# GeoPano: Towards Geometrically Accurate 3D Reconstruction from a Single Panorama

Jing OU<sup>1∗</sup> Zidong Cao<sup>1∗</sup> Liaoyuan Fan<sup>3</sup> Zhuoxiao Li<sup>1</sup> Shuai Zhang<sup>1</sup> Yinrui Ren<sup>2</sup> Dongli Wu<sup>1</sup> Haoang Li<sup>1</sup> Hui Xiong<sup>1†</sup> Wufan Zhao<sup>1†</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou), China <sup>2</sup>Sun Yat-sen University, Guangzhou, China <sup>3</sup>The Hong Kong University, Hong Kong <sup>∗</sup>Equal contribution <sup>†</sup>Corresponding author

![](images/aa25997885228e9830d3dd74226cbde4d4501d72c42bc98aec385793f7a2cf03.jpg)  
Zoom in from this view

Figure 1: For each scene, our GeoPano takes a single panorama as input, directly recovers distortionfree 3D point clouds, which can be easily re-projected into the sphere plane to obtain the ERP depth map. Given diverse in-the-wild panoramas from both indoor and outdoor scenes, GeoPano can robustly reconstruct high-fidelity 3D geometry, showing its impressive zero-shot capabilities.

## Abstract

Panoramic 3D reconstruction enables the holistic recovery of complete 360 <sup>◦</sup> scene geometry, offering significant benefits for robotics and AR/VR applications.

However, directly estimating geometry from ERP images is inherently challenged by spherical distortions. To reduce spherical projection distortions, a common strategy is to decompose panoramas into distortion-free perspective views, but independently processing these views often introduces cross-view geometric inconsistencies. To address these challenges, we introduce GeoPano, a feed-forward framework for single-view panoramic 3D reconstruction that recovers a geometrically accurate and consistent point cloud from a single panorama. Our approach is driven by two key innovations: First, we propose a panoramic-aware attention aggregator, featuring a streamlined attention architecture equipped with a novel Pano-Attention mechanism, which explicitly exploits the intrinsic spherical geometry of the panorama to guide cross-view interactions, thereby suppressing geometric inconsistencies. Furthermore, we formulate a hybrid supervision strategy to jointly enforce fine-grained local details in perspective space and global structural coherence in spherical space. Experiments demonstrate that our method significantly outperforms previous SOTA approaches in reconstruction accuracy, geometric consistency, and visual fidelity.

## 1 Introduction

Omnidirectional vision has garnered significant attention due to its capability to capture a comprehensive 360<sup>◦</sup> field-of-view, acting as a critical enabler for holistic scene understanding [Ai et al., 2025, Gao et al., 2022]. Consequently, panoramic monocular 3D reconstruction, or depth estimation has emerged as a cornerstone task for numerous downstream applications, ranging from autonomous driving [Kumar et al., 2021, Zheng et al., 2023] to immersive virtual reality (VR) [Yang et al., 2022, 2024, Cao et al., 2023] and visual navigation [Li and Bansal, 2023]. In current pipelines, the equirectangular projection (ERP) is widely used to represent spherical data on 2D planes. However, this projection is fundamentally hindered by non-uniform geometric deformations, particularly the extreme stretching concentrated at the polar regions [Ai et al., 2025]. Such inherent spherical distortions disrupt the expected spatial distribution of visual features, causing conventional perspective 3D reconstruction to degrade significantly when applied directly to ERP imagery.

To circumvent these projection artifacts, a prevailing strategy utilizes perspective projection to extract a set of distortion-free patches from the panorama. Nevertheless, processing these extracted views in isolation disrupts global contextual dependencies, invariably leading to cross-view inconsistencies and boundary seams. Existing methods attempt to resolve this fragmentation through two distinct paradigms: end-to-end supervised fusion and zero-shot training-free inference. End-to-end supervised architectures [Wang et al., 2020, 2022, Jiang et al., 2021, Li et al., 2022b, Ai et al., 2023] attempt to bridge this spatial gap by explicitly fusing global ERP context with local extracted features during training. However, their efficacy is heavily bottlenecked by the prohibitive cost of dense 360<sup>◦</sup> depth annotations [Chang et al., 2017, Armeni et al., 2017, Zheng et al., 2020], resulting in limited cross-domain generalization. Furthermore, the massive computational overhead of multi-branch networks typically restricts these models to low-resolution inputs. Conversely, zero-shot training-free approaches [Rey-Area et al., 2022, Peng and Zhang, 2023, Jung et al., 2025] bypass domain-specific training by leveraging pre-trained perspective foundation models [Ranftl et al., 2020, Yin et al., 2021] to process the decomposed perspective patches independently. While this effectively avoids the need for 360<sup>◦</sup> datasets, these methods face their own critical bottlenecks: reconciling scale ambiguities and enforcing cross-view alignment heavily relies on either computationally expensive iterative post-processing [Jung et al., 2025, Rey-Area et al., 2022] or auxiliary panoramic models as references [Peng and Zhang, 2023].

Currently, the advent of feed-forward 3D reconstruction architectures (e.g., VGGT [Wang et al., 2025a] and π<sup>3</sup> [Wang et al., 2025b]) has established a new frontier in visual geometry, operating as 3D foundation models. These networks excel at reconstructing globally consistent 3D structures from uncalibrated multi-view images by intrinsically capturing cross-view geometric relations. Motivated by this, we reformulate panoramic 3D reconstruction as a sparse multi-view problem via view decomposition, enabling us to directly leverage the robust geometric priors of foundation models for holistic scene recovery. Furthermore, benefiting from extensive pre-training on millions of real-world 3D scenes, these foundation models inherently possess exceptional sparse-view geometric reasoning capabilities. Such a rich, data-driven prior allows these frameworks to readily adapt to a sparse decomposed-view setup requiring only minimal fine-tuning data.

Fundamentally, the multi-view reasoning prowess of these foundation models is rooted in their strong attention aggregators. To adapt them for panoramic scenes, we analyze how their attention aggregator captures global information across decomposed panoramic views. This exploration exposes two bottlenecks: (1) early global attention in $\pi ^ { 3 }$ fails to capture geometric correlations such as spatial overlap; and (2) while later global attention layers exhibit a clear preference for interacting within overlapping views, interactions stemming from entirely non-overlapping views remain significant, which we argue is detrimental to achieving cross-view consistency.

To directly address these two bottlenecks, we propose a novel panoramic-aware Attention Aggregator. Specifically, to prevent the ineffective and diffuse interactions observed in early layers, our architecture initially employs a sequence of intra-frame self-attention layers. This stage focuses entirely on extracting reliable local geometry, explicitly bypassing premature cross-view interference. Subsequently, to tackle the residual noise in later stages, we introduce alternating attention layers augmented with our Pano-Attention. Integrated via residual connections, this Pano-Attention explicitly guides query tokens to interact strictly with geometrically overlapping regions across the decomposed views. Consequently, it effectively suppresses irrelevant interactions and significantly enhances globally consistent scene reconstruction. In summary, our contributions are three-fold: (1) A Panoramic-Aware Attention Aggregator that explicitly incorporates spherical geometry to guide cross-view feature interactions. This design mitigates irrelevant token interactions and improves geometric consistency across decomposed perspective views. (2) A hybrid supervision strategy that jointly optimizes local perspective-space point clouds and global spherical-space structure. This strategy preserves fine-grained local geometry while enhancing globally consistent 3D reconstructions. (3) Extensive experiments on standard benchmarks demonstrate that our method achieves superior performance. It consistently outperforms prior single-view panoramic approaches in reconstruction accuracy, geometric consistency, and overall visual fidelity.

## 2 Related Work

## 2.1 Panoramic Monocular Depth Estimation

Panoramic monocular depth estimation reconstructs holistic 3D scene depth from single equirectangular projection (ERP) images. Current solutions generally follow two main tracks: direct ERP-domain estimation and multi-view fusion.

ERP-domain estimation. ERP-based methods preserve global consistency but suffer from spherical distortions and data scarcity. To mitigate distortions, prior works employ architectural adaptations like distortion-aware convolution and spherical encoding [Cao and Wang, 2024, Eder and Frahm, 2019, Sun et al., 2021, Tateno et al., 2018, Yun et al., 2023]. To address data limitations, recent studies focus on data synthesis and knowledge transfer, such as generating pseudo-labels from perspective estimators [Cao et al., 2025b], expanding datasets via diffusion-based outpainting [Li et al., 2025], or introducing large-scale panoramic datasets [Lin et al., 2025]. Despite these efforts, inherent distortions and data scarcity continue to hinder depth accuracy in highly distorted regions.

Multi-view Fusion. Conversely, multi-view fusion paradigms mitigate spherical distortions by decomposing panoramas into perspective views [Rey-Area et al., 2022, Ai et al., 2023, Jiang et al., 2021, Li et al., 2022b, Mohadikar et al., 2024, Wang et al., 2018, 2020, 2022]. To address spatial fragmentation, recent methods explore advanced fusion strategies, such as post-processing optimization for view alignment [Jung et al., 2025] or representing static panoramas into sequential video frames to utilize video depth models [Cao et al., 2025a]. Despite these efforts, the multi-view fusion strategy inherently disrupts global structural consistency. Consequently, these approaches often rely on time-consuming post-processing or cross-domain feature fusion, which struggles to enforce accurate global consistency in complex scenes.

## 2.2 Feed-forward 3D Reconstruction Models

DUST3R [Wang et al., 2024] introduces a transformative paradigm by directly regressing 3D point maps and camera poses from uncalibrated images. Subsequent works such as Fast3R [Yang et al., 2025] and SLAM3R [Liu et al., 2025] extend this to multi-view and video inputs via global fusion transformers. Scaling this further, VGGT [Wang et al., 2025a] leverages massive 3D datasets to achieve robust in-the-wild reconstruction. To resolve the performance degradation caused by reference-frame dependency in these architectures, $\pi ^ { 3 }$ [Wang et al., 2025b] proposes a permutationequivariant design. Despite their generalization strengths, these regression-based methods often lack explicit geometric constraints tailored for the spherical topology of panoramas, potentially leading to structural drift in $3 6 0 ^ { \circ }$ reconstruction.

![](images/31abe297d19bd65e9890d7080acf4f993e6c8032beffb603ba7b47eeb0287714.jpg)

![](images/4f7c0831bce5c42d41d8e057244f1e0f9dc553009a3033d5d992a57122f78d17.jpg)  
Figure 2: Statistics of global attention cross-view interactions in $\pi ^ { 3 }$

Recently, researchers have extended this feed-forward paradigm to the spherical domain for holistic 360<sup>◦</sup> reconstruction. PanoSplatt3R [Ren et al., 2025] adapts perspective pre-training for unposed widebaseline panorama 3D reconstruction, enabling high-fidelity novel view synthesis. PanoVGGT [Guo et al., 2026] proposes an end-to-end Transformer to jointly predict camera poses, dense depth, and 3D point clouds directly from unordered panoramas by mitigating spherical distortions. VGGT-360 [Yuan et al., 2026] achieves zero-shot panoramic depth estimation by reformulating the task as a 3D reconstruction problem utilizing perspective foundation models.

## 3 Methodology

## 3.1 Analysis of Cross-View Interactions

We adopt $\pi ^ { 3 }$ [Wang et al., 2025b] as our base model, owing to its strong robustness in sparse-view 3D reconstruction. Given N perspective views, $\pi ^ { 3 }$ utilizes a DINOv2 encoder [Oquab et al., 2023] to extract feature tokens. These tokens are then processed by an attention aggregator comprising 18 alternating intra-frame and global attention blocks for feature aggregation. Finally, camera and point map heads decode the aggregated features into camera poses and 3D point maps.

To better adapt the multi-view geometric reasoning capability of $\pi ^ { 3 }$ to panoramic reconstruction, we analyze the behavior of global attention across different layers of the attention aggregator, inspired by the analysis in AVGGT [Sun et al., 2025]. We conduct an exploratory study on a small-scale dataset, which ultimately motivates the design of our proposed attention aggregator. Specifically, we select one token from view 4 as the query token. For each global attention layer, we separately extract the top-100 key/value tokens with the highest attention weights for this query token, and then compute their view-wise distribution.

As illustrated in Fig. 2 (a), in the early global attention layers (Layers $0 \sim 8 .$ , blue in the figure), the attention weights are uniformly distributed across the first eight horizontal views $( \mathrm { v } 0 \sim \mathrm { v } 7 )$ . The v8 (upward) and v9 (downward) receive minimal attention likely because these low-texture regions (e.g., skies or grounds) semantically diverge from the horizontal views. Consequently, this early-stage attention exhibits no preference for views with actual spatial overlap, resulting in a diffuse distribution.

In contrast, the later global attention layers (Layers 9 ∼ 17, other colors in Fig. 2 (a)) demonstrate an explicit interaction bias. The attention becomes highly concentrated on the source view (v4) and its spatially overlapping neighbors (v5), while interactions with irrelevant views are suppressed. The Global Decay Slope metric quantitatively confirms this trend. A higher slope means the attention is more tightly focused. As shown in Fig. 2 (b), the slope increases significantly in later layers, proving that the model successfully concentrates on geometrically relevant regions.

Consequently, we conclude that under decomposed perspective views, early global attention in $\pi ^ { 3 }$ fails to capture geometric correlations such as spatial overlap. Furthermore, we find that even in the later global attention layers, interactions stemming from entirely non-overlapping views are still significant, which we argue is detrimental to achieving cross-view consistency. Motivated by these two findings, we subsequently introduce the design of our attention aggregator, aiming to enhance the efficiency of cross-view feature interactions and improve the global consistency of the reconstruction.

![](images/5a377452f69d9dd1e68017bbccb0bc595839c772471bf649bf7e043b6ff7fc36.jpg)  
Figure 3: Overview of the framework. Given a panorama, we decompose it into perspective views and encode them using DINOv2. The extracted features are then aggregated by a panoramic-aware attention aggregator. Based on the aggregated features, the model predicts local point maps, which are fused into a global point cloud. We further introduce a hybrid loss to jointly enforce local details and global consistency. G-Attn and F-Attn denote Global Attention and Frame Attention.

![](images/a3db9c16d40bb021f25dc4094af0392998ca992e1af22b5df0601e19b650a511.jpg)  
Figure 4: Left: View decomposition. The input panorama is projected into 10 perspective views: 8 evenly distributed horizontally, plus upward and downward views. Right: Token mask construction. Candidate masks are constructed by thresholding the pairwise cosine similarity of token ray directions.

## 3.2 Framework Overview

As illustrated in Fig. 3, our framework operates through a three-stage pipeline: view decomposition, local 3D reconstruction, and panoramic fusion. The input panorama is first decomposed into 10 overlapping perspective views (8 horizontal views, 1 upward, and 1 downward, as detailed in Fig. 4). During the reconstruction phase, these decomposed views are tokenized into high-dimensional features by DINOv2 [Oquab et al., 2023]. These tokens are subsequently processed by our proposed panoramic-aware attention aggregator, which explicitly guides multi-view interactions to enforce geometric consistency. A local point map head then decodes these aggregated features into perview 3D point clouds. Finally, these per-view point clouds are re-projected back into the spherical coordinate system and seamlessly fused into a globally aligned, holistic 3D point cloud. The entire framework is optimized via a hybrid loss, which simultaneously supervises the local perspective point clouds and the global panoramic point cloud.

## 3.3 Panoramic-aware Attention Aggregator

Base Architecture. Motivated by the findings in Sec.3.1, the panoramic-aware attention aggregator is specifically designed to address two key issues: (1) preventing ineffective and diffuse interactions in the early global attention layers, and (2) suppressing residual interactions across non-overlapping regions in the later stages. As shown in Fig. 3, we design the base architecture with nine layers of intra-frame self-attention followed by nine layers of alternating attention. In layers 10, 12, 14, and 16 of the Attention Aggregator, we compute Pano-Attention and add it to the output of the alternating attention via a residual connection. The first nine layers focus on extracting reliable intra-frame geometry, while the subsequent alternating attention layers jointly refine inter-frame consistency and fine-grained intra-frame features. Importantly, the introduced Pano-Attention explicitly guides the query tokens to interact with tokens in cross-view overlapping regions, thereby enhancing geometry-aware feature interactions and suppressing interactions with irrelevant tokens.

Pano-Attention. We introduce Pano-Attention, which restricts token interactions to geometrically coherent regions. Specifically, As shown in Fig. 4 we construct a candidate token mask M, where the i-th row specifies the key/value tokens (candidate tokens 1∼10 in the figure) that can be queried by the i-th query token. We apply this mask to the attention logits before the softmax operation, as illustrate in Fig. 3. We explicitly mask invalid token pairs by setting their pre-softmax scores $\mathrm { t o } - \infty$ ensuring their resulting attention weights are strictly zero. In this way, each token only aggregates information from geometrically valid candidates, reducing unreliable interactions and improving cross-view consistency.

To construct the candidate mask, we leverage rays in the spherical space to introduce global geometric information. Specifically, since all decomposed perspective views share an identical camera projection center, two patches from adjacent views are considered geometrically aligned if they correspond to the same spherical ray direction. Based on this property, we can establish accurate correspondences between any two patches, determining whether they are geometrically relevant in the spherical space.

Formally, we quantify this geometric relevance by measuring the cosine similarity between their associated spherical rays. To construct the candidate mask M, as shown in Fig. 4, we first compute the cosine similarity $s _ { i j }$ between the ray of each token i and token $j$ to form a global cosine similarity matrix S. Subsequently, the candidate mask M is calculated via Eq. 1:

$$
s _ { i j } = \frac { \mathbf { r } _ { i } ^ { \top } \mathbf { r } _ { j } } { \| \mathbf { r } _ { i } \| _ { 2 } \| \mathbf { r } _ { j } \| _ { 2 } } , \quad \mathbf { M } _ { i j } = \left\{ { 1 , \quad s _ { i j } > \tau } , \right.\tag{1}
$$

where τ is a threshold used to determine whether these two tokens are geometrically correlated.

## 3.4 Hybrid Loss

Building upon $\pi ^ { 3 }$ [Wang et al., 2025b], we first apply a local point cloud loss to supervise the fine-grained geometry of each decomposed view. However, optimizing solely in isolated perspective spaces inevitably fractures global spatial alignment. To overcome this, we introduce a panoramic point cloud loss in spherical space, effectively coupling local precision with holistic consistency. The total loss $\mathcal { L } _ { \mathrm { t o t a l } }$ is defined as a weighted combination of the local point cloud loss $\mathcal { L } _ { \mathrm { l o c a l } }$ and the panoramic point cloud loss $\mathcal { L } _ { \mathrm { p a n o } }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \lambda \mathcal { L } _ { \mathrm { l o c a l } } + \beta \mathcal { L } _ { \mathrm { p a n o } } , } \end{array}\tag{2}
$$

where λ and $\beta$ control the relative contributions of the local and panoramic losses.

Local Point Cloud Loss. We first compute an optimal scale $s _ { \mathrm { l o c a l } } ^ { * }$ by aligning the predicted points $\hat { P } _ { i } ^ { \mathrm { l o c a l } }$ with the ground-truth points $P _ { i } ^ { \mathrm { l o c a l } }$ with inverse depth weights $\frac { 1 } { z _ { i } }$ . Based on the scale-aligned points, we compute the local point cloud loss $\mathcal { L } _ { \mathrm { l o c a l } } \mathrm { : }$

$$
s _ { \mathrm { l o c a l } } ^ { * } = \arg \operatorname* { m i n } _ { s } \sum _ { i = 1 } ^ { N } \frac { 1 } { z _ { i } } \| s \hat { P } _ { i } ^ { \mathrm { l o c a l } } - P _ { i } ^ { \mathrm { l o c a l } } \| _ { 1 } , \qquad \mathcal { L } _ { \mathrm { l o c a l } } = \sum _ { i = 1 } ^ { N } \frac { 1 } { z _ { i } } \| s _ { \mathrm { l o c a l } } ^ { * } \hat { P } _ { i } ^ { \mathrm { l o c a l } } - P _ { i } ^ { \mathrm { l o c a l } } \| _ { 1 }\tag{3}
$$

Panoramic Point Cloud Loss. For panoramic point cloud loss, We first project the predicted perspective-view point clouds $\hat { P } _ { v }$ from multiple views back into a spherical coordinate system using the inverse projection operator ${ \mathcal { F } } ^ { - 1 }$

$$
\hat { P } ^ { \mathrm { p a n } } = \frac { 1 } { V } \sum _ { v = 1 } ^ { V } \mathcal { F } ^ { - 1 } ( \hat { P } _ { v } ) , \quad s _ { \mathrm { p a n o } } ^ { * } = \operatorname { a r g m i n } \sum _ { s } ^ { N } \frac { 1 } { i = 1 } \hat { d } _ { i } \Vert s \hat { P } _ { i } ^ { \mathrm { p a n o } } - P _ { i } ^ { \mathrm { p a n o } } \Vert _ { 1 } , \mathcal { L } _ { \mathrm { p a n o } } = \sum _ { i = 1 } ^ { N } \frac { 1 } { d _ { i } } \Vert s _ { \mathrm { p a n o } } ^ { * } \hat { P } _ { i } ^ { \mathrm { p a n o } } - P _ { i } ^ { \mathrm { p a n o } } \Vert _ { 1 }\tag{4}
$$

where $V$ denotes the total number of decomposed views. Subsequently, we align the predicted point cloud $\hat { P } _ { i } ^ { \mathsf { p a n o } }$ and ground-truth point cloud $P _ { i } ^ { \mathsf { p a n o } }$ within the spherical space, weighted by the panoramic inverse depth $\frac { 1 } { d _ { i } }$ , to derive the optimal scale factor $s _ { \mathrm { p a n o } } ^ { * }$ . Based on these scale-aligned points, we then compute the panoramic point cloud loss $\mathcal { L } _ { \mathrm { p a n o } }$

## 4 Experiments

## 4.1 Experiment Setup

Training Datasets. For training, we use Holo360D [Ou et al., 2026], a large-scale real-world panoramic 3D dataset comprising 75 indoor and outdoor scenes. It provides panoramic RGB images with corresponding depth maps, each captured at a resolution of 2880 × 1440. We train our model using 23,000 panoramic frames.

Evaluation Datasets and Evaluation Metrics. For evaluation, following prior work [Lin et al., 2025], we test our model on three 360<sup>◦</sup> depth datasets: Matterport3D [Chang et al., 2017] (indoor) and Stanford2D3D [Armeni et al., 2017] (indoor), and Deep360 [Li et al., 2022a] (outdoor). To comprehensively assess model performance, we evaluate both point cloud using 3D metrics [Jung et al., 2025] such as Chamfer Distance, F-Score, and Intersection over Union (IoU), and depth using standard 2D metrics including Absolute Relative Error (AbsRel), Root Mean Squared Error (RMSE), and the accuracy under a threshold $\delta _ { i }$ with i=1.25.

Implementation details. For training, each model is trained using four NVIDIA A800 GPUs for 20 epochs, with each epoch consisting of 1,000 iterations. During evaluation, inference is performed on a single A800 GPU, while monitoring GPU memory usage and runtime performance.

## 4.2 Qualitative and Quantitative Evaluation

Comparative Baselines. We compare GeoPano with recent state-of-the-art panoramic approaches. We consider both methods operating directly on the equirectangular projection (ERP) format (PanDA [Cao et al., 2025b], $\mathrm { D A } ^ { 2 }$ [Li et al., 2025], DAP [Lin et al., 2025], and PanoVGGT [Guo et al., 2026]) and those relying on view decomposition into pinhole projections (360MonoDepth [Rey-Area et al., 2022], RPG360 [Jung et al., 2025]). PanoVGGT is evaluated only on Deep360, as its custom dataset splits for Matterport3D and Stanford2D3D are not publicly available, preventing a fair comparison on these benchmarks.

Table 1: Quantitative evaluation on 3D and 2D metrics. ERP and Pinhole denote methods using ERP projection images and pinhole images decomposed from panoramas as input, respectively.
<table><tr><td colspan="2">Method</td><td colspan="6">Matterport3D (Indoor)</td><td colspan="6">Stanford2D3D (Indoor)</td><td colspan="6">Deep360 (Outdoor)</td></tr><tr><td></td><td></td><td colspan="3">3D Metrics</td><td colspan="3">2D Metrics</td><td colspan="3">3D Metrics</td><td colspan="3">2D Metrics</td><td colspan="3">3D Metrics</td><td colspan="3">2D Metrics</td></tr><tr><td>Model</td><td>Input</td><td>Chamfer↓</td><td>F-Score↑</td><td>IoU↑</td><td>| AbsRel.↓</td><td>RMSE↓</td><td>δ1↑</td><td>Chamfer↓</td><td>F-Score↑</td><td>IoU↑</td><td>AbsRel↓</td><td>RMSE↓</td><td>δ1↑</td><td>Chamfer.↓</td><td>F-Score↑</td><td>IoU↑</td><td>AbsRel↓</td><td>RMSE↓ δ1↑</td></tr><tr><td>PanDA</td><td></td><td>0.553</td><td>26.270</td><td>15.710</td><td>0.0888</td><td>0.3325</td><td>0.9209</td><td>0.727</td><td>15.990 8.850</td><td>0.1648</td><td>0.2364</td><td>0.7326</td><td>2.350</td><td></td><td>18.420 10.810</td><td>0.3319</td><td></td><td>10.7668 0.4498</td></tr><tr><td>DA2</td><td>ERP</td><td>0.180</td><td>77.450</td><td>67.700</td><td>0.0667</td><td>0.2882</td><td>0.9561</td><td>0.145 80.310</td><td>69.360</td><td>0.0723</td><td>0.1400</td><td>0.9545</td><td>4.036</td><td>3.520</td><td>1.920</td><td>0.1991</td><td>12.3898</td><td>0.8155</td></tr><tr><td>DAP</td><td></td><td>0.269</td><td>63.260</td><td>49.930</td><td>0.1186</td><td>0.7510</td><td>0.8518</td><td>0.487 33.120</td><td>20.600</td><td>0.0921</td><td>0.3820</td><td>0.9135</td><td>2.214</td><td>34.180</td><td>21.850</td><td>0.1213</td><td>7.2308</td><td>0.9159</td></tr><tr><td>PanoVGGT</td><td></td><td></td><td></td><td></td><td>-</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.537</td><td>69.094</td><td>55.587</td><td>0.1618</td><td>6.1886</td><td>0.7561</td></tr><tr><td>360MonoDepth</td><td>Pinhole</td><td>0.632</td><td>28.050</td><td>16.990</td><td>0.2083</td><td>0.7909</td><td>0.6558</td><td>0.576 25.230</td><td>14.670</td><td>0.1650</td><td>0.2823</td><td>0.7456</td><td>3.761</td><td>25.450</td><td>15.080</td><td>1.5836</td><td>17.2300</td><td>0.1412</td></tr><tr><td>RPG360</td><td></td><td>0.337</td><td>58.810</td><td>44.910</td><td>0.2030</td><td>0.6670</td><td>0.8590</td><td>0.206</td><td>73.910 61.880</td><td>0.1016</td><td>0.1924</td><td>0.9104</td><td>3.430</td><td>62.990</td><td>48.110</td><td>0.2375</td><td>10.0562</td><td>0.6377</td></tr><tr><td>Ours</td><td></td><td>0.153</td><td>82.250</td><td>72.920</td><td>0.0597</td><td>0.2695</td><td>0.9648</td><td>0.127</td><td>85.040</td><td>75.670 0.06361</td><td>0.1326</td><td>0.9539</td><td>1.202</td><td>80.560</td><td>68.710</td><td>0.1130</td><td>5.5837</td><td>0.8289</td></tr></table>

Evaluation. As reported in Tab. 1, GeoPano establishes a new state-of-the-art, demonstrating superior performance across all 3D and 2D metrics. Additionally, as illustrated in Fig. 5 and Fig. 6, our approach significantly mitigates spherical distortions and preserves geometric consistency, yielding high-fidelity 3D structures across diverse indoor and outdoor environments. As shown in the first two rows of Fig. 5, GeoPano achieves accurate geometric reconstructions even in challenging outdoor long-range scenes.

Ground Truth  
RPG360  
Ours  
DA2  
DAP  
![](images/4fd434309fc6596de57e82fc71a6f97b403d106d803af1e3863a17bf90710013.jpg)  
Figure 5: Qualitative comparison in outdoor scenarios. We compare GeoPano against $\mathrm { D A } ^ { 2 }$ , DAP, and RPG360. As illustrated, our model produces more precise 3D geometry and significantly mitigates structural distortions in long-range regions.

![](images/0e3baebb0e4c6e4382bf0b79d47b87d709fcaf0f3fc5b4c71b885ff1baab95cf.jpg)  
Figure 6: Qualitative comparison in indoor scenes. Compared to $\mathrm { D A } ^ { 2 }$ , DAP, and RPG360, our method achieves superior global structural coherence and recovers fine-grained local geometries.

## 4.3 Ablation Studies

Main Ablation. To evaluate the contribution of each proposed component of GeoPano, we conduct a ablation study conducted on Matterport3D, summarized in Tab. 2. We start from a panoramic finetuing version of $\pi ^ { 3 }$ on our training data as the baseline and add: (1) dropping early global attention, Pano-Attention, and (3) hybrid loss. All introduced modules effectively improve reconstruction accuracy; notably, Pano-Attention intrinsically maintains spatial consistency across the decomposed views, with qualitative comparisons shown in Fig. 7.

Effectiveness of Early Global Attention. We investigate the effect of dropping early global attention in the baseline $\pi ^ { 3 }$ architecture for single-frame panoramic 3D reconstruction, as shown in Tab. 3. Specifically, we train and evaluate several variants where global attention is dropped from the initial layers, including layers $0 \sim 4 , 0 \sim 6 , 0 \sim 8 ,$ , and $0 \sim 1 0$ . Dropping global attention from layers $0 \sim 8$ yields the best trade-off between computational efficiency and geometric accuracy. This observation supports our analysis of $\pi ^ { 3 } .$ , suggesting that the diffuse interactions induced by early global attention can adversely affect global consistency in the panoramic setting.

Table 2: Ablation study of the proposed components. Each designed module effectively improves the reconstruction accuracy. Compared with the baseline, our full model reduces the inference time by 10.3%, with only a 5.6% increase in memory consumption.
<table><tr><td>Drop Global Attention</td><td>Pano-Attention</td><td>Hybrid Loss</td><td>Chamfer↓</td><td>F-Score↑</td><td>IoU↑</td><td>Inference Time (s)</td><td>GPU Mem. (GB)</td></tr><tr><td>×</td><td>×</td><td>X</td><td>0.164</td><td>81.110</td><td>71.640</td><td>0.500</td><td>4.890</td></tr><tr><td>√</td><td>X</td><td>X</td><td>0.157</td><td>81.780</td><td>72.370</td><td>0.370</td><td>4.250</td></tr><tr><td>√</td><td>√</td><td>X</td><td>0.155</td><td>82.010</td><td>72.710</td><td>0.450</td><td>5.170</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.153</td><td>82.250</td><td>72.920</td><td>0.450</td><td>5.170</td></tr></table>

Table 3: Ablation study on the effect of dropping early global attention layers in alternating attention.
<table><tr><td>Layers w/o Global Attntion</td><td>Chamfer↓</td><td>F-Score↑</td><td>IoU↑</td><td>Inference Time (s)</td><td>GPU Mem. (GB)</td></tr><tr><td>Baseline</td><td>0.164</td><td>81.110</td><td>71.640</td><td>0.500</td><td>4.890</td></tr><tr><td> $0 \sim 4$ </td><td>0.161</td><td>81.530</td><td>71.970</td><td>0.440</td><td>4.540</td></tr><tr><td> $0 \sim 6$ </td><td>0.157</td><td>82.050</td><td>72.680</td><td>0.410</td><td>4.440</td></tr><tr><td> $0 \sim 8$ </td><td>0.157</td><td>81.780</td><td>72.370</td><td>0.390</td><td>4.250</td></tr><tr><td> $0 \sim 1 0$ </td><td>0.159</td><td>81.450</td><td>72.030</td><td>0.350</td><td>4.000</td></tr></table>

Effectiveness of Pano-Attention. We investigate the effectiveness of Pano-Attention and its optimal configuration. Specifically, we compare four settings: (1) no Pano-Attention, (2) Pano-Attention applied at every other layer (layers 10, 12, 14, 16), (3) Pano-Attention applied at every layer (layers 9 ∼ 16), and (4) standard global attention applied at every other layer (layers 10, 12, 14, 16). The results show that applying Pano-Attention at every other layer achieves the best performance, indicating that moderate geometric constraints better balance reliable cross-view interaction and global information aggregation. In contrast, the standard global attention baseline (configure 4) fails to yield comparable gains, demonstrating that the improvement stems from our geometry-constrained crossview interaction rather than merely stacking additional attention layers.

Effectiveness of hybrid loss. Tab. 5 validates the necessity of our dual-constraint design, demonstrating that the combination of local and global losses is strictly superior to single-loss baselines. We observe that peak performance is achieved with $\lambda = 1$ and $\beta = 0 . 7 5$ . This optimal weighting strategy explicitly navigates the trade-off between local and global representation learning, ensuring that high-fidelity local geometric structures are flawlessly preserved while still satisfying strict global coherence.

![](images/f1e2cfff7846c397b5dca4e35f5185edcf4a7ed280e92b1fd9732bbca08f5e66.jpg)  
(1) w/o Pano-Attention (2) with Pano-Attention  
Figure 7: Qualitative comparison with and without Pano-Attention. Integrating our proposed Pano-Attention significantly enhances the geometric consistency.

Table 4: Ablation study on the configuration of Pano-Attention layers.
<table><tr><td>Pano-Attention Layers</td><td>Chamfer↓</td><td>F-Score↑</td><td>IoU↑</td></tr><tr><td>(1) 0 Pano-Attention</td><td>0.157</td><td>81.780</td><td>72.370</td></tr><tr><td>(2) 4 Pano-Attention (10,12,14,16)</td><td>0.153</td><td>82.250</td><td>72.920</td></tr><tr><td>(3) 8 Pano-Attention (9,10,11,12,13,14,15,16)</td><td>0.156</td><td>81.590</td><td>72.190</td></tr><tr><td>(4) 4 Global Attention (10,12,14,16)</td><td>0.160</td><td>81.470</td><td>71.920</td></tr></table>

Table 5: Ablation study with different loss weightings.
<table><tr><td>λ</td><td>β</td><td>Chamfer↓</td><td>F-Score↑</td><td>IoU↑</td></tr><tr><td>1</td><td>0</td><td>0.155</td><td>82.010</td><td>72.710</td></tr><tr><td>1</td><td>0.5</td><td>0.154</td><td>82.030</td><td>72.670</td></tr><tr><td>1</td><td>0.75</td><td>0.153</td><td>82.250</td><td>72.920</td></tr><tr><td>1</td><td>1</td><td>0.156</td><td>81.500</td><td>72.000</td></tr><tr><td>0</td><td>1</td><td>0.154</td><td>81.960</td><td>72.550</td></tr></table>

## 5 Conclusion

We proposed GeoPano, a feed-forward framework for single-view panoramic 3D reconstruction, featuring a Panoramic-Aware Attention Aggregator that leverages spherical geometry to improve cross-view consistency. A hybrid supervision strategy combines local and global point cloud losses, ensuring fine-grained details and holistic structural coherence. Experiments show that GeoPano outperforms prior single-view panoramic methods in accuracy, consistency, and visual fidelity while maintaining practical efficiency.

## References

Hao Ai, Zidong Cao, Yan-Pei Cao, Ying Shan, and Lin Wang. Hrdfuse: Monocular 360deg depth estimation by collaboratively learning holistic-with-regional depth distributions. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13273–13282, 2023.

Hao Ai, Zidong Cao, and Lin Wang. A survey of representation learning, optimization strategies, and applications for omnidirectional vision: H. ai et al. International Journal of Computer Vision, 133 (8):4973–5012, 2025.

Iro Armeni, Sasha Sax, Amir R Zamir, and Silvio Savarese. Joint 2d-3d-semantic data for indoor scene understanding. arXiv preprint arXiv:1702.01105, 2017.

Zidong Cao and Lin Wang. Monocular 360 depth estimation via spherical fully-connected crfs. IEEE Robotics and Automation Letters, 10(2):1409–1416, 2024.

Zidong Cao, Hao Ai, Yan-Pei Cao, Ying Shan, Xiaohu Qie, and Lin Wang. Omnizoomer: Learning to move and zoom in on sphere at high-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12897–12907, 2023.

Zidong Cao, Jinjing Zhu, Hao Ai, Lutao Jiang, Yuanhuiyi Lyu, and Hui Xiong. st<sup>2</sup>360 d: Spatialto-temporal consistency for training-free 360 monocular depth estimation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a.

Zidong Cao, Jinjing Zhu, Weiming Zhang, Hao Ai, Haotian Bai, Hengshuang Zhao, and Lin Wang. Panda: Towards panoramic depth anything with unlabeled panoramas and mobius spatial augmentation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 982–992, 2025b.

Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158, 2017.

Marc Eder and Jan-Michael Frahm. Convolutions on spherical images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 1–5, 2019.

Shaohua Gao, Kailun Yang, Hao Shi, Kaiwei Wang, and Jian Bai. Review on panoramic imaging and its applications in scene understanding. IEEE Transactions on Instrumentation and Measurement, 71:1–34, 2022.

Yijing Guo, Mengjun Chao, Luo Wang, Tianyang Zhao, Haizhao Dai, Yingliang Zhang, Jingyi Yu, and Yujiao Shi. Panovggt: Feed-forward 3d reconstruction from panoramic imagery. arXiv preprint arXiv:2603.17571, 2026.

Hualie Jiang, Zhe Sheng, Siyu Zhu, Zilong Dong, and Rui Huang. Unifuse: Unidirectional fusion for 360 panorama depth estimation. IEEE Robotics and Automation Letters, 6(2):1519–1526, 2021.

Dongki Jung, Jaehoon Choi, Yonghan Lee, and Dinesh Manocha. Rpg360: Robust 360 depth estimation with perspective foundation models and graph optimization. arXiv preprint arXiv:2509.23991, 2025.

Varun Ravi Kumar, Senthil Yogamani, Hazem Rashed, Ganesh Sitsu, Christian Witt, Isabelle Leang, Stefan Milz, and Patrick Mäder. Omnidet: Surround view cameras based multi-task visual perception network for autonomous driving. IEEE Robotics and Automation Letters, 6(2):2830– 2837, 2021.

Haodong Li, Wangguangdong Zheng, Jing He, Yuhao Liu, Xin Lin, Xin Yang, Ying-Cong Chen, and Chunchao Guo. Da <sup>2</sup>: Depth anything in any direction. arXiv preprint arXiv:2509.26618, 2025.

Jialu Li and Mohit Bansal. Panogen: Text-conditioned panoramic environment generation for visionand-language navigation. Advances in neural information processing systems, 36:21878–21894, 2023.

Ming Li, Xueqian Jin, Xuejiao Hu, Jingzhao Dai, Sidan Du, and Yang Li. Mode: Multi-view omnidirectional depth estimation with 360<sup>◦</sup> cameras. In European Conference on Computer Vision, pages 197–213. Springer, 2022a.

Yuyan Li, Yuliang Guo, Zhixin Yan, Xinyu Huang, Ye Duan, and Liu Ren. Omnifusion: 360 monocular depth estimation via geometry-aware fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2801–2810, 2022b.

Xin Lin, Meixi Song, Dizhe Zhang, Wenxuan Lu, Haodong Li, Bo Du, Ming-Hsuan Yang, Truong Nguyen, and Lu Qi. Depth any panoramas: A foundation model for panoramic depth estimation. arXiv preprint arXiv:2512.16913, 2025.

Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yingda Yin, Yanchao Yang, Qingnan Fan, and Baoquan Chen. Slam3r: Real-time dense scene reconstruction from monocular rgb videos. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 16651–16662, 2025.

Payal Mohadikar, Chuanmao Fan, and Ye Duan. Ms360: A multi-scale feature fusion framework for 360 monocular depth estimation. In Proceedings ofthe 50th Graphics Interface Conference, pages 1–11, 2024.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Jing Ou, Zidong Cao, Yinrui Ren, Zhuoxiao Li, Jinjing Zhu, Tongyan Hua, Shuai Zhang, Hui Xiong, and Wufan Zhao. Holo360d: A large-scale real-world dataset with continuous trajectories for advancing panoramic 3d reconstruction and beyond, 2026. URL https://arxiv.org/abs/ 2604.22482.

Chi-Han Peng and Jiayao Zhang. High-resolution depth estimation for 360deg panoramas through perspective and panoramic depth images registration. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3116–3125, 2023.

René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020.

Jiahui Ren, Mochu Xiang, Jiajun Zhu, and Yuchao Dai. Panosplatt3r: Leveraging perspective pretraining for generalized unposed wide-baseline panorama reconstruction. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 28959–28969, 2025.

Manuel Rey-Area, Mingze Yuan, and Christian Richardt. 360monodepth: High-resolution 360deg monocular depth estimation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3762–3772, 2022.

Cheng Sun, Min Sun, and Hwann-Tzong Chen. Hohonet: 360 indoor holistic understanding with latent horizontal features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2573–2582, 2021.

Xianbing Sun, Zhikai Zhu, Zhengyu Lou, Bo Yang, Jinyang Tang, Liqing Zhang, He Wang, and Jianfu Zhang. Avggt: Rethinking global attention for accelerating vggt. arXiv preprint arXiv:2512.02541, 2025.

Keisuke Tateno, Nassir Navab, and Federico Tombari. Distortion-aware convolutional filters for dense prediction in panoramic images. In Proceedings of the European Conference on Computer Vision (ECCV), pages 707–722, 2018.

Fu-En Wang, Hou-Ning Hu, Hsien-Tzu Cheng, Juan-Ting Lin, Shang-Ta Yang, Meng-Li Shih, Hung-Kuo Chu, and Min Sun. Self-supervised learning of depth and camera motion from 360 videos. In Asian Conference on Computer Vision, pages 53–68. Springer, 2018.

Fu-En Wang, Yu-Hsuan Yeh, Min Sun, Wei-Chen Chiu, and Yi-Hsuan Tsai. Bifuse: Monocular 360 depth estimation via bi-projection fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 462–471, 2020.

Fu-En Wang, Yu-Hsuan Yeh, Yi-Hsuan Tsai, Wei-Chen Chiu, and Min Sun. Bifuse++: Selfsupervised and efficient bi-projection fusion for 360 depth estimation. IEEE transactions on pattern analysis and machine intelligence, 45(5):5448–5460, 2022.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025a.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697–20709, 2024.

Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Scalable permutation-equivariant visual geometry learning. arXiv e-prints, pages arXiv–2507, 2025b.

Bangbang Yang, Yinda Zhang, Yijin Li, Zhaopeng Cui, Sean Fanello, Hujun Bao, and Guofeng Zhang. Neural rendering in a room: amodal 3d understanding and free-viewpoint rendering for the closed scene composed of pre-captured objects. ACM Transactions on Graphics (TOG), 41(4): 1–10, 2022.

Bangbang Yang, Wenqi Dong, Lin Ma, Wenbo Hu, Xiao Liu, Zhaopeng Cui, and Yuewen Ma. Dreamspace: Dreaming your room space with text-driven panoramic texture propagation. In 2024 IEEE Conference Virtual Reality and 3D User Interfaces (VR), pages 650–660. IEEE, 2024.

Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21924–21935, 2025.

Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 204–213, 2021.

Jiayi Yuan, Haobo Jiang, De Wen Soh, and Na Zhao. Vggt-360: Geometry-consistent zero-shot panoramic depth estimation. arXiv preprint arXiv:2603.18943, 2026.

Ilwi Yun, Chanyong Shin, Hyunku Lee, Hyuk-Jae Lee, and Chae Eun Rhee. Egformer: Equirectangular geometry-biased transformer for 360 depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6101–6112, 2023.

Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3d: A large photo-realistic dataset for structured 3d modeling. In European Conference on Computer Vision, pages 519–535. Springer, 2020.

Xu Zheng, Jinjing Zhu, Yexin Liu, Zidong Cao, Chong Fu, and Lin Wang. Both style and distortion matter: Dual-path unsupervised domain adaptation for panoramic semantic segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1285–1295, 2023.