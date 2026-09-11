# ReconPlusGen: Injecting Reconstruction Prior into Multi-view 3D Generation through Noise Inversion and Modulation

Jiarui Liu<sup>1\*</sup> Heng Li<sup>1\*</sup> Weiyu Li<sup>12</sup> Keng Deng<sup>1</sup> Junyuan Deng<sup>1</sup> Zheng Zhongxing<sup>3</sup> Junyu Huang<sup>3</sup> Jiahao Chang<sup>4</sup> Xiaoguang Han<sup>4</sup> Ping Tan<sup>1†</sup> <sup>1</sup>HKUST <sup>2</sup>LightIllusions <sup>3</sup>BYD <sup>4</sup>CUHK-Shenzhen <sup>\*</sup>Core contributions <sup>†</sup>Corresponding author.

![](images/544eb00e2256d510bc400059d32167f7be342af5f7da4ac14d265e21e3fcdf1d.jpg)  
Figure 1. Qualitative results and an illustration of our core idea. Top left: reconstruction results on benchmark images. Top right: reconstruction results on real-world images. Bottom: illustration of reconstruction-guided noise initialization and modulation. Given multiple input images, we predict a point cloud in canonical space, deterministically inject the predicted geometry into the diffusion process through noise inversion, and modulate the resulting noise to preserve the generative flexibility required to complete unobserved regions and refine visible geometry.

## Abstract

Multi-view 3D reconstruction and 3D generation offer complementary properties: reconstruction models preserve observation-grounded geometry but often fail to recover unobserved regions, whereas generative models synthesize plausible 3D assets but struggle to maintain pixel-level geometric fidelity because of the stochastic nature of diffusion. Existing methods attempt to combine the strengths by conditioning generative models on reconstruction features but still treat both observed and unobserved regions stochastically. We present RECONPLUSGEN, a unified framework that deterministically injects reconstruction priors into a

3D generative model through its initial-noise space. Given a set of unposed images, our approach reconstructs the visible geometry and transforms it into a structured diffusion initialization via noise inversion. To adaptively balance reconstruction fidelity and generative flexibility, we introduce a confidence-guided, spatially varying noise modulation scheme that strongly constrains reliable regions, relaxes constraints in uncertain regions, and preserves generative freedom in unobserved regions. Finally, a multiview-conditioned diffusion model refines geometric details while maintaining consistency with the input observations. Extensive experiments on synthetic and real-world benchmarks demonstrate that RECONPLUSGEN consistently outperforms existing reconstruction-guided and generationbased approaches. These results establish initial-noise control as a simple yet effective mechanism for faithful, highquality 3D asset generation from multi-view observations.

## 1. Introduction

3D object reconstruction has long been a fundamental problem in 3D computer vision, with broad applications in VR/AR and content creation. Conventional reconstruction methods [37–39] rely on reliable cross-view correspondences to recover the 3D structure. Such pipelines are brittle in low-texture regions, and need multi-stage processing or iterative optimization [13, 15]. Recent learning-based methods have made strong progress in sparse-view 3D object reconstruction. Regression models [7, 30, 46, 48, 59] infer 3D attributes from unposed image sets, improving performance under challenging imaging conditions. Nevertheless, these approaches are inherently observation-bounded: they can only predict 3D attributes for observed pixels, resulting in incomplete reconstructions.

Recent advances in diffusion-based 3D generative models [27, 41, 43, 60, 63] offer a compelling way to predict complete shapes from limited observations. By learning strong 3D priors from large-scale 3D data, these methods can generate complete 3D content conditioned on sparseview images. Such generative priors can synthesize unobserved regions with high-quality geometry and appearance, improving reconstruction completeness by filling in missing structures and details. Some works [3, 5] attempt to improve the pixel-level reconstruction alignment by replacing 2D image features with 3D point cloud features. However, diffusion inference conditioned on features is inherently stochastic, which introduces substantial uncertainty and sample-to-sample variability. This variability makes it difficult to achieve the pixel-level and multi-view alignment needed for precise geometric consistency.

Instead of conditioning the denoising process on geometry features, we take a different perspective: our geometrygrounded generation framework, RECONPLUSGEN, injects deterministic reconstruction cues at the source of stochasticity—the initial noise of the generative process. Specifically, we first train a VGGT-style geometry predictor, termed Canonical-Aligned VGGT (CA-VGGT), to estimate geometry directly in a shared canonical space from unposed observations, providing an initial shape for noise inversion. We then apply confidence-aware spatial noise modulation to inversion noise to enforce geometric fidelity in reliable regions, relax constraints in uncertain regions, and preserve generative flexibility in unobserved regions. Finally, a multi-view diffusion recovers fine-grained geometric details while maintaining global and cross-view consistency from the modulated initial noise. This design yields a controllable yet expressive generation process that improves both the accuracy of the reconstruction and the perceptual quality.

Extensive experiments on synthetic and real-world image benchmarks demonstrate that RECONPLUSGEN achieves state-of-the-art performance in unposed sparseview settings. Compared with prior feature-conditioned pipelines, our method provides stronger geometric fidelity, improved image alignment, and more robust completion of unseen regions. Our contributions are summarized as follows:

• We introduce a novel paradigm for geometry injection in 3D diffusion models: reconstructing visible geometry in an object-centric canonical space and inverting it into a corresponding initial noise to provide a deterministic anchor for the diffusion trajectory.

• We propose a confidence-aware, spatially varying noise modulation scheme that adaptively controls the influence of the reconstructed geometry according to its estimated reliability, preserving fidelity in confident regions while retaining generative flexibility in uncertain and unobserved regions.

• We achieve state-of-the-art performance across diverse sparse-view 3D object reconstruction benchmarks.

## 2. Related Works

## 2.1. 3D Reconstruction

Traditional Optimization-based reconstruction methods [49, 61, 62] can recover highly accurate geometry, but typically rely on dense observations and known camera parameters. Recent approaches [13, 15] extend this paradigm to sparse-view settings by incorporating priors from pretrained geometry models. Nevertheless, their costly per-scene optimization still limits both computational efficiency and applicability to downstream tasks. Recent feed-forward models [10, 50] directly predict depth maps and camera parameters from unposed images. Subsequent approaches [7, 9, 33, 48] improve scalability to multi-view inputs and even support large-scale reconstruction from thousands of images. Despite their strong performance, these models remain observation-bounded: they primarily recover geometry corresponding to visible pixels and do not explicitly address complete object reconstruction.

Different from scene-level reconstruction methods above, object-centric reconstruction methods instead model object shapes in a normalized canonical space, usually defined in [−1, 1]. LucidFusion [17] proposes to predict object geometry in canonical space coordinates from sparse-view images. Large reconstruction models [18, 26, 40, 52, 56, 58] further focus on recovering complete geometries. However, their outputs are often overly smooth and fail to preserve fine geometric details.

![](images/710b2f2d5536f3f1dfe0b8c615cf89d2eef1fdaf45878f99ff8552e08cdff17f.jpg)  
Figure 2. Pipeline of RECONPLUSGEN. Given multiple input images, our canonical-aligned geometry prediction model first reconstructs the visible geometry directly in canonical space. We then inject the predicted geometry into the diffusion process through noise inversion and modulation. Starting from the modulated noise, a multi-view image-conditioned diffusion model generates the final shape while preserving observed geometry and completing unobserved regions.

## 2.2. 3D Generative Models

Large-scale 3D generative models provide a complementary direction by learning strong shape and appearance priors from extensive 3D datasets. Shape2VecSet [60] encodes signed distance fields into compact sets of latent tokens, enabling diffusion models to operate directly in this compressed representation space. Building on this paradigm, Clay [63] generates complete textured 3D meshes from a single image. Subsequent works [8, 27, 29, 42, 55, 67] further improve generation quality through advanced architectures and sparse representations. However, most of these methods are conditioned on a single image or assume restrictive camera configurations [42], and their generated assets may deviate from the observed geometry.

To support arbitrary unposed input images, previous approaches bridge reconstruction and generation by conditioning the denoising process on geometry-aware features [3]. Some of them improve geometric consistency by finding 2D-3D correspondences [5, 19, 25], others [21, 31] explicitly learn the alignment between reconstruction and generation representations. Although these designs improve controllability, they still operate mainly at feature condition level. Therefore, structural sampling can remain unstable due to stochastic diffusion dynamics. Our method differs by injecting reconstruction priors into the initial noise, which provides stronger geometric control with less reliance on strict calibration.

## 2.3. Initial Noise in Diffusion Models.

Recent studies on diffusion models show that the initial noise strongly affects the generation result. Pioneer works [23, 57] found that the noise space itself contains structural cues, motivating research in test-time optimization, learned noise priors, and diffusion inversion. Test-time methods [2, 16, 44] use gradient-based optimization to refine the initial noise for better alignment, while Reno [11] extends this idea to one-step models with multiple reward signals. However, these methods introduce additional computation during inference. To reduce this cost, several approaches [1, 12, 66] train lightweight networks to predict improved noise in a single pass. DeepInv [64] uses a selfsupervised trainable solver for faster image-to-noise mapping. NoiseAR [28] instead learns an autoregressive noise prior for controllable generation and reinforcement learning. Meanwhile, diffusion inversion methods map an image back to its initial noise. This has also become a common foundation for image editing [14, 35, 45]. Recent works further improve inversion fidelity, editability, and efficiency through different schedules [20, 22], high-order solvers for rectified-flow models [47], and learned one-step inversion [36]. Together, these studies show that controlling the initial noise can improve both generation quality and controllability. Inspired by this finding, we extend initial-noise control to reconstruction-guided 3D generation by converting predicted geometry into structured noise.

![](images/b6ffb5b58ab4faa1b876eaa108730aa122ffd9c614b31a0ad23ec44e744ac2cd.jpg)  
Figure 3. Spatial Locality of 3DShape2VecSet [60] based diffusion models. (a) Given a partial point cloud, (b) we sample query points q using farthest point sampling [34] (c) We encode the point <sup>Partial</sup> <sup>Point</sup> <sup>Cloud</sup> <sup>Query</sup> <sup>Points</sup>  <sub>Decoded Mesh Inversion</sub> <sub>and</sub> <sub>D</sub>cloud into the latent space with q, and decode it into a mesh. (d) We perform noise inversion [47] and recover the shape through denoising and decoding.

## 3. Method

We present RECONPLUSGEN, a geometry-grounded 3D <sup>Noise</sup> <sup>Inversion</sup> <sup>and</sup> <sup>Denoising</sup>object generation framework that takes a set of unposed input images and produces a high-fidelity, pixel-aligned, and multi-view-consistent 3D mesh. Given a set of unposed images, we first recover camera poses and reconstruct a point cloud in the canonical space via Canonical-Aligned VGGT. Building upon this geometric prior, we describe in Sec. 3.1 how to obtain a deterministic initial noise from the reconstructed point cloud. Starting from this geometryinformed initialization, we further modulate the initial noise in Sec. 3.2 to facilitate the recovery of geometry in unobserved regions. Finally, Sec. 3.3 details the diffusion process conditioned on multi-view inputs with camera embeddings. We adopt the 3DShape2VecSet [60] based 3D generation framework [43] as our backbone.

## 3.1. Noise Initialization with Reconstruction

Deterministic Noise Initialization. Starting from a random initial noise, a common practice for 3D generation [3, 25] is to inject feature conditions via cross-attention at each diffusion step. However, such feature conditioning is often insufficient for precise geometric control because the initial state is still unconstrained noise, introducing stochastic variation in both visible and unobserved regions, as shown in Fig. 4. Recent studies have shown that diffusion outputs are strongly determined by initial noise [1, 12, 28, 66]. This observation motivates us to inject the condition in an alternative manner: recovering the corresponding deterministic noise from a given explicit geometry prior.

Given a reconstructed point cloud S, we first encode it in the latent representation $Z _ { 0 } = \mathrm { E n c o d e r } ( S , q )$ and then recover the corresponding initial noise $Z ^ { \mathrm { i n v } } ~ = ~ I n v ( Z _ { 0 } )$ through noise inversion, denoted by Inv(·). Here, the query points q are selected from S using farthest point sampling [34] method and provided as input to the shape encoder. We adapt the inversion scheme of [47] to native 3D diffusion models, approximating the velocity term through

a Taylor expansion:

$$
Z _ { t _ { i + 1 } } = Z _ { t _ { i } } + \sum _ { k = 0 } ^ { n - 1 } \frac { ( ( t _ { i + 1 } - t _ { i } ) ^ { k + 1 } ) } { ( k + 1 ) ! } v _ { \theta } ^ { ( k ) } ( Z _ { t _ { i } } , t _ { i } ) + \mathcal { O } ( h _ { i } ^ { n + 1 } ) ,\tag{1}
$$

where $Z _ { t _ { i + 1 } }$ denotes the latent vector at timestep $i , v _ { \theta } ^ { ( k ) } ( \cdot )$ denotes the k-th derivative of the velocity, $v _ { \theta } ^ { ( 0 ) } ( \cdot )$ is predicted by flow-matching model $\mathcal { D } _ { \theta } . ~ \mathcal { O } ( h _ { i } ^ { n + 1 } )$ is the Peano remainder which is omitted in the computation. n denotes the Taylor expansion order, where we empirically set the expansion order at $n = 2$ . We approximate the higher-order terms using finite differences:

$$
v _ { \theta } ^ { ( k + 1 ) } ( Z _ { t } , t ) \approx \frac { v _ { \theta } ^ { ( k ) } ( Z _ { t } , t ) - v _ { \theta } ^ { ( k ) } ( Z _ { t + \Delta t } , t + \Delta t ) } { \Delta t }\tag{2}
$$

We visualize the input point cloud S, the mesh reconstructed by the decoder $M = { \mathrm { D e c o d e r } } ( Z _ { 0 } )$ , and the mesh generated from the inverted noise $M = { \mathrm { D e c o d e r } } ( { \mathrm { D i T } } ( Z _ { T } ) )$ in Fig. 3. The results demonstrate that the inverted noise initialization can faithfully preserve the given geometry prior: the reconstructed region aligns well with the input point cloud, while the unobserved regions remain empty.

Canonical Point Cloud Generation. Another challenge in noise inversion lies in the misalignment between the coordinate system of the input reconstructed point cloud and the canonical space required by the diffusion network. The point cloud from the estimated method is usually defined up to a scale [38] or with respect to a specific frame coordinate [46]. Directly using such point clouds for noise inversion leads to a catastrophic failure of diffusion prediction due to the discrepancy in the input domains, as noted in RecGen3D [21]. To address this misalignment of coordinate system, we train a Canonical-Aligned VGGT (CA-VGGT) to estimate geometry from unposed images in canonical space. Instead of assuming the camera pose of the first frames as identity, we train CA-VGGT directly in the standard canonical coordinate range of [−1, 1]. We predict point clouds and camera poses with different heads, as well as the corresponding confidence maps, denoted as C. Those confidence maps are designed to be proportional to the model’s prediction error [46] and to provide an indicator of reconstruction quality in Sec 3.2. Please refer to the supplementary material for more details about CA-VGGT.

As a result, for a given set of input images, our method identifies a deterministic diffusion initialization that produces a shape well aligned with the observation, eliminating frame-dependent position, scale, and pose ambiguity and reducing stochastic variation in visible regions, thereby substantially improving reconstruction-to-generation consistency.

![](images/b702e83f67e21012b5a71d016174749e20403b084753dc6951e8249a2e9b1f00.jpg)  
Figure 4. Continuity of noise space in a 3DShape2VecSet [60] based diffusion model. We compute inversion noise from ground truth mesh(GT mesh), interpolate between the resulting inversion noise and a random noise, and conduct denoising starting from each interpolated noise. Both inversion and denoising process is conditioned with the same image(shown in rightmost).

## Inversion Noise Patial Tokens Complete Tokens3.2. Confidence-aware Noise Modulation

Sec. 3.1 provides a deterministic initialization that anchors the diffusion process to the given geometry prior. However, the point cloud estimated from the images is inherently incomplete and contains prediction errors, which leads to inferior quality of the generated mesh. To address these limitations, we introduce a spatially varying, confidence-aware noise modulation scheme that preserves reliable geometric cues while relaxing constraints in uncertain and unobserved regions, with respect to two key properties of the 3DShape2VecSet [60] based diffusion models: Spatial Locality and Noise Initialization Continuity.

Spatial Locality. As shown in Fig. 3, each query point in q constitutes a latent token that controls a spatially localized region during encoding. Regions devoid of query points cannot be adequately represented by latent tokens. This property remains after noise inversion. Consequently, we selectively employ a subset of query points (tokens) $\tilde { q } \subset q$ to encode the observed portion of the point cloud, perform noise inversion, while leaving the remaining tokens as random noise. This strategy, termed noise padding (Noise-Pad), enables the model to synthesize geometry in unseen regions from scratch:

$$
Z _ { T } = [ Z ^ { \mathrm { i n v } } = i n v ( \operatorname { E n c o d e r } ( S , \tilde { q } ) ) , ~ Z ^ { \mathrm { r a n d o m } } ] .\tag{3}
$$

Note that determining the completeness ratio of the input point cloud is non-trivial. For simplicity, we use 2, 048 query points and 2, 048 random tokens, forming a total of 4, 096 latent tokens for the 3DShape2VecSet VAE [43].

Noise Initialization Continuity. Although deterministic noise initialization on the given points provides a strong prior for 3D generation, a noisy or inaccurate prior can lead to inferior results. Note that both $Z ^ { \mathrm { i n v } }$ and $Z ^ { \mathrm { { r a n d o m } } }$ are drawn from $\mathcal { N } ( 0 , I )$ . By the closure property of Gaussian distributions under linear combinations, any convex combination of them remains Gaussian. This motivates us to interpolate between $Z ^ { \mathrm { i n v } }$ and $Z ^ { \mathrm { { r a n d o m } } }$ according to the reconstruction quality of each token. The confidence map $C$ predicted by CA-VGGT provides a natural basis for determining the interpolation weights: it reflects the reliability of the reconstructed geometry and is spatially aligned with each point in the reconstructed point cloud S. We therefore use these confidence scores to modulate the geometric constraints as follows:

$$
Z ^ { \mathrm { r e c o n } } = { \sqrt { \alpha } } \cdot Z ^ { \mathrm { i n v } } + { \sqrt { ( 1 - \alpha ) } } \cdot Z ^ { \mathrm { r a n d o m } } ,\tag{4}
$$

where $\alpha = f ( C ( \tilde { q } ) ) , C ( \tilde { q } )$ denotes the confidence scores associated with the selected query points, and $f ( \cdot )$ is a linear function that maps confidence values to interpolation weights: $f ( c ) = c ^ { m i n } + ( 1 - c ^ { m a x } ) \cdot n ( c )$ . Here, $n ( c ) \in [ 0 , 1 ]$ denotes the normalized value of $c .$ We visualize the effect of interpolating $Z ^ { \mathrm { r e c o n } }$ in Fig. 4. Starting from each interpolated noise, we perform denoising under the same image conditioning and decode the resulting latent representation into a mesh. As the initialization shifts from inversion noise toward random noise, the generated shapes change smoothly, demonstrating the continuity of the noise space. This transition gradually transfers geometric control from the reconstruction prior encoded in the inversion noise to the learned generative prior. Consequently, high-confidence tokens remain close to the inversion noise, whereas low-confidence tokens contain more random noise and can be refined more freely by the generative model. Finally, we replace $Z ^ { \mathrm { i n v } }$ in Eq. 3 with $Z ^ { \mathrm { r e c o n } }$ , yielding the following modulated noise tokens:

$$
Z _ { T } = [ Z ^ { \mathrm { r e c o n } } , Z ^ { \mathrm { r a n d o m } } ] .\tag{5}
$$

Overall, the modulated tokens preserve the reconstructed geometry in reliable regions while delegating the refinement of uncertain regions and the completion of unseen regions to the learned diffusion prior, thereby achieving a favorable balance between input alignment and geometric quality.

## 3.3. Multi-View Diffusion Refinement

Starting from the noise in Eq. 5, the original single-view diffusion model without fine-tuning is already sufficient to generate the complete mesh, making our method entirely training-free. However, a multi-view diffusion model remains indispensable for complementing the reconstruction by correcting residual errors and enriching fine-scale geometric details. To this end, we extend the single-view image conditioning to a multi-view setting by incorporating camera pose embeddings. Specifically, we convert the camera poses $\tilde { \mathbf { P } }$ estimated by CA-VGGT into view-specific Plucker ¨ ray maps as explicit pose embeddings. These pose embeddings are added to the DINO image features channel-wise. We then concatenate the conditioning tokens across views and feed them into the diffusion model:

$$
V = { \mathcal { D } } _ { \theta } \left( Z ; { \mathrm { D I N O } } ( \mathbf { I } ) \oplus { \mathrm { P l i c k e r E m b e d } } ( { \tilde { \mathbf { P } } } ) \right) ,\tag{6}
$$

where $\mathcal { D } _ { \theta }$ denotes the multi-view diffusion model, Z is the latent token set at timestep t, I denotes the input RGB images, and P<sup>˜</sup> denotes the corresponding camera poses. The operator ⊕ indicates channel-wise addition. This design allows the diffusion model to exploit both semantic context and explicit geometric cues, facilitating the recovery of finegrained surface details while maintaining multi-view consistency.

## 4. Experiments

## 4.1. Implementation Details

Training. We select 130K high-quality meshes from Objaverse [6] as training set and render 60 random views per mesh with camera jittering. This dataset is used for training both CA-VGGT and the multi-view diffusion model. All meshes are normalized so that their vertex coordinates fall within [−1, 1]. We initialize CA-VGGT with the pretrained weights of VGGT [46]. During training, we dynamically sample one to eight input views for each batch. Within each batch, we randomly select one image as the first frame defining the canonical axes and transform the camera poses and point clouds of all views into the corresponding canonical coordinate system. We train CA-VGGT on 16 H100 GPUs for 17k iterations using a learning rate of $1 \times 1 0 ^ { - 5 }$ For the multi-view diffusion model, we fine-tune the diffusion model from Hunyuan3D-2.1 [43] with the multi-view condition for 110K iterations on 16 H100 GPUs.

Evaluation. Following ReconViaGen [3], we randomly sample 300 objects from DoraBench [4], 200 objects from OmniObject3D [54] and 300 objects from Objaverse [6]. For DoraBench, we render 24 images per object using the TRELLIS [55] rendering protocol. For OmniObject3D, we randomly select four views from the 24-view Blender renderings contained in the original dataset. We evaluate on a held-out set of meshes sampled from Objaverse, ensuring no overlap with the training data, and render them using the same configuration as the training set. We adopt the $\ell _ { 1 }$ Chamfer distance and the F-score at a threshold of 0.1 as evaluation metrics. Before computing these metrics, we rescale each prediction and align it with the ground-truth shape using a scaled ICP initialized from four different orientations, thus mitigating the scale and orientation ambiguities introduced by different reconstruction models.

## 4.2. Quantitative Results

Comparison with Baseline Methods. We compare our method with four categories of baselines: reconstructionbased methods, image-conditioned 3D generation models, point-cloud-conditioned generation models, and recent approaches that integrate reconstruction and generation models. We report the results in Tab. 1 and Tab. 2.

The reconstruction-based baselines include large feedforward reconstruction models [40, 56] and optimizationbased reconstruction methods [13, 15]. Large reconstruction models produce coarse mesh surfaces due to the limited representational capacity of its triplane representation, while optimization-based methods fail to generate reasonable surfaces in unseen regions and result in unsatisfactory indicators even with ground-truth camera poses.

For 3D generative models, we compare against singleview-conditioned approaches [27, 43] and multi-viewconditioned approaches [42, 55]. Although these methods achieve marginally better quantitative results, as shown in Fig. 5, they tend to recover incorrect geometric details due to the lack of an explicit geometric prior.

Point-cloud-conditioned methods [41, 53, 65] require explicit point clouds as input and reconstruct the corresponding complete mesh. We therefore use the point clouds predicted by CA-VGGT as input. The aforementioned methods fail to achieve satisfactory alignment with the input observations due to their stochastic nature, which demonstrates that our deterministic noise initialization can provide stronger control given a geometric prior.

We also compare our method with recent approaches that combine reconstruction and generation models [3, 24, 31]. ReconViaGen [3] and Mix3R [31] combine reconstruction prior by conditioning the generation process on geometry regression features [46, 51], While MV-SAM3D [24] fuse conditions of each view by adaptive weighting strategies. Our method outperforms this feature-conditioning approach by deterministically injecting reconstruction prior into the visible regions while retaining the flexibility to complete and refine the remaining part.

To further explore the full potential of our approach, we additionally evaluate our method using a ground-truth visible point cloud as the reconstruction prior. The results in this setting provide an empirical upper bound and demonstrate that improvements in reconstruction quality directly translate into better generation performance. Overall, our method consistently outperforms all competing approaches by a clear margin, and the advantages become more pronounced as the number of views increases.

Different Number of Views. To evaluate robustness with respect to the number of input views, we compare our method with baseline methods that are acceptable with different numbers of views under varying view counts. We exclude LGM [40] and Hunyuan2.0-MV because they are designed for exactly four input views captured from predefined poses. As demonstrated in Tab. 2, in general, our method consistently achieves strong performance across all view counts, demonstrating its robustness to different observation settings. In particular, compared to recent methods that combine reconstruction and generation, our advantage becomes more pronounced as the number of input views increases. We attribute this trend to the fact that crossattention-based approaches must aggregate a growing number of conditioning features, whereas our method converts the increasingly accurate multi-view reconstruction into a fixed-size set of initial-noise tokens, allowing it to benefit more effectively from additional observations.

Table 1. Quantitative comparison on three benchmarks with the 4-view setting. We report Chamfer Distance (CD, ↓) and F-Score (↑, threshold=0.1). Best results under the standard setting are shown in bold. Results shown in gray report the empirical upper bound obtained using ground-truth point clouds. ✓ indicates methods that require ground-truth camera poses, and <sup>∗</sup> indicates methods conditioned on point clouds predicted by CA-VGGT.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="2">Objaverse</td><td colspan="2">Dorabench</td><td colspan="2">OmniObject3D</td></tr><tr><td>CD↓</td><td>F-Score ↑</td><td>CD↓</td><td>F-Score ↑</td><td>CD↓</td><td>F-Score ↑</td></tr><tr><td rowspan="4">Reconstruction Methods</td><td>LGM [40] √</td><td>0.246</td><td>0.562</td><td>0.077</td><td>0.918</td><td>0.090</td><td>0.878</td></tr><tr><td>InstantMesh [56] √</td><td>0.242</td><td>0.581</td><td>0.094</td><td>0.874</td><td>0.071</td><td>0.934</td></tr><tr><td>MAtCha GS [15] √</td><td>0.124</td><td>0.814</td><td>0.097</td><td>0.863</td><td>0.104</td><td>0.868</td></tr><tr><td>DP-GS [13] √</td><td>0.140</td><td>0.776</td><td>0.099</td><td>0.859</td><td>0.113</td><td>0.841</td></tr><tr><td rowspan="4">3D Generative Models</td><td>Craftsman [27]</td><td>0.141</td><td>0.774</td><td>0.123</td><td>0.820</td><td>0.109</td><td>0.849</td></tr><tr><td>Hunyuan2.1 [43]</td><td>0.087</td><td>0.880</td><td>0.084</td><td>0.893</td><td>0.078</td><td>0.913</td></tr><tr><td>Hunyuan2.0-MV [42]</td><td>0.085</td><td>0.890</td><td>0.092</td><td>0.87</td><td>0.075</td><td>0.924</td></tr><tr><td>Trellis-MV [55]</td><td>0.088</td><td>0.880</td><td>0.065</td><td>0.941</td><td>0.070</td><td>0.940</td></tr><tr><td rowspan="3">Point Cloud Conditioned Models</td><td>HunyuanOmni [41]*</td><td>0.077</td><td>0.905</td><td>0.076</td><td>0.905</td><td>0.059</td><td>0.945</td></tr><tr><td>DeepMesh [65]*</td><td>0.117</td><td>0.803</td><td>0.141</td><td>0.774</td><td>0.143</td><td>0.789</td></tr><tr><td>BPT [53]*</td><td>0.101</td><td>0.839</td><td>0.112</td><td>0.823</td><td>0.086</td><td>0.884</td></tr><tr><td rowspan="5">Combined Methods</td><td>ReconViaGen [3]</td><td>0.053</td><td>0.961</td><td>0.052</td><td>0.965</td><td>0.052</td><td>0.976</td></tr><tr><td>Mix3R [31]</td><td>0.073</td><td>0.912</td><td>0.056</td><td>0.949</td><td>0.054</td><td>0.960</td></tr><tr><td>MV-SAM3D [24]</td><td>0.078</td><td>0.907</td><td>0.061</td><td>0.945</td><td>0.075</td><td>0.926</td></tr><tr><td>Ours*</td><td>0.040</td><td>0.974</td><td>0.042</td><td>0.966</td><td>0.044</td><td>0.981</td></tr><tr><td>Ours(GT Point Cloud)</td><td>0.017</td><td>0.990</td><td>0.026</td><td>0.982</td><td>0.020</td><td>0.991</td></tr></table>

Table 2. Quantitative comparison on Dorabench with baseline methods with different numbers of views. We report Chamfer Distance (CD, ↓) and F-Score (↑, threshold=0.1). Best results are shown in bold. In the Type column, R, G, and C denote Reconstruction Methods, Multi-view Conditioned 3D Generative Models, and Combined Methods, respectively. ✓ indicates methods that require ground-truth camera poses.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="2">2 views</td><td colspan="2">8 views</td><td colspan="2">16 views</td><td colspan="2">24 views</td></tr><tr><td>CD↓</td><td>F-Score ↑</td><td>CD↓</td><td>F-Score ↑</td><td>CD↓</td><td>F-Score ↑</td><td>CD↓</td><td>F-Score ↑</td></tr><tr><td>R</td><td>InstantMesh [56] √</td><td>0.083</td><td>0.908</td><td>0.070</td><td>0.932</td><td>0.072</td><td>0.928</td><td>0.154</td><td>0.758</td></tr><tr><td>R</td><td>MAtCha GS [15] √</td><td>0.169</td><td>0.723</td><td>0.059</td><td>0.942</td><td>0.040</td><td>0.963</td><td>0.040</td><td>0.960</td></tr><tr><td>R</td><td>DP-GS [13] √</td><td>0.251</td><td>0.593</td><td>0.061</td><td>0.925</td><td>0.042</td><td>0.967</td><td>0.042</td><td>0.964</td></tr><tr><td>G</td><td>Trellis-MV [55]</td><td>0.066</td><td>0.931</td><td>0.057</td><td>0.952</td><td>0.063</td><td>0.940</td><td>0.084</td><td>0.901</td></tr><tr><td>C</td><td>ReconViaGen [3]</td><td>0.062</td><td>0.952</td><td>0.047</td><td>0.971</td><td>0.044</td><td>0.977</td><td>0.046</td><td>0.974</td></tr><tr><td>C</td><td>Mix3R [31]</td><td>0.068</td><td>0.929</td><td>0.054</td><td>0.955</td><td>0.054</td><td>0.955</td><td>0.054</td><td>0.957</td></tr><tr><td>C</td><td>MV-SAM3D [24]</td><td>0.070</td><td>0.927</td><td>0.055</td><td>0.955</td><td>0.056</td><td>0.956</td><td>0.056</td><td>0.953</td></tr><tr><td>C</td><td>Ours</td><td>0.056</td><td>0.946</td><td>0.031</td><td>0.978</td><td>0.027</td><td>0.982</td><td>0.029</td><td>0.980</td></tr></table>

## 4.3. Visualization Results

We provide qualitative comparisons to further demonstrate the effectiveness of our approach. To visualize geometric consistency, we compute the per-point distance error and color-code the errors on the generated surfaces, clipping values to [0, 0.1]. As shown in Fig. 5, geometry-aware initialization effectively complements the generative prior, producing shapes that align more faithfully with the input observations. The improvement is particularly evident in fine structural details and repeated patterns, such as the number of rocket support legs and the pillars on the back of the chair.

Table 3. Ablation study of the individual components. We conduct experiments using both a single-view diffusion model [43] and our multi-view diffusion model. Noise modulation is decomposed into noise padding and noise interpolation. In VGGT+ICP mode, we generate an anchor mesh first and align the point cloud predicted by origion VGGT [46] to the anchor using scaled ICP.
<table><tr><td rowspan="2">Method</td><td colspan="4">Diffusion Model</td></tr><tr><td>Single-view Diffusion</td><td></td><td>Multi-view Diffusion</td><td></td></tr><tr><td></td><td>CD↓</td><td>F-score↑</td><td>CD↓</td><td>F-score↑</td></tr><tr><td>VGGT [46]+ICP</td><td>0.107</td><td>0.848</td><td>0.110</td><td>0.846</td></tr><tr><td>Random Noise</td><td>0.084</td><td>0.893</td><td>0.060</td><td>0.944</td></tr><tr><td>Random Noise(GT Cam Pose)</td><td></td><td></td><td>0.059</td><td>0.944</td></tr><tr><td>w Noise Inv</td><td>0.070</td><td>0.916</td><td>0.056</td><td>0.952</td></tr><tr><td>w Noise Padding</td><td>0.049</td><td>0.956</td><td>0.048</td><td>0.964</td></tr><tr><td>w Noise Interp</td><td>0.045</td><td>0.965</td><td>0.042</td><td>0.966</td></tr></table>

To evaluate robustness in real-world scenarios, we further test our model on in-the-wild images, as shown in Fig. 1. We extract the foreground objects using an existing segmentation method [32] before applying our reconstruction pipeline. The results demonstrate that our method generalizes effectively to natural out-of-domain inputs.

## 4.4. Ablation Studies

To isolate the contribution of each component, we conduct ablation studies on the same 300 Dorabench objects under the 4-view setting and report the results in Tab. 3. We evaluate our design with both the off-the-shelf single-view diffu-

Input Images

Hunyuan2.1

Mix3R

Error:

MV-SAM3D

ReconViaGen

low(0)

high(0.1)

Ours

![](images/02f4a0c486be0a605761a94e67431252684362496d03e8336fa9fd114f7a6921.jpg)  
Figure 5. Qualitative comparison with baseline approaches. Per-point distance errors are color-coded to visualize geometric discrepancies. Our method achieves more faithful alignment with the input observations while better preserving fine-grained details and repeated structures. Refer to supplementary for more comparison results.

sion model [43] and our trained multi-view diffusion model. Experiments on single-view diffusion demonstrate that our noise-space injection mechanism can serve as a plug-andplay approach for incorporating multi-view geometric information into existing diffusion models without additional training. Our trained multi-view diffusion model provides further gains through explicit multi-view conditioning.

In both settings, directly using the unmodulated inversion noise overconstrains the diffusion process to the incomplete and imperfect CA-VGGT reconstruction. Noise padding preserves the capacity for completing unobserved regions, while confidence-aware noise interpolation reduces the influence of unreliable geometric predictions. Notably, noise interpolation provides a larger improvement with the multi-view diffusion model, which we attribute to the stronger guidance supplied by multi-view conditioning.

CA-VGGT resolves the coordinate mismatch between reconstruction and generation models by directly predicting point clouds in canonical space. To evaluate this design, we compare it with a post-hoc alignment baseline. Specifically, we first generate an anchor shape in canonical space from random noise and then align the point cloud reconstructed by the original VGGT to this anchor using scaled ICP. The inferior performance of this baseline highlights the importance of predicting geometry directly in canonical-space. We further evaluate the multi-view diffusion model using both ground-truth camera poses and poses predicted by CA-VGGT to assess its robustness to small pose errors.

## 5. Conclusion

In this paper, we introduce ReconPlusGen, a novel unified framework that bridges 3D reconstruction and generation by initializing the diffusion process with partial reconstructions. Unlike existing methods, we generate deterministic initial noise from partial reconstructions and assign different range of flexibility through noise modulation, effectively shielding observed regions from the stochasticity of diffusion models while leveraging generative imagination to complete unobserved parts. We validated our approach through extensive experiments. Both quantitative and qualitative results demonstrate that ReconPlusGen achieves state-of-the-art performance, significantly outperforming existing baselines.

## References

[1] Donghoon Ahn, Jiwon Kang, Sanghyun Lee, Jaewon Min, Minjae Kim, Wooseok Jang, Hyoungwon Cho, Sayak Paul, SeonHwa Kim, Eunju Cha, et al. A noise is worth diffusion guidance. arXiv preprint arXiv:2412.03895, 2024. 3, 4

[2] Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-flow: Differentiating through flows for controlled generation. arXiv preprint arXiv:2402.14017, 2024. 3

[3] Jiahao Chang, Chongjie Ye, Yushuang Wu, Yuantao Chen, Yidan Zhang, Zhongjin Luo, Chenghong Li, Yihao Zhi, and Xiaoguang Han. Reconviagen: Towards accurate multiview 3d object reconstruction via generation. arXiv preprint arXiv:2510.23306, 2025. 2, 3, 4, 6, 7

[4] Rui Chen, Jianfeng Zhang, Yixun Liang, Guan Luo, Weiyu Li, Jiarui Liu, Xiu Li, Xiaoxiao Long, Jiashi Feng, and Ping Tan. Dora: Sampling and benchmarking for 3d shape variational auto-encoders. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 16251– 16261, 2025. 6

[5] Yiwen Chen, Zhihao Li, Yikai Wang, Hu Zhang, Qin Li, Chi Zhang, and Guosheng Lin. Ultra3d: Efficient and highfidelity 3d generation with part attention, 2025. 2, 3

[6] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects, 2022. 6

[7] Junyuan Deng, Heng Li, Tao Xie, Weiqiang Ren, Qian Zhang, Ping Tan, and Xiaoyang Guo. Sail-recon: Large sfm by augmenting scene regression with localization. arXiv preprint arXiv:2508.17972, 2025. 2

[8] Ken Deng, Yuan-Chen Guo, Jingxiang Sun, Zi-Xin Zou, Yangguang Li, Xin Cai, Yan-Pei Cao, Yebin Liu, and Ding Liang. Detailgen3d: Generative 3d geometry enhancement via data-dependent flow. arXiv preprint arXiv:2411.16820, 2024. 3

[9] Kai Deng, Zexin Ti, Jiawei Xu, Jian Yang, and Jin Xie. Vggt-long: Chunk it, loop it, align it–pushing vggt’s limits on kilometer-scale long rgb sequences. arXiv preprint arXiv:2507.16443, 2025. 2

[10] Bardienus Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jerome Revaud. Mast3rsfm: a fully-integrated solution for unconstrained structurefrom-motion. arXiv preprint arXiv:2409.19152, 2024. 2

[11] Luca Eyring, Shyamgopal Karthik, Karsten Roth, Alexey Dosovitskiy, and Zeynep Akata. Reno: Enhancing one-step text-to-image models through reward-based noise optimization. Advances in Neural Information Processing Systems, 37:125487–125519, 2024. 3

[12] Luca Eyring, Shyamgopal Karthik, Alexey Dosovitskiy, Nataniel Ruiz, and Zeynep Akata. Noise hypernetworks: Amortizing test-time compute in diffusion models. arXiv preprint arXiv:2508.09968, 2025. 3, 4

[13] Bowen Gao, Zhicheng Lu, Mingyi He, and Yuchao Dai. Dpgs: Depth-prior & perception-guided gaussian splatting for sparse-view novel view synthesis. In 2025 Asia Pacific Signal and Information Processing Association Annual Summit

and Conference (APSIPA ASC), pages 1975–1980, 2025. 2, 6, 7

[14] Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel Cohen-Or. Renoise: Real image inversion through iterative noising. In European Conference on Computer Vision, pages 395–413. Springer, 2024. 3

[15] Antoine Guedon, Tomoki Ichikawa, Kohei Yamashita, and´ Ko Nishino. Matcha gaussians: Atlas of charts for high quality geometry and photorealism from sparse views. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6001–6011. IEEE, 2025. 2, 6, 7

[16] Xiefan Guo, Jinlin Liu, Miaomiao Cui, Jiankai Li, Hongyu Yang, and Di Huang. Initno: Boosting text-to-image diffu sion models via initial noise optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9380–9389, 2024. 3

[17] Hao He, Yixun Liang, Luozhou Wang, Yuanhao Cai, Xinli Xu, Hao-Xiang Guo, Xiang Wen, and Yingcong Chen. Lu cidfusion: Reconstructing 3d gaussians with arbitrary un posed images, 2025. 2

[18] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 2

[19] Binbin Huang, Haobin Duan, Yiqun Zhao, Zibo Zhao, Yi Ma, and Shenghua Gao. Cupid: Generative 3d reconstruction via joint object and pose modeling, 2025. 3

[20] Jiancheng Huang, Yi Huang, Jianzhuang Liu, Donghao Zhou, Yifan Liu, and Shifeng Chen. Dual-schedule inver sion: Training- and tuning-free inversion for real image editing. In IEEE/CVF Winter Conference on Applications of Computer Vision, 2025. 3

[21] Zhisheng Huang, Jiahao Chen, Cheng Lin, Chenyu Hu, Hanzhuo Huang, Zhengming Yu, Mengfei Li, Yuheng Liu, Zekai Gu, Zibo Zhao, Yuan Liu, Xin Li, and Wenping Wang. Recgen3d: Reconstruction-guided 3d generation in a shared canonical space. In SIGGRAPH Asia 2026 Conference Pa pers, 2026. 3, 4

[22] Wonjun Kang, Kevin Galim, and Hyung Il Koo. Eta inversion: Designing an optimal eta function for diffusion-based real image editing. In European Conference on Computer Vision, 2024. 3

[23] Sang-gil Lee, Heeseung Kim, Chaehun Shin, Xu Tan, Chang Liu, Qi Meng, Tao Qin, Wei Chen, Sungroh Yoon, and Tie Yan Liu. Priorgrad: Improving conditional denoising diffusion models with data-dependent adaptive prior. arXiv preprint arXiv:2106.06406, 2021. 3

[24] Baicheng Li, Dong Wu, Jun Li, Shunkai Zhou, Zecui Zeng, Lusong Li, and Hongbin Zha. Mv-sam3d: Adaptive multiview fusion for layout-aware 3d generation. arXiv preprint arXiv:2603.11633, 2026. 6, 7

[25] Dong-Yang Li, Wang Zhao, Yuxin Chen, Wenbo Hu, Meng Hao Guo, Fang-Lue Zhang, Ying Shan, and Shi-Min Hu. Pixal3d: Pixel-aligned 3d generation from images. In Pro ceedings of the Special Interest Group on Computer Graph ics and Interactive Techniques Conference Conference Papers, pages 1–12, 2026. 3, 4

[26] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214, 2023. 2

[27] Weiyu Li, Jiarui Liu, Hongyu Yan, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Craftsman3d: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024. 2, 3, 6, 7

[28] Zeming Li, Xiangyue Liu, Xiangyu Zhang, Ping Tan, and Heung-Yeung Shum. Noisear: Autoregressing initial noise prior for diffusion models. arXiv preprint arXiv:2506.01337, 2025. 3, 4

[29] Zhihao Li, Yufei Wang, Heliang Zheng, Yihao Luo, and Bihan Wen. Sparc3d: Sparse representation and construction for high-resolution 3d shapes modeling, 2025. 3

[30] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025. 2

[31] Siyou Lin, Zhou Xue, Hongwen Zhang, Liang An, Dongping Li, Shaohui Jiao, and Yebin Liu. Mix3r: Mixing feedforward reconstruction and generative 3d priors for joint multi-view aligned 3d reconstruction and pose estimation, 2026. 3, 6, 7

[32] Yi Liu, Lutao Chu, Guowei Chen, Zewu Wu, Zeyu Chen, Baohua Lai, and Yuying Hao. Paddleseg: A high-efficient development toolkit for image segmentation, 2021. 7

[33] Dominic Maggio, Hyungtae Lim, and Luca Carlone. Vggtslam: Dense rgb slam optimized on the sl (4) manifold. arXiv preprint arXiv:2505.12549, 2025. 2

[34] Carsten Moenning and Neil A Dodgson. Fast marching farthest point sampling. Technical report, University of Cambridge, Computer Laboratory, 2003. 4

[35] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6038–6047, 2023. 3

[36] Trong-Tung Nguyen, Quang Nguyen, Khoi Nguyen, Anh Tran, and Cuong Pham. Swiftedit: Lightning fast text-guided image editing via one-step diffusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 3

[37] Linfei Pan, Daniel Barath, Marc Pollefeys, and Johannes Lutz Schonberger. Global Structure-from-Motion¨ Revisited. In European Conference on Computer Vision (ECCV), 2024. 2

[38] Johannes Lutz Schonberger and Jan-Michael Frahm.¨ Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 4

[39] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3d. In ACM siggraph 2006 papers, pages 835–846. 2006. 2

[40] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation, 2024. 2, 6, 7

[41] Tencent Hunyuan3D Team. Hunyuan3d-omni: A unified framework for controllable generation of 3d assets, 2025. 2, 6, 7

[42] Tencent Hunyuan3D Team. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation, 2025. 3, 6, 7

[43] Tencent Hunyuan3D Team. Hunyuan3d 2.1: From images to high-fidelity 3d assets with production-ready pbr material, 2025. 2, 4, 5, 6, 7, 8

[44] Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhil Naik. End-to-end diffusion latent optimization improves classifier guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7280–7290, 2023. 3

[45] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22532–22541, 2023. 3

[46] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 2, 4, 6, 7, 3

[47] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing, 2025. 3, 4

[48] Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schonberger, Patrick Labatut, Piotr Bo-¨ janowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. Vggt-omega. arXiv preprint arXiv:2605.15195, 2026. 2

[49] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. NeurIPS, 2021. 2

[50] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 2

[51] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning. In International Conference on Learning Representations, pages 10481–10497, 2026. 6

[52] Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. Meshlrm: Large reconstruction model for high quality meshes, 2025. 2

[53] Haohan Weng, Zibo Zhao, Biwen Lei, Xianghui Yang, Jian Liu, Zeqiang Lai, Zhuo Chen, Yuhong Liu, Jie Jiang, Chunchao Guo, et al. Scaling mesh generation via compressive tokenization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11093– 11103, 2025. 6, 7

[54] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Liang Pan Jiawei Ren, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian,

Dahua Lin, and Ziwei Liu. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 6

[55] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024. 3, 6, 7

[56] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024. 2, 6, 7

[57] Katherine Xu, Lingzhi Zhang, and Jianbo Shi. Secret seeds in text-to-image diffusion models. In MINT: Foundation Model Interventions. 3

[58] Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, et al. Dmv3d: Denoising multi-view diffusion using 3d large reconstruction model. arXiv preprint arXiv:2311.09217, 2023. 2

[59] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. arXiv preprint arXiv:2501.13928, 2025. 2

[60] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions On Graphics (TOG), 42(4):1–16, 2023. 2, 3, 4, 5

[61] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, and Ping Tan. Rade-gs: Rasterizing depth in gaussian splatting. arXiv preprint arXiv:2406.01467, 2024. 2

[62] Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, and Ping Tan. Geometry-grounded gaussian splatting. arXiv preprint arXiv:2601.17835, 2026. 2

[63] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024. 2, 3

[64] Ziyue Zhang, Luxi Lin, Xiaolin Hu, Chao Chang, HuaiXi Wang, Yiyi Zhou, and Rongrong Ji. Deepinv: A novel selfsupervised learning approach for fast and accurate diffusion inversion. arXiv preprint arXiv:2601.01487, 2026. 3

[65] Ruowen Zhao, Junliang Ye, Zhengyi Wang, Guangce Liu, Yiwen Chen, Yikai Wang, and Jun Zhu. Deepmesh: Autoregressive artist-mesh creation with reinforcement learning. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10612–10623. IEEE, 2025. 6, 7

[66] Zikai Zhou, Shitong Shao, Lichen Bai, Shufei Zhang, Zhiqiang Xu, Bo Han, and Zeke Xie. Golden noise for diffusion models: A learning framework. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 17688–17697, 2025. 3, 4

[67] Junzhe Zhu, Peiye Zhuang, and Sanmi Koyejo. Hifa: High fidelity text-to-3d generation with advanced diffusion guid ance. arXiv preprint arXiv:2305.18766, 2023. 3

# ReconPlusGen: Injecting Reconstruction Prior into Multi-view 3D Generation through Noise Inversion and Modulation

Supplementary Material

In this supplement, we first provide more experimental results in Sec. 6, including analyzes of the mapping function used in noise inversion and modulation and additional comparisons with point-cloud-conditioned methods. We then provide further details on the implementation in Sect. 7, including the CA-VGGT orientation design and a more detailed mathematical explanation of noise modulation. Finally, we discuss the strengths, limitations, and potential extensions of our method in Sec. 8.

## 6. More Experimental Results

## 6.1. More visualization Results

We report more visualization results in Fig. 7. Our method consistently outperforms the baseline methods, particularly in preserving complex structures and maintaining accurate proportions among different object parts.

## 6.2. Expansion Order in Noise Inversion

In Sec. 3.1 of the main paper, we initialize the diffusion process with noise obtained through inversion, where the inversion velocity is approximated using a Taylor expansion. To visualize the effects of different expansion orders, we select several ground-truth meshes, perform noise inversion followed by denoising and decoding, and measure the errors introduced at each order. We color-code these distance errors as described in Sec. 7.3 and present the results in Fig. 7. In most cases, the error introduced by the second-order Taylor expansion is sufficiently small for our task. Note that the inversion and generation processes use the same image conditioning.

## 6.3. Ablation of Noise Mapping Function in Noise Modulation

As discussed in Sec.3.4 in main context, we employ a noise modulation scheme that allows the diffusion process to rectify prediction errors from CA-VGGT. We introduce a mapping function $f ( \cdot )$ that converts confidence scores into interpolation weights. Given a set of query points q, point cloud S and confidence maps C, we search 15 nearest neightbors in the predicted point clouds for each query point, retrieve their corresponding confidence value, and map the NN-averaged VGGT confidence c to a interpolation weight via a mapping function. We conduction ablations for several mapping functions, which is detailed in below.

Linear mapping. Each sample is first min–max normalized, then affinely rescaled to $[ c _ { \mathrm { m i n } } , c _ { \mathrm { m a x } } ]$ :

$$
\tilde { c } = \frac { c - c _ { \mathrm { d a t a , m i n } } } { c _ { \mathrm { d a t a , m a x } } - c _ { \mathrm { d a t a , m i n } } } ,\tag{7}
$$

$$
\begin{array} { r } { \alpha = \tilde { c } \left( c _ { \operatorname* { m a x } } - c _ { \operatorname* { m i n } } \right) + c _ { \operatorname* { m i n } } . } \end{array}\tag{8}
$$

Since we fully trust the reconstructed geometry in highconfidence regions, we always set $c _ { \operatorname* { m a x } } = 1$ . Thus, α spans $[ c _ { \mathrm { m i n } } , c _ { \mathrm { m a x } } ]$ for each sample, independent of the absolute scale of c.

Zero-anchored sigmoid. In this setting, we map the raw confidence value to [0, 1]:

$$
\begin{array} { c l c r } { \alpha = \left( 2 \sigma ( s \cdot c ) - 1 \right) _ { [ 0 , 1 ] } , } \\ { = \operatorname { t a n h } \left( s c / 2 \right) _ { c > 0 } . } \end{array}\tag{9}
$$

where $\sigma ( z ) ~ = ~ ( 1 + e ^ { - z } ) ^ { - 1 }$ and $s > 0$ is a scale. This function is smooth, monotonic, and bounded in [0, 1], while s controls how sharply the interpolation weight responds to changes in confidence.

Exponential.

$$
\alpha = 1 - \exp ( - s \cdot c ) .\tag{10}
$$

This function increases rapidly at low confidence values and gradually saturates toward one, with s controlling the rate of saturation.

We evaluate the different mapping functions under the same 4-view DoraBench dataset and report the results in Tab. 4. At every tested scale, replacing the original linear mapping with either the zero-anchored sigmoid or exponential mapping increases the Chamfer distance and decreases the F-score. The performance gap is smallest at $s = 0 . 1$ which produces the most gradual nonlinear mappings, and largest at $s = 1 0$ . We therefore retain the original linear mapping with $c _ { \mathrm { m i n } } = 0 . 5$

We further analyze the effect of $c _ { \mathrm { m i n } }$ on the mapping function and report the results in Tab. 5. As shown, $c _ { \operatorname* { m i n } } =$ 0.5 achieves the best performance. We therefore use $c _ { \operatorname* { m i n } } =$ 0.5 for all other experiments reported in the main paper.

## 7. More Implementation Details

## 7.1. Data Preparation

While rendering images for training, we uniformly sampled cameras on a sphere of radius 7.5 centered at the origin and independently applied positional jitter along each

Input Images

Error:

Hunyuan2.1

Mix3R

MV-SAM3D

low(0)

ReconViaGen

high(0.1)

Ours

![](images/fe0c4ae080f0bba77362577a4fa30fd3ceb2c29349db2fc1d105fdb0a4b7d36c.jpg)  
Figure 6. More comparison results of our method and other baselines. We draw distance error with color map for visualization.

![](images/ee7be671705bfaf525f6aa64ffac920cedc4d13786c82ba034f39eefba8fe9bc.jpg)  
Figure 7. Vision and numerical results of noise inversion. In most cases, noise inversion with expansion order as 2 is sufficient for geometry info injection.

Table 4. Ablation study of different mapping functions on DoraBench. We report Chamfer Distance (CD, ↓) and F-Score (↑, threshold=0.1). Best results are shown in bold.
<table><tr><td>Mapping</td><td>Scale s</td><td>CD↓</td><td>F-score ↑</td></tr><tr><td>Linear mapping</td><td>一</td><td>0.0424</td><td>0.9659</td></tr><tr><td>Sigmoid</td><td>0.1</td><td>0.0490</td><td>0.9533</td></tr><tr><td>Sigmoid</td><td>1</td><td>0.0607</td><td>0.9359</td></tr><tr><td>Sigmoid</td><td>10</td><td>0.0622</td><td>0.9324</td></tr><tr><td>Exponential</td><td>0.1</td><td>0.0512</td><td>0.9489</td></tr><tr><td>Exponential</td><td>1</td><td>0.0616</td><td>0.9334</td></tr><tr><td>Exponential</td><td>10</td><td>0.0623</td><td>0.9334</td></tr></table>

Table 5. Quantitative comparison on different number of $c _ { m i n }$ . We evaluate on Dorabench with 4 rendering images for each shape, and report Chamfer Distance (CD, ↓) and F-Score (↑, threshold=0.01). We set δ = 0.5 by default.
<table><tr><td></td><td></td><td colspan="2">0.22</td><td colspan="2">0.42</td><td colspan="2">0.52</td><td colspan="2">0.62</td><td colspan="2">0.82</td></tr><tr><td>CD↓</td><td>F-Score ↑</td><td>CD↓</td><td>F-Score ↑</td><td>CD↓</td><td>F-Score ↑</td><td>CD↓</td><td>F-Score ↑</td><td>CD↓</td><td>F-Score ↑</td><td>CD↓</td><td>F-Score ↑</td></tr><tr><td>0.056</td><td>0.940</td><td>0.052</td><td>0.947</td><td>0.048</td><td>0.957</td><td>0.042</td><td>0.966</td><td>0.046</td><td>0.963</td><td>0.047</td><td>0.960</td></tr></table>

![](images/8829c2b29bb122e799e7cc3734b13a07d17d4476171b27ad7891705707582814.jpg)  
(a)

![](images/7d8826cbd38dc0b160068eb7907bd4e18ca88ac5dc5f5dc83993d7980f2c0f66.jpg)  
(b)  
Figure 8. Camera Coordinate. (a) The original VGGT [46] represents cameras in a relative coordinate system defined with respect to a reference view. (b) In contrast, our method expresses camera poses directly in a shared, object-centric canonical coordinate system.

axis within [−0.2, 0.2]. For each RGB image, we also rendered the corresponding depth map. To simulate images captured by different real-world devices, we randomly varied the field of view (FOV) between 38<sup>◦</sup> and 63<sup>◦</sup>. We sampled 81,920 surface points from each shape for shape encoding.

## 7.2. Necessity and Design Details of CA-VGGT

A fundamental obstacle in integrating large feed-forward 3D foundation models into 3D object generation is the coordinate system mismatch between the two model families, as shown in Fig. 8. Feed-forward 3D foundation models, such as VGGT [46], typically designate the first view as the reference frame, while 3D object generation models are typically trained within a unified, object-canonical coordinate space. This mismatch introduces systematic transformation ambiguities (rotation, scale, and axis orientation). Existing methods either overlook this issue or rely on heuristic posthoc alignment [3], which cannot provide a principled and stable solution.

We present the details of our canonical space definition. Due to the ambiguity of the “front” direction for many objects and the lack of orientation annotated 3D datasets, we do not attempt to learn a model that predicts geometry in an unified object canonical space (where the object’s front faces the −Z axis, assuming objects are Y-up). Instead, we relax the definition of canonical space by allowing the front of the object to align with any of four possible axes: X, −X, Z, or −Z. We refer to this formulation as relative canonical space.

Specifically, during data preparation, we assume the object is placed in one of these four orientations. We then sample camera poses and render corresponding RGB and depth images accordingly. Note that the camera pose is defined with respect to a global world coordinate system, which can be aligned with the relative canonical space via a global rotation. During training, we randomly sample n images and determine the relative canonical space by collapsing the azimuth angle of the camera pose of the first image lie within [−45<sup>◦</sup>, 45<sup>◦</sup>].

## 7.3. Evaluation and Visualization

For the 4-view DoraBench benchmark, we use views with indices [9, 18, 19, 20]. To evaluate performance under varying numbers of input views, we begin with the 2-view configuration [9, 18] and progressively add views to increase object coverage and reduce unobserved regions. The full 24-view configuration is [9, 18, 19, 20, 22, 23, 24, 25, 30, 31, 34, 35, 36, 37, 38, 39, 0, 1, 2, 3, 4, 5, 6, 7].

When comparing our method with MAtCha Gaussian, we introduce a mask loss during the Gaussian splatting optimization stage with weight 0.1 to suppress background floaters, which would otherwise produce spurious triangles in the final mesh. For Mix3R, we evaluate the V2 implementation released on GitHub, as it supports more flexible camera configurations. To visualize the Chamfer distance, we color each vertex of the predicted shape according to its error. For each predicted vertex, the error is defined as the maximum of two values: (1) its minimum distance to the ground-truth shape and (2) the distance from a ground-truth vertex for which it is the nearest vertex on the predicted shape.

## 8. Discussion

In this paper, we propose a novel method that directly predicts shapes in canonical space and injects the predicted geometry through noise inversion and modulation. Our approach excels at preserving reconstruction fidelity by eliminating the stochasticity of diffusion models, while retaining the ability to complete unobserved regions. Despite these promising results, we identified several limitations. Specifically, we encode the predicted point clouds using an offthe-shelf Shape VAE, which is originally trained on clean, complete shapes. When the input prediction is sparse and noisy, the VAE tends to preserve these errors, leading to imperfect surfaces in the final output. Fine-tuning the Shape VAE on our VGGT predictions seems to be a clear direction for future improvement.