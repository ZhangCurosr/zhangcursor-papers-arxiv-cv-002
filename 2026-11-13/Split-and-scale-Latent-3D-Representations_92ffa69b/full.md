# Split-and-scale Latent 3D Representations

Dorian Chan, Muhammed Kocabas, Xiaoming Zhao, Oncel Tuzel, Jen-Hao Rick Chang

Apple

Latent 3D representations have significantly expanded the capabilities of modern shape modeling, enabling compact encoding of 3D scenes and supporting powerful 3D generative models. Despite this progress, current representations struggle to capture the fine details of complex objects, fundamentally limited by the capacity of their latents. However, scaling latent capacity at training time is computationally expensive and often infeasible under memory limits, and naively increasing the number of latents at inference fails to make effective use of the added capacity. In this paper, we address both challenges with split-and-scale, a method that effectively expands latent capacity at inference while keeping training cost tractable. Our key observation is that by representing 3D scenes as continuous functions of 3D coordinates, a single pretrained tokenizer can be applied at inference to arbitrarily small sub-regions of a scene, allocating the same latent capacity to a smaller spatial extent and thus capturing far higher detail. We validate this insight by conducting extensive ablations to study the trade-offs between compute, memory, latent capacity, and reconstruction quality, demonstrating our inference-time scaling method achieves quality competitive with much larger models that exceed single-GPU memory limits. We further demonstrate that split-and-scale supports high-quality generative modeling, training an autoregressive 3D generator that performs competitively with state-of-the-art baselines.

one model, inference-time scaling  
![](images/cc180ccc368b5f4b53250b6022bed0e39e96e05d3c9d96f8fe2caf094c2d163d.jpg)  
Figure 1 Split-and-scale reuses the same tokenizer across spatial scales to efectively increase latent capacity at inference time. Recommend viewing the figure on monitor, zoomed in. Asset credit (Cherchye, 2016).

## 1 Introduction

Latent representations have become central to modern 3D shape modeling (Zhang et al., 2023; Chang et al., 2024; Xiang et al., 2025). By encoding geometry and appearance into a compact set of tokens, they support a wide range of tasks, from shape reconstruction to 3D generation (Zhao et al., 2023; Chang et al., 2026; Xiang et al., 2026; Zhao et al., 2025; Chen et al., 2025b; Li et al., 2025c; Lai et al., 2025; Jia et al., 2025; Chen et al., 2025a; Wu et al., 2025; Malik et al., 2026).

Capturing the fine details of real-world objects, however, demands far more representational capacity than current latent representations typically provide. For example, consider building a latent representation for a small town, with buildings and trees and signs (Figure 1). Clearly, very large encodings are likely necessary to accurately capture all the detail in such a scene. How can we scale the latent representations of today to handle such a scenario? Since capacity in these token-based architectures is governed by the number of latent tokens, the standard recipe is to scale up the token count. This recipe is far from free: computation grows quadratically and memory linearly with the number of tokens, and existing mitigations like curriculum learning (Wu et al., 2025) or sophisticated attention patterns (Chang et al., 2026) only partially help — GPU memory ultimately caps the trainable latent size $( e . g . , \sim 8 1 9 2$ tokens for LiTo on an 80GB H100). While certain architectures $( e . g .$ , Perceiver-style encoders (Zhang et al., 2023; Chang et al., 2026; Malik et al., 2026; Lai et al., 2025)) permit larger token counts at inference than at training, in practice the resulting quality gains saturate quickly and can even reverse when scaled too far (Figure 3).

In this paper, we propose split-and-scale (Figure 1), a technique that enables efective inference-time scaling of latent capacity. Our key observation is that a latent 3D representation parameterized by continuous coordinates can be applied at any spatial scale: the same encoder operates on a full scene or any sub-region, after appropriate normalization of the input coordinates. Critically, normalizing a sub-region of width $1 / n$ back to 1 lowers the content frequency seen by the encoder by a factor of n. The same network thus represents detail at n× finer spatial resolution within each sub-region, and produces proportionally more total tokens across the full scene — efectively increasing latent capacity with no change to the network or its weights. In practice, because 3D scenes are dominated by surfaces, only $O ( n ^ { 2 } )$ of the $n ^ { 3 }$ sub-regions are non-empty so the total token count grows with surface area rather than volume, keeping inference cost tractable even at fine subdivisions.

Beyond inference, split-and-scale also enables eficient training. Because the encoder operates on normalized sub-regions, it can be trained primarily at coarser subdivisions and still generalize to finer ones at inference, as we will demonstrate empirically. Furthermore, we encode and decode each sub-region independently while computing the reconstruction loss on the assembled full scene. Independent encoding lets us backpropagate gradients through only a small subset of sub-regions per iteration, sharply reducing compute and memory; the full-scene loss simultaneously teaches the network to handle inter-region boundaries. Together, these properties let split-and-scale train comfortably within a single 80GB GPU (e.g., A100 and H100), even at token counts that would otherwise be infeasible to train under standard architectures.

Through extensive reconstruction experiments, we show that split-and-scale consistently outperforms naive inference-time scaling and approaches the quality of much larger baselines that cannot be trained within a single 80GB GPU. Detailed ablations further analyze the design choices behind split-and-scale and characterize the trade-ofs between latent capacity, training compute, memory, runtime, and reconstruction quality. We additionally demonstrate the representation’s broader applicability by training an autoregressive 3D generative model on top of split-and-scale latents.

In summary, our main contributions are:

• We propose split-and-scale, a latent 3D representation that reuses a single tokenizer across spatial scales — enabling memory- and compute-eficient training while supporting much higher-detail reconstruction at inference.

• Through thorough ablation studies, we map the trade-ofs between latent capacity, compute, memory, and reconstruction quality, and show that split-and-scale consistently outperforms naive inference-time scaling of the underlying tokenizer.

• We show a potential use case of the representation by training an autoregressive 3D generative model on top of split-and-scale latents.

## 2 Related work

In recent years, a significant body of work has started to explore the landscape of 3D latent representations. For instance, in the context of geometry, a number of approaches explore the encoding of signed-distance functions or occupancy fields (Zhang et al., 2023, 2024; Li et al., 2025b; He et al., 2025; Zhao et al., 2025;

Meng et al., 2025; Hui et al., 2024), often over some form of sparse grid structure for eficiency (Wu et al., 2024, 2025; Ren et al., 2024; Li et al., 2025c; Lai et al., 2025). Other works attempt to model 3D surfaces as probability distributions, leveraging generative architectures to sample from them (Cai et al., 2020; Chang et al., 2024; Yang et al., 2019).

In recent work, latent representations for both geometry and appearance have seen more traction. A number of approaches have explored how to encode both meshes and their textures. For example, many geometryonly representations have been extended with multi-view texture generation pipelines to add material to generated assets (Zhang et al., 2024; Zhao et al., 2025; Li et al., 2025b; Yang et al., 2025); however, such approaches do not leverage any potential compression in terms of shared information between geometry and appearance. To tackle this aspect, recent work has explored more joint encodings by baking the geometry and material of an input mesh into an intermediate primitive before latent compression. 3DTopia-XL (Chen et al., 2025c) proposed a PrimX primitive that locally encodes both SDF and material parameters, on top of which they train a variational autoencoder (VAE) (Kingma and Welling, 2013) for further compression. TRELLIS-2 (Xiang et al., 2026) proposed a dual contouring-like O-Voxel representation that they then train a VAE upon. While these mesh-based approaches often produce high-quality, plausible assets, they typically fail to reconstruct the appearance of input objects even under ground truth lighting, as they rely on simplified material models that do not capture the complexities of not only real objects but also many synthetic assets. Furthermore, generative models trained upon these representations do not output the surrounding lighting. This makes such models impractical for tasks like digital tourism or digital memory preservation, where a user would need to guess the lighting that best reproduces their target scene.

Concurrently, another line of research has explored directly encoding multi-view RGBD images as a more accessible proxy of both geometry and appearance. TRELLIS (Xiang et al., 2025) proposed encoding voxelized DINOv2 features (Oquab et al., 2023) into a structured latent representation, where features are encoded over a sparse grid — the representation is then supervised via renderings of decoded 3D Gaussians. SAM 3D (Chen et al., 2025a) built a generation pipeline for a TRELLIS-style representation with a sophisticated post-training process in order to maximize generation quality, improve input image fidelity and handle occlusions. To improve visual realism, LiTo (Chang et al., 2026) proposed using the surface light field as the input encoder feature, and decoding to view-dependent Gaussians. Relative to TRELLIS, this allowed their method to produce much better highlights and reflections, and also enabled the training of a generative model with much stronger input image fidelity thanks to the continuous nature of the surface light field input. Velox (Malik et al., 2026) extended the LiTo architecture to 4D scenes, demonstrating strong performance in tracking and generation tasks. Broadly speaking, our work is mostly orthogonal to these directions for improving latent 3D representations, and can potentially be used to scale any of them in a computationally eficient manner. In this paper, we chose to build of LiTo (Chang et al., 2026) as a base thanks to its strong performance in reconstruction and generation tasks — we discuss more details of how we leverage it in the next section.

More related to our work, a few approaches have begun exploring how objects can be generated in a partbased fashion (Tang et al., 2025; Chen et al., 2025b; Lin et al., 2025; He et al., 2026), typically by using diferent sets of latent tokens for diferent components. However, these approaches do not consider scaling the overall resolution of the entire object like we do, and require part-based annotations that may limit available 3D training data. Perhaps most similar to our approach, a few world-building approaches have adapted object-level tokenizers and generators, e.g., TRELLIS (Xiang et al., 2025), by applying them at the block level and combining the results to compose entire scenes (Engstler et al., 2025; Zheng et al., 2025; Yoon et al., 2026; Li et al., 2026). While analogous in principle, unlike our work, they are focused on scene generation, and must handle seams between diferent blocks in ad hoc fashions (e.g., via RePaint (Lugmayr et al., 2022) or SDEdit (Meng et al., 2021)) because they reuse pre-existing object tokenizers. In contrast, our work focuses on scaling the quality of object representations by building a tiling-friendly tokenizer and generative model from ground up. Ultimately, we believe our pipelines and trained models may be useful for future such world-building approaches.

![](images/0abcdbea7c1ae5c29a31c33181c29da8bf201f13c99d989faf2730444c6554dc.jpg)  
Figure 2 Naive scaling vs. split-and-scale. (Left) Naive scaling increases number of latents by using more queries to the tokenizer. As the tokenizer is not trained with the higher number of queries, the quality improvement is limited. (Right) Split-and-scale encodes/decodes subregions of the scene with the same tokenizer/decoder and the designed number of queries, efectively increasing latent capacity. Asset credit (Studio, 2022).

## 3 Method

## 3.1 Preliminary: LiTo (Chang et al., 2026)

While split-and-scale can potentially be applied to various existing methods, we focus on LiTo (Chang et al., 2026), a recent 3D latent representation for geometry and (view-dependent) appearance, as it takes a point cloud with continuous coordinates as input. Given a set of points $\mathcal X = \{ ( { \bf x } _ { i } , { \bf f } _ { i } ) \} _ { i = 1 \dots n }$ , where $\mathbf { x } _ { i } \in \mathbb { R } ^ { 3 }$ is the 3D coordinate and $\mathbf { f } _ { i } ~ \in ~ \mathbb { R } ^ { c }$ is the associated feature containing color and view direction, a Perceiver encoder (Jaegle et al., 2022; Zhang et al., 2023) E takes X as keys and values and subsamples X as the initial queries. It then outputs the latent representation s. The latent representation s is composed of a set of tokens, $\{ s _ { i } \in \mathbb { R } ^ { d } \} _ { i = 1 \dots m }$ , and can be decoded by diferent decoders into point clouds and Gaussian splats. At a high level, we have:

$$
\{ ( \mathbf { x } _ { i } , f _ { i } ) \} _ { i } \xrightarrow { \varepsilon } \textbf { s } \xrightarrow { \mathcal { D } } \textbf { y } ,\tag{3.1}
$$

where y can be a point cloud, a mesh, or Gaussian splats. s is supervised via 1) novel view renderings from the decoded Gaussians and 2) points sampled from a flow-matching decoder (Chang et al., 2024), given a dataset of RGBD images rendered and points sampled from a collection of synthetic objects. Mesh and voxel occupancy decoders are later trained with a frozen encoder, similar to TRELLIS (Xiang et al., 2025).

To scale LiTo to represent more complex objects, one potential solution is to simply increase the number of latent tokens m (by increasing the number of queries subsampled from X). However, it is often infeasible during training, as the compute grows quadratically due to the use of attention and memory linearly (see Table 1). Thereby, existing works (Zhang et al., 2024; Li et al., 2025b) often increase number of latents during inference by sampling more queries from the input point cloud. However, because the tokenizer is not trained with additional number of queries, the improvement is limited and can negatively impact the quality if used aggressively, as shown in Figure 3.

## 3.2 Split-and-scale 3D Representations

Can we efectively increase the number of latents during inference while maintaining training eficiency? Our strategy is to keep encoder operating exactly as at training — with the same number of queries $m -$ and invoke the tokenizer multiple times, once per sub-region of the scene (see Figure 2). Concretely, we subdivide the scene (ranging from -1 to 1) into $n ^ { 3 }$ non-overlapping voxels of size $\textstyle { \frac { 2 } { n } }$ and encode each independently: $s _ { \mathrm { s p l i t - s c a l e } } [ i ] = \mathcal { E } ( \mathcal { V } _ { i } )$ , where $\nu _ { i }$ is the content of the ith voxel. To decode, we similarly run the decoder independently for each voxel, and then compute losses over the re-assembled full scene. For eficiency, we skip empty voxels, so the total number of latents grows directly with the number of occupied voxels — due to the surface-dominated nature of 3D scenes, only $\mathcal { O } ( n ^ { 2 } )$ of the $n ^ { 3 }$ voxels are occupied in practice. Output resolution $( e . g .$ , number of Gaussians produced by the decoder) also grows correspondingly with $n .$

![](images/eceae9d2c56c30615c30b3c6d924138b0d95d004c9c037f757f317a2ad5031ed.jpg)

![](images/4f01a5ecbbc1fc1755aa634527ddcc36d1f83e1b81457f51ec72593a1326af85.jpg)

![](images/54f32ac55e9f477794e8be93dae85351bab74819438c496725fa832bb1153dc3.jpg)  
Figure 3 Quantitative comparison versus naive token count scaling. We train a split-and-scale model that outputs 1024 latents per voxel. We compare with LiTo-1024, LiTo-8192, LiTo-32768 that use 1024, 8192, 32768 tokens during training, respectively. We also evaluate these models with naive inference-time scaling. Note that training LiTo-32768 requires larger memory than all other methods, and its training takes 2.7× more time. All models are trained to the same number of iterations. As shown, the performance of the proposed method (blue curve) increases consistently as we increase number of tokens during inference, whereas naive scaling quickly hits bottleneck.

Table 1 Training compute comparison versus naive token count scaling. Our split-and-scale recipe induces a similar computational and memory cost to an 8192 token version of LiTo (Chang et al., 2026), and can be comfortably trained on an 80GB GPU unlike a 32768 token LiTo. Runtime is computed over a single forward + backward pass. FLOPs are computed assuming 2× forward cost for backward.
<table><tr><td>Method</td><td>Peak memory (GB)</td><td>Avg runtime (s)</td><td>Avg GFLOPs</td><td>LPIPS (32768 inf. tokens)</td></tr><tr><td>LiTo (8192 tokens)</td><td>68.22</td><td>1.18</td><td>45.51</td><td>0.106</td></tr><tr><td>Split-and-scale (≈ 14k train. tokens)</td><td>73.51</td><td>1.32</td><td>49.71</td><td>0.091</td></tr><tr><td>LiTo (32768 tokens)</td><td>145.58</td><td>3.54</td><td>306.23</td><td>0.084</td></tr></table>

However, splitting alone is not enough: as reported in Table 4a (“no scaling”), LPIPS degrades sharply from 0.107 to 0.161 at $n = 4 .$ . The cause is a frequency mismatch. A voxel of width $\textstyle { \frac { 1 } { n } }$ packs the same amount of geometric and textural detail into a spatial extent n times smaller, so the encoder is fed content at substantially higher spatial frequencies than any it was exposed to during training. As n grows, the input distribution drifts out-of-domain and quality collapses.

We correct this mismatch via the space-frequency duality of the Fourier transform: dilating a signal in space by a factor of n compresses its spectrum by the same factor. We therefore shift and rescale each voxel back to the original scene extent $[ - 1 , { \bar { 1 } } ] ^ { 3 }$ with a normalization $\mathcal { M } _ { i }$ before encoding, and invert it at the decoder:

$$
s _ { \mathrm { s p l i t - s c a l e } } [ i ] = \mathcal { E } ( \mathcal { M } _ { i } ( \mathcal { V } _ { i } ) ) , \qquad \mathcal { D } _ { \mathrm { s p l i t - s c a l e } } \big ( s _ { \mathrm { s p l i t - s c a l e } } \big ) = \bigcup _ { i } \mathcal { M } _ { i } ^ { - 1 } \big ( \mathcal { D } ( s _ { \mathrm { s p l i t - s c a l e } } [ i ] ) \big ) .\tag{3.2}
$$

Note that the transformation $\mathcal { M } _ { i }$ only depends on the size and location of the voxel, not its content. Under this rescaling, every voxel presents the encoder with content whose frequency range matches what it saw at training, regardless of voxel size.

Interestingly, the same network captures n× finer detail inside each voxel while reusing its pretrained weights, efectively expanding total latent capacity at inference without any retraining. By choosing n, we directly control the level of detail.

## 3.2.1 Computational strengths of split-and-scale

Strictly speaking, a traditional train-time-scaling approach for n tokens $( i . e .$ , directly training a tokenizer to use an increased number of tokens) should outperform a split-and-scale representation at n tokens. Under split-and-scale, tokens are partitioned across subregions and restricted to intra-region attention, while tokens in a traditional approach can attend globally across the entire object allowing for more potential compression. However, when implemented appropriately, the proposed method has important computational strengths, as we will discuss below:

Coarse-scale training. As the encoder and decoder functions are fixed across all scales, they can be trained at coarser scales only, but still generalize to finer subdivisions at inference. The smaller number of latent tokens at coarser scales allows for much faster training and reduced memory usage. In practice, slight distributional diferences can be introduced due to subdividing the scene. First, at $n > 1$ , geometry can cross diferent voxels, and the latent encoder/decoder must therefore be able to handle inter-voxel boundaries. Second, if the available number of points is fixed for an object regardless of $n ,$ the density of X can change with diferent scales, $i . e . ,$ splitting an object into $n ^ { 3 }$ voxels also splits input points into $1 / n ^ { 3 }$ per voxel. Both can be easily mitigated by training with some iterations of $n \geq 2$ and training coarser levels with a random number of input points.

Voxel subset training. The encoder/decoder are applied independently to each subvoxel of the object. Thus, rather than training the entire object as would be required under standard token scaling, we can train just a subset of voxels, allowing for significantly reduced memory usage even when using many latent tokens. Note the forward pass still includes all active voxels (but without storing their activations), as we only have RGB renderings of full objects (not individual voxels).

Implementation and ablation. We build of LiTo to implement our encoder and decoder; see Section A.2 for details. Following this above regimen, training comfortably fits within the memory limits of an 80GB GPU (Table 1) — an important threshold given their relative availability $( e . g .$ , A100 and H100 GPUs) compared to GPUs with larger memory sizes. Simultaneously, compute time and FLOP usage remain comparable to a standard LiTo architecture. However, performance scales well even up to 131072 total latent tokens $( n \approx 8 )$ as visualized in Figure 3.

We ablate our design choices in downscaled experiments in Table 4a, where we train tokenizers without various components of our final recipe for 44k iterations. Voxel normalization M and training with both $n = 1$ and $n = 2$ can be seen to be crucial for performance, while training with $n = 4$ and random point counts yield smaller improvements.

## 3.3 Autoregressive generation

While split-and-scale provides a computationally-eficient avenue for training latent representations with a high number of latents, training a generative model for these larger latents is not straightforward. For instance, $n = 4$ and 1024 tokens per voxel translates to approximately 32768 total tokens on average in our experiments, naively scaling the LiTo difusion transformer to generate 32768 tokens induces a peak training memory usage of nearly 160 GBs, far outpacing the capacity of most commonly available GPUs. To tackle this challenge, we propose an autoregressive voxel-based formulation with flow matching-based sampling (Li et al., 2024) and cross-attention conditioning, as visualized in Figure 4. In particular, we train a flow-matching model (Lipman et al., 2022) based on a difusion transformer (Peebles and Xie, 2023) that models the following distribution: $P ( \mathbf { v } _ { \mathrm { t a r g e t } } \mid \mathbf { v } _ { \mathrm { c o n t e x t } } , O , I )$ , where $\mathbf { v } _ { \mathrm { t a r g e t } }$ denotes the latents of voxels to be generated, $\mathbf { v } _ { \mathrm { c o n t e x t } }$ the already-generated voxel latents, O the $n ^ { 3 }$ coarse occupancy (as empty voxels are not reconstructed in split-and-scale), and I the input image. We use self attention between noisy $\mathbf { v } _ { \mathrm { t a r g e t } }$ tokens with cross attention to $\mathbf { v } _ { \mathrm { c o n t e x t } }$ , mitigating the memory costs of larger latent generation. To apply other conditioning, we use cross attention with I and self attention with the fewer token O.

Implementation and ablation. We train this autoregressive model for split-and-scale latents from $n = 4$ with 1024 tokens per voxel, using randomly selected context and target voxels. To minimize the overall number of autoregressive steps, we sample 8 voxels at a time, resulting in 4 flow-matching samplings per object on average. We train this model for 450k iterations with a batch size of 256. Note that training this model can again be performed on H100 80GB GPUs — please see the appendix for more details on our network architecture, which mostly follows that of LiTo (Chang et al., 2026).

![](images/bf40422cb4f0330aab057d3ebc9fd131eeba2a2c9b6ea9c0fb8ab7d2607cbd4a.jpg)  
Figure 4 Visualization of our proposed generation procedure. To reduce memory requirements, we propose autoregressive voxel-based generation with flow matching-based sampling and cross-attention conditioning. Given an input image, we start by first generating coarse $4 ^ { 3 }$ occupancy. For the occupied voxels, we then autoregressively sample voxel latents conditioned on previously-generated latents, until all voxels are filled. Asset credit (Studio, 2022).

In Table 4b, to evaluate our proposed autoregressive method, we also implement a naive approach that directly generates n = 4 split-and-scale latents with a DiT, but with reduced hidden dimension such that the model can be trained on an 80GB GPU. Our autoregressive approach can be seen to comfortably outperform this baseline when both models have been trained for 135k iterations. For an equivalent amount of training time, our autoregressive model can also be seen to be competitive with a DiT trained to directly generate 32768 token LiTo latents, even though neither the representation nor the generative model fit within 80GB memory.

We also ablate potential inference voxel sampling orders in Table 4b. Generating voxels in random orders versus in raster order away from the camera seems to behave comparably — we use raster order for the results shown in this paper. To leverage this model in practice, we also train a small flow-matching model that generates the 4 × 4 × 4 occupancy of an object, conditioned on an input image. We ablate the impact of this additional generation step on overall performance in Table 4b.

## 4 Results

## 4.1 Reconstruction

Dataset. We build our split-and-scale representation of TRELLIS’s (Xiang et al., 2025) training subset of Objaverse-XL (Deitke et al., 2023), which we split into 10% validation, 10% test and 80% training sets following LiTo (Chang et al., 2026). For quantitative evaluation, we randomly select 200 objects from the 1000 test set objects with highest aesthetic score as computed by TRELLIS.<sup>1</sup> We follow a simplified LiTo pipeline and render 1036×1036 resolution images under random lighting. To get the input point cloud, we backproject 150 views rendered on a sphere and randomly select 16 million points. We supervise the Gaussian decoder with 100 images rendered from random views.

Results. For quantitative evaluation, we first compare our split-and-scale model with baseline LiTo models (Chang et al., 2026) trained for a variety of diferent latent counts, under both equal iteration (Figure 3) and equal compute time (Figure 9(a)) settings. In both cases, it can be seen that for an equivalent total latent count, split-and-scale is competitive with a LiTo model trained with that total latent count. Additionally, split-and-scale outperforms the comparable LiTo model when scaled to use more tokens, e.g., split-and-scale with 4 × 4 × 4 voxels (≈ 32768 tokens) performs better than a LiTo model trained to use 8192 tokens, even when that LiTo model is applied with more latents at inference. For completeness, we also include a 32768 token variant of LiTo (LiTo-32768), that notably cannot be trained on an H100 unlike the other methods shown. Nevertheless, split-and-scale still performs comparably. We also evaluate the geometry quality via Chamfer distance between ground truth point clouds and point clouds sampled from the flow-matching decoder of LiTo and split-and-scale. As shown in Figure 8, point clouds sampled from split-and-scale can be seen to significantly outperform those sampled from comparable LiTo models.

![](images/97a6b41414dbedf99540aed05d319336d1c11f750cbbd620255e656ae416e5a2.jpg)  
GT  
LiTo-8192  
(8192 inf.)  
LiTo-8192  
(32768 inf.)  
Split-and-scale  
(32768 inf.)  
Split-and-scale  
(131072 inf.)  
TRELLIS  
TRELLIS-2  
Figure 5 Qualitative reconstruction comparison. For comparison, we visualize a LiTo model trained with 8192 tokens (“LiTo-8192”), that can be trained under similar compute limits as our split-and-scale model. Note that the metrics for TRELLIS and TRELLIS-2 are quite poor due to color shift. Asset credits (okajima, 2019; MrsM, 2020; andrea.notarstefano, 2020).

In Table 2, we compare with other baselines more broadly (we include our LiTo implementation trained with 8192 tokens, i.e., “LiTo-8192”, for completeness). To compare with the geometry of mesh-based approaches, e.g. TRELLIS (Xiang et al., 2025) or TRELLIS-2 (Xiang et al., 2026), we also train a mesh decoder from our split-and-scale representation (network architecture discussed in appendix). Our approach can be seen to be broadly competitive across both appearance and geometry.

For qualitative evaluation, in Figure 5 and the supplemental website, we visualize the results of split-and-scale relative to LiTo-8192 as well as recent baselines (Xiang et al., 2025, 2026). As can be seen, split-and-scale performs well in regions of high detail compared to a computationally-similar LiTo model trained with 8192 tokens. Simultaneously, it lacks the color drift issues of TRELLIS (Xiang et al., 2025) and TRELLIS-2 (Xiang et al., 2026)<sup>2</sup>. We also show a comparison with the upper-bound LiTo-32768 in Figure 10.

Table 2 Quantitative reconstruction comparison with other baselines.
<table><tr><td>Method</td><td>Infer. latent size</td><td>LPIPS↓</td><td>Chamfer (mesh)↓</td><td>Chamfer (flow)↓</td></tr><tr><td>LiTo-8192</td><td> $8 1 9 2 \times 3 2$ </td><td>0.124</td><td></td><td> $2 . 8 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>LiTo-8192</td><td> $3 2 7 6 8 \times 3 2$ </td><td>0.106</td><td></td><td> $2 . 7 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>TRELLIS</td><td> $\approx 2 0 \mathrm { k } \times 1 1$ </td><td>0.230</td><td> $1 . 2 8 \times 1 0 ^ { - 2 }$ </td><td>=</td></tr><tr><td>TRELLIS-2</td><td> $\approx \mathrm { 1 0 k \times 3 2 }$ </td><td>0.184</td><td> $1 . 5 3 \times 1 0 ^ { - 2 }$ </td><td>1</td></tr><tr><td> $S p l i t - a n d - s c a l e$ </td><td> $8 1 9 2 \times 3 2$ </td><td>0.124</td><td> $1 . 7 0 \times 1 0 ^ { - 3 }$ </td><td> $2 . 4 6 \times 1 0 ^ { - 3 }$ </td></tr><tr><td> $S p l i t - a n d - s c a l e$ </td><td> $3 2 7 6 8 \times 3 2$ </td><td>0.091</td><td> $1 . 5 7 \times 1 0 ^ { - 3 }$ </td><td> $2 . 0 9 \times 1 0 ^ { - 3 }$ </td></tr></table>

## 4.2 Generation

Dataset. We use the same dataset as reconstruction, but we render additional conditioning views at random distances away from the object, with field-of-view set such that the object takes up the majority of the frame. We use two sets of cameras for evaluation. To capture the global structure of generated objects, we use “horizon” cameras sampled from a circle centered around the object, with a fixed field-of-view that captures the entire object silhouette. To measure the quality of finer details, we also use random poses that are randomly placed closer to the object than the horizon views. Our quantitative metrics are computed over the 2000 test set objects with highest aesthetic scores.

Results. Figure 6 shows example results from our proposed autoregressive generation. More results can be found in the supplemental website. In general, we observe that our proposed method generates output assets with finer details than a 8192-token LiTo model (“LiTo-8192”), thanks to the increased capacity of the latent space. We also compare to generations from TRELLIS (Xiang et al., 2025) and TRELLIS-2 (Xiang et al., $2 0 2 6 ) ^ { 3 }$ our method seems to produce much less color shift relative to the input images compared to both methods. However, as shown in Figure 7(b), we sometimes observe structural flaws in the output assets from our generative model, $e . g .$ , duplicated features or asymmetry. We believe such degradations are perhaps expected due to our autoregressive formulation, where later voxel generations cannot revise earlier ones.

For quantitative evaluation, Table 3 shows the performance of our proposed autoregressive voxel generation method compared to existing 3D generation approaches. Our model can be seen to outperform TRELLIS and TRELLIS-2 baselines, while being competitive with LiTo-8192. In particular, our method seems to be roughly on par with LiTo-8192 in terms of input view metrics, while it begins to outperform for novel views.

Table 3 Quantitative generation comparison. CFG scale is 3.
<table><tr><td></td><td colspan="2">Input view</td><td colspan="2">Novel view (horizon)</td><td colspan="2">Novel view (random)</td></tr><tr><td>Method</td><td>FID↓</td><td> $\mathbf { K I D _ { 1 0 ^ { - 3 } \downarrow } }$ </td><td>FID↓</td><td> $\mathbf { K I D _ { 1 0 ^ { - 3 } \downarrow } }$ </td><td>FID↓</td><td> $\mathbf { K I D _ { 1 0 ^ { - 3 } } } \downarrow$ </td></tr><tr><td>LiTo-8192</td><td>16.78</td><td>0.42</td><td>6.33</td><td>0.97</td><td>7.95</td><td>1.77</td></tr><tr><td>TRELLIS</td><td>38.09</td><td>3.94</td><td>10.32</td><td>2.17</td><td>15.35</td><td>5.02</td></tr><tr><td>TRELLIS-2</td><td>42.84</td><td>3.76</td><td>10.39</td><td>1.81</td><td>14.71</td><td>3.79</td></tr><tr><td>Split-and-scale</td><td>17.98</td><td>0.23</td><td>6.26</td><td>0.83</td><td>7.70</td><td>1.83</td></tr></table>

## 5 Limitations and future work

While split-and-scale is a promising approach for scalable latent representations, it is not without its limitations. First, because every subregion is encoded and decoded independently, subtle seam artifacts can

![](images/64a6d51dbf5b3b8409eb49b763dae1957bb0828b76770fab0f2f0314b577fccf.jpg)  
Input images

![](images/bac2f103271a1b2bc5eaa3b3c051810affa3e83d1c5f05c4b381a491d8bf155a.jpg)  
Split-and-scale  
LiTo-8192  
TRELLIS  
TRELLIS-2  
Split-and-scale  
LiTo-8192  
TRELLIS  
TRELLIS-2

Figure 6 Qualitative generation comparison. A generative model trained on split-and-scale can reproduce fine details. In contrast, competitor methods sufer from either poor detail or poor material estimation. Asset credits (wribeast, 2020; slasmir, 2020; bobrik00, 2020).

arise, particularly in areas with significant view dependence (Figure 7(a)). Visually, these artifacts appear as highlights that are subtly cut of at voxel boundaries. They can be mitigated in a compute-eficient way by training a decoder that conditions on neighboring latents. Second, the current implementation of split-andscale subdivides objects into voxels of equal size for simplicity, regardless of the content complexity. While our proposed adaptive latent allocation (see Section A.2) can help mitigate any information asymmetry across diferent voxels, this regular division fundamentally increases the complexity of the distribution the splitand-scale tokenizer must be able to encode. Intelligent divisions, such as K-d tree or part-based approaches, could improve performance, but may introduce extra complexity for generative modeling or limit available training data. Third, in our current split-and-scale implementation, the number of output Gaussians grows with the subdivision level (i.e., each decoder run produces a full-sized set of splats). For a large n, this can generate too many Gaussians to render eficiently in practice. A smarter implementation where the decoder can output a variable number of splats depending on n would mitigate this problem.

![](images/4858d24b53474810c623136bea92ab72f1c6365f7de0677f39856be91127f05a.jpg)  
Figure 7 Failure cases. (a) In tokenization, because each voxel is independently processed, subtle seams can appear in areas with significant view-dependent appearance. (b) Because our proposed generation model is autoregressive, sometimes it can experience the structural flaws common to such models. This manifests in the form of repeated geometry or poor symmetry. Asset credits (3Dji, 2025; Sohaibakr, 2019; Nikkip, 2021).

Our training recipe for autoregressive generation is also quite simplistic, and can be further improved. For example, a common challenge with autoregressive models is that later generations cannot revise earlier ones, occasionally leading to global structural flaws (Figure 7(b)). One scenario that occasionally appears is where the object looks reasonable from the input view, but occluded regions show slight asymmetries or floating geometry relative to the input. A more complex design that enables the autoregressive model to generate global structures early in the sampling sequence could help remedy such issues, e.g., following recent work in fractal (Li et al., 2025a) and discrete difusion models. Another common challenge with such models is the requisite teacher forcing during training. Adapting recent techniques from other domains that attempt to deal with this training-inference gap (Huang et al., 2025; Chen et al., 2024) could improve quality.

## Acknowledgments

We thank Byeongjoo Ahn, Raviteja Vemulapalli and Anagh Malik for their detailed feedback. All funding for this work came from Apple.

## Author contributions

All authors were involved in project planning and progress discussions.

Scope. Rick Chang and Oncel Tuzel set the overarching direction of the work.

Method. Rick Chang formulated the main split-and-scale methodology, and Dorian Chan formalized the computational benefits. Dorian Chan conceived the autoregressive generation architecture. Muhammed Kocabas, Xiaoming Zhao and Oncel Tuzel provided idea, design and architecture feedback.

Writing. Dorian Chan wrote the initial draft of the paper, which Rick Chang and Muhammed Kocabas later refined. Oncel Tuzel critiqued the manuscript.

Model training. Dorian Chan led model training for both the tokenizer and the generative model. Rick Chang and Xiaoming Zhao developed the original training framework.

Experiments. Dorian Chan developed the evaluation framework for all ablations. Muhammed Kocabas implemented the comparisons with TRELLIS and TRELLIS-2. Dorian Chan and Rick Chang created all qualitative figures. Dorian Chan created the website.

## References

3Dji. Cofee grinder, 2025. URL https://skfb.ly/pzpn7. Licensed under Creative Commons Attribution.

andrea.notarstefano. Tibetan teapot, 2020. URL https://skfb.ly/6WQ7s. Licensed under Creative Commons Attribution.

bobrik00. Windmill, 2020. URL https://skfb.ly/6SQLq. Licensed under Creative Commons Attribution.

Ruojin Cai, Guandao Yang, Hadar Averbuch-Elor, Zekun Hao, Serge Belongie, Noah Snavely, and Bharath Hariharan. Learning gradient fields for shape generation. In ECCV, pages 364–381. Springer, 2020.

Jen-Hao Chang, Xiaoming Zhao, Dorian Chan, and Oncel Tuzel. Lito: Surface light field tokenization. In ICLR, volume 2026, pages 100644–100671, 2026.

Jen-Hao Rick Chang, Yuyang Wang, Miguel Angel Bautista Martin, Jiatao Gu, Xiaoming Zhao, Josh Susskind, and Oncel Tuzel. 3d shape tokenization via latent flow matching. arXiv preprint arXiv:2412.15618, 2024.

Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Difusion forcing: Next-token prediction meets full-sequence difusion. NeurIPS, 37:24081–24125, 2024.

Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, et al. SAM 3d: 3dfy anything in images. arXiv preprint arXiv:2511.16624, 2025a.

Yiwen Chen, Zhihao Li, Yikai Wang, Hu Zhang, Qin Li, Chi Zhang, and Guosheng Lin. Ultra3d: Eficient and high-fidelity 3d generation with part attention. arXiv preprint arXiv:2507.17745, 2025b.

Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, et al. 3dtopia-xl: Scaling high-quality 3d asset generation via primitive difusion. In CVPR, pages 26576–26586, 2025c.

Sooi Cherchye. City scene tokyo, 2016. URL https://skfb.ly/JMC6. Licensed under Creative Commons Attribution.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. NeurIPS, 36: 35799–35813, 2023.

Paul Engstler, Aleksandar Shtedritski, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Syncity: Training-free generation of 3d worlds. In ICCV, pages 27585–27595. IEEE, 2025.

Xianglong He, Zi-Xin Zou, Chia-Hao Chen, Yuan-Chen Guo, Ding Liang, Chun Yuan, Wanli Ouyang, Yan-Pei Cao, and Yangguang Li. Sparseflex: High-resolution and arbitrary-topology 3d shape modeling. arXiv preprint arXiv:2503.21732, 2025.

Xufan He, Yushuang Wu, Xiaoyang Guo, Chongjie Ye, Jiaqing Zhou, Tianlei Hu, Xiaoguang Han, and Dong Du. Unipart: Part-level 3d generation with unified 3d geom-seg latents. In CVPR, pages 34227–34236, 2026.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion. arXiv preprint arXiv:2506.08009, 2025.

Ka-Hei Hui, Aditya Sanghi, Arianna Rampini, Kamal Rahimi Malekshan, Zhengzhe Liu, Hooman Shayani, and Chi Wing Fu. Make-a-shape: a ten-million-scale 3d shape model. In ICML, 2024.

Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, et al. Perceiver IO: a General Architecture for Structured Inputs & Outputs. In ICLR, 2022.

Tanghui Jia, Dongyu Yan, Dehao Hao, Yang Li, Kaiyi Zhang, Xianyi He, Lanjiong Li, Yuhan Wang, Jinnan Chen, Lutao Jiang, et al. UltraShape 1.0: High-fidelity 3d shape generation via scalable geometric refinement. arXiv preprint arXiv:2512.21185, 2025.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Zeqiang Lai, Yunfei Zhao, Zibo Zhao, Haolin Liu, Qingxiang Lin, Jingwei Huang, Chunchao Guo, and Xiangyu Yue. LATTICE: Democratize high-fidelity 3d generation at scale. arXiv preprint arXiv:2512.03052, 2025.

Sikuang Li, Chen Yang, Jiemin Fang, Taoran Yi, Jia Lu, Jiazhong Cen, Lingxi Xie, Wei Shen, and Qi Tian. Worldgrow: Generating infinite 3d world. In AAAI, volume 40, pages 6433–6441, 2026.

Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. NeurIPS, 37:56424–56445, 2024.

Tianhong Li, Qinyi Sun, Lijie Fan, and Kaiming He. Fractal generative models. arXiv preprint arXiv:2502.17437, 2025a.

Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models. IEEE TPAMI, 2025b.

Zhihao Li, Yufei Wang, Heliang Zheng, Yihao Luo, and Bihan Wen. Sparc3d: Sparse representation and construction for high-resolution 3d shapes modeling. arXiv preprint arXiv:2505.14521, 2025c.

Yuchen Lin, Chenguo Lin, Panwang Pan, Honglei Yan, Feng Yiqiang, Yadong Mu, and Katerina Fragkiadaki. Partcrafter: Structured 3d mesh generation via compositional latent difusion transformers. NeurIPS, 38:35387– 35415, 2025.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising difusion probabilistic models. In CVPR, pages 11451–11461. IEEE, 2022.

Anagh Malik, Dorian Chan, Xiaoming Zhao, David B Lindell, Oncel Tuzel, and Jen-Hao Rick Chang. Velox: Learning representations of 4d geometry and appearance. arXiv preprint arXiv:2605.04527, 2026.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic diferential equations. arXiv preprint arXiv:2108.01073, 2021.

Quan Meng, Lei Li, Matthias Nießner, and Angela Dai. LT3SD: Latent trees for 3d scene difusion. In CVPR, pages 650–660, 2025.

MrsM. Tazza e piattino, 2020. URL https://skfb.ly/6WqJz. Licensed under Creative Commons Attribution-NonCommercial.

Nikkip. Tropical pot plant, 2021. URL https://skfb.ly/o6rwy. Licensed under Creative Commons Attribution.

okajima. [chocolate cake], 2019. URL https://skfb.ly/6WODV. Licensed under Creative Commons Attribution.

Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

William Peebles and Saining Xie. Scalable difusion models with transformers. In ICCV, pages 4195–4205, 2023.

Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies. In CVPR, 2024.

Tianchang Shen, Jacob Munkberg, Jon Hasselgren, Kangxue Yin, Zian Wang, Wenzheng Chen, Zan Gojcic, Sanja Fidler, Nicholas Sharp, and Jun Gao. Flexible isosurface extraction for gradient-based mesh optimization. ACM TOG, 42(4):1–16, 2023.

slasmir. Procedural voxel tree, 2020. URL https://skfb.ly/6SCH9. Licensed under Creative Commons Attribution.

Sohaibakr. Snowman, 2019. URL https://skfb.ly/6G7HB. Licensed under Creative Commons Attribution.

Second Studio. Dry flower, 2022. URL https://skfb.ly/owzYE. Licensed under Creative Commons Attribution.

Jiaxiang Tang, Ruijie Lu, Max Li, Zekun Hao, Xuan Li, Fangyin Wei, Shuran Song, Gang Zeng, Ming-Yu Liu, and Tsung-Yi Lin. Eficient part-level 3d object generation via dual volume packing. NeurIPS, 38:27115–27137, 2025.

wribeast. Day 2/ sweet, 2020. URL https://skfb.ly/6SpIw. Licensed under Creative Commons Attribution.

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3D: Scalable Image-to-3D Generation via 3D Latent Difusion Transformer. In NeurIPS, 2024.

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Xun Cao, Philip Torr, et al. Direct3d-s2: Gigascale 3d generation made easy with spatial sparse attention. arXiv preprint arXiv:2505.17412, 2025.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In CVPR, pages 21469–21480, 2025.

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, and Jiaolong Yang. Native and compact structured latents for 3d generation. In CVPR, pages 14419–14429, 2026.

Guandao Yang, Xun Huang, Zekun Hao, Ming-Yu Liu, Serge Belongie, and Bharath Hariharan. PointFlow: 3D point cloud generation with continuous normalizing flows. In ICCV, 2019.

Jiayu Yang, Taizhang Shang, Weixuan Sun, Xibin Song, Ziang Chen, Senbo Wang, Shenzhou Chen, Weizhe Liu, Hongdong Li, and Pan Ji. Pandora3D: A Comprehensive Framework for High-Quality 3D Shape and Texture Generation. arXiv preprint arXiv:2502.14247, 2025.

Seungwoo Yoon, Jinmo Kim, and Jaesik Park. Extend3d: Town-scale 3d generation. arXiv preprint arXiv:2603.29387, 2026.

Biao Zhang, Jiapeng Tang, Matthias Nießner, and Peter Wonka. 3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Difusion Models. ACM TOG, 2023.

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets. ACM TOG, 2024.

Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, BIN FU, Tao Chen, Gang YU, and Shenghua Gao. Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation. In NeurIPS, 2023.

Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3D 2.0: Scaling Difusion Models for High Resolution Textured 3D Assets Generation. arXiv preprint arXiv:2501.12202, 2025.

Kaizhi Zheng, Ruijian Zhang, Jing Gu, Jie Yang, and Xin Eric Wang. Constructing a 3d town from a single image. arXiv preprint arXiv:2505.15765, 2025.

## A Appendix

## A.1 Additional experiments

Table 4 details our ablation studies for both our tokenizer architecture as well as our generative model, as already discussed in the main manuscript. In the reconstruction ablation, one interesting point is that $n = 4 \AA$ only training underperforms training with only n = 1 and n = 2. We believe the reasons to be twofold. First, coarse-level training may improve the quality of fine details; in particular, since we split the scene and scale each voxel back to the full range $( [ - 1 , 1 ] )$ and encode each voxel separately, each rescaled voxel of $n = 4$ will contain more smoothed geometry compared to n = 1 and $n = 2$ . For example, a voxel in $n = 1 \ \mathrm { m a y }$ contain a chair, but a voxel in $n = 4$ might only contain the seat back (which has lower geometry frequency). This means the training data in $n = 4$ is of lower frequency than $n = 1 { \mathrm { ~ o r ~ } } n = 2$ . Thus, training with $n = 1$ gives the highest content frequency, which is important for handling any downstream scenarios with extremely detailed objects. To handle boundaries between voxels, it is important to train with $n = 2$ and $n = 4 ,$ , as done in the full recipe. A second potential reason is compute-related implementation tradeofs: specifically, in order to fit training within 80GB GPU memory (e.g., A100 or H100 GPU) as well as increase speed, we detach all but a single voxel at every iteration when $n = 4$ (see Section A.2). In contrast, we can jointly train 8 voxels at a time when $n = 2$ , resulting in a stronger training signal. This diference in trainable voxels across diferent n can be attributed to the larger number of Gaussian activations that must be stored for backprop when more voxels are used, even when they are detached. This efect is validated in a small-scale experiment in Table 5.

Figure 8 and Figure 9 are extensions to Figure 3. Figure 9 shows the results under an equal compute setting, where LiTo-1024 has been trained to 200k iterations, LiTo-8192 to 100k iterations, LiTo-32768 to

Table 4 Ablation study. We ablate our proposed designs for reconstruction as well as generation. In (b), both LiTo-32768 variants cannot be trained on an 80GB GPU, for both the representation and generative model.
<table><tr><td rowspan="2">Recipe</td><td colspan="2"> $n = 4 ,$  ≈ 32768 tokens</td><td colspan="2"> $n = 8 ,$  ≈ 131072 tokens</td></tr><tr><td>LPIPS</td><td>Chamfer</td><td>LPIPS</td><td>Chamfer</td></tr><tr><td>Full recipe</td><td>0.107</td><td> $2 . 1 9 \times 1 0 ^ { - 3 }$ </td><td>0.081</td><td> $2 . 0 5 { \times } 1 0 ^ { - 3 }$  </td></tr><tr><td>no random point count</td><td>0.110</td><td> $2 . 2 3 \times 1 0 ^ { - 3 }$ </td><td>0.092</td><td> $2 . 0 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>no scaling</td><td>0.161</td><td> $2 . 9 2 \times 1 0 ^ { - 3 }$ </td><td>0.164</td><td> $2 . 8 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td colspan="5">Training scales:</td></tr><tr><td>Only n = 1 and n = 2 training</td><td>0.113</td><td>_  $2 . 2 1 \times 1 0 ^ { - 3 }$ </td><td>0.088</td><td> $2 . 0 6 \times 1 0 ^ { - 3 }$  </td></tr><tr><td>Only n = 1 training</td><td>0.145</td><td> $3 . 3 4 \times 1 0 ^ { - 3 }$ </td><td>0.133</td><td> $2 . 7 1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Only n = 4 training</td><td>0.135</td><td> $2 . 2 9 \times 1 0 ^ { - 3 }$ </td><td>0.101</td><td> $2 . 0 8 \times 1 0 ^ { - 3 }$ </td></tr></table>

(a) Reconstruction ablation (44k iterations)

<table><tr><td rowspan="2"></td><td colspan="2">Input view</td><td colspan="2">Novel view (horizon)</td><td colspan="2">Novel view (random)</td></tr><tr><td>FID↓</td><td>KID↓</td><td>FID↓</td><td>KID↓</td><td>FID↓</td><td>KID↓</td></tr><tr><td colspan="7">Architecture (135k iterations):</td></tr><tr><td>Autoregressive</td><td>21.56</td><td> $6 . 3 4 \times 1 0 ^ { - 4 }$ </td><td>8.63</td><td> $1 . 8 1 \times 1 0 ^ { - 3 }$ </td><td>11.67</td><td> $3 . 8 0 \times 1 0 ^ { - 3 }$  –</td></tr><tr><td>Direct with smaller DiT</td><td>27.67</td><td> $7 . 3 5 { \times } 1 0 ^ { - 4 }$ </td><td>21.89</td><td> $8 . 6 5 \times 1 0 ^ { - 3 }$ </td><td>32.63</td><td> $1 . 5 8 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>LiTo-32768 DiT, equal time</td><td>21.95</td><td> $7 . 3 5 \times 1 0 ^ { - 4 }$ </td><td>8.76</td><td> $1 . 6 2 \times 1 0 ^ { - 3 }$ </td><td>11.38</td><td> $3 . 4 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>LiTo-32768 DiT, equal iter</td><td>18.50</td><td> $2 . 3 0 \times 1 0 ^ { - 4 }$ </td><td>6.53</td><td> $8 . 2 9 \times 1 0 ^ { - 4 }$ </td><td>7.95</td><td> $1 . 7 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td colspan="7">Generation order:</td></tr><tr><td>Random</td><td>18.07</td><td> $2 . 7 0 \times 1 0 ^ { - 4 }$ </td><td>6.18</td><td> $8 . 2 6 \times 1 0 ^ { - 4 }$ </td><td>7.89</td><td> $1 . 9 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Raster</td><td>17.87</td><td> $2 . 4 5 { \times } 1 0 ^ { - 4 }$ </td><td>6.21</td><td> $8 . 2 1 \times 1 0 ^ { - 4 }$ </td><td>7.72</td><td> $1 . 8 1 \times 1 0 ^ { - 3 }$  一</td></tr><tr><td colspan="7">Occupancy impact:</td></tr><tr><td>GT occupancy</td><td>17.95</td><td> $2 . 4 8 \times 1 0 ^ { - 4 }$ </td><td>6.15</td><td> $7 . 7 8 \times 1 0 ^ { - 4 }$ </td><td>7.75</td><td> $1 . 8 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Generated occupancy</td><td>17.98</td><td> $2 . 2 6 \times 1 0 ^ { - 4 }$ </td><td>6.26</td><td> $8 . 2 9 \times 1 0 ^ { - 4 }$ </td><td>7.70</td><td> $1 . 8 3 \times 1 0 ^ { - 3 }$ </td></tr></table>

(b) Generation ablation

34k iterations and split-and-scale to the standard 90k iterations. Split-and-scale can be seen to be very strong. Figure 8 explores the scaling performance of split-and-scale versus naive token scaling in the context of geometry only — split-and-scale can be seen to significantly outperform the standard approach, in both equal compute and equal iteration settings. Both figures also show the impact of adaptive allocation, as will be discussed in Section A.2.

In Figure 10, we also show an additional qualitative comparison of split-and-scale with 32768 inference tokens, and the upper-bound LiTo-32768 which cannot be trained in an 80GB GPU (Table 1). Split-and-scale remains competitive in visual performance.

To validate the importance of our multiscale training recipe, we also test whether the pretrained TREL-LIS (Xiang et al., 2025) tokenizer can be used in a subdivision, tile-based fashion out of the box with $n = 2$ and $n = 4 .$ . Specifically, as the base TRELLIS uses a $6 4 ^ { 3 }$ grid, for $n = 2$ , we construct a $1 2 8 ^ { 3 }$ grid and reconstruct individual $6 4 ^ { 3 }$ subgrids using TRELLIS. While the level of fine detail can be seen to qualitatively improve, large blur artifacts appear along object and voxel boundaries that worsen with n. Introducing overlapping reconstructions across diferent voxels does not seem to mitigate these issues and in fact worsens performance — blurry tile seams actually cover larger portions of the object than if they were not overlapped. This shows that tiling-aware training, as we developed in this paper, is important for scaling latent sizes as we have proposed.

split-and-scale split-and-scale (adaptive)

![](images/aeb4bf71cf35fcfbaac661e130773ffee0397d04169991b92093a64f6f878839.jpg)

![](images/eb6379108aa7d728b69f73b85e01d1a67dae70ae05bedc9845316a64f6987b67.jpg)  
(a) Equal compute time

![](images/1d8be26195ac976984d1b6c3be7283a74b2c19976ab5bda24cfb545158b7565e.jpg)

![](images/7905b95429b9fe6334bbb55a08e14fba3741231449c7ddb941be41584fe3cd90.jpg)

![](images/577e11875947ed7fceb28eb05574a9846317b07d27997bac18eccc07f64bce64.jpg)  
(b) Equal iteration

![](images/d4d9e2161f5bd4fe811fc2130129221975c669aa9b76d690fbf9dc7c8fe4b1bf.jpg)  
Figure 8 Quantitative geometry comparison versus naive token count scaling. We compare the quality of sampled point clouds from split-and-scale versus comparable LiTo models (Chang et al., 2026).

## A.2 Split-and-scale implementation details

Latent representation recipe. Our work builds of the LiTo representation (Chang et al., 2026), with a Perceiver encoder, a Gaussian decoder, and a flow matching-based point-cloud decoder (Chang et al., 2024). We randomly select a grid size of $n = 1$ , 2 or 4 at every iteration; we train for a total of 90k iterations. When $n = 4 ~ ( \approx 3 2 { }$ active voxels on average), we detach all but one voxel. During training, we select anywhere between $2 ^ { 1 0 } – 2 ^ { 2 0 }$ total input points, and randomly select a latent size between 256-4096 tokens per voxel. For supervision, we decode Gaussians for every voxel, render them jointly, and compute PSNR and LPIPS losses against the ground truth. We also apply a flow-matching loss through the point-cloud decoder (Chang et al., 2024) for every voxel. During inference we use 1024 tokens per voxel, and a maximum of 1 million points per voxel.

One potential downside of naive split-and-scale is that a fixed m latent tokens are used per voxel, when some voxels may contain more details and content than others. To tackle this, we can also train the encoder to handle a variable number of latent tokens, and at inference select the number of tokens per voxel according to some metric, $e . g .$ , the number of input points per voxel. The improvement aforded by this adaptive allocation is shown in Figure 9. Unless otherwise noted, we use this variant for all reconstruction results shown in the paper. For generation, we do not use this adaptation to avoid any additional prerequisite generation outside coarse $4 \times 4 \times 4$ occupancy.

## A.2.1 Network architectures

Our work is built of LiTo (Chang et al., 2026), and primarily inherits the same encoder/decoder/generative architectures from their work. Please refer to their paper for detailed architecture diagrams. Here, we summarize how each one works, and any modifications we have made.

Encoder: A Perceiver encoder receives a surface light field as input, which it subsamples to use as queries. Each query cross attends to only the points that it is the closest query to — this configuration can be interpreted as a sort of 3D patchification. Instead of the voxel-based attention pattern leveraged by LiTo for computational eficiency, we instead use standard full attention, as our per-voxel latent sizes are small enough for it to not become a bottleneck.

Point-cloud decoder: A noisy input point is positionally encoded and then passed into a Transformer archi tecture, where it only cross attends to the corresponding voxel latent (there is no self attention). The model conditions on the flow-matching timestep via gating and layernorm parameters predicted by a small MLP.

Gaussian decoder: A Perceiver encoder receives $6 4 ^ { 3 }$ coarse occupancy as input, cross attending to the latent of the corresponding voxel. It decodes a feature for every coarse location, each of which is then processed by an MLP to produce 64 Gaussians. Note that this decoder runs once for every occupied voxel, i.e., the number of Gaussians scales with the number of occupied voxels and therefore latent capacity.

Voxel decoder: A Perceiver architecture is used to decode a sparse structure TRELLIS latent. A pretrained TRELLIS sparse-structure decoder is then used to convert this latent to a full occupancy $\mathrm { g r i d }$ . To train this voxel decoder, at every iteration we select a random voxel, compute a split-and-scale latent for it and then decode a single sparse structure latent — we do not predict sparse structure latents for all voxels as we found such a procedure to be too memory ineficient for 80GB GPU training. We use this voxel decoder to decode Gaussians from generated latents.

Mesh decoder: A Perceiver takes in coarse occupancy for queries (like the Gaussian decoder) and outputs low resolution shape features, which are then upsampled by sparse convolutional layers to Flexicube parameters (He et al., 2025; Shen et al., 2023). We follow the split-and-scale recipe described in the main paper to train this mesh decoder, i.e., we pick a random n, decode up to $n ^ { 3 }$ sets of Flexicube parameters, construct a mesh from their union, and compute losses on rendered depth and silhouettes. We do not compute losses on rendered normal, because we observe that the normals and depths of synthetic geometry do not often align $( e . g .$ ., bump or normal maps).

Generative model: Like $\operatorname { L i T o } ,$ our work leverages a standard Difusion Transformer architecture, but to sample in an autoregressive fashion. Unlike LiTo, we do not use a positional embedding for each noisy token, but instead a “voxel” embedding on which voxel each token should correspond to — we use a sinusoidal positional embedding of the voxel center, with 32 frequencies sampled from $2 ^ { 0 }$ to $2 ^ { 1 2 }$ . We use a hidden dimension of 1152, with 16 self attention and cross attention heads. Time conditioning is implemented via adaLN Zero (Peebles and Xie, 2023). We input in $\mathbf { v } _ { \mathrm { c o n t e x t } }$ and I via cross attention, where the image is preprocessed with DINOv2-large (Oquab et al., 2023) and a learnable patchification layer, following LiTo (Chang et al., 2026). I and $\mathbf { v } _ { \mathrm { c o n t e x t } }$ are given separate key-value and output projections, and their output cross attentions are summed. O is input via self attention with a single positionally-encoded learnable token per active voxel. To generate $O ,$ we use the same flow-matching architecture as above, with O represented as $4 ^ { 3 }$ binary tokens — we threshold with 0.5 to obtain the active voxels. We train this occupancy generator for 100k iterations.

## A.3 Baseline implementation Details

LiTo (Chang et al., 2026): We retrain LiTo variants with the same training data as split-and-scale. We primarily reuse the exact Perceiver-IO architectures used by LiTo for the encoder and decoder — we also use the same architectures for our split-and-scale encoder/decoder for fair comparison. When training variants with diferent latent counts, we also adjust the resolution of the occupancy grid that is input to the Gaussian decoder, in order to match the equivalent occupancy grid resolution used by split-and-scale for a particular latent count, i.e., 1024 tokens $\to 6 4 ^ { 3 }$ , 8192 tokens $\to 1 2 8 ^ { 3 }$ , 32768 tokens $\to 2 5 6 ^ { 3 }$ . We use ground truth occupancy as input to the Gaussian decoder for reconstruction results, and the output of a voxel decoder for generation results.

For the generative model, we use the same DiT architecture as LiTo (Chang et al., 2026), except without positional embedding to aid with set generation. We also use $1 0 3 6 \times 1 0 3 6$ image inputs, rather than the $5 1 8 \times 5 1 8$ used in the original paper for fairness. We train this model for a total of 240k iterations — note that our AR model, at 450k iterations, has only been trained for roughly half the total object data of this LiTo model, as it is trained to generate only a subset of a given sample at every iteration.

TRELLIS (Xiang et al., 2025): We use the publicly released TRELLIS pipelines for image-to-3D and the tokenizer. For the tokenizer, TRELLIS expects a sparse $6 4 ^ { 3 }$ occupancy grid plus per-view DINO features. We voxelize the ground-truth point cloud directly into TRELLIS’s $6 4 ^ { 3 }$ format. The DINO pool consumes 150 dense input views (i.e., the same views used to generate the point cloud). We evaluate the results using TRELLIS’ own Gaussian decoder. We render the output gaussians at our target cameras using TRELLIS’s own splat rasterizer for a fair comparison against split-and-scale renders.

For image-to-3D, TRELLIS produces results in its own canonical frame $( [ - 0 . 5 , 0 . 5 ] ^ { 3 } , \mathrm { y - u p } )$ , which does not align with our input-view coordinate system; we account for this with a fixed similarity transform (axis flip and ×2 scale) before rendering. This rigid alignment is exact for the tokenizer (which is conditioned on the GT geometry), but for the generative model the orientation of the asset relative to the camera is not constrained by the conditioning image. We therefore only report quantitative gen-numbers using set-level metrics (FID/KID) that are invariant to such a global rotation. For qualitative results, we manually realign each gen output.

TRELLIS-2 (Xiang et al., 2026): We use the publicly released checkpoints of TRELLIS-2. The tokenizer has the spatial downsample factor of 16 and 32 latent channels per active voxel. We run the tokenizer at $1 0 2 4 ^ { 3 }$ voxel resolution, which after the $f 1 6$ downsample yields a $\bar { 6 } 4 ^ { 3 }$ sparse latent grid populated only at voxels intersected by the surface.

For image-to-3D, we run TRELLIS-2 with the 1024<sup>3</sup> voxel resolution. Because TRELLIS-2 represents appearance through PBR materials, evaluating it against our Gaussian renderings requires re-rendering all TRELLIS-2 outputs under the same lighting as our GT. We therefore re-render every reconstructed/generated GLB through Blender Cycles, using the exact per-sample HDRI light that were used to render our GT views. In practice TRELLIS-2’s material-light disentanglement is not perfect: even with the GT lighting, the decoded base-color/metallic/roughness channels often produce appearance noticeably diferent from our difuse GT, which inflates TRELLIS-2’s pixel-space and FID/KID numbers in ways that are not entirely attributable to the tokenizer/generator itself.

As with TRELLIS, TRELLIS-2’s gen output lives in its own canonical frame; we apply a fixed rotation/scale alignment for the qualitative figures, and rely on rotation-invariant set metrics (FID/KID) for the quantitative gen comparison. All renders use 1036×1036 resolution to match split-and-scale’s GT.

Table 5 Few-iteration training scale ablation. We evaluate the impact of training at n = 4 without detaching any voxels. While this makes training infeasible on most standard 80GB GPUs, it provides the best performance.
<table><tr><td>Training recipe</td><td>Fits 80 GB?</td><td>n = 4 LPIPS</td><td>n = 4 PSNR</td><td>n = 8 LPIPS</td><td>n = 8 PSNR</td></tr><tr><td>Our recipe</td><td>yes</td><td>0.129</td><td>29.58</td><td>0.102</td><td>31.26</td></tr><tr><td>Only n = 1 training</td><td>yes</td><td>0.168</td><td>27.60</td><td>0.158</td><td>28.82</td></tr><tr><td>Only n = 1 and  $n = 2$  training</td><td>yes</td><td>0.131</td><td>29.61</td><td>0.106</td><td>31.37</td></tr><tr><td>Only n = 4 training, detach</td><td>yes</td><td>0.161</td><td>27.17</td><td>0.128</td><td>28.98</td></tr><tr><td>Only n = 4 training, no detach</td><td>no</td><td>0.111</td><td>29.72</td><td>0.080</td><td>31.69</td></tr></table>

Table 6 TRELLIS tiling experiment. We evaluate whether the vanilla TRELLIS (Xiang et al., 2025) tokenizer can be applied in a similar tile-based fashion as split-and-scale. Quantitatively, performance can be seen to drop with the amount of tiling, even with overlapping reconstructions across diferent voxels.
<table><tr><td>Method</td><td>LPIPS</td><td>PSNR</td></tr><tr><td>original TRELLIS (Xiang et al., 2025)</td><td>0.212</td><td>20.86</td></tr><tr><td rowspan="3">TRELLIS, n = 2 TRELLIS, n = 2 + overlapping TRELLIS, n = 4</td><td>0.254</td><td>19.63</td></tr><tr><td>0.284</td><td>18.69</td></tr><tr><td>0.295</td><td>17.86</td></tr></table>

![](images/655fc1ad198b81d373d43d4e7712ca3aae35ab23de4b4dbcceb32d42bfa3bbbe.jpg)

LiTo (8192 training tokens)  
![](images/ce5c643d7c1a8410176c3846506754d5524897cc4cf01645a71c3c8e8d7b6f59.jpg)  
(a) Equal compute time

LiTo (32768 training tokens, B200)  
![](images/05019b58d32b792ad19647aed3125f1ac815ca5ab8ef8a4d09bca2175735eb21.jpg)

![](images/f8188059ec2f79b0abf3920a150e4731faa73752d93c3ca9e904f37b963bbc6a.jpg)

![](images/22dee090bdaaaa8e3a086577c0953cd659d16395c8c925b699f6715ae276ebe3.jpg)  
(b) Equal iteration

![](images/49743ce6aed314be4fd43328942fecc236a21181175e491aad86977378bc606c.jpg)  
Figure 9 Quantitative comparison versus naive token count scaling. Following the recipe in Section 3.2, we train a split-and-scale model that outputs 1024 latents per voxel at inference. For comparison, we train 1024 and 8192 token LiTo models (Chang et al., 2026) that use a similar amount of memory/compute during training. In both (a) equal compute time and (b) equal iteration scenarios, split-and-scale can be seen to scale with token count better than the baseline models. Although it cannot be trained within the memory limits of an 80GB GPU, we include a 32768 token variant of LiTo (Chang et al., 2026) as a performance upper-bound; split-and-scale is still competitive.

GT  
![](images/c71ffc24b99821f3877967d9c76b3f924868a62e22e4424b1f1f69fd81bd37cf.jpg)  
Split-and-scale (32768 inf.)  
LiTo-32768 (32768 inf.)

![](images/28a4fcb3b95866bb5173b186da2ebf5b952d8919dc1adaa746ca3e08bb677285.jpg)  
GT

![](images/085638e18bf0ef92895b3a6ddadcadaca19069df4b9d1fd9a6a9c7d7c7e64f53.jpg)  
Split-and-scale (32768 inf.)

![](images/a0cd0f9d6ba7c14bd6f1d9f640ce4fbfee0fd4fe9c9fcb93f420daf561593b9b.jpg)  
LiTo-32768 (32768 inf.)

![](images/7816448d8d075335e1d03a5671534caedbfcedb538fd2881ad96b2eb3fa05433.jpg)  
GT

![](images/46aabd8c65dee538cb707f0c0ba4201bad148902d6bbf48115eda6c4ae4b3046.jpg)  
Split-and-scale (32768 inf.)

![](images/61663059f8cdd5bcc6979d7c426f03123f00ca155f876e20404043d156e3347b.jpg)  
LiTo-32768 (32768 inf.)  
Figure 10 Upper-bound qualitative reconstruction comparison. Building of Figure 5, although it cannot be trained within the memory limits of an 80GB GPU, we show a 32768 token variant of LiTo (“LiTo-32768”) as an upper-bound for performance; split-and-scale is still competitive. Asset credits (okajima, 2019; MrsM, 2020; andrea.notarstefano, 2020).