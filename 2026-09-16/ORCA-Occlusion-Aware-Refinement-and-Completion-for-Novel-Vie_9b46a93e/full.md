Input image

# ORCA: Occlusion-Aware Refinement and Completion for Novel View Synthesis

Weronika Jakubowska<sup>1</sup> Maciej Zi˛eba<sup>1</sup> Przemysław Spurek<sup>2,3</sup>

<sup>1</sup>Wrocław University of Science and Technology <sup>2</sup>Jagiellonian University <sup>3</sup>IDEAS Research Institute weronika.jakubowska@pwr.edu.pl

## Abstract

Novel-view synthesisfrom a single image is afundamentally ambiguous problem. As the camera moves away from the input viewpoint, previously hidden regions become visible, exposing missing geometry and holes in the reconstructed scene. Existing methods often rely on generative models to complete such regions. However, many of these artifacts are small gaps near depth boundaries and do not require generating new scene content.

In order to eliminate expensive process of generating image we introduce ORCA, an occlusion-aware method for reconstructing and completing explorable 3D scenes from a single image. ORCA first introduces 3D structure into a Gaussian-anchor representation using monocular depth while preserving the original camera-ray correspondence. During scene exploration, missing regions are handled based on their size and structure. Small disocclusions are repaired using RGB-D information already available in the reconstruction, while generative inpainting is reserved for larger regions that cannot be reliably recovered from the scene. New Gaussian anchors are added and optimized locally without modifying the existing representation. By reducing unnecessary reliance on generative inpainting, ORCA limits generation-induced hallucinations and better preserves the content and structure ofthe original scene.

On DIV2K, ORCA improves novel-view quality over VistaDream across all reported metrics, increasing MUSIQ from 61.60 to 68.71 and CLIP-IQA from 0.474 to 0.574. These results show that many novel-view artifacts can be repaired effectively by reusing information already present in the reconstructed scene.

## 1. Introduction

Reconstructing a 3D scene from a single image is difficult because the input provides only one view of the scene. Depth is ambiguous, and large parts of the scene may be hidden behind foreground objects. These limitations become visible as soon as the camera starts to move. A novel viewpoint can reveal incorrect geometry, gaps between surfaces, or regions that were never observed in the input image. Figure 1 illustrates how camera motion exposes disoccluded regions and how ORCA repairs them using either existing scene information or generative completion.

![](images/292631e95c75b9180f5fb52bd7777f7a09b341e8abec804128206d8c64a6bcfc.jpg)  
Figure 1. Camera motion reveals disoccluded regions that are not covered by the initial reconstruction. ORCA reuses existing scene information whenever possible and applies generative completion only to regions that cannot be reliably recovered from the reconstructed scene.

Recent methods often address this problem by generating additional views or completing missing regions with generative models [18, 20, 22]. This makes it possible to extend the scene beyond the original observation, but generated content is not guaranteed to agree with the geometry that has already been reconstructed. A generated image may look plausible from one viewpoint while introducing inconsistencies when the scene is viewed from another.

We found that generation is not necessary for every region exposed by camera motion. Many artifacts that appear in novel views are small gaps around object boundaries. In such cases, the missing appearance is often already present in the reconstruction. The problem is instead that the existing geometry does not provide enough coverage from the new viewpoint. Larger occluded regions are different: they may contain content that was never visible and therefore cannot be recovered from the reconstructed scene alone.

This distinction motivates our approach, named ORCA, an Occlusion-Aware Refinement and Completion method for reconstructing explorable 3D scenes from a single image. ORCA uses information already present in the reconstruction whenever possible and introduces generated content only when it is needed. Small disocclusions are repaired using nearby RGB-D information from the reconstructed scene. We select background samples according to depth so that the added geometry is placed behind foreground objects rather than extending their surfaces into the missing region. Larger regions that cannot be recovered in this way are completed using generative inpainting.

The initial scene is represented using neural Gaussian anchors and is reconstructed from the input image together with an outpainted extension. Since both images correspond to the same camera viewpoint, they provide additional appearance information but no multi-view depth constraints. We therefore use monocular depth to introduce 3D structure. Gaussian positions are moved along their original camera rays according to the estimated relative depth, preserving their correspondence with the input view. The deformed geometry is then kept fixed while the remaining representation is fine-tuned.

Once the initial scene has been reconstructed, we render it along a camera trajectory and identify regions that become uncovered. Missing regions are repaired one at a time. New Gaussian anchors are added only inside the selected region and optimized without changing the existing scene representation. After each repair, the trajectory is rendered again before the next region is selected. This allows later repairs to take into account geometry that has already been added.

Unlike approaches that rely on generation whenever a new viewpoint reveals a hole, ORCA first checks whether the missing region can be recovered from the current reconstruction. This makes it possible to repair many disocclusions without synthesizing new appearance and limits generative completion to regions where the scene does not provide enough information.

Our main contributions are:

• We present a method for reconstructing a coherent and explorable 3D scene from a single image. Monocular depth is used to introduce 3D structure into a neural Gaussiananchor representation while preserving the appearance observed in the input view.

• We introduce an occlusion-repair mechanism that treats small geometric gaps differently from regions containing genuinely unseen content. Small disocclusions are reconstructed using RGB-D information already present in the scene, while generative completion is used only when the missing region cannot be recovered reliably from the existing representation.

## 2. Related Work

## 2.1. Neural Scene Representations

Neural Radiance Fields (NeRF) model a scene as a continuous function of density and view-dependent appearance and render novel views through volumetric integration [12]. In contrast, 3D Gaussian Splatting (3DGS) represents the scene explicitly with anisotropic Gaussian primitives, enabling fast rendering while maintaining high visual quality [9].

More recent approaches combine explicit geometric structures with learned neural features. Affine-Equivariant Kernel Space Encoding uses Gaussian kernels as a spatial support for locally transformable neural features [26], while GaINeR combines trainable Gaussian primitives with a neural implicit image representation and demonstrates geometry-aware lifting and depth-guided editing [7]. Our method builds on IRIS [21], which represents the scene using Gaussian neural anchors and aggregates their features through ray-Gaussian interactions. We use this representation as the basis for the depth deformation and disocclusion repair stages described in Sec. 3.

## 2.2. Single-Image 3D Scene Reconstruction

Reconstructing a 3D scene from a single image is inherently ambiguous because the input provides neither multiview geometry nor observations of occluded regions. One common strategy is therefore to generate additional views and use them as supervision for reconstruction. Sync-Dreamer [11] generates multi-view consistent observations in an object-centric setting, while ZeroNVS [14] targets zero-shot novel-view synthesis of real scenes under larger viewpoint changes. CAT3D [5] follows a generate-thenreconstruct strategy, where a set of generated views is first produced and then used to recover a 3D representation.

Several recent methods make the generation process more explicitly aware of camera geometry. MVGenMaster [3] incorporates camera parameters and 3D priors into multi-view diffusion, Stable Virtual Camera [25] generates views conditioned on target camera trajectories, and 3D-Adapter [4] introduces geometric feedback during multiview generation. Director3D [10] explores a related setting in which scene content and camera trajectories are generated jointly.

A second line of work focuses more directly on constructing explorable 3D scenes. WonderWorld [22] combines image outpainting with geometric initialization and depth guidance to progressively extend the reconstructed scene. ExScene [6] first constructs a global scene representation from generated appearance and depth and then refines it using generative priors. One2Scene [20] similarly uses generated panoramic content to construct a geometrically consistent Gaussian scaffold that guides subsequent view generation. Other approaches explore related directions, including camera-controlled video generation in DimensionX [16], instance-aware depth-guided reconstruction in DepR [24], and direct diffusion-based generation of Gaussian representations in DiffusionGS [2].

VistaDream [18] is particularly related to our setting. It starts from a zoomed-out and inpainted image, estimates depth, and builds an initial 3D scaffold before progressively completing the scene with additional RGB-D observations. The resulting Gaussian representation is further refined using multi-view consistency sampling. RealmDreamer [15] also combines Gaussian scene representations with image inpainting and depth-based guidance, although its primary focus is text-driven 3D scene generation.

## 2.3. Generative Scene Completion

Novel viewpoints often reveal regions that were not visible in the original observation. Generating plausible content for such regions is only part of the problem: the newly synthesized appearance must also remain consistent with the existing geometry and with neighboring viewpoints.

Existing methods address this issue in different ways. VistaDream [18] repeatedly extends an RGB-D reconstruction and uses multi-view consistency sampling to refine the generated observations. ExScene [6] combines generative priors with an explicit 3D Gaussian representation, while One2Scene [20] conditions further view generation on an already reconstructed Gaussian scaffold. Related approaches such as 3D-Adapter [4] introduce explicit geometric feedback directly into the generation process.

In contrast, ORCA uses generation only when the current reconstruction does not provide sufficient information to recover the missing region. Small disocclusions are repaired directly from existing RGB-D scene information, while generative completion is reserved for larger unseen regions.

## 3. Method

Our pipeline consists of three stages, as shown in Figure 2. We first optimize an initial Gaussian representation using the input image and its outpainted extension. We then deform the Gaussian geometry according to estimated monocular depth and fine-tune the resulting representation. Finally, we reconstruct regions revealed by viewpoint changes, with particular focus on previously occluded areas.

## 3.1. Initial Scene Representation

Given an input image $I _ { \mathrm { o r i g i n a l } } .$ , we generate an outpainted version $I _ { \mathrm { o u t p a i n t } }$ . The planar Gaussian cloud is initialized from the outpainted image grid, while both the original and outpainted images are used as training observations. No depth information is used at this stage. The Gaussian means are initially placed on a planar grid corresponding to the image plane.

We represent the scene using neural anchors parameterized by 3D Gaussians, following the base architecture [21]. The representation is defined as:

$$
\pmb { \mathcal { P } } = \{ ( \pmb { N } ( \pmb { \mu } _ { i } , \pmb { \Sigma } _ { i } ) , \omega _ { i } , \pmb { f } _ { i } ) \} _ { i = 1 } ^ { N } .\tag{1}
$$

Here, $\pmb { \mu } _ { i }$ and $\Sigma _ { i }$ denote the mean and covariance of the $i -$ th Gaussian, ω<sub>i</sub> is a logit controlling its opacity, and $\pmb { f } _ { i }$ is its latent feature. During training, latent features are obtained from a learnable multi-resolution hash encoding queried at the Gaussian positions. During rendering, the Ray Intersection Selector identifies ray–Gaussian interactions. Features associated with neighboring intersected anchors are aggregated by the Ray-Coherent Aggregation module and decoded into density and view-dependent color, which are then integrated using volumetric rendering.

Both $I _ { \mathrm { o r i g i n a l } }$ and $I _ { \mathrm { o u t p a i n t } }$ are assigned the same camera pose. For the outpainted image, the camera intrinsics are adjusted to account for the scale and position of the original image within the extended canvas. Because both observations share the same camera center and orientation, they provide no multi-view parallax and therefore do not constrain scene depth.

The first stage yields a Gaussian representation consistent with both training images, but its depth remains underconstrained because both observations share the same viewpoint. We address this in the second stage using monocular depth.

## 3.2. Depth-Based Gaussian Deformation

To introduce depth structure, we estimate a depth map D from the outpainted image $I _ { \mathrm { o u t p a i n t } }$ using Depth Pro [1]. For each Gaussian mean, its x- and $y -$ coordinates are independently normalized with respect to the spatial extent of the current Gaussian cloud and mapped to normalized image coordinates. The depth map is then bilinearly sampled at the corresponding location.

![](images/ecf444a64647c35d2c05b45ccb8188f92fdef2c8ed351c09f84331b5817f5d5b.jpg)  
Figure 2. Starting from the original and outpainted images, we initialize and optimize a planar Gaussian scene representation. Monocular depth is then used to deform the learned Gaussian geometry, and the resulting representation is fine-tuned while keeping the deformed Gaussian positions fixed. During novel-view exploration, small disocclusions are repaired using RGB-D information already available in the scene, whereas larger unseen regions are completed with generative inpainting.

Rather than using the metric depth values directly, we convert them to inverse depth and normalize them into a relative nearness measure. This allows us to use the predicted relative scene structure while adapting the magnitude of the depth deformation to the scale of the Gaussian cloud. Let

$$
q _ { i } = \frac { 1 } { \operatorname* { m a x } ( d _ { i } , \epsilon ) } ,\tag{2}
$$

where $d _ { i }$ is the depth value bilinearly sampled for the i-th Gaussian and ϵ is a small positive constant preventing division by zero. Larger $q _ { i }$ corresponds to points closer to the camera. We then compute

$$
s _ { i } = \mathrm { c l i p } \left( \frac { q _ { i } - q _ { \mathrm { f a r } } } { q _ { \mathrm { n e a r } } - q _ { \mathrm { f a r } } } , 0 , 1 \right) ,\tag{3}
$$

where $q _ { \mathrm { f a r } }$ and $q _ { \mathrm { n e a r } }$ are robust lower and upper quantiles of the inverse-depth distribution. This step reduces the influence of extreme depth predictions and expresses the estimated geometry as a relative displacement within the scale of the Gaussian representation.

The normalized value $s _ { i }$ determines the target $z -$ coordinate of each Gaussian:

$$
\tilde { z } _ { i } = z _ { \mathrm { b a s e } } + \lambda L _ { x y } s _ { i } ,\tag{4}
$$

where $z _ { \mathrm { b a s e } }$ is the median z-coordinate of the current Gaussian cloud,

$$
L _ { x y } = \operatorname* { m a x } \left( x _ { \operatorname* { m a x } } - x _ { \operatorname* { m i n } } , y _ { \operatorname* { m a x } } - y _ { \operatorname* { m i n } } \right) ,\tag{5}
$$

is the maximum spatial extent of the cloud along the xand y-axes, and λ controls the magnitude of the deformation.

Changing only the z-coordinate would move a Gaussian away from its original viewing ray. We therefore displace each Gaussian along the ray passing through its current position. Let c denote the camera center, with $c _ { z }$ its z-coordinate, and let $\mu _ { i , z }$ denote the current z-coordinate of the i-th Gaussian mean. We compute:

$$
\gamma _ { i } = \frac { c _ { z } - \tilde { z } _ { i } } { c _ { z } - \mu _ { i , z } } ,\tag{6}
$$

and update the Gaussian mean as:

$$
\tilde { \pmb { \mu } } _ { i } = \mathbf { c } + \gamma _ { i } \left( \pmb { \mu } _ { i } - \mathbf { c } \right) .\tag{7}
$$

This construction places the Gaussian at the target $z -$ coordinate while preserving its direction from the camera center. The image-space correspondence established during the first stage is therefore retained while the Gaussian cloud is deformed according to the estimated scene geometry.

The Stage-1 checkpoint is then fine-tuned using the deformed Gaussian positions. We replace the original means with $\tilde { \pmb { \mu } } _ { i }$ and keep them fixed during this stage, while the remaining model parameters continue to be optimized. Finetuning uses the same RGB reconstruction objective as in the first stage, without introducing an additional depth loss. The depth estimate is used only to modify the Gaussian geometry before fine-tuning and is not used as a direct supervision signal.

## 3.3. Disocclusion Repair

After depth-guided fine-tuning, moving the camera can reveal regions that are not covered by the existing Gaussian cloud. These regions usually appear as thin gaps near foreground–background boundaries or as larger areas that were not visible in the input image. The third stage detects and repairs these disocclusions.

We render the reconstructed scene along a predefined camera trajectory. At each iteration, we select the view with the largest detected disocclusion and split its repair mask into connected components. Small or elongated components are treated as local geometric gaps, while large and compact components are considered for generative completion. Local components are processed first, since many of these gaps can be repaired using information already present in the reconstructed scene.

For each local component, we build an RGB-D target using the surrounding reconstructed background. We first extract a ring of valid pixels around the repair region and select donors from the farther part of the local depth distribution. Using the nearest depth at an occlusion boundary can incorrectly extend the foreground surface into the missing region. Selecting farther background samples reduces this effect and places the added geometry behind the foreground object. RGB and depth values for the repair region are taken from nearby valid donors. If not enough valid background samples are available, we use nearest-neighbor filling instead.

New Gaussian anchors are initialized from the resulting RGB-D target and added only in the missing region. The newly added Gaussians are then optimized locally, while the existing scene representation remains unchanged. The optimization matches the target RGB values, increases coverage inside the repaired region, and preserves the appearance around its boundary. We use wider, overlapping Gaussians for these local repairs to reduce thin gaps that can remain visible after a viewpoint change.

Some regions cannot be recovered reliably from the local background. If a remaining component is sufficiently large and compact, we inpaint it using Stable Diffusion. Inpainting is applied to a crop around the selected component rather than to the whole rendered image, and only pixels inside the missing region are replaced. We allow at most two generative inpainting operations per reconstructed scene. After this budget is exhausted, remaining regions are handled using the local geometry-based repair.

For generated regions, we estimate depth using Depth Pro. The predicted depth is aligned with the depth of the current reconstruction using a local affine transformation,

$$
d _ { \mathrm { a l i g n e d } } = a d _ { \mathrm { p r e d } } + b ,\tag{8}
$$

where a and b are estimated from valid background pixels around the repaired region. Outliers are removed during the fitting. Near the boundary of the repaired region, the aligned depth is blended with the reconstructed background depth. The completed RGB-D region is then converted into additional Gaussian anchors and optimized locally. If inpainting or depth estimation fails, we fall back to the local background-based repair. After each repair, we render the camera trajectory again and search for the next view with remaining disocclusions. The newly added Gaussians are therefore evaluated from the full trajectory before another region is selected. The procedure stops when no sufficiently large disocclusion remains or when the maximum number of repair iterations is reached.

## 4. Experiments

We evaluate ORCA from three complementary perspectives: the perceptual quality of rendered novel views, crossview geometric consistency, and qualitative behavior under camera motion. We compare against VistaDream on two datasets using matched camera trajectories for both methods.

## 4.1. Datasets

We evaluate our method on two sets: DIV2K [17] and a subset of scenes released with RealmDreamer [15]. DIV2K provides a larger and more diverse collection of natural images, while the RealmDreamer subset provides a smaller set of synthetic scenes. For DIV2K, we use the validation split of the 2× bicubic downsampling setting. It contains 100 images obtained by bicubic downsampling of the corresponding high-resolution images, with spatial resolutions ranging from 408 × 1020 to 1020 × 1020. The variety of scenes and image content allows us to evaluate the method on a broad set of natural inputs. VistaDream failed to produce a valid reconstruction for one sky-dominated image in the DIV2K validation set. We therefore exclude this scene from the comparison for both methods and report quantitative results on the remaining 99 images. We additionally evaluate on 11 scenes from RealmDreamer: bathroom, bear, bedroom3, bust, car, kitchen, lavender, living\_room, piano, steampunk, and victorian. All images have a resolution of 512 × 512 pixels. These scenes include both indoor environments and object-centered compositions.

## 4.2. Implementation Details

The initial Gaussian representation is trained for at most 30,000 iterations. Early stopping is based on a smoothed PSNR score evaluated every 1,000 iterations, with a minimum improvement of 0.03 dB and a patience of four evaluations.

After the initial training stage, Depth Pro is applied to the outpainted image and the resulting depth estimate is used to deform the Gaussian positions as described in Sec. 3. The depth-guided representation is then fine-tuned for at most 20,000 iterations. Early stopping is enabled after 10,000 iterations, using the same stopping parameters as in the first stage. During fine-tuning, the deformed Gaussian means remain fixed while the remaining model parameters are optimized.

For disocclusion repair, local geometry-based completion is used by default. A missing region is considered for generative completion only if it contains at least 9,000 pixels, occupies at least 45% of its bounding box, and the shorter side of the bounding box is at least 64 pixels. All other components are repaired using information from the surrounding reconstructed background.

For local repair, RGB-D donor pixels are searched within a ring around the missing region, with inner and outer radii of 3 and 28 pixels, respectively. We require at least 32 valid donor pixels and preferentially select samples from the farther part of the local depth distribution, using the 78-th depth percentile as the default threshold. Newly inserted Gaussians use a covariance scale multiplier of 1.7 and a minimum opacity of 0.9. At most 5,000 Gaussians are added in a single local repair, and only the newly inserted Gaussians are optimized, for up to 24 steps.

For generative completion, we use the Stable Diffusion 1.5 inpainting model [13]. Inpainting is performed on a crop around the selected missing region using 30 inference steps, a guidance scale of 7.0, and a maximum resolution of 512 pixels. At most two generative completion operations are allowed per reconstructed scene. For generated regions, Depth Pro is used to estimate depth, which is locally aligned to the depth of the current reconstruction before the corresponding Gaussian anchors are inserted.

All reported experiments were run on a single NVIDIA A40 GPU. We additionally verified that the full pipeline can be executed on an NVIDIA RTX 4060 GPU.

## 4.3. Quantitative Results

We compare our method with VistaDream [18] on both DIV2K and RealmDreamer datasets. For both methods, we evaluate rendered camera trajectories using the same set of image-quality metrics. To ensure a fair comparison, we render VistaDream [18] using the same camera trajectories as those used for our method. We report MUSIQ [8] and CLIP-IQA [19], together with the five LLaVA-

IQA criteria used in VistaDream [18] : Noise-Free (NF), Edge, Structure, Detail, and overall Quality. For each reconstructed scene, the metrics are averaged over 50 frames sampled from the rendered video. We then report the mean score over all scenes in each dataset. Table 1 summarizes the results on DIV2K and RealmDreamer. ORCA consistently outperforms VistaDream across all reported metrics on both datasets. On DIV2K, MUSIQ increases from 61.60 to 68.71, while CLIP-IQA improves from 0.474 to 0.574. The LLaVA-IQA scores show a similar trend, with overall Quality increasing from 0.407 to 0.630, together with substantial improvements in Edge and Structure. On Realm-Dreamer, ORCA improves MUSIQ from 68.66 to 72.85 and CLIP-IQA from 0.378 to 0.457, while overall Quality increases from 0.573 to 0.851. These results indicate that the proposed reconstruction and repair strategy improves both perceptual image quality and the preservation of scene structure during novel-view exploration.

Since RealmDreamer contains a small set of predefined scenes, we additionally report scene-level results in Table 2. This allows us to examine how the two methods perform across different scene types rather than relying only on the average score. The per-scene results show that the improvement is consistent across different scene types. ORCA achieves higher MUSIQ and CLIP-IQA scores than VistaDream on all 11 RealmDreamer scenes. The largest improvements are observed for several of the more challenging scenes, including bear and piano, where VistaDream exhibits substantial degradation under viewpoint changes. While individual LLaVA-IQA criteria occasionally remain unchanged or favor VistaDream, the overall trend strongly favors ORCA.

We additionally evaluate cross-view geometric consistency using TSED [23]. Unlike the image-quality metrics reported above, TSED measures whether correspondences across rendered views are consistent with the underlying camera geometry. The results are reported in Table 3. ORCA also achieves higher cross-view geometric consistency. On DIV2K, TSED increases from 0.8265 for VistaDream to 0.9980 for ORCA, while on RealmDreamer it increases from 0.9864 to 1.0000. The particularly large improvement on DIV2K indicates that the gains in perceptual quality are accompanied by substantially higher cross-view consistency rather than being limited to individual rendered frames. The improvements across both perceptual and geometric metrics suggest that ORCA improves not only individual frame quality but also the stability of the reconstructed scene across viewpoints.

## 4.4. Qualitative Results

Figure 3 presents representative novel views rendered along matched camera trajectories. The qualitative results are consistent with the quantitative evaluation. As the camera moves away from the input viewpoint, VistaDream can exhibit uncovered regions, local Gaussian-like artifacts, and substantial blurring in newly exposed areas. These artifacts are particularly visible near occlusion boundaries and in views requiring larger viewpoint changes.

Table 1. Quantitative comparison with VistaDream on DIV2K and RealmDreamer. Results are averaged over all evaluated scenes in each dataset. Higher values are better for all metrics.
<table><tr><td>Dataset</td><td>Method</td><td>MUSIQ↑</td><td>CLIP-IQA ↑</td><td>NF↑</td><td>Edge ↑</td><td>Structure ↑</td><td>Detail ↑</td><td>Quality ↑</td></tr><tr><td>DIV2K</td><td>VistaDream</td><td>61.6023</td><td>0.4735</td><td>0.5808</td><td>0.3663</td><td>0.4337</td><td>0.7206</td><td>0.4065</td></tr><tr><td rowspan="2">RealmDreamer</td><td>Ours VistaDream</td><td>68.7133</td><td>0.5735</td><td>0.7606</td><td>0.5980</td><td>0.6539</td><td>0.8962</td><td>0.6295</td></tr><tr><td>Ours</td><td>68.6576 72.8514</td><td>0.3775 0.4567</td><td>0.7982 0.9636</td><td>0.4291 0.7182</td><td>0.4818 0.6855</td><td>0.9327 1.0000</td><td>0.5727 0.8509</td></tr></table>

Table 2. Per-scene comparison with VistaDream on the 11 RealmDreamer scenes. Higher values are better for all metrics.
<table><tr><td>Scene</td><td>Method</td><td>MUSIQ↑</td><td>CLIP-IQA ↑</td><td>NF↑</td><td>Edge ↑</td><td>Structure ↑</td><td>Detail ↑</td><td>Quality ↑</td></tr><tr><td rowspan="2">bathroom</td><td>VistaDream</td><td>70.8680</td><td>0.4333</td><td>1.00</td><td>0.60</td><td>0.70</td><td>1.00</td><td>0.92</td></tr><tr><td>Ours</td><td>73.8245</td><td>0.4643</td><td>1.00</td><td>1.00</td><td>0.98</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="3">bear</td><td>VistaDream</td><td>61.5041</td><td>0.4851</td><td>0.38</td><td>0.22</td><td>0.00</td><td>0.52</td><td>0.22</td></tr><tr><td>Ours</td><td>71.0969</td><td>0.6452</td><td>0.96</td><td>0.80</td><td>0.00</td><td>1.00</td><td>0.68</td></tr><tr><td>VistaDream</td><td>72.8725</td><td>0.2940</td><td>0.96</td><td>0.70</td><td>0.90</td><td>1.00</td><td>0.88</td></tr><tr><td rowspan="2">bedroom3 bust</td><td>Ours</td><td>75.0659</td><td>0.3245</td><td>1.00</td><td>0.96</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>VistaDream</td><td>71.8854</td><td>0.3467</td><td>0.90</td><td>0.46</td><td>0.54</td><td>1.00</td><td>0.54</td></tr><tr><td rowspan="2">car</td><td>Ours</td><td>75.7703</td><td>0.4273</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>VistaDream</td><td>58.1490</td><td>0.4131</td><td>0.30</td><td>0.00</td><td>0.00</td><td>0.88</td><td>0.02</td></tr><tr><td rowspan="2">kitchen</td><td>Ours</td><td>64.8457</td><td>0.4805</td><td>0.64</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.00</td></tr><tr><td>VistaDream</td><td>64.4678</td><td>0.3558</td><td>1.00</td><td>0.80</td><td>1.00</td><td>1.00</td><td>0.74</td></tr><tr><td rowspan="2">lavender</td><td>Ours</td><td>69.6226</td><td>0.4644</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.96</td></tr><tr><td>VistaDream</td><td>66.6912</td><td>0.3659</td><td>1.00</td><td>0.72</td><td>0.52</td><td>1.00</td><td>0.42</td></tr><tr><td rowspan="2">living_room</td><td>Ours</td><td>70.2838</td><td>0.3958</td><td>1.00</td><td>0.78</td><td>0.70</td><td>1.00</td><td>0.78</td></tr><tr><td>VistaDream</td><td>71.8811</td><td>0.2763</td><td>0.96</td><td>0.58</td><td>0.80</td><td>0.90</td><td>0.74</td></tr><tr><td rowspan="2">piano</td><td>Ours</td><td>75.7188</td><td>0.3015</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>VistaDream</td><td>69.4828</td><td>0.3858</td><td>0.56</td><td>0.38</td><td>0.38</td><td>0.96</td><td>0.38</td></tr><tr><td rowspan="2"></td><td>Ours</td><td>75.1793</td><td>0.5491</td><td>1.00</td><td>0.98</td><td>1.00</td><td>1.00</td><td>0.94</td></tr><tr><td>VistaDream</td><td>75.8560</td><td>0.4399</td><td>1.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.96</td></tr><tr><td rowspan="2">steampunk victorian</td><td>Ours</td><td>75.9137</td><td>0.5457</td><td>1.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>1.00</td></tr><tr><td>VistaDream</td><td>71.5758</td><td>0.3568</td><td>0.72</td><td>0.26</td><td>0.46</td><td>1.00</td><td>0.48</td></tr><tr><td rowspan="2"></td><td>Ours</td><td>74.0436</td><td>0.4259</td><td>1.00</td><td>0.38</td><td>0.86</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>1.00</td><td>1.00</td></tr></table>

ORCA produces more complete and visually stable novel views. Small disocclusions are filled without unnecessarily changing the surrounding appearance, while larger missing regions are integrated into the reconstructed scene without introducing the same degree of visible degradation. Across the shown trajectories, scene structure and object boundaries remain more consistent as the viewpoint changes.

Table 3. Geometric consistency measured with TSED on DIV2K and RealmDreamer. Higher values indicate better cross-view geometric consistency.
<table><tr><td>Dataset</td><td>VistaDream</td><td>Ours</td></tr><tr><td>DIV2K</td><td>0.8265</td><td>0.9980</td></tr><tr><td>RealmDreamer</td><td>0.9864</td><td>1.0000</td></tr></table>

## 5. Conclusion

We presented ORCA, an occlusion-aware method for reconstructing explorable 3D scenes from a single image. Starting from a planar Gaussian representation, ORCA introduces scene depth using monocular depth estimates while preserving the original camera-ray correspondence. During novel-view exploration, the method treats small disocclusions differently from regions containing genuinely unseen content. Small gaps are repaired using RGB-D information already present in the reconstruction, while generative inpainting is used only when the missing region cannot be recovered reliably from the scene. Newly added geometry is optimized locally without modifying the existing representation.

![](images/eb8b7f06b4af36a6e1a25b292190e74d2984ac557486e029548f6a1233bf87c0.jpg)  
Figure 3. Qualitative comparison with VistaDream. Each pair of rows shows matched novel views rendered by VistaDream (top) and ORCA (bottom) along the same camera trajectory. VistaDream exhibits uncovered regions, Gaussian-like artifacts, and blurring in challenging viewpoints, whereas ORCA produces more complete and spatially coherent renderings while better preserving scene structure across views.

Experiments on DIV2K and RealmDreamer show that ORCA improves both perceptual novel-view quality and cross-view geometric consistency compared with VistaDream. The results support the main motivation of our approach: many artifacts revealed by camera motion can be repaired using information already available in the reconstructed scene, reducing the need to generate new content.

## References

[1] Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second, 2025. 3

[2] Yuanhao Cai, He Zhang, Kai Zhang, Yixun Liang, Mengwei Ren, Fujun Luan, Qing Liu, Soo Ye Kim, Jianming Zhang, Zhifei Zhang, Yuqian Zhou, Yulun Zhang, Xiaokang Yang, Zhe Lin, and Alan Yuille. Baking gaussian splatting into diffusion denoiser for fast and scalable single-stage imageto-3d generation and reconstruction, 2025. 3

[3] Chenjie Cao, Chaohui Yu, Shang Liu, Fan Wang, Xiangyang Xue, and Yanwei Fu. Mvgenmaster: Scaling multi-view generation from any image via 3d priors enhanced diffusion model, 2025. 3

[4] Hansheng Chen, Bokui Shen, Yulin Liu, Ruoxi Shi, Linqi Zhou, Connor Z. Lin, Jiayuan Gu, Hao Su, Gordon Wetzstein, and Leonidas Guibas. 3d-adapter: Geometryconsistent multi-view diffusion for high-quality 3d generation, 2025. 3

[5] Ruiqi Gao\*, Aleksander Holynski\*, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul P. Srinivasan, Jonathan T. Barron, and Ben Poole\*. Cat3d: Create anything in 3d with multi-view diffusion models. Advances in Neural Information Processing Systems, 2024. 2

[6] Tianyi Gong, Boyan Li, Yifei Zhong, and Fangxin Wang. Exscene: Free-view 3d scene reconstruction with gaussian splatting from a single image, 2025. 3

[7] Weronika Jakubowska, Mikołaj Zielinski, Rafał Tobiasz,´ Krzysztof Byrski, Maciej Zi˛eba, Dominik Belter, and Przemysław Spurek. Gainer: Geometry-aware implicit network representation, 2026. 2

[8] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer, 2021. 6

[9] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 2

[10] Xinyang Li, Zhangyu Lai, Linning Xu, Yansong Qu, Liujuan Cao, Shengchuan Zhang, Bo Dai, and Rongrong Ji. Director3d: Real-world camera trajectory and 3d scene generation from text, 2024. 3

[11] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image, 2024. 2

[12] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis, 2020. 2

[13] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 6

[14] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry La gun, Li Fei-Fei, Deqing Sun, and Jiajun Wu. ZeroNVS: Zero-shot 360-degree view synthesis from a single real im age. CVPR, 2024, 2023. 2

[15] Jaidev Shriram, Alex Trevithick, Lingjie Liu, and Ravi Ramamoorthi. Realmdreamer: Text-driven 3d scene generation with inpainting and depth diffusion, 2025. 3, 5

[16] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhu, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with decoupled video diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 13695–13706, 2025. 3

[17] Radu Timofte, Eirikur Agustsson, Luc Van Gool, Ming Hsuan Yang, and Lei Zhang. Ntire 2017 challenge on single image super-resolution: Methods and results. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2017. 5

[18] Haiping Wang, Yuan Liu, Ziwei Liu, Wenping Wang, Zhen Dong, and Bisheng Yang. Vistadream: Sampling multiview consistent images for single-view scene reconstruction, 2024. 2, 3, 6

[19] Jianyi Wang, Kelvin C. K. Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images, 2022. 6

[20] Pengfei Wang, Liyi Chen, Zhiyuan Ma, Yanjun Guo, Guowen Zhang, and Lei Zhang. One2scene: Geometric con sistent explorable 3d scene generation from a single image, 2026. 2, 3

[21] Grzegorz Wilczynski, Mikołaj Zieli´ nski, Krzysztof Byrski,´ Joanna Waczynska, Dominik Belter, and Przemysław´ Spurek. Iris: Intersection-aware ray-based implicit editable scenes, 2026. 2, 3

[22] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T. Freeman, and Jiajun Wu. Wonderworld: Interactive 3d scene generation from a single image, 2025. 2, 3

[23] Jason J. Yu, Fereshteh Forghani, Konstantinos G. Derpanis, and Marcus A. Brubaker. Long-term photometric consistent novel view synthesis with diffusion models, 2023. 6

[24] Qingcheng Zhao, Xiang Zhang, Haiyang Xu, Zeyuan Chen, Jianwen Xie, Yuan Gao, and Zhuowen Tu. Depr: Depth guided single-view scene reconstruction with instance-level diffusion, 2025. 3

[25] Jensen Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models, 2025. 3

[26] Mikołaj Zielinski, Krzysztof Byrski, Tomasz Szczepanik,´ Dominik Belter, and Przemysław Spurek. Affine-equivariant kernel space encoding for nerf editing, 2026. 2