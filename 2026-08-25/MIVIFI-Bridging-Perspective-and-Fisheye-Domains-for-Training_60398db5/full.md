# MIVIFI: Bridging Perspective and Fisheye Domains for Training Multi-View Fisheye Image Generation Models

Matthias Neuwirth-Trapp<sup>1,2,∗,</sup> <sup>B</sup>, Begüm Altunbas<sup>2,4,∗</sup>, Jiayi Wang<sup>2</sup>, Yan Xia<sup>4</sup>, Maarten Bieshaar<sup>2</sup>, Xinyu Huang<sup>3</sup>, and Daniel Cremers<sup>4</sup>

Abstract— Achieving 360° coverage is critical for the visual perception systems of autonomous vehicles. Fisheye cameras offer a cost-effective solution by enabling full surround coverage with as few as two sensors. However, existing multi-view fisheye datasets are limited, and synthesizing rare corner cases typically requires computationally expensive 3D simulations, hindering the training. While generative models have achieved significant success in standard perspective imagery, their application to wide-angle distortion remains unexplored. In this work, we formally introduce the novel problem of multi-view fisheye image generation conditioned on volumetric semantic representations and present two distinct methods. We first propose SyntheOcc-FE, which adapts the SyntheOcc architecture to fisheye data. While effective, this method is constrained by the scarcity of fisheye datasets, which limits its generalization. To overcome these limitations, we propose our second method, MIVIFI (multi-view fisheye), which leverages crossdomain learning with Equirectangular Projections. By bridging the gap between dataset domains using KITTI-360 fisheye images alongside nuScenes multi-view standard images, our approach enables high-fidelity manipulation of scene content. This framework enables the structural modification of semantic occupancy inputs to introduce or eliminate specific actors and facilitates the rendering of diverse meteorological conditions and illumination scenarios absent in the limited fisheye datasets. Quantitative and qualitative experiments demonstrate that our methods achieve robust photorealistic multi-view fisheye image generation and highlight the specific advantages of our crossdomain strategy for handling data scarcity.

## I. INTRODUCTION

The robustness of perception systems in autonomous vehicles is fundamentally dependent on the quality and diversity of their training data. While surround-view systems are standard, fisheye cameras are increasingly adopted for their cost-effectiveness and expansive field of view [11]. This capability is critical for achieving comprehensive 360<sup>◦</sup> coverage and eliminating blind spots during complex maneuvers like parking and navigating tight urban spaces. However, there is a pronounced scarcity of large-scale, richly annotated public datasets compared with their perspectivecamera counterparts, such as nuScenes [1].

Recent advances in generative modeling, particularly diffusion-based frameworks such as Stable Diffusion [19], have shown promise for data augmentation and corner-case synthesis in autonomous driving [2]. Unlike text-to-image models, architectures conditioned on 3D geometric and semantic information, such as semantic occupancy grids, offer precise spatial control over the generated output [2], [12], [15]. This paradigm enables rigorous scene editing: by manipulating the underlying volumetric semantic representation, one can systematically add or remove dynamic agents, alter infrastructure, or modify the scene layout before rendering the final image. Yet, applying these capabilities to multi-view fisheye imagery remains an unexplored and critical research gap. Direct application of existing methods is nontrivial due to severe lens distortion. Furthermore, a significant domain gap exists between visually diverse perspective-view datasets (e.g., nuScenes [1]) and more homogeneous fisheye datasets (e.g., KITTI-360 [13]), complicating knowledge transfer.

![](images/aa6fe38a2f260413016bed2a7ed992c7cea3bfd5f70ed5a5469f11265c9c1ada.jpg)  
Fig. 1: Controllable multi-view fisheye image generation. We introduce the task of generating controllable multiview fisheye images and present the first methods for it, (1) SyntheOcc-FE, an adaptation of SyntheOcc [12] trained on fisheye datasets. To mitigate the limited availability of multi-view fisheye data and improve generalization, we further propose (2) MIVIFI, which builds on SyntheOcc-FE and uses equirectangular projections to enable training on both fisheye and standard-lens datasets (a). (b) Both methods are conditioned on semantic occupancy maps and text prompts and (c) train a diffusion model to synthesize multi-view fisheye images (d). The occupancy conditioning provides the image semantic layout, supporting various downstream applications (f).

![](images/7e9da1e45756fbd48139a28c3432a19229775922ccae01fde995c4beb5c033c4.jpg)  
Fig. 2: SyntheOcc-FE and MIVIFI overview. (a) Geometric conditioning: a 3D semantic occupancy map is projected into a fisheye-aware multi-plane image (MPI) by discretizing depth and stacking the per-plane semantic layers (for each target view), yielding the structural control signal. (b) Semantic enrichment: a vision-language model provides training captions, while arbitrary prompts are used at inference to steer appearance. The fisheye MPI and text prompt jointly condition a diffusion model to generate multi-view fisheye images (d). (c) Post-processing: an optional binary vignette mask mimics mechanical lens vignetting and suppresses boundary artifacts. MIVIFI uses the same MPI+text conditioning but maps both fisheye and perspective data to a shared equirectangular projection (ERP) canvas, enabling joint training with large-scale perspective datasets via stitched ERP views and masked supervision in unobserved regions.

This paper addresses the scarcity of annotated multi-view fisheye datasets by proposing, to the best of our knowledge, the first two generation-based approaches for controllable multi-view fisheye images synthesis conditioned on 3D semantic occupancy grids. We first introduce SyntheOcc-FE, which adapts the SyntheOcc architecture to generate two $1 8 0 ^ { \circ } \times 1 8 0 ^ { \circ }$ fisheye images (cf . Figure 1). While this approach establishes a strong baseline for geometrically consistent generation, it is constrained by the limited diversity of existing fisheye data (e.g., KITTI-360), which consists of daytime, clear-weather scenes. This restricts the model’s ability to synthesize critical long-tail scenarios, such as nighttime or rainy conditions, or to add new content.

To overcome these limitations and address the scarcity of diverse fisheye data, we propose a second novel method, MIVIFI, that leverages cross-domain training with fisheye and perspective multi-view images (cf., Figure 1). By utilizing a shared Equirectangular Projection (ERP) space [5], MIVIFI integrates fisheye and standard-perspective images into a unified representation. This approach not only enables the transfer of visual attributes, such as weather and lighting effects, from the perspective domain to the fisheye domain but also enables advanced scene editing. We can structurally modify the input semantic occupancy, such as by inserting specific traffic participants or removing occlusions, and render these altered geometries under diverse atmospheric conditions absent from the original fisheye dataset. MIVIFI learns two wide-view ERP images that can be converted into two fisheye images, thereby achieving multi-view coverage.

Our experiments demonstrate that both of our methods generate high-quality images, with SyntheOcc-FE focusing on geometric fidelity and MIVIFI on generation diversity and semantic manipulation. For the first time, we can controllably synthesize corner cases, such as nighttime and rainy weather, for multi-view fisheye cameras by transferring knowledge from datasets in which these conditions were never captured with such sensors. This provides a scalable solution to the critical data scarcity problem for fisheye sensors, enabling the creation of diverse, edited datasets needed to train and validate safer, more reliable autonomous systems.

## In summary, our main contributions are:

• We formulate the novel task of generating controllable multi-view fisheye images conditioned on 3D semantic occupancy grids.

• We introduce SyntheOcc-FE, a geometry-aware framework that adapts latent diffusion models to enforce strict fisheye lens characteristics and wide-angle consistency.

• We propose MIVIFI, a unified training strategy that uses ERP to fuse fisheye and perspective datasets, enabling cross-domain transfer of diverse environmental semantics and facilitating precise structural editing of scene content.

## II. RELATED WORK

## A. General Image Generation

Early image synthesis relied on Generative Adversarial Networks (GANs) [4], which often suffered from training instability and limited mode coverage. The advent of Denoising Diffusion Probabilistic Models (DDPMs) [7] and Stable Diffusion (SD) [19] revolutionized the field by enabling stable and diverse generation via iterative denoising in latent space. ControlNet [25] further enhanced this by injecting spatial conditions into the generation process to guide structure. While these foundational models enable high-fidelity synthesis, they inherently lack the geometric constraints required to model the nonlinear distortions of fisheye lenses without the explicit adaptation provided by our methods.

## B. Image Generation for Autonomous Driving

In the autonomous driving domain, generative models have shifted toward multi-view consistency to simulate full vehicle surroundings. Methods like BEVGen [20] and BEV-Control [24] utilize Bird’s Eye View layouts to control scene synthesis, while SyntheOcc [12], MagicDrive [2], and WoVoGen [15] employ 3D-aware conditioning to maintain geometric consistency across multiple cameras. These approaches establish strong baselines for multi-view generation but are designed exclusively for perspective cameras. They fail to account for the severe non-linear distortion and wide field of view characteristic of fisheye sensors, which we explicitly model.

## C. Fisheye Image Generation

Fisheye cameras are indispensable in modern autonomous vehicles for providing 360<sup>◦</sup> surround coverage and eliminating blind spots with minimal hardware. Despite their importance, generative research for this domain remains sparse. Curved Diffusion [21] introduced a lens-aware framework capable of simulating arbitrary optical geometries, but it does not focus on automotive applications. Similarly, [10] explored fisheye generation for edge-case detection but restricted their scope to static surveillance settings rather than dynamic driving environments. We address this gap by proposing the first framework specifically tailored for multiview fisheye generation in autonomous driving which ensures both geometric accuracy and semantic consistency across overlapping wide-angle views.

## III. METHODOLOGY

In this section, we introduce SyntheOcc-FE and MIVIFI, the two methods proposed for multi-view fisheye generation.

## A. First Method: SyntheOcc-FE

We propose SyntheOcc-FE, an adaptation of the SyntheOcc [12] framework tailored to the complex geometry of fisheye cameras. The pipeline consists of three stages: (1) geometric conditioning, (2) semantic enrichment, and (3) domain-aware fine-tuning.

First, we discretize the scene into Multi-Plane Images (MPI) with N layers, which serve as the primary spatial conditioning signal. Unlike standard approaches, we modify the MPI generation to explicitly account for wide-angle distortion. Second, to capture visual diversity and scene semantics, we initialize our Stable Diffusion backbone (ControlNet and U-Net) with weights pre-trained on nuScenes and augment the training data with rich, vision-language-modelgenerated text descriptions. Finally, we fine-tune the model on our fisheye dataset to align the generative priors with the target sensor characteristics. The overall training setup is shown in Figure 2.

Geometric Conditioning via Fisheye MPI. To ensure the MPI conditioning is geometrically consistent with the sensor, we replace standard perspective projection with the Mei fisheye model [16]. This allows us to accurately map semantic labels from the voxel grid to the distorted image plane $( c f .$ Figure 2 (a)). For a specific depth slice $d _ { i } .$ , we first undistort a pixel’s coordinates $( u , v )$ using the intrinsic matrix and distortion coefficients $( k _ { 1 } , k _ { 2 } , p _ { 1 } , p _ { 2 } )$ to obtain normalized coordinates $( x _ { n } , y _ { n } )$ . These are mapped onto the unit sphere following the Mei projection:

$$
\mathbf { r } = { \bigg [ } { \begin{array} { l } { x _ { n } \cdot c } \\ { y _ { n } \cdot c } \\ { c - \xi } \end{array} } { \bigg ] } \ ,\tag{1}
$$

where $\xi$ represents the mirror parameter and the scalar c is defined as:

$$
c = { \frac { \xi \pm { \sqrt { 1 + ( 1 - \xi ^ { 2 } ) ( x _ { n } ^ { 2 } + y _ { n } ^ { 2 } ) } } } { 1 + x _ { n } ^ { 2 } + y _ { n } ^ { 2 } } } .\tag{2}
$$

The resulting ray r is scaled by the target depth $d _ { i }$ and transformed into world coordinates to sample the correct semantic label. This process ensures that the generated views preserve the specific non-linear distortion of the fisheye lens. Figure 2 (a) shows an example of the final fisheye MPI, which is similar to a semantic segmentation but obtained from the 3D occupancy map.

Network Training and Post-Processing. With the geometric conditioning established, we leverage a Vision-Language Model (VLM) to annotate the fisheye dataset. These descriptions serve as text prompts that, combined with the MPI control signal, condition the diffusion model during fine-tuning. See Figure 2 (b) for a VLM-generated sample caption. This could be further combined with prompt tuning approaches [8], [17].

As a final step to enforce sensor realism, we apply a binary mask to the generated output, setting all pixels within the mechanical vignetting area to black to match the physical constraints of the lens assembly (cf. Figure 2 (c)). The general training pipeline is similar to SyntheOcc [12] and is schematically shown in Figure 2, in which the MPI of the occupancy map and the text prompt condition an imagegeneration model to obtain synthetic multi-view fisheye images.

## B. Second Method: MIVIFI

To extend SyntheOcc-FE and incorporate the diversity of standard multi-view datasets (e.g., nuScenes), we introduce MIVIFI. While fisheye datasets offer structural realism, the available datasets often lack the semantic variety found in large-scale perspective collections. Building on SyntheOcc-FE, MIVIFI bridges this gap by adopting Equirectangular Projection (ERP) as a unified representation. By projecting both fisheye and perspective data onto a shared spherical manifold, we create a consistent domain for joint training. This allows the model to learn high-fidelity fisheye structures while benefiting from the rich object diversity of perspective data, conditioned on the same VLM-based text prompts used in SyntheOcc-FE, and optionally on the same circle mask to match the physical constraints of the lens assembly.

![](images/4c681edbae2f34d99f161cc155b815b806bd7cb2cb5348350c3a79ac1c44f467.jpg)  
Fig. 3: Unified ERP Data Preparation. (a) Original perspective cameras and (b) their MPI. (c) Perspective cameras mapped to the ERP frame and (d) their corresponding MPI. (e) A fisheye image and (f) its MPI. (g) The fisheye image mapped to the ERP frame and (h) its corresponding MPI.

Unified ERP Representation. We standardize all inputs into a common $1 8 0 ^ { \circ } \times 1 8 0 ^ { \circ }$ Equirectangular Projection (ERP) layout. This projection maps the 3D spherical environment onto a flat 2D grid by treating longitude and latitude directly as horizontal and vertical image coordinates.

Wide-angle fisheye images map directly onto this expansive canvas, whereas standard perspective cameras with narrower fields of view yield sparse projections individually. To approximate wide-angle coverage, we stitch adjacent perspective cameras with overlapping sightlines into composite ERP patches. This strategy minimizes empty padding and ensures spatially comparable supervision across varying sensor types by masking any remaining unobserved regions. Furthermore, this masking approach allows the inclusion of single-view datasets for training, thereby significantly expanding the available data pool.

For example, Figure 3 shows the direct mapping of our wide-angle fisheye images onto the full ERP canvas, while stitching three adjacent perspective cameras from nuScenes to form a comparable wide-field representation.

Masked Supervision and Out-of-FoV Completion. Despite stitching, perspective ERP projections inevitably contain unobserved regions (padding). To prevent the model from overfitting to these artifacts, we apply a masked loss that restricts supervision to valid pixel regions. Conversely, fisheye images provide valid signals across the entire canvas.

The bottom row in Figure 3 shows on the left the stitched perspective images with large black background sections, for which the training signal will be removed. Crucially, while the RGB input for perspective data is masked, we project the corresponding occupancy MPIs across the full $1 8 0 ^ { \circ } \times 1 8 0 ^ { \circ }$ domain. This creates a training setup in which the model is supervised by complete structural information (MPI), even when visual information (RGB) is missing. This implicitly encourages the model to perform out-of-FoV completion, i.e., hallucinating consistent RGB content in empty regions by leveraging cross-domain knowledge transfer from fully realized fisheye examples.

![](images/88439047df7c186389881410080cc5b66376f6c92325cebe4eea8b2ea50a556f.jpg)

![](images/e40a16e068f5b0f5932cc18514e92f4b463e379666faec45710b6c6f9df7939f.jpg)  
Fig. 4: Qualitative Results of SyntheOcc-FE. (a, d) Input MPI, (b, e) generated fisheye images, and (c, f) the corresponding real multi-view fisheye images for reference.

## IV. EXPERIMENTS

## A. Datasets

Our experiments leverage KITTI-360 [13] for fisheye training and evaluation, and nuScenes [1] to broaden scene diversity via ERP-based joint training in MIVIFI. KITTI-360 extends the original KITTI [3] benchmark with suburban driving sequences recorded in Germany and provides leftand right-facing fisheye cameras, each offering a $1 8 0 ^ { \circ } \times 1 8 0 ^ { \circ }$ field-of-view. Dense accumulated LiDAR scans are converted into semantic occupancy voxel grids at 0.2 m resolution, which serve as our primary conditioning input. nuScenes comprises urban scenes from Boston and Singapore with six perspective cameras at approximately $7 0 ^ { \circ } \times 4 0 ^ { \circ }$ field-ofview; for this dataset, we adopt the dense occupancy voxel grids released by SurroundOcc [23] at 0.5 m resolution.

## B. Implementation Details

We build our model upon Stable Diffusion v2.1 [19] and represent occupancy as MPI of size $1 0 0 \times 1 0 0 \times 2 5 6$ . Training utilizes a base batch size of 4 with gradient accumulation over 8 steps for an effective batch size of 32. To ensure balanced contributions to MIVIFI, we employ a batch sampler that selects equal numbers of samples from KITTI-360 and nuScenes. We set the learning rate to $1 \times 1 0 ^ { - 6 }$ . For inference, we use the UniPC scheduler [27] with a classifierfree guidance scale of 7.0 and 20 denoising steps.

![](images/a37a794ffe2d7225be2f9057dbdd0d681192fb4735a4c3175211e072a18c330b.jpg)

![](images/484fffc4fd5ad29597730bd90c23193724d3174c8fa00a713d2a9f86dd819b85.jpg)  
Fig. 5: Qualitative Results of MIVIFI. Comparison of generated outputs for KITTI-360 (a)–(d) and nuScenes (e)– (i). Both datasets display (a, e) the ERP MPI, (b, f) the generated ERP, and (c, g) the ERP converted to fisheye. For comparison, KITTI-360 includes (d) the real fisheye image, while nuScenes includes (h) the converted perspective and (i) the real reference image. These results demonstrate visual consistency, high quality, and faithful input adherence, as well as highlighting the ability to convert perspective images into multi-view fisheye images covering a $3 6 0 ^ { \circ } \times 1 8 0 ^ { \circ }$ FoV.

## C. Metrics

We assess quality, semantics, and diversity using four standard metrics within the valid fisheye region:

• FID: The Fréchet Inception Distance [6] measures the distance between real and generated image distributions; lower values imply higher fidelity.

• SSIM: The Structural Similarity Index [22] quantifies perceptual similarity in terms of luminance, contrast, and structure.

• LPIPS: We use Learned Perceptual Image Patch Similarity [26] to evaluate diversity. We compute pairwise distances between multiple images generated from the same input using different random seeds, with higher scores indicating greater variation.

• mIoU: We use mean Intersection over Union (mIoU) to assess semantic accuracy by comparing the semantic segmentation of generated and corresponding real images. For reliability, we filter out the ego-vehicle and mechanical lens vignette, and we exclude areas without an occupancy map due to weak LiDAR coverage, as these regions provide insufficient conditioning signal.

## D. Qualitative Results

a) General Generation Quality: Figure 4 and 5 show the qualitative results of SyntheOcc-FE and MIVIFI, respectively. Both approaches generate high-quality images that respect the given conditioning. Because only the MPI with semantic and spatial information and the textual prompt with an abstract description were provided, it is to be expected that the generated images differ slightly from the ground truth. This is further desirable, as it allows the generation of diverse datasets with the same semantic layout. We excluded the lens mask for the results here to show that the model, in general, learns to set these areas to black. We further want to highlight the out-of-FoV completion in 5 (g). The generated fisheye images cover a larger area, whereas the nuScenes dataset contains only images of the central region. Compare to Figure 3 to see how many pixels are generated and do not have a real counterpart.

b) Adding New Objects: We evaluate the insertion of new objects by modifying the underlying occupancy map. Figure 6 (a–f) illustrates the addition of a car and a person using SyntheOcc-FE. The model successfully generates a car that remains consistent with the MPI. However, the person (highlighted in red) is absent from the generated output. This limitation stems from the KITTI-360 training data, which lacks pedestrian instances.

To address this issue, we utilize MIVIFI, which incorporates the nuScenes dataset during training. This extension enables the successful generation of pedestrians as demonstrated in Figure 6 (g–j). These experiments confirm that scenes can be effectively manipulated by adding or removing objects within the occupancy map.

c) Shifting the Time-of-Day: We control visual properties through text prompts by combining the time-agnostic MPI with textual descriptors such as “nighttime.” Figure 7 compares the performance of SyntheOcc-FE and MIVIFI when transforming a daytime scene. SyntheOcc-FE fails to synthesize night imagery due to limited training data in this modality. In contrast, MIVIFI leverages a broader training distribution from nuScenes to generate realistic nocturnal environments.

d) Modifiying the Weather: We further manipulate environmental attributes via text prompts by integrating the time-agnostic MPI with specific weather descriptors, such as “heavy rain.” Figure 8 illustrates the contrast between SyntheOcc-FE and MIVIFI when adapting a clear-weather scene. While SyntheOcc-FE struggles to generate plausible precipitation effects due to data sparsity in this domain, MIVIFI leverages a more comprehensive training distribution provided by nuScenes.

## E. Quantitative Results

We present quantitative results using the metrics described in Section IV-C. Table I compares SyntheOcc-FE and MIV-IFI. As SyntheOcc-FE is specialized for fisheye images, it achieves higher scores across all quality-based metrics. In contrast, MIVIFI emphasizes generalization and diversity, as evidenced by lower quality but higher LPIPS scores.

TABLE I: Image quality results for SyntheOcc-FE and MIVIFI. SyntheOcc-FE demonstrates superior generation quality, while MIVIFI achieves greater diversity. Best results are in bold.
<table><tr><td colspan="5"></td><td colspan="3">mIoU Classes ↑</td><td rowspan="2">LPIPS ↑</td></tr><tr><td>Method</td><td>FID ↓</td><td>SSIM ↑</td><td>mIoU ↑</td><td>Road ↑</td><td>Car ↑</td><td>Background ↑</td><td></td></tr><tr><td>SyntheOcc-FE (ours)</td><td>10.77</td><td>0.49</td><td>0.34</td><td>0.65</td><td>0.49</td><td>0.74</td><td></td><td>0.30</td></tr><tr><td>MIVIFI (ours)</td><td>17.05</td><td>0.41</td><td>0.27</td><td>0.56</td><td>0.35</td><td></td><td>0.64</td><td>0.36</td></tr></table>

![](images/f28e9bed4911d2a0dc05c962cb9c26e028ccb1861ab0db62c35edce4ccfdec59.jpg)

![](images/329807482e80a1869d4d9fcff4d9a90ca1d7f623f6e8f79564456af43b8665c0.jpg)

![](images/565f46fc9fd4eeaab0365e3b7a81ecab1a97cd34c11358dc39811451c200833f.jpg)  
Fig. 6: Comparison of Object Insertion with SyntheOcc-FE and MIVIFI. Left: Results using SyntheOcc-FE. (a, d) Modified MPIs for the left and right views, showing an inserted car (blue) and person (red). (b, e) The respective generated images, demonstrating successful car insertion (b) but failed person insertion (e). (c, f) The original multi-view reference images before modification. Right: Results using MIVIFI. (g) Original MPI and (h) the corresponding real image. (i) Modified MPI with an inserted person (red) and (j) the generated output. The expanded training dataset allows MIVIFI to successfully synthesize pedestrians.

Table III compares methods on a downstream 2D object detection task. Following the COCO protocol, we report mean average precision over multiple intersection-over-union thresholds, defined as mAP = mAP@[.5:.05:.95] [14]. We trained YOLOv11 [9] on 4,000 images from the real KITTI-360 dataset or generated samples. We test it on the KITTI-360 test dataset. While real data outperforms synthetic, synthetic results remain competitive. SyntheOcc-FE generally yields better results than MIVIFI. However, because KITTI-360 is scarce in pedestrians, MIVIFI demonstrates the benefit of diverse training data for this specific class. Overall, both generative approaches yield viable results but fall short of the real training data.

![](images/4bc57d7911a4c1588ea1b8e0d3f789536189c6ef433030d736df0726d636fbc7.jpg)

![](images/1a1e0e5a580041e15a145bdf872f228c8813c69980a45083ab44d3e46bd2b6bd.jpg)  
Fig. 7: Nighttime Synthesis Comparison. Transformation of daytime multi-view fisheye imagery via text-prompt adaptation, comparing SyntheOcc-FE (a)–(d) and MIVIFI (e)–(h). (a, b) and (e, f) show the generated multi-view fisheye images conditioned by including “nighttime” in the prompt, while (c, d) and (g, h) are the real daytime images for comparison.

## F. Ablations

We evaluate several architectural modifications to analyze their impact on performance (cf. Table II).

• Replacing VLM-generated prompts with generic prompts significantly degrades image quality, although Background mIoU remains high due to the broad nature of that category.

• Removing the nuScenes-pretrained weights in addition to using generic prompts further reduces performance. • Eliminating the lens mask increases the FID to 18.68.

• Conditioning image generation on a pseudo-depth map from Depth Any Camera [5], cropped at the top and bottom to exclude unreliable regions, slightly increases the FID despite providing additional geometric guidance alongside the MPI.

• Incorporating a cosine-similarity loss to improve overlap consistency also increases the FID, indicating that the optimization prioritizes loss minimization over overall image quality.

## V. DISCUSSION

a) Regarding Expandability: Unlike SyntheOcc-FE, which is constrained by the scarcity of fisheye datasets, MIVIFI can be trained on any perspective dataset, as demonstrated here on nuScenes. Extending this training to additional datasets promises further improvements in generalization and image quality. Furthermore, MIVIFI could be trained with single-view images. This flexibility would further enable extension to other sensors and simulations,

TABLE II: Ablation Study on SyntheOcc-FE. Removing VLM prompts and lens masks degrades overall performance. Conversely, introducing depth conditioning and multi-view cosine similarity yields no significant quantitative gains. SSIM and mIoU are overall more stable than FID. Best in bold.
<table><tr><td rowspan="2">Experiment</td><td colspan="6"></td></tr><tr><td>FID ↓</td><td>SSIM ↑</td><td>mIoU ↑</td><td>Road</td><td>Car</td><td>Backgr.</td></tr><tr><td>SyntheOcc-FE (ours)</td><td>10.77</td><td>0.49</td><td>0.34</td><td>0.65</td><td>0.49</td><td>0.74</td></tr><tr><td>SyntheOcc-FE (ours) w/o VLM prompts</td><td>21.25</td><td>0.48</td><td>0.33</td><td>0.55</td><td>0.45</td><td>0.74</td></tr><tr><td>SyntheOcc-FE (ours) w/o VLM prompts &amp; w/o nuScenes-pretrained weights</td><td>23.32</td><td>0.40</td><td>0.28</td><td>0.47</td><td>0.36</td><td>0.72</td></tr><tr><td>SyntheOcc-FE (ours) w/ 2D depth condition</td><td>11.59</td><td>0.48</td><td>0.34</td><td>0.65</td><td>0.48</td><td>0.74</td></tr><tr><td>SyntheOcc-FE (ours) w/ multi-view cosine similarity</td><td>13.91</td><td>0.49</td><td>0.34</td><td>0.64</td><td>0.51</td><td>0.74</td></tr></table>

![](images/b4f56b504fc66f58b064658a0316ed511a106d3ade9e358f4aef7bac007fec94.jpg)

![](images/a6889e7ec15e908ad3f69c1d169619f7e0dfa22081616890086693d5ce2fa5c8.jpg)  
Fig. 8: Rain Synthesis Comparison. Transformation of sunny multi-view fisheye imagery via text-prompt adaptation to rain, comparing SyntheOcc-FE (a)–(d) and MIVIFI (e)– (h). (a, b) and (e, f) show the generated multi-view fisheye images conditioned by including “rain” in the prompt, while (c, d) and (g, h) are the real sunny images for comparison. SyntheOcc-FE is unable to generate rain, while MIVIFI successfully renders the weather modification realistic.

TABLE III: Downstream 2D Object Detection. mAP on the KITTI-360 test set for models trained on real KITTI-360 versus generated training data. Best in bold.
<table><tr><td>Training data</td><td></td><td>Vehicle ↑ Person ↑ Bicycle ↑ Avg ↑</td><td></td></tr><tr><td>KITTI-360 dataset</td><td>27.7</td><td>10.7</td><td>0.1 12.9</td></tr><tr><td>SyntheOcc-FE</td><td>25.6</td><td>0.6</td><td>5.5 10.6</td></tr><tr><td>MIVIFI</td><td>20.7</td><td>4.0</td><td>1.1 8.6</td></tr></table>

for example, those given by the RICO benchmark [18].   
MIVIFI is extensible with fisheye and non-fisheye datasets.

b) Regarding Generalization vs. Specialization: The results from SyntheOcc-FE and MIVIFI indicate a general trade-off between generalization and specialization. SyntheOcc-FE yields high-quality results due to its specialization in KITTI-360. MIVIFI, on the other hand, is trained on more diverse data and demonstrates the benefits of this generalization in object, time, and weather edits, but lags slightly in visual quality. Joint training induces domain conflicts, as shown in Tables I and III. SyntheOcc-FE is more specialized, and MIVIFI is more general.

c) Regarding Perspective Images: In Figure 3, it can be seen that the three perspective images do not match perfectly, creating a faulty training signal, which hinders training. A possible approach would be to train on a single image and mask out more area, or to improve the alignment of the overlapping regions. The same issues would arise when training a model to convert perspective to fisheye images. The ERP approach enables perspective image training but has alignment shortcomings.

d) Regarding Sequences: With semantic MPI conditioning, new trajectories can be simulated from the occupancy map or existing sequences can be adapted. However, in the current setup, the frames would not be consistent. This consistency can be achieved by incorporating conditioning on previous frames. While SyntheOcc-FE and MIVIFI can be extended to include additional conditioning, this is beyond the scope of this work. Consistent sequence editing would require additional conditioning through previous frames.

e) Regarding Downstream Application: We demonstrated that synthetic images provide a useful training signal for 2D object detection. Due to the semantic occupancy map, this can be further extended to panoptic segmentation. Although the overall object detection mAP is low, the results on real and synthetic data are comparable. We suspect that downstream results could be improved by optimizing the training setup, enhancing generation models, or using more data (which is particularly feasible for our methods). The semantic conditioning enables SyntheOcc-FE and MIVIFI for object detection and panoptic segmentation.

## VI. CONCLUSION

In this work, we introduce the problem of multi-view fisheye image generation and present two approaches to address it: SyntheOcc-FE and MIVIFI. SyntheOcc-FE is trained solely on fisheye data and achieves slightly better visual quality. MIVIFI combines SyntheOcc-FE with Equirectangular Projection, enabling training on additional multi view perspective datasets such as nuScenes. We show the benefits of this general approach for scene editing, offering a practical method for extending limited fisheye datasets and generating rare edge cases. Future work will expand the training data, incorporate image conditioning for consistent sequence editing, and place greater emphasis on downstream tasks. Overall, our cross-domain strategy reduces the need for costly 3D simulation and helps bridge the gap between scarce fisheye data and the diverse conditions required for training robust 360° perception models.

## Appendix

## Appendix A. Single-View ERP

MIVIFI can train on single-view images with a full semantic occupancy map and does not require multi-view perspective datasets. Figure A.1 shows a single-view ERP example. This enables single-view training and Out-of-FoV completion.

## Appendix B. Adapting Sequenzes

Figure A.2 shows sequence adaptation with SyntheOcc-FE. It generates high-quality images and inserts the car correctly, but struggles with the person due to limited data. Without visual conditioning, the images are not consistent.

## Appendix C. Qualitative Object Detection Results

Figure A.3 shows qualitative results for SyntheOcc-FE and MIVIFI for the downstream object detection task. SyntheOcc-FE detects no pedestrians.

![](images/b266b4d3a6b9e8c3e02ab7d0cb68bf4a13f3ad144b6f9a6de7ffe40b57fc5715.jpg)  
Fig. A.1: Single-View ERP visualization. Top: perspective image and corresponding MPI. Bottom: single-view mapped to ERP with masked regions and full-screen MPI.

![](images/da782494ecc6d3fd897b824282d15cab8dfe03959c823b63370d8aa4a0c7412e.jpg)  
Fig. A.2: SyntheOcc-FE object insertion. Top: occupancy map with added car (blue) and person (red). Middle: generated multi-view fisheye images. Bottom: real scene. The car renders, the person is omitted due to limited training data, and results vary without adjacent-frame conditioning.

![](images/74fc9f0d521c651c5ef7cb1296255e80d913ca3bd102bab4d8bd3fbd11b6864d.jpg)  
Fig. A.3: Qualitative Object Detection Results. SyntheOcc-FE (left) struggles more with people than MIVIFI (right).

## References

[1] Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., Beijbom, O.: nuscenes: A multimodal dataset for autonomous driving. In: CVPR (2020)

[2] Gao, R., Chen, K., Xie, E., Hong, L., Li, Z., Yeung, D.Y., Xu, Q.: MagicDrive: Street view generation with diverse 3d geometry control. In: ICLR (2024)

[3] Geiger, A., Lenz, P., Stiller, C., Urtasun, R.: Vision meets robotics: The KITTI dataset. IJRR (2013)

[4] Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. NeurIPS (2014)

[5] Guo, Y., Garg, S., Miangoleh, S.M.H., Huang, X., Ren, L.: Depth any camera: Zero-shot metric depth estimation from any camera. In: CVPR (2025)

[6] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS (2017)

[7] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: NeurIPS (2020)

[8] Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N.: Visual prompt tuning. In: ECCV (2022)

[9] Jocher, G., Qiu, J.: Ultralytics yolo11 (2024)

[10] Kim, S., Go, K.: Edge-case Synthesis for Fisheye Object Detection: A Data-centric Perspective. arXiv preprint arXiv:2507.16254 (2025)

[11] Kumar, V.R., Eising, C., Witt, C., Yogamani, S.K.: Surround-view fisheye camera perception for automated driving: Overview, survey & challenges. IEEE Trans. Intell. Transp. Syst. 24(4), 3638–3659 (2023)

[12] Li, L., Qiu, W., Cai, Y., Yan, X., Lian, Q., Liu, B., Chen, Y.C.: Syntheocc: Synthesize geometric-controlled street view images through 3d semantic mpis. arXiv preprint arXiv:2410.00337 (2024)

[13] Liao, Y., Xie, J., Geiger, A.: KITTI-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d. PAMI (2022)

[14] Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common Objects in Context. In: Eccv (2014)

[15] Lu, J., Huang, Z., Zhang, J., Yang, Z., Zhang, L.: Wovogen: World volume-aware diffusion for controllable multi-camera driving scene generation. In: ECCV (2024)

[16] Mei, C., Rives, P.: Single view point omnidirectional camera calibration from planar grids. In: ICRA (2007)

[17] Neuwirth-Trapp, M., Bieshaar, M., Paudel, D.P., Van Gool, L.: Incremental object detection with prompt-based methods. In: ICCVW (2025)

[18] Neuwirth-Trapp, M., Bieshaar, M., Paudel, D.P., Van Gool, L.: Rico: Two realistic benchmarks and an in-depth analysis for incremental learning in object detection. In: ICCVW (2025)

[19] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: Highresolution image synthesis with latent diffusion models. In: CVPR (2022)

[20] Swerdlow, A., Xu, R., Zhou, B.: Street-view image generation from a bird’s-eye view layout. Robotics and Automation Letters (2024)

[21] Voynov, A., Hertz, A., Arar, M., Fruchter, S., Cohen-Or, D.: Curved diffusion: A generative model with optical geometry control. In: ECCV (2024)

[22] Wang, Z., Bovik, A., Sheikh, H., Simoncelli, E.: Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600–612 (2004). https://doi.org/10.1109/TIP.2003.819861

[23] Wei, Y., Zhao, L., Zheng, W., Zhu, Z., Zhou, J., Lu, J.: Surroundocc: Multi-camera 3d occupancy prediction for autonomous driving. arXiv preprint arXiv:2303.09551 (2023)

[24] Yang, K., Ma, E., Peng, J., Guo, Q., Lin, D., Yu, K.: Bevcontrol: Accurately controlling street-view elements with multi-perspective consistency via bev sketch layout. arXiv preprint arXiv:2308.01661 (2023)

[25] Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to textto-image diffusion models. In: ICCV (2023)

[26] Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: CVPR (2018)

[27] Zhao, W., Bai, L., Rao, Y., Zhou, J., Lu, J.: Unipc: A unified predictorcorrector framework for fast sampling of diffusion models. NeurIPS (2023)