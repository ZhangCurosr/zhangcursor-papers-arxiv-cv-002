# Plant Orbiter: 3D Plant Reconstruction and Generation with Video Difusion Orbit Cameras and 3D Gaussian Splatting

Zane K.J. Hartley<sup>a,1</sup>, Yong En Kok<sup>a,1</sup>, Lewis A. G. Stuart<sup>a</sup>, Andrew P. French<sup>a,b</sup> and Michael P. Pound<sup>a,∗</sup>

<sup>a</sup>University of Nottingham, School of Computer Science, Jubilee Campus, Nottingham, NG8 1BB, United Kingdom

<sup>b</sup>University of Nottingham, School of Biosciences, Sutton Bonnington Campus, Loughborough, LE12 5RD, United Kingdom

## A R T I C L E I N F O

Keywords:   
Digital Phenotyping   
3D Reconstruction   
Monocular 3D Reconstruction   
Video Difusion   
3D Gaussian Splatting   
Text-to-3D generation   
Synthetic Data

## A BS T RA C T

Modern digital phenotyping pipelines require the precise measurement of a range of 3D morphological plant traits. The accurate, high throughput acquisition of these traits helps breeders and researchers accelerate the development of new varieties, techniques and products. Recent advances in 3D technologies have enabled a rapid transition from 2D to 3D pipelines, but high throughput acquisition of digital 3D twins of plants remains a significant challenge due to complex morphology, thin structures and occlusions.

In this paper we present Plant Orbiter, a new approach to generating 3D plants that leverages recent advances in video difusion to produce high quality 3D Gaussian Splats of plants from a single image. We show that our method more accurately recreates plant morphology than other state-of-theart methods, achieving the highest 3D F-score against all baselines, while being more practical for high throughput pipelines due to easier alignment and reduced cost of camera placement. Combined with a 2D difusion model, we show that our method also achieves state-of-the-art results in the generation of 3D plant models from a text prompt.

## 1. Introduction

Digital 3D phenotyping presents substantial advantages in crop understanding over existing 2D analysis tools. Noninvasive methods for extracting accurate morphology enables a deep understanding of plant health [21], and can accelerate the development of new varieties, as well as providing better monitoring tools for growers [19]. 3D plant analysis unlocks many of the most important measures for growers, such as leaf area, and fruit volume while handling occlusion for dense canopies better than can be achieved with a single view. Within mainstream computer vision, the rapid development of 3D reconstruction [44, 27] and generation [70, 80] techniques has already lead to widespread adoption of 3D models for both generation and analysis.

Despite this, no current approach on plants achieves suficient morphological accuracy, or the rapid acquisition, that would be required for the high throughput phenotyping pipelines at scale needed for agricultural settings. Existing methods in 3D plant capture often exhibit one of a range of limiting factors. Hardware dependant approaches such as laser scanners or robotic capture environments have high upfront costs, and are often low throughput. Multi-view methods like Neural Radiance Fields (NeRF) [44] or 3D Gaussian Splatting (3DGS) [27] directly learn a 3D representation from image sequences around a plant, but rely on computationally expensive camera recovery from structurefrom-motion, which has high failure rates on complex biological structures. To tackle this, single image or monocular reconstruction methods have emerged in recent years to avoid the need for complex calibration of camera parameters. Image-to-3D generators [25, 69] attempt to learn a direct mapping from a single image to a 3D model, but often sufer from poor performance on plants, and their output is challenging to realign and scale to the input image [14]. Novel view difusion methods [8, 65], such as one included in our own pipeline, have not previously been applied to challenging domains like plants.

To improve the performance of reconstruction and other digital phenotyping tasks, the use of synthetic training data has proven increasingly popular due to a lack of available training datasets [64, 18]. 3D data is often more challenging to produce and label, so this scarcity is even more pronounced and has motivated recent work in the generation of synthetic 3D plants using procedural and generative models [17] alongside wider research in text-to-3D models [80, 77]. Despite recent advances, these methods sufer from high upfront costs, the need for hand-built per species rules, and in the case of generic models, poor performance on highly specific plant domains.

In this paper we present Plant Orbiter, a versatile pipeline for 3D reconstruction of accurate, automatically aligned plants. We leverage recent advances in video difusion models to produce an accurate, geometrically consistent orbital video around the plant based on a single input image, from which 3D Gaussian splatting can produce an accurate model. Our method both reduces the practical and computational costs compared to multi-image capture setups, and recreates plants with higher geometric accuracy compared to previous camera guidance methods [81, 65].

![](images/644e76ebe169e7de09b7fc263a6f7cf32897080cec8fd71ef0cd0bede93f1eea.jpg)  
Figure 1: The plant orbiter pipeline, showing both the reconstruction and text-to-3D setups. In step 1, we acquire an appropriate input image of a plant, either a single photograph or the output of a high-quality 2D difusion model. Step 2 and 3 show how a pretrained LoRA model is used in conjunction with MiniMax H3 to generate a 73 (figure shows 12 for simplicity) frame video with each frame at a predetermined position, forming an orbit around the photographed plant. Finally in step 4 we reconstruct the plant from the video using 3D Gaussian splatting, using the knowledge of the camera placements to both determine camera positions during initialisation, and to align the output plant to the input images frame.

By prepending a text-to-image difusion model to our pipeline, we also create a best in class model for the generation of entirely synthetic 3D plants that accurately adhere to a text prompt. We evaluate both the quality of synthetic plants but also the ability to prompt for specific phenotypic measurements, such as the number of fruits, and show that Plant Orbiter outperforms other leading text-to-3D models for both shape and prompt adherence.

The contributions of this paper can be summarised as follows:

• A single image 3D reconstruction pipeline that achieves state-of-the-art geometric accuracy on reconstruction of plants.

• Experimental results for highly accurate height estimation from single image reconstruction, showing that video difusion methods using our pipeline outperform 3D generation methods.

• A synthetic data generation pipeline that produces realistic synthetic plants, as well as experimental results showing our pipeline scores the highest for prompt adherence for phenotypical measurements.

## 2. Related Work

3D plant reconstruction. 3D plant reconstruction makes it possible to capture detailed, precise plant morphology traits and supports automated, high-throughput crop phenotyping [32]. 3D plants models have traditionally been obtained with laser scanning [35], structured light or RGB-D sensors [9], or from many overlapping photographs using structure-from-motion and multi-view stereo [68]. These approaches can be highly accurate, but they need dedicated equipment, controlled conditions and many viewpoints per plant. Plant structure also makes them fragile; thin leaves, overlapping organs and self-occlusion result in gaps and distortions in the reconstruction [32, 46].

Neural radiance fields (NeRF) [44] and 3D Gaussian splatting (3DGS) [27] have since substantially raised the fidelity of plant reconstruction from ordinary photographs. While NeRF variants successfully model complex appearance without depth supervision, their implicit representation makes it hard to extract the explicit surfaces needed for trait measurement [5, 54]. 3DGS instead models the scene with explicit Gaussian primitives, which train quickly and render in real time, recovering traits such as plant height and leaf dimensions with accuracy close to laser scanning [58, 41, 75]. Robotic capture systems extend them to occluded organs and growth tracking [3, 4]. However, these gains in fidelity have not reduced the cost of acquisition, as current methods still require tens to hundreds of calibrated photographs of each plant.

3D generation from images and text. Generative 3D models learn shape and appearance from 3D training data that let them produce a complete object, including its unseen sides, from a single image or a text prompt. Because 3D training datasets are far scarcer than image-text pairs, early text-to-3D methods optimised a 3D representation for each prompt by distilling a pretrained 2D difusion model through score distillation sampling (SDS) [48]. GaussianDreamer [77] applies SDS to 3D Gaussians for speed. MVDream [56] distils from a camera-conditioned multi-view difusion model, which reduces the multi-face artefacts that arise when each view is supervised independently. In parallel, imageconditioned methods learned to synthesise novel views from one image [37, 55, 38, 40] and then lifted them to 3D in a single forward pass [22, 62, 71]. TRELLIS [70] decodes structured sparse-voxel latents into Gaussians or meshes from an image or a prompt, and TRELLIS.2 [69] extends this to open and non-manifold surfaces. Hunyuan3D 2.1 [25] first generates the shape with a difusion transformer and then textures it in a separate stage, and DifSplat [34] repurposes an image difusion model to generate Gaussian splats.

Multi-view image models generate only a handful of views, whereas video difusion models, trained on vast realworld footage, learn how objects transform under continuous camera motion and can synthesise dense novel-view trajectories [7]. SV3D [65] and V3D [8] fine-tune Stable Video Difusion on rendered 360° orbits of synthetic objects, so that a single image yields a full orbit at known camera angles, to which a NeRF or Gaussian splat is then fitted. Later work improves the resolution and consistency of these orbits [74, 82, 42, 66], and SEVA [81] extends them to arbitrary camera trajectories from one or more input views. Camera control can also be added to general-purpose video generators, either by injecting camera poses [67, 20] or by training lightweight LoRAs for specific camera motions [60]. VIST3A [16] skips explicit frames and decodes scene-level 3D Gaussians directly from the latents of a textto-video model.

These models are trained mainly on synthetic 3D assets [12] and evaluated on general objects, and plants, with their thin, overlapping and highly variable organs, expose the limits of this training [17, 14]. Single-image methods built for plants do exist, but they mostly target trees and rely on hand-designed botanical priors [31, 33, 30, 39]. From text, general 3D generators also ofer little control over specific plant traits, such as the number of fruit, whereas text-toimage models, trained on far more data, follow such detailed prompts much more closely [79]. Our aim is therefore to generate 3D plants that are faithful to an input image or text prompt, leveraging the priors of existing text-to-image and video difusion models without requiring specialised 3D plant training data.

## 3. Method

## 3.1. Single Image Orbital Video

Our method leverages the strong understanding of 3D shape that has been learned by powerful video difusion models. We use Minimax H3 [45] along with a LoRA trained to produce orbital camera videos around the plant [11]. For each video the model is prompted with a single image of the front side of a given plant. We preprocess each image to remove all background elements using provided masks or SAM 2.1 [52] where masks are not provided. We then undistort the image, add it to a white canvas, and resize it to fit Minimax’s native 768 × 768 resolution. The input image is given the Rotary Position Embedding (RoPE) [59] position for the first and last frame in the sequence, ensuring the camera arc returns perfectly to the original location in the loop. Additionally, the model receives the text prompt used to train the original LoRA, conditioning the model to produce a consistent, level camera arc around the object while keeping the plant still to discourage motion that would otherwise reduce the model quality and enforce a static scene.

## 3.2. Known Pose Camera Recovery

Generally, the first stage of 3DGS pipelines is the extraction of camera positions using COLMAP [53], using structure from motion to correctly identify the cameras position and rotation within the scene. For 3D plant reconstruction, this method struggles to extract features on the repetitive, detailed structures of the plant surfaces, as well as being computationally expensive. Instead we exploit the consistent, known parameters of the orbital video to skip this stage of reconstruction, and use easily calculated camera poses based on a known camera path. With 73 frames including the duplicated frame for 0 and 72, each frame represents a uniformly spaced 5 degree increment in the orbit around the plant starting from an initial pose 1 arbitrary unit from the rotational axis. The camera’s path maintains a fixed distance and directed at this centre axis, which the video difusion model takes to be the object at the horizontal midpoint of the frame.

For efective training, 3D Gaussians are conventionally initialised from an initial point cloud generated from the structure-from-motion process employed to extract the camera poses. However, as COLMAP is skipped, we replace its initial Gaussians with ones we compute from a visual hull of the plant, ensuring high-quality 3D reconstruction. Each video frame is used to carve a silhouette cone [29] from a voxel grid, with each surviving voxel used as an initial Gaussian [73, 8]. As the video model does not output its focal length or elevation, we also calculate an estimate of these values during this process by determining the values that cause the projected carved hull to intersect most accurately with the plant masks across all viewpoints.

## 3.3. 3D Reconstruction

Training each Gaussian splat is performed using NerfStudio’s [61] Splatfacto [27, 76] model with default parameters, apart from enabling antialiased rasterisation and scale regularisation (with maximum Gaussian axis ratio of 3) to remove long, needle like Gaussians. We perform masked training so that only the plant is reconstructed, and the 3D model is trained for 30,000 iterations, which is standard practice for 3DGS reconstruction. For each plant, we also produce a matching 2D Gaussian Splat (2DGS) [23] model, which replaces the 3D gaussians with flat 2D disks and is trained with additional depth-normal consistency and depth distortion losses. Because of this 2DGS models produce more surface-aligned reconstructions compared to the original 3DGS implementation, allowing for a mesh to be extracted with truncated signed distance function (TSDF) [10] after training which we then use for our 3D geometric evaluation.

## 3.4. Alignment

In order to accurately evaluate the 3D plant reconstructions, alignment between generated models and the ground truth must first be established. All 3D reconstruction models return the plant within their own arbitrary canonical frame, which require a transform (scale, rotation, and translation) to bring them into the frame of the real cameras for evaluation. This reflects real world usage where the ability to accurately phenotype from a reconstruction relies on it being correctly scaled and positioned within its frame; however precise realignment is a known challenge [14].

To precisely position reconstructed plants within the real scene, a known real world measure is required to determine the scale of the reconstructed object. In a capture environment designed with this in mind, an object of known dimensions can be used as a reference. As a reference is not provided in each dataset we use, we instead calculate the real world scale of each plant from the plant’s distance from the input camera, which we derived from each dataset’s calibrated camera pose data. We then scale each reconstructed object such that the orbit radius matches this distance, adjusting for diferent focal lengths. After scaling, each reconstruction must be orientated and positioned to match the position of the ground truth. For Plant Orbiter, we initialise this process by aligning the first orbit camera to the input image’s camera. Afterwards, we fit the reconstruction’s silhouette to the plant mask as seen from the input image (or for oracle placement, all held out view) over 300 steps using an Adam Optimiser. This gives a similarity correction using uniform scale, 3D rotation and translation which provides our final alignment.

For the 3D generator baselines evaluated against in this paper we cannot rely on a known start position or radius, since these models produce a 3D reconstruction in an unknown canonical frame depending on the plant and generator. Instead we perform scaling by matching the reconstructed plant’s height against the pixel height in the input image. Then to normalise each reconstruction, we orient the plant upright using the generators default up axis, or using its longest axis with its pot as the base where this method fails. We then iterate over 10° yaw increments over 0.8, 1.0 and 1.25 scales and rank for silhouette IoU against the alignment view(s). We double the number of steps taken during correction to maximise the precision of the final alignment, ensuring a fair comparison between plant reconstructions.

To match a real world scenario where only a single image is captured, we compute this similarity against only the single image used to generate the 3D model (experiments labelled input view placement), however we also complete a second evaluation round aligning against all held out views (experiments labelled oracle), to determine true accuracy score assuming an upper bound alignment.

## 3.5. Text-to-3D generation

Since Plant Orbiter needs only a single image, it can be turned into a text-to-3D generator by first generating a synthetic image from a text prompt using a text-to-image model, and then lifting it to 3D with the pipeline described earlier. This approach inherits the strong prompt adherence of standard 2D text-to-image models, which are trained on far more data than is available in 3D datasets, allowing finegrained attributes such as the number of fruit to be specified in the prompt.

We use Qwen-Image-2.1 [50] with default parameters to generate 2048 × 2048 pixel input images. To ensure the generated image matches the input expected by the orbital video model, we append a fixed framing sufix to the content prompt. For example, the prompt for bean becomes “A photo of bean plants. Single potted plant, whole plant and pot fully in frame and centred with margin around it, eye-level side view, isolated on a plain white background, studio lighting.” The sufix is identical for every plant species generated, and adds only framing, background and lighting terms, so the plant description itself is the same prompt that every baseline receives.

## 4. Single-Image Reconstruction Experiments

## 4.1. Datasets

For our experiments, we evaluate over a range of diferent multi-view plant datasets captured using a combination of cameras mounted on robot arms, turntables, bespoke multi-camera capture environments and handheld capture. For each plant we select one real photo showing the entire plant, which is then used as the input for all single view reconstruction.

The PlantDreamer [17] dataset constitutes the primary dataset we evaluated, containing multi-view captures of bean, mint and kale plants captured using two Universal Robots UR5 robotic arms. The overall dataset contains 21 individual plants, including 12 instances of bean plants, 6 instances of mint and 3 instances of kale. For our experiments, we omit a single instance of bean due to incorrect mask data included in the published dataset. Each plant is captured with approximately 220 views at a range of elevations, and 27 images were included in the evaluation dataset for assessing the photometric quality of the reconstruction. For a fair comparison, these evaluation images were solely used as the reference images in the single-image generation models. PlantDreamer does not come with a true 3D ground truth and instead provides point clouds derived from 3D Gaussian splats [1], which are produced using the images provided which we use as a 3D reference during experimentation. During experimentation we observed that the masks provided with PlantDreamer were poorly cropped to the plant, resulting in poor performence, so applied an intersection between the provided masks and IS-Net [49] for all experiments to improve the segmentation.

The Berkeley Botany-Bot [3] dataset contains 8 diferent houseplants and was captured by placing the plant on a turntable, acquiring images at four elevations. Each plant has 91-96 total views, and we evaluate using all held-out samples for each experiment. Botany-Bot neither provides a 3D ground truth nor an appropriate 3D reference, so instead we generate our own 2DGS from the real photos provided as a 3D reference.

The PLANesT-3D [43] dataset contains 10 instances of potted pepper and rose plants, each captured manually using a handheld DSLR camera. These plants were captured from 180-400 viewpoints and evaluated using 64 evenly spaced evaluation views taken from the unused images. We exclude a single pepper instance during 3D evaluation because its point cloud data could not be aligned to its photos, suggesting movement during capture. For 3D, PLANesT provides labelled Metashape point clouds which are registered to the cameras by a silhouette fit.

The GrowSplat [4] dataset, by contrast to the previous methods, is a time series dataset showing a quinoa and a sequoia plant each growing over 17 time steps. For the purposes of our experiments, we remove the earliest time step for quinoa, as the plant at that stage was too small to be reconstructed by any of the methods evaluated in this paper. The images were captured using an imaging box that contained fixed cameras at five positions around the plant, at three fixed heights, resulting in a total of 15 views per plant. We evaluate over all 14 unused held out images during our evaluations. Each plant in GrowSplat also provides a metric plant point cloud using an accurate LiDAR scanner, which we use as a height reference for our downstream evaluations.

For all datasets the background was digitally removed, using either provided masks to isolate the plant and pot from the background, or our own masks generated with a SAM 2.1 [52] segmentation pipeline. This was done to cleanly produce a Gaussian splat of the target plant and remove background artifacts in the orbit video that did not consistently rotate with the plant.

## 4.2. Baselines

To evaluate our method we select a number of SotA monocular 3D reconstruction methods to compare against. We include TRELLIS [70], TRELLIS.2 [69], and Hunyuan3D 2.1 [25] (the most recent version available for local use), as these are currently the highest performing available models that could be employed as part of a 3D phenotyping pipeline, and have seen interest in previous phenotyping works for single-image reconstruction [14]. TRELLIS produces its own Gaussians and a mesh for 3D scoring, while both TRELLIS.2 and Hunyuan3D 2.1 produce textured meshes which we convert into Gaussians for rendering, and are run using default settings as described in their documentation. Each method receives the same input image having undergone the same preprocessing steps as Plant Orbiter. We opt to give each baseline our own cropped image, rather than relying on built in background removal, as we found poorer performance at this stage unfairly penalised these models relative to our own.

We additionally test against SEVA [81] and SV3D [65], two other methods that use video difusion for 3D reconstruction. SEVA is a multi-view difusion model that we configure to generate 72 views around the object, matching our setup, while SV3D is a video difusion pipeline, from which we configured to generate 21 frames around the plant at 10° of elevation. As neither of these models natively generate a 3D reconstruction, we append our downstream pipeline, including use of known poses, as well as splatting and meshing pipelines to produce the outputs used in our evaluation. As such, this test primarily evaluates the accuracy of the orbital LoRA compared to standard video difusion approaches.

To fairly evaluate the baseline outputs, proper alignment between the 3D plant models was needed. Reconstruction models generally produce output in their own canonical frame, and reorienting and scaling them is a known challenge [14]. A core advantage of video difusion method is the implicit alignment with the input image that is automatically achieved, while baseline methods were aligned by matching the reconstructions silhouette against the 2D mask. Full details on alignment are found in section 3.4. Finally, to improve reliability of our results, all experiments on reconstruction quality in sections 4.4 and 4.5 were carried out using three diferent seeds.

## 4.3. Evaluation Metrics

We perform evaluation across three sets of experiments shown in tables 1, 2, and 3, using metrics defined below.

2D evaluation is performed using a range of metrics which directly compare renders of the reconstruction against ground truth held out images of the same camera pose. We use four metrics: IoU (the overlap between the render and ground truth silhouettes divided by their union), PSNR (the pixelwise error between the render and ground truth), LPIPS [78] (deep feature distance between the render and ground truth image), and DISTS [13] (comparison of structure and texture between render and ground truth, included for being more tolerant to small misalignments). For fairness, especially with pixel error metrics, we apply colour correction to the renders of each method’s output using a transform fitted to each input image.

While 2D IoU already gives some indication of 3D structural accuracy, we also include 3D evaluation where we compare the aligned 3D reconstruction against a ground truth or derived 3D representation from each full dataset. For this we use Chamfer Distance (the average distance between each point on the reconstruction to the nearest point on the ground truth, and again from the ground truth to the reconstruction both averaged, normalised to the reference bounding box and shown as $\times 1 0 ^ { 3 } )$ and F-score (the harmonic mean of the fraction of points in the reconstruction within � of the reference and vice versa, where � is 1% of the reference bounding box diagonal).

## 4.4. Single Image Reconstruction

Results in Table 1 demonstrate that for the input-view case, we score best for shape measures for PlantDreamer (0.552 vs 0.529) and Botany-Bot (0.556 vs 0.504) and score second on the PLANesT (0.343 vs 0.348). For DISTS measurements of perceptual similarity, we score best across all datasets, while for pixel metrics PSNR and LPIPS we are among the top three best performing models. Visualisations of all models results for all datasets can be seen in Fig. 2.

Table 1 shows separate results for input-view placement and oracle placement. These results show substantial performance improvements are gained by all models through more accurate placement, and notably that the improvement of the TRELLIS and Hunyuan3D models was larger than those of our method, with TRELLIS.2 now leading in some metrics having received the largest improvement due to alignment. By contrast, Plant Orbiter continues to lead Botany-Bot across all four metrics, as well leading in IoU for PLANesT. This shows that placement from a single frame is unreliable for the generator baselines, whereas our method is much stronger in input-view placement due to orientation being known by construction. We find that the oracle placements for the generator models have median $1 6 ^ { \circ }$ (TRELLIS.2), 27° (TRELLIS) and $4 8 ^ { \circ }$ (Hunyuan3D) diference between input placement and oracle placement, demonstrating that single image alignment is repeatedly insuficient to align the outputs of these models.

The input-view alignment best reflects the real world situation where an agronomist would want to take measurements using a single hand-captured image, and it is in this setting that our method performs best. We also observe that even with Oracle placement, our method still achieved the second highest score for PlantDreamer, and the highest score for Botany-Bot and PLANesT, despite the latter being the most challenging dataset. This shows that our methods advantage is not from alignment alone, but also matches or exceeds the strongest generator on shape, and we argue geometric measures are the most important for downstream phenotyping which is supported further in section 4.5.

We do observe that while our method performs best in spatial and perceptual metrics (IoU and DISTS respectively), our method is less comparatively strong in pixel similarity metrics such as PSNR, where SV3D is the strongest overall method. Characteristics of pixel similarity metrics mean that fine details with small misalignments are punished more, while lower detail, and blurred geometry is more highly rewarded. We also observe that the absolute PSNR values for all methods are generally poor, likely caused by the invented details on the occluded side of the plant. We believe for these reasons PSNR is less reliable as a signal for the overall quality of reconstruction for these datasets.

Overall, our results show that our method is the best performing model on shape and perceptual accuracy, with substantial gains over the currently favoured generator models, such as TRELLIS.2, for the real world case of single image generation and alignment. We also show that our method creates the geometrically highest performing reconstructions on most datasets, even in the ideal case of oracle alignment.

## 4.5. 3D Geometry

In order to evaluate our output 3D accuracy, we carry out an additional evaluation on Plant Orbiter against the baselines. Since the 3D Gaussian splats produced by our method have no explicit surface, we instead train a 2D Gaussian splat using the same generated orbit video as in our previous approach and extract a mesh from it, giving a surface that can be directly compared against the output from the other baselines. We evaluate over Chamfer distance and F-score to evaluate the true accuracy in 3D of our method and the baselines. Following alignment described in 3.4, we perform an additional normalisation step followed by 200 iterations of the rigid iterative closest point algorithm to properly align the 3D model with the reference point cloud. For PLANesT Dataset, we additionally crop the pot from the 3D model to match the contents of the reference point cloud.

We report all results in Table 2 with best performing methods highlighted in bold. The results show that our method outperforms all baselines in F-score and has the best Chamfer distance on all but PlantDreamer, where we score slightly worse than TRELLIS.2 (18.3 to 17.7). Within each dataset, we individually beat all baselines on a majority of individual plants, with the clearest wins on Botany-Bot and PLANesT. This highlights how our method excels at generating geometrically consistent plants between frames, which is important for accurate reconstruction and phenotyping.

## 4.6. Downstream Trait Evaluation

To evaluate our model for representing real downstream plant traits using 3D reconstruction, we select plant height estimation (a common type of spatial phenotyping task) that is well suited for evaluating 3D reconstruction approaches. For these experiments, we use the dataset GrowSplat, a temporal dataset showing two growing plants over a series of 17 time steps. By evaluating the ability of each reconstruction technique to measure the growing height of each plant over many time steps, we seek to verify firstly, to what extent 3D reconstruction methods can be used to measure major phenotypic traits such as height, and additionally to measure the performance of PlantOrbiter with respect to the baselines. We measure each plant from the base of the plant, as taken from the reference, removing the pot and soil prior to measuring. To remove any individual stray Gaussians above the top of the plant, we measure the height by downsampling to one point per 5mm voxel and take the 99.5th percentile of the points above its base.

We show numerical results in Table 3 and additional time series data in Fig. 3. The results show that the performance for downstream evaluation from all three camerabased methods (mean MAPE 3.7 - 5.9%) are much more closely able to estimate the plant heights compared to the generator methods (mean MAPE 12.7 - 15.8%). While both SV3D and SEVA are the best models overall, our method is similarly much more performant in these tasks compared to Hunyuan3D 2.1, the current best generator model. For plant height in particular, our method closely tracks the recorded growth curve, with a slight 6% overestimate of height, while in Fig. 3 we see SEVA has less consistent tracking despite the lower average error. As all three of the best performing model use our reconstruction and camera placement models, we attribute their relatively much higher performance to our pipeline as opposed to the specific video models.

Given / reference

Image-to-3D generators

View-synthesis models

![](images/de368ad3013b54e294f34af55d7c6a9de1ed209c9e0e800b9d61b0586cf26fee.jpg)  
Figure 2: Outputs of our model and all baselines showing examples from each test set. Column 1 shows the input image used for single image 3D reconstruction and column 2 shows the held-out image from an unseen view. For each output we render the reconstruction from the held-out view using oracle placement, demonstrating each objects ability to accurately recreate the occluded side and plant geometry from the front image.

## Table 1

Monocular Reconstruction scored in 2D on unseen images. Results shown for input-view placement, where alignment is performed using the input image only and oracle-view placement, where all held out images are used. We evaluate 2D geometry (IoU), pixel accuracy (PSNR) and feature alignment (LPIPS and DISTS). Best reconstruction results for each dataset and metric are shown in bold, second best are shown underlined. PlantDreamer and Botany-Bot are abbreviated to PD and BB respectively. Full capture indicates scores for splats reconstructed from all non-test real images and acts as an upper bound.
<table><tr><td></td><td colspan="3">loU↑</td><td colspan="3">DISTS↓</td><td colspan="3">PSNR↑</td><td colspan="3">LPIPS↓</td></tr><tr><td>Method</td><td>PD</td><td>BB</td><td>PLANesT</td><td>PD</td><td>BB</td><td>PLANesT</td><td>PD</td><td>BB</td><td>PLANesT</td><td>PD</td><td>BB</td><td>PLANesT</td></tr><tr><td colspan="10">Input-view placement</td><td></td><td></td><td></td></tr><tr><td>TRELLIS</td><td>0.406</td><td>0.446</td><td>0.269</td><td>0.225</td><td>0.229</td><td>0.246</td><td>5.75</td><td>7.86</td><td>5.24</td><td>0.083</td><td>0.089</td><td>0.099</td></tr><tr><td>TRELLIS.2</td><td>0.385</td><td>0.504</td><td>0.281</td><td>0.227</td><td>0.220</td><td>0.238</td><td>5.44</td><td>8.56</td><td>5.62</td><td>0.084</td><td>0.085</td><td>0.095</td></tr><tr><td>Hunyuan3D 2.1</td><td>0.336</td><td>0.482</td><td>0.288</td><td>0.262</td><td>0.241</td><td>0.239</td><td>5.01</td><td>8.78</td><td>5.98</td><td>0.089</td><td>0.087</td><td>0.094</td></tr><tr><td>SEVA</td><td>0.448</td><td>0.442</td><td>0.285</td><td>0.215</td><td>0.213</td><td>0.246</td><td>6.57</td><td>8.66</td><td>5.40</td><td>0.082</td><td>0.083</td><td>0.097</td></tr><tr><td>SV3D</td><td>0.529</td><td>0.495</td><td>0.348</td><td>0.201</td><td>0.217</td><td>0.234</td><td>8.25</td><td>9.27</td><td>6.23</td><td>0.072</td><td>0.081</td><td>0.093</td></tr><tr><td>Ours</td><td>0.552</td><td>0.556</td><td>0.343</td><td>0.199</td><td>0.194</td><td>0.226</td><td>6.75</td><td>9.19</td><td>5.74</td><td>0.082</td><td>0.080</td><td>0.095</td></tr><tr><td colspan="10">Oracle placement</td><td colspan="3"></td></tr><tr><td>TRELLIS</td><td>0.585</td><td>0.557</td><td>0.333</td><td>0.190</td><td>0.207</td><td>0.216</td><td>8.02</td><td>9.08</td><td>6.73</td><td>0.074</td><td>0.081</td><td>0.091</td></tr><tr><td>TRELLIS.2</td><td>0.620</td><td>0.581</td><td>0.344</td><td>0.179</td><td>0.208</td><td>0.204</td><td>9.02</td><td>9.63</td><td>7.38</td><td>0.069</td><td>0.079</td><td>0.087</td></tr><tr><td>Hunyuan3D 2.1</td><td>0.522</td><td>0.578</td><td>0.366</td><td>0.215</td><td>0.227</td><td>0.206</td><td>7.65</td><td>9.75</td><td>7.68</td><td>0.076</td><td>0.081</td><td>0.086</td></tr><tr><td>SEVA</td><td>0.547</td><td>0.543</td><td>0.363</td><td>0.197</td><td>0.205</td><td>0.219</td><td>8.37</td><td>9.83</td><td>6.81</td><td>0.073</td><td>0.079</td><td>0.091</td></tr><tr><td>SV3D</td><td>0.591</td><td>0.575</td><td>0.391</td><td>0.197</td><td>0.211</td><td>0.221</td><td>9.26</td><td>9.87</td><td>7.18</td><td>0.069</td><td>0.077</td><td>0.089</td></tr><tr><td>Ours</td><td>0.612</td><td>0.643</td><td>0.408</td><td>0.188</td><td>0.188</td><td>0.205</td><td>8.21</td><td>10.19</td><td>6.87</td><td>0.074</td><td>0.074</td><td>0.090</td></tr><tr><td>Full capture</td><td>0.920</td><td>0.925</td><td>0.784</td><td>0.087</td><td>0.070</td><td>0.082</td><td>18.47</td><td>20.32</td><td>17.29</td><td>0.030</td><td>0.029</td><td>0.033</td></tr></table>

Monocular Reconstruction scored in 3D using reference point clouds. All reconstructed plants are aligned with oracle placement and further aligned with rigid ICP. For PLANesT we evaluate the plant only and digitally remove the pot for evaluation. We show results for geometric accuracy in 3D evaluated with Chamfer Distance and F-score. Best result for each dataset and metric are shown in bold, second best are shown underlined.
<table><tr><td rowspan="2">Method</td><td colspan="2">PlantDreamer (n=20)</td><td colspan="2">Botany-Bot (n=8)</td><td colspan="2">Pepper  $\scriptstyle ( n = 9 )$ </td><td colspan="2">Rose  $\scriptstyle ( n = 1 0 )$ </td></tr><tr><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>TRELLIS</td><td>20.6</td><td>0.367</td><td>22.8</td><td>0.426</td><td>68.5</td><td>0.142</td><td>52.2</td><td>0.168</td></tr><tr><td>TRELLIS.2</td><td>17.7</td><td>0.427</td><td>22.6</td><td>0.464</td><td>65.6</td><td>0.161</td><td>59.6</td><td>0.157</td></tr><tr><td>Hunyuan3D 2.1</td><td>34.8</td><td>0.300</td><td>24.4</td><td>0.424</td><td>44.5</td><td>0.208</td><td>51.0</td><td>0.204</td></tr><tr><td>SEVA</td><td>30.8</td><td>0.332</td><td>28.5</td><td>0.378</td><td>39.5</td><td>0.216</td><td>42.0</td><td>0.217</td></tr><tr><td>SV3D</td><td>20.9</td><td>0.410</td><td>23.5</td><td>0.434</td><td>35.7</td><td>0.230</td><td>40.4</td><td>0.215</td></tr><tr><td>Ours</td><td>18.3</td><td>0.443</td><td>15.9</td><td>0.540</td><td>30.2</td><td>0.252</td><td>37.1</td><td>0.257</td></tr></table>

Plant height estimation from one photo 3D reconstructions, as tested on GrowSplat (using 16 time steps for quinoa and 17 time steps for sequoia) and PLANesT pepper (n=9) and rose (n=10). We evaluate over Mean Absolute Percent Error (MAPE) and Spearman Rank Correlation �. For these experiments all reconstructions are aligned with input-view placement, to best represent single image performance in a downstream setting. Best result for each dataset and metric are shown in bold, second best are shown underlined.
<table><tr><td></td><td colspan="4">GrowSplat (time series)</td><td colspan="4">PLANesT-3D (hand-held)</td><td rowspan="3">Mean</td></tr><tr><td></td><td colspan="2">Quinoa</td><td colspan="2">Sequoia</td><td colspan="2">Pepper</td><td colspan="2">Rose</td></tr><tr><td>Method</td><td>MAPE↓</td><td>ρ↑</td><td>MAPE↓</td><td>ρ↑</td><td>MAPE↓</td><td>ρ↑</td><td>MAPE↓</td><td>ρ↑</td></tr><tr><td>TRELLIS</td><td>7.4</td><td>0.98</td><td>11.2</td><td>0.95</td><td>28.9</td><td>-0.13</td><td>12.7</td><td>0.90</td><td>15.1</td></tr><tr><td>TRELLIS.2</td><td>9.8</td><td>0.96</td><td>12.9</td><td>0.74</td><td>23.9</td><td>-0.02</td><td>16.6</td><td>0.83</td><td>15.8</td></tr><tr><td>Hunyuan3D 2.1</td><td>17.0</td><td>0.97</td><td>12.0</td><td>0.57</td><td>6.5</td><td>0.73</td><td>15.1</td><td>0.75</td><td>12.7</td></tr><tr><td>SEVA</td><td>4.3</td><td>0.99</td><td>4.0</td><td>0.89</td><td>2.5</td><td>0.95</td><td>4.8</td><td>0.96</td><td>3.9</td></tr><tr><td>SV3D</td><td>3.9</td><td>0.99</td><td>2.5</td><td>0.91</td><td>2.5</td><td>0.95</td><td>6.0</td><td>0.95</td><td>3.7</td></tr><tr><td>Ours</td><td>7.6</td><td>0.98</td><td>5.9</td><td>0.94</td><td>3.0</td><td>0.95</td><td>6.9</td><td>0.93</td><td>5.9</td></tr></table>

## 5. Text-to-3D Generation Experiments

## 5.1. Baselines

We compare our text-to-3D pipeline against SotA textto-3D methods: GaussianDreamer [77], DifSplat (SD3.5m) [ TRELLIS-text (xlarge) [70], and MVDream [56]. All of them output 3D Gaussian splats except MVDream, which outputs a NeRF (marked <sup>†</sup>). Each baseline is run with the default parameters and with the same seed numbers as our method.

Every baseline receives the prompt “A photo of ⟨species⟩ plants. Single potted plant, whole plant.” The first sentence is the guidance prompt of PlantDreamer [17], whose real plants serve as our reference set in Section 5.2, and the second asks every method for a single, complete potted plant, matching more closely to the reference plants against which all methods are scored. Our method receives the same content prompt followed by the framing sufix described in Section 3.5. PlantDreamer itself is not included as a baseline, since it requires an L-system mesh as input rather than text alone, and it was trained on the same plants that form our reference set.

In the fruit-count experiment (Section 5.3) we additionally include TRELLIS.2 [69], one of the strongest of the 3D generators evaluated in Section 4, and give it the same Qwen-Image-2.1 input images as our method. Since both methods then start from an identical image, any diference between them in the fruit count is attributable to the 3D lifting stage alone.

We generate bean, kale and mint, the three species of the PlantDreamer dataset (Section 4.1), so that the generated plants can be compared with real ones. Each method generates 12 assets per species, with seeds 1–11 and 42, giving 36 assets per method.

Appearance is assessed with three image metrics, each averaged over the 72 views of an asset: the CLIP score [51] (100× the cosine similarity between the image and prompt embeddings of OpenAI CLIP ViT-L/14); ImageReward [72], a learned model of human preference for text-to-image outputs; and the VBench imaging-quality score [24], a noreference measure of distortions such as blur and noise. For every method, CLIP and ImageReward score each view against the bare prompt “A photo of ⟨species⟩ plants”, so that neither the object sentence nor our framing sufix afects these scores.

For realism, we compute the kernel distance (KD) between the rendered views and real photographs of the same species [6]. The reference set consists of photographs of the 11 bean, 3 kale and 6 mint plants of the PlantDreamer dataset described in Section 4.1, with the background removed in the same way, excluding the photographs used to train PlantDreamer and restricted to camera elevations of −5<sup>◦</sup> to 35<sup>◦</sup> to match the renders. We compute KD on Inceptionv3 pool features (reported ×100) and on DINOv2-B CLS features [47].

![](images/ea13452846d37d88b767061cf348412f71ac9b4ed82ef6d25bce35b54761563f.jpg)  
Figure 3: Plant height over time on GrowSplat, reconstructed from a single photo per imaging date.

To assess shape, we score each asset against Plant-Dreamer’s real point clouds of the same species with the F-score [63], after scaling both clouds to unit height and aligning them rigidly. As a generated plant is compared with a diferent real plant rather than with its own reference, we use a looser threshold, � = 2% of plant height, chosen on real plants alone: two real plants of the same species score 0.66 at this threshold, against 0.54 at 1% and a saturated 0.85 at 5%. Following [2], MMD-F averages over the real plants the F-score of the closest generated asset, measuring how closely the best generated shapes match real plants, and COV is the fraction of real plants that are the nearest neighbour of at least one generated asset, measuring how many distinct real shapes are covered. As a reference level, the real plants compared with one another reach an MMD-F of 0.659 and a COV of 0.510.

Table 4 shows that our method achieves the highest CLIP score and ImageReward, above every baseline. Our assets correspond most closely to the requested species and are the most preferred under a learned model of human preference. They are, however, less sharp than TRELLIStext, MVDream and DifSplat. A likely cause is that our splats are fitted to a single, nearly level orbit, so the views at the higher evaluation elevations lie outside the generated trajectory, where our reconstructions are prone to artefacts.

Realism depends on the feature space. ImageNet-trained Inception features favour texture over shape [15], while DINOv2 features capture object structure and track human judgements more closely [57]. Using DINOv2 features, our model ranks second, behind only TRELLIS-text. Conversely, on Inception features, it ranks last among the five methods. This disparity is mainly driven by the "mint" (30.54). Our method generates dense, bushy plants, often in terracotta pots, whereas the reference mint plants are sparse and leggy and sit in black pots.

On shape, our method achieves the highest MMD-F, showing our closest generated assets match real plant shapes most accurately. However, its lower coverage indicates our shapes are less diverse across seeds than baseline outputs.

Figure 4 shows the generated plant samples per species for each method, together with real PlantDreamer photographs. Compared with these real plants, our assets closely resemble those of the real species, though real mint samples are noticeably taller and sparser than ours, which is consistent with our larger KD on mint.

In summary, although our method trails the TRELLIStext slightly in realism, it demonstrates superior prompt adherence in generating specified plant species and achieves the most accurate structural match to real plant geometry.

## 5.3. Synthetic Fruit Counting Experiments

To test whether a fine-grained attribute specified in the prompt is preserved in 3D, we prompt each method for a given number of fruit, using “A photo of ⟨species⟩ plants with ⟨�⟩ ⟨fruit⟩. Single potted plant, whole plant.” (e.g. “with five tomatoes”). We write � as a word, since textto-image models follow number words more accurately than digits [26], which we also observed for Qwen-Image-2.1 in a pilot study. We generate tomato and lemon plants for � = 1 to 7 with seeds 42, 1 and 2, giving 42 assets per method. As in Section 5.2, our method receives the same content prompt followed by the framing sufix.

Fruit are counted by eye by one author, blind to the requested number and method, from rendered views covering each asset from all sides. Each distinct fruit of any colour is counted once, including fused, detached and blurred fruit, but not colour on a leaf or stem and specks visible from only one angle. A second author, independently re-counted one asset per method, species and � (84 of the 252 assets) and 14 of the 42 input images. The two counters agreed exactly on 91.7% of the assets (Krippendorf’s � = 0.98 [28]) and on all 14 input images. Every disagreement concerned the ill-formed fruit of DifSplat (5 of 14 assets) or MVDream (2 of 14), yet none changed whether an asset matched the requested number, so each method’s exact-count rate on the subset is the same under either counter. Table 5 uses the reported counts of the first counter.

Text-to-3D plant generation: 12 seeds per species (bean, kale, mint), $n { = } 3 6$ assets per method. Appearance: mean over 72 views per asset. Realism: kernel distance to PlantDreamer’s real photos, mean over species, 95% bootstrap interval in brackets; Shape: against PlantDreamer’s real point clouds; real plants vs each other: MMD-F 0.659, COV 0.510. Bold / underline: best / second-best text-only method. <sup>†</sup>NeRF output.
<table><tr><td></td><td colspan="3">Appearance</td><td colspan="2">Realism</td><td colspan="2">Shape</td></tr><tr><td>Method</td><td>CLIP↑</td><td>ImageReward↑</td><td>Imaging↑</td><td> $\mathsf { K D } _ { \mathsf { I n c e p } } \downarrow$ </td><td> ${ \mathsf { K D } } _ { \mathsf { D I N O v } 2 } \downarrow$ </td><td>MMD-F↑</td><td>COV↑</td></tr><tr><td>GaussianDreamer</td><td>23.41</td><td>-0.935</td><td>0.382</td><td>22.74 [20.8, 25.0]</td><td>53.44 [51.5, 55.4]</td><td>0.432</td><td>0.308</td></tr><tr><td>DiffSplat</td><td>20.54</td><td>-1.340</td><td>0.457</td><td>24.83 [22.8, 27.3]</td><td>58.44 [56.6, 60.2]</td><td>0.480</td><td>0.475</td></tr><tr><td>TRELLIS-text</td><td>19.89</td><td>-1.806</td><td>0.600</td><td>18.20 [15.8, 21.1]</td><td>35.26 [32.8, 37.9]</td><td>0.514</td><td>0.419</td></tr><tr><td>MVDream†</td><td>20.55</td><td>-1.349</td><td>0.542</td><td>23.99 [22.1, 26.2]</td><td>60.97 [59.5, 62.5]</td><td>0.491</td><td>0.338</td></tr><tr><td>Ours</td><td>24.99</td><td>-0.687</td><td>0.412</td><td>25.30 [22.8, 28.1]</td><td>39.46 [36.9, 41.8]</td><td>0.539</td><td>0.288</td></tr></table>

We report the percentage of the percentage of assets whose counted fruit equals the requested � (Exact Number Generation), the mean absolute count error (MAE) and, for our method and TRELLIS.2, the percentage whose 3D count matches their input image (=input). CLIP, ImageReward and VQAScore [36], which uses a vision-language model to estimate the probability that an image shows the prompt, are averaged over the 72 views and computed without the count (e.g. “A photo of tomato plants with tomatoes”), so that they measure image quality and the presence of fruit.

Our method preserves the fruit count of its input image in 40 of the 42 assets (95.2%; Table 5).The two exceptions are both tomato assets whose input image shows six fruit and whose 3D asset gains one, giving seven (see Fig. 5). For the � = 6 asset, this turns a correct count into an error, while for the � = 7 asset it turns an error into a correct count. The two changes cancel out, so our Exact Number Generation and MAE are identical to those of the input images. TRELLIS.2, lifting the same images, preserves the count in only 23 of 42 assets (54.8%), and 18 of its 19 errors add fruit. We attribute this diference to how each method uses the input image. In our pipeline, the input image is pinned as the first and last frame of the orbit video, which only moves the camera around a static scene (Section 3.1), so the visible fruit are carried into the splat. TRELLIS.2 instead uses the image only as conditioning and generates the whole asset, including its hidden side, from its 3D prior, leaving it free to add fruit.

Because our 3D counts follow the input images so closely, our remaining errors originate in the text-to-image stage. As Fig. 5 shows, these errors depend on the species, with tomato images having at most one fruit too few and lemon images up to two too many, and they become more frequent as � grows.

End-to-end, our method achieves the highest Exact Number Generation and the lowest MAE of all methods. Notably, TRELLIS-text, which produced the most realistic renders in Section 5.2, never produces the requested count (0.0%, MAE 3.62), as it mostly omits the fruit altogether (Fig. 5). On the quality metrics, our method achieves the highest VQAScore, close to that of its input images, and ranks second on CLIP and ImageReward, behind TREL-LIS.2. These diferences are small, and as TRELLIS.2 is rendered as a textured mesh and our method as a splat, they may partly reflect the diferent output representations.

Figure 6 shows examples for � = 3 and � = 6. The text-to-3D baselines often generate malformed or missing fruit, and mostly match the requested count at a rate close to or below random chance (14.3%). These failures are consistent with the observation in Section 2 that text-to-3D models, trained on far less data than text-to-image models, ofer little control over specific plant traits. Our method and TRELLIS.2 avoid this limitation by starting from an image generated by a strong text-to-image model.

Overall, our method follows the requested fruit count more accurately than every text-to-3D baseline and almost always preserves the count of its input image, so its accuracy is limited mainly by the text-to-image model and should improve with it.

## 6. Discussion

## 6.1. Advantages over existing methods

We believe that Plant Orbiter presents several advantages over existing approaches to 3D reconstruction of plants. Firstly, monocular methods of 3D reconstruction present numerous practical advantages for phenotyping applications. High throughput pipelines, like those normally used in plant science and agriculture, must work eficiently and at scale, making capture using complex imaging setups or expensive robots impractical. Eficient, one or few shot reconstruction approaches, captured with a handheld camera present a much more practical approach in a wide range of settings.

Relative to other generation methods such as TRELLIS or Hunyuan3D, our work ofers a substantial practical advantage by being able to easy align the model to its input image; even with respect to other video difusion models, the ability to predict camera poses based on the known path provides computational benefits at scale.

When compared to other reconstruction from video, including real and synthetic methods like SEVA or SV3D,

![](images/fe3e0d2b40742fb72587e81681e26b158a0ce7a4279dd0b01f6785f65d2bf44f.jpg)  
Figure 4: Text-to-3D samples (seeds 1 and 2). Every baseline receives the prompt “A photo of ⟨species⟩ plants. Single potted plant, whole plant.” Ours receives the same text with the final period replaced by its text-to-image sufix (“. . . whole plant and pot fully in frame and centred with margin around it, eye-level side view, isolated on a plain white background, studio lighting.”), generates an image from it and lifts that image to 3D. <sup>†</sup>NeRF output.

## Table 5

Prompted fruit counts (requested � = 1–7; 3 diferent seeds per species (tomato, lemon); �=42 assets per method). Fruit counted by eye over each asset’s 36 evaluation views and turntable, each fruit once, all assets included; agreement with a blinded second counter on a subset: Krippendorf’s � = 0.98. Exact Number Generation: count = � (Random Chance, uniform guess over 1–7: 14.3%). MAE: mean absolute count error. =input: the 3D count equals the count in the input image. Input image: the 2D image that ours and TRELLIS.2 start from. CLIP, ImageReward and VQAScore use the prompt without the count. Bold / underline: best / second-best 3D method. <sup>†</sup>NeRF output. <sup>‡</sup>mesh output.
<table><tr><td>Method</td><td>Exact Number Generation (%)↑</td><td>MAE↓</td><td>=input (%)↑</td><td>CLIP↑</td><td>ImageReward↑</td><td>VQAScore↑</td></tr><tr><td>GaussianDreamer</td><td>23.8</td><td>1.88</td><td>一</td><td>25.37</td><td>-0.26</td><td>0.765</td></tr><tr><td>DiffSplat</td><td>14.3</td><td>2.17</td><td>一</td><td>23.98</td><td>-1.57</td><td>0.483</td></tr><tr><td>TRELLIS-text</td><td>0.0</td><td>3.62</td><td>一</td><td>19.93</td><td>-2.10</td><td>0.128</td></tr><tr><td>MVDream†</td><td>7.1</td><td>3.29</td><td>一</td><td>21.82</td><td>-1.65</td><td>0.336</td></tr><tr><td>TRELLIS.2</td><td>40.5</td><td>1.02</td><td>54.8</td><td>28.35</td><td>-0.02</td><td>0.607</td></tr><tr><td>Ours</td><td>50.0</td><td>0.60</td><td>95.2</td><td>27.58</td><td>-0.13</td><td>0.867</td></tr><tr><td>Input image (2D)</td><td>50.0</td><td>0.60</td><td>一</td><td>29.14</td><td>0.47</td><td>0.913</td></tr></table>

![](images/c6be92bc606183720811150a3be038e60025ebb6b08b0ba91c17efb669b561ad.jpg)  
One asset Ours, TRELLIS.2: 3D input Count = N

Figure 5: Counted fruit against the requested number � for every asset. Each dot is one asset; Rows: tomato and lemon, 21 assets each per method (� = 1–7, seeds 42, 1 and 2). Dots on the dashed line have exactly � fruit; Input image: the count in the input image, which was then used as input to ours and TRELLIS.2. Hollow dots: the 3D asset’s count difers from its input image’s (ours: 2 of 42, both one more; TRELLIS.2: 19 of 42, 18 more and 1 fewer). <sup>†</sup>NeRF output.  
![](images/5684d047f7b7f094d432c8636fb287447711a80e9879ff27bf1a5c841002fed1.jpg)  
Figure 6: Text-to-3D samples for prompted fruit counts: tomato and lemon with � = 3 and � = 6. Every baseline receives “A photo of ⟨species⟩ plants with ⟨�⟩ ⟨fruit⟩. Single potted plant, whole plant.” with the number written as a word; The input image (left) is generated by Qwen-Image-2.1 [50] from the same text, with the final period replaced by the text-to-image sufix (“. . . whole plant and pot fully in frame and centred with margin around it, eye-level side view, isolated on a plain white background, studio lighting.”); ours and TRELLIS.2 both lift this image to 3D. Two views of ours and TRELLIS.2 from opposite sides (elevation 15<sup>◦</sup>, azimuth 30<sup>◦</sup> and 210<sup>◦</sup>), one (30<sup>◦</sup>) of each text-to-3D baseline. <sup>†</sup>NeRF output.

which inherit part of our pipeline, our method beats both methods across a majority of the evaluations conducted. Our strongest results against these methods come from Shape and Perceptual measures, for which we beat SEVA in all tests, and SV3D in all but one. All three video methods use our known poses and therefore skip the lengthy COLMAP process, a stage that can often lead to reconstruction failure. For complex plant scenes it is often challenging to reconstruct camera poses using COLMAP, and the inclusion of our known pose approach greatly strengthens this family of methods.

For text-to-3D generation, our results suggest that the priors of 2D and video models can stand in for scarce 3D training data. The text-to-image model provides control over the species and traits of the plant, such as the number of fruit, and the video model, by orbiting the input image rather than regenerating the plant, carries that control into 3D. The prompt therefore remains an accurate label for the generated plant, as labelled synthetic data for phenotyping requires, and the pipeline should benefit directly from future text-toimage models.

For both reconstruction and generative pipelines we present, our method is especially of interest as it achieves these results without any plant specific training. Many generative [17] models for plants require training from plant specific datasets that are often species and setup dependant, meaning that these models do not generalise well to unseen cases. By contrast, our method performs well across a wide variety of plant species without requiring additional training.

## 6.2. Limitations

The biggest limitation of this work is that in phenotyping pipelines, monocular approaches can only capture details included in the input image, and details of the reverse side, or occluded features must be inferred. For this reason our method is best suited to specific use cases, like height measurement as shown in the paper, and specific plants that do not self occlude significantly.

Another limitation of our method is that the current path predicted captures a single orbit of the plant at a static height. Since Gaussian splatting is trained from this video from the specific orbital perspective, the reconstructions often have artefacts and loss of realism, when viewed from above and below. Though the impact of this limitation will vary in importance depending on the specific downstream phenotyping task being attempted.

## 6.3. Future Work

One of the primary limitations of any monocular approach is that the rear view of the plant will, by the nature of this approach, be invented. As such, a video difusion model that takes an additional image of the reverse view of the plant may be required for further phenotyping tasks, or dense plant structures where additional knowledge of the far side is required. An adapted version of our model that pins an additional view could allow for this, but would need to overcome additional challenges in plant consistency, and the ability to determine the placement of the camera in relation to each other, and track the camera path between them.

Additional work on synthetic data generated this way is also highly promising. Compared to other methods of generating synthetic plant data, for example, procedural generative methods, a crucial advantage of text-to-3D generative models is the ability to prompt for specific phenotypic characteristics easily. Further investigation into the level of accuracy (in the case of geometric measures) or realism when generating physiological phenotypes that can be achieved with our method could strengthen our methods potential for synthetic data generation.

## 6.4. Conclusion

In this paper we have presented Plant Orbiter, to our knowledge the first monocular 3D reconstruction method for plants using video difusion. Our experimental results show that our method competes strongly against existing SotA methods, often exceeding their performance in crucial metrics such as 3D geometry across varied datasets.

Our method generates a short 73 frame video around a target plant, from which we generate a 3D Gaussian splat, using the known orbit from around the plant to predict accurate camera positions and easily align the generated 3D plant. When tested on plants from a wide variety of species, we find we are able to achieve the highest accuracy in IoU and DISTS for most datasets and the best F-scores on all tested datasets. We demonstrate how this method can be used to measure real world plant traits such as plant height and achieve a mean height error of just 5.9% (0.9-2.3cm) across our experiments.

By prepending a text-to-image model, the same pipeline also generates 3D plants from text, inheriting the prompt adherence of 2D models without requiring scarce 3D training data. Among the text-to-3D methods tested, it best matches both the requested species and the shapes of real plants, reproduces a desired number of fruit accurately, and preserves the fruit count of its input image in 95% of assets.

Overall, these results show an accurate and realistic new way to reconstruct digital twins, perform 3D phenotyping, and generate synthetic data using a method not yet applied in this field, while also providing meaningful practical benefits to users.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used Claude (Anthropic) in order to assist with coding, organising and documenting the code repository, and figure improvement. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## Data Availability

All datasets used in this work are from previous publications and are publicly available. All code as well as orbit videos and reconstructions will be available from the corresponding author upon reasonable request.

## Funding

Yong En Kok is supported by the Biotechnology and Biological Sciences Research Council [UKRI3716] and Lewis Stuart is supported by the Engineering and Physical Sciences Research Council [EP/W524402/1].

## CRediT authorship contribution statement

Zane K.J. Hartley Conceptualisation, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Visualisation, Writing - original draft, Writing - review and editing. Yong En Kok Conceptualisation, Methodology, Software, Validation, Formal Analysis, In vestigation, Data Curation, Visualisation, Writing - original draft, Writing - review and editing. Lewis A.G. Stuart Conceptualisation, Writing - review and editing Andrew P. French Supervision, Writing - review and editing Michael P. Pound Supervision, Writing - review and editing

## Declaration of competing interest

The authors of this paper have no competing interests to declare.

## References

[1] A G Stuart, L., Morton, A., Stavness, I., Pound, M.P., 2025. 3dgsto-pc: 3d gaussian splatting to dense point clouds, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pp. 3730–3739.

[2] Achlioptas, P., Diamanti, O., Mitliagkas, I., Guibas, L., 2018. Learning representations and generative models for 3d point clouds, in: International conference on machine learning, PMLR. pp. 40–49.

[3] Adebola, S., Kim, C.M., Kerr, J., Xie, S., Akella, P., Rincon, J.L.S., Solowjow, E., Goldberg, K., 2025a. Botany-bot: Digital twin monitoring of occluded and underleaf plant structures with gaussian splats, in: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE. pp. 1839–1846.

[4] Adebola, S., Xie, S., Kim, C.M., Kerr, J., van Marrewijk, B.M., van Vlaardingen, M., van Daalen, T., van Loo, E., Rincon, J.L.S., Solowjow, E., et al., 2025b. Growsplat: Constructing temporal digital twins of plants with gaussian splats, in: 2025 IEEE 21st International Conference on Automation Science and Engineering (CASE), IEEE. pp. 1766–1773.

[5] Arshad, M.A., Jubery, T., Aful, J., Jignasu, A., Balu, A., Ganapathysubramanian, B., Sarkar, S., Krishnamurthy, A., 2024. Evaluating neural radiance fields for 3d plant geometry reconstruction in field conditions. Plant Phenomics 6, 0235.

[6] Bińkowski, M., Sutherland, D.J., Arbel, M., Gretton, A., 2018. Demystifying mmd gans. arXiv preprint arXiv:1801.01401 .

[7] Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al., 2023. Stable video difusion: Scaling latent video difusion models to large datasets. arXiv preprint arXiv:2311.15127 .

[8] Chen, Z., Wang, Y., Wang, F., Wang, Z., Liu, H., 2024. V3d: Video difusion models are efective 3d generators. arXiv preprint arXiv:2403.06738 .

[9] Choi, H.B., Park, J.K., Park, S.H., Lee, T.S., 2024. Nerf-based 3d reconstruction pipeline for acquisition and analysis of tomato crop morphology. Frontiers in Plant Science 15, 1439086.

[10] Curless, B., Levoy, M., 1996. A volumetric method for building complex models from range images, in: Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, pp. 303–312.

[11] Dawson, P., 2026. MiniMax H3 360 orbit LoRA. Civitai, https:// civitai.com/models/2976490. Version v1 (minimax\_h3\_flf2v\_lora\_v1); released 27 September 2026; accessed October 2026.

[12] Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., Vander-Bilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A., 2023. Objaverse: A universe of annotated 3D objects, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13142–13153. doi:10.1109/CVPR52729.2023.01263.

[13] Ding, K., Ma, K., Wang, S., Simoncelli, E.P., 2020. Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence 44, 2567–2581.

[14] Gao, Z., Hartley, Z.K.J., French, A.P., 2026. Evaluation of one-image 3D reconstruction for plant model generation. Plant Methods 22, 18. doi:10.1186/s13007-025-01482-6.

[15] Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F.A., Brendel, W., 2018. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. arXiv preprint arXiv:1811.12231 .

[16] Go, H., Narnhofer, D., Bhat, G., Truong, P., Tombari, F., Schindler, K., 2025. Vist3a: Text-to-3d by stitching a multi-view reconstruction network to a video generator. arXiv e-prints , arXiv–2510.

[17] Hartley, Z.K., AG Stuart, L., French, A.P., Pound, M.P., 2025. Plantdreamer: Achieving realistic 3d plant models with difusion-guided gaussian splatting, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7128–7138.

[18] Hartley, Z.K., Lind, R.J., Pound, M.P., French, A.P., 2024. Domain targeted synthetic plant style transfer using stable difusion, lora and controlnet, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), IEEE. pp. 5375–5383.

[19] Hartley, Z.K., Lind, R.J., Smith, N., Collison, B., French, A.P., 2023. Unlocking comparative plant scoring with siamese neural networks and pairwise pseudo labelling, in: 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), IEEE. pp. 678–684.

[20] He, H., Xu, Y., Guo, Y., Wetzstein, G., Dai, B., Li, H., Yang, C., 2025a. Cameractrl: Enabling camera control for video difusion models, in: International Conference on Learning Representations, pp. 100433– 100464.

[21] He, J., Tempelaere, A., Vignero, J., Verboven, P., Nicolai, B., 2025b. Multimodal fusion of x-ray transmission and dark field imaging for apple internal disorders detection, in: 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), IEEE. pp. 7229–7238.

[22] Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., Tan, H., 2024. Lrm: Large reconstruction model for single image to 3d, in: International Conference on Learning Representations, pp. 50678–50702.

[23] Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S., 2024a. 2d gaussian splatting for geometrically accurate radiance fields, in: ACM SIGGRAPH 2024 conference papers, pp. 1–11.

[24] Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al., 2024b. Vbench: Comprehensive benchmark suite for video generative models, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE. pp. 21807–21818.

[25] Hunyuan3D, T., Yang, S., Yang, M., Feng, Y., Huang, X., Zhang, S., He, Z., Luo, D., Liu, H., Zhao, Y., et al., 2025. Hunyuan3d 2.1: From images to high-fidelity 3d assets with production-ready pbr material. arXiv preprint arXiv:2506.15442 .

[26] Kajić, I., Wiles, O., Albuquerque, I., Bauer, M., Wang, S., Pont-Tuset, J., Nematzadeh, A., 2024. Evaluating numerical reasoning in text-toimage models. Advances in Neural Information Processing Systems 37, 42211–42224.

[27] Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G., 2023. 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42, 139:1–139:14. doi:10.1145/3592433.

[28] Krippendorf, K., 2018. Content Analysis: An Introduction to Its Methodology. 4 ed., SAGE Publications.

[29] Laurentini, A., 1994. The visual hull concept for silhouette-based image understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence 16, 150–162. doi:10.1109/34.273735.

[30] Lee, J.J., Li, B., Beery, S., Huang, J., Fei, S., Yeh, R.A., Benes, B., 2024. Tree-d fusion: Simulation-ready tree dataset from single images with difusion priors, in: European Conference on Computer Vision (ECCV), pp. 439–460. doi:10.1007/978-3-031-72940-9\_25.

[31] Li, B., Kałużny, J., Klein, J., Michels, D.L., Pałubicki, W., Benes, B., Pirk, S., 2021. Learning to reconstruct botanical trees from single images. ACM Transactions on Graphics (Proc. SIGGRAPH Asia) 40.

[32] Li, J., Qi, X., Nabaei, S.H., Liu, M., Chen, D., Sun, Q., Zhang, X., Yin, X., Li, Z., 2025. A survey on 3d reconstruction techniques in plant phenotyping: From classical methods to neural radiance fields (nerf), 3d gaussian splatting (3dgs), and beyond. Plant Phenomics , 100137.

[33] Li, Y., Liu, Z., Benes, B., Zhang, X., Guo, J., 2024. Svdtree: Semantic voxel difusion for single image tree reconstruction, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4692–4702.

[34] Lin, C., Pan, P., Yang, B., Li, Z., Mu, Y., 2025. Difsplat: Repurposing image difusion models for scalable gaussian splat generation, in: International Conference on Learning Representations, pp. 40745– 40767.

[35] Lin, Y., 2015. Lidar: An important tool for next-generation phenotyping technology of high potential for plant phenomics? Computers and electronics in Agriculture 119, 61–73.

[36] Lin, Z., Pathak, D., Li, B., Li, J., Xia, X., Neubig, G., Zhang, P., Ramanan, D., 2024. Evaluating text-to-visual generation with imageto-text generation, in: European Conference on Computer Vision, Springer. pp. 366–384.

[37] Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C., 2023. Zero-1-to-3: Zero-shot one image to 3d object, in: 2023 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE. pp. 9264–9275.

[38] Liu, Y., Lin, C., Zeng, Z., Long, X., Liu, L., Komura, T., Wang, W., 2024. Syncdreamer: Generating multiview-consistent images from a single-view image, in: International conference on learning representations, pp. 27676–27697.

[39] Liu, Z., Cheng, Z., Yokoya, N., 2025. Neural hierarchical decomposition for single image plant modeling, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 733–742.

[40] Long, X., Guo, Y.C., Lin, C., Liu, Y., Dou, Z., Liu, L., Ma, Y., Zhang, S.H., Habermann, M., Theobalt, C., et al., 2024. Wonder3d: Single image to 3d using cross-domain difusion, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE. pp. 9970–9980.

[41] Ma, J., Hu, X., Shi, L., Zhang, Y., Jiang, Y., Zhang, H., Duan, S., 2026. Plant3r: Fusing 3d feature learning with gaussian splatting to enhance wheat plant 3d reconstruction precision. Plant Phenomics , 100200.

[42] Melas-Kyriazi, L., Laina, I., Rupprecht, C., Neverova, N., Vedaldi, A., Gafni, O., Kokkinos, F., 2024. Im-3d: Iterative multiview difusion and reconstruction for high-quality 3d generation. arXiv preprint arXiv:2402.08682 .

[43] Mertoğlu, K., Şalk, Y., Sarıkaya, S.K., Turgut, K., Evrenesoğlu, Y., Çevikalp, H., Gerek, Ö.N., Dutağacı, H., Rousseau, D., 2024. PLANesT-3D: A new annotated dataset for segmentation of 3D plant point clouds. arXiv preprint arXiv:2407.21150 V2, December 2025.

[44] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R., 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65, 99–106.

[45] MiniMax, 2026. MiniMax H3: A general-purpose omni-modal generative model. https://huggingface.co/MiniMaxAI/MiniMax-H3. Openweight release, 3 August 2026. Accessed: 2026-10-13.

[46] Okura, F., 2022. 3d modeling and reconstruction of plants and trees: A cross-cutting review across computer graphics, vision, and plant phenotyping. Breeding Science 72, 31–47.

[47] Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al., 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 .

[48] Poole, B., Jain, A., Barron, J.T., Mildenhall, B., 2022. Dreamfusion: Text-to-3d using 2d difusion. arXiv preprint arXiv:2209.14988 .

[49] Qin, X., Dai, H., Hu, X., Fan, D.P., Shao, L., Van Gool, L., 2022. Highly accurate dichotomous image segmentation, in: European conference on computer vision, Springer. pp. 38–56.

[50] Qwen Team, 2026. Qwen-image-2.1. https://huggingface.co/Qwen/ Qwen-Image-2.1. GitHub: https://github.com/QwenLM/Qwen-Image-2.1; Blog: https://qwen.ai/blog?id=qwen-image-2.1. Accessed 2026-10- 29.

[51] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al., 2021. Learning transferable visual models from natural language supervision, in: International conference on machine learning, PmLR. pp. 8748–8763.

[52] Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al., 2025. Sam 2: Segment anything in images and videos, in: International Conference on Learning Representations, pp. 28085–28128.

[53] Schonberger, J.L., Frahm, J.M., 2016. Structure-from-motion revisited, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4104–4113.

[54] Shen, P., Jing, X., Deng, W., Jia, H., Wu, T., 2025. Plantgaussian: Exploring 3d gaussian splatting for cross-time, cross-scene, and realistic 3d plant visualization and beyond. The Crop Journal 13, 607–618.

[55] Shi, R., Chen, H., Zhang, Z., Liu, M., Xu, C., Wei, X., Chen, L., Zeng, C., Su, H., 2023. Zero123++: a single image to consistent multi-view difusion base model. arXiv preprint arXiv:2310.15110 .

[56] Shi, Y., Wang, P., Ye, J., Mai, L., Li, K., Yang, X., 2024. Mvdream: Multi-view difusion for 3d generation, in: International conference on learning representations, pp. 39838–39859.

[57] Stein, G., Cresswell, J., Hosseinzadeh, R., Sui, Y., Ross, B., Villecroze, V., Liu, Z., Caterini, A.L., Taylor, E., Loaiza-Ganem, G., 2023. Exposing flaws of generative model evaluation metrics and their unfair treatment of difusion models. Advances in Neural Information Processing Systems 36, 3732–3784.

[58] Stuart, L.A., Wells, D.M., Atkinson, J.A., Castle-Green, S., Walker, J., Pound, M.P., 2025. High-fidelity wheat plant reconstruction using 3d gaussian splatting and neural radiance fields. GigaScience 14, giaf022.

[59] Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y., 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063.

[60] Sun, W., Chen, S., Liu, F., Chen, Z., Duan, Y., Zhu, J., Zhang, J., Wang, Y., 2025. Dimensionx: Create any 3d and 4d scenes from a single image with decoupled video difusion, in: 2025 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE. pp. 13695–13706.

[61] Tancik, M., Weber, E., Ng, E., Li, R., Yi, B., Kerr, J., Wang, T., Kristofersen, A., Austin, J., Salahi, K., Ahuja, A., McAllister, D., Kanazawa, A., 2023. Nerfstudio: A modular framework for neural radiance field development, in: ACM SIGGRAPH 2023 Conference Proceedings. doi:10.1145/3588432.3591516.

[62] Tang, J., Chen, Z., Chen, X., Wang, T., Zeng, G., Liu, Z., 2024. Lgm: Large multi-view gaussian model for high-resolution 3d content creation, in: European Conference on Computer Vision, Springer. pp. 1–18.

[63] Tatarchenko, M., Richter, S.R., Ranftl, R., Li, Z., Koltun, V., Brox, T., 2019. What do single-view 3d reconstruction networks learn?, in: 2019 IEEE/CVF conference on computer vision and pattern recognition (CVPR), IEEE. pp. 3400–3409.

[64] Ubbens, J., Cieslak, M., Prusinkiewicz, P., Stavness, I., 2018. The use of plant models in deep learning: an application to leaf counting in rosette plants. Plant methods 14, 6.

[65] Voleti, V., Yao, C.H., Boss, M., Letts, A., Pankratz, D., Tochilkin, D., Laforte, C., Rombach, R., Jampani, V., 2024. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video difusion, in: European Conference on Computer Vision, Springer. pp. 439–457.

[66] Wang, R., Zha, R., Cheng, Z., Yang, J., Purkait, P., Li, H., 2026. Towards realistic and consistent orbital video generation via 3d foundation priors. arXiv preprint arXiv:2604.12309 .

[67] Wang, Z., Yuan, Z., Wang, X., Li, Y., Chen, T., Xia, M., Luo, P., Shan, Y., 2024. Motionctrl: A unified and flexible motion controller for video generation, in: ACM SIGGRAPH 2024 Conference Papers, pp. 1–11.

[68] Wang, Z., Zhang, H., Bian, L., Zhou, L., Ge, Y., 2025. Accurate plant 3d reconstruction and phenotypic traits extraction via stereo imaging and multi-view point cloud alignment. Frontiers in Plant Science 16, 1642388.

[69] Xiang, J., Chen, X., Xu, S., Wang, R., Lv, Z., Deng, Y., Zhu, H., Dong, Y., Zhao, H., Yuan, N.J., et al., 2026. Native and compact structured latents for 3d generation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14419– 14429.

[70] Xiang, J., Lv, Z., Xu, S., Deng, Y., Wang, R., Zhang, B., Chen, D., Tong, X., Yang, J., 2025. Structured 3d latents for scalable and versatile 3d generation, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE. pp. 21469–21480.

[71] Xu, J., Cheng, W., Gao, Y., Wang, X., Gao, S., Shan, Y., 2024. Instantmesh: Eficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191 .

[72] Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y., 2023. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems 36, 15903–15935.

[73] Yang, C., Li, S., Fang, J., Liang, R., Xie, L., Zhang, X., Shen, W., Tian, Q., 2024a. GaussianObject: High-quality 3D object reconstruction from four views with Gaussian splatting. ACM Transactions on Graphics 43. SIGGRAPH Asia 2024.

[74] Yang, H., Chen, Y., Pan, Y., Yao, T., Chen, Z., Ngo, C.W., Mei, T., 2024b. Hi3d: Pursuing high-resolution image-to-3d generation with video difusion models, in: Proceedings of the 32nd ACM international conference on multimedia, pp. 6870–6879.

[75] Yang, Y., Shinoda, R., Santo, H., Okura, F., 2025. Gaussianplant: Structure-aligned gaussian splatting for 3d reconstruction of plants. arXiv preprint arXiv:2512.14087 .

[76] Ye, V., Li, R., Kerr, J., Turkulainen, M., Yi, B., Pan, Z., Seiskari, O., Ye, J., Hu, J., Tancik, M., Kanazawa, A., 2025. gsplat: An open-source library for Gaussian splatting. Journal of Machine Learning Research 26, 1–17.

[77] Yi, T., Fang, J., Wang, J., Wu, G., Xie, L., Zhang, X., Liu, W., Tian, Q., Wang, X., 2024. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d difusion models, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE. pp. 6796–6807.

[78] Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O., 2018. The unreasonable efectiveness of deep features as a perceptual metric, in: 2018 IEEE/CVF conference on computer vision and pattern recognition, IEEE. pp. 586–595.

[79] Zhao, B., Wu, C., Li, D., Meng, H., Li, J., Zhang, J., Zhou, J., Lin, J., Gao, K., Cao, K., et al., 2026. Qwen-image-2.0 technical report. arXiv preprint arXiv:2605.10730 .

[80] Zhao, Z., Lai, Z., Lin, Q., Zhao, Y., Liu, H., Yang, S., Feng, Y., Yang, M., Zhang, S., Yang, X., et al., 2025. Hunyuan3d 2.0: Scaling difusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202 .

[81] Zhou, J., Gao, H., Voleti, V., Vasishta, A., Yao, C.H., Boss, M., Torr, P., Rupprecht, C., Jampani, V., 2025. Stable virtual camera: Generative view synthesis with difusion models, in: 2025 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE. pp. 12405–12414.

[82] Zuo, Q., Gu, X., Qiu, L., Dong, Y., Yuan, W., Peng, R., Zhu, S., Bo, L., Dong, Z., Huang, Q., et al., 2024. Videomv: Consistent multi-view generation based on large video generative model .