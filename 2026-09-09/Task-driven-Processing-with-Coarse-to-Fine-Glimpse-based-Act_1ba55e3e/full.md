# Task-driven Processing with Coarse-to-Fine Glimpse-based Active Perception

Oleh Kolner<sup>1,2</sup>, Thomas Ortner<sup>1</sup>, Stanisław Woźniak<sup>1</sup>, and Angeliki Pantazi<sup>1</sup>

<sup>1</sup> IBM Research, Zurich, Switzerland <sup>2</sup> Graz University of Technology, Graz, Austria olk@zurich.ibm.com

Abstract. State-of-the-art vision models process images in their entirety, lacking the ability to selectively zoom in on relevant regions. This limitation is particularly acute in scenarios where processing must be conditioned on a specific task – such as instance detection, which requires localizing a specific object in a high-resolution, cluttered scene. In such settings, critical details are easily lost as images are often resized to match the model dimensions and computational constraints. We introduce Coarse-to-Fine Glimpse-based Active Perception (CF-GAP), a task-driven front-end that enhances high-resolution processing of existing instance detectors. CF-GAP selectively directs a sequence of limitedview glimpses across the scene, utilizing task information to iteratively refine focus on the most relevant regions. These localized regions are then processed at high resolution by a downstream instance detector. By avoiding full-image processing and eliminating irrelevant confounding information, CF-GAP improves Average Precision (AP) by up to 20% across various state-of-the-art instance detectors on the HR-InsDet and Robotools benchmarks, while further enabling lightweight detectors to outperform their larger counterparts.

Keywords: Active perception · Bio-inspired vision · Instance detection

## 1 Introduction

State-of-the-art vision models process entire images uniformly, lacking the ability to selectively zoom into task-relevant regions for detailed analysis. Moreover, they require images to be resized to the fixed input dimensions used during pretraining, often obscuring small but important regions. These limitations hinder performance on high-resolution images where task-relevant content is small or cluttered [37, 43, 50]. A further shortcoming is that task-specific information (e.g., a textual prompt or a specific object to be localized) is typically incorporated only after a costly feature extraction stage. However, early integration of such information could significantly reduce the image area requiring intensive computation and mitigate the influence of irrelevant or distracting features.

Human vision, by contrast, is inherently task-driven: we actively seek visual information based on what we aim to accomplish, rather than passively processing every detail [11, 45]. For example, in an ofice environment shown in Fig. 1, one often needs to find a specific box or tool – not just any object from those categories. This targeted search corresponds to instance detection, where the goal is to localize a specific object instance given a few visual examples, as opposed to classical object detection, which seeks to identify all objects in a scene. As opposed to typical vision models, humans actively leverage instancespecific features as task cues for eficient visual search [41]. A further hallmark of human vision is the foveal structure of the eye, which provides non-uniform resolution – highest at the center and decreasing toward the periphery – balancing detailed information with a broad field of view [10,36]. Humans exploit this balance through eye movements (saccades) that follow a coarse-to-fine strategy: macrosaccades, guided by low-resolution cues, direct gaze to promising areas, which are then analyzed in detail through microsaccades [28, 34, 46].

![](images/53a6308d6446e0b6cbf53a75125d932915cc334128328ad40cccb9e51efdbd39.jpg)  
Fig. 1: Coarse-to-Fine Glimpse-based Active Perception (CF-GAP) iteratively directs a series of glimpses across a high-resolution scene, using task-driven search maps to progressively narrow focus onto the likely search target. The resulting regions of interest are passed at high resolution to the downstream architecture for the final detection.

Inspired by saccadic processing, the recently proposed Glimpse-based Active Perception (GAP) model [20] fixates on salient – not necessarily task-relevant – image regions to solve synthetic visual reasoning tasks, demonstrating strong out-of-distribution generalization and, thus, the efectiveness of processing only selected image parts. Building on this, we propose Coarse-to-Fine GAP (CF-GAP), which replaces uniform image processing by selectively directing a sequence of limited-view glimpses (Fig. 1). This sequence is generated through a nested process of coarse and fine glimpsing, guided by a task-specific object to be localized, referred to as a search target. Coarse glimpsing relies on correlations between the search target’s features and the downscaled scene to identify an initial coarse glimpse location. There, a fovea-inspired log-polar sensor extracts a limited-view glimpse from the full-resolution scene, magnifying the visual details around the glimpse location while preserving peripheral information at diminishing resolution. Operating on this focused view, fine glimpsing directs the log-polar sensor in a closed loop to iteratively refine the initial location toward a likely search target. The resulting fine glimpse locations define a localized region of interest (RoI) that is passed to a downstream architecture at high-resolution to determine whether the search target is present at those locations.

Our work focuses on the instance detection problem, where visual processing must be conditioned on a specific search target. We design CF-GAP as a frontend module that integrates seamlessly with existing instance detectors as downstream architectures, providing them with targeted, high-resolution input. We combine CF-GAP with several state-of-the-art detectors and evaluate on HR-InsDet [37] and Robotools [21] as two challenging benchmarks. We observe significant performance improvement across all models. Notably, integrating CF-GAP with small models optimized for edge devices allowed them to match, and in some dificult cases, even surpass the performance of their larger versions.

To summarize, our contributions are the following:

1. We introduce a bio-inspired, coarse-to-fine processing scheme to selectively explore high-resolution scenes, replacing exhaustive full-scene processing.

2. We design our model as a front-end module that can be seamlessly integrated with any existing instance detector, enhancing it with targeted, highresolution input.

3. We show that our approach significantly boosts the performance of state-ofthe-art instance detectors, particularly in cluttered and complex scenes.

## 2 Related Work

Early instance detection methods trained CNN-based object detectors with a separate class per object instance, using the cut-paste-learn (CPL) framework [6], where objects are pasted into random backgrounds. State-of-the-art instance detection methods [24,37,38] adopt a two-stage pipeline. First, all candidate objects in the scene are detected using foundation models pre-trained on large-scale data, in particular Segment Anything Model (SAM) [19,32] and GroundingDINO [23]. Second, the candidates are matched against visual examples of the search target using DINOv2 [29] features, with the best match returned as the final prediction. Recent extensions improve the matching stage by fine-tuning DINOv2 [38] or training a weight adapter [24] on specific object instances. Critically, all the aforementioned methods are from the ground up not task-driven, since the first stage of proposal detection involves exhaustive analysis of the entire image to detect all possible objects irrespective of the search target. Our approach leverages the information about the task of detecting a specific search target to steer detection toward only promising regions.

Multiple approaches drew high-level inspiration from saccadic processing and showed compelling results across various tasks, including image classification [8, 27, 44], object detection [2, 13, 14], visual exploration [31], and visual reasoning [20, 42]. However, none of them allows for conditioning by task information (e.g. by a search target) at inference to guide the image exploration. While our approach is built upon GAP [20], the original framework lacks a coarse-to-fine, task-driven search strategy.

Fovea-inspired vision with non-uniform resolution was explored for guiding the visual search only in simple images [1, 3]. Various methods explored fovealike log-polar imagery and its rotational and scaling equivariance properties for image classification, object detection, and image correspondence [7, 9, 18]. Another approach [35] proposed a sophisticated tokenization for vision transformers (ViTs) [5], splitting the image into patches of various sizes resembling foveated image structure. While prior work used fovea-inspired imagery mainly to encode a fixed view, we use it instead to navigate the coarse-to-fine glimpsing process.

## 3 Method

Conceptually, CF-GAP can be viewed as an active process of steering a virtual log-polar sensor across a scene to acquire high-quality information related to the search target (Fig. 1). More specifically, CF-GAP directs a sequence of glimpses – limited views of an image taken at specific locations – to pinpoint a likely search target location. This location defines the center of a RoI that is passed to the downstream architecture, allowing it to isolate and verify the candidate object, without exhaustively processing the entire scene. CF-GAP operates in two nested stages of coarse and fine glimpsing guided by distinct search maps. The search maps are 2D heatmaps that highlight regions with a high probability of containing the search target. The search map extraction follows a common scheme from [47, 51]: a scene encoder and a search target encoder produce features that are compared via convolution (Fig. 2A). The coarse and fine glimpsing stages difer in how this scheme is instantiated, as described below, with further technical details provided in Appendix A.

## 3.1 Coarse glimpsing

CF-GAP begins with computing a coarse search map from a downscaled version of the scene and sample images of the search target taken from diferent viewpoints (referred to as search target examples). The scene and search target encoders (Fig. 2A) are instantiated with the lightweight MobileNet-V3 [12]. The features of the search target examples are averaged across spatial dimensions into a single feature vector, which is then convolved over the scene features, producing the coarse search map. This map drives the iterative selection of coarse glimpse locations: at each iteration, the location of the highest value is selected via the winner-takes-all (WTA) strategy, followed by inhibition-of-return (IoR) that masks its surrounding region to prevent repeated selection, similar to [15, 20].

Since coarse glimpse locations are derived from low-resolution imagery, they represent only rough estimates of the search target’s position. Therefore, each coarse glimpse location initiates a closer inspection with fine glimpsing.

## 3.2 Fine glimpsing

Unlike coarse glimpsing, which operates on a static downscaled scene, fine glimpsing operates on glimpses of both the scene and the search target. It is a closedloop process: at each iteration, a fine search map is generated from the current scene glimpse, and its 2D centroid determines the next fine glimpse location. The log-polar sensor then moves to that location and extracts a new scene glimpse, from which the next iteration proceeds. The sensor employs a fovea-like logpolar transformation to extract high-resolution detail near the glimpse location while preserving distant context at progressively lower resolution. The resulting focused view makes fine search map generation robust against peripheral distractors, and we analyze its advantages over cartesian cropping in Sec. 5.

![](images/274f92592d4477affa6499fe298d761e37bd5a166a0dbc13e0020d2156f9660d.jpg)

![](images/fb7a7069b8c4da7406e86d57d995fe13c4d405740188e7b2dee274b8e9bd5e76.jpg)  
Fig. 2: (A) General architecture for extracting coarse and fine search maps. (B) Scene encoder compresses the scene glimpse into a compact set of latent embeddings via crossattention, where a small number of learnable embeddings attend to the full spatial input $( N \ll H \times W )$ . The compressed representation is then processed through selfattention. A second cross-attention decodes the resulting representation back to the original spatial dimensions, where a distinct set of learnable value embeddings defines the output feature space. $\mathrm { Q } , \mathrm { K } ,$ , and V denote queries, keys, and values of each attention block. (C) Search target encoder shares the cross- and self-attention blocks with the scene encoder (marked by the same color) and iteratively attends to multiple search target glimpses to produce compact target features to be compared with the glimpse feature map.

Since the fine search map extraction (Fig. 2A) operates on log-polar glimpses, using CNN-based scene and search target encoders (as for the coarse search map) becomes problematic. Specifically, CNNs assume that the input image has uniform resolution, treating all regions equally regardless of their position within that image. By contrast, log-polar glimpses preserve fine detail near the glimpse location while compressing the periphery into much fewer pixels. We therefore propose modules based on Perceiver [16], illustrated in Fig. 2B-C.

Each glimpse is partitioned into non-overlapping patches projected into a D-dimensional space, $\pmb { F } _ { s } \in \mathbb { R } ^ { H \times W \times D }$ and $\{ F _ { m } \} _ { m = 1 } ^ { \bar { M } }$ denote the corresponding representations of the scene glimpse and M search target glimpses, respectively. The scene encoder (Fig. 2B) first compresses the scene glimpse into a compact set of N latent embeddings via cross-attention, where a small number of learnable queries attend to the full spatial input, and then processes them through selfattention:

$$
\tilde { h } _ { s } = \mathscr { C } ( \mathsf { Q } = e _ { q } , \mathsf { K V } = F _ { s } ) , \qquad h _ { s } = \mathscr { S } \Big ( \mathsf { Q } \mathsf { K V } = \tilde { h } _ { s } \Big ) ,\tag{1}
$$

where $\boldsymbol { e } _ { q } \in \mathbb { R } ^ { N \times D }$ is a set of learnable embeddings and $N \ll H \times W$ . The compression via cross-attention alleviates the quadratic cost of the subsequent self-attention. The self-attention, in turn, integrates fine local details near the glimpse center with the broader peripheral context. Finally, the latent embeddings are decoded back to the original spatial dimensions through a second cross-attention, producing the glimpse feature map $\boldsymbol { F } _ { s } ^ { * }$ as the final output:

$$
\pmb { F } _ { s } ^ { * } = \mathcal { C } _ { \mathrm { d e c } } ( \mathsf { Q } = \pmb { F } _ { s } , \mathsf { K } = \pmb { h } _ { s } , \mathsf { V } = \pmb { e } _ { o } ) .\tag{2}
$$

The decoding cross-attention uses a distinct set of learnable embeddings $e _ { o } \in$ $\mathbb { R } ^ { N \times D }$ as values to decouple the feature space used for latent compression from the one used for the output feature map. This is intended to let the output feature map be optimized specifically for comparison with features extracted from the search target.

The search target encoder (Fig. 2C) follows the same compress-and-process pattern with shared cross- and self-attention blocks, but instead of decoding back to spatial dimensions, it iteratively attends to M search target glimpses to produce a set of compact target features. At each iteration m, the cross-attention receives the search target glimpse ${ \pmb F } _ { m }$ as keys and values and the previous output as queries, followed by self-attention:

$$
\tilde { h } _ { T } ^ { ( m ) } = \mathscr { C } \Bigl ( \mathsf { Q } = h _ { T } ^ { ( m - 1 ) } , \mathsf { K V } = \pmb { F } _ { m } \Bigr ) , \qquad h _ { T } ^ { ( m ) } = \mathscr { S } \Bigl ( \mathsf { Q } \mathsf { K V } = \tilde { h } _ { T } ^ { ( m ) } \Bigr ) ,\tag{3}
$$

with the query at the first iteration initialized to $h _ { T } ^ { ( 0 ) } = e _ { q }$ and the final output $\pmb { h } _ { T } = \pmb { h } _ { T } ^ { ( M ) } , \pmb { h } _ { T } \in \mathbb { R } ^ { N \times D }$ used as the set of target features. These target features are then correlated with the glimpse feature map via convolution, yielding N correlation maps that are averaged into a single fine search map.

## 3.3 Downstream architecture

At the end of each fine-glimpsing loop, CF-GAP provides the downstream architecture with three inputs. First, a fixed-size RoI cropped around the last fine glimpse location. Second, the fine glimpse location itself, which, depending on the downstream architecture, serves either as a spatial prompt (e.g. for SAM-like models) or as a bounding box filter to constrain object detection. Third, multiple visual examples of the search target from diferent viewpoints, used for matching with the detected candidate object. After a predefined number of coarse and fine glimpsing iterations, the best-matched candidate object is returned. Importantly, CF-GAP is agnostic to the choice of downstream architecture. During evaluation, we employ several state-of-the-art detectors as described in Sec. 4.

## 4 Experiments

Instance detection consists of individual tasks, each defined by a few visual examples of a specific object to be localized in an input scene. Unlike classical object detection, which identifies all instances of an object category, instance detection is conditioned on a particular object instance. We consider two benchmarking datasets. The first, HR-InsDet [37], contains 100 object instances with 24 visual examples each, and 160 high-resolution scenes spanning 14 indoor scenarios. For training, the dataset provides 200 images with random backgrounds to synthesize training data via the cut-paste-learn strategy [6], where search targets are resized and pasted onto arbitrary backgrounds. The dataset is split into subsets by the level of clutter and occlusion – easy and hard – and by object size – small, medium, and large. We report results for each subset following the HR-InsDet evaluation protocol. The second benchmark, Robotools [21], contains 20 object instances and 1581 test images from 24 indoor scenarios. Unlike HR-InsDet, Robotools prohibits using its 20 search targets for training, thereby evaluating generalization to novel objects. Accordingly, we train CF-GAP using only objects from HR-InsDet. We report average precision (AP) at Intersectionover-Union (IoU) thresholds from 0.5 to 0.95 in steps of 0.05, as well as $\mathrm { A P _ { 5 0 } }$ at an IoU threshold of 0.5.

Baselines. The strongest baselines, OTS-FM [37], IDOW [38], and NIDS-Net [24], employ pre-trained foundation models to process the entire scene, first detecting bounding boxes for all object-like regions (proposals). A feature extractor then generates embeddings for each proposal, which are matched to the search target’s examples via Stable Matching [25]. All methods use either SAM [19] or GroundingDINO [23] for proposals and DINOv2 [29] for feature extraction. IDOW extends OTS-FM by fine-tuning DINOv2 on search targets from HR-InsDet. However, we exclude IDOW from our evaluations as its finetuned weights are unavailable, precluding integration with our CF-GAP frontend. We do include NIDS-Net, which follows a similar but higher-performing approach: it trains a weight adapter for DINOv2 and additionally uses SAM to mask out backgrounds within each proposal. The core baseline set comprises $\mathrm { O T S - F M _ { S A M } }$ and $\mathrm { O T S - F M _ { G r o u n d i n g D I N O } }$ , depending on proposal detector, and NIDS-Net, which uses GroundingDINO for proposal detection. We additionally consider two eficient SAM variants as OTS-FM backbones: MobileSAM [48], a distilled version of SAM, and Segment This Thing (STT) [35], which uses a fovea-inspired tokenization that partitions the image into patches of increasing size with distance from a given location. These are denoted $\mathrm { O T S - F M _ { M o b i l e S A M } }$ and $\mathrm { O T S - F M _ { S T T } }$ . Since STT requires a location input for the tokenization, it can only be evaluated in combination with $\mathrm { C F { - } G A P }$ and is thus excluded from the main pairwise comparisons between standalone baselines and their CF-GAP extensions. Unless stated otherwise, all baseline results are reproduced using the publicly available code.

Setup. We integrate CF-GAP with each baseline as its downstream architecture. For the $\mathrm { O T S - F M } _ { \mathrm { M o b i l e S A M } }$ and $\mathrm { O T S - F M _ { S A M } }$ baselines, CF-GAP changes the input structure: standalone baselines receive the full high-resolution scene together with a coarse grid of 2D point prompts to detect proposals at each grid location. By contrast, CF-GAP provides only a small RoI along with fine glimpse locations as 2D point prompts at the end of each fine glimpsing loop, improving both eficiency and focus. As the $\mathrm { O T S - F M } _ { \mathrm { G r o u n d i n g D I N O } }$ baseline does not support point-based prompting, the fine glimpse locations are used to filter out detected bounding boxes that do not contain them.

In HR-InsDet and Robotools, scenes are sized at 6144×8192 and 1920×1080 pixels, respectively, whereas baseline models require resizing below $2 0 4 8 \times 2 0 4 8$ CF-GAP supports flexible input sizes; for faster experimentation, HR-InsDet scenes were resized to $4 0 9 6 \times 5 4 6 0$ while Robotools scenes were kept at original resolution. Coarse glimpsing operates on scenes downscaled by a factor of 2; fine glimpsing operates at full resolution. The log-polar sensor diameter is set to 4096 pixels, and the resulting glimpses are resized to $2 4 5 \times 2 4 5$ . Each scene undergoes $N _ { c }$ coarse glimpses, each followed by $N _ { f }$ fine glimpses, with $N _ { c } { = } 3 0$ and $N _ { f } { = } 3$ by default unless stated otherwise. Further details are provided in Appendix B.

## 5 Results

## 5.1 Benchmarking results

As shown in Figs. 3 and 4, CF-GAP consistently improves all baseline models, demonstrating the efectiveness of the coarse-to-fine glimpsing. In evaluations on HR-InsDet dataset, the biggest benefits are observed in scenes with small-sized objects and hard scenes, i.e. scenes with high clutter and partial occlusions. OTS-$\mathrm { F M _ { M o b i l e S A M } }$ , the smallest baseline, benefits the most, achieving up to 20% AP improvement on hard scenes. With CF-GAP, it surpasses both $\mathrm { O T S - F M _ { S A M } }$ and $\mathrm { O T S - F M _ { G r o u n d i n g D I N O } }$ on the dificult subsets while performing comparably on simpler ones. On Robotools, CF-GAP also consistently improves AP of all baselines. Lightweight OTS-FM<sub>MobileSAM</sub> paired with CF-GAP achieves $\mathrm { A P _ { 5 0 } }$ comparable even with the heavier baselines; its AP, however, stays below theirs, indicating less precise bounding boxes due to its weaker detector backbone. We provide more extensive tabular comparisons in Appendix C.

![](images/13dfa7d96ecfcee63bc0db59bdfcbc3212a4df80e049e860073ec2a30ec4d621.jpg)  
Fig. 3: Performance on HR-InsDet dataset across two dataset groupings – by search target sizes, and by scene types.

![](images/87991ab198c0e2cb632f82cdb6c5fb72cc739e0f02f735193ddde910f4e1c179.jpg)  
Fig. 4: Performance on Robotools dataset, testing generalization to novel objects unseen during training. Baseline results in dashed bars are taken from [24, 38]

## 5.2 Analysis and Ablations

Naive high-resolution baseline. To demonstrate the importance of CF-GAP, we compare it against a straightforward alternative for processing high-resolution images. Baseline models cannot handle scenes at their original resolution because their ViT-based backbones decompose images into a fixed number of patches to limit the quadratic cost of self-attention. A naive solution is to split each scene into smaller overlapping patches and let the baseline treat each patch as a separate RoI. Tab. 1 reports results for this approach, where the full-sized (6144×8192) scenes are split into 1024×1024 patches with 50% overlap, yielding 165 RoIs per scene – over 5× more than the 30 RoIs produced by CF-GAP. OTS-FM models without fine-tuned matching do not benefit from such patching, as the larger number of candidate proposals across all RoIs leads to increased matching errors. NIDS-Net, whose fine-tuned features better discriminate the search target, does benefit – particularly for small objects – but still falls behind its CF-GAP extension, especially in hard, cluttered scenes. These results confirm that while exhaustive patching can in principle recover lost high-resolution detail, CF-GAP is both more efective – achieving higher AP, and more eficient – passing over 5× fewer RoIs to the downstream architecture.

Table 1: Comparison to patched baselines that split the scene into smaller, separately processed patches. Results on the two most challenging HR-InsDet subsets.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>APsmall hard</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { \overline { { O T S - F M _ { M o b i l e S A M } } } }$ Patched OTS-FMMobileSAM $\mathrm { \underline { { C F \mathrm { - } G A P \mathrm { ~ + ~ { O T S \mathrm { - } F M } _ { M o b i l e S A M } } } } }$ </td><td rowspan=1 colspan=1>12.4 22.018.4 23.229.3 41.7</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { O T S - F M } _ { \mathrm { S A M } } } }$  $\mathrm { P a t c h e d ~ O T S  – F M _ { S A M } }$  $\mathrm { C F - G A P + O T S \mathrm { - } F M _ { S A M } }$ </td><td rowspan=1 colspan=1>14.628.018.924.232.4 47.1</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { O T S - F M } _ { \mathrm { G r o u n d i n g D I N O } } } }$  $\mathrm { P a t c h e d ~ O T S  – F M _ { G r o u n d i n g D I N O } }$  $\mathrm { C F - G A P } + \mathrm { O T S - F M } _ { \mathrm { G r o u n d i n g D I N O } }$ </td><td rowspan=1 colspan=1>28.8 37.222.4 28.139.650.2</td></tr><tr><td rowspan=1 colspan=1>NIDS-NetPatched NIDS-Net $\mathrm { C F - G A P + N I D S  – N e t }$ </td><td rowspan=1 colspan=1>32.4 39.950.1 49.152.1 63.2</td></tr></table>

Glimpse locations. In addition to providing targeted RoIs to downstream architectures, CF-GAP also provides fine glimpse locations that specify where objects have to be detected within each RoI. For SAM-based models, this improves the eficiency by replacing the dense grid of point prompts with only a few glimpse locations. For GroundingDINO-based models, the fine glimpse locations act as spatial filters that exclude candidate objects that do not overlap with them, substantially reducing the number of candidates and thereby improving matching efectiveness. This is confirmed in Tab. 2, which reports higher performance when the downstream architecture receives both the RoIs and the glimpse locations, compared to receiving only the RoIs.

Table 2: Ablation of inputs provided to the downstream architecture. Results are shown for CF-GAP+NIDS-Net evaluated on HR-InsDet.
<table><tr><td>Downstream architecture input</td><td colspan="4">AP</td></tr><tr><td rowspan="2">w/o locations with locations</td><td>small medium large easy</td><td></td><td></td><td>hard</td></tr><tr><td>41.8 52.1</td><td>73.1 79.7</td><td>86.0 74.5 86.1 78.2</td><td>49.5 63.2</td></tr></table>

Fine glimpsing. Given that CF-GAP consists of coarse and fine glimpsing processes, it is important to demonstrate the need for the latter. Tab. 3 shows that fine glimpsing is most beneficial in hard scenes and scenes with small-sized objects. This is because the low resolution of the coarse search map can yield glimpse locations that are ofset from the actual object, disrupting downstream architectures that are prompted to detect objects at specific locations. Fine glimpsing corrects for this spatial error (see Fig. 7 for examples). In simpler cases where objects are easy to find, fine glimpsing provides marginal benefit.

Table 3: Ablation of fine glimpsing. Results are shown for CF-GAP+NIDS-Net evaluated on HR-InsDet.
<table><tr><td>Fine glimpsing</td><td colspan="4">AP</td></tr><tr><td></td><td>small medium large easy</td><td></td><td></td><td>hard</td></tr><tr><td>no</td><td>44.6 77.3</td><td></td><td>85.5 75.6</td><td>56.2</td></tr><tr><td>yes</td><td>52.1 79.7</td><td></td><td>86.1 78.2</td><td>63.2</td></tr></table>

Log-polar glimpses. Another study shows the advantage of extracting logpolar glimpses. Compared to a naive approach of extracting a small cartesian crop, the log-polar representation eliminates the need to tune the crop size, which would otherwise be highly sensitive to the proximity of the initial coarse glimpse location to the object and to the object’s size. For example, the crop size can be either too small, providing too little information, or too big, providing too much distraction (Fig. 5, top row). By contrast, due to logarithmically diminishing resolution at the periphery, the log-polar glimpses consistently maintain the focus on nearby regions regardless of the area captured in the full-resolution scene. This is visually apparent in the bottom row of Fig. 5, where log-polar glimpses remain visually similar regardless of their size (diameter), as opposed to crop-based glimpses in the top row. As a result, the fine search map extraction becomes more robust against the distracting peripheral information as indicated by the resulting fine glimpse locations in Fig. 5. Fig. 6 confirms this quantitatively, showing more stable performance over a broader range of glimpse sizes for log-polar compared to cropped-based glimpses. In addition, in Appendix D, we empirically justify our architecture for processing log-polar glimpses (Fig. 2B-C) by comparing it to a CNN-based model, showing that the latter is less efective.

![](images/dd511e2e9708622d55c1e605dfb320478c1e7d33b256f64c9c83566727403fc2.jpg)  
Fig. 5: Qualitative comparison between crop-based and log-polar glimpses across different sizes. Red dots mark the initial coarse glimpse location, ofset from the search target, and green triangles mark the subsequent fine glimpse location based on fine search maps extracted from each glimpse. Unlike crop-based glimpses (top row), which change drastically with size, log-polar glimpses (bottom row) remain visually stable across diferent diameters, as peripheral diferences are compressed into the far-right part of each log-polar image. The middle row shows the regions covered by the log polar glimpses.

![](images/a9c59049fe9944144ae94ab7d8703a76a93211e3dff8ccabc4a84d45fb31d622.jpg)  
Fig. 6: Performance comparison for diferent glimpse types and their sizes. Results are for CF-GAP+NIDS-Net evaluated on the two most challenging HR-InsDet subsets.

Factoring out the downstream architecture. Instance detection can be decomposed into two stages: 1) finding a candidate region likely to contain the search target, and 2) matching it against the search target’s examples for recognition. Since CF-GAP primarily improves the first stage, we measure how often the search target is found in the scene regardless of whether it is successfully recognized during matching. We note that we cannot report more standard average recall metrics, as CF-GAP does not directly detect bounding boxes. Focusing on the most challenging HR-InsDet subsets, Tab. 4 shows that CF-GAP locates small objects and objects in hard scenes more frequently than standalone baselines. These results also reflect the upper-bound AP achievable under a perfect matching stage. The last row of Tab. 4 further highlights the importance of fine glimpsing for precise localization.

Table 4: Percentage of scenes in the two most challenging HR-InsDet subsets, where the search target was found, but not necessarily correctly recognized.
<table><tr><td>Model</td><td>|Target found, % small hard</td></tr><tr><td>OTS-FMSAM</td><td>42.1 56.3</td></tr><tr><td>NIDS-Net</td><td>73.3 75.2</td></tr><tr><td>CF-GAP</td><td>81.9 89.6</td></tr><tr><td>CF-GAP w/o fine glimpsing</td><td>69.5 74.6</td></tr></table>

![](images/f9b5fedfc98804287a35088833c0818a74917d0f38665e5aa1926a94679b7073.jpg)  
Fig. 7: Coarse and fine glimpse locations for two scenes where baseline models failed to detect the search target (left: HR-InsDet, right: Robotools). Glimpse locations concentrate on a small fraction of the scene rather than spreading uniformly, and fine glimpse locations visibly correct the spatial imprecision of coarse ones.

Qualitative inspection. We provide visualizations of coarse and fine glimpse locations for a couple of scenes in Fig. 7. Note that the glimpses are not spread out across entire scenes meaning that only a subset of the entire scene will be passed to downstream architectures. This, in turn, implies the reduction of irrelevant and potentially distracting information. The visualizations in zoom-in panels also illustrate how fine glimpsing corrects for the spatial imprecision of coarse glimpsing. More examples are provided in Appendix E. Failed cases are shown and discussed in Appendix F.

## 5.3 Computational Cost

While CF-GAP significantly improves performance of the baselines, it incurs additional computational cost of repeatedly running the downstream architecture after each coarse glimpse. To fully leverage the power of CF-GAP, one has to select the downstream architecture wisely. In particular, it is costly to use a heavy, ineficient model designed and trained to handle complex scenery with numerous objects and various visual intricacies. In fact, this is unnecessary since CF-GAP provides targeted, high-quality information stripped of irrelevant details. Hence, a less powerful but more eficient downstream architecture can sufice. To illustrate this, we compare SAM with its two more eficient versions – MobileSAM and STT – as OTS-FM’s backbones combined with CF-GAP. The comparison is made both in terms of performance and eficiency, with the latter being represented via end-to-end FLOPS for all models. As can be seen in Fig. 8, CF-GAP makes the more eficient versions surpass the original $\mathrm { O T S - F M _ { S A M } }$ model in terms of both eficiency and performance. Although, to the best of our knowledge, there are no eficient versions currently available for GroundingDINO, we expect to observe similar trends to those of SAM and its eficient alternatives.

![](images/ae480c9fc18e9b9f503c9c280d9a71134c99951039a0700e85ba295f83a17ab5.jpg)  
Fig. 8: CF-GAP with eficient $\mathrm { O T S - F M _ { S A M } }$ variants evaluated on the two most challenging HR-InsDet subsets. Each dot corresponds to a specific number of coarse glimpses (1 to 30). Other baselines are omitted as no eficient variants are available.

More generally, with each coarse glimpse invoking the downstream archi tecture, CF-GAP directly trades eficiency for performance. Fig. 9 traces this trade-of along the number of coarse glimpses $N _ { c }$ . The comparison between the standalone baseline and its CF-GAP extension at the matched compute budget corresponds to the case of using a single coarse glimpse. In addition, we break

![](images/052906abd0d0c5642352e8d8e4b9d6c6474d3079d17f17fc439e7f7f945a5572.jpg)

![](images/80be2111ca13cc48c16d485c54895922e80f5371d4ec8c86af9650f2560b8e8f.jpg)  
Cumulative % of hitting the target AP, %

Fig. 9: Impact of the number of coarse glimpses on performance. Results are shown for CF-GAP+NIDS-Net evaluated on the two most challenging HR-InsDet subsets. The stars mark the performance of standalone NIDS-Net.

Table 5: Cost breakdown for CF-GAP paired with downstream architectures. $N _ { c }$ and $N _ { f }$ correspond to the number of coarse and fine glimpses, respectively.
<table><tr><td>Component</td><td>GFLOPs</td></tr><tr><td>Coarse search map, SMc (MobileNet)</td><td>60 (per scene)</td></tr><tr><td>Fine search map, SMf (Perceiver-based encoders)</td><td>1 (per glimpse)</td></tr><tr><td>CF-GAP internal cost:  $\mathrm { S M } _ { c } + N _ { c } { \times } N _ { f } { \times } \mathrm { S M } _ { f }$ </td><td> $6 0 + N _ { c } { \times } N _ { f } { \times } 1$ </td></tr><tr><td>Downstream architecture (DA)</td><td> $\left\{ 8 0 \mid 1 3 0 \mid 5 8 0 0 \mid 8 0 0 \right\}$ </td></tr><tr><td> $\{ \mathrm { O T S } _ { \mathrm { S T T } } \mid \mathrm { O T S } _ { \mathrm { M o b i l e S A M } } \mid \mathrm { O T S } _ { \mathrm { S A M } } \mid \mathrm { N I D S } \mathrm { - N e t } \}$  Total cost: [CF-GAP internal  $\mathrm { c o s t } ] + N _ { c } \times \mathrm { [ D A ~ c o s t ] }$ </td><td></td></tr></table>

down the compute cost in Tab. 5, showing that the total cost is dominated by $N _ { c }$ invocations of the downstream architecture. We report the cost in FLOPS, since wall-clock runtime depends on implementation-specific optimizations beyond the scope of this work. We also note that two further costs, shared by all downstream architectures, are omitted from the table: encoding the search target examples and the candidate objects with DINOv2, both of which vary across datasets and scenes.

Lastly, Fig. 9 shows the cumulative percentage of hitting the search target (in gray): in ∼50% of scenes, CF-GAP finds search targets within the first 8- 10 glimpses. This hints that the computational cost can be reduced, given a more robust matching stage that could terminate the glimpsing process once the search target is recognized.

## 6 Discussion

Our results demonstrate that CF-GAP consistently boosts existing instance detectors, with the largest gains in the most challenging settings of small-sized objects and cluttered scenes. Combining CF-GAP with instance detectors as downstream architectures exhibits a functional dichotomy of looking and seeing: CF-GAP looks for task-relevant regions and directs the downstream architecture as a seeing component to analyze them in high resolution. This division of labor, in turn, allows for employing lighter, distilled models such as OTS-FM<sub>MobileSAM</sub> as a downstream architecture, achieving competitive performance compared to larger models. Moreover, one can also use large downstream architectures such as $\mathrm { O T S - F M _ { S T T } }$ with advanced fovea-inspired tokenization techniques that allow to retain their expressivity while making them very eficient. Hence, the combination of our lightweight CF-GAP with such downstream architectures paves the way to eficient yet powerful task-driven models.

Limitations and future work. In its current form, CF-GAP relies solely on texture-based guidance. However, the human visual system is known to leverage high-level semantics about objects, spatial layouts of scenes, and many other features when searching for task-relevant information. In addition, the inhibitionof-return mechanism only suppresses previously visited locations and their immediate neighborhoods, rather than entire task-irrelevant regions. This can cause the glimpsing process to repeatedly revisit the same distractor object. Another limitation is that CF-GAP passes a fixed-size crop as RoI to the downstream architecture, requiring the crop to be conservatively large to accommodate objects of varying sizes. An adaptive mechanism that adjusts the crop based on the content of the task-relevant region would improve both eficiency and precision. Finally, invoking the downstream architecture after every coarse glimpse incurs a computational cost that grows linearly with the number of coarse glimpses. A more robust matching stage that halts the glimpsing process once the search target is confidently recognized would alleviate this cost overhead. Addressing these limitations and extending CF-GAP to other task definitions, such as text-based queries, are promising directions for future work.

## References

1. Akbas, E., Eckstein, M.P.: Object detection through search with a foveated visual system. PLoS computational biology 13(10), e1005743 (2017)

2. Ba, J., Mnih, V., Kavukcuoglu, K.: Multiple object recognition with visual attention. arXiv preprint arXiv:1412.7755 (2014)

3. Cheung, B., Weiss, E., Olshausen, B.: Emergence of foveal image sampling from learning to attend in visual scenes. arXiv preprint arXiv:1611.09430 (2016)

4. Dai, J., Qi, H., Xiong, Y., Li, Y., Zhang, G., Hu, H., Wei, Y.: Deformable convolutional networks. In: Proceedings of the IEEE international conference on computer vision. pp. 764–773 (2017)

5. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021)

6. Dwibedi, D., Misra, I., Hebert, M.: Cut, paste and learn: Surprisingly easy synthesis for instance detection. In: Proceedings of the IEEE international conference on computer vision. pp. 1301–1310 (2017)

7. Ebel, P., Mishchuk, A., Yi, K.M., Fua, P., Trulls, E.: Beyond cartesian representations for local descriptors (2019)

8. Elsayed, G., Kornblith, S., Le, Q.V.: Saccader: Improving accuracy of hard attention models for vision. Advances in neural information processing systems 32 (2019)

9. Esteves, C., Allen-Blanchette, C., Zhou, X., Daniilidis, K.: Polar transformer networks. arXiv preprint arXiv:1709.01889 (2017)

10. Freeman, J., Simoncelli, E.P.: Metamers of the ventral stream. Nature neuroscience 14(9), 1195–1201 (2011)

11. Hayhoe, M., Ballard, D.: Eye movements in natural behavior. Trends in cognitive sciences 9(4), 188–194 (2005)

12. Howard, A., Sandler, M., Chu, G., Chen, L.C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., et al.: Searching for mobilenetv3. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1314–1324 (2019)

13. Ibrayev, T., Mukherjee, A., Aketi, S.A., Roy, K.: Toward two-stream foveationbased active vision learning. IEEE Transactions on Cognitive and Developmental Systems 16(5), 1843–1860 (2024)

14. Ibrayev, T., Nagaraj, M., Mukherjee, A., Roy, K.: Exploring foveation and saccade for improved weakly-supervised localization. In: Gaze Meets Machine Learning Workshop. pp. 61–89. PMLR (2024)

15. Itti, L., Koch, C., Niebur, E.: A model of saliency-based visual attention for rapid scene analysis 20(11), 1254–1259 (1998), conference Name: IEEE Transactions on Pattern Analysis and Machine Intelligence

16. Jaegle, A., Borgeaud, S., Alayrac, J.B., Doersch, C., Ionescu, C., Ding, D., Koppula, S., Zoran, D., Brock, A., Shelhamer, E., et al.: Perceiver io: A general architecture for structured inputs & outputs. arXiv preprint arXiv:2107.14795 (2021)

17. Kim, D., Lin, T.Y., Angelova, A., Kweon, I.S., Kuo, W.: Learning open-world object proposals without learning to classify. IEEE Robotics and Automation Letters 7(2), 5453–5460 (2022)

18. Kim, J., Jung, W., Kim, H., Lee, J.: Cycnn: A rotation invariant cnn using polar mapping and cylindrical convolution layers. arXiv preprint arXiv:2007.10588 (2020)

19. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF international conference on computer vision (2023)

20. Kolner, O., Ortner, T., Woźniak, S., Pantazi, A.: Mind the GAP: Glimpse-based active perception improves generalization and sample eficiency of visual reasoning. In: The Thirteenth International Conference on Learning Representations (2025)

21. Li, B., Wang, J., Hu, Y., Wang, C., Scherer, S.: Voxdet: Voxel learning for novel instance detection. Advances in Neural Information Processing Systems 36, 10604– 10621 (2023)

22. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object detection. In: Proceedings of the IEEE international conference on computer vision. pp. 2980–2988 (2017)

23. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In: European conference on computer vision. pp. 38–55. Springer (2024)

24. Lu, Y., Guo, Y., Ruozzi, N., Xiang, Y., et al.: Adapting pre-trained vision models for novel instance detection and segmentation. In: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 13341–13348. IEEE (2025)

25. McVitie, D.G., Wilson, L.B.: The stable marriage problem. Communications of the ACM 14(7), 486–490 (1971)

26. Mercier, J.P., Garon, M., Giguere, P., Lalonde, J.F.: Deep template-based object instance detection. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 1507–1516 (2021)

27. Mnih, V., Heess, N., Graves, A., kavukcuoglu, k.: Recurrent Models of Visual Attention. In: Advances in Neural Information Processing Systems. vol. 27. Curran Associates, Inc. (2014)

28. Navon, D.: Forest before trees: The precedence of global features in visual perception. Cognitive psychology 9(3), 353–383 (1977)

29. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

30. Osokin, A., Sumin, D., Lomakin, V.: Os2d: One-stage one-shot object detection by matching anchor features. In: European Conference on Computer Vision. pp. 635–652. Springer (2020)

31. Pardyl, A., Wronka, M., Wołczyk, M., Adamczewski, K., Trzciński, T., Zieliński, B.: Adaglimpse: Active visual exploration with arbitrary glimpse position and scale. In: European Conference on Computer Vision. pp. 112–129. Springer (2025)

32. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024)

33. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems 28 (2015)

34. Rolfs, M.: Microsaccades: small steps on a long way. Vision research 49(20), 2415– 2441 (2009)

35. Schmidt, T., Newcombe, R.: Segment this thing: Foveated tokenization for eficient point-prompted segmentation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 29428–29437 (2025)

36. Schwartz, E.L.: Spatial mapping in the primate sensory projection: analytic structure and relevance to perception. Biological cybernetics 25(4), 181–194 (1977)

37. Shen, Q., Zhao, Y., Kwon, N., Kim, J., Li, Y., Kong, S.: A high-resolution dataset for instance detection with multi-view object capture. Advances in Neural Information Processing Systems 36, 42064–42076 (2023)

38. Shen, Q., Zhao, Y., Kwon, N., Kim, J., Li, Y., Kong, S.: Solving instance detection from an open-world perspective. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 9901–9910 (2025)

39. Tian, Z., Shen, C., Chen, H., He, T.: Fcos: Fully convolutional one-stage object detection. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9627–9636 (2019)

40. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in Neural Information Processing Systems 30 (2017)

41. Wolfe, J.M.: Visual search. Current biology 20(8), R346–R349 (2010)

42. Woźniak, S., Jónsson, H., Cherubini, G., Pantazi, A., Eleftheriou, E.: On the visual analytic intelligence of neural networks. Nature Communications 14(1), 5978 (2023)

43. Wu, P., Xie, S.: V\*: Guided visual search as a core mechanism in multimodal llms. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

44. Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., Zemel, R., Bengio, Y.: Show, attend and tell: Neural image caption generation with visual attention. In: Bach, F., Blei, D. (eds.) Proceedings of the 32nd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 37, pp. 2048–2057. PMLR, Lille, France (07–09 Jul 2015)

45. Yarbus, A.L.: Eye Movements and Vision. Springer US (1967)

46. Yu, X., Zhou, Z., Becker, S.I., Boettcher, S.E., Geng, J.J.: Good-enough attentional guidance. Trends in Cognitive Sciences 27(4), 391–403 (2023)

47. Zeng, A., Florence, P., Tompson, J., Welker, S., Chien, J., Attarian, M., Armstrong, T., Krasin, I., Duong, D., Sindhwani, V., et al.: Transporter networks: Rearranging the visual world for robotic manipulation. In: Conference on Robot Learning. pp. 726–747. PMLR (2021)

48. Zhang, C., Han, D., Qiao, Y., Kim, J.U., Bae, S.H., Lee, S., Hong, C.S.: Faster segment anything: Towards lightweight sam for mobile applications. arXiv preprint arXiv:2306.14289 (2023)

49. Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., Shum, H.Y.: Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605 (2022)

50. Zhang, J., Khayatkhoei, M., Chhikara, P., Ilievski, F.: MLLMs know where to look: Training-free perception of small visual details with multimodal LLMs. In: The Thirteenth International Conference on Learning Representations (2025)

51. Zhang, M., Feng, J., Ma, K.T., Lim, J.H., Zhao, Q., Kreiman, G.: Finding any waldo with zero-shot invariant and eficient visual search. Nature Communications 9(1), 3730 (2018)

52. Zhou, X., Wang, D., Krähenbühl, P.: Objects as points. arXiv preprint arXiv:1904.07850 (2019)

## A Model Details

The code can be accessed via this link.

## A.1 Log-polar sensor

The log-polar sensor samples pixels based on the log-polar layout around the glimpse location, oversampling regions that are closer to the glimpse location and undersampling ones that are farther. More specifically, given an image I of size $H \times W$ and a glimpse location $( x , y )$ , the sensor samples pixels from the image I according to the log-polar coordinate transform as in [9]:

$$
x _ { s } = x + e ^ { \log ( \rho ) x _ { t } / W } \cos ( \frac { 2 \pi y _ { t } } { H } )\tag{4}
$$

$$
y _ { s } = y + e ^ { \log ( \rho ) x _ { t } / W } \sin ( \frac { 2 \pi y _ { t } } { H } )\tag{5}
$$

where $( x _ { s } , y _ { s } )$ denote the sampled points from $( x _ { t } , y _ { t } )$ coordinates in the image $^ { I , }$ and $\rho$ is a hyper-parameter that defines the radius of the region around the glimpse location in I from which the pixels are to be sampled. Compared to the cartesian grid-based layout, the log-polar design ofers a better resolution vs. field of view balance. In particular, it magnifies regions near glimpse locations, see Fig. 10 for an illustration.

## A.2 Fine search map

Input preparation. The scene and search target encoders for fine search map generation, shown in Fig. 2B-C, receive a scene glimpse $G _ { s } \in \mathbb { R } ^ { H _ { \theta } \times W _ { \rho } \times 3 }$ and a sequence of M search target glimpses $\{ G _ { m } \} _ { m = 1 } ^ { M ^ { - } } , \ : \bar { G } _ { m } \in \mathbb { R } ^ { H _ { \theta } \times W _ { \rho } \times 3 }$ , respectively. Each glimpse is an RGB log-polar image of size $H _ { \theta } \times W _ { \rho } ,$ where $H _ { \theta }$ is the angular dimension and $W _ { \rho }$ is the radial dimension (see Fig. 10). The search target glimpses are extracted from an image of the search target by applying the log-polar sensor at the center of the object as well as at its topmost and bottommost locations, yielding $M = 3$ glimpses. These locations are determined from the binary segmentation mask provided for each search target image. During training, a fourth glimpse $\scriptstyle ( M = 4 )$ is additionally sampled at a random target location to define the localization target, as detailed in Sec. A.4.

All glimpses are partitioned into non-overlapping patches of size $P \times P$ , each of which is projected into a D-dimensional feature space. This produces the scene feature grid $\dot { \boldsymbol { F } _ { s } } \in \mathbb { R } ^ { H _ { \theta } ^ { \prime } \times W _ { \rho } ^ { \prime } \times D }$ and the search target feature grids $\{ F _ { m } \} _ { m = 1 } ^ { M } ,$ $\boldsymbol { F } _ { m } \in \mathbb { R } ^ { H _ { \theta } ^ { \prime } \times W _ { \rho } ^ { \prime } \times D }$ , where the new size $H _ { \theta } ^ { \prime } \times W _ { \rho } ^ { \prime }$ accounts for patching. Note that in Sec. 3 the subscripts $\cdot _ { \theta }$ and $\cdot _ { \rho }$ were omitted for the sake of simplicity. In the technical description below, we restate Eqs. 1-3 for the reader’s convenience.

Scene encoder. The scene encoder (Fig. 2B) receives the scene feature grid $\pmb { F } _ { s }$ as input and outputs a glimpse feature map $\boldsymbol { F } _ { s } ^ { * } \in \mathbb { R } ^ { H _ { \theta } ^ { \prime } \times W _ { \rho } ^ { \prime } \times D }$ through three stages: compression, processing, and decoding.

![](images/a1be6d33026eb6757bdfa64b9fd0f2f8d1de3de2bdc43dcc9a0ed2f05deae02a.jpg)  
Fig. 10: Log-polar images produced around the red dots in the original images, with the dashed contours corresponding to the border of the area from which the pixels are sampled. Note that the red dots and dashed contours in the original images are mapped to the red solid and dashed lines, respectively, in the log-polar images.

Compression. The feature grid $\pmb { F } _ { s }$ is first flattened into a sequence of $H _ { \theta } ^ { \prime } \times W _ { \rho } ^ { \prime }$ tokens and compressed into a compact set of N latent embeddings via crossattention:

$$
\tilde { h } _ { s } = \mathcal { C } ( \mathsf { Q } = e _ { q } , \mathsf { K V } = F _ { s } ) ,\tag{6}
$$

where $\boldsymbol { e } _ { q } \in \mathbb { R } ^ { N \times D }$ is a set of learnable embeddings. Since $N \ll H _ { \theta } ^ { \prime } \times W _ { \rho } ^ { \prime } ,$ this step serves as a bottleneck that alleviates the quadratic cost of the subsequent self-attention.

Processing. The latent embeddings $\tilde { { h } } _ { s } ~ \in ~ \mathbb { R } ^ { N \times D }$ are subsequently processed through self-attention:

$$
h _ { s } = { \cal S } \Big ( \mathsf { Q K V } = \tilde { h } _ { s } \Big ) ,\tag{7}
$$

where $\pmb { h } _ { s } \in \mathbb { R } ^ { N \times D }$ . This stage enables the integration of fine local details encoded near the glimpse center with the broader peripheral context.

Decoding. The processed latent embeddings $h _ { s }$ are decoded back to the original spatial resolution using another cross-attention and scene feature grid $\boldsymbol { F } _ { s }$ as queries:

$$
\pmb { F } _ { s } ^ { * } = \mathcal { C } _ { \mathrm { d e c } } ( \mathsf { Q } = \pmb { F } _ { s } , \mathsf { K } = \pmb { h } _ { s } , \mathsf { V } = \pmb { e } _ { o } ) ,\tag{8}
$$

where $\boldsymbol { e } _ { o } \in \mathbb { R } ^ { N \times D }$ is a distinct set of learnable embeddings that defines the output space of the final glimpse feature map $\boldsymbol { F } _ { s } ^ { * }$

Throughout, $\mathcal { C } ( \cdot )$ and $\boldsymbol { \mathcal { S } } ( \cdot )$ denote cross-attention and self-attention blocks, respectively, each implemented as a single-layer transformer [40]. These blocks are shared with the search target decoder described below. The decoding crossattention $\mathcal { C } _ { \mathrm { d e c } } ( \cdot )$ uses a separate set of parameters.

Search target encoder. The search target encoder (Fig. 2C) iteratively processes the sequence of search target feature grids $\{ \boldsymbol { F } _ { m } \} _ { m = 1 } ^ { M ^ { - } } , \boldsymbol { F } _ { m } \in \mathbb { R } ^ { H _ { \theta } ^ { \prime } \times \hat { W } _ { \rho } ^ { \prime } \times D }$ 2 through cross-attention and self-attention blocks that are shared with the scene encoder, Eqs. (6) and (7). At each iteration $m ,$ the cross-attention block receives the flattened feature grid ${ \pmb F } _ { m }$ as keys and values, and the output of the previous iteration as queries:

$$
\tilde { \boldsymbol { h } } _ { T } ^ { ( m ) } = \mathcal { C } \left( \mathsf { Q } = \boldsymbol { h } _ { T } ^ { ( m - 1 ) } , \mathsf { K V } = \boldsymbol { F } _ { m } \right) ,\tag{9}
$$

$$
\pmb { h } _ { T } ^ { ( m ) } = \pmb { \mathcal { S } } \left( \mathsf { Q } \mathsf { K } \mathsf { V } = \tilde { \pmb { h } } _ { T } ^ { ( m ) } \right) ,\tag{10}
$$

where $\tilde { h } _ { T } ^ { ( m ) } , h _ { T } ^ { ( m ) } \in \mathbb { R } ^ { N \times D }$ . At the first iteration $( m { = } 1 )$ , the query is initialized with the same learnable embeddings used by the scene encoder, i.e. ${ h _ { T } ^ { ( 0 ) } = e _ { q } } .$ Unlike the scene encoder, the search target encoder omits the decoding stage; instead, the output of the final iteration $\bar { \pmb { h } } _ { T } = \pmb { h } _ { T } ^ { ( M ) } \in \mathbb { R } ^ { N \times D }$ is used directly as the set of target features for feature comparison.

Feature comparison. The target features $\pmb { h } _ { T } \in \mathbb { R } ^ { N \times D }$ are spatially correlated with the glimpse feature map $\breve { F _ { s } ^ { * } } \in \mathbb { R } ^ { H _ { \theta } ^ { \prime } \times W _ { \rho } ^ { \prime } \times \bar { D } }$ to produce the fine search map. Each of the $N$ target embeddings $\pmb { h } _ { T } ^ { ( n ) } \in \mathbb { R } ^ { D } , n = 1 , \dots , N$ , is treated as a 1×1 convolution kernel and convolved over $\boldsymbol { F } _ { s } ^ { * }$ , yielding N correlation maps:

$$
\begin{array} { r } { C _ { n } ( i , j ) = \pmb { h } _ { T } ^ { ( n ) } \cdot \pmb { F } _ { s } ^ { \ast } ( i , j ) , \quad C _ { n } \in \mathbb { R } ^ { H _ { \theta } ^ { \prime } \times W _ { \rho } ^ { \prime } } , } \end{array}\tag{11}
$$

where · denotes the dot product. These correlation maps are averaged into a single fine search map $\begin{array} { r } { S _ { \mathrm { f i n e } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } C _ { n } } \end{array}$

To improve robustness to viewpoint variation, search target glimpses are extracted from four images of the search target, each depicting the object from a diferent viewpoint. The above procedure is applied independently to each viewpoint example so that the final fine search map is averaged across maps produced for each example.

Positional Embeddings. While omitted in the description above, we add positional embeddings to the feature grids $\{ F _ { m } \} _ { m = 1 } ^ { M }$ and $\pmb { F } _ { s }$ . We adapt the standard sinusoidal positional embeddings (SPEs) from [5, 40] to the log-polar grid. Let $\pmb { F } \in \mathbb { R } ^ { H _ { \theta } ^ { \prime } \times \bar { W } _ { \rho } ^ { \prime } \times D }$ be a log-polar feature grid of size $H _ { \theta } ^ { \prime } \times W _ { \rho } ^ { \prime }$ , where $H _ { \theta } ^ { \prime }$ is the number of patches along the angular dimension and $W _ { \rho } ^ { j }$ is the number of patches along the radial dimension. For a patch at grid position $( p , q )$ , where $p \in$ $\{ 0 , \ldots , H _ { \theta } ^ { \prime } - 1 \}$ and $q \in \{ 0 , \ldots , W _ { \rho } ^ { \prime } { - } 1 \}$ , we compute its D-dimensional positional embedding $P E _ { ( p , q ) }$ by decomposing it into two independent 1D embeddings. Specifically, we partition the embedding dimension D into $D = D _ { \theta } + D _ { \rho }$ . The final embedding is the concatenation of the θ-dimension embedding and the ρ-dimension embedding:

$$
P E _ { ( p , q ) } = [ P E _ { \theta } ( p ) \oplus P E _ { \rho } ( q ) ]\tag{12}
$$

where $\oplus$ denotes vector concatenation. The θ-dimension of the log-polar grid is cyclic, meaning that the position $p = 0$ is adjacent to $p = H _ { \theta } ^ { \prime } - 1$ . Using the standard sinusoidal formula would create an artificial “seam”, incorrectly signaling a large distance between these adjacent patches. To resolve this issue, we define $P E _ { \theta }$ with cyclic frequencies. Specifically, at each position $p$ and frequency index $k \in \{ 1 , \ldots , D _ { \theta } / 2 \}$ , the components of $P E _ { \theta }$ are defined as:

$$
P E _ { \theta } ( p , 2 ( k - 1 ) ) = \sin \left( { \frac { p \cdot 2 \pi k } { H _ { \theta } ^ { \prime } } } \right)\tag{13}
$$

$$
P E _ { \theta } ( p , 2 ( k - 1 ) + 1 ) = \cos \left( \frac { p \cdot 2 \pi k } { H _ { \theta } ^ { \prime } } \right)\tag{14}
$$

This formulation ensures that $P E _ { \theta } ( p ) = P E _ { \theta } ( p + H _ { \theta } ^ { \prime } )$ , providing a continuous and cyclic representation of the angular position in the log-polar grid. $P E _ { \rho }$ is defined by the standard 1D SPE: for a position q and embedding dimension index $k ^ { \prime } \in \{ 0 , \ldots , D _ { \rho } / 2 - 1 \}$ , the components are:

$$
P E _ { \rho } ( q , 2 k ^ { \prime } ) = \sin ( q / 1 0 0 0 0 ^ { 2 k ^ { \prime } / D _ { \rho } } )\tag{15}
$$

$$
P E _ { \rho } ( q , 2 k ^ { \prime } + 1 ) = \cos ( q / 1 0 0 0 0 ^ { 2 k ^ { \prime } / D _ { \rho } } )\tag{16}
$$

The resulting positional embeddings are added to the log-polar feature grid $\pmb { F } \in \mathbb { R } ^ { H _ { \theta } ^ { \prime } \times W _ { \rho } ^ { \prime } \times D }$

## A.3 Location selection

Coarse and fine glimpsing employ diferent mechanisms to select the next glimpse location based on their respective search maps.

Coarse glimpsing. At each coarse glimpse iteration $i = \{ 1 , \ldots , N _ { c } \}$ , a glimpse location $\boldsymbol { l } _ { i } \in \mathbb { R } ^ { 2 }$ is selected using winner-takes-all (WTA), i.e. picking the location in the coarse search map $\bar { S _ { c } } \in \mathbb { R } ^ { H _ { c } \times W _ { c } }$ <sup>c</sup> with the highest value:

$$
l _ { i } = \mathrm { W T A } ( S _ { c } ^ { ( i ) } ) \stackrel { \mathrm { d e f } } { = } \arg \operatorname* { m a x } _ { p q } ( S _ { c , p q } ^ { ( i ) } )\tag{17}
$$

where $S _ { c } ^ { ( i ) }$ is the state of the coarse search map at iteration i. The WTA operation is followed by inhibition-of-return (IoR), which applies a mask $M ( l _ { i } ) \in \mathbf { \dot { \mathbb { R } } } ^ { H _ { c } \times W _ { c } }$ around location $\mathbf { \xi } _ { l _ { i } }$ to prevent it from repetitive selection:

$$
S _ { c } ^ { ( i + 1 ) } = S _ { c } ^ { ( i ) } \odot M ( l _ { i } )\tag{18}
$$

where ⊙ is the element-wise product and the value at each mask location $p , q$ is given by the inverted exponential radial kernel $1 - e ^ { - \epsilon \| ( p , q ) - l _ { i } | }$ <sup>∥2</sup> with ϵ being a hyperparameter and $( p , q ) \in \mathbb { R } ^ { 2 }$ a vector of location $( p , q )$

Fine glimpsing. At each fine glimpse iteration $j = \{ 1 , \ldots , N _ { f } \}$ , the fine glimpse location $\boldsymbol { l } _ { i } ^ { f } = \left( c _ { \theta } , c _ { \rho } \right)$ is computed as a centroid of the fine search map $S _ { f } \in$ $\mathbb { R } ^ { H _ { \theta } \times W _ { \rho } }$ , where $H _ { \theta }$ and $W _ { \rho }$ are the angular and radial dimensions of the logpolar image, respectively. The fine search map is first normalized using spatial softmax so that the normalized values $\hat { S } _ { f , \ p q }$ at each location $( p , q )$ become:

$$
\hat { S } _ { f , \ p q } = \frac { \exp ( S _ { f , \ p q } ) } { \sum _ { p ^ { \prime } = 1 } ^ { H _ { \theta } } \sum _ { q ^ { \prime } = 1 } ^ { W _ { \rho } } \exp ( S _ { f , \ p ^ { \prime } q ^ { \prime } } ) }\tag{19}
$$

The radial index of the centroid $c _ { \rho }$ is then defined as:

$$
c _ { \rho } = \sum _ { p = 1 } ^ { H _ { \theta } } \sum _ { q = 1 } ^ { W _ { \rho } } q \cdot \hat { S } _ { f , ~ p q }\tag{20}
$$

To account for the periodicity of the angular dimension $H _ { \theta }$ , we define the angular position $\alpha _ { p }$ for row $p$ of the log-polar image as $\alpha _ { p } = 2 \pi p / H _ { \theta }$ and compute the weighted sum of sine and cosine components:

$$
\mathbf { v } = \sum _ { p = 1 } ^ { H _ { \theta } } \sum _ { q = 1 } ^ { W _ { \rho } } \hat { S } _ { f , \ p q } \left[ \cos ( \alpha _ { p } ) \right]\tag{21}
$$

The resulting angular centroid $c _ { \theta }$ is calculated as:

$$
c _ { \theta } = { \frac { H _ { \theta } } { 2 \pi } } ( \mathrm { a t a n 2 } ( \mathbf { v } _ { y } , \mathbf { v } _ { x } ) { \pmod { 2 \pi } } )\tag{22}
$$

## A.4 Training details

The model for coarse search map generation, MobileNetV3, is used with its pretrained weights without any further fine-tuning. The model for the fine search map generation is trained from scratch using a dataset synthesized with a cutpaste-learn (CPL) framework [6]. The synthetic scenes were generated by pasting original images of the search targets onto random backgrounds. The training objective was to predict a search map that highlights a specific location of the search target within the synthetic scene.

Given a synthetic scene $\pmb { I } \in \mathbb { R } ^ { H \times W \times 3 }$ , we generate a scene glimpse $G _ { s } \in$ $\mathbb { R } ^ { H _ { \theta } \times W _ { \rho } \times 3 }$ by applying the log-polar sensor at a randomly selected location $l _ { s } \in \mathbb { R } ^ { 2 }$ . Note that $H _ { \theta }$ and $W _ { \rho }$ denote the angular and radial dimensions of the resulting log-polar image, respectively.

Each training sample consists of a scene glimpse $G _ { s }$ and a sequence of $M = 4$ search target glimpses $\{ G _ { m } \} _ { m = 1 } ^ { M } , G _ { m } \in \bar { \mathbb { R } } ^ { H _ { \theta } \times W _ { \rho } \times 3 }$ . The first three glimpses $( m = 1 , 2 , 3 )$ are generated by applying the log-polar sensor at the central, top, and bottom locations of the original search target image ${ \mathbf { } } I _ { T }$ to provide textural information. The fourth glimpse $( m = 4 )$ is generated by applying the log-polar sensor at a randomly picked location $\boldsymbol { l } _ { t } \in \mathbb { R } ^ { 2 }$ on the search target. This fourth glimpse specifies the exact part of the target to be localized within the scene glimpse $G _ { s }$

The ground truth label is a spatial map $Y \in \mathbb { R } ^ { H _ { \theta } \times W _ { \rho } }$ . This map consists of zeros except for a single element set to one at location $\ b { l } ^ { * }$ . The coordinate $\ b { l } ^ { * }$ is obtained by mapping the randomly picked target location $\mathbf { \xi } _ { l _ { t } }$ (from ${ \cal I } _ { T } )$ into the coordinate system of the scene glimpse $G _ { s }$ . The model is trained using cross-entropy loss between the predicted flattened fine search map and the flattened one-ground truth map $Y .$ , treating each spatial location as a distinct class. Training is performed using objects and random backgrounds from the HR-InsDet dataset [37].

## B Experimental details

In HR-InsDet and Robotools, scenes are sized at $6 1 4 4 \times 8 1 9 2$ and $1 9 2 0 \times 1 0 8 0$ pixels. While CF-GAP supports flexible image sizes, for faster experimentation, HR-InsDet scenes were resized to $4 0 9 6 \times 5 4 6 0$ . The Robotools scenes were kept in original resolution. The coarse glimpsing operates on scenes downscaled by a factor of $^ { 2 , }$ while the fine glimpsing operates on full-resolution scenes. The size of the log-polar images is $H _ { \theta } \times W _ { \rho } = 2 4 5 \times 2 4 5$ , and the radius $\rho$ of the log-polar sensor is set to half of the smallest dimension of the scenes, i.e. to 2048 and 540 for HR-InsDet and Robotools, respectively. Each scene undergoes $N _ { c } = 3 0$ coarse glimpses, with $N _ { f } = 3$ fine glimpses per coarse one. The size of the crop to be passed to the downstream architecture at the end each fine glimpsing loop is set $8 0 0 \times 8 0 0$ and $3 6 0 \times 3 6 0$ for HR-InsDet and Robotools, respectively. The cross- and self-attention components of the fine search map are implemented as simple single-layer transformer networks with 4 heads and feature dimensionality set to $D = 9 6$ . The patching size is $P \times P = 5 \times 5$ , the number of learnable embeddings is $N = 1 2 8$

## C Additional Results

Table 6: Performance on HR-InsDet for all our models and models evaluated in prior work. Results for models that were subsequently evaluated with CF-GAP are reproduced using publicly available code. Note that OTS-FM<sub>STT</sub> can be evaluated only in combination with CF-GAP as its STT backbone requires glimpse locations as inputs.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Venue and Year</td><td colspan="5"> $\mathrm { A P }$ </td><td rowspan="2"> $\mathrm { A P _ { 5 0 } }$ </td></tr><tr><td> $\mathrm { a v g }$ </td><td>hard</td><td></td><td>easy small medium large</td><td></td></tr><tr><td>CPLFasterRCNN [6, 33]</td><td>NeurIPS 2015</td><td>19.5</td><td>10.3 23.8</td><td>5.0</td><td>22.2</td><td>38.0</td><td>29.2</td></tr><tr><td>CPLRetinaNet [6, 22]</td><td>ICCV 2017</td><td>22.2</td><td>14.9 26.5</td><td>5.5</td><td>25.8</td><td>42.7</td><td>31.2</td></tr><tr><td> $\mathrm { C P L } _ { \mathrm { C e n t e r N e t ~ } } \mathrm { \bar { [ 6 , 5 2 ] } }$ </td><td>CVPR 2019</td><td>21.1</td><td>11.9 25.7</td><td>5.9</td><td>24.2</td><td>40.4</td><td>32.7</td></tr><tr><td>CPLFCOs [6,39]</td><td>ICCV 2019</td><td>22.4</td><td>13.2 28.7</td><td>6.2</td><td>26.5</td><td>38.1</td><td>32.8</td></tr><tr><td>CPLDINO [6, 49]</td><td>ICLR 2023</td><td>28.0 17.9</td><td>32.7</td><td>11.5</td><td>31.5</td><td>48.4</td><td>39.6</td></tr><tr><td> $\mathrm { O T S - F M \mathrm { _ { M o b i l e S A M } \ [ 3 7 , 4 8 ] } }$ </td><td>NeurIPS 2023</td><td>37.0 22.0</td><td>43.1</td><td>12.4</td><td>42.4</td><td>63.2</td><td>46.1</td></tr><tr><td> $\mathrm { O T S - F M _ { S A M } \ [ 1 9 , 3 7 ] }$ </td><td>NeurIPS 2023</td><td>41.6 28.0</td><td>47.6</td><td>14.6</td><td>45.8</td><td>69.1</td><td>49.1</td></tr><tr><td> $\mathrm { O T S - F M _ { G r o u n d i n g D I N O } \ [ 2 3 , 3 8 ] }$ </td><td>CVPR 2025</td><td>51.7 37.2</td><td>58.7</td><td>28.8</td><td>58.6</td><td>69.2</td><td>62.5</td></tr><tr><td> $\mathrm { I D O W _ { S A M } \ [ 1 9 , 3 8 ] }$ </td><td>CVPR 2025</td><td>48.8 32.1</td><td>56.5</td><td>20.8</td><td>55.3</td><td>73.4</td><td>49.1</td></tr><tr><td> $\mathrm { I D O W } _ { \mathrm { G r o u n d i n g D I N O } } \ [ 2 3 , 3 8 ]$ </td><td>CVPR 2025</td><td>57.0 40.7</td><td>64.4</td><td>35.3</td><td>63.0</td><td>73.6</td><td>69.3</td></tr><tr><td>NIDS-Net [24]</td><td>IROS 2025</td><td>63.8 39.9</td><td>74.6</td><td>32.4</td><td>72.7</td><td>86.1</td><td>69.8</td></tr><tr><td> $\overline { { \mathrm { C F - G A P \mathrm { ~ + ~ } O T S - F M _ { S A M } } } }$ </td><td></td><td>56.5 47.1</td><td>60.7</td><td>32.4</td><td>62.6</td><td>75.6</td><td>61.0</td></tr><tr><td> $\mathrm { C F \mathrm { - } G A P \mathrm { ~ + ~ } O T S \mathrm { - } F M _ { G r o u n d i n g D I N O } }$ </td><td></td><td>58.8 50.2</td><td>62.7</td><td>39.6</td><td>63.5</td><td>74.8</td><td>63.0</td></tr><tr><td> $\mathrm { C F \mathrm { - } G A P \mathrm { ~ + ~ } O T S \mathrm { - } F M _ { M o b i l e S A M } }$ </td><td></td><td>51.8 41.7</td><td>56.4</td><td>29.3</td><td>57.9</td><td>68.6</td><td>58.1</td></tr><tr><td> $\mathrm { C F - G A P + O T S \mathrm { - } F M _ { S T T } }$ </td><td></td><td>52.6 42.7</td><td>57.0</td><td>30.2</td><td>58.8</td><td>68.8</td><td>60.5</td></tr><tr><td></td><td></td><td>73.3</td><td>63.2 78.2 52.1</td><td></td><td>79.7</td><td></td><td></td></tr><tr><td> $\mathrm { C F - G A P + N I D S  – N e t }$ </td><td></td><td></td><td></td><td></td><td></td><td>86.1</td><td>78.4</td></tr></table>

Table 7: Performance on Robotools for all our models and models evaluated in prior work.
<table><tr><td>Model OS2D [30]</td><td>Venue and Year ECCV 2020</td><td>AP</td><td> $\mathrm { A P _ { 5 0 } }$ </td></tr><tr><td>DTOID [26]</td><td>WACV 2021</td><td>2.9 3.6</td><td>6.5 9.0</td></tr><tr><td> $\mathrm { O L N } _ { \mathrm { C o r r } } \ [ 1 7 ]$ </td><td>RA-L 2022</td><td>14.4</td><td>18.1</td></tr><tr><td> $\mathrm { V o x D e t } \ [ 2 1 ]$ </td><td>NeurIPS 2023</td><td>18.7</td><td>23.6</td></tr><tr><td> $\mathrm { O T S - F M _ { M o b i l e S A M } \ [ 3 7 , 4 8 ] }$ </td><td>NeurIPS 2023</td><td>35.5 46.9</td><td></td></tr><tr><td> $\mathrm { O T S - F M _ { S A M } \ [ 1 9 , 3 7 ] }$   $\mathrm { O T S - F M _ { G r o u n d i n g D I N O } \ [ 2 3 , 3 8 ] }$ </td><td>NeurIPS 2023</td><td>46.5 55.9</td><td></td></tr><tr><td> $\mathrm { I D O W _ { S A M } \ [ 1 9 , 3 8 ] }$ </td><td>CVPR 2025</td><td>56.7 64.8</td><td></td></tr><tr><td>IDOWGroundingDINO [23, 38]</td><td>CVPR 2025</td><td>51.9 63.8</td><td></td></tr><tr><td> $\mathrm { N I D S - N e t \ [ 2 4 ] }$ </td><td>CVPR 2025</td><td>59.0 67.8</td><td></td></tr><tr><td> $\overline { { \mathrm { C F - G A P \mathrm { ~ + ~ } O T S - F M _ { S A M } } } }$ </td><td>IROS 2025</td><td>64.9</td><td>79.4</td></tr><tr><td></td><td></td><td>55.2</td><td>68.1</td></tr><tr><td> $\mathrm { C F \mathrm { - } G A P \mathrm { ~ + ~ } O T S \mathrm { - } F M _ { G r o u n d i n g D I N O } }$ </td><td></td><td>62.1</td><td>70.8</td></tr><tr><td> $\mathrm { C F \mathrm { - } G A P \mathrm { ~ + ~ } O T S \mathrm { - } F M _ { M o b i l e S A M } }$ </td><td></td><td>49.4 63.5</td><td></td></tr><tr><td> $\mathrm { C F - G A P + O T S \mathrm { - } F M _ { S T T } }$ </td><td></td><td>46.559.9</td><td></td></tr><tr><td> $\mathrm { C F - G A P + N I D S - N e t }$ </td><td></td><td>70.3 79.4</td><td></td></tr></table>

Table 8: Sensitivity analysis of diferent hyperparameters. Results are shown for CF-GAP+NIDS-Net evaluated on the hard subset of HR-InsDet.
<table><tr><td>Hyperparameter</td><td>Values</td><td colspan="4"> $\mathrm { A P }$ </td></tr><tr><td>IoR kernel €</td><td> $\{ 0 . 1 \mid 0 . 5 \mid 1 \mid 1 0 \}$ </td><td></td><td></td><td></td><td>{63.0 | 63.2 | 63.2 | 59.9}</td></tr><tr><td># fine glimpses  $N _ { f }$ </td><td> $\{ 0 \mid 1 \mid 2 \mid 3 \}$ </td><td>{56.2 | 61.6 | 62.7 | 63.2}</td><td></td><td></td><td></td></tr><tr><td>Log-polar radius  $\rho$ </td><td>{256 | 512 | 1024 | 2048}</td><td></td><td></td><td></td><td>{60.9 | 61.8 | 61.8 | 63.2}</td></tr><tr><td>RoI size</td><td> $\{ 4 0 0 ^ { 2 } \ | \ 8 0 0 ^ { 2 } \ | \ 1 2 0 0 ^ { 2 } \ | \ 1 6 0 \dot { 0 } ^ { 2 } \}$ </td><td></td><td></td><td></td><td>{57.4 | 63.2 | 60.4 | 58.5}</td></tr></table>

## D Justifying the architecture for log-polar processing

Generating high-quality fine search maps requires an architecture for scene and search target encoders that is suitable to process log-polar glimpses. As argued in Sec. 3, using CNN-based models for this purpose may not be the best solution due to non-uniform resolution within a log-polar image. We experimented with diferent types of CNN architectures and found particularly beneficial to use deformable convolutions [4] that can adjust the kernels depending on the spatial position of the receptive field within the log-polar image. However, even this best-performing CNN configuration lags behind the attention-based architecture shown in Fig. 2B-C, as reported in Tab. 9. We attribute this gap to the fact that the attention-based model allows for integrating fine local details near the glimpse location with the broader context of the more distant periphery.

Table 9: Performance comparison when using CNN-based model for scene and search target encoders. Results are shown for CF-GAP+NIDS-Net evaluated on the two most challenging HR-InsDet subsets.
<table><tr><td>Scene &amp; search target encoders</td><td>AP [small hard</td></tr><tr><td>CNN</td><td>47.5 58.6</td></tr><tr><td>Cross- &amp; self-attention blocks</td><td>52.1 63.2</td></tr></table>

## E Qualitative examples

Fine glimpse locations

Coarse glimpse locations

![](images/be86dcf067ac2bab965f12dc07c310dcc1448e0336f1a8eb4528217dcdbec6d7.jpg)  
Fig. 11: Coarse and fine glimpse locations for two scenes where baseline models failed to detect the search target, while our approach succeeded. Top four examples are from HR-InsDet, and two examples at the bottom are from Robotools.

## F Failed cases

![](images/d298cca1bedc5e0870baeedc5e492b5d9a27b5f9991bfd5bdf2b6581a061322c.jpg)  
Fig. 12: Failed cases where CF-GAP could not find the search target, i.e. no glimpse location landed on the search target’s surface.

Fig. 12 illustrates cases where CF-GAP failed to find the search target, meaning that no glimpse location landed on the search target’s surface. We observe two main reasons for failures. First, small objects can be surrounded by heavy clutter or be strongly occluded (two examples on the right) so that the glimpsing is diverted towards distracting regions whose texture resembles that of the search target. Second, search targets can be barely distinguishable from their background (two examples on the left).