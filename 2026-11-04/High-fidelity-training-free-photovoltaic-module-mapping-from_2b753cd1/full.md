# High-fidelity training-free photovoltaic module mapping from multi-view UAV imagery

WenHao Duan<sup>a</sup>, Qi Chen<sup>a,∗</sup>, Penglei Zhang<sup>b,c</sup>, Ruizhe Deng<sup>b,c,d</sup>, Zhiling Guo<sup>d</sup>, Haoran Zhang<sup>e</sup>, Yuntian Chen<sup>b,c</sup>

<sup>a</sup>School of Geography and Information Engineering, China University of Geosciences (Wuhan), Wuhan 430074, China

<sup>b</sup>Ningbo Institute of Digital Twin, Eastern Institute of

Technology, Ningbo, Zhejiang 315200, China

<sup>c</sup>Zhejiang Key Laboratory of Industrial Intelligence and Digital Twin, Eastern Institute of Technology, Ningbo, Zhejiang 315200, China

<sup>d</sup>Department of Building Environment and Energy Engineering, The Hong Kong Polytechnic University, Kowloon, Hong Kong, China

<sup>e</sup>School of Urban Planning and Design, Peking University, Shenzhen, 518055, China

## Abstract

High-fidelity photovoltaic (PV) module mapping requires complete instance recovery and accurate delineation. Vision foundation models ofer zero-shot segmentation, but converting incomplete and inconsistent predictions across views into a complete, geometrically accurate map of individual modules remains challenging. We propose the Multi-View PV Module Mapper (MV-PVMapper), a training-free framework coupling vision foundation models with unmanned aerial vehicle (UAV) photogrammetry. First, Segment Anything Model 3 (SAM 3) generates candidates from original images using text prompts. Their contours are then transferred into object space through per-module slope correction to limit distortion from surface-model noise, and verified against representative module dimensions to guide cross-view instance selection. Finally, accepted instances return to all covering images as box prompts for SAM 3, supplementing text prompts with spatial cues for iterative segmentation, verification and fusion. On five-direction surveys of three operational rooftop PV plants at approximately 1 cm nadir ground sampling distance, two feedback iterations recover 7 710 of 7 711 reference modules with no false positives at an intersection-over-union (IoU) matching threshold of 0.5. Macro-averaged area precision and recall reach 99.00% and 98.52%, respectively. Averaged across sites, the aggregated Jaccard

index reaches 97.55%, compared with 60.73% and 41.24% for orthomosaic segmentation and multi-view pixel-voting baselines, respectively; average precision at the stricter IoU threshold of 0.95 reaches 95.67%. Ablation studies show that module-size constraints are critical for instance separation, while slope correction and feedback to all covering views improve boundary accuracy. These results are achieved without task-specific image annotation or segmentation-model weight updates.

Keywords: Photovoltaic module mapping, UAV photogrammetry, Vision foundation models, Zero-shot instance segmentation, Multi-view fusion

## 1. Introduction

The global energy transition is rapidly expanding the installed base of photovoltaic (PV) systems. By the end of 2025, global cumulative PV capacity had approached 3 TW, after an estimated 698 GW was installed in that year alone(de l’Epine et al., 2026). As the installed base grows, maintaining reliable performance over the lifetime of these assets requires inspection and maintenance at an increasingly large scale(Gallardo-Saavedra et al., 2018; Chen et al., 2023; Barraz et al., 2025a).

Within large PV installations, tasks such as targeted diagnosis, repair and replacement require inspection findings to be linked to individual modules and their locations. Module-level georeferencing supports this link between image-based inspection and field operations(Barraz et al., 2025a; Bommes et al., 2022). A module map provides a common spatial representation in which each module has its own boundary and location. Its quality depends on recovering the module inventory while preserving the separation and geometry of individual instances, rather than only identifying the area occupied by PV arrays. Unmanned aerial vehicle (UAV) surveys can cover an entire plant eficiently, but converting UAV images into such a map remains dificult. Modules are densely and repeatedly arranged, their gaps may span only a few pixels, and their apparent scale and shape change under oblique viewing.

PV module mapping involves both extracting individual instances and establishing their geographic locations. Rule-based methods can avoid labelled data, but depend on assumptions about colour, line structure, shape or array layout(Wang et al., 2018; Cardoso et al., 2024). Supervised detectors and segmentation networks can learn more flexible appearance features, but require annotated UAV images. Even domain-generalization methods rely on training data collected across diferent module types and imaging conditions to improve performance on unseen sites(Barraz et al., 2024). Beyond recognising and delineating modules, mapping also requires their image-space detections to be placed in a common geographic frame. Photogrammetry provides this spatial reference for UAV imagery. Structure-from-motion (SfM) estimates camera poses and reconstructs scene geometry, supporting both the georeferencing of modules detected in individual images and the generation of orthomosaics. These products support diferent mapping workflows: modules can be extracted from a georeferenced digital orthophoto map (DOM)(Cardoso et al., 2024), or observations in multiple images can be associated and located using the reconstructed geometry(Bommes et al., 2022; Kozák et al., 2026).

Promptable visual foundation models (VFMs) ofer a way to reduce the annotation burden. The Segment Anything Model (SAM) supports zero-shot transfer to new image distributions and tasks through prompts(Kirillov et al., 2023). SAM 3 further accepts concept prompts and can detect and segment instances named by a short noun phrase(Carion et al., 2025). Recent work has shown the potential of this training-free, open-vocabulary approach by formulating remote-sensing image interpretation as two-dimensional segmentation(Li et al., 2025). Within a photogrammetric workflow, the same model can be applied directly to a georeferenced DOM, so the predicted masks already have ground locations and physical extents. However, segmentation on a DOM uses a single orthorectified representation rather than explicitly combining predictions from the overlapping source images. For densely arranged PV modules, observations from diferent viewpoints can provide complementary evidence where individual views leave boundaries unclear.

A multi-view UAV oblique survey provides overlapping images that can observe the same module from several directions, while camera poses and a digital surface model (DSM) relate these observations within a common objectspace frame. In addition, PV modules within a plant commonly follow one or a few standardized size specifications, allowing representative dimensions to be established from a small number of field measurements. Using these sources jointly for module mapping, however, raises two related challenges. The first is to transfer predictions reliably into object space despite noise in the reconstructed surface, which can distort their locations and shapes. The second is to combine predictions from views with diferent visibility and image quality, so that complementary observations reinforce one another while conflicting results are suppressed.

This study aims to achieve high-fidelity PV module mapping by combining zero-shot VFM segmentation with redundant observations from UAV photogrammetry, without task-specific training. We propose the Multi-View PV Module Mapper (MV-PVMapper), which links image-space segmentation and object-space geometric verification through iterative multi-view feedback. First, SAM 3 generates module candidates from the original images using a text concept prompt, without site-specific image annotation or segmentation-model weight updates. Second, per-module slope correction transfers candidate contours into object space while limiting deformation caused by DSM noise. Third, segmentation confidence and consistency with representative module dimensions guide cross-view fusion, retaining plausible module instances while suppressing conflicting and duplicate predictions. Finally, accepted object-space instances are projected into covering images as box prompts; the resulting masks are transferred back for geometric verification and fusion, providing updated spatial evidence for subsequent segmentation rounds. We evaluate MV-PVMapper on five-direction oblique photogrammetric data from three operational rooftop PV plants, with a nadir ground sampling distance (GSD) of approximately 1 cm/pixel. After two feedback iterations, MV-PVMapper matches 7 710 of the 7 711 reference modules across all three sites, with no false positives at an intersection-overunion (IoU) threshold of 0.5. It achieves macro-averaged area precision and recall of 99.00% and 98.52%, respectively. Its macro-averaged aggregated Jaccard index (AJI) and average precision at the stricter IoU threshold of 0.95 $( A P _ { 9 5 } )$ reach 97.55% and 95.67%, compared with 60.73% and 60.51% for an orthomosaic segmentation baseline (DOM-IterSeg), and 41.24% and 27.52% for a multi-view pixel-voting baseline (MV-PixelVote). Together, these results show near-complete instance recovery and accurate delineation on the evaluated sites, rather than high pixel precision alone.

The main contributions are as follows:

• We propose MV-PVMapper, a training-free framework that couples segmentation by a pretrained VFM with multi-view instance fusion in an iterative refinement loop for PV module mapping. By feeding geometrically verified instances back to all covering views as spatial prompts, the framework uses complementary observations to recover missed modules and refine their boundaries.

• We introduce a per-module slope-correction strategy for projecting PV masks from image space to object space. By adapting the projection to each module’s local surface, it preserves module geometry and improves robustness to DSM noise.

• We convert readily available module dimensions into an object-space geometry score that guides candidate selection, cross-view association and feedback. This geometry-guided strategy suppresses over- and under-segmentation and helps limit error accumulation across iterations.

## 2. Related Work

We review PV extraction methods, prompting strategies for VFMs, and multi-view fusion with geometric priors. The review distinguishes task-specific learning from training-free inference and examines how spatial evidence can support instance association and delineation, providing the context for MV-PVMapper.

## 2.1. PV Module Extraction and Mapping

Image-based PV extraction supports applications ranging from installation inventories to component inspection (Chen et al., 2023; Barraz et al., 2025a). Traditional methods identify panels through explicit appearance and geometric rules. Association between region and line primitives can support automatic template generation and matching without training samples (Wang et al., 2018). Edge detection, morphological processing and contour extraction provide another route to delineating individual panels (Cardoso et al., 2024). These methods make use of the regular appearance of PV installations, but their applicability depends on the chosen rules and thresholds. Changes in illumination, panel appearance or the visibility of narrow gaps can weaken the cues on which extraction relies.

Supervised learning replaces much of this manual feature design with representations learned from task data. For UAV thermal imagery, training across diferent module types and acquisition conditions has been used to improve generalisation to unseen installations (Barraz et al., 2024). Other approaches reduce the cost of adapting models to new data: weakly supervised domain adaptation transfers rooftop PV segmentation from densely labelled source imagery to target imagery with sparse point annotations (Wang et al., 2026), while parameter-eficient VFM adaptation has been explored for PV change detection (Huang et al., 2026). These developments address trainingdata diversity and annotation cost, but still require task-specific learning. Their practical deployment therefore still involves obtaining suitable training data and adapting the model.

## 2.2. Prompting and Refinement with Visual Foundation Models

SAM allows points, boxes and masks to specify the object to be segmented (Kirillov et al., 2023). Evaluations on overhead imagery show useful zero-shot transfer, but also failures associated with the characteristics of remote-sensing targets (Ren et al., 2024a). One response is to learn task-specific prompts or output representations. Category-aware prompt generation supports automatic instance segmentation (Chen et al., 2024a), while automatic box prompting and a decoder trained to predict vertices and boundaries support polygonal building extraction (Wang et al., 2024). In PV imagery, learned multi-scale and edge-pyramid prompts incorporate scale and boundary information into SAM adaptation (Yao et al., 2026). These methods use the foundation model as a basis for task-specific learning.

Language prompts ofer a convenient way to specify a target category without placing a spatial prompt on each object. This can be implemented by coupling a text-conditioned detector with SAM (Ren et al., 2024b), or through the unified concept-prompting interface of SAM 3 (Carion et al., 2025). Direct application of SAM 3 to remote-sensing semantic segmentation has also been explored without additional training (Li et al., 2025). PVspecific vision–language pretraining ofers a complementary approach: PANEL supports downstream zero- and few-shot interpretation after learning from a PV image–text corpus (Deng et al., 2026). MV-PVMapper instead uses a generic VFM without PV-specific training and coordinates its predictions across UAV views to recover georeferenced module instances.

The prompt interface also supports refinement: previous masks and corrective prompts can be supplied in subsequent segmentation rounds (Kirillov et al., 2023). The quality of this refinement depends on the evidence supplied to the model. In spatial mapping, observations from other views and geometric information provide possible sources of such evidence, alongside the prediction in the current image.

## 2.3. Multi-view Fusion and Geometric Priors

Photogrammetric geometry allows observations from diferent images to be compared in a common spatial frame. In PV mapping, detections have been georeferenced through tracking and SfM, or associated across images using semantic keypoints and plant structure (Bommes et al., 2022; Kozák et al., 2026). Learned fusion also uses this spatial correspondence to combine information before prediction. Image features can be backprojected and pooled on a voxel grid together with geometric features (Dai and Nießner, 2018), or aggregated on 3D points with weights learned from viewing conditions (Robert et al., 2022). For aerial building mapping, MVMapper combines original-image and object-space features using supervised learning and piecewise afine projection to reduce DSM-induced misalignment (Chen et al., 2024b). MV-PVMapper shares MVMapper’s use of a common spatial frame to exploit complementary aerial observations, but operates on predicted module instances rather than learned feature fusion. Geometric verification and cross-view feedback coordinate these instances while the segmentation model remains frozen.

At the instance level, fusion must distinguish repeated detections from neighbouring objects. Non-maximum suppression (NMS) ranks candidates and removes those with excessive overlap; Soft-NMS instead reduces their scores according to overlap (Bodla et al., 2017). For multi-view masks, the evidence can extend beyond pairwise overlap. Consistency with masks from several views can guide the merging of 3D regions (Yin et al., 2024), and agreement among views can define graph connections for clustering masks into instances (Yan et al., 2024). Spatial information can also guide segmentation itself: SAMPro3D projects shared 3D prompts into images and selects and consolidates them using multi-view mask evidence, without additional training (Xu et al., 2025). Thus, geometric information can support both instance association and prompt selection.

Module dimensions and regular layouts provide constraints beyond visual appearance. In PV applications, physical module dimensions have been used to generate training boxes automatically (Barraz et al., 2025b). Detected templates and layout constraints can also guide the completion of module maps (Cardoso et al., 2024). Such priors can complement VFMs: SinkSAM-Net uses topographic depressions to generate prompts and shape information to filter SAM masks before training a downstream segmenter (Rafaeli et al., 2025). MV-PVMapper uses representative dimensions to verify object-space candidates and select instances for mapping and subsequent spatial prompting.

## 3. Datasets

We constructed a UAV oblique photogrammetric dataset of three operational rooftop PV plants. For each plant, it contains image-space data, namely the original nadir and oblique images, and object-space data, namely the camera poses, DSM and DOM reconstructed from these images, together with a reference polygon for every module.

## 3.1. Study Sites and UAV Acquisition

The three plants are installed on building roofs and contain 7 711 modules in total (Table 1). Sites 1 and 2 contain 1 772 and 2 305 modules, respectively, each mounted at a single height, whereas the 3 634 modules at Site 3 are mounted at two heights about 10 m apart.

All sites were surveyed with a DJI Mavic 3E, whose wide-angle colour camera records 5280 × 3956 pixel images at a focal length of 12.29 mm. Each survey combined nadir images with oblique images from four directions at 80% forward and 70% side overlap, giving 2 165 images for the three sites. The median flight heights above the take-of point were 35.0 to 41.8 m, corresponding to a nadir GSD of 0.93 to 1.11 cm/pixel. The oblique gimbal pitch was −45<sup>◦</sup> at Sites 1 and 2 and $- 6 0 ^ { \circ }$ at Site 3. Site 3 was surveyed in August 2024, and Sites 1 and 2 in October and November 2025.

Table 1: Study sites and UAV acquisition. Module size is the representative length × width of the modules at each site, taken in this study as the mean length × width of the reference modules.
<table><tr><td></td><td>Site 1</td><td>Site 2</td><td>Site 3</td></tr><tr><td>Reference modules</td><td>1772</td><td>2305</td><td>3634</td></tr><tr><td>Module size (m) Mounting levels</td><td> $2 . 2 2 \times 1 . 0 9$  1</td><td> $2 . 1 7 \times 1 . 1 0$  1</td><td> $2 . 2 0 \times 1 . 0 9$  2</td></tr><tr><td>Mounting type</td><td>Roof-flush</td><td>Racked</td><td>Roof-flush</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Survey date</td><td>2025-10</td><td>2025-11</td><td>2024-08</td></tr><tr><td>Relative flight height (m)</td><td>35.0</td><td>38.0</td><td>41.8</td></tr><tr><td>Oblique gimbal pitch</td><td>-45°</td><td>-45°</td><td>-60°</td></tr><tr><td>Nadir GSD (cm/pixel)</td><td>0.93</td><td>1.01</td><td>1.11</td></tr><tr><td>Images (nadir / oblique)</td><td>62 / 266</td><td>94 / 289</td><td>329 / 1125</td></tr><tr><td>Coordinate system (EPSG code)</td><td>4550</td><td>4550</td><td>4547</td></tr></table>

## 3.2. Photogrammetric Processing and Reference Generation

The images of each site were processed in ContextCapture. Aerotriangulation of all nadir and oblique images estimated the camera poses, with the image positions recorded by the onboard real-time kinematic receiver providing the georeferencing. Dense matching produced the DSM, from which a true orthophoto, referred to below as the DOM, was generated. Figure 1 shows the DOM and the reconstructed 3D model of each site. Every module is represented by one reference polygon, giving 7 711 reference polygons for the three sites.

![](images/02fb11dcbbcda61cded3d9228455cf16a2bc331bc367bb498a97626a8e3a548f.jpg)  
Figure 1: DOMs (top) and oblique views of the reconstructed 3D models (bottom) of Sites 1–3.

## 4. Methods

MV-PVMapper takes overlapping UAV images, calibrated camera poses, a georeferenced DSM and representative module dimensions as inputs, and outputs a map of separate, georeferenced module polygons (Figure 2). It alternates between prompted segmentation in image space and geometric verification and fusion in object space. Per-module slope correction transfers candidate contours to a common spatial frame, where segmentation confidence and module geometry guide cross-view instance selection. The accepted instances return to the covering images as box prompts for further segmentation. The final module map is the accepted instance set from the last fusion round.

![](images/92964420f86d36f1b95f880751e613a08c7c1fa22d2dee0141b575dd20df693e.jpg)  
Figure 2: Workflow of MV-PVMapper. Panels (a)–(d) correspond to Sections 4.1–4.4. The snowflake marks the frozen weights of the segmentor, the promptable segmentation model described in Section 4.1. $C _ { v }$ is the camera pose of image $v , \mathcal { M } _ { v } ^ { k }$ is the mask of its k-th candidate, and L and W are the representative module length and width.

## 4.1. View-Wise Prompted Segmentation

Iterative segmentation requires prompts that can evolve as spatial evidence becomes available. A text prompt identifies the target category and provides initial candidates without known module locations. Once reliable instances have been identified, box prompts add location cues that a repeated category description alone cannot provide. The segmentation model must therefore accept text and box prompts and return instance masks with confidence scores. We use SAM 3, whose concept-prompting and interactive segmentation capabilities support these requirements (Carion et al., 2025).

For the initial pass, SAM 3 is queried with the text prompt a PV panel. Inference uses overlapping windows to preserve the narrow gaps between neighbouring modules, which may become dificult to resolve if an entire UAV image is resized to the model input. The last row and column of windows are aligned with the image boundary, and the predicted masks are restored to the original image coordinates using the window ofsets. Candidates below the confidence or mask-area threshold are discarded.

In subsequent rounds, accepted object-space instances are projected back into the covering images to generate box prompts (Section 4.4). By conveying module locations established from other views, these box prompts guide the re-segmentation of modules previously missed or poorly delineated in the target image. Each box is clipped to every window it intersects. For each window, all available box prompts are used unless their number exceeds $N _ { b }$ in which case $N _ { b }$ prompts are randomly selected. Windows with box prompts use both spatial and text prompts, whereas windows without box prompts use text-only inference. Duplicates from overlapping windows are retained until object-space fusion, where they are resolved together with cross-view duplicates (Section 4.3). Numerical settings are given in Section 5.1.

## 4.2. Per-Module Slope Correction Using DSM

Photogrammetric DSMs can smooth sharp height discontinuities at object edges. Intersecting each contour vertex’s viewing ray with such a surface can therefore displace or distort an otherwise accurate image-space contour (Chen et al., 2024b). In dense PV arrays, these errors afect module edges and the narrow gaps between rows.

We introduce a per-module slope-correction strategy that uses the approximately planar module surface to project PV mask contours into object space. For each mask, a z-bufer identifies visible DSM points that fall within the mask when projected into the image. We then fit a local plane to these points using random sample consensus (RANSAC) (Fischler and Bolles, 1981), reducing the influence of outliers near module edges or on surrounding surfaces. Candidates are excluded if the fitted plane is insuficiently supported by the points or has an implausibly steep slope. For each retained candidate, the camera pose and interior orientation define a viewing ray through each mask-contour vertex. We intersect these rays with the fitted plane and use the planimetric coordinates of the intersections to construct the object-space polygon. The contour is thus transferred using a common local plane, rather than independent intersections with the DSM, preserving straight edges without imposing a rectangular shape. Whether the resulting polygon corresponds to one physical module is assessed in Section 4.3.

Figure 3 illustrates the efect of the local surface model. Collinearitybased projection transfers local height errors to the contour vertices and can bend edges across module gaps. A two-dimensional image-to-object afine transformation (Chen et al., 2024b) preserves straight edges, but DSM points from the surrounding roof can bias the estimated afine parameters. Slope correction projection instead uses RANSAC to fit the surface from visible DSM points that project into the mask, giving closer agreement with the module boundaries in this example.

![](images/13cff65670271eb1660b06d33ca8f51073b6833f1ba3f0b21500fdd7d2433a13.jpg)  
(a) Image space

![](images/ef8f13dfd1c72ffce1658a314b3711c5c66a1a0e9570bdaa78accb239db73b7e.jpg)  
(b) Collinearity-based projection

![](images/0fb01507e3b6017de817ecdf04a281acf96e754868a96be1b4b39507c0d784b7.jpg)  
(c) Afine-based projection

![](images/1495ba4931906791429a97450b4199f0717289c9284765f8b720f8b22df31e83.jpg)  
(d) Slope correction projection  
Figure 3: Object-space contours of a row of five modules, transferred from the image-space masks in (a) by (b) intersecting the viewing ray of each contour vertex with the DSM, (c) a two-dimensional image-to-object afine transformation estimated for each candidate from DSM point correspondences and (d) per-module slope correction.

## 4.3. Geometry-Guided Scoring and Cross-View Fusion

After slope correction, candidates from all images share one object-space coordinate system. Some still represent module fragments or merged groups, while others are repeated observations of the same module. We score their geometric plausibility before using NMS to select representative instances across views.

Rooftop PV arrays often use repeated module specifications and regular installation patterns. Representative dimensions can be obtained by measuring a module or consulting its product specification, providing a low-cost indication of the expected size of one instance. We use these dimensions as a soft constraint, not a requirement that every module have exactly the same size. The evaluated sites contain modest dimensional variation, which is accommodated through a tolerance in the score.

Let L and W be the representative module length and width, with nominal area $A _ { \mathrm { r e f } } = L W$ . For a candidate with planimetric area A and minimum rotated bounding-rectangle area $A _ { \mathrm { { r e c t } } }$ , the area score $s _ { A }$ and rectangularity score $s _ { R }$ are

$$
s _ { A } = \exp \left( - \frac { \vert A - A _ { \mathrm { r e f } } \vert } { \delta _ { A } A _ { \mathrm { r e f } } } \right) ,\tag{1}
$$

$$
s _ { R } = \exp \left( - \frac { \lvert A / A _ { \mathrm { r e c t } } - 1 \rvert } { \delta _ { R } } \right) ,\tag{2}
$$

where the positive tolerances $\delta _ { A }$ and $\delta _ { R }$ control the decay of the two scores. The area score penalises fragments and merged groups that depart from the expected module size. The rectangularity score penalises irregular contours, including those with a plausible area. We combine them with the model confidence as

$$
S _ { \mathrm { g e o } } = \frac { w _ { A } s _ { A } + w _ { R } s _ { R } } { w _ { A } + w _ { R } } ,\tag{3}
$$

$$
\begin{array} { r l } { S } & { { } = \frac { w _ { \mathrm { c o n f } } s _ { \mathrm { c o n f } } + w _ { \mathrm { g e o } } S _ { \mathrm { g e o } } } { w _ { \mathrm { c o n f } } + w _ { \mathrm { g e o } } } , } \end{array}\tag{4}
$$

where $S _ { \mathrm { g e o } }$ is the combined geometry score, $s _ { \mathrm { c o n f } } \in [ 0 , 1 ]$ is the segmentationmodel confidence, and S is the final candidate score. The non-negative weights $w _ { A }$ and $w _ { R }$ balance the geometric terms, while $w _ { \mathrm { c o n f } }$ and $w _ { \mathrm { g e o } }$ balance confidence and geometry; each pair has a positive sum. Candidates with $S < S _ { \mathrm { m i n } }$ are discarded, where $S _ { \mathrm { m i n } }$ is the acceptance threshold.

NMS processes the retained candidates in descending order of S and suppresses overlapping lower-scoring candidates above a polygon-IoU threshold $\theta _ { \mathrm { I o U } }$ . Polygon overlap is used because axis-aligned bounding boxes can overlap substantially even for separate modules in a rotated array. An R-tree index limits overlap tests to spatially intersecting bounding boxes. The highestscoring retained observation represents each module; this instance set supplies both the current map and the next round of prompts.

## 4.4. Iterative Covering-View Feedback

The iteration index t counts feedback rounds, with $t = 0$ denoting the initial text-only pass. Segmentation, slope correction, scoring and fusion yield the accepted instance set for that round. Although each retained polygon originates from one image, its role is not confined to that source view: it is projected into every image whose ground footprint intersects the instance. The image-space bounding box of each projected polygon is then used as a spatial prompt for re-segmentation.

The next round re-segments all windows using the prompting strategy in Section 4.1. The resulting candidates undergo projection, geometric verification and cross-view fusion again, updating the map and the evidence available for subsequent rounds. Routing each instance to all covering views allows a successful detection in one image to guide another image in which that module was previously missed or poorly delineated. The last accepted set is the output. We evaluate successive rounds separately in Section 5.5 to distinguish gains in module recovery from gains in boundary accuracy.

## 5. Results and Discussion

The evaluation examines whether MV-PVMapper produces a complete module map with separate, geometrically consistent instances. We first compare MV-PVMapper with DOM-IterSeg and MV-PixelVote, then examine the three main components of the framework, the number of feedback iterations and the available camera directions.

## 5.1. Configuration and Implementation Details

All methods use the same pretrained SAM 3 model without task-specific weight updates. The experiments process 1024 × 1024-pixel windows with 50% overlap and use the text prompt a PV panel. The confidence threshold is 0.45, the minimum mask area is 500 pixels, and each window receives at most $N _ { b } = 5$ box prompts. Projected boxes with either side shorter than 50 pixels are discarded. The representative module dimensions are listed in Table 1.

For geometric scoring, we set $\delta _ { A } = 0 . 2 5 , \delta _ { R } = 0 . 1 , ( w _ { A } , w _ { R } ) = ( 1 0 , 5 )$ and $( w _ { \mathrm { c o n f } } , w _ { \mathrm { g e o } } ) = ( 0 . 2 , 0 . 8 )$ . The acceptance threshold is $S _ { \mathrm { m i n } } = 0 . 7$ , and the polygon-IoU threshold for NMS is $\theta _ { \mathrm { I o U } } = 0 . 2$ . Unless specified otherwise, the reported maps are obtained after two feedback rounds $( t = 2 )$ , following the initial text-only pass. The iteration experiment also reports the intermediate results at $t = 0$ and $t = 1$

All experiments were run on one NVIDIA RTX 5880 Ada GPU (48 GB) and 128 GB of RAM, under Ubuntu 22.04.5 LTS with Python 3.12.12 and PyTorch 2.6.0 (CUDA 11.8). SAM 3 is used through its oficial implementation and released weights (Carion et al., 2025).

## 5.2. Evaluation Metrics and Baseline Design

All methods are evaluated as georeferenced polygons in the same objectspace coordinate system. We assess PV coverage first and individual module instances second. Metrics are computed separately for each site; overall results are unweighted averages of the three site-level values, reported as percentages to two decimal places.

## 5.2.1. Area-Level Agreement

Area-level evaluation follows the foreground-overlap principle used in semantic segmentation (Long et al., 2015). Predicted polygons are merged into one coverage region, and reference polygons into another, without retaining instance identities. Area IoU is their intersection area divided by their union area. Area precision is the fraction of predicted coverage supported by the reference, area recall is the fraction of reference coverage recovered, and area $F _ { 1 }$ is their harmonic mean. These metrics assess coverage but do not distinguish boundaries between neighbouring modules.

## 5.2.2. Instance-Level Agreement

We use recognition quality (RQ), segmentation quality (SQ) and panoptic quality (PQ) to separate instance recovery from boundary accuracy (Kirillov et al., 2019). Predictions and references are matched one-to-one at an IoU threshold of 0.5. True positives (TP) are matched predictions, false positives (FP) are unmatched predictions, and false negatives (FN) are unmatched reference modules. Let $\tau$ denote the set of matched pairs $( d , g )$ , where d is a predicted polygon and g its reference polygon. Then

$$
\mathrm { R Q } = \frac { 2 \mathrm { T P } } { 2 \mathrm { T P } + \mathrm { F P } + \mathrm { F N } } , \quad \mathrm { S Q } = \frac { 1 } { \mathrm { T P } } \sum _ { ( d , g ) \in \mathcal { T } } \mathrm { I o U } ( d , g ) , \quad \mathrm { P Q } = \mathrm { S Q } \times \mathrm { R Q } .\tag{5}
$$

RQ is the harmonic mean of instance precision and recall. SQ measures the mean overlap of successfully matched modules and must therefore be read together with RQ. PQ combines both aspects.

We also report AJI, which aggregates overlap by instance rather than over the merged foreground (Kumar et al., 2017). For consistency across our instance metrics, we use the same thresholded one-to-one matching defined above:

$$
{ \mathrm { A J I } } = { \frac { \sum _ { ( d , g ) \in { \mathcal { T } } } \left| d \cap g \right| } { \sum _ { ( d , g ) \in { \mathcal { T } } } \left| d \cup g \right| + \sum _ { u \in { \mathcal { U } } } \left| u \right| } } ,\tag{6}
$$

where | · | denotes polygon area and $\mathcal { U }$ contains all unmatched predicted and reference polygons. This matching convention is an adaptation of the original AJI definition, which assigns predictions by maximum overlap without the present 0.5 threshold. The denominator penalises unmatched regions as well as imperfect matched boundaries. Finally, $A P _ { 9 5 }$ measures average precision under a stricter IoU threshold of 0.95, assessing the recovery of accurately delineated modules.

## 5.2.3. Baseline Design

The comparison tests two alternative mapping strategies while retaining SAM 3 as the segmentation model. The baselines adapt established processing ideas to the present training-free setting; they are not reproductions of the cited systems or direct comparisons with those papers’ published results.

• DOM-IterSeg: iterative segmentation on an orthophoto. Extracting modules from a georeferenced orthomosaic is an established PV mapping route (Cardoso et al., 2024). We implement it with iterative SAM 3 segmentation and the module-geometry prior on each site’s DOM. This baseline tests how far prompted segmentation and geometric screening can proceed within one orthorectified representation, without cross-view feedback from the source images.

• MV-PixelVote: multi-view pixel-level decision fusion. Projection-based semantic mapping can combine predictions from multiple views at shared spatial locations (Lawin et al., 2017). We adapt this decision-fusion idea by segmenting the same perspective images used by MV-PVMapper, transferring their masks to object space and combining them by pixel-level voting. Voting thresholds are tuned separately for each site. This baseline uses multi-view evidence but aggregates foreground support at individual pixels rather than selecting module instances using their geometry.

MV-PVMapper uses the original perspective images, camera geometry and DSM; it does not use the DOM as a segmentation input. Thus, DOM-IterSeg assesses the orthophoto route, whereas MV-PixelVote assesses pixel-level aggregation of the original multi-view predictions. All three methods run two feedback rounds after the initial text-only pass and are compared at $t = 2$ . In each feedback round, DOM-IterSeg uses the bounding boxes of the previous round’s polygons, converted to DOM pixel coordinates, as box prompts. MV-PixelVote counts the projected polygons covering each cell of a 0.02 m object-space grid. Cells with at least 20, 27 and 30 votes at Sites 1, 2 and 3, respectively, are retained, and their connected components are converted to polygons. Polygons smaller than $0 . 6 \mathrm { m ^ { 2 } }$ are removed.

## 5.3. Main Results and Cross-Site Comparison

Table 2 compares the three mapping strategies. MV-PVMapper achieves an AJI of 97.55% and an $A P _ { 9 5 }$ of 95.67%, combining near-complete instance recovery with accurate boundaries. The baselines reveal two diferent limitations. DOM-IterSeg retains high area precision but recovers only 60.77% of the reference area. MV-PixelVote recovers more coverage, yet its AJI falls to 41.24% because foreground agreement does not ensure that neighbouring modules remain separate.

Table 2: Module-mapping accuracy of MV-PVMapper and the two baselines at $t = 2 .$ Values (%) are macro-averages over the three sites. Higher values are better; bold marks the highest value in each column.
<table><tr><td></td><td colspan="4">Area-level</td><td colspan="5">Instance-level</td></tr><tr><td>Method</td><td> $\mathrm { \ I o U _ { a r e a } }$ </td><td> $F _ { 1 }$ </td><td> $\mathrm { P _ { a r e a } }$ </td><td> $\mathrm { R _ { a r e a } }$ </td><td>RQ</td><td>SQ</td><td>PQ</td><td>AJI</td><td> $A P _ { 9 5 }$ </td></tr><tr><td>DOM-IterSeg</td><td>60.73</td><td>72.58</td><td>99.80</td><td>60.77</td><td>73.93</td><td>96.97</td><td>71.66</td><td>60.73</td><td>60.51</td></tr><tr><td>MV-PixelVote</td><td>86.11</td><td>92.19</td><td>96.79</td><td>89.11</td><td>49.40</td><td>96.03</td><td>47.64</td><td>41.24</td><td>27.52</td></tr><tr><td>MV-PVMapper</td><td>97.55</td><td>98.76</td><td>99.00</td><td>98.52</td><td>99.99</td><td>97.56</td><td>97.56</td><td>97.55</td><td>95.67</td></tr></table>

Figure 4a shows that DOM-IterSeg retains high area precision but has much lower area recall at Sites 1 and 3, whereas MV-PixelVote has comparatively high area IoU and $F _ { 1 }$ but markedly lower RQ, AJI and $A P _ { 9 5 }$ at Sites 2 and 3. The reference-module bars in Figure 4b explain these contrasting profiles: DOM-IterSeg misses 670 and 2 480 modules at Sites 1 and 3, respectively, while MV-PixelVote merges or splits many reference modules, including 461 merged modules at Site 1 and 561 split modules at Site 2. The separate prediction bars also reveal surplus MV-PixelVote predictions, particularly at Site 2. MV-PVMapper maintains high scores across all three sites, with almost entirely matched reference-module bars and no unmatched predictions.

![](images/8ef70331d633d34e009f671430a59e60855de1314a6adb3b7f682e83afc023cc.jpg)  
Figure 4: Per-site comparison of the three methods at t = 2. (a) Metric profiles; IoU, F1, P and R are the area-level metrics of Table 2, and the dashed circle marks 50%. (b) Reference-module outcomes and prediction-side false positives. Reference and prediction bars are normalised by their respective populations. Numbers below the reference bars give matched, merged, split and missed counts; numbers below the prediction bars give unmatched predictions (FP).

Across all three sites, MV-PVMapper matches 7 710 of the 7 711 reference modules, with no false positives and no merged or split modules. The sitewide maps in Figure 5 show the diferent spatial patterns of baseline errors: DOM-IterSeg leaves scattered gaps at Site 1 and misses large portions of the three roof-wide arrays at Site 3, while MV-PixelVote covers more of these arrays but merges neighbouring modules, including 3 050 reference modules at Site 3.

![](images/68452efc5eaa7efaa70751dd9bc8bb822a62db04e34a3a8cc34f9bc8a190dfb2.jpg)  
Figure 5: Site-wide module maps at t = 2, overlaid on the DOM. Columns show Sites 1–3 and rows show the reference modules, DOM-IterSeg, MV-PixelVote and MV-PVMapper.

The paired crops in Figure 6 make the instance errors visible: MV-PixelVote fills narrow inter-module gaps at Sites 1 and 3 and produces fragmented predictions at Site 2, where DOM-IterSeg retains mostly separate modules. MV-PVMapper preserves both the site-wide array pattern and the individual module contours, showing why foreground coverage alone is insuficient to assess module mapping.

![](images/ded21986cac6d33144478ef9df8a77c48745b83141ed5668715292992f6f70a0.jpg)  
Figure 6: Detail views of the module maps in Figure 5. Each site is shown by two 5 m-scale crops; rows show the reference modules, DOM-IterSeg, MV-PixelVote and MV-PVMapper.

## 5.4. Ablation Studies

The ablations isolate contour projection, the module-geometry prior and feedback routing. Each changes one component of the full framework while retaining the remaining configuration. In each ablation, the configuration labelled ours corresponds to the MV-PVMapper result in Table 2. Results are evaluated after two feedback rounds, except for the no-feedback control, and the tables report macro-averages over the three sites.

## 5.4.1. Contour Projection

This experiment tests whether a module-specific surface model improves contour transfer and subsequent instance selection. We compare:

• Collinearity-based projection: intersect each contour vertex’s viewing ray directly with the DSM.

• Afine-based projection: fit a two-dimensional afine transformation from DSM-derived image–object correspondences within each candidate’s image bounding box. This adapts the afine-mapping idea of MVMapper (Chen et al., 2024b) to individual candidates.

• Slope correction projection (ours): fit a local module plane from visible DSM points that project into the mask and intersect the contour rays with that plane.

Table 3 shows that slope correction projection reaches 97.55% area IoU and 95.67% $A P _ { 9 5 }$ , compared with 97.24% and 93.46% for afine-based projection and 89.94% and 46.56% for collinearity-based projection.

Table 3: Ablation of contour projection. All configurations use the full geometry prior and feedback to all covering views and are evaluated at t = 2. Values (%) are macro-averages over the three sites; bold marks the highest value in each column.
<table><tr><td></td><td colspan="4">Area-level</td><td colspan="5">Instance-level</td></tr><tr><td>Projection</td><td>IoUarea</td><td> $F _ { 1 }$ </td><td>P area</td><td>Rarea</td><td>RQ</td><td>SQ</td><td>PQ</td><td>AJI</td><td> $A P _ { 9 5 }$ </td></tr><tr><td>Collinearity-based projection</td><td>89.94</td><td>94.52</td><td>96.78</td><td>92.56</td><td>96.85</td><td>94.45</td><td>91.54</td><td>87.70</td><td>46.56</td></tr><tr><td>Affine-based projection</td><td>97.24</td><td>98.60</td><td>98.82</td><td>98.38</td><td>99.98</td><td>97.28</td><td>97.26</td><td>97.24</td><td>93.46</td></tr><tr><td>Slope correction projection (ours)</td><td>97.55</td><td>98.76</td><td>99.00</td><td>98.52</td><td>99.99</td><td>97.56</td><td>97.56</td><td>97.55</td><td>95.67</td></tr></table>

Figure 7a–c shows that collinearity-based projection mainly reduces boundary accuracy (lower SQ and $A P _ { 9 5 } )$ at Sites 1 and 3. Site 2 additionally exhibits a pronounced detection failure: 15.44% of reference modules are missed and 2.41% of predictions are false positives, with RQ falling to 90.56% and $A P _ { 9 5 }$ to 19.82%. The geometric reasons for this site-specific failure are further examined through the module-level 3D geometry and elevation profiles in Figure 9. Slope correction recovers all modules at Site 2 without false positives and raises $A P _ { 9 5 }$ to 93.20%.

![](images/e39165f5b772a620f51a7ce84d9e99cd77b0e23c9d81697552cece86f41e7308.jpg)  
Figure 7: Per-site results of the contour-projection ablation at $t = 2 .$ (a) RQ and SQ; (b) $A P _ { 9 5 } ; { \mathrm { ( c ) } }$ missed reference modules (Missed/GT) and false positives among the predictions $\left( \mathrm { F P / P r e d } \right)$ .

Figure 8 distinguishes boundary discrepancies from whole-module omissions. At Sites 1 and 3, FP and FN pixels are concentrated along module perimeters, while most module interiors remain correctly covered. At Site 2, the collinearity-based result contains entire module interiors labelled as FN, whereas afine-based and slope correction projection recover these regions. The residual diferences between the latter two methods occur mainly in narrow boundary bands, illustrating why similar instance-recovery rates can coexist with diferences in $A P _ { 9 5 }$

![](images/7fea13daf8a3796518f12969d3439407b924372ee19ee700c679b95b39808a3c.jpg)  
Figure 8: Object-space contours obtained with the three projection methods (rows) in two windows per site (columns); TP, FP and FN are shown at the pixel level.

The module-level 3D profiles in Figure 9 complement the quantitative results and object-space contours by showing the installation geometry underlying the diferences between sites. At Sites 1 and 3, modules lie close to the roof surface. At Site 2, each rack-mounted row has an upper edge approximately 1.5 m above the roof, producing an abrupt height transition. Collinearity-based projection can then distort an otherwise valid candidate enough to reduce its geometry score. Consistent with this mechanism, it misses 356 modules at Site 2, whereas slope correction projection matches all 2 305.

![](images/389bb70f4e5c0a825099ed2bf271639e3d237cdcbc052203d7d52bfe77868079.jpg)

![](images/1e8dfb9660e456a1f4a4d8c6029e246744bd1cdf16f6bed90c6778a195892ad9.jpg)

![](images/3e47f07772fb55d1cbd93b74caf291b43eb01dc5643f9335ec2f7ec574e176a8.jpg)

![](images/a37cb7d66da11e19819faa1cc5dad813781bcb0d009c500f5d9978ab01bab2d1.jpg)  
Figure 9: Module-level 3D geometry at Sites 1–3 (columns). (a) Recovered module planes above the DSM, coloured by tilt on a separate scale for each site; the yellow dashed line marks the transect. (b) Elevation of the DSM and the module planes along the transect.

## 5.4.2. Module-Geometry Prior

This experiment separates the roles of size and shape in deciding whether a candidate represents one module. The four configurations are:

• No geometry prior: use segmentation confidence without the area or rectangularity term.

• Rectangularity only (+ rect): retain the rectangularity term but remove the area term.

• Area only (+ area): retain the area term but remove the rectangularity term.

• Full prior (+ rect + area; ours): combine both terms as in MV-PVMapper.

Without the geometry prior, area recall remains 98.09%, but RQ falls to 5.16% and AJI to 19.63% (Table 4). The area term supplies the missing instance-scale constraint: on its own, it restores RQ to 99.95%. Combining it with rectangularity further improves $A P _ { 9 5 }$ from 90.81% to 95.67%.

Table 4: Ablation of the module-geometry prior. + rect and + area each use only the corresponding term. All configurations use slope correction and feedback to all covering views and are evaluated at $t = 2$ . Values (%) are macro-averages over the three sites; bold marks the highest value in each column.
<table><tr><td></td><td colspan="4">Area-level</td><td colspan="5">Instance-level</td></tr><tr><td>Geometry prior</td><td> $\mathrm { I o U } _ { \mathrm { a r e a } }$ </td><td> $F _ { 1 }$ </td><td> $\mathrm { P _ { a r e a } }$ </td><td> $\mathrm { R _ { a r e a } }$ </td><td>RQ</td><td>SQ</td><td>PQ</td><td>AJI</td><td> $A P _ { 9 5 }$ </td></tr><tr><td>No geometry prior</td><td>72.99</td><td>84.38</td><td>74.05</td><td>98.09</td><td>5.16</td><td>91.91</td><td>4.84</td><td>19.63</td><td>3.80</td></tr><tr><td>+rect</td><td>95.34</td><td>97.60</td><td>98.18</td><td>97.04</td><td>84.03</td><td>97.09</td><td>81.64</td><td>83.69</td><td>87.54</td></tr><tr><td>+ area</td><td>97.20</td><td>98.58</td><td>98.63</td><td>98.53</td><td>99.95</td><td>97.29</td><td>97.24</td><td>97.19</td><td>90.81</td></tr><tr><td>+ rect + area (ours)</td><td>97.55</td><td>98.76</td><td>99.00</td><td>98.52</td><td>99.99</td><td>97.56</td><td>97.56</td><td>97.55</td><td>95.67</td></tr></table>

Without the geometry prior, Figure 10a shows prediction counts 5.82– 7.21 times the reference counts despite area $F _ { 1 }$ remaining above 83% at all three sites. Panels (b)–(c) reveal low RQ and AJI alongside extensive merging and splitting, whereas the area term brings prediction counts close to the reference counts and eliminates the reported partition errors. Adding rectangularity to the area term further increases $A P _ { 9 5 }$ at every site, indicating an additional benefit for boundary accuracy beyond recovering the correct number of instances.

![](images/08c740655922ed6c189e3bec4478009ba51ea3b1b2db86033a843cce13984f09.jpg)  
Figure 10: Per-site results of the module-geometry prior ablation at t = 2. (a) Quantity and coverage: number of predictions relative to the number of reference modules (#Pred/#GT) and area F . (b) Instance quality: RQ and SQ (top), AJI and $A P _ { 9 5 }$ (bottom). (c) Partition errors: merged and split reference modules as percentages of the reference count.

Rectangularity alone removes irregular candidates but cannot distinguish one module from a rectangular fragment or a rectangle spanning several modules. The examples in Figure 11 show these scale errors remaining on the arrays despite high area precision.

![](images/8b3ab6ddc3b008b1148bad41dddec3800e5a412ba79979e8cd37350df44ccc94.jpg)  
Figure 11: Module maps of the four geometry-prior configurations (rows) in two windows per site and one window over a non-PV area (columns). Predictions are coloured by outcome as shown in the legend.

The two terms therefore have distinct roles. Area is central to recovering the correct module inventory, while rectangularity improves the selection of well-delineated candidates at the expected scale. Neither strong coverage metrics nor the quality of matched pairs alone adequately describes the failures observed when these constraints are removed.

## 5.4.3. Feedback Routing

This experiment asks whether feedback benefits from cross-view evidence, rather than simply from repeated model calls. We compare four ways of generating and routing prompts:

• No feedback: retain the initial text-only result at t = 0.

• Image-space self re-prompting: derive each image’s boxes from its own predictions in the previous round, without using the fused object-space result.

• Source-view routing: return each accepted object-space instance only to the image that supplied its representative polygon.

• Covering-view routing (ours): return each accepted object-space instance to all images that cover its location.

Table 5 shows that the restricted routes improve instance recovery but do not match the boundary accuracy of covering-view routing. Image-space self re-prompting reaches a higher RQ than source-view routing, yet its SQ remains slightly below the no-feedback result. Their $A P _ { 9 5 }$ values are 79.88% and 86.77%, compared with 95.67% for MV-PVMapper. Thus, repeated prompting or returning a fused result only to its source image does not provide the full benefit of cross-view feedback.

Table 5: Ablation of feedback routing. No feedback is evaluated at $t = 0$ and the other configurations at t = 2. All configurations use slope correction and the full geometry prior. Values (%) are macro-averages over the three sites; bold marks the highest value in each column.
<table><tr><td></td><td colspan="4">Area-level</td><td colspan="5">Instance-level</td></tr><tr><td>Feedback routing</td><td> $\mathrm { I o U } _ { \mathrm { a r e a } }$ </td><td> $F _ { 1 }$ </td><td> $\mathrm { P _ { a r e a } }$ </td><td> $\mathrm { R _ { a r e a } }$ </td><td>RQ</td><td>SQ</td><td>PQ</td><td>AJI</td><td> $A P _ { 9 5 }$ </td></tr><tr><td>No feedback  $( t = 0 )$ </td><td>83.19</td><td>89.52</td><td>98.83</td><td>83.99</td><td>91.05</td><td>96.67</td><td>88.04</td><td>83.18</td><td>71.96</td></tr><tr><td>Image-space self re-prompting</td><td>95.36</td><td>97.62</td><td>98.74</td><td>96.53</td><td>99.35</td><td>96.53</td><td>95.90</td><td>95.32</td><td>79.88</td></tr><tr><td>Source-view routing</td><td>94.80</td><td>97.30</td><td>98.84</td><td>95.86</td><td>98.77</td><td>97.05</td><td>95.86</td><td>94.80</td><td>86.77</td></tr><tr><td>Covering-view routing (ours)</td><td>97.55</td><td>98.76</td><td>99.00</td><td>98.52</td><td>99.99</td><td>97.56</td><td>97.56</td><td>97.55</td><td>95.67</td></tr></table>

Figure 12 distinguishes the two efects by site. Sites 1 and 2 already recover all modules in the initial pass, so feedback mainly improves their boundaries. At Site 3, self re-prompting and source-view routing still miss 117 and 258 modules, respectively, whereas covering-view routing misses only one.

![](images/747d8403251922685fa92f4b6704977e1f21b4b0b39e73dca389f85bb4ae2403.jpg)  
Figure 12: Per-site results of the feedback-routing ablation. No feedback is evaluated at $t = 0$ and the other routes at t = 2. (a) SQ and $A P _ { 9 5 } ;$ (b) missed reference modules (Missed/GT) and false positives among the predictions $\left( \mathrm { F P / P r e d } \right)$

Figure 13 shows the corresponding residual gaps and boundary slivers. Sending verified instances to images that did not initially detect them is therefore important for both recovering dense arrays and refining their contours.

![](images/fc22f389ffe62a52590f21354929211d1aba4050140758d08e11aa2ca3e435c7.jpg)  
Figure 13: Comparison of the four feedback routes (rows) in two windows per site (columns); TP, FP and FN are shown at the pixel level.

## 5.5. Iteration Count and Viewpoint Configuration

This section examines how the number of feedback iterations and the number of camera directions afect MV-PVMapper. The iteration experiment uses the complete five-direction image set and evaluates the results at t = 0, 1 and 2; the t = 2 result is the main-experiment MV-PVMapper configuration in Table 2. The viewpoint experiment uses nested image subsets that start from the nadir images and add the four oblique directions one at a time in a fixed order. All subsets use camera poses and the DSM reconstructed from the complete five-direction survey; only the images used for segmentation vary.

Feedback both recovers missed modules and improves boundary accuracy (Table 6). From $t = 0$ to $t = 2$ , the number of missed modules summed over the three sites falls from 1 537 to 125 and then to one, and the macro-averaged $A P _ { 9 5 }$ rises by 23.71 percentage points, while area precision remains above 98.8%. With a single missed module but an AJI of 97.55% at $t = 2$ , the remaining error lies mainly at the module boundaries. The box prompts of each iteration mark every accepted module in all covering images, so a module that was missed or poorly delineated in one view but accepted from others is segmented again in that view; detection and boundary accuracy therefore improve together.

Table 6: Module-mapping accuracy at $t = 0 ,$ 1 and 2 using all five viewing directions. The $t = 2$ row corresponds to MV-PVMapper in Table 2. Values (%) are macro-averages over the three sites; bold marks the highest value in each column.
<table><tr><td></td><td colspan="4">Area-level</td><td colspan="5">Instance-level</td></tr><tr><td>Iteration</td><td> $\mathrm { I o U } _ { \mathrm { a r e a } }$ </td><td> $F _ { 1 }$ </td><td> $\mathrm { P _ { a r e a } }$ </td><td> $\mathrm { R _ { a r e a } }$ </td><td>RQ</td><td>SQ</td><td>PQ</td><td>AJI</td><td> $A P _ { 9 5 }$ </td></tr><tr><td> $t = 0$ </td><td>83.19</td><td>89.52</td><td>98.83</td><td>83.99</td><td>91.05</td><td>96.67</td><td>88.04</td><td>83.18</td><td>71.96</td></tr><tr><td> $t = 1$ </td><td>96.33</td><td>98.12</td><td>99.01</td><td>97.26</td><td>99.42</td><td>97.43</td><td>96.86</td><td>96.33</td><td>92.44</td></tr><tr><td> $t = 2$ </td><td>97.55</td><td>98.76</td><td>99.00</td><td>98.52</td><td>99.99</td><td>97.56</td><td>97.56</td><td>97.55</td><td>95.67</td></tr></table>

Figure 14 shows that the two efects occur at diferent sites. Sites 1 and 2 have no missed modules at any iteration, so feedback acts there mainly on the boundaries: $A P _ { 9 5 }$ rises by 4.76 and 15.01 percentage points, almost entirely in the first iteration. The second iteration does not change the recovered module counts at these two sites.

![](images/fff66fdd923b45304264fe8e360646c493662c27c821b75c7bf6cc27e10d58fd.jpg)

![](images/3c5717450cb15a1e48f7eeb50cc16eaba0c143f0ccb9aace23c1c6df8c7eb278.jpg)

![](images/8c13a4191c935ac38a31bfcbcc29b1a1c33d7c5a5f55b75f740cbadf7d2b339d.jpg)  
Figure 14: Per-site accuracy of MV-PVMapper at t = 0, 1 and 2. Each panel shows one metric for the three sites; Missed/GT is the percentage of reference modules that are missed. The table under each panel gives the change in percentage points made by the first (0 → 1) and second (1 → 2) iteration. In the Missed/GT panel, the lines of Sites 1 and 2 coincide at zero.

All omissions occur at Site 3, where the initial pass misses 42.29% of the modules, mostly in the interiors of the densely packed arrays that cover the entire roofs of three buildings (Figure 15). Feedback reduces this share to 0.03%, and the second iteration still adds 9.40 percentage points of $A P _ { 9 5 }$ at this site.

![](images/5a68ada6ae2e013ca0ca10bc66ef264f717ffacb8e250b06c18eae3af3e9502f.jpg)  
Figure 15: Module maps at $t = 0 ,$ 1 and 2, overlaid on the DOM of each site. Each site is shown by two crops; TP, FP and FN are shown at the pixel level.

Figure 16 compares the nested viewpoint subsets at $t = 0 .$ , 1 and 2, showing how feedback changes the benefit of additional directions. At $t = 0$ , adding the first oblique direction raises area IoU above 94% at Sites 1 and 2, but even all five directions leave 42.29% of modules missed at Site 3. Feedback substantially reduces these omissions: at Site 3, nadir images alone reach an area IoU of 95.05% at $t = 2$ , compared with 56.08% for all five directions at $t = 0$ . The closer spacing of the $t = 2$ results across viewpoint subsets, especially after the first oblique direction is added, indicates that feedback reduces the dependence on the number of input directions.

![](images/3bbbbb38b7b9aeb2facb44823b87ca421dfff25adf41f2defccf5e91e4d756e0.jpg)  
Figure 16: Per-site accuracy of the nested viewpoint subsets at $t = 0 , 1$ and 2. The horizontal axis gives the number of viewpoints: the nadir images alone, followed by one to four oblique directions added in a fixed order. The number at the end of each line is its change in percentage points from one to five viewpoints. Missed/GT is the percentage of reference modules that are missed.

The benefit of a further feedback round nevertheless depends on the site and viewpoint configuration. At Site 3, the $t = 1$ and $t = 2$ curves remain clearly separated in missed-module rate and $A P _ { 9 5 }$ , whereas at Site 2 the nadir-only configuration misses no module at $t = 1$ but 4.16% at $t = 2$ . At t = 2, increasing the number of directions produces only modest changes in mapping accuracy, particularly beyond the first oblique direction, with no consistent increase in $A P _ { 9 5 }$ . A possible explanation is that modules at these three sites are largely unoccluded in oblique views, so additional directions ofer limited opportunities to reveal previously hidden module regions.

## 5.6. Implications, Applicability and Limitations

On the three evaluated rooftop plants, a frozen VFM combined with photogrammetric geometry and limited dimensional information produces a nearly complete module inventory with accurate boundaries. The geometry prior does not replace image interpretation; it helps select candidates at the scale of one physical module, while covering-view feedback allows reliable observations to improve other views. The resulting georeferenced instances provide the spatial objects needed to associate inspection findings with individual assets. Here, training-free means that no task-specific model training or weight updates are performed; it does not mean that the method requires no site information.

The input requirement also defines the scope of deployment. MV-PVMapper is designed for overlapping oblique UAV imagery with reconstructed camera poses and a DSM, rather than an isolated image or an orthophoto alone. Photogrammetric reconstruction is already used in UAV-based PV mapping and inspection workflows (Bommes et al., 2022; Cardoso et al., 2024). Where such a survey is available, the method reuses the redundant source observations and reconstructed geometry; its additional site-specific input is a representative module size. The acquisition and reconstruction efort remains necessary where these data do not yet exist. Each feedback round processes every sliding window again, making the present implementation more suitable for ofline map generation than real-time inspection.

The evaluated sites do not cover rooftops with highly irregular installations and several substantially diferent module specifications. A single representative size may then reject valid modules or admit fragments of larger ones. Lower image resolution, severe occlusion and larger installation tilts also require further evaluation.

We use SAM 3 as a representative model with the required prompting interface rather than compare a large set of segmentation backbones. The present experiments isolate the contribution of photogrammetric coordination while holding the backbone fixed. Near-complete recovery on these sites leaves limited scope for distinguishing models by instance count alone. Later or alternative promptable models may still improve boundary accuracy or reduce inference cost.

## 6. Conclusion

This study presents MV-PVMapper, a training-free framework for extracting georeferenced PV module instances from UAV multi-view images. It connects prompt-driven SAM 3 segmentation with object-space geometric verification and returns the fused modules to covering views as spatial prompts. Per-module slope correction transfers image contours through a locally coherent DSM surface, while the module-geometry score guides candidate selection, cross-view association and feedback. The resulting workflow links the complementary observations of a photogrammetric survey to a common module map without updating the segmentation-model weights.

On three rooftop plants containing 7 711 reference modules, two feedback iterations produce 7 710 matched instances with no false positives and one false negative. The framework achieves a macro-averaged AJI of 97.55% and an $A P _ { 9 5 }$ of 95.67%, compared with 60.73% and 60.51% for DOM-IterSeg and 41.24% and 27.52% for MV-PixelVote. The ablations support the combined roles of covering-view feedback, local surface correction and geometric screening. These findings indicate that a pretrained image model can support module-level mapping when its predictions are assessed and coordinated using photogrammetric geometry.

## Acknowledgements

This work is supported by the National Natural Science Foundation of China under Grant 42371475.

## References

Barraz, Z., Sebari, I., Ait El Kadi, K., Ait Abdelmoula, I., 2025a. Towards a holistic approach for UAV-based large-scale photovoltaic inspection: A review on deep learning and image processing techniques. Technologies 13, 117.

Barraz, Z., Sebari, I., Ait El Kadi, K., Lamrini, N., Ait Abdelmoula, I., 2024. One to segment them all: A data-based domain generalization approach for solar module segmentation using thermal UAV images, in: 2024 10th International Conference on Control, Decision and Information Technologies (CoDIT), IEEE. pp. 1751–1755.

Barraz, Z., Sebari, I., Lamrini, N., Ait El Kadi, K., Ait Abdelmoula, I., 2025b. Fast and automatic solar module geo-labeling for optimized large-scale photovoltaic systems inspection from UAV thermal imagery using deep

learning segmentation. Cleaner Engineering and Technology 28, 101048. doi:10.1016/j.clet.2025.101048.

Bodla, N., Singh, B., Chellappa, R., Davis, L.S., 2017. Soft-NMS – improving object detection with one line of code, in: Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 5561–5569. URL: https://arxiv.org/abs/1704.04503.

Bommes, L., Buerhop-Lutz, C., Pickel, T., Hauch, J., Brabec, C., Peters, I.M., 2022. Georeferencing of photovoltaic modules from aerial infrared videos using structure-from-motion. Progress in Photovoltaics: Research and Applications 30, 1122–1135.

Cardoso, A., Jurado-Rodríguez, D., López, A., Ramos, M.I., Jurado, J.M., 2024. Automated detection and tracking of photovoltaic modules from 3D remote sensing data. Applied Energy 367, 123242.

Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., et al., 2025. SAM 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 .

Chen, K., Liu, C., Chen, H., Zhang, H., Li, W., Zou, Z., Shi, Z., 2024a. RSPrompter: Learning to prompt for remote sensing instance segmentation based on visual foundation model. IEEE Transactions on Geoscience and Remote Sensing 62, 1–17.

Chen, Q., Gan, W., Tao, P., Zhang, P., Huang, R., Wang, L., 2024b. End-toend multiview fusion for building mapping from aerial images. Information Fusion 111, 102498. doi:10.1016/j.inffus.2024.102498.

Chen, Q., Li, X., Zhang, Z., Zhou, C., Guo, Z., Liu, Z., Zhang, H., 2023. Remote sensing of photovoltaic scenarios: Techniques, applications and future directions. Applied Energy 333, 120579. URL: https://www.sciencedirect.com/science/article/pii/ S0306261922018360, doi:10.1016/j.apenergy.2022.120579.

Dai, A., Nießner, M., 2018. 3DMV: Joint 3D-multi-view prediction for 3D semantic scene segmentation, in: Proceedings of the European Conference on Computer Vision (ECCV), pp. 452–468. URL: https://arxiv.org/ abs/1803.10409.

Deng, R., Guo, Z., Zhang, P., Li, J., Xu, X., Chen, Q., Chen, Y., Yan, J., 2026. PANEL: A photovoltaic-specific vision–language model for zero-shot and few-shot PV interpretation tasks in remote sensing. ISPRS Journal

of Photogrammetry and Remote Sensing 238, 337–355. doi:10.1016/j. isprsjprs.2026.05.012.

Fischler, M.A., Bolles, R.C., 1981. Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM 24, 381–395. doi:10.1145/358669. 358692.

Gallardo-Saavedra, S., Hernández-Callejo, L., Duque-Perez, O., 2018. Technological review of the instrumentation used in aerial thermographic inspection of photovoltaic plants. Renewable and Sustainable Energy Reviews 93, 566–579. URL: https://www.sciencedirect.com/science/article/ pii/S1364032118303708, doi:10.1016/j.rser.2018.05.027.

Huang, R., Guo, J., Yue, H., Lu, X., Yang, J., 2026. PVP-Adapter: Dualbranch spatial–frequency adaptation for change detection of photovoltaic panels. ISPRS Journal of Photogrammetry and Remote Sensing 240, 360– 377. doi:10.1016/j.isprsjprs.2026.07.032.

Kirillov, A., He, K., Girshick, R., Rother, C., Dollár, P., 2019. Panoptic segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9404–9413. doi:10.1109/CVPR.2019. 00963.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al., 2023. Segment anything, in: Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015–4026.

Kozák, V., Chudoba, J., Přeučil, L., 2026. Detailed aerial mapping of photovoltaic power plants through semantically significant keypoints. International Journal of Engineering and Geosciences 11, 352–362. doi:10. 26833/ijeg.1737764.

Kumar, N., Verma, R., Sharma, S., Bhargava, S., Vahadane, A., Sethi, A., 2017. A dataset and a technique for generalized nuclear segmentation for computational pathology. IEEE Transactions on Medical Imaging 36, 1550–1560. doi:10.1109/TMI.2017.2677499.

Lawin, F.J., Danelljan, M., Tosteberg, P., Bhat, G., Khan, F.S., Felsberg, M., 2017. Deep projective 3D semantic segmentation, in: Computer Analysis of Images and Patterns, pp. 95–107. doi:10.1007/978-3-319-64689-3\_8.

de l’Epine, M., Dominguez, G., Kaizuka, I., Jäger-Waldau, A., 2026. A Snapshot of Global PV Markets. Technical Report T1-49:2026. IEA Photovoltaic Power Systems Programme. URL: https://iea-pvps.org/ snapshot-reports/snapshot-2026/, doi:10.69766/WXLW1389.

Li, K., Zhang, S., Deng, Y., Wang, Z., Meng, D., Cao, X., 2025. SegEarth-OV3: Exploring SAM 3 for open-vocabulary semantic segmentation in remote sensing images. arXiv preprint arXiv:2512.08730 .

Long, J., Shelhamer, E., Darrell, T., 2015. Fully convolutional networks for semantic segmentation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3431–3440. doi:10.1109/ CVPR.2015.7298965.

Rafaeli, O., Svoray, T., Nahlieli, A., 2025. SinkSAM-Net: Knowledge-driven self-supervised sinkhole segmentation using topographic priors and Segment Anything Model. ISPRS Journal of Photogrammetry and Remote Sensing 228, 1–15. doi:10.1016/j.isprsjprs.2025.06.035.

Ren, S., Luzi, F., Lahrichi, S., Kassaw, K., Collins, L.M., Bradbury, K., Malof, J.M., 2024a. Segment anything, from space?, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 8355–8365.

Ren, T., Liu, S., Zeng, A., Lin, J., Li, K., Cao, H., Chen, J., Huang, X., Chen, Y., Yan, F., et al., 2024b. Grounded SAM: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159 .

Robert, D., Vallet, B., Landrieu, L., 2022. Learning multi-view aggregation in the wild for large-scale 3D semantic segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5575–5584. URL: https://arxiv.org/abs/2204.07548.

Wang, C., Chen, J., Meng, Y., Deng, Y., Li, K., Kong, Y., 2024. SAM-PolyBuild: Adapting the Segment Anything Model for polygonal building extraction. ISPRS Journal of Photogrammetry and Remote Sensing 218, 707–720. doi:10.1016/j.isprsjprs.2024.09.018.

Wang, M., Cui, Q., Sun, Y., Wang, Q., 2018. Photovoltaic panel extraction from very high-resolution aerial imagery using region–line primitive association analysis and template matching. ISPRS Journal of Photogrammetry and Remote Sensing 141, 100–111.

Wang, S., Cai, B., Luo, Z., Hou, D., Zheng, X., Liu, J., Shao, Z., 2026. Edgepreserving and uncertainty-guided domain adaptation for weakly supervised rooftop photovoltaic mapping. ISPRS Journal of Photogrammetry and Remote Sensing 240, 31–53. doi:10.1016/j.isprsjprs.2026.07.008.

Xu, M., Yin, X., Qiu, L., Liu, Y., Tong, X., Han, X., 2025. SAMPro3D: Locating SAM prompts in 3D for zero-shot instance segmentation, in: International Conference on 3D Vision (3DV). arXiv:2311.17707.

Yan, M., Zhang, J., Zhu, Y., Wang, H., 2024. MaskClustering: View consensus based mask graph clustering for open-vocabulary 3D instance segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). arXiv:2401.07745.

Yao, X., Zhang, S., Liang, Z., Li, J., Liu, C., 2026. PVSAM: Adapting geometric prompts to Segment Anything Model for photovoltaic detection in remote sensing imagery. Applied Energy 404, 127137.

Yin, Y., Liu, Y., Xiao, Y., Cohen-Or, D., Huang, J., Chen, B., 2024. SAI3D: Segment any instance in 3D scenes, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). arXiv:2312.11557.