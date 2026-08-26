# ROI-Gated SAHI: Content-Adaptive Slicing-Based Inference for Efficient Object Detection

Rashid Riyadh<sup>1</sup>, Abd Ullah Khan<sup>2,3</sup>, Imad Gohar<sup>4,∗</sup>, Muzammil Behzad<sup>5,6,∗</sup>

<sup>1</sup> Faculty of Information Technology, City University, Malaysia

<sup>2</sup> Department of Information Convergence Engineering, Kyung Hee University, South Korea

<sup>3</sup> Department of Computer Science, National University of Sciences and Technology, Pakistan

<sup>4</sup> School of Computing and Artificial Intelligence, Sunway University, Malaysia

<sup>5</sup> King Fahd University of Petroleum and Minerals, Saudi Arabia

<sup>6</sup> SDAIA-KFUPM Joint Research Center for Artificial Intelligence, Saudi Arabia

Abstract—Slicing-Aided Hyper Inference (SAHI) improves small object detection in high-resolution images but often spends substantial compute on background tiles. We propose region-ofinterest (ROI)-Gated SAHI, an inference-time framework that introduces a lightweight proposer to localize foreground regions and restrict sliced refinement to informative areas. We evaluate the framework in two settings. On the COCO128 full split dataset comprising 128 images, static ROI-gating is slower on average than Full SAHI, achieving a speed ratio of 0.88, and yields a lower mAP@0.5 of 0.6602 compared with 0.7569 for Full SAHI. A simple adaptive routing policy with τ = 0.4 reduces the mean latency, achieving a slight gain of 1.02× over Full SAHI. On a three-image sparse-to-dense case study, ROI-gating achieves speedups ranging from 0.96× to 6.90× with a mean speedup of 3.41×. These results show that ROI-gating is most beneficial in sparse scenes and requires policy-based routing for robust average behavior.

Index Terms—object detection, sliced inference, SAHI, efficiency, YOLO, edge computing.

## I. INTRODUCTION

Detecting small objects in high-resolution images is a challenging problem due to the trade-off between spatial resolution and computational efficiency [1]. For such detection, the fine-grained spatial details must be maintained, which makes processing images at high resolutions inevitable [2]. However, processing such high-resolution images using conventional object detectors is computationally expensive, leading to memory cost and inference latency [3]. These constraints make it challenging to deploy object detectors in real-world particularly on resource-constrained platforms such as mobile devices, embedded systems, aerial platforms, and edge computing environments [4].

To address this challenge, slicing-based inference has recently emerged, which improves small object detection without architectural changes or retraining. Slicing-Aided Hyper Inference (SAHI) [5] is one of the slicing-based inference approaches. SAHI first partitions high-resolution images into overlapping tiles. Next, it applies object detection independently to each tile, and then merges predictions through postprocessing. On one hand, by increasing the size of small objects within each slice, SAHI enhances detection accuracy. On the other hand, due to computationally efficiency, it maintains compatibility with off-the-shelf detectors. However, the computational cost of such slicing-based inference approaches increases as the number of tiles increases. As a result, substantial overhead is generated when it is uniformly applied across the entire image.

![](images/b2d86aea99b185b112a3645b0311c1c0cc8738224629abbc520310a9d06f5acc.jpg)  
Fig. 1: ROI-Gated SAHI pipeline: (top) baseline full-image patching; (bottom) proposer, ROI merge, coverage check with fallback, and selective refinement. Red dashed box marks injected ROI-gating components.

In addition to the overhead problem, another challenge associated with the existing slicing-based methods is that they perform tiled inference without discriminating the image content. As a result, background regions having little or no relevant information are processed with the same computational cost as foreground, information-rich regions. This approach becomes particularly computation-inefficient in spatially sparse or clustered objects, where a large fraction of inference time is spent on redundant computation [6]. This renders slicing-based approaches inapplicable for applications with latency and real-time constraints. To circumvent these challenges, inference strategies that adapt computation based on scene content are required.

Given the above challenges, an effective slicing-based technique is required that can reduce unnecessary computation in background regions while maintaining high accuracy [7]. Second, the technique must be focused on inference time, and must not require retraining or architectural changes of detectors, thereby facilitating deployment flexibility [8]. Third, its performance should remain robust across diverse scene densities, including cases where foreground objects densely occupy the image [9]. Finally, the technique should be seamlessly compatible with existing slicing frameworks and detectors [10].

To this end, we propose a content-adaptive inference framework called region-of-interest (ROI)-Gated SAHI. The framework conditions sliced inference on foreground presence to satisfy the above mentioned requirements. For this purpose, it introduces a lightweight ROI estimation that identifies candidate foreground regions before performing the computationintensive tiled inference process. Specifically, it avoids slicing the entire image, by performing slicing-based refinement on the selectively applied regions, i.e., regions with overlapping ROIs. This way, it bypasses background-only areas. To ensure robust behavior in dense scenes where foreground coverage is high, an intelligent fallback mechanism dynamically reverts to conventional full-image SAHI when selective gating becomes counterproductive.

The proposed framework is designed as an external inference-time system that remains fully compatible with existing detectors and slicing pipelines. By decoupling ROI reasoning from detector architectures, ROI-Gated SAHI enables adaptive computation without compromising detection completeness or requiring model retraining. This design reframes inference efficiency as a content-adaptive routing problem rather than a model-level optimization, opening a new direction for efficient deployment of slicing-based object detection systems.

The contributions are summarized as follows.

## A. Contributions

• We propose ROI-Gated SAHI, an inference-time framework that integrates lightweight ROI estimation with slicing-based inference, enabling selective activation of computationally expensive tiled refinement only in foreground-relevant regions. This design directly addresses redundant computation in background-dominated scenes while remaining fully compatible with off-theshelf object detectors. The framework is enhanced by introducing an intelligent fallback strategy, which enables the framework to dynamically reverts to conventional full-image SAHI when the foreground size exceeds a threshold. This way, a stable and predictable performance across diverse scenes is achieved, and performance degradation in dense or cluttered scenarios is mitigated.

• Through experimental evaluations, we exhaustively analyse the relationship between ROI coverage, tile count reduction, and inference latency using multiple image resolutions and foreground densities. We further provide insights into when ROI-gated inference should be applied for optimal results. Additionally, end-to-end latency, detection accuracy, and qualitative analysis of all the inference pipeline stages are provided to demonstrate the performance improvement of the proposed framework in

terms of computational efficiency and detection accuracy. The rest of the paper is organized as follows. Section II presents the related work. Section III details the proposed framework. Section IV presents results and discussion, and Section V concludes the paper.

## II. RELATED WORK

## A. Small Object Detection in High-Resolution Images

Small object detection in high-resolution images has been a crucial challenge in computer vision. Specifically, the low pixel representation, foreground–background imbalance, and large-scale spatial downsampling in convolutional networks make it challenging to detect small objects in such images [2]. These challenges particularly impact aerial, satellite, and surveillance images, where objects of interest often occupy only a small fraction of the entire image [11]. The existing literature deals with this problem by making architectural enhancements [12]–[14]. For instance, multi-scale feature pyramids is used in [15], context aggregation modules in used in [16], attention mechanisms is utilized in [17], and loss reweighting schemes is introduced in [18]. All these approaches improve feature representation and detection accuracy for small objects. However, they do not consider the computational cost associated with high-resolution inputs, which occurs when inference is performed densely over the entire image. Consequently, these methods cannot be efficiently applied for latency-sensitive or resource-constrained applications.

## B. Slicing-Based Inference for Small Object Detection

Slicing-based inference is an effective approach to improve small object detection without changing the detector’s architecture. Utilizing this idea, a SAHI framework is proposed that converts high-resolution images into overlapping tiles, applies object detection independently on each tile, and merges predictions through post-processing [5]. Specifically, small objects within each slice are enlarged to enable feature extraction, facilitating high accuracy gains on various detectors and datasets.

However, SAHI brings about computational overhead. For instance, the inference cost increases with increase in the number of tiles. This overhead leads to high inefficiency when the image contains sparse scenes, or when most of the image contains only background. To address this problem, adaptive SAHI (ASAHI) [19] is proposed that dynamically adjusts slice size or the number of tiles based on image resolution. Though, this approach reduces redundancy up to some extent, it uniformly applies slicing on the entire image, irrespective of foreground and background regions. As a result, the background regions still get unnecessary inference related computation. Another SAHI-based framework called S<sup>3</sup>AHI is proposed in [20], which generates high-resolution patches, and improves detection of small objects during testtime adaptation. The framework uses a teacher–student model to refine predictions. Moreover, semantic correlation graphs are constructed over sliced detections to enhance consistency. The enhanced features, however, introduce additional computational and architectural complexity to the conventional SAHI.

## C. ROI Mechanisms in Object Detection

ROI methods are crucial in object detection as they focus on image regions that contain objects. In two-stage detectors, such as Faster R-CNN [21], this is done using region proposal networks (RPNs), which first identifies potential object regions. These regions are then refined to produce the final detections. These methods improve detection accuracy and proposal quality. However, they have to perform high inference-time computations for high-resolution images. This is because the feature extraction process is performed over the entire image. The latest literature explores ROI-based filtering and foreground–background separation to improve efficiency, especially in remote sensing and edge-computing applications. Techniques such as background suppression [22], foreground enhancement [23], and ROI-based preprocessing have shown remarkable performance to reduce data transfer and processing costs. However, these methods require modifications in the detector’s architecture, or retraining/task-specific tuning, which limit their applicability to the existing available models. Contrarily, ROI-Gated SAHI introduces ROI reasoning at the inference pipeline level, which is external to the detector. Hence, it saves computations, and bypasses changes in the model’s architecture or retraining.

## D. Positioning of ROI-Gated SAHI

While slicing-based methods improve accuracy and ROIbased methods aim to reduce internal feature redundancy, existing approaches do not explicitly integrate lightweight foreground estimation with slicing-based inference to selectively activate expensive tiled inference [24]. ROI-Gated SAHI bridges this gap by combining (1) Lightweight ROI estimation to approximate foreground distribution at negligible cost; (2) Selective slicing, in which background-only tiles are skipped entirely; and (3) A robustness-aware fallback mechanism that reverts to full-image SAHI under dense foreground conditions [25]. This design allows ROI-Gated SAHI to achieve substantial inference-time speedups in sparse scenes while preserving detection completeness in dense scenarios [26]. Unlike prior work that uniformly applies slicing or requires architectural changes, our framework adapts computation dynamically to scene content and remains fully compatible with existing detectors [27].

The existing research on small object detection either improves accuracy at the expense of increased computation or reduces complexity through architectural redesign and retraining. However, the proposed ROI-Gated SAHI framework differs from these approaches by formulating efficiency as a content-adaptive inference problem. Specifically, it exploits foreground sparsity to gate slicing operations, aiming to reduce the redundant computation in the background-dominated regions while enhancing the accuracy benefits of slicing-based inference. This makes the proposed framework suitable for deployment in resource-constrained systems.

Motivated by this research gap, we propose a contentadaptive inference-time optimization of sliced object detection framework, called ROI-Gated SAHI, that dynamically adjusts slicing configuration according to the presence of foreground. During this operation, the framework doesn’t require detector retraining or changes to the basic detector architecture. By incorporating lightweight ROI reasoning and an adaptive routing mechanism, the framework seeks to improve computational efficiency, preserve detection accuracy, and ensure practical deployment within a unified single system.

## III. METHODOLOGY

## A. Overview of ROI-Gated SAHI

The proposed framework reduces redundant computation in the background regions by selecting foreground regions for processing. Unlike baseline SAHI, which applies uniform tiling across the entire image, our framework uses adaptive gating that restricts the processing to ROI only. The framework consists of multiple stages, including a lightweight proposer for fast localization and a high-resolution refiner for accurate detection. To maintain steady performance in dense scenarios, an adaptive fallback mechanism is also incorporated. Figure 1 illustrates the architectural difference between baseline SAHI and the proposed ROI-Gated SAHI framework.

The proposed framework keeps the standard training procedure intact, and leverages pre-trained models from Ultralytics [28], initialized with COCO pre-trained weights. Both YOLOv8n and YOLOv8s are used without any additional training or fine-tuning. YOLOv8n, with 3.2M parameters, serves as a fast proposer for ROI generation, while YOLOv8s, with 11.2M parameters, acts as a high-quality refiner. The proposed ROI-gated SAHI is incorporated exclusively during the inference stage.

## B. Inference Pipeline

The complete methodology consists of five sequential stages executed during inference, as illustrated in Figure 2.

Stage A. In Stage A, referred to as the proposer stage, the objective is to efficiently generate candidate object regions from the input image. The pipeline begins with a highresolution image $I \ \in \ \mathbb { R } ^ { H \times W \times \hat { 3 } }$ , which is downsampled to a fixed resolution before inference, i.e.,

$$
\begin{array} { r } { I ^ { \prime } = \mathcal { D } ( I ) , \quad I ^ { \prime } \in \mathbb { R } ^ { 4 1 6 \times 4 1 6 \times 3 } } \end{array}\tag{1}
$$

The downsampled image $I ^ { \prime }$ is then processed by a region proposer to outputs a set of candidate region proposals, i.e.,

$$
\pmb { { B } } = \{ b _ { i } , c _ { i } \} _ { i = 1 } ^ { n }\tag{2}
$$

where $b _ { i }$ denotes the predicted bounding boxes and $c _ { i }$ represents their corresponding confidence scores. The resulting proposals provide coarse estimates of object locations with minimal computational overhead. At this stage, enhancing recall is prioritized over precision, so that as many foreground regions are captured as possible before refinement in the

![](images/3c96831d684bae40098610f5c77d70acd657b0db3d91174a8bc100219bd05dc1.jpg)  
Fig. 2: Complete overview of the proposed ROI-Gated SAHI framework, illustrating the sequential flow from Stage A to Stage E. The pipeline begins with high-resolution image input, followed by Stage A (Proposer), where a lightweight model generates coarse ROI candidates. Stage B computes ROI coverage and determines the routing criterion. Based on the threshold condition, the pipeline adaptively switches between Stage C (ROI-Gated SAHI), which selectively processes foreground regions, and Stage D (Full-SAHI), which processes the entire image grid. Finally, Stage E performs global detection fusion by combining outputs from the proposer and refiner branches using NMS to produce the final detections.

subsequent stages.

Stage B. This is the decision stage, and is used for consolidation and coverage estimation. Here, the redundant proposals are removed using Non-Maximum Suppression (NMS), i.e.,

$$
\mathrm { I o U } ( b _ { i } , b _ { j } ) = { \frac { | b _ { i } \cap b _ { j } | } { | b _ { i } \cup b _ { j } | } }\tag{3}
$$

Boxes with $\mathrm { I o U } > 0 . 5$ are suppressed, yielding a refined set $B ^ { * }$ . Each bounding box is then expanded by a margin $\alpha =$ 0.15. Consequently,

$$
w ^ { \prime } = ( 1 + 2 \alpha ) w , \quad h ^ { \prime } = ( 1 + 2 \alpha ) h .\tag{4}
$$

For a box with width w and height h, the expanded dimensions become $w ^ { \prime } = 1 . 3 w$ and $h ^ { \prime } = 1 . 3 h ,$ , centered on the original box center. The lightweight proposer may produce slightly tight bounding boxes that crop object edges. A 15% margin ensures objects near ROI boundaries remain fully visible to the refiner. As a result, adequate contextual coverage is ensured while reducing boundary artifacts. The total ROI coverage is computed as a ratio of the aggregated ROI areas to the input image area, and is used as a metric to guide adaptive routing.

$$
R = { \frac { \sum _ { i } \mathrm { A r e a } ( R O I _ { i } ) } { \mathrm { A r e a } ( I ) } }\tag{5}
$$

Stage C. This stage selectively processes informative regions while reducing unnecessary computation. It consists of three main steps: (i) ROI-based patch selection focusing on foreground regions, (ii) reduced tile processing compared to Full-SAHI, and (iii) refinement using a high-capacity detector.

If the value of R is less than a specific threshold $\tau ,$ , i.e, $R < \tau .$ , selective SAHI is applied only within ROI regions. Each ROI is partitioned into overlapping tiles of size 640×640 pixels, with an overlap of $5 0 \% - 7 5 \%$ to preserve contextual continuity. Let N denote the total number of tiles generated in Full-SAHI over the entire image. Since only a fraction of the image is processed, the number of tiles in the ROI-Gated setting is approximated as

$$
N _ { \mathrm { R O I } } \approx R \cdot N ,\tag{6}
$$

where $N _ { \mathrm { R O I } }$ represents the number of tiles processed within the ROI regions, and $R \in [ 0 , 1 ]$ is the ratio of total ROI area to the full image area. Each tile is then processed using YOLOv8s to produce high-resolution detections.

Stage D. This stage complements Stage C, and is activated when comprehensive coverage of the image is required. Unlike the ROI-Gated approach, Full-SAHI processes the entire image without any spatial filtering. In this setting, the full image is partitioned into a complete grid of overlapping tiles, ensuring that no region is skipped. Each tile is processed independently, enabling thorough exploration of dense and complex scenes where objects may be distributed across the entire image. All generated patches are passed through the refiner model, YOLOv8s, to produce high-resolution detections. Stage E. In this stage, detections obtained from both the proposer and refiner branches are consolidated to produce the final output. The combined detection set is defined as

$$
\mathcal { D } _ { f i n a l } = \mathrm { N M S } ( \mathcal { D } _ { p r o p } \cup \mathcal { D } _ { r e f } ) ,\tag{7}
$$

where $\mathcal { D } _ { p r o p }$ and $\mathcal { D } _ { r e f }$ denote the detections from the proposer and refiner stages, respectively. NMS is applied on the unified set with an IoU threshold of approximately 0.45 to eliminate redundant overlapping detections and retain the most confident predictions.

## C. Model Latency and Analysis

Let t, C, and R respectively represent the per-tile inference time, constant overhead, and ROI ratio, we have,

$$
L _ { \mathrm { F u l l } } = N \cdot t\tag{8}
$$

$$
L _ { \mathrm { R O I } } = C + R \cdot N \cdot t\tag{9}
$$

where $L _ { \mathrm { F u l l } }$ and $L _ { \mathrm { R O I } }$ represent latency in Full SAHI and ROI-Gated SAHI, respectively. The break-even analysis of ROI ratio defines the threshold below which ROI-gating becomes more efficient than Full SAHI in terms of latency. Hence,

$$
R _ { \mathrm { b r e a k - e v e n } } = 1 - { \frac { C } { N \cdot t } }\tag{10}
$$

## D. Fallback Strategy

A threshold $\tau = 0$ .40 is used in this stage, i.e., if $R \geq 0 . 4 0 .$ Full SAHI is applied; otherwise ROI-Gated SAHI is applied. The value of τ is selected from empirical latency calibration on the COCO128 full split to improve average-case robustness. The selected value of $\tau$ accounts for proposer recall degradation in dense scenes, variability in computational overhead, and the effects of resolution-dependent tile distribution. The value of τ is empirically selected to balance computational efficiency and detection reliability.

The proposed method reuses the same trained detectors in complementary roles, employing a fast proposer for coarse localization and a high-resolution refiner for precise detection. This way, computational overhead is reduced by selectively applying full inference only when necessary, while preserving detection fidelity, particularly in sparse scenes.

## E. Principle of ROI-Gated Inference

The ROI-Gated SAHI pipeline described in Section III is utilized in the inference stage. First, the input image is processed by the proposer to identify candidate ROI. Next, the overlapping regions are merged. Depending on the ROI coverage, slicing is applied only to selected regions using SAHI. Subsequently, the detections obtained from the processed patches are combined using global NMS. The SAHI-based slicing is applied only during inference. Figure 3 illustrates this process, where slicing is restricted to ROI regions rather than the entire image. This way, the associated computation is reduced while keeping preserving detection rate.

As shown in Fig. 3, only ROI regions are processed instead of the entire image. In the left panel, the full SAHI (baseline) divides the entire image into overlapping patches, and, as a result, all the regions are processed. In the right panel, which represents our ROI-Gated approach, a fast proposer first identifies a candidate region (green dashed boundary), and patches are generated only within that region. The detected object is shown by the red bounding box (rectangle). The figure clearly highlights the difference between processing the entire image and processing only the selected regions.

![](images/00c7970d305a0645e5560aa21c1534e390b818ac36e44665c6ce923321e0657e.jpg)  
Fig. 3: Principle of ROI-Gated SAHI sliced inference. Left: Baseline SAHI processes the full image grid (blue). Right: ROI-Gated SAHI first identifies a candidate region (green dashed boundary) and then processes only ROI tiles. Both paths detect the same object (red rectangle) in this example, illustrating how selective tiling reduces computation cost in sparse scenes.

## F. Implementation Details

To implement the proposed framework, we developed a Python-based inference pipeline using publicly-available object detection and slicing libraries. The developed inference pipeline is reproducible across different hardware types, and is outlined in Algorithm 1.

As this work focuses on inference-time optimization, we utilize YOLOv8n and YOLOv8s from Ultralytics [28] as the proposer and refiner, respectively. Both of the models were initialized with weights pre-trained on the COCO dataset. Py-Torch is used for fast inferential computations, while NumPy and OpenCV are used for image preprocessing and bounding box operations. Detection accuracy and execution time were used as performance indicators for all experiments.

## G. Evaluation Strategy

Two types of evaluation settings are adopted to ensure a comprehensive evaluation. The first one is a dataset-based evaluation and the second one is a case-study-based evaluation. The former is conducted on the full COCO128 dataset. This dataset contains on 128 images. This evaluation highlights the overall system performance. The second evaluation is based on three high-resolution images representing sparse, moderatedense (called moderate for brevity), and full-dense (called dense for brevity) object distributions. This configuration is essential to carry out a detailed analysis of latency and detection behavior under diverse spatial environments. These two evaluation settings are reported separately to avoid conflating aggregate dataset-level statistics with localized behavior observed in specific scene configurations.

## Algorithm 1 ROI-Gated SAHI

Input: Input image I, threshold τ , proposer $M _ { p } ,$ , refiner M<sub>r</sub> Output: Final detections $D _ { \mathrm { f i n a l } }$

1: Generate candidate detections using the fast proposer:

$$
D _ { p } \gets M _ { p } ( I )
$$

2: Merge and expand proposal boxes to obtain ROI regions:

$$
R O I  \mathrm { M e r g e E x p a n d } ( D _ { p } )
$$

3: Compute the ROI coverage ratio:

$$
R \gets \frac { \mathrm { A r e a } ( \cup _ { i } R O I _ { i } ) } { \mathrm { A r e a } ( I ) }
$$

4: if $R < \tau$ then

5: Generate SAHI tiles only from ROI-overlapping regions:

$$
T \gets \mathrm { R O I - G a t e d S A H I } ( I , R O I )
$$

6: else

7: Generate SAHI tiles over the full image:

$$
T \gets \mathrm { F u l l S A H I } ( I )
$$

8: end if

9: Refine selected tiles using the high-resolution detector:

$$
D _ { r } \gets M _ { r } ( T )
$$

10: Fuse proposer and refiner detections:

$$
D _ { \mathrm { f i n a l } }  \mathrm { N M S } ( D _ { p } \cup D _ { r } )
$$

11: return $D _ { \mathrm { f i n a l } }$

## IV. RESULTS AND DISCUSSION

## A. Quantitative Results

The proposed method is evaluated on the COCO128 test set and compared against standard SAHI-based inference. The comparison is conducted from three complementary perspectives, namely overall detection accuracy, inference efficiency in terms of processing speed, and robustness under varying object sparsity conditions in high-resolution images. To provide a more fine-grained analysis, we further consider three representative scenarios corresponding to dense, moderate, and sparse object distributions. These cases are used to systematically assess the behavior of the proposed approach under different spatial complexity levels.

## B. Overall Accuracy

Table I shows that the static ROI-Gated SAHI configuration is overall slower than Full SAHI, with a speed ratio of 0.88 (298.24 ms vs. 263.73 ms) also shown in Figure 4, while achieving lower detection accuracy in terms of mAP@0.5 (0.6602 vs. 0.7569). The observed latency improvement is limited to sparse scenes $( R < 0 . 3 0 )$ , whereas both moderate and dense regimes consistently exhibit higher inference time compared to Full SAHI on average. In contrast, the adaptive routing strategy with $\tau = 0 . 4 0$ improves overall efficiency, reducing the average latency to 258.46 ms as shown in Figure 4, corresponding to a 1.02× speedup relative to Full SAHI, by selectively routing 26 out of 128 images to the ROIbased processing branch.

Figure 5 presents a regime-wise comparison of mean latency between the Full SAHI and ROI-Gated approaches across sparse, moderate, and dense subsets, as well as the overall dataset, with the corresponding speed ratio (Full/ROI) overlaid. In the sparse regime, ROI-Gated processing achieves a clear latency advantage, yielding a speed ratio greater than 1.0 and demonstrating the benefit of selectively processing only limited regions of interest. However, as the scene complexity increases, this advantage diminishes: in the moderate and dense regimes, the ROI-Gated approach incurs higher latency than Full SAHI, reflected by speed ratios below 1.0. This reason for this degradation is the increased number of regions required to be processed. Processing such a higher number of regions gives rise to routing and coordination overheads. The performance gap is more visible in dense scenes. This shows that ROI-based approach doesn’t perform better when the majority area inside the image contains relevant objects. The overall results also show this fact, i.e., on average, ROI-Gated processing slightly performs lower than Full SAHI. The results imply that ROI-based routing is highly efficient in sparse scenes. Also, they show that ROI-based routing brings high gains in sparse regions but becomes less advantageous as image density increases.

Figure 6 shows the system performance as a function of the routing threshold $\tau .$ Both the mean adaptive latency and the number of images routed to the ROI pipeline are shown in the figure. As can be seen, as $\tau$ increases, the number of images for ROI processing increases, also increasing the number of routed images. Aside from this, the mean latency initially decreases because low-confidence regions are processed in a selective fashion. This decrease touches a minimum around $\tau \approx 0 . 5$ . Thereafter, the mean latency increases because the increased number of routing introduces extra computational overhead. The results shows that trade-off exists between selective processing and routing cost. The small value of threshold limits efficiency gains, while its high value increases efficiency. Our selected operating point $( \tau = 0 . 4 0 )$ is near the optimal region. This operating point achieves near-minimal latency. It also maintains stable routing behavior. As a result, it provides a good balance between performance and computational cost.

Figure 7 shows how the per-image speed ratio (Full SAHI / ROI) changes as a function of ROI coverage. Each dot/point represents an image. The horizontal dotted line at the speed ratio of 1 represents the break-even point. The value above this point indicates that ROI-based approach is more efficient while values below this point indicates that full SAHI processing is more efficient. The vertical dotted lines divide the data into low-, medium-, and high-coverage regions. As can be noted from the figure, for images at low ROI coverage, most points lie above the break-even line. This implies that ROI-based approach yields consistent speed improvements.

This is because, ROI-based approach restricts the required computations to a small subset of the images. As ROI coverage increases, the acquired speed improvement becomes smaller. In the medium coverage regions, the results are more diverse. In this region, the points are distributed on both sides of the break-even line.

In the high coverage region, most points lie below the break-even line. This implies that the extra overheads resulting from ROI extraction and routing exceeds its benefits, which makes full SAHI more faster and efficient. Overall, these results suggest taht the ROI-GAted SAHI is more effective in sparse scenes. Its effectiveness decreases as ROI coverage increases, i.e., when a large part of the image requires detailed processing. In such a case, the advantages of selective routing decreases.

![](images/a4798856a24f10533cc1d77d02a8fa1057f3d42720d521923e47d59128abc907.jpg)  
Fig. 4: Latency Comparison: average latency for Full SAHI, Static ROI-Gated SAHI, and Adaptive routing (threshold 0.40). This figure compares the average runtime of the three options on the full dataset. Static ROI-Gated is slower overall, while Adaptive routing is the fastest on average because it uses ROI only when ROI area is small.

![](images/d34fdb661987c2a740dbad66f3609b3a76b18ed1ca7965e1ec512a3fd7ad36ac.jpg)  
Fig. 5: COCO128 full split (128 images): adaptive threshold sweep showing mean adaptive latency and number of images routed to ROI. This figure shows how performance changes when we move the routing threshold. As threshold changes, the number of images sent to ROI changes, and latency follows a curve. The paper setting (0.40) gives near-best latency while keeping routing behavior stable.

TABLE I: Quantitative evaluation on COCO128. Speed is reported as mean Full/ROI latency (> 1 favors ROI-Gated). ROI Faster Count reflects per-image gains (Adaptive: routing). $R = \{ 0 . 3 0 , 0 . 7 5 \}$ $\tau = 0 . 4 0 .$ F1 and Mean IoU report agreement with Full SAHI.
<table><tr><td>Setting</td><td>Images ROI (%)</td><td></td><td>Full mAP@0.5</td><td>ROI mAP@0.5</td><td>Full Lat. (ms)</td><td>ROI Lat. (ms)</td><td>Speed</td><td>ROI Faster Count</td><td>Agreement F1</td><td>Agreement Mean IoU</td></tr><tr><td>Overall</td><td>128</td><td>68.79</td><td>0.7569</td><td>0.6602</td><td>263.73</td><td>298.24</td><td>0.88</td><td>45/128</td><td>0.7828</td><td>0.9100</td></tr><tr><td>Sparse regime  $( R < 0 . 3 0 )$ </td><td>18</td><td>&lt; 30</td><td>0.6459</td><td>0.3630</td><td>182.42</td><td>154.38</td><td>1.18</td><td>11/18</td><td>0.6981</td><td>0.7856</td></tr><tr><td>Moderate regime  $( 0 . 3 0 \leq R < 0 . 7 5 )$ </td><td>45</td><td>30-75</td><td>0.7265</td><td>0.5605</td><td>250.27</td><td>285.22</td><td>0.88</td><td>20/45</td><td>0.6727</td><td>0.9108</td></tr><tr><td>Dense regime (  $R \geq 0 . 7 5 )$ </td><td>65</td><td>&gt; 75</td><td>0.8063</td><td>0.7712</td><td>295.56</td><td>347.09</td><td>0.85</td><td>14/65</td><td>0.8824</td><td>0.9439</td></tr><tr><td>Adaptive policy  $( \tau = 0 . 4 0 )$ </td><td>128</td><td>dynamic</td><td>0.7569</td><td>0.7305</td><td>263.73</td><td>258.46</td><td>1.02</td><td>routed 26/128</td><td>0.9381</td><td>0.9638</td></tr></table>

Note: Full SAHI values are shown in the Full mAP@0.5 and Full Lat. columns. Agreement metrics (F1, Mean IoU) compare ROI-Gated outputs against Full SAHI outputs. In the Adaptive row, ROI mAP@0.5 and agreement metrics are computed for the mixed routing policy output.

![](images/c167875a08d3cb06e06734626dd449a08a8cbd6c713f897b7230a596a165d0e7.jpg)  
Fig. 6: Thresholding for τ : adaptive threshold sweep showing mean adaptive latency and number of images routed to ROI. This figure shows how performance changes when we move the routing threshold. As threshold changes, the number of images sent to ROI changes, and latency follows a curve. The paper setting (0.40) gives near-best latency while keeping routing behavior stable.

![](images/1f583fe350a4d9ad7281b2c0bc6ba2e0a88ef7ba16cbdc9040d1b4218874435b.jpg)  
Fig. 7: Speed Ratio: per-image speed ratio versus ROI coverage, with break-even and regime boundaries. Each point is one image. Points above 1.0 mean ROI is faster; below 1.0 mean Full SAHI is faster. The plot shows that ROI tends to help more when ROI coverage is low, and tends to lose advantage as ROI coverage increases.

TABLE II: Three-image case-study frame statistics and detection counts.
<table><tr><td>Image</td><td>Resolution</td><td>Full detection</td><td>ROI detection</td><td>Speed (ms)</td><td>Speedup (x)</td><td>ROI%</td></tr><tr><td>image1</td><td>1200×800</td><td>102</td><td>72</td><td>341/355</td><td>0.96</td><td>69.9</td></tr><tr><td>image2</td><td>1536×1024</td><td>13</td><td>4</td><td>579/244</td><td>2.38</td><td>26.4</td></tr><tr><td>image3</td><td>1024×1536</td><td>2</td><td>1</td><td>527/76</td><td>6.90</td><td>2.7</td></tr></table>

TABLE III: Aggregate performance summary for the threeimage case study.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Frames Tested Mean Speedup</td><td>3 3.41×</td></tr><tr><td>Mean ROI Coverage</td><td>33.0%</td></tr><tr><td>Total Full SAHI Detections</td><td>117</td></tr><tr><td>Total ROI-Gated Detections</td><td>77</td></tr><tr><td>Detection Agreement (Mean F1)</td><td>0.544</td></tr></table>

## C. Performance Dynamics Across ROI Regimes

To evaluate the efficiency of the proposed method under varying scene complexities, we conducted an analysis on three representative high-resolution images from the COCO128 dataset. These images were selected to represent three distinct operating regimes: sparse, moderate, and dense Region of Interest (ROI) coverage. We quantify the performance using latency, speedup, ROI coverage, and detection agreement as summarized in Tables II, III, IV and V.

1) Latency and Speedup Analysis: The evaluation demonstrates that ROI-Gated SAHI achieves a mean speedup of 3.41× across the selected test cases with an average ROI coverage of 33.0% (see Table III). As illustrated in Figure 8, the proposed method significantly reduces inference latency in sparse and moderate scenes.

For the sparse scene (image3.jpeg), where the ROI covers only 2.7% of the image, latency drops from 526.52 ms to 76.25 ms, yielding a 6.90× speedup. In the moderate scene (image2.jpeg) with 26.4% ROI coverage, latency is reduced by more than half, from 579.01 ms to 243.69 ms (2.38× speedup). Conversely, in the dense scene (image1.jpeg) where ROI coverage reaches 69.9%, the overhead of the gating mechanism results in latency comparable to the baseline (0.96× speedup).

The relationship between foreground density and efficiency is further explored in Figure 9. The data confirms that speedup is inversely proportional to ROI coverage; maximum computational gains are achieved when the target objects occupy a small portion of the total image area, thereby allowing the system to skip a larger number of tiles.

TABLE IV: Per-image latency and ROI coverage.
<table><tr><td>Image</td><td>Full latency (ms)</td><td>ROI latency (ms)</td><td>Speed (×)</td><td>ROI (%)</td></tr><tr><td>img1.jpeg</td><td>342</td><td>355</td><td>0.96</td><td>69.9</td></tr><tr><td>img2.jpeg</td><td>579</td><td>244</td><td>2.38</td><td>26.4</td></tr><tr><td>img3.jpeg</td><td>526.52</td><td>76.25</td><td>6.90</td><td>2.7</td></tr></table>

![](images/7c308a7a5c298cc671564319edbb2a35c1239bf8b3d4f0ab38265871ff2ec4af.jpg)  
Fig. 8: Per-image latency comparison between Full SAHI and ROI-Gated SAHI.

![](images/a7a2661fd71b516b2e7d4c90621b5969271b31cece566fc398250f6f8db1dbb2.jpg)  
Fig. 9: (Speedup vs ROI Coverage) Per-image speedup across different ROI regimes.

2) Detection Agreement and Robustness: To ensure that computational efficiency does not come at the cost of detection quality, we measured the agreement between ROI-Gated SAHI and the Full SAHI baseline. As shown in Table V, while the F1 scores reflect differences in total detection counts (mean F1 of 0.544), the spatial alignment of the resulting bounding boxes remains exceptionally high.

The Mean IoU ranges from 0.87 to 0.98, indicating that when objects are detected, their localization is highly consistent with the full-inference baseline. This suggests that the ROI-Gating mechanism effectively preserves spatial accuracy while successfully discarding redundant background processing.

## D. Qualitative Results

Qualitative results demonstrate that ROI-Gated SAHI focuses computation on relevant regions while preserving spatial localization quality, as depicted in Figures 10–15. The sparsescene efficiency gain is consistent with Fig. 8 and Table IV, where the 2.7% ROI case improves from 526.52 ms to 76.25 ms (6.90× speedup).

TABLE V: Agreement of ROI-gated SAHI vs. full-image SAHI (IoU matching).
<table><tr><td>Image</td><td>F1</td><td>Precision</td><td>Recall</td><td>Mean IoU</td></tr><tr><td>image1.jpeg</td><td>0.49</td><td>0.60</td><td>0.42</td><td>0.89</td></tr><tr><td>image2.png</td><td>0.47</td><td>1.00</td><td>0.31</td><td>0.87</td></tr><tr><td>image3.png</td><td>0.67</td><td>1.00</td><td>0.50</td><td>0.98</td></tr></table>

## E. Discussion

Agreement Metrics. Agreement F1 measures the consistency between Full SAHI and ROI-Gated outputs in terms of detected objects, capturing how often both methods identify the same instances (including missed or additional detections). Agreement Mean IoU, on the other hand, evaluates the spatial alignment of matched detections by quantifying the overlap in bounding box position and size. High agreement means that ROI-Gated SAHI produces results similar to those of Full SAHI. Lower agreement indicates differences resulting from missed detections or changes in object localization. It is worth noting that these metrics compare the two methods with each other. Hence, these metrics do not measure absolute detection performance.

Impact of ROI Coverage: The results show three distinct performance regimes based on coverage. In sparse scenes where the coverage is < 30%, the proposed method achieves the highest speedups, reaching up to 6.90×. This is because only a small part of the image requires processing.

In moderately dense scenes where the coverage is between 30% and 75%, the method still provide stable performance gains, typically between 2 times and 3×. However, the results become more variable as the fraction of image requiring processing increases. In dense scenes with coverage > 75%, the benefits of ROI-gating considerably decreases because most of the image requires processing. Consequently, the effectiveness of selective computation decreases. This regimedependent trend appears consistently in both the aggregate statistics and the case study results.

Efficiency–Accuracy Trade-off: ROI-Gated SAHI reduces computation by requiring fewer tiles to be processed. However, this computation improvement also affects detection accuracy. The main reason is that the lightweight proposer may fail to identify some object regions. On the COCO128 full split, the mAP@0.5 decreases from 0.7569 for Full SAHI to 0.6602 for ROI-Gated SAHI. This suggests that some objects objects or regions are missed/not covered and not passed to the refinement stage, thereby propagating error to the refinement stage. Nevertheless, for objects detected by both methods, the localization quality of the objects for both methods is closely matched. This is indicated by the close alignment in IoU-based metrics.

Role of Adaptive Fallback: Static ROI-gating does not perform well, particularly when ROI coverage is high. To address this issue, the adaptive fallback mechanism is introduced, which dynamically routes high-coverage images to Full SAHI. With $\tau = 0 . 4 0$ , the average speed ratio increases from 0.88× for static ROI-gating to 1.02× relative to Full SAHI. As a result, average-case slowdowns is removed while the the performance gain of ROI-Gating is retained for sparse scenes.

![](images/0ec26f7759e5037d424e327f20d36a87fd17ec14d04b12357a2e702c8a54bf54.jpg)  
(a) Input

![](images/72be640a9ec61bbeed3159d6f8e3c38423ab6ca473ed8906a403ca3054918397.jpg)  
(b) Proposer Detections

![](images/5e1a0f07cb8d31a4887eb1dc22eeab8061e71d5f24b114f0486578895576532f.jpg)  
(c) Merged ROI Regions

![](images/bae9f021f5e34ac0cb8eae23770a3a2cabd85d40c4ce3507722c7820dfa9d490.jpg)  
(d) ROI-Gated SAHI

![](images/e422a20ace22ccd61cc2f405977004380004811fa58f5293b8e92de120c91b58.jpg)  
(e) Full SAHI Baseline

Fig. 10: Case 1. Dense-scene (69.9% ROI) qualitative comparison showing the Input Image, Proposer Detections, Merged RO Regions, ROI-Gated SAHI result, and Full SAHI baseline. ROI-gated processing yields near-baseline latency due to limited tile-saving potential.  
![](images/15bb8a73fd0e4fc95917821cc174da2858a20dbcecb4972445d9433b5c64612f.jpg)  
(a) Input

![](images/585b19aa3a88bb9937e32b39918a26c3fb121065bbc374b38d4932dcf6ad8005.jpg)  
(b) Proposer

![](images/37c62cf396488ae9f4cb080b11c299b0e921b3c343ade95cf2b01204f5486c52.jpg)  
(c) ROI Boxes

![](images/b08b534e5453313356fb61bfdfef5f94cb835f4e618458e971b97a59553faddb.jpg)  
(d) ROI-Gated

![](images/6c96c7f154e8d193ce635d7de9db33a297816567b9b3925464e980e3fb0df93c.jpg)  
(e) Full SAHI

Fig. 11: Case 2. The moderate-density (26.4% ROI), where selective slicing reduces computation while preserving detection localization quality.  
![](images/e8b02f22145c703fbe4547eff9a85fbf72f1bac59e11629179aeb0f95968ce5f.jpg)  
(a) Input

![](images/a10114d91bb60aa15962db92f9ff9411a635e0a8a3f59ba2afdc889fdba92007.jpg)  
(b) Proposer

![](images/3f929773e96667cda94739890e2a7caea973fa1c8e99f0c704594c198e834b8d.jpg)  
(c) ROI boxes

![](images/147a33c08a49ba04ba582c86f2c7a8280f32e8a897b2ffb43c82b6e3337a623e.jpg)  
(d) ROI gated

![](images/e504edc774ad7382a8a1d502401b1d4687ca44ed733108b9d0f5f6669e348e67.jpg)  
(e) ROI SAHI  
Fig. 12: Case 3. Sparse-scene case (2.7% ROI), where skipping background regions produces the largest efficiency gain.

Practical Implications: The proposed approach is suitable for applications where scene density varies and low latency is critical. Particularly, in sparse scenes, the approach can achieve considerable computational saving without reducing detection quality. For datasets with both sparse and dense scenes, adaptive or policy-based routing should be considered to maintain stable performance. By dynamically selecting the most appropriate strategy for each image, such adaptive routing helps avoid regressions associated with static configurations.

Limitations: The proposed framework relies on the recall of the lightweight proposer,. Missed regions at the proposal stage cannot be recovered in the later stages. Additionally, the framework uses fixed thresholds, such as ROI coverage threshold and ROI expansion ratio. These configurations may limit its generalizability, affecting its performance on datasets with different scene characteristics. Moreover, the given evaluations only consider YOLO-based detectors, and does not include alternative detection architectures to validate the generalizability of the proposed framework.

Overall, ROI-Gated SAHI reduces computational cost by limiting processing to selected regions of an image. At the same time, it maintains competitive detection performance. When integrated with adaptive routing, the proposed framework effectively operate under different scene densities. This characteristics make the framework a practical option to resourceconstrained applications.

![](images/87ce84ee096ce9034787f8644a9e3588968a6c6fa24e37a24b9d91541b4a3704.jpg)  
(a) Input

![](images/d39f27781aaff62443b4b994165d776f332d3769b49ced723f3e623a396af6fc.jpg)  
(b) Proposer

![](images/bc0528a17adadf78f6e39ef290ae3bca80fae4b2da8a9f21fdfbfd966d5304ac.jpg)  
(c) ROI Regions

![](images/f6ba0330d4115f753e0af9bd31096b5319c08657a6822b62229a433c9a009613.jpg)  
(d) ROI-Gated

![](images/eefba8bdd03e9c6de0f7389e3ccf8e8d77e17208dda176a6d47c531d685559c1.jpg)  
(e) Full SAHI

Fig. 13: Case 2. Dense-scene qualitative comparison showing the Input Image, Proposer Detections, Merged ROI Regions, ROI-Gated SAHI result, and Full SAHI baseline.  
![](images/b4f59d088e7e8af5c9ac40b59edc11553376cfdbc8e3798e5fd7d4d216b4baaf.jpg)  
(a) Input

![](images/4e4ce3588af358f3230c324a19242cd4e2e2f97c22b74b858aca650ac29e6be9.jpg)  
(b) Proposer

![](images/4f9b8d1532bfc4d83524178feac027537ce59a645f461fc7de55a79f6ba032ce.jpg)  
(c) Merged ROI Regions

![](images/db3cd3efe9407635b93f69c97a69e6f6de30c6de6ee2f6b792e5b7a754b114db.jpg)  
(d) ROI-Gated SAHI

![](images/6001a2951e8684a4c6d6cc55b42247f97f66d7a5b5a4b024034a936d37d6c118.jpg)  
(e) Full SAHI

Fig. 14: Case 1. Dense-scene qualitative comparison showing the input image, proposer detections, merged ROI regions, ROI Gated SAHI result, and Full SAHI baseline.  
![](images/294ccb9591df12338765494ce6f93a52250f302c639ee38a79eea4543789c005.jpg)  
(a) Input

![](images/3afba5ea03aa83dc04d731a2add5cb3dd8c58a2aa4e3b86687df797c3be137bd.jpg)  
(b) Proposer Detections

![](images/39217c5c2dd2eb9e5f665e421278228ffb58be3541e87cd1811237bb8ed88ff4.jpg)  
(c) Merged ROI Regions

![](images/45819e318dc5829e872dcb61467ba94b81cf88e50f0b0015f8cae3dc1f75c9b3.jpg)  
(d) ROI-Gated SAHI

![](images/d9869852be27a73d52a10a6fa3f5cb1abe9f28aabf15870f4bd197fe8f5cc791.jpg)  
(e) Full SAHI Baseline  
Fig. 15: Case 2. Moderate-scene qualitative comparison showing the Input Image, Proposer Detections, Merged ROI Regions, ROI-Gated SAHI result, and Full SAHI baseline.

## V. CONCLUSION

In this paper, we proposed ROI-Gated SAHI, a contentadaptive inference framework for computational-efficient slicing-based object detection. The framework introduces ROI proposer that estimates foreground distribution in an image before selectively applying sliced inference to those regions. This way, the framework reduces the redundant computations required in the conventional full-image SAHI where the entire image is processed, while preserving detection robustness across varying scene densities. For dense scenes, the framework uses an adaptive fallback mechanism, wherein processing is switched to full SAHI when ROI coverage becomes high. Experimental evaluation shows that on the COCO128 full split, static ROI-gating is slower than Full SAHI on average, achieving a high speed ratio of 0.88×, and producing a lower mAP@0.5, decreasing from 0.6602 to 0.7569. The results also show that adaptive routing further improves this trend. Using a threshold of $\tau ~ = ~ 0 . 4 0$ , the average speed ratio increases to 1.02×. This suggests that policy-based adaptive routing can increase average performance. Aside from this, a case study is conducted using three images with sparse, moderate, and dense ROI coverage, in which the speedups is observed between 0.96× and 6.90× with a mean value of mean 3.41×. The largest gains occur in sparse scenes, where only a small fraction of the image requires processing.

These results show that ROI-gated slicing is most effective when scene content is sparse. They also indicate that adaptive routing is important for maintaining stable performance across different scene conditions. Future work will investigate adaptive and learned gating strategies. Additional directions include multi-scale ROI selection, class-aware ROI reasoning, and the application of ROI-gated inference to video data. These extensions may further improve efficiency and robustness in practical deployment scenarios.

## REFERENCES

[1] W. Zhu and K. Chen, “Real-time object detection for unmanned aerial vehicles based on vision transformer and edge computing,” Scientific Reports, vol. 16, no. 1, p. 6814, 2026.

[2] M. Nikouei, B. Baroutian, S. Nabavi, F. Taraghi, A. Aghaei, A. Sajedi, and M. E. Moghaddam, “Small object detection: A comprehensive survey on challenges, techniques and real-world applications,” Intelligent Systems with Applications, vol. 27, p. 200561, 2025.

[3] A. Dede, H. Nunoo-Mensah, E. T. Tchao, A. S. Agbemenu, P. E. Adjei, F. A. Acheampong, and J. J. Kponyo, “Deep learning for efficient highresolution image processing: A systematic review,” Intelligent Systems with Applications, vol. 26, p. 200505, 2025.

[4] C. Kong, F. Li, X. Yan, J. Yang, P. Mo, Q. Luo, and R. Mao, “Object detection on low-compute edge socs: a reproducible benchmark and deployment guidelines,” Scientific Reports, vol. 16, no. 1, p. 5875, 2026.

[5] F. C. Akyon, S. O. Altinuc, and A. Temizel, “Slicing aided hyper inference and fine-tuning for small object detection,” in 2022 IEEE International Conference on Image Processing (ICIP), pp. 966–970, 2022.

[6] J. Liu, Y. Chen, X. Ye, Z. Tian, X. Tan, and X. Qi, “Spatial pruned sparse convolution for efficient 3d object detection,” in Advances in Neural Information Processing Systems, vol. 35, pp. 6735–6748, Curran Associates, Inc., 2022.

[7] S. V. Mahadevkar, B. Khemani, S. Patil, K. Kotecha, D. R. Vora, A. Abraham, and L. A. Gabralla, “A review on machine learning styles in computer vision—techniques and future directions,” IEEE Access, vol. 10, pp. 107293–107329, 2022.

[8] Y. Bi, B. Xue, P. Mesejo, S. Cagnoni, and M. Zhang, “A survey on evolutionary computation for computer vision and image analysis: Past, present, and future trends,” IEEE Transactions on Evolutionary Computation, vol. 27, no. 1, pp. 5–25, 2023.

[9] X. Pan, T. T. Yang, J. Li, C. Ventura, C. Malaga-Chuquitaype, C. Li, R. K. L. Su, and S. Brzev, “A review of recent advances in data-driven computer vision methods for structural damage evaluation: algorithms, applications, challenges, and future opportunities,” Archives of Computational Methods in Engineering, vol. 32, no. 7, pp. 4587–4619, 2025.

[10] R. Vogg, T. Lueddecke, J. Henrich, S. Dey, M. Nuske, V. Hassler, D. Murphy, J. Fischer, J. Ostner, O. Schuelke, et al., “Computer vision for primate behavior analysis in the wild,” Nature Methods, vol. 22, no. 6, pp. 1154–1166, 2025.

[11] W. Hua and Q. Chen, “A survey of small object detection based on deep learning in aerial images,” Artificial Intelligence Review, vol. 58, no. 6, p. 162, 2025.

[12] S. Wu, H. Yang, L. Liao, C. Song, Q. Liu, J. Fu, and T. Li, “Dynamic small object feature enhancement and detection for remote sensing images,” Scientific Reports, vol. 15, no. 1, p. 37225, 2025.

[13] A. Kos, K. Majek, and D. Belter, “Enhanced lightweight detection of small and tiny objects in high-resolution images using object trackingbased region of interest proposal,” Engineering Applications ofArtificial Intelligence, vol. 153, p. 110852, 2025.

[14] Y. Zhou and Y. Wei, “Uav-detr: An enhanced rt-detr architecture for efficient small object detection in uav imagery,” Sensors, vol. 25, no. 15, 2025.

[15] N. Zeng, P. Wu, Z. Wang, H. Li, W. Liu, and X. Liu, “A smallsized object detection oriented multi-scale feature fusion approach with application to defect detection,” IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1–14, 2022.

[16] M. Yang, H. Bai, J. Hu, and D. Li, “A dynamic context-aware aggregation strategy for small object detection,” Pattern Recognition, vol. 170, p. 112127, 2026.

[17] H. Gong, T. Mu, Q. Li, H. Dai, C. Li, Z. He, W. Wang, F. Han, A. Tuniyazi, H. Li, X. Lang, Z. Li, and B. Wang, “Swin-transformerenabled YOLOv5 with attention mechanism for small object detection on satellite images,” Remote Sensing, vol. 14, no. 12, 2022.

[18] Z. Chen, C. Xu, H. Zhu, Y. Li, and W. Yang, “Refocal: Addressing learning imbalances for accurate tiny object detection in aerial imagery,” IEEE Geoscience and Remote Sensing Letters, vol. 22, pp. 1–5, 2025.

[19] H. Zhang, C. Hao, W. Song, B. Jiang, and B. Li, “Adaptive slicing-aided hyper inference for small object detection in high-resolution remote sensing images,” Remote Sensing, vol. 15, no. 5, 2023.

[20] H. Ding, G. Yang, X. Tu, Y. Huang, and X. Ding, “S3AHI: Sourcefree domain adaptive small object detection with slicing aided hyper inference,” in 2024 International Joint Conference on Neural Networks (IJCNN), pp. 1–8, 2024.

[21] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time object detection with region proposal networks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 6, pp. 1137– 1149, 2017.

[22] K. Naganuma and S. Ono, “Robust foreground-background separation for severely-degraded videos using convolutional sparse representation modeling,” 2025.

[23] J. Zhang, X. Zhang, S. Liu, B. Pan, and Z. Shi, “Fie-net: Foreground instance enhancement network for domain adaptation object detection in remote sensing imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–14, 2025.

[24] M. Muzammul, A. Algarni, Y. Y. Ghadi, and M. Assam, “Enhancing uav aerial image analysis: Integrating advanced SAHI techniques with realtime detection models on the visdrone dataset,” IEEE Access, vol. 12, pp. 21621–21633, 2024.

[25] L. Li, L. Liu, F. Cheng, Y. He, and Z. Zhong, “Cn-unet: Convnext unet with slicing-aided hyper segmentation for infrared small target detection,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 19, pp. 84–98, 2025.

[26] J. Suarez-Ram´ ´ırez, D. Santana-Cedres, and N. Monz´ on, “DAHI: a fast´ and efficient density aided hyper inference technique for large scene object detection,” Pattern Recognition, p. 112228, 2025.

[27] A. Tureckovˇ a, T. Ture´ cek, and Z. K. Oplatkovˇ a, “Artificial size slic-´ ing aided fine tuning (ASSAFT) and hyper inference (ASSAHI) in tomato detection,” Computers and Electronics in Agriculture, vol. 225, p. 109280, 2024.

[28] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” 2023. Available: https://github.com/ultralytics/ultralytics.