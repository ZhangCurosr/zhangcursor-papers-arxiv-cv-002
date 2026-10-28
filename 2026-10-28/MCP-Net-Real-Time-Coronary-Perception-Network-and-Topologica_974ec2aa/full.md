# MCP-Net: Real-Time Coronary Perception Network and Topological Analysis for Robotic Guidewire Navigation

Sahar Nasirihaghighi, Tianyuan Wang, Kenan Niu and Stefano Stramigioli<sup>1</sup>

Abstract— Coronary guidewire navigation during percutaneous coronary intervention requires continuous interpretation of vessel anatomy, guidewire position, and target stenosis under two-dimensional X-ray fluoroscopy. While existing approaches commonly address vessel segmentation or device localization as separate tasks, a robotic system requires a more structured perception of the coronary scene to support downstream navigation. In this work, we present a simulation-driven framework centered on a Multi-Task Coronary Perception Network (MCP-Net) for robotic guidewire navigation, integrating coronary vessel segmentation, vascular topology extraction, and roadmap generation. Starting from three-dimensional coronary CT angiography, we generate simulated fluoroscopy images containing vascular structures, simulated guidewires, stenotic lesions with varying severity, and diverse vascular geometries, together with simulation-derived ground truth. MCP-Net jointly performs coronary vessel segmentation, guidewire-tip localization, and stenosis localization from a single fluoroscopic frame. These predictions are subsequently converted into a structured vascular graph containing centerlines, bifurcations, endpoints, branch directions, and branch angles. By associating the predicted stenosis with this graph, the framework generates a roadmap from the automatically selected graph root toward the target lesion. The results demonstrate that the proposed framework can jointly recover coronary perception targets and transform them into an interpretable vascular representation suitable for downstream robotic navigation.

## I. INTRODUCTION

Coronary artery disease remains a major cause of morbidity and mortality worldwide, and percutaneous coronary intervention (PCI) is one of the primary procedures used to restore blood flow through stenotic coronary arteries [1], [2]. During PCI, the interventionalist must advance a thin guidewire through a complex, branching coronary vasculature toward the target lesion under two-dimensional Xray fluoroscopic guidance. This task requires continuous interpretation of vessel geometry, identification of the lesion, and selection of the appropriate branch at successive bifurcations. The difficulty is compounded by vessel overlap, low image contrast, respiratory and cardiac motion, and the loss of depth information inherent to projection imaging [3], [4]. These challenges have also motivated the development of robotic-assisted PCI (RA-PCI), which enables remote manipulation of coronary guidewires and devices and has demonstrated clinical feasibility while reducing operator radiation exposure [5]–[7]. However, increasing the level of robotic assistance requires reliable intraoperative perception capable of extracting actionable vascular information directly from fluoroscopic images.

For robotic guidewire navigation, however, vessel segmentation alone is not sufficient. A navigation system must reason about both the device and its surrounding anatomy: where the guidewire tip is located, where the target stenosis lies, how the coronary tree branches, and which sequence of vessel segments connects the coronary root to the target. Such information must also be available at interactive rates if it is to support intraoperative robotic decision-making. Existing work has investigated coronary vessel segmentation, guidewire or catheter localization, and vascular analysis [8]– [10], but these components are commonly considered as separate problems. Consequently, even an accurate segmentation mask does not directly provide the structured anatomical representation required for downstream navigation, including vessel connectivity, bifurcation structure, branch orientation, and a path toward the target lesion.

A second challenge is the availability of suitable training and evaluation data. Dense annotation of coronary fluoroscopy is difficult and time-consuming, particularly for thin or overlapping vessels. Precise annotation of guidewire tips, stenosis location, and stenosis severity introduces additional ambiguity, while annotations of centerlines and vascular topology require further expert effort. More importantly, the ground-truth geometry underlying a two-dimensional projection is generally unavailable in real fluoroscopic data. These limitations make it difficult to jointly train and quantitatively evaluate all of the perception and geometric components required by a robotic navigation pipeline.

In this work, we address these challenges through a simulation-driven, multi-task framework for coronary perception, topological analysis, and real-time roadmap generation. Starting from three-dimensional coronary CT angiography, we generate fluoroscopy-like projections containing coronary vessels, simulated guidewires, and stenotic lesions with different severity levels across diverse vascular geometries. Since the underlying scene geometry is known, task-specific ground truth can be generated automatically for vessel segmentation, guidewire-tip localization, stenosis localization, and geometric analysis. These data are used to train the proposed Multi-Task Coronary Perception Network (MCP-Net), which jointly predicts the vessel mask, guidewire-tip location, and stenosis landmarks from a single fluoroscopy frame. The predicted vasculature is subsequently converted into a structured centerline graph containing bifurcations, endpoints, vessel geometry, and branch information, from which a target-oriented vascular roadmap is generated.

This design deliberately separates learned image perception from geometric vascular reasoning. MCP-Net performs the image-dependent tasks jointly in a single forward pass, while centerline extraction, graph construction, and roadmap generation operate directly on the resulting vascular representation. The combination provides not only pixel-level predictions, but also a compact and interpretable description of the coronary anatomy that can serve as a perception interface for downstream robotic planning.

The main contributions of this work are:

• We develop a CT-based coronary fluoroscopy simulation framework that generates synthetic intervention scenes containing coronary vasculature, simulated guidewires, and stenotic lesions with varying severity, together with automatically generated task-specific ground truth.

• We propose MCP-Net, a multi-task coronary perception network that jointly performs coronary vessel segmentation, guidewire-tip localization, and stenosis localization within a unified architecture.

• We introduce a real-time vascular topology and roadmap generation framework that converts the predicted vasculature into a structured graph containing centerlines, vessel calibers, bifurcations, endpoints, and branch geometry, and generates a navigation-oriented path toward the detected stenosis.

• We provide a comprehensive evaluation on the simulated fluoroscopy dataset, including comparisons with established segmentation architectures, vessel segmentation and connectivity, guidewire-tip and stenosis localization, lesion geometry, vascular topology, roadmap generation, and end-to-end computational performance.

The remainder of this paper is organized as follows. Section II reviews related work, and Section III presents the proposed framework. Sections IV and V describe the experimental setup and results, respectively. Section VI discusses the findings, limitations, and future directions, and Section VII concludes the paper.

## II. RELATED WORK

## A. Coronary Vessel Segmentation in X-ray Angiography

Deep learning has substantially improved coronary vessel segmentation in X-ray angiography, with recent work focusing on vessel continuity, temporal information, and reducing dependence on manual annotations. He et al. proposed TVS-Net, which incorporates temporal information from angiographic sequences together with a connectivitypreserving objective [3]. Li et al. further introduced a subpixel-level framework for X-ray coronary angiography, combining vessel landmark localization with topology-aware contour estimation for quantitative vessel analysis [11].

The difficulty of obtaining dense annotations from clinical angiography has also motivated simulation- and selfsupervision-based approaches. Zhang et al. used digitally reconstructed radiographs generated from CT data together with domain randomization and self-supervised learning to train coronary vessel segmentation without manual angiographic annotations [12]. More recently, XA-Sim2Real generates simulated X-ray angiograms from coronary CT angiography and learns adaptive representations between simulated and real images [13]. These approaches demonstrate the value of CT-derived supervision, but primarily target vessel segmentation. In contrast, our simulation pipeline additionally models the interventional guidewire and stenotic lesion, providing simulation-derived supervision for multiple perception tasks and subsequent vascular topology analysis.

## B. Guidewire and Lesion Localization

Reliable localization of interventional devices is another important component of image-guided vascular procedures. Zhang et al. proposed a two-stage real-time framework for guidewire tracking and segmentation in intraoperative X-ray images using spatial-temporal refinement and guidewire-specific feature enhancement [14]. Yao et al. subsequently introduced a Siamese attention network for real-time guidewire-tip tracking [15]. Recent work by Robertshaw et al. integrated catheter/guidewire segmentation, skeletonization, and tip extraction into a real-time fluoroscopic tracking pipeline intended for downstream autonomous navigation [16]. Multi-task learning has also been explored for interventional device perception; Xi et al. jointly performed catheter segmentation and electrode localization using a shared network with multiple prediction heads [17].

Stenosis detection has largely been studied separately from device perception. DeepCoro employs vessel tracking and video-based models for automatic localization and assessment of coronary stenoses [18], while Molenaar et al. recently developed and externally validated deep-learning models for both coronary artery segmentation and significant stenosis detection in invasive coronary angiography [4]. In contrast to these task-specific or paired formulations, MCP-Net jointly estimates the coronary vessel mask, guidewiretip location, and stenosis location from a single fluoroscopy frame using a shared representation.

## C. Vascular Topology and Robotic Endovascular Navigation

Beyond pixel-level segmentation, structured vascular representations are important for anatomical reasoning and robotic navigation. Hampe et al. represented coronary artery trees as graphs for automatic extraction and anatomical labeling in coronary CT angiography [19]. Such centerlineand graph-based representations provide explicit information about vessel connectivity and branching structure that is not directly available from a binary segmentation mask.

In robotic endovascular intervention, Li et al. proposed a vascular centerline-guided autonomous navigation method, demonstrating the utility of centerline geometry for robotassisted navigation [20]. Scarponi et al. further investigated reinforcement learning for autonomous guidewire navigation in unseen vascular geometries [21] and extended this direction to deforming anatomy under single-view fluoroscopic imaging [22]. These works highlight the importance of reliable geometric and device state information for robotic navigation. Our work focuses on producing such information directly from fluoroscopic perception: the predicted vessel mask is converted into a structured graph containing centerlines, bifurcations, endpoints, vessel calibers, and branch geometry, from which a root-to-lesion roadmap is derived.

![](images/a3ba097abcebaef98b91668b1353d6796e1ab655625fa09f789811c408293922.jpg)  
Fig. 1. Overview of the proposed framework. Coronary CT angiography is used to generate simulated fluoroscopy images and task-specific annotations for MCP-Net training. During real-time inference, the trained MCP-Net jointly predicts the vessel mask, guidewire-tip location, and stenosis landmarks from a single fluoroscopy frame. The predicted vasculature is converted into a structured centerline graph and combined with the predicted stenosis target to generate a navigation-oriented roadmap.

## III. METHODS

The proposed framework consists of an offline simulation and training stage followed by a real-time perception and vascular-analysis pipeline, as illustrated in Fig. 1. First, simulated fluoroscopy images and their task-specific annotations are generated from coronary CT angiography (Section III-A) and used to train MCP-Net. During real-time inference, the trained MCP-Net processes each incoming fluoroscopy frame in a single forward pass to predict the coronary vessel mask, guidewire-tip location, and stenosis landmarks (Section III-B). The predicted vessel mask is subsequently converted into a centerline-based vascular graph containing topological and geometric information (Section III-C). The graph is then combined with the localized target to generate a navigation-oriented roadmap toward the stenosis (Section III-D). Section III-E summarizes the complete real-time inference pipeline.

## A. Data Simulation

Real fluoroscopy with dense, per-pixel and per-point annotations is scarce and impractical to label by hand. We therefore synthesize training data from three-dimensional coronary CT angiography [23]. For each CT volume, multiple two-dimensional fluoroscopy-like projections are generated using Tomosipo [24], with the underlying projection operations performed by the ASTRA Toolbox [25]. Vessel centerlines and local radius profiles are extracted using the Vascular Modeling Toolkit (VMTK) [26]. Stenotic lesions are simulated by locally reducing the vessel radius along a selected centerline segment, allowing lesions with different severity levels and locations to be generated. A simulated guidewire is subsequently introduced and advanced toward the stenotic region using the coronary centerline as a geometric reference. Because the underlying anatomy and simulated intervention components are known, exact simulation-derived annotations can be generated automatically for each projection. These include the coronary vessel mask, guidewire-tip location, and stenosis landmarks, together with additional geometric information required for downstream analysis. This simulation stage is performed offline and provides the dataset used to train MCP-Net and evaluate the complete framework.

## B. Multi-Task Coronary Perception Network (MCP-Net)

Given a single fluoroscopy frame I, MCP-Net jointly predicts the coronary vessel mask, guidewire-tip location, and stenosis landmarks. A shared feature representation is followed by three task-specific prediction heads for vessel segmentation, guidewire-tip localization, and stenosis localization. The vessel branch employs a DeepLabV3+ decoder with atrous spatial pyramid pooling (ASPP) to produce a dense binary coronary segmentation. The vessel branch is optimized using the combination of binary cross-entropy and Dice losses,

$$
\mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { B C E } } + \mathcal { L } _ { \mathrm { D i c e } } ,\tag{1}
$$

which balances pixel-wise classification with region overlap and is well suited to the sparse, thin vessel structures.

The two localization branches predict spatial heatmaps for the guidewire tip and stenosis landmarks. Rather than treating localization as pixel-wise classification, we recover continuous image coordinates from each predicted heatmap using a differentiable spatial soft-argmax (DSNT),

$$
\hat { \mathbf { u } } = \sum _ { \mathbf { p } } \mathbf { p } \ \mathrm { s o f t m a x } ( \hat { H } ) _ { \mathbf { p } } .\tag{2}
$$

where $\hat { H }$ denotes the predicted heatmap and p denotes a pixel coordinate. The guidewire branch predicts a single tip location, whereas the stenosis branch predicts three landmarks corresponding to the proximal boundary, lesion throat, and distal boundary, thereby representing both the location and extent of the stenosis. Localization losses are applied only to valid annotated targets.

The complete network is trained end-to-end with the joint objective

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s e g } } + \lambda _ { \mathrm { t i p } } \mathcal { L } _ { \mathrm { t i p } } + \lambda _ { \mathrm { s t e n } } \mathcal { L } _ { \mathrm { s t e n } } ,\tag{3}
$$

where $\lambda _ { \mathrm { t i p } } = \lambda _ { \mathrm { s t e n } } = 5$ in our experiments. At run time, only the trained MCP-Net checkpoint is used, and all three perception outputs are obtained in a single forward pass.

## C. Vascular Topology and Geometric Analysis

The predicted vessel mask is converted into a structured vascular representation using deterministic geometric processing. Small disconnected components are first removed, after which morphological skeletonization reduces the vessel tree to a one-pixel-wide centerline. A Euclidean distance transform of the binary vessel mask is used to estimate the local vessel caliber. For a centerline point p, the local diameter is defined as

$$
d ( \mathbf { p } ) = 2 D ( \mathbf { p } ) ,\tag{4}
$$

where $D ( \mathbf { p } )$ is the Euclidean distance from p to the nearest background pixel.

The extracted centerline is subsequently converted into a vascular graph $\mathcal { G } = ( \nu , \mathcal { E } )$ . Graph nodes represent vessel bifurcations and terminal endpoints, while edges correspond to the centerline segments connecting them. Each edge retains its centerline trajectory together with geometric attributes including segment length and local vessel diameter. Short spurious segments are removed and nearby junctions are consolidated to reduce artifacts introduced by discrete skeletonization.

Local vessel directions are estimated from the centerline geometry around each junction and are used to compute bifurcation and take-off angles. The resulting graph therefore captures both the connectivity and local geometry of the predicted coronary tree and provides the structured representation required for subsequent landmark association and roadmap generation.

## D. Roadmap Generation

The vascular graph and predicted stenosis location are combined to construct an automatic navigation roadmap toward the target lesion. The start point is defined by the automatically selected graph root, which serves as a proxy for the coronary ostium, while the predicted stenosis throat defines the target.

Both locations are projected onto the nearest vessel centerline points. If either lies within a graph edge, a temporary node is inserted at the projected location to preserve the exact start and end positions. A shortest-path search is then performed using vessel-segment arc length as the edge weight, producing a continuous centerline path from the root to the stenosis.

Along the resulting path, traversed bifurcations are identified and local vessel directions are used to compute branch angles and classify each navigation decision as left, right, or straight. The resulting roadmap therefore provides both the target-oriented vessel path and bifurcation-level navigation cues for downstream robotic planning or guidewire control.

## E. Real-Time Inference

During online operation, each fluoroscopy frame is processed by the trained MCP-Net in a single forward pass to obtain the vessel mask, guidewire-tip location, and stenosis location.

The predicted vessel mask is then converted into a centerline-based vascular graph containing the root, bifurcations, endpoints, vessel calibers, and branch geometry. The predicted stenosis throat is associated with this graph, and the roadmap-generation procedure in Section III-D is applied automatically.

The final output consists of the root-to-stenosis path, path length, traversed bifurcations, branch angles, and left/right/straight navigation cues. These outputs can be overlaid with the guidewire-tip and stenosis predictions on the fluoroscopy image. No network training or manual landmark selection is performed during online inference.

## IV. EXPERIMENTAL SETUPS

## A. Dataset

The simulated fluoroscopy dataset is generated from coronary CT angiography volumes of ImageCAS [23] using the procedure described in Section III-A. In total, 376 CT volumes are used, with four simulated projection views generated per volume, resulting in 1,504 images of size 256 × 256 pixels. The split is performed at the CT-volume level to avoid anatomical overlap between sets: 208 volumes (832 images) are used for training, 84 volumes (336 images) for validation, and 84 volumes (336 images) for testing.

Each image is accompanied by simulation-derived annotations including a binary vessel mask, guidewire-tip heatmap, and three stenosis landmarks corresponding to the proximal boundary, throat, and distal boundary. These annotations provide supervision for all perception tasks without manual fluoroscopy labeling.

## B. Implementation Details

MCP-Net uses DeepLabV3+ with a ResNet-50 backbone and 256 × 256 inputs. Training is performed for 40 epochs with batch size 4 using SGD $( \mathrm { l r } = 5 \times 1 0 ^ { - 3 }$ , momentum 0.9, weight decay $1 0 ^ { - 4 } )$ and cosine annealing to $1 0 ^ { - 5 }$ . The segmentation branch uses BCE+Dice loss, while the guidewiretip and stenosis branches use DSNT-based localization losses with $\lambda _ { \mathrm { t i p } } = \lambda _ { \mathrm { s t e n } } = 5$

At inference, vessel probabilities are thresholded at 0.5, while landmark coordinates are recovered using an 11 × 11 local soft-argmax window. The predicted vessel mask is subsequently skeletonized and converted into a vascular graph, from which vessel connectivity, local caliber, and branch geometry are extracted. For the architecture comparison, DeepLabV3+ [27], U-Net [28], SegFormer [29], and FPN [30] use identical multi-task heads, losses, data splits, and training schedules. All experiments were performed on an NVIDIA GB10 GPU with CUDA 13.0.

## C. Evaluation Metrics

Vessel segmentation is evaluated using Dice, IoU, and centerline Dice (clDice). Guidewire-tip and stenosis-throat localization are evaluated using Euclidean pixel error, reported as median and interquartile range (IQR), together with PCK@5 and PCK@10. For stenosis localization, we additionally report the fraction of predicted throat points lying within the ground-truth lesion.

Lesion extent is evaluated using the absolute and relative error of the proximal-to-distal landmark distance. For the subset of stenosis predictions with throat error ≤ 8 pixels, we additionally report the median localization error. Computational efficiency is evaluated using the end-to-end latency and speed of the complete perception, topology, and roadmap pipeline.

## V. EXPERIMENTAL RESULTS

Table I summarizes the test-set performance of the final MCP-Net configuration using DeepLabV3+. For vessel segmentation, it achieved a Dice score of 90.3±3.0%, an IoU of $8 2 . 5 \pm 4 . 8 \%$ , and a clDice score of $9 6 . 0 \pm 3 . 0 \%$ . The higher clDice relative to the region-overlap metrics indicates that the predicted masks preserve the centerline structure of the coronary tree particularly well. This property is important for the subsequent skeletonization and graph construction, where discontinuities in thin vessel branches can directly affect the extracted vascular topology.

Guidewire-tip localization was accurate for most valid test samples, achieving a median error of 1.09 px (IQR: 0.73–1.95 px) and a PCK@5 of 88.1%. The narrow IQR indicates that the majority of tip predictions are concentrated close to the ground-truth location. In comparison, stenosisthroat localization was more challenging, with a median error of 3.82 px (IQR: 1.88–27.1 px), PCK@5 of 52.7%, and PCK@10 of 64.4%. The considerably larger upper quartile indicates the presence of a subset of localization errors.

TABLE I  
PERFORMANCE OF THE DEEPLABV3+-BASED MCP-NET ON THE SIMULATED TEST SET (n = 336). LOCALIZATION ERRORS ARE REPORTED AS MEDIAN (IQR).
<table><tr><td>Task</td><td>Metric</td><td>Value</td></tr><tr><td rowspan="3">Vessel segmentation</td><td>Dice</td><td>90.3 ± 3.0%</td></tr><tr><td>clDice</td><td>96.0 ± 3.0%</td></tr><tr><td>IoU</td><td>82.5 ± 4.8%</td></tr><tr><td rowspan="2">Guidewire tip (n = 327)</td><td>Error</td><td>1.09 (0.73–1.95) px</td></tr><tr><td>PCK@5</td><td>88.1%</td></tr><tr><td rowspan="4">Stenosis throat (n = 320)</td><td>Error</td><td>3.82 (1.88–27.1) px</td></tr><tr><td>PCK@5 / PCK@10</td><td>52.7/64.4%</td></tr><tr><td>Detection rate</td><td>70.6%</td></tr><tr><td>Error if  $\leq 8 ~ \mathrm { p x }$ </td><td>2.13 (1.29–3.23) px</td></tr><tr><td rowspan="2">Lesion length</td><td>Absolute error</td><td>4.60 (1.48–11.30) px</td></tr><tr><td>Relative error</td><td>17.7%</td></tr></table>

![](images/3500b97c9fb245af3b979c26e73d98e8af8640c63957cc18d9bd3dfaebe60d18.jpg)  
Fig. 2. Localization accuracy as a function of pixel tolerance for guidewiretip and stenosis-throat localization on the simulated test set.

Despite this wider error distribution, 70.6% of predicted throat locations fell within the ground-truth lesion extent. This suggests that some predictions identify the correct stenotic region without precisely localizing the lesion throat. For the subset of predictions with throat error $\leq ~ 8$ px, the median error decreased to 2.13 px (IQR: 1.29–3.23 px), showing substantially tighter localization among welllocalized cases. The proximal and distal stenosis landmarks additionally provided an estimate of lesion extent. Across the valid lesion-length cases $( n \ = \ 3 1 6 )$ , the proximal-todistal distance had a median absolute error of 4.60 px and a median relative error of 17.7%.

Fig. 2 further illustrates the different localization behavior of the two tasks. Guidewire-tip accuracy rises rapidly at small pixel tolerances and then approaches a plateau, consistent with its relatively compact error distribution. In contrast, stenosis-throat localization improves more gradually as the tolerance increases, reflecting the broader error distribution observed in Table I. Together, these results indicate that guidewire-tip localization is the more stable of the two localization tasks, while precise identification of the stenosis throat remains more challenging.

TABLE II  
COMPARISON OF MULTI-TASK ARCHITECTURES ON THE SIMULATED TEST SET (n = 336). BEST RESULTS ARE SHOWN IN BOLD.
<table><tr><td>Network</td><td>Dice (%)</td><td>clDice (%)</td><td>IoU (%)</td><td>Tip error (px) ↓</td><td>Tip PCK@5↑</td><td>Throat error (px) ↓</td><td>Throat PCK@10 ↑</td><td>Throat in lesion (%) ↑</td><td>Error &gt;15 px (%) ↓</td></tr><tr><td>DeepLabV3+</td><td>90.3</td><td>96.0</td><td>82.5</td><td>1.09</td><td>88.1</td><td>3.82</td><td>64.4</td><td>70.6</td><td>32.2</td></tr><tr><td>U-Net</td><td>87.7</td><td>94.4</td><td>78.3</td><td>1.79</td><td>82.3</td><td>5.92</td><td>62.2</td><td>71.5</td><td>34.7</td></tr><tr><td>SegFormer</td><td>86.5</td><td>93.1</td><td>76.5</td><td>2.06</td><td>74.3</td><td>6.37</td><td>57.5</td><td>68.0</td><td>35.0</td></tr><tr><td>FPN</td><td>85.7</td><td>93.4</td><td>75.2</td><td>1.54</td><td>79.5</td><td>8.97</td><td>53.1</td><td>66.5</td><td>41.6</td></tr></table>

![](images/0ba987a0f7d48400a50fd42d1aca1975b7f4ec32d27ff38e492a612533d92d66.jpg)  
Fig. 3. Qualitative comparison of the multi-task architectures on representative simulated fluoroscopy images. For each test case, the input image and ground-truth vessel mask are followed by vessel-segmentation and guidewire-tip/stenosis localization predictions from DeepLabV3+, U-Net, SegFormer, and FPN.

Table II compares DeepLabV3+, U-Net, SegFormer, and FPN using the same multi-task formulation and training conditions. DeepLabV3+ achieved the highest vesselsegmentation performance, with Dice, clDice, and IoU scores of 90.3%, 96.0%, and 82.5%, respectively. It also obtained the lowest median guidewire-tip error (1.09 px) and stenosisthroat error (3.82 px), together with the lowest proportion of large throat-localization errors (> 15 px).

The alternative architectures showed different trade-offs. FPN produced the second-lowest guidewire-tip error (1.54 px), but showed the largest stenosis-throat error (8.97 px) and the highest proportion of errors above 15 px. U-Net was comparatively competitive for stenosis localization and achieved a slightly higher fraction of predictions within the ground-truth lesion than DeepLabV3+ (71.5% versus 70.6%), although its median throat error remained higher (5.92 px). Considering both vessel segmentation and landmark localization performance, DeepLabV3+ was selected for the final MCP-Net configuration.

The qualitative comparison in Fig. 3 is consistent with the quantitative results. The same fluoroscopy cases are processed by all four architectures, enabling direct comparison of vessel segmentation and landmark localization. Across the examples, DeepLabV3+ maintains the major vessel structure and thin branches more consistently, while also preserving the guidewire-tip and stenosis predictions required by the subsequent analysis. The comparison also illustrates that differences between architectures are not limited to pixelwise vessel overlap, but can influence the vascular structure available for downstream graph extraction.

Beyond point localization, the MCP-Net outputs are associated with the predicted vessel geometry to obtain additional navigation-relevant information. Representative examples are shown in Fig. 4. For the guidewire tip, the framework relates the predicted location to the vessel centerline and estimates the local vessel width. This analysis indicates whether the guidewire tip remains near the vessel centerline or deviates toward the vessel boundary, providing useful feedback for downstream guidewire navigation. For the stenosis, the proximal, throat, and distal landmarks provide estimates of lesion extent, throat diameter, and local reference diameter, and percentage diameter stenosis.

Fig. 5 qualitatively demonstrates the vascular topology obtained from the predicted vessel masks. The centerline representation is converted into a graph containing the selected root, bifurcations, endpoints, and connecting vessel segments, while the graph edges retain local geometric information such as vessel caliber. The two examples show that the same processing pipeline can represent coronary trees with different branching complexity. This structured representation provides the connectivity and branch geometry required for the subsequent root-to-stenosis path analysis.

Finally, Fig. 6 demonstrates the complete perceptionto-roadmap workflow. The predicted vessel representation and stenosis location are used to determine a root-to-target centerline path, while the traversed bifurcations are annotated with branch angles and left/right/straight navigation cues. The roadmap is displayed both on the simulated fluoroscopy image and on the extracted vessel representation, providing an interpretable output that combines the learned perception results with the subsequent geometric analysis.

![](images/9f263b52df01d4e801c8fb676a3f6271a8235ac4a40db069452ba80979439d4c.jpg)

Fig. 4. Representative guidewire-tip and stenosis geometric analysis. Predicted landmarks are associated with the vessel geometry to estimate guidewire-tip position relative to the vessel, local vessel width, stenosis throat and reference diameters, lesion extent, and descriptive image-based percentage diameter stenosis (%DS).  
![](images/8730683921b1e547239e5f9dc4774d658292e9ed2e5d24c8d6d0be1afe07d2ea.jpg)  
Fig. 5. Representative vascular topology extracted from MCP-Net vessel predictions. The predicted mask is skeletonized and converted into a graph containing the automatically selected root, bifurcations, endpoints, and connecting vessel segments. Graph edges are color-coded according to their vessel diameter.

The complete real-time pipeline, including MCP-Net inference, vascular topology extraction, and roadmap generation, achieved a mean end-to-end latency of 26.90 ms per frame, corresponding to 37.2 FPS on the NVIDIA GB10 platform. This demonstrates that the complete perception-to-roadmap pipeline can operate at real-time rates.

## VI. DISCUSSION

The results show that MCP-Net can jointly recover vessel structure, guidewire-tip location, and stenosis information from a single simulated fluoroscopy frame. The high clDice score indicates good preservation of vessel connectivity, which is particularly important because missing or disconnected branches can directly affect the subsequent graph and navigation path. Guidewire-tip localization was robust, whereas stenosis localization exhibited a heavier error tail, indicating that precise lesion localization remains the more challenging perception task.

![](images/8f7c36a26c9f5c05762682c7a00f8c08603bfb3492fc574305fa51853ff42533.jpg)  
Fig. 6. Real-time roadmap visualization for two representative cases. The root-to-stenosis path is overlaid on the simulated fluoroscopy image (left) and predicted vessel mask (right), together with the guidewire tip, stenosis, start and end points, and bifurcation-level navigation cues.

The proposed framework is particularly relevant to robotic-assisted interventions because it converts imagelevel predictions into an explicit vascular graph and targetoriented roadmap. Centerline-based representations have previously supported autonomous endovascular navigation [20], while simulation has enabled guidewire-navigation strategies across varying vascular geometries [21], [22]. Our framework complements these approaches by deriving vessel geometry, device location, and the target directly from fluoroscopic perception. Since simulation provides known anatomy, imaging geometry, and automatically generated supervision, the same principle could also be extended to other image-guided robotic procedures, such as TAVI or bronchoscopy, using task-specific anatomical representations.

The main limitation is that training and evaluation are currently restricted to simulated fluoroscopy. Although simulation enables controlled variation and exact geometric supervision, a domain gap to clinical images remains. Simulationto-real approaches such as XA-Sim2Real [13] provide a possible direction for addressing this limitation. Future work will therefore focus on phantom and clinical validation, simulation-to-real adaptation, and integration of the generated roadmap with closed-loop robotic guidewire control.

## VII. CONCLUSION

This work presented a simulation-driven framework for real-time coronary perception, vascular topology analysis, and roadmap generation from X-ray fluoroscopy. MCP-Net jointly estimates the coronary vessel structure, guidewiretip location, and target stenosis, while subsequent geometric processing transforms these predictions into an interpretable vascular graph and root-to-target navigation roadmap. Guidewire-tip localization additionally provides information about the current device position relative to the vessel centerline, which can support guiding the guidewire along the generated roadmap toward the target lesion. The results demonstrate the feasibility of integrating learned fluoroscopic perception with structured vascular reasoning to provide navigation-relevant information for robotic-assisted PCI.

## REFERENCES

[1] G. A. Mensah, V. Fuster, C. J. Murray, G. A. Roth, Y. H. Abate, M. Abbasian, F. Abd-Allah, A. Abdollahi, M. Abdollahi, D. M. Abdulah et al., “Global burden of cardiovascular diseases and risks, 1990-2022,” Journal of the American College of Cardiology, vol. 82, no. 25, pp. 2350–2473, 2023.

[2] W. C. Members, J. S. Lawton, J. E. Tamis-Holland, S. Bangalore, E. R. Bates, T. M. Beckie, J. M. Bischoff, J. A. Bittl, M. G. Cohen, J. M. DiMaio et al., “2021 acc/aha/scai guideline for coronary artery revascularization: a report of the american college of cardiology/american heart association joint committee on clinical practice guidelines,” Journal of the American College of Cardiology, vol. 79, no. 2, pp. e21–e129, 2022.

[3] H. He, A. Banerjee, R. P. Choudhury, and V. Grau, “Deep learning based coronary vessels segmentation in x-ray angiography using temporal information,” Medical Image Analysis, vol. 102, p. 103496, 2025.

[4] M. A. Molenaar, E. Hebbo, J. L. Selder, N. Shekiladze, P. B. Sandesara, W. J. Nicholson, F. W. Asselbergs, S. Ahmad, D. A. Gold, S. M. Sakr et al., “Deep learning–based segmentation of coronary arteries and stenosis detection in x-ray coronary angiography,” JACC: Advances, vol. 4, no. 12 Part 2, p. 102360, 2025.

[5] A. Stevenson, A. Kirresh, M. Ahmad, and L. Candilio, “Roboticassisted pci: the future of coronary intervention?” Cardiovascular Revascularization Medicine, vol. 35, pp. 161–168, 2022.

[6] J. D. Haner, L. R¨ aber, C. Moro, S. Losdat, and S. Windecker, “Robotic-¨ assisted percutaneous coronary intervention: experience in switzerland,” Frontiers in cardiovascular medicine, vol. 10, p. 1294930, 2023.

[7] B. Bay, M. Seiffert, and F. J. Brunner, “Robotic-assisted percutaneous coronary intervention: current evidence and future directions,” Frontiers in Cardiovascular Medicine, vol. 13, p. 1827660, 2026.

[8] M. Nobre Menezes, J. L. Silva, B. Silva, T. Rodrigues, C. Guerreiro, J. P. Guedes, M. O. Santos, A. L. Oliveira, and F. J. Pinto, “Coronary x-ray angiography segmentation using artificial intelligence: a multicentric validation study of a deep learning model,” The international journal of cardiovascular imaging, vol. 39, no. 7, pp. 1385–1396, 2023.

[9] Y.-J. Zhou, X.-L. Xie, X.-H. Zhou, S.-Q. Liu, G.-B. Bian, and Z.-G. Hou, “Pyramid attention recurrent networks for real-time guidewire segmentation and tracking in intraoperative x-ray fluoroscopy,” Computerized Medical Imaging and Graphics, vol. 83, p. 101734, 2020.

[10] H. Ma, I. Smal, J. Daemen, and T. van Walsum, “Dynamic coronary roadmapping via catheter tip tracking in x-ray fluoroscopy with deep learning based bayesian filtering,” Medical image analysis, vol. 61, p. 101634, 2020.

[11] C. Li, M. Chu, X. Liu, B. Yang, Y. Li, G. Li, and S. Tu, “Straightening the path to clarity: A subpixel-level vessel segmentation framework in x-ray coronary angiography,” Computerized Medical Imaging and Graphics, p. 102634, 2025.

[12] Z. Zhang, B. Zhang, M. F. Azampour, S. Faghihroohi, A. Tomczak, H. Schunkert, and N. Navab, “Self-supervised vessel segmentation from x-ray images using digitally reconstructed radiographs,” in BVM Workshop. Springer, 2024, pp. 220–225.

[13] B. Zhang, Z. Zhang, S. Liu, S. Faghihroohi, H. Schunkert, and N. Navab, “Xa-sim2real: adaptive representation learning for vessel segmentation in x-ray angiography,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 747–756.

[14] B. Zhang, M. Bui, C. Wang, F. Bourier, H. Schunkert, and N. Navab, “Real-time guidewire tracking and segmentation in intraoperative xray,” in Medical Imaging 2022: Image-Guided Procedures, Robotic Interventions, and Modeling, vol. 12034. SPIE, 2022, pp. 283–290.

[15] T. Yao, Z. Pei, Y. Li, Y. Yuan, and P. Qi, “Real-time guidewire tip tracking using a siamese network for image-guided endovascular procedures,” Advanced Intelligent Systems, vol. 7, no. 10, p. 2500425, 2025.

[16] H. Robertshaw, Y. Hao, W. Deng, B. Jackson, S. Sadati, N. Fischer, T. Vercauteren, A. Granados, and T. C. Booth, “Toward real-time autonomous navigation: transformer-based catheter tip tracking in fluoroscopy,” International Journal of Computer Assisted Radiology and Surgery, pp. 1–9, 2026.

[17] L. Xi, Y. Ma, E. Koland, S. Howell, A. Rinaldi, and K. S. Rhode, “Catheter detection and segmentation in x-ray images via multi-task learning,” International Journal of Computer Assisted Radiology and Surgery, vol. 21, no. 1, pp. 163–173, 2026.

[18] E. Labrecque Langlais, D. Corbin, O. Tastet, A. Hayek, G. Doolub,<sup>´</sup> S. Mrad, J.-C. Tardif, J.-F. Tanguay, G. Marquis-Gravel, G. H. Tison et al., “Evaluation of stenoses using ai video models applied to coronary angiography,” NPJ digital medicine, vol. 7, no. 1, p. 138, 2024.

[19] N. Hampe, S. G. van Velzen, J. M. Wolterink, C. Collet, J. P. Henriques, N. Planken, and I. Isgum, “Graph neural networks forˇ automatic extraction and labeling of the coronary artery tree in ct angiography,” Journal of Medical Imaging, vol. 11, no. 3, pp. 034 001– 034 001, 2024.

[20] N. Li, Y. Wang, H. Cheng, H. Zhao, and H. Ding, “Vascular centerlineguided autonomous navigation methods for robot-lead endovascular interventions,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 11 578–11 584.

[21] V. Scarponi, M. Duprez, F. Nageotte, and S. Cotin, “A zero-shot reinforcement learning strategy for autonomous guidewire navigation,” International Journal of Computer Assisted Radiology and Surgery, vol. 19, no. 6, pp. 1185–1192, 2024.

[22] V. Scarponi, F. Lecomte, M. Duprez, F. Nageotte, and S. Cotin, “Autonomous guidewire navigation in dynamic environments,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 6931–6938.

[23] A. Zeng, C. Wu, G. Lin, W. Xie, J. Hong, M. Huang, J. Zhuang, S. Bi, D. Pan, N. Ullah et al., “Imagecas: A large-scale dataset and benchmark for coronary artery segmentation based on computed tomography angiography images,” Computerized Medical Imaging and Graphics, vol. 109, p. 102287, 2023.

[24] A. A. Hendriksen, D. Schut, W. J. Palenstijn, N. Vigano, J. Kim, ´ D. M. Pelt, T. Van Leeuwen, and K. Joost Batenburg, “Tomosipo: fast, flexible, and convenient 3d tomography for complex scanning geometries in python,” Optics Express, vol. 29, no. 24, pp. 40 494– 40 513, 2021.

[25] W. Van Aarle, W. J. Palenstijn, J. De Beenhouwer, T. Altantzis, S. Bals, K. J. Batenburg, and J. Sijbers, “The astra toolbox: A platform for advanced algorithm development in electron tomography,” Ultramicroscopy, vol. 157, pp. 35–47, 2015.

[26] R. Izzo, D. Steinman, S. Manini, and L. Antiga, “The vascular modeling toolkit: a python library for the analysis of tubular structures in medical images,” Journal of Open Source Software, vol. 3, no. 25, p. 745, 2018.

[27] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-decoder with atrous separable convolution for semantic image segmentation,” in European conference on computer vision. Springer, 2018, pp. 833–851.

[28] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.

[29] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Advances in neural information processing systems, vol. 34, pp. 12 077–12 090, 2021.

[30] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” in 2017 IEEE conference on computer vision and pattern recognition (CVPR). Ieee, 2017, pp. 936–944.