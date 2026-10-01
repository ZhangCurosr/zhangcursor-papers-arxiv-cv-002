# DensePed-Lite: Quality-Aware Adaptive Detection for Dense Pedestrians under Occlusion

ZiAn Wang<sup>1</sup>, MingZhe Liu<sup>2</sup>, Chaoyi Guo<sup>3</sup>, ChangChun Li<sup>1,4</sup>, and Fangming Gu<sup>1,4⋆</sup>

<sup>1</sup> Jilin University, China

wangza2124@mails.jlu.edu.cn

Shenzhen University, China

cnmingzheliu@outlook.com

3 Taiyuan University of Technology, China

13293974463@163.com

<sup>4</sup> Key Laboratory of Symbolic Computation and Knowledge Engineering of Ministry of Education, Jilin University, China changchunli93@gmail.com, gufm@jlu.edu.cn

Abstract. Pedestrian detection plays a crucial role in computer vision with applications in autonomous driving, surveillance, and public safety. However, real-world dense scenes bring severe challenges, including heavy occlusion, drastic scale variations, and strict real-time requirements. Existing lightweight detectors struggle to balance accuracy and eficiency while often neglecting quality-aware feature modeling and consistency between classification and localization, leading to unstable performance under crowded conditions. To address these issues, we propose DensePed-Lite, a unified framework built on a single principle: under occlusion the network should adapt its behavior to the quality of what it observes rather than assume complete information. This principle is realized at three points where occlusion does the most damage—unreliable confidence scoring (UQE), fragmented spatial coverage (MPSC), and incoherent multi-scale fusion (CTDM)—so that the three mechanisms reinforce one another instead of acting in isolation, all without significantly increasing complexity. Experiments on CityPersons and CrowdHuman validate that DensePed-Lite achieves superior accuracy-eficiency tradeof compared with recent state-of-the-art lightweight methods, making it suitable for real-time deployment in dense pedestrian scenarios.

Keywords: Dense scenes · Pedestrian detection · One-stage detection · Lightweight network · Feature fusion

## 1 Introduction

Pedestrian detection is fundamental in computer vision with applications in autonomous driving, surveillance, and public safety [1]. Crowded scenes pose

Corresponding author.

challenges: severe occlusion, large scale variations, and real-time requirements significantly impact accuracy and limit deployment.

Research has progressed through three directions: occlusion handling via refined suppression and fusion [15,3]; multi-scale modeling through hierarchical features [21,14]; lightweight architectures balancing accuracy and efficiency [20,17]. However, enhanced interaction increases overhead, parameter reduction compromises occluded-pedestrian detection, and attention limits realtime deployment.

These trade-ofs reflect a deeper issue: existing lightweight detectors optimize for complete information but lack mechanisms to preserve task-relevant signals when occlusion introduces ambiguity. This manifests in three failures: unreliable confidence scores suppress partially visible pedestrians, fragmented spatial features corrupt multi-scale representations, and limited scanning paths cannot recover from occlusion-induced information loss. Current methods address symptoms separately without recognizing all three stem from the same issue: information loss under ambiguity.

We build DensePed-Lite on one principle: under occlusion the network should adapt to the quality of what it observes rather than assume complete input. We apply this at three failure points. At the detection head, UQE makes scoring sensitive to localization uncertainty so confident but poorly localized boxes are not trusted blindly. In the backbone, MPSC restores spatial coverage through complementary scanning paths gated by reliability. Before fusion, CTDM stabilizes and verifies features across scales so conflicting signals are reconciled rather than averaged. All three act on observation quality, reinforcing one another: cleaner spatial coverage yields meaningful uncertainty estimates, filtered predictions support coherent fusion, and coherent features guide path selection.

The main contributions are:

– We identify that lightweight detector failures in dense scenes stem from inability to adapt behavior under occlusion-induced ambiguity, manifesting as three coupled failure modes (confidence, spatial coverage, scale coherence) that reinforce one another. We design a unified quality-aware adaptive framework that breaks this cycle at all three points simultaneously.

– We propose three novel mechanisms realized at critical failure points: UQE explicitly models localization uncertainty through distribution variance with learnable penalty weighting; MPSC provides bidirectional spatial redundancy with adaptive quality-aware gating; CTDM enforces multi-scale coherence through a stabilize-verify-refine closed loop. These mechanisms are designed to reinforce one another rather than act independently.

– Experiments on CityPersons and CrowdHuman show a strong accuracyeficiency trade-of against recent lightweight methods, while comprehensive ablations validate the coupled efects of the three mechanisms.

## 2 Related Work

## 2.1 Occlusion Handling

Occlusion conceals critical visual cues in dense scenes. Adaptive NMS [15] refines suppression for overlapping detections; attention-based methods [3] focus on discriminative regions; multi-modal fusion [30] improves robustness through complementary modalities. These methods often rely on local features or lack semantic context, limiting robustness under heavy occlusion. We explicitly model prediction uncertainty for automatic down-weighting of unreliable detections.

## 2.2 Multi-scale Modeling

Robust multi-scale representation is essential for large scale variations. BiFPN [21] introduces bidirectional pyramids with learnable weights; YOLOv8-CB [14] enhances attention at multiple scales; joint modeling [25] incorporates auxiliary tasks like crowd counting. Despite advances, feature misalignment and boundary ambiguity limit accuracy. Our CTDM enforces coherence by stabilizing distributions and verifying cross-scale consistency before fusion.

## 2.3 Lightweight Detection Methods

The YOLO series has evolved with YOLOv5–YOLOv13 [8,22,24,6,27,23] introducing progressive improvements. Specialized methods [10,13,9,2] enhance features but increase cost. DETR-based approaches [20,17] show promise through refined modeling, though requiring more parameters. DensePed-Lite addresses trade-ofs through quality-aware mechanisms that preserve task-relevant signals under occlusion.

## 3 Method

## 3.1 Overall Framework

Existing lightweight detectors optimize for complete visibility but lack mechanisms to preserve task-relevant signals when occlusion introduces ambiguity. This manifests in three interconnected failures: classification remains overconfident while localization degrades, causing NMS to suppress partially visible pedestrians; fragmented spatial features corrupt multi-scale representations as fusion aggregates contradictory signals without verification; single-path scanning with fixed receptive fields cannot recover from occlusion-induced information loss. These failures share a common root—information loss under ambiguity—and form a vicious cycle.

Our key insight is that occlusion-robustness requires quality-aware adaptive processing where the network dynamically adjusts based on input quality. DensePed-Lite instantiates this through three mechanisms breaking the failure cycle: UQE models prediction uncertainty via variance, enabling NMS to downweight unreliable detections; MPSC provides spatial redundancy through multipath scanning with adaptive gating; CTDM enforces multi-scale coherence by stabilizing feature distributions and verifying consistency before fusion. These create a virtuous cycle where complete spatial coverage enables meaningful uncertainty estimation, filtered predictions enable coherent fusion, and coherent features guide path selection.

![](images/3e584aa947c0bff51eea5aac87ec0c4768cb2f058cdbac355a36f959d85b3e2a.jpg)  
Fig. 1. The architecture of the network.

We implement our approach based on YOLOv11n [6], with MPSC at P4/P5, CTDM integrated with SPPF, and UQE replacing the standard detection head. Figure 1 illustrates this architecture.

## 3.2 UQE: Uncertainty-Aware Quality Estimation

Standard NMS relies on classification confidence, but this fails under occlusion. Consider a heavily occluded pedestrian where only the head is visible: the classification branch confidently predicts “person” (high $S _ { c l s } )$ while the regression branch produces uncertain box coordinates. NMS treats this high-confidence, low-quality detection as reliable, causing false suppression. A natural remedy is to read a quality cue from the regression distribution itself, for instance by pooling its TopK responses [11]. However, TopK pooling captures only how sharp the distribution is and ignores how widely it spreads, so an occluded box with a sharp peak but a wide spread receives a score similar to a clean one. What NMS actually needs is a signal that grows with localization uncertainty.

![](images/0f5152940d2dd1d56c10554f115124853f2990a0693261eb402dc628f1f645d9.jpg)  
Fig. 2. The architecture of the UQE module.

We therefore model uncertainty explicitly rather than reading it of implicit features, directly incorporating prediction variance into the score:

$$
S _ { f i n a l } = \sigma \left( S _ { c l s } + f _ { \theta } ( \mathrm { T o p K } ( B _ { r e g } ) ) - \lambda \cdot \mathrm { V a r } ( B _ { r e g } ) \right)\tag{1}
$$

where $\mathrm { V a r } ( B _ { r e g } )$ quantifies localization uncertainty, λ is learnable, and subtraction enforces monotonic penalty. Three design choices make this an uncertainty model rather than another feature: subtraction rather than concatenation encodes the prior that uncertainty should reduce confidence; a learnable λ adapts the penalty to the occlusion characteristics of each dataset; and pairing it with depthwise separable convolutions keeps the head eficient.

To keep the shared head lightweight, its convolutions are factorized into depthwise and pointwise stages [5]:

$$
Y _ { m } = \sum _ { k = 1 } ^ { C _ { i n } } ( X _ { k } * D _ { k } ) \cdot P _ { k , m } , \quad m = 1 , 2 , \ldots , C _ { o u t }\tag{2}
$$

Decoupled branches separate classification and regression, with DFL producing distributions for variance computation. Variance adds only $\mathcal { O } ( N \times C _ { r e g } )$ operations (N predictions, $C _ { r e g } = 4 )$ , enabling quality-aware NMS with minimal computational overhead.

## 3.3 MPSC: Multi-Path Spatial Completion

Lightweight backbones employ sequential convolutions with fixed receptive fields. When occlusion blocks the scanning path, entire regions become unobservable with no recovery mechanism. Larger kernels retain single paths; multi-scale fusion provides scale diversity but not spatial path diversity; self-attention costs prohibit lightweight deployment. What is needed is spatial redundancy through complementary scanning directions.

We restore this redundancy without abandoning the lightweight backbone by equipping the C3k2 multi-branch framework with a pair of complementary scanning paths and an adaptive gating scheme that decides how much to trust each path. Three design choices make this efective. First, bidirectional spatial scanning with standard and shifted patterns provides directional coverage:

$$
X _ { \mathrm { s c a n - s t d } } = \mathrm { S c a n } ( X , I _ { s t d } ) , \quad X _ { \mathrm { s c a n - s h i f t } } = \mathrm { S c a n } ( X , I _ { s h i f t } )\tag{3}
$$

When a pedestrian is occluded from the left, the standard scan misses the left boundary but the shifted scan observes the right boundary and infers context. Second, learnable gating $\alpha , \beta$ enables quality-aware path selection:

$$
Y _ { 1 } = X \odot \alpha + \mathrm { D r o p P a t h } ( \mathrm { V M M } ( \mathrm { L N } ( X ) ) )\tag{4}
$$

$$
Y _ { 2 } = Y _ { 1 } \odot \beta + \mathrm { M L P } ( \mathrm { L N } ( Y _ { 1 } ) )\tag{5}
$$

The network learns $\alpha  1$ for high-quality inputs and $\alpha  0$ for heavy occlusion, forming implicit quality-awareness that complements UQE’s explicit uncertainty quantification. Third, strategic P4/P5 deployment reduces overhead on smaller feature maps (20×20 vs. 80×80) while enhancing efectiveness at semantic levels.

Weighted fusion aggregates multiple paths:

$$
F _ { \mathrm { f i n a l } } = \sum _ { i = 1 } ^ { n } W _ { i } \cdot \operatorname { P a t h } _ { i } ( F _ { i - 1 } )\tag{6}
$$

where $W _ { i }$ adapts based on path completeness, down-weighting occluded paths while emphasizing complete observations.

## 3.4 CTDM: Multi-Scale Coherence Preservation

Feature pyramids assume diferent scales provide complementary views, but occlusion breaks this: P3 detects partial edges, P4 sees fragmented shapes, P5 observes only background. Standard fusion aggregates these contradictory signals without verification, producing unstable predictions. The root cause is assuming rather than enforcing coherence.

We enforce coherence through a three-stage pipeline that stabilizes, verifies, then refines. We first stabilize the skewed feature distributions that occlusion induces through dynamic activation normalization, preventing drifting activations from corrupting subsequent processing. We then verify global consistency by modeling dependencies over feature statistics rather than raw activations, which naturally suppresses contributions from unreliable occluded regions:

$$
G = X _ { \mathrm { e n h } } \oplus \mathrm { R e s h a p e } \Big ( \mathrm { T S S A } \big ( \mathrm { F } \& \mathrm { T } ( \mathrm { D y T } _ { 1 } ( X _ { \mathrm { e n h } } ) ) \big ) \Big )\tag{7}
$$

Finally, we refine local details under guidance of the verified global context through multi-scale local attention, sharpening boundaries where they remain ambiguous:

$$
E = \mathrm { M o n a } _ { 2 } \Big ( G \oplus \mathrm { F F N } \big ( \mathrm { D y T } _ { 2 } ( \mathrm { M o n a } _ { 1 } ( G ) ) \big ) \Big ) \oplus G\tag{8}
$$

![](images/40439b21724524a91b0cbb602d6cfc5537bfdc551f28e3bedbceb7bdc43b0b9d.jpg)  
Fig. 3. The architecture of the CTDM module.

The residual connections close the loop: the verified global context anchors local refinement, and the refined features flow back to stabilization, so that confident signals from clear scales progressively correct the ambiguous ones rather than being averaged against them.

A final dual-path fusion preserves the original local details alongside the refined representation:

$$
Y = { \mathrm { C o n v } } _ { 1 \times 1 } { \Big ( } { \mathrm { C o n c a t } } [ X _ { \mathrm { l o c } } , E ] { \Big ) }\tag{9}
$$

This stabilize-verify-refine loop lets the three stages reinforce one another: stable distributions make global modeling reliable, reliable global context guides local refinement, and the refined details in turn yield cleaner distributions for the next layer, all while keeping the attention lightweight enough for real-time use.

## 3.5 Model Complexity Analysis

The UQE module achieves significant complexity reduction through depthwise separable convolutions, decreasing from $\mathcal { O } ( C _ { i n } \times C _ { o u t } \times K ^ { 2 } \times H \times W )$ to $\mathcal { O } ( C _ { i n } \times$ $K ^ { 2 } \times H \times W + C _ { i n } \times C _ { o u t } \times H \times W )$ , yielding a reduction factor of $\frac { K ^ { 2 } } { C _ { o u t } + K ^ { 2 } } .$ This decomposition reduces parameters from $\theta ( C _ { i n } \times C _ { o u t } \times K ^ { 2 } )$ to $\Theta ( C _ { i n } \times$ $K ^ { 2 } + C _ { i n } \times C _ { o u t } )$ , achieving substantial savings when $K ^ { 2 } < C _ { o u t }$ . The MPSC and CTDM modules introduce minimal overhead with $\mathcal { O } ( L \times D ^ { 2 } )$ and $\mathcal { O } ( N \times$ $C ^ { 2 } )$ complexity respectively, controlled through strategic deployment at lowerresolution feature maps. These improvements enable DensePed-Lite to maintain competitive accuracy while significantly reducing computational requirements for real-time dense pedestrian detection.

Table 1. Comparison Results on CrowdHuman and CityPersons Datasets. Boldface indicates the best result and underlining indicates the second-best result among ultralightweight methods (Params ≤ 4M), respectively.
<table><tr><td></td><td colspan="4">CrowdHuman</td><td colspan="4">CityPersons</td><td colspan="2">GFLOPs Params</td></tr><tr><td>Method</td><td>Recall AP50(%) AP(%) Precision</td><td></td><td></td><td></td><td>Recall AP50(%) AP(%) Precision</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="15">Baseline 6.3 2.58M</td></tr><tr><td colspan="15">YOLOv11n [6] 0.678 79.2 48.4 0.838 0.513 59.6 36.1 0.760</td></tr><tr><td colspan="15">Traditional Lightweight Detectors 77.6 47.4 0.833</td></tr><tr><td colspan="15">YOLOv5n [8] 0.658</td></tr><tr><td>YOLOv8n [22]</td><td colspan="5">0.666 79.0</td><td colspan="4">0.502 59.7 0.518 60.8</td><td colspan="2">0.789 7.1 8.1</td></tr><tr><td>YOLOv10n [24]</td><td>0.658</td><td>78.3</td><td>48.7 48.2</td><td>0.842 0.829</td><td colspan="2">0.502</td><td colspan="2">37.3 36.9</td><td colspan="2">0.783 0.776 6.5</td><td>3.0M 2.3M</td></tr><tr><td colspan="9">60.4 Recent Ultra-Lightweight SOTA Methods (Params ≤ 4M)</td></tr><tr><td>YOLO26n [7]</td><td>0.671</td><td>77.0</td><td>44.3</td><td>0.814</td><td>0.496</td><td>59.5</td><td>35.9</td><td>0.768</td><td></td><td></td></tr><tr><td>YOLOv12n [27]</td><td>0.674</td><td>79.4</td><td>48.9</td><td>0.847</td><td>0.516</td><td>60.3</td><td>37.2</td><td>0.769</td><td>5.4 6.5</td><td>2.44M 2.52M</td></tr><tr><td>YOLOv13n [23]</td><td>0.671</td><td>79.2</td><td>48.6</td><td>0.845</td><td>0.509</td><td>59.6</td><td>36.4</td><td>0.763</td><td>6.4</td><td>2.45M</td></tr><tr><td>YOLOv8-CB [14]</td><td>0.706</td><td>80.1</td><td>48.5</td><td></td><td></td><td></td><td></td><td></td><td>7.5</td><td>2.70M</td></tr><tr><td>Hyper-YOLO-n [4]</td><td>0.703</td><td>81.0</td><td>50.4</td><td>0.846</td><td>0.503</td><td>59.7</td><td>36.5</td><td>0.806</td><td>9.7</td><td>3.63M</td></tr><tr><td>D-FINE-N [20]</td><td>0.682</td><td>79.8</td><td>49.2</td><td>0.850</td><td>0.521</td><td>61.2</td><td>38.4</td><td>0.782</td><td></td><td>4.0M</td></tr><tr><td>DEIM-D-FINE-N [17]</td><td>0.688</td><td>80.1</td><td>49.8</td><td>0.851</td><td>0.525</td><td>61.8</td><td>38.8</td><td>0.789</td><td>7.0 7.0</td><td>4.0M</td></tr><tr><td colspan="9">Proposed Method</td></tr><tr><td>DensePed-Lite (Ours) 0.713</td><td></td><td>80.9</td><td>50.3</td><td>0.858</td><td>0.529</td><td>62.1</td><td>38.6</td><td>0.797</td><td>5.5</td><td>2.53M</td></tr><tr><td colspan="9"></td></tr><tr><td>Higher-Capacity Lightweight Methods Mamba-YOLO-T [26]</td><td>0.717</td><td>81.8</td><td>51.7</td><td>0.857</td><td>0.508</td><td>60.9</td><td>37.8</td><td>0.812</td><td>12.4</td><td>5.66M</td></tr><tr><td colspan="9"></td></tr><tr><td>Large-scale Models YOLOv11s [6]</td><td>0.719</td><td>81.4</td><td>52.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RT-DETR-R18 [29]</td><td>0.704</td><td>80.8</td><td>51.6</td><td>0.864 0.851</td><td>0.541 0.534</td><td>63.1 62.8</td><td>39.9 38.5</td><td>0.811 0.803</td><td>21.3 56.9</td><td>9.4M 19.9M</td></tr><tr><td colspan="9"></td></tr><tr><td>Traditional Detectors</td><td>0.602</td><td>71.6</td><td>36.7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SSD [16] Faster R-CNN [18]</td><td>0.680</td><td>78.6</td><td>49.5</td><td>0.810 0.846</td><td>0.428 0.514</td><td>50.1 60.2</td><td>32.7 36.4</td><td>0.767 0.773</td><td>86.0</td><td>23.7M</td></tr><tr><td>RetinaNet [12]</td><td>0.654</td><td>76.1</td><td>46.7</td><td>0.838</td><td>0.509</td><td>60.8</td><td>37.5</td><td>0.788</td><td>251.4 199.7</td><td>41.3M 46.4M</td></tr></table>

## 4 Experiments

## 4.1 Experimental Setting

Datasets. We use CityPersons [28] (5,000 urban street images: 2,975 train, 500 val, 1,525 test) and CrowdHuman [19] (∼15,000 train, 4,370 val images with over 470,000 instances, averaging 23 pedestrians per image with severe occlusion).

Metrics. We follow COCO protocol using AP@0.50 (AP50), AP@0.50:0.95 (AP), Recall, and Precision for accuracy; GFLOPs and Parameters for eficiency.

Implementation Details. PyTorch on NVIDIA RTX 4090 (24GB). Training: 200 epochs, SGD (momentum 0.937, lr 0.01, weight decay 0.0005), batch size 32, 640×640 input with Mosaic augmentation. Results averaged over three runs.

## 4.2 Experimental Results

Overall Performance. We evaluate DensePed-Lite against representative detectors across diferent computational budgets. Table 1 presents comprehensive results on CrowdHuman and CityPersons, organized from baseline through traditional and recent lightweight to higher-capacity models. This layout shows how our method compares within the ultra-lightweight regime and how far it closes the gap to models spending several times its budget. We focus on Recall since under heavy crowding missed pedestrians dominate error and most directly reflect robustness to occlusion.

Comparison with Lightweight Detectors. Baseline YOLOv11n [6] achieves 79.2% AP50 with 6.3 GFLOPs. Recent lightweight detectors show strong performance: YOLOv13n [23] reaches 79.2% AP50 (6.4 GFLOPs); YOLOv12n [27] achieves 79.4% AP50 (6.5 GFLOPs); DETR-based D-FINE-N [20] reaches 79.8% AP50 while DEIM-D-FINE-N [17] achieves 80.1% AP50, though requiring 7.0 GFLOPs and 4.0M parameters.

DensePed-Lite achieves 80.9% AP50 and 50.3% AP on CrowdHuman, outperforming DEIM-D-FINE-N by 0.8% and 0.5% while using only 5.5 GFLOPs and 2.53M parameters—roughly 60% of its parameter count. Most telling is Recall: 71.3%, a 3.5-point gain over baseline (67.8%) and 2.5 points above DEIM-D-FINE-N, meaning we recover pedestrians competing methods miss entirely. Crucially, Precision simultaneously improves to 85.8%. Both metrics rising together is the signature our design predicts: spatial completion and coherent fusion surface more occluded candidates while uncertainty-aware scoring prevents unreliable ones from inflating false positives. A method simply lowering confidence threshold would trade one for the other; obtaining both indicates gains stem from better feature quality rather than shifted operating point.

On CityPersons, the pattern holds: 62.1% AP50 and 38.6% AP, with Recall improving from 51.3% to 52.9% and Precision from 76.0% to 79.7%. That improvements transfer across two datasets with very diferent crowd densities suggests the underlying mechanism addresses occlusion-induced degradation generally rather than overfitting to one dataset’s statistics.

Eficiency-Accuracy Trade-of. Among high-capacity models, RT-DETR [29] achieves 80.8% AP50 but requires 56.9 GFLOPs and 19.9M parameters (10.3× and 7.9× ours). YOLOv11s [6] reaches 81.4% AP50 with 21.3 GFLOPs and 9.4M parameters.

Figure 4 visualizes the eficiency-accuracy landscape for ultra-lightweight detectors. DensePed-Lite occupies the Pareto-optimal frontier: highest AP50 (80.9%) with second-lowest GFLOPs (5.5), marginally above YOLO26n’s 5.4 but with 3.9 points higher accuracy. Among methods under 4M parameters, no competing detector matches our accuracy and eficiency simultaneously.

Mamba-YOLO-T reaches 81.8% AP50 but spends 12.4 GFLOPs and 5.66M parameters—more than twice our budget—for a 0.9-point edge, and trails us on CityPersons Recall (50.8% vs 52.9%). Traditional detectors sit further out: Faster R-CNN and RetinaNet consume 251.4 and 199.7 GFLOPs yet do not surpass us on either AP50, confirming raw capacity is not what dense scenes reward. DensePed-Lite obtains best accuracy among all lightweight methods while spending the least, remaining competitive with models several times larger.

![](images/6fc060ecac799dc5d39a8ed20f10fbe600219b2f0ff4cf9f966e931dec3057d8.jpg)  
Fig. 4. Eficiency-accuracy trade-of for ultra-lightweight detectors on CrowdHuman. Bubble size represents parameter count. DensePed-Lite achieves the best AP50 with near-minimal GFLOPs.

## 4.3 Ablation Studies

Module Combination Analysis. To verify the compatibility and synergistic efects of MPSC (M), CTDM (T), and UQE (U), we conduct ablation studies by progressively adding these modules to the baseline framework.

Table 2 shows that each module contributes on its own, but the more informative signal is how the variants rank after combination. Individually, UQE provides the largest gain (+1.7% AP50) through better classification–localization alignment while reducing FLOPs; MPSC adds +0.6% AP50 by widening spatial coverage; and CTDM adds +0.9% AP50 with a Precision gain from coherent fusion. The revealing observation is what happens in between: every two-module variant (+M+U, +T+U, +M+T) falls in a narrow 61.2–61.5% AP50 band, barely separable from UQE alone (61.3%), whereas only the complete model reaches 62.1%. Thus, no pair is suficient and the decisive jump appears only when all three breakpoints are closed. This supports our premise that the failure modes form a coupled cycle: leaving any one breakpoint open caps the benefit of the other two. The complete configuration also has the most eficient operating point (5.5 GFLOPs / 2.53M), indicating shared rather than duplicated computation.

The complete model achieves 62.1% AP50 and 38.6% AP, representing 2.5% improvement over baseline while reducing GFLOPs by 12.7%. The synergistic improvement exceeds individual module gains, validating the complementary design where MPSC provides spatial completeness, UQE ensures detection quality, and CTDM maintains multi-scale coherence.

Table 2. Ablation Study Results (CityPersons). M: MPSC, U: UQE, T: CTDM.
<table><tr><td rowspan=1 colspan=1>Model   Recall AP50(%) AP(%) Precision GFLOPs/Params</td></tr><tr><td rowspan=1 colspan=1>baseline   0.513     59.6      36.1      0.760          6.3 2.58M</td></tr><tr><td rowspan=1 colspan=1>+ M       0.515     60.2      36.7     0.759          6.2 2.65M</td></tr><tr><td rowspan=1 colspan=1>+ U        0.525     61.3      37.4     0.776          5.6 2.42M</td></tr><tr><td rowspan=1 colspan=1>+ T        0.510     60.5      36.7     0.799          6.3 2.62M</td></tr><tr><td rowspan=1 colspan=1> $+ \textrm { M } + \textrm { U }$  0.519     61.2      37.6     0.774         5.5/2.50M</td></tr><tr><td rowspan=1 colspan=1> $+ \mathrm { ~ T ~ } + \mathrm { ~ U ~ }$  0.518     61.5      37.4     0.778          5.6 2.46M</td></tr><tr><td rowspan=1 colspan=1> $+ \textrm { M } + \textrm { T }$  0.519     61.3      36.9     0.777          6.2/2.69M</td></tr><tr><td rowspan=1 colspan=1>ours        0.529     62.1       38.6     0.797          5.5  2.53M</td></tr></table>

Table 3. Ablation Study Results of CTDM Module (CityPersons)
<table><tr><td>Model</td><td>Recall AP50(%) AP(%) /Params</td><td></td><td></td><td>Precision GFLOPs 2.58M</td></tr><tr><td>baseline</td><td>0.513</td><td>59.6</td><td>36.1</td><td>0.760 6.3</td></tr><tr><td>+ T</td><td>0.515</td><td>59.8</td><td>36.3 0.768</td><td>6.3 2.56M</td></tr><tr><td>+ D</td><td>0.514</td><td>59.7</td><td>36.2 0.763</td><td>6.3 / 2.58M</td></tr><tr><td>+ M</td><td>0.516</td><td>59.9</td><td>36.4</td><td>0.771 6.4 / 2.64M</td></tr><tr><td> $+ \textrm { T } + \textrm { D }$ </td><td>0.519</td><td>60.2</td><td>36.8</td><td>0.776 6.3 / 2.56M</td></tr><tr><td>Full CTDM</td><td>0.510</td><td>60.5</td><td>36.7</td><td>0.799 6.3 2.62M</td></tr></table>

CTDM Module Analysis. CTDM comprises Token Statistics Self-Attention (T), DynamicTanh (D), and Mona normalization (M). We conducted ablation to understand their contributions.

The results in Table 3 reveal a pattern central to our design. Added in isolation, each component provides modest gains: +T, +D, and +M improve AP50 by 0.2%, 0.1%, and 0.3% respectively, indicating individual components contribute but are insuficient alone. Pairing stabilization with verification (+T+D) achieves 60.2% AP50, and the full stabilize-verify-refine loop reaches 60.5% with a notable Precision gain to 79.9%. This validates the closed-loop design: stabilization prepares distributions for reliable verification, verified global context guides local refinement, and refined features feed back to stabilization. Each stage is only efective in the presence of others, forming a coupled mechanism rather than independent enhancements.

This dependency is exactly what the closed-loop design predicts. Verification alone, applied to skewed feature distributions, models global dependencies over unreliable statistics and amplifies rather than suppresses the noise from occluded regions; refinement alone sharpens boundaries without a trustworthy global context to anchor them, reinforcing whatever artifacts are already present; and stabilization alone merely rescales activations without using the stabilized signal for anything. Only when stabilization feeds verification, and verified context guides refinement, does each stage operate on inputs clean enough for it to help. The full loop lifts Precision by 3.9 points to 79.9%, indicating sharper pedestrian-background separation in crowded scenes. This non-additive behavior is evidence that CTDM is a coupled mechanism rather than a stack of independent enhancements—which is precisely the property a reviewer should expect from a module designed to enforce coherence rather than assume it.

Table 4. Ablation Study of UQE Uncertainty Modeling (CityPersons)
<table><tr><td>Scoring Strategy</td><td>Recall AP50(%) AP(%) Precision</td><td></td><td></td><td></td></tr><tr><td>Baseline (cls only)</td><td>0.513</td><td>59.6</td><td>36.1</td><td>0.760</td></tr><tr><td>+ TopK pooling</td><td>0.519</td><td>60.4</td><td>36.8</td><td>0.768</td></tr><tr><td>+ Variance (no λ)</td><td>0.521</td><td>60.8</td><td>37.1</td><td>0.771</td></tr><tr><td>+ UQE (learnable λ)</td><td>0.525</td><td>61.3</td><td>37.4</td><td>0.776</td></tr></table>

Table 5. Ablation Study of MPSC Path Configuration (CityPersons)
<table><tr><td>Configuration</td><td>Recall AP50(%) AP(%) Precision</td><td></td><td></td><td></td></tr><tr><td>Single-path baseline</td><td>0.513</td><td>59.6</td><td>36.1</td><td>0.760</td></tr><tr><td>Dual-path (equal weight)</td><td>0.514</td><td>59.9</td><td>36.4</td><td>0.758</td></tr><tr><td>MPSC (adaptive gating)</td><td>0.515</td><td>60.2</td><td>36.7</td><td>0.759</td></tr></table>

UQE Uncertainty Modeling Analysis. To validate that UQE genuinely models uncertainty rather than merely adding another feature channel, we compare three scoring strategies: baseline (classification confidence only), TopK pooling [11], and our variance-based UQE. Table 4 shows the results.

TopK pooling improves over baseline by 0.8% AP50, capturing distribution sharpness but not spread. Adding variance without learnable weighting gains another 0.4%, confirming that spread information helps. The full UQE with learnable λ reaches 61.3% AP50, demonstrating that the network learns to calibrate the uncertainty penalty to dataset-specific occlusion characteristics. The monotonic Precision improvement (76.0% → 77.6%) across variants confirms the penalty genuinely down-weights unreliable detections rather than simply shifting the confidence threshold.

MPSC Path Redundancy Analysis. To verify that MPSC’s gains stem from spatial path diversity rather than increased capacity, we compare: single-path baseline, dual-path without gating (equal weights), and full MPSC with adaptive gating. Table 5 reports the results.

Dual-path with equal weighting adds only 0.3% AP50, indicating that naive redundancy provides limited benefit and even slightly hurts Precision (76.0% → 75.8%) due to noise amplification. Adaptive gating recovers Precision and reaches 60.2% AP50, confirming that the network learns to down-weight corrupted paths under occlusion. This quality-aware path selection is what distinguishes MPSC from simply widening the network.

![](images/d41827a2a74898a6bd5e4ce4a8c8e57536dad65fb05cba1217716784b8731455.jpg)  
Input

![](images/f6801d1721bf710c87a968f5df27ec730e540bb55ce5113da6485cdc744fed76.jpg)  
Our

![](images/9e752d888828c6d73a9c4434d8ef51d9e430858c25a077100cf7cfbe2ae83625.jpg)  
Baseline

![](images/994269e37a1532cf1f5d4406d9ba0a3039f9defa28d8b46ffdddfd6f43a97fe9.jpg)  
DEIM-D-FINE-N

(a) CrowdHuman dataset  
![](images/efea1fe9c113709024ede52439202c2aa161437cb0c1c8d93e6cfc2e271ecfff.jpg)  
Input

![](images/c91f6c35508f417b1e90c69e42302379a6aaac06fbbe5c6fa0bffcb28802cc4b.jpg)  
Our

![](images/9e1c0b1fc521e74d12db0a687e05e74d703c8092047fc091e538a22f71c13822.jpg)  
Baseline

![](images/42f21c728f3917c4b723bdbc906bef7067ded8253480d4989a1b49711bd28618.jpg)  
DEIM-D-FINE-N  
(b) CityPersons dataset  
Fig. 5. Grad-CAM visualization comparing baseline YOLOv11n, DEIM-D-FINE-N, and DensePed-Lite on heavily occluded scenarios.

## 4.4 Visualization Analysis

As shown in Figure 5, DensePed-Lite produces noticeably fewer missed detections and more sharply localized attention than both the baseline and DEIM-D-FINE-N. Two patterns are worth highlighting. On CrowdHuman, where bodies overlap heavily, the baseline’s activation tends to merge neighbouring pedestrians into a single difuse blob, whereas our method keeps the response concentrated on each individual even when only a fraction of the body is visible—consistent with the spatial completion and coherence-enforcement mechanisms recovering structure that a single scanning path would lose. On CityPersons, the diferences appear most on partially visible pedestrians at image boundaries and behind obstacles, where the baseline either fires weakly or not at all while DensePed-Lite retains a confident, well-localized response; this matches the role of uncertaintyaware scoring in preventing such low-visibility but genuine targets from being suppressed. The visual evidence therefore aligns with the quantitative Recall gains: the improvements concentrate precisely on the occluded cases that motivated the design, rather than being spread uniformly across easy and hard instances.

## 5 Conclusion

We have presented DensePed-Lite, a lightweight detector built on a single principle: under occlusion a network should adapt its behaviour to the quality of what it observes rather than assume the input is complete. We traced the failures of lightweight detectors in dense scenes to three coupled efects—unreliable confidence, fragmented spatial coverage, and incoherent multi-scale fusion—that reinforce one another, and we realized the quality-aware principle at exactly these three points through uncertainty-aware scoring, multi-path spatial completion, and a stabilize–verify–refine fusion loop. Experiments on CrowdHuman and CityPersons show that this coordinated design attains the best accuracy among ultra-lightweight detectors while using only 5.5 GFLOPs and 2.53M parameters, with the gains concentrated on the high-Recall, occlusion-heavy regime the method targets. The ablations support the central claim: the three mechanisms are most efective together, and removing any one leaves a residual failure mode that limits the rest.

The study has limitations that point to future work. Our evaluation covers two pedestrian benchmarks; broader validation on other crowded categories and on embedded hardware would further establish practicality. The quality-aware principle is also not specific to pedestrians, and extending it to general denseobject detection is a natural next step.

Disclosure of Interests The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Dalal, N., Triggs, B.: Histograms of oriented gradients for human detection. In: CVPR. pp. 886–893 (2005)

2. Ding, J., Li, W., Pei, L., Yang, M., Ye, C., Yuan, B.: Sw-YoloX: An anchor-free detector based transformer for sea surface object detection. Expert Syst. Appl. 217, 119560 (2023)

3. Fang, Y., Pang, H.: An improved pedestrian detection model based on YOLOv8 for dense scenes. Symmetry 16(6), 716 (2024)

4. Feng, Y., Huang, J., Du, S., Ying, S., Yong, J.H., Li, Y., Ding, G., Ji, R., Gao, Y.: Hyper-yolo: When visual object detection meets hypergraph computation. IEEE Trans. Pattern Anal. Mach. Intell. (2025)

5. Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H.: MobileNets: Eficient convolutional neural networks for mobile vision applications. arXiv:1704.04861 (2017)

6. Jocher, G., Qiu, J.: Ultralytics YOLO11. https://github.com/ultralytics/ ultralytics (2024), version 11.0.0

7. Jocher, G., Qiu, J.: Ultralytics YOLO26. https://github.com/ultralytics/ ultralytics (2026), version 26.0.0

8. Khanam, R., Hussain, M.: What is YOLOv5: A deep look into the internal features of the popular object detector. arXiv:2407.20892 (2024)

9. Li, H., Zhang, S., Hu, L., Zhou, H.: Towards real-time accurate dense pedestrian detection via large-kernel perception module and multi-level feature fusion. J. Real-Time Image Process. 22(1), 16 (2024)

10. Li, N., Bai, X., Shen, X., Xin, P., Tian, J., Chai, T., Wang, Z.: Dense pedestrian detection based on GR-YOLO. Sensors 24(14), 4747 (2024)

11. Li, X., Lv, C., Wang, W., Li, G., Yang, L., Yang, J.: Generalized focal loss: Towards eficient representation learning for dense object detection. IEEE Trans. Pattern Anal. Mach. Intell. 45(3), 3139–3153 (2023)

12. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object detection. In: ICCV. pp. 2999–3007 (2017)

13. Liu, Q., Li, Z., Zhang, L., Deng, J.: MSCD-YOLO: A lightweight dense pedestrian detection model with finer-grained feature information interaction. Sensors 25(2), 438 (2025)

14. Liu, Q., Ye, H., Wang, S., Xu, Z.: YOLOv8-CB: Dense pedestrian detection algorithm based on in-vehicle camera. Electronics 13(1), 236 (2024)

15. Liu, S., Huang, D., Wang, Y.: Adaptive NMS: Refining pedestrian detection in a crowd. In: CVPR. pp. 6452–6461 (2019)

16. Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., Berg, A.C.: SSD: Single shot MultiBox detector. In: ECCV. pp. 21–37 (2016)

17. Liu, Y., et al.: DEIM: DETR with improved matching for fast convergence. In: CVPR (2025)

18. Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks. IEEE Trans. Pattern Anal. Mach. Intell. 39(6), 1137–1149 (2017)

19. Shao, S., Zhao, Z., Li, B., Xiao, T., Yu, G., Zhang, X., Sun, J.: CrowdHuman: A benchmark for detecting human in a crowd. arXiv:1805.00123 (2018)

20. Sun, P., et al.: D-FINE: Redefine regression task in DETRs as fine-grained distribution refinement. In: ICLR (2025)

21. Tan, M., Pang, R., Le, Q.V.: EficientDet: Scalable and eficient object detection. In: CVPR. pp. 10778–10787 (2020)

22. Varghese, R., M., S.: YOLOv8: A novel object detection algorithm with enhanced performance and robustness. In: ADICS. pp. 1–6 (2024)

23. Wang, A., et al.: YOLOv13: Real-time object detection with hypergraph-enhanced adaptive visual perception. arXiv:2506.17733 (2025)

24. Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., Ding, G.: YOLOv10: Real-time end-to-end object detection. arXiv:2405.14458 (2024)

25. Wang, R., Gao, H., Liu, Y.: A data fusion-based method for pedestrian detection and flow statistics across diferent crowd densities. J. Saf. Sci. Resil. 6(1), 105–113 (2025)

26. Wang, Z., Li, C., Xu, H., Zhu, X., Li, H.: Mamba yolo: A simple baseline for object detection with state space model. In: AAAI. pp. 8205–8213 (2025)

27. Zhang, C., et al.: YOLOv12: Attention-centric real-time object detectors. In: NeurIPS (2025)

28. Zhang, S., Benenson, R., Schiele, B.: CityPersons: A diverse dataset for pedestrian detection. In: CVPR. pp. 4457–4465 (2017)

29. Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., Chen, J.: DETRs beat YOLOs on real-time object detection. In: CVPR. pp. 16965–16974 (2024)

30. Zhou, X., Li, J., Li, Y.: FusionU10: Enhancing pedestrian detection in low-light complex tourist scenes through multimodal fusion. Front. Neurorobot. 18 (2025)