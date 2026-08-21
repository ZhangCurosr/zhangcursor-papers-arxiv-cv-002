# Unified and Eficient Point-Line Local Features

François Costa<sup>1,⋆</sup>, Raphael Kreft<sup>1,∗</sup>, Eckhard Goedeke<sup>1</sup>, Felix Möller<sup>1</sup>, Hardik Shah<sup>1</sup>, Ramanathan Rajaraman<sup>1</sup>, Shaohui Liu<sup>1</sup>, Rémi Pautrat<sup>2</sup>, and Marc Pollefeys<sup>1,2</sup>

<sup>1</sup> Department of Computer Science, ETH Zurich, Switzerland <sup>2</sup> Microsoft Spatial AI Lab

Abstract. Multi-view computer vision pipelines typically rely on accurate sparse keypoints and robust descriptors. While incorporating line features has shown clear benefits for matching and pose estimation, existing point-line approaches remain ineficient: they detect points and lines separately, use increasingly heavy networks, and depend on CPU-bound heuristics that hinder real-time performance. We introduce a Unified Eficient Points and Lines (UPAL) feature extractor that jointly extracts keypoints, line segments, and feature descriptors within a single lightweight architecture. A shared backbone provides common representations that feed diferent branches for point and line features. Line segments are recovered through an accelerated post-processing stage, an enhanced and highly eficient variant of the LSD algorithm. UPAL matches or exceeds state-ofthe-art performance in both point and line applications while significantly reducing computational cost, achieving, for instance, a 4× speedup and 10× smaller memory footprint over the ALIKED + DeepLSD pipeline. Code is publicly available at https://github.com/francois141/upal.

Keywords: Fast local features · Point and line detectors and descriptors

## 1 Introduction

Extracting salient points and their corresponding descriptors is a fundamental step in many vision tasks involving multi-view geometry. Beyond 2D points, line segments are another type of feature that can complement points due to their prevalence in human-made structures, resulting in more informative image representations for geometric tasks. Accurately localizing line segments in images remains an active field of research due to several challenges, such as their extended size across images, their susceptibility to occlusion, and their sensitivity to perspective changes. In spite of these challenges, recent methods combining points and lines have demonstrated state-of-the-art results in multiple fields, such as feature matching [24, 41, 45, 67], 3D reconstruction [38, 39], visual localization [2,19,39,51,68,82,83], relative pose estimation [26], and Simultaneous Localization and Mapping (SLAM) [18, 22, 34, 36, 49, 52, 70, 71, 86].

![](images/5e9959ad9de950064060e941efea0a252d4c416ed41901ba34967fc5e29ca57c.jpg)  
Fig. 1: In this paper, we propose UPAL, a joint point-line feature extractor that outperforms most existing methods (here, on hybrid point-line visual localization with pose accuracy under 5cm / 5<sup>◦</sup>) while being substantially more lightweight. This is enabled by our compact architecture, joint point-line training with distilled supervision, and an eficient variant of the LSD algorithm. Results are shown on the Stairs scene of the 7Scenes dataset [62] with the setup described in Sec. 4.4.

Despite their efectiveness, such vision pipelines using both points and lines sufer from notable computational drawbacks. They often require two separate extractors: one for keypoint detection and description, and another for line localization. This approach is ineficient, as the two tasks are strongly related and could share computations, since points and lines are both localized in regions with strong image gradients. Moreover, recent network architectures tend to grow in size, making both point and line extraction computationally expensive, requiring high-end GPUs to achieve satisfactory throughput. While recent line detectors have pushed performance for feature extraction and matching, little efort has been invested in reducing the size of the networks and making them eficient. Furthermore, some state-of-the-art deep line detectors reuse classical algorithms such as LSD [20] to extract their line segments [44, 61, 65]. However, LSD is a CPU-only algorithm that contains operations that could be implemented more eficiently on GPU. In addition, it sufers from quadratic complexity with respect to the size of the image.

Motivated by these observations, we introduce Unified Eficient Points and Lines (UPAL) local features, a novel lightweight feature extractor that extracts keypoints, line segments, and feature descriptors jointly in a single forward pass. UPAL leverages the architecture of a point feature extractor and adds a lightweight line detection branch on top of it. Without resorting to costly learned line descriptors or matchers [45, 67, 76], we argue that lines can be eficiently matched by first matching their endpoints and then finding the optimal assignment, thus requiring no extra line descriptors. While we are not the first to propose joint point-line feature extraction [17, 70], previous attempts failed to reach state-of-the-art results for both features, despite maintaining high throughput. With UPAL, we show how to reach such goals with a streamlined design. Through distillation, UPAL combines the best of both worlds: it leverages an eficient and lightweight neural network while being supervised by powerful state-of-the-art models. We carefully reviewed the best point and line extractors to date, and selected the best-performing combination to distill their knowledge into our architecture. By eliminating redundant computations in the extraction of the two kinds of features (points and lines), UPAL preserves state-of-the-art performance while improving eficiency and memory footprint. In particular, we demonstrate that state-of-the-art results in line detection can be achieved by adding only three convolutional layers to an existing point extractor.

Inspired by DeepLSD [44], we leverage a post-processing step based on the LSD [20] algorithm to extract the final line segments. We introduce three major improvements over the original LSD and DeepLSD implementations. First, we show that discarding the branch predicting an angle field can simplify the architecture and at the same time improve the performance. Second, we move most of LSD’s image processing to the GPU and leave only the line-growing stage on CPU. Lastly, we propose a simple but efective heuristic to reduce the number of seed points in LSD, increasing the eficiency of the detector without any loss in performance.

We evaluate UPAL on both low-level metrics and geometric downstream tasks, demonstrating its efectiveness across multiple challenging datasets. Combined with our optimized line extraction module, UPAL achieves a 4× speedup over the current state of the art, with similar or better performance. Moreover, approaches with comparable performance have a memory footprint that is at least one order of magnitude higher.

In summary, our contributions are as follows:

1. We demonstrate how to distill state-of-the-art point and line detectors and descriptors into a single lightweight network, preserving, or even improving, their original accuracy. This design substantially simplifies existing pipelines, requiring only a few additional convolutions for line extraction.

2. We introduce an enhanced variant of LSD that provides significantly higher eficiency through GPU-accelerated processing.

3. UPAL achieves state-of-the-art performance for both point and line features, across low-level metrics and downstream tasks, while running much faster than most prior approaches and being significantly lighter, making it more suitable for embedded applications.

## 2 Related work

Point Feature Extractors. Feature points are traditionally detected and described using patterns in the image gradient [5, 25, 40, 54]. The deep learning era has subsequently introduced novel methods to jointly detect and describe feature points within a single network. The pioneering work SuperPoint [10] introduced the process of homography adaptation to create a pseudo ground truth of corner-like keypoints. D2-Net [11] showed how to describe first, then extract keypoints from the features. R2D2 [53] introduced the concept of reliability to only keep the most relevant keypoints. DISK [66] revisited how to learn reliable keypoints through reinforcement learning. ALIKED [80] was proposed as a lightweight network and remains among the top performers, which motivated our decision to follow the same architecture. SiLK [21] demonstrated that a simple setup can already lead to a strong performance. Progress has also been made in terms of eficiency with XFeat [48], which made it possible to run feature extraction even on embedded devices. The DeDoDe work [14] advocates for using independent networks for detection and description to maximize performance. Finally, a recent keypoint detector, DaD [13], revisited the reinforcement learning objective to learn more robust points. In this work, we reuse the architecture of ALIKED including its sparse descriptors branch, and leverage both the SuperPoint and DaD keypoints to supervise our keypoint detector.

![](images/9da057cc90a250cdaa5c8bc2c8fb337a885efe83bf7f255b041ea0e251ccfe73.jpg)  
Fig. 2: Overview of our network architecture. We use a shared convolutional encoder to generate a multi-scale feature map (orange). We then use three decoder branches: a keypoint detection branch (blue) with a Score Map Head (SMH) followed by Diferentiable Keypoint Detection (DKD), a Sparse Deformable Descriptor Head (SDDH) to obtain descriptors (purple), and a branch predicting a distance field (green). The latter is then passed to our fast LSD to extract the line segments.

Line Feature Extractors. Line features have recently emerged as an alternative to point features. As with points, traditional line detectors rely on image gradients to group pixels with strong, coherent gradient magnitude and orientation [3, 16, 20, 55, 64, 78]. With the rise of deep learning, wireframe extraction, detecting interconnected lines that capture the main structure of indoor scenes, has become a popular task [27], leading to numerous wireframe-detection networks [27, 42, 73–75, 79, 84]. In parallel, more general deep line detectors have been proposed to detect all kinds of lines, not only structural ones [9, 23, 28, 30, 44, 61, 65, 72]. Following a trend similar to point features, joint line detection and description has also been explored [1, 47, 77]. Among these methods, DeepLSD [44] and ScaleLSD [30] stand out for geometric tasks: the former combines a learned image gradient with LSD [20], while the latter scales network capacity and training data. In this work, we follow the DeepLSD design by predicting a deep image gradient and using LSD to extract segments, but introduce multiple performance and eficiency improvements to both the learned and handcrafted components.

Joint Point-Line Applications. Points and lines have long been shown to be complementary across a wide range of applications, consistently outperforming approaches relying on points or lines alone. PLNet [70] and Wireframe [17] pioneered the joint detection and description of points and lines within a single network. However, this joint prediction came at the cost of lower performance compared to individual best-in-class baselines. Joint point-line matching is a key component in many multi-view pipelines and has recently been addressed using graph neural networks [24, 41, 45, 67]. This complementarity is particularly valuable in SLAM, where the longer spatial extent of lines improves feature tracking and reduces drift [18, 22, 34, 36, 49, 52, 70, 71, 86]. Lines also provide additional geometric structure beneficial for 3D reconstruction, and recent work has shown that combining points and lines improves both accuracy and completeness [38,39]. Although lines alone do not fully constrain relative pose between two views, they ofer useful cues through endpoints and vanishing points, efectively complementing points [26]. Finally, point-line combinations have been widely explored for visual localization [2, 19, 51, 68, 82, 83], where lines are advantageous in low-texture scenes, an established failure mode for point-based methods.

## 3 Method

We present a unified pipeline that jointly predicts points, lines, and feature descriptors using a single neural network, reducing inference time and memory footprint while preserving or exceeding the performance of separate detectors. The model is trained in a student-teacher framework that distills multiple stateof-the-art keypoint and line detectors. An overview of the architecture is provided in Fig. 2.

Design Choices. Our objective is to match the performance of independent feature extractors while distilling their capabilities into a single lightweight network for faster inference and a lower memory footprint. As dense matching remains computationally costly, we focus on the classical paradigm of detecting and describing features in a single image. We extensively evaluated point detectors, line detectors, and feature descriptors in isolation and identified the following strong performers among existing methods: SuperPoint [10] and DaD [13] for keypoints, ALIKED [80] for descriptors, and DeepLSD [44] for line detection. To unify all methods, we adopt the architecture of ALIKED [80] for its low latency and memory footprint and extend it with additional heads for the new feature types.

![](images/4a3ca67100f29a7776982d9a2367c976d50251bb7a286a80bb1471dea8e70130.jpg)  
(a) Input image

![](images/ca91122282736c5e3a981a1ece1f2bea37d10ebb52c8fdfad703066589db198f.jpg)  
(b) Bottleneck features

![](images/3bf40b90e275b6ffae7cd4b2bd5a5daed7cf7df3f753d7895add4a9bfce0be49.jpg)  
(c) Keypoint score map

![](images/dfa7d1009099da4feffa0fcd9ce5c51396d2964d885fd9eb772d8d6396a07365.jpg)  
(d) Line distance field  
Fig. 3: Visualization of the outputs of the network. The shared backbone receives an input image (a) and encodes it into bottleneck features (b). We observe that the predicted keypoint score map (c) and line distance field (d) show some similarity (e.g. keypoints are often line endpoints or intersections), justifying their joint prediction.

## 3.1 Encoder

The encoder follows ALIKED [80] and consists of four convolutional blocks with progressively reduced resolution; the last two employ Deformable Convolutions (DCN) [85] to improve geometric invariance. Each block’s output is passed through a $1 \times 1$ convolution, then upsampled to the original resolution and finally concatenated with the other blocks’ outputs to form the bottleneck F. This representation captures both low- and high-level cues and supports accurate point detection, descriptor computation, and distance-field prediction. Figure 3 illustrates a keypoint score map, a line distance field, and a PCA visualization of F, showing that the bottleneck can encode information for both points and lines, as keypoints frequently lie on line segments.

## 3.2 Point Extraction

Starting from the bottleneck features F, we use the Score Map Head (SMH) from ALIKED [80], a lightweight convolutional head, to extract a keypoint probability heatmap $\mathbf { \bar { S } } \in [ 0 , 1 ] ^ { H \times W }$ . This map is then fed into a Diferentiable Keypoint Detection (DKD) module [81], which applies Non-Maximum Suppression (NMS) thresholding to obtain a set of N keypoints $\{ \mathbf { p _ { i } } \} , i \in \{ 1 , \dots , N \}$ , and softargmax within a $3 \times 3$ window to obtain subpixel-accurate coordinates.

As in ALIKED [80], we use the Sparse Deformable Descriptor Head (SDDH). For each predicted keypoint p<sub>i</sub>, a $K \times K$ window is extracted around $p _ { i }$ from the bottleneck F, and a small module predicts M sampling positions. These sampling positions are then used to sample the bottleneck features with bilinear interpolation, followed by a $1 \times 1$ convolution and SELU activation. The sampled features are then aggregated into a single descriptor $\mathbf { d _ { i } } \in \mathcal { R } ^ { D }$ through a weighted sum with learned weights. This produces a deformable descriptor that can incorporate information from various parts of the image and thus easily adapt to geometric deformations.

## 3.3 Distance Field Prediction

State-of-the-art line detectors such as DeepLSD [44] and ScaleLSD [30] predict an attraction field before extracting line segments from it. DeepLSD regresses in particular both a distance and an angle field, corresponding to dense maps where each pixel value encodes the distance to the nearest line and the orientation of that line. While experimenting with DeepLSD, we discovered that the angle field predicted by DeepLSD is detrimental, as will be shown in Table 8. Instead of predicting an angle field, we observed that directly using image-gradient orientation yielded better performance, while reducing the complexity of the network. Thus, the line decoder branch of our architecture only predicts a line distance field and we compute the angle field using a GPU-based image gradient angle computation.

Furthermore, DeepLSD relies on a deep and computationally expensive U-Net backbone, which contains roughly 10× more learnable parameters than ALIKED. Since predicting a distance field is a relatively low-level task, we argue and demonstrate that a smaller-scale backbone is suficient to produce equivalent results. In Sec. 4 we show that our lighter backbone network, followed by the same line decoder branch as in DeepLSD, can obtain results similar to or better than those of DeepLSD, while reducing the number of learnable parameters by a factor of 10. This line decoder consists only of 3 convolutions with batch normalization and ReLU activation, making it extremely lightweight. Overall, this makes the network architecture very lightweight and eficient.

## 3.4 Line Extraction

Similar to DeepLSD [44], our network predicts the distance field and delegates line extraction to the handcrafted LSD detector [20]. LSD is an expensive, twostage, CPU-only algorithm. It first computes the image gradient (or receives it from a learned detector), and performs an approximate sorting of the gradient magnitude of the pixels. Then, it iterates over the nearly sorted points to start drawing lines, starting from the most promising ones with the highest gradient magnitudes. This sorting-and-seeding step is expensive on CPU and the current LSD implementation has a quadratic cost with respect to the image side lengths, as it is proportional to the number of pixels in the image.

In this work, we propose an accelerated variant of LSD, which removes the expensive approximate sorting by moving the extraction of useful seed points directly to the GPU. Following the observation that most points with low gradient values never contribute to lines and can therefore be discarded, we reduce the number of seed points. We empirically determined that downsampling the grid of seed points with a stride of 2 and keeping the top 20% of pixels with the lowest distance-field values has no impact on the performance of the detector, while improving the eficiency by a factor of 3.

## 3.5 Losses

The keypoint heatmap \mathbf {S} is supervised using the teacher prediction $\hat { \bf S }$ with a weighted binary cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { w b c e } } ( \mathbf { S } , \hat { \mathbf { S } } ) = - \lambda \hat { \mathbf { S } } \log ( \mathbf { S } ) - ( 1 - \hat { \mathbf { S } } ) \log ( 1 - \mathbf { S } ) .\tag{1}
$$

![](images/784d1188604086ade728058ce28a28c7a5196f6821ff9de22d7b0dfaf40c5f27.jpg)  
Fig. 4: Line detection examples. In spite of its eficiency, UPAL outputs generic lines with quality similar to more expensive methods.

The weighting factor \lambda compensates for the imbalance between positive keypoint pixels and the predominantly near-zero background values in the pseudo-ground truth. Motivated by our empirical analysis of keypoint detectors, we construct the teacher heatmap by taking the element-wise maximum of the predictions from SuperPoint [10] and DaD [13].

The line distance field \mathbf {D} is supervised with an L1 loss against the groundtruth map \mathbf {\hat {D} generated by DeepLSD [44]. Following DeepLSD, we restrict this loss to pixels within a radius $r = 5$ of each ground-truth line, which improves convergence and accuracy by focusing the supervision on the most informative regions. We also normalize both the prediction and the ground truth to yield more meaningful gradients in the vicinity of lines:

$$
\mathcal { L } _ { D } = \left. \hat { \mathbf { D _ { n } } } - \mathbf { D _ { n } } \right. , \qquad \mathbf { D _ { n } } = - \log \left( \frac { \mathbf { D } } { r } \right) .\tag{2}
$$

Local descriptors $\mathbf { d _ { i } }$ are supervised using ground-truth descriptors $\hat { \mathbf { d _ { i } } }$ with an L1 loss averaged over the top $n = 1 0 0 0$ detected keypoints:

$$
\mathcal { L } _ { \mathrm { d e s c } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left\| \hat { \mathbf { d } } _ { \mathbf { i } } - \mathbf { d } _ { \mathbf { i } } \right\| .\tag{3}
$$

The ground-truth descriptors are obtained by querying ALIKED-n32 [80] at the predicted keypoint locations.

The final training objective is the sum of all components:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { w b c e } } + \mathcal { L } _ { D } + \mathcal { L } _ { \mathrm { d e s c } } . } \end{array}\tag{4}
$$

## 3.6 Implementation Details

Our encoder and keypoint branch follow the ALIKED-n16 architecture [80], while the distance-field branch is adapted from DeepLSD [44]. We train on 10k distractor images from the Oxford-Paris dataset [50] at a resolution of 800 × 800. This dataset contains a mix of indoor and outdoor scenes, making our model suitable for both conditions. We set $K = 3$ in the SDDH, use 128-dimensional descriptors, apply a keypoint loss weight of $\lambda = 2 0 0$ , and optimize with Adam at a learning rate of 1e−4.

Table 1: Homography estimation on HPatches [4]. We report the latency in ms and the Area Under the Curve (AUC) of correctly predicted homographies for various pixel error thresholds. Methods are grouped into lightweight and heavyweight categories. Within each category, the best result is shown in bold, and the second best is underlined.
<table><tr><td>Method</td><td>AUC @1px ↑</td><td>AUC @3px ↑</td><td>AUC @5px ↑</td><td>Latency (ms) ↓</td></tr><tr><td colspan="5">Lightweight Methods</td></tr><tr><td>SIFT [40]</td><td>35.1</td><td>65.0</td><td>75.3</td><td>237</td></tr><tr><td>SuperPoint [10]</td><td>35.6</td><td>65.2</td><td>75.7</td><td>50</td></tr><tr><td>DISK [66]</td><td>29.9</td><td>61.8</td><td>73.2</td><td>100</td></tr><tr><td>XFeat [48]</td><td>34.4</td><td>62.2</td><td>73.3</td><td>44</td></tr><tr><td>Wireframe [17]</td><td>38.3</td><td>66.4</td><td>76.1</td><td>308</td></tr><tr><td>PLNet [70]</td><td>35.5</td><td>67.0</td><td>77.4</td><td>158</td></tr><tr><td>UPAL (Ours)</td><td>39.2</td><td>67.9</td><td>77.7</td><td>62</td></tr><tr><td colspan="5">Heavyweight Methods</td></tr><tr><td>DeDoDe v2 [15]</td><td>40.9</td><td>69.1</td><td>78.4</td><td>348</td></tr><tr><td>DaD [13] + DeDoDe v2 [15]</td><td>39.7</td><td>68.6</td><td>78.6</td><td>164</td></tr><tr><td>Ripe [31]</td><td>37.5</td><td>58.9</td><td>69.3</td><td>193</td></tr><tr><td>RDD [6]]</td><td>34.6</td><td>60.0</td><td>68.4</td><td>522</td></tr></table>

Training proceeds in two stages: we first learn the distance-field branch for about one hour, using early stopping after a single epoch, then freeze the encoder and distance-field module and train the remaining components for 60 epochs (about 12 hours). ALIKED-derived components are initialized with their pretrained weights. Training is performed on four NVIDIA TITAN GPUs (24 GB each) with a batch size of 4 per GPU.

## 4 Experiments

We evaluate UPAL, first showing that it matches or exceeds the state of the art in point features, line segment detection, and point-line applications. We then highlight its eficiency benefits in terms of inference speed and memory footprint and provide an ablation study of our design choices. A visual comparison of predicted lines is shown in Figure 4.

## 4.1 Point Evaluation

We begin by assessing UPAL against state-of-the-art point extractors, verifying that adding the line-detection branch does not degrade point performance. We use three benchmarks that evaluate point matching, jointly measuring keypoint and descriptor quality. For each benchmark, point features are extracted from image pairs and matched via mutual nearest neighbors. We compare UPAL to the following lightweight methods: SIFT [40], SuperPoint [10], DISK [66], ALIKEDn16 [80] (sharing the same architecture as ours), XFeat [48], the Wireframe point detector and descriptor [17], and PLNet [70]. We also add more heavyweight baselines (more than 10M params) as a comparison: DeDoDe v2 [15], DaD [13] keypoints paired with DeDoDe v2 descriptors, Ripe [31], and RDD [6].

Table 2: Relative pose estimation on MegaDepth [35]. The latency and AUC of correctly retrieved poses under various angular error thresholds are reported. Methods are grouped into lightweight and heavyweight categories. Within each category, the best result is highlighted in bold and the second best is underlined. Results reported in parentheses are not considered for ranking due to overlap between the training data and the MegaDepth benchmark test set.
<table><tr><td>Method</td><td>AUC @5°↑</td><td>AUC @10°↑</td><td>AUC @20°↑</td><td>Latency (ms) ↓</td></tr><tr><td colspan="5">Lightweight Methods</td></tr><tr><td>SIFT [40]</td><td>40.1</td><td>53.6</td><td>64.4</td><td>729</td></tr><tr><td>SuperPoint [10]</td><td>51.3</td><td>64.4</td><td>73.7</td><td>78</td></tr><tr><td>DISK [66]</td><td>53.9</td><td>66.1</td><td>75.3</td><td>335</td></tr><tr><td>ALIKÊD [80]</td><td>59.8</td><td>72.0</td><td>81.1</td><td>79</td></tr><tr><td>XFeat [48]</td><td>44.6</td><td>59.0</td><td>70.2</td><td>28</td></tr><tr><td>Wireframe [17]</td><td>47.7</td><td>60.7</td><td>70.8</td><td>307</td></tr><tr><td>PLNet [70]</td><td>37.1</td><td>51.6</td><td>63.9</td><td>288</td></tr><tr><td>UPAL (Ours)</td><td>58.2</td><td>70.8</td><td>79.4</td><td>80</td></tr><tr><td colspan="5">Heavyweight Methods</td></tr><tr><td>Ripe [31]</td><td>58.2</td><td>70.7</td><td>79.8</td><td>751</td></tr><tr><td>DeDoDe v2 [15]</td><td>57.2</td><td>69.6</td><td>78.8</td><td>1142</td></tr><tr><td>DaD [13]+DeDoDe v2 [15]</td><td>60.3</td><td>72.8</td><td>81.6</td><td>485</td></tr><tr><td>RDD [6]3</td><td>(64.8)</td><td>(77.2)</td><td>(85.8)</td><td>04</td></tr></table>

Homography Estimation. The HPatches [4] dataset contains 580 image pairs, with either illumination or viewpoint variations. We resize the longest side of the images to 800 pixels while keeping the aspect ratio, and extract the top 1024 keypoints per image. The benchmark evaluates the estimation of the dominant planar homography from keypoint correspondences. After matching, we use PoseLib [32] to estimate the homography, and evaluate the Area Under the Curve (AUC) of the reprojection error of the four image corners at thresholds of 1, 3, and 5 pixels, as in [58]. Results can be found in Tab. 1.

Outdoor Relative Pose Estimation. The MegaDepth [35] benchmark for feature matching contains 1,500 image pairs sampled with various levels of overlap, taken from two outdoor landmark scenes of the original MegaDepth dataset. For this benchmark, we resize images to 1600 pixels on the long side and use the top 2048 keypoints for each extractor. After matching, we leverage PoseLib [32] to estimate the relative translation and rotation from the correspondences, using the 5-point algorithm [43] to obtain a robust estimate. The results are returned as the AUC of the pose error (maximum of the angular error in both translation and rotation) at three error thresholds. Results are shown in Tab. 2.

Table 3: Relative pose estimation on ScanNet [8]. The latency and AUC of correctly estimated poses under various angular error thresholds are reported. Methods are grouped into lightweight and heavyweight categories. Within each category, the best result is highlighted in bold and the second best is underlined.
<table><tr><td>Method</td><td>AUC @5°↑</td><td>AUC @10°↑</td><td>AUC @20°↑</td><td>Latency (ms)</td></tr><tr><td colspan="5">Lightweight Methods</td></tr><tr><td>SIFT [40]</td><td>6.7</td><td>14.9</td><td>24.2</td><td>742</td></tr><tr><td>SuperPoint [10]</td><td>15.5</td><td>29.8</td><td>43.4</td><td>83</td></tr><tr><td>DISK [66]</td><td>10.1</td><td>19.2</td><td>29.5</td><td>348</td></tr><tr><td>ALIKÊD [80]</td><td>6.9</td><td>13.3</td><td>20.1</td><td>82</td></tr><tr><td>XFeat [48]</td><td>16.4</td><td>31.2</td><td>45.7</td><td>27</td></tr><tr><td>Wireframe [17]</td><td>11.5</td><td>22.3</td><td>34.5</td><td>293</td></tr><tr><td>PLNet [70]</td><td>9.5</td><td>20.7</td><td>34.8</td><td>427</td></tr><tr><td>UPAL (Ours)</td><td>15.6</td><td>28.9</td><td>41.4</td><td>83</td></tr><tr><td colspan="5">Heavyweight Methods</td></tr><tr><td>Ripe [31]</td><td>5.5</td><td>11.6</td><td>18.9</td><td>782</td></tr><tr><td>DeDoDe v2 [15]</td><td>9.6</td><td>18.5</td><td>28.4</td><td>1135</td></tr><tr><td>DaD [13]+DeDoDe v2 [15]</td><td>14.6</td><td>27.9</td><td>41.6</td><td>462</td></tr><tr><td>RDD [6]</td><td>14.2</td><td>28.2</td><td>42.7</td><td>5</td></tr></table>

Indoor Relative Pose Estimation. ScanNet [8] is a large-scale RGB-D indoor dataset, from which [58] sampled 1,500 overlapping image pairs with ground-truth poses. Indoor scenes are particularly challenging due to repetitive structures and texture-less regions, although line features are abundant and can efectively complement point features. The evaluation protocol is identical to that of the previous section, and Tab. 3 shows the results.

Discussion of the Point Evaluation. While UPAL is mainly distilled from existing methods for the point branch, it consistently ranks in the top three among all these benchmarks. The first reason is that we explicitly selected the best teacher for each component. For example, while ALIKED [80] has generally the best descriptors overall, we noticed that its points underperform in indoor scenarios, as can be seen in Tab. 3, probably because of its unsupervised training and lack of indoor training data. In contrast, SuperPoint [10] keypoints are supervised as corners and naturally work better in indoor scenarios where corners are abundant. We also empirically found that mixing SuperPoint with DaD [13] keypoints could further boost the overall performance. A second reason for the increased performance of our approach is that jointly detecting points and lines benefits the network by helping it identify more stable keypoints along line structures, which are naturally abundant in indoor environments.

Table 4: Line detection evaluation on HPatches [4] and RDNIM [46]. We compare the line localization error (Loc Err) on a fixed set of l lines, homography estimation (H Estim), and line repeatability (Rep) at various pixel thresholds. Horizontal lines separate Traditional / Learned / Joint methods.
<table><tr><td rowspan="3"></td><td colspan="9">HPatches</td><td colspan="9">RDNIM</td></tr><tr><td colspan="2">Loc Err</td><td colspan="3">H Estim</td><td colspan="3">Rep</td><td>Time</td><td colspan="2">Loc Err</td><td colspan="3">H Estim</td><td colspan="3">Rep</td><td>Time</td></tr><tr><td>501</td><td>3001</td><td>1px</td><td>3px</td><td>5px</td><td>1px</td><td>3px</td><td>5px</td><td>(ms)</td><td>501</td><td>3001</td><td>1px</td><td>3px</td><td>5px</td><td>1px</td><td>3px</td><td>5px</td><td>(ms)</td></tr><tr><td>LSD [20] ELSED [64]</td><td>0.50 0.71</td><td>1.42 1.63</td><td>58.3 40.7</td><td>88.3 65.8</td><td>90.9</td><td>26.4</td><td>53.6</td><td>61.6</td><td>150</td><td>1.78</td><td>1.85</td><td>7.4</td><td>44.8</td><td>57.6</td><td>9.2</td><td>30.0</td><td>37.7</td><td>85</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>67.9</td><td>18.8</td><td>41.8</td><td>50.8</td><td>67</td><td>1.92</td><td>1.96</td><td>5.9</td><td>30.8</td><td>40.9</td><td>8.9</td><td>25.9</td><td>30.9</td><td>40</td></tr><tr><td>TP-LSD [28]</td><td>1.49</td><td>1.61</td><td>10.9</td><td>34.6</td><td>42.6</td><td>11.6</td><td>42.9</td><td>57.2</td><td>65</td><td>1.93</td><td>1.93</td><td>0.2</td><td>6.3</td><td>14.2</td><td>4.9</td><td>33.3</td><td>47.5</td><td>52</td></tr><tr><td>SOLD2 [47]</td><td>1.09</td><td>1.29</td><td>23.7</td><td>48.5</td><td>55.4</td><td>19.8</td><td>44.9</td><td>51.5</td><td>745</td><td>1.39</td><td>1.41</td><td>0.6</td><td>9.4</td><td>16.0</td><td>8.8</td><td>20.0</td><td>24.0</td><td>546</td></tr><tr><td>M-LSD [23]</td><td>2.10 0.55</td><td>2.21</td><td>15.0</td><td>54.1</td><td>63</td><td>8.9</td><td>35.6</td><td>51.2</td><td>91</td><td>2.41</td><td>2.41</td><td>0.5</td><td>13.4</td><td>26.0</td><td>4.3</td><td>22.6</td><td>34.9</td><td>70</td></tr><tr><td>DeepLSD [44]</td><td>1.81</td><td>1.43 2.03</td><td>58.1</td><td>87.4</td><td>90.6</td><td>27.2</td><td>54.1</td><td>61.8</td><td>473</td><td>1.73</td><td>1.81</td><td>9.4</td><td>45.3</td><td>60.5</td><td>10.1</td><td>31.8</td><td>39.5</td><td>294</td></tr><tr><td>Linea [29]</td><td>0.57</td><td>1.56</td><td>15.9</td><td>46.5</td><td>53.0</td><td>8.3</td><td>35.3</td><td>50.0</td><td>63</td><td>2.25</td><td>2.26</td><td>0.3</td><td>9.0</td><td>17.1</td><td>2.6</td><td>25.0</td><td>40.0</td><td>44</td></tr><tr><td>ScaleLSD [30]</td><td></td><td></td><td>53.3</td><td>87.6</td><td>91.1</td><td>20.2</td><td>46.5</td><td>55.2</td><td>177</td><td>1.71</td><td>1.83</td><td>9.8</td><td>47.4</td><td>63.4</td><td>9.1</td><td>25.0</td><td>31.4</td><td>166</td></tr><tr><td>Wireframe [17]</td><td>1.05</td><td>1.67</td><td>33.7</td><td>72.8</td><td>79.4</td><td>16.7</td><td>43.7</td><td>52.3</td><td>294</td><td>1.88</td><td>1.90</td><td>2.0</td><td>20.4</td><td>31.4</td><td>7.3</td><td>25.0</td><td>31.9</td><td>291</td></tr><tr><td>PLNet [70]</td><td>0.39</td><td>1.57</td><td>41.9</td><td>67.0</td><td>71.7</td><td>11.5</td><td>30.4</td><td>40.0</td><td>240</td><td>0.99</td><td>2.35</td><td>8.3</td><td>49.4</td><td>66.8</td><td>4.0</td><td>16.1</td><td>24.4</td><td>199</td></tr><tr><td>UPAL (Ours)</td><td>0.51</td><td>1.42</td><td>57.0</td><td>88.1</td><td>90.9</td><td>27.2</td><td>54.1</td><td>61.9</td><td>141</td><td>1.62</td><td>1.69</td><td>9.6</td><td>45.3</td><td>58.0</td><td>13.3</td><td>35.9</td><td>43.6</td><td>83</td></tr></table>

## 4.2 Line Evaluation

We evaluate our line detection on two standard datasets to test the robustness of the methods: HPatches [4] and RDNIM [46]. The latter consists of 1,722 image pairs related by a homography and with challenging day-night variations.

Following [44, 47], we evaluate the repeatability and localization error metrics. Since our teacher’s lines are semi-supervised and generic [44], evaluating on a biased set of lines such as in the Wireframe dataset [27] would not provide a fair comparison for generic line detection, and we instead report quality metrics of our lines for downstream tasks. Given a pair of images with ground-truth homography, we detect line segments in both images and establish one-to-one correspondences by warping lines between frames. To decide which pairs should be considered as a match, we compute the symmetric orthogonal distance: for each pair of lines, we calculate the average distance from both endpoints of a segment to the other line, compute this bidirectionally, and take the mean. Repeatability measures the ratio of lines whose match has an error below 1, 3, and 5 pixels, and the localization error returns the average distance of the 10, 50, and 300 most accurate matches. We also compute a homography estimation score, as in [10, 44]. Reusing the previous line matches, we sample minimal sets of 4 line matches and run LO-RANSAC [33] with the orthogonal line distance as reprojection error to estimate a homography.

Tab. 4 compares UPAL to two classical detectors: LSD [20] and ELSED [64]; the recent learned baselines TP-LSD [28], SOLD2 [47], M-LSD tiny [23], DeepLSD [44], Linea [29], and ScaleLSD [30]; and the joint point-line extractors Wireframe [17] and PLNet [70]. For LSD, we use the implementation available at https: //github.com/iago- suarez/pytlsd and otherwise use the authors’ implementations with default parameters for all the other models. Despite its light backbone, UPAL has the highest repeatability across the board and often ranks second in terms of localization error and homography estimation. This shows that line detection is a low-level task that can be solved with as few as 780k

Table 5: Visual localization on Stairs scene of the 7Scenes dataset [62]. We report the median translation / rotation errors (cm / deg) and pose accuracy at 5 cm / 5° threshold. Please refer to our supplementary material for results on all scenes.
<table><tr><td></td><td>Method</td><td>T / R err. ↓</td><td>Acc. ↑</td></tr><tr><td rowspan="5">Points</td><td>SuperPoint [10]</td><td>6.2 /1.65</td><td>36.6</td></tr><tr><td>DISK [66]</td><td>7.2 / 2.16</td><td>28.5</td></tr><tr><td>ALIKÈD [80]</td><td>7.8 / 1.98</td><td>35.0</td></tr><tr><td>PLNet [70]</td><td>6.2 / 1.68</td><td>38.2</td></tr><tr><td>UPAL (Ours)</td><td>5.1 / 1.39</td><td>49.1</td></tr><tr><td rowspan="8">Points</td><td>SuperPoint [10]+DeepLSD [44]</td><td>5.0 / 1.33</td><td>49.6</td></tr><tr><td>ALIKED [80]+TP-LSD [28]</td><td>5.2 / 1.48</td><td>48.8</td></tr><tr><td>ALIKED [80]+M-LSD [23]</td><td>5.2 / 1.52</td><td>48.5</td></tr><tr><td>ALIKED [80]+DeepLSD [44]</td><td>5.1 / 1.48</td><td>49.6</td></tr><tr><td>Wireframe [17]</td><td>4.6 / 1.30</td><td>53.8</td></tr><tr><td>PLNet [70]</td><td>5.0 / 1.36</td><td>50.1</td></tr><tr><td>UPAL (Ours)</td><td>4.4 / 1.23</td><td>54.6</td></tr><tr><td></td><td></td><td></td></tr></table>

parameters. Notably, UPAL improves over its teacher, DeepLSD [44], on the RDNIM dataset. This is due to the removal of the angle field prediction, which yields inaccurate predictions on such challenging images.

## 4.3 Point-Line Applications

Visual Localization. We evaluate the combination of points and lines for the task of visual localization using lightweight detectors. We use the 7Scenes dataset [62], consisting of 7 indoor scenes with RGB images and ground-truth poses. We focus here on the scene with the most lines, Stairs, and report the other scenes in the supplementary. We do not use depth information in this experiment. We run the point SfM on database images and find the camera poses of query images using the hloc [56, 57] framework. We then add line segments to the 3D model and perform pose estimation of the query images with a combination of points and lines using the LIMAP framework [39]. Line matching is performed as originally proposed in [45]: we extract descriptors at line endpoints, create an assignment matrix between the lines of both images by summing the matching scores of both endpoints, and finally solve for the best assignment with the Sinkhorn algorithm [7, 63]. For each image, we extract a maximum of 2048 keypoints and 3000 lines, and resize the maximum side of the image to 1024 pixels. Camera poses are estimated using the solvers in PoseLib [32]. We report the median translation and rotation error on the retrieved camera pose, as well as the pose estimation accuracy at a 5cm $/ 5 ^ { \circ }$ threshold.

In Tab. 5 we compare the results of point-only pipelines with point-line approaches and first observe that UPAL achieves the best results among all point baselines. This aligns with Sec. 4.1, where UPAL excels in indoor scenarios. We attribute this to joint training with lines, encouraging keypoints to be better localized on lines, which are abundant in indoor scenarios. Second, the addition of lines improves results over points in isolation, especially in scenes that are challenging for keypoints but rich in lines, like the Stairs scene. This aligns with existing works [39, 45] which demonstrated the complementarity of lines with respect to points. Finally, UPAL with points and lines has the best overall results, showcasing the synergy of jointly learned features.

Table 6: Evaluation on multi-view point triangulation on ETH3D [60].
<table><tr><td>Method</td><td colspan="3">Completeness (%)</td><td colspan="3">Accuracy (%)</td></tr><tr><td></td><td>1cm</td><td>2cm</td><td>5cm</td><td>1cm</td><td>2cm</td><td>5cm</td></tr><tr><td>SuperPoint [10]</td><td>0.17</td><td>0.81</td><td>4.34</td><td>49.52</td><td>64.73</td><td>80.01</td></tr><tr><td>DISK [66]</td><td>0.33</td><td>1.18</td><td>4.41</td><td>69.22</td><td>81.10</td><td>91.29</td></tr><tr><td>ALIKÈD [80]</td><td>0.13</td><td>0.56</td><td>2.76</td><td>56.61</td><td>70.78</td><td>85.18</td></tr><tr><td>Wireframe [17]</td><td>0.19</td><td>0.98</td><td>5.41</td><td>50.98</td><td>66.44</td><td>82.50</td></tr><tr><td>UPAL (Ours)</td><td>0.30</td><td>1.09</td><td>4.45</td><td>64.80</td><td>77.57</td><td>88.87</td></tr></table>

3D Reconstruction. We further evaluate 3D point-cloud reconstruction from multi-view posed images. We evaluate the accuracy and completeness of the point triangulation [59] on ETH3D [60] following existing practices [12,37]. All detectors are tested with mutual nearest neighbor matching with a maximum number of keypoints of 2048. Tab. 6 shows that we achieve strong performance both in terms of accuracy and completeness, consistently exceeding SuperPoint [10] and ALIKED [80], while falling behind DISK [66], which performs particularly well under this setup. This further proves the applicability of UPAL for accurate 3D reconstruction with a compact, unified architecture.

## 4.4 Eficiency

We benchmark eficiency against combinations of state-of-the-art models on an NVIDIA RTX 2080 Ti GPU (11 GB) as well as on an Intel Core i9-13900K CPU, on 500 images from the Oxford-Paris dataset [50] with a batch size of one. Independent point and line extractors are run sequentially. Images are resized to 800 × 800. As shown in Tab. 7, UPAL achieves a mean GPU inference time of 70 ms, providing roughly a 4× speedup over the SOTA ALIKED [80] + DeepLSD [44]. This eficiency stems from joint inference, a smaller network architecture, and an optimized line post-processing pipeline. UPAL is also competitive with lightweight approaches such as ALIKED [80] + M-LSD [23], while obtaining better performance (e.g. in Tab. 4). Furthermore, our number of parameters is lower by an order of magnitude compared to previous joint extractors (Wireframe [17] and PLNet [70]), and GPU latency is 37% lower.

## 4.5 Ablation study

We validate our design choices for the line detection implementation on the HPatches dataset [4]. In Tab. 8 we compare our final model to various versions of our line post-processing. First, we show that DeepLSD without an angle field prediction (2) obtains a higher repeatability and lower localization error than the vanilla DeepLSD (1). Next, we show the benefits of our enhancements over the vanilla LSD (3). For each modification, computing the gradient angle in parallel (4), sampling seed points with a stride of 2 (5), selecting only a subset of the seed points (6), and ofloading the image preprocessing to the GPU (7), we observe consistent improvements in latency, with no degradation in performance.

Table 7: Runtime evaluation. Parameter counts are reported, and latencies are averaged over 500 sequential detections on an NVIDIA RTX 2080 Ti GPU and Intel Core i9-13900K CPU.
<table><tr><td rowspan="2"></td><td rowspan="2">Nb params ↓</td><td colspan="2">Latency (ms) ↓</td></tr><tr><td>GPU</td><td>CPU</td></tr><tr><td>SuperPoint [10] + DeepLSD [44]</td><td>9.8M</td><td>293</td><td>2259</td></tr><tr><td>ALIKED [80] + DeepLSD [44]</td><td>9.2M</td><td>286</td><td>2239</td></tr><tr><td>ALIKED [80] + M-LSD [23]</td><td>1.3M</td><td>54</td><td>525</td></tr><tr><td>ALIKED [80] + TP-LSD [28]</td><td>25M</td><td>56</td><td>1582</td></tr><tr><td>DaD [13] + DeDoDe v2 [15] + ScaleLSD [30]</td><td>120M</td><td>147</td><td>&gt;200K</td></tr><tr><td>Wireframe [17]</td><td>5.8M</td><td>266</td><td>2493</td></tr><tr><td>PLNet [70]</td><td>7.5M</td><td>112</td><td>6233</td></tr><tr><td>UPAL (Ours)</td><td>0.78M</td><td>70</td><td>976</td></tr></table>

Table 8: Ablation study on the HPatches dataset [4]. We compare the localization error, homography estimation, repeatability, and latency for variants of our approach.
<table><tr><td></td><td></td><td></td><td>Loc Error ↓ H estimation ↑</td><td>Repeatability ↑</td><td>Latency (ms) ↓</td></tr><tr><td>(1)</td><td>DeepLSD [44]</td><td>1.432</td><td>90.6</td><td>27.2</td><td>233</td></tr><tr><td>(2)</td><td>DeepLSD no AF</td><td>1.341</td><td>90.6</td><td>28.2</td><td>218</td></tr><tr><td>(3)</td><td>LSD [20]</td><td>1.414</td><td>91.1</td><td>26.3</td><td>36</td></tr><tr><td>(4)</td><td>(3) + parall. ang. comp.</td><td>1.414</td><td>91.1</td><td>26.3</td><td>27</td></tr><tr><td>(5)</td><td>(4) + stride 2</td><td>1.455</td><td>90.6</td><td>26.3</td><td>19</td></tr><tr><td>(6)</td><td>(5) + subset</td><td>1.476</td><td>90.6</td><td>26.3</td><td>17</td></tr><tr><td>(7)</td><td>(6) + GPU</td><td>1.343</td><td>90.7</td><td>28.9</td><td>11</td></tr></table>

## 5 Conclusion

In this work, we introduced UPAL, a unified architecture capable of jointly extracting keypoints, line segments, and feature descriptors, while maintaining high eficiency and preserving the performance of each individual component. Our method achieves a 4× speedup compared to combining existing standalone components, while reducing the memory footprint by at least one order of magnitude. Looking ahead, we plan to integrate UPAL with a lightweight, jointly learned matcher to enable end-to-end point-line image matching. We believe that our eficient, joint feature extractor marks an important step toward broadening access to advanced geometric perception, ultimately facilitating the deployment of point-line applications on embedded platforms and in low-compute environments.

Acknowledgments: We would like to thank Viktor Larsson for valuable discussions regarding the LSD line detector. During our eforts to develop a faster variant of LSD, he suggested precomputing the distance field on the GPU. This insight ultimately helped shape the accelerated version of LSD presented in this work. We also thank Olivia Buset for her help during this project.

## References

1. Abdelbaki, H., Frohlich, R., Vilajosana, V., Katona, Z.: L2d2: Learnable line detector and descriptor. In: 3DV (2021)

2. Agostinho, S., Gomes, J., Del Bue, A.: Cvxpnpl: A unified convex solution to the absolute pose estimation problem from point and line correspondences (2019)

3. Akinlar, C., Topal, C.: Edlines: A real-time line segment detector with a false detection control. In: ICIP (2011)

4. Balntas, V., Lenc, K., Vedaldi, A., Mikolajczyk, K.: Hpatches: A benchmark and evaluation of handcrafted and learned local descriptors. In: CVPR (2017)

5. Bay, H., Ess, A., Tuytelaars, T., Van Gool, L.: Speeded-up robust features (surf). CVIU (2008)

6. Chen, G., Fu, T., Chen, H., Teng, W., Xiao, H., Zhao, Y.: Rdd: Robust feature detector and descriptor using deformable transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR). pp. 6394–6403 (June 2025)

7. Cuturi, M.: Sinkhorn distances: Lightspeed computation of optimal transport. NeurIPS (2013)

8. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In: CVPR (2017)

9. Dai, X., Yuan, X., Gong, H., Ma, Y.: Fully convolutional line parsing. In: IPIN (2019)

10. DeTone, D., Malisiewicz, T., Rabinovich, A.: Superpoint: Self-supervised interest point detection and description. CVPRW (2018)

11. Dusmanu, M., Rocco, I., Pajdla, T., Pollefeys, M., Sivic, J., Torii, A., Sattler, T.: D2-Net: A Trainable CNN for Joint Detection and Description of Local Features. In: CVPR (2019)

12. Dusmanu, M., Schönberger, J.L., Pollefeys, M.: Multi-view optimization of local feature geometry. In: ECCV (2020)

13. Edstedt, J., Bökman, G., Wadenbäck, M., Felsberg, M.: DaD: Distilled Reinforcement Learning for Diverse Keypoint Detection. arXiv (2025)

14. Edstedt, J., Bökman, G., Wadenbäck, M., Felsberg, M.: DeDoDe: Detect, Don’t Describe - Describe, Don’t Detect for Local Feature Matching. In: 3DV (2024)

15. Edstedt, J., Bökman, G., Zhao, Z.: DeDoDe v2: Analyzing and Improving the DeDoDe Keypoint Detector . In: CVPRW (2024)

16. Elder, J.H., Almazan, E.J., Qian, Y., Tal, R.: Mcmlsd: A probabilistic algorithm and evaluation framework for line segment detection. arXiv (2020)

17. Ferre, I., Baumela, L., Suárez, I.: Learning to detect and describe a wireframe. In: Proceedings of the 12th Iberian Conference on Pattern Recognition and Image Analysis (IbPRIA) (2025)

18. Fu, Q., Wang, J., Yu, H., Ali, I., Guo, F., He, Y., Zhang, H.: Pl-vins: Real-time monocular visual-inertial slam with point and line features (2020)

19. Gao, S., Wan, J., Ping, Y., Zhang, X., Dong, S., Yang, Y., Ning, H., Li, J., Guo, Y.: Pose refinement with joint optimization of visual points and lines. In: IROS (2022)

20. von Gioi, R.G., Jakubowicz, J., Morel, J.M., Randall, G.: Lsd: A fast line segment detector with a false detection control. IEEE TPAMI (2010)

21. Gleize, P., Wang, W., Feiszli, M.: Silk – simple learned keypoints. In: ICCV (2023)

22. Gomez-Ojeda, R., Moreno, F.A., Zuniga-Noël, D., Scaramuzza, D., Gonzalez Jimenez, J.: Pl-slam: A stereo slam system through the combination of points and line segments. IEEE Transactions on Robotics 35(3), 734–746 (2019)

23. Gu, G., Ko, B., Go, S., Lee, S.H., Lee, J., Shin, M.: Towards real-time and lightweight line segment detection. In: AAAI (2022)

24. Guo, Z., Lu, H., Yu, Q., Guo, R., Xiao, J., Yu, H.: HDPL: a hybrid descriptor for points and lines based on graph neural networks. Industrial Robot: the International Journal of Robotics Research and Application (2021)

25. Harris, C., Stephens, M.: A combined corner and edge detector. In: Alvey Vision Conference (1988)

26. Hrubý, P., Liu, S., Pautrat, R., Pollefeys, M., Baráth, D.: Handbook on leveraging lines for two-view relative pose estimation. In: 3DV (2023)

27. Huang, K., Wang, Y., Zhou, Z., Ding, T., Gao, S., Ma, Y.: Learning to parse wireframes in images of man-made environments. In: CVPR (2018)

28. Huang, S., Qin, F., Xiong, P., Ding, N., He, Y., Liu, X.: Tp-lsd: Tri-points based line segment detector. In: Computer Vision – ECCV 2020 (2020)

29. Janampa, S., Pattichis, M.: Linea: Fast and accurate line detection using scalable transformers (2025), https://arxiv.org/abs/2505.16264

30. Ke, Z., Tan, B., Zheng, X., Shen, Y., Wu, T., Xue, N.: Scalelsd: Scalable deep line segment detection streamlined. In: CVPR (2025)

31. Künzel, J., Hilsmann, A., Eisert, P.: Ripe: Reinforcement learning on unlabeled image pairs for robust keypoint extraction (2025), https://arxiv.org/abs/2507. 04839

32. Larsson, V.: Poselib-minimal solvers for camera pose estimation (2020), available at https://github.com/vlarsson/poselib

33. Lebeda, K., Matas, J., Chum, O.: Fixing the Locally Optimized RANSAC. In: BMVC (2012)

34. Li, W., Shao, Y., Wang, Y., Wang, S., Bai, X., Li, D.: Idls: Inverse depth line based visual-inertial slam (2023)

35. Li, Z., Snavely, N.: Megadepth: Learning single-view depth prediction from internet photos. CVPR (2018)

36. Lim, H., Jeon, J., Myung, H.: Uv-slam: Unconstrained line-based slam using vanishing points for structural mapping. IEEE Robotics and Automation Letters 7(2), 1518–1525 (2022)

37. Lindenberger, P., Sarlin, P.E., Larsson, V., Pollefeys, M.: Pixel-perfect structurefrom-motion with featuremetric refinement. In: ICCV (2021)

38. Liu, S., Gao, Y., Zhang, T., Pautrat, R., Schönberger, J.L., Larsson, V., Pollefeys, M.: Robust incremental structure-from-motion with hybrid features. In: ECCV (2024)

39. Liu, S., Yu, Y., Pautrat, R., Pollefeys, M., Larsson, V.: 3d line mapping revisited. In: CVPR (2023)

40. Lowe, D.G.: Distinctive image features from scale-invariant keypoints. IJCV 60(2), 91–110 (2004)

41. Ma, Q., Jiang, G., Wu, J., Cai, C., Lai, D., Bai, Z., Chen, H.: WGLSM: An endto-end line matching network based on graph convolution. Neurocomputing 453, 195–208 (2021)

42. Meng, Q., Zhang, J., Hu, Q., He, X., Yu, J.: Lgnn: A context-aware line segment detector. arXiv abs/2008.05892 (2020)

43. Nistér, D.: An eficient solution to the five-point relative pose problem. In: CVPR (2003)

44. Pautrat, R., Larsson, V., Yu, Y., Pollefeys, M.: Deeplsd: Line segment detection and refinement with deep image gradients. In: CVPR (2023)

45. Pautrat\*, R., Suárez\*, I., Yu, Y., Pollefeys, M., Larsson, V.: GlueStick: Robust image matching by sticking points and lines together. In: ICCV (2023)

46. Pautrat, R., Larsson, V., Oswald, M.R., Pollefeys, M.: Online invariance selection for local feature descriptors. In: ECCV (2020)

47. Pautrat, R., Lin, J.T., Larsson, V., Oswald, M.R., Pollefeys, M.: SOLD2: Selfsupervised occlusion-aware line description and detection. In: CVPR (2021)

48. Potje, G., Cadar, F., Araujo, A., Martins, R., Nascimento, E.R.: Xfeat: Accelerated features for lightweight image matching. In: CVPR (2024)

49. Pumarola, A., Vakhitov, A., Agudo, A., Sanfeliu, A., Moreno-Noguer, F.: Pl-slam: Real-time monocular visual slam with points and lines. In: ICRA (2017)

50. Radenović, F., Iscen, A., Tolias, G., Avrithis, Y., Chum, O.: Revisiting oxford and paris: Large-scale image retrieval benchmarking. In: CVPR (2018)

51. Ramalingam, S., Bouaziz, S., Sturm, P.: Pose estimation using both points and lines for geo-localization. In: ICRA (2011)

52. Ren, G., Cao, Z., Liu, X., Tan, M., Yu, J.: Plj-slam: Monocular visual slam with points, lines, and junctions of coplanar lines. IEEE Sensors Journal 22(15) (2022)

53. Revaud, J., Weinzaepfel, P., de Souza, C.R., Humenberger, M.: R2D2: repeatable and reliable detector and descriptor. In: NeurIPS (2019)

54. Rublee, E., Rabaud, V., Konolige, K., Bradski, G.: Orb: An eficient alternative to sift or surf. In: ICCV (2011)

55. Salaün, Y., Marlet, R., Monasse, P.: Multiscale line segment detector for robust and accurate SfM. In: ICPR (2016)

56. Sarlin, P.E.: Visual localization made easy with hloc. https://github.com/cvg/ Hierarchical-Localization

57. Sarlin, P.E., Cadena, C., Siegwart, R., Dymczyk, M.: From coarse to fine: Robust hierarchical localization at large scale. In: CVPR (2019)

58. Sarlin, P.E., DeTone, D., Malisiewicz, T., Rabinovich, A.: SuperGlue: Learning feature matching with graph neural networks. In: CVPR (2020)

59. Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: CVPR (2016)

60. Schops, T., Schönberger, J.L., Galliani, S., Sattler, T., Schindler, K., Pollefeys, M., Geiger, A.: A multi-view stereo benchmark with high-resolution images and multi-camera videos. In: CVPR (2017)

61. Sha, Z., Zhong, B., Chen, X., Wang, Z.: Dealsd: A deep edge assisted line segment detector. Expert Systems with Applications 297 (2025)

62. Shotton, J., Glocker, B., Zach, C., Izadi, S., Criminisi, A., Fitzgibbon, A.: Scene coordinate regression forests for camera relocalization in RGB-D images. In: CVPR (2013)

63. Sinkhorn, R., Knopp, P.: Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics 21(2), 343–348 (1967)

64. Suárez, I., Buenaposada, J.M., Baumela, L.: Elsed: Enhanced line segment drawing. PR 127 (2022)

65. Teplyakov, L., Erlygin, L., Shvets, E.: Lsdnet: Trainable modification of lsd algorithm for real-time line segment detection. IEEE Access 10 (2022)

66. Tyszkiewicz, M., Fua, P., Trulls, E.: Disk: Learning local features with policy gradient. In: NeurIPS (2020)

67. Ubingazhibov, A., Pautrat, R., Suárez, I., Liu, S., Pollefeys, M., Larsson, V.: Lightgluestick: a fast and robust glue for joint point-line matching. In: CVPRW (2025)

68. Vakhitov, A., Funke, J., Moreno-Noguer, F.: Accurate and linear time pose estimation from points and lines. In: ECCV (2016)

69. Wang, C., Xu, R., Zhang, Y., Xu, S., Meng, W., Fan, B., Zhang, X.: Mtldesc: Looking wider to describe better (2022), https://arxiv.org/abs/2203.07003

70. Xu, K., Hao, Y., Yuan, S., Wang, C., Xie, L.: AirSLAM: An eficient and illuminationrobust point-line visual slam system. arXiv preprint arXiv:2408.03520 (2024), https://arxiv.org/abs/2408.03520

71. Xu, L., Yin, H., Shi, T., Jiang, D., Huang, B.: Eplf-vins: Real-time monocular visualinertial slam with eficient point-line flow features. IEEE Robotics and Automation Letters 8(2) (2023)

72. Xu, Y., Xu, W., Cheung, D., Tu, Z.: Line segment detection using transformers without edges. arXiv abs/2101.01909 (2021)

73. Xue, N., Bai, S., Wang, F., Xia, G.S., Wu, T., Zhang, L.: Learning attraction field representation for robust line segment detection. In: CVPR (2019)

74. Xue, N., Wu, T., Bai, S., Wang, F.D., Xia, G.S., Zhang, L., Torr, P.H.: Holisticallyattracted wireframe parsing: From supervised to self-supervised learning. arXiv (2022)

75. Xue, N., Wu, T., Bai, S., Wang, F., Xia, G.S., Zhang, L., Torr, P.H.: Holisticallyattracted wireframe parsing. In: CVPR (2020)

76. Yoon, S., Kim, A.: Line as a visual sentence: Context-aware line descriptor for visual localization. IEEE Robotics and Automation Letters (RAL) 6(4), 8726–8733 (2021)

77. Zhang, H., Luo, Y., Qin, F., He, Y., Liu, X.: Elsd: Eficient line segment detector and descriptor. In: ICCV (2021)

78. Zhang, Y., Wei, D., Li, Y.: AG3line: Active grouping and geometry-gradient combined validation for fast line segment extraction. PR 113 (2021)

79. Zhang, Z., Li, Z., Bi, N., Zheng, J., Wang, J., Huang, K., Luo, W., Xu, Y., Gao, S.: Ppgnet: Learning point-pair graph for line segment detection. In: CVPR (2019)

80. Zhao, X., Wu, X., Chen, W., Chen, P.C., Xu, Q., Li, Z.: Aliked: A lighter keypoint and descriptor extraction network via deformable transformation. TIM (2023)

81. Zhao, X., Wu, X., Miao, J., Chen, W., Chen, P.C.Y., Li, Z.: Alike: Accurate and lightweight keypoint detection and descriptor extraction. IEEE Transactions on Multimedia (2022)

82. Zhou, L., Wang, S., Kaess, M.: A fast and accurate solution for pose estimation from 3d correspondences. In: ICRA (2020)

83. Zhou, L., Ye, J., Kaess, M.: A stable algebraic camera pose estimation for minimal configurations of 2d/3d point and line correspondences. In: ACCV (2018)

84. Zhou, Y., Qi, H., Ma, Y.: End-to-end wireframe parsing. ECCV (2019)

85. Zhu, X., Hu, H., Lin, S., Dai, J.: Deformable convnets v2: More deformable, better results. In: CVPR (2019)

86. Zuo, X., Xie, X., Liu, Y., Huang, G.: Robust visual slam with point and line features. In: IROS (2017)

## Supplementary Material

We provide additional analyses of our proposed UPAL feature extractor in the following. Sec. A provides additional information about the distillation process. Sec. B justifies the joint point and line training. Sec. C provides another perspective on the inference cost, by comparing baselines on a low-end GPU, simulating an embedded GPU. Sec. D presents the evaluation of point-line visual localization on the full 7Scenes dataset. Sec. E displays additional visualizations of the predicted lines. Finally, Sec. F shows qualitative examples of point and line matching.

## A Score map distillation

We use the raw score map predicted by DaD [13] when fusing its keypoint heatmap with the SuperPoint [10] output. This score map can be interpreted as a prior probability indicating whether a pixel corresponds to a keypoint, similarly to the formulation used in SuperPoint. Therefore, taking the maximum response between the two maps is a meaningful fusion strategy.

The choice of λ = 200 for balancing positive and negative keypoints follows the setting introduced in MTLDesc [69]. For images of size $8 0 0 \times 8 0 0$ , this corresponds to approximately 3,200 keypoints.

## B Joint point and line training

Beyond computational eficiency, joint point-and-line training also improves the quality of the learned point features. As shown in Tab. 9, the jointly trained model consistently outperforms its point-only counterpart on MegaDepth relative pose estimation, demonstrating that line supervision provides complementary geometric information during training. Lines encode structural cues such as edges, boundaries, and intersections, which help the network localize keypoints more accurately and learn features that are more geometrically consistent across views. This additional supervision acts as a form of regularization, encouraging the shared backbone to capture both local appearance and higher-level scene geometry.

Table 9: Impact of joint point and line training on MegaDepth [35] relative pose estimation.
<table><tr><td>Method</td><td> $\mathbf { A U C @ 5 ^ { \circ } } \uparrow$ </td><td> $\operatorname { A U C @ 1 0 ^ { \circ } } \uparrow$ </td><td> $\operatorname { A U C @ 2 0 } ^ { \circ } \uparrow$ </td></tr><tr><td>Ours (points only)</td><td>55.2</td><td>67.6</td><td>76.4</td></tr><tr><td>Ours (joint training)</td><td>58.2</td><td>70.8</td><td>79.4</td></tr><tr><td>Improvement</td><td>+3.0</td><td>+3.2</td><td>+3.0</td></tr></table>

## C Eficiency on constrained resources

Table 10: Runtime evaluation. Parameter counts and latencies (ms) are averaged over 500 sequential detections on an NVIDIA GeForce GTX 1050 Ti with 4 GB of VRAM. UPAL runs in 186 ms, about 18× faster than DaD + ScaleLSD.
<table><tr><td></td><td>Nb params ↓ Latency ↓</td><td></td></tr><tr><td>SuperPoint [10] + DeepLSD [44]</td><td> $9 . 8 \times 1 0 ^ { 6 }$ </td><td>707</td></tr><tr><td>ALIKED [80] + DeepLSD [44]</td><td> $9 . 2 \times { { 1 0 } ^ { 6 } }$ </td><td>716</td></tr><tr><td>DaD [13] + DeDoDe v2 [15] + ScaleLSD [30]</td><td> $1 . 2 \times 1 0 ^ { 8 }$ </td><td>3474</td></tr><tr><td>UPAL (Ours)</td><td> $7 . 8 \times 1 0 ^ { 5 }$ </td><td>186</td></tr></table>

To assess eficiency in low-resource environments, representative of embedded applications or consumer devices, we additionally benchmark eficiency using combinations of state-of-the-art models with large backbones on an NVIDIA GeForce GTX 1050 Ti GPU with 4 GB of VRAM. Similar to the setup of Table 7 of the main paper, we use 500 images from the Oxford-Paris dataset [50] with a batch size of 1 to measure latency. Independent point and line extractors are executed sequentially, and all images are resized to 800 × 800.

As shown in Table 10, UPAL achieves a mean inference time of 186 ms, yielding roughly a 4× speedup over ALIKED + DeepLSD. More importantly, we observe a significant speed diference between UPAL and DaD + ScaleLSD with up to 18× speedup. The reason is that DaD + ScaleLSD, and especially ScaleLSD, rely on modern powerful GPUs to be eficient and perform poorly on limited hardware due to their large model size. In contrast, UPAL performs well in both settings due to its lightweight configuration.

## D Visual localization

Tab. 5 of the main paper reports the results for visual localization on the Stairs scene of the 7Scenes [62] dataset for both points-only and points+lines jointly. In Tab. 11, we give the results for all seven scenes of the dataset.

The results demonstrate that our method achieves highly competitive performance across visual localization tasks on the 7Scenes dataset, consistently outperforming or matching strong baselines in both point-only and points+lines configurations. In the point-only category, UPAL secures the best accuracy for five of the seven scenes. While SuperPoint and PLNet are the strongest competing baselines, DISK and especially ALIKED fall behind. In general, translation and rotation errors show less variance. Similar to the accuracy results, UPAL, SuperPoint, and PLNet consistently achieve the best results.

Looking at points+lines performance, we observe that SuperPoint+DeepLSD and PLNet represent the strongest competing baselines and outperform UPAL on

Chess, Heads, and Ofice, with SuperPoint+DeepLSD showing especially strong results on Chess, while PLNet scores best in the Heads and Ofice scenes, where long structural lines are abundant. However, our method demonstrates clear advantages on scenes with complex textures and varying illumination conditions such as Fire, Pumpkin, and RedKitchen, where we outperform these baselines in accuracy while maintaining comparable or better translation/rotation errors. We see the strong points-only performance of UPAL, SuperPoint, and PLNet translating to the points+lines scenario as well. Overall, UPAL performs best on most scenes for points-only and points+lines localization on 7Scenes while being significantly faster and more lightweight than the baselines.

## E Qualitative comparison of line detection results

Similar to Fig. 4, Fig. 5 depicts additional line detection results. All detectors are run with their default settings. The qualitative detection results clearly show that ScaleLSD and Wireframe consistently detect significantly fewer lines in the images. They focus on long structural lines and miss many generic valid lines that could be useful for some applications (e.g. visual localization). LSD, DeepLSD and UPAL show a higher recall of line segments, even in structural regions. The diference between LSD, DeepLSD, and UPAL is less pronounced; nonetheless, we observe that DeepLSD and UPAL can recall more line segments in areas of subtle image gradients (see second row of Fig. 5). We explain this with the strong learned line distance field informing the LSD algorithm in low gradient areas.

Overall, UPAL produces qualitative results superior to Wireframe and ScaleLSD in particular, while showing DeepLSD-level detection quality. Notably, we achieve this while being significantly faster and substantially more lightweight, as shown in Tab. 7. This advantage is even more pronounced on lower-tier hardware as shown in Section C.

## F Point and line matching qualitative examples

We provide qualitative examples of point and line matching with UPAL on image pairs from HPatches [4], with both illumination-varying and viewpoint-varying pairs. Features are extracted after resizing each image so that its longer side is 800 pixels while preserving the aspect ratio, as in our HPatches evaluation (Sec. 4). Keypoints are matched by mutual nearest neighbors over their descriptors, and line segments likewise over their two endpoint descriptors. For better visibility, each pair is shown at its original aspect ratio, and only the top-300 matches are drawn, ranked by detector scores for points and by segment length for lines. Correct matches are drawn in yellow (points) and green (lines), and incorrect ones in red, where a match is deemed correct when its reprojection error (points) or orthogonal line distance (lines) under the ground-truth homography is below 3 pixels. Fig. 6 shows point correspondences and Fig. 7 shows line correspondences, including both favorable and challenging cases.

Notably, the same viewpoint-varying pairs (artwork, sculpture, and stained glass) that are matched reliably by points are almost entirely mismatched as lines. This follows from our design: UPAL has no dedicated line descriptor and matches segments only through the descriptors at their two endpoints [45] (Sec. 4). As these endpoints are LSD segment terminations rather than repeatable keypoints, a perspective change shifts them along the segment and alters their surrounding appearance, breaking the endpoint correspondences; the point branch is unafected, as it matches repeatable keypoints directly. The simple mutualnearest-neighbor matcher used here further limits robustness, and we expect a lightweight learned point-line matcher, which we leave to future work (Sec. 5), to improve these results.

Table11:Visuallocalizationonthefull7Scenesdataset[62].Wereportthemediantranslation/rotationerrors(cm/deg)and poseaccuracy(%)at5cm/5°thresholdforeachscene.Wedenotethebestmethodforeachsceneandcategory(PointsorPoints+Lines) <sub>and</sub> <sub>the</sub> <sub>second-best</sub> m<sup>ethod</sup> <sup>is</sup> <sup>un</sup>
<table><tr><td></td><td colspan="2">Chess</td><td colspan="2">Fire</td><td colspan="2">Heads</td><td colspan="2">Office</td><td colspan="2">Pumpkin</td><td colspan="2">RedKitchen</td><td colspan="2">Stairs</td><td colspan="2">Total</td></tr><tr><td>Method</td><td>T/R↓</td><td>Acc. ↑</td><td>T/R↓</td><td>Acc. ↑</td><td>T/R↓</td><td>Acc. ↑</td><td>T/R ↓</td><td>Acc. ↑</td><td>T/R↓</td><td>Acc. ↑</td><td>T/R↓</td><td>Acc. ↑</td><td>T/R↓</td><td>Acc. ↑</td><td>T/R↓</td><td>Acc. ↑</td></tr><tr><td>SuperPoint [10]</td><td>2.5/0.87</td><td>92.2</td><td>2.3/0.90</td><td>89.5</td><td>1.1/0.80</td><td>95.6</td><td>3.2/0.93</td><td>74.4</td><td>5.2/1.40</td><td>48.2</td><td>4.4/1.43</td><td>56.7</td><td>6.2/1.65</td><td>36.6</td><td>3.56/1.14</td><td>70.5</td></tr><tr><td>DISK [66]</td><td>2.4/0.87</td><td>92.2</td><td>2.4/0.96</td><td>86.4</td><td>1.2/0.86</td><td>89.9</td><td>3.8/1.06</td><td>64.9</td><td>5.4/1.47</td><td>46.0</td><td>4.6/1.49</td><td>55.2</td><td>7.2 /2.16</td><td>28.5</td><td>3.86/1.27</td><td>66.2</td></tr><tr><td>ALIKED [80]</td><td>2.6/0.92</td><td>86.3</td><td>2.7/1.06</td><td>80.3</td><td>1.4/1.00</td><td>77.3</td><td>5.7/1.71</td><td>45.8</td><td>5.3/1.44</td><td>47.1</td><td>5.2/1.65</td><td>48.0</td><td>7.8/1.98</td><td>35.0</td><td>4.39/1.39</td><td>60.0</td></tr><tr><td>Points PLNet [70]</td><td>2.5/0.86</td><td>92.2</td><td>2.3/0.91</td><td>88.6</td><td>1.1/0.79</td><td>95.2</td><td>3.1/0.93</td><td>74.9</td><td>5.2/1.44</td><td>47.5</td><td>4.4/1.42</td><td>56.6</td><td>6.2 /1.68</td><td>38.2</td><td>3.54/1.15</td><td>70.5</td></tr><tr><td>UPAL (Ours)</td><td>2.5/0.88</td><td>93.4</td><td>2.1/0.85</td><td>91.9</td><td>1.0/0.76</td><td>93.9</td><td>3.3/0.96</td><td>72.9</td><td>4.9/1.25</td><td>51.2</td><td>4.3/1.44</td><td>58.0</td><td>5.1/1.39</td><td>49.1</td><td>3.31/1.08</td><td>72.9</td></tr><tr><td>SuperPoint [10]+DeepLSD</td><td>2.5/0.87</td><td>92.5</td><td>2.1/0.85</td><td></td><td></td><td>93.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>[44] ALIKED [80]+DeepLSD [44]</td><td>2.6/0.90</td><td>88.0</td><td>2.4/0.95</td><td>93.2 85.2</td><td>1.1/0.79 1.4/1.00</td><td>74.5</td><td>3.1/0.91 5.3/1.62</td><td>76.0 47.8</td><td>4.9/1.31 5.0/1.38</td><td>51.4 49.8</td><td>4.3/1.43 5.1/1.61</td><td>57.5 49.1</td><td>5.0/1.33 5.1/1.48</td><td>49.6 49.6</td><td>3.29/1.07 3.84/1.28</td><td>73.4</td></tr><tr><td>ALIKED [80] ]+TP-LSD [28]</td><td>2.6/0.89</td><td>88.2</td><td>2.4/0.96</td><td>84.3</td><td>1.4/0.98</td><td>74.3</td><td>5.3/1.64</td><td>48.1</td><td>5.2/1.38</td><td>48.5</td><td>5.1/1.61</td><td>48.8</td><td>5.2/1.48</td><td>48.8</td><td>3.89/1.28</td><td>63.4 63.0</td></tr><tr><td>ALIKED [80]+M-LSD [23]</td><td>2.6/0.89</td><td>88.2</td><td>2.5/0.96</td><td>84.2</td><td>1.4/0.99</td><td>73.9</td><td>5.3/1.62</td><td>47.6</td><td>5.2/1.37</td><td>48.5</td><td>5.1/1.6</td><td>48.9</td><td>5.2 /1.52</td><td>48.5</td><td>3.90/1.28</td><td>62.8</td></tr><tr><td>Wireframe [17]</td><td>2.5/0.86</td><td>91.9</td><td>2.2/0.87</td><td>90.4</td><td>1.1/0.79</td><td>89.4</td><td>3.5/0.98</td><td>70.8</td><td>5.1/1.40</td><td>49.3</td><td>4.5/1.49</td><td>55.6</td><td>4.6/1.30</td><td>53.8</td><td>3.36/1.10</td><td>71.6</td></tr><tr><td>PLNet [70]</td><td>2.5/0.89</td><td>92.3</td><td>2.1/0.85</td><td>92.4</td><td>1.1/0.80</td><td>94.0</td><td>3.1/0.91</td><td>76.6</td><td>4.9/1.36</td><td>50.7</td><td>4.3/ /1.42</td><td>57.8</td><td>5.0/1.36</td><td>50.1</td><td>3.29/1.08</td><td>73.4</td></tr><tr><td>Poinnnes UPAL (Ours)</td><td>2.5/0.88</td><td>92.2</td><td>2.0/0.79</td><td>94.9</td><td>1.0/0.77</td><td>92.8</td><td>3.2/0.93</td><td>75.8</td><td>4.5/1.17</td><td>54.8</td><td>4.2/1.41</td><td>59.6</td><td>4.4/1.23</td><td>54.6</td><td>3.11/1.03</td><td>75.0</td></tr></table>

LSD [20]  
ScaleLSD [30]  
Wireframe [17]  
DeepLSD [44]  
UPAL (Ours)  
![](images/98178b34e52968b4ce1bc6586bc902ff56029c225d94ffc1c4fdadc2b17eb30d.jpg)  
Fig. 5: Line detection examples. Unlike wireframe-based methods [17,30] that focus solely on structural lines, UPAL produces more general and repeatable line detections that better support downstream tasks, while remaining significantly more lightweight. The last four images are challenging examples for UPAL and other detectors.

![](images/5ca013f776a1369423ab7914291b17aca3462a2857441cc0e72e012de1c1a791.jpg)  
artwork, viewpoint

![](images/cc6f812a7a3fd0c7ac94a887af049bd16a45e519e453a0e737f5825a44808f7f.jpg)  
sculpture, viewpoint

![](images/9f7599126a900530c29e092a98dd7c22bf37312b26bdb6e9077e1ac3b417f4f9.jpg)  
resort, illumination

![](images/30701a12cd03b8cf0f07e78b58c95a9ad5d06211b34df1e48b06f95c281171f7.jpg)  
tools, illumination

![](images/699c3866041306642d19ad4a7becf9b1e2b4c5f1f0fa861f54b8ebb5171d5f4d.jpg)  
stained glass, viewpoint

![](images/c9c6b7fa08476b8c8bc3f022223cf53e715223d824a60506fba200a49b9de5ed.jpg)  
brooklyn bridge, illumination

![](images/66aee1782293a2cb84f3ca362ba1e9eb28dbcfc5f14fa069560ef12c32d0675e.jpg)  
tower bridge, illumination

![](images/96c497727911774be1ad5f6c5280bca62f5d5014e06141a13a568b76395eba76.jpg)  
bark, viewpoint

![](images/4dc6d33ea96c9918fbb4b42ad5971ec1bc950b90cf86f652e0b7a80fe5ce0745.jpg)  
grafiti, viewpoint  
Fig. 6: Point matching on HPatches (yellow = correct, red = wrong with respect to the ground-truth homography; only the most salient matches are shown). UPAL generally matches keypoints reliably under both viewpoint and illumination changes. On repetitive structures the salient corners still match, with errors confined to weaker detections, whereas specular surfaces and especially self-similar texture under large viewpoint changes are harder and yield many incorrect matches.

![](images/c68bb926e7d0e2a74f244e5d922cc867b6015111cbbabd53762f271785c90376.jpg)  
resort, illumination

![](images/c862f607288a9627d9eba2209c7d516c8332755afd32b70ddf11a453eed2cc6d.jpg)  
product, illumination

![](images/2d3eb604d976c2563314f9cbd95395ca55447bbc4a037b6023e23679a6c430b9.jpg)  
stairs, illumination

![](images/6f04d594cfa82763f2a25ba0b2456ae3a3439ea80f0b715a04b9ce9c3628651e.jpg)  
books, illumination

![](images/e889e744f18f3904e0b3a935d37716d05bde60a00fb4a9986a3b97a000a69cc7.jpg)  
village, illumination

![](images/4900e5693190dbefa5b3c6bcac03551bcb01f06ccbd49a3286e884c521b81e7f.jpg)  
artwork, viewpoint

![](images/64d3245272c3534b7a7d71151f462d579b0ad7c87f735b6e912d130ddb7aacdd.jpg)

![](images/92a39dff150f58043e5ff0a10f9e1b17e783741e452ba2564167323c99b2f991.jpg)  
stained glass, viewpoint  
Fig. 7: Line matching on HPatches (green = correct, red = wrong with respect to the ground-truth homography; only the longest matched segments are shown). UPAL matches lines well under illumination changes on structured man-made scenes, but its line branch is not viewpoint-robust: the same viewpoint pairs that are matched reliably by points are almost entirely mismatched as lines.