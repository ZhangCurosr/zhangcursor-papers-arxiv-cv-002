# ILLRAW: Illumination-Aware RAW Object Detection

Xinzhe Wang <sup>1\*</sup>, Xiao Liang <sup>1\*</sup>, Xiaoli Zhang <sup>3</sup>, and Kang Ma <sup>1,2†</sup>

<sup>1</sup> Beijing Institute of Technology

State Key Laboratory of CNS/ATM, Beijing Institute of Technology 3 Independent Researcher

<sup>1</sup>xinzwang123@gmail.com, <sup>1</sup>{liangxiao,makang}@bit.edu.cn <sup>3</sup>z.xiaolilili@gmail.com

Abstract. Existing RAW-based detectors typically introduce learnable ISP modules to convert RAW inputs into sRGB-like formats to fit ofthe-shelf detectors. However, this RAW-to-sRGB paradigm decouples representation learning from the downstream detection task. Crucially, it forces the network to inherit human-centric visual priors while discarding intrinsic RAW advantages, such as linear sensor responses and high-bit-depth, minimally processed scene cues. In this paper, we propose ILLRAW, an ILLumination-aware end-to-end RAW object detection framework. Instead of reconstructing visually plausible sRGB images, ILLRAW directly learns detection-oriented RAW features through a lightweight feature adapter inserted at the input stage of existing detectors. ILLRAW estimates an illumination mask and introduces channelsensitive embeddings to model Bayer-channel responses. These features are fused by an illumination attention fusion module. Experiments show clear improvements on the diverse AODRaw benchmark and competitive performance on LOD and PascalRAW, with particularly strong results under dark exposure.

Keywords: RAW object detection · Illumination-aware Feature Adaptation · end-to-end RAW detection

## 1 Introduction

Object detection is a crucial and fundamental task in computer vision and has achieved significant progress [30, 40–42, 58]. However, most existing object detection methods are developed on sRGB images, whose robustness under challenging imaging conditions is often overlooked [27, 53]. RAW images are the direct outputs of camera sensors [3], and their pixel values are linearly related to incident light intensity and preserve high-bit-depth radiometric information together with unprocessed sensor noise [7, 60]. Such sensor-level light measurements provide richer illumination cues than sRGB images, especially in dark regions, high-dynamic-range scenes, and areas close to saturation. By directly preserving sensor measurements and richer radiometric cues, RAW images provide a promising input space for robust object detection [7, 21, 27, 47, 53] under challenging real-world conditions, such as high dynamic range scenes [7,53], lowlight environments [21, 47], and adverse weather [27].

In modern camera systems, RAW images are processed by the ISP and converted into sRGB images for human viewing. A typical ISP pipeline includes denoising, demosaicking, white balancing, gamma correction, and compression [14, 60]. Although these operations improve visual quality, they inevitably reshape the original light distribution and may discard task-relevant details. For example, denoising suppresses sensor noise but can also remove fine structures [1], particularly in dark areas and near overexposed regions.

To improve detection on RAW images, previous methods often couple object detectors with imaging systems by tuning the hyperparameters of the ISP or the camera pipeline [33, 51, 54]. Although these methods can be regarded as RAW-aware detection approaches, they still heavily rely on ISP processing and do not achieve fully end-to-end RAW object detection. A straightforward alternative is to directly train existing sRGB-based detectors on RAW images. However, this usually leads to degraded performance [7, 27, 53], mainly for two reasons. First, conventional detectors are designed for sRGB images and are not tailored to the unique characteristics of RAW data. Second, most detectors benefit from ImageNet-pretrained weights [10], while transferring sRGB-pretrained models to RAW inputs introduces a clear domain gap. To mitigate these issues, recent methods commonly introduce learnable ISPs that convert RAW data into sRGB-like images by mimicking human visual priors, such as denoising, gamma correction, and white balance [7, 36, 53]. Although efective, this RAW-to-sRGB paradigm optimizes RAW representation learning toward image reconstruction rather than the detection objective. As a result, it may introduce information bottlenecks and inherit human-centric priors that are not necessarily optimal for machine vision models.

Instead of reconstructing visually plausible sRGB images, we argue that RAW images should be directly adapted to detection-oriented feature learning. From the perspective of detector architectures, most modern detectors consist of a backbone network for feature extraction [20, 31], followed by necks, detection heads, or Transformer encoder-decoder modules for object localization and classification [17, 58]. These downstream components are largely designed for high-level semantic reasoning and should not be heavily modified merely to accommodate RAW inputs. Notably, within the backbone, the first convolutional layer directly interacts with image measurements. This observation motivates us to design a lightweight RAW feature adapter at the input stage, which addresses the unique characteristics of RAW data while preserving the general architecture of existing detectors.

We further analyze the inherent illumination characteristics of RAW images. As shown in Figure 1(a), the intensity distribution of RAW images is often highly concentrated within a narrow range, especially under low-light conditions.

![](images/1d6125dd7d68fc813051040ad04cf52f2cf0de0123c55956954a4dc1ecec585d.jpg)

![](images/eb0bfe43596281753bfaf8039c7fe9f0756c08f6558425c63a32d54428915a2e.jpg)  
(a) Histogram of pixel values

<table><tr><td>sRGB</td><td>RAW</td></tr><tr><td>Red 1.001</td><td>0.788</td></tr><tr><td>Green</td><td>1.032 1.257</td></tr><tr><td>Blue 0.967</td><td>0.955</td></tr></table>

(b) Normalized channel intensity

![](images/a9eef08c33cfd3831dd6541f19485c9b0a7a9c0afb5dff8a4574a32ee700420d.jpg)  
sRGB

![](images/fb1733dfc3bb4a7a519c88b90439f2a3efd6d714b970973f270e1d6026bf8f23.jpg)  
RAW-Red

![](images/0177801b0acd593f47420178891b073ab8e0632a0409de3e4db8908baf2b56c7.jpg)  
RAW-Green

![](images/bd2758e2e4439fa082677807431edaa10a65e945522f2901c029ba77332cb889.jpg)  
RAW-Blue  
(c) Visualization of diferent channels in RAW images  
Fig. 1: Comparison of sRGB and RAW images. (a) RAW pixel values are concentrated within a small range. (b) RAW images exhibit significant channel diferences. (c) In the example scene, the green channel is brighter, while the red channel is darker.

In contrast, sRGB images benefit from ISP operations such as gamma correction and tone mapping, which redistribute the dynamic range and produce a more balanced brightness distribution, thereby improving detection performance [26]. However, the concentrated RAW intensity distribution does not indicate the absence of useful information. Instead, RAW pixels preserve a linear response to incident light intensity, making their brightness values closely related to the physical illumination of the scene. Therefore, the key challenge is how to effectively perceive and exploit spatial brightness variations in RAW images. To this end, we introduce an illumination mask to identify regions with diferent brightness levels and use it to guide feature extraction under varying lighting conditions.

In addition to spatial brightness variation, RAW images also exhibit distinct channel-dependent light sensitivity. As shown in Figure 1(b), the green channel typically has higher average intensity values, suggesting that it preserves more information in darker regions. In contrast, the red and blue channels are more responsive in brighter regions that are close to saturation. This phenomenon is further illustrated in Figure 1(c), where the red channel clearly delineates object boundaries in bright areas where the green channel tends to become saturated. Based on these observations, we propose an illumination-aware end-to-end RAW object detection method, termed ILLRAW. Unlike previous approaches that rely on learnable ISPs to bridge RAW and sRGB domains, ILLRAW directly integrates with existing detectors through a lightweight feature adapter. Specifically, we first introduce an illumination mask estimation module to capture spatial light intensity distributions. Then, a channel-sensitive embedding module is designed to extract features with diferent light sensitivities under the guidance of the green and red/blue channels. To efectively integrate these complementary representations, we propose an illumination attention fusion mechanism that adaptively aggregates features according to the estimated illumination mask. Moreover, re-parameterization techniques are employed to minimize inference overhead. The proposed method can efectively process RAW data with low computational cost and good compatibility across diferent detection architectures. Extensive experiments on three RAW object detection datasets demonstrate that ILLRAW achieves state-of-the-art performance, validating its efectiveness and generalizability.

In summary, our main contributions are as follows:

– We propose ILLRAW, an illumination-aware RAW object detection framework that leverages spatial and channel-wise illumination diferences to enhance semantic understanding of RAW images.

– We design an illumination estimation module to extract spatial illumination cues, and a channel-sensitive embedding module to decouple features with diferent light sensitivities under channel-specific guidance.

– We introduce an illumination attention fusion mechanism, which utilizes the estimated illumination mask to adaptively fuse features from regions with varying brightness levels.

## 2 Related Work

In this section, we first review the fundamentals of RAW image processing. Then, we summarize current research on both sRGB-based and RAW-based object detection.

RAW-based Image Processing. Modern camera imaging systems comprise three main components, namely the optical lens, the image sensor, and the image signal processor. When light hits the sensor, it is converted into electrical signals, then into a raw digital signal by an analog-to-digital converter (ADC). This raw signal is called a RAW image [50]. The RAW image is processed by the ISP and converted into an sRGB image suitable for human viewing. Traditional ISP designs are based on how humans perceive color and brightness [24,34,38]. The ISP comprises several processing stages, each serving a specific purpose, although accurately performing these operations remains challenging. Since sRGB images are widely used, many image enhancement methods focus on this format [12, 19, 37, 55, 57]. Recently, there is growing interest in using RAW images for tasks such as denoising [1, 23, 50], high dynamic range (HDR) imaging [25, 43, 60], and general image enhancement [3, 5, 6, 22]. RAW images have clear benefits in handling real-world problems, especially in HDR and low-light situations, but their potential for visual recognition tasks remains underexplored.

sRGB-based Object Detection. Object detection has greatly benefited from large-scale sRGB benchmarks [15, 30, 42]. Early object detection methods primarily relied on hand-crafted features and sliding window techniques [9, 16, 45]. Although these approaches have low computational cost, their limited generalization ability makes them less efective in complex scenarios. With the advent of deep learning, CNN-based detectors have achieved significant progress. These can be broadly divided into two-stage detectors [2,18,28,41] and one-stage detectors [13,17,29,39,40,46]. More recently, transformer-based detectors [31,56,58,59] have introduced global modeling capabilities and reformulated object detection as a set prediction problem, eliminating the need for anchors and non-maximum suppression (NMS). Despite the impressive performance of these methods, their robustness under real-world conditions remains a challenge [21, 27, 32].

RAW-based Object Detection. RAW-based object detection holds promise for addressing challenging real-world scenarios. A straightforward approach is to train detectors directly on RAW images [21, 53]. However, due to the lack of designs tailored specifically for RAW data, these methods often achieve limited performance. Some approaches focus on tuning non-diferentiable ISP components [33, 44], while others learn diferentiable ISP networks [7, 11, 48, 51, 54] to bridge sRGB detectors with RAW inputs. For example, Xu et al. [53] explore image- and pixel-level information to adjust the dynamic range of RAW data, while Cui et al. [7] attempt to adapt sRGB-pretrained models to RAW images. These methods incorporate priors inspired by human visual perception, yet there is no evidence that such priors are optimal for machine vision tasks. Moreover, this decoupling strategy can cause information bottlenecks and may lead to ineficiencies. In contrast, we propose a universal RAW detection method that directly leverages the inherent characteristics of RAW images to improve performance.

## 3 Method

In this section, we first present the motivation behind our design. We then provide a detailed description of the proposed ILLRAW framework, followed by the training and inference strategies. The architecture is illustrated in Figure 2.

## 3.1 Motivation

Existing RAW-based detectors commonly rely on learnable ISPs to convert RAW data into sRGB-like images by mimicking human visual priors, such as denoising, gamma correction, and white balance [7,53]. Although efective, this RAWto-sRGB paradigm decouples RAW representation learning from the detection objective and may introduce information bottlenecks. Instead of reconstructing visually plausible sRGB images, we aim to directly adapt RAW inputs to detection-oriented feature learning. Since most modern detectors consist of a backbone for feature extraction [20,31], followed by task-specific necks, heads, or Transformer encoder-decoder modules for localization and classification [17, 58], we preserve these downstream components and focus on the input stage, where the detector first interacts with image measurements.

Our adapter design is motivated by two RAW-specific properties. First, RAW pixels are linearly related to incident light intensity and exhibit strong spatial brightness variations, motivating an illumination mask estimation module.

![](images/bd5a6809dac91cfadbde81bcfc4678d18b8399d0044a3cf4df51d5c1de43d5c7.jpg)  
Fig. 2: Overview of the proposed ILLRAW. It leverages illumination variations across the channel and spatial dimensions to efectively extract semantic information from RAW images.

Second, RAW channels have diferent light sensitivities: the green channel and red/blue channels provide complementary cues under diferent illumination conditions, motivating a channel-sensitive embedding module. Finally, we introduce an illumination attention fusion module to adaptively aggregate these complementary features according to the estimated illumination mask.

## 3.2 Architecture Design

We propose a feature adaptation module that bridges the gap between sRGBtrained detectors and RAW images. This module replaces the detector’s initial layer and is jointly optimized with the rest of the network, enabling seamless integration with various detection architectures while maintaining strong performance.

Given an input RAW image X, we first estimate an illumination mask using an illumination estimation module (IE). Next, a channel-sensitive embedding (CSE) module extracts features with varying sensitivity to bright and dark regions, guided by the red/blue and green channels, respectively. Finally, an illumination attention fusion (IAF) module selectively integrates the extracted features under the guidance of the estimated illumination mask. The overall architecture is illustrated in Figure 2.

Illumination Estimation. RAW images contain abundant information in both bright and dark regions, which pose distinct challenges for object detectors. Specifically, dark regions often exhibit considerable sensor noise, while bright regions are prone to overexposure. To enable region-specific processing, we estimate an illumination mask that highlights the spatial distribution of brightness levels across the scene. Previous methods typically rely on Retinex theory [49] to learn illumination estimators via deep networks. However, these approaches tend to introduce significant computational overhead. In contrast, we adopt a lightweight yet efective strategy by utilizing the mean value of the RAW image along the channel dimension as an illumination prior. This prior is concatenated with the original input and passed to a compact estimator E to produce the illumination mask.

$$
\begin{array} { r } { \mathbf { M } = \mathcal { E } \Big ( \operatorname { C o n c a t } \big ( \mathbf { X } , \operatorname { M e a n } _ { c } ( \mathbf { X } ) \big ) \Big ) , } \end{array}\tag{1}
$$

where $\mathcal { E }$ denotes the illumination estimator, M is the predicted illumination mask, and Concat and Mea $\mathrm { n } _ { c }$ represent the channel-wise concatenation and mean operations, respectively.

The architecture of the illumination estimator E is illustrated in Figure $\mathrm { 2 \ ( a ) }$ It first employs a $1 \times 1$ convolutional layer to fuse the concatenated input. To improve discrimination between bright and dark regions, a $5 \times 5$ convolutional layer is then applied to capture spatial context. Finally, another $1 \times 1$ convolutional layer is used to generate the illumination mask. Notably, we avoid any nonlinear activation functions in the design of $\mathcal { E } .$ , which enables the illumination prior and estimator $\mathcal { E }$ to be reparameterized into a single convolution during inference. More details on this reparameterization are provided in Section 3.3.

Channel-Sensitive Embedding. Backbones designed for sRGB images, such as CNNs and Transformers, typically adopt a single convolutional layer to embed input images into feature space, implicitly treating all three color channels with equal importance. However, this assumption does not hold for RAW images, which exhibit substantial variations across diferent color channels. Notably, the green channel contains twice as many samples as the red or blue channels due to the Bayer pattern. Since the color channels share similar edge structures, modern ISPs often use the green channel to guide the spatial reconstruction of red and blue channels. However, this domain-specific prior has not been explicitly utilized in existing RAW-based object detection methods. Moreover, the three color channels demonstrate distinct sensitivities to illumination intensity. Motivated by these observations, we propose to learn independent embedding layers, each guided by either the green or the red/blue channels, thereby enhancing the model’s capability to recognize objects in regions with varying brightness levels.

The proposed channel-sensitive embedding module is illustrated in Figure 2 (b). It begins by unpacking the input RAW image into three branches. The basic branch includes all color channels and applies a $3 \times 3$ convolution to extract a general feature map $\mathbf { Y } _ { R G B G }$ . Two guidance branches are constructed, one for the green channel and another for the red and blue channels. Each branch passes through a $3 \times 3$ convolution to generate features $\mathbf { Y } _ { G }$ and ${ \bf Y } _ { R B }$ , respectively. Next, the features from the basic branch are concatenated with those from each guidance branch and fused through lightweight aggregation modules.

$$
\begin{array} { r l } & { \mathbf { Y } _ { R B } ^ { \prime } = \mathcal { F } _ { R B } \Big ( \operatorname { C o n c a t } ( \mathbf { Y } _ { R B } , \mathbf { Y } _ { R G B G } ) \Big ) } \\ & { ~ \mathbf { Y } _ { G } ^ { \prime } = \mathcal { F } _ { G } \Big ( \operatorname { C o n c a t } ( \mathbf { Y } _ { G } , \mathbf { Y } _ { R G B G } ) \Big ) , } \end{array}\tag{2}
$$

where $\mathcal { F } _ { R B }$ and $\mathcal { F } _ { G }$ denote the aggregation modules for the red/blue and green branches, respectively. $\mathbf { Y } _ { R B } ^ { \prime }$ and $\mathbf { Y } _ { G } ^ { \prime }$ are the resulting channel-sensitive embedded features. For simplicity and eficiency, each aggregation module is implemented as a $, ~ 1 \times 1$ convolution, enabling reparameterization before inference. This design allows the module to be equivalently simplified into two independent $3 \times 3$ convolutions at test time, without additional computational cost.

Illumination Attention Fusion. After obtaining the illumination mask M and the embedded features $\mathbf { Y } _ { R B } ^ { \prime }$ and $\mathbf { Y } _ { G } ^ { \prime } .$ , we perform feature fusion under illumination guidance. Conventional addition or concatenation applies fixed fusion rules and cannot adapt to spatial brightness variations. To address this limitation, the proposed IAF module uses M to guide dynamic and global feature interaction.

As illustrated in Figure 2 (c), IAF is built upon cross-attention. Specifically, the illumination mask is projected into the query Q, which provides spatial intensity guidance, while the two embedded features are separately projected into key–value pairs $\left( \mathbf { K } _ { R B } , \mathbf { V } _ { R B } \right)$ and $( \mathbf { K } _ { G } , \mathbf { V } _ { G } )$ . The keys encode global channel relationships for attention estimation, whereas the values retain the branch-specific features to be fused. To reduce computational complexity, adaptive average pooling is applied only to the query and key branches. Following the channel-first layout used in our implementation and omitting the batch dimension, let the pooled spatial size be $h \times w$ , with $N = h w$ , and let $H \times W$ denote the original resolution. For each attention head, $\mathbf { Q } , \mathbf { K } _ { R B } , \mathbf { K } _ { G } \in \mathbb { R } ^ { d _ { k } \times N }$ , where $d _ { k }$ is the channel dimension of each head. The two keys are concatenated along the channel dimension as $\mathbf { K } \in \mathbb { R } ^ { 2 d _ { k } \times N }$ . Meanwhile, $\dot { \mathbf { V } } _ { R B } , \mathbf { V } _ { G } \in \mathbb { R } ^ { d _ { k } \times H W }$ retain the original spatial resolution. They are modulated by M and $1 - \mathbf { M } .$ , respectively, and concatenated along the channel dimension as $\mathbf { V } \in \mathbb { R } ^ { 2 d _ { k } \times H \dot { W } }$ . The fused feature is computed as

$$
\mathbf { Y } = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d _ { k } } } \right) \mathbf { V } ,\tag{3}
$$

where the softmax is applied along the last dimension of $\mathbf { Q } \mathbf { K } ^ { \top }$ . Accordingly, $\mathbf { Q } \mathbf { K } ^ { \top } \in \mathbb { R } ^ { d _ { k } \times 2 d _ { k } }$ captures global interactions between the query channels and the concatenated RB/G key channels, while $\mathbf { Y } \in \mathbb { R } ^ { d _ { k } \times H W }$ preserves the original spatial resolution. Finally, the outputs of all heads are concatenated and projected by a $3 \times 3$ convolution.

## 3.3 Training and Inference

Training details. The proposed method is jointly optimized with the detector and can be integrated into representative CNN- and Transformer-based detection frameworks by replacing the input embedding layer. It utilizes the default loss functions of the underlying detector without requiring additional modifications. Although detector performance benefits significantly from ImageNet-pretrained weights, our method modifies the original network architecture, which may limit the full utilization of pretrained features and exacerbate the domain gap. Nevertheless, we observe that ImageNet-pretrained weights still provide substantial performance gains for our approach. A detailed analysis is presented in Section 4.3.

Inference. To reduce computational overhead during inference, we employ a reparameterization strategy for both the illumination estimation (IE) and channel-sensitive embedding (CSE) modules. These modules are designed exclusively with convolutional and concatenation operations, enabling straightforward fusion. The channel-wise mean operation can be represented as a $1 \times 1$ convolution with kernel size $1 \times C \times 1 \times 1$ , where all kernel weights are set to $1 / C$ . For the green and red/blue guidance branches, missing color channels are aligned with the basic branch by introducing zero-valued convolutional kernels, allowing the branches to be merged seamlessly. Consequently, at inference time, the IE module requires only a single $5 \times 5$ convolution, and the CSE module involves two $3 \times 3$ convolutions, thereby substantially reducing computational cost.

## 4 Experiments

In this section, we first describe the experimental setup. Next, we present detailed results of our experiments. Finally, we perform ablation studies to analyze the contributions of key components.

## 4.1 Experimental Setup

Dataset. We conduct experiments on three RAW datasets, namely AODRaw [27], PascalRAW [35], and LOD [21]. The AODRaw dataset comprises 7,785 real RAW images annotated across four categories, covering diverse challenging conditions such as daylight, low-light, rain, and fog. We adhere to the oficial data splits provided. PascalRAW is a widely used RAW object detection benchmark. Following the evaluation protocol in [7], we assess performance under three conditions, namely normal, overexposure, and dark. Note that the latter two conditions involve synthetically generated images based on [8]. The LOD dataset focuses on real-world low-light object detection and contains 2,230 RAW images with annotations for eight categories. Following [7], we use 1,800 images for training and 430 for testing.

Comparison Methods. To evaluate the efectiveness of the proposed method, we compare it against various open-source ISP pipelines and RAW-based object detection approaches. Karaimer et al. [24] introduce a manually designed traditional ISP pipeline. InvISP [52] represents a network-based ISP method. RAOD [53] and RAW-Adapter [7] are two state-of-the-art RAW detection methods. For a comprehensive evaluation, experiments are conducted using multiple detectors. On the AODRaw dataset, we employ Cascade-RCNN [2], a two-stage CNN-based detector, and Deformable DETR [58], a Transformer-based detector. For the PascalRAW and LOD datasets, we adopt RetinaNet [29], a one-stage CNN-based detector. For methods reproduced in our implementation, we use the same data augmentation protocols and training configurations. Results taken directly from previous studies are explicitly marked with \* in the corresponding tables.

Implementation Details. We implement all methods using the open-source object detection toolbox MMDetection [4]. For the AODRaw dataset, Cascade-RCNN is trained for 12 epochs, while Deformable DETR is trained for 25 epochs. For the PascalRAW and LOD datasets, the detectors are trained for 50 and 35

Table 1: Comparison results on the AODRaw dataset. We use $\mathrm { m A P }$ and AP under diferent weather conditions to evaluate the detection performance. Results marked with \* correspond to the values reported in [27]. Here, C-RCNN and D-DETR denote Cascade-RCNN and Deformable DETR. Params and FPS denote the number of model parameters in millions and the inference speed measured on one NVIDIA RTX 3090 GPU, respectively.
<table><tr><td>Detector</td><td>Method</td><td>|mAP</td><td> $\mathrm { A P _ { 5 0 } }$ </td><td> $\mathrm { A P } _ { 7 5 }$ </td><td>[Normal</td><td>Low</td><td>Rain</td><td> $\mathrm { F o g }$ </td><td>|Params</td><td>FPS</td></tr><tr><td>C-RCNN</td><td>sRGB</td><td>18.4</td><td>29.5</td><td>19.5</td><td>21.0</td><td>16.1</td><td>15.7</td><td>14.4</td><td>60.96</td><td>40.8</td></tr><tr><td>C-RCNN</td><td>RAW*</td><td>18.1</td><td>30.9</td><td>18.3</td><td>20.6</td><td>16.7</td><td>17.3</td><td>14.6</td><td>60.96</td><td>40.8</td></tr><tr><td>C-RCNN</td><td>Karaimer et al.</td><td>18.3</td><td>29.7</td><td>19.6</td><td>22.1</td><td>16.1</td><td>16.1</td><td>13.8</td><td>60.96</td><td>40.8</td></tr><tr><td>C-RCNN</td><td>InvISP</td><td>17.4</td><td>28.7</td><td>18.3</td><td>21.3</td><td>15.6</td><td>13.1</td><td>13.0</td><td>61.10</td><td>30.7</td></tr><tr><td>C-RCNN</td><td>RAOD</td><td>21.1</td><td>33.6</td><td>22.2</td><td>23.8</td><td>19.3</td><td>19.9</td><td>17.4</td><td>61.03</td><td>35.6</td></tr><tr><td>C-RCNN</td><td>RAW-Adapter*</td><td>19.9</td><td>33.2</td><td>20.1</td><td>22.3</td><td>18.1</td><td>20.8</td><td>16.9</td><td>61.11</td><td>36.2</td></tr><tr><td>C-RCNN</td><td>ILLRAW</td><td>23.3</td><td>36.0</td><td>24.8</td><td>26.7</td><td>20.9</td><td>23.3</td><td>18.4</td><td>61.10</td><td>38.1</td></tr><tr><td>D-DETR sRGB*</td><td></td><td>16.6</td><td>31.9</td><td>15.6</td><td>18.3</td><td>15.2</td><td>16.4</td><td>13.1</td><td>40.11</td><td>43.1</td></tr><tr><td>D-DETR</td><td>RAW</td><td>20.5</td><td>36.9</td><td>20.5</td><td>23.9</td><td>18.5</td><td>19.6</td><td>16.2</td><td>40.11</td><td>43.1</td></tr><tr><td>D-DETR</td><td>Karaimer et al.</td><td>20.4</td><td>36.9</td><td>20.4</td><td>23.5</td><td>18.8</td><td>18.1</td><td>16.5</td><td>40.11</td><td>43.1</td></tr><tr><td>D-DETR</td><td>InvISP</td><td>19.8</td><td>35.3</td><td>19.3</td><td>23.6</td><td>18.0</td><td>17.8</td><td>14.9</td><td>40.25</td><td>37.1</td></tr><tr><td>D-DETR</td><td>RAOD</td><td>17.2</td><td>30.6</td><td>17.1</td><td>19.6</td><td>15.6</td><td>14.5</td><td>12.1</td><td>40.19</td><td>41.2</td></tr><tr><td>D-DETR</td><td>RAW-Adapter</td><td>18.0</td><td>32.7</td><td>17.4</td><td>20.8</td><td>17.0</td><td>15.6</td><td>13.0</td><td>40.68</td><td>39.3</td></tr><tr><td>D-DETR ILLRAW</td><td></td><td>22.3</td><td>36.6</td><td>23.5</td><td>25.4</td><td></td><td>20.1 22.4 18.6</td><td></td><td>40.26</td><td>42.3</td></tr></table>

epochs, respectively. A batch size of 4 is used for all detectors. Data augmentation includes random horizontal flipping and random resizing. All models reproduced in our implementation are initialized with ImageNet-1K-pretrained weights by default. Experiments are conducted on NVIDIA RTX 3090 GPUs.

## 4.2 Experimental Results

We perform comprehensive quantitative and qualitative evaluations to compare the performance of all methods on three benchmark datasets.

Quantitative Comparison. Table 1 presents the results on the AODRaw dataset. The proposed method achieves highly competitive results across all evaluated detectors. Specifically, with the Cascade-RCNN detector, ILLRAW outperforms the strongest baseline RAOD by 2.2 mAP (improving from 21.1 to 23.3). Training directly on RAW data yields an mAP of 18.1. This value is 0.3 lower than the mAP of 18.4 obtained by training on sRGB images. The opensource ISP methods achieve moderate accuracy on Cascade-RCNN and do not surpass the sRGB-based detector. With Deformable DETR, direct RAW training and the Karaimer pipeline achieve 20.5 and 20.4 mAP, respectively, both outperforming the sRGB baseline of 16.6 mAP. This architecture-dependent result suggests that the relative efectiveness of RAW processing strategies varies with the underlying detector. Regarding RAW-specific detection methods, RAOD and RAW-Adapter achieve improvements of 3.0 and 1.8 mAP, respectively, over direct RAW training on Cascade-RCNN, demonstrating meaningful gains. With Deformable DETR, however, they achieve 17.2 and 18.0 mAP and therefore do not improve upon direct RAW training. In contrast, ILLRAW reaches 22.3 mAP, exceeding direct RAW training by 1.8 mAP. Analyzing performance under diferent weather conditions, our method maintains strong accuracy across all scenarios. On Cascade-RCNN, ILLRAW achieves the highest result in normal lighting conditions (26.7 AP), significantly outperforming other methods. Although performance drops under foggy conditions (18.4 AP), it still leads the comparison. Similar trends are observed with Deformable DETR, highlighting the robustness and adaptability of our approach to diverse environmental conditions. In terms of model complexity and computational eficiency, all methods introduce only marginal increases in parameter count compared to baseline detectors. However, inference speeds vary. InvISP, due to its reliance on a heavy open-source ISP pipeline, exhibits the lowest FPS. RAOD and RAW-Adapter also incur noticeable frame rate reductions. Our method yields the smallest decrease in inference speed, with a drop of only 2.7 FPS on Cascade-RCNN and 0.8 FPS on Deformable DETR, demonstrating an excellent trade-of between accuracy and eficiency.

Table 2: Comparison results on the LOD dataset. The detector is RetinaNet [29] with a ResNet-50 backbone. We report the AP for each category and the overall mAP. Results marked with \* correspond to the values reported in [27].
<table><tr><td>Method</td><td>|Bicycle</td><td>Car</td><td>Motor.</td><td>Chair</td><td>Dining</td><td>Bottle</td><td>TV</td><td>Bus</td><td>Total</td></tr><tr><td>sRGB*</td><td>67.6</td><td>89.1</td><td>59.9</td><td>77.3</td><td>39.8</td><td>34.8</td><td>50.9</td><td>55.9</td><td>59.4</td></tr><tr><td>RAW*</td><td>71.9</td><td>89.3</td><td>57.7</td><td>74.6</td><td>37.1</td><td>31.0</td><td>43.6</td><td>62.4</td><td>58.5</td></tr><tr><td>Karaimer et al.*</td><td>66.9</td><td>87.3</td><td>57.6</td><td>71.3</td><td>30.2</td><td>29.7</td><td>35.8</td><td>56.0</td><td>54.4</td></tr><tr><td>InvISP*</td><td>72.6</td><td>89.2</td><td>59.9</td><td>76.8</td><td>38.9</td><td>31.9</td><td>31.4</td><td>56.6</td><td>57.1</td></tr><tr><td>RAOD</td><td>71.0</td><td>89.9</td><td>28.2</td><td>83.4</td><td>32.4</td><td>31.2</td><td>34.8</td><td>63.4</td><td>54.2</td></tr><tr><td>RAW-Adapter*</td><td>76.2</td><td>89.7</td><td>58.0</td><td>77.4</td><td>40.6</td><td>43.2</td><td>46.0</td><td>65.9</td><td>62.1</td></tr><tr><td>ILLRAW</td><td>73.6</td><td>90.1</td><td>59.2</td><td>76.8</td><td>38.0</td><td>30.6</td><td>63.3</td><td>66.9</td><td>62.3</td></tr></table>

Table 3: Comparison results on the PascalRAW dataset. The detector is RetinaNet [29] with a ResNet-18 backbone. We present AP under three diferent illumination conditions. Results marked with \* correspond to the values reported in [7].
<table><tr><td>Method</td><td>[Normal Overexposure Dark</td><td></td></tr><tr><td>sRGB*</td><td>88.3</td><td>1</td></tr><tr><td>RAW*</td><td>87.7</td><td>87.7 80.3</td></tr><tr><td>Karaimer et al.*</td><td>88.1</td><td>85.6 78.8</td></tr><tr><td>InvISP*</td><td>85.4</td><td>86.6 70.9</td></tr><tr><td>RAOD</td><td>88.1</td><td>88.1 83.9</td></tr><tr><td>RAW-Adapter*</td><td>88.7</td><td>88.7 82.5</td></tr><tr><td>ILLRAW (Ours)</td><td>88.7</td><td>88.5 84.7</td></tr></table>

We evaluate the per-class performance on the LOD dataset using RetinaNet with a ResNet-50 backbone. As shown in Table 2, ILLRAW achieves the highest overall accuracy and outperforms RAW-Adapter by 0.2 mAP. Across categories, it achieves the best results in the car, TV monitor, and bus classes, while maintaining competitive performance in the remaining classes. In the TV monitor category, ILLRAW surpasses the strongest baseline by 12.4 AP. We also observe that open-source ISP methods fail to outperform even the simple RAW baseline.

Table 3 reports the results on PascalRAW using RetinaNet with a ResNet-18 backbone. Under normal exposure, ILLRAW achieves 88.7 AP, tying RAW-Adapter for the best result. Direct RAW training obtains 87.7 AP, which is 0.6 AP lower than the sRGB baseline. Under overexposure, ILLRAW reaches 88.5 AP, 0.2 AP below RAW-Adapter and 0.4 AP above RAOD. Under dark exposure, ILLRAW achieves the best result of 84.7 AP, outperforming RAOD and RAW-Adapter by 0.8 and 2.2 AP, respectively. From overexposure to dark exposure, ILLRAW decreases by 3.8 AP, the smallest drop among the methods evaluated under both conditions. These results indicate that ILLRAW is particularly robust under dark exposure while remaining competitive under normal and overexposed conditions.

![](images/3e8a8c4f004b378fe763629bbbfdbf14ef56910870fee913ab1d53b80c98b378.jpg)

![](images/afe3794a6ca71399685cff56693f0077922346322049365a28cfcbef371ac452.jpg)

![](images/f676e344d2e97048208970fa98c7c443d72f5d06dd59116f2867c11620296936.jpg)

![](images/21806ee99bbd46a4a11ac1ac80976c892feef0e37f30e213b320f341e5f69764.jpg)  
(e) InvISP

![](images/5a4206dd521521d028e320ca0628ef36b3e60c1dfd36dd0016f0eaa6432e0bae.jpg)  
(f ) RAOD

![](images/69c88d2aab9d4390c073425673625fe2c1cc58f4a56e679718842bc6b065afc3.jpg)

![](images/8186abc37bfc9ac2e62d2dcf7f0559436fc0389d6b9d56327eee41f5f2a45300.jpg)  
(g) RAW-Adapter

![](images/4c89e15e6563f9649308cf2fe25b5c9c74092b0a109b100e917267a4011d701b.jpg)  
(h) ILLRAW (Ours)  
Fig. 3: Visual examples of detection results with all the comparison methods. Please zoom in for confidence scores and class predictions.

Visual Comparison. We further visualize detection results in Figure 3. The camera-generated sRGB image shows higher visual quality, with better color reproduction and sharper edges. However, it fails to detect most small objects, such as trafic lights and surveillance cameras. In contrast, most RAW-based methods detect the majority of these small objects. The two open-source ISP methods, Karaimer et al. [24] and InvISP [52], improve image brightness and enhance recognition in dark regions. However, their processing changes spatial information and leads to several false positives, such as trees misclassified as trafic cones and vehicles incorrectly detected near image borders. The RAWbased detectors RAOD [53] and RAW-Adapter [7] show promising results but still fail to handle dark regions efectively; several objects in the darker left area remain undetected. Our method recalls some objects missing from the dataset annotations, such as a trafic light in the left dark region and distant vehicles. These objects are inherently dificult to detect, and this result highlights the robustness of our method rather than annotation errors. Figure 4 shows the estimated illumination mask M and its complement 1 − M. These masks clearly separate regions with diferent brightness levels and guide the model to apply illumination-aware processing.

![](images/6451f7bd8e2b9d3c2251a23171baf0817f8984e833fde4ba55b113429310335f.jpg)  
(a) M

![](images/3124fcc8647a47f3b7856df69c443d7da580ec364f9c44ce090be303cb5ea5c2.jpg)  
(b) 1 − M  
Fig. 4: Visualization of the illumination mask M and its complement 1 − M. They efectively distinguish diferent brightness regions in the image, guiding the model to apply distinct operations for areas with varying illumination.

## 4.3 Ablation Studies

In this section, we perform ablation studies to validate the efectiveness of the proposed model components. Experiments are conducted on the AODRaw dataset using Cascade-RCNN as the detector.

Efects of CSE. We first assess the contribution of CSE. The RAW baseline of 18.1 mAP in Table 1 is reported by [27] using gamma-corrected RAW input. In contrast, the 16.8 mAP baseline in Table 4 is retrained without gamma correction; ILLRAW and all ablation variants follow this same setting. As presented in Table 4, introducing either the green (G) or red/blue (RB) guidance branches to the baseline significantly improves performance. Furthermore, combining both guidance branches achieves the best results, even when using a simple summation for fusion, yielding an improvement of 5.2 mAP.

Table 4: Ablation study on AODRaw without gamma correction.
<table><tr><td colspan="2">CSE</td><td colspan="4">Fusion module</td><td rowspan="2">mAP</td></tr><tr><td>G</td><td>RB</td><td>Mask</td><td>Sum</td><td>Cat</td><td>IAF</td></tr><tr><td>一</td><td></td><td></td><td>一</td><td></td><td></td><td>16.8</td></tr><tr><td>√</td><td>一</td><td></td><td>一</td><td>一</td><td>一</td><td>20.4</td></tr><tr><td>一</td><td>√</td><td>-</td><td>-</td><td>-</td><td>一</td><td>21.1</td></tr><tr><td>√</td><td>√</td><td>一</td><td>√</td><td></td><td></td><td>22.0</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>一</td><td></td><td>22.1</td></tr><tr><td>√</td><td>√</td><td>-</td><td>一</td><td>√</td><td>一</td><td>22.4</td></tr><tr><td>√</td><td>√</td><td>一</td><td>一</td><td>一</td><td>√</td><td>22.8</td></tr><tr><td>√</td><td>√</td><td>√</td><td>一</td><td>一</td><td>√</td><td>23.3</td></tr></table>

Efects of IAF. We investigate the benefits of the proposed IAF by comparing three fusion strategies. These strategies are direct feature summation, weighted summation guided by the illumination mask, and concatenation followed by convolutional aggregation. Given that the illumination mask plays a key role in guiding feature fusion within the IAF module, we conduct an ablation study to evaluate its impact alongside the fusion methods. For the “IAF on, Mask of” setting, the query is generated from the embedded features $\mathbf { Y } _ { R B } ^ { \prime }$ and $\mathbf { Y } _ { G } ^ { \prime }$ instead of

M, and neither value branch is modulated by M or 1 − M; all key/value projections and attention operations remain unchanged. As shown in Table 4, all three fusion strategies achieve comparable accuracy. Weighted summation improves mAP by 0.1, enhancing the model’s sensitivity to bright regions. Concatenationbased fusion ofers greater flexibility, resulting in 22.4 mAP. Our proposed IAF module surpasses concatenation by 0.9 mAP, reaching 23.3 mAP. This gain is attributed to the IAF’s dynamic utilization of the illumination mask for attentionguided fusion. Furthermore, removing the illumination mask from our method causes a 0.5 mAP drop, underscoring the critical role of the illumination mask in fusion.

Impact of pretrained weights. We analyze the impact of backbone pretraining on our method using a ResNet-18 backbone. Pretrained weights usually improve detector performance, but our modification of the first layer reduces the direct applicability of standard ImageNet weights. We compare four initialization strategies. These strategies are random initialization, ResNet-18 pretrained weights, ILLRAW

Table 5: Impact of backbone pretraining on our method.
<table><tr><td>Method</td><td>mAP</td><td> $\mathrm { A P _ { 5 0 } }$ </td><td> $\mathrm { A P _ { 7 5 } }$ </td></tr><tr><td>Random</td><td>21.7</td><td>34.7</td><td>22.8</td></tr><tr><td>ResNet-18</td><td>22.4</td><td>36.1</td><td>24.0</td></tr><tr><td>sRGB</td><td>23.0</td><td>36.2</td><td>24.7</td></tr><tr><td>Synthesized RAW</td><td>23.3</td><td>36.0</td><td>24.8</td></tr></table>

pretrained on ImageNet in the sRGB domain, and ILLRAW pretrained on ImageNet with synthesized RAW images [1]. The results in Table 5 show that our method achieves competitive performance without pretrained weights. Using ResNet pretrained weights provides a small improvement to 22.4 mAP. Pretraining ILLRAW on sRGB images adds another 0.6 mAP, which compensates for the missing weights in the modified layer. Pretraining on synthesized RAW images gives the best performance at 23.3 mAP, likely because the pretraining domain matches RAW data and improves illumination modeling and noise robustness.

## 5 Conclusion

In this paper, we propose a novel illumination-aware RAW object detection method, addressing the unique challenges posed by the inherent properties of RAW images. Unlike prior approaches that rely on human visual priors to design learnable ISPs, our method, ILLRAW, leverages the intrinsic variations in light intensity perception across both channel and spatial dimensions of RAW data. We introduce an innovative illumination mask estimation module alongside a channel-sensitive embedding strategy, efectively capturing and processing the semantic information embedded within RAW images. Furthermore, an illumination attention fusion mechanism adaptively enhances feature integration across regions with varying brightness levels. Collectively, these contributions significantly advance the accuracy and eficiency of RAW-based object detection.

Acknowledgments. This work was funded by the China Postdoctoral Science Foundation 2025M784248.

## References

1. Brooks, T., Mildenhall, B., Xue, T., Chen, J., Sharlet, D., Barron, J.T.: Unprocessing images for learned raw denoising. In: CVPR. pp. 11036–11045 (2019)

2. Cai, Z., Vasconcelos, N.: Cascade r-cnn: Delving into high quality object detection. In: CVPR. pp. 6154–6162 (2018)

3. Chen, C., Chen, Q., Xu, J., Koltun, V.: Learning to see in the dark. In: CVPR. pp. 3291–3300 (2018)

4. Chen, K., Wang, J., Pang, J., Cao, Y., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Xu, J., et al.: Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155 (2019)

5. Cheng, Y., Liu, X., Yang, J.: Recaptured raw screen image and video demoireing via channel and spatial modulations. NeurIPS 36, 40414–40425 (2023)

6. Conde, M.V., Vasluianu, F., Timofte, R.: Bsraw: Improving blind raw image superresolution. In: WACV. pp. 8500–8510 (2024)

7. Cui, Z., Harada, T.: Raw-adapter: Adapting pre-trained visual model to camera raw images. In: ECCV. pp. 37–56. Springer (2024)

8. Cui, Z., Qi, G.J., Gu, L., You, S., Zhang, Z., Harada, T.: Multitask aet with orthogonal tangent regularity for dark object detection. In: ICCV. pp. 2553–2562 (2021)

9. Dalal, N., Triggs, B.: Histograms of oriented gradients for human detection. In: CVPR. vol. 1, pp. 886–893. Ieee (2005)

10. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: CVPR. Ieee (2009)

11. Diamond, S., Sitzmann, V., Julca-Aguilar, F., Boyd, S., Wetzstein, G., Heide, F.: Dirty pixels: Towards end-to-end image processing and perception. ACM TOG 40(3), 1–15 (2021)

12. Dong, C., Loy, C.C., He, K., Tang, X.: Learning a deep convolutional network for image super-resolution. In: ECCV. pp. 184–199. Springer (2014)

13. Duan, K., Bai, S., Xie, L., Qi, H., Huang, Q., Tian, Q.: Centernet: Keypoint triplets for object detection. In: ICCV. pp. 6569–6578 (2019)

14. Ehret, T., Davy, A., Arias, P., Facciolo, G.: Joint demosaicking and denoising by fine-tuning of bursts of raw images. In: ICCV. pp. 8868–8877 (2019)

15. Everingham, M., Eslami, S.A., Van Gool, L., Williams, C.K., Winn, J., Zisserman, A.: The pascal visual object classes challenge: A retrospective. IJCV 111, 98–136 (2015)

16. Felzenszwalb, P.F., Girshick, R.B., McAllester, D., Ramanan, D.: Object detection with discriminatively trained part-based models. IEEE TPAMI 32(9), 1627–1645 (2009)

17. Ge, Z., Liu, S., Wang, F., Li, Z., Sun, J.: Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430 (2021)

18. Girshick, R.: Fast r-cnn. In: ICCV. pp. 1440–1448 (2015)

19. He, K., Sun, J., Tang, X.: Single image haze removal using dark channel prior. IEEE TPAMI 33(12), 2341–2353 (2010)

20. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR. pp. 770–778 (2016)

21. Hong, Y., Wei, K., Chen, L., Fu, Y.: Crafting object detection in very low light. In: BMVC. vol. 1, p. 3 (2021)

22. Huang, H., Yang, W., Hu, Y., Liu, J., Duan, L.Y.: Towards low light enhancement with raw images. IEEE TIP 31, 1391–1405 (2022)

23. Jin, X., Han, L.H., Li, Z., Guo, C.L., Chai, Z., Li, C.: Dnf: Decouple and feedback network for seeing in the dark. In: CVPR. pp. 18135–18144 (2023)

24. Karaimer, H.C., Brown, M.S.: A software platform for manipulating the camera imaging pipeline. In: ECCV. pp. 429–444. Springer (2016)

25. Kronander, J., Gustavson, S., Bonnet, G., Unger, J.: Unified hdr reconstruction from raw cfa data. In: ICCP. pp. 1–9. IEEE (2013)

26. Li, Z., Lu, M., Zhang, X., Feng, X., Asif, M.S., Ma, Z.: Eficient visual computing with camera raw snapshots. IEEE TPAMI 46(7), 4684–4701 (2024)

27. Li, Z.Y., Jin, X., Sun, B., Guo, C.L., Cheng, M.M.: Towards raw object detection in diverse conditions. arXiv preprint arXiv:2411.15678 (2024)

28. Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: CVPR. pp. 2117–2125 (2017)

29. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object detection. In: ICCV. pp. 2980–2988 (2017)

30. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: ECCV. pp. 740–755. Springer (2014)

31. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: ICCV. pp. 10012–10022 (2021)

32. Loh, Y.P., Chan, C.S.: Getting to know low-light images with the exclusively dark dataset. Computer Vision and Image Understanding 178, 30–42 (2019)

33. Mosleh, A., Sharma, A., Onzon, E., Mannan, F., Robidoux, N., Heide, F.: Hardware-in-the-loop end-to-end optimization of camera image processing pipelines. In: CVPR. pp. 7529–7538 (2020)

34. Nishimura, J., Gerasimow, T., Sushma, R., Sutic, A., Wu, C.T., Michael, G.: Automatic isp image quality tuning using nonlinear optimization. In: ICIP. pp. 2471– 2475. IEEE (2018)

35. Omid-Zohoor, A., Ta, D., Murmann, B.: Pascalraw: raw image database for object detection. Stanford Digital Repository 1(3), 4 (2014)

36. Onzon, E., Mannan, F., Heide, F.: Neural auto-exposure for high-dynamic range object detection. In: CVPR. pp. 7710–7720 (2021)

37. Peng, F., Zhang, M., Lai, S., Tan, H., Yan, S.: Deep hdr reconstruction of dynamic scenes. In: ICIVC. pp. 347–351. IEEE (2018)

38. Ramanath, R., Snyder, W.E., Yoo, Y., Drew, M.S.: Color image processing pipeline. IEEE Signal processing magazine 22(1), 34–43 (2005)

39. Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Unified, real-time object detection. In: CVPR. pp. 779–788 (2016)

40. Redmon, J., Farhadi, A.: Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767 (2018)

41. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. NeurIPS 28 (2015)

42. Shao, S., Li, Z., Zhang, T., Peng, C., Yu, G., Zhang, X., Li, J., Sun, J.: Objects365: A large-scale, high-quality dataset for object detection. In: ICCV. pp. 8430–8439 (2019)

43. Shu, Y., Shen, L., Hu, X., Li, M., Zhou, Z.: Towards real-world hdr video reconstruction: A large-scale benchmark dataset and a two-stage alignment network. In: CVPR. pp. 2879–2888 (2024)

44. Tseng, E., Yu, F., Yang, Y., Mannan, F., Arnaud, K.S., Nowrouzezahrai, D., Lalonde, J.F., Heide, F.: Hyperparameter optimization in black-box image processing using diferentiable proxies. ACM TOG 38(4), 27–1 (2019)

45. Viola, P., Jones, M.: Rapid object detection using a boosted cascade of simple features. In: CVPR. vol. 1, pp. I–I. Ieee (2001)

46. Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., et al.: Yolov10: Real-time end-to-end object detection. NeurIPS 37, 107984–108011 (2025)

47. Wang, X., Ma, K., Liu, Q., Zou, Y., Fu, Y.: Multi-object tracking in the dark. In: CVPR. pp. 382–392 (2024)

48. Wang, Y., Xu, T., Fan, Z., Xue, T., Gu, J.: Adaptiveisp: Learning an adaptive image signal processor for object detection. NeurIPS 37, 112598–112623 (2025)

49. Wei, C., Wang, W., Yang, W., Liu, J.: Deep retinex decomposition for low-light enhancement. arXiv preprint arXiv:1808.04560 (2018)

50. Wei, K., Fu, Y., Yang, J., Huang, H.: A physics-based noise formation model for extreme low-light raw denoising. In: CVPR. pp. 2758–2767 (2020)

51. Wu, C.T., Isikdogan, L.F., Rao, S., Nayak, B., Gerasimow, T., Sutic, A., Ainkedem, L., Michael, G.: Visionisp: Repurposing the image signal processor for computer vision applications. In: ICIP. pp. 4624–4628. IEEE (2019)

52. Xing, Y., Qian, Z., Chen, Q.: Invertible image signal processing. In: CVPR. pp. 6287–6296 (2021)

53. Xu, R., Chen, C., Peng, J., Li, C., Huang, Y., Song, F., Yan, Y., Xiong, Z.: Toward raw object detection: A new benchmark and a new model. In: CVPR. pp. 13384– 13393 (2023)

54. Yoshimura, M., Otsuka, J., Irie, A., Ohashi, T.: Dynamicisp: dynamically controlled image signal processor for image recognition. In: ICCV. pp. 12866–12876 (2023)

55. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.: Restormer: Eficient transformer for high-resolution image restoration. In: CVPR. pp. 5728– 5739 (2022)

56. Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., Shum, H.Y.: Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605 (2022)

57. Zhang, K., Luo, W., Zhong, Y., Ma, L., Stenger, B., Liu, W., Li, H.: Deblurring by realistic blurring. In: CVPR. pp. 2737–2746 (2020)

58. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159 (2020)

59. Zong, Z., Song, G., Liu, Y.: Detrs with collaborative hybrid assignments training. In: ICCV. pp. 6748–6758 (2023)

60. Zou, Y., Yan, C., Fu, Y.: Rawhdr: High dynamic range image reconstruction from a single raw image. In: ICCV. pp. 12334–12344 (2023)