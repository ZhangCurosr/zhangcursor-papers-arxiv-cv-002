# Repurposing RGB-based Foundation Model for Depth Estimation on Thermal Images Using Hierarchical Supervision

Jie Hong<sup>1</sup>, Tingtian Li<sup>1</sup>, Xuesong Li<sup>2</sup> and Xiao Li<sup>1</sup>

Abstract— Depth estimation from thermal images is highly valuable for robotic applications in adverse conditions, such as nighttime and rainy weather. Recent studies have sought to transfer knowledge from RGB-based foundation models to thermal modalities, yet the rich hierarchical representations these models encode remain underutilized. To address this limitation, we propose RGB-HS, a novel framework for thermalimage depth estimation that leverages hierarchical supervision from an RGB-based foundation model. Specifically, we first replace the baseline thermal encoder with a foundational model and introduce a parallel RGB branch that also employs a foundational model as an encoder of the same architecture, taking RGB images as input. The alignment is then performed across multiple levels between the tokens of the two encoders, allowing the thermal student branch to capture both structural precision and semantic abstraction from the RGB teacher branch. Furthermore, we introduce verification to refine the alignment process by weighting tokens from the RGB branch based on RGB image quality. Extensive experiments on the popular benchmark demonstrate that RGB-HS achieves competitive performance and more effectively exploits the representational capacity of RGB-based foundation models for depth estimation on thermal images.

## I. INTRODUCTION

Depth estimation is an important task in computer vision, with applications in robotic systems for real-world scenarios [1], [2], [3], [4], [5]. While significant progress has been achieved in predicting depth from RGB images [6], [7], [8], these methods typically rely on the RGB modality and struggle under adverse conditions such as nighttime or rain [9]. In contrast, thermal imaging provides robust perception in low-illumination or visually degraded environments by capturing scene structure through infrared radiation. However, estimating depth directly from thermal images remains challenging due to the inherent sparsity of texture and the limited availability of thermal data.

Recent advances in RGB-based vision foundation models [10], [11], [12] have opened new opportunities for crossmodal learning in thermal-image depth estimation. Largescale pre-trained models, such as DINO [13], [14], [11], encode rich representations that generalize across diverse visual domains. Several studies have attempted to exploit these models for depth estimation on thermal images [15], [16]. For instance, GTDE [15] employs DINOv2 features to enable zero-shot generalization, while RGB-MDE [16] distills knowledge from an RGB–depth branch to guide thermal prediction. Despite their progress, these approaches face two limitations: they do not fully leverage the RGB modality or the multi-level, structural, and semantic information embedded in RGB-based foundation models, and they treat all RGB supervision equally, overlooking the varying quality of RGB data and feature representations across different environmental conditions.

![](images/f56eca78d2f516d19c91af781b302f0f8eaebbb0b72f791b2c9dc131eaf0146d.jpg)  
Fig. 1. Frameworks for depth estimation on thermal images. (a) The baseline model adopts an encoder–decoder architecture to predict depth from thermal input. (b) The baseline encoder is replaced with a powerful foundation encoder for enhanced representation learning. (c) The alignment is introduced to transfer knowledge from the RGB modality. (d) The verification is incorporated to highlight high-quality RGB data, thereby ensuring more reliable alignment.

To overcome these limitations, we propose RGB-HS, a framework for depth estimation on thermal images via Hierarchical Supervision from RGB-based foundation models. The key idea is to transfer structural and semantic knowledge from a frozen RGB teacher encoder to a thermal student encoder through multi-level alignment, while adaptively verifying the quality of such alignment. Specifically, as illustrated in Figure 1, RGB-HS replaces the baseline encoder with a pre-trained foundational encoder and introduces a parallel RGB branch that shares the same architecture as the thermal branch. We then design an alignment mechanism that aligns thermal and RGB features across both map-level and latent-level spaces, enabling the thermal encoder to capture structural precision and semantic abstraction simultaneously. Furthermore, we propose a verification module that evaluates the confidence of the RGB supervision at the image level, ensuring that unreliable RGB signals are down-weighted during knowledge transfer. In summary, our main contributions are summarized as follows:

• We propose RGB-HS, a framework that leverages hierarchical supervision (i.e., verification-guided alignment) from an RGB-based foundation model to improve depth estimation for thermal images.

• For the proposed hierarchical supervision, we introduce an alignment mechanism that transfers structural and semantic cues from the map and latent levels. We also design a verification strategy that adaptively filters unreliable RGB supervision, thereby improving alignment quality.

• Extensive experiments on the Multi-Spectral Stereo $( \mathbf { M } \mathbf { S } ^ { 2 } )$ dataset [9] demonstrate the effectiveness of RGB-HS in learning from RGB modality.

## II. RELATED WORKS

1) Depth Estimation on RGB Images: Depth estimation from RGB images has been extensively studied [17], [18], [19], [20], [21]. The first deep learning approach for this task was introduced by Eigen et al. [6]. Subsequent works have sought to enhance the prediction. For instance, Xian et al. [22] proposed a structure-guided ranking loss with edge- and instance-guided sampling to emphasize geometric details. AdaBins [7] introduced a transformer-based architecture that adaptively partitions the depth range into imagespecific bins. NeWCRF [23] formulated fully-connected conditional random fields (CRFs) within local windows, making them computationally feasible, and integrated them with multi-head attention for monocular depth estimation.

Recently, with the rise of RGB-based foundation models, researchers have explored adapting large pre-trained vision models for depth prediction [24], [8], [25], [11]. Marigold [24] repurposes a pre-trained Stable Diffusion v2 model [26] via efficient fine-tuning, enabling monocular depth estimation by leveraging rich visual priors. DepthAnything [8] trains a foundation model using large-scale unlabeled RGB images with strong data perturbations, while DepthAnythingV2 [25] replaces labeled data with synthetic samples to further enhance generalization. Both methods adopt DINOv2 [14] as their backbone. More recently, DI-NOv3 [11] has been repurposed for monocular depth estimation, achieving state-of-the-art (SOTA) performance. Despite these advances, depth estimation from thermal images remains relatively underexplored.

2) Depth Estimation on Thermal Images: In contrast to RGB-based research, depth estimation on thermal images has only recently gained attention [9], [15], [16]. MTN [27], UDE [28], AMA [29], and MSCRF [9] are the very first deep learning frameworks capable of predicting depth from either a single thermal image or a stereo thermal pair. In this work, we adopt MSCRF [9] as the baseline. GTDE [15] proposes a two-stage strategy that leverages DINOv2 [14] to achieve impressive zero-shot generalization across multiple thermal datasets. RGB-MDE [16] introduces confidenceaware knowledge distillation from an RGB–depth branch and employs DINOv2 as its backbone. ThermoStereoRT [30] also applies knowledge distillation for the task. Beyond depth estimation, RGB-thermal fusion has been explored in tasks such as semantic segmentation [31], object tracking [32], [33], and object detection [34], [35].

Although GTDE [15] utilizes intermediate tokens from a foundation model, it overlooks the potential of the RGB modality itself. RGB-MDE [16] addresses this issue by transferring knowledge from RGB images, but it neglects the information encoded within the intermediate tokens of the foundation model. Moreover, RGB-MDE or ThermoStereoRT does not identify and exclude low-quality RGB samples during knowledge transfer. These limitations motivate our proposed RGB-HS, which leverages hierarchical supervision to exploit both the feature- and data-level advantages of the RGB modality.

## III. METHOD

In this section, we propose RGB-HS, a depth estimation framework for thermal images that leverages hierarchical supervision from an RGB-based foundation model. As shown in Figure 1, RGB-HS consists of three components: a foundation encoder, alignment, and verification.

## A. Foundational Encoder

In this work, we use MSCRF [9] as the baseline, the first deep-learning framework for thermal-image depth estimation. We replace its encoder with an RGB-based foundation model (see Figure 1 (b)). MSCRF originally uses a Swin-L [36] encoder trained from scratch. In RGB-HS, we adopt DINOv3 [11], which achieves state-of-the-art results in RGB-based depth estimation. Specifically, we load a pre-trained ViT-B/16 backbone. The strong representation capability of DINOv3 may provide structural and semantic priors that are beneficial for depth estimation in thermal images.

## B. Hierarchical Supervision: Alignment

Depth estimation from thermal images is inherently challenging due to the lack of texture and color information provided by the RGB sensor. Thermal imagery captures only temperature contrast, leading to ambiguous object boundaries and weak spatial gradients. To better transfer knowledge from the RGB modality and make this transfer reliable, as shown in Figure 1 (c) and (d), we propose a crossmodal framework that transfers the representational power of an RGB-based foundation model (i.e., DINOv3 [11]) to a thermal encoder-decoder network.

More details of our method are depicted in Figure 2. The proposed RGB-HS follows a teacher-student paradigm, where the frozen RGB teacher branch provides rich information supervision, and the thermal student branch reproduces representations for depth prediction by learning from the RGB-branch encoder. We introduce a two-level hierarchical alignment scheme: 1) Map-level alignment refines local structure by ensuring spatial correspondence, and 2) Latentlevel alignment aligns global, semantic relationships between the RGB and thermal modalities. Map-level alignment employs a correlation loss that enforces structural consistency. The latent-level alignment employs cosine similarity and

![](images/007f3efdfa6f791593d7a9308d1c37a9250dadc22865bca6b19faea9c84cf3e2.jpg)  
Fig. 2. Overview of the proposed RGB-HS framework. Built upon an encoder–decoder architecture, RGB-HS introduces an RGB-modality branch that transfers comprehensive, reliable knowledge through alignment and verification modules. Note that the RGB branch, alignment module, and verification module will not be used during inference.

Kullback–Leibler (KL) divergence loss, which enforces directional and distributional consistency.

Given a paired RGB-thermal image pair (I , I ), the RGB and thermal encoders extract hierarchical features: $\mathbf { F } _ { \mathrm { R G B } } \ = \ f _ { \theta _ { \mathrm { R G B } } } ( \mathbf { I } _ { \mathrm { R G B } } ) , \ \mathbf { F } _ { \mathrm { T H R } } \ = \ f _ { \theta _ { \mathrm { T H R } } } ( \mathbf { I } _ { \mathrm { T H R } } )$ where $f _ { \theta _ { \mathrm { R G B } } }$ and $f _ { \theta _ { \mathrm { T H R } } }$ denote the RGB and thermal branch encoders, respectively. Each encoder produces multi-level features ${ \bf F } =$ $\{ \mathbf { F } _ { l } \} _ { l = 1 } ^ { L }$ from its L layers. For the lth layer, the DINOv3 transformer [11] outputs a sequence of tokens. We extract only the spatial patch tokens $\mathbf { T } _ { l } \in \mathbb { R } ^ { N _ { l } \times C _ { l } }$ , excluding class and register tokens, where $N _ { l } = H _ { l } \times W _ { l }$ is the number of spatial positions. To preserve the 2D spatial structure of the original image, these tokens are reshaped into spatial feature maps as follows:

$$
\mathbf { F } _ { l } = \mathrm { R e s h a p e } ( \mathbf { T } _ { l } ) \in \mathbb { R } ^ { C _ { l } \times H _ { l } \times W _ { l } } ,\tag{1}
$$

The decoder $g _ { \phi }$ then processes the thermal features to estimate the dense depth map: $\begin{array} { r l } { \hat { \bf D } } & { { } = } \end{array}$ $g _ { \phi } ( \mathbf { F } _ { \mathrm { T H R , 1 } } , \mathbf { F } _ { \mathrm { T H R , 2 } } , . . . , \mathbf { F } _ { \mathrm { T H R , } L } )$ , where D<sup>ˆ</sup> denotes the predicted dense depth map. Parallel to the thermal branch giving F , RGB-branch encoder $f _ { \theta _ { \mathrm { R G B } } } ( . )$ produces corresponding feature maps $\{ \mathbf { F } _ { \mathrm { R G B } , 1 } , \mathbf { F } _ { \mathrm { R G B } , 2 } , \dots , \mathbf { F } _ { \mathrm { R G B } , L } \}$

1) Map-level Alignment: For depth estimation, it is natural to focus on fine-grained details such as object edges or depth discontinuities. To enhance such details in features, we introduce map-level alignment, which aims to transfer rich structural correspondence from the teacher to the student feature maps (reshaped tokens). We then compute the localized alignment term: correlation alignment loss.

Correlation Loss. The mismatch between RGB and thermal images exists. To avoid pixel-wise alignment but still learn local details, we align the spatial dependency of two feature maps from two modalities:

$$
\mathcal { L } _ { \mathrm { c o r r } } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } { \left. \mathbf { f } _ { \mathrm { T H R } , c } ^ { \top } \mathbf { f } _ { \mathrm { T H R } , c } - \mathbf { f } _ { \mathrm { R G B } , c } ^ { \top } \mathbf { f } _ { \mathrm { R G B } , c } \right. } _ { 2 } ^ { 2 } ,\tag{2}
$$

where $\mathbf { f } _ { \mathrm { T H R } , c }$ and $\mathbf { f } _ { \mathrm { R G B } , c }$ are the single feature maps in ${ \bf F } _ { \mathrm { T H R } }$ or ${ \bf F } _ { \mathrm { R G B } }$ at channel c. The map-level alignment loss is defined as:

$$
{ \mathcal { L } } _ { \mathrm { m a p } } = \lambda ^ { l } { \mathcal { L } } _ { \mathrm { c o r r } } ,\tag{3}
$$

where $\lambda ^ { l }$ is the coefficient. The map-level loss ${ \mathcal { L } } _ { \mathrm { m a p } }$ sharpens the structure of the feature maps learned by the thermal student branch relative to that of the RGB teacher branch.

2) Latent-level Alignment: While the map-level alignment promotes learning local structures, it overlooks the global semantic consistency. Hence, we propose the latent-level alignment, which focuses on transferring high-level semantic representations from the teacher to the student. Here, we aim to align the global latent distributions that encode scene-wide context or semantic cues, rather than spatial pixels. Each feature map is first aggregated via spatial-wise max-pooling to form a global latent representation:

$$
\begin{array} { r } { \bar { \bar { \mathbf { f } } } _ { \mathrm { T H R } } = \mathbf { M a x P o o l } ( \mathbf { F } _ { \mathrm { T H R } } ) , \quad \bar { \bar { \mathbf { f } } } _ { \mathrm { R G B } } = \mathbf { M a x P o o l } ( \mathbf { F } _ { \mathrm { R G B } } ) , } \end{array}\tag{4}
$$

where $\bar { \mathbf { f } } \in \mathbb { R } ^ { C }$ summarizes the overall scene of each feature map at channel c. Here, we use cosine similarity loss and the KL divergence loss. The KL divergence loss helps

align the probabilistic distributions of feature activations across channels, guiding the student to mimic the teacher’s allocation of semantic emphasis.

Cosine Similarity Loss. Similar to building the map-level loss, we then enforce latent-level alignment between the thermal and RGB embeddings using cosine similarity loss:

$$
\mathcal { L } _ { \mathrm { c o s } } ^ { g } = 1 - \frac { \langle \bar { \bf f } _ { \mathrm { T H R } } , \bar { \bf f } _ { \mathrm { R G B } } \rangle } { \Vert \bar { \bf f } _ { \mathrm { T H R } } \Vert _ { 2 } \Vert \bar { \bf f } _ { \mathrm { R G B } } \Vert _ { 2 } } ,\tag{5}
$$

where the cosine similarity loss $\mathcal { L } _ { \mathrm { { c o s } } } ^ { g }$ encourages alignment in the latent semantic space, forcing the student’s global feature vectors to lie in similar directions to those of the teacher.

KL Divergence Loss. To enhance the student’s sensitivity to distributional cues from the teacher, we also have KL divergence loss as follows:

$$
\mathcal { L } _ { \mathrm { K L } } ^ { g } = \mathrm { K L } \bigg ( \mathrm { S o f t m a x } \bigg ( \frac { \bar { \bf f } _ { \mathrm { R G B } } } { T } \bigg ) ~ \bigg \| \mathrm { S o f t m a x } \bigg ( \frac { \bar { \bf f } _ { \mathrm { T H R } } } { T } \bigg ) \bigg ) ,\tag{6}
$$

where $\mathrm { K L } ( . \| . )$ computes the KL divergence between embeddings and $T$ controls the softness of the probability distribution. This term could align the channel-wise probability distributions, ensuring the student captures a similar distributional knowledge from the teacher. The objective of latent-level alignment is defined as follows:

$$
\mathcal { L } _ { \mathrm { l a t e n t } } = \lambda _ { 1 } ^ { g } \mathcal { L } _ { \mathrm { c o s } } ^ { g } + \lambda _ { 2 } ^ { g } \mathcal { L } _ { \mathrm { K L } } ^ { g } ,\tag{7}
$$

where $\lambda _ { 1 } ^ { g }$ and $\lambda _ { 2 } ^ { g }$ are coefficients. By transferring latent-level information, the thermal student might gain global semantic awareness, thereby improving depth estimation in textureless or low-contrast thermal scenes.

## C. Hierarchical Supervision: Verification

A critical challenge in aligning RGB-thermal features arises from the heterogeneous reliability of the RGB modality across varying scene conditions. While the frozen RGB teacher branch yields robust semantic representations under normal conditions, its features become less informative in low-light or adverse weather conditions. Thus, aligning thermal features to RGB features in all scenarios risks suppressing the thermal modality’s unique capabilities. To address this, we propose an adaptive confidence mechanism that dynamically modulates the alignment strength based on confidence signals.

Brightness–contrast Confidence. To assess the perceptual quality of each RGB input, we introduce a simple yet effective brightness–contrast confidence measure. Given an RGB image $\mathbf { \bar { I } } _ { \mathrm { R G B } } \in \mathbb { R } ^ { 3 \times H \times W }$ with intensity values in [0, 255], we first normalize it to [0, 1] and compute its luminance map as follows:

$$
\mathbf { Y } = 0 . 2 1 2 6 \mathbf { R } + 0 . 7 1 5 2 \mathbf { G } + 0 . 0 7 2 2 \mathbf { B } ,\tag{8}
$$

where Y, R, G, and $\mathbf { B } \in \mathbb { R } ^ { H \times W }$ denote the red, green, and blue channels after normalization, respectively. The average luminance and its standard deviation are used to describe the image’s brightness and contrast:

$$
C _ { \mathrm { B T } } = \frac { 1 } { H W } \sum _ { i , j } \mathbf { Y } _ { i , j } ,\tag{9}
$$

$$
C _ { \mathrm { C T } } = \sqrt { \frac { 1 } { H W } \sum _ { i , j } ( { \bf Y } _ { i , j } - C _ { \mathrm { B T } } ) ^ { 2 } } ,\tag{10}
$$

where $C _ { \mathrm { B T } } , C _ { \mathrm { C T } } \in \mathbb { R }$ . To ensure comparability across images, $C _ { \mathrm { C T } }$ is then normalized into [0, 1]. To reflect both image exposure quality and contrast sharpness, the brightness–contrast confidence is the average of these two terms:

$$
C _ { \mathrm { R G B } } = \frac { C _ { \mathrm { B T } } + C _ { \mathrm { C T } } } { 2 } ,\tag{11}
$$

where $C _ { \mathrm { R G B } } \ \in \ \mathbb { R } .$ . Intuitively, C<sub>RGB</sub> is high when the image is neither underexposed nor overexposed and exhibits adequate contrast. It can serve as a simple and interpretable probabilistic indicator of the visual quality of $\mathbf { I } _ { \mathrm { R G B } }$ . We then have $\mathbf { F } _ { \mathrm { T H R } } \  \ C _ { \mathrm { R G B } } \cdot \mathbf { F } _ { \mathrm { T H R } }$ as verified features for the alignment loss calculation.

## D. Overall Loss

The complete training objective integrates supervised depth learning of the baseline with both map- and latentlevel alignment terms:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { b a s e l i n e } } + \frac { 1 } { N L } \big ( \sum _ { n = 1 } ^ { N } \sum _ { l = 1 } ^ { L } \mathcal { L } _ { \mathrm { m a p } } ^ { ( n , l ) } + \sum _ { n = 1 } ^ { N } \sum _ { l = 1 } ^ { L } \mathcal { L } _ { \mathrm { l a t e n t } } ^ { ( n , l ) } \big ) ,\tag{12}
$$

where $\mathcal { L } _ { \mathrm { b a s e l i n e } }$ denotes the supervised regression loss of the baseline, L is the number of encoder layers, N is the batchsize. The hierarchical design allows the thermal branch to learn both structural precision and semantic abstraction from the RGB branch. At inference time, only the thermal encoder–decoder is retained, eliminating any dependency on RGB input.

## IV. EXPERIMENTS

In this section, we present the experimental results of the proposed RGB-HS framework. Experiments are conducted on the large-scale ${ \bf M } { \bf S } ^ { 2 }$ dataset [9]. We evaluate RGB-HS on both monocular and stereo depth estimation tasks and conduct ablation studies to assess the effectiveness of the alignment and verification mechanisms.

## A. Dataset

All experiments are performed on the ${ \bf M } { \bf S } ^ { 2 }$ dataset [9], a large-scale multimodal outdoor benchmark for autonomous driving. The dataset provides synchronized data from stereo RGB, near-infrared (NIR), and thermal cameras, together with stereo LiDAR ground truth. Importantly, ${ \bf M } { \bf S } ^ { 2 }$ includes diverse illumination and weather conditions $( e . g .$ , nighttime and rain), making it suitable for evaluating thermal-based depth estimation. Following the official split, we use 26K, $4 \mathrm { K } ,$ and 17.8K paired samples for training, validation, and testing, respectively.

## B. Implementations

The proposed RGB-HS utilized the MSCRF framework [9] as the baseline. Hence, we adopt the same training settings and hyperparameters as those used in MSCRF. The AdamW optimizer [40] is used with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ . For the proposed alignment and verification modules, the weights of the loss terms are empirically set as $\lambda ^ { l } = 0 . 5 , \lambda _ { 1 } ^ { g } = 1 . 0 .$ , and $\lambda _ { 2 } ^ { g } ~ = ~ 0 . 1$ , based on validation performance. All experiments are conducted on a single NVIDIA RTX PRO 6000 GPU. Our implementation is based on PyTorch [41].

DEPTH ESTIMATION RESULTS FOR THERMAL IMAGES ON THE ${ \bf M } { \bf S } ^ { 2 }$ [9] DATASET. THE BEST TWO NUMBERS IN $\mathbf { \ddot { A V G } } ^ { 5 5 }$ ARE IN BOLD. THE BEST NUMBERS ARE IN RED, AND THE SECOND-BEST ARE IN BLUE. THE RESULTS OF DORN [37], BTS [38], ADABINS [7], NEWCRF [23], AND MSCRF [9] ARE PROVIDED IN [9]. NOTE THAT RGB-HS DOES NOT USE THE RGB IMAGE AS INPUT DURING INFERENCE IN ALL EXPERIMENTS.
<table><tr><td rowspan="2">Method</td><td rowspan="2">TestSet</td><td colspan="4">Error ↓</td><td colspan="3">Accuracy ↑</td></tr><tr><td>AbsRel</td><td>SqRel</td><td>RMSE</td><td>RMSElog</td><td> $\delta < 1 . 2 5$ </td><td> $\delta < 1 . 2 5 ^ { 2 }$ </td><td> $\delta < 1 . 2 5 ^ { 3 }$ </td></tr><tr><td rowspan="4">DORN [37]</td><td>Day</td><td>0.144</td><td>1.288</td><td>5.483</td><td>0.230</td><td>0.856</td><td>0.941</td><td>0.970</td></tr><tr><td>Night</td><td>0.136</td><td>1.136</td><td>5.290</td><td>0.212</td><td>0.863</td><td>0.950</td><td>0.976</td></tr><tr><td>Rain</td><td>0.180</td><td>1.934</td><td>6.735</td><td>0.276</td><td>0.781</td><td>0.910</td><td>0.955</td></tr><tr><td>Avg</td><td>0.151</td><td>1.419</td><td>5.776</td><td>0.237</td><td>0.837</td><td>0.935</td><td>0.968</td></tr><tr><td rowspan="4">BTS [38]</td><td>Day</td><td>0.122</td><td>0.905</td><td>4.923</td><td>0.198</td><td>0.857</td><td>0.951</td><td>0.980</td></tr><tr><td>Night</td><td>0.114</td><td>0.798</td><td>4.701</td><td>0.184</td><td>0.870</td><td>0.959</td><td>0.984</td></tr><tr><td>Rain</td><td>0.157</td><td>1.395</td><td>6.053</td><td>0.243</td><td>0.791</td><td>0.926</td><td>0.969</td></tr><tr><td>Avg</td><td>0.129</td><td>1.008</td><td>5.169</td><td>0.206</td><td>0.843</td><td>0.947</td><td>0.978</td></tr><tr><td rowspan="4">AdaBins [7]</td><td>Day</td><td>0.129</td><td>0.976</td><td>5.108</td><td>0.205</td><td>0.847</td><td>0.947</td><td>0.979</td></tr><tr><td>Night</td><td>0.119</td><td>0.822</td><td>4.749</td><td>0.187</td><td>0.864</td><td>0.958</td><td>0.984</td></tr><tr><td>Rain</td><td>0.168</td><td>1.545</td><td>6.336</td><td>0.254</td><td>0.771</td><td>0.918</td><td>0.965</td></tr><tr><td>Avg</td><td>0.137</td><td>1.084</td><td>5.330</td><td>0.212</td><td>0.831</td><td>0.943</td><td>0.977</td></tr><tr><td rowspan="4">NeWCRF [23]</td><td>Day</td><td>0.120</td><td>0.864</td><td>4.852</td><td>0.195</td><td>0.858</td><td>0.952</td><td>0.982</td></tr><tr><td>Night</td><td>0.112</td><td>0.755</td><td>4.594</td><td>0.179</td><td>0.875</td><td>0.961</td><td>0.985</td></tr><tr><td>Rain</td><td>0.155</td><td>1.352</td><td>5.956</td><td>0.240</td><td>0.795</td><td>0.929</td><td>0.970</td></tr><tr><td>Avg</td><td>0.127</td><td>0.965</td><td>5.077</td><td>0.202</td><td>0.846</td><td>0.949</td><td>0.980</td></tr><tr><td rowspan="4">MSCRF (Mono) [9]</td><td>Day</td><td>0.115</td><td>0.983</td><td>4.895</td><td>0.201</td><td>0.882</td><td>0.952</td><td>0.977</td></tr><tr><td>Night</td><td>0.107</td><td>0.850</td><td>4.658</td><td>0.185</td><td>0.894</td><td>0.961</td><td>0.981</td></tr><tr><td>Rain</td><td>0.152</td><td>1.567</td><td>6.020</td><td>0.247</td><td>0.822</td><td>0.928</td><td>0.964</td></tr><tr><td>Avg</td><td>0.123</td><td>1.103</td><td>5.134</td><td>0.208</td><td>0.869</td><td>0.948</td><td>0.975</td></tr><tr><td rowspan="4">MSCRF (Stereo) [9]</td><td>Day</td><td>0.113</td><td>0.948</td><td>4.852</td><td>0.200</td><td>0.884</td><td>0.953</td><td>0.977</td></tr><tr><td>Night</td><td>0.105</td><td>0.811</td><td>4.584</td><td>0.183</td><td>0.896</td><td>0.961</td><td>0.981</td></tr><tr><td>Rain</td><td>0.149</td><td>1.499</td><td>5.940</td><td>0.245</td><td>0.826</td><td>0.929</td><td>0.965</td></tr><tr><td>Avg</td><td>0.120</td><td>1.057</td><td>5.068</td><td>0.207</td><td>0.872</td><td>0.949</td><td>0.975</td></tr><tr><td rowspan="4">RGB-HS (Mono)</td><td>Day</td><td>0.112</td><td>0.616</td><td>3.750</td><td>0.151</td><td>0.868</td><td>0.974</td><td>0.994</td></tr><tr><td>Night</td><td>0.099</td><td>0.452</td><td>3.223</td><td>0.131</td><td>0.899</td><td>0.985</td><td>0.997</td></tr><tr><td>Rain</td><td>0.141</td><td>0.883</td><td>4.446</td><td>0.181</td><td>0.818</td><td>0.962</td><td>0.990</td></tr><tr><td>Avg</td><td>0.117</td><td>0.650</td><td>3.806</td><td>0.154</td><td>0.862</td><td>0.974</td><td>0.994</td></tr><tr><td rowspan="4">RGB-HS (Stereo)</td><td>Day</td><td>0.098</td><td>0.517</td><td>3.434</td><td>0.128</td><td>0.894</td><td>0.981</td><td>0.997</td></tr><tr><td>Night</td><td>0.093</td><td>0.404</td><td>3.038</td><td>0.120</td><td>0.912</td><td>0.989</td><td>0.998</td></tr><tr><td>Rain</td><td>0.124</td><td>0.797</td><td>4.172</td><td>0.156</td><td>0.856</td><td>0.973</td><td>0.993</td></tr><tr><td>Avg</td><td>0.105</td><td>0.572</td><td>3.548</td><td>0.134</td><td>0.887</td><td>0.981</td><td>0.996</td></tr></table>

## C. Depth Estimation on Thermal Images

We first evaluate the proposed RGB-HS for monocular and stereo depth estimation using thermal images. Quantitative results on the ${ \bf M } { \bf S } ^ { 2 }$ dataset [9] are provided in Table I. We compare RGB-HS against several representative depth estimation methods on thermal images, including DORN [37], BTS [38], AdaBins [7], NeWCRF [23], and MSCRF [9]. Note that RGB-HS adopts MSCRF as the baseline. Moreover, it does not use the RGB teacher branch and RGB images as input during inference.

As shown in Table I, RGB-HS achieves the best performance under most evaluation metrics and testing conditions. Specifically, the proposed RGB-HS (Stereo) achieves the best average performance with an AbsRel of 0.105, SqRel of 0.572, RMSE/RMSElog of 3.548/0.134, and $\delta <$ $1 . 2 5 / 1 . 2 5 ^ { 2 } / 1 . 2 5 ^ { 3 }$ accuracies of 0.887/0.981/0.996. Even under the monocular mode, RGB-HS (Mono) outperforms previous stereo-based models. For example, compared with MSCRF (Stereo), RGB-HS (Mono) achieves an average SqRel of 0.650 and an RMSE of 3.806, reducing AbsRel and RMSE by 0.407 and 1.262, respectively. The above results demonstrate the effectiveness of leveraging hierarchical supervision from the RGB-based foundation model to enhance depth perception on thermal images. There is a clean version of $\mathrm { \bar { M } S ^ { \bar { 2 } } }$ in which the depth of the training data is filtered [42], [16]. We provide results for this clean version of the dataset in Table II, in which the SOTA method RGB-MDE [16] is compared.

The consistent gains across monocular and stereo settings demonstrate that RGB-HS generalizes well across diverse input configurations. It is also observed from Table I that

![](images/b0701b21ef0c85882536e3910b3006f8e6ec96750dcf037d71fb8b87c1a3d53b.jpg)  
Fig. 3. Visualizations of depth estimation results on the MS<sup>2</sup> dataset [9]. Each row shows one example under different environmental conditions (daytime, nighttime, and rainy scenes). From left to right: RGB image (for reference), thermal input image, and predicted depth maps from MSCRF [9], MSCRF with DINOv3 ViT-B/16 backbone, and our RGB-HS (Ours). We highlight the representative regions with the red dashed boxes.

## TABLE II

DEPTH ESTIMATION RESULTS FOR THERMAL IMAGES ON THE CLEANED VERSION OF THE DATASET ${ \bf M } { \bf S } ^ { 2 }$ [9]. THE BEST TWO NUMBERS IN “AVG” ARE IN BOLD. THE BEST NUMBERS ARE IN RED, AND THE SECOND-BEST ARE IN BLUE. THE RESULTS OF DORN [37], BTS [38], ADABINS [7],

NEWCRF [23], ZOEDEPTH [39], DEPTHANYTHING [8], AND RGB-MDE [16] ARE PROVIDED IN [16].
<table><tr><td rowspan="2">Method</td><td rowspan="2">Modality</td><td colspan="4">Error ↓</td></tr><tr><td>AbsRel</td><td>SqRel</td><td>RMSE</td><td>RMSElog</td></tr><tr><td>DORN [37]</td><td>Ther</td><td>0.109</td><td>0.540</td><td>3.660</td><td>0.144</td></tr><tr><td>BTS [38]</td><td>Ther</td><td>0.086</td><td>0.380</td><td>3.163</td><td>0.117</td></tr><tr><td>AdaBins [7]</td><td>Ther</td><td>0.088</td><td>0.377</td><td>3.152</td><td>0.119</td></tr><tr><td>NeWCRF [23]</td><td>Ther</td><td>0.080</td><td>0.331</td><td>2.937</td><td>0.109</td></tr><tr><td>Zoedepth [39]</td><td>Ther</td><td>0.091</td><td>0.425</td><td>3.202</td><td>0.123</td></tr><tr><td>DepthAnything [8]</td><td>Ther</td><td>0.075</td><td>0.287</td><td>2.719</td><td>0.103</td></tr><tr><td>RĠB-MDE [16]</td><td>Ther</td><td>0.072</td><td>0.275</td><td>2.677</td><td>0.100</td></tr><tr><td>RGB-HS (Ours)</td><td>Ther</td><td>0.072</td><td>0.283</td><td>2.595</td><td>0.100</td></tr></table>

RGB-HS consistently maintains high performance across diverse environmental conditions. For instance, the model achieves strong accuracy at night with RMSE=3.038 and in rainy scenes with RMSE=4.172, whereas MSCRF achieves RMSE=4.584 and 5.940. These results suggest that hierarchical supervision enables the model to capture robust informative cues that generalize beyond variations in illumination and weather.

We present qualitative comparisons in Figure 3. As shown in the figure, the proposed RGB-HS generates more accurate depth maps, with more precise object boundaries than other methods. The red dashed boxes highlight representative areas, such as vehicles, cyclists, and road fences, where previous methods often produce blurred or inconsistent depth. In contrast, RGB-HS better preserves structure and maintains global depth continuity, demonstrating the effectiveness of hierarchical supervision in leveraging the knowledge of RGB-based foundation model for thermal depth estimation. For example, as shown in the third row of Figure 3, RGB-HS more accurately predicts the car on the right than the baselines.

## D. Ablation Study

Foundational Encoder. The results obtained with different foundational encoders are presented in Table III. The baseline MSCRF [9] utilizes Swin-L [36] as its backbone, whereas our RGB-HS is constructed upon a pre-trained DINOv3 ViT encoder [11]. We also replace the encoder in MSCRF with ViT models of varying scales. As shown in Table III, to some extent, the performance improves with larger ViT capacities, suggesting that the foundation model’s strong visual priors benefit depth learning in the thermal domain. Notably, RGB-HS with ViT-B/16 achieves excellent overall performance, surpassing MSCRF by significant margins (SqRel=0.572, RMSE=3.548 vs. SqRel=1.057, RMSE=5.068) while using fewer parameters (86M vs. 197M). This suggests that the hierarchical supervision from the RGB-based foundation model effectively transfers the knowledge to the thermal modality.

Map- and Latent-level Alignments. We next analyze the impact of the proposed alignment module. Table IV summarizes the results when enabling or disabling the maplevel loss ${ \mathcal { L } } _ { \mathrm { m a p } }$ and the latent-level loss $\mathcal { L } _ { \mathrm { l a t e n t } }$ . Compared to the baseline MSCRF without alignment, adding either loss alone marginally improves both error (i.e., AbsRel, SqRel, and RMSE) and accuracy performance. The combination of both yields the best performance across all metrics, suggesting complementary roles. The map-level alignment locally enforces structural consistency with RGB features, while the latent-level alignment provides higher-level semantic alignment, jointly enhancing the thermal encoder’s representational power.

TABLE III  
ABLATION STUDY: FOUNDATIONAL ENCODER. WE REPLACE THE BASELINE BACKBONE WITH A VARIETY OF DINOV3 ENCODERS AND COMPARETHEIR PERFORMANCE. THE RESULTS OF STEREO DEPTH ESTIMATION ARE PROVIDED. THE BEST-PERFORMING NUMBERS ARE IN BOLD.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Encoder</td><td rowspan="2">Encoder params</td><td colspan="4">Error ↓</td><td colspan="3">Accuracy ↑</td></tr><tr><td>AbsRel</td><td>SqRel</td><td>RMSE</td><td>RMSElog</td><td> $\delta < 1 . 2 5$ </td><td> $\delta < 1 . 2 5 ^ { 2 }$ </td><td> $\delta < 1 . 2 5 ^ { 3 }$ </td></tr><tr><td>MSCRF [9]</td><td>Swin-L</td><td>197M</td><td>0.120</td><td>1.057</td><td>5.068</td><td>0.207</td><td>0.872</td><td>0.949</td><td>0.975</td></tr><tr><td rowspan="3">MSCRF [9]</td><td>ViT-S/16</td><td>29M</td><td>0.116</td><td>0.700</td><td>4.345</td><td>0.160</td><td>0.851</td><td>0.974</td><td>0.995</td></tr><tr><td>ViT-B/16</td><td>86M</td><td>0.111</td><td>0.643</td><td>3.826</td><td>0.144</td><td>0.873</td><td>0.978</td><td>0.995</td></tr><tr><td>ViT-L/16</td><td>300M</td><td>0.110</td><td>0.649</td><td>3.939</td><td>0.147</td><td>0.869</td><td>0.978</td><td>0.995</td></tr><tr><td>RGB-HS</td><td>ViT-B/16</td><td>86M</td><td>0.105</td><td>0.572</td><td>3.548</td><td>0.134</td><td>0.887</td><td>0.981</td><td>0.996</td></tr></table>

TABLE IV

ABLATION STUDY: MAP- AND LATENT-LEVEL ALIGNMENTS. THE RESULTS OF STEREO DEPTH ESTIMATION ARE PROVIDED. ALL MODELS USE PRE-TRAINED DINOV3 VIT-B/16 [11] AS THE ENCODER. IN THIS CASE, RGB-HS IS RUNNING WITHOUT VERIFICATION. THE BEST-PERFORMING NUMBERS ARE IN BOLD.
<table><tr><td rowspan="2">Method</td><td rowspan="2"> ${ \mathcal { L } } _ { \mathrm { m a p } }$ </td><td rowspan="2"> $\mathcal { L } _ { \mathrm { l a t e n t } }$ </td><td colspan="3">Error ↓</td></tr><tr><td>AbsRel</td><td>SqRel</td><td>RMSE</td></tr><tr><td>MSCRF [9]</td><td>-</td><td>-</td><td>0.111</td><td>0.643</td><td>3.826</td></tr><tr><td rowspan="3">RGB-HS</td><td>√</td><td>-</td><td>0.116</td><td>0.629</td><td>3.674</td></tr><tr><td>-</td><td>√</td><td>0.116</td><td>0.631</td><td>3.726</td></tr><tr><td>√</td><td>√</td><td>0.114</td><td>0.607</td><td>3.646</td></tr></table>

TABLE V

ABLATION STUDY: EFFECTS OF VERIFICATION MODULES. THE RESULTS OF STEREO DEPTH ESTIMATION ARE PROVIDED. ALL MODELS USE PRE-TRAINED DINOV3 VIT-B/16 [11] AS THE ENCODER. THE BEST-PERFORMING NUMBERS ARE IN BOLD.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Alignment</td><td rowspan="2">Verification</td><td colspan="3">Error ↓</td></tr><tr><td>AbsRel</td><td> $\mathbf { S q R e l }$ </td><td>RMSE</td></tr><tr><td>MSCRF [9]</td><td>-</td><td>-</td><td>0.111</td><td>0.643</td><td>3.826</td></tr><tr><td rowspan="2">RGB-HS</td><td>√</td><td>-</td><td>0.114</td><td>0.607</td><td>3.646</td></tr><tr><td>√</td><td>√</td><td>0.105</td><td>0.572</td><td>3.548</td></tr></table>

Effects of Verification Module. Here, we evaluate the contribution of the proposed verification modules. As shown in Table V, introducing RGB-image-quality verification yields improvements over the version without verification. The results indicate that the verification mechanism acts as a quality filter, promoting the use of RGB data for alignment and enhancing the reliability of cross-modal supervision.

Model Complexity Comparison. As shown in Table VI, the proposed RGB-HS achieves the best complexity performance in terms of model size and FLOPs. Even though RGB-HS’s depth estimation performance is close to that of RGB-MDE (see Table II), it significantly outperforms RGB-MDE across all complexity metrics.

TABLE VI  
MODEL COMPLEXITY COMPARISON OF MONOCULAR DEPTH ESTIMATION METHODS ON THERMAL IMAGES.
<table><tr><td>Method</td><td>Model size (M) ↓</td><td>Inference time (ms/per image) ↓</td><td>FLOPs (T) ↓</td></tr><tr><td>MSCRF [9]</td><td>284</td><td>31.86</td><td>0.32</td></tr><tr><td>RGB-MDE [16]</td><td>666</td><td>698.22</td><td>0.72</td></tr><tr><td>RGB-HS (Ours)</td><td>249</td><td>44.46</td><td>0.23</td></tr></table>

## V. CONCLUSION

This work presents RGB-HS, a framework for depth estimation on thermal images that leverages hierarchical supervision from RGB-based foundation models. By introducing a frozen RGB teacher encoder and transferring knowledge to a thermal student encoder via alignment, RGB-HS enables the thermal branch to learn both structural precision and semantic abstraction. Furthermore, the proposed verification mechanism adaptively reduces the weight of unreliable RGB supervision by evaluating the confidence of image-level representation, thereby ensuring more accurate alignment. Extensive experiments on the ${ \bf M } { \bf S } ^ { 2 }$ benchmark demonstrate that RGB-HS not only achieves competitive performance but also better exploits the representational capacity of RGB foundation models for thermal perception. Extending similar hierarchical supervision to other tasks, such as thermal-based semantic segmentation, detection, and tracking, could be a promising future research direction that further bridges the gap between visible- and infrared-based perception.

## REFERENCES

[1] Y. Wang, W.-L. Chao, D. Garg, B. Hariharan, M. Campbell, and K. Q. Weinberger, “Pseudo-lidar from visual depth estimation: Bridging the gap in 3d object detection for autonomous driving,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 8445–8453.

[2] S. Chen, J. Hong, X. Liu, J. Li, T. Zhang, D. Wang, and Y. Guan, “A framework for 3d object detection and pose estimation in unstructured environment using single shot detector and refined linemod template matching,” in 2019 24th IEEE International Conference on Emerging Technologies and Factory Automation (ETFA). IEEE, 2019, pp. 499– 504.

[3] D. Wofk, F. Ma, T.-J. Yang, S. Karaman, and V. Sze, “Fastdepth: Fast monocular depth estimation on embedded systems,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 6101–6108.

[4] X. Li, J. Guivant, and S. Khan, “Real-time 3d object proposal generation and classification using limited processing resources,” Robotics and Autonomous Systems, vol. 130, p. 103557, 2020.

[5] S. Yu, Y. Guan, Z. Yang, C. Liu, J. Hu, J. Hong, H. Zhu, and T. Zhang, “Multiseam tracking with a portable robotic welding system in unstructured environments,” The International Journal ofAdvanced Manufacturing Technology, vol. 122, no. 3, pp. 2077–2094, 2022.

[6] D. Eigen, C. Puhrsch, and R. Fergus, “Depth map prediction from a single image using a multi-scale deep network,” Advances in neural information processing systems, vol. 27, 2014.

[7] S. F. Bhat, I. Alhashim, and P. Wonka, “Adabins: Depth estimation using adaptive bins,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 4009–4018.

[8] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth anything: Unleashing the power of large-scale unlabeled data,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 10 371–10 381.

[9] U. Shin, J. Park, and I. S. Kweon, “Deep depth estimation from thermal image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1043–1053.

[10] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.

[11] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose,´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.

[12] P. Han, C. Ye, J. Tong, C. Jiang, J. Hong, L. Fang, and X. Li, “Enhancing features in long-tailed data using large vision model,” in 2025 International Joint Conference on Neural Networks (IJCNN). IEEE, 2025, pp. 1–9.

[13] M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal, P. Bojanowski,´ and A. Joulin, “Emerging properties in self-supervised vision transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 9650–9660.

[14] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[15] R. Fan, W. Zhao, M. Lin, Q. Wang, Y.-J. Liu, and W. Wang, “Generalizable thermal-based depth estimation via pre-trained visual foundation model,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 14 614–14 621.

[16] X. Zuo, N. Ranganathan, C. Lee, G. Gkioxari, and S.-J. Chung, “Monother-depth: Enhancing thermal depth estimation via confidenceaware distillation,” IEEE Robotics and Automation Letters, 2025.

[17] B. Li, C. Shen, Y. Dai, A. Van Den Hengel, and M. He, “Depth and surface normal estimation from monocular images using regression on deep features and hierarchical crfs,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 1119–1127.

[18] W. Yin, Y. Liu, C. Shen, and Y. Yan, “Enforcing geometric constraints of virtual normal for depth prediction,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 5684–5693.

[19] X. Yang, Z. Ma, Z. Ji, and Z. Ren, “Gedepth: Ground embedding for monocular depth estimation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 12 719–12 727.

[20] S. Shao, Z. Pei, W. Chen, X. Wu, and Z. Li, “Nddepth: Normaldistance assisted monocular depth estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7931–7940.

[21] Z. Li, X. Wang, X. Liu, and J. Jiang, “Binsformer: Revisiting adaptive bins for monocular depth estimation,” IEEE Transactions on Image Processing, vol. 33, pp. 3964–3976, 2024.

[22] K. Xian, J. Zhang, O. Wang, L. Mai, Z. Lin, and Z. Cao, “Structureguided ranking loss for single image depth prediction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 611–620.

[23] W. Yuan, X. Gu, Z. Dai, S. Zhu, and P. Tan, “Neural window fullyconnected crfs for monocular depth estimation,” in Proceedings of the

IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 3916–3925.

[24] B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, and K. Schindler, “Repurposing diffusion-based image generators for monocular depth estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 9492– 9502.

[25] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” Advances in Neural Information Processing Systems, vol. 37, pp. 21 875–21 911, 2024.

[26] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[27] N. Kim, Y. Choi, S. Hwang, and I. S. Kweon, “Multispectral transfer network: Unsupervised depth estimation for all-day vision,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.

[28] Y. Lu and G. Lu, “An alternative of lidar in nighttime: Unsupervised depth estimation based on single thermal image,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2021, pp. 3833–3843.

[29] U. Shin, K. Park, B.-U. Lee, K. Lee, and I. S. Kweon, “Self-supervised monocular depth estimation from thermal images via adversarial multi-spectral adaptation,” in Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, 2023, pp. 5798–5807.

[30] A. Hu, A. Li, X. Jin, and D. Zou, “Thermostereort: Thermal stereo matching in real time via knowledge distillation and attention-based refinement,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 3766–3772.

[31] Q. Zhang, S. Zhao, Y. Luo, D. Zhang, N. Huang, and J. Han, “Abmdrnet: Adaptive-weighted bi-directional modality difference reduction network for rgb-t semantic segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 2633–2642.

[32] T. Zhang, H. Guo, Q. Jiao, Q. Zhang, and J. Han, “Efficient rgbt tracking via cross-modality distillation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 5404–5413.

[33] X. Li and J. E. Guivant, “Efficient and accurate object detection with simultaneous classification and tracking under limited computing power,” IEEE Transactions on Intelligent Transportation Systems, vol. 24, no. 6, pp. 5740–5751, 2023.

[34] D. P. Do, T. Kim, J. Na, J. Kim, K. Lee, K. Cho, and W. Hwang, “D3t: Distinctive dual-domain teacher zigzagging across rgb-thermal gap for domain-adaptive object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 23 313–23 322.

[35] K. Zhou, F. Yang, S. Wang, B. Wen, C. Zi, L. Chen, Q. Shen, and X. Cao, “M-specgene: Generalized foundation model for rgbt multispectral vision,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 7861–7872.

[36] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10 012–10 022.

[37] H. Fu, M. Gong, C. Wang, K. Batmanghelich, and D. Tao, “Deep ordinal regression network for monocular depth estimation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 2002–2011.

[38] J. H. Lee, M.-K. Han, D. W. Ko, and I. H. Suh, “From big to small: Multi-scale local planar guidance for monocular depth estimation,” arXiv preprint arXiv:1907.10326, 2019.

[39] S. F. Bhat, R. Birkl, D. Wofk, P. Wonka, and M. Muller, “Zoedepth:¨ Zero-shot transfer by combining relative and metric depth,” arXiv preprint arXiv:2302.12288, 2023.

[40] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019.

[41] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[42] U. Shin and J. Park, “Deep depth estimation from thermal image: Dataset, benchmark, and challenges,” arXiv preprint arXiv:2503.22060, 2025.