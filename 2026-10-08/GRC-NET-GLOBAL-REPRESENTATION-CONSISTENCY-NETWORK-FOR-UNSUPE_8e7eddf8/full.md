# GRC-NET: GLOBAL REPRESENTATION CONSISTENCY NETWORK FOR UNSUPERVISED MULTIMODAL ANOMALY DETECTION

Seyoung Jeong<sup>1</sup>, Jong Pil Yun<sup>2,3</sup>, Sang Jun Lee<sup>1∗</sup>

<sup>1</sup>Jeonbuk National University, Jeonju, Republic of Korea <sup>2</sup>Korea Institute of Industrial Technology (KITECH), Incheon, Republic of Korea <sup>3</sup>Chung-Ang University, Seoul, Republic of Korea

## ABSTRACT

Automated quality inspection is essential for ensuring product reliability in manufacturing. While image-based methods effectively capture appearance-related defects, these methods are limited in detecting structural and geometric anomalies, motivating multimodal approaches incorporating 3D information. However, existing methods mainly rely on local patch-level representations, which often lead to unstable reconstruction errors even in normal regions. To address this limitation, we propose GRC-Net, which integrates a globalattention MLP to enforce global representation consistency across patch embeddings with a stable reconstruction module to improve reconstruction stability. The proposed method captures holistic contextual information through a global token and suppresses reconstruction noise by minimizing discrepancies between original and predicted embeddings. Experiments on MVTec 3D-AD and Eyecandies demonstrate that GRC-Net consistently outperforms existing methods at both image and pixel levels. Qualitative results further demonstrate reduced reconstruction errors in normal regions and more distinct reconstruction differences between normal and anomalous regions.

Index Terms— Computer Vision, Anomaly Detection, Unsupervsied Learning, Multimodal

## 1. INTRODUCTION

In manufacturing, surface quality inspection is essential for product reliability. However, manual inspection is often inconsistent and inefficient, while diverse defect types and high labeling costs make large-scale defect data collection difficult. Therefore, unsupervised anomaly detection using only normal samples has been widely studied, with RGB-based methods serving as a practical and cost-effective solution for industrial inspection. [1–3] However, RGB-based methods rely on surface color and texture information, limiting their ability to detect geometric anomalies such as subtle object deformations [4]. In addition, lighting variations, shadows, and reflections often alter pixel values and cause false detections in normal products. Accordingly, multimodal anomaly detection methods incorporating 3D information have gained increasing attention for detecting geometric anomalies and improving robustness to environmental variations [5].

Recent studies have proposed various multimodal anomaly detection approaches [6–8]. M3DM [6] stores modalityspecific features in memory banks and compares them with test features for anomaly detection. However, this leads to high memory consumption in multimodal settings, while misaligned feature spaces often destabilize distance-based comparisons. CFM [9] learns cross-modal feature reconstruction from normal data, but has difficulty reconstructing inherent variations such as surface curvature and material diversity in real-world manufacturing environments.

We propose the global representation consistency network (GRC-Net), a reconstruction-based multimodal anomaly detection method using RGB images and 3D point clouds. GRC-Net incorporates a global attention MLP (GA-MLP) to generate a global token that captures the intrinsic variability of normal samples through global representation consistency across patch embeddings. A stable reconstruction module (SRM) further improves reconstruction stability and enlarges the reconstruction gap between normal and anomalous regions. We evaluate GRC-Net on the MVTec 3D-AD [10] and Eyecandies [11] benchmarks. Experimental results demonstrate that the proposed method outperforms existing multimodal anomaly detection methods, particularly in imagelevel AUROC. The main contributions of this study are as follows:

• We propose a GA-MLP that generates a global token from feature embeddings of normal data and enforces global representation consistency to predict features of the counterpart modality.

• We propose a SRM that suppresses noise in anomaly maps of normal data and enhances the reconstruction gap between normal and anomalous regions.

• We conduct extensive experiments on MVTec 3D-AD and Eyecandies, two representative benchmarks for unsupervised multimodal anomaly detection. Our method achieves strong performance on both datasets.

![](images/5f586a2392ad4ac4d5a95156818e930ebe8d5a6644538b87cc64b48ea4c11e19.jpg)  
Fig. 1. Overview of the proposed GRC-Net architecture. The framework incorporates a GA-MLP to capture global contextual relationships and a SRM to suppress unstable reconstruction errors.

## 2. METHODOLOGY

We propose GRC-Net, a multimodal anomaly detection framework designed to improve reconstruction stability and clarify the reconstruction gap between normal and anomalous region. The overall architecture is shown in Figure 1. Feature embeddings are first extracted using pretrained modalityspecific encoders, and global tokens are generated to enforce global representation consistency across patch embeddings. Reconstruction is then performed using globally informed embeddings, while the SRM suppresses reconstruction noise and promotes accurate reconstruction of normal data. During inference, modality-specific anomaly maps are obtained from reconstruction errors and combined through pixel-wise multiplication to produce the final anomaly map.

## 2.1. Multimodal feature extraction and alignment

We extract and align features from RGB images and point clouds using pretrained encoders. Given an RGB image $I _ { i }$ of size $2 2 4 \times 2 2 4 \times 3 .$ , a DINO-pretrained ViT-B/8 [12, 13] on ImageNet [14] extracts a $2 8 \times 2 8 \times 7 6 8$ feature map, which is upsampled by bilinear interpolation to obtain $F _ { \mathrm { r g b } }$ of size $2 2 4 \times 2 2 4 \times 7 6 8$ . For a point cloud $P _ { i }$ of size $N \times 3 ,$ we use Point-MAE [15] pretrained on ShapeNet [16]. The point cloud is divided into 1,024 groups via farthest point sampling, with 32 points per group, and Point-MAE extracts a 1,152-dimensional feature for each representative center point c. Distance-based interpolation is then applied to obtain $P _ { \mathrm { p t } } .$

which consists of N interpolated 1,152-dimensional feature vectors. Using the known pixel correspondence of each 3D point, $P _ { \mathrm { p t } }$ is projected onto the image plane to form $F _ { \mathrm { p t } }$ of size 224 $\times 2 2 4 \times 1 1 5 2$ . Finally, the pixel-wise aligned feature maps $F _ { \mathrm { r g b } }$ and $F _ { \mathrm { p t } }$ are obtained, and the corresponding pixel-level feature vectors frgb and fpt are used as network inputs.

## 2.2. Global attention MLP

In multimodal reconstruction-based anomaly detection, capturing global contextual information is essential for modeling the intrinsic variability of normal samples. To enhance reconstruction stability by promoting global representation consistency, we propose a GA-MLP that injects a global token summarizing the feature distribution of each modality. The architecture of GA-MLP for the processing of $f _ { \mathrm { p t } }$ is presented in Fig. 1. (a), and the same GA-MLP is also applied to the RGB feature representation $f _ { \mathrm { r g b } }$ . Specifically, we generate a global token ${ \pmb { g } } _ { \mathrm { r g b } }$ and $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } _ { \mathrm { p t } } }$ using the modality-specific feature vectors $f _ { \mathrm { r g b } }$ and $f _ { \mathrm { p t } }$ . The equation is defined as follows:

$$
\begin{array} { r l } & { { \pmb { g } } _ { \mathrm { r g b } } = \phi _ { \mathrm { r g b } } \Big ( \mathrm { c o n c a t } \big ( \mathrm { M e a n } ( { \pmb f } _ { \mathrm { r g b } } ) , \mathrm { M a x } ( { \pmb f } _ { \mathrm { r g b } } ) \big ) \Big ) , } \\ & { { \pmb { g } } _ { \mathrm { p t } } = \phi _ { \mathrm { p t } } \Big ( \mathrm { c o n c a t } \big ( \mathrm { M e a n } ( { \pmb f } _ { \mathrm { p t } } ) , \mathrm { M a x } ( { \pmb f } _ { \mathrm { p t } } ) \big ) \Big ) . } \end{array}\tag{1}
$$

Mean pooling and max pooling are applied over all patches of each sample to summarize global context and local saliency, respectively. The resulting vectors are concatenated along the channel dimension and projected back to the original feature dimension to form the final global token.

Subsequently, all patch tokens are used as queries, while the global token is projected into key–value pairs, allowing each patch to align with the global representation and enforce global representation consistency before being fed into the MLP. Specifically, the global embeddings $f _ { g , \mathrm { r g b } }$ and $f _ { g , \mathrm { p t } }$ are defined as follows:

$$
f _ { g , \mathrm { r g b } } = f _ { \mathrm { r g b } } + \mathrm { s o f t m a x } \left( \frac { \phi _ { q } ^ { \mathrm { r g b } } ( f _ { \mathrm { r g b } } ) \phi _ { k } ^ { \mathrm { r g b } } ( g _ { \mathrm { r g b } } ) ^ { \top } } { \sqrt { C _ { \mathrm { r g b } } } } \right) \phi _ { v } ^ { \mathrm { r g b } } ( g _ { \mathrm { r g b } } ) ,\tag{2}
$$

$$
f _ { g , \mathrm { p t } } = f _ { \mathrm { p t } } + \mathrm { s o f t m a x } \left( \frac { \phi _ { q } ^ { \mathrm { p t } } ( f _ { \mathrm { p t } } ) \phi _ { k } ^ { \mathrm { p t } } ( g _ { \mathrm { p t } } ) ^ { \top } } { \sqrt { C _ { \mathrm { p t } } } } \right) \phi _ { v } ^ { \mathrm { p t } } ( g _ { \mathrm { p t } } ) ,\tag{3}
$$

In Eq. $( 3 ) , \phi _ { q } ( \cdot ) , \phi _ { k } ( \cdot )$ , and $\phi _ { v } ( \cdot )$ are implemented as modality-specific learnable linear layers with identical input and output dimensions, mapping features to the query, key, and value spaces, respectively, and $C _ { \mathrm { r g b } }$ and $C _ { \mathrm { p t } }$ denote the channel dimensions of the RGB and point-cloud feature embeddings. The proposed mechanism reduces visual bias and local noise from patch-level embeddings while improving representation consistency by sharing global patterns. The enhanced patch embeddings are then projected into the other modality through a deep nonlinear MLP, enabling more stable and semantically aligned cross-modal mapping while capturing the natural variability of normal samples.

## 2.3. Stable reconstruciton module

Existing reconstruction-based anomaly detection methods assume complete reconstruction of normal data, but this assumption is often violated, resulting high anomaly scores for normal patches. We therefore align patch embeddings across RGB images and point clouds through bidirectional feature mapping while reducing reconstruction errors in normal data.

To effectively learn the bidirectional mapping structure, we jointly employ a cross-modal projection loss for semantic alignment between modalities and a stable reconstruction loss to stabilize reconstruction errors on normal data. The structure of the SRM is presented in Fig. 1. (b). First, we define a cosine-based projection loss that maximizes the directional similarity between predicted features and their corresponding ground-truth features by projecting the patch embeddings $f _ { g , \mathrm { r g b } }$ and $f _ { g , \mathrm { p t } }$ of each modality into the feature space of the other modality. The equation is defined as follows:

$$
L _ { \mathrm { p r o j } } ^ { i } = \left( 1 - \frac { f _ { g , \mathrm { r g b } } ^ { i } \cdot \hat { f } _ { \mathrm { p t } } ^ { i } } { \Vert f _ { g , \mathrm { r g b } } ^ { i } \Vert \Vert \hat { f } _ { \mathrm { p t } } ^ { i } \Vert } \right) + \left( 1 - \frac { f _ { g , \mathrm { p t } } ^ { i } \cdot \hat { f } _ { \mathrm { r g b } } ^ { i } } { \Vert f _ { g , \mathrm { p t } } ^ { i } \Vert \Vert \hat { f } _ { \mathrm { r g b } } ^ { i } \Vert } \right)\tag{4}
$$

In Eq. (4), $\hat { f } _ { \mathrm { r g b } } ^ { i }$ and $\hat { f } _ { \mathrm { p t } } ^ { i }$ denote the predicted features at pixel $i ,$ obtained by projecting the patch embeddings of the opposite modality through the corresponding GA-MLP, where the loss is computed independently for each pixel.

However, to compensate for magnitude differences that cannot be captured by cosine-based alignment, we define a stable reconstruction Loss that directly minimizes the normalized difference between the predicted embedding and the ground-truth embedding. The equation is defined as follows:

$$
d _ { m } ^ { i } = \left\| \frac { f _ { g , m } ^ { i } } { \| f _ { g , m } ^ { i } \| } - \frac { \hat { f } _ { m } ^ { i } } { \| \hat { f } _ { m } ^ { i } \| } \right\| _ { 2 } , \quad m \in \{ \mathrm { r g b , p t } \} .\tag{5}
$$

$$
L _ { s r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Big ( ( d _ { \mathrm { r g b } } ^ { i } ) ^ { 2 } + ( d _ { \mathrm { p t } } ^ { i } ) ^ { 2 } \Big ) .\tag{6}
$$

$$
L = L _ { \mathrm { p r o j } } ^ { \mathrm { r g b } } + L _ { \mathrm { p r o j } } ^ { \mathrm { p t } } + \alpha L _ { \mathrm { s r } } .\tag{7}
$$

In Eq. (7), α denotes the weight of the loss term and is set to 0.2 in our experiments. By jointly optimizing the projection and reconstruction objectives, the proposed loss formulation stabilizes reconstruction quality for normal data and reduces false anomaly responses caused by unstable embeddings.

## 3. EXPERIMENTS

Experimental details. We evaluate the proposed method on MVTec 3D-AD [10] and Eyecandies [11], standard benchmarks for unsupervised multimodal anomaly detection that provide pixel-wise aligned RGB images and point clouds. Performance is evaluated using AUROC and AUPRO. I-AUROC measures image-level anomaly detection, while P-AUROC evaluates pixel-level anomaly localization. AUPRO measures the overlap between predicted and ground-truth anomalous regions, with the threshold set to 0.3. All experiments were conducted on an NVIDIA H100 GPU, and the model was trained for 250 epochs using Adam with a learning rate of 0.001.

## 3.1. Experimental results

We present quantitative and qualitative comparisons with previous methods on the MVTec 3D-AD and Eyecandies datasets. Table 1 compares performance in terms of I-AUROC, P-AUROC, and AUPRO. Overall, GRC-Net achieves strong performance across both datasets, with particularly high results in image-level and pixel-level AUROC. These results demonstrate that the GA-MLP and SRM effectively suppress spurious anomaly responses in normal regions and improve distinction between normal and anomalous areas.

Table 1. Comparison of quantitative results on the MVTec 3D-AD and Eyecandies datasets. The highest scores are shown in bold, and the second-highest scores are underlined. “–” denotes results not reported in the original paper.
<table><tr><td rowspan="2">Method</td><td colspan="3">MVTec 3D-AD</td><td colspan="3">Eyecandies</td></tr><tr><td>I-AUROC</td><td>P-AUROC</td><td>AUPRO</td><td>I-AUROC</td><td>P-AUROC</td><td>AUPRO</td></tr><tr><td>AST [17]</td><td>0.937</td><td>0.976</td><td>0.944</td><td>0.780</td><td>0.902</td><td>0.744</td></tr><tr><td>M3DM [6]</td><td>0.945</td><td>0.992</td><td>0.964</td><td>0.897</td><td>0.977</td><td>0.882</td></tr><tr><td>EasyNet [18]</td><td>0.926</td><td></td><td>0.821</td><td>0.869</td><td></td><td></td></tr><tr><td>CFM [9]</td><td>0.954</td><td>0.993</td><td>0.971</td><td>0.881</td><td>0.974</td><td>0.887</td></tr><tr><td>2M3DF [19]</td><td>0.966</td><td></td><td>0.975</td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>0.972</td><td>0.998</td><td>0.976</td><td>0.918</td><td>0.982</td><td>0.888</td></tr></table>

![](images/15cd0183eaf93e0317d4ab5b92cc79956c0c4036a9cf8a47e889ba5ef6a028f0.jpg)  
Fig. 2. Qualitative comparison of anomaly localization results on the MVTec 3D-AD dataset.

Qualitative comparisons of anomaly maps on MVTec 3D-AD are presented in Fig. 2. Existing methods often generate noisy responses on irregular or curved surfaces, such as cable gland, peach, and tire. In contrast, the proposed method suppresses high anomaly scores in normal regions and more clearly localizes true anomalies. In particular, the stable reconstruction loss reduces noisy responses caused by illumination and surface variations, resulting in more refined anomaly maps. Fig. 3 presents qualitative comparisons of anomaly maps on the Eyecandies dataset. M3DM assigns high anomaly scores to large normal regions, while CFM reduces some noise but still shows false responses and blurred boundaries. In contrast, the proposed method suppresses noise in normal regions and generates anomaly maps that better align with the ground truth. These results demonstrate that the GA-MLP and SRM improve global representation consistency and reconstruction stability.

## 3.2. Ablation study

We conduct ablation studies to evaluate the contributions of GA-MLP and SRM. Table 2 reports I-AUROC, P-AUROC, and AUPRO on MVTec 3D-AD and Eyecandies. Applying GA-MLP or SRM individually improves performance over the baseline, while their combination achieves the best results on both datasets. These results confirm that GA-MLP enhances global representation consistency and SRM improves reconstruction stability and anomaly localization. We also analyze the effect of the SRM loss coefficient α. As shown in Table 3, the best performance on both datasets is obtained at α = 0.2. Smaller values provide insufficient regularization, while larger values overly emphasize reconstruction stability and reduce anomaly sensitivity. Therefore, α is fixed at 0.2 for all experiments.

![](images/1b9e95af2fb70bc706b7ef69856ac216148e482c67cc87b6bd535d439aba8f13.jpg)  
Fig. 3. Qualitative comparison of anomaly localization results on the Eyecandies dataset.

Table 2. Ablation study on the effectiveness of GA-MLP and SRM.
<table><tr><td rowspan="2">GA-MLP</td><td rowspan="2">SRM</td><td colspan="3">MVTec 3D-AD</td><td colspan="2">Eyecandies</td></tr><tr><td>I-AUROC</td><td>P-AUROC</td><td>AUPRO</td><td>I-AUROC P-AUROC</td><td>AUPRO</td></tr><tr><td rowspan="3">√</td><td></td><td>0.954</td><td>0.993</td><td>0.971</td><td>0.881 0.974</td><td>0.887</td></tr><tr><td></td><td>0.965</td><td>0.995</td><td>0.974</td><td>0.909 0.980</td><td>0.888</td></tr><tr><td>√</td><td>0.967</td><td>0.995</td><td>0.976</td><td>0.917 0.981</td><td>0.888</td></tr><tr><td>√</td><td>√</td><td>0.972</td><td>0.998</td><td>0.976</td><td>0.918 0.982</td><td>0.888</td></tr></table>

Table 3. Ablation study on the effect of the SR loss weight α in the SRM.
<table><tr><td rowspan="2">α</td><td colspan="3">MVTec 3D-AD</td><td colspan="3">Eyecandies</td></tr><tr><td>I-AUROC</td><td>P-AUROC</td><td>AUPRO</td><td>I-AUROC</td><td>P-AUROC</td><td>AUPRO</td></tr><tr><td>0.1</td><td>0.968</td><td>0.995</td><td>0.975</td><td>0.915</td><td>0.981</td><td>0.883</td></tr><tr><td>0.2</td><td>0.972</td><td>0.998</td><td>0.976</td><td>0.918</td><td>0.982</td><td>0.888</td></tr><tr><td>0.3</td><td>0.964</td><td>0.995</td><td>0.974</td><td>0.917</td><td>0.980</td><td>0.881</td></tr><tr><td>0.4</td><td>0.966</td><td>0.995</td><td>0.975</td><td>0.916</td><td>0.980</td><td>0.883</td></tr></table>

## Conclusion

In this paper, we propose GRC-Net, a multimodal anomaly detection framework using RGB images and point clouds. GRC-Net integrates GA-MLP to enforce global representation consistency and SRM to improve reconstruction stability, reducing reconstruction noise in normal regions and enhancing the reconstruction gap between normal and anomalous regions. Experiments on MVTec 3D-AD and Eyecandies demonstrate improved image-level and pixel-level anomaly detection performance over existing methods, while qualitative results confirm more accurate anomaly localization.

## 4. REFERENCES

[1] Tran Dinh Tien, Anh Tuan Nguyen, Nguyen Hoang Tran, Ta Duc Huy, Soan Duong, Chanh D Tr Nguyen, and Steven QH Truong, “Revisiting reverse distillation for anomaly detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 24511–24520.

[2] Jiawei Yu, Ye Zheng, Xiang Wang, Wei Li, Yushuang Wu, Rui Zhao, and Liwei Wu, “Fastflow: Unsupervised anomaly detection and localization via 2d normalizing flows,” arXiv preprint arXiv:2111.07677, 2021.

[3] Jiarui Lei, Xiaobo Hu, Yue Wang, and Dong Liu, “Pyramidflow: High-resolution defect contrastive localization using pyramid normalizing flow,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 14143–14152.

[4] Eliahu Horwitz and Yedid Hoshen, “Back to the feature: classical 3d features are (almost) all you need for 3d anomaly detection,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 2968–2977.

[5] Paul Bergmann and David Sattlegger, “Anomaly detection in 3d point clouds using deep geometric descriptors,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 2613–2623.

[6] Chengjie Wang, Haokun Zhu, Jinlong Peng, Yue Wang, Ran Yi, Yunsheng Wu, Lizhuang Ma, and Jiangning Zhang, “M3dm-nr: Rgb-3d noisy-resistant industrial anomaly detection via multimodal denoising,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[7] Vitjan Zavrtanik, Matej Kristan, and Danijel Skocaj,ˇ “Cheating depth: Enhancing 3d surface anomaly detection via depth simulation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 2164–2172.

[8] Vitjan Zavrtanik, Matej Kristan, and Danijel Skocaj,ˇ “Keep dræming: discriminative 3d anomaly detection through anomaly simulation,” Pattern Recognition Letters, vol. 181, pp. 113–119, 2024.

[9] Alex Costanzino, Pierluigi Zama Ramirez, Giuseppe Lisanti, and Luigi Di Stefano, “Multimodal industrial anomaly detection by crossmodal feature mapping,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 17234– 17243.

[10] Paul Bergmann, Xin Jin, David Sattlegger, and Carsten Steger, “The mvtec 3d-ad dataset for unsupervised 3d anomaly detection and localization,” arXiv preprint arXiv:2112.09045, 2021.

[11] Luca Bonfiglioli, Marco Toschi, Davide Silvestri, Nicola Fioraio, and Daniele De Gregorio, “The eyecandies dataset for unsupervised multimodal anomaly detection and localization,” in Proceedings of the Asian Conference on Computer Vision, 2022, pp. 3586–3602.

[12] Alexey Dosovitskiy, “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[13] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve´ Jegou, Julien Mairal, Piotr Bojanowski, and Armand´ Joulin, “Emerging properties in self-supervised vision transformers,” in Proceedings ofthe IEEE/CVF international conference on computer vision, 2021, pp. 9650– 9660.

[14] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision andpattern recognition. Ieee, 2009, pp. 248–255.

[15] Yatian Pang, Eng Hock Francis Tay, Li Yuan, and Zhenghua Chen, “Masked autoencoders for 3d point cloud self-supervised learning,” World Scientific Annual Review of Artificial Intelligence, vol. 1, pp. 2440001, 2023.

[16] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al., “Shapenet: An information-rich 3d model repository,” arXiv preprint arXiv:1512.03012, 2015.

[17] Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn, and Bastian Wandt, “Asymmetric student-teacher networks for industrial anomaly detection,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2023, pp. 2592–2602.

[18] Ruitao Chen, Guoyang Xie, Jiaqi Liu, Jinbao Wang, Ziqi Luo, Jinfan Wang, and Feng Zheng, “Easynet: An easy network for 3d industrial anomaly detection,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 7038–7046.

[19] Mujtaba Asad, Waqar Azeem, He Jiang, Hafiz Tayyab Mustafa, Jie Yang, and Wei Liu, “2m3df: Advancing 3d industrial defect detection with multi perspective multimodal fusion network,” IEEE Transactions on Circuits and Systemsfor Video Technology, 2025.