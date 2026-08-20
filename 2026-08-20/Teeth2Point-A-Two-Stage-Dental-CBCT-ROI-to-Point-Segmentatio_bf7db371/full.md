# Teeth2Point: A Two-Stage Dental CBCT ROI-to-Point Segmentation Framework

Qi Ma<sup>1,2[0009−0005−4028−6917]</sup>, Shipra Jain<sup>2[0009−0008−1833−3416]</sup>, Niko Benjamin Huber<sup>2[0009−0007−2089−0150]</sup>, and Ender Konukoglu<sup>1[0000−0002−2542−3611]</sup>

Computer Vison Lab, ETH Zurich, Switzerland {fname.lname}@vision.ee.ethz.ch

2 Align Technology {sjain, nhuber}@aligntech.com

Abstract. Modern deep learning architectures have demonstrated strong performance in dental CBCT segmentation. One remaining crucial challenge is accurate tooth labeling in cases with missing or malpositioned teeth, which are highly relevant for dental practice. Transformer-based architectures should in theory be able to resolve such ambiguities using global anatomical context. However, due to the high resolution of CBCT volumes and the wide spatial distribution of teeth within volumes, dense patch-based volumetric processing faces an inherent trade-of. Computational costs limit the number of patches that can be used in self-attention and thus, one can either increase the extent of the context captured in self-attention or capture fine-grained structural details by using small patches, but not both. In this work, we present Teeth2Point, an eficient point-based transformer framework for dental CBCT semantic segmentation that can avoid this trade-of. Teeth2Point first localizes volumetric regions of interest (ROIs) surrounding teeth using a convolutional model, then converts ROIs into point tokens using adaptive sampling. A transformer model predicts accurate segmentations using the point tokens, which allow capturing global context while retaining high resolution. The transformer is first pretrained using self-supervised learning (SSL), in the style of DINO but using domain-specific augmentation strategies, followed by supervised finetuning. The SSL pretraining, which includes random token masking, provides robustness to complex anatomical variations. Compared with the strongest two-stage baseline, Teeth2Point improves abnormal-case performance by 1.44 DSC points on average across four datasets; relative to the first-stage nnU-Net, the gain is 1.9 points.

Keywords: 3D Medical Image Analysis · CBCT Semantic Segmentation · Point Cloud Segmentation.

## 1 Introduction

Volumetric Cone Beam Computed Tomography (CBCT) is widely used in dental practice and segmentation of such images is an important step for quantitative analysis and treatment planning. Advanced deep learning models have recently shown promising performance in this segmentation task. Despite the advances, consistent instance-level tooth labeling in dificult but relevant cases, particularly when teeth are missing or malpositioned, remains challenging. In such structural variations, using global anatomical context based on a large field of view, including bilateral symmetry and inter-arch relationships across the dentition, can help resolve ambiguities and improve robustness. However, using large context in volumetric segmentation with today’s algorithms is challenging. Enlarging the context through downsampling sacrifices fine surface details required for accurate boundary segmentation whereas increasing the patch size can improve performance [1, 14] but incurs substantial computational cost for dense 3D volumetric processing. To improve scalability, prior work has explored hierarchical windowed attention [9] and more recently Mamba-based state-space models [18]. Yet, processing large volumes remains expensive. This challenge is particularly evident in dental CBCT, where teeth occupy only a small fraction of the volume (as low as 0.5%, see Tab. 1) while being widely dispersed across the scan.

Two-stage methods [5,12,14,16] address this sparsity by first extracting ROIs and then performing fine-grained segmentation in the ROIs. However, these approaches retain a dense volumetric representation, where background voxels remain dominant when foreground structures are spatially sparse, as in dental CBCT. In contrast, point-based representations focus on foreground structures and ofer a promising alternative for balancing resolution and global context without excessive computation. However, existing volume-to-point methods [10, 12] primarily rely on local feature modeling, limiting their ability to capture long-range anatomical relationships critical for instance-level tooth segmentation. Moreover, they do not leverage self-supervised pretraining to improve robustness in challenging regions with limited annotations.

To overcome these limitations, we propose Teeth2Point, a two-stage ROIto-point segmentation framework that adopts transformer-based architecture with DINOv2-style self-supervised pretraining. During training, we first learn a foreground extraction model (e.g., nnU-Net [14]) to compute the tooth ROI, which is expanded via dilation and converted into a point-based representation. The unlabeled data are used for self-supervised pretraining, followed by supervised fine-tuning separately on each benchmark dataset. In addition to CBCT-specific volumetric augmentations, we adopt adaptive sampling before tokenization, which allocates more points to high-detail surface regions. We progressively finetune the pretrained transformer, first on ground-truth foreground points and subsequently on predicted ROIs. During inference, the extracted foreground points undergo the same sampling procedure and are forwarded through the point-based network in a single pass to produce point-level predictions, which are subsequently interpolated and mapped back to the volumetric space.

We evaluate Teeth2Point on four dental CBCT datasets, achieving a 1.44 DSC-point gain in missing and malpositioned cases compared with the strongest two-stage baseline. Inference averages 14 seconds per volume, with the second stage running in under one second. Additionally, representations learned through self-supervised training improve label eficiency, achieving a Dice score of 86.7 using only 10 labeled samples.

Table 1: Dataset Characteristics. The Region of Interest (ROI) refers to the cropped volume encompassing all foreground structures. Foreground Ratio (FR) denotes the proportion of foreground voxels in the entire volume, while FR-ROI refers to the foreground ratio within the region of interest. Abnormal Ratio (AR) measures the proportion of cases with missing or malpositioned teeth.
<table><tr><td>Dataset</td><td></td><td>|Sample Mean Shape</td><td>ROI Shape</td><td>FR</td><td>FR-ROI</td><td>AR</td></tr><tr><td>STS2024 [8, 25]</td><td>27</td><td>(584,584,445)</td><td>(253,202,158)</td><td>0.5%</td><td>9.3%</td><td>82.5%</td></tr><tr><td>ToothFairy2 [2, 3, 17]</td><td>480</td><td>(386,348,182)</td><td>(214,169,113)</td><td>1.4%</td><td>8.3%</td><td>80.8%</td></tr><tr><td>ToothFairy3 [2, 3, 17]</td><td>532</td><td>(386,349,181)</td><td>(215,170,112)</td><td>1.3%</td><td>7.7%</td><td>23.1%</td></tr><tr><td>Internal Dataset</td><td>1170</td><td>(472,369,347)</td><td>(286,232,203)</td><td>0.97%</td><td>4.3%</td><td>9.0%</td></tr></table>

## 2 Related work

Volumetric Medical Image Segmentation. CNN-based models such as nnU-Net [14] established strong performance in volumetric CT/MRI segmentation. Transformer-based methods (e.g., Swin UNETR, VideoTeeth) [9, 19] improved global context modeling in 3D volumes, while hybrid architectures like MedNeXt [23] achieved strong benchmark performance. More recently, state-space models such as U-Mamba [18] used long-range context with linear complexity, making them suitable for dense 3D segmentation. Although these approaches show promising results for high-resolution dental CBCT segmentation [2, 8, 20, 24], performance often degrades in abnormal cases, where accurate instance-level labeling requires stronger global anatomical awareness.

Two-Stage Framework and Volume-to-Point method. Two-stage segmentation frameworks typically follow a coarse-to-fine strategy and are widely used in tumor [5] and dental CBCT segmentation [16]. Diferent than other organs, teeth are sparse and widely distributed and even in an ROI(defined as a cropped volumetric bounding box covering them), they constitute less than 10% of the ROI volume (Tab. 1). To exploit this property, we adopt a point-based representation in a second stage as in [10, 12]. However, these approaches employ point-MLP-based networks [13, 22], which are mainly designed for local aggregation and provide limited modeling of global anatomical structure.

Self-Supervised Pretraining for Volumetric Medical Modeling. SSL can reduce annotation requirements in volumetric segmentation. Beyond contrastive [6] and masked reconstruction [11], DINO-style self-distillation [21] learns strong semantic and global representations. However, its application to pointbased representations in medical imaging remains underexplored. Motivated by recent advances shown for natural scenes [27], we adopt a DINOv2-style pretraining framework for dental CBCT point representations. We hypothesize that enforcing invariance to strong augmentations and masking encourages the model to rely on stable anatomical cues (e.g., bilateral symmetry and inter-arch spatial relationships), leading to more robust representations in challenging regions.

![](images/443d790b478076fd245b76f3e3092989ceef0d4b96df727e235da268db3b381c.jpg)  
Fig. 1: Teeth2Point Overview. The first stage uses a convolutional model for ROI extraction, followed by adaptive sampling to obtain a sparse, surfacefocused representation. Point-wise predictions from the point-based transformer network are then mapped back to a volumetric form for evaluation.

![](images/195e7b264b46ae0bf3a1a6f6c5072b486708474bf744c6706943d5aed6b438a1.jpg)  
Fig. 2: Self-distillation pretraining and Gradient-Guided Adaptive Sampling illustration. DINOv2-style pretraining is applied to large-scale unlabeled ROI points (left), while GGAS allocates more samples to high-gradient boundary regions compared to uniform sampling (right).

## 3 Teeth2Point: Two-Stage Segmentation Framework

We illustrate the Teeth2Point method in Fig. 1 and provide details below. CNN Based ROI Segmentation: A CBCT volume is denoted as $V \in$ R<sup>H×W×D×1</sup>. We first train a volumetric segmentation model $v _ { \phi }$ to obtain an initial segmentation: $\mathbf { O } = v _ { \phi } ( V ) \in \mathbb { R } ^ { H \times W \times D \times ( N + 1 ) }$ , which contains logits for N tooth classes and one background class. (Due to memory constraints, we use sliding-window inference and average logits in overlapping regions.) Voxelwise predictions are obtained by $\mathbf { P } = \arg \operatorname* { m a x } ( \mathbf { O } )$ , and the foreground mask is defined as $\textbf { M } = \textbf { P } \neq$ Background, i.e., all voxels that were assigned to a tooth by the CNN model. To obtain the final ROI, we further dilate the mask $\mathbf { M } _ { \mathrm { a u g } } = D i l a t e ( \mathbf { M } )$ using a 3D 6-connected (cross-shaped) kernel, expanding it by one voxel (≈ 0.3mm) to avoid missing structures. The point-based representation X is obtained by collecting all voxels inside the augmented mask. Each point $\mathbf { x } _ { i } \in X$ is associated with voxel coordinate $\mathbf { c } _ { i } \in \mathbb { R } ^ { 3 }$ , a normalized intensity $I _ { i }$ (CBCT intensity clipped to $[ - 1 0 0 0 , 3 5 1 2 ]$ and global z-score normalized), and intensity gradient vector $g _ { i }$ , forming the feature tuple $\mathbf x _ { i } = ( \mathbf c _ { i } , I _ { i } , g _ { i } )$ , and the point label $y _ { i }$ from the corresponding voxel label for supervised fine-tuning.

Table 2: Quantitative Results. We evaluate single- and two-stage methods on STS2024, ToothFairy2/3, and an internal dataset. nnU-Net results (marked with • ) are used for second-stage ROI cropping. We report the Dice similarity coeficient (DSC) for all tooth classes and for molars separately, along with the average number of sliding windows (SW) and time for inference.
<table><tr><td rowspan="2">Method</td><td rowspan="2"> $\mathrm { S W }$ </td><td rowspan="2">Time|</td><td colspan="2">STS2024</td><td colspan="2">ToothFairy2</td><td colspan="2">ToothFairy3</td><td colspan="2">Internal</td><td colspan="2">Average</td></tr><tr><td>(s) |DSC-R DSC-A|DSC-R DSC-A|DSC-R DSC-A|DSC-R DSC-A|DSC-R DSC-A</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Full-Volume CBCT</td><td colspan="5"></td></tr><tr><td>• nnU-Net [14]</td><td></td><td>72</td><td>12.8</td><td>97.93</td><td>93.38</td><td>97.64</td><td>93.03</td><td>97.12</td><td>93.66</td><td>94.01</td><td>87.22</td><td>96.68</td><td>91.82</td></tr><tr><td>Toothseg [20]</td><td></td><td>72</td><td>28.3</td><td>96.76</td><td>93.05</td><td>98.46</td><td>93.22</td><td>98.08</td><td>91.96</td><td>93.93</td><td>87.17</td><td>96.81</td><td>91.35</td></tr><tr><td>MedNeXt [23]</td><td></td><td>72</td><td>6.7</td><td>93.42</td><td>90.03</td><td>96.38</td><td>89.95</td><td>94.19</td><td>88.77</td><td>92.09</td><td>83.61</td><td>94.02</td><td>88.09</td></tr><tr><td>VideoTeeth [19]</td><td></td><td>149</td><td>51.2</td><td>96.37</td><td>94.16</td><td>97.29</td><td>91.50</td><td>94.51</td><td>91.19</td><td>94.12</td><td>85.07</td><td>95.57</td><td>90.48</td></tr><tr><td>U-Mamba1 [18]</td><td></td><td>75</td><td>33.9</td><td>96.87</td><td>94.47</td><td>98.21</td><td>92.17</td><td>95.54</td><td>94.20</td><td>94.22</td><td>86.85</td><td>96.21</td><td>91.92</td></tr><tr><td>U-Mamba2 [24]</td><td></td><td>24</td><td>10.9</td><td>98.85</td><td>94.27</td><td>98.61</td><td>92.33</td><td>96.92</td><td>91.84</td><td>93.68</td><td>86.67</td><td>97.02</td><td>91.28</td></tr><tr><td colspan="10">• ROI-Cropped CBCT</td><td colspan="5"></td></tr><tr><td>nnU-Net [14]</td><td></td><td>18</td><td>8.1</td><td>98.01</td><td>93.65</td><td>98.59</td><td>93.83</td><td>97.89</td><td>94.27</td><td>94.13</td><td>87.27</td><td>97.16</td><td>92.26</td></tr><tr><td>Point-UNet [12]</td><td></td><td>1</td><td>0.7</td><td>97.32</td><td>91.21</td><td>92.02</td><td>86.15</td><td>91.71</td><td>85.87</td><td>90.04</td><td>81.38</td><td>92.77</td><td>86.15</td></tr><tr><td>Teeth2Point</td><td></td><td>1</td><td>0.9</td><td>98.43</td><td>95.01</td><td>98.93 95.17</td><td></td><td>97.84</td><td>95.86</td><td>94.08</td><td>88.74</td><td>97.32</td><td>93.70</td></tr></table>

Adaptive Sampling. Processing all foreground points is computationally expensive, and discriminative cues for tooth segmentation are primarily located near structural boundaries. We therefore propose Gradient-Guided Adaptive Sampling (GGAS). Unlike uniform sampling, GGAS adapts sampling according to the local gradient distribution (Fig. 2, right). To this end, the gradient magnitudes of ROI points, $\left| g _ { i } \right|$ , are computed and normalized across all points via min-max normalization. We then create a coarser grid where each cell contains $\epsilon ^ { 3 }$ voxels. For each cell, a normalized gradient magnitude histogram is computed from ROI points that fall in the cell (Fig. 2c). Histograms are constructed using fixed-width bins (bin width α). In each cell, one point is sampled from each nonempty histogram bin with a probability proportional to $| g _ { i } |$ . This produces X<sup>ˆ</sup> with more sampled points from cells with higher gradient magnitude variation and only one sample from cells with homogeneous gradient magnitude.

Architecture: A linear embedding layer is used to convert point attributes into point tokens, followed by an encoder–decoder backbone [26, 27], which utilizes spatial pooling (max feature pooling within each voxel) and unpooling (nearestneighbor interpolation) with skip connections for multi-level feature integration.

Self-Distillation Pretraining: Self-distillation [21] learns augmentation invariant representations by training a student network $\theta _ { s }$ to match an EMA teacher $\theta _ { t }$ [7]. Given the point sets $X ,$ we construct two augmented views: a local view $X _ { l }$ and a global view $X _ { g } ,$ using domain-specific intensity and spatial transformations described further below. In addition, we apply random rotations and elastic transformations to the points, separately for local and global views. In case where adaptive sampling is not used, additional spherical neighborhood cropping is applied by randomly selecting a center point and sampling neighboring points according to a predefined ratio (local views: 0.1–0.4 and global views: 0.4–1.0). As illustrated in fig. 2, $\theta _ { t }$ encodes $X _ { g }$ into features $F _ { g }$ , while $\theta _ { s }$ encodes $X _ { l }$ into $F _ { l }$ . Instead of class tokens as in $\left[ 2 1 \right]$ , we utilize point-level distillation. Each point feature in $F _ { l }$ is associated with its nearest neighbor in $F _ { g } \mathbf { ; }$ ; these paired features are converted into probabilities, and $\theta _ { s }$ is optimized by minimizing the KL-divergence at point-level. Following [21], we stabilize training with Sinkhorn-Knopp centering and clustering-based assignment normalization [4], which help avoid representation collapse and encourage discriminative embeddings. Moreover, we construct a masked global view $X _ { g } ^ { m }$ by randomly masking points with a training-scheduled ratio r [28]. We insert a learnable mask token at the masked locations and $\theta _ { s }$ embeds the masked points to $F _ { g } ^ { m }$ In addition to the point-level distillation, $\theta _ { s }$ is also trained to minimize the difference between $F _ { g }$ and $F _ { g } ^ { m }$ , using the same objective as the distillation. This masked prediction task encourages the model to capture richer contextual geometry and learn robust point representations. Volumetric Medical Domain Augmentation: We adapt commonly used volumetric CBCT augmentations to point attributes $( \mathbf { c } _ { i } , I _ { i } , g _ { i } )$ during training. Intensity transformations (e.g., gamma correction and brightness scaling) perturb per-point intensities. We further introduce two dental-specific strategies: Symmetry Augmentation, which flips points with label remapping to exploit bilateral tooth structure, and Artifact Augmentation, which simulates metal-induced streak artifacts common in clinical CBCT by randomly selecting a tooth class and amplifying the intensities of its associated points.

Progressive finetuning: A two-step strategy is employed for supervised finetuning, where both stages optimize a combined Dice and cross-entropy loss with equal weighting. In the first (GT-ROI) stage, labels contain only foreground points. In the second adaptation stage, the input is derived from predicted ROIs that may be inaccurate; consequently, the corresponding point labels, inherited from voxel annotations, include background and incomplete structures.

Voxelization: The model produces point-wise predictions on the sampled points. Predictions are propagated to unsampled points using K-nearest neighbors interpolation. The point-wise predictions are then voxelized into a volumetric representation and combined with background labels from the first-stage results, i.e., where $M _ { a u g } ( \mathbf { p } ) = 0$

## 4 Experiments

Implementation Details. All volumes have an isotropic spacing of 0.3 mm. We set ϵ = 2 for coordinate discretization, α = 0.125 for gradient histogram binning, and use K = 4 for prediction voxelization. For pretraining, we use 3,224 internal unlabeled CBCT scans with a batch size of 32 and a cosine schedule that increases the mask ratio from 0.3 to 0.7, trained for 200 epochs on 8 NVIDIA A6000 GPUs. Fine-tuning is performed with a batch size of 8 for 400 epochs GT-ROI training, followed by 100 adaptation epochs on a single A6000 GPU, conducted separately for each benchmark dataset. All supervised comparisons use the same 70/30 train-test split within each dataset and small-component removal post-processing [15]. Because this short workshop paper reports one split, the gains should be interpreted as descriptive rather than inferential.

Main Results: We compare Teeth2Point against full-volume and ROI-cropped nnU-Net baselines [14] and other single- and two-stage baselines (Tab. 2). These include ToothSeg and MedNeXt [23] (CNN-based), VideoTeeth [19] (transformerbased), and U-Mamba 1&2 [18, 24] (state-space model). We report DSC-R and DSC-A separately for regular and abnormal cases, where all baselines show a clear performance drop, with scores 5.2% below the overall average. The ROIcropped nnU-Net improves over the full-volume nnU-Net, potentially because it focuses more on the foreground region, whereas Point-UNet performs comparatively worse, as RandLA-Net [13] primarily emphasizes local feature aggregation. Teeth2Point achieves the best average DSC-A and improves upon the strongest ROI-cropped baseline by 1.44 DSC points. It also improves upon the first-stage nnU-Net by 0.6 points in DSC-R and 1.9 points in DSC-A, reducing the regular-abnormal performance gap by approximately 26%. As shown in Fig. 3, the pretrained representations capture pronounced left-right symmetry (a) and upper-lower structural relationships (b). The final encoder attention maps further show that molar-region tokens attend to anatomically corresponding counterparts across horizontal and vertical directions (c-d). These results support the narrower interpretation that Teeth2Point can access global anatomical context, while causal attribution of the gain to remote context requires additional controlled experiments. The point stage is eficient, requiring one pass and 0.9 seconds, while the full pipeline requires approximately 13.7 seconds including first-stage ROI extraction. We were unable to submit to the Grand Challenge leaderboard as the provided NVIDIA T4 GPUs lack FlashAttention-2 support required to meet the memory demands of large-scale token attention.

![](images/275aaee0fc87b71197c20340ba54e1207c6ed8d19973e79888e1b26fa0e7b990.jpg)  
(a)

![](images/48578c874d31dc93bc71884e53727fe75533066bb40467c0b6fb8cce8e58dcd1.jpg)  
(b)

![](images/72979c872fb384d9647cdb032c5287934f82ed3bce10f713ae13351ea06b14b7.jpg)  
(c)

![](images/474b83607e1927c0b34d9c6dbfe8f89e4c1c01fbf260d3d5ad6619a2e758aa2c.jpg)  
(d)  
Fig. 3: Visualization of pretrained representations and attention weights. We show PCA projections of feature dimensions 1–3 (a) and 4–6 (b), along with query–key attention scores in (c) and (d).

![](images/a4c33a1a7bd2f3202ab2283a86e37f6f6d0634403a055f0e7fd7d189613712a3.jpg)  
Fig. 4: Qualitative Results. Teeth2Point shows strong performance among two-stage methods (numbers denote Average DSC / Molar DSC).

![](images/ac6e54cd8172437fae5c043fae528c0659333dadcea109a4f94b711f9c5d5ec3.jpg)

![](images/e5fb5a907a6d019b9f1c0d648955e32937299f69df446b3a8ca4d1c0479c0f21.jpg)  
Fig. 5: Experiments of Ablation (left) and Label Eficiency (right).

Ablation. Teeth2Point was initially trained on point clouds extracted from GT ROIs. When evaluated on points from predicted ROIs, the DSC decreased by over 5.6% on the internal dataset (Fig. 5 left). Fine-tuning on imperfect predictions with adapted CBCT augmentation and gradient-guided sampling largely recovered the performance loss and notably improved results in the molar region. In addition, the pretrained model further contributed to the overall performance. Label eficiency. We report the efectiveness of the pretrained model in Fig. 5 (right). The pretrained model consistently outperforms training from scratch under the same subset protocol. Notably, with only 10 labeled samples for finetuning, it achieves an average DSC of 86.7%, which is comparable to training from scratch with 100 samples. This suggests reduced annotation needs, although repeated subset draws would be required to quantify uncertainty.

## 5 Conclusion

We present Teeth2Point, a two-stage framework that converts dense CBCT volumes into point-based representations and processes them with a self-distillation pretrained transformer. Several design components improve robustness and global context modeling. Experiments show improved performance, particularly in abnormal cases. A current limitation is that the framework is not fully end-to-end, as it relies on a separate ROI extraction model in the first stage; if a tooth is entirely missed during ROI extraction, the second stage cannot recover it.

## References

1. Bolelli, F., et al.: Segmenting the Inferior Alveolar Canal in CBCTs Volumes: the ToothFairy Challenge. IEEE Transactions on Medical Imaging pp. 1–17 (2024). https://doi.org/10.1109/TMI.2024.3523096

2. Bolelli, F., et al.: Segmenting Maxillofacial Structures in CBCT Volumes. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1–10. IEEE (2025)

3. Bolelli, F., et al.: Multi-structure segmentation in CBCT volumes: The ToothFairy2 challenge. Medical Image Analysis 112, 104095 (2026). https://doi.org/10.1016/j.media.2026.104095

4. Caron, M., et al.: Unsupervised learning of visual features by contrasting cluster assignments (2021)

5. Casamitjana, A., et al.: Cascaded v-net using roi masks for brain tumor segmentation. In: International MICCAI Brainlesion Workshop. pp. 381–391. Springer (2017)

6. Chen, T., et al.: A simple framework for contrastive learning of visual representations (2020)

7. Chen, X., He, K.: Exploring simple siamese representation learning (2020)

8. Cui, W., et al.: Ctooth+: a large-scale dental cone beam computed tomography dataset and benchmark for tooth volume segmentation. In: MICCAI Workshop on Data Augmentation, Labelling, and Imperfections. pp. 64–73. Springer (2022)

9. Hatamizadeh, A., et al.: Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images (2022)

10. He, J., et al.: Learning Hybrid Representations for Automatic 3D Vessel Centerline Extraction, p. 24–34. Springer International Publishing (2020). https://doi.org/10.1007/978-3-030-59725-2\_3

11. He, K., et al.: Masked autoencoders are scalable vision learners (2021)

12. Ho, N.V., et al.: Point-unet: A context-aware point-based neural network for volumetric segmentation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 644–655. Springer (2021)

13. Hu, Q., et al.: Randla-net: Eficient semantic segmentation of large-scale point clouds (2020)

14. Isensee, F., et al.: nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18(2), 203–211 (2021)

15. Isensee, F., et al.: Scaling nnu-net for cbct segmentation (2024)

16. Lin, X., et al.: Accurate mandibular canal segmentation of dental cbct using a two-stage 3d-unet based segmentation framework. BMC Oral Health 23(1), 551 (2023)

17. Lumetti, L., et al.: Enhancing Patch-Based Learning for the Segmentation of the Mandibular Canal. IEEE Access pp. 1–12 (2024). https://doi.org/10.1109/ACCESS.2024.3408629

18. Ma, J., Li, F., Wang, B.: U-mamba: Enhancing long-range dependency for biomedical image segmentation. arXiv:2401.04722 (2024)

19. Ma, Q., et al.: Video foundation model for medical 3d segmentation. In: Supervised and Semi-supervised Multi-structure Segmentation and Landmark Detection in Dental Data. pp. 72–88. Springer Nature Switzerland (2025)

20. van Nistelrooij, N., et al.: Toothseg: Robust tooth instance segmentation and numbering in cbct using deep learning and self-correction. IEEE Journal of Biomedical and Health Informatics (2025). https://doi.org/10.1109/JBHI.2025.3650444

21. Oquab, M., et al.: Dinov2: Learning robust visual features without supervision (2024)

22. Qi, C.R., et al.: Pointnet++: Deep hierarchical feature learning on point sets in a metric space (2017)

23. Roy, S., et al.: Mednext: Transformer-driven scaling of convnets for medical image segmentation (2024)

24. Tan, Z.Q., et al.: U-mamba2: Scaling state space models for dental anatomy segmentation in cbct. In: Medical Image Computing and Computer Assisted Intervention (MICCAI), Workshop on Oral and Dental Image Analysis (ODIN) (2025)

25. Wang, Y., et al.: Sts miccai 2023 challenge: Grand challenge on 2d and 3d semisupervised tooth segmentation. arXiv:2407.13246 (2024)

26. Wu, X., et al.: Point transformer v3: Simpler, faster, stronger. In: CVPR (2024)

27. Wu, X., et al.: Sonata: Self-supervised learning of reliable point representations. In: CVPR (2025)

28. Zhou, J., et al.: ibot: Image bert pre-training with online tokenizer (2022)