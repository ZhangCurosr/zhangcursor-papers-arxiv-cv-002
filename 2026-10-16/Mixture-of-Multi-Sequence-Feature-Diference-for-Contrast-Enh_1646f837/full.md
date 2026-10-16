# Mixture of Multi-Sequence Feature Diference for Contrast-Enhanced MRI Synthesis

Jonghun Kim , Junmo Kwon , and Hyunjin Park <sup>⋆</sup>

Department of Electrical and Computer Engineering, Sungkyunkwan University, Suwon, Korea {iproj2,skenfn1231,hyunjinp}@skku.edu

Abstract. Contrast-enhanced magnetic resonance imaging (CE-MRI) is crucial for accurately diagnosing and segmenting brain tumors. However, the acquisition of CE-MRI requires injecting contrast agents, which poses potential health risks and increases scan costs. Here, we propose a CE-MRI synthesis method from non-contrast sequences, reducing both risk and cost. Unlike existing methods that rely on channel-wise concatenation to fuse multi-sequence information for synthesis, we introduce a Mixture of Feature Diference (MoFD) module. The proposed module uses a dynamic weighting strategy to adaptively select and aggregate informative features from both individual source sequences and explicit diferences among them. Quantitative and qualitative evaluations demonstrate that our proposed method outperforms baselines in terms of image fidelity and perceptual quality on the UCSF-PDGM and UPenn-GBM datasets. Our method successfully reconstructs tumor regions and achieves segmentation performance comparable to that obtained with real CE-MRI. Our ablation studies show that although adversarial learning improves perceptual quality, a regression-based variant of our method achieves higher structural fidelity. Our code available at github.com/jongdory/MoFD.

Keywords: Contrast-Enhanced MRI Synthesis · Medical Image Translation · Mixture of Feature Diference

## 1 Introduction

Contrast-enhanced magnetic resonance imaging (CE-MRI) is a pivotal tool that uses a contrast agent injected into the bloodstream to enhance image clarity [9,30,31,51]. Although non-contrast MRI is suficient for observing anatomical structures, it often lacks the sensitivity needed to identify vascular conditions and subtle tissue discrepancies. It is dificult to distinguish tumors or inflammation from healthy tissue on non-contrast scans because these conditions often have similar intensities [10]. CE-MRI addresses this issue by using contrast agents that make blood vessels and lesions hyperintense on T1-weighted images. This hyperintensity enables the precise identification of the location and extent of these lesions [38]. As shown in Fig. 1, tumor boundaries are ambiguous in other non-contrast MRI sequences, whereas the contrast-enhanced image clearly delineates the tumor region. Despite these advantages, administering contrast agents raises safety concerns because gadolinium may not be fully excreted, potentially leading to accumulation in the brain and bones [25, 40]. Furthermore, CE-MRI requires longer acquisition times and incurs higher costs.

![](images/ceb0353b2c038ea11652bde685d39884b130a2a21243fb2f13781936cade14fd.jpg)  
Fig. 1: Illustration of non-contrast multi-sequence MRI and contrast-enhanced MRI. Red bounding boxes show zoomed-in views of the tumor region. Unlike the non-contrast sequence images, the T1 contrast-enhanced image clearly delineates tumor boundaries.

To address these issues, recent studies have explored generative model-based medical image translation for synthesizing CE-MRI from non-CE images [27]. In particular, Generative Adversarial Network (GAN) [17] and difusion model [21] based generative models have been widely investigated for synthesizing target modalities by concatenating source modalities along the channel dimension as input [11,13,29,48,53]. Ea-GAN [53] is a 3D Pix2Pix [23]-based model that integrates edge information, efectively preserving structural details. MM-GAN [48] utilizes a 3D GAN framework that combines information from all available MRI sequences to simultaneously synthesize arbitrary missing sequences. Chen et al. [11] have proposed a 3D convolutional network designed using a fusion method that extracts unique information feature maps through independent encoders for each input modality and then combines them in the channel direction to synthesize CE-MRI within the regression framework. ResViT [13] has been introduced as a GAN-based generative model for multi-modal image synthesis that combines Vision Transformers [15] with the convolutional networks to leverage global context. ALDM [29] applies a 3D Latent Difusion Model [46] incorporating switchable SPADE [42] blocks within the latent space, enabling the translation of a single source image into various target images. Recent studies [44] have suggested that, in medical image translation, fidelity is more critical than perceptual quality because generative models for restoration tasks are subject to the perception-distortion tradeof [4]. Accordingly, regression losses can be more important than adversarial learning, which primarily aims to improve perceptual quality [44].

Most approaches concatenate multi-sequence images along the channel dimension for use as input. This often overlooks the distinct contrast characteristics of each sequence. For example, T1-weighted images depict fat as hyperintense, while water, cerebrospinal fluid (CSF), edema, and inflammation appear dark [16, 43]. T2-weighted images depict water, lesions, and fat as hyperintense, while air and blood appear hypointense [3,43]. T2-FLAIR images also highlight lesions and fat, but water and CSF appear dark [14]. Furthermore, the contrast enhancement of pathologies such as inflammation and edema varies significantly across these sequences, serving as an important biomarker [12]. These abnormal regions can be explicitly modeled using diference maps between sequences, making them useful for accurate contrast-enhanced image synthesis. Therefore, to accurately synthesize contrast-enhanced regions, it is crucial to explicitly incorporate the diverse contrast variations of distinct sequences and their explicit diferences rather than relying on channel-wise concatenation. Because channelwise concatenation captures pixel-level relationships between source modalities, it can be sensitive to intensity scaling and noise. In contrast, modeling diferences in a feature space can provide an enhanced contextual representation of inter-sequence contrast changes.

We propose the Mixture of Feature Diference (MoFD) module that efectively leverages the complementary information from multiple sequences for CE-MRI synthesis. Unlike conventional methods that concatenate sequences across channels, our approach uses an encoder to extract sequence-specific features from each input sequence capturing unique contrast characteristics. We further calculate the feature diferences between sequences and dynamically allocate importance to sequence-specific features and their feature diferences, enabling flexible and selective feature integration instead of fixed averaging. We validate our method with regression-based and GAN-based approaches to investigate the impact of adversarial training on improving perceptual quality. We also evaluate tumor segmentation performance on the synthesized images using nnU-Net [22] to assess synthesis quality in tumor regions.

## Our Contributions:

• We propose a method that explicitly leverages feature diferences for multisequence MRI, enabling the model to capture distinct contrast variations across MRI sequences.

• We introduce the MoFD module to prioritize informative features via adaptive and selective integration.

• We analyze the impact of regression-based and GAN-based objectives on CE-MRI synthesis.

• We extensively validate our method on UCSF-PDGM and UPENN-GBM datasets for CE-MRI synthesis.

## 2 Related Work

## 2.1 Medical Image Translation for CE-MRI Synthesis

Gadolinium-based contrast agents improve lesion visibility but raise safety concerns, motivating recent studies on CE-MRI synthesis from non-contrast sequences. However, most existing methods still rely on channel-wise concatenation of input modalities, which can overlook the distinct contrast characteristics of each sequence. Bone et al. [5] synthesized full-dose T1 contrast-enhanced (T1ce) brain MRI from either non-contrast images or low-dose contrast-enhanced images to reduce concerns related to gadolinium retention and nephrotoxicity [34]. Their study considered a clinical scenario in which T1, T2-FLAIR, and apparent difusion coeficient (ADC) images were acquired before contrast injection, followed by a 25% low-dose T1ce scan and an additional 75% injection for the full-dose T1ce reference. The source modalities were concatenated along the channel dimension and used as input. Chen et al. [11] proposed a 3D fully convolutional network to synthesize CE-MRI from T1, T2, and ADC images. Although regression-based methods can preserve anatomical structures, they often produce over-smoothed images and struggle to synthesize fine details in localized enhancing lesions. To improve perceptual quality, Jiao et al. [24] proposed a GAN-based model for synthesizing abdominal contrast-enhanced T1ce images from non-contrast T1 and T2 images, using separate encoders and a sparse attention fusion module to combine modality-specific features. Li et al. [35] demonstrated that virtual contrast-enhanced MRI can precisely delineate tumors for nasopharyngeal carcinoma radiotherapy, eliminating the risk of gadoliniumbased contrast agent. Bottani et al. [6] compared three conditional GAN-based generators for translating non-contrast MRI into CE-MRI, each built on a 3D U-Net backbone with residual connections, attention gates, or transformer layers. Most existing methods rely on GAN-based frameworks. However, due to the perception-distortion trade-of [4], adversarial learning tends to only improve perceptual image quality at the cost of reduced fidelity. This trade-of is important in medical image translation, where fidelity is more critical than perceptual quality. Accordingly, regression losses can be more important than adversarial learning, which primarily aims to improve perceptual quality [44].

## 2.2 Feature Fusion and Diference Modeling

Medical image translation has been studied to reduce scan time, cost, and patient risk by generating missing or target modalities from acquired sequences [13, 28, 32]. Early methods used convolutional networks and GANs to translate non-contrast sequences into target images, while later studies introduced more advanced fusion strategies to combine complementary information across modalities. For example, Hi-Net [55] used hybrid feature fusion to capture modalityspecific and cross-modal information. Han et al. [18,19] introduced a task-specific fusion method that adaptively weights each modality according to its contribution to the target image. Rokuss et al. [45] introduced a diference weighting block to explicitly incorporate visual and temporal changes between baseline and follow-up scans into the model architecture. Their method computed featuremap diferences between the two time points and uses them to generate an attention map that emphasizes changed regions. Mixture-of-Experts (MoE) [49] is an efective module for scaling model capacity while keeping the number of activated parameters during inference relatively small, often leading to improved performance. Oh et al. [41] interpreted MoE from a control-theoretic perspective, formulating it as a feedback control mechanism that dynamically assigns the influence of each controller according to the current state to optimize the overall system. Building on these prior studies, we propose a module for CE-MRI synthesis that explicitly computes feature-map diferences across modalities and uses a gating network to dynamically select informative features from multisequence MRI.

![](images/0770341e4da74a78e127a5f20c8deb2cd20ba182b96706b8651e559c99c79ef8.jpg)  
Fig. 2: Illustration of the proposed method for CE-MRI synthesis. Weight-sharing encoders extract features from each input sequence (T1, T2, and T2-FLAIR). The model uses the diferences between these features and employs a softmax function to assign adaptive weights dynamically. These weighted features are then aggregated and fed into the decoder to synthesize the CE-MRI.

## 3 Method

Our model architecture is based on the U-Net [47], as illustrated in Fig. 2. Unlike previous approaches that concatenate multi-sequence images along the channel dimension and process them through a single encoder, our model processes each sequence individually to extract distinct features. This strategy allows for the explicit separation of sequence-specific features, which are then utilized for contrast-enhanced image synthesis. We employ a weight-sharing encoder across all input sequences. This design ensures that diferent input sequences are mapped into a unified latent feature space under a consistent criterion, while maintaining parameter eficiency [36, 37].

## 3.1 Mixture of Feature Diference

We compute both the encoder features and feature diferences and pass their mixture to the decoder. Let ${ \cal { S } } = \{ I _ { 1 } , I _ { 2 } , \dots , I _ { N } \}$ be the set of input sequences, where N is the number of sequences and $I ^ { k }$ denotes the image of the k-th sequence. The encoder extracts multi-scale features from each $\bar { I } ^ { k }$ . To capture the inter-sequence contrast variations, we define the feature diference set $D _ { i }$ at the i-th scale as the collection of pairwise diferences between sequence features:

$$
\mathrm { E n c o d e r } ( I ^ { k } ) = \{ f _ { 1 } ^ { k } , f _ { 2 } ^ { k } , \ldots , f _ { L } ^ { k } \} , \quad D _ { i } = \{ f _ { i } ^ { k } - f _ { i } ^ { l } | 1 \leq k < l \leq N \} ,\tag{1}
$$

where $f _ { i } ^ { k }$ represents the latent feature of the k-th sequence at the i-th scale and L is the total number of feature scales.

Instead of simply averaging the features defined in $\operatorname { E q . }$ 1 and passing them to the decoder, we design a mechanism to dynamically select and prioritize significant features. To this end, we compute importance scores for each feature map and forward a weighted sum of these features to the decoder. Both the individual features of a given sequence and the feature diferences between sequences are first processed through a convolution block with pre-normalization to ensure consistent scaling. We concatenate the feature set $F _ { i }$ and the feature diference set $D _ { i }$ to form the combined representation $C _ { i }$ . This is then processed through a convolution block to yield the refined feature map ${ \tilde { F } } _ { i } .$ , as formulated in:

$$
\begin{array} { r } { C _ { i } = \mathrm { C o n c a t } ( { F } _ { i } , D _ { i } ) , \quad \tilde { F } _ { i } = \mathrm { C o n v } ( C _ { i } ) . } \end{array}\tag{2}
$$

We compute the attention weights $a _ { i }$ by applying Global Average Pooling (GAP) followed by a softmax. The final aggregated feature ${ \bar { f } } _ { i } ,$ which is forwarded to the decoder, is obtained as a weighted sum of these features:

$$
\boldsymbol { A _ { i } } = \mathrm { S o f t m a x } ( \mathrm { G A P } ( \boldsymbol { \tilde { F } _ { i } } ) ) , \quad \boldsymbol { \bar { f } _ { i } } = \sum _ { m = 1 } ^ { M } \boldsymbol { a _ { i } ^ { m } } \cdot \boldsymbol { \tilde { f } _ { i } ^ { m } } ,\tag{3}
$$

where $a _ { i } ^ { m } \in A _ { i }$ denotes the weight assigned to the m-th feature at the i-th scale and M is the total number of elements in the combined set, defined as the sum of the number of elements in $F _ { i }$ and $D _ { i }$ . The aggregated multi-scale features are then passed to the decoder to synthesize the CE-MRI. Let $\bar { F } = \{ \bar { f } _ { 1 } , \bar { f } _ { 2 } , \dots , \bar { f } _ { L } \}$ denote the set of aggregated features across all scales. The prediction $\hat { Y }$ is defined as follows:

$$
{ \hat { Y } } = \operatorname { D e c o d e r } ( { \bar { F } } ) = \operatorname { D e c o d e r } ( { \bar { f } } _ { 1 } , { \bar { f } } _ { 2 } , \dots , { \bar { f } } _ { L } ) .\tag{4}
$$

Rather than merging all input sequences at the image level, the weight-sharing encoder first maps each sequence into a common feature space, enabling the model to extract comparable representations, while preserving the distinctive contrast information of each modality. The MoFD module then exploits both individual encoder features and their pairwise diferences to identify complementary information for contrast-enhanced image synthesis. By adaptively weighting these features at multiple scales, the decoder receives a refined representation that emphasizes relevant cross-modality cues while reducing the influence of redundant or less informative features. This design is particularly efective for multi-sequence MRI synthesis, where the target contrast is determined not by a single input modality but by complex relationships among anatomical structure, tissue contrast, and lesion appearance across sequences.

## 3.2 Structural Alignment

Although diferent MRI sequences exhibit varying contrast, they share the underlying anatomical structure. Therefore, we aim to enforce structural consistency across multiple sequences for a single subject. Inspired by previous work [52], we focus on the structure comparison component of the Structural Similarity Index Measure (SSIM) [50] for the sequence feature $f _ { i } ^ { k }$ . We exclude the luminance and contrast terms from the original SSIM to focus on preserving structural alignment despite intensity variations between sequences. We compute localized statistics using a Gaussian kernel. The alignment loss is applied to all pairwise combinations of sequence features, as shown in $\mathrm { F i g . \ 3 \ ( a ) }$ . The alignment loss for structural similarity is defined as follows:

$$
s ( f _ { i } ^ { k } , f _ { i } ^ { l } ) = \frac { \sigma ( f _ { i } ^ { k } , f _ { i } ^ { l } ) + \epsilon } { \sigma ( f _ { i } ^ { k } ) \sigma ( f _ { i } ^ { l } ) + \epsilon } , ~ \mathcal { L } _ { \mathrm { a l i g n } } = 1 - s ( f _ { i } ^ { k } , f _ { i } ^ { l } ) ,\tag{5}
$$

(a) Structural Alignment Loss  
![](images/4438089cbd13dc41e8c697e52443b325467cfe55fa150b136e75c41b30e2c35c.jpg)

![](images/65bdeae5c74d221b0d922413d5f0640063e8880e0162bab64fa0b48909cb4298.jpg)  
(b) Reconstruction and GAN Loss  
Fig. 3: Illustration of the training objectives. (a) The structural alignment loss encourages structural information in sequence features to remain consistent across input sequences. (b) The reconstruction loss encourages the prediction to match the ground truth, while the optional GAN loss promotes realistic textures in the synthesized image.

where $\sigma ( f _ { i } ^ { k } , f _ { i } ^ { l } )$ denotes the covariance between $f _ { i } ^ { k }$ and $f _ { i } ^ { l } , ~ \sigma ( f _ { i } ^ { k } )$ and $\sigma ( f _ { i } ^ { l } )$ represent their respective standard deviations, and $\epsilon = 1 0 ^ { - 6 }$ is a small constant. The structural alignment loss encourages features from diferent sequences of the same subject to maintain consistent anatomical structure, which helps the model preserve patient-specific structure during synthesis.

## 3.3 Training Strategy

To ensure high-fidelity synthesis of CE-MRI, we optimize the model using a reconstruction loss. This objective encourages the predicted image $\hat { Y }$ to closely match the ground truth $Y :$

$$
\mathcal { L } _ { \mathrm { r e c } } = | | Y - \hat { Y } | | _ { 1 } .\tag{6}
$$

Additionally, we incorporate the structural consistency constraint defined in Eq. 5 to enforce structural coherence across diferent sequences. The total objective function is formulated as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { a l i g n } } .\tag{7}
$$

Although using a GAN loss can improve the perceptual quality of synthesized images, it is subject to the perception-distortion trade-of [4]. In medical imaging, prioritizing perceptual quality can result in hallucinated details or artifacts that do not exist in the ground truth, compromising pixel-level fidelity. Therefore, as shown in Fig. 3 (b), we define the adversarial loss using a PatchGAN [23] discriminator D as follows, treating it as an optional component:

$$
\mathcal { L } _ { \mathrm { g a n } } = \mathbb { E } _ { Y } [ \log \mathcal { D } ( Y ) ] + \mathbb { E } _ { \hat { Y } } [ \log ( 1 - \mathcal { D } ( G ( S ) ) ] ,\tag{8}
$$

where S is set of input sequences. The total objective function with the GAN term is formulated as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { a l i g n } } + \mathcal { L } _ { \mathrm { g a n } } .\tag{9}
$$

These components allow the proposed method to balance structural fidelity and contrast generation. The reconstruction objective promotes voxel-level accuracy, the alignment objective regularizes anatomical consistency, and the optional adversarial objective can be used when improved perceptual realism is desired.

## 4 Experiments

## 4.1 Datasets.

We used two public datasets: UCSF-PDGM [7, 8] and UPenn-GBM [1, 2]. For the experiments, the UCSF-PDGM dataset was divided into 210 subjects for training, 23 for validation, and 54 for testing. The UPenn-GBM dataset was divided into 210 training, 21 validation, and 52 testing subjects. Both datasets consist of multi-sequence MRI, including T1, T2, T2-FLAIR, and T1ce images. In addition, both datasets provide glioblastoma tumor masks annotated into whole tumor (WT), enhancing tumor (ET), and tumor core (TC) regions. All volumes were preprocessed with co-registration to the same anatomical template, skull stripping, and resampling to an isotropic resolution of $1 \times 1 \times 1 ~ \mathrm { m m ^ { 3 } }$

## 4.2 Implementation details.

The model was trained for 500 epochs using the Adam [33] optimizer with a batch size of 1 and a learning rate of 1e-4. For training and inference, input images were cropped to minimize background regions and adjusted to a spatial resolution of $1 9 2 \times 1 9 2 \times 1 2 8$ . Our model was built on a 3D U-Net [47] backbone, with feature channel dimensions set to [32, 64, 128, 256, 512] at each stage. The proposed MoFD module was inserted into the skip connections at all stages, allowing encoder features to be refined before being passed to the corresponding decoder layers.

## 4.3 Metrics.

To evaluate image quality, the 3D PSNR and SSIM metrics were used. Perceptual quality was assessed using the LPIPS [54] metric with a 3D ResNet-50 [20] trained on RadImageNet [39]. For downstream evaluation, we trained an nnU-Net [22] on single-input T1ce images from the training set and evaluated tumor segmentation performance using the Dice score for the WT, ET, and TC regions.

## 4.4 Comparison methods.

We compared our method with representative MRI synthesis models spanning both 3D and 2D architectures. Among the 3D methods, Ea-GAN [53] is based on a 3D pix2pix [23], Chen et al. [11] is a regression-based model and ALDM [29] employs a 3D latent difusion model [46]. In contrast, MM-GAN [48] and ResViT [13] are 2D GAN-based methods that operate on axial images. HiNet [55] and MMgSN-Net [35] are also 2D GAN-based approaches and they incorporate modality fusion modules to integrate information from multiple input modalities.

![](images/4344ee96e6685f7ce4032f74de57aac55bf6f5fe3209a04aa02846bdb13210ff.jpg)  
Fig. 4: Qualitative results of T1ce synthesis with comparison methods on the UCSF-PDGM [7, 8]. The bounding box (bottom left) shows a zoomed-in view of the tumor region from the T1ce. The bottom right image shows the corresponding tumor segmentation results by nnU-Net [22]. Without GAN, the results are blurry. With GAN, however, the perceptual quality improves.

Table 1: Quantitative performance of T1ce synthesis with comparison methods on the UCSF-PDGM [7, 8]. PSNR and SSIM are measured and reported as 3D metrics in mean ± std. Bold indicates the best, underlined denotes the second-best performance. WT: Whole Tumor, ET: Enhancing Tumor, TC: Tumor Core.
<table><tr><td>Dataset</td><td colspan="7">UCSF-PDGM [7,8]</td></tr><tr><td>Tasks</td><td colspan="3">T1ce Synthesis</td><td colspan="4">Tumor Segmentation</td></tr><tr><td>Model</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>WT</td><td>ET</td><td>TC</td><td>Avg.</td></tr><tr><td>Ea-GAN [53]</td><td>23.59 ± 0.99</td><td>0.872 ± 0.029</td><td>0.047 ± 0.022</td><td>0.844 ± 0.069</td><td>0.318 ± 0.257</td><td>0.125 ± 0.217</td><td>0.429 ± 0.161</td></tr><tr><td>MM-GAN [48]</td><td>24.24 ± 1.43</td><td>0.877 ± 0.033</td><td>0.042 ± 0.025</td><td>0.747 ± 0.140</td><td>0.094 ± 0.186</td><td>0.046 ± 0.137</td><td>0.296 ± 0.117</td></tr><tr><td>Chen et al. [11]</td><td>24.54 ± 1.38</td><td>0.880 ± 0.027</td><td>0.025 ± 0.016</td><td>0.852 ± 0.093</td><td>0.711 ± 0.195</td><td>0.532 ± 0.268</td><td>0.698 ± 0.160</td></tr><tr><td>ResViT [13]</td><td>24.46 ± 1.34</td><td>0.871 ± 0.026</td><td>0.025 ± 0.020</td><td>0.852 ± 0.079</td><td>0.741 ± 0.140</td><td>0.551 ± 0.200</td><td>0.714 ± 0.120</td></tr><tr><td>ALDM [29]</td><td>23.65 ± 0.88</td><td>0.848 ± 0.026</td><td>0.025 ± 0.017</td><td>0.803 ± 0.108</td><td>0.679 ± 0.190</td><td>0.443 ± 0.252</td><td>0.642 ± 0.141</td></tr><tr><td>Hi-Net [55]</td><td>24.12 ± 1.45</td><td>0.874 ± 0.036</td><td>0.029 ± 0.017</td><td>0.826 ± 0.102</td><td>0.581 ± 0.257</td><td>0.410 ± 0.277</td><td>0.606 ± 0.193</td></tr><tr><td>MMgSN-Net [35]</td><td>22.59 ± 1.94</td><td>0.858 ± 0.095</td><td>0.049 ± 0.027</td><td>0.842 ± 0.090</td><td>0.662 ± 0.270</td><td>0.519 ± 0.266</td><td>0.674 ± 0.187</td></tr><tr><td>MoFD (w/o GAN)</td><td>25.00 ± 1.26 0.894 ± 0.027</td><td></td><td>0.024 ± 0.020</td><td>0.867 ± 0.080 0.769 ± 0.165</td><td></td><td></td><td>0.588 ± 0.214 0.741 ± 0.134</td></tr><tr><td>MoFD (w GAN)</td><td>24.62 ± 1.19</td><td>0.881 ± 0.029</td><td>0.022 ± 0.017</td><td>0.821 ± 0.104</td><td>0.763 ± 0.146</td><td>0.559 ± 0.174</td><td>0.714 ± 0.107</td></tr></table>

## 5 Results

## 5.1 Results on the Comparison Methods

We evaluated the performance of T1ce synthesis against baseline models quantitatively (Table 1) and qualitatively (Fig. 4) on the UCSF-PDGM [7, 8] dataset. Most baseline models were trained using T1, T2, and T2-FLAIR sequences fused via channel-wise concatenation, whereas Hi-Net and MMgSN-Net used the multimodal fusion strategies proposed in their respective studies. As shown in Table 1, our regression-based method (MoFD w/o GAN) achieved the best fidelity performance, with the highest PSNR and SSIM scores. The second-best fidelity results were obtained by Chen et al. [11], a regression-based model for T1ce synthesis. In general, models trained with adversarial loss are subject to the perception-distortion trade-of [4]: they can improve perceptual quality, as reflected by lower LPIPS scores, but often reduce pixel-level fidelity in terms of PSNR and SSIM. Despite this tendency, our GAN-based method (MoFD w/ GAN) remained competitive, achieving the best LPIPS score and the secondbest PSNR and SSIM scores on UCSF-PDGM. Qualitative comparisons in Fig. 4 further support these quantitative findings. While several comparison methods fail to capture detailed tumor appearance, our method reconstructs both anatomical structures and contrast-enhanced tumor regions more faithfully. In particular, Hi-Net struggles to delineate the tumor core region, whereas MMgSN-Net fails to adequately represent the edema component within the whole tumor region. These qualitative observations are consistent with the segmentation results in Table 1, where the proposed model achieves the best Dice scores for WT, ET, TC, and their average, indicating that its synthesized T1ce images preserve tumor structures more accurately than the comparison methods. A similar trend was observed on the UPENN-GBM dataset, as shown in Fig. 5 and Table 2.

![](images/7cacf896015ac33f2b82273c779fef3e92111d59a78c61bbe8912081ea4f071f.jpg)  
Fig. 5: Qualitative results of T1ce synthesis with comparison methods on the UPENN-GBM [1, 2]. The bounding box (bottom left) shows a zoomed-in view of the tumor region from the T1ce. The bottom right image shows the corresponding tumor segmentation results by nnU-Net [22].

Table 2: Quantitative performance of T1ce synthesis with comparison methods on the UPENN-GBM [1, 2]. PSNR and SSIM are measured and reported as 3D metrics in mean ± std. Bold indicates the best, underlined denotes the second-best performance.
<table><tr><td>Dataset</td><td colspan="7">UPENN-GBM [1, 2]</td></tr><tr><td>Tasks</td><td colspan="3">T1ce Synthesis</td><td colspan="4">Tumor Segmentation</td></tr><tr><td>Model</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>WT</td><td>ET</td><td>TC</td><td>Avg.</td></tr><tr><td>Ea-GAN [53]</td><td>23.08 ± 0.82</td><td>0.833 ± 0.026</td><td>0.047 ± 0.022</td><td> $0 . 7 2 6 \pm 0 . 1 4 1$ </td><td>0.454 ± 0.276</td><td>0.213 ± 0.218</td><td>0.464 ± 0.181</td></tr><tr><td>MM-GAN [48]</td><td>23.82 ± 1.22</td><td>0.840 ± 0.029</td><td> $0 . 0 3 6 \pm 0 . 0 2 2$ </td><td> $0 . 6 8 9 \pm 0 . 1 8 6$ </td><td> $0 . 5 7 0 \pm 0 . 3 5 6$ </td><td> $0 . 3 1 2 \pm 0 . 2 5 9$ </td><td> $0 . 5 2 4 \pm 0 . 2 4 2$ </td></tr><tr><td>Chen et al. [1i]</td><td>24.35 ± 1.25</td><td>0.842 ± 0.026</td><td> $0 . 0 2 5 \pm 0 . 0 1 6$ </td><td> $0 . 6 1 6 \pm 0 . 2 1 8$ </td><td> $0 . 6 1 8 \pm 0 . 3 3 3$ </td><td> $0 . 3 7 2 \pm 0 . 3 1 6$ </td><td> $0 . 5 3 5 \pm 0 . 2 4 7$ </td></tr><tr><td>ResViT [13]</td><td>24.28 ± 1.15</td><td>0.828 ± 0.027</td><td> $0 . 0 2 5 \pm 0 . 0 1 4$ </td><td> $0 . 7 1 8 \pm 0 . 2 3 7$ </td><td> $\underline { { 0 . 6 5 2 \pm 0 . 3 2 7 } }$ </td><td> $\underline { { 0 . 4 7 3 \pm 0 . 2 8 9 } }$ </td><td> $0 . 6 1 4 \pm 0 . 2 5 6$ </td></tr><tr><td>ALDM [29]</td><td>23.55 ± 0.77</td><td>0.814 ± 0.024</td><td> $0 . 0 2 6 \pm 0 . 0 1 7$ </td><td> $0 . 6 6 5 \pm 0 . 2 3 2$ </td><td> $0 . 6 4 6 \pm 0 . 3 2 5$ </td><td> $0 . 4 1 2 \pm 0 . 2 8 2$ </td><td> $0 . 5 7 4 \pm 0 . 2 5 2$ </td></tr><tr><td>Hi-Net [55]</td><td> $2 3 . 4 9 \pm 1 . 0 1$ </td><td>0.845 ± 0.026</td><td> $0 . 0 2 6 \pm 0 . 0 1 8$ </td><td> $0 . 7 1 8 \pm 0 . 1 8 9$ </td><td> $0 . 5 5 0 \pm 0 . 3 0 7$ </td><td> $0 . 3 5 3 \pm 0 . 2 9 9$ </td><td> $0 . 5 4 1 \pm 0 . 2 4 3$ </td></tr><tr><td>MMgSN-Net [35]</td><td> $2 2 . 1 6 \pm 0 . 9 1$ </td><td>0.810 ± 0.029</td><td> $0 . 0 2 9 \pm 0 . 0 1 8$ </td><td> $0 . 6 2 8 \pm 0 . 3 1 2$ </td><td> $0 . 5 4 3 \pm 0 . 3 7 2$ </td><td> $0 . 3 4 4 \pm 0 . 3 0 9$ </td><td> $0 . 5 0 5 \pm 0 . 3 0 3$ </td></tr><tr><td>MoFD (w/o GAN)</td><td>24.80 ± 1.10 0.861 ± 0.025</td><td></td><td> $\underline { { 0 . 0 2 4 } } \pm \mathbf { 0 . 0 1 5 }$ </td><td> $\mathbf { 0 . 7 6 3 \ : \pm { \ : 0 . 1 3 7 } }$ </td><td>0.721 ± 0.277 0.520 ± 0.262</td><td></td><td> $\mathbf { 0 . 6 6 8 \ : \pm { \ : 0 . 2 0 6 } }$ </td></tr><tr><td>MoFD (w GAN)</td><td> $2 4 . 3 3 \pm 1 . 0 1$ </td><td>0.839 ± 0.026</td><td>0.022 ± 0.014</td><td> $\underline { { 0 . 7 6 0 \pm 0 . 1 4 8 } }$ </td><td>0.626 ± 0.371</td><td> $0 . 4 6 4 \pm 0 . 3 3 2$ </td><td> $\underline { { 0 . 6 1 6 \pm 0 . 2 6 4 } }$ </td></tr></table>

![](images/94e3e44c5c63003aae3a472ac22464f01e176be23a466b634b96234bb3abb64b.jpg)  
Fig. 6: Qualitative results of T1ce synthesis with various input sequence combinations on the UCSF-PDGM [7, 8]. The bounding boxes display a zoomed-in view of the boxed area in the bottom left. The bottom right image shows the corresponding tumor segmentation results by nnU-Net [22].

Table 3: Quantitative results of T1ce synthesis and segmentation with various input sequence combinations on the UCSF-PDGM [7, 8].
<table><tr><td colspan="3">Modalities</td><td colspan="3">T1ce Synthesis</td><td colspan="4">Tumor Segmentation</td></tr><tr><td>T1 T2 T2f </td><td></td><td></td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>WT</td><td>ET</td><td>TC</td><td>Avg.</td></tr><tr><td>●</td><td>o</td><td>o</td><td>23.770 ± 1.485</td><td>0.857 ± 0.045</td><td>0.031 ± 0.020</td><td>0.788 ± 0.115</td><td>0.674 ± 0.176</td><td>0.410 ± 0.287</td><td>0.624 ± 0.153</td></tr><tr><td>o</td><td>●</td><td>o</td><td>23.723 ± 0.942</td><td>0.868 ± 0.025</td><td>0.032 ± 0.016</td><td>0.808 ± 0.091</td><td>0.724 ± 0.151</td><td>0.385 ± 0.224</td><td>0.639 ± 0.116</td></tr><tr><td>o</td><td>o</td><td>●</td><td>22.660 ± 0.890</td><td>0.843 ± 0.027</td><td>0.037 ± 0.019</td><td>0.815 ± 0.088</td><td>0.728 ± 0.134</td><td>0.221 ± 0.224</td><td>0.588 ± 0.118</td></tr><tr><td>●</td><td>●</td><td>o</td><td>24.360 ± 1.225</td><td>0.880 ± 0.028</td><td>0.024 ± 0.014</td><td>0.807 ± 0.096</td><td>0.723 ± 0.143</td><td>0.496 ± 0.178</td><td>0.675 ± 0.110</td></tr><tr><td>●</td><td>o</td><td></td><td>23.678 ± 1.283</td><td>0.859 ± 0.034</td><td>0.025 ± 0.015</td><td>0.813 ± 0.089</td><td>0.738 ± 0.134</td><td>0.480 ± 0.184</td><td>0.677 ± 0.100</td></tr><tr><td>o</td><td></td><td>C</td><td>24.091 ± 0.864</td><td>0.877 ± 0.022</td><td>0.025 ± 0.013</td><td>0.801 ± 0.093</td><td>0.724 ± 0.131</td><td>0.445 ± 0.212</td><td>0.657 ± 0.104</td></tr><tr><td>●</td><td></td><td>C</td><td>24.626 ± 1.197</td><td>0.881 ± 0.029</td><td>0.022 ± 0.017</td><td>0.821 ± 0.104</td><td>0.763 ± 0.146</td><td>0.559 ± 0.174</td><td>0.714 ± 0.107</td></tr><tr><td colspan="2">GT T1ce</td><td></td><td>inf</td><td>1.00</td><td>0</td><td>0.826 ± 0.095</td><td>0.767 ± 0.121</td><td>0.840 ± 0.121</td><td>0.811 ± 0.088</td></tr></table>

## 5.2 Multi-Sequence Combinations

We investigated the impact of diferent input sequence combinations on synthesis quality and downstream segmentation performance. We used an nnU-Net [22] trained on ground truth T1ce images to segment the synthesized T1ce images. Table 3 presents the quantitative comparison. The model using all sequences (T1, T2, and T2-FLAIR) achieved the highest scores for fidelity, perceptual quality, and downstream segmentation. For segmentation, the model showed high performance for both WT and ET. However, the model showed limited performance on the TC, the smallest of the subregions. Note that although using only T1 as an input generally improves image fidelity due to its structural consistency with T1ce, it leads to poor segmentation results. This discrepancy arises because T1 images provide insuficient contrast information for distinguishing tumor features. As illustrated in Fig. 6, the qualitative results confirm that the multi-sequence input enables the model to reconstruct tumor details.

## 5.3 Ablation Study

MoFD Architecture. Table 4 details the ablation results regarding the architecture of the proposed module (Fig. 2). Omitting the convolution block leads to inconsistent feature scales due to the lack of normalization, which negatively afects the synthesis quality. Using a weighted sum to calculate the final output significantly outperforms the approach of averaging features, confirming the efectiveness of our dynamic feature selection mechanism.

Table 4: Ablation study on the MoFD Architecture.
<table><tr><td>Dataset</td><td colspan="7">UCSF-PDGM [7, 8]</td></tr><tr><td>Tasks</td><td colspan="3">T1ce Synthesis</td><td colspan="4">Tumor Segmentation</td></tr><tr><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>WT</td><td>ET</td><td>TC</td><td>Avg.</td></tr><tr><td>w/o Conv Block</td><td> $2 3 . 5 9 0 \pm 1 . 0 5 9$ </td><td> $0 . 8 5 9 \pm 0 . 0 2 6$ </td><td>0.028 ± 0.014</td><td> $0 . 8 2 0 \pm 0 . 0 9 7$ </td><td> $0 . 7 0 9 \pm 0 . 1 3 1$ </td><td> $0 . 5 0 1 \pm 0 . 1 8 5$ </td><td>0.677 ± 0.093</td></tr><tr><td>Average Feature</td><td> $2 4 . 3 0 1 \pm 1 . 1 9 8$ </td><td> $0 . 8 7 5 \pm 0 . 0 2 9$ </td><td> $0 . 0 2 3 \pm 0 . 0 1 9$ </td><td> $\mathbf { 0 . 8 2 4 \ : \pm { \ : 0 . 0 9 4 } }$ </td><td> $0 . 7 1 6 \pm 0 . 1 4 6$ </td><td> $0 . 5 1 2 \pm 0 . 1 7 5$ </td><td>0.685 ± 0.098</td></tr><tr><td>Ours</td><td>24.626 ± 1.197 0.881 ± 0.029 0.022 ± 0.017|</td><td></td><td></td><td></td><td>0.821 ± 0.104 0.763 ± 0.146 0.559 ± 0.174 0.714 ± 0.107</td><td></td><td></td></tr></table>

![](images/7581ba22502aa27a53f83ebf6843eada11b0bd9068b0102a55858e4909acdf84.jpg)  
Fig. 7: Visualization of features, feature diferences, and weighted sum features derived from source T1, T2, and FLAIR modalities. $f _ { w s u m }$ denotes the weighted sum feature. The red bounding boxes indicate zoomed-in views of the tumor region.

Loss Function. Table 5 presents the ablation study on the three components of $\mathcal { L } _ { \mathrm { t o t a l } } ~ \mathrm { ( E q . ~ 9 ) }$ . Excluding the ${ \mathcal { L } } _ { \mathrm { g a n } }$ yielded superior fidelity scores in T1ce synthesis. While incorporating the ${ \mathcal { L } } _ { \mathrm { g a n } }$ improved perceptual quality, it reduced fidelity. Notably, incorporating $\mathcal { L } _ { \mathrm { a l i g n } }$ improved both fidelity and quality metrics in all cases. Interestingly, the regression-based model $( \mathrm { w } / \mathrm { o } \ \mathcal { L } _ { \mathrm { g a n } } )$ demonstrated better tumor segmentation performance than the GAN-based model. This indicates that although adversarial training improves perceptual realism, it risks introducing realistic but fake details that deviate from the ground truth, thereby degrading segmentation accuracy. Thus, for tasks requiring structural precision, a regression-based approach might be better.

Top-K Selection. Table 6 summarizes the performance of the Top-K feature selection strategy, which is similar to the MoE [49] framework. We evaluated the model by varying K to select the top-weighted features. The results indicate that the best performance was achieved when all six available features, comprising three sequence features and three diference maps, across both datasets was better than selecting a partial subset.

Table 5: Ablation study on the Loss function.
<table><tr><td colspan="3">Dataset</td><td colspan="8">UCSF-PDGM [7, 8]</td></tr><tr><td colspan="3">Tasks</td><td colspan="3">T1ce Synthesis</td><td colspan="4">Tumor Segmentation</td></tr><tr><td> $\mathcal { L } _ { r e c } ~ \mathcal { L } _ { a l i g n } ~ \mathcal { L } _ { g a n }$ </td><td></td><td></td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>WT</td><td>ET</td><td>TC</td><td></td><td>Avg.</td></tr><tr><td>√</td><td></td><td></td><td> $2 4 . 8 3 3 \pm 1 . 2 6 3$ </td><td> $0 . 8 9 1 \pm 0 . 0 2 9$ </td><td> $0 . 0 2 7 \pm 0 . 0 1 4$ </td><td> $0 . 8 5 9 \pm 0 . 0 7 8$ </td><td></td><td> $0 . 7 5 7 \pm 0 . 1 1 7$ </td><td> $0 . 4 7 4 \pm 0 . 2 6 4$ </td><td> $0 . 6 9 6 \pm 0 . 1 1 5$ </td></tr><tr><td>√</td><td>√</td><td></td><td> $\mathbf { 2 5 . 0 0 9 \ : \pm { \ : 1 . 2 6 3 } }$ </td><td> $\mathbf { 0 . 8 9 4 \ : \pm { \ : 0 . 0 2 7 } }$ </td><td> $0 . 0 2 4 \pm 0 . 0 2 0$ </td><td></td><td>0.867 ± 0.080 0.769 ± 0.165</td><td></td><td> $\mathbf { 0 . 5 8 8 \ : \pm { \ : 0 . 2 1 4 } }$ </td><td> $\mathbf { 0 . 7 4 1 \pm 0 . 1 3 4 }$ </td></tr><tr><td>√</td><td></td><td>√</td><td> $2 4 . 0 2 5 \pm 1 . 2 1 1$ </td><td> $0 . 8 8 8 \pm 0 . 0 2 6$ </td><td> $0 . 0 2 3 \pm 0 . 0 1 3$ </td><td> $0 . 8 2 4 \pm 0 . 0 9 2$ </td><td></td><td> $0 . 7 1 8 \pm 0 . 1 4 5$ </td><td> $0 . 5 4 5 \pm 0 . 1 5 2$ </td><td> $0 . 6 9 5 \pm 0 . 0 9 8$ </td></tr><tr><td></td><td>√</td><td>√</td><td> $2 4 . 6 2 6 \pm 1 . 1 9 7$ </td><td> $0 . 8 8 1 \pm 0 . 0 2 9$ </td><td> $\mathbf { 0 . 0 2 2 \sigma \pm 0 . 0 1 7 }$ </td><td></td><td> $0 . 8 2 1 \pm 0 . 1 0 4$ </td><td> $0 . 7 6 3 \pm 0 . 1 4 6$  </td><td> $0 . 5 5 9 \pm 0 . 1 7 4$ </td><td> $0 . 7 1 4 \pm 0 . 1 0 7$ </td></tr></table>

Table 6: Ablation study on the Top-K Selection.
<table><tr><td rowspan="2">Dataset Top-K</td><td colspan="3">UCSF-PDGM [7, 8]</td><td colspan="3">UPenn-GBM [1, 2]</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>1</td><td> $2 4 . 2 7 0 \pm 1 . 1 9 9$ </td><td> $0 . 8 7 5 \pm 0 . 0 3 1$ </td><td> $0 . 0 2 5 \pm 0 . 0 1 5$ </td><td> $2 3 . 9 0 2 \pm 1 . 0 4 4$ </td><td> $0 . 8 3 0 \pm 0 . 0 2 7$ </td><td> $0 . 0 2 3 \pm 0 . 0 1 9$ </td></tr><tr><td>2</td><td> $2 4 . 2 9 4 \pm 1 . 2 0 2$ </td><td> $0 . 8 7 5 \pm 0 . 0 3 1$ </td><td> $0 . 0 2 4 \pm 0 . 0 1 5$ </td><td> $2 3 . 9 4 0 \pm 0 . 9 6 1$ </td><td> $0 . 8 3 1 \pm 0 . 0 2 7$ </td><td> $0 . 0 2 3 \pm 0 . 0 1 6$ </td></tr><tr><td>3</td><td> $2 4 . 4 7 6 \pm 1 . 1 8 1$ </td><td> $0 . 8 8 0 \pm 0 . 0 3 0$ </td><td> $0 . 0 2 3 \pm 0 . 0 1 3$ </td><td> $2 4 . 0 0 1 \pm 0 . 9 5 5$ </td><td> $0 . 8 3 3 \pm 0 . 0 2 7$ </td><td> $0 . 0 2 2 \pm 0 . 0 1 7$ </td></tr><tr><td>4</td><td> $2 4 . 4 6 1 \pm 1 . 1 9 2$ </td><td> $0 . 8 8 0 \pm 0 . 0 3 0$ </td><td> $0 . 0 2 3 \pm 0 . 0 1 4$ </td><td> $2 4 . 0 9 2 \pm 0 . 9 7 3$ </td><td> $0 . 8 3 5 \pm 0 . 0 2 6$ </td><td> $0 . 0 2 2 \pm 0 . 0 1 6$ </td></tr><tr><td>5</td><td> $2 4 . 4 3 7 \pm 1 . 2 0 2$ </td><td> $0 . 8 7 8 \pm 0 . 0 3 0$ </td><td> $0 . 0 2 4 \pm 0 . 0 1 5$ </td><td> $2 4 . 2 0 2 \pm 0 . 9 8 2$ </td><td> $0 . 8 3 6 \pm 0 . 0 2 7$ </td><td> $0 . 0 2 2 \pm 0 . 0 1 7$ </td></tr><tr><td>6</td><td> $\mathbf { 2 4 . 6 2 6 \\\\pm 1 . 1 9 7 }$ </td><td> $\mathbf { 0 . 8 8 1 \ : \pm { \ : 0 . 0 2 9 } }$ </td><td> $\mathbf { 0 . 0 2 2 \sigma \pm 0 . 0 1 7 }$ </td><td> $\mathbf { 2 4 . 3 3 6 \ : \pm { \ : 1 . 0 1 1 } }$  </td><td> $\mathbf { 0 . 8 3 9 \ : \pm { \ : 0 . 0 2 6 } }$  </td><td> $\mathbf { 0 . 0 2 2 \lambda \pm 0 . 0 1 4 }$ </td></tr></table>

![](images/b1cbdf69b99d3f7455bf1161e3e71bf72d587def353074d51d97fb26377c4dc0.jpg)  
Fig. 8: Visualization of the feature selection probability in the MoFD module. At low-level scales (scales 0 and 1), the selection probabilities are relatively evenly distributed across features and feature diferences. As the network becomes deeper and approaches higher-level representations, the selection probabilities for T2 and T2- FLAIR features increase. Since image translation requires accurate pixel-level synthesis, low-level features passed through the skip connections remain important and the proposed module efectively uses all available features across scales.

## 5.4 Analysis of Feature Selection

We visualized the features and feature diferences produced by the MoFD module, as shown in Fig. 7. In the tumor region, each feature and feature diference captures the tumor appearance from a diferent representation. The weightedsum feature is then generated by adaptively combining these features according to their selection probabilities. Notably, this aggregated feature closely resembles the tumor region in the target image, suggesting that the proposed module efectively extracts complementary representations and dynamically integrates them to produce a final feature representation suitable for target T1ce synthesis.

We further visualized which features were selected and passed to the decoder at each scale, as shown in Fig. 8. At the low-level scales, i.e., scales 0 and 1, the selection probabilities are relatively evenly distributed across the individual features and feature diferences, indicating that the MoFD module flexibly selects complementary information to support target synthesis. As the features become deeper, the selection probabilities for T2 and T2-FLAIR tend to increase. This may be because these modalities exhibit contrast patterns similar to T1ce in non-tumor structural regions. However, voxel-level accuracy is particularly important for tumor regions, detailed texture synthesis, and the delineation of enhancing tumor and tumor core regions. Therefore, low-level features can play a more critical role. The relatively balanced feature usage at low-level scales suggests that the proposed module efectively exploits diverse modality-specific and diference-based cues for fine-grained synthesis. This observation is also consistent with Kim et al. [26], who showed that coarse layers mainly contribute to forming global image structure, whereas their influence can be reduced when the input image already provides strong structural guidance for image-to-image translation.

## 6 Conclusion

We propose a framework for synthesizing T1ce images from non-contrast multisequence MRI using the MoFD module. Instead of relying on simple channel-wise concatenation, MoFD explicitly models both sequence-specific features and their pairwise feature diferences, allowing the network to adaptively select complementary information useful for contrast-enhanced image synthesis. Experiments on the UCSF-PDGM and UPenn-GBM datasets showed that our method consistently outperformed baseline approaches in image fidelity and improved downstream tumor segmentation performance. The ablation studies further demonstrated the contribution of the proposed design: feature normalization and dynamic weighting are important for stable fusion, all input sequences and their diference features are beneficial for synthesis quality, and regression-based training can better preserve structural fidelity than adversarial training when accurate tumor representation is required.

## Acknowledgement

This study was supported by the SMC-SKKU Future Convergence Research Program Grant (SMO125123). It was also supported by the National Research Foundation of Korea (NRF) grant (RS-2025-00518533), the AI Graduate School Support Program (Sungkyunkwan University) (RS-2019-II190421), the National Research Foundation of Korea (RS-2024-00408040, RS-2026-25598771), the ICT Creative Consilience Program Grant (IITP-2026-RS-2020-II201821), the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant (RS-2026-25528384, Resource-Intensive AI Technologies Based on Sustainable GPU Integrated Platforms), the IITP grant (IITP-2026-RS-2026-25615334, AI Star Fellowship Support Program), and an IITP grant for the AI Computing Support Project for R&D (RS-2026-25505492, High-Performance Research AI Computing Infrastructure Support at the 2 PFLOPS Scale), all funded by the Ministry of Science and ICT (MSIT), Republic of Korea.

## References

1. Bakas, S., et al.: Multi-parametric magnetic resonance imaging (mpmri) scans for de novo glioblastoma (gbm) patients from the university of pennsylvania health system (upenn-gbm). The Cancer Imaging Archive 10 (2021) 8, 10, 13

2. Bakas, S., et al.: The university of pennsylvania glioblastoma (upenn-gbm) cohort: advanced mri, clinical, genomics, & radiomics. Scientific data 9(1), 453 (2022) 8, 10, 13

3. Bitar, R., et al.: Mr pulse sequences: what every radiologist wants to know but is afraid to ask. Radiographics 26(2), 513–537 (2006) 2

4. Blau, Y., Michaeli, T.: The perception-distortion tradeof. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6228–6237 (2018) 2, 4, 7, 10

5. Bône, A., Ammari, S., Lamarque, J.P., Elhaik, M., Chouzenoux, É., Nicolas, F., Robert, P., Balleyguier, C., Lassau, N., Rohé, M.M.: Contrast-enhanced brain mri synthesis with deep learning: key input modalities and asymptotic performance. In: 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI). pp. 1159–1163. IEEE (2021) 3

6. Bottani, S., Thibeau-Sutre, E., Maire, A., Ströer, S., Dormont, D., Colliot, O., Burgos, N., Group, A.S.: Contrast-enhanced to non-contrast-enhanced image translation to exploit a clinical data warehouse of t1-weighted brain mri. BMC Medical Imaging 24(1), 67 (2024) 4

7. Calabrese, E., et al.: The university of california san francisco preoperative difuse glioma mri (ucsf-pdgm)(version 4)[dataset]. The Cancer Imaging Archive 10 (2022) 8, 9, 11, 12, 13

8. Calabrese, E., et al.: The university of california san francisco preoperative difuse glioma mri dataset. Radiology: Artificial Intelligence 4(6), e220058 (2022) 8, 9, 11, 12, 13

9. Caravan, P., et al.: Gadolinium (iii) chelates as mri contrast agents: structure, dynamics, and applications. Chemical reviews 99(9), 2293–2352 (1999) 1

10. Carr, D., Brown, J., Bydder, G., Weinmann, H., Speck, U., Thomas, D., Young, I.: Intravenous chelated gadolinium as a contrast agent in nmr imaging of cerebral tumours. The Lancet 323(8375), 484–486 (1984) 1

11. Chen, C., et al.: Synthesizing mr image contrast enhancement using 3d highresolution convnets. IEEE Transactions on Biomedical Engineering 70(2), 401–412 (2022) 2, 4, 8, 9, 10

12. Cho, N.S., et al.: Quantification of t2-flair mismatch in nonenhancing difuse gliomas using digital subtraction. American Journal of Neuroradiology 45(2), 188– 197 (2024) 3

13. Dalmaz, O., Yurt, M., Çukur, T.: Resvit: Residual vision transformers for multimodal medical image synthesis. IEEE Transactions on Medical Imaging 41(10), 2598–2614 (2022) 2, 4, 8, 9, 10

14. De Coene, B., Hajnal, J.V., Gatehouse, P., Longmore, D.B., White, S.J., Oatridge, A., Pennock, J., Young, I., Bydder, G.: Mr of the brain using fluid-attenuated inversion recovery (flair) pulse sequences. American journal of neuroradiology 13(6), 1555–1564 (1992) 2

15. Dosovitskiy, A.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020) 2

16. Ginat, D.T., Meyers, S.P.: Intracranial lesions with high signal intensity on t1- weighted mr images: diferential diagnosis. Radiographics 32(2), 499–516 (2012) 2

17. Goodfellow, I.J., et al.: Generative adversarial nets. Advances in neural information processing systems 27 (2014) 2

18. Han, L., Tan, T., Zhang, T., Huang, Y., Wang, X., Gao, Y., Teuwen, J., Mann, R.: Synthesis-based imaging-diferentiation representation learning for multi-sequence 3d/4d mri. Medical Image Analysis 92, 103044 (2024) 4

19. Han, L., Zhang, T., Huang, Y., Dou, H., Wang, X., Gao, Y., Lu, C., Tan, T., Mann, R.: An explainable deep framework: towards task-specific fusion for multito-one mri synthesis. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 45–55. Springer (2023) 4

20. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016) 8

21. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020) 2

22. Isensee, F., et al.: nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18(2), 203–211 (2021) 3, 8, 9, 10, 11

23. Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with conditional adversarial networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1125–1134 (2017) 2, 7, 8

24. Jiao, C., Ling, D., Bian, S., Vassantachart, A., Cheng, K., Mehta, S., Lock, D., Zhu, Z., Feng, M., Thomas, H., et al.: Contrast-enhanced liver magnetic resonance image synthesis using gradient regularized multi-modal multi-discrimination sparse attention fusion gan. Cancers 15(14), 3544 (2023) 4

25. Kanda, T., Ishii, K., Kawaguchi, H., Kitajima, K., Takenaka, D.: High signal intensity in the dentate nucleus and globus pallidus on unenhanced t1-weighted mr images: relationship with increasing cumulative dose of a gadolinium-based contrast material. Radiology 270(3), 834–841 (2014) 2

26. Kim, G., Kim, B., Park, E., Cho, S.: Difusion model compression for image-toimage translation. In: Proceedings of the Asian Conference on Computer Vision. pp. 2105–2123 (2024) 14

27. Kim, J.: Posterior samplings are missing modalities generators for medical image translation. arXiv preprint arXiv:2607.18763 (2026) 2

28. Kim, J., Na, I., Ko, E.S., Park, H.: Tumor synthesis conditioned on radiomics. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 3635–3646. IEEE (2025) 4

29. Kim, J., Park, H.: Adaptive latent difusion model for 3d medical image to image translation: Multi-modal magnetic resonance imaging study. In: Proceedings of the IEEE/CVF Winter conference on applications of computer Vision. pp. 7604–7613 (2024) 2, 8, 9, 10

30. Kim, J., Park, H.: Radiomics-guided multimodal self-attention network for predicting pathological complete response in breast mri. In: 2024 IEEE International Symposium on Biomedical Imaging (ISBI). pp. 1–5. IEEE (2024) 1

31. Kim, J., Park, H.: Simulating post-neoadjuvant chemotherapy breast cancer mri via difusion model with prompt tuning. In: 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI). pp. 1–5. IEEE (2025) 1

32. Kim, J., Park, H.: Visual prompt tuning for task-flexible medical image synthesis. Computer Methods and Programs in Biomedicine p. 109244 (2026) 4

33. Kingma, D.P.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014) 8

34. Ledneva, E., Karie, S., Launay-Vacher, V., Janus, N., Deray, G.: Renal safety of gadolinium-based contrast media in patients with chronic renal insuficiency. Radiology 250(3), 618–628 (2009) 3

35. Li, W., Zhao, D., Zeng, G., Chen, Z., Huang, Z., Lam, S., Cheung, A.L.Y., Ren, G., Liu, C., Liu, X., et al.: Evaluating virtual contrast-enhanced magnetic resonance imaging in nasopharyngeal carcinoma radiation therapy: a retrospective analysis for primary gross tumor delineation. International Journal of Radiation Oncology\* Biology\* Physics 120(5), 1448–1457 (2024) 4, 8, 9, 10

36. Liu, M.Y., Breuel, T., Kautz, J.: Unsupervised image-to-image translation networks. Advances in neural information processing systems 30 (2017) 5

37. Liu, M.Y., Tuzel, O.: Coupled generative adversarial networks. Advances in neural information processing systems 29 (2016) 5

38. Lohrke, J., et al.: 25 years of contrast-enhanced mri: developments, current challenges and future perspectives. Advances in therapy 33(1), 1–28 (2016) 2

39. Mei, X., et al.: Radimagenet: an open radiologic deep learning research dataset for efective transfer learning. Radiology: Artificial Intelligence 4(5), e210315 (2022) 8

40. Murata, N., Gonzalez-Cuyar, L.F., Murata, K., Fligner, C., Dills, R., Hippe, D., Maravilla, K.R.: Macrocyclic and other non–group 1 gadolinium contrast agents deposit low levels of gadolinium in brain and bone tissue: preliminary results from 9 patients with normal renal function. Investigative radiology 51(7), 447–453 (2016) 2

41. Oh, Y., Jin, P., Park, S., Kim, S., yoon, S., Kim, J.S., Kim, K., Li, X., Li, Q.: Distribution-aware fairness learning in medical image segmentation from a controltheoretic perspective. In: Forty-second International Conference on Machine Learning (2025) 4

42. Park, T., Liu, M.Y., Wang, T.C., Zhu, J.Y.: Semantic image synthesis with spatially-adaptive normalization. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2337–2346 (2019) 2

43. Pooley, R.A.: Fundamental physics of mr imaging. Radiographics 25(4), 1087–1099 (2005) 2

44. Rassmann, S., Kügler, D., Ewert, C., Reuter, M.: Regression is all you need for medical image translation. IEEE Transactions on Medical Imaging (2026) 2, 4

45. Rokuss, M.R., Kirchhof, Y., Roy, S., Kovacs, B., Ulrich, C., Wald, T., Zenk, M., Denner, S., Isensee, F., Vollmuth, P., et al.: Longitudinal segmentation of ms lesions via temporal diference weighting. In: International Conference on Medical Image Computing and Computer-Assisted Intervention - MICCAI 2024 Workshops. pp. 64–74. Springer (2024) 4

46. Rombach, R., et al.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022) 2, 8

47. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015) 5, 8

48. Sharma, A., Hamarneh, G.: Missing mri pulse sequence synthesis using multi-modal generative adversarial network. IEEE transactions on medical imaging 39(4), 1170– 1183 (2019) 2, 8, 9, 10

49. Shazeer, N., et al.: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In: International Conference on Learning Representations (2017) 4, 12

50. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004) 6

51. Weinmann, H.J., Brasch, R.C., Press, W.R., Wesbey, G.E.: Characteristics of gadolinium-dtpa complex: a potential nmr contrast agent. American journal of roentgenology 142(3), 619–624 (1984) 1

52. Xiong, H., Wang, Y., Shen, Z., Sun, K., Fang, Y., Chen, Y., Shen, D., Wang, Q.: Learning contrast and content representations for synthesizing magnetic resonance image of arbitrary contrast. Medical Image Analysis p. 103635 (2025) 6

53. Yu, B., Zhou, L., Wang, L., Shi, Y., Fripp, J., Bourgeat, P.: Ea-gans: edge-aware generative adversarial networks for cross-modality mr image synthesis. IEEE transactions on medical imaging 38(7), 1750–1762 (2019) 2, 8, 9, 10

54. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018) 8

55. Zhou, T., Fu, H., Chen, G., Shen, J., Shao, L.: Hi-net: hybrid-fusion network for multi-modal mr image synthesis. IEEE transactions on medical imaging 39(9), 2772–2781 (2020) 4, 8, 9, 10