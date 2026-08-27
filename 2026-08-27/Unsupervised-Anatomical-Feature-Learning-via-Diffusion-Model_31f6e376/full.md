Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000. Digital Object Identifier 10.1109/ACCESS.2024.0429000

# Unsupervised Anatomical Feature Learning via Diffusion Models: Enhanced Medical Image Segmentation with Denoising Diffusion Probabilistic Models

AKSHAT G<sup>1</sup>, DIVYANSH GUPTA<sup>1</sup>, SHALEEN BHATNAGAR<sup>1</sup> (Senior Member, IEEE) , SHILPA ANKALAKI<sup>1</sup>, AND TUSAR KANTI MISHRA<sup>1</sup> (Senior Member, IEEE)0

<sup>1</sup>Manipal Institute of Technology Bengaluru, Manipal Academy of Higher Education, Manipal, India

Corresponding author: Shaleen Bhatnagar (shaleen.bhatnagar@manipal.edu)

This work was supported by Manipal Academy of Higher Education (Open Access Funding).

ABSTRACT The process of acquiring pixel and voxel level annotations is a bottleneck when it comes to medical image segmentation as it is extremely expensive. Furthermore, it requires expert diagnosis to manually delineate anatomical structures across three-dimensional volumes. Although traditional neural network architectures like U-net are affective, they operate as black boxes that learn local texture patterns. These models lack awareness of global anatomical structures. This leads to potential catastrophic failures in boundary delineation and poor generalization when quantity of labeled data is limited. This study proposes utilizing unsupervised Denoising Diffusion Probabilistic Models (DDPMs) to extract anatomical features. The proposed framework trains a DDPM on 21 unlabeled abdominal CT scans to learn structural representations through the iterative denoising process, which later transfers the learned encoder weights to perform a down-stream segmentation task. The framework is evaluated using data from the BTCV multi organ dataset. Diffusion pretraining significantly improved liver segmentation where Dice increased from 0.75±0.36 to 0.93±0.16 (p < 5.33×10<sup>−26</sup>, with 0.529 Cohen’s d), reduced 95th-percentile Hausdorff Distance (HD95) by 45% (11.76 to 6.50 mm), and decreased Average Surface Distance (ASD) by 66% (4.38 to 1.51 mm). For kidney segmentation, Dice improved from 0.90±0.19 to 0.95±0.10 (p < 4.01×10<sup>−11</sup>, with 0.271 Cohen’s d), with 37% HD95 reduction. Multi-organ pooled performance showed the most dramatic improvement, where Dice increased from 0.78±0.23 to 0.95±0.07, and representing a 68% reduction in variance with 74% improvement in boundary precision. Furthermore, frozen encoder models achieved >80% of fine-tuned performance without seeing any segmentation labels, demonstrating the existence of learned anatomical priors. In low-data regimes, diffusion-pretrained models maintained robust performance with only 50% (Dice: 0.92 liver, 0.94 kidney), 25% (Dice: 0.90 liver, 0.81 kidney), and even 10% of labeled data (Dice: 0.89 liver, 0.71 kidney), substantially outperforming random initialization baselines. Using unlabeled images for diffusion-based pretraining helps embed robust, interpretable anatomical features into encoder network prior to any human supervision. The performance retention when compared to low-data scenarios, and the graceful degradation even with only 10% of labelled data confirms that the learned representations greatly improve segmentation accuracy, boundary precisions and model robustness. The diffusion pretraining effectively helps transform U-Nets from texture matching networks to anatomy aware systems.

INDEX TERMS Diffusion models, medical image segmentation, unsupervised learning, transfer learning, abdominal CT imaging, U-Net, DDPM, anatomical feature learning.

## I. INTRODUCTION

mentation, pixel level segmentation of anatomical structures and pathologies are required, which in turn creates a severe bottleneck [1]. Unlike image classification where labels can be crowd-sourced, medical segmentation required trained radiologists to manually trace out and segment organ boundaries across three dimensional volumes, one slice at a time. A single CT scan containing anywhere between 100 to 300 slices may require an expert radiologist 30 to 60 minutes of annotation time, with costs often exceeding 100 dollars per volume [1]–[3].

Inter labeler variability introduces additional challenges, previous studies have shown that even expert radiologists disagree on precise boundary delineation [4], [5], particularly for organs with diffused edges like liver. This variability in turn propagates into the training data, resulting in inconsistent signals that limit model performance and reliability [6], [7]. In a resource constrained clinical setting, or for rare pathologies, obtaining sufficient annotated data becomes practically infeasible, such scenarios are where automated assistance would provide the greatest clinical value [5], [7].

Since 2015, the U-Net architecture and its variants have been leading choice of use for medical image segmentation tasks [8]. This is because of their ability to achieve competitive results on benchmarking datasets [8], [9]. However, these models have a few fundamental limitations [9], [10]. A randomly initialized standard U-Net learns individual features which are optimized to minimize pixel-wise classification loss on training data. This optimization process helps the network in identifying local texture patterns, edge orientations, intensity gradients, and co-occurrence statistics that correlate with organ boundaries in the training set.

This texture-focused learning reveals three significant failure modes:

• Boundary failures: Models usually struggle with segmenting boundaries precisely in case of organs with irregular shapes or variable contrasts. A high 95thpercentile Hausdorff Distance (HD95) suggests that although overall segmentation may seem reasonable, inaccuracies such as missing lobes, incorrect attachments, or disconnected regions may be present at the boundaries [9], [11].

• Lack of global spatial context: Convolution inherently is a local operation. While the skip connections from the U-Net help propagate features, randomly initialized encoders lack any prior understanding of anatomical topology, such as livers occupy specific regions, kidneys come in pairs, or that specific organs follow predictable size ranges [12], [13].

• Poor performance in low-data regimes: When ground truth for a specific task is limited, texture-based discrimination becomes unreliable. Models will tend to overfit spurious correlations in limited training examples. This results in failures to generalize to unseen anatomical variations, imaging protocols or certain patient demographics [9], [14].

This study proposes a paradigm shift where rather than training segmentation models from random initialization, a DDPM [15] is deployed to learn unsupervised anatomical representations. Diffusion models have recently been adopted for generative modeling, and producing synthetic images by learning to reverse a gradual noise corruption process [16]– [21]. However, the objective of this study focuses on leveraging the reverse diffusion process not to generate synthetic images, but to force an encoder to learn the underlying structural patterns of abdominal anatomy.

To predict and remove noise at intermediate steps, the network must learn anatomy such as organ shapes, positions, relative arrangements. The model must also learn boundary characteristics of organs, and background with respect to each other along with global spatial relationships and size constraints. The learning process in DDPM occurs without manual annotation, and the training signal comes purely from the data distribution itself. After pretraining the model with images, the encoder weights are transferred to downstream segmentation task, where even with limited labeled data, the encoder weights can be fine tuned with anatomically aware features for precise boundary prediction.

The main contributions of this study are:

1) Statistically significant improvements in spatial boundary precision: The results inferred from the study demonstrate that diffusion pretraining produces statistically substantial and reproducible improvements in segmentation accuracy across multiple metrics like Dice [22], IoU [23] and predominantly in boundary precision metrics like HD95 [24] and ASD [25]. Statistical analysis using Wilcoxon signed-rank tests with Bonferroni correction confirms significance at (p < 10<sup>−10</sup>) [26], with medium effect sizes (Cohen’s d = 0.271–0.529) [27].

2) Variance reduction and improved robustness: Diffusion pretraining reduces prediction variance by 47– 68%, inferred with improved generalization across diverse anatomical configurations and imaging conditions.

3) Evidence for learned anatomical priors: Frozen encoder experiments demonstrate that DDPM-pretrained encoders retain over 80% of the performance of fully fine-tuned models for liver, despite never being trained on segmentation label. This provides direct evidence that the unsupervised pretraining phase learns meaningful anatomical structure, not just generic image features.

4) Multi-organ generalization: This study shows that single diffusion pretraining benefits multiple anatomically distinct organs (liver and kidney), with multiorgan models achieving the best overall performance. This suggests that diffusion models learn generalizable abdominal anatomy representations rather than organspecific patterns.

## II. RELATED WORK

## A. DEEP LEARNING IN MEDICAL IMAGE SEGMENTATION

Fully Convolutional Networks(FCNs) [28]–[32] were one of the earliest applications of deep-learning for medical image segmentation. This field was later redefined by the introduction of the U-Net architecture [8]. The skip connections first hypothesized in the U-Net combine high-resolution spatia information from the encoder with semantic features from the decoder. This concept standardized well with the requirements of precise boundary localization in medical imaging.

Subsequent work resulting from U-Net has explored numerous architectural variations. V-Net extended U-Net to 3D volumetric segmentation with residual connections [33]. Attention U-Net incorporated attention mechanisms to help the network focus on relevant spatial regions [34]. nnU-Net systematized hyperparameter selection and preprocessing, achieving strong performance across diverse medical imaging tasks through careful engineering rather than architectural novelty [35]. More recently, Vision Transformers have been adapted for medical segmentation, with architectures like Swin-UNETR demonstrating that self-attention mechanisms can capture long-range dependencies that convolutions miss [36], [37].

Despite these advances, a common limitation for these models is the fact that they require substantial amounts of labeled training data and learn features from random initialization [38]–[41].

## B. SELF-SUPERVISED REPRESENTATION LEARNING

The goal of Self-Supervised Learning (SSL) [42], [43] is to learn useful representation from unlabeled data by defining a pretext task. This pretext task provides sensory input without the requirement of manual annotation. Learning methods like SimCLR and Moco have achieved success in computer vision and contrastive learning tasks. They do this by learning representations that maximize agreement between different augmented views of the same image while also minimizing the similarity between two different augmented views of different images [44]–[46].

Several SSL approaches have been explored prior for medical imaging. Autoencoders learn compressed representations by reconstructing input images, forcing the bottleneck to capture essential information [47], [48]. Contrastive learning has been applied using domain-specific augmentations like random cropping, rotation, and intensity transformations [44]. Masked Autoencoders (MAE), inspired by BERT in NLP, randomly mask patches of images and train networks to reconstruct the missing regions [49], [50].

However, these methods have limitations for dense prediction tasks. Contrastive learning optimizes for image-level representations, potentially discarding fine-grained spatial information crucial for segmentation [51]. Standard autoencoders with MSE reconstruction loss may learn to reproduce textures and intensities without capturing higher-level anatomical structure. MAE’s discrete masking strategy can miss subtle boundary information [52].

Alternatively, the gradual denoising process of diffusion requires learning of hierarchical representations at different scales, from fine to global textures. This denoising objective explicitly encourages the network to understand spatial relationships and structural coherence, which makes it particularly advantageous for segmentation pretraining.

## C. DIFFUSION MODELS FOR FEATURE EXTRACTION

Denoising Diffusion Probabilistic Models (DDPMs) [15] are used to generate high-quality image data, quickly exceeding GANs in sample quality and training stability [53]–[55]. This fundamental approach is based on the principle of gradually corrupting data with noise over numerous iterations and timesteps, followed by training a neural network to reverse this process.

Recent work have started exploring the application of diffusion models for purposes beyond generation. Evidence suggests that the features acquired from diffusion models are effectively transferable to classification tasks involving natural images [56]. Furthermore, it has been observed that diffusion models implicitly learn sematic segmentation during the generation process [57]. In medical imaging, diffusion models have been directly applied to the segmentation tasks, using the reverse process to iteratively enhance segmentation [58].

The current study expands upon these foundations while also diverging in different key aspects. Instead of utilizing diffusion for comprehensive segmentation, which is computationally expensive at inference time, an unsupervised pretraining method is used, focusing completely on extracting only the learned encoder weights.

This approach provides the benefits of diffusion’s comprehensive learned representations while ensuring the effectiveness of conventional segmentation architectures during deployment. Furthermore, this study focuses on limited data, critical boundary precision requirements, and the need for interpretable anatomically grounded features.

## III. METHODS

## A. DATASET AND PREPROCESSING PIPELINE

## 1) Dataset Description

The methodology was evaluated on the Beyond the Cranial Vault (BTCV) multi-organ abdominal CT segmentation dataset [59], a widely-used benchmark in medical image segmentation research. Sample images from the dataset are shown in Fig. 1. The dataset comprises 30 abdominal CT scans from patients undergoing radiotherapy treatment, with expert annotations for 13 anatomical structures including liver, kidneys, spleen, stomach, gallbladder, pancreas, and various vascular structures. Images were acquired using clinical protocols with varying slice thickness (2.5–5.0 mm), inplane resolution (0.54–0.98 mm), and scanner manufacturers, providing realistic heterogeneity representative of clinical deployment scenarios [60].

## 2) Preprocessing Pipeline

A domain-specific preprocessing pipeline, as demonstrated in Fig. 2, was utilized to improve soft tissue visualization. In addition to image-level processing, patient-level preprocessing was performed to ensure consistency across volumetric samples. Algorithm 1 summarizes the full preprocessing pipeline, including both image-level and patient-level operations.

![](images/23d982ac9d4dd34bbe40d42a04eea17a08603065b775b3accc4633fae9805ec0.jpg)  
FIGURE 1. Sliced 2D images from a single patient in the dataset.

![](images/541017ea65ba956593a837d0215379ac0457324b6acdf9e24cde4ab78e894e92.jpg)  
FIGURE 2. A domain-specific preprocessing pipeline optimized for soft tissue visualization.

Sub-heading: Image Level Preprocessing. Initially threedimensional NIfTI volumes are converted to 2D axial slices. CT intensities were clipped to the range [−100, 400] HU [61], [62], optimized for soft tissue contrast. After windowing, intensities were linearly normalized to [0, 1] to standardize input ranges across different scanners and acquisition protocols. All CT slices were resized to 256×256 pixels using bilinear interpolation, balancing computation efficiency and preserving anatomical details. Only axial slices containing liver or kidney annotations were retained based on groundtruth masks, discarding empty slices to focus computational resources on relevant anatomy. This filtering was performed using the ground-truth masks to ensure consistent slice selection across train/validation/test splits.

Patient Level Data Splitting. Prevention of data leaks is critical to this experimental design process. Patient Level Splitting was the focus of the current study, as opposed to Slice Level or Volume Level Splitting. In Patient Level Splitting, each slice comes from only one patient, in either a training, validation or test set.

This is not the case for the Slice Level and Volume Level Splits where multiple patients’ slices are considered in training, validation, and test sets. A large portion of the data is from adjacent slices of the same patient; they contain very similar anatomical features, have minor differences in position and appearance. If Slice Level Splits were done, the model would memorize specific anatomical characteristics associated with training patients, and then would be evaluated against patientspecific samples (both during testing) resulting in inflated performance metrics.

In this study, a patient level split of 70%, 15% and 15% was achieved for the training, validation, and test sets respectively. Each of the patients was logged to ensure reproducibility of the analysis. Random seeds were fixed across each experiment to ensure deterministic results. The test set was never used for selection of the optimal model, hyperparameters or any early stopping decisions.

## B. SELF-SUPERVISION USING DIFFUSION

## 1) Mathematical Formulation

Denoising Diffusion Probabilistic Models (DDPMs) consist of two parameterized Markov chains: a forward diffusion process that systematically corrupts the data, and a learned reverse process that reconstructs it.

The Forward Process. Given an uncorrupted input image x<sub>0</sub> sampled from the real data distribution q(x), the forward process q gradually adds Gaussian noise over T timesteps. The amount of noise at each step is controlled by a fixed variance schedule $\beta _ { 1 } , \ldots , \beta _ { T } \in ( 0 , 1 )$ . The transition probability for a single step is defined as:

```latex
Algorithm 1 Abdominal CT Image Preprocessing Pipeline
Require: Set of 3D CT Volumes I and corresponding
Masks M for patients P, Target Labels $\begin{array} { r l } { L _ { t a r g e t } } & { { } = } \end{array}$
{6(Liver), 2(R. Kidney), 3(L. Kidney)}
Ensure: Preprocessed 2D slice set S, Patient-level data splits
$( P _ { t r a i n } , P _ { \nu a l } , P _ { t e s t } )$
1: $S \gets \emptyset , P _ { \nu a l i d } \gets \emptyset$
2: for each patient $p \in P$ do
3: Extract 3D volume $I _ { p }$ and mask $M _ { p }$
4: if dim $( I _ { p } ) = \dim ( M _ { p } )$ then
5: $Z $ number of axial slices in $I _ { p }$
6: has_target ← False
7: for z = 1 to Z do
8: $I _ { s l i c e } \gets I _ { p } [ : , : , z ]$
9: $M _ { s l i c e }  \cdot \_ M _ { p } [ : , : , z ]$
10: if $M _ { s l i c e } \cap \mathop { \blacktriangle } _ { t a r g e t } \not = \emptyset$ then
11: has_target ← True
{1. Hounsfield Unit Windowing}
12: $I _ { s l i c e } \gets \operatorname* { m a x } ( \operatorname* { m i n } ( I _ { s l i c e } , 4 0 0 ) , - 1 0 0 )$
{2. Min-Max Normalization}
13: $\begin{array} { r } { \dot { I } _ { s l i c e }  \frac { I _ { s l i c e } - ( - 1 0 0 ) } { 4 0 0 - ( - 1 0 0 ) } } \end{array}$
{3. Spatial Resizing}
14: $I _ { s l i c e } \gets \mathrm { R e s i z e } ( I _ { s l i c e } , 2 5 6 \times 2 5 6 ,$ method =
$\mathrm { \ A r e a ) }$
15: $M _ { s l i c e } \gets \mathrm { R e s i z e } ( M _ { s l i c e } , 2 5 6 \times 2 5 6 ,$ , method =
Nearest)
{4. Class Mapping}
16: $M _ { f i n a l } \gets \mathrm { z e r o s \_ l i k e } ( M _ { s l i c e } )$
17: $\bar { M _ { f i n a l } } \bar { [ M _ { s l i c e } = = 6 ] }  1$ {Map Liver}
18: $\bar { M _ { f i n a l } } [ M _ { s l i c e } \in \{ 2 , 3 \} ]  2$ {Map Kidneys}
19: Add $\left( I _ { s l i c e } , M _ { f i n a l } \right)$ to S
20: end if
21: end for
22: if has_target then
23: Add $p$ to $P _ { \nu a l i d }$
24: end if
25: end if
26: end for
{5. Strict Patient-Level Splitting}
27: $P _ { t r a i n } , P _ { t e m p } \gets \mathrm { S p l i t } ( P _ { \nu a l i d } , \mathrm { r a t i o = 7 0 : 3 0 ) }$
28: $P _ { \nu a l } , P _ { t e s t } \gets \mathrm { S p l i t } ( P _ { t e m p } , \mathrm { r a t i o = 5 0 : 5 0 } )$
29: return $S , P _ { t r a i n } , P _ { \nu a l } , P _ { t e s t }$
```

$$
q ( x _ { t } | x _ { t - 1 } ) = \mathcal { N } ( x _ { t } ; \sqrt { 1 - \beta _ { t } } x _ { t - 1 } , \beta _ { t } I )\tag{1}
$$

The complete forward process is the product of these conditional probabilities. A critical mathematical property of this process is that it allows us to sample $x _ { t }$ at any arbitrary timestep directly from $x _ { 0 }$ without iterating through all previous steps. By defining $\alpha _ { t } = 1 - \beta _ { t }$ and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } } \end{array}$ , the marginal distribution becomes:

$$
q ( x _ { t } | x _ { 0 } ) = \mathcal { N } ( x _ { t } ; \sqrt { \bar { \alpha } _ { t } } x _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) I )\tag{2}
$$

Using the reparameterization trick, we can express this as $x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$ , where $\epsilon \sim \mathcal { N } ( 0 , I )$ . As $t  \infty .$ , the parameter $\bar { \alpha } _ { t } \to 0$ , forcing the data distribution to converge to an isotropic standard normal distribution $\mathcal { N } ( 0 , I )$

The Reverse Process. The objective of the network is to reverse this noise addition to maximize the log-likelihood of generating a real sample, log $p _ { \theta } ( x _ { 0 } )$ . Since the true reversal step $q ( x _ { t - 1 } | x _ { t } )$ is computationally intractable, it is approximated using a neural network $p _ { \theta } \colon$

$$
p _ { \theta } ( x _ { 0 : T } ) = p ( x _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \theta } ( x _ { t - 1 } | x _ { t } )\tag{3}
$$

$$
p _ { \theta } ( x _ { t - 1 } | x _ { t } ) = \mathcal { N } ( x _ { t - 1 } ; \mu _ { \theta } ( x _ { t } , t ) , \Sigma _ { \theta } ( x _ { t } , t ) )\tag{4}
$$

KL Divergence and the True Posterior. Training the network requires minimizing the negative log-likelihood $- \log p _ { \theta } ( x _ { 0 } )$ . Using Bayesian variational inference, this is bounded by minimizing a combination of terms, the most critical being the Kullback-Leibler (KL) divergence. The KL divergence measures the ‘‘distance in probability space’’ between the network’s prediction and the true posterior conditioned on x<sub>0</sub>:

$$
L _ { t } = \mathbb { E } _ { q } \left[ D _ { \mathrm { K L } } ( q ( x _ { t - 1 } | x _ { t } , x _ { 0 } ) \parallel p _ { \theta } ( x _ { t - 1 } | x _ { t } ) ) \right]\tag{5}
$$

Here, $q ( x _ { t - 1 } | x _ { t } , x _ { 0 } )$ represents the forward process utilizing the known original image $x _ { 0 }$ to perfectly calculate the previous step distribution. While using $x _ { 0 }$ during generation is impossible, it provides a perfect target during training to approximate the parameters θ.

Reparameterization to Simplified Loss. Both the true posterior and the predicted distribution are Gaussian distributions: $q ( x _ { t - 1 } | x _ { t } , x _ { 0 } ) = \mathcal { N } ( \tilde { \mu } _ { t } , \tilde { \beta } _ { t } I )$ and $p _ { \theta } ( x _ { t - 1 } | x _ { t } ) = \mathcal { N } ( \mu _ { \theta } , \sigma _ { \theta } ^ { 2 } I )$ Consequently, minimizing their KL divergence simplifies to minimizing the sum of L2 distances between their means:

$$
L _ { t } = \mathbb { E } _ { q } \left[ \frac { 1 } { 2 \sigma _ { t } ^ { 2 } } \lVert \tilde { \mu } _ { t } - \mu _ { \theta } ( x _ { t } , t ) \rVert ^ { 2 } \right]\tag{6}
$$

Through further algebraic reparameterization, rather than predicting the mean $\mu ,$ the neural network $\epsilon _ { \theta } ( x _ { t } , t )$ is trained to directly predict the noise ϵ that was added at timestep t. Removing the variance-weighting factors yields the final, simplified loss function:

$$
L _ { \mathrm { s i m p l e } } = \mathbb { E } _ { t , x _ { 0 } , \epsilon } \left[ \left\| \epsilon - \epsilon _ { \theta } \big ( \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , t \big ) \right\| ^ { 2 } \right] _ { }\tag{7}
$$

This robust, unweighted objective improves training stability and forces the encoder to learn the underlying global anatomical structures required to successfully denoise the image across all possible noise scales.

![](images/e0a2037c6c01cb579beae8ec964b584f01983aaf60a6fc70c780382d5d10bdee.jpg)  
FIGURE 3. DDPM architecture employing U-Net backbone.

## 2) Architecture and Implementation

The DDPM architecture employs a U-Net backbone with several key design choices optimized for transfer to downstream segmentation tasks:

1) Encoder-decoder structure: The network follows the standard U-Net architecture with skip connections, ensuring that pretrained encoder can later integrate seamlessly with a segmentation decoder.

2) Timestep conditioning: The timestep t is embedded using sinusoidal position encodings and inject exclusively at the bottleneck layer. This maintains −1 to 1 structural compatibility with standard U-Nets providing the temporal conditioning necessary for diffusion training.

3) Reduced timesteps: T = 500 timesteps are used rather than the standard 1,000. This reduces computational cost by 2 fold while maintaining sufficient granularity for learning meaningful denoising features.

4) Training data: The DDPM is trained on all available abdominal CT slices without using any annotations. This unsupervised phase sees the full data distribution regardless of which subset will later be used for supervised fine-tuning.

Training proceeded for 500 epochs using the Adam optimizer with learning rate 10<sup>−4</sup>. Reconstruction quality was monitored by visualizing denoised samples at every t = 50 to verify that the model learned meaningful structure at different noise levels.

## C. SUPERVISED FINE-TUNING AND TRANSFER LEARNING

After performing unsupervised DDPM pretraining, the weights learned in the encoder are transferred to downstream

segmentation tasks. This study performs three distinct transfer strategies to understand how diffusion contributes to segmentation performance. These include:

## 1) Transfer Study 1: Random Initialization

Models M1–M3 mentioned in Table 1 represent baseline for the study. The entire U-Net is initialized with random weights (He initialization [63]) and trained end-to-end using only labeled segmentation data. This represents the standard supervised learning approach and provides the performance reference for measuring pretraining benefits.

## 2) Transfer Study 2: Fine-Tune DDPM Encoder

Models M4–M6 as mentioned in Table 1, initialize the encoder with DDPM-pretrained weights while randomly initializing the decoder and segmentation head. During training, all parameters, encoder, decoder, and head; are fine-tuned using the segmentation loss (Equation 8). This strategy allows the network to adapt pretrained anatomical features to the specific requirements of pixel-wise segmentation while preserving learned structural knowledge.

## 3) Transfer Study 3: Frozen DDPM Encoder

Models M7–M9 as mentioned in Table 1 use DDPMpretrained encoders with frozen weights encoder parameters are not updated during segmentation training. Only the decoder and segmentation head learn from labeled data. This experimental condition directly tests whether unsupervised diffusion pretraining creates useful anatomical features without any task-specific adaptation. Strong performance from frozen encoders would provide compelling evidence for learned anatomical priors.

TABLE 1. Model Names and Descriptions.
<table><tr><td rowspan=1 colspan=1>Model Name</td><td rowspan=1 colspan=7>Initialization</td><td rowspan=1 colspan=1>Organ</td></tr><tr><td rowspan=1 colspan=1>M1</td><td rowspan=1 colspan=7>Random</td><td rowspan=12 colspan=1>LiverKidneyMulti OrganLiverKidneyMulti OrganLiverKidneyMulti Organ</td></tr><tr><td rowspan=1 colspan=1>M2</td><td rowspan=4 colspan=7>Random</td></tr><tr><td rowspan=2 colspan=1>M3</td></tr><tr><td rowspan=2 colspan=6>Fine-Tuned Diffusion Encoder</td></tr><tr><td rowspan=1 colspan=1>M4</td></tr><tr><td rowspan=1 colspan=1>M5</td><td rowspan=2 colspan=4>Fine-Tuned Diffusiusi</td><td></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1>Kid</td></tr><tr><td rowspan=1 colspan=1>M6</td><td></td></tr><tr><td rowspan=3 colspan=1>M7</td><td rowspan=3 colspan=5>Erozen Diffusion Enco</td><td rowspan=5 colspan=3>Frozen Diffusion EncoderFrozen Diffusion EncoderFrozen Diffusion Encoder</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>er</td><td rowspan=1 colspan=1>Liver</td></tr><tr><td rowspan=1 colspan=1>M8</td><td rowspan=1 colspan=2>Frozer</td><td rowspan=1 colspan=2>zen Diffusi</td><td></td></tr><tr><td rowspan=1 colspan=1>M9</td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 4) Segmentation Training Details

All segmentation models were trained using the Dice loss function, which directly optimizes the evaluation metric:

$$
L _ { \mathrm { { D i c e } } } = 1 - { \frac { 2 \sum _ { i } p _ { i } g _ { i } + \varepsilon } { \sum _ { i } p _ { i } ^ { 2 } + \sum _ { i } g _ { i } ^ { 2 } + \varepsilon } }\tag{8}
$$

where $p _ { i }$ is the predicted probability for pixel $i , g _ { i }$ is the ground truth binary label, and $\varepsilon = 1 0 ^ { - 5 }$ prevents division by zero. Dice loss is particularly effective for medical segmentation because it handles class imbalance naturally and directly optimizes region overlap.

The study trained three task variants for each of the transfer strategy, (1) single-organ liver segmentation, (2) single-organ kidney segmentation and (3) multi-organ joint segmentation with three output classes. The respective correspondence for each model has been detailed in Table 1. All models were trained for 50 epochs using Adam optimizer with learning rate $1 0 ^ { - 4 }$ , batch size 8, and standard data augmentation comprising random horizontal flips, random rotations (±15<sup>◦</sup>), and random scaling (0.9–1.1×).

## D. LOW-DATA SIMULATION

To evaluate the robustness of diffusion pretraining under data scarcity, this study conducts semantic experiments with restricted fine-tuning data. The DDPM pretraining phase was conducted with the complete data, only the supervisedfinetuning was performed on the restricted label set.

TABLE 2. Model names and percentage of labeled data for Low Data Simulation Study.
<table><tr><td>Model Name</td><td>Percentage of labeled data</td><td>Organ</td></tr><tr><td>M10</td><td>50%</td><td>Liver</td></tr><tr><td>M11</td><td>50%</td><td>Kidney</td></tr><tr><td>M12</td><td>25%</td><td>Liver</td></tr><tr><td>M13</td><td>25%</td><td>Kidney</td></tr><tr><td>M14</td><td>10%</td><td>Liver</td></tr><tr><td>M15</td><td>10%</td><td>Kidney</td></tr></table>

## E. EVALUATION METRICS AND STATISTICAL FRAMEWORK

## 1) Segmentation Quality Metrics

Four metrics were used to systematically evaluate segmentation performance, capturing both region overlap and boundary accuracy. These metrics include:

1) Dice Coefficient [22]: The primary metric for measuring volumetric overlap, ranging from 0 (no overlap) to 1 (perfect agreement).

2) Intersection over Union (IoU) [23]: Also known as the Jaccard index, IoU is more sensitive to errors than Dice, penalizing false positives and false negatives more heavily.

3) 95th Percentile Hausdorff Distance (HD95) [24]: Measures boundary accuracy using the 95th percentile of point-to-surface distances, particularly important for detecting huge boundary errors that Dice might miss.

4) Average Surface Distance (ASD) [25]: Calculates the mean distance between corresponding boundary points, offering a smooth and reliable assessment of the overall quality of the boundary.

## 2) Statistical Significance Testing

The rigorous hypothesis testing employed a non-parametric Wilcoxon signed-rank test for paired comparisons [26], which is suitable for bounded metrics such as Dice scores. To account for multiple comparisons, a Bonferroni correction was applied. Cohen’s d effect sizes [27] quantified practical significance, interpreted as small $( 0 . 2 \leq | d | < 0 . 5 )$ , medium $( 0 . 5 \leq | d | < 0 . 8 )$ , or large $( | d | \geq 0 . 8 )$

## IV. RESULTS

## A. IMPLEMENTATION DETAILS

All experiments were carried out in accordance with the methodology described in Section III. The DDPM pretraining phase employed the entire unlabeled training set of 21 patients from the BTCV dataset with 1000 diffusion timesteps and trained for 500 epochs on NVIDIA A2000 GPUs. For the downstream segmentation tasks, models M1==M9 were trained on the complete labeled training set, which consisted of a 70% patient-level split involving 21 patients. In contrast, models M10–M15 were designed to simulate lowdata regimes, being trained on 50%, 25% and 10% of the labeled data, respectively. All segmentation models implemented Dice loss optimization (Equation 8) alongside Adam optimizer [64] (learning rate $1 0 ^ { - 4 } )$ , a batch size of $^ { 8 , }$ and were trained for 50 epochs while applying standard augmentation techniques, including horizontal flips, $\pm 1 5 ^ { \circ }$ rotation, 0.9–1.1× scaling. The assessment of statistical significance was conducted using Wilcoxon signed-rank tests, applying Bonferroni correction for multiple comparisons $( \alpha = 0 . 0 5 )$ , and effect sizes were measured using Cohen’s d.

## B. QUANTITATIVE PERFORMANCE ANALYSIS

## 1) Comprehensive Segmentation Metrics

Table 3 presents the complete segmentation performance of the 15 models. The models were evaluated on four metrics, being, Dice Coefficient and IoU for volumetric overlap, and HD95 and ASD for boundary precision.

![](images/0921ad7b2d8ae608ca94a469a19c1fff06a9e448d0ade3f2b6f9d01bac9471b3.jpg)  
FIGURE 4. Qualitative Segmentation Predictions Across All 15 Models on Three Representative Test Cases. Axial CT slices with ground truth masks (red: single class representing kidney/liver; red and green for multi class, where red represents liver and green represents kidney) overlaid with predictions from models M1–M15. Columns represent the three test cases; rows show predictions from baseline (M1–M3), fine-tuned diffusion (M4–M6), frozen diffusion (M7–M9), and low-data models (M10–M15). Visual inspection reveals systematic differences in boundary precision, false positive rates, and anatomical plausibility.

TABLE 3. Comprehensive Segmentation Performance Across All Models.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Dice</td><td rowspan=1 colspan=1>IoU</td><td rowspan=1 colspan=1>HD95 (mm)</td><td rowspan=1 colspan=1>ASD (mm)</td></tr><tr><td rowspan=1 colspan=1>M1</td><td rowspan=1 colspan=1>0.751±0.363</td><td rowspan=1 colspan=1>0.703±0.354</td><td rowspan=1 colspan=1>11.76±16.44</td><td rowspan=2 colspan=1>4.38±11.113.75±8.13</td></tr><tr><td rowspan=1 colspan=1>M2</td><td rowspan=1 colspan=1>0.899±0.192</td><td rowspan=1 colspan=1>0.853±0.216</td><td rowspan=1 colspan=1>14.00±27.32</td></tr><tr><td rowspan=1 colspan=1>M3</td><td rowspan=1 colspan=1>0.783±0.225</td><td rowspan=1 colspan=1>0.724±0.236</td><td rowspan=1 colspan=1>15.49±18.26</td><td rowspan=5 colspan=1>4.26±5.691.51±2.082.47±7.551.42±2.556.83±9.07</td></tr><tr><td rowspan=1 colspan=1>M4</td><td rowspan=1 colspan=1>0.928±0.160</td><td rowspan=1 colspan=1>0.891±0.178</td><td rowspan=1 colspan=1>6.50±11.13</td></tr><tr><td rowspan=1 colspan=1>M5</td><td rowspan=1 colspan=1>0.950±0.101</td><td rowspan=1 colspan=1>0.916±0.129</td><td rowspan=1 colspan=1>8.89±25.25</td></tr><tr><td rowspan=1 colspan=1>M6</td><td rowspan=1 colspan=1>0.950±0.071</td><td rowspan=1 colspan=1>0.916±0.081</td><td rowspan=1 colspan=1>6.35±12.92</td></tr><tr><td rowspan=1 colspan=1>M7</td><td rowspan=1 colspan=1>0.806±0.284</td><td rowspan=1 colspan=1>0.740±0.278</td><td rowspan=1 colspan=1>29.04±28.75</td></tr><tr><td rowspan=1 colspan=1>M8</td><td rowspan=1 colspan=1>0.654±0.409</td><td rowspan=1 colspan=1>0.604±0.393</td><td rowspan=1 colspan=1>22.05±28.84</td><td rowspan=1 colspan=1>6.32±12.89</td></tr><tr><td rowspan=1 colspan=1>M9</td><td rowspan=1 colspan=1>0.739±0.249</td><td rowspan=1 colspan=1>0.681±0.244</td><td rowspan=1 colspan=1>23.96±24.28</td><td rowspan=1 colspan=1>6.15±7.97</td></tr><tr><td rowspan=1 colspan=1>M10</td><td rowspan=1 colspan=1>0.922±0.173</td><td rowspan=1 colspan=1>0.884±0.182</td><td rowspan=1 colspan=1>6.44±9.80</td><td rowspan=4 colspan=1>1.47±2.012.12±3.703.29±4.911.72±3.60</td></tr><tr><td rowspan=1 colspan=1>M11</td><td rowspan=1 colspan=1>0.904±0.199</td><td rowspan=1 colspan=1>0.860±0.203</td><td rowspan=1 colspan=1>9.09±14.40</td></tr><tr><td rowspan=1 colspan=1>M12</td><td rowspan=1 colspan=1>0.886±0.193</td><td rowspan=1 colspan=1>0.830±0.204</td><td rowspan=1 colspan=1>14.19±18.35</td></tr><tr><td rowspan=1 colspan=1>M13</td><td rowspan=1 colspan=1>0.941±0.124</td><td rowspan=1 colspan=1>0.905±0.143</td><td rowspan=1 colspan=1>8.08±19.07</td></tr><tr><td rowspan=1 colspan=1>M14</td><td rowspan=1 colspan=1>0.814±0.341</td><td rowspan=1 colspan=1>0.783±0.338</td><td rowspan=1 colspan=1>9.89±23.61</td><td rowspan=2 colspan=1>3.24±10.208.60±15.04</td></tr><tr><td rowspan=1 colspan=1>M15</td><td rowspan=1 colspan=1>0.712±0.363</td><td rowspan=1 colspan=1>0.653±0.364</td><td rowspan=1 colspan=1>27.05±36.32</td></tr></table>

## 2) Statistical Significance Analysis

Three primary statistical comparisons were conducted. The results from these statistical comparisons have been tabulated in Table 4. The first comparison axis was the baseline versus diffusion-pretrained models. The second comparison was between fine-tuned versus frozen encoder configurations, and the third comparison was for data-efficiency in degraded patterns. All comparisons achieved statistical significance (p < 0.05 after Bonferroni correction), with effect sizes ranging from small to large.

## 3) Performance Metric Analysis

For liver segmentation, the Dice Coefficient improved from 0.751±0.363 to 0.928±0.160, which represents a net increase of 23.6% relative to the base mode. Furthermore, variance reduced by over 55% (0.363 to 0.160 standard deviation). These results reveal improvements that exceed statistical significance and the reduction in variance indicates significantly improved reliability and clinical relevance. For multi-organ model (M3 vs M6) the Dice increased from 0.783±0.225 to 0.950±0.071, which represents a mean improvement of 21.3%. Variance reduced by over 68% and thus provides conclusive evidence for model performance.

For liver, diffusion pretraining reduces HD95 by 44.7% (11.76 to 6.50 mm) and ASD by 65.5% (4.38 to 1.51 mm). The multi-organ model had greater improvements with respect to boundary metrics, where, HD95 reduced by over 59.0% from (15.49 to 6.35 mm) and ASD reduced by over 66.6% (4.26 to 1.42 mm). The sub-2 millimeter ADS values are in the range usually considered as inter-rater variability in manual annotations. This suggests that diffusion-pretrained models achieve close to human-level boundary precision.

![](images/9bc8bbabecb041d154daa3443879498a833987d634a862f910a1a68f9f60bd5d.jpg)  
FIGURE 5. Grad-CAM Attention Heatmaps from Bottleneck Layer Across All 15 Models. Heatmaps overlaid on input CT slices for models M1–M15 on three representative cases. Warm colors (red/yellow) indicate high feature activation; cool colors (blue/green) show low activation. The heatmaps reveal where the network’s deepest convolutional features focus when making segmentation predictions.

TABLE 4. Statistical Significance Analysis (Wilcoxon Signed-Rank Test with Bonferroni Correction).
<table><tr><td>Comparison</td><td>Models</td><td>Raw p-value</td><td>Adjusted p-value</td><td>Cohen&#x27;s d</td><td>Effect Size</td></tr><tr><td colspan="6">Baseline vs Fine-Tuned Diffusion</td></tr><tr><td>Liver</td><td>M1 vs M4</td><td> $\overline { { 2 . 6 6 \times 1 0 ^ { - 2 6 } } }$ </td><td> $\overline { { 3 . 2 0 \times 1 0 ^ { - 2 5 } } }$ </td><td>0.529</td><td>Medium</td></tr><tr><td>Kidney</td><td>M2 vs M5</td><td> $2 . 0 0 \times 1 0 ^ { - 1 1 }$ </td><td> $2 . 4 1 \times 1 0 ^ { - 1 0 }$ </td><td>0.271</td><td>Small</td></tr><tr><td>Multi-Organ</td><td>M3 vsM6</td><td> $9 . 4 0 \times 1 0 ^ { - 3 1 }$ </td><td> $1 . 1 3 \times 1 0 ^ { - 2 9 }$ </td><td>0.766</td><td>Medium</td></tr><tr><td colspan="6">Fine-Tuned vs Frozen Encoder</td></tr><tr><td>Liver</td><td>M4vsM7</td><td> $\overline { { { 6 . 3 4 \times 1 0 ^ { - 2 8 } } } }$ </td><td> $\overline { { 7 . 6 1 \times 1 0 ^ { - 2 7 } } }$ </td><td>-0.475</td><td>Small</td></tr><tr><td>Kidney</td><td>M5 vsM8</td><td> $4 . 2 2 \times 1 0 ^ { - 2 7 }$ </td><td> $5 . 0 6 \times 1 0 ^ { - 2 6 }$ </td><td>-0.737</td><td>Medium</td></tr><tr><td>Multi-Organ</td><td>M6 vsM9</td><td> $4 . 7 4 \times 1 0 ^ { - 3 4 }$ </td><td> $5 . 6 9 \times 1 0 ^ { - 3 3 }$ </td><td>-0.897</td><td>Large</td></tr><tr><td colspan="6">Data Efficiency (Liver)</td></tr><tr><td>100% vs 50%</td><td>M4 vs M10</td><td> $\overline { { 4 . 7 9 \times 1 0 ^ { - 8 } } }$ </td><td> $\overline { { 5 . 7 5 \times 1 0 ^ { - 7 } } }$ </td><td>-0.071</td><td>Small</td></tr><tr><td>100% vs 25%</td><td>M4 vs M11</td><td> $5 . 7 8 \times 1 0 ^ { - 1 5 }$ </td><td> $6 . 9 4 \times 1 0 ^ { - 1 4 }$ </td><td>-0.118</td><td>Small</td></tr><tr><td>100% vs 10%</td><td>M4 vs M12</td><td> $7 . 3 8 \times 1 0 ^ { - 2 2 }$ </td><td> $8 . 8 6 \times 1 0 ^ { - 2 1 }$ </td><td>-0.226</td><td>Small</td></tr><tr><td colspan="6">Data Efficiency (Kidney)</td></tr><tr><td>100% vs 50%</td><td>M5 vsM13</td><td> $\overline { { 2 . 0 5 \times 1 0 ^ { - 4 } } }$ </td><td> $2 . 4 6 \times 1 0 ^ { - 3 }$ </td><td>-0.071</td><td>Small</td></tr><tr><td>100% vs 25%</td><td>M5 vsM14</td><td> $4 . 6 2 \times 1 0 ^ { - 1 4 }$ </td><td> $5 . 5 4 \times 1 0 ^ { - 1 3 }$ </td><td>-0.436</td><td>Small</td></tr><tr><td>100% vs 10%</td><td>M5 vs M15</td><td> $2 . 2 0 \times 1 0 ^ { - 2 5 }$ </td><td> $2 . 6 4 \times 1 0 ^ { - 2 4 }$ </td><td>-0.686</td><td>Medium</td></tr></table>

It is important to emphasize that the noted increase in boundary measures compared to Dice indicates an important discovery – namely, the fact that diffusion pre-training not only improves the number of correct pixel classification (and thus results in a linear increase in the value of Dice), but also changes the distribution of the mistakes. While baseline networks show completely random mistakes and non-uniform boundaries, resulting in a high HD95 value with low Dice values, diffusion networks create spatially consistent predictions, with plausible anatomy.

The frozen encoder experiments (M7–M9) provide the most compelling evidence that DDPM pretraining learns genuine anatomical structure rather than generic image features. The organ-specific results reveal nuanced insights:

1) Liver (M7): A Dice of 0.806±0.284 represents 86.9% of the fine-tuned performance (M4: 0.928), with the frozen encoder outperforming the random baseline (M1: 0.751) by 7.3%. This demonstrates that unsupervised diffusion pretraining alone; without ever seeing segmentation labels; learns liver-specific features sufficient for reasonable segmentation. The Cohen’s d = −0.475 (M4 vs M7) indicates a small-to-medium performance gap, suggesting that task-specific finetuning provides incremental rather than transformative benefit.

2) Kidney (M8): The dramatic failure of the frozen kidney encoder (Dice = 0.654, worse than baseline M2: 0.899) with Cohen’s d = −0.737 (medium effect) reveals that kidney segmentation requires more precise, task-specific features than liver. We hypothesize this stems from anatomical differences: kidneys are smaller, bilaterally symmetric, and have more complex beanshaped morphology with the renal hilum requiring fine-grained boundary precision. The diffusion-learned global structural priors are insufficient without supervised adaptation.

3) Multi-Organ (M9): The large effect size (Cohen’s d = −0.897) between frozen and fine-tuned multi-organ models indicates that organ disambiguation; distinguishing liver from kidney despite similar tissue densities; requires supervised signal that pure generative pretraining cannot provide. However, M9 still outperforms the random baseline M3 on variance (0.249 vs 0.225), suggesting partial anatomical understanding.

In all the analyses conducted, diffusion pretraining invariably leads to variance reduction: liver 55.9%, kidney 47.3%, and multi-organ 68.4%. This variance reduction is significant from a clinical point of view. Consider an analysis with a mean Dice score of 0.95 and a standard deviation of 0.30. Such a system will definitely fail in 5% of the test cases, making it necessary to manually check each prediction done. On the other hand, M6 demonstrates a very narrow distribution (0.950 ± 0.071), and thus enables safe automated screening where only anomalies need human intervention. The reduction in variance indicates that diffusion-based features include organ invariant information.

## 4) Low-Data Performance

Organ-Specific Data Efficiency Patterns. The low-data experiments (M10–M15) reveal striking differences in how liver and kidney segmentation degrade under label scarcity. Table 5 isolates these models for focused analysis:

TABLE 5. Data Efficiency Analysis (Dice Coefficient Only).
<table><tr><td>Model</td><td>Organ</td><td>Labeled Data</td><td>Dice</td><td>% Performance</td></tr><tr><td>M4</td><td>Liver</td><td>100%</td><td>0.928±0.160</td><td>100.0%</td></tr><tr><td>M10</td><td>Liver</td><td>50%</td><td>0.922±0.173</td><td>99.4%</td></tr><tr><td>M11</td><td>Liver</td><td>25%</td><td>0.904±0.199</td><td>97.4%</td></tr><tr><td>M12</td><td>Liver</td><td>10%</td><td>0.886±0.193</td><td>95.5%</td></tr><tr><td>M5</td><td>Kidney</td><td>100%</td><td>0.950±0.101</td><td>100.0%</td></tr><tr><td>M13</td><td>Kidney</td><td>50%</td><td>0.941±0.124</td><td>99.1%</td></tr><tr><td>M14</td><td>Kidney</td><td>25%</td><td>0.814±0.341</td><td>85.7%</td></tr><tr><td>M15</td><td>Kidney</td><td>10%</td><td>0.712±0.363</td><td>75.0%</td></tr></table>

Liver segmentation shows impressive data efficiency, maintaining 99.4% of its full performance even when it only gets half of the labeled data. It can still do well with just 10% of the labels (approximately 3–4 annotated patients). Looking at the results in Table 4, with Cohen’s d between −0.071 to −0.226, we see that performance dips gradually. Hence the features learned through diffusion work great for livers and are very adaptable to similar tasks.

For kidneys, once the dataset drops below 50%, performance takes a nosedive. With 25% of the data, it only keeps 85.7% of its ability, dropping further to 75.0% at 10%. As the data shrink, The Cohen’s d progression (−0.071 to −0.436 to −0.686) indicates accelerating degradation. In addition, the larger variance seen at lower data points indicates some serious unpredictability (M15: 0.363 standard deviation vs M5: 0.101), suggesting potential problems with prediction accuracy. This extra vulnerability for kidney segmentation could be due to the kidneys being more complex. Their bilateral symmetry, changing positions, and detailed internal structures probably need far more training samples to obtain reliable results.

Liver segmentation achieve a Dice score over 0.92 with only 50% labeled data, that means you only really need around 17 to 18 annotated CT scans rather than 35. Factoring in annotation costs of \$100-\$150 per volume and around 45- 60 minutes of a radiologist’s time, going with less labeled data saves roughly 50% on cash (\$3,500 vs \$1,750) and time (26 hours versus 13 hours), all while barely giving up any performance accuracy.

## C. QUALITATIVE VISUAL ANALYSIS

Fig. 4 presents segmentation predictions from all 15 models on three representative test cases selected to demonstrate: (1) typical anatomy with clear boundaries, (2) challenging anatomy with irregular liver morphology, and (3) a case with adjacent liver-kidney boundaries prone to organ confusion.

## D. INTERPRETABILITY ANALYSIS VIA GRAD-CAM

To elucidate the mechanistic basis for diffusion pretraining’s performance improvements, we applied Mask-Guided Gradient-weighted Class Activation Mapping (Grad-CAM) to the bottleneck layer of all models. Standard Grad-CAM often fails in dense prediction tasks due to background dominance; therefore, gradients were backpropagated explicitly through ground-truth masks to isolate attention maps specifically responsible for predicting target organs. Fig. 5 presents these attention visualizations across all 15 models on the same three test cases as Fig. 4.

One interesting phenomenon observed in the attention maps of the diffusion-pretrained network was the presence of negative encoding or a stencil effect at the bottleneck. Unlike baseline models that predominantly attend to internal organ textures, the diffusion pretrained encoder exhibited high activation in the surrounding anatomical context, while suppressing activations within the organ itself. This strongly suggests that the diffusion pretraining shifts the network’s delineation strategy from local texture-matching to global contextual boundary-delineation. It identifies an organ by confidently mapping the diverse anatomical structures that surround it.

Grad-CAM analysis supports the main idea that diffusion pretraining adds global anatomical priors to encoder networks, changing how features are learned. Moving from scattered baseline attention to confined diffusion attention means DDPM pretraining makes the network understand organ placement along with their looks. This spatial info acts as a strong regularizer during fine-tuning. Thus, it prevents overfitting to local textures and lets the model make sensible, anatomy-based predictions.

The organ-specific differences in frozen encoder performance (liver: 86.9% retention, kidney: catastrophic failure) suggest that anatomical complexity and organ size influence the quality of diffusion-learned priors. Large, spatially extended organs with relatively simple topology (liver: single irregular polyhedron) may be easier to learn unsupervised than small, bilaterally symmetric organs with complex morphology (kidneys: paired bean-shaped structures with intricate hilum). This insight may guide future work on selective pretraining strategies or hybrid approaches where different organ types receive customized pretraining.

## V. DISCUSSION

This study proves that unsupervised pre-training using diffusion has the capacity to change the way medical image segmentation works by introducing anatomical information into encoders before performing any supervised optimization. These results support the core premise of this work: generative models can serve as unsupervised learners of anatomical properties used in later discriminative tasks when they have been trained for image reconstruction.

## A. PRINCIPAL FINDINGS AND MECHANISTIC INTERPRETATION

The consistent improvements achieved in Dice score, IoU, HD95, and ASD metrics show that diffusion pre-training helps overcome the main problems faced by randomly initialized U-Nets. In supervised training, the goal is focused on achieving high pixel-wise accuracy, and, therefore, the network tries to match local textures while providing little consideration of the wider anatomical context. This approach is clearly visible from the distribution of the Grad-CAM maps in the baseline models.

But diffusion pretraining shifts this around. The goal here is predicting and eliminating Gaussian noise across a thousand steps. That doesn’t work just by matching local textures. To handle a messed-up liver image at step 500, say, the net needs to understand that livers have certain shapes and typical spots where they are anatomically located; and what intensities those spots should have. As a result, the encoder learns various structural bits: simple details like texture and edges; more complex things like liver lobes and vessels; and, finally, the complete anatomical setup.

When transferred to segmentation, these pretrained features provide powerful priors. The inverse bounded Grad-CAM attention patterns in M4–M6 reveal that diffusion encoders inherently suppress activations in anatomically implausible regions; not because supervision taught them to, but because the denoising objective required learning spatial constraints. This explains the disproportionate improvement in boundary metrics (HD95: −44.7% to −68.3%, ASD: −65.5% to −74.3%) relative to Dice: diffusion features encode smooth, anatomically coherent boundaries as fundamental structural units, whereas baseline models must infer boundaries indirectly from pixel-wise classification gradients.

## B. INFERENCES FROM FROZEN ENCODER EXPERIMENT

The empirical results with the frozen encoder represent the most convincing evidence. M7 reaches 86.9% of M4 performance in spite of the absence of segmentation labels which shows that diffusion pre-training captures some real anatomical structures and not just general image features which are applicable to different tasks.

Differences in performance of the frozen encoders designed for particular organs (positive results in case of liver and negative in case of kidney) represent some interesting details. It is hypothesized that the reasons behind the differences have something to do with structural complexities:

Properties of the liver which make it easy to learn in an unsupervised way:

• Large spatial footprint (spans 10–15 axial slices)

• Unique anatomical position (only organ in right upper quadrant)

• Relatively simple topology (single connected component, though irregular)

• High visual distinctiveness (adjacent to low-density lung, fat)

Properties of the kidney which make unsupervised learning difficult:

• Small spatial footprint (5–7 axial slices)

• Bilateral symmetry requiring paired representation learning

• Complex morphology (bean-shaped with concave hilum, intricate vasculature)

• Variable positioning (can shift 2–3 cm with respiration)

• Lower visual contrast (surrounded by psoas muscle, perirenal fat with similar densities)

It shows that pre-training via unsupervised diffusion is particularly beneficial for large organs with unique anatomy, while the smaller organs or organs with different anatomy gain from being subjected to supervised training. Future research should target organ-specific pre-training methods or hybrid methods where diffusion-based features are combined with specific supervised information. Increasing the complexity of the model and training it using labels for the whole abdomen might also bring about some improvement.

## C. ANALYSIS OF VARIANCE REDUCTION

While mean performance improvements are scientifically important, the 47–68% variance reduction is clinically more significant. In deployment scenarios compared to peak performance, metrics like consistency across diverse anatomies, imaging protocols, and patient populations often matters more. A model achieving a Dice of 0.95±0.30 requires manual verification of every prediction to catch the 5% catastrophic failures; one achieving 0.95±0.07 enables automated screening where only outliers trigger human review.

The variance collapse in diffusion-pretrained models suggests they encode anatomical invariants robust to patientspecific variations. We hypothesize three mechanisms:

1) Distributional priors suppress outliers: Diffusion models learn the data distribution during pretraining, hence when fine-tuning on segmentation the encoders weight represent a degree of distribution knowledge. Hence, predictions deviating from typical anatomy receive lower confidence.

2) Global structural constraints prevent local overfitting: As seen from the Grad-CAM analysis, diffusion used a stencil effect, which intern learnt the global anatomical relationships. It prevents diffusion from learning spurious correlations in limited training data, therefore a prediction is penalized not only for pixelwise loss but also for violating learned spatial priors.

3) Multi-scale feature hierarchies enable graceful degradation: Diffusion encoders learn features at multiple spatial scales (due to the multi-timestep denoising objective). When encountering unusual anatomy, the model can fall back to coarse-scale features (approximate organ position) even if fine-scale features (precise boundary texture) prove unreliable. Baseline models lack this hierarchical fallback mechanism.

## D. ANALYSIS OF DATA EFFICIENCY

The low-data experiments reveal that diffusion pretraining’s value compounds when labeled data is scarce. Maintaining 99.4% liver performance with 50% labels and 95.5% with only 10% labels demonstrates that learned anatomical priors can partially substitute for supervised examples. In a standard learning paradigm a model must learn everything from labels, to what organs look like, where they are located and how they vary across patients. In low data regimes, the performance of traditional models collapse, variance explores and predictions become clinically useless.

Diffusion pretraining breaks this dependency, where the encoder is initialized with rich anatomical priors such as organ shape, location and variability. The limited segmentation labels need only refine these priors for task-specific discrimination, rather than learning anatomical structure from scratch. This explains the small Cohen’s d effect sizes for liver data efficiency (d = −0.071 to −0.226). Performance degradation is minimal because the hard work of anatomical learning occurred during unsupervised pretraining.

The organ-specific degradation patterns (liver: graceful, kidney: catastrophic) likely reflect anatomical complexity interacting with data efficiency. Liver’s large spatial extent and distinctive position may enable robust unsupervised learning, allowing fine-tuning with minimal supervision. Kidney’s small size, bilateral symmetry, and morphological complexity may require more extensive supervised signals to anchor the diffusion priors to task-specific features. This suggests data efficiency is not uniform across anatomy. Organs with favorable structural characteristics may enable extreme labe reduction, while complex structures still benefit substantially from richer supervision.

## VI. LIMITATIONS

Evaluation focused on liver and kidney which are two anatomically favorable organs (large, well-contrasted). Generalization to smaller structures (pancreas, adrenal glands), pathological tissues (tumors with irregular boundaries), or organs with lower contrast (spleen, bowel) requires validation. The failure of the frozen encoder for kidney suggest that organ-specific characteristics do influence diffusion pretraining effectiveness.

Furthermore, all experiments used BTCV dataset sourced from a single institution. Conducting cross-dataset validation, which involves different scanners, protocols and patient demographics, is crucial for evaluating true generalization beyond the training distribution. The observed reduction in variance may partially reflect the memorization of datasetspecific characteristics rather than pure anatomical learning.

Our slice-based approach compromises the volumetric context. Although the segmentation of individual axial slices is performed with precision, the model is unable to enforce 3D anatomical constraints. An extension to 3D diffusion models can capture the volumetric relationships, thereby enhancing robustness and allowing for accurate 3D boundary metrics.

While Grad-CAM visualizations offer interpretable attention maps, they are approximate explanations, illustrating which areas the network considers significant. The bounded attention patterns observed in diffusion models may either represent genuine anatomical priors or suggest the presence of learned shortcuts.

Future research directions include extending to 3D volumetric diffusion, segmentation of pathological tissues, multimodal pretraining, and investigations into cross-dataset generalization.

## VII. CONCLUSION

This study illustrates that unsupervised diffusion-based pretraining effectively integrates robust and interpretable anatomical features into encoder networks, fundamentally altering the approach to medical image segmentation from a focus on texture-matching to anatomy-aware prediction. The evidence supporting this transformation is both multifaceted and convergent:

Quantitative: There are statistically significant enhancements across all metrics (Dice: +5.7% to +23.6%, p < 10<sup>−10</sup>; boundary precision: -38% to -74%), accompanied by medium-to-large effect sizes and a notable reduction in variance (47-68

Qualitative: Visual assessments indicate that diffusion models produce smooth, anatomically plausible boundaries while preserving global structural coherence. This eliminates the scattered errors typically associated with baseline models.

Mechanistic: Grad-CAM visualizations reveal that diffusion encoders develop bounded, organ-specific attention patterns even when in a frozen state. This thereby demonstrating that unsupervised anatomical learning takes place during the pretraining phase.

Practical: The remarkable data efficiency (achieving 95.5% liver performance with only 10% of labels) addresses significant challenges to clinical implementation in resource-limited environments.

Achieving 87% of fine-tuned liver performance without utilizing labels serves as compelling evidence that the generation-as-pretext-task is effective. The denoising objective forces networks to internalize anatomical structures as a prerequisite for successful reconstruction. When these learned priors are applied to segmentation tasks, they provide substantial regularization, mitigating the risk of overfitting to misleading texture correlations and enables robust generalization across diverse anomalies.

In addition to its direct clinical applications, this works establishes a broader principle: generative models trained on unsupervised reconstruction learn structural representations that are applicable to discriminative dense prediction tasks. As diffusion models progress, their role as unsupervised feature learners for spatial understanding tasks in medical imaging and potentially other domains deserve increasing attention from both research and clinical communities.

## ACKNOWLEDGMENT

The authors would like to acknowledge Manipal Institute of Technology Bengaluru, Manipal Academy of Higher Education, Manipal, India, for providing the necessary facilities and computational support to conduct this research. We express our sincere gratitude to the creators of the Beyond the Cranial Vault (BTCV) dataset [59] for making their data publicly accessible. Additionally, we would like to thank the develop ers of PyTorch, CUDA, and other open-source packages that enabled our model development and experiments.

## REFERENCES

[1] Y. Deng, Y. Sun, Y. Zhu, Y. Xu, Q. Yang, S. Zhang, Z. Wang, J. Sun, W. Zhao, X. Zhou, and K. Yuan, ‘‘A new framework to reduce doctor’s workload for medical image annotation,’’ IEEE Access, vol. 7, pp. 107 097– 107 104, 2019.

[2] A. Lawley, R. Hampson, K. Worrall, and G. Dobie, ‘‘A cost focused framework for optimizing collection and annotation of ultrasound datasets,’ Biomedical Signal Processing and Control, vol. 92, p. 106048, 2024.

[3] C. Qu, T. Zhang, H. Qiao, Y. Tang, A. Yuille, and Z. Zhou, ‘‘AbdomenAtlas-8K: Annotating 8,000 CT volumes for multi-organ seg mentation in three weeks,’’ Preprint, 2023.

[4] A. Schmidt, P. Morales-Álvarez, and R. Molina, ‘‘Probabilistic modeling of inter- and intra-observer variability in medical image segmentation,’ in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 21 040–21 049.

[5] F. Yang, G. Zamzmi, S. Angara, S. Rajaraman, A. Aquilina, Z. Xue, S. Jaeger, E. Papagiannakis, and S. Antani, ‘‘Assessing inter-annotator agreement for medical image segmentation,’’ IEEE Access, vol. 11, pp. 21 300–21 312, 2023.

[6] P. Roshanzamir, H. Rivaz, J. Ahn, H. Mirza, N. Naghdi, M. Anstruther, M. Battié, M. Fortin, and Y. Xiao, ‘‘How inter-rater variability relates to aleatoric and epistemic uncertainty: a case study with deep learning-based paraspinal muscle segmentation,’’ arXiv:2308.06964, pp. 74–83, 2023.

[7] M. Riera-Marín et al., ‘‘Calibration and uncertainty for multirater volume assessment in multiorgan segmentation (CURVAS) challenge results,’ Computers in Biology and Medicine, vol. 197, p. 111024, 2025.

[8] O. Ronneberger, P. Fischer, and T. Brox, ‘‘U-Net: Convolutional networks for biomedical image segmentation,’’ in International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI). Cham: Springer International Publishing, 2015, pp. 234–241.

[9] R. Azad, E. Aghdam, A. Rauland, Y. Jia, A. Avval, A. Bozorgpour, S. Karimijafarbigloo, J. Cohen, E. Adeli, and D. Merhof, ‘‘Medical image segmentation review: The success of U-Net,’’ IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, pp. 10 076–10 095, 2022.

[10] N. Ibtehaz and M. Rahman, ‘‘MultiResUNet: Rethinking the U-Net architecture for multimodal biomedical image segmentation,’’ Neural Networks, vol. 121, pp. 74–87, 2019.

[11] T. Hussain and H. Shouno, ‘‘MAGRes-UNet: Improved medical image segmentation through a deep learning paradigm of multi-attention gated residual U-Net,’’ IEEE Access, vol. 12, pp. 40 290–40 310, 2024.

[12] J. Chen, J. Mei, X. Li, Y. Lu, Q. Yu, Q. Wei, X. Luo, Y. Xie, E. Adeli, Y. Wang, M. Lungren, S. Zhang, L. Xing, L. Lu, A. Yuille, and Y. Zhou, ‘‘TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers,’’ Medical Image Analysis, vol. 97, p. 103280, 2024.

[13] Z. Zhou, M. Siddiquee, N. Tajbakhsh, and J. Liang, ‘‘UNet++: Redesigning skip connections to exploit multiscale features in image segmentation,’ IEEE Transactions on Medical Imaging, vol. 39, pp. 1856–1867, 2019.

[14] L. Dai, M. Johar, and M. Alkawaz, ‘‘Review of semi-supervised medical image segmentation based on the U-Net,’’ Academic Journal of Science and Technology, 2024.

[15] J. Ho, A. Jain, and P. Abbeel, ‘‘Denoising diffusion probabilistic models,’ in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 6840–6851.

[16] F.-A. Croitoru, V. Hondru, R. T. Ionescu, and M. Shah, ‘‘Diffusion models in vision: A survey,’’ IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, pp. 10 850–10 869, 2022.

[17] A. Kazerouni, E. K. Aghdam, M. Heidari, R. Azad, I. Hacihaliloglu, and D. Merhof, ‘‘Diffusion models in medical imaging: A comprehensive survey,’’ Medical image analysis, vol. 88, p. 102846, 2022.

[18] Y. Huang, J. Huang, Y. Liu, M. Yan, J. Lv, J. Liu, W. Xiong, H. Zhang, S. Chen, and L. Cao, ‘‘Diffusion model-based image editing: A survey,’’ IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, pp. 4409–4437, 2024.

[19] C. F. Higham, D. J. Higham, and P. Grindrod, ‘‘Diffusion models for generative artificial intelligence: An introduction for applied mathematicians,’ SIAM Rev., vol. 67, pp. 607–623, 2025.

[20] H. Chung and J.-C. Ye, ‘‘Score-based diffusion models for accelerated mri,’’ Medical image analysis, vol. 80, p. 102479, 2021.

[21] V. Purma, S. Srinath, S. Srirangarajan, A. Kakkar, and A. Prathosh, ‘‘Genselfdiff-his: generative self-supervision using diffusion for histopathological image segmentation,’’ IEEE Transactions on Medical Imaging, vol. 44, no. 2, pp. 618–631, 2024.

[22] L. R. Dice, ‘‘Measures of the amount of ecologic association between species,’’ Ecology, vol. 26, no. 3, pp. 297–302, 1945.

[23] P. Jaccard, ‘‘Étude comparative de la distribution florale dans une portion des Alpes et des Jura,’’ Bulletin de la Société Vaudoise des Sciences Naturelles, vol. 37, pp. 547–579, 1901.

[24] F. Hausdorff, Grundzüge der Mengenlehre. Leipzig: Veit & Comp., 1914.

[25] T. Heimann and H.-P. Meinzer, ‘‘Statistical shape models for 3D medical image segmentation: A review,’’ Medical Image Analysis, vol. 13, no. 4, pp. 543–563, 2009.

[26] R. A. Fisher, Statistical Methods for Research Workers. Edinburgh: Oliver and Boyd, 1925.

[27] J. Cohen, Statistical Power Analysis for the Behavioral Sciences, 2nd ed Lawrence Erlbaum Associates, 1988.

[28] Z. Guo, X. Li, H. Huang, N. Guo, and Q. Li, ‘‘Deep learning-based image segmentation on multimodal medical imaging,’’ IEEE Transactions on Radiation and Plasma Medical Sciences, vol. 3, pp. 162–169, 2019.

[29] R. Gu, G. Wang, T. Song, R. Huang, M. Aertsen, J. Deprest, S. Ourselin, T. Vercauteren, and S. Zhang, ‘‘Ca-net: Comprehensive attention convolutional neural networks for explainable medical image segmentation,’’ IEEE Transactions on Medical Imaging, vol. 40, pp. 699–711, 2020.

[30] H. Roth, C. Shen, H. Oda, M. Oda, Y. Hayashi, K. Misawa, and K. Mori, ‘‘Deep learning and its application to medical image segmentation,’’ ArXiv, vol. abs/1803.08691, 2018.

[31] M. A. Abdou, ‘‘Literature review: efficient deep neural networks techniques for medical image analysis,’’ Neural Computing and Applications, vol. 34, pp. 5791 – 5812, 2022.

[32] Y. Xu, R. Quan, W. Xu, Y. Huang, X. Chen, and F. Liu, ‘‘Advances in medical image segmentation: A comprehensive review of traditional, deep learning and hybrid approaches,’’ Bioengineering, vol. 11, 2024.

[33] F. Milletari, N. Navab, and S. A. Ahmadi, ‘‘V-Net: Fully convolutional neural networks for volumetric medical image segmentation,’’ in 2016 Fourth International Conference on 3D Vision (3DV). IEEE, 2016, pp. 565–571.

[34] N. Siddique, S. Paheding, C. Elkin, and V. Devabhaktuni, ‘‘U-Net and its variants for medical image segmentation: A review of theory and applications,’’ IEEE Access, vol. 9, pp. 82 031–82 057, 2020.

[35] F. Isensee, P. Jaeger, S. Kohl, J. Petersen, and K. Maier-Hein, ‘‘nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,’’ Nature Methods, vol. 18, pp. 203–211, 2020.

[36] A. Lin, B. Chen, J. Xu, Z. Zhang, G. Lu, and D. Zhang, ‘‘DS-TransUNet: Dual swin transformer U-Net for medical image segmentation,’’ IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1–15, 2021.

[37] A. Shaker, M. Maaz, H. Rasheed, S. Khan, M. Yang, and F. Khan, ‘‘UN-ETR++: Delving into efficient and accurate 3D medical image segmentation,’’ IEEE Transactions on Medical Imaging, vol. 43, pp. 3377–3390, 2022.

[38] L. Alzubaidi, J. Bai, A. Al-Sabaawi, J. I. Santamaría, A. Albahri, B. S. Aldabbagh, M. Fadhel, M. Manoufali, J. Zhang, A. H. Al-timemy, Y. Duan, A. Abdullah, L. Farhan, Y. Lu, A. Gupta, F. Albu, A. Abbosh, and Y. Gu, ‘‘A survey on deep learning tools dealing with data scarcity: definitions, challenges, solutions, tips, and applications,’’ Journal ofBig Data, vol. 10, pp. 1–82, 2023.

[39] M. A. Bansal, D. R. Sharma, and D. M. Kathuria, ‘‘A systematic review on data scarcity problem in deep learning: Solution and applications,’’ ACM Computing Surveys (CSUR), vol. 54, pp. 1 – 29, 2022.

[40] S. E. Whang and J.-G. Lee, ‘‘Data collection and quality challenges for deep learning,’’ Proceedings ofthe VLDB Endowment, vol. 13, pp. 3429 – 3432, 2020.

[41] C. Janiesch, P. Zschech, and K. Heinrich, ‘‘Machine learning and deep learning,’’ Electronic Markets, vol. 31, pp. 685 – 695, 2021.

[42] J. Gui, T. Chen, J. Zhang, Q. Cao, Z. Sun, H. Luo, and D. Tao, ‘‘A survey on self-supervised learning: Algorithms, applications, and future trends,’’ IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 9052–9071, 2024.

[43] ‘‘Dive into the details of self-supervised learning for medical image analysis,’’ Medical Image Analysis, 2023.

[44] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, ‘‘A simple framework for contrastive learning of visual representations,’’ in Proceedings of

the 37th International Conference on Machine Learning (ICML), 2020. [Online]. Available: https://arxiv.org/abs/2002.05709

[45] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, ‘‘Momentum contrast for unsupervised visual representation learning,’’ in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. [Online]. Available: https://arxiv.org/abs/1911.05722

[46] L. Jing and Y. Tian, ‘‘Self-supervised visual feature learning with deep neural networks: A survey,’’ IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020. [Online]. Available: https: //arxiv.org/abs/1902.06162

[47] G. E. Hinton and R. R. Salakhutdinov, ‘‘Reducing the dimensionality of data with neural networks,’’ Science, vol. 313, no. 5786, pp. 504–507, 2006.

[48] X. Chen, L. Yao, and Y. Zhang, ‘‘Residual attention U-Net for automated multi-class segmentation of COVID-19 chest CT images,’’ Preprint, 2019.

[49] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, ‘‘Masked autoencoders are scalable vision learners,’’ in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. [Online]. Available: https://arxiv.org/abs/2111.06377

[50] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, ‘‘BERT: Pretraining of deep bidirectional transformers for language understanding,’’ in Proceedings of NAACL-HLT, 2019. [Online]. Available: https://arxiv. org/abs/1810.04805

[51] X. Chen, H. Fan, R. Girshick, and K. He, ‘‘Improved baselines with momentum contrastive learning,’’ arXiv:2003.04297, 2020. [Online]. Available: https://arxiv.org/abs/2003.04297

[52] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, ‘‘Extracting and composing robust features with denoising autoencoders,’’ in Proceedings ofthe 25th International Conference on Machine Learning (ICML), 2008.

[53] G. Müller-Franzes, J. Niehues, F. Khader, S. T. Arasteh, C. Haarburger, C. Kuhl, T. Wang, T. Han, S. Nebelung, J. N. Kather, and D. Truhn, ‘‘A multimodal comparison of latent denoising diffusion probabilistic models and generative adversarial networks for medical image synthesis,’’ Scien tific Reports, vol. 13, 2022.

[54] K. Gong, K. A. Johnson, G. E. Fakhri, Q. Li, and T. Pan, ‘‘Pet image denoising based on denoising diffusion probabilistic model,’’ European Journal of Nuclear Medicine and Molecular Imaging, vol. 51, pp. 358– 368, 2022.

[55] W. Wang, J. Bao, W. gang Zhou, D. Chen, D. Chen, L. Yuan, and H. Li, ‘‘Semantic image synthesis via diffusion models,’’ ArXiv, vol. abs/2207.00050, 2022.

[56] D. Baranchuk, I. Rubachev, A. Voynov, V. Khrulkov, and A. Babenko, ‘‘Label-efficient semantic segmentation with diffusion models,’ arXiv:2112.03126, 2021.

[57] T. Amit, T. Shaharbany, and L. Wolf, ‘‘SegDiff: Image segmentation with diffusion probabilistic models,’’ arXiv:2112.00390, 2021.

[58] J. Wu, H. Fu, J. Xu, and Y. Zhang, ‘‘MedSegDiff: Medical image segmentation with diffusion probabilistic models,’’ arXiv:2301.11798, 2023.

[59] E. Gibson, F. Giganti, Y. Hu, E. Bonmati, S. Bandula, K. Gurusamy, B. Davidson, S. P. Pereira, M. J. Clarkson, and D. C. Barratt, ‘‘Multiorgan abdominal ct reference standard segmentations,’’ 2018. [Online]. Available: https://doi.org/10.5281/zenodo.1169361

[60] S. Pan, C.-W. Chang, T. Wang, J. Wynne, M. Hu, Y. Lei, T. Liu, P. Patel, J. Roper, and X. Yang, ‘‘Abdomen CT multi-organ segmentation using token-based MLP-Mixer,’’ Medical Physics, vol. 50, no. 5, pp. 3027–3038, 2023.

[61] J. T. Bushberg, J. A. Seibert, E. M. Leidholdt, and J. M. Boone, The Essential Physics of Medical Imaging, 3rd ed. Lippincott Williams & Wilkins, 2011.

[62] G. Litjens, T. Kooi, B. E. Bejnordi, A. A. A. Setio, F. Ciompi, M. Ghafoorian, J. A. W. M. van der Laak, B. van Ginneken, and C. I. Sánchez, ‘‘A survey on deep learning in medical image analysis,’’ Medical Image Analysis, vol. 42, pp. 60–88, 2017.

[63] K. He, X. Zhang, S. Ren, and J. Sun, ‘‘Delving deep into rectifiers: Surpassing human-level performance on imagenet classification,’’ in Proceedings ofthe IEEE international conference on computer vision, 2015, pp. 1026– 1034.

[64] D. P. Kingma and J. Ba, ‘‘Adam: A method for stochastic optimization,’ arXiv preprint arXiv:1412.6980, 2014.

## DECLARATIONS

## A. AUTHOR CONTRIBUTIONS STATEMENT

A. G. conceived the study, designed the methodology, implemented the DDPM architecture and transfer learning, performed experiments, analyzed the results, prepared the figures, and wrote the initial manuscript draft. D. G. aided in tabulating the conducted experiments results, analyzed the results, reviewed the figures and tables, and aided in the initial manuscript draft. S. B., S. A., and T. K. M. supervised the research, provided conceptual guidance and contributed to the interpretation of results, reviewed and edited the manuscript, and approved the final version.

## B. ETHICS STATEMENT

This article does not contain any studies with human participants or animals performed by any of the authors.

## ADDITIONAL INFORMATION

• Funding: Open access funding is provided by Manipal Academy of Higher Education.

• Conflicts of Interest: The authors declare that they have no competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

• Data Availability: The experimental evaluations were performed using the Beyond the Cranial Vault (BTCV) multi-organ abdominal CT dataset, which is publicly accessible via the official Synapse repository at https: //www.synapse.org/#!Synapse:syn3193805.

• ORCID iDs: Akshat G: https://orcid.org/0009-0000-9244-1513.

![](images/5cee05004f7dbc11f5fa5aa3c3786d3c248336b3b083ac3afbe59dc7c7a600c0.jpg)  
AKSHAT G is currently pursuing the B.Tech Honors degree in computer science (minor in A.I. for Healthcare) with Manipal Institute of Technology Bengaluru, India. He has worked as an ML Research Intern, developing hierarchical deep learning architectures for multi label defect classification, and build end to end consumer products for his startup. His research interests include semi-supervised learning, self-supervised learning, data-centric AI, computer vision, deep learn-  
ing, foundational AI and focus on developing scalable AI applications for real world challenges.

![](images/f774ad9c9c8a4c5dde1845137dd3bc7d44a89731c071441c25e46889683c1b33.jpg)  
els.

DIVYANSH GUPTA is currently pursuing a thirdyear B.Tech. degree in Computer Science and Engineering at the Manipal Institute of Technology, Bengaluru, Manipal Academy of Higher Education (MAHE), Manipal, India. He is passionate about artificial intelligence and focuses on creating scalable machine learning solutions for real-world challenges. His areas of interest include machine learning and deep learning, particularly in the development of robust and application-oriented mod-

![](images/4a9213dcdfbd9e98f2559f5aacac47ea0d8e03a3424c9c22c1c1a6bd21bc12e0.jpg)

SHALEEN BHATNAGAR is an Assistant Professor (Senior Scale) with the School of Computer Engineering, Manipal Institute of Technology, Bengaluru, Manipal Academy of Higher Education (MAHE), Manipal, India. She has more than 14 years of experience in teaching and research. Her research focuses on biometrics, computer vi sion, pattern recognition, and image processing, with recent emphasis on machine learning, deep learning, and quantum computing for real-world

applications. She has authored and co-authored several research papers in these areas. She is a Senior Member of IEEE and actively contributes to academic and professional activities within the computing research community.

![](images/636c9c9c8ae9ed563704d0af90bc5b304a003903700282f5f907843623c29c17.jpg)

SHILPA ANKALAKI received the Ph.D. degree in computer science and engineering from Visveswaraya Technological University, Belagavi, India. She is currently an Assistant Professor (Senior Scale) with the School of Computer Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education, Manipal, Bengaluru. She has authored several research arti cles published in various international journals and conferences. Her research interests include ma

chine learning, deep learning, data mining, artificial intelligence, and com puter vision.

![](images/f4b5f4263d6bfce354f6a2fda22c4feada164d8c08b9b898d73e939d3b144247.jpg)

TUSAR KANTI MISHRA (Senior Member, IEEE) received the B.Tech. and M.Tech. degrees, and the Ph.D. degree from NIT Rourkela, in 2015, along with an MHRD Fellowship. He is currently an Associate Professor with the School of Com puter Engineering, Manipal Institute of Technology Bengaluru. He has more than 20 years of teaching experience. His research expertise is of ten years in the domain of artificial intelligence, computer vision, pattern recognition, along with

allied application areas. He has availed the Erasmus Fellowship and worked under Prof. L. R. B. Schomaker, the Director of the ALICE Laboratory, The Netherlands, for a period of six months. He feels proud of being a student of renowned researchers, such as Dr. B. Majhi and Dr. Arun K. Pujari. He has been a reviewer for multiple reputed journals. So far, he has filed multiple patents and two copyrights with the IPR, Government of India. He has more than 35 research publications and counting. He has bagged a research funding grant of INR 30 lakhs from SERB, Government of India. He is also a member of the BoS, Sambalpur University Institute of Information Technology