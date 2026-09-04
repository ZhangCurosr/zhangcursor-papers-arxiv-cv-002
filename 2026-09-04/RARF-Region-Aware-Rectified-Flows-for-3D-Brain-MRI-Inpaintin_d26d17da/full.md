# RARF: Region-Aware Rectified Flows for 3D Brain MRI Inpainting

Tomas Guija-Valiente<sup>1[0009−0000−0911−3317]</sup>, Blanca Rodriguez-Gonzalez<sup>1[0009−0007−0982−1293]</sup>, Norberto Malpica<sup>1[0000−0003−4618−7459]</sup>, and Angel Torrado-Carvajal<sup>1[0000−0002−1540−2809]</sup>

Medical Image Analysis and Biometry Lab, Universidad Rey Juan Carlos, Madrid, Spain

tomas.guija@urjc.es

Abstract. Medical image inpainting has the potential to improve automated brain MRI analysis by reconstructing healthy tissue within pathological regions. We introduce RARF, a task-agnostic region-aware rectified flow framework for masked data generation. We instantiate the framework for 3D brain MRI inpainting as our submission to the BraTS Inpainting Challenge 2026. RARF restricts the stochastic interpolation process to the inpainting region, while the observed voxels remain fixed and provide patient-specific anatomical context. A three-dimensional neural network receives the partially voided image, with Gaussian noise filling the missing region, together with the inpainting mask and the corresponding timestep. The model is trained using masked flow-matching and reconstruction-consistency objectives, combined with mask-aware preprocessing and data augmentation. During inference, the learned velocity field transports the initial noise toward a plausible reconstruction of the missing tissue, which is then combined with the unchanged observed anatomy. Experiments under the BraTS evaluation protocol show that the proposed approach produces competitive reconstructions while maintaining anatomical consistency. Source code is available at: https://github.com/TomasGuija/rarf.

Keywords: Rectified flow · Image inpainting · Medical image synthesis · Brain MRI

## 1 Introduction

Many automated brain Magnetic Resonance Imaging (MRI) analysis methods are designed under the assumption that the input image represents healthy anatomy. This assumption is problematic in neuro-oncology, where healthy images are generally unavailable and downstream tools for tasks such as brain extraction, tissue segmentation, or anatomical parcellation may be unreliable in

Preprint version corresponding to the initial submission prior to peer review. The final accepted version will be openly available in the oficial MICCAI proceedings on the conference website.

the presence of lesions [15]. In this context, healthy tissue synthesis can provide a subject-specific anatomical proxy to mitigate pathology-induced bias in subsequent analyses.

The Brain Tumor Segmentation (BraTS) Challenge: Local Synthesis of Healthy Brain Tissue via Inpainting [9,2] addresses this limitation by formulating healthy tissue synthesis as a standardized inpainting task: given a partially masked brain MRI, the objective is to reconstruct the tumor-afected region with anatomically plausible, tumor-free tissue.

Classical approaches based on difusion, fast marching, exemplar filling, or smoothness-regularized interpolation have been applied to both natural images and medical imaging problems, including the recovery of missing voxels in brain MRI [3,16,4,17]. However, these methods are often insuficient for large pathological regions, where complete anatomical structures must be synthesized rather than locally propagated. Deep learning approaches address this limitation by learning data-driven priors, with mask-aware convolutional and difusion models proving efective for irregular regions in natural images [10,12] and for lesion filling, pathology-free reconstruction, and healthy tissue synthesis in medical imaging [1,13,15,5]. In BraTS, existing methods mainly rely on direct U-Net-based reconstruction [18,19,20] or iterative difusion-based restoration [5]. More recently, region-aware difusion (RAD) has shown the value of adapting the generative process to the mask geometry [8], which is particularly relevant when synthesizing plausible healthy anatomy while preserving the surrounding patient-specific context.

Motivated by this region-aware perspective, we propose RARF, a Region-Aware Rectified Flow framework for localized image inpainting [11]. RARF is designed to be task-agnostic and supports multiple training and inference strategies. In the BraTS instantiation considered here, rectified-flow interpolation and supervision are restricted to the target region, while the observed anatomy is preserved as conditioning context. The model therefore learns to generate plausible healthy tissue within the missing region without modifying the surrounding anatomy. An overview of the resulting training and inference pipeline is shown in Fig. 1. Preliminary results suggest that this BraTS-specific instantiation produces anatomically consistent reconstructions.

## 2 Methods

## 2.1 Data and pre-processing

We use the BraTS Local Inpainting dataset [9], derived from the BraTS glioma collection [2]. It contains 1,251 training and 219 validation skull-stripped, coregistered T1-weighted MRI volumes with a common shape of 240 × 240 × 155 voxels and 1mm isotropic spacing. Ground-truth volumes are not publicly available for the validation set.

For each training case, the dataset provides the original T1 volume, a healthy tissue mask h, an unhealthy tissue mask, and their union m, defining the complete inpainting region. The input is voided over m, while voxel-wise supervision is restricted to h, since tumor intensities are not valid targets for healthy-tissue reconstruction. We use all 1,251 oficial training cases for optimization. To monitor overfitting without withholding training data, we use the challenge validation cases by placing synthetic masks over visible healthy tissue, creating known regions for supervised validation. The masks are generated and placed following the procedure described in Section 2.2.

![](images/858bf052cf35288a457643ce4de00fa4838d5727e760ce98781e4cb2500db9d6.jpg)  
Fig. 1: Overview of the BraTS instantiation of our method. During training, RARF interpolation is applied within the inpainting mask while the observed anatomy remains fixed. The network predicts the masked flow velocity from the interpolated image, mask, and timestep, with supervision restricted to healthy tissue. At inference, the missing region is initialized with noise, integrated from t = 0 to t = 1, and composed with the unchanged visible anatomy.

All case volumes and masks are cropped identically using only information available at inference. The foreground is defined as the union of visible nonzero voxels in the voided image and the inpainting mask, and the crop is centered on the midpoint of its axis-aligned bounding box. Volumes are cropped or zeropadded to 166 × 196 × 152 voxels.

Intensities are normalized independently for each case. We compute the 0.5th and 99.5th percentiles of the complete voided volume before cropping, including zero-valued background voxels, and constrain the lower bound to be nonnegative. Intensities are clipped to these bounds and linearly mapped to [0, 1]. During training, the same transformation is applied to the voided input and clean target.

## 2.2 Healthy-mask augmentation

Following the mask-generation procedure of Zhang et al. [20], we generate five mask variants per training case, including the original healthy mask. Additional masks are created from transformed connected components of training-set lesions, and placed in healthy tissue under constraints on lesion distance, brain coverage, overlap, and diversity. Their union with the unhealthy mask defines the conditioning region, and one variant is sampled per training instance.

## 2.3 Region-aware rectified flow

Let $x _ { 1 } \in \mathbb { R } ^ { H \times W \times D }$ denote the normalized T1 volume, $m \in \{ 0 , 1 \} ^ { H \times W \times D }$ the complete inpainting mask, and $h \in \{ 0 , 1 \} ^ { H \times W \times D }$ the healthy-tissue mask, with $h \subseteq m$ . The voided volume is

$$
y = \left( 1 - m \right) \odot x _ { 1 } .\tag{1}
$$

For the BraTS instantiation, inspired by the spatially varying generative process of RAD [8], we define a localized rectified-flow path in which only the masked region evolves, while the observed context remains fixed during training and inference. Given a Gaussian noise image

$$
x _ { 0 } \sim \mathcal { N } ( 0 , I )\tag{2}
$$

and a flow time $t \in [ 0 , 1 ]$ , where $t = 0$ denotes the source distribution and $t = 1$ the data distribution, we define

$$
x _ { t } = m \odot \left[ ( 1 - t ) x _ { 0 } + t x _ { 1 } \right] + y .\tag{3}
$$

Thus, interpolation is restricted to m, while the visible anatomy is copied from the voided input.

Equivalently, the interpolation can be expressed through the spatial time map

$$
\tau _ { t } = m t + ( 1 - m ) ,\tag{4}
$$

which assigns time t to masked voxels and fixes visible voxels at the clean endpoint. Since $\tau _ { t }$ is completely determined by t and $m ,$ , it is not provided as a separate network input. Instead, the model receives the channel-wise concatenation

$$
z _ { t } = \mathrm { c o n c a t } ( x _ { t } , m )\tag{5}
$$

together with an embedding of the scalar flow time.

The target rectified-flow velocity along the linear path is:

$$
v ^ { \star } = m \odot ( x _ { 1 } - x _ { 0 } ) .\tag{6}
$$

The network is trained to approximate this velocity within the healthy region,

$$
\begin{array} { r } { v _ { \theta } ( z _ { t } , t ) \approx v ^ { \star } . } \end{array}\tag{7}
$$

Although m determines the entire region to be generated, only h contributes to the training objective.

At inference, the region defined by m is initialized with Gaussian noise and the learned velocity field is integrated from $t = 0$ to $t = 1$ . At every integration step, the predicted update is multiplied by m, and the observed voxels are restored from y. Consequently, the visible anatomical context remains unchanged throughout generation.

## 2.4 Network architecture

We parameterize the velocity field using a 3D U-Net. Its input has two channels, corresponding to the current state $x _ { t }$ and the complete mask m, and its output is a single-channel velocity volume with the same spatial dimensions.

The encoder has three resolution levels with channel widths 32, 64, and 128. Each level contains two residual blocks composed of group normalization, SiLU activations, and $3 \times 3 \times 3$ convolutions. Downsampling is performed using stridetwo $3 \times 3 \times 3$ convolutions. The decoder mirrors the encoder and combines its features with the corresponding encoder activations through U-Net skip connections. Upsampling uses nearest-neighbor interpolation followed by a $3 \times 3 \times 3$ convolution.

The scalar flow time is multiplied by $T \ : = \ : 1 0 0 0$ and encoded using sinusoidal features. A two-layer multilayer perceptron maps this encoding to a 128- dimensional time representation, which is projected and added to every residual block. The final prediction head applies group normalization, a SiLU activation, and a zero-initialized $3 \times 3 \times 3$ convolution.

## 2.5 Training objective

Although the complete mask m defines the region to be synthesized, supervision is restricted to the healthy-tissue mask $h \subseteq m$ . This prevents pathological tissue from contributing directly to the learning objective. We define the masked meansquared and mean-absolute errors as

$$
\mathrm { M S E } _ { h } ( a , b ) = \frac { \left\| h \odot ( a - b ) \right\| _ { 2 } ^ { 2 } } { \left\| h \right\| _ { 1 } } , \qquad \mathrm { M A E } _ { h } ( a , b ) = \frac { \left\| h \odot ( a - b ) \right\| _ { 1 } } { \left\| h \right\| _ { 1 } } .\tag{8}
$$

For the noise sample $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ and clean target $x _ { 1 } .$ , the region-aware interpolation has the constant target velocity $v ^ { \star }$ defined in $\mathrm { E q . ~ 6 }$ . The primary rectified-flow objective is therefore

$$
\mathcal { L } _ { \mathrm { R F } } = \mathrm { M S E } _ { h } \left( v _ { \theta } ( z _ { t } , t ) , v ^ { \star } \right) .\tag{9}
$$

We supplement this squared flow-matching objective with voxel-wise and structural auxiliary terms. Given the predicted velocity, the corresponding clean endpoint is estimated as

$$
\widehat { x } _ { 1 } = x _ { t } + \left( 1 - \tau _ { t } \right) \odot v _ { \theta } ( z _ { t } , t ) .\tag{10}
$$

Along the linear region-aware path, its endpoint error can be written as

$$
\widehat { \boldsymbol { x } } _ { 1 } - \boldsymbol { x } _ { 1 } = \left( 1 - \tau _ { t } \right) \odot \left( \boldsymbol { v } _ { \theta } ( \boldsymbol { z } _ { t } , t ) - \boldsymbol { v } ^ { \star } \right) .\tag{11}
$$

To remove the dependence of the endpoint error on the remaining trajectory length, we define the normalized endpoint error

$$
\widetilde { e } _ { t } = \frac { \widehat { x } _ { 1 } - x _ { 1 } } { \operatorname* { m a x } ( 1 - \tau _ { t } , \epsilon ) } , \qquad \epsilon = 1 0 ^ { - 3 } ,\tag{12}
$$

where the maximum and division are applied element-wise. Within the supervised region and away from the numerical clamp, $\widetilde { e } _ { t }$ is equivalent to $v _ { \theta } ( z _ { t } , t ) - v ^ { \star }$

We apply a masked mean-absolute error to this normalized endpoint error:

$$
\mathcal { L } _ { \mathrm { M A E } } = w ( t ) \mathrm { M A E } _ { h } \left( \widetilde { e } _ { t } , 0 \right) , \qquad w ( t ) = 1 + \frac { 1 } { 2 } t ^ { 2 } .\tag{13}
$$

This term complements the squared flow-matching objective with an $\ell _ { 1 }$ penalty, providing direct voxel-wise supervision while reducing the influence of isolated large errors.

To additionally encourage preservation of local anatomical structure and contrast, we compare the estimated and target endpoints using a masked structuralsimilarity loss:

$$
\mathcal { L } _ { \mathrm { S S I M } } = w ( t ) \mathrm { m e a n } _ { h } \left[ 1 - \mathrm { S S I M } _ { \mathrm { m a p } } \left( \widehat { x } _ { 1 } , x _ { 1 } \right) \right] .\tag{14}
$$

Before computing the structural similarity index measure (SSIM), both endpoints are clipped to the normalized intensity range [0, 1] and set to zero outside h. The resulting SSIM map is averaged only over the supervised region.

The weighting factor $w ( t )$ mildly emphasizes later trajectory stages, where the evolving sample is closer to the clean endpoint and fine reconstruction errors become more relevant. Training times are sampled from a mixture distribution: with probability 0.75, t is sampled uniformly from $\mathcal { U } ( 0 , 1 )$ ; otherwise, it is sampled as $t = 1 - u ^ { 2 }$ , with $u \sim \mathcal { U } ( 0 , 1 )$ , increasing the frequency of examples near the clean endpoint.

The complete training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { R F } } + \lambda _ { \mathrm { M A E } } \mathcal { L } _ { \mathrm { M A E } } + \lambda _ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { S S I M } } .\tag{15}
$$

## 2.6 Inference

At inference, the masked region is initialized with Gaussian noise and the learned velocity field is integrated from $t = 0$ to $t = 1$ . After each update, voxels outside the mask are restored from the observed image.

The number of integration steps balances accuracy and runtime. Based on validation results, we use four integration time points with midpoint integration, which provided the best empirical performance while remaining substantially faster than conventional denoising difusion probabilistic model (DDPM) sampling.

Since diferent initial noise realizations may produce diferent plausible completions, we also consider drawing K independent samples $\{ \hat { x } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ . Their voxel-wise average is

$$
\bar { x } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \hat { x } ^ { ( k ) } .\tag{16}
$$

This Monte Carlo estimate approaches the conditional mean, which minimizes expected squared error and can therefore improve distortion-based metrics such as MSE and PSNR. However, when multiple anatomical completions are plausible, it may blur fine structures, illustrating the distortion–perception trade-of in image restoration [6].

To retain more sample-level detail, we evaluate sample selection strategies. First, we select the generated sample closest to the sample mean within the inpainting mask:

$$
\hat { x } _ { \mathrm { m e d o i d } } = \arg \operatorname* { m i n } _ { \hat { x } ^ { ( j ) } } \mathrm { M S E } _ { m } \left( \hat { x } ^ { ( j ) } , \bar { x } \right) ,\tag{17}
$$

where $\mathrm { M S E } _ { m }$ denotes the MSE restricted to the complete inpainting mask.

We also evaluate kernel density steering (KDS), an inference-time scaling method for image restoration based on mode seeking in the sample distribution [7]. KDS jointly evolves multiple samples and, at each integration step, compares their predicted clean endpoints using a Gaussian kernel. A mean-shift update then steers the samples toward regions of higher estimated density. After integration, we compute the voxel-wise mean of the final particles within the inpainting region and select the particle with the smallest mean-squared distance to this mean.

Finally, we consider a minimum Bayes-risk (MBR) selection strategy inspired by agreement-based decoding [14]. Each generated completion is compared with the remaining samples using a weighted combination of masked MSE and SSIM, and the candidate with the lowest average pairwise risk is selected. This yields a central representative completion without averaging voxel intensities.

Section 3.2 compares these inference strategies using the same trained checkpoint, isolating their efects from diferences in model training.

## 3 Results

## 3.1 Experimental setup and metrics

We conduct all experiments on the BraTS Local Inpainting dataset described in Sec. 2.1, using the training and local-validation setup, preprocessing pipeline, and mask-generation strategy introduced therein. Since the oficial validation targets are withheld, local validation is used to monitor training and select model configurations. Inference ablations are evaluated through separate submissions to the oficial Synapse platform using the same trained checkpoint. We report mean squared error (MSE), mean absolute error (MAE), peak signal-to-noise ratio (PSNR), and structural similarity index measure (SSIM).

## 3.2 Ablation study

Our ablation study focuses on the inference strategies described in Sec. 2.6, all evaluated using the same trained checkpoint. All reported results use an exponential moving average (EMA) of the model parameters, with decay $\beta = 0 . 9 9 9$ We compare them quantitatively through separate submissions to the Synapse evaluation platform and qualitatively through representative reconstructed volumes. Ablations of the broader training modes supported by RARF are left for future work, as they fall outside the scope of this challenge submission.

Table 1: Inference ablation results on the oficial BraTS validation set.
<table><tr><td>Inference strategy</td><td>MSE</td><td>MAE</td><td>↓PSNR ↑</td><td>SSIM↑</td></tr><tr><td>30-sample average</td><td>0.006</td><td>0.018</td><td>23.830</td><td>0.823</td></tr><tr><td>30-sample closest to mean</td><td>0.009</td><td>0.021</td><td>21.911</td><td>0.780</td></tr><tr><td>KDS</td><td>0.009</td><td>0.021</td><td>22.096</td><td>0.784</td></tr><tr><td>MBR</td><td>0.009</td><td>0.022</td><td>21.982</td><td>0.781</td></tr><tr><td>Regular inference</td><td>0.010</td><td>0.022</td><td>21.942</td><td>0.777</td></tr></table>

Table 1 shows that distortion-based metrics benefit from averaging multiple samples, consistent with interpreting the sample mean as a Monte Carlo estimate of the conditional expectation. However, averaging also produces smoother reconstructions and may blur fine anatomical structures, illustrating the distortion– perception trade-of. Representative-sample strategies provide a compromise by improving distortion metrics over single-sample inference while better preserving sample-level detail. Nevertheless, with $K = 3 0$ , all multi-sample strategies require substantially greater inference time.

Figure 2 compares standard single-sample inference with K-sample averaging, together with the voided input and ground truth.

## 3.3 Main challenge results

Table 2: Oficial BraTS validation performance of RARF using the released checkpoint and $K = 5 0$ sample averaging.

$$
\frac { \mathbf { \overline { { M S E } } } \downarrow \mathbf { P S N R } \uparrow \mathbf { S S I M } \uparrow } { 0 . 0 0 6 2 4 . 0 0 8 0 . 8 3 2 }
$$

Table 2 summarizes the oficial validation performance of our challenge submission. This submission uses a separately trained checkpoint and averages $K = 5 0$ samples per case to prioritize the distortion-based evaluation metrics. For reproducibility, we publicly release the corresponding model weights together with the source code.

![](images/b3577dca3ee79ee8a59c7a48c8aa1413a5a9598eeb0b95e0daa1bfbe1352acb9.jpg)  
Fig. 2: Qualitative comparison between single-sample inference and averaging K = 30 samples across three representative cases. Absolute-error maps are restricted to the synthesized region and shown using a common color scale, with white indicating zero error and increasingly saturated red indicating larger discrepancies. Averaging generally improves distortion-based fidelity, although it may smooth fine anatomical structures relative to individual generated samples.

## 4 Discussion

RARF is conceived as a general region-aware inpainting framework rather than a method tailored exclusively to the BraTS benchmark. Its flexible formulation supports diferent training and inference configurations and can be applied to arbitrary inpainting masks. We believe this broader perspective is important, as medical imaging can both benefit from and contribute to advances in general computer vision.

Medical imaging also provides a particularly demanding benchmark domain because of its high dimensionality, complex spatial structure, and strict anatomical constraints. In inpainting, the masked region corresponds to a specific underlying anatomy rather than one of many equally valid completions. This makes reference-based distortion metrics more informative than in less constrained natural-image settings.

Nevertheless, our ablation shows that lower distortion does not necessarily imply better perceptual quality or anatomical plausibility. Averaging multiple samples improved MSE and PSNR but produced smoother, blurrier reconstructions that could still rank above sharper predictions. This motivates complementing voxel-wise metrics with perceptual, anatomy-aware, and downstreamtask evaluation.

Acknowledgments. This study has been funded by the MAGERIT-CM project (TEC2024/COM-44), funded by Comunidad de Madrid.

Disclosure of Interests. Authors declare no conflict of interests relevant to this research.

## References

1. Armanious, K., Kumar, V., Abdulatif, S., Hepp, T., Gatidis, S., Yang, B.: ipamedgan: Inpainting of arbitrary regions in medical imaging. In: 2020 IEEE international conference on image processing (ICIP). pp. 3005–3009. IEEE (2020)

2. Baid, U., Ghodasara, S., Mohan, S., et al.: The rsna-asnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification (2021), https://arxiv.org/abs/2107.02314

3. Bertalmío, M., Sapiro, G., Caselles, V., Ballester, C.: Image inpainting. pp. 417–424 (01 2000)

4. Criminisi, A., Perez, P., Toyama, K.: Region filling and object removal by exemplarbased image inpainting. IEEE Transactions on Image Processing 13(9), 1200–1212 (2004). https://doi.org/10.1109/TIP.2004.833105

5. Durrer, A., Wolleb, J., Bieder, F., Friedrich, P., et al.: Denoising difusion models for 3d healthy brain tissue inpainting. In: MICCAI Workshop on Deep Generative Models. pp. 87–97. Springer (2024)

6. Freirich, D., Michaeli, T., Meir, R.: A theory of the distortion-perception tradeof in wasserstein space. In: Advances in Neural Information Processing Systems. vol. 34, pp. 25661–25672. Curran Associates, Inc. (2021)

7. Hu, Y., Mei, K., Sahraee-Ardakan, M., Kamilov, U., Milanfar, P., Delbracio, M.: Kernel density steering: Inference-time scaling via mode seeking for image restoration. In: Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025 (2025)

8. Kim, S., Suh, S., Lee, M.: Rad: Region-aware difusion models for image inpainting. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2439–2448 (2025). https://doi.org/10.1109/CVPR52734.2025.00233

9. Kofler, F., Meissen, F., Steinbauer, F., et al.: The brain tumor segmentation (brats) challenge: Local synthesis of healthy brain tissue via inpainting (2024), https:// arxiv.org/abs/2305.08992

10. Liu, G., Reda, F.A., Shih, K.J., Wang, T.C., Tao, A., Catanzaro, B.: Image inpainting for irregular holes using partial convolutions. In: Computer Vision – ECCV 2018. pp. 89–105. Springer International Publishing, Cham (2018)

11. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. ArXiv abs/2209.03003 (2022), https://api. semanticscholar.org/CorpusID:252111177

12. Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., Van Gool, L.: Repaint: Inpainting using denoising difusion probabilistic models. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 11451–11461 (2022). https://doi.org/10.1109/CVPR52688.2022.01117

13. Manjón, J.V., Romero, J.E., Vivo-Hernando, R., Rubio, G., Aparici, F., de la Iglesia-Vaya, M., Tourdias, T., Coupé, P.: Blind mri brain lesion inpainting using deep learning. In: Simulation and Synthesis in Medical Imaging. pp. 41–49. Springer International Publishing, Cham (2020)

14. Natsumi, K., Deguchi, H., Sakai, Y., Kamigaito, H., Watanabe, T.: Agreementconstrained probabilistic minimum Bayes risk decoding. In: Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics. pp. 484–493. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics, Mumbai, India (Dec 2025). https://doi.org/10.18653/v1/2025.ijcnlp-short.39

15. Pollak, C., Kügler, D., Bauer, T., Rüber, T., Reuter, M.: FastSurfer-LIT: Lesion inpainting tool for whole-brain MRI segmentation with tumors, cavities, and abnormalities. Imaging Neuroscience 3, imag\_a\_00446 (Jan 2025). https: //doi.org/10.1162/imag\_a\_00446

16. Telea, A.: An image inpainting technique based on the fast marching method. Journal of Graphics Tools 9 (01 2004). https://doi.org/10.1080/10867651.2004. 10487596

17. Torrado-Carvajal, A., Albrecht, D., Lee, J., Andronesi, O., Ratai, E.M., Napadow, V., Loggia, M.: Inpainting as a technique for estimation of missing voxels in chemical shift imaging (02 2020). https://doi.org/10.1101/2020.02.17.952325

18. Zhang, J., Chen, K., Weng, Y.: Synthesis of healthy tissue within tumor area via unet. In: Brain Tumor Segmentation, and Cross-Modality Domain Adaptation for Medical Image Segmentation. pp. 233–240. Springer Nature Switzerland, Cham (2024)

19. Zhang, J., Weng, Y., Chen, K.: U-net based healthy 3d brain tissue inpainting. ArXiv abs/2507.18126 (2025), https://api.semanticscholar.org/CorpusID: 280017988

20. Zhang, J., Weng, Y., Chen, K.: Robust 3d brain mri inpainting with random masking augmentation. In: Segmentation, Classification, and Synthesis for Brain Tumors and Traumatic Brain Injuries. pp. 102–109. Springer Nature Switzerland, Cham (2026)