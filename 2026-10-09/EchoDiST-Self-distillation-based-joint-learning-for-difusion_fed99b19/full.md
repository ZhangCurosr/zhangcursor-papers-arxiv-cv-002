# EchoDiST: Self-distillation-based joint learning for difusion-conditioned echocardiographic myocardial motion estimation

Feiyue Qi<sup>a,1</sup>, Xingyue Wei<sup>a,1</sup> and Jianwen Luo<sup>a,∗</sup>

<sup>a</sup>School of Biomedical Engineering, Tsinghua Medicine, Tsinghua University, Beijing, 100084, China

## A R T I C L E I N F O

Keywords:   
Echocardiography   
Myocardial motion estimation   
Self-distillation   
Joint learning   
Difusion-conditioned

## A BS T RA C T

Motion estimation in echocardiography is essential for quantitative assessment of cardiac function and myocardial mechanics. However, accurate myocardial motion estimation remains challenging due to image artifacts, limited image information, speckle decorrelation, and the scarcity of ground-truth myocardial displacement fields. Anatomy-guided approaches can provide additional structural information for motion estimation, but often depend on expert-labeled myocardial segmentations, which limits their scalability in clinical applications. To address these challenges, we propose EchoDiST, a framework for unsupervised echocardiographic myocardial motion estimation that integrates a self distillation-based joint learning strategy with a difusion-conditioned motion estimation network. Here, unsupervised motion estimation refers to learning without ground-truth displacement fields. The self-distillation-based joint learning strategy enables anatomical segmentation and myocardial motion estimation to be jointly optimized under limited anatomical annotations. The difusion-conditioned motion estimation network incorporates difusion-based conditioning into the estimation process, with stochastic difusion perturbations used only during training, while inference is performed through a i l d i i i f d i h i i dif i li i l l ated EchoDiST on three echocardiographic datasets, including two external test datasets evaluated under cross-view and cross-dataset settings. Compared with seven representative learning-based methods, EchoDiST demonstrates consistent improvements in myocardial motion estimation across multiple evaluation levels, including anatomical alignment, myocardial strain assessment, and motionderived functional and cardiac-phase assessment. These gains are statistically significant, further supporting the efectiveness and robustness of EchoDiST across the evaluated tasks and datasets. Overall, EchoDiST provides an efective approach for reliable myocardial motion estimation under limited anatomical supervision and supports downstream quantitative assessment of cardiac function.

## 1. Introduction

Cardiovascular diseases (CVDs) remain the leading cause of death worldwide and continue to impose substantial economic and societal burdens (Tsao et al., 2022). Quantitative assessment of myocardial motion can reveal regional and global mechanical abnormalities, thereby providing clinically useful information for disease detection, functional assessment, prognostic evaluation, and prediction of therapeutic response (Risum et al., 2012; Park et al., 2018; Zhang et al., 2018; van Mourik et al., 2019). Echocardiography is particularly well suited to myocardial motion assessment because it is non-invasive, widely available and costefective, while also providing information on both global cardiac function and regional myocardial mechanics (Ahn et al., 2023). Reliable myocardial motion estimation from echocardiographic sequences is therefore essential for the quantitative assessment of cardiac function (Ta et al., 2024).

Despite its importance, accurate myocardial motion estimation from echocardiographic sequences remains challenging. Echocardiographic images are inherently afected by artifacts (e.g., shadowing and reverberation), limited image information, and speckle decorrelation. These factors can alter the appearance of the same myocardial tissue across cardiac phases, making reliable motion estimation dificult, particularly during rapid or rotational motion (Evain et al.,

2022). Meanwhile, the scarcity of ground-truth myocardial displacement fields in clinical echocardiography limits the feasibility of fully supervised motion estimation (Ahn et al., 2023).

To address these challenges, anatomical or shape information, particularly myocardial segmentation, has been incorporated into cardiac motion estimation to improve its accuracy (Papademetris et al., 2002; Ta et al., 2021, 2024). However, inaccurate segmentation may in turn lead to unreliable motion estimation (Ta et al., 2024). Previous studies have further exploredjoint optimization of segmentation and motion estimation, allowing the two tasks to benefit from each other (Qin et al., 2018; Ta et al., 2020, 2024). However, existing anatomy-guided motion estimation approaches often rely on expert-labeled myocardial segmentations to provide structural supervision (Ta et al., 2021; Ahn et al., 2023). Obtaining such manual annotations at scale is dificult, labor-intensive, and time-consuming (Xu and Niethammer, 2019; Ta et al., 2021). Reducing this dependence while maintaining efective joint learning therefore remains a key challenge.

In this work, we propose EchoDiST (Echocardiographic Difusion-conditioned Self-distillation), a framework for unsupervised echocardiographic myocardial motion estimation that integrates a self-distillation-based joint learning strategy with a difusion-conditioned motion estimation network. Here, unsupervised motion estimation refers to learning myocardial motion without ground-truth displacement fields. The self-distillation-based joint learning strategy enables anatomical segmentation and myocardial motion estimation to be jointly optimized under limited anatomical annotations. The difusion-conditioned motion estimation network incorporates difusion-based conditioning into the estimation process, providing an alternative to conventional direct regression-based motion estimation. To the best of our knowledge, difusion-conditioned motion estimation has not previously been explored for myocardial motion estimation. Comprehensive experiments on three echocardiographic datasets demonstrate the efectiveness of EchoDiST in myocardial motion estimation across multiple evaluation levels, including anatomical alignment, myocardial strain assessment, and motion-derived functional and cardiac-phase assessment.

The main contributions of this work are summarized as follows:

• We propose EchoDiST for echocardiographic myocardial motion estimation. The framework incorporates difusion-based conditioning into motion estimation and integrates anatomical segmentation and motion estimation to improve reliability.

• We develop a self-distillation-based joint learning strategy that simultaneously optimizes anatomical segmentation and unsupervised myocardial motion estimation under limited anatomical annotations. The proposed strategy uses the segmentation results and estimated motion to provide mutual learning signals, thereby reducing the reliance on extensive manual segmentation annotations.

• We conduct a comprehensive evaluation of EchoDiST across three echocardiographic datasets, including two external test datasets. The results demonstrate consistent improvements in myocardial motion estimation across multiple evaluation levels, including anatomical alignment, myocardial strain assessment, and motion-derived functional and cardiac-phase assessment.

## 2. Related Work

## 2.1. Echocardiographic Motion Estimation Methods

Echocardiographic motion estimation aims to estimate myocardial displacement across cardiac phases, providing a basis for subsequent cardiac functional analysis. Conventional approaches primarily depend on intensity- or featurebased tracking. Huang et al. (2014) employed sparse representation and dictionary learning for echocardiographic contour tracking, while Parajuli et al. (2016) integrated dynamic shape tracking with radio-frequency speckle tracking for cardiac motion analysis. Despite these advances, conventional motion-tracking approaches can be computationally intensive and often rely on prior knowledge or extensively hand-crafted features (Ta et al., 2024).

Deep learning-based methods have subsequently been developed to estimate myocardial motion directly from echocardiographic data. Some supervised approaches rely on synthetic data with reference displacement fields; for example, Evain et al. (2022) developed a deep-learning framework for 2D echocardiographic motion estimation using realistic synthetic B-mode sequences with reference myocardial motion. However, reliable dense motion annotations are dificult to obtain from clinical echocardiography, motivating unsupervised approaches that estimate motion without direct supervision from ground-truth displacement fields. Ahn et al. (2023), for instance, proposed a co-attention spatial transformer network for unsupervised motion tracking in 3D echocardiography by leveraging inter-frame spatiotemporal information. Although such approaches alleviate the need for motion annotations, their optimization is mainly driven by image appearance or spatiotemporal correspondence, with limited explicit anatomical supervision.

Anatomical labels have therefore been introduced as auxiliary supervision for echocardiographic motion estimation. Ta et al. (2021), for instance, used manually traced segmentations to provide shape guidance for left-ventricular motion estimation in 3D+Time echocardiography. Furthermore, Ta et al. (2024) showed that anatomical segmentation and motion estimation can be complementary when jointly optimized through a multi-task learning framework. However, extending such supervision to large-scale motion estimation would require a large number of expert-annotated ground-truth segmentation masks, which are costly and labor-intensive to obtain.

To reduce this dependence, our framework learns anatomical segmentations from a limited labeled subset and uses the resulting predictions to provide structural guidance over a larger set of training pairs for motion estimation.

## 2.2. Difusion-Based Motion Estimation Methods

Difusion models learn to reverse a progressive noising process through a denoising objective (Ho et al., 2020). Originally developed for generative modeling, difusion models have subsequently been adapted to dense motion estimation between medical images, where difusion-derived features can provide spatial information useful for correspondence learning. DifuseMorph, for example, models the conditional score associated with the spatial transformation between a moving–fixed image pair and uses the resulting latent feature to generate a dense displacement field (Kim et al., 2022).

Subsequent studies have explored diferent ways of incorporating difusion into motion estimation. FSDifReg employs multi-scale semantic features derived from a difusion model to guide dense displacement prediction, while the difusion score further contributes to topology-preserving optimization (Qin and Li, 2023). In contrast, DifuseReg directly formulates the displacement field as the difusion target and progressively recovers it through iterative denoising (Zhuo and Shen, 2024). More recently, LDM-Morph incorporates semantic features extracted from a latent diffusion model and integrates them with global image features through cross-attention, together with a hierarchical similarity measure defined in both pixel and latent-feature spaces (Wu et al., 2026).

Collectively, these studies demonstrate the potential of difusion-based modeling for dense motion estimation between medical images, yet its application to echocardiographic motion estimation remains relatively underexplored, particularly for difusion-conditioned learning of myocardial motion. Motivated by this gap, we employ difusionconditioned learning for direct echocardiographic motion estimation while retaining deterministic single-pass inference.

## 2.3. Joint Learning of Segmentation and Registration with Limited Annotations

Anatomical guidance is widely used to impose structural constraints on deformable registration, but its reliance on dense segmentation annotations limits its use in label-scarce settings. To reduce the dependence of anatomy-guided registration on densely annotated data, several studies have explored learning anatomical supervision from limited labels and extending it to unlabeled images. DeepAtlas (Xu and Niethammer, 2019) jointly learned weakly supervised registration and semi-supervised segmentation, allowing predicted segmentations to provide anatomical guidance when manual annotations were unavailable. He et al. (2022) further introduced a few-shot deformable registration framework that decouples anatomical perception from image correspondence and uses reverse teaching to transfer structural knowledge from only a few labeled images to unlabeled data. Shi et al. (2023) incorporated self-training into joint registration and segmentation, where quality-assessed pseudo-labels extend anatomical supervision to unlabeled images while supporting weakly supervised registration. Alongside these learning strategies, pretrained medical vision models have provided an additional source of anatomical information for registration. Xu et al. (2025) employed text-prompted segmentation masks generated by a vision foundation model to explicitly encode anatomical structures for deformable registration, using the resulting masks for feature fusion, semantic prototype alignment, and contour-aware optimization.

Inspired by these studies, we propose EchoDiST, which jointly learns anatomical segmentation and unsupervised motion estimation through EMA-based self-distillation. Such a mechanism can achieve efective learning with limited annotations.

## 3. Methods

Given a moving echocardiographic frame $I _ { m }$ and a fixed frame $I _ { f }$ , our goal is to estimate a dense displacement field $\mathbf { u } _ { m  f }$ that establishes spatial correspondence between the two frames. No ground-truth displacement fields are used for motion-estimation learning. As illustrated in Fig. 1, the proposed EchoDiST follows a three-stage training pipeline:

1. Segmentation Pretraining Stage: A segmentation network is pretrained on a limited set of annotated echocardiographic images to provide structural guidance for subsequent motion estimation.

2. Motion-Estimation Pretraining Stage: The difusionconditioned motion-estimation network is pretrained with the segmentation masks predicted by the segmentation network as anatomical guidance. Gaussian perturbations at randomly sampled difusion timesteps facilitate motion estimation across diferent perturbation levels.

3. Self-distillation-based Joint Learning Stage: The pretrained segmentation and motion-estimation networks are jointly optimized through EMA-based self-distillation and reliability-weighted bidirectional cross-frame anatomical consistency.

## 3.1. Segmentation Pretraining

In Stage 1, a segmentation network is pretrained using a limited set of manually annotated echocardiographic images. To enhance the robustness of anatomical segmentation, its encoder adopts the same architecture as EchoFM (Kim et al., 2025), a foundation model pretrained on a large-scale collection of echocardiographic images. An FPN-based decoder (Lin et al., 2017) integrates multi-level features for anatomical segmentation. The main segmentation head predicts four classes: background, LV cavity, myocardium, and LA. Auxiliary segmentation heads provide deep supervision at intermediate FPN levels, while a lightweight boundary branch encourages boundary-aware learning.

The segmentation-pretraining objective $\mathcal { L } _ { \mathrm { s e g } }$ comprises the main segmentation loss ${ \mathcal { L } } _ { \mathrm { m a i n } } ,$ auxiliary deep-supervision loss ${ \mathcal { L } } _ { \mathrm { a u x } } ,$ , and boundary loss ${ \mathcal { L } } _ { \mathrm { b n d } }$

$$
\mathcal { L } _ { \mathrm { m a i n } } = \mathcal { L } _ { \mathrm { w c e } } ( P , Y ) + \mathcal { L } _ { \mathrm { d i c e } } ( P , Y ) ,\tag{1}
$$

$$
\mathcal { L } _ { \mathrm { a u x } } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \left[ \mathcal { L } _ { \mathrm { w c e } } ( P _ { s } ^ { \mathrm { a u x } } , Y _ { s } ) + \mathcal { L } _ { \mathrm { d i c e } } ( P _ { s } ^ { \mathrm { a u x } } , Y _ { s } ) \right] ,\tag{2}
$$

$$
\mathcal { L } _ { \mathrm { b n d } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ b _ { i } \log \hat { b } _ { i } + ( 1 - b _ { i } ) \log ( 1 - \hat { b } _ { i } ) \right] ,\tag{3}
$$

$$
\mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { m a i n } } + \lambda _ { \mathrm { a u x } } \mathcal { L } _ { \mathrm { a u x } } + \lambda _ { \mathrm { b n d } } \mathcal { L } _ { \mathrm { b n d } } .\tag{4}
$$

Here, � and � denote the main segmentation prediction and ground-truth target, respectively. $P _ { s } ^ { \mathrm { a u x } }$ denotes the prediction from the �-th auxiliary head with target $Y _ { s }$ , and � is the number of auxiliary heads. $b _ { i }$ and $\hat { b } _ { i }$ are the groundtruth and predicted boundary values at pixel $i ,$ with � denoting the number of pixels. $\mathcal { L } _ { \mathrm { w c e } }$ and ${ \mathcal { L } } _ { \mathrm { d i c e } }$ denote the weighted cross-entropy and foreground Dice losses, while $\lambda _ { \mathrm { a u x } }$ and $\lambda _ { \mathrm { b n d } }$ weight the corresponding auxiliary and boundary terms.

The resulting segmentation model is used to generate anatomical pseudo masks for Stage 2 and to initialize the Student network in Stage 3.

![](images/01ec1812f44b59dd9bc6970eda1cf7bd4cb7b820c065ef10f8ade907db35267e.jpg)

Stage 3: Joint Anatomy–Motion Learning  
![](images/5486aa75e1299c9408c7bcf845ca344658c5f05e0d4b370cefc5b6fb69232425.jpg)  
Figure 1: Overview of the proposed EchoDiST framework for echocardiographic motion estimation. (a) Stage 1 pretrains an EchoFM-based segmentation network using a limited set of manually annotated images to generate anatomical pseudo masks. (b) Stage 2 pretrains the difusion-conditioned motion-estimation network to estimate the displacement field, with anatomical pseudo masks providing structural supervision. (c) Stage 3 jointly optimizes anatomical segmentation and motion estimation through the EMA-based self-distillation strategy.

## 3.2. Motion-Estimation Pretraining

## 3.2.1. Difusion-Conditioned Motion Estimation Network

Given $I _ { m }$ and $I _ { f } ,$ a perturbed fixed image $I _ { f } ^ { t }$ is generated using the standard forward difusion process (Ho et al., 2020). For each training pair, a difusion timestep � and Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ are randomly sampled:

$$
I _ { f } ^ { t } = \sqrt { \bar { \alpha } _ { t } } I _ { f } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon ,\tag{5}
$$

where $\bar { \alpha } _ { t }$ denotes the cumulative product of the noiseschedule coeficients. Following FSDifReg (Qin and Li, 2023), the moving image, clean fixed image, and perturbed fixed image are concatenated as

$$
X _ { t } = \mathrm { C o n c a t } \left( I _ { m } , I _ { f } , I _ { f } ^ { t } \right) .\tag{6}
$$

The overall architecture of the proposed difusion-conditioned motion-estimation network is illustrated in Fig. 2.

The sampled timestep � is embedded and injected into the residual encoding blocks. A shared encoder extracts multiscale difusion-conditioned features from $X _ { t } ,$ , and the deepest feature map is further processed by a Swin Transformer V2 bottleneck (Liu et al., 2022) to capture long-range spatial dependencies.

To complement these learned features with echocardiography-specific structural information, the clean fixed image $I _ { f }$ is additionally processed by a pretrained EchoFM encoder (Kim et al., 2025). The EchoFM encoder remains frozen during training and inference. Its high-level feature $F _ { E }$ is projected to match the dimensionality of the bottleneck feature $F _ { B }$ and fused with $F _ { B }$ through cross-attention:

$$
\widetilde { F } _ { B } = \mathcal { F } _ { \mathrm { C A } } ( F _ { B } , F _ { E } ) ,\tag{7}
$$

where $\mathcal { F } _ { \mathrm { { C A } } } ( \cdot , \cdot )$ denotes the cross-attention fusion operation.

The fused bottleneck feature ${ \widetilde { F } } _ { B }$ is subsequently decoded through two task-specific streams. The difusion stream predicts the injected noise $\boldsymbol { \hat { \epsilon } } _ { t } .$ , while the motion-estimation stream combines multi-scale decoder features with encoder skip connections to estimate the dense moving-to-fixed displacement field $\mathbf { u } _ { m  f }$ . The displacement field is progressively refined in a coarse-to-fine manner and is finally used to warp the moving image toward the fixed image:

$$
\hat { I } _ { f } = \mathscr { W } ( I _ { m } , \mathbf { u } _ { m  f } ) ,\tag{8}
$$

where $\mathcal { W } ( \cdot , \cdot )$ denotes diferentiable spatial warping.

The stochastic difusion perturbation is used only during training; inference does not require iterative reversedifusion sampling. During inference, � is fixed to 0 and the perturbed input is replaced by the clean fixed image, yielding $X _ { 0 } = \mathrm { C o n c a t } ( I _ { m } , I _ { f } , I _ { f } )$ . The displacement field is therefore predicted through a single deterministic forward pass.

![](images/f1c6d537863540cfdfccdda8159324abf8a3c17ab9f65578d2c4078252d034f2.jpg)  
Figure 2: Architecture of the proposed difusion-conditioned motion-estimation network. The moving image $I _ { m } ,$ fixed image $I _ { f } ,$ and perturbed fixed image $I _ { f } ^ { t }$ form the difusion-conditioned input $X _ { t } .$ . The network contains a difusion stream for noise prediction and a motion-estimation stream for motion estimation, together with coarse-to-fine feedback refinement. The difusion timestep embedding is injected into the residual encoding blocks, and frozen EchoFM features are fused with the bottleneck feature through cross-attention.

## 3.2.2. Feedback Refinement

Inspired by the feedback-attention strategy of (Hasan et al., 2025), which uses residual registration errors after an initial alignment to guide subsequent refinement, we incorporate misalignment-aware feedback into the coarseto-fine decoder. At decoder scale �, the displacement $\mathbf { u } _ { i - 1 }$ and feedback $\overline { { \mathcal { M } } } _ { i - 1 }$ propagated from the preceding coarser scale are resampled to the current resolution and combined with the current difusion-stream decoder feature ${ \mathcal { F } } _ { i } ^ { \mathrm { d i f f } }$ to obtain a preliminary displacement:

$$
\mathbf { u } _ { i } ^ { \mathrm { p r e } } = \mathcal { V } \left( \mathbf { u } _ { i - 1 } \right) + \mathcal { H } _ { i } ^ { \mathrm { p r e } } \left( \mathcal { F } _ { i } ^ { \mathrm { d i f f } } , \mathcal { V } \left( \mathbf { u } _ { i - 1 } \right) , \mathcal { V } \left( \overline { { \mathcal { M } } } _ { i - 1 } \right) \right) ,\tag{9}
$$

where $\tau ( \cdot ) , \overline { { ( \cdot ) } } .$ , and $\mathcal { H } _ { i } ^ { \mathrm { p r e } } ( \cdot )$ denote resampling, averaging, and the initial residual-flow head, respectively. $\overline { { \mathcal { M } } } _ { i - 1 }$ represents the average response of residual misalignment at the corresponding scale.

The preliminary displacement is then used to warp the corresponding encoder features. The original and warped features are integrated and subsequently processed by the feedback-attention module to generate the feedback map $\mathcal { M } _ { i , j } \colon$

$$
\begin{array} { r } { \mathcal { M } _ { i , j } = 1 - \mathcal { A } _ { i , j } . } \end{array}\tag{10}
$$

The scale-level feedback is then combined with the motion-estimation-stream decoder feature ${ \mathcal { F } } _ { i } ^ { \mathrm { m o t } }$ and the preliminary displacement to refine the displacement:

$$
\mathbf { u } _ { i } = \mathbf { u } _ { i } ^ { \mathrm { p r e } } + \mathcal { H } _ { i } ^ { \mathrm { r e f } } \left( \mathcal { F } _ { i } ^ { \mathrm { m o t } } , \mathbf { u } _ { i } ^ { \mathrm { p r e } } , \overline { { \mathcal { M } } } _ { i } \right) ,\tag{11}
$$

where $\mathcal { H } _ { i } ^ { \mathrm { r e f } } ( \cdot )$ denotes the residual-flow refinement head. The refined displacement $\mathbf { u } _ { i }$ is propagated to the next finer scale, while a detached copy of $\overline { { \mathcal { M } } } _ { i }$ serves as the feedback signal.

## 3.2.3. Motion-Estimation Objective

The motion-estimation objective combines difusion noise prediction, image similarity, displacement-field regularization, and anatomical correspondence. The anatomical correspondence term is defined as the Dice loss between the warped moving pseudo mask and the corresponding fixed pseudo mask, both obtained from the pretrained segmentation network. The difusion stream is supervised by the mean squared error between the sampled Gaussian noise � and the predicted noise $\hat { \epsilon } _ { t }$

$$
\mathcal { L } _ { \mathrm { n o i s e } } = \mathbf { M } \mathbf { S } \mathrm { E } \left( \epsilon , \hat { \epsilon } _ { t } \right) .\tag{12}
$$

The predicted noise is further used to modulate the image-similarity objective, following the difusion-guided weighting strategy of FSDifReg (Qin and Li, 2023). Specifically, it is transformed into a bounded spatial weighting map:

$$
\omega _ { t } = \left[ \sigma \left( \hat { \epsilon } _ { t } \right) \right] ^ { \gamma } ,\tag{13}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function and $\gamma$ controls the weighting strength. Rather than treating $\hat { \epsilon } _ { t }$ as an explicit score function, $\omega _ { t }$ is used as a noise-predictionguided weighting signal for local image similarity. Given the warped moving image $\hat { I } _ { f } = \mathscr { W } ( I _ { m } , \mathbf { u } _ { m  f } )$ , the weighted similarity loss is defined as

$$
\mathcal { L } _ { \mathrm { s i m } } = - \operatorname* { m e a n } \left[ \omega _ { t } \odot \mathrm { L C C } _ { 9 \times 9 } \left( \hat { I } _ { f } , I _ { f } \right) \right] ,\tag{14}
$$

where $\mathrm { L C C _ { 9 \times 9 } }$ denotes the local cross-correlation computed within a $9 \times 9$ window. Together with the displacement-field regularization and anatomical correspondence terms, $\mathcal { L } _ { \mathrm { n o i s e } }$ and $ { \mathcal { L } } _ { \mathrm { s i m } }$ constitute the motion-estimation objective ${ \mathcal { L } } _ { \mathrm { m o t } } .$

## 3.3. Self-distillation-based Joint Learning

As illustrated in Fig. 3, anatomical segmentation and unsupervised motion estimation are jointly refined through

![](images/9bff8a9f2d2654291ef6908b1a76629e311934e109b8f21a5f48967a6983e48a.jpg)  
Figure 3: The proposed self-distillation-based joint learning strategy. Student and Teacher segmentation networks are employed, with the Teacher updated using EMA. Reliability-aware self-distillation and bidirectional anatomy–motion consistency are introduced in EchoDiST. Anatomical consistency is imposed in both moving-to-fixed and fixed-to-moving directions, while only the moving-to-fixed branch is shown for clarity. The Student prediction is warped to the fixed frame using the estimated displacement field and constrained by an EMA Teacher-derived anatomical target. A sample-level reliability weight modulates the combined region-level Dice and pixel-wise MSE consistency losses, enabling joint optimization of anatomical segmentations and motion estimation.

EMA-based self-distillation and reliability-weighted crossframe consistency.

To provide more stable anatomical guidance, we adopt an EMA-based self-distillation scheme following the Mean Teacher paradigm (Tarvainen and Valpola, 2017). Let $f _ { S } ( \cdot ; \theta _ { S } )$ and $f _ { T } ( \cdot ; \theta _ { T } )$ denote the Student and Teacher segmentation networks, respectively. The Student is initialized from the segmentation model pretrained in Stage 1, with the Teacher initialized using the same parameters. During joint training, the Student is updated by gradient descent, whereas the Teacher parameters are updated as an exponential moving average of the Student:

$$
\theta _ { T }  \alpha \theta _ { T } + ( 1 - \alpha ) \theta _ { S } ,\tag{15}
$$

where $\alpha$ denotes the EMA decay factor. The Teacher therefore provides EMA-stabilized anatomical targets without direct back-propagation.

Given $I _ { m }$ and $I _ { f } , f _ { S } ( \cdot ; \theta _ { S } )$ predicts the moving-frame foreground probability map $P _ { m } ^ { S }$ , while the motion-estimation network predicts the forward displacement field $\mathbf { u } _ { m  f } .$ . The probability map $P _ { m } ^ { S }$ is then propagated to the fixed frame as

$$
\widetilde { P } _ { m  f } ^ { S } = \mathcal { W } ( P _ { m } ^ { S } , \mathbf { u } _ { m  f } ) .\tag{16}
$$

Meanwhile, the Teacher prediction for $I _ { f }$ is converted into a hard foreground target $\hat { Y } _ { f } ^ { T }$

The forward anatomical consistency loss is defined as

$$
\mathcal { L } _ { \mathrm { c o n } } ^ { m  f } = \mathcal { L } _ { \mathrm { d i c e } } ( \widetilde { P } _ { m  f } ^ { S } , \hat { Y } _ { f } ^ { T } ) + \lambda _ { c } \mathcal { L } _ { \mathrm { m s e } } ( \widetilde { P } _ { m  f } ^ { S } , \hat { Y } _ { f } ^ { T } )\tag{17}
$$

The Dice term constrains region-level anatomical overlap, while the MSE term further regularizes pixel-wise agreement. Because $\mathbf { u } _ { m  f }$ remains attached to the computational graph during warping, this consistency loss jointly optimizes anatomical segmentation and motion estimation. The same constraint is applied in the reverse direction to obtain $\mathscr { L } _ { \mathrm { c o n } } ^ { f  m }$

Since Teacher-derived anatomical targets may vary in reliability across image pairs and motion directions, we further introduce a reliability-aware weighting strategy inspired by uncertainty-aware self-ensembling (Yu et al., 2019). For each direction, three normalized reliability scores are computed from the Teacher outputs: $q _ { \mathrm { c o n f } }$ measures the foreground-conditioned confidence of the moving and fixed predictions, while $q _ { \mathrm { a g r } }$ and $q _ { \mathrm { s i z e } }$ respectively quantify the anatomical agreement and foreground-area consistency between the warped moving and fixed predictions. The sample-level reliability is defined as

$$
q _ { m  f } = \frac { \alpha _ { \mathrm { c o n f } } q _ { \mathrm { c o n f } } + \alpha _ { \mathrm { a g r } } q _ { \mathrm { a g r } } + \alpha _ { \mathrm { s i z e } } q _ { \mathrm { s i z e } } } { \alpha _ { \mathrm { c o n f } } + \alpha _ { \mathrm { a g r } } + \alpha _ { \mathrm { s i z e } } } , q _ { m  f } \in [ 0 , 1 ] .\tag{18}
$$

The same formulation is applied to the reverse direction to obtain $q _ { f  m } .$ . Each component is mapped to [0, 1] before weighted aggregation. The displacement field used for reliability estimation is detached from the computational graph, and the resulting reliability score is treated as a stop-gradient sample weight in the subsequent anatomical consistency objective.

The overall joint-training objective is

$$
\mathcal { L } _ { \mathrm { j o i n t } } = \mathcal { L } _ { \mathrm { m o t } } + \lambda _ { \mathrm { s e g } } \mathcal { L } _ { \mathrm { s e g } } + q _ { m  f } \lambda _ { \mathrm { m f } } \mathcal { L } _ { \mathrm { c o n } } ^ { m  f } + q _ { f  m } \lambda _ { \mathrm { f m } } \mathcal { L } _ { \mathrm { c o n } } ^ { f  m } .\tag{19}
$$

## 4. Experiments

## 4.1. Datasets and Preprocessing

Three echocardiographic datasets were used in this study, comprising two clinical cohorts, i.e., CAMUS (Leclerc et al., 2019) and EchoNet-Dynamic (Ouyang et al., 2020), and a CAMUS simulation dataset (Evain et al., 2022). The CAMUS dataset was used for model development, with its apical four-chamber (4CH) sequences used for training and in-domain evaluation and its apical two-chamber (2CH) sequences used for cross-view evaluation. The CA-MUS simulation dataset and EchoNet-Dynamic dataset were treated as external test datasets and used to evaluate crossdataset generalization without dataset-specific fine-tuning. All images in the three datasets were resized to $2 5 6 \times 2 5 6$ pixels using bilinear interpolation. Grayscale intensities were scaled to [0, 1] for anatomical segmentation. For motion-estimation training, each image was independently intensity-normalized and linearly mapped to [−1, 1]. Unless otherwise specified, motion estimation was performed from end-diastole (ED) to end-systole (ES), with the ED frame treated as the moving frame and the ES frame as the fixed frame. For full-cycle analyses, ED served as the reference frame, and motion was estimated from ED to each frame throughout the cardiac cycle.

CAMUS. The CAMUS dataset (Leclerc et al., 2019) comprises apical two-chamber (2CH) and four-chamber (4CH) echocardiographic sequences from 500 patients, with manual delineations of the left ventricular (LV) cavity, myocardium, and left atrium (LA) provided at ED and ES. The CAMUS-4CH cohort was split at the patient level into 350 training, 50 validation, and 100 held-out test subjects. Of the 350 training subjects, 50 with manual anatomical annotations were used for segmentation pretraining, while all 350 subjects were used for motion-estimation training. The corresponding 100-subject CAMUS-2CH test cohort was used exclusively for cross-view evaluation without additional fine-tuning.

EchoNet-Dynamic. EchoNet-Dynamic (Ouyang et al., 2020) comprises 10,030 apical 4CH echocardiographic videos with cardiac functional measurements and expert LV tracings at ED and ES. We used a fixed subset of 110 cases with complete videos and corresponding LV tracings. This subset was defined independently of model predictions and downstream evaluation metrics. The oficial LV tracings were converted into closed contours and binary cavity masks for evaluation. The annotated frames with the largest and smallest LV cavity areas were identified as ED and ES, respectively.

![](images/fd4f8e58e8f36c72b891099602a77cafaa1216f4e854402d83ba260c2874758b.jpg)  
Figure 4: Reference myocardial point configuration in the CAMUS simulation dataset, comprising five transmural layers with 36 points distributed along each layer. The enlarged region illustrates the local radial-tangential basis, where the radial direction $\mathbf { e } _ { r }$ is defined from the corresponding inner-layer point toward the outer-layer point, and the tangential direction $\mathbf { e } _ { t }$ is defined as the in-plane direction orthogonal to $\mathbf { e } _ { r }$

CAMUS simulation dataset. The CAMUS simulation dataset (Evain et al., 2022) contains simulated apical 4CH echocardiographic sequences with reverberation artifacts for 100 patients from the CAMUS dataset. Each case comprises a B-mode image sequence, time-varying myocardial meshes, and ED/ES frame indices, from which reference myocardial motion throughout the cardiac cycle can be derived. The myocardial meshes are represented by 180 points organized into five transmural layers at each frame, with 36 points distributed circumferentially along each layer. We denote each point as $p _ { i , j } .$ , where $i \in \{ 0 , \ldots , 4 \}$ indexes the transmural layer from the inner to the outer myocardial boundary and $j \in \{ 0 , \ldots , 3 5 \}$ denotes the circumferential position within each layer. Points sharing the same circumferential index � across diferent layers define corresponding transmural locations, as illustrated in Fig. 4.

## 4.2. Implementation Details

The experiments were conducted on an RTX A6000- equipped server for training and evaluation, with all network code implemented in PyTorch.

Training followed the three-stage strategy described in Section 3. In Stage 1, the anatomical segmentation network was pretrained for 100 epochs using AdamW with a batch size of 8 and a weight decay of $1 \times 1 0 ^ { - 4 }$ . The learning rates were set to $1 \times 1 0 ^ { - 5 }$ for the pretrained EchoFM backbone and $1 \times 1 0 ^ { - 4 }$ for the segmentation heads. The motion-estimation network was optimized for 100 epochs in both Stage 2 and Stage 3 using Adam with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 1. During Stage 2, the forward difusion process used 1000 timesteps with a linear variance schedule from $1 \times 1 0 ^ { - 6 } \mathrm { t o } 1 \times 1 0 ^ { - 2 }$ . In Stage 3, the Student segmentation network was optimized using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ , a weight decay of $1 \times 1 0 ^ { - 4 }$ , and a batch size of 8, while the Teacher network was updated after each optimization step using EMA with a decay factor of 0.995.

The EchoFM encoder in the motion-estimation network remained frozen during Stages 2 and 3. At inference, motion estimation was performed by a single deterministic forward pass without reverse-difusion sampling.

## 4.3. Evaluation Metrics

## 4.3.1. Anatomical Alignment

In our experiments, we employed three metrics to assess anatomical alignment performance: the Dice similarity coeficient (Dice) (Dice, 1945), the 95th-percentile Hausdorf distance (HD95) (Huttenlocher et al., 1993), and the average symmetric surface distance (ASSD) (Heimann et al., 2009). Metrics were averaged over the LV cavity, myocardium, and LA for CAMUS; for the CAMUS simulation dataset and EchoNet-Dynamic, they were computed on the myocardium and LV cavity, respectively. HD95 and ASSD were reported in millimeters for CAMUS and the CAMUS simulation dataset and in pixels for EchoNet-Dynamic. A better alignment is characterized by a higher Dice score and lower HD95 and ASSD values.

For the CAMUS simulation dataset with available myocardial point correspondences, endpoint error (EPE) was used to quantify motion accuracy (Baker et al., 2011). EPE was defined as the Euclidean distance between corresponding predicted and reference points, computed in physical coordinates and reported in millimeters.

## 4.3.2. Myocardial Strain Assessment

Myocardial strain was evaluated on the CAMUS simulation dataset at both local and global levels.

Local myocardial strain. Local myocardial strain was quantified in the ED-defined radial-tangential coordinate system illustrated in Fig. 4. At each myocardial position, the radial direction was defined from the corresponding inner-layer point toward the outer-layer point, while the tangential direction was defined as the in-plane direction orthogonal to the radial direction. Consistent with previous echocardiographic motion studies that derive directional myocardial strain from estimated motion fields in a cardiac coordinate system (Ahn et al., 2023), local displacement gradients were used to derive the Green-Lagrange strain tensor, which was then transformed from the axial-lateral image coordinates into the local radial-tangential coordinate system. The radial and tangential components were used for evaluation, while the associated shear component was not included in the reported metrics. The same procedure was applied to the predicted and reference motion fields. ES radial- and tangential-strain errors were quantified over the 180 myocardial points using root mean square error (RMSE) and reported in percent (%).

Global myocardial strain. ED was used as the reference frame for full-cycle strain estimation, consistent with standard echocardiographic displacement analysis (Mor-Avi et al., 2011; Voigt et al., 2015). Myocardial motion was estimated directly from ED to each subsequent frame. Global tangential strain was defined as the spatial average of local tangential strain over the myocardium at each frame. Agreement between predicted and reference strain trajectories was evaluated using curve RMSE and Pearson correlation �, with RMSE reported in percent (%).

Global longitudinal strain (GLS) was derived geometrically from the relative change in endocardial contour length with respect to its ED reference length. The endocardial contour was represented as an open curve extending from one mitral annular point, through the ventricular apex, to the opposite mitral annular point, without closure across the mitral valve plane. Predicted and reference GLS trajectories were compared using curve RMSE and Pearson correlation $r ,$ with RMSE reported in percent (%).

## 4.3.3. Motion-Derived Functional and Cardiac-Phase Assessment

Area-based functional accuracy. On EchoNet-Dynamic, we assessed whether the estimated motion preserved LV systolic contraction using the available expert ED/ES annotation. Similar to the relative ED-to-ES volume change used in conventional ejection fraction, an area-based ejection fraction (Area-EF) was defined as

$$
{ \mathrm { A r e a - E F } } = { \frac { A _ { \mathrm { E D } } - A _ { \mathrm { E S } } } { A _ { \mathrm { E D } } } } \times 1 0 0 \% .\tag{20}
$$

Reference Area-EF was computed from the annotated ED and ES cavity areas, whereas predicted Area-EF was computed from the ED cavity and its motion-propagated ES area. Agreement was evaluated using RMSE and Pearson correlation �, with RMSE reported in percent (%).

ED/ES frame prediction. We further assessed whether the proposed method can identify the ED and ES frames. Given an annotated LV mask, the estimated displacement fields were used to propagate the mask to diferent time points and obtain the corresponding LV-area trajectory. Since the LV area is generally largest at ED and smallest at ES, the frames with the maximum and minimum predicted LV areas were identified as the predicted ED and ES frames, respectively. ES localization error quantified the absolute frame diference between the annotated ES frame and the frame of minimum predicted LV area within the ED-to-ES interval, while monotonic consistency measured the proportion of consecutive frame transitions over this interval for which the predicted LV area did not increase.

## 4.3.4. Statistical Analysis

Statistical comparisons were conducted at the case level. Continuous paired endpoints were analyzed using two-sided Wilcoxon signed-rank tests. Multiple comparisons were corrected using the Holm procedure, and adjusted �-values below 0.05 were considered statistically significant. Diferences between dependent Pearson correlations on EchoNet-Dynamic were assessed using paired case-level bootstrap resampling.

## 4.4. Comparative Evaluation

## 4.4.1. Comparison Methods

We compared EchoDiST with seven representative learningbased methods: DdC-AC-DLIR (Hasan et al., 2025), NICE-Trans (Meng et al., 2023), NICE-Net (Meng et al., 2022),

Quantitative comparison of anatomical alignment on CAMUS-4CH, CAMUS-2CH, EchoNet-Dynamic, and the CAMUS simulation dataset. All metrics are reported as mean ± standard deviation over 100, 100, 110, and 100 test cases for the four datasets, respectively, with one ED-to-ES image pair evaluated per case. �-values indicate paired statistical comparisons of Dice between EchoDiST and each comparison method.
<table><tr><td rowspan="2">Method</td><td colspan="4">CAMUS-4CH</td><td colspan="4">CAMUS-2CH</td></tr><tr><td>Dice↑</td><td>HD95↓</td><td>ASSD↓</td><td>p-value</td><td>Dice↑</td><td>HD95↓</td><td>ASSD↓</td><td>p-value</td></tr><tr><td>DdC-AC-DLIR</td><td> $0 . 8 8 6 \pm 0 . 0 3 3$ </td><td> $7 . 4 7 \pm 2 . 7 2$ </td><td> $2 . 8 8 \pm 0 . 9 5$ </td><td>&lt; 0.001</td><td> $0 . 8 7 2 \pm 0 . 0 3 9$ </td><td> $8 . 6 6 \pm 2 . 9 8$ </td><td> $3 . 4 4 \pm I . I 6$ </td><td>&lt; 0.001</td></tr><tr><td>NICE-Trans</td><td> $0 . 8 6 3 \pm 0 . 0 4 9$ </td><td> $9 . 2 5 \pm 3 . 8 9$ </td><td> $3 . 4 6 \pm 1 . 5 5$ </td><td>&lt; 0.001</td><td> $0 . 8 4 8 \pm 0 . 0 5 2$ </td><td> $1 0 . 4 3 \pm 4 . 1 8$ </td><td> $4 . 0 7 \pm 1 . 6 5$ </td><td>&lt; 0.001</td></tr><tr><td>NICE-Net</td><td> $0 . 8 4 9 \pm 0 . 0 4 8$ </td><td> $9 . 9 2 \pm 3 . 8 5$ </td><td> $3 . 7 0 \pm 1 . 4 3$ </td><td>&lt; 0.001</td><td> $0 . 8 3 9 \pm 0 . 0 5 2$ </td><td> $1 1 . 0 5 \pm 4 . 2 8$ </td><td> $4 . 2 2 \pm 1 . 5 6$ </td><td>&lt; 0.001</td></tr><tr><td>CorrMLP</td><td> $0 . 8 4 8 \pm 0 . 0 5 3$ </td><td> $1 0 . 2 2 \pm 4 . 1 7$ </td><td> $3 . 8 3 \pm 1 . 6 5$ </td><td>&lt; 0.001</td><td> $0 . 8 3 5 \pm 0 . 0 5 7$ </td><td> $1 0 . 9 6 \pm 4 . 0 8$ </td><td> $4 . 3 7 \pm 1 . 7 6$ </td><td>&lt; 0.001</td></tr><tr><td>SDHNet</td><td> $0 . 8 6 8 \pm 0 . 0 4 7$ </td><td> $9 . 7 3 \pm 4 . 4 0$ </td><td> $3 . 4 2 \pm 1 . 2 8$ </td><td>&lt; 0.001</td><td> $0 . 8 6 1 \pm 0 . 0 4 6$ </td><td> $1 0 . 2 3 \pm 4 . 2 7$ </td><td> $3 . 7 9 \pm 1 . 3 8$ </td><td>&lt; 0.001</td></tr><tr><td>TransMorph2D</td><td> $0 . 8 4 9 \pm 0 . 0 5 5$ </td><td> $1 0 . 7 3 \pm 4 . 8 0$ </td><td> $4 . 0 3 \pm 1 . 6 9$ </td><td>&lt; 0.001</td><td> $0 . 8 4 5 \pm 0 . 0 5 4$ </td><td> $1 0 . 9 2 \pm 4 . 5 1$ </td><td> $4 . 3 2 \pm 1 . 6 9$ </td><td>&lt; 0.001</td></tr><tr><td>VoxelMorph</td><td> $0 . 8 2 4 \pm 0 . 0 5 1$ </td><td> $1 2 . 7 5 \pm 4 . 3 9$ </td><td> $4 . 6 4 \pm 1 . 5 7$ </td><td>&lt; 0.001</td><td> $0 . 8 1 9 \pm 0 . 0 5 2$ </td><td> $1 2 . 8 7 \pm 4 . 4 1$ </td><td> $5 . 0 2 \pm 1 . 6 5$ </td><td>&lt; 0.001</td></tr><tr><td>Ours</td><td> ${ \bf 0 . 9 0 9 \pm 0 . 0 2 2 }$ </td><td> ${ \bf 5 . 9 6 \pm 1 . 8 0 }$ </td><td> $\mathbf { 2 . 2 9 \pm 0 . 6 8 }$ </td><td>ref.</td><td> $\mathbf { 0 . 8 9 2 \pm 0 . 0 3 2 }$ </td><td> ${ \bf 7 . 4 7 \pm 2 . 6 5 }$ </td><td> $\mathbf { 2 . 8 9 \pm 1 . 0 3 }$ </td><td>ref.</td></tr><tr><td rowspan="2"></td><td colspan="4">EchoNet-Dynamic</td><td colspan="4">CAMUS Simulation</td></tr><tr><td>Dice↑</td><td>HD95↓</td><td>ASSD↓</td><td>p-value</td><td>Dice↑</td><td>HD95↓</td><td>ASSD↓</td><td>p-value</td></tr><tr><td>DdC-AC-DLIR</td><td> $0 . 7 8 1 \pm 0 . 0 7 7$ </td><td> $7 . 3 7 \pm 2 . 3 4$ </td><td> $3 . 3 1 \pm 1 . 1 3$ </td><td>&lt; 0.001</td><td> $0 . 7 8 1 \pm 0 . 0 6 0$ </td><td> $5 . 7 0 \pm 1 . 6 9$ </td><td> $2 . 1 2 \pm 0 . 5 8$ </td><td>&lt; 0.001</td></tr><tr><td>NICE-Trans</td><td> $0 . 7 8 3 \pm 0 . 0 7 4$ </td><td> $7 . 7 7 \pm 2 . 5 0$ </td><td> $3 . 2 1 \pm 1 . 1 5$ </td><td>&lt; 0.001</td><td> $0 . 8 2 0 \pm 0 . 0 5 9$ </td><td> $5 . I 2 \pm I . 6 2$ </td><td> $I . 7 3 \pm 0 . 5 5$ </td><td>&lt; 0.001</td></tr><tr><td>NICE-Net</td><td> $0 . 7 6 0 \pm 0 . 0 7 5$ </td><td> $7 . 5 5 \pm 2 . 3 7$ </td><td> $3 . 6 5 \pm 1 . 1 6$ </td><td>&lt; 0.001</td><td> $0 . 7 8 2 \pm 0 . 0 5 7$ </td><td> $5 . 7 1 \pm 1 . 4 8$ </td><td> $2 . 1 5 \pm 0 . 5 4$ </td><td>&lt; 0.001</td></tr><tr><td>CorrMLP</td><td> $0 . 7 8 4 \pm 0 . 0 7 5$ </td><td> $7 . 7 0 \pm 2 . 4 8$ </td><td> $3 . 1 4 \pm 1 . 1 4$ </td><td>&lt; 0.001</td><td> $0 . 7 9 7 \pm 0 . 0 5 8$ </td><td> $5 . 8 0 \pm 1 . 6 5$ </td><td> $1 . 9 4 \pm 0 . 5 6$ </td><td>&lt; 0.001</td></tr><tr><td>SDHNet</td><td> $0 . 7 0 7 \pm 0 . 0 8 6$ </td><td> $9 . 1 4 \pm 2 . 5 3$ </td><td> $4 . 3 9 \pm 1 . 2 8$ </td><td>&lt; 0.001</td><td> $0 . 6 8 9 \pm 0 . 0 6 9$ </td><td> $7 . 6 1 \pm 1 . 8 3$ </td><td> $2 . 9 7 \pm 0 . 6 0$ </td><td>&lt; 0.001</td></tr><tr><td>TransMorph2D</td><td> $0 . 8 3 5 \pm 0 . 0 7 I$ </td><td> $6 . 3 6 \pm 2 . 6 3$ </td><td> $2 . 5 0 \pm { I . O 8 }$ </td><td>&lt; 0.001</td><td> $0 . 8 0 6 \pm 0 . 0 5 4$ </td><td> $5 . 6 1 \pm 1 . 8 9$ </td><td> $1 . 9 8 \pm 0 . 5 7$ </td><td>&lt; 0.001</td></tr><tr><td>VoxelMorph</td><td> $0 . 7 7 9 \pm 0 . 0 7 2$ </td><td> $7 . 3 9 \pm 2 . 4 5$ </td><td> $3 . 3 9 \pm 1 . 1 5$ </td><td>&lt; 0.001</td><td> $0 . 7 7 2 \pm 0 . 0 6 1$ </td><td> $5 . 7 8 \pm 1 . 5 8$ </td><td> $2 . 2 3 \pm 0 . 5 6$ </td><td>&lt; 0.001</td></tr><tr><td>Ours</td><td> $\mathbf { 0 . 8 8 8 \pm 0 . 0 5 4 }$ </td><td> $\mathbf { 4 . 1 9 } \pm 2 . 0 4$ </td><td> ${ \bf 1 . 6 7 \pm 0 . 8 4 }$ </td><td>ref.</td><td> ${ \bf 0 . 8 7 8 \pm 0 . 0 3 7 }$ </td><td> ${ \bf 3 . 5 8 \pm 1 . 2 4 }$ </td><td> ${ \bf 1 . 2 0 \pm 0 . 3 3 }$ </td><td>ref.</td></tr></table>

CorrMLP (Meng et al., 2024), SDHNet (Zhou et al., 2023), TransMorph2D (Chen et al., 2022), and VoxelMorph (Balakrishnan et al., 2019). For a fair comparison, methods requiring retraining were trained using the same CAMUS-4CH training dataset as EchoDiST, while their original method-specific objectives and optimization settings were retained.

## 4.4.3. Myocardial Motion and Strain Assessment

## 4.4.2. Anatomical Alignment and Generalization

Table 2 reports the quantitative results of myocardial point motion and local strain estimation on the CAMUS simulation dataset. EchoDiST achieved the lowest point-motion error as well as the lowest radial- and tangential-strain errors, indicating consistently improved accuracy in both myocardial motion tracking and local strain estimation. Subjectlevel comparisons further showed significant improvements in EPE over all comparison methods $( p ~ < ~ 0 . 0 0 1 )$ , with significant improvements also observed for tangential strain $( p < 0 . 0 5 )$

The anatomical alignment performance of various methods across four datasets is detailed in Table 1. For alignment accuracy, our EchoDiST achieves the highest Dice and the lowest HD95 and ASSD across all four evaluation settings. DdC-AC-DLIR achieves the strongest baseline performance on CAMUS, whereas NICE-Trans and Trans-Morph2D are more competitive on the CAMUS simulation dataset and EchoNet-Dynamic, respectively. In contrast, our approach consistently outperforms on all metrics, demonstrating more robust anatomical alignment across diferent views and datasets. Subject-level paired statistical analysis further confirms that the improvements over all comparison methods are significant for all three metrics on each dataset (all Holm-adjusted $p < 0 . 0 0 1 )$ .

Fig. 5 and 6 visually compare the alignment results of diferent methods, showing that EchoDiST achieves closer agreement between the warped and target anatomical boundaries, particularly around the ventricular wall and apex.

The qualitative results obtained with selected representative methods in Fig. 7 further support these findings. EchoDiST more closely reproduces the reference myocardial motion in both displacement direction and magnitude, while preserving the spatial organization of motion across the myocardial wall. The corresponding radial and tangential strain maps also show spatial patterns and regional variations that are more consistent with the reference distributions, whereas the comparison methods exhibit larger local discrepancies.

The advantage extends to full-cycle global strain assessment. As reported in Table 3, EchoDiST achieved the lowest curve RMSE and the highest Pearson correlation for both GTS and GLS, demonstrating closer agreement with the reference global strain trajectories. These improvements were statistically significant compared with all comparison methods for both global strain measures $( p < 0 . 0 0 1 )$

Fig. 8 illustrates the GTS and GLS trajectories estimated by EchoDiST and the comparison methods over the cardiac cycle. EchoDiST closely follows the reference trajectories, capturing both the magnitude and cardiac-phase evolution of myocardial strain. In contrast, the comparison methods show larger deviations and fluctuations from the reference curves over the cardiac cycle.

![](images/06176227ff9a764c4d6d99c0549e3d816050a63e399c8fc4c9a0054e67cc4b8f.jpg)  
Figure 5: Qualitative comparison of diferent methods on the CAMUS dataset. The first column shows the ES fixed image and the ED moving image with anatomical contours, and the remaining columns show the results of the diferent methods. The dataset corresponding to each two-row group is indicated along the left margin. Within each group, the first row presents the warped moving image with the fixed anatomy in green and the warped moving anatomy in red, while the second row presents the corresponding displacement grid. The numbers displayed on the upper-left corner of each warped-image panel indicate the Dice score for the representative ED-to-ES image pair shown.

![](images/19cc08fd1fd36d1bab566aa25ae0a60f376155050613e6dbc7ea9b2d825922c1.jpg)  
Figure 6: Qualitative comparison of diferent methods on EchoNet-Dynamic and the CAMUS simulation dataset. The first column shows the ES fixed image and the ED moving image with anatomical contours, and the remaining columns show the results of the diferent methods. The dataset corresponding to each two-row group is indicated along the left margin. Within each group, the first row presents the warped moving image with the fixed anatomy in green and the warped moving anatomy in red, while the second row presents the corresponding displacement grid. The numbers displayed on the upper-left corner of each warped-image panel indicate the Dice score for the representative ED-to-ES image pair shown.

## 4.4.4. Motion-Derived Functional and Cardiac-Phase Assessment

Table 4 presents the motion-derived functional and cardiac-phase assessment of the comparison methods on EchoNet-Dynamic. EchoDiST achieved the lowest Area-EF RMSE and the highest Pearson correlation with the reference

Table 2  
Quantitative comparison of myocardial point motion and local strain estimation on the CAMUS simulation dataset. All metrics are reported as mean ± standard deviation over 100 test cases, with 180 myocardial points evaluated per case. �-values indicate paired statistical comparisons of EPE between EchoDiST and each comparison method.
<table><tr><td>Method</td><td>EPE (mm)↓</td><td>Radial RMSE (%)↓</td><td>Tangential RMSE (%)↓</td><td>p-value</td></tr><tr><td>DdC-AC-DLIR</td><td> $7 . 1 0 \pm 2 . 3 1$ </td><td> $I 7 . 4 6 \pm 5 . 5 2$ </td><td> $1 8 . 4 3 \pm 5 . 8 3$ </td><td>&lt; 0.001</td></tr><tr><td>NICE-Trans</td><td> $6 . 3 3 \pm 2 . 4 5$ </td><td> $2 4 . 5 6 \pm 6 . 1 1$ </td><td> $1 9 . 7 8 \pm 5 . 6 5$ </td><td>&lt; 0.001</td></tr><tr><td>NICE-Net</td><td> $7 . 3 1 \pm 2 . 2 0$ </td><td> $2 1 . 6 4 \pm 5 . 4 9$ </td><td> $2 2 . 6 1 \pm 5 . 7 3$ </td><td>&lt; 0.001</td></tr><tr><td>CorrMLP</td><td> $6 . 8 1 \pm 2 . 5 1$ </td><td> $2 7 . 7 7 \pm 6 . 4 9$ </td><td> $2 1 . 4 5 \pm 5 . 5 9$ </td><td>&lt; 0.001</td></tr><tr><td>SDHNet</td><td> $9 . 0 4 \pm 2 . 2 3 $ </td><td> $2 5 . 6 1 \pm 6 . 5 3$ </td><td> $2 5 . 0 2 \pm 6 . 3 2$ </td><td>&lt; 0.001</td></tr><tr><td>TransMorph2D</td><td> $6 . 7 4 \pm 2 . 4 2$ </td><td> $2 0 . 4 0 \pm 5 . 2 6$ </td><td> $I 8 . 6 7 \pm 5 . 9 I$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>VoxelMorph</td><td> $7 . 5 5 \pm 2 . 3 0$ </td><td> $2 5 . 0 4 \pm 6 . 4 6$ </td><td> $2 2 . 8 6 \pm 6 . 1 7$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>Ours</td><td> ${ \bf 4 . 9 8 \pm 1 . 5 8 }$ </td><td> ${ \bf 1 6 . 5 9 \pm 4 . 2 7 }$ </td><td> ${ \bf 1 6 . 5 0 \pm 5 . 3 3 }$ </td><td>ref.</td></tr></table>

![](images/ebf29695b9c1cb9162b03f9e7defd59162f858502bfc28d4f0ce4d4bb452088f.jpg)  
Figure 7: Qualitative comparison of selected representative methods for myocardial point motion and local strain estimation on the CAMUS simulation dataset. The first row shows ED-to-ES myocardial point-motion vectors, with reference and predicted motions shown in green and red, respectively. The second and third rows show the corresponding radial and tangential strain maps. EPE and strain RMSE are computed over the 180 myocardial points of the representative case and displayed on the upper-left corner of the corresponding motion and strain panels, respectively.

Quantitative comparison of full-cycle global myocardial strain estimation on the CAMUS simulation dataset. All metrics are reported as mean ± standard deviation over 100 test cases, with global tangential and longitudinal strain curves evaluated over the full cardiac cycle for each case. �-values indicate paired statistical comparisons of Curve RMSE between EchoDiST mode and each comparison method
<table><tr><td rowspan="2">Method</td><td colspan="3">Global Tangential Strain</td><td colspan="3">Global Longitudinal Strain</td></tr><tr><td>Curve RMSE (%)↓</td><td>Pearson correlation r ↑</td><td>p-value</td><td>Curve RMSE (%)↓</td><td>Pearson correlation r ↑</td><td>p-value</td></tr><tr><td>DdC-AC-DLIR</td><td> $6 . 6 7 \pm 2 . 6 8$ </td><td> $0 . 6 6 6 \pm 0 . 3 6 3$ </td><td>&lt; 0.001</td><td> $8 . 7 4 \pm 3 . 3 9$ </td><td> $0 . 6 4 7 \pm 0 . 3 6 I$ </td><td>&lt; 0.001</td></tr><tr><td>NICE-Trans</td><td> $5 . 5 7 \pm 2 . 6 7$ </td><td> $0 . 6 6 3 \pm 0 . 3 6 0$ </td><td>&lt; 0.001</td><td> $8 . 2 4 \pm 3 . 4 2$ </td><td> $0 . 5 3 7 \pm 0 . 4 2 2$ </td><td>&lt; 0.001</td></tr><tr><td>NICE-Net</td><td> $7 . 1 0 \pm 2 . 7 6$ </td><td> $0 . 6 0 5 \pm 0 . 4 4 9$ </td><td>&lt; 0.001</td><td> $9 . 7 6 \pm 3 . 5 0$ </td><td> $0 . 4 6 3 \pm 0 . 5 1 1$ </td><td>&lt; 0.001</td></tr><tr><td>CorrMLP</td><td> $6 . 7 4 \pm 2 . 9 5$ </td><td> $0 . 4 4 1 \pm 0 . 4 4 1$ </td><td>&lt; 0.001</td><td> $9 . 3 3 \pm 3 . 5 5$ </td><td> $0 . 3 6 4 \pm 0 . 4 7 9$ </td><td>&lt; 0.001</td></tr><tr><td>SDHNet</td><td> $1 0 . 0 4 \pm 3 . 3 8$ </td><td> $- 0 . 3 7 7 \pm 0 . 4 4 2$ </td><td>&lt; 0.001</td><td> $1 2 . 6 6 \pm 4 . 1 7$ </td><td> $- 0 . 4 0 7 \pm 0 . 4 9 0$ </td><td>&lt; 0.001</td></tr><tr><td>TransMorph2D</td><td> $5 . 5 3 \pm 3 . 0 0$ </td><td> $0 . 6 0 6 \pm 0 . 4 7 6$ </td><td>&lt; 0.001</td><td> $7 . 0 2 \pm 3 . 8 0$ </td><td> $0 . 5 8 8 \pm 0 . 4 9 8$ </td><td>&lt; 0.001</td></tr><tr><td>VoxelMorph</td><td> $7 . 3 9 \pm 2 . 9 2$ </td><td> $0 . 3 6 1 \pm 0 . 5 7 9$ </td><td>&lt; 0.001</td><td> $9 . 5 5 \pm 3 . 8 0$ </td><td> $0 . 2 0 4 \pm 0 . 5 9 1$ </td><td>&lt; 0.001</td></tr><tr><td>Ours</td><td> ${ \pm } \mathbf { 1 . 4 8 \pm 1 . 5 7 }$ </td><td> ${ \bf 0 . 9 4 6 \pm 0 . 0 9 5 }$ </td><td>ref.</td><td> $\pm \mathbf { \delta . 9 8 \pm 1 . 9 9 }$ </td><td> $\mathbf { 0 . 9 5 6 \pm 0 . 0 6 0 }$ </td><td>ref.</td></tr></table>

![](images/5fe62c609bcce7202608c1518b68c68d41b8f864c741b2d1053ff03bb08564fc.jpg)

![](images/978b6ad37ca85b415f16e964e8fc1adea6a557540015a23683377a2c5234545e.jpg)  
Figure 8: Qualitative comparison of full-cycle global myocardial strain trajectories estimated by diferent methods on the CAMUS simulation dataset. (a) Global tangential strain and (b) Global longitudinal strain estimated by diferent methods. The reference trajectory is shown in black, and the ES frame is indicated by a vertical reference line.

Table 4  
Quantitative comparison of area-based functional accuracy and cardiac-phase consistency on EchoNet-Dynamic. All metrics are evaluated over 110 test cases. Area-based functional metrics are reported at the cohort level, and temporal consistency metrics as mean ± standard deviation. �-values indicate case-level paired statistical comparisons for Area-EF between EchoDiST and each comparison method.
<table><tr><td rowspan="2">Method</td><td colspan="3">Area-based Functional Accuracy (Area-EF)</td><td colspan="2">Temporal Consistency</td></tr><tr><td>Area-EF RMSE (%)↓</td><td>Area-EF r ↑</td><td>p-value</td><td>ES Localization Error↓</td><td>Monotonic Consistency↑</td></tr><tr><td>DdC-AC-DLIR</td><td>26.90</td><td>0.383</td><td>&lt; 0.001</td><td> $2 . 5 3 \pm 4 . 3 4$ </td><td> $0 . 8 3 0 \pm 0 . 2 1 1$ </td></tr><tr><td>NICE-Trans</td><td>13.41</td><td>0.355</td><td>0.011</td><td> $1 . 1 7 \pm 0 . 5 7$ </td><td> $0 . 8 4 6 \pm 0 . 0 6 8$ </td></tr><tr><td>NICE-Net</td><td>33.93</td><td>0.212</td><td> $< 0 . 0 0 1$ </td><td> $3 . 1 7 \pm 5 . 5 4$ </td><td> $0 . 7 6 8 \pm 0 . 1 9 3$ </td></tr><tr><td>CorrMLP</td><td>18.57</td><td>0.472</td><td> $< 0 . 0 0 1$ </td><td> $I . I 6 \pm 0 . 3 4$ </td><td> $0 . 8 5 8 \pm 0 . 0 3 8$ </td></tr><tr><td>SDHNet</td><td>38.32</td><td>-0.209</td><td> $< 0 . 0 0 1$ </td><td> $6 . 1 3 \pm 7 . 9 4$ </td><td> $0 . 6 2 6 \pm 0 . 2 6 5$ </td></tr><tr><td>TransMorph2D</td><td>22.59</td><td>0.406</td><td> $< 0 . 0 0 1$ </td><td> $1 . 3 2 \pm 0 . 8 6$ </td><td> $0 . 8 5 0 \pm 0 . 0 7 6$ </td></tr><tr><td>VoxelMorph</td><td>33.88</td><td>0.303</td><td> $< 0 . 0 0 1$ </td><td> $3 . 5 1 \pm 4 . 7 7$ </td><td> $0 . 7 4 6 \pm 0 . 1 6 8$ </td></tr><tr><td>Ours</td><td>10.39</td><td>0.686</td><td>ref.</td><td> ${ \bf 1 . 0 8 \pm 2 . 8 5 }$ </td><td> $\mathbf { 0 . 8 8 0 \pm 0 . 1 2 4 }$ </td></tr></table>

![](images/30e3cfc503da763df9935d20e2261b4f643144fba3e264bc8b304512dcd57392.jpg)  
Figure 9: Qualitative comparison of normalized left ventricular area trajectories estimated by diferent methods on EchoNet-Dynamic. Reference LV-area annotations at ED and ES are shown as discrete black markers.

Area-EF, indicating more accurate recovery of ED-to-ES chamber contraction. This reduction in Area-EF RMSE was statistically significant compared with all comparison methods (all Holm-adjusted $p < 0 . 0 5 )$ .

Similar improvements were observed in cardiac-phase assessment, i.e., ED/ES prediction. EchoDiST yielded the lowest ES localization error and the highest monotonic consistency, indicating its potential for cardiac-phase assessment.

Fig. 9 compares the normalized LV-area trajectories estimated by the comparison methods. EchoDiST shows a more consistent decrease from ED to ES, with an LVarea reduction closer to the reference (black solid markers in Fig. 9). In contrast, the comparison methods tend to underestimate the reduction or exhibit larger frame-to-frame fluctuations.

## 4.5. Ablation Studies

To conduct the ablation study, joint anatomy-motion estimation and feedback refinement were selectively enabled or disabled to form four configurations, while all other components and training settings were kept unchanged. For convenience, the corresponding configurations are referred to as w/o Joint & Feedback, w/o Joint, w/o Feedback, and Full.

Table 6  
Quantitative ablation results of anatomical alignment on CAMUS-4CH, CAMUS-2CH, EchoNet-Dynamic, and the CAMUS simulation dataset. All metrics are reported as mean ± standard deviation over 100, 100, 110, and 100 test cases for the four datasets, respectively, with one ED-to-ES image pair evaluated per case. �-values indicate paired statistical comparisons of Dice between the full EchoDiST model and each ablated variant.
<table><tr><td></td><td></td><td colspan="4">CAMUS-4CH</td><td colspan="4">CAMUS-2CH</td></tr><tr><td></td><td>Joint Feedback</td><td>Dice↑</td><td>HD95↓</td><td>ASSD↓</td><td>p-value</td><td>Dice↑</td><td>HD95↓</td><td>ASSD↓</td><td>p-value</td></tr><tr><td></td><td></td><td> $0 . 8 8 8 \pm 0 . 0 3 2$ </td><td> $7 . 1 8 \pm 2 . 2 3$ </td><td> $2 . 8 5 \pm 0 . 8 9$ </td><td> $< 0 . 0 0 1$ </td><td> $0 . 8 6 9 \pm 0 . 0 4 0$ </td><td> $8 . 8 1 \pm 3 . 0 1$ </td><td> $3 . 5 4 \pm 1 . 2 7$ </td><td>&lt; 0.001</td></tr><tr><td></td><td>√</td><td> $0 . 9 0 8 \pm 0 . 0 2 4$ </td><td> $6 . 1 0 \pm 2 . 0 5$ </td><td> $2 . 3 1 \pm 0 . 7 3$ </td><td>0.479</td><td> $0 . 8 8 5 \pm 0 . 0 3 0$ </td><td> $8 . 2 6 \pm 2 . 7 9$ </td><td> $3 . 1 9 \pm 1 . 0 1$ </td><td>0.027</td></tr><tr><td>√</td><td></td><td> $0 . 8 6 9 \pm 0 . 0 4 1$ </td><td> $8 . 1 5 \pm 2 . 8 6$ </td><td> $3 . 3 6 \pm 1 . 2 2$ </td><td> $< 0 . 0 0 1$ </td><td> $0 . 8 5 4 \pm 0 . 0 4 3$ </td><td> $9 . 4 7 \pm 3 . 4 7$ </td><td> $4 . 0 0 \pm 1 . 4 6$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>√</td><td>√</td><td> ${ \bf 0 . 9 0 9 \pm 0 . 0 2 2 }$ </td><td> ${ \bf 5 . 9 6 \pm 1 . 8 0 }$ </td><td> $\mathbf { 2 . 2 9 \pm 0 . 6 8 }$ </td><td>ref.</td><td> $\mathbf { 0 . 8 9 2 \pm 0 . 0 3 2 }$ </td><td> ${ \bf 7 . 4 7 \pm 2 . 6 5 }$ </td><td> ${ \bf 2 . 8 9 \pm 1 . 0 3 }$ </td><td>ref.</td></tr><tr><td></td><td></td><td colspan="4">EchoNet-Dynamic</td><td colspan="4">CAMUS Simulation</td></tr><tr><td></td><td>Joint Feedback</td><td>Dice↑</td><td>HD95↓</td><td>ASSD↓</td><td>p-value</td><td>Dice↑</td><td>HD95↓</td><td>ASSD↓</td><td>p-value</td></tr><tr><td></td><td></td><td> $0 . 7 9 4 \pm 0 . 0 6 9$ </td><td> $6 . 7 0 \pm 2 . 1 7$ </td><td> $3 . 0 5 \pm 0 . 9 8$ </td><td>&lt; 0.001</td><td> $0 . 7 4 8 \pm 0 . 0 4 9$ </td><td> $7 . 2 2 \pm 1 . 6 1$ </td><td> $2 . 5 3 \pm 0 . 5 0$ </td><td>&lt; 0.001</td></tr><tr><td></td><td>√</td><td> $0 . 8 3 0 \pm 0 . 0 6 3$ </td><td> $6 . 0 1 \pm 2 . 0 3$ </td><td> $2 . 5 1 \pm 0 . 9 1$ </td><td>0.002</td><td> $0 . 7 8 6 \pm 0 . 0 4 7$ </td><td> $6 . 7 8 \pm 1 . 8 1$ </td><td> $2 . 1 6 \pm 0 . 5 1$ </td><td>&lt; 0.001</td></tr><tr><td>√</td><td></td><td> $0 . 8 3 0 \pm 0 . 0 6 6$ </td><td> $5 . 8 4 \pm 2 . 2 4$ </td><td> $2 . 5 7 \pm 1 . 0 2$ </td><td> $< 0 . 0 0 1$ </td><td> $0 . 8 0 8 \pm 0 . 0 4 3$ </td><td> $5 . 4 3 \pm 1 . 3 8$ </td><td> $1 . 9 3 \pm 0 . 4 1$ </td><td>&lt; 0.001</td></tr><tr><td>√</td><td>√</td><td> $\mathbf { 0 . 8 8 8 \pm 0 . 0 5 4 }$ </td><td> $\mathbf { 4 . 1 9 } \pm 2 . 0 4$ </td><td> ${ \bf 1 . 6 7 \pm 0 . 8 4 }$ </td><td>ref.</td><td> ${ \bf 0 . 8 7 8 \pm 0 . 0 3 7 }$ </td><td> ${ \bf 3 . 5 8 \pm 1 . 2 4 }$ </td><td> ${ \bf 1 . 2 0 \pm 0 . 3 3 }$ </td><td>ref.</td></tr></table>

Quantitative ablation results of myocardial point motion and local strain estimation on the CAMUS simulation dataset. All metrics are reported as mean ± standard deviation over 100 test cases, with 180 myocardial points evaluated per case. �-values indicate paired statistical comparisons of EPE between the full EchoDiST model and each ablated variant.
<table><tr><td>Joint</td><td>Feedback</td><td>EPE (mm)↓</td><td>Radial RMSE (%)↓</td><td>Tangential RMSE (%)↓</td><td>p-value</td></tr><tr><td></td><td></td><td> $8 . 1 0 \pm 2 . 0 0$ </td><td> $1 9 . 7 1 \pm 5 . 8 0$ </td><td> $2 1 . 0 0 \pm 4 . 8 0$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td></td><td>√</td><td> $7 . 8 3 \pm 2 . 0 3$ </td><td> $2 6 . 2 1 \pm 5 . 4 0$ </td><td> $2 7 . 1 6 \pm 5 . 2 1$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>√</td><td></td><td> $6 . 2 3 \pm 1 . 7 4$ </td><td> $1 8 . 8 2 \pm 4 . 4 2$ </td><td> $1 7 . 6 9 \pm 3 . 9 2$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>√</td><td>√</td><td> ${ \bf 4 . 9 8 \pm 1 . 5 8 }$ </td><td> ${ \bf 1 6 . 5 9 \pm 4 . 2 7 }$ </td><td> ${ \bf 1 6 . 5 0 \pm 5 . 3 3 }$ </td><td>ref.</td></tr></table>

Quantitative ablation results of full-cycle global myocardial strain estimation on the CAMUS simulation dataset. All metrics are reported as mean ± standard deviation over 100 test cases, with global tangential and longitudinal strain curves evaluated over the full cardiac cycle for each case. �-values indicate paired statistical comparisons of Curve RMSE between the full EchoDiST model and each ablated variant.
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Global Tangential Strain</td><td colspan="3">Global Longitudinal Strain</td></tr><tr><td></td><td>Joint Feedback Curve RMSE (%)↓ Pearson correlation r ↑ p-value</td><td></td><td></td><td>Curve RMSE (%)↓ Pearson correlation r ↑ p-value</td><td></td></tr><tr><td></td><td></td><td> $8 . 3 6 \pm 2 . 4 0$ </td><td> $0 . 5 7 1 \pm 0 . 4 1 9$ </td><td> $< 0 . 0 0 1$ </td><td> $1 0 . 6 9 \pm 2 . 9 7$ </td><td> $0 . 6 0 5 \pm 0 . 3 8 7$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td></td><td>√</td><td> $8 . 0 0 \pm 2 . 7 3$ </td><td> $0 . 5 0 5 \pm 0 . 4 8 1$ </td><td> $< 0 . 0 0 1$ </td><td> $1 0 . 1 3 \pm 3 . 2 3$ </td><td> $0 . 5 8 5 \pm 0 . 4 0 9$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>V√</td><td></td><td> $5 . 7 1 \pm 2 . 1 1$ </td><td> $0 . 8 0 2 \pm 0 . 2 2 8$ </td><td> $< 0 . 0 0 1$ </td><td> $7 . 2 7 \pm 2 . 7 6$ </td><td> $0 . 8 2 9 \pm 0 . 1 6 3$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td></td><td>√</td><td> ${ \pm } \mathbf { 1 . 4 8 \pm 1 . 5 7 }$ </td><td> ${ \bf 0 . 9 4 6 \pm 0 . 0 9 5 }$ </td><td>ref.</td><td> $\mathbf { 2 . 9 8 \pm 1 . 9 9 }$ </td><td> $\mathbf { 0 . 9 5 6 \pm 0 . 0 6 0 }$ </td><td>ref.</td></tr></table>

Table 5 reports the anatomical alignment results across the four datasets. Feedback refinement alone recovered much of the alignment performance, particularly on CAMUS-4CH, whereas joint learning alone showed a less uniform efect across datasets. Combining both components yielded the best performance. The Full model significantly outperformed w/o Joint & Feedback and w/o Feedback for all three metrics across all datasets. Fig. 10 and 11 provide further qualitative evidence for the benefit of combining the two components.

For myocardial motion and strain assessment, the two components showed diferent efects when evaluated individually. Retaining joint anatomy-motion estimation reduced both point-motion and strain errors, whereas feedback refinement alone slightly reduced point-motion error but did not improve local strain estimation. As summarized in Tables 6 and 7, combining the two components yielded the lowest EPE and local strain errors, together with the lowest curve RMSE and the highest correlation for GTS and GLS. The Full model also significantly outperformed all three ablation variants for both global strain measures (all Holm-adjusted $\begin{array} { r l r } { p } & { { } < } & { 0 . 0 0 1 ) } \end{array}$ ). Fig. 12 and 13 further support these findings, showing more accurate point motion and more consistent local strain distributions and full-cycle strain trajectories.

GT  
w/o Joint & Feedback  
w/o Joint  
![](images/a5d4234a0e2d434ec2091bfe7209c268e47a02a1ad6e800c15aa1389b538a830.jpg)  
Figure 10: Qualitative comparison of diferent model variants on the CAMUS dataset. The first column shows the ES fixed image and the ED moving image with anatomical contours, and the remaining columns show the alignment results of diferent model variants. The dataset corresponding to each two-row group is indicated along the left margin. Within each group, the first row presents the warped moving image with the fixed anatomy in green and the warped moving anatomy in red, while the second row presents the corresponding displacement grid. The numbers displayed on the upper-left corner of each warped-image panel indicate the Dice score for the representative ED-to-ES image pair shown.

Table 8  
Quantitative ablation results of area-based functional accuracy and cardiac-phase consistency on EchoNet-Dynamic. All metrics are evaluated over 110 test cases. Area-based functional metrics are reported at the cohort level, and temporal consistency metrics as mean ± standard deviation. �-values indicate case-level paired statistical comparisons for Area-EF between the full EchoDiST model and each ablated variant.
<table><tr><td rowspan="2">Joint</td><td rowspan="2">Feedback</td><td colspan="3">Area-based Functional Accuracy (Area-EF)</td><td colspan="2">Temporal Consistency</td></tr><tr><td>Area-EF RMSE (%)↓</td><td>Area-EF r ↑</td><td>p-value</td><td>ES Localization Error↓</td><td>Monotonic Consistency↑</td></tr><tr><td></td><td></td><td>19.43</td><td>0.620</td><td>&lt; 0.001</td><td> $1 . 1 5 \pm 0 . 6 5$ </td><td> $0 . 7 9 1 \pm 0 . 0 3 4$ </td></tr><tr><td></td><td>√</td><td>17.42</td><td>0.683</td><td> $< 0 . 0 0 1$ </td><td> $1 . 5 3 \pm 1 . 8 7$ </td><td> $0 . 8 6 4 \pm 0 . 0 6 4$ </td></tr><tr><td>√</td><td></td><td>35.20</td><td>0.533</td><td> $< 0 . 0 0 1$ </td><td> $5 . 8 6 \pm 8 . 2 9$ </td><td> $0 . 6 4 1 \pm 0 . 2 7 1$ </td></tr><tr><td>√</td><td>√</td><td>10.39</td><td>0.686</td><td>ref.</td><td> ${ \bf 1 . 0 8 \pm 2 . 8 5 }$ </td><td> $\mathbf { 0 . 8 8 0 \pm 0 . 1 2 4 }$ </td></tr></table>

This complementary benefit was also evident on EchoNet-Dynamic. With both components enabled, EchoDiST achieved the strongest overall functional and cardiac-phase performance (Table 8), with a significant improvement in Area-EF RMSE over all ablation variants. Fig. 14 further illustrates a smoother and more consistent ED-to-ES LV-area trajectory with fewer frame-to-frame fluctuations.

Overall, the ablation results indicate complementary contributions of joint anatomy-motion estimation and feedback refinement, with their combination providing the most consistent performance across the evaluated tasks.

## 5. Discussion

The main findings of this study are as follows: 1) EchoDiST improves echocardiographic motion estimation by jointly learning anatomical segmentation and motion under limited anatomical supervision; 2) maintains consistent performance across diferent echocardiographic views and external test datasets, with the improvements extending beyond anatomical alignment to myocardial motion and mechanical assessment; and 3) supports motion-derived functional and cardiac-phase assessment on independently acquired clinical echocardiographic data.

![](images/826a29931fb9a919edb945af0ed068072d1c113adaabe71276e786ccd811af92.jpg)  
Figure 11: Qualitative comparison of diferent model variants on EchoNet-Dynamic and the CAMUS simulation dataset. The first column shows the ES fixed image and the ED moving image with anatomical contours, and the remaining columns show the alignment results of diferent model variants. The dataset corresponding to each two-row group is indicated along the left margin. Within each group, the first row presents the warped moving image with the fixed anatomy in green and the warped moving anatomy in red, while the second row presents the corresponding displacement grid. The numbers displayed on the upper-left corner of each warped-image panel indicate the Dice score for the representative ED-to-ES image pair shown.

EchoDiST employs a self-distillation-based joint learning mechanism for anatomical segmentation and myocardial motion estimation rather than treating anatomy as an independent auxiliary task. Anatomical segmentation provides structural information about cardiac regions that complements intensity-based correspondence in echocardiographic images. By incorporating this information into motion optimization, cross-frame correspondence is learned from both image intensity and cardiac structure.

The self-distillation-based joint learning mechanism forms a reciprocal interaction. During optimization, the estimated motion propagates anatomical segmentations across frames, while the agreement between the propagated and target anatomy contributes in turn to motion estimation. In this way, anatomical segmentations and motion estimation constrain each other rather than being optimized independently. This interaction may help reduce local inconsistencies in myocardial motion that have limited influence on segmentation accuracy but become more apparent in strain assessment. Feedback refinement serves a diferent role by correcting residual mismatches during coarse-to-fine motion estimation.

Although this study evaluated the proposed method on three datasets, the available reference data impose important limitations. In the CAMUS simulation dataset, dense point trajectories facilitate controlled evaluation of myocardial motion and strain, but these trajectories are generated by the simulation procedure and should not be interpreted as physiological ground-truth motion. Evain et al. (2022) also noted that local motion may difer from realistic myocardial behavior despite plausible global motion patterns, and that the simulated reverberation model does not capture the full complexity of artifacts encountered in clinical echocardiography. On EchoNet-Dynamic, expert annotations are available only at ED and ES, precluding direct validation of myocardial motion at intermediate frames.

![](images/5272f37fca9f14bef33e4335f003fe3fb87c428b7340573b79c0d01e3c0e9f84.jpg)  
Figure 12: Qualitative comparison of diferent model variants for myocardial point motion and local strain estimation on the CAMUS simulation dataset. The first row shows ED-to-ES myocardial point-motion vectors, with reference and predicted motions shown in green and red, respectively. The second and third rows show the corresponding radial and tangential strain maps. EPE and strain RMSE are computed over the 180 myocardial points of the representative case and displayed on the upper-left corner of the corresponding motion and strain panels, respectively.

![](images/5b344d8d6381598b8bd8096650a8d28cc2412acb37f2a2ace1b79407888fed1d.jpg)

![](images/6ed20c3a14831f634d9e3594f478760841656a4d3fa9b510c808e7cd8aa761b9.jpg)  
Figure 13: Qualitative comparison of full-cycle global myocardial strain trajectories estimated by diferent model variants on the CAMUS simulation dataset. (a) Global tangential strain and (b) Global longitudinal strain estimated by diferent model variants. The reference trajectory is shown in black, and the ES frame is indicated by a vertical reference line.

Three additional limitations should be noted. First, the multi-stage training procedure and difusion-conditioned motion estimation increase the computational cost of model development, although inference requires only a single deterministic forward pass. Second, model development was restricted to CAMUS-4CH. Although evaluation on CAMUS-2CH, the CAMUS simulation dataset, and EchoNet-Dynamic without dataset-specific fine-tuning demonstrates transfer across the tested settings, broader generalization across institutions, scanners, acquisition protocols, image quality, and patient populations remains to be established. In addition, the current framework can only estimate two-dimensional in-plane motion and cannot capture out-of-plane myocardial displacement. Future work will focus on simplifying the training procedure and extending validation to larger multicenter and more heterogeneous echocardiographic cohorts, as well as to multi-view or three-dimensional echocardiography with richer frame-wise clinical motion references.

![](images/976dae246c6b1fa5b36a141db89f089a38e00264bf135f276bc4ee2d7b4ceca3.jpg)  
Figure 14: Qualitative comparison of normalized left ventricular area trajectories estimated by diferent model variants on EchoNet-Dynamic. Reference LV-area annotations at ED and ES are shown as discrete black markers.

## 6. Conclusion

This study proposes a difusion-conditioned framework (EchoDiST) for echocardiographic motion estimation. It combines a difusion-conditioned motion-estimation network that uses difusion perturbations as auxiliary conditioning with an EMA-based self-distillation strategy for jointly learning anatomical segmentation and motion estimation. The self-distillation strategy enables joint learning using only limited annotations, allowing the two tasks to be optimized together without extensive manual labeling.

Experimental results across diferent echocardiographic views and datasets demonstrate consistent improvements in myocardial motion estimation as well as in mechanical, functional, and cardiac-phase assessment. These findings support the potential of EchoDiST for quantitative analysis of cardiac motion and function from echocardiographic images.

## CRediT authorship contribution statement

Feiyue Qi: Writing – review & editing, Writing – original draft, Visualization, Software, Methodology, Formal analysis, Data curation, Conceptualization. Xingyue Wei: Writing – review & editing, Methodology, Data curation, Investigation, Formal analysis. Jianwen Luo: Writing – review & editing, Funding acquisition, Methodology, Supervision, Resources.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data Availability

The datasets analyzed in this study are publicly available from their respective repositories. CAMUS, EchoNet-Dynamic, and the CAMUS simulation dataset were used in accordance with their respective data-access requirements. No new patient data were collected for this study. The source code for EchoDiST is not publicly available at the time of submission and will be released upon publication.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used OpenAI ChatGPT in order to assist with language refinement and manuscript editing. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (NSFC) under Grants 62531004 and 62027901.

## References

Ahn, S.S., Ta, K., Thorn, S.L., Onofrey, J.A., Melvinsdottir, I.H., Lee, S., Langdon, J., Sinusas, A.J., Duncan, J.S., 2023. Co-attention spatial transformer network for unsupervised motion tracking and cardiac strain analysis in 3D echocardiography. Med. Image Anal. 84, 102711. doi:10. 1016/j.media.2022.102711.

Baker, S., Scharstein, D., Lewis, J.P., Roth, S., Black, M.J., Szeliski, R., 2011. A database and evaluation methodology for optical flow. Int. J. Comput. Vis. 92, 1–31. doi:10.1007/s11263-010-0390-2.

Balakrishnan, G., Zhao, A., Sabuncu, M.R., Guttag, J., Dalca, A.V., 2019. VoxelMorph: a learning framework for deformable medical image registration. IEEE Trans. Med. Imaging 38, 1788–1800. doi:10.1109/TMI. 2019.2897538.

Chen, J., Frey, E.C., He, Y., Segars, W.P., Li, Y., Du, Y., 2022. TransMorph: Transformer for unsupervised medical image registration. Med. Image Anal. 82, 102615. doi:10.1016/j.media.2022.102615.

Dice, L.R., 1945. Measures of the amount of ecologic association between species. Ecology 26, 297–302. doi:10.2307/1932409.

Evain, E., Sun, Y., Faraz, K., Garcia, D., Saloux, E., Gerber, B.L., De Craene, M., Bernard, O., 2022. Motion estimation by deep learning in 2D echocardiography: Synthetic dataset and validation. IEEE Trans. Med. Imaging 41, 1911–1924. doi:10.1109/TMI.2022.3151606.

Hasan, M.K., Luo, Y., Yang, G., Yap, C.H., 2025. Feedback attention to enhance unsupervised deep learning image registration in 3D echocardiography. IEEE Trans. Med. Imaging 44, 2230–2243. doi:10.1109/TMI. 2025.3530501.

He, Y., Li, T., Ge, R., Yang, J., Kong, Y., Zhu, J., Shu, H., Yang, G., Li, S., 2022. Few-shot learning for deformable medical image registration with perception-correspondence decoupling and reverse teaching. IEEE J. Biomed. Health Inform. 26, 1177–1187. doi:10.1109/JBHI.2021.3095409.

Heimann, T., van Ginneken, B., Styner, M.A., Arzhaeva, Y., Aurich, V., Bauer, C., Beck, A., Becker, C., Beichel, R., Bekes, G., Bello, F., Binnig, G., Bischof, H., Bornik, A., Cashman, P.M.M., Chi, Y., Cordova, A., Dawant, B.M., Fidrich, M., Furst, J.D., Furukawa, D., Grenacher, L., Hornegger, J., Kainmüller, D., Kitney, R.I., Kobatake, H., Lamecker, H., Lange, T., Lee, J., Lennon, B., Li, R., Li, S., Meinzer, H.P., Nemeth, G., Raicu, D.S., Rau, A.M., van Rikxoort, E.M., Rousson, M., Rusko,

L., Saddi, K.A., Schmidt, G., Seghers, D., Shimizu, A., Slagmolen, P., Sorantin, E., Soza, G., Susomboon, R., Waite, J.M., Wimmer, A., Wolf, I., 2009. Comparison and evaluation of methods for liver segmentation from CT datasets. IEEE Trans. Med. Imaging 28, 1251–1265. doi:10. 1109/TMI.2009.2013851.

Ho, J., Jain, A., Abbeel, P., 2020. Denoising difusion probabilistic models, in: Advances in Neural Information Processing Systems, pp. 6840–6851.

Huang, X., Dione, D.P., Compas, C.B., Papademetris, X., Lin, B.A., Bregasi, A., Sinusas, A.J., Staib, L.H., Duncan, J.S., 2014. Contour tracking in echocardiographic sequences via sparse representation and dictionary learning. Med. Image Anal. 18, 253–271. doi:10.1016/j.media.2013.10. 012.

Huttenlocher, D.P., Klanderman, G.A., Rucklidge, W.J., 1993. Comparing images using the Hausdorf distance. IEEE Trans. Pattern Anal. Mach. Intell. 15, 850–863. doi:10.1109/34.232073.

Kim, B., Han, I., Ye, J.C., 2022. DifuseMorph: Unsupervised deformable image registration using difusion model, in: European Conference on Computer Vision, Springer. pp. 347–364. doi:10.1007/ 978-3-031-19821-2\_20.

Kim, S., Jin, P., Song, S., Chen, C., Li, Y., Ren, H., Li, X., Liu, T., Li, Q., 2025. EchoFM: Foundation model for generalizable echocardiogram analysis. IEEE Trans. Med. Imaging 44, 4049–4062. doi:10.1109/TMI. 2025.3580713.

Leclerc, S., Smistad, E., Pedrosa, J., Østvik, A., Cervenansky, F., Espinosa, F., Espeland, T., Berg, E.A.R., Jodoin, P.M., Grenier, T., Lartizien, C., Dhooge, J., Lovstakken, L., Bernard, O., 2019. Deep learning for segmentation using an open large-scale dataset in 2D echocardiography. IEEE Trans. Med. Imaging 38, 2198–2210. doi:10.1109/TMI.2019. 2900516.

Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S., 2017. Feature pyramid networks for object detection, in: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE. pp. 936–944. doi:10.1109/CVPR.2017.106.

Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang, Z., Dong, L., Wei, F., Guo, B., 2022. Swin Transformer V2: Scaling up capacity and resolution, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE. pp. 11999– 12009. doi:10.1109/CVPR52688.2022.01170.

Meng, M., Bi, L., Feng, D., Kim, J., 2022. Non-iterative coarse-to-fine registration based on single-pass deep cumulative learning, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 88–97. doi:10.1007/978-3-031-16446-0\_9.

Meng, M., Bi, L., Fulham, M., Feng, D., Kim, J., 2023. Non-iterative coarse-to-fine transformer networks for joint afine and deformable image registration, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 750–760. doi:10.1007/978-3-031-43999-5\_71.

Meng, M., Feng, D., Bi, L., Kim, J., 2024. Correlation-aware coarse-to-fine MLPs for deformable medical image registration, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE. pp. 9645–9654. doi:10.1109/CVPR52733.2024.00921.

Mor-Avi, V., Lang, R.M., Badano, L.P., Belohlavek, M., Cardim, N.M., Derumeaux, G., Galderisi, M., Marwick, T., Nagueh, S.F., Sengupta, P.P., Sicari, R., Smiseth, O.A., Smulevitz, B., Takeuchi, M., Thomas, J.D., Vannan, M., Voigt, J.U., Zamorano, J.L., 2011. Current and evolving echocardiographic techniques for the quantitative evaluation of cardiac mechanics: ASE/EAE consensus statement on methodology and indications endorsed by the japanese society of echocardiography. Eur. J. Echocardiogr. 12, 167–205. doi:10.1093/ejechocard/jer021.

van Mourik, M.J.W., Zaar, D.V.J., Smulders, M.W., Heijman, J., Lumens, J., Dokter, J.E., Passos, V.L., Schalla, S., Knackstedt, C., Schummers, G., Gjesdal, O., Edvardsen, T., Bekkers, S.C.A.M., 2019. Adding speckletracking echocardiography to visual assessment of systolic wall motion abnormalities improves the detection of myocardial infarction. J. Am. Soc. Echocardiogr. 32, 65–73. doi:10.1016/j.echo.2018.09.007.

Ouyang, D., He, B., Ghorbani, A., Yuan, N., Ebinger, J., Langlotz, C.P., Heidenreich, P.A., Harrington, R.A., Liang, D.H., Ashley, E.A., Zou, J.Y., 2020. Video-based AI for beat-to-beat assessment of cardiac

function. Nature 580, 252–256. doi:10.1038/s41586-020-2145-8.

Papademetris, X., Sinusas, A.J., Dione, D.P., Constable, R.T., Duncan, J.S., 2002. Estimation of 3-d left ventricular deformation from medical images using biomechanical models. IEEE Trans. Med. Imaging 21, 786–800. doi:10.1109/TMI.2002.801163.

Parajuli, N., Lu, A., Stendahl, J.C., Zontak, M., Boutagy, N., Eberle, M., Alkhalil, I., O’Donnell, M., Sinusas, A.J., Duncan, J.S., 2016. Integrated dynamic shape tracking and RF speckle tracking for cardiac motion analysis, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 431–438. doi:10.1007/ 978-3-319-46720-7\_50.

Park, J.J., Park, J.B., Park, J.H., Cho, G.Y., 2018. Global longitudinal strain to predict mortality in patients with acute heart failure. J. Am. Coll. Cardiol. 71, 1947–1957. doi:10.1016/j.jacc.2018.02.064.

Qin, C., Bai, W., Schlemper, J., Petersen, S.E., Piechnik, S.K., Neubauer, S., Rueckert, D., 2018. Joint learning of motion estimation and segmentation for cardiac mr image sequences, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 472–480. doi:10.1007/978-3-030-00934-2\_53.

Qin, Y., Li, X., 2023. FSDifReg: Feature-wise and score-wise difusionguided unsupervised deformable image registration for cardiac images, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 655–665. doi:10.1007/ 978-3-031-43999-5\_62.

Risum, N., Jons, C., Olsen, N.T., Fritz-Hansen, T., Bruun, N.E., Hojgaard, M.V., Valeur, N., Kronborg, M.B., Kisslo, J., Sogaard, P., 2012. Simple regional strain pattern analysis to predict response to cardiac resynchronization therapy: rationale, initial results, and advantages. Am. Heart J. 163, 697–704. doi:10.1016/j.ahj.2012.01.025.

Shi, H., Lu, L., Yin, M., Zhong, C., Yang, F., 2023. Joint few-shot registration and segmentation self-training of 3D medical images. Biomed. Signal Process. Control 80, 104294. doi:10.1016/j.bspc.2022.104294.

Ta, K., Ahn, S.S., Stendahl, J.C., Sinusas, A.J., Duncan, J.S., 2020. A semi-supervised joint network for simultaneous left ventricular motion tracking and segmentation in 4d echocardiography, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 468–477. doi:10.1007/978-3-030-59725-2\_45.

Ta, K., Ahn, S.S., Stendahl, J.C., Sinusas, A.J., Duncan, J.S., 2021. Shaperegularized unsupervised left ventricular motion network with segmentation capability in 3D+time echocardiography, in: 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), IEEE. pp. 536–540. doi:10.1109/ISBI48211.2021.9433888.

Ta, K., Ahn, S.S., Thorn, S.L., Stendahl, J.C., Zhang, X., Langdon, J., Staib, L.H., Sinusas, A.J., Duncan, J.S., 2024. Multi-task learning for motion analysis and segmentation in 3D echocardiography. IEEE Trans. Med. Imaging 43, 2010–2020. doi:10.1109/TMI.2024.3355383.

Tarvainen, A., Valpola, H., 2017. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results, in: Advances in Neural Information Processing Systems, pp. 1195–1204.

Tsao, C.W., Aday, A.W., Almarzooq, Z.I., Alonso, A., Beaton, A.Z., Bittencourt, M.S., Boehme, A.K., Buxton, A.E., Carson, A.P., Commodore-Mensah, Y., Elkind, M.S.V., Evenson, K.R., Eze-Nliam, C., Ferguson, J.F., Generoso, G., Ho, J.E., Kalani, R., Khan, S.S., Kissela, B.M., Knutson, K.L., Levine, D.A., Lewis, T.T., Liu, J., Loop, M.S., Ma, J., Mussolino, M.E., Navaneethan, S.D., Perak, A.M., Poudel, R., Rezk-Hanna, M., Roth, G.A., Schroeder, E.B., Shah, S.H., Thacker, E.L., VanWagner, L.B., Virani, S.S., Voecks, J.H., Wang, N.Y., Yafe, K., Martin, S.S., 2022. Heart disease and stroke statistics—2022 update: a report from the american heart association. Circulation 145, e153–e639. doi:10.1161/CIR.0000000000001052.

Voigt, J.U., Pedrizzetti, G., Lysyansky, P., Marwick, T.H., Houle, H., Baumann, R., Pedri, S., Ito, Y., Abe, Y., Metz, S., Song, J.H., Hamilton, J., Sengupta, P.P., Kolias, T.J., d’Hooge, J., Aurigemma, G.P., Thomas, J.D., Badano, L.P., 2015. Definitions for a common standard for 2D speckle tracking echocardiography: consensus document of the EACVI/ASE/Industry task force to standardize deformation imaging. Eur. Heart J. Cardiovasc. Imaging 16, 1–11. doi:10.1093/ehjci/jeu184.

Wu, J., Pan, T., Gong, K., 2026. LDM-Morph: Latent difusion model guided deformable image registration. Pattern Recognit. 174, 112925. doi:10.1016/j.patcog.2025.112925.

Xu, H., Xue, T., Fan, J., Liu, D., Chen, Y., Zhang, F., Westin, C.F., Kikinis, R., O’Donnell, L.J., Cai, W., 2025. Medical image registration meets vision foundation model: Prototype learning and contour awareness, in: International Conference on Information Processing in Medical Imaging, Springer. pp. 79–93. doi:10.1007/978-3-031-96628-6\_6.

Xu, Z., Niethammer, M., 2019. DeepAtlas: Joint semi-supervised learning of image registration and segmentation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 420–429. doi:10.1007/978-3-030-32245-8\_47.

Yu, L., Wang, S., Li, X., Fu, C.W., Heng, P.A., 2019. Uncertainty-aware self-ensembling model for semi-supervised 3D left atrium segmentation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 605–613. doi:10.1007/ 978-3-030-32245-8\_67.

Zhang, K.W., Finkelman, B.S., Gulati, G., Narayan, H.K., Upshaw, J., Narayan, V., Plappert, T., Englefield, V., Smith, A.M., Zhang, C., Hundley, W.G., Ky, B., 2018. Abnormalities in 3-dimensional left ventricular mechanics with anthracycline chemotherapy are associated with systolic and diastolic dysfunction. JACC Cardiovasc. Imaging 11, 1059–1068. doi:10.1016/j.jcmg.2018.01.015.

Zhou, S., Hu, B., Xiong, Z., Wu, F., 2023. Self-distilled hierarchical network for unsupervised deformable image registration. IEEE Trans. Med. Imaging 42, 2162–2175. doi:10.1109/TMI.2023.3244333.

Zhuo, Y., Shen, Y., 2024. DifuseReg: Denoising difusion model for obtaining deformation fields in unsupervised deformable image registration, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 597–607. doi:10.1007/ 978-3-031-72069-7\_56.