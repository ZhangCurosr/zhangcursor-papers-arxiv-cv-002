# SPARC: Slice-to-volume Pipeline for Automated Reconstruction of gated 3D+time fetal Cardiac MRI

Arnaud Boutillon<sup>a,b,∗,1</sup>, Naomi Clarke<sup>a,c,1</sup>, Tomas Woodgate<sup>c</sup>, Daniel West<sup>b,d</sup>, Alina Schneider<sup>b</sup>, Rachael Franklin<sup>b</sup>, Anthony Price<sup>b,d</sup>, Jo Hajnal<sup>b,c</sup>, Kuberan Pushparajah<sup>c</sup>, David Lloyd<sup>c</sup> and Maria Deprez<sup>a,c</sup>

<sup>a</sup>Biomedical Computing Research Department, School ofBiomedical Engineering and Imaging Sciences, King’s College London, London, United Kingdom

<sup>b</sup>Imaging Physics and Engineering Research Department, School of Biomedical Engineering and Imaging Sciences, King’s College London, London, United Kingdom

<sup>c</sup>Early Life Imaging Research Department, School ofBiomedical Engineering and Imaging Sciences, King’s College London, London, United Kingdom <sup>d</sup>Medical Physics and Clinical Engineering, Guys and St Thomas’ NHS Foundation Trust, London, United Kingdom, London, United Kingdom

## A R T I C L E I N F O

Keywords:   
Fetal cardiac MRI   
Doppler-gated acquisition   
Slice-to-volume reconstruction   
Deep learning   
Domain adaptation   
Anatomical reorientation

## A BS T RA C T

Fetal cardiac MRI (fCMR) provides valuable diagnostic information complementary to echocardiography, particularly for complex congenital heart disease (CHD). Dynamic cine imaging captures cardiac motion essential for assessment of cardiac function; however, the reconstruction of 3D+time cine volumes from 2D+time acquired slices remains challenging due to unpredictable fetal motion and the absence of automated and robust processing tools suitable for clinical deployment. We present the SPARC pipeline (Slice-to-volume Pipeline for Automated Reconstruction of gated 3D+time fetal Cardiac MRI) which combines physics-informed slice-to-volume reconstruction (SVR) of Doppler ultrasound (DUS) gated stacks of slices, assisted by deep learning (DL) models for thoracic segmentation and anatomical reorientation. The proposed SVR algorithm achieves a tenfold reduction in reconstruction time relative to existing frame-wise approaches (4.8 ± 1.0 vs 49.0 ± 14.1 min, � < 0.0001) while improving the reconstruction quality. Thoracic segmentation performance using ensemble aggregation exceeded inter-rater agreement (Dice 84.7 ± 3.9% vs 81.4 ± 7.7%, � < 0.05), while anatomical reorientation achieved a success rate of 90.1%. End-to-end evaluation on a large held-out clinical cohort (� = 121) demonstrated fully automatic processing in 82.6% of cases with a mean runtime of 7.1 ± 1.3 min, compatible with clinical deployment. The complete SPARC pipeline is publicly available as a Docker container https://hub.docker.com/r/aboutill/sparc and is currently deployed at our institution as a clinical research tool.

## 1. Introduction

Congenital heart disease (CHD) afects approximately 1% of births worldwide (Liu et al., 2019), and prenatal detection is associated with improved postnatal outcomes (Gardiner et al., 2014; Holland et al., 2015; Lloyd et al., 2019). Two-dimensional (2D) echocardiography remains the primary modality for prenatal cardiac screening due to its widespread availability and ease of use. However, there is a longstanding need for complementary three-dimensional (3D) visualisation of fetal cardiac anatomy due to the inherent constraints of 2D imaging (Lloyd et al., 2016, 2019). Magnetic resonance imaging (MRI) has emerged as a valuable adjunct to fetal echocardiography (Roy et al., 2019; Dong et al., 2020; Mamalis et al., 2022; Lim et al., 2024; Vollbrecht et al., 2024a; Woodgate et al., 2024), with demonstrated utility in predicting the need for early postnatal intervention (Lloyd et al., 2021) and good agreement with both echocardiographic and postnatal findings (Moscatelli et al., 2023).

The fetal heart is a small, complex, and dynamic organ that presents unique imaging challenges (van Amerom et al., 2018; Woodgate et al., 2024). Fetal echocardiography is limited by acoustic window quality, which deteriorates towards the end of pregnancy, as well as maternal obesity and unfavourable fetal positions (Moscatelli et al., 2023). Fetal cardiac magnetic resonance (fCMR) faces distinct but complementary challenges, including maternal respiratory motion, fetal motion (Votino et al., 2012), the absence of an electrocardiogram (ECG) gating signal, and maternal discomfort (Vollbrecht et al., 2024a). Maternal breath-holds and fetal sedation can improve image quality but increase discomfort and anxiety (van Amerom et al., 2018).

In the non-fetal context, dynamic cardiac MRI is well established using ECG gating to synchronise acquisition with the cardiac cycle and reconstruct 3D+time cine volumes. ECG cannot be applied in the fetal context, necessitating alternative synchronisation strategies. For example, balanced steady-state free precession (bSSFP) cine sequences have enabled recognition of basic cardiovascular anatomy but remain susceptible to fetal motion (Geiger et al., 2023). Selfgating techniques, including metric optimised gating (MOG) and other image-based approaches (Roy et al., 2013; Haris et al., 2017; van Amerom et al., 2018), address the absence of ECG but introduce additional post-processing complexity and can perform suboptimally under maternal free-breathing conditions (Haris et al., 2020).

![](images/91e94c37df9faa5fd6b691cb273389416a2d01d58fd4c6888c0ba989277167ba.jpg)  
Figure 1: Doppler-gated acquisition of a single 2D+time slice. The bSSFP sequence acquires one �-space segment per TR, with the number of acquired cardiac phases per heartbeat given by $T _ { \mathsf { a c q } } = \lfloor \mathsf { R R } _ { \mathsf { I n s t } } / \mathsf { T R } \rfloor$ , where $\mathsf { R R } _ { \mathsf { I n s t } }$ is the instantaneous RR interval. Since $\mathsf { R R } _ { \mathsf { I n s t } }$ varies across heartbeats, $T _ { \mathsf { a c q } }$ is not constant. Temporal interpolation of each �-space line to � uniformly spaced cardiac phases, followed by a 2D Fourier transform, yields the reconstructed Doppler-gated 2D+time slice.

The recent introduction of an MR-compatible Doppler ultrasound (DUS) gating device (Kording et al., 2018a,b), that provides a cardiac gating signal analogous to the ECG (Figure 1), enables dynamic cine acquisition of the fetal heart. DUS-gated fCMR has been shown to provide cardiac output measurements within physiologically plausible ranges and additional diagnostic value over echocardiography alone (Ryd et al., 2021; Vollbrecht et al., 2023; Minocha et al., 2024). Existing studies combining DUS-gating with a breath-holding to acquire 2D+time standard views demonstrated clinical feasibility but remain limited by motion (Ryd et al., 2021; Vollbrecht et al., 2023). The feasibility of free-breathing 2D+time DUS-gated radial acquisition with within-slice Fourier domain motion compensation has been demonstrated by Haris et al. (2020). However, the extension of DUS-gated fCMR to motion-compensated 3D+time imaging has not yet been proposed.

In this work, we aim to reconstruct motion-compensated 3D+time fCMR from stacks of overlapping DUS-gated 2D+time slices acquired in multiple orientations in the presence of maternal respiration and fetal motion. Sliceto-volume reconstruction (SVR) methods were originally developed for 3D fetal brain reconstruction (Rousseau et al., 2005; Kuklisova-Murgasova et al., 2012; Gholipour and Warfield, 2019) and applied to reconstruct static 3D fetal heart (Lloyd et al., 2019; Uus et al., 2022). SVR has been extended for 3D+time fetal cardiac MRI (van Amerom et al., 2019), to reconstruct a cine volume spanning a single cardiac cycle from free-running, un-gated acquisitions by retrospectively estimating and synchronising the cardiac phase of each image frame using image-derived self-gating. While this approach demonstrated the feasibility of 3D+time fetal cardiac cine reconstruction, it required long processing times and substantial manual input including segmentation of the chest and heart and reorientation to canonical coordinate system, limiting its applicability in clinical practice.

Deep learning has recently been applied to fCMR to improve image quality and reduce acquisition time. The work of Vollbrecht et al. (2024b) demonstrated that deep learning denoising and super-resolution applied to Dopplergated fCMR produced reconstructions of higher diagnostic confidence. However, the absence of motion mitigation meant that fetal movement remained a significant source of image artefacts. Berggren et al. (2022) proposed an alternative strategy to reduce motion sensitivity by shortening scan times, although, re-acquisition or additional postprocessing was still required when fetal movement occurred. These approaches all underscore the importance of motion correction.

Uus et al. (2022) proposed an automated SVR pipeline for static 3D fCMR in which deep learning models provided automatic region of interest segmentation and landmark detection to recover large inter-stack displacements. Recent work has also formulated SVR as an optimisation problem using implicit neural representations (INR) (Sitzmann et al., 2020), ofering a promising alternative paradigm (Xu et al., 2023). However, such approaches have not yet been extended to fetal cardiac 3D+time MR imaging.

## 1.1. Contributions

In this work, we propose Slice-to-volume Pipeline for Automated Reconstruction of gated 3D+time fetal Cardiac MRI (SPARC), the first automated end-to-end pipeline for localisation, reconstruction and reorientation of 3D+time fetal cardiac cine volumes from Doppler-gated MRI stacks (Figure 2). This hybrid pipeline combines a physics-informed slice-to-volume reconstruction algorithm with deep learning models for thoracic segmentation and anatomical reorientation. The main contributions of this work are as follows:

![](images/3f1d8b2d4cb0b70cc4376557b656108a179883f0d7475f0cee756e0ab0b49f74.jpg)  
Figure 2: Multiple gated 2D+time stacks acquired in orthogonal orientations are combined via the SPARC pipeline to produce a 3D+time cine volume with � cardiac phases reoriented in canonical fetal coordinate system.

![](images/c563841c8412c7f3ea4a9932049fc4220d49ebaf2d5f373d26e717da74787dfe.jpg)  
Figure 3: Overview of the proposed SPARC pipeline.

1. A parsimonious slice-wise spatio-temporal forward model specifically adapted to Doppler-gated fetal cardiac cine reconstruction achieving a tenfold reduction in reconstruction time relative to frame-wise approaches (van Amerom et al., 2019) with improved generalisation to unseen data.

2. A transfer learning strategy for training deep learning models (i.e., thoracic segmentation, cardiac centre localisation, and reorientation) for Doppler-gating data, to take advantage of a larger free-running domain.

3. An end-to-end clinically validated pipeline, packaged as a Docker container and currently deployed at our institution, evaluated on a large held-out clinical cohort (� = 121) with 82.6% fully automatic success and a mean processing time of 7.1±1.3 min compatible with clinical deployment requirements.

## 2. Methodology

The proposed SPARC pipeline consists of three fully automated steps that transform stacks of thick, gated 2D+time slices into an isotropic cine volume of the fetal heart, reoriented into a standard fetal coordinate space (Figure 3). The steps are: (1) template stack selection and segmentation of fetal thorax (Sec. 2.2 & Sec. 2.3), (2) reconstruction of fetal heart 3D+time volume spanning a single cardiac cycle (Sec. 2.4), and (3) reorientation to the fetal anatomical coordinate system (Sec. 2.5). Before describing these modules, we briefly review Doppler-ultrasound gated MR acquisition and introduce the mathematical notation used in this work.

## 2.1. Doppler-gated MR acquisition

During acquisition, the Doppler-ultrasound device measures fetal cardiac wall motion and flow velocity to derive a gating signal transmitted to the MRI scanner (Kording et al., 2018a,b). During the acquisition of each 2D+time slice, the Doppler device triggers acquisition of part of the �-space at each cardiac cycle (Figure 1). Since the cardiac cycle duration varies across heartbeats due to physiological heart rate variability, the number of acquired cardiac phases per cycle is not constant. Temporal interpolation in �-space to a fixed number of reconstructed frames � is therefore applied. Ultimately, each slice spans a single cardiac cycle, with each frame composed from �-space samples collected over multiple heartbeats, and there is a one-to-one correspondence between temporal indices $t \in \{ 1 , \ldots , T \}$ and cardiac phases.

In this study, multiple stacks $\mathbf { s } _ { l }$ of thick, Doppler-gated 2D+time slices $\mathbf { y } _ { k }$ covering the fetal thorax were acquired in diferent orientations (i.e., axial, sagittal, coronal, and oblique). Let $\mathbf { y } _ { k } = ( \mathbf { y } _ { k , 1 } ^ { \top } , \ldots , \mathbf { y } _ { k , T } ^ { \top } ) ^ { \top } \in \bar { \mathbb { R } } ^ { N _ { k } \times T }$ denote the vector representation of the �-th slice, composed of � frames with $N _ { k }$ voxels each, and $\mathbf { s } _ { l } = \{ \mathbf { y } _ { k } ~ | ~ k \in \mathcal { E } _ { l } \}$ denote the �-th stack, where $\mathcal { E } _ { l } \subset \{ 1 , \ldots , K \}$ is the index set of slices belonging to that stack.

As a preprocessing step, all stacks were corrected for slice wise bias field inhomogeneity using the N4 algorithm (Tustison et al., 2010). Owing to the gated acquisition, the bias field was estimated from the first frame and the resulting correction was applied to all frames.

## 2.2. Motion-robust template selection

The axial stack exhibiting the lowest level of motion artefacts was automatically selected as the reconstruction template. Motion in each time-averaged stack $\overline { { \bf s } } _ { l }$ was evaluated as the normalised variance of adjacent slice diferences:

$$
\Delta _ { l } ^ { 2 } = \frac { \mathrm { V a r } ( \{ \| \bar { \mathbf { y } } _ { k } - \bar { \mathbf { y } } _ { k + 1 } \| _ { 1 } \mid ( k , k + 1 ) \in \mathcal { E } _ { l } ^ { 2 } \} ) } { ( \mathrm { A v g } ( \{ \| \bar { \mathbf { y } } _ { k } \| _ { 1 } \mid k \in \mathcal { E } _ { l } \} ) ) ^ { 2 } }\tag{1}
$$

This variance-based �-smoothness metric captures the instability between neighbouring slices induced by fetal motion. A low $\Delta _ { \scriptscriptstyle { l } } ^ { \scriptscriptstyle { 2 } }$ indicates stable fetal positioning throughout the stack, which is the primary requirement for a reliable reconstruction template.

## 2.3. Thoracic segmentation via transfer learning

Automatic thoracic segmentation was subsequently performed on the chosen template stack to define the region

SPARC: Automatic 3D+time fetal cardiac MRI reconstruction

![](images/2be6070db03c0409e4376c68a6e338844423b662121de41869bec77cd71f5c41.jpg)  
Figure 4: Overview of the slice-to-volume reconstruction algorithm.

of interest for reconstruction. A lightweight 3D UNet (Ronneberger et al., 2015) was trained on stacks acquired in axial, coronal, sagittal and oblique directions. Training incorporated extensive data augmentation, including random translations, rotations (in-plane and through-plane), scaling, and flipping, as well as intensity-based gamma transformation and bias-field augmentation. The agreement between predicted and manually annotated ground-truth masks was measured using an equally weighted combination of Dice and cross-entropy (CE) losses (Cardoso et al., 2022).

Due to the scarcity of annotated Doppler-gated data, a transfer learning strategy was adopted. The network was first trained on a large-scale free-running (i.e., non-gated) fetal cardiac dataset acquired on a diferent scanner (source domain $\mathit { D _ { S } } )$ , and subsequently fine-tuned on the target domain $D _ { T }$ (i.e., Doppler-gated) using transfer learning (Guan and Liu, 2022).

An ensemble strategy was adopted to enhance robustness and generalizability (Cardoso et al., 2022). Networks were trained on distinct splits of the training data, and their predictions were aggregated at inference via majority voting. The largest connected component was retained, and residual mask holes were filled to promote anatomical plausibility (Cardoso et al., 2022). A final binary morphological dilation (1×voxel size in-plane) was applied to the thorax mask to provide a safety margin against boundary uncertainty, reducing the risk of excluding cardiac structures during reconstruction when the heart lies adjacent to the thoracic wall.

## 2.4. Motion-corrected cine reconstruction

## 2.4.1. Spatio-temporalforward model

Acquired stacks of 2D+time slices are afected by interslice motion, as the fetus is moving freely and the mother is breathing normally. Fetal motion is assumed to be rigid since the fetal thorax is contained within the rib-cage, and the whole 2D+time slice is assumed to be in the same motionstate, because Fourier samples are acquired continuously and then retrospectively binned into cardiac phases.

The forward model is a linear matrix operation that simulates intensity-corrected frames $\mathbf { y } _ { k , t } ^ { * }$ from the 3D+time reconstructed cardiac anatomy $\mathbf { x } \in \mathbb { R } ^ { N \times T }$

$$
\begin{array} { r } { \mathbf { y } _ { k , t } ^ { * } = \mathbf { M } _ { k } \mathbf { x } _ { t } ; \quad \mathbf { y } _ { k , t } ^ { * } = a _ { k } \exp ( - \mathbf { b } _ { k } ) \circ \mathbf { y } _ { k , t } } \end{array}\tag{2}
$$

where $\mathbf { x } _ { t }$ is a reconstructed 3D frame corresponding to cardiac phase �. The sparse matrix $\mathbf { M } _ { k } \in \mathbb { R } ^ { N _ { k } \times \mathbf { \bar { N } } }$ is composed of rows of vectorised 3D point spread functions (PSF) corresponding to voxels of the acquired slice $\mathbf { y } _ { k } .$ , reoriented using motion parameters and sampled on the grid of reconstructed image �. Note that PSF is not dependent on time � and is applied to all frames equally. The PSF is determined by the acquisition protocol and it is approximated by a truncated 3D Gaussian function with full width at half maximum equal to the slice-thickness in the through-plane direction and 1.2×voxel size in-plane respectively (Kuklisova-Murgasova et al., 2012).

Intensity correction is modeled by applying of slicewise scales $a _ { k }$ and spatially smooth slice-wise bias fields $\mathbf { b } _ { k } .$ , where exp(.) is applied element wise to the vector $- \mathbf { b } _ { k }$ and ◦ denotes element wise multiplication. These intensity correction parameters model interactions of MRI signal with fetal motion, such as partial slice dropouts, spin history efects and diferential transmit and receive MRI field inhomogeneities (Kuklisova-Murgasova et al., 2012). These efects are assumed to be independent of the cardiac phase � and are applied to all frames of the 2D+time slices equally.

## 2.4.2. Slice-to-volume motion estimation

Motion correction is initialised by volumetric registration to align each stack $\mathbf { s } _ { l }$ to the previously selected template stack (Sec. 2.2), enabling an initial volume reconstruction, followed by iterations of slice-to-volume registration interleaved with super-resolution reconstruction (Fig. 4). Rigid transformations with normalized mutual information similarity metric were adopted for both stages (Kuklisova-Murgasova et al., 2012; van Amerom et al., 2018). Given the gated acquisition, all frames within a slice are assumed to share a common motion state. Consequently, a single 3D rigid transformation estimated from a single temporal frame is suficient to align the entire set of frames associated with that slice.

## 2.4.3. Spatio-temporal super-resolution reconstruction

Assuming that the spatial alignment, and consequently also PSF matrices $\mathbf { M } _ { k }$ are known, the cine volume � can be estimated by minimizing the weighted sum of squared residuals between the intensity-corrected acquired slices $\mathbf { y } _ { k t } ^ { * }$ and simulated slices $\mathbf { M } _ { k } \mathbf { x } _ { t }$ (Eq. 2) to ensure their consistency assumed in the forward model:

$$
\mathcal { L } _ { \mathrm { R e c o n s t r u c t i o n } } ( \mathbf { x } ) = \sum _ { k = 1 } ^ { K } \sum _ { t = 1 } ^ { T } \| \mathbf { W } _ { k , t } ( \mathbf { y } _ { k , t } ^ { * } - \mathbf { M } _ { k } \mathbf { x } _ { t } ) \| _ { 2 } ^ { 2 } + \lambda \mathcal { R } ( \mathbf { x } )\tag{3}
$$

where $\mathbf { W } _ { k t }$ are diagonal matrices with slice-voxel level weights $w _ { j , k , t } = q _ { k } p _ { j , k , t }$ with $p _ { j , k , t }$ representing voxel-level and $q _ { k }$ slice-level confidence in the acquired data.

Robust statistics matrices $\mathbf { W } _ { k } .$ were estimated within an expectation–maximization (EM) framework following (Kuklisova-Murgasova et al., 2012; van Amerom et al., 2018), by classification of each voxel and slice as an inlier or outlier based on the error terms $\mathbf { y } _ { k , t } ^ { * } - \mathbf { M } _ { k } \mathbf { x } _ { t }$ , thus rejecting corrupted and misaligned slices from the reconstruction.

Edge-preserving regularization (.) (Charbonnier et al., 1997; Kuklisova-Murgasova et al., 2012) was applied independently to each 3D cardiac frame to stabilize the reconstruction while limiting over-smoothing of anatomical structures. The regularization term was computed for each voxel based on its 26 spatial neighbours. Parameter � controls the strength of regularization, and was set to $\lambda = 0 . 0 2$

The intensity correction parameters $\mathbf { b } _ { k }$ and $a _ { k }$ are also estimated by minimising the loss (Eq. 3) jointly with the reconstruction in an interleaved manner (Fig. 4) following the approach of Kuklisova-Murgasova et al. (2012).

As a post-processing step, the cine volume is corrected for volumetric bias field inhomogeneity using the N4 algorithm (Tustison et al., 2010). Given the temporal consistency aforded by gated acquisition, the bias field was estimated from the first frame and propagated to all frames.

## 2.5. Anatomical reorientation

The position and orientation of the fetus in the template stack, and consequently in the reconstructed volume �, are unpredictable a priori. The cine volume must therefore be reoriented to a canonical fetal coordinate system to enable consistent anatomical interpretation and quantitative analysis. The proposed anatomical reorientation process was decomposed into two steps: (1) alignment to the fetal heart centre of mass to correct large translations (Sec. 2.5.1), and (2) reorientation to the fetal coordinate system using a Transformer network (sec. 2.5.2). The two stages were performed on the time-averaged reconstructed volume �̄ to improve signal-to-noise ratio (SNR) and enhance the robustness of anatomical landmark detection. �̄ is treated in this section as a 3D image (not a flattened vector).

## 2.5.1. Cardiac centre localization

We employed the same transfer-learning strategy to train a 3D U-Net segmentation of the fetal heart as in Sec. 2.3.

Extensive intensity and spatial augmentations, including full 3D rotations, were applied during training to promote generalisability. During inference, predictions from separate heart-segmentation networks trained on distinct data splits were combined using a majority voting strategy. Anatomical plausibility of the predicted heart mask was promoted via largest connected component selection, and residual mask holes filling. Fetal heart centre of mass was extracted from the predicted mask and used to recentre the image prior to reorientation.

## 2.5.2. Transformer-based 3D reorientation

Anchor point representation. Reorientation was formulated as an image regression task, where predicted transformations were parametrised using anchor-points similarly to Hou et al. (2018a). Three anchor points were defined: the image centre (corresponding to the fetal cardiac centre), and the left- and right-posterior midpoints of the image. These non-collinear points uniquely determine a rigid transformation. An orthonormal reference frame was constructed via Gram–Schmidt orthogonalization from the predicted anchors, from which the translation vector and rotation matrix were recovered. The obtained transformation enables anatomical reorientation of the image to the canonical fetal coordinate system.

Training strategy. We employed a standard vision Transformer (ViT) architecture to predict the spatial orientation. Manually curated fetal heart volumes in a canonical coordinate system were used for training. To simulate arbitrary rigid poses, random translations and rotations were applied, and the network was trained to predict the corresponding anchor-point locations in the transformed volume (Hou et al., 2018a,b). Translations were sampled uniformly within 20% of the image size along each axis. This restricted range was suficient due to previous centering step. Rotations were sampled over SO(3) using an Euler-angle parameterization, with $\theta \sim \mathcal { V } ( 0 , 2 \pi ) , \cos ( \varphi ) \sim \mathcal { V } ( - 1 , 1 )$ and $\psi \sim \mathcal { V } ( - \pi , \pi )$ ensuring uniform coverage of 3D orientations.

To promote invariance to mask shape and background context, mask-based augmentations were applied, including random erosion of the thorax mask and random dilation of the heart mask. Intensity augmentations, such as contrast perturbation and simulated bias-field inhomogeneity, were also employed to improve robustness and generalization.

Lossfunction and optimisation. The network parameters $\mathbf { \Theta } _ { \mathrm { V i T } }$ were optimized using a weighted combination of four complementary loss terms:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R e o r i e n t a t i o n } } ( \Theta _ { \mathrm { V i T } } ) = \lambda _ { 1 } \mathcal { L } _ { \mathrm { A n c h o r } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { G e o d e s i c } } } \\ { + \lambda _ { 3 } \mathcal { L } _ { \mathrm { T r a n s l a t i o n } } + \lambda _ { 4 } \mathcal { L } _ { \mathrm { I m a g e } } } \end{array}\tag{4}
$$

Here, ${ \mathcal { L } } _ { \mathrm { A n c h o r } }$ denotes the Euclidean distance between predicted and ground-truth anchor points; $\mathcal { L } _ { \mathrm { G e o d e s i c } }$ is the geodesic distance between the predicted and ground-truth rotation matrices; $\mathcal { L } _ { \mathrm { T r a n s l a t i o n } }$ measures the Euclidean distance between predicted and ground-truth translations; and $\mathcal { L } _ { \mathrm { I m a g e } }$ corresponds to the $L ^ { 1 }$ norm between the reoriented and canonical images. The geodesic distance was defined as the angle arccos $\left( \frac { \mathrm { t r } ( \mathbf { R } ^ { \top } \hat { \mathbf { R } } ) - 1 } { 2 } \right)$ between the predicted and ground-truth rotation matrices �<sup>̂</sup> and � (Hartley et al., 2013). The proposed multi-objective loss enforces consistency at both the parameter and image levels. Furthermore, $\mathcal { L } _ { \mathrm { A n c h o r } }$ provides spatial supervision of the anchor configuration, while explicit geodesic and translation losses regularize the induced rigid transformation in SE(3), stabilizing training and preventing degenerate anchor configurations. Optimization was carried out via the Adam algorithm (Kingma and Ba, 2017) and the hyperparameters were set empirically to: $\lambda _ { 1 } = 0 . 0 1$ $\lambda _ { 2 } = 0 . 1$ $\lambda _ { 3 } = 0 . 0 1$ , and $\lambda _ { 4 } = 0 . 1$

Ensembling. Separate networks, initialized via transfer learning from $D _ { S } ,$ were trained on distinct $D _ { T }$ data splits, and an ensemble strategy was adopted at inference. For each model, the predicted anchor points were converted into the corresponding translation vector and rotation matrix. The translation vectors were aggregated by simple arithmetic averaging, while the rotation matrices were combined using chordal rotation averaging (Hartley et al., 2013), defined as:

$$
\bar { \bf R } _ { \mathrm { V i T } } = \underset { { \bf R } \in \mathrm { S O } ( 3 ) } { \arg \operatorname* { m i n } } \sum _ { i } \| { \bf R } - { \bf R } _ { \mathrm { V i T } _ { i } } \| _ { F } ^ { 2 }\tag{5}
$$

where $\| . \| _ { F }$ is the Frobenius norm.

This ensemble aggregation reduced variance in reorientation while preserving geometric consistency in SO(3). The optimisation problem in Eq. 5 is equivalent to finding the rotation matrix nearest to the arithmetic mean of the individual rotation matrices, and admits a closed-form solution analogous to the orthogonal Procrustes problem via singular value decomposition (SVD) (Hartley et al., 2013). Chordal rotation averaging assumes that the predictions ${ \mathbf { R } } _ { \mathrm { V i T } _ { i } }$ are relatively concentrated, which holds when the input is unambiguous. In contrast, predictions tend to be more dispersed for scans with unusual fetal anatomy or intensity distribution. Although various rotation-averaging strategies exist in the literature (Hartley et al., 2013), we adopted chordal averaging as our ensemble aggregation strategy, as empirically validated in Section 4.4.

## 3. Experiments

## 3.1. Fetal cardiac MRI datasets

## 3.1.1. Doppler-gated dataset – Target domain $D _ { T }$

Cine stacks of 2D+time slices were acquired from 199 subjects (gestational age: $3 2 . 5 \pm 1 . 5 ~ [ 2 7 . 4 , 3 7 . 0 ]$ weeks), comprising 18 healthy volunteers and 181 fetuses with congenital heart disease and related conditions, on a 1.5T MAGNETOM Sola (Siemens Healthineers, Germany) using Doppler ultrasound gated (Kording et al., 2018a,b) balanced steady-state free precession (bSSFP) sequence (TE 1.93 ms, TR 56.3 ms, flip angle 60°, and acceleration factor 3, spatial resolution $2 \times 2 \times 6 ~ \mathrm { m m } ^ { 3 }$ , � = 20 temporal frames after resampling), see Figure 1. Four to 16 stacks were acquired in multiple orientations. Each 2D+time slice spanned a single cardiac cycle, with a per-slice acquisition time of approximately 4 seconds.

<table><tr><td rowspan=2 colspan=1>Domain</td><td rowspan=2 colspan=1>Evaluation</td><td rowspan=1 colspan=3>Subjects n</td></tr><tr><td rowspan=1 colspan=1>5-fold cross-validation set</td><td rowspan=1 colspan=1>Held-outtest set</td><td rowspan=1 colspan=1>Uniquesubjects</td></tr><tr><td rowspan=3 colspan=1>Source</td><td rowspan=1 colspan=1>Thoracic segmentation</td><td rowspan=1 colspan=1>116 [483]</td><td rowspan=1 colspan=1>23 (92)</td><td rowspan=3 colspan=1>141</td></tr><tr><td rowspan=1 colspan=1>Cardiac segmentation</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>22</td></tr><tr><td rowspan=1 colspan=1>Reorientation</td><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=1>21</td></tr><tr><td rowspan=6 colspan=1>Target</td><td rowspan=1 colspan=1>Thoracic segmentation</td><td rowspan=1 colspan=1>25 [222]</td><td rowspan=1 colspan=1>5 (36)</td><td rowspan=4 colspan=1>60</td></tr><tr><td rowspan=1 colspan=1>Reconstruction</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>Cardiac segmentation</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>Reorientation</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>End-to-end pipeline</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>121</td><td rowspan=1 colspan=1>121</td></tr><tr><td rowspan=1 colspan=1>Excluded</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>18</td></tr></table>

Dataset splits for each evaluated task in the source (freerunning; � = 141 subjects) and target (Doppler-gated; � = 199 subjects) domains. Numbers of stacks are shown in brackets.

Manual thoracic and cardiac segmentation masks, along with canonical reorientation, were produced using ITK-SNAP (Yushkevich et al., 2006).The first rater annotated both the training and test sets. A second rater independently annotated the test data in order to assess inter-rater agreement. The SPARC pipeline was additionally evaluated on a large held-out subset not used at any stage of network training or evaluation, for which no ground truth annotations were available. Dataset splits are described in Table 1.

## 3.1.2. Free-running dataset – Source domain $\mathcal { D } _ { S }$

Free-running cine stacks of 2D+time slices (4-8 stacks in multiple orientations) were acquired from 141 subjects (gestational age: 32.3 ± 1.7 [27.9, 39.3] weeks), comprising 6 healthy volunteers and 135 fetuses with congenital heart disease and related conditions, on a 1.5T Ingenia (Philips Healthcare, Netherlands) using a 2D bSSFP sequence (TE 1.9 ms, TR 3.8 ms, flip angle 60°, spatial resolution $2 \times$ $2 \times 6 ~ \mathrm { m m } ^ { 3 }$ , slice overlap 2-3 mm, under-sampling factor of 8, temporal resolution 72 ms, 96 frames per slice) and reconstructed using a �-� SENSE which included motion correction, cardiac synchronization using image-base gating and volumetric reconstruction (van Amerom et al., 2018, 2019; Roberts et al., 2020). Acquisition of a typical stack took 155s. Manual cardiac and thoracic segmentation masks, were available for all subjects. All cine volumes were manually reoriented to a canonical coordinate system using ITK-SNAP (Yushkevich et al., 2006).

## 3.2. Experimental setup

## 3.2.1. Deep learning models

For each deep learning task (Sec.2.3, 2.5.1 and 2.5.2), five networks were trained and validated on distinct subjectlevel splits of the cross-validation set (Table 1), used for both hyperparameter tuning and model selection. The resulting five trained networks were subsequently evaluated on a heldout test set using an ensemble strategy: majority voting for segmentation tasks and chordal rotation averaging for reorientation (sec. 2.5.2). Cross-validation and test subjects were selected to reflect the overall distribution of the cohort in terms of gestational age and cardiac condition (volunteer vs. CHD).

We investigated four domain adaptation strategies: (1) Source-only networks trained exclusively on $D _ { S } ; ( 2 )$ Targetonly networks trained exclusively on $D _ { T } ; ( 3 )$ Joint networks trained on the combined domain $\mathit { D } _ { S } \cup \mathit { D } _ { T } ;$ (4) Transfer networks pre-trained on $\mathit { D _ { S } }$ and subsequently fine-tuned on $D _ { T }$ . All approaches were evaluated on the target test set. Approaches (1) and (3) were evaluated on the source test set, providing a benchmark in-domain performance.

To assessed the benefit of ensemble aggregation ensemble performance was compared against the mean performance of the five individual networks. For reorientation specifically, three ensemble aggregation strategies were compared: chordal rotation averaging, quaternion averaging, and geodesic averaging (Hartley et al., 2013).

## 3.2.2. Reconstruction algorithm

The proposed reconstruction algorithm (Sec.2.4) was compared against the method of (van Amerom et al., 2018), which assumes distinct motion states for each temporal frame within a slice and incorporates a sinc temporal point spread function (tPSF), a design choice reflecting its original development for free-running acquisitions. In contrast to the proposed approach, the method of (van Amerom et al., 2018) does not share parameters across temporal frames, resulting in a less parsimonious model with a substantially larger parameter space. The same hyperparameters were used to define the iterative reconstruction process of both methods.

The comparison was performed on $n = 1 0$ target domain cases (Table 1) using a leave-stack-out design, with a cine volume reconstructed from all but one randomly selected withheld stack. The withheld stack was subsequently simulated from the reconstructed volume and compared against the original acquired stack, assessing how well the reconstruction generalises to unseen data. Additionally, data consistency was calculated by comparing the included acquired stacks against their simulated counterparts. In both cases, comparison between acquired and simulated stacks incorporated slice registration and intensity correction to ensure a fair and consistent evaluation. To limit computational cost, a single leave-stack-out experiment was performed per subject rather than evaluating all possible stack combinations. The same stack was left-out in both methods.

## 3.2.3. End-to-endpipeline

The complete SPARC pipeline was evaluated on a large target domain held-out subset $( n ~ = ~ 1 3 9 )$ not used in any prior training or evaluation step. Eighteen cases were excluded on the basis of insuficient input data, falling into two categories: fewer than five stacks with adequate cardiac coverage and acceptable motion levels within stacks $( n ~ =$ 14), and loss of gating signal and consequently synchronicity between stacks $( n = 4 )$ . These conditions prevent reliable slice-to-volume reconstruction, as suficient spatial coverage and consistent cardiac phase synchronisation are required to recover a coherent cine volume. In $n = 6$ cases, one or more input stacks were manually excluded by the experienced observer due to excessive within-stack motion corruption. The total $n \ = \ 1 2 1$ cases were processed through the full pipeline and form the basis of the end-to-end evaluation.

To assess the contribution of the cardiac centre localisation step, an additional ablation was performed by disabling the heart segmentation network prior to Transformer-based reorientation, and running the SPARC pipeline on the same $n = 1 2 1$ cases. This allowed direct comparison of reorientation performance and failure rates with and without explicit cardiac centring, as reported in Section 4.5.

## 3.3. Quantitative evaluation

## 3.3.1. Assessment of segmentation networks

Segmentation performance was assessed by comparing predicted delineations against manually annotated groundtruth masks, using the Dice similarity coeficient (DSC), the $9 5 ^ { \mathrm { t h } }$ percentile symmetric Hausdorf distance $( \mathrm { H D _ { 9 5 \% } } )$ , and the average symmetric surface distance (ASSD) (Cardoso et al., 2022) averaged over the test set. For cardiac segmentation, the centre distance (CD) between the predicted and ground-truth centres of mass was additionally reported.

For thoracic segmentation, performance was compared across methods within each domain using a linear mixedefects model with method as a fixed efect, subject as a random efect, and Bonferroni correction applied for multiple comparisons (36 stacks nested within 5 subjects), with the proposed Transfer approach (ensemble) compared to all other methods within the target domain. Inter-rater agreement was assessed by comparing two independent sets of manual annotations on the target domain test stacks. For cardiac segmentation, given the limited size of the target domain test set $\mathbf { \xi } ( n ~ \mathbf { \xi } = ~ 5$ subjects), statistical comparison between methods was not performed.

## 3.3.2. Assessment of reconstruction algorithm

Reconstruction performance was evaluated by computing similarity metrics between the acquired and simulated stacks, measured within manually delineated cardiac masks. Two complementary metrics were computed: normalised cross-correlation (NCC), which measures intensity agreement, and normalised root mean squared error (NRMSE), which penalises large intensity discrepancies (Kuklisova-Murgasova et al., 2012). Both metrics were computed separately for the leave-stack-out and data consistency evaluations described in Section 3.2.2. Reconstruction time and the proportions of included and excluded slices were recorded to characterise the computational eficiency and robust weighting behaviour of each method. Statistical comparison between methods was performed using paired �-tests.

## 3.3.3. Assessment of reorientation network

Reorientation performance was evaluated by by applying random translations and rotations to manually reoriented fetal heart volumes (see Section 2.5.2), enabling evaluation across the full range of possible input orientations. For each test case, 250 random rigid transformations were sampled and held fixed across all compared methods. The error between the ground-truth and predicted rigid transformations was measured using the geodesic distance (GD) for rotations and the cardiac centre distance (CD) for translations (Hou et al., 2018a).

The proposed Transfer approach with chordal ensemble was compared to all other methods within the target domain using a linear mixed-efects model with method as a fixed efect, subject as a random efect, and Bonferroni correction applied for multiple comparisons (1750 samples corresponding to 250 simulated rigid transformations per subject). Inter-rater agreement was evaluated by computing the GD between two manual transformations to a canonical orientation on the target domain test volumes.

## 3.4. Qualitative evaluation of end-to-end pipeline

We visually evaluated each stage of the pipeline using a binary success/failure decision by two independent raters. In cases of disagreement, a decision was reached by consensus. A failure was recorded as follows: (1) Template selection: an axial stack exhibited less severe motion than the automatically selected template; (2) Thoracic segmentation: the predicted mask resulted in partial or complete exclusion of the fetal heart; (3) Slice-to-volume reconstruction: the reconstructed cine volume did not reflect of the anatomy visible in the template stack; (4) Anatomical reorientation: the reoriented volume deviated substantially from the canonical fetal coordinate system.

The automated pipeline was considered successful if no failure was identified. Each failure prompted a manual intervention at the corresponding stage, and the pipeline resumed from that point.

## 3.5. Implementation details

## 3.5.1. Deep learning models

All deep learning architectures were implemented in PyTorch using the MONAI framework (Paszke et al., 2019; Cardoso et al., 2022) and trained on an NVIDIA RTX A5000 GPU with 24 GB of memory. For thoracic and cardiac segmentation, a standard 3D U-Net architecture (Ronneberger et al., 2015; Milletari et al., 2016) was employed, comprising $3 \times 3 \times 3$ convolutional layers, batch normalisation (Iofe and Szegedy, 2015) and ReLU activations, in both encoder and decoder paths. The encoder path comprised strided convolutions to downsample the data. The decoder path used strided transpose convolutions to upscale the output to its original spatial dimensions. The number of features doubled at each encoding stage, resulting in a bottleneck consisting of 256 channels. For reorientation we employed a standard ViT architecture comprising 12 transformer blocks, each with 12 self-attention heads. The input volume was partitioned into $1 6 \times 1 6 \times 1 6$ patches and augmented with fixed sinusoidal positional embeddings (Dosovitskiy et al., 2020).

Hyperparameters (e.g. loss weighting coeficients, batch size, number of training epochs, and learning rate) were selected based on performance on the validation set. With the exception of the number of training epochs, all hyperparameters were kept identical across the four training strategies (source-only, target-only, joint, and transfer learning). The batch size and learning rate were set to 64 and $1 0 ^ { - 3 }$ for the segmentation networks, and to 1024 and $1 0 ^ { - 4 }$ for the reorientation network. For the reorientation network, the efective batch size exceeded the number of available training images. To compensate, each image was sampled multiple times per batch using independently drawn random rigid transformations, ensuring dense coverage of SE(3) at each gradient update. This sampling strategy was found to be critical for stabilising optimisation, particularly given the limited size of the target domain training set. The number of training epochs was set to 200 for thoracic segmentation, 400 for cardiac segmentation, and 2500 for reorientation under the source-only andjoint training configurations; these values were doubled for the target-only and transfer learning configurations to account for the smaller target domain training set. The model checkpoint corresponding to the lowest validation loss was retained at the end of training.

Prior to all training and testing experiments, images were corrected for bias-field inhomogeneity (Tustison et al., 2010) and resampled to isotropic $2 \times 2 \times 2 \mathrm { m m } ^ { 3 }$ resolution. Segmentation inference was performed using a sliding-window strategy (Cardoso et al., 2022). Code for the segmentation and anatomical reorientation components is publicly available at https://github.com/aboutill/SPARC.

## 3.5.2. Slice-to-volume reconstruction

The proposed reconstruction algorithm was implemented in C++ using a multi-threaded design, with image representation and registration handled via the MIRTK $\mathrm { l i b r a r y } ^ { 2 }$ The method of van Amerom et al. (2018) was obtained from SVRTK repository<sup>3</sup>. Experiments were performed on a system equipped with 62 GB of RAM, a 2.40 GHz CPU, and 24 threads. For both methods, three interleaved registrationreconstruction cycles were performed, with the number of super-resolution reconstruction iterations set to 7, 7, and 21 in successive cycles respectively. Cine volumes were reconstructed at an isotropic spatial resolution of $1 \times 1 \times 1 \mathrm { m m } ^ { 3 }$ The code is publicly available at https://github.com/baby-MedIA/svr-lite.

## 3.5.3. Pipeline

End-to-end pipeline experiments were conducted in semi-automatic mode on a system equipped with 96 GB of RAM, a 2.40 GHz CPU, and 24 threads. Timing was also recorded for the full end-to-end pipeline as well as for the reconstruction step. The SPARC pipeline is available in as a Docker container at https://hub.docker.com/r/aboutill/sparc.

## 4. Results

## 4.1. Thoracic segmentation

Thoracic segmentation results are summarised in Table 2. The transfer learning approach achieved the best performance on the target domain across all three metrics $( \mathrm { D S C } 8 4 . 7 \pm 3 . 9 \% , \mathrm { H D } _ { 9 5 \% } 5 . 4 \pm 2 . 4 \mathrm { m m } ,$ , and ASSD 1.1 ± 0.4 mm), with statistically significant improvements over all other approaches $( p < 0 . 0 5 )$ ). Notably, this performance is comparable the source-only model evaluated on the source domain $( \mathrm { D S C } \ 8 4 . 9 \pm 7 . 0 \% , \mathrm { H D } _ { 9 5 \% } \ 4 . 1 \pm 3 . 0 \mathrm { m m }$ , ASSD $1 . 1 \pm 0 . 7 \mathrm { m m } )$ , and exceeds inter-rater agreement metrics (DSC 81.4±7.7%, HD 6.6±4.0 mm, ASSD 1.5±0.9 mm).

SPARC: Automatic 3D+time fetal cardiac MRI reconstruction
<table><tr><td rowspan=1 colspan=2>Domain</td><td rowspan=2 colspan=1>Ensemble</td><td rowspan=2 colspan=1>DSC [%] ↑</td><td rowspan=2 colspan=1> $\mathrm { H D } _ { 9 5 \% } \ [ \mathsf { m m } ] \ \downarrow$ </td><td rowspan=2 colspan=1>ASSD [mm] ↓</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>Training</td></tr><tr><td rowspan=2 colspan=1>Source</td><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1> $\overline { { \checkmark } }$ </td><td rowspan=1 colspan=1> $\overline { { 8 4 . 9 \pm 7 . 0 } }$ </td><td rowspan=1 colspan=1> $\overline { { 4 . 1 \pm 3 . 0 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 1 \pm 0 . 7 } }$ </td></tr><tr><td rowspan=1 colspan=1>Joint</td><td rowspan=1 colspan=1> $\overline { { \checkmark } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 8 5 . 7 \pm 6 . 0 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 4 . 1 \pm 3 . 2 } }$ </td><td rowspan=1 colspan=1> $1 . 1 \pm 0 . 7$ </td></tr><tr><td rowspan=6 colspan=1>Target</td><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1> $\overline { { \checkmark } }$ </td><td rowspan=1 colspan=1> $\overline { { 7 9 . 2 \pm 7 . 3 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 . 8 \pm 1 0 . 6 } }$ </td><td rowspan=1 colspan=1> $2 . 1 \pm 2 . 1$ </td></tr><tr><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1> $\overline { { \checkmark } }$ </td><td rowspan=1 colspan=1> $8 1 . 8 \pm 4 . 5$ </td><td rowspan=1 colspan=1> $\overline { { 6 . 5 \pm 2 . 8 } }$ </td><td rowspan=1 colspan=1> $1 . 4 \pm 0 . 4$ </td></tr><tr><td rowspan=1 colspan=1>Joint</td><td rowspan=1 colspan=1> $\overline { { \checkmark } }$ </td><td rowspan=1 colspan=1> $\overline { { 8 4 . 3 \pm 3 . 9 } }$ </td><td rowspan=1 colspan=1> ${ \overline { { 5 . 9 \pm 2 . 8 } } }$ </td><td rowspan=1 colspan=1> $1 . 2 \pm 0 . 4$ </td></tr><tr><td rowspan=2 colspan=1>Transfer</td><td rowspan=1 colspan=1> $\overline { x }$ </td><td rowspan=1 colspan=1> $\overline { { 8 3 . 2 \pm 4 . 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { 6 . 9 \pm 4 . 4 } }$ </td><td rowspan=1 colspan=1> $1 . 4 \pm 0 . 8$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { \checkmark } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 8 4 . 7 \pm 3 . 9 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 5 . 4 \pm 2 . 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 1 . 1 \pm 0 . 4 } } }$ </td></tr><tr><td rowspan=1 colspan=2>Inter-rater</td><td rowspan=1 colspan=1> $\overline { { 8 1 . 4 \pm 7 . 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { 6 . 6 \pm 4 . 0 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 5 \pm 0 . 9 } }$ </td></tr></table>

Quantitative evaluation of thoracic segmentation models on held-out test sets from the source (free-running; $m = 9 2$ stacks across $n = 2 3$ subjects) and target (Doppler-gated; $m = 3 6$ stacks across � = 5 subjects) domains shown as mean±standard deviation. The best performance within each domain is shown in bold (linear mixed-efect model with Bonferroni correction, $p < 0 . 0 5 )$ .

Table 2
<table><tr><td rowspan=1 colspan=2>Method</td><td rowspan=1 colspan=1>van Amerom</td><td rowspan=1 colspan=1>Proposed</td></tr><tr><td rowspan=2 colspan=1>Modelassumptions</td><td rowspan=1 colspan=1>Motion states</td><td rowspan=1 colspan=1>Frame-wise</td><td rowspan=1 colspan=1>Slice-wise</td></tr><tr><td rowspan=1 colspan=1>tPSF</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>x</td></tr><tr><td rowspan=2 colspan=1>Stepcomplexity</td><td rowspan=1 colspan=1>Slice/frame-to-volumeregistration</td><td rowspan=1 colspan=1>O(KT)</td><td rowspan=1 colspan=1>O(K)</td></tr><tr><td rowspan=1 colspan=1>Super-resolutionreconstruction</td><td rowspan=1 colspan=1>O(NT2)</td><td rowspan=1 colspan=1>O(NT)</td></tr><tr><td rowspan=2 colspan=1>Leave-stack-out</td><td rowspan=1 colspan=1>NCC ↑</td><td rowspan=1 colspan=1> $\overline { { 0 . 8 9 \pm 0 . 0 4 } }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 1 \pm 0 . 0 3 }$ </td></tr><tr><td rowspan=1 colspan=1>NRMSE↓</td><td rowspan=1 colspan=1> $\overline { { 0 . 1 3 \pm 0 . 0 3 } }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 1 1 \pm 0 . 0 2 }$ </td></tr><tr><td rowspan=2 colspan=1>Dataconsistency</td><td rowspan=1 colspan=1>NCC ↑</td><td rowspan=1 colspan=1> $\overline { { { \bf 0 . 9 7 \pm 0 . 0 1 } } }$ </td><td rowspan=1 colspan=1> $0 . 9 6 \pm 0 . 0 1$ </td></tr><tr><td rowspan=1 colspan=1>NRMSE↓</td><td rowspan=1 colspan=1> $\mathbf { \overline { { 0 . 0 9 \pm 0 . 0 1 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 1 0 \pm 0 . 0 2 } }$ </td></tr><tr><td rowspan=2 colspan=1>Slice/frameratios</td><td rowspan=1 colspan=1>Included [%] ↑</td><td rowspan=1 colspan=1> $\overline { { 5 5 . 0 \pm 5 . 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { 7 8 . 2 \pm 5 . 9 } }$ </td></tr><tr><td rowspan=1 colspan=1>Excluded [%] ↓</td><td rowspan=1 colspan=1> $\overline { { 4 5 . 0 \pm 5 . 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 1 . 8 \pm 5 . 9 } }$ </td></tr><tr><td rowspan=1 colspan=2>Time [min] ↓</td><td rowspan=1 colspan=1> $\overline { { 4 9 . 0 \pm 1 4 . 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 4 . 8 \pm 1 . 0 } }$ </td></tr></table>

Table 3  
Quantitative evaluation of the reconstruction algorithms on the target domain test set (Doppler-gated; $n = 1 0$ subjects) shown as mean±standard deviation. The best performance within each domain is shown in bold (paired �-test, � < 0.05)

The substantially degraded performance of the sourceonly model on the target domain $( \mathrm { H D } _ { 9 5 \% } \ 1 0 . 8 \pm 1 0 . 6 \mathrm { m m } ,$ DSC 79.2±7.3%) confirms a need for our domain adaptation strategy. Figure S1 provides a qualitative comparison of the four training strategies on three representative target domain subjects, illustrating the superior boundary delineation of the Transfer approach.

## 4.2. Reconstruction

Reconstruction results are summarised in Table 3. The proposed method achieved significantly higher leave-stackout $\operatorname { N C C } \left( 0 . 9 1 \pm 0 . 0 3 \ \mathrm { v s } \ 0 . 8 9 \pm 0 . 0 4 \right)$ and lower NRMSE $( 0 . 1 1 \pm 0 . 0 2 \mathrm { \ v s \ } 0 . 1 3 \pm 0 . 0 3 )$ , indicating superior generalisation to unseen data and more faithful recovery of the underlying anatomy. This shows that assuming a shared motion state per slice rather than per frame reduces the risk of overfitting to the input data. Marginally higher consistency combined with smaller fraction of included input data in the method of van Amerom et al. (2019) points to overfitting rather then genuinely better performance compared to our method. In addition, the proposed method achieved a tenfold reduction in reconstruction time $( 4 . 8 { \pm } 1 . 0 \mathrm { v s } 4 9 . 0 { \pm } 1 4 . 1 $ min, $p < 0 . 0 5 )$ , directly attributable to the reduced computational complexity: from (��) to (�) for slice-to-volume registration, and from $\mathcal { O } ( N T ^ { 2 } )$ to (��) for super-resolution reconstruction (with $T = 2 0 )$ . Figure 5 demonstrates that our method produces notably smoother temporal transitions in the cardiac phase profile than van Amerom et al. (2019), with comparable spatial sharpness.

## 4.3. Cardiac segmentation

Cardiac segmentation results are summarised in Table 4. The transfer learning approach with ensemble achieved the best performance on the target domain across all metrics $( \mathrm { D S C } 8 2 . 2 \pm 2 . 1 \% , \mathrm { H D } _ { 9 5 \% } 5 . 1 \pm 0 . 6 \mathrm { m m } , \mathrm { A S S D } 2 . 1 \pm 0 . 4 $ mm, CD $6 . 3 \pm 2 . 9 \mathrm { { m m } ) }$ , reaching performance comparable to that of the source-only model on the source domain (DSC 83.4 ± 4.9%, HD<sub>95%</sub> 4.8 ± 1.9 mm, ASSD 1.7 ± 0.4 mm, CD $4 . 2 { \pm } 2 . 3 \mathrm { m m }$ , while significantly outperforming target-only (DSC $7 3 . 8 \pm 9 . 0 \% )$ and source-only (DSC 20.0 ± 30.8%) model. These findings are consistent with the results for thoracic segmentation.

The relatively low centre distance achieved by the proposed approach demonstrates that cardiac segmentation provides a robust spatial initialisation for the subsequent reorientation network, as further confirmed by the ablation study (Section 4.5).

## 4.4. Reorientation

Reorientation results are summarised in Table 5. The transfer-learning approach with chordal averaging achieved statistically superior performance on the target domain across all metrics (GD $1 4 . 3 \pm 1 0 . 9 ^ { \circ }$ , CD $2 . 4 \pm 1 . 9$ mm, $p ~ < ~ 0 . 0 5 )$ , and reached performance comparable to that of the source-only model evaluated on the source domain (GD $1 3 . 1 \pm 6 . 3 ^ { \circ } , \mathrm { C D } 1 . 6 \pm 1 . 5 \mathrm { m m } )$ . The source-only model

![](images/cd719873bab45cb204f10637c351407978de6211fe5058b263e692a50202541e.jpg)  
Figure 5: Qualitative comparison of cine volumes reconstructed by van Amerom et al. (2019) and our proposed methods, for two representative target domain cases from the leave-stack-out experiment. Three orthogonal views are shown, alongside a temporal intensity profile computed along a fixed line ( ) across all � = 20 cardiac phases.

<table><tr><td rowspan=1 colspan=2>Domain</td><td rowspan=2 colspan=1>Ensemble</td><td rowspan=2 colspan=1>DSC [%] ↑</td><td rowspan=2 colspan=1> $\mathrm { H D } _ { 9 5 \% } \ [ \mathsf { m m } ] \ \downarrow$ </td><td rowspan=2 colspan=1>ASSD [mm] ↓</td><td rowspan=2 colspan=1>CD [mm] ↓</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>Training</td></tr><tr><td rowspan=2 colspan=1>Source</td><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1> $\overline { { 8 3 . 4 \pm 4 . 9 } }$ </td><td rowspan=1 colspan=1> $\overline { { 4 . 8 \pm 1 . 9 } }$ </td><td rowspan=1 colspan=1> $1 . 7 \pm 0 . 4$ </td><td rowspan=1 colspan=1> $\overline { { 4 . 2 \pm 2 . 3 } }$ </td></tr><tr><td rowspan=1 colspan=1>Joint</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1> $\overline { { 8 3 . 8 \pm 5 . 0 } }$ </td><td rowspan=1 colspan=1> $\overline { { 4 . 7 \pm 1 . 5 } }$ </td><td rowspan=1 colspan=1> $1 . 6 \pm 0 . 4$ </td><td rowspan=1 colspan=1> $\overline { { 4 . 0 \pm 2 . 3 } }$ </td></tr><tr><td rowspan=5 colspan=1>Target</td><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1> $2 0 . 0 \pm 3 0 . 8$ </td><td rowspan=1 colspan=1> $2 7 . 8 \pm 1 2 . 0$ </td><td rowspan=1 colspan=1> $\overline { { 1 4 . 4 \pm 7 . 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { 3 9 . 5 \pm 2 2 . 3 } }$ </td></tr><tr><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1> $7 3 . 8 \pm 9 . 0$ </td><td rowspan=1 colspan=1> $\overline { { 8 . 2 \pm 3 . 6 } }$ </td><td rowspan=1 colspan=1> $3 . 2 \pm 1 . 4$ </td><td rowspan=1 colspan=1> $\overline { { 1 2 . 3 \pm 5 . 2 } }$ </td></tr><tr><td rowspan=1 colspan=1>Joint</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1> $\overline { { 7 1 . 0 \pm 7 . 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 9 . 2 \pm 1 . 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { 3 . 5 \pm 0 . 9 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 1 . 1 \pm 7 . 6 } }$ </td></tr><tr><td rowspan=2 colspan=1>Transfer</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1> $\overline { { 7 9 . 6 \pm 2 . 5 } }$ </td><td rowspan=1 colspan=1> $\overline { { 6 . 2 \pm 1 . 3 } }$ </td><td rowspan=1 colspan=1> $2 . 4 \pm 0 . 5$ </td><td rowspan=1 colspan=1> $\overline { { 8 . 4 \pm 2 . 1 } }$ </td></tr><tr><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1> $\overline { { 8 2 . 2 \pm 2 . 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 5 . 1 \pm 0 . 6 } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 . 1 \pm 0 . 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { 6 . 3 \pm 2 . 9 } }$ </td></tr></table>

Table 4  
Quantitative evaluation of cardiac segmentation models on held-out test sets from the source (free-running; � = 22 subjects) and target (Doppler-gated; � = 5 subjects) domains shown as mean±standard deviation.

<table><tr><td rowspan=1 colspan=2>Domain</td><td rowspan=2 colspan=1>Ensemble</td><td rowspan=2 colspan=1>GD [°] ↓</td><td rowspan=2 colspan=1>CD [mm] ↓</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>Training</td></tr><tr><td rowspan=2 colspan=1>Source</td><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>Chordal</td><td rowspan=1 colspan=1> $1 3 . 1 \pm 6 . 3$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 6 \pm 1 . 5 } }$ </td></tr><tr><td rowspan=1 colspan=1>Joint</td><td rowspan=1 colspan=1>Chordal</td><td rowspan=1 colspan=1> $\overline { { { \bf 1 0 . 5 \pm 5 . 8 } } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 1 . 1 \pm 1 . 0 } } }$ </td></tr><tr><td rowspan=8 colspan=1>Target</td><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>Chordal</td><td rowspan=1 colspan=1> $\overline { { 7 4 . 9 \pm 5 2 . 8 } }$ </td><td rowspan=1 colspan=1> $\overline { { 5 . 2 \pm 4 . 5 } }$ </td></tr><tr><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>Chordal</td><td rowspan=1 colspan=1> $3 3 . 4 \pm 3 6 . 4$ </td><td rowspan=1 colspan=1> $\overline { { 4 . 4 \pm 4 . 5 } }$ </td></tr><tr><td rowspan=1 colspan=1>Joint</td><td rowspan=1 colspan=1>Chordal</td><td rowspan=1 colspan=1> $\overline { { 1 6 . 0 \pm 1 1 . 8 } }$ </td><td rowspan=1 colspan=1> $2 . 9 \pm 2 . 5$ </td></tr><tr><td rowspan=4 colspan=1>Transfer</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1> $\overline { { 1 8 . 9 \pm 1 3 . 5 } }$ </td><td rowspan=1 colspan=1> $\overline { { 6 . 9 \pm 3 . 9 } }$ </td></tr><tr><td rowspan=1 colspan=1>Quaternion</td><td rowspan=1 colspan=1> $\overline { { 1 6 . 7 \pm 1 5 . 3 } }$ </td><td rowspan=1 colspan=1> $2 . 4 \pm 1 . 9$ </td></tr><tr><td rowspan=1 colspan=1>Geodesic</td><td rowspan=1 colspan=1> $\overline { { 1 6 . 3 \pm 1 3 . 5 } }$ </td><td rowspan=1 colspan=1> $2 . 4 \pm 1 . 9$ </td></tr><tr><td rowspan=1 colspan=1>Chordal</td><td rowspan=1 colspan=1> $\overline { { { \bf 1 4 . 3 \pm 1 0 . 9 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 . 4 \pm 1 . 9 } }$ </td></tr><tr><td rowspan=1 colspan=2>Inter-rater†</td><td rowspan=1 colspan=1> $\overline { { 7 . 7 \pm 2 . 9 } }$ </td><td rowspan=1 colspan=1>一</td></tr></table>

Table 5

Quantitative evaluation of reorientation models on held-out test sets from the source (free-running; � = 21 subjects) and target (Doppler-gated; � = 7 subjects) domains shown as mean±standard deviation. Each image was sampled 250 times using independently drawn random rigid transformations. The best performance within each domain is shown in bold (linear mixed-efect model comparison with Bonferroni correction, $p ~ < ~ 0 . 0 5 )$ <sup>†</sup>Target domain GD inter-rater agreement was computed on a single reorientation per subject (� = 7) and is not directly comparable to network performance assessed over whole range of rotations.

fails on the target domain (GD $7 4 . 9 \pm 5 2 . 8 ^ { \circ } )$ , while the comparatively poor performance of the target-only model (GD $3 3 . 4 \pm 3 6 . 4 ^ { \circ } )$ reflects the limited size of the target domain training set. All three ensemble strategies (quaternion, geodesic, and chordal averaging (Hartley et al., 2013)) provided consistent improvements over the mean performance of individual networks. The best performing chordal averaging was selected and the ensemble method of choice.

Figure 6 illustrates the superior visual agreement of the proposed approach with the ground truth reorientation across all cases and orientations. Furthermore, Figure S2 reveals that GD remains low and approximately constant across the full range of simulated orientations, with the majority of error concentrated in a small number of challenging subjects.

## 4.5. End-to-end pipeline

The results of the end-to-end pipeline evaluation are summarised in Table 6. Prior to end-to-end pipeline evaluation, input stacks were manually excluded in $\ n \ = \ 6$ cases due to excessive within-stack motion corruption. Both automatic template selection and thoracic segmentation was successful in 118 out of 121 cases (97.5%). Slice-to-volume reconstruction was successful in all 121 cases. Mean reconstruction time was $4 . 9 \pm 1 . 0$ min per case. The cardiac centre localisation step was critical in reducing the reorientation failure rate from 25.6% (31∕121) to 9.9% (12∕121).

The complete SPARC pipeline ran fully automatically in 100 out of 121 cases (82.6%), with a mean end-to-end processing time of 7.1±1.3 min per case. With manual intervention, all subjects whose data quality passed the inclusion criteria were successfully reconstructed.

![](images/5d1202450e9cd35151810a487948abba52709b91c1aa6b462a607e143b6e814f.jpg)  
Figure 6: Qualitative comparison of reorientation for three target domain subjects across the four training strategies (source-only, target-only, joint, and transfer). Cases were selected to represent varying levels of image contrast with distinct simulated random transformation. Time-averaged cine volumes are shown in canonical fetal coordinate system across three orthogonal views. Images resolution is set to the input resolution of the ViT reorientation network, 2 × 2 × 2 mm.

## 5. Discussion

This study introduces an end-to-end, deep learningenabled pipeline for the acquisition and reconstruction of 3D+time fetal cardiac MRI. Leveraging a recently introduced, CE-marked Doppler ultrasound gating device (Kording et al., 2018a,b), the proposed framework provides a solution deployable in routine clinical practice and large-scale research studies. By reducing fully automated reconstruction times to minutes while ofering eficient manual intervention options, the pipeline ensures robust processing for cases meeting minimal inclusion criteria (at least four stacks without major motion corruption and successful Doppler gating).

## 5.1. Clinical translation

The SPARC pipeline is currently deployed at our institution as part of fetal cardiac MRI screening. The Doppler ultrasound gating device used in this study has already been adopted at multiple fetal cardiac MRI centres (Tavares

De Sousa et al., 2019; Vollbrecht et al., 2024a), providing a natural pathway for deployment beyond our institution. The reconstruction algorithm requires no site-specific adaptation and can be used out-of-the-box. While the deep learning components are susceptible to domain shifts, our proposed transfer learning approach allows eficient retraining with a small amount of locally acquired and annotated data, using our publicly available model weights and training code.

Additionally, the competitive results of joint training strategy provide evidence that pooling heterogeneous multisite data is a viable generalisation strategy (Wang et al., 2023). The public availability of the model weights and training code also supports federated learning (Guan et al., 2024), as institutions can initialise from the provided weights and contribute site-specific fine-tuning without exposing patient data. In parallel, the emergence of foundation models pre-trained on large and diverse medical imaging datasets may reduce the annotation burden at new sites (Chen et al., 2026), either as initialisations for fine-tuning or as zero-shot segmentation tools.

Automated quality control could further enhance deployability to clinical practice by automatically highlighting reconstructions that require further inspection. Our current

<table><tr><td rowspan=1 colspan=4>SPARC pipeline evaluation on clinical cohort (n = 121)</td></tr><tr><td rowspan=1 colspan=4>1) Manual intervention summary</td></tr><tr><td rowspan=1 colspan=2>Step</td><td rowspan=1 colspan=1>n (%)</td><td rowspan=1 colspan=1>Time per case</td></tr><tr><td rowspan=1 colspan=2>Manual stack exclusion</td><td rowspan=1 colspan=1>6 (5.0%)</td><td rowspan=1 colspan=1>~ 30 s</td></tr><tr><td rowspan=1 colspan=2>Manual template selection</td><td rowspan=1 colspan=1>3 (2.5%)</td><td rowspan=1 colspan=1>~ 60 s</td></tr><tr><td rowspan=1 colspan=2>Manual thoracic segmentation</td><td rowspan=1 colspan=1>3 (2.5%)</td><td rowspan=1 colspan=1>~ 120 s</td></tr><tr><td rowspan=1 colspan=2>Manual anatomical reorientation</td><td rowspan=1 colspan=1>12 (9.9%)</td><td rowspan=1 colspan=1>~ 60 s</td></tr><tr><td rowspan=1 colspan=2>Multiple interventions</td><td rowspan=1 colspan=1>2 (1.7%)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Unique cases</td><td rowspan=1 colspan=1>21 (17.4%)</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=2>2) Template se</td><td rowspan=1 colspan=2>lection</td></tr><tr><td rowspan=1 colspan=2>Template stack evaluation</td><td rowspan=1 colspan=2>n (%)</td></tr><tr><td rowspan=1 colspan=2>Failure</td><td rowspan=1 colspan=2>3 (2.5%)</td></tr><tr><td rowspan=1 colspan=2>Success</td><td rowspan=1 colspan=2>118 (97.5%)</td></tr><tr><td rowspan=1 colspan=2>3) Thoracic seg</td><td rowspan=1 colspan=2>mentation</td></tr><tr><td rowspan=1 colspan=2>Segmentation evaluation</td><td rowspan=1 colspan=2>n (%)</td></tr><tr><td rowspan=1 colspan=2>Failure</td><td rowspan=1 colspan=2>3 (2.5%)</td></tr><tr><td rowspan=1 colspan=2>Success</td><td rowspan=1 colspan=2>118 (97.5%)</td></tr><tr><td rowspan=1 colspan=2>4) Slice-to-volume r</td><td rowspan=1 colspan=2>econstruction</td></tr><tr><td rowspan=1 colspan=2>Reconstruction evaluation</td><td rowspan=1 colspan=1>n (%)</td><td rowspan=1 colspan=1>Time [min]</td></tr><tr><td rowspan=1 colspan=2>Failure</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Success</td><td rowspan=1 colspan=1>121 (100%)</td><td rowspan=1 colspan=1>4.9 ± 1.0</td></tr><tr><td rowspan=1 colspan=2>5) Anatomical reo</td><td rowspan=1 colspan=2>rientation</td></tr><tr><td rowspan=1 colspan=1>Cardiaclocalization</td><td rowspan=1 colspan=1>Reorientationevaluation</td><td rowspan=1 colspan=2>n (%)</td></tr><tr><td rowspan=2 colspan=1>x</td><td rowspan=1 colspan=1>Failure</td><td rowspan=1 colspan=2>31 (25.6%)</td></tr><tr><td rowspan=1 colspan=1>Success</td><td rowspan=1 colspan=2>90 (74.4%)</td></tr><tr><td rowspan=2 colspan=1>√</td><td rowspan=1 colspan=1>Failure</td><td rowspan=1 colspan=2>12 (9.9%)</td></tr><tr><td rowspan=1 colspan=1>Success</td><td rowspan=1 colspan=2>109 (90.1%)</td></tr><tr><td rowspan=1 colspan=2>6) End-to-end pipe</td><td rowspan=1 colspan=2>line summary</td></tr><tr><td rowspan=1 colspan=2>Pipeline evaluation</td><td rowspan=1 colspan=1>n (%)</td><td rowspan=1 colspan=1>Time [min]</td></tr><tr><td rowspan=1 colspan=2>Semi-automatic</td><td rowspan=1 colspan=1>21 (17.4)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Fully-automatic</td><td rowspan=1 colspan=1>100 (82.6)</td><td rowspan=1 colspan=1>7.1 ± 1.3</td></tr></table>

End-to-end SPARC pipeline evaluation on a held-out clinical cohort (� = 121 subjects). 1) Manual intervention summary: the number and estimated duration of manual interventions required at each pipeline stage. 2) Template stack selection evaluation. 3) Thoracic segmentation evaluation. 4) Slice-tovolume reconstruction evaluation with mean processing time. 5) Anatomical reorientation evaluation stratified by cardiac centre localisation ablation. 6) End-to-end pipeline summary: overall counts and mean processing time for fully automatic and semi-automatic runs.

implementation makes use of agreements between our ensemble deep learning models and consistency metrics in super-resolution reconstruction (see Figure S3). By identifying suitable thresholds for these QC metrics, we can establish safe operating zones where the fully automated pipeline can be trusted, thus significantly reducing the workload of clinicians.

## 5.2. Limitations

Our pipeline relies on successful Doppler gating, which can be lost with extreme fetal motion. Subsequent repositioning of the device results in temporal misalignment of the stacks. Automatic detection and re-alignment of temporally misaligned stacks (Jansz et al., 2010) represents a natural direction for future work.

We adopted a rigid motion model under the assumption of limited anatomical deformation within the thoracic cage.

While investigating deformable slice-to-volume registration (Uus et al., 2020) could yield further improvements, it introduces a significant computational penalty that currently conflicts with our goal of rapid clinical turnaround.

The SPARC pipeline is currently tailored for thirdtrimester fetuses where gross fetal mobility is relatively restricted. Recent developments combining slice-to-volume reconstruction with initial landmark-based registration have shown promise in handling the larger inter-slice motion characteristic of earlier gestational ages (Uus et al., 2022), providing a clear pathway for future work to expand the applicable gestational age range.

## 6. Conclusion

We presented a hybrid pipeline combining physicsinformed slice-to-volume reconstruction with deep learning components for automated 3D+time fetal cardiac cine reconstruction. By exploiting the temporal consistency afforded by Doppler gating, the proposed model achieves a tenfold reduction in computational complexity relative to existing approaches without loss in reconstruction quality and superior generalisation performance on held-out data. Deep learning components were trained under a restricted annotation budget via transfer learning from a larger source domain, enabling fully automatic processing in 82.6% of clinical cases, reaching 100% success for all cases meeting minimal inclusion criteria through eficient manual intervention, with a mean end-to-end processing time of 7.1 ± 1.3 min compatible with clinical deployment requirements. The current pipeline enables visualisation of intracardiac anatomy and function. Future work will focus on extracting downstream clinical metrics, such as ventricular volumes, ejection fraction, and flow velocity maps, to further establish its diagnostic value in assessing congenital heart disease.

The complete SPARC pipeline is publicly available as a Docker container<sup>4</sup> and is currently deployed at our institution as a clinical diagnostic and research tool for fetal cardiac magnetic resonance imaging.

## 7. Acknowledgement

This work was funded by core funding from the Wellcome/EPSRC Centre for Medical Engineering [WT203148/Z/16/Z] and by the National Institute for Health and Care Research (NIHR) Clinical Research Facility (CRF) and HealthTech Research Centre in Cardiovascular and Respiratory Medicine (HRC) at Guy’s and St Thomas’ NHS Foundation Trust. The views expressed are those of the author(s) and not necessarily those of the NHS, the NIHR or the Department of Health and Social Care.

## References

van Amerom, J.F., Lloyd, D.F., Price, A.N., Kuklisova Murgasova, M., Aljabar, P., Malik, S.J., Lohezic, M., Rutherford, M.A., Pushparajah, K., Razavi, R., Hajnal, J.V., 2018. Fetal cardiac cine imaging using

highly accelerated dynamic MRI with retrospective motion correction and outlier rejection. Magnetic Resonance in Medicine 79, 327–338. doi:10.1002/mrm.26686.

van Amerom, J.F.P., Lloyd, D.F.A., Deprez, M., Price, A.N., Malik, S.J., Pushparajah, K., van Poppel, M.P.M., Rutherford, M.A., Razavi, R., Hajnal, J.V., 2019. Fetal whole-heart 4D imaging using motion-corrected multi-planar real-time MRI. Magnetic Resonance in Medicine 82, 1055– 1072. doi:10.1002/mrm.27798.

Berggren, K., Ryd, D., Heiberg, E., Aletras, A.H., Hedström, E., 2022. Super-Resolution Cine Image Enhancement for Fetal Cardiac Magnetic Resonance Imaging. Journal of Magnetic Resonance Imaging 56, 223– 231. doi:10.1002/jmri.27956.

Cardoso, M.J., Li, W., Brown, R., Ma, N., Kerfoot, E., Wang, Y., Murrey, B., Myronenko, A., Zhao, C., Yang, D., Nath, V., He, Y., Xu, Z., Hatamizadeh, A., Myronenko, A., Zhu, W., Liu, Y., Zheng, M., Tang, Y., Yang, I., Zephyr, M., Hashemian, B., Alle, S., Darestani, M.Z., Budd, C., Modat, M., Vercauteren, T., Wang, G., Li, Y., Hu, Y., Fu, Y., Gorman, B., Johnson, H., Genereaux, B., Erdal, B.S., Gupta, V., Diaz-Pinto, A., Dourson, A., Maier-Hein, L., Jaeger, P.F., Baumgartner, M., Kalpathy-Cramer, J., Flores, M., Kirby, J., Cooper, L.A.D., Roth, H.R., Xu, D., Bericat, D., Floca, R., Zhou, S.K., Shuaib, H., Farahani, K., Maier-Hein, K.H., Aylward, S., Dogra, P., Ourselin, S., Feng, A., 2022. MONAI: An open-source framework for deep learning in healthcare. doi:10.48550/arXiv.2211.02701. arXiv:2211.02701 [cs].

Charbonnier, P., Blanc-Feraud, L., Aubert, G., Barlaud, M., 1997. Deterministic edge-preserving regularization in computed imaging. IEEE Transactions on Image Processing 6, 298–311. doi:10.1109/83.551699.

Chen, Q., Liu, A., Zhang, J., Yang, C., Zhang, Y., 2026. Foundation models in medical imaging: A review. EngMedicine 3, 100123. doi:10.1016/j. engmed.2026.100123.

Dong, S.Z., Zhu, M., Ji, H., Ren, J.Y., Liu, K., 2020. Fetal cardiac MRI: a single center experience over 14-years on the potential utility as an adjunct to fetal technically inadequate echocardiography. Scientific Reports 10, 12373. doi:10.1038/s41598-020-69375-3.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N., 2020. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, in: International Conference on Learning Representations. URL: https://openreview.net/ forum?id=YicbFdNTTy.

Gardiner, H.M., Kovacevic, A., Heijden, L.B.v.d., Pfeifer, P.W., Franklin, R.C., Gibbs, J.L., Averiss, I.E., LaRovere, J.M., 2014. Prenatal screening for major congenital heart disease: assessing performance by combining national cardiac audit with maternity data. Heart 100, 375–382. doi:10. 1136/heartjnl-2013-304640.

Geiger, J., Tuura, R.O., Callaghan, F.M., Burkhardt, B.E., Kellenberger, C.J., Valsangiacomo Buechel, E.R., 2023. Feasibility of Non-Gated Dynamic Fetal Cardiac MRI for Identification of Fetal Cardiovascular Anatomy. Fetal Diagnosis and Therapy 50, 8–16. doi:10.1159/000528966.

Gholipour, A., Warfield, S.K., 2019. Motion-corrected foetal cardiac MRI. Nature Biomedical Engineering 3, 852–854. doi:10.1038/ s41551-019-0476-2.

Guan, H., Liu, M., 2022. Domain Adaptation for Medical Image Analysis: A Survey. IEEE Transactions on Biomedical Engineering 69, 1173– 1185. doi:10.1109/TBME.2021.3117407.

Guan, H., Yap, P.T., Bozoki, A., Liu, M., 2024. Federated learning for medical image analysis: A survey. Pattern Recognition 151, 110424. doi:10.1016/j.patcog.2024.110424.

Haris, K., Hedström, E., Bidhult, S., Testud, F., Maglaveras, N., Heiberg, E., Hansson, S.R., Arheden, H., Aletras, A.H., 2017. Self-gated fetal cardiac MRI with tiny golden angle iGRASP: A feasibility study: Self-Gated Fetal Cardiac MRI With iGRASP. Journal of Magnetic Resonance Imaging 46, 207–217. doi:10.1002/jmri.25599.

Haris, K., Hedström, E., Kording, F., Bidhult, S., Steding-Ehrenborg, K., Ruprecht, C., Heiberg, E., Arheden, H., Aletras, A.H., 2020. Freebreathing fetal cardiac MRI with doppler ultrasound gating, compressed sensing, and motion compensation. Journal of Magnetic Resonance Imaging 51, 260–272. doi:10.1002/jmri.26842.

Hartley, R., Trumpf, J., Dai, Y., Li, H., 2013. Rotation Averaging. International Journal of Computer Vision 103, 267–305. doi:10.1007/ s11263-012-0601-0.

Holland, B.J., Myers, J.A., Woods Jr, C.R., 2015. Prenatal diagnosis of critical congenital heart disease reduces risk of death from cardiovascular compromise prior to planned neonatal cardiac surgery: a meta-analysis. Ultrasound in Obstetrics & Gynecology 45, 631–638. doi:10.1002/uog. 14882.

Hou, B., Khanal, B., Alansary, A., McDonagh, S., Davidson, A., Rutherford, M., Hajnal, J.V., Rueckert, D., Glocker, B., Kainz, B., 2018a. 3-D Reconstruction in Canonical Co-Ordinate Space From Arbitrarily Oriented 2-D Images. IEEE Transactions on Medical Imaging 37, 1737– 1750. doi:10.1109/TMI.2018.2798801.

Hou, B., Miolane, N., Khanal, B., Lee, M.C.H., Alansary, A., McDonagh, S., Hajnal, J.V., Rueckert, D., Glocker, B., Kainz, B., 2018b. Computing CNN Loss and Gradients for Pose Estimation with Riemannian Geometry, in: Medical Image Computing and Computer Assisted Intervention, pp. 756–764. doi:10.1007/978-3-030-00928-1\_85.

Iofe, S., Szegedy, C., 2015. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. doi:10.48550/ arXiv.1502.03167. arXiv:1502.03167 [cs].

Jansz, M.S., Seed, M., Van Amerom, J.F.P., Wong, D., Grosse-Wortmann, L., Yoo, S., Macgowan, C.K., 2010. Metric optimized gating for fetal cardiac MRI. Magnetic Resonance in Medicine 64, 1304–1314. doi:10. 1002/mrm.22542.

Kingma, D.P., Ba, J., 2017. Adam: A Method for Stochastic Optimization. doi:10.48550/arXiv.1412.6980. arXiv:1412.6980 [cs].

Kording, F., Schoennagel, B.P., de Sousa, M.T., Fehrs, K., Adam, G., Yamamura, J., Ruprecht, C., 2018a. Evaluation of a Portable Doppler Ultrasound Gating Device for Fetal Cardiac MR Imaging: Initial Results at 1.5T and 3T. Magnetic resonance in medical sciences 17, 308–317. doi:10.2463/mrms.mp.2017-0100.

Kording, F., Yamamura, J., De Sousa, M.T., Ruprecht, C., Hedström, E., Aletras, A.H., Ellen Grant, P., Powell, A.J., Fehrs, K., Adam, G., Kooijman, H., Schoennagel, B.P., 2018b. Dynamic fetal cardiovascular magnetic resonance imaging using Doppler ultrasound gating. Journal of Cardiovascular Magnetic Resonance 20, 17. doi:10.1186/ s12968-018-0440-4.

Kuklisova-Murgasova, M., Quaghebeur, G., Rutherford, M.A., Hajnal, J.V., Schnabel, J.A., 2012. Reconstruction of fetal brain MRI with intensity matching and complete outlier removal. Medical Image Analysis 16, 1550–1564. doi:10.1016/j.media.2012.07.004.

Lim, Z.N., Woodgate, T., Paul, J., Poppel, M.V., Steinweg, J., Skelton, E., Sharland, G., Miller, O., Eglof, A., Rutherford, M., Zidere, V., Vigneswaran, T., Simpson, J., Pushparajah, K., Lloyd, D., 2024. Fetal Cardiac MRI in Clinical Practice: Report of over 350 Fetal CMR Scans in a Large Tertiary Fetal and Paediatric Cardiology Centre. Journal of Cardiovascular Magnetic Resonance 26, 100131. doi:10.1016/j.jocmr. 2024.100131.

Liu, Y., Chen, S., Zühlke, L., Black, G.C., Choy, M.k., Li, N., Keavney, B.D., 2019. Global birth prevalence of congenital heart defects 1970–2017: updated systematic review and meta-analysis of 260 studies. International Journal of Epidemiology 48, 455–463. doi:10.1093/ije/ dyz009.

Lloyd, D.F., Van Poppel, M.P., Pushparajah, K., Vigneswaran, T.V., Zidere, V., Steinweg, J., Van Amerom, J.F., Roberts, T.A., Schulz, A., Charakida, M., Miller, O., Sharland, G., Rutherford, M., Hajnal, J.V., Simpson, J.M., Razavi, R., 2021. Analysis of 3-Dimensional Arch Anatomy, Vascular Flow, and Postnatal Outcome in Cases of Suspected Coarctation of the Aorta Using Fetal Cardiac Magnetic Resonance Imaging. Circulation: Cardiovascular Imaging 14. doi:10.1161/CIRCIMAGING.121.012411.

Lloyd, D.F.A., Pushparajah, K., Simpson, J.M., Van Amerom, J.F.P., Van Poppel, M.P.M., Schulz, A., Kainz, B., Deprez, M., Lohezic, M., Allsop, J., Mathur, S., Bellsham-Revell, H., Vigneswaran, T., Charakida, M., Miller, O., Zidere, V., Sharland, G., Rutherford, M., Hajnal, J.V., Razavi, R., 2019. Three-dimensional visualisation of the fetal heart using prenatal MRI with motion-corrected slice-volume registration: a prospective, single-centre cohort study. The Lancet 393, 1619–1627.

doi:10.1016/S0140-6736(18)32490-5.

Lloyd, D.F.A., Van Amerom, J.F.P., Pushparajah, K., Simpson, J.M., Zidere, V., Miller, O., Sharland, G., Allsop, J., Fox, M., Lohezic, M., Murgasova, M., Malamateniou, C., Hajnal, J.V., Rutherford, M., Razavi, R., 2016. An exploration of the potential utility of fetal cardiovascular MRI as an adjunct to fetal echocardiography. Prenatal Diagnosis 36, 916–925. doi:10.1002/pd.4912.

Mamalis, M., Bedei, I., Schoennagel, B., Kording, F., Reitz, J.G., Wolter, A., Schenk, J., Axt-Fliedner, R., 2022. The Evolution and Developing Importance of Fetal Magnetic Resonance Imaging in the Diagnosis of Congenital Cardiac Anomalies: A Systematic Review. Journal of Clinical Medicine 11, 7027. doi:10.3390/jcm11237027.

Milletari, F., Navab, N., Ahmadi, S.A., 2016. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. doi:10. 48550/arXiv.1606.04797. arXiv:1606.04797 [cs].

Minocha, P.K., Englund, E.K., Friesen, R.M., Fujiwara, T., Smith, S.A., Meyers, M.L., Browne, L.P., Barker, A.J., 2024. Reference alues for Fetal Cardiac Dimensions, Volumes, Ventricular Function and Left Ventricular Longitudinal Strain Using Doppler Ultrasound Gated Cardiac Magnetic Resonance Imaging in Healthy Third Trimester Fetuses. Journal of Magnetic Resonance Imaging 60, 365–374. doi:10.1002/jmri. 29077.

Moscatelli, S., Leo, I., Lisignoli, V., Boyle, S., Bucciarelli-Ducci, C., Secinaro, A., Montanaro, C., 2023. Cardiovascular Magnetic Resonance from Fetal to Adult Life—Indications and Challenges: A State-of-the-Art Review. Children 10, 763. doi:10.3390/children10050763.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S., 2019. PyTorch: an imperative style, high-performance deep learning library, in: Proceedings of the 33rd International Conference on Neural Information Processing Systems. 721, pp. 8026–8037.

Roberts, T.A., Van Amerom, J.F.P., Uus, A., Lloyd, D.F.A., Van Poppel, M.P.M., Price, A.N., Tournier, J.D., Mohanadass, C.A., Jackson, L.H., Malik, S.J., Pushparajah, K., Rutherford, M.A., Razavi, R., Deprez, M., Hajnal, J.V., 2020. Fetal whole heart blood flow imaging using 4D cine MRI. Nature Communications 11. doi:10.1038/s41467-020-18790-1.

Ronneberger, O., Fischer, P., Brox, T., 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation, in: Medical Image Computing and Computer-Assisted Intervention, pp. 234–241. doi:10.1007/ 978-3-319-24574-4\_28.

Rousseau, F., Glenn, O., Iordanova, B., Rodriguez-Carranza, C., Vigneron, D., Barkovich, J., Studholme, C., 2005. A Novel Approach to High Resolution Fetal Brain MR Imaging, in: Medical Image Computing and Computer-Assisted Intervention, Berlin, Heidelberg. pp. 548–555. doi:10.1007/11566465\_68.

Roy, C.W., Seed, M., Van Amerom, J.F.P., Al Nafisi, B., Grosse-Wortmann, L., Yoo, S., Macgowan, C.K., 2013. Dynamic imaging of the fetal heart using metric optimized gating. Magnetic Resonance in Medicine 70, 1598–1607. doi:10.1002/mrm.24614.

Roy, C.W., Van Amerom, J.F., Marini, D., Seed, M., Macgowan, C.K., 2019. Fetal Cardiac MRI: A Review of Technical Advancements. Topics in Magnetic Resonance Imaging 28, 235–244. doi:10.1097/RMR. 0000000000000218.

Ryd, D., Fricke, K., Bhat, M., Arheden, H., Liuba, P., Hedström, E., 2021. Utility of Fetal Cardiovascular Magnetic Resonance for Prenatal Diagnosis of Complex Congenital Heart Defects. JAMA Network Open 4, e213538. doi:10.1001/jamanetworkopen.2021.3538.

Sitzmann, V., Martel, J., Bergman, A., Lindell, D., Wetzstein, G., 2020. Implicit Neural Representations with Periodic Activation Functions, in: Advances in Neural Information Processing Systems, pp. 7462–7473. URL: https://proceedings.neurips.cc/paper/2020/hash/ 53c04118df112c13a8c34b38343b9c10-Abstract.html.

Tavares De Sousa, M., Hecher, K., Yamamura, J., Kording, F., Ruprecht, C., Fehrs, K., Behzadi, C., Adam, G., Schoennagel, B.P., 2019. Dynamic fetal cardiac magnetic resonance imaging in four-chamber view using Doppler ultrasound gating in normal fetal heart and in congenital

heart disease: comparison with fetal echocardiography. Ultrasound in Obstetrics & Gynecology 53, 669–675. doi:10.1002/uog.20167.

Tustison, N.J., Avants, B.B., Cook, P.A., Zheng, Y., Egan, A., Yushkevich, P.A., Gee, J.C., 2010. N4ITK: Improved N3 Bias Correction. IEEE Transactions on Medical Imaging 29, 1310–1320. doi:10.1109/TMI.2010. 2046908.

Uus, A., Zhang, T., Jackson, L.H., Roberts, T.A., Rutherford, M.A., Hajnal, J.V., Deprez, M., 2020. Deformable Slice-to-Volume Registration for Motion Correction of Fetal Body and Placenta MRI. IEEE transactions on medical imaging 39, 2750–2759. doi:10.1109/TMI.2020.2974844.

Uus, A.U., Grigorescu, I., Van Poppel, M.P., Steinweg, J.K., Roberts, T.A., Rutherford, M.A., Hajnal, J.V., Lloyd, D.F., Pushparajah, K., Deprez, M., 2022. Automated 3D reconstruction of the fetal thorax in the standard atlas space from motion-corrupted MRI stacks for 21–36 weeks GA range. Medical Image Analysis 80, 102484. doi:10.1016/j.media. 2022.102484.

Vollbrecht, T.M., Bissell, M.M., Kording, F., Geipel, A., Isaak, A., Strizek, B.S., Hart, C., Barker, A.J., Luetkens, J.A., 2024a. Fetal Cardiac MRI Using Doppler US Gating: Emerging Technology and Clinical Implications. Radiology: Cardiothoracic Imaging 6, e230182. doi:10. 1148/ryct.230182.

Vollbrecht, T.M., Hart, C., Zhang, S., Katemann, C., Isaak, A., Pieper, C.C., Kuetting, D., Faridi, B., Strizek, B., Attenberger, U., Kipfmueller, F., Herberg, U., Geipel, A., Luetkens, J.A., 2023. Fetal Cardiac Cine MRI with Doppler US Gating in Complex Congenital Heart Disease. Radiology: Cardiothoracic Imaging 5, e220129. doi:10.1148/ryct.220129.

Vollbrecht, T.M., Hart, C., Zhang, S., Katemann, C., Sprinkart, A.M., Isaak, A., Attenberger, U., Pieper, C.C., Kuetting, D., Geipel, A., Strizek, B., Luetkens, J.A., 2024b. Deep learning denoising reconstruction for improved image quality in fetal cardiac cine MRI. Frontiers in Cardiovascular Medicine 11, 1323443. doi:10.3389/fcvm.2024.1323443.

Votino, C., Jani, J., Damry, N., Dessy, H., Kang, X., Cos, T., Divano, L., Foulon, W., De Mey, J., Cannie, M., 2012. Magnetic resonance imaging in the normal fetal heart and in congenital heart disease. Ultrasound in Obstetrics & Gynecology 39, 322–329. doi:10.1002/uog.10061.

Wang, J., Lan, C., Liu, C., Ouyang, Y., Qin, T., Lu, W., Chen, Y., Zeng, W., Yu, P.S., 2023. Generalizing to Unseen Domains: A Survey on Domain Generalization. IEEE Transactions on Knowledge and Data Engineering 35, 8052–8072. doi:10.1109/TKDE.2022.3178128.

Woodgate, T., Steinweg, J., Franklin, R., Price, A., Roberts, T., Boutillon, A., Uus, A., Hajnal, J., Deprez, M., Pushparajah, K., Lloyd, D., 2024. Time-resolved 3D MRI Volumetry of the Fetal Heart Using a Novel Combination of Doppler Ultrasound Gating and Motion-corrected Slicevolume Registration. Journal of Cardiovascular Magnetic Resonance 26. doi:10.1016/j.jocmr.2024.100552.

Xu, J., Moyer, D., Gagoski, B., Iglesias, J.E., Grant, P.E., Golland, P., Adalsteinsson, E., 2023. NeSVoR: Implicit Neural Representation for Slice-to-Volume Reconstruction in MRI. IEEE Transactions on Medical Imaging 42, 1707–1719. doi:10.1109/TMI.2023.3236216.

Yushkevich, P.A., Piven, J., Hazlett, H.C., Smith, R.G., Ho, S., Gee, J.C., Gerig, G., 2006. User-guided 3D active contour segmentation of anatomical structures: Significantly improved eficiency and reliability. NeuroImage 31, 1116–1128. doi:10.1016/j.neuroimage.2006.01.015.

## Supplementary Material

Case 1  
Case 2  
Case 3  
![](images/ed7cca9804d9d2b807747e92c81259aac2752de36405427a09ddbdda8a5f310f.jpg)  
Figure S1: Qualitative comparison of thoracic segmentation contours for three target domain subjects across the four training strategies (source-only, target-only, joint, and transfer). Time-averaged stacks were selected to represent distinct stack orientations and fetal position. Ground truth delineations are in red ( ) while predictions appear in green ( ).

![](images/06fd7cb67a030b97e3eebe65c0e3d3eb5197e04a52cc5e775a3bf5049dab3e7a.jpg)  
Figure S2: Geodesic distance (GD) of the Transfer approach as a function of simulated rotation magnitude � , evaluated on the held-out target domain test set (Doppler-gated; � = 7 subjects, 250 randomly sampled rigid transformations per subject). Each point represents one simulated transformation, with subjects colour-coded.

![](images/528bda1f5ededbc81099f57521659568e64353076a47a5567a6d1daf7557688f.jpg)

![](images/b5547c7de253237a4b502a42e8f48d35eadd196ec3a27557828ba8f47e7b6192.jpg)

![](images/eeb0a196b8aed26e23687adcbb449e69523b0396b451e9c9b64d5e157baaa720.jpg)  
Figure S3: Automatic quality control metrics for the three main SPARC pipeline stages, evaluated on the end-to-end clinical cohort (� = 121 subjects). A) Thoracic segmentation: mean pairwise inter-model DSC versus $\mathrm { H D } _ { 9 5 \% }$ for each subject, colour-coded by segmentation outcome. The post-hoc acceptance region is define by $\mathsf { D S C } > 8 5 \%$ and $\mathrm { H D } _ { 9 5 \% } < 5 \mathsf { m m . B } )$ Cine reconstruction: data consistency metrics (NCC, NRMSE) and proportion of included slices for pipeline-accepted $\left( n = 1 2 1 \right)$ and pipeline-rejected $( n = 1 8 )$ cases. C) Reorientation: mean pairwise inter-model GD versus CD for each subject, colour-coded by reorientation outcome. The post-hoc acceptance region was defined as $\mathsf { G D } < 2 5 ^ { \circ }$ and $\mathrm { C D } < 5 \mathrm { m m }$