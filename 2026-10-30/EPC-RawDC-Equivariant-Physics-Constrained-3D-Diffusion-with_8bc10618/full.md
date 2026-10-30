# EPC-RawDC: Equivariant Physics Constrained 3D Diffusion with Raw Data Consistency for CBCT to CT Image Synthesis

Alzahra Altalib, Chunhui Li, Alessandro Perelli

Abstract—Cone beam computed tomography (CBCT) is widely used for image guided and adaptive radiotherapy, but scatter, noise, reconstruction artifacts, and inaccurate Hounsfield units limit its quantitative use. Recent diffusion approaches for CBCT to CT synthesis remain largely image driven, with limited use of the underlying acquisition physics. We propose EPC-RawDC, a conditional 3D latent diffusion framework that combines volumetric generative modeling with two complementary constraints derived from the same scanner specific CBCT forward model. First, geometry equivariant projection constraint exploits the rotational equivariance of the CBCT forward operator by supervising the synthesized CT through correspondingly transformed reference projections. Second, raw data consistency uses noisy projection measurements of the individual acquisition as scan specific supervision during training. The framework was evaluated on two independent clinical cohorts with distinct scanner geometries and projection sources. For JUST, the physics constraints used measured clinical raw CBCT projections acquired with an Elekta system, whereas for SynthRAD2025 the projection data were simulated from the reference CT using a Varian geometry with Poisson photon statistics. EPC-RawDC achieved an SSIM of 0.97 in both cohorts and improved PSNR over DDPM by 2.66 dB on SynthRAD2025 and 2.10 dB on JUST, while reducing mean MAE by 20.3% and 19.0%, respectively. On SynthRAD2025, projection consistency and equivariant errors were reduced by 55.0% and 41.4% relative to DDPM. These results demonstrate that combining a volumetric diffusion prior with scanner geometry and scan specific projection measurements improves quantitative CBCT to CT synthesis across both simulated and measured projection domains. Code available: https://github.com/ALZAHRAALTALIB/EPC-RawDC.

Index Terms—CBCT to CT Image Synthesis, Conditional Diffu sion models, Equivariant Imaging, Physics consistent optimization, head and neck, Radiotherapy

## I. INTRODUCTION

Cone beam computed tomography (CBCT) is widely used for image guidance and adaptive radiotherapy because it provides volumetric information directly at the treatment position. Its quantitative use, however, remains limited by scatter, beam hardening, photon noise, truncation, detector effects, and reconstruction artifacts. These degradations reduce soft tissue contrast and disturb the relationship between reconstructed intensity and tissue attenuation, resulting in inaccurate Hounsfield units (HU) and limiting applications such as dose calculation and online treatment adaptation. Synthetic CT (sCT) generation from CBCT has therefore become an important approach for obtaining CT like quantitative information from daily treatment imaging. The SynthRAD2025 challenge demonstrated substantial progress in this area using large multicenter datasets, while also showing that image similarity alone does not necessarily predict dosimetric accuracy [1]. This motivates methods that preserve image quality while incorporating stronger knowledge of the underlying acquisition process.

Early learning based CBCT to CT methods relied primarily on convolutional networks and generative adversarial models. Cycle consistent architectures enabled unpaired image translation and substantially improved HU accuracy over uncorrected CBCT [2], while more recent approaches have incorporated transformer architectures, anatomical constraints, and uncertainty estimation [3]. Diffusion models have subsequently emerged as a powerful alternative because of their stable denoising objective and ability to model complex conditional image distributions. Peng et al. demonstrated conditional denoising diffusion for CBCT based CT generation with improved image quality and HU fidelity [4]. More recent studies have explored anatomical guidance, frequency information, patient adaptation, latent representations, and external clinical validation [5]. 3D latent diffusion has also been investigated for volumetric CBCT to CT synthesis [6], while multicenter conditional diffusion studies have shown the feasibility of validation across independent clinical populations [7]. Consequently, neither latent diffusion nor multicenter evaluation alone constitutes the principal novelty of the present work.

Despite these advances, most CBCT to CT synthesis methods remain predominantly image driven. A generated CT can be anatomically plausible and visually realistic without necessarily satisfying the physical relationships imposed by the CBCT acquisition system or remaining compatible with the projection measurements from which the input CBCT was reconstructed. Latent diffusion provides an efficient mechanism for modeling complete three dimensional volumes [8], but the learned generative prior alone does not guarantee consistency with scanner geometry or patient specific measurements. This limitation is particularly relevant to CBCT, where the acquisition trajectory and forward projection model provide strong physical structure that is known independently of the training data.

Equivariance provides one mechanism for incorporating this information. Equivariant imaging demonstrated that symmetries of an imaging operator can constrain learned inverse problems beyond conventional image domain objectives [9]. In CBCT to CT synthesis, EqDiff-CT introduced rotational equivariance at the network representation level [10]. Our previous EPC-3D-Diff framework instead derived a projection domain constraint from the rotational equivariance of the CBCT acquisition operator [11]. An in plane rotation of the three dimensional object about the scanner axis produces a corresponding angular displacement of its projections. EPC-3D-Diff used this scanner derived transformation law to construct geometry transformed projection supervision within a conditional three dimensional latent diffusion model, rather than generic image domain rotation augmentation.

The geometry derived term in EPC-3D-Diff uses a noise free reference projection transformed according to the deterministic symmetry of the scanner operator; it does not incorporate the noisy projection realization measured for an individual scan. The present work addresses this limitation by introducing raw data consistency (RawDC). Both terms use the same scanner specific forward model, but RawDC adds patient and scan specific measurement supervision that is absent from the geometry transformed reference projection term. The acquired photon counts originate from Poisson statistics, while the RawDC objective uses the corresponding post-log quadratic Gaussian approximation developed in Section II-A. Importantly, the implemented forward model preserves scanner specificity by evaluating projections at the native detector resolution of the corresponding acquisition system.

## A. Positioning Against Existing Methods

Existing CBCT to CT synthesis methods based on GANs, transformers, and conditional diffusion primarily learn image domain mappings [2]–[5] and volumetric latent diffusion improves the efficiency of three dimensional modeling [6]. Projection consistency has also been incorporated into tomographic reconstruction methods [12], [13], but these approaches reconstruct images directly from projection data rather than perform conditional CBCT to CT synthesis. EPC-RawDC combines a volumetric latent diffusion prior, geometry transformed projection supervision derived from scanner equivariance, and scan specific raw projection consistency within a single synthesis framework.

Relative to EqDiff and EPC-3D-Diff [10], [11], the present work introduces scan specific raw data consistency that directly constrains the synthesized CT using the projection measurements of the corresponding acquisition.

## B. Main Contributions

The principal novelty of EPC-RawDC is the integration of an equivariance derived projection constraint with scan specific raw projection consistency within a common acquisition CBCT forward model for the conditional 3D latent diffusion.

The contributions are threefold. First, we formulate conditional 3D latent diffusion for volumetric CBCT to CT synthesis. Second, we impose geometry transformed projection supervision using the physical relationship between object rotation and projection angle shift induced by the scanner operator. Third, RawDC constrains the predicted CT during training using the projection measurements of the corresponding acquisition. Both physics terms use the same acquisition faithful forward operator while encoding complementary reference geometry and scan specific measurement information. The framework is evaluated on independent datasets with distinct scanner geometries using controlled ablations of volumetric diffusion, geometry derived projection supervision, and raw data consistency.

Together, these contributions establish EPC-RawDC as a measurement grounded generative framework in which the sCT is constrained jointly by the learned CT distribution, the known scanner geometry, and the projection measurements of the individual acquisition.

## II. PHYSICS BASED FORMULATION

## A. Statistical CT and CBCT Measurement Model

Following the notation of EPC-3D-Diff, let ${ \bf x } _ { r }$ denote the attenuation volume and $\mathbf { A } _ { r }$ the corresponding X-ray forward operator, with $r \in \{ 0 , c \}$ indicating CT and CBCT, respectively. Under the Beer–Lambert model, the detected photon counts are modeled as

$$
\mathbf { p } _ { r } \mid \mathbf { x } _ { r } \sim \operatorname { P o i s s o n } \left( \mathbf { I } _ { r } \odot \exp \left( - \mathbf { A } _ { r } \mathbf { x } _ { r } \right) \right) ,\tag{1}
$$

where ${ { \mathbf { I } } _ { r } }$ denotes the incident photon fluence and $\odot$ denotes elementwise multiplication. After logarithmic normalization,

$$
\mathbf { y } _ { r } = - \log \left( \mathbf { p } _ { r } \oslash \mathbf { I } _ { r } \right) ,\tag{2}
$$

where $\oslash$ denotes elementwise division. For sufficiently high photon counts, applying the Gaussian approximation to the Poisson distribution together with a first order approximation of the logarithmic transform gives

$$
\begin{array} { r } { \mathbf { y } _ { r } \approx \mathbf { A } _ { r } \mathbf { x } _ { r } + \mathbf { n } _ { r } , \qquad \mathbf { n } _ { r } \sim \mathcal { N } \left( \mathbf { 0 } , \pmb { \Sigma } _ { r } \right) , } \end{array}\tag{3}
$$

with approximately signal dependent variance $\sigma _ { r , j } ^ { 2 } \simeq 1 / \bar { p } _ { r , j }$ Defining ${ \mathbf W } _ { r } = { \boldsymbol \Sigma } _ { r } ^ { - 1 }$ , the corresponding Gaussian negative log likelihood reduces, up to additive and multiplicative constants, to the weighted least squares data fidelity

$$
\mathcal { L } _ { \mathrm { W L S } } = \left. \mathbf { W } _ { r } ^ { 1 / 2 } \left( \mathbf { y } _ { r } - \mathbf { A } _ { r } \mathbf { x } _ { r } \right) \right. _ { 2 } ^ { 2 } .\tag{4}
$$

Thus, although the underlying photon statistics are Poisson, the post-log measurements admit a convenient heteroscedastic Gaussian approximation, providing the statistical basis for quadratic projection domain consistency terms [14], [15]. When uniform weighting is adopted $\mathbf { W } _ { r } = \mathbf { I } ,$ Eq. (4) reduces to the standard least squares objective.

## B. CBCT Geometry Induced Projection Equivariance

Equivariance in the proposed framework is not introduced as a generic image augmentation prior. Instead, it follows directly from the rotational symmetry of the CBCT acquisition geometry. In a circular CBCT scan, the X-ray source and detector rotate around a fixed scanner axis while maintaining their relative source, isocenter, and detector geometry [9]. Consequently, rotating the object around the axis produces an equivalent angular displacement of its projection data.

Let $\mathbf { R } _ { \phi }$ denote a three dimensional rotation of an attenuation volume by angle ϕ about the CBCT rotation axis, and let $\mathbf { A } _ { c }$ denote the CBCT forward projector at the native detector resolution. For a projection acquired at gantry angle $\eta ,$ , the continuous circular acquisition geometry satisfies

$$
\left[ \mathbf { A } _ { c } \mathbf { R } _ { \phi } \mathbf { x } \right] ( \eta , u , v ) = \left[ \mathbf { A } _ { c } \mathbf { x } \right] ( \eta - \phi , u , v ) ,\tag{5}
$$

where $( u , v )$ are detector coordinates. Defining the angular translation operator $\left[ \mathbf { T } _ { \phi } \mathbf { y } \right] ( \eta , u , v ) : = \mathbf { y } ( \eta - \phi , u , v )$ , Eq. (5) gives the scanner geometry induced relation

$$
\mathbf { A } _ { c } \mathbf { R } _ { \phi } = \mathbf { T } _ { \phi } \mathbf { A } _ { c } .\tag{6}
$$

Equation (6) is an intertwining relation between rotations in the image domain and angular translations in the projection domain. It characterizes equivariance of the CBCT acquisition operator, is determined by the circular scanner geometry rather than learned from the training data, and forms the physical basis of the geometry derived projection constraint introduced in Section III-B.

In the ideal continuous circular geometry, Eq. (6) is exact. In the discrete implementation it is satisfied approximately because the projection angles, detector grid, volume and rotational interpolation are discretized. Accordingly, $\mathbf { T } _ { \phi }$ is implemented as an integer angular shift when ϕ coincides with the acquisition sampling and by fractional angular interpolation otherwise.

Importantly, Eq. (6) describes the acquisition operator and does not by itself imply equivariance of the learned CBCT to CT mapping. In the proposed training objective, this operator relation is used to construct geometry transformed reference projection supervision, while RawDC introduces scan specific measurement supervision. The two terms therefore use the same scanner model but constrain the prediction with different projection domain targets.

## III. PROPOSED EPC-RAWDC FRAMEWORK

Building on the statistical measurement model and the scanner geometry induced equivariance established in Section II-B, EPC-RawDC combines conditional 3D latent diffusion with two complementary projection domain physics terms. The first provides geometry transformed reference projection supervision derived from the rotational equivariance of the acquisition oper ator, whereas RawDC uses the noisy projection measurements of the individual scan as training supervision.

Let $\mathbf { x } _ { \mathrm { 0 } }$ denote the reference CT volume, $\mathbf { x } _ { c }$ the spatially aligned CBCT volume, and A the scanner specific CBCT forward projector at the native detector resolution.

## A. Conditional 3D Latent Diffusion Model

Direct diffusion on full resolution 3D volumes is computationally demanding. We therefore perform the generative process in a compact volumetric latent space, following the latent diffusion principle [8] and the conditional formulation introduced in EPC-3D-Diff. A shared encoder $\mathcal { E } _ { \psi }$ maps the paired CT and CBCT volumes to

$$
\begin{array} { r } { \mathbf { z } _ { 0 } = \mathcal { E } _ { \psi } ( \mathbf { x } _ { 0 } ) , \qquad \mathbf { z } _ { c } = \mathcal { E } _ { \psi } ( \mathbf { x } _ { c } ) , } \end{array}\tag{7}
$$

with $\mathbf { z } _ { 0 } , \mathbf { z } _ { c } \in \mathbb { R } ^ { C \times N ^ { \prime } \times M ^ { \prime } \times S } , N ^ { \prime } \mathit { \Theta } < N , M ^ { \prime } \mathit { \Theta } < M ,$ and C the number of latent channels. The in plane dimensions are compressed while volumetric depth is retained, allowing the diffusion model to exploit inter slice context at lower memory cost. After autoencoder pretraining, the encoder $\mathcal { E } _ { \psi }$ and decoder $\mathcal { D } _ { \omega }$ are fixed during diffusion training.

A variance preserving diffusion process [16] is applied to the target CT latent $\mathbf { z } _ { 0 }$ . Defining $\alpha _ { t } = 1 - \beta _ { t }$ and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } , } \end{array}$ the forward process is $q ( \mathbf { z } _ { t } \mid \mathbf { z } _ { 0 } ) = \mathcal { N } \left( \sqrt { \bar { \alpha } _ { t } } \mathbf { z } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } \right)$ , or equivalently,

$$
\mathbf { z } _ { t } = \sqrt { \bar { \alpha } _ { t } } \mathbf { z } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \qquad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) .\tag{8}
$$

The reverse process is parameterized by a conditional 3D UNet $\epsilon _ { \theta }$ that receives the noisy CT latent, the diffusion time step, and the CBCT latent condition, $\hat { \epsilon } = \epsilon _ { \theta } \left( { \bf z } _ { t } , t , { \bf z } _ { c } \right)$ . The corresponding estimate of the clean CT latent is

$$
\hat { \bf z } _ { 0 } = \frac { { \bf z } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { \theta } ( { \bf z } _ { t } , t , { \bf z } _ { c } ) } { \sqrt { \bar { \alpha } _ { t } } } ,\tag{9}
$$

which is decoded to the predicted synthetic CT, $\hat { \mathbf { x } } _ { 0 } = \mathcal { D } _ { \omega } \left( \hat { \mathbf { z } } _ { 0 } \right)$ The conditional denoising objective is

$$
\mathcal { L } _ { \mathrm { D D P M } } = \mathbb { E } _ { t , \mathbf { z } _ { 0 } , \epsilon } \left[ \Vert \epsilon - \epsilon _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { z } _ { c } ) \Vert _ { 2 } ^ { 2 } \right] .\tag{10}
$$

Importantly, the physics constraints introduced below are evaluated on $\hat { \mathbf { x } } _ { 0 }$ rather than directly on the noisy latent $\mathbf { z } _ { t } .$ Consequently, the acquisition model acts on a physical image domain estimate of the attenuation volume while the generative prior is learned efficiently in latent space.

## B. Geometry Derived Equivariance Projection Constraint

As established in Section II-B, a rotation $\mathbf { R } _ { \phi }$ of the attenuation volume about the CBCT scanner axis induces a corresponding angular translation $\mathbf { T } _ { \phi }$ in the projection domain. This property is inherited from the circular source detector trajectory and is therefore a deterministic property of the acquisition operator, rather than an image augmentation heuristic. To distinguish the geometry transformed reference target from the noisy CBCT measurements used by RawDC, we define the noise free projection of the reference CT under the CBCT geometry as $\bar { \mathbf { y } } _ { 0  c } : = \mathbf { A } _ { c } \mathbf { x } _ { 0 }$

For an admissible rotation $\phi _ { i }$ , the forward projection of the rotated prediction should agree with the corresponding angular translation of this reference projection

$$
{ \bf A } _ { c } { \bf R } _ { \phi _ { i } } \hat { \bf x } _ { 0 } \simeq { \bf T } _ { \phi _ { i } } \bar { \bf y } _ { 0 \to c } .\tag{11}
$$

The geometry equivariant projection loss is therefore

$$
\mathcal { L } _ { \mathrm { e q } } = \frac { 1 } { N _ { \phi } } \sum _ { i = 1 } ^ { N _ { \phi } } \left. \mathbf { A } _ { c } \mathbf { R } _ { \phi _ { i } } \hat { \mathbf { x } } _ { 0 } - \mathbf { T } _ { \phi _ { i } } \bar { \mathbf { y } } _ { 0 \to c } \right. _ { 2 } ^ { 2 } ,\tag{12}
$$

![](images/a64fea2e931dd6ccf485d6c6dffe9143875a6e94bef89ed01db74aeb5939378e.jpg)  
Fig. 1. Overview of the proposed EPC-RawDC framework. The reference CT $\mathbf { x } _ { 0 }$ and input $\mathbf { C B C T } \mathbf { x } _ { c }$ are encoded into volumetric latent representations, and conditional 3D diffusion produces the synthesized CT xˆ<sub>0</sub>. Geometry equivariance links an image domain rotation $\mathbf { R } _ { \phi }$ to the corresponding angular translation $\mathbf { T } _ { \phi }$ in projection space, while RawDC constrains $\hat { \mathbf { x } } _ { 0 }$ using the measured CBCT projections.

where $N _ { \phi }$ is the number of sampled rotations. To make the role of Eq. (12) explicit, define the discrete operator residual $\begin{array} { r } { \pmb { \delta } _ { \phi } ( \mathbf { x } ) = \mathbf { A } _ { c } \mathbf { R } _ { \phi } \mathbf { x } - \mathbf { T } _ { \phi } \mathbf { A } _ { c } \mathbf { x } . } \end{array}$ . Then

$$
\mathbf { A } _ { c } \mathbf { R } _ { \phi } \hat { \mathbf { x } } _ { 0 } - \mathbf { T } _ { \phi } \mathbf { A } _ { c } \mathbf { x } _ { 0 } = \mathbf { T } _ { \phi } \mathbf { A } _ { c } ( \hat { \mathbf { x } } _ { 0 } - \mathbf { x } _ { 0 } ) + \delta _ { \phi } ( \hat { \mathbf { x } } _ { 0 } ) .\tag{13}
$$

In the ideal continuous circular geometry, $\delta _ { \phi } \ = \ 0 ;$ if $\mathbf { T } _ { \phi }$ is norm preserving, Eq. (12) reduces to reference projection supervision $\| \mathbf { A } _ { c } ( \hat { \mathbf { x } } _ { 0 } - \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 }$ . Accordingly, we interpret $\mathcal { L } _ { \mathrm { e q } }$ as geometry transformed projection supervision derived from equivariance of the CBCT forward operator, rather than as a direct constraint on equivariance of the learned synthesis mapping. In the discrete implementation, finite angular and detector sampling together with spatial and angular interpolation give $\delta _ { \phi } \neq 0$ . Therefore, the transformed formulation is not exactly equivalent to unrotated projection matching. The notation $\mathcal { L } _ { \mathrm { e q } }$ is retained for continuity with EPC-3D-Diff.

## C. Raw Data Consistency Constraint

The equivariant term uses a transformed reference projection target. Therefore, the predicted sCT volume is not necessarily consistent with the raw noisy measurements observed for a particular patient. To introduce scan specific information, we additionally constrain $\hat { \mathbf { x } } _ { 0 }$ using the measured CBCT projection data $\mathbf { y } _ { c } .$ Following the model in Section II-A, the underlying photon counts obey Poisson statistics, while the post-log measurements are approximated by ${ \bf y } _ { c } = { \bf A } _ { c } { \bf x } + { \bf n } _ { c } , ~ { \bf n } _ { c } \sim$ $\mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { c } )$ . The corresponding general weighted least squares RawDC data fidelity term is

$$
{ \mathcal { L } } _ { \mathrm { R a w D C } } = { \frac { 1 } { M _ { c } } } \left\| \mathbf { W } _ { c } ^ { 1 / 2 } \left( \mathbf { y } _ { c } - \mathbf { A } _ { c } { \hat { \mathbf { x } } } _ { 0 } \right) \right\| _ { 2 } ^ { 2 } , \ \mathbf { W } _ { c } = \pmb { \Sigma } _ { c } ^ { - 1 }\tag{14}
$$

where $M _ { c }$ denotes the number of projection samples used in the loss. The normalization makes the magnitude of the loss less dependent on the number of views and detector elements.

RawDC therefore complements Eq. (12) by changing the projection domain target from the transformed noise free reference projection to the particular noisy realization $\mathbf { y } _ { c }$ . Both terms use the same scanner specific forward model, but the geometry derived term provides reference based projection supervision whereas RawDC introduces patient and scan specific measurement evidence.

## D. Overall Training Objective

In addition to the conditional diffusion objective, image domain regularization is retained from EPC-3D-Diff to preserve quantitative voxel accuracy and local anatomical structure. For the predicted clean CT xˆ<sub>0</sub>, we use

$$
\begin{array} { r l } & { \quad \mathcal { L } _ { 1 } = \left\| \hat { \mathbf { x } } _ { 0 } - \mathbf { x } _ { 0 } \right\| _ { 1 } , } \\ & { \quad \mathcal { L } _ { \mathrm { e d g e } } = \left\| \nabla \hat { \mathbf { x } } _ { 0 } - \nabla \mathbf { x } _ { 0 } \right\| _ { 1 } , \quad \mathcal { L } _ { \mathrm { l a p } } = \left\| \Delta \hat { \mathbf { x } } _ { 0 } - \Delta \mathbf { x } _ { 0 } \right\| _ { 1 } , } \end{array}\tag{15}
$$

where $\nabla$ and $\Delta$ denote the 3D finite difference gradient and Laplacian operators, respectively.

The complete EPC-RawDC training objective is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { D D P M } } + \lambda _ { 1 } \mathcal { L } _ { 1 } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { e d g e } } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { l a p } } } \\ & { ~ + \lambda _ { \mathrm { e q } } \mathcal { L } _ { \mathrm { e q } } + \lambda _ { \mathrm { R a w D C } } \mathcal { L } _ { \mathrm { R a w D C } } , } \end{array}\tag{16}
$$

where $\lambda _ { 1 } , ~ \lambda _ { 2 } , ~ \lambda _ { 3 } , ~ \lambda _ { \mathrm { e q } }$ , and $\lambda _ { \mathrm { R a w D C } }$ control the relative contribution of the corresponding constraints.

The terms in Eq. (16) act at complementary levels. L<sub>DDPM</sub> learns the conditional latent CT distribution, the image domain losses preserve voxel and structural fidelity, $\mathcal { L } _ { \mathrm { e q } }$ provides geometry transformed reference projection supervision derived from scanner equivariance, and $\mathcal { L } _ { \mathrm { R a w D C } }$ provides scan specific supervision from noisy projection measurements. Consequently, EPC-RawDC combines a learned 3D generative prior with scanner derived geometry and patient specific measurement information within a single training objective.

During inference, only the reconstructed CBCT volume $\mathbf { x } _ { c }$ is required as conditioning input. Sampling is performed in latent space and the final latent estimate is decoded to obtain $\begin{array} { r } { \hat { \mathbf { x } } _ { 0 } . } \end{array}$ The projection operators and raw measurements are used only as training supervision; therefore, projection agreement for a new patient is encouraged by the learned conditional model rather than explicitly enforced at inference.

## IV. EXPERIMENTAL DATA AND ACQUISITION GEOMETRY

The framework was evaluated on two independent head and neck cohorts SynthRAD2025 [17] and JUST with distinct clinical populations and CBCT geometries as in Table I. All volumes were resampled to a common grid of $5 0 \times 2 5 6 \times 2 5 6$ voxels with $1 \times 1 \times 3$ mm spacing. Image intensities were clipped to [−1000, 2000] HU and linearly mapped to [−1, 1]. Training and testing were separated at the patient level.

TABLE I  
SUMMARY OF THE CLINICAL DATASETS AND CBCT ACQUISITION MODELS USED IN THE STUDY.
<table><tr><td></td><td>SynthRAD2025</td><td>JUST</td></tr><tr><td>Source</td><td>Multicenter clinical CT</td><td>Clinical CT and CBCT</td></tr><tr><td>Patients</td><td>96 train / 24 test</td><td>32 train / 8 test</td></tr><tr><td>3D grid</td><td> $5 0 \times 2 5 6 \times 2 5 6$ </td><td> $5 0 \times 2 5 6 \times 2 5 6$ </td></tr><tr><td>Voxel spacing</td><td> $1 \times 1 \times 3 ~ \mathrm { m m }$ </td><td> $1 \times 1 \times 3 ~ \mathrm { m m }$ </td></tr><tr><td>CBCT geometry</td><td>Varian</td><td>Elekta</td></tr><tr><td>Detector</td><td> $7 6 8 \times 1 0 2 4$ </td><td> $5 1 2 \times 5 1 2$ </td></tr><tr><td>Projection data</td><td>Poisson  $I _ { 0 } = 1 0 ^ { 4 }$ </td><td>Clinical raw projections</td></tr></table>

## A. SynthRAD2025 Cohort

SynthRAD2025 provides multicenter clinical CT and CBCT data for synthetic CT generation in radiotherapy [17]. Because raw projection measurements are not distributed with the dataset, the reference CT volumes were used to generate a controlled CBCT acquisition domain using a Varian cone beam geometry. The scanner forward operator was implemented with the ASTRA Toolbox [18]. Projection measurements were generated from the CT attenuation volumes using Poisson photon statistics with an incident fluence of $I _ { 0 } = 1 0 ^ { 4 }$ photons per ray, followed by FDK reconstruction. The native simulated detector consisted of 768 × 1024 elements.

## B. JUST Clinical Cohort

The JUST cohort consists of paired clinical CT and CBCT examinations acquired at King Abdullah University Hospital, Jordan, with ethical approval Mar2026/191-38. In contrast to SynthRAD2025, the projection measurements used for the physics constraints were obtained from the clinical raw CBCT data. Therefore, the noise characteristics of the measured acquisition were retained. The corresponding Elekta acquisition model contains 220 projection views and a native $5 1 2 \times 5 1 2$ detector. The ASTRA Toolbox was used to implement the scanner specific forward operator employed by the geometry derived projection and RawDC constraints. The paired JUST CT and CBCT volumes were spatially aligned and resampled to the same grid to ensure voxelwise correspondence. Using a common image representation, while retaining the corresponding Varian and Elekta forward operators, allowed the same 3D diffusion architecture to be evaluated across both cohorts, without removing differences in scanner acquisition physics.

## V. RESULTS

We evaluated EPC-RawDC through four complementary experiments assessing: 1) component contributions under matched training and testing conditions; 2) generalization across independent clinical datasets and acquisition geometries; 3) joint multi domain training; and 4) volumetric 3D versus corresponding 2D modeling. Quantitative metrics were computed for each patient within the experiment. Images were clipped to [−1000, 2000] HU and normalized to [0, 1] for SSIM and PSNR, using a data range of 1, while MAE was reported in HU. One value per metric was obtained for each patient, and cohort results are reported as mean ± standard deviation.

## A. Implementation Details

The same network and optimization configuration was used for SynthRAD2025 and JUST, with the dataset and scanner specific geometry changed accordingly. Input volumes were encoded into 4 channel latent representations of size 4 × 50 × $1 2 8 \times 1 2 8 .$ , preserving the complete through plane dimension. The pretrained encoder and decoder, implemented with 4 latent, 32 hidden channels and 500 epochs, were fixed during diffusion training. The conditional 3D UNet used 64 base channels and a 256 dimensional sinusoidal time embedding. Diffusion was performed over 1000 time steps using a linear variance schedule from $\beta _ { 1 } = 1 0 ^ { - 4 }$ to $\beta _ { T } = 5 \times 1 0 ^ { - 3 }$ . Training used Adam with a learning rate of $5 \times 1 0 ^ { - 6 }$ and batch size one. The rotational projection constraint used one randomly signed rotation of $2 ^ { \circ } - 1 0 ^ { \circ }$ . Validation sampling used a 100 step DDIM procedure.

## B. Single Domain Performance and Component Ablation

We first evaluated all methods in the single domain setting, where training and testing were performed on the same clinical cohort using the same CBCT acquisition geometry. This experiment assesses reconstruction performance before introducing scanner or domain mismatch.

For the JUST cohort, the models were trained using the Elekta acquisition geometry and evaluated on eight independent patients that were not included during training. For SynthRAD2025, training and testing were performed using the corresponding acquisition setting, with 24 independent patients included in the final evaluation. In both cohorts, the data split was defined at the patient level to prevent information leakage between training and testing.

The ablation study included four diffusion based variants: DDPM [4], EPC-3D-Diff [11], RawDC and the complete EPC-RawDC model. The comparison models included a volumetric residual UNet (3D ResUNet), implemented as a 3D extension of residual UNet architectures previously investigated for CBCT based synthetic CT generation [19]. A volumetric CycleGAN was also implemented using 3D convolutional generators and discriminators, consistent with recent applications in medical image synthesis [20].

TABLE II
<table><tr><td rowspan="2">Data</td><td rowspan="2">Metrics</td><td rowspan="2">CBCT vs CT</td><td colspan="4">CT vs Synth CT (Ablation)</td><td colspan="2">CT vs Synth CT (Comparison)</td></tr><tr><td>DDPM</td><td>EPC-3D-Diff</td><td>RawDC</td><td>EPC-RawDC</td><td>3D ResUNet</td><td>3D CycleGAN</td></tr><tr><td rowspan="3">SynthRAD25</td><td>SSIM</td><td> $\overline { { 0 . 6 6 \pm 0 . 0 3 } }$ </td><td> $\overline { { 0 . 8 4 \pm 0 . 0 4 } }$ </td><td> $0 . 9 6 \pm 0 . 0 2$ </td><td> $\overline { { 0 . 9 6 \pm 0 . 0 2 } }$ </td><td> $\mathbf { 0 . 9 7 \pm 0 . 0 2 }$ </td><td> $\overline { { 0 . 9 1 \pm 0 . 0 2 } }$ </td><td> $\overline { { 0 . 9 5 \pm 0 . 0 2 } }$ </td></tr><tr><td>PSNR (dB)</td><td> $2 7 . 6 5 \pm 0 . 9 6$ </td><td> $3 6 . 5 2 \pm 1 . 1 8$ </td><td> $3 7 . 9 4 \pm 1 . 6 4$ </td><td> $3 9 . 1 2 \pm 1 . 7 2$ </td><td> $\mathbf { 3 9 . 1 8 \pm 1 . 8 2 }$ </td><td> $3 4 . 7 4 \pm 0 . 8 3$ </td><td> $3 5 . 9 1 \pm 1 . 5 9$ </td></tr><tr><td>MAE (HU)</td><td> $3 7 . 5 3 \pm 3 . 5 0$ </td><td> $1 0 . 8 8 \pm 2 . 6 5$ </td><td> $8 . 7 1 \pm 3 . 2 0$ </td><td> $8 . 7 0 \pm 2 . 8 0$ </td><td> ${ \bf 8 . 6 7 \pm 2 . 8 1 }$ </td><td> $1 3 . 8 1 \pm 1 . 5 9$ </td><td> $1 2 . 0 0 \pm 3 . 3 1$ </td></tr><tr><td rowspan="3">JUST</td><td>SSIM</td><td> $\overline { { 0 . 6 0 \pm 0 . 0 2 } }$ </td><td> $\overline { { 0 . 9 2 \pm 0 . 0 0 5 } }$ </td><td> $\overline { { 0 . 9 6 \pm 0 . 0 1 } }$ </td><td> $\overline { { 0 . 9 6 \pm 0 . 0 2 } }$ </td><td> $\mathbf { \overline { { 0 . 9 7 \pm 0 . 0 1 } } }$ </td><td> $\overline { { 0 . 9 5 \pm 0 . 0 1 } }$ </td><td> $\overline { { 0 . 9 5 \pm 0 . 0 1 } }$ </td></tr><tr><td>PSNR (dB)</td><td> $2 5 . 5 0 \pm 0 . 5 4$ </td><td> $3 5 . 9 0 \pm 1 . 0 9$ </td><td> $3 7 . 9 0 \pm 3 . 6 5$ </td><td> $3 7 . 9 9 \pm 3 . 7 8$ </td><td> ${ \bf 3 8 . 0 0 \pm 3 . 6 8 }$ </td><td> $3 5 . 5 9 \pm 1 . 0 3$ </td><td> $3 6 . 0 6 \pm 0 . 8 2$ </td></tr><tr><td>MAE (HU)</td><td> $4 9 . 2 6 \pm 3 . 7 3$ </td><td> $1 1 . 4 6 \pm 3 . 3 1$ </td><td> $9 . 8 3 \pm 2 . 8 0$ </td><td> $9 . 6 5 \pm 2 . 9 2$ </td><td> ${ \bf 9 . 2 8 \pm 2 . 9 1 }$ </td><td> $1 1 . 3 2 \pm 1 . 6 8$ </td><td> $1 1 . 1 2 \pm 1 . 6 5$ </td></tr></table>

AVERAGE AND STANDARD DEVIATION QUANTITATIVE COMPARISON AT PATIENT LEVEL: DDPM, EPC-3D-DIFF, RAWDC, EPC-RAWDC, 3D RESUNET, 3D CYCLEGAN, ON SINGLE DOMAIN SYNTHRAD2025 AND JUST TEST SETS FOR AXIAL ORIENTATION.

![](images/363143e1cab0f6cf084fdfe8f88b85335e06dce7acb55a6fb69a0641c9e14557.jpg)  
Fig. 2. Qualitative comparison after single domain training for the SynthRAD2025 and JUST clinical test sets using (a) the full HU range and (b) the soft tissue window. Columns show the reference CT (x ), CBCT (x ), synthesized CT (xˆ ) for the ablation and comparison methods, and the corresponding absolute error maps.

Table II summarizes the patient level quantitative results for both cohorts. All synthesis methods substantially improved the input CBCT, with the physics informed diffusion variants showing the strongest overall performance. EPC-3D-Diff and RawDC both improved upon DDPM, indicating that geometry derived reference projection supervision and scan specific projection supervision each contribute to reconstruction quality. Their combination in EPC-RawDC provided the best overall results in both datasets, reaching an SSIM of 0.97 in both cohorts and the highest PSNR of 39.18 dB for SynthRAD2025 and 38.00 dB for JUST. Compared with DDPM, EPC-RawDC reduced the mean MAE by approximately 20.3% for SynthRAD2025 and 19.0% for JUST.

The comparison models also improved substantially over the input CBCT, but their overall performance remained below EPC-RawDC. Across both datasets, EPC-RawDC achieved higher PSNR and lower MAE than 3D UNet and 3D Cycle-GAN, while also providing the highest SSIM. The consistent behaviour across SynthRAD2025 and JUST suggests that the benefit of combining equivariant projection supervision with RawDC is maintained across both acquisition settings.

These findings support the complementary use of the two projection domain terms. EPC-3D-Diff uses geometry transformed supervision from the reference CT, whereas RawDC uses the projection measurements of the acquisition. Their combination produced the best mean performance, although the patient level analysis below shows that some incremental differences relative to RawDC were not statistically significant.

Figure 2 presents representative axial examples from the JUST and SynthRAD2025 test cohorts. For each case, the reference CT, input CBCT, synthesized CT images, and absolute error maps are shown. The examples illustrate reconstruction behaviour in soft tissue, bone, air cavities, and regions affected by CBCT intensity artifacts. Images are displayed using both the full HU range [−1000, 2000] HU and the soft tissue range [−160, 240] HU. The full HU range provides an overall view of the reconstructed anatomy, whereas the soft tissue window enhances smaller intensity differences. MAE values computed within these windows are therefore not directly comparable, as the full HU range spans substantially larger intensity variations.

## C. Cross Dataset Generalization under Fixed Geometry

Cross dataset generalization was evaluated in both transfer directions while preserving the acquisition geometry used during training. The model trained on SynthRAD2025 with the Varian geometry was evaluated on 8 subjects from JUST dataset represented using the same Varian forward model, whereas the model trained on JUST with the Elekta geometry was evaluated on 24 SynthRAD2025 patients represented using the corresponding Elekta forward model. The results are reported in Tables III and IV.

For transfer from SynthRAD2025 to JUST, EPC-RawDC showed a clear improvement over DDPM, increasing SSIM by approximately 5.7%, increasing PSNR by 1.32 dB, and reducing MAE by 23.7%. Relative to RawDC, EPC-RawDC maintained the same SSIM at the reported precision, while improving PSNR by 0.47 dB and reducing MAE by approximately 5.6%. This indicates that the combined constraints remained beneficial when the model was applied to an independent clinical population.

The reverse transfer from JUST to SynthRAD2025 showed the same overall trend, although the improvement was smaller. Compared with DDPM, EPC-RawDC increased SSIM by approximately 5.6% and reduced MAE by 6.8%, while PSNR increased by 0.04 dB. Compared with RawDC, EPC-RawDC further improved SSIM by approximately 1.1% and reduced MAE by 2.5%. The consistent performance in both transfer directions suggests that the physics constrained framework improves robustness to changes in patient population while the CBCT forward geometry is maintained.

TABLE III  
CROSS DATASET GENERALIZATION FROM SYNTHRAD2025 TRAINING TO THE JUST TEST COHORT USING VARIAN GEOMETRY.
<table><tr><td>Method</td><td>SSIM ↑</td><td></td><td>PSNR (dB)↑ MAE (HU) ↓</td></tr><tr><td>CBCT vs CT</td><td> $\overline { { 0 . 6 3 \pm 0 . 0 2 } }$ </td><td> $2 7 . 8 8 \pm 0 . 6 6$ </td><td> $\overline { { 3 8 . 8 4 \pm 3 . 1 1 } }$ </td></tr><tr><td>DDPM</td><td> $0 . 8 7 \pm 0 . 0 1$ </td><td> $3 4 . 8 0 \pm 0 . 5 6$ </td><td> $1 6 . 7 0 \pm 0 . 8 8$ </td></tr><tr><td>RawDC</td><td> $0 . 9 2 \pm 0 . 0 2$ </td><td> $3 5 . 6 5 \pm 0 . 6 4$ </td><td> $1 3 . 5 1 \pm 1 . 2 7$ </td></tr><tr><td>EPC-RawDC</td><td> ${ \bf 0 . 9 2 \pm 0 . 0 2 }$ </td><td> ${ \bf 3 6 . 1 2 \pm 0 . 6 3 }$ </td><td> ${ \bf 1 2 . 7 5 \pm 1 . 3 1 }$ </td></tr></table>

TABLE IV  
CROSS DATASET GENERALIZATION FROM JUST TRAINING TO SYNTHRAD2025 TEST COHORT USING ELEKTA GEOMETRY.
<table><tr><td>Method</td><td>SSIM ↑</td><td>PSNR (dB) ↑</td><td>MAE (HU) ↓</td></tr><tr><td>CBCT vs CT</td><td> $\overline { { 0 . 5 7 \pm 0 . 0 6 } }$ </td><td> $\overline { { 2 5 . 7 4 \pm 0 . 8 2 } }$ </td><td> $\overline { { 4 7 . 8 9 \pm 7 . 1 6 } }$ </td></tr><tr><td>DDPM</td><td> $0 . 9 0 \pm 0 . 0 1$ </td><td> $3 3 . 9 8 \pm 1 . 3 9$ </td><td> $1 6 . 2 3 \pm 3 . 6 4$ </td></tr><tr><td>RawDC</td><td> $0 . 9 4 \pm 0 . 0 1$ </td><td> $3 3 . 9 8 \pm 1 . 4 8$ </td><td> $1 5 . 5 1 \pm 3 . 9 1$ </td></tr><tr><td>EPC-RawDC</td><td> $\mathbf { 0 . 9 5 \pm 0 . 0 1 }$ </td><td> $\mathbf { 3 4 . 0 2 \pm 1 . 5 1 }$ </td><td> ${ \bf 1 5 . 1 3 \pm 4 . 0 4 }$ </td></tr></table>

## D. Joint Multi Domain Training

To assess whether a single model could accommodate heterogeneous acquisition physics, EPC-RawDC was trained jointly using the SynthRAD2025 and JUST cohorts, with each sample retaining its corresponding scanner specific forward operator. The shared model was then evaluated independently on the held out test cohort from each domain.

As shown in Table V, EPC-RawDC consistently improved upon DDPM after joint training. On SynthRAD2025, EPC-RawDC increased SSIM by approximately 1.1%, improved PSNR by 0.49 dB, and reduced MAE by 6.5%. A similar trend was observed on JUST, where SSIM increased by approximately 1.1%, PSNR improved by 0.35 dB, and MAE decreased by 5.8%.

TABLE V  
PERFORMANCE OF THE JOINTLY TRAINED MODELS ON THE HELD OUT SYNTHRAD2025 AND JUST TEST COHORTS. THE MODELS WERE TRAINED USING BOTH DATASETS AND EVALUATED ON EACH DOMAIN.
<table><tr><td colspan="2">Method</td><td>SSIM ↑</td><td>PSNR (dB) ↑ MAE (HU) ↓</td><td></td></tr><tr><td rowspan="3">RA025</td><td>CBCT vs CT</td><td> $0 . 6 6 \pm 0 . 0 3$ </td><td> $2 7 . 6 5 \pm 0 . 9 6$ </td><td> $3 7 . 5 3 \pm 3 . 5 0$ </td></tr><tr><td>DDPM</td><td> $0 . 9 4 \pm 0 . 0 2$ </td><td> $3 4 . 0 8 \pm 1 . 0 3$ </td><td> $1 6 . 8 6 \pm 3 . 3 6$ </td></tr><tr><td>EPC-RawDC</td><td> ${ \bf 0 . 9 5 \pm 0 . 0 2 }$ </td><td> $\mathbf { 3 4 . 5 7 \pm 1 . 0 6 }$ </td><td> ${ \bf 1 5 . 7 6 \pm 3 . 1 6 }$ </td></tr><tr><td rowspan="4"></td><td>CBCT vs CT</td><td> $0 . 6 0 \pm 0 . 0 2$ </td><td> $2 5 . 5 0 \pm 0 . 5 4$ </td><td> $4 9 . 2 6 \pm 3 . 7 3$ </td></tr><tr><td>DDPM</td><td> $0 . 9 4 \pm 0 . 0 1$ </td><td> $3 4 . 1 9 \pm 1 . 1 9$ </td><td> $1 6 . 6 5 \pm 2 . 6 0$ </td></tr><tr><td>EPC-RawDC</td><td> $\mathbf { 0 . 9 5 \pm 0 . 0 1 }$ </td><td> $\mathbf { 3 4 . 5 4 \pm 1 . 4 1 }$ </td><td> ${ \bf 1 5 . 6 8 \pm 2 . 7 6 }$ </td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

The comparison with domain specific training in Table VI showed a clear advantage for specialization to a single acquisition domain. Relative to the corresponding single domain EPC-RawDC models, joint training reduced SSIM by approximately 2.1% in both cohorts and decreased PSNR by 4.61 dB on SynthRAD2025 and 3.46 dB on JUST. MAE increased by approximately 81.8% on SynthRAD2025 and 69.0% on JUST. These results indicate that although a single

EPC-RawDC model can accommodate heterogeneous scanner geometries, domain specific training provides greater structural similarity, contrast fidelity, and voxelwise intensity accuracy when sufficient scanner specific training data are available.

TABLE VI  
COMPARISON OF SINGLE DOMAIN AND JOINT DOMAIN EPC-RAWDCTRAINING ACROSS THE SYNTHRAD2025 AND JUST GEOMETRIES.
<table><tr><td>Test</td><td>Training</td><td>SSIM ↑</td><td>PSNR ↑</td><td>MAE ↓</td></tr><tr><td rowspan="2"></td><td>JUST + RAD2025</td><td> $0 . 9 5 \pm 0 . 0 2$ </td><td> $3 4 . 5 7 \pm 1 . 0 6$ </td><td> $1 5 . 7 6 \pm 3 . 1 6$ </td></tr><tr><td>RAD2025 only</td><td> $0 . 9 7 \pm 0 . 0 2$ </td><td> $3 9 . 1 8 \pm 1 . 8 2$ </td><td> $8 . 6 7 \pm 2 . 8 1$ </td></tr><tr><td rowspan="2">UV RA</td><td>JUST + RAD2025</td><td> $\overline { { 0 . 9 5 \pm 0 . 0 1 } }$ </td><td> $\overline { { 3 4 . 5 4 \pm 1 . 4 1 } }$ </td><td> $\overline { { 1 5 . 6 8 \pm 2 . 7 6 } }$ </td></tr><tr><td>JUST only</td><td> $0 . 9 7 \pm 0 . 0 1$ </td><td> $3 8 . 0 0 \pm 3 . 6 8$ </td><td> $9 . 2 8 \pm 2 . 9 1$ </td></tr></table>

## E. Volumetric 3D Versus Slice Wise 2D Modeling

The contribution of volumetric context was evaluated by comparing the proposed 3D framework with corresponding slice wise 2D synthesis methods. The 2D EqDiff model employs rotational equivariance directly in the image domain through a $C _ { 4 }$ group equivariant denoising network [10], whereas EPC-3D-Diff uses translated projection supervision derived from image rotational equivariance of the CBCT forward operator within a volumetric latent diffusion framework [11]. EPC-RawDC further incorporates patient specific projection consistency into the 3D formulation.

As summarized in Table VII, the coronal 2D versus 3D evaluation across all 24 SynthRAD2025 test subjects showed a clear advantage for volumetric 3D modeling.

Relative to 2D EqDiff, EPC-RawDC improved PSNR by 14.24 dB and reduced MAE by approximately 75.1%. Within the 3D physics constrained models, EPC-RawDC further improved PSNR by 1.32 dB and reduced MAE by 12.5% relative to EPC-3D-Diff, while reducing MAE by 32.9% relative to 3D DDPM.

The coronal examples in Fig. 3 further illustrate the effect of volumetric modeling on through plane consistency. The 2D approaches show stronger slice dependent intensity variation and discontinuity, particularly around the mandible, cervical spine, and surrounding soft tissue, whereas the 3D methods preserve more coherent anatomical structure across the coronal plane. EPC-RawDC provides the closest visual agreement with the reference CT in both the full HU and soft tissue displays. Across the displayed examples, EPC-RawDC achieved the highest PSNR and SSIM. Its error maps also showed the lowest MAE in every example compared with 2D and 3D DDPMs, with reduced residual errors around high contrast bone boundaries and soft tissue regions.

These findings indicate that the principal improvement arises from exploiting volumetric anatomical context, while RawDC provides a further gain beyond the geometry derived projection term alone. Such volumetric consistency is particularly relevant to radiotherapy applications requiring spatially coherent image volumes for registration, contour propagation, and dose related analysis.

TABLE VII  
PATIENT LEVEL COMPARISON OF 2D AND 3D CBCT TO CT SYNTHESIS ACROSS THE SYNTHRAD2025 TEST SUBJECTS UNDER THE CORONAL EVALUATION PROTOCOL.
<table><tr><td>Method</td><td>SSIM ↑</td><td>PSNR (dB) ↑</td><td>MAE (HU) ↓</td></tr><tr><td>CBCT vs CT</td><td> $\overline { { 0 . 6 6 \pm 0 . 0 4 } }$ </td><td> $\overline { { 2 6 . 9 1 \pm 0 . 9 1 } }$ </td><td> $\overline { { 4 3 . 0 2 \pm 6 . 5 0 } }$  </td></tr><tr><td>2D DDPM</td><td> $0 . 6 6 \pm 0 . 1 2$ </td><td> $2 1 . 8 1 \pm 1 2 . 6 9$ </td><td> $5 8 . 3 5 \pm 2 6 . 9 1$ </td></tr><tr><td>2D EqDiff</td><td> $0 . 7 3 \pm 0 . 0 5$ </td><td> $2 4 . 2 8 \pm 1 . 0 3$ </td><td> $4 1 . 2 7 \pm 7 . 5 7$ </td></tr><tr><td>2D CycleGAN</td><td> $0 . 7 8 \pm 0 . 0 5$ </td><td> $2 6 . 5 5 \pm 1 . 5 0$ </td><td> $3 5 . 4 4 \pm 9 . 5 0$ </td></tr><tr><td>3D DDPM</td><td> $\overline { { 0 . 8 9 \pm 0 . 0 2 } }$ </td><td> $\overline { { 3 6 . 3 7 \pm 1 . 1 9 } }$ </td><td> $\overline { { 1 5 . 2 8 \pm 1 . 9 2 } }$ </td></tr><tr><td>3D CycleGAN</td><td> $0 . 9 4 \pm 0 . 0 2$ </td><td> $3 5 . 1 4 \pm 1 . 3 7$ </td><td> $1 4 . 2 7 \pm 3 . 4 5$ </td></tr><tr><td>3D EPC-3D-Diff</td><td> $0 . 9 5 \pm 0 . 0 2$ </td><td> $3 7 . 2 0 \pm 1 . 3 7$ </td><td> $1 1 . 7 2 \pm 3 . 1 0$ </td></tr><tr><td>3D EPC-RawDC</td><td> ${ \bf 0 . 9 7 \pm 0 . 0 2 }$ </td><td> ${ \bf 3 8 . 5 2 \pm 1 . 5 5 }$ </td><td> ${ \bf 1 0 . 2 6 \pm 2 . 7 5 }$  1</td></tr></table>

## F. Physics Specific Consistency Analysis

Since the main contribution of EPC-RawDC is the incorporation of acquisition physics into training, we evaluated both projection domain physics terms directly in addition to conventional image quality metrics.

Projection consistency was assessed using the same scanner specific forward model employed by RawDC during training. For each generated CT volume, the image was converted to attenuation coefficients and forward projected at the native detector resolution using $\mathbf { A } _ { c } .$ . The projection consistency error was defined as $\begin{array} { r } { \mathcal { E } _ { \mathrm { p r o j } } = \bar { \frac { 1 } { M _ { c } } } \left\| \mathbf { y } _ { c } - \bar { \mathbf { A } } _ { c } \bar { \hat { \mathbf { x } } } _ { 0 } \right\| _ { 2 } ^ { 2 } . } \end{array}$ where $M _ { c }$ denotes the number of projection samples and lower values indicate greater agreement with the projection data.

As shown in Table VIII, DDPM produced a mean projection consistency error of $2 . 2 7 \times 1 0 ^ { - 3 }$ across the 24 SynthRAD2025 test cases. This decreased to $1 . 0 5 \times 1 0 ^ { - 3 }$ for EPC-3D-Diff and $1 . 0 2 \times 1 0 ^ { - 3 }$ for EPC-RawDC, corresponding to a 55.0% reduction relative to DDPM. The 3D CycleGAN achieved error of $1 . 1 1 \times 1 0 ^ { - 3 }$ . These results indicate that physics constrained models generate CT estimates that are more consistent with projection data than the unconstrained diffusion baseline.

TABLE VIII  
DIRECT EVALUATION OF THE PHYSICAL CONSISTENCY CONSTRAINTS ACROSS THE 24 SYNTHRAD2025 TEST CASES.
<table><tr><td>Method</td><td>Projection error  $( \mathbf { 1 0 } ^ { - 3 } ) \downarrow$ </td><td>Equivariance error  $( \mathbf { 1 0 } ^ { - 2 } ) \downarrow$ </td></tr><tr><td>DDPM</td><td> $\overline { { 2 . 2 7 \pm 0 . 3 4 } }$ </td><td> $\overline { { 7 . 1 8 \pm 1 . 6 9 } }$ </td></tr><tr><td>3D CycleGAN</td><td> $1 . 1 1 \pm 0 . 7 2$ </td><td> $7 . 3 8 \pm 1 . 5 9$ </td></tr><tr><td>EPC-3D-Diff</td><td> $1 . 0 5 \pm 0 . 7 1$ </td><td> $5 . 1 9 \pm 1 . 7 0$ </td></tr><tr><td> $\scriptstyle \mathrm { E P C - R a w D C }$ </td><td> ${ \bf 1 . 0 2 \pm 0 . 5 0 }$ </td><td> ${ \bf 4 . 2 1 \pm 1 . 6 7 }$ </td></tr></table>

The geometry derived term was evaluated using the transformed projection residual. For each generated CT, the volume was rotated by a prescribed angle ϕ and forward projected using the same scanner specific operator $\mathbf { A } _ { c } .$ The result was compared with the corresponding angular translation of the reference CT projection: $\begin{array} { r } { \bar { \mathcal { E } } _ { \mathrm { g e o } } = \frac { 1 } { N _ { \wedge } } \sum _ { i = 1 } ^ { N _ { \phi } } \big \| \mathbf { A } _ { c } \mathbf { R } _ { \phi _ { i } } \hat { \mathbf { x } } _ { 0 } - \mathbf { T } _ { \phi _ { i } } \bar { \mathbf { y } } _ { 0 \to c } \big \| _ { 2 } ^ { 2 } , } \end{array}$ where $\bar { \bf y } _ { 0  c } = { \bf A } _ { c } { \bf x } _ { 0 }$ . This metric quantifies agreement with the geometry transformed reference projection and is not a direct measurement of equivariance of the learned synthesis mapping. DDPM produced a mean geometry transformed projection error of $7 . 1 8 \times 1 0 ^ { - 2 }$ . EPC-3D-Diff reduced this value to $5 . 1 9 \times 1 0 ^ { - 2 }$ corresponding to a 27.7% reduction, while EPC-RawDC further reduced it $\mathrm { t o ~ 4 . 2 1 \times 1 0 ^ { - 2 } }$ , corresponding to reductions of 41.4% relative to DDPM and 18.9% relative to EPC-3D-Diff. The 3D

![](images/cd093498d1ac28e414807af02c958f72fba7d47c52b49b680953be26d3ed582d.jpg)  
Fig. 3. Coronal comparison of 2D and 3D CBCT to CT synthesis on representative SynthRAD2025 slices using full HU and soft tissue window. Columns show the reference CT, CBCT, 2D DDPM, 2D EqDiff, 2D CycleGAN, 3D DDPM, EPC-3D-Diff, 3D CycleGAN, and EPC-RawDC, and the absolute HU error for 2D DDPM, 3D DDPM, and EPC-RawDC. PSNR and SSIM are reported for the synthesis images and MAE for the error maps.

![](images/0d63dc6f9151ddfb8904767999d6c85978913cb454efb5d295bc57ffa11f7571.jpg)  
Fig. 4. Patient level distributions of SSIM, PSNR, and MAE for (a) SynthRAD2025 and (b) JUST. DDPM, EPC-3D-Diff, RawDC, and EPC-RawDC are compared using matched patient level measurements. Boxes indicate the median and interquartile range, whiskers extend to 1.5 times the interquartile range, and individual points denote test patients. The annotation “EPC-RawDC vs DDPM” indicates the number of patients for which EPC-RawDC outperformed DDPM for the corresponding metric.

CycleGAN produced $7 . 3 8 \times 1 0 ^ { - 2 }$ . Together, the two metrics assess complementary forms of projection domain consistency.

## G. Patient Level Statistical Analysis

Paired patient level comparisons used matched measurements from each test subject. Normality of paired differences was assessed using Shapiro Wilk test, followed by a two sided paired t test for normal differences or a two sided Wilcoxon signed rank test otherwise. Within each cohort, Holm Bonferroni correction was applied across 9 prespecified comparisons: EPC RawDC vs DDPM, RawDC, and EPC-3D-Diff for SSIM, PSNR and MAE. Significance was defined as $p _ { \mathrm { a d j } } < 0 . 0 5$

As shown in Fig. 4, EPC-RawDC consistently improved upon DDPM across both cohorts. On SynthRAD2025, EPC-RawDC outperformed DDPM in all 24 patients for SSIM, PSNR, and MAE, with all comparisons remaining significant after correction $( p _ { \mathrm { a d j } } < 0 . 0 0 1 )$ . EPC-RawDC also showed sig nificant improvements over EPC-3D-Diff for all three metrics $( p _ { \mathrm { a d j } } < 0 . 0 0 1 )$ . Relative to RawDC, the improvement remained significant for SSIM $( p _ { \mathrm { a d j } } = 0 . 0 0 9 )$ , whereas the differences in PSNR and MAE were not significant $( p _ { \mathrm { a d j } } = 0 . 2 6 1$ and $p _ { \mathrm { a d j } } = 0 . 5 8 4$ , respectively).

The JUST cohort showed a similar but more conservative pattern. EPC-RawDC outperformed DDPM in all 8 patients for all three metrics; however, after correction, the improvement was significant for SSIM $( p _ { \mathrm { a d j } } < 0 . 0 0 1 )$ and MAE $( p _ { \mathrm { a d j } } =$ 0.002), but not for PSNR $( p _ { \mathrm { a d j } } = 0 . 0 7 3 )$ . Compared with the physics constrained alternatives, EPC-RawDC achieved significantly higher SSIM than both EPC-3D-Diff and RawDC (both $p _ { \mathrm { a d j } } < 0 . 0 0 1 $ ) and significantly lower MAE than both methods (both $p _ { \mathrm { a d j } } = 0 . 0 3 9 )$ ), while the corresponding PSNR differences were not significant $( p _ { \mathrm { a d j } } = 0 . 0 7 3$ versus EPC-3D-Diff and $p _ { \mathrm { a d j } } = 0 . 2 7 5$ versus RawDC).

Overall, the patient level analysis confirms that the improvement over the unconstrained DDPM baseline is highly consistent across subjects. The smaller differences relative to EPC-3D-Diff and RawDC indicate that the incremental benefit of combining equivariant projection supervision with raw data consistency is most evident in structural similarity and, for the JUST cohort, in absolute intensity accuracy.

## VI. CONCLUSION

This work introduced EPC-RawDC, a conditional 3D latent diffusion framework that integrates equivariance derived projection supervision with raw data consistency through a common acquisition CBCT forward operator. Across SynthRAD2025 and JUST, EPC-RawDC provided the best mean performance among the evaluated diffusion and comparison methods, reducing MAE relative to DDPM by 20.3% and 19.0%, respectively. Direct physics evaluation further demonstrated improved agreement with the projection measurements and geometry transformed reference projections. Cross dataset, joint domain, and volumetric experiments further characterized the generalization and multi domain behaviour of the framework, while domain specific training remained advantageous for quantitative accuracy. These findings support measurement grounded diffusion as a promising approach for quantitative CBCT to CT synthesis. Future work will investigate broader multicenter validation and the impact of the synthesized CT on downstream radiotherapy tasks, including dose calculation and adaptive treatment workflows.

## ACKNOWLEDGEMENT

The authors thank Prof. Haytham Alewaidat, Prof. Khaled Alawneh, Eng. Ahmad Qandeel for help to collect the clinical JUST data and Mr. Mohammad Alhyari for the manuscript review.

## REFERENCES

[1] E. M. C. Huijben, M. L. Terpstra, A. J. Galapon et al., “Generating syn thetic computed tomography for radiotherapy: SynthRAD2023 challenge report,” Medical Image Analysis, vol. 97, p. 103276, 2024.

[2] X. Liang, L. Chen, D. Nguyen, Z. Zhou, X. Gu, M. Yang, J. Wang, and S. Jiang, “Generating synthesized computed tomography (CT) from cone-beam computed tomography (CBCT) using CycleGAN for adaptive radiation therapy,” Physics in Medicine and Biology, vol. 64, 2019.

[3] B. Rusanov, G. M. Hassan, M. Reynolds, M. Sabet, P. Rowshanfarzad, N. Bucknell, S. Gill, J. Dass, and M. Ebert, “Transformer CycleGAN with uncertainty estimation for CBCT based synthetic CT in adaptive radiotherapy,” Physics in Medicine and Biology, vol. 69, no. 3, 2024.

[4] J. Peng, R. L. J. Qiu, J. F. Wynne, C.-W. Chang, S. Pan, T. Wang, J. Roper, T. Liu, P. R. Patel, D. S. Yu, and X. Yang, “CBCT-based synthetic CT image generation using conditional denoising diffusion probabilistic model,” Medical Physics, vol. 51, pp. 1847–1859, 2024.

[5] A. Altalib, C. Li, and A. Perelli, “Conditional diffusion models for CT image synthesis from CBCT: A systematic review,” Tomography, vol. 12, no. 5, p. 64, 2026.

[6] M. Al-Shalabi, M. A. Mahdi, R. Elbarougy, E. T. Alnfrawy, M. U. Hadi, and R. F. Ali, “Cone-beam CT to synthetic CT translation using conditional 3d latent diffusion-based model,” IEEE Access, vol. 14, pp. 12 680–12 693, 2026.

[7] H. Xie, H. Chen, Q. Li, Z. Chen, Y. Sun, and T. Tan, “Conditional diffusion models for high-fidelity synthetic CT generation from CBCT in nasopharyngeal carcinoma adaptive radiotherapy: a multi-center study,” Physics in Medicine and Biology, vol. 71, no. 18, p. 185013, 2026.

[8] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 684–10 695.

[9] D. Chen, J. Tachella, and M. E. Davies, “Equivariant imaging: Learning beyond the range space,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 4379–4388.

[10] A. Altalib, C. Li, and A. Perelli, “Equivariant conditional diffusion model for head and neck CT image synthesis from CBCT,” Medical Physics, vol. 53, no. 8, p. e70617, 2026.

[11] A. Altalib, C. Li, H. Al Ewaidat, K. Alawneh, A. Qendel, and A. Perelli, “EPC-3D-Diff: Equivariant physics consistent conditional 3D latent diffusion for CBCT to CT synthesis,” MICCAI Proceedings, 2026.

[12] B. Zhou, S. K. Zhou, J. S. Duncan, and C. Liu, “Limited view tomographic reconstruction using a cascaded residual dense spatialchannel attention network with projection data fidelity layer,” IEEE Transactions on Medical Imaging, vol. 40, no. 7, pp. 1792–1804, 2021.

[13] H. Chen, Z. Hao, L. Guo, and L. Xiao, “Mitigating data consistency induced discrepancy in cascaded diffusion models for sparse-view CT reconstruction,” IEEE Transactions on Medical Imaging, vol. 44, no. 7, pp. 3012–3024, 2025.

[14] J. Nuyts, B. De Man, J. A. Fessler, W. Zbijewski, and F. J. Beekman, “Modelling the physics in iterative reconstruction for transmission computed tomography,” Physics in Medicine and Biology, vol. 58, 2013.

[15] J. Wang, T. Li, H. Lu, and Z. Liang, “Penalized weighted least-squares approach to sinogram noise reduction and image reconstruction for low-dose x-ray computed tomography,” IEEE Transactions on Medical Imaging, vol. 25, no. 10, pp. 1272–1283, 2006.

[16] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, 2020.

[17] A. Thummerer, E. van der Bijl, A. J. Galapon, F. Kamp, M. Savenije, C. Muijs, S. Aluwini, R. J. H. M. Steenbakkers, S. Beuel, M. P. W. Intven, J. A. Langendijk, S. Both, S. Corradini, V. Rogowski, M. Terpstra, N. Wahl, C. Kurz, G. Landry, and M. Maspero, “Synthrad2025 grand challenge dataset: Generating synthetic CTs for radiotherapy from head to abdomen,” Medical Physics, vol. 52, no. 7, p. e17981, 2025.

[18] W. Van Aarle, W. J. Palenstijn, J. Cant, E. Janssens, F. Bleichrodt, A. Dabravolski, J. De Beenhouwer, K. Joost Batenburg, and J. Sijbers, “Fast and flexible X-ray tomography using the ASTRA toolbox,” Optics express, vol. 24, no. 22, pp. 25 129–25 147, 2016.

[19] M. C. Caglar, K. S. Ertas, M. S. Cebe, I. Kara, N. Kheradmand, and E. Metcalfe, “Synthetic CT generation from CBCT using deep learning for adaptive radiotherapy in prostate cancer,” Front. Radiol., vol. 5, 2025.

[20] J. M. Molina-Maza, D. Viar-Hernandez, B. Rodriguez-Gonzalez, J. Sequeiro-Gonzalez, J. A. Vera-Sanchez, A. Mazal, N. Malpica, A. Torrado-Carvajal, J. M. Perez-Moreno, and B. Rodriguez-Vila, “Enhancing CBCT-based CT synthesis using planning MRI in adaptive proton therapy for head cancer: A deep learning approach,” J. Appl. Clin. Med. Phys., vol. 26, no. 12, p. e70367, 2025.