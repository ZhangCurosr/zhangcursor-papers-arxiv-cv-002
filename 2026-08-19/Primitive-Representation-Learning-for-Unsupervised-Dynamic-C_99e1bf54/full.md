# Primitive Representation Learning for Unsupervised Dynamic Contrast Enhanced MRI Reconstruction

Veronika Spieker<sup>1,2,3</sup>, Wenqi Huang<sup>2</sup>, Cemre Ariyurek<sup>3</sup>, Liam Timms<sup>3</sup>, Daniel Rueckert<sup>2,4,5</sup>, Onur Afacan<sup>3</sup>, Julia A. Schnabel<sup>1,2,4,6</sup>, and Sila Kurugol<sup>3</sup>

1 Institute of Machine Learning in Biomedical Imaging, Helmholtz Munich, Germany 2 School of Computation and Information Technology, Technical University of Munich, Germany

<sup>3</sup> Department of Radiology, Boston Children’s Hospital and Harvard Medical School, USA

4 Munich Center for Machine Learning, Technical University of Munich, Germany 5 Department of Computing, Imperial College London, United Kingdom   
6 School of Biomedical Engineering and Imaging Sciences, King’s College London, United Kingdom y.spieker@tum.de

Abstract. Reliable quantitative analysis of dynamic contrast-enhanced MRI requires high-quality spatiotemporal reconstructions at high undersampling rates. Scan-specific reconstructions using Gaussian and Gabor primitives have shown promising results without the need for large training datasets, but have not addressed the additional dimension of dynamic contrast. We propose a multi-dimensional, primitive based framework for dynamic contrast-enhanced MRI reconstruction that disentangles the underlying anatomy, the dynamic contrast enhancement, and residual motion into separate temporal basis functions, thereby enabling a geometrical interpretation of the representation. We show that this architecture achieves performance competitive with conventional reconstruction methods, both in reconstruction quality and in the accuracy of extracted aorta and kidney enhancement curves. The modular tier design extends naturally to additional dynamic factors and higher acceleration rates. Code available at https://github.com/compai-lab/ 2026-GaborDCE-spieker.

Keywords: Representation Learning · MRI Reconstruction · Gaussian Splatting · Gabor Primitives · Unsupervised Learning · MR Multitasking.

## 1 Introduction

Magnetic resonance imaging (MRI) ofers high diagnostic value, but its long acquisition times remain a fundamental limitation. Therefore, acceleration strategies are widely employed, although they often involve a trade-of between spatial resolution, temporal resolution and temporal signal fidelity. This trade-of is especially consequential in dynamic contrast-enhanced (DCE) MRI, where preserving the peak and shape of the arterial input function (AIF) is essential for reliable quantitative parameter estimation, e.g., in MR urography for renal function assessment [6, 7, 14, 15]. Standard clinical reconstructions such as GRASP and XD-GRASP [5] prioritize diagnostic image quality, often over-smoothing temporal dynamics and degrading downstream parameter estimation.

Recently, unsupervised deep learning methods have shown promising results for spatiotemporal reconstruction [8, 16]. Rather than relying on large training datasets, these approaches fit continuous representations on a scan-specific basis. This is particularly advantageous for abdominal MRI, where such datasets are dificult to obtain, as acquiring motion-free, fully sampled reference data is especially challenging. One widely applied architecture for spatiotemporal reconstruction is based on implicit neural representations (INRs), which use an MLP to encode intensity values at arbitrary spatial locations (pixels in 2D or voxels in 3D) [2, 4, 9, 17].

However, as purely coordinate-based functions optimized without an explicit spatial prior, INRs are prone to sub-resolution overfitting and ofer limited interpretability. Primitive-based methods instead represent the image with an explicit set of kernels whose shape prior removes this failure mode. More recently, a novel representation based on Gaussian primitives has been proposed. Inspired by Gaussian splatting, the image space is parametrized through the properties of multiple Gaussian kernels, namely their weights and geometric parameters, which define their spatial extent and contribution. The final image is rendered by summing all primitives at the desired image coordinates [20]. Initial applications of primitive representations to MR reconstruction have covered MR slice-tovolume reconstruction (SVR) [3] and dynamic cardiac MR reconstruction [10, 18]. Each primitive has an explicit position, scale and frequency, making the representation directly inspectable, and its reduced parameters train faster than an INR MLP [10].

Primitive-based MR reconstruction performance was further improved by modulating each Gaussian primitive with a frequency, yielding Gabor primitives. This approach was applied to cardiac cine imaging, leveraging temporal redundancies through low-rank modeling of the temporal dimension. However, such modeling is not suficient to also capture contrast enhancement dynamics, which can be considered a decoupled dynamic process. Unlike motion, contrast enhancement is neither smooth nor periodic, but instead requires flexible signal modeling that can accommodate sharp transitions, such as those occurring upon bolus arrival.

We propose to adapt the Gabor primitive idea to DCE-MRI, leveraging the inherent temporal redundancy by modeling the dynamic behavior of the primitives. To this end, we extend the motion-aware reconstruction framework with an additional DCE signal tier, which models the rapidly arriving high-intensity changes on top of the base signal representing the general anatomy. To account for these distinct dynamics, each tier is modeled with its own temporal basis, and the temporal regularization is adapted so that the sharp contrast wash-in is preserved rather than smoothed away. Overall, our contributions are three-fold: 1. we propose the first primitive-based reconstruction method for DCE-MRI;

2. we present a novel multi-dimensional design and training strategy for a geometrically interpretable, unsupervised primitive-based representation;

3. we evaluate on in-vivo pediatric abdominal data, covering both reconstruction quality and downstream quantitative parameters as well as geometrical analysis of the learned representation.

## 2 Methods

## 2.1 Primitive Modeling for MR Reconstruction

As introduced in [10], we model the dynamic MR image across T frames as a sum of N Gabor primitives whose weights and geometry may vary across frames:

$$
x _ { t } ( \pmb { r } ) = \sum _ { n = 1 } ^ { N } w _ { n , t } P _ { n } \big ( \pmb { r } ; \mu _ { n , t } , \pmb { s } _ { n , t } , \theta _ { n , t } , \pmb { \xi } _ { n , t } \big ) , \quad t = 1 , \dots , T ,\tag{1}
$$

where $w _ { n , t } ~ \in ~ \mathbb { C }$ is the complex weight and $P _ { n } ( r ; \mu , s , \theta , \xi )$ are anisotropic Gabor primitives with its corresponding geometrical parameters $\textstyle \mu _ { n , t } ,$ scale $s _ { n , t } .$ rotation $\theta _ { n , t }$ and Gabor frequency $\xi _ { n , t }$ (see [10] for detailed definition).

## 2.2 Spatiotemporal Forward Model

Originally, temporal variation is modeled through two low-rank temporal bases: a contrast basis modeling signal-intensity variations, and a geometry basis capturing cardiac motion [10]. In the following, we expand the model to account for dynamic contrast signal enhancement. In DCE-MRI, the injected contrast produces a sharp enhancement at time of injection, followed by a slow decay. This temporal behavior is diferent from the general (non-contrast) anatomy changes, e.g., due to motion, which motivates modeling it as a separate tier. At the same time, the signal enhancement is spatially constrained by the underlying anatomical regions. Assuming that a well-fitted primitive model captures this structure geometrically, contrast enhancement would be expected to manifest simply as an increase in the intensity weight of the corresponding primitives.

Contrast Basis. Therefore, we model the complex signal intensity weight $w _ { n , t }$ as a composition of three tiers (see Fig. 1A): A base anatomy signal representing the pre-contrast anatomy, a motion-coupled signal accounting for motioninduced intensity changes, and a dedicated dynamic contrast signal capturing the enhancement and its subsequent decay. As each tier exhibits a distinct temporal behavior, all are modeled with their own temporal basis, resulting in the following summation for the overall intensity weight $w _ { n , t } \colon$

$$
w _ { n , t } = \underbrace { \pmb { u } _ { n } ^ { [ b ] } \cdot \pmb { v } _ { b , t } } _ { \mathrm { b a s e ~ t i e r } } + \underbrace { \pmb { u } _ { n } ^ { [ g ] } \cdot \pmb { v } _ { g , t } } _ { \mathrm { m o t i o n ~ t i e r } } + \underbrace { \pmb { u } _ { n } ^ { [ c ] } \cdot \pmb { v } _ { c , t } } _ { \mathrm { D C E ~ t i e r } } \mathrm { ~ , ~ }\tag{2}
$$

![](images/1e1d4c5da3652bccfa20528fdf9edecc162563ee72e44ac5f3a6d0e4992ba51b.jpg)  
Fig. 1: Method Overview. (A) Spatiotemporal DCE Primitive Representation consists of a contrast basis and geometry basis, modeling the primitives intensity and the primitives shape/position, respectively. (B) The weighted sum of all rendered primitives results in the DCE reconstruction time series, which is transformed into k-space to compute the unsupervised loss to the acquired data.

where $\pmb { v } _ { b , t } \in \mathbb { C } ^ { R _ { b } } , \pmb { v } _ { g , t } \in \mathbb { C } ^ { R _ { g } }$ and ${ \pmb v } _ { c , t } \in \mathbb { R } ^ { R _ { c } }$ are the t-th rows of the shared base, motion and contrast bases $\mathbf { V } _ { b } \in \mathbb { C } ^ { T \times R _ { b } } , \mathbf { V } _ { g } \in \mathbb { C } ^ { T \times R _ { g } }$ and $\mathbf { V } _ { c } \in \mathbb { C } ^ { T \times R _ { c } }$ and $u _ { n } ^ { [ b ] } , u _ { n } ^ { [ g ] }$ and $\mathbf { \Delta } _ { \mathbf { \Delta } u _ { n } ^ { [ c ] } }$ are per-primitive coeficient vectors.

Geometry Basis. The parameters defining the geometry of the primitives $( \mu _ { n , t } ,$ scale $s _ { n , t }$ , rotation $\theta _ { n , i }$ and Gabor frequency $\xi _ { n , t } )$ are each composed of a static base component and a temporally varying component modeling motion:

$$
\begin{array} { r l r } & { } & { \pmb { \mu } _ { n , t } = \pmb { \mu } _ { n } + \mathbf { C } _ { n } ^ { [ \mu ] } \pmb { v } _ { g , t } , \qquad \theta _ { n , t } = \theta _ { n } + \pmb { C } _ { n } ^ { [ \theta ] } \cdot \pmb { v } _ { g , t } , } \\ & { } & { \log \pmb { s } _ { n , t } = \log \pmb { s } _ { n } + \mathbf { C } _ { n } ^ { [ s ] } \pmb { v } _ { g , t } , \quad \pmb { \xi } _ { n , t } = \pmb { \xi } _ { n } + \mathbf { C } _ { n } ^ { [ \xi ] } \pmb { v } _ { g , t } , } \end{array}\tag{3}
$$

where log denotes the element-wise logarithm and $\mathbf { C } _ { n } ^ { [ \cdot ] }$ are per-primitive coeficient matrices for the motion track. Reusing the geometric temporal base $v _ { g , i }$ ties the intensity enhancement and the geometric deformation of an enhancing primitive to a single time course.

## 2.3 Multi-dimensional training strategy

The primitives are fit by minimizing a density-compensated k-space objective,

$$
\mathcal { L } = \sum _ { c , t } \left\| \sqrt { d } \odot \left( A _ { t } ( S _ { c } \odot x _ { t } ) - y _ { c , t } \right) \right\| _ { 2 } ^ { 2 } + \sum _ { i \in \{ b , g , c \} } \lambda _ { i } \mathcal { R } _ { i } ( \mathbf { V } _ { i } ) + \lambda _ { x } \sum _ { t } \left\| x _ { t + 1 } - x _ { t } \right\| _ { 1 } ,\tag{4}
$$

where $y _ { c , t }$ are the measured radial samples of coil c at frame $t ,$ d the densitycompensation weights, $\boldsymbol { A } _ { t }$ the per-frame encoding operator, $S _ { c }$ the coil sensitivity, and $x _ { t }$ the reconstructed frame. The regularizer $\mathcal { R } _ { i }$ is a temporal regularization penalty applied to each tier’s temporal basis $\mathbf { V } _ { i } ,$ with $i \in \{ b , g , c \}$ indexing the base, geometry, and contrast tiers (detailed below). The final term is an image-domain temporal total variation, an $\ell _ { 1 }$ penalty on the frame-to-frame diference of the rendered images $x _ { t }$ , weighted by $\lambda _ { x }$

Temporal regularization. We penalize each basis’s frame-to-frame diferences with a Huber kernel $H _ { \delta }$ to suppress temporal noise while preserving the sharp bolus arrival: diferences below δ are treated quadratically (damped more strongly than $\ell _ { 1 } )$ , while the large bolus jump stays in the linear $( \ell _ { 1 } )$ regime and is left un-smoothed. For each tier $i \in \{ b , g , c \}$ we apply ${ \mathcal R } _ { i } ( { \mathbf V } _ { i } ) = \bar { V } _ { i } ^ { - 1 } \langle H _ { \delta } ( \varDelta _ { t } { \mathbf V } _ { i } ) \rangle$ 2 where $\bar { V } _ { i } = \langle | \mathbf { V } _ { i } | \rangle$ normalizes by each tier’s scale so one δ yields equal relative smoothing across tiers of difering magnitude. All three bases share this penalty; only the weights difer, with the base and geometry bases weighted more strongly than the contrast basis $\left( \lambda _ { c } , \lambda _ { g } \ll \lambda _ { b } \right)$ to keep the enhancement sharp.

Staged optimization. Optimizing the three tiers of Eq. (2) jointly from scratch is unstable because the high-capacity base tier tends to absorb the contrast enhancement that should be captured by the dedicated contrast tier, collapsing the intended separation. We therefore adopt a staged schedule in the following order: base/pre-contrast → DCE contrast → motion-geometry. First, the static base tier $\mathbf { V } _ { s }$ is fit on the pre-contrast frames with the other tiers frozen, so it commits to the pre-enhancement anatomy. The contrast tier is then unfrozen over all $T$ frames, forcing the bolus wash-in into $\mathbf { V } _ { c } ,$ and finally the motioncoupled geometry $\mathbf { V } _ { g }$ is released to absorb residual deformation. This sequential unfreezing keeps the anatomy, contrast, and motion tiers from competing for the same signal while the separation is most fragile.

## 3 Experimental Setup

## 3.1 Data

We evaluate on five in-house pediatric MRI acquisitions using a golden angle stack-of-stars 3D FLASH prototype sequence with a multichannel body-matrix coil (3T Siemens VIDA/PRISMA) to acquire 32 coronal slices with a voxel size of $1 . 2 5 \times 1 . 2 5 \times 3 \mathrm { { m m ^ { 3 } } }$ ， $\mathrm { T R } / \mathrm { T E } / \mathrm { F A } = 3 . 5 6 / 1 . 3 9 \mathrm { m s } / 1 2 ^ { \circ }$ . The acquired spokes were retrospectively binned to 34 spokes per frame, resulting in a temporal resolution of ∼4.2-4.8 seconds per volume. For an image size of 448×448 pixels, this corresponds to an undersampling factor of $R \approx 2 1$ relative to the Nyquist criterion. Reconstructions were performed on all slices for the complete field-ofview and then cropped to a 224×224 region of interest. To reduce computational overhead, coil compression was applied to retain 80% of the coil information, and coil sensitivities were estimated via ESPIRiT [19]. Aorta/kidney segmentations were obtained with nnU-Net [11] and verified manually.

## 3.2 Evaluation Methods

Methods are qualitatively compared on representative aorta and kidney slices, both spatially and spatiotemporally. Since no ground truth is available, we compute PSNR and SSIM against GRASP as an indicator of reconstruction similarity. To assess the clinical potential for urography, including tracer kinetic modeling [15], we further analyze the temporal behavior of the aorta and kidney in the spatiotemporal reconstruction. To this end, we reconstruct the whole volume and extract the aorta and kidney signal intensities using segmentation masks, retaining the top 20% of signal intensities within each mask. Additionally, we quantify the temporal sharpness relevant for AIF estimation by computing the peak-to-baseline ratio (P/B), the wash-in slope and the temporal variation after contrast arrival $( T V _ { p o s t } )$ . Comparison methods include the current clinical standard GRASP [5] with low temporal regularization used for accurate kidney function quantification, L+S [13] as a conventional decomposition-based method, and Hash-INR as an alternative scan-specific learning-based method, based on an implicit neural representation with multi-resolution hash-grid encoding [12]. Also, a Gaussian primitive representation $( \xi _ { n } = 0 )$ is computed.

![](images/a241a0e063e4d315aa97540d2e8293e71afb9dbfe86c970ee220b3c7445f5f84.jpg)  
Fig. 2: Qualitative spatial (xy) and spatiotemporal (xt) reconstruction results for an exemplary subject (kidneys/aorta on the top/bottom, respectively). While INR results in competitive but slightly noisy reconstructions compared to the conventional references, it blurs spatiotemporally (arrows). Gabor sharpens reconstructions compared to Gaussian and maintains temporal sharpness.

## 3.3 Implementation Details

All methods were implemented in PyTorch on an NVIDIA RTX A5000 GPU. Custom CUDA kernels were used for primitive rasterization. Our reconstruction is initialized with 15000 Gabor primitives on a regular grid and adaptively densified up to 20000 during optimization (19000 and 25000 for Gaussian primitives to match parameter count, respectively). Adaptive density control prunes low-contribution primitives and splits/clones high-gradient ones every 300 iterations, active between 25% and 50% of training. The base, contrast and geometry tiers use temporal ranks $R _ { b } { = } 3 , R _ { c } { = } 4$ and $R _ { g } { = } 1$ . All parameters are optimized with Adam for 5000 iterations under a cosine learning-rate schedule, with pergroup rates. Optimization begins with 500 pre-contrast iterations on the static tier alone, followed by 500 iterations with the DCE tier unfrozen, before the full model is optimized from iteration 1000 onward. Huber total-variation (δ=0.05) is weighted by $\lambda _ { b } , \lambda _ { g } = 5 { \times } 1 0 ^ { - 4 }$ and $\lambda _ { c } = 1 \times 1 0 ^ { - 4 }$ , and $\lambda _ { x }$ is set to $\mathrm { 1 \times 1 0 ^ { - 3 } }$ GRASP reference reconstructions were obtained using a public implementation [1], with a relative temporal lambda of 0.05. L+S regularization parameters were set to $\lambda _ { L } = 0 . 0 1 / \lambda _ { S } = 0 . 0 0 5$ . Hash-INR (16 hash levels and $2 ^ { 1 9 }$ map size for encoding, 2-hidden-layer MLP with 128 nodes) was trained for 5k iterations.

Table 1: Aorta TIC metrics. Mean ± std across 5 subjects (ROI-averaged).
<table><tr><td>Metric</td><td>GRASP</td><td>LS</td><td>INR</td><td>Gaussian</td><td>Gabor</td></tr><tr><td></td><td></td><td></td><td></td><td> $\mathrm { P / B \uparrow } \qquad 3 . 5 0 \pm 0 . 8 4 2 . 8 9 \pm 0 . 6 2 3 . 0 1 \pm 0 . 5 8 3 . 4 1 \pm 0 . 8 3 3 . 3 7 \pm 0 . 7 8$ </td><td></td></tr><tr><td>wash-in ↑</td><td> $0 . 3 5 \pm 0 . 1 5 0 . 1 7 \pm 0 . 0 6 0 . 1 2 \pm 0 . 0 3 0 . 3 0 \pm 0 . 1 3 0 . 2 9 \pm 0 . 1 3$ </td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $\mathrm { T V _ { p o s t } } \uparrow 0 . 1 5 \pm 0 . 0 5 0 . 1 2 \pm 0 . 0 5 0 . 0 9 \pm 0 . 0 3 0 . 1 4 \pm 0 . 0 4 0 . 1 4 \pm 0 . 0 4$ </td><td></td><td></td><td></td><td></td></tr></table>

![](images/d19b8dccfd19ff332aa4a24075ca7a2d75856e561bff7dd52088c9f4f0cdeee8.jpg)  
Fig. 3: Aorta and kidney time intensity curve comparison. A sharp aortic peak is expected as soon as the contrast is injected (∼35 seconds). INR over-smooths the signal, while Gaussian and Gabor maintain a sharper peak than the reference GRASP. Kidney curves confirm similar dynamic behavior.

## 4 Results

Reconstruction quality. Figure 2 shows spatial (xy) and spatiotemporal (xt) reconstructions for two slices. Hash-INR is competitive spatially but slightly noisier than the GRASP reference and blurs along the temporal dimension. Among the primitive models, Gabor recovers sharper spatial detail than the Gaussian variant and preserves the temporal sharpness that the INR loses. Quantitative image similarity confirms the sharper reconstruction, yet these should be regarded as indicative only, since GRASP does not constitute a true ground truth.

![](images/2ea5fd7cf953b98c770aa283e5beefd9bf531151310686e9ae7cbae5792b8a14.jpg)  
Fig. 4: Gabor representation analysis. (A) Location and scale of the top-20 enhanced primitives overlaid on the reconstruction. (B) Base- and DCE-tier contrast intensity over time for the top-10% enhancing primitives; the base stays flat while the DCE tier rises at injection. (C) Learned motion-tier temporal basis (general drift, slice 05; respiratory motion, slice 12).

Time-intensity curves. Figure 3 compares aorta and kidney time-intensity curves; a sharp aortic peak is expected shortly after injection (∼35 s). Hash-INR and L+S oversmoothen this peak, whereas the Gaussian and Gabor reconstructions retain a peak similar to the GRASP reference, and the kidney curves show the same behaviour. Quantitative values (Table 1) show similar peak behavior as well as post contrast temporal smoothness of GRASP, Gabor and Gaussian. Representation analysis. Figure 4 inspects the learned representation for a subject. Panel (A) overlays the most enhancing primitives on the reconstruction, indicating that their locations coincide with the expected enhancing anatomy. Panel (B) shows the contrast intensity contributed by the base and DCE tiers over time for these primitives. The base tier stays flat while the DCE tier rises at injection, showing that the model separates the contrast dynamics from the static anatomy. Panel (C) visualizes the learned motion basis, which exhibits a periodic pattern for slice 13 that may be attributed to respiratory motion.

## 5 Discussion and Conclusion

We presented the first primitive-based framework for DCE-MRI reconstruction. By disentangling the representation into a base, a dynamic-contrast, and a geometry-coupled tier, the model reconstructs highly undersampled acquisitions with quality competitive with the clinical GRASP reference, while preserving the sharp aortic and renal enhancement that the comparison methods over-smooth. The recovered contrast primitives localize to the expected anatomical regions and separate the enhancement from the static anatomy, yielding a direct geometric interpretation. The learned motion basis further indicates that a primitive representation can jointly capture contrast and motion in a disentangled manner.

Yet, the present evaluation is limited by the absence of a ground-truth reference: the quantitative comparison is made against GRASP, which itself temporally smooths the bolus, and is restricted to a small in-vivo cohort. Rendering every frame also couples the reconstruction cost to the temporal resolution, trading training time for finer dynamics and requires further hyperparameter tuning.

Overall, we have shown the potential of representing undersampled DCE-MRI with an explicit geometric meaning, a representation that inherently supports higher acceleration through explicit regularization. Extending the framework to further multi-dimensional MRI settings and driving downstream tasks such as tracer-kinetic parameter estimation motivate further exploration.

## References

1. Ariyurek, C., Koçanaoğulları, A., Sari, C.T., Vasylechko, S., Afacan, O., Kurugol, S.: PyGRASP: A standalone Python image reconstruction library for DCE-MRI acquired with radial sampling. In: Proc. Intl. Soc. Mag. Reson. Med. (ISMRM). p. 2404 (2023), software: https://github.com/quin-med-harvard-edu/pyGRASP

2. Catalán, T., Courdurier, M., Osses, A., Fotaki, A., Botnar, R., Sahli-Costabal, F., Prieto, C.: Unsupervised reconstruction of accelerated cardiac cine mri using neural fields. Computers in biology and medicine 185, 109467 (2025)

3. Dannecker, M., Jia, S., Stolt-Ansó, N., Girard, N., Auzias, G., Rousseau, F., Rueckert, D.: Fast and explicit: Slice-to-volume reconstruction via 3D Gaussian primitives with analytic point spread function modeling. arXiv preprint arXiv:2512.11624 (2025)

4. Feng, J., Feng, R., Wu, Q., Shen, X., Chen, L., Li, X., Feng, L., Chen, J., Zhang, Z., Liu, C., et al.: Spatiotemporal implicit neural representation for unsupervised dynamic MRI reconstruction. IEEE Transactions on Medical Imaging 44(5), 2143– 2156 (2025)

5. Feng, L., Axel, L., Chandarana, H., Block, K.T., Sodickson, D.K., Otazo, R.: XD-GRASP: Golden-angle radial MRI with reconstruction of extra motion-state dimensions using compressed sensing. Magnetic Resonance in Medicine 75(2), 775– 788 (2016)

6. Grattan-Smith, J.D., Chow, J., Kurugol, S., Jones, R.A.: Quantitative renal magnetic resonance imaging: magnetic resonance urography. Pediatric Radiology 52(2), 228–248 (2022)

7. Gunwhy, E.R., Kurugol, S., Serai, S., van der Molen, A.J., Abou El-Ghar, M., Buckley, D.L., Hockings, P.D., Jones, R.A., Lim, R.P., Mendichovszky, I.A., et al.: Consensus-based technical recommendations for clinical translation of renal dynamic contrast-enhanced (DCE) MRI. medRxiv pp. 2026–05 (2026)

8. Heckel, R., Jacob, M., Chaudhari, A., Perlman, O., Shimron, E.: Deep learning for accelerated and robust MRI reconstruction. Magnetic Resonance Materials in Physics, Biology and Medicine 37(3), 335–368 (2024)

9. Huang, W., Li, H.B., Pan, J., Cruz, G., Rueckert, D., Hammernik, K.: Neural implicit k-space for binning-free non-Cartesian cardiac MR imaging. In: International Conference on Information Processing in Medical Imaging (IPMI). pp. 548–560. Springer (2023)

10. Huang, W., Spieker, V., Stolt-Ansó, N., Niessen, N., Dannecker, M., Kafali, S.G., Kurugol, S., Schnabel, J.A., Rueckert, D.: Gabor primitives for accelerated cardiac cine MRI reconstruction. arXiv preprint arXiv:2603.05681 (2026)

11. Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., Maier-Hein, K.H.: nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18(2), 203–211 (2021)

12. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (TOG) 41(4), 1– 15 (2022)

13. Otazo, R., Candès, E., Sodickson, D.K.: Low-rank plus sparse matrix decomposition for accelerated dynamic MRI with separation of background and dynamic components. Magnetic Resonance in Medicine 73(3), 1125–1136 (2015)

14. Rifel, P., Zoellner, F.G., Budjan, J., Grimm, R., Block, T.K., Schoenberg, S.O., Hausmann, D.: "One-Stop Shop": Free-breathing dynamic contrast-enhanced magnetic resonance imaging of the kidney using iterative reconstruction and continuous golden-angle radial sampling. Investigative Radiology 51(11), 714–719 (2016)

15. Sourbron, S.P., Buckley, D.L.: Classic models for dynamic contrast-enhanced MRI. NMR in Biomedicine 26(8), 1004–1027 (2013)

16. Spieker, V., Eichhorn, H., Hammernik, K., Rueckert, D., Preibisch, C., Karampinos, D.C., Schnabel, J.A.: Deep learning for retrospective motion correction in MRI: a comprehensive review. IEEE Transactions on Medical Imaging 43(2), 846–859 (2023)

17. Spieker, V., Huang, W., Eichhorn, H., Stelter, J., Weiss, K., Zimmer, V.A., Braren, R.F., Karampinos, D.C., Hammernik, K., Schnabel, J.A.: ICoNIK: Generating respiratory-resolved abdominal MR reconstructions using neural implicit representations in k-space. In: Deep Generative Models (DGM4MICCAI), MICCAI Workshop. pp. 183–192. Lecture Notes in Computer Science, Springer (2024)

18. Terpstra, M., van den Berg, C.: Fast undersampled dynamic MRI reconstruction using explicit representation learning with Gaussian splatting. In: ISMRM Workshop on Data Sampling and Image Reconstruction. Sedona, AZ, USA (2026)

19. Uecker, M., Lai, P., Murphy, M.J., Virtue, P., Elad, M., Pauly, J.M., Vasanawala, S.S., Lustig, M.: ESPIRiT – an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA. Magnetic Resonance in Medicine 71(3), 990– 1001 (2014)

20. Zhang, X., Ge, X., Xu, T., He, D., Wang, Y., Qin, H., Lu, G., Geng, J., Zhang, J.: GaussianImage: 1000 FPS image representation and compression by 2D Gaussian splatting. In: European Conference on Computer Vision (ECCV) (2024)