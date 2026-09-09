# SynthRCT: Scalable Conditional Deformation Synthesis for Synthetic Repeat CT Generation

Tomas Guija-Valiente<sup>1[0009−0000−0911−3317]</sup>, Blanca Rodriguez-Gonzalez<sup>1[0009−0007−0982−1293]</sup>, and Norberto Malpica<sup>1[0000−0003−4618−7459]</sup>

Medical Image Analysis and Biometry Lab, Universidad Rey Juan Carlos, Madrid, Spain

tomas.guija@urjc.es

Abstract. In proton therapy, plans are typically optimized on a single planning CT, making robustness evaluation essential under anatomical changes. However, current scenarios often rely on simplified perturbations that poorly capture complex, patient-specific variability. We propose SynthRCT, a scalable conditional generative framework for 3D anatomical deformation synthesis. Based on a conditional variational autoencoder, SynthRCT learns a latent deformation space and decodes sampled latent codes into local stationary velocity fields conditioned on an input anatomy. Local fields are assembled into coherent full-volume transformations, enabling memory-scalable generation for large field-ofview CT data. We validate the approach on respiratory 4DCT data with multiple breathing-phase anatomies per subject. SynthRCT enables patient-specific sampling of plausible anatomical transformations beyond predefined robustness scenarios. Code available at: https:// github.com/TomasGuija/SynthRCT.

Keywords: Adaptive Proton Therapy · Generative AI.

## 1 Introduction

Proton therapy can concentrate dose around the target with high precision, but its finite beam range makes treatment plans highly sensitive to anatomical deviations between the planning CT and treatment anatomy [22,14]. Inter- and intra-treatment anatomical variations can therefore compromise plans optimized on a single planning CT [15], making robustness evaluation essential to assess whether target coverage and organ-at-risk sparing are preserved under plausible treatment conditions [10]. Current workflows often define such conditions through predefined perturbations of the planning image, but these simplified scenarios only approximate the complex, patient-specific anatomical variability observed during treatment. A more flexible alternative is to generate plausible deformations directly from the planning image, enabling evaluation under a richer set of synthetic but realistic anatomical scenarios.

![](images/e7256ffb9b20b3c20b1e11c2eef6d9a85b3ff70217139c5b36c6d51a77aa6075.jpg)  
Fig. 1: Overview of the proposed SynthRCT framework. A latent deformation code is inferred from a moving–fixed CT pair during training, or sampled from the anatomy-conditioned prior at inference. The code conditions slab-wise SVF decoding, followed by SVF stitching, integration, and diferentiable warping to generate synthetic repeat CT anatomies.

Deformable image registration provides the standard framework for estimating transformations between observed images. Widely used methods such as Demons and SyN solve an optimization problem for each image pair [23,2]. Learning-based methods such as VoxelMorph replace iterative optimization with neural networks that predict dense displacement or velocity fields from image pairs [4,9,13,8,19,24]. However, these methods estimate one transformation between two observed images, rather than sampling multiple plausible transformations from a reference anatomy.

A complementary line of work models anatomical variability, using statistical deformation models [5,21,18], latent probabilistic models [20,11,16], or deformationbased difusion models [25] to sample plausible anatomical changes. Despite these advances, scalable conditional deformation generation for large-field-ofview 3D CT remains challenging, since high-resolution volumetric fields are memory-intensive and local predictions must be combined into coherent fullvolume transformations.

We propose SynthRCT, a scalable conditional generative model for 3D anatomical deformation synthesis. SynthRCT learns global latent deformation modes from intra-patient CT pairs and decodes sampled codes into local SVFs over axial slabs, i.e., sub-volumes that cover the full in-plane field of view but only a limited range of axial slices. Local stationary velocity fields are assembled and integrated into a coherent full-volume transformation, reducing memory requirements while preserving high-resolution local deformation modeling, making the approach suitable for large field-of-view CT data.

Our main contributions are a conditional generative framework for patientspecific repeat CT synthesis and a scalable local SVF generation and composition strategy that produces coherent full-volume deformations. We validate the approach through experiments assessing registration accuracy, deformation regularity, latent-space consistency, and landmark-distribution agreement.

## 2 Methods

## 2.1 SynthRCT Overview

## Conditional latent deformation model

Let M, $F : \varOmega \subset \mathbb { R } ^ { 3 } \to \mathbb { R }$ denote two intra-patient CT volumes, where M is the moving image and $F$ is the fixed image. SynthRCT learns a conditional distribution of plausible target anatomies F given an input anatomy M, induced by latent deformation codes $z \in \mathbb { R } ^ { d }$

$$
p _ { \theta } ( F \mid M ) = \int p _ { \theta } ( F \mid z , M ) p _ { \theta } ( z \mid M ) d z .\tag{1}
$$

Here, $p _ { \theta } ( z \mid M )$ is an anatomy-conditioned prior learned by a CNN encoder that receives the moving image as input, and is regularized toward $\mathcal { N } ( 0 , I )$ to keep the latent space compact and well behaved. During training, we introduce an approximate posterior distribution $q _ { \psi } ( z \mid F , M )$ , parameterized by a CNN encoder that observes the moving–fixed pair $( M , F )$ . This posterior approximates the latent deformation codes that explain the transformation from M to F. At inference time, F is not available; latent codes are therefore sampled from the anatomy-conditioned prior $p _ { \theta } ( z \mid M )$

## SVF parameterization and integration.

SynthRCT parameterizes anatomical change through stationary velocity fields (SVFs). Given an SVF $v : \varOmega \to \mathbb { R } ^ { 3 }$ , the transformation is obtained as:

$$
\phi = \exp ( v ) ,\tag{2}
$$

which we approximate using scaling and squaring [1]. Since a suficiently smooth velocity field integrates to a smooth invertible transformation, operating in the SVF domain reduces the risk of non-physical foldings compared with directly predicting an unconstrained displacement field.

## Slab-wise deformation generator.

The likelihood $p _ { \theta } ( F \mid z , M )$ is defined implicitly by generating an SVF, integrating it into a transformation, and warping the moving image. To scale synthesis to large CT volumes, SynthRCT predicts SVFs locally over axial slabs. An axial slab $M _ { \mathrm { l o c } }$ is defined as s contiguous slices of the moving image M along the axial direction. The prior and posterior encoders operate on the full-volume anatomies M and $F ,$ so that the latent variable z captures global deformation modes. In contrast, the deformation generator $G _ { \theta }$ decodes each axial slab $M _ { \mathrm { l o c } }$ together with the sampled latent code z into a local SVF.

The generator $G _ { \theta }$ is implemented as a convolutional U-Net encoder–decoder with skip connections. Its encoder $E _ { \theta }$ extracts multi-resolution features from $M _ { \mathrm { l o c } } ,$ while the decoder $D _ { \theta }$ maps these to a local SVF. The latent code z is injected into the decoder through Feature-wise Linear Modulation (FiLM) [17] allowing the same local anatomy to be deformed according to diferent sampled deformation modes. This design combines global latent conditioning with local anatomical decoding, enabling coherent respiratory motion modeling while preserving local detail and scaling to large fields of view.

## Overlap refinement and full-volume assembly.

The slab-wise generator produces local SVF predictions, but the final synthetic CT requires a single coherent full-volume deformation. We therefore decode axial slabs with an overlap of O slices and merge their predictions in the SVF domain before integration. Consider two neighboring overlapping slabs, denoted $M _ { 1 }$ and $M _ { 2 }$ . Both are decoded with the same global latent code z:

$$
v _ { 1 } = G _ { \theta } ( M _ { 1 } , z ) , \qquad v _ { 2 } = G _ { \theta } ( M _ { 2 } , z ) .\tag{3}
$$

Although the shared latent code encourages global consistency, independently decoded slabs may still present small discontinuities at their overlap. To reduce these artifacts, we introduce a convolutional refiner $S _ { \eta }$ that receives the local SVF predictions together with the corresponding moving-image context $M _ { 1 : 2 }$ and predicts a residual correction for the overlapping region:

$$
\varDelta { v } _ { 1 : 2 } = S _ { \eta } \left( v _ { 1 } , v _ { 2 } , M _ { 1 : 2 } \right) .\tag{4}
$$

The refiner modifies only the overlap. Let $v _ { 1 } ^ { \mathrm { o v } }$ and $v _ { 2 } ^ { \mathrm { o v } }$ denote the restrictions of the two local SVFs to their common overlapping region. The refined overlap is obtained by adding the predicted residual to the average transition:

$$
v _ { 1 : 2 } ^ { \mathrm { o v } } = \frac { 1 } { 2 } \left( v _ { 1 } ^ { \mathrm { o v } } + v _ { 2 } ^ { \mathrm { o v } } \right) + \varDelta v _ { 1 : 2 } .\tag{5}
$$

At inference time, all overlapping slabs are first decoded independently using the same sampled latent code $z .$ . The refiner is then applied to each neighboring slab pair, replacing each averaged overlap with its refined counterpart. The resulting regions are assembled into a single full-volume $\operatorname { S V F } v ^ { \mathrm { f u l l } }$

The synthetic repeat CT is finally obtained by warping the input image:

$$
\phi ^ { \mathrm { f u l l } } = \exp ( v ^ { \mathrm { f u l l } } ) , \qquad \hat { F } = M \circ \phi ^ { \mathrm { f u l l } } .\tag{6}
$$

## 2.2 Training Objective

The model is trained from intra-patient moving–fixed CT pairs (M, F). For each pair, a latent code z is sampled from the posterior $q _ { \psi } ( z \mid F , M )$ and shared across all sampled local regions, encouraging z to represent global deformation patterns rather than independent local perturbations.

During training, we sample neighboring overlapping slabs from the moving image and decode them with the shared latent code. Their predicted SVFs are refined and assembled into a single local SVF $v ^ { \mathrm { p a i r } }$ . This SVF is integrated and used to warp the corresponding moving region:

$$
\phi ^ { \mathrm { p a i r } } = \exp \left( \upsilon ^ { \mathrm { p a i r } } \right) , \qquad \hat { F } ^ { \mathrm { p a i r } } = M ^ { \mathrm { p a i r } } \circ \phi ^ { \mathrm { p a i r } } ,\tag{7}
$$

where $M ^ { \mathrm { p a i r } }$ denotes the moving-image region covered by the sampled slab pair.

Image supervision is imposed using local normalized cross-correlation (LNCC). In addition to the final-resolution prediction, the decoder produces intermediateresolution SVF’s used for deep supervision. The similarity loss is therefore

$$
\mathcal { L } _ { \mathrm { s i m } } = \mathcal { L } _ { \mathrm { L N C C } } ^ { \mathrm { f i n e } } + \lambda _ { \mathrm { c o a r s e } } \mathcal { L } _ { \mathrm { L N C C } } ^ { \mathrm { c o a r s e } } ,\tag{8}
$$

where the coarse term supervises the intermediate decoder outputs after integration and warping at the corresponding resolution.

The latent space is regularized by matching the training posterior to the anatomy-conditioned inference prior, and by anchoring the prior to a standard Gaussian:

$$
{ \mathcal { L } } _ { \mathrm { K L } } = D _ { \mathrm { K L } } \left( q _ { \psi } ( z \mid F , M ) \parallel p _ { \theta } ( z \mid M ) \right) + \alpha D _ { \mathrm { K L } } \left( p _ { \theta } ( z \mid M ) \parallel { \mathcal { N } } ( 0 , I ) \right) .\tag{9}
$$

The final objective is

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { s i m } } \mathcal { L } _ { \mathrm { s i m } } + \beta ( t ) \mathcal { L } _ { \mathrm { K L } } , } \end{array}\tag{10}
$$

where $\lambda _ { \mathrm { s i m } }$ controls image supervision, $\lambda _ { \mathrm { c o a r s e } }$ weights deep supervision, α regularizes the conditional prior, and $\beta ( t )$ is a KL warm-up factor.

## 3 Experiments

We evaluate SynthRCT on respiratory 4DCT data, using $d = 3 2$ latent dimensions and axial slabs of 48 slices with 50% overlap. Experiments assess image alignment, deformation regularity, latent-space behaviour, and agreement between sampled and observed respiratory landmark distributions. Additional implementation details on the model architecture and training strategy are provided in the publicly available source code repository.

## 3.1 Dataset

Experiments use the DIR4DCT dataset [6,7], with 10 thoracic 4DCT patients, 10 respiratory phases, and 75 annotated landmark trajectories per subject. Volumes are resampled to 1.5 mm isotropic spacing, normalized to [0, 1], and crop/padded around a patient-specific anatomical center to 256×256×207 voxels. Foreground masks are obtained by intensity thresholding and used to restrict image similarity supervision. Training and evaluation pairs are intra-patient phase pairs separated by at least two respiratory positions to avoid near-identity cases.

## 3.2 Registration Accuracy and Deformation Regularity

Although SynthRCT is not a deterministic registration method, we evaluate posterior-encoded alignment to verify that its latent space reconstructs meaningful respiratory transformations. The posterior encodes the pair $( M , F )$ , but the generator receives only the moving slab and latent code $z ;$ the fixed image can influence the deformation only through this low-dimensional bottleneck. Therefore, lower alignment accuracy than registration methods such as SyN [2,3], Demons [23,12], and VoxelMorph [4,9] is expected.

![](images/72a203e4cb8c614d5c8a48180a781aecfba54be59f83e587c7458d33376784f9.jpg)  
(a) LNCC

![](images/e85ddde58d6b09f43bd53f172d3586c8501381758c2fc747b891347a40d2b005.jpg)  
(b) RMSE

![](images/3c384c73e46f8a65b566ecc611487e2cb126121ea26b1c11e9787867f6b116e6.jpg)  
(c) Lung Dice  
Fig. 2: Alignment evaluation on held-out DIR4DCT patients. SynthRCT is compared with the unregistered moving image and deterministic registration methods. Candles show the mean and standard deviation across held-out patients, where each patient-level value is obtained by averaging over all evaluated moving–fixed respiratory phase pairs.

We report LNCC, RMSE, and lung Dice overlap (Fig. 2). SynthRCT improves over the unregistered moving image while remaining below deterministic registration baselines, consistent with its generative objective.

Deformation regularity is assessed through the Jacobian determinant of the transformation ϕ. In particular, we measure the folding percentage, defined as the fraction of voxels satisfying det $\nabla \phi \leq 0$ . Such voxels correspond to local orientation reversals and therefore indicate non-physical topology violations. SynthRCT produced approximately zero folding across the evaluated held-out patients, indicating topology-preserving full-volume transformations.

## 3.3 Landmark distribution evaluation

To evaluate whether sampled deformations reproduce realistic respiratory motion beyond image similarity, we compare real and generated landmark configurations. For each reference phase M, we sample N latent codes $z _ { i } \sim p _ { \theta } ( z \mid M )$ ， decode them into full-volume SVFs, integrate them, and warp the $\left( \mathrm { L } \mathrm { = } 7 5 \right)$ reference landmarks. This yields generated configurations $\hat { \mathcal X } = \{ \hat { X } _ { i } \} _ { i = 1 } ^ { N }$ , which are compared with the real respiratory configurations $\mathcal { X } = \{ X _ { j } \} _ { j = 1 } ^ { R }$ . Distances are computed in physical space using the root-mean-square distance.

Table 1 reports distribution-level metrics between $\hat { \mathcal X }$ and X. Energy distance $D _ { E }$ and Wasserstein-1 distance $W _ { 1 }$ measure global discrepancy between the real and generated landmark distributions. Coverage $d _ { \mathrm { c o v } }$ is the average distance from each real configuration to its nearest generated sample, assessing whether generated samples span the observed respiratory states. Precision $d _ { \mathrm { p r e c } }$ is the reverse nearest-neighbor distance, measuring whether generated samples remain close to the real distribution. The spread ratio $\rho _ { \mathrm { s p r e a d } }$ compares generated and real pairwise variability, with values close to one indicating matched motion diversity.

Table 1: Landmark distribution evaluation on the held-out validation patient. Distance-based metrics are reported in millimetres.
<table><tr><td>Case  $D _ { E }$ </td><td>[mm] ↓  $W _ { 1 }$  [mm] ↓  $d _ { \mathrm { c o v } }$  [mm] ↓  $d _ { \mathrm { p r e c } }$  [mm] ↓ ρspread → 1</td></tr><tr><td>Held-out patient 1.44</td><td>2.61 1.69 1.78</td></tr></table>

The generated distribution closely matches the observed respiratory landmark configurations, with millimetre-scale configuration discrepancies.

## 3.4 Latent Space Study

To assess the generative behaviour of SynthRCT, we analyze the learned latent deformation space through interpolation and latent-code statistics.

![](images/2203bd605b7b8dc4f6ecf8a3257f979b90a5736f55be8bbd8ad193502f97122f.jpg)  
Fig. 3: PCA analysis of the learned latent deformation space.

We test whether linear interpolation in latent space produces smooth respiratory trajectories. For a held-out patient, we interpolate between the identity deformation and the largest observed respiratory deformation, decode the interpolated codes into full-volume SVFs, integrate them, and warp the reference image. As shown in Fig. 4, the generated anatomies follow intermediate respiratory phases, with SVF magnitude increasing gradually along the trajectory.

![](images/dfa393562eca7f0866918c3148681c4a6f4b68782f4165f37bfa09f5f1e6d8c0.jpg)  
Fig. 4: Latent-space interpolation on a held-out patient. Columns show linear interpolation between the identity deformation and the largest observed respiratory deformation. Rows show the ground-truth phase, generated warped image, 3D overlay, and SVF magnitude.

We further analyze latent-space organization by applying PCA to more than 300 posterior latent codes encoded from moving–fixed training pairs. Each code is projected onto the first principal component and compared with deformationderived statistics.

Fig. 3 shows that displacement along the main latent direction is associated with increasing deformation magnitude. The color pattern further suggests that opposite directions along this component correspond to diferent volumetric behaviours, consistent with respiratory expansion and contraction.

## 4 Conclusion

We presented SynthRCT, a conditional probabilistic framework for synthesizing plausible 3D anatomical deformations from a single input CT. The method learns global deformation modes in a latent space while decoding local stationary velocity fields that are assembled before integration into coherent full-volume transformations. This enables sampling and interpolation of anatomical deformations for large field-of-view CT data, reducing peak allocated GPU memory by 41.6% compared with full-volume decoding. Although pairwise registration is not the main objective, the generated transformations achieved competitive alignment quality while preserving spatial regularity.

The main limitation is the restricted variability of the respiratory 4DCT data, where breathing motion dominates the learned deformation space. Future work should evaluate larger and more diverse datasets to assess whether the latent space can disentangle multiple deformation modes. Additional conditioning signals, such as segmentation masks, could enable controllable deformation synthesis, while more expressive generative models, including flow-matching or difusion-based approaches, may further improve generation quality.

Acknowledgments. This study has been funded by the MAGERIT-CM project (TEC2024/COM-44), funded by Comunidad de Madrid.

Disclosure of Interests. Authors declare no conflict of interests relevant to this research.

## References

1. Arsigny, V., Commowick, O., Pennec, X., Ayache, N.: A log-euclidean framework for statistics on difeomorphisms. In: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2006. Lecture Notes in Computer Science, vol. 4190, pp. 924–931. Springer, Berlin, Heidelberg (2006). https://doi.org/ 10.1007/11866565\_113

2. Avants, B.B., Epstein, C.L., Grossman, M., Gee, J.C.: Symmetric difeomorphic image registration with cross-correlation: Evaluating automated labeling of elderly and neurodegenerative brain. Medical Image Analysis 12(1), 26–41 (2008). https: //doi.org/10.1016/j.media.2007.06.004

3. Avants, B.B., Tustison, N.J., Song, G., Cook, P.A., Klein, A., Gee, J.C.: A reproducible evaluation of ants similarity metric performance in brain image registration. NeuroImage 54(3), 2033–2044 (2011). https://doi.org/https: //doi.org/10.1016/j.neuroimage.2010.09.025, https://www.sciencedirect. com/science/article/pii/S1053811910012061

4. Balakrishnan, G., Zhao, A., Sabuncu, M.R., Guttag, J., Dalca, A.V.: VoxelMorph: A learning framework for deformable medical image registration. IEEE Transactions on Medical Imaging 38(8), 1788–1800 (2019). https://doi.org/10.1109/ TMI.2019.2897538

5. Budiarto, E., Keijzer, M., Storchi, P.R.M., Hoogeman, M.S., Bondar, L., Mutanga, T.F., de Boer, H.C.J., Heemink, A.W.: A population-based model to describe geometrical uncertainties in radiotherapy: Applied to prostate cases. Physics in Medicine and Biology 56(4), 1045–1061 (2011). https://doi.org/10.1088/ 0031-9155/56/4/011

6. Castillo, E., Castillo, R., Martinez, J., Shenoy, M., Guerrero, T.: Four-dimensional deformable image registration using trajectory modeling. Physics in Medicine and Biology 55(1), 305–327 (2010). https://doi.org/10.1088/0031-9155/55/1/018

7. Castillo, R., Castillo, E., Guerra, R., Johnson, V.E., McPhail, T., Garg, A.K., Guerrero, T.: A framework for evaluation of deformable image registration spatial accuracy using large landmark point sets. Physics in Medicine and Biology 54(7), 1849–1870 (2009). https://doi.org/10.1088/0031-9155/54/7/001

8. Chen, J., Frey, E.C., He, Y., Segars, W.P., Li, Y., Du, Y.: TransMorph: Transformer for unsupervised medical image registration. Medical Image Analysis 82, 102615 (2022). https://doi.org/10.1016/j.media.2022.102615

9. Dalca, A.V., Balakrishnan, G., Guttag, J., Sabuncu, M.R.: Unsupervised learning of probabilistic difeomorphic registration for images and surfaces. Medical Image Analysis 57, 226–236 (2019). https://doi.org/10.1016/j.media.2019.07.006

10. van Herk, M., Remeijer, P., Lebesque, J.V.: Inclusion of geometric uncertainties in treatment plan evaluation. International Journal of Radiation Oncology Biology Physics 52(5), 1407–1422 (2002). https://doi.org/10.1016/S0360-3016(01) 02805-X

11. Krebs, J., Delingette, H., Mailhé, B., Ayache, N., Mansi, T.: Learning a probabilistic model for difeomorphic registration. IEEE Transactions on Medical Imaging 38(9), 2165–2176 (2019). https://doi.org/10.1109/TMI.2019.2897112

12. Lowekamp, B., Chen, D., Ibanez, L., Blezek, D.: The design of simpleitk. Frontiers in neuroinformatics 7, 45 (12 2013). https://doi.org/10.3389/fninf.2013. 00045

13. Mok, T.C.W., Chung, A.C.S.: Large deformation difeomorphic image registration with laplacian pyramid networks. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2020. Lecture Notes in Computer Science, vol. 12263, pp. 211–221. Springer, Cham (2020). https://doi.org/10.1007/ 978-3-030-59716-0\_21

14. Paganetti, H., Botas, P., Sharp, G.C., Winey, B.: Adaptive proton therapy. Physics in Medicine and Biology 66(22), 22TR01 (2021). https://doi.org/10.1088/ 1361-6560/ac344f

15. Pakela, J.M., Knopf, A., Dong, L., Rucinski, A., Zou, W.: Management of motion and anatomical variations in charged particle therapy: Past, present, and into the future. Frontiers in Oncology 12, 806153 (2022). https://doi.org/10.3389/fonc. 2022.806153

16. Pastor-Serrano, O., Habraken, S., Hoogeman, M., Lathouwers, D., Schaart, D., Nomura, Y., Xing, L., Perkó, Z.: A probabilistic deep learning model of interfraction anatomical variations in radiotherapy. Physics in Medicine and Biology 68(8), 085018 (2023). https://doi.org/10.1088/1361-6560/acc71d

17. Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A.C.: FiLM: Visual reasoning with a general conditioning layer. In: Proceedings of the AAAI Conference on Artificial Intelligence (2018)

18. Rios, R., de Crevoisier, R., Ospina, J.D., Commandeur, F., Lafond, C., Simon, A., Haigron, P., Espinosa, J., Acosta, O.: Population model of bladder motion and deformation based on dominant eigenmodes and mixed-efects models in prostate cancer radiotherapy. Medical Image Analysis 38, 133–149 (2017). https://doi. org/10.1016/j.media.2017.03.001

19. Skibbe, H., Byra, M., Watakabe, A., Yamamori, T., Reisert, M.: Memory eficient training for 3d brain image registration networks using PatchMorph. Scientific Reports 16, 14386 (2026). https://doi.org/10.1038/s41598-026-44858-x

20. Sohn, K., Lee, H., Yan, X.: Learning structured output representation using deep conditional generative models. In: Advances in Neural Information Processing Systems. vol. 28 (2015)

21. Szeto, Y.Z., Witte, M.G., van Herk, M., Sonke, J.J.: A population based statistical model for daily geometric variations in the thorax. Radiotherapy and Oncology 123(1), 99–105 (2017). https://doi.org/10.1016/j.radonc.2017.02.012

22. Unkelbach, J., Paganetti, H.: Robust proton treatment planning: Physical and biological optimization. Seminars in Radiation Oncology 28(2), 88–96 (2018). https://doi.org/10.1016/j.semradonc.2017.11.005

23. Vercauteren, T., Pennec, X., Perchant, A., Ayache, N.: Difeomorphic demons: Eficient non-parametric image registration. NeuroImage 45(1), S61–S72 (2009). https://doi.org/10.1016/j.neuroimage.2008.10.040

24. Wu, J., Zhou, S., Lin, L., Wang, X., Tan, W.: Fast difeomorphic image registration using patch based fully convolutional networks. In: Proceedings of the 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society. pp. 1–4 (2024). https://doi.org/10.1109/EMBC53108.2024.10781975

25. Zheng, J.Q., Mo, Y., Sun, Y., Li, J., Wu, F., Wang, Z., Vincent, T., Papież, B.W.: Deformation-recovery difusion model (DRDM): Instance deformation for image manipulation and synthesis. Medical Image Analysis 110, 103987 (2026). https: //doi.org/10.1016/j.media.2026.103987