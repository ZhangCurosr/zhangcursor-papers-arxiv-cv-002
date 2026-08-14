# Splat-based Metal Artifact Reduction in Cone-Beam CT via Polychromatic Modeling

Kiseok Choi

![](images/ac1a38d1ac5f1a5f6faa5e15035df08880d9cdc2c787387fdbb50a277864245f.jpg)  
Walnut

Inchul Kim

![](images/dad90822a806ec368b52fe31576c55eda8043648771963dccf033ad93e09c367.jpg)  
FDK (reference)

Jaemin Cho

![](images/b7343b79cf8549c14802c28aa62019ad30437ace19d7e1de413851c346349edc.jpg)  
FDK  
KAIST

Hyeongjun Cho

![](images/adc308ab501345aea7a02e7c88996508a19e454948366d8b9ad15c312bad6e9e.jpg)

Min H. Kim

Polyner  
![](images/4540b534e6d8389afcc9e0b4428085e32a348c33ec481071b49cb4e4cb3c398d.jpg)  
Park et al.

![](images/cb0b10c5f34801f9a0f285370d99384f334d4a98a7bef8d3047328927b330765.jpg)

![](images/3a4709b4e6cfde7cad1513110cf3aaa99012fb4b9a9087444b3e27eab150a08d.jpg)  
Ours

Figure 1: Qualitative comparison ofCBCT reconstruction resultsfor a real walnut object with inserted metal pins. The leftmost column shows the optical photograph ofthe walnut, where the blue and red planes denote the positions ofthe horizontal and vertical cutsfor visualization. Each reconstruction method is visualized with two slices: a horizontal slice (top row) and a vertical slice (bottom row)from the reconstructed volume. The FDK (reference) result corresponds to a baseline scan ofthe walnut without metal pins and serves as a ground-truth proxyfree from metal artifacts. FDK [FDK84] applied to the metal-inserted scan exhibits severe beam hardening artifacts, including dark streaks and intensity distortions. Polyner [WCW∗23] reduces some artifacts but shows noticeable blurring and oversmoothing offine structures. Park e al. [PSJ] minimizes artifacts but considerably compromises the overall image structures, suferingfrom noise. Our method delivers the mos faithful reconstruction by efectively suppressing artifacts while preserving structural detail in both axial and sagittal views, demonstrating its robustness for cone-beam CT with severe metal-induced beam hardening.

## Abstract

Cone-beam computed tomography (CBCT) enables volumetric reconstructionfrom X-ray projections, but sufersfrom severe artifacts–especially beam hardening–when imaging materials with high attenuation such as metals. These artifacts arisefrom the polychromatic nature ofX-rays and are not properly addressed by conventional monochromatic reconstruction algorithms. While recent neural representation-based methods ofer improved reconstruction quality, they are computationally expensive and often impractical for deployment. We propose a novel physics-inspired, self-calibrating metal artifact reduction method that eficiently reconstructs 3D CBCT volumes while correcting beam hardening artifacts. Our method integrates a polychromatic X-ray projection model, material-dependent attenuation profiles, and system response modeling into a Gaussian Splatting framework. Unlike prior work, we eliminate the need for manual metal masks or strong prior assumptions, and we optimize both reconstruction parameters and X-ray spectral characteristics jointly during training. We further introduce a high-fidelity synthetic CBCT dataset generation pipeline validated on Monte-Carlo x-ray simulation toolbox and release new datasets with severe metal-induced artifacts to support the community. This is thefirst splat-based methodfor reducing beam hardening in CBCT. Extensive experiments on both synthetic and real-world datasets demonstrate that our method outperforms state-of-the-ar approaches in artifact suppression and reconstruction accuracy.

## CCS Concepts

<sup>•</sup> <sup>Social</sup> <sup>and</sup> <sup>professional</sup> <sup>topics</sup> → <sup>Medical</sup> <sup>technologies;</sup> <sup>•</sup> <sup>Computing</sup> <sup>methodologies</sup> → <sup>Reconstruction;</sup> <sup>Volumetric</sup> <sup>models;</sup>

## 1. Introduction

X-rays are high-energy electromagnetic waves capable of penetrating various materials. As X-rays propagate through matter, they are attenuated according to the Beer-Lambert law, which models the exponential decay of energy along the path proportional to material density. X-ray computed tomography (CT) reconstructs internal volumetric structures by analyzing the attenuation. Among several CT configurations—parallel-beam, fan-beam, cone-beam, and helicalbeam—cone-beam CT (CBCT) has attracted particular interest due to its ability to capture dense projection views in a single scan via conic emission and 2D planar detection, enabling eficient volumetric acquisition. This faster acquisition also contributes to reduced Xray exposure, making CBCT an attractive modality for low-dose tomography applications.

While beam hardening artifacts are a common challenge in most X-ray CT systems due to the polychromatic nature of X-ray sources, where low-energy photons are absorbed more readily than highenergy ones, CBCT is particularly susceptible to such artifacts. Its wide cone angle and increased scatter amplify these nonlinear attenuation efects, leading to dark streaks, cupping artifacts, and geometric distortions that significantly degrade image quality (Figure 1). These artifacts not only impair diagnostic interpretation but also hinder downstream computational analysis such as segmentation or 3D modeling.

Traditional beam hardening correction (BHC) methods, includ ing sinogram interpolation using predefined metal masks [KHE87, MRL∗10], are limited by their reliance on heuristic preprocessing. Data-driven approaches [LLZL20, WLH∗22, WLMZ22, WXZ∗24] have demonstrated improved artifact removal using supervised learning or dictionary-based priors, but often require metal masks and paired training data, limiting generalization across materials and geometries. More recently, physically-motivated methods have been proposed to address beam hardening during reconstruction. Polyner [WCW∗23] reconstructs energy-dependent voxel attenuations by assuming known X-ray spectra, and Park et al. [PSJ] leverage a simplified linear attenuation-energy model under fixed energy assumptions. While these methods show promise, they often depend on prior spectral calibration, operate under restrictive fan-beam geometries, or require computationally expensive neural volumetric representations, making them less suitable for practical CBCT systems.

In this work, we tackle the longstanding challenge of beam hardening artifact reduction in CBCT by introducing a physicsinspired reconstruction framework that models both the polychromatic X-ray spectrum and material-specific attenuation behavior. Our method is built upon a diferentiable Gaussian Splatting (GS) framework [ZLC∗24], and incorporates an energy-aware attenuation model as well as a self-calibrating system response formulation. This unified framework allows us to jointly reconstruct high-quality attenuation volumes and correct beam hardening artifacts, without requiring any prior knowledge of the X-ray spectrum or manual metal masks.

In summary, we present a physics-inspired reconstruction framework that directly targets the beam hardening problem in cone-beam CT, a long-standing and challenging artifact in low-dose imaging. By integrating a polychromatic forward model with materialspecific attenuation behavior into a diferentiable Gaussian Splatting framework, our method achieves high-fidelity reconstruction while efectively suppressing metal-induced artifacts. This is the first splatbased method for reducing beam hardening artifacts in CBCT. A key component of our approach is a self-calibrating system response model that eliminates the need for prior spectral information or manual metal masking, significantly improving both robustness and practicality for real-world CBCT systems.

## 2. Related Work

Beam Hardening Artifact Reduction. Beam hardening artifacts are a persistent challenge in CT imaging, particularly when scanning high-attenuation materials such as metals. The underlying cause is the polychromatic nature of X-ray sources: lower-energy photons are more readily absorbed, leading to nonlinear attenuation that violates the assumptions of monochromatic models and results in dark streaks and geometric distortions.

Early correction methods such as LIMAR [KHE87] and NMAR [MRL∗10] perform sinogram interpolation using predefined metal masks, but sufer from poor generalizability and reliance on accurate mask segmentation. With the rise of deep learning, data-driven BHC methods have emerged, including ADN [LLZL20], InDuDoNet [WLZ∗21], DICDNet [WLH∗22], ACDNet [WLMZ22], and OSCNet [WXZ∗24]. These approaches attempt to learn artifactfree mappings but often require paired training data and metal masks, limiting their applicability to varied clinical scenarios.

More recent strategies tackle beam hardening directly during reconstruction by incorporating physical priors. Polyner [WCW∗23] reconstructs energy-dependent attenuation volumes assuming a known X-ray spectrum and metal mask, using a NeRF-style rep resentation. Park et al. [PSJ] adopt a simplified linear attenuation model and fixed energy assumption, avoiding prior calibration but still relying on heavy NeRF-based volumetric rendering. While these methods reduce artifacts to some extent, they are computationally expensive and often constrained to fan-beam geometries, which are incompatible with modern CBCT systems.

In contrast, our method targets beam hardening reduction for conebeam CT by integrating a polychromatic X-ray model and materialspecific attenuation profiles into a lightweight Gaussian Splatting framework [ZLC∗24]. Our system includes a self-calibrating component that jointly estimates spectral and attenuation parameters without requiring any prior spectral information or metal masks, ofering both generality and practical deployability.

CT Reconstruction Paradigms. CT reconstruction algorithms can be broadly categorized into backprojection-based, iterative, and neural volumetric methods. Backprojection-based methods like FBP [Gul79] and FDK [FDK84] use analytical models rooted in the Fourier slice theorem, but they perform poorly when projection inconsistencies arise, such as those induced by metal artifacts. Iterative methods [AK84, Gil72] treat reconstruction as an inverse problem and refine the volume estimate through multiple projection-matching steps. Edge-preserving regularizations [SP08, LMFL12, BT09] such as total variation improve performance, but they still struggle under severe nonlinearity from beam hardening. Neural representation approaches like SAX-NeRF [CWY∗24] apply neural rendering to

CT reconstruction, achieving high accuracy but incurring substantial computational cost. Variants such as NAF [ZZL22] and multiresolution hash-based methods [MESK22] improve runtime at the cost of fidelity. More recently, Gaussian Splatting (GS) [KKLD23] has been proposed as a neural-free, real-time approach to novel view synthesis. This method has been adapted to CT in works like $\mathbf { R } ^ { 2 } \cdot$ -Gaussian [ZLC∗24], which combines explicit Gaussian representations with the Beer-Lambert law to reconstruct attenuation volumes. However, $\textstyle \mathbf { R } ^ { 2 } .$ -Gaussian assumes a monochromatic model and struggles under real-world polychromatic conditions. Our work builds upon $\mathbf { R } ^ { 2 } .$ -Gaussian by extending it with a physics-inspired polychromatic attenuation model and a self-calibrated system response formulation, enabling artifact-robust cone-beam CT reconstruction under severe beam hardening conditions.

## 3. Background

## 3.1. Polychromatic X-ray Projection Model

When an X-ray propagates through an object, its energy is attenuated according to the Beer-Lambert law:

$$
I = I _ { 0 } \exp \left( - \int _ { L } \mu ( l ) d l \right) ,\tag{1}
$$

where $I _ { 0 }$ is the emission intensity of the source, $\mu ( l )$ is the attenuation coeficient along the line segment �� of the path �, and � is the intensity measured at the detector. This expression can be rearranged into a log-transformed form:

$$
P = - \log \frac { I } { I _ { 0 } } = \int _ { L } \mu ( l ) d l .\tag{2}
$$

The resulting quantity $P$ is referred to as the projection image or sinogram. However, since real-world X-ray sources emit a spectrum of energies, and the attenuation coeficient $\mu$ depends on energy, Equation (2) cannot capture the polychromatic nature of X-ray propagation. To address this, the attenuation process must be expressed as an integral over the energy range $\varepsilon { \mathrm { : } }$

$$
P = - \log \int _ { \mathcal { E } } \eta ( E ) \exp \left( - \int _ { L } \mu ( l , E ) d l \right) d E ,\tag{3}
$$

where $\eta ( E )$ denotes the system response at energy level �, including the X-ray source distribution, filter characteristics, and detector sensitivity. The attenuation coeficient $\mu ( l , E )$ is now energy-dependent and referred to as the linear attenuation coeficient (LAC). The measured polychromatic projection is obtained by integrating across the entire spectrum . The LAC is a material-dependent property that varies significantly with energy due to fundamental physical interactions, primarily the photoelectric efect and Compton scattering. This energy dependence of attenuation serves as a strong prior in estimating material-specific attenuation from polychromatic projections, as described in Equation (3).

## 3.2. R<sup>2</sup>-Gaussian

In $\mathbf { R } ^ { 2 }$ -Gaussian [ZLC∗24], the attenuation field in 3D space is modeled as a sum of anisotropic Gaussian primitives:

$$
\mu ( \mathbf { x } ) = \sum _ { i } ^ { M } \rho _ { i } \exp \left( - \frac { 1 } { 2 } ( \mathbf { x } - \mathbf { p } _ { i } ) ^ { T } \Sigma _ { i } ^ { - 1 } ( \mathbf { x } - \mathbf { p } _ { i } ) \right) ,\tag{4}
$$

where x is the spatial query point, � is the number of Gaussians, ρ<sub>�</sub> is the density of the �-th Gaussian, and $\mathbf { p } _ { i } , \boldsymbol { \Sigma } _ { i }$ denote its center and covariance, respectively. The forward projection of a ray is then computed by integrating the contribution of all Gaussians along the ray path:

$$
P \left( \hat { \mathbf { x } } \right) = \int _ { L } \mu ( l ) d l \approx \sum _ { i } ^ { M } \sqrt { \frac { 2 \pi | \tilde { \Sigma } _ { i } | } { | \hat { \Sigma } _ { i } | } } \rho _ { i } \exp \left( - \frac { 1 } { 2 } ( \hat { \mathbf { x } } - \hat { \mathbf { p } } _ { i } ) ^ { T } \hat { \Sigma } _ { i } ^ { - 1 } ( \hat { \mathbf { x } } - \hat { \mathbf { p } } _ { i } ) \right) ,\tag{5}
$$

where $\hat { \mathbf { x } }$ is the projected pixel position, $\tilde { \Sigma } _ { i }$ is the view-space covariance, $\hat { \Sigma } _ { i }$ is the projected covariance, and $\hat { \mathbf { p } } _ { i }$ is the projected Gaussian center. The parameters $\rho _ { i } , \mathbf { p } _ { i }$ , and $\Sigma _ { i }$ are optimized such that the computed projections from Equation (5) match the observed projection images. After optimization, the final attenuation volume is recovered by evaluating the sum of the fitted Gaussian components.

## 4. Method

## 4.1. Overview

The overall framework is illustrated in Figure 2. Our method consists of two types of integrals: projection and reconstruction. In the projection phase, the Gaussians are aggregated using our diferentiable polychromatic forward projector, and the resulting projections are compared against ground-truth log-transformed measurements acquired from the CT system. Importantly, while the per-Gaussian parameters are optimized, the global X-ray system response $\eta ( \bar { E } )$ is jointly optimized to achieve more efective reduction of metal artifacts. In the reconstruction phase, all optimized Gaussians are integrated to estimate the linear attenuation coeficient at each voxel location, which constitutes the final CT image output.

## 4.2. System Response and Polychromatic Attenuation

System Response. To accurately model polychromatic X-ray attenuation, it is essential to incorporate the spectral characteristics of the CT system. The system response is defined as the product of the X-ray source spectrum, filter transmission, and detector sensitivity [CAS∗19], which are collectively represented by the function $\eta ( E )$ . The source spectrum mainly consists of a continuous Bremsstrahlung component with several discrete characteristic peaks [IG24]. Since the contribution of these peaks spans only a narrow energy range, they can be neglected in our case. The filter shifts the Bremsstrahlung spectrum toward higher energies [AS18]; however, for beam hardening correction, the filter thickness is typically small and its efect can be approximated as negligible. The detector sensitivity decreases gradually with photon energy [WMM∗11], which, when combined with the Bremsstrahlung spectrum, results in an approximately linear decay. At higher energies, material attenuation becomes very weak, so the exponential term exp $\scriptstyle \left( - \int _ { L } \mu ( l , E ) d l \right)$ approaches unity. Consequently, spectral components in this range cannot be efectively distinguished by gradient-based optimization. Since the spectrum in this region is also close to linearly decaying, we approximate it with a constant plateau, which improves both computational robustness and physical plausibility. Based on these observations, we approximate the system response using a soft, diferentiable piecewise function composed of a linearly decaying component and a constant plateau:

$$
\eta ( \bar { E } ) = \ell _ { \bar { E } _ { t h } , r } ( \bar { E } ) \cdot \left( 1 - \sigma ( \bar { E } - \bar { E } _ { t h } ) \right) + \ell _ { \bar { E } _ { t h } , r } ( \bar { E } _ { t h } ) \cdot \sigma ( \bar { E } - \bar { E } _ { t h } ) ,\tag{6}
$$

![](images/a88263934e8bba35792b46bdc3a8a99de5bf9d328fc8fde8f6e56647ffeb644a.jpg)  
Figure 2: Method overview diagram. Our method consists oftwo types ofintegrals: projection and reconstruction. In the projection phase, the Gaussians are aggregated using our diferentiable polychromatic forward projector, and the resulting projections are compared against ground-truth measurements acquired from the CT system. Importantly, while the per-Gaussian parameters are optimized, the global X-ray system response � � is jointly optimized to achieve more efective reduction of metal artifacts. In the reconstruction phase, all optimized Gaussians are integrated to estimate the linear attenuation coeficient at each voxel location, which constitutes thefinal CT image output.

![](images/da1e53c012a69f9a287e76235d768d32969165698348ca2e530e4d48b8067dc6.jpg)  
Figure 3: Illustration ofour polychromatic attenuation model. The material-dependent linear attenuation coeficient $\mu ( \mathbf { x } , \bar { E } )$ is decomposed into two components: a constant Compton term $\mu _ { a } ( \mathbf { x } )$ and an energy-dependent photoelectric term $\mu _ { b } ( { \bf x } ) \bar { E } ^ { - 3 } .$ . Each Gaussian primitive is assigned these parameters to model spatially varying material attenuation.

where $\ell _ { \bar { E } _ { t h } , r }$ models the linear decay of the response with increasing energy, and � is a sigmoid function that smoothly transitions from the decaying region to a constant value. For numerical stability and bounded computation, we normalize the energy � (in keV) to $\bar { E } ,$ which ranges over $\left[ 1 - \gamma , 1 + \gamma \right]$ . The parameter � controls the relative intensity at the threshold energy $\bar { E } _ { t h }$ compared to the peak, while � defines the half-range of the normalized energy domain. Both � and $\bar { E } _ { t h }$ are optimized during reconstruction. We adopt this simple model for two main reasons: (1) it reduces model complexity and mitigates overfitting, and (2) it decreases the number of estimated parameters, enabling more eficient optimization.

Polychromatic Attenuation. To account for the energy-dependent attenuation characteristics of materials, we incorporate the two dominant interaction mechanisms in CBCT energy ranges: the photoelectric efect and Compton scattering, as discussed in Section 3.1. The photoelectric efect dominates at lower energies and exhibits an inverse cubic dependence on energy, while Compton scattering is more prevalent at higher energies [Spr12]. We model the total attenuation coeficient accordingly:

$$
\mu ( { \bf x } , \bar { E } ) = \mu _ { a } ( { \bf x } ) + \frac { 1 } { \bar { E } ^ { 3 } } \mu _ { b } ( { \bf x } ) ,\tag{7}
$$

where $\mu _ { a } ( \mathbf { x } )$ approximates the Compton scattering component and $\mu _ { b } ( { \bf x } )$ represents the photoelectric efect. The illustration of our model is shown in Figure 3. To ensure numerical stability during optimization, we approximate the Compton term with a constant value rather than an energy-dependent formulation. This simplification is justified because the Compton component remains nearly constant within the practical X-ray energy range used in medical CT, where the maximum photon energy is below 120 keV [Spr12, WCW∗23]. We incorporate this model into the reconstruction framework by assigning each Gaussian primitive a density $\rho _ { i }$ that varies with energy:

$$
\rho _ { i } ( \bar { E } ) = \rho _ { i } ^ { a } + \frac { 1 } { \bar { E } ^ { 3 } } \rho _ { i } ^ { b } ,\tag{8}
$$

where $\rho _ { i } ^ { a }$ and $\rho _ { i } ^ { b }$ correspond to the terms $\mu _ { a }$ and $\mu _ { b }$ in Equation (7) respectively. Using the Gaussian-based attenuation coeficient (Equation (4)) at $\bar { E }$ and Equation (8), we derive Equation (7):

$$
\begin{array} { l } { { \displaystyle \mu ( { \bf x } , \bar { E } ) = \sum _ { i } ^ { M } c _ { i } ( { \bf x } ) \rho _ { i } ( \bar { E } ) = \sum _ { i } ^ { M } c _ { i } ( { \bf x } ) \left( \rho _ { i } ^ { a } + \frac { 1 } { \bar { E } ^ { 3 } } \rho _ { i } ^ { b } \right) } \ ~ } \\ { { \displaystyle ~ = \sum _ { i } ^ { M } c _ { i } ( { \bf x } ) \rho _ { i } ^ { a } + \frac { 1 } { \bar { E } ^ { 3 } } \sum _ { i } ^ { M } c _ { i } ( { \bf x } ) \rho _ { i } ^ { b } } , \ ~ } \\ { { \displaystyle ~ \triangleq \mu _ { b } ( { \bf x } ) } } \end{array}\tag{9}
$$

where $c _ { i } \left( \mathbf { x } \right) = \exp \left( - \textstyle { \frac { 1 } { 2 } } ( \mathbf { x } - \mathbf { p } _ { i } ) ^ { T } \Sigma _ { i } ^ { - 1 } ( \mathbf { x } - \mathbf { p } _ { i } ) \right)$ that determines the shape of the $\mathrm { i ^ { t h } }$ Gaussian. Since the attenuation coeficient in Equation (9) is a multiplication of the geometric Gaussian weight and the density term, it is regularized by the term �<sub>�</sub> x during optimization.

## 4.3. Projection

We define the polychromatic forward projection model for beam hardening correction. By substituting the polychromatic attenuation model from Equation (9) into the forward model (Equation (3)), we

obtain:

$$
\begin{array} { l } { { P = \displaystyle - \log \int _ { \bar { \mathcal { E } } } \eta ( \bar { E } ) \exp \left( - \int _ { L } \mu ( l , \bar { E } ) d l \right) d \bar { E } } } \\ { { \displaystyle ~ = - \log \int _ { \bar { \mathcal { E } } } \eta ( \bar { E } ) \exp \left( - \int _ { L } \left( \mu _ { a } ( l ) + \frac { 1 } { \bar { E } ^ { 3 } } \mu _ { b } ( l ) \right) d l \right) d \bar { E } } } \\ { { \displaystyle ~ = \int _ { L } \mu _ { a } ( l ) d l - \log \int _ { \bar { \mathcal { E } } } \eta ( \bar { E } ) \exp \left( - \frac { 1 } { \bar { E } ^ { 3 } } \int _ { L } \mu _ { b } ( l ) d l \right) d \bar { E } , } } \end{array}\tag{10}
$$

<sup>where</sup> E<sup>¯</sup> <sup>is</sup> <sup>the</sup> <sup>integration</sup> <sup>range</sup> <sup>in</sup> <sup>the</sup> <sup>normalized</sup> <sup>energy</sup> <sup>levels</sup> <sup>(Fig-</sup> ure 3). To compute the projection on the projected coordinate xˆ using the Gaussian parameters, we plug in Equation (5) into Equation (10):

$$
P \left( \hat { \mathbf { x } } \right) \approx \sum _ { i } ^ { M } w _ { i } \left( \hat { \mathbf { x } } \right) \rho _ { i } ^ { a } - \log \int _ { \bar { \varepsilon } } \eta ( \bar { E } ) \exp \left( - \frac { 1 } { \bar { E } ^ { 3 } } \sum _ { i } ^ { M } w _ { i } \left( \hat { \mathbf { x } } \right) \rho _ { i } ^ { b } \right) d \bar { E } ,\tag{11}
$$

where $\begin{array} { r } { w _ { i } ( \hat { \mathbf { x } } ) = \sqrt { 2 \pi | \widetilde { \Sigma } _ { i } | \int | \widehat { \Sigma } _ { i } | } \exp \left( - \frac { 1 } { 2 } ( \hat { \mathbf { x } } - \hat { \mathbf { p } } _ { i } ) ^ { T } \hat { \Sigma } _ { i } ^ { - 1 } ( \hat { \mathbf { x } } - \hat { \mathbf { p } } _ { i } ) \right) } \end{array}$ is the weight at the projected coordinate xˆ of the $\mathrm { i ^ { t h } }$ Gaussian primitive. Note that since we approximate the Compton term as a constant variable $\mu _ { a } \left( \mathbf { x } \right)$ , we can take that term out of the exponential function as shown in the third line of the equation above, which is significantly beneficial in numerical stability during optimization. To compute $P \left( \hat { \mathbf { x } } \right)$ , we partition the range of normalized energy levels �<sup>¯</sup> by � to discretize Equation (11):

$$
\begin{array} { l } {displaystyle { P \left( \hat { \mathbf { x } } \right) = \sum _ { i } ^ { M } w _ { i } \left( \hat { \mathbf { x } } \right) \rho _ { i } ^ { a } } } \\ { \displaystyle { - \log \sum _ { k = 0 } ^ { N - 1 } \eta _ { k } \exp \left( - \frac { 1 } { \left( \left( \frac { 2 k } { N - 1 } - 1 \right) \gamma + 1 \right) ^ { 3 } } \sum _ { i } ^ { M } w _ { i } \left( \hat { \mathbf { x } } \right) \rho _ { i } ^ { b } \right) } , } \end{array}\tag{12}
$$

where $\eta _ { k }$ is the normalized system response weight such that $\begin{array} { r } { \sum _ { k = 0 } ^ { N - 1 } \eta _ { k } = 1 } \end{array}$

## 4.4. Optimization

We jointly optimize the system response parameters � and $\bar { E } _ { t h }$ along with the Gaussian parameters $\{ \rho _ { i } ^ { a } , \rho _ { i } ^ { b } , \mathbf { p } _ { i } , \Sigma _ { i } \mid i = 1 , . . . , M \}$ The total loss is a weighted combination of pixel-wise �<sub>1</sub> loss and structural similarity (SSIM):

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { 1 } ( P _ { G T } , P ) + \lambda _ { s s i m } \mathcal { L } _ { s s i m } ( P _ { G T } , P ) ,\tag{13}
$$

where $\lambda _ { s s i m }$ is fixed to 0.25 in all experiments. We include the gradient derivation of � with respect to the Gaussian parameters in the supplemental material to support backpropagation through the $\begin{array} { r } { \mathsf { R } ^ { 2 } \cdot \qquad } \end{array}$ -Gaussian CUDA pipeline. After optimization, the final voxelized attenuation field is obtained via Equation (9).

## 4.5. Implementation

Our method is implemented in PyTorch [PGM∗19] with CUDA [Gui13]. We set the normalized energy half-range � to $\frac { ( E _ { \mathrm { m a x } } - 1 0 ) } { ( E _ { \mathrm { m a x } } + 1 0 ) }$ , assuming that the minimum photon energy in the spectrum is 10 keV and that $E _ { \mathrm { m a x } }$ (the maximum photon energy) is known in advance. The number of discrete energy components � used for spectrum integration is fixed to 15. The parameters for the system response are initialized as $r = 0 . 2$ and $\bar { E } _ { t h } = 1 . 0$ . The Gaussian parameters in our model are initialized following the same procedure as the base method, $\textstyle \mathrm { \mathrm { R } } ^ { 2 } .$ Gaussian [ZLC∗24]. Specifically, we first reconstruct an initial volume using the FDK from the input projection images. From the reconstructed volume, we randomly sample 50,000 positions in the voxel grid where the voxel intensity is greater than 0.005. One Gaussian is then initialized at each of these sampled positions. For each Gaussian, the value of $\rho _ { a }$ is initialized by multiplying the voxel intensity by 0.075, and $\rho _ { b }$ is initialized with the voxel intensity multiplied by 0.0075. The initialization of the Gaussian covariance follows the same routine used in the original R2-Gaussian algorithm. We adopt R<sup>2</sup>-Gaussian [ZLC∗24]’s learning rate scheme, setting $( \rho _ { a } , \rho _ { b } )$ to (0.01, 0.1), and $( r , \bar { E } _ { t h } )$ to 0.001. Learning rates decay exponentially over 20k steps.

## 5. Validation

## 5.1. Synthetic Dataset Generation

To validate our method, we implement polychromatic CBCT synthetic dataset generator based on TIGRE [BDHS16], incorporating Poisson and Gaussian noise to mimic realistic acquisition conditions. This simulation pipeline follows established practices in prior work [LLP∗19, LLZL20, ZY18], and we further extend it to the CBCT configuration. For validating our simulator, we employ the OpenGATE package [ope25] that is based on Geant4 x-ray simulation library which supports Monte-Carlo (MC) projection. We first generate the X-ray spectrum using XrayPhysics library [KC23]. The spectrum ranges between 10–90keV with the take-of angle of 11 degree. The filter consists of aluminum with 0.5mm thickness. We set the material of the detector using GOS (O2SGd2) scintillator with 0.1mm thickness. We set the X-ray source’s shape as isotropic to simulate a conic beam, and the half angle of the cone is set to 13 degree. The total number of particles is set to 1 billion for the MC integration. We use a synthetic scene as Figure 4 (a). In the scene, three aluminum rods are inserted into the large cylinder, and the remaining part of the cylinder is filled with water, and the rest of the space consists of air. The volume’s physical size is 60 mm 60 mm 60 mm and the voxelized space has 256 256 256 voxels. We run both our projector and the MC-based one to generate raw detector signals and convert them into projection images by taking negative logarithm. Then, we compute the PSNR between them and get 32.94 dB (Figure 4 (b)), which shows strong agreement of our projection with the MC-based result.

![](images/f3aa41d6597217d2dc06405bbed2f9c42de8c8405c3b72696bd70604c366cb34.jpg)

![](images/82e9f92bf7749254ec11f5874fb50ed8dac3b4236d0eb61857220ced3b520877.jpg)

![](images/c4de7434139c624681e7239263834f4ac5c7a4f34524b0f325f9c4ca90e597ba.jpg)  
Figure 4: Validation of our synthetic projection image against one generated by Monte Carlo (MC) simulation using the OpenGATE package. A PSNR of32.94 dB indicates strong agreement between the two results.

## 5.2. System Response Model Selection

We adopt the simple yet physics-inspired approximation model for the system response described in Section 4.2, which has only two parameters. To validate the superiority of this choice, we evaluate reconstruction performance using more flexible alternatives, including a Gaussian Mixture Model (GMM) and a model with free parameters. As shown in Table 1, our model outperforms these alternatives.

Table 1: Validation ofour system response model against alternative flexible models (scene: Lung).
<table><tr><td>System response</td><td>PSNR3D</td><td>SSIM3D</td></tr><tr><td>GMM(w/ 5 Gaussians)</td><td>26.51</td><td>0.989</td></tr><tr><td>Free parameters</td><td>27.74</td><td>0.991</td></tr><tr><td>Our model</td><td>29.19</td><td>0.993</td></tr></table>

## 5.3. Generalizability to Multiple CT Systems

We validate the robustness of our method using synthetically generated projections that emulate diferent detector types and ranges of X-ray energies, as shown in Table 2. Our method shows consistent performance across various hardware configurations.

Table 2: Validation ofour method’s generalizability under various acquisition conditions (scene: Lung).
<table><tr><td>Case</td><td>Voltage (kV)</td><td>Filter (mm)</td><td>Detector</td><td>PSNR3D</td><td>SSIM3D</td></tr><tr><td>1</td><td>90</td><td>A1 0.5</td><td>GOS</td><td>29.12</td><td>0.993</td></tr><tr><td>2</td><td>90</td><td>Al 1.0</td><td>GOS</td><td>30.12</td><td>0.994</td></tr><tr><td>3</td><td>90</td><td>A1 0.5</td><td>CsI</td><td>29.38</td><td>0.993</td></tr><tr><td>4</td><td>120</td><td>A1 0.5</td><td>GOS</td><td>28.91</td><td>0.990</td></tr><tr><td>5</td><td>60</td><td>Cu 0.5</td><td>GOS</td><td>31.17</td><td>0.995</td></tr></table>

## 5.4. Real Attenuation Estimation

In the real scene Metal Rods (the third row in Figure 8), water and aluminum rods are used. After running our reconstruction, we segment the reconstructed volume into regions corresponding to water and aluminum via thresholding. We then compare the reconstructed attenuation values for each segment with the groundtruth linear attenuation coeficients of water and aluminum from the NIST database [SHS88] (at the center energy 50keV). As shown in the Figure 5, the estimated values are well aligned with the real-world attenuation coeficients in the homogeneous regions.

![](images/b5eaea3218f531cd09b05c49da4e995a2eecefa99adf47b29fa609531059f67e.jpg)  
Figure 5: Validation ofreconstructed attenuation coeficients against the NIST database [SHS88]. All values are reported in mm−<sup>1</sup>

## 6. Results

## 6.1. Dataset

![](images/70cec799c4484f6cd0bcbb39f48002ea23669ccd177392e4691d8fbc9403b198.jpg)  
Figure 6: Bruker SKYSCAN 1273 CBCT system.

We validate our method both on synthetic datasets and real datasets. For the synthetic evaluation, we utilize the LIDC dataset [AIMB∗11], the X-plant dataset [VDH∗22], and the ZCB100 dataset [LSZ∗16]. We evaluate our method on three synthetic CBCT phantoms (Lung, Teeth, and Broccoli) and five real-world scans: a walnut with metal pins (Walnut), a metal-embedded cylinder (Metal Rods), a chicken wrapped in metal wire (Chicken), a bell pepper with metal rivets (Bell Pepper), and a broccoli with metal rivets (Broccoli). Synthetic projections are generated using our validated simulator introduced in Section 5.1. Real projections are acquired using the Bruker SKYSCAN 1273 CBCT system (Figure 6) at a tube voltage of 90 kVp with a 0.5 mm aluminum filter. Each scan consists of 720 projections at a resolution of 512 512. We scan each specimen both with and without inserted metal materials to observe metal-induced beam hardening efects. To validate the reconstruction accuracy of our method on real objects, we fabricated a 3D-printed cylindrical phantom containing three aluminum alloy rods to induce severe artifacts (see the lower left of Figure 8). Figure 8 (third column) shows the volume reconstruction exhibiting prominent beam hardening patterns. All projection images are resized to the resolution of 512 512, and each scene has 720 projections.

## 6.2. Synthetic Experiment

We evaluate the performance of our method on three synthetic CBCT scenes—Lung, Teeth, and Broccoli—each containing embedded metallic objects made of iron (Fe), titanium (Ti), and aluminum (Al), respectively. The volumes are constructed with anatomical and organic structures, and cylindrical or spherical metal implants are synthetically inserted to induce beam hardening artifacts. CBCT projections are simulated under realistic polychromatic conditions using XrayPhysics [CAS∗19]. Figure 7 compares the reconstruction quality of various methods on these datasets. FDK [FDK84] exhibits severe streaking artifacts, even in slices not directly intersected by metal due to 3D projection efects inherent in cone-beam geometry. While methods such as LIMAR [KHE87], NMAR [MRL∗10], and Polyner [WCW∗23] partially mitigate artifacts, they still sufer from residual distortions or blurring. Park et al. [PSJ], originally developed for fan-beam CT, struggle to generalize to cone-beam settings, resulting in poor reconstruction quality. In contrast, our method produces high-fidelity reconstructions with accurate volume recovery and significantly suppressed metal-induced artifacts across all scenes. Table 3 quantitatively reports 3D PSNR and SSIM metrics. Our method consistently achieves the highest scores in all scenes, surpassing existing classical, learning-based, and physicsguided baselines. These results highlight the efectiveness of jointly modeling polychromatic attenuation and system response within a physics-inspired reconstruction framework.

Table 3: Quantitative results on synthetic CBCT datasets. 3D PSNR and SSIM scores for the Lung, Teeth, and Broccoli scenes corresponding to Figure 7. Our method achieves the highest accuracy across all datasets, demonstrating strong artifact suppression and structural fidelity.
<table><tr><td rowspan="2">Method</td><td colspan="2">Lung</td><td colspan="2">Teeth</td><td colspan="2">Broccoli</td></tr><tr><td>PSNR3D</td><td>SSIM3D</td><td>PSNR3D</td><td>SSIM3D</td><td>PSNR3D</td><td>SSIM3D</td></tr><tr><td>FDK [FDK84]</td><td>19.04</td><td>0.924</td><td>32.84</td><td>0.917</td><td>11.66</td><td>0.892</td></tr><tr><td>LIMAR [KHE87]</td><td>19.83</td><td>0.935</td><td>32.97</td><td>0.921</td><td>13.60</td><td>0.922</td></tr><tr><td>NMAR [MRL*10]</td><td>20.49</td><td>0.942</td><td>33.05</td><td>0.923</td><td>15.81</td><td>0.951</td></tr><tr><td>ACDNet [WLMZ22]</td><td>13.88</td><td>0.644</td><td>26.28</td><td>0.602</td><td>14.37</td><td>0.943</td></tr><tr><td>DICDNet [WLH*22]</td><td>13.48</td><td>0.605</td><td>25.33</td><td>0.423</td><td>13.93</td><td>0.933</td></tr><tr><td>OSCNet [WXZ*24]</td><td>13.28</td><td>0.592</td><td>25.34</td><td>0.422</td><td>13.83</td><td>0.931</td></tr><tr><td>Polyner [WCW*23]</td><td>20.43</td><td>0.955</td><td>30.18</td><td>0.959</td><td>18.88</td><td>0.987</td></tr><tr><td>Park et al. [PSJ]</td><td>13.59</td><td>0.699</td><td>23.15</td><td>0.524</td><td>5.29</td><td>0.526</td></tr><tr><td>Ours</td><td>29.19</td><td>0.993</td><td>35.97</td><td>0.992</td><td>23.22</td><td>0.995</td></tr></table>

## 6.3. Real Experiment

To evaluate the practical applicability of our method, we conduct experiments on real CBCT measurements acquired with the aforementioned X-ray scanner. We test five physical scenes as described in Section 6.1. All scenes except Metal Rods are also scanned separately without metal for reference, while the artifact-free version of the cylindrical phantom is synthesized in simulation for ground-truth comparison. Note that the outer shell of the cylindrical phantom is made of polylactide, whose attenuation is very similar to that of water. Figure 8 presents qualitative comparisons across multiple reconstruction methods. FDK and conventional algorithms (e.g., LIMAR [KHE87], NMAR [MRL∗10]) exhibit severe streaking and low-frequency distortions that obscure important structures in both axial and sagittal views. The data-driven method [WXZ∗24] fails to generalize to volume reconstruction due to out-of-distribution issues. Polyner [WCW∗23] achieves reasonable reconstructions but leaves residual artifacts, while Park et al. [PSJ] struggles to adapt to the CBCT setting. In contrast, our method consistently produces volumetrically accurate reconstructions with sharp structural boundaries and substantially fewer artifacts across objects. These results highlight the robustness and real-world transferability of our framework, even under hardware and acquisition conditions not seen during development. Figure 9 further shows results for three organic objects across joint reconstruction and metal artifact reduction methods. Odd columns display representative slices reconstructed by each method, with red boxes marking regions of interest; even columns show the corresponding zoom-in patches. Additional slices are provided in the supplemental material. Our method produces clearer reconstructions with fewer artifacts compared to prior approaches.

## 6.4. Computation Time

We compare the computation time of our method with state-of-theart joint reconstruction and metal artifact correction approaches in Table 4. All measurements are performed on an Intel Xeon 4214R

CPU and an NVIDIA RTX A6000 GPU. Our method achieves an order-of-magnitude speedup across all scenes.

Table 4: Comparisons ofcomputation time. Our method isfaster than the other joint reconstruction and metal artifact reduction methods.
<table><tr><td></td><td>Scene</td><td>Polyner [WCW*23]</td><td>Park et al. [PSJ]</td><td>Ours</td></tr><tr><td rowspan="3">Synetc</td><td>Broccoli</td><td>1h 35m 58s</td><td>1h 33m 41s</td><td>16m 08s</td></tr><tr><td>Lung</td><td>1h 18m 19s</td><td>1h 15m 35s</td><td>22m 16s</td></tr><tr><td>Teeth</td><td>1h 19m 50s</td><td>1h 18m 34s</td><td>23m 38s</td></tr><tr><td rowspan="5">Real</td><td>Walnut</td><td>1h 11m 58s</td><td>1h 10m 20s</td><td>17m 51s</td></tr><tr><td>Metal Rods</td><td>1h 27m 14s</td><td>1h 37m 56s</td><td>30m 41s</td></tr><tr><td>Broccoli</td><td>2h 06m 55s</td><td>1h 59m 16s</td><td>16m 16s</td></tr><tr><td>Bell Pepper</td><td>2h 01m 00s</td><td>2h 10m 11s</td><td>15m 33s</td></tr><tr><td>Chicken</td><td>2h 02m 55s</td><td>2h 09m 43s</td><td>16m 35s</td></tr></table>

## 7. Ablation Study

## 7.1. Component Contribution

To evaluate the efectiveness of each core component in our framework, we perform an ablation study on the Lung scene by systematically adding the proposed modules to the baseline R<sup>2</sup>-Gaussian algorithm. Specifically, we investigate the individual and combined impacts of the system response model and the polychromatic attenu ation model. As shown in Table 5, incorporating the system response model alone yields a modest improvement in both PSNR and SSIM, suggesting that response-aware weighting improves projection fidelity. In contrast, the addition of the polychromatic attenuation model leads to a substantial gain in reconstruction quality, highlighting the importance of accurate energy-dependent material modeling in suppressing beam hardening artifacts. When both modules are integrated, we observe the highest reconstruction performance, achieving a PSNR of 29.19 and an SSIM of 0.993, thereby confirming the complementary nature of the two components.

Table 5: Component contribution analysis. We examine the individual contributions of the system response model and polychromatic attenuation model. Both components contribute positively to reconstruction accuracy, with their combination yielding the best overall performance (scene: Lung).
<table><tr><td>Module</td><td>PSNR3D</td><td>SSIM3D</td></tr><tr><td>Baseline</td><td>20.68</td><td>0.954</td></tr><tr><td>Baseline with response model</td><td>21.21</td><td>0.963</td></tr><tr><td>Baseline with attenuation model</td><td>27.73</td><td>0.991</td></tr><tr><td>Baseline with both</td><td>29.19</td><td>0.993</td></tr></table>

## 7.2. Number of Spectrum Components

We conduct an experiment to determine the optimal number of spectrum components (�). As shown in Table 6, the best reconstruction quality is achieved when � = 15. When � exceeds 15, the performance starts getting saturated and slightly degraded.

Park et al.  
![](images/f6bf8326229600b7c669feebf3e901e6743c8c4bf5166b17398668c72d31a52f.jpg)  
FDK  
Polyner  
Ours  
GT  
Figure 7: Qualitative comparison on synthetic CBCT datasets. Reconstruction results on Lung, Teeth, and Broccoli scenes with metallic implants ofdiferent materials. Our method achieves superior artifact suppression and structuralfidelity compared to prior methods.

Table 6: Performance impact ofthe number ofspectral components. The best performance is achieved at � = 15 (scene: Lung).
<table><tr><td># of Spectrum Components (N)</td><td>PSNR3D</td><td>SSIM3D</td></tr><tr><td>7</td><td>27.86</td><td>0.988</td></tr><tr><td>15</td><td>29.19</td><td>0.993</td></tr><tr><td>31</td><td>28.70</td><td>0.993</td></tr><tr><td>63</td><td>28.25</td><td>0.992</td></tr></table>

## 7.3. Performance Comparison to Baseline

We compare our method against the baseline approach [ZLC∗24] on scenes containing metal artifacts. By incorporating a polychromatic attenuation model and system response, our method consistently achieves superior performance across diferent scenes, as shown in Table 7. We also compare the computation time between our

Table 7: Comparison ofthe baseline [ZLC∗24] and our method on PSNR3D and SSIM3D across diferent scenes.
<table><tr><td rowspan="2">Scene (synthetic)</td><td colspan="2">Baseline [ZLC*24]</td><td colspan="2">Ours</td></tr><tr><td>PSNR3D</td><td>SSIM3D</td><td>PSNR3D</td><td>SSIM3D</td></tr><tr><td>Broccoli</td><td>17.67</td><td>0.984</td><td>23.22</td><td>0.995</td></tr><tr><td>Lung</td><td>20.68</td><td>0.954</td><td>29.19</td><td>0.993</td></tr><tr><td>Teeth</td><td>33.53</td><td>0.964</td><td>35.97</td><td>0.992</td></tr></table>

Walnut

Metal Rods

![](images/b9d412e6626c3a6be50b0256ac3b96c0596d38bf6645228d7ded981745334008.jpg)

![](images/814c58961c4d109197a16b9e98e4fd53ec291d111487e1c022897d45676b2fcd.jpg)  
Reference

![](images/36e3510f4399e7224a9804fa0a0a21ec584b561b1d6fe72533a9d59208583cf7.jpg)  
FDK

![](images/6802ddabde874477d47f0a535dce56e3cfeaacc38659a657db9c2f5044e367bd.jpg)  
LIMAR

![](images/c42dda18b57349f7eb66c893a94a60ffc7ded5e711a99a6fa65967099b5d8669.jpg)  
OSCNET

![](images/45e2eb748dd3133b901361b287c5c40e43e1cd796ad6db29c26806d5f3a15932.jpg)  
Polyner

![](images/f3ed5d1ff88305b4ddc60d5d29f53fe27def5c64a3cf9d6118b9fbb57aa59f88.jpg)  
Park et al.

![](images/0c6ce2e7fe77bb6f306302725caefea8161d9213dafcf3e795b479c3eb73a38b.jpg)  
Ours

Figure 8: Qualitative comparison on real CBCT datasets. Reconstruction resultsfor walnut and Metal Rods phantoms. Reference volumes are obtainedfrom either separate scans or synthetic generation. Our method achieves clearer structures and stronger artifact suppression compared to prior approaches

method and the baseline [ZLC∗24] as Table 8. Due to the additional computation of the polychromatic model in our forward projection compared to the baseline method, each optimization iteration takes slightly longer. To mitigate this overhead, we implement several eficiency-oriented strategies: we analytically simplify the forward projection and its corresponding backward gradient computations, and eliminate redundant operations to reduce computation costs. For faster convergence, we conduct extensive experiments to identify optimal learning rates for $\rho _ { a }$ and $\rho _ { b }$ . As a result, despite the periteration overhead, our model converges within 20K iterations, whereas the baseline method typically requires around 30K. This demonstrates that, with proper hyperparameter tuning, our model achieves comparable or even faster convergence.

Table 8: Convergence time comparison between the baseline and our method across diferent scenes.
<table><tr><td>Type</td><td>Scene</td><td>Baseline [ZLC*24]</td><td>Ours</td></tr><tr><td>Synthetic</td><td>Broccoli Lung Teeth</td><td>18m 44s 20m 17s 21m 35s</td><td>16m 08s 22m 16s 23m 38s</td></tr><tr><td>Real</td><td>Walnut Metal Rods Broccoli Bell Pepper</td><td>20m 52s 27m 07s 17m 52s 19m 53s 21m 59s</td><td>17m 51s 30m 41s 16m 16s 15m 33s</td></tr></table>

## 7.4. Sensitivity on �

We normalize the energy spectrum so that its center becomes 1 and the minimum and maximum values are set to 1 � and 1 �, respectively. If � is not set correctly and does not match the actual spectrum range, reconstruction quality may deteriorate. We evaluate the sensitivity by running our method using projections (acquired under a 10–90 keV spectrum) with various � values corresponding to maximum energies of 60, 90, 120, and 150 keV. As shown in Table 9, the model shows the best performance when � is set near the true spectrum range (i.e., 90 keV), which validates the significance of the choice of �.

Table 9: Efect ofvarying � and corresponding maximum energy $E _ { m a x }$ on reconstruction quality (scene: Broccoli).
<table><tr><td>γ</td><td>Corresponding  $E _ { \mathrm { m a x } }$  (keV)</td><td>PSNR3D</td><td>SSIM3D</td></tr><tr><td>0.714</td><td>60</td><td>21.85</td><td>0.9943</td></tr><tr><td>0.800</td><td>90</td><td>23.22</td><td>0.9953</td></tr><tr><td>0.846</td><td>120</td><td>22.83</td><td>0.9946</td></tr><tr><td>0.875</td><td>150</td><td>22.38</td><td>0.9939</td></tr></table>

## 8. Discussion

Due to limitations in accessing clinical CT datasets, we were unable to conduct comprehensive evaluations across diverse real medical use cases. Instead, we have validated our method on synthetic humanbody datasets containing metal implants (e.g., in soft tissue and dental regions). Additionally, we have tested on a real-world dataset where metallic wires were inserted into a chicken to induce metal artifacts in biological tissues. These experiments show that our method generalizes well to various anatomical objects and metals. We are actively seeking access to clinical data to further investigate the generalizability of our method under real-world scenarios for future work.

While our method is built upon a physics-inspired polychromatic projection model, its assumptions may break down under extreme conditions such as photon starvation or partial volume efects. Also, it relies on approximations of system response and material attenuation models. As a result, the reconstructed linear attenuation coeficients may deviate from the true physical values for specific materials.

To enable fair comparison, we re-implemented Polyner [WCW∗23] and Park et al. [PSJ], as no CBCT-compatible public code was available. After careful tuning, our Polyner reproduction matched or exceeded the original performance in CBCT settings. In contrast,

![](images/ee30f95e98f97fcbdf5c80479fa047c5ee1c676a1b95c0fb16af4198586e705b.jpg)  
Figure 9: Qualitative comparison on real CBCT datasets. The slicing locations are displayed on each picture ofthe real objects in thefirst row. Odd columns show representative slices reconstructed by each method, with red boxes highlighting regions of interest. Even columns display the corresponding zoom-in patches. Additional slices are provided in the supplemental document. Our method produces clearer reconstructions with fewer artifacts compared to prior approaches.

Park et al.’s fan-beam-based method showed limited generalization to CBCT, highlighting the challenge of adapting fan-beam approaches to cone-beam geometries under severe artifacts. Details of the comparison implementation are provided in the supplemental document.

## 9. Conclusion

We have presented a CBCT reconstruction method that reduces metal-induced beam hardening artifacts via polychromatic modeling and Gaussian splatting. Our framework jointly estimates volume structure, system response, and attenuation parameters—without requiring metal masks, paired supervision, nor spectral priors. Extensive evaluations on synthetic and real data confirm that our method outperforms state-of-the-art baselines in reconstruction accuracy and artifact suppression. We also release new CBCT datasets with realistic artifacts to support future research in artifact-resilient reconstruction.

## Acknowledgements

Min H. Kim acknowledges the Samsung Research Funding & Incubation Center (SRFC-IT2402-02), the Korea NRF grant (RS-2024-00357548), the MSIT/IITP of Korea (RS-2022-00155620, RS-2024-00398830, RS-2024-00436680, and 2017-0-00072), and Microsoft Research Asia.

## References

[AIMB∗11] Armato III S. G., McLennan G., Bidaut L., McNitt-Gray M. F., Meyer C. R., Reeves A. P., Zhao B., Aberle D. R., Henschke C. I., Hoffman E. A., et al.: The lung image database consortium (lidc) and image database resource initiative (idri): a completed reference database of lung nodules on ct scans. Medical physics 38, 2 (2011), 915–931. 6

[AK84] Andersen A. H., Kak A. C.: Simultaneous algebraic reconstruction technique (sart): A superior implementation of the art algorithm. Ultrasonic Imaging 6, 1 (1984), 81–94. 2

[AS18] Ahmed O., Song Y.: A review of common beam hardening correction methods for industrial x-ray computed tomography. Sains Malaysiana 47, 8 (2018), 1883–1890. 3

[BDHS16] Biguri A., Dosanjh M., Hancock S., Soleimani M.: Tigre: a matlab-gpu toolbox for cbct image reconstruction. Biomedical Physics & Engineering Express 2, 5 (2016), 055010. 5

[BT09] Beck A., Teboulle M.: A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM J. Imaging Sci. 2, 1 (2009), 183–202. 2

[CAS∗19] Champley K. M., Azevedo S. G., Seetho I. M., Glenn S. M., McMichael L. D., Smith J. A., Kallman J. S., Brown W. D., Martz H. E.: Method to extract system-independent material properties from dual-energy x-ray ct. IEEE Transactions on Nuclear Science 66, 3 (2019), 674–686. 3, 6

[CWY∗24] Cai Y., Wang J., Yuille A., Zhou Z., Wang A.: Structureaware sparse-view x-ray 3d reconstruction. In CVPR (2024). 2

[FDK84] Feldkamp L. A., Davis L. C., Kress J. W.: Practical cone-beam algorithm. Journal of the Optical Society of America A 1, 6 (1984), 612–619. 1, 2, 6, 7

[Gil72] Gilbert P.: Iterative methods for the three-dimensional reconstruction of an object from projections. Journal ofTheoretical Biology 36, 1 (1972), 105–117. 2

[Gui13] Guide D.: Cuda c programming guide. NVIDIA, July 29 (2013), 31. 5

[Gul79] Gullberg G. T.: The reconstruction of fan-beam data by filtering the back-projection. Computer Graphics and Image Processing 10, 1 (1979), 30–47. 2

[IG24] Iniewski K., Gadey H.: Emerging Radiation Detection: Technology and Applications. Springer Nature, 2024. 3

[KC23] Kim H., Champley K.: Diferentiable forward projector for x-ray computed tomography. arXiv preprint arXiv:2307.05801 (2023). 5

[KHE87] Kalender W. A., Hebel R., Ebersberger J.: Reduction of ct artifacts caused by metallic implants. Radiology 164, 2 (1987), 576–577. 2, 6, 7

[KKLD23] Kerbl B., Kopanas G., Leimkühler T., Drettakis G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42, 4 (2023), 139–1. 3

[LLP∗19] Lin W.-A., Liao H., Peng C., Sun X., Zhang J., Luo J., Chellappa R., Zhou S. K.: Dudonet: Dual domain network for ct metal artifact reduction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2019), pp. 10512–10521. 5

[LLZL20] Liao H., Lin W.-A., Zhou S. K., Luo J.: Adn: Artifact disentanglement network for unsupervised metal artifact reduction. IEEE Transactions on Medical Imaging 39, 3 (2020), 634–643. 2, 5

[LMFL12] Liu Y., Ma J., Fan Y., Liang Z.: Adaptive-weighted total variation minimization for sparse data toward low-dose x-ray computed tomography image reconstruction. Physics in Medicine & Biology 57, 23 (nov 2012), 7923. 2

[LSZ∗16] Luo T., Shi C., Zhao X., Zhao Y., Xu J.: Automatic synthesis of panoramic radiographs from dental cone beam computed tomography data. PloS one 11, 6 (2016), e0156976. 6

[MESK22] Müller T., Evans A., Schied C., Keller A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph. 41, 4 (July 2022), 102:1–102:15. 3

[MRL∗10] Meyer E., Raupach R., Lell M., Schmidt B., Kachelrieß M.: Normalized metal artifact reduction (nmar) in computed tomography. Medical physics 37, 10 (2010), 5482–5493. 2, 6, 7

[ope25] Opengate documentation, 2025. URL: https://opengate. readthedocs.io/en/latest/introduction.html. 5

[PGM∗19] Paszke A., Gross S., Massa F., Lerer A., Bradbury J., Chanan G., Killeen T., Lin Z., Gimelshein N., Antiga L., Desmaison A., Kopf A., Yang E., DeVito Z., Raison M., Tejani A., Chilamkurthy S., Steiner B., Fang L., Bai J., Chintala S.: Pytorch: An imperative style, high-performance deep learning library. 5

[PSJ] Park H. S., Seo J. K., Jeon K.: Implicit neural representation-based method for metal-induced beam hardening artifact reduction in x-ray ct imaging. Medical Physics. 1, 2, 6, 7, 9

[SHS88] Saloman E., Hubbell J., Scofield J.: X-ray attenuation cross sections for energies 100 ev to 100 kev and elements z= 1 to z= 92, 1988. 6

[SP08] Sidky E. Y., Pan X.: Image reconstruction in circular conebeam computed tomography by constrained, total-variation minimization. Physics in Medicine & Biology 53, 17 (2008), 4777. 2

[Spr12] Sprawls P.: Interaction of radiation with matter. Physical Principles ofMedical Imaging (2012), 141–157. 4

[VDH∗22] Verboven P., Dequeker B., He J., Pieters M., Pols L., Tempelaere A., Van Doorselaer L., Van Cauteren H., Verma U., Xiao H., et al.: www. x-plant. org-the ct database of plant organs. In 6th Symposium on X-ray Computed Tomography: Inauguration of the KU Leuven XCT Core Facility, Location: Leuven, Belgium (2022). 6

[WCW∗23] Wu Q., Chen L., Wang C., Wei H., Zhou S. K., Yu J., Zhang Y.: Unsupervised polychromatic neural representation for CT metal artifact reduction. In Thirty-seventh Conference on Neural Information Processing Systems (2023). 1, 2, 4, 6, 7, 9

[WLH∗22] Wang H., Li Y., He N., Ma K., Meng D., Zheng Y.: Dicdnet: Deep interpretable convolutional dictionary network for metal artifact reduction in ct images. IEEE Transactions on Medical Imaging 41, 4 (2022), 869–880. 2, 7

[WLMZ22] Wang H., Li Y., Meng D., Zheng Y.: Adaptive convolutional dictionary network for ct metal artifact reduction. In The 31st International Joint Conference on Artificial Intelligence (2022), IEEE. 2, 7

[WLZ∗21] Wang H., Li Y., Zhang H., Chen J., Ma K., Meng D., Zheng Y.: Indudonet: An interpretable dual domain network for ct metal artifact reduction. In International Conference on Medical Image Computing and Computer-Assisted Intervention (2021), Springer, pp. 107–118. 2

[WMM∗11] Wang X., Meier D., Mikkelsen S., Maehlum G., Wagenaar D., Tsui B., Patt B., Frey E.: Microct with energy-resolved photon counting detectors. Physics in Medicine & Biology 56, 9 (2011), 2791. 3

[WXZ∗24] Wang H., Xie Q., Zeng D., Ma J., Meng D., Zheng Y.: Oscnet: Orientation-shared convolutional network for ct metal artifact learning. IEEE Transactions on Medical Imaging 43, 1 (2024), 489–502. 2, 7

[ZLC∗24] Zha R., Lin T. J., Cai Y., Cao J., Zhang Y., Li H.: R<sup>2</sup>-gaussian: Rectifying radiative gaussian splatting for tomographic reconstruction. In Advances in Neural Information Processing Systems (NeurIPS) (2024). 2, 3, 5, 8, 9

[ZY18] Zhang Y., Yu H.: Convolutional neural network based metal artifact reduction in x-ray computed tomography. IEEE transactions on medical imaging 37, 6 (2018), 1370–1381. 5

[ZZL22] Zha R., Zhang Y., Li H.: Naf: neural attenuation fields for sparse-view cbct reconstruction. In International Conference on Medical Image Computing and Computer-Assisted Intervention (2022), Springer, pp. 442–452. 3

# Supplementary Material: Splat-based Metal Artifact Reduction in Cone-Beam CT via Polychromatic Modeling

Kiseok Choi Inchul Kim Jaemin Cho Hyeongjun Cho Min H. Kim

KAIST

## 1. Overview of Our Optimization Pipeline

We illustrate our overall optimization pipeline in Figure 1. Our method consists of two stages, following a structure similar to the baseline approach [ZLC∗24]. In the first stage, a conventional reconstruction method, such as FDK [FDK84], is employed to obtain an initial volumetric reconstruction. From this volume, we randomly sample a subset of voxel locations and initialize 3D Gaussian primitives with their centers placed at the sampled positions. These Gaussians are forward-projected onto the projection image domain under a cone-beam geometry using the given system response parameters. The resulting synthetic projections are compared with the measured projection images, and the reconstruction loss is backpropagated to jointly optimize the corresponding 3D Gaussian parameters and the system response parameters. During this iterative optimization, our adaptive density control strategy is applied to prune the negligible Gaussians and to split or clone Gaussians in order to better represent the underlying volume. The output of the first stage is a set of optimized 3D Gaussian parameters. In the second stage, the optimized Gaussians are evaluated at the voxel locations of interest, from which the final reconstructed volume is obtained.

![](images/73e28676b2236687d57d342d3c639e54c428e801d417f9b8662cc7440fac37f5.jpg)  
Figure 1: Overview of our optimization pipeline.

2 of8 K. Choi, I. Kim, J. Cho, H. Cho, & M.H. Kim/Supplementary Material: Splat-basedMetalArtifactReduction in Cone-Beam CTvia Polychromatic Modeling

## 2. Additional Ablation Results

Figure 2 provides additional qualitative results that support the ablation experiments reported by Table 5 in the main paper. The numbers shown in the upper-right corners indicate the corresponding 3D PSNR values. As shown in this figure, the baseline reconstruction is noticeably blurred and sufers from severe metal artifacts. When only our response model is applied, finer details are recovered; however, prominent metal artifacts remain since the beam-hardening is not taken into account. Applying only the attenuation model significantly reduces these artifacts, although minor residual artifacts persist near metallic regions. When both models are jointly employed, metal artifacts are further suppressed and the overall intensity level closely matches the ground-truth volume, resulting in the highest PSNR among all ablation settings.

![](images/951dc185fc51e6e4f23e81870a3d865da55dca1929bbb5bf063b75d30d1ea67f.jpg)  
Ground truth

![](images/d1bda2804f7e86ba6ac76b86b0e07638dacc24ce0f6806917a8dc7b86d483401.jpg)  
Baseline

![](images/30288b15a413c4092eb7f5c2f53cf2f6f42144f46d81c308aa6e804d201917a5.jpg)  
With response model

![](images/918ce2fa546f207631856be63e8f559f77381b803fc0721af4b62e7c7617df74.jpg)  
With attenuation model

![](images/727918b0e76e09f67989c4fb67594b530b51fe22bcf8107be22961eb7b43395e.jpg)  
With both models  
Figure 2: Qualitative results of the ablations reported by Table 5 in the main paper.

## 3. Derivation of Backward Propagation

We implement a custom CUDA pipeline to enable automatic diferentiation of the parameters in the discrete polychromatic forward projection model using the Gaussian primitives as Equation (1):

$$
P ( \hat { \mathbf { x } } ) = \sum _ { i } ^ { M } s _ { i } G _ { i } ( \hat { \mathbf { x } } ) \rho _ { i } ^ { a } - \log \sum _ { k = 0 } ^ { N - 1 } \eta _ { k } \exp \left( - \frac { 1 } { \left( \left( \frac { 2 k } { N - 1 } - 1 \right) \gamma + 1 \right) ^ { 3 } } \sum _ { i } ^ { M } s _ { i } G _ { i } ( \hat { \mathbf { x } } ) \rho _ { i } ^ { b } \right) ,\tag{1}
$$

where $\begin{array} { r } { s _ { i } = \sqrt { \frac { 2 \pi | \tilde { \Sigma } _ { i } | } { | \Sigma _ { i } | } } , G _ { i } ( \hat { \bf x } ) = \exp \left( - \frac { 1 } { 2 } ( \hat { \bf x } - \hat { \bf p } _ { i } ) ^ { T } \tilde { \Sigma } _ { i } ^ { - 1 } ( \hat { \bf x } - \hat { \bf p } _ { i } ) \right) , \sum _ { k = 0 } ^ { N - 1 } \eta _ { k } = 1 . } \end{array}$

$\textstyle \mathbf { R } ^ { 2 } .$ -Gaussian [ZLC∗24] consists of the Preprocessor that transforms the position and shape of each Gaussian, and the Rasterizer that computes the intensity projected onto each detector pixel. While the Preprocessor reuses the original implementation, we design a

K. Choi, I. Kim, J. Cho, H. Cho, & M.H. Kim/Supplementary Material: Splat-basedMetalArtifactReduction in Cone-Beam CTvia Polychromatic Modeling3 of8

new Rasterizer model and manually derive the gradients with respect to each intermediate parameter as Equations (2)–(6):

$$
\left. \frac { d \mathcal { L } } { d \rho _ { i } ^ { a } } \right| _ { \hat { \mathbf { x } } } = \frac { d \mathcal { L } } { d P ( \hat { \mathbf { x } } ) } \cdot \frac { d P ( \hat { \mathbf { x } } ) } { d \rho _ { i } ^ { a } } = \frac { d \mathcal { L } } { d P ( \hat { \mathbf { x } } ) } \cdot \left( s _ { i } G _ { i } ( \hat { \mathbf { x } } ) \right) ,\tag{2}
$$

$$
\frac { d \mathcal { L } } { d \rho _ { i } ^ { b } } \Bigg \vert _ { \hat { \mathbf { x } } } = \frac { d \mathcal { L } } { d P ( \hat { \mathbf { x } } ) } \cdot \frac { d P ( \hat { \mathbf { x } } ) } { d \rho _ { i } ^ { b } } = \frac { d \mathcal { L } } { d P ( \hat { \mathbf { x } } ) } \cdot \left( - \frac { 1 } { A ( \hat { \mathbf { x } } ) } s _ { i } G _ { i } ( \hat { \mathbf { x } } ) \sum _ { k = 0 } ^ { N - 1 } \left( B _ { k } ( \hat { \mathbf { x } } ) C _ { k } \right) \right) ,\tag{3}
$$

$$
\frac { d \mathcal { L } } { d s _ { i } } \bigg \vert _ { \hat { \mathbf { x } } } = \frac { d \mathcal { L } } { d P ( \hat { \mathbf { x } } ) } \cdot \frac { d P ( \hat { \mathbf { x } } ) } { d s _ { i } } = \frac { d \mathcal { L } } { d P ( \hat { \mathbf { x } } ) } \cdot \left( \rho _ { i } ^ { a } G _ { i } ( \hat { \mathbf { x } } ) - \frac { 1 } { A ( \hat { \mathbf { x } } ) } \rho _ { i } ^ { b } G _ { i } ( \hat { \mathbf { x } } ) \sum _ { k = 0 } ^ { N - 1 } \left( B _ { k } ( \hat { \mathbf { x } } ) C _ { k } \right) \right) ,\tag{4}
$$

$$
\left. \frac { d \mathcal { L } } { d G _ { i } } \right| _ { \hat { \mathbf { x } } } = \frac { d \mathcal { L } } { d P ( \hat { \mathbf { x } } ) } \cdot \frac { d P ( \hat { \mathbf { x } } ) } { d G _ { i } ( \hat { \mathbf { x } } ) } = \frac { d \mathcal { L } } { d P ( \hat { \mathbf { x } } ) } \cdot \left( \rho _ { i } ^ { a } s _ { i } - \frac { 1 } { A ( \hat { \mathbf { x } } ) } \rho _ { i } ^ { b } s _ { i } \sum _ { k = 0 } ^ { N - 1 } \left( B _ { k } ( \hat { \mathbf { x } } ) C _ { k } \right) \right) ,\tag{5}
$$

$$
\left. \frac { d \mathcal { L } } { d \eta _ { k } } \right| _ { \hat { \mathbf { x } } } = \frac { d \mathcal { L } } { d P ( \hat { \mathbf { x } } ) } \cdot \frac { d P ( \hat { \mathbf { x } } ) } { d \eta _ { k } } = \frac { d \mathcal { L } } { d P ( \hat { \mathbf { x } } ) } \cdot \left( - \frac { 1 } { A ( \hat { \mathbf { x } } ) } \frac { B _ { k } ( \hat { \mathbf { x } } ) } { \eta _ { k } } \right) ,\tag{6}
$$

where $\begin{array} { r l } & { A ( \hat { \bf x } ) = \sum _ { k = 0 } ^ { N - 1 } B _ { k } ( \hat { \bf x } ) , B _ { k } ( \hat { \bf x } ) = \eta _ { k } \exp \left( - C _ { k } \sum _ { i } ^ { M } \rho _ { i } ^ { b } s _ { i } G _ { i } ( \hat { \bf x } ) \right) , C _ { k } = - \frac { 1 } { \left( \left( \frac { 2 k } { N - 1 } - 1 \right) \gamma + 1 \right) ^ { 3 } } } \end{array}$ . All these equations are then implemented in the CUDA backward function. The final gradient with respect to $\eta _ { k }$ is obtained by summing the pixel-wise partial derivatives:

$$
\frac { d \mathcal { L } } { d \eta _ { k } } = \sum _ { \hat { \mathbf { x } } } \left. \frac { d \mathcal { L } } { d \eta _ { k } } \right| _ { \hat { \mathbf { x } } } = \sum _ { \hat { \mathbf { x } } } \frac { d \mathcal { L } } { d P ( \hat { \mathbf { x } } ) } \cdot \left( - \frac { 1 } { A ( \hat { \mathbf { x } } ) } \cdot \frac { B _ { k } ( \hat { \mathbf { x } } ) } { \eta _ { k } } \right)\tag{7}
$$

## 4. Comparison Implementation Detail

Since both Polyner [WCW∗23] and Park et al. [PSJ] are originally designed for the fan-beam geometry, we extend their implementations to support the cone-beam configuration for fair comparison with our method. Specifically, we constrain the sampling range that is originally set to (-SOD,SOD) where SOD is Source-to-Object Distance, to a narrower interval that better matches the actual object region to reduce sensitivity to variations across diferent datasets. To improve the reconstruction quality under the cone-beam setting, which is more challenging to reconstruct a volume due to its 3D geometry of beams, we also increase the depth of the network and expand the hash encoding parameters used in the reconstruction. As the oficial implementation of Park et al. is not publicly available, we instead adapt the extended cone-beam implementation of Polyner and fine-tune several network and training-related parameters for fair comparison. For learning-based methods such as ACDNet, DICDNet, and OSCNet, we employ publicly available pre-trained models. As these baselines only support metal artifact reduction on 2D slices, we first reconstruct a 3D volume using FDK and then apply each model independently to the individual slices.

## 5. Geometry Configuration

The geometry configurations for the datasets are provided in Tables 1–2. To validate the robustness of our method, we conduct experiments under various geometric setups.

## 6. GPU Memory Usage

We report the memory footprint of the baseline and our method in Table 3. Although our model is more complex than the baseline, its memory requirement increases only marginally, indicating that the additional polychromatic features do not compromise memory eficiency.

## 7. Additional Discussion

We compute the PSNR and SSIM for the 3D volume to compare the reconstruction performance on the synthetic dataset. For the PSNR calculation, the ground truth (GT) is determined based on the center energy, and following the evaluation methods of existing metal artifact reduction techniques [WCW∗23] and [PSJ], PSNR is measured for all regions except for the metal mask area. Since there is no proper GT available for the real dataset, we do not conduct quantitative evaluations, but qualitative comparison evaluations are done instead.

## 8. Additional Result

Figure 3 presents the objects for which we provide additional results. Figures 4–6 show qualitative comparisons of our method against FDK [FDK84] and the state-of-the-art joint reconstruction and metal artifact reduction methods: Polyner [WCW∗23], and Park et al. [PSJ].

Table 1: The scanning configuration of our synthetic datasets.
<table><tr><td rowspan="2">Parameters</td><td colspan="3">Synthetic Dataset</td></tr><tr><td>Broccoli</td><td>Lung</td><td>Teeth</td></tr><tr><td>Source voltage (kV)</td><td>90</td><td>←</td><td>←</td></tr><tr><td>Aluminium filter (mm)</td><td>0.5</td><td>←</td><td>←</td></tr><tr><td>Metal material</td><td>Aluminum</td><td>Steel</td><td>Titanium</td></tr><tr><td>Volume resolution</td><td>256×256×256</td><td>←</td><td>←</td></tr><tr><td>Volume size (mm)</td><td>50.0×50.0×50.0</td><td>←</td><td>←</td></tr><tr><td>Angle range (°)</td><td>[0, 360)</td><td>←</td><td>←</td></tr><tr><td>Number of angles</td><td>720</td><td>←</td><td>←</td></tr><tr><td>Detector resolution</td><td>512×512</td><td>←</td><td>←</td></tr><tr><td>Detector size (mm)</td><td>145.485</td><td>←</td><td>←</td></tr><tr><td>Source-to-object distance (mm)</td><td>200.962</td><td>←</td><td>←</td></tr><tr><td>Source-to-detector distance (mm)</td><td>501.309</td><td>←</td><td>←</td></tr></table>

Table 2: The scanning configuration ofour real datasets.
<table><tr><td rowspan="2">Parameters</td><td colspan="5">Real Dataset</td></tr><tr><td>Walnut</td><td>Metal Rods</td><td>Chicken (Wire)</td><td>Bell Pepper (Rivet)</td><td>Broccoli (Rivet)</td></tr><tr><td>Source voltage (kV)</td><td>90</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Source current (μA)</td><td>166</td><td>←</td><td>300</td><td>←</td><td>←</td></tr><tr><td>Aluminum filter (mm)</td><td>1.0</td><td>0.5</td><td>1.0</td><td>←</td><td>←</td></tr><tr><td>Metal material</td><td>Steel</td><td>Aluminum</td><td>Steel</td><td>←</td><td>←</td></tr><tr><td>Volume resolution</td><td>256×256×256</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Volume size (mm)</td><td>40.0×40.0×40.0</td><td>60.0×60.0×60.0</td><td>120.0×120.0×120.0</td><td>←</td><td>←</td></tr><tr><td>Angle range (°)</td><td>[0, 360)</td><td>←</td><td>←</td><td></td><td>←</td></tr><tr><td>Number of angles</td><td>720</td><td>←</td><td>←</td><td></td><td>←</td></tr><tr><td>Detector resolution</td><td>512×512</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Detector size (mm)</td><td>104.773</td><td>145.485</td><td>229.902</td><td>145.485</td><td>229.902</td></tr><tr><td>Source-to-object distance (mm)</td><td>200.962</td><td>←</td><td>334.942</td><td>←</td><td>←</td></tr><tr><td>Source-to-detector distance (mm)</td><td>501.309</td><td>←</td><td>←</td><td>←</td><td>←</td></tr></table>

## References

[FDK84] Feldkamp L. A., Davis L. C., Kress J. W.: Practical cone-beam algorithm. Journal ofthe Optical Society ofAmerica A 1, 6 (1984), 612–619. 1, 3

[PSJ] Park H. S., Seo J. K., Jeon K.: Implicit neural representation-based method for metal-induced beam hardening artifact reduction in x-ray ct imaging. Medical Physics. 3

[WCW∗23] Wu Q., Chen L., Wang C., Wei H., Zhou S. K., Yu J., Zhang Y.: Unsupervised polychromatic neural representation for CT metal artifact reduction. In Thirty-seventh Conference on Neural Information Processing Systems (2023). 3

[ZLC∗24] Zha R., Lin T. J., Cai Y., Cao J., Zhang Y., Li H.: R<sup>2</sup>-gaussian: Rectifying radiative gaussian splatting for tomographic reconstruction. In Advances in Neural Information Processing Systems (NeurIPS) (2024). 1, 2, 5

Table 3: GPU Memory usage (in MB) comparison between the baseline [ZLC∗24] and our method across diferent scenes.
<table><tr><td>Type</td><td>Scene</td><td>Baseline (MB)</td><td>Ours (MB)</td></tr><tr><td rowspan="3">Synthetic</td><td>Broccoli</td><td>1212</td><td>1402</td></tr><tr><td>Lung</td><td>1242</td><td>1404</td></tr><tr><td>Teeth</td><td>1272</td><td>1398</td></tr><tr><td rowspan="5">Real</td><td>Walnut</td><td>1308</td><td>1390</td></tr><tr><td>Metal Rods</td><td>1330</td><td>1406</td></tr><tr><td>Chicken (Wire)</td><td>1196</td><td>1382</td></tr><tr><td>Bell Pepper (Rivet)</td><td>1210</td><td>1388</td></tr><tr><td>Broccoli (Rivet)</td><td>1256</td><td>1340</td></tr></table>

![](images/495ad0a79eb68fc2de8ba49ac3d95db5342978bc0897f5eab9cabcb8e03a9aa1.jpg)  
Chicken (Wire)

![](images/0f6e12fbcda950c9aac5100883ec7686d1eb7056df55cc5bfe19da243c09dde3.jpg)  
Bell Pepper (Rivet)

![](images/cbbcabf54079bea99608864d0b9cb9558508180ce77740e0fd8acac48aca4db2.jpg)  
Broccoli (Rivet)  
Figure 3: The objects used for our additional results. The type of metallic objects used for each scene is shown in the parentheses.

![](images/28bb14eb2a2c1d620e876f1e816db133ab126559f46875898b82d748e8a98f86.jpg)  
Figure 4: Qualitative comparison on the real dataset: Chicken (Wire). The odd columns show the sampled slices from the reconstructed volumes by each algorithm, and the red boxes indicate the close-up regions. The even columns show the zoom-in patches from the corresponding slices.

![](images/a3c15d6758776d2ff32fb2291a2e23dc40abcfeeabfc0120e6353a9a42f1f051.jpg)  
Figure 5: Qualitative comparison on the real dataset: Bell Pepper (Rivet). The odd columns show the sampled slicesfrom the reconstructed volumes by each algorithm, and the red boxes indicate the close-up regions. The even columns show the zoom-in patchesfrom the corresponding slices.

![](images/828751d8d91d3df41bda511336c3e52c7b72520841649b6937ba4dfc9ee1b121.jpg)  
Figure 6: Qualitative comparison on the real dataset: Broccoli (Rivet). The odd columns show the sampled slicesfrom the reconstructed volumes by each algorithm, and the red boxes indicate the close-up regions. The even columns show the zoom-in patchesfrom the corresponding slices.