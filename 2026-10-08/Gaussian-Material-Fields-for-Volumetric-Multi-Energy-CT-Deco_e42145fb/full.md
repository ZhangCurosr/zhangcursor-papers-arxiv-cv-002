# Gaussian Material Fields for Volumetric Multi-Energy CT Decomposition

Jian Lin, Jiancheng Fang, Hongming Shan, Senior Member, IEEE, Shaoyu Wang, Yang Chen, Senior Member, IEEE, and Qiegen Liu, Senior Member, IEEE

Abstract—Volumetric material decomposition in multi-energy computed tomography requires a representation that organizes multiple three-dimensional material fields in a common spatial domain while retaining differences in composition and local structure. We observe that spatial primitives can be shared across materials without tying their coefficients, but their local capacity must respond to material-specific reconstruction needs. We introduce Gaussian material fields, which represent multiple material distributions with shared anisotropic 3D Gaussian primitives and independent nonnegative material coefficients. The shared geometry defines a continuous spatial basis, while the coefficients determine each primitive’s contribution to the individual material fields. To reconstruct this representation from multi-energy projections, a differentiable spectral forward model combines Gaussian material path integrals with a calibrated basis matrix, enabling joint optimization of spatial geometry and material composition. Material-aware adaptive density control retains material-specific refinement evidence before aggregation and adjusts local representation capacity to accommodate both spatially extensive components and sparse details. Experiments use synthesized multi-energy projections generated from pseudoreference material maps constructed by conventional methods from publicly available CT data. Across 15 cases, our approach improves average PSNR by 4.03 dB and SSIM by 4.96% over the strongest baseline, while reducing NRMSE by 33.45%. Materialwise comparisons and component ablations support improved recovery of localized structures, while runtime and memory measurements show favorable computational scaling. These results establish Gaussian material fields as an explicit, adaptive representation for volumetric multi-material reconstruction.

Index Terms—Multi-energy computed tomography, material decomposition, Gaussian material fields, shared geometry, adaptive density control.

## I. INTRODUCTION

ULTI-ENERGY computed tomography (CT), includ dependent X-ray attenuation to recover basis-material distributions beyond conventional attenuation imaging [1]–[5]. In volumetric material decomposition, the unknown is a collection of three-dimensional material coefficient fields. Recovering these fields requires more than reconstructing several scalar volumes: the representation must organize material composition within a common spatial domain while preserving the distinct distributions and local structures of the individual components.

![](images/83d0207a183c845cbb8b4e02256a94c6e48eda5ea575a702b6b5095d6976c511.jpg)  
Fig. 1. Two organizations of volumetric material reconstruction. (a) A two-stage image-domain pipeline reconstructs energy-dependent attenuation volumes before voxel-wise material decomposition. (b) Gaussian material fields represent material composition on shared spatial primitives and optimize geometry and material-specific coefficients from multi-energy projections.

Existing reconstruction routes organize these unknowns in different ways. Image-domain pipelines first estimate energydependent attenuation volumes and then decompose their voxel values. Direct model-based methods instead optimize material volumes through a coupled spectral forward model [6]–[9]. Dense voxel grids provide an explicit spatial layout, but assign material variables on a fixed lattice. Neural material fields replace that lattice with a continuous, coordinate-based function [10], [11], coupling representation and optimization through repeated network queries. Gaussian representations now support both attenuation reconstruction [12]–[15] and joint spectral basis estimation [16]. This motivates the question of how a shared spatial basis should be adapted when refinement demands differ across materials.

Our central observation is that spatial organization can be shared while material composition remains independently parameterized. A common set of primitives can define where and over what spatial extent the fields are represented, while separate coefficients determine each primitive’s contribution to each material. Shared geometry therefore need not impose equal material values or identical material boundaries. This organization also creates an adaptation requirement: a region that is adequately represented for one material may still lack capacity for another. In particular, a localized component may require refinement even when other materials provide little evidence at the same location. A shared representation should retain this material-specific demand when deciding where to add or remove capacity.

Anisotropic Gaussian primitives provide a concrete way to realize this organization. Their centers, orientations, and scales are explicit spatial parameters, and their overlapping contributions form continuous fields. Material-specific coefficients can weight the same kernels differently, while local primitive edits can change the spatial basis during reconstruction. Differentiable Gaussian X-ray rendering provides the connection to projection measurements [13]. Together, these properties make Gaussian primitives suitable not only for compact storage, but also for jointly fitting spatial structure and adapting the basis to heterogeneous material distributions.

We introduce Gaussian material fields, which combine shared anisotropic Gaussian geometry with independent nonnegative material coefficients. As illustrated in Fig. 1, the material fields themselves are the reconstruction variables. A differentiable spectral forward model projects their shared spatial basis, forms material path integrals, and mixes them through a calibrated basis matrix. Projection errors then update geometry and material composition jointly. This coupling also avoids relying on a final image-domain decomposition of separately reconstructed energy fields, where attenuation errors can be amplified. To adapt the shared basis, material-aware adaptive density control retains material-specific evidence before aggregation and reallocates local capacity through cloning, splitting, and pruning. Representation, joint fitting, and capacity adjustment thus address complementary parts of the same multi-material reconstruction problem.

The main contributions are summarized as follows:

• We introduce Gaussian material fields, an explicit representation that organizes multiple continuous material distributions through shared spatial primitives and independent material-specific coefficients. A differentiable spectral forward model makes this representation jointly optimizable from multi-energy projections.

• We develop material-aware adaptive density control for the shared representation. Material-specific reconstruction evidence guides cloning, splitting, and pruning so that local capacity can respond to both spatially extensive components and sparse material structures.

• We evaluate the representation and its reconstruction mechanisms using synthesized multi-energy data derived from public CT data and conventionally constructed pseudo-reference material maps. Overall comparisons, material-wise analysis, component ablations, and configuration studies demonstrate reconstruction quality, detail recovery, and favorable computational characteristics.

## II. RELATED WORK

## A. Spectral CT Material Decomposition and Reconstruction

Spectral CT relates energy-dependent attenuation measurements to basis-material composition [1], [2], [4], [5]. Imagedomain decomposition estimates material coefficients from reconstructed energy images. It inherits reconstruction errors despite statistical weighting and cross-channel regularization [17], [18]. Projection-domain decomposition estimates material path integrals before reconstruction, whereas joint reconstruction directly estimates material volumes. Representative methods include E-ART [19], IFBP [20], SOMA [21], statistical formulations [6], [7], constrained and preconditioned inversion [8], [9], and total-variation regularization [22], [23]. Recent networks address image-domain classification and concentration estimation [24] or learn dual-energy decomposition through polychromatic projection consistency without reference material images [25].

Continuous neural representations replace fixed voxel grids with coordinate-dependent fields. NeRF introduced this paradigm [26], and NeAT, NAF, and IntraTomo adapted neural or self-supervised representations to CT [27]–[29]. NeMCoF and neural base-material fields support per-instance material reconstruction without voxel-wise reference supervision [10], [11]. JSolver jointly estimates material volume fractions and the X-ray spectrum from single-energy projections [30]. Diffusion approaches instead introduce learned priors through spectral posterior sampling and memory-efficient volumetric extensions [31], [32]. In contrast, Gaussian material fields retain continuous material queries through explicit shared primitives, avoiding dense trainable grids, coordinate-MLP evaluation, and diffusion sampling.

## B. Gaussian-Based CT Reconstruction

3D Gaussian Splatting demonstrated differentiable rendering with explicit anisotropic primitives [33]. Radiative Gaussian Splatting (X-GS) adapted this representation to X-ray novel-view synthesis [12], and $\mathbb { R } ^ { 2 } .$ -Gaussian developed a rectified X-ray rasterizer for tomographic reconstruction [13]. Extensions include Gaussian volumetric reconstruction in 3DGR-CT, discretized reconstruction in DGR, coronary reconstruction, and direction-dependent radiograph synthesis [14], [15], [34], [35]. Recent work covers faster rasterization and voxelization in FaCT-GS [36], learned initialization and projection-domain refinement in SPARK [37], and maskguided initialization with loss scaling in LINGO [38].

SpectralCTGaussians fits photoelectric, Compton, and optional K-edge coefficients on shared Gaussians using a polychromatic model, then derives material labels by clustering [16]. Our method uses a calibrated log-domain model to reconstruct material coefficient fields. Shared geometry organizes multiple continuous material fields, while independent coefficients preserve their distinct compositions. Materialaware density control retains material-branch refinement evidence before aggregation to adapt the shared spatial basis.

(a) Error amplification in image-domain decomposition  
![](images/26ffa1eaeed33f537c9ab77cf56ee3875ed598565ad42326bc7efe254d248f83.jpg)

![](images/8d4225066e265bb4608d717dbb711f0c5b4d87b6825f4bd3f585d6f004a564ae.jpg)

![](images/a375c312e6fcb10b006668eb155ebbbd5932c2d92a1e63853e9bc42b1282e15f.jpg)

(b) Material-wise gradient aggregation  
![](images/846a9fd3a4a03a3677325ffc5d030d39d31a9443cfbaf8a15ea15f9ea8eae2ac.jpg)

![](images/5d015ab53e1df90cdb08136be6ad83ccc07ce6060f2902746fd44d03a7a89d88.jpg)  
Fig. 2. (a) Reconstructed energy-dependent attenuation fields $V _ { e } ,$ corresponding basis-material coefficient fields $C _ { m }$ , local error-magnitude visualizations, and quantitative summaries of error transfer across energies and materials. (b) Co-located basis-material coefficient patches and material-specific refinement evidence for a representative Gaussian $G _ { i }$ , including naive material averaging and material-wise maximum aggregation.

## III. METHOD

## A. Motivation

Gaussian material fields separate shared spatial organization from material-specific composition. The method first defines this representation, then fits its geometry and coefficients through a spectral forward model, and finally adjusts its local capacity using material-specific evidence. Fig. 3 shows the representation and joint fitting loop. Fig. 2 illustrates the errortransfer and local-refinement considerations addressed by the two reconstruction mechanisms.

## B. Shared-Geometry Gaussian Material Field

The representation must place all material fields in a common spatial basis without tying their values. We first specify the measurement model, then define the shared Gaussian kernels and the material-specific coefficients that together form the reconstruction variables.

We reconstruct the nonnegative material field $\begin{array} { r l } { \mathbf { c } ( \mathbf { x } ) } & { { } = } \end{array}$ $[ C _ { 1 } ( \mathbf { x } ) , \ldots , C _ { M } ( \mathbf { x } ) ] ^ { \mathsf { T } }$ directly from E co-registered energy channels, with material indices $\mathcal { M } = \{ 1 , \dots , M \}$ . Let $\textbf { B } \in$ $\mathbb { R } _ { \geq 0 } ^ { E \times M }$ be the fixed calibrated basis matrix and ${ \mathbf b } _ { e } ^ { \top }$ its eth row.

For ray $\mathbf { x } _ { r } ( t ) = \mathbf { o } _ { r } + t \mathbf { d } _ { \mathrm { \Omega } }$ <sub>r</sub> with $\| \mathbf { d } _ { \boldsymbol { r } } \| _ { 2 } = 1$ , the effective linear basis model gives

$$
\begin{array} { l } { { \displaystyle { \bf v } ( { \bf x } ) = { \bf B } { \bf c } ( { \bf x } ) } , } \\ { { p _ { e } } ( r ) = { \bf b } _ { e } ^ { \top } \displaystyle \int _ { t _ { r } ^ { - } } ^ { t _ { r } ^ { + } } { \bf c } ( { \bf x } _ { r } ( t ) ) \mathrm { d } t + \varepsilon _ { e } ( r ) . } \end{array}\tag{1}
$$

Here $\mathbf { v } = [ V _ { 1 } , \ldots , V _ { E } ] ^ { \mathsf { T } }$ and $\varepsilon _ { e }$ includes noise and model mismatch. This calibrated log-domain model [1], [6] is not an exact polychromatic photon-count model. Energy-dependent volumes are derived quantities rather than independent reconstruction variables.

All materials share N anisotropic Gaussian kernels, but each kernel carries its own material vector $\mathbf { a } _ { i } = [ a _ { i , 1 } , . . . , a _ { i , M } ] ^ { \mathsf { T } }$ With center $\pmb { \mu } _ { i }$ , positive scales $\sigma _ { i } .$ and rotation $\mathbf { R } _ { i }$ obtained from a normalized nonzero quaternion $\mathbf { q } _ { i } ,$ define

$$
\begin{array} { r l } & { K _ { i } ( { \bf x } ) = \exp \left[ - \frac { 1 } { 2 } ( { \bf x } - { \pmb \mu } _ { i } ) ^ { \sf T } { \pmb \Sigma } _ { i } ^ { - 1 } ( { \bf x } - { \pmb \mu } _ { i } ) \right] , } \\ & { \quad \quad { \pmb \Sigma } _ { i } = { \bf R } _ { i } \mathrm { d i a g } ( \pmb \sigma _ { i } ^ { 2 } ) { \bf R } _ { i } ^ { \sf T } \succ 0 , } \\ & { \quad a _ { i , m } = \mathrm { s o f t p l u s } ( \beta _ { i , m } ) . } \end{array}\tag{2}
$$

We collect the geometric parameters in $\Theta = \{ \pmb { \mu } _ { i } , \mathbf { q } _ { i } , \pmb { \eta } _ { i } \} _ { i = 1 } ^ { N } ,$ where $\eta _ { i }$ parameterizes the Gaussian scales. The material coefficients form the matrix $\begin{array} { r } { \begin{array} { l } { \mathbf { A } } \end{array} = \begin{array} { l } { \left[ \mathbf { a } _ { 1 } , \ldots , \mathbf { a } _ { N } \right] ^ { \mathsf { T } } } \end{array} = } \end{array}$ softplus $( \mathbf { Z } ) \in \mathbb { R } _ { > 0 } ^ { N \times M }$ , where $\mathbf { Z } = [ \beta _ { i , m } ]$ . The material and attenuation fields are then given by

$$
\begin{array} { r l } & { { \displaystyle { \bf c } _ { \Theta , { \bf A } } ( { \bf x } ) = \sum _ { i = 1 } ^ { N } K _ { i } ( { \bf x } ) { \bf a } _ { i } = { \bf A } ^ { \top } { \bf k } ( { \bf x } ; \Theta ) } , } \\ & { { \displaystyle { \bf v } _ { \Theta , { \bf A } } ( { \bf x } ) = { \bf B } { \bf A } ^ { \top } { \bf k } ( { \bf x } ; \Theta ) , } } \end{array}\tag{3}
$$

where $\mathbf { k } = [ K _ { 1 } , \ldots , K _ { N } ] ^ { \mathsf { T } }$ . Shared geometry establishes spatial correspondence without imposing equal material values or a sum-to-one composition. The kernels are unnormalized: $a _ { i , m }$ is a peak amplitude, not integrated material content. Softplus allows arbitrarily small, but not exactly zero, amplitudes at finite logits. Parameter maps and their numerical boundary conventions are discussed in the Appendix.

In Eq. (3), the kernels determine the shared spatial organization, while the coefficient vectors determine the individual material distributions. Each material field is a differently weighted combination of the same kernels, rather than a copy of a common scalar field. Moving or reshaping a primitive changes the basis available to all materials, whereas changing a material-specific coefficient changes only that primitive’s contribution to the corresponding field. This separation supplies the geometric and compositional variables for joint projectiondomain fitting.

## C. Projection-Domain Joint Reconstruction

To fit the material representation, the forward model must connect its shared spatial basis and material-specific coefficients to the energy-resolved measurements. Fig. 2(a) illustrates why this coupling is useful: errors in reconstructed attenuation fields can be amplified during subsequent imagedomain decomposition, with different transfer sensitivities across materials. We instead evaluate projection consistency from the material fields themselves, using the Gaussian ray response and its rasterized implementation below.

![](images/22ac2419325398399fee2ed2c0f6f3d8c8402f49233eaa91fe450a5fc7d43619.jpg)  
Fig. 3. Gaussian material fields and their projection-domain optimization. Shared anisotropic primitives carry material-specific coefficients. X-ray projectio forms material path integrals, and calibrated spectral mixing produces predicted energy projections. The projection loss updates geometry and coefficients jointly, while the optimized field provides the material outputs.

Substitution of Eq. (3) into Eq. (1) reduces each material path integral to a weighted sum of Gaussian ray responses. For $\mathbf { Q } _ { i } = \Sigma _ { i } ^ { - 1 } , \delta _ { i , r } = \mathbf { o } _ { r } - \pmb { \mu } _ { i }$ , and $\alpha _ { i , r } = \mathbf { d } _ { r } ^ { \mathsf { T } } \mathbf { Q } _ { i } \mathbf { d } _ { r }$ , completing the square yields

$$
\begin{array} { l } { { \displaystyle \ell _ { i , r } ^ { \infty } = \sqrt { \frac { 2 \pi } { \alpha _ { i , r } } } \exp ( - \Delta _ { i , r } / 2 ) } , } \\ { { \displaystyle \Delta _ { i , r } = \delta _ { i , r } ^ { \top } \left( \mathbf Q _ { i } - \frac { \mathbf Q _ { i } \mathbf d _ { r } \mathbf d _ { r } ^ { \top } \mathbf Q _ { i } } { \alpha _ { i , r } } \right) \delta _ { i , r } } . } \end{array}\tag{4}
$$

This is the exact infinite-line response. The finite-segment response $\begin{array} { r } { \ell _ { i , r } ^ { \mathrm { s e g } } = \int _ { t _ { - } ^ { - } } ^ { t _ { r } ^ { + } } K _ { i } ( \mathbf { x } _ { r } ( t ) ) } \end{array}$ dt and an explicit omitted-tail bound are established in Proposition 1. Replacing it by Eq. (4) requires negligible tails beyond the acquisition segment.

To exploit rasterization, we use the local affine ray-space approximation of R<sup>2</sup>-Gaussian [13]. Let $\mathbf W _ { v } \in \mathbb R ^ { 3 \times 3 }$ be the rotational part of the viewing transform and $\mathbf { J } _ { i , v } \in \mathbb { R } ^ { 3 \times 3 }$ the ray-space Jacobian at primitive i. With projected center $\widehat { \mu } _ { i , v } ,$ detector covariance $\widehat { \Sigma } _ { i , v } ,$ , and $\begin{array} { r } { \pmb { \rho } _ { i , r } = \mathbf { u } _ { r } - \hat { \pmb { \mu } } _ { i , v } } \end{array}$ for ray $r =$ $\left( v , \mathbf { u } _ { r } \right)$

$$
\begin{array} { r l } & { \widetilde { \boldsymbol { \Sigma } } _ { i , v } = \boldsymbol { \mathrm { J } } _ { i , v } \mathbf { W } _ { v } \boldsymbol { \Sigma } _ { i } \mathbf { W } _ { v } ^ { \top } \mathbf { J } _ { i , v } ^ { \top } , } \\ & { \Phi _ { r , i } = \zeta _ { i , v } \exp \left[ - \frac { 1 } { 2 } \rho _ { i , r } ^ { \top } \widehat { \boldsymbol { \Sigma } } _ { i , v } ^ { - 1 } \boldsymbol { \rho } _ { i , r } \right] , } \\ & { \zeta _ { i , v } = \left( \displaystyle \frac { 2 \pi \left| \widetilde { \boldsymbol { \Sigma } } _ { i , v } \right| } { \left| \widehat { \boldsymbol { \Sigma } } _ { i , v } \right| } \right) ^ { 1 / 2 } . } \end{array}\tag{5}
$$

The determinant ratio follows from Gaussian marginalization (Proposition 2), rather than an empirical amplitude correction. The third ray-space coordinate measures physical length. The matrix $\widehat { \Sigma } _ { i , v }$ is the leading $2 \times 2$ covariance block. Indexing all detector samples by $r = 1 , \ldots , R ,$ the shared additive

projector gives

$$
\begin{array} { r l } & { \widehat { \mathbf { L } } = \boldsymbol { \Phi } ( \boldsymbol { \Theta } ) \mathbf { A } , } \\ & { \widehat { \mathbf { P } } = \boldsymbol { \Phi } ( \boldsymbol { \Theta } ) \mathbf { A } \mathbf { B } ^ { \top } , \qquad \boldsymbol { \Phi } \in \mathbb { R } ^ { R \times N } . } \end{array}\tag{6}
$$

Equation (6) is exact for the defined rasterized model, not for the original finite-ray integral. Spectral mixing remains linear, whereas spatial approximation includes local linearization, omitted tails, and numerical footprint truncation. The factorization assumes common geometric footprints and additive material rendering. The matrix denotes an operator applied view by view, not a dense stored array.

Equation (6) assigns a distinct role to each part of the representation. The shared projector describes how spatial kernels intersect the acquisition rays. The coefficient matrix turns those responses into material path integrals. The calibrated basis matrix mixes the material contributions into energy channels. Geometry and composition therefore affect the same predicted measurements, allowing a common projection objective to update both rather than fitting separate energydependent geometries.

For fixed energy weights $\mathbf { W } _ { E } = \mathrm { d i a g } ( w _ { 1 } , \dots , w _ { E } ) , w _ { e } >$ 0, the inverse problem at a fixed primitive count is

$$
\begin{array} { r l } & { \underset { \Theta , \mathbf { Z } } { \mathrm { m i n i m i z e } } \quad \mathcal { L } _ { \mathrm { p r o j } } ( \Theta , \mathbf { Z } ) = \displaystyle \frac { 1 } { R } \| \mathbf { E } _ { p } \mathbf { W } _ { E } \| _ { 1 , 1 } , } \\ & { \mathbf { E } _ { p } = \Phi ( \Theta ) \mathrm { s o f t p l u s } ( \mathbf { Z } ) \mathbf { B } ^ { \top } - \mathbf { P } , } \end{array}\tag{7}
$$

where $\| \mathbf { X } \| _ { 1 , 1 }$ sums absolute entries. No volumetric ground truth or auxiliary volume/support loss is used. Define the projection sensitivity $S _ { r , e } = ( w _ { e } / R ) \operatorname { s g n } ( E _ { p , r , e } )$ , with $\operatorname { s g n } ( 0 ) =$ $0 ,$ and its material-domain counterpart $\mathbf { D } = \mathbf { S } \mathbf { B }$ . On a fixed active set and away from zero residuals, the first variation is

$$
\begin{array} { r l } & { \mathrm { d } \mathcal { L } _ { \mathrm { p r o j } } = \langle \mathbf { D } , ( \mathrm { d } \Phi ) \mathbf { A } + \Phi \mathrm { d } \mathbf { A } \rangle _ { F } } \\ & { \qquad = \langle \mathbf { D } \mathbf { A } ^ { \top } , \mathrm { d } \Phi \rangle _ { F } + \langle \mathbf { \Phi } ^ { \top } \mathbf { D } , \mathrm { d } \mathbf { A } \rangle _ { F } . } \end{array}\tag{8}
$$

![](images/1f60debe42ba7e89e3a94918c6ea4bbef29d7d963ad96b883d03301188a21d4c.jpg)  
Fig. 4. Material-aware adaptive density control. Material-specific projection sensitivities are retained before aggregation and combined with Gaussian scale to guide cloning and splitting. Pruning tests material amplitudes across all channels.

Thus, for $\pmb \theta _ { i } \in \{ \pmb \mu _ { i } , \mathbf q _ { i } , \pmb \eta _ { i } \}$

$$
\begin{array} { r l } & { \nabla _ { \mathbf { A } } \mathcal { L } _ { \mathrm { p r o j } } = \Phi ^ { \mathsf { T } } \mathbf { D } , } \\ & { \nabla _ { \mathbf { Z } } \mathcal { L } _ { \mathrm { p r o j } } = ( \Phi ^ { \mathsf { T } } \mathbf { D } ) \odot \mathrm { s i g m o i d } ( \mathbf { Z } ) , } \\ & { \nabla _ { \theta _ { i } } \mathcal { L } _ { \mathrm { p r o j } } = { \displaystyle \sum _ { r = 1 } ^ { R } } ( \mathbf { D } \mathbf { A } ^ { \mathsf { T } } ) _ { r , i } \nabla _ { \theta _ { i } } \Phi _ { r , i } . } \end{array}\tag{9}
$$

At zero residuals these expressions use the selected absolutevalue subgradient. Since $\begin{array} { r } { D _ { r , m } ~ = ~ \sum _ { e } S _ { r , e } B _ { e , m } } \end{array}$ , material coefficients receive spectrally specific updates while geometry receives coupled evidence from every material. Full geometric differentials, including the position dependence of $\mathbf { J } _ { i , v } ,$ are given in Proposition 4. Fixed-geometry separation and conditioning are analyzed in Proposition 3. This analysis neither asserts global uniqueness nor replaces the actual $\ell _ { 1 }$ objective by least squares.

We initialize from energy-wise FDK followed by voxelwise NNLS [39], [40], then optimize geometry and composition jointly with Adam using separate parameter groups and exponential learning-rate schedules. Initialization details are deferred to the Appendix. Parameter updates are interleaved with the density control below. At the final iterate, material values are queried directly as $\widehat { \mathbf { c } } ( \mathbf { x } ) = \mathbf { A } _ { T } ^ { \mathsf { T } } \mathbf { k } ( \mathbf { x } ; \boldsymbol { \Theta } _ { T } )$ . No final independent decomposition is required.

## D. Material-Aware Adaptive Density Control

Joint parameter updates fit the current spatial basis, but cannot by themselves change how many primitives represent a local structure. Capacity adjustment must also respect material differences: a region may be adequately represented in one field yet unresolved in another. Fig. 2(b) illustrates how averaging across materials can suppress a localized refinement response. We retain the material branches when constructing refinement evidence, then use their aggregation and primitive scale to guide the shared representation’s adaptation, as illustrated in Fig. 4.

Equation (9) updates parameters within the current representation, whereas density control adjusts its local capacity. To retain material identity, let $\mathbf { u } _ { i , m } ^ { ( v ) }$ denote the projected-center variable in material branch m, evaluated at the common center $\widehat { \mu } _ { i , v }$ . Holding projected covariance and amplitude correction fixed for this partial derivative, Eqs. (5) and (8) give

$$
\begin{array} { r l } & { \mathbf { h } _ { i , m } ^ { ( v ) } : = \nabla _ { \mathbf { u } _ { i , m } ^ { ( v ) } } \mathcal { L } _ { \mathrm { p r o j } } } \\ & { \qquad = a _ { i , m } \displaystyle \sum _ { r \in \mathcal { R } _ { v } } D _ { r , m } \Phi _ { r , i } \widehat { \boldsymbol { \Sigma } } _ { i , v } ^ { - 1 } \rho _ { i , r } , } \end{array}\tag{10}
$$

where $\mathcal { R } _ { v }$ indexes the view’s detector samples. This projectedcenter contribution is not the full three-dimensional position gradient. Summing the vectors across materials before forming a refinement score can cancel responses. Averaging their norms can dilute a response confined to one material. Writing $\mathbf { h } _ { i , m } ^ { ( t ) }$ for the branch derivative at iteration t, we use

$$
\begin{array} { r } { g _ { i } ^ { ( t ) } = \displaystyle \operatorname* { m a x } _ { m \in \mathcal { M } } \| \mathbf { h } _ { i , m } ^ { ( t ) } \| _ { 2 } , } \\ { \bar { g } _ { i } = \displaystyle \frac { \sum _ { t \in \mathcal { W } } \nu _ { i } ^ { ( t ) } g _ { i } ^ { ( t ) } } { \sum _ { t \in \mathcal { W } } \nu _ { i } ^ { ( t ) } + \epsilon } , } \end{array}\tag{11}
$$

with zero evidence for invisible branches, visibility indicator $\nu _ { i } ^ { ( t ) }$ , accumulation window W, and $\epsilon > 0$ . The maximum is taken before temporal accumulation. Proposition 5 states precisely what this aggregation preserves.

For $s _ { i } ~ = ~ \| \pmb { \sigma } _ { i } \| _ { \infty }$ and thresholds $\tau _ { g } , \tau _ { s }$ , the pre-pruning refinement rule is

$$
\mathcal { D } _ { i } = \left\{ \begin{array} { l l } { \mathrm { c l o n e , } } & { \bar { g } _ { i } \geq \tau _ { g } , s _ { i } \leq \tau _ { s } , } \\ { \mathrm { s p l i t , } } & { \bar { g } _ { i } \geq \tau _ { g } , s _ { i } > \tau _ { s } , } \\ { \mathrm { k e e p , } } & { \bar { g } _ { i } < \tau _ { g } . } \end{array} \right.\tag{12}
$$

These edits remain connected to the same forward model. Replacing primitive i by children $\mathcal { C } _ { i }$ changes the predicted data by

$$
\delta \widehat { \mathbf { P } } _ { i } = \sum _ { j \in \mathcal { C } _ { i } } \phi _ { i , j } ( \mathbf { B } \mathbf { a } _ { i , j } ) ^ { \mathsf { T } } - \phi _ { i } ( \mathbf { B } \mathbf { a } _ { i } ) ^ { \mathsf { T } } ,\tag{13}
$$

where $\phi _ { i } = \Phi _ { : , i }$ Cloning creates two identical kernels with activated coefficients $\mathbf { a } _ { i } / 2$ , so Eq. (13) vanishes at insertion under unchanged linear rendering. Splitting samples center offsets using the parent covariance, reduces scales, and divides coefficients across children. It preserves the coefficient sum, but generally changes both the field integral and its projections. Exact rules and their consequences are established in Proposition 6.

TABLE I  
SIMULATED CASES. E/M GIVES ENERGY/MATERIAL CHANNELS.
<table><tr><td>Case</td><td>E/M</td><td>Detector</td><td>Volume</td></tr><tr><td>AAPM</td><td>2/2</td><td> $5 1 2 ^ { 2 }$ </td><td> $2 5 6 ^ { 3 }$ </td></tr><tr><td>Lizard head, mouse forelimb</td><td>3/3</td><td> $1 2 8 ^ { 2 }$ </td><td> $8 0 ^ { 3 }$ </td></tr><tr><td>Powder phantom</td><td>4/4</td><td> $1 2 8 ^ { 2 }$ </td><td> $8 0 ^ { 3 }$ </td></tr><tr><td>MUSIC 3D</td><td>6/6</td><td> $2 5 6 ^ { 2 }$ </td><td> $2 5 6 ^ { 3 }$ </td></tr><tr><td>Walnut (cubic)</td><td>2/2</td><td> $2 5 6 ^ { 2 }$ </td><td> $2 5 6 ^ { 3 }$ </td></tr><tr><td>Ore (cubic)</td><td>2/2</td><td> $1 2 8 ^ { 2 }$ </td><td> $8 0 ^ { 3 }$ </td></tr></table>

A low-content primitive is pruned only when $\operatorname* { m a x } _ { m } a _ { i , m } <$ $\tau _ { a } .$ The geometric guard also removes centers outside the valid reconstruction box Ω. Thus, a non-negligible response in any material prevents low-content pruning. Refinement is restricted by the prescribed iteration window and Gaussian budget. Optional size tests apply only when enabled. These residual-guided edits change the approximation space between parameter updates, without constituting an exact minimization over N or guaranteeing a loss decrease. Together, Eqs. (7) and (13) describe fitting and adapting one continuous material representation.

## IV. EXPERIMENTS

## A. Datasets and Experimental Settings

We report material reconstruction on 15 simulated multienergy cone-beam cases. Of these, 9 anatomical softtissue/calcium phantoms are constructed from AAPM Low Dose CT Grand Challenge CT volumes [41] and have known synthetic material maps. The remaining 6 cases use material maps derived from lizard-head data [42], [43], powderphantom data [43], [44], mouse-limb data [45], [46], MUSIC 3D [47], walnut data [48], and ore data [49]. We refer to these derived maps as pseudo-reference material maps. The material maps generate the multi-energy projections and serve as the evaluation references. Table I summarizes the channel counts and reconstruction grids.

For every case, 50 uniformly spaced training views cover a full rotation and 100 nonoverlapping views form the test set. The AAPM training projections include the prescribed Poisson and Gaussian noise, whereas their test projections and the projections in the other 6 cases are clean. The conebeam simulator uses source-to-detector and source-to-origin distances of 7 and 5, respectively, and a 4 × 4 detector extent in simulation coordinates. Detector sampling and volume resolution are listed in Table I.

The comparison includes FDK+NNLS [39], [40], ASD-POCS+NNLS [23], NeRF+NNLS [26], R2+NNLS [13], AD-JUST [50], CIL SIRT+NNLS [51], and the proposed method. The 5 NNLS baselines reconstruct energy volumes before material decomposition. The dataset’s material basis matrix and channel order are fixed for each case and used by all methods that require the basis. The compared runs use 6 ASD-POCS iterations, 1000 NeRF epochs, 15,000 R2 iterations per energy channel, 1000 outer and 10 inner ADJUST iterations, 50 CIL SIRT iterations, and 15,000 iterations for the proposed method. FDK is a direct reconstruction. These are method-specific stopping settings rather than a common iteration budget. The source code is available at https://github.com/yqx7150/MGS.

We evaluate every method on the same reference material channels for each case. For each channel, PSNR uses the range of its reference volume, NRMSE is voxelwise RMSE divided by that range, and SSIM averages nonempty reference slices along the 3 volume axes. We report the arithmetic mean of each metric across a case’s material channels. Family-level summaries average 1 score per case configuration.

## B. Comparative Results

Table II presents all 15 simulated-projection cases at 50 training views. Each cell gives material-averaged full-volume PSNR, SSIM, and NRMSE, and the final column averages the 15 cases equally. The proposed method leads all 3 metrics in each case.

On the 9 AAPM phantoms, the proposed method’s mean PSNR margin over the strongest competing method in each case is 2.38 dB. The 2 animal cases also favor the proposed method. The largest gaps occur for the powder phantom and walnut, where the proposed method reaches 41.39 and 37.77 dB, versus 32.98 and 29.86 dB for the strongest alternatives. These results show improved recovery of the constructed and pseudo-reference material volumes across the tested configurations.

In Fig. 5, the proposed method preserves the thin calcification arc and cortical boundary more continuously than competing methods. It also retains the forelimb contour and internal structure with fewer spurious responses.

In Fig. 6, the proposed method better preserves narrow powder features against a smoother background and recovers a more continuous inner ore-Pb ring.

In Fig. 7, the proposed method retains compact MUSIC boundaries with less mottling and a more continuous walnut rim, while several baselines introduce signal inside the shell.

We also evaluate full-rotation sparse-view reconstruction on the ore and lizard-head cases using 25, 50, and 75 uniformly spaced training views. Within each case, the material basis, channel order, volume grid, and 100 disjoint test views stay fixed. The proposed method, R2+NNLS, and ASD-POCS+NNLS are evaluated with 1 seed each and run for 15,000 iterations, 15,000 iterations per energy channel, and 6 iterations, respectively. Fig. 8 shows the material-averaged full-volume PSNR, SSIM, and NRMSE across view counts. The proposed method leads all 3 metrics at the 6 tested case/view settings.

Fig. 9 compares selected 25-view material slices with the same slice, ROI, and grayscale window across methods within each case. The proposed method retains the inner ore-Pb ring and the lizard-head cortical edge more continuously than the 2 shown baselines. Ore Pb uses the same fixed location as its 50-view row in Fig. 6.

Fig. 10 reports the complete method on 4 distinct simulatedprojection cases with 2, 3, 4, and 6 energy/material channels.

TABLE II  
50-VIEW MATERIAL-VOLUME PSNR (DB)/SSIM/NRMSE ON 15 CASES.
<table><tr><td>Method</td><td>L096</td><td>L109</td><td>L143</td><td>L192</td><td>L286</td></tr><tr><td>FDK+NNLS</td><td>23.58/0.3316/0.0705</td><td>18.33/0.2113/0.1279</td><td>17.10/0.1789/0.1489</td><td>19.90/0.2393/0.1065</td><td>18.01/0.1943/0.1329</td></tr><tr><td>ADJUST</td><td>31.15/0.8710/0.0370</td><td>26.68/0.8034/0.0586</td><td>24.15/0.6455/0.0850</td><td>27.20/0.7694/0.0528</td><td>25.20/0.6877/0.0692</td></tr><tr><td>NeRF+NNLS</td><td>31.10/0.7799/0.0294</td><td>27.66/0.6679/0.0434</td><td>28.29/0.6608/0.0409</td><td>29.28/0.7530/0.0359</td><td>28.56/0.6764/0.0393</td></tr><tr><td>CIL SIRT+NNLS</td><td>31.35/0.8079/0.0314</td><td>28.10/0.6754/0.0428</td><td>27.31/0.5913/0.0495</td><td>27.88/0.7211/0.0428</td><td>27.52/0.6415/0.0458</td></tr><tr><td>R2+NNLS</td><td>34.02/0.8610/0.0216</td><td>31.32/0.7987/0.0289</td><td>29.95/0.7538/0.0351</td><td>31.76/0.8331/0.0274</td><td>31.29/0.7959/0.0296</td></tr><tr><td>ASD-POCS+NNLS</td><td>33.60/0.9011/0.0243</td><td>30.75/0.8340/0.0319</td><td>30.17/0.8384/0.0375</td><td>30.25/0.8698/0.0328</td><td>30.19/0.8551/0.0342</td></tr><tr><td>Ours</td><td>37.01/0.9340/0.0169</td><td>33.55/0.8939/0.0243</td><td>32.82/0.8797/0.0286</td><td>33.68/0.9109/0.0232</td><td>33.62/0.8923/0.0246</td></tr></table>

<table><tr><td>Method</td><td>L291</td><td>L310</td><td>L333</td><td>L506</td><td>Lizard</td></tr><tr><td>FDK+NNLS</td><td>24.48/0.3551/0.0632</td><td>16.83/0.1630/0.1522</td><td>19.19/0.2180/0.1155</td><td>18.69/0.2163/0.1221</td><td>27.68/0.6972/0.0449</td></tr><tr><td>ADJUST</td><td>31.71/0.8971/0.0322</td><td>24.12/0.6371/0.0835</td><td>28.16/0.8303/0.0486</td><td>25.13/0.7236/0.0652</td><td>25.59/0.7971/0.0620</td></tr><tr><td>NeRF+NNLS</td><td>31.78/0.8238/0.0270</td><td>28.25/0.6357/0.0408</td><td>28.78/0.7207/0.0380</td><td>27.73/0.7169/0.0428</td><td>27.77/0.7940/0.0452</td></tr><tr><td>CIL SIRT+NNLS</td><td>30.90/0.8302/0.0313</td><td>27.33/0.5912/0.0473</td><td>27.74/0.7035/0.0438</td><td>27.10/0.6988/0.0462</td><td>29.82/0.8683/0.0325</td></tr><tr><td>R2+NNLS</td><td>34.28/0.8828/0.0205</td><td>30.90/0.7681/0.0312</td><td>31.79/0.8303/0.0275</td><td>30.92/0.8199/0.0300</td><td>29.91/0.8719/0.0357</td></tr><tr><td>ASD-POCS+NNLS</td><td>33.02/0.9056/0.0244</td><td>30.38/0.8613/0.0346</td><td>30.17/0.8681/0.0334</td><td>29.57/0.8607/0.0349</td><td>31.47/0.9321/0.0268</td></tr><tr><td>Ours</td><td>36.65/0.9431/0.0165</td><td>33.65/0.8841/0.0255</td><td>33.85/0.9084/0.0232</td><td>33.06/0.9053/0.0248</td><td>33.92/0.9581/0.0203</td></tr></table>

<table><tr><td>Method</td><td>Mouse FL</td><td>Powder</td><td>MUSIC</td><td>Walnut</td><td>Ore</td><td>Average</td></tr><tr><td>FDK+NNLS</td><td>30.05/0.7778/0.0326</td><td>26.55/0.6505/0.0746</td><td>42.96/0.8691/0.0078</td><td>26.68/0.4796/0.0470</td><td>27.68/0.7866/0.0413</td><td>23.85/0.4246/0.0859</td></tr><tr><td>ADJUST</td><td>23.00/0.7398/0.0723</td><td>28.10/0.8178/0.0572</td><td>46.67/0.9890/0.0051</td><td>21.94/0.7501/0.0810</td><td>24.22/0.7264/0.0657</td><td>27.54/0.7790/0.0584</td></tr><tr><td>NeRF+NNLS</td><td>22.33/0.7117/0.0879</td><td>27.44/0.7122/0.0622</td><td>44.17/0.9719/0.0071</td><td>22.00/0.7011/0.0797</td><td>32.44/0.8814/0.0239</td><td>29.17/0.7471/0.0429</td></tr><tr><td>CIL SIRT+NNLS</td><td>31.83/0.9324/0.0266</td><td>29.50/0.7772/0.0469</td><td>43.52/0.9832/0.0072</td><td>27.14/0.8110/0.0444</td><td>31.81/0.8885/0.0257</td><td>29.92/0.7681/0.0376</td></tr><tr><td>R2+NNLS</td><td>24.88/0.7760/0.0661</td><td>29.19/0.7261/0.0523</td><td>48.86/0.9901/0.0040</td><td>22.20/0.7209/0.0781</td><td>33.34/0.9071/0.0216</td><td>31.64/0.8224/0.0340</td></tr><tr><td>ASD-POCS+NNLS</td><td>33.01/0.9481/0.0230</td><td>32.98/0.8826/0.0322</td><td>45.58/0.9904/0.0056</td><td>29.86/0.8756/0.0324</td><td>31.24/0.8751/0.0274</td><td>32.15/0.8865/0.0290</td></tr><tr><td>Ours</td><td>35.58/0.9711/0.0174</td><td>41.39/0.9579/0.0108</td><td>51.18/0.9964/0.0029</td><td>37.77/0.9712/0.0133</td><td>35.01/0.9515/0.0178</td><td>36.18/0.9305/0.0193</td></tr></table>

![](images/7b51504722d0796b8b931cc7a65bbfb9a3583b098604213c48ec9fa12684a6ec.jpg)  
Fig. 5. Matched 50-view slices. Rows, top to bottom: AAPM L506 calcification, lizard-head bone, mouse-forelimb background. Columns, left to right: GT FDK, ADJUST, NeRF, CIL, R2, ASD-POCS, Ours. Main/ROI views are at left/upper right. Auxiliary structural differences are at lower right (red: missing, blue: extra). Window [0, 1]. Full-slice 2D PSNR (dB)/SSIM (%).

![](images/963e13ed30996e9ff29d849281d91ed3e94d1109f84fdbad1687271016aa9e41.jpg)  
Fig. 6. 50-view material-slice comparisons. Rows show powder-phantom Al, powder-phantom Fe, and ore Pb, respectively. Enlarged regions highlight narrow powder features and the inner ore-Pb ring.

ASD-POCS+NNLS  
![](images/a0f689792279756cd10640d4419c899b234f820eea482c8bf94ede9769f3afb4.jpg)  
Fig. 7. 50-view material-slice comparisons. Rows show MUSIC 3D water-like, MUSIC 3D iodine-like, and walnut shell, respectively. Enlarged regions highlight compact material boundaries and the thin walnut-shell rim.

![](images/579f7006b077811320c7e2c3f682d4ef922506c59ac33f00d6ea8e907425a677.jpg)

Fig. 8. Material-volume quality at 25/50/75 views: lizard head (left) and ore (right). Rings show PSNR, SSIM, and NRMSE $\times 1 0 ^ { 3 }$ from outside in. Symbols mark methods. Darker shading is better within each metric. Each setting uses 1 seed.  
![](images/79b0a0cee770d9217af803e5859d2b1726c01f4a4f2857c0d64651b09f308472.jpg)  
Fig. 9. Matched 25-view slices: ore Pb and lizard-head bone, top to bottom. Columns: GT, R2, ASD-POCS, Ours.

Each uses 50 training views, 100 disjoint test views, 1 seed, and 15,000 iterations. The material basis is fixed within each case. These observations demonstrate reconstruction for the 4 configurations, but their different scenes and bases, along with the different grid used in 1 case, prevent attributing case differences to channel count alone.

TABLE III  
MATERIAL-VOLUME MEAN ± SD ACROSS 3 SEEDS. $\mathrm { N R M S E } \times \mathrm { 1 0 ^ { - 3 } }$ BOLD/UNDERLINE MARK BEST/SECOND MEANS.
<table><tr><td>Setting</td><td>PSNR (dB)</td><td>SSIM</td><td>NRMSE  $( 1 0 ^ { - 3 } )$ </td></tr><tr><td>Lizard head</td><td></td><td></td><td></td></tr><tr><td>Full</td><td> $3 3 . 9 1 { \pm } 0 . 0 3 $ </td><td> $0 . 9 5 8 1 3 { \scriptstyle \pm 0 . 0 0 0 2 1 }$ </td><td>20.368±0.071</td></tr><tr><td>Fixed set</td><td> $3 3 . 3 3 { \pm } 0 . 0 1 $ </td><td> $0 . 9 5 4 7 0 { \scriptstyle \pm 0 . 0 0 0 2 1 }$ </td><td>21.778±0.027</td></tr><tr><td>Mean agg.</td><td> $3 4 . 0 6 { \pm } 0 . 0 4 $ </td><td> $0 . 9 6 1 0 7 { \scriptstyle \pm 0 . 0 0 0 1 5 }$ </td><td>20.045±0.083</td></tr><tr><td>Material  $\ell _ { 1 }$ </td><td> ${ \bf 3 4 . 3 0 { \pm } 0 . 0 1 }$ </td><td> $\mathbf { 0 . 9 6 3 9 6 { \scriptstyle \pm 0 . 0 0 0 0 2 } }$ </td><td>19.538±0.027</td></tr><tr><td>Energy  $\ell _ { 1 }$ </td><td> $2 9 . 6 6 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $0 . 8 8 7 4 1 { \scriptstyle \pm 0 . 0 0 0 3 1 }$ </td><td>34.538±0.095</td></tr><tr><td>Powder phantom</td><td></td><td></td><td></td></tr><tr><td>Full</td><td>41.55±0.16</td><td> $0 . 9 5 8 2 0 { \scriptstyle \pm 0 . 0 0 0 2 4 }$ </td><td>10.728±0.102</td></tr><tr><td>Fixed set</td><td>40.85±0.03</td><td> $0 . 9 6 6 1 7 { \scriptstyle \pm 0 . 0 0 0 1 9 }$ </td><td>11.023±0.031</td></tr><tr><td>Mean agg.</td><td>41.90±0.21</td><td> $0 . 9 6 6 0 4 { \scriptstyle \pm 0 . 0 0 0 4 7 }$ </td><td>10.267±0.265</td></tr><tr><td>Material  $\ell _ { 1 }$ </td><td>42.59±0.14</td><td> $\mathbf { 0 . 9 6 7 3 0 { \scriptstyle \pm 0 . 0 0 0 2 7 } }$ </td><td>9.589±0.069</td></tr><tr><td>Energy  $\ell _ { 1 }$ </td><td> $3 3 . 1 3 { \pm } 0 . 0 8 $ </td><td> $0 . 8 0 2 2 6 { \scriptstyle \pm 0 . 0 0 0 2 0 }$ </td><td>29.940±0.287</td></tr><tr><td>MUSIC 3D</td><td></td><td></td><td></td></tr><tr><td>Full</td><td> $5 1 . 2 2 { \pm } 0 . 0 6$ </td><td> $0 . 9 9 6 4 4 { \scriptstyle \pm 0 . 0 0 0 0 2 }$ </td><td>2.865±0.022</td></tr><tr><td>Fixed set</td><td> $5 0 . 3 6 { \pm } 0 . 0 7$ </td><td> $0 . 9 9 7 2 0 { \scriptstyle \pm 0 . 0 0 0 0 1 }$ </td><td>3.115±0.023</td></tr><tr><td>Mean agg.</td><td> $5 1 . 4 3 { \pm } 0 . 0 2 $ </td><td> $0 . 9 9 7 5 8 { \scriptstyle \pm 0 . 0 0 0 0 1 }$ </td><td>2.784±0.012</td></tr><tr><td>Material  $\ell _ { 1 }$ </td><td> ${ \overline { { 5 1 . 5 9 } } } \pm 0 . 0 4$ </td><td> $\mathbf { 0 . 9 9 7 9 1 } { \pm } 0 . 0 0 0 0 1$ </td><td>2.731±0.009</td></tr><tr><td>Energy  $\ell _ { 1 }$ </td><td> $5 1 . 1 1 { \pm } 0 . 0 2 $ </td><td> $0 . 9 9 5 0 0 { \scriptstyle \pm 0 . 0 0 0 0 2 }$ </td><td>2.926±0.003</td></tr></table>

On the 50-view ore case, Fig. 11 compares reconstruction quality and observed cost on the same RTX 5090 GPU. Primary time spans worker start to final metric. The proposed method attains 35.01 dB mean PSNR across 3 repeats, versus 33.34 dB for R2+NNLS, while its median primary time is 354.514 s versus 183.081 s. The observed primary-time ranges are 348.035–449.293 s and 180.119–292.759 ${ \bf S } ,$ respectively. For these 2 methods, quality is the repeat mean and time the median. The remaining 4 methods have 1 observation each.

## C. Ablation Studies

We compare the complete method with 4 controlled variants on lizard head $( E = M = 3 )$ , powder phantom $( E = M = 4 )$ and MUSIC 3D $( E = M = 6 )$ . The variants hold the Gaussian set fixed, replace maximum with mean gradient aggregation, use a material-domain $\ell _ { 1 }$ objective, or use an energy-domain $\ell _ { 1 }$ objective. We use 3 paired seeds per case with the same saved 50,000-point initialization across settings within each case and seed. All settings run for 15,000 iterations. Table III reports the material-volume metrics. Fig. 12 shows each variant’s paired PSNR and SSIM differences from the complete setting.

![](images/6ec27bd14f9a5d1b3e5a791fc0fb19be6ad70fd553d9604e991f3ae223a2b751.jpg)  
Fig. 10. Per-material quality in 4 scene-specific energy/material configurations. Bars show PSNR (dB), dashed lines case means, shaded strips SSIM, and dots unscaled NRMSE. S/B denote soft tissue/bone, Ce/Zn denote $\mathrm { C e O _ { 2 } / Z n O , }$ and W/T denote water-like/tungsten-like. MUSIC uses a 256<sup>3</sup> grid, the others $8 0 ^ { 3 }$  
PSNR (dB)  
Ore V50 / interim

![](images/d7905a0afb3fad14dfb577914c20ad987c1f80deaeef6a83ebe2d000d0407208.jpg)

<table><tr><td rowspan=1 colspan=7>SSIM     NRMSE       MiB       n</td></tr><tr><td rowspan=1 colspan=1>FDK</td><td rowspan=1 colspan=1>0.787</td><td rowspan=1 colspan=1>0.041</td><td></td><td rowspan=1 colspan=1>500</td><td></td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>R2</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.022</td><td></td><td rowspan=1 colspan=1>1302</td><td></td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>CIL</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.026</td><td></td><td rowspan=1 colspan=1>504</td><td></td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>NeRF</td><td rowspan=1 colspan=1>0.885</td><td rowspan=1 colspan=1>0.024</td><td></td><td rowspan=1 colspan=1>10340</td><td></td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>ASD</td><td rowspan=1 colspan=1>0.875</td><td rowspan=1 colspan=1>0.027</td><td></td><td rowspan=1 colspan=1>504</td><td></td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>★Ours</td><td rowspan=1 colspan=1>0.951</td><td rowspan=1 colspan=1>0.018</td><td rowspan=1 colspan=2>1390</td><td></td><td rowspan=1 colspan=1>3</td></tr></table>

Fig. 11. Ore quality versus log time at 50 views on RTX 5090 with methodspecific budgets. Bubble area encodes maximum sampled process memory across repeats, not CUDA peak. The lower matrix gives SSIM, unscaled NRMSE, MiB, and repeat count. For $n = 3 ,$ time is the median and quality is the mean.

![](images/4b5124ab97a1eed33fd852230fb6f54f717e0cb061bba473e0066cb68d62f132.jpg)  
Fig. 12. Paired variant-minus-Full contrasts across 3 cases and seeds. Upper/lower maps show PSNR (dB)/SSIM differences. Blue is negative, ochre positive. Signed-asinh scales are centered at zero, with parameters 0.3 dB and 0.003 SSIM.

Material-domain $\ell _ { 1 }$ has the highest 3-seed mean PSNR and SSIM and the lowest mean NRMSE in each case. Mean aggregation also improves all 3 mean metrics relative to the complete setting, although its powder-phantom PSNR is slightly lower for 1 seed. Holding the Gaussian set fixed reduces mean PSNR in all cases, while its mean SSIM exceeds the complete setting for powder and MUSIC 3D. The energydomain $\ell _ { 1 }$ variant performs worse on all 3 mean metrics. Final representation sizes differ across settings, so these results do not isolate the quality effect of aggregation from the number of Gaussians.

## V. CONCLUSION

We introduced Gaussian material fields for volumetric multi-energy CT decomposition, combining shared anisotropic geometry with independent material coefficients. A differentiable spectral forward model enabled joint fitting from multi-energy projections, while material-aware density control adapted local capacity using material-specific evidence. Experiments using synthesized projections from public-data-derived pseudo-reference material maps showed improved material reconstruction and localized detail recovery, supported by material-wise comparisons and ablations. Runtime and memory measurements demonstrated computational benefits in the tested settings. This representation unified spatial organization, material composition, and local adaptation.

## APPENDIX

This appendix provides the derivations and proofs supporting the method in Sec. III, together with parameterization and implementation details.

Proposition 1 (Finite Gaussian ray response): For a positive definite $\Sigma _ { i } , \mathbf { a }$ unit ray direction, and fixed limits $t _ { r } ^ { - } < t _ { r } ^ { + }$ define $\begin{array} { r } { \beta _ { i , r } = \mathbf { d } _ { r } ^ { \mathsf { T } } \mathbf { Q } _ { i } \delta _ { i , r } } \end{array}$ and $\gamma _ { i , r } = \delta _ { i , r } ^ { \mathsf { T } } \mathbf { Q } _ { i } \delta _ { i , r }$ . The exact finite response is

$$
\begin{array} { r l } & { \ell _ { i , r } ^ { \mathrm { s e g } } = \sqrt { \frac { \pi } { 2 \alpha _ { i , r } } } e ^ { - \Delta _ { i , r } / 2 } \left[ \mathrm { e r f } ( z _ { i , r } ^ { + } / \sqrt { 2 } ) - \mathrm { e r f } ( z _ { i , r } ^ { - } / \sqrt { 2 } ) \right] , } \\ & { z _ { i , r } ^ { \pm } = \frac { \alpha _ { i , r } t _ { r } ^ { \pm } + \beta _ { i , r } } { \sqrt { \alpha _ { i , r } } } , \qquad \Delta _ { i , r } = \gamma _ { i , r } - \frac { \beta _ { i , r } ^ { 2 } } { \alpha _ { i , r } } . } \end{array}\tag{14}
$$

Its infinite-limit value is Eq. (4).

Proof: Suppressing subscripts, expand and complete the square:

$$
\begin{array} { c } { ( \delta + t \mathbf { d } ) ^ { \mathsf { T } } \mathbf { Q } ( \delta + t \mathbf { d } ) = \alpha t ^ { 2 } + 2 \beta t + \gamma } \\ { = \alpha ( t + \beta / \alpha ) ^ { 2 } + \Delta . } \end{array}\tag{15}
$$

The substitution $z = \sqrt { \alpha } ( t + \beta / \alpha )$ gives Eq. (14). The errorfunction bracket approaches 2 at infinite limits. If $F$ is the standard normal CDF, the omitted relative tail is

$$
\begin{array} { r l } & { 1 - \ell ^ { \mathrm { s e g } } / \ell ^ { \infty } = F ( z ^ { - } ) + 1 - F ( z ^ { + } ) } \\ & { ~ \leq \mathrm { e r f c } ( k / \sqrt { 2 } ) , ~ z ^ { - } \leq - k , ~ z ^ { + } \geq k . } \end{array}\tag{16}
$$

The bound holds for $k > 0$ by CDF monotonicity. It separates segment truncation from the later affine projection approximation.

Proposition 2 (Covariance-dependent rectification): For the positive definite locally transformed covariance, write

$$
\widetilde { \pmb { \Sigma } } = \left[ \mathbf { b } ^ { \mathsf { T } } \quad \mathbf { b } \right] , \qquad \xi = c - \mathbf { b } ^ { \mathsf { T } } \mathbf { S } ^ { - 1 } \mathbf { b } > 0 .\tag{17}
$$

Integrating the corresponding unit-peak Gaussian over the third coordinate gives the rectification factor $\begin{array} { r l } { \sqrt { 2 \pi \xi } } & { { } = } \end{array}$ $( 2 \pi | \widetilde { \pmb { \Sigma } } | / | \mathbf { S } | ) ^ { 1 / 2 }$ in Eq. (5).

Proof: For displacement $[ \mathbf { h } ^ { \mathsf { T } } , z ] ^ { \mathsf { T } }$ , the block inverse identity yields

$$
\begin{array} { r l } & { \left[ \underset { z } { \mathbf { h } } \right] ^ { \mathsf { T } } \widetilde { \boldsymbol { \Sigma } } ^ { - 1 } \left[ \underset { z } { \mathbf { h } } \right] = \mathbf { h } ^ { \mathsf { T } } \mathbf { S } ^ { - 1 } \mathbf { h } + \frac { ( z - \mathbf { b } ^ { \mathsf { T } } \mathbf { S } ^ { - 1 } \mathbf { h } ) ^ { 2 } } { \xi } , } \\ & { \qquad | \widetilde { \boldsymbol { \Sigma } } | = | \mathbf { S } | \xi . } \end{array}\tag{18}
$$

The integral of the second Gaussian factor over $z ~ \in ~ \mathbb { R }$ is $\scriptstyle { \sqrt { 2 \pi \xi } }$ . The first factor is independent of z. This proves the stated amplitude and detector-plane exponent. The identity is exact for the affine Gaussian. Its use for perspective rays retains the local approximation in the ray-space transform [13].

Proposition 3 (Conditional separation and conditioning): For fixed geometry and column-major vectorization,

$$
\begin{array} { r l r } & { } & { \mathrm { v e c } ( \widehat { \bf P } ) = ( { \bf B } \otimes \Phi ) \mathrm { v e c } ( { \bf A } ) , } \\ & { } & { \mathrm { r a n k } ( { \bf B } \otimes \Phi ) = \mathrm { r a n k } ( { \bf B } ) \mathrm { r a n k } ( \Phi ) . } \end{array}\tag{19}
$$

The unrestricted coefficient map is injective precisely when both factors have full column rank, requiring $E \ge M$ and

$R \ \geq \ N$ . For the diagnostic quadratic objective $\boldsymbol { \mathcal { Q } } ( \mathbf { A } ) \ =$ $\begin{array} { r } { \frac 1 2 \| \Phi \mathbf { A } \mathbf { B } ^ { \mathsf { T } } - \mathbf { P } \| _ { F } ^ { 2 } } \end{array}$ , its coefficient Hessian obeys

$$
\begin{array} { r } { \mathbf H _ { A } = ( \mathbf B ^ { \mathsf T } \mathbf B ) \otimes ( \mathbf \Phi ^ { \mathsf T } \mathbf \Phi ) , } \\ { \kappa _ { 2 } ( \mathbf H _ { A } ) = \kappa _ { 2 } ( \mathbf B ) ^ { 2 } \kappa _ { 2 } ( \mathbf \Phi ) ^ { 2 } \quad } \end{array}\tag{20}
$$

under those full-column-rank assumptions.

Proof: The vectorization identity gives the first line of Eq. (19). Applying thin SVDs to the two factors shows that nonzero singular values of the Kronecker product are their pairwise products. This proves the rank and injectivity claims. The quadratic Hessian is $( \mathbf { B } \otimes \Phi ) ^ { \mathsf { T } } ( \mathbf { B } \otimes \Phi )$ , giving Eq. (20). The corresponding unrestricted least-squares perturbation satisfies

$$
\begin{array} { r } { \delta \mathbf { A } _ { \mathrm { L S } } = \Phi ^ { \dagger } \delta \mathbf { P } ( \mathbf { B } ^ { \dagger } ) ^ { \mathsf { T } } , \quad \quad } \\ { \| \delta \mathbf { A } _ { \mathrm { L S } } \| _ { F } \leq \frac { \| \delta \mathbf { P } \| _ { F } } { \sigma _ { \operatorname* { m i n } } ( \Phi ) \sigma _ { \operatorname* { m i n } } ( \mathbf { B } ) } . } \end{array}\tag{21}
$$

The norm bound follows from submultiplicativity.

The quadratic model is used only to diagnose conditioning. It is not the Hessian of the actual $\ell _ { 1 }$ loss or a replacement training objective. Nonnegativity does not generally remove rank-deficient ambiguity at a strictly positive coefficient matrix: a sufficiently small nullspace perturbation preserves positivity. Boundary-constrained cases may differ. These results do not establish global uniqueness with trainable geometry. In particular, identical cloned kernels give duplicate columns of Φ at insertion. The fixed spectral basis is never jointly learned.

Proposition 4 (Geometric differentials): For fixed ray limits, exact Gaussian responses admit the derivatives below. For the rasterized model, Eq. (9) uses the full differential of Eq. (5), including its rectification factor and position-dependent covariance.

Proof: For $\mathbf { y } ( t ) = \mathbf { o } + t \mathbf { d } - \mu ,$ differentiation of the unitpeak kernel gives

$$
\begin{array} { r } { \nabla _ { \pmb { \mu } } K = K \mathbf { Q } \mathbf { y } , \qquad \nabla _ { \pmb { \Sigma } } K = \frac { 1 } { 2 } K \mathbf { Q } \mathbf { y } \mathbf { y } ^ { \mathsf { T } } \mathbf { Q } . } \end{array}\tag{22}
$$

There is no probability-density normalization term. Set $t _ { 0 } =$ $- \beta / \alpha , \bar { \mathbf { y } } = \delta + t _ { 0 } \mathbf { d } , k ^ { \pm } = K ( \mathbf { x } _ { r } ( t ^ { \pm } ) )$ , and

$$
\begin{array} { l } { { M _ { 0 } = \ell ^ { \mathrm { s e g } } , \qquad M _ { 1 } = ( k ^ { - } - k ^ { + } ) / \alpha , } } \\ { { M _ { 2 } = \{ M _ { 0 } + ( t ^ { - } - t _ { 0 } ) k ^ { - } - ( t ^ { + } - t _ { 0 } ) k ^ { + } \} / \alpha . } } \end{array}\tag{23}
$$

Integration by parts identifies $\begin{array} { r } { M _ { j } = \int _ { t ^ { - } } ^ { t ^ { + } } ( t - t _ { 0 } ) ^ { j } K } \end{array}$ dt for $j =$ $0 , 1 , 2 .$ . Differentiating under the integral in Eq. (22) therefore yields

$$
\begin{array} { r l } & { \nabla _ { \pmb { \mu } } \ell ^ { \mathrm { s e g } } = \mathbf { Q } ( M _ { 0 } \bar { \mathbf { y } } + M _ { 1 } \mathbf { d } ) , } \\ & { \nabla _ { \pmb { \Sigma } } \ell ^ { \mathrm { s e g } } = \frac { 1 } { 2 } \mathbf { Q } \big [ M _ { 0 } \bar { \mathbf { y } } \bar { \mathbf { y } } ^ { \mathsf { T } } + M _ { 2 } \mathbf { d } \mathbf { d } ^ { \mathsf { T } } } \\ & { \qquad + M _ { 1 } ( \bar { \mathbf { y } } \mathbf { d } ^ { \mathsf { T } } + \mathbf { d } \bar { \mathbf { y } } ^ { \mathsf { T } } ) \big ] \mathbf { Q } . } \end{array}\tag{24}
$$

The infinite-line case has $M _ { 1 } = 0$ and $M _ { 2 } = \ell ^ { \infty } / \alpha .$

For the rectified response, write $\mathbf { S } = { \widehat { \boldsymbol { \Sigma } } } , \mathbf { r } = \mathbf { S } ^ { - 1 } ( \mathbf { u } - { \widehat { \boldsymbol { \mu } } } )$ and let ${ \bf E } _ { 2 } = [ { \bf I } _ { 2 } \ { \bf 0 } ]$ select the first two coordinates. Logarithmic differentiation gives

$$
\begin{array} { r l } & { \mathrm { d } \phi = \phi \big [ \mathbf { r } ^ { \mathsf { T } } \mathrm { d } \widehat { \pmb { \mu } } + \langle \mathbf { H } , \mathrm { d } \widetilde { \pmb { \Sigma } } \rangle _ { F } \big ] , } \\ & { \mathbf { H } = \frac { 1 } { 2 } \big [ \widetilde { \pmb { \Sigma } } ^ { - 1 } + \mathbf { E } _ { 2 } ^ { \mathsf { T } } ( \mathbf { r } \mathbf { r } ^ { \mathsf { T } } - \mathbf { S } ^ { - 1 } ) \mathbf { E } _ { 2 } \big ] . } \end{array}\tag{25}
$$

The first inverse-covariance term comes from log $| \widetilde { \pmb { \Sigma } } |$ , and the negative $\mathbf { S } ^ { - 1 }$ term comes from − log |S|. Both would be lost

by freezing the amplitude correction. With $\mathbf { T } = \mathbf { J } \mathbf { W } _ { v }$ and $\Lambda = \mathrm { d i a g } ( \sigma ^ { 2 } )$ , the chain rule is

$$
\begin{array} { r l } & { \mathrm { d } \tilde { \Sigma } = \mathbf { T } \mathrm { d } \Sigma \mathbf { T } ^ { \mathsf { T } } + ( \mathrm { d } \mathbf { T } ) \Sigma \mathbf { T } ^ { \mathsf { T } } + \mathbf { T } \Sigma ( \mathrm { d } \mathbf { T } ) ^ { \mathsf { T } } , } \\ & { \mathrm { d } \Sigma = ( \mathrm { d } \mathbf { R } ) \Lambda \mathbf { R } ^ { \mathsf { T } } + \mathbf { R } \Lambda ( \mathrm { d } \mathbf { R } ) ^ { \mathsf { T } } } \\ & { \qquad + \mathbf { R } \mathrm { d i a g } ( 2 \sigma \odot \mathrm { d } \sigma ) \mathbf { R } ^ { \mathsf { T } } , } \\ & { \mathrm { d } \bar { \mathbf { q } } = \frac { \mathbf { I } _ { 4 } - \bar { \mathbf { q } } \bar { \mathbf { q } } ^ { \mathsf { T } } } { \| \mathbf { q } \| _ { 2 } } \mathrm { d } \mathbf { q } , \qquad \mathrm { d } \mathbf { T } = ( \mathrm { d } \mathbf { J } ) \mathbf { W } _ { v } . } \end{array}\tag{26}
$$

Here dR follows the normalized quaternion convention. In perspective geometry J depends on the center, so its differential cannot generally be discarded. Substituting these differentials into Eq. (8) proves the parameter gradients in Eq. (9). They hold locally with fixed culling decisions. Hard active-set changes are not differentiated.

Proposition 5 (Material-specific refinement evidence): For the separate projected-center branches in Eq. (10), their sum is the shared projected-center partial derivative. Their maximum norm satisfies

$$
\left\| \frac { 1 } { M } \sum _ { m } \mathbf { h } _ { i , m } \right\| _ { 2 } \leq \frac { 1 } { M } \sum _ { m } \| \mathbf { h } _ { i , m } \| _ { 2 } \leq \operatorname* { m a x } _ { m } \| \mathbf { h } _ { i , m } \| _ { 2 } .\tag{27}
$$

Proof: The projected-center derivative of a footprint with fixed covariance is $\nabla _ { \widehat { \mu } } \Phi _ { r , i } = \Phi _ { r , i } \widehat { \boldsymbol { \Sigma } } ^ { - 1 } \rho _ { i , r }$ . Its coefficient in material branch m is $a _ { i , m } D _ { r , m }$ , giving Eq. (10). Summing branches gives the shared-center contribution. The triangle inequality and the bound of an arithmetic mean by the largest entry prove Eq. (27).

The result prevents averaging-induced dilution, but it does not guarantee accurate recovery or optimal allocation. Temporal averaging and material maximization do not commute: responses (2, 0) and (0, 2) over two iterations have mean-ofmaxima 2 but max-of-means 1 before denominator damping. Equation (11) uses the former.

Proposition 6 (Effect ofprimitive edits): Under additive rendering with unchanged other primitives, a clone with identical kernels and halved activated coefficients preserves the field and prediction at insertion. The splitting rule in Sec. III-D conserves coefficient sums but not spatial integrals in general. For any resulting prediction perturbation,

$$
\big | \mathcal { L } ( \widehat { \mathbf { P } } + \delta \widehat { \mathbf { P } } ) - \mathcal { L } ( \widehat { \mathbf { P } } ) \big | \leq \frac { 1 } { R } \| \delta \widehat { \mathbf { P } } \mathbf { W } _ { E } \| _ { 1 , 1 } .\tag{28}
$$

Here $\mathcal { L } ( \widehat { \mathbf { P } } ) = \| ( \widehat { \mathbf { P } } - \mathbf { P } ) \mathbf { W } _ { E } \| _ { 1 , 1 } / R .$

Proof: The field perturbation corresponding to Eq. (13) is

$$
\delta \mathbf { c } _ { i } ( \mathbf { x } ) = \sum _ { j \in \mathcal { C } _ { i } } K _ { i , j } ( \mathbf { x } ) \mathbf { a } _ { i , j } - K _ { i } ( \mathbf { x } ) \mathbf { a } _ { i } .\tag{29}
$$

For a clone, $K _ { i , 1 } = K _ { i , 2 } = K _ { i }$ and $\mathbf { a } _ { i , 1 } = \mathbf { a } _ { i , 2 } = \mathbf { a } _ { i } / 2$ making Eqs. (13) and (29) zero. This identity requires identical masks and no amplitude-dependent cutoff.

For a split, the parent is replaced by $N _ { c } ~ = ~ 2$ children according to

$$
\begin{array} { l } { { \pmb { \mu } } _ { i , j } = { \pmb \mu } _ { i } + { \bf R } _ { i } \mathrm { d i a g } ( { \pmb \sigma } _ { i } ) { \pmb \epsilon } _ { j } , \quad { \pmb \epsilon } _ { j } \sim \mathcal { N } ( { \bf 0 } , { \bf I } _ { 3 } ) , } \\ { { \pmb \sigma } _ { i , j } = { \pmb \sigma } _ { i } / k _ { c } , \qquad k _ { c } = 0 . 8 N _ { c } , } \\ { { \bf R } _ { i , j } = { \bf R } _ { i } , \qquad { \bf a } _ { i , j } = { \bf a } _ { i } / N _ { c } . } \end{array}\tag{30}
$$

For the unnormalized kernels in Eq. (2),

$$
\begin{array} { c } { \displaystyle \mathbf { m } _ { i } : = \int _ { \mathbb { R } ^ { 3 } } K _ { i } ( \mathbf { x } ) \mathbf { a } _ { i } \mathrm { d } \mathbf { x } = ( 2 \pi ) ^ { 3 / 2 } | \Sigma _ { i } | ^ { 1 / 2 } \mathbf { a } _ { i } , } \\ { \displaystyle \sum _ { j } \mathbf { a } _ { i , j } = \mathbf { a } _ { i } , \qquad \sum _ { j } \mathbf { m } _ { i , j } = k _ { c } ^ { - 3 } \mathbf { m } _ { i } . } \end{array}\tag{31}
$$

The second identity follows from $\Sigma _ { i , j } = \Sigma _ { i } / k _ { c } ^ { 2 }$ , provided no additional scale clipping changes Eq. (30). For $N _ { c } = 2 $ the whole-space integral ratio is $1 / 1 . 6 ^ { 3 } \approx 0 . 2 4 4 1 4$ , not one. Finally, applying $| | x + y | - | x | | \leq | y |$ entrywise proves Eq. (28).

A clone does not increase the column span at insertion, and identical copies may remain identical if all updates and optimizer states are symmetric. A split generally changes the loss, with no sign guaranteed for that change. These properties are not convergence claims.

For pruning a primitive, $\delta \widehat { \mathbf { P } } _ { i } = - \phi _ { i } ( \mathbf { B } \mathbf { a } _ { i } ) ^ { \mathsf { T } }$ . Nonnegativity implies the exact perturbation size

$$
\frac { 1 } { R } \| \delta \widehat { \mathbf { P } } _ { i } \mathbf { W } _ { E } \| _ { 1 , 1 } = \frac { \| \phi _ { i } \| _ { 1 } } { R } \mathbf { w } ^ { \top } \mathbf { B } \mathbf { a } _ { i } ,\tag{32}
$$

where $\mathbf { w } = [ w _ { 1 } , \ldots , w _ { E } ] ^ { \mathsf { T } }$ . If all amplitudes are below $\tau _ { a } ,$ replace ${ \bf a } _ { i }$ by $\tau _ { a } \mathbf { 1 } _ { M }$ to obtain an upper bound. Thus small peak amplitude alone is not a scale-independent error guarantee. Similarly, a center outside Ω need not have zero Gaussian contribution inside it.

Parameter maps and projection-only initialization. With finite logits, the scale parameterizations are

$$
\pmb { \sigma } _ { i } = \left\{ \begin{array} { l l } { \sigma _ { \operatorname* { m i n } } \mathbf { 1 } _ { 3 } + ( \sigma _ { \operatorname* { m a x } } - \sigma _ { \operatorname* { m i n } } ) \mathrm { s i g m o i d } ( \eta _ { i } ) , } & { \mathrm { b o u n d e d } , } \\ { \exp ( \eta _ { i } ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{33}
$$

where $0 ~ < ~ \sigma _ { \mathrm { m i n } } ~ < ~ \sigma _ { \mathrm { m a x } }$ . Their differentials are obtained elementwise: dσ $= \sigma \mathrm { d } \eta$ in the exponential branch, or dσ = $( \sigma _ { \mathrm { m a x } } - \sigma _ { \mathrm { m i n } } ) s ( 1 - s ) \mathrm { d }$ η with $s = \mathrm { s i g m o i d } ( \eta )$ in the bounded branch.

Initialization uses only measured projections:

$$
\begin{array} { r l } & { \qquad V _ { e } ^ { ( 0 ) } = \mathrm { F D K } ( \mathbf { p } _ { e } ) , } \\ & { \qquad \mathbf { c } ^ { ( 0 ) } ( \mathbf { x } ) \in \underset { \mathbf { z } \geq 0 } { \mathrm { a r g m i n } } \ : \| \mathbf { B } \mathbf { z } - \mathbf { v } ^ { ( 0 ) } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } . } \end{array}\tag{34}
$$

Set $s ^ { ( 0 ) } = \mathbf { 1 } _ { M } ^ { \top } \mathbf { c } ^ { ( 0 ) }$ and $\Omega _ { \tau } = \{ \mathbf { x } : s ^ { ( 0 ) } ( \mathbf { x } ) > \tau _ { \mathrm { i n i t } } \}$ . Initial centers are sampled from $\Omega _ { \tau }$ . When candidates are insufficient, highest-score locations and sampling with replacement are permitted. With nearest-neighbor distance $\begin{array} { r } { d _ { i } = \operatorname* { m i n } _ { j \neq i } \| \pmb { \mu } _ { i } ^ { ( \dot { 0 } ) } - } \end{array}$ $\mu _ { j } ^ { ( 0 ) } \lVert _ { 2 }$ , the initialization targets are

$$
\begin{array} { r l r } & { \widetilde { \mathbf { a } } _ { i } ^ { ( 0 ) } = \eta _ { \mathrm { i n i t } } \mathbf { c } ^ { ( 0 ) } ( \pmb { \mu } _ { i } ^ { ( 0 ) } ) , } & \\ & { \widetilde { \pmb { \sigma } } _ { i } ^ { ( 0 ) } = \mathrm { c l i p } ( d _ { i } , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ) \mathbf { 1 } _ { 3 } , \quad } & { \mathbf { R } _ { i } ^ { ( 0 ) } = \mathbf { I } _ { 3 } . } & \end{array}\tag{35}
$$

For positive amplitudes the inverse-Softplus map is

$$
\beta = \log ( e ^ { a } - 1 ) = a + \log ( 1 - e ^ { - a } ) , \qquad a > 0 .\tag{36}
$$

A zero NNLS coefficient is not representable by finite Softplus logits, and a clipped scale at a sigmoid endpoint requires an interior numerical convention. Finite initialization therefore requires a positive numerical floor and an interior convention for bounded scales. The tildes in Eq. (35) distinguish targets from finite activated parameters. Cloning and splitting divide activated coefficients, not their logits. Bound-violating split scales likewise require explicit handling. The integral identity in Eq. (31) applies to the stated, unclipped split rule.

Continuous queries and computational accounting. For arbitrary query locations, set $[ \mathbf { K } _ { Q } ] _ { q , i } = K _ { i } ( \mathbf { x } _ { q } ; \Theta _ { T } )$ . The final material and energy samples are

$$
\begin{array} { r } { { \bf C } _ { Q } = { \bf K } _ { Q } { \bf A } _ { T } , \qquad { \bf V } _ { Q } = { \bf C } _ { Q } { \bf B } ^ { \top } . } \end{array}\tag{37}
$$

These are point samples. Denser queries do not establish additional physical resolution. The stored trainable scalar counts are $N ( 1 0 + M )$ for the shared representation and 11MN for M independent single-material sets of N primitives each. They are not intrinsic degrees of freedom because quaternion normalization is redundant. Separate material rasterization with $K _ { v }$ effective Gaussian–pixel interactions per view costs

$$
\begin{array} { r l } & { \mathrm { ~ v i e w ~ e v a l u a t i o n : ~ } \mathcal { O } ( M K _ { v } ) + \mathcal { O } ( E M H W ) , } \\ & { \mathrm { t r a i n a b l e ~ s t o r a g e : ~ } \mathcal { O } ( N ( 1 0 + M ) ) . } \end{array}\tag{38}
$$

The second work term is mixing on an $H \times W$ detector. Sorting/culling, optimizer states, initialization, and output arrays are additional costs. Explicitly storing material samples requires $\mathcal { O } ( Q M )$ space. No fused multi-channel acceleration or material-count-independent runtime is assumed.

## REFERENCES

[1] R. E. Alvarez and A. Macovski, “Energy-selective reconstructions in X-ray computerized tomography,” Phys. Med. Biol., vol. 21, no. 5, pp. 733–744, 1976.

[2] C. H. McCollough, S. Leng, L. Yu, and J. G. Fletcher, “Dual- and multienergy CT: principles, technical approaches, and clinical applications,” Radiology, vol. 276, no. 3, pp. 637–653, 2015.

[3] M. J. Willemink, M. Persson, A. Pourmorteza, N. J. Pelc, and D. Fleischmann, “Photon-counting CT: technical principles and clinical prospects,” Radiology, vol. 289, no. 2, pp. 293–312, 2018.

[4] C. H. McCollough et al., “Principles and applications of multienergy CT: report of AAPM task group 291,” Med. Phys., vol. 47, no. 7, pp. e881–e912, 2020.

[5] A. So and S. Nicolaou, “Spectral computed tomography: fundamental principles and recent developments,” Korean J. Radiol., vol. 22, no. 1, pp. 86–96, 2021.

[6] Y. Long and J. A. Fessler, “Multi-material decomposition using statistical image reconstruction for spectral CT,” IEEE Trans. Med. Imaging, vol. 33, no. 8, pp. 1614–1626, 2014.

[7] K. Mechlem et al., “Joint statistical iterative material image reconstruction for spectral computed tomography using a semi-empirical forward model,” IEEE Trans. Med. Imaging, vol. 37, no. 1, pp. 68–80, 2018.

[8] R. F. Barber, E. Y. Sidky, T. G. Schmidt, and X. Pan, “An algorithm for constrained one-step inversion of spectral CT data,” Phys. Med. Biol., vol. 61, no. 10, pp. 3784–3818, 2016.

[9] M. Tivnan, W. Wang, and J. W. Stayman, “A preconditioned algorithm for model-based iterative CT reconstruction and material decomposition from spectral CT data,” arXiv preprint arXiv:2010.01371, 2020. [Online]. Available: https://arxiv.org/abs/2010.01371

[10] T. Hotta, T. Yatagawa, Y. Ohtake, and T. Aoki, “NeMCoF: neural material composition fields for material decomposition in sparse-view spectral X-ray CT,” J. Nondestruct. Eval., vol. 44, 2025, Art. no. 122.

[11] L. Shi, C. Liu, P. Yang, J. Qiu, and X. Zhao, “Ray-driven spectral CT reconstruction based on neural base-material fields,” arXiv preprint arXiv:2404.06991v1, 2024. [Online]. Available: https://arxiv.org/abs/ 2404.06991v1

[12] Y. Cai et al., “Radiative Gaussian splatting for efficient X-ray novel view synthesis,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 283–299.

[13] R. Zha, T. J. Lin, Y. Cai, J. Cao, Y. Zhang, and H. Li, “R<sup>2</sup>-Gaussian: rectifying radiative Gaussian splatting for tomographic reconstruction,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 37, 2024, pp. 44 907– 44 934.

[14] Y. Li, X. Fu, H. Li, S. Zhao, R. Jin, and S. K. Zhou, “3DGR-CT: sparseview CT reconstruction with a 3D Gaussian representation,” Med. Image Anal., vol. 103, 2025, Art. no. 103585.

[15] S. Wu et al., “Discretized Gaussian representation for tomographic reconstruction,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 25 073–25 082.

[16] R. Vos, S. N. Sinha, and M. Weinmann, “SpectralCTGaussians: projection-domain reconstruction and basis material decomposition for spectral CT using 3D Gaussian splatting,” arXiv preprint arXiv:2609.29638, 2026. [Online]. Available: https://arxiv.org/abs/2609. 29638

[17] T. Niu, X. Dong, M. Petrongolo, and L. Zhu, “Iterative image-domain decomposition for dual-energy CT,” Med. Phys., vol. 41, no. 4, 2014, Art. no. 041901.

[18] Y. Xue et al., “Statistical image-domain multimaterial decomposition for dual-energy CT,” Med. Phys., vol. 44, no. 3, pp. 886–901, 2017.

[19] Y. Zhao, X. Zhao, and P. Zhang, “An extended algebraic reconstruction technique (E-ART) for dual spectral CT,” IEEE Trans. Med. Imaging, vol. 34, no. 3, pp. 761–768, 2015.

[20] M. Li, Y. Zhao, and P. Zhang, “Accurate iterative FBP reconstruction method for material decomposition of dual energy CT,” IEEE Trans. Med. Imaging, vol. 38, no. 3, pp. 802–812, 2019.

[21] H. Pan, S. Zhao, W. Zhang, H. Zhang, and X. Zhao, “Fast iterative reconstruction for multi-spectral CT by a Schmidt orthogonal modification algorithm (SOMA),” Inverse Probl., vol. 39, no. 8, 2023, Art. no. 085001.

[22] L. I. Rudin, S. Osher, and E. Fatemi, “Nonlinear total variation based noise removal algorithms,” Physica D, vol. 60, no. 1–4, pp. 259–268, 1992.

[23] E. Y. Sidky and X. Pan, “Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization,” Phys. Med. Biol., vol. 53, no. 17, pp. 4777–4807, 2008.

[24] J. R. Rajagopal et al., “Development of a deep learning based approach for multi-material decomposition in spectral CT: a proof of principle in silico study,” Sci. Rep., vol. 15, no. 1, 2025, Art. no. 28814.

[25] L. Hellwege, J. C. Engster, M. Schaar, T. M. Buzug, and M. Stille, “Unsupervised physics-informed deep learning for dual-energy CT material decomposition,” arXiv preprint arXiv:2604.25356, 2026. [Online]. Available: https://arxiv.org/abs/2604.25356

[26] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: representing scenes as neural radiance fields for view synthesis,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 405–421.

[27] D. Ruckert, Y. Wang, R. Li, R. Idoughi, and W. Heidrich, “NeAT: neural¨ adaptive tomography,” ACM Trans. Graph., vol. 41, no. 4, pp. 55:1– 55:13, 2022.

[28] R. Zha, Y. Zhang, and H. Li, “NAF: neural attenuation fields for sparseview CBCT reconstruction,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2022, pp. 442–452.

[29] G. Zang, R. Idoughi, R. Li, P. Wonka, and W. Heidrich, “IntraTomo: self-supervised learning-based tomography via sinogram synthesis and prediction,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 1940–1950.

[30] Q. Wu, H. Wei, J. Yu, S. K. Zhou, and Y. Zhang, “JSolver: joint spectrum estimation and multi-material decomposition from single-energy CT projections,” arXiv preprint arXiv:2505.08123v2, 2025. [Online]. Available: https://arxiv.org/abs/2505.08123v2

[31] X. Jiang, G. J. Gang, and J. W. Stayman, “Multi-material decomposition using spectral diffusion posterior sampling,” IEEE Trans. Biomed. Eng., vol. 72, no. 8, pp. 2447–2461, Aug. 2025.

[32] X. Jiang, G. J. Gang, and J. W. Stayman, “Volumetric material decomposition using spectral diffusion posterior sampling with a compressed polychromatic forward model,” arXiv preprint arXiv:2503.22392, 2025. [Online]. Available: https://arxiv.org/abs/2503.22392

[33] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3D Gaussian¨ splatting for real-time radiance field rendering,” ACM Trans. Graph., vol. 42, no. 4, pp. 139:1–139:14, 2023.

[34] X. Fu et al., “3DGR-CAR: coronary artery reconstruction from ultrasparse 2D X-ray views with a 3D Gaussians representation,” arXiv preprint arXiv:2410.00404, 2024. [Online]. Available: https://arxiv.org/ abs/2410.00404

[35] Z. Gao, B. Planche, M. Zheng, X. Chen, T. Chen, and Z. Wu, “DDGS-CT: direction-disentangled Gaussian splatting for realistic volume rendering,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 37, 2024, pp. 39 281–39 302.

[36] P. T. Pieta, R. J. Pedersen, S. Borgi, J. S. Jørgensen, J. W. Andreasen, and V. A. Dahl, “FaCT-GS: fast and scalable CT reconstruction with

Gaussian splatting,” arXiv preprint arXiv:2604.01844v2, 2026, accepted at ECCV. [Online]. Available: https://arxiv.org/abs/2604.01844v2

[37] H. Zhou et al., “Structurally informed 3-D Gaussian splatting for limited-angle CBCT,” IEEE Trans. Med. Imaging, vol. 45, no. 8, pp. 4482–4493, Aug. 2026.

[38] L. Xing, D. Jin, K. Bu, P. He, and S. Ying, “LINGO: latent initialization and gradient optimization for sparse-view X-ray novel view synthesis and CT reconstruction with 3D Gaussian splatting,” arXiv preprint arXiv:2609.22849, 2026. [Online]. Available: https://arxiv.org/abs/2609. 22849

[39] L. A. Feldkamp, L. C. Davis, and J. W. Kress, “Practical cone-beam algorithm,” J. Opt. Soc. Am. A, vol. 1, no. 6, pp. 612–619, 1984.

[40] C. L. Lawson and R. J. Hanson, Solving Least Squares Problems. Philadelphia, PA, USA: Society for Industrial and Applied Mathematics, 1995.

[41] C. H. McCollough et al., “Low-dose CT for the detection and classification of metastatic liver lesions: results of the 2016 low dose ct grand challenge,” Med. Phys., vol. 44, no. 10, pp. e339–e352, 2017.

[42] R. Warr et al., “Hyperspectral X-ray CT dataset of a single, iodinestained lizard head sample,” Zenodo, data set, 2021. [Online]. Available: https://doi.org/10.5281/zenodo.5013680

[43] R. Warr et al., “Enhanced hyperspectral tomography for bioimaging by spatiospectral reconstruction,” Sci. Rep., vol. 11, 2021, Art. no. 20818.

[44] R. Warr, J. S. Jørgensen, E. Papoutsellis, E. Ametova, R. Cernik, and P. J. Withers, “Hyperspectral X-ray CT datasets of an aluminium phantom containing three metal-based powders,” Zenodo, data set, 2022, version 3. [Online]. Available: https://doi.org/10.5281/zenodo.5825464

[45] R. Warr, S. Handschuh, M. Glosmann, R. J. Cernik, and P. J. Withers,¨ “Hyperspectral X-ray CT datasets for a set of multiply-stained mouse limb specimens,” Zenodo, data set, 2022. [Online]. Available: https:// doi.org/10.5281/zenodo.6787594

[46] R. Warr, S. Handschuh, M. Glosmann, R. J. Cernik, and P. J. Withers,¨ “Quantifying multiple stain distributions in bioimaging by hyperspectral X-ray tomography,” Sci. Rep., vol. 12, 2022, Art. no. 21945.

[47] C. Kehl, W. Mustafa, J. Kehres, A. B. Dahl, and U. L. Olsen, “Multispectral imaging via computed tomography (MUSIC) – comparing unsupervised spectral segmentations for material differentiation,” arXiv preprint arXiv:1810.11823, 2018. [Online]. Available: https://arxiv.org/ abs/1810.11823

[48] E. Zhou et al., “A cone-beam photon-counting CT dataset for spectral image reconstruction and deep learning,” Sci. Data, vol. 12, 2025, Art. no. 1955.

[49] R. Warr et al., “Hyperspectral X-ray CT data set of mineralised ore sample with Au and Pb deposits,” Zenodo, data set, 2020. [Online]. Available: https://doi.org/10.5281/zenodo.4157615

[50] M. T. Zeegers, A. Kadu, T. van Leeuwen, and K. J. Batenburg, “ADJUST: a dictionary-based joint reconstruction and unmixing method for spectral tomography,” Inverse Probl., vol. 38, no. 12, 2022, Art. no. 125002.

[51] J. S. Jørgensen et al., “Core Imaging Library – part I: a versatile Python framework for tomographic imaging,” Philos. Trans. R. Soc. A, vol. 379, no. 2204, 2021, Art. no. 20200192.