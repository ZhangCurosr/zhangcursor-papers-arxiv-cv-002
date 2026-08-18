# Shared-Structure 4D Spectral Gaussian Representation for Sparse-View Spectral CT Reconstruction

Jiancheng Fang, Shaoyu Wang, Wenjun Xia, Member, IEEE, Yang Chen, Senior Member, IEEE, and Qiegen Liu, Senior Member, IEEE

Abstract—Sparse-view spectral computed tomography (CT) reconstructs energy-resolved attenuation volumes from limited projection views, requiring simultaneous handling of angular undersampling and spectral coupling. We propose a Shared-Structure 4D Spectral Gaussian Representation (4D-SG) that learns shared Gaussian geometry from full spectrum structural projections and uses a Gaussian-wise Spectral Density Curve Network (GSC-Net) to predict Gaussian raw density transformations. This factorization separates shared spatial structure from spectral attenuation variation, avoids independent channel geometry optimization, and establishes a continuous 4D-SG representation from discrete spectral measurements for unobserved spectral channel queries. Experiments on six synthesized, simulated projection, and real projection datasets with 50 views demonstrate the best average performance. Compared with the strongest Gaussian baseline, 4D-SG improves PSNR from 35.56 dB to 36.61 dB, increases SSIM from 0.909 to 0.914, and reduces LPIPS from 0.208 to 0.194, demonstrating its effectiveness for sparse-view spectral CT reconstruction.

Index Terms—Sparse-view reconstruction, spectral CT, Gaus sian splatting, spectral modeling.

## I. INTRODUCTION

PECTRAL computed tomography (CT), including dual-S energy CT, multi-energy CT, and photon-counting CT, provides energy-resolved measurements beyond conventional attenuation-based CT [1]–[3]. Its energy-dependent attenuation enables multi-energy attenuation imaging, material discrimination, virtual monoenergetic imaging, and other quantitative spectral imaging applications. Reconstruction requires multiple correlated attenuation volumes with different spectral responses, making spectral modeling across channels essential.

![](images/24353519d20ccd255f480f3c8d485b2ca68d8a4cd49b830a3374f638778a0b28.jpg)  
Fig. 1. Conceptual comparison. (a) Conventional methods reconstruct each spectral channel through an independent iterative process. (b) 4D-SG learns shared Gaussian geometry and GSC-Net from discrete spectral measurements and sparse-view projections and generates the Gaussian representation of a target channel for voxelization, supporting reconstruction and prediction of observed and unobserved spectral channels.

Sparse-view acquisition reduces radiation dose, scan time, and system complexity in spectral CT. However, angular undersampling and spectral channel separation occur simultaneously: each channel contains incomplete projection information, while all channels correspond to the same object. Reconstruction must therefore suppress sparse-view artifacts, preserve structural consistency across channels, and retain attenuation differences between spectral channels [4]–[6].

Existing spectral CT methods exploit physical forward models, correlations across channels, or learned image priors. Model-based approaches include statistical reconstruction, joint regularization, tensor representations, basis-image estimation, and calibrated spectral forward models [4], [5], [7]–[10], while learning-based approaches use deep spectral imaging pipelines, model-based unrolling, learned structural priors, and generative constraints [6], [11]–[15]. Most of these methods still represent spectral CT volumes as voxel grids, image tensors, or separate channel outputs, coupling shared spatial structure with spectral attenuation variation and limiting controlled reuse of common 3D structure.

Representation learning offers an alternative for sparse-view tomographic reconstruction. Implicit neural representations model continuous attenuation fields as functions of spatial coordinates and have been applied to sparse-view CT, cone-beam CT, and X-ray 3D reconstruction [16]–[19]. More recently, 3D Gaussian Splatting (3DGS) has provided compact learnable Gaussian primitives and efficient differentiable rendering [20], and Gaussian representations have been extended to X-ray imaging and CT reconstruction [21]–[24]. However, existing methods mainly address generic tomography, cone-beam CT, or single-energy sparse-view CT rather than shared object geometry with varying spectral attenuation responses.

We propose a Shared-Structure 4D Spectral Gaussian Representation (4D-SG) for sparse-view spectral CT reconstruction. Instead of assigning an independent volumetric representation to each spectral channel, 4D-SG learns shared Gaussian geometry for common 3D structure and Gaussian raw density curves for spectral attenuation variation, as illustrated in Fig. 1.

The main contributions are summarized as follows:

• We introduce, to the best of our knowledge, the first Gaussian-splatting framework for 4D spectral CT reconstruction; 4D-SG separates shared Gaussian geometry from attenuation responses across spectral channels.

• We develop a Gaussian-wise Spectral Density Curve Network (GSC-Net) to predict raw density transformations over the ordered spectral axis, using channel correlations while preserving channel-specific attenuation differences.

• We establish a continuous 4D-SG representation over the spectral axis from discrete measurements, supporting reconstruction conditioned on the spectral coordinate and queries of unobserved channels for flexible recovery of multi-energy attenuation volumes.

## II. RELATED WORK

## A. Spectral CT Reconstruction

Existing spectral CT reconstruction methods can be broadly categorized into model-based and learning-based approaches. Spectral CT, including dual-energy, multi-energy, and photoncounting CT, reconstructs multiple energy-dependent attenuation images, making channel correlation particularly important under sparse-view acquisition [1], [3], [25].

Model-based approaches include statistical image reconstruction for material decomposition [7], joint multi-channel or total-image-constrained regularization [4], [26], tensor-based dictionary learning [5], basis-image estimation [8], and joint statistical iterative reconstruction with calibrated forward models [9]. More recent methods jointly estimate image content and spectral response parameters [10]. Although physically interpretable and effective at exploiting spectral coupling, these approaches require repeated projection and backprojection over multiple channels and can be computationally costly for 3D sparse-view reconstruction.

Learning-based spectral CT reconstruction methods [11], [27] include deep spectral imaging pipelines and challengedriven reconstruction [28], [29], model-based unrolling [12], [30], learned texture or structural priors [13], [31], and generative or structural constraints for sparse-view spectral reconstruction [6], [14], [15]. They improve artifact suppression and detail recovery but generally rely on voxel grids, image tensors, or separate spectral-channel outputs, coupling shared spatial structure with spectral attenuation variation.

![](images/fece099b5ddf17668704cc6caa5a1c3a84ec18a4950ec8ba4f304f27011a8386.jpg)

![](images/5ea869d99b5119df3261e33b7b881901920dd53f85148cc689492011fbe6f625.jpg)  
Fig. 2. Empirical analysis of structural consistency across channels and attenuation variation by region in spectral CT. (a) Representative spectral CT slices, corresponding edge maps, and distributions of the shared structural edge ratio under different thresholds k for the minimum number of concurring channels. Violin plots, samples from individual slices, box plots, and median markers summarize the distributions; the dashed line denotes 0.95, and the bottom annotations report the proportion of samples with low ratios and the minimum ratio. (b) Four fixed regions of interest, enlarged patches, and their mean globally normalized attenuation values across spectral channels, with the measured values annotated at each channel.

## B. Implicit Neural and Gaussian Representations

Implicit neural representations model attenuation fields as continuous spatial functions and have been applied to sparseview and cone-beam CT reconstruction [16]–[19]. Despite reducing dependence on voxel grids, they require dense coordinate sampling and repeated multilayer perceptron evaluations, limiting reconstruction and rendering efficiency.

3DGS represents geometry and attributes using learnable Gaussian primitives, providing a compact parameterization and efficient differentiable rendering [20]. Gaussian representations have been extended to X-ray novel view synthesis, tomographic reconstruction, extremely sparse-view cone-beam CT reconstruction, and sparse-view CT reconstruction [21]–[24]. However, these methods mainly target generic tomography, cone-beam CT, or single-energy sparse-view CT. Separation of shared Gaussian geometry and spectral attenuation variation remains unaddressed in spectral CT.

![](images/91fb3afa1683ade7264523e52d3f929cbd19a01e7ad8726f3cf8d5d936291f3e.jpg)  
Fig. 3. Workflow of 4D-SG. Shared Gaussian geometry and common raw densities are first learned from structural projections over the full spectrum using projection fidelity and volumetric regularization. With the Gaussian centers, scales, and rotations fixed, GSC-Net predicts raw density transformations for each Gaussian conditioned on the common raw densities, spectral grid, and target channel. The resulting density coefficients for the target channel are combined with the fixed shared geometry and optimized using spectral projection fidelity, curve regularization, and volumetric regularization. Gaussian voxelization reconstructs the queried spectral channel, supporting reconstruction of observed spectral channels and queries of unobserved spectral channels.

## III. METHOD

## A. Motivation

Spectral CT channels depict the same object and therefore share a common spatial support, whereas the attenuation values vary as the spectral coordinate changes. Fig. 2 provides empirical evidence for this distinction between shared spatial structure and spectral attenuation variation.

As shown in Fig. 2(a), spectral slices preserve consistent boundaries, and the shared structural edge ratio remains high across different thresholds k for the minimum number of concurring channels. This consistency supports reusing Gaussian geometry across channels. Independent optimization introduces redundant parameters and may cause structural drift between channels. Meanwhile, Fig. 2(b) shows that the mean globally normalized attenuation of four fixed regions of interest (ROIs) follows systematic, ordered changes across spectral channels. This phenomenon supports modeling attenuation as a correlated function along the spectral axis rather than as unrelated values across channels.

Based on the observations, 4D-SG factorizes the spectral CT representation into shared Gaussian geometry and raw density transformation curves for each Gaussian. The shared Gaussian geometry captures the 3D structure common to all spectral channels, while the transformation curves describe spatially varying attenuation as it changes along the spectral axis.

## B. Theoretical Rationale

The structural consistency and ordered attenuation variation in Fig. 2 motivates separating shared spatial structure from spectral attenuation variation. With s denoting a generic spectral coordinate, a material basis model represents the attenuation field as $\begin{array} { r } { \mu ( \mathbf { x } , s ) = \sum _ { b = 1 } ^ { B } \rho _ { b } ( \mathbf { x } ) f _ { b } ( s ) } \end{array}$ , where $\rho _ { b } ( { \bf x } )$ and $f _ { b } ( s )$ are the spatial distribution and spectral response of the b-th basis component, respectively. The basis distributions are approximated using a shared set of Gaussian primitives:

$$
\rho _ { b } ( \mathbf { x } ) \approx \sum _ { i = 1 } ^ { K } \beta _ { i , b } G _ { i } ( \mathbf { x } ; \pmb { \mu } _ { i } , \pmb { \Sigma } _ { i } ) .\tag{1}
$$

The Gaussian kernel and covariance matrix are defined as

$$
G _ { i } ( \mathbf { x } ; { \boldsymbol { \mu } } _ { i } , { \boldsymbol { \Sigma } } _ { i } ) = \exp \left[ - { \frac { 1 } { 2 } } ( \mathbf { x } - { \boldsymbol { \mu } } _ { i } ) ^ { \top } { \boldsymbol { \Sigma } } _ { i } ^ { - 1 } ( \mathbf { x } - { \boldsymbol { \mu } } _ { i } ) \right] , \qquad { \boldsymbol { \Sigma } } _ { i } = \mathbf { R } ( \mathbf { q } _ { i } ) \mathrm { d i a g } ( \mathbf { s } _ { i } ^ { 2 } ) \mathbf { R } ( \mathbf { q } _ { i } ) ^ { \top } .\tag{2}
$$

Thus, $\{ ( \pmb { \mu } _ { i } , \mathbf { s } _ { i } , \mathbf { q } _ { i } ) \} _ { i = 1 } ^ { K }$ defines the shared Gaussian geometry. We write $G _ { i } ( \mathbf { x } ) = G _ { i } ( \mathbf { x } ; \pmb { \mu } _ { i } , \pmb { \Sigma } _ { i } )$ when the geometry is fixed. Substituting Eq. (1) into the material basis model gives

$$
\hat { \mu } ( \mathbf { x } , s ) = \sum _ { b = 1 } ^ { B } \sum _ { i = 1 } ^ { K } \beta _ { i , b } f _ { b } ( s ) G _ { i } ( \mathbf { x } ) .\tag{3}
$$

With $\begin{array} { r } { \alpha _ { i } ( s ) = \sum _ { b = 1 } ^ { B } \beta _ { i , b } f _ { b } ( s ) } \end{array}$ , the representation is written as follows:

$$
\hat { \mu } ( \mathbf { x } , s ) = \sum _ { i = 1 } ^ { K } \alpha _ { i } ( s ) G _ { i } ( \mathbf { x } ) .\tag{4}
$$

This representation separates shared Gaussian geometry from spectral density coefficients. To examine geometry sharing, we consider a general representation in which both density coefficients and Gaussian geometry vary with s:

$$
\widetilde { \mu } ( \mathbf { x } , s ) = \sum _ { i = 1 } ^ { K } \alpha _ { i } ( s ) G _ { i } \left( \mathbf { x } ; \pmb { \mu } _ { i } ( s ) , \pmb { \Sigma } _ { i } ( s ) \right) .\tag{5}
$$

Define $\mathbf { r } _ { i } ( s ) = \mathbf { x } - { \pmb \mu } _ { i } ( s ) , \mathbf { M } _ { i } ( s ) = \pmb { \Sigma } _ { i } ( s ) ^ { - 1 }$ , and $G _ { i } ( { \bf x } , s ) =$ $G _ { i } ( { \bf x } ; { \pmb \mu } _ { i } ( s ) , { \pmb \Sigma } _ { i } ( s ) )$ . Let $\begin{array} { r } { \phi _ { i } ( s ) = - \frac { 1 } { 2 } \mathbf r _ { i } ( s ) ^ { \top } \mathbf M _ { i } ( s ) \mathbf r _ { i } ( s ) } \end{array}$ . The argument s is omitted below when unambiguous. Applying the matrix product rule gives

$$
\phi _ { i } ^ { \prime } = - \frac { 1 } { 2 } \left[ \mathbf { r } _ { i } ^ { \prime \top } \mathbf { M } _ { i } \mathbf { r } _ { i } + \mathbf { r } _ { i } ^ { \top } \mathbf { M } _ { i } ^ { \prime } \mathbf { r } _ { i } + \mathbf { r } _ { i } ^ { \top } \mathbf { M } _ { i } \mathbf { r } _ { i } ^ { \prime } \right] .\tag{6}
$$

Using $\mathbf { r } _ { i } ^ { \prime } = - \pmb { \mu } _ { i } ^ { \prime }$ and $\mathbf { M } _ { i } ^ { \prime } = - \mathbf { M } _ { i } \pmb { \Sigma } _ { i } ^ { \prime } \mathbf { M } _ { i }$ , Eq. (6) becomes

$$
\phi _ { i } ^ { \prime } = \mu _ { i } ^ { \prime \top } \mathbf { M } _ { i } \mathbf { r } _ { i } + \frac { 1 } { 2 } \mathbf { r } _ { i } ^ { \top } \mathbf { M } _ { i } \Sigma _ { i } ^ { \prime } \mathbf { M } _ { i } \mathbf { r } _ { i } .\tag{7}
$$

![](images/608670b8883efa691376a44b53003143757c8db2a05eed98cde8fe69573d07e4.jpg)  
Fig. 4. Architecture of GSC-Net. The common raw densities and normalized spectral grid are embedded into spectral token sequences for each Gaussian and processed by the gated spectral SSM backbone. The density curve head combines token, low-rank basis, and residual responses to predict raw density transformations for each Gaussian. Observed spectral channels are queried directly from the predicted curves, whereas target spectral channels are queried from the predicted raw density curves before conversion to non-negative density coefficients.

Using the symmetry of $\mathbf { M } _ { i }$ and $G _ { i } ( { \bf x } , s ) = \exp ( \phi _ { i } ( s ) )$ , the chain rule gives

$$
\frac { \partial G _ { i } ( \mathbf { x } , s ) } { \partial s } = G _ { i } ( \mathbf { x } , s ) \pmb { \mu } _ { i } ^ { \prime \top } \mathbf { M } _ { i } \mathbf { r } _ { i } + \frac { 1 } { 2 } G _ { i } ( \mathbf { x } , s ) \mathbf { r } _ { i } ^ { \top } \mathbf { M } _ { i } \Sigma _ { i } ^ { \prime } \mathbf { M } _ { i } \mathbf { r } _ { i } .\tag{8}
$$

With $\mathbf { R } _ { i } ( s ) = \mathbf { R } ( \mathbf { q } _ { i } ( s ) )$ and $\mathbf { D } _ { i } ( s ) = \mathrm { d i a g } ( \mathbf { s } _ { i } ( s ) ^ { 2 } )$ . from $\Sigma _ { i } ( s ) = \mathbf { R } _ { i } ( s ) \mathbf { D } _ { i } ( s ) \mathbf { R } _ { i } ( s ) ^ { \top }$ , we obtain

$$
\begin{array} { r } { \pmb { \Sigma } _ { i } ^ { \prime } = \mathbf { R } _ { i } ^ { \prime } \mathbf { D } _ { i } \mathbf { R } _ { i } ^ { \top } + \mathbf { R } _ { i } \mathbf { D } _ { i } ^ { \prime } \mathbf { R } _ { i } ^ { \top } + \mathbf { R } _ { i } \mathbf { D } _ { i } \mathbf { R } _ { i } ^ { \prime \top } , } \end{array}\tag{9}
$$

where $\mathbf { D } _ { i } ^ { \prime } \ = \ 2 \mathrm { d i a g } ( \mathbf { s } _ { i } \odot \mathbf { s } _ { i } ^ { \prime } )$ . Differentiating Eq. (5) and substituting Eq. (8) yields

$$
\begin{array} { r l r } {  { \frac { \partial \widetilde { \mu } ( { \bf x } , s ) } { \partial s } = \sum _ { i = 1 } ^ { K } \alpha _ { i } ^ { \prime } G _ { i } ( { \bf x } , s ) + \sum _ { i = 1 } ^ { K } \alpha _ { i } G _ { i } ( { \bf x } , s ) { \pmb \mu } _ { i } ^ { \prime \top } { \bf M } _ { i } { \bf r } _ { i } } } \\ & { } & { + \frac { 1 } { 2 } \displaystyle \sum _ { i = 1 } ^ { K } \alpha _ { i } G _ { i } ( { \bf x } , s ) { \bf r } _ { i } ^ { \top } { \bf M } _ { i } \Sigma _ { i } ^ { \prime } { \bf M } _ { i } { \bf r } _ { i } . } \end{array}\tag{10}
$$

The three terms represent density variation, Gaussian center displacement, and covariance deformation, respectively.

In 4D-SG, the Gaussian centers, scales, and rotations are independent of s. The last two terms in Eq. (10) therefore vanish:

$$
\frac { \partial \hat { \mu } ( \mathbf { x } , s ) } { \partial s } = \sum _ { i = 1 } ^ { K } \alpha _ { i } ^ { \prime } ( s ) G _ { i } ( \mathbf { x } ) .\tag{11}
$$

Thus, spectral variation is represented only by Gaussian density changes over fixed spatial primitives. Integrating Eq. (11)

over $[ s _ { a } , s _ { b } ]$ gives

$$
\hat { \mu } ( \mathbf { x } , s _ { b } ) - \hat { \mu } ( \mathbf { x } , s _ { a } ) = \int _ { s _ { a } } ^ { s _ { b } } \frac { \partial \hat { \mu } ( \mathbf { x } , s ) } { \partial s } \mathrm { d } s = \sum _ { i = 1 } ^ { K } \left[ \alpha _ { i } ( s _ { b } ) - \alpha _ { i } ( s _ { a } ) \right] G _ { i } ( \mathbf { x } ) .\tag{12}
$$

Hence, any two spectral channels remain represented by the same Gaussian primitives.

For spectral weights $\lbrace w _ { m } \rbrace _ { m = 1 } ^ { M }$ , linear spectral aggregation is defined as

$$
\bar { \mu } ( { \bf x } ) = \sum _ { m = 1 } ^ { M } w _ { m } \hat { \mu } ( { \bf x } , s _ { m } ) .\tag{13}
$$

Substituting Eq. (4) and rearranging the summations gives

$$
\bar { \mu } ( { \bf x } ) = \sum _ { i = 1 } ^ { K } \left[ \sum _ { m = 1 } ^ { M } w _ { m } \alpha _ { i } ( s _ { m } ) \right] G _ { i } ( { \bf x } ) = \sum _ { i = 1 } ^ { K } \bar { \alpha } _ { i } G _ { i } ( { \bf x } ) ,\tag{14}
$$

where $\begin{array} { r } { \bar { \alpha } _ { i } = \sum _ { m = 1 } ^ { M } w _ { m } \alpha _ { i } ( s _ { m } ) } \end{array}$ . Thus, linear spectral aggregation changes only the Gaussian density coefficients and preserves the shared geometry. Overall, the shared geometry representation confines spectral variation to Gaussian density changes and preserves the same spatial primitives under both spectral differencing and linear aggregation.

## C. Shared-Structure 4D-SG Reconstruction

An overview of the framework is shown in Fig. 3. Let $\Theta = \{ \theta _ { n } \} _ { n = 1 } ^ { N _ { \theta } }$ denote the sparse projection views and ${ \boldsymbol { s } } =$ $\{ s _ { m } \} _ { m = 1 } ^ { M }$ denote the observed spectral channels. The input projections are denoted by $\mathcal { P } = \{ P _ { \theta } ^ { ( s _ { m } ) } \ | \ \theta \in \Theta , s _ { m } \in \mathcal { S } \}$ . As illustrated in Fig. 3, the projections at each view are aggregated into a structural projection over the full spectrum for establishing a common Gaussian representation. The learned Gaussian geometry is reused across spectral channels, while GSC-Net models the raw density curves for each Gaussian used for reconstruction of observed spectral channels and prediction of unobserved spectral channels.

The common Gaussian representation is defined as

$$
{ \mathcal G } ^ { ( c ) } = \left\{ \left( { \pmb { \mu } } _ { i } , { \bf s } _ { i } , { \bf q } _ { i } , u _ { i } ^ { ( c ) } \right) \right\} _ { i = 1 } ^ { K } ,\tag{15}
$$

where $\mu _ { i } , \mathbf { s } _ { i } , \mathbf { q } _ { i } .$ , and $u _ { i } ^ { ( c ) }$ denote the Gaussian center, scale, rotation quaternion, and common raw density, respectively. The centers, scales, and rotations constitute the shared Gaussian geometry, and the corresponding common density coefficient is $\boldsymbol \alpha _ { i } ^ { ( c ) } \stackrel { \cdot } { = } \boldsymbol \sigma _ { + } ( u _ { i } ^ { ( c ) } )$

For each projection view θ, the structural projection over the full spectrum is obtained as $\bar { P } _ { \theta } = \mathcal { A } _ { \mathrm { f s } } ( \{ P _ { \theta } ^ { ( s _ { m } ) } \} _ { s _ { m } \in \mathcal { S } } )$ . For photon-counting $\mathrm { C T } ,  { \mathcal { A } } _ { \mathrm { f s } } ( \cdot )$ returns the projection over the full spectrum, whereas for conventional spectral CT, it computes the average projection as $\begin{array} { r } { \bar { P } _ { \theta } = \frac { 1 } { M } \sum _ { m = 1 } ^ { \mathbf { \bar { M } } } P _ { \theta } ^ { ( s _ { m } ) } } \end{array}$ . Both forms are used to emphasize the structural information shared across spectral channels. The common Gaussian representation is then supervised through

$$
\operatorname* { m i n } _ { \mathcal { G } ^ { ( c ) } } \mathcal { L } _ { \mathrm { g e o m } } = \frac { 1 } { | \Theta | } \sum _ { \theta \in \Theta } \mathcal { L } _ { \mathrm { i m g } } \left( \mathcal { R } _ { \theta } ( \mathcal { G } ^ { ( c ) } ) , \bar { P } _ { \theta } \right) + \lambda _ { \mathrm { t v } } \mathcal { L } _ { \mathrm { t v } } ^ { ( c ) }\tag{16}
$$

Here, $\mathcal { R } _ { \theta } ( \cdot )$ and Q(·) denote the Gaussian rendering and voxelization operators, respectively, following $\mathbb { R } ^ { 2 } .$ -Gaussian [22], and ${ \mathcal { L } } _ { \mathrm { i m g } }$ consists of an $\ell _ { 1 }$ term and an optional SSIM term. Adaptive densification and pruning adjust the allocation of Gaussian primitives during this optimization.

Once the common Gaussian representation is established, the centers, scales, and rotations $\{ ( \pmb { \mu } _ { i } , \mathbf { s } _ { i } , \mathbf { q } _ { i } ) \} _ { i = 1 } ^ { K }$ are fixed to preserve Gaussian correspondence across spectral channels. The common raw densities and GSC-Net remain learnable. For the observed spectral grid, GSC-Net predicts the raw density transformation vector of each Gaussian:

$$
\Delta \mathbf { u } _ { i } = \left[ \Delta u _ { i } ^ { ( 1 ) } , \ldots , \Delta u _ { i } ^ { ( M ) } \right] .\tag{17}
$$

For spectral channel $s _ { m } ,$ its raw density is obtained as

$$
u _ { i } ^ { ( m ) } = u _ { i } ^ { ( c ) } + \Delta u _ { i } ^ { ( m ) } .\tag{18}
$$

The corresponding non-negative density coefficient is

$$
\alpha _ { i } ^ { ( m ) } = \sigma _ { + } \left( u _ { i } ^ { ( m ) } \right) .\tag{19}
$$

The entries of $\Delta { \mathbf { u } } _ { i }$ form the raw density curve of each Gaussian over the observed spectral grid. For spectral channel $s _ { m }$ , the corresponding Gaussian representation is $\hat { \mathcal { G } } ^ { ( s _ { m } ) } =$ $\{ ( \pmb { \mu } _ { i } , \mathbf { s } _ { i } , \mathbf { q } _ { i } , \pmb { \alpha } _ { i } ^ { ( \bar { m } ) } ) \} _ { i = 1 } ^ { K }$ . Its rendered projection and queried attenuation volume are denoted by $\hat { P } _ { \theta } ^ { ( s _ { m } ) } = \mathcal { R } _ { \theta } ( \hat { \mathcal { G } } ^ { ( s _ { m } ) } )$ and $\hat { V } ^ { ( s _ { m } ) } = \mathcal { Q } ( \hat { \mathcal { G } } ^ { ( s _ { m } ) } )$ , respectively.

Algorithm 1 Optimization and spectral querying of 4D-SG   
Require: P, Θ, observed grid $s ,$ target $( s ^ { \star } , \tau ^ { \star } )$   
Ensure: $\{ \hat { V } ^ { ( s _ { m } ) } \} _ { m = 1 } ^ { M }$ and $\hat { V } ^ { ( s ^ { \star } ) }$   
1: Construct $\{ \bar { P } _ { \theta } \}$ <sub>θ∈Θ</sub> and initialize $\mathcal G ^ { ( c ) }$ as described in   
Sec. III-C   
2: repeat   
Optimize $\mathcal G ^ { ( c ) }$ using Eq. (16), including scheduled   
densification and pruning   
4: until convergence   
5: Fix the shared Gaussian geometry   
6: repeat   
7: Sample $\theta \sim \Theta$ and predict the spectral curves using   
GSC-Net   
8: Render $\{ s _ { m } ~ \in ~ S _ { \theta } \}$ and optimize the common raw   
densities and GSC-Net using Eq. (20)   
9: until convergence   
10: Query $\{ \hat { V } ^ { ( s _ { m } ) } \} _ { m = 1 } ^ { M }$ using Eqs. (18)–(19)   
11: Query $\stackrel { \wedge } { V } ^ { ( s ^ { \star } ) }$ using Eq. (27)

At each iteration, a projection view is sampled and all available spectral channels at that view contribute to the projection supervision. The spectral optimization objective is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { s p e c } } = \mathbb { E } _ { \boldsymbol { \theta } \sim \boldsymbol { \Theta } } \left[ \displaystyle \frac { 1 } { | \boldsymbol { S } _ { \boldsymbol { \theta } } | } \sum _ { s _ { m } \in \boldsymbol { S } _ { \boldsymbol { \theta } } } \mathcal { L } _ { \mathrm { i m g } } \left( \hat { P } _ { \boldsymbol { \theta } } ^ { ( s _ { m } ) } , P _ { \boldsymbol { \theta } } ^ { ( s _ { m } ) } \right) \right] } \\ & { \quad \quad \quad + \lambda _ { \mathrm { c u r v e } } \mathcal { L } _ { \mathrm { c u r v e } } + \lambda _ { \mathrm { t v } } \mathcal { L } _ { \mathrm { t v } } . } \end{array}\tag{20}
$$

Here, $\scriptstyle { \mathcal { S } } _ { \theta }$ denotes the spectral channels available at view θ. The term $\mathcal { L } _ { \mathrm { c u r v e } }$ regularizes the raw density curves of individual Gaussians along the ordered spectral axis, while ${ \mathcal { L } } _ { \mathrm { t v } }$ regularizes the queried attenuation volumes. For an observed spectral channel, the corresponding raw density transformation is queried directly from the learned curve. For an unobserved target spectral coordinate $s ^ { \star }$ with normalized coordinate $\tau ^ { \star }$ $\Delta u _ { i } ( \tau ^ { \star } )$ is queried from the learned raw density curve and combined with the same shared Gaussian geometry according to Eq. (4). Algorithm 1 summarizes the optimization and spectral querying procedure.

## D. Gaussian-wise Spectral Density Curve Network

As illustrated in Fig. 4, GSC-Net maps the common raw density $u _ { i } ^ { ( c ) }$ and the normalized spectral grid values $\lbrace \tau _ { m } \rbrace _ { m = 1 } ^ { M }$ to the raw density transformations $\{ \Delta u _ { i } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ for each Gaussian. Here, $\tau _ { m }$ denotes the normalized spectral coordinate associated with $s _ { m }$ . The predicted transformations form a raw density curve over the ordered spectral axis, while the shared Gaussian geometry remains fixed.

Tokenization: The common raw density and each normalized spectral grid value are encoded separately and combined by broadcasted addition:

$$
\begin{array} { r } { \mathbf { x } _ { i , m } ^ { ( 0 ) } = \gamma _ { u } \left( u _ { i } ^ { ( c ) } \right) + \gamma _ { s } ( \tau _ { m } ) . } \end{array}\tag{21}
$$

The resulting tokens are arranged according to the ordered spectral grid:

$$
\mathbf { X } _ { i } ^ { ( 0 ) } = \left[ \mathbf { x } _ { i , 1 } ^ { ( 0 ) } , \ldots , \mathbf { x } _ { i , M } ^ { ( 0 ) } \right] \in \mathbb { R } ^ { M \times d } .\tag{22}
$$

TABLE I  
SUMMARY OF THE DATASETS, SPECTRAL CONFIGURATIONS, AND ACQUISITION GEOMETRIES. CHEST, HEAD, WALNUT, AND ORDINARY OBJECTS USE THE SAME SIMULATED CONE-BEAM SYSTEM MATRIX, WHILE FULL-VIEW FDK RECONSTRUCTIONS ARE USED AS REFERENCE VOLUMES FOR INSECT AND MOUSE CHEST.
<table><tr><td>Dataset</td><td>Data Source</td><td>Spectral Setting</td><td>SOD/SDD(mm)</td><td>Detector</td><td>Reference Volume</td></tr><tr><td>Chest [32]</td><td>Synthesized from a single-energy CT volume</td><td>3 monoenergetic channels: 40, 70, and 100 keV</td><td>1000/1500</td><td>512 × 512 pixels, 1-mm pixel size</td><td>Synthesized GT</td></tr><tr><td>Head [33]</td><td>Synthesized from a single-energy CT volume</td><td>3 monoenergetic channels: 40, 70, and 100 keV</td><td>1000/1500</td><td>512 × 512 pixels, 1-mm pixel size</td><td>Synthesized GT</td></tr><tr><td>Walnut [34]</td><td>Simulated from a spectral CT volume</td><td>8 monoenergetic channels: 10–80 keV at 10-keV intervals</td><td>1000/1500</td><td>512 × 512 pixels, 1-mm pixel size</td><td>Spectral CT volume</td></tr><tr><td></td><td>Ordinary Objects [35] Simulated from dual-energy CT volumes</td><td>5 mixed-energy channels from 100/140-kVp volumes, with λ = {0, 0.3, 0.5, 0.7, 1.0}</td><td>1000/1500</td><td>512 × 512 pixels, 1-mm pixel size</td><td>Spectral CT volume</td></tr><tr><td>Insect</td><td>Real multi-kVp cone-beam CT projections</td><td>4 tube-voltage settings: 40, 60, 80, and 100 kVp</td><td>310/510</td><td>2882 × 2340 pixels, 0.05-mm pixel size</td><td>Full-view FDK</td></tr><tr><td>Mouse Chest</td><td>Real photon-counting CT projections</td><td>4 energy windows: 20–90, 30–90, 45–90, and 76.28/361.1 60–90 keV</td><td></td><td>2062 × 2062 pixels, 0.1-mm pixel size</td><td>Full-view FDK</td></tr></table>

TABLE II

QUANTITATIVE COMPARISON ON ALL DATASETS USING 50 VIEWS. HIGHER PSNR AND SSIM VALUES ARE BETTER, WHEREAS LOWER LPIPS VALUES ARE BETTER. METRICS ON INSECT AND MOUSE CHEST ARE COMPUTED WITH RESPECT TO FULL-VIEW FDK REFERENCES.
<table><tr><td rowspan="2">Method</td><td colspan="3">Chest</td><td colspan="3">Head</td><td colspan="3">Walnut</td><td colspan="3">Ordinary Objects</td><td colspan="3">Insect</td><td colspan="3">Mouse Chest</td><td colspan="3">Average</td></tr><tr><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td><td>PSNR↑</td><td></td><td>SSIM↑ LPIPS↓</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td>PSNR↑ SSIM↑</td><td></td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td></tr><tr><td>FDK</td><td>27.00</td><td>0.546</td><td>0.402</td><td>29.41</td><td>0.702</td><td>0.346</td><td>24.77</td><td>0.540</td><td>0.393</td><td>29.85</td><td>0.625</td><td>0.326</td><td>27.09</td><td>0.526</td><td>0.440</td><td>22.61</td><td>0.516</td><td>0.432</td><td>26.79</td><td>0.576</td><td>0.390</td></tr><tr><td>CGLS</td><td>33.43</td><td>0.858</td><td>0.244</td><td>35.08</td><td>0.909</td><td>0.256</td><td>29.99</td><td>0.813</td><td>0.283</td><td>37.74</td><td>0.829</td><td>0.243</td><td>30.50</td><td>0.877</td><td>0.239</td><td>27.43</td><td>0.669</td><td>0.419</td><td>32.36</td><td>0.826</td><td>0.281</td></tr><tr><td>SART</td><td>32.85</td><td>0.886</td><td>0.228</td><td>36.73</td><td>0.925</td><td>0.256</td><td>32.76</td><td>0.897</td><td>0.244</td><td>39.26</td><td>0.943</td><td>0.225</td><td>31.96</td><td>0.900</td><td>0.236</td><td>28.78</td><td>0.705</td><td>0.361</td><td>33.72</td><td>0.876</td><td>0.258</td></tr><tr><td>NAF</td><td>35.56</td><td>0.896</td><td>0.192</td><td>37.35</td><td>0.942</td><td>0.215</td><td>33.68</td><td>0.939</td><td>0.199</td><td>40.12</td><td>0.968</td><td>0.193</td><td>32.74</td><td>0.914</td><td>0.194</td><td>28.80</td><td>0.715</td><td>0.349</td><td>34.71</td><td>0.896</td><td>0.224</td></tr><tr><td>R²-Gaussian</td><td>35.77</td><td>0.911</td><td>0.174</td><td>38.15</td><td>0.947</td><td>0.209</td><td>34.93</td><td>0.948</td><td>0.186</td><td>40.34</td><td>0.974</td><td>0.170</td><td>35.06</td><td>0.924</td><td>0.146</td><td>29.08</td><td>0.751</td><td>0.360</td><td>35.56</td><td>0.909</td><td>0.208</td></tr><tr><td>4D-SG (Ours)</td><td>37.57</td><td>0.923</td><td>0.162</td><td>39.54</td><td>0.959</td><td>0.203</td><td>35.65</td><td>0.955</td><td>0.177</td><td>41.04</td><td>0.974</td><td>0.179</td><td>36.54</td><td>0.936</td><td>0.112</td><td>29.35</td><td>0.736</td><td>0.331</td><td>36.61</td><td>0.914</td><td>0.194</td></tr></table>

The encoders $\gamma _ { u } ( \cdot )$ and $\gamma _ { s } ( \cdot )$ include the corresponding embedding, linear projection, and activation operations shown in Fig. 4. The resulting sequence preserves the order of the spectral grid values and is conditioned on the common raw density of each Gaussian.

Spectral SSM Backbone: The token sequence is processed along the spectral axis by a spectral state space model (SSM) backbone:

$$
\begin{array} { r l } & { \mathbf { H } _ { i } = \left[ \mathbf { h } _ { i , 1 } , . . . , \mathbf { h } _ { i , M } \right] } \\ & { \quad = \Phi _ { \mathrm { p o s t } } \left( \Phi _ { \mathrm { s s m } } \left( \Phi _ { \mathrm { c o n v } } \left( \mathbf { X } _ { i } ^ { ( 0 ) } \right) \right) \odot \Phi _ { \mathrm { g a t e } } \left( \mathbf { X } _ { i } ^ { ( 0 ) } \right) \right) } \end{array}\tag{23}
$$

Here, $\Phi _ { \mathrm { c o n v } } ( \cdot )$ captures local interactions between neighboring spectral tokens, $\Phi _ { \mathrm { s s m } } ( \cdot )$ models their ordered dependencies, and $\Phi _ { \mathrm { g a t e } } ( \cdot )$ modulates the resulting features. The output projection $\Phi _ { \mathrm { p o s t } } ( \cdot )$ produces the spectral features of the tokens. A global feature $\mathbf { p } _ { i } = \mathrm { P o o l } ( \mathbf { H } _ { i } )$ is further obtained for curve decoding.

Density Curve Head: The density curve head combines token, low-rank basis, and residual responses. The token response $\delta _ { i , m } ^ { \mathrm { t o k } }$ is decoded from $\mathbf { h } _ { i , m }$ , while the pooled feature $\mathbf { p } _ { i }$ provides the coefficients of the basis response and the residual response. The low-rank basis response is

$$
\delta _ { i , m } ^ { \mathrm { { b a s i s } } } = \sum _ { r = 1 } ^ { R } a _ { i , r } B _ { r , m } .\tag{24}
$$

The raw density transformation for each Gaussian is obtained as

$$
\Delta u _ { i } ^ { ( m ) } = \delta _ { i , m } ^ { \mathrm { t o k } } + \delta _ { i , m } ^ { \mathrm { b a s i s } } + \delta _ { i , m } ^ { \mathrm { r e s } } .\tag{25}
$$

Here, $\mathbf { B } \in \mathbb { R } ^ { R \times M }$ is a learnable spectral basis matrix, R denotes the basis rank, and $a _ { i , r }$ denotes the corresponding coefficient for each Gaussian. For each channel, the raw density and non-negative density coefficient are obtained as $u _ { i } ^ { ( m ) } \dot { = } u _ { i } ^ { ( c ) } + \Delta u _ { i } ^ { ( \check { m } ) }$ and $\alpha _ { i } ^ { ( m ) } \mathbf { \dot { \alpha } } = \sigma _ { + } ( u _ { i } ^ { ( m ) } )$ , respectively.

For an observed spectral channel $s _ { m }$ , the corresponding transformation $\Delta u _ { i } ^ { ( m ) }$ is queried directly from the predicted curve. For an unobserved target spectral channel, the raw density transformation is queried from the predicted curve as described in Sec. III-E, and the resulting raw density is mapped to the corresponding density coefficient through the softplus activation.

A first-order smoothness term constrains adjacent values of the raw density curve of each Gaussian:

$$
\mathcal { L } _ { \mathrm { c u r v e } } = \frac { 1 } { K ( M - 1 ) } \sum _ { i = 1 } ^ { K } \sum _ { m = 1 } ^ { M - 1 } \left| \Delta u _ { i } ^ { ( m + 1 ) } - \Delta u _ { i } ^ { ( m ) } \right| .\tag{26}
$$

This term suppresses abrupt variation between neighboring spectral grid values while preserving the spectral responses of individual Gaussians.

## E. Querying Unobserved Spectral Channels

Given an unobserved target spectral channel $s ^ { \star }$ , let $\tau ^ { \star }$ denote its normalized spectral grid value. For $\begin{array} { r } { \tau _ { k } \le \tau ^ { \star } \le \tau _ { k + 1 } , } \end{array}$ the raw density transformation of the i-th Gaussian is queried from the learned curve using piecewise-linear interpolation:

$$
\Delta u _ { i } ( \tau ^ { \star } ) = \frac { \tau _ { k + 1 } - \tau ^ { \star } } { \tau _ { k + 1 } - \tau _ { k } } \Delta u _ { i } ^ { ( k ) } + \frac { \tau ^ { \star } - \tau _ { k } } { \tau _ { k + 1 } - \tau _ { k } } \Delta u _ { i } ^ { ( k + 1 ) } .\tag{27}
$$

The queried transformation is combined with the common raw density and mapped to a non-negative density coefficient using the raw density parameterization in Eqs. (18)–(19). The resulting coefficient is combined with the shared Gaussian geometry to query the target attenuation volume using the same Gaussian voxelization procedure described in Sec. III-C. No additional optimization is required.

![](images/206adb5a6811dc31a0876f5ce46d2e6a4a36b087dad01d4d64ad5ab483bfffcc.jpg)  
Fig. 5. Representative qualitative comparison on 3D volume datasets with simulated projections using 50 views. The upper two groups show Walnut at 30 keV and 70 keV, respectively, and the lower two groups show Ordinary Objects with mixing coefficients of 0.0 and 1.0, respectively. For each spectral setting, the first row shows the reconstructed slice and the second row shows the corresponding zoomed ROI. The red boxes indicate the enlarged regions, and the red arrows highlight representative local structures. The PSNR and SSIM values annotated in each panel are computed on the corresponding entire 3D volume at the displayed spectral channel, rather than on the shown slice only. HU values are normalized to [0,1].

![](images/57c22fcdbcd53acfb0f2d93d77dcdc370485c3edfa3fc1899603e6adca3ceb37.jpg)  
Fig. 6. Representative qualitative comparison on synthesized multi-energy datasets using 50 views. The upper two groups show Chest at 40 keV and 100 keV, respectively, and the lower two groups show Head at 40 keV and 100 keV, respectively. For each spectral setting, the first row shows the reconstructed slice and the second row shows the corresponding zoomed ROI. The red boxes indicate the enlarged regions, and the red arrows highlight representative local structures. The PSNR and SSIM values annotated in each panel are computed on the corresponding entire 3D volume at the displayed spectral channel, rather than on the shown slice only. HU values are normalized to [0,1].

## IV. EXPERIMENTS

## A. Datasets and Experimental Settings

The proposed method is evaluated on six datasets spanning synthesized data, data with simulated projections, and real projection measurements, as summarized in Table I. Unless otherwise specified, all experiments are conducted using 50 projection views. For the datasets with simulated projections, the views are uniformly distributed over 360<sup>◦</sup>. For the real projection datasets, 50 views are uniformly selected from the original full-view measurements. All reconstructed volumes are resized or cropped to a spatial resolution of $2 5 6 ^ { 3 }$ for evaluation. The spectral channels of the Insect dataset are labeled in kVp because each tube voltage setting corresponds to a polychromatic rather than monoenergetic X-ray spectrum.

The proposed method is implemented in PyTorch, and all experiments are performed on an NVIDIA GeForce RTX 5070 Ti GPU. The source code is available at https://github.com/yqx7150/4D-SG. The optimization consists of two stages. We first optimize the shared Gaussian geometry for 30,000 iterations and then optimize the raw density curves for each Gaussian for 5,000 iterations. The Gaussian primitives are initialized from an FDK reconstruction, and the same hyperparameter settings are used for all datasets unless a dataset is specified otherwise.

## B. Comparative Results

The proposed method is compared with FDK, CGLS, SART, NAF [36], and R<sup>2</sup>-Gaussian [22]. Since NAF and R<sup>2</sup>-Gaussian do not use spectral reconstruction with shared structure, they are applied to each spectral channel independently. PSNR, SSIM, and LPIPS are averaged over all reconstructed slices and spectral channels after intensity normalization. For the qualitative figures, the annotated PSNR and SSIM values are computed on the corresponding entire 3D volume at the displayed spectral setting.

![](images/c7a7eebb2d022e26ad7e1113624245fda10f6c2fd5f7c1bb1b744aacd911135d.jpg)  
Fig. 7. Representative qualitative comparison on real projection datasets using 50 views. The upper two groups show Insect acquired at 40 kVp and 100 kVp, respectively, and the lower two groups show Mouse Chest at the 20–90 keV and 60–90 keV energy windows, respectively. For each spectral setting, the first row shows the reconstructed slice and the second row shows the corresponding zoomed ROI. The red boxes indicate the enlarged regions, and the red arrows highlight representative local structures. Since clean ground-truth volumes are unavailable for the real projection datasets, the reference volumes are full-view FDK reconstructions. The PSNR and SSIM values annotated in each panel are computed on the corresponding entire 3D volume at the displayed spectral channel, rather than on the shown slice only.

(a) Representative qualitative results  
![](images/8d1f8b166c0ea00855a13a9560d4183b6a2d33cdd73e8a3fb3ae7a500f9240ab.jpg)  
Fig. 8. Reconstruction and quantitative evaluation across spectral channels. (a) Reconstructed and reference Walnut slices. (b) PSNR, SSIM, and LPIPS by channel for each dataset.

Across Figs. 5–7, FDK exhibits severe sparse-view artifacts, while CGLS and SART reduce global artifacts but blur or distort fine structures. NAF and $\mathbb { R } ^ { 2 } .$ -Gaussian further improve structural continuity, whereas 4D-SG preserves sharper boundaries, local contrast, and structural consistency across channels on synthesized data and data with simulated projections. The same advantage remains visible on real projection data despite acquisition noise, system blur, and mismatch in spectral response.

Table II reports the results using 50 views. The proposed method achieves the best average PSNR, SSIM, and LPIPS. On synthesized and simulated projection datasets, the improvements indicate better fidelity to the available reference volumes. On Insect and Mouse Chest, metrics use full-view FDK references and should be interpreted as reference comparisons. Overall, the results consistently indicate improved sparseview spectral CT reconstruction across synthesized, simulated projection, and real projection scenarios.

![](images/d098f9560f59a2a0df7ec1aca38b26d214c06c027b96ce68b6e3b5924963da7d.jpg)  
Fig. 9. Ablation on Gaussian parameter freezing during raw density curve learning. Rows report PSNR, SSIM, and LPIPS. P/R/S/D denote position, rotation, scale, and raw density; L/F denote learnable/frozen.

## C. Ablation Studies

Channel reconstruction, Gaussian parameter freezing strategies, and the contribution of GSC-Net are examined. Scalability with respect to spectral channel count and interpolation at unobserved channels are also evaluated.

Spectral Channel Reconstruction and Metric Analysis by Dataset: Fig. 8 shows that the reconstructed slices preserve object boundaries and internal structures across spectral channels while reproducing spectral contrast changes. Stable PSNR, SSIM, and LPIPS across data types further indicate that the learned Gaussian raw density curves coherently represent attenuation variation along the spectral axis.

![](images/54d11ab130e74bfa359841638a6b9a7dfb027dc3bf61d376cffcacacb749aa0a.jpg)  
Fig. 10. Comparison between GSC-Net and a direct MLP alternative under different numbers of projection views. Higher values are better for PSNR and SSIM, whereas lower values are better for LPIPS.

![](images/237ad407de0368756bf5dc10eb5251d5ee55846f1057d209137fd673248cc4d2.jpg)

![](images/556917b7a5dd908a6cb4bdffc761b126f045866e05f2234aa29bc8e138719b98.jpg)  
Fig. 11. Comparison of training time and reconstruction quality for different spectral channel counts and configurations. The top plot reports how total training time changes as the number of spectral channels increases. It compares independent R<sup>2</sup>-Gaussian training against shared geometry variants using either MLP or GSC-Net for spectral adaptation. The bottom plots report PSNR, SSIM, and LPIPS. These results indicate that reusing shared Gaussian geometry improves training efficiency while maintaining reconstruction accuracy.

Effect of Freezing Different Gaussian Parameters: After learning the shared Gaussian geometry, we evaluate several parameter freezing strategies during the subsequent spectral density curve learning stage, as shown in Fig. 9.

Freezing the raw density leads to a substantial degradation in reconstruction quality. The best performance is obtained when the Gaussian positions, rotations, and scales are fixed, while the raw density remains learnable, achieving 36.61 dB PSNR, 0.914 SSIM, and 0.194 LPIPS. Allowing the geometric parameters to remain learnable does not provide any further improvement, which supports the use of fixed shared Gaussian geometry during spectral density curve learning.

Effect of GSC-Net and Number of Views: We evaluate the effect of GSC-Net by replacing it with a direct MLP while keeping the remaining pipeline unchanged. We also compare the two variants under different numbers of projection views. The results are shown in Fig. 10.

![](images/3729561e3d6887b608d0aa5984bb6f71578506d125f17a9c9698cb7de184f737.jpg)  
(a) Projection Residual Comparison

![](images/5319065f49149126fb6bd99960fcfad487132dd3b2d4516b7bce3c8309ca852a.jpg)  
(b) Boxplots of Projection Metrics  
Fig. 12. Prediction of projections for an unobserved spectral channel across six datasets. (a) Predicted projections, corresponding reference projections, and residual maps. (b) Projection-level metrics evaluated across multiple views.

GSC-Net outperforms the direct MLP at all tested view counts. At 50 views, it improves PSNR from 36.06 to 36.61 dB and SSIM from 0.905 to 0.914, while reducing LPIPS from 0.195 to 0.194. At 20 views, PSNR increases from 29.58 to 31.01 dB and LPIPS decreases from 0.303 to 0.292. Thus, modeling Gaussian raw density curves is more effective than direct density regression, particularly when angular sampling is severely limited.

Training Efficiency versus Number of Spectral Channels: We compare independent channel training with $\mathrm { { R ^ { 2 } } \mathrm { { - } } }$ Gaussian against MLP and GSC-Net variants with shared Gaussian geometry in training time and quality (Fig. 11).

Independent $\mathrm { R ^ { 2 } \mathrm { - G a u s s i a n } }$ training grows rapidly with spectral channel count, whereas shared geometry variants scale more slowly. For eight channels, $\mathbf { R } ^ { 2 } \cdot$ -Gaussian with 30K iterations requires 107 min 47 s, compared with 49 min 10 s for 30K geometry + GSC-Net. This variant gives the best shared geometry result (36.61 dB PSNR, 0.914 SSIM, and 0.194 LPIPS), reducing redundant channel optimization without sacrificing accuracy.

Unobserved Spectral Channel Projection Prediction: We evaluate target spectral channel prediction by withholding one intermediate spectral channel during spectral density curve learning. The remaining channels are used as observed channels, and the withheld projection is generated by querying the learned raw density transformation curves for each Gaussian.

Across the six datasets shown in Fig. 12, predicted projections closely match the ground-truth or reference projections, with residuals mainly near high-contrast boundaries and fine structures. Stable PSNR, SSIM, and LPIPS at the projection level across views confirm prediction of an unobserved spectral channel without direct supervision from that channel.

## V. CONCLUSION

We proposed 4D-SG for sparse-view spectral CT reconstruction, which learned shared Gaussian geometry to represent common 3D structure and employed GSC-Net to predict raw density transformation curves for each Gaussian to model attenuation variation across spectral channels. Experiments on synthesized datasets, datasets with simulated projections, and real projection datasets demonstrated improved reconstruction quality, spectral consistency across channels, and computational efficiency over conventional baselines based on voxel grids and Gaussian representations. Beyond sparse-view reconstruction, the learned shared-structure 4D spectral Gaussian representation may also provide a useful basis for downstream spectral CT applications, including material decomposition, virtual monoenergetic imaging, and other spectral reconstruction and analysis tasks, which will be explored in future work.

## REFERENCES

[1] C. H. McCollough, S. Leng, L. Yu, and J. G. Fletcher, “Dual-and multienergy CT: principles, technical approaches, and clinical applications,” Radiology, vol. 276, no. 3, pp. 637–653, 2015.

[2] M. J. Willemink, M. Persson, A. Pourmorteza, N. J. Pelc, and D. Fleischmann, “Photon-counting CT: technical principles and clinical prospects,” Radiology, vol. 289, no. 2, pp. 293–312, 2018.

[3] J. Greffier, N. Villani, D. Defez, D. Dabli, and S. Si-Mohamed, “Spectral CT imaging: technical principles of dual-energy CT and multi-energy photon-counting CT,” Diagnostic and Interventional Imaging, vol. 104, no. 4, pp. 167–177, 2023.

[4] D. S. Rigie and P. J. La Riviere, “Joint reconstruction of multi-channel, spectral CT data via constrained total nuclear variation minimization,” Physics in Medicine & Biology, vol. 60, no. 5, pp. 1741–1762, 2015.

[5] Y. Zhang, X. Mou, G. Wang, and H. Yu, “Tensor-based dictionary learning for spectral CT reconstruction,” IEEE Transactions on Medical Imaging, vol. 36, no. 1, pp. 142–154, 2016.

[6] Y. Liu, X. Zhou, C. Wei, and Q. Xu, “Sparse-view spectral CT reconstruction and material decomposition based on multi-channel SGM,” IEEE Transactions on Medical Imaging, vol. 43, no. 10, pp. 3425–3435, 2024.

[7] Y. Long and J. A. Fessler, “Multi-material decomposition using statistical image reconstruction for spectral CT,” IEEE Transactions on Medical Imaging, vol. 33, no. 8, pp. 1614–1626, 2014.

[8] T. G. Schmidt, R. F. Barber, and E. Y. Sidky, “A spectral CT method to directly estimate basis material maps from experimental photon-counting data,” IEEE Transactions on Medical Imaging, vol. 36, no. 9, pp. 1808– 1819, 2017.

[9] K. Mechlem, S. Ehn, T. Sellerer, E. Braig, D. Munzel, F. Pfeiffer, and¨ P. B. Noel, “Joint statistical iterative material image reconstruction for¨ spectral computed tomography using a semi-empirical forward model,” IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 68–80, 2017.

[10] L. Shen, Y. Xing, and L. Zhang, “Joint reconstruction and spectrum refinement for photon-counting-detector spectral CT,” IEEE Transactions on Medical Imaging, vol. 42, no. 9, pp. 2653–2665, 2023.

[11] W. Wu, D. Hu, C. Niu, L. Vanden Broeke, A. P. Butler, P. Cao, J. Atlas, A. Chernoglazov, V. Vardhanabhuti, and G. Wang, “Deep learning based spectral CT imaging,” Neural Networks, vol. 144, pp. 342–358, 2021.

[12] X. Chen, W. Xia, Z. Yang, H. Chen, Y. Liu, J. Zhou, Z. Wang, Y. Chen, B. Wen, and Y. Zhang, “SOUL-net: A sparse and low-rank unrolling network for spectral CT image reconstruction,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 12, pp. 18 620– 18 634, 2023.

[13] Y. Shi, Y. Gao, Q. Xu, Y. Li, X. Mou, and Z. Liang, “Learned tensor neural network texture prior for photon-counting CT reconstruction,” IEEE Transactions on Medical Imaging, vol. 43, no. 11, pp. 3830–3842, 2024.

[14] J. Guo, Y. Wang, S. Wang, Z. Zheng, L. Li, A. Cai, and B. Yan, “Sparseview spectral CT reconstruction via a coupled subspace representation and score-based generative model,” Quantitative Imaging in Medicine and Surgery, vol. 15, no. 6, pp. 5474–5495, 2025.

[15] X. Zhang, L. Li, S. Wang, N. Liang, A. Cai, and B. Yan, “One-step inverse generation network for sparse-view dual-energy CT reconstruction and material imaging,” Physics in Medicine & Biology, vol. 69, no. 14, p. 145012, 2024.

[16] B. Song, L. Shen, and L. Xing, “PINER: Prior-informed implicit neural representation learning for test-time adaptation in sparse-view CT reconstruction,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 1928–1938.

[17] Y. Lin, Z. Luo, W. Zhao, and X. Li, “Learning deep intensity field for extremely sparse-view CBCT reconstruction,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2023, pp. 13–23.

[18] Y. Lin, J. Yang, H. Wang, X. Ding, W. Zhao, and X. Li, “C<sup>2</sup>RV: Crossregional and cross-view learning for sparse-view CBCT reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 11 205–11 214.

[19] Y. Cai, J. Wang, A. Yuille, Z. Zhou, and A. Wang, “Structure-aware sparse-view X-ray 3D reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 11 174–11 183.

[20] B. Kerbl, G. Kopanas, T. Leimkuhler, G. Drettakis¨ et al., “3D Gaussian splatting for real-time radiance field rendering,” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.

[21] Y. Cai, Y. Liang, J. Wang, A. Wang, Y. Zhang, X. Yang, Z. Zhou, and A. Yuille, “Radiative Gaussian splatting for efficient X-ray novel view synthesis,” in European Conference on Computer Vision. Springer, 2024, pp. 283–299.

[22] R. Zha, T. J. Lin, Y. Cai, J. Cao, Y. Zhang, and H. Li, “R<sup>2</sup>-Gaussian: Rectifying radiative gaussian splatting for tomographic reconstruction,” arXiv preprint arXiv:2405.20693, 2024.

[23] Y. Lin, H. Wang, J. Chen, and X. Li, “Learning 3D Gaussians for extremely sparse-view cone-beam CT reconstruction,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 425–435.

[24] Y. Li, X. Fu, H. Li, S. Zhao, R. Jin, and S. K. Zhou, “3DGR-CT: Sparseview CT reconstruction with a 3D gaussian representation,” Medical Image Analysis, vol. 103, p. 103585, 2025.

[25] A. S. Wang and N. J. Pelc, “Spectral photon counting CT: Imaging algorithms and performance assessment,” IEEE Transactions on Radiation and Plasma Medical Sciences, vol. 5, no. 4, pp. 453–464, 2020.

[26] J. Liu, H. Ding, S. Molloi, X. Zhang, and H. Gao, “TICMR: Total image constrained material reconstruction via nonlocal total variation regularization for spectral CT,” IEEE Transactions on Medical Imaging, vol. 35, no. 12, pp. 2578–2586, 2016.

[27] A. Bousse, V. S. S. Kandarpa, S. Rit, A. Perelli, M. Li, G. Wang, J. Zhou, and G. Wang, “Systematic review on learning-based spectral CT,” IEEE Transactions on Radiation and Plasma Medical Sciences, vol. 8, no. 2, pp. 113–137, 2023.

[28] X. Hu and X. Jia, “Spectral CT image reconstruction using a constrained optimization approach—an algorithm for AAPM 2022 spectral CT grand challenge and beyond,” Medical Physics, vol. 51, no. 5, pp. 3376–3390, 2024.

[29] E. Y. Sidky and X. Pan, “Report on the aapm deep-learning spectral CT grand challenge,” Medical Physics, vol. 51, no. 2, pp. 772–785, 2024.

[30] T. Ge, R. Liao, M. Medrano, D. G. Politte, J. F. Williamson, and J. A. O’Sullivan, “MB-DECTNet: a model-based unrolling network for accurate 3D dual-energy CT reconstruction from clinically acquired helical scans,” Physics in Medicine & Biology, vol. 68, no. 24, p. 245009, 2023.

[31] Y. Liu, X. Zhou, C. Wang, C. Wei, and Q. Xu, “Low-dose spectral CT reconstruction based on structural prior network,” Medical Physics, vol. 52, no. 10, p. e70061, 2025.

[32] American Association of Physicists in Medicine, “Low dose CT grand challenge,” [Online], 2017, accessed: Apr. 6, 2017. [Online]. Available: http://www.aapm.org/GrandChallenge/LowDoseCT/

[33] P. Klacansky, “Scientific Visualization Datasets,” [Online], 2022. [Online]. Available: https://klacansky.com/open-scivis-datasets/

[34] E. Zhou, W. Li, W. Xu, K. Wan, Y. Lu, S. Chen, G. Zheng, T. Xie, and Q. Liu, “A cone-beam photon-counting CT dataset for spectral image reconstruction and deep learning,” Scientific Data, vol. 12, no. 1, p. 1955, 2025.

[35] D. Volgyes, “Dual energy CT scan of ordinary objects,” jun 2018.¨ [Online]. Available: https://doi.org/10.5281/zenodo.1253035

[36] R. Zha, Y. Zhang, and H. Li, “NAF: Neural attenuation fields for sparseview CBCT reconstruction,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2022, pp. 442–452.