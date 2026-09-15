# TRACE: Two-Stage Detector-Response Estimation With Angular Cosine Expansion for Ring Artifact Correction in Photon-Counting CT

Jigang Duan, Heran Wang, Ligen Shi, Zheng Sun, Ping Yang, and Xing Zhao

Abstract—Detector response nonuniformity introduces systematic projection errors and ring artifacts in photon-counting detector computed tomography (PCD-CT). In measured PCD-CT data, residual stripe amplitudes vary slowly with projection angle, which fixed-bias models cannot adequately capture. We propose TRACE, a two-stage unsupervised sinogram decomposition method for estimating and correcting these response-related errors. TRACE represents stripes as a fixed bias plus low-order discrete cosine transform (DCT) components, using a small number of coefficients to describe angular variations at each detector element. A learnable analysis–synthesis architecture represents the ideal projections, while two-stage optimization separates them from fixed and then dynamic stripes. An angular-gradient soft orthogonality constraint suppresses correlated variations within the shared DCT gradient subspace, reducing the leakage of object structures into the artifact estimate. All parameters are optimized directly on the measured sinogram without paired training data. Experiments on measured QRM mouse phantom and porcine trotter data show that TRACE suppresses ring artifacts and improves image uniformity while preserving edge sharpness, softtissue texture, and trabecular detail.

Index Terms—Photon-counting CT, detector response nonuniformity, ring artifact correction, sinogram decomposition, discrete cosine transform, unsupervised learning

## I. INTRODUCTION

Response nonuniformity among detector elements introduces systematic errors into computed tomography (CT) measurements. These errors appear as stripes extending along the angular direction in sinograms and form rings around the rotation center after reconstruction [1], [2]. Unlike conventional energy-integrating detector CT, photon-counting detector CT (PCD-CT) registers individual X-ray-induced electrical pulses and discriminates their amplitudes using energy thresholds. Differences in counting response between detector elements can depend on energy threshold and incident flux, so fixed calibration parameters may not fully compensate for response nonuniformity throughout a scan. Residual response errors can produce stripes that vary with energy window and projection angle [3]–[5]. These artifacts impair image uniformity, distort reconstructed attenuation values, and obscure low-contrast structures. Correcting residual response errors while preserving object information is therefore important for reliable PCD-CT measurements.

## A. Existing Methods for Ring Artifact Correction

Existing methods include detector calibration, sinogramdomain preprocessing, CT image postprocessing, dual-domain iterative methods, and data-driven methods.

Detector calibration compensates for response differences between detector elements through dynamic flat-field correction based on eigen flat fields [6] or nonlinear response fitting based on mean projections [7]. For PCD-CT, phantom-based calibration corrects response differences and count-rate nonlinearity [4], while physical models estimate energy threshold deviations [8]. Redundant ray sampling also enables gain selfcalibration from scan data [5]. However, fixed parameters may fail to track changing system and scan conditions, whereas dynamic calibration often requires additional data, redundant sampling, or accurate registration.

Sinogram-domain preprocessing suppresses stripes before reconstruction through wavelet–Fourier filtering [2], extensions using projection grouping and weighted filtering [9], or response equalization, detection, and interpolation [10]. For PCD-CT, smoothed mean projections can provide detector response correction coefficients [11]. Other approaches use multiscale BM3D for correlated stripe noise [12], lowrank Tucker decomposition with spatial–sequential total variation [13], adaptive normalization [14], or adaptive frequencydomain patch filtering combined with spatial-domain stripe filtering [15]. However, directional or frequency overlap between stripes and object projections can lead to residual artifacts or loss of useful projection information.

CT image postprocessing typically converts rings into approximate stripes in polar coordinates. Representative methods use sliding-window artifact estimation [1], unidirectional variational decomposition with sparsity constraints [16], or relative total variation (RTV) templates for iterative stripe extraction [17]. Structure preservation is addressed through sparse-domain regularized decomposition with guided filtering [18], structure-aware guided filtering [19], or directional gradient-domain optimization [20]. Superpixel segmentation with adaptive RTV further treats strong and weak rings separately [21]. These methods require no raw projections, but polar-coordinate interpolation may blur details, and circular object structures may be suppressed. They also lack direct constraints from measured projections.

Dual-domain iterative methods jointly optimize image reconstruction and projection correction. Early studies introduced explicit ring variables into compressed sensing reconstruction [22] or combined image-domain ring total variation with detector correction vectors [23]. Extensions incorporate unidirectional total variation and group sparsity into spectral PCD-CT material decomposition [24], combine stripe and image sparsity priors [25], or couple polynomial detector response models with polar-domain smoothing [26]. Other formulations combine group sparsity and low-rank priors on three-dimensional projection stripes with image regularization [27], or integrate generative and ring artifact model priors for sparse-view reconstruction [28]. Repeated projection, backprojection, and subproblem optimization increase computational cost, while performance depends on the system model and regularization. Separate component priors may also permit object structures to enter artifact estimates.

Data-driven methods learn correction mappings or artifact representations using neural networks. Early approaches fused precorrected sinograms with reconstructed images [29] or combined random detector translation with neural artifact suppression and denoising [30]. Sinogram networks estimate stripes by combining wavelet decomposition with residual learning [31] or global–local feature interaction [32]. Imagedomain methods include a polar-domain Transformer with a vertical-gradient loss [33], Cartesian–polar Mamba networks [34], and dual-branch networks exploiting adjacent slices and enhancing central image regions [35]. Multistage networks also process projections, sinograms, and reconstructed images sequentially [36]. Syn2Real uses realistic image-domain ring synthesis to reduce the synthetic–real gap [37], while conditional flow matching refines reconstructions after random neighborhood sampling decouples projection stripes [38].

Unsupervised alternatives use dual contrastive learning for unpaired polar-domain correction [39] or implicit neural representations for single-sinogram decomposition into ideal projections and stripes [40]. Supervised methods depend on training data size, quality, and representativeness and remain sensitive to synthetic–real distribution differences. Unsupervised methods reduce reliance on paired data but still depend on network representations, losses, and priors.

Overall, two challenges remain difficult to address simultaneously: modeling residual detector response variations along the scan sequence with controlled degrees of freedom, and preventing object structures from entering artifact estimates when stripes and object projections exhibit similar variation patterns.

## B. Motivation

To address these challenges, we propose TRACE, a twostage unsupervised method that estimates residual detector response errors through sinogram decomposition. Fig. 1 summarizes its three key ideas: an adaptive representation of ideal projections, compact angular response modeling, and reduced structure leakage.

To relate residual response errors to projection stripes, consider an equivalent monoenergetic model for a fixed energy channel. Omitting the angle index and random counting fluctuations, the transmitted count at detector element d is modeled as

![](images/d2b67a74fa40a707c0922e1f054a94e8922653c5542590a7900d27df8b8228c6.jpg)  
Fig. 1. Motivation and key ideas of TRACE. (A) A learnable analysis– synthesis representation combines a low-frequency backbone with recovered details to estimate the ideal sinogram (IS). (B) Two-stage optimization extends fixed stripe biases with low-order discrete cosine transform (DCT) components to capture smooth angular variations. (C) Soft orthogonality between the projected angular gradients of IS and dynamic stripe artifacts (SA) reduces structure leakage. The cyan and yellow vectors represent these gradients, respectively, in the subspace spanned by the angular gradients of the loworder DCT basis functions.

$$
I _ { d } = C _ { d } I _ { 0 , d } \exp \left( - \int _ { L _ { d } } \mu ( x ) \mathrm { d } l \right) ,\tag{1}
$$

where $I _ { 0 , d }$ is the incident reference count and $C _ { d } ~ > ~ 0$ is the effective relative response coefficient after conventional calibration, with an ideal value of 1. The linear attenuation coefficient is denoted by $\mu ( x )$ , and $L _ { d }$ is the corresponding ray path. This effective model links residual response errors to additive biases after logarithmic conversion:

$$
\int _ { L _ { d } } \mu ( x ) \mathrm { d } l = - \log \left( \frac { I _ { d } } { I _ { 0 , d } } \right) + \log \left( C _ { d } \right) .\tag{2}
$$

Thus, the measured logarithmic projection is the sum of the ideal line integral and the stripe component − log(C<sub>d</sub>). Across all views and detector elements, these components form the ideal sinogram (IS) and stripe artifacts (SA), respectively.

Along the detector direction, adjacent object projections are usually locally continuous, giving the IS a predominantly low-frequency structure. Narrow stripes exhibit more rapid variations between detector elements. As illustrated in Fig. 1(A), predefined wavelet bases can separate frequency components [2], but their fixed supports and frequency responses limit adaptation to different object structures and stripe widths. A learnable analysis–synthesis representation can adapt to the measured sinogram while a detail branch recovers structural information.

Along the angular direction, the residual stripes considered here contain fixed biases and slowly varying components. Fig. 1(B) illustrates the modeling trade-off: fixed biases cannot capture angular variations, whereas independent per-view estimates introduce many degrees of freedom and may absorb object structures. Moreover, IS and dynamic SA can share low angular frequencies, as shown in Fig. 1(C). Separate smoothness or sparsity constraints therefore do not fully resolve their ambiguity.

Based on these observations, TRACE first estimates fixed stripes and then introduces dynamic components. In Stage I, the learnable analysis–synthesis architecture represents IS, while each detector element’s logarithmic response error is approximated by a fixed bias:

$$
- \log { ( C _ { d } ) } \approx b _ { d } ,\tag{3}
$$

where $b _ { d }$ is the fixed stripe coefficient. Fitting the sum of IS and fixed SA to the measured sinogram limits the initial degrees of freedom of the artifact branch and provides a stable starting point for dynamic estimation.

Following Fig. 1(B), Stage II extends the response model using low-order discrete cosine transform (DCT) components:

$$
- \log \left[ C _ { d } ( \boldsymbol { v } ) \right] \approx b _ { d } + \sum _ { k = 1 } ^ { K } a _ { d , k } \phi _ { k } ( \boldsymbol { v } ) ,\tag{4}
$$

where v is the view index, $\phi _ { k } ( v )$ is the kth non-DC DCT basis function, and ${ a } _ { d , k }$ is its coefficient. The number of basis functions, K, is much smaller than the number of views. Excluding the DC basis avoids redundancy with the fixed bias, yielding a compact representation of smooth angular response variations.

Finally, Fig. 1(C) illustrates the angular-gradient soft orthogonality constraint. The angular gradients of IS and dynamic SA are projected onto the subspace spanned by the gradients of the retained DCT basis functions. Penalizing their normalized correlation reduces the leakage of object structures into dynamic SA. Angular curvature regularization controls oscillations, while data consistency and sorting-based detectordirection smoothness stabilize the decomposition.

## C. Our Contributions

The main contributions are as follows:

• We develop a two-stage unsupervised sinogram decomposition framework that combines a learnable analysis– synthesis representation with progressive fixed and dynamic stripe estimation. All parameters are fitted to the measured sinogram without paired data or additional flatfield acquisitions.

• A fixed–dynamic response-error model uses a truncated non-DC DCT basis to describe smooth angular stripe variations with few coefficients per detector element, extending fixed-bias correction while controlling model complexity.

• An angular-gradient soft orthogonality constraint reduces ambiguity between IS and dynamic SA. Together with angular curvature regularization, data consistency, and detector-direction smoothness, it promotes artifact separation while preserving object structures.

## II. METHOD

As shown in Fig. 2, TRACE separates IS and SA by exploiting their variation patterns along the detector and angular directions. A learnable IS branch represents object projections, while two-stage optimization progressively estimates fixed and dynamic stripes. All parameters are optimized directly on the measured sinogram.

## A. Sinogram Decomposition and Learnable IS Representation

As shown in Fig. 2(A), let the measured logarithmic sinogram be $\pmb { Y } \in \mathbb { R } ^ { V \times D }$ , where V and D denote the numbers of projection views and detector elements, respectively. We use the additive decomposition

$$
\pmb { Y } = \pmb { P } + \pmb { S } ,\tag{5}
$$

where P and S denote IS and SA, respectively.

Fig. 2(B) details the IS branch. Since IS typically exhibits local continuity and predominantly low-frequency content along the detector direction, a learnable convolutional analysis–synthesis architecture first extracts its low-frequency backbone:

$$
P _ { \mathrm { l o w } } = \mathcal { G } _ { \pmb { \theta } _ { s } } \left[ \mathcal { H } _ { \pmb { \theta } _ { a } } ( \pmb { Y } ) \right] ,\tag{6}
$$

where $\mathcal { H } _ { \theta _ { a } }$ comprises analysis convolutions and downsampling, and $\mathcal { G } _ { \pmb { \theta } _ { i } }$ comprises upsampling and synthesis convolutions. Inspired by wavelet analysis and synthesis, this architecture uses learnable convolution kernels to adapt the representation to the measured sinogram.

To recover structural details lost during analysis and synthesis, we compute the residual

$$
\pmb { R } = \pmb { Y } - \pmb { P } _ { \mathrm { l o w } } .\tag{7}
$$

As shown in the lower branch of Fig. 2(B), the residual passes through a learnable vertical band-stop module $B _ { \theta _ { \tau } }$ to suppress stripes extending along the angular direction. The result is added to the low-frequency backbone to obtain the IS estimate:

$$
\widehat { P } _ { \Theta } = P _ { \mathrm { l o w } } + \mathcal { B } _ { \theta _ { r } } ( R ) ,\tag{8}
$$

where $\boldsymbol { \Theta } = \{ \pmb { \theta } _ { a } , \pmb { \theta } _ { s } , \pmb { \theta } _ { r } \}$ contains all learnable parameters of the IS branch.

## B. Fixed–Dynamic Modeling of SA

Residual response errors can contain both fixed biases and smooth angular variations. Following the stripe decomposition in Fig. 2(A), we represent SA using fixed column biases and low-order DCT components:

$$
\widehat { S } ( b , A ) = \underbrace { \mathbf { 1 } _ { V } { b } ^ { T } } _ { \widehat { S } _ { \mathrm { f i x } } } + \underbrace { \Phi _ { K } A ^ { T } } _ { \widehat { S } _ { \mathrm { d y n } } } ,\tag{9}
$$

where $\mathbf { 1 } _ { V }$ is an all-ones vector, $\pmb { b } \in \mathbb { R } ^ { D }$ contains the fixed biases, $\pmb { A } \in \mathbb { R } ^ { D \times K }$ is the dynamic coefficient matrix, and $\Phi _ { K } ~ \in ~ \mathbb { R } ^ { V \times K }$ contains the first K non-DC DCT basis functions, with entries

$$
[ \Phi _ { K } ] _ { v , k } = \alpha _ { k } \cos \left[ \frac { \pi ( 2 v + 1 ) k } { 2 V } \right] ,\tag{10}
$$

![](images/c971d508f934a89baa9322e78b1bfefd5aa5e8c2c8c59f32bb6d2c9c13eff7d9.jpg)  
Fig. 2. Overview of the proposed two-stage unsupervised sinogram decomposition method. (A) IS–SA decomposition of the measured sinogram and the fixed–dynamic stripe model. (B) IS representation using a learnable analysis–synthesis architecture and a residual branch. (C) Two-stage optimization: fixed stripes are first separated, followed by the introduction of dynamic DCT components and angular constraints to jointly refine IS and SA. (D) Final IS and SA estimates and the corrected image reconstructed from IS using FBP.

where $v = 0 , \ldots , V - 1 , k = 1 , \ldots , K , \alpha _ { k }$ is a normalization coefficient, and $K \ll V$ . Here, $[ \Phi _ { K } ] _ { v , k } = \phi _ { k } ( v )$ in (4), and $d = 1 , \dotsc , D$ indexes detector elements. Excluding the DC basis yields $\pmb { \Phi } _ { K } ^ { T } \mathbf { 1 } _ { V } = \mathbf { 0 }$ , so the dynamic component has zero angular mean, avoiding redundancy with the fixed biases.

In Stage II of Fig. 2(C), replicating b across views forms fixed SA, while $\Phi _ { K } A ^ { T }$ generates dynamic SA. Their sum describes the response-related projection bias at each detector element.

## C. Regularization and Two-Stage Optimization

IS–SA separation is guided by data consistency and directional constraints. First, data consistency penalizes the difference between their sum and the measured sinogram:

$$
\mathcal { L } _ { \mathrm { d c } } ( P , S ) = \frac { 1 } { V D } \left\| Y - P - S \right\| _ { 1 } .\tag{11}
$$

Here, $\| \cdot \| _ { 1 }$ denotes the entrywise $L _ { 1 }$ norm.

To reduce the influence of angle-dependent shifts in projection structures when comparing adjacent detector elements, we first sort the projection values in each IS column along the angular direction:

$$
{ \widetilde { \pmb { P } } } = \mathrm { s o r t } _ { v } ( { \pmb { P } } ) .\tag{12}
$$

Differences between adjacent sorted columns are then penalized using a sorting-based detector-direction smoothness constraint:

$$
\mathcal { R } _ { \mathrm { d e t } } ( \boldsymbol { P } ) = \frac { 1 } { V ( D - 1 ) } \left. \widetilde { \boldsymbol { P } } \boldsymbol { D } _ { d } ^ { T } \right. _ { F } ^ { 2 } ,\tag{13}
$$

where $\pmb { { D } } _ { d } \in \mathbb { R } ^ { ( D - 1 ) \times D }$ is the first-order difference matrix along the detector direction. This term promotes continuity between the projection-value distributions of adjacent detector elements.

The low-order DCT representation restricts the angular frequency range of dynamic SA. We further introduce angular curvature regularization to suppress excessive oscillations within this range:

$$
\mathcal { R } _ { \mathrm { c u r v } } ( \pmb { A } ) = \frac { 1 } { ( V - 2 ) D } \left\| D _ { \theta } ^ { 2 } \Phi _ { K } \pmb { A } ^ { T } \right\| _ { F } ^ { 2 } ,\tag{14}
$$

where $D _ { \theta } \in \mathbb { R } ^ { ( V - 1 ) \times V }$ and $D _ { \theta } ^ { 2 } \in \mathbb { R } ^ { ( V - 2 ) \times V }$ denote the first- and second-order differences along the view sequence, respectively. The superscript 2 denotes the difference order, not a matrix square. The notation $\Vert \cdot \Vert _ { F }$ denotes the Frobenius norm.

Both true projection structures and dynamic SA may contain low angular frequencies, so smoothness alone may still allow structural content to enter the SA branch. To address this, as shown in the branch imposing the orthogonality constraint in Fig. 2(C), we first construct an orthonormal basis for the angular-gradient subspace of dynamic SA:

$$
Q = \operatorname { o r t h } \left( D _ { \theta } \Phi _ { K } \right) .\tag{15}
$$

For the dth detector element, the coordinates of their projected angular gradients in this basis are

$$
\begin{array} { r l } & { \boldsymbol { u } _ { d } = \boldsymbol { Q } ^ { T } \boldsymbol { D } _ { \theta } \widehat { \boldsymbol { P } } _ { \Theta , : , d } , } \\ & { \boldsymbol { v } _ { d } = \boldsymbol { Q } ^ { T } \boldsymbol { D } _ { \theta } \widehat { \boldsymbol { S } } _ { \mathrm { d y n } , : , d } , } \end{array}\tag{16}
$$

where $\pmb { Q } \in \mathbb { R } ^ { ( V - 1 ) \times K }$ and $\boldsymbol { u } _ { d } , \boldsymbol { v } _ { d } \in \mathbb { R } ^ { K }$ . Their normalized inner product is

$$
\rho _ { d } = \frac { \pmb { u } _ { d } ^ { T } \pmb { v } _ { d } } { \sqrt { ( \| \pmb { u } _ { d } \| _ { 2 } ^ { 2 } + \varepsilon ) ( \| \pmb { v } _ { d } \| _ { 2 } ^ { 2 } + \varepsilon ) } } ,\tag{17}
$$

which defines the angular-gradient soft orthogonality constraint:

$$
\mathcal { L } _ { \perp } = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } \rho _ { d } ^ { 2 } ,\tag{18}
$$

where $\varepsilon > 0$ ensures numerical stability. This soft constraint penalizes aligned and oppositely aligned gradients in the shared subspace, reducing ambiguity between structures and stripes. Since the first- and second-order angular differences of fixed SA are zero, the orthogonality and curvature constraints connected to total SA in the diagram effectively act on the dynamic component.

As shown in Fig. 2(C), these constraints are introduced in two stages. In Stage I, SA is represented only by fixed column biases, and the IS branch parameters Θ and fixed biases b are jointly optimized:

$$
\mathcal { L } _ { \mathrm { I } } = \mathcal { L } _ { \mathrm { d c } } \left( \widehat { P } _ { \Theta } , \mathbf { 1 } _ { V } \boldsymbol { b } ^ { T } \right) + \lambda _ { \mathrm { d e t } } \mathcal { R } _ { \mathrm { d e t } } \left( \widehat { P } _ { \Theta } \right) .\tag{19}
$$

This stage limits the degrees of freedom of SA and provides an initial decomposition for subsequent dynamic stripe estimation.

Stage II retains Θ and b from Stage I, initializes the newly introduced dynamic coefficients A to zero, and jointly updates all three parameter groups:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { I I } } = \mathcal { L } _ { \mathrm { d c } } \left( \widehat { P } _ { \Theta } , \widehat { S } \right) + \lambda _ { \mathrm { d e t } } \mathcal { R } _ { \mathrm { d e t } } \left( \widehat { P } _ { \Theta } \right) } \\ & { ~ + \lambda _ { \mathrm { c u r v } } \mathcal { R } _ { \mathrm { c u r v } } ( A ) + \lambda _ { \perp } \mathcal { L } _ { \perp } , } \end{array}\tag{20}
$$

where $\lambda _ { \mathrm { d e t } } , \lambda _ { \mathrm { c u r v } }$ , and $\lambda _ { \perp }$ are the corresponding weights. Introducing the dynamic component and its angular constraints after the initial separation of fixed stripes helps prevent dynamic SA from prematurely absorbing true structures.

After optimization, the final IS and SA estimates are $P ^ { * } = \widehat { P } _ { \Theta }$ ∗ and $S ^ { * } = { \widehat S } ( b ^ { * } , A ^ { * } )$ , respectively. As shown in Fig. 2(D), $P ^ { * }$ is reconstructed using filtered backprojection (FBP) to obtain the corrected CT image. The complete optimization procedure is summarized in Algorithm 1.

## III. EXPERIMENTS

All experiments used measured PCD-CT data to retain detector-specific response errors and their angular variations. Synthetic stripes may not reproduce these patterns and can favor the assumptions of the model used to generate them.

Algorithm 1 TRACE   
Require: Measured sinogram Y, number of non-DC DCT   
basis functions K, Stage I and II iteration counts $T _ { 1 }$ and   
$T _ { 2 } ,$ weights $\lambda _ { \mathrm { d e t } } , \lambda _ { \mathrm { c u r v } } ,$ and $\lambda _ { \perp }$   
Ensure: Final IS $P ^ { * }$ and $\mathbf { S A } \textbf { } S ^ { * }$   
1: Initialize IS branch parameters Θ and fixed biases b   
2: Stage I: Fixed stripe decomposition   
3: for t = 1 to $T _ { 1 }$ do   
4: Compute IS $\hat { P } _ { \Theta }$ using (6)– (8)   
5: Compute $\hat { \boldsymbol { S } } _ { \mathrm { f i x } }$ using the fixed component in (9)   
6: Compute ${ \mathcal { L } } _ { \mathrm { I } }$ using (19)   
7: Update Θ and b   
8: end for   
9: Stage II: Joint fixed–dynamic stripe refinement   
10: Retain Θ and b from Stage I   
11: Initialize $\mathbf A = \mathbf 0$   
12: Construct Φ and Q using (10) and (15)   
13: for t = 1 to $T _ { 2 }$ do   
14: Compute IS ${ \hat { P } } _ { \Theta }$   
15: Compute fixed SA, dynamic SA, and their sum $\widehat { \boldsymbol { S } }$   
using (9)   
16: Compute ${ \mathcal { L } } _ { \mathrm { I I } }$ using (20)   
17: Update Θ, b, and A   
18: end for   
19: Recompute the estimates using the final parameters:   
$P ^ { * }  \widehat { P } _ { \Theta ^ { * } }$ and $S ^ { * } \gets \widehat { S } ( b ^ { * } , A ^ { * } )$   
20: return $P ^ { * }$ and $S ^ { * }$

![](images/aa015cb660e7099557b880fc3229bc33dc8ca0a7ea06ea22cbaefcf5c5233573.jpg)  
(a)

![](images/a840bdfdbf619f8175e957f0b012156661095731ffc5a2e33a36a02e552c1fe9.jpg)  
(b)

![](images/4de73467bc7539a8331548f9c0b834b1e7b1fc978678ee7cc80cb0687946f77e.jpg)  
(c)  
Fig. 3. Experimental system and scanned objects. (a) In-house PCD-CT system. (b) QRM mouse phantom. (c) Porcine trotter specimen.

We used two complementary datasets: a QRM mouse phantom with homogeneous regions and well-defined boundaries to evaluate artifact suppression and edge sharpness, and a porcine trotter specimen with soft tissue, bone, and trabeculae to assess structure preservation in a complex background. The experiments examine the number of DCT basis functions, comparisons with representative methods, and the contribution of dynamic stripe modeling.

## A. Experimental Setup

Data were acquired using the in-house PCD-CT system in Fig. 3(a), comprising an L10101 X-ray source (Hamamatsu Photonics, Japan) and an EIGER2 1M-W R photoncounting detector (DECTRIS, Switzerland). The Micro-CT Mouse Phantom (QRM-70137, QRM GmbH, Germany) and porcine trotter specimen in Fig. 3(b) and (c) were scanned at 100 and 80 kVp, respectively. Both scans used a tube current of 300 $\mu \mathrm { A }$ and an exposure time of 1 s per view. Detector thresholds were set to 10 and 30 keV, and only the counts recorded with the 10 keV threshold were used.

![](images/f69390312bd86ae76ab1c5fc445f156e311d805b8e4d7a5ca1e19c46526f7ad1.jpg)  
Fig. 4. Effect of the number of DCT basis functions on the porcine trotter results. From left to right: the uncorrected result and corrected results for $\bar { K = 1 , \ldots , 9 } .$ . The first row shows reconstructed images, and the second row shows magnified views of the same region marked by the red box in the uncorrected image. The display windows are [0, 1] and [0.10, 0.45], respectively.

![](images/cb82bc0a55fe188f5326e904274ff9bc6bee6c35cc62bbd08500aa2af425bbc3.jpg)

![](images/3568a83b81aaabf81e9cf008d391283c57aca01f101ad0eb6ca65baecef483cc.jpg)

![](images/7123487b9ca4992712ee780d7d4c68e1fd2512ad805c83e52af7830a944ece9d.jpg)

![](images/d6eca4f43d171fd4f488ab7de78e9b8331e724181d99a3e5bfccc6ca3f4c6560.jpg)

![](images/6566cc8b6866013f51e7910fb19afa7965328c8c3f8ce6a5c6d3c389ffa2c018.jpg)

![](images/14f00f7a15cd9cda022f87b0ee62858f3ec5b1a8154e3332e4b10f0b2056242a.jpg)  
Fig. 5. Effect of the number of DCT basis functions on stripe estimation. (a)–(e) Estimated stripe artifact amplitude versus projection angle at detector elements 650, 850, 1000, 1200, and 1450, respectively, for $K = 1 , \ldots , 9 .$ The amplitude is the dimensionless logarithmic projection residual obtained by subtracting the corrected projection from the original projection. (f) Relative $L _ { 2 }$ change (%) between consecutive values of K, where the horizontal coordinate K denotes the comparison from K − 1 to K. Colored curves correspond to the five detector elements, and the black curve shows their arithmetic mean relative change.

Both scans used source-to-rotation-center and source-todetector distances of 299.064 and 418.657 mm, respectively, with 720 uniformly spaced views over 360<sup>◦</sup>. The central three detector rows were averaged to form fan-beam projections, and 2068 lateral samples with a detector element width of 0.075 mm were used for reconstruction. FBP with a Ram– Lak filter produced 2048 × 2048 images covering approximately 110.8 mm × 110.8 mm, with a square pixel width of approximately 0.0541 mm.

TRACE was compared with TV-L<sup>1</sup>aG [11], RAS [10], VBSC [16], DDR [25], and INR [40], covering sinogram preprocessing, image postprocessing, dual-domain iteration, and unsupervised learning. Supervised methods requiring paired training data were excluded because paired artifact-free references were unavailable. TRACE is labeled Ours in the comparison figures.

QRM evaluation used the percentage of ring artifacts correction (PRAC) [41] and the modulation transfer function (MTF). PRAC measures changes in local intensity fluctuations in polar-transformed images relative to the uncorrected result, without requiring artifact-free ground truth. Mean PRAC was computed in the homogeneous bone-equivalent region using a stabilizing constant of $C = 0 . 5 .$ . Higher values indicate greater artifact suppression. Since smoothing can also increase PRAC, MTF was estimated from the circular edge of the same region. MTF50 and MTF10 are the spatial frequencies at which the normalized MTF falls to 50% and 10%, respectively, and were used to assess edge sharpness. All methods used the same rotation center, polar transformation, and evaluation region.

Porcine trotter evaluation used reconstructed images, image residuals, and magnified views of three regions of interest (ROIs): low-contrast soft tissue, trabecular bone, and the region near the rotation center. Image residuals were defined as corrected minus uncorrected reconstructions, using identical ROIs and display windows for all methods. Angular projection residuals were defined as original minus corrected logarithmic projections to show the removed projection components. TRACE used $K = 7$ except in the $K = 1 , \ldots , 9$ comparison. An ablation compared fixed stripe correction with joint fixed and dynamic stripe correction.

![](images/d9742c415e58c986e0a868463f48ed4569b3cb4c65e6398621412ddeee6b0313.jpg)  
Fig. 6. Ring artifact correction results of different methods on the QRM mouse phantom. The first row shows reconstructed images, and the second row shows magnified views of the circular water-equivalent region marked by the red box in the uncorrected image. The display windows are [0, 0.4] and [0.2, 0.28], respectively.

![](images/7531e3de61f443db05d48186555e96d740c8c9e458bd5a2a4796a84191929773.jpg)

![](images/79f6ea6018d7d5fcd21147a2d32912149a6f9f8570c91ae787c7731a159aff28.jpg)

![](images/4fc9830176d7ac961088d6fa35f22dec98a84c0560480e0433882a7f9c3f59df.jpg)  
Fig. 7. Quantitative evaluation of the bone-equivalent region in the QRM mouse phantom: (a) PRAC. (b) MTF50. (c) MTF10. PRAC is computed relative to the uncorrected result (Original) and reported for the five comparison methods and TRACE (labeled Ours). MTF50 and MTF10 are expressed in lp/mm. Horizontal dashed lines indicate the corresponding values for Original.

## B. Effect of the Number of DCT Basis Functions

To assess the effect of the number of DCT basis functions, we compared the porcine trotter results for $K = 1 , \ldots , 9$ . As shown in Fig. 4, smaller values of K leave visible residual rings, particularly in low-contrast soft tissue. The rings generally weaken as K increases, whereas visual differences in both reconstructed images and magnified regions become small for $K \geq 7$

Fig. 5(a)–(e) shows the estimated stripe curves at five detector elements. Fewer basis functions mainly capture the overall trend. As more basis functions are included, local variations become better resolved and the main peaks and troughs stabilize. To quantify changes between consecutive settings, let $\pmb { s } _ { d } ^ { ( K ) } \in \mathbb { R } ^ { V }$ denote the original-minus-corrected projection residual at detector element d using K DCT basis functions. The relative change is defined as

$$
r _ { d } ( K ) = \frac { \left\| s _ { d } ^ { ( K ) } - s _ { d } ^ { ( K - 1 ) } \right\| _ { 2 } } { \left\| s _ { d } ^ { ( K - 1 ) } \right\| _ { 2 } } \times 1 0 0 \% , \qquad K = 2 , \ldots , 9 .\tag{21}
$$

As shown in Fig. 5(f), the mean relative change across the five detector elements generally decreases as K increases. The estimate at element 1450 still changes appreciably from $K = 6$ to $K = 7$ , whereas the mean relative changes from $K = 7$ to K = 8 and from K = 8 to K = 9 are both below 10%. Given the limited further improvement in the reconstructed images, we selected $K = 7$ to balance correction performance, stripe estimation stability, and model complexity.

## C. Comparison with Representative Methods

We first evaluated artifact suppression and spatial resolution preservation using homogeneous regions and well-defined boundaries in the QRM mouse phantom, then assessed correction performance in the complex tissue background of the porcine trotter specimen.

1) QRM Mouse Phantom: Fig. 6 shows the reconstructed QRM mouse phantom images and magnified views of the circular water-equivalent region. Ring artifacts disrupt intensity uniformity within homogeneous materials in the uncorrected image. The magnified views reveal residual arcs of varying severity with TV-L<sup>1</sup>aG, RAS, and DDR, pronounced smoothing and blurring with VBSC, and additional linear artifacts with INR. TRACE more effectively suppresses ring-induced fluctuations in the water-equivalent region while preserving sharp boundaries.

DDR  
![](images/088e30b931662f688b1775b9cbd74f5e2187330e579c17bfda0ec36d6b0e7a68.jpg)  
Fig. 8. Ring artifact correction results of different methods on the porcine trotter specimen. The first row shows reconstructed images with a display window of [0, 1]. Blue, yellow, and red boxes in the uncorrected image indicate regions of interest (ROIs) 1, 2, and 3, respectively. The second row shows image residuals, defined as the corrected reconstruction minus the uncorrected reconstruction, with a display window of [−0.1, 0.1]. Red and blue indicate positive and negative values, respectively.

TV-L <sup>1</sup> aG  
VBSC  
![](images/ab79d65c7130b1f352cdcf93e53228cd40a8918237fa6d698930487830a997d7.jpg)  
Fig. 9. Magnified views of the porcine trotter specimen. From top to bottom: ROI 1, ROI 2, and ROI 3, corresponding to the blue, yellow, and red boxes in Fig. 8 and showing low-contrast soft tissue, trabecular bone, and the region near the rotation center, respectively. The display windows are [0.1, 0.5], [0.4, 0.6], and [0.2, 0.4], respectively.

Fig. 7 presents the quantitative results for the boneequivalent region. VBSC achieves the highest PRAC (1.86%), but its MTF50 and MTF10 decrease to 2.16 and 3.72 lp/mm, respectively. This agrees with the observed blurring and indicates that its improvement in uniformity comes at the cost of spatial resolution. TRACE achieves a PRAC of 0.98%, comparable to RAS and higher than TV-L<sup>1</sup>aG, DDR, and INR. Its MTF50 and MTF10 are 3.31 and 5.77 lp/mm, respectively, compared with 3.27 and 5.54 lp/mm for the uncorrected image. Together with the visual results, these findings indicate that TRACE balances ring artifact suppression and edge sharpness preservation.

2) Porcine Trotter Specimen: Fig. 8 shows the reconstructed porcine trotter images and image residuals. Dense ring artifacts extend across soft tissue and bone in the uncorrected image, obscuring tissue organization and fine details. The first row shows that TRACE effectively suppresses rings across different regions while preserving clear soft-tissue contours and internal bone texture. The other methods exhibit local residual artifacts, blurred details, or texture distortion.

The residual images in the second row further reveal the components altered by correction. Bone contours and tissue textures remain discernible in the residuals of some comparison methods, indicating that true structures are also affected. The residuals of TRACE are dominated by ring components with few visible traces of bone or soft-tissue structures, supporting selective artifact separation and structure preservation.

Fig. 9 further compares three ROIs. In the low-contrast soft-tissue region, ROI 1, TRACE substantially reduces arcshaped artifacts and reveals tissue organization previously obscured by rings. In the trabecular bone region, ROI 2, TRACE preserves fine trabeculae and intertrabecular spaces, whereas VBSC shows pronounced blurring and INR exhibits substantial texture distortion. Near the rotation center, ROI 3, VBSC leaves strong concentric rings, TV-L<sup>1</sup>aG leaves a central dark spot, and INR produces pronounced local structural distortion. TRACE effectively suppresses the central artifacts while preserving local structural continuity. These results support effective ring suppression with detail preservation in complex biological tissue.

![](images/ee8c6d31abcdc23214d2be9828cfc3c45d14b908e3e639a114dfeb45edd4f90e.jpg)  
Fig. 10. Angular projection residuals of different methods on the porcine trotter data. (a) Original sinogram, with yellow, green, and cyan vertical lines marking detector elements 700, 1000, and 1300, respectively. (b)–(d) Dimensionless logarithmic projection residuals versus projection angle at the corresponding detector elements. Residuals are defined as the original projection minus the projection corrected by each method and represent the removed projection components.

Fixed + dynamic stripe  
![](images/6281f819ea53446e04a8cf5be4241c39e24a4d6a7d3352e99d4964071abdf19d.jpg)  
Fig. 11. Effect of dynamic stripe modeling on the porcine trotter results. From left to right: no correction, fixed stripe correction only, and joint fixed and dynamic stripe correction. The first row shows reconstructed images, and the second row shows magnified views of the region marked by the red box in the uncorrected image. The display windows are [0, 1] and [0.2, 0.5], respectively.

## D. Analysis of Angle-Dependent Stripe Artifacts

Projection-domain stripe amplitudes can vary with projection angle, which a fixed column bias cannot fully describe. We examined the role of dynamic modeling using angular projection residuals and an ablation comparing fixed stripe correction with joint fixed and dynamic stripe correction.

Fig. 10 shows the projection residuals at detector elements 700, 1000, and 1300. The residuals of TV-L<sup>1</sup>aG are nearly constant, whereas those of VBSC and INR exhibit substantial local fluctuations. The low-order DCT expansion allows TRACE to estimate a smooth, distinct angular correction profile for each detector element. For example, the estimated curve at element 1000 has a trough near $2 3 0 ^ { \circ }$ , whereas that at element 1300 changes from positive to negative and then rises again. These variations cannot be represented by a single fixed bias, illustrating the ability of the dynamic component to capture angle-dependent corrections.

Fig. 11 further demonstrates the contribution of dynamic modeling to the reconstructed images. Fixed stripe correction alone removes most strong concentric rings, but dense arcshaped artifacts remain in the lower part of the image and are particularly evident in the magnified view. Adding the dynamic component further suppresses these artifacts and improves soft-tissue uniformity while retaining visible tissue texture and a clear outer boundary. Together, the angular residuals and reconstructed images support joint fixed and dynamic modeling for suppressing artifacts left by fixed-bias correction.

## IV. CONCLUSION

TRACE estimates and corrects response-related projection errors in PCD-CT through unsupervised sinogram decomposition. A low-order DCT expansion captures smooth angular stripe variations with few coefficients per detector element, while two-stage optimization and angular-gradient soft orthogonality reduce structure leakage into the artifact estimate. Experiments on measured QRM mouse phantom and porcine trotter data show that dynamic modeling suppresses rings left by fixed-bias correction. The resulting images exhibit improved uniformity with preserved edge sharpness, softtissue texture, and trabecular detail.

The current method uses a predefined number of DCT basis functions, and validation is limited to two-dimensional data from a single energy threshold. Future work will explore adaptive selection of the number of basis functions according to angular stripe variations and extend the framework to joint correction of multi-energy three-dimensional PCD-CT data by exploiting correlations across energy channels and detector rows.

## REFERENCES

[1] J. Sijbers and A. Postnov, “Reduction of ring artefacts in high resolution micro-CT reconstructions,” Physics in Medicine and Biology, vol. 49, no. 14, pp. N247–N253, 2004.

[2] B. Münch, P. Trtik, F. Marone, and M. Stampanoni, “Stripe and ring artifact removal with combined wavelet–Fourier filtering,” Optics Express, vol. 17, no. 10, pp. 8567–8591, 2009.

[3] G.-H. Chen, R. Lai, and K. Li, “A physics primer on photon-counting detectors in CT: Physics, signal formation, and performance,” Medical Physics, vol. 53, no. 8, p. e70586, 2026.

[4] D. Lee, X. Zhan, W. Y. Tai, S. Subramanian, W. Zbijewski, and K. Taguchi, “Calibration method for reducing ring artifacts in energy bin images for photon-counting CT,” IEEE Transactions on Medical Imaging, vol. 45, no. 2, pp. 569–582, 2026.

[5] S. S. Hsieh, J. Day, X. Deng, and M. Bazalova-Carter, “Ring artifact reduction in photon counting CT using redundant sampling and autocalibration,” IEEE Transactions on Medical Imaging, vol. 45, no. 5, pp. 2532–2541, 2026.

[6] V. Van Nieuwenhove, J. De Beenhouwer, F. De Carlo, L. Mancini, F. Marone, and J. Sijbers, “Dynamic intensity normalization using eigen flat fields in X-ray imaging,” Optics Express, vol. 23, no. 21, pp. 27 975– 27 989, 2015.

[7] X. Guo, W. Wu, X. Duan, H. Yu, D. Chang, P. He, J. Wang, R. Zhou, Y. Du, and K. An, “Nonlinear deviation correction for ring-artifact removal with cone-beam computed tomography,” IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1–10, 2022.

[8] Y. Chen, Y. Xing, L. Zhang, Z. Deng, and H. Gao, “Energy-threshold bias calculator: A physics-model-based adaptive correction scheme for photon-counting CT,” IEEE Transactions on Medical Imaging, vol. 45, no. 5, pp. 2385–2402, 2026.

[9] H. Guo, D. Zeng, H. Zhang, J. Huang, J. Zhang, and J. Ma, “CT ring artifact reduction using an improved wavelet filtering in the sinogram domain,” Journal of Southern Medical University, vol. 35, no. 9, pp. 1258–1262, Sep. 2015. [Online]. Available: https://doi.org/10.3969/j.issn.1673-4254.2015.09.07

[10] N. T. Vo, R. C. Atwood, and M. Drakopoulos, “Superior techniques for eliminating ring artifacts in X-ray micro-tomography,” Optics Express, vol. 26, no. 22, pp. 28 396–28 412, 2018.

[11] K. An, J. Wang, R. Zhou, F. Liu, and W. Wu, “Ring-artifacts removal for photon-counting CT,” Optics Express, vol. 28, no. 17, pp. 25 180–25 193, Aug. 2020. [Online]. Available: https://doi.org/10.1364/OE.400108

[12] Y. Mäkinen, S. Marchesini, and A. Foi, “Ring artifact reduction via multiscale nonlocal collaborative filtering of spatially correlated noise,” Journal of Synchrotron Radiation, vol. 28, no. 3, pp. 876–888, 2021.

[13] Y. Li, Y. Zhao, D. Ji, S. Han, M. Zheng, W. Lv, X. Xin, F. Li, X. Zhao, D. Liu, and C. Hu, “3D ring artifacts removal algorithm combined low-rank tensor decomposition with spatial–sequential total variation regularization and its application in phase-contrast microtomography,” Medical Physics, vol. 49, no. 1, pp. 393–410, 2022.

[14] D. Kazimirov, D. Polevoy, A. Ingacheva, M. Chukalina, and D. Nikolaev, “Adaptive automated sinogram normalization for ring artifacts suppression in CT,” Optics Express, vol. 32, no. 10, pp. 17 606–17 643, 2024.

[15] X. Liu, S. Zhao, D. Xia, W. Zhang, and F. An, “Adaptive patch-based ring artifacts removal for photon-counting CT,” Optics Express, vol. 33, no. 1, pp. 18–33, 2025.

[16] L. Yan, T. Wu, S. Zhong, and Q. Zhang, “A variation-based ring artifact correction method with sparse constraint for flat-detector CT,” Physics in Medicine and Biology, vol. 61, no. 3, pp. 1278–1292, Feb. 2016. [Online]. Available: https://doi.org/10.1088/0031-9155/61/3/1278

[17] X. Liang, Z. Zhang, T. Niu, S. Yu, S. Wu, Z. Li, H. Zhang, and Y. Xie, “Iterative image-domain ring artifact removal in cone-beam CT,” Physics in Medicine and Biology, vol. 62, no. 13, pp. 5276–5292, 2017.

[18] Y. Li, Y. Zhao, D. Ji, W. Lv, X. Xin, X. Zhao, D. Liu, Z. Ouyang, and C. Hu, “Sparse-domain regularized stripe decomposition combined with guided-image filtering for ring artifact removal in propagation-based Xray phase-contrast CT,” Physics in Medicine and Biology, vol. 66, no. 10, p. 105011, 2021.

[19] Y. Zhao, C. Ma, D. J. Ji, Y. Peng, F. Li, Y. Li, and C. Hu, “Structureaware guided filtering for a ring artifact correction in synchrotron Xray microtomography,” Applied Optics, vol. 62, no. 28, pp. 7400–7410, 2023.

[20] Y. Wang, Z. Chen, H. Gao, Y. Deng, and L. Zhang, “An analytical form of ring artifact correction for computed tomography based on directional gradient domain optimization,” Medical Physics, vol. 51, no. 6, pp. 4121–4132, 2024.

[21] N. Li, Y. Yin, J. Zhao, and J. Xia, “Computed tomography ring artifact correction method with super-pixel segmentation and adaptive relative total variation,” Quantitative Imaging in Medicine and Surgery, vol. 15, no. 4, pp. 2889–2904, 2025.

[22] P. Paleo and A. Mirone, “Ring artifacts correction in compressed sensing tomographic reconstruction,” Journal of Synchrotron Radiation, vol. 22, no. 5, pp. 1268–1278, 2015.

[23] M. Salehjahromi, Q. Wang, Y. Zhang, L. A. Gjesteby, D. Harrison, G. Wang, P. M. Edic, and H. Yu, “A new iterative algorithm for ring artifact reduction in CT using ring total variation,” Medical Physics, vol. 46, no. 11, pp. 4803–4815, 2019.

[24] T. Sun, X. Lu, X. Yu, and Y. Zhao, “Ring-artifacts removal for spectral photon counting CT,” Optics Express, vol. 33, no. 4, pp. 7792–7812, 2025.

[25] X. Lu, H. Zhu, Y. Qin, X. Yu, T. Sun, and Y. Zhao, “A dual-domain regularization method for ring artifact removal of X-ray CT,” Medical Physics, vol. 52, no. 9, p. e18065, 2025.

[26] S. Kan, C. Ren, Z. Liu, Y. Lu, S. Luo, X. Ji, and Y. Chen, “DuDo-RAC: Dual-domain optimization for ring artifact correction in photon counting CT,” Computer Methods and Programs in Biomedicine, vol. 263, p. 108636, 2025.

[27] Y. Qin, X. Su, X. Lu, B. Yu, Y. Zhao, and F. Meng, “Dual domain optimization algorithm for CBCT ring artifact correction,” IEEE Transactions on Image Processing, vol. 35, pp. 1190–1205, 2026.

[28] F. Li, Y. Li, Z. Wang, C. Ma, D. Ji, W. Lv, Y. He, J. Jian, X. Zhao, C. Hu, and Y. Zhao, “Data-driven and model-guided iterative reconstruction framework for simultaneous sparse-view and ring artifact reduction in

synchrotron X-ray microtomography,” Optics Express, vol. 33, no. 2, pp. 3145–3161, 2025.

[29] S. Chang, X. Chen, J. Duan, and X. Mou, “A CNN-based hybrid ring artifact reduction algorithm for CT images,” IEEE Transactions on Radiation and Plasma Medical Sciences, vol. 5, no. 2, pp. 253–260, 2021.

[30] Y. Liu, C. Wei, and Q. Xu, “Detector shifting and deep learning based ring artifact correction method for low-dose CT,” Medical Physics, vol. 50, no. 7, pp. 4308–4324, 2023.

[31] T. Fu, Y. Wang, K. Zhang, J. Zhang, S. Wang, W. Huang, Y. Wang, C. Yao, C. Zhou, and Q. Yuan, “Deep-learning-based ring artifact correction for tomographic reconstruction,” Journal of Synchrotron Radiation, vol. 30, no. 3, pp. 620–626, 2023.

[32] C. Su, Y. Liu, P. Yang, L. Shi, and X. Zhao, “Ring artifacts correction based on global–local feature interaction guidance in the projection domain,” IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1–14, 2025.

[33] J. Sha and J. Li, “Removing ring artifacts in CBCT images via transformer with unidirectional vertical gradient loss,” Medical Physics, vol. 51, no. 9, pp. 6149–6160, 2024.

[34] Y. Zhang, G. Liu, Y. Liu, S. Xie, J. Gu, Z. Huang, X. Ji, T. Lyu, Y. Xi, S. Zhu, J. Yang, and Y. Chen, “GDP-Net: Global dependencyenhanced dual-domain parallel network for ring artifact removal,” IEEE Transactions on Medical Imaging, vol. 44, no. 6, pp. 2718–2731, 2025.

[35] Y. Zhang, G. Liu, Z. Chen, Z. Huang, S. Kan, X. Ji, S. Luo, S. Zhu, J. Yang, and Y. Chen, “Inter-slice complementarity enhanced ring artifact removal using central region reinforced neural network,” Physics in Medicine and Biology, vol. 70, no. 21, p. 215016, 2025.

[36] J. Shi, D. M. Pelt, and K. J. Batenburg, “Multi-stage deep learning artifact reduction for parallel-beam computed tomography,” Journal of Synchrotron Radiation, vol. 32, no. 2, pp. 442–456, 2025.

[37] D. Hein, S. Holmin, V. Prochazka, Z. Yin, M. Danielsson, M. Persson, and G. Wang, “Syn2Real: Synthesis of CT image ring artifacts for deep learning-based correction,” Physics in Medicine and Biology, vol. 70, no. 4, p. 04NT01, 2025.

[38] M. Li, G. Ma, Y. Zhu, and X. Zhao, “Feature decoupling through random sampling for high-fidelity flow matching-based artifact removal,” IEEE Transactions on Computational Imaging, vol. 12, pp. 660–672, 2026.

[39] T. Wang, X. Liu, C. Zhang, Y. He, Y. Chan, Y. Xie, and X. Liang, “Ring artifacts correction for computed tomography image using unsupervised contrastive learning,” Physics in Medicine and Biology, vol. 68, no. 20, p. 205008, 2023.

[40] L. Shi, X. Jiang, Y. Liu, C. Liu, P. Yang, S. Guo, and X. Zhao, “Ring artifacts removal based on implicit neural representation of sinogram data,” IEEE Transactions on Image Processing, vol. 34, pp. 4080–4091, 2025.

[41] Y. Zou, M. Qi, S. Wang, J. Zhang, H. Jiang, and S. Yao, “Percentage of ring artifacts correction (PRAC): A quantitative specific evaluation metric for the effect of ring artifacts correction in X-ray CT images,” Optics Express, vol. 33, no. 16, pp. 33 844–33 858, 2025.