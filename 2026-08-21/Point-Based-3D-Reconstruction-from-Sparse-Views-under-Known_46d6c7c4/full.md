# Point-Based 3D Reconstruction from Sparse Views under Known Illumination

Magnus Kaufmann Gjerde<sup>1</sup> , Joakim Bruslund Haurum<sup>2,5</sup> , Jeppe Revall Frisvad<sup>3</sup> , Markus Worchel<sup>4</sup> , J. Andreas Bærentzen<sup>3</sup> , and Thomas B. Moeslund<sup>1,5</sup>

<sup>1</sup> Visual Analysis & Perception Lab, Aalborg University Centre for Software Technology, University of Southern Denmark 3 Department of Applied Mathematics and Computer Science, Technical University of Denmark <sup>4</sup> Technische Universität Berlin <sup>5</sup> Pioneer Centre for Artificial Intelligence

Abstract. Sparse view 3D reconstruction is commonly addressed with neural implicit surfaces or dense point-based representations such as Gaussian splatting. Surface-aware splatting methods improve extracted geometry through oriented primitives and regularization, while RadiosityGS incorporates diferentiable light transport through a radiosity inspired finite-element surfel formulation. We propose a diferentiable point rendering method based on opacity-bearing beta surfels. An opacity explicit adjoint light transport formulation provides gradients for surfel geometry and appearance parameters, allowing physically based light transport to constrain reconstruction. Across five synthetic objects reconstructed from ten posed views, our method achieves the lowest mean symmetric Chamfer distance among the evaluated baselines and reduces mean Chamfer distance by 28.5% relative to the strongest point-based baseline while using only 267 surfels on average, approximately 161× fewer primitives. Directional Chamfer results further show improved accuracy and competitive completion relative to related point-based methods. These results show that, in the controlled direct illumination setting, compact beta surfels combined with transport-based optimization can recover surfaces without relying on the tens to hundreds of thousands of primitives used by the evaluated baselines.

Keywords: Diferentiable rendering · Point-based reconstruction · Inverse rendering · Adjoint light transport · Gaussian splatting

## 1 Introduction

Recovering accurate geometry from a small set of images remains challenging because appearance alone does not uniquely determine surface shape, reflectance, and visibility. Recent point-based methods, such as Gaussian Splatting (GS) [8], can synthesize high-quality images by optimizing large sets of primitives. However, image fidelity does not necessarily imply accurate geometry: Optimized primitives may compensate for photometric error through opacity and their appearance parameters rather than converging to a coherent surface. Consequently, meshes extracted from appearance-driven splatting representations can contain floaters, fragmented regions, and excessive geometric detail. Surface-aware variants such as 2DGS [5] and SuGaR [4] improve geometric behavior through oriented primitives and regularization, but they still rely on rasterization-based image formation rather than physically based light transport.

![](images/8b1a9ee54bfa2a69ab835b83b40c39dea6069f51ff03b67e432fe9e5d01c8a6e.jpg)  
Fig. 1: Overview of our optimization pipeline. Starting from a sparse beta-surfel initialization, we alternate primal rendering, residual evaluation, adjoint transport, and parameter updates. The adjoint source is the image-space residual between the rendered and reference images.

Known illumination provides an additional source of geometric supervision. Under calibrated point lighting and Lambertian reflectance, surface shape and orientation afect observed radiance through shading and visibility. Physicsbased diferentiable rendering provides a principled way to exploit these efects by diferentiating image formation through physical light transport [11, 27].

RadiosityGS [6] combines Gaussian surfels with a radiosity-inspired finiteelement formulation of diferentiable light transport for relighting and geometry reconstruction. Its formulation represents radiometric quantities using persurfel coeficients and assumes spatially constant material, emission, and outgoing radiance within each Gaussian-surfel support. This enables eficient viewindependent transport computation and motivates our alternative formulation, which, in contrast, expresses transport through surface-level quantities evaluated at the specific local coordinates of each surfel interaction, allowing radiance and material response to vary across the surfel support.

We adopt this local-coordinate view for compact surface recovery. Our representation uses opacity-bearing beta surfels [12] with spatially varying local support. Rather than assigning one outgoing radiance value to an entire primitive, we evaluate scattering, transmittance, and visibility at sampled surfel coordinates. We then specialize continuous adjoint light transport to this representation, retaining segment transmittance explicitly to obtain analytical parameter gradients for surfel geometry, material, opacity, and local opacity-profile shape. We estimate the resulting transport integrals with Monte Carlo path sampling and evaluate the direct illumination specialization under calibrated point lights.

Figure 1 summarizes the optimization loop. Starting from a sparse beta-surfel initialization, primal rendering produces image residuals that define the adjoint source. The adjoint transport pass then provides gradients used to update surfel geometry, material, opacity, and kernel shape.

We evaluate the method on five synthetic objects reconstructed from ten posed input views under calibrated static near-field point illumination. Our method obtains the lowest mean symmetric Chamfer distance among the evaluated baselines while using only 267 surfels on average. In comparison, the standard 2DGS 30K configuration uses 51k primitives on average. Directional Chamfer measurements show the lowest reconstruction-to-ground-truth error and competitive ground-truth-to-reconstruction error. Although we focus on isolated objects rather than large-scale scenes, we address object-centric 3D reconstruction by recovering geometry with orders of magnitude fewer primitives than the evaluated point-based baselines while maintaining better reconstruction results.

## Our contributions are:

– A method for sparse-view reconstruction under known direct illumination using compact, opacity-bearing beta surfels, with explicit scattering, visibility, and transmittance and without assuming constant outgoing radiance over each surfel.

– An opacity-explicit adjoint light-transport formulation that provides gradients with respect to surfel geometry, albedo, opacity, and local opacity-profile shape.

## 2 Related Work

Physics-based diferentiable rendering. This approach optimizes scene parameters by diferentiating image formation with respect to geometry, materials, or illumination. Early Monte Carlo methods developed estimators for visibility changes and transport derivatives under complex light transport [11, 26, 27]. Adjoint formulations provide an alternative perspective in which gradients are derived at the level of the continuous transport operator before discretization [14, 18]. Recent work has extended diferentiable light transport to point-based scene representations. In particular, RadiosityGS [6] formulates transport over Gaussian surfels using a radiosity-inspired finite-element approximation. In contrast, our method instead evaluates transport using generic Monte Carlo path tracing, without a constant local radiance assumption.

Surface reconstruction from posed images. Neural implicit methods such as NeuS [21] and NeuS2 [22] optimize continuous implicit surfaces from posed images, while GeoSVR [10] uses sparse voxels and geometry-oriented regularization. These methods contextualize our explicit surfel representation, which directly exploits calibrated point-light transport.

Point-based reconstruction and Gaussian splatting. Point-based rendering has a long history in graphics, and Gaussian splatting methods have recently enabled high-quality novel-view synthesis with optimized primitive sets [8, 28]. Surface-aware variants improve geometric behavior by using oriented primitives or additional regularization. In particular, 2DGS [5] represents scenes using oriented 2D Gaussian disks and introduces depth-distortion and normal-consistency losses for mesh reconstruction. SuGaR [4] aligns Gaussian primitives with an extracted surface to improve mesh quality. Our representation is similarly surfaceoriented, but difers in its compact-support opacity kernel, continuous transmittance model, and physics-based optimization under known illumination.

## 3 Method

Our goal is to recover scene parameters $\psi$ (geometry, materials, etc.), whose rendered measurements match the posed reference images under known direct illumination, while evaluating transport at local surfel coordinates rather than assuming constant radiance over each support. We first discuss how we define our geometry primitive before proceeding to diferentiate light transport with respect to the scene parameters.

## 3.1 Point Geometry Parameterization

We introduce our beta surfel elements by expressing them as regular surfaces. We do so by using the theory of Do Carmo [3]. A beta surfel represents a finite planar surface element embedded in $\mathbb { R } ^ { 3 }$ and parameterized by a local tangent frame. Each surfel k is defined by a learnable set of parameters:

$$
\boldsymbol { \psi } _ { k } = \left\{ \mathbf { p } _ { k } , \ \mathbf { t } _ { u , k } , \ \mathbf { t } _ { v , k } , \ \mathbf { s } _ { k } , \ \rho _ { k } , \ \eta _ { k } , \ \beta _ { k } \right\} ,\tag{1}
$$

where $\mathbf { p } _ { k } \in \mathbb { R } ^ { 3 }$ is the surfel center, $\mathbf { t } _ { u , k }$ and $\mathbf { t } _ { v , k }$ are orthonormal tangent directions, defining the surfel normal by $\mathbf { n } _ { k } = \mathbf { t } _ { u , k } \times \mathbf { t } _ { v , k } , \mathbf { s } _ { k } = \left( s _ { u , k } , s _ { v , k } \right)$ are scale parameters for the tangent directions, $\rho _ { k }$ is a difuse albedo, $\eta _ { k } \in [ 0 , 1 ]$ is a uniform opacity factor, and $\beta _ { k }$ controls the shape of the opacity kernel.

We define a local chart of the regular surface:

$$
\begin{array} { r } { \varPhi _ { k } ( u , v ) = \mathbf { p } _ { k } + s _ { u , k } \mathbf { t } _ { u , k } u + s _ { v , k } \mathbf { t } _ { v , k } v , } \end{array}\tag{2}
$$

with $( u , v ) \in \mathbb { R } ^ { 2 }$ as the local coordinates of the surfel. We also define a constant Jacobian determinant which is found by analyzing the surface with its first fundamental form

$$
\mathrm { d } A = s _ { u , k } s _ { v , k } \mathrm { d } u \mathrm { d } v ,\tag{3}
$$

which we use for reparameterizing the transport operator integral over beta surfels.

We use the local coordinates $( u , v )$ to model continuous visibility and transmittance $\alpha _ { k } ^ { \mathrm { g e o m } }$ . While Gaussian kernels are commonly used in point-based rendering, we instead use a beta kernel presented by Liu et al. [12]:

$$
\alpha _ { k } ^ { \mathrm { g e o m } } ( u , v ; \beta _ { k } ) = \left\{ \begin{array} { l l } { ( 1 - r ^ { 2 } ) ^ { b ( \beta _ { k } ) } , } & { r ^ { 2 } \le 1 , } \\ { 0 , } & { r ^ { 2 } > 1 , } \end{array} \right.\tag{4}
$$

with

$$
b ( \beta _ { k } ) = 4 e ^ { \beta _ { k } } \qquad \mathrm { a n d } \qquad r ^ { 2 } = u ^ { 2 } + v ^ { 2 } .\tag{5}
$$

When $\beta _ { k } = 0$ the kernel starts with a Gaussian-like function. The final efective opacity of the surfel is

$$
\alpha _ { k } ( u , v ) = \eta _ { k } \alpha _ { k } ^ { \mathrm { g e o m } } ( u , v ; \beta _ { k } ) ,\tag{6}
$$

where $0 \leq \alpha _ { k } \leq 1$ . Later in ablation studies it is shown that optimizing the beta profile improves reconstruction on our data relative to the fixed Gaussian-like $\beta _ { k } = 0$ profile. Furthermore, a benefit of the compactly supported beta kernel is that it removes the need for a heuristic footprint cutof which can vary between diferent Gaussian Splatting implementations.

As shown with our definition of opacity, a beta surfel is not an opaque surface element. Its local opacity profile both determines scattering at a sampled point and attenuates every light or camera connection passing through its support. Consequently, changing a surfel parameter afects image formation in two ways, either through local scattering or segment transmittance. Monte Carlo implementations can account for opacity through stochastic path continuation which is suficient for rendering, but it does not directly expose the derivatives of the continuous transmittance needed to diferentiate analytically. We therefore retain a segment transmittance defined as $\tau _ { \psi }$ explicitly before forming the light transport gradients.

This also difers from the primitive-level finite-element discretization used by RadiosityGS [6], which represents radiometric quantities using per-surfel coeficients and assumes them spatially constant within each Gaussian-surfel support. In contrast, we estimate transport at sampled local coordinates within each betasurfel support.

## 3.2 Light Transport Gradients with Explicit Transmittance

The gradients considered here could alternatively be obtained by applying reversemode automatic diferentiation to a sampled rendering computation. While this can be practical at first, diferentiating recursive transport becomes increasingly demanding as longer path histories and their intermediate states must be stored. We instead follow the adjoint perspective of Radiative Backpropagation [14,18], while retaining explicit transmittance $\tau _ { \psi }$ . We subsequently show by example the zeroth-order term of the adjoint Neumann series to show that an appropriate parameterization of the transport operator keeps the sensor sample fixed, thereby avoiding camera-response derivatives induced by a moving surface projection.

We optimize scene parameters $\psi$ for all surfels described in the previous section, to obtain a rendered image matching an observed reference image. We write beam transmittance as $\tau _ { \psi }$ in the rendering equation, where the $\psi$ subscript denotes the dependence on the scene parameters $\psi .$ . This makes the transmittance terms involving $\partial \tau _ { \psi } / \partial \psi$ explicit and connects the continuous adjoint formulation to our opacity-bearing point primitives. Assuming a pinhole camera at the position $\mathbf { x _ { c } }$ , the intensity captured in one of its pixels is an integral of incident radiance $L _ { i }$ across points $\mathbf { x _ { p } }$ in the projected pixel area, or equivalently an integral of outgoing radiance $L _ { o }$ from observed surface positions $\mathbf { x _ { o } }$ found in the ray direction $\omega = ( \mathbf { x _ { p } } - \mathbf { x _ { c } } ) / t _ { p }$ with $t _ { p } = \| \mathbf { x _ { p } } - \mathbf { x _ { c } } \|$ , such that $\mathbf { x _ { p } } = \mathbf { x _ { c } } + t _ { p }$ ω and ${ \bf x _ { o } } = { \bf x _ { c } } + t _ { o } \omega$ . A least squares cost function for our optimization is then

$$
\mathcal { I } ( L _ { o } , \psi ) = \int _ { A ( \psi ) } \int _ { 4 \pi } W _ { c } ( \mathbf { x } , \omega ) J ( L _ { o } , \psi ) \mathrm { d } \omega d A\tag{7}
$$

$$
\begin{array} { r } { J ( L _ { o } , \psi ) = \frac { 1 } { 2 } ( L _ { o } ( \mathbf { x } _ { p } , - \omega , \psi ) - L _ { \mathrm { r e f } } ( \mathbf { x } _ { p } , - \omega ) ) ^ { 2 } \ . } \end{array}\tag{8}
$$

where $\mathbf { x }$ is a point in $A ( \psi )$ , which is the union of $\mathbf { x } _ { c }$ and all surface positions in the scene. The domain 4π refers to the full solid angle of directions $\omega , L _ { \mathrm { r e f } }$ is radiance sampled in $\mathbf { x } _ { p }$ of the reference image, and $W _ { c } = \left[ { \pmb { \omega } } \in \mathcal { \Omega } \right] \delta ( { \mathbf { x } } - { \mathbf { x } } _ { c } )$ is the camera response function, defined using an Iverson bracket and a Dirac delta function to only include paths observed at $\mathbf { x } _ { c }$ with $\omega$ in the solid angle $\varOmega$ subtended by the camera sensor (the film). The optimal parameters are obtained by solving

$$
\psi ^ { \star } = \arg \operatorname* { m i n } _ { \psi } \mathcal { I } ( L _ { o } , \psi ) \quad \mathrm { s . t . } \quad R ( L _ { o } , \psi ) = 0 ,\tag{9}
$$

where the constraint function R is based on light transport.

In the original discussion of methods for solving the rendering equation, Kajiya [7] mentioned that one option is to use a linear operator $\mathcal { T } _ { \psi }$ to represent the integral in the reflected radiance term. We write the rendering equation including transmittance as:

$$
\begin{array} { r } { L _ { o } = \tau _ { \psi } ( L _ { e } + \mathcal { T } _ { \psi } L _ { o } ) , } \end{array}\tag{10}
$$

where $L _ { e }$ is emitted radiance and $\tau _ { \psi }$ is beam transmittance induced by the opacity-bearing scene representation. This is our main addition to Stam’s formulation [18]. The transport operator $\mathcal { T } _ { \psi }$ acts on outgoing radiance and includes propagation and scattering between surface elements. Writing $\tau _ { \psi }$ explicitly is not required in all presentations of light transport, but it is necessary here because the opacity parameters of the point primitives are optimized and therefore contribute directly to the gradient.

Using Lagrange multipliers and the continuous adjoint representation we end up with the following expression (more details in supplemental Appendix A):

$$
\frac { \partial \mathcal { L } } { \partial \psi } = \frac { \partial \mathcal { I } } { \partial \psi } + \left. \frac { \partial ( \mathcal { T } _ { \psi } ^ { * } \tau _ { \psi } ) } { \partial \psi } p , L _ { o } \right. + \left. p , \frac { \partial \tau _ { \psi } } { \partial \psi } L _ { e } \right. ,\tag{11}
$$

where ∗ denotes the adjoint and $p$ is a Lagrangian multiplier traced as incident directional importance. The appeal of the adjoint formulation is that it eliminates the need to compute the variation of $L _ { o }$ with the parameters $\psi$ explicitly. We only need derivatives of the transport operator kernel for each surfel, yielding favorable scaling properties with respect to the number of optimized parameters.

For a fixed camera and camera response function $W _ { c } .$ the measurement operator and the image domain are independent of the scene parameters $\psi .$ . Consequently, the cost functional $\mathcal { I }$ has no explicit dependence on $\psi .$ , and the first term of Eq. (11) vanishes.

## 3.3 Transport and Transmittance Derivatives

As derived in Sec. A.1 of the supplementary material, the Lagrangian multiplier $p$ is given as a Neumann series:

$$
p = \sum _ { n = 0 } ^ { \infty } ( \mathcal T _ { \psi } ^ { * } \tau _ { \psi } ) ^ { n } \frac { \partial J } { \partial L _ { o } } W _ { c } = \sum _ { n = 0 } ^ { \infty } ( \mathcal T _ { \psi } ^ { * } \tau _ { \psi } ) ^ { n } ( L _ { o } - L _ { \mathrm { r e f } } ) W _ { c } .
$$

To clearly show how the gradient connects to propagation of this adjoint variable, we take a special look at the zeroth-order term of this Neumann series, and denote it by

$$
p _ { 0 } = \left( { \mathcal T } _ { \psi } ^ { * } \tau _ { \psi } \right) ^ { 0 } \left( L _ { o } - L _ { \mathrm { r e f } } \right) W _ { c } = \left( L _ { o } - L _ { \mathrm { r e f } } \right) W _ { c } .\tag{12}
$$

Using the zeroth-order adjoint approximation $p _ { 0 } \approx p ,$ the gradient of the Lagrangian $\mathcal { L }$ with respect to the scene parameters ψ is then

$$
\begin{array} { r l r } {  {  \frac { \partial \mathcal { L } } { \partial \psi } | _ { p \approx p _ { 0 } } =  \frac { \partial ( \mathcal { T } _ { \psi } ^ { * } \tau _ { \psi } ) } { \partial \psi } p _ { 0 } , L _ { o }  +  p _ { 0 } , \frac { \partial \tau _ { \psi } } { \partial \psi } L _ { e }  \cdot } } \\ & { } & { =  \frac { \partial \mathcal { T } _ { \psi } ^ { * } } { \partial \psi } \tau _ { \psi } p _ { 0 } , L _ { o }  +  p _ { 0 } , \frac { \partial \tau _ { \psi } } { \partial \psi } ( L _ { e } + \mathcal { T } _ { \psi } L _ { o } )  . } \end{array}\tag{13}
$$

The first term includes derivatives of transport from the camera to the scene. The second term includes derivatives of the transmittance of directly visible emitters and surfaces. In the following, we describe the computation of the transport and transmittance derivatives.

Transmittance derivative. Here, $\alpha _ { i } ( u _ { i } , v _ { i } )$ is the opacity of a surfel, and i is the i-th surfel in the front-to-back composition between $\mathbf { x } _ { \mathrm { 0 } }$ and $\mathbf { x } _ { 1 }$ :

$$
\tau _ { \psi } ( \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } ) = \prod _ { i \in \mathcal { H } } \left( 1 - \alpha _ { i } ( u _ { i } , v _ { i } ) \right) .\tag{14}
$$

The transmittance derivative, where \protect \mathcal {H} is the fixed set of segment intersections assuming the ordering and membership of intersected surfels remain unchanged on the path between $\mathbf { x } _ { \mathrm { 0 } }$ and $\mathbf { x } _ { 1 }$ , is

$$
\frac { \partial \tau _ { \psi } ( { \bf x } _ { 0 } , { \bf x } _ { 1 } ) } { \partial \psi } = - \sum _ { k \in \mathcal { H } } \frac { \partial \alpha _ { k } ( u _ { k } , v _ { k } ) } { \partial \psi } \prod _ { \stackrel { i \in \mathcal { H } } { i \neq k } } \left( 1 - \alpha _ { i } ( u _ { i } , v _ { i } ) \right) .\tag{15}
$$

Transport derivative. The adjoint transport operator admits equivalent hemispherical and surface-area parameterizations. Here, θ denotes the direction at which the operator is evaluated. In the hemispherical form, ϕ parameterizes the

direction from $\mathbf { x } _ { \mathrm { 0 } }$ and determines the next interaction $\mathbf { x } _ { 1 } = r _ { \psi } ( \mathbf { x } _ { 0 } , \phi )$ . The two forms are

$$
\mathcal { T } _ { \psi } ^ { * } p _ { 0 } = \int _ { \mathbb { S } ^ { 2 } } p _ { 0 } ( r _ { \psi } ( \mathbf { x } _ { 0 } , \phi ) , - \phi ) f _ { s } ( \mathbf { x } _ { 0 } , \theta , - \phi ) \left| \cos ( \mathbf { N } _ { \mathbf { x } _ { 0 } } , \phi ) \right| \mathrm { d } \omega _ { \phi } ,\tag{16}
$$

$$
\mathcal { T } _ { \psi } ^ { * } p _ { 0 } = \int _ { A } p _ { 0 } ( \mathbf { x } _ { 1 } , - \phi ) f _ { s } ( \mathbf { x } _ { 0 } , \pmb { \theta } , - \phi ) G ( \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } ) \mathrm { d } A _ { \mathbf { x } _ { 1 } } .\tag{17}
$$

For primal rendering, either formulation may be used since both represent the same transport operator. Under diferentiation, however, the choice of parameterization becomes important, as discussed by Worchel et al. [23]. We use diferent parameterizations for diferent path segments: Recursive surface-to-surface transport is evaluated using the surface-area form, which permits reparameterization over fixed local surfel coordinates, while the primary camera-to-scene connection is evaluated using the hemispherical form. For recursive surface-tosurface transport, we locally denote the current interaction by $\mathbf { x } _ { \mathrm { 0 } }$ and the next interaction by $\mathbf { x } _ { 1 } = \varPhi _ { k } ( u , v )$ . Its derivative is

$$
\frac { \partial T _ { \psi } ^ { * } } { \partial \psi } p _ { 0 } = \int _ { U } \partial _ { \psi } \left[ p _ { 0 } ( \mathbf { x } _ { 1 } , - \phi ) f _ { s } ( \mathbf { x } _ { 0 } , \pmb { \theta } , - \phi ) G ( \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } ) J _ { \psi } \right] \mathrm { d } u \mathrm { d } v .\tag{18}
$$

Here, U denotes the local surfel-coordinate domain and $J _ { \psi } = s _ { u } s _ { v }$ is the corresponding determinant of the surface-area Jacobian. For the primary camerato-scene propagation, we instead diferentiate the hemispherical form. Let $\mathbf { x } _ { 1 } =$ $r _ { \psi } ( \mathbf { x } _ { c } , \phi )$ , where the camera position $\mathbf { x } _ { c }$ and sampled sensor direction $\pmb \theta$ are held fixed:

$$
\frac { \partial T _ { \psi } ^ { * } } { \partial \psi } p _ { 0 } = \int _ { \mathbb { S } ^ { 2 } } \partial _ { \psi } \left[ p _ { 0 } ( \mathbf { x } _ { 1 } , - \phi ) f _ { s } ( \mathbf { x } _ { 1 } , \pmb { \theta } , - \phi ) \left| \cos ( \mathbf { N } _ { \mathbf { x } _ { 1 } } , \phi ) \right| \right] \mathrm { d } \omega _ { \phi } .\tag{19}
$$

A surface-area parameterization would instead evaluate the adjoint source at the projection of a parameter-dependent surface point, $p _ { 0 } ( \mathbf { X } _ { \psi } ( u , v ) , \omega _ { \psi } ( u , v ) )$ . Differentiating this expression introduces chain-rule terms through the parameterdependent evaluation of $p _ { 0 } .$ , for example $\begin{array} { r } { \nabla _ { \mathbf { x } } p _ { 0 } \frac { \partial \mathbf { X } _ { \psi } } { \partial \psi } + \nabla _ { \omega } p _ { 0 } \frac { \partial \omega _ { \psi } } { \partial \psi } } \end{array}$ , analogous to the pixel-filter derivative discussed by Yu et al. [25]. For a discontinuous box reconstruction filter, this also introduces the associated pixel-boundary contribution. By keeping the sampled camera ray fixed, the hemispherical parameterization evaluates the reconstruction filter at a fixed sensor sample. The geometry dependence is instead represented by the moving camera-ray intersection $\mathbf { x } _ { 1 } = r _ { \psi } ( \mathbf { x } _ { c } , \phi )$

## 4 Problem Setup and Training

The preceding formulation is written for recursive light transport and therefore admits multi-bounce path contributions. In this work, however, we instantiate its direct-illumination specialization for sparse-view geometry reconstruction. Given

![](images/4dd6c04c437f5af95c6551866d66bbbefde5474050bc1f90a731975a4ecdc6f7.jpg)  
Fig. 2: Overview of the synthetic sparse-view object reconstruction dataset. Each scene is reconstructed from ten posed input views rendered with known static point-light illumination and Lambertian materials. For visual clarity, we show one representative training image per scene.

posed reference images, calibrated camera intrinsics and extrinsics, and known static point lights, we optimize the beta-surfel parameters $\psi _ { k }$ defined in Eq. (1).

The optimized objective combines image reconstruction with geometric regularization:

$$
{ \mathcal { L } } _ { \mathrm { t r a i n } } = { \mathcal { I } } _ { \mathrm { r g b } } + \lambda _ { d } { \mathcal { L } } _ { d } + \lambda _ { n } { \mathcal { L } } _ { n } + \lambda _ { \mathrm { w e a k } } { \mathcal { L } } _ { \alpha } .\tag{20}
$$

where $\mathcal { I } _ { \mathrm { r g b } }$ is the image reconstruction loss across all RGB channels, $\mathcal { L } _ { d }$ is the depth-distortion regularizer, ${ \mathcal { L } } _ { n }$ is the normal-consistency regularizer adopted from 2DGS [5], and ${ \mathcal { L } } _ { \alpha }$ is a weak visibility-weighted opacity regularizer.

Weak Opacity Regularizer A black background introduces an opacity/radiance ambiguity where a partially transparent surfel with high reflected radiance can produce a similar image contribution to a more opaque surfel with lower reflected radiance. We therefore introduce a weak visibility-weighted regularizer that encourages opacity only for surfels that contribute to observed training pixels:

$$
\mathcal { L } _ { \alpha } = \sum _ { p \in \mathcal { P } } \sum _ { i \in \mathcal { H } ( p ) } w _ { p , i } \left( 1 - \eta _ { i } \right) ^ { 2 } ,\tag{21}
$$

where $\mathcal { P }$ denotes the set of training pixel rays and $\mathcal { H } ( \boldsymbol { p } )$ is the front-to-back ordered list of surfels intersected by ray p. The visibility weight is

$$
w _ { p , i } = \tau _ { p , i } \alpha _ { p , i } ,\tag{22}
$$

where $\tau _ { p , i }$ is the accumulated transmittance before surfel i and $\alpha _ { p , i }$ is its efective opacity at the ray–surfel intersection. We stop gradients through $w _ { p , i }$ when evaluating this prior. The regularizer therefore cannot be reduced by changing the ray composition and instead directly encourages the opacity parameter of contributing surfels. The weight $\lambda _ { \mathrm { w e a k } }$ in Eq. (20) is kept small, so the prior is intended to regularize the opacity/radiance ambiguity while retaining a small relative weight.

Densification and Pruning We adapt the number of surfels during optimization using clone-only densification based on position gradient and scale-based pruning. A surfel that is incorrectly positioned can reduce its image-space and light-transport contribution by shrinking its footprint. We therefore remove surfels whose scale falls below a fixed minimum threshold, as these primitives have become efectively unsupported by the reconstruction objective.

We do not use opacity pruning or periodic opacity resets commonly used by GS methods [8]. Given a background color in the renderer, a surfel contribution is jointly controlled by opacity and reflected radiance, so reducing opacity can mimic changes in albedo or incident illumination. In preliminary experiments, opacity resets and opacity-based pruning removed useful surfels or encouraged low-opacity configurations that destabilized the normal consistency regularizer. We therefore avoid resetting opacity and remove unsupported surfels only through scale-based pruning.

## 4.1 Dataset, Baselines, and Evaluation

We evaluate on five synthetic object scenes: Teapot [20], LEGO [13], Dragon [2], Horse, and Plant. The Horse and Plant assets were obtained from Blendkit under their respective licenses. We release only rendered data, calibration, and scene-generation code, not the original meshes or textures. Figure 2 shows one representative training image per scene. Each object is reconstructed from ten posed images at 500 × 500 resolution. The camera intrinsics and extrinsics, together with the positions and intensities of three to four static near-field point lights, are known to our method.

The scenes contain isolated Lambertian objects on black backgrounds. Although the geometry is relatively simple, ten views leave substantial portions of the surface weakly observed or unobserved. This setting therefore retains ambiguity between geometry, reflectance, and visibility while allowing controlled evaluation of known-light supervision.

Reference images are rendered in Blender Cycles using the same direct pointlight illumination model assumed during optimization, Lambertian materials, and no indirect illumination. Blender serves as an independent renderer and avoids coupling the dataset-generation implementation to our optimization renderer.

Initialization. For all scenes, we initialize every point-based method from the same set of 25 white primitives positioned on a uniform grid within the scene bounding volume. The primitives share identical initial positions, white appearance, scale 0.1 , opacity 0.5 , and vertical orientation. For our beta surfels, we additionally initialize the shape parameter as $\beta _ { k } = 0$

Baselines. Our primary comparisons are against the point-based methods 2D-GS [5], SuGaR [4], RadiosityGS [6], and GOF [24]. We additionally include NeuS [21] and NeuS2 [22], which use implicit neural surface representations, and GeoSVR [10], which uses a sparse-voxel representation, to contextualize the point-based results within the broader landscape of surface reconstruction methods.

For 2DGS, we report the standard 30K-iteration configuration from the original method. We additionally report a 7K baseline which is the same iteration count that SuGaR uses for 3DGS before applying their meshes extraction procedure. For GeoSVR, we use the oficial implementation without its monocular depth constraint. Monocular depth predictions were unreliable for our black background object images without foreground masks and introduced incorrect geometric constraints. No foreground masks or external monocular depth priors are provided to the evaluated methods.

## 4.2 Implementation Details

All experiments use a custom physically based renderer implemented in modern C++ and accelerated with SYCL [19]. We tested the same implementation on AMD and NVIDIA GPUs using the corresponding SYCL backends. Optimization requires approximately 5–8 GB of VRAM and uses Adam [9] through PyTorch [15], with default optimizer settings except for parameter-specific learning rates.

We selected the regularization weights empirically in preliminary experiments and fixed them across all reported scenes avoiding per scene specific tuning: $\lambda _ { d } = 1 0 0 , \lambda _ { n } = 0 . 0 0 5$ , and $\lambda _ { \mathrm { w e a k } } = 0 . 0 5$ in Eq. (20). Our optimization runs for 60K iterations. We extract meshes by truncated signed distance function (TSDF) fusion of depth maps rendered from the optimized surfels at the input camera poses, following the depth-fusion procedure used by 2DGS [5]. Ground-truth geometry is not used during optimization or mesh extraction.

Evaluation. We report symmetric Chamfer distance between reconstructed and ground-truth meshes. Accuracy denotes reconstruction-to-ground-truth distance and penalizes spurious geometry, while completion denotes ground-truth-toreconstruction distance and penalizes missing surface regions.

## 5 Results

Geometry Reconstruction Table 1 compares our method with implicit, sparsevoxel, and point-based reconstruction baselines. Our method achieves the lowest mean symmetric Chamfer distance and the best result on three scenes. Among the point-based methods, it performs best on four scenes, with SuGaR performing slightly better on LEGO.

NeuS2 performs poorly without foreground masks or monocular-depth supervision. NeuS is the strongest non-point-based baseline, performing slightly better on LEGO and Plant, while our method performs better on the remaining scenes. GeoSVR is less accurate on average, particularly on Plant, however, we omit its monocular-depth prior because its predictions were unreliable on these images.

Table 1: Main geometry reconstruction results for sparse-view object reconstruction. Entries report symmetric Chamfer distance (CD), scaled by $1 0 ^ { 2 }$ for readability; lower is better. The mean is computed across all five scenes. The final column reports the average number of optimized primitives for point-based methods. Within the pointbased methods, the best , second-best , and third-best results are highlighted.
<table><tr><td>Method</td><td>Teapot</td><td>LEGO</td><td>Horse</td><td>Dragon</td><td>Plant</td><td>Mean↓</td><td>Avg pts ↓</td></tr><tr><td>NeuS2 [22]</td><td>95.850</td><td>112.344</td><td>90.242</td><td>101.955</td><td>93.212</td><td>98.721</td><td></td></tr><tr><td>NeuS [21]</td><td>1.396</td><td>1.424</td><td>2.022</td><td>1.397</td><td>2.180</td><td>1.684</td><td></td></tr><tr><td>GeoSVR [10] †</td><td>3.667</td><td>1.446</td><td>2.054</td><td>3.255</td><td>10.296</td><td>4.144</td><td></td></tr><tr><td>2DGS 7K [5]</td><td>2.471</td><td>1.429</td><td>1.702</td><td>2.251</td><td>2.891</td><td>2.149</td><td>43.0k</td></tr><tr><td>2DGS 30K [5]</td><td>53.570</td><td>30.563</td><td>39.075</td><td>53.149</td><td>15.917</td><td>38.455</td><td>51.0k</td></tr><tr><td>SuGaR [4]</td><td>5.837</td><td>1.406</td><td>1.993</td><td>2.613</td><td>3.468</td><td>3.064</td><td>421.4k</td></tr><tr><td>GOF [24]</td><td>3.450</td><td>65.990</td><td>2.394</td><td>2.074</td><td>3.186</td><td>15.419</td><td>72.4k</td></tr><tr><td>RadiosityGS [6]</td><td>31.621</td><td>24.400</td><td>27.327</td><td>28.256</td><td>22.867</td><td>26.894</td><td>107.4k</td></tr><tr><td>Ours</td><td>1.387</td><td>1.426</td><td>1.319</td><td>1.297</td><td>2.252</td><td>1.536</td><td>267</td></tr></table>

† GeoSVR is evaluated without its monocular-depth prior; see Section 4.1.

Table 2: Unscaled mean directional Chamfer distance for the point-based methods across five scenes. Accuracy penalizes spurious geometry, while completion penalizes missing regions.
<table><tr><td>Method</td><td>Accuracy ↓</td><td>Completion↓</td></tr><tr><td>2DGS 7K [5]</td><td>0.020</td><td>0.023</td></tr><tr><td>2DGS 30K [5]</td><td>0.743</td><td>0.026</td></tr><tr><td>SuGaR [4]</td><td>0.037</td><td>0.024</td></tr><tr><td>GOF [24]</td><td>0.295</td><td>0.014</td></tr><tr><td>RadiosityGS [6]</td><td>0.512</td><td>0.026</td></tr><tr><td>Ours</td><td>0.017</td><td>0.014</td></tr></table>

We use the standard 30K-iteration 2DGS schedule as the primary baseline configuration. We additionally report the 7K checkpoint to illustrate an optimization behavior observed in our sparse-view setting. Although the 30K schedule further reduces the image-space objective, later iterations continue to introduce primitives near object boundaries that help explain the black background and reduce photometric residuals. These additional primitives do not improve the extracted surface and instead lead to substantially worse mesh accuracy.

The 7K checkpoint therefore achieves lower Chamfer error in our evaluation and indicates that continued densification and improved image fitting do not necessarily yield better geometry. Figure 3 illustrates this distinction for the 2DGS-30K and 7K results. In contrast, our method is optimized for 60K iterations without the same late-stage degradation in extracted geometry.

RadiosityGS is the closest physics-based point-based baseline. In our setting, it exhibits a related mismatch between image-space optimization and extracted

GT

Ours

SuGaR

2DGS-7K

![](images/714e6822021bbb12d5fc4d2b293c47e8bc9c942ca33a9aee2fb99fee0f0f4ff7.jpg)  
Fig. 3: Qualitative reconstruction comparison across synthetic object scenes. We compare the ground-truth geometry with our reconstruction and baseline methods. Our method preserves the main object structure while using fewer optimized primitives.

mesh quality similar to 2DGS-30k. This suggests that, as with 2DGS, continued primitive growth can improve image fitting without improving the recovered surface. Nevertheless, the Dragon reconstruction in Figure 3 shows a notable benefit. RadiosityGS and our method are the only evaluated approaches that separate the object from the background plane, whereas the remaining methods fuse the two surfaces.

The 2DGS-30K reconstructions can appear smoother in some examples, but visual smoothness does not consistently correspond to geometric fidelity. The directional Chamfer results in Table 2 show that our improvement is not obtained merely by omitting uncertain regions: our method improves accuracy while maintaining competitive completion. Table 1 further shows that this improvement is achieved using only 267 surfels on average, compared with tens to hundreds of thousands of primitives for the point-based baselines. This combination of lower geometric error and a substantially smaller representation suggests a favorable scaling direction where increasing scene complexity need not require the rapid primitive growth observed for the evaluated splatting-based approaches.

Table 3: Ablation study on the Horse scene using ten input views. $\mathcal { L } _ { d } \colon$ depth distortion loss; $\mathcal { L } _ { n } \colon$ normal consistency loss; $\mathcal { L } _ { \alpha } \colon$ weak opacity prior. CD denotes unscaled symmetric Chamfer distance; lower is better.
<table><tr><td>Setting</td><td> $\mathcal { L } _ { d }$ </td><td> ${ \mathcal { L } } _ { n }$ </td><td> $\mathcal { L } _ { \alpha }$ </td><td>Learn β</td><td>CD↓</td><td>Acc. ↓</td><td>Comp. ↓</td></tr><tr><td>Full</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.0132</td><td>0.0138</td><td>0.0125</td></tr><tr><td>w/o  $\mathcal { L } _ { d }$ </td><td></td><td>√</td><td>√</td><td>√</td><td>0.0162</td><td>0.0171</td><td>0.0152</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { n }$ </td><td>√</td><td></td><td>√</td><td>√</td><td>0.0185</td><td>0.0193</td><td>0.0178</td></tr><tr><td>w/o  $\mathcal { L } _ { \alpha }$ </td><td>√</td><td>√</td><td></td><td>√</td><td>0.0241</td><td>0.0187</td><td>0.0295</td></tr><tr><td>Fixed  $\beta$ </td><td>√</td><td>√</td><td>√</td><td></td><td>0.0278</td><td>0.0245</td><td>0.0312</td></tr></table>

These results indicate accurate recovery of the overall object geometry and major surface structures rather than fine-scale detail, making the method more suitable for applications such as coarse path planning, collision-aware navigation, and compact scene representation than high-fidelity digitization.

Ablations Table 3 evaluates the main components of the method on the Horse scene. We include depth distortion and normal regularization to show that they also benefit our optimization similar to 2DGS [5]. The weak visibility-weighted opacity regularizer has a larger impact on the result than depth and normal regularization and shows that for our simple scenes reducing the opacity/radiance ambiguity is beneficial. On Horse, fixing the beta profile similarly degrades reconstruction quality, indicating that the chosen opacity kernel is beneficial over a Gaussian-like shape only. This isolates the benefit of adapting the local opacity fallof during optimization.

## 6 Conclusion

We presented a geometry first point-based reconstruction method based on compact opacity-bearing beta surfels and adjoint light transport with explicit transmittance. Across five synthetic objects reconstructed from ten posed views, the method achieves the lowest mean symmetric Chamfer distance among the evaluated baselines while using only 267 surfels on average. These results indicate that physically informed optimization can recover accurate coarse scale geometry without relying on dense primitive sets.

The present system is a deliberately constrained first instantiation and is not intended as the final form of the method. We evaluate isolated synthetic objects under known direct illumination and Lambertian reflectance, although the formulation admits recursive transport. Extending the system to indirect illumination, jointly estimated illumination, non-Lambertian reflectance, real captures, and more adaptive surfel refinement may provide stronger reconstruction cues. The current results therefore establish the viability of compact transportoptimized surfels while leaving substantial transport and representational capacity unexplored.

## References

1. Christensen, P.H.: Adjoints and importance in rendering: An overview. IEEE Transactions on Visualization and Computer Graphics 9(3), 329–340 (2003). https://doi.org/10.1109/TVCG.2003.1207441

2. Curless, B., Levoy, M.: A volumetric method for building complex models from range images. In: SIGGRAPH ’96. pp. 303–312. ACM (1996). https://doi.org/ 10.1145/237170.237269

3. Do Carmo, M.P.: Diferential Geometry of Curves and Surfaces. Dover Publications, revised and updated second edn. (2016)

4. Guédon, A., Lepetit, V.: SuGaR: Surface-aligned Gaussian splatting for eficient 3D mesh reconstruction and high-quality mesh rendering. In: CVPR. pp. 5354–5363 (2024). https://doi.org/10.1109/CVPR52733.2024.00512

5. Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S.: 2D Gaussian splatting for geometrically accurate radiance fields. In: SIGGRAPH 2024 Conference Papers. pp. 32:1–32:11. ACM (2024). https://doi.org/10.1145/3641519.3657428

6. Jiang, K., Sun, J.M., Li, Z., Wang, D., Li, T.M., Ramamoorthi, R.: Diferentiable light transport with Gaussian surfels via adapted radiosity for eficient relighting and geometry reconstruction. ACM Transactions on Graphics 44(6), 210:1–210:25 (2025). https://doi.org/10.1145/3763305

7. Kajiya, J.T.: The rendering equation. Computer Graphics (SIGGRAPH ’86) 20(4), 143–150 (1986). https://doi.org/10.1145/15886.15902

8. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3D Gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4), 139:1– 139:14 (2023). https://doi.org/10.1145/3592433

9. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: Proceedings of International Conference for Learning Representations (ICLR) (2015). https: //doi.org/10.48550/arXiv.1412.6980

10. Li, J., Zhang, J., Zhang, Y., Bai, X., Zheng, J., Yu, X., Gu, L.: GeoSVR: Taming sparse voxels for geometrically accurate surface reconstruction. In: Advances in Neural Information Processing Systems. vol. 38, pp. 108809–108837. Curran Associates (2025)

11. Li, T.M., Aittala, M., Durand, F., Lehtinen, J.: Diferentiable Monte Carlo ray tracing through edge sampling. ACM Transactions on Graphics 37(6), 222:1–222:11 (2018). https://doi.org/10.1145/3272127.3275109

12. Liu, R., Sun, D., Chen, M., Wang, Y., Feng, A.: Deformable beta splatting. In: SIGGRAPH 2025 Conference Papers. pp. 101:1–101:11 (2025). https://doi.org/ 10.1145/3721238.3730716

13. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: NeRF: Representing scenes as neural radiance fields for view synthesis. In: ECCV. Lecture Notes in Computer Science, vol. 12346. Springer (2020). https: //doi.org/10.1007/978-3-030-58452-8\_24

14. Nimier-David, M., Speierer, S., Ruiz, B., Jakob, W.: Radiative backpropagation: An adjoint method for lightning-fast diferentiable rendering. ACM Transactions on Graphics 39(4), 146:1–146:15 (2020). https://doi.org/10.1145/3386569. 3392406

15. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: PyTorch: An imperative style, highperformance deep learning library. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 32. Curran Associates (2019)

16. Plessix, R.E.: A review of the adjoint-state method for computing the gradient of a functional with geophysical applications. Geophysical Journal International 167(2), 495–503 (2006). https://doi.org/10.1111/j.1365-246X.2006.02978.x

17. Sokolowski, J., Zolésio, J.P.: Introduction to Shape Optimization: Shape Sensitivity Analysis. Springer (1992). https://doi.org/10.1007/978-3-642-58106-9

18. Stam, J.: Computing light transport gradients using the adjoint method. arXiv:2006.15059 [cs.GR] (2020). https://doi.org/10.48550/arXiv.2006.15059

19. The Khronos SYCL Working Group: SYCL™ 2020 Specification. https:// registry.khronos.org/SYCL/specs/sycl-2020/pdf/sycl-2020.pdf (2025), revision 11, accessed July 3, 2026

20. Turk, G., Levoy, M.: Zippered polygon meshes from range images. In: SIGGRAPH ’94. pp. 311–318. ACM (1994). https://doi.org/10.1145/192161.192241

21. Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., Wang, W.: NeuS: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 34, pp. 27171– 27183. Curran Associates (2023)

22. Wang, Y., Han, Q., Habermann, M., Daniilidis, K., Theobalt, C., Liu, L.: NeuS2: Fast learning of neural implicit surfaces for multi-view reconstruction. In: ICCV. pp. 3272–3283. IEEE (2023). https://doi.org/10.1109/ICCV51070.2023.00305

23. Worchel, M., Finnendahl, U., Alexa, M.: Radiative backpropagation with nonstatic geometry. In: Eurographics Symposium on Rendering (EGSR). The Eurographics Association (2025). https://doi.org/10.2312/sr.20251198

24. Yu, Z., Sattler, T., Geiger, A.: Gaussian opacity fields: Eficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics 43(6), 271:1– 271:13 (2024). https://doi.org/10.1145/3687937

25. Yu, Z., Zhang, C., Nowrouzezahrai, D., Dong, Z., Zhao, S.: Eficient diferentiation of pixel reconstruction filters for path-space diferentiable rendering. ACM Trans. Graph. 41(6) (Nov 2022). https://doi.org/10.1145/3550454.3555500, https: //doi.org/10.1145/3550454.3555500

26. Zhang, C., Miller, B., Yan, K., Gkioulekas, I., Zhao, S.: Path-space diferentiable rendering. ACM Transactions on Graphics 39(4), 143:1–143:19 (2020). https:// doi.org/10.1145/3386569.3392383

27. Zhang, C., Wu, L., Zheng, C., Gkioulekas, I., Ramamoorthi, R., Zhao, S.: A diferential theory of radiative transfer. ACM Transactions on Graphics 38(6), 227:1– 227:16 (2019). https://doi.org/10.1145/3355089.3356522

28. Zwicker, M., Pfister, H., Van Baar, J., Gross, M.: EWA splatting. IEEE Transactions on Visualization and Computer Graphics 8(3), 223–238 (2002). https: //doi.org/10.1109/TVCG.2002.1021576

## A Continuous Adjoint Formulation with Explicit Transmission

This section provides the derivation of the opacity-explicit adjoint gradient used in the main paper. For completeness, we restate the rendering equation as

$$
\boldsymbol { L _ { o } } = \tau _ { \psi } \left( \boldsymbol { L _ { e } } + \mathcal { T _ { \psi } } \boldsymbol { L _ { o } } \right) ,\tag{23}
$$

where $L _ { o }$ is outgoing radiance, $L _ { e }$ is emitted radiance, $\mathcal { T } _ { \psi }$ is the light-transport operator, and $\tau _ { \psi }$ is the beam-transmittance operator induced by the opacitybearing scene representation. The image-space objective is written as

$$
\mathcal { I } ( L _ { o } , \psi ) = \left. W _ { c } , J ( L _ { o } , \psi ) \right. ,\tag{24}
$$

where $W _ { c }$ is the camera response and J is the per-measurement loss.

From $\operatorname { E q . }$ (23), the corresponding state constraint is

$$
R ( L _ { o } , \psi ) = - L _ { o } + \tau _ { \psi } \mathcal { T } _ { \psi } L _ { o } + \tau _ { \psi } L _ { e } .\tag{25}
$$

To obtain an unconstrained problem, we introduce a Lagrange multiplier $p .$ The Lagrangian is

$$
\begin{array} { r l } & { \mathcal { L } ( L _ { o } , p , \psi ) = \mathcal { I } ( L _ { o } , \psi ) + \langle p , R ( L _ { o } , \psi ) \rangle } \\ & { \quad \quad \quad = \langle W _ { c } , J ( L _ { o } , \psi ) \rangle + \langle p , R ( L _ { o } , \psi ) \rangle , } \end{array}\tag{26}
$$

with

$$
\langle f , g \rangle = \int _ { A ( \psi ) } \int _ { 4 \pi } f ( x , \omega ) g ( x , \omega ) \mathrm { d } \omega \mathrm { d } A .\tag{27}
$$

Stationarity of the Lagrangian with respect to radiance, $\partial \mathcal { L } / \partial L _ { o } = 0 _ { ; }$ , together with Eq. (25), gives the adjoint equation

$$
\frac { \partial J } { \partial L _ { o } } W _ { c } = - \left( \frac { \partial R } { \partial L _ { o } } \right) ^ { * } p = - \left( \tau _ { \psi } \mathcal { T } _ { \psi } - \mathcal { T } \right) ^ { * } p = \left( \mathcal { T } - \mathcal { T } _ { \psi } ^ { * } \tau _ { \psi } \right) p ,\tag{28}
$$

where $^ *$ denotes the adjoint and $\mathcal { T }$ is the identity operator. We treat beam transmittance as self-adjoint, $\tau _ { \psi } ^ { * } = \tau _ { \psi } . \mathrm { ~ A ~ }$ more detailed account of the transport operator and its adjoint is provided by Christensen [1].

When the adjoint equation is satisfied, stationarity with respect to the scene parameters is an optimality condition, and $\partial \mathcal { L } / \partial \psi$ equals the gradient of the cost function [16]. Before reparameterization, this derivative can be written as

$$
\frac { \partial \mathcal { L } } { \partial \psi } = \frac { \partial \mathcal { I } } { \partial \psi } + \left. p , \frac { \partial R } { \partial \psi } \right. + \left. p , R \left( \nabla _ { A } \cdot \frac { \partial x } { \partial \psi } \right) \right. ,\tag{29}
$$

where the final term accounts for the parameter dependence of the integration domain $A ( \psi ) \left[ 1 7 \right]$ . In our implementation, surface integrals over individual surfels are reparameterized over fixed local coordinates. The corresponding parameter dependence is therefore represented through the diferentiated integrand and the local surface-area Jacobian, rather than through an explicit moving-domain term.

## A.1 Adjoint Light Transport

The rendering equation in Eq. (23) admits the Neumann-series expansion [7]

$$
L _ { o } = ( \mathcal { T } - \tau _ { \psi } \mathcal { T } _ { \psi } ) ^ { - 1 } \tau _ { \psi } L _ { e } = \sum _ { n = 0 } ^ { \infty } \left( \tau _ { \psi } \mathcal { T } _ { \psi } \right) ^ { n } \tau _ { \psi } L _ { e } ,\tag{30}
$$

where each application of the transport operator corresponds to one light-transport bounce. As observed by Stam [18], the adjoint equation has the same structure but uses the adjoint transport operator $\tau _ { \psi } ^ { * }$ . The adjoint operator propagates an incident quantity, such as radiance or importance, rather than an outgoing quantity [1]. The adjoint variable is therefore

$$
p = \sum _ { n = 0 } ^ { \infty } \left( \mathcal T _ { \psi } ^ { * } \tau _ { \psi } \right) ^ { n } \frac { \partial J } { \partial L _ { o } } W _ { c } = \sum _ { n = 0 } ^ { \infty } \left( \mathcal T _ { \psi } ^ { * } \tau _ { \psi } \right) ^ { n } \left( L _ { o } - L _ { \mathrm { r e f } } \right) W _ { c } .\tag{31}
$$

Importantly, tracing $p$ depends neither on the number of optimized parameters nor on their individual derivatives.

Because eye-path tracing is the natural implementation strategy in our differentiable renderer, we express the gradient using derivatives of $\tau _ { \psi } ^ { * }$ rather than derivatives of $\mathcal { T } _ { \psi }$ . Substituting Eq. (25) into Eq. (29), taking the partial derivative with respect to $\psi$ while holding the radiance state $L _ { o }$ fixed, and applying the defining property of adjoint operators gives

$$
\frac { \partial \mathcal { L } } { \partial \psi } = \frac { \partial \mathcal { I } } { \partial \psi } + \left. p , \frac { \partial } { \partial \psi } \left( - L _ { o } + \tau _ { \psi } \mathcal { T } _ { \psi } L _ { o } + \tau _ { \psi } L _ { e } \right) \right. .\tag{32}
$$

This simplifies to

$$
\frac { \partial \mathcal { L } } { \partial \psi } = \frac { \partial \mathcal { I } } { \partial \psi } + \left. \frac { \partial \left( \mathcal { T } _ { \psi } ^ { * } \tau _ { \psi } \right) } { \partial \psi } p , L _ { o } \right. + \left. p , \frac { \partial \tau _ { \psi } } { \partial \psi } L _ { e } \right. ,\tag{33}
$$

where the light sources are assumed not to be parameterized. The first innerproduct term diferentiates transported adjoint importance through scattering and transmission, while the second diferentiates the transmission of emitted radiance. The surface-position integral in the source term for p collapses through the Dirac delta contained in W<sub>c</sub> [Eq. (31)], allowing the gradient to be evaluated by diferentiable path tracing from the camera.