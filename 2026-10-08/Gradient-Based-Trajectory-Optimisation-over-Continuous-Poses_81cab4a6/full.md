# Gradient-Based Trajectory Optimisation over Continuous Poses for Sparse-View Cone-Beam CT

Linda-Sophie Schneider, Simon Wittl, Gabriel Herl and Andreas Maier

Abstract—Trajectory optimisation for cone-beam computed tomography (CT) determines which information sparse-view scans acquire. Fixed candidate pools prevent off-grid refinement and require new object-specific precomputation for each acquisition manifold. We make every source pose an individual continuous variable and move all poses jointly by gradient ascent on the scanner’s kinematic manifold. The objective combines soft-Tuy plane coverage, continuous View Covariance Loss, and an analytic attenuation-aware ray-bundle penalty. The same optimiser handles circular, limited C-arm, two-axis, and freesphere parametrisations. On a Defrise flange, continuous selection recovers laminar defects invisible to a circular orbit, matches discrete swap search on the free sphere at the sparser budget, and leads at the denser one, with the same objective evaluated in every arm. A moderate elevation band already recovers most of the free-sphere gain at the defects, so the same optimiser transfers to bounded scanner envelopes. Photon noise preserves the ordering on the flange and compresses it on a dense fuel nozzle. Sparseprescan planning benefits from matching prescan and planned acquisition manifolds. Selection takes seconds rather than minutes without an object-specific reconstruction basis. Prescan-planned poses were executed on a robot CT bench and reconstructed in a common frame, demonstrating feasibility but no consistent metric gain over uniform band sampling. Continuous pose optimisation incorporates attenuation and scanner constraints directly into sparse-view acquisition design.

Index Terms—Trajectory optimisation, cone-beam computed tomography, sparse-view CT, view selection, differentiable optimisation, optimal experimental design, robotic CT, industrial CT.

## I. INTRODUCTION

Trajectory optimisation for computed tomography (CT) decides where the X-ray source is placed before a scan is acquired. Robot-based and multi-axis systems reach almost any pose on a two-dimensional manifold, so a trajectory is given by the poses themselves rather than by a curve through them [1], [2]. The problem becomes acute under a sparse view budget, in industrial non-destructive evaluation with long acquisition times and in interventional imaging with strict dose and acquisition-speed constraints [3], [4]. Reconstruction from few projections has been studied extensively, with analytical, iterative, and learning-based approaches [5]–[10]. Optimal source placement remains much less settled.

A natural geometric anchor is Tuy completeness [11], which guarantees exact cone-beam reconstruction if every plane through the region of interest intersects the source path. Discrete coverage scores use this condition by sampling plane normals on the sphere [12], [13], later augmented with attenuation-based validity constraints [1], [14]. Our prior work sharpens the resulting discrete selection problem with a soft coverage score, termed soft-Tuy in the following, together with complexity results and certified small-instance optima [15]. Whereas coverage scores reason purely geometrically, the View Covariance Loss Swap Search (VCLS) of Lin et al. [16] selects views by minimising a closed-form surrogate of reconstruction error and performs strongly on simulated and measured cone-beam data. Both select from a fixed discrete candidate pool, the soft-Tuy approach from arbitrary but predefined geometries, VCLS in its published implementation from a circular candidate set, although the VCL criterion itself admits arbitrary view parameters, and it additionally requires an object-specific quadratic precompute and does not model photon starvation.

A fixed pool limits the geometry to the positions it already contains, refining it only enlarges a selection problem that is already NP-complete [15], and every new acquisition manifold needs a new pool. A pool-free search alone does not resolve this. Linde et al. [17] optimise three global orbit parameters by derivative-free differential evolution, so all views move together and none can be steered towards a direction that only a single defect requires. Making each view free instead yields 3k unknowns, the sample count of a population-based search grows with that dimension, and every sample costs a full objective evaluation on the volume. What is missing is a search direction that costs one objective evaluation per step instead of one per sampled perturbation, and an analytic gradient delivers exactly that.

We remove both restrictions by making every source pose an individual continuous variable of a differentiable objective. This matters when views must be refined off-grid, when the acquisition manifold is richer than a circular orbit, and when planning should account for the attenuation the object imposes on each direction.

The objective combines three differentiable terms on the source positions. The soft-Tuy surrogate, a sampled centrepoint plane-coverage score, promotes angular coverage. An analytic bundle term penalises, under a monochromatic Beer– Lambert transmission model, the directions along which the transmitted intensity is lowest. A continuous extension of the VCL information criterion of Lin et al. [16] rewards views that carry reconstruction-relevant information in the noisefree case. Coverage sees only geometry, the bundle term sees only the attenuation along its rays, and the information term assumes noise-free data. One shared objective puts all three

into the same gradient step.

All results in this paper start the optimiser from the greedy soft-Tuy selector of [15], a geometric initialisation that needs no precompute, and refine the source poses on an idealised differentiable kinematic manifold. We realise this with four standard models, namely the single-axis circle, the limited Carm, the two-axis gantry, and the free sphere.

This paper makes four contributions.

• A gradient-based framework for trajectory optimisation. Every source pose is a continuous variable on an idealised kinematic manifold, so the refinement stage needs no candidate pool.

• A differentiable objective that puts geometric coverage, attenuation, and reconstruction information into one gradient step. The absorption term has an analytic gradient and needs no step-size calibration.

• A matched comparison against the discrete state of the art. On the free sphere our selector reaches its image quality without any object-specific precompute, so it selects in seconds where the discrete pipeline needs minutes and its margin grows with the reconstruction grid.

• A measured robot-based study. Pose sets planned from a prescan are acquired on a custom bench and reconstructed in a common reference frame.

All proofs, the algorithm listing, and extended tables and analyses appear in the supplementary material. The code is available on GitHub.

## II. RELATED WORK

Sparse-view CT reconstruction spans analytical Feldkamp– Davis–Kress (FDK) and iterative regularisation [5], [7], learned and known-operator methods [8]–[10], [18], and differentiable CT operators as reusable building blocks [19]. These methods ask how to reconstruct incomplete data; we address the complementary question of which projections should be acquired.

Adaptive acquisition design is well established, and almost all of it selects from a discrete pool. Batenburg et al. [20], Haque et al. [21], and Lin et al. [22] select angles from information, spectral, or CAD-edge criteria; Burger et al. [23] formulate sequential X-ray view selection as Bayesian A- or D-optimal design. Robot and C-arm studies optimise object-specific trajectories and out-of-plane adjustments [24]– [28], characterise non-circular orbits [29], evaluate families of spherical trajectories under a fixed pose budget [30], and plan orbits scored on a prior reconstruction under strong kinematic constraints [31], [32]. Learned selectors choose views sequentially, through reinforcement-learning policies [33], [34] or a recurrent network [35]; they amortise selection over a task distribution but have so far required training per application and acquisition manifold.

Continuous and differentiable geometry optimisation has also been studied before. Stayman et al. [3] use the populationbased, derivative-free covariance matrix adaptation evolution strategy (CMA-ES) of Hansen and Ostermeier [36] to maximise a task-driven detectability index over continuous trajectory parameters expressed as periodic or B-spline coefficients;

Gang et al. [37] optimise tube-current and reconstructionkernel modulation for a fixed orbit, sweeping orbital tilt over a small discrete set. Thies et al. [38], [39] study gradient-based geometry correction, and Fathi and van Leeuwen [40] optimise a fixed-budget laminography design. Our own differentiableranking work [41] makes discrete projection selection differentiable. However, its search variables are inclusion scores over a fixed candidate pool, and its task-based objective must be learned per application. Linde et al. [17] search robot-CT trajectories by differential evolution over three continuous orbit parameters on a Radon-space completeness criterion from a prior object model; companion work addresses space-restricted environments and few-view budgets [42], [43]. In closely related work first presented in mid-2025, Yoo et al. [44] combine a sampled Tuy-completeness score with hard rejection of metal-artefact-inducing directions on a robotic CBCT system, again selecting from discrete candidate positions. In conebeam CT, however, no prior method optimises the source poses themselves as continuous variables of a gradient-based, fixedbudget selection. Even where the search space is continuous, Stayman et al. [3] and Linde et al. [17] optimise a handful of global orbit-shape parameters through a finite population of candidate solutions, whereas our optimiser updates one point per source through a smooth gradient and never instantiates any candidate set.

The closest reconstruction-driven baseline is the View Covariance Loss (VCL) of Lin et al. [16]. Its VCLS swap search minimises a closed-form filtered-backprojection error surrogate over a fixed candidate set, and voxel subsampling makes that search tractable. Discrete coverage work supplies a complementary geometric baseline, grounding selection in Tuy’s completeness condition [11] through sampled completeness criteria and greedy selection [1], [12]–[15], [45]. In particular, our prior work [15] shows that the decision version of the discrete problem is NP-complete and gives a near-optimal submodular greedy method. Clackdoyle and Noo [46] quantify cone-beam incompleteness on a theoretically grounded graded scale, but as an evaluation measure that is not differentiable in the source positions. We retain both the reconstructiondriven and the geometric perspective, and make the source positions themselves differentiable variables. Reconstruction information, soft-Tuy coverage, and attenuation are combined in one objective, without rebuilding candidate pools or objectspecific precomputations for new acquisition manifolds.

## III. METHODS

Selecting X-ray source positions for cone-beam CT is usually posed as a combinatorial search over a finite candidate pool. We instead treat the source positions as continuous variables on a kinematic manifold and express coverage, absorption, and view information as differentiable functions of these positions. View selection then becomes a gradientbased optimisation problem, as illustrated in Fig. 1. Sec. III-A introduces the differentiable kinematic parametrisation. The objective combines a smooth soft-Tuy coverage surrogate with a closed-form tangential gradient in Secs. III-B to III-C, an analytic bundle absorption term in Sec. III-D, and a continuous extension of the VCL score of Lin et al. in Sec. III-E. The combined objective is maximised with Adam under a cosine schedule and manifold-aware refinement, as detailed in Sec. III-F; the complete algorithm is provided in the supplementary material.

## A. Problem Formulation

We build on the discrete framework of [15], which maximises a soft Tuy-coverage surrogate over a finite candidate set. Our change is the search space, optimising the source positions themselves. Let $V = \{ s _ { 1 } , \ldots , s _ { k } \}$ with $s _ { i } \in \mathbb { R } ^ { 3 }$ be the current view set. Further, let $x _ { \mathrm { r o i } } \in \mathbb { R } ^ { 3 }$ denote the regionof-interest (ROI) centre, $r _ { \mathrm { s i d } } ~ > ~ 0$ the source-to-isocentre distance, and $\{ n _ { j } \} _ { j = 1 } ^ { z }$ with $n _ { j } \in \mathbb { S } ^ { 2 }$ a Fibonacci lattice of Radon-plane normals on the unit sphere [1], with z samples in total. The $n _ { j }$ are the normals of the Radon planes through the ROI centre; since all these planes share that anchor, only the orientation remains free. Each pose is described by a kinematic parameter vector $\theta _ { i } \in \mathbb { R } ^ { p }$ , where $p$ depends on the chosen acquisition manifold. Source positions are parametrised through a differentiable kinematic map

$$
s _ { i } = \Gamma ( \theta _ { i } ) ,\tag{1}
$$

where Γ encodes scanner kinematics. The manifolds used here share the map

$$
\Gamma _ { 2 \mathrm { a x } } ( \theta , \varphi ) = r _ { \mathrm { s i d } } \left[ - \cos \varphi \sin \theta \quad \cos \varphi \cos \theta \quad \sin \varphi \right] ^ { \top } .\tag{2}
$$

The single-axis circle is $\Gamma _ { \mathrm { c i r c } } ( \theta ) = \Gamma _ { 2 \mathrm { a x } } ( \theta , 0 )$ with θ periodic modulo 2π. The unconstrained two-axis gantry uses the same periodic map; the canonical reporting ranges are $\theta \in [ - \pi , \pi )$ and $\varphi \in [ - \pi / 2 , \pi / 2 ]$ . An envelope-restricted axis is expressed by a chart that represents the envelope itself rather than by restricting an unbounded one. For an axis admitted on $[ a , b ]$ we optimise a free coordinate $\tilde { \alpha } \in \mathbb { R }$ and read the angle off

$$
\begin{array} { r } { \alpha ( \tilde { \alpha } ) = \frac { 1 } { 2 } ( a + b ) + \frac { 1 } { 2 } ( b - a ) \operatorname { t a n h } \tilde { \alpha } , } \end{array}
$$

so every iterate is admissible by construction, the gradient is everywhere tangent to the reachable set, and no feasibility repair enters the update. The limited C-arm uses this chart on both axes, with $\theta \in [ - 1 1 0 ^ { \circ } , 1 1 0 ^ { \circ } ]$ and $\varphi \in [ - 4 5 ^ { \circ } , 4 5 ^ { \circ } ] ;$ the elevation band of our measured bench uses it on $\varphi$ alone and keeps θ periodic, since a bounded chart would break the wrap-around of a full revolution. Throughout, the isocentre coincides with the ROI centre $x _ { \mathrm { r o i } } .$ , which the two-axis map places at the origin. The free sphere is $\{ s \mid \| s - x _ { \mathrm { r o i } } \| = r _ { \mathrm { s i d } } \}$ its ambient-coordinate update is retracted by $\Pi _ { \mathbb { S } ^ { 2 } } ( s ) = x _ { \mathrm { r o i } } +$ $r _ { \mathrm { s i d } } ( s - x _ { \mathrm { r o i } } ) / \lVert s - x _ { \mathrm { r o i } } \rVert$ . Each source pose induces the standard cone-beam detector geometry, an untilted flat detector at fixed source-to-detector distance, centred on the principal ray through $x _ { \mathrm { r o i } }$ and oriented by $u _ { i } \propto e _ { z } \times \left( s _ { i } - x _ { \mathrm { r o i } } \right)$ , with $u _ { i } = e _ { x }$ at the two poles where the cross product vanishes. These detector parameters remain fixed, so the optimisation acts only on the source positions. The continuous view-selection problem is then

$$
\operatorname* { m a x } _ { \theta _ { 1 } , \dots , \theta _ { k } } \ { \mathcal { L } } ( \Gamma ( \theta _ { 1 } ) , \dots , \Gamma ( \theta _ { k } ) ) ,\tag{3}
$$

with the objective $\mathcal { L }$ built term by term in the following subsections.

## B. Differentiable Soft Coverage

For source $s _ { i } \in V$ and ROI centre $x _ { \mathrm { r o i } }$ , define the displacement, distance, and unit viewing direction by

$$
r _ { i } = s _ { i } - x _ { \mathrm { { r o i } } } , \qquad \rho _ { i } = \left\| r _ { i } \right\| , \qquad d _ { i } = r _ { i } / \rho _ { i } .\tag{4}
$$

Here, $d _ { i }$ is the unit viewing direction from the ROI centre to source $s _ { i }$ . For each Radon-plane normal $n _ { j }$ , the quantity $g _ { i j } = n _ { j } ^ { \top } d _ { i }$ measures how close this viewing direction is to the corresponding plane; exact alignment gives $g _ { i j } = 0$ . The hinge-based coverage score of [15] is piecewise linear, not differentiable at the tolerance boundary, and carries no gradient information beyond it. Therefore, we replace it by the smooth Gaussian kernel

$$
\psi _ { \sigma } ( g ) = \exp \left( - g ^ { 2 } / 2 \sigma ^ { 2 } \right) ,\tag{5}
$$

with Gaussian bandwidth σ. We match it to a target tolerance $\tau = \sin ( \Delta \gamma )$ by imposing $\psi _ { \sigma } ( \tau ) = \epsilon ,$ which yields

$$
\sigma = \tau / \sqrt { 2 \log ( 1 / \epsilon ) } .\tag{6}
$$

Here, $\Delta \gamma \in \left( 0 , \pi / 2 \right)$ sets the angular tolerance and $\epsilon \in ( 0 , 1 )$ its corresponding score, so that $\sigma > 0$ is well defined. We then accumulate the contributions of all views in V along direction $n _ { j }$ as $\begin{array} { r } { \Sigma _ { j } ( V ) = \sum _ { i = 1 } ^ { k } { \psi _ { \sigma } ( g _ { i j } ) } } \end{array}$ . Applying the smooth saturation function $\phi ( x ) = 1 - e ^ { - x }$ gives the geometry-only coverage surrogate

$$
\tilde { C } _ { \mathrm { g e o } } ( V ) = \frac { 1 } { z } \sum _ { j = 1 } ^ { z } \phi ( \Sigma _ { j } ( V ) ) \in [ 0 , 1 ) .\tag{7}
$$

${ \tilde { C } } _ { \mathrm { g e o } }$ averages the saturated coverage over the z sampled Radon-plane normals and is the smooth analogue of the hingebased coverage of [15]. Because every view contributes nonnegatively to each $\Sigma _ { j }$ and the saturation $\phi$ is increasing and concave, $\tilde { C } _ { \mathrm { g e o } }$ is monotone and submodular as a set function in the selected sources [47].

## C. Closed-Form Tangential Coverage Gradient

The coverage term depends on the viewing direction $d _ { i } ,$ not directly on the source-to-ROI distance $\rho _ { i }$ . The first step is therefore to understand how $d _ { i }$ changes when the source position moves. Only tangential motion changes the viewing direction; radial motion changes the distance to the ROI but not the direction itself. This geometric separation is captured by the Jacobian of the unit direction with respect to the source position.

Proposition 1 (Tangential Jacobian). The derivative of the unit viewing direction with respect to the source position is the orthogonal projection onto the tangent plane at $d _ { i } ,$ , scaled by the inverse source-to-ROI distance, and is given by

$$
\partial d _ { i } / \partial s _ { i } = \rho _ { i } ^ { - 1 } ( I _ { 3 } - d _ { i } d _ { i } ^ { \top } ) = \rho _ { i } ^ { - 1 } P _ { d _ { i } } ^ { \bot } ,
$$

where $P _ { d _ { i } } ^ { \perp } = I _ { 3 } - d _ { i } d _ { i } ^ { \top }$ is the orthogonal projector onto the tangent plane of $\mathbb { S } ^ { 2 }$ at $d _ { i }$

The proof is provided in the supplementary material.

![](images/eb2f2965bfcbbfe97e00e21fe45e44b1b7f9c60c63dd5f4cc3ce957ee8c5b3f5.jpg)  
Fig. 1. Overview of the gradient-based trajectory-optimisation pipeline. (a) Attenuation model $\pmb { \mu } .$ (b) The viewing sphere carries two families of symbols. Red dots are the source positions $s _ { i } ,$ and blue arrows from the centre are the sampled Radon normals $n _ { j } ,$ , one for each plane whose coverage is scored. Thick red arrows are the coverage gradient, which is tangential to the sphere by Proposition 2. Every source carries such a gradient, and three are drawn so that the panel stays readable. (c) The objective combines the soft-Tuy coverage surrogate, the attenuation-aware bundle penalty, and the continuous VCL information score; Adam updates the kinematic parameters on the manifold. (d) Two outputs from separate runs. Above, red dots are the optimised source positions $V ^ { \star }$ of a coverage-only free-sphere run at k=24, thin red lines are their rays to the region of interest, and the black cross marks its centre. Below, SART reconstructions at k=80 for the two-axis gantry and the free sphere, named in the image.

Applying this Jacobian to $g _ { i j } = n _ { j } ^ { \top } d _ { i }$ gives the gradient of the Radon-plane mismatch with respect to the source position,

$$
\nabla _ { s _ { i } } g _ { i j } = \frac { \partial { d _ { i } } ^ { \top } } { \partial s _ { i } } n _ { j } = \frac { 1 } { \rho _ { i } } \big ( n _ { j } - g _ { i j } d _ { i } \big ) ,\tag{8}
$$

where the second equality uses $P _ { d _ { i } } ^ { \perp } n _ { j } = n _ { j } - g _ { i j } d _ { i }$ . Differentiating the Gaussian score $\psi _ { \sigma }$ then gives the single-direction coverage gradient

$$
\nabla _ { s _ { i } } \psi _ { \sigma } ( g _ { i j } ) = - \frac { g _ { i j } } { \sigma ^ { 2 } } \psi _ { \sigma } ( g _ { i j } ) \frac { 1 } { \rho _ { i } } \big ( n _ { j } - g _ { i j } d _ { i } \big ) .\tag{9}
$$

The kernel derivative $| \psi _ { \sigma } ^ { \prime } ( g ) | = | g | \psi _ { \sigma } ( g ) / \sigma ^ { 2 }$ vanishes at perfect coverage and peaks at |g| = σ. Applying $\phi ^ { \prime } ( \Sigma _ { j } ) = e ^ { - \Sigma _ { j } }$ yields the full coverage gradient

$$
\begin{array} { l } { { \nabla _ { s _ { i } } \tilde { C } _ { \mathrm { g e o } } = \displaystyle \frac { 1 } { z } \sum _ { j = 1 } ^ { z } e ^ { - \Sigma _ { j } } \ \nabla _ { s _ { i } } \psi _ { \sigma } ( g _ { i j } ) } } \\ { { \mathrm { } = - \displaystyle \frac { 1 } { z \rho _ { i } } \sum _ { j = 1 } ^ { z } \frac { g _ { i j } \psi _ { \sigma } ( g _ { i j } ) e ^ { - \Sigma _ { j } } } { \sigma ^ { 2 } } \ ( n _ { j } - g _ { i j } d _ { i } ) . } } \end{array}\tag{10}
$$

The saturation factor $e ^ { - \Sigma _ { j } }$ discourages further investment in well-covered directions, the scalar factor $- ( g _ { i j } / \sigma ^ { 2 } ) \psi _ { \sigma } ( g _ { i j } )$ weights each direction by the kernel sensitivity, and the direction factor $n _ { j } - g _ { i j } d _ { i } = P _ { d _ { i } } ^ { \perp } n _ { j }$ lies in the tangent plane at $d _ { i }$ , so the full coverage gradient remains tangential to the viewing sphere.

Proposition 2 (Tangential Coverage Gradient). The coverage gradient with respect to the source position is orthogonal to the viewing direction, so that

$$
\nabla _ { s _ { i } } \tilde { C } _ { \mathrm { g e o } } \perp d _ { i }
$$

for every source position $s _ { i }$ with $\rho _ { i } > 0 \quad$

The proof is provided in the supplementary material.

For a kinematic parametrisation $s _ { i } = \Gamma ( \theta _ { i } )$ , the parameterspace gradient is

$$
\nabla _ { \theta _ { i } } \tilde { C } _ { \mathrm { g e o } } = \left( { \partial \Gamma ( \theta _ { i } ) } / { \partial \theta _ { i } } \right) ^ { \top } \nabla _ { s _ { i } } \tilde { C } _ { \mathrm { g e o } } ,\tag{11}
$$

which maps the source-space gradient to the kinematic coordinates by the chain rule.

The centred surrogate of Eq. (7) is the form used in all virtual studies. The measured study of Sec. V-D averages the same construction over sampled ROI points instead of a single centre; at the 5 mm ROI size and $\Delta \gamma = 1 5 ^ { \circ }$ smoothing used here, the two forms agree in attained coverage to within $1 0 ^ { - 3 }$

## D. Analytic Bundle Absorption Penalty

Geometry-only coverage treats every direction as equally informative, although long paths through dense material transmit fewer photons even where coverage is good. We therefore add an object-aware penalty, a forward projection along a small bundle of rays from each source to a target patch around the ROI. Unlike a general projection operator [19], which is differentiated with respect to the volume, the rays are parametrised directly in $s _ { i } ,$ so the source gradient is available in closed form at the cost of a few tens of rays per view.

To place a square patch of bundle targets around the ROI, we construct for source $s _ { i }$ an orthonormal frame $( \hat { u } _ { i } , \hat { v } _ { i } , d _ { i } )$ with $( \hat { u } _ { i } , \hat { v } _ { i } )$ spanning the tangent plane $d _ { i } ^ { \perp }$ , using a crossproduct construction with a reference axis $\hat { r } _ { i }$

$$
\boldsymbol { \hat { u } _ { i } } = \frac { d _ { i } \times \boldsymbol { \hat { r } _ { i } } } { \lVert d _ { i } \times \boldsymbol { \hat { r } _ { i } } \rVert } , \qquad \boldsymbol { \hat { v } _ { i } } = d _ { i } \times \boldsymbol { \hat { u } _ { i } } ,\tag{12}
$$

where $\hat { r } _ { i } = \hat { z } \mathrm { ~ i f ~ } | d _ { i } ^ { \top } \hat { z } | < 0 . 9 5$ and $\hat { r } _ { i } ~ = ~ \hat { y }$ otherwise; the fallback avoids the polar degeneracy of the frame construction. Let $\{ ( g _ { u } ^ { ( r ) } , g _ { v } ^ { ( r ) } ) \} _ { r = 1 } ^ { | \mathcal { \bar { R } } | }$ be a uniform $n _ { u } \times n _ { \tau }$ grid on $[ - 1 , 1 ] ^ { 2 }$

and let $R _ { \mathrm { r o i } }$ denote the half-width of the square bundle target patch in the plane $d _ { i } ^ { \perp }$ . The corresponding target points are

$$
\begin{array} { r } { c _ { i } ^ { ( r ) } = x _ { \mathrm { r o i } } + R _ { \mathrm { r o i } } \left( g _ { u } ^ { ( r ) } \hat { u } _ { i } + g _ { v } ^ { ( r ) } \hat { v } _ { i } \right) , \quad r = 1 , \ldots , | \mathcal { R } | , } \end{array}\tag{13}
$$

and the line integral along ray $r ,$ parametrised by $p _ { i } ^ { ( r ) } ( t ) =$ $s _ { i } + t \left( c _ { i } ^ { ( r ) } - s _ { i } \right)$ with $t \in ( 0 , 1 ]$ , is

$$
\tau _ { i } ^ { ( r ) } = \Big \| c _ { i } ^ { ( r ) } - s _ { i } \Big \| \int _ { 0 } ^ { 1 } \mu \Big ( p _ { i } ^ { ( r ) } ( t ) \Big ) ~ \mathrm { d } { t } ,\tag{14}
$$

where $\mu$ is the attenuation field of the reference volume available at planning time. The bundle-mean line integral

$$
\bar { \tau } _ { i } = \frac { 1 } { | \mathcal { R } | } \sum _ { r = 1 } ^ { | \mathcal { R } | } \tau _ { i } ^ { \left( r \right) }\tag{15}
$$

yields one dimensionless optical-depth score per source. Eq. (14) is evaluated with the midpoint rule on the subsegment $[ t _ { \mathrm { i n } } ^ { ( r ) } , t _ { \mathrm { o u t } } ^ { ( r ) } ] \subseteq [ 0 , 1 ]$ of each ray inside the reference volume, obtained by intersecting the ray with the volume bounding box, at the $N _ { t }$ regularly spaced samples $t _ { n } = t _ { \mathrm { i n } } ^ { ( r ) } +$ $( n - \textstyle \frac { 1 } { 2 } ) \overline { { \big ( } } t _ { \mathrm { o u t } } ^ { ( r ) } - t _ { \mathrm { i n } } ^ { ( r ) } \big ) / N _ { t }$ . The attenuation values are obtained by trilinear interpolation of $\mu .$ Interpolation is what makes the quadrature differentiable with respect to the source position, since the sample points move with the source and a nearestneighbour lookup would be piecewise constant with vanishing gradient almost everywhere. The resulting approximation is

$$
\tau _ { i } ^ { ( r ) } \approx \frac { \Big \lVert c _ { i } ^ { ( r ) } - s _ { i } \Big \rVert \left( t _ { \mathrm { o u t } } ^ { ( r ) } - t _ { \mathrm { i n } } ^ { ( r ) } \right) } { N _ { t } } \sum _ { n = 1 } ^ { N _ { t } } \mu \Big ( p _ { i } ^ { ( r ) } ( t _ { n } ) \Big ) .\tag{16}
$$

Restricting the quadrature to the ray–volume intersection loses nothing physically, because $\mu$ vanishes outside the reference volume, and it ensures every sample carries attenuation information rather than reading clamped border values. A preregistered convergence study against a 4096-sample reference, reported in the supplementary material, fixes $N _ { t } = 2 5 6 ;$ at this value the integrals, their source gradients, and the induced view rankings are converged. For this trilinear interpolation, let $q ( p ) = ( p - p _ { 0 } ) / \Delta _ { v }$ denote the volume-grid coordinates of a world point $p ,$ where $p _ { 0 }$ is the volume origin and $\Delta _ { v }$ the voxel size. We then split q into the floor index $\lfloor q \rfloor \in \mathbb { Z } ^ { 3 }$ and the fractional offsets $\mathbf { \bar { f } } ( p ) = \mathbf { \bar { q } } - \lfloor \mathbf { \bar { q } } \rfloor \in [ 0 , 1 ) ^ { 3 }$ . The floor index identifies the voxel cell that contains $p ,$ and the fractional offsets locate $p$ inside that cell. The interpolated value

$$
\begin{array} { l } { \displaystyle \mu ( \boldsymbol { p } ) = \sum _ { c \in \{ 0 , 1 \} ^ { 3 } } \mu ( v _ { c } ) w _ { c } ( \boldsymbol { p } ) , } \\ { \displaystyle w _ { c } ( \boldsymbol { p } ) = \prod _ { \xi } ( c _ { \xi } f _ { \xi } + ( 1 - c _ { \xi } ) ( 1 - f _ { \xi } ) ) , } \end{array}\tag{17}
$$

is differentiable within each voxel cell with respect to the fractional offsets $f ( \boldsymbol p )$ . Here, $v _ { c } = \lfloor q \rfloor + c$ are the eight cell corners and ξ indexes the axes. Differentiating the weights $w _ { c } ( p )$ therefore gives the spatial gradient

$$
\frac { \partial \mu ( p ) } { \partial p } = \Delta _ { v } ^ { - 1 } \sum _ { c } \mu ( v _ { c } ) \frac { \partial w _ { c } } { \partial f } ,\tag{18}
$$

where each partial derivative follows by differentiating one linear factor of $w _ { c }$ . Coordinates are clipped independently to the closed volume bounds before interpolation. This amounts to a constant boundary extension, where an outside-volume sample takes the nearest boundary value and its derivative in a clipped coordinate is zero. Within the volume, floor indices remain constant under differentiation and enter only through the gathered corner values $\mu ( v _ { c } )$

The bundle mean $\bar { \tau } _ { i }$ is differentiated end-to-end with respect to $s _ { i }$ . The hard reference-axis branch in Eq. (12) makes the construction piecewise differentiable, with a possible discontinuity at $| d _ { i } ^ { \top } \hat { z } | = 0 . 9 5$

We use the raw attenuation score directly instead of wrapping it in a bounded gate, since a gate such as $\nu _ { i } = \exp ( - \alpha \bar { \tau } _ { i } )$ saturates on highly absorbing directions and its gradient vanishes there as well. We define the bundle penalty as

$$
\mathcal { L } _ { \mathrm { b u n d l e } } ( V ) = - \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \bar { \tau } _ { i } .\tag{19}
$$

A natural alternative is to finite-difference the source coordinates through an existing cone-beam forward projector, and Supplementary ?? reports that comparison in matched form with a step size of $\varepsilon = 0 . 5$ mm selected by an explicit sweep. This mode acts on the absorption objective alone and is separate from the VCL geometry difference of Sec. III-E. The analytic bundle instead provides the gradient directly through the attenuation field $\mu ,$ without differentiating through a full projector.

## E. Continuous View Covariance Loss

The VCL of Lin et al. [16] is a sample-based approximation to the normalised mean-squared error of a linear filteredbackprojection (FBP) reconstruction at the candidate view set.

Let $\mathbf { A } _ { s } ~ \in ~ \mathbb { R } ^ { m \times N }$ be the cone-beam forward projector at source position s, H a one-dimensional ramp filter applied along each detector row, and $A _ { s } ^ { \top }$ its matched-footprint adjoint. The per-view FBP operator is

$$
\pmb { T _ { s } } = \pmb { A } _ { s } ^ { \top } \pmb { H } \pmb { A _ { s } } .\tag{20}
$$

For a fixed voxel set $\Omega \subset \{ 1 , \ldots , N \}$ of size $n = \left\lceil r _ { 1 } N \right\rceil$ drawn uniformly at random once per run with sampling fraction $r _ { 1 }$ , let $\dot { S _ { \Omega } } \dot { \in } \{ 0 , 1 \} ^ { n \times N }$ be the corresponding sampling operator and ${ \pmb x } _ { \Omega } = { \pmb S } _ { \Omega } { \pmb x }$ . We form the normalised per-view response

$$
\begin{array} { r l } & { \displaystyle { q _ { i } ( s _ { i } ) = \frac { S _ { \Omega } T _ { s _ { i } } x } { \| S _ { \Omega } T _ { s _ { i } } x \| + \varepsilon _ { \mathrm { n } } } , } } \\ & { \displaystyle { Q ( V ) = \left[ \begin{array} { l } { \pmb { q } _ { 1 } ( s _ { 1 } ) ^ { \top } } \\ { \vdots } \\ { \pmb { q } _ { k } ( s _ { k } ) ^ { \top } } \end{array} \right] \in \mathbb { R } ^ { k \times n } , } } \end{array}\tag{21}
$$

with $\varepsilon _ { \mathrm { n } } = 1 0 ^ { - 9 }$ . The score requires $\| \pmb { x } _ { \Omega } \| > 0$ , and the implementation rejects a zero-norm sample. The regularised view– view and view–reference correlations used by the optimiser are

$$
\begin{array} { r } { \begin{array} { l } { { \pmb R } ( V ) = { \pmb Q } ( V ) { \pmb Q } ( V ) ^ { \top } + \lambda _ { \mathrm { R } } { \pmb I } _ { k } , \lambda _ { \mathrm { R } } = 1 0 ^ { - 3 } , } \\ { \gamma ( V ) = { \pmb Q } ( V ) \frac { \pmb x _ { \Omega } } { \| \pmb x _ { \Omega } \| } . } \end{array} } \end{array}\tag{22}
$$

The continuous information score is

$$
I _ { \mathrm { v c l } } ( V ) = \gamma ( V ) ^ { \top } R ( V ) ^ { - 1 } \gamma ( V ) \ \in \ [ 0 , 1 ] .\tag{23}
$$

The bounds follow from $Q Q ^ { \top } \preceq R .$ In the unregularised limit, this is the squared cosine between the sampled reference and its projection onto the span of the per-view responses. Row normalisation leaves that ideal score invariant but fixes the scale of the ridge term in Eq. (22). The discrete VCLS of [16] maximises this score over a candidate pool by swap search; here, $I _ { \mathrm { v c l } }$ enters the continuous objective.

The matched-footprint backprojector exposed by diffct [19] provides a vector–Jacobian product (VJP) with respect to the projection data but no complete geometry VJP. Propagating only through the forward projection would omit the first term in

$$
\mathrm { D } _ { s } \big ( \boldsymbol { A } _ { s } ^ { \top } \boldsymbol { H } \boldsymbol { A } _ { s } \pmb { x } \big ) = \mathrm { D } _ { s } ( \boldsymbol { A } _ { s } ^ { \top } ) \boldsymbol { H } \boldsymbol { A } _ { s } \pmb { x } + \boldsymbol { A } _ { s } ^ { \top } \boldsymbol { H } \mathrm { D } _ { s } ( \boldsymbol { A } _ { s } ) \pmb { x } .\tag{24}
$$

We therefore estimate the geometry derivative of the complete normalised response in Eq. (21) by a central difference, which re-evaluates the entire pipeline at perturbed source positions and thus captures both terms of Eq. (24),

$$
\widehat { \frac { \partial \pmb q _ { i } } { \partial s _ { i , a } } } = \frac { \pmb q _ { i } ( s _ { i } + \varepsilon _ { \mathrm { v c l } } \pmb e _ { a } ) - \pmb q _ { i } ( s _ { i } - \varepsilon _ { \mathrm { v c l } } \pmb e _ { a } ) } { 2 \varepsilon _ { \mathrm { v c l } } } ,\tag{25}
$$

where $\varepsilon _ { \mathrm { v c l } } = 0 . 5$ mm. Each perturbation rebuilds the sourcedependent detector frame and re-evaluates the complete response pipeline. Since response row i depends only on source i, every view can be perturbed along the same coordinate simultaneously. The VJP therefore uses six additional basis evaluations per Adam step, independent of k.

The outer VCL derivative remains analytical, using a custom vector–Jacobian product for $f ( \gamma , R ) = \gamma ^ { \top } R ^ { - 1 } \gamma$ instead of autograd through the explicit inverse.

Proposition 3 (VJP of the VCL Quadratic Form). For symmetric positive-definite R, as in $E q .$ (22), let

$$
f ( \gamma , { \pmb R } ) = \gamma ^ { \top } { \pmb R } ^ { - 1 } \gamma , \qquad { \pmb u } = { \pmb R } ^ { - 1 } \gamma .
$$

Then

$$
\frac { \partial f } { \partial \boldsymbol { \gamma } } = 2 \boldsymbol { u } , \qquad \frac { \partial f } { \partial \boldsymbol { R } } = - \boldsymbol { u } \boldsymbol { u } ^ { \intercal } .\tag{26}
$$

The proof follows from the differential identity for $\pmb { R } ^ { - 1 }$ and is provided in the supplementary material.

The outer reverse pass therefore requires a single linear solve $R u = \gamma$ and one outer product ${ \mathbf { } } { \mathbf { } } { \mathbf { } } u { \mathbf { } } u ^ { \top }$ , with no $k \times k$ inverse ever materialised explicitly. The resulting cotangent with respect to the rows of Q is contracted with Eq. (25) to obtain the source update.

## F. Composite Objective and Manifold Refinement

The terms combine additively into the ascent objective

$$
{ \mathcal { L } } ( V ) = \lambda _ { \mathrm { c o v } } { \tilde { C } } _ { \mathrm { g e o } } ( V ) + \lambda _ { \mathrm { v c l } } I _ { \mathrm { v c l } } ( V ) + \lambda _ { \mathrm { b u n d l e } } { \mathcal { L } } _ { \mathrm { b u n d l e } } ( V ) ,\tag{27}
$$

with the continuous VCL score $I _ { \mathrm { v c l } }$ of Eq. (23), which carries no noise model. Attenuation-awareness, a monochromatic photon-starvation proxy, enters through the explicit absorption penalty $\lambda _ { \mathrm { b u n d l e } } \mathcal { L } _ { \mathrm { b u n d l e } } .$

Eq. (27) covers all continuous variants. Inactive terms are switched off by setting their weights to zero, and the weights are fixed during a given optimisation run.

A discrete swap search can score the same weighted sum, but every candidate swap then pays for a full evaluation of each active term.

We maximise L by differentiating through the chosen kinematic map $s _ { i } = \Gamma ( \theta _ { i } )$ and updating the parameter vectors $\theta _ { i }$ with Adam [48], using a cosine learning-rate schedule with patience-based early stopping on the best objective seen so far; the complete loop is listed in the supplementary material. The concrete settings are given in Sec. IV-D.

The Adam learning rate is a displacement in the units of the chart, so mixed-chart implementations advance poses at different angular rates per step. Scaling the Cartesian chart by the sphere radius removes the discrepancy, and we report all comparisons under that scaling. Feasibility on the circle and the C-arm is enforced by Γ itself; on the free sphere each source is re-normalised to the viewing sphere after each step by the closest-point projection $\Pi _ { \mathbb { S } ^ { 2 } }$

The Adam loop can be initialised from a converged discrete VCLS solution or from a discrete soft-coverage greedy solution, and the two starts place the optimiser in different basins of the objective landscape. All reported results use the greedy start.

## IV. EXPERIMENTAL SETUP

We compare continuous and discrete selection at matched view budgets on two virtual benchmarks, with and without photon noise, and ablate the derivative estimator and optimiser. Further studies test kinematic constraints, sparse-prescan planning, selection cost, and physical acquisition on a robot bench.

## A. Virtual Benchmarks and Test Objects

The Defrise flange (Fig. 2) tests missing elevation: planar lack-of-fusion (LoF) voids in its off-plane disks are invisible to an equatorial orbit, while tilted cracks and central pores serve as reachable controls. Steel inserts create attenuation shadows. Table I and Fig. 3 gives acquisition specifications and directional absorption; Supplementary ?? provides the full phantom design.

The Oak Ridge National Laboratory (ORNL) fuel nozzle [49] is a near-axisymmetric dense-metal counterpart. Its FDK volume provides the attenuation reference; all virtual comparisons forward-project the reference volumes with matched diffct operators [19] and reconstruct using 15 simultaneous algebraic reconstruction technique (SART) iterations [50], relaxation 0.9, and non-negativity. Supplementary ?? gives implementation details.

The photon-noise model and its evaluation on both objects are reported with the complete results in Supplementary ??.

## B. Measured Robot-Based CT Study

The robot CT bench (Fig. 4) scans a digital camera over 360<sup>◦</sup> azimuth and $\pm 3 0 ^ { \circ }$ elevation. A 120-view circular prescan is the planning prior; a separate 1200-view circular FDK reference is withheld until evaluation.

![](images/f9fc5cc163b1e708cf090852f3cb8b5a2a23562e96f5a78d8b843113dc0303b8.jpg)

TABLE I  
CONFIGURATION OF THE THREE TEST OBJECTS. THE FLANGE IS SIMULATED UNDER A MONOCHROMATIC MODEL, AND THE CAMERA DETECTOR IS MEAN-POOLED BY A FACTOR OF FOUR BEFORE RECONSTRUCTION.
<table><tr><td>Specification</td><td>Defrise flange</td><td>Fuel nozzle</td><td>Digital camera</td></tr><tr><td>Material</td><td>aluminium, steel</td><td>316L stainless steel</td><td>plastics, circuit boards, metal</td></tr><tr><td>Max. photon energy (keV)</td><td>monochromatic</td><td>180</td><td>175</td></tr><tr><td>Detector dimensions (pixels)</td><td>256 × 256</td><td>1024 × 1024</td><td>3072× 3072</td></tr><tr><td>Pixel pitch in detector (mm)</td><td>0.5</td><td>0.2</td><td>0.139</td></tr><tr><td>Source-to-detector distance (mm)</td><td>900</td><td>808.508</td><td>1995.6</td></tr><tr><td>Source-to-isocentre distance (mm)</td><td>500</td><td>243.307</td><td>997.1</td></tr><tr><td>Reconstruction grid</td><td>384³</td><td>5123</td><td>7683</td></tr><tr><td>Voxel pitch in object (mm)</td><td>0.3</td><td>0.118</td><td>0.278</td></tr><tr><td>Reference volume</td><td>phantom ground truth</td><td>FDK of 1050 views</td><td>FDK of 1200 views</td></tr></table>

![](images/4313fb2c6c6526b8c0bbe89c40d4facaf8f3c307ebc12337cc8cf44079cb038b.jpg)  
Fig. 2. The Defrise flange phantom. (a) Three-dimensional view of the aluminium body in translucent blue with two press-fit steel inserts in red. (b) Central sagittal slice with the two LoF stacks far off the source plane and the two tilted control cracks in the column. (c) Axial slice through the upper flange disk with the LoF layer, the steel insert, and the pore cluster in the attenuation shadow of the insert.

![](images/85d69f453388f050c825f49061af4b566238b8eb4207ed218ba963b6c748404e.jpg)

![](images/56a1fa8d3f6469dcc3ee31651f85e0427153a90c1ea849132131ec506c8f491e.jpg)  
Fig. 3. Directional absorption of the Defrise flange, shown as the bundle path integral τ¯ towards three target points as a function of the source azimuth and elevation on the acquisition sphere. Dark is transparent. Towards the isocentre the transparent region is the equatorial band, since only the column lies in the way there. Towards an off-plane defect the minimum moves into the hemisphere of that defect and leaves a narrow equatorial notch. We aim the term at the isocentre because that choice needs no assumption about which defect matters.

Coverage–bundle and full-composite pose sets are acquired at $k \in \{ 1 0 0 , 4 0 0 \}$ . Baselines are equidistant circular-reference subsets and uniform-on-band sets formed by matching Fibonacci targets to distinct views pooled from the four band acquisitions (mean mismatch 3–3.5<sup>◦</sup>). All sparse arms use totalvariation (TV) regularised adaptive steepest descent projection onto convex sets (ASD-POCS) with the TV weight chosen by data residual, without reference access. The planned arms use the historical 32-sample bundle rule of Supplementary ??. Reconstructions are intensity-matched to the reference before scoring. Supplementary ?? gives acquisition and reconstruction details.

![](images/42ffe3b0e194e4c6061ad8be8eb4cc0e844d42a09253348f0e750e25eb078e1a.jpg)  
(a)

![](images/0acd96b57f6600ef0db3ff047cbf6abee8aea6a6f97f59a163a74acd956e17c5.jpg)  
(b)  
Fig. 4. The scanned object mounted on the rotation stage (a) and the custom robot-based cone-beam CT test bench with the two-axis source and detector manipulators (b).

## C. Compared Methods and Evaluation

We compare discrete VCLS [16], discrete soft-Tuy greedy [15], and greedy-initialised continuous coverage plus VCL, bundle, or both. Discrete pools contain 720 Fibonaccisphere candidates, or 360 feasible candidates in the constrained study. All virtual selectors use the same SART reconstruction; Supplementary ?? reports the derivative and optimiser ablation.

Metrics are peak signal-to-noise ratio (PSNR), threedimensional structural similarity (SSIM) [51], both higher is better, and the high-frequency error norm (HFEN) [52], lower is better, with defect-resolved spherical ROIs on the flange. Uncertainties are population standard deviations over three seeds unless stated otherwise.

## D. Optimisation Settings and Implementation

Virtual selection uses z = 2000 Radon normals, 100 Adam steps, and a cosine schedule from $\eta _ { 0 } = 0 . 0 5$ in the two-axis chart. Coverage has unit weight, VCL weight 0.2 when active with $r _ { 1 } = 1 0 ^ { - 3 }$ , and the 5×9 bundle weight is 0.2/ median ¯τ. Measured planning uses 150 steps, the constrained-elevation chart, and a 128<sup>3</sup> VCL model with voxel sampling fraction $r _ { 1 } ~ = ~ 1 0 ^ { - 3 }$ and ridge $\lambda _ { R } ~ = ~ 1 0 ^ { - 3 }$ and $\varepsilon _ { \mathrm { v c l } } ~ = ~ 0 . 5$ mm. Implementation uses MLX [53] with Torch/CUDA projections and reconstructions on a Linux workstation with two NVIDIA RTX PRO 6000 GPUs.

TABLE II  
SPHERE-BASED SELECTORS ON BOTH VIRTUAL OBJECTS, COLD-STARTED, MEAN ± POPULATION STANDARD DEVIATION. PSNR IN DB AND SSIM AREHIGHER-IS-BETTER, HFEN IS LOWER-IS-BETTER. THE SOFT-TUY GREEDY ROW IS THE DISCRETE FIXED-GRID SELECTOR ON THE SAME 720-POINTSPHERE AND THE INITIALISATION OF EVERY CONTINUOUS ROW, SO THE THREE ARMS BELOW IT MEASURE WHAT LEAVING THE CANDIDATE GRID ADDS.BOLD MARKS THE BEST VALUE PER COLUMN WITHIN EACH OBJECT.
<table><tr><td></td><td></td><td colspan="3"> $k = 4 0$ </td><td colspan="3"> $k = 8 0$ </td></tr><tr><td>Object</td><td>Selector</td><td>PSNR</td><td>SSIM</td><td>HFEN</td><td>PSNR</td><td>SSIM</td><td>HFEN</td></tr><tr><td rowspan="6">Defrise flange,  $3 8 4 ^ { 3 }$ </td><td>VCLS, discrete</td><td>45.82±0.02</td><td>0.981±0.000</td><td> $1 . 8 1 { \pm } 0 . 0 2 $ </td><td> $4 9 . 7 6 { \pm } 0 . 0 9$ </td><td> $\mathbf { 0 . 9 9 2 } { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 9 1 { \pm } 0 . 0 2$ </td></tr><tr><td>soft-Tuy greedy, discrete</td><td>41.32±0.56</td><td>0.951±0.003</td><td> $3 . 4 6 \pm 0 . 3 7$ </td><td> $4 5 . 2 5 { \pm } 0 . 3 9$ </td><td>0.980±0.000</td><td>1.87±0.15</td></tr><tr><td>cov+bundle</td><td>44.66±0.60</td><td>0.963±0.004</td><td> $2 . 0 0 { \pm } 0 . 1 5 $ </td><td> $4 7 . 0 7 { \pm } 0 . 3 9$ </td><td>0.981±0.000</td><td>1.43±0.10</td></tr><tr><td>cov+VCL</td><td>43.91±0.53</td><td>0.962±0.002</td><td> $2 . 3 6 \pm 0 . 2 1$ </td><td> $4 9 . 3 6 { \pm } 0 . 1 5 $ </td><td>0.989±0.000</td><td>0.95±0.02</td></tr><tr><td>cov+VCL+bundle</td><td>45.72±0.25</td><td>0.970±0.002</td><td> ${ \bf 1 . 7 8 \pm 0 . 0 7 }$ </td><td>50.32±0.10</td><td>0.988±0.000</td><td>0.81±0.00</td></tr><tr><td>VCLS, discrete</td><td>28.99±0.17</td><td>0.770±0.004</td><td>11.19±0.16</td><td> $3 4 . 8 1 { \pm } 0 . 1 1 $ </td><td> $0 . 8 8 3 { \pm } 0 . 0 0 2$ </td><td>7.94±0.11</td></tr><tr><td rowspan="5">Fuel nozzle,  $5 1 2 ^ { 3 }$ </td><td>soft-Tuy greedy, discrete</td><td>29.19±0.14</td><td>0.770±0.000</td><td>11.33±0.19</td><td> $3 5 . 0 4 { \pm } 0 . 5 6 $ </td><td> $0 . 8 8 9 { \pm } 0 . 0 0 5$ </td><td>7.94±0.54</td></tr><tr><td>cov+bundle</td><td>28.83±0.06</td><td>0.772±0.006</td><td>11.80±0.13</td><td> $3 1 . 4 4 { \pm } 0 . 0 6$ </td><td> $0 . 8 3 8 { \pm } 0 . 0 0 1$ </td><td>9.20±0.10</td></tr><tr><td>cov+VCL</td><td>29.94±0.49</td><td> $0 . 7 8 4 { \pm } 0 . 0 1 1$ </td><td>10.99±0.53</td><td> $\mathbf { 3 5 . 2 7 \pm 0 . 2 4 }$ </td><td> $\mathbf { 0 . 8 9 1 } { \scriptstyle \pm 0 . 0 0 2 }$ </td><td>7.63±0.17</td></tr><tr><td>cov+VCL+bundle</td><td> $\mathbf { 3 0 . 0 8 } { \pm } 0 . 1 3$ </td><td> $\mathbf { 0 . 7 9 2 } { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 1 0 . 5 5 \pm 0 . 1 1 }$ </td><td> $3 3 . 1 9 { \pm } 0 . 3 4$ </td><td> $0 . 8 6 6 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>7.86±0.24</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## V. RESULTS

## A. Noise-Free Benchmarks

The near-axisymmetric fuel nozzle is the conservative case for a continuous selector, because a circular orbit draws an object-specific geometric advantage on this part, so we compare the sphere-based selectors among themselves. Table II reports all five families on both noise-free benchmarks.

On the nozzle the information term is what carries the continuous arms. Coverage and VCL exceed the discrete sphere baseline at both budgets, so a continuous search reaches this object without the candidate basis the swap search needs. Coverage and absorption alone fall behind that baseline, at the larger budget by more than three decibels. A nearaxisymmetric part offers few geometrically distinguishable directions, so a purely geometric criterion has little to work with. Adding the information term recovers the loss at the sparse budget, where the three-term arm is the best selector on all three metrics. At the larger budget it does not, and the absorption term then costs about two decibels against coverage and VCL alone. The same arm and protocol produce the analytic column of Supplementary ??.

The soft-Tuy greedy row prices the off-grid step, since it is the selection every continuous arm starts from. On the flange the best continuous arm gains four decibels over it at the sparse budget and five at the larger one, cutting HFEN by about half. On the nozzle the same step buys less than a decibel, because the greedy grid solution there is already ahead of the swap search. Whether continuous refinement pays is therefore a property of the object, and it pays where the informative directions lie between the candidates.

Supplementary ?? compares the analytic gradient with finite differences and CMA-ES under matched objectives and evaluation budgets.

Photon noise preserves the PSNR ordering of the selectors on the flange and compresses it on the dense nozzle; Supplementary ?? reports the noise model, protocol, and full comparison.

## B. Constrained Acquisition

We evaluate the native Defrise flange at $K _ { \operatorname* { m a x } } = 3 6 0$ and $k ~ \in ~ \{ 4 0 , 8 0 \}$ on four reachable sets: the equatorial circle, the bench’s $\pm 3 0 ^ { \circ }$ elevation band, a C-arm envelope with $\pm 1 1 0 ^ { \circ }$ orbital rotation and $\pm 4 5 ^ { \circ }$ elevation, and the full sphere. Table III isolates the effect of the reachable set under a fixed selector, with defect-resolved metrics.

![](images/504f50d258b1a64cde5402115f809df0fa238451108313c0a8becf7b28c586ac.jpg)  
Fig. 5. Noise-free reconstructions at $k = 4 0$ , the nozzle rendered at $2 5 6 ^ { 3 }$ for legibility. The top row is a central axial slice of the fuel nozzle with the cooling-hole ring in the inset, the bottom row a central sagittal slice of the Defrise flange with the upper lack-of-fusion stack. The nozzle selections are visually equivalent, as the sub-decibel margins of Table II predict, whereas the flange separates them in the streak background.

At the LoF stacks, the equatorial circle trails the out-ofplane sets by five to nine decibels in ROI-PSNR at $k = 4 0$ and by more than ten at $k ~ = ~ 8 0$ . The growing deficit indicates missing Radon data rather than undersampling; bulkdominated global metrics understate it by at least a factor of two. Its bulk-dominated SSIM advantage persists only at the sparse budget.

The bench band recovers most of the full-sphere gain and matches the larger C-arm envelope at the defects. The Carm’s global PSNR nevertheless falls below the circle at k = 40: coverage drives about a third of its poses towards the azimuth limit, concentrating viewing directions. The full sphere remains the strongest reachable set at both budgets, globally and at the defects. Fig. 6 pairs each reconstruction with its trajectory.

Poses move off-grid by median great-circle angles of roughly one to three candidate spacings on the twodimensional reachable sets (Fig. 6); the circle remains unchanged. On the free sphere, 74% of poses finish within $5 ^ { \circ }$ of the source plane, but eleven of eighty lie at the LoF-stack heights of 20–27 mm and seven remain above $6 0 ^ { \circ }$ elevation. These out-of-plane poses carry the defect gain.

SELECTION WALL-CLOCK IN SECONDS AT $K _ { \operatorname* { m a x } } = 7 2 0$ ON ONE LINUX WORKSTATION WITH THE TORCH/CUDA BACKEND, RECONSTRUCTION EXCLUDED. VCLS IS SPLIT INTO THE OBJECT-SPECIFIC  
TABLE III  
KINEMATIC STUDY ON THE DEFRISE FLANGE, NOISE-FREE, NATIVE 384<sup>3</sup>, MEAN ± POPULATION STANDARD DEVIATION. ROI-PSNR IS MEASURED AT THE UPPER LACK-OF-FUSION STACK, THE DESIGNED CIRCLE-BLIND DEFECT FAMILY; THE LOWER STACK BEHAVES ALIKE. EVERY ROW USES THE SAME COMPOSITE OBJECTIVE, GREEDY INITIALISATION AND POSE PARAMETRISATION; ONLY THE REACHABLE SET CHANGES, ENTERING THE REFINEMENT STAGE AS BOUNDS. BOLD MARKS THE BEST VALUE PER COLUMN.
<table><tr><td></td><td colspan="4">k = 40</td><td colspan="4">k = 80</td></tr><tr><td>Reachable set</td><td>PSNR</td><td>SSIM</td><td>HFEN</td><td>ROI-PSNR</td><td>PSNR</td><td>SSIM</td><td>HFEN</td><td>ROI-PSNR</td></tr><tr><td>equatorial circle</td><td> $4 2 . 5 6 { \pm } 0 . 0 0$ </td><td>0.972±0.000</td><td>2.91±0.00</td><td>25.36±0.00</td><td>44.67±0.00</td><td>0.985±0.000</td><td>2.14±0.00</td><td> $2 6 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>band, ±30° elevation</td><td> $4 3 . 9 3 { \pm } 0 . 2 1 $ </td><td>0.958±0.002</td><td> $2 . 2 6 { \pm } 0 . 0 9$ </td><td> $3 0 . 6 1 { \pm } 1 . 0 1 $ </td><td>48.55±0.06</td><td> $0 . 9 8 7 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 1 1 { \pm } 0 . 0 2 $ </td><td> $3 6 . 8 3 { \pm } 0 . 8 4$ </td></tr><tr><td> $\mathrm { C } { \cdot } \mathrm { a r m } , \pm 1 1 0 ^ { \circ } / \pm 4 5 ^ { \circ }$ </td><td> $4 2 . 0 8 { \pm } 0 . 5 1 $ </td><td> $0 . 9 4 6 { \pm } 0 . 0 0 4$ </td><td> $2 . 8 8 \pm 0 . 2 0 $ </td><td> $3 0 . 4 5 { \pm } 1 . 3 4$ </td><td>46.98±0.66</td><td> $0 . 9 8 0 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $1 . 4 8 { \pm } 0 . 1 7 $ </td><td> $3 6 . 9 1 { \pm } 1 . 7 0 $ </td></tr><tr><td>full sphere</td><td>45.61±0.34</td><td>0.968±0.003</td><td> ${ \bf 1 . 7 8 \pm 0 . 0 8 }$ </td><td> $\mathbf { 3 4 . 6 4 } \pm 0 . 7 4$ </td><td>51.01±0.28</td><td> $\mathbf { 0 . 9 9 0 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td></td><td>0.72±0.04 39.56±0.98</td></tr></table>

![](images/4796f3704a2cc1e30d2f7bb619f0c12bae86f242e067efb08d7558f554ec5336.jpg)  
Fig. 6. Defrise flange at $k = 8 0 ,$ native $3 8 4 ^ { 3 } .$ . Each geometry column pairs the sagittal reconstruction (top, with LoF inset) with its source poses (bottom). The equatorial circle loses laminar contrast; the bench band already resolves the layers. Open blue circles show Tuy-greedy poses, filled red dots the optimised poses, grey lines their displacement, and dashed lines the reachable limits. Trajectory panels share the elevation scale.

TABLE IV  
PRESCAN SENSITIVITY ON THE DEFRISE FLANGE AT NATIVE 384<sup>3</sup>, k = 40, $K _ { \operatorname* { m a x } } = 3 6 0$ . EACH ENTRY IS THE PAIRED PSNR DIFFERENCE IN DB BETWEEN PLANNING ON A PRESCAN RECONSTRUCTION FROM n<sub>s</sub> VIEWS AND PLANNING ON THE TRUE VOLUME, SO ZERO IS THE ORACLE PLAN AND NEGATIVE IS WORSE, MEAN OVER FIVE SEEDS. ONLY OUR OWN SELECTORS ARE LISTED.
<table><tr><td rowspan="2">Selector</td><td rowspan="2">Prescan</td><td colspan="3"> $n _ { \mathrm { s } }$  prescan views</td></tr><tr><td>8</td><td>16 32</td><td>64</td></tr><tr><td rowspan="2">cov+bundle</td><td>circle</td><td>-2.67</td><td>-1.05 -1.69</td><td>-2.10</td></tr><tr><td>band</td><td>-0.04</td><td>-0.81 -1.24</td><td>-0.02</td></tr><tr><td rowspan="2">cov+VCL+bundle</td><td>circle</td><td>-5.62</td><td>-2.37 -3.21</td><td>-4.86</td></tr><tr><td>band</td><td>-3.20</td><td>-2.35 -2.75</td><td>-1.40</td></tr></table>

## C. Scout Robustness and Selection Cost

We plan on 15-iteration SART prescans of the native Defrise flange with ${ n _ { \mathrm { s } } } \in \{ 8 , 1 6 , 3 2 , 6 4 \}$ views, then evaluate against the true volume. Circular and $\pm 3 0 ^ { \circ }$ -band prescans use equal view budgets, isolating prescan geometry from measurement cost.

TABLE V  
CANDIDATE-BASIS PRECOMPUTE AND THE SWAP LOOP THAT USES IT. OURS IS THE GREEDY-INITIALISED COVERAGE PLUS BUNDLE ARM. BEST PER ROW IN BOLD, SECOND BEST UNDERLINED.
<table><tr><td colspan="3"></td><td colspan="3">VCLS</td></tr><tr><td>Phantom</td><td>Grid</td><td>k</td><td>Precomp. Swap</td><td>Total</td><td>Ours Total Gain</td></tr><tr><td rowspan="2">Defrise flange 3843</td><td rowspan="2"></td><td>40</td><td>98.1</td><td colspan="2">0.6 98.7</td></tr><tr><td>80</td><td>98.1</td><td>5.0 103.1 10.1</td><td>5.1 19× 10×</td></tr><tr><td rowspan="2">Fuel nozzle</td><td rowspan="2"> $5 1 2 ^ { 3 }$ </td><td>40</td><td>262.1</td><td>1.2 263.3</td><td>5.6 47×</td></tr><tr><td>80</td><td>262.1</td><td>3.6 265.8</td><td>13.0 20×</td></tr></table>

Table IV shows that prescan geometry matters more than budget. Circular prescans cost the three-term selector several decibels, and additional views do not reliably recover the loss. The attenuation map remains strongly correlated with the true one, but missing-cone artefacts mislead the information term and drive the planned poses to higher elevations. A band prescan at the same measurement cost removes most of the penalty at 8 and 64 views, less at 32, and none at 16. The flange is deliberately an extreme case.

Table V prices selection itself, with reconstruction excluded and the VCLS precompute separated from the swap loop it feeds. Our objective scales mainly with the view budget, not the reconstruction grid, whereas VCLS ties its precompute to the grid and must be rebuilt for each new part. The margin therefore grows with resolution instead of shrinking.

TABLE VI  
MEASURED CAMERA EXPERIMENT AGAINST THE QUANTITATIVE1200-VIEW FDK REFERENCE. THE CIRCULAR SUBSETS SHARE THEREFERENCE ACQUISITION. EVERY METRIC IS COMPUTED ON THE SAMEREFERENCE-FRAME FULL-VOLUME PAIR. BEST VALUES WITHIN EACHBUDGET ARE BOLD AND SECOND-BEST VALUES ARE UNDERLINED.
<table><tr><td>Trajectory</td><td>k</td><td>PSNR SSIM</td><td>HFEN</td></tr><tr><td>circular subset uniform on band planned bundle planned full composite</td><td>100 50.02 100 45.53 100 44.22</td><td>0.9785 0.9552 0.9495 0.9510</td><td>16.93 27.21 31.31</td></tr><tr><td>circular subset uniform on band planned bundle</td><td>100 400 400 400</td><td>44.09 51.49 0.9795 45.09 0.9523 43.71 0.9485</td><td>32.95 12.11 27.58 32.45</td></tr></table>

## D. Measured Robot-Based CT

Planning uses only the 120-view prescan of Sec. IV-B, and the 1200-view reference is withheld until evaluation. The optimised poses span the available elevation band.

On the measured prescan, coverage saturates within about thirty steps. $\mathrm { A t } \ k \ \mathrm { ~ = ~ } \ 4 0 0$ the optimisation then reduces the mean bundle integral by roughly a third while coverage changes by less than one percent, so the two terms separate cleanly. This is geometric evidence only, and the acquired scans decide whether it improves image quality.

At k = 100, uniform sampling leads both planned arms on every metric; at $k ~ = ~ 4 0 0$ it leads the bundle-only arm throughout and the three-term arm on two of three metrics. The reference-based metrics therefore show no consistent gain over uniform sampling on the same band, though the imperfect reference discussed below warrants caution in reading them. The visual comparison in Fig. 7 adds a distinction the aggregate numbers miss; the three-term arm reconstructs the pins in the upper enlargement most completely, at the shown budget and at $k = 4 0 0$ alike, whereas the bundle-only arm yields a clean overall slice whose fine detail remains imperfect in places.

The circular subset is best on every metric, but is not a clean selector comparison because it is drawn directly from the reference acquisition, avoids cross-session alignment, and does not use the $\pm 3 0 ^ { \circ }$ elevation range. The reference also contains radial streaks that are strongest for the circular subset and progressively weaker for uniform band sampling and the planned arms, consistent with suppression by out-of-plane views. Since the metrics reward agreement with this artefactbearing reference, they favour the circular subset and do not reflect absolute fidelity. We therefore rely primarily on fixedbudget comparisons among the separately acquired band pose sets.

## VI. DISCUSSION

The matched benchmarks show that continuous selection reaches the quality of discrete swap search. VCL approximates normalised mean-squared error [16], so a strong VCL solution leaves little PSNR headroom. Off-grid refinement adds four to five decibels on the asymmetric flange but less than one on the near-axisymmetric nozzle. The estimator ablation attributes final quality to the objective and selection cost to the gradient; photon noise compresses the margins.

The reachable set outweighs the selector. Leaving the equatorial plane recovers the designed laminae, and the moderate bench band captures most of the free-sphere gain. On bounded envelopes, however, coverage drives poses towards the azimuth limit and lowers global performance relative to the full sphere. Planning-prior geometry matters for the same reason: circularprescan artefacts mislead the continuous information term, while distributing the same prescan views over the reachable band removes most of the penalty at most budgets. Selection cost follows the view budget rather than the reconstruction grid, so the advantage over the discrete precompute grows with resolution.

The measured study demonstrates physical execution and reconstruction in a common frame, but no consistent metric gain over uniform band sampling. The circular reference contains streaks suppressed by the planned arms, so referencebased metrics partly measure artefact agreement. The threeterm arm resolves the fine pins most completely in Fig. 7.

Geometry-only coverage is the default without a prior or under tight selection cost; the bundle term is preferable when a prior exists and attenuation or occlusion matters. Attenuation can also pull poses into unfavourable geometry, as on the Defrise phantom, where adding the information term keeps poses out of plane.

Five limitations bound these claims. The optimiser is local and the objective is degenerate at these budgets. Photon noise is modelled on the virtual benchmarks only, as monochromatic Poisson transmission noise at one idealised incident fluence, without scatter, detector effects, beam hardening or a calibrated polychromatic spectrum. The measured comparison confounds several effects, since the circular subsets share the reference acquisition and the band budgets come from independently acquired sessions. The measured study plans with the historical bundle estimator. Finally, every study plans for the object that is subsequently scanned, so what the prescan study bounds is the sparsity of the prior and not a mismatch between prior and part.

Two directions follow. Ordering the poses and interpolating them into a mechanically smooth path remains a subsequent step; optimising the coefficients of a trajectory parametrisation instead, for instance within the space-filling source paths of Kingston et al. [2], would impose continuity and velocity constraints directly and is also the most promising route to reducing the initialisation dependence. Differentiating through a limited number of reconstruction iterations would admit taskspecific image losses and opens the way to replanning the remaining poses from the projections already measured, so that the planning prior is progressively replaced by data from the object itself.

![](images/b4063554d97dc0ca13997a89d029827ec8eff9c9050e34d93630a357901d8769.jpg)  
Fig. 7. Measured central axial slices at $k = 1 0 0 .$ , showing the quantitative 1200-view FDK reference, the reference-coupled circular subset, same-manifold uniform sampling, and the two acquired planned arms. Gold boxes and the strip above show the complete pin region. Cyan boxes and the strip below show a high-attenuation component above the lens together with the streaks around it. All panels use the common reference grey-value window. Band acquisitions are reconstructed natively in the reference frame, and all panels use the residual-selected TV weight.

## VII. CONCLUSION

Optimising source poses on the scanner’s kinematic manifold provides a portable alternative to fixed candidate pools and their object-specific precomputation. The continuous selector reaches discrete-search quality on the free sphere while incorporating attenuation and scanner limits directly into selection. Reachable geometry and the planning prior govern the remaining differences. Continuous planning is therefore useful when per-part precomputation cannot be amortised, informative directions lie between candidates, or scanner kinematics must enter the objective. The robot study establishes acquisition feasibility, while consistent measured image-quality gains remain to be demonstrated.

## CODE AND DATA AVAILABILITY

The implementation, tests, paper configurations, and rerun manifest are available in the authors’ GitHub repository. The ORNL fuel-nozzle volume is available through the release cited in [49]. Access to the phantom data can be requested from the corresponding author, subject to the source data and institutional redistribution conditions.

## ACKNOWLEDGMENT

The authors used Claude, a large language model by Anthropic [54], for textual refinement and shortening of the manuscript and for correction and structuring of the research code. The model produced no experimental results, no measured numbers, and no scientific claims. The authors reviewed and verified all content and take full responsibility for it.

## REFERENCES

[1] G. Herl, J. Hiller, and A. Maier, “Scanning trajectory optimisation using a quantitative Tuybased local quality estimation for robot-based X-ray computed tomography,” Nondestructive Testing and Evaluation, vol. 35, no. 3, pp. 287–303, jun 21 2020.

[2] A. M. Kingston, G. R. Myers, S. J. Latham, B. Recur, H. Li, and A. P. Sheppard, “Space-Filling X-Ray Source Trajectories for Efficient Scanning in Large-Angle Cone-Beam Computed Tomography,” IEEE Transactions on Computational Imaging, vol. 4, no. 3, pp. 447–458, 2018.

[3] J. W. Stayman, S. Capostagno, G. J. Gang, and J. H. Siewerdsen, “Taskdriven source–detector trajectories in cone-beam computed tomography: I. Theory and methods,” Journal of Medical Imaging, vol. 6, no. 02, p. 1, may 2 2019.

[4] S. Hatamikia, A. Biguri, G. Herl, G. Kronreif, T. Reynolds, J. Kettenbach, T. Russ, A. Tersol, A. Maier, M. Figl, J. H. Siewerdsen, and W. Birkfellner, “Source-detector trajectory optimization in cone-beam computed tomography: a comprehensive review on today’s state-of-theart,” Physics in Medicine & Biology, vol. 67, no. 16, p. 16TR03, aug 16 2022.

[5] L. A. Feldkamp, L. C. Davis, and J. W. Kress, “Practical Cone-Beam Algorithm,” Journal of the Optical Society of America A, vol. 1, no. 6, pp. 612–619, 1984.

[6] F. Natterer and F. Wubbeling,¨ Mathematical Methods in Image Reconstruction. Philadelphia: Society for Industrial and Applied Mathematics, 2001.

[7] E. Y. Sidky and X. Pan, “Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization,” Physics in Medicine and Biology, vol. 53, no. 17, pp. 4777–4807, aug 13 2008.

[8] K. H. Jin, M. T. McCann, E. Froustey, and M. Unser, “Deep Convolutional Neural Network for Inverse Problems in Imaging,” IEEE Transactions on Image Processing, vol. 26, no. 9, pp. 4509–4522, 9 2017.

[9] J. Adler and O. Oktem, “Learned Primal-Dual Reconstruction,” IEEE Transactions on Medical Imaging, vol. 37, no. 6, pp. 1322–1332, 6 2018.

[10] C. Ye, L.-S. Schneider, Y. Sun, M. Thies, S. Mei, and A. Maier, “DRACO: differentiable reconstruction for arbitrary CBCT orbits,” Physics in Medicine & Biology, vol. 70, no. 7, p. 075005, mar 20 2025.

[11] H. K. Tuy, “An Inversion Formula for Cone-Beam Reconstruction,” SIAM Journal on Applied Mathematics, vol. 43, no. 3, pp. 546–552, 6 1983.

[12] A. Maier, P. Kugler, G. Lauritsch, and J. Hornegger, Discrete Estimation of Data Completeness for 3D Scan Trajectories with Detector Offset. Springer Berlin Heidelberg, 2015, pp. 47–52.

[13] B. Liu, J. Bennett, G. Wang, B. De Man, K. Zeng, Z. Yin, P. Fitzgerald, and H. Yu, “Completeness map evaluation demonstrated with candidate nextgeneration cardiac CT architectures,” Medical Physics, vol. 39, no. 5, pp. 2405–2416, apr 12 2012.

[14] G. Herl, A. Maier, and S. Zabler, “X-ray CT data completeness condition for sets of arbitrary projections,” SPIE, Tech. Rep., oct 18 2022.

[15] L.-S. Schneider and A. Maier, “Soft Tuy-Completeness for Robust Projection Selection in Cone-Beam CT,” arXiv [cs.CV], 2026, arXiv:2605.24023, submitted to Inverse Problems. [Online]. Available: https://arxiv.org/abs/2605.24023

[16] J. Lin, A. Ziabari, S. V. Venkatakrishnan, O. Rahman, G. T. Buzzard, and C. A. Bouman, “Tomographic Sparse View Selection Using the View Covariance Loss,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–11, 2025.

[17] M. Linde, W. Wiest, A. Trauth, and M. G. R. Sause, “Automated Robot-Based Computed Tomography Trajectory Optimization using Differential Evolution in 3d Radon Space,” Journal of Nondestructive Evaluation, vol. 44, no. 3, jun 10 2025.

[18] Y. S. Han, J. Yoo, and J. C. Ye, “Deep Residual Learning for Compressed Sensing CT Reconstruction via Persistent Homology Analysis,” arXiv:1611.06391, 2016. [Online]. Available: https://arxiv. org/abs/1611.06391

[19] Y. Sun, L.-S. Schneider, C. Ye, and A. Maier, “diffct: Differentiable CT Operators from Circular Orbits to Arbitrary Trajectories,” Preprints, 5 2026.

[20] K. J. Batenburg, W. J. Palenstijn, P. Balazs, and J. Sijbers, “Dynamic´ Angle Selection in Binary Tomography,” Computer Vision and Image Understanding, vol. 117, no. 4, pp. 306–318, 2013.

[21] M. A. Haque, M. O. Ahmad, M. N. S. Swamy, M. K. Hasan, and S. Y. Lee, “Adaptive Projection Selection for Computed Tomography,” IEEE Transactions on Image Processing, vol. 22, no. 12, pp. 5085–5095, 2013.

[22] J. Lin, S. V. Venkatakrishnan, G. T. Buzzard, A. K. Ziabari, and C. A. Bouman, “Edge Projection-Based Adaptive View Selection for Cone-Beam CT,” in 58th Asilomar Conference on Signals, Systems, and Computers, 2024, pp. 658–663.

[23] M. Burger, A. Hauptmann, T. Helin, N. Hyvonen, and J.-P. Puska, “Se-¨ quentially Optimized Projections in X-Ray Imaging,” Inverse Problems, vol. 37, no. 7, p. 075006, 2021.

[24] A. Fischer, T. Lasser, M. Schrapp, J. Stephan, and P. B. Noel, “Object¨ Specific Trajectory Optimization for Industrial X-ray Computed Tomography,” Scientific Reports, vol. 6, no. 1, jan 28 2016.

[25] F. Bauer, D. Forndran, T. Schromm, and C. U. Grosse, “Practical Part-Specific Trajectory Optimization for Robot-Guided Inspection via Computed Tomography,” Journal of Nondestructive Evaluation, vol. 41, no. 3, 2022.

[26] G. Herl, J. Hiller, M. Thies, J.-N. Zaech, M. Unberath, and A. Maier, “Task-Specific Trajectory Optimisation for Twin-Robotic X-Ray Tomography,” IEEE Transactions on Computational Imaging, vol. 7, pp. 894– 907, 2021.

[27] J.-N. Zaech, C. Gao, B. Bier, R. Taylor, A. Maier, N. Navab, and M. Unberath, “Learning to Avoid Poor Images: Towards Task-Aware C-Arm Cone-Beam CT Trajectories,” in Medical Image Computing and Computer Assisted Intervention (MICCAI). Springer, 2019, pp. 11–19.

[28] M. Thies, J.-N. Zach, C. Gao, R. Taylor, N. Navab, A. Maier, and¨ M. Unberath, “A Learning-Based Method for Online Adjustment of C-Arm Cone-Beam CT Source Trajectories for Artifact Avoidance,” International Journal of Computer Assisted Radiology and Surgery, vol. 15, pp. 1787–1796, 2020.

[29] P. Wu, N. Sheth, A. Sisniega, T. Wang, A. Uneri, R. Han, R. Vijayan, P. Vagdargi, B. Kreher, H. Kunze, G. Kleinszig, S. Vogt, S.-F. L. Lo, N. Theodore, and J. H. Siewerdsen, “C-Arm Non-Circular Orbits: Geometric Calibration, Image Quality, and Avoidance of Metal Artifacts,” in Proceedings of the 6th International Conference on Image Formation in X-Ray Computed Tomography, 2020, arXiv:2010.00175.

[30] F. Bauer, M. Goldammer, and C. U. Grosse, “Selection and Evaluation of Spherical Acquisition Trajectories for Industrial Computed Tomography,” Proceedings of the Royal Society A, vol. 477, no. 2250, p. 20210192, 2021.

[31] S. Hatamikia, A. Biguri, G. Kronreif, J. Kettenbach, T. Russ, H. Furtado, L. K. Shiyam Sundar, M. Buschmann, E. Unger, M. Figl, D. Georg, and W. Birkfellner, “Optimization for customized trajectories in cone beam computed tomography,” Medical Physics, vol. 47, no. 10, pp. 4786– 4799, aug 29 2020.

[32] S. Hatamikia, A. Biguri, G. Kronreif, M. Figl, T. Russ, J. Kettenbach, M. Buschmann, and W. Birkfellner, “Toward on-the-fly trajectory optimization for C-arm CBCT under strong kinematic constraints,” PLOS ONE, vol. 16, no. 2, p. e0245508, feb 9 2021.

[33] Z. Shen, Y. Wang, D. Wu, X. Yang, and B. Dong, “Learning to Scan: A Deep Reinforcement Learning Approach for Personalized Scanning in CT Imaging,” Inverse Problems and Imaging, vol. 16, no. 1, p. 179, 2022.

[34] T. Wang, F. Lucka, and T. van Leeuwen, “Sequential Experimental Design for X-Ray CT Using Deep Reinforcement Learning,” IEEE Transactions on Computational Imaging, vol. 10, pp. 953–968, 2024.

[35] Y. Yuan, L.-S. Schneider, and A. Maier, “Application of Gated Recurrent Units for CT Trajectory Optimization,” in Proceedings of the 8th International Conference on Image Formation in X-Ray Computed Tomography, 2024, arXiv:2405.09333.

[36] N. Hansen and A. Ostermeier, “Completely Derandomized Self-Adaptation in Evolution Strategies,” Evolutionary Computation, vol. 9, no. 2, pp. 159–195, 2001.

[37] G. J. Gang, J. W. Stayman, T. Ehtiati, and J. H. Siewerdsen, “Task-Driven Image Acquisition and Reconstruction in Cone-Beam CT,” Physics in Medicine and Biology, vol. 60, no. 8, pp. 3129–3150, 2015.

[38] M. Thies, F. Wagner, N. Maul, L. Pfaff, L.-S. Schneider, C. Syben, and A. Maier, “Optimizing CT Scan Geometries With and Without Gradients,” in Proceedings of the 17th International Meeting on Fully Three-Dimensional Image Reconstruction in Radiology and Nuclear Medicine, 2023.

[39] M. Thies, F. Wagner, N. Maul, L. Folle, M. Meier, M. Rohleder, L.- S. Schneider, L. Pfaff, M. Gu, J. Utz, F. Denzinger, M. Manhart, and A. Maier, “Gradient-Based Geometry Learning for Fan-Beam CT Reconstruction,” Physics in Medicine and Biology, vol. 68, no. 20, p. 205004, 2023.

[40] H. Fathi and T. van Leeuwen, “Source Design Optimization for Depth Image Reconstruction in X-Ray Imaging,” Mathematics, vol. 12, no. 10, p. 1524, 2024.

[41] L.-S. Schneider, M. Thies, C. Syben, R. Schielein, M. Unberath, and A. Maier, “Task-Based Generation of Optimized Projection Sets Using Differentiable Ranking,” in Proceedings of the 17th International Meeting on Fully Three-Dimensional Image Reconstruction in Radiology and Nuclear Medicine, 2023, arXiv:2303.11724.

[42] M. Linde, W. Wiest, A. Trauth, and M. G. R. Sause, “Selecting Feasible Trajectories for Robot-Based X-ray Tomography by Varying Focus-Detector-Distance in Space Restricted Environments,” Journal of Nondestructive Evaluation, vol. 43, no. 2, 2024.

[43] ——, “Trajectory Optimization for Few-View Robot-Based CT: Transitioning from Static to Object-Specific Acquisition Geometries,” Tomography of Materials and Structures, vol. 7, p. 100058, 2025.

[44] S. Yoo, S. Oh, S. Park, J. Lee, and H. K. Kim, “Optimizing Robot-CT Trajectories for Data Completeness,” Journal ofInstrumentation, vol. 21, no. 2, p. C02010, 2026.

[45] L.-S. Schneider, G. Herl, and A. K. Maier, “Integer Optimization of CT Trajectories using a Discrete Data Completeness Formulation,” in Proceedings of the 8th International Conference on Image Formation in X-Ray Computed Tomography, 2024, arXiv:2402.10223.

[46] R. Clackdoyle and F. Noo, “Quantification of Tomographic Incompleteness in Cone-Beam Reconstruction,” IEEE Transactions on Radiation and Plasma Medical Sciences, vol. 4, no. 1, pp. 63–80, 1 2020.

[47] G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher, “An analysis of approximations for maximizing submodular set functions—I,” Mathematical Programming, vol. 14, no. 1, pp. 265–294, 12 1978.

[48] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in International Conference on Learning Representations (ICLR), 2015.

[49] A. Ziabari, O. Rahman, S. V. Venkatakrishnan, and R. Dehoff, “X-ray Computed Tomography Data of Dense Metallic Components,” 6 2025.

[50] A. H. Andersen and A. C. Kak, “Simultaneous Algebraic Reconstruction Technique (SART): A Superior Implementation of the ART Algorithm,” Ultrasonic Imaging, vol. 6, no. 1, pp. 81–94, 1984.

[51] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[52] S. Ravishankar and Y. Bresler, “Mr Image Reconstruction From Highly Undersampled k-Space Data by Dictionary Learning,” IEEE Transactions on Medical Imaging, vol. 30, no. 5, pp. 1028–1041, 2011.

[53] A. Hannun, J. Digani, A. Katharopoulos, and R. Collobert, “Mlx: Efficient and Flexible Machine Learning on Apple Silicon,” https://github.com/ml-explore/mlx, 2023.

[54] Anthropic, “Claude,” 2025. [Online]. Available: https://www.anthropic. com/claude