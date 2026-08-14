# Structure-aware Riemannian Growth Fields for 4D Plant Modeling

![](images/372d4d419df9cd2c9ffe7f01ea52989c77fef0ab06a790dbe4f0a2d95ca697fe.jpg)  
Figure 1. Our method enables spatio-temporal, dense and holistic 4D plant generation from as few as two sparse timepoint observations. The results show that the method can track correspondences as time progress, which would otherwise be hard to achieve with conventiona methods. Nodes on the topological skeleton are color-coded by their tracking identities to demonstrate structural consistency.

## Abstract

In this paper, we introduce a novelframeworkfor 4D plant growth modeling that reconstructs the continuous geometric and topological evolution ofplants from sparse temporal observations. Existing methods mainly rely on dense registration, yet reliable dense sequences are hard to obtain due to scanning constraints and self-occlusions, leaving these approaches struggling under large temporal gaps where rapid organ emergence violates local rigidity. To overcome this, we bridge these gaps by formulating plant morphogenesis as a continuous procedural process on a structure-aware Riemannian growth field; this jointly models topology evolution and geometric deformation, preserving botanical hierarchies and stable spatio-temporal correspondences across distant timepoints. Our key idea is to ground symbolic growth rules within a continuous geodesic flow, where organ development follows biologically modulated trajectories that preserve structural coherence under topological changes. We further contribute a 10-day dualspecies dataset with dense geometric and semantic annotations. Experiments demonstrate that our method accurately tracks individual organ growth over time and significantly outperforms state-of-the-art baselines in both geometric accuracy and correspondence consistency.

## 1. Introduction

Plant modeling plays an increasingly important role in applications ranging from precision agriculture and autonomous greenhouse management to environmental biophysical simulation. Recovering high-fidelity plant geometry remains challenging due to intricate branching structures, severe self-occlusions, and texture sparsity. Traditional photogrammetry and LiDAR methods often struggle to capture these thin, overlapping organs, whereas recent 3D Gaussian Splatting (3DGS) approaches [8, 20, 32, 34] have demonstrated promising results through explicit surface primitives and high-quality rendering. Nevertheless, existing frameworks largely treat plants as static entities. Modeling continuous plant growth is essential for understanding developmental dynamics, yet it remains fundamentally difficult due to large non-rigid deformations and persistent topological changes inherent in biological morphogenesis.

Recent work in 4D plant modeling seeks to reconstruct continuous growth from discrete data, yet acquiring dense temporal captures is often infeasible due to hardware constraints and severe self-occlusions over cultivation cycles. Under such sparse constraints, registration methods [1, 4, 21] rely on local geometric approximations, failing across distant timepoints where abrupt organ emergence violates the local-rigidity assumption. While skeletal frameworks [11] introduce formal shape metrics, their optimization lacks biological constraints and is prone to local minima that violate temporal causality without dense inputs, causing branch sliding and corrupted correspondences. A separate line of work injects botanical priors through rulebased grammars [15, 26]: classical differential L-systems [24] model growth via forward simulation of hand-specified symbolic rules in flat Euclidean space, whereas recent datadriven approaches [6, 14] predict static geometry from 3D data—neither optimizing a developmental trajectory over the intrinsically curved manifolds of shape space.

This raises a fundamental question: Can we reconstruct continuous, biologically plausible development given reliable data at only afew sparse timepoints? In this paper, we introduce a novel framework for reconstructing continuous 4D plant morphogenesis from as few as two sparse temporal observations (Fig. 1). Rather than treating growth as registered snapshots or static grammar predictions, we jointly model topology evolution and geometric deformation, enabling stable spatio-temporal correspondences throughout development. Our key idea is to formulate plant development as a continuous procedural process on a structureaware Riemannian growth field. By replacing discrete Euclidean rules with continuous geodesic dynamics, organ elongation and expansion follow energetically optimal trajectories. This bridges high-level botanical hierarchy and geometric evolution, yielding a unified 4D representation that preserves structural identity even as topology changes.

More specifically, we represent the evolving plant skeleton as a dynamic graph in which the birth of each node is triggered by a symbol-specific wait interval relative to its parent. To ensure realistic deformation, we derive a continuous growth velocity field that minimizes a morphological smoothness energy across a stratified manifold, where new structural degrees of freedom emerge at birth events. The velocity field is further governed by a sigmoid growth modulation that enforces biologically plausible maturation limits. Building on these structural trajectories, we generate a holistic 4D representation of stems and leaves, whose surface evolution follows anisotropic expansion aligned with intrinsic growth directions.

We collect a 10-day dual-species dataset comprising daily growth measurements of bean and broccoli specimens. This dataset provides comprehensive 4D annotations spanning dense geometry, extracted skeletal topologies, and semantic labels for individual organs. Experimental results demonstrate that our method reconstructs continuous growth with stable spatio-temporal correspondences, improved geometric fidelity, and topological consistency—outperforming conventional methods that struggle under sparse, large-displacement settings.

## 2. Related Works

## 2.1. Static Plant 3D Reconstruction

Recent research on plant reconstruction has transitioned from traditional approaches such as Structure-from-Motion (SfM) and LiDAR [5, 29, 30, 35] toward high-fidelity neural representations [1, 8, 9, 16, 20, 31, 32, 34] that allow for explicit surface extraction. In particular, 3D Gaussian Splatting (3DGS) [10] has emerged as a pivotal technique for representing botanical scenes through anisotropic Gaussian primitives. Splanting [20] introduced a fast-capture phenotyping pipeline based on 3DGS, while Zhang et al. [34] integrated 3DGS with the Segment Anything Model (SAM) [13] for automated wheat head segmentation and trait extraction. Despite their exceptional geometric detail, these representations treat the plant as a static snapshot and lack a temporal mechanism to model the continuous growth processes of the specimen.

## 2.2. Procedural Plant Growth Modeling

Most botanical frameworks simulate organs [12, 17] or global architecture [15, 18, 23, 26] via manual biological priors. To model continuous temporal changes, classical differential L-systems [24] combine symbolic rules with differential equations; however, they operate as handspecified forward simulations in flat Euclidean space, failing to capture how biological forms develop on the intrinsically curved manifolds of shape space [7]. While recent data-driven methods [6, 14] learn grammars or parameterized representations directly from 3D target observations, they rely on a discrete sequence-generation paradigm, producing static geometry without optimizing the deformation trajectory. In contrast, we ground data-driven topological rule generation within a continuous geodesic flow on a stratified Riemannian manifold, optimizing across sparse temporal endpoints into an energy-optimal, biologically consistent growth trajectory.

## 2.3. 4D Plant Modeling and Tracking

Reconstructing temporally consistent plant models is challenging due to the non-rigid deformations and topological changes inherent in biological growth. Geometry-based registration frameworks [1, 4, 21] rely on local approximations and require dense temporal sampling, failing across wide observation gaps where rapid organ emergence violates local rigidity. The closest work to ours is the Riemannian framework by Khanam et al. [11], which uses Square Root Velocity Function Trees (SRVFT) for non-rigid alignment. Though providing a rigorous metric for skeletal trajectories, its shape-space optimization lacks biological priors. Under sparse inputs, this geometric matching struggles with the search space of discrete organ appearance and is prone to local minima that violate temporal causality, causing branch sliding and corrupted correspondences. In contrast, our framework couples botanical hierarchies with continuous Riemannian flow on a stratified manifold, leveraging symbolic growth rules to recover intermediate birth events and maintain dense, biologically faithful spatiotemporal correspondences from sparse observations.

![](images/57755db0637af778255b21fdb8dbda2b5dee5649b5a1cbbdb5c0aaf337b322ca.jpg)  
Figure 2. Overview of our 4D Plant Modeling Pipeline. Given as few as two sparse temporal semantic point clouds, we first recover the structural scaffold and perform continuous topological growth inference to optimize a temporal L-system L(t) (Sec. 3). These symbolic transitions are then grounded in a structure-aware Riemannian growth field on a stratified manifold M(x(t)) (Sec. 4). By integrating a continuous growth velocity field Θ(<sup>˙</sup> x(t)) governed by morphological smoothness and sigmoid maturation priors, our framework generates a holistic 4D representation with dense spatio-temporal correspondences (Sec. 5).

## 3. Spatio-Temporal Structural Inference

To model morphogenesis as a continuous trajectory, we first establish a consistent topological transition between discrete observations. This section details the reconstruction of a structural scaffold from raw sensor data and the subsequent inference of the temporal rules governing its evolution. By mapping unstructured point clouds onto a dynamic symbolic representation, we provide the discrete boundary conditions required for the high-dimensional Riemannian growth fields described in Section 4. In Fig. 2, we illustrates the overall pipeline of our 4D plant modeling.

## 3.1. Topological Recovery and Graph Initialization

To establish a rigorous basis for 4D growth inference, given two semantic point clouds $P _ { t _ { 0 } }$ and $P _ { t _ { 1 } }$ , we first recover the underlying structural skeletal representation. We employ Laplacian-based contraction (LBC) [19] to derive an initial set of 1D curves that approximate the plant’s medial axis. To mitigate structural discontinuities arising from sensor occlusions, we perform a directional search along the local tangent of orphaned terminal nodes. This process yields a canonical skeletal graph $G = ( V , E )$ where each vertex is assigned a semantic attribute (e.g., internode, leaf).

By serializing G through a recursive depth-first traversal, we generate the bracketed L-string that defines the plant’s symbolic state. This initialization transforms the unstructured spatial data into a topological scaffold, providing the discrete boundary conditions necessary for the subsequent temporal inference.

## 3.2. Continuous Topological Growth Inference

Given L-strings at the observed time instances, we infer the production rules governing the transition between these states. Whereas prior L-systems [24, 25] rely on handspecified symbolic rules, ours are recovered directly from real data, modeling topological evolution as continuous, time-dependent transitions. At this stage, we treat the Lsystem as a symbolic scaffold devoid of geometric attributes such as branch length.

## 3.2.1. Temporal L-system

We represent the time-varying symbolic string of the evolving plant as a function of time $L ( t )$ . We define the maturation birth time $\tau _ { v }$ for each node v through a recursive temporal propagation from its parent $v _ { p } \mathrm { . }$

$$
\tau _ { v } = \tau _ { v _ { p } } + \Delta \tau _ { \mu ( v ) } ,\tag{1}
$$

where $\Delta \tau _ { \mu ( v ) }$ is a symbol-specific wait interval modeling organ-specific dormancy. In this formulation, the emergence of each node is no longer constrained to global discrete generation steps. Instead, each node’s birth is triggered by the completion of its parent’s growth, forming a sequential chain of topological events.

## 3.2.2. Dynamic Graph Representation

The birth time $\tau _ { v }$ defines the evolution of the semantic plant graph $G ( t ) = ( V ( t ) , E ( t ) )$ . At any time t, the graph consists of elements from the potential structure $( \nu , \mathcal { E } )$ that have reached their birth threshold:

$$
\begin{array} { l } { { V ( t ) = \{ v \in \mathcal { V } \mid t \geq \tau _ { v } \} , } } \\ { { E ( t ) = \{ ( v _ { p } , v ) \in \mathcal { E } \mid t \geq \tau _ { v } \} . } } \end{array}\tag{2}
$$

## 3.2.3. Optimization

We employ a Genetic Algorithm (GA) to optimize the parameter set $\Omega \ = \ \{ { \mathcal P } , \Delta \tau _ { \mu } \}$ , where $\mathcal { P }$ denotes the set of production rules, by minimizing the structural discrepancy at the observed boundary timestamps:

$$
\underset { \Omega } { \operatorname { a r g m i n } } \sum _ { i } \mathrm { D i s t } _ { \mathrm { p a t h } } \big ( L \big ( t _ { i } ; L _ { t _ { 0 } } , \Omega \big ) , L _ { t _ { i } } \big ) ,\tag{3}
$$

where we specifically evaluate the case $i = 1$ against the target string $L _ { t _ { 1 } }$ evolved from the source state $L _ { t _ { 0 } } . \ \mathrm { D i s t _ { p a t h } }$ is the structural distance defined by the Dice coefficient over the sets of unique positional paths H:

$$
\mathrm { D i s t } _ { \mathrm { p a t h } } ( L _ { A } , L _ { B } ) = 1 - \frac { 2 | \mathcal { H } ( L _ { A } ) \cap \mathcal { H } ( L _ { B } ) | } { | \mathcal { H } ( L _ { A } ) | + | \mathcal { H } ( L _ { B } ) | } .\tag{4}
$$

Each path in H encodes the symbolic identity of a node along with its hierarchical address and sibling order to ensure a topology-aware comparison. To strictly enforce structural identity, we introduce a thresholding gate where the distance is fixed to a small epsilon unless $\mathcal { H } ( L ( t _ { i } ; L _ { t _ { 0 } } , \Omega ) ) = \mathcal { H } ( L _ { t _ { i } } )$ , at which point the discrepancy reaches zero. This yields a stable topological sequence that serves as the prerequisite for subsequent geometric modeling in Riemannian space.

## 4. Structure-aware Riemannian Growth Fields

To ensure geometric consistency and avoid non-physical artifacts, such as discontinuous erratic shrinking, we model morphogenesis as a continuous trajectory within a stratified Riemannian manifold. This geometric flow is governed by the structural scaffold $L ( t )$ inferred in the previous section, ensuring that the plant’s evolving topology remains tightly coupled with its physical deformation.

## 4.1. Plant Riemannian State Space

We define the state of the plant at time t as $x ( t ) \ =$ $( G ( t ) , \Theta ( t ) ) ,$ , where the geometric parameters $\Theta ( t )$ are

structured to ensure developmental inheritance. For each edge $\boldsymbol { e } = \left( v _ { p } , v \right)$ , the parameters define a relative transformation from the terminal frame of its parent $v _ { p } \colon$

$$
\begin{array} { r l } & { \Theta ( t ) = \left( { \bf T } _ { \mathrm { r o o t } } ( t ) , \{ \theta _ { e } ( t ) \vert e \in E ( t ) \} \right) , } \\ & { \theta _ { e } ( t ) = ( s _ { e } ( t ) , { \bf R } _ { e } ( t ) , \kappa _ { e } ( t ) ) , } \end{array}\tag{5}
$$

where $\mathbf { T } _ { \mathrm { r o o t } } ( t ) \in S E ( 3 )$ is the global root pose, $s _ { e } ( t ) =$ log $\ell _ { e } ( t ) \in$ R is the log-length of edge $e ,$ ensuring $\ell _ { e } > 0$ [22], ${ \bf R } _ { e } ( t ) \in S O ( 3 )$ is the relative rotation with respect to the parent’s terminal tangent, and $\kappa _ { e } ( t ) \in \mathbb { R } ^ { k }$ represents the B-spline parameters (Hermite Splines) governing local curvature.

The plant evolves within a stratified Riemannian manifold $\mathcal { M } ( \boldsymbol { x } ( t ) )$ . Rather than a single fixed-dimensional space, $\mathcal { M } ( \boldsymbol { x } ( t ) )$ is a collection of strata where each stratum’s geometry is induced by the current state $x ( t )$ . Specifically, the state space at time t is formed by the product of the root pose and the individual edge manifolds: $\begin{array} { r } { \mathcal { M } ( { \boldsymbol { x } } ( t ) ) = S E ( 3 ) \times \prod _ { e \in E ( t ) } \left( \mathbb { R } \times S O ( 3 ) \times \mathbb { R } ^ { k } \right) } \end{array}$ . That is, the “sequential chain of topological events” corresponds to the trajectory $x ( t )$ transitioning between strata of increasing dimensionality. At each birth threshold $\tau _ { v } ,$ a new manifold factor $( \mathbb { R } \times S O ( 3 ) \times \mathbb { R } ^ { k } )$ is appended to the product. Continuity is preserved by initializing these new dimensions at the singular boundary where the edge length is approximated to zero $( s _ { e }  - \infty )$ , ensuring the plant geometry unfolds smoothly from the progenitor.

## 4.2. Continuous Growth Velocity Field

While the state space $\mathcal { M } ( \boldsymbol { x } ( t ) )$ ) defines the static configurations of the specimen, capturing morphogenesis requires a formal description of how these states evolve over time. To model plant morphogenesis as a continuous biological process, as shown in Fig. 3, we represent growth as a flow generated by a time-dependent velocity field $\dot { \Theta } ( x ( t ) , t )$ within the tangent space of $\mathcal { M } ( \boldsymbol { x } ( t ) )$ ). Rather than performing discrete interpolation between topological states, the trajectory $x ( t )$ is defined as the integral of this field over the growth interval $[ t _ { 0 } , t _ { 1 } ] ;$

$$
x ( t _ { 1 } ) = x ( t _ { 0 } ) + \int _ { t _ { 0 } } ^ { t _ { 1 } } \dot { \Theta } ( x ( t ) , t ) d t .\tag{6}
$$

The velocity vector $\dot { \Theta } = ( \dot { \bf T } _ { \mathrm { r o o t } } , \dot { s } _ { e } , \omega _ { e } , \dot { \kappa } _ { e } )$ captures the expansion and deformation rates of each organ. By parameterizing $\dot { \Theta }$ as a smooth function of time, we ensure that growth remains geometrically consistent even as the underlying graph $G ( t )$ undergoes topological transitions. Specifically, at the birth of a new edge $e \in E ( t )$ , the velocity field handles the activation of its growth potential, ensuring a smooth transition in the physical coordinate space as the organ elongates from its parent.

For each active edge $e ,$ the growth rates are defined as follows:

• Elongation Rate: The term $\begin{array} { r } { \dot { s } _ { e } ( t ) ~ = ~ \frac { d } { d t } \log \ell _ { e } ( t ) } \end{array}$ represents the relative growth rate. In log-space, this linear rate translates to exponential expansion in physical space.

• Angular Velocity: The rotational rate $\omega _ { e } ( t ) \in \mathfrak { s o } ( 3 )$ governs the orientation via $\dot { \bf R } _ { e } ( t ) = { \bf R } _ { e } ( t ) [ \omega _ { e } ( t ) ] \times$ , ensuring smooth tropism along manifold geodesics.

• Curvature Evolution: The term $\dot { \kappa } _ { e } ( t ) \in \mathbb { R } ^ { k }$ captures the temporal deformation of the organ’s medial axis. Since the B-spline parameters reside in a Euclidean subspace of the stratum, their evolution represents the continuous bending of the organ over time.

## 4.3. Morphological Smoothness

Integrating the velocity field yields a family of kinematically valid trajectories; however, additional constraints are required to filter for paths that honor the metabolic and structural limits of real botanical systems.

To ensure the inferred morphogenesis is biologically plausible, we define $E _ { \mathrm { s m o o t h } }$ as a penalty on the total variation of the growth velocity field. This term favors trajectories that exhibit steady growth and minimal unnecessary deformation, effectively acting as a “minimum-jerk” prior for plant development. We define the smoothness energy as the integral of the squared norm of the manifold acceleration as follows:

$$
E _ { \mathrm { s m o o t h } } = \int _ { t _ { 0 } } ^ { t _ { 1 } } \left\| \frac { D \dot { \Theta } ( t ) } { d t } \right\| _ { \mathcal { M } } ^ { 2 } d t ,\tag{7}
$$

where $\textstyle { \frac { D } { d t } }$ denotes the covariant derivative. Expanding this into our parameter space, the term penalizes the secondorder derivatives of the log-length, orientation, and curvature, ensuring that growth remains steady across the stratified manifold:

$$
\begin{array} { l } { \displaystyle \left. \frac { D \dot { \Theta } ( t ) } { d t } \right. _ { \mathcal { M } } ^ { 2 } = } \\ { \displaystyle \sum _ { e \in G ( t ) } \left( \alpha _ { e } | \ddot { s } _ { e } ( t ) | ^ { 2 } + \beta _ { e } \| \dot { \omega } _ { e } ( t ) \| _ { \mathfrak { s o } ( 3 ) } ^ { 2 } + \eta _ { e } \| \ddot { \kappa } _ { e } ( t ) \| ^ { 2 } \right) , } \end{array}\tag{8}
$$

where $\alpha _ { e } , \beta _ { e }$ , and $\eta _ { e }$ are the semantic-dependent weights representing axial, flexural, and configurational stiffness within the range [0.1, 1.0]. In our experiments, we set high penalties $( \alpha _ { \mathrm { e } } = \beta _ { \mathrm { e } } = \eta _ { \mathrm { e } } = 1 . 0 )$ for primary structural elements such as internodes to enforce structural rigidity. Conversely, we assign lower weights for terminal organs $( \alpha _ { e } ~ = ~ 0 . 1$ and $\beta _ { e } = \eta _ { e } = 0 . 2 )$ for leaves or petioles to accommodate complex environmental deformations (e.g., phototropic reorientation) without incurring excessive smoothness penalties.

![](images/51ba314ffc12117e1a87c0054be86e9114042ad22b472ee04652a454f7afa4f0.jpg)  
Figure 3. Continuous Growth Velocity Field. The developmental trajectory is modeled as a continuous geodesic path across time-varying topological states $x ( t )$ on a stratified manifold. The growth intensity, governed by the modulation factor $\xi ,$ regulates the transition from rapid organ elongation to biological maturation.

Sigmoid Growth Modulation. We assume that the growth follows a sigmoid characteristic [22], typically decaying as $\theta _ { e } ( t )$ approaches a saturation. To ensure this property and simultaneously preserve the flexibility of the intrinsic elongation potential, we define the modulation factor $\xi _ { \theta _ { \epsilon } }$ based on the derivative of the log-sigmoid function. Here, we introduce the component-specific absolute peak growth timing $T _ { \theta _ { e } } = \tau _ { v } + ( 1 - \tau _ { v } ) t _ { m , \theta _ { e } }$ , where $t _ { m } \in [ 0 , 1 ]$ is a global parameter normalized relative to the remaining lifespan from the inception $\tau _ { v } .$

$$
\begin{array} { l } { { \displaystyle \xi ( k _ { \theta _ { e } } , T _ { \theta _ { e } } ) = \frac { d } { d t } \log \left( \frac { 1 } { 1 + e ^ { - k _ { \theta _ { e } } ( t - T _ { \theta _ { e } } ) } } \right) } } \\ { { \displaystyle = \frac { k _ { \theta _ { e } } } { 1 + e ^ { k _ { \theta _ { e } } ( t - T _ { \theta _ { e } } ) } } . } } \end{array}\tag{9}
$$

The velocity field is then updated as $\dot { \theta } _ { e } ( t )  \dot { \theta } _ { e } ( t ) \cdot \xi _ { \theta _ { e } }$ To regularize the trajectory within the energy functional $E _ { \mathrm { s m o o t h } } .$ , the second-order derivative (acceleration) for any growth component $\theta _ { e } ( t )$ is given by:

$$
\ddot { \theta } _ { e } ( t ) \simeq - \dot { \theta } _ { e } ( t ) \frac { k _ { \theta _ { e } } ^ { 2 } e ^ { k _ { \theta _ { e } } ( t - T _ { \theta _ { e } } ) } } { ( 1 + e ^ { k _ { \theta _ { e } } ( t - T _ { \theta _ { e } } ) } ) ^ { 2 } } .\tag{10}
$$

This generalized form ensures that the growth of length, curvature, and torsion are all governed by the same underlying biological saturation logic, while allowing for distinct parameters $\{ k , t _ { m } \}$ for each physical property.

## 5. Holistic 4D Plant Modeling

Building upon the structural trajectories, we expand our skeletal manifold into a holistic 4D representation by modeling deformation of stems and leaves across the temporal domain.

## 5.1. Deformable Leaf

We treat each leaf as a deformable template $P _ { l e a f } ( t )$ anchored to a specific node v. To maintain consistency with the continuous Riemannian framework, the global trajectory of any point $\mathbf { p } ( t )$ on the leaf surface is defined by the composition of the skeletal rigid-body motion and an intrinsic anisotropic expansion flow:

$$
\mathbf { p } ( t ) = \mathbf { x } _ { v } ( t ) + \mathbf { R } _ { l e a f } ( t ) \cdot \mathbf { A } ( t ) \cdot \mathbf { p } _ { l o c a l } ,\tag{11}
$$

where $\mathbf { x } _ { v } ( t )$ is the absolute position of the anchor node and $\mathbf p l o c a l$ is the canonical template point. The leaf-specific orientation ${ \bf R } _ { l e a f } ( t )$ is governed by the angular velocity $\dot { \omega } _ { l e a f } ( t )$ with sigmoid modulation, coupled with the primary axis connecting the anchor node to the distal leaf tip.

The intrinsic growth of the leaf blade is modeled as an anisotropic scaling $f l o w$ in the local canonical space. We define the deformation tensor $\begin{array} { r l } { \mathbf { \boldsymbol { \Lambda } } ( t ) } & { { } = } \end{array}$ diag $( e ^ { s _ { l o n g } ( t ) } , e ^ { s _ { l a t } ( t ) } , e ^ { s _ { l a t } ( t ) } )$ , where $s l o n g$ and $s _ { l a t }$ represent the longitudinal (midrib) and latitudinal (lamina) logscales, respectively. To ensure biological plausibility, these scaling rates are slaved to the corresponding organ-specific sigmoid modulation ξ(t) (Sec. 4.3):

$$
[ \dot { s } _ { l o n g } ( t ) \dot { s } _ { l a t } ( t ) ]  [ \dot { s } _ { l o n g } \dot { s } _ { l a t } ] \cdot \xi ( t ) ,\tag{12}
$$

where s˙ represents the constant growth velocity in the logscale space. For leaves emerging post-birth $( \tau _ { v } > t _ { 0 } )$ , the scale is initialized to an infinitesimal value ϵ, representing the transition from a dormant bud to an expanding lamina.

## 5.2. Deformable Stem

We model the stem as a continuous assembly of Generalized Cylinders parameterized by the underlying skeletal splines [28]. To ensure structural continuity at branching junctions, we define the radius $r _ { v } ( t )$ at each node v rather than per edge. For an edge e connecting parent $v _ { p }$ to child v, the radius at any longitudinal position $u \in [ 0 , 1 ]$ is determined by the interpolation: $r _ { e } ( u , t ) = ( 1 - u ) r _ { v _ { p } } ( t ) +$ u $r _ { v } ( t )$ . By inheriting the propagation logic from the skeletal graph, $r _ { v } ( t )$ scales from its initial measured value at $t _ { 0 }$ (or an small value ϵ for nodes emerging post-birth) toward its terminal thickness at $t _ { 1 }$

The stem surface $S _ { e }$ is procedurally generated at time t via:

$$
\begin{array} { c } { { S _ { e } ( u , \phi , t ) = \gamma _ { e } ( u , t ) + } } \\ { { { \displaystyle r _ { e } ( u , t ) \left[ \cos ( \phi ) { \bf N } ( u , t ) + \sin ( \phi ) { \bf B } ( u , t ) \right] } } } \end{array}\tag{13}
$$

where $\gamma _ { e } ( u , t )$ is the skeletal Hermite spline, $\phi \in [ 0 , 2 \pi )$ is the azimuthal angle, and {N, B} represents the orthonormal basis of the Bishop Frame propagated via parallel transport [2]. This procedural mapping ensures that surface points automatically scale with the skeletal growth, eliminating the need for explicit point-level deformation.

## 5.2.1. Optimization

By combining these dynamic flow constraints with structural priors, we can formulate a global energy that reconciles the model with observed physical evidence. The optimization is performed over the control coefficients for all the modulation parameters $\{ k , t _ { m } \}$ and is defined as:

$$
\begin{array} { r } { E ( k , t _ { m } ) = E _ { \mathrm { d a t a } } + \lambda _ { 1 } E _ { \mathrm { s m o o t h } } , } \end{array}\tag{14}
$$

where $E _ { \mathrm { d a t a } } = d ( \hat { P } ( t ) , P _ { t } )$ represents the data fidelity term. The function $d ( \cdot )$ computes the Chamfer distance between the estimated point cloud $\hat { P } ( t )$ —integrated forward from the fixed initial state $\Theta ( t _ { 0 } )$ —and the corresponding target observation $P _ { t }$

## 6. Experiments

In this section, we provide a comprehensive quantitative and qualitative evaluation of our 4D growth model, assessing its performance in terms of topological stability, morphological fidelity, and temporal correspondence.

## 6.1. Real Plant Dataset Acquisition and Processing

To evaluate our 4D growth model, we utilized both the publicly available Pheno4D tomato dataset [27] and a customcollected dual-species temporal dataset. As shown in Fig. 4, our dataset consists of daily growth measurements of bean and broccoli specimens over 10 days. We acquired about 30 multi-view images per specimen per day using a Sony α7 ILCE-7K. We generated sparse 3D reconstructions via 3D Gaussian Splatting (3DGS) [33]. To ensure geometric density and semantic purity, 3D geometries were projected into 2D space, refined using the Segment Anything Model 3 (SAM 3) [3], and back-projected to 3D. Absolute scale was preserved via ChArUCO marker calibration. Skeletons were extracted using Laplacian-based contraction [19] and refined via directional search (Sec. 3.1). Ground truth semantic labels (stem, leaf, cotyledons) of point clouds were manually annotated to provide a baseline for topological recovery.

## 6.2. Quantitative Evaluation

We quantitatively evaluate the modeling accuracy and robustness of our method on both Pheno4D tomato specimens [27] and our multi-species dataset. Tab. 1 reports our method against skeletal [11] and holistic registration [21] baselines. Errors are reported as the mean Chamfer Distance (CD) in mm. The results demonstrate the fundamental advantage of our approach in integrating a kinematically valid growth trajectory across time, outperforming baselines that struggle with the large displacements and topological changes inherent in botanical development, especially across distant timepoints.

![](images/661bf5c7b91a90b698ef2f83919a7c5b87d3b5cbf61478e85b7e1bf87e8a5188.jpg)  
Figure 4. Bean and broccoli data collection over 10 days. Each day, for every specimen, we capture about 30 multi-view RGB images, extract masks using SAM 3 [3], reconstruct the 3D geometry through a Gaussian Splatting–based multi-view pipeline, extract the skeleton, and perform instance annotation, yielding the temporal sequence of 3D geometry. The same process generalizes to other species.

Table 1. Quantitative comparison of 4D reconstruction accuracy. We report Skeletal Chamfer Distance (Sk.) and holistic surface CD (Hl.) in mm. Note that [11] reports only skeletal results, while [21] reports only holistic results. Our method achieves higher accuracy.
<table><tr><td>Method</td><td>Tomato ↓</td><td>Bean↓</td><td>Broccoli↓</td></tr><tr><td>Pan et al. [21] Hl.</td><td>12.68</td><td>16.11</td><td>6.32</td></tr><tr><td>Khanam et al. [11] Sk.</td><td>21.12</td><td>20.23</td><td>2.09</td></tr><tr><td>Ours Sk.</td><td>2.09</td><td>8.83</td><td>1.01</td></tr><tr><td>Ours Hl.</td><td>5.45</td><td>7.86</td><td>1.92</td></tr></table>

Fig. 5 presents the robustness analysis. In Fig. 5(a), we report the mean temporal Chamfer Distance under Gaussian noise $( \sigma \in [ 0 , 1 0 ]$ mm) injected into the input point-cloud pairs for broccoli, bean, and tomato; error increases linearly with noise magnitude. Fig. 5(b) evaluates the GA-recovered rules under input corruption: on a tomato with 11 leaves, we randomly drop leaf labels at both timepoints and report the mean structural error via the Dice coefficient (higher is better) against ground truth. Performance degrades smoothly as corruption increases. Moreover, in Fig. 5(c), we report the estimation error for different numbers of input observations. The results show that modeling accuracy improves as the number of inputs increases. Tab. 2 investigates the impact of the sigmoid modulation factor ξ by comparing our full model against a baseline that assumes a constant growth rate. Our full model reduces errors across all species, validating the effectiveness of the modulation.

## 6.3. Qualitative Evaluation

As shown in Fig. 6 and Fig. 7, our model accurately captures the non-linear morphological progression of tomato and bean specimens. The continuous Riemannian flow preserves surface manifold integrity across all observed time instances. In contrast, baseline methods [11, 21] either exhibit tracking drift, where branches slide laterally or translate upward, or produce fragmented intermediate geometry when interpolating between distant snapshots (indicated by red arrows). This structural stability is particularly vital during organ inception. By initializing growth at the singular boundary of the stratified manifold, our model ensures that new leaves and internodes unfold smoothly from their progenitors, eliminating the discrete popping or volume loss. Fig. 8 compares dense correspondences recovered from two sparse timepoint observations. While our method yields consistent dense correspondences, the baselines produce spurious matches that erroneously link noncorresponding points, validating our approach’s ability to accurately track growth trajectories across development.

![](images/fef6851f5127003fb032fd6fe7531182e84ba5dad7b99c2557afa4f384180841.jpg)  
(a)

![](images/855b3eda5515d4a59d3ec9322ed80b89cb4354a2c6ca99aeb65d0fbe8e9b08c9.jpg)  
(b)

![](images/aae3f4b937a3701104e6832bae9e34a7a89097be8aeec24bb26bdde34e55616b.jpg)  
(c)  
Figure 5. Robustness analysis. (a) Mean estimation errors at different noise levels. (b) GA robustness under missing semantic labels. (c) Mean estimation errors at different number of inputs. Colors denote different species.

Table 2. Ablation study on geometric priors. We report mean Skeletal (Sk.) and Holistic (Hl.) Chamfer Distances (mm) w/ and w/o Sigmoid Modulation ξ.
<table><tr><td rowspan="2">Method</td><td colspan="2">Tomato</td><td colspan="2">Bean Broccoli</td></tr><tr><td>Sk. ↓ Hl. ↓</td><td>Sk. ↓ HI. ↓</td><td>Sk. ↓</td><td>Hl. ↓</td></tr><tr><td>wloξ</td><td>2.30 5.85</td><td>9.49 8.44</td><td>1.11</td><td>2.01</td></tr><tr><td>Ours</td><td>2.09 5.45</td><td>8.83 7.86</td><td>1.01</td><td>1.92</td></tr></table>

![](images/8336bd40bb1a7602df0c680dfd44d2dd784fd38af1cad5c61fca1a24392a751f.jpg)  
Figure 6. Qualitative 4D Reconstruction of Pheno4D Tomato. We show recovered growth trajectories of a tomato specimen over time. The input pair is highlighted with red boxes. From top to bottom, the rows display: GT temporal 3D point clouds; skeletal reconstruction from Khanam et al. [11]; spatiotemporal registration results from Pan et al. [21]; our generated continuous holistic geometry; and our recovered topological skeletons with invariant node identities (indicated by consistent color-coding). Our approach better recovers temporally continuous geometry.

## 7. Conclusion

In this paper, we presented a novel framework for reconstructing continuous 4D plant morphogenesis from sparse temporal observations. By formulating plant development as geodesic evolution on a structure-aware Riemannian growth field, our method jointly models topology evolution and geometric deformation within a single continuous representation, enabling stable dense correspondences and precise organ-level growth tracking. Experimental result demonstrate the effectiveness of our approach in capturing complex growth dynamics. Yet, challenges remain — severe self-occlusions and reconstruction noise can affect reconstruction stability. Moreover, fine-scale phenomena such as leaf unflurring or subtle surface unfolding are not explicitly modeled. Addressing these limitations through tighter integration of biomechanical and appearance-aware dynamics remains an important direction for future work. Despite these challenges, we believe our formulation establishes a principled foundation for 4D plant modeling and future predictive digital plant twins.

![](images/139d73a14023606f9eac3b71ea96e6bcd901ccc3a746c481a0f44501b43dfe01.jpg)

Figure 7. Qualitative 4D Reconstruction of Bean. We visualize the recovered growth trajectories of bean specimens over time. The input pair is highlighted with red boxes. Note that our Riemann flow preserves better branch topology and identity.  
![](images/c1ec4f22eee7e02fe89b6bddd22779d39168491f004295e4cca01d83ea53315c.jpg)  
Figure 8. Qualitative comparison of dense correspondences recovered from two sparse timepoints (matching colors denote identical points). Our method yields markedly more consistent correspondences, whereas baselines produce non-corresponding matches.

Acknowledgment This work was in part supported by the Organization for the Promotion of Gender Equality at Nara Women’s University.

## References

[1] Simeon Adebola, Shuangyu Xie, Chung Min Kim, Justin Kerr, Bart M. van Marrewijk, Mieke van Vlaardingen, Tim van Daalen, E.N. van Loo, Jose Luis Susa Rincon, Eugen Solowjow, Rick van Zedde, and Ken Goldberg. Growsplat: Constructing temporal digital twins of plants with gaussian splats. In 2025 IEEE 21st International Conference on Automation Science and Engineering (CASE), pages 1766– 1773, 2025. 1, 2

[2] Richard L Bishop. There is more than one way to frame a curve. The American Mathematical Monthly, 82(3):246– 251, 1975. 6

[3] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025. 6, 7

[4] Nived Chebrolu, Thomas Labe, and Cyrill Stachniss. Spatio-¨ temporal non-rigid registration of 3d point clouds of plants. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 3112–3118. IEEE, 2020. 1, 2

[5] Jordi Gene-Mola, Eduard Gregorio, Fernando Auat Cheein,´ Javier Guevara, Jordi Llorens, Ricardo Sanz-Cortiella, Alexandre Escola, and Joan R Rosell-Polo. Fruit detection,\` yield prediction and canopy geometric characterization using lidar with forced air flow. Computers and Electronics in Agriculture, 168:105121, 2020. 2

[6] Samara Ghrer, Christophe Godin, and Stefanie Wuhrer. Learning to infer parameterized representations of plants from 3d scans. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 42236– 42245, 2026. 2

[7] Christophe Godin and Fred´ eric Boudon. Riemannian l-´ systems: modelling growing forms in curved spaces. Quantitative Plant Biology, 6:e38, 2025. 2

[8] Zane KJ Hartley, Lewis AG Stuart, Andrew P French, and Michael P Pound. Plantdreamer: Achieving realistic 3d plant models with diffusion-guided gaussian splatting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7069–7079, 2025. 1, 2

[9] Takahiro Isokane, Fumio Okura, Ayaka Ide, Yasuyuki Matsushita, and Yasushi Yagi. Probabilistic plant modeling via multi-view image-to-image translation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2906–2915, 2018. 2

[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 2

[11] Tahmina Khanam, Hamid Laga, Mohammed Bennamoun, Guanjin Wang, Ferdous Sohel, Farid Boussaid, Guan Wang, and Anuj Srivastava. A riemannian approach for spatiotemporal analysis and generation of 4d tree-shaped structures. In European Conference on Computer Vision, pages 326–341. Springer, 2024. 2, 6, 7, 8

[12] Daeyeoul Kim and Jinmo Kim. Procedural modeling and

visualization of multiple leaves. Multimedia Systems, 23(4): 435–449, 2017. 2

[13] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings ofthe IEEE/CVF international confer ence on computer vision, pages 4015–4026, 2023. 2

[14] Jae Joong Lee, Bosheng Li, and Bedrich Benes. Latent lsystems: Transformer-based tree generator. ACM Transactions on Graphics, 43(1):1–16, 2023. 2

[15] Bernd Lintermann and Oliver Deussen. Interactive modeling of plants. IEEE Computer Graphics and Applications, 19(1): 56–65, 1999. 2

[16] Zhihao Liu, Zhanglin Cheng, and Naoto Yokoya. Neural hi erarchical decomposition for single image plant modeling. Proceedings of the IEEE/CVF conference on computer vi sion and pattern recognition (CVPR), 2025. 2

[17] Shenglian Lu, Chunjiang Zhao, and Xinyu Guo. Venation skeleton-based modeling plant leaf wilting. International Journal of Computer Games Technology, 2009(1):890917, 2009. 2

[18] Radom´ır Mech and Przemyslaw Prusinkiewicz. Visual mod-ˇ els of plants interacting with their environment. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, pages 397–410, 1996. 2

[19] Lukas Meyer, Andreas Gilson, Oliver Scholz, and Marc Stamminger. Cherrypicker: Semantic skeletonization and topological reconstruction of cherry trees. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 6244–6253. IEEE, 2023. 3, 6

[20] Tommy Ojo, Thai La, Andrew Morton, and Ian Stavness. Splanting: 3d plant capture with gaussian splatting. In SIG-GRAPH Asia 2024 Technical Communications, pages 1–4. 2024. 1, 2

[21] Haolin Pan, Franck Hetroy-Wheeler, Julie Charlaix, and´ David Colliaux. Multi-scale space-time registration of growing plants. In 3DV, pages 310–319, 2021. 1, 2, 6, 7, 8

[22] JH Priestley and WH Pearsall. Growth studies: Ii. an interpretation of some growth-curves. Annals of Botany, 36(142): 239–249, 1922. 4, 5

[23] Przemyslaw Prusinkiewicz and Aristid Lindenmayer. The algorithmic beauty ofplants. Springer Science & Business Media, 2012. 2

[24] Przemyslaw Prusinkiewicz, Mark S Hammel, and Eric Mjolsness. Animation of plant development. In Proceedings of the 20th annual conference on Computer graphics and interactive techniques, pages 351–360, 1993. 2, 3

[25] Przemyslaw Prusinkiewicz, Jim Hanan, Mark Hammel, and Radomir Mech. L-systems: from the theory to visual models of plants. 1996. 3

[26] Przemyslaw Prusinkiewicz, Jim Hanan, and Radom´ır Mech.ˇ An l-system-based plant modeling language. In International workshop on applications of graph transformations with industrial relevance, pages 395–410. Springer, 1999. 2

[27] David Schunck, Federico Magistri, Radu Alexandru Rosu, Andre Cornelißen, Nived Chebrolu, Stefan Paulus, Jens´

Leon, Sven Behnke, Cyrill Stachniss, Heiner Kuhlmann,´ et al. Pheno4d: A spatio-temporal dataset of maize and tomato plant point clouds for phenotyping and advanced plant analysis. Plos one, 16(8):e0256340, 2021. 6

[28] Uri Shani and Dana H Ballard. Splines as embeddings for generalized cylinders. Computer Vision, Graphics, and Image Processing, 27(2):129–156, 1984. 6

[29] Zhongzhen Tang, Tianyou Jiang, Yongzhen Wang, and Xiaoyong Sun. Lidar: a new player in analyzing plant phenotypes. Trends in plant science, 29(12):1383–1384, 2024. 2

[30] Yinghua Wang, Songtao Hu, He Ren, Wanneng Yang, and Ruifang Zhai. 3dphenomvs: A low-cost 3d tomato phenotyping pipeline using 3d reconstruction point cloud based on multiview images. Agronomy, 12(8):1865, 2022. 2

[31] Yang Yang, Dongni Mao, Hiroaki Santo, Yasuyuki Matsushita, and Fumio Okura. Neuraleaf: Neural parametric leaf models with shape and deformation disentanglement. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 28167–28176, 2025. 2

[32] Yang Yang, Risa Shinoda, Hiroaki Santo, and Fumio Okura. Gaussianplant: Structure-aligned gaussian splatting for 3d reconstruction of plants. arXiv preprint arXiv:2512.14087, 2025. 1, 2

[33] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, et al. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):1–17, 2025. 6

[34] Daiwei Zhang, Joaquin Gajardo, Tomislav Medic, Isinsu Katircioglu, Mike Boss, Norbert Kirchgessner, Achim Walter, and Lukas Roth. Wheat3dgs: In-field 3d reconstruction, instance segmentation and phenotyping of wheat heads with gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5360–5370, 2025. 1, 2

[35] Yonglong Zhang, Yaling Xie, Jialuo Zhou, Xiangying Xu, and Minmin Miao. Cucumber seedling segmentation network based on a multiview geometric graph encoder from 3d point clouds. Plant Phenomics, 6:0254, 2024. 2