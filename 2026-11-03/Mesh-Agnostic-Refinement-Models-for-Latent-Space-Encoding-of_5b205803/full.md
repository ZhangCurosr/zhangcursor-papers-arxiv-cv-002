# Mesh-Agnostic Refinement Models for Latent Space Encoding of 3D Human Body Shapes

Chaima Zouaoui<sup>d,e,1</sup>, Jonathan Cerqueira<sup>b,1</sup>, Sylvain Arguillère<sup>a</sup>, Martin Bauer<sup>b</sup>, Nicolas Charon<sup>c</sup>, Mohamed Daoudi<sup>d,e</sup> and Emmanuel Hartman<sup>c</sup>

<sup>a</sup>Université de Lille, Département de Mathématiques, Bâtiment M2, Avenue Carl Gauss, Cité Scientifique, Villeneuve-d’Ascq, 59655, Nord, France

<sup>b</sup>Florida State University, Department of Mathematics, 208 Love Building, 1017 Academic Way, Tallahassee, 32306-4510, Florida, United States

<sup>c</sup>University ofHouston, Department ofMathematics, 3551 Cullen Blvd, PGH Room 641, Houston, 77204-3008, Texas, United States

<sup>d</sup>Univ. Lille, CNRS, Centrale Lille, Institut Mines-Télécom, UMR 9189 CRIStAL, F-59000 Lille, France

<sup>e</sup>IMT Nord Europe, Institut Mines-Télécom, Univ. Lille, Centrefor Dihal Systems, F-59000 Lille, France

## A R T I C L E I N F O

Keywords:   
SMPL model   
Latent-space correction   
Corrective deformation fields   
Elastic Shape Analysis   
Human body reconstruction   
Neural networks   
Clothed human scans

## A BS T RA C T

Parametric human body models such as the Skinned Multi-Person Linear (SMPL) model provide compact representations of human shape and pose, but their reconstructions remain afected by both errors in estimated latent parameters and limitations in the decoder’s expressive capacity. We propose a framework for refining these reconstructions by exploiting the geometric discrepancy between an initial decoded surface and a target mesh, without requiring vertex correspondences between them. The framework comprises three complementary approaches. Latent-VaRefine uses varifold-distance gradients to correct the estimated latent code while preserving the original parametric representation. Vertex-VaRefine processes multiscale varifold-gradient features with a DifusionNet backbone to predict vertex displacements in a canonical pose, yielding an approximately pose-equivariant refinement pipeline. ElaOpt directly optimizes surface deformations under elastic regularization and normal-direction constraints. On AMASS, Vertex-VaRefine reduces the mean vertex-to-vertex error of OmniFit reconstructions from 1.181 to 0.342 cm, a 71.0% reduction, while combining Latent-VaRefine with ElaOpt reduces the asymmetric Chamfer distance of ArtEq reconstructions from 1.681 to 0.717 cm, a 57.3% reduction. Evaluations with SMPL-H and SMPL-X demonstrate the framework’s adaptability to diferent body models, while additional experiments on remeshed targets and raw scans assess its robustness to changes in discretization and geometry beyond the parametric model. These results highlight the complementary strengths of latent-space correction, learned vertex refinement, and elastic optimization, ofering diferent trade-ofs between preserving the model parameterization and improving geometric fidelity. Our code is publicly available at https://github.com/chaimazouaoui 9/mesh-agnostic-body-refinement.

## 1. Introduction

The accurate capture of 3D deformable objects is commonly done through registration of a given target  (e.g. a triangular mesh produced by a 3D scanner) onto a 3D mesh with a fixed, canonical mesh topology. For human body shapes, this is most often done by relying on a parametric representation such as the Skinned Multi-Person Linear model (SMPL) (Loper, Mahmood, Romero, Pons-Moll and Black, 2015) and its extensions SMPL-H (Romero, Tzionas and Black, 2017) and SMPL-X (Pavlakos, Choutas, Ghorbani, Bolkart, Osman, Tzionas and Black, 2019). Such a model is given by a latent space $\mathcal { L } \ : = \ : \mathbb { R } ^ { N }$ and a decoder $\mathrm { D e c } ^ { \mathcal { L } } : \mathcal { L } \xrightarrow { } \mathbb { R } ^ { V \times 3 }$ which, to a given latent code, associates a triangular mesh with � vertices. Registration of  then becomes equivalent to code retrieval, i.e., to finding the latent code whose decoded mesh best represents the target.

In recent years, many methods for latent code retrieval have been proposed. Optimization approaches such as (Bogo, Kanazawa, Lassner, Gehler, Romero and Black, 2016; Pavlakos et al., 2019; Kanazawa, Black, Jacobs and Malik, 2018; Feng, Choutas, Bolkart, Tzionas and Black, 2021) require heavy user supervision. Learning-based methods, on the other hand, have recently made notable breakthroughs (Bhatnagar, Sminchisescu, Theobalt and Pons-Moll, 2020). Stateof-the-art frameworks based on equivariant networks such as ArtEq (Feng, Kulits, Liu, Black and Abrevaya, 2023) have been shown to generalize well to unseen poses. Despite these advances, reconstruction errors can arise from two distinct sources. First, the retrieval algorithm may fail to identify the parameters that best explain the observed target, even when its geometry is representable by the model. Second, the target may contain geometric details that lie outside the decoder’s representational capacity. These two sources of error call for complementary corrections: refining the latent code addresses estimation errors, whereas recovering geometry beyond the model requires additional surface deformations.

However, a limitation of all latent code retrieval methods is their inability to retrieve details not representable by the latent space, such as raw scans, or meshes from other, incompatible databases. Hence, a natural question is how to account for these various extra details without losing advantages intrinsic to the parametric model, such as the possibility of animating a human shape, once registered onto the model, through some evolution in the pose parameters of the parametric model. A suficiently accurate initial reconstruction provides more than an estimate of the target: it also supplies a structured reference on which the remaining geometric discrepancy can be represented. This motivates a residual refinement strategy that uses the reconstructed surface as a target-adapted reference. The central challenge is to encode and exploit this discrepancy without requiring vertex correspondences between the source and target, whose sampling and connectivity may difer. A further challenge for learned corrections is to account for articulation, so that similar geometric discrepancies can be processed consistently across diferent body poses.

![](images/23d895878db98270755d9fb3ae8162c1eae13ecd237afe811f674fbe2fb2ea54.jpg)  
Figure 1: Overview of our refinement framework. (1) Given a target mesh  of arbitrary topology, (2) an existing encoder provides an initial reconstruction, the source mesh  = Dec(�), with the fixed topology of the body model. (3) Our three methods refine this reconstruction: Latent-VaRefine corrects the latent code, Vertex-VaRefine predicts a per-vertex correction in the canonica T-pose that is then re-posed by linear blend skinning, and ElaOpt deforms the mesh vertices by elastic optimization.

In this work, we introduce a collection of complementary tools that can be integrated with any latent code retrieval algorithm that produces a decoded surface  = Dec(�) approximating a target shape  within a chosen latent representation. Our approaches exploit the geometric proximity between  and  to estimate corrections, either in the latent space or directly on the vertices, leading to a corrected surface <sup>̂</sup> capable of recovering finer geometric details of . Central to our framework are mesh-agnostic representations of the discrepancies between  and , which allow the correction process to train in a way that remains independent of the topology of the target mesh. For the learningbased approaches, we encode the source–target discrepancy through varifold-distance gradients, evaluated with respect to either latent parameters or surface vertices. For vertexwise prediction, we additionally express these features in a canonical pose and re-pose the predicted correction, encouraging approximate consistency across articulations. The optimization-based approach instead fits the target directly through a geometric objective regularized by elastic deformation energies. Together, these mechanisms address diferent requirements: preserving the original parametric representation, learning transferable surface corrections, or refining an individual reconstruction without training. Our three main contributions are the following:

• Latent-VaRefine: a learning-based method that corrects the latent code $C \in { \mathcal { L } }$ directly in the latent space, yielding a refined code $C + \Delta _ { C }$

• Vertex-VaRefine: an equivariant, learning-based method that predicts a per-vertex displacement field Δ ∈ $\mathbb { R } ^ { V \times \hat { 3 } }$ , in which the mesh-agnostic representation is used directly as a per-vertex signal and processed by DifusionNet.

• ElaOpt: an alternative optimization-based strategy relying on a simple elastic energy formulation that deforms  toward  while discouraging unrealistic distortions and retaining reposability.

A schematic description of these three methods can be seen in Fig. 1. They may be used individually or in concert, depending on the objective: if retaining a low-dimensional latent code representation is key, then Latent-VaRefine alone may be best. If, instead, simply acquiring accurate, low-noise meshes is the objective, then Vertex-VaRefine may be more practical. Combinations of the methods may also be useful. For example, as will be seen in our results, the strongest improvement in various error measures came from first applying Latent-VaRefine to obtain a better starting point for optimization and then using ElaOpt. These alternatives expose a practical trade-of between preserving the original model parameterization and increasing geometric flexibility. Latent-space refinement retains the decoder’s pose and shape parameterization, whereas vertex-wise refinement introduces additional degrees of freedom to better approximate the target surface. The latter retains the source mesh connectivity and can reuse its skinning structure, but the corrected surface need not correspond to a latent code of the original model. The appropriate refinement therefore depends on the representation required by downstream applications, as well as on reconstruction accuracy.

We evaluate the proposed refinement methods, individually and in combination, on AMASS using ArtEq and OmniFit as initial reconstruction methods. Vertex-VaRefine reduces the mean vertex-to-vertex error of OmniFit reconstructions from 1.181 to 0.342 cm, corresponding to a 71.0% reduction. Starting from ArtEq reconstructions, combining Latent-VaRefine with ElaOpt reduces the asymmetric Chamfer distance from 1.681 to 0.717 cm, a 57.3% reduction. These results illustrate the complementary strengths of learned vertex refinement for correspondence accuracy and latent-space refinement followed by elastic optimization for surface fidelity. Further evaluations with SMPL-H and SMPL-X demonstrate the adaptability of the framework to diferent body models, while experiments on remeshed targets and raw scans assess its robustness to changes in target discretization and its ability to refine geometry beyond the parametric model.

## 2. Background and Related Work

## 2.1. Human Body Latent Spaces

Statistical parametric body models constitute the dominant paradigm for low-dimensional human body representation. SMPL (Loper et al., 2015) introduced a skinned linear representation where shape coeficients � are obtained by PCA over a population of registered scans, while pose coeficients � encode joint-angle rotations driving a linear blend skinning deformation. SMPL was later extended to SMPL-H (Romero et al., 2017) for articulated hands and further to SMPL-X (Pavlakos et al., 2019) for unified body, hand, and facial expression modeling. This was made more sparse in STAR (Osman, Bolkart and Black, 2020). While these parametric models provide a compact and diferentiable parameterization, the joint-angle pose space � is itself unconstrained: arbitrary parameter values may correspond to implausible meshes. To address this, learned priors over the pose manifold have been proposed, including the VAEbased VPoser (Pavlakos et al., 2019), the GAN-based adversarial pose prior of Davydov, Remizova, Constantin, Honari, Salzmann and Fua (2022), and the neural distance field formulation Pose-NDF (Tiwari, Antic, Lenssen, Sarafianos, Tung and Pons-Moll, 2022). Yet, the expressive power of the latent space in these approaches remains limited by the model’s parameterization, and any geometry lying outside this manifold, fine surface detail, and clothing cannot be recovered through latent codes alone. In the present work, we will largely make use of the base SMPL model, which is described in more details below. The methods we propose are, however, not specific to SMPL and apply to any latentspace body model, such as SMPL-H and SMPL-X (see Section 10).

The SMPL, SMPL-H and SMPL-X Latent Spaces and Decoders. The Skinned Multi-Person Linear model (SMPL), as introduced in (Loper et al., 2015) and its extensions SMPL-H (Romero et al., 2017) and SMPL-X (Pavlakos et al., 2019) share the same construction. Each consists of latent codes $C ~ = ~ ( \beta , \pmb \theta ) ~ \in ~ \mathbb R ^ { | \beta | } \times \mathbb R ^ { K \times 3 }$ and a decoder that maps any such code to the positions of the � vertices of a mesh, i.e. an element of $\mathbb { R } ^ { \bar { V } \times 3 }$ , where the number of vertices � is fixed for each model $( V ~ = ~ 6 8 9 0$ for SMPL and SMPL-H, and $V \ = \ 1 0 4 7 5$ for SMPL-X). The $\beta$ are shape parameters, the number of which, $| \beta |$ , can range up to 300 and is commonly set to 10. The pose parameter vector $\theta$ encodes the relative rotation of each of the � articulated joints relative to its parent in the kinematic tree, in axis-angle representation, with $K = 2 3$ for SMPL, $K = 5 1$ for SMPL-H (articulated fingers) and $K = 5 4$ for SMPL-X (additional jaw and eye joints). The model uses PCA to account for variations in body shape, and Linear Blend Skinning (LBS), $W ( \cdot )$ , to pose the mesh. We denote the resulting maps from latent codes to meshes by $\mathrm { D e c } ^ { \mathrm { S M P L } } ( \cdot ) , \ \mathrm { D e c } ^ { \mathrm { S M P L - H } } ( \cdot )$ and $\mathrm { D e c } ^ { \mathrm { S M P L - X } } ( \cdot )$ , respectively; it is computed in three steps.

A rest-pose mesh is first produced by

$$
\mathbf { M } ( C ) = \mathbf { M } ( \beta , \theta ) = \bar { \mathbf { M } } + B _ { S } ( \beta ) + B _ { P } ( \theta ) ,\tag{1}
$$

where $\bar { \mathbf { M } } \in \mathbb { R } ^ { V \times 3 }$ is the template mesh, $B _ { S } ( \beta )$ is the linear transformation from shape parameters $\beta$ to shape displacements, and $B _ { P } ( \theta )$ is the linear transformation from pose parameters to blendshape correctives that account for softtissue deformations due to pose. Next, joint locations are obtained for the rest-pose mesh via the linear joint regressor $J ( \pmb { \beta } ) : \mathbb { R } ^ { | \pmb { \beta } | }  \mathbb { R } ^ { ( K + 1 ) \times 3 } , J ( \pmb { \beta } ) = ( t _ { 0 } , \dots , t _ { K } )$ , where $t _ { 0 }$ is the root joint. Finally, Linear Blend Skinning (LBS), denoted $W ( \cdot )$ , is applied to the rest-pose mesh with skinning weights  to obtain the output vertices. For SMPL, for instance, this gives:

$$
\mathrm { D e c } ^ { \mathrm { S M P L } } ( C ) = \mathrm { D e c } ^ { \mathrm { S M P L } } ( \beta , \theta ) = W ( \mathbf { M } ( \beta , \theta ) , J ( \beta ) , \theta , \mathcal { W } ) .\tag{2}
$$

and $\mathrm { D e c } ^ { \mathrm { S M P L - H } }$ and $\mathrm { D e c } ^ { \mathrm { S M P L - X } }$ are obtained analogously with their own template, blendshapes, joint regressor and skinning weights.

## 2.2. Latent Code Retrieval

Given a target mesh or point cloud, retrieving the latent code that best reconstructs it is a non-trivial inverse problem. Classical optimization-based fitters such as SM-PLify (Bogo et al., 2016) and SMPLify-X (Pavlakos et al., 2019) minimize a 2D/3D keypoint-based energy regularized by a pose prior. Regression-based approaches such as HMR (Kanazawa et al., 2018) and PIXIE (Feng et al., 2021) predict SMPL latent codes in a single forward pass from images, while IPNet (Bhatnagar et al., 2020) regresses SMPL latent codes from point clouds via implicit part segmentation. Recently, ArtEq (Feng et al., 2023) introduced an articulated SE(3)-equivariant architecture that generalizes the SMPL fitting to unseen poses by enforcing equivariance in the encoder. VariShaPE (Hartman, Charon and Bauer, 2025) proposed an alternative approach via the VariGrad descriptor, a vertex-wise feature that is robust to changes in mesh connectivity and triangulation from (Hartman and Pierson, 2023). A common limitation persists across these encoders in that they are constrained by the expressive capacity of the underlying body model. The recovered latent code can only describe geometry that lies within the manifold spanned by the decoder. Our work takes a fixed, chosen encoders’ outputs as input and introduces a learned correction module that recovers the residual geometry left out by the latent decoder.

## 3. Problem Formulation and Setup

In this section we will present a rigorous mathematical formulation of the latent code recovery problem.

## 3.1. Latent Code Recovery

Consider some target mesh  for which one would like to identify the latent code  in some latent space $\mathcal { L } = \mathbb { R } ^ { N }$ so that the decoded shape $S = \mathrm { D e c } ^ { \mathcal { L } } ( C )$ will best approximate . Note that, by construction,  has a fixed mesh topology that is typically determined by the template surface from the decoder, e.g. �<sup>̄</sup> for the SMPL model in Eq. (1). This generally prevents, outside of artificial examples, the reconstructed  to match  exactly and often makes the estimation of the optimal latent code a dificult problem to solve in practice. Further, due to the expressive limits of the latent space and decoder, retaining fine details from the target mesh is typically impossible, even with optimal recovery. Our goal is thus to propose diferent general strategies for refining estimated latent codes or their corresponding meshes.

## 3.2. Setup

In the rest of the paper, the target mesh, which represents the shape to be registered on by the model will always be denoted . Our second input is a source mesh $S = \mathrm { D e c } ^ { \mathcal { L } } ( C )$ where $C \in { \mathcal { L } }$ is some latent code in the chosen latent-space $\mathcal { L } = \mathbb { R } ^ { N }$ and obtained by some (possibly suboptimal) latent code estimation method. The mesh structure of the source is prescribed by the given decoder. In most of our experiments, we focus specifically on the SMPL latent-space and its decoder $\mathrm { D e c } ^ { \mathrm { S M P L } }$ . Our methods are, however, not specific to SMPL, (theoretically, nor are they even restricted to latent spaces representing human body shapes); to illustrate this, we also apply them to two other latent-space models, SMPL-H and SMPL-X (Section 10). Our only base assumption is that  and  remain suficiently “close” to one-another (for example, in terms of Chamfer distance). In most of our experiments, � is obtained by applying ArtEq (Feng et al., 2023), a state-of-the-art latent code retrieval method, to .

We propose several strategies which may be combined or used independently to bridge the remaining gap between  and : (1) a method that corrects the estimated latent code �, and (2) two methods that operate directly on the vertices. Among those, deep learning based methods will require an approach for consistently encoding the relationship between the source S and the target shape T. Indeed, unlike S, T is typically not assumed to have a fixed mesh structure. To address this, we shall rely on VariGrad, an invariant mesh representation framework based on geometric measure theory, which we introduce in the next section.

## 4. The VariGrad representation

We now recall the definition of the varifold distance and its gradient, VariGrad, which will be an essential ingredient for our learning based approaches. We also explain its equivariant properties and deduce an approximately pose-invariant version of VariGrad. As shown in Feng et al. (2023), this property allows for better generalization of learned methods, which also translate into requiring a smaller training sample. Indeed, theoretically, with perfect pose equivariance, only training on surfaces with a neutral pose would be required.

## 4.1. Varifold distance

The family of reproducing kernel varifold distances originates from the representation of shapes, in particular surfaces, as a specific type of geometric measures known as varifolds. We refer to Kaltenmark, Charlier and Charon (2017) for a more complete introduction to these metrics. The key feature of this distance is its (quasi-)invariance to resampling/re-meshing. The following is only a brief description of how they are computed in practice.

For a mesh �, we first compute, for each face $k \mathbf { \Sigma } =$ $1 , \ldots , F .$ , its barycenter $c _ { k } \in \mathbb { R } ^ { 3 }$ , its oriented unit normal vector $N _ { k } \in \mathbb S ^ { 2 }$ and its area $r _ { k } > 0$ . Similarly, for a second mesh �<sup>′</sup> we denote by $c _ { \ell } ^ { \prime } , N _ { \ell } ^ { \ i }$ and $r _ { \ell } ^ { \prime }$ the barycenters, unit normals and areas of its faces $\ell = 1 , \ldots , F ^ { \prime }$ . The family of varifold distances we consider in the work are defined as follows:

$$
\begin{array} { l } { { \displaystyle { d _ { \sigma } ^ { \mathrm { V a r } } ( \mathbf { V } , \mathbf { V } ^ { \prime } ) ^ { 2 } = \sum _ { k = 1 } ^ { F } \sum _ { k ^ { \prime } = 1 } ^ { F } G _ { \sigma } ( \| c _ { k } - c _ { k ^ { \prime } } \| ) \Gamma ( N _ { k } \cdot N _ { k ^ { \prime } } ) r _ { k } r _ { k ^ { \prime } } } } } \\ { { \displaystyle ~ + \sum _ { \ell = 1 } ^ { F ^ { \prime } } \sum _ { \ell ^ { \prime } = 1 } ^ { F ^ { \prime } } G _ { \sigma } ( \| c _ { \ell } ^ { \prime } - c _ { \ell ^ { \prime } } ^ { \prime } \| ) \Gamma ( N _ { \ell } ^ { \prime } \cdot N _ { \ell ^ { \prime } } ^ { \prime } ) r _ { \ell ^ { \prime } } ^ { \prime } r _ { \ell ^ { \prime } } ^ { \prime } } }  \\ { { \displaystyle ~ - 2 \sum _ { k = 1 } ^ { F } \sum _ { \ell = 1 } ^ { F ^ { \prime } } G _ { \sigma } ( \| c _ { k } - c _ { \ell } ^ { \prime } \| ) \Gamma ( N _ { k } \cdot N _ { \ell } ^ { \prime } ) r _ { k } r _ { \ell } ^ { \prime } ~ ( 3 ) } } \end{array}
$$

where $G _ { \sigma } : \mathbb { R } _ { \ge 0 } \to$ ℝ is a spatial kernel, taken throughout to be the Gaussian kernel of width � given by $G _ { \sigma } ( t ) = e ^ { - t ^ { 2 } / \sigma ^ { 2 } }$ and $\Gamma : [ - 1 , 1 ] \to \mathbb { R }$ is an orientation kernel. Taking in Eq. (3), as an orientation kernel, $\Gamma ( N \cdot N ^ { \prime } ) = e ^ { ( N \cdot N ^ { \prime } - 1 ) / \tau ^ { 2 } }$ for some angular width $\tau > 0$ yields the oriented varifold distance. This orientation kernel distinguishes aligned from anti-aligned normals. Both the Latent-VaRefine model of Section 5 and the multi-scale VariGrad features of Vertex-VaRefine (Section 6.1) rely on this oriented formulation.

## 4.2. VariGrad

As explained above, we assume that every source mesh  produced by any latent-space decoder has a fixed, canonical topology. In contrast, we allow the target mesh, as it is typically the output of some scanning acquisition procedure, to have both arbitrary connectivity and an arbitrary number of vertices. The VariGrad framework of Hartman and Pierson (2023) proposes to represent  via the gradient of the above varifold distance. More specifically, the VariGrad representation of the target mesh  with respect to the source S is the vector field (on S) denoted

$$
v \doteq \mathrm { V a r i G r a d } _ { \sigma } ( S , T ) \in \mathbb { R } ^ { V \times 3 }
$$

and defined as the gradient of the squared varifold distance between S and T with respect to the vertices $x _ { 1 } , \ldots , x _ { V }$ of . In other words,

$$
v = ( v _ { j } ) \in \mathbb { R } ^ { V \times 3 } \mathrm { ~ w i t h : ~ } v _ { j } = \nabla _ { x _ { j } } d _ { \sigma } ^ { \mathrm { V a r } } ( S , T ) ^ { 2 } .
$$

which we interpret as a vector field on the reference mesh $S .$ $\operatorname { A s } d _ { \sigma } ^ { \operatorname { V a r } }$ is (quasi-)invariant to resampling and re-meshing of $\tau ,$ so is $\operatorname { V a r i G r a d } _ { \sigma } ( S , \mathcal { T } )$

In the present work, we will take advantage of diferent variants of the original VariGrad model from Hartman and Pierson (2023) described above. In Section 5, following the VariShaPE model developed in (Hartman et al., 2025), we shall consider VariGrad representations taken with respect to the latent space coordinate basis of SMPL at � (where ${ \boldsymbol { S } } =$ $\mathrm { D e c } ^ { \mathrm { S M P L } } ( \dot { C } ) )$ , yielding a lower-dimensional representations of $\tau$ . Second, in Section 6.1, we leverage a multi-scale extension $v = ( v ^ { \sigma _ { 1 } } , \ldots , v ^ { \sigma _ { \ell } } ) \in \mathbb { R } ^ { V \times 3 \times \ell }$ that concatenates several VariGrads with respect to a common reference  but with various diferent spatial scales $\sigma _ { j } .$ , which better captures both coarse body proportions and finer features.

## 4.3. Equivariance to Rigid Motions

Let $g = ( R , t ) \in \mathrm { S E } ( 3 )$ be a rigid motion, with rotation � in SO(3) and translation � in $\mathbb { R } ^ { \breve { 3 } }$ , and have � act on any mesh  with vertices $( x _ { i } ) _ { i = 1 } ^ { V } \mathrm { b y } g \cdot \mathcal { M } \doteq ( R x _ { i } + t ) _ { i = 1 } ^ { V }$ . That is, it acts as a rigid motion of the vertices that leaves the topology (faces and connectivity) unchanged. Then for the �-th face its barycenter $c _ { k } ^ { \mathcal { M } }$ , unit normal $\grave { N } _ { k } ^ { \mathcal { M } }$ , and area $r _ { k } ^ { \mathcal { M } }$ are transformed under $g$ so that

$$
\left\{ \begin{array} { l l } { c _ { k } ^ { g \cdot \mathcal { M } } = g ( c _ { k } ^ { \mathcal { M } } ) = R c _ { k } ^ { \mathcal { M } } + t , } \\ { N _ { k } ^ { g \cdot \mathcal { M } } = R N _ { k } ^ { \mathcal { M } } , } \\ { r _ { k } ^ { g \cdot \mathcal { M } } = r _ { k } ^ { \mathcal { M } } . } \end{array} \right.\tag{4}
$$

The varifold distance between two meshes  and  is known (see Kaltenmark et al. (2017)) to be invariant to the action of a joint rigid motion on the two meshes, that is,

$$
d _ { \sigma } ^ { \operatorname { V a r } } ( g \cdot S , g \cdot \mathcal { T } ) = d _ { \sigma } ^ { \operatorname { V a r } } ( S , \mathcal { T } ) , \qquad \forall g \in \operatorname { S E } ( 3 ) .\tag{5}
$$

Diferentiating identity (5) with respect to the vertex coordinates $x _ { j }$ of , and using the chain rule for the action $x _ { j } \ \mapsto \ g ( \bar { x _ { j } } ) \ = \ R x _ { j } + t .$ shows that the gradient field inherits an equivariant transformation law. Writing $v =$ $\operatorname { V a r i G r a d } _ { \sigma } ( T ; S )$ and $v ^ { \prime } = \mathrm { V a r i G r a d } _ { \sigma } ( g \cdot \mathcal { T } ; g \cdot S )$ , we obtain

$$
v _ { j } ^ { \prime } = R v _ { j } , \qquad j = 1 , \ldots , V ,\tag{6}
$$

i.e., the VariGrad vector attached to the transformed vertex $R x _ { i } + t$ is exactly the original VariGrad vector at $x _ { j }$ , rotated by $\dot { \boldsymbol { R } }$ — the translation drops out, as expected for the gradient of a translation-invariant energy.

## 4.4. Approximate Pose Invariant Representation

We first heuristically argue that VariGrad is also approximately equivariant to a common change of pose. For example, consider two surfaces $\bar { S }$ and <sup>̄</sup> corresponding to the neutral pose of some human body model. Now, bend the left arm for each surface, obtaining posed surfaces  and . For most points on the left arm of either surface, (say, $\bar { \boldsymbol { x } } _ { i } \in \bar { \mathcal { S } }$ and $\bar { y } _ { j } \in \bar { \mathcal { T } }$ respectively) this corresponds to applying a rigid transformation, say $g = ( R , t ) \colon x _ { j } = R \bar { x } _ { j } + t .$ $y _ { j } = R { \bar { y } } _ { j } + t .$ Then, the Gaussian factors that appear in the computation of $\bar { v } = \mathrm { V a r i G r a d } _ { \sigma } ( \bar { S } , \bar { \tau } )$ mean that its value $\bar { v } _ { j }$ at a vertex $\bar { x } _ { j }$ depends mainly on points located in a close neighborhood of this vertex, all of which are transformed by $g .$ Combining this fact with the global equivariance of VariGrad to common rigid transformations, this leads us to assert that VariGrad components attached to those points should undergo approximately the same rotation: if $v =$ $\operatorname { V a r i G r a d } _ { \sigma } ( S , \mathcal { T } )$ , then $v _ { j } \simeq R \bar { v } _ { j }$

In the case where the pose of $S = \mathrm { D e c } ^ { \mathcal { L } } ( \beta , \theta )$ is given by joint rotations and Linear Blend Skinning (as is the case for SMPL and its variants), the rigid transformation $g ~ = ~ g _ { j } ( \pmb \theta ) ~ = ~ ( R _ { j } ( \pmb \theta ) , t _ { j } ( \pmb \theta ) )$ is determined by the pose coeficients, so we can write $v _ { j } ~ \simeq ~ R _ { j } ( \theta ) \bar { v } _ { j }$ . This leads us to define the approximate pose-invariant version of the VariGrad representation $\phi \in \mathbb { R } ^ { V \times 3 }$ by projecting the raw representation back into the frame of the neutral pose (which we call canonical coordinates):

$$
{ \boldsymbol { \phi } } _ { j } ~ \doteq ~ R _ { j } ( { \boldsymbol { \theta } } ) ^ { \intercal } { \boldsymbol { v } } _ { j } \left( \simeq R _ { j } ( { \boldsymbol { \theta } } ) ^ { \intercal } R _ { j } ( { \boldsymbol { \theta } } ) { \boldsymbol { \bar { v } } } _ { j } = { \boldsymbol { \bar { v } } } _ { j } \right) .\tag{7}
$$

On those vertices that are moved by a locally rigid transformation during a pose change, $\dot { \phi _ { j } } \simeq \mathrm { V a r i G r a } \dot { { \bf d } } _ { \sigma } ( \bar { S } , \bar { T } )$ Applying this reasoning to any common pose change in both source and target, we obtain with the projection $\phi$ an approximate pose invariant representation of  onto .

To confirm this heuristic reasoning, we numerically validate this approximate pose-invariance in Table 1. We select 100 real source and target pairs from AMASS and compute their representation $\mathrm { V a r i G r a d } _ { \sigma } ^ { 0 }$ at the reference pose $\begin{array} { r } { \theta = \mathbf { 0 } , } \end{array}$ obtaining reference values for comparison. We then apply a common pose change to both the source and the target, rotating a random number of joints (up to 10) by a random angle drawn from $[ 1 0 ^ { \circ } , 2 0 ^ { \circ } ] , [ 1 0 ^ { \circ } , 4 0 ^ { \circ } ] \mathrm { o r } [ 1 0 ^ { \circ } , 6 0 ^ { \circ } ]$ Finally, we compute the raw representation VariGrad of each new target pose onto their respective new source pose, and its projection $\phi$ onto canonical coordinates (Eq. (7)). We then compute both of their per-vertex relative error with respect to our reference values. We find that the median pervertex relative error of $\phi$ remains below 10% for all tested intervals for the approximate invariant representation, and is about four times smaller than that of the raw VariGrad. The mean error is noticeably larger than the median. This reflects a minority of vertices located in the blending regions

![](images/5bcb05553d5f4ed3475cfb7c894d59985e244f16c9b6e1051b4db65f6a35100e.jpg)  
Figure 2: Overview of Latent-VaRefine: Given a source mesh of fixed SMPL topology  with coeficients � and a target mesh of arbitrary topology , we encode the reconstruction error by the low-dimensional VariGrad � with respect to the coeficients � (as defined in the last paragraph of Section 4.2). The pair $( C , v )$ is then used to find $\Delta _ { C }$ via a fully connected MLP. The coeficient correction $\Delta _ { C }$ is then added to the initial coeficients � to produce a latent code $C + \Delta C$ yielding a better mesh reconstruction.

Numerical validation of the approximate pose-invariance of the canonical-coordinate Varigrad representation VariGrad<sup>0</sup> (Eq. (7)), on 100 real AMASS source/target pairs. For each pair, a random number of joints (between 1 and 10) are rotated by a random angle within the given interval, identically for both meshes. We report the per-vertex relative error (%) between the VariGrad representation recomputed at this new pose and its value at the reference pose $\theta \ = \ \mathbf { 0 }$ , for both the approximately invariant and raw versions of VariGrad. The reduction is the ratio of the raw median error to the median error between the two versions.

<table><tr><td>Angle interval (deg)</td><td>Invariant proj. mean median</td><td>Raw VariGrad mean median</td><td></td><td>Reduction (median)</td></tr><tr><td>[10,20]</td><td>22.99 4.37</td><td>32.84</td><td>17.99</td><td>x4.12</td></tr><tr><td>[10,40]</td><td>37.29 7.17</td><td>53.44</td><td>27.89</td><td>×3.89</td></tr><tr><td>[10,60]</td><td>49.62 9.98</td><td>71.66</td><td>38.50</td><td>×3.86</td></tr></table>

between joints, where linear blend skinning mixes several joint transformations and the pose-corrective blendshapes introduce non-rigid deformations that a single local rotation per vertex cannot capture.

## 5. Latent-VaRefine: Latent Space Correction via Adaptive VariGrads

In this section, we introduce Latent-VaRefine, a learning based latent space correction framework that refines an initial latent code using a VariGrad representation. More specifically, given a target mesh  and an initial latent code $C \in \mathcal { L } = \mathbb { R } ^ { N }$ , our goal is to estimate a correction vector $\Delta _ { C } ~ \in ~ \mathbb { R } ^ { N }$ such that the refined latent space coeficients $\hat { C } = C + \Delta _ { C }$ generate a mesh $\hat { S } = \mathrm { D e c } ^ { \mathcal { L } } ( C + \Delta _ { C } )$ that more accurately approximates the target mesh .

Our approach is inspired by the VariShaPE framework proposed in (Hartman et al., 2025), which employs the

VariGrad representation with respect to the latent-space coordinate basis and a fixed reference mesh  to derive a fixeddimensional representation of unregistered meshes. This representation can subsequently serve as input to a neural network for learning an embedding into the latent space. A limitation of this representation arises when the target shape difers significantly from the reference mesh, in which case the VariGrad representation may produce suboptimal results. The main idea behind Latent-VaRefine is therefore to apply this approach as a refinement stage on top of an existing latent space encoder. In the examples presented later in Section 7, we will use the SMPL representation for human body shapes together with ArtEq (Feng et al., 2023) as the primary encoder/code retriever, but the framework is equally compatible with any alternative encoder.

This two-step process involves the computation of the VariGrad representation relative to the surface $S = \mathrm { D e c } ^ { \mathcal { L } } ( C )$ generated by the latent space decoder using the given estimate rather than with respect to a fixed template mesh. Assuming  is a reasonably accurate estimate of the target , the anchor point underlying the VariGrad representation is already close to the target mesh. We subsequently use both the latent code estimate � and the corresponding VariGrad representation � as input to a neural network that learns a correction to the estimated latent code, as illustrated in Figure 2. To ensure that Latent-VaRefine does not produce a worse mesh than we began with, we use a symmetric chamfer measurement to decide whether to keep the resulting refined latent code or return to the original latent code and its corresponding mesh. While inference with the proposed pipeline can be performed on arbitrary unregistered meshes, training the network requires data with ground-truth latent space coeficients. In our experiments we use data from the AMASS dataset, for which we have access to those ground truth SMPL coeficients.

This type of training data enables us to train the network using a loss function that combines the mean vertex error between the corrected mesh and the ground-truth mesh with a latent-space error term, i.e., the discrepancy between the estimated latent-space parameters and the corresponding ground-truth parameters.

## 6. Vertex-wise Correction

In this section, we present two diferent Vertex Correction Frameworks that refine an initial source mesh ${ \textbf { \textit { S } } } =$ $\mathrm { D e c } ^ { \mathcal { L } } ( C )$ reconstructed from a latent code estimate � by using a geometry-aware deformation field to better match the target mesh $\tau .$ . Unlike the previous latent space correction scheme, the main objective is here to allow the reconstruction of retrieve geometric details outside of the latent model. First, we introduce Vertex-VaRefine, a learningbased approach based on an equivariant neural network model. Second, we consider an alternative optimizationbased approach involving elastic energies and second-order surface metrics as regularizers. In the following, we denote by $( x _ { 1 } , \hdots , x _ { V } ) \in \mathbb { R } ^ { \breve { V } \times 3 }$ the vertices of the source mesh $S .$ The objective is to predict a per-vertex displacement field that we shall write $\bar { \mathbf { \Delta } } \in \mathbb { R } ^ { V \times \bar { 3 } }$ so that the refined mesh ${ \hat { S } } ,$ with vertices $\hat { x } _ { i } = x _ { i } + \Delta _ { i }$ , results in a reconstruction closer to $\tau$

## 6.1. Vertex-VaRefine: An Equivariant Learning Based Approach for Vertex Correction

Figure 3 illustrates the overall architecture of our proposed deformation refinement framework. It combines three key components:

Multiscale VariGrad:. We choose a set of scales $\sigma _ { 1 } , \ldots , \sigma _ { \ell }$ and compute, for each $\sigma _ { j } .$ , the VariGrad descriptors

$$
v _ { j } = \operatorname { V a r i G r a d } _ { \sigma _ { j } } ( \mathcal T ; S ) = ( v _ { j , 1 } , \dots , v _ { j , V } ) \in \mathbb { R } ^ { V \times 3 }
$$

from the source to the target (see Section 4.2), which encode the geometric diferences between the two shapes. We obtain a total of $3 \times \ell$ features on each vertex by concatenation: $\boldsymbol { v } = ( v _ { 1 } , \dots , v _ { \ell } ) \in \mathbb { R } ^ { V \times 3 \times \ell }$

Once the source–target residual � is encoded vertexwise, we need a network that can propagate information across the surface while respecting its intrinsic geometry. Standard MLPs treat vertices independently and thus ignore neighborhood structure, while spectral graph convolutions are computationally expensive and basis-dependent.

We chose DifusionNet (Sharp, Attaiki, Crane and Ovsjanikov, 2022) as the backbone that propagates the encoded discrepancy across the surface, but other options such as a SpiralNet++ (Gong, Chen, Bronstein and Zafeiriou, 2019) encoder, or PoissonNet (Maesumi, Makadia, Groueix, Kim, Ritchie and Aigerman, 2025), could also be valid alternatives (see Section 9.3 and Appendix C).

DifusionNet. DifusionNet is a general-purpose architecture for deep learning on 3D surfaces, introduced by Sharp et al. (2022). Its central idea is that a simple learned difusion layer provides an efective mechanism for spatial communication across a surface: at each block, per-vertex features are difused for a learned time scale, so that the spatial support is optimized as a continuous network parameter ranging from purely local to fully global, removing the need to handpick neighborhood sizes. Each DifusionNet block combines this difusion step with spatial gradient features that support directional (anisotropic) filters and a spatially shared pointwise MLP applied independently at every vertex, and several identical blocks are stacked to form the network. Because difusion is expressed through intrinsic surface operators (the Laplace–Beltrami operator and mass matrix), which can be discretized on triangle meshes as well as on point clouds (Sharp and Crane, 2020), the resulting networks are robust to changes in resolution and sampling of a surface. Rather than overfitting to a particular mesh connectivity, the model learns a representation-agnostic solution and can even be trained on one representation, such as a triangle mesh, and evaluated on another, such as a point cloud. This is achieved without any explicit surface convolutions or pooling hierarchies, which makes the architecture simple, eficient, and well suited to our setting.

Pose-Equivariant Prediction and Re-posing. Rather than predicting the correction directly on the posed source mesh, we predict it on the canonical neutral pose $( \theta = 0 )$ shape $\bar { \cal S }$ obtained from the SMPL shape blendshapes of , and recover the final displacement by re-posing through linear blend skinning. Concretely, the VariGrad field � computed on the posed source $s$ is first projected onto the per-vertex rest-pose frame $R _ { j } ( \theta )$ obtained from linear blend skinning (Eq. (7)), yielding a feature field ${ \boldsymbol { \phi } } = ( \phi _ { 1 } , \ldots , \phi _ { V } )$ that ${ \mathrm { i s } } ,$ as discussed in Section 4.4, approximately invariant to the pose �. The backbone (here, DifusionNet) processes $\phi$ into a corrected neutral pose shape $\mathbf { M } = \bar { \cal { S } } + { \cal { D } } .$ , and the final prediction is obtained by re-posing � with the same pose parameters � from the source shape , in other words:

$$
\hat { S } = \mathrm { L B S } ( \mathbf { M } , \pmb { \theta } ) = \mathrm { L B S } ( \bar { S } + D , \pmb { \theta } ) .\tag{8}
$$

Since both $\phi$ and � live in the canonical T-pose frame, and pose is only reintroduced via the fixed LBS skinning map, the correction pipeline is by construction approximately pose-equivariant: applying it to the same subject in a diferent pose $\theta ^ { \prime }$ yields approximately the same Tpose correction $D ,$ simply re-posed through Eq. (8). This approximate equivariance arises because $\phi$ is itself only approximately pose-invariant (Eq. (7)), a discrepancy that is quantified numerically in Table 1 and whose practical impact on the trained model we isolate via an ablation in Section 9.2.

## 6.2. ElaOpt: An optimization Based Approach Using Elastic Energies

Finally, we introduce ElaOpt, an optimization-based vertex-wise correction model with regularization based on an elastic deformation energy. The novelty of ElaOpt lies in the combination of three components: (i) a meshagnostic elastic registration term rooted in mathematical shape analysis; (ii) restricting the admissible deformations to the surface-normal direction; and (iii) an adaptive mask that enforces stronger regularization in high-curvature or noiseprone regions, such as the face and hands. Given a source shape  and a target shape $\tau$ , we seek a correction field � that minimizes the energy

![](images/6af2f150da3421cf2db5257afa51cd4d6320ed900a7ad4e75890c9116dcc2275.jpg)  
Figure 3: Overview of Vertex-VaRefine. Given a source mesh $S = \mathrm { D e c } ^ { \mathcal { L } } ( C )$ of prescribed topology and a target mesh  of arbitrary topology, we compute multi-scale VariGrad descriptors $v \in \mathbb { R } ^ { V \times 3 \times \ell }$ on  and project them onto the per-vertex rest-pose frames, $\phi _ { j } = R _ { j } ( \theta ) ^ { \top } v _ { j }$ , giving approximately pose-invariant features on the neutral-pose shape <sup>̄</sup>. A DifusionNet backbone (� blocks of learned difusion, spatial gradient features and a shared pointwise MLP) maps � to a displacement $\boldsymbol { D } \in \mathbb { R } ^ { V \times 3 }$ in T-pose space. The corrected shape $\mathbf { M } = \bar { S } + D$ is re-posed with the source pose, $\hat { S } = \dot { \mathrm { L B S } } ( \mathbf { M } , \theta )$

![](images/8553dee1a12ef5c6292a5589169f939eaf73ac432f34d022aa2c4851acdd80bc.jpg)  
Figure 4: Schematic of the optimization based approach for vertex based mesh correction. The correction vector field is estimated by minimizing the sum of a similarity and a regularization energy.

$$
\mathcal { E } ( \Delta ) = \mathcal { E } ^ { \mathrm { s i m } } ( S + \Delta , \mathcal { T } ) + \mathcal { E } ^ { \mathrm { r e g } } ( S , \Delta ) .
$$

The first term, ${ \mathcal { E } } ^ { \mathrm { { s i m } } }$ , measures the similarity between the corrected source surface $S + \Delta$ and the target surface $\tau _ { \ast }$ while the second term, $\mathcal { E } ^ { \mathrm { r e g } }$ , regularizes the deformation field �. Note that the regularization term depends not only on the correction vector �, but takes into account the geometry of the source . We further emphasize that the correction field is constrained to share the mesh structure of the source surface, whereas the target surface may possess a diferent triangulation. Consequently, the similarity measure must compare surfaces with non-matching mesh discretizations, as is the case for e.g. the Chamfer distance.

In the following, we provide a high-level description of the chosen energy terms, while full details of the precise energy formulation are deferred to 7.4. We employ the Chamfer distance to define the similarity energy due to its simplicity and computational eficiency, although alternative geometric shape metrics, such as the varifold distance from Section 4, could also be used. For the regularization term, we draw inspiration from elastic shape analysis. Specifically, we combine a reparameterization-invariant Sobolev $H ^ { 2 }$ Riemannian metric (Hartman, Sukurdeep, Klassen, Charon and Bauer, 2023)—defined using the geometry of the source surface —with the Square Root Normal Field (SRNF) distance (Jermyn, Kurtek, Klassen and Srivastava, 2012; Kurtek, Jermyn, Klassen and Srivastava, 2011; Bauer, Charon, Harms and Hsieh, 2021) between the source surface and its vertex-corrected counterpart. An ablation study investigating the influence of the various components of the energy is provided in 9.4.

In addition to the regularization energy described above, we employ a hybrid deformation model in which specific regions are subject to additional deformation constraints. To this end, the vertices are partitioned using a mask $m \in$ $\{ 0 , 1 \} ^ { V }$ into masked regions (e.g. hands, feet, and face for human body shapes) on which $m _ { i } = 0$ and unmasked regions where $m _ { i } = 1$ . To prevent excessive deformations within the masked regions, we impose an additional penalty on the $L ^ { 2 } -$ norm of the correction vector field restricted to these regions.

Furthermore, within the unmasked regions, we restrict the correction vector field to act purely in the normal direction of the source surface, i.e., $\Delta _ { i } = m _ { i } \alpha N _ { i } ^ { S } + ( 1 - m _ { i } ) \delta _ { i } , i =$

$1 , \ldots , V$ , where $N ^ { S }$ denotes the normal vector field of  and � is a scalar field defined on the surface, while � fits the chosen constraints on the masked regions. This additional restriction was seen to promote smoother and more regular corrected surfaces. The unconstrained optimization problem itself is tackled with the L-BFGS method. An schematic of the full pipeline is shown in Figure 4.

## 7. Implementation Details

## 7.1. Baseline Encoder

As a baseline encoder we employ the state-of-the-art ArtEq model. Since the original ArtEq framework reconstructs meshes with the SMPL male model and is trained on meshes constructed with the male SMPL model, we re-trained ArtEq with the SMPL neutral model to support both male and female body shapes and ensure compatibility with our datasets. To verify that our reported gains do not simply stem from the additional training data used by the two VaRefine models, we also report a second baseline, denoted $\mathrm { { A r t E q } ^ { + } }$ , which is trained on the same data as the full ArtEq + VaRefine pipeline with the SMPL neutral model. We emphasize that our methods are not dependent on these specific choices of latent space and encoder, but could be employed on top of any others as well. To illustrate this, we use a converter that maps SMPL codes to corresponding SMPL-H or SMPL-X representations (Appendix B). It allows us to transpose both the ground-truth codes of the targets and the ArtEq predictions into these other latent spaces, and thus to run the same experiments on those. We also use the very recently proposed OmniFit (Cai, Xiu, Wang, Shao, Li, Yu, Xu, Liu, Sun, Yang and Zhang, 2026) as an alternative choice of baseline encoder to find initial SMPL-X latent codes (Section 10.2).

## 7.2. Latent-VaRefine

The first layer of the model consists of the computation of VariGrad between the given source-target pair. Here, we specifically choose a single fixed spatial scale of $\sigma =$ 0.75cm with an oriented Gaussian kernel as mentioned in Section 4.1 above. Furthermore, the VariGrad is here computed with respect to the full 375-SMPL latent parameters rather than by-vertex: this yields a smaller vector in $\mathbb { R } ^ { 3 7 5 }$ instead of ℝ<sup>3×6890</sup> for the vertex-wise version. The obtained VariGrad is then concatenated with the latent code � from the baseline encoder output. For ArtEq this consists of 82 total parameters: 3 global orientation parameters, 69 pose (�) parameters, and 10 shape (�) parameters. In total, this concatenated input is a 457 dimensional vector. A fully connected MLP with output layers of dimensions {914, 1828, 3656, 1828, 1828, 1371, 82} with RELU activation functions then performs the inference, where the 82- dimensional output vector is the parameter correction $\Delta _ { C }$

Training is performed using a weighted sum of two loss terms. The first is a mean squared error distance between the refined mesh $\hat { S } = \mathrm { D e c } ^ { \mathrm { S M P L } } ( C + \Delta _ { C } )$ and the target mesh $\tau { : }$

$$
\mathcal { L } _ { \mathrm { M S E } } ( \hat { S } , \mathcal { T } ) = \frac { 1 } { V } \sum _ { i = 1 } ^ { V } \left. \hat { \mathbf { x } } _ { i } - \mathbf { y } _ { i } \right. _ { 2 } ^ { 2 } .
$$

The second is a weighted sum of squared latent parameter errors. Here �<sup>̄</sup> are the ground truth coeficients.

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S M P L } } ( \bar { C } , C + \Delta _ { C } ) = w _ { \mathrm { o r i } } \displaystyle \sum _ { i = 1 } ^ { 3 } ( \bar { C } _ { i } - ( C _ { i } + \Delta _ { C i } ) ) ^ { 2 } } \\ { + w _ { \mathrm { p o s } } \displaystyle \sum _ { i = 4 } ^ { 7 2 } ( \bar { C } _ { i } - ( C _ { i } + \Delta _ { C i } ) ) ^ { 2 } } \\ { + w _ { \beta } \displaystyle \sum _ { i = 7 3 } ^ { 8 2 } ( \bar { C } _ { i } - ( C _ { i } + \Delta _ { C i } ) ) ^ { 2 } } \end{array}
$$

We take $w _ { \mathrm { o r i } } = w _ { \beta } = 1$ and $w _ { \mathrm { p o s } } = 3$ . What remains to set is the relative weighting of MSE and latent code parameter loss. We take a weight of $w _ { \mathrm { M S E } } = 1 0 0 0 0$ . Loss is then given by

$$
\mathcal { L } ( \Delta _ { C } ) = w _ { \mathrm { M S E } } \cdot \mathcal { L } _ { \mathrm { M S E } } ( \hat { S } , \mathcal { T } ) + \mathcal { L } _ { \mathrm { S M P L } } ( \bar { C } , C + \Delta _ { C } ) .
$$

Training is performed for 180 epochs using the Adam optimizer with a learning rate of $1 0 ^ { - 3 . 5 }$ , a batch size of 3000, and a learning rate decay factor of .1 every 60 epochs.

## 7.3. Vertex-VaRefine

At the first stage, multi-scale VariGrad descriptors are computed between the posed source and target meshes using 3 scales: 3cm, 6cm and 10cm. Each vertex-wise VariGrad vector is then projected onto the per-vertex rest-pose frame obtained from linear blend skinning of the pose � (Eq. (7)), producing a 9-dimensional, approximately pose-invariant feature $\phi _ { j }$ per vertex of the canonical T-pose shape <sup>̄</sup>. This feature field is processed by a DifusionNet backbone composed of 4 difusion blocks of width 128: each block combines a learned-time heat difusion step – parametrized by $\begin{array} { r } { k \ = \ 1 2 8 } \end{array}$ Laplace–Beltrami eigenfunctions of the Tpose SMPL template, precomputed once and cached across the dataset – with spatial gradient features and a pointwise MLP shared across vertices. A final linear layer maps the 128-dimensional per-vertex features to a 3-dimensional pervertex displacement �, predicted in T-pose space. The corrected T-pose shape $\mathbf { M } = \bar { S } + D$ is finally re-posed by linear blend skinning with the same pose � (Eq. (8)) to produce the refined reconstruction <sup>̂</sup>. Training is performed for 300 epochs using the Adam optimizer with a learning rate of $4 \times 1 0 ^ { - 4 }$ , batch size 4, gradient-norm clipping at 1.0, and a learning rate decay factor of 0.9 every 50 epochs. The loss function combines a vertex reconstruction term and an intrinsic gradient regularization term weighted by $\lambda _ { v } \ : = \ : 1$ and $\lambda _ { g } = 0 . 0 8$

Vertex-VaRefine is trained on a database with point-topoint correspondences between the targets (or ground truths) and the source meshes. As a result, we use the following loss:

$$
\mathcal { L } = \frac { \lambda _ { v } } { 3 } \mathcal { L } _ { \mathrm { M S E } } + \lambda _ { g } \mathcal { L } _ { \mathrm { g r a d } } ,
$$

where $\lambda _ { v } \ = \ 1$ and $\lambda _ { g } ~ = ~ 0 . 0 8$ are weighting coeficients balancing the vertex reconstruction term and the intrinsic gradient regularization term, respectively. Here $\mathcal { L } _ { \mathrm { M S E } }$ is the same Mean Squared Error as in 7.2, and the factor of 3 accounts for the 3 coordinate channels so that both terms are normalized as a mean over individual scalar entries. $c _ { \mathrm { g r a d } }$ is the area-weighted average of the squared Euclidean distance between the discrete gradient of the predicted mesh and those of the ground truth i.e.:

$$
\mathcal { L } _ { \mathrm { g r a d } } = \frac { 1 } { 3 \sum _ { k = 1 } ^ { F } r _ { k } } \sum _ { k = 1 } ^ { F } r _ { k } \left\| ( \nabla \hat { x } ) _ { k } - ( \nabla y ) _ { k } \right\| _ { 2 } ^ { 2 } ,
$$

where ̂� and $y$ denote the vertices of the predicted mesh $\hat { S }$ and of the ground-truth mesh, respectively, and $F$ denotes the number of mesh faces. The operator ∇ denotes the intrinsic discrete mesh gradient: it is computed face-wise, with two vector components per face expressed in a local tangent basis, so that $( \nabla \hat { x } ) _ { k }$ and $( \nabla y ) _ { k }$ are the �-th of these � components and $r _ { k }$ is the area of the corresponding face. This term regularizes the deformation at the diferential level by matching intrinsic surface gradients. Together, these two losses encourage both accurate and geometrically smooth mesh deformations.

## 7.4. ElaOpt

The correction field $\Delta$ is constrained to act in the normal direction of the source surface, i.e.

$$
\begin{array} { r } { \Delta _ { i } = \alpha _ { i } N _ { i } ^ { S } , \qquad i = 1 , \dots , V , } \end{array}
$$

where $N _ { i } ^ { S }$ denotes the unit normal vector of the source surface at vertex $i ,$ and � is a scalar field defined on the surface. The optimization energy is given by

$$
\mathcal { E } ( \Delta ) = \mathcal { E } ^ { \mathrm { s i m } } ( S + \Delta , \mathcal { T } ) + \mathcal { E } ^ { \mathrm { r e g } } ( S , \Delta ) ,
$$

where the similarity term is defined using the Chamfer distance

$$
\mathcal { E } ^ { \mathrm { s i m } } ( S + \Delta , \mathcal { T } ) = \lambda d _ { \mathrm { C h a m f e r } } ( S + \Delta , \mathcal { T } ) ,
$$

and the regularization energy is given by

$$
\begin{array} { l } { { \displaystyle \mathcal E ^ { \mathrm { r e g } } ( S , { \mathbf { \Delta } } { \mathbf { \Delta } } ) = \mu _ { \mathrm { h f } } \sum _ { i \in \mathcal { M } _ { \mathrm { h f } } } \| { \mathbf { \Delta } } \mathbf { \mathbf { \Delta } } _ { i } \| ^ { 2 } + \mu _ { \mathrm { f a } } \sum _ { i \in \mathcal { M } _ { \mathrm { f a } } } \| { \mathbf { \Delta } } \mathbf { \mathbf { \Delta } } _ { i } \| ^ { 2 } } } \\ { ~ + \frac { 1 } { 1 0 ^ { 8 } } \Big ( \| { \mathbf { \Delta } } { \mathbf { \Delta } } \| _ { H ^ { 1 } ( S ) } ^ { 2 } + \| { \mathbf { \Delta } } { \mathbf { \Delta } } \| _ { H ^ { 2 } ( S ) } ^ { 2 } \Big ) } \\ { ~ + \frac { 1 } { 1 0 ^ { 6 } } \| Q ( S + { \mathbf { \Delta } } { \mathbf { \Delta } } ) - Q ( S ) \| _ { L ^ { 2 } } ^ { 2 } , ~ } \end{array}
$$

where �(⋅) denotes the Square Root Normal Field (SRNF) representation (Jermyn et al., 2012; Kurtek et al., 2011; Bauer et al., 2021), $\dot { \| } \cdot \| _ { H ^ { k } ( S ) } ^ { 2 }$ for $k \in \{ 1 , 2 \}$ denotes the Sobolev norms calculated w.r.t. the induced surface metric of  (Hartman et al., 2023), and $\mathcal { M } _ { \mathrm { h f } }$ and $\mathcal { M } _ { \mathrm { f a } }$ denote masked regions (of e.g. the hands/feet and face) with corresponding weights $\mu _ { \mathrm { h f } }$ and $\mu _ { \mathrm { f a } }$ respectively.

To improve optimization stability, the energy is minimized iteratively using a L-BFGS method and a continuation strategy with increasing weights $\lambda \in \{ 1 0 ^ { - 8 } , 1 0 ^ { - 6 } , 1 0 ^ { - 4 } \}$ where the solution obtained for one value of � is used to initialize the subsequent optimization stage.

## 8. Results

## 8.1. Datasets

We use meshes reconstructed by a latent-space retrieval method as source shapes, while the corresponding groundtruth meshes are used as targets. More specifically, given a target mesh $\tau _ { \ast }$ we first obtain an initial reconstruction using one of the state-of-the-art latent-space body retrieval algorithm ArtEq. This gives a latent code $C = ( \beta , \theta )$ , which is in turn used to generate a source mesh $S = \mathrm { D e c } ^ { \mathrm { S M P L } } ( C )$ The objective of the proposed methods is to predict a code correction $\Delta _ { C }$ and/or a deformation field Δ along  that refines it toward $\tau .$

To this end, we first train ArtEq on the DFAUST dataset (Bogo, Romero, Pons-Moll and Black, 2017), following the protocol from the original paper (Feng et al., 2023). Among our proposed methods, both VaRefine algorithms require training. For this, we use the BMLmovi dataset (Ghorbani, Mahdaviani, Thaler, Kording, Cook, Blohm and Troje, 2020), a subset of AMASS (Mahmood, Ghorbani, Troje, Pons-Moll and Black, 2019) similar to DFAUST. For each mesh $\tau _ { i }$ in BMLmovi, we first extract a latent code $C _ { i }$ through the trained ArtEq network, which corresponds to a source mesh $S _ { i }$ . Since AMASS contains a very large number of frames, we perform frame sampling to reduce training and testing set sizes. More precisely, we randomly select five frames from each motion sequence. As a result, the final training set contains 9005 target meshes $\tau _ { i } ,$ 9005 SMPL codes $C _ { i }$ and their corresponding decoded source meshes $S _ { i } , i = 1 , \ldots , 9 0 0 5$

For evaluation, we perform a large-scale evaluation on (randomly selected frames of) all of AMASS, excluding the BMLmovi dataset which was used for training. We also give separate results for multiple AMASS sub-datasets in order to assess the generalization capability of our methods across diverse motion distributions and capture conditions. The final evaluation set contains 80 072 meshes. For each sample target  in the dataset, we first apply the trained ArtEq network to obtain a latent SMPL code � and corresponding mesh , then apply/combine each of the three methods in order to compute a correction to the code � and/or each vertex position of , then compare the V2V and asymmetric Chamfer distance between the corrected prediction and .

## 8.2. Computational Cost

In Table 2 we present the computational cost of the proposed refinement frameworks in terms of model complexity and inference eficiency. All experiments were conducted on a machine equipped with an NVIDIA RTX 5000 Ada Generation laptop GPU. For training measurements, the gradient descriptors are assumed to be precomputed. In addition to the reported training cost, the learning based methods required the pre-computation of VariGrad representations for the entire training set. For Latent-VaRefine, where Vari-Grad representations are computed in the SMPL parameter space using a single scale, this step required 108.48 minutes to compute the descriptors for the full dataset of 89 077 meshes. In contrast, Vertex-VaRefine computes VariGrad descriptors on mesh vertices using 3 scales, resulting in a preprocessing time of 118.77 minutes for the same dataset. Vertex-VaRefine additionally pre-computes, for each training sample, the T-pose shape, the frame-projected VariGrad features, and the intrinsic gradient and mass operators used by the gradient regularization term $\mathcal { L } _ { \mathrm { g r a d } } ;$ this step took 2.43 minutes for the 9 005 training meshes.

Table 2  
Table 3  
Computational cost of the proposed refinement frameworks. We report the number of trainable parameters, the average training time per epoch, and the average inference time per sample including VariGrad calculations.
<table><tr><td>Method</td><td>#Params (M)</td><td>Train Time/Epoch(s)</td><td>Inference (ms)</td></tr><tr><td>Latent-VaRefine</td><td>21.43</td><td>8.2</td><td>73.1</td></tr><tr><td>Vertex-VaRefine</td><td>0.46</td><td>260</td><td>82.4</td></tr><tr><td>ElaOpt</td><td>n/A</td><td>n/A</td><td>2006.8</td></tr></table>

## 8.3. Comparison of Proposed Refinement Frameworks

We evaluate the proposed refinement frameworks on a large collection of AMASS sub-datasets using the SMPL target representation. The average across AMASS and several sub-dataset results are reported in Table 3 in terms of vertexto-vertex error (V2V) and asymmetric Chamfer Distance (CD), both expressed in centimeters, together with the relative improvement over the ArtEq baseline. The remaining sub-datasets are found in Tables 13, 14 of appendix.

Overall, all refinement strategies improve the reconstruction quality across all evaluated datasets, demonstrating strong generalization capabilities across diverse motion distributions and capture conditions. Among the learningbased approaches, Vertex-VaRefine consistently achieves the lowest V2V values. In particular, large improvements are observed on challenging datasets such as GRAB, TCD, and SFU, where Vertex-VaRefine reduces the V2V by up to 57.1%, 56.4%, and 52.0%, respectively. The optimization based method, ElaOpt, achieves the best Chamfer Distance results across most datasets, with improvements exceeding 50% on several benchmarks. Latent-VaRefine already provides a noticeable improvement over the initial ArtEq reconstructions, but remains less accurate than Vertex-VaRefine on both V2V and CD metrics, and than ElaOpt on CD. Its main advantage, however, is that it stays within the SMPL latent space: it outputs refined SMPL parameters rather than a per-vertex displacements, thereby preserving their interpretability. These observations confirm that directly refining the reconstructed geometry at the vertex level leads to more accurate and stable surface corrections than operating only in the latent space, at the cost of leaving the original parametric representation. Finally, we also report results of combining Latent-VaRefine with ElaOpt, by running these two frameworks in succession, which turns out to lead to the consistently best CD results across all data sets and both comparison methods. We also retrain ArtEq on the additional data used in our pipeline (i.e., on DFAUST+BMLmovi), indicated as ArtEq<sup>+</sup>. Although this does reduce ArtEq’s error, it clearly remains behind Vertex-VaRefine, Latent-VaRefine + ElaOpt, Latent-VaRefine + Vertex-VaRefine applied on top of the original ArtEq, on all sub-datasets, corroborating the fact that the improvement is not explained by the additional training data alone. In addition to these quantitative results, we present in Figure 5 a qualitative comparison, which demonstrates that all proposed methods efectively reduce the per-vertex reconstruction errors across a wide variety of body shapes and poses compared to the initial ArtEq reconstructions.

Comparison of reconstruction performance on representative AMASS sub-datasets. ArtEq<sup>+</sup> denotes ArtEq retrained with the additional BMLmovi training frames. We report the vertexto-vertex error (V2V) and Asymmetric Chamfer Distance (CD), both expressed in centimeters (cm), together with the relative improvement with respect to ArtEq. The column labeled AMASS does not include the BMLmovi sub-dataset as that was used as the training set for the two VaRefine methods.
<table><tr><td>Method</td><td>|Metric</td><td>|AMASS|</td><td>GRAB</td><td>TCD</td><td>HumanEva</td><td>EKUT</td><td>SFU</td></tr><tr><td rowspan="2">ArtEq</td><td>|V2V</td><td>3.168</td><td>3.163 5.637</td><td></td><td>2.566</td><td>1.938</td><td>4.188</td></tr><tr><td>CD</td><td>1.681</td><td>1.708</td><td>2.689</td><td>1.327</td><td>1.087</td><td>2.223</td></tr><tr><td rowspan="2">ArtEq⁺</td><td>|V2V</td><td>2.401</td><td>1.994</td><td>3.092</td><td>2.257</td><td>2.043</td><td>3.031</td></tr><tr><td>CD</td><td>1.376</td><td>1.190</td><td>1.655</td><td>1.294</td><td>1.162</td><td>1.748</td></tr><tr><td rowspan="4">Latent-VaRefine</td><td>|V2V</td><td>2.511</td><td>2.499 3.620</td><td></td><td>2.305</td><td>1.678</td><td>3.356</td></tr><tr><td>Gain (%)</td><td>20.7</td><td>21.0</td><td>35.8</td><td>10.2</td><td>13.4</td><td>19.9</td></tr><tr><td>CD</td><td>1.414</td><td>1.407</td><td>1.762</td><td>1.252</td><td>1.013</td><td>1.899</td></tr><tr><td>Gain (%)</td><td>15.9</td><td>17.6</td><td>34.5</td><td>5.7</td><td>6.8</td><td>14.6</td></tr><tr><td rowspan="4">Vertex-VaRefine</td><td>|V2V Gain (%)</td><td>1.552 51.0</td><td></td><td>1.356 2.457</td><td>1.423</td><td>1.094</td><td>2.012</td></tr><tr><td></td><td></td><td>57.1</td><td>56.4</td><td>44.5</td><td>43.6</td><td>52.0</td></tr><tr><td>|CD</td><td>0.943</td><td>0.870</td><td>1.300</td><td>0.856</td><td>0.746</td><td>1.211</td></tr><tr><td>Gain (%)</td><td>43.9</td><td>49.1</td><td>51.7</td><td>35.5</td><td>31.4</td><td>45.5</td></tr><tr><td rowspan="4">ElaOpt</td><td>|V2V</td><td>2.652</td><td>2.596</td><td>4.316</td><td>1.927</td><td>1.634</td><td>3.330</td></tr><tr><td>Gain (%)</td><td>16.3</td><td>17.9</td><td>23.4</td><td>24.9</td><td>15.7</td><td>20.5</td></tr><tr><td>CD</td><td>0.793</td><td>0.787</td><td>1.072</td><td>0.663</td><td>0.622</td><td>0.885</td></tr><tr><td>Gain (%)</td><td>52.8</td><td>53.9</td><td>60.1</td><td>50.0</td><td>42.8</td><td>60.2</td></tr><tr><td rowspan="4">Latent-VaRefine + ElaOpt</td><td>|V2V Gain (%)</td><td>2.008</td><td>1.979</td><td>2.576</td><td>1.673</td><td>1.331</td><td>1.943</td></tr><tr><td></td><td>36.6</td><td>37.4</td><td>54.3</td><td>34.8</td><td>31.3</td><td>53.6</td></tr><tr><td>|CD</td><td>0.717</td><td>0.710</td><td>0.805</td><td>0.652</td><td>0.604</td><td>0.743</td></tr><tr><td>Gain (%)</td><td>57.3</td><td>58.4</td><td>70.1</td><td>50.9</td><td>44.4</td><td>66.6</td></tr><tr><td rowspan="4">Latent-VaRefine + Vertex-VaRefine</td><td>|V2V</td><td>1.715</td><td>1.558</td><td>2.529</td><td>1.593</td><td>1.264</td><td>2.599</td></tr><tr><td>Gain (%)</td><td>45.9</td><td>50.7</td><td>55.1</td><td>37.9</td><td>34.8</td><td>37.9</td></tr><tr><td>CD</td><td>1.010</td><td>0.946</td><td>1.317</td><td>0.923</td><td>0.819</td><td>1.497</td></tr><tr><td>Gain (%)</td><td>39.9</td><td>44.6</td><td>51.0</td><td>30.4</td><td>24.7</td><td>32.7</td></tr></table>

## 8.4. Evaluation with Additional Metrics

In order to test the independence of our results to the choice of metric, we report the results on one subset of AMASS using an additional set of metrics that are independent of the training objectives. For the SSM\_synced subset of AMASS, Table 4 reports the relative improvement over ArtEq under Hausdorf distance, vertex-to-vertex supremum, and both oriented and unoriented varifold metrics (using � = .1). Interestingly, the improvements under these metrics are of the same order as those obtained with the V2V and Chamfer distances used in the design and training of the methods. The descriptions of these distances can be found in the Appendix A.

![](images/ae4f2f5a8c138d9364175d620c32e51d5a31c096e0185c9a0d3fb1af620ef84f.jpg)  
Figure 5: Qualitative comparison between ArtEq, latent-VaRefine, Vertex-VaRefine, ElaOpt, Latent-VaRefine + ElaOpt, and Latent-VaRefine + Vertex-VaRefine on diferent body shapes and poses. Color maps show the per-vertex reconstruction error with respect to the ground-truth surface in centimeters (cm).

Table 4  
Relative improvement (%) over the ArtEq baseline on the SSM\_synced subset of AMASS, under metrics independent of the training objective.
<table><tr><td>Metric</td><td>Latent-VaRefine Vertex-VaRefine ElaOpt</td><td></td><td></td></tr><tr><td>Hausdorff</td><td>43.5</td><td>57.1</td><td>60.1</td></tr><tr><td>V2V supremum</td><td>41.3</td><td>55.9</td><td>18.3</td></tr><tr><td>Varifold (oriented)</td><td>32.7</td><td>59.0</td><td>75.4</td></tr><tr><td>Varifold (unoriented)</td><td>32.7</td><td>57.2</td><td>75.2</td></tr></table>

## 9. Ablation Studies

## 9.1. Number of VariGrad Scales

We analyze here the contribution of the proposed components through a series of ablation studies. First, we evaluate the impact of the number of VariGrad scales used during refinement. We study the efect of diferent multiscale VariGrad configurations on the performance of the Latent-VaRefine and Vertex-VaRefine models. In the latent refinement setting, increasing the number of scales beyond a single-scale representation slightly degrades performance. In contrast, the vertex refinement model benefits from multiscale VariGrad features with the 3-scale configuration providing the best trade-of between accuracy and stability across the AMASS benchmark and is therefore adopted in the final model.

Latent-VaRefine. For Latent-VaRefine, we consider 3 and 5 scale variants and also perform an ablation to test the value of including the coeficient fidelity term. In addition to the base .75 cm base scale, we include half and double scale (.375 cm and 1.5 cm) VariGrad descriptors to obtain a 3-scale VariGrad layer. To these are further added the 5 cm and 15 cm VariGrads to obtain a 5-scale model. The ablation to test the value of the coeficient fidelity term is performed using the same single .75 cm VariGrad as the standard Latent-VaRefine with the only diference being that only MSE is used for training loss in this case.

The comparison of the standard 1 scale Latent-VaRefine to the 3 and 5 scale versions is found in Table 5, which also includes the MSE-only ablation results. In all cases, the standard Latent-VaRefine performs best by a slim margin. It is likely that the wider model needed to accommodate the additional input data is the cause of the lower performance of the 3 and 5 scale variants. The inclusion of the coeficient fidelity term appears to improve both MVE and Chamfer error as the term likely both regularizes the training and directly encourages improvement as errors in coeficients lead to higher MVE and Chamfer error.

Vertex-VaRefine. Similar to the study on Latent-VaRefine, we wish to determine the potential efect of adding more scales in the VariGrad representation. To that end, we consider multiple uniformly spaced scales in order to capture geometric discrepancies at diferent spatial resolutions. We thus consider three specific configurations: a single scale {5 cm}, three scales {3 cm, 6 cm, 10 cm}, and five scales {2 cm, 4 cm, 6 cm, 8 cm, 10 cm}. Table 5 reports the reconstruction performance obtained with each configuration using both the vertex-to-vertex (V2V) and the asymmetric Chamfer Distance (CD). These results show that multiscale VariGrad features consistently improve reconstruction performance over a single scale. Notably, the performance gain between the single-scale and the 3-scale configuration (V2V: 1.734 vs. 1.552 on AMASS) is substantially larger than the performance gain between the 3-scale and 5-scale configurations (V2V: 1.552 vs. 1.527), indicating that most of the benefit of multi-scale VariGrad features is already achieved with 3 scales, with diminishing returns beyond that point. While the 5-scale configuration achieves the best results on most datasets, this additional improvement over the 3-scale configuration is modest and comes at a noticeably higher memory and training-time cost, since each additional scale requires computing, storing, and projecting an extra VariGrad feature channel per vertex. We therefore adopt the 3-scale configuration as our final choice, as it ofers the best trade-of between reconstruction accuracy and computational cost.

Ablation studies on the number of VariGrad scales for Latent-VaRefine and Vertex-VaRefine, and on the coeficient loss term of Latent-VaRefine. We report vertex-to-vertex (V2V) and asymmetric Chamfer Distance (CD) errors, both in cm, on representative datasets and on the entire AMASS benchmark (excluding BMLmovi). Best results are in bold, second-best are underlined.
<table><tr><td>Model</td><td>Configuration</td><td>Metric</td><td>AMASS</td><td>GRAB</td><td>EKUT</td><td>SFU</td></tr><tr><td>ArtEq</td><td colspan="2">V2V CD</td><td>3.168 1.681</td><td>3.163 1.708</td><td>1.938 1.087</td><td>4.188 2.223</td></tr><tr><td rowspan="6">Latent- VaRefine</td><td>1 scale</td><td>V2V CD</td><td>2.511 1.414</td><td>2.499 1.407</td><td>1.678 1.013</td><td>3.356 1.899</td></tr><tr><td>1 scale (MSE loss only)</td><td>V2V CD</td><td>2.625 1.470</td><td>2.577 1.466</td><td>1.722 1.024</td><td>3.525 1.965</td></tr><tr><td>3 scales</td><td>V2V CD</td><td>2.545 1.433</td><td>2.528 1.425</td><td>1.703 1.021</td><td>3.431 1.934</td></tr><tr><td>5 scales</td><td>V2V CD</td><td>2.590 1.440</td><td>2.529 1.417</td><td>1.721 1.022</td><td>3.394 1.918</td></tr><tr><td>1 scale</td><td>V2V CD</td><td>1.734 1.043</td><td>1.466 0.950</td><td>1.153 0.788</td><td>2.268 1.338</td></tr><tr><td>3 scales</td><td>V2V CD</td><td>1.552 0.943</td><td>1.356 0.870</td><td>1.094</td><td>2.012</td></tr><tr><td rowspan="2">VaRefine</td><td></td><td>V2V</td><td>1.527</td><td>1.280</td><td>0.746 0.964</td><td>1.211</td></tr><tr><td>5 scales</td><td>CD</td><td>0.933</td><td>0.828</td><td>0.700</td><td>1.959 1.226</td></tr></table>

## 9.2. Efect of the Equivariance in Vertex-VaRefine

Section 6.1 argues that Vertex-VaRefine is approximately pose-equivariant by construction, but not fully equivariant; this is confirmed by Table 1, with a median relative error of 4–10% and a mean relative error of 23–50%. To determine if this approximate equivariance is genuinely beneficial rather than merely a design choice, we trained an otherwise identical variant that entirely omits the canonicalframe projection and the neutral pose/re-posing step. In this version, the DifusionNet backbone processes raw, unprojected VariGrad descriptors computed directly on the posed source mesh and predicts displacement directly in posed space, providing no equivariance guarantees. Both variants maintain the same backbone architecture, VariGrad scales, loss functions, and training data; the only diference lies in the pose-equivariant wrapping.

Table 6 demonstrates that omitting this construction impairs both metrics across all sub-datasets. On AMASS, the V2V gain decreases from 51% to 47.4%, and the CD gain from 43.9% to 40.5%. The decline is most pronounced on SFU, where the V2V gain falls from 52.0% to 39.4%. Consequently, despite the input features being only approximately

## Table 6

Ablation of the pose-equivariant construction in Vertex-VaRefine (DifusionNet backbone), evaluated on the same representative AMASS sub-datasets as Table 3. Both variants share identical backbone, VariGrad features, training data, and schedule; the only diference is the canonical T-pose projection and re-posing step of Section 6.1. We report the vertex-tovertex error (V2V) and Asymmetric Chamfer Distance (CD), in centimeters (cm), together with the relative gain (%) over the ArtEq baseline. The AMASS column is the average over the AMASS sub-datasets, excluding BMLmovi, which was used for training. Removing the equivariant construction degrades both metrics on every sub-dataset.
<table><tr><td>Method</td><td>|Metric</td><td>|AMASS|</td><td>GRAB</td><td>TCD</td><td>HumanEva EKUT</td><td></td><td>SFU</td></tr><tr><td>ArtEq</td><td>|V2V CD</td><td>3.168 1.681</td><td>3.163 1.708</td><td>5.637 2.689</td><td>2.566 1.327</td><td>1.938 1.087</td><td>4.188 2.223</td></tr><tr><td rowspan="3">Vertex-VaRefine (w/ equiv.)</td><td>|V2V Gain (%)</td><td>1.552</td><td>1.356</td><td>2.457</td><td>1.423</td><td>1.094</td><td>2.012</td></tr><tr><td>CD</td><td>51.0 0.943</td><td>57.1 0.870</td><td>56.4 1.300</td><td>44.5 0.856</td><td>43.6</td><td>52.0</td></tr><tr><td>Gain (%)</td><td>43.9</td><td>49.1</td><td>51.7</td><td>35.5</td><td>0.746 31.4</td><td>1.211 45.5</td></tr><tr><td rowspan="3">Vertex-VaRefine (w/o equiv.)</td><td>V2V Gain (%)</td><td>1.666</td><td>1.554</td><td>2.670</td><td>1.607</td><td>1.192</td><td>2.540</td></tr><tr><td>CD</td><td>47.4</td><td>50.9</td><td>52.6</td><td>37.4</td><td>38.5</td><td>39.4</td></tr><tr><td>Gain (%)</td><td>1.001 40.5</td><td>0.975 42.9</td><td>1.385 48.5</td><td>0.941 29.1</td><td>0.782 28.1</td><td>1.455 34.5</td></tr></table>

pose-invariant, the proposed construction yields a consistent performance improvement.

## 9.3. Backbone Choice: DifusionNet vs. SpiralNet vs. PoissonNet

Vertex-VaRefine’s pipeline – VariGrad computation, projection onto per-vertex rest-pose frames (Eq. (7)), T-pose prediction, and re-posing by linear blend skinning (Eq. (8)) – is agnostic to the network processing �. We assess this choice by replacing the DifusionNet backbone (Section 6.1) with two alternatives, both are described in Appendix C, trained and evaluated identically as in Section 7. PoissonNet uses a PoissonNet (Maesumi et al., 2025) backbone of the same width and depth as DifusionNet $( C _ { \mathrm { w i d t h } } { = } 1 2 8 ,$ $N _ { \mathrm { b l o c k } } { = } 4 )$ , replacing learned difusion with a per-block Poisson solve – involving one Cholesky factorization of the cotangent Laplacian per mesh (Nicolet, Jacobson and Jakob, 2021) – followed by the same linear displacement head. We also test the SpiralNet approach with a Spiral-Net++ (Gong et al., 2019) encoder (three spiral convolution layers, {32, 32, 64}, ReLU, no downsampling), whose pervertex output is averaged per face into 64-dim descriptors and fed to a Neural Jacobian Field (Aigerman, Gupta, Kim, Chaudhuri, Saito and Groueix, 2022) decoder (4 FC layers, 128 units, GroupNorm, ReLU) that predicts a per-face 3×3 Jacobian. Unlike the other two backbones, the displacement is not a direct output: the Jacobian field is integrated via a Poisson solve factorized once on a fixed template ${ \mathit { S } } _ { 0 } ,$ and the resulting correction Δ is added to the per-sample neutral pose shape, � ≐ Δ, � = <sup>̄</sup> + Δ – keeping this head compatible with the shared equivariant pipeline. We report the results in Table 7, confirming DifusionNet as the optimal choice for this framework.

Table 8  
Ablation on the backbone processing the frame-projected VariGrad features �. All three variants share the same descriptors, rest-pose frame projection, T-pose prediction, reposing, loss and training schedule; only the backbone (and, for SpiralNet++, the decoding head it requires) changes. V2V and asymmetric Chamfer Distance in centimeters.
<table><tr><td rowspan="2"></td><td rowspan="2">Subset Metric ArtEq</td><td rowspan="2">DiffusionNet</td><td rowspan="2">SpiralNet PoissonNet</td><td rowspan="2"></td></tr><tr><td>(Vertex-VaRefine)</td></tr><tr><td rowspan="2">AMASS</td><td>V2V 3.168</td><td>1.552</td><td>2.019</td><td>2.118</td></tr><tr><td>CD 1.681</td><td>0.943</td><td>1.162</td><td>1.205</td></tr><tr><td rowspan="2">SFU</td><td>V2V 4.188</td><td>2.012</td><td>2.612</td><td>2.823</td></tr><tr><td>CD 2.223</td><td>1.211</td><td>1.486</td><td>1.592</td></tr></table>

## 9.4. ElaOpt Ablations

We perform two ablation studies on ElaOpt. The first is a simple comparison against SMPL+D, and the latter is a term-wise examination, to qualitatively inspect the value of the individual terms and their quantitative impact on reported metrics such as V2V error and Chamfer error.

Comparison With SMPL+D. The results of an unrestricted SMPL+D-style fitting are less accurate and regular than those of ElaOpt; for example, on the SFU dataset a SMPL+D style fitting with no regularizer and an SRNF regularizer produces a CD improvement of 24%, 31% respectively.

Termwise Ablation. For the second ablation, we consider six diferent variants of ElaOpt. The first is ElaOpt as presented in subsection 7.4. The second is plain, normalrestrained optimization. To this are added masking and the $H ^ { 1 }$ $H ^ { 2 }$ , and SRNF terms. All six cases retain the constraint to the normal direction, diferentiating them from an entirely free optimization.

A sample of qualitative results can be seen in Figure 7 and the quantitative results across the SFU dataset can be found in Table 8. Due to the nature of optimization methods, it can be noted that while the quantitative results of less restrained methods can get lower numerically, this is not always preferable if it comes at the cost of a messier, qualitatively worse result.

Although quantitatively ElaOpt does not win in V2V nor Chamfer metrics, the qualitative value of the fully regularized method can be seen in Figure 7. In this figure, we can see the value of the masks in retaining an identifiable face and the regularization terms for their avoidance of numerical instability which can be seen in the figure as jagged artifacts.

## 10. Generalizability

As a final set of evaluation tests, we examine the generalizability of our methods. We show that they work with a diferent choice of decoder (OmniFit (Cai et al., 2026)), of latent space (SMPL-H/X), and with targets outside of the reach of the decoder (due to either a change of resolution or belonging to a more general category, such as raw scans).

Numerical evaluation on the efect of the ElaOpt terms on V2V and Chamfer. All computations are on the SFU dataset and reported in cm.
<table><tr><td colspan="3">ArtEq ElaOpt Nothing +Masks  $+ H ^ { 1 }$   $+ H ^ { 2 }$  +SRNF</td></tr><tr><td>V2V 4.188 3.330 3.270</td><td>3.314 3.265 3.268</td><td>3.314</td></tr><tr><td>CD 2.223 0.885 0.848</td><td>0.861 0.850 0.908</td><td>0.851</td></tr></table>

## 10.1. Generalizability to a diferent latent space

To demonstrate that our framework is not specific to SMPL, we apply all three methods (Latent-VaRefine, Vertex-VaRefine and ElaOpt) to two richer body models. SMPL-H shares the SMPL template topology, but articulates the fingers. SMPL-X additionally models the jaw, eyes, and facial expressions. Since ArtEq is trained for SMPL, we do not retrain it for these models. Instead, the SMPL-H and SMPL-X codes of both the source and target meshes are obtained from the corresponding SMPL codes with the SMPL→SMPL-H and SMPL→SMPL-X converters of Appendix B: the source codes are converted ArtEq predictions, and the target codes are converted ground-truth SMPL codes. The two learningbased methods are retrained with the SMPL-H/SMPL-X versions of the BMLmovi data, keeping the architectures of Section 7. In the case of Vertex-VaRefine, hyper-parameters are also kept identical. For Latent-VaRefine the learning rate and relative coeficient-loss/MSE-loss weighting were adjusted. For SMPL-X we set $w _ { \mathrm { M S E } } = 2 0$ while for SMPL-H we set $w _ { \mathrm { M S E } } = 1$ . For both, the initial learning rate was set to $1 0 ^ { - 4 . { \overset { . } { 0 } } { } }$ and an additional weight was needed for the new hand parameters for the latent space loss. As there are 90 hand parameters in SMPL-H and SMPL-X and these parameters have little efect on overall mesh correctness, this weight was set to 0. ElaOpt requires no training and is applied directly to the converted source and target meshes. All methods are evaluated on the remaining AMASS subdatasets, and are showed to perform similarly well as on their SMPL version.

## 10.2. Use of a diferent encoder: OmniFit

To further assess the generality of our framework beyond the ArtEq encoder, we evaluate it on top of OmniFit (Cai et al., 2026), a recent multi-modal, scale-agnostic method that fits the SMPL-X body model to 3D human assets. We treat its reconstruction as the source mesh $S = \mathrm { D e c } ^ { \mathcal { L } } ( C )$ and apply our refinement methods to recover the residual geometry, without any change to their architecture.

The learning-based methods (Latent and Vertex-VaRefine) are trained on the OmniFit outputs of the BML\_movi, while ElaOpt, being optimization-based, requires no training; all methods are then evaluated on the remaining subsets (GRAB, TCD, HumanEva, EKUT, SFU), so that the reported scores measure generalization to unseen sequences. Results are given in Table 10, using the vertex-to-vertex (V2V) and the Chamfer distance (CD), both in centimeters. Interestingly, we obtain even better improvements than for

Table 9  
Extension to SMPL-H and SMPL-X (neutral models). V2V and asymmetric Chamfer Distance (CD) in cm, with relative improvement (Gain) over the initial source reconstruction. The AMASS column does not include BMLmovi, which is used for training.
<table><tr><td>Method</td><td>Metric</td><td>SMPL-H AMASS TCD</td><td>SFU</td><td>SMPL-X AMASS</td><td>TCD SFU</td></tr><tr><td>ArtEq</td><td>|V2V CD</td><td>3.167 5.637 1.681 2.689</td><td>4.188 2.223</td><td>2.575 1.296</td><td>4.282 3.805 1.908 1.807</td></tr><tr><td rowspan="2">Latent- VaRefine</td><td>|V2V Gain (%)</td><td>2.675 15.5</td><td>4.103 3.600 27.2 14.0</td><td>2.138 17.0</td><td>2.923 3.072 31.7 19.3</td></tr><tr><td>CD Gain (%)</td><td>1.480 2.035 12.0 24.3</td><td>2.012 9.5</td><td>1.119 13.7</td><td>1.361 1.546 28.7 14.4</td></tr><tr><td rowspan="2">Vertex- VaRefine</td><td>|V2V Gain (%)</td><td>1.445 54.4</td><td>2.605 2.219 53.8 47.0</td><td>1.188 53.9</td><td>1.798 1.849 58.0 51.4</td></tr><tr><td>|CD Gain (%)</td><td>0.904 1.354 46.2 49.6</td><td>1.364 38.6</td><td>0.688 46.9</td><td>0.918 1.042 51.9 42.3</td></tr><tr><td rowspan="2">ElaOpt</td><td>|V2V Gain (%)</td><td>2.616 17.4</td><td>4.426 3.442 21.5 17.8</td><td>2.063 19.9</td><td>4.097 2.998 4.3 21.2</td></tr><tr><td>|CD Gain (%)</td><td>0.803 52.2 55.2</td><td>1.204 0.978 56.0</td><td>0.608 53.1</td><td>0.997 0.831 47.7 54.0</td></tr></table>

Table 10

Per-dataset evaluation on the AMASS subsets. OmniFit gives the initial reconstruction; we report the error after each refinement method and the relative gain over OmniFit. Lower is better; all values in cm.
<table><tr><td>Method</td><td>Metric</td><td>AMASS</td><td>GRAB</td><td>TCD</td><td>HumanEva</td><td>EKUT</td><td>SFU</td></tr><tr><td rowspan="2">OmniFit</td><td>V2V</td><td>1.181</td><td>1.130</td><td>1.112</td><td>1.167</td><td>1.592</td><td>1.162</td></tr><tr><td>CD</td><td>0.649</td><td>0.641</td><td>0.706</td><td>0.616</td><td>0.727</td><td>0.655</td></tr><tr><td rowspan="3">Latent-VaRefine</td><td>V2V Gain (%)</td><td>0.822</td><td>0.825</td><td>1.092</td><td>0.853</td><td>0.822</td><td>0.868</td></tr><tr><td></td><td>30.4</td><td>27.0</td><td>1.8</td><td>26.9</td><td>48.4</td><td>25.3</td></tr><tr><td>CD Gain (%)</td><td>0.533 17.9</td><td>0.529 17.5</td><td>0.700 0.8</td><td>0.482 21.8</td><td>0.520 28.5</td><td>0.572 12.7</td></tr><tr><td rowspan="4">Vertex-VaRefine</td><td>V2V</td><td>0.342</td><td>0.328</td><td></td><td></td><td>0.339</td><td>0.419</td></tr><tr><td>Gain (%)</td><td>71.0</td><td>71.0</td><td>0.400 64.0</td><td>0.311</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>73.4</td><td>78.7</td><td>63.9</td></tr><tr><td>CD Gain (%)</td><td>0.274 57.8</td><td>0.253</td><td>0.329</td><td>0.255</td><td>0.274</td><td>0.323</td></tr><tr><td rowspan="4">ElaOpt</td><td>V2V</td><td></td><td>60.5</td><td>53.4</td><td>58.6</td><td>62.3</td><td>50.7</td></tr><tr><td>Gain (%)</td><td>0.967</td><td>0.954</td><td>0.902</td><td>0.934</td><td>0.982</td><td>0.930</td></tr><tr><td></td><td>18.1</td><td>15.6</td><td>18.9</td><td>20.0</td><td>38.3</td><td>20.0</td></tr><tr><td>CD Gain (%)</td><td>0.289 55.5</td><td>0.245 61.8</td><td>0.308 56.4</td><td>0.281 54.4</td><td>0.381 47.6</td><td>0.311 52.5</td></tr></table>

ArtEq, showcasing the generalizability of our method to both a diferent latent space (SMPL-X) and a completely diferent decoder.

## 10.3. Robustness to Target Mesh Variations

Mesh Resolution. To test that all three refining methods are robust to remeshing, we apply them to both upsampled and downsampled versions of 100 random target meshes from the SFU dataset. We report the results in Table 11, which confirms our statement.

Raw Scan Targets. We test our methods on 10 raw scans from FAUST, i.e., targets completely outside of any parametric body model. The scans were chosen so that ArtEq gives a relatively accurate pose, before checking the improvements of the proposed methods (Table 12, Figure 6). In this case, only (asymmetric) Chamfer distance makes sense as a measurement. As expected, Latent-VaRefine cannot improve the result by much, as it is constraiend to stay within the SMPL space. On the other hand, both vertex-based methods give marked improvement, with the Optimizationbased method logically being the most generalizable to this far-from-training situation.

Table 11  
Robustness to mesh resolution. Relative improvement (%) in asymmetric Chamfer Distance (CD) over ArtEq on 100 target meshes from the SFU dataset, for the original, downsampled and upsampled targets. The methods are applied as trained on SMPL targets.
<table><tr><td>Method</td><td>Original</td><td>Downsampled</td><td>Upsampled</td></tr><tr><td>Latent-VaRefine</td><td>17.8</td><td>18.5</td><td>17.5</td></tr><tr><td>Vertex-VaRefine</td><td>54.7</td><td>46.3</td><td>46.9</td></tr><tr><td>ElaOpt</td><td>65.7</td><td>66.1</td><td>65.8</td></tr></table>

Table 12

Generalization to non-SMPL targets. Relative improvement (%) in asymmetric Chamfer Distance (CD) over the ArtEq prediction on 10 raw scans from FAUST. The scans were chosen so that ArtEq gives a relatively accurate pose, before evaluating any of the proposed methods.
<table><tr><td>Metric</td><td></td><td>Latent-VaRefine Vertex-VaRefine</td><td>ElaOpt</td></tr><tr><td>CD Gain (%)</td><td>3.9</td><td>29.1</td><td>65.2</td></tr></table>

## 11. Limitations and Future work

Individual Weakness of Each Approach. From the results of the previous sections, one can conclude that Latent-VaRefine preserves the latent representation but is less accurate, though it can be easily combined with the two other frameworks to efect; Vertex-VaRefine achieves the lowest vertex-to-vertex errors; while ElaOpt yields the best Chamfer distances and is extendable to clothed meshes (see Appendix D), at the cost of a substantially slower inference and potentially more irregular output meshes. Combining a learning based initialization with few subsequent optimization steps is a natural direction for future work. Another common limitation to all proposed methods is the assumption that the source mesh produced by the base latent code retrieval method is not too far from the ground truth, especially in terms of the predicted pose. This could possibly be improved by adapting Latent-VaRefine to capture bigger changes, which we intend to investigate.

Shape and Motion Generation. In their current form, all three methods allow for animation of the refined shape using the pose parameter of the latent space from which the source shape is obtained. This could be used in conjunction with motion generation algorithms such as ARDY (Zhao, Petrovich, Zhang, Wang, Tang and Rempe, 2026) to animate general body shapes. On the other hand, the generation of more general human body shapes using generative methods could be possible, though it will probably require adapting the Vertex-VaRefine approach to allow for a useful latent space on which to work. This is another topic of interest as a natural follow-up to this work.

![](images/1c7ee4b090e0573980a438f3031bcb86e4d9f1f8ac30fc7d65c31f6a6be881cf.jpg)  
Figure 6: Qualitative comparison on 10 raw FAUST scans. Color maps show the per-vertex asymmetric Chamfer distance to the ground-truth scan surface, in centimeters (cm).

Clothed Human Shapes and Scans. While the addition of clothes is an obvious next step for our vertex-based frameworks, preliminary results have only showed good predictions for ElaOpt so far; we provide some of those results and examples in Appendix D. We intend to pursue this both with the optimization framework and by a possible change of paradigm in Vertex-VaRefine.

## 12. Conclusion

This work proposed diferent approaches for the postcorrecting and refining of latent space representations for human body scans estimated by currently available latent code retrieval algorithms. We specifically address the problem of improving the code prediction via a correction within the considered latent space, which we complement with two diferent methods that operate at the vertex level and allow the capture of missing details beyond the limitations of the latent model itself.

The results reveal a practical trade-of between preserving the original parametric representation and improving geometric fidelity. Latent-VaRefine retains the model parameterization, Vertex-VaRefine provides strong vertex correspondence accuracy, and ElaOpt favors surface fidelity through direct geometric optimization. Their complementarity supports choosing the refinement strategy according to the requirements of the downstream application.

## 12.1. Acknowledgments

Chaima Zouaoui, Sylvain Arguillère, and Mohamed Daoudi were supported by the project 4DSHAPE ANR-24-CE23-5907 of the French National Research Agency (ANR). M. Bauer and J. Cerqueira were supported by the National Science Foundation through grant 2426549. N. Charon and E. Hartman were supported by the National Science Foundation through grant 2426550.

## 12.2. Statement on AI usage

During the preparation of this manuscript, the authors used AI tools solely for language editing to improve clarity and readability. AI tools were not used to generate scientific content, conduct analyses, or draw conclusions. The authors reviewed and verified all edits and take full responsibility for the content of the published article.

## References

Aigerman, N., Gupta, K., Kim, V.G., Chaudhuri, S., Saito, J., Groueix, T., 2022. Neural jacobian fields: learning intrinsic mappings of arbitrary meshes. ACM Trans. Graph. 41, 109:1–109:17. URL: https://doi.or g/10.1145/3528223.3530141, doi:10.1145/3528223.3530141.

Bauer, M., Charon, N., Harms, P., Hsieh, H.W., 2021. A numerical framework for elastic surface matching, comparison, and interpolation. International Journal of Computer Vision 129, 2425–2444.

Bhatnagar, B.L., Sminchisescu, C., Theobalt, C., Pons-Moll, G., 2020. Combining implicit function learning and parametric models for 3D human reconstruction, in: ECCV, Springer.

Bogo, F., Kanazawa, A., Lassner, C., Gehler, P., Romero, J., Black, M.J., 2016. Keep it SMPL: Automatic estimation of 3D human pose and shape from a single image, in: European Conference on Computer Vision (ECCV).

Bogo, F., Romero, J., Pons-Moll, G., Black, M.J., 2017. Dynamic FAUST: Registering human bodies in motion, in: IEEE Conf. on Computer Vision and Pattern Recognition (CVPR).

Cai, Z., Xiu, Y., Wang, R., Shao, Z., Li, X., Yu, S., Xu, C., Liu, Y., Sun, B., Yang, J., Zhang, Z., 2026. OmniFit: Multi-modal 3D Body Fitting via Scale-agnostic Dense Landmark Prediction, in: The European Conference on Computer Vision (ECCV).

Davydov, A., Remizova, A., Constantin, V., Honari, S., Salzmann, M., Fua, P., 2022. Adversarial parametric pose prior, in: IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR).

Feng, H., Kulits, P., Liu, S., Black, M.J., Abrevaya, V.F., 2023. Generalizing neural human fitting to unseen poses with articulated SE(3) equivariance, in: IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, IEEE. pp. 7943–7954.

Feng, Y., Choutas, V., Bolkart, T., Tzionas, D., Black, M.J., 2021. Collaborative regression of expressive bodies using moderation, in: International Conference on 3D Vision (3DV).

Ghorbani, S., Mahdaviani, K., Thaler, A., Kording, K., Cook, D.J., Blohm, G., Troje, N.F., 2020. MoVi: A large multipurpose motion and video dataset. arXiv preprint arXiv: 2003.01888 .

Gong, S., Chen, L., Bronstein, M., Zafeiriou, S., 2019. Spiralnet++: A fast and highly eficient mesh convolution operator, in: Proceedings of

![](images/9e9eb58c6f5f59ac8b5f0094f6275d589dc8283b15504e1e846386390a3d8359.jpg)

Figure 7: Comparison of the methods used in the ElaOpt term-wise ablation on four meshes. On the left in gray is the target to be optimized to and on the right, in green, is the base ArtEq mesh each method will be applied to. In red, from left to right are: ElaOpt, plain normal-constrained optimization, normal-constrained optimization with masks, with �<sup>1</sup>, with �<sup>2</sup>, and with SNRF. Note the numerical instability artifacts in the plain optimization as well as the distorted faces in those methods not using masks.

the IEEE International Conference on Computer Vision Workshops, pp. 0–0.

Hartman, E., Charon, N., Bauer, M., 2025. Self supervised networks for learning latent space representations of human body scans and motions. URL: https://arxiv.org/abs/2411.03475, arXiv:2411.03475.

Hartman, E., Pierson, E., 2023. Varigrad: A novel feature vector architecture for geometric deep learning on unregistered data, in: Fugacci, U., Lavoué, G., Veltkamp, R.C. (Eds.), 16th Eurographics Workshop on 3D Object Retrieval, 3DOR 2023 - Short Papers, Lille, France, August 31 - September 1, 2023, Eurographics Association.

Hartman, E., Sukurdeep, Y., Klassen, E., Charon, N., Bauer, M., 2023. Elastic shape analysis of surfaces with second-order sobolev metrics: A comprehensive numerical framework. International Journal of Computer Vision 131, 1183–1209.

Jermyn, I.H., Kurtek, S., Klassen, E., Srivastava, A., 2012. Elastic shape matching of parameterized surfaces using square root normal fields. Computer Vision – ECCV 2012 , 804–817.

Kaltenmark, I., Charlier, B., Charon, N., 2017. A general framework for curve and surface comparison and registration with oriented varifolds, in: Proceedings of the IEEE Conference on Computer Vision and Pattern

Recognition, pp. 3346–3355.

Kanazawa, A., Black, M.J., Jacobs, D.W., Malik, J., 2018. End-to-end recovery of human shape and pose, in: IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR).

Kurtek, S., Jermyn, I.H., Klassen, E., Srivastava, A., 2011. A riemannian framework for shape analysis of 3d surfaces. Computer Vision and Image Understanding 115, 1637–1651.

Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., Black, M.J., 2015. SMPL: A skinned multi-person linear model. ACM Transactions on Graphics (Proc. SIGGRAPH Asia) 34, 248:1–248:16.

Ma, Q., Yang, J., Ranjan, A., Pujades, S., Pons-Moll, G., Tang, S., Black, M.J., 2020. Learning to dress 3D people in generative clothing, in: Computer Vision and Pattern Recognition (CVPR).

Maesumi, A., Makadia, T., Groueix, T., Kim, V.G., Ritchie, D., Aigerman, N., 2025. Poissonnet: A local-global approach for learning on surfaces. ACM Transactions on Graphics (Proceedings ofSIGGRAPH Asia 2025)

Mahmood, N., Ghorbani, N., Troje, N.F., Pons-Moll, G., Black, M.J., 2019. AMASS: Archive of motion capture as surface shapes, in: International Conference on Computer Vision (ICCV), pp. 5442–5451.

Nicolet, B., Jacobson, A., Jakob, W., 2021. Large steps in inverse rendering of geometry. ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia) 40.

Osman, A.A.A., Bolkart, T., Black, M.J., 2020. STAR: Sparse trained articulated human body regressor, in: European Conference on Computer Vision (ECCV).

Pavlakos, G., Choutas, V., Ghorbani, N., Bolkart, T., Osman, A.A.A., Tzionas, D., Black, M.J., 2019. Expressive body capture: 3D hands, face, and body from a single image, in: IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR).

Romero, J., Tzionas, D., Black, M.J., 2017. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics (Proc. SIGGRAPH Asia) 36.

Sharp, N., Attaiki, S., Crane, K., Ovsjanikov, M., 2022. Difusionnet: Discretization agnostic learning on surfaces. ACM Transactions on Graphics (TOG) 41, 1–16.

Sharp, N., Crane, K., 2020. A Laplacian for Nonmanifold Triangle Meshes. Computer Graphics Forum (SGP) 39.

Tiwari, G., Antic, D., Lenssen, J.E., Sarafianos, N., Tung, T., Pons-Moll, G., 2022. Pose-NDF: Modeling human pose manifolds with neural distance fields, in: European Conference on Computer Vision (ECCV).

Yu, T., Zheng, Z., Guo, K., Liu, P., Dai, Q., Liu, Y., 2021. Function4d: Realtime human volumetric capture from very sparse consumer rgbd sensors, in: IEEE Conference on Computer Vision and Pattern Recognition (CVPR2021).

Zhao, K., Petrovich, M., Zhang, H., Wang, T., Tang, S., Rempe, D., 2026. Ardy: Autoregressive difusion with hybrid representation for interactive human motion generation. ACM Transactions on Graphics (TOG) 45. doi:10.1145/3811284.

## Appendix

## A. Distances Between Surface Meshes

In this section, we go over the diferent notions of distances between surface meshes (registered or not) that are used throughout the paper, both to quantify reconstruction errors and as loss functions for the proposed neural network models. In all the following, we consider two meshes with vertex sets $\mathbf { V } = \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { M }$ and $\mathbf { V } ^ { \prime } = \{ \mathbf { x } _ { j } ^ { \prime } \} _ { j = 1 } ^ { N } .$

Vertex-to-Vertex Distance (V2V). When the two meshes share the same topology and their vertices are in one-to-one correspondence, then the vertex-to-vertex error measures the average Euclidean distance between corresponding vertices:

$$
\mathrm { V } 2 \mathrm { V } ( \mathbf { V } , \mathbf { V } ^ { \prime } ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left\| \mathbf { x } _ { i } - \mathbf { x } _ { i } ^ { \prime } \right\| _ { 2 }
$$

where $\| \cdot \| _ { 2 }$ denotes the Euclidean norm. Throughout the article, results using this measurement will be reported in centimeters.

Asymmetric Chamfer Distance (CD). A common alternative to the above is the asymmetric Chamfer Distance defined by

$$
\mathrm { C D } ( { \bf V } , { \bf V } ^ { \prime } ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \operatorname* { m i n } _ { 1 \leq i \leq M } \left\| { \bf x } _ { j } ^ { \prime } - { \bf x } _ { i } \right\| _ { 2 }
$$

The asymmetric Chamfer Distance measures how well the surface � covers $\mathbf { V } ^ { \prime } ,$ , and does not require vertex correspondences or shared topology. Being one-directional, it is robust to case where the target mesh is incomplete or contains holes. One can also define the symmetric Chamfer distance by simply summing the two asymmetric distances i.e. $\operatorname { S C D } ( V , V ^ { \prime } ) \ = \ \operatorname { C D } ( V , V ^ { \prime } ) + \operatorname { C D } ( V ^ { \prime } , V )$ . Results for either version of the Chamfer distance will be reported in centimeters.

Vertex-to-Vertex Supremum. Under the same correspondence assumption as the V2V error (Section A), replacing the average with a maximum gives the worst-case point-wise deviation:

$$
\mathrm { V } 2 \mathrm { V } _ { \mathrm { s u p } } ( \mathbf { V } , \mathbf { V } ^ { \prime } ) = \operatorname* { m a x } _ { 1 \leq i \leq M } \left\| { \bf x } _ { i } - { \bf x } _ { i } ^ { \prime } \right\| _ { 2 }
$$

Unlike the mean V2V error, this metric is highly sensitive to localized reconstruction failures, such as a single poorly reconstructed limb or extremity. Results are reported in centimeters.

Hausdorf Distance. The Hausdorf distance is a symmetric, worst-case counterpart to the Chamfer distance (Section A): rather than averaging nearest-neighbor distances, it retains their maximum in each direction, then takes the larger

of the two:

$$
d _ { H } ( \mathbf { V } , \mathbf { V } ^ { \prime } ) = \operatorname* { m a x } \Big ( \operatorname* { m a x } _ { 1 \leq i \leq M } \operatorname* { m i n } _ { 1 \leq j \leq N } \left\| \mathbf { x } _ { i } - \mathbf { x } _ { j } ^ { \prime } \right\| _ { 2 } ,
$$

Like the Chamfer distance, it does not require vertex correspondence, but it is considerably more sensitive to outliers and localized failures. Results are reported in centimeters.

Varifold Distances. The varifold distance $d _ { \sigma } ^ { \operatorname { V a r } }$ is defined in Eq. (3) (Section 4) and compares both the positions and the orientations of the surfaces being compared. We consider its two standard variants: the oriented distance, obtained with the orientation kernel $\Gamma ( N \cdot N ^ { \prime } ) = e ^ { ( N \cdot N ^ { \prime } - 1 ) / \tau ^ { 2 } }$ used throughout the paper, and the unoriented distance, obtained with $\Gamma ( N \cdot N ^ { \prime } ) = ( N \cdot N ^ { \prime } ) ^ { 2 }$ . The latter depends only on the square of the inner product between normals and is therefore invariant to a flip of the normal orientation of either mesh.

## B. SMPL to SMPL-H and SMPL-X Converters

The parameters of SMPL latent codes can be used naively as SMPL-X or SMPL-H parameters. One simply matches up shape (�) parameters and assigns the pose (�) parameters of one latent space to the relevant pose parameters of the other. The immediate issue is the simple architectural diferences between the diferent latent spaces. SMPL ofers 6 coeficients for hands, but SMPL-X and SMPL-H ofer as many as 90, 45 for either hand, if one does not use PCA. SMPL-X adds expression coeficients for the face and parameters for jaw and eye position. Even if one chooses, as a matter of convention, to simply discard hand and expression information and match only the common joints and the shape parameters, the result is not the identity. That is, the exact mesh encoded by a set of shape (�) and joint (�) parameters depends on the decoder used even if one restricts to parameters held in common. See Figure 8 for a typical example.

To avoid retraining ArtEq for SMPL-X or performing long intensive computations to find optimal matchings between the thousands of relevant SMPL and SMPL-X parameters, we resort to training a simple neural network for the task. The fact that makes this workable is that while the meshes encoded by SMPL and SMPL-X by a common set of parameters will difer, they do not difer extremely. The poses are nearly right. Additionally, there is a matrix, which we will here denote , provided by the SMPL(- X) ecosystem to convert a generated SMPL-X mesh into one with the topology of an SMPL mesh. This allows us to train an autoencoder to find parameters correspondences with MSE loss:

$$
\operatorname { L o s s } ( C ^ { \prime } ) = \operatorname { M S E } ( \operatorname { D e c } ^ { \operatorname { S M P L } } ( C ) , \mathcal { M } \operatorname { D e c } ^ { \operatorname { S M P L X } } ( C ^ { \prime } ) ) .
$$

These losses are computed after an area-weighted centroid based centering, as SMPL and SMPL-X do not have a common center.

![](images/ea88ebe217799b76e34fa0f25b4311b85a8fd8bfb5e5ff5425f5cf8942bea807.jpg)  
Figure 8: The above is an SMPL mesh (left) in gray along with the results of naive parameter reuse into SMPL-X (center) and an SMPL-X mesh (right) that is the result of an optimization on the SMPL-X parameters to minimize MSE with the ground truth. As SMPL and SMPL-X have diferent topologies, this MVE and V2V distance are calculated after first applying SMPL’s supplied matrix to the mesh to convert its mesh topology to that of SMPL. The plotted error is V2V distance in meters under a log of base 10.

Naively attempting to train with the above loss with a single large network appears to be slow and prone to local optima. More efective is to train in three stages. We trained the model to convert SMPL latent codes with 10 shape parameters into SMPL-X latent codes with 10 shape parameters. The SMPL latent code is 82 parameters: 3 parameters corresponding to a global orientation, 63 corresponding to non-hand joints, 6 for SMPL’s simple hand joints, and 10 shape parameters. The SMPL-X latent code is 166 parameters: 3 parameters corresponding to a global orientation, 63 corresponding to non-hand joints, 90 for SMPL-H style hands, and 10 shape parameters. The first stage is a pair of MLPs, one MLP converting the 6 SMPL hand joint parameters into 90 SMPL-H style parameters and one MLP converting the 10 shape parameter from SMPL parameters to SMPL-X parameters. Training this stage alone will find parameters such that the shape will better match than naive parameter matching. The body pose can then be corrected by training an MLP to find a correcting term for the pose parameters. Finally, once both are roughly correct, the end to end combination of the two stages can be trained together for a final performance increase. The overall architecture is shown in Figure 9 and an example result can be seen in Figure 10

We can perform a nearly identical process to create a converter from SMPL parameters to SMPL-H parameters. As we did not use expression or jaw parameters above, we have the same 166 parameters for a 10-shape parameter SMPL-H latent code. Unlike in the SMPL-X case, there is no diference of mesh topology, so our loss is simply

$$
\operatorname { L o s s } ( C ^ { \prime } ) = \operatorname { M S E } ( \operatorname { D e c } ^ { \operatorname { S M P L } } ( C ) , \operatorname { D e c } ^ { \operatorname { S M P L H } } ( C ^ { \prime } ) ) .
$$

Naive parameter reuse is closer to the identity for SMPL-H. It largely loses only hand information. Due to this, the converter performs even better than in the SMPL-X case as can be seen in Figure 11.

![](images/c089601bf11e0627eff167bcff8ea6f23ad9edfbe32e2d023ec983e51314e36b.jpg)  
Figure 9: Overview of the autoencoder’s architecture. The upper left subnetwork is for $\beta$ conversion, the center left subnetwork is for hands. Together their outputs form an initial guess, with the pose and orientation retained naively. The pose parameters are then corrected by the larger network in the center of the figure.

![](images/9a30edf90b9659b32acce1dc64a50a3133ce451f7cd69305e0391dcf036422e5.jpg)  
Figure 10: The above is an SMPL mesh (left) in gray along with the results of naive parameter reuse in SMPL-X (center) and the SMPL-X mesh (right) that produced by the autoencoder. The result here is nearly as good as the optimized result above in Figure 8. The plotted error is V2V distance in meters under a log of base 10.

![](images/4f0ee60ba0771bb09fd4d1791447929d14ac6fe459749a0d42491123fed6ef71.jpg)  
Figure 11: Similar architecture may be trained to allow for conversion to SMPL-H. Note here the naive result is much closer to correct and the converter largely serves to fix the incorrect hands.

## C. Alternative Backbones for Vertex-VaRefine

The choice of DifusionNet as the backbone of Vertex-VaRefine results from the ablation study of Section 9.3. We describe here the two alternative backbones evaluated in place of DifusionNet, as well as the two decoding heads used to obtain �: a direct per-vertex displacement and a Neural Jacobian Field.

PoissonNet. PoissonNet (Maesumi et al., 2025) is an intrinsic mesh neural network that replaces the spectral diffusion operator of DifusionNet (Sharp et al., 2022) with a diferentiable Poisson equation solving as its mechanism for information propagation across the surface. Given pervertex scalar features $\begin{array} { r l r } { { \boldsymbol { x } } } & { { } \in } & { \mathbb { R } ^ { V \times C } } \end{array}$ , each Poisson block proceeds in three steps: (i) the intrinsic gradient operator $\dot { G } \in \mathbb { R } ^ { 2 F \times V }$ maps vertex features to per-face tangent-plane gradients, $\nabla x ~ = ~ G x$ , represented as complex numbers in a local 2D basis per face; (ii) a complex-valued MLP transforms these tangent-vector features while preserving the ��(2)-equivariance of the local frame; (iii) a Poisson equation $L u \ : = \ : \mathrm { d i v } ( A $ ⋅ grads) is solved in closed form via a pre-factorized (Cholesky) cotangent Laplacian � (Nicolet et al., 2021) – one factorization per mesh, reused across all channels and both forward/backward passes – to obtain a globally smoothed vertex signal �. The block output is $\mathbf { M L P } ( [ x , u ] ) + x$ . Stacking � such blocks after a linear input projection gives the PoissonNet backbone; a final linear head maps $C _ { \mathrm { w i d t h } }$ -dimensional vertex features to the task output.

SpiralNet. SpiralNet++ (Gong et al., 2019) ofers an attractive compromise: it defines a fast, anisotropic mesh convolution by following spiral sequences of neighbouring vertices on a mesh of fixed topology. In the case where the source shape has a fixed mesh template, like in ours (the source mesh  is always decoded from a given latent space and therefore has consistent mesh structure), the spirals can be precomputed once and reused across the dataset, leading to an eficient and topology-aware encoder particularly suited to body meshes. The final output then needs to be as detailed as possible, so we forego the usual downsampling of SpiralNet. This lets us directly obtain a per-vertex processed feature $\boldsymbol { z } = ( z _ { 1 } , \ldots , z _ { V } ) \in \mathbb { R } ^ { K \times V }$ . Finally, to prime this code for the next module, we turn it into per-face feature $Z = ( Z _ { 1 } , \dots , Z _ { F } ) \in \mathbb { R } ^ { K \times F }$ . For each face index $f = 1 \ldots F$ of the mesh, $Z _ { f }$ is the average of the three codes $z _ { i } s$ attached to the three vertices of that face.

Decoding Head: Direct Displacement vs. Neural Jacobian Fields. For DifusionNet and PoissonNet, the backbone directly outputs the per-vertex T-pose displacement $\textit { \textbf { D } } =$ Net $( \boldsymbol { \phi } ) \in \bar { \mathbb { R } } ^ { V \times 3 }$ through a final linear layer, with no further post-processing. For SpiralNet++, directly regressing � from per-vertex features produces noisy and spatially inconsistent corrections. To enforce smoothness without sacrificing expressivity, we instead adopt a slightly modified version of the Neural Jacobian Field (Aigerman et al., 2022) formulation on top of the SpiralNet++ encoder. Rather than predicting the displacement directly, the network outputs a per-triangle 3×3 Jacobian matrix $J _ { f } ~ = ~ \mathrm { M L P } ( Z _ { f } , c _ { f } )$ $f = 1 , \ldots , F .$ , from the per-face encoder features $\dot { Z _ { f } }$ and face centers $c _ { f }$ introduced above. The columns of $J _ { f }$ are interpreted as the surface gradient of each coordinate of the T-pose correction. The deformation is then recovered by solving the Poisson system

$$
L _ { S _ { 0 } } \Delta = \mathcal { A } _ { S _ { 0 } } \nabla _ { S _ { 0 } } ^ { T } J _ { f } ,
$$

where $L _ { S _ { 0 } }$ is the surface Laplacian, $\mathcal { A } _ { S _ { 0 } }$ the diagonal matrix of face areas, and $\nabla ^ { T }$ the transpose of the surface gradient operator, all computed once on a fixed reference template $S _ { 0 }$ (the template SMPL mesh) shared by every training sample. Since , and hence <sup>̄</sup>, always share the SMPL topology, this Poisson system is factorized only once, independently of the training set size, which keeps memory and training time tractable. Unlike the original, non-equivariant formulation, the resulting correction Δ is here added to the per-sample T-pose shape rather than to $S _ { 0 }$ itself, $D \ \doteq \ \Delta$ with $T =$ $\bar { S } + \Delta$ – this is what makes this decoding head compatible with the pose-equivariant pipeline described above. This approach has two advantages: it introduces a regularization prior on the deformation appropriate for body geometry, and it preserves the intrinsic diferential structure of the source surface, so fine-scale features are not distorted during refinement.

## D. Clothed Human Scans and Scan Animation

In a manner similar to SMPL+D and Bhatnagar et al. (2020), per vertex correction preserves the original joint rigging and skinning weights allowing the reconstruction be reposed or animated. Additionally, unlike methods constrained to the latent space’s representative limits, these vertex corrections can be applied to capture clothing or other details outside the latent space constraints. These are items we will explore in future work, and we present in this section some promising early proof-of-concept eforts.

Figure 12 a-b), presents qualitative reconstructions of scans from the CAPE (Ma, Yang, Ranjan, Pujades, Pons-Moll, Tang and Black, 2020) and THuman (Yu, Zheng, Guo, Liu, Dai and Liu, 2021) datasets. Starting from minimally clothed SMPL meshes, each method predicts deformation fields that recover the geometry of the target clothed scans. Quantitatively, the Vertex-VaRefine framework reduces the vertex-to-vertex Error (V2V) from 1.74 cm to 1.61 cm, corresponding to a relative improvement of approximately 7.5%. Meanwhile, the ElaOpt method reduces the V2V to 1.41 cm corresponding to a relative improvement of 19%. Qualitatively, VaRefine captures the overall clothing geometry, but not enough to convincingly dress the source mesh. Fine-scale cloth wrinkles and high-frequency surface details remain challenging. ElaOpt on the other hand, yields much better-looking results, both in terms of large scale geometry and wrinkles.

Since the ElaOpt method deforms the shape in the normal direction of the SMPL representation, these deformations can be applied to a sequence of pose parameters to produce 4D sequences of clothed meshes. A qualitative evaluation of the ElaOpt method’s ability to capture clothing details and an example of a resulting transfer of clothes to a 4D sequences are presented in Figure 12 c).

![](images/1710bed70df55d254d249aaa262ebeafda2ea741a42b8c552b68df45387d2a0b.jpg)  
Figure 12: In a), we present six examples from the CAPE Dataset. The ground truth clothed meshes are in red, the SMPL approximations are presented in yellow, the refined meshes via Vertex-Varefine are presented in blue, and the refined meshes produced by $\mathsf { E l a O p t }$ are presented in green. In b), we present two examples from the THumans dataset. Again, the raw scans are displayed in red, the SMPL approximations are displayed in yellow, and the reconstruction using ElaOpt is displayed in green. In c), we display the transfer of a motions from the CMU dataset to the clothed meshes using the outputs of the ElaOpt method from b).

## E. Additional Results

This appendix provides the complete quantitative evaluation of Latent-VaRefine, Vertex-VaRefine, ElaOpt, Latent-VaRefine + ElaOpt, and Latent-VaRefine + Vertex-VaRefine on all evaluated AMASS sub-datasets. Tables 13 and 14 list the detailed reconstruction errors for the remaining AMASS sub-datasets not reported in Section 8.3 of the main paper.

Table 13  
Detailed evaluation on additional AMASS sub-datasets (SMPL target). We report vertex-to-vertex (V2V, in cm) for ArtEq, Latent-VaRefine, ElaOpt, and Vertex-VaRefine together with the corresponding relative improvements.
<table><tr><td>Dataset</td><td>ACCAD</td><td>BML_rub</td><td>CMU</td><td>CNRS</td><td>DFAUST</td><td>DanceDB</td></tr><tr><td>ArtEq</td><td>2.615</td><td>3.750</td><td>3.054</td><td>3.426</td><td>2.017</td><td>4.900</td></tr><tr><td>Latent-VaRefine Improvement (%)</td><td>2.329 10.9</td><td>2.917 22.2</td><td>2.463 19.4</td><td>3.216 6.1</td><td>1.729 14.3</td><td>4.533 7.5</td></tr><tr><td>Vertex-VaRefine</td><td>1.447</td><td>1.653</td><td>1.437</td><td>1.627</td><td>1.100</td><td>3.149</td></tr><tr><td>Improvement (%) ElaOpt</td><td>44.7 1.983</td><td>55.9 3.217</td><td>52.9 2.564</td><td>52.5 2.911</td><td>45.5 1.622</td><td>35.7 3.327</td></tr><tr><td>Improvement (%) Latent-VaRefine + ElaOpt</td><td>24.2 1.695</td><td>14.2 2.464</td><td>16.0 1.938</td><td>15.0 2.633</td><td>19.6 1.177</td><td>32.1 2.812</td></tr><tr><td>Improvement (%) Latent-VaRefine + Vertex-VaRefine</td><td>35.2 1.665</td><td>34.3 1.875</td><td>36.5</td><td>23.1</td><td>41.6</td><td>42.6</td></tr><tr><td>Improvement (%)</td><td>36.3</td><td>50.0</td><td>1.736 43.2</td><td>1.784 47.9</td><td>1.438 28.7</td><td>3.459 29.4</td></tr><tr><td>Dataset</td><td>EYES_JAPAN Human4D</td><td></td><td>KIT</td><td>MPI_HDM</td><td>MPI Limits</td><td>MPI_mosh</td></tr><tr><td>ArtEq</td><td>5.400</td><td>3.774</td><td>2.595</td><td>3.994</td><td>4.299</td><td>2.849</td></tr><tr><td>Latent-VaRefine Improvement (%)</td><td>4.356 19.3</td><td>3.402 9.9</td><td>2.083</td><td>3.047</td><td>4.045</td><td>2.429</td></tr><tr><td>Vertex-VaRefine</td><td>2.854</td><td>1.912</td><td>19.7 1.555</td><td>23.7 2.091</td><td>5.9 2.864</td><td>14.7 1.400</td></tr><tr><td>Improvement (%)</td><td>47.1</td><td>49.3</td><td>40.1</td><td>47.6</td><td>33.4</td><td>50.9</td></tr><tr><td>ElaOpt Improvement (%)</td><td>4.541 15.9</td><td>3.102 17.8</td><td>2.178 16.1</td><td>3.463</td><td>2.449 43.0</td><td>2.155 24.4</td></tr><tr><td>Latent-VaRefine + ElaOpt</td><td>3.488</td><td>2.563</td><td>1.658</td><td>13.3 2.826</td><td>2.451</td><td>2.194</td></tr><tr><td>Improvement (%)</td><td>35.4</td><td>32.1</td><td>36.1</td><td>29.2</td><td>43.0</td><td>23.0</td></tr><tr><td>Latent-VaRefine + Vertex-VaRefine</td><td>3.127</td><td>2.304</td><td>1.467</td><td>2.066</td><td>3.322</td><td>1.813</td></tr><tr><td>Improvement (%)</td><td>42.1</td><td>39.0</td><td>43.5</td><td>48.3</td><td>22.7</td><td>36.4</td></tr><tr><td>Dataset</td><td>SOMA</td><td>Transitions BML</td><td></td><td></td><td>_handball WEIZMANN TotalCapture</td><td>SSM</td></tr><tr><td>ArtEq</td><td>2.650</td><td>2.940</td><td>2.667</td><td>2.989</td><td>3.244</td><td>4.321</td></tr><tr><td>Latent-VaRefine</td><td>2.427</td><td>2.200</td><td>2.187</td><td></td><td>2.290</td><td>2.601</td></tr><tr><td>Improvement (%)</td><td>8.4</td><td>25.2</td><td>18.0</td><td>2.192 26.7</td><td>29.4</td><td>39.8</td></tr><tr><td>Vertex-VaRefine</td><td>1.271</td><td>1.073</td><td>1.419</td><td>1.152</td><td>1.123</td><td>1.647</td></tr><tr><td>Improvement (%)</td><td>52.0</td><td>63.5</td><td>46.8</td><td>61.5</td><td>65.4</td><td>61.9</td></tr><tr><td>ElaOpt</td><td>2.257</td><td>2.209</td><td>2.029</td><td>2.587</td><td>2.950</td><td>3.877</td></tr><tr><td>Improvement (%)</td><td>14.8</td><td>24.9</td><td>23.9</td><td>13.4</td><td>9.06</td><td>10.3</td></tr><tr><td>Latent-VaRefine + ElaOpt</td><td>1.961</td><td>1.431</td><td>1.757</td><td>1.741</td><td>1.634</td><td>1.837</td></tr><tr><td>Improvement (%)</td><td>26.0</td><td>51.3</td><td>34.1</td><td>41.8</td><td>49.6</td><td>57.5</td></tr><tr><td>Latent-VaRefine + Vertex-VaRefine</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>1.599</td><td>1.301</td><td>1.512</td><td>1.491</td><td>1.447</td><td>1.893</td></tr><tr><td>Improvement (%)</td><td>39.7</td><td>55.7</td><td>43.3</td><td>50.1</td><td>55.4</td><td>56.2</td></tr></table>

Table 14  
Detailed evaluation on additional AMASS sub-datasets (SMPL target). We report asymmetric chamfer distance (in cm) for ArtEq, Latent-VaRefine, ElaOpt, and Vertex-VaRefine together with the corresponding relative improvements.
<table><tr><td>Dataset</td><td>ACCAD</td><td>BML_rub</td><td>CMU</td><td>CNRS</td><td>DFAUST</td><td>DanceDB</td></tr><tr><td>ArtEq</td><td>1.409</td><td>1.955</td><td>1.645</td><td>1.443</td><td>1.245</td><td>2.627</td></tr><tr><td>Latent-VaRefine Improvement (%)</td><td>1.302 7.59</td><td>1.605 17.9</td><td>1.374 16.5</td><td>1.396 3.26</td><td>1.119 10.1</td><td>2.436 7.27</td></tr><tr><td>Vertex-VaRefine</td><td>0.893</td><td>0.987</td><td>0.875</td><td>0.898</td><td>0.776</td><td>1.730</td></tr><tr><td>Improvement (%) ElaOpt</td><td>36.6 0.705</td><td>49.5 0.876</td><td>46.8 0.763</td><td>37.8 0.703</td><td>37.7</td><td>34.1</td></tr><tr><td>Improvement (%)</td><td>50.0</td><td>55.2</td><td>53.6</td><td>51.3</td><td>0.637 48.8</td><td>0.898 65.8</td></tr><tr><td>Latent-VaRefine + ElaOpt Improvement (%)</td><td>0.674 52.2</td><td>0.783 59.9</td><td>0.710 56.8</td><td>0.707 51.0</td><td>0.604 51.5</td><td>0.804 69.4</td></tr><tr><td>Latent-VaRefine + Vertex-VaRefine Improvement (%)</td><td>0.983 30.2</td><td>1.082 44.7</td><td>1.018 38.1</td><td>0.945</td><td>0.926</td><td>1.918</td></tr><tr><td>Dataset</td><td>EYES_JAPAN Human4D</td><td></td><td>KIT</td><td>34.5</td><td>25.6</td><td>27.0</td></tr><tr><td>ArtEq</td><td>2.638</td><td>1.927</td><td>1.375</td><td>MPI HDM 1.853</td><td>MPI Limits</td><td>MPI mosh</td></tr><tr><td>Latent-VaRefine</td><td>2.182</td><td>1.753</td><td>1.194</td><td>1.504</td><td>2.089 1.933</td><td>1.606 1.395</td></tr><tr><td>Improvement (%) Vertex-VaRefine</td><td>17.3 1.410</td><td>9.0 1.084</td><td>13.2 0.947</td><td>18.8 1.091</td><td>7.5 1.396</td><td>13.1</td></tr><tr><td>Improvement (%) ElaOpt</td><td>46.6 1.112</td><td>43.7 0.901</td><td>31.1</td><td>41.1</td><td>33.2</td><td>0.917 42.9</td></tr><tr><td>Improvement (%)</td><td>57.8</td><td>53.2</td><td>0.713 48.1</td><td>0.818 55.9</td><td>0.711 66.0</td><td>0.682 57.5</td></tr><tr><td>Latent-VaRefine + ElaOpt Improvement (%)</td><td>0.680 74.2</td><td>0.776 59.7</td><td>0.658 52.1</td><td>0.735 60.3</td><td>0.717 65.7</td><td>0.680 57.7</td></tr><tr><td>Latent-VaRefine + Vertex-VaRefine Improvement (%)</td><td>1.542 41.5</td><td>1.225 36.4</td><td>0.900 34.5</td><td>1.101</td><td>1.600</td><td>1.078</td></tr><tr><td>Dataset</td><td>SOMA</td><td>Transitions BML</td><td></td><td>40.6</td><td>23.4 _handball WEIZMANN TotalCapture</td><td>32.9 SSM</td></tr><tr><td>ArtEq</td><td>1.532</td><td>1.664</td><td>1.524</td><td>1.664</td><td>1.631</td><td>2.308</td></tr><tr><td>Latent-VaRefine</td><td>1.418 7.4</td><td>1.294</td><td>1.255</td><td>1.379</td><td>1.289</td><td>1.605</td></tr><tr><td>Improvement (%) Vertex-VaRefine</td><td>0.849</td><td>22.2 0.742</td><td>17.7 0.891</td><td>17.1</td><td>21.0</td><td>30.5</td></tr><tr><td>Improvement (%)</td><td>44.6</td><td>55.4</td><td>41.5</td><td>0.805 51.6</td><td>0.765 53.1</td><td>1.045 54.7</td></tr><tr><td>ElaOpt Improvement (%)</td><td>0.728 52.5</td><td>0.752 54.8</td><td>0.668 56.2</td><td>0.823 50.5</td><td>0.797 51.1</td><td>1.034 55.2</td></tr><tr><td>Latent-VaRefine + ElaOpt</td><td>0.700</td><td>0.651</td><td>0.651</td><td>0.714</td><td>0.671</td><td>0.685</td></tr><tr><td>Improvement (%)</td><td>54.3</td><td>60.9</td><td>57.3</td><td>57.1</td><td>58.9</td><td>70.3</td></tr><tr><td>Latent-VaRefine + Vertex-VaRefine Improvement (%)</td><td>0.984 35.8</td><td>0.855</td><td>0.923</td><td>0.931</td><td>1.317</td><td>1.184</td></tr><tr><td></td><td></td><td>48.6</td><td>39.4</td><td>44.1</td><td>19.3</td><td>48.7</td></tr></table>