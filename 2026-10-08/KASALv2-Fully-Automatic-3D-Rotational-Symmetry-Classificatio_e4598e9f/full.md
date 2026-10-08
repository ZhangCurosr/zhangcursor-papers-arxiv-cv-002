# KASALv2: Fully Automatic 3D Rotational Symmetry Classification and Axis Localization

Mengxin Zhang<sup>\*</sup> Yulin Wang<sup>\*</sup> Chen Luo<sup>†</sup> Yongzhe Li Yijun Zhou<sup>†</sup> School of Mechanical Engineering, Southeast University Nanjing, China

{mx.zhang, yulinwang, chenluo, yongzheli, zhouyj}@seu.edu.cn

## Abstract

Rotational symmetry is an important prior in 6D pose estimation, improving pose accuracy and supporting symmetry-aware evaluation. However, current symmetry annotations for 3D objects remain largely manual or semi-automatic, often requiring predefined types or orders, which limits scalability. This work introduces a fully automatic, reference-free framework for symmetry-type classi fication, rotational-order identification, andfull-axis localization across all eight canonical 3D rotational symmetry types. The method localizes a dominant high-order axis, infers its rotational order through self-consistency analysis, and reconstructs the complete symmetry structure under a hierarchy-guidedformulation. A texture-aware extension further models appearance-induced reductions in rotational order while preserving axis orientations. Experiments on idealized and real-world datasets demonstrate strong accuracy and generalization, achieving 94.75% accuracy on 438 symmetric objects in GSO. Training FoundationPose with these priors improves accuracy by up to 0.9% across five BOP datasets, showing that automatically estimated rotational priors improve downstream 6D pose estimation. Code is available at https://github.com/ WangYuLin-SEU/KASAL.

## 1. Introduction

Rotational symmetry is a key geometric prior in 6D pose estimation[1–7]. It resolves pose ambiguity, stabilizes optimization, and supports symmetry-aware metrics such as ADD-S[8], MSSD, and MSPD[9]. Recent systems showing HccePose(BF)[10] and ZebraPose[11] use symmetry priors to improve accuracy, indicating that reliable symmetry information benefits high-precision estimation. Meanwhile, the field is moving toward unseen-object generalization and large-scale synthetic training. These trends rely on datasets with tens of thousands of 3D models, where per-object rotational symmetry is needed but manual annotation is infeasible. Such scalable symmetry annotation also benefits mesh generation and large-scale 3D asset processing.

![](images/adecb31251ecf4712fb9314c8d82cd3bb8b7c06e5404140a015b3885fcd19a28.jpg)  
Figure 1. Comparative landscape of rotational symmetry analysis methods categorized by geometric reference dependency and task capability. The proposed method supports all eight canonical types without geometric references and handles texture-induced symmetries.

In addition, rotational symmetry is structurally diverse, spanning eight canonical forms[12–16] with distinct axis multiplicities, rotational orders, and geometric configurations. This diversity produces many subtypes and requires a unified framework capable of joint reasoning over type and axis structure. However, as shown in Fig. 1, existing methods remain limited in automation and coverage: many address only subsets of symmetry types,[17–23] others depend on geometric references,[24, 25] and recent all-type approaches[26] still require manual specification of rotational orders or categories. For large-scale unseen-object pose estimation, an automatic system capable of full symmetry classification and axis localization is needed.

To address this requirement, we extend the key-axisbased localization method of Wang et al. [26], which relies on manually specified symmetry types and orders, and present KASALv2, a unified framework for complete rotational-symmetry inference across all eight canonical symmetry types without any prior input or predefined geometric references. KASALv2 transforms the task from type-conditioned axis localization into fully automatic analysis that jointly determines the symmetry type, rotational order, and the full set of canonical axes. Beyond removing manual input, it replaces the exhaustive directional enumeration of [26] with the Geometric Degeneration Hierarchy (GDH), a computable reasoning mechanism that encodes type-specific geometric relations and guides the symmetry analysis. Guided by GDH, the system localizes a dominant high-order axis, estimates its rotational order through periodic self-consistency, infers the corresponding symmetry family, and reconstructs all canonical axes in a unified manner, with a texture-aware extension that accounts for appearance-induced reductions in rotational order.

Extensive experiments on DSRSTO[26], BOP[27], and GSO[28] validate generalization and scalability, achieving 94.75% accuracy on real-world symmetric objects and demonstrating robust performance on the DSRSTO texture subset under appearance-induced degradation. When used as symmetry priors for FoundationPose,[29] the method improves BOP accuracy by up to 0.9%, confirming the value of precise symmetry estimation for downstream pose prediction. Our contributions are summarized as follows:

• A unified and fully automatic framework for 3D rotational symmetry analysis that performs symmetry-type classification, rotational-order identification, and fullaxis localization across all eight canonical types in a reference-free pipeline.

• A geometry-driven formulation, the Geometric Degeneration Hierarchy (GDH), introduced as a computable geometric hierarchy that organizes rotational symmetries via structured axis reduction and provides the algorithmic pathway for mapping an unknown object to its canonical symmetry family.

• An operational instantiation of GDH enabling consistent type determination, stable high-order axis localization, and a texture-aware extension that models appearance-induced reductions in rotational order while preserving axis orientations.

• A scalable and annotation-free source of rotational symmetry priors for unseen-object pose estimation. Training FoundationPose with these priors on a onemillion-image corpus yields up to a 0.9% gain on BOP, highlighting the value of symmetry-aware augmentation for foundation-level models.

## 2. Related Work

Rotational symmetry analysis in 3D vision comprises two fundamental tasks: symmetry-type identification[17, 18,

23–25], which infers the underlying rotational structure from data, and symmetry-axis localization[19–22, 26], which assumes the type is known and estimates its corresponding axes. Existing approaches addressing one or both tasks vary mainly along two design dimensions: (i) reliance on geometric priors such as centroids, keypoints, or object frames, and (ii) coverage of the eight canonical rotational symmetry types. These dimensions strongly influence robustness and generality.

## 2.1. Symmetry Type Identification

Symmetry type detection identifies whether an object exhibits rotational symmetry and, if so, infers its specific type and order. Existing approaches fall into two categories: reference-based and reference-free. Reference-based methods [24, 25] rely on geometric anchors such as centroids, midpoints, or principal axes. They perform well under controlled conditions but are highly sensitive to surface noise, occlusion, and misalignment, which limits their applicability in real-world 6D pose estimation. Reference-free methods [17, 18, 23] infer symmetry directly from surface geometry through optimization, spectral analysis, or voting. They achieve greater robustness by avoiding predefined anchors but still lack generality across all canonical rotational types. For example, Hruda et al.[18] proposed a reflection-based framework that fails on continuously symmetric objects such as cylinders or cones, while Rajkumar et al.[17] presented a hybrid approach limited to continuous and second-order discrete conical symmetries.

## 2.2. Symmetry Axis Localization

Most existing methods address only limited cases[19–22, 30] and are typically designed for specific scenarios. For example, Sipiran[20, 21] proposed approaches for cultural heritage restoration that focuses on continuous types. These methods, however, cannot accommodate texture-induced symmetries. Although Wang et al.[26] introduced a method capable of handling all symmetry types, including texturerelated cases, it still requires manual specification of symmetry type and order. This dependency hinders full automation and leads to annotation ambiguity, particularly for objects with high-order or composite symmetries that even experts may find difficult to classify accurately.

## 2.3. Rotational Symmetry in Pose Estimation

In 6D pose estimation, existing methods for handling rotationally symmetric objects fall into two categories. The first does not use explicit symmetry priors[3, 4, 31, 32] but incorporates architectural or learning designs that implicitly capture latent symmetry cues. Examples include EPOS[32] and GDR-Net[31]. The second[1, 2, 7, 10, 11, 33, 34] explicitly leverages known symmetry priors to resolve pose ambiguities and improve accuracy. Methods such as Hcce-

![](images/95e1ef95d6b583a49d08bed02cfcdeec577382fe703cb43b287f49b82346917c.jpg)  
Figure 2. The pipeline integrates a Geometric Degeneration Hierarchy (GDH) and an operational framework. The GDH establishes the hierarchical taxonomy of 3D rotational symmetry, while the operational framework implements it. Given a 3D object of unknown symmetry, the system first localizes a dominant high-order axis and infers its rotational order through self-consistency optimization. Based on this order, candidate symmetry types are derived and verified by checking whether a secondary axis satisfies the corresponding geometric constraints. This hierarchical verification yields type classification and reconstruction of all canonical axes under the identified geometric relations. Finally, the texture-symmetry module refines the classification by modeling appearance-induced degradation, covering reduced invariant, and discretized texture cases.

Pose (BF)[10] and ZebraPose[11] use predefined symmetry matrices to transform ground-truth rotations, selecting the variant closest to the identity as the training target. This strategy reduces symmetry-induced ambiguity and leads to more stable training and higher accuracy than prior-free methods. Symmetry priors also underpin evaluation protocols. Metrics such as ADD-S[8], MSSD[9], and MSPD[9] define pose equivalence based on symmetry-aware criteria; without them, physically identical yet numerically distinct poses would be misjudged. Hence, symmetry priors are essential not only for accurate estimation but also for consistent evaluation.

Despite their importance, existing symmetry annotations remain largely manual or require predefined types or orders, limiting scalability. These limitations motivate the development of a fully automatic framework for robust object symmetry analysis without human intervention. To this end, KASALv2 is introduced, enabling automatic detection of symmetry types and precise axis localization across all eight canonical rotational types, thereby providing reliable symmetry priors for downstream pose estimation.

## 3. Methodology

Our framework performs reference-free rotational symmetry analysis for 3D objects. Central to the method is the Geometric Degeneration Hierarchy (GDH), a computable structure encoding all canonical rotational symmetry families and the degeneration conditions between them. Guided by GDH, the method localizes the dominant high-order axis, estimates its order, determines the symmetry type, and reconstructs the full canonical axis configuration. A texture-aware extension further models appearance-induced order reductions while preserving geometric axis orientations. The overall workflow is shown in Fig. 2.

This section follows a theory-to-operation progression. Sec. 3.1 introduces GDH, which defines the structural relations and degeneration pathways among all rotational symmetry families. Sec. 3.2 and Sec. 3.3 then apply these relations to localize the dominant axis, infer its order, classify the symmetry type, and recover all canonical axes. Sec. 3.4 extends the framework to appearance-aware settings using texture-induced order reductions.

## 3.1. Geometric Degeneration Hierarchy (GDH) for 3D Rotational Symmetry

Rotational symmetry in three-dimensional space corresponds to the orientation-preserving subgroups of SO(3), including the continuous types $C _ { \infty }$ and $D _ { \infty }$ , the finite cyclic and dihedral groups $C _ { n }$ and $D _ { n }$ , and the rotational groups of the Platonic solids $( A _ { 4 } , S _ { 4 } , A _ { 5 } )$ [12–16, 35, 36]. These families cover all canonical rotational symmetries considered in this work, with classical subgroup embeddings such as $A _ { 4 } ~ \subset ~ S _ { 4 }$ and $A _ { 4 } ~ \subset ~ A _ { 5 } ~ [ 3 7$ , 38]. Building on this foundation, we introduce the Geometric Degeneration Hierarchy (GDH), a geometry-driven and algebraically complete hierarchy that reorganizes these $S O ( 3 )$ subgroups into an operational form. Rather than a purely conceptual taxonomy, GDH encodes how axis multiplicity, rotational order, and characteristic inter-axis relations evolve under geometric degenerations, providing computable constraints that restrict feasible symmetry families and support typespecific axis inference in our fully automatic framework.

At the top of this hierarchy lies the maximally symmetric sphere, where every direction is a valid rotational axis. From this state, GDH considers two geometric degeneration lineages: the dihedral–cyclic family, characterized by reductions from $C _ { \infty }$ and $D _ { \infty }$ to their finite counterparts, and the Platonic family, representing discrete reductions of multi-axis structures such as $A _ { 5 } , S _ { 4 }$ , and $A _ { 4 }$

The dihedral–cyclic lineage begins with the cylindrical form $D _ { \infty }$ , characterized by continuous rotation about a single principal axis. From this state, symmetry may evolve along two geometric reduction paths. The first is the discretization of continuous rotation,

$$
D _ { \infty } \Rightarrow D _ { n } , \qquad C _ { \infty } \Rightarrow C _ { n } , \qquad n \in \mathbb { N } , \ n \geq 2 ,\tag{1}
$$

producing finite n-fold rotational structures with or without orthogonal two-fold axes. This discretization relation allows the dominant axis order to restrict the feasible discrete families during type inference.

The second path corresponds to the loss of orthogonal axes,

$$
D _ { \infty } \Rightarrow C _ { \infty } , \qquad D _ { n } \Rightarrow C _ { n } ,\tag{2}
$$

yielding single-axis rotational configurations.

In parallel, the Platonic lineage models the geometric reduction of multiple high-order axes present in fully symmetric structures. This leads to the icosahedral $\left( A _ { 5 } \right)$ , octahedral (S ), and tetrahedral $\left( A _ { 4 } \right)$ symmetries, which exhibit progressively fewer and lower-order rotational axes:

$$
A _ { 5 } \Rightarrow A _ { 4 } , \qquad S _ { 4 } \Rightarrow A _ { 4 } .\tag{3}
$$

Each downward transition reflects a geometric simplification in both the number and orientation of rotational axes. Together, the dihedral–cyclic and Platonic lineages form the Geometric Degeneration Hierarchy (GDH), a geometrybased organizational framework encompassing all eight canonical types of 3D rotational symmetry.

Within this hierarchy, the localized dominant high-order axis provides the primary constraint that narrows the feasible subtype of symmetry types, while its estimated rotational order further restricts the set of candidates. The identification of a secondary axis that satisfies the corresponding geometric relations is then used to verify the specific type. Once the type is determined, its intrinsic axis relations uniquely specify the remaining canonical axes, enabling complete reconstruction of the object’s rotational structure.

![](images/4299ba2ff7bc4a11916a1b49a6741d703459693fd3d9b5ddc4bee7f9c1eca337.jpg)  
Figure 3. Aggregated alignment-loss distribution on the unit sphere. Low-loss regions (orange) cluster around the true highorder axis, illustrating the self-consistent nature of high-order symmetries.

Through these components, GDH supplies a structured set of geometric constraints that make the subsequent classification and axis reconstruction steps operationally tractable within our fully automatic framework.

## 3.2. High-order Axis Localization and Order Identification

## 3.2.1. Dominant Axis Localization

Guided by the GDH framework introduced in Sec. 3.1, this stage localizes the dominant high-order rotational axis, which anchors subsequent order identification and type classification. Unlike reference-based methods that depend on pre-aligned templates or centroids, our approach operates entirely in a reference-free manner by optimizing a global alignment objective directly on the unit sphere.

## Alignment Principle.

To locate the true rotational axis without prior knowledge of symmetry order, candidate directions are evaluated using a mixed-order alignment loss that aggregates alignment errors across multiple potential orders:

$$
L _ { m i x } \left( a \right) = \sum _ { n \in N } L _ { n } \left( a \right)\tag{4}
$$

$$
L _ { n } \left( a \right) = \sum _ { k = 1 } ^ { n - 1 } C h a m f e r \left( P , R \left( a , \theta _ { k } \right) P \right)\tag{5}
$$

where $\theta _ { k } = 2 \pi k / n$ and $R \left( a , \theta _ { k } \right)$ denotes rotation of the point cloud P around axis a by $\theta _ { k }$ and Chamfer(, ) denotes the Chamfer distance[39].

Because high-order axes inherently satisfy the periodicity of lower-order rotations, their aggregated loss tends to be smaller, forming natural descent basins in the loss landscape (Fig. 3). As a result, optimization is naturally biased toward these dominant high-order directions during search. This self-guiding property enables high-order symmetries to emerge automatically, thereby supporting fully referencefree localization of the dominant axis.

## Two-Stage Sampling and Optimization.

To balance global exploration and computational efficiency, axis localization is carried out in two stages: global coarse sampling followed by local fine refinement, with a final gradient-based optimization for convergence stability.

In the first stage, global coarse sampling uniformly explores the unit sphere to identify regions of low mixed-order alignment loss, yielding the $\mathrm { t o p } { - } k _ { c a n d }$ candidate directions. The second stage performs local fine sampling around each candidate by resampling a spherical cap of angular radius r and re-evaluating $\mathrm { t o p } { - } k _ { i n i t }$ directions under the same loss. This hierarchical search strategy enables efficient localization of promising minima while maintaining broad coverage of the loss landscape. The refined candidates are then optimized using the Adam algorithm to further minimize the aggregated loss and stabilize the alignment result.

Through this progressive sampling-to-optimization pipeline, the framework robustly localizes the dominant high-order rotational axis, which subsequently serves as the geometric anchor for order estimation.

## 3.2.2. Rotational Order Identification

After the dominant axis $a ^ { * }$ is localized, rotational order $N _ { \mathrm { e s t } }$ is inferred by analyzing the object’s periodic self-alignment under densely sampled rotations. The object is rotated around $a ^ { * }$ through a dense set of uniformly spaced angles $\phi ~ \in ~ [ 0 , 2 \pi )$ (typically 360 samples per full revolution), and the Chamfer distance between the original and rotated shapes is computed at each step to form a rotational selfsimilarity signal. The dominant frequency of this signal $\mathrm { \Delta \ r e \mathrm { - } }$ veals the object’s fundamental rotational periodicity.

A coarse estimate $N _ { f f t }$ is obtained from the dominant frequency component and further refined through direct alignment evaluation over candidate orders $\{ N _ { f f t } , \ N _ { f f t } \pm 1 \}$ The order that minimizes the reconstruction error is selected as the final $N _ { \mathrm { e s t } }$ . Ambiguous or degenerate cases are resolved through hierarchical verification: when $N _ { \mathrm { { e s t } } } ~ \geq ~ 3 ,$ the object is considered discretely rotationally symmetric; when $N _ { \mathrm { e s t } } = 2 , \mathrm { ~ a ~ } 1 8 0 ^ { \circ }$ rotation test distinguishes genuine two-fold symmetry from reflection-like structures; and when $N _ { \mathrm { e s t } } ~ \leq ~ 1$ , a small-angle (45<sup>◦</sup>) rotation test differentiates continuous from asymmetric shapes.

This procedure provides a robust, reference-free estimation of both the dominant axis and its rotational order, forming a reliable basis for the hierarchical symmetry classification and full-axis reconstruction introduced in Sec. 3.3.

## 3.3. Hierarchical Type Classification and Full-Axis Reconstruction

Building upon the Geometric Degeneration Hierarchy (GDH) introduced in Section 3.1, each symmetry type is characterized by the number and relative orientation of its rotational axes. Once the dominant high-order axis $a ^ { * }$ is localized and its order $N _ { \mathrm { e s t } }$ is inferred, the GDH restricts the hypothesis space to the subset of types consistent with $N _ { \mathrm { e s t } }$

The remaining ambiguity concerns the orientation of a secondary high-order axis relative to $a ^ { * }$ . Under GDH, each candidate symmetry type prescribes a characteristic interaxis inclination $\alpha _ { F }$ between $a ^ { * }$ and a secondary axis $b ^ { * }$ which may share the same or a lower rotational order. Because these inclinations are invariant to pose and scale, verifying the existence of a secondary axis at the prescribed $\alpha _ { F }$ serves to identify the correct symmetry type and resolve the remaining ambiguity.

## 3.3.1. Secondary-Axis Identification and Hierarchical Type Classification

The secondary axis $b ^ { * }$ is searched along a circular locus on the unit sphere, defined by a fixed inclination $\alpha _ { F }$ from the dominant axis $a ^ { * }$ , as prescribed by the corresponding GDH type. The optimal direction is obtained by minimizing the alignment loss

$$
\boldsymbol { b } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { b } \in S ( \alpha _ { F } ) } L _ { \mathrm { C h a m f e r } } \left( \boldsymbol { b } ; \boldsymbol { N } \right)\tag{6}
$$

where $N$ denotes the hypothesized order of b (equal to $\mathrm { N } _ { \mathrm { e s t } }$ for Platonic families or a lower order for prismatic and pyramidal ones).

Candidate directions with the lowest losses are preserved as top-k seeds. Each seed undergoes a one-degree-offreedom gradient refinement along the ring parameter axis, inherently enforcing the inclination constraint. The refinement, performed using the Adam optimizer, minimizes $L _ { \mathrm { C h a m f e r } } \left( b ; N \right)$ . Once $b ^ { * }$ is obtained, the symmetry type is determined from the order pair $( N _ { a ^ { * } } , N _ { b ^ { * } } )$ and the theoretical inclination $\alpha _ { F }$ defined in the GDH.

Continuous symmetries $( C _ { \infty } , D _ { \infty } )$ are recognized when no discrete periodicity is found along $\mathbf { a } ^ { * }$ . The presence of additional perpendicular two-fold axes distinguishes cylindrical $( D _ { \infty } )$ from circular $( C _ { \infty } )$ symmetry, while complete isotropy corresponds to the spherical case.

Discrete symmetries $( C _ { n } , \ D _ { n } )$ are identified when $N _ { e s t } \geq 2 .$ Orthogonal two-fold axes indicate dihedral $( D _ { n } )$ structures, whereas their absence yields cyclic $( C _ { n } )$ symmetry.

Platonic types $( A _ { 4 } , S _ { 4 } , A _ { 5 } )$ are verified by comparing the measured inter-axis inclination with the theoretical constants $7 0 . 5 3 ^ { \circ }$ (tetrahedral, $A _ { 4 } ) , 9 0 ^ { \circ }$ (octahedral/cubic, $S _ { 4 } )$ and $6 3 . 4 3 ^ { \circ }$ (icosahedral, $A _ { 5 } )$

This geometric classification maps the detected axis configuration to its canonical rotational type, completing symmetry-type determination before the full-axis reconstruction in Sec. 3.3.2.

## 3.3.2. Canonical Axis Reconstruction

Once the symmetry type is determined, the GDH provides its canonical axis configuration. With the dominant axis $a ^ { * }$ and secondary axis $b ^ { * }$ already localized, the remaining axes are recovered by aligning the canonical arrangement associated with the identified type. Each type includes a predefined axis template specifying the relative orientations and multiplicities of its rotational axes. A rigid transformation Q is estimated to align the template pair $( \hat { a } , \hat { b } )$ with the detected pair $( a ^ { * } , b ^ { * } )$ , yielding the full set of axes as

$$
u _ { i } ^ { \mathrm { f i n a l } } = Q \widehat { u _ { i } } .\tag{7}
$$

This alignment transfers type-specific geometric relations to the object, ensuring consistency with its rotational structure. The resulting reconstruction completes full-axis localization in a compact, reference-free form suitable for symmetry reasoning and pose evaluation.

## 3.4. Texture-Symmetry Extension for Appearance-Aware Analysis

Real-world objects rarely exhibit perfect geometric regularity. Surface appearance, including texture and color patterns, can break rotational invariance in the image domain. To model this effect, texture is treated as a secondary order reduction applied on top of geometric symmetry. Following the GDH, this reduction alters only the rotational order of each axis while preserving its orientation. The effective orders must lie in the divisor set of the geometric order:

$$
n _ { \mathrm { t e x } } \in \{ d | d \mathrm { d i v i d e s } n _ { \mathrm { g e o } } \} \cup \{ 1 \} .\tag{8}
$$

Thus, a geometrically six-fold object may appear threeor two-fold under periodic textures, and continuous symmetries such as cylinders can manifest as discrete $D _ { n }$ patterns when repeated surface structure is present. Given the recovered geometric type and axis configuration, texture refinement is computed analytically by evaluating appearance consistency around each axis, yielding a unified representation that incorporates both geometry- and texture-induced symmetry.

## 4. Experiments

This section presents the experimental setup, ablation studies, and cross-dataset evaluations. We first outline the implementation and evaluation settings, then conduct ablations on the DSRSTO [26] dataset. To assess generalization, we further evaluate on GSO[28] and BOP[27], which include diverse real-world objects with varying symmetry characteristics.

## 4.1. Experimental Setup

## 4.1.1. Implementation Details

All experiments are implemented in Python/PyTorch and executed on an Intel i5-13400F / RTX 3050 GPU and 32

GB RAM. In the high-order axis detection stage, 128 coarse directions are uniformly sampled on the upper hemisphere. The top six are refined within $1 0 ^ { \circ }$ spherical caps using eight local samples, and the best three undergo Adam optimization (lr=0.04, 5 steps). The mixed-order alignment loss uses the candidate order set $N = \{ 3 , 4 , 5 , 6 \}$ For secondaryaxis localization, Adam is applied with lr=0.05 for 5 steps. The number of top candidates equals twice the estimated rotational order, and each region is refined by eight local samples.

## 4.1.2. Adaptive Sampling Mechanism

High-Order Axis Detection. A single spherical-cap radius r governs all sampling budgets within the valid range $r \ \in \ [ r _ { m i n } , r _ { m a x } ] .$ , where coverage remains geometrically meaningful. The theoretical lower bound of the coarsesampling count is derived from the ratio between the sphere surface and the cap area,

$$
N _ { m i n } = \frac { 2 } { 1 - \cos r }\tag{9}
$$

representing the minimal number of caps required for full coverage. Sampling is limited to the upper hemisphere to avoid antipodal redundancy, and $N _ { m i n }$ is rounded to the nearest multiple of eight for computational convenience.

The radius r jointly determines the coarse-sampling count $N _ { m i n } \left( r \right)$ , candidate directions $k _ { \mathrm { c a n d } } \left( r \right)$ , local fine samples $n _ { \mathrm { l o c a l } } \left( r \right) ~ \geq ~ 4$ , and Adam initializations $k _ { \mathrm { i n i t } } \left( r \right)$ Smaller r increases $k _ { \mathrm { c a n d } } \left( r \right)$ to capture narrow high-order axes, while larger r raises $k _ { \mathrm { i n i t } } \left( r \right)$ for broader exploration. Both are capped at six, consistent with the fact that in 3D finite rotational symmetries the number of maximal-order $( \geq 3 )$ axes never exceeds six (achieved only by the icosahedral case). This radius-driven formulation balances sampling density and computational efficiency, ensuring stable localization across geometries.

Secondary-Axis Localization. Sampling density scales linearly with the target rotational order, allocating proportionally more candidates and local samples for higher orders to maintain consistent angular resolution.

## 4.1.3. Metrics and Datasets

To quantitatively evaluate the proposed framework, both classification and localization metrics are employed.

Classification Metrics. To assess symmetry-type and order prediction, we report three accuracies:

$$
a c c _ { t } = \frac { N _ { \mathrm { c o r r e c t } { \mathrm { t y p e } } } } { N _ { \mathrm { a l l } } } ,\tag{10a}
$$

$$
a c c _ { o } = \frac { N _ { \mathrm { c o r r e c t ~ o r d e r ~ | ~ c o r r e c t ~ t y p e } } } { N _ { \mathrm { c o r r e c t ~ t y p e } } } ,\tag{10b}
$$

$$
a c c _ { T } = \frac { N _ { \mathrm { c o r r e c t ~ t y p e ~ \& ~ o r d e r } } } { N _ { \mathrm { a l l } } } .\tag{10c}
$$

These metrics respectively reflect type accuracy, orderconsistent accuracy, and overall correctness.

Localization Metric. Axis localization is evaluated using the Average Distance of Model Points (ADI), which measures the discrepancy between the original model and its reconstructed symmetric instances. Given the predicted symmetry orders, directions, and centers, the rotational symmetry is represented by a discrete rotation set $T _ { p } = T _ { i }$ The localization error is

$$
e _ { \mathrm { A D I } } = \frac { \sum _ { T _ { i } \in T _ { p } } { \mathrm { A D I } ( T _ { i } , I ) } } { | T _ { p } | } ,\tag{11}
$$

where $\mathrm { A D I } ( T _ { i } , I )$ denotes the mean pointwise distance between the model and its rotated counterpart.

Misclassified objects yield incorrect canonical axis groups and thus invalid rotation sets, making $e _ { \mathrm { A D I } } / d$ meaningless. Accordingly, $e _ { \mathrm { A D I } } / d$ is reported only for samples correctly identified in both type and order.

Datasets. The DSRSTO[26] dataset includes 38 CAD objects across seven categories, covering all canonical 3D rotational symmetry types with 27 geometrically symmetric and 11 texture-symmetric models. As a manually designed benchmark, it serves to evaluate symmetry classification and axis localization. To assess performance in real scenarios, tests are conducted on common 6D pose benchmarks such as ITODD[40], T-LESS[41], LM-O[42], HB[43], YCB-V[8], and IC-BIN[44], which contain realworld objects but limited symmetry diversity. The GSO[28] dataset extends the evaluation to 944 scanned models, among which 438 exhibit rotational symmetry, providing a large-scale testbed for generalization across geometries.

## 4.2. Ablation Studies

Ablation experiments on the DSRSTO dataset were conducted to analyze the influence of core design choices on classification accuracy, axis-localization precision, and computational efficiency. The evaluation considers overall accuracy acc<sub>T</sub>, normalized geometric alignment error $\mathbf { e _ { A D I } } / d$ (where d is the object diameter), and average runtime per object. Two factors were examined: the adaptive radius-driven sampling configuration and the composition of the candidate order set.

## 4.2.1. Sampling Configuration

The effect of the spherical-cap radius r was evaluated within $[ 5 ^ { \circ } , 3 0 ^ { \circ } ]$ to assess its impact on coverage, accuracy, and efficiency. As shown in Fig. 4, performance forms a single peak near $r = 1 0 ^ { \circ }$ . Both total accuracy and normalized alignment error remain stable between $9 ^ { \circ }$ and $1 1 ^ { \circ } .$ achieving the optimum at $r ~ = ~ 1 0 ^ { \circ } ~ ( a c c _ { T } ~ = ~ 8 1 . 4 8 \%$ $e _ { A D I } / d = 4 . 1 8 { \times } 1 0 ^ { - 3 } )$ . Smaller radii $( \leq 8 ^ { \circ } )$ lead to redundant refinements without accuracy gain, while larger ones $( \geq ~ 1 2 ^ { \circ } )$ reduce coverage and precision. Runtime varies slightly (≈ 1.3 − 1.8s per model). The interval $9 ^ { \circ } - 1 1 ^ { \circ }$ thus provides the best trade-off among coverage, accuracy, and efficiency, and $r = 1 0 ^ { \circ }$ is adopted as the default.

![](images/804de29615d43326182f127b5bb15853829e39d32273ff1407064a851a32e243.jpg)  
Figure 4. Effect of the spherical-cap radius r on sampling performance. Dashed curves indicate fitted trends for total accuracy (blue) and normalized alignment error $e _ { \mathrm { A D I } } / d$ (orange). The optimal configuration appears near $r = 1 0 ^ { \circ }$

Table 1. Ablation experiments on candidate order sets ${ \mathcal { N } } .$
<table><tr><td rowspan=1 colspan=1>Order Set N</td><td rowspan=1 colspan=1>accT (%) |</td><td rowspan=1 colspan=1> $e _ { \mathrm { A D I } } / d$ </td><td rowspan=1 colspan=1>Time (s)</td></tr><tr><td rowspan=4 colspan=1>{3, 4}{3, 4, 5}{3, 5, 6}{3, 4, 5, 6}</td><td rowspan=1 colspan=1>70.37</td><td rowspan=1 colspan=1>0.00419</td><td rowspan=1 colspan=1>1.02</td></tr><tr><td rowspan=1 colspan=1>77.78</td><td rowspan=1 colspan=1>0.00423</td><td rowspan=1 colspan=1>1.20</td></tr><tr><td rowspan=1 colspan=1>77.78</td><td rowspan=1 colspan=1>0.00411</td><td rowspan=1 colspan=1>1.69</td></tr><tr><td rowspan=1 colspan=1>81.48</td><td rowspan=1 colspan=1>0.00418</td><td rowspan=2 colspan=1>1.461.77</td></tr><tr><td rowspan=3 colspan=1>{3, 4, 5, 6, 7}{3, 4, 5, 6, 7, 9}{2, 3, 4, 5, 6}</td><td rowspan=1 colspan=1>81.48</td><td rowspan=1 colspan=1>0.00419</td></tr><tr><td rowspan=1 colspan=1>81.48</td><td rowspan=1 colspan=1>0.00429</td><td rowspan=1 colspan=1>2.40</td></tr><tr><td rowspan=1 colspan=1>77.78</td><td rowspan=1 colspan=1>0.00659</td><td rowspan=1 colspan=1>2.28</td></tr></table>

## 4.2.2. Order-Set Composition

Table 1 summarizes the evaluation of different candidate order sets ${ \mathcal { N } } ,$ , obtained by systematically reducing, expanding, and cross-validating their elements, including configurations with and without the 2-fold term.

Reducing $\mathcal { N }$ to {3, 4} or {3, 4, 5} caused clear accuracy drops, indicating that {3, 4, 5, 6} is the minimal effective configuration. Expanding the set to {3, 4, 5, 6, 7} or {3, 4, 5, 6, 7, 9} increased runtime without measurable benefit, while {3, 5, 6} further confirmed the necessity of the 4- fold term for distinguishing cubic and dihedral symmetries. Including $n = 2$ introduced instability, as trivial half-turn alignments dominated optimization and led to high-order shapes, particularly several 3-fold models, being misclassified as 2-fold. (Detailed analyses are provided in the supplementary material.) Overall, $\mathcal { N } = \{ 3 , 4 , 5 , 6 \}$ provides the best balance among accuracy, stability, and computational efficiency, while adding 2-fold or rare high-order terms introduces misalignment or unnecessary cost.

Table 2. Comprehensive evaluation of symmetry classification and axis localization across datasets. $\mathrm { D S R S T O _ { \mathrm { t e x } } } \mathrm { ; }$ texture subset.
<table><tr><td rowspan="2">Method</td><td colspan="2">Dataset Name</td><td colspan="3">Category Statistics</td><td colspan="6">Dataset-Level Performance</td></tr><tr><td></td><td>Object Type</td><td>Count Errors</td><td></td><td>accT</td><td>acct</td><td>acco</td><td>accT</td><td></td><td>time eAD1/d (ours / Wang et al. [26])</td></tr><tr><td rowspan="5">BOP (original)</td><td rowspan="5">BOP[27]</td><td> $C _ { \infty }$   $D _ { \infty }$ </td><td>7 3</td><td>0 0</td><td>100.00% 100.00%</td><td rowspan="5">80.00%</td><td rowspan="5">100.00%</td><td rowspan="5">80.00%</td><td rowspan="5"></td><td rowspan="5"></td><td rowspan="5"></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td> $C _ { n }$ </td><td>35</td><td>9</td><td>74.29%</td></tr><tr><td> $D _ { n }$ </td><td>5</td><td>1</td><td>80.00%</td></tr><tr><td> $C _ { \infty }$ </td><td></td><td></td><td>100.00%</td></tr><tr><td rowspan="5">KASALv2(ours)</td><td rowspan="5">BOP[27]</td><td> $D _ { \infty }$ </td><td>7 3</td><td>0 0</td><td>100.00%</td><td rowspan="5">92.00%</td><td rowspan="5">91.30%</td><td rowspan="5"></td><td rowspan="5">84.00% 1.61</td><td rowspan="5"></td><td rowspan="5">0.00256/0.00290</td></tr><tr><td> $C _ { n }$ </td><td>35 5</td><td>8</td><td>77.14%</td></tr><tr><td> $D _ { n }$ </td><td></td><td>0</td><td>100.00%</td></tr><tr><td> $\overline { { C _ { \infty } } }$ </td><td>118</td><td>3</td><td>97.46%</td></tr><tr><td> $D _ { \infty }$ </td><td>11</td><td>0</td><td>100.00%</td></tr><tr><td rowspan="5">KASALv2(ours)</td><td rowspan="5">GSO[28]</td><td> $C _ { n }$ </td><td>85</td><td>14</td><td>83.53%</td><td rowspan="5">96.58%</td><td rowspan="5">98.10%</td><td rowspan="5"></td><td rowspan="5">94.75% 1.37</td><td rowspan="5"></td><td rowspan="5">0.00262/0.00256</td></tr><tr><td> $D _ { n }$ </td><td>222 2</td><td>6 0</td><td>97.30% 100.00%</td></tr><tr><td> $S _ { 4 }$ </td><td></td><td></td><td></td></tr><tr><td> $C _ { \infty }$ </td><td></td><td>0 0</td><td>100.00%</td></tr><tr><td> $D _ { \infty }$ </td><td></td><td>5</td><td>100.00%</td></tr><tr><td rowspan="6">KASALv2(ours) KASALv2(ours)</td><td rowspan="6">DSRSTO[26]</td><td> $C _ { n }$   $D _ { n }$ </td><td>1 7 12</td><td></td><td>28.57% 100.00%</td><td rowspan="6">81.48%</td><td rowspan="6"></td><td rowspan="6">100.00% 81.48% 1.46</td><td rowspan="6"></td><td rowspan="6"></td><td rowspan="6">0.00212/0.00172 0.00195/0.00181</td></tr><tr><td> $A _ { 4 }$ </td><td>1 3</td><td>0 0</td><td>100.00%</td></tr><tr><td> $S _ { 4 }$ </td><td></td><td></td><td>100.00%</td></tr><tr><td> $A _ { 5 }$ </td><td>2</td><td>0</td><td>100.00%</td></tr><tr><td> $D _ { n }$ </td><td>11</td><td>3</td><td>72.73%</td></tr><tr><td> $\mathrm { D S R S T O _ { t e x } } [ 2 6 ]$ </td><td></td><td></td><td></td></tr></table>

## 4.3. Cross-Dataset Evaluation

Cross-dataset evaluations were conducted on DSRSTO, BOP, and GSO. Table 2 reports the metrics defined in Sec. 4.1.3, the average runtime per object, and localization accuracy compared with the reference method [26].

On DSRSTO, the method achieved 81.48% accuracy with $e _ { \mathrm { A D I } } / d = 0 . 0 0 2 1 2$ On BOP, under our stricter reannotation of 50 symmetric instances, it reached 84.00% accuracy, improving over 80.00% under the original labels and reducing $e _ { \mathrm { A D I } } / d$ from 0.00290 to 0.00256. On GSO, it obtained 94.75% accuracy with $e _ { \mathrm { A D I } } / d = 0 . 0 0 2 6 2$ . The 72.32% accuracy on $\mathrm { D S R S T O _ { \mathrm { t e x } } }$ further confirms applicability to texture-level symmetries. The lower DSRSTO performance mainly stems from challenging high-order and visually ambiguous $C _ { n }$ cases; consistently, $C _ { n }$ is also the lowest-accuracy category in BOP. Details of the original BOP annotations, our re-annotation, and the full error analysis are provided in the supplementary material.

## 4.4. Effect of Symmetry Priors on Pose Estimation

To assess the downstream value of automatically estimated symmetry priors, we train two identical Foundation-Pose [29] models on a synthetic GSO training set of about one million rendered images, differing only in the use of KASALv2-derived rotational-symmetry priors. The prior is injected only at the label level via ZebraPose’s symmetryaware label canonicalization module, which maps each GT pose to a canonical representative in its symmetry equivalence class for both discrete and continuous symmetries. All other settings remain unchanged.

Table 3 reports a retrained multi-checkpoint evaluation on five BOP datasets. Averaged over five checkpoints from the same training run, the results improve on all five datasets, with the largest mean gain of 0.9 points on IC-BIN.

Table 3. FoundationPose with and without KASALv2 symmetry priors on five BOP datasets. Results are reported as mean (range) over five checkpoints from the same training run (seed 0).
<table><tr><td>Dataset</td><td>w/ KASALv2 mean (range)</td><td>w/o KASALv2 mean (range)</td><td>Diff.</td></tr><tr><td>LM-0</td><td>71.9 (71.7–72.0)</td><td>71.6 (71.5–71.8)</td><td>+0.3</td></tr><tr><td>IC-BIN</td><td>64.7 (64.4–65.3)</td><td>63.8 (63.2–64.4)</td><td>+0.9</td></tr><tr><td>YCB-V</td><td>87.7 (87.6–87.9)</td><td>87.6 (87.5–87.8)</td><td>+0.1</td></tr><tr><td>ITODD</td><td>65.8 (65.1–66.4)</td><td>65.0 (64.9–65.2)</td><td>+0.8</td></tr><tr><td>T-LESS</td><td>59.3 (58.6–60.1)</td><td>58.8 (58.3–59.3)</td><td>+0.5</td></tr></table>

## 5. Conclusion

This work presents a unified and fully automatic framework for 3D rotational symmetry analysis, jointly addressing symmetry-type classification, rotational-order identification, and axis localization across all eight canonical types. The Geometric Degeneration Hierarchy (GDH), together with an integrated inference pipeline, enables complete and reference-free symmetry reasoning for real-world 3D objects. Extensive experiments show strong accuracy and generalization, reaching 94.75% on GSO, and the symmetry priors produced by KASALv2 improve FoundationPose accuracy by up to 0.9% across five BOP datasets. Despite these results, characteristic failure modes remain, including the underestimation of high rotational orders and sensitivity to local geometric irregularities. Future work will explore more frequency-sensitive order estimation and improved robustness to locally perturbed structures.

## 6. Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant No. 52375487 and the Special Fund of Jiangsu Province for Key Research and Development under Grant No. BE2023041.

The authors also thank the reviewers for their thoughtful and constructive comments, which have helped improve this paper. Due to space limitations, several complementary analyses and extended discussions could not be included in the present conference version and will be reported in a future journal extension, including posterior verification for continuous symmetries and secondary two-fold axes, robustness under geometric perturbations, additional optimizer and runtime analyses, large-scale rotationalsymmetry annotation on Objaverse, and broader downstream evaluation studies.

## References

[1] Tijn Bertens, Brandon Caasenbrood, Alessandro Saccon, and Andrei Jalba. Symmetry-induced ambiguity in orientation estimation from RGB images. Mach. Vis. Appl., 36:40, 2025. 1, 2

[2] Wadim Kehl, Fabian Manhardt, Federico Tombari, Slobodan Ilic, and Nassir Navab. SSD-6D: Making RGB-based 3D detection and 6D pose estimation great again. In ICCV, pages 1530–1538, 2017. 2

[3] Mahdi Rad and Vincent Lepetit. BB8: A scalable, accurate, robust to partial occlusion method for predicting the 3D poses of challenging objects without using depth. In ICCV, pages 3828–3836, 2017. 2

[4] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3D reconstruction in function space. In CVPR, pages 4460–4470, 2019. 2

[5] Fabian Manhardt, Diego Martin Arroyo, Christian Rupprecht, Benjamin Busam, Tolga Birdal, Nassir Navab, and Federico Tombari. Explaining the ambiguity of object detection and 6D pose from visual data. In ICCV, pages 6840– 6849, 2019.

[6] Tsu-Ching Hsiao, Hao-Wei Chen, Hsuan-Kung Yang, and Chun-Yi Lee. Confronting ambiguity in 6D object pose estimation via score-based diffusion on SE(3). In CVPR, pages 352–362, 2024.

[7] Giorgia Pitteri, Michael Ramamonjisoa, Slobodan Ilic, and¨ Vincent Lepetit. On object symmetries and 6D pose estimation from images. In 3DV, pages 614–622, 2019. 1, 2

[8] Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, and Dieter Fox. PoseCNN: A convolutional neural network for 6D object pose estimation in cluttered scenes. In Proceedings ofRobotics: Science and Systems, 2018. 1, 3, 7

[9] Toma´s Hodan, Martin Sundermeyer, Bertram Drost, Yannˇ Labbe, Eric Brachmann, Frank Michel, Carsten Rother, and´ Jiˇr´ı Matas. BOP challenge 2020 on 6D object localization. In Computer Vision – ECCV 2020 Workshops, pages 577–594, 2020. 1, 3

[10] Yulin Wang, Mengting Hu, Hongli Li, and Chen Luo. HccePose(BF): Predicting front & back surfaces to construct ultra-dense 2D-3D correspondences for pose estimation. In ICCV, pages 7166–7175, 2025. 1, 2, 3

[11] Yongzhi Su, Mahdi Saleh, Torben Fetzer, Jason Rambach, Nassir Navab, Benjamin Busam, Didier Stricker, and Federico Tombari. ZebraPose: Coarse to fine surface encoding for 6DoF object pose estimation. In CVPR, pages 6728– 6738, 2022. 1, 2, 3

[12] James E. Humphreys. Reflection Groups and Coxeter Groups. Cambridge University Press, 1990. 1, 3

[13] Harold Scott Macdonald Coxeter. Regular Polytopes. Dover Publications, 1973.

[14] Jr. Miller, Willard. Symmetry Groups and Their Applica tions. Academic Press, 1972.

[15] Deane Montgomery and C. T. Yang. A theorem on the action of SO(3). Pacific J. Math., 12(4):1385–1400, 1962.

[16] Hermann Weyl. Symmetry. Princeton University Press, 1952. 1, 3

[17] Rajkumar Gothandaraman, Rohitkumar Jha, and Sreekumar Muthuswamy. Reflectional and rotational symmetry detection of cad models based on point cloud processing. In IEEE 4th Conf. Inf. Commun. Technol., pages 1–5, 2020. 1, 2

[18] Luka´s Hruda, Ivana Kolingerovˇ a, Miroslav L´ avi´ cka, andˇ Martin Manˇak. Rotational symmetry detection in 3D us-´ ing reflectional symmetry candidates and quaternion-based rotation parameterization. Comput. Aided Geom. Des., 98:102138, 2022. 2

[19] Aleksandrs Ecins, Cornelia Fermuller, and Yiannis Aloi-¨ monos. Seeing behind the scene: Using symmetry to reason about objects in cluttered environments. In IEEE/RSJ Int. Conf. Intell. Robots Syst., pages 7193–7200, 2018. 2

[20] Ivan Sipiran. Completion of cultural heritage objects with rotational symmetry. In Eurographics Workshop on 3D Object Retrieval, pages 87–93, 2018. 2

[21] Ivan Sipiran. Analysis of partial axial symmetry on 3D sur faces and its application in the restoration of cultural heritage objects. In IEEE Int. Conf. Comput. Vis. Worksh., pages 2925–2933, 2018. 2

[22] Saeid Mansouri and Hossein Ebrahimnezhad. Efficient axial symmetry aware mesh approximation with application to 3D pottery models. Multimed. Tools Appl., 75(14):8347–8379, 2016. 2

[23] Yifei Shi, Junwen Huang, Hongjia Zhang, Xin Xu, Szymon Rusinkiewicz, and Kai Xu. SymmetryNet: Learning to predict reflectional and rotational symmetries of 3D shapes from single-view RGB-D images. ACM TOG, 39(6):213:1– 213:14, 2020. 1, 2

[24] Aurelien Martinet, Cyril Soler, Nicolas Holzschuch, and´ Franc¸ois X. Sillion. Accurate detection of symmetries in 3D shapes. ACM TOG, 25(2):439–464, 2006. 1, 2

[25] Simon Korman, Roee Litman, Shai Avidan, and Alex Bron stein. Probably approximately symmetric: Fast rigid symme try detection with global guarantees. Comput. Graph. Forum, 34(1):2–13, 2015. 1, 2

[26] Yulin Wang and Chen Luo. Key-axis-based localization of symmetry axes in 3D objects utilizing geometry and texture. IEEE TIP, 33:6720–6733, 2024. 1, 2, 6, 7, 8

[27] Toma´s Hodaˇ n, Frank Michel, Eric Brachmann, Wadim Kehl,ˇ Anders Glent Buch, Dirk Kraft, Bertram Drost, Joel Vidal, Stephan Ihrke, Xenophon Zabulis, Caner Sahin, Fabian Manhardt, Federico Tombari, Tae-Kyun Kim, Jiˇr´ı Matas, and Carsten Rother. BOP: Benchmark for 6D object pose estimation. In ECCV, pages 19–35, 2018. 2, 6, 8

[28] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B. McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3D scanned household items. In IEEE Int. Conf. Robot. Autom., pages 2553–2560, 2022. 2, 6, 7, 8

[29] Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. FoundationPose: Unified 6D pose estimation and tracking of novel objects. In CVPR, pages 17868–17879, 2024. 2, 8

[30] Seyed M Hosseini, Seyed-Mahdi Nasiri, Reshad Hosseini, and Hadi Moradi. Single-view 3D reconstruction of surface of revolution. Vis. Comput., 41:8961–8973, 2025. 2

[31] Gu Wang, Fabian Manhardt, Federico Tombari, and Xiangyang Ji. GDR-Net: Geometry-guided direct regression network for monocular 6D object pose estimation. In CVPR, pages 16606–16616, 2021. 2

[32] Toma´s Hodaˇ n, Dˇ aniel Bar´ ath, and Ji´ ˇr´ı Matas. EPOS: Estimating 6D pose of objects with symmetries. In CVPR, pages 11700–11709, 2020. 2

[33] Ningkai Mo, Wanshui Gan, Naoto Yokoya, and Shifeng Chen. ES6D: A computation efficient and symmetry-aware 6D pose regression framework. In CVPR, pages 6718–6727, 2022. 2

[34] Jesse Richter-Klug and Udo Frese. Handling object symmetries in CNN-based pose estimation. In IEEE Int. Conf. Robot. Autom., pages 13850–13856, 2021. 2

[35] Felix Klein. Lectures on the Icosahedron and the Solution of Equations ofthe Fifth Degree. Dover Publications, 1956. 3

[36] Ludwig Schlafli. ¨ Theorie der vielfachen Kontinuitat¨ . Zurcher¨ & Furrer, 1901. 3

[37] Hans Frederik Blichfeldt. Finite Collineation Groups: With an Introduction to the Theory of Groups of Operators and Substitution Groups. The University of Chicago Press, 1917. 3

[38] John Horton Conway and Neil James Alexander Sloane. Sphere Packings, Lattices and Groups. Springer, 1999. 3

[39] Harry G. Barrow, Jay M. Tenenbaum, Robert C. Bolles, and Helen C. Wolf. Parametric correspondence and chamfer matching: Two new techniques for image matching. In IJ-CAI, pages 659–663, 1977. 4

[40] Bertram Drost, Markus Ulrich, Paul Bergmann, Philipp Hartinger, and Carsten Steger. Introducing MVTec¨ ITODD—a dataset for 3D object recognition in industry. In IEEE Int. Conf. Comput. Vis. Worksh., pages 2200–2208, 2017. 7

[41] Toma´s Hoda ˇ n, Pavel Haluza, ˇ St<sup>ˇ</sup> epˇ an Obdr ´ zˇalek, Ji ´ ˇr´ı Matas, Manolis Lourakis, and Xenophon Zabulis. T-LESS: An RGB-D dataset for 6D pose estimation of texture-less objects. In IEEE Winter Conf. Appl. Comput. Vis., pages 880– 888, 2017. 7

[42] Eric Brachmann, Alexander Krull, Frank Michel, Stefan Gumhold, Jamie Shotton, and Carsten Rother. Learning

6D object pose estimation using 3D object coordinates. In ECCV, pages 536–551, 2014. 7

[43] Roman Kaskman, Sergey Zakharov, Ivan Shugurov, and Slo bodan Ilic. HomebrewedDB: RGB-D dataset for 6D pose estimation of 3D objects. In IEEE/CVF Int. Conf. Comput. Vis. Worksh., pages 2767–2776, 2019. 7

[44] Andreas Doumanoglou, Rigas Kouskouridas, Sotiris Malas siotis, and Tae-Kyun Kim. Recovering 6D object pose and predicting next-best-view in the crowd. In CVPR, pages 3583–3592, 2016. 7