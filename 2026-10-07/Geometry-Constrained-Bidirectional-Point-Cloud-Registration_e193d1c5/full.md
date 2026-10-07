# Geometry-Constrained Bidirectional Point Cloud Registration for Thin, Sheet-Like Heritage Artifacts

YUEZHE ZHANG, Northwestern Polytechnical University, China

LEI WEI , Xi’an International University, China

JINGNAN DU, Northwestern Polytechnical University, China

SHUAI WAN, Northwestern Polytechnical University, China

Non-contact three-dimensional reconstruction of thin, sheet-like heritage artifacts poses significant geometric and registration challenges. Due to their fragility, these artifacts cannot be suspended or equipped with artificial markers, necessitating independent acquisition of their front and back surfaces. Subsequent registration proves dificult due to the limited number of shared geometric features and the scarcity of explicit physical constraints, which may result in rotational ambiguity, instability, and structural collapse during iterative optimization. To address these challenges, we propose a geometry-constrained bidirectional point cloud registration method specifically tailored for thin, sheet-like heritage artifacts. The method integrates semantic-guided preprocessing, Principal Component Analysis (PCA)-based geometric normalization, and a thickness-aware registration strategy. The estimated physical thickness is incorporated as a geometric constraint to preserve structural integrity during registration. Rotational ambiguity is resolved by evaluating a finite set of global rotation hypotheses, each refined using the point-to-plane Iterative Closest Point (ICP) algorithm, with the optimal transformation selected via a geometry-aware fitness criterion consistent with the thickness scale. Experimental results show that the proposed method achieves competitive or improved performance in most cases, particularly in projected area consistency and physically plausible front–back alignment. In addition, the thickness-aware constraint and rotation hypothesis evaluation reduce the risk of degenerate configurations in which the two surfaces are incorrectly flipped while still yielding deceptively acceptable numerical scores, supporting reliable non-contact digitization of delicate and thin heritage artifacts. Implementation details are available at https://zyz-nwpu.github.io/GCBPCR/.

CCS Concepts: • Computing methodologies → Point-based models; Shape analysis; • Applied computing → Digital libraries and archives.

Additional Key Words and Phrases: Geometry-Constrained Registration, Bidirectional Point Cloud Registration, Thickness-Constrained Optimization, PCA-Based Normalization, Thin Heritage Artifacts

© 2026 Copyright held by the owner/author(s). Journal on Computing and Cultural Heritage (2026). https://doi.org/10.1145/3836772

## 1 Introduction

The digital preservation of cultural heritage artifacts has become an important component of modern conservation practice, enabling systematic documentation, structural analysis, virtual restoration, and long-term digital archiving [20]. Accurate three-dimensional (3D) models allow fragile historical objects to be studied and preserved while minimizing physical handling. Early large-scale digitization eforts relied on active range sensing technologies such as laser scanning [12]. Structured-light scanning systems were later introduced to provide dense and accurate surface measurements for smaller artifacts and museum collections [24]. Due to their ability to capture precise geometric details, these systems have been widely adopted in cultural heritage digitization workflows [8]. However, active scanning methods typically require specialized hardware, controlled acquisition environments, and complex calibration procedures.

With the rapid development ofcomputer vision, image-based reconstruction has become a flexible solution for heritage digitization. Structure-from-Motion (SfM) estimates camera poses and sparse geometry from unordered images [28], while later work improved the robustness and scalability of SfM pipelines for large datasets [25]. Eficient incremental systems further enabled large-scale reconstruction from thousands of images [31]. Multi-View Stereo (MVS) methods reconstruct dense surface geometry from calibrated images [7]. Subsequent improvements introduced more robust view selection and pixelwise optimization strategies for dense reconstruction [26]. These SfM–MVS pipelines have become a standard workflow for image-based 3D reconstruction and are widely used for digitizing archaeological sites and museum artifacts [11]. Close-range photogrammetry is also widely used in heritage documentation due to its portability and relatively low cost [13]. Although these techniques are mature, fragile artifacts often require non-contact acquisition from multiple orientations, making point cloud registration necessary to recover complete geometry.

Among these objects, thin sheet-like artifacts such as bronze fragments, ceramic shards, and thin wooden panels present particular challenges for 3D reconstruction. Their extremely small thickness relative to lateral dimensions results in near-planar geometry with limited structural depth. Due to conservation constraints, these fragile artifacts cannot be suspended or equipped with artificial markers during acquisition and must instead be placed on a supporting surface. Consequently, the front and back sides are captured separately under non contact conditions consistent with non-invasive heritage digitization principles [8]. Because the two sides are reconstructed independently, their point clouds lie in diferent local coordinate systems and require accurate rigid registration to recover the complete geometry.

Estimating this transformation remains challenging for thin-sheet artifacts due to their near-planar geometry and limited thickness variation, which reduce geometric saliency and weaken feature-based registration methods. Classical optimization approaches such as Iterative Closest Point (ICP) estimate rigid transformations by minimizing distances between corresponding points [4], but they do not incorporate thickness-aware constraints and may lead to degenerate solutions. Marker-based registration is unsuitable because ataching artificial mark ers violates the non-invasive principles of cultural heritage digitization [20]. Learning-based registration meth ods have also been proposed, but they typically require large annotated datasets and do not explicitly enforce geometry-aware thickness constraints [18]. Therefore, robust registration frameworks for thin-sheet heritage artifacts acquired under non-contact conditions remain limited.

To address these limitations, this work proposes a geometry-constrained bidirectional point cloud registration method specifically designed for thin, sheet-like heritage artifacts. The proposed method integrates semantic-guided preprocessing, Principal Component Analysis (PCA)-based geometric normalization, and thickness-aware registration.

The main contributions of this work are summarized as follows:

Thickness-aware geometric constraint modeling: We propose a robust percentile-based thickness estimation method within a PCA-normalized coordinate system and explicitly incorporate the estimated physical thickness into the registration process as a geometric prior, efectively preventing non-physical structural col lapse during iterative optimization.

Global rotation hypothesis evaluation for ambiguity resolution: We introduce a finite rotation hypothesis strategy derived from principal-axis sign permutations to systematically resolve orientation ambiguity in symmetric thin structures. Each hypothesis is refined via point-to-plane ICP, and the optimal transformation is selected using a geometry-consistent fitness criterion.

Semantic-guided structural purification: We integrate image-level semantic masking with threedimensional spatial clustering to obtain structurally clean point clouds prior to registration, enabling non contact background removal without the need for physical markers.

A practical non-invasive method for thin-sheet heritage digitization: The proposed geometryconstrained bidirectional registration pipeline provides a stable and physically consistent approach for frontback reconstruction of fragile thin-sheet artifacts under non-contact acquisition conditions.

This paper is organized as follows. Section 2 reviews related work on point cloud registration and challenges associated with thin structures. Section 3 presents the proposed geometry-constrained registration method, including semantic-guided purification, PCA-based canonical registration, thickness estimation, rotation hypothesis search, and ICP refinement under physical constraints. Section 4 reports experimental results and ablation studies on representative thin-sheet heritage objects. Finally, Section 5 concludes the paper and discusses limi tations and future work

## 2 Related Work

## 2.1 Point Cloud Registration

Point cloud registration aims to estimate the rigid transformation that aligns multiple three-dimensional point clouds into a common coordinate system. It is a fundamental problem in computer vision, robotics, and 3D per ception, playing a critical role in applications such as 3D reconstruction, SLAM, autonomous driving, and robotic manipulation. Over the past several decades, extensive research has been conducted on point cloud registration, leading to a wide variety of algorithms that can generally be categorized into local iterative methods, global registration approaches, handcrafted feature-based techniques, and more recently learning-based methods.

One of the earliest and most widely used registration algorithms is the Iterative Closest Point (ICP) method introduced by Besl and McKay [4]. ICP iteratively establishes correspondences between two point clouds and estimates the rigid transformation that minimizes the Euclidean distance between matched points. Due to its conceptual simplicity and practical efectiveness, ICP has become a fundamental component in many 3D process ing pipelines. Numerous variants have been proposed to improve its convergence speed and robustness. Chen and Medioni [5] proposed the point-to-plane ICP formulation, which minimizes the distance between points and their corresponding tangent planes, leading to significantly faster convergence. Segal et al. [27] further proposed the Generalized ICP framework, which unifies point-to-point and point-to-plane formulations under a probabilistic model and improves robustness in noisy environments. Rusinkiewicz and Levoy [21] also presented an eficient variant of ICP that improves performance through optimized correspondence selection and error metrics.

Despite their efectiveness, ICP-based methods generally require a good initial alignment and suficient overlap between point clouds. To overcome this limitation, global registration methods have been proposed to estimate transformations without prior alignment. Aiger et al. [1] introduced the Four-Point Congruent Sets (4PCS) algorithm, which identifies congruent geometric configurations between two point clouds by matching sets of four coplanar points. This approach enables robust registration even under large transformations and limited overlap. Mellado et al. [14] later proposed Super4PCS, which significantly improves the eficiency of the original algorithm through smart indexing and geometric filtering, enabling near-linear computational complexity.

Another important class of methods relies on local geometric descriptors to establish correspondences between point clouds. These feature-based approaches compute invariant descriptors that capture the geometric properties of local neighborhoods. Rusu et al. [22] proposed the Fast Point Feature Histogram (FPFH), which eficiently encodes angular relationships between neighboring points and has been widely used for coarse alignment in 3D registration tasks. Tombari et al. [29] introduced the Unique Shape Context (USC) descriptor, which improves robustness for object recognition and registration tasks. Salti et al. [23] proposed the SHOT descriptor, which combines histograms of surface orientations with spatial partitioning to produce highly discriminative local descriptors

In recent years, deep learning has emerged as a powerful paradigm for point cloud registration. Instead of relying on handcrafted descriptors, learning-based methods aim to learn feature representations directly from raw point cloud data. Aoki et al. [2] introduced PointNetLK, which integrates the PointNet architecture with the classical Lucas–Kanade optimization framework for rigid point cloud alignment. Choy et al. [6] proposed Fully Convolutional Geometric Features (FCGF), which learns dense local features for robust correspondence matching across point clouds. Bai et al. [3] further proposed D3Feat, which jointly learns keypoint detection and descriptor extraction for 3D point clouds.

More recent work has focused on improving correspondence estimation through atention mechanisms and transformer-based architectures. Huang et al. [10] proposed Predator, which introduces overlap-aware atention mechanisms to improve feature matching under low-overlap scenarios. Qin et al. [18] introduced Geo-Transformer, which leverages geometric atention to capture long-range spatial relationships and significantly improves robustness under large viewpoint changes.

Although these methods have achieved remarkable progress, most existing registration algorithms still rely on assumptions such as suficient overlap and distinctive geometric features between point clouds. In practical scenarios involving thin or near-planar structures, these assumptions are often violated. In such cases, the lack of distinctive geometric features can lead to ambiguous correspondences and unstable optimization, motivating the development of more specialized registration frameworks.

## 2.2 Registration of Thin and Low-Overlap Structures

Although substantial progress has been made in point cloud registration, most existing methods implicitly assume that the target objects exhibit suficiently rich three-dimensional geometry and overlapping surface regions. These assumptions are often violated when dealing with thin, sheet-like artifacts or near-planar structures, where geometric degeneracy significantly complicates the registration process.

A fundamental challenge arises from the limited geometric variation present in thin or planar surfaces. When the majority of points lie close to a two-dimensional manifold, the covariance structure of the point cloud becomes highly anisotropic, and the smallest principal component often corresponds to the thickness direction of the surface. Under such conditions, classical registration algorithms such as ICP may sufer from rank-deficient optimization problems, leading to unstable convergence or ambiguous transformations [21]. Similar degeneracy issues have been discussed in geometric modeling and surface alignment literature, where planar or symmetric structures reduce the observability of rigid transformations [17].

Another dificulty is the scarcity of distinctive local geometric features. Feature-based registration methods rely on discriminative descriptors computed from local neighborhoods, such as FPFH [22] or SHOT [23]. However, when applied to smooth or low-curvature surfaces, these descriptors often become less distinctive due to the lack ofgeometric variation, resulting in unreliable correspondences. This limitation has been widely reported in empirical evaluations of registration algorithms on structured environments containing large planar regions [16].

More recent learning-based approaches have atempted to address challenging scenarios such as partial overlap and large viewpoint changes. For example, Predator [10] introduces an overlap-aware atention mechanism to improve correspondence estimation in low-overlap situations, while GeoTransformer [18] leverages geometric atention to capture long-range spatial relationships between points. Despite their strong performance in many benchmark datasets, these methods still assume the existence of at least partial overlapping regions be tween point clouds. In scenarios involving double-sided acquisition of thin objects, the front and back surfaces may share litle or no overlapping geometry, making correspondence-based matching fundamentally ill-posed.

Furthermore, thin sheet-like objects introduce additional physical constraints that are rarely considered in conventional registration frameworks. In practical digitization workflows, the two sides of an artifact are often captured independently due to conservation constraints, and the resulting point clouds correspond to opposite surfaces separated by a small estimated physical thickness. Without incorporating such geometric priors, standard registration algorithms may produce physically implausible solutions, including surface interpenetration or collapse along the thickness direction.

These limitations highlight the need for specialized registration strategies that explicitly consider the geometric characteristics of thin structures and the acquisition conditions of double-sided scanning. In particular, incorporating structural priors such as thickness constraints or canonical geometric normalization can help stabilize the alignment process and reduce ambiguity in under-constrained configurations.

## 3 Proposed Method

The proposed method consists of three main stages, as illustrated in Fig. 1. First, a semantic-guided 2D–3D purification process is performed to remove background structures and reconstruction artifacts and obtain artifactonly point clouds (Section 3.1). As further detailed in Fig. 2, this stage expands the artifact-aware multi-view reconstruction and purification part of Fig. 1, including double-sided image acquisition, SfM–MVS reconstruction, semantic masking, and mask-guided point cloud purification. Next, PCA-based canonical normalization is performed to establish intrinsic geometric axes and estimate the artifact thickness (Section 3.2). Finally, a geometry-constrained bidirectional registration strategy aligns the front and back surfaces by combining rotation hypothesis evaluation with point-to-plane ICP refinement under a thickness-aware constraint (Section 3.3).

## 3.1 Semantic-Guided 2D–3D Purification

Following the artifact-aware pipeline in Fig. 1, we treat the front and back surfaces of thin, sheet-like heritage artifacts as two independent acquisition-and-reconstruction sessions, and we purify each session in its own coordinate system before any cross-side registration is atempted. This separation is essential because the supporting plane is present in both sessions but appears diferently under viewpoint and illumination changes, and the reconstructed shells may contain side-specific background adhesion and outliers. Accordingly, we define two disjoint multi-view datasets and process them with the same purification operator, instantiated once for the front side and once for the back side.

For the front side, let the captured front-side image sequence be

$$
\mathcal { I } _ { f } = \{ I _ { f } ^ { k } \} _ { k = 1 } ^ { N _ { f } } ,\tag{1}
$$

where each image is defined as

$$
I _ { f } ^ { k } : \Omega _ { f } ^ { k } \subset \mathbb { R } ^ { 2 } \to \mathbb { R } ^ { 3 } ,\tag{2}
$$

with $\Omega _ { f } ^ { k }$ denoting the pixel domain and $\mathbb { R } ^ { 3 }$ representing RGB color space.

To suppress background interference introduced by the supporting plane, we employ the Segment Anything Model 2 (SAM2) [19] to obtain artifact-aware semantic masks. Specifically, foreground and background prompts are manually provided on the first image of the sequence, and SAM2 propagates the segmentation to the remaining frames, producing a binary mask for each image.

Let the resulting mask set be

$$
\mathcal { M } _ { f } = \{ { M } _ { f } ^ { k } \} _ { k = 1 } ^ { N _ { f } } , \qquad M _ { f } ^ { k } : \Omega _ { f } ^ { k } \to \{ 0 , 1 \} ,\tag{3}
$$

where 1 denotes artifact pixels and 0 denotes background.

The front-side image sequence is processed using a standard SfM–MVS pipeline, producing a dense point cloud representing the front surface of the artifact

6 • Yuezhe Zhang, Lei Wei, Jingnan Du, and Shuai Wan

![](images/01ba744710b2b8a4cf5be180f347b9f029ca9ebde3e5a7107df94ece4b3d2458.jpg)  
Fig. 1. Overview of the proposed geometry-constrained bidirectional point cloud registration pipeline for thin sheet-like artifacts. The workflow begins with independent front- and back-side multi-view reconstruction using an SfM–MVS pipeline. Semantic-guided purification is then applied to remove background structures and reconstruction artifacts, producing artifact-only point clouds for both sides. Next, PCA-based canonical normalization aligns each point cloud to its intrinsic geometric axes and enables robust estimation of the artifact thickness. Based on this canonical representation, a geometryconstrained registration stage evaluates a finite set of rotation hypotheses and refines each candidate using point-to-plane ICP initialized with a thickness-aware ofset. The optimal transformation is selected using a thickness-aware fitness crite rion, and the aligned point clouds are finally merged to obtain the complete artifact model.

$$
\mathcal { P } _ { f } = \{ \mathbf { X } _ { i } ^ { ( f ) } \in \mathbb { R } ^ { 3 } \} _ { i = 1 } ^ { M _ { f } } ,\tag{4}
$$

together with the corresponding camera parameters

$$
\{ \mathbf { K } _ { f } ^ { k } , \mathbf { R } _ { f } ^ { k } , \mathbf { t } _ { f } ^ { k } \} _ { k = 1 } ^ { N _ { f } } ,\tag{5}
$$

where $N _ { f }$ denotes the number of front-side images. For each view �, $\mathbf { K } _ { f } ^ { k } \in \mathbb { R } ^ { 3 \times 3 }$ represents the camera intrinsic matrix, while $\mathbf { R } _ { f } ^ { k } \in S O ( 3 )$ and $\mathbf { t } _ { f } ^ { k } \in \mathbb { R } ^ { 3 }$ describe the rigid transformation from the world coordinate system to the camera coordinate system.

Because the supporting plane and surrounding environment may also be reconstructed during the MVS process, a semantic-guided filtering step is performed to retain only points belonging to the artifact. The basic idea is to reproject each reconstructed 3D point into all images and evaluate whether the projected pixel falls inside the artifact mask.

For a 3D point $\mathbf { X } \in \mathcal { P } _ { f }$ , its coordinates in the camera coordinate system of view � are first obtained through the rigid transformation

$$
{ \bf X } _ { f , k } ^ { c } = { \bf R } _ { f } ^ { k } { \bf X } + { \bf t } _ { f } ^ { k } ,\tag{6}
$$

where $\mathbf { X } _ { f , k } ^ { c } = [ x _ { f , k } ^ { c } , y _ { f , k } ^ { c } , z _ { f , k } ^ { c } ] ^ { \top }$ . Using the standard perspective projection model, the normalized image coordinates are computed as

$$
\mathbf { x } _ { f , k } ^ { n } = \left[ \begin{array} { l } { \boldsymbol { x } _ { f , k } ^ { c } / \boldsymbol { z } _ { f , k } ^ { c } } \\ { \boldsymbol { y } _ { f , k } ^ { c } / \boldsymbol { z } _ { f , k } ^ { c } } \end{array} \right] .\tag{7}
$$

Geometry-Constrained Bidirectional Point Cloud Registration for Thin, Sheet-Like Heritage Artifacts

![](images/a48f7b96d13d3d8836c6d99feb884aa4010d4d23024a478b227b6c001df26ca5.jpg)  
Fig. 2. Artifact-aware multi-view reconstruction and purification pipeline. The front and back surfaces are digitized through double-sided independent multi-view acquisition. The captured image sets are processed using an SfM–MVS reconstruction framework to generate dense point clouds. Semantic masks are predicted for each image and integrated with 3D structural filtering to refine the reconstructed data, yielding artifact-only point clouds for subsequent registration.

With the camera intrinsic matrix

$$
\begin{array} { r } { \mathbf { K } _ { f } ^ { k } = \left[ \begin{array} { c c c } { f _ { x } } & { 0 } & { c _ { x } } \\ { 0 } & { f _ { y } } & { c _ { y } } \\ { 0 } & { 0 } & { 1 } \end{array} \right] , } \end{array}\tag{8}
$$

the projected pixel coordinate is obtained by

$$
\mathbf { u } _ { f , k } = \left[ \overset { u _ { f , k } } { \boldsymbol { \nu } _ { f , k } } \right] = \left[ \overset { f _ { x } x _ { f , k } ^ { c } / z _ { f , k } ^ { c } + c _ { x } } { f _ { y } y _ { f , k } ^ { c } / z _ { f , k } ^ { c } + c _ { y } } \right] .\tag{9}
$$

Let $\Omega _ { f } ^ { k } \subset \mathbb { R } ^ { 2 }$ denote the image domain of view �. A projection is considered geometrically valid only if the point lies in front of the camera and its projection falls inside the image domain. We therefore define the visibility indicator function

$$
\delta _ { f , k } ( \mathbf { X } ) = \boldsymbol { 1 } ( z _ { f , k } ^ { c } > \epsilon _ { z } ) \cdot \boldsymbol { 1 } ( \mathbf { u } _ { f , k } \in \Omega _ { f } ^ { k } ) ,\tag{10}
$$

where $\epsilon _ { z } > 0$ is a small positive constant used to exclude points located behind the camera, and �(⋅) denotes the indicator function

For each valid projection, the binary artifact mask $M _ { f } ^ { k } : \Omega _ { f } ^ { k }  \{ 0 , 1 \}$ is queried at the projected pixel location

$$
m _ { f , k } ( \mathbf { X } ) = M _ { f } ^ { k } ( \mathbf { u } _ { f , k } ) .\tag{11}
$$

Using the visibility indicator, the number of valid observations of point � is defined as

$$
{ \cal N } _ { \mathrm { v i s } } ^ { ( f ) } ( { \bf { X } } ) = \sum _ { k = 1 } ^ { N _ { f } } \delta _ { f , k } ( { \bf { X } } ) ,\tag{12}
$$

while the number of observations that support the hypothesis that the point belongs to the artifact is

$$
{ \cal N } _ { \mathrm { a r t } } ^ { ( f ) } ( { \mathbf { X } } ) = \sum _ { k = 1 } ^ { N _ { f } } \delta _ { f , k } ( { \mathbf { X } } ) m _ { f , k } ( { \mathbf { X } } ) .\tag{13}
$$

Based on these statistics, we define a semantic consistency score

$$
S _ { f } ( \mathbf { X } ) = \frac { N _ { \mathrm { a r t } } ^ { ( f ) } ( \mathbf { X } ) } { N _ { \mathrm { v i s } } ^ { ( f ) } ( \mathbf { X } ) + \varepsilon } ,\tag{14}
$$

where $\varepsilon > 0$ is a small constant introduced to avoid division by zero.

Intuitively, $s _ { f } ( \mathbf { X } )$ measures the proportion of views in which the projected point is classified as belonging to the artifact. Points that receive consistent support across multiple views are therefore more likely to correspond to the artifact surface.

Finally, points whose semantic consistency score exceeds a predefined threshold � are retained

$$
\widetilde { \mathcal { P } } _ { f } ^ { \mathrm { s e m } } = \{ \mathbf { X } \in \mathcal { P } _ { f } | S _ { f } ( \mathbf { X } ) \geq \tau \} .\tag{15}
$$

This multi-view semantic filtering procedure efectively removes points reconstructed from the supporting plane and surrounding background while preserving geometrically consistent artifact points.

The same procedure is applied independently to the back-side image sequence. Let

$$
\mathcal { I } _ { b } = \{ I _ { b } ^ { k } \} _ { k = 1 } ^ { N _ { b } } , \quad \mathcal { M } _ { b } = \{ M _ { b } ^ { k } \} _ { k = 1 } ^ { N _ { b } } , \quad \mathcal { P } _ { b } = \{ \mathbf { X } _ { i } ^ { ( b ) } \in \mathbb { R } ^ { 3 } \} _ { i = 1 } ^ { M _ { b } } ,\tag{16}
$$

denote the back-side images, artifact masks, and reconstructed dense point cloud, respectively, with corresponding camera parameters

$$
\{ \mathbf { K } _ { b } ^ { k } , \mathbf { R } _ { b } ^ { k } , \mathbf { t } _ { b } ^ { k } \} _ { k = 1 } ^ { N _ { b } } .\tag{17}
$$

Following the same projection and visibility formulation defined for the front side, each point $\mathbf { X } \in \mathcal { P } _ { b }$ is reprojected into every image to determine whether it is visible and whether the projected pixel belongs to the artifact mask.

The number of valid observations and artifact-supporting observations are defined as

$$
{ \cal N } _ { \mathrm { v i s } } ^ { ( b ) } ( { \bf X } ) = \sum _ { k = 1 } ^ { N _ { b } } \delta _ { b , k } ( { \bf X } ) ,\tag{18}
$$

$$
{ \cal N } _ { \mathrm { a r t } } ^ { ( b ) } ( { \mathbf { X } } ) = \sum _ { k = 1 } ^ { N _ { b } } \delta _ { b , k } ( { \mathbf { X } } ) m _ { b , k } ( { \mathbf { X } } ) ,\tag{19}
$$

where $\delta _ { b , k } ( { \mathbf X } )$ denotes the visibility indicator and $m _ { b , k } ( { \mathbf { X } } )$ is the artifact mask value sampled at the projected pixel.

The semantic consistency score for the back-side point cloud is therefore

$$
S _ { b } ( \mathbf { X } ) = \frac { N _ { \mathrm { a r t } } ^ { ( b ) } ( \mathbf { X } ) } { N _ { \mathrm { v i s } } ^ { ( b ) } ( \mathbf { X } ) + \varepsilon } .\tag{20}
$$

Finally, points whose score exceeds a threshold � are retained:

$$
\begin{array} { r } { \tilde { \mathcal { P } } _ { b } ^ { \mathrm { s e m } } = \{ \mathbf { X } \in \mathcal { P } _ { b } \mid S _ { b } ( \mathbf { X } ) \geq \tau \} . } \end{array}\tag{21}
$$

The threshold � controls the strictness of mask-guided point cloud purification. In all experiments, we set $\tau = 0 . 9$ to retain points with highly consistent foreground support across valid projections. This seting is used to suppress residual background and boundary artifacts, and its influence is evaluated in Section 4.3.1.

Although semantic masking efectively suppresses a large portion of the supporting plane and surrounding background, the reconstructed point clouds may still contain geometric artifacts and sparse outliers introduced by the SfM–MVS pipeline. In particular, weakly textured regions, grazing-angle observations, and imperfect depth fusion near the artifact boundary can generate isolated floating points or small background fragments that remain atached to the artifact surface. Therefore, beyond the semantic consistency filtering in $\tilde { \mathcal { P } } _ { s } ^ { s e m } ( s \in \{ f , b \} )$ we further enforce geometric cleanliness via a structured 3D purification procedure that removes statistical outliers and retains only the dominant connected structure corresponding to the artifact.

We first apply Statistical Outlier Removal (SOR) to $\tilde { \mathcal { P } } _ { s } ^ { s \mathrm { e m } }$ . For a point $\mathbf { X } \in \tilde { \mathcal { P } } _ { s } ^ { s \mathrm { e m } }$ , let $\mathcal { N } _ { K } ^ { ( s ) } ( { \bf { X } } )$ denote its �-nearest neighbors, and define the local mean distance

$$
\bar { d } _ { s } ( \mathbf { X } ) = \frac { 1 } { K } \sum _ { \mathbf { Y } \in \mathcal { N } _ { K } ^ { ( s ) } ( \mathbf { X } ) } \| \mathbf { X } - \mathbf { Y } \| _ { 2 } .\tag{22}
$$

Let $\mu _ { s }$ and $\sigma _ { s }$ be the mean and standard deviation of $\{ \bar { d } _ { s } ( \mathbf { X } ) \}$ over $\tilde { \mathcal { P } } _ { s } ^ { s \mathrm { e m } }$ . Points whose neighborhood statistics deviate significantly from the global distribution are regarded as outliers and removed by

$$
\tilde { \mathcal { P } } _ { s } ^ { \mathrm { s o r } } = \left\{ \mathbf { X } \in \tilde { \mathcal { P } } _ { s } ^ { \mathrm { s e m } } \Big | \bar { d } _ { s } ( \mathbf { X } ) \leq \mu _ { s } + \alpha \sigma _ { s } \right\} , \qquad s \in \{ f , b \} ,\tag{23}
$$

where $\alpha > 0$ controls the trimming strength.

After statistical outlier removal, the point cloud may still contain residual background structures that remain atached to the artifact surface. These structures typically appear as small connected fragments or thin surface patches originating from partial reconstruction of the supporting plane near the artifact boundary. Because such remnants may form locally coherent structures rather than isolated points, they are not reliably eliminated by purely local statistical filtering.

To address this issue, we further enforce structural consistency by retaining only the dominant connected component corresponding to the artifact geometry. Since clustering directly on the full-resolution point cloud can be computationally expensive, we first construct a coarse geometric representation through voxel downsampling. Let $\mathcal { D } ( \cdot )$ denote the voxel downsampling operator and define

$$
\mathcal { Q } _ { s } = \mathcal { D } ( \tilde { \mathcal { P } } _ { s } ^ { s \mathrm { o r } } ) .\tag{24}
$$

We then perform density-based spatial clustering on the downsampled point set $\mathcal { Q } _ { s }$ using DBSCAN, which partitions the points into spatially connected clusters $\{ \mathcal { C } _ { s , m } \} _ { m = 1 } ^ { L _ { s } }$ according to a neighborhood radius and a minimum density criterion.

After semantic filtering and statistical denoising, the artifact typically forms the largest and most coherent connected structure in the scene. We therefore identify the dominant cluster by selecting the component with the maximum cardinality

$$
\mathcal { C } _ { s , \star } = \arg \operatorname* { m a x } _ { m } | \mathcal { C } _ { s , m } | , \qquad s \in \{ f , b \} .\tag{25}
$$

Finally, the selected structural component is propagated back to the original-resolution point cloud. Specifically, we retain points in $\tilde { \mathcal { P } } _ { s } ^ { \mathrm { s o r } }$ whose distance to the dominant cluster is smaller than a spatial tolerance $\eta > 0 ;$

$$
\tilde { \mathcal { P } } _ { s } = \left\{ \mathbf { X } \in \tilde { \mathcal { P } } _ { s } ^ { \mathrm { s o r } } \bigg | \operatorname* { m i n } _ { \mathbf { Y } \in \mathcal { C } _ { s , * } } \| \mathbf { X } - \mathbf { Y } \| _ { 2 } < \eta \right\} , \qquad s \in \{ f , b \} .\tag{26}
$$

The resulting $\tilde { \mathcal { P } } _ { f }$ and $\mathcal { \tilde { P } } _ { b }$ are thus purified artifact-only point clouds for the front and back sessions, respectively. Since the above operations are performed independently for each side $s \in \{ f , b \}$ , the purification stage does not assume any cross-side correspondences or overlap; instead, it ensures that the subsequent registration is initialized from two structurally consistent and background-suppressed surfaces, which is particularly important under the near-zero overlap condition induced by double-sided acquisition.

## 3.2 PCA-Based Canonical Normalization and Thickness Estimation

After semantic-guided 2D–3D purification, we obtain two artifact-only point clouds

$$
\tilde { \mathcal { P } } _ { f } \subset \mathbb { R } ^ { 3 } , \qquad \tilde { \mathcal { P } } _ { b } \subset \mathbb { R } ^ { 3 } ,
$$

corresponding to the front and back acquisition sessions. Because the two sides are reconstructed independently, the resulting point sets are defined in diferent session-specific coordinate systems. Consequently, the relative pose between them is unknown, and small scale inconsistencies may also arise during reconstruction. To stabilize later processing, both point clouds are mapped into canonical coordinate systems defined by their intrinsic geometric structure. This normalization step reduces the ambiguity induced by arbitrary reconstruction axes and provides a consistent reference frame for estimating the physical thickness of the artifact.

For each side $s \in \{ f , b \} .$ , the purified point cloud is denoted as $\tilde { \mathcal { P } } _ { s } = \{ \mathbf { X } _ { i } ^ { ( s ) } \} _ { i = 1 } ^ { N _ { s } }$ , where $\mathbf { X } _ { i } ^ { ( s ) } \in \mathbb { R } ^ { 3 }$ . We apply PCA to the centered point cloud by computing

$$
\mathbf { c } _ { s } = \frac { 1 } { N _ { s } } \sum _ { i = 1 } ^ { N _ { s } } \mathbf { X } _ { i } ^ { ( s ) } , \quad \Sigma _ { s } = \frac { 1 } { N _ { s } } \sum _ { i = 1 } ^ { N _ { s } } ( \mathbf { X } _ { i } ^ { ( s ) } - \mathbf { c } _ { s } ) ( \mathbf { X } _ { i } ^ { ( s ) } - \mathbf { c } _ { s } ) ^ { \top } .\tag{27}
$$

The covariance matrix is decomposed as

$$
\Sigma _ { s } = \mathrm { E } _ { s } \Lambda _ { s } \mathrm { E } _ { s } ^ { \top } , \quad \mathrm { E } _ { s } = [ \mathbf { e } _ { s , 1 } , \mathbf { e } _ { s , 2 } , \mathbf { e } _ { s , 3 } ] , \quad \lambda _ { s , 1 } \geq \lambda _ { s , 2 } \geq \lambda _ { s , 3 } \geq 0 .\tag{28}
$$

Here, $\mathbf { e } _ { s , 1 }$ and $\mathbf { e } _ { s , 2 }$ represent the dominant surface directions, while $\mathbf { e } _ { s , 3 }$ corresponds to the least-variance direction and is used as the thickness axis.

For thin sheet-like artifacts, this ordering has a clear geometric interpretation. The eigenvectors associated with the two largest eigenvalues span the dominant surface plane of the artifact, while the eigenvector corresponding to the smallest eigenvalue defines the least-variance direction and therefore approximates the esti mated physical thickness axis. In practice, eigen-decomposition may introduce a reflection due to sign ambiguity. Following the implementation, we enforce a right-handed canonical basis by requiring the rotation matrix to satisfy det $\begin{array} { r } { \langle \mathbf { E } _ { s } \rangle = 1 . \operatorname { I f } \operatorname* { d e t } ( \mathbf { E } _ { s } ) < 0 , } \end{array}$ , the sign of one axis (typically $\mathbf { e } _ { s , 3 } )$ is flipped so that the resulting canonical frame is a proper rotation and preserves geometric orientation consistency.

Using this observation, a canonical coordinate transformation is defined as

$$
\mathbf { X } _ { s } ^ { \mathrm { p c a } } ( \mathbf { X } ) = \mathbf { E } _ { s } ^ { \top } ( \mathbf { X } - \mathbf { c } _ { s } ) , \qquad \mathbf { X } \in \tilde { \mathcal { P } } _ { s } .\tag{29}
$$

Applying this transformation to all points produces the PCA-normalized point cloud

$$
\begin{array} { r } { \tilde { \mathcal { P } } _ { s } ^ { \mathrm { p c a } } = \left\{ \mathbf { X } _ { s } ^ { \mathrm { p c a } } ( \mathbf { X } _ { i } ^ { ( s ) } ) \right\} _ { i = 1 } ^ { N _ { s } } . } \end{array}\tag{30}
$$

The transformed cloud is centered at the origin and registered with the principal axes of the object geometry. This canonicalization has two practical consequences. First, it makes the dominant surface directions comparable across sessions despite arbitrary reconstruction axes. Second, the third canonical coordinate is consistent with the estimated thickness axis, enabling thickness statistics to be computed by a one-dimensional analysis along this axis.

Since independent reconstructions may exhibit scale drift, we additionally estimate a robust isotropic scale factor between the two PCA-normalized clouds using radial statistics in canonical space. Let

$$
\rho _ { s , i } = \left\| \mathbf { X } _ { s } ^ { \mathrm { p c a } } ( \mathbf { X } _ { i } ^ { ( s ) } ) \right\| _ { 2 } , \qquad \mathcal { R } _ { s } = \{ \rho _ { s , i } \} _ { i = 1 } ^ { N _ { s } } .
$$

We define the scale factor that maps the back-side canonical cloud to the front-side canonical scale as

$$
\alpha = \frac { \mathrm { p e r c } _ { 9 0 } ( \mathcal { R } _ { f } ) } { \mathrm { p e r c } _ { 9 0 } ( \mathcal { R } _ { b } ) } .\tag{31}
$$

Using the 90th percentile rather than the maximum avoids sensitivity to isolated outliers or residual floating points, and matches the robust-radius strategy used in the implementation.

$$
\begin{array} { r } { \mathbf { X } _ { s } ^ { \mathrm { p c a } } ( \mathbf { X } _ { i } ^ { ( s ) } ) = \left[ x _ { s , i } ^ { \mathrm { p c a } } \quad y _ { s , i } ^ { \mathrm { p c a } } \quad z _ { s , i } ^ { \mathrm { p c a } } \right] ^ { \top } , } \end{array}\tag{32}
$$

and denote the set of thickness-axis coordinates as

$$
\mathcal { Z } _ { s } = \{ z _ { s , i } ^ { \mathrm { p c a } } \} _ { i = 1 } ^ { N _ { s } } .\tag{33}
$$

Directly using the minimum and maximum values of $\mathcal { Z } _ { s }$ is sensitive to reconstruction noise and extreme artifacts near grazing angles. To obtain a robust estimate, we compute percentile-based bounds

$$
z _ { s } ^ { \mathrm { l o w } } = \mathrm { p e r c } _ { 2 } ( \mathcal { Z } _ { s } ) , \qquad z _ { s } ^ { \mathrm { h i g h } } = \mathrm { p e r c } _ { 9 8 } ( \mathcal { Z } _ { s } ) .\tag{34}
$$

The 2nd–98th percentile interval is used as a robust empirical seting for thickness-axis extent estimation. This interval excludes sparse extreme points caused by reconstruction noise, boundary artifacts, or grazingangle observations, while preserving the main distribution of the front and back surfaces along the thickness axis. The influence of this parameter is further evaluated in Section 4.3.2.

The thickness extent of each side is then defined as

$$
\Delta _ { s } = z _ { s } ^ { \mathrm { h i g h } } - z _ { s } ^ { \mathrm { l o w } } , \qquad s \in \{ f , b \} .\tag{35}
$$

To account for scale drift, we aggregate the two side-specific extents after bringing them to a common scale. Specifically, the back-side thickness extent is scaled by �, yielding a global thickness estimate

$$
h = \frac { \Delta _ { f } + \alpha \Delta _ { b } } { 2 } .\tag{36}
$$

This definition is consistent with the implementation logic that first normalizes scale using a robust radius statistic and then evaluates thickness using canonical �-axis variation. The resulting estimate ℎ provides a physically meaningful thickness proxy that is stable under mild reconstruction noise and inter-session scaling diferences

Finally, the thickness estimate is converted into an explicit separation magnitude along the canonical thickness axis:

$$
z _ { \mathrm { o f f } } = \frac { 1 } { 2 } h .\tag{37}
$$

The value $z _ { \mathrm { o f f } }$ is retained as a thickness-derived geometric prior for downstream use. It encodes a physically plausible half-thickness separation in the canonical coordinate system and is derived solely from intrinsic statistics of the purified point clouds, without assuming any cross-side correspondences

## 3.3 Geometry-Constrained Bidirectional Registration

After PCA-based canonical normalization, the purified front and back point clouds are represented in their respective PCA coordinate systems:

$$
\begin{array} { r } { \tilde { \mathcal P } _ { f } ^ { \mathrm { p c a } } = \bigl \{ \mathbf { X } _ { f , i } ^ { \mathrm { p c a } } \bigr \} _ { i = 1 } ^ { N _ { f } } , \qquad \tilde { \mathcal P } _ { b } ^ { \mathrm { p c a } } = \bigl \{ \mathbf { X } _ { b , j } ^ { \mathrm { p c a } } \bigr \} _ { j = 1 } ^ { N _ { b } } . } \end{array}\tag{38}
$$

In this canonical representation, both point clouds are centered at the origin and registered with their principal axes. For thin sheet-like objects, the third PCA axis corresponds to the direction of minimum geometric variance and therefore approximates the physical thickness direction of the artifact. This canonicalization significantly reduces the pose ambiguity between the independently reconstructed front and back surfaces.

However, eigenvectors obtained from PCA are defined only up to sign. As a result, the PCA coordinate frames of the two surfaces may difer by axis flips, which may cause the back surface to appear mirrored relative to the front surface along one or more principal axes. To resolve this ambiguity, we evaluate a finite set of global orientation hypotheses. As shown in Fig. 3, diferent axis-flip hypotheses lead to diferent point cloud registration results.

For thin sheet-like artifacts captured through double-sided acquisition, the physically plausible configurations can be restricted to 180<sup>∘</sup> rotations around the canonical axes. This yields the following hypothesis set:

$$
\mathcal { R } = \left\{ \mathbf { I } , \mathbf { R } _ { x } ( \pi ) , \mathbf { R } _ { y } ( \pi ) , \mathbf { R } _ { z } ( \pi ) \right\} .\tag{39}
$$

$$
\mathbf { R } _ { x } ( \pi ) = { \left[ \begin{array} { l l l } { 1 } & { 0 } & { 0 } \\ { 0 } & { - 1 } & { 0 } \\ { 0 } & { 0 } & { - 1 } \end{array} \right] } , \qquad \mathbf { R } _ { y } ( \pi ) = { \left[ \begin{array} { l l l } { - 1 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } \\ { 0 } & { 0 } & { - 1 } \end{array} \right] } , \qquad \mathbf { R } _ { z } ( \pi ) = { \left[ \begin{array} { l l l } { - 1 } & { 0 } & { 0 } \\ { 0 } & { - 1 } & { 0 } \\ { 0 } & { 0 } & { 1 } \end{array} \right] } .\tag{40}
$$

Each $\mathbf { R } _ { k } \in \mathcal { R }$ defines a candidate orientation of the back surface relative to the front surface in PCA space. Before ICP refinement, each rotation hypothesis is initialized with the thickness-derived ofset $z _ { \mathrm { o f f } }$ defined in Eq. (37), which separates the two surfaces along the canonical thickness axis:

$$
\mathbf { T } _ { k } ^ { ( 0 ) } = \left[ \mathbf { R } _ { k } \quad \mathbf { t } _ { k } ^ { ( 0 ) } \right] , \quad \mathbf { t } _ { k } ^ { ( 0 ) } = \left[ \begin{array} { l } { 0 } \\ { 0 } \\ { z _ { \mathrm { o f f } } } \end{array} \right] , \quad z _ { \mathrm { o f f } } = \frac { 1 } { 2 } h .\tag{41}
$$

The initialized back-side point cloud under hypothesis ${ \bf R } _ { k }$ is then given by

$$
\begin{array} { r } { \hat { \mathcal { P } } _ { b , k } ^ { ( 0 ) } = \left\{ \mathbf { R } _ { k } \mathbf { X } _ { b , j } ^ { \mathrm { p c a } } + \mathbf { t } _ { k } ^ { ( 0 ) } \right\} _ { j = 1 } ^ { N _ { b } } . } \end{array}\tag{42}
$$

Starting from the initialized configuration, each hypothesis is refined by point-to-plane ICP, with the front surface used as the fixed reference and the transformed back surface as the source. The optimization minimizes

$$
E _ { k } ( { \bf R , t } ) = \sum _ { j = 1 } ^ { N _ { b } } \left( { \bf n } _ { j } ^ { \top } \left( { \bf R } { \bf x } _ { j } + { \bf t } - { \bf y } _ { j } \right) \right) ^ { 2 } , \quad { \bf x } _ { j } = { \bf R } _ { k } { \bf X } _ { b , j } ^ { \mathrm { p c a } } + { \bf t } _ { k } ^ { ( 0 ) } .\tag{43}
$$

After ICP refinement, the resulting transformation for hypothesis � is writen as

$$
\mathbf { T } _ { \mathrm { I C P } , k } = \left[ \begin{array} { c c } { \mathbf { R } _ { \mathrm { I C P } , k } } & { \mathbf { t } _ { \mathrm { I C P } , k } } \\ { \mathbf { 0 } ^ { \top } } & { 1 } \end{array} \right] , \quad \mathbf { T } _ { k } = \mathbf { T } _ { \mathrm { I C P } , k } \mathbf { T } _ { k } ^ { ( 0 ) } .\tag{44}
$$

After ICP refinement, the quality of each hypothesis is evaluated using a thickness-aware fitness criterion. Specifically, correspondences whose residual distance is smaller than a thickness-scaled threshold are considered inliers:

$$
\mathcal { I } _ { k } = \left\{ j \Big | \Big \| \mathbf { T } _ { k } \mathbf { X } _ { b , j } ^ { \mathrm { p c a } } - \mathbf { y } _ { j } \Big \| _ { 2 } < d _ { \operatorname* { m a x } } \right\} , \qquad \mathrm { F i t n e s s } _ { k } = \frac { | \mathcal { I } _ { k } | } { N _ { b } } .\tag{45}
$$

The threshold is defined as

$$
d _ { \mathrm { m a x } } = \gamma h .\tag{46}
$$

![](images/fd36412a7e2f8b9c5d035f8889121e5c2f3c9bd6a91e9deb7b1fffd4e9b69aca.jpg)  
Fig. 3. Visualization of point cloud registration under diferent hypotheses. The first column shows the direct registration results of the two point clouds. The second column presents an exploded view where the two point clouds are spatially separated with a large ofset to beter observe their relative rotations. The best alignment hypothesis is highlighted.

In our implementation, � is set to 2.0 unless otherwise specified. This value is selected as an empirical balance between tolerance and discriminability in the thickness-scaled fitness evaluation. A smaller $\gamma$ makes the correspondence threshold overly strict and may reject valid correspondences afected by reconstruction noise, boundary artifacts, or residual ICP errors. A larger � makes the threshold overly permissive and may cause fitness saturation among diferent rotation hypotheses, thereby weakening the ability to distinguish the correct orientation. The sensitivity of $\gamma$ is analyzed in Section 4.3.3.

Among all hypotheses, the transformation that produces the highest fitness score is selected as the final solution:

$$
k ^ { \ast } = \arg \operatorname* { m a x } _ { k } { \mathrm { F i t n e s s } _ { k } } , \qquad \mathbf { T } _ { b  f } ^ { \mathrm { p c a } } = \mathbf { T } _ { k ^ { \ast } } .\tag{47}
$$

Finally, the optimal transformation obtained in PCA space is mapped back to the original world coordinate systems of the two reconstruction sessions:

$$
{ \bf X } _ { s } ^ { \mathrm { p c a } } = { \bf E } _ { s } ^ { \top } ( { \bf X } - { \bf c } _ { s } ) , \qquad { \bf X } = { \bf E } _ { s } { \bf X } _ { s } ^ { \mathrm { p c a } } + { \bf c } _ { s } .\tag{48}
$$

$$
\begin{array} { r } { \mathbf { T } _ { b  f } ^ { \mathrm { w o r l d } } = [ \begin{array} { l } { \mathbf { E } _ { f } \ \mathbf { c } _ { f } } \\ { \mathbf { 0 } ^ { \top } 1 } \end{array} ] \mathbf { T } _ { b  f } ^ { \mathrm { p c a } } [ \begin{array} { l l } { \mathbf { E } _ { b } ^ { \top } \ - \mathbf { E } _ { b } ^ { \top } \mathbf { c } _ { b } } \\ { \mathbf { 0 } ^ { \top } } & { 1 } \end{array} ] . } \end{array}\tag{49}
$$

Applying $\mathbf { T } _ { b  f } ^ { \mathrm { w o r l d } }$ to $\mathcal { \tilde { P } } _ { b }$ registers the back point cloud with the front surface in the same coordinate system, producing the final stitched artifact model.

## 4 Experimental Results

## 4.1 Experimental Setup

The artifact samples were digitized through independent front-side and back-side acquisition sessions. During each session, the artifact was placed on a flat supporting surface and captured from multiple viewpoints, following the non-contact acquisition requirement for fragile heritage materials.

Images were captured using a Redmi K70 Pro smartphone camera. For each side, the camera was moved around the artifact to acquire a multi-view image sequence with suficient overlap for stable SfM–MVS reconstruction. The front-side and back-side image sequences were processed independently, producing two dense point clouds corresponding to the two surfaces of the artifact.

The reconstructed point clouds were then used as input for the registration experiments. Semantic masks for background removal were generated using SAM2, and the mask inference was performed on a workstation equipped with an NVIDIA RTX 5090D GPU. The same hardware and computational environment were used for subsequent point cloud purification, registration, and baseline comparison to ensure consistent experimental conditions.

The proposed method and all baseline approaches were evaluated on the same set of purified front–back point cloud pairs obtained from the above acquisition and reconstruction workflow.

The artifact samples used in the experiments are shown in Fig. 4. For each object, both front-side and backside views are presented to illustrate the physical samples considered in the evaluation. These samples represent typical flat and fragile cultural heritage artifacts that require independent double-sided acquisition during noncontact digitization.

## 4.2 Comparison with Existing Registration Methods

To assess the efectiveness of the proposed registration method, we compare it with six representative rigid point cloud registration approaches, including Coherent Point Drift (CPD) [15] followed by Iterative Closest Point (ICP), Fast Global Registration (FGR) [33] with ICP refinement, Fast Point Feature Histogram (FPFH)-based registration with ICP refinement, Generalized ICP (GICP), Predator, and GeoTransformer. These baselines cover probabilistic registration, feature-based global alignment, iterative geometric optimization, and learning-based correspondence estimation.

It should be noted that these methods were originally developed for general point cloud registration scenarios in which reliable correspondences or partially overlapping regions are usually available. In contrast, the task considered in this paper focuses on front–back registration of thin, sheet-like heritage artifacts, where the two reconstructed surfaces often share very limited common geometry. Therefore, the comparison is intended to examine the applicability of representative registration methods under this challenging domain-specific seting, rather than to negatively assess their performance in their original target scenarios.

All methods are evaluated using the same purified front–back point cloud pairs described in Section 4.1. Each baseline first estimates a rigid transformation according to its original formulation. To ensure a consistent evaluation protocol, the estimated transformation is further refined using the same point-to-plane ICP procedure. For the learning-based methods, we follow the oficial implementations and use the released pretrained weights for evaluation.

![](images/8e1b9176cf5541c230c6c6ecb9661b4e9c112bbcb4878bf2e9c86cf30abfae15.jpg)  
Fig. 4. Front and back views of all artifacts involved in the experiments.

Since the objective of this work is to obtain a registered front–back model that is consistent with the original image observations, we evaluate the registration results using a rendering-based evaluation procedure. Specif ically, each registered point cloud is rendered from selected calibrated viewpoints using the camera poses estimated during reconstruction. The resulting image-space representations, including rendered images and foreground silhouetes, are then compared with the corresponding captured images and object masks derived from the acquisition data.

This rendering-based evaluation procedure is consistent with the practical acquisition seting of thin, sheetlike heritage artifacts. In this study, the available observations mainly consist of calibrated multi-view images acquired under non-contact conditions. Obtaining an additional independent geometric reference model would require extra scanning, fabrication, and calibration procedures. Active scanning or replica-based validation may provide useful complementary evidence, but the accuracy of the resulting reference model can also be afected by surface accessibility, boundary stability, scanning resolution, and fabrication quality. Therefore, this study focuses on evaluating whether the registered and rendered model remains consistent with the original image observations, while more extensive reference-based 3D geometric validation is left for future work.

Based on this evaluation procedure, we employ both image-space metrics and projection-based geometric metrics. PSNR, SSIM, and LPIPS are used to evaluate the visual fidelity between rendered and captured images. IoU and AreaRatio are further used to assess silhouete overlap and projected area consistency. Although these metrics cannot replace direct 3D geometric measurements, they provide practical evidence for contour alignment, projected scale consistency, and image-plane geometric agreement under the available acquisition conditions.

Peak Signal-to-Noise Ratio (PSNR) [9] is used to measure the pixel-level fidelity between the rendered image <sup>̂</sup> � and the corresponding reference image �. It is defined as

$$
P S N R = 1 0 \log _ { 1 0 } \left( \frac { M A X _ { I } ^ { 2 } } { M S E } \right) ,\tag{50}
$$

where $M A X _ { I }$ denotes the maximum possible pixel intensity, and the mean squared error (MSE) is computed as

$$
M S E = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( I _ { i } - \hat { I _ { i } } \right) ^ { 2 } .\tag{51}
$$

Structural Similarity Index (SSIM) [30] evaluates the structural consistency between two images by considering luminance, contrast, and structural information. For two image patches � and �, SSIM is defined as

$$
S S I M ( x , y ) = \frac { ( 2 \mu _ { x } \mu _ { y } + C _ { 1 } ) ( 2 \sigma _ { x y } + C _ { 2 } ) } { ( \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + C _ { 1 } ) ( \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + C _ { 2 } ) } ,\tag{52}
$$

where $\mu _ { x }$ and $\mu _ { y }$ denote the mean intensities of � and $y , \sigma _ { x } ^ { 2 }$ and $\sigma _ { y } ^ { 2 }$ denote the corresponding variances, and $\sigma _ { x y }$ denotes their covariance.

Learned Perceptual Image Patch Similarity (LPIPS) [32] is adopted to evaluate perceptual similarity in a deep feature space. Let $\phi _ { l } ( \cdot )$ denote the feature map extracted from layer �. LPIPS is computed as

$$
L P I P S ( x , y ) = \sum _ { l } \frac { 1 } { H _ { l } W _ { l } } \sum _ { h , w } \left\| w _ { l } \odot \left( \phi _ { l } ( x ) _ { h , w } - \phi _ { l } ( y ) _ { h , w } \right) \right\| _ { 2 } ^ { 2 } ,\tag{53}
$$

where �<sub>�</sub> denotes the learned channel-wise weights, and $( h , w )$ indexes the spatial locations of the feature map. In addition to image-space similarity, silhouete consistency is evaluated by comparing the rendered projection mask with the artifact mask extracted from the real image. Let $S _ { r }$ and $S _ { g }$ denote the foreground silhouete regions

![](images/4e8cacce6c7de006365ebabb10b70d8ae4f3104d6597d40428fa1eb593093892.jpg)  
Fig. 5. Visual comparison of reconstruction results. (a) Result produced by the proposed method. (b) Reconstruction without thickness constraints, resulting in surface interpenetration. (c) Reconstruction without hypothesis flipping, leading to inaccurate stitching.

of the rendered image and the reference image, respectively. The intersection-over-union is defined as

$$
I o U = \frac { | S _ { r } \cap S _ { g } | } { | S _ { r } \cup S _ { g } | } .\tag{54}
$$

To measure projected area consistency, we further introduce the AreaRatio metric. Let $A _ { r }$ and $A _ { g }$ denote the number of foreground pixels in the rendered mask and the reference mask, respectively. AreaRatio is defined as

$$
A r e a R a t i o = \frac { A _ { r } } { A _ { g } } .\tag{55}
$$

A value closer to 1 indicates that the rendered projection has a projected area more consistent with the observed artifact region.

The quantitative comparison results are summarized in Table 1, where all metrics are averaged over the evaluated viewpoints. The results show that registration performance varies with artifact shape, thickness, texture, and projected extent. In addition, IoU and AreaRatio provide complementary evidence: AreaRatio reflects pro jected area consistency, whereas IoU further measures spatial silhouete overlap. The qualitative results in Fig. 5 further demonstrate the efectiveness of the proposed method under challenging thin-sheet registration conditions.

## 4.3 Parameter Sensitivity and Robustness Evaluation

To address the influence of heuristic parameter choices and evaluate the robustness of the proposed pipeline, we conduct additional sensitivity and robustness analyses. Specifically, we analyze three representative parameters used in diferent stages of the method: the semantic consistency threshold � in mask-guided point cloud purification, the percentile thresholds used for thickness estimation, and the � parameter used in the thickness scaled fitness evaluation. These parameters respectively afect artifact-only point cloud extraction, thickness-axis extent estimation, and rotation hypothesis evaluation. In addition, we conduct a synthetic noise perturbation experiment to evaluate the stability of the proposed registration method under geometric noise. The objects used in these analyses are selected from the experimental dataset as representative examples. The artifacts in our dataset can be roughly divided into four groups according to their size, thickness, texture richness, and material characteristics. The first group includes Artifacts 1, 4, and 5, which are small thin-sheet relics with relatively rich surface textures. The second group includes Artifacts 2 and 6, which are larger and relatively thicker fragments, but their surface textures are comparatively weak. Artifact 3 represents a small compact relic category, with both limited surface area and small thickness, as well as weak texture information. Artifact 7 has material and surface characteristics similar to Artifacts 1, 4, and 5, but it is larger in scale and contains richer texture details.

Table 1. Comparison across diferent artifact cases. Within each artifact group, the top three results are emphasized by red-colored text and orange-shaded backgrounds with decreasing intensity, where darker shades denote higher-ranked performance.
<table><tr><td rowspan=1 colspan=7>Object    Baseline methods      PSNR↑         SSIM↑         LPIPS↓        IoU→ 1       AreaRatio→ 1</td></tr><tr><td rowspan=2 colspan=1>FPFH+ICPCPD+ICP</td><td rowspan=1 colspan=2>12.812          0.477</td><td rowspan=1 colspan=2>0.620           0.292</td><td rowspan=1 colspan=2>3.134</td></tr><tr><td rowspan=1 colspan=2>10.668           0.358</td><td rowspan=1 colspan=1>0.611</td><td rowspan=1 colspan=1>0.303</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.607</td></tr><tr><td rowspan=1 colspan=1>FGR+ICP</td><td rowspan=1 colspan=2>10.585          0.298</td><td rowspan=1 colspan=1>0.545</td><td rowspan=1 colspan=1>0.497</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.941</td></tr><tr><td rowspan=2 colspan=1>Artifact1  GICPGeoTransformer</td><td rowspan=1 colspan=2>11.648          0.434</td><td rowspan=1 colspan=2>0.564           0.305</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.338</td></tr><tr><td rowspan=1 colspan=2>12.694          0.436</td><td rowspan=1 colspan=2>0.494           0.486</td><td rowspan=1 colspan=2>2.379</td></tr><tr><td rowspan=1 colspan=1>Predator</td><td rowspan=1 colspan=2>13.137          0.483</td><td rowspan=1 colspan=2>0.485           0.459</td><td rowspan=1 colspan=2>2.166</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=2>15.535          0.536</td><td rowspan=1 colspan=2>0.327           0.600</td><td rowspan=1 colspan=2>1.075</td></tr><tr><td rowspan=1 colspan=1>FPFH+ICP</td><td rowspan=1 colspan=2>12.601           0.395</td><td rowspan=1 colspan=2>0.558           0.510</td><td rowspan=1 colspan=2>0.510</td></tr><tr><td rowspan=1 colspan=1>CPD+ICP</td><td rowspan=1 colspan=1>11.193</td><td rowspan=1 colspan=1>0.409</td><td rowspan=1 colspan=1>0.586</td><td rowspan=1 colspan=1>0.376</td><td rowspan=1 colspan=2>1.376</td></tr><tr><td rowspan=1 colspan=1>FGR+ICP</td><td rowspan=1 colspan=1>14.630</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=1>0.445</td><td rowspan=1 colspan=1>0.734</td><td rowspan=1 colspan=2>1.038</td></tr><tr><td rowspan=2 colspan=1>Artifact2  GICPGeoTransformer</td><td rowspan=1 colspan=1>13.362</td><td rowspan=1 colspan=1>0.414</td><td rowspan=1 colspan=1>0.474</td><td rowspan=1 colspan=1>0.582</td><td rowspan=1 colspan=2>1.469</td></tr><tr><td rowspan=1 colspan=1>14.412</td><td rowspan=1 colspan=1>0.414</td><td rowspan=1 colspan=1>0.449</td><td rowspan=1 colspan=1>0.738</td><td rowspan=1 colspan=2>0.932</td></tr><tr><td rowspan=2 colspan=1>PredatorOurs</td><td rowspan=1 colspan=1>14.839</td><td rowspan=1 colspan=1>0.481</td><td rowspan=1 colspan=1>0.438</td><td rowspan=1 colspan=1>0.733</td><td rowspan=1 colspan=2>1.166</td></tr><tr><td rowspan=1 colspan=1>15.606</td><td rowspan=1 colspan=1>0.349</td><td rowspan=1 colspan=2>0.367           0.757</td><td rowspan=1 colspan=2>0.987</td></tr><tr><td rowspan=1 colspan=1>FPFH+ICP</td><td rowspan=1 colspan=1>13.900</td><td rowspan=1 colspan=1>0.416</td><td rowspan=1 colspan=1>0.547</td><td rowspan=1 colspan=1>0.500</td><td rowspan=1 colspan=2>2.007</td></tr><tr><td rowspan=1 colspan=1>CPD+ICP</td><td rowspan=1 colspan=1>15.573</td><td rowspan=1 colspan=1>0.490</td><td rowspan=1 colspan=1>0.545</td><td rowspan=1 colspan=1>0.186</td><td rowspan=1 colspan=2>1.623</td></tr><tr><td rowspan=1 colspan=1>FGR+ICP</td><td rowspan=1 colspan=1>15.128</td><td rowspan=1 colspan=1>0.234</td><td rowspan=1 colspan=1>0.677</td><td rowspan=1 colspan=1>0.383</td><td rowspan=1 colspan=2>2.018</td></tr><tr><td rowspan=1 colspan=1>Artifact3  GICP</td><td rowspan=1 colspan=1>13.866</td><td rowspan=1 colspan=1>0.400</td><td rowspan=1 colspan=1>0.547</td><td rowspan=1 colspan=1>0.504</td><td rowspan=1 colspan=2>2.013</td></tr><tr><td rowspan=1 colspan=1>GeoTransformer</td><td rowspan=1 colspan=1>18.116</td><td rowspan=1 colspan=1>0.474</td><td rowspan=1 colspan=1>0.496</td><td rowspan=1 colspan=1>0.627</td><td rowspan=1 colspan=2>1.805</td></tr><tr><td rowspan=1 colspan=1>Predator</td><td rowspan=1 colspan=1>15.940</td><td rowspan=1 colspan=1>0.390</td><td rowspan=1 colspan=1>0.549</td><td rowspan=1 colspan=1>0.651</td><td rowspan=1 colspan=2>1.614</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>20.909</td><td rowspan=1 colspan=1>0.619</td><td rowspan=1 colspan=1>0.342</td><td rowspan=1 colspan=1>0.689</td><td rowspan=1 colspan=2>1.073</td></tr><tr><td rowspan=3 colspan=1>FPFH+ICP</td><td rowspan=3 colspan=1>18.235</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>0.750</td><td rowspan=2 colspan=1>0.301</td><td rowspan=2 colspan=1>0.840</td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>1.680</td></tr><tr><td rowspan=1 colspan=1>CPD+ICP</td><td rowspan=1 colspan=1>12.452</td><td rowspan=1 colspan=1>0.318</td><td rowspan=1 colspan=1>0.704</td><td rowspan=1 colspan=1>0.378</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.504</td></tr><tr><td rowspan=1 colspan=1>FGR+ICP</td><td rowspan=1 colspan=1>15.560</td><td rowspan=1 colspan=1>0.397</td><td rowspan=1 colspan=1>0.359</td><td rowspan=1 colspan=1>0.521</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.254</td></tr><tr><td rowspan=1 colspan=1>Artifact4  GICP</td><td rowspan=1 colspan=1>13.144</td><td rowspan=1 colspan=1>0.283</td><td rowspan=1 colspan=1>0.538</td><td rowspan=1 colspan=1>0.885</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.445</td></tr><tr><td rowspan=1 colspan=1>GeoTransformer</td><td rowspan=1 colspan=1>15.119</td><td rowspan=1 colspan=1>0.435</td><td rowspan=1 colspan=1>0.403</td><td rowspan=1 colspan=1>0.730</td><td rowspan=1 colspan=2>1.338</td></tr><tr><td rowspan=1 colspan=1>Predator</td><td rowspan=1 colspan=1>15.413</td><td rowspan=1 colspan=1>0.350</td><td rowspan=1 colspan=1>0.455</td><td rowspan=1 colspan=1>0.773</td><td rowspan=1 colspan=2>1.175</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>18.489</td><td rowspan=1 colspan=1>0.762</td><td rowspan=1 colspan=1>0.289</td><td rowspan=1 colspan=1>0.681</td><td rowspan=1 colspan=2>1.114</td></tr><tr><td rowspan=1 colspan=1>FPFH+ICP</td><td rowspan=1 colspan=1>14.668</td><td rowspan=1 colspan=1>0.354</td><td rowspan=1 colspan=1>0.496</td><td rowspan=1 colspan=1>0.471</td><td rowspan=1 colspan=2>1.820</td></tr><tr><td rowspan=1 colspan=1>CPD+ICP</td><td rowspan=1 colspan=1>15.388</td><td rowspan=1 colspan=1>0.400</td><td rowspan=1 colspan=1>0.438</td><td rowspan=1 colspan=1>0.496</td><td rowspan=1 colspan=2>1.144</td></tr><tr><td rowspan=1 colspan=1>FGR+ICP</td><td rowspan=1 colspan=1>15.439</td><td rowspan=1 colspan=1>0.362</td><td rowspan=1 colspan=1>0.431</td><td rowspan=1 colspan=1>0.601</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.229</td></tr><tr><td rowspan=1 colspan=1>Artifact5  GICP</td><td rowspan=1 colspan=1>14.910</td><td rowspan=1 colspan=1>0.357</td><td rowspan=1 colspan=1>0.445</td><td rowspan=1 colspan=1>0.559</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.713</td></tr><tr><td rowspan=1 colspan=1>GeoTransformer</td><td rowspan=1 colspan=1>15.875</td><td rowspan=1 colspan=1>0.403</td><td rowspan=1 colspan=1>0.369</td><td rowspan=1 colspan=1>0.709</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.392</td></tr><tr><td rowspan=1 colspan=1>Predator</td><td rowspan=1 colspan=1>14.974</td><td rowspan=1 colspan=1>0.344</td><td rowspan=1 colspan=1>0.487</td><td rowspan=1 colspan=1>0.552</td><td rowspan=1 colspan=2>1.300</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>14.602</td><td rowspan=1 colspan=1>0.286</td><td rowspan=1 colspan=1>0.497</td><td rowspan=1 colspan=1>0.438</td><td rowspan=1 colspan=2>0.982</td></tr><tr><td rowspan=3 colspan=1>FPFH+ICP</td><td rowspan=3 colspan=1>15.529</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>0.536</td><td rowspan=2 colspan=1>0.466</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.520</td><td rowspan=1 colspan=2>1.971</td></tr><tr><td rowspan=1 colspan=1>CPD+ICP</td><td rowspan=1 colspan=1>15.193</td><td rowspan=1 colspan=1>0.509</td><td rowspan=1 colspan=1>0.522</td><td rowspan=1 colspan=1>0.370</td><td rowspan=1 colspan=2>2.029</td></tr><tr><td rowspan=1 colspan=1>FGR+ICP</td><td rowspan=1 colspan=1>16.327</td><td rowspan=1 colspan=1>0.489</td><td rowspan=1 colspan=1>0.532</td><td rowspan=1 colspan=1>0.772</td><td rowspan=1 colspan=2>1.254</td></tr><tr><td rowspan=1 colspan=1>Artifact6  GICP</td><td rowspan=1 colspan=1>15.473</td><td rowspan=1 colspan=1>0.526</td><td rowspan=1 colspan=1>0.466</td><td rowspan=1 colspan=1>0.515</td><td rowspan=1 colspan=2>1.947</td></tr><tr><td rowspan=1 colspan=1>GeoTransformer</td><td rowspan=1 colspan=1>16.710</td><td rowspan=1 colspan=1>0.474</td><td rowspan=1 colspan=1>0.463</td><td rowspan=1 colspan=1>0.717</td><td rowspan=1 colspan=2>1.453</td></tr><tr><td rowspan=1 colspan=1>Predator</td><td rowspan=1 colspan=1>15.937</td><td rowspan=1 colspan=1>0.434</td><td rowspan=1 colspan=1>0.546</td><td rowspan=1 colspan=1>0.548</td><td rowspan=1 colspan=2>1.444</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>18.177</td><td rowspan=1 colspan=1>0.499</td><td rowspan=1 colspan=1>0.396</td><td rowspan=1 colspan=1>0.710</td><td rowspan=1 colspan=2>1.013</td></tr><tr><td rowspan=2 colspan=1>FPFH+ICP</td><td rowspan=2 colspan=1>17.939</td><td rowspan=2 colspan=1>0.565</td><td rowspan=2 colspan=1>0.530</td><td></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=2>1.166</td></tr><tr><td rowspan=1 colspan=1>CPD+ICP</td><td rowspan=1 colspan=1>17.566</td><td rowspan=1 colspan=1>0.551</td><td rowspan=1 colspan=1>0.552</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=2>1.033</td></tr><tr><td rowspan=1 colspan=1>FGR+ICP</td><td rowspan=1 colspan=1>17.113</td><td rowspan=1 colspan=1>0.501</td><td rowspan=1 colspan=1>0.583</td><td rowspan=1 colspan=1>0.377</td><td rowspan=1 colspan=2>1.611</td></tr><tr><td rowspan=1 colspan=1>Artifact7   GICP</td><td rowspan=1 colspan=1>18.204</td><td rowspan=1 colspan=1>0.586</td><td rowspan=1 colspan=1>0.496</td><td rowspan=1 colspan=1>0.228</td><td rowspan=1 colspan=2>2.240</td></tr><tr><td rowspan=1 colspan=1>GeoTransformer</td><td rowspan=1 colspan=1>17.305</td><td rowspan=1 colspan=1>0.514</td><td rowspan=1 colspan=1>0.523</td><td rowspan=1 colspan=1>0.171</td><td rowspan=1 colspan=2>1.992</td></tr><tr><td rowspan=2 colspan=1>PredatorOurs</td><td rowspan=1 colspan=1>18.204</td><td rowspan=1 colspan=1>0.593</td><td rowspan=1 colspan=1>0.522</td><td rowspan=1 colspan=1>0.088</td><td rowspan=1 colspan=2>2.151</td></tr><tr><td rowspan=1 colspan=2>19.666          0.677</td><td rowspan=1 colspan=1>0.504</td><td rowspan=1 colspan=3>0.790             0.865</td></tr></table>

![](images/ed8e962746dd09c78487467a31644c386b6fed02bceafed0f1261b35d88633f5.jpg)  
τ = 0.4

![](images/71ea1a0f3365c3db32372bc7d281943b4cbbdeafeec53b484dfcabb49fa21dea.jpg)  
τ = 0.5

![](images/d1d59d04c2b160953c485c7122448d334254ee008674d7a826813281da182fe7.jpg)  
τ = 0.6

![](images/02dc81c0f7bdd68ab124da70e28c44b10113d6f3715ba871d3ff9ca58001ee84.jpg)  
τ = 0.9

![](images/2f893c5a7f246050ccbd209efc64730cefadefb25e96af03e4eb848d7211926e.jpg)  
τ = 0.8

![](images/d1fb7864a022a5f6ed291039ffa555e92639428c11de0451afb3557c5cdebed5.jpg)  
τ = 0.7  
Fig. 6. Sensitivity of mask-guided point cloud purification to diferent semantic consistency thresholds �. As � increases from 0.4 to 0.9, the purified point clouds remain visually stable, with only minor changes near the boundary regions.

4.3.1 Sensitivity to the Semantic Consistency Threshold. The semantic consistency threshold � is used in the mask-guided point cloud purification stage. It determines whether a reconstructed 3D point should be retained according to the proportion of valid visible views in which its projection falls inside the artifact mask. A smaller � preserves more points but may retain residual background or boundary artifacts, whereas a larger � applies a stricter multi-view consistency requirement and removes more uncertain points.

To evaluate the influence of this parameter, we test diferent � values from 0.4 to 0.9. The corresponding purifi cation results are shown in Fig. 6. As � increases, the retained point cloud becomes slightly cleaner near the object boundary, while the main artifact structure remains stable. This indicates that the mask-guided purification result is not highly sensitive to the specific choice of � within this range. In particular, when � is increased to 0.9, the visual result is already very similar to those obtained with neighboring high-threshold setings, suggesting that further increasing the threshold does not provide a noticeable improvement in artifact purification.

However, seting � to 1.0 is not used as a valid experimental seting in our implementation. Such an excessively strict threshold would retain only points with nearly perfect semantic consistency across valid projections, leaving only a very small number of points after purification. Consequently, � = 1.0 is regarded as a degenerate case rather than a meaningful purification seting.

Based on these observations, � = 0.9 is adopted as the default seting. This value corresponds to a strict multiview semantic consistency criterion: a point is retained only when it is classified as belonging to the artifact in at least 90% of its valid projections. This seting efectively suppresses residual background points and boundary artifacts while preserving the main artifact structure. Therefore, $\tau \ = \ 0 . 9$ provides a conservative but stable purification seting for generating artifact-only point clouds before registration.

Table 2. Efect of percentile intervals on thickness-axis extent estimation and rotation hypothesis selection.
<table><tr><td>Object</td><td>Percentile interval</td><td>Front extent</td><td>Back extent</td><td>Selected hypothesis</td></tr><tr><td rowspan="5">Artifact2</td><td>1%-99%</td><td>0.30191</td><td>0.24756</td><td>Identity hypothesis</td></tr><tr><td>2%-98%</td><td>0.27836</td><td>0.23128</td><td>Identity hypothesis</td></tr><tr><td>3%-97%</td><td>0.26240</td><td>0.22051</td><td>Identity hypothesis</td></tr><tr><td>5%-95%</td><td>0.24273</td><td>0.20518</td><td>Identity hypothesis</td></tr><tr><td>10%-90%</td><td>0.20628</td><td>0.17403</td><td>Identity hypothesis</td></tr><tr><td rowspan="5">Artifact4</td><td>1%-99%</td><td>0.06878</td><td>0.13458</td><td>Flip-Y hypothesis</td></tr><tr><td>2%-98%</td><td>0.06330</td><td>0.04050</td><td>Flip-Z hypothesis</td></tr><tr><td>3%-97%</td><td>0.05917</td><td>0.03443</td><td>Flip-Z hypothesis</td></tr><tr><td>5%-95%</td><td>0.05044</td><td>0.02903</td><td>Flip-Y hypothesis</td></tr><tr><td>10%-90%</td><td>0.03712</td><td>0.02038</td><td>Flip-Y hypothesis</td></tr><tr><td rowspan="5">Artifact6</td><td>1%-99%</td><td>0.16997</td><td>0.21118</td><td>Identity hypothesis</td></tr><tr><td>2%-98%</td><td>0.14866</td><td>0.17846</td><td>Flip-Y hypothesis</td></tr><tr><td>3%-97%</td><td>0.13794</td><td>0.16133</td><td>Flip-Y hypothesis</td></tr><tr><td>5%-95%</td><td>0.12503</td><td>0.14470</td><td>Flip-Y hypothesis</td></tr><tr><td>10%-90%</td><td>0.10709</td><td>0.11877</td><td>Flip-Y hypothesis</td></tr></table>

4.3.2 Sensitivity to Percentile Thresholds in Thickness Estimation. The percentile thresholds used in thickness estimation determine the percentile-based bounds along the PCA-derived thickness axis. Since the resulting thickness-axis extent is further used to compute the thickness-derived ofset and the ICP correspondence thresh old during hypothesis evaluation, this parameter may afect both the estimated extent and the selected rotation hypothesis.

Table 2 reports the influence of diferent percentile intervals on representative objects. Artifacts 2, 4, and 6 were selected because they cover a broad range of thickness scales and geometric conditions in the dataset. Arti fact 4 represents a typical thin-sheet case, whereas Artifacts 2 and 6 are larger and relatively thicker fragments. This selection allows us to evaluate whether the percentile interval remains robust for both normal thin-sheet artifacts and fragments with larger thickness-axis extents. The results show that, as the percentile interval becomes narrower, the front-side and back-side thickness-axis extents generally decrease. This indicates that the percentile interval directly controls how much of the thickness-axis distribution is retained for thickness estimation.

When the interval is too wide, sparse extreme points caused by reconstruction noise or boundary artifacts may still be included. In this case, the thickness-axis extent can be overestimated, which may further afect the thickness-derived ofset and the ICP correspondence threshold. As a result, the subsequent hypothesis evaluation may select a diferent rotation hypothesis.

When the interval is too narrow, more points near the two ends of the thickness-axis distribution are removed. Although this can suppress sparse extreme points, it may also discard valid geometric variation along the thickness direction. This can lead to an underestimated thickness-axis extent and may also change the selected rotation hypothesis.

The results in Table 2 show that the adopted 2%–98% interval provides a balanced seting. It removes sparse extreme points more efectively than overly wide intervals, while avoiding the excessive trimming caused by overly narrow intervals. Therefore, the 2%–98% interval is used as the default percentile interval for robust thickness estimation in the proposed pipeline.

Table 3. Sensitivity of fitness evaluation to diferent � values on representative objects.
<table><tr><td>Object</td><td>γ</td><td>Identity</td><td>Flip-X</td><td>Flip-Y</td><td>Flip-Z</td><td>Selected hypothesis</td></tr><tr><td rowspan="6">Artifact4</td><td>0.5</td><td>0.76839</td><td>0.84200</td><td>0.89447</td><td>0.81110</td><td>Flip-Y hypothesis</td></tr><tr><td>1.0</td><td>0.91305</td><td>0.92541</td><td>0.97597</td><td>0.95562</td><td>Flip-Y hypothesis</td></tr><tr><td>1.5</td><td>0.94665</td><td>0.94321</td><td>0.98011</td><td>0.97681</td><td>Flip-Y hypothesis</td></tr><tr><td>2.0</td><td>0.95818</td><td>0.95464</td><td>0.99229</td><td>0.99823</td><td>Flip-Z hypothesis</td></tr><tr><td>2.5</td><td>0.97021</td><td>0.96180</td><td>0.99949</td><td>1.00000</td><td>Flip-Z hypothesis</td></tr><tr><td>3.0</td><td>0.98178</td><td>0.97244</td><td>1.00000</td><td>1.00000</td><td>Ambiguous</td></tr><tr><td rowspan="6">Artifact6</td><td>0.5</td><td>0.89332</td><td>0.90663</td><td>0.96612</td><td>0.88177</td><td>Flip-Y hypothesis</td></tr><tr><td>1.0</td><td>0.98077</td><td>0.95568</td><td>0.98939</td><td>0.96194</td><td>Flip-Y hypothesis</td></tr><tr><td>1.5</td><td>0.99745</td><td>0.98316</td><td>1.00000</td><td>0.98573</td><td>Flip-Y hypothesis</td></tr><tr><td>2.0</td><td>0.99998</td><td>0.99765</td><td>1.00000</td><td>0.99515</td><td>Flip-Y hypothesis</td></tr><tr><td>2.5</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>0.99914</td><td>Ambiguous</td></tr><tr><td>3.0</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>Ambiguous</td></tr><tr><td rowspan="6">Artifact7</td><td>0.5</td><td>0.53114</td><td>0.87536</td><td>0.97764</td><td>0.58902</td><td>Flip-Y hypothesis</td></tr><tr><td>1.0</td><td>0.77289</td><td>0.93829</td><td>0.98575</td><td>0.83786</td><td>Flip-Y hypothesis</td></tr><tr><td>1.5</td><td>0.87849</td><td>0.96224</td><td>0.98895</td><td>0.88839</td><td>Flip-Y hypothesis</td></tr><tr><td>2.0</td><td>0.94078</td><td>0.97790</td><td>0.99150</td><td>0.92091</td><td>Flip-Y hypothesis</td></tr><tr><td>2.5</td><td>0.96627</td><td>0.98882</td><td>0.99265</td><td>0.95209</td><td>Flip-Y hypothesis</td></tr><tr><td>3.0</td><td>0.98313</td><td>0.99278</td><td>0.99355</td><td>0.97528</td><td>Flip-Y hypothesis</td></tr></table>

4.3.3 Sensitivity to the � Parameter in Fitness Evaluation. The parameter � is used in the fitness evaluation to define the thickness-scaled ICP correspondence threshold. Specifically, the maximum correspondence distance is defined as $d _ { \mathrm { m a x } } = \gamma h$ , where ℎ denotes the estimated thickness. Therefore, � controls the tolerance used to count inliers during rotation hypothesis evaluation. A smaller � leads to a stricter correspondence threshold, whereas a larger � produces a more permissive threshold.

To analyze the influence of this parameter, we evaluate � values from 0.5 to 3.0 with an interval of 0.5. For each value, all rotation hypotheses are refined using point-to-plane ICP, and the corresponding fitness scores are reported in Table 3. The hypothesis with the highest fitness score is selected. Objects 4, 6, and 7 were selected because they provide diferent thickness scales and hypothesis ambiguity levels. Object 4 represents a typical thin-sheet case, Object 6 represents a larger and relatively thicker fragment, and Object 7 provides a larger texture-rich artifact. This selection allows us to examine whether the default � value provides a balanced thickness-scaled fitness evaluation across diferent geometric conditions, avoiding both overly strict correspondence rejection and excessive fitness saturation. When two or more hypotheses obtain the same highest fitness score, the result is marked as Ambiguous, because the fitness evaluation cannot uniquely determine a single rotation hypothesis.

The results show that smaller � values produce lower fitness scores. This is because the correspondence threshold is more restrictive, and valid correspondences may be rejected under reconstruction noise, boundary artifacts, or residual alignment errors after ICP refinement. In this case, the fitness evaluation may become overly strict and sensitive to local reconstruction errors, which can afect the ranking of rotation hypotheses.

As � increases, the fitness scores generally become higher because more correspondences are counted as inliers. However, excessively large � values make the correspondence threshold too permissive. This reduces the discriminative ability of the fitness evaluation, since diferent rotation hypotheses may obtain identical or nearly identical high fitness scores. For example, in Table 3, Object 4 becomes Ambiguous when $\gamma = 3 . 0 $ , and Object 6 becomes Ambiguous when $\gamma = 2 . 5$ and $\gamma = 3 . 0$ . These cases indicate that overly large � values can lead to fitness saturation, making the selected hypothesis unreliable.

![](images/76e7b883543722b6f6a9db36f7e6c10cafad39d2295ddfdce95502a886b1a772.jpg)  
Fig. 7. Robustness evaluation under diferent noise perturbations. Two types of noise are introduced to the purified front– back point clouds, and the registration results are compared under increasing noise levels (0%, 1% and 2%). The proposed method successfully aligns and stitches the point clouds even in the presence of noise, demonstrating stable registration performance.

The adopted seting $\gamma = 2 . 0$ provides a balanced choice in the proposed pipeline. Compared with smaller values, it provides suficient tolerance for valid correspondences afected by reconstruction noise and residual misalignment. Compared with larger values, it avoids the stronger ambiguity caused by excessive relaxation of the correspondence threshold. Therefore, $\gamma = 2 . 0$ is used as an empirical default seting for fitness evaluation in the following experiments.

4.3.4 Robustness to Synthetic Noise. To further evaluate the robustness of the proposed registration method under noisy conditions, additional experiments were conducted by introducing synthetic noise to the purified front–back point clouds. Specifically, random perturbations were added to the coordinates ofboth point clouds to simulate measurement errors that commonly occur in practical 3D reconstruction. Noise levels were considered, corresponding to 0%, 1% and 2% of the scene scale. For each noise level, the disturbed point clouds were reregistered and stitched using the proposed method. The experimental results in Fig. 7 show that the method successfully aligns and merges the front and back point clouds across all tested noise levels. Even when the noise intensity increases from 0% to 1% and further to 2%, the registration process remains stable and produces consistent alignment without structural misalignment or failure. These results demonstrate that the proposed method exhibits robustness to moderate geometric noise in the input point clouds.

## 4.4 Ablation Study

To analyze the contribution of each component in the proposed method, we conduct an ablation study by progressively removing key modules from the full pipeline. The proposed method integrates two major components:

PCA-based canonical normalization with thickness estimation and geometry-constrained bidirectional registration with rotation hypothesis evaluation. To understand the individual impact ofthese components, we construct two degraded variants of the method and compare them with the full model.

First, we remove the thickness prior used in the registration stage. In this seting, the algorithm relies solely on geometric correspondence during point cloud alignment without enforcing the thickness-aware separation constraint described in Section 3.3. This variant evaluates the importance of incorporating physical thickness information for stabilizing the registration of thin sheet-like structures.

Second, we disable the rotation hypothesis enumeration step and perform registration using only a single initial orientation. Since PCA-based coordinate frames may exhibit sign ambiguity, the absence of hypothesis evaluation may cause the two surfaces to be incorrectly flipped or misaligned. This experiment evaluates the role of the rotation hypothesis search strategy in resolving orientation ambiguity between independently reconstructed surfaces.

As shown in Fig. 8, we select two representative objects to conduct the ablation experiments and visually analyze the influence of each component in the proposed pipeline.

![](images/867d1b070622a14c4294db80189c4cd6f20022e9312a68df32450df15ae0b0a1.jpg)  
Fig. 8. Ablation study on two objects. From left to right: full model, without thickness prior, and without rotation hypothesis evaluation. Removing the thickness constraint leads to noticeable surface interpenetration, while further removing the rotation hypothesis causes severe misalignment between the reconstructed front and back surfaces.

After removing the thickness prior, we observe that the alignment becomes less constrained along the surface normal direction. As a result, the two reconstructed surfaces tend to interpenetrate, and noticeable surface piercing artifacts appear in the merged model. This phenomenon indicates that the thickness-aware constraint plays an important role in preventing unrealistic overlap between the front and back surfaces and helps stabilize the registration of thin structures. Furthermore, when the rotation hypothesis evaluation is also removed, the align ment quality deteriorates significantly. Without exploring multiple orientation candidates, the two surfaces may be incorrectly flipped or misoriented, leading to obvious mismatches between the front and back reconstructions. In several cases, the two sides of the object fail to align properly, suggesting that rotation hypothesis enumeration is important for mitigating orientation ambiguity introduced by PCA-based canonical frames.

The ablation results indicate that the thickness prior and rotation hypothesis evaluation are both important for stable front–back registration. Removing the thickness prior tends to produce physically implausible alignments, where the front and back surfaces collapse toward each other or partially interpenetrate. Disabling rotation hypothesis enumeration makes the registration more susceptible to orientation ambiguity, leading to incorrect front–back alignment. In contrast, the complete method preserves a more plausible thickness separation and yields visually more stable stitching results, demonstrating the efectiveness of combining PCA-based geometric normalization, rotation hypothesis evaluation, and thickness-aware registration.

## 5 Conclusion and Future Work

This paper presents a geometry-constrained bidirectional point cloud registration method for thin, sheet-like cultural heritage artifacts. It addresses the challenges of non-contact double-sided digitization, where the front and back surfaces are reconstructed independently with litle or no geometric overlap.

The method combines semantic-guided purification, PCA-based canonical normalization, and thickness-aware registration. Semantic purification removes the supporting plane and background structures, while PCA normalization establishes a canonical coordinate system for geometric analysis and robust thickness estimation. Using this geometric prior, rotation hypothesis evaluation and point-to-plane ICP refinement are applied to resolve orientation ambiguity and reduce non-physical surface interpenetration during alignment.

Experimental results show that the proposed method achieves competitive or improved performance in most tested cases, particularly in projected area consistency and physically plausible front–back alignment. Compared with representative conventional and learning-based registration baselines, the proposed thickness-aware strategy reduces the risk of structural collapse when registering thin-sheet objects with limited geometric features, although certain baselines may obtain beter image-space or silhouete metrics for some artifacts. Despite its efectiveness, the method may still fail for artifacts with strong geometric symmetry or weak boundary cues, where front–back orientation ambiguity can lead to inaccurate stitching. Further improvements are therefore needed to enhance robustness across diverse thin-sheet reconstruction scenarios.

Future work will explore several extensions. First, reference objects with known geometry, such as highprecision 3D-printed replicas, could provide more direct validation through explicit 3D geometric error metrics in addition to the current rendering- and projection-based metrics. Second, integrating the proposed registration strategy with neural rendering techniques such as 3D Gaussian Splating may improve the visual fidelity of digital heritage models. Finally, the method could be extended to multi-fragment alignment for reconstructing fragmented cultural artifacts from independently scanned pieces.

## References

[1] Dror Aiger, Niloy J. Mitra, and Daniel Cohen-Or. 2008. 4-Points Congruent Sets for Robust Pairwise Surface Registration. ACM Transactions on Graphics 27, 3, Article 85 (2008), 10 pages. doi:10.1145/1360612.1360684

[2] Yasuhiro Aoki, Hunter Goforth, Rangaprasad Arun Srivatsan, and Simon Lucey. 2019. PointNetLK: Robust & Eficient Point Cloud Registration Using PointNet. In 2019 IEEE/CVF Conference on Computer Vision and Patern Recognition (CVPR). IEEE, 7156–7165. doi:10. 1109/CVPR.2019.00733

[3] Xuyang Bai, Zixin Luo, Lei Zhou, Hongbo Fu, Long Quan, and Chiew-Lan Tai. 2020. D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features. In 2020 IEEE/CVF Conference on Computer Vision and Patern Recognition (CVPR). IEEE, 6358–6366. doi:10.1109/CVPR42600.2020.00639

[4] Paul J. Besl and Neil D. McKay. 1992. A Method for Registration of 3-D Shapes. IEEE Transactions on Patern Analysis and Machine Intelligence 14, 2 (1992), 239–256. doi:10.1109/34.121791

[5] Yang Chen and Gérard Medioni. 1992. Object Modelling by Registration of Multiple Range Images. Image and Vision Computing 10, 3 (1992), 145–155. doi:10.1016/0262-8856(92)90066-C

[6] Christopher Choy, Jaesik Park, and Vladlen Koltun. 2019. Fully Convolutional Geometric Features. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 8957–8965. doi:10.1109/ICCV.2019.00905

[7] Yasutaka Furukawa and Jean Ponce. 2010. Accurate, Dense, and Robust Multiview Stereopsis. IEEE Transactions on Patern Analysis and Machine Intelligence 32, 8 (2010), 1362–1376. doi:10.1109/TPAMI.2009.161

[8] Gabriele Guidi, Michele Russo, and Davide Angheleddu. 2014. 3D Survey and Virtual Reconstruction of Archeological Sites. Digital Applications in Archaeology and Cultural Heritage 1, 2 (2014), 55–69. doi:10.1016/j.daach.2014.01.001

[9] Alain Hore and Djemel Ziou. 2010. Image Quality Metrics: PSNR vs. SSIM. In 2010 20th International Conference on Patern Recognition. IEEE, 2366–2369. doi:10.1109/ICPR.2010.579

[10] Shengyu Huang, Zan Gojcic, Mikhail Usvyatsov, Andreas Wieser, and Konrad Schindler. 2021. PREDATOR: Registration of 3D Point Clouds with Low Overlap. In 2021 IEEE/CVF Conference on Computer Vision and Patern Recognition (CVPR). IEEE, 4265–4274. doi:10. 1109/CVPR46437.2021.00425

[11] Anestis Koutsoudis, Blaž Vidmar, George Ioannakis, Fotis Arnaoutoglou, George Pavlidis, and Christodoulos Chamzas. 2014. Multi-Image 3D Reconstruction Data Evaluation. Journal ofCultural Heritage 15, 1 (2014), 73–79. doi:10.1016/j.culher.2012.12.003

[12] Marc Levoy, Kari Pulli, Brian Curless, Szymon Rusinkiewicz, David Koller, Lucas Pereira, Mat Ginzton, Sean Anderson, James Davis, Jeremy Ginsberg,Jonathan Shade, and Duane Fulk. 2000. The Digital Michelangelo Project: 3D Scanning ofLarge Statues. In Proceedings ofthe 27th Annual Conference on Computer Graphics and Interactive Techniques. ACM Press/Addison-Wesley Publishing Co., 131–144. doi:10.1145/344779.344849

[13] Thomas Luhmann, Stuart Robson, Stephen Kyle, and Jan Boehm. 2023. Close-Range Photogrammetry and 3D Imaging (4 ed.). De Gruyter, Berlin, Boston. doi:10.1515/9783111029672

[14] Nicolas Mellado, Dror Aiger, and Niloy J. Mitra. 2014. Super 4PCS: Fast Global Pointcloud Registration via Smart Indexing. Computer Graphics Forum 33, 5 (2014), 205–215. doi:10.1111/cgf.12446

[15] Andriy Myronenko and Xubo Song. 2010. Point Set Registration: Coherent Point Drift. IEEE Transactions on Patern Analysis and Machine Intelligence 32, 12 (2010), 2262–2275. doi:10.1109/TPAMI.2010.46

[16] François Pomerleau, Francis Colas, and Roland Siegwart. 2015. A Review of Point Cloud Registration Algorithms for Mobile Robotics. Foundations and Trends in Robotics 4, 1 (2015), 1–104. doi:10.1561/2300000035

[17] François Pomerleau, Francis Colas, Roland Siegwart, and Stéphane Magnenat. 2013. Comparing ICP Variants on Real-World Data Sets: Open-Source Library and Experimental Protocol. Autonomous Robots 34, 3 (2013), 133–148. doi:10.1007/s10514-013-9327-2

[18] Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing Peng, and Kai Xu. 2022. Geometric Transformer for Fast and Robust Point Cloud Registration. In 2022 IEEE/CVF Conference on Computer Vision and Patern Recognition (CVPR). IEEE, 11133–11142. doi:10.1109/ CVPR52688.2022.01086

[19] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. 2025. SAM 2: Segment Anything in Images and Videos. In The Thirteenth International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=Ha6RTeWMd0

[20] Fabio Remondino. 2011. Heritage Recording and 3D Modeling with Photogrammetry and 3D Scanning. Remote Sensing 3, 6 (2011), 1104–1138. doi:10.3390/rs3061104

[21] Szymon Rusinkiewicz and Marc Levoy. 2001. Eficient Variants of the ICP Algorithm. In Proceedings of the Third International Conference on 3-D Digital Imaging and Modeling. IEEE, 145–152. doi:10.1109/IM.2001.924423

[22] Radu Bogdan Rusu, Nico Blodow, and Michael Beetz. 2009. Fast Point Feature Histograms (FPFH) for 3D Registration. In 2009 IEEE International Conference on Robotics and Automation. IEEE, 3212–3217. doi:10.1109/ROBOT.2009.5152473

[23] Samuele Salti, Federico Tombari, and Luigi Di Stefano. 2014. SHOT: Unique Signatures of Histograms for Surface and Texture Descrip tion. Computer Vision and Image Understanding 125 (2014), 251–264. doi:10.1016/j.cviu.2014.04.011

[24] Joaquim Salvi, Carles Matabosch, David Fofi, and Josep Forest. 2007. A Review of Recent Range Image Registration Methods with Accuracy Evaluation. Image and Vision Computing 25, 5 (2007), 578–596. doi:10.1016/j.imavis.2006.05.012

[25] Johannes L. Schönberger and Jan-Michael Frahm. 2016. Structure-from-Motion Revisited. In 2016 IEEE Conference on Computer Vision and Patern Recognition (CVPR). IEEE, 4104–4113. doi:10.1109/CVPR.2016.445

[26] Johannes L. Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. 2016. Pixelwise View Selection for Unstructured Multi-View Stereo. In Computer Vision – ECCV2016 (Lecture Notes in Computer Science, Vol. 9907), Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling (Eds.). Springer International Publishing, Cham, 501–518. doi:10.1007/978-3-319-46487-9\_31

[27] Aleksandr V. Segal, Dirk Haehnel, and Sebastian Thrun. 2009. Generalized-ICP. In Proceedings of Robotics: Science and Systems V. Robotics: Science and Systems Foundation, Seatle, Washington, USA. doi:10.15607/RSS.2009.V.021

[28] Noah Snavely, Steven M. Seitz, and Richard Szeliski. 2006. Photo Tourism: Exploring Photo Collections in 3D. ACM Transactions on Graphics 25, 3 (2006), 835–846. doi:10.1145/1141911.1141964

[29] Federico Tombari, Samuele Salti, and Luigi Di Stefano. 2010. Unique Shape Context for 3D Data Description. In Proceedings ofthe ACM Workshop on 3D Object Retrieval. Association for Computing Machinery, 57–62. doi:10.1145/1877808.1877821

[30] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. 2004. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing 13, 4 (2004), 600–612. doi:10.1109/TIP.2003.819861

[31] Changchang Wu. 2013. Towards Linear-Time Incremental Structure from Motion. In 2013 International Conference on 3D Vision (3DV). IEEE, 127–134. doi:10.1109/3DV.2013.25

[32] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Efectiveness of Deep Features as a Perceptual Metric. In 2018 IEEE/CVF Conference on Computer Vision and Patern Recognition. IEEE, 586–595. doi:10.1109/CVPR. 2018.00068

[33] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. 2016. Fast Global Registration. In Computer Vision – ECCV 2016 (Lecture Notes in Computer Science, Vol. 9906), Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling (Eds.). Springer International Publishing, Cham, 766–782. doi:10.1007/978-3-319-46475-6\_47