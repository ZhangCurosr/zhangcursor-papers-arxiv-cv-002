# LoomSC: Scalable Deep Subspace Clustering with Projector Factorization and Exact Spectral Reduction

Nairouz Mrabah, Youssef Melki, Mohamed Bouguessa, Riadh Ksantini, Shakeeb Murtaza, and Tehseen Zia

Abstract—Dense self-expression matrices and full-affinity spectral clustering limit the scalability of subspace clustering. We introduce the Latent Orthogonal Optimization Model for Subspace Clustering (LoomSC), a framework that addresses both bottlenecks through projector factorization and exact spectral reduction. Motivated by the spectral structure of leastsquares regression, LoomSC jointly learns latent features and a projector self-representation through two thin factors. Alternating Procrustes and least-squares updates preserve the sample factor’s orthogonality while keeping the coefficient matrix implicit. We construct a nonnegative quadratic affinity that preserves the projector’s support. An exact feature map then reduces its normalized spectral problem to an eigenproblem whose dimension depends only on the factor width. Neither the full affinity nor the sample Laplacian needs to be formed. Our analysis quantifies the projector approximation and identifies conditions for subspace preservation and within-subspace connectivity. For fixed dimensions and iteration budgets, the complete pipeline has linear time and memory complexity in the number of samples. Across five image-clustering benchmarks, LoomSC ranks first or second in all 15 dataset–metric comparisons against 9 state-ofthe-art baselines. Its mean accuracy exceeds the highest baseline mean by 6.66 percentage points. Synthetic experiments scale to 500,000 samples while maintaining at least 99.8% accuracy.

## I. INTRODUCTION

In several cases, high-dimensional data can be modeled by a generation process from the union of multiple low-dimensional linear subspaces. In this context, subspace clustering is a prominent unsupervised strategy that aims to partition the data into clusters by identifying these subspaces, where each cluster is defined by exactly one low-dimensional subspace. The subspace assumption is widely used in visual analysis and has been successfully exploited in several applications, such as image segmentation [1], motion segmentation [2], [3], and face clustering [3], [4]. For example, face images of a person can be represented by a low-dimensional subspace for a fixed pose and under Lambertian reflectance [5].

The most recent subspace clustering methods [6]–[8] follow a two-step strategy. First, these methods learn an affinity matrix that captures the similarity between points within the same subspace. Second, a graph clustering approach, such as spectral clustering, is used to identify the clusters. The affinity matrix is constructed according to the self-expression property [8], [9] of the subspaces. More precisely, each data point can be expressed as a linear combination of points in the same subspace. Given a dataset $X \in \mathbb { R } ^ { d _ { x } \times n }$ with n data points of dimensionality $d _ { x } ,$ the optimization problem associated with this strategy is:

$$
\operatorname* { m i n } _ { c } \frac { 1 } { 2 } \left\| X - X C \right\| _ { F } ^ { 2 } + \lambda \theta ( C ) ,\tag{1}
$$

where $C = ( c _ { i j } ) \in \mathbb { R } ^ { n \times n }$ is a matrix that captures the selfexpression property based on the first term; $\left\| . \right\| _ { F }$ denotes the Frobenius norm, which is defined for any matrix $M = ( m _ { i j } )$ by $\begin{array} { r } { \| M \| _ { F } = ( \sum _ { i = 1 } \sum _ { j = 1 } m _ { i j } ^ { 2 } ) ^ { \frac { 1 } { 2 } } } \end{array}$ ; the second term, controlled by the function $\theta ,$ serves as a regularization to avoid trivial solutions (e.g., $C = I )$ and to adjust the structure of the selfrepresentation matrix $C ; \lambda \in \mathbb { R } _ { + } ^ { * }$ is a balancing hyperparameter between the two terms. Based on the self-representation matrix, the affinity graph is constructed by computing its adjacency matrix $A _ { \mathrm { a b s } } = ( a _ { i j } ^ { \mathrm { a b s } } ) \in \mathbb { R } ^ { n \times n }$ , such that $\begin{array} { r } { \mathbf { \bar { \rho } } _ { A _ { \mathrm { a b s } } } = \frac { | C | + \mathbf { \bar { | } } C ^ { T } | } { 2 } } \end{array}$

Different regularizers encode complementary structural preferences. Sparse Subspace Clustering (SSC) [8] promotes sparse coefficients through an $\ell _ { 1 }$ penalty. Low-Rank Representation (LRR) [10] favors a low-rank coefficient matrix through nuclear-norm regularization. Multi-Subspace Representation (MSR) [11] combines these penalties, whereas Block Diagonal Representation (BDR) [6] directly encourages a block structure. Least Squares Regression (LSR) [9] uses squared Frobenius regularization and admits a closed-form solution. Its grouping effect encourages correlated samples to receive similar coefficients. These formulations seek to suppress cross-subspace connections while retaining informative relations within each subspace. Both requirements matter because a sparse affinity can separate samples from different subspaces yet fragment samples from the same subspace [12].

The self-expression framework nevertheless introduces two distinct computational bottlenecks. Explicitly storing a dense coefficient matrix requires $\scriptstyle { \mathcal { O } } ( n ^ { 2 } )$ memory, even when its estimation admits an efficient solver. Constructing a full affinity retains this quadratic storage cost. The spectral stage adds a second bottleneck. A dense eigendecomposition requires $\mathcal { O } ( n ^ { 3 } )$ operations, while partial eigensolvers still perform repeated products with the sample-level affinity. Consequently, accelerating coefficient estimation alone does not ensure a scalable clustering pipeline. Both self-expression and spectral assignment must avoid dense sample-by-sample computations.

Several strategies reduce these costs. Sampling methods replace the full dictionary with a smaller set of representatives [13]–[16]. Subspace-preserving selection seeks representatives that span the underlying subspaces [17]. Sparse solvers accelerate coefficient estimation through greedy pursuit [18], active-set selection [19], or stochastic dictionaries [20].

Structure-aware methods exploit compact graph models [21] or structured affinity estimation [22]. Neural coefficient predictors [23] provide another route by learning a mapping that replaces repeated regression.

Representative-based reduction is particularly attractive because it replaces an all-sample dictionary with $m \ll n$ representatives. Each sample is then described by m coefficients instead of $n .$ Anchor-graph methods can also obtain the spectral embedding from a compact factor [15], [16]. Their effectiveness depends on how well the representatives capture the subspaces in the supplied feature space. Learning the coefficients improves the graph, but does not by itself adapt the image representation. This motivates a compact model in which both the representation and the directions used for self-expression are learned from the data.

The representation itself is important because complex appearance variation and data corruption can weaken the linear subspace structure of raw observations. Kernel methods address nonlinear structure through an implicit feature mapping [24]. Deep subspace clustering [25]–[27] instead couples an autoencoder with self-expression in the latent space. The reconstruc tion objective preserves information about the inputs. The selfexpression objective encourages latent features that conform to a subspace model. Joint optimization therefore adapts the features to clustering rather than imposing self-expression on a fixed representation.

Joint learning also makes computational compactness essential. A dense self-expression layer introduces $n ^ { 2 }$ sampledependent coefficients that must be optimized together with the encoder [25], [26]. Neural parameterizations and minibatch training can reduce training costs [28], but do not by themselves eliminate full-affinity spectral inference. Our goal is to retain both pairwise self-expression and normalized spectral clustering while learning nonlinear representations, without constructing dense sample-level matrices at either stage.

To this end, we introduce the Latent Orthogonal Optimization Model for Subspace Clustering (LoomSC). Motivated by the spectral structure of LSR, LoomSC represents self-expression through a projector $C = P P ^ { \top }$ . The sample factor $P \in \mathbb { R } ^ { n \times m }$ has orthonormal columns and a prescribed width $m$ . An equivalent two-factor objective separates latent directions from sample coefficients and couples both to the auto-encoder. Alternating Procrustes and least-squares updates refine these factors while preserving orthogonality. Neither the coefficient matrix nor a sample Gram matrix needs to be formed. The latent directions are updated using all samples rather than remaining fixed to selected representatives. Our analysis quantifies the projector approximation to LSR and identifies conditions under which the full row-space projector preserves subspace memberships and within-subspace connectivity.

LoomSC extends this compact representation to spectral assignment through a nonnegative quadratic affinity. Squaring the projector coefficients preserves their zero pattern and thus retains the connections encoded by self-expression. An exact feature map represents this affinity using $q = m ( m + 1 ) / 2$ features per sample. Its normalized spectral problem then reduces to a q-dimensional eigenproblem. This reduction avoids forming either the full affinity or the sample Laplacian. For fixed dimensions and iteration budgets, the complete pipeline has linear time and memory complexity in n. The same compact representation therefore supports feature learning, selfexpression, and global spectral assignment.

Contributions. (1) We introduce a two-factor projector model for joint feature and self-expression learning with orthogonality-preserving updates. (2) We construct a supportpreserving nonnegative affinity and derive an exact reduction of its normalized spectral problem without forming full pairwise matrices. (3) We quantify the projector approximation to LSR and establish conditions for subspace preservation and connectivity. We also prove linear time and memory bounds for the complete pipeline under fixed dimensions and iteration budgets. (4) Across five image-clustering benchmarks, LoomSC ranks first or second in all 15 dataset–metric comparisons against nine baselines. Its mean accuracy exceeds the highest baseline mean by 6.66 percentage points. Experiments on synthetic data reach 500,000 samples while maintaining at least 99.8% accuracy throughout the tested range.

## II. RELATED WORK

In line with the focus of this work, we discuss the advantages and limitations of two existing subspace clustering categories: the scalable subspace clustering category and the deep subspace clustering category. In that respect, we highlight the pertinence of introducing a subspace clustering approach that combines the advantages of both categories.

## A. Scalable subspace clustering:

Least-squares regression (LSR) [9] provides an efficient alternative to sparse coefficient estimation. Its squared Frobenius regularization encourages a grouping effect among correlated samples and admits a closed-form solution. However, solving the regression analytically does not eliminate the dense $n \times n$ coefficient matrix, where $n$ is the number of samples. Its explicit construction and storage remain quadratic in n. Thus, improving the regression solver alone does not resolve the full-affinity bottleneck. LoomSC instead adopts a prescribedwidth projector model and learns its thin factors directly. This changes the representation of self-expression rather than merely accelerating a dense coefficient solve.

Sparse methods reduce storage by retaining a small number of coefficients per sample. SSC-OMP [18] replaces $\ell _ { 1 }$ optimization with orthogonal matching pursuit and establishes subspacepreserving guarantees under suitable conditions. EnSC [19] combines $\ell _ { 1 }$ and squared $\ell _ { 2 }$ regularization to balance subspace preservation and graph connectivity. Its oracle-guided active-set solver restricts optimization to candidate dictionary elements. Both approaches substantially accelerate coefficient estimation. Nevertheless, their dictionary searches still compare target residuals with the full dataset. Repeating these searches across samples retains a quadratic sample-size dependence for fixed sparsity and search budgets. Sparse spectral solvers reduce the subsequent graph-processing cost, but do not remove this coefficient-estimation cost. LoomSC avoids these separate fulldictionary searches by optimizing a shared factorization through thin matrix products and an orthogonal Procrustes update.

Stochastic sparse subspace clustering (SSSC) [20] addresses weak within-subspace connectivity through dictionary dropout. For normalized samples, the expected dropout objective induces an additional squared $\ell _ { 2 }$ penalty, which encourages denser representations. Its scalable implementation combines coefficients from multiple reduced dictionaries through a consensus formulation. This improves connectivity while preserving a sparse affinity. However, with a fixed retained fraction, each dictionary still grows with n, and the reduced regression must be repeated for every target. Parallel execution accelerates these subproblems without changing their aggregate arithmetic complexity. LoomSC obtains compactness through a common learned latent factor rather than repeated stochastic dictionary reduction. The resulting sample relations are represented jointly without an ensemble of coefficient estimates.

Representative-based approaches reduce the dictionary size more directly. LMVSC [15] learns nonnegative sample-toanchor coefficients and computes its spectral embedding from a compact normalized factor. SGL [16] further couples anchor graph learning with a spectral constraint that targets k connected components. These methods achieve linear sample-size scaling for fixed anchor and optimization budgets. Their graph weights are learned, but their reconstruction is performed using supplied features and an anchor dictionary. Consequently, they do not jointly adapt the image representation to the self-expression objective. LoomSC preserves a compact representation while learning both the encoder and the latent factor. After initialization, the columns of the latent factor are updated from all samples rather than remaining fixed anchor representatives. Unlike the nonnegative anchor weights in LMVSC and SGL, its sample-factor coefficients can be signed. This permits linear combinations beyond convex anchor assignments.

Selective sampling-based scalable sparse subspace clustering $( \mathrm { { } S ^ { 5 } C ) }$ [29] selects representatives using stochastic estimates of subgradient violations. It then solves restricted sparse regressions for all samples. Its spectral stage exploits a graph with only $O ( n m )$ nonzero entries, where m is the representative budget. Thus, $\mathrm { \bf S ^ { 5 } C }$ reduces both regression and spectral costs without requiring a dense affinity. However, its representation remains tied to selected samples in the feature space. LoomSC replaces this exemplar dictionary with adaptive latent directions and jointly refines the features used to construct them.

A-DSSC [22] addresses a different source of difficulty: the compatibility between self-expression coefficients and spectral normalization. It first estimates coefficients and then obtains a doubly stochastic affinity through quadratically regularized optimal transport. A support-restricted solver exploits sparsity to accelerate this second stage. This gives a principled alterna tive to heuristic coefficient normalization, but retains separate coefficient-estimation and transport problems. LoomSC instead derives the affinity and its normalization from the same learned projector. For its squared affinity, each degree equals the squared norm of the corresponding sample-factor row. Thus, degree computation requires only the thin factor and no additional transport optimization.

Compact spectral computation also appears in scalable attributed-graph subspace clustering (SAGSC) [30]. SAGSC constructs a projector from graph-smoothed features and uses a quadratic kernel to obtain a nonnegative factorized affinity. This permits spectral clustering without explicitly assembling pairwise weights. LoomSC applies this spectral principle to jointly learned image representations rather than fixed graphsmoothed features. Its two-factor objective allows the projector to evolve with the encoder, and its analysis connects the learned factors to dominant latent directions and subspace structure. The distinction is therefore the integration of adaptive representation learning with a compact spectral construction, rather than factorization alone.

## B. Deep subspace clustering:

DSCNet [25] integrates a differentiable self-expression layer into a convolutional auto-encoder. This couples feature learning with the reconstruction of each latent sample from the others. S<sup>2</sup>ConvSCN [26] additionally uses spectral assignments to supervise feature learning and coefficient estimation. These formulations make the representation responsive to clustering, but retain a sample-to-sample self-expression matrix. Additional supervision does not remove its quadratic parameter storage. LoomSC retains joint feature and self-expression learning while replacing this layer with two thin factors. Its factor updates admit closed-form solutions, and their dimensions are controlled by the prescribed factor width.

Neural coefficient prediction offers another route to scalability. SENet [23] parameterizes pairwise coefficients with a neural network that can generalize to unseen samples. Its objective learns a coefficient-prediction rule for the supplied representation rather than jointly learning a new feature space for self-expression. This separates the number of network parameters from the number of coefficient entries. However, a compact predictor does not itself provide a compact spectral representation of the resulting affinity. LoomSC instead learns a sample factor whose algebraic structure supports both selfexpression and the subsequent spectral reduction.

PRO-DSC [28] jointly learns latent representations and selfexpression coefficients. A log-determinant regularizer controls representation collapse under the conditions established in its analysis. Its implementation predicts coefficients through neural mappings and Sinkhorn normalization, which supports mini-batch training. The distinction from LoomSC arises at full-set clustering. PRO-DSC constructs the coefficient matrix over the clustering set and applies spectral clustering to the resulting affinity. For n samples, this inference route retains quadratic matrix storage despite efficient training. LoomSC does not reconstruct a full coefficient matrix after learning. The final sample factor directly defines a nonnegative affinity whose normalized spectral problem is solved in a feature space determined by the factor width. Thus, the computational reduction extends through spectral assignment rather than ending with network training.

Other deep methods reduce cost by changing how the final partition is obtained. The large-scale extension of pseudosupervised deep subspace clustering [27] trains on a subset and assigns the remaining latent samples through nearest-neighbor classification. LoomSC instead retains an explicit model of sample relations while avoiding their dense storage. All samples participate in factor learning, and the final partition is obtained by spectral clustering of the learned affinity rather than by extending labels from a training subset. Its efficiency follows from representing that affinity compactly while preserving the self-expression formulation.

## III. LOOMSC

The proposed method develops a compact self-representation by exploiting the spectral structure of least-squares regression (LSR). We first analyze the LSR solution and motivate a projector approximation. We then express this approximation through two thin factors that are learned jointly with a deep auto-encoder. Finally, we construct a nonnegative affinity from the learned factors and obtain the spectral embedding through a reduced eigenproblem. Both phases operate directly on the factors, which makes their computational cost linear.

Let the matrix $X = [ x _ { 1 } , \ldots , x _ { n } ] \in \mathbb { R } ^ { d _ { x } \times n }$ contain the input samples $x _ { i } \in \mathbb { R } ^ { d _ { x } }$ , and let k denote the number of clusters.

## A. A Projector Approximation Motivated by LSR

LSR [9] models each sample through a linear combination of the data collection and penalizes the energy of the coefficients. For the input X, its diagonal-unconstrained formulation is

$$
\operatorname* { m i n } _ { C \in \mathbb { R } ^ { n \times n } } \ \frac { 1 } { 2 } \| X - X C \| _ { F } ^ { 2 } + \lambda \| C \| _ { F } ^ { 2 } , \qquad \lambda > 0 .\tag{2}
$$

The reconstruction term encourages self-expression. The Frobenius penalty controls coefficient magnitudes while allowing correlated samples to contribute jointly. Differentiation with respect to $C$ gives the closed-form solution of this problem

$$
\begin{array} { r } { C _ { \mathrm { L S R } } ^ { \star } = ( X ^ { \top } X + 2 \lambda I _ { n } ) ^ { - 1 } X ^ { \top } X , } \end{array}\tag{3}
$$

where $I _ { n }$ is the $n \times n$ identity matrix. A direct evaluation of Eq. (3) requires forming the Gram matrix $X ^ { \top } X$ , at a cost of $O ( d _ { x } n ^ { 2 } )$ , followed by the solution of an $n \times n$ linear system, which requires $O ( n ^ { 3 } )$ operations using a dense factorization. Moreover, the resulting coefficient matrix contains $n ^ { 2 }$ entries and therefore requires $O ( n ^ { 2 } )$ memory. Although the inversion cost can be reduced in particular dimensional regimes, explicitly constructing the dense self-representation remains quadratic in the number of samples. Thereofre, we examine the spectral structure of $C _ { \mathrm { L S R } } ^ { \star }$ to derive a compact representation.

Let $r \ = \ \operatorname { r a n k } ( X )$ . Since $X ^ { \top } X$ is symmetric positive semidefinite with rank r, its compact eigendecomposition is

$$
\begin{array} { r } { X ^ { \top } X = V _ { x } \mathrm { d i a g } ( \beta _ { 1 } , . . . , \beta _ { r } ) V _ { x } ^ { \top } , } \end{array}\tag{4}
$$

where $V _ { x } \in \mathbb { R } ^ { n \times r }$ has orthonormal columns given by the eigenvectors of $X ^ { \top } X$ associated with its positive eigenvalues, and $\beta _ { 1 } \geq \cdot \cdot \cdot \geq \beta _ { r } > 0$ are the corresponding eigenvalues.

Proposition 1 (Spectral structure of LSR). For $\lambda > 0 ,$ , the LSR solution is symmetric and positive semidefinite, with rank r. Its nonzero eigenvalues are

$$
\tau _ { j } = \frac { \beta _ { j } } { \beta _ { j } + 2 \lambda } , \qquad j = \{ 1 , \dots , r \} ,\tag{5}
$$

and $C _ { \mathrm { L S R } } ^ { \star } = V _ { x }$ diag(τ<sub>1</sub>, . . . , τ<sub>r</sub>)V<sup>⊤</sup><sub>x</sub> .

The eigenvectors identify the sample relationships represented by LSR. Eq. (5) also shows that directions with $\beta _ { j }$ large relative to 2λ have eigenvalues close to one. This observation motivates replacing their spectral weights by one and retaining the corresponding subspace. For the full row space, define $C _ { \mathrm { r o w } } = V _ { x } V _ { x } ^ { \top } \in \mathbb { R } ^ { n \times n }$ . The approximation error is

$$
\| C _ { \mathrm { L S R } } ^ { \star } - C _ { \mathrm { r o w } } \| _ { F } ^ { 2 } = \sum _ { j = 1 } ^ { r } \left( \frac { 2 \lambda } { \beta _ { j } + 2 \lambda } \right) ^ { 2 } .\tag{6}
$$

Thus, the projector approaches the LSR representation as the regularization becomes small relative to the positive data spectrum. This gives a direct spectral justification for an idempotent self-representation.

To obtain a compact model, choose a factor width m with $k \leq m < n$ . When $m \leq r ,$ let $V _ { x , m } \in \mathbb { R } ^ { n \times m }$ contain the first m columns of $V _ { x } .$ . The corresponding approximation satisfies

$$
\| C _ { \mathrm { L S R } } ^ { \star } - V _ { x , m } V _ { x , m } ^ { \top } \| _ { F } ^ { 2 } = \sum _ { j = 1 } ^ { m } ( 1 - \tau _ { j } ) ^ { 2 } + \sum _ { j = m + 1 } ^ { r } \tau _ { j } ^ { 2 } .\tag{7}
$$

For the retained directions $j ~ \leq ~ m ,$ the approximation replaces the LSR eigenvalue $\tau _ { j }$ by the projector eigenvalue 1. The first term in Eq. (7) measures the resulting error. The remaining directions $j > m$ are discarded, and the second term measures the corresponding loss of the LSR spectral components. Since $\tau _ { j }$ is monotonically increasing with $\beta _ { j }$ the ordering of the $\tau _ { j } \mathrm { ' s }$ is identical to that of the positive eigenvalues of $X ^ { \top } X$ . Hence, for a fixed width m, retaining the first m eigenvectors gives the closest projector among the rank-m projectors formed from the eigenvectors of $X ^ { \top } X$

Setting $\textit { m } = \textit { r }$ recovers the full row-space projector $C _ { \mathrm { r o w } } = V _ { x } V _ { x } ^ { \top }$ . However, constructing this projector requires determining r and computing all r nonzero right singular vectors of X. For a dense $X \in \mathbb { R } ^ { d _ { x } \times n }$ , computing the full singular subspace costs $O ( d _ { x } n \operatorname* { m i n } \{ d _ { x } , n \} )$ ), while storing $V _ { x }$ requires $O ( n r )$ memory. These costs become prohibitive when r grows with the data size. Thus, LoomSC does not attempt to recover the complete rank-r projector. Instead, m is treated as a prescribed model width, and the corresponding thin factors are learned directly in the subsequent formulation.

Accordingly, we parameterize the self-representation as

$$
C = P P ^ { \top } , \qquad P \in \mathbb { R } ^ { n \times m } , \qquad P ^ { \top } P = I _ { m } .\tag{8}
$$

The factor $P$ stores the relationships between all samples through nm coefficients. Its orthogonality makes C symmetric, positive semidefinite, and idempotent, with rank m. It also fixes the coefficient energy:

$$
\| P P ^ { \top } \| _ { F } ^ { 2 } = \mathrm { t r } ( P ^ { \top } P ) = m .\tag{9}
$$

Substituting Eq. (8) into the LSR objective in Eq. (2) yields the following optimization over fixed-width projectors:

$$
\operatorname* { m i n } _ { P ^ { \top } P = I _ { m } } ~ \| X - X P P ^ { \top } \| _ { F } ^ { 2 } .\tag{10}
$$

Since $P P ^ { \top }$ is an orthogonal projector, Eq. (10) is equivalent to maximizing $\operatorname { t r } ( P ^ { \top } X ^ { \top } X P )$ . Hence, for $m \leq r ,$ an optimal solution is $P ^ { \star } = V _ { x , m } R .$ , where $R \in \mathbb { R } ^ { m \times m }$ is any

orthogonal matrix. All such solutions span the same dominant m-dimensional subspace and yield

$$
P ^ { \star } P ^ { \star \top } = V _ { x , m } V _ { x , m } ^ { \top } .\tag{11}
$$

The minimum reconstruction error is $\textstyle \sum _ { j = m + 1 } ^ { r } \beta _ { j }$ . Thus, the exact solution requires computing the dominant right singular subspace of the current representation. This is practical when the representation is fixed, but it becomes less attractive when the features are learned jointly, since the dominant subspace must be recomputed as the representation evolves. To address this problem, LoomSC replaces the repeated spectral solution with a two-factor formulation. The resulting updates involve only thin matrix products and an $n \times m$ Procrustes problem, while the full coefficient matrix C remains implicit.

## B. Learning a Latent Representation

We apply the compact projector model to a learned latent space. This lets the self-expression structure guide feature extraction. We use an encoder E and a decoder D with parameter collections $W$ and $\widehat { W }$ , respectively. They define

$$
\begin{array} { l } { Z = \mathcal { E } ( X ; W ) \in \mathbb R ^ { d _ { z } \times n } , } \\ { \widehat { X } = \mathcal { D } ( Z ; \widehat { W } ) \in \mathbb R ^ { d _ { x } \times n } , } \end{array}\tag{12}
$$

where $d _ { z }$ is the latent dimension and $z _ { i } ~ \in ~ \mathbb { R } ^ { d _ { z } }$ is the representation of $x _ { i }$ . The vector $\widehat { x } _ { i } \in \mathbb { R } ^ { d _ { x } }$ is the $i ^ { \mathrm { { t h } } }$ column of the matrix $\widehat { X }$ . For image inputs, convolutional layers extract spatial features, which are flattened to form the columns of the latent representation matrix Z. The decoder reconstructs the input from these features. Its reconstruction objective is

$$
J _ { \mathrm { r e c } } ( W , \widehat { W } ) = \frac { 1 } { n } \| X - \widehat { X } \| _ { F } ^ { 2 } .\tag{13}
$$

This objective trains the representation to retain information about the observed samples. The projector objective in Eq. (10), applied to Z, imposes a complementary subspace structure.

## C. Self-Expression Through Two Thin Factors

To optimize projector self-expression in the latent space, we introduce a latent factor $L \in \mathbb { R } ^ { d _ { z } \times m }$ . Its columns represent learned directions in the latent space. The following decomposition connects the resulting two-factor model to projector self-expression:

$$
\lVert Z - L P ^ { \top } \rVert _ { F } ^ { 2 } = \lVert Z - Z P P ^ { \top } \rVert _ { F } ^ { 2 } + \lVert L - Z P \rVert _ { F } ^ { 2 } .\tag{14}
$$

The identity in Eq. (14) holds for every P satisfying $P ^ { \top } P =$ $I _ { m }$ . The proof is provided in Appendix B. Minimizing its left hand side with respect to L gives $L = Z P$ . Hence, we obtain:

$$
\operatorname* { m i n } _ { \substack { L , P ^ { \top } P = I _ { m } } } \Vert Z - L P ^ { \top } \Vert _ { F } ^ { 2 } = \operatorname* { m i n } _ { \substack { P ^ { \top } P = I _ { m } } } \Vert Z - Z P P ^ { \top } \Vert _ { F } ^ { 2 } .\tag{15}
$$

The two-factor formulation separates the latent directions represented by L from the sample coefficients encoded by $P .$ This separation gives simple block updates and requires only products involving thin matrices.

For fixed Z, let $r _ { z } = \mathrm { r a n k } ( Z )$ and write $Z = U _ { z } \Sigma _ { z } V _ { z } ^ { \top }$ The matrices $U _ { z } \in \mathbb { R } ^ { d _ { z } \times r _ { z } }$ and $V _ { z } \in \mathbb { R } ^ { n \times r _ { z } }$ have orthonormal columns. The diagonal matrix $\begin{array} { r } { \Sigma _ { z } \ \in \ \mathbb { R } ^ { r _ { z } \times r _ { z } } } \end{array}$ <sup>z</sup> contains the positive singular values $\sigma _ { 1 } \geq \cdot \cdot \cdot \geq \sigma _ { r _ { z } } > 0$

Proposition 2 (Reconstruction through dominant directions). For a fixed latent matrix $Z ,$ the optimal factor residual is

$$
\operatorname* { m i n } _ { \substack { L , P ^ { \top } P = I _ { m } } } \| Z - L P ^ { \top } \| _ { F } ^ { 2 } = \sum _ { j = m + 1 } ^ { r _ { z } } \sigma _ { j } ^ { 2 } ,\tag{16}
$$

where an empty sum is zero. For $m \leq r _ { z } ,$ , a minimizing $P$ spans an m-dimensional dominant right singular subspace of $Z .$ For $m \geq r _ { z } ,$ , an optimal projector contains the row space of Z and satisfies $Z { \bar { P } } P ^ { \top } = { \bar { Z } } .$ In particular, $m = r _ { z }$ gives $C = V _ { z } V _ { z } ^ { \top }$

Proposition 2 connects the learned factors to the dominant spectral directions of Z. Thus, the spectral approximation developed above applies directly to the learned representation. It also quantifies how the width controls the amount of latent information represented by the model. The proofs of (14) and Proposition 2 are given in Appendix B.

We couple the factor objective with the auto-encoder to learn the latent features and their self-representation jointly:

$$
\operatorname* { m i n } _ { W , \widehat { W } , L , P } \quad J ( W , \widehat { W } , L , P ) = J _ { \mathrm { r e c } } ( W , \widehat { W } ) + \| Z - L P ^ { \top } \| _ { F } ^ { 2 } ,
$$

$$
{ \mathrm { s u b j e c t ~ t o ~ } } P ^ { \top } P = I _ { m } .\tag{17}
$$

The reconstruction term retains information about the input samples. The factor term encourages the latent codes to admit a common compact self-representation. Joint optimization allows the encoder to adjust this representation as the factors are refined. For sample i, the row coefficients $p _ { i } = P _ { i : } ^ { \top } \in \mathbb { R } ^ { m }$ reconstruct its latent vector as $L p _ { i }$

## D. Alternating Optimization

We first pretrain the auto-encoder with (13). We then initialize the columns of L using m latent samples selected by k-means++ seeding [31]. This spreads the initial directions across the observed latent features. Appendix E specifies the sampling rule. The factor P is initialized by its update below. Each joint iteration updates the network parameters, refreshes $Z ,$ and then updates P and L in sequence.

a) Network parameters: With L and P fixed, an Adam step [32] updates the encoder and decoder:

$$
( W , \widehat { W } ) \longleftarrow \mathrm { A d a m } _ { t } \Big ( ( W , \widehat { W } ) , \nabla _ { W , \widehat { W } } J \Big ) ,\tag{18}
$$

where t is the joint iteration index. The factor term contributes the latent gradient $2 ( Z - L P ^ { \top } )$ , which encourages the encoder to produce representations consistent with the current factors. To control activation memory, the samples can be processed in chunks $\{ B _ { b } \}$ while keeping L and P fixed:

$$
J = \sum _ { b } \sum _ { i \in \mathcal { B } _ { b } } \left( \frac { 1 } { n } \| x _ { i } - \widehat { x } _ { i } \| _ { 2 } ^ { 2 } + \| z _ { i } - L p _ { i } \| _ { 2 } ^ { 2 } \right) .\tag{19}
$$

The gradients are accumulated over all chunks before a single update of $( W , \widehat { W } )$ . Hence, chunking reduces memory usage without changing the full-data objective. After the update, the encoder recomputes Z before updating P and L.

Algorithm 1 LoomSC   
Require: $X \in \mathbb { R } ^ { d _ { x } \times n } , k \leq m < n , T _ { \mathrm { p r e } } , T .$   
Ensure: Cluster labels $\gamma _ { 1 } , \ldots , \gamma _ { n } .$   
1: Pretrain $( W , \widehat { W } )$ for $T _ { \mathrm { p r e } }$ updates using (13).   
2: Compute $Z$ using (12).   
3: Initialize L with m sample columns selected by applying   
k-means++ to $Z .$   
4: Initialize $P$ using (21).   
5: for $t = 1 , \dots , T$ do   
6: Fix L and $P$ and accumulate the gradient using (19).   
7: Update $( W , \widehat { W } )$ using (18).   
8: Recompute $Z$ using (12).   
9: Update $P$ using the thin-SVD step in (21).   
10: Update $L \gets Z P$ using (22).   
11: end for   
12: Obtain the cluster labels from Algorithm 2 using the final   
P and $Z .$   
13: return $\gamma _ { 1 } , \ldots , \gamma _ { n } .$

b) Sample factor: For fixed Z and $L ,$ orthogonality gives

$$
\| Z - L P ^ { \top } \| _ { F } ^ { 2 } = \| Z \| _ { F } ^ { 2 } + \| L \| _ { F } ^ { 2 } - 2 \operatorname { t r } ( P ^ { \top } Z ^ { \top } L ) .\tag{20}
$$

Thus, updating P amounts to maximizing the trace term under $P ^ { \top } P = I _ { m }$ . This is an orthogonal Procrustes problem. To solve it, we form the thin product $Z ^ { \top } L \in \mathbb { R } ^ { n \times m }$ and compute

$$
\begin{array} { r } { Z ^ { \top } L = U _ { P } \Sigma _ { P } V _ { P } ^ { \top } , \qquad P  U _ { P } V _ { P } ^ { \top } , } \end{array}\tag{21}
$$

where $U _ { P } ~ \in ~ \mathbb { R } ^ { n \times m }$ and $V _ { P } ~ \in ~ \mathbb { R } ^ { m \times m }$ have orthonormal columns. The diagonal matrix $\Sigma _ { P } ~ \in ~ \mathbb { R } ^ { m \times m }$ contains the singular values. This update attains the maximum trace and preserves the orthogonality constraint.

c) Latent factor: For fixed $Z$ and $P ,$ differentiation gives $\nabla _ { L } \| Z - L P ^ { \top } \| _ { F } ^ { 2 } = 2 ( L - Z P )$ . The minimizing update is

$$
L \gets Z P .\tag{22}
$$

Each column of L is refined using the current latent samples and their coefficients. Together, the two updates adapt the representation to the full dataset using an n × m decomposition and thin matrix products.

Proposition 3 (Descent of the factor updates). For fixed $Z ,$ each update in (21) and (22) minimizes the factor objective with respect to its block. Repeating these updates yields a nonincreasing sequence of objective values that converges to a finite limit.

Appendix B proves this result. Algorithm 1 combines the factor updates with representation learning for a prescribed number of joint iterations.

## E. Subspace Structure of the Learned Projector

The projector interpretation also explains how self-expression can preserve subspace memberships. Consider latent samples spanning subspaces $S _ { 1 } , \ldots , S _ { k } \subseteq \mathbb { R } ^ { d _ { z } }$ with dimensions

$r _ { 1 } , \ldots , r _ { k }$ . For independent subspaces, their sum is direct and

$$
\dim ( S _ { 1 } + \cdot \cdot \cdot + S _ { k } ) = \sum _ { a = 1 } ^ { k } r _ { a } = r _ { z } .\tag{23}
$$

Proposition 4 (Subspace preservation and connectivity). Sup pose all latent samples are nonzero and span the direct sum in (23). If $m \ = \ r _ { z } \ < \ n$ and the fixed-Z factor objective attains its minimum, then $C = P P$ ⊤ is block diagonal after grouping samples by subspace. Within a subspace, its nonzero off-diagonal coefficients define a connected graph whenever that subspace’s samples admit no partition into two nonempty sets whose spans form a direct sum.

The result identifies conditions under which the projector both separates the generating subspaces and connects the samples within each subspace. Its proof uses the block structure of the row-space projector and appears in Appendix C. The next phase converts these coefficients into nonnegative graph weights while preserving their nonzero connections.

## F. Scalable Spectral Clustering from the Learned Factors

The learned factor P provides the pairwise coefficients through $C = P P ^ { \top }$ . We now use this structure to obtain the final partition directly from compact matrices. The construction has two steps. First, we convert the coefficients into nonnegative affinities while preserving their support. We then express the normalized spectral problem through a quadratic feature map whose dimension depends only on m.

a) A nonnegative affinity with an explicit feature map: The magnitude of $C _ { i j } = p _ { i } ^ { \top } p _ { j }$ measures the strength of the relation between samples i and $j .$ Squaring this coefficient gives the symmetric affinity

$$
A = C \circ C \in \mathbb { R } ^ { n \times n } , \qquad A _ { i j } = ( p _ { i } ^ { \top } p _ { j } ) ^ { 2 } ,\tag{24}
$$

where ◦ denotes entrywise multiplication. This construction assigns nonnegative weights to both positive and negative coefficients. It also preserves their zero pattern. Thus, any separation between subspaces encoded by C is retained in $A .$ Moreover, the affinity is invariant to an orthogonal change of basis in $P .$

The quadratic form in (24) admits an explicit feature representation. Let $q = m ( m + 1 ) / 2$ . For a symmetric matrix $S \in \mathbb { R } ^ { m \times m }$ $\mathrm { s v e c } ( S ) \in \mathbb { R } ^ { q }$ contains its m diagonal entries followed by its upper-triangular entries. Each off-diagonal entry is multiplied by ${ \sqrt { 2 } } .$ . This scaling preserves the Frobenius inner product between symmetric matrices. We define

$$
\begin{array} { r l r } & { \phi ( \boldsymbol { p } ) = \operatorname { s v e c } ( \boldsymbol { p } \boldsymbol { p } ^ { \top } ) \in \mathbb { R } ^ { q } , } & \\ & { F _ { i : } = \phi ( p _ { i } ) ^ { \top } , \qquad F \in \mathbb { R } ^ { n \times q } . } & \end{array}\tag{25}
$$

Consequently, $\phi ( p _ { i } ) ^ { \top } \phi ( p _ { j } ) = ( p _ { i } ^ { \top } p _ { j } ) ^ { 2 }$ , which gives

$$
A = F F ^ { \top } .\tag{26}
$$

Hence, the $n \times n$ affinity matrix is represented exactly through q features per sample and does not need to be formed explicitly.

We retain the diagonal entries of A as self-loop weights. The projector identity $C ^ { 2 } = C$ then gives a particularly simple

degree formula,

$$
\delta _ { i } = \sum _ { j = 1 } ^ { n } A _ { i j } = ( C ^ { 2 } ) _ { i i } = C _ { i i } = \| p _ { i } \| _ { 2 } ^ { 2 } , \qquad \sum _ { i = 1 } ^ { n } \delta _ { i } = m .\tag{27}
$$

Each degree therefore requires only a row norm of $P .$ Computing all degrees takes $O ( n m )$ time. Let ${ \mathcal { T } } _ { + } = \{ i : \delta _ { i } > 0 \}$ denote the indices with positive degrees and let $n _ { + } = | \mathcal { I } _ { + } |$ Spectral normalization is performed on these samples. Zerodegree samples receive labels through their latent features after clustering.

b) An exact reduced spectral problem: Let $F _ { + } \in \mathbb { R } ^ { n _ { + } \times q }$ contain the rows indexed by $\mathcal { T } _ { + }$ in increasing order. Define

$$
\begin{array} { r l } & { \Delta = \mathrm { d i a g } ( \delta _ { i } : i \in \mathcal { T } _ { + } ) \in \mathbb { R } ^ { n _ { + } \times n _ { + } } , } \\ & { B = \Delta ^ { - 1 / 2 } F _ { + } \in \mathbb { R } ^ { n _ { + } \times q } , } \\ & { G = B ^ { \top } B \in \mathbb { R } ^ { q \times q } . } \end{array}\tag{28}
$$

The normalized affinity is $B B ^ { \top }$ , and its normalized Laplacian is $I _ { n + } \mathrm { ~ - ~ } B B ^ { \top }$ . Thus, the smallest Laplacian eigenvalues correspond to the largest eigenvalues of $B B ^ { \top }$ . The following result transfers their computation to G.

Proposition 5 (Exact spectral reduction). For $P ^ { \top } P = I _ { m } ,$ , the normalized feature matrix satisfies

$$
m \leq \operatorname { r a n k } ( B ) \leq \operatorname* { m i n } \{ n _ { + } , q \} .\tag{29}
$$

The positive eigenvalues of G and $B B ^ { \top }$ coincide, including multiplicities. If $G v _ { j } = \nu _ { j } v _ { j }$ with $\nu _ { j } > 0$ and orthonormal $v _ { j } \in \mathbb { R } ^ { q }$ , then $B v _ { j } / \sqrt { \nu _ { j } }$ are orthonormal eigenvectors of $B B ^ { \top }$ with eigenvalues $\nu _ { j } .$

The proof is provided in Appendix D. Since $k \leq m ,$ , the reduced problem supplies at least k positive eigenvalues. The embedding dimension is $k ,$ whereas m controls the factor width and q determines the reduced spectral dimension.

The degree identity also gives one dominant eigenpair analytically,

$$
v _ { 1 } = \frac { \mathrm { s v e c } ( I _ { m } ) } { \sqrt { m } } , \qquad \nu _ { 1 } = 1 , \qquad B v _ { 1 } = \frac { ( \sqrt { \delta _ { i } } ) _ { i \in \mathcal { T } _ { + } } } { \sqrt { m } } .\tag{30}
$$

We retain this vector and compute the largest $k - 1$ eigenpairs on $v _ { 1 } ^ { \perp }$ . This gives k dominant modes while providing a strictly positive coordinate for every active sample. The sample embedding and its normalized rows are

$$
\begin{array} { r l } & { H = B [ v _ { 1 } , \dots , v _ { k } ] \mathrm { d i a g } ( \nu _ { 1 } , \dots , \nu _ { k } ) ^ { - 1 / 2 } \in \mathbb { R } ^ { n _ { + } \times k } , } \\ & { \widehat { H } _ { i : } = H _ { i : } / \| H _ { i : } \| _ { 2 } , \qquad i = 1 , \dots , n _ { + } . } \end{array}\tag{31}
$$

Equation (30) ensures that every row norm is positive. Retain ing the eigenvalue-one modes also preserves the component indicators when the affinity graph has k connected components. In that case, row normalization maps each component to a distinct embedding point, as established in Appendix D.

Following normalized spectral clustering [33], we apply k-means to the rows of ${ \widehat { H } } .$ . Let $\gamma _ { i } \in \{ 1 , \ldots , k \}$ denote the resulting label of active sample i. For a zero-degree sample, we use its distance to the latent centers of these clusters,

Algorithm 2 Spectral clustering through the quadratic feature   
map   
Require: $P \in \mathbb { R } ^ { n \times m }$ with $P ^ { \top } P = I _ { m } , Z \in \mathbb { R } ^ { d _ { z } \times n } , k \le m .$   
$R _ { \mathrm { k m } } , T _ { \mathrm { k m } } .$   
Ensure: Cluster labels $\gamma _ { 1 } , \ldots , \gamma _ { n } .$   
1: Compute $\delta _ { i }$ by (27) and retain the indices $\mathcal { T } _ { + }$ with $\delta _ { i } > 0 .$   
2: Set $q = m ( m + 1 ) / 2$ and the feature ordering in (25).   
3: Set $v _ { 1 } , \nu _ { 1 }$ using (30).   
4: if $k > 1$ then   
5: if direct eigensolver then   
6: Accumulate $\begin{array} { r } { G = \sum _ { i \in \mathcal { I } _ { + } } \phi ( p _ { i } ) \phi ( p _ { i } ) ^ { \top } / \delta _ { i } } \end{array}$ using   
(28).   
7: Compute its largest k − 1 eigenpairs on $v _ { 1 } ^ { \perp }$   
8: else   
9: Compute the largest k − 1 eigenpairs on $v _ { 1 } ^ { \perp }$ with   
a symmetric partial eigensolver using (33).   
10: end if   
11: end if   
12: Recover H and normalize its rows by (31).   
13: Run $R _ { \mathrm { k m } }$ seeded k-means restarts using (54), with at most   
$T _ { \mathrm { k m } }$ iterations each.   
14: Retain the restart with minimum distortion and restore the   
original sample indices.   
15: Assign zero-degree samples by (32).   
16: return $\gamma _ { 1 } , \ldots , \gamma _ { n } .$

$$
\begin{array} { r l r } & { \mathcal { T } _ { a } = \{ i \in \mathbb { Z } _ { + } : \gamma _ { i } = a \} , } & \\ & { \zeta _ { a } = \displaystyle \frac { 1 } { | \mathcal { T } _ { a } | } \sum _ { i \in \mathcal { I } _ { a } } z _ { i } \in \mathbb { R } ^ { d _ { z } } , } \\ & { \gamma _ { i } \in \arg \operatorname* { m i n } _ { a \in \{ 1 , \dots , k \} } \| z _ { i } - \zeta _ { a } \| _ { 2 } ^ { 2 } , \quad } & { i \notin \mathbb { Z } _ { + } . } \end{array}\tag{32}
$$

The clustering conventions in Appendix E ensure nonempty active clusters and specify initialization and tie-breaking. Together, these steps assign every sample to a cluster. The corresponding membership matrix is $\bar { \Gamma } \in \{ 0 , 1 \} ^ { n \times k }$ with $\Gamma _ { i a } = \mathbf { 1 } \{ \gamma _ { i } = a \}$

c) Direct and implicit implementations: When q is modest, we accumulate G from normalized feature rows and solve its symmetric eigenproblem directly. For larger q, the same operator can be applied by streaming the rows of P. Let smat : $\mathbb { R } ^ { q }  \mathbb { R } ^ { m \times m }$ be the inverse of svec on symmetric matrices. For any $v \in \mathbb { R } ^ { q }$

$$
G v = \mathrm { s v e c } \left( \sum _ { i \in \mathbb { Z } _ { + } } \frac { p _ { i } ^ { \top } \operatorname { s m a t } ( v ) p _ { i } } { \delta _ { i } } p _ { i } p _ { i } ^ { \top } \right) .\tag{33}
$$

Each operator evaluation takes $O ( n m ^ { 2 } )$ time. A symmetric partial eigensolver uses this action on $v _ { 1 } ^ { \perp }$ to recover the remaining dominant modes. Both implementations solve the spectral problem defined by (24). They use compact factors throughout, with $C ,$ A, and the sample Laplacian represented implicitly.

## G. Complete Algorithm and Computational Complexity

Algorithm 1 combines representation learning with spectral clustering. Let $T _ { \mathrm { p r e } }$ and $T$ be the numbers of pretraining updates and joint outer iterations, respectively. The spectral routine is given in Algorithm 2. It uses $R _ { \mathrm { k m } }$ clustering restarts with at most $T _ { \mathrm { k m } }$ Lloyd iterations per restart.

a) Representation learning: Let $c _ { \mathrm { n e t } }$ bound the persample cost of network evaluation and backpropagation in one update, including the refreshed encoder evaluation. Pretraining takes $O ( T _ { \mathrm { p r e } } n c _ { \mathrm { n e t } } )$ time. Seeding requires $O ( n d _ { z } m )$ time, and the initial Procrustes update requires $O ( n d _ { z } m + n m ^ { 2 } + m ^ { 3 } )$ Each joint iteration takes

$$
O \big ( n c _ { \mathrm { n e t } } + n d _ { z } m + n m ^ { 2 } + m ^ { 3 } \big ) .\tag{34}
$$

These terms account for network training, the products $Z ^ { \top } L$ and $Z P ;$ , and the thin SVD. The representation cost is the sum of pretraining and initialization costs plus $T$ times (34). Gradient accumulation allows the activation workspace to be controlled through the chunk size.

b) Spectral computation: Computing the degrees requires $O ( n m )$ time. In the direct implementation, constructing the reduced matrix $G \in \mathbb { R } ^ { q \times q }$ costs $O ( n q ^ { 2 } )$ , and its eigendecomposition costs $O ( q ^ { 3 } )$ ). Recovering the k-dimensional sample embedding requires $O ( n q k )$ operations. The total complexity is therefore

$$
O ( n m + n q ^ { 2 } + q ^ { 3 } + n q k ) , \qquad q = m ( m + 1 ) / 2 .\tag{35}
$$

The dependence on n appears only in the construction of the reduced matrix and the recovery of the sample embedding. The eigendecomposition is performed in the q-dimensional space.

For the implicit implementation, the matrix G is not formed explicitly. Let $T _ { \mathrm { e i g } }$ denote the total number of matrix– vector products required by the iterative eigensolver, $s _ { \mathrm { e i g } }$ the maximum Krylov basis size, and $R _ { \mathrm { e i g } }$ the number of restarts. Including operator evaluations, orthogonalization, and the projected eigenproblems, the total complexity is

$$
O \big ( n m + T _ { \mathrm { e i g } } ( n q + q s _ { \mathrm { e i g } } ) + R _ { \mathrm { e i g } } ( q s _ { \mathrm { e i g } } ^ { 2 } + s _ { \mathrm { e i g } } ^ { 3 } ) + n q k \big ) .\tag{36}
$$

Each vector processed by a block eigensolver contributes one matrix–vector product to $T _ { \mathrm { e i g } }$ . The quantities $T _ { \mathrm { e i g } } , \ s _ { \mathrm { e i g } } ,$ , and $R _ { \mathrm { e i g } }$ depend on the convergence of the eigensolver and the required numerical accuracy. Since the implicit implementation evaluates products with G only when needed, it avoids storing the dense $q \times q$ matrix and is preferable when q is large.

c) Clustering and memory: The k-dimensional embedding gives a clustering cost of $O ( R _ { \mathrm { k m } } ( T _ { \mathrm { k m } } + 1 ) n k ^ { 2 } )$ . Computing latent centers and assigning zero-degree samples adds at most $O ( n d _ { z } k )$ . Let $M _ { \mathrm { n e t } }$ include network parameters, optimizer state and the activation workspace. A resident-data memory bound is

$$
O ( n ( d _ { x } + d _ { z } + m + k ) + d _ { z } m + M _ { \mathrm { n e t } } ) ,\tag{37}
$$

plus $O ( q ^ { 2 } + q k )$ for direct reduction or $O ( q s _ { \mathrm { e i g } } + q k )$ for the implicit eigensolver. Quadratic feature rows can be evaluated as needed, so storing $F$ is optional. The output labels require $O ( n )$ memory.

Together, these bounds establish linear time and memory in n for fixed dimensions, architecture, factor width, cluster count and iteration budgets. The factor width m is a chosen model parameter, so both phases operate without estimating the data rank. Their shared use of compact factors makes scalability extend from representation learning to the final partition.

## IV. RESULTS AND DISCUSSION

a) Protocol: We evaluate LoomSC on five imageclustering benchmarks. The face datasets are Extended YaleB (EYaleB), ORL, and UMIST, with $( n , k ) \ = \ ( 2 , 4 3 2 , 3 8 )$ (400, 40), and (575, 20), respectively. We also use COIL-100 with 7,200 images from 100 object classes and a balanced subset of Fashion-MNIST with $5 , 0 0 0$ images from 10 classes. Each sample is normalized to unit $\ell _ { 2 }$ norm. Tables I and IV specify the sample counts used in the comparisons.

We compare with nine baselines that distinguish deep representation learning from computational scalability. The scalable non-deep methods comprise two groups. LMVSC [15], SGL [16], and $\mathrm { \bf S ^ { 5 } C }$ [29] use sampling or anchor-based reduction. EnSC [19], SSC-OMP [18], A-DSSC [22], and SSSC [20] accelerate self-expression estimation without learning deep representations. These methods target scalability, but do not all achieve linear complexity for the complete clustering pipeline.

The non-linear-time full-affinity references are LSR [9] and PRO-DSC [28]. LSR is non-deep and computes a dense selfexpression matrix. PRO-DSC is deep and supports mini-batch training. Its published spectral inference nevertheless constructs pairwise coefficients over the clustering set and processes the resulting affinity. Thus, scalable training does not eliminate its full-affinity spectral bottleneck. LoomSC combines deep representation learning with factorized self-expression and spectral assignment. Its complete pipeline has linear time and memory complexity in n for fixed architecture, dimensions, factor width, and iteration budgets.

A detailed comparison of computational and matrix-storage complexities is provided in Appendix F.

For LoomSC, we use the same architecture and optimization hyperparameters across all datasets. The encoder contains $L =$ 3 convolutional layers with 10, 20, and 30 output channels and kernel sizes $5 \times 5 , 3 \times 3$ , and $3 \times 3$ , respectively. Each layer uses stride 2 and a ReLU activation, with no batch normalization. The decoder mirrors the encoder using transposed convolutions. The resulting convolutional representation has size $3 0 \times 6 \times 6 ,$ which gives a flattened latent dimension of $d _ { z } = 1 0 8 0$ . We set the factor width to $m = 3 8 0$ for all datasets. The auto-encoder is first pretrained for 5,000 epochs using full-batch Adam with a learning rate of $1 0 ^ { - 3 }$ . No data augmentation, weight decay, or learning-rate scheduling is used.

After pretraining, k-means with k-means++ initialization is applied to the latent representations, and its m centroids initialize the columns of L. The factor $P$ is then initialized by the orthogonal Procrustes solution. Joint optimization is performed for another $5 { , } 0 0 0$ epochs using full-batch Adam with a learning rate of $1 0 ^ { - 3 }$ . The reconstruction and selfexpression terms have equal weights. At each joint-training epoch, the network parameters are first updated by Adam.

TABLE I  
CLUSTERING QUALITY — ACC (%) ON THE FIVE BENCHMARKS. HIGHER IS BETTER. MEAN ± STANDARD DEVIATION OVER THREE SEEDS WHERE AVAILABLE. BOLD INDICATES THE LARGEST MEAN IN EACH ROW.
<table><tr><td colspan="8"></td><td colspan="4"></td><td colspan="2">Deep methods</td></tr><tr><td>Dataset</td><td>n</td><td>k</td><td>EnSC</td><td>SSC-OMP</td><td>LMVSC</td><td>SGL</td><td>S⁵C</td><td>LSR</td><td>A-DSSC</td><td>SSSC</td><td>PRO-DSC</td><td></td><td>LoomSC</td></tr><tr><td>Fashion-MNIST 5,000</td><td></td><td>10</td><td> $6 1 . 3 5 \pm 0 . 0 0$ </td><td> $5 6 . 1 9 \pm 1 . 3 6$ </td><td>59.18</td><td> $5 4 . 4 7 \pm 0 . 3 6$ </td><td> $5 4 . 6 2 \pm 1 . 9 9$ </td><td>59.52 ± 0.00</td><td> $3 9 . 0 1 \pm 2 . 2 0$ </td><td></td><td> $4 2 . 0 1 \pm 1 . 7 0$ </td><td> $5 2 . 1 2 \pm 2 . 9 2$ </td><td> ${ \bf 6 5 . 3 3 \pm 1 . 7 9 }$ </td></tr><tr><td>ORL</td><td>400</td><td>40</td><td> $7 8 . 0 0 \pm 1 . 5 9$ </td><td> $7 3 . 7 5 \pm 0 . 8 2$ </td><td> $5 5 . 2 5 \pm 0 . 0 0$ </td><td>62.25 ± 0.00</td><td> $4 2 . 5 0 \pm 0 . 0 0$ </td><td>82.50 ± 0.00</td><td></td><td> $5 0 . 7 5 \pm 0 . 0 0$ </td><td> $4 6 . 5 0 \pm 1 . 5 4$ </td><td> $5 2 . 0 0 \pm 7 . 9 1$ </td><td> $\mathbf { 8 6 . 1 7 \ : \pm { 2 . 0 1 } }$ </td></tr><tr><td>UMIST</td><td>575</td><td>20</td><td> $7 2 . 5 2 \pm 0 . 0 0$ </td><td> $6 7 . 7 0 \pm 0 . 0 0$ </td><td> $6 0 . 6 2 \pm 0 . 0 0$ </td><td></td><td></td><td>65.00 ± 0.00 40.87 ± 0.00 58.26 ± 0.00</td><td></td><td> $5 2 . 5 2 \pm 0 . 0 0$ </td><td> $2 1 . 1 6 \pm 1 . 3 1$ </td><td> ${ \bf 8 1 . 3 9 \pm 1 . 5 8 }$ </td><td> $8 0 . 0 0 \pm 2 . 1 9$ </td></tr><tr><td>EYaleB</td><td>2,432</td><td>38</td><td> $8 8 . 1 9 \pm 0 . 0 0$ </td><td> $8 0 . 8 4 \pm 1 . 3 4$ </td><td></td><td></td><td></td><td></td><td>21.50 ± 0.00 21.50 ± 0.00 22.41 ± 0.00 96.26 ± 0.00 61.88 ± 0.00</td><td></td><td> $2 5 . 1 1 \pm 2 . 2 7$ </td><td> $7 7 . 6 3 \pm 0 . 7 3$ </td><td> $9 4 . 7 3 \pm 1 . 2 6$ </td></tr><tr><td>COIL-100</td><td>7,200</td><td>100</td><td> $6 7 . 5 0 \pm 0 . 6 9$ </td><td> $5 9 . 4 6 \pm 0 . 0 0$ </td><td></td><td></td><td>49.50 ± 0.00 49.26 ± 0.00 42.04 ± 0.00 61.10 ± 0.00</td><td></td><td></td><td> $4 8 . 5 7 \pm 0 . 0 0$ </td><td> $1 3 . 8 1 \pm 1 . 3 1$ </td><td> ${ \bf 8 2 . 3 1 \pm 0 . 3 0 }$ </td><td> $7 4 . 6 4 \pm 1 . 3 4$ </td></tr></table>

TABLE II

CLUSTERING QUALITY — NMI (%) ON THE FIVE BENCHMARKS. HIGHER IS BETTER. MEAN ± STANDARD DEVIATION OVER THREE SEEDS. BOLD INDICATES THE LARGEST MEAN IN EACH ROW.
<table><tr><td rowspan="2">Dataset</td><td colspan="8">Non-deep methods</td><td colspan="2">Deep methods</td></tr><tr><td>EnSC</td><td>SSC-OMP</td><td>LMVSC</td><td>SGL</td><td>S⁵C</td><td>LSR</td><td>A-DSSC</td><td>SSSC</td><td>PRO-DSC</td><td>LoomSC</td></tr><tr><td>Fashion-MNIST</td><td> $6 0 . 2 9 \pm 0 . 8 8$ </td><td> $5 7 . 9 3 \pm 0 . 4 1$ </td><td>64.37 ± 0.63</td><td> $6 2 . 5 1 \pm 0 . 4 2$ </td><td> $5 4 . 8 2 \pm 2 . 3 3$ </td><td> $6 3 . 3 3 \pm 0 . 0 0$ </td><td> $5 3 . 8 3 \pm 1 . 0 0$ </td><td> $4 6 . 8 8 \pm 0 . 4 5$ </td><td> $5 6 . 4 5 \pm 3 . 0 8$ </td><td> $6 3 . 9 1 \pm 0 . 8 2$ </td></tr><tr><td>ORL</td><td> $8 9 . 2 5 \pm 0 . 0 7$ </td><td> $8 4 . 5 4 \pm 0 . 3 7$ </td><td> $7 4 . 5 3 \pm 0 . 0 0$ </td><td> $7 8 . 2 4 \pm 0 . 0 0$ </td><td> $6 4 . 7 6 \pm 0 . 0 0$ </td><td>90.07 ± 0.00</td><td> $7 7 . 9 3 \pm 0 . 0 0$ </td><td> $6 6 . 8 4 \pm 1 . 3 4$ </td><td> $7 1 . 7 1 \pm 5 . 3 9$ </td><td> ${ \bf 9 1 . 7 6 \pm 0 . 8 8 }$ </td></tr><tr><td>UMIST</td><td> $8 7 . 0 6 \pm 0 . 0 0$ </td><td> $6 3 . 6 1 \pm 0 . 5 3$ </td><td> $6 2 . 4 9 \pm 0 . 0 0$ </td><td> $5 6 . 8 4 \pm 0 . 0 0$ </td><td> $5 6 . 7 2 \pm 0 . 0 0$ </td><td>75.40 ± 0.00</td><td> $6 6 . 0 1 \pm 0 . 0 0$ </td><td> $3 2 . 8 9 \pm 1 . 4 0$ </td><td> ${ \bf 9 0 . 0 2 \pm 0 . 2 0 }$ </td><td> $8 7 . 8 3 \pm 1 . 4 4$ </td></tr><tr><td>EYaleB</td><td> $7 4 . 3 1 \pm 0 . 0 3$ </td><td> $8 3 . 1 7 \pm 0 . 5 5$ </td><td> $1 3 . 3 7 \pm 0 . 0 0$ </td><td> $6 . 4 4 \pm 0 . 0 0$ </td><td> $3 7 . 9 2 \pm 0 . 0 0$ </td><td>95.44 ± 0.00</td><td> $7 5 . 6 1 \pm 0 . 0 0$ </td><td> $3 4 . 2 4 \pm 2 . 4 2$ </td><td> $8 2 . 3 8 \pm 0 . 9 1$ </td><td> ${ \bf 9 5 . 4 7 \pm 0 . 4 3 }$ </td></tr><tr><td>COIL-100</td><td> $9 0 . 9 5 \pm 0 . 0 2$ </td><td> $5 7 . 1 1 \pm 0 . 1 6$ </td><td> $7 4 . 6 9 \pm 0 . 0 0$ </td><td></td><td>74.54 ± 0.00 68.03 ± 0.00 85.21 ± 0.00 74.31 ± 0.00</td><td></td><td></td><td> $3 7 . 8 0 \pm 1 . 0 8$ </td><td> ${ \bf 9 5 . 7 5 \pm 0 . 0 5 }$ </td><td> $9 0 . 9 5 \pm 0 . 4 1$ </td></tr></table>

TABLE III

CLUSTERING QUALITY — ARI (%) ON THE FIVE BENCHMARKS. HIGHER IS BETTER. MEAN ± STANDARD DEVIATION OVER THREE SEEDS. BOLD INDICATES THE LARGEST MEAN IN EACH ROW.
<table><tr><td></td><td colspan="8">Non-deep methods</td><td colspan="2">Deep methods</td></tr><tr><td>Dataset</td><td>EnSC</td><td>SSC-OMP</td><td>LMVSC</td><td>SGL</td><td> $\mathrm { { \bf S } } ^ { 5 } \mathrm { { C } }$ </td><td>LSR</td><td>A-DSSC</td><td>SSSC</td><td>PRO-DSC</td><td>LoomSC</td></tr><tr><td>Fashion-MNIST</td><td> $4 3 . 2 1 \pm 1 . 0 4$ </td><td> $4 2 . 0 0 \pm 0 . 1 6$ </td><td> $4 5 . 3 5 \pm 0 . 8 7$ </td><td> $4 2 . 7 5 \pm 0 . 2 2$ </td><td> $4 0 . 2 7 \pm 1 . 5 0$ </td><td> $4 6 . 5 3 \pm 0 . 0 0$ </td><td> $2 2 . 2 4 \pm 1 . 2 9$ </td><td> $1 9 . 4 1 \pm 1 . 4 9$ </td><td> $3 7 . 7 6 \pm 4 . 0 7$ </td><td> ${ \bf 5 1 . 8 9 \pm 1 . 4 3 }$ </td></tr><tr><td>ORL</td><td> $6 9 . 5 0 \pm 0 . 5 5$ </td><td> $6 0 . 2 2 \pm 0 . 7 4$ </td><td> $3 9 . 2 2 \pm 0 . 0 0$ </td><td> $4 7 . 6 0 \pm 0 . 0 0$ </td><td> $2 2 . 6 6 \pm 0 . 0 0$ </td><td> $7 4 . 1 0 \pm 0 . 0 0$ </td><td> $2 1 . 4 9 \pm 0 . 0 0$ </td><td> $1 5 . 3 2 \pm 4 . 9 8$ </td><td> $3 5 . 5 5 \pm 8 . 7 5$ </td><td> ${ \bf 7 8 . 0 5 \pm 2 . 6 7 }$ </td></tr><tr><td>UMIST</td><td> $6 6 . 9 1 \pm 0 . 0 0$ </td><td> $3 4 . 8 1 \pm 0 . 4 7$ </td><td> $3 2 . 1 2 \pm 0 . 0 0$ </td><td> $2 5 . 1 9 \pm 0 . 0 0$ </td><td> $2 6 . 6 3 \pm 0 . 0 0$ </td><td> $4 6 . 8 9 \pm 0 . 0 0$ </td><td> $2 2 . 9 9 \pm 0 . 0 0$ </td><td> $1 . 0 7 \pm 0 . 3 1$ </td><td> ${ \bf 7 9 . 3 6 \pm 0 . 6 5 }$ </td><td> $7 5 . 5 5 \pm 2 . 6 9$ </td></tr><tr><td>EYaleB</td><td> $1 9 . 6 7 \pm 0 . 0 6$ </td><td> $5 3 . 9 1 \pm 5 . 1 1$ </td><td> $1 . 8 7 \pm 0 . 0 0$ </td><td> $- 0 . 4 0 \pm 0 . 0 0$ </td><td> $1 3 . 6 5 \pm 0 . 0 0$ </td><td> ${ \bf 9 2 . 1 9 \pm 0 . 0 0 }$ </td><td> $2 1 . 6 7 \pm 0 . 0 0$ </td><td> $2 . 1 6 \pm 0 . 3 1$ </td><td> $6 4 . 1 5 \pm 1 . 2 8$ </td><td> $9 1 . 7 6 \pm 0 . 9 0$ </td></tr><tr><td>COIL-100</td><td> $6 1 . 3 8 \pm 1 . 8 4$ </td><td> $2 1 . 9 8 \pm 0 . 1 3$ </td><td> $4 3 . 0 0 \pm 0 . 0 0$ </td><td> $4 3 . 9 3 \pm 0 . 0 0$ </td><td> $3 5 . 0 5 \pm 0 . 0 0$ </td><td> $5 4 . 0 3 \pm 0 . 0 0$ </td><td> $7 . 8 4 \pm 0 . 0 0$ </td><td> $1 . 5 9 \pm 0 . 0 4$ </td><td> ${ \bf 8 0 . 6 3 \pm 1 . 2 8 }$ </td><td> $6 6 . 6 0 \pm 0 . 6 6$ </td></tr></table>

The factor $P$ is then updated by the exact Procrustes solution, followed by the closed-form update $L = Z P ,$ , as described in Algorithm 1. After joint optimization, spectral assignment is performed following Algorithm 2. The spectral embedding dimension is set to the target number of clusters k, after which k-means produces the final partition.

We report clustering accuracy (ACC), normalized mutual information (NMI), and adjusted Rand index (ARI) as percent ages. Higher values indicate better clustering. Repeated-run results are summarized by their mean and standard deviation over three seeds. A scalar entry indicates that no standard deviation is reported. Differences between percentage scores are expressed in percentage points. The synthetic experiments additionally report the connectivity diagnostic (CONN) [34], for which higher values are preferred.

LoomSC achieves higher ACC than every scalable nondeep baseline on all five datasets. The improvements over the strongest method in this group range from 3.98 to 8.17 points. The gains over sampling-based methods are larger. Relative to the best ACC among LMVSC, SGL, and ${ \mathrm { { S } } } ^ { 5 } { \mathrm { { C } } } ,$ , LoomSC improves by 23.92 points on ORL and 25.14 points on COIL-100. The corresponding gains are 15.00 points on UMIST and 72.32 points on EYaleB. These comparisons show that a compact model need not be restricted to the clustering quality of fixed-representation sampling methods. LoomSC retains compactness while jointly refining the latent features and their self-expression factors.

## A. Clustering quality

Tables I–III show that LoomSC ranks first or second in all 15 dataset–metric comparisons, including a shared second place for COIL-100 NMI. No other evaluated method maintains this ranking across all five datasets and three metrics. Its unweighted means are 80.17% ACC, 85.98% NMI, and 72.77% ARI. These exceed the highest corresponding baseline means by 6.66, 4.09, and 10.02 points. Thus, the compact formulation provides consistently strong clustering quality while retaining linear-inn computation under the conditions stated above.

On Fashion-MNIST, LoomSC obtains the highest ACC of $6 5 . 3 3 \pm 1 . 7 9 \%$ and ARI of $5 1 . 8 9 \pm 1 . 4 3 \%$ . These exceed the strongest competing results by 3.98 and 5.36 points, respectively. Its NMI of $6 3 . 9 1 \pm 0 . 8 2 \%$ is within 0.46 points of the highest value. The agreement between ACC and ARI supports the quality of the partition beyond a single evaluation criterion.

On ORL, LoomSC leads all three metrics with 86.17±2.01% ACC, 91.76±0.88% NMI, and 78.05±2.67% ARI. It improves upon LSR, the strongest alternative for each metric, by 3.67, 1.69, and 3.95 points. These results are obtained with only ten images per subject. On EYaleB, LoomSC reaches 94.73% ACC, 95.47% NMI, and 91.76% ARI. Its NMI closely matches LSR at 95.44%, and their ARI scores differ by only 0.43 points.

TABLE IV  
MEASURED WALL-CLOCK TIME (S). LOWER IS BETTER. BOLD INDICATES THE SMALLEST MEAN IN EACH ROW.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">n EnSC</td><td colspan="8">Non-deep methods</td><td colspan="2">Deep methods</td></tr><tr><td></td><td>SSC-OMP</td><td>LMVSC</td><td>SGL</td><td>S⁵C</td><td>LSR</td><td>A-DSSC</td><td>SSSC</td><td>PRO-DSC</td><td>LoomSC</td></tr><tr><td>Fashion-MNIST 5,000</td><td></td><td> $3 2 1 9 . 7 5 2 \pm 4 6 . 9 4 8$ </td><td> ${ \bf 5 . 4 4 7 \pm 0 . 6 8 1 }$ </td><td> $1 6 . 9 6 5 \pm 0 . 1 7 9$ </td><td> $1 7 . 2 0 3 \pm 0 . 2 1 9$ </td><td> $8 . 2 8 0 \pm 1 . 0 0 0$ </td><td> $6 . 2 6 0 \pm 0 . 1 5 7$ </td><td> $3 5 . 6 1 2 \pm 0 . 5 8 4$ </td><td> $2 2 3 1 . 3 1 4 \pm 6 9 . 8 7 7$ </td><td> $5 6 3 . 3 3 8 \pm 1 . 3 9 2$ </td><td> $9 . 0 0 8 \pm 0 . 9 8 7$ </td></tr><tr><td>ORL</td><td>400</td><td> $7 . 9 5 3 \pm 0 . 5 1 4$ </td><td> $1 . 8 5 6 \pm 0 . 0 5 8$ </td><td> $0 . 9 2 9 \pm 0 . 0 3 1$ </td><td> $0 . 9 5 8 \pm 0 . 0 0 3$ </td><td> $7 . 4 4 0 \pm 0 . 4 1 9$ </td><td> $\mathbf { 0 . 0 4 9 \ : \pm { \ : 0 . 0 0 1 } }$ </td><td> $0 . 6 9 9 \pm 0 . 0 0 5$ </td><td> $7 . 0 8 4 \pm 0 . 1 8 3$ </td><td> $1 0 2 . 7 9 7 \pm 2 . 6 7 1$ </td><td> $0 . 2 3 9 \pm 0 . 0 8 9$ </td></tr><tr><td>UMIST</td><td>575</td><td> $2 4 . 3 6 6 \pm 2 6 . 8 9 9$ </td><td> $0 . 4 9 8 \pm 0 . 0 1 2$ </td><td> $1 . 0 8 6 \pm 0 . 0 3 3$ </td><td> $1 . 2 6 3 \pm 0 . 0 4 3$ </td><td> $1 0 . 2 4 9 \pm 4 . 9 4 6$ </td><td> $\mathbf { 0 . 0 7 7 \pm 0 . 0 0 2 }$ </td><td> $1 . 0 2 6 \pm 0 . 0 2 7$ </td><td> $9 . 7 7 4 \pm 1 . 4 2 3$ </td><td> $7 8 . 6 8 9 \pm 0 . 1 7 2$ </td><td> $0 . 2 0 3 \pm 0 . 0 0 3$ </td></tr><tr><td>EYaleB</td><td>2,432</td><td> $4 3 . 8 4 3 \pm 0 . 3 9 8$ </td><td> $1 2 . 6 6 6 \pm 0 . 9 7 5$ </td><td> $9 . 1 2 4 \pm 0 . 0 6 3$ </td><td> $9 . 2 2 1 \pm 0 . 0 8 3$ </td><td> $4 7 1 . 7 6 8 \pm 2 4 . 6 4 7$ </td><td> $2 . 3 8 9 \pm 0 . 5 4 2$ </td><td> $1 3 . 9 1 8 \pm 0 . 1 3 2$ </td><td> $2 2 8 6 . 4 2 1 \pm 9 4 . 4 8 8$ </td><td> $3 1 3 . 6 4 8 \pm 2 . 6 7 0$ </td><td> $\mathbf { 2 . 2 4 0 \ : \pm { \ : 0 . 5 8 1 } }$ </td></tr><tr><td>COIL-100</td><td>7,200</td><td> $5 0 . 8 5 8 \pm 1 . 5 2 3$ </td><td> $1 3 1 9 . 5 8 6 \pm 1 6 4 . 8 6 3$ </td><td> ${ \bf 2 1 . 4 2 1 \pm 0 . 1 4 3 }$ </td><td> $2 2 . 4 9 8 \pm 0 . 3 9 0$ </td><td> $4 7 0 9 . 5 6 8 \pm 1 6 9 . 0 2 3$ </td><td> $4 9 . 8 8 8 \pm 5 . 4 5 1$ </td><td> $1 7 5 . 3 2 8 \pm 1 . 5 5 1$ </td><td> $2 4 0 9 . 5 9 9 \pm 2 4 5 . 4 9 2$ </td><td> $7 4 4 3 . 6 2 7 \pm 4 . 9 9 2$ </td><td> $2 5 . 3 1 1 \pm 9 . 9 7 6$ </td></tr></table>

Thus, the thin-factor representation can match or improve upon dense least-squares self-expression on several clustering measures without storing all pairwise coefficients.

The comparison with PRO-DSC assesses whether this efficiency is compatible with deep clustering quality. LoomSC exceeds PRO-DSC on all three metrics for Fashion-MNIST, ORL, and EYaleB. The ACC gains are 13.21, 34.17, and 17.10 points, respectively. Across the five datasets, its average improvements over PRO-DSC are 11.08 ACC points, 6.72 NMI points, and 13.28 ARI points. Therefore, the advantage over the non-deep baselines is accompanied by a strong comparison with a deep full-affinity method.

PRO-DSC attains the highest scores on UMIST and COIL 100, where LoomSC ranks second on all three metrics. On UMIST, LoomSC reaches 80.00% ACC compared with 81.39% for PRO-DSC. On COIL-100, it obtains 74.64% ACC, 90.95% NMI, and 66.60% ARI. Its COIL-100 ACC exceeds every non-deep baseline, with gains of 7.14 points over EnSC and 13.54 points over LSR. These results establish competitive deep clustering performance on both datasets. As quantified next, LoomSC attains this performance at a small fraction of PRO-DSC’s measured runtime and without its non-linear-time full-affinity spectral stage.

## B. Computational efficiency

Table IV reports the measured wall-clock times. LoomSC requires 0.239 s on ORL and 0.203 s on UMIST. Its mean time is 2.240 s on EYaleB, 9.008 s on Fashion-MNIST, and 25.311 s on COIL-100. No competing method combines higher mean ACC with lower mean runtime on any of these datasets. Thus, LoomSC lies on the empirical ACC–runtime Pareto frontier for every benchmark among the evaluated implementations.

The distinction from PRO-DSC concerns the complete clustering pipeline rather than network training alone. PRO-DSC reduces training costs through mini-batches, but its published inference constructs a full pairwise affinity before spectral clustering [28]. Explicitly storing this affinity requires $O ( n ^ { 2 } )$ memory for a clustering set of size n. LoomSC instead retains $P \in \mathbb { R } ^ { n \times m }$ and $L \in \bar { \mathbb { R } } ^ { d _ { z } \times m }$ during representation learning. Its quadratic feature map transfers spectral computation to dimension $q = m ( m + 1 ) / 2$ , which is independent of n for fixed m. Consequently, it removes the full-affinity bottleneck rather than only accelerating its construction. Section III-G establishes linear time and memory in n under fixed dimensions and iteration budgets.

The reported runtimes show substantial savings relative to the deep baseline. LoomSC is 62.5× faster than PRO-DSC on Fashion-MNIST while improving ACC by 13.21 points. On

ORL, it is 430.1× faster and improves ACC by 34.17 points. On EYaleB, the corresponding speedup is 140.0×, together with a 17.10-point ACC gain. Hence, on three datasets, the computational reduction accompanies higher clustering quality rather than a loss of accuracy.

The datasets where PRO-DSC achieves its strongest results further illustrate the scalability advantage. On UMIST, LoomSC attains 80.00% ACC in 0.203 s, compared with 81.39% in 78.689 s for PRO-DSC. This is a 387.6× speedup with an ACC difference of 1.39 points. On COIL-100, LoomSC obtains the second-highest ACC while reducing the measured time from 7,443.627 to 25.311 s, a 294.1× speedup. Across all five datasets, the speedups over PRO-DSC range from 62.5× to 430.1×. Therefore, competitive deep clustering performance is retained without the runtime of the full-affinity reference.

LoomSC also improves the quality–cost balance relative to scalable non-deep methods. On Fashion-MNIST, it exceeds EnSC by 3.98 ACC points with a 357.4× speedup. On COIL-100, it improves ACC over EnSC and LSR by 7.14 and 13.54 points while approximately halving their runtimes. Compared with LMVSC, the fastest method on COIL-100, LoomSC requires only 3.89 additional seconds and improves ACC by 25.14 points. These results show that LoomSC combines runtimes close to or below those of sampling methods with substantially stronger clustering. Its compact pipeline also remains far faster than the deep full-affinity reference.

## C. Scalability Analysis

We first vary the sampling density in a controlled union-ofsubspaces setting. Following the synthetic-data configuration in [20], we generate k = 10 subspaces with random dimensions between 6 and 12 in an ambient space of dimension $d _ { x } = 7 8 4$ Each subspace contains $n _ { i } ~ \in ~ \{ 3 0 , 5 0 , 1 0 0 , 2 0 0 , 5 0 0 , 1 0 0 0 \}$ samples. The total sample count is therefore $n = 1 0 n _ { i } ,$ ranging from 300 to 10,000. Figure 1 compares ACC, connectivity, and CPU and GPU runtimes.

LoomSC maintains 99%–100% ACC across the entire density range. Its connectivity remains high, at 0.68–0.75. Several competing methods attain high ACC at sufficiently large sample counts but have much smaller connectivity scores. Thus, the two diagnostics provide complementary evidence: LoomSC produces accurate partitions while retaining strong graph connectivity under the reported measure. This behavior is consistent with the goal of preserving informative sample relations through the factorized representation.

The comparison with the SDSNet [35] reference is particularly informative at low sampling densities. With 30 samples per subspace, LoomSC reaches 99% ACC, compared with about 85% for SDSNet. With 50 samples per subspace, its ACC is 100%, compared with about 87%. At the largest size, the two approaches both retain near-perfect accuracy and high connectivity. LoomSC nevertheless reduces the measured runtime from 143 to 39 s in both the CPU and GPU panels. The improvement is visible within each hardware setting rather than arising only from a CPU-to-GPU comparison.

![](images/23deb9426dad2889adaa5d62ae5b9253c5e20c8f932b530b92251cbca2068adf.jpg)

c))(b)  
![](images/beb02eb23d0e26312fcbbe60da4accb4c0e1d5f3e595c260ce6b7cf7411e8a22.jpg)

(c)(  
![](images/11c9e4ce70197d05d357c0613ef3a9c3e33d2134cad949fac334c6c41a95107d.jpg)

)(d)  
![](images/672716b14f3bb33756ac910ef7e5aff475b92bd8f34cf686cdd00b9886b6681a.jpg)  
Fig. 1. Synthetic-data comparison with ten subspaces and n samples per subspace. Panels show (a) ACC, (b) connectivity, (c) CPU runtime, and (d) GPU runtime. “Ours” denotes LoomSC.

The same experiment contrasts LoomSC with the fullaffinity deep baseline. PRO-DSC also reaches near-perfect ACC at higher sampling densities. However, its CPU runtime is substantially larger at the highest density. LoomSC combines high clustering accuracy across the full density range with lower computational cost at the largest size in this experiment.

Figure 2 extends the sample-size experiment from 1,000 to 500,000 points. LoomSC maintains 99.8% ACC or higher throughout this 500-fold increase. Its runtime grows from about 57 s at 50,000 samples to 362 s at 500,000 samples. At the largest size, it retains near-perfect accuracy in approximately six minutes. $\mathrm { \bf S ^ { 5 } C }$ reaches comparable accuracy but requires about 1,054 s. LMVSC has a lower runtime of about 125 s, with ACC near 94%. SGL also reaches near-perfect ACC at large sizes, whereas LoomSC maintains this accuracy regime across the full tested range, including the smallest datasets.

The large-sample runtime trend is consistent with the linearin-n bounds in Section III-G. For fixed factor width, the quadratic feature dimension $q = m ( m + 1 ) / 2$ does not grow with the number of samples. Neither factor learning nor the reduced spectral stage requires storage of all pairwise affinities. With fixed dimensions and iteration budgets, the sample-dependent work remains linear. The experiment shows that this design supports increasing the dataset size without a corresponding loss of clustering accuracy.

![](images/430bc6bae4bcecad849802bf4d43e75615f904eac28996db5bb66fc63f1d711d.jpg)

![](images/32b23fd4ed62c1b4014c7d0d48be7bc1469b71339a9e8ac5233df7b5d53c8cdc.jpg)

Fig. 2. Synthetic-data scalability from 1,000 to 500,000 samples: (a) runtime and (b) clustering accuracy. The horizontal-axis symbol N denotes the total sample count n. The reference curves indicate linear and quadratic growth. “Ours” denotes LoomSC.  
![](images/31a69239c0642ed0c290e8c13820d2a4e5dca76bed55b9db944e4f9c30144f4d.jpg)

![](images/94aa018348dfef5edf3230577ab7413c0677ae6a01fbff55b7157866bb9ec1e6.jpg)  
Fig. 3. Relative Frobenius change $\| A _ { t } - A _ { t - 1 } \| _ { F } / \| A _ { t - 1 } \| _ { F }$ on (a) synthetic datasets with ten subspaces and (b) real datasets. Here, $A _ { t }$ denotes the affinity matrix in (24) at outer iteration $t ,$ and $n _ { i }$ is the number of samples in synthetic subspace i.

## D. Convergence of the Affinity Matrix

Figure 3 examines the relative Frobenius change between successive affinity iterates. The diagnostic divides the Frobenius norm of the difference by the norm of the previous iterate. This difference measures the stability of the sample relationships used for clustering, rather than the scale of the matrix alone.

On synthetic data, all six sampling densities exhibit a short initial transient followed by gradual stabilization. The relative change falls to a few $1 0 ^ { - 3 }$ within roughly the first 100 outer iterations and approaches $1 0 ^ { - 3 }$ by the end of the run. The curves reach similar terminal scales despite the increase from 30 to 1,000 samples per subspace. Thus, the larger synthetic datasets retain the same qualitative stabilization pattern.

The real-data curves show similarly small late-stage changes. UMIST falls below $1 0 ^ { - 3 }$ after about 100 iterations. Fashion-

MNIST reaches a low baseline with temporary excursions, while EYaleB settles around a few $1 0 ^ { - 3 }$ . The dominant pattern is a rapid reduction in the update magnitude followed by sustained small changes. This supports the practical stability of the learned affinities during joint training. Proposition 3 provides a complementary descent result for the exact factor updates with fixed latent features. The observed curves show that small affinity changes are also attained when these updates are coupled with representation learning.

## E. Affinity-matrix structure

Figures 4–6 visualize the learned sample relations at epochs 10, 100, and 500 after pretraining. The face-data panels use tenclass subsets, and the synthetic panels contain ten subspaces. Samples are grouped by their classes, with class boundaries overlaid. These plots examine whether the representation retains class-aligned relations throughout optimization.

![](images/3a16b2bacfff85f0be6c01ffc804681a1d432b5f469e994d9fb5bbe151222600.jpg)  
(a) Epoch 10

![](images/d5033b28cfd5a56ccb78a592d183558dfcfb401ec68aaf7a0197b0b306235da1.jpg)  
(b) Epoch 100

![](images/4f4119ea09eb010fea101124bc1341cb4bfd92f516f4eedd1720df8f1014c1ff.jpg)  
(c) Epoch 500  
Fig. 4. Learned sample relations on a ten-class EYaleB subset at epochs 10, 100, and 500. Class boundaries are overlaid.

![](images/8285458abdfcf887c8da17df50b0b1467d66d037b2181650edf370720e09529e.jpg)  
(a) Epoch 10

![](images/955b593129a53ccfb051d24585d23cbfeffa1ec0aaa50da84175a1deaff8718a.jpg)  
(b) Epoch 100

![](images/fcbc2c9c57ae15f571171b2d1516bdca854f054a060e066c594d64619e121551.jpg)  
(c) Epoch 500  
Fig. 5. Learned sample relations on a ten-class ORL subset at epochs 10, 100, and 500. Class boundaries are overlaid.

![](images/c1c7bc2c2ab63e443a6357b1d28edcaf7e8140497b8b4495825cd68fc14597e3.jpg)  
(a) Epoch 10

![](images/a859c3a2426281b9c61c6d77184c4d6c1d3fbf50b218f12817de90a1513e2eea.jpg)  
(b) Epoch 100

![](images/ee528096d3eab98ced29c30f7dd8447e889f4e6c0d492948c5b174d82f6219a8.jpg)  
(c) Epoch 500  
Fig. 6. Learned sample relations on synthetic data with ten subspaces at epochs 10, 100, and 500. Subspace boundaries are overlaid.

![](images/474989e3c7a7539d7514f4abdf3ce18dfc865aff080717e9f7fbd2731127a450.jpg)  
(a) n = 200

![](images/03d360bc65eb493b61b1083f5026a6496fda00459dcc90081bef3006aeb84e56.jpg)  
(b) n = 300

![](images/7bd0ac353ac004d0f48d52ca2e8c9e0879871fb61f8cbf437b67934c5ebe340f.jpg)  
(c) n = 500  
Fig. 7. Learned sample relations on synthetic data with ten subspaces and increasing total sample count at epoch 500. Subspace boundaries are overlaid

On EYaleB, the diagonal blocks are already visible after 10 epochs and remain clearly identifiable at later epochs. The within-class regions contain more pronounced structure than most off-block regions. ORL also retains recognizable diagonal blocks despite having only ten images per subject. This visual organization is consistent with the strong face-clustering scores in Tables I–III. In particular, the displayed relations extend beyond individual self-connections and include associations between distinct samples from the same class.

The synthetic panels exhibit an even clearer separation between dense diagonal blocks and the surrounding regions. This organization persists across the training stages (i.e., 10, 100, and 500 epochs). Figure 7 compares n = 200, 300, and 500 at epoch 500. The 10 diagonal blocks remain identifiable as more samples are added, and the larger panels show a clearer visual contrast between within-subspace structure and off-block regions. Together with the quantitative connectivity results, these visualizations support the ability of the affinity matrix to retain structured sample relations as the dataset grows.

## V. CONCLUSION

This work presents LoomSC, a scalable subspace clustering framework that combines projector-based self-expression with exact spectral reduction. Jointly learning latent representations and two thin factors avoids explicit sample-by-sample coefficient matrices. A nonnegative quadratic affinity preserves the projector’s support and enables normalized spectral clustering without forming the full affinity or sample Laplacian. For fixed dimensions and iteration budgets, the complete pipeline has linear time and memory complexity in the number of samples. Our analysis quantifies the projector approximation to regularized least squares. It also identifies conditions under which the full row-space projector preserves subspace memberships and ensures within-subspace connectivity.

Across five image-clustering benchmarks, LoomSC achieves the highest average ACC, NMI, and ARI among the evaluated methods. Synthetic experiments extend to 500,000 samples while maintaining at least 99.8% accuracy throughout the tested range. These findings demonstrate that compact self-expression can combine strong clustering quality with scalability across both representation learning and spectral assignment.

## REFERENCES

[1] Y. Ma, H. Derksen, W. Hong, and J. Wright, “Segmentation of multivariate mixed data via lossy data coding and compression,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 29, no. 9, pp. 1546–1562, 2007.

[2] S. Rao, R. Tron, R. Vidal, and Y. Ma, “Motion segmentation in the presence of outlying, incomplete, or corrupted trajectories,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 32, no. 10, pp. 1832–1845, 2010.

[3] R. Vidal, “Subspace clustering,” IEEE Signal Process. Mag., vol. 28, no. 2, pp. 52–68, 2011.

[4] T. Ding, D. Lim, R. Vidal, and B. D. Haeffele, “Understanding doubly stochastic clustering,” in Proc. Int. Conf. Mach. Learn. (ICML), ser. Proceedings of Machine Learning Research, vol. 162, 2022, pp. 5153– 5165.

[5] R. Basri and D. W. Jacobs, “Lambertian reflectance and linear subspaces,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 25, no. 2, pp. 218–233, 2003.

[6] C. Lu, J. Feng, Z. Lin, T. Mei, and S. Yan, “Subspace clustering by block diagonal representation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 2, pp. 487–501, 2019.

[7] E. Elhamifar and R. Vidal, “Sparse subspace clustering,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2009, pp. 2790–2797.

[8] ——, “Sparse subspace clustering: Algorithm, theory, and applications,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 35, no. 11, pp. 2765–2781, 2013.

[9] C.-Y. Lu, H. Min, Z.-Q. Zhao, L. Zhu, D.-S. Huang, and S. Yan, “Robust and efficient subspace segmentation via least squares regression,” in Eur. Conf. Comput. Vis. (ECCV), ser. Lecture Notes in Computer Science, vol. 7578, 2012, pp. 347–360.

[10] G. Liu, Z. Lin, and Y. Yu, “Robust subspace segmentation by low-rank representation,” in Proc. Int. Conf. Mach. Learn. (ICML), 2010, pp. 663–670.

[11] D. Luo, F. Nie, C. Ding, and H. Huang, “Multi-subspace representation and discovery,” in Machine Learning and Knowledge Discovery in Databases, ser. Lecture Notes in Computer Science, vol. 6912. Springer, 2011, pp. 405–420.

[12] B. Nasihatkon and R. Hartley, “Graph connectivity in sparse subspace clustering,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2011, pp. 2137–2144.

[13] X. Peng, L. Zhang, and Z. Yi, “Scalable sparse subspace clustering,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2013, pp. 430–437.

[14] X. Peng, H. Tang, L. Zhang, Z. Yi, and S. Xiao, “A unified framework for representation-based subspace clustering of out-of-sample and largescale data,” IEEE Trans. Neural Netw. Learn. Syst., vol. 27, no. 12, pp. 2499–2512, 2016.

[15] Z. Kang, W. Zhou, Z. Zhao, J. Shao, M. Han, and Z. Xu, “Large-scale multi-view subspace clustering in linear time,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 34, no. 4, 2020, pp. 4412–4419.

[16] Z. Kang, Z. Lin, X. Zhu, and W. Xu, “Structured graph learning for scalable subspace clustering: From single view to multiview,” IEEE Trans. Cybern., vol. 52, no. 9, pp. 8976–8986, 2022.

[17] S. Wang, B. Tu, C. Xu, and Z. Zhang, “Exact subspace clustering in linear time,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 28, no. 1, 2014, pp. 2113–2120.

[18] C. You, D. P. Robinson, and R. Vidal, “Scalable sparse subspace clustering by orthogonal matching pursuit,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 3918–3927.

[19] C. You, C.-G. Li, D. P. Robinson, and R. Vidal, “Oracle based active set algorithm for scalable elastic net subspace clustering,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 3928–3937.

[20] Y. Chen, C.-G. Li, and C. You, “Stochastic sparse subspace clustering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 4155–4164.

[21] A. Adler, M. Elad, and Y. Hel-Or, “Linear-time subspace clustering via bipartite graph modeling,” IEEE Trans. Neural Netw. Learn. Syst., vol. 26, no. 10, pp. 2234–2246, 2015.

[22] D. Lim, R. Vidal, and B. D. Haeffele, “Doubly stochastic subspace clustering,” arXiv:2011.14859, 2020.

[23] S. Zhang, C. You, R. Vidal, and C.-G. Li, “Learning a self-expressive network for subspace clustering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 12 393–12 403.

[24] Z. Kang, H. Xu, B. Wang, H. Zhu, and Z. Xu, “Clustering with similarity preserving,” Neurocomputing, vol. 365, pp. 211–218, 2019.

[25] P. Ji, T. Zhang, H. Li, M. Salzmann, and I. Reid, “Deep subspace clustering networks,” in Adv. Neural Inf. Process. Syst. (NeurIPs), vol. 30, 2017, pp. 24–33.

[26] J. Zhang, C.-G. Li, C. You, X. Qi, H. Zhang, J. Guo, and Z. Lin, “Self-supervised convolutional subspace clustering network,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 5473–5482.

[27] J. Lv, Z. Kang, X. Lu, and Z. Xu, “Pseudo-supervised deep subspace clustering,” IEEE Trans. Image Process., vol. 30, pp. 5252–5263, 2021.

[28] X. Meng, Z. Huang, W. He, X. Qi, R. Xiao, and C.-G. Li, “Exploring a principled framework for deep subspace clustering,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[29] S. Matsushima and M. Brbic, “Selective sampling-based scalable sparse subspace clustering,” in Adv. Neural Inf. Process. Syst. (NeurIPs), vol. 32, 2019.

[30] C. Fettal, L. Labiod, and M. Nadif, “Scalable attributed-graph subspace clustering,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 37, no. 6, 2023, pp. 7559–7567.

[31] D. Arthur and S. Vassilvitskii, “k-means++: The advantages of careful seeding,” in Proc. 18th Annu. ACM-SIAM Symp. Discrete Algorithms (SODA), 2007, pp. 1027–1035.

[32] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.

[33] A. Y. Ng, M. I. Jordan, and Y. Weiss, “On spectral clustering: Analysis and an algorithm,” in Adv. Neural Inf. Process. Syst. (NeurIPs), vol. 14, 2001, pp. 849–856.

[34] Y. Chen, C.-G. Li, and C. You, “Stochastic sparse subspace clustering,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 4154–4163.

[35] N. Mrabah, M. Bouguessa, and S. Sami, “Scalable deep subspace clustering network,” in IEEE International Conference on Data Science and Advanced Analytics (DSAA), 2025.

## APPENDIX A

SPECTRAL ANALYSIS OF THE LSR APPROXIMATION

The normal equation of (2) is

$$
( X ^ { \top } X + 2 \lambda I _ { n } ) C = X ^ { \top } X .\tag{38}
$$

The coefficient matrix is positive definite for $\lambda > 0 ,$ , so the solution is unique. Since $\bar { X ^ { \top } } X = V _ { x } \mathrm { d i a g } ( \beta _ { 1 } , \dots , \beta _ { r } ) V _ { x } ^ { \top }$ , the vectors in $V _ { x }$ diagonalize the solution on the row space of X. Along its jth eigenvector direction, the normal equation becomes $( \beta _ { j } + 2 \lambda ) \tau _ { j } = \beta _ { j }$ . In the orthogonal complement, the solution has an eigenvalue of zero. This proves Proposition 1.

The LSR solution and $C _ { \mathrm { r o w } }$ have the same eigenvectors on the row space. The invariance of the Frobenius norm under orthonormal changes of coordinates therefore gives

$$
\| C _ { \mathrm { L S R } } ^ { \star } - C _ { \mathrm { r o w } } \| _ { F } ^ { 2 } = \sum _ { j = 1 } ^ { r } ( 1 - \tau _ { j } ) ^ { 2 } ,\tag{39}
$$

which proves (6). For the rank-m projector $V _ { x , m } V _ { x , m } ^ { \top }$ , the first m eigenvalues are one and the remaining eigenvalues are zero. Splitting the squared difference over these two sets proves (7). These identities quantify the spectral approximation used to motivate the factor model.

## APPENDIX B PROOFS FOR THE FACTOR MODEL

For fixed $Z \in \mathbb { R } ^ { d _ { z } \times n }$ and feasible $P \in \mathbb { R } ^ { n \times m }$ , decompose the residual as

$$
Z - L P ^ { \top } = Z ( I _ { n } - P P ^ { \top } ) + ( Z P - L ) P ^ { \top } .\tag{40}
$$

The two terms are orthogonal in the Frobenius inner product because $( I _ { n } - P P ^ { \top } ) P = 0$ . Moreover, $\| ( Z P - L ) P ^ { \top } \| _ { F } ^ { 2 } =$ $\| Z P - { \dot { L } } \| _ { F } ^ { 2 }$ since $\begin{array} { r l } { P ^ { \top } P \ = } & { { } I _ { m } } \end{array}$ . Taking squared norms proves (14). The second term in that identity is strictly convex in $L .$ . Thus, $L = Z P$ is its unique minimizer. The equality $L P ^ { \top } = Z P P ^ { \top }$ holds after this update.

For fixed $L ,$ expansion gives (20). Let $Z ^ { \top } L = U _ { P } \Sigma _ { P } V _ { P } ^ { \top }$ have the dimensions specified in (21). The trace inequality yields

$$
\mathrm { t r } ( P ^ { \top } Z ^ { \top } L ) \leq \mathrm { t r } ( \Sigma _ { P } ) .\tag{41}
$$

The feasible matrix $P = U _ { P } V _ { P } ^ { \top }$ attains equality. This proves the global optimality of the Procrustes update. When $\bar { Z } ^ { \top } L$ is rank deficient, zero singular directions must be included so that both singular bases have m columns. Different orthonormal completions can yield different minimizers.

To prove Proposition 2, eliminate L and expand the remaining objective:

$$
\| Z - Z P P ^ { \top } \| _ { F } ^ { 2 } = \| Z \| _ { F } ^ { 2 } - \operatorname { t r } ( P ^ { \top } Z ^ { \top } Z P ) .\tag{42}
$$

The variational characterization of eigenvalues maximizes the trace over an m-dimensional dominant eigenspace of $Z ^ { \top } Z$ Its eigenvalues are $\sigma _ { 1 } ^ { 2 } , \ldots , \sigma _ { r _ { z } } ^ { 2 }$ , followed by zeros. Subtracting their largest m values from $\| Z \| _ { F } ^ { 2 }$ gives (16). When $m < r _ { z } ,$ a spectral tie at the truncation boundary can make the optimal subspace nonunique.

When $m \ = \ r _ { z }$ , the minimum residual is zero. Hence, every global minimizer satisfies $Z P P ^ { \top } = Z$ , which implies $\mathrm { r a n g e } ( Z ^ { \top } ) \subseteq \mathrm { r a n g e } ( P )$ . Both spaces have dimension $r _ { z } ,$ so they coincide. Consequently, $\dot { C ^ { \mathrm { ~ } } } = P P ^ { \top } = V _ { z } V _ { z } ^ { \top } = Z ^ { \dagger } Z$ is the unique row-space projector, where † denotes the Moore– Penrose pseudoinverse. Its orthonormal basis P remains nonunique. If $m > r _ { z }$ , the same inclusion holds, but range(P) also contains $m - r _ { z }$ directions orthogonal to the row space. This proves the remaining assertions of Proposition 2.

For Proposition 3, each exact block update minimizes the same fixed-Z objective with its other block held constant. Neither update can increase that objective. Its successive values are nonnegative and nonincreasing, so they converge.

## APPENDIX C SUBSPACE PRESERVATION AND CONNECTIVITY

Reorder the samples by subspace and write $Z \quad =$ $[ Z _ { 1 } , \ldots , Z _ { k } ]$ , where $\begin{array} { r } { \overline { { \ u { Z _ { a } } } } \in \mathbb { R } ^ { d _ { z } \times \bar { n _ { a } } } , \sum _ { a = 1 } ^ { k } n _ { a } \ = \ n _ { \cdot } } \end{array}$ , and span $( Z _ { a } ) = S _ { a }$ . Under Proposition $4 , S _ { a }$ has dimension $r _ { a } > 0$ and the subspaces form a direct sum. Choose a basis matrix $\Omega _ { a } \in \mathbb { R } ^ { d _ { z } \times r _ { a } }$ and write $Z _ { a } = \Omega _ { a } \Xi _ { a }$ , where $\Xi _ { a } \in \mathbb { R } ^ { r _ { a } \times n _ { c } }$ has full row rank. Define

$$
\begin{array} { r l } & { \Omega = [ \Omega _ { 1 } , \dots , \Omega _ { k } ] \in \mathbb R ^ { d _ { z } \times r _ { z } } , } \\ & { \Xi = \mathrm { b l k d i a g } ( \Xi _ { 1 } , \dots , \Xi _ { k } ) \in \mathbb R ^ { r _ { z } \times n } . } \end{array}\tag{43}
$$

The direct-sum condition makes Ω full column rank. Therefore, $Z = \Omega \Xi { \mathrm { ~ a n d } } \Xi = \Omega ^ { \dagger } Z$ , which imply row $( Z ) = \operatorname { r o w } ( \Xi )$ . Their orthogonal row-space projectors coincide:

$$
\begin{array} { r l } & { Z ^ { \dagger } Z = \Xi ^ { \dagger } \Xi } \\ & { \qquad = \mathrm { b l k d i a g } ( Z _ { 1 } ^ { \dagger } Z _ { 1 } , \dots , Z _ { k } ^ { \dagger } Z _ { k } ) . } \end{array}\tag{44}
$$

The second equality uses the block structure of $\Xi$ and row $\mathbf { \partial } ( \Xi _ { a } ) \ = \ \operatorname { r o w } ( Z _ { a } )$ . At a global fixed-Z optimum with $m = r _ { z }$ , Appendix B gives $C = Z ^ { \dagger } Z$ . Equation (44) therefore proves subspace preservation.

For connectivity, the stated sample condition can be written

$$
\mathrm { r a n k } ( Z _ { \mathcal { A } } ) + \mathrm { r a n k } ( Z _ { \mathcal { B } } ) > r _ { a }\tag{45}
$$

for every partition of a class’s indices into two nonempty sets A and B. Here $Z _ { A }$ and $Z _ { B }$ contain the corresponding columns of $Z _ { a }$ . Indeed, their spans sum to $S _ { a }$ . Equality of their rank sum with $r _ { a }$ is equivalent to those spans forming a direct sum.

Set $\begin{array} { r c l } { { C _ { a } } } & { { = } } & { { Z _ { a } ^ { \dagger } Z _ { a } } } \end{array}$ . Suppose the graph defined by its nonzero off-diagonal entries is disconnected. There is a nontrivial partition A, B for which $C _ { a }$ is block diagonal. Since row $( C _ { a } ) \ = \ \operatorname { r o w } ( Z _ { a } )$ , this row space then splits into two spaces supported on the respective coordinate sets. Coordinate projection onto A has dimension rank $( Z _ { A } )$ , which also equals the rank of the corresponding diagonal block of $C _ { a }$ The analogous equality holds for B. Adding the two block ranks gives rank $( Z _ { \mathcal { A } } ) + \mathrm { r a n k } ( Z _ { \mathcal { B } } ) = r _ { a } ,$ contradicting (45). Conversely, equality in this rank sum gives a direct-sum partition. Applying (44) to its two groups makes $C _ { a }$ block diagonal. Thus, the sample condition is also necessary for connectivity of the full row-space projector graph. A singleton containing a nonzero sample has a connected one-vertex graph and satisfies the condition vacuously.

Every diagonal entry of $C _ { a }$ is positive when all samples are nonzero. A zero diagonal in an orthogonal projector forces the corresponding row and column to vanish. The identity $Z _ { a } = Z _ { a } C _ { a }$ would then make the corresponding sample zero, a contradiction. Positive row rescaling and entrywise squaring preserve the off-diagonal support. The affinity specified in the method therefore has exactly k connected components under these assumptions.

The eigenspace of eigenvalue one of its normalized affinity is spanned by the degree-weighted component indicators. Any orthonormal basis of this eigenspace differs from the normalized indicator basis by an orthogonal rotation. Row normalization maps every sample of one component to the same direction. Distinct components yield distinct directions. Consequently, the ideal embedding admits zero-distortion kmeans assignments equal to the components, up to a label permutation.

## APPENDIX D

## PROPERTIES OF THE QUADRATIC AFFINITY AND SPECTRALREDUCTION

a) Affinity and degrees: The definition of svec gives $\operatorname { s v e c } ( R ) ^ { \top } \operatorname { s v e c } ( S ) = \operatorname { t r } ( R ^ { \top } S )$ for symmetric $R , S \in \mathbb { R } ^ { \bar { m } \times m }$ Applying this identity to the rank-one matrices $p _ { i } p _ { i } ^ { \top }$ and $p _ { j } p _ { j } ^ { \intercal }$ proves (26). Hence A is positive semidefinite. Its entrywise nonnegativity follows from (24). Let $\boldsymbol { e } _ { i } ~ \in ~ \mathbb { R } ^ { n }$ be the ith standard basis vector. Since $C ^ { 2 } = C$

$$
\begin{array} { c } { \displaystyle \delta _ { i } = e _ { i } ^ { \top } C C ^ { \top } e _ { i } = e _ { i } ^ { \top } C e _ { i } = \| p _ { i } \| _ { 2 } ^ { 2 } , } \\ { \displaystyle \sum _ { i } \delta _ { i } = \mathrm { t r } ( C ) = m . } \end{array}\tag{46}
$$

The diagonal entries of an orthogonal projector belong to $[ 0 , 1 ]$ so $0 \leq \delta _ { i } \leq 1$ . Moreover, $\delta _ { i } = 0$ if and only if $p _ { i } = 0$ , which makes the entire corresponding affinity row zero. Removing these rows, therefore, leaves the degrees of all active samples unchanged.

b) Proof of Proposition 5: Let $s = { \mathrm { r a n k } } ( F _ { + } )$ . Choose s rows of $F _ { + }$ spanning its row space. The corresponding s rank-one matrices span all matrices $p _ { i } p _ { i } ^ { \intercal }$ . In particular, their span contains

$$
\sum _ { i \in \mathbb { Z } _ { + } } p _ { i } p _ { i } ^ { \top } = P ^ { \top } P = I _ { m } .\tag{47}
$$

A linear combination of s rank-one matrices has rank at most s. Thus, $m \leq s .$ . The upper bound follows from the dimensions of $F _ { + }$ . Since ∆ is positive definite, B and $F _ { + }$ have the same rank.

If $G v = \nu v$ for $\nu > 0 ,$ then $( B B ^ { \top } ) ( B v ) = B ( G v ) = \nu B v$ and $\| B v \| _ { 2 } ^ { 2 } = \nu \| v \| _ { 2 } ^ { 2 }$ . Conversely, a positive eigenpair of $B B ^ { \top }$ maps to an eigenpair of G by multiplication by $B ^ { \top }$ . For orthonormal eigenvectors,

$$
\frac { \boldsymbol { v } _ { a } ^ { \top } \boldsymbol { B } ^ { \top } \boldsymbol { B } \boldsymbol { v } _ { b } } { \sqrt { \nu _ { a } \nu _ { b } } } = 1 \{ a = b \} .\tag{48}
$$

This proves the correspondence, multiplicities, and orthonormality.

c) Leading vector and spectral convention: Let $\omega =$ $\mathrm { s v e c } ( I _ { m } )$ . We have $F _ { + } ^ { \top } 1 _ { n _ { + } } ~ = ~ \omega , ~ F _ { + } \omega ~ = ~ ( \delta _ { i } ) _ { i \in \mathbb { Z } _ { + } }$ , and

$\| \boldsymbol { \omega } \| _ { 2 } ^ { 2 } = m$ . Therefore,

$$
\begin{array} { r } { G \omega = F _ { + } ^ { \top } \Delta ^ { - 1 } F _ { + } \omega = F _ { + } ^ { \top } \mathbf { 1 } _ { n _ { + } } = \omega . } \end{array}\tag{49}
$$

This proves (30). Let $A _ { + + } ~ \in ~ \mathbb { R } ^ { n _ { + } \times n _ { + } }$ be the principal submatrix of A indexed by $\mathcal { T } _ { + }$ . The normalized affinity $B B ^ { \bar { \top } }$ is similar to the row-stochastic matrix $\Delta ^ { - 1 } A _ { + + }$ . Its eigenvalues have a magnitude of at most one. Positive semidefiniteness further restricts them to [0, 1]. Thus, the retained eigenvalue is dominant. Locking this vector remains valid when the eigenvalue one is repeated. Every active row then contains the strictly positive coordinate $\sqrt { \delta _ { i } / m } ,$ , which justifies the row normalization in (31).

Proposition 6 (Conditional recovery on the active graph). Suppose $A _ { + + }$ has exactly k connected components. The row-normalized embedding in (31), computed from exact eigenvectors, has one distinct row per component. An exact solution of k-means therefore recovers those components up to label permutation.

Proof. Let $\Upsilon \in \{ 0 , 1 \} ^ { n _ { + } \times k }$ be the component membership matrix. The eigenvalue-one eigenspace of $\bar { B } B ^ { \top }$ has dimension k and an orthonormal basis

$$
\Delta ^ { 1 / 2 } \Upsilon ( \Upsilon ^ { \top } \Delta \Upsilon ) ^ { - 1 / 2 } O , \qquad O ^ { \top } O = I _ { k } .\tag{50}
$$

The diagonal matrix $\Upsilon ^ { \top } \Delta \Upsilon$ contains the positive component volumes. Within component $^ { a , }$ each row of this basis is a positive scalar multiple of the ath row of $O .$ Row normalization removes that scalar. The k resulting rows are distinct because the rows of $O$ are orthonormal. Clustering these rows into k nonempty groups has minimum within-cluster sum of squares zero, attained exactly by the component partition up to permutation. □

d) Implicit reduced operator: Using (25) and the Frobenius inner-product identity, for each original index $i \in \mathcal { Z } _ { + }$ , the corresponding row of $F _ { + } v$ equals $p _ { i } ^ { \top }$ smat(v)p<sub>i</sub>. Multiplying by $F _ { + } ^ { \dagger } \Delta ^ { - 1 }$ yields (33). This proves the implicit operator identity.

## APPENDIX E IMPLEMENTATION CONVENTIONS

The implementation specifies the network architecture, preprocessing, initialization, random seeds, Adam settings, and the iteration budgets in Algorithm 1. Each training update follows the stated order. When gradients are accumulated over sample chunks, their contributions in (19) are summed before the parameter update. Joint training starts with zero Adam moments.

a) Sample-based initialization: For k-means++ seeding of $L ,$ select the first index uniformly from $\{ 1 , \ldots , n \}$ . Let S be the indices already selected. For each remaining index, compute its squared distance to the closest selected column,

$$
d _ { i } ^ { \mathrm { s e e d } } = \operatorname* { m i n } _ { j \in \mathcal { S } } \| z _ { i } - z _ { j } \| _ { 2 } ^ { 2 } , \qquad \operatorname* { P r } ( i \mid \mathcal { S } ) = \frac { d _ { i } ^ { \mathrm { s e e d } } } { \sum _ { j \not \in \mathcal { S } } d _ { j } ^ { \mathrm { s e e d } } } .\tag{51}
$$

Use this distribution until m distinct indices have been selected. When its denominator is zero, sample uniformly from the remaining indices. Put the selected columns into L in selection order. This is the squared-distance seeding rule [31].

b) Adam update: For completeness, let $w _ { t }$ be the vector containing the current network parameters and let $g _ { t }$ be its gradient of the specified objective. These are vectorizations used only to define the optimizer. Let $a _ { t }$ and $b _ { t }$ be its firstand second-moment vectors. With $a _ { 0 } = b _ { 0 } = 0$ , coefficient inputs $0 \le \rho _ { 1 } , \rho _ { 2 } < 1$ , step size $\eta _ { t } > 0$ , and $\epsilon _ { \mathrm { A d a m } } > 0$ , one update is

$$
\begin{array} { r l } & { a _ { t } = \rho _ { 1 } a _ { t - 1 } + ( 1 - \rho _ { 1 } ) g _ { t } , } \\ & { b _ { t } = \rho _ { 2 } b _ { t - 1 } + ( 1 - \rho _ { 2 } ) ( g _ { t } \odot g _ { t } ) , } \\ & { \widehat { a } _ { t } = a _ { t } / ( 1 - \rho _ { 1 } ^ { t } ) , ~ \widehat { b } _ { t } = b _ { t } / ( 1 - \rho _ { 2 } ^ { t } ) , } \\ & { w _ { t + 1 } = w _ { t } - \eta _ { t } \widehat { a } _ { t } \oslash ( \sqrt { \widehat { b } _ { t } } + \epsilon _ { \mathrm { A d a m } } ) . } \end{array}\tag{52}
$$

Products $\odot ,$ divisions $\oslash ,$ and the square root are elementwise. These equations specify Adam [32], without an added weightdecay penalty. For pretraining, g<sub>t</sub> differentiates (13); for joint training, it differentiates (17) with fixed factors. The moment vectors have the same dimension as $w _ { t }$

c) Factor accuracy: The thin decomposition in (21) uses m orthonormal columns, completing the singular bases when zero singular values occur. After each factor update, we check the residual $\| P ^ { \top } P - I _ { m } \| _ { F }$ at the chosen working precision.

d) Spectral accuracy and reproducibility: The eigensolver retains the analytical mode in (30) and computes the remaining dominant modes on its orthogonal complement. For prescribed residual, orthogonality, and positive-eigenvalue tolerances $\varepsilon _ { \mathrm { e i g } } > 0 , \epsilon _ { \mathrm { o r t h } } > 0 .$ , and $\varepsilon _ { \mathrm { r a n k } } > 0$ , respectively, we accept computed eigenpairs satisfying

$$
\begin{array} { r l r } & { } & { \underset { 1 \leq j \leq k } { \operatorname* { m a x } } \| G v _ { j } - \nu _ { j } v _ { j } \| _ { 2 } \leq \varepsilon _ { \mathrm { e i g } } , } \\ & { } & { \underset { 1 \leq a , b \leq k } { \operatorname* { m a x } } | v _ { a } ^ { \top } v _ { b } - \mathbf { 1 } \{ a = b \} | \leq \epsilon _ { \mathrm { o r t h } } , } \\ & { } & { \nu _ { k } > \varepsilon _ { \mathrm { r a n k } } . } \end{array}\tag{53}
$$

The iteration budget or working precision is increased as needed to meet these criteria. Eigenvector ordering and random seeds are recorded for reproducibility. When an eigenvalue is repeated at the truncation boundary, the solver fixes a reproducible orthonormal basis of the selected modes. The active set contains every strictly positive degree in (27). Rows with zero degree are assigned by (32).

e) Final clustering: Let ι(i) denote the row of $F _ { + }$ corresponding to the original sample index $i \in \mathcal { T } _ { + }$ . The same ordering is used in H and $\widehat { H }$ . With spectral centers $\boldsymbol { \mu _ { a } } \in \mathbb { R } ^ { k }$ the clustering objective and Lloyd updates are

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \{ \gamma _ { i } \} , \{ \mu _ { a } \} } } & { \displaystyle \sum _ { i \in \mathcal { Z } _ { + } } \| \widehat { H } _ { \iota ( i ) : } ^ { \top } - \mu _ { \gamma _ { i } } \| _ { 2 } ^ { 2 } , } \\ { \displaystyle \gamma _ { i }  \arg \operatorname* { m i n } _ { 1 \leq a \leq k } \| \widehat { H } _ { \iota ( i ) : } ^ { \top } - \mu _ { a } \| _ { 2 } ^ { 2 } , } \\ { \displaystyle \mu _ { a }  \frac { 1 } { | \mathcal I _ { a } | } \sum _ { i \in \mathcal I _ { a } } \widehat { H } _ { \iota ( i ) : } ^ { \top } . } \end{array}\tag{54}
$$

Use $R _ { \mathrm { k m } } ~ \ge ~ 1$ recorded k-means++ restarts on these rows. Apply the seeding rule (51) to the spectral rows, selecting k seeds among the $n _ { + }$ active samples. Each restart takes at most $T _ { \mathrm { k m } }$ Lloyd steps. Ties between centers use the smallest center index. Repair an empty cluster before its mean update by moving the sample with the largest current squared residual from a cluster containing at least two samples, with ties resolved by the original sample index. Then recompute affected centers. Stop early when assignments do not change. Keep the restart with minimum within-cluster sum of squares. This procedure returns k nonempty groups for $n _ { + } \geq k ,$ as ensured by (29).

## APPENDIX FCOMPLEXITY COMPARISON WITH BASELINES

Table V separates representation learning from spectral computation. The bounds count sequential arithmetic operations. The numbers of training passes and optimization iterations are held fixed to isolate sample-size scaling. Eigensolver budgets are retained explicitly.

To compare dense, sparse, and implicit spectral operators on the same basis, define

$$
\begin{array} { r } { \mathcal { C } _ { \mathrm { s p } } ( d _ { \mathrm { o p } } , \omega ) = T _ { \mathrm { e i g } } ( \omega + d _ { \mathrm { o p } } s _ { \mathrm { e i g } } ) + R _ { \mathrm { e i g } } ( d _ { \mathrm { o p } } s _ { \mathrm { e i g } } ^ { 2 } + s _ { \mathrm { e i g } } ^ { 3 } ) . } \end{array}\tag{55}
$$

Here, $d _ { \mathrm { o p } }$ is the operator dimension and $\omega$ bounds the cost of one matrix–vector product. The solver-specific budgets $T _ { \mathrm { e i g } } ,$ $s _ { \mathrm { e i g } } ,$ , and $R _ { \mathrm { e i g } }$ have the meanings given in Section III-G. For an explicitly stored sparse graph, ω is its number of nonzero entries. A dense graph instead has $\omega = O ( n ^ { 2 } )$ . Thus, partial eigensolvers retain the benefit of sparse affinities. The common final k-means cost, $O ( R _ { \mathrm { k m } } ( T _ { \mathrm { k m } } + 1 ) n k ^ { 2 } )$ , is added once to the sum of the two time columns for the complete pipeline.

PRO-DSC [28] trains on mini-batches and constructs its final affinity over the complete clustering set. Combining its regularizer costs with neural computation and matrix products gives the following per-batch computational-cost bound:

$$
\begin{array} { r l } & { c _ { \mathrm { P R O } } ( n _ { b } ) = n _ { b } c _ { \mathrm { n e t } } + n _ { b } ^ { 2 } d _ { z } + \operatorname* { m i n } \{ n _ { b } ^ { 3 } , d _ { z } ^ { 3 } \} } \\ & { \qquad + \mathcal { C } _ { \mathrm { s p } } ( n _ { b } , n _ { b } ^ { 2 } ) . } \end{array}\tag{56}
$$

For $1 \ \leq \ n _ { b } \ \leq \ n ,$ the $n / n _ { b }$ multiplier counts batches in a full training pass. A fixed number of Sinkhorn iterations is absorbed into the matrix-product term. The separate $n ^ { 2 } d _ { z }$ term accounts for constructing the full-set affinity at inference. The per-sample network cost $c _ { \mathrm { n e t } }$ is evaluated for each method’s own architecture.

The regression-based methods differ in how they reduce optimization cost. $\mathrm { \bf S ^ { 5 } C }$ [29] uses a selection budget of m and solves the final restricted regression for every sample. Its bound includes both selection and regression costs. For EnSC [19], the active-set solver is left explicit because the original algorithm permits different restricted solvers. A-DSSC [22] likewise permits different coefficient estimators. Each support-restricted transport evaluation is linear in $e _ { \mathrm { a c t } } . \mathrm { ~ A ~ }$ complete support check additionally costs $O ( n ^ { 2 } )$ with stored coefficients, or $O ( d _ { x } n ^ { 2 } )$ when feature-space LSR entries are evaluated on demand. For LSR [22] with $d _ { x } < n ,$ a feature-space solve reduces coefficient construction to $O ( n d _ { x } ^ { 2 } + d _ { x } ^ { 3 } + n ^ { 2 } d _ { x } )$ . Explicit assembly and storage of the full coefficient matrix remain quadratic in n.

The matrix-storage column isolates the representation responsible for affinity scaling. Total working memory also includes the input data, spectral embedding, eigensolver workspace, and regression-solver workspace. Deep methods additionally retain latent features and network state. For LoomSC, these terms are specified in (37), with the corresponding eigensolver workspace. Quadratic feature rows can be generated as needed, so an $n \times q$ feature matrix need not be stored.

TABLE V  
TIME AND REPRESENTATION-STORAGE COMPLEXITY OF THE COMPARED METHODS. ALL ENTRIES ARE INSIDE $O ( \cdot )$ . THE TWO TIME COLUMNS ARE ADDITIVE, WITH THE COMMON FINAL k-MEANS COST ADDED ONCE. FIXED OPTIMIZATION BUDGETS ARE SUPPRESSED. SPECTRAL BUDGETS ARE INCLUDED THROUGH (55). THE LAST COLUMN COUNTS THE PRINCIPAL COEFFICIENT, FACTOR, AND AFFINITY MATRICES, NOT TOTAL WORKING MEMORY. BOUNDS ARE OBTAINED FROM THE ORIGINAL ALGORITHMS UNDER THE IMPLEMENTATION CONVENTIONS STATED BELOW.
<table><tr><td>Method</td><td>Deep</td><td>Representation / affinity learning</td><td>Spectral computation</td><td>Matrix storage</td></tr><tr><td>LMVSC [15]</td><td>No</td><td> $n d _ { x } m + n m ^ { 3 }$ </td><td> $n m ^ { 2 } + m ^ { 3 }$ </td><td>nm</td></tr><tr><td>SGL [16]</td><td>No</td><td> $n d _ { x } m + n m ^ { 3 }$ </td><td> $n m ^ { 2 } + m ^ { 3 \mathrm { a } }$ </td><td>nm</td></tr><tr><td> $\mathrm { s } ^ { 5 } \mathrm { C }$  [29]</td><td>No</td><td> $n ( d _ { x } m + c _ { \ell _ { 1 } } )$ </td><td> $\mathcal { C } _ { \mathrm { s p } } ( n , n m )$ </td><td>nm</td></tr><tr><td>EnSC [19]</td><td>No</td><td> $n ^ { 2 } d _ { x } + n c _ { \mathrm { E N } }$ </td><td> $\mathcal { C } _ { \mathrm { s p } } ( n , n s )$ </td><td>ns</td></tr><tr><td>SSC-OMP [18]</td><td>No</td><td> $n ^ { 2 } d _ { x } s$ </td><td> $\mathcal { C } _ { \mathrm { s p } } ( n , n s )$ </td><td>ns</td></tr><tr><td>A-DSSC [22]</td><td>No</td><td> $\mathcal { C } _ { C } + \mathcal { C } _ { \mathrm { c h k } } + e _ { \mathrm { a c t } }$ </td><td> $\mathcal { C } _ { \mathrm { s p } } ( n , e _ { \mathrm { a c t } } )$ </td><td> $\mathcal { M } _ { C } + e _ { \mathrm { a c t } }$ </td></tr><tr><td>SSSC [20]</td><td>No</td><td> $R _ { \mathrm { d r o p } } \rho n ^ { 2 } d _ { x } s$ </td><td> $\mathcal { C } _ { \mathrm { s p } } ( n , e _ { \mathrm { d r o p } } )$ </td><td>edrop</td></tr><tr><td>LSR [9]</td><td>No</td><td> $n ^ { 2 } d _ { x } + \operatorname* { m i n } \{ n ^ { 3 } , n d _ { x } ^ { 2 } + d _ { x } ^ { 3 } \} ^ { \mathrm { b } }$ </td><td> $\mathcal { C } _ { \mathrm { s p } } ( n , n ^ { 2 } )$ </td><td> $n ^ { 2 }$ </td></tr><tr><td>PRO-DSC [28]</td><td>Yes</td><td> $( n / n _ { b } ) c _ { \mathrm { P R O } } ( n _ { b } )$ </td><td> $\mathcal { C } _ { \mathrm { s p } } ( n , n ^ { 2 } )$ </td><td> $n ^ { 2 }$ </td></tr><tr><td>LoomSC</td><td>Yes</td><td> $+ n c _ { \mathrm { n e t } } + n ^ { 2 } d _ { z }$   $n c _ { \mathrm { n { e t } } } + n d _ { z } m + n m ^ { 2 } + m ^ { 3 }$ </td><td> $n m + \mathcal { C } _ { \mathrm { s p } } ( q , n q )$   $+ \it { n q k }$ </td><td> $n m + d _ { z } m$ </td></tr></table>

For LoomSC, $q = m ( m + 1 ) / 2 .$ The table reports spectral computation using the operator in (33), without assembling G. s bounds the nonzeros per sampl for EnSC and SSC-OMP, and per dropout subproblem for SSSC. For ${ \mathrm { S S S C } } , \bar { \rho }$ is the expected retained dictionary fraction and $R _ { \mathrm { d r o p } }$ is the number of dropout dictionaries, giving $e _ { \mathrm { d r o p } } = \bar { \operatorname* { m i n } } \{ n ^ { 2 } , n s R _ { \mathrm { d r o p } } \}$ . The SSSC row gives the expected sequential computational cost at a fixed consensus budget. $c _ { \ell _ { 1 } }$ and c<sub>EN</sub> are the costs of one restricted Lasso and elastic-net solve, respectively. The former uses at most m representatives; the latter depends on the active-set size. For $\mathbf { A } – \mathbf { D } \mathbf { S } \mathbf { S } \mathbf { C } , \mathcal { C } _ { C }$ and $\mathcal { M } _ { C }$ denote the chosen coefficient solver’s computational cost and matrix storage. $\mathcal { C } _ { \mathrm { c h k } }$ includes support initialization and global checks, and $e _ { \mathrm { a c t } }$ bounds the entries processed by its support-restricted transport solver. $n _ { b }$ is PRO-DSC’s training batch size, and c<sub>PRO</sub> is defined in (56). Anchor-method spectral costs use a direct thin SVD of the n × m factor. <sup>a</sup>SGL updates this representation within its graph-learning loop. <sup>b</sup>LSR uses the cheaper of the sample-space and feature-space linear solves. Both routes explicitly form the coefficient matrix.

The comparison identifies the source of LoomSC’s scalability advantage. LMVSC [15], SGL [16], and $\mathrm { \bf S ^ { 5 } C }$ [29] already have linear sample-size dependence for fixed representative and solver budgets. LoomSC retains this dependence while jointly learning nonlinear features and the two self-expression factors. EnSC [19] and SSC-OMP [18] can reduce coefficient storage through sparsity, but their dictionary scans still compare each target with the full dataset. SSSC reduces the dictionary size by dropout, giving the explicit $R _ { \mathrm { d r o p } } \rho n ^ { 2 }$ dependence in the table [20]. PRO-DSC [28] instead improves the training stage through mini-batches, while its full-affinity spectral inference retains quadratic storage. LoomSC avoids explicitly constructing either $n \times n$ matrix. Its q-dimensional reduction is exact for the defined squared affinity. For fixed m and solver budgets, both its learned factors and its reduced spectral computation scale linearly with n. Thus, LoomSC combines deep representation learning with scalability through the final clustering stage.