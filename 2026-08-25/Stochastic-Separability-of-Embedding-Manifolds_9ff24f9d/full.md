# Stochastic Separability of Embedding Manifolds

Liqing Zhang<sup>∗</sup> School of Computer Science Shanghai Jiao Tong University

August 25, 2026

## Abstract

Neurobiological studies and representation learning have observed that representations of objects belonging to the same category in high-dimensional neural spaces exhibit low-dimensional object manifold characteristics, and diferent object manifolds are linearly separable in these neural spaces. However, these experimentally observed phenomena lack rigorous theoretical validation to date.

This paper proposes a new stochastic separability theorem for embedding manifolds of two diferent object categories. First, we establish a projection measure concentration theorem for embedding manifolds under general conditions. We develop a new two-layer measure concentration analysis technique, which unifies two estimation bounds via the law of total expectation to derive measure concentration inequalities.

Based on the measure concentration theorem, we further prove a stochastic separability theorem for embedding manifolds of two diferent object categories. If two datasets have distinct means and bounded total variances, their samples become linearly separable with high probability, provided that the projection direction satisfies a non-singularity condition.

The main contributions of this paper are twofold:

1. We prove the projection concentration properties of embedding manifolds in high-dimensional spaces by using two-lawyer tail-bound inequalities.

2. We identify a non-singularity condition for the stochastic separability between embedding manifolds, and rigorously prove the stochastic projection separability theorem.

The theorem not only uncovers geometric and statistical properties of the object embedding manifolds, but also provides a novel mechanism for representation learning in deep networks.

Keywords: Projection Measure Concentration, Stochastic Embedding Manifolds, Stochastic Separability, Bounded Total Variance.

## 1 Introduction

In the era of deep learning and large foundation models, embedding multi-modal data into high-dimensional feature spaces has become a critical approach for machine learning modeling and inference. Although high-dimensional modeling confronts intricate challenges such as the curse of dimensionality, the interplay between randomness and distribution concentration in high-dimensional spaces can drastically simplify numerous high-dimensional machine learning problems—provided we relax error estimation requirements and only demand bounds to hold with high probability. This analytical paradigm greatly facilitates modeling and model evaluation for high-dimensional systems, such a positive efect of the high dimensionality is known as the blessing of dimensionality.

This idea was first proposed by Kainen [18], who noted that having many parameters actually facilitates computation and uncovered intrinsic correlations between randomness and geometric properties of high-dimensional variables. Gorban & Tyukin[13] and Donoho [8] further refined and extended error bound estimation techniques for high-dimensional space models.

High-dimensional measure concentration theory not only reveals statistical distributions and geometric structures of high-dimensional data, but also furnishes new tools for highdimensional modeling and generalization analysis. Its core modeling principle bears profound implications for understanding internal architectures and statistical regularities of deep learning models.

Measure concentration theory characterizes distribution properties of data following common distributions: uniform measures on unit balls/cubes, and Gaussian distributions. The concept of measure concentration originated with Milman [21, 22], who revisited Paul L´evy’s spherical isoperimetric inequalities within local Banach space theory and formalized spherical measure concentration as a universal mathematical phenomenon. Subsequent work by Ledoux et al. [20, 19] generalized proof strategies for uniform spherical concentration, refining concentration bounds via logarithmic Sobolev inequalities [16] and entropy methods [19].

Measure concentration on the unit sphere manifests in two key ways. First, nearly all volume mass of an n-dimensional unit ball $\mathbb { B } ^ { n }$ is concentrated within a very thin shell adjacent to the sphere’s surface, while the central core of the unit ball occupies a volume that vanishes exponentially. Second, mass of the unit ball accumulates in thin equatorial slices passing through the origin, as illustrated in Figure 1. For any fixed $\epsilon > 0$ , the fraction of mass contained within a $2 \epsilon { \mathrm { - } } \mathrm { t h i c k }$ slice converges to 1 as the dimension tends to infinity. This property is known as equatorial/waist concentration, which provides the foundation for proving the embedding manifold concentration in this work.

Diferent random variables exhibit disparate concentration behavior. In $\mathbb { R } ^ { n }$ , the Euclidean norm $\| X \| _ { 2 }$ of a standard Gaussian random vector concentrates tightly around ${ \sqrt { n } } ,$ , forming the so-called Gaussian annulus [26]:

$$
p \left( \left| \frac { \| X \| _ { 2 } } { \sqrt { n } } - 1 \right| \geqslant \epsilon \right) \leqslant 2 \exp \left( - c n \epsilon ^ { 2 } \right) , \quad \epsilon \in ( 0 , 1 ) ,\tag{1}
$$

where constant $c > 0$ and $X \sim { \mathcal { N } } ( 0 , I _ { n } )$

![](images/7c6f864c6a2fe182a74b0dd51bfe762e7b256905af57a611b987aaaef6b7aeb2.jpg)  
Figure 1: Illustration of equatorial/waist concentration: The mass of the unit ball concentrates in the thin layer of the blue cross-section

While isotropic Gaussian vectors concentrate within a spherical annulus, Lipschitz functions defined on Gaussian vectors concentrate around their expectation $\mathbb { E } [ f ( X ) ]$ , with tail behavior fully controllable—analogous to univariate Gaussian random variables with wellbehaved mean and variance bounds.

A complementary property tied to measure concentration is probabilistic convexity of random high-dimensional samples, formalized as the stochastic separation theorem by Gorban & Tyukin [12] in 2017. This theorem reveals a counterintuitive high-probability linear separability in high dimensional spaces: under uniform distribution assumptions, any single sample within a finite dataset can be linearly separated from all remaining samples with arbitrarily high probability when the ambient dimension is suficiently large.

Let $X \in \mathbb { R } ^ { n }$ denote an n-dimensional random vector with independent, centrally symmetric components and bounded second moments. Let $\mathcal { D } = \left\{ \pmb { x } _ { k } \right\} _ { k = 1 } ^ { N }$ denote N independent identically distributed (i.i.d.) samples of X. For arbitrarily small $\bar { \epsilon } \in ( 0 , 1 )$ , there exists a threshold dimension such that any single sample $\mathbfit { \Delta } \mathbfit { x } _ { k }$ can be Fisher-separated from the rest of the dataset with probability at least $1 - \epsilon .$ . Fisher separability requires the existence of a weight vector w satisfying $\langle { \pmb w } , { \pmb x } _ { k } \rangle > \mathrm { m a x } _ { i \neq k } \langle { \pmb w } , { \pmb x } _ { i } \rangle$ , equivalent to existence of a separating linear hyperplane for $\scriptstyle { \mathbf { { \mathit { x } } } } _ { k }$

Grechuk et al. (2021) [14] extended this result to broader distribution families and derived tight finite-dimensional separation probability bounds. Their work relaxes symmetry and sub-Gaussian constraints from the original theorem, generalizing stochastic separation to log-concave, mixture, and clustered non-independent datasets with computable upper/lower bounds on separation probability. The generalized stochastic separation theorem is applied to correct errors and analyze instabilities in artificial intelligence systems.

Stochastic separation theorems provide a number of potential applications in the field of machine learning, such as sample correction, intrinsic dimension estimation, and few-shot learning. Gorban et al. [10] [11] leveraged Fisher discriminants for one-shot learning of model correctors, with stochastic separability furnishing complete mathematical foundations for such frameworks. For fine-grained clustered data, A novel stochastic separation theorem is developed for designing a multi-corrector framework validated on deep convolutional neural network error correction and novel class incremental learning using the CIFAR-10 benchmark.

Albergante et al. [1] proposed an intrinsic dimension estimation algorithm using Fisher separability from stochastic separation theory, matching state-of-the-art performance across standard and real biological datasets with eficient computation and robustness to noisy samples. Tyukin et al. [25] investigated information processing properties of convergent neural pathways onto single neurons, indicating that via the measure concentration and the stochastic separation models, individual neurons can selectively encode, recognize, and memorize arbitrary information units through associative dynamic learning of new content.

Classical measure concentration theory typically assumes uniform supported on bounded domains or Gaussian distributions. Although stochastic separation theorems characterize probabilistic convexity and separability within a single dataset, they cannot be applied directly to separability between two distinct real-world datasets. To characterize separability between sample sets of two distinct object classes in the embedded feature spaces, we first analyze distribution properties of samples belonging to a single object category.

Samples of images from same-class objects do not follow uniform distributions in highdimensional embedding spaces. For computer vision, images of the same object share structural attributes; continuous pose variation induces continuous shifts in embedding representations, forming low-dimensional sub-manifolds within feature spaces. For example, neural population responses to images of cats versus dogs, each form distinct object manifolds, reformulating object recognition to a separation problem between pairs of embedding manifolds.

Since latent parameters governing same-category images are finite-dimensional, the intrinsic dimension of the embedding manifold of same-category object images remains bounded and does not grow to infinity with ambient feature space dimension. In addition, Noise and background interference prevent these distributions from forming perfect mathematical manifolds—instead, samples cluster within neighborhoods of the underlying manifold, which we term stochastic embedding manifolds.

Low-dimensional manifold embedding representations for same-category object images are not merely theoretical machine learning assumptions; they are substantiated by extensive neurobiological and systems biology experiments. Research across neuroscience, singlecell transcriptomics, and metabolomics confirms pervasive structural redundancy in highdimensional biological observational data, where functional states and temporal evolution are governed by a small set of latent variables, naturally inducing low-dimensional manifold distributions.

Within visual hierarchical systems, neuronal population responses evoked by identical objects under varying pose collectively form object manifolds. DiCarlo & Cox [6] integrated neurophysiology and computational theory to define object manifolds as continuous surfaces traced by neuronal responses to objects under translation, scaling, and pose variation in high-dimensional spaces. Each transformation layer within the ventral visual stream aims to disentangle object manifolds to enable rapid visual categorization, converting initially non-separable manifolds into linearly separable structures.

This embedding manifold representation is corroborated by studies of primary visual cortex and motor cortex neural representations [7, 23, 15, 17] , as well as deep neural network representation learning literature[24, 9, 2]. The perceptual manifold geometric frameworks [4, 5] quantify three measurable manifold attributes: intrinsic manifold dimension, manifold radius, and inter-manifold correlation. Separability between distinct perceptual manifolds depends jointly on intrinsic dimension, radius, and cross-manifold correlation—this theory provides analytical tools to explain manifold disentanglement and improved separability within deep network representations. Chung & Abbott [3] demonstrated perceptual manifold properties extend beyond visual stimulus encoding to motor and cognitive brain regions. They systematically investigated neural manifold methods across perceptual disentanglement, manifold classification theory, abstract encoding in cognitive systems, topological cognitive maps, and dynamic motor disentanglement, formalizing mathematical links between geometric representation properties and encoded task information.

In summary, neurobiological experiments have revealed that same-category objects induce low-dimensional manifold neural representations in high-dimensional deep brain spaces, with distinct object manifolds linearly separable at deeper layers. Deep embedding experiments similarly show that well-trained representation models produce embedding manifolds for separate object classes that become linearly separable with high probability, with the separation probability rising alongside the ambient feature dimensionality.

Despite consistent experimental evidence for low-dimensional embedding manifolds and high-probability separability between cross-class manifolds, rigorous theoretical proofs remain absent in the prior literature. This paper derives suficient projection concentration conditions for the embedding manifold of same-category objects and proves the projection measure concentration under general conditions. For two-class separability between stochastic embedding manifolds, we prove that if two datasets have distinct means and bounded total variances, their embedding manifolds become linearly separable with arbitrarily high probability when ambient feature dimension is suficiently large—provided projection directions satisfy a non-singularity criterion. We formalize this result as the embedding manifold stochastic separability theorem.

Although the theorem states that linear separability improves with embedding dimension, higher dimensionality does not universally imply superior model performance, which must be evaluated holistically via generalization and interpretability metrics. The stochastic separability theorem delivers a new mechanism for representation learning: minimize the intrinsic dimension of each object’s manifold during embedding learning. Additionally, highdimensional embeddings allow the deployment of simple linear classifiers for downstream categorization, reducing overfitting risks.

## 2 Projection Measure Concentration of Embedding Manifolds

High-dimensional embeddings are fundamental to deep learning and generative artificial intelligence. The curse of dimensionality renders most low-dimensional data algorithms computationally intractable in high dimensions, yet the unique geometric and statistical properties of high-dimensional spaces unlock novel analytical frameworks for statistical learning and inference. Randomness induced by high ambient dimensionality enables innovative methodologies for high-dimensional data analysis.

Let $X = ( x _ { 1 } , \cdot \cdot \cdot , x _ { n } ) ^ { \intercal } \in \mathbb { R } ^ { n }$ denote an n-dimensional random vector with independent components, mean $\pmb { \mu } _ { X }$ , and component variances $\{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { n }$ . As established in Section 1, samples of natural objects lie within neighborhoods of low-dimensional embedding manifolds whose intrinsic dimension remains finite and independent of ambient feature dimension. We characterize each stochastic embedding manifold via a probability density function $p _ { X } ( \pmb { x } )$ describing the distribution of samples drawn from the manifold. Bounded intrinsic dimensionality of a stochastic embedding manifold is captured via a total variance constraint on the n-dimensional random vector X: there exists a constant $C _ { 0 }$ such that

$$
\sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 2 } \leqslant C _ { 0 } , \quad \forall n .\tag{2}
$$

We term this the bounded total variance condition. Let $\mathcal { D } _ { X } = \left\{ \pmb { x } _ { k } \right\} _ { k = 1 } ^ { N }$ denotes N i.i.d. samples of X distributed over the stochastic embedding manifold. We use $\mathbb { N } _ { + }$ for positive integers and $\mathbb { S } ^ { n - 1 } \mathrm { t o }$ denote the unit n-dimensional sphere.

Theorem 1 (Projection Measure Concentration for Embedding Manifolds) Let X be an n-dimensional random vector with independent components, mean $\pmb { \mu } _ { X }$ , and bounded total variance. For arbitrary $t > 0 , \epsilon , \delta \in ( 0 , 1 )$ , and any projection vector $\pmb { w } \in \mathbb { S } ^ { n - 1 }$ , there exists $n _ { 0 } \in \mathbb { N } _ { + }$ such that for all $n > n _ { 0 }$ , the inequality below holds with probability at least $1 - \delta$

$$
p \left( \left\{ \left| w ^ { \top } ( X - \pmb \mu _ { X } ) \right| \geqslant t \right\} \right) \leqslant \epsilon .\tag{3}
$$

Proof: Given a projection vector $\pmb { w } \in \mathbb { S } ^ { n - 1 }$ , consider the random variable ${ \pmb w } ^ { \top } ( X - { \pmb \mu } _ { X } )$ . By Chebyshev’s inequality, for any $t > 0 \colon$

$$
p \left( \left. \pmb { w } ^ { \top } ( X - \pmb { \mu } _ { X } ) \right. \geqslant t \right) \leqslant \frac { \mathbb { V } a r [ \pmb { w } ^ { \top } ( X - \pmb { \mu } _ { X } ) ] } { t ^ { 2 } } ,\tag{4}
$$

where the variance term Var $\begin{array} { r } { \left[ { \pmb w } ^ { \top } ( X - { \pmb \mu } _ { X } ) \right] = \sum _ { i = 1 } ^ { n } w _ { i } ^ { 2 } \sigma _ { i } ^ { 2 } } \end{array}$ and $\mathbf { \nabla } w ^ { \top } w = 1$ . We next apply measure concentration properties of uniformly random vectors w on $\mathbb { S } ^ { n - 1 }$ to bound the random quantity $Z _ { w } ~ = ~ \mathbb { V } a r \left[ { \pmb w } ^ { \top } ( X - { \pmb \mu } _ { X } ) \right]$ Derivations in the appendix yield moment identities for spherical components:

$$
\mathbb { E } [ w _ { i } ^ { 2 } ] = \frac { 1 } { n } ; \qquad \mathbb { E } [ w _ { i } ^ { 4 } ] = \frac { 3 } { n ( n + 2 ) } .\tag{5}
$$

The expectation and variance of $Z _ { w } = \mathbb { V } a r \left[ { \pmb w } ^ { \top } ( X - { \pmb \mu } _ { X } ) \right]$ are computed as:

$$
\mathbb { E } [ Z _ { w } ] ~ = ~ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 2 } \triangleq \frac { 1 } { n } C _ { 2 }\tag{6}
$$

$$
\begin{array} { r c l } { \displaystyle \mathbb { V } a r [ Z _ { w } ] } & { = } & { \displaystyle \frac { 2 } { n ( n + 2 ) } \left( \sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 4 } - \frac { 1 } { n } \Big ( \sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 2 } \Big ) ^ { 2 } \right) \leqslant \frac { 2 } { n ( n + 2 ) } C _ { 4 } . } \end{array}\tag{7}
$$

with shorthand $\begin{array} { r } { C _ { 2 } = \sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 2 } , ~ C _ { 4 } = \sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 4 } } \end{array}$ . Under the bounded total-variance condition, both $C _ { 2 } , C _ { 4 }$ are bounded constants, so $\mathbb { E } [ Z _ { w } ] , \mathbb { V } a r [ Z _ { w } ]  0$ as the dimension n increases, implying $Z _ { w }$ converges to zero with high probability.

To estimate the tail bound of $Z _ { w }$ , we apply Chebyshev’s inequality to $Z _ { w }$ , for any $\eta > 0$

$$
p \big ( | Z _ { w } - \mathbb { E } [ Z _ { w } ] | \geqslant \eta \big ) \leqslant \frac { \mathbb { V } a r [ Z _ { w } ] } { \eta ^ { 2 } } \leqslant \frac { 2 } { n ( n + 2 ) } \frac { C _ { 4 } } { \eta ^ { 2 } } .\tag{8}
$$

Set $\eta = 1 / \sqrt { n }$ to guarantee a tight concentration around $\mathbb { E } [ Z _ { w } ]$ . There exists a threshold integer $n _ { 1 }$ such that for all $n \geqslant n _ { 1 }$ ，

$$
\frac { 2 } { n ( n + 2 ) } \frac { C _ { 4 } } { \eta ^ { 2 } } \leqslant \delta , \forall n \geqslant n _ { 1 } .\tag{9}
$$

Substituting (9) into equation (8) yields

$$
p \big ( | Z _ { w } - \mathbb { E } [ Z _ { w } ] | \geqslant 1 / \sqrt { n } \big ) \leqslant \delta , ~ \forall ~ n \geqslant n _ { 1 } .\tag{10}
$$

Equivalently:

$$
p \big ( | Z _ { w } - \mathbb { E } [ Z _ { w } ] | < 1 / \sqrt { n } \big ) > 1 - \delta , ~ \forall ~ n \geqslant n _ { 1 } .\tag{11}
$$

Taking the one-side bound for random variable $Z _ { W }$ , we obtain

$$
p \big ( Z _ { w } < \mathbb { E } [ Z _ { w } ] + 1 / \sqrt { n } \big ) > 1 - \delta .\tag{12}
$$

Using the expectation identity (6), we derive that there exists $n _ { 2 } \in \mathbb { N } _ { + }$ such that for all $n \geqslant n _ { 2 } :$

$$
\mathbb { E } [ Z _ { w } ] + 1 / { \sqrt { n } } = { \frac { C _ { 2 } } { n } } + { \frac { 1 } { \sqrt { n } } } \leqslant t ^ { 2 } \ \epsilon .\tag{13}
$$

Define $n _ { 0 } = \operatorname* { m a x } \{ n _ { 1 } , n _ { 2 } \}$ . Then combining two inequalities (12) and (13) yields that for all $n \geqslant n _ { 0 }$

$$
p \big ( Z _ { w } < t ^ { 2 } \epsilon \big ) > 1 - \delta .\tag{14}
$$

We now establish event inclusion relations to combine the dual probability bounds. Define three events:

$$
\begin{array} { r c l } { { E _ { 0 } } } & { { = } } & { { \left\{ w \mid p \left( \left| w ^ { \top } ( X - \pmb \mu _ { X } ) \right| \geqslant t \right) \leqslant \epsilon \right\} , } } \\ { { E _ { 1 } } } & { { = } } & { { \left\{ w \mid p \left( \left| w ^ { \top } ( X - \pmb \mu _ { X } ) \right| \geqslant t \right) \leqslant Z _ { w } / t ^ { 2 } \right\} , } } \\ { { E _ { 2 } } } & { { = } } & { { \left\{ w \mid Z _ { w } < t ^ { 2 } \epsilon \right\} . } } \end{array}
$$

The conjunction $E _ { 1 } \cap E _ { 2 }$ implies $E _ { 0 }$ , i.e., $E _ { 0 } \supseteq E _ { 1 } \cap E _ { 2 }$ . Under the theorem conditions, $E _ { 1 }$ holds for all $w \in S ^ { n - 1 }$ , thus we obtain

$$
p ( E _ { 0 } ) \geqslant p ( E _ { 1 } \cap E _ { 2 } ) = p ( E _ { 2 } ) > 1 - \delta .\tag{15}
$$

This confirms the following target bound holds with probability at least $1 - \delta$

$$
p \left( \left| \pmb { w } ^ { \top } ( X - \pmb { \mu } _ { X } ) \right| \geqslant t \right) \leqslant \epsilon .\tag{16}
$$

Thus, this completes the proof.

The proof employs two nested layers of measure concentration estimation: first, Chebyshev’s inequality for ${ \pmb w } ^ { \top } ( X - { \pmb \mu } _ { X } )$ provides a tail bound of projected data. The tail bound

$Z _ { w }$ depends on spherical projection ${ \pmb w } .$ . Second, we apply concentration inequalities to $Z _ { w }$ itself, showing its mean and variance vanish as the dimension n goes to infinity, resulting in $Z _ { w } < t ^ { 2 } \epsilon$ with high probability for suficiently large n.

We name the set $E _ { 0 }$ the set of good projection samples ${ \pmb w } .$ , since arbitrary small $\delta$ ensures random vector w falls within $E _ { 0 }$ with probability at least $1 - \delta$ . All good sample projections satisfy the projection concentration bound (3). The complement $E _ { 0 } ^ { c }$ contains bad sample projections that violate the concentration inequality. The typical bad vectors are the standard coordinate vectors $e _ { i }$ for each coordinate axis—projection onto a single coordinate axis recovers the marginal distribution of $X _ { i } .$ , which lacks the measure concentration property.

We can reformulate the tail bound (3) of Theorem 1 into its equivalent form: for any random sampled $\pmb { w }$ , fixed $t > 0$ , and arbitrarily small $\epsilon , \delta \in ( 0 , 1 )$

$$
p \left( \left| \pmb { w } ^ { \top } ( X - \pmb { \mu } _ { X } ) \right| < t \right) > 1 - \epsilon .\tag{17}
$$

This highlights a critical distinction between projection concentration of bounded-totalvariance embedding manifolds and isotropic Gaussian distributions: projections of isotropic Gaussian vectors remain univariate Gaussian and do not concentrate within arbitrarily small intervals around their projection mean. Under the condition of the bounded total-variance, embedding manifold projections concentrate within arbitrarily narrow bands around the projected mean for suficiently high ambient dimension $n ,$ for nearly all random projection directions w.

Theorem 1 describes the tail bound for measure concentration in the nested dual-probability way. We next derive a lemma converting dual probability bounds into single-layer concentration inequalities.

Lemma 1 Suppose conditional tail bound $p ( | f ( X , W ) | \geqslant t \ | \ W = w ) \leqslant g ( w ) \ ( \leqslant 1 )$ , and $p ( g ( W ) \leq \epsilon ) \geq 1 - \delta$ , then

$$
p ( | f ( X , W ) | \geqslant t ) \leqslant \epsilon + \delta .\tag{18}
$$

Proof: Apply the law of total expectation to the tail event $E = \left\{ | f ( X , W ) | \geqslant t \right\}$

$$
p ( \left| f ( X , W ) \right| \geqslant t ) = \mathbb { E } _ { W } { \big [ } p ( \left| f ( X , W ) \right| \geqslant t { \big | } \ W ) { \big ] } .\tag{19}
$$

By the conditional bound $p ( | f ( X , W ) | \geqslant t \ | \ W = w ) \leqslant g ( w )$ , we obtain

$$
p ( | f ( X , W ) | \geqslant t ) = \mathbb { E } _ { W } { \big [ } p ( | f ( X , W ) | \geqslant t | W ) { \big ] } \leqslant \mathbb { E } _ { W } { \big [ } g ( W ) { \big ) } { \big ] } .\tag{20}
$$

Decompose the expectation over disjoint indicator events $\{ g ( W ) \leqslant \epsilon \}$ and $\{ g ( W ) > \epsilon \}$ :

$$
p \big ( | f ( X , W ) | \geqslant t \big ) \leqslant \mathbb { E } \big [ g ( W ) \cdot \mathbf { 1 } _ { \{ g ( W ) \leqslant \epsilon \} } \big ] + \mathbb { E } \big [ g ( W ) \cdot \mathbf { 1 } _ { \{ g ( W ) > \epsilon \} } \big ] .\tag{21}
$$

The first term is bounded via $g ( W ) \leqslant \epsilon$ on its support ${ \mathbf { 1 } } _ { \{ g ( W ) \leqslant \epsilon \} } ;$

$$
\begin{array} { r } { \mathbb { E } \big [ g ( W ) \cdot \mathbf { 1 } _ { \{ g ( W ) \leq \epsilon \} } \big ] \leq \epsilon p ( g ( W ) \leqslant \epsilon ) . } \end{array}\tag{22}
$$

The second term uses the trivial upper bound $g ( W ) \leqslant 1$

$$
\begin{array} { r } { \mathbb { E } \big [ g ( W ) \cdot \mathbf { 1 } _ { \{ g ( W ) > \epsilon \} } \big ] \leq 1 \cdot p \big ( g ( W ) > \epsilon \big ) . } \end{array}\tag{23}
$$

Combining the above two estimates,

$$
p \big ( | f ( X , W ) | \geqslant t \big ) \leqslant \epsilon \cdot p ( g ( W ) \le \epsilon ) + p ( g ( W ) > \epsilon ) .\tag{24}
$$

The lemma’s premise $p ( g ( W ) \leq \epsilon ) \geq 1 - \delta$ implies $p ( g ( W ) > \epsilon ) \leq \delta$ , hence:

$$
p ( | f ( X , W ) | \geqslant t ) \leq \epsilon \cdot 1 + \delta = \epsilon + \delta .\tag{25}
$$

This completes the proof.

Theorem 2 ( Projection Concentration Theorem) Let X be an n-dimensional random vector with independent components, mean µ , and bounded total variance. Given $\pmb { \mu } _ { X }$ $t > 0 , \epsilon \in$ (0, 1), there exists $n _ { 0 } \in  { \mathbb { N } } _ { + }$ such that for all $n > n _ { 0 }$

$$
p \left( \left\{ \left| W ^ { \top } ( X - \pmb { \mu } _ { X } ) \right| \geqslant t \right\} \right) \leqslant \epsilon\tag{26}
$$

Proof: Define $f ( X , W ) = W ^ { \top } ( X - \pmb { \mu } _ { X } ) , g ( W ) = \mathbb { V } a r [ W ^ { \top } ( X - \pmb { \mu } _ { X } ) / t ^ { 2 }$ . From Theorem 1’s proof, for arbitrary $\epsilon _ { 1 } , \delta _ { 1 } \in ( 0 , 1 )$ , there exists $n _ { 0 } \in  { \mathbb { N } } _ { + }$ satisfying:

$$
p \left( \left| f ( X , W ) \right| \geqslant t \mid W = \pmb { w } \right) \leqslant g ( \pmb { w } ) ,\tag{27}
$$

$$
p \big ( g ( W ) \leqslant \epsilon _ { 1 } \big ) \geqslant 1 - \delta _ { 1 } .\tag{28}
$$

If we choose $\epsilon _ { 1 } = \epsilon / 2 , \delta _ { 1 } = \epsilon / 2$ , directly applying Lemma 1 yields the tail bound (26). □

The tail bound inequality (26) in Theorem 2 can be rewritten in its equivalent form:

$$
p \left( { \big \{ } | W ^ { \top } ( X - \pmb { \mu } _ { X } ) | < t { \big \} } \right) > 1 - \epsilon .\tag{29}
$$

This inequality states that for arbitrarily small concentration radius t and error tolerance ϵ, projected samples concentrate within an arbitrarily narrow band around the projected mean with probability at least $1 - \epsilon .$ , provided ambient embedding dimension n is suficiently large. The critical assumption is bounded total variance: indicating the low intrinsic dimensionality of embedding manifolds in high-dimensional spaces, should be independent of ambient dimension growth. From a high-dimensional perspective, low-dimensional manifolds occupy compact subspaces whose samples concentrate tightly around their global mean—a behavior observable via random linear projections.

## 3 Stochastic Projection Separability of Embedding Manifolds

The preceding section established projective measure concentration for single embedding manifolds under bounded total variance. We now prove that two distinct stochastic embedding manifolds with distinct means and bounded total variances become linearly separable with high probability when ambient embedding dimension is suficiently large.

Let $X = ( x _ { 1 } , \ldots , x _ { n } ) ^ { \intercal } \in \mathbb { R } ^ { n }$ denote an n-dimensional random vector with independent components, mean $\pmb { \mu } _ { X }$ , component-wise variances $\{ \sigma _ { i } ^ { 2 } \} _ { i = 1 } ^ { n }$ . Let $Y = ( y _ { 1 } , \ldots , y _ { n } ) ^ { \top } \in \mathbb { R } ^ { n }$ denote a second independent n-dimensional random vector with mean $\pmb { \mu } _ { Y }$ , component-wise variances $\{ \rho _ { i } ^ { 2 } \} _ { i = 1 } ^ { n }$ Let $\mathcal { D } _ { X } ~ = ~ \{ x _ { k } \} _ { k = 1 } ^ { N }$ denote N i.i.d. samples of X from its stochastic embedding manifold; let $\mathcal { D } _ { Y } = \{ y _ { k } \} _ { k = 1 } ^ { M }$ denote M i.i.d. samples of Y from its manifold.

Theorem 3 (Stochastic Projection Concentration for embedding Manifolds) Suppose random vectors X,Y are component-independent, and satisfy bounded total variance and $d i f -$ ferent means $\pmb { \mu } _ { X } \neq \pmb { \mu } _ { Y }$ . Given any $t > 0 , \ \epsilon , \ \delta \in ( 0 , 1 )$ and any projection vector $\pmb { w } \in \mathbb { S } ^ { n - 1 }$ there exists $n _ { 0 } \in  { \mathbb { N } } _ { + }$ such that for all $n > n _ { 0 }$ , the joint concentration event below holds with probability at least $1 - \delta$

$$
p \left( \left\{ \left. w ^ { \top } ( X - \mu _ { X } ) \right. < t \right\} \cap \left\{ \left. w ^ { \top } ( Y - \mu _ { Y } ) \right. < t \right\} , \forall X \in \mathcal { D } _ { X } , Y \in \mathcal { D } _ { Y } \right) > 1 - \epsilon .\tag{30}
$$

Proof: Apply Theorem 1 separately to X and Y. Given $t > 0 , \ \epsilon _ { 1 } = \epsilon / 2 , \ \delta _ { 1 } = \delta / 2 \in ( 0 , 1 )$ there exist dimension thresholds $n _ { 1 } , n _ { 2 } \in \mathbb { N } _ { + }$ such that the following inequalities hold with probability at least $1 - \delta _ { 1 }$

$$
p \Big ( \big | \pmb { w } ^ { \top } ( { \boldsymbol X } - { \pmb \mu } _ { { \boldsymbol X } } ) \big | \geqslant t \Big ) \leqslant \epsilon _ { 1 } , \quad \forall n \geqslant n _ { 1 } ,\tag{31}
$$

$$
p \Big ( \big | \pmb { w } ^ { \top } ( \pmb { Y } - \pmb { \mu } _ { \pmb { Y } } ) \big | \geqslant t \Big ) \leqslant \epsilon _ { 1 } , \quad \forall n \geqslant n _ { 2 } .\tag{32}
$$

Define good-projection events for each manifold,

$$
\begin{array} { r } { E _ { X } = \Big \{ { \pmb w } \Big | p \big ( \big | { \pmb w } ^ { \top } ( X - { \pmb \mu } _ { X } ) \big | \geqslant t \big ) \leqslant \epsilon _ { 1 } \Big \} , } \\ { E _ { Y } = \Big \{ { \pmb w } \Big | p \big ( \big | { \pmb w } ^ { \top } ( Y - { \pmb \mu } _ { Y } ) \big | \geqslant t \big ) \leqslant \epsilon _ { 1 } \Big \} . } \end{array}
$$

Using the probability of union bounds, we have

$$
p ( E _ { X } \cap E _ { Y } ) \geqslant p ( E _ { X } ) + p ( E _ { Y } ) - 1 > ( 1 - \delta _ { 1 } ) + ( 1 - \delta _ { 1 } ) - 1 = 1 - \delta .\tag{33}
$$

When both $E _ { X } , ~ E _ { Y }$ hold simultaneously, apply complement probability identities for joint events. Set $n _ { 0 } = \operatorname* { m a x } \{ n _ { 1 } , n _ { 2 } \}$ . For all $n \geqslant n _ { 0 }$ 2

$$
\begin{array} { r l } & { p \Big ( \big \{ \big | { \pmb w } ^ { \top } ( X - { \pmb \mu } _ { X } ) \big | < t \big \} \cap \big \{ \big | { \pmb w } ^ { \top } ( Y - { \pmb \mu } _ { Y } ) \big | < t \big \} \Big ) } \\ & { \quad = 1 - p \Big ( \neg \{ \big | { \pmb w } ^ { \top } ( X - { \pmb \mu } _ { X } ) \big | < t \big \} \cup \neg \big \{ \big | { \pmb w } ^ { \top } ( Y - { \pmb \mu } _ { Y } ) \big | < t \big \} \Big ) } \\ & { \quad \geqslant 1 - \Big [ p \big ( \big | { \pmb w } ^ { \top } ( X - { \pmb \mu } _ { X } ) \big | \geqslant t \big ) + p \big ( \big | { \pmb w } ^ { \top } ( Y - { \pmb \mu } _ { Y } ) \big | \geqslant t \big ) \Big ] } \\ & { \quad > 1 - \big ( \epsilon _ { 1 } + \epsilon _ { 1 } \big ) = 1 - \epsilon . } \end{array}\tag{34}
$$

This completes the proof.

This theorem infers a general result under mild conditions. If two datasets satisfy the bounded total variance, projections of two embedding manifolds concentrate tightly around their corresponding projected means with high probability provided the ambient dimension n is suficiently large. Although low-dimensional embedding manifolds can be projected into narrow equatorial disks centered at their projected means, the marginal projection bands may overlap. Joint concentration alone does not guarantee linear separability between the two classes, an additional non-singularity condition is required:

$$
\pmb { w } ^ { \top } ( \pmb { \mu } _ { X } - \pmb { \mu } _ { Y } ) \neq 0 .\tag{35}
$$

This condition enforces separation between the two projected cluster centroids, termed the non-singular projection condition.

Theorem 4 (Stochastic Projection Separability Theorem) Let X,Ydenote componentindependent random vectors with bounded total variance and distinct means $\pmb { \mu } _ { X } \neq \pmb { \mu } _ { Y }$ . Given ϵ, $\delta \in ( 0 , 1 )$ , and any non-singular projection vector w $\in \mathbb { S } ^ { n - 1 }$ satisfying w $^ { \top } ( { \pmb { \mu } } _ { X } - { \pmb { \mu } } _ { Y } ) \neq 0$ there exists $n _ { 0 } \in  { \mathbb { N } } _ { + }$ such that for all $n > n _ { 0 }$ , the following inequalities hold with probability at least $1 - \delta$

$$
p \left( \left\{ \pmb { w } ^ { \top } \pmb { X } > \pmb { w } ^ { \top } \pmb { Y } \right\} \right) \geqslant 1 - \epsilon , \quad i f \ \pmb { w } ^ { \top } ( \pmb { \mu } _ { X } - \pmb { \mu } _ { Y } ) > 0 ;\tag{36}
$$

$$
p \left( \left\{ \pmb { w } ^ { \top } \pmb { X } < \pmb { w } ^ { \top } \pmb { Y } \right\} \right) \geqslant 1 - \epsilon , \quad i f \ \pmb { w } ^ { \top } ( \pmb { \mu } _ { X } - \pmb { \mu } _ { Y } ) < 0 .\tag{37}
$$

Proof: Without loss of generality, assume ${ \pmb w } ^ { \top } ( { \pmb \mu } _ { X } - { \pmb \mu } _ { Y } ) > 0$ . Define the concentration radius $t = { \pmb w } ^ { \top } ( { \pmb \mu } _ { X } - { \pmb \mu } _ { Y } ) / 2$ . By Theorem 3 there exists threshold dimension $n _ { 0 }$ such that for all $n > n _ { 0 }$ , the following inequality holds with probability at least $1 - \delta$ 2

$$
p \Big ( \big \{ | w ^ { \top } ( X - \mu _ { X } ) | < t \big \} \cap \big \{ \big | w ^ { \top } ( Y - \mu _ { Y } ) \big | < t \big \} , \forall X \in \mathcal { D } _ { X } , Y \in \mathcal { D } _ { Y } \Big ) > 1 - \epsilon .\tag{38}
$$

The joint concentration event implies two strict inequalities:

$$
\pmb { w } ^ { \top } \boldsymbol { X } > \pmb { w } ^ { \top } \pmb { \mu } _ { \boldsymbol { X } } - t = \pmb { w } ^ { \top } \cdot \frac { \pmb { \mu } _ { \boldsymbol { X } } + \pmb { \mu } _ { \boldsymbol { Y } } } { 2 } ,\tag{39}
$$

$$
\pmb { w } ^ { \top } \pmb { Y } < \pmb { w } ^ { \top } \pmb { \mu } _ { Y } + t = \pmb { w } ^ { \top } \cdot \frac { \pmb { \mu } _ { X } + \pmb { \mu } _ { Y } } { 2 } .\tag{40}
$$

Combining these inequalities yields $\pmb { w } ^ { \top } \pmb { X } > \pmb { w } ^ { \top } \pmb { Y }$ . We infer the following event inclusion relation,

$$
\left\{ { \pmb w } ^ { \top } { \boldsymbol X } > { \pmb w } ^ { \top } { \boldsymbol Y } \right\} \supseteq \left\{ \left| { \pmb w } ^ { \top } ( { \boldsymbol X } - { \pmb \mu } _ { { \boldsymbol X } } ) \right| < t \right\} \cap \left\{ \left| { \pmb w } ^ { \top } ( { \boldsymbol Y } - { \pmb \mu } _ { { \boldsymbol Y } } ) \right| < t \right\} .\tag{41}
$$

Thus, the corresponding probabilities satisfy the following inequality

$$
p \big ( { \pmb w } ^ { \top } { \boldsymbol X } > { \pmb w } ^ { \top } { \boldsymbol Y } \big ) \geqslant p \Big ( \big \{ \big | { \pmb w } ^ { \top } ( { \boldsymbol X } - { \pmb \mu } _ { { \boldsymbol X } } ) \big | < t \big \} \cap \big \{ \big | { \pmb w } ^ { \top } ( { \boldsymbol Y } - { \pmb \mu } _ { { \boldsymbol Y } } ) \big | < t \big \} \Big ) > 1 - \epsilon .\tag{42}
$$

On the other hand, we define two event families to quantify confidence bounds:

$$
\begin{array} { r l r } { E _ { C } } & { \triangleq } & { \Big \{ w \Big | p \Big ( \big \{ \vert w ^ { \top } ( X - \mu _ { X } ) \big \vert < t \big \} \cap \big \{ \big \vert w ^ { \top } ( Y - \mu _ { Y } ) \big \vert < t \big \} \Big ) > 1 - \epsilon \Big \} , } \end{array}\tag{43}
$$

$$
E _ { S } \triangleq \big \{ \pmb { w } \big | p ( \pmb { w } ^ { \top } \pmb { X } > \pmb { w } ^ { \top } \pmb { Y } ) > 1 - \epsilon \big \} .\tag{44}
$$

Any projection vector $w \in E _ { C }$ also satisfies the separability bound $p ( \pmb { w } ^ { \top } \boldsymbol { X } > \pmb { w } ^ { \top } \boldsymbol { Y } ) > 1 - \epsilon$ in $E _ { S }$ , that is $E _ { C } \subseteq E _ { S }$ , implying $p ( E _ { S } ) \geqslant p ( E _ { C } ) > 1 - \delta$ . Thus the conclusion follows. □

This theorem proposes general suficient conditions for linear separability between two embedding manifolds: bounded total variance for each class and non-singular projection direction. Under the conditions of theorem 4, given a target classification accuracy $1 - \epsilon$ the theorem indicates that the two-class data are linearly separable with probability at least $1 - \epsilon$ along good non-singular projection w, provided the ambient embedding dimension n is suficiently large. From the proof of theorem 3, the probability of good projection samples is very high. We can easily find those good projection samples by combining the canonical projection vector with the random samples. We define a linear discriminant function as

$$
\ell ( X ) = { \pmb w } ^ { \top } \Big ( X - \frac { { \pmb \mu } _ { X } + { \pmb \mu } _ { Y } } { 2 } \Big ) .\tag{45}
$$

The samples are assigned to class $\mathcal { D } _ { X }$ if $\ell ( X ) > 0$ , and to $\mathcal { D } _ { Y }$ if $\ell ( X ) < 0$

Two assumptions are irreplaceable: bounded total variance and projective non-singularity.

1. Bounded total variance enforces projective measure concentration for embedding manifolds. If this condition fails, projections do not concentrate within arbitrarily small intervals around the projected mean even as the dimension n grows. For example, isotropic Gaussian distributions assign equal variance to every coordinate, violating bounded total variance. Accordingly, the projected data of isotropic Gaussian distributions follow a single-variable Gaussian distribution without tight concentration around the mean.

2. Projective non-singularity ensures the projected class centroids do not collapse onto identical points. Uniform random sampling of w on $\mathbb { S } ^ { n - 1 }$ yields near-orthogonality with $\pmb { \mu } _ { X } - \pmb { \mu } _ { Y }$ in high dimensions, i.e., ${ \pmb w } ^ { \top } ( { \pmb \mu } _ { X } - { \pmb \mu } _ { Y } ) \approx 0$ , violating non-singularity. To construct valid random non-singular projections, first build a canonical separating direction,

$$
{ \pmb w } _ { 0 } = \frac { { \pmb \mu } _ { X } - { \pmb \mu } _ { Y } } { \| { \pmb \mu } _ { X } - { \pmb \mu } _ { Y } \| _ { 2 } } ,\tag{46}
$$

which trivially satisfies ${ \pmb w } _ { 0 } ^ { \top } ( { \pmb \mu } _ { X } - { \pmb \mu } _ { Y } ) > 0$ Then we can randomly generate $\epsilon \sim$ $\mathcal { N } ( \mathbf { 0 } , \pmb { \sigma } _ { \epsilon } ^ { 2 } )$ , where $\pmb { \sigma } _ { \epsilon } ^ { 2 }$ is supposed to be bounded total-variance. Thus we construct valid projection vectors as $\pmb { w } = \pmb { \mathrm { \pmb { w } } } _ { 0 } + \pmb { \mathrm { \pmb { \epsilon } } }$ , followed by unit normalization. Such constructed projection vectors are able to balance randomness and non-singularity.

## 4 Discussions and Perspectives

The embedding vectors of images of same-category objects in high-dimensional feature spaces are located in the vicinity of a low-dimensional manifold, so the probabilistic distribution of the embedding vectors is not isotropic with the coordinate-wise variances. The stochastic separation theorem proposed by Gorban & Tyukin [12] deals with uniform distributed samples on unit spheres, whereas this paper addresses a more realistic two-distribution separability problem for object embedding manifolds.

This work addresses two core problems for stochastic embedding manifolds: projection measure concentration and cross-manifold stochastic separability. Diferent from the isotropic Gaussian distributions, where the projection of Gaussian random vector becomes uni-variate Gaussian without tight concentration around the projected mean, the probability distribution of stochastic embedding manifolds with the bounded total-variance enables projected data of the embedding manifolds to concentrate in arbitrarily narrow bands around the projected class mean, provided the ambient dimension is suficiently large.

Gorban & Tyukin’s stochastic separation theory characterizes intra-dataset probabilistic convexity: each sample is Fisher linearly separable from the rest of its dataset with high probability, provided the ambient dimension is suficiently large. Our work extends this to interdataset separability between two distinct manifold distributions, proving high-probability linear separation with arbitrarily small classification error under bounded total-variance and projection non-singularity, provided the ambient dimension is large enough.

We develop a novel two-layer nested measure concentration analysis approach, applying Chebyshev’s inequality sequentially for dual-layer tail bounds and leveraging spherical projection vector concentration to integrate the concentration inequalities into one inequality.

We should mention here that the stochastic separability theory not only reveals the stochastic separability properties in high dimensional spaces, but provides a new mechanism for representation learning of large foundation models. To maximize cross-class manifold separability, one should minimize the intrinsic dimension of each class’s embedding manifold to reduce the bounded total-variance.

There are a few perspective research directions related to the stochastic separability. First one is how to extend the stochastic separability theory to disentangled embedding manifolds for improved inference robustness. Second one is to investigate the optimal separating projection directions beyond random projections. Still other one is to estimate the minimal embedding dimension n, given ϵ and δ.

## References

[1] Luca Albergante, Jonathan Bac, and Andrei Zinovyev. Estimating the efective dimension of large biological datasets using fisher separability analysis. In 2019 International Joint Conference on Neural Networks (IJCNN), pages 1–8, 2019.

[2] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8):1798–1828, 2013.

[3] SueYeon Chung and Larry F Abbott. Neural population geometry: An approach for understanding biological and artificial neural networks. Current opinion in neurobiology, 70:137–144, 2021.

[4] SueYeon Chung, Daniel D Lee, and Haim Sompolinsky. Classification and geometry of general perceptual manifolds. Physical Review X, 8(3):031003, 2018.

[5] Uri Cohen, SueYeon Chung, Daniel D. Lee, and Haim Sompolinsky. Separability and geometry of object manifolds in deep neural networks. Nature Communications, 11(1):746, 2020.

[6] James J DiCarlo and David D Cox. Untangling invariant object recognition. Trends in Cognitive Sciences, 11(8):333–341, 2007.

[7] James J DiCarlo, Davide Zoccolan, and Nicole C Rust. How does the brain solve visual object recognition? Neuron, 73(3):415–434, 2012.

[8] David L Donoho et al. High-dimensional data analysis: The curses and blessings of dimensionality. AMS math challenges lecture, 1(2000):1–32, 2000.

[9] Ian Goodfellow, Honglak Lee, Quoc Le, Andrew Saxe, and Andrew Ng. Measuring invariances in deep networks. In Advances in Neural Information Processing Systems, volume 22, pages 646–654, 2009.

[10] Alexander N. Gorban, Alexey Golubkov, Bogdan Grechuk, Evgeny M. Mirkes, and Ivan Y. Tyukin. Correction of ai systems by linear discriminants: Probabilistic foundations. Information Sciences, 466:303–322, 2018.

[11] Alexander N. Gorban, Bogdan Grechuk, Evgeny M. Mirkes, Sergey V. Stasenko, and Ivan Y. Tyukin. High-dimensional separability for one- and few-shot learning. Entropy, 23(8):1090, 2021.

[12] Alexander N Gorban and Ivan Y Tyukin. Stochastic separation theorems. Neural Networks, 94:255–259, 2017.

[13] Alexander N. Gorban and Ivan Yu. Tyukin. Blessing of dimensionality: mathematical foundations of the statistical physics of data. Philosophical Transactions of the Royal Society A, 376(2118):20170237, 2018.

[14] Bogdan Grechuk, Alexander N Gorban, and Ivan Y Tyukin. General stochastic separation theorems with optimal bounds. Neural Networks, 138:33–56, 2021.

[15] Tijl Grootswagers, Anna K Robinson, Samuel M Shatek, and Thomas A Carlson. Untangling featural and conceptual object representations. NeuroImage, 202:116083, 2019.

[16] Leonard Gross. Logarithmic sobolev inequalities. American Journal of Mathematics, 97:1061–1083, 1975.

[17] Olivier J H´enaf, Robbe L T Goris, and Eero P Simoncelli. Perceptual straightening of natural videos. Nature Neuroscience, 22(6):984–991, 2019.

[18] Paul C. Kainen. Utilizing Geometric Anomalies of High Dimension: When Complexity Makes Computation Easier, pages 283–294. Birkh¨auser Boston, Boston, MA, 1997.

[19] Michel Ledoux. On talagrand’s deviation inequalities for product measures. ESAIM: Probability and statistics, 1:63–87, 1997.

[20] Michel Ledoux and Michel Talagrand. Probability in Banach Spaces: Isoperimetry and Processes. Springer, New York, NY, 1991.

[21] V. D Milman. Geometric theory of banach spaces. part ii. geometry of the unit sphere. Russian Math. Surveys, 26:79–163, 1971.

[22] Vitali D Milman. A new proof of a. dvoretzky’s theorem on cross-sections of convex bodies. Funkcional. Anal. i Prilozen, 5:28–37, 1971.

[23] Marino Pagan, Luke S Urban, Margot P Wohl, and Nicole C Rust. Signals in inferotemporal and perirhinal cortex suggest an untangling of visual target information. Nature neuroscience, 16(8):1132–1139, 2013.

[24] Marc’Aurelio Ranzato, Fu Jie Huang, Y-Lan Boureau, and Yann LeCun. Unsupervised learning of invariant feature hierarchies with applications to object recognition. In 2007 IEEE conference on computer vision and pattern recognition, pages 1–8. IEEE, 2007.

[25] Ivan Tyukin, Alexander N. Gorban, Carlos Calvo, Julia Makarova, and Valeri Makarov. High-dimensional brain: A tool for encoding and rapid learning of memories by single neurons. Bulletin of Mathematical Biology, 81:4856–4888, 2019.

[26] Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science, 2nd Edition. Cambridge University Press, 2026.

# A Fourth Moment Identities of Uniform Variables on $\mathbb { S } ^ { n - 1 }$

Let random vector $\mathbf { x } \in \mathbb { S } ^ { n - 1 }$ follow uniform distribution, satisfying:

$$
\sum _ { i = 1 } ^ { n } x _ { i } ^ { 2 } = 1 .\tag{47}
$$

Taking an expectation of both sides of the above equation, we have

$$
\sum _ { i = 1 } ^ { n } \mathbb { E } [ x _ { i } ^ { 2 } ] = 1 .\tag{48}
$$

Using the component symmetry $\mathbb { E } [ x _ { i } ^ { 2 } ] = \mathbb { E } [ x _ { j } ^ { 2 } ]$ , we have

$$
\mathbb { E } [ x _ { i } ^ { 2 } ] = { \frac { 1 } { n } } .\tag{49}
$$

Define the fourth moment of $x _ { i }$ as $M _ { 4 } = \mathbb { E } [ x _ { 1 } ^ { 4 } ]$ , and denote $V _ { 1 2 } = \mathbb { E } [ x _ { 1 } ^ { 2 } x _ { 2 } ^ { 2 } ]$ . Squaring identity (47), and taking expectation on both sides of the equation, we obtain

$$
n M _ { 4 } + n ( n - 1 ) V _ { 1 2 } = 1 ,\tag{50}
$$

where we utilize the moment symmetry $( M _ { 4 } = \mathbb { E } [ x _ { i } ^ { 4 } ] , V _ { 1 2 } = \mathbb { E } [ x _ { i } ^ { 2 } x _ { j } ^ { 2 } ] , \forall i \neq j )$ . By rotational symmetry of the unit sphere, rotating coordinates $( x _ { 1 } , x _ { 2 } )$ by $\pi / 4$ preserves cross-moment $V _ { 1 2 }$

$$
\mathbb { E } \left[ \left( \frac { x _ { 1 } + x _ { 2 } } { \sqrt { 2 } } \right) ^ { 2 } \left( \frac { x _ { 1 } - x _ { 2 } } { \sqrt { 2 } } \right) ^ { 2 } \right] = V _ { 1 2 } .\tag{51}
$$

Expand the left-hand side:

$$
\frac { 1 } { 4 } \left( \mathbb { E } \left[ x _ { 1 } ^ { 4 } \right] + \mathbb { E } \left[ x _ { 2 } ^ { 4 } \right] - 2 \mathbb { E } \left[ x _ { 1 } ^ { 2 } x _ { 2 } ^ { 2 } \right] \right) = V _ { 1 2 } .\tag{52}
$$

Solving the above equation leads to the following relation

$$
M _ { 4 } = 3 V _ { 1 2 } .\tag{53}
$$

Substituting the relation into (50), we obtain the closed-form solutions for the moments:

$$
M _ { 4 } = \frac { 3 } { n ( n + 2 ) } , \quad V _ { 1 2 } = \frac { 1 } { n ( n + 2 ) } .\tag{54}
$$

As ambient dimension $n \to \infty$ , both second and fourth component moments decay to zero.