# How Do DiTs Denoise? An Aggregator–Predictor View of Attention and MLPs

Gyeonghun Kang<sup>1</sup> William Xingxu Chen<sup>2</sup> Xiang Cheng<sup>3</sup>

<sup>1</sup>Department of Statistical Science, Duke University gk149@duke.edu

<sup>2</sup>Department of Computer Science, Duke University wc187@duke.edu

<sup>3</sup>Department of Electrical and Computer Engineering, Duke University xiang.cheng@duke.edu

How do difusion transformers (DiTs) represent image content and learn rules for reusing it to generalize? We develop an aggregator–predictor view of DiTs as input-adaptive local denoisers, in which context aggregation and patch prediction provide functional roles for interpreting attention and MLPs. We first establish that, for Gaussian Markov random field (GMRF) priors, the global Bayes denoiser can be approximated using only a subset of context patches, with error decaying exponentially in context radius. We further show that patchwise linear denoisers, often invoked to explain difusion generalization, recover the empirical local denoiser with suficiently informative aggregation. Within this view, AdaLN-modulated attention naturally parameterizes an input- and noise-dependent aggregator, while the MLP admits a learned key–value representation for tokenwise patch prediction. Our experiments support this view. DiTs trained on real images exhibit noise-dependent denoising contexts resembling those in the analytic GMRF setting, while attention mass exhibits a qualitatively corresponding broadening pattern. In a controlled CelebA64 study, shared-memory patch predictors used in place of DiT MLPs achieve competitive denoising and generative performance. Finally, cross-dataset component transplantation reveals compatibility patterns consistent with the proposed roles of QK in spatial weighting, VO in feature transport, and MLPs in patch prediction. Together, these results provide a component-level account of DiT denoising and motivate architecture-specific fine-tuning and controlled intervention.

Last updated: October 22, 2026 Code: https://github.com/hun-learning94/dit-denoise

Duke

## 1 Introduction

Difusion and flow-matching models are widely used for image generation (Song et al., 2021; Lipman et al., 2023), with Difusion Transformers (DiTs) becoming a standard backbone (Peebles and Xie, 2023; Esser et al., 2024; Xie et al., 2025; Li and He, 2026). Growing evidence connects their generalization to representations of images and rules for reusing them. Difusion features encode semantic and geometric information (Baranchuk et al., 2022; Tang et al., 2023), and aligning them with pretrained representations improves training (Yu et al., 2025). Models also generate combinations absent from the training data, as shown by analyses of synthetic hierarchical data (Okawa et al., 2023; Favero et al., 2025). How components of DiTs realize these capabilities remains unclear.

Existing explanations connect difusion-model generalization to geometry-adaptive harmonic or balanced representations (Kadkhodaie et al., 2024; Zhang et al., 2026), and inductive biases toward spatial locality and translation equivariance (Kamb and Ganguli, 2025; Niedoba et al., 2025). These accounts leave open how DiT components divide the denoising computation. Rosu et al. (2025) show that attention can implement exact score-based denoising, but they consider whole-image tokens, leaving interactions among spatial patches unresolved. An et al. (2025) associate DiT generalization with attention mimicking CNN-like spatial locality, but attention can also select input-dependent, spatially non-contiguous context (Sec. 5.1). We ask: How do attention, MLPs, and adaptive layer normalization (AdaLN) divide the work of representing image content and learning context-dependent rules for generalization?

Understanding these component roles matters for intervention: domain adaptation, concept editing, and controlled generation require deciding which computations to modify and which to preserve. In language models, interpreting MLPs as key–value memories has motivated methods for editing stored associations (Geva et al., 2021; Meng et al., 2022, 2023), with related analyses in multimodal transformers (Pan et al., 2024a; Basu et al., 2024a). A functional account of DiTs could similarly guide component-wise adaptation across domains and interventions that change image content while preserving desired spatial relationships. Related work is reviewed in Appendix A.

![](images/e446c38059b3a12d4bcf504cc34d1cd5148b62f9399f091a21b07fd0b9e05200.jpg)  
Figure 1 In each row, a CelebA source checkpoint is fine-tuned on the AAHQ, FFHQ, AFHQ, or STL-10 dataset. The source and fine-tuned generated images are shown in columns (000) and (111). In the middle columns, attention QK/VO and MLP weights of the source checkpoint are replaced with the corresponding fine-tuned weights; e.g., (001) means only MLP is replaced, while QK/VO are retained. For face-to-face fine-tuning, replacing only QK generates source-like faces, while replacing MLP brings visual changes: cartoon faces, bald heads, or dogs. Among the tested CelebA→STL-10 hybrids, coherent target-like structure appears primarily in the joint endpoint (111); reverting VO (101) leaves fragmented texture. See Sec. 5.3 and Appendix E.4.

We address this question by proposing an aggregator–predictor view of DiTs as input-adaptive local denoisers, supported by a statistical foundation and a set of architectural and empirical observations:

(1) A statistical foundation for local denoising. We justify local denoiser, in which each patch is denoised using only a subset of context patches, by identifying its statistical target and establishing when it approximates the global Bayes denoiser. Specifically, for Gaussian Markov random field (GMRF) image priors, the global denoiser can be approximated by a radius-r local denoiser with error decaying exponentially in r (Thm. 1). We further connect existing patchwise retrieval-based explanations of generalization to this target by showing that, with suficiently informative aggregation, a patchwise linear denoiser exactly recovers the empirical local denoiser (Thm. 3).

(2) An aggregator–predictor interpretation of DiT components. We decompose the local denoising into context aggregation followed by patch prediction (Def. 2). Under this view, the algebraic structure of AdaLN-modulated attention implements an input- and noise-dependent aggregation rule: QK parameterizes spatial compatibility weights, while VO transforms the corresponding features (Sec. 4.1). In turn, MLP admits a key–value memory interpretation and operates on the resulting contextualized token (Sec. 4.2). These algebraic correspondences are exact for the isolated branches considered in Sec. 4; in a multilayer residual stack, the proposed roles interact and need not remain functionally separable (Appendix D.5). This view extends existing fixed spatial-locality accounts by allowing the relevant context to depend on the image and noise level and to include spatially non-contiguous patches. It further suggests the possibility of replacing DiT MLPs with more interpretable memory-based components, as supported by our experiments (Sec. 5.2).

(3) Empirical evidence consistent with the proposed component roles. Across real-image datasets, DiT sensitivity fields exhibit noise-dependent context broadening resembling the GMRF setting, while attention mass shows a related qualitative trend (Sec. 5.1). We further construct explicit shared-memory patch predictors that retain competitive denoising and generation performance in a controlled CelebA64 study (Sec. 5.2). Finally, cross-domain component transplantation reveals systematic component-compatibility patterns associated with source–target spatial similarity. These patterns are consistent with, but do not uniquely identify, the proposed roles in context aggregation and patch prediction (Sec. 5.3).

## 2 Background and Problem Setup

Let $X _ { 0 } = ( X _ { 0 , 1 } , \hdots , X _ { 0 , M } ) \in \mathbb { R } ^ { M d _ { p } }$ be a clean image with M patches, where $X _ { 0 , m } \in \mathbb { R } ^ { d _ { p } }$ is patch m. At time $t ,$ consider $X _ { t } = a _ { t } X _ { 0 } + b _ { t } \varepsilon$ , with $a _ { t } , b _ { t } > 0$ and $\varepsilon \sim \mathcal { N } ( 0 , I _ { M d _ { p } } )$ independent of $X _ { 0 } .$ . This formulation includes variance-exploding (VE) and variance-preserving (VP) difusion (Song et al., 2021), as well as linear Gaussian interpolation paths used in flow matching (Lipman et al., 2023). We define the normalized observation $Y _ { t } : = X _ { t } / a _ { t } = X _ { 0 } + \sigma _ { t } \varepsilon$ and the efective noise level $\sigma _ { t } : = b _ { t } / a _ { t }$ . Under standard squared-loss objectives, population-optimal clean-image, noise, and velocity predictions are related by known afine transformations to the denoiser (Li and He, 2026)

$$
D _ { m , t } ^ { * } ( y ) : = \mathbb { E } [ X _ { 0 , m } \mid Y _ { t } = y , t ] , \qquad m \in \{ 1 , \ldots , M \} .\tag{1}
$$

Throughout the paper, unless otherwise specified, expectations involving $Y _ { t }$ are over $X _ { 0 } ~ \sim ~ p _ { 0 }$ and the independent Gaussian corruption. We call (1) a population global denoiser because the prediction conditions on all noisy patches.

For training images $\{ x ^ { ( i ) } \} _ { i = 1 } ^ { n }$ and the empirical prior $\begin{array} { r } { \hat { p } _ { 0 } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } \delta _ { x ^ { ( i ) } } } \end{array}$ , the empirical version of the global denoiser (1) is a weighted sum of whole images:

$$
\hat { D } _ { m , t } ( \boldsymbol { y } ) = \sum _ { i = 1 } ^ { n } \operatorname { s m } _ { i } \left\{ - \frac { 1 } { 2 \sigma _ { t } ^ { 2 } } \left. \boldsymbol { y } - \boldsymbol { x } ^ { ( j ) } \right. ^ { 2 } \right\} _ { j \in [ n ] } \boldsymbol { x } _ { m } ^ { ( i ) } ,\tag{2}
$$

where the softmax sm is taken over the n training images. This creates a tension between training-loss minimization and generalization in difusion models: under exact reverse dynamics and the corresponding terminal initialization, the empirical denoiser reproduces training images at the clean endpoint (Biroli et al., 2024; Kamb and Ganguli, 2025). Even when these conditions are not met, the shared mixture weights across patches keep the full-image prediction in the convex hull of whole training images, restricting its ability to recombine their local content independently (Kamb and Ganguli, 2025; Niedoba et al., 2025).

This raises the question of why trained models often generalize beyond memorizing training images. Several works attribute this to compositional generation, in which retrieval weights vary across patch locations (Niedoba et al., 2025; Kamb and Ganguli, 2025; Lukoianov et al., 2025). We formalize the estimator underlying the compositional accounts as a patchwise linear denoiser (PLD). PLD depends on y only through the context patches $y z _ { m , \ l }$ and allows diferent output patches to combine content from diferent training images. It is linear in how the context patches are aggregated.

Definition 1 (Patchwise linear denoiser (PLD)). Fix a context set $B _ { m , t } \subseteq \{ 1 , \dots , M \}$ and a matrix $S _ { m , t } \in \mathbb { R } ^ { d _ { h } \times M d _ { p } }$ with full row rank whose lth block of columns, $[ S _ { m , t } ] _ { l } \in \mathbb { R } ^ { d _ { h } \times d _ { p } }$ , is zero for $l \notin B _ { m , t } .$ Given a symmetric positive definite $\Gamma _ { m , t } \in \mathbb { R } ^ { d _ { h } \times d _ { h } }$ , define the patchwise linear denoiser (PLD) by

$$
\hat { D } _ { m , t } ^ { \mathrm { p l d } } ( y ) = \sum _ { i = 1 } ^ { n } \mathrm { s m } _ { i } \left\{ - \frac { 1 } { 2 } \left. S _ { m , t } ( y - x ^ { ( j ) } ) \right. _ { \Gamma _ { m , t } } ^ { 2 } \right\} _ { j \in [ n ] } x _ { m } ^ { ( i ) } ,\tag{3}
$$

where $\| u \| _ { \Gamma _ { m , t } } ^ { 2 } : = u ^ { \top } \Gamma _ { m , t } u$ and $x _ { m } ^ { ( i ) }$ is patch m of training image i.

However, it is not clear when PLD achieves low population denoising risk or how a neural architecture can implement it. In the following sections, we address three central questions: what statistical target does the PLD approximate, when is that target close to the population denoiser, and how can a DiT implement such an estimator?

![](images/79245419a9996683d6f22427ec45ae736c1c8ddb0c6b30820573c80a10f768a9.jpg)

![](images/b68e819fc2f1fdd7f755293a2507902bb2e8f6d935aa7c1c108cf2287c77d46e.jpg)

![](images/c4f4a53fa34ca8ea1efe13b7eec30dcdb0383e2ba978e4078f3e5968c8b5a8b5.jpg)  
Figure 2 Exact context set in a Gaussian MRF prior. The known graph and precision matrix allow exact analysis of context set in this synthetic setting. (a) Radius-3 contexts $B _ { 3 } ( m )$ for two selected patches. Defined by graph distance, these sets need not be spatially contiguous. (b) Exact and certified radii at $\epsilon = { 1 0 } ^ { - 3 }$ . Corollary 2 guarantees that the certified radius is nondecreasing up to $\sigma _ { \mathrm { a b s } }$ (dotted lines); the exact radii can be substantially smaller. (c) 90%/95% sensitivity radii (Sec. 5.1), showing that relative influence spreads over a broader context as noise increases. Details in Appendix E.1.

## 3 Local denoiser can approximate global denoiser

We first show when global denoising can be accurately approximated using only a local subset of patches. Suppose that the clean patches jointly follow a Markov random field (MRF) with respect to a graph $G _ { 0 } = ( [ M ] , E _ { 0 } )$ , and let $B _ { r } ( m ) \subseteq [ M ]$ be a graph-radius context set, containing all patches within graph distance r of patch m. This subset need not be spatially contiguous (Fig. 2(a)). Define

$$
D _ { m , t } ^ { * } ( y ) : = \mathbb { E } \bigl [ X _ { 0 , m } \mid Y _ { t } = y , t \bigr ] , \qquad D _ { m , t } ^ { * , r } ( y ) : = \mathbb { E } \bigl [ X _ { 0 , m } \mid Y _ { t , B _ { r } ( m ) } = y _ { B _ { r } ( m ) } , t \bigr ]
$$

as the global denoiser and $B _ { r } ( m )$ -local denoiser, respectively. We define their mean-squared discrepancy $\mathbb { E } \big \| D _ { m , t } ^ { * } ( Y _ { t } ) - D _ { m , t } ^ { * , r } ( Y _ { t } ) \big \| _ { 2 } ^ { 2 }$ as the locality error and say that $p _ { 0 }$ is exponentially local with respect to $G _ { 0 }$ if the following holds.

Assumption 1 (Exponential locality). For every $t ,$ there exists a constant $c _ { t } > 0$ such that, for every patch m and integer $r \geq 0 , \mathbb { E } \left. D _ { m , t } ^ { * } ( Y _ { t } ) - D _ { m , t } ^ { * , r } ( Y _ { t } ) \right. _ { 2 } ^ { 2 } \lesssim _ { t } \exp ( - c _ { t } r )$

An explicit example is a Gaussian MRF image prior $X _ { 0 } \sim \mathcal { N } ( 0 , Q ^ { - 1 } )$ with positive definite block precision $Q \in \bar { \mathbb { R } ^ { M d _ { p } \times M d _ { p } } }$ supported on $G _ { 0 }$ . Here, the global denoiser has the closed form $D _ { m , t } ^ { * } ( y ) = [ ( I + \sigma _ { t } ^ { 2 } { Q } ) ^ { - 1 } y ] _ { m }$ Although this denoiser generally depends on all noisy patches, analyticity of the inverse function and the graph sparsity of $Q$ allow approximation using only radius-r context, with error decaying exponentially in r (Benzi and Razouk, 2007).

Theorem 1 (Exponential locality of GMRF, informal). A Gaussian MRF is exponentially local in the sense of Assump. 1. The full bound and conditions are in Appendix C.1.

Intuitively, noisier patches provide weaker evidence individually, making broader context useful for denoising. The following guarantee formalizes this intuition for the bound; the optimal local denoiser may achieve the same tolerance with less context (Fig. 2(b)).

Corollary 2 (Bounded-range monotonicity of the locality radius, informal). For any error tolerance $\epsilon > 0$ , the smallest radius $r _ { \epsilon } ( \sigma _ { t } )$ certified by the explicit exponential bound is nondecreasing in noise on $0 \leq \sigma _ { t } \leq \sigma _ { \mathrm { a b s } } .$ , where $\sigma _ { \mathrm { a b s } } > 0$ depends on Q. The full statement, radius formula, and noise threshold are in Appendix C.2.

This guarantee concerns a bounded noise range: at extreme noise, the posterior mean approaches the zero prior mean and the absolute bound vanishes. Fig. 2 illustrates the context $B _ { r } ( m )$ defined by graph distance and the radius guarantee in Cor. 2 using a synthetic GMRF prior. Similar locality patterns appear in DiTs trained on real images (Sec. 5.1).

## 3.1 Aggregator–predictor decomposition of local denoiser

We next interpret PLD and DiTs as estimators of the same local denoiser. Without assuming that the image prior is an MRF, fix a patch m, a time $t ,$ and a context set $B _ { m , t }$ . We define the following population target as the ${ \cal { B } } _ { m , \pmb { \imath } }$ <sub>t</sub>-local denoiser

$$
D _ { m , t } ^ { * , \mathcal { B } } ( y ) : = \mathbb { E } [ X _ { 0 , m } \mid Y _ { t , \mathcal { B } _ { m , t } } = y _ { \mathcal { B } _ { m , t } } , t ] .\tag{4}
$$

We decompose its estimation in two stages: local aggregation of the relevant context, followed by patch prediction from the aggregated statistic.

Specifically, a measurable map $S _ { m , t } : \mathbb { R } ^ { M d _ { p } }  \mathbb { R } ^ { d _ { h } }$ is a $B _ { m , \cdot }$ -local aggregator if $y _ { B _ { m , t } } ~ = ~ y _ { B _ { m , t } } ^ { \prime }$ implies $S _ { m , t } ( y ) = S _ { m , t } ( y ^ { \prime } )$ for all $y , y ^ { \prime } .$ . It depends only on the context patches and maps them, possibly nonlinearly, to a statistic in $\mathbb { R } ^ { d _ { h } }$ . A measurable map $g _ { m , t } : \mathbb { R } ^ { d _ { h } }  \mathbb { R } ^ { d _ { p } }$ , called a patch predictor, then predicts the clean patch from this statistic. For a given local aggregator, a predictor recovering the local conditional mean exists if and only if $\operatorname { s f } ( D _ { m , t } ^ { * , \mathcal { B } } ( Y _ { t } ) ) \subseteq \operatorname { s f } ( \mathbf { \bar { S } } _ { m , t } ( Y _ { t } ) )$ ; that is, local aggregation preserves all information needed to recover the local conditional mean. The aggregator may discard details that are unnecessary for determining this mean. In that case, the tower property shows that such predictor can be chosen as $g _ { m , t } ^ { * } ( s ) = \operatorname { \mathbb { E } } [ X _ { 0 , m } \mid S _ { m , t } ( Y _ { t } ) = s , t ]$

Definition 2 (Aggregator–predictor decomposition). A pair of measurable maps $( S _ { m , t } ^ { * } , g _ { m , t } ^ { * } )$ is an aggregator–predictor decomposition of $D _ { m , t } ^ { * , B }$ if $S _ { m , t } ^ { * }$ is $B _ { m , t }$ -local and, for $P _ { Y _ { t } }$ -almost every y, $D _ { m , t } ^ { * , B } ( y ) =$ $( g _ { m , t } ^ { * } \circ S _ { m , t } ^ { * } ) ( y )$

This decomposition is generally nonunique but provides a framework for separately estimating the aggregator and predictor of the local denoiser (4). The PLD is one instance; for a patch m and time t with $\sigma _ { t } > 0$ setting $\Gamma _ { m , t } = ( \sigma _ { t } ^ { 2 } S _ { m , t } S _ { m , t } ^ { \top } ) ^ { - 1 }$ in Def. 1, the PLD is decomposed into $S _ { m , t } ^ { \mathrm { l i n } } ;$ the linear aggregator, and $g _ { m , t } ^ { \mathrm { r b f } }$ the radial basis function (RBF) predictor:

$$
\begin{array} { r l } & { \hat { D } _ { m , t } ^ { \mathrm { p l i d } } ( y ) = \left( g _ { m , t } ^ { \mathrm { r b f } } \circ S _ { m , t } ^ { \mathrm { l i n } } \right) ( y ) , } \\ & { S _ { m , t } ^ { \mathrm { l i n } } ( y ) = S _ { m , t } y , \qquad g _ { m , t } ^ { \mathrm { r b f } } ( s ) = \displaystyle \sum _ { i = 1 } ^ { n } \operatorname { s m } _ { i } \left\{ - \frac { 1 } { 2 } \left. s - S _ { m , t } ^ { \mathrm { l i n } } ( x ^ { ( j ) } ) \right. _ { ( \sigma _ { t } ^ { 2 } S _ { m , t } S _ { m , t } ^ { \top } ) ^ { - 1 } } ^ { 2 } \right\} _ { j \in [ n ] } x _ { m } ^ { ( i ) } . } \end{array}\tag{5}
$$

Bayes’ rule identifies the predictor $g _ { m , t } ^ { \mathrm { r b f } } ( s )$ with $\mathbb { E } _ { \hat { p } _ { 0 } } [ X _ { 0 , m } \mid S _ { m , t } ^ { \mathrm { l i n } } ( Y _ { t } ) = s , t ]$ , the conditional mean given the aggregated statistic s under the empirical prior; see Appendix C.3. This correspondence has been noted in prior work (Niedoba et al., 2025; Kamb and Ganguli, 2025; Lukoianov et al., 2025). In our framework, these approaches correspond to using cropping matrices, prescribed neighborhood masks, or thresholded Wiener-filter sensitivity masks for $S _ { m , t }$ , respectively. We go a step further and show that, with a suficiently expressive aggregator, $\hat { D } _ { m , t } ^ { \mathrm { p l d } }$ exactly recovers the local denoiser under the empirical observation law.

Theorem 3 (PLD recovers the empirical local denoiser). Under the setup above, let $\sigma _ { t } > 0$ and take $\Gamma _ { m , t } = ( \sigma _ { t } ^ { 2 } S _ { m , t } \tilde { S } _ { m , t } ^ { \top } ) ^ { - 1 }$ in Def. 1. $I f S _ { m , t } ^ { \mathrm { l i n } }$ preserves the empirical local conditional mean, i.e., s $\mathrm { f } ( \mathbb { E } _ { \hat { p } _ { 0 } } [ X _ { 0 , m } \mid$ $Y _ { t , { \mathscr B } _ { m , t } } , t ] ) \subseteq \operatorname { s f } ( S _ { m , t } ^ { \mathrm { l i n } } ( Y _ { t } ) )$ , then, almost surely under the empirical observation law, $\hat { D } _ { m , t } ^ { \mathrm { p l d } } ( y ) = \mathbb { E } _ { \hat { p } _ { 0 } } [ X _ { 0 , m } \mid$ $Y _ { t , B _ { m , t } } = y _ { B _ { m , t } } , t ]$

The condition holds whenever the full local observation can be recovered from the aggregate, as with a patch-selection matrix. A compressive linear aggregator can also sufice: when the training contexts restricted to $B _ { m , t }$ lie in a lower-dimensional afine subspace, a fixed linear aggregator whose row space contains their diference subspace is suficient; see (17).

Theorem 3 therefore connects the PLD, used to explain generalization by composition, to the population local denoiser (4): with suficient aggregation, retrieval recovers its empirical counterpart exactly. However, requiring all training-context diferences to lie in a single low-dimensional subspace is restrictive. This motivates the nonlinear DiT aggregator considered next.

## 4 DiTs as Adaptive Local Denoisers

We next relate DiT attention and MLPs to the aggregator and predictor roles of the local-denoising framework. We consider a single layer without residual connections $\hat { D } _ { m , t } ^ { \mathrm { d i } \hat { \mathrm { t } } } ( y ) = ( g _ { t } ^ { \mathrm { m l p } } \circ S _ { m , t } ^ { \mathrm { a t t n } } ) ( y )$ , where the attention aggregator $S _ { m , t } ^ { \mathrm { a t t n } }$ and MLP predictor ${ g } _ { t } ^ { \mathrm { m l p } }$ are defined in the following sections. Appendix D gives the derivation and residual-stack extension; Fig. 6 summarizes the component mapping. Note that in a residual stack, VO and MLP updates can alter later QK routing, and attention can afect later MLP updates, so the branch-level algebra does not necessarily imply a functionally separable decomposition of the full network.

## 4.1 Attention as an input-adaptive aggregator

PLD’s linear aggregator (5) uses a fixed context rule across inputs. A more flexible denoiser can adapt its context to the input, drawing on diferent patches as their relevance changes with the image and noise level. We express this through an input-adaptive aggregator

$$
\mathcal { S } _ { m , t } ( Y _ { t } ) = \sum _ { l = 1 } ^ { M } S _ { m l } ( Y _ { t } ; t ) \Phi ( Y _ { t , l } ; t , l ) .\tag{6}
$$

Here $\Phi ( Y _ { t , l } ; t , l ) \in \mathbb { R } ^ { d _ { h } }$ represents patch l together with its position and noise level, while $S _ { m l } ( Y _ { t } ; t ) \in \mathbb { R } ^ { d _ { h } \times d _ { h } }$ determines how it contributes to the context for denoising patch m.

DiT attention, incorporating patch, position, and time embeddings together with AdaLN conditioning, provides a factorized parameterization of such an input-adaptive aggregator. Let $E \in \mathbb { R } ^ { d _ { h } \times d _ { p } }$ be the patch projection, $e _ { l } \in \mathbb { R } ^ { d _ { h } }$ the positional embedding, and $\phi _ { t , l } ( Y _ { t } ) : = \Phi ^ { \mathrm { a t t n } } ( E Y _ { t , l } ; e _ { l } , t )$ the feature formed by patch embedding, position embedding, and AdaLN. For each of $\tilde { H }$ equal-width heads, $\tilde { W } _ { h } ^ { Q } , \tilde { W } _ { h } ^ { K }$ , and $\tilde { W } _ { h } ^ { V }$ are the query (Q), key (K), and value (V) projections, and $\tilde { W } _ { h } ^ { O }$ is the output (O) projection. Writing the attention AdaLN residual gate as $\tilde { \alpha } _ { t } \in \mathbb { R } ^ { d _ { h } }$ , the attention branch constitutes an attention aggregator defined as

$$
\begin{array} { l } { { \displaystyle { \mathcal S } _ { m , t } ^ { \mathrm { a t t n } } ( Y _ { t } ) = \sum _ { l = 1 } ^ { M } S _ { m l } ^ { \mathrm { a t t n } } ( Y _ { t } ; t ) \phi _ { t , l } ( Y _ { t } ) } , } \\ { { \displaystyle S _ { m l } ^ { \mathrm { a t t n } } ( Y _ { t } ; t ) = \sum _ { h = 1 } ^ { \tilde { H } } \mathrm { s m } _ { l } \left\{ \frac { \langle \tilde { W } _ { h } ^ { Q } \phi _ { t , m } ( Y _ { t } ) , \tilde { W } _ { h } ^ { K } \phi _ { t , j } ( Y _ { t } ) \rangle } { ( d _ { h } / \tilde { H } ) ^ { 1 / 2 } } \right\} _ { j \in [ M ] } \mathrm { d i a g } ( \tilde { \alpha } _ { t } ) { \tilde { W } } _ { h } ^ { O } { \tilde { W } } _ { h } ^ { V } . } } \end{array}\tag{7}
$$

A single attention branch separates QK spatial weighting from VO value transformation and mixing (7). The QK projections determine the compatibility scores of patches through $( \tilde { W } _ { h } ^ { Q } ) ^ { \top } \tilde { W } _ { h } ^ { K }$ , and softmax converts these scores into spatial weights. By smoothly suppressing patches with low relative scores, attention performs soft thresholding and forms an approximately local aggregator when its weights concentrate on a suitable context. In this branch-level sense, QK parameterizes which patch positions receive greater or smaller weight in the constructed context, depending on image content, position, and noise level. Appendix D.3 details the local approximation.

VO transforms and combines the selected features through the shared, time-dependent channel operators $\mathrm { d i a g } ( \tilde { \alpha } _ { t } ) \tilde { W } _ { h } ^ { O } \tilde { W } _ { h } ^ { V }$ . Attention thus combines $\tilde { H }$ shared channel operators with input-dependent spatial weights, rather than learning an independent operator for every patch pair. This decomposition motivates an aggregator interpretation: for fixed input features, QK directly controls relative spatial weights, while VO applies patch shared channel operators to the weighted features. The downstream impact of an attention weight also depends on the value features, channel operators, and residual interactions. These roles interact through learned features and across layers, but the distinction motivates examining QK and VO separately in Sec. 5.3.

## 4.2 MLP and shared-memory patch predictors

Within the single-layer model, the DiT MLP gtp ${ g } _ { t } ^ { \mathrm { m l p } }$ acts on the attention-aggregated context $s = S _ { m , t } ^ { \mathrm { a t t n } } ( Y _ { t } )$ in the patch-predictor role; in a residual stack, its output is a tokenwise feature update. The framework also allows alternative patch predictors, such as the RBF predictor $g _ { m , t } ^ { \mathrm { r b f } }$ in (5), which explicitly retrieves from training patches. In a multilayer residual stack (Appendix D.5), these predictors can share memory across layers, just as $g _ { m , t } ^ { \mathrm { r b f } }$ would consult the same training patches at each layer. This motivates a sequence of shared-memory patch predictors connecting explicit patch retrieval to an ordinary MLP. We present simplified forms below; Appendix D.4 gives the full parameterizations, including biases, output gates, compatibility projections, and multiple heads.

We first note that ${ g } _ { t } ^ { \mathrm { m l p } }$ and $g _ { m , t } ^ { \mathrm { r b f } }$ admit similar key–value memory interpretations (Geva et al., 2021). If coupled with the attention aggregator, $g _ { m , t } ^ { \mathrm { r b f } }$ retrieves from patch-specific values $\{ x _ { m } ^ { ( i ) } \} _ { 1 : n }$ using contextualized training images $S _ { m , t } ^ { \mathrm { a t t n } } ( x ^ { ( i ) } )$ as keys. A transformer MLP of hidden width R likewise consists of learnable key–value pairs $\{ k ^ { ( r ) } , v ^ { ( r ) } \in \mathbb { R } ^ { d _ { h } } \} _ { r = 1 } ^ { R }$ . In DiTs, AdaLN modulates the memory query to produce time-dependent retrieval weights:

$$
g _ { t } ^ { \mathrm { m l p } } ( s ) = \sum _ { r = 1 } ^ { R } \mathrm { g e l u } \Big ( \mathrm { A d a L N } _ { t } ( s ) ^ { \top } k ^ { ( r ) } \Big ) v ^ { ( r ) } .\tag{8}
$$

Compared with $g _ { m , t } ^ { \mathrm { r b f } }$ in (5), there are three diferences: (i) the memory is learned rather than drawn from training patches, (ii) the same bank of memory is used across patches, and (iii) the retrieval weights are unnormalized and may have either sign. These diferences motivate three patch-predictor variants that start from $g _ { m , t } ^ { \mathrm { r b f } }$ and progressively resemble ${ \overset { \sim } { g _ { t } ^ { \mathrm { m l p } } } }$ : we first make the memory learnable, then share it across patches, and finally replace normalized retrieval weights with unnormalized, potentially signed coeficients.

All predictors act on the same contextualized patch $s = S _ { m , t } ^ { \mathrm { a t t n } } ( Y _ { t } )$ ). Let $W ^ { Q } , W ^ { K } , W ^ { V }$ be layerwise projections, with softmax taken over the R memory entries.

$$
g _ { m , t } ^ { \mathrm { l p l d } } ( s ) = \sum _ { r = 1 } ^ { R } \mathrm { s m } _ { r } \left\{ \mathrm { A d a L N } _ { t } ( s ) ^ { \top } ( W ^ { Q } ) ^ { \top } W ^ { K } { \mathcal S } _ { m , t } ^ { \mathrm { a t t n } } ( z ^ { ( j ) } ) \right\} _ { j \in [ R ] } W ^ { V } z _ { m } ^ { ( r ) } ,\tag{9}
$$

$$
g _ { t } ^ { \mathrm { s c a } } ( s ) = \sum _ { r = 1 } ^ { R } \mathrm { s m } _ { r } \left\{ \mathrm { A d a L N } _ { t } ( s ) ^ { \top } ( W ^ { Q } ) ^ { \top } W ^ { K } k ^ { ( j ) } \right\} _ { j \in [ R ] } W ^ { V } v ^ { ( r ) } ,\tag{10}
$$

$$
g _ { t } ^ { \mathrm { s m l p } } ( s ) = \sum _ { r = 1 } ^ { R } \mathrm { g e l u } \Big ( \mathrm { A d a L N } _ { t } ( s ) ^ { \top } ( W ^ { Q } ) ^ { \top } W ^ { K } k ^ { ( r ) } \Big ) W ^ { V } v ^ { ( r ) } .\tag{11}
$$

Compared to PLD, the first variant (9) replaces the training images with a learned memory bank $\{ z ^ { ( r ) } \in$ $\mathbb { R } ^ { M d _ { h } ^ { \ f } } \} _ { r = 1 } ^ { R }$ shared across layers, and RBF retrieval with softmax. Just as a training image $x ^ { ( i ) }$ consists of M patches, each memory entry $z ^ { ( r ) }$ consists of M learnable patches $z _ { m } ^ { ( r ) } \in \mathbb { R } ^ { d _ { h } ^ { \prime } }$ . The memory width $d _ { h } ^ { \prime }$ may equal the context width $d _ { h }$ or difer from it; a learned compatibility projection shared across layers accommodates difering widths. Together, these choices define the learned PLD (LPLD) predictor $g _ { m , t } ^ { \mathrm { l p } \mathrm { i d } }$ . LPLD uses the attention-aggregated memory $z ^ { ( r ) }$ to form keys, whereas the value is taken directly from its mth patch.

Storing a separate set of memory patches at each patch position can be redundant, especially given that patchification is an architectural choice. In (10) and (11), the patch-specific memory bank in $g _ { m , t } ^ { \mathrm { l p l d } }$ is replaced with an untied key–value bank $\{ k ^ { ( r ) } , v ^ { ( r ) } \in \mathbb { R } ^ { d _ { h } ^ { \prime } } \} _ { r = 1 } ^ { R }$ shared across patches and layers. Coupled with the softmax retrieval, this gives a shared cross-attention (SCA) predictor $g _ { t } ^ { \mathrm { s c a } }$ . Finally, we replace normalized softmax weights with unnormalized GELU coeficients for a more flexible retrieval rule. This gives a shared MLP (SMLP) predictor $g _ { t } ^ { \mathrm { s m l p } }$ . Absorbing the projection matrices into the keys and values allows SMLP to recover the MLP form (8). For all three variants, sharing the memory bank across layers can reduce the parameter count, allowing us to allocate more capacity to predictor heads at matched parameter budgets.

## 5 Experiments

Our experiments examine whether trained DiT denoisers exhibit locality patterns resembling those of the oracle GMRF denoiser (Sec. 5.1), and whether, as suggested by our aggregator–predictor framework, layerwise

![](images/3d79723da580b230d68284603c6fe4011567c6f5e5f6345bad7ed6ed3bf6f15e.jpg)  
Figure 3 Sensitivity-defined contexts broaden with noise, with attention mass showing a related trend. Gold/white contours enclose 50%/95% of normalized squared sensitivity or attention mass; ticks point inward and cyan dots mark target patches. Right: 95% radii averaged over target patches; Details in Appendix E.2.

MLPs can be replaced with more interpretable, shared-memory patch predictors (Sec. 5.2). Finally, we use component transplantation to examine whether compatibility patterns among jointly adapted QK, VO, and MLP parameters are consistent with this framework (Sec. 5.3). Experimental details and additional results are in Appendix E.

## 5.1 Real-image denoising echoes GMRF locality

An exact analysis as in Fig. 2 is unavailable for real images. Instead, we identify context sets using sensitivity weights, which quantify how the prediction of patch m responds to perturbations of input patch l, based on a trained DiT’s Jacobian, similar to Niedoba et al. (2025). The population denoiser’s Jacobian measures posterior dependence between clean patches given the noisy patches (Dytso et al., 2021). Based on these weights, we define the q-sensitivity radius as the smallest radius of a square centered on patch m containing at least a fraction q of the total weight. The sensitivity context set contains the fewest patches carrying at least that fraction.

Denoising sensitivity exhibits GMRF-like broadening; attention shows a related trend. For the synthetic GMRF where the true Jacobian is available, Fig. 2(c) shows that sensitivity spreads over broader contexts as noise increases. Denoising real images echoes this pattern: Fig. 3 shows concentrated sensitivity at low noise and broader, sometimes disconnected contexts at higher noise, together with increasing sensitivity radii. Notably, replacing sensitivity weights with attention mass yields analogous attention contexts and radii, which also generally broaden with noise. Together, these observations support the GMRF setting as a tractable reference model for denoising locality and the interpretation of attention as an input-adaptive context aggregator. Sensitivity and attention mass measure diferent quantities, so the comparison concerns a shared qualitative trend rather than identical context sets. Additional results appear in Appendix E.2.

![](images/b54299c02c1a2a4b1be3bcc91434333020d045f49b386cf0c6ee24121ade2525.jpg)

![](images/2bfc7154632243790e18fce33bd1d6349e1eca647fb29c335ffa7fceaf28a0e4.jpg)

![](images/a45b4e093d3e6b1360c293d756e23cd2a87564152cb12c51fc9826ee8fee7d73.jpg)  
Figure 4 Shared-memory patch predictors at comparable parameter budgets. Models are trained on CelebA64 for 400 epochs; see Appendix E.3 for details.

(b) VO coordination

## 5.2 Shared-memory predictors can replace MLPs in a controlled setting

We replace DiT MLPs with LPLD, SCA, and SMLP on CelebA64. We compare similar parameter budgets using a common backbone architecture, with modest models and one training seed per configuration for unconditional 64 × 64 face generation. Figure 4 reports held-out velocity loss, denoising error, and FID at epoch 400. Three patterns appear; first, the patch-shared SCA and SMLP banks outperform the positionspecific LPLD bank at comparable parameter budgets, showing that competitive prediction need not assign separate memory for every patch position. Second, SMLP improves on SCA, supporting unnormalized GELU retrieval as an efective alternative to softmax. Third, SMLP with H = 8 achieves competitive denoising and lower FID than the ordinary MLP, despite sharing its base memory across layers. Multiple predictor heads generally improve performance, suggesting that multiple projections of the shared bank can be useful. Together, these results demonstrate that an explicit, layer-shared memory bank can support competitive patch prediction in this controlled setting. We note that these results by themselves do not establish that layerwise MLP parameters generally encode a single shared memory or that this design transfers unchanged to larger conditional DiTs. Training details and limitations are in Appendix E.3.

![](images/1cdbbda02dc50245cbeadd91b610901d8bab6971054bf0be119191783b15b407.jpg)

![](images/407391c87e9daee5333a1f5f7daa6966b7717c8bdbe68b508be6e81220672067.jpg)

![](images/4689b90a90d081f17a85c8c4041103d3aaea10ad4a3ecbf871582e1cdd585603.jpg)  
Figure 5 QK, VO, and MLP component transplants. (a) Source-QK compatibility is higher across spatially similar datasets. (b) VO reversion costs more as joint non-additivity increases. (c) Natural-image source QK retains partial agreement on face targets. See Appendix E.4.2.

## 5.3 Component transplants support the aggregator–predictor view

Transplanting jointly adapted components probes compatibility and coadaptation. To examine compatibility patterns in relation to the proposed component roles, we fine-tune a pretrained source model on a target dataset. We then transplant combinations of the adapted components into the source checkpoint. When the datasets have similar spatial structure, our framework suggests that source QK may remain compatible with adapted VO and MLP parameters. Larger spatial shifts may be associated with stronger coadaptation. We study twelve transfer directions between face and natural-image datasets. For clarity, we jointly fine-tune only QK, VO, and MLP, leaving AdaLN and all other parameters fixed, assuming that the learned time conditioning transfers across domains. Each of the eight resulting hybrids is labeled by QK/VO/MLP bits: 0 for source weights and 1 for jointly adapted weights. These transplants probe the compatibility and coadaptation of parameters learned jointly; they do not measure the performance obtainable by training QK-, VO-, or MLP-only models. Separately, under the same target-domain update budget, QK-only and QK+VO-only fine-tuning generally fall further behind joint adaptation on larger domain shifts (Appendix E.4.5).

Following work on patch self-similarity and spatial normalization (Tumanyan et al., 2022; Singh et al., 2026), we measure spatial distance between datasets using $\mathrm { M M D ^ { 2 } }$ of patch self-similarity descriptors. We compare each hybrid’s outputs with the fully adapted model’s using matched initial noise. We measure this agreement using DINOv2 global features for endpoint agreement (EA) and patch self-similarity patterns for spatial agreement (SA). Both scores assign 0 to the source and 1 to the fully adapted model. Full protocols and results are in Appendix E.4.

QK compatibility is higher for spatially similar datasets, while VO reversion penalties track component coadaptation. With source QK retained (011), CelebA→AAHQ/FFHQ/AFHQ produces recognizable faces. In contrast, among the tested CelebA→STL-10 hybrids, coherent target-like structure emerges primarily only when all jointly adapted components are present (111; Figs. 1 and 13). Consistent with this pattern, the

QK reversion penalty tends to increase with spatial distance (Fig. 5(a)), which also accompanies stronger component coadaptation (Appendix E.4.3). Additionally, restoring source VO (101) yields fragmented textures in the natural-image-to-face transfers (Figs. 14 and 15). Its reversion penalty increases with joint non-additivity—the gain from joint adaptation beyond the sum of isolated component changes (Fig. 5(b)).

Source QK from broader natural-image domains retains partial compatibility on face targets. With source QK retained (011), STL-10 hybrids recover human and animal face structure, including a blurred face for STL-10→CelebA, while joint adaptation produces clearer results (Fig. 14). This contrasts with the poorly formed scene in the reverse CelebA→STL-10 direction (Fig. 1); additional examples in Appendix E.4.4. Figure 5(c) supports this asymmetry: source QK from STL-10 retains greater endpoint agreement on CelebA than in the reverse direction, while QK from natural-image domains retains partial agreement on face targets at levels comparable to CelebA QK.

## 6 Conclusion and limitations

We presented an aggregator–predictor view of difusion transformers as input-adaptive local denoisers. Our theory justifies local denoising under GMRF priors and connects patchwise linear denoisers to the empirical local denoising target. Within this framework, QK determines spatial context weights, VO transports features, and MLPs perform contextualized patch prediction. Experiments provide complementary evidence through noise-dependent denoising contexts, competitive shared-memory predictors, and cross-domain component compatibility patterns. Together, these findings ofer a statistical and architectural perspective on DiT denoising and motivate further work on interpretable predictors and targeted model adaptation.

Our experiments focus on modest-sized, unconditional DiTs operating on 64 × 64 images across a limited collection of face and natural-image datasets. In particular, the shared-memory predictor comparison uses CelebA64 with one training seed per configuration, leaving its robustness across datasets, model scales, and training seeds unresolved. The transfer study examines twelve directions while keeping AdaLN and global parameters fixed; its component transplants probe compatibility among jointly adapted weights and do not uniquely identify the proposed functional roles. Testing these patterns in larger conditional models, at higher resolutions, and under broader adaptation protocols is an important next step.

## AI use statement

Generative AI tools were used as research assistants for literature review, refining the presentation of the work, writing and cleaning up code, assisting with experiment execution, cleaning up proofs, providing feedback on the manuscript, and polishing the text. All AI-assisted results were independently reviewed and verified by the authors. The authors made the final scientific and methodological decisions and take full responsibility for the contents of the paper.

## Reproducibility statement

Assumptions and proofs are in Appendix B. Appendix E documents the datasets, model configurations, training and evaluation protocols, and metric definitions for all three experiments. We provide code and reproduction instructions for dataset preparation, training, evaluation, and regeneration of the figures and tables, together with the small numerical summaries used to render the reported plots.

## References

Jie An, De Wang, Pengsheng Guo, Jiebo Luo, and Alex Schwing. On inductive biases that enable generalization in difusion transformers. In Advances in Neural Information Processing Systems, volume 38, pages 31738–31757. Curran Associates, Inc., 2025. doi: 10.52202/085713-1066. https://papers.neurips.cc/paper\_files/paper/ 2025/hash/2dc52e27a26afabb0bf2123bceaf9208-Abstract-Conference.html.

Harry C Andrews and Bobby R Hunt. Digital image restoration. Prentice-Hall, 1977.

Haim Avron and Sivan Toledo. Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix. Journal of the ACM, 58(2):8:1–8:17, 2011. doi: 10.1145/1944345.1944349. https://www.cs. tau.ac.il/\~stoledo/Bib/Pubs/randomized-trace.pdf.

Dmitry Baranchuk, Ivan Rubachev, Andrey Voynov, Valentin Khrulkov, and Artem Babenko. Label-eficient semantic segmentation with difusion models. In International Conference on Learning Representations, 2022.

Samyadeep Basu, Martin Grayson, Cecily Morrison, Besmira Nushi, Soheil Feizi, and Daniela Massiceti. Understanding information storage and transfer in multi-modal large language models. Advances in Neural Information Processing Systems, 37, 2024a.

Samyadeep Basu, Keivan Rezaei, Priyatham Kattakinda, Vlad I. Morariu, Nanxuan Zhao, Ryan A. Rossi, Varun Manjunatha, and Soheil Feizi. On mechanistic knowledge localization in text-to-image generative models. In Proceedings of the 41st International Conference on Machine Learning, 2024b.

Samyadeep Basu, Nanxuan Zhao, Vlad I. Morariu, Soheil Feizi, and Varun Manjunatha. Localizing and editing knowledge in text-to-image generative models. In International Conference on Learning Representations, 2024c.

Michele Benzi and Nader Razouk. Decay bounds and O(n) algorithms for approximating functions of sparse matrices. Electronic Transactions on Numerical Analysis, 28:16–39, 2007.

Julian Besag. Spatial interaction and the statistical analysis of lattice systems. Journal of the Royal Statistical Society: Series B (Methodological), 36(2):192–236, 1974.

Giulio Biroli, Tony Bonnaire, Valentin de Bortoli, and Marc Mézard. Dynamical regimes of difusion models. Nature Communications, 15:9957, 2024. doi: 10.1038/s41467-024-54281-3. https://www.nature.com/articles/ s41467-024-54281-3.

Tony Bonnaire, Raphaël Urfin, Giulio Biroli, and Marc Mézard. Why difusion models don’t memorize: The role of implicit dynamical regularization in training. In Advances in Neural Information Processing Systems, 2025. https://arxiv.org/abs/2505.17638.

Rama Chellappa and Shankar Chatterjee. Classification of textures using Gaussian Markov random fields. IEEE Transactions on Acoustics, Speech, and Signal Processing, 33(4):959–963, 1985. doi: 10.1109/TASSP.1985.1164641.

Xinlei Chen, Zhuang Liu, Saining Xie, and Kaiming He. Deconstructing denoising difusion models for self-supervised learning. In International Conference on Learning Representations, 2025. https://proceedings.iclr.cc/paper\_ files/paper/2025/hash/8b52c29d1a77e3fcc6c8f90fa0ac0e8e-Abstract-Conference.html.

Yunjin Chen, René Ranftl, and Thomas Pock. Insights into analysis operator learning: From patch-based sparse models to higher-order MRFs. IEEE Transactions on Image Processing, 23(3):1060–1072, 2014.

Caroline Choi, Yoonho Lee, Annie Chen, Allan Zhou, Aditi Raghunathan, and Chelsea Finn. AutoFT: Learning an objective for robust fine-tuning, 2024. https://arxiv.org/abs/2401.10220.

Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. StarGAN v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. https://arxiv.org/abs/1912.01865.

Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. A downsampled variant of ImageNet as an alternative to the CIFAR datasets. arXiv preprint arXiv:1707.08819, 2017. https://arxiv.org/abs/1707.08819.

Adam Coates, Honglak Lee, and Andrew Ng. An analysis of single-layer networks in unsupervised feature learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pages 215–223, 2011. https://proceedings.mlr.press/v15/coates11a. html.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8493–8502, 2022.

DLMF. NIST Digital Library of Mathematical Functions. Release 1.2.8, 2026. https://dlmf.nist.gov/18.12.E8. Released September 15, 2026. F. W. J. Olver, A. B. Olde Daalhuis, D. W. Lozier, B. I. Schneider, R. F. Boisvert, C. W. Clark, B. R. Miller, B. V. Saunders, H. S. Cohl, and M. A. McClain, editors.

Alex Dytso, H. Vincent Poor, and Shlomo Shamai (Shitz). A general derivative identity for the conditional mean estimator in Gaussian noise and some applications. arXiv preprint arXiv:2104.01883, 2021. https://arxiv.org/ abs/2104.01883.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 12606–12633. PMLR, 2024. https://proceedings.mlr.press/v235/esser24a.html.

Alessandro Favero, Antonio Sclocchi, Francesco Cagnetta, Pascal Frossard, and Matthieu Wyart. How compositional generalization and creativity improve as difusion models are trained. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 16286–16306. PMLR, 2025. https://proceedings.mlr.press/v267/favero25a.html.

Chaofan Gan, Yuanpeng Tu, Xi Chen, Tieyuan Chen, Yuxi Li, Mehrtash Harandi, and Weiyao Lin. Unleashing difusion transformers for visual correspondence by modulating massive activations. In Advances in Neural Information Processing Systems, volume 38, 2025. https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ a654298de44ec95dec28aa37e8d0af51-Abstract-Conference.html.

Stuart Geman and Donald Geman. Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-6(6):721–741, 1984. doi: 10.1109/TPAMI. 1984.4767596.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484–5495, 2021.

Mor Geva, Avi Caciularu, Kevin Ro Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022.

Klaus Gref, Rupesh K. Srivastava, and Jürgen Schmidhuber. Highway and residual networks learn unrolled iterative estimation. In International Conference on Learning Representations, 2017. https://openreview.net/forum?id= Skn9Shcxe.

Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Sch"olkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 13:723–773, 2012. https://www.jmlr.org/papers/v13/ gretton12a.html.

Xiangming Gu, Chao Du, Tianyu Pang, Chongxuan Li, Min Lin, and Ye Wang. On memorization in difusion models. Transactions on Machine Learning Research, 2025. https://openreview.net/forum?id=D3DBqvSDbj.

Andi Han, Wei Huang, Yuan Cao, and Difan Zou. On the feature learning in difusion models. In International Conference on Learning Representations, 2025a. https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html.

Yujin Han, Andi Han, Wei Huang, Chaochao Lu, and Difan Zou. Can difusion models learn hidden inter-feature rules behind images? In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 21704–21732. PMLR, 2025b. https://proceedings.mlr.press/ v267/han25b.html.

Dominik Hintersdorf, Lukas Struppek, Kristian Kersting, Adam Dziedzic, and Franziska Boenisch. Finding NeMo: Localizing neurons responsible for memorization in difusion models. In Advances in Neural Information Processing Systems, 2024.

Benjamin Hoover, Hendrik Strobelt, Dmitry Krotov, Judy Hofman, Zsolt Kira, and Duen Horng Chau. Memory in plain sight: Surveying the uncanny resemblances of associative memories and difusion models, 2023. https: //arxiv.org/abs/2309.16750.

Stanislaw Jastrzebski, Devansh Arpit, Nicolas Ballas, Vikas Verma, Tong Che, and Yoshua Bengio. Residual connections encourage iterative inference. In International Conference on Learning Representations, 2018. https: //openreview.net/forum?id=SJa9iHgAZ.

Zahra Kadkhodaie, Florentin Guth, Eero P. Simoncelli, and Stéphane Mallat. Generalization in difusion models arises from geometry-adaptive harmonic representations. In International Conference on Learning Representations, 2024. https://openreview.net/forum?id=ANvmVS2Yr0.

Mason Kamb and Surya Ganguli. An analytic theory of creativity in convolutional difusion models. In Proceedings of

the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 28795–28831. PMLR, 2025. https://proceedings.mlr.press/v267/kamb25a.html.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. https://arxiv. org/abs/1812.04948.

Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. In Advances in Neural Information Processing Systems, 2021. https://nvlabs. github.io/stylegan3/.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image difusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recogni tion, 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Kumari\_Multi-Concept\_Customization\_ of\_Text-to-Image\_Diffusion\_CVPR\_2023\_paper.html

Stan Z Li. Markov random field modeling in image analysis. Springer, 3rd edition, 2009.

Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 36115–36125, June 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Li\_Back\_to\_Basics\_Let\_Denoising\_ Generative\_Models\_Denoise\_CVPR\_2026\_paper.html.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. https://openreview.net/forum?id= PqvMRDCJT9t.

Mingcong Liu, Qiang Li, Zekui Qin, Guoxin Zhang, Pengfei Wan, and Wen Zheng. BlendGAN: Implicitly GAN blending for arbitrary stylized face generation. In Advances in Neural Information Processing Systems, 2021. https://arxiv.org/abs/2110.11728.

Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones: Concept neurons in difusion models for customized generation. In Proceedings of the 40th International Conference on Machine Learning, 2023.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision, 2015. https://liuziwei7.github.io/projects/ FaceAttributes.html.

Artem Lukoianov, Chenyang Yuan, Justin Solomon, and Vincent Sitzmann. Locality in image difusion models emerges from data statistics. In Advances in Neural Information Processing Systems, 2025. https://papers.nips.cc/ paper\_files/paper/2025/hash/88ad9774ffcb7a272457e9396f793a07-Abstract-Conference.html.

Song Mei. U-Nets as belief propagation: Eficient classification, denoising, and difusion in generative hierarchical models. In International Conference on Learning Representations, 2025. https://proceedings.iclr.cc/paper\_ files/paper/2025/hash/cb3658b9983f677670a246c46ece553d-Abstract-Conference.html.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. Advances in Neural Information Processing Systems, 35:17359–17372, 2022.

Kevin Meng, Arnab Sen Sharma, Alex J Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. In The Eleventh International Conference on Learning Representations, 2023.

Yuki Monma, Kan Aro, and Muneki Yasuda. Hierarchical Gaussian Markov random field for image denoising. IEICE Transactions on Information and Systems, E105-D(3):689–699, 2022. doi: 10.1587/transinf.2021EDP7172.

Matthew Niedoba, Berend Zwartsenberg, Kevin Patrick Murphy, and Frank Wood. Towards a mechanistic explanation of difusion model generalization. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 46389–46411. PMLR, 2025. https://proceedings.mlr. press/v267/niedoba25a.html.

Maya Okawa, Ekdeep Singh Lubana, Robert P. Dick, and Hidenori Tanaka. Compositional abilities emerge multi plicatively: Exploring difusion models on a synthetic task. In Advances in Neural Information Processing Systems, 2023.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. https://openreview.net/forum?id=a68SUt6zFt.

Haowen Pan, Yixin Cao, Xiaozhi Wang, Xun Yang, and Meng Wang. Finding and editing multi-modal neurons in pre-trained transformers. In Findings of the Association for Computational Linguistics: ACL 2024, pages 1012–1037, 2024a.

Kaihang Pan, Zhaoyu Fan, Juncheng Li, Qifan Yu, Hao Fei, Siliang Tang, Richang Hong, Hanwang Zhang, and Qianru Sun. Towards unified multimodal editing with enhanced knowledge collaboration. In Advances in Neural Information Processing Systems, volume 37, 2024b.

William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023. https://openaccess.thecvf.com/content/ ICCV2023/html/Peebles\_Scalable\_Diffusion\_Models\_with\_Transformers\_ICCV\_2023\_paper.html.

Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller Language models as knowledge bases? In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019.

Jakiw Pidstrigach. Score-based generative models detect manifolds. In Advances in Neural Information Processing Systems, 2022.

Emile Pierret and Bruno Galerne. Difusion models for Gaussian distributions: Exact solutions and Wasserstein errors. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 49355–49381. PMLR, 2025. https://proceedings.mlr.press/v267/pierret25a.html.

Paul Rosu, Lawrence Carin, and Xiang Cheng. From softmax to score: Transformers can efectively implement incontext denoising steps. In Advances in Neural Information Processing Systems, volume 38, 2025. https://papers. nips.cc/paper\_files/paper/2025/hash/d27af2bebeab8a3e6f3848fc71736235-Abstract-Conference.html.

Stefan Roth and Michael J. Black. Fields of experts: A framework for learning image priors. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition, volume 2, pages 860–867, 2005. doi: 10.1109/CVPR.2005. 160. https://download.visinf.tu-darmstadt.de/papers/2005-cvpr-roth-fields\_of\_experts-preprint.pdf.

Håvard Rue and Leonhard Held. Gaussian Markov Random Fields: Theory and Applications. Chapman & Hall/CRC, 2005.

Sarah Schwettmann, Neil Chowdhury, Samuel Klein, David Bau, and Antonio Torralba. Multimodal neurons in pretrained text-only transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 2862–2867, 2023.

Joonghyuk Shin, Alchan Hwang, Yujin Kim, Daneul Kim, and Jaesik Park. Exploring multimodal difusion trans formers for enhanced prompt-based image editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Shin\_Exploring\_Multimodal\_ Diffusion\_Transformers\_for\_Enhanced\_Prompt-based\_Image\_Editing\_ICCV\_2025\_paper.html.

Hayaru Shouno. Bayesian image restoration for Poisson corrupted image using a latent variational method with Gaussian MRF. IPSJ Online Transactions, 8:15–24, 2015. doi: 10.2197/ipsjtrans.8.15.

Jaskirat Singh, Xingjian Leng, Zongze Wu, Liang Zheng, Richard Zhang, Eli Shechtman, and Saining Xie. What matters for representation alignment: Global information or spatial structure? In International Conference on Learning Representations, 2026. https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 3929a7785bd56f57edcff0152ab41289-Abstract-Conference.html.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic diferential equations. In International Conference on Learning Representations, 2021. https://openreview.net/forum?id=PxTIG12RRHS.

Sainbayar Sukhbaatar, Edouard Grave, Guillaume Lample, Hervé Jégou, and Armand Joulin. Augmenting self-attention with persistent memory. arXiv preprint arXiv:1907.01470, 2019.

Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image difusion. In Advances in Neural Information Processing Systems, volume 36, 2023. https://proceedings. neurips.cc/paper\_files/paper/2023/hash/0503f5dce343a1d06d16ba103dd52db1-Abstract-Conference.html.

Marshall F. Tappen, Ce Liu, Edward H. Adelson, and William T. Freeman. Learning Gaussian Conditional Random Fields for low-level vision. In IEEE Conference on Computer Vision and Pattern Recognition, 2007. doi: 10.1109/ CVPR.2007.382979. https://people.csail.mit.edu/billf/www/papers/cvpr07-gcrf.pdf.

Narek Tumanyan, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Splicing ViT features for semantic appearance transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Tumanyan\_Splicing\_ViT\_Features\_for\_ Semantic\_Appearance\_Transfer\_CVPR\_2022\_paper.html.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play difusion features for text-driven imageto-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Tumanyan\_Plug-and-Play\_ Diffusion\_Features\_for\_Text-Driven\_Image-to-Image\_Translation\_CVPR\_2023\_paper.html.

Evelyn Turri, Davide Bucciarelli, Sara Sarto, Lorenzo Baraldi, and Marcella Cornia. Few channels draw the whole picture: Revealing massive activations in difusion transformers, 2026. https://arxiv.org/abs/2605.13974.

Binxu Wang and John J. Vastola. The unreasonable efectiveness of Gaussian score approximation for difusion models and its applications. Transactions on Machine Learning Research, 2024.

Yair Weiss and William T Freeman. What makes a good model of natural images? In 2007 IEEE Conference on Computer Vision and Pattern Recognition, pages 1–8, 2007.

Norbert Wiener. Extrapolation, interpolation, and smoothing of stationary time series. MIT Press, 1949.

Wikipedia contributors. Chebyshev polynomials. Wikipedia, The Free Encyclopedia, 2026. https://en.wikipedia. org/wiki/Chebyshev\_polynomials. Accessed September 2, 2026.

Yu-Han Wu, Pierre Marion, Gérard Biau, and Claire Boyer. Taking a big step: Large learning rates in denoising score matching prevent memorization. In Proceedings of the Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 5718–5756. PMLR, 2025. https://proceedings.mlr.press/ v291/wu25a.html.

Weilai Xiang, Hongyu Yang, Di Huang, and Yunhong Wang. Denoising difusion autoencoders are unified self-supervised learners. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han. SANA: Eficient high-resolution text-to-image synthesis with linear difusion transformers. In International Conference on Learning Representations, 2025. https://proceedings.iclr.cc/paper\_files/paper/ 2025/hash/34345e243156da67605d4b63d71c8d98-Abstract-Conference.html.

Muneki Yasuda, Junpei Watanabe, Shun Kataoka, and Kazuyuki Tanaka. Linear-time algorithm in bayesian image denoising based on Gaussian Markov random field. IEICE Transactions on Information and Systems, E101-D(6): 1629–1639, 2018. doi: 10.1587/transinf.2017EDP7346.

TaeHo Yoon, Joo Young Choi, Sehyun Kwon, and Ernest K. Ryu. Difusion probabilistic models generalize when they fail to memorize. In ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling, 2023.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training difusion transformers is easier than you think. In International Conference on Learning Representations, 2025. https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ d9e42b4d7163931f3689d6d6fbaa11d0-Abstract-Conference.html.

Arman Zarei, Samyadeep Basu, Keivan Rezaei, Zihao Lin, Sayan Nag, and Soheil Feizi. Localizing knowledge in difusion transformers. In Advances in Neural Information Processing Systems, volume 38, 2025. https://proceedings. neurips.cc/paper\_files/paper/2025/hash/157a6107a95883f6ccc0e38e80a24295-Abstract-Conference.html.

Zekai Zhang, Xiao Li, Xiang Li, Lianghe Shi, Meng Wu, Molei Tao, and Qing Qu. Generalization of difusion models arises with a balanced representation space. In International Conference on Learning Representations, 2026. https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ a5e3cf29c269b041ccd644b6beaf5c42-Abstract-Conference.html.

Yi Zheng, Yifan Zhao, Mengyuan Ren, He Yan, Xiangju Lu, Junhui Liu, and Jia Li. Cartoon face recognition: A benchmark dataset. In Proceedings of the 28th ACM International Conference on Multimedia, pages 2264–2272, 2020. doi: 10.1145/3394171.3413726. https://doi.org/10.1145/3394171.3413726.

## Appendix contents

A Related Literature

B Proofs 18   
B.1 Notation . 18   
B.2 Background facts 18   
C Main proofs 19   
C.1 Full statement and proof of Theorem 1 19   
C.2 Full statement and proof of Corollary 2 22   
C.3 Proof of Theorem 3 . 22   
D From the aggregator–predictor construction to implementation 23   
D.1 Patch, position, and time conditioning 24   
D.2 Spatial weights and shared channel operators 25   
D.3 Soft locality of Attention Aggregator 25   
D.4 General predictor parameterizations 27   
D.5 From a single layer to a residual stack 29   
E Experimental details 29   
E.1 Experiment 1-1: Exact denoising context under synthetic GMRF priors 30   
E.2 Experiment 1-2: Denoising context in DiTs trained on real-image datasets 32   
E.3 Experiment 2: Replacing MLPs with shared-memory predictors . 36   
E.4 Experiment 3: Component compatibility in cross-dataset adaptation 39

## A Related Literature

Representations and compositional reuse. Difusion models learn internal features useful for semantic segmentation and discriminative probing (Baranchuk et al., 2022; Xiang et al., 2023), and semantic and geometric correspondence can emerge without correspondence supervision (Tang et al., 2023). Chen et al. (2025) connect this ability to classical denoising autoencoders by simplifying difusion models and studying which components support self-supervised representation learning. Han et al. (2025a) provide theoretical and empirical evidence that denoising encourages representations covering multiple data features, including features that classifiers may neglect. REPA connects representation quality to generative training by aligning hidden states with pretrained visual features in difusion and flow-based transformers (Yu et al., 2025). Singh et al. (2026) further identify the spatial structure of the alignment target as a strong predictor of generation performance.

Compositional generation provides evidence of reuse: Okawa et al. (2023) demonstrate unseen combinations in a synthetic task, and Favero et al. (2025) analyze the learning of composition rules in a hierarchical grammar, where longer-range dependencies require more data. The extent of this ability depends on the task; Han et al. (2025b) find that difusion models can learn coarse relationships while failing finer dependencies, including reflection and lighting consistency. These findings motivate the introduction’s representation-and-reuse framing. In our denoising account, a reuse rule has a specific meaning: it determines which context patches supply evidence for predicting a clean patch from a new noisy image.

Memorization and generalization. Under a finite empirical prior, the squared-error denoising objective is minimized by a weighted average of training images, with weights determined by whole-image comparisons as in (2). With the corresponding terminal initialization, exact reverse dynamics recover the empirical distribution (Pidstrigach, 2022; Yoon et al., 2023; Gu et al., 2025; Biroli et al., 2024; Kamb and Ganguli, 2025). Several explanations address why learned denoisers nevertheless generalize. Training dynamics can favor generalizing solutions before memorization, with larger datasets delaying memorization (Bonnaire et al., 2025); large learning rates can also steer training away from the memorizing solution (Wu et al., 2025). Other accounts characterize the learned estimator: Kadkhodaie et al. (2024) connect generalization to geometry-adaptive harmonic representations, while Zhang et al. (2026) prove a distinction between sample storage and balanced representations of local statistics for a two-layer ReLU denoising autoencoder, and investigate analogous structure in larger models. Our framework concerns how DiT components implement denoising computations compatible with such statistical structure.

Local denoising and its statistical target. Local empirical denoisers explain novel generation through patch recombination. Niedoba et al. (2025) use cropped neighborhoods to approximate trained denoisers; Kamb and Ganguli (2025) derive estimators from locality and translation equivariance; and Lukoianov et al. (2025) choose context from the sensitivity of the optimal linear denoiser, emphasizing the role of data statistics. These constructions motivate the patchwise linear denoiser (PLD) in Section 2. In (3), a softmax comparison through $S _ { m , t }$ determines how stored training patches $x _ { m } ^ { ( i ) }$ contribute to a prediction. Setting $S _ { m , t } = I$ and $\Gamma _ { m , t } = \sigma _ { t } ^ { - 2 } I$ recovers whole-image retrieval; local contexts allow the weights to difer across patches and hence permit recombination.

Our statistical contribution identifies a population target for this construction. Theorem 1 bounds the discrepancy between population local and global denoisers under a GMRF prior. Theorem 3 identifies when suficient aggregation lets a PLD recover the empirical local conditional mean exactly; this identity alone is not a finite-sample bound on population risk. Together, the results motivate learning both the context aggregator and the patch predictor. The stored patches and context comparisons in the PLD provide an explicit example of content and reuse; Section 4 develops their learned counterparts in a DiT.

Architectural accounts of denoising. Mei (2025) show how U-Net encoder–decoder paths and skip connections can implement belief-propagation denoising in hierarchical graphical models, providing a direct precedent for interpreting architecture through a statistical denoising algorithm. For transformers, Rosu et al. (2025) show that attention can implement exact score-based denoising; their image experiments use whole-image tokens, although an anisotropic extension permits patchwise pixel updates. An et al. (2025) associate DiT generalization with local attention in early layers and show benefits from restricting attention windows with limited training data. Gan et al. (2025) link concentrated large feature activations to AdaLN and use channel modulation to improve DiT features for correspondence. These works explain architectural or conditioning mechanisms relevant to denoising and representation learning. Our account connects attention, MLPs, and AdaLN through context aggregation followed by patch prediction, allowing context to vary with the input and noise level, including the non-contiguous patterns in Section 5.1.

MLPs as learned memory. Language-model MLPs have been interpreted as key–value memories that store and retrieve learned associations (Petroni et al., 2019; Geva et al., 2021, 2022); Sukhbaatar et al. (2019) express them as attention over persistent memory vectors. This interpretation has motivated locating and editing facts through MLP weights (Dai et al., 2022; Meng et al., 2022, 2023), with related analyses and interventions in multimodal models (Schwettmann et al., 2023; Pan et al., 2024a; Basu et al., 2024a; Pan et al., 2024b). At the level of the generative process, Hoover et al. (2023) survey connections between difusion dynamics and retrieval in energy-based associative memories. Our DiT account assigns a memory role specifically to the MLP’s patch prediction from aggregated context (Section 4.2). Replacing layerwise MLPs with explicit memory shared across layers tests this interpretation at the level of the predictor component (Section 5.2). The representation may be distributed across learned memory entries; the interpretation does not require individual entries to correspond to named concepts.

Localization, editing, and component interventions. Several analyses of text-to-image difusion localize visual attributes in text-encoder self-attention or U-Net cross-attention (Basu et al., 2024c,b), including neurons associated with concepts or memorized training images (Liu et al., 2023; Hintersdorf et al., 2024). For DiTs, Zarei et al. (2025) identify blocks associated with specific concepts and use localized updates for personalization and unlearning. Their component ablations also intervene on self-attention, normalization, and MLP outputs, with cross-attention giving the strongest localization signal in the tested PixArt setting. Thus, component intervention already has direct precedents in DiTs.

Related editing methods illustrate the practical value of these distinctions. Custom Difusion adapts selected parameters for concept customization (Kumari et al., 2023); Plug-and-Play Difusion Features injects spatial features and self-attention to preserve layout during translation (Tumanyan et al., 2023); and Shin et al. (2025) analyze text–image attention blocks to design editing interventions for multimodal DiTs. Turri et al. (2026) further show that transferring high-magnitude hidden-state channels between prompt-conditioned trajectories can steer image semantics in DiT and related flow-based models. These activation interventions complement our parameter transplants, which ask which QK, VO, and MLP computations remain compatible across image domains, guided by the aggregator–predictor account (Section 5.3). The transplants probe jointly adapted components with AdaLN fixed, so their outcomes measure reuse and coadaptation under that intervention protocol (Appendix E.4).

Gaussian and Markov image priors. The GMRF prior in Theorem 1 combines joint Gaussian statistics with conditional independence between non-neighboring patches given their graph neighbors. Markov image models developed from spatial statistics (Besag, 1974) into methods for restoration, texture modeling, and segmentation (Geman and Geman, 1984; Li, 2009), with higher-order models extending interactions to larger neighborhoods and patches (Roth and Black, 2005; Weiss and Freeman, 2007; Chen et al., 2014). Gaussian MRFs (Rue and Held, 2005) have been used for texture modeling (Chellappa and Chatterjee, 1985), Bayesian restoration (Shouno, 2015; Yasuda et al., 2018; Monma et al., 2022), and learned conditional denoising (Tappen et al., 2007). The Gaussian optimal denoiser is the Wiener filter (Wiener, 1949; Andrews and Hunt, 1977); its tractability also supports exact analyses of difusion sampling (Pierret and Galerne, 2025) and approximations to learned scores at high noise (Wang and Vastola, 2024). We use this tractable prior with patches as variables on an arbitrary graph to establish a locality guarantee, and examine the corresponding qualitative patterns on real images separately.

## B Proofs

## B.1 Notation

For a positive integer M, let $[ M ] : = \{ 1 , \dots , M \}$ . Vectors in $\mathbb { R } ^ { M d _ { p } }$ are partitioned into M blocks of dimension $d _ { p } ; x _ { m }$ denotes block m of a vector x, and $A _ { m l }$ denotes the $( m , l )$ block of a conformably partitioned matrix A. The identity matrix is denoted by I, with its dimension determined by context, and $A ^ { \top }$ denotes the transpose of A.

For a vector $x , \ \Vert x \Vert _ { 2 }$ is its Euclidean norm. For a matrix A, $\begin{array} { r } { \| A \| _ { 2 } : = \operatorname* { s u p } _ { \| x \| _ { 2 } = 1 } \| A x \| _ { 2 } } \end{array}$ is the induced operator norm and $\| A \| _ { F }$ is the Frobenius norm. We write spec(A) for the spectrum of $A , \lambda _ { \mathrm { m i n } } ( A )$ and $\lambda _ { \mathrm { m a x } } ( A )$ for the smallest and largest eigenvalues of a real symmetric matrix, and $\operatorname { t r } ( A )$ for its trace. For real symmetric matrices, $A \preceq B$ means that $B - A$ is positive semidefinite.

For the graph $G _ { 0 } = ( [ M ] , E _ { 0 } ) , d _ { G _ { 0 } } ( m , l )$ is the length of a shortest path between vertices m and $l ,$ and $B _ { r } ( m ) : = \left\{ l \in [ M ] : d _ { G _ { 0 } } ( m , l ) \leq r \right\}$ is the closed graph ball of radius r centered at $m$ . For a block vector $x ,$ $x _ { B _ { r } ( m ) }$ collects the blocks indexed by $B _ { r } ( m )$ . Finally, E denotes expectation and $\mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ denotes a Gaussian distribution with mean $\mu$ and covariance $\Sigma .$

For the conditional-mean decompositions, work at fixed time t on a probability space $( \Omega , \mathcal { F } , \mathbb { P } )$ carrying $X _ { 0 }$ and the independent Gaussian noise in the corruption model, and assume $\mathbb { E } \| X _ { 0 } \| _ { 2 } ^ { 2 } < \infty$ . All Euclidean spaces are equipped with their Borel σ-fields, and measurability of maps between them is understood in this sense. For a random vector $Z$ , write sf $( Z ) \subseteq { \mathcal { F } }$ for the σ-field it generates, and let $P _ { Y _ { t } }$ denote the law of $Y _ { t }$ Inclusions between σ-fields are understood modulo P-null sets.

## B.2 Background facts

## B.2.1 Chebyshev polynomials and the reciprocal expansion

The Chebyshev polynomials are a family of orthogonal polynomials on $[ - 1 , 1 ]$ that are useful for constructing near-minimax polynomial approximations. We use the Chebyshev polynomials of the first kind, defined by the recurrence $T _ { 0 } ( s ) = 1 , T _ { 1 } ( s ) = s .$ and $T _ { r + 1 } ( s ) = 2 s T _ { r } ( s ) - T _ { r - 1 } ( s )$ for $r \geq 1$ . Equivalently, $T _ { r } ( \cos \theta ) = \cos ( r \theta )$ for $\theta \in [ 0 , \pi ]$ . T has degree r and $| T _ { r } ( s ) | \leq 1$ for $s \in [ - 1 , 1 ]$ . For general background, see Wikipedia contributors (2026). The ordinary generating function for $T _ { r }$ is given by (DLMF, 2026, Eq. 18.12.8)

$$
\sum _ { r = 0 } ^ { \infty } T _ { r } ( s ) z ^ { r } = { \frac { 1 - s z } { 1 - 2 s z + z ^ { 2 } } } , \qquad | z | < 1 , \quad s \in [ - 1 , 1 ] .\tag{12}
$$

We now introduce a useful lemma on the Chebyshev approximation of the reciprocal function with exponentially decaying error in the degree r.

Lemma 4. Let $0 < h _ { - } < h _ { + }$ and $r \geq 0$ . There exists a polynomial $\psi _ { r } ( x )$ of degree at most r such that

$$
\operatorname* { s u p } _ { x \in [ h _ { - } , h _ { + } ] } \left. \frac { 1 } { x } - \psi _ { r } ( x ) \right. = \frac { 2 } { \sqrt { h _ { - } h _ { + } } ( 1 - \rho ) } \rho ^ { r + 1 } ,\tag{13}
$$

where $\rho = ( \sqrt { h _ { + } / h _ { - } } - 1 ) / ( \sqrt { h _ { + } / h _ { - } } + 1 )$ is the Chebyshev convergence factor.

Proof. Let $s ( x ) = ( 2 x - h _ { + } - h _ { - } ) / ( h _ { + } - h _ { - } )$ be the linear transformation that maps $[ h _ { - } , h _ { + } ] \mathrm { t o } [ - 1 , 1 ]$ . Since $T _ { 0 } ( s ) = 1$ and the series converges absolutely and uniformly over $s \in [ - 1 , 1 ]$ , from (12), we have

$$
1 + 2 \sum _ { r = 1 } ^ { \infty } z ^ { r } T _ { r } ( s ) = { \frac { 1 - z ^ { 2 } } { 1 - 2 s z + z ^ { 2 } } } , \qquad | z | < 1 , \quad s \in [ - 1 , 1 ] .
$$

Substituting $z = - \rho$ and $s = s ( x )$ and simplifying gives

$$
\frac { 1 } { x } = \frac { 1 } { \sqrt { h _ { - } h _ { + } } } \left( 1 + 2 \sum _ { r = 1 } ^ { \infty } ( - \rho ) ^ { r } T _ { r } ( s ( x ) ) \right) , \qquad x \in [ h _ { - } , h _ { + } ] .
$$

Now, define the degree-r truncated Chebyshev polynomial as

$$
\psi _ { r } ( x ) : = \frac { 1 } { \sqrt { h _ { - } h _ { + } } } \left( 1 + 2 \sum _ { k = 1 } ^ { r } ( - \rho ) ^ { k } T _ { k } ( s ( x ) ) \right) .\tag{14}
$$

Then, since $| T _ { k } ( s ( x ) ) | \le 1$ , we have, for $x \in [ h _ { - } , h _ { + } ]$

$$
\left| \frac { 1 } { x } - \psi _ { r } ( x ) \right| = \frac { 2 } { \sqrt { h _ { - } h _ { + } } } \left| \sum _ { k = r + 1 } ^ { \infty } ( - \rho ) ^ { k } T _ { k } ( s ( x ) ) \right| \leq \frac { 2 } { \sqrt { h _ { - } h _ { + } } } \sum _ { k = r + 1 } ^ { \infty } \rho ^ { k } = \frac { 2 } { \sqrt { h _ { - } h _ { + } } ( 1 - \rho ) } \rho ^ { r + 1 } .
$$

At $x = h _ { - }$ , we have $s ( h _ { - } ) = - 1$ and $T _ { k } ( - 1 ) = ( - 1 ) ^ { k }$ , so $( - \rho ) ^ { k } T _ { k } ( s ( h _ { - } ) ) = \rho ^ { k }$ for all $k \geq 1$ , and hence the upper bound is tight. □

## C Main proofs

## C.1 Full statement and proof of Theorem 1

Theorem 1 (Exponential locality of GMRF; full statement). Suppose $X _ { 0 } \sim \mathcal { N } ( 0 , Q ^ { - 1 } )$ , where $Q \in$ $\mathbb { R } ^ { M d _ { p } \times M d _ { p } }$ is symmetric positive definite and its of-diagonal patch blocks vanish outside the edges of $G _ { 0 } = ( [ M ] , E _ { 0 } )$ . Let $X _ { t } = a _ { t } X _ { 0 } + b _ { t } \varepsilon$ with $a _ { t } , b _ { t } > 0$ and independent $\varepsilon \sim \mathcal { N } ( 0 , I _ { M d _ { p } } )$ , and set $Y _ { t } = X _ { t } / a _ { t }$ and $\sigma _ { t } = b _ { t } / a _ { t }$ . For $B _ { r } ( m ) = \{ l : d _ { G _ { 0 } } ( m , l ) \leq r \}$ , define

$$
D _ { m , t } ^ { * } ( y ) = \mathbb { E } [ X _ { 0 , m } \mid Y _ { t } = y , t ] , \qquad D _ { m , t } ^ { * , r } ( y ) = \mathbb { E } [ X _ { 0 , m } \mid Y _ { t , B _ { r } ( m ) } = y _ { B _ { r } ( m ) } , t ] .
$$

Let $q _ { - } : = \lambda _ { \operatorname* { m i n } } ( Q )$ and $q _ { + } : = \lambda _ { \operatorname* { m a x } } ( Q )$ satisfy $0 < q _ { - } < q _ { + } < \infty$ . Then for every patch m and every integer radius $r \geq 0$

$$
\mathbb { E } \left\| D _ { m , t } ^ { * } ( Y _ { t } ) - D _ { m , t } ^ { * , r } ( Y _ { t } ) \right\| _ { 2 } ^ { 2 } \leq \frac { d _ { p } } { q _ { - } } \left[ \frac { \left( 1 - 1 / \sqrt { \chi _ { t } } \right) ^ { 2 } } { 1 + \sigma _ { t } ^ { 2 } q _ { - } } \right] \rho _ { t } ^ { 2 r } ,\tag{15}
$$

where $\chi _ { t } = ( 1 + \sigma _ { t } ^ { 2 } q _ { + } ) / ( 1 + \sigma _ { t } ^ { 2 } q _ { - } )$ is the condition number of $I + \sigma _ { t } ^ { 2 } Q$ and $\rho _ { t } = ( \sqrt { \chi _ { t } } - 1 ) / ( \sqrt { \chi _ { t } } + 1 )$ is the Chebyshev convergence factor. In the notation of Assump. 1, the bound is $C _ { t } \exp ( - c _ { t } r )$ with

$$
C _ { t } = \frac { d _ { p } ( 1 - 1 / \sqrt { \chi _ { t } } ) ^ { 2 } } { q _ { - } ( 1 + \sigma _ { t } ^ { 2 } q _ { - } ) } , \qquad c _ { t } = 2 \log ( 1 / \rho _ { t } ) > 0 .
$$

Proof. Since $Y _ { t } = X _ { 0 } + \sigma _ { t } \varepsilon$ , Bayes’ rule gives posterior precision $Q + \sigma _ { t } ^ { - 2 } I = \sigma _ { t } ^ { - 2 } ( I + \sigma _ { t } ^ { 2 } Q )$ and posterior mean

$$
D _ { m , t } ^ { * } ( y ) = \bigl [ ( I + \sigma _ { t } ^ { 2 } Q ) ^ { - 1 } y \bigr ] _ { m } .
$$

Let $H _ { t } : = I + \sigma _ { t } ^ { 2 } Q$ . The matrix $H _ { t }$ is real symmetric positive definite, and

$$
\operatorname { s p e c } ( H _ { t } ) \subseteq [ 1 + \sigma _ { t } ^ { 2 } q _ { - } , 1 + \sigma _ { t } ^ { 2 } q _ { + } ] .
$$

Write its spectral decomposition as $H _ { t } = U \Lambda _ { t } U ^ { \top }$ , where $U$ is orthogonal and $\Lambda _ { t }$ is diagonal. We first approximate the scalar inverse function with the degree-r polynomial in (14), and then apply that polynomial to $H _ { t }$

Take $h _ { - , t } : = 1 + \sigma _ { t } ^ { 2 } q _ { - }$ and $h _ { + , t } : = 1 + \sigma _ { t } ^ { 2 } q _ { + }$ , so that $\chi _ { t } = h _ { + , t } / h _ { - , t }$ . Then by the Chebyshev approximation in (13), the degree-r Chebyshev polynomial $\psi _ { r , t } ( x )$ as given in (14) (with $h _ { - }$ and $h _ { + }$ replaced by $h _ { - , t }$ and $h _ { + , t } )$ satisfies

$$
\begin{array} { r l } & { \| H _ { t } ^ { - 1 } - \psi _ { r , t } ( H _ { t } ) \| _ { 2 } = \| U \left( \Lambda _ { t } ^ { - 1 } - \psi _ { r , t } ( \Lambda _ { t } ) \right) U ^ { \top } \| _ { 2 } } \\ & { \qquad = \underset { \lambda \in \mathrm { s p e c } ( H _ { t } ) } { \operatorname* { m a x } } \left| \frac { 1 } { \lambda } - \psi _ { r , t } ( \lambda ) \right| } \\ & { \qquad \le \frac { 2 \rho _ { t } ^ { r + 1 } } { \sqrt { h _ { - , t } h _ { + , t } } ( 1 - \rho _ { t } ) } . } \end{array}
$$

On the other hand, $H _ { t }$ has the same of-diagonal block sparsity pattern as $Q , { \mathrm { s o } } .$ for every integer $k \geq 0$ $[ H _ { t } ^ { k } ] _ { m l } = 0$ whenever $d _ { G _ { 0 } } ( m , l ) > k$ . Indeed, each nonzero term in the blockwise expansion of $\big [ H _ { t } ^ { k } \big ] _ { m l }$ corresponds to a length-k walk from m to $l ,$ with diagonal blocks allowing the walk to remain at its current vertex. Because $\psi _ { r , t }$ has degree at most $r ,$ the predictor

$$
\widetilde { D } _ { m , t } ^ { r } ( y ) : = [ \psi _ { r , t } ( H _ { t } ) y ] _ { m }
$$

depends only on $y _ { B _ { r } ( m ) }$ and is therefore measurable with respect to $Y _ { t , B _ { r } ( m ) }$ . Graph connectivity determines which patches lie within each radius, while the conditioning of $I + \sigma _ { t } ^ { 2 } Q$ controls the decay rate: worse conditioning slows the certified decay and can require broader context

By the tower property, $D _ { m , t } ^ { \ast , r } ( Y _ { t } ) = \operatorname { \mathbb { E } } [ D _ { m , t } ^ { \ast } ( Y _ { t } ) \mid Y _ { t , B _ { r } ( m ) } , t ]$ . The projection property of conditional expectation therefore implies

$$
\begin{array} { r } { \mathbb { E } \| D _ { m , t } ^ { * } ( Y _ { t } ) - D _ { m , t } ^ { * , r } ( Y _ { t } ) \| _ { 2 } ^ { 2 } \leq \mathbb { E } \| D _ { m , t } ^ { * } ( Y _ { t } ) - \widetilde { D } _ { m , t } ^ { r } ( Y _ { t } ) \| _ { 2 } ^ { 2 } . } \end{array}
$$

Construct $P _ { m } \in \mathbb { R } ^ { d _ { p } \times M d _ { p } }$ so that it selects patch m of a noisy image $y \in \mathbb { R } ^ { M d _ { p } }$ , i.e., $P _ { m } y = y _ { m }$ . Then $D _ { m , t } ^ { \ast } ( Y _ { t } ) - \widetilde { D } _ { m , t } ^ { r } ( Y _ { t } ) = P _ { m } \left( H _ { t } ^ { - 1 } - \psi _ { r , t } ( H _ { t } ) \right) Y _ { t }$ . Since $Y _ { t }$ is zero-mean Gaussian with covariance $Q ^ { - 1 } + \sigma _ { t } ^ { 2 } I$ we have

$$
\begin{array} { r l } & { \mathbb { E } \| D _ { m , t } ^ { * } ( Y _ { t } ) - \widetilde { D } _ { m , t } ^ { r } ( Y _ { t } ) \| _ { 2 } ^ { 2 } = \operatorname { t r } \bigg ( P _ { m } \left( H _ { t } ^ { - 1 } - \psi _ { r , t } ( H _ { t } ) \right) ( Q ^ { - 1 } + \sigma _ { t } ^ { 2 } I ) \left( H _ { t } ^ { - 1 } - \psi _ { r , t } ( H _ { t } ) \right) P _ { m } ^ { \top } \bigg ) } \\ & { \qquad = \left\| P _ { m } \left( H _ { t } ^ { - 1 } - \psi _ { r , t } ( H _ { t } ) \right) ( Q ^ { - 1 } + \sigma _ { t } ^ { 2 } I ) ^ { 1 / 2 } \right\| _ { F } ^ { 2 } } \\ & { \qquad \leq \| Q ^ { - 1 } + \sigma _ { t } ^ { 2 } I \| _ { 2 } \left\| P _ { m } \left( H _ { t } ^ { - 1 } - \psi _ { r , t } ( H _ { t } ) \right) \right\| _ { F } ^ { 2 } . } \end{array}
$$

Since $P _ { m } ( H _ { t } ^ { - 1 } - \psi _ { r , t } ( H _ { t } ) ) \in \mathbb { R } ^ { d _ { p } \times M d _ { p } }$ has $d _ { p }$ rows, the Frobenius–operator norm inequality and the Chebyshev approximation bound give

$$
\begin{array} { r l } {  { \big \| P _ { m } ( H _ { t } ^ { - 1 } - \psi _ { r , t } ( H _ { t } ) ) \big \| _ { F } ^ { 2 } \leq d _ { p } \| P _ { m } ( H _ { t } ^ { - 1 } - \psi _ { r , t } ( H _ { t } ) ) \| _ { 2 } ^ { 2 } } } \\ & { \leq d _ { p } \| H _ { t } ^ { - 1 } - \psi _ { r , t } ( H _ { t } ) \| _ { 2 } ^ { 2 } } \\ & { \leq d _ { p } \frac { 4 \rho _ { t } ^ { 2 r + 2 } } { h _ { - , t } h _ { + , t } ( 1 - \rho _ { t } ) ^ { 2 } } . } \end{array}
$$

Also, since $Q ^ { - 1 } + \sigma _ { t } ^ { 2 } I$ is real symmetric positive definite,

$$
\| Q ^ { - 1 } + \sigma _ { t } ^ { 2 } I \| _ { 2 } = \lambda _ { \operatorname* { m a x } } ( Q ^ { - 1 } + \sigma _ { t } ^ { 2 } I ) = q _ { - } ^ { - 1 } + \sigma _ { t } ^ { 2 } .
$$

Combining the above bounds, we have

$$
\mathbb { E } \| D _ { m , t } ^ { * } ( Y _ { t } ) - \widetilde { D } _ { m , t } ^ { r } ( Y _ { t } ) \| _ { 2 } ^ { 2 } \leq d _ { p } ( q _ { - } ^ { - 1 } + \sigma _ { t } ^ { 2 } ) \frac { 4 \rho _ { t } ^ { 2 r + 2 } } { h _ { - , t } h _ { + , t } ( 1 - \rho _ { t } ) ^ { 2 } } .
$$

Finally, the identities

$$
q _ { - } ^ { - 1 } + \sigma _ { t } ^ { 2 } = \frac { h _ { - , t } } { q _ { - } } , \qquad \frac { 2 \rho _ { t } } { \sqrt { h _ { - , t } h _ { + , t } } ( 1 - \rho _ { t } ) } = \frac { 1 - 1 / \sqrt { \chi _ { t } } } { h _ { - , t } }
$$

reduce the preceding display to (15).

## C.2 Full statement and proof of Corollary 2

Corollary 2 (Bounded-range monotonicity of the locality radius; full statement). Under the hypotheses of the full Thm. 1 above, with $0 < q _ { - } < q _ { + }$ , define $\sigma _ { \mathrm { a b s } } ^ { 2 } : = ( q _ { - } ^ { 1 / 3 } + q _ { + } ^ { 1 / 3 } ) / ( q _ { - } ^ { 2 / 3 } q _ { + } ^ { 2 / 3 } )$ . For $\epsilon > 0$ and $0 < \sigma _ { t } \le \sigma _ { \mathrm { a b s } }$ , the smallest radius $r _ { \epsilon } ( \sigma _ { t } )$ such that the upper bound in (15) is at most ϵ is the least nonnegative integer satisfying

$$
r _ { \epsilon } ( \sigma _ { t } ) \geq \frac { 1 } { 2 } \log \left( \frac { d _ { p } \left( 1 - 1 / \sqrt { \chi _ { t } } \right) ^ { 2 } } { \epsilon q _ { - } ( 1 + \sigma _ { t } ^ { 2 } q _ { - } ) } \right) \bigg / \log \left( \frac { 1 } { \rho _ { t } } \right)\tag{16}
$$

The resulting $r _ { \epsilon } ( \sigma _ { t } )$ is nondecreasing in $\sigma _ { t }$ on the same range. At $\sigma _ { t } = 0$ , radius zero is exact.

Proof. We show that the upper bound in (15) is nondecreasing in $\sigma _ { t }$ for $0 < \sigma _ { t } \le \sigma _ { \mathrm { a b s } }$ . First, for $\sigma _ { t } > 0$

$$
\frac { 1 } { \chi _ { t } } \frac { d \chi _ { t } } { d \sigma _ { t } } = \frac { 2 \sigma _ { t } ( q _ { + } - q _ { - } ) } { ( 1 + \sigma _ { t } ^ { 2 } q _ { + } ) ( 1 + \sigma _ { t } ^ { 2 } q _ { - } ) } \geq 0 ,
$$

and hence $d \chi _ { t } / d \sigma _ { t } \geq 0$ . Next,

$$
\begin{array} { r l r } {  { \frac { d } { d \sigma _ { t } } \log [ \frac { ( 1 - 1 / \sqrt { \chi _ { t } } ) ^ { 2 } } { ( 1 + \sigma _ { t } ^ { 2 } q _ { - } ) } ] = \frac { 1 } { \sqrt { \chi _ { t } } - 1 } ( \frac { 1 } { \chi _ { t } } \frac { d \chi _ { t } } { d \sigma _ { t } } ) - \frac { 2 \sigma _ { t } q _ { - } } { 1 + \sigma _ { t } ^ { 2 } q _ { - } } } } \\ & { } & { = \frac { 2 \sigma _ { t } } { 1 + \sigma _ { t } ^ { 2 } q _ { - } } [ \frac { q _ { + } - q _ { - } } { ( 1 + \sigma _ { t } ^ { 2 } q _ { + } ) ( \sqrt { \chi _ { t } } - 1 ) } - q _ { - } ] } \end{array}
$$

is nonnegative when $q _ { + } - q _ { - } \ge q _ { - } ( 1 + \sigma _ { t } ^ { 2 } q _ { + } ) ( \sqrt { \chi _ { t } } - 1 )$ . Using

$$
1 + \sigma _ { t } ^ { 2 } q _ { + } = \chi _ { t } \frac { q _ { + } - q _ { - } } { q _ { + } - q _ { - } \chi _ { t } } , \qquad q _ { + } - q _ { - } \chi _ { t } = \frac { q _ { + } - q _ { - } } { 1 + \sigma _ { t } ^ { 2 } q _ { - } } ,
$$

this condition is equivalent to $q _ { + } \ge q _ { - } \chi _ { t } ^ { 3 / 2 }$ , which simplifes to $\sigma _ { t } ^ { 2 } \le \sigma _ { \mathrm { a b s } } ^ { 2 }$

For the rest of the term in the upper bound, it is straightforward to see that $d \rho _ { t } / d \chi _ { t } \geq 0$ , and since $d \chi _ { t } / d \sigma _ { t } \geq 0$ , we have $d \rho _ { t } / d \sigma _ { t } \geq 0$ . Consequently, if $0 < \sigma _ { 1 } \leq \sigma _ { 2 } \leq \sigma _ { \mathrm { a b s } } .$ , every radius whose upper bound is at most ϵ at $\sigma _ { 2 }$ also has upper bound at most ϵ at $\sigma _ { 1 }$ . Henc $r _ { \epsilon } ( \sigma _ { 2 } ) \geq r _ { \epsilon } ( \sigma _ { 1 } )$ □

## C.3 Proof of Theorem 3

Proof. Fix m and t with $\sigma _ { t } > 0 .$ , and use the stacked-vector convention throughout. Under the empirical prior, let J be uniform on [n] and set $X _ { 0 } ~ = ~ x ^ { ( J ) }$ . Since $Y _ { t } = X _ { 0 } + \sigma _ { t } \varepsilon$ , the linear aggregate satisfies $\bar { S } _ { m , t } ^ { \mathrm { l i n } } ( Y _ { t } ) \mid J = i \sim \mathcal { N } \big ( S _ { m , t } x ^ { ( i ) } , \sigma _ { t } ^ { 2 } S _ { m , t } S _ { m , t } ^ { \top } \big )$ . The covariance is positive definite because $\sigma _ { t } > 0$ and $S _ { m , t }$ has full row rank. With the prescribed precision $\Gamma _ { m , t } = ( \sigma _ { t } ^ { 2 } S _ { m , t } S _ { m , t } ^ { \top } ) ^ { - 1 }$ , the Gaussian log-likelihood is $- \frac { 1 } { 2 } \| s - S _ { m , t } x ^ { ( i ) } \| _ { \Gamma _ { m , : } } ^ { 2 }$ up to a term independent of i. The uniform prior and common Gaussian normalizing factor cancel in Bayes’ rule, giving

$$
\mathrm { P r } _ { \hat { p } _ { 0 } } \left[ J = i \mid { \cal S } _ { m , t } ^ { \mathrm { l i n } } ( Y _ { t } ) = s , t \right] = \mathrm { s m } _ { i } \left\{ - \frac { 1 } { 2 } \left. s - { \cal S } _ { m , t } ^ { \mathrm { l i n } } ( x ^ { ( j ) } ) \right. _ { \Gamma _ { m , t } } ^ { 2 } \right\} _ { j \in [ n ] } .
$$

Consequently,

$$
\mathbb { E } _ { \hat { p } _ { 0 } } \left[ X _ { 0 , m } ~ | ~ \mathcal { S } _ { m , t } ^ { \mathrm { l i n } } ( Y _ { t } ) = s , t \right] = \sum _ { i = 1 } ^ { n } \operatorname* { P r } _ { \hat { p } _ { 0 } } \left[ J = i ~ | ~ \mathcal { S } _ { m , t } ^ { \mathrm { l i n } } ( Y _ { t } ) = s , t \right] x _ { m } ^ { ( i ) } = g _ { m , t } ^ { \mathrm { r b f } } ( s ) .
$$

Because the aggregate is a measurable function of $Y _ { t , { B _ { m , t } } }$ , the tower property gives the first equality below. The assumed sigma-field inclusion makes the inner conditional mean measurable with respect to the aggregate, giving the second equality:

$$
\begin{array} { r l } & { \mathbb { E } _ { \hat { p } _ { 0 } } \left[ X _ { 0 , m } \mid \mathcal { S } _ { m , t } ^ { \mathrm { l i n } } ( Y _ { t } ) , t \right] = \mathbb { E } _ { \hat { p } _ { 0 } } \left[ \mathbb { E } _ { \hat { p } _ { 0 } } \left[ X _ { 0 , m } \mid Y _ { t } , { \mathcal { B } } _ { m , t } , t \right] \mid { \mathcal { S } } _ { m , t } ^ { \mathrm { l i n } } ( Y _ { t } ) , t \right] } \\ & { \qquad = \mathbb { E } _ { \hat { p } _ { 0 } } \left[ X _ { 0 , m } \mid Y _ { t , { \mathcal { B } } _ { m , t } } , t \right] . } \end{array}
$$

Together with $\hat { D } _ { m , t } ^ { \mathrm { p l d } } = g _ { m , t } ^ { \mathrm { r b f } } \circ S _ { m , t } ^ { \mathrm { l i n } }$ , this proves the theorem.

To make the inclusion condition concrete, restrict $S _ { m , t }$ to its context columns, denoted by $[ S _ { m , t } ] _ { B _ { m , t } }$ . Locality implies $S _ { m , t } y = [ S _ { m , t } ] B _ { m , t } y _ { B _ { m , t } }$ . Bayes’ rule for the full context gives

$$
\begin{array} { r l r } & { } & { \mathrm { P r } _ { \hat { p } _ { 0 } } \left[ J = i \mid Y _ { t , \mathcal { B } _ { m , t } } = y _ { \mathcal { B } _ { m , t } } , t \right] = \mathrm { s m } _ { i } \left\{ - \displaystyle \frac { 1 } { 2 \sigma _ { t } ^ { 2 } } \left\| y _ { \mathcal { B } _ { m , t } } - x _ { \mathcal { B } _ { m , t } } ^ { ( j ) } \right\| _ { 2 } ^ { 2 } \right\} _ { j \in [ n ] } } \\ & { } & { = \mathrm { s m } _ { i } \left\{ \frac { 2 y _ { \mathcal { B } _ { m , t } } ^ { \top } \left( x _ { \mathcal { B } _ { m , t } } ^ { ( j ) } - x _ { \mathcal { B } _ { m , t } } ^ { ( 1 ) } \right) - \| x _ { \mathcal { B } _ { m , t } } ^ { ( j ) } \| _ { 2 } ^ { 2 } + \| x _ { \mathcal { B } _ { m , t } } ^ { ( 1 ) } \| _ { 2 } ^ { 2 } } { 2 \sigma _ { t } ^ { 2 } } \right\} _ { j \in [ n ] } . } \end{array}
$$

The second equality subtracts the log-likelihood of sample 1 from every logit, which leaves the softmax unchanged. Thus, the posterior weights depend on $y _ { B _ { m , t } }$ only through its inner products with the trainingcontext diferences. Suppose that

$$
V _ { m , t } : = \mathrm { s p a n } \left\{ x _ { \mathcal { B } _ { m , t } } ^ { ( j ) } - x _ { \mathcal { B } _ { m , t } } ^ { ( 1 ) } : j \in [ n ] \right\} \subseteq \mathrm { r o w } \left( [ S _ { m , t } ] _ { \mathcal { B } _ { m , t } } \right) .\tag{17}
$$

For each $j ,$ there is then a vector $a _ { j } \in \mathbb { R } ^ { d _ { h } }$ such that $x _ { B _ { m , t } } ^ { ( j ) } - x _ { B _ { m , t } } ^ { ( 1 ) } = [ S _ { m , t } ] _ { B _ { m , t } } ^ { \top } a _ { j }$ . Hence $y _ { B _ { m , t } } ^ { \top } ( x _ { B _ { m , t } } ^ { ( j ) } - x _ { B _ { m , t } } ^ { ( 1 ) } ) =$ $a _ { j } ^ { \top } [ S _ { m , t } ] s _ { m , t } y _ { \mathcal { B } _ { m , t } } = a _ { j } ^ { \top } S _ { m , t } y$ . Every posterior weight is therefore a continuous function of $S _ { m , t } y$ , as is their weighted sum $\mathbb { E } _ { \hat { p } _ { 0 } } [ X _ { 0 , m } ^ { \mathrm { ~ ~ } } \mid Y _ { t , { \mathcal { B } } _ { m , t } } = y _ { { \mathcal { B } } _ { m , t } } , t ]$ . This proves the required sigma-field inclusion: the local conditional mean can be recovered from the aggregate even when the full context cannot.

This construction is strictly compressive when $0 < v _ { m , t } : =$ dim $V _ { m , t } < | B _ { m , t } | d _ { p } \large { : }$ choose the $v _ { m , t }$ rows of $[ S _ { m , t } ] _ { B _ { m } , }$ to be an orthonormal basis of $V _ { m , t }$ , so that $d _ { h } = v _ { m , t } < | \boldsymbol { B _ { m , t } } | d _ { p }$ . All clean training contexts have the same projection onto $V _ { m , t } ^ { \perp }$ . Under isotropic Gaussian corruption, the discarded coordinates consequently contribute the same likelihood factor for every sample index, which cancels from the posterior weights. For example, if $| \boldsymbol { B _ { m , t } } | d _ { p } = 2$ and $\boldsymbol { x } _ { B _ { m , t } } ^ { ( j ) } = ( u _ { j } , c )$ with fixed c and varying $u _ { j }$ , then $[ S _ { m , t } ] _ { B _ { m , t } } = ( 1 , 0 )$ retains only the first noisy coordinate and still determines all posterior weights. If $v _ { m , t } = 0$ , all training contexts coincide and the local conditional mean is constant. In general $v _ { m , t } \leq \operatorname* { m i n } ( n - 1 , | \boldsymbol { \mathcal { B } } _ { m , t } | d _ { p } )$ , but $v _ { m , t }$ can equal $| B _ { m , t } | d _ { p } ;$ in that case this row-space condition requires retaining the full context dimension and provides no strict compression. We note that the condition (17) is suficient, not necessary for the theorem: it preserves the entire posterior over sample indices, whereas the theorem requires only preservation of the target-patch conditional mean. □

## D From the aggregator–predictor construction to implementation

This section connects the aggregator–predictor construction in Secs. 4 to the implemented architectures. We first expand a single attention branch applied to the noisy patches $Y _ { t } .$ then explain how aggregation and patch prediction extend to residual stacks, and finally give the full predictor parameterizations. All patches, tokens, embeddings, biases, and aggregates are column vectors, consistent with Appendix B.1; projection matrices act from the left. The width $d _ { h }$ denotes the full token and aggregate dimension, and the number of attention heads H<sup>˜</sup> divides $d _ { h }$

![](images/99d5d30fc7e5797634073a7c54f49f89352215363d105cdadd9484466e75e481.jpg)  
Figure 6 Component map for the aggregator–predictor view. At the branch level, AdaLN-conditioned QK supplies spatial weights, VO transforms the weighted features, and the MLP admits a learned-memory interpretation for tokenwise prediction. The patchwise linear denoiser (PLD) instead uses a prescribed context and retrieves training patches.

## D.1 Patch, position, and time conditioning

The initial token at patch l is $E Y _ { t , l } + e _ { l }$ , where $E \in \mathbb { R } ^ { d _ { h } \times d _ { p } }$ and $\boldsymbol { e } _ { l } \in \mathbb { R } ^ { d _ { h } }$ . A shared patch-projection bias, if present, can be included in each $e _ { l }$ . If the network receives $X _ { t } = a _ { t } Y _ { t }$ or another known scalar preconditioning of $Y _ { t }$ , include that time-dependent scalar before the patch projection; the same derivation applies.

Let $\tau ( t ) \in \mathbb { R } ^ { d _ { h } }$ denote a fixed sinusoidal time embedding. The time-embedding and the following AdaLN parameters can be written as

$$
e _ { t } ^ { \mathrm { t i m e } } = W _ { 2 } ^ { \mathrm { t i m e } } \mathrm { S i L U } \big ( W _ { 1 } ^ { \mathrm { t i m e } } \tau ( t ) + b _ { 1 } ^ { \mathrm { t i m e } } \big ) + b _ { 2 } ^ { \mathrm { t i m e } } ,\tag{18}
$$

$$
\begin{array} { r l } { \left[ \tilde { \beta } _ { t } ^ { \top } \quad \tilde { \gamma } _ { t } ^ { \top } \quad \tilde { \alpha } _ { t } ^ { \top } \quad \beta _ { t } ^ { \top } \quad \gamma _ { t } ^ { \top } \quad \alpha _ { t } ^ { \top } \right] ^ { \top } = W ^ { \mathrm { a d a } } e _ { t } ^ { \mathrm { t i m e } } + b ^ { \mathrm { a d a } } . } \end{array}\tag{19}
$$

Here $W _ { 1 } ^ { \mathrm { t i m e } } \in \mathbb { R } ^ { 4 d _ { h } \times d _ { h } } , W _ { 2 } ^ { \mathrm { t i m e } } \in \mathbb { R } ^ { d _ { h } \times 4 d _ { h } }$ , and their biases have dimensions $4 d _ { h }$ and $d _ { h }$ , respectively. Also $W ^ { \mathrm { a d a } } \in \mathbf { \bar { \mathbb { R } } } ^ { 6 d _ { h } \times d _ { h } }$ and $b ^ { \mathrm { a d a } } \in \mathbb { R } ^ { 6 d _ { h } }$ , so every shift $\beta ,$ scale adjustment $\gamma ,$ , and residual gate α is a column in $\mathbb { R } ^ { d _ { h } }$ . The tilded rows condition spatial attention; the untilded rows condition the predictor branch. Layer indices are suppressed in this single-layer expansion; in a stack, the AdaLN map and both sets of conditioning parameters are layer-specific.

With these notations, the feature entering attention can be written as

$$
\phi _ { t , l } ( Y _ { t } ) : = \Phi ^ { \mathrm { a t t n } } ( E Y _ { t , l } ; e _ { l } , t ) = \left( 1 + \tilde { \gamma } _ { t } \right) \odot \mathrm { L N } ( E Y _ { t , l } + e _ { l } ) + \tilde { \beta } _ { t } .\tag{20}
$$

Layer normalization acts over the $d _ { h }$ channels of this token, with a positive numerical stabilizer and no learned afine parameters; ⊙ denotes elementwise multiplication. Consequently, at fixed t, this feature depends only on patch l and its position. The gate $\tilde { \alpha } _ { t }$ is applied after attention output mixing.

## D.2 Spatial weights and shared channel operators

For head h, the Q, K, and V matrices have shape $( d _ { h } / \tilde { H } ) \times d _ { h }$ , and the output projection matrix $\tilde { W } _ { h } ^ { O }$ has shape $d _ { h } \times ( d _ { h } / \tilde { H } )$ . Including the QK biases $\tilde { b } _ { h } ^ { Q } , \tilde { b } _ { h } ^ { K } \in \mathbb { R } ^ { d _ { h } / \tilde { H } }$ , we define the scalar spatial weights

$$
A _ { m l , t } ^ { ( h ) } ( Y _ { t } ) = \mathrm { s m } _ { l } \left\{ \frac { \langle \tilde { W } _ { h } ^ { Q } \phi _ { t , m } ( Y _ { t } ) + \tilde { b } _ { h } ^ { Q } , \tilde { W } _ { h } ^ { K } \phi _ { t , j } ( Y _ { t } ) + \tilde { b } _ { h } ^ { K } \rangle } { \sqrt { d _ { h } / \tilde { H } } } \right\} _ { j \in [ M ] } .\tag{21}
$$

These weights are positive and sum to one over l. Without QK biases, the numerator is $\phi _ { t , m } ( Y _ { t } ) ^ { \top } ( \tilde { W } _ { h } ^ { Q } ) ^ { \top }$ $\tilde { W } _ { h } ^ { K } \phi _ { t , l } ( Y _ { t } )$ . The matrix $( \tilde { W } _ { h } ^ { Q } ) ^ { \top } \tilde { W } _ { h } ^ { K } \in \mathbb { R } ^ { d _ { h } \times d _ { h } }$ <sup>h</sup> therefore controls compatibility between query and context features.

Rearranging the finite head and patch sums gives

$$
\begin{array} { r l r } & { } & { S _ { m , t } ^ { \mathrm { a t t n } } ( Y _ { t } ) = \mathrm { d i a g } ( \tilde { \alpha } _ { t } ) \displaystyle \sum _ { h = 1 } ^ { \tilde { H } } \tilde { W } _ { h } ^ { O } \left( \sum _ { l = 1 } ^ { M } A _ { m l , t } ^ { ( h ) } ( Y _ { t } ) \tilde { W } _ { h } ^ { V } \phi _ { t , l } ( Y _ { t } ) \right) } \\ & { } & { \qquad = \displaystyle \sum _ { l = 1 } ^ { M } \left[ \displaystyle \sum _ { h = 1 } ^ { \tilde { H } } A _ { m l , t } ^ { ( h ) } ( Y _ { t } ) \mathrm { d i a g } ( \tilde { \alpha } _ { t } ) \tilde { W } _ { h } ^ { O } \tilde { W } _ { h } ^ { V } \right] \phi _ { t , l } ( Y _ { t } ) . } \end{array}\tag{22}
$$

Each bracketed matrix in the second line is $d _ { h } \times d _ { h }$ , and each summand is a column in $\mathbb { R } ^ { d _ { h } }$ . Setting the QK biases to zero recovers (7); with biases, the same decomposition holds using the weights in (21). If included, the value biases $\tilde { b } _ { h } ^ { V } \in \tilde { \mathbb { R } } ^ { d _ { h } / \tilde { H } }$ and output bias $\tilde { b } ^ { O } \in \mathbb { R } ^ { d _ { h } }$ add the column ofset diag $\begin{array} { r } { \because ( \tilde { \alpha } _ { t } ) \bigl ( \sum _ { h } \tilde { W } _ { h } ^ { O } \tilde { b } _ { h } ^ { V } + \tilde { b } ^ { O } \bigr ) } \end{array}$ because each attention row sums to one. This ofset depends on time but not on the noisy image or patch location, so it does not change the spatial selection mechanism.

As explained in Sec. 4.1, we can read of the roles of QK and VO from (22). At fixed t, every $S _ { m l } ^ { \mathrm { a t t n } } ( Y _ { t } ; t )$ lies in the linear subspace spanned by the time-conditioned VO operators, span $\{ \mathrm { d i a g } ( \tilde { \alpha } _ { t } ) \tilde { W } _ { h } ^ { O } \tilde { W } _ { h } ^ { V } : \tilde { h } \in [ \tilde { H } ] \}$ whose dimension is at most min $( \tilde { H } , d _ { h } ^ { 2 } )$ . Thus, each head supplies a patch-shared, time-conditioned channel operator $\mathrm { d i a g } ( \tilde { \alpha } _ { t } ) \tilde { W } _ { h } ^ { O } \tilde { W } _ { h } ^ { V }$ and an independently normalized spatial weighting. Before output mixing, a head forms a convex combination of its projected patch features $\tilde { \tilde { W } } _ { h } ^ { V } \phi _ { t , l } ( Y _ { t } )$ . However, the full branch (22) need not be a convex combination of the patch features, since the channel operators are unconstrained. With the value vectors and channel operators held fixed, the attention matrix $A _ { h }$ determines how the head distributes across the value vectors associated with diferent patches. QK therefore directly controls which patches receive more or less exposure in the constructed context, even though the downstream efect of an individual weight also depends on value magnitudes, channel operators, and residual interactions.

## D.3 Soft locality of Attention Aggregator

Section D.2 explained how, for fixed features, QK controls the relative logits and hence which patches receive appreciable weight. Accordingly, low-scoring patches are suppressed smoothly by exponentiation and normalization, rather than removed by a literal threshold. This provides an input-adaptive analogue of a local selector, but without explicit thresholding unmasked attention remains globally supported and need not solely concentrate on a local context set.

To relate this interpretation to our definition of locality, fix a nonempty context $B _ { m , t }$ containing m and mask every head to that context before softmax. Define $S _ { m , t } ^ { \mathrm { a t t n } , B }$ using the same features, channel operators, and biases, but with weights proportional to $A _ { m l , t } ^ { ( h ) } ( Y _ { t } )$ over $l \in B _ { m , t }$ and zero otherwise. The renormalization cancels the original global softmax denominator. Since each retained logit depends only on patches m and l, this masked branch is exactly $\boldsymbol { B } _ { m , t } \mathrm { - l o c a l }$ . The following bound records the expected squared error of this hard-threshold masking approximation. The bound depends on discarded mass over the observation law, together with the feature and channel-operator magnitudes. The notation ${ \lesssim } _ { t }$ hides a constant depending only on the fixed network parameters and time t, not on the observation or the context.

Theorem 5 (Expected squared error of context masking). Fix t, a patch $m ,$ , the attention parameters, and a nonempty context $B _ { m , t } \subseteq [ M ]$ containing m. Then $S _ { m , t } ^ { \mathrm { a t t n } , B }$ is $B _ { m , t }$ -local, and

$$
\mathbb { E } \left\| S _ { m , t } ^ { \mathrm { a t t n } } ( Y _ { t } ) - S _ { m , t } ^ { \mathrm { a t t n } , \mathcal { B } } ( Y _ { t } ) \right\| _ { 2 } ^ { 2 } \lesssim t \sum _ { h = 1 } ^ { \tilde { H } } \mathbb { E } \left[ \left( \sum _ { l \not \in \mathcal { B } _ { m , t } } A _ { m l , t } ^ { ( h ) } ( Y _ { t } ) \right) ^ { 2 } \right] .\tag{23}
$$

All expectations are over the observation law of Y<sub>t</sub> at the fixed time t.

Proof. Before applying the time-conditioned scale and shift, the layer-normalized feature has norm at most $\sqrt { d _ { h } }$ , so (20) gives

$$
\operatorname* { m a x } _ { l \in [ M ] } \| \phi _ { t , l } ( Y _ { t } ) \| _ { 2 } \leq \sqrt { d _ { h } } \| 1 + \tilde { \gamma } _ { t } \| _ { \infty } + \| \tilde { \beta } _ { t } \| _ { 2 } .
$$

Fix an observation $Y _ { t } .$ . For each head, let $\mu _ { h } ^ { \mathrm { i n } }$ and $\mu _ { h } ^ { \mathrm { o u t } }$ be the attention-weighted averages of $\phi _ { t , l } ( Y _ { t } )$ over the retained and discarded patches, respectively, each normalized by its total attention mass. The retained mass is positive because the context is nonempty and softmax weights are positive. If the discarded set is empty, take $\mu _ { h } ^ { \mathrm { o u t } } = 0$ , in which case the approximation is exact. Note that both averages satisfy the same norm bound as the features. Also, the masked head average is $\mu _ { h } ^ { \mathrm { i n } }$ , since the retained weights are renormalized to sum to one. Thus the diference between the full and masked head averages is

$$
\begin{array} { r l } {  { \sum _ { l = 1 } ^ { M } A _ { m l , t } ^ { ( h ) } ( Y _ { t } ) \phi _ { t , l } ( Y _ { t } ) - \mu _ { h } ^ { \mathrm { i n } } } } \\ & { = ( 1 - \sum _ { l \not \in \mathcal { B } _ { m , t } } A _ { m l , t } ^ { ( h ) } ( Y _ { t } ) ) \mu _ { h } ^ { \mathrm { i n } } + ( \sum _ { l \not \in \mathcal { B } _ { m , t } } A _ { m l , t } ^ { ( h ) } ( Y _ { t } ) ) \mu _ { h } ^ { \mathrm { o u t } } - \mu _ { h } ^ { \mathrm { i n } } } \\ & { = ( \sum _ { l \not \in \mathcal { B } _ { m , t } } A _ { m l , t } ^ { ( h ) } ( Y _ { t } ) ) ( \mu _ { h } ^ { \mathrm { o u t } } - \mu _ { h } ^ { \mathrm { i n } } ) . } \end{array}
$$

The value- and output-bias ofsets are the same in both branches, since both the full and masked attention rows sum to one, so these ofsets cancel. The triangle inequality and the spectral-norm bound therefore give

$$
\begin{array} { r l } {  { \Big \| S _ { m , t } ^ { \mathrm { a t t } } ( Y _ { t } ) - S _ { m , t } ^ { \mathrm { a t t } , \boldsymbol { B } } ( Y _ { t } ) \Big \| _ { 2 } } \qquad } & { } \\ & { \leq \displaystyle \sum _ { h = 1 } ^ { \tilde { H } } \| \mathrm { d i a g } ( \tilde { \alpha } _ { t } ) \tilde { W } _ { h } ^ { O } \tilde { W } _ { h } ^ { V } \| _ { 2 } ( \sum _ { l \not \in B _ { m , t } } A _ { m l , t } ^ { ( h ) } ( Y _ { t } ) ) \| \mu _ { h } ^ { \mathrm { o u t } } - \mu _ { h } ^ { \mathrm { i n } } \| _ { 2 } } \\ & { \leq 2 ( \sqrt { d _ { h } } \| 1 + \tilde { \gamma } _ { t } \| _ { \infty } + \| \tilde { \beta } _ { t } \| _ { 2 } ) \displaystyle \sum _ { h = 1 } ^ { \tilde { H } } \| \mathrm { d i a g } ( \tilde { \alpha } _ { t } ) \tilde { W } _ { h } ^ { O } \tilde { W } _ { h } ^ { V } \| _ { 2 } \sum _ { l \not \in B _ { m , t } } A _ { m l , t } ^ { ( h ) } ( Y _ { t } ) . } \end{array}
$$

Squaring and taking expectations gives

$$
\begin{array} { r l r } {  { \mathbb { E } \| \mathcal { S } _ { m , t } ^ { \mathrm { a t t n } } ( Y _ { t } ) - \mathcal { S } _ { m , t } ^ { \mathrm { a t t n } , \mathcal { S } } ( Y _ { t } ) \| _ { 2 } ^ { 2 } } } \\ & { } & { \leq 4 ( \sqrt { d _ { h } } \| 1 + \widetilde { \gamma } _ { t } \| _ { \infty } + \| \widetilde { \beta } _ { t } \| _ { 2 } ) ^ { 2 } \mathbb { E } [ ( \displaystyle \sum _ { h = 1 } ^ { \widetilde { H } } \| \mathrm { d i a g } ( \widetilde { \alpha } _ { t } ) \widetilde { W } _ { h } ^ { \mathcal { O } } \tilde { W } _ { h } ^ { V } \| _ { 2 } \displaystyle \sum _ { l \not \in \mathcal { B } _ { m , t } } A _ { m l , t } ^ { ( h ) } ( Y _ { t } ) ) ^ { 2 } ] . } \end{array}
$$

Applying Cauchy–Schwarz to the head sum yields (23), with implicit constant

$$
4 \left( \sqrt { d _ { h } } \| 1 + \tilde { \gamma } _ { t } \| _ { \infty } + \| \tilde { \beta } _ { t } \| _ { 2 } \right) ^ { 2 } \sum _ { h = 1 } ^ { \tilde { H } } \big \| \mathrm { d i a g } ( \tilde { \alpha } _ { t } ) \tilde { W } _ { h } ^ { O } \tilde { W } _ { h } ^ { V } \big \| _ { 2 } ^ { 2 } .
$$

Finally, after masking, the softmax denominator contains only retained logits. Each such logit depends only on patches m and $l \in B _ { m , t } ,$ and each retained feature depends only on its own patch. Thus the masked branch depends only on $Y _ { t , B _ { m , t } } ;$ , establishing locality. □

## D.4 General predictor parameterizations

We now write the full formulation of the shared-memory patch predictors at layer $\ell \in [ L ]$ , where L denotes the number of layers, expanding the AdaLN conditioning and restore the afine biases, output gates, separate output projections, and multiple heads, suppressed in Sec. 4.2. We let s denote the input to a patch predictor at layer ℓ. Its time-conditioned version is AdaL $\ N _ { \ell , t } ( s ) = ( 1 + \gamma _ { \ell , t } ) \odot \mathrm { L N } _ { \ell } ( s ) + \beta _ { \ell , t }$ . The AdaLN scale and shift, together with the channelwise output gate $\alpha _ { \ell , t }$ , are supplied by the untilded rows in (19).

For compact notation, suppressing the layer index and biases, define the retrieval functions

$$
\kappa _ { t } ^ { \mathrm { s m } } ( s , k ) : = \exp \left( \mathrm { A d a L N } _ { t } ( s ) ^ { \top } k \right) ,
$$

$$
\kappa _ { t } ^ { \mathrm { g e l u } } ( s , k ) : = \mathrm { g e l u } \big ( \mathrm { A d a L N } _ { t } ( s ) ^ { \top } k \big ) .
$$

Softmax retrieval normalizes $\kappa _ { t } ^ { \mathrm { s m } }$ over memory entries, whereas GELU retrieval uses $\kappa _ { t } ^ { \mathrm { g e l u } }$ directly. The full expressions below make the headwise projections, scaling, biases, and output gates explicit.

Throughout the predictors, we fix the same token dimension $d _ { h }$ . The predictors are specified by the following design choices:

<table><tr><td>Notation</td><td>Design choice</td></tr><tr><td>H</td><td>Number of predictor heads</td></tr><tr><td>R</td><td>Number of learnable memory entries</td></tr><tr><td> $d _ { h } ^ { \prime }$ </td><td>Width of learnable memory</td></tr></table>

The ordinary MLP illustrates these choices. It has a single predictor head $H = 1$ , and $R = 4 d _ { h }$ layer-specific keys and values $\{ k _ { \ell } ^ { ( r ) } , v _ { \ell } ^ { ( r ) } \in \mathbb { R } ^ { d _ { h } ^ { \prime } } \} _ { r = 1 } ^ { R }$ of width $d _ { h } ^ { \prime } = d _ { h }$

$$
g _ { \ell , t } ^ { \mathrm { m l p } } ( s ) = \alpha _ { \ell , t } \odot \left[ \sum _ { r = 1 } ^ { R } \mathrm { g e l u } \Big ( \mathrm { A d a L N } _ { \ell , t } ( s ) ^ { \top } k _ { \ell } ^ { ( r ) } + b _ { \ell } ^ { ( r ) } \Big ) v _ { \ell } ^ { ( r ) } + b _ { \ell } ^ { o } \right] .\tag{24}
$$

These keys and values are shared across patches within a layer but are learned independently at each layer. If $H > 1$ , as with multi-head attention, each predictor head projects queries, keys, and values to width $d _ { h } / H$ assuming $d _ { h }$ is divisible by H.

Table 1 compares the memory banks and layerwise parameter counts for each predictor across all L layers. The spatial-attention backbone and AdaLN parameters are common to all four models and are excluded.
<table><tr><td>Predictor</td><td></td><td>Memory bank Bank parameters</td><td>Layerwise parameters</td></tr><tr><td>LPLD  $g _ { \ell , m , t } ^ { \mathrm { l p l d } }$ </td><td> $\{ z ^ { ( r ) } \} _ { r = 1 } ^ { R }$ </td><td> $R M d _ { h } ^ { \prime }$ </td><td> $3 d _ { h } ^ { 2 } + d _ { h } d _ { h } ^ { \prime } + 4 d _ { h }$ </td></tr><tr><td>SCA  $g _ { \ell , t } ^ { \mathrm { s c a } }$ </td><td> $\{ k ^ { ( r ) } , v ^ { ( r ) } \} _ { r = 1 } ^ { R }$ </td><td> $2 R d _ { h } ^ { \prime }$ </td><td> $2 d _ { h } ( d _ { h } + d _ { h } ^ { \prime } ) + 5 d _ { h } + H$ </td></tr><tr><td>smlp SMLP ge,t</td><td> $\{ k ^ { ( r ) } , v ^ { ( r ) } \} _ { r = 1 } ^ { R }$ </td><td> $2 R d _ { h } ^ { \prime }$ </td><td> $2 d _ { h } ( d _ { h } + d _ { h } ^ { \prime } ) + H d _ { h } ^ { \prime } + 2 d _ { h }$ </td></tr><tr><td>MLP  $g _ { \ell , t } ^ { \mathrm { d i t } }$ </td><td> $\{ k _ { \ell } ^ { ( r ) } , v _ { \ell } ^ { ( r ) } \} _ { \ell , r }$ </td><td> $2 L R d _ { h }$ </td><td>0</td></tr></table>

Table 1 Predictor parameter comparison. The layerwise column includes $\mathrm { Q / K / V / o u t p u t }$ projections, their biases, and SCA’s headwise gain parameters at each layer. For LPLD, a learned compatibility map adds $d _ { h } d _ { h } ^ { \prime }$ parameters once; this cost is zero when $A = I$

## D.4.1 Learned PLD (LPLD)

At each layer, LPLD uses cross-attention to retrieve from a memory bank $\{ \boldsymbol { z } ^ { ( r ) } \in \mathbb { R } ^ { M d _ { h } ^ { \prime } } \} _ { r = 1 } ^ { R }$ shared across layers. Each entry $z ^ { ( r ) } = [ z _ { 1 } ^ { ( r ) } , \dots , z _ { M } ^ { ( r ) } ]$ contains M memory patches $z _ { m } ^ { ( r ) } \in \mathbb { R } ^ { d _ { h } ^ { \prime } }$ , whose width $d _ { h } ^ { \prime }$ may difer from the token width $d _ { h }$ . The query is projected from the conditioned layer input $\mathrm { A d a L N } _ { \ell , t } ( s )$ , while the value is projected directly from the raw memory patch $z _ { m } ^ { ( r ) }$ . To form the key, a layer-shared compatibility map $A \in \mathbb { R } ^ { d _ { h } \times d _ { h } ^ { \prime } }$ first maps each memory patch to token width $d _ { h }$ . We then add positional embeddings, apply the same attention aggregator (including attention AdaLN) used for the input tokens, and apply the same predictor AdaLN used to form the query. We write $S _ { \ell , m , t } ^ { \mathrm { a t t n } } ( A \dot { z } ^ { ( r ) } )$ for this memory attention context. With H

predictor heads, the full predictor is

$$
\begin{array} { r l r } & { q _ { \ell , h , t } ( s ) = W _ { \ell , h } ^ { Q } \mathrm { ~ A d a L N } _ { \ell , t } ( s ) + b _ { \ell , h } ^ { Q } , } & \\ & { k _ { \ell , h , m , t } ^ { ( r ) } = W _ { \ell , h } ^ { K } \mathrm { ~ A d a L N } _ { \ell , t } \bigg ( S _ { \ell , m , t } ^ { \mathrm { a t t n } } ( A z ^ { ( r ) } ) \bigg ) + b _ { \ell , h } ^ { K } , } & \\ & { v _ { \ell , h , m } ^ { ( r ) } = W _ { \ell , h } ^ { V } z _ { m } ^ { ( r ) } + b _ { \ell , h } ^ { V } , } & \\ & { g _ { \ell , m , t } ^ { \mathrm { l p l d } } ( s ) = \alpha _ { \ell , t } \odot \left[ \displaystyle \sum _ { h = 1 } ^ { H } W _ { \ell , h } ^ { O } \left( \displaystyle \sum _ { r = 1 } ^ { R } \mathrm { s m } _ { r } \left\{ \frac { q _ { \ell , h , t } ( s ) ^ { \top } k _ { \ell , h , m , t } ^ { ( j ) } } { \sqrt { d _ { h } / H } } \right\} _ { j \in [ R ] } v _ { \ell , h , m } ^ { ( r ) } \right) + b _ { \ell } ^ { O } \right] . } & \end{array}\tag{25}
$$

Here A acts patchwise, $W _ { \ell , h } ^ { Q } , W _ { \ell , h } ^ { K } \in \mathbb { R } ^ { ( d _ { h } / H ) \times d _ { h } }$ , and $W _ { \ell , h } ^ { V } \in \mathbb { R } ^ { ( d _ { h } / H ) \times d _ { h } ^ { \prime } }$ . When $d _ { h } ^ { \prime } = d _ { h }$ , we may take $A = I$ without adapter parameters; otherwise, A is learned. The shared spatial bank costs $R M d _ { h } ^ { \prime }$ parameters once, plus $d _ { h } d _ { h } ^ { \prime }$ if the compatibility map A is learned.

## D.4.2 Shared cross-attention (SCA)

SCA uses untied key and value banks $\{ k ^ { ( r ) } , v ^ { ( r ) } \in \mathbb { R } ^ { d _ { h } ^ { \prime } } \} _ { r = 1 } ^ { R }$ shared across all layers and patch positions. With H predictor heads, the full predictor is

$$
\begin{array} { r l } & { q _ { \ell , h , t } ( s ) = W _ { \ell , h } ^ { Q } \mathrm { ~ A d a L N } _ { \ell , t } ( s ) + b _ { \ell , h } ^ { Q } , } \\ & { \qquad k _ { \ell , h } ^ { ( r ) } = W _ { \ell , h } ^ { K } k ^ { ( r ) } + b _ { \ell , h } ^ { K } , } \\ & { \qquad v _ { \ell , h } ^ { ( r ) } = W _ { \ell , h } ^ { V } v ^ { ( r ) } + b _ { \ell , h } ^ { V } , } \\ & { \qquad g _ { \ell , t } ^ { \mathrm { s c a } } ( s ) = \alpha _ { \ell , t } \odot \left[ \displaystyle \sum _ { h = 1 } ^ { H } W _ { \ell , h } ^ { O } \left( \eta _ { \ell , h , t } ( s ) \sum _ { r = 1 } ^ { R } \mathrm { s m } _ { r } \left\{ \frac { q _ { \ell , h , t } ( s ) ^ { \top } k _ { \ell , h } ^ { ( j ) } } { \sqrt { d _ { h } / H } } \right\} _ { j \in [ R ] } v _ { \ell , h } ^ { ( r ) } \right) + b _ { \ell } ^ { O } \right] . } \end{array}\tag{26}
$$

Here $W _ { \ell , h } ^ { Q } \in \mathbb { R } ^ { ( d _ { h } / H ) \times d _ { h } }$ and $W _ { \ell , h } ^ { K } , W _ { \ell , h } ^ { V } \in \mathbb { R } ^ { ( d _ { h } / H ) \times d _ { h } ^ { \prime } }$ . The positive headwise gain is

$$
\eta _ { \ell , h , t } ( s ) = \mathrm { s o f t p l u s } ( a _ { \ell , h } ^ { \top } q _ { \ell , h , t } ( s ) + b _ { \ell , h } ^ { \eta } ) ,
$$

with learned $a _ { \ell , h } \in \mathbb { R } ^ { d _ { h } / H }$ and scalar $b _ { \ell , h } ^ { \eta } .$ The shared libraries cost $2 R d _ { h } ^ { \prime }$ parameters once, while their projections and the query/output maps remain layer-specific.

## D.4.3 Shared MLP (SMLP)

SMLP uses the same shared-library structure and H predictor heads as SCA but replaces normalized softmax retrieval with unnormalized GELU retrieval. With H predictor heads, the full predictor is

$$
\begin{array} { r l r } & { q _ { \ell , h , t } ( s ) = W _ { \ell , h } ^ { \cal Q } \mathrm { A d a L N } _ { \ell , t } ( s ) + b _ { \ell , h } ^ { \cal Q } , } & \\ & { } & { k _ { \ell , h } ^ { ( r ) } = W _ { \ell , h } ^ { \cal K } k ^ { ( r ) } , } \\ & { } & { v _ { \ell , h } ^ { ( r ) } = W _ { \ell , h } ^ { \cal V } v ^ { ( r ) } , } \\ & { } & { g _ { \ell , t } ^ { \mathrm { s m l p } } ( s ) = \alpha _ { \ell , t } \odot \left[ \displaystyle \sum _ { h = 1 } ^ { H } W _ { \ell , h } ^ { \cal O } \left( \displaystyle \sum _ { r = 1 } ^ { R } \mathrm { g e l u } \left\{ \frac { q _ { \ell , h , t } ( s ) ^ { \top } k _ { \ell , h } ^ { ( r ) } } { \sqrt { d _ { h } / H } } + ( b _ { \ell , h } ^ { \cal K } ) ^ { \top } k ^ { ( r ) } \right\} v _ { \ell , h } ^ { ( r ) } \right) + b _ { \ell } ^ { \cal O } \right] . } \end{array}\tag{27}
$$

K and V projections are bias-free; the Q and O projections have ordinary afine biases. $( b _ { \ell , h } ^ { K } ) ^ { \top }$ is an additional learned row of the key projection that generates the GELU bias from the shared key prototype. Note that all predictor heads are evaluated for every token: there is no expert router, softmax normalization, or separate retrieval gain. Both the single-head and multihead forms retain the $\mathrm { M L P ^ { \prime } s }$ unnormalized GELU expansion while sharing the underlying representations across depth. The shared libraries cost $2 R d _ { h } ^ { \prime }$ parameters once. Including the layer-specific weight matrices but excluding biases and AdaLN, the cost is $2 R \ddot { d } _ { h } ^ { \prime } + 2 L d _ { h } ( d _ { h } + d _ { h } ^ { \prime } )$ compared with $2 L R d _ { h }$ for independent width-R MLPs.

## D.5 From a single layer to a residual stack

The construction in Sec. 4 isolates aggregation and prediction in a single layer without residual connections.   
In this section, we explain its extension to a residual stack, a multilayer architecture with residual connections.

Let $h _ { \ell }$ denote the hidden token sequence entering layer ℓ at difusion time t. We absorb normalization, AdaLN, and output gates into the attention branch $S _ { \ell , t } ^ { \mathrm { a t t n } }$ and the tokenwise predictor $g _ { \ell , t }$ (an MLP in DiT). The two residual additions give

$$
\begin{array} { r } { \begin{array} { r l } & { \quad s _ { \ell } = h _ { \ell } + S _ { \ell , t } ^ { \mathrm { a t t n } } ( h _ { \ell } ) , } \\ & { \quad h _ { \ell + 1 } = s _ { \ell } + g _ { \ell , t } ( s _ { \ell } ) = h _ { \ell } + \underbrace { S _ { \ell , t } ^ { \mathrm { a t t n } } ( h _ { \ell } ) + g _ { \ell , t } \big ( h _ { \ell } + S _ { \ell , t } ^ { \mathrm { a t t n } } ( h _ { \ell } ) \big ) } _ { F _ { \ell , t } ( h _ { \ell } ) } . } \end{array} } \end{array}\tag{28}
$$

The aggregator interpretation also applies to $s _ { \ell } \colon$ it combines the token’s existing state with attention-selected context, so the identity path retains a contribution from the token itself. The predictor then maps this contextualized state to a residual feature update.

We emphasize that (28) motivates interpreting the residual stack as iterative denoising in feature space: successive aggregation and prediction updates refine the internal residual stream from which the final cleanimage estimate is read out. This view is consistent with interpretations of residual networks as unrolled iterative estimation and feature refinement (Gref et al., 2017; Jastrzebski et al., 2018). This interpretation also motivates consulting a shared memory across depth, providing a reusable bank of learned representations at each refinement step.

However, the clean factorization in a single layer does not necessarily extend to a residual stack. Patch projection and positional embedding are applied once at network entry; later layers operate on already contextualized tokens. Although the shared-channel-operator decomposition of attention still holds layerwise, a token can depend on many original patches, and VO or predictor updates can alter later QK routing. The division of labor between QK and VO is therefore less clear-cut than in a single layer. Similarly, the MLP in one layer can afect attention in subsequent layers, and attention can in turn afect subsequent MLP updates.

## E Experimental details

Datasets. Table 2 summarizes the seven datasets used in our real-image experiments. All real images are represented as 64 × 64 RGB arrays with pixel values scaled to [−1, 1]; training is unconditional and does not use class, identity, or attribute labels.
<table><tr><td>Dataset</td><td>Content</td><td>Training</td><td>Evaluation</td></tr><tr><td>CelebA (Liu et al., 2015)</td><td>Aligned celebrity photographs</td><td>162,770</td><td>19,962</td></tr><tr><td>FFHQ (Karras et al., 2019)</td><td>Photographic faces</td><td>60,000</td><td>5,000</td></tr><tr><td>AAHQ (Liu et al., 2021)</td><td>Artistic portraits</td><td>15,366</td><td>1,921</td></tr><tr><td>AFHQ-v2 (Choi et al., 2020; Karras et al., 2021)</td><td>Cat, dog, and wild-animal faces</td><td>14,335</td><td>1,467</td></tr><tr><td>iCartoonFace (Zheng et al., 2020)</td><td>Cartoon-face recognition crops</td><td>388,710</td><td>22,496</td></tr><tr><td>STL-10 (Coates et al., 2011)</td><td>Natural-object images</td><td>104,500</td><td>8,000</td></tr><tr><td>ImageNet64 (Chrabaszcz et al., 2017)</td><td>Downsampled natural images</td><td>1,281,167</td><td>50,000</td></tr></table>

Table 2 Datasets used throughout the paper. Counts refer to prepared images after filtering. Evaluation uses the held-out splits described below; the ImageNet64 evaluation set is its oficial validation split.

We retain CelebA’s oficial train/test splits and center-crop its images to 178 × 178 before bilinear resizing. For FFHQ, we retain the oficial 60,000-image training partition and divide the remaining 10,000 images equally into validation and test sets. For AAHQ, crops from the same source artwork remain in the same split. AFHQ-v2 pools all three animal categories, retains the oficial evaluation split, and removes exact pixel duplicates. iCartoonFace uses its oficial train/test splits after filtering invalid crops and exact pixel duplicates;

identities may overlap across splits. STL-10 training combines 100,000 unlabeled images with 4,500 labeled training images, reserving the remaining 500 labeled images for validation and retaining the oficial test split. The synthetic GMRF experiment instead uses the analytically specified prior in Appendix E.1 and does not train a DiT.

Common training protocol. All real-image DiTs divide each image into $M = 2 5 6$ nonoverlapping 4×4 patches, giving a $1 6 \times 1 6$ patch grid with $d _ { p } = 3 \cdot 4 ^ { 2 } = 4 8$ coordinates per patch. Following the direct-clean-image formulation in Sec. 2, we use the linear Gaussian interpolation

$$
X _ { t } = t X _ { 0 } + ( 1 - t ) \varepsilon , \qquad \varepsilon \sim { \mathcal N } ( 0 , I ) , \qquad \mathrm { l o g i t } ( t ) \sim { \mathcal N } ( - 0 . 8 , 0 . 8 ^ { 2 } ) .\tag{29}
$$

Thus $a _ { t } = t , b _ { t } = 1 - t ,$ and the normalized observation $Y _ { t } = X _ { t } / t$ has noise level $\sigma _ { t } = ( 1 - t ) / t$ . Let $f _ { \theta } ( X _ { t } , t )$ denote the model’s direct clean-image prediction. The implementation’s VP velocity objective uses $c ( t ) = \operatorname* { m a x } ( 1 - t , 0 . 0 5 )$ and minimizes

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( \theta ) = \frac { 1 } { M d _ { p } } \mathbb { E } \left\| \frac { f _ { \theta } ( X _ { t } , t ) - X _ { t } } { c ( t ) } - \frac { X _ { 0 } - X _ { t } } { c ( t ) } \right\| _ { 2 } ^ { 2 } } \\ { \displaystyle = \frac { 1 } { M d _ { p } } \mathbb { E } \left[ \frac { \| f _ { \theta } ( X _ { t } , t ) - X _ { 0 } \| _ { 2 } ^ { 2 } } { c ( t ) ^ { 2 } } \right] . } \end{array}\tag{30}
$$

The denominator floor is applied to both predicted and target velocities. Away from the floored region, the target velocity is $X _ { 0 } - \varepsilon$ . The floor changes the time weighting but retains the conditional mean as the pointwise population optimum. Unless specified otherwise below, we use AdamW with a constant learning rate of $2 \times 1 0 ^ { - 4 } , ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ , zero weight decay, gradient clipping at norm 1, and global batch size 32. We maintain an exponential moving average (EMA) of the weights with decay 0.999 and use EMA weights for all reported measurements and generated samples. The experiment-specific settings below give the model sizes, training durations, seeds, precision, and any departures from these defaults.

## E.1 Experiment 1-1: Exact denoising context under synthetic GMRF priors

We conduct experiments to illustrate the concept of local context and examine the prediction of Cor. 2: within a specified noise interval, the radius certifying an upper bound on locality error, which we call the certified radius, is nondecreasing as the efective noise level increases.

## E.1.1 Synthetic Gaussian MRF prior.

We consider the Gaussian MRF image prior introduced in Thm. 1, for a single-channel image consisting of $1 6 \times 1 6$ scalar patches of dimension $d _ { p } = 1$ . We draw a graph $G _ { 0 }$ by sampling 384 distinct unordered pairs uniformly without replacement from the $\binom { 2 5 6 } { 2 }$ possible pairs, and independently draw each retained edge’s connection strength as $\omega _ { i j } \sim$ Uniform(0.5, 1.5). We choose 384 edges to give an average of three neighbors per patch: each undirected edge contributes to the degree of both endpoints, so the mean degree is $2 \times 3 8 4 / 2 5 6 = 3$ Then, let $A _ { i j } = A _ { j i } = \omega _ { i j }$ for each sampled edge $\{ i , j \}$ and zero otherwise. We define the prior by

$$
X _ { 0 } \sim { \mathcal { N } } ( 0 , Q ^ { - 1 } ) , \qquad Q = 0 . 2 I + \mathrm { d i a g } ( A { \bf 1 } ) - A .\tag{31}
$$

Here 1 is the all-ones vector, so A1 contains the row sums of A. We use the weighted graph Laplacian $\mathrm { d i a g } ( A { \bf 1 } ) - A$ so that the prior penalizes diferences between connected patches, with larger edge weights encouraging stronger agreement. The positive diagonal shift ensures that Q is positive definite even if the graph is disconnected. We evaluate 41 logarithmically spaced noise levels from $1 0 ^ { - 2 }$ to $1 0 ^ { 2 }$ , together with zero and the corollary’s endpoint $\sigma _ { \mathrm { a b s } } ,$ giving 43 noise levels in total.

## E.1.2 Exact and certified radii for absolute locality error

Since we know the population prior and its graph, we can compute the global population denoiser and the exact local denoiser for each graph-radius context $B _ { r } ( m ) = \{ l : d _ { G _ { 0 } } ( m , l ) \leq r \}$ . Here $d _ { G _ { 0 } }$ is the unweighted shortest-path distance; vertices in diferent components have infinite distance and zero cross-component influence. This allows us to compare the minimum radius achieving a given locality error with the radius certifying its upper bound in Cor. 2. Specifically, observe $Y _ { \sigma } = X _ { 0 } + \sigma \varepsilon$ with independent $\varepsilon \sim \mathcal { N } ( 0 , I )$ , and write $J _ { \sigma } = ( I + \sigma ^ { 2 } Q ) ^ { - 1 }$ for the population denoiser’s Jacobian. The global denoiser and the local denoiser on $B _ { r } ( m )$ are

$$
D _ { m , \sigma } ^ { * } ( y ) = ( J _ { \sigma } y ) _ { m } ,\tag{32}
$$

$$
D _ { m , \sigma } ^ { * , r } ( y ) = ( Q ^ { - 1 } ) _ { m , B _ { r } ( m ) } \bigl [ ( Q ^ { - 1 } ) _ { B _ { r } ( m ) , B _ { r } ( m ) } + \sigma ^ { 2 } I \bigr ] ^ { - 1 } y _ { B _ { r } ( m ) } .\tag{33}
$$

Gaussian conditioning gives the locality error in closed form:

$$
\begin{array} { r l } & { \mathbb { E } \big | D _ { m , \sigma } ^ { * } ( Y _ { \sigma } ) - D _ { m , \sigma } ^ { * , r } ( Y _ { \sigma } ) \big | ^ { 2 } } \\ & { \quad = ( J _ { \sigma } Q ^ { - 1 } ) _ { m m } - ( Q ^ { - 1 } ) _ { m , B _ { r } ( m ) } \big [ ( Q ^ { - 1 } ) _ { B _ { r } ( m ) , B _ { r } ( m ) } + \sigma ^ { 2 } I \big ] ^ { - 1 } ( Q ^ { - 1 } ) _ { B _ { r } ( m ) , m } . } \end{array}\tag{34}
$$

Based on this expression, we compute the exact radius for an absolute tolerance $\epsilon > 0$ as

$$
r _ { \mathrm { e x t } } ( m , \sigma ; \epsilon ) : = \operatorname* { m i n } \left. r \in \mathbb { Z } _ { \geq 0 } : \mathbb { E } \big | D _ { m , \sigma } ^ { * } ( Y _ { \sigma } ) - D _ { m , \sigma } ^ { * , r } ( Y _ { \sigma } ) \big | ^ { 2 } \leq \epsilon \right. .\tag{35}
$$

Corollary 2 also gives the certified radius. For this calculation we use $q _ { - } = 0 . 2$ and $q _ { + } = 0 . 2 + 2 \operatorname* { m a x } _ { i } \sum _ { j } A _ { i j }$ ≈ 21.420774. The former is the exact minimum eigenvalue of $Q ,$ while the latter is a conservative upper bound on its maximum eigenvalue; the theorem’s bound remains valid with this spectral enclosure. Using $\chi _ { t }$ and $\rho _ { t }$ as defined in Appendix C.1, evaluated at $\sigma _ { t } = \sigma$ , the certified radius is

$$
r _ { \mathrm { c r t } } ( \sigma ; \epsilon ) : = \operatorname* { m i n } \left. r \in \mathbb { Z } _ { \geq 0 } : \frac { ( 1 - 1 / \sqrt { \chi _ { t } } ) ^ { 2 } } { q _ { - } ( 1 + \sigma ^ { 2 } q _ { - } ) } \rho _ { t } ^ { 2 r } \leq \epsilon \right. .\tag{36}
$$

At zero noise, both radii are zero. The certified radius is common for all target patches $m$ . We use tolerances $\epsilon = 1 0 ^ { - 3 }$ in the main figure.

## E.1.3 Sensitivity radius as a measure of statistical dependence

Along with these exact and certified radii, we define a sensitivity radius based on the input–output Jacobian, which measures how perturbing each observed patch afects the predicted target patch. For this Gaussian prior, $( J _ { \sigma } ) _ { m l } = \partial D _ { m , \sigma } ^ { * } ( y ) / \partial y _ { l }$ , with $J _ { \sigma } = ( I + \sigma ^ { 2 } Q ) ^ { - 1 }$ . We square its entries to measure sensitivity energy and normalize over input patches as $( J _ { \sigma } ) _ { m l } ^ { 2 } / [ \sum _ { j } ( J _ { \sigma } ) _ { m j } ^ { 2 } ]$ . This normalized derivative energy is the fraction of the target m patch’s total squared sensitivity attributable to input patch l. It measures relative influence, so a common reduction in the magnitude of all derivatives does not by itself change the normalized map. Using this normalization, we define the q-sensitivity radius:

$$
r _ { \mathrm { s n s } } ( m , \sigma ; q ) : = \operatorname* { m i n } \left\{ r \in \mathbb { Z } _ { \geq 0 } : \sum _ { l \in B _ { r } ( m ) } ( J _ { \sigma } ) _ { m l } ^ { 2 } \Bigg / \left[ \sum _ { j } ( J _ { \sigma } ) _ { m j } ^ { 2 } \right] \geq q \right\} .\tag{37}
$$

The self-patch is included, and we evaluate $q \in \{ 0 . 9 0 , 0 . 9 5 \}$ . Unlike the exact and certified radii, this radius concerns relative derivative energy rather than an absolute locality-error tolerance, so the three radii need not agree. For any prior with finite second moments, the population denoiser satisfies the posterior-covariance identity (Dytso et al., 2021)

$$
\frac { \partial D _ { m , t } ^ { * } ( y ) } { \partial y _ { l } } = \sigma _ { t } ^ { - 2 } \operatorname { C o v } ( X _ { 0 , m } , X _ { 0 , l } \mid Y _ { t } = y , t ) , \qquad \sigma _ { t } > 0 .\tag{38}
$$

Thus, its Jacobian reflects the dependence between clean patches that remains after observing the noisy image. For the Gaussian prior considered here, this Jacobian is independent of $y .$

## E.1.4 Interpretation of Fig. 2

Fig. 2(a) Local context need not be spatially contiguous. The first panel shows the radius-3 local contexts of the two selected patches, containing 12 and 50 nodes, respectively, including the targets. Because these sets are defined by graph distance, they need not be square or spatially contiguous: a patch that is distant on the image grid can still be nearby in $G _ { 0 }$ . Each ball is nevertheless connected in the graph. Thus, locality is relative to the dependence structure of the prior, rather than necessarily to proximity on the image grid.

Fig. 2(b) The certified radius is nondecreasing over a bounded noise interval. The second panel compares the exact $r _ { \mathrm { e x t } }$ and certified radii $r _ { \mathrm { c r t } }$ at the absolute error tolerance $\epsilon = 1 0 ^ { - 3 }$ . At low noise, the target patch’s own observation is accurate, so little additional context is needed. As noise increases, observations of correlated patches become more useful, and both radii are nondecreasing over the specified interval in this realization. $r _ { \mathrm { c r t } }$ is conservative: its maximum over the sweep is 16 hops, compared with 4 hops for each exact radius. The dotted line marks $\sigma _ { \mathrm { a b s } } \approx 1 . 1 2 8 9 6 2$ , the endpoint specified in Cor. 2. The corollary guarantees monotonicity of the certified radius on $[ 0 , \sigma _ { \mathrm { a b s } } ] ;$ ; the exact-radius behavior is an empirical observation in this example. Beyond this interval, the radii eventually decrease as the observations provide progressively less information about the clean signal. Indeed, $J _ { \sigma } \sim \sigma ^ { - 2 } Q ^ { - 1 } \mathrm { a s } \sigma \to \infty ,$ so both the global and local posterior means approach the zero prior mean and their absolute diference shrinks. Note that this does not mean that reconstruction improves: locality error measures the discrepancy between the two denoisers, not their error relative to the clean image.

Fig. 2(c) Relative sensitivity reveals how the denoiser distributes influence across patches. The third panel shows that the sensitivity radius $r _ { \mathrm { s n s } }$ can remain broad even when the absolute locality error becomes small. As noise increases from low levels, a larger graph neighborhood is needed to capture the same fraction of the total derivative energy, consistent with the initial growth of $r _ { \mathrm { e x t } }$ and $r _ { \mathrm { c r t } }$ . At extreme noise, however, the absolute response to every observation weakens, while its relative contribution approaches a constant:

$$
\frac { ( J _ { \sigma } ) _ { m l } ^ { 2 } } { \sum _ { j } ( J _ { \sigma } ) _ { m j } ^ { 2 } } \longrightarrow \frac { ( ( Q ^ { - 1 } ) _ { m l } ) ^ { 2 } } { \sum _ { j } ( ( Q ^ { - 1 } ) _ { m j } ) ^ { 2 } } \qquad \mathrm { a s ~ } \sigma \to \infty .\tag{39}
$$

At extreme noise, the posterior covariance approaches the prior covariance $Q ^ { - 1 }$ , while the factor $\sigma ^ { - 2 }$ in the Jacobian makes its absolute entries vanish. This common scale cancels under normalization, leaving a pattern determined by the prior’s covariances, which can extend to distant patches. These results show why a broad sensitivity radius can coexist with a small absolute locality error: the radius describes the distribution of influence, while the error also depends on its absolute strength. Note that in this synthetic setting, we do not claim that the sensitivity radius must increase monotonically with noise, or that observations outside the corresponding neighborhood can be removed without afecting denoising accuracy.

This interpretation provides a link to denoisers trained on real-image datasets: their input–output Jacobians also describe responses to infinitesimal perturbations, so the sensitivity radius $r _ { \mathrm { s n s } }$ can be defined in the same way to characterize local functional context even when the population prior and exact local conditional means are unknown. Furthermore, due to (38), to the extent that learned denoisers approximate the population denoisers, their Jacobian-based sensitivity provides a measure of statistical dependence relevant to denoising.

## E.2 Experiment 1-2: Denoising context in DiTs trained on real-image datasets

We ask whether the denoising-relevant statistical dependence in image datasets exhibits a similar pattern to the exponentially local GMRF model studied in Appendix E.1. We use trained DiTs as estimators of the population denoiser and its Jacobian; through the posterior-covariance identity (38), their sensitivity fields provide an empirical view of this dependence. We use a DiT with approximately 20M parameters, comprising $L = 1 0$ transformer layers, token dimension $d _ { h } = 3 2 0$ , H<sup>˜</sup> = 10 attention heads, and MLP hidden width 1280. We denote its clean-patch prediction by $D _ { m , t } ^ { \mathrm { d i t } }$ and use the common patch representation and training protocol at the start of Appendix E. Following the normalized-observation convention in Sec. 2, $D _ { m , t } ^ { \mathrm { d i t } } ( y )$ denotes the prediction obtained by feeding ty and t to the trained network; thus, all input derivatives below are with respect to y.

Experiment-specific training. We train models on CelebA, iCartoonFace, ImageNet64, and STL-10 using the datasets and splits described above. We analyze checkpoints at epochs 400, 400, 800, and 800, respectively. The non-ImageNet runs use batch size 32 and FP32 on single GPUs, mostly RTX 2080 Ti GPUs; the ImageNet checkpoint uses batch size 256 and BF16 on an H200.

## E.2.1 Sensitivity context set based on input-output Jacobian

Our primary measurement is the Jacobian of output patch m with respect to input patch l, extending the scalar-patch sensitivity in Appendix E.1. For real-image priors, the population derivative $\partial D _ { m , t } ^ { * } ( y ) / \partial y _ { l }$ is

unavailable, so we estimate it using the trained denoiser’s Jacobian,

$$
( J _ { t } ( y ) ) _ { m l } : = \frac { \partial D _ { m , t } ^ { \mathrm { d i t } } ( y ) } { \partial y _ { l } } \in \mathbb { R } ^ { d _ { p } \times d _ { p } } .\tag{40}
$$

Since each patch contains multiple spatial and color coordinates, we measure its sensitivity by the squared Frobenius norm of this block, which sums the squared derivatives over all input–output coordinate pairs. The squared Frobenius norm extends the squared scalar Jacobian entries used in Appendix E.1 to vector-valued patches. This also follows the use of input-gradient sensitivity to study denoiser locality in Niedoba et al. (2025, Sec. 3.1).

For each dataset, we evaluate 1,024 held-out images at 12 noise levels from $\sigma _ { t } = 0 . 0 2$ to 49. In practice, we estimate the squared Frobenius norm using vector–Jacobian products with four independent Rademacher output probes per target patch, following Hutchinson-type Monte Carlo estimation (Avron and Toledo, 2011). We divide by $d _ { p }$ to obtain sensitivity per output coordinate, average over held-out clean images and Gaussian corruption noise, and then normalize over input patches to define the normalized sensitivity weight:

$$
w _ { m l } ( t ) : = \frac { d _ { p } ^ { - 1 } \mathbb { E } _ { X _ { 0 } , \varepsilon } \| ( J _ { t } ( Y _ { t } ) ) _ { m l } \| _ { F } ^ { 2 } } { \sum _ { j = 1 } ^ { M } d _ { p } ^ { - 1 } \mathbb { E } _ { X _ { 0 } , \varepsilon } \| ( J _ { t } ( Y _ { t } ) ) _ { m j } \| _ { F } ^ { 2 } } .\tag{41}
$$

Thus, $w _ { m l } ( t )$ is the fraction of patch m’s expected squared sensitivity attributable to input patch l, with $\begin{array} { r } { \sum _ { l } w _ { m l } ( t ) = 1 } \end{array}$ whenever the total sensitivity is positive.

For real images, the prior’s dependence graph is unknown so we cannot compute the exact radius of a local context as in the synthetic case. Instead, we describe local context in terms of spatial distance and patch count. Specifically, we use the Chebyshev grid distance, a patch hop, defined as $d _ { \mathrm { g r i d } } ( m , l ) = \operatorname* { m a x } \{ | i _ { m } - i _ { l } | , | j _ { m } - j _ { l } | \}$ where $i _ { m } , j _ { m }$ denote the row and column indices of patch m, and define $B _ { r } ( m ) = \{ l : d _ { \mathrm { g r i d } } ( m , l ) \leq r \}$ . These neighborhoods are square regions clipped to the image boundary, with radius measured in patch hops. Using the normalized derivative energy in (41), we define the sensitivity radius by

$$
r _ { \mathrm { s n s } } ( m , t ; q ) : = \operatorname* { m i n } \left\{ r \in \mathbb { Z } _ { \geq 0 } : \sum _ { l \in B _ { r } ( m ) } w _ { m l } ( t ) \geq q \right\} .\tag{42}
$$

The radius is the smallest centered Chebyshev ball containing at least a fraction q of the normalized derivative energy. This ball measures the spatial extent of the context, but the functionally relevant local context set need not be ball-shaped. To allow arbitrary shapes, we also define a context set containing the fewest patches that together account for at least a fraction q of the normalized derivative energy and the sensitivity count as

$$
\mathcal { B } _ { m , t } ( q ) \in \mathop { \mathrm { a r g ~ m i n } } _ { B \subseteq [ M ] } \left\{ | B | : \sum _ { l \in B } w _ { m l } ( t ) \geq q \right\} , \qquad K _ { q } ( m , t ) : = | \mathcal { B } _ { m , t } ( q ) | .\tag{43}
$$

We construct this set by adding patches in descending order of $w _ { m l } ( t )$ until their cumulative energy reaches q, breaking ties by patch index. The selected patches may be scattered or disconnected; the target patch is eligible for selection. For the population curves, we compute each target’s radius and count from the image-averaged energy in (41), then average these quantities over all 256 target patches. For the displayed sensitivity heatmaps, we apply the same construction to the normalized derivative energy of the individual noisy image.

Together, these quantities distinguish spatially broad context from sparse influence carried by a few distant patches. The posterior-covariance identity (38) connects these measurements to statistical dependence remaining after observing the noisy image. Under the Jacobian approximation described above, these sensitivity-based contexts estimate the patches carrying most of the expected squared posterior crosscovariance with the target. They thus characterize the dataset’s denoising-relevant dependence through a learned predictor.

## E.2.2 Attention context set based on attention mass

Additionally, we average the normalized attention matrices computed from the residual-stream features of $D _ { m , t } ^ { \mathrm { d i t } }$ over heads and layers, and use the resulting attention mass in place of $w _ { m l } ( t )$ . Specifically, let $A _ { m l , t } ^ { ( \ell , h ) } ( y )$ denote the softmax attention weight from query patch m to source patch l in head h of layer $\ell ,$ evaluated when the network receives ty and time t. These are the spatial weights in (21), computed from each layer’s features after layer normalization and AdaLN modulation, as described in Appendix D.5. Averaging over the same held-out images and corruption noise gives the mean attention weight:

$$
w _ { m l } ^ { \mathrm { a t t n } } ( t ) : = \mathbb { E } _ { X _ { 0 } , \varepsilon } \left[ \frac { 1 } { L \tilde { H } } \sum _ { \ell = 1 } ^ { L } \sum _ { h = 1 } ^ { \tilde { H } } A _ { m l , t } ^ { ( \ell , h ) } ( Y _ { t } ) \right] .\tag{44}
$$

Each attention row sums to one over source patches, so $\begin{array} { r } { \sum _ { l } w _ { m l } ^ { \mathrm { a t t n } } ( t ) = 1 } \end{array}$ without further normalization. We average the softmax weights themselves, without squaring them or weighting them by residual gates or value/output projections; this measures how attention distributes mass across token positions along the residual stream. Note that both $w _ { m l } ^ { \mathrm { a t t n } } ( t )$ and $w _ { m l } ( t )$ assign normalized weights to input patches l for a given target patch $m ,$ , but they measure diferent quantities: the former describes how attention distributes mass within the network, whereas the latter measures the relative squared sensitivity of the final denoised prediction to each input patch. In this sense, although both weights lie in [0, 1] and sum to one, equal values do not imply equal influence.

Using the same neighborhoods and coverage threshold as for the sensitivity radius, we define the attention radius by

$$
r _ { \mathrm { a t t n } } ( m , t ; q ) : = \operatorname* { m i n } \left\{ r \in \mathbb { Z } _ { \geq 0 } : \sum _ { l \in B _ { r } ( m ) } w _ { m l } ^ { \mathrm { a t t n } } ( t ) \geq q \right\} .\tag{45}
$$

Mirroring (43), we additionally define the attention context set and attention count as

$$
\mathcal { B } _ { m , t } ^ { \mathrm { a t t n } } ( q ) \in \underset { B \subseteq [ M ] } { \mathrm { a r g } \mathrm { m i n } } \left\{ | B | : \sum _ { l \in B } w _ { m l } ^ { \mathrm { a t t n } } ( t ) \geq q \right\} , \qquad K _ { q } ^ { \mathrm { a t t n } } ( m , t ) : = | \mathcal { B } _ { m , t } ^ { \mathrm { a t t n } } ( q ) | .\tag{46}
$$

The radius measures the smallest centered square capturing a fraction q of the mean attention weight, while the count measures the fewest patches carrying that fraction, allowing arbitrary spatial arrangements. We obtain the attention context set by selecting patches in descending order of $w _ { m l } ^ { \mathrm { a t t n } } ( t )$ , with the same tie-breaking rule as in (43). These quantities are computed after averaging the attention maps over heads, layers, and images; the population curves then average the resulting radii or counts over all 256 target patches. For an individual-image heatmap, we average only over heads and layers before selecting its context set.

## E.2.3 Interpretation of Fig. 3

Estimated dependence fields resemble the exponentially local GMRF setting. Figure 3 reveals a pattern of denoising-relevant statistical dependence similar to that of the synthetic GMRF prior. Sensitivity is concentrated at lower noise, while its radius generally increases as noise grows, echoing the broadening of relative sensitivity in Fig. 2. Through the posterior-covariance interpretation in (38), this suggests that predicting a patch relies on a broader set of statistical dependencies when the observation is noisier. Figures 7 and 8 corroborate this overall broadening across additional datasets and at both 90% and 95% coverage curves. The analytic GMRF setting thus provides a useful model for interpreting the estimated dependence fields of real-image datasets, beyond its role as a tractable synthetic example.

The context set need not be spatially contiguous. The estimated dependence fields in the heatmaps show that patches carrying most of the sensitivity can form irregular or disconnected regions rather than a centered square. At lower noise, influence is concentrated near the target and selected nearby structures; at higher noise, the displayed context sets spread over a larger part of the image. The radius and patch count therefore describe complementary aspects of context: its spatial extent and the number of patches carrying a specified fraction of influence.

Contours: gold 50% · white 95% · Ticks point inside · Cyan dots: target patches  
![](images/801d550202070c84ce28a8e7c037d53f0f50a3dc56b950b807c3e97d08739499.jpg)

![](images/6bbb6306e169b6ca3cc8fbe9533552a8b91acdda8b37eb6c70d7045796ae6c1f.jpg)

![](images/6a5dc1406ec057caaa939e184428a8666da62caf0537890e4eab346aa2b2129e.jpg)

![](images/bc0f83f3277e208a845af6598a1896040613f47257cd3f6dd454ea6dbcaf6fef.jpg)  
Figure 7 Sensitivity and attention radii and patch counts on four datasets. Solid/dashed curves show 90%/95% coverage, computed from population maps over 1,024 held-out images and averaged over all 256 target patches.

![](images/1fbb59b494350ea42786db0ee3700103969a66737dab5cc76aeed3db0767690c.jpg)  
Figure 8 Additional context maps for iCartoonFace and STL-10 at $\sigma _ { t } = 0 . 7 5 , 4 9$ , complementing the CelebA and ImageNet examples in Fig. 3. Colors are logarithmic; gold/white contours mark 50%/95% coverage, ticks point inward, and cyan dots mark target patches.

Attention exhibits a related pattern of context formation. Head- and layer-averaged attention also tends to spread over a broader region at higher noise, although its mass is already more difuse than sensitivity at low noise. Attention weights and input–output Jacobians are diferent quantities: the former are softmaxnormalized logits between intermediate tokens, whereas the latter measure how the final denoised prediction responds to the original input. There is therefore no requirement that their spatial patterns agree. Their observed agreement in overall broadening is consistent with attention adapting its aggregation to the dataset’s denoising-relevant dependence structure and contributing to the formation of noise-dependent local context, as suggested by the aggregator interpretation in Sec. 4.1 and Appendix D.5. The additional radius/count curves and heatmaps in Figs. 7 and 8 support this interpretation. Note that the comparison concerns a shared qualitative trend rather than equality of context sets: 95% attention mass and 95% squared derivative energy describe diferent distributions.

## E.3 Experiment 2: Replacing MLPs with shared-memory predictors

## E.3.1 Motivation

Our aggregator–predictor framework decomposes the local conditional mean into a local aggregator and a patch predictor (Def. 2). Sections 4.1 and 4.2 associate these operations with attention and the DiT MLP, respectively. This interpretation motivates replacing the MLP with more interpretable predictors based on retrieval from a reusable bank of learned representations, as introduced in Sec. 4.2, with the aim of attaining comparable denoising and generative performance at a similar trainable-parameter budget. If the proposed alternatives were to fail to do so, this would indicate that their representation or training omits useful properties of an ordinary MLP for denoising. Conversely, a successful replacement provides evidence that the framework supports a useful, explicitly structured alternative. Additionally, this comparison lets us examine the efects of three design choices on empirical performance: patch-specific versus patch-shared storage, softmax cross-attention versus unnormalized GELU retrieval, and a base memory shared across layers versus independent layerwise memory. We take a baseline DiT and replace only its patch predictor with an alternative predictor, leaving everything else fixed; the following section provides the details.

## E.3.2 Settings

Dataset and common backbone. We use the CelebA splits, preprocessing, and common training protocol described at the start of Appendix E, without horizontal-flip augmentation. All models use token width $d _ { h } ~ = ~ 2 5 6$ , depth $L = 8 .$ , and $\tilde { H } = 8$ spatial-attention heads. The patch embedding, spatial-attention architecture, AdaLN conditioning, residual structure, and image-output head are common across families. Each model is trained jointly from initialization; the common backbone specification does not mean that attention weights are frozen or shared between runs.

Predictor configurations and training. We use the implementations in Appendix D.4, with the residual composition in Appendix D.5. Table 3 lists all 21 configurations. For each shared-memory family, we compare $H = 1$ and $H = 8$ predictor heads at the same stored width, with projected width $d _ { h } / H$ per head. Thus, the total projected query, key, and value width stays at 256 rather than growing eightfold with the number of heads. We vary stored width $d _ { h } ^ { \prime }$ and choose MLP hidden widths to obtain three approximate parameter budgets around 8.6–8.9M, 10.2M, and 13.0–13.4M parameters. The table reports exact counts, including the common backbone and all predictor parameters; matching is approximate, rather than exact equality or equal computation.

For training, we use the common objective in Eq. (30) and train all configurations to 400 epochs, with 5,086 optimizer updates per recorded epoch and 2,034,400 updates in total. The global batch size is 32, and the default learning rate is constant except for the LPLD recovery intervention below. All runs use training seed 0.

LPLD computation and noise groups. LPLD’s key path applies spatial attention to each of the R full memory grids at every layer and noise level. This is substantially more expensive than projecting a patch-shared prototype bank: distinct sampled times require distinct conditioned memory computations. We therefore draw four independent times per global batch and share each time among eight images, retaining independent

<table><tr><td>Predictor</td><td>R</td><td> $d _ { h } ^ { \prime }$ </td><td>Parameters, H = 1</td><td>Parameters,  $H = 8$ </td></tr><tr><td>LPLD</td><td>32</td><td>128</td><td>8,935,472</td><td>8,935,472</td></tr><tr><td>LPLD</td><td>32</td><td>256</td><td>10,213,424</td><td>10,213,424</td></tr><tr><td>LPLD</td><td>32</td><td>512</td><td>12,965,936</td><td>12,965,936</td></tr><tr><td>SCA</td><td>1024</td><td>256</td><td>8,642,616</td><td>8,642,672</td></tr><tr><td>SCA</td><td>1024</td><td>512</td><td>10,215,480</td><td>10,215,536</td></tr><tr><td>SCA</td><td>1024</td><td>1024</td><td>13,361,208</td><td>13,361,264</td></tr><tr><td>SMLP</td><td>1024</td><td>256</td><td>8,638,512</td><td>8,652,848</td></tr><tr><td>SMLP</td><td>1024</td><td>512</td><td>10,213,424</td><td>10,242,096</td></tr><tr><td>SMLP</td><td>1024</td><td>1024</td><td>13,363,248</td><td>13,420,592</td></tr><tr><td>MLP</td><td>640</td><td>256</td><td>8,639,536</td><td></td></tr><tr><td>MLP</td><td>1024</td><td>256</td><td>10,215,472</td><td></td></tr><tr><td>MLP</td><td>1792</td><td>256</td><td>13,367,344</td><td></td></tr></table>

Table 3 Complete patch-predictor grid. Each numeric entry in the last two columns specifies one trained configuration. All models use $d _ { h } = 2 5 6$ 2 $L = 8 ,$ , and ${ \tilde { H } } = 8 ;$ predictor head width is 256 for $H = 1$ and 32 for $H = 8$ . R counts image-shaped memories for LPLD, shared prototype pairs for $\mathrm { S C A / S M L P }$ , and hidden units for MLP.

Gaussian noise for each image. This allows the memory-key computation to be reused within a time group, while leaving per-image marginal objective in Eq. (30) unchanged.

During training, the original $H = 8 , d _ { h } ^ { \prime } = 2 5 6 ~ \mathrm { L P L D }$ run developed severe generation artifacts and degraded denoising during training. We exclude that run from the displayed comparisons and use a separate recovery that resumes its epoch-225 checkpoint, including optimizer and EMA state, at learning rate $1 0 ^ { - 4 }$ . The recovery reaches cumulative epoch 400 after 175 additional epochs and has FID 29.197 at that endpoint. Only completed post-restart measurements are drawn for its trajectory; they are not spliced onto the original curve.

Evaluation. All measurements use EMA weights and the same 19,962 held-out test images. We evaluate the velocity objective above and the raw, unclipped clean-image MSE across 32 logarithmically spaced normalized noise levels $\sigma \in [ 0 . 0 5 , 4 ]$ . The denoising score is the trapezoidal integral of this MSE with respect to log σ, divided by $\log ( 4 / 0 . 0 5 )$ , so it is an average error over the specified noise range. At each noise level we use $t = 1 / ( 1 + \sigma )$ to map the normalized observation to the model’s input convention. FID-50k compares 50,000 generated samples with the common real-test feature statistics. Generation uses 50 uniform time intervals from noise to data, with Heun updates and an Euler update on the last interval, and the same denominator floor of 0.05. We evaluate the standard training trajectories every 25 epochs. Figure 4 uses epoch 400 for every displayed metric and model in the current snapshot, rather than selecting each model’s best intermediate FID.

## E.3.3 Results and interpretation

Figure 9 shows that all reported families learn substantial denoising improvements, but their attained errors and generation trajectories difer. Held-out losses generally decrease smoothly, whereas FID is less monotone. The following comparisons use the common epoch-400 snapshot in Fig. 4.

(1) Diferent patches can retrieve from a shared patch memory. The patch-shared SCA and SMLP banks are more competitive than LPLD’s position-specific image-shaped memory in the corresponding parameter-budget groups. At the largest budget with H = 8, LPLD has denoising AUC 0.013087 and FID 19.848, compared with 0.012873 and 17.145 for SCA, and 0.012777 and 15.767 for SMLP. Thus, allocating a separate stored value vector to every patch position is not necessary for competitive denoising in this setting. Sharing prototypes across positions permits a much larger bank—1,024 entries rather than 32—at a similar total parameter count. Overall, the weaker LPLD results show that the most literal learned-image retrieval construction is not automatically the easiest or most efective mechanism for patch predictor.

(2) GELU retrieval is more efective here than softmax retrieval. At every tested $( H , d _ { h } ^ { \prime } )$ pair, SMLP has lower held-out velocity loss, denoising AUC, and FID than SCA. For example, at $H = 8 , d _ { h } ^ { \prime } = 5 1 2 .$ , SCA and SMLP have FIDs 18.413 and 16.652, respectively, with denoising AUCs 0.012882 and 0.012716. Although

![](images/28e7378244c018fd799ea03c70db8c166f5d58abb71e9c7424292ff299d51f67.jpg)  
Figure 9 Training trajectories for the patch-predictor comparison. Held-out velocity loss, denoising log-σ AUC MSE, and FID-50k on CelebA64. Colors identify predictor families; circles and triangles indicate H = 1 and H = 8. Solid, dashed, and dotted curves denote increasing $d _ { h } ^ { \prime }$ within each shared-memory family, or increasing hidden width R for MLP.

SCA already includes the positive, query-dependent headwise gain $\eta _ { \ell , h , t } ( s )$ specified in Appendix D.4, its within-head softmax coeficients remain nonnegative and coupled by normalization. In contrast, GELU allows unnormalized, potentially signed coeficients, providing a more flexible way to combine the shared values. The observed advantage is consistent with this flexibility being useful during training.

(3) Multiple predictor heads are usually beneficial. Holding the stored width fixed, H = 8 improves both held-out denoising measures over H = 1 at every tested width in all three shared-memory families. For SCA and SMLP, it also improves FID at every width. For example, at $d _ { h } ^ { \prime } = 1 0 2 4$ , increasing H from one to eight reduces FID from 22.508 to 17.145 for SCA and from 19.556 to 15.767 for SMLP. LPLD FID improves from 29.269 to 27.167 at $d _ { h } ^ { \prime } = 1 2 8$ and from 24.976 to 19.848 at $d _ { h } ^ { \prime } = 5 1 2$ . This suggests that multiple projections of the shared bank can be useful while keeping the total projected width fixed.

(4) The base memory can be shared across layers. The central comparison is between SMLP, whose base key–value bank is reused at every layer, and the ordinary MLP, whose vectors are independently stored in each layer. For H = 8 SMLP at $d _ { h } ^ { \prime } \in \{ 2 5 6 , 5 1 2 , 1 0 2 4 \}$ , epoch-400 FIDs are 15.017, 16.652, and 15.767, compared with 19.550, 17.574, and 16.060 for the respective MLP controls. At the smallest budget, their denoising AUCs are 0.0127499 and 0.0127496, and their velocity losses are 0.076336 and 0.076360. At the middle and largest budgets, SMLP has approximately 0.24% and 1.33% higher denoising AUC, respectively, despite its lower FID. Thus the shared-memory predictor is competitive in both denoising and generation, with a small denoising tradeof at the larger budgets. We emphasize that the result is a constructive demonstration that independently stored memory at every layer is not required for this level of performance.

Scope. We close by noting the scope of this experiment. It is a controlled study of unconditional face generation at $6 4 \times 6 4$ resolution, with modest model sizes and one training seed per configuration. These limits constrain how broadly the findings of our experiment can be applied. We therefore do not claim that all DiT MLPs should share memory. Instead, the results support considering an explicit shared memory architecture when a common representation bank would be useful for inspection or future intervention. Our experiment shows that this architecture is feasible at least in the setting studied here.

## E.4 Experiment 3: Component compatibility in cross-dataset adaptation

Motivation and design. The aggregator–predictor framework separates forming a patch’s local context from predicting its clean value using learned representations. Section 4.1 associates the first operation with attention and Sec. 4.2 associates the second with the MLP. Experiment 1 finds that DiT sensitivity and attention maps exhibit related noise-dependent broadening on real images; Experiment 2 shows that explicit retrieval from a shared memory supports competitive denoising and generation quality in a controlled CelebA64 study. We now ask whether component compatibility across image domains is consistent with the proposed attention and MLP roles. This question also matters when target data or adaptation compute are limited: fine-tuning methods must decide which pretrained parameters to update, and the best choice can depend on the source and target distributions (Kumari et al., 2023; Choi et al., 2024).

The attention decomposition in (7) gives a more specific hypothesis. Within an isolated branch, QK controls spatial weights and VO transforms and transports features; the MLP admits a key–value representation for tokenwise updates. This motivates the hypothesis that source QK may remain compatible with adapted VO and MLP parameters when source and target have similar spatial context, with larger shifts associated with reduced compatibility. Source-QK compatibility may also be directional: a broadly trained source may support contextual relations needed by a narrower target, whereas the reverse transfer need not hold. We probe these hypotheses by adapting QK, VO, and MLP jointly to each target, then replacing one or more adapted groups with their source weights without further optimization. The eight resulting hybrids show how much each subset of jointly learned changes matters for reproducing the jointly adapted model.

Training and endpoint selection. We use the 20M-class DiT of Appendix E.2 on 64 × 64 RGB images: $L = 1 0 ,$ $d _ { h } = 3 2 0 , \tilde { H } = 1 0 .$ , and MLP hidden width 1280. The twelve directions are CelebA→AAHQ/FFHQ/AFHQv2/STL-10, STL-10→AAHQ/FFHQ/AFHQ-v2/CelebA, and ImageNet64→AAHQ/FFHQ/AFHQ-v2/CelebA. Source training uses seed $0 ;$ adaptation starts from EMA checkpoints at source epochs 400, 800, and 800, respectively, and is repeated with two adaptation seeds. Only the attention QK and VO projections along with MLP block weights are updated jointly. Patch, position, and time embeddings, AdaLN, the output head, and other parameters stay fixed. Freezing AdaLN keeps its learned time conditioning while holding the denoising schedule constant; it assumes that this aspect of time dependence transfers across datasets.

Training uses the common VP objective, global batch 32, AdamW with learning rate $2 \times 1 0 ^ { - 4 }$ , and a fresh optimizer for adaptation. For source and target training-set sizes $N _ { \mathrm { s r c } } , N _ { \mathrm { t g t } }$ , one reference epoch is $s _ { \mathrm { r e f } } = \mathrm { m i n } ( \lfloor N _ { \mathrm { s r c } } / 3 2 \rfloor , \lfloor N _ { \mathrm { t g t } } / 3 2 \rfloor )$ updates; adaptation stops by 200 reference epochs, with checkpoints every 25. For each direction, we select the checkpoint minimizing the joint model’s two-seed mean FID-50k on the monitored target split, with earlier epochs breaking ties, and use that same reference epoch for both seeds (Table 4). Each eight-state hybrid cube uses the selected checkpoints, EMA weights, and the same 1,024 Gaussian initial latents per state, sampled in FP32 with 50-step VP Heun. Note that since the three architectural components are adapted jointly, these eight-state swaps measure compatibility among jointly adapted components; they do not measure what independently fine-tuned restricted models could learn.

Scope. Component transplantation diagnoses compatibility among parameter groups obtained by joint adaptation; it neither determines their performance when optimized in isolation nor uniquely assigns computation to components in a multilayer residual network. Separate restricted-training comparisons are reported in Appendix E.4.5. Spatial MMD, EA, and SA are descriptive proxies for dataset geometry and endpoint agreement, not direct measurements of denoising context. The twelve transfer directions share datasets and pretrained sources, and the component contrasts share cube terms; their correlations are descriptive rather than independent-sample estimates of a population relationship.

## E.4.1 Spatial MMD for dataset similarity

To describe spatial similarity between unpaired real-image datasets, we compare distributions of whole-image patch self-similarity descriptors using maximum mean discrepancy (MMD). For an individual clean image $x _ { 0 } ,$ let $u _ { m } ( x _ { 0 } )$ be its final DINOv2 ViT-B/14 feature at patch position m after square resize to 224 × 224, yielding a $1 6 \times 1 6$ patch grid (Oquab et al., 2024). For each feature channel, subtract its mean over the 256 patch positions and divide by its spatial sample standard deviation plus 10<sup>−6</sup>; then normalize each patch vector to unit length, obtaining $\widehat { u } _ { m } ( x _ { 0 } )$ . For patch positions $m , l \in [ 2 5 6 ]$ , we form $K _ { x _ { 0 } } ( m , l ) = \widehat { u } _ { m } ( x _ { 0 } ) ^ { \top } \widehat { u } _ { l } ( x _ { 0 } )$ and let $v ( x _ { 0 } ) \in \mathbf { \tilde { R } ^ { 3 2 6 4 0 } }$ collect the upper of-diagonal entries of $K _ { x _ { 0 } }$ in a fixed raster order. Thus each descriptor retains where pairs of patches have similar features. Patch self-similarity has been used to describe image structure (Tumanyan et al., 2022), and spatially normalized patch features have been studied in difusion representation alignment (Singh et al., 2026); applying MMD to these descriptors for cross-dataset comparison is our construction.

<table><tr><td>Transfer</td><td>Epoch</td><td>FID</td><td>Transfer</td><td>Epoch</td><td>FID</td></tr><tr><td> $\mathrm { C e l e b A } {  } \mathrm { A A H Q }$ </td><td>75</td><td>40.767</td><td> $\mathrm { S T L - 1 0 {  } A A H Q }$ </td><td>150</td><td>40.592</td></tr><tr><td> $\mathrm { C e l e b A {  } F F H Q }$ </td><td>100</td><td>25.252</td><td> $\mathrm { S T L - 1 0 {  } F F H Q }$ </td><td>100</td><td>28.231</td></tr><tr><td> $\mathrm { C e l e b A {  } A F H Q { - } v 2 }$ </td><td>200</td><td>44.821</td><td> $\mathrm { S T L \mathrm { - } 1 0 {  } A F H Q \mathrm { - } v 2 }$ </td><td>200</td><td>49.119</td></tr><tr><td> $\mathrm { C e l e b A { \to } S T L { \ - } 1 0 }$ </td><td>200</td><td>68.050</td><td> $\mathrm { S T L - 1 0 {  } C e l e b A }$ </td><td>200</td><td>17.262</td></tr><tr><td> $\mathrm { I m a g e N e t 6 4 {  } A A H Q }$ </td><td>150</td><td>34.493</td><td> $\mathrm { I m a g e N e t 6 4 {  } F F H Q }$ </td><td>100</td><td>22.640</td></tr><tr><td> $\mathrm { I m a g e N e t 6 4 {  } A F H Q { \mathrm { - } } v 2 }$ </td><td>200</td><td>38.840</td><td> $\mathrm { I m a g e N e t 6 4 {  } C e l e b A }$ </td><td>100</td><td>14.620</td></tr></table>

Table 4 Selected joint-adaptation endpoints. Epochs use the smaller source/target training set as reference; FID is the two-seed mean FID-50k used for checkpoint selection. STL-10 and ImageNet64 source checkpoints are both from epoch 800.

For descriptors $v , v ^ { \prime } ,$ , we use the Gaussian kernel $\kappa ^ { \mathrm { s p } } ( v , v ^ { \prime } ) = \exp [ - | | v - v ^ { \prime } | | _ { 2 } ^ { 2 } / ( 2 \cdot 3 2 6 4 0 \tau ^ { 2 } ) ]$ . We fix $\tau ^ { 2 }$ to the median of $\| v - v ^ { \prime } \| _ { 2 } ^ { 2 } / 3 2 6 4 0$ over all distinct pairs in a balanced calibration bank of 128 held-out images from each of six datasets, separately for normalized and raw patch descriptors. Let $A = \{ a _ { i } \} _ { i = 1 } ^ { n _ { A } }$ and $B = \{ b _ { j } \} _ { j = 1 } ^ { n _ { B } }$ be independent evaluation banks of real-image descriptors. For distinct datasets, $n _ { A } = n _ { B } = 1 0 0 0 ;$ for a diagonal cell, the banks are disjoint halves of one dataset and $n _ { A } = n _ { B } = 5 0 0$ . Conditional on the calibrated kernel bandwidth, the following is an unbiased estimator of the population squared MMD between the corresponding distributions of $v ( X _ { 0 } )$ (Gretton et al., 2012):

$$
\widehat { \mathrm { M M D } } _ { \mathrm { s p } } ^ { 2 } ( A , B ) = \frac { \sum _ { i \neq j } \kappa ^ { \mathrm { s p } } ( a _ { i } , a _ { j } ) } { n _ { A } ( n _ { A } - 1 ) } + \frac { \sum _ { i \neq j } \kappa ^ { \mathrm { s p } } ( b _ { i } , b _ { j } ) } { n _ { B } ( n _ { B } - 1 ) } - \frac { 2 \sum _ { i , j } \kappa ^ { \mathrm { s p } } ( a _ { i } , b _ { j } ) } { n _ { A } n _ { B } } .\tag{47}
$$

Smaller values indicate more similar distributions of these fixed-grid descriptors, not necessarily more similar physical geometry or semantic content. The estimate may be slightly negative for finite banks. Figure 10 shows the spatially normalized version, with each diagonal comparing two disjoint halves of the same dataset.

![](images/5b9b91f029f1266b7f0e55609bc37f4c3be42662656a2574bd3380ae55f56d6e.jpg)  
Figure 10 Spatial self-similarity distance between real datasets. Unbiased Gaussian-kernel $\mathrm { M M D ^ { 2 } }$ of spatially normalized, fixed-grid DINOv2 patch descriptors. Cell and colorbar values are in units of $1 0 ^ { - 2 }$ . Of-diagonal cells compare 1,000 unpaired images from each dataset; diagonal cells compare disjoint 500-image halves. Lower means more similar descriptor distributions.

## E.4.2 Endpoint agreement and component contrasts

For each transfer direction and adaptation seed, write $\mathbf { b } = ( b _ { Q } , b _ { V } , b _ { M } ) \in \{ 0 , 1 \} ^ { 3 }$ for a hybrid state in QK/VO/MLP order: 0 restores source weights and 1 retains jointly adapted weights. A bit string such as 011 abbreviates $\mathbf { b } = ( 0 , 1 , 1 )$ . Let $x _ { 0 } ^ { ( \mathbf { b } ) } ( \xi _ { i } )$ be the clean image generated by hybrid b from the ith shared Gaussian sampling initialization $\xi _ { i } ; x _ { 0 } ^ { ( 0 0 0 ) } ( \xi _ { i } )$ is the source output and $x _ { 0 } ^ { ( 1 1 1 ) } ( \xi _ { i } )$ is the joint-adaptation output. Visual comparisons use the same latents but can be ambiguous, so we also measure their agreement with the paired joint endpoint. Let $f _ { \mathrm { g } } ( x _ { 0 } )$ be the unit-normalized final DINOv2 class token and let $v ( x _ { 0 } )$ be the spatial descriptor above. For N = 1024 matched latents, define

$$
\mathcal { E } _ { \mathbf { b } } ^ { \mathrm { g } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| f _ { \mathrm { g } } ( x _ { 0 } ^ { ( \mathbf { b } ) } ( \xi _ { i } ) ) - f _ { \mathrm { g } } ( x _ { 0 } ^ { ( 1 1 1 ) } ( \xi _ { i } ) ) \| _ { 2 } ^ { 2 } , \qquad \mathrm { E A } _ { \mathbf { b } } = 1 - \frac { \mathcal { E } _ { \mathbf { b } } ^ { \mathrm { g } } } { \mathcal { E } _ { 0 0 0 } ^ { \mathrm { g } } } ,\tag{48}
$$

$$
\mathcal { E } _ { \mathbf { b } } ^ { \mathrm { s p } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { \| v ( x _ { 0 } ^ { ( \mathbf { b } ) } ( \xi _ { i } ) ) - v ( x _ { 0 } ^ { ( 1 1 1 ) } ( \xi _ { i } ) ) \| _ { 2 } ^ { 2 } } { 3 2 6 4 0 } ,
$$

$$
\mathrm { S A } _ { \mathbf { b } } = 1 - { \frac { \mathcal { E } _ { \mathbf { b } } ^ { \mathrm { s p } } } { \mathcal { E } _ { 0 0 0 } ^ { \mathrm { s p } } } } .\tag{49}
$$

We compute EA only when $\mathcal { E } _ { 0 0 0 } ^ { \mathrm { g } } \geq 0 . 0 1$ and SA only when $\mathcal { E } _ { 0 0 0 } ^ { \mathrm { s p } } \geq 1 0 ^ { - 6 }$ . Both scores set the source to 0 and the joint endpoint to 1; negative values are retained if a hybrid is farther from the joint endpoint than the source. EA describes global feature agreement and SA describes agreement of patch self-similarity at fixed image coordinates. SA and the dataset MMD use the same spatial descriptor but answer diferent questions: SA compares matched generated images, while MMD compares distributions of unpaired real images.

For either score $\mathrm { A g r } _ { \mathbf { b } } \in \{ \mathrm { E A } _ { \mathbf { b } } , \mathrm { S A } _ { \mathbf { b } } \}$ , finite diferences across cube states quantify component dependence. The penalty for reverting one adapted group from the joint model is $R _ { Q } = \mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 0 1 1 } , R _ { V } = \mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 1 0 1 }$ or $R _ { M } = \mathrm { { A g r } _ { 1 1 1 } - \mathrm { { A g r } _ { 1 1 0 } ; } }$ we also use $R _ { \operatorname* { m i n } } = \operatorname* { m i n } ( R _ { Q } , R _ { V } , R _ { M } )$ to ask whether all three reversions are costly. The gain beyond adding the three isolated changes is

$$
\begin{array} { r l } & { \mathrm { J N A } ( \mathrm { A g r } ) = \mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 1 0 0 } - \mathrm { A g r } _ { 0 1 0 } - \mathrm { A g r } _ { 0 0 1 } + 2 \mathrm { A g r } _ { 0 0 0 } , } \\ & { \mathrm { T W I } ( \mathrm { A g r } ) = \mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 1 1 0 } - \mathrm { A g r } _ { 1 0 1 } - \mathrm { A g r } _ { 0 1 1 } + \mathrm { A g r } _ { 1 0 0 } + \mathrm { A g r } _ { 0 1 0 } + \mathrm { A g r } _ { 0 0 1 } - \mathrm { A g r } _ { 0 0 0 } . } \end{array}
$$

Here JNA measures joint non-additivity relative to the three isolated gains, while TWI isolates the three-way interaction after subtracting pairwise interactions. Positive $R _ { \mathrm { m i n } }$ means each single reversion lowers endpoint agreement; positive JNA means the joint gain exceeds the sum of isolated gains. All contrasts are computed within each seed’s cube before averaging seeds. Figure 11 gives the two agreement scores for every hybrid state and transfer direction; interpretation of their component patterns follows in the next section.

## E.4.3 Results and interpretation

(1) Higher spatial similarity is associated with QK compatibility; larger shifts accompany coadaptation. Figure 5(a) shows that spatial MMD<sup>2</sup> correlates with the QK reversion diference $\mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 0 1 1 } \ : ( \rho _ { \mathrm { E A } } = 0 . 7 3 9$ $\rho _ { \mathrm { S A } } = 0 . 8 4 4 )$ : source QK remains more compatible with adapted VO+MLP when dataset geometry is similar. The selected CelebA→AAHQ/FFHQ images retain recognizable faces in state 011, whereas CelebA→STL-10 has little recognizable structure until joint adaptation (111; Figs. 1 and 13). Across all twelve nested state diferences, this QK diference ranks first by the weaker of its two absolute correlations (Table 5). Figure 12 extends the pattern to joint non-additivity, the minimum single-component reversion penalty, and threeway interaction: all increase with spatial distance $( \rho _ { \mathrm { E A } } / \rho _ { \mathrm { S A } } = 0 . 6 3 4 / 0 . 6 5 5 , 0 . 7 3 9 / 0 . 8 4 4$ , and 0.641/0.774, respectively). These observations are consistent with stronger component coadaptation across larger spatial shifts, subject to the dependence limitations noted above.

(2) VO dependence increases with functional coadaptation. In Fig. 5(b), joint non-additivity correlates strongly with the VO reversion diference $\mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 1 0 1 } \ ( \rho _ { \mathrm { E A } } = 0 . 8 3 2 , \rho _ { \mathrm { S A } } = 0 . 9 3 0 )$ . Table 6 shows that this association exceeds those for reverting QK or MLP in both scores. The generated images illustrate the same dependence: reverting VO in state 101 leaves fragmented texture for CelebA→STL-10 and distorted outputs for STL-10/ImageNet64→CelebA, compared with the coherent joint endpoints in 111 (Figs. 1, 14, and 15).

(3) Broad-domain source QK retains partial compatibility on face targets. Figure 5(c) shows partial source-QK agreement across AAHQ, FFHQ, and AFHQ for all three sources $( \mathrm { E A _ { 0 1 1 } } = 0 . 4 6 5 - 0 . 8 1 5 , \mathrm { S A _ { 0 1 1 } } = 0 . 5 0 7$ 0.686). For STL-10→CelebA, retained agreement $( \mathrm { E A _ { 0 1 1 } , S A _ { 0 1 1 } } ) = ( 0 . 2 6 7 , 0 . 3 4 0 )$ exceeds the reciprocal

<table><tr><td></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.37</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.56</td><td rowspan=2 colspan=1>1.001.00</td></tr><tr><td></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.28</td><td rowspan=1 colspan=1>0.46</td></tr><tr><td></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.49</td><td rowspan=2 colspan=1>1.001.00</td></tr><tr><td></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.48</td></tr><tr><td rowspan=6 colspan=1>STL-10 → CelebAImageNet64 → AAHQImageNet64 → FFHQmageNet64 → AFHQ-v2ImageNet64 → CelebA</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.52</td><td rowspan=2 colspan=1>1.001.00</td></tr><tr><td rowspan=1 colspan=2>0.000.02</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.39</td></tr><tr><td rowspan=1 colspan=8>000 100 010 110 001 101 011 111State (QK / VO / MLP)</td></tr></table>

<table><tr><td colspan="16"></td></tr><tr><td></td><td colspan="7">Spatial agreement (SA) 0.00 0.12 0.14</td><td colspan="2">0.50 0.69</td><td colspan="3">1.00</td></tr><tr><td></td><td>0.00</td><td>0.11</td><td>0.09</td><td>0.21</td><td>0.41</td><td>0.43</td><td></td><td>0.60</td><td>1.00</td><td rowspan="3"></td></tr><tr><td></td><td>0.00</td><td>0.10</td><td>0.11</td><td>0.25</td><td>0.26</td><td>0.37</td><td>0.53</td><td>1.00</td><td></td></tr><tr><td>1.0</td><td>0.00</td><td>0.28</td><td>0.02</td><td>0.30</td><td>-0.17</td><td>0.00</td><td>0.02</td><td>1.00</td><td>1.0</td></tr><tr><td>0.8</td><td>0.00</td><td>0.08</td><td>0.11</td><td>0.23</td><td>0.21</td><td>0.34</td><td></td><td>0.57</td><td>1.00</td><td rowspan="4">0.8 0.6 -0.4 0.2</td></tr><tr><td>0.6</td><td>0.00</td><td>0.10</td><td>0.14</td><td>0.21</td><td>0.17</td><td></td><td>0.32</td><td>0.51</td><td>1.00</td></tr><tr><td>0.4</td><td>0.00</td><td>0.08</td><td>0.18</td><td>0.32</td><td>0.14</td><td></td><td>0.24</td><td>0.53</td><td>1.00</td></tr><tr><td>0.2</td><td>0.00</td><td>0.09</td><td>0.04</td><td>0.11</td><td>0.03</td><td>0.20</td><td></td><td>0.34</td><td>1.00</td></tr><tr><td rowspan="4">0.0</td><td>0.00</td><td>0.07</td><td>0.14</td><td>0.24</td><td>0.14</td><td></td><td>0.27</td><td>0.63</td><td>1.00</td><td rowspan="4"></td></tr><tr><td>0.00</td><td>0.10</td><td>0.16</td><td>0.26</td><td>0.10</td><td></td><td>0.26</td><td>0.61</td><td>1.00</td></tr><tr><td>0.00</td><td>0.08</td><td>0.17</td><td>0.32</td><td></td><td>0.08</td><td>0.23</td><td>0.58</td><td>1.00</td></tr><tr><td>0.00</td><td>0.10</td><td>0.07</td><td>0.15</td><td>-0.01</td><td></td><td>0.14</td><td>0.44</td><td>1.00</td></tr><tr><td>000</td><td></td><td>100</td><td>0i0</td><td>110</td><td>001</td><td>State (QK / VO / MLP)</td><td>101</td><td>011</td><td>111</td><td></td></tr></table>

Figure 11 Agreement of all hybrid states with the jointly adapted endpoint. Global-feature endpoint agreement EA (left) and relative spatial endpoint agreement SA (right), averaged over two adaptation seeds. Each row is a transfer direction, and columns give QK/VO/MLP bits (0: source, 1: jointly adapted). The same 1,024 noise latents are used for all eight states of each cube. Each score is 0 for the source and 1 for the joint endpoint; negative values are possible.

Table 5 Spatial distance and nested state diferences. Spearman correlations with dataset spatial $\mathrm { M M D ^ { 2 } }$ use 12 two-seed direction means for each score. Each row changes one component with its partners fixed. Rows are sorted by decreasing min $\big ( \big | \rho _ { \mathrm { E A } } \big | , \big | \rho _ { \mathrm { S A } } \big | \big )$
<table><tr><td>Interpretation</td><td>State difference</td><td>ρEA ρSA</td></tr><tr><td>QK reversion penalty†</td><td> $\mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 0 1 1 }$ </td><td>0.739 0.844</td></tr><tr><td>MLP adaptation cost (VO adapted) †</td><td> $\mathrm { A g r } _ { 0 1 0 } - \mathrm { A g r } _ { 0 1 1 }$ </td><td>0.718 0.809</td></tr><tr><td>VO adaptation cost (QK adapted)</td><td> $\mathrm { A g r } _ { 1 0 0 } - \mathrm { A g r } _ { 1 1 0 }$ </td><td>0.771 0.683</td></tr><tr><td>MLP-only adaptation cost</td><td> $\mathrm { A g r } _ { 0 0 0 } - \mathrm { A g r } _ { 0 0 1 }$ </td><td>0.746 0.673</td></tr><tr><td>MLP adaptation cost (QK adapted)</td><td> $\mathrm { A g r } _ { 1 0 0 } - \mathrm { A g r } _ { 1 0 1 }$ </td><td>0.644 0.673</td></tr><tr><td>VO reversion penalty</td><td> $\mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 1 0 1 }$ </td><td>0.641 0.673</td></tr><tr><td>VO-only adaptation cost</td><td> $\mathrm { A g r } _ { 0 0 0 } - \mathrm { A g r } _ { 0 1 0 }$ </td><td>0.455 0.424</td></tr><tr><td>QK adaptation cost (VO adapted)</td><td> $\mathrm { A g r } _ { 0 1 0 } - \mathrm { A g r } _ { 1 1 0 }$ </td><td>0.676 0.378</td></tr><tr><td>MLP reversion penalty</td><td> $\mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 1 1 0 }$ </td><td>0.602 0.375</td></tr><tr><td>QK adaptation gain (MLP adapted)</td><td> $\mathrm { A g r } _ { 1 0 1 } - \mathrm { A g r } _ { 0 0 1 }$ </td><td>0.266 0.676</td></tr><tr><td>QK-only adaptation gain</td><td> $\mathrm { A g r } _ { 1 0 0 } - \mathrm { A g r } _ { 0 0 0 }$ </td><td>0.081 0.179</td></tr><tr><td>VO adaptation cost (MLP adapted)</td><td> $\mathrm { A g r } _ { 0 0 1 } - \mathrm { A g r } _ { 0 1 1 }$ </td><td>0.168 0.018</td></tr></table>

![](images/dd775eeae97ed22159895d484c663cfb7fe83d035764bac7c6d0c16469b202ea.jpg)

![](images/982d1eb5fd03e45754278fe1730fef33caf91f14f8d46ffe89b8a5adb99b9aaf.jpg)

![](images/8e49c6a6fc24b87f0ea6cfadf04b9b5400db2ac9b2c735105120c66f154f5fd3.jpg)  
Figure 12 Spatial distance and functional coordination. Spatial MMD<sup>2</sup> versus (a) joint non-additivity, (b) minimum reversion penalty, and (c) three-way interaction, defined in Appendix E.4.2. Points average two adaptation seeds; colors, markers, and lines follow Fig. 5.

CelebA→STL-10 values (0.143, 0.021); ImageNet64→CelebA also retains partial agreement (0.387, 0.444). The selected natural-source images show varying face or animal structure with source QK in 011, while 111 generally yields clearer endpoints; some STL-10→CelebA examples remain weak in 011 (Fig. 14).

Table 6 Component dependence and joint non-additivity. Spearman correlations between $\mathrm { J N A _ { A g r } }$ and each displayed diference use 12 two-seed direction means, separately for Agr = EA, SA. Note that since all quantities share cube terms, the diferences between correlations are descriptive.
<table><tr><td>Reverted component</td><td>State difference</td><td>ρEA</td></tr><tr><td>VO</td><td> $\mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 1 0 1 }$ </td><td>0.832</td></tr><tr><td>QK</td><td> $\mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 0 1 1 }$ </td><td>0.601</td></tr><tr><td>MLP</td><td> $\mathrm { A g r } _ { 1 1 1 } - \mathrm { A g r } _ { 1 1 0 }$ </td><td>0.664</td></tr></table>

## E.4.4 Additional generated-image cubes

Figures 13–15 extend Fig. 1 with more latents and all eight component states per target. They use the same layout and selected endpoints (Table 4), with EMA weights, FP32 sampling, and 50-step VP Heun.

![](images/194ab3587cc11ba3eb641048a12186a9c206fb092706cc3c008800a6d86dbea1.jpg)  
Figure 13 CelebA source: three additional latents across four targets.

![](images/22be6a5893004b07700e324497c5e31cdc8ddb1991f91f50de2466bb419b1f39.jpg)  
Figure 14 STL-10 source: three latents across four targets.

![](images/7c70f3fde38779dd78537c813c0be283a3c4367f74c772a57419ac3eafbe0683.jpg)  
Figure 15 ImageNet64 source: three latents across four targets.

## E.4.5 Training trajectories and restricted fine-tuning

We compare joint QK/VO/MLP fine-tuning (111) with training all parameters from scratch and with independent fine-tuning restricted to QK or QK+VO, using the same target-domain update budget (Figs. 16– 18). Joint fine-tuning is comparable to or better than scratch training in held-out denoising and usually in FID-50k; STL-10→CelebA is an exception for FID at the selected joint endpoint. QK-only and QK+VO-only fine-tuning generally lag further behind on larger domain shifts under the tested training protocol and update budget. Each target has one row, with denoising AUC, VP velocity loss, and FID-50k in columns; lower is better. The two loss axes use $1 0 ^ { - 2 }$ units, as in Fig. 9. Solid and dashed curves denote adaptation seeds 0 and 1, and dotted scratch curves use seed 0. Vertical lines mark the selected joint endpoints; source pretraining is excluded from the target-domain update count.

![](images/56268bb284228ffcda33e3342f519a9d2e6ea2da8e4e232508423fbd24975f42.jpg)  
Figure 16 CelebA-source adaptation trajectories. Rows are AAHQ, FFHQ, AFHQ-v2, and STL-10 targets.

![](images/ddebd3a4475f5aceb49a62358fff679ad0e17882885142826365f527dc5e9ac7.jpg)  
Figure 17 STL-10-source adaptation trajectories. The source checkpoint is from epoch 800.

![](images/642eea767c1ac05579b03cbd83a96718cb58fd7bf70044b8f61184a56570aec4.jpg)  
Figure 18 ImageNet64-source adaptation trajectories. The source checkpoint is from epoch 800; rows are AAHQ, FFHQ, AFHQ-v2, and CelebA targets.