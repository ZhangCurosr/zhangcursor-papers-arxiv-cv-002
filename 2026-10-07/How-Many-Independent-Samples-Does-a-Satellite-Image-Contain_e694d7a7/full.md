# How Many Independent Samples Does a Satellite Image Contain? Generalization Bounds for Spatially Dependent Data

Robin Young

Department of Computer Science and Technology

University of Cambridge, UK

ray25@cam.ac.uk

Abstract—Machine learning classifiers for remote sensing imagery are typically evaluated as though every pixel were an independent sample. Spatial autocorrelation violates this assumption, since neighboring pixels carry redundant information which inflates sample sizes. How many independent samples does a satellite image actually contain? For an $n \times n$ image whose spatial correlation persists over a range of r pixels, the effective sample size is $\Theta ( n ^ { 2 } / r ^ { 2 } )$ , not $n ^ { 2 } .$ . We prove this as a finite-sample upper bound for classifiers on spatially correlated data, and show via a matching lower bound that the rate is tight, and no algorithm can do better. We extend the results to images with directional correlation and spatially varying correlation structure. Our result justifies spatial cross-validation since block holdout with separation proportional to the correlation range achieves optimal generalization guarantees, while random holdout can underestimate confidence interval widths by a factor proportional to r. We validate the theory on synthetic data and satellite image tiles from three sensors (Landsat 8, Sentinel-2, and Sentinel-1).

Keywords: Generalization bounds, spatial dependence, sample complexity, spatial autocorrelation

## I. INTRODUCTION

A persistent methodological issue in remote sensing is spatial autocorrelation, namely the mismatch between the number of labeled pixels available and the amount of statistically independent information they contain. An $n \times n$ satellite image nominally provides $n ^ { 2 }$ labeled samples, but neighboring pixels are spatially correlated. Adjacent forest pixels look alike, adjacent urban pixels look alike, because a scene is composed of internally homogeneous objects (fields, forest stands) whose size and spacing set the distance over which correlation persists [1]. Standard learning-theoretic guarantees, that is, high-probability bounds on how far a classifier’s test error can depart from its training error, assume independent and identically distributed samples and therefore do not apply.

In practice, this manifests as a well-documented bias. Models evaluated by random holdout on spatially autocorrelated imagery report higher accuracy than models evaluated by spatially structured holdout [2, 3, 4, 5]. The recognition that covariance structure differs across strata, and that treating a scene as globally homogeneous is inadequate, has a long history in geostatistics [6]. Studies of sampling design under spatial dependence have shown empirically that prediction accuracy degrades sharply once the sampling interval approaches the autocorrelation range [7]. Concurrently, Liao et al. [8] have established an empirical scaling law relating sample size to classifier performance in remote sensing. Our results complement theirs by providing a theoretical explanation for why the effective sample size entering such scaling relationships is $\Theta ( n ^ { 2 } / r ^ { 2 } )$ rather than the nominal pixel count. The remote sensing community has converged on the recommendation to use spatial cross-validation, but a characterization of the cost of ignoring spatial structure, how much independent information an image truly contains, and a proof that the standard spatial CV prescription is optimal has remained open. Meanwhile, the statistical learning theory literature has developed generalization bounds for dependent data under mixing conditions [9, 10, 11], but almost exclusively for temporally indexed (one-dimensional) sequences.

We bridge this gap. Our contributions are a finite-sample generalization bound for classifiers on a two-dimensional $\beta -$ mixing random field, with an explicit dependence on the spatial correlation range r, a matching minimax lower bound showing the rate is tight up to logarithmic factors, and a corollary providing theoretical justification for spatial crossvalidation in remote sensing and quantifying the overoptimism of random holdout. For a typical Sentinel-2 scene with correlation range 30 pixels, the effective sample size may be much smaller than the pixel count, meaning reported confidence intervals could be too narrow.

## II. SETUP

Let $\{ ( X _ { s } , Y _ { s } ) \} _ { s \in \mathbb { Z } ^ { 2 } }$ be a strictly stationary random field, where $X _ { s } \in \mathcal { X } \subseteq \mathbb { R } ^ { d }$ is a feature vector (e.g. reflectance) and $Y _ { s } \in \{ 1 , \ldots , K \}$ is a land cover label. We observe this field on the lattice $\Lambda _ { n } = \{ 1 , \ldots , n \} ^ { 2 }$

Definition 1 (Spatial β-mixing with latent alignment). Let τ be a random variable on a finite set T (possibly a single point, in which case all conditioning below is vacuous<sup>2</sup>) such that, conditional on $\tau \ = \ t ,$ the single-site law of $( X _ { s } , Y _ { s } )$ is the same for all s and t. For finite $S , T \subset \mathbb { Z } ^ { 2 }$ with $\begin{array} { r } { d _ { \infty } ( S , T ) \ = \ \operatorname* { m i n } _ { s \in S , t ^ { \prime } \in T } \| s - t ^ { \prime } \| _ { \infty } \ \geq \ g , } \end{array}$ let $\beta ( \sigma _ { S } , \sigma _ { T } \ )$ $\begin{array} { r } { \tau = t ) = \mathbb { E } \big [ \operatorname* { s u p } _ { B \in \sigma _ { T } } \vert P ( B \mid \sigma _ { S } , \tau = t ) - P ( B \mid \tau = t ) \vert \big \vert \tau = t \big \vert } \end{array}$ where $\sigma _ { S } \overset { \cdot } { = } \sigma ( \{ ( \boldsymbol { X } _ { s } , Y _ { s } ) : s \in S \} )$ , and set

$$
\beta ( g ) = \underset { t \in \mathcal { T } } { \operatorname* { s u p } } \ \underset { S , T \ f n i t e } { \operatorname* { s u p } } \ \beta ( \sigma _ { S } , \sigma _ { T } \mid \tau = t ) .\tag{1}
$$

The mixing coefficient measures how close to independent two regions of the image become as you move them apart. For degenerate τ this is the standard spatial β-mixing coefficient. The alignment τ models a global nuisance given which the field mixes (for instance, the unknown phase of the object grid like field boundaries or forest stands relative to the pixel lattice). No analyst knows where scene objects fall relative to the sensor grid, so this is a natural feature of remote sensing data rather than a technical artifact. The single-site condition ensures the risk defined below is independent of s and τ.

Let H be a hypothesis class of classifiers $h \colon { \mathcal { X } } \ \to$ $\{ 1 , \ldots , K \}$ with VC dimension $d _ { \mathrm { v c } } < \infty$ . Define the 0–1 loss $\ell ( h , s ) = \dot { \bf 1 } [ h ( X _ { s } ) \neq Y _ { s } ]$ and the true risk $R ( h ) = \mathbb { E } [ \ell ( h , s ) ]$ For a stationary Gaussian random field with covariance $C ( { \bf h } ) \ = \ \sigma ^ { 2 } e ^ { - \| { \bf h } \| / r }$ (exponential model with range r) and spectral density bounded away from zero, we have $\beta ( g ) \le$ $c _ { 0 } e ^ { - c _ { 1 } g / r }$ for constants $c _ { 0 } , c _ { 1 } > 0 [ 1 2 , 1 3 ]$ , with degenerate τ . For the Matern model with smoothness´ ν, polynomial mixing rates $\beta ( g ) \sim g ^ { - \alpha }$ hold with α depending on $\nu$ and the spatial dimension.

## III. UPPER BOUND

The challenge is that standard generalization bounds assume independent samples, yet nearby pixels in a satellite image are correlated. Our proof strategy converts the dependent-data problem into an independent-data problem at a controlled cost. We partition the $n \times n$ image into small blocks separated by gaps of width $g \ ( \mathrm { F i g . \ 1 } )$ . If $g$ is large enough relative to the correlation range r, the blocks are nearly independent, and we can apply standard concentration inequalities as if the blocks were i.i.d. samples. The cost is that the gaps consume pixels and an image with $n ^ { 2 }$ pixels yields only $\bar { m } \approx n ^ { 2 } / g ^ { 2 }$ usable blocks. The proof optimizes this tradeoff where wider gaps give better independence but fewer blocks and shows that the optimal gap is $g = \Theta ( r \log n )$ , yielding $m = \Theta ( n ^ { 2 } / r ^ { 2 } )$ effectively independent blocks up to logarithmic factors.

Formally, the argument has three components: a coupling lemma that bounds the total variation distance between the dependent blocks and an i.i.d. surrogate (Lemma 1), a complexity bound on the block-averaged loss class (Lemma 2), and an optimization over the block geometry.

Fix integers $a \ge 1$ (block side length) and $g \geq 1$ (gap width). Set $p = a + g$ . Define non-overlapping blocks

$$
B _ { i j } = \left\{ \left( i p + 1 , \ldots , i p + a \right) \times \left( j p + 1 , \ldots , j p + a \right) \right\}\tag{2}
$$

for $i , j = 0 , \ldots , M { \ - } 1$ , where $M = \lfloor n / p \rfloor$ . This yields $m =$ $M ^ { 2 }$ blocks, each containing $a ^ { 2 }$ pixels, with any two distinct

![](images/d837ae8e5cab895d6c19941999e4387c858d4290cdfddd1c42254a4b2d2cd766.jpg)  
Fig. 1. Graphical illustration of the blocking construction in Thm. 1.

blocks separated by $d _ { \infty } ( B _ { i j } , B _ { i ^ { \prime } j ^ { \prime } } ) \ \geq \ g$ . Define the block empirical risk

$$
\hat { R } _ { \mathrm { b l o c k } } ( h ) = \frac { 1 } { m } \sum _ { i , j = 0 } ^ { M - 1 } Z _ { i j } ( h ) , \quad Z _ { i j } ( h ) = \frac { 1 } { a ^ { 2 } } \sum _ { s \in B _ { i j } } \ell ( h , s ) .\tag{3}
$$

By stationarity, $\mathbb { E } [ \hat { R } _ { \mathrm { b l o c k } } ( h ) ] = R ( h )$ . The standard telescoping argument [14] extends directly to the 2D setting:

Lemma 1 (Coupling). Conditionally on $\tau = t ,$ for every $t \in$ $\tau ,$ there exist i.i.d. random variables $\{ Z _ { i j } ^ { \prime } \}$ with $Z _ { i j } ^ { \prime } \ { \overset { d } { = } } \ Z _ { i j }$ such that

$$
\begin{array} { r } { d _ { \mathrm { T V } } \big ( \mathcal { L } ( Z _ { i j } : i , j ) , \ \prod _ { i , j } \mathcal { L } ( Z _ { i j } ) \big ) \leq ( m - 1 ) \beta ( g ) . } \end{array}\tag{4}
$$

Proof. Enumerate the blocks as $Z _ { 1 } , \dots , Z _ { m } ;$ all laws below are conditional on $\tau \ = \ t .$ By the triangle inequality, $\begin{array} { r } { d _ { \mathrm { T V } } ( \mu _ { 0 } , \mu _ { m } ) \ \leq \ \sum _ { k = 2 } ^ { m } d _ { \mathrm { T V } } ( \mathcal { L } ( Z _ { k } \ | \ Z _ { 1 } , \ldots , Z _ { k - 1 } ) , \mathcal { L } ( Z _ { k } ) ) } \end{array}$ Each $Z _ { k }$ is $\sigma _ { B _ { k } }$ -measurable, $\cup _ { j < k } B _ { j }$ is finite at $\ell _ { \infty } { \tt - d i s t a n c e }$ $\geq g$ from $B _ { k }$ , so each term is $\underline { { \underline { { \mathbf { \Pi } } } } } \le \mathbf { \dot { \rho } } \beta ( g )$ □

Lemma 2 (Complexity of Block-Averaged Class). The pseudo-dimension of $\begin{array} { r } { \mathcal { F } = \{ ( x _ { s } , y _ { s } ) _ { s \in B } \mapsto \frac { 1 } { a ^ { 2 } } \sum _ { s \in B } \ell ( h , s ) } \end{array}$ $h \in \mathcal H \}$ satisfies Pdim $( \mathcal { F } ) \leq 2 d _ { \mathrm { v c } } \log _ { 2 } ( e a ^ { 2 } )$

Proof. Shattering k block-samples with thresholds requires all $2 ^ { k }$ sign patterns of $\begin{array} { r } { \frac { 1 } { a ^ { 2 } } \sum _ { s } \ell ( h , s ) - t _ { i } \colon } \end{array}$ ; each h induces a binary labeling of $k a ^ { 2 }$ pixels, so $2 ^ { k } \leq S _ { \mathcal { H } } ( k a ^ { 2 } ) \leq ( e k a ^ { 2 } / d _ { \mathrm { v c } } ) ^ { d _ { \mathrm { v c } } }$ by Sauer’s lemma. Then $k \leq d _ { \mathrm { v c } } \log _ { 2 } ( e k a ^ { 2 } / d _ { \mathrm { v c } } )$ , and $x \le$ $A \log _ { 2 } ( B x ) \Rightarrow x \leq 2 A \log _ { 2 } ( A B )$ for $A , B \geq 1$ gives the claim. □

Remark 1 (On generality). We emphasize that the correlation range r and the number of pixels $n ^ { 2 }$ enter the effective sample size $n _ { \mathrm { e f f } } = \Theta ( n ^ { 2 } / r ^ { 2 } )$ independently of the classifier. The hypothesis class affects the bound only through its VC dimension $d _ { \mathrm { v c } } ,$ which multiplies the deviation as $\sqrt { d _ { \mathrm { v c } } }$ but does not change the scaling in r. The rate is therefore a property of the spatial dependence structure, not of any particular learning algorithm.

Theorem 1 (Upper bound). Let $\big \{ ( X _ { s } , Y _ { s } ) \big \}$ be a strictly stationary random field on $\Lambda _ { n } ~ = ~ \{ 1 , \ldots , n \} ^ { 2 }$ satisfying Definition 1 with $\beta ( g ) \le c _ { 0 } e ^ { - c _ { 1 } g / r }$ for constants $c _ { 0 } , c _ { 1 } > 0$ and correlation range $r \geq 1$ . Let H have VC dimension $d _ { \mathrm { v c } } .$ For any $\delta \in ( 0 , 1 )$ , choosing block side length a and gap g optimally, with probability at least $1 - \delta \cdot$

$$
\operatorname* { s u p } _ { h \in \mathcal { H } } \bigl | R ( h ) - \hat { R } _ { \mathrm { b l o c k } } ( h ) \bigr | \leq C _ { 1 } \frac { r \log ( n / \delta ) } { n } \sqrt { d _ { \mathrm { v c } } \log n } ,\tag{5}
$$

where $C _ { 1 }$ depends only on c<sub>0</sub> and $c _ { 1 } .$ . Equivalently, the effective sample size is $n _ { \mathrm { e f f } } = \Theta \bigl ( n ^ { 2 } / ( r ^ { 2 } \log ^ { 2 } n ) \bigr )$

Proof. Fix $t \in \tau$ and argue conditionally on $\tau = t$ . Combining Lemmas 1 and 2 with the standard pseudo-dimension bound for i.i.d. bounded variables:

$$
\begin{array} { r l } & { P \bigg ( \underset { h \in \mathcal { H } } { \operatorname* { s u p } } | R ( h ) - \hat { R } _ { \mathrm { b l o c k } } ( h ) | > \epsilon \bigg | \tau = t \bigg ) } \\ & { \quad \le 8 \bigg ( \displaystyle \frac { 4 e m } { d ^ { \prime } } \bigg ) ^ { d ^ { \prime } } \exp \bigg ( - \displaystyle \frac { m \epsilon ^ { 2 } } { 3 2 } \bigg ) + ( m - 1 ) \beta ( g ) } \end{array}\tag{6}
$$

where $m = \lfloor n / ( a + g ) \rfloor ^ { 2 }$ and $d ^ { \prime } \leq 2 d _ { \mathrm { v c } } \log _ { 2 } ( e a ^ { 2 } )$

Optimization. The coupling constraint $( m - 1 ) \beta ( g ) \leq \delta / 2$ requires $\begin{array} { r } { g \ge \frac { r } { c _ { 1 } } \log \frac { 2 c _ { 0 } m } { \delta } = \Theta ( r \log ( n / \delta ) ) } \end{array}$ ; the concentration term scales as $\sqrt { d ^ { \prime } \log m / m }$ . We set $a \ = \ 1$ : single-pixel blocks may seem counterintuitive, since averaging over larger blocks reduces each block’s variance, but larger blocks also inflate the pseudo-dimension (Lemma 2) and reduce the block count for a given gap, and the variance reduction cannot compensate for these two costs. With $a = 1 , d ^ { \prime } = d _ { \mathrm { v c } }$ and $m \overset { \cdot } { = } \Theta ( n ^ { 2 } / ( r ^ { 2 } \log ^ { 2 } ( n / \delta ) ) )$ is maximized.

With $a = 1$ , each $Z _ { i j } ( h ) = \ell ( h , s _ { i j } )$ is a single-site loss, whose conditional law is independent of t with mean $R ( h )$ by the single-site condition of Definition 1. Hence the coupled surrogates are conditionally i.i.d. with mean $R ( h )$ , the right side of (6) is uniform in t, and averaging over τ preserves the bound. Setting (6) equal to δ yields the result. □

## IV. LOWER BOUND

To show that no algorithm can beat the $\Theta ( n ^ { 2 } / r ^ { 2 } )$ effective sample size, we construct a worst-case random field that is stationary, satisfies the mixing condition, and yet contains $\lfloor n / r \rfloor ^ { 2 }$ independent pieces of information. The construction tiles the image with $\boldsymbol { r } \times \boldsymbol { r } ^ { \mathrm { ~ \scriptsize ~ . ~ } }$ “super-pixels,” each carrying an independent draw from a hard classification instance. All pixels within a super-pixel share the same feature and label, so observing the full $n ^ { 2 }$ pixels is no more informative than observing the $\lfloor n / r \rfloor ^ { 2 }$ super-pixel summaries. A random spatial offset τ ensures strict stationarity, and conditionally on $\tau$ the field is block-independent, so the construction satisfies Definition 1 with τ as the latent alignment, the offset is the unknown grid phase that the definition models. Applying Assouad’s lemma to the resulting i.i.d. super-pixel problem gives a minimax lower bound of $\Omega ( r \sqrt { d _ { \mathrm { v c } } } / n )$ , matching the upper bound up to logarithmic factors.

Theorem 2 (Minimax lower bound). Let H have VC dimension $d _ { \mathrm { v c } } \geq 1$ . For any $r \geq 1$ and $n \geq 2 r \sqrt { d _ { \mathrm { v c } } } ,$ there exists a strictly stationary random field on $\Lambda _ { n }$ satisfying Definition 1 with $\beta ( g ) = 0 f o r$ all $g \geq r$ hence lying in the model class of Theorem 1 such that

$$
\operatorname* { i n f } _ { \hat { h } } \operatorname* { s u p } _ { P } \mathbb { E } \big [ R ( \hat { h } ) - R ( h _ { P } ^ { * } ) \big ] \geq \frac { 1 } { 1 2 } \frac { r \sqrt { d _ { \mathrm { v c } } } } { n }\tag{7}
$$

where the infimum is over all learning algorithms and the supremum is over all distributions within the model class.

Proof. Since $\mathrm { V C } ( \mathcal { H } ) = d _ { \mathrm { v c } }$ , there exist points $z _ { 1 } , \ldots , z _ { d _ { \mathrm { v c } } } \in$ X shattered by H.

Step 1: Construction. Let τ be uniform on $\{ 0 , \ldots , r { - } 1 \} ^ { 2 }$ independent of all other randomness. Define super-pixel assignments $\kappa ( s ) ~ = ~ \lfloor ( s - \tau ) / r \rfloor$ . For each super-pixel k, draw independently $J _ { k } \sim$ Uniform $\{ 1 , \ldots , d _ { \mathrm { v c } } \}$ and $\xi _ { k } \sim$ Bernoulli(η), and set $X _ { s } = z _ { J _ { \kappa ( s ) } } , Y _ { s } = c _ { J _ { \kappa ( s ) } } \oplus \xi _ { \kappa ( s ) }$ , where $c \in \{ 0 , 1 \} ^ { d _ { \mathrm { v c } } }$ is the unknown concept.

Step 2: Stationarity. Shifting s by $v \in \mathbb { Z } ^ { 2 }$ is equivalent to shifting τ by −v; since τ is uniform on a period-r grid, the field is strictly stationary.

Step 3: Mixing. Condition on $\tau = t ,$ , so $\kappa _ { t } ( s ) ~ = ~ \lfloor ( s ~ -$ $t ) / r ]$ is deterministic. The conditional single-site law is that of $( z _ { J } , c _ { J } \oplus \xi )$ with J ∼ Uniform $\{ 1 , \ldots , d _ { \mathrm { v c } } \} , \ \xi \ \sim$ Bernoulli(η), independent of s and t. If $d _ { \infty } ( S , T ) \geq r .$ , the index sets $\kappa _ { t } ( S )$ and $\kappa _ { t } ( T )$ are disjoint, and since the marks $\left( J _ { k } , \xi _ { k } \right)$ are i.i.d. across super-pixels, $\beta ( g \mid \tau = t ) = 0$ for $g \geq r$ , uniformly in t. (Unconditionally the field is not mixing since observing one region reveals the grid phase, which is informative about joint laws in distant regions. Conditioning on τ removes this nuisance.)

Step 4: Sufficient statistic. Reveal τ to the learner; this only strengthens it, so any lower bound survives. Conditionally on $\tau ,$ the data reduce to $N \geq \lfloor n / r \rfloor ^ { 2 }$ i.i.d. pairs $( J _ { k } , Y _ { k } )$ , one per super-pixel, and the reduction does not depend on c.

Step 5: Assouad’s lemma. Work conditionally on τ; the excess risk is τ -free by Step 3, so the bound below is uniform in τ and averaging gives the theorem. The $2 ^ { d _ { \mathrm { v c } } }$ concepts form a hypercube. Flipping bit $j$ changes the excess risk by $\alpha _ { j } = ( 1 - 2 \eta ) / ( 2 d _ { \mathrm { v c } } )$ . The number of super-pixels informative about bit $j$ is $N _ { j } \sim$ Binomial $( N , 1 / d _ { \mathrm { v c } } )$ , and by Pinsker, $\mathrm { T V } _ { j } \leq \sqrt { N _ { j } \mathrm { K L } ( \eta \| 1 - \eta ) / 2 }$ . Set $\eta = 1 / 2 - \gamma$ with $\begin{array} { r } { \gamma = \frac { 1 } { 6 } \sqrt { d _ { \mathrm { v c } } / N } \doteq 1 / 1 2 } \end{array}$ (as $N \geq 4 d _ { \mathrm { v c } } )$ , so $\mathrm { K L } ( \eta \vert \vert 1 { - } \eta ) ~ \leq$ $\begin{array} { r } { \frac { \left( 1 - 2 \eta \right) ^ { 2 } } { \eta \left( 1 - \eta \right) } \ \leq \ 1 7 \gamma ^ { 2 } } \end{array}$ . Since $\mathrm { T V } _ { j } ^ { 2 } ~ \leq ~ 1 7 \gamma ^ { 2 } N _ { j } / 2$ given $N _ { j }$ and $\sqrt { \cdot }$ is concave, Jensen gives $\mathbb { E } [ \mathrm { T V } _ { j } ] \le \sqrt { 1 7 \gamma ^ { 2 } \mathbb { E } [ N _ { j } ] / 2 } =$ $\sqrt { 1 7 \gamma ^ { 2 } N / ( 2 d _ { \mathrm { v c } } ) } = \sqrt { 1 7 / 7 2 } < 1 / 2$ . The expectation form of Assouad’s lemma [15] then gives

$$
\begin{array} { r } { \operatorname* { i n f } _ { \hat { h } } \operatorname* { m a x } _ { c } \mathbb { E } [ R ( \hat { h } ) - R ( h _ { c } ^ { * } ) ] \geq d _ { \mathrm { v c } } \alpha _ { j } \cdot \frac { 1 } { 2 } = \frac { \gamma } { 2 } = \frac { 1 } { 1 2 } \sqrt { \frac { d _ { \mathrm { v c } } } { N } } , } \end{array}\tag{8}
$$

and $N \leq n ^ { 2 } / r ^ { 2 }$ completes the proof.

Comparing Theorems 1 and 2, the upper bound is $\tilde { O } ( r \sqrt { d _ { \mathrm { v c } } } / n )$ and the lower bound is $\Omega ( r \sqrt { d _ { \mathrm { v c } } } / n )$ : the rates match and the gap is polylogarithmic.

## V. EXTENSIONS TO NON-IDEAL SETTINGS

The results of Sections III–IV assume an isotropic, globally stationary random field. We now relax both. Section V-A

extends the bounds to anisotropic fields with directional correlation, and Section V-B extends them to piecewise stationary fields with spatially varying correlation structure.

## A. Anisotropic Fields

Many remote sensing modalities exhibit directional correlation structure. SAR images have different correlation lengths in range and azimuth, and landscapes shaped by rivers, coastlines, or mountain ridges are naturally anisotropic. We replace the isotropic exponential covariance with

$$
C ( \mathbf { h } ) = \sigma ^ { 2 } \exp \Bigl ( - \sqrt { h _ { 1 } ^ { 2 } / r _ { 1 } ^ { 2 } + h _ { 2 } ^ { 2 } / r _ { 2 } ^ { 2 } } \Bigr ) ,\tag{9}
$$

where $r _ { 1 } , r _ { 2 } \ge 1$ are the directional correlation ranges. The corresponding mixing coefficient satisfies

$$
\beta ( g _ { 1 } , g _ { 2 } ) \leq c _ { 0 } \exp \Bigl ( - c _ { 1 } \sqrt { g _ { 1 } ^ { 2 } / r _ { 1 } ^ { 2 } + g _ { 2 } ^ { 2 } / r _ { 2 } ^ { 2 } } \Bigr ) ,\tag{10}
$$

where $g _ { k }$ denotes the separation between two regions in direction k.

The blocking construction generalizes by using rectangular blocks and gaps. Fix unit block size $a _ { 1 } ~ = ~ a _ { 2 } ~ = ~ 1$ and directional gaps $g _ { k } \geq 1$ . Define blocks

$$
B _ { i j } = \{ i ( 1 + g _ { 1 } ) + 1 \} \times \{ j ( 1 + g _ { 2 } ) + 1 \}\tag{11}
$$

for $i = 0 , \ldots , M _ { 1 } - 1$ and $j = 0 , \ldots , M _ { 2 } { - } 1$ , where $M _ { k } \ =$ $\lfloor n / ( 1 + g _ { k } ) \rfloor$ . This yields $m = M _ { 1 } M _ { 2 }$ single-pixel blocks, any two of which are separated by at least $( g _ { 1 } , g _ { 2 } )$ in the respective directions.

Theorem 3 (Anisotropic upper bound). Let $\big \{ ( X _ { s } , Y _ { s } ) \big \}$ be a stationary random field on $\Lambda _ { n }$ satisfying Definition 1 with directional rate (10) and directional correlation ranges $r _ { 1 } , r _ { 2 } \ge 1$ . Let H have VC dimension $d _ { \mathrm { v c } } .$ For any $\delta \in ( 0 , 1 )$ with probability at least $1 - \delta \colon$

$$
\operatorname* { s u p } _ { h \in \mathcal H } \left| R ( h ) - \hat { R } _ { \mathrm { b l o c k } } ( h ) \right| \leq C _ { 1 } \frac { \sqrt { r _ { 1 } r _ { 2 } } \log ( n / \delta ) } { n } \sqrt { d _ { \mathrm { v c } } \log n } ,\tag{12}
$$

where $C _ { 1 }$ depends only on $c _ { 0 } , c _ { 1 }$ . The effective sample size is $n _ { \mathrm { e f f } } = \Theta \big ( n ^ { 2 } / ( r _ { 1 } r _ { 2 } ) \big )$ up to logarithmic factors.

Proof. The coupling lemma (Lemma 1) applies with the same telescoping argument where each pair of blocks is separated by $( g _ { 1 } , g _ { 2 } )$ , so each coupling step incurs error at most $\beta ( g _ { 1 } , g _ { 2 } )$ The constraint $( m - 1 ) \beta ( g _ { 1 } , g _ { 2 } ) \leq \delta / 2$ is satisfied by setting $g _ { k } \ = \ \Theta ( r _ { k } \log ( n / \delta ) )$ for each k. Since $a _ { 1 } = a _ { 2 } = 1$ , the pseudo-dimension bound (Lemma 2) gives $d ^ { \prime } \ = \ d _ { \mathrm { v c } }$ . The number of blocks is

$$
m = \Big \lfloor \frac { n } { 1 + g _ { 1 } } \Big \rfloor \Big \lfloor \frac { n } { 1 + g _ { 2 } } \Big \rfloor = \Theta \Big ( \frac { n ^ { 2 } } { r _ { 1 } r _ { 2 } \log ^ { 2 } ( n / \delta ) } \Big ) .\tag{13}
$$

Substituting into (6) yields the stated bound. For nondegenerate τ, condition as in Theorem 1. Since $a _ { 1 } = a _ { 2 } = 1$ , the argument is identical. □

Theorem 4 (Anisotropic lower bound). Under the conditions of Theorem 2 with directional ranges $r _ { 1 } , r _ { 2 } \ \geq \ 1$ and $n \geq$ $2 \operatorname* { m a x } ( r _ { 1 } , r _ { 2 } ) \sqrt { d _ { \mathrm { v c } } } .$

$$
\operatorname* { i n f } _ { \hat { h } } \operatorname* { s u p } _ { P } \mathbb { E } \big [ R ( \hat { h } ) - R ( h _ { P } ^ { \ast } ) \big ] \geq \frac { 1 } { 1 2 } \frac { \sqrt { r _ { 1 } r _ { 2 } d _ { \mathrm { v c } } } } { n } .\tag{14}
$$

Proof. Modify the construction of Theorem 2 by using rectangular super-pixels. Let $\tau$ be uniform on $\{ 0 , \ldots , r _ { 1 } - 1 \} \ \times$ $\{ 0 , \ldots , r _ { 2 } { - } 1 \}$ and define super-pixel assignments $\kappa ( s ) \ =$ $( \lfloor ( s _ { 1 } - \tau _ { 1 } ) / r _ { 1 } \rfloor , \lfloor ( s _ { 2 } - \tau _ { 2 } ) / r _ { 2 } \rfloor )$ . Each super-pixel has size $r _ { 1 } \times r _ { 2 }$ . Stationarity follows from the uniformity of τ (shifting by v is equivalent to shifting $\tau \ \mathsf { b y } \ - v$ , which preserves the distribution since $\tau _ { k }$ is uniform on a period-r<sub>k</sub> grid). Conditionally on τ, super-pixel indices of S and T are disjoint whenever $d _ { \infty } ( S , T ) \geq \operatorname* { m a x } ( r _ { 1 } , r _ { 2 } )$ , giving $\beta ( g \mid \tau ) = 0$ for $g ~ \ge ~ \operatorname* { m a x } ( r _ { 1 } , r _ { 2 } )$ uniformly in $\tau ;$ the learner is augmented with $\tau$ as in Step 4 of Theorem 2. The number of superpixels is $N ~ = ~ \lfloor n / r _ { 1 } \rfloor \cdot \lfloor n / r _ { 2 } \rfloor ~ \leq ~ n ^ { 2 } / ( r _ { 1 } r _ { 2 } )$ . The Assouad argument proceeds identically, yielding $\Omega ( \sqrt { d _ { \mathrm { v c } } / N } ) =$ $\Omega ( \sqrt { r _ { 1 } r _ { 2 } d _ { \mathrm { v c } } } / n )$ □

Theorems 3 and 4 together establish that $\begin{array} { r l } { n _ { \mathrm { e f f } } } & { { } = } \end{array}$ $\Theta ( n ^ { 2 } / ( r _ { 1 } r _ { 2 } ) )$ for anisotropic fields, matching up to polylogarithmic factors. The isotropic result is recovered when $r _ { 1 } = r _ { 2 } = r$

The practical implication for spatial cross-validation is that block separation should be direction-dependent where folds should be separated by $g _ { k } = \Theta ( r _ { k } \log n )$ in direction $k ,$ leading to elongated validation blocks aligned with the anisotropy. For SAR imagery, where the range direction may have $r _ { 1 } = 5 0$ pixels and the azimuth $r _ { 2 } = 1 0 ,$ , this yields $n _ { \mathrm { e f f } } = n ^ { 2 } / 5 0 0$ rather than the isotropic estimate $n ^ { 2 } / 2 5 0 0$ obtained by using $r _ { \operatorname* { m a x } } = 5 0$ in both directions.

## B. Piecewise Stationary Fields

Real remote sensing scenes are rarely globally stationary, and correlation structure varies across land cover types, and the range fitted over cropland may differ substantially from that over forest or urban areas. We extend the theory to fields that are stationary within each of K homogeneous regions.

Definition 2 (Piecewise stationary field). A random field $\big \{ ( X _ { s } , Y _ { s } ) \big \}$ on $\Lambda _ { n }$ is $( K , \{ r _ { k } \} )$ -piecewise stationary if there exists a partition $\textstyle \Lambda _ { n } = \bigcup _ { k = 1 } ^ { K } \Lambda _ { k }$ such that, within each region $\Lambda _ { k }$ (containing $| \Lambda _ { k } | = n _ { k } ^ { 2 } \ p i x e l s )$ , the field is stationary and satisfies Definition 1 (alignment $\tau _ { k } )$ with $\beta _ { k } ( g ) \le c _ { 0 } e ^ { - c _ { 1 } g / r _ { k } }$ and, conditionally on the tuple $\tau = ( \tau _ { 1 } , \dots , \tau _ { K } )$ , the restrictions of the field to distinct regions are mutually independent.

Without global stationarity, the expected loss $\mathbb { E } [ \ell ( h , s ) ]$ depends on s. The natural target becomes the domain-average risk:

$$
\bar { R } ( h ) = \frac { 1 } { n ^ { 2 } } \sum _ { s \in \Lambda _ { n } } \mathbb { E } [ \ell ( h , s ) ] = \sum _ { k = 1 } ^ { K } w _ { k } R _ { k } ( h ) ,\tag{15}
$$

where $w _ { k } = n _ { k } ^ { 2 } / n ^ { 2 }$ and $R _ { k } ( h ) = \mathbb { E } _ { s \sim \Lambda _ { k } } [ \ell ( h , s ) ]$ is the withinregion risk.

Within each region $\Lambda _ { k } ,$ , we construct blocks with gap $g _ { k } \ = \ \Theta ( r _ { k } \log ( n / \delta ) )$ , yielding $m _ { k } ~ = ~ \Theta ( n _ { k } ^ { 2 } / ( r _ { k } ^ { 2 } \log ^ { 2 } \stackrel { . } { n } ) )$ blocks. The per-region block empirical risk is $\hat { R } _ { k } ^ { \mathrm { b i } \mathrm { { o c k } } } ( h ) =$ $\begin{array} { r } { \frac { 1 } { m _ { k } } \sum _ { ( i , j ) \in \Lambda _ { k } } Z _ { i j } ^ { ( k ) } ( h ) } \end{array}$ , and the overall estimator is

$$
\hat { \bar { R } } ( h ) = \sum _ { k = 1 } ^ { K } w _ { k } \hat { R } _ { k } ^ { \mathrm { b l o c k } } ( h ) .\tag{16}
$$

By per-region stationarity, $\mathbb { E } [ \hat { \bar { R } } ( h ) ] = \bar { R } ( h )$

Theorem 5 (Piecewise stationary upper bound). $L e t$ $\big \{ ( X _ { s } , Y _ { s } ) \big \}$ be a $\left( K , \left\{ r _ { k } \right\} \right)$ )-piecewise stationary field on $\Lambda _ { n } ,$ with each region $\Lambda _ { k }$ containing $n _ { k } ^ { 2 }$ pixels and having correlation range $r _ { k } \ge 1$ . Let H have VC dimension $d _ { \mathrm { v c } } .$ . For any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta \colon$

$$
\operatorname* { s u p } _ { h \in \mathcal { H } } \left. \bar { R } ( h ) - \hat { \bar { R } } ( h ) \right. \leq C _ { 2 } \frac { \log ( n / \delta ) } { n ^ { 2 } } \sqrt { d _ { \mathrm { v c } } \log n \cdot \sum _ { k = 1 } ^ { K } n _ { k } ^ { 2 } r _ { k } ^ { 2 } } ,\tag{17}
$$

where $C _ { 2 }$ depends only on $c _ { 0 } , c _ { 1 }$ . The effective sample size is

$$
n _ { \mathrm { e f f } } = \frac { n ^ { 4 } } { \sum _ { k = 1 } ^ { K } n _ { k } ^ { 2 } r _ { k } ^ { 2 } }\tag{18}
$$

up to logarithmic factors.

Proof. Within each region $\Lambda _ { k } .$ , apply the coupling lemma (Lemma 1) with gap $g _ { k } \ = \ \Theta ( r _ { k } \log ( n K / \delta ) )$ to replace the $m _ { k }$ dependent block variables $\{ Z _ { i j } ^ { ( k ) } \}$ with independent copies. The total coupling error across all regions is at most $\begin{array} { r } { \sum _ { k } ( m _ { k } - 1 ) \beta _ { k } ( g _ { k } ) \le \delta / 2 } \end{array}$ by setting the gaps appropriately.

After coupling, $\hat { \bar { \cal R } } ( h ) - \bar { \cal R } ( h )$ is a weighted sum of independent bounded random variables. Specifically, for fixed h:

$$
\hat { \bar { R } } ( h ) - \bar { R } ( h ) = \sum _ { k = 1 } ^ { K } \frac { w _ { k } } { m _ { k } } \sum _ { ( i , j ) \in \Lambda _ { k } } \big ( Z _ { i j } ^ { ( k ) } ( h ) - R _ { k } ( h ) \big ) .\tag{19}
$$

Each term $w _ { k } Z _ { i j } ^ { ( k ) } / m _ { k }$ is bounded in $[ 0 , w _ { k } / m _ { k } ]$ . By Hoeffding’s inequality for independent (not necessarily identically distributed) bounded random variables:

$$
P \big ( | \hat { \bar { R } } ( h ) - \bar { R } ( h ) | > \epsilon \big ) \leq 2 \mathrm { e x p } \left( - \frac { 2 \epsilon ^ { 2 } } { \sum _ { k = 1 } ^ { K } w _ { k } ^ { 2 } / m _ { k } } \right)\tag{20}
$$

For uniform convergence over ${ \mathcal { H } } ,$ we use the standard symmetrization and covering number argument, which only requires independence and boundedness. Since $a = 1$ (unit blocks), the loss class has pseudo-dimension $d ^ { \prime } \ = \ d _ { \mathrm { v c } }$ by Lemma 2. The total number of blocks is $\begin{array} { r } { \begin{array} { r c l } { m } & { = } & { \sum _ { k } m _ { k } } \end{array} } \end{array}$ and the covering number of the VC class at scale ϵ satisfies $\mathcal { N } ( \epsilon ) \leq ( e m / d _ { \mathrm { v c } } ^ { - } ) ^ { d _ { \mathrm { v c } } }$ . Combining via a union bound over an $\epsilon / 2 \cdot$ -cover:

$$
\begin{array} { r l } & { P \biggl ( \underset { h \in \mathcal { H } } { \operatorname* { s u p } } | \bar { R } ( h ) - \hat { \bar { R } } ( h ) | > \epsilon \biggr ) } \\ & { \quad \le 2 \biggl ( \displaystyle \frac { 2 e m } { d _ { \mathrm { v c } } } \biggr ) ^ { d _ { \mathrm { v c } } } \exp \biggl ( - \frac { \epsilon ^ { 2 } / 2 } { \sum _ { k } w _ { k } ^ { 2 } / m _ { k } } \biggr ) + \frac { \delta } { 2 } . } \end{array}\tag{21}
$$

Substituting $w _ { k } = n _ { k } ^ { 2 } / n ^ { 2 }$ and $m _ { k } = \Theta ( n _ { k } ^ { 2 } / ( r _ { k } ^ { 2 } \log ^ { 2 } n ) )$ :

$$
\sum _ { k = 1 } ^ { K } \frac { w _ { k } ^ { 2 } } { m _ { k } } = \sum _ { k = 1 } ^ { K } \frac { n _ { k } ^ { 4 } } { n ^ { 4 } } \cdot \frac { r _ { k } ^ { 2 } \log ^ { 2 } n } { n _ { k } ^ { 2 } } = \frac { \log ^ { 2 } n } { n ^ { 4 } } \sum _ { k = 1 } ^ { K } n _ { k } ^ { 2 } r _ { k } ^ { 2 } .\tag{22}
$$

Setting (21) equal to $\delta$ and solving for ϵ gives the stated bound. □

The effective sample size (18) has a natural interpretation. It is the reciprocal of the weighted variance $\textstyle \sum _ { k } w _ { k } ^ { 2 } / m _ { k }$ . Regions with large correlation range $r _ { k }$ contribute fewer effective

samples per unit area and thereby reduce the overall effective sample size, even if they occupy a small fraction of the image.

Theorem 6 (Piecewise stationary lower bound). Under the conditions of Theorem 5, for any region $\Lambda _ { j }$ with $n _ { j } \mathrm { ~  ~ { ~ \geq ~ } ~ }$ $2 r _ { j } \sqrt { d _ { \mathrm { v c } } } .$

$$
\operatorname* { i n f } _ { \hat { h } } \operatorname* { s u p } _ { P } \mathbb { E } \big [ \bar { R } ( \hat { h } ) - \bar { R } ( h _ { P } ^ { * } ) \big ] \geq \frac { w _ { j } } { 1 2 } \frac { r _ { j } \sqrt { d _ { \mathrm { v c } } } } { n _ { j } } = \frac { n _ { j } r _ { j } \sqrt { d _ { \mathrm { v c } } } } { 1 2 n ^ { 2 } } .\tag{23}
$$

In particular, taking the maximum over regions:

$$
\operatorname* { i n f } _ { \hat { h } } \operatorname* { s u p } _ { P } \mathbb { E } \big [ \bar { R } ( \hat { h } ) - \bar { R } ( h _ { P } ^ { * } ) \big ] \geq \frac { \sqrt { d _ { \mathrm { v c } } } } { 1 2 n ^ { 2 } } \operatorname* { m a x } _ { 1 \leq k \leq K } n _ { k } r _ { k } .\tag{24}
$$

Proof. For a chosen region $\Lambda _ { j }$ , construct a distribution in which the classification task within $\Lambda _ { j }$ follows the super-pixel construction of Theorem 2 with range $r _ { j }$ , and the task is trivial in all other regions (i.e. $R _ { k } ( h ) = R _ { k } \bar { ( h ^ { * } ) }$ for $k \neq j$ under every hypothesis). Then $\bar { R } ( \hat { h } ) - \bar { R } ( h ^ { * } ) = w _ { j } ( R _ { j } ( \hat { h } ) - R _ { j } ( h ^ { * } ) )$ and the minimax rate for estimating $R _ { j }$ over $\Lambda _ { j }$ alone is $\Omega ( r _ { j } \sqrt { d _ { \mathrm { v c } } } / n _ { j } )$ by Theorem 2. Multiplying by $\overset { \mathbf { \tilde { w } } _ { j } } { = } n _ { j } ^ { 2 } / n ^ { 2 }$ gives the per-region lower bound. Taking the maximum over j yields (24). □

Comparing the upper bound (17) and the lower bound (24), the upper bound depends on $\sqrt { \sum _ { k } n _ { k } ^ { 2 } r _ { k } ^ { 2 } }$ and the lower bound on ma $_ { \textrm { \tiny A } } n _ { k } r _ { k }$ . Since max $\begin{array} { r } { \mathrm { \Large ~ \dot { \chi } ~ } n _ { k } r _ { k } \le \sqrt { \sum _ { k } n _ { k } ^ { 2 } r _ { k } ^ { 2 } } \le \sqrt { K } } \end{array}$ max n r , the gap between the bounds is at most $\sqrt { K }$ which is a small constant for typical land cover classifications $( K \le 1 0 )$

Remark 2 (Practical interpretation). Given a scene with estimated per-stratum correlation ranges $\hat { r } _ { k } ,$ , the effective sample size should be computed as $n _ { \mathrm { e f f } } \ \stackrel { \cdot } { = } \ n ^ { 4 } / \sum _ { k } n _ { k } ^ { 2 } \hat { r } _ { k } ^ { 2 }$ , not from a global variogram. A scene that is half cropland $( r = 4 0 )$ and half forest $( r = 5 )$ has $n _ { \mathrm { e f f } } = 2 n ^ { 2 } / 1 6 2 5 \approx n ^ { 2 } / 8 1 2 ,$ , whereas a global variogram might estimate $r \approx 2 2 ,$ , giving $n ^ { 2 } / 4 8 4$ overestimating of the effective sample size by nearly a factor of two.

## VI. IMPLICATIONS FOR SPATIAL CROSS-VALIDATION

The main practical consequence of the preceding results concerns validation methodology. We state implications for each setting.

Corollary 1 (Spatial cross-validation in the stationary case). Under the conditions of Theorem 1:

1) Spatial block holdout with block separation $\textit { g } \geq \textit { r }$ O(log n) achieves the optimal generalization bound with effective sample size $n _ { \mathrm { e f f } } = \Theta ( n ^ { 2 } / r ^ { 2 } )$ up to log factors.

2) Random holdout treats all $n ^ { 2 }$ pixels as independent. Any uniform deviation bound based on the nominal sample size $n ^ { 2 }$ underestimates the true generalization error by a factor of up to $\sim r$ log n.

Proof. Statement (i) is immediate from Theorem 1. $\hat { R } _ { \mathrm { b l o c k } }$ with separation $g \ = \ \Theta ( r \log n )$ achieves the bound. For (ii), a random subset of $n ^ { 2 } / 2$ test pixels drawn without spatial structure includes $\sim n ^ { 2 } / ( 2 r ^ { 2 } )$ effectively independent observations (by the lower bound, this cannot yield better than $\Omega ( r \sqrt { d _ { \mathrm { v c } } } / n )$ accuracy in estimating $R ( h ) )$ , but the practitioner computes a standard error as if all $n ^ { 2 } / 2$ pixels were independent, underestimating the confidence interval width by $\sqrt { n ^ { 2 } / n _ { \mathrm { e f f } } } = \Theta ( r )$ □

Corollary 2 (Spatial cross-validation in the piecewise stationary case). Under the conditions of Theorem ${ } ^ { 5 , }$ spatial block holdout with per-stratum block separation $g _ { k } = \Theta ( r _ { k } \log n )$ within each region $\Lambda _ { k }$ achieves the generalization bound (17) with effective sample size $n _ { \mathrm { e f f } } = n ^ { 4 } \bar { / } \sum _ { k } n _ { k } ^ { 2 } r _ { k } ^ { 2 }$ . In particular: 1) Let $\begin{array} { r } { \bar { r } = \sum _ { k } w _ { k } r _ { k } } \end{array}$ denote the area-weighted mean range, with $w _ { k } = \ddot { n _ { k } ^ { 2 } } / n ^ { 2 }$ . Then

$$
n _ { \mathrm { e f f } } ~ \le ~ \frac { n ^ { 2 } } { \bar { r } ^ { 2 } } ,
$$

with equality if and only if all $r _ { k }$ are identical. Thus, whenever the per-stratum ranges are heterogeneous, substituting any single averaged range into the stationary formula $\bar { n ^ { 2 } } / \bar { r } ^ { 2 }$ strictly overestimates the effective sample size.

2) Random holdout underestimates the width of confidence intervals for the generalization error by a factor of up to $\begin{array} { r } { \sqrt { n ^ { 2 } / n _ { \mathrm { e f f } } } = \sqrt { \sum _ { k } n _ { k } ^ { 2 } r _ { k } ^ { 2 } / n _ { \mathrm { . } } } } \end{array}$

Proof. Statement (i): since $\textstyle \sum _ { k } w _ { k } \ = \ 1$ and $x \ \mapsto \ x ^ { 2 }$ is strictly convex, Jensen’s inequality gives $\begin{array} { r } { \bar { r } ^ { 2 } = \left( \sum _ { k } w _ { k } r _ { k } \right) ^ { 2 } \le } \end{array}$ $\begin{array} { r } { \sum _ { k } \dot { w _ { k } } r _ { k } ^ { 2 } = \sum _ { k } n _ { k } ^ { 2 } r _ { k } ^ { 2 } / n ^ { 2 } } \end{array}$ , with equality if and only if the $r _ { k }$ are all equal. Hence $\textstyle \tilde { n ^ { 2 } } / \bar { r } ^ { 2 } \geq n ^ { 4 } / \bar { \sum _ { k } { n _ { k } ^ { 2 } r _ { k } ^ { 2 } } } = n _ { \mathrm { e f f } }$ . Statement (ii) follows from the same argument as Corollary 1(ii).

Remark 3 (Anisotropic block separation). For anisotropic fields (Theorem 3), the block separation should be directiondependent: $g _ { k } = \Theta ( r _ { k } \log n )$ in direction k. This yields elongated validation blocks aligned with the correlation structure. For SAR imagery with range correlation $r _ { 1 } = 5 0$ pixels and azimuth correlation $r _ { 2 } = 1 0 \ p i x e l s ,$ using isotropic separation $g = 5 0$ in both directions discards ∼ 80% of the available effective samples compared to the anisotropic prescription.

a) Practical workflow: The theoretical results suggest the following procedure for spatial cross-validation on a new scene:

1) Stratify. Partition the image into approximately homogeneous regions $\Lambda _ { 1 } , \ldots , \Lambda _ { K }$ (e.g. by land cover class or ecological zone). If directional correlation is suspected, note the principal axes.

2) Estimate correlation ranges. Within each region $\Lambda _ { k } .$ compute the empirical variogram $\hat { \gamma } _ { k } ( h )$ and fit a parametric model to obtain $\hat { r } _ { k }$ (or directional ranges $\hat { r } _ { k , 1 } , \hat { r } _ { k , 2 }$ if anisotropic). A robust choice is the practical range, defined as the lag at which the fitted variogram reaches 95% of its sill.

3) Compute effective sample size. Use $n _ { \mathrm { e f f } } = n ^ { 4 } / \sum _ { k } n _ { k } ^ { 2 } \hat { r } _ { k } ^ { 2 }$ (or $n ^ { 4 } / \sum _ { k } n _ { k } ^ { 2 } \hat { r } _ { k , 1 } \hat { r } _ { k , 2 }$ for anisotropic strata.

4) Set block separation. Within each region, separate spatial CV folds by at least $g _ { k } \ = \ c \cdot { \hat { r } } _ { k }$ log n pixels, where $c \geq 1 / c _ { 1 }$ depends on the covariance model.

5) Report corrected confidence intervals. Replace the nominal sample size $n ^ { 2 }$ with $n _ { \mathrm { e f f } }$ when computing standard errors and confidence intervals for accuracy metrics.

The bound is continuous in $r ,$ and replacing $r _ { k }$ with $\hat { r } _ { k } ( 1 \pm \epsilon )$ changes $n _ { \mathrm { e f f } }$ by a factor of at most $( 1 \pm \epsilon ) ^ { - 2 }$ , so moderate variogram estimation error produces proportional, not catastrophic, error in the bound.

b) Worked examples: For a Sentinel-2 scene $( n = 1 0 , 0 0 0$ pixels per side at 10m resolution) with landscape correlation ranges of 200–500m $( r = 2 0 – 5 0$ pixels), the effective sample size under global stationarity is $n _ { \mathrm { e f f } }$ ≈ $n ^ { 2 } / r ^ { 2 }$ ≈ 40,000– 250,000, roughly two to three orders of magnitude smaller than the pixel count of $1 0 ^ { 8 }$ . For a heterogeneous scene with three strata, suppose we have cropland (40% of pixels, $r _ { 1 } = 4 0 )$ forest $( 4 0 \% , r _ { 2 } = 8 )$ , and urban $( 2 0 \% , r _ { 3 } = 1 5 )$ . Writing the stratum pixel counts as $n _ { k } ^ { 2 } \ = \ w _ { k } n ^ { 2 }$ , the stratified effective sample size from (18) is

$$
{ \begin{array} { r } { n _ { \mathrm { e f f } } = { \frac { n ^ { 4 } } { 0 . 4 n ^ { 2 } \cdot 4 0 ^ { 2 } + 0 . 4 n ^ { 2 } \cdot 8 ^ { 2 } + 0 . 2 n ^ { 2 } \cdot 1 5 ^ { 2 } } } } \\ { = { \frac { n ^ { 2 } } { 0 . 4 \cdot 1 6 0 0 + 0 . 4 \cdot 6 4 + 0 . 2 \cdot 2 2 5 } } \approx { \frac { n ^ { 2 } } { 7 1 1 } } } \end{array} }
$$

A global variogram on this scene might estimate the areaweighted mean range $\bar { r } \approx 2 2$ , giving $n ^ { \bar { 2 } } / \bar { r } ^ { 2 } \approx n ^ { 2 } / 4 9 3$ , which overestimates $n _ { \mathrm { e f f } }$ by a factor of $\sim 1 . 4$ . Whenever the perstratum ranges are heterogeneous, any single averaged range yields confidence intervals that are too narrow. Per-stratum estimation avoids this bias.

## VII. SYNTHETIC EXPERIMENTS

We first validate the theoretical results with a controlled simulation. The goals are to (1) confirm that random holdout overestimates classification accuracy on spatially correlated imagery, with the bias growing in the correlation range $r ;$ (2) verify the effective-sample-size scaling $n _ { \mathrm { e f f } } \sim n ^ { 2 } / r ^ { 2 } ;$ and (3) demonstrate that validation optimism decays with distance from the training region, reaching zero once the distance exceeds r.

We generate synthetic $n \times n$ images $\scriptstyle ( n = 5 0 0 )$ from a stationary Gaussian random field with exponential covariance $C ( d ) ~ = ~ \exp ( - d / r )$ and $d { = } 6$ spectral bands, each drawn independently from the same covariance structure via circulant embedding (FFT). The true label at each pixel is $\begin{array} { r l } { Y _ { s } } & { { } = } \end{array}$ $\mathbf { 1 } [ \mathbf { w } ^ { \top } \mathbf { X } _ { s } + \epsilon _ { s } > 0 ]$ , where $\mathbf { w } \in \mathbb { R } ^ { d }$ is a fixed unit-norm weight vector and $\epsilon _ { s } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ with $\sigma { = } 0 . 5$ is i.i.d. label noise. The linear boundary ensures that the Bayes-optimal classifier is well-defined, while the label noise makes the classification task nontrivial (Bayes error ≈15%). The correlation range r is the parameter we vary across experiments, taking values in $\{ 1 , 3 , 5 , 1 0 , \dots , 7 0 \}$

We use a decision tree (depth 15, minimum leaf size 3) as the classifier. The high capacity of the tree is crucial here as a linear classifier trained on data generated from a linear boundary converges to the Bayes-optimal decision surface regardless of spatial structure and therefore exhibits negligible optimism. The decision tree can actually memorize spatially correlated noise patterns in the training set, and this overfitting to local spatial structure is the mechanism that inflates randomholdout accuracy.

True generalization error is estimated by evaluating each trained classifier on five independently generated images drawn from the same distribution (same w, fresh randomfield realizations). All reported results are averaged over 500 replications per configuration with error bars denoting ±1 standard error. Figure 2 visualizes the experiments.

![](images/2dff54ab10400e00d8343fa127bf271e5643d48dd2d9e8ab29fcba317145260b.jpg)

![](images/d33a6b6035b3b543051dfa0763a39df8797ae033fe9009ae937e5f75be4dbe9c.jpg)

![](images/2d218ab54972bcd1bf1b144803115abaf643919ff64dfafbbf69db02f8c6d290.jpg)  
Fig. 2. Simulation validation of the results. (A) Random holdout error (red) diverges from generalization error (green) as the correlation range r increases. Shaded area is the optimism gap. (B) Empirical effective sample size (blue) scales log-log slope $= - 1 . 7 8$ vs. theory −2. (C) Validation optimism decays with distance from the training boundary, reaching zero at d ≈ r (vertical line). The gray curve shows ∝ $e ^ { - d / r }$ . All experiments use n=500, d=6 spectral bands, exponential covariance.

a) Panel (A): Random holdout vs. true error: For each value of $r ,$ we train the decision tree on a random 50% of the pixels in a single image and evaluate on the remaining 50%. The red curve shows the random-holdout error estimate and the green curve shows the true generalization error measured on fresh images. At r=1 (nearly i.i.d. pixels) the two coincide. As r increases, they diverge: the random-holdout estimate decreases while the true error increases. The shaded region between them is the optimism gap, namely the amount by which random holdout underestimates the true classification error. At $r { = } 7 0$ the gap reaches approximately 2.8 percentage points. As anticipated, the holdout estimate falls (the interleaved test set rewards the memorization) while true error rises (more correlation, more structured noise to memorize).

b) Panel (B): Effective sample size: To measure $n _ { \mathrm { e f f } }$ empirically we train a logistic regression classifier on one image and evaluate it on a second, independently generated image, repeating this 800 times per value of $r _ { \cdot }$ The variance of the test error across repetitions gives $n _ { \mathrm { e f f } } = p ( 1 { - } p ) / \mathrm { V a r } ( \hat { e } )$ , where $p$ is the mean error. We use logistic regression rather than a decision tree here to minimize the contribution of training-set variability to the total variance, isolating the evaluation-side effect. The empirical $n _ { \mathrm { e f f } }$ (blue) tracks the theoretical $n ^ { 2 } / r ^ { 2 }$ line (dashed) in slope, with a log-log slope of −1.78, the shortfall from the predicted −2 reflecting finite-r curvature from the $l o g ^ { 2 } n$ term, which flattens the local slope when $r$ is not small relative to $n .$ The constant-factor offset (roughly one order of magnitude) is consistent with the polylogarithmic factors.

c) Panel (C): Optimism decay with distance: We train the decision tree on the left half of the image (columns 1 to $n / 2 )$ and measure the test error in narrow 15-pixel-wide bands at increasing distances from the training boundary. At each distance $d ,$ we compute the optimism as the difference between the true generalization error and the band-specific test error. Positive optimism indicates that the band benefits from spatial leakage. The optimism is largest immediately adjacent to the training region (+1.2% at distance 7) and decays to zero by distance ≈ $r { = } 5 0$ , closely following the exponential decay ∝ $e ^ { - d / r }$ predicted by the covariance model (gray curve). The vertical dashed line marks $d { = } r$ . Beyond d ≈ $r ,$ the optimism fluctuates around zero, confirming that pixels separated from the training data by more than the correlation range provide an unbiased estimate of generalization performance. The slight negative drift at distances $d > 2 r$ reflects a secondary effect. At large separations the test pixels inhabit a sufficiently different realization of the random field that the classifier, which has partially memorized the training-half realization, performs slightly worse than on a generic fresh image. This is the distribution shift phenomenon we discuss in connection with Wadoux et al. [3].

## VIII. EMPIRICAL EXPERIMENTS

We then validate the theoretical predictions on satellite imagery spanning seven ecoregions, three sensors, and diverse land cover types. All tiles are approximately $5 0 0 ~ \times ~ 5 0 0$ pixels, extracted via Google Earth Engine [16] from cloudfree summer (May–September 2021) median composites, with land cover labels from the 2021 National Land Cover Database (NLCD) [17].

We use three sensors: Landsat 8 surface reflectance [18] (30m, six spectral bands), Sentinel-2 surface reflectance [19] (10m, ten bands), and Sentinel-1 GRD SAR [20] (10m, VV/VH polarizations). Tiles are drawn from seven regions selected for ecological diversity. We use Iowa (flat cropland), Colorado Front Range (mountainous forest/shrub), Chesapeake Bay (coastal wetland/forest/cropland), Arizona (arid desert/shrub), New England (mixed deciduous/urban), Georgia (pine plantation/wetland), and California Central Valley (irrigated agriculture). In total, 43 tiles were analyzed with 25 Landsat, 10 Sentinel-2 (co-located with Landsat tiles for cross-sensor comparison), and 8 Sentinel-1 (one per region, for anisotropy analysis). Table I summarizes the results.

Effective sample size  
TABLE I  
SUMMARY OF CORRELATION RANGES AND EFFECTIVE SAMPLE SIZES ACROSS ALL STUDY TILES. rˆ: FITTED EXPONENTIAL RANGE FROM THE NDVI (OR VV) VARIOGRAM. $n _ { \mathrm { e f f } } :$ : BLOCK-BOOTSTRAP EFFECTIVE SAMPLE SIZE AT BLOCK SIZE $b = { \hat { r } } .$
<table><tr><td>Region</td><td>Tile</td><td>Sensor</td><td>Scale</td><td>τ (px)</td><td>τ (m)</td><td>ntest</td><td> $n _ { \mathrm { e f f } }$ </td></tr><tr><td rowspan="10"></td><td>IA-L1</td><td>Landsat 8</td><td>30m</td><td>15.0</td><td>449</td><td>145684</td><td>1930</td></tr><tr><td>IA-L2</td><td>Landsat 8</td><td>30m</td><td>25.0</td><td>751</td><td>139876</td><td>610</td></tr><tr><td>IA-L3</td><td>Landsat 8</td><td>30m</td><td>13.0</td><td>390</td><td>147744</td><td>2687</td></tr><tr><td>IA-L4</td><td>Landsat 8</td><td>30m</td><td>23.2</td><td>696</td><td>141620</td><td>1861</td></tr><tr><td>IA-L5</td><td>Landsat 8</td><td>30m</td><td>5.6</td><td>168</td><td>149865</td><td>11915</td></tr><tr><td>IA-L6</td><td>Landsat 8</td><td>30m</td><td>20.8</td><td>625</td><td>142002</td><td>2463</td></tr><tr><td>IA-L7</td><td>Landsat 8</td><td>30m</td><td>18.2</td><td>545</td><td>144530</td><td>2448</td></tr><tr><td>IA-S2a</td><td>Sentinel-2</td><td>10m</td><td>35.8</td><td>358</td><td>135520</td><td>955</td></tr><tr><td>IA-S2b</td><td>Sentinel-2</td><td>10m</td><td>19.4</td><td>194</td><td>142485</td><td>1215</td></tr><tr><td>IA-S1a</td><td>Sentinel-1</td><td>10m 10m</td><td>18.0</td><td>180</td><td>144232</td><td>1384 2467</td></tr><tr><td rowspan="5"></td><td>IA-S1b</td><td>Sentinel-1</td><td></td><td>11.9</td><td>119</td><td>146349</td><td></td></tr><tr><td>CO-L1</td><td>Landsat 8</td><td>30m</td><td>26.1</td><td>782</td><td>143674</td><td>1758</td></tr><tr><td>CO-L2</td><td>Landsat 8</td><td>30m</td><td>14.3</td><td>430</td><td>149108</td><td>3168</td></tr><tr><td>CO-L3</td><td>Landsat 8</td><td>30m</td><td>15.4</td><td>461</td><td>148632</td><td>2691</td></tr><tr><td>CO-S2 CO-S1</td><td>Sentinel-2 Sentinel-1</td><td>10m 10m</td><td>36.9 15.7</td><td>369 157</td><td>139095 149238</td><td>1285 2677</td></tr><tr><td rowspan="5">Chesapeake</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CB-L1</td><td>Landsat 8</td><td>30m</td><td>48.2</td><td>1445</td><td>138308</td><td>1608</td></tr><tr><td>CB-L2</td><td>Landsat 8</td><td>30m</td><td>11.5</td><td>345</td><td>156006</td><td>6235</td></tr><tr><td>CB-L3</td><td>Landsat 8 Sentinel-2</td><td>30m</td><td>101.1 17.8</td><td>3034</td><td>112497</td><td>171</td></tr><tr><td>CB-S2 CB-S1</td><td>Sentinel-1</td><td>10m 10m</td><td>14.9</td><td>178 149</td><td>153405 154866</td><td>2810 4167</td></tr><tr><td rowspan="5">Arizona</td><td></td><td></td><td></td><td></td><td></td><td></td><td>2355</td></tr><tr><td>AZ-L1</td><td>Landsat 8</td><td>30m</td><td>10.7</td><td>321</td><td>157464</td><td>1740</td></tr><tr><td>AZ-L2 AZ-L3</td><td>Landsat 8 Landsat 8</td><td>30m</td><td>18.9</td><td>566 1299</td><td>153260 141135</td><td>4269</td></tr><tr><td>AZ-S2</td><td>Sentinel-2</td><td>30m 10m</td><td>43.3 28.6</td><td>286</td><td>148716</td><td>373</td></tr><tr><td>AZ-S1</td><td>Sentinel-1</td><td>10m</td><td>11.4</td><td>114</td><td>156978</td><td>1898</td></tr><tr><td rowspan="5">New England</td><td></td><td></td><td></td><td></td><td></td><td></td><td>659</td></tr><tr><td>NE-L1 NE-L2</td><td>Landsat 8 Landsat 8</td><td>30m</td><td>28.6</td><td>859</td><td>144248</td><td>4330</td></tr><tr><td>NE-L3</td><td>Landsat 8</td><td>30m 30m</td><td>12.0 21.2</td><td>360 635</td><td>154256 148500</td><td>2027</td></tr><tr><td>NE-S2</td><td>Sentinel-2</td><td>10m</td><td>32.1</td><td>321</td><td>142560</td><td>529</td></tr><tr><td>NE-S1</td><td>Sentinel-1</td><td>10m</td><td>23.0</td><td>230</td><td>147015</td><td>472</td></tr><tr><td rowspan="6">Georgia</td><td>GA-L1</td><td></td><td></td><td></td><td></td><td></td><td>842</td></tr><tr><td>GA-L2</td><td>Landsat 8 Landsat 8</td><td>30m 30m</td><td>35.2 8.4</td><td>1057 253</td><td>139173 156806</td><td>5065</td></tr><tr><td>GA-L3</td><td>Landsat 8</td><td>30m</td><td>21.5</td><td>646</td><td>148004</td><td>6550</td></tr><tr><td>GA-S2a</td><td>Sentinel-2</td><td>10m</td><td>101.7</td><td>1017</td><td>112440</td><td>91</td></tr><tr><td>GA-S2b</td><td>Sentinel-2</td><td>10m</td><td>18.9</td><td>189</td><td>152312</td><td>958</td></tr><tr><td>GA-S1</td><td>Sentinel-1</td><td>10m</td><td>70.9</td><td>709</td><td>127342</td><td>134</td></tr><tr><td rowspan="7">California</td><td>CA-L1</td><td>Landsat 8</td><td>30m</td><td>37.6</td><td>1127</td><td>146421</td><td>955</td></tr><tr><td>CA-L2</td><td>Landsat 8</td><td>30m</td><td>23.3</td><td>698</td><td>153634</td><td>1842</td></tr><tr><td>CA-L3</td><td>Landsat 8</td><td>30m</td><td>14.6</td><td>439</td><td>157760</td><td>2621</td></tr><tr><td>CA-S2a</td><td>Sentinel-2</td><td>10m</td><td>25.3</td><td>253</td><td>152337</td><td>858</td></tr><tr><td>CA-S2b</td><td>Sentinel-2</td><td>10m</td><td>23.7</td><td>237</td><td>153634</td><td>694</td></tr><tr><td>CA-S1</td><td></td><td>10m</td><td>25.6</td><td></td><td></td><td>638</td></tr><tr><td></td><td>Sentinel-1</td><td></td><td></td><td>256</td><td>152337</td><td></td></tr></table>

## A. Variogram Estimation and $n ^ { 2 } / r ^ { 2 }$ Scaling

For each tile we compute the NDVI (or VV backscatter for SAR) and estimate an isotropic empirical variogram from ∼200,000 random pixel pairs, binned at 2-pixel lag intervals. We fit an exponential model with nugget, $\gamma ( h ) = c _ { 0 } + ( c _ { 1 } -$ $c _ { 0 } ) ( 1 - e ^ { - h / r } )$ , by nonlinear least squares.

The fitted correlation ranges vary substantially both within and across regions. For Landsat tiles, the median rˆ per region ranges from ∼15 pixels (450m) in Iowa to ∼43 pixels (1300m) in parts of Arizona, with within-region variability often spanning a factor of 2–3. Iowa Landsat tiles yield $\hat { r } \in [ 5 . 6 , 2 5 . 0 ]$ pixels across seven tiles, reflecting local differences in field size and land cover heterogeneity. Cross-sensor comparisons on co-located tiles show that physical correlation lengths are broadly consistent between Landsat and Sentinel-2 (e.g. Iowa: ∼450m vs. ∼280m), while SAR consistently shows shorter correlation lengths (∼150–180m), consistent with radar sensitivity to fine-scale surface structure.

![](images/f74de573fccf82e2664552f5478ae4ea1a78f80963445206625bbe0cfdb515b5.jpg)  
Fig. 3. Block-bootstrap effective sample size as a function of block size b for four tiles spanning different regions.

To test the $n ^ { 2 } / r ^ { 2 }$ prediction, we employ a spatial block bootstrap on all tiles. For each tile, a logistic regression classifier is trained on the left third of the image (with a gap of rˆ pixels to prevent leakage) and evaluated on the remaining test region. For each block size b, we tile the test region into nonoverlapping $b \times b$ blocks, resample blocks with replacement 500 times, and compute $n _ { \mathrm { e f f } } = \hat { p } ( 1 - \hat { p } ) / \mathrm { V a r } ( \hat { e } )$

Fig. 3 shows the result for four representative tiles spanning Iowa, Chesapeake, California, and New England. In all cases, $n _ { \mathrm { e f f } }$ decreases as a power law in the block size, running parallel to the $b ^ { - 2 }$ reference line. The log-log slopes range from −1.6 to −1.9 across tiles, consistent with the theoretical prediction of −2 up to the polylogarithmic correction in Theorem 1. The scaling holds across all three sensors, all seven regions, and for both binary and multi-class classification tasks. Across all tiles, $n _ { \mathrm { e f f } }$ ranges from 91 (GA-S2a, $\hat { r } = 1 0 2$ pixels) to 11915 (IA-L5, rˆ = 5.6 pixels), compared to test set sizes of 112000–158000 pixels, confirming that effective sample sizes are typically 1–3 orders of magnitude smaller than pixel counts.

## B. Anisotropy Validation

Sentinel-1 SAR imagery provides a natural test case for the anisotropic extension (Theorem 3), since SAR backscatter can exhibit directional correlation structure from both the sensor geometry and landscape features. For each of the eight SAR tiles, we compute directional variograms along the N–S and E–W axes (with $\mathrm { a \pm 3 0 ^ { \circ } }$ angular tolerance) and fit exponential models independently to obtain directional ranges $\hat { r } _ { 1 }$ (N–S) and ${ \hat { r } } _ { 2 } \ ( \mathrm { E } { - } \mathrm { W } )$

Table II summarizes the results. The anisotropy ratio $\operatorname* { m a x } ( \hat { r } _ { 1 } , \hat { r } _ { 2 } ) / \operatorname* { m i n } ( \hat { r } _ { 1 } , \hat { r } _ { 2 } )$ ranges from 1.0 (isotropic) to 8.2, and the fraction of effective samples wasted by isotropic blocking ranges from 0% to 88%. Chesapeake exhibits the strongest anisotropy (ratio 8.2x, 88% waste), with the E–W correlation extending along the coastal shoreline $( \hat { r } _ { 2 } ~ = ~ 7 3 ~ $ pixels) while the N–S direction decorrelates rapidly across the land–water boundary $( \hat { r } _ { 1 } ~ = ~ 9$ pixels). Colorado shows a similar pattern along mountain ridges (ratio 5.9x, 83% waste). California and Iowa are near-isotropic (ratios 1.1x and 1.4x), consistent with the regular geometry of agricultural fields. These results confirm that the anisotropic prescription $n _ { \mathrm { e f f } } = n ^ { 2 } / ( r _ { 1 } r _ { 2 } )$ from Theorem 3 can recover substantially more effective samples than isotropic blocking with $r _ { \mathrm { m a x } }$ in landscapes with directional structure.

TABLE II  
DIRECTIONAL CORRELATION RANGES FROM SENTINEL-1 SAR. ${ \hat { r } } _ { 1 } ,$ rˆ<sub>2</sub>: N–S AND E–W EXPONENTIAL RANGES. RATIO: max / min OF THE TWO RANGES. WASTE: FRACTION OF EFFECTIVE SAMPLES LOST BY USING ISOTROPIC BLOCKING WITH $r _ { \mathrm { m a x } }$ INSTEAD OF THE ANISOTROPIC $r _ { 1 } r _ { 2 }$
<table><tr><td>Tile</td><td> $\hat { r } _ { 1 }$  (px)</td><td> $\hat { r } _ { 2 }$  (px)</td><td>Ratio</td><td> $r _ { 1 } r _ { 2 }$ </td><td>Waste</td></tr><tr><td>IA-S1a</td><td>17</td><td>12</td><td>1.4x</td><td>208</td><td>27%</td></tr><tr><td>IA-S1b</td><td>75</td><td>75</td><td>1.0x</td><td>5563</td><td>0%</td></tr><tr><td>CO-S1</td><td>60</td><td>10</td><td>5.9x</td><td>604</td><td>83%</td></tr><tr><td>CB-S1</td><td>9</td><td>73</td><td>8.2x</td><td>643</td><td>88%</td></tr><tr><td>AZ-S1</td><td>9</td><td>21</td><td>2.3x</td><td>185</td><td>57%</td></tr><tr><td>NE-S1</td><td>47</td><td>36</td><td>1.3x</td><td>1696</td><td>22%</td></tr><tr><td>GA-S1</td><td>159</td><td>60</td><td>2.7x</td><td>9515</td><td>63%</td></tr><tr><td>CA-S1</td><td>24</td><td>27</td><td>1.1x</td><td>647</td><td>10%</td></tr></table>

## C. Piecewise Stationarity Validation

To validate the piecewise stationary extension (Theorem 5), we partition selected tiles by NLCD land cover class and fit per-stratum variograms. For strata with at least 500 valid pixels, we estimate $\hat { r } _ { k }$ and compute the stratified effective sample size $n _ { \mathrm { e f f } } ~ = ~ n _ { \mathrm { t o t a l } } ^ { 2 } / \sum _ { k } n _ { k } \hat { r } _ { k } ^ { 2 }$ , where $n _ { k }$ is the pixel count in stratum k. Strata with non-convergent variogram fits are excluded.

Table III shows the per-stratum breakdown for six tiles from Iowa, Colorado, and Chesapeake. The correlation ranges vary dramatically within individual scenes. In Iowa, cropland has $\hat { r } ~ = ~ 9 . 7$ pixels while developed land has $\hat { r } ~ = ~ 3 . 4 .$ , and in Colorado, herbaceous cover reaches $\hat { r } = 6 0$ pixels while shrub is only 4.6. The dominant stratum’s correlation range does not determine the global estimate.

The ratio of stratified to global $n _ { \mathrm { e f f } }$ ranges from $0 . 4 \times$ (CO-L3) to 5.2× (CO-L2), demonstrating that the global variogram can err in either direction. When most strata have shorter ranges than the global estimate (IA-L1, CO-L1), the stratified $n _ { \mathrm { e f f } }$ is higher, so the global variogram is overly conservative. When a minority stratum with long-range correlation is averaged away (CO-L3), the global estimate is anticonservative and the true effective sample size is lower than reported. The Chesapeake tile (CB-L1) illustrates a third scenario where water $( { \hat { r } } = 3 2 ,$ 52% of pixels) and wetland (rˆ = 32, 12%) dominate the denominator despite cropland $( { \hat { r } } = 7 )$ covering nearly a quarter of the scene.

These results confirm that per-stratum estimation is necessary for heterogeneous scenes and that the formula from Theorem 5 captures effects invisible to a global variogram.

TABLE III  
STRATIFIED VS. GLOBAL EFFECTIVE SAMPLE SIZE FOR SIX HETEROGENEOUS TILES. r¯: GLOBAL VARIOGRAM RANGE. $r _ { k }$ RANGE: MIN–MAX OF PER-STRATUM RANGES (NLCD CLASSES WITH ≥ 500 PIXELS).
<table><tr><td>Tile</td><td>K</td><td>r (px)</td><td> $r _ { k }$  range</td><td> $n _ { \mathrm { e f f } } ^ { \mathrm { g l o b a l } }$ </td><td> $n _ { \mathrm { e f f } } ^ { \mathrm { s t r a t } } ~ / ~ n _ { \mathrm { e f f } } ^ { \mathrm { g l o b a l } }$ </td><td></td></tr><tr><td>IA-L1</td><td>6</td><td>14.0</td><td>3.2-17.6</td><td>1161</td><td></td><td>2.4x</td></tr><tr><td>CO-L1</td><td>5</td><td>17.9</td><td>4.6–32.1</td><td>730</td><td></td><td>4.3x</td></tr><tr><td>CO-L2</td><td>5</td><td>28.2</td><td>3.4-28.8</td><td>295</td><td></td><td>5.2x</td></tr><tr><td>CO-L3</td><td>6</td><td>17.1</td><td>11.3-60.3</td><td>802</td><td></td><td>0.4x</td></tr><tr><td>CO-S2</td><td>4</td><td>21.2</td><td>2.2-17.7</td><td>522</td><td></td><td>2.2x</td></tr><tr><td>CB-L1</td><td>5</td><td>50.3</td><td>6.6-32.4</td><td>96</td><td></td><td>3.5x</td></tr></table>

## IX. DISCUSSION

We established that the effective sample size of a spatially correlated remote sensing image is $\Theta ( n ^ { 2 } / r ^ { 2 } )$ , where r is the correlation range, and that this rate is minimax optimal. The result provides a rigorous statistical justification for spatial cross-validation in the remote sensing context and gives quantitative guidance on the required block separation.

The bound takes the correlation range r as known. In practice, r can be estimated from an empirical variogram. We can compute the semivariance $\begin{array} { r } { \hat { \gamma } ( h ) = \frac { 1 } { 2 \left| N ( h ) \right| } \sum _ { ( s , t ) \in N ( h ) } ( X _ { s } - } \end{array}$ $X _ { t } ) ^ { 2 }$ at each lag $h ,$ fit a parametric model (exponential, Matern, etc.), and read off the practical range at which correla-´ tion drops to 5% of the sill. Since the bound is continuous in $r ,$ moderate estimation error in $\hat { r }$ produces proportional error in the bound by replacing r with $\hat { r } ( 1 \pm \epsilon )$ changes $n _ { \mathrm { e f f } }$ by a factor of $( 1 \pm \epsilon ) ^ { - \bar { 2 } }$

A subtlety concerns which spatial field the correlation range $r$ refers to. The bounds are driven by the mixing rate of the loss field $\{ \ell ( h , s ) \}$ , whereas the variograms in Section VIII are computed on the raw features, which may not coincide. The feature range bounds the loss range from above when the label depends on the features through a fixed decision boundary. However, when intra-class variability is small, the loss field decorrelates faster than the feature field, and the feature-variogram range $\hat { r }$ overestimates the effective correlation range of the loss, yielding a conservative $n _ { \mathrm { e f f } }$ . Estimating the loss-field range directly, e.g. from the variogram of held-out residuals, would tighten the bound, at the cost of requiring a fitted classifier. We use the feature range throughout as it is model-free, computable prior to training, and its conservatism errs in the safe direction for confidence intervals.

Wadoux et al. [3] argued that spatial cross-validation can be pessimistic for map accuracy assessment. This is consistent with our analysis once the prediction task is distinguished. Theorem 1 characterizes generalization to new draws from the same stationary field. When the goal is instead extrapolation to geographically disjoint regions (new biomes, different continents), the relevant difficulty is distributional shift, not spatial correlation, and domain adaptation bounds are more appropriate. Spatial CV with large block separations can conflate the two effects, penalizing a classifier for distribution shift that the user does not intend to generalize across. Our result clarifies that the block separation should be scaled to the correlation range r, not to the size of the study area.

We conjecture the blocking argument generalizes to spacetime lattices $\Lambda _ { n } \times \{ 1 , . . . , T \}$ with anisotropic mixing spatial range $r _ { s }$ and temporal range $r _ { t } . \ \mathrm { A p p l y i n g }$ 3D blocking gives an effective sample size $n _ { \mathrm { e f f } } \sim n ^ { 2 } T / ( r _ { s } ^ { 2 } r _ { t } )$ , quantifying the marginal value of each additional revisit. For a Sentinel-2 time series with $T = 3 6$ revisits per year and temporal correlation of $r _ { t } \approx 3$ revisits, a temporal stack multiplies the effective sample size by $\sim T / r _ { t } \approx 1 2$ relative to a single image.

Our results have several limitations. The bounds are worstcase and conservative, with unoptimized constants and polylogarithmic factors. The key value is the $n _ { \mathrm { e f f } } = \Theta ( n ^ { 2 } / r ^ { 2 } )$ scaling and the spatial-CV guidance, not a sharp constant. The correlation range r must be estimated from an empirical variogram; while the bound is continuous in r and moderate estimation error propagates proportionally (Remark 2), a poorly specified variogram will bias $n _ { \mathrm { e f f } }$ . We assume label noise is conditionally independent given features, but tasks with spatial label structure (e.g. segmentation) may exhibit different mixing behavior. Investigating such settings is left to future work.

## X. CONCLUSION

We derived finite-sample generalization bounds for classifiers trained on spatially dependent remote sensing data. The main result is that the effective sample size of an $n \times n$ image with spatial correlation range r is $\dot { \Theta ( n ^ { 2 } / r ^ { 2 } ) }$ , and that this rate is minimax optimal. For anisotropic fields, the effective sample size is $\Theta ( n ^ { 2 } / ( r _ { 1 } r _ { 2 } ) )$ ), and for piecewise stationary scenes per-stratum correlation ranges must be combined through a variance-weighted formula. As a corollary, spatial block holdout with separation proportional to r achieves optimal generalization guarantees, while random holdout can underestimate confidence intervals by $\Theta ( r )$ . These results provide theoretical footing and quantitative guidance for spatial crossvalidation in remote sensing accuracy assessments.

## REFERENCES

[1] D. Jupp, A. Strahler, and C. Woodcock, “Autocorrelation and regularization in digital images. i. basic theory,” IEEE Transactions on Geoscience and Remote Sensing, vol. 26, no. 4, pp. 463–473, 1988.

[2] D. R. Roberts, V. Bahn, S. Ciuti, M. S. Boyce, J. Elith, G. Guillera-Arroita, S. Hauenstein, J. J. Lahoz-Monfort, B. Schroder, W. Thuiller, D. I. Warton,¨ B. A. Wintle, F. Hartig, and C. F. Dormann, “Crossvalidation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” Ecography, vol. 40, no. 8, pp. 913–929, 2017. [Online]. Available: https://doi.org/10.1111/ecog.02881

[3] A. M. J.-C. Wadoux, G. B. M. Heuvelink, S. de Bruin, and D. J. Brus, “Spatial cross-validation is not the right way to evaluate map accuracy,” Ecological Modelling,

vol. 457, p. 109692, 2021. [Online]. Available: https: //doi.org/10.1016/j.ecolmodel.2021.109692

[4] P. Ploton, F. Mortier, M. Rejou-M´ echain, N. Barbier,´ N. Picard, V. Rossi, C. F. Dormann, G. Cornu, G. Viennois, N. Bayol, A. Lyapustin, S. Gourlet-Fleury, and R. Pelissier, “Spatial validation reveals´ poor predictive performance of large-scale ecological mapping models,” Nature Communications, vol. 11, p. 4540, 2020. [Online]. Available: https://doi.org/10.1038/ s41467-020-18321-y

[5] H. Meyer and E. Pebesma, “Predicting into unknown space? estimating the area of applicability of spatial prediction models,” Methods in Ecology and Evolution, vol. 12, no. 9, pp. 1620–1633, 2021. [Online]. Available: https://besjournals.onlinelibrary. wiley.com/doi/abs/10.1111/2041-210X.13650

[6] J.-F. Wang, G. Christakos, and M.-G. Hu, “Modeling spatial means of surfaces with stratified nonhomogeneity,” IEEE Transactions on Geoscience and Remote Sensing, vol. 47, no. 12, pp. 4167–4174, 2009.

[7] A. D. Rocha, T. A. Groen, A. K. Skidmore, and L. Willemen, “Role of sampling design when predicting spatially dependent ecological data with remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 1, pp. 663–674, 2021.

[8] Y. Liao, Y. Bai, S. Chen, J. Wang, Y. You, Z. Chen, H. Wu, H. Huang, W. Li, M. Zang, Z. Wang, X. Zhang, B. Xu, and P. Gong, “The scaling law in remote sensing mapping for model performance forecasting and reference sample optimization,” Remote Sensing of Environment, vol. 344, p. 115529, 2026.

[9] M. Mohri and A. Rostamizadeh, “Stability bounds for stationary ϕ-mixing and β-mixing processes,” J. Mach. Learn. Res., vol. 11, p. 789–814, Mar. 2010. [Online]. Available: https://jmlr.org/papers/volume11/mohri10a/mohri10a.pdf

[10] V. Kuznetsov and M. Mohri, “Generalization bounds for non-stationary mixing processes,” Machine Learning, vol. 106, no. 1, pp. 93–117, 2017. [Online]. Available: https://doi.org/10.1007/s10994-016-5588-2

[11] D. S. Modha and E. Masry, “Minimum complexity regression estimation with weakly dependent observations,” IEEE Transactions on Information Theory, vol. 42, no. 6, pp. 2133–2145, November 1996. [Online]. Available: https://doi.org/10.1109/18.556602

[12] P. Doukhan, Mixing: Properties and Examples, ser. Lecture Notes in Statistics. New York, NY: Springer, 1994, vol. 85. [Online]. Available: https://doi.org/10. 1007/978-1-4612-2642-0

[13] R. C. Bradley, Introduction to Strong Mixing Conditions, Volume 1. Kendrick Press, 2007.

[14] B. Yu, “Rates of convergence for empirical processes of stationary mixing sequences,” The Annals of Probability, vol. 22, no. 1, pp. 94–116, 1994. [Online]. Available: https://www.jstor.org/stable/2244496

[15] A. B. Tsybakov, Introduction to Nonparametric Estimation, ser. Springer Series in Statistics. New York, NY: Springer, 2009. [Online]. Available:

[16] N. Gorelick, M. Hancher, M. Dixon, S. Ilyushchenko, D. Thau, and R. Moore, “Google earth engine: Planetary-scale geospatial analysis for everyone,” Remote Sensing of Environment, 2017. [Online]. Available: https://doi.org/10.1016/j.rse.2017.06.031

[17] U.S. Geological Survey (USGS), “Annual NLCD collection 1 science products,” 2024. [Online]. Available: https://doi.org/10.5066/P94UXNTS

[18] D. P. Roy, M. A. Wulder, T. R. Loveland, C. E. Woodcock, R. G. Allen, M. C. Anderson, D. Helder, J. R. Irons, D. M. Johnson, R. Kennedy, T. A. Scambos, C. B. Schaaf, J. R. Schott, Y. Sheng, E. F. Vermote, A. S. Belward, R. Bindschadler, W. B. Cohen, F. Gao, J. D. Hipple, P. Hostert, J. Huntington, C. O. Justice, A. Kilic, V. Kovalskyy, Z. P. Lee, L. Lymburner, J. G. Masek, J. McCorkel, Y. Shuai, R. Trezza, J. Vogelmann, R. H. Wynne, and Z. Zhu, “Landsat-8: Science and product vision for terrestrial global change research,” Remote Sensing of Environment, vol. 145, pp. 154–172, 2014.

[19] M. Drusch, U. D. Bello, S. Carlier, O. Colin, V. Fernandez, F. Gascon, B. Hoersch, C. Isola, P. Laberinti, P. Martimort, A. Meygret, F. Spoto, O. Sy, F. Marchese, and P. Bargellini, “Sentinel-2: ESA’s Optical High-Resolution Mission for GMES Operational Services,” Remote Sensing of Environment, vol. 120, pp. 25–36, 2012.

[20] R. Torres, P. Snoeij, D. Geudtner, D. Bibby, M. Davidson, E. Attema, P. Potin, B. Rommen, N. Floury, M. Brown, I. N. Traver, P. Deghaye, B. Duesmann, B. Rosich, N. Miranda, C. Bruno, M. L’Abbate, R. Croci, A. Pietropaolo, M. Huchler, and F. Rostan, “GMES Sentinel-1 mission,” Remote Sensing of Environment, vol. 120, pp. 9–24, 2012.

[21] S. Shalev-Shwartz and S. Ben-David, Understanding Machine Learning: From Theory to Algorithms. Cambridge, UK: Cambridge University Press, 2014. [Online]. Available: https://doi.org/10.1017/CBO9781107298019

[22] M. Mohri, A. Rostamizadeh, and A. Talwalkar, Foundations of Machine Learning, 2nd ed. Cambridge, MA: The MIT Press, 2018.

[23] V. N. Vapnik and A. Y. Chervonenkis, “On the uniform convergence of relative frequencies of events to their probabilities,” Theory of Probability & Its Applications, vol. 16, no. 2, pp. 264–280, 1971. [Online]. Available: https://doi.org/10.1137/1116025

[24] G. Douzas, F. Bacao, J. Fonseca, and M. Khudinyan, “Imbalanced learning in land cover classification: Improving minority classes’ prediction accuracy using the geometric smote algorithm,” Remote Sensing, vol. 11, no. 24, p. 3040, 2019. [Online]. Available: https://doi.org/10.3390/rs11243040

[25] A. Mellor, S. Boukir, A. Haywood, and S. Jones, “Exploring issues of training data imbalance and mislabelling on random forest performance for large area land cover classification using the ensemble margin,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 105, pp. 155–168, Jul. 2015. [Online].

## APPENDIX

This section briefly glosses the learning-theoretic machinery used in the main text, for readers who work with machine learning in practice but have not seen the underlying theory. Readers seeking a fuller treatment may consult the accessible introduction of Shalev-Shwartz and Ben-David [21], whose early chapters develop PAC learning, VC dimension, and uniform convergence from first principles (a free copy is hosted by the authors here), or the more advanced account of Mohri et al. [22], which also covers the learning theoretic treatment of dependent data underlying the mixing bounds we build on.

Anyone who has held out a test set has estimated a generalization gap. A classifier h has a true risk $R ( h )$ , or its error rate on the whole population, which we never observe, and an empirical risk $\hat { R } ( h )$ , which is the error rate on the finite sample in hand. Reporting held-out accuracy is the act of using $\hat { R } ( h )$ as a stand-in for $R ( h )$ , and the hope that the two are close is the hope that the gap $R ( h ) - { \hat { R } } ( h )$ is small. A generalization bound makes that hope quantitative as a guarantee, holding with high probability, on how large the gap can be and, crucially, one that holds simultaneously for every h the training procedure might have selected. If we only controlled the gap for a single h fixed in advance, the training procedure’s freedom to select the h that happens to look best on this particular sample would be exactly the loophole overfitting exploits. The guarantee must therefore survive a worst-case choice of h, which is what the supremum in (S1) encodes.

Everyone knows that a model flexible enough to memorize the training set will report an optimistically low training error, and VC dimension is the formal handle on “flexible enough.” Say that H shatters a set of points if, for every possible assignment of binary labels to them, some $h \in \mathcal H$ classifies all of them correctly, then the class is flexible enough to realize any labeling, including the adversarial ones. The VC dimension $d _ { \mathrm { v c } }$ of H [23] is the size of the largest set it can shatter. A linear classifier in the plane has $d _ { \mathrm { v c } } = 3$ , which means any labeling of three points is separable by a line, but no line separates four points labeled in the XOR pattern. A deep decision tree can shatter large sets and so has large $d _ { \mathrm { v c } } . ~ \mathrm { A }$ class that shatters sets of every size has $d _ { \mathrm { v c } } ~ = ~ \infty$ and can fit pure noise. It is a measure of capacity that, unlike a raw parameter count, directly predicts overfitting. The classical guarantee is that, for n i.i.d. samples, with probability at least $1 - \delta ,$

$$
\underbrace { \operatorname* { s u p } _ { h \in \mathcal { H } } \bigl | R ( h ) - \hat { R } ( h ) \bigr | } _ { \mathrm { w o r s t - c a s e ~ g a p ~ o v e r ~ t h e ~ c l a s s } } \ \lesssim \ \sqrt { \frac { \overset { \mathrm { c a p a c i t y } } { \widetilde { d } _ { \mathrm { v c } } \log n } + \log ( 1 / \delta ) } { \underset { \mathrm { s a m p l e ~ s i z e } } { \overset { n } { n } } } } .\tag{S1}
$$

Holding n fixed, a more flexible class (larger $d _ { \mathrm { v c } } )$ admits a larger gap, which is the formal version of “flexible models overfit more.” Holding $d _ { \mathrm { v c } }$ fixed, the gap shrinks as $1 / { \sqrt { n } } , \mathbf { e . g . }$ quadrupling the sample halves the error bar. The rate $1 / { \sqrt { n } } ,$ and specifically the appearance of n in the denominator, is the object our main results correct.

The n in (S1) arrives through a concentration inequality. The average of n independent bounded quantities concentrates around its mean at rate $1 / { \sqrt { n } } .$ . This is the one step that assumes independence, and it is the step spatial correlation breaks. When neighboring pixels are correlated, the $n ^ { 2 }$ pixels of an image do not supply $n ^ { 2 }$ independent quantities to average, adjacent pixels largely repeat information already present in their neighbors. The image behaves as though it held some smaller number $m \ < \ n ^ { 2 }$ of independent observations. The technical contribution of Section III is to make “some smaller number” formally defined.

A blocking construction extracts $\begin{array} { l l l } { m } & { \approx } & { n ^ { 2 } / r ^ { 2 } } \end{array}$ nearlyindependent units, and the bound (S1) then holds with m in place of n. m here is not a subset of privileged pixels one could point to on the image, but the number of actually independent observations whose averaged error would carry the same variance as the full correlated scene. This variancebased reading is what the piecewise-stationary formula of Section V makes explicit, where $n _ { \mathrm { e f f } }$ appears as the reciprocal of a weighted variance.

The blocking argument shows the $n ^ { 2 } / r ^ { 2 }$ rate is achievable. Section IV shows it cannot be beaten, and the reason is that an adversary may build a scene in which every $r \times r$ patch is internally identical and drawn independently of the others, then all $n ^ { 2 }$ pixels encode only $\lfloor n / r \rfloor ^ { 2 }$ free pieces of information, and no algorithm, however clever, can recover structure that was never present. The effective sample size is thus a property of the spatial dependence, not of the learner.

The generalization bounds of Section III are stated for a hypothesis class of K-way classifiers and hold for any number of classes $K \geq 2$ . The number of classes enters the bound only through the VC dimension $d _ { \mathrm { v c } } ,$ which sits inside the constant. The dependence of the effective sample size on the correlation range, $n _ { \mathrm { e f f } } \sim n ^ { 2 } / r ^ { 2 }$ , is independent of $K$ . This section confirms that invariance empirically, and additionally checks that class imbalance does not alter the scaling.

## A. Setup

We reuse the field-generation and estimation procedure of the binary effective-sample-size experiment (Section VII, Panel B) unchanged, varying only the labeling rule. For each value of the correlation range r, we draw an n×n image $( n =$ 500) with $d = 6$ spectral bands from a stationary Gaussian random field with exponential covariance $C ( d ) = \exp ( - d / r )$ each band generated independently by circulant embedding. The latent score $\boldsymbol { \ell } _ { s } = \mathbf { w } ^ { \intercal } \mathbf { X } _ { s } + \boldsymbol { \epsilon } _ { s }$ is formed from a fixed weight vector w and i.i.d. label noise $\epsilon _ { s } \sim \mathcal { N } ( 0 , 0 . 5 ^ { 2 } )$ , exactly as in the binary experiment.

The three labeling rules differ only in how $\ell _ { s }$ is discretized:

• Binary $( K = 2 ) \colon$ a single threshold at the median of $\ell _ { s } .$ reproducing the rule $Y _ { s } = \mathbf { 1 } [ \boldsymbol { \ell } _ { s } > t ]$ of Section VII.

• Four-class balanced $( K \ : = \ : 4 ) \colon$ three thresholds at the 25th, 50th, and 75th percentiles of $\ell _ { s } ,$ giving four classes of equal prior probability.

• Four-class imbalanced $( K = 4 )$ : three thresholds at the 70th, 80th, and 90th percentiles, giving a $7 0 / 1 0 / 1 0 / 1 0$ split in which one class dominates. This is the regime in which practitioners most often worry about imbalance.

Thresholds are estimated once from a pilot draw so that the target class proportions are met on average. Each condition is fit with multinomial logistic regression and evaluated on a second, independently drawn image. We use 600 replications per value of $r ~ \in ~ \{ 1 , 2 , 3 , 5 , 8 , 1 0 , 1 5 , 2 0 , 3 0 , 4 0 , 6 0 , 8 0 \}$ and estimate $n _ { \mathrm { e f f } } = \bar { p } ( 1 - \bar { p } ) / \mathrm { V a r } ( \hat { e } )$ from the across-replication variance of the test error, where $\bar { p }$ is the mean error rate. This estimator is valid for any K since the per-pixel error indicator is Bernoulli regardless of how many classes produced a given misclassification.

## B. Results

Figure S1 shows the three curves. In every condition, $n _ { \mathrm { e f f } }$ decreases as a power law in r, running parallel to the $n ^ { 2 } / r ^ { 2 }$ reference and far below the naive pixel count $n ^ { 2 }$ . The fitted log-log slopes (over $r \geq 3 .$ , where finite-r curvature is mild) are −1.85 for the binary task, −1.87 for the four-class balanced task, and −1.88 for the four-class imbalanced task, and all consistent with the theoretical exponent of −2 up to the polylogarithmic correction of Theorem 1.

The level of $n _ { \mathrm { e f f } }$ does depend on the task, as expected. The mean error rate rises from $\bar { p } ~ \approx ~ 0 . 1 3$ in the binary case to $\bar { p } \approx 0 . 3 3$ for four balanced classes (four classes are intrinsically harder to separate), and returns to $\bar { p } \approx 0 . 1 3$ for the imbalanced split, whose dominant class is easily classified. These shifts move the curves vertically but leave their slope unchanged. Neither the number of classes nor the degree of class imbalance affects the $n ^ { 2 } / r ^ { 2 }$ scaling. The cost of spatial correlation is governed by the correlation range r alone, with K and the class distribution absorbed into the constant, exactly as the theory predicts.

Class imbalance is a well-recognized difficulty in remotesensing land-cover classification, where minority classes are systematically under-learned and are commonly addressed by resampling or margin-based methods [24, 25]. Our experiment shows that this concern, while important for per-class accuracy, is orthogonal to the spatial-correlation correction studied here. Tmbalance shifts the level of $n _ { \mathrm { e f f } }$ through the base error rate but leaves the $n ^ { 2 } / r ^ { 2 }$ scaling unchanged, so the blockseparation prescription applies regardless of class balance.

Effective sample size is class-count invariant in slope  
![](images/b9ae38c1a612c9c5729b7673025786988fc218679efef144267a6e09cad793c5.jpg)  
Fig. S1. Effective sample size versus correlation range for binary and fourclass classification at n = 500.