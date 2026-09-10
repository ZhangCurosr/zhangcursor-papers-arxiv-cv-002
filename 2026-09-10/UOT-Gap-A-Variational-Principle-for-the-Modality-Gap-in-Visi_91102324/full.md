# UOT-Gap: A Variational Principle for the Modality Gap in Vision–Language Models via Unbalanced Optimal Transport

Zonglin Yang <sup>B</sup>, Huilan Ma, Xudan Zheng, and Yuejun Xie

Guangdong Police College

3258244847@qq.com

1451727521@qq.com

3113434702@qq.com

18320142522@163.com

Abstract. Vision–language models such as CLIP embed images and text in a shared space, where modality-specific distributions often remain separated. Existing accounts connect this modality gap to initialization, contrastive dynamics, and information imbalance, while its distributional and pairwise contributions to retrieval remain unresolved. We introduce UOT-Gap, a training-free variational diagnostic that models frozen image and text embeddings with unbalanced entropic optimal transport (UOT). The UOT optimum separates transport, coupling complexity, and marginal mass variation; a complementary pair-aware residual compares observed image–caption pairs with the UOT soft matching. On Flickr8K and COCO-1K with frozen CLIP, OpenCLIP, and SigLIP encoders, cap tion degradation reduces Flickr8K Recall@1 from 0.559 to 0.003. Across six dataset–model conditions, the pair-aware residual tracks retrieval degradation with mean absolute Spearman 0.973, compared with 0.392 for the mean gap. The association remains stable across five random COCO-1K subsets at 0.954 ± 0.026, with a minimum of 0.943. UOT barycentric updates reduce the transport objective while degrading retrieval, distinguishing geometric objective descent from task improvement. These results establish UOT-Gap as a diagnostic for caption quality, modality alignment, and retrieval robustness.

Keywords: Vision-language models · Modality gap · Unbalanced optimal transport · Cross-modal retrieval

## 1 Introduction

Contrastive vision–language models (VLMs), such as CLIP [22] and SigLIP [27], map images and captions into a shared representation space. Their embeddings often form modality-specific clouds, producing the modality gap [17]. Previous studies connect this structure to initialization and contrastive optimization [17], gradient-flow dynamics and mismatched pairs [26], and information imbalance between images and captions [23]. A remaining diagnostic challenge is to identify a variational quantity that separates distributional mismatch from broken image– caption correspondence.

![](images/a4680aa9b4d921d3a9c7f766f9a84ea42ac54e1648c585e378857316923e7e67.jpg)  
Fig. 1. Overview. UOT-Gap interprets the modality gap as the residual of an unbal anced transport problem. Distributional components diagnose marginal and coupling mismatch; the pair-aware residual diagnoses whether observed image–caption pairs are worse than the best soft UOT matching.

We address this challenge with unbalanced entropic optimal transport (UOT). Balanced optimal transport enforces complete mass conservation, whereas VLM representations contain modality-specific factors: images may include visual content omitted by captions, and captions may express abstractions without localized visual counterparts. UOT assigns a penalty to mass deletion or creation and thereby represents this information asymmetry directly.

Given frozen image embeddings $X = \{ x _ { i } \} _ { i = 1 } ^ { n }$ and text embeddings $Y =$ $\{ y _ { j } \} _ { j = 1 } ^ { m }$ on the unit sphere, empirical weights $a , b .$ , and cosine cost $C _ { i j } = 1 \mathrm { - } \langle x _ { i } , y _ { j } \rangle$ ， we solve

$$
\pi ^ { \star } = \operatorname * { a r g m i n } _ { \pi \geq 0 } \left. C , \pi \right. + \varepsilon \mathrm { K L } ( \pi \| a b ^ { \top } ) + \rho _ { x } \mathrm { K L } ( \pi \mathbf { 1 } \| a ) + \rho _ { y } \mathrm { K L } ( \pi ^ { \top } \mathbf { 1 } \| b ) .\tag{1}
$$

The optimized objective yields transport, coupling, and marginal components. For paired retrieval datasets, we further define a pair-aware residual that compares the observed diagonal pairing cost with the UOT matching cost. A random caption permutation isolates the resulting distinction by preserving the text marginal distribution while disrupting pairwise retrieval.

Using only frozen embeddings, UOT-Gap can screen caption quality, monitor modality alignment during training, and audit retrieval robustness without encoder fine-tuning.

Our contributions are:

1. We formulate the VLM modality gap as the residual of an unbalanced entropic optimal transport problem between frozen image and text embeddings, and introduce a pair-aware UOT residual that separates marginal distribution mismatch from broken image–caption alignment.

2. We prove three results that justify the diagnostic: transport bounds the classical mean gap, missing modality-specific mass induces an unavoidable UOT cost or marginal penalty, and UOT barycenters provide a first-order descent direction for the UOT objective.

3. We validate the diagnostic across Flickr8K, COCO-1K, synthetic controlled data, three frozen encoders, five random COCO subsets, caption degradation, hyperparameter sweeps, and post-hoc correction baselines, showing that the pair-aware residual tracks retrieval degradation more reliably than the classical mean gap.

## 2 Related Work

Modality gap in VLMs. CLIP learns transferable visual representations by aligning images and text with a contrastive loss [22]. Liang et al. [17] showed that modalities can remain separated in the shared embedding space and connected this gap to initialization and optimization. Yaras et al. [26] analyze gradientflow dynamics and identify mismatched pairs and temperature as mechanisms that preserve the gap. Schrodi et al. [23] link information imbalance to both modality gap and object bias. These causal accounts motivate our complementary variational diagnostic, whose residuals quantify the cost and mass variation required for cross-modal matching.

Vision–language pretraining and retrieval. Vision–language pretraining has evolved from cross-modal Transformer encoders [19,1] to large-scale dual encoders and frozen-backbone image–text models [13,16,15]. These models improve retrieval and transfer through greater alignment capacity. UOT-Gap complements this progress with a post-hoc diagnostic that distinguishes marginal distribution shift from broken pair identity across frozen encoders.

Optimal transport and unbalanced matching. Optimal transport provides a geometry for comparing distributions [25,20]. Entropic regularization enables fast Sinkhorn scaling [3], and unbalanced optimal transport relaxes exact marginal constraints with divergence penalties [2,21,12]. Sinkhorn divergences and their sample behavior motivate practical regularized distribution comparison [5,7]. Practical solvers are available in the Python Optimal Transport library [6]. We treat the optimized UOT residual as a diagnostic signal that exposes the structure of relaxed cross-modal matching.

Multimodal identifiability and information asymmetry. Multimodal contrastive learning may identify shared latent factors while leaving modality-specific factors unresolved [4]. This view motivates our missing-attribute analysis: private factors make relaxed mass matching the appropriate object, allowing UOT to separate alignable shared mass from residual modality-specific mass.

## 3 Method

## 3.1 Embeddings and UOT Objective

Let $f _ { \theta }$ be an image encoder and $g _ { \phi }$ a text encoder. For image–caption pairs $\{ ( I _ { i } , T _ { i } ) \} _ { i = 1 } ^ { n }$ , define normalized embeddings

$$
x _ { i } = \frac { f _ { \theta } ( I _ { i } ) } { \| f _ { \theta } ( I _ { i } ) \| _ { 2 } } , \qquad y _ { i } = \frac { g _ { \phi } ( T _ { i } ) } { \| g _ { \phi } ( T _ { i } ) \| _ { 2 } } , \qquad x _ { i } , y _ { i } \in \mathbb { S } ^ { d - 1 } .\tag{2}
$$

The empirical measures are $\begin{array} { r } { \hat { \mu } = \sum _ { i } a _ { i } \delta _ { x _ { i } } } \end{array}$ and $\begin{array} { r } { \hat { \nu } = \sum _ { j } b _ { j } \delta _ { y _ { j } } } \end{array}$ , typically with uniform weights. The classical mean gap is

$$
\mathsf { G a p } _ { \mathrm { m e a n } } ( \hat { \mu } , \hat { \nu } ) = \left\| \sum _ { i } a _ { i } x _ { i } - \sum _ { j } b _ { j } y _ { j } \right\| _ { 2 } .\tag{3}
$$

The mean gap captures global centroid displacement; higher-order geometry, neighborhood structure, and pair identity remain outside this first-moment statistic.

For a nonnegative coupling $\pi \in \mathbb { R } _ { + } ^ { n \times m }$ , denote row and column marginals by $r ( \pi ) = \pi \mathbf { 1 } _ { m }$ and $c ( \pi ) = \pi ^ { \top } \mathbf { 1 } _ { n }$ . With generalized KL divergence, the UOT objective is

$$
\mathcal { I } _ { \varepsilon , \rho _ { x } , \rho _ { y } } ( \pi ; X , Y ) = \langle C , \pi \rangle + \varepsilon \mathrm { K L } ( \pi \| a b ^ { \top } ) + \rho _ { x } \mathrm { K L } ( r ( \pi ) \| a ) + \rho _ { y } \mathrm { K L } ( c ( \pi ) \| b ) .\tag{4}
$$

Let $\pi ^ { \star }$ minimize Eq. (4). We define

$$
\begin{array} { r } { \mathsf { C o s t } = \langle C , \pi ^ { \star } \rangle , \quad \mathsf { C o u p l } = \mathrm { K L } ( \pi ^ { \star } \| a b ^ { \top } ) , \quad \mathsf { M a r g } = \mathrm { K L } ( r ( \pi ^ { \star } ) \| a ) + \mathrm { K L } ( c ( \pi ^ { \star } ) \| b ) . } \end{array}\tag{5}
$$

These terms form a distributional decomposition: Cost measures cross-modal matching cost, Coupl measures complexity relative to independent matching, and Marg measures mass variation through deletion or creation.

## 3.2 Pair-Aware UOT Residual

Retrieval depends on both the marginal distributions and the observed image– caption pairing. A random caption permutation preserves the text distribution while collapsing retrieval. We therefore report the paired cost

$$
\mathsf { C o s t } _ { \mathrm { p a i r } } ( X , Y ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } C _ { i i } ,\tag{6}
$$

and the pair-aware UOT residual

$$
{ \mathsf { G a p } } _ { \mathrm { p a i r } } ( X , Y ) = { \mathsf { C o s t } } _ { \mathrm { p a i r } } ( X , Y ) - { \frac { { \mathsf { C o s t } } ( X , Y ) } { \sum _ { i j } \pi _ { i j } ^ { \star } } } .\tag{7}
$$

Intuitively, ${ \mathsf { G a p } } _ { \mathrm { p a i r } }$ is large when the observed pairs are much worse than the best soft cross-modal matching implied by UOT. The reported score is

$$
\operatorname { U O T - G A P } _ { \varepsilon , \rho } ( X , Y ) = \left[ \operatorname { G a p } _ { \mathsf { p a i r } } ( X , Y ) + \mathsf { M a r g } ( X , Y ) + 0 . 1 \mathsf { C o u p l } ( X , Y ) \right] _ { + } ^ { 1 / 2 } .\tag{8}
$$

For distributional diagnosis, the components should be inspected separately. For retrieval degradation diagnosis, ${ \mathsf { G a p } } _ { \mathrm { p a i r } }$ is the primary UOT-derived statistic because it explicitly evaluates the observed image–caption pair.

## 3.3 Barycentric Correction

The UOT plan also induces a lightweight post-hoc correction. For rows with positive transported mass $\begin{array} { r } { r _ { i } ^ { \star } = \sum _ { j } \pi _ { i j } ^ { \star } } \end{array}$ , define the text barycenter

$$
b _ { i } = \frac { \sum _ { j } \pi _ { i j } ^ { \star } y _ { j } } { r _ { i } ^ { \star } } .\tag{9}
$$

Because CLIP-like embeddings are normalized, we move along the tangent direction on the sphere:

$$
v _ { i } = P _ { x _ { i } } ^ { \perp } b _ { i } = b _ { i } - \left. { x _ { i } , b _ { i } } \right. x _ { i } , \qquad x _ { i } ^ { \prime } = \frac { x _ { i } + \alpha v _ { i } } { \| x _ { i } + \alpha v _ { i } \| _ { 2 } } .\tag{10}
$$

This update serves as a diagnostic intervention that tests whether the UOT matching geometry reduces transport residuals and modality separability. Because its direction follows soft matching instead of observed diagonal pairs, we evaluate retrieval jointly with every gap metric.

## 4 Theory

We state three results that motivate the diagnostic. Full derivations use Jensen’s inequality, Pinsker’s inequality, pointwise minimization over retained private mass, and the envelope theorem for the optimized UOT value.

Theorem 1 (Balanced transport certificate). Let $\mu , \nu$ be probability measures $o n \mathbb { S } ^ { d - 1 }$ , and let $c ( x , y ) = 1 - \langle x , y \rangle$ ⟩. For any coupling $\gamma \in \pi ( \mu , \nu )$

$$
\left\| m _ { \mu } - m _ { \nu } \right\| _ { 2 } ^ { 2 } \leq 2 \int c ( x , y ) \mathrm { d } \gamma ( x , y ) .\tag{11}
$$

Consequently ${ \mathsf { G a p } } _ { \mathrm { m e a n } } ^ { 2 } ( \mu , \nu ) \leq 2 0 { \mathsf { T } } _ { c } ( \mu , \nu )$

The theorem provides a transport certificate for the mean gap: low cosine transport cost implies small centroid separation. Its scope is global displacement; local mismatch and missing modality-specific factors require the relaxed decomposition developed below.

Proof sketch. Let $( X , Y ) \sim \gamma$ . Since $\gamma$ has marginals $\mu$ and $\nu , \mathbb { E } [ X - Y ] =$ $m _ { \mu } - m _ { \nu }$ . Jensen’s inequality gives $\left\| m _ { \mu } - m _ { \nu } \right\| _ { 2 } ^ { 2 } \leq \mathbb { E } \left\| X - Y \right\| _ { 2 } ^ { 2 }$ . On the unit sphere, $\| x - y \| _ { 2 } ^ { 2 } = 2 - 2 \left. x , y \right. = 2 c ( x , y )$ . Taking the infimum over couplings yields the OT bound.

Theorem 2 (UOT certificate for the mean gap). Let $\mu , \nu$ be probability measures on $\mathbb { S } ^ { d - 1 }$ . Let π be any nonzero finite coupling whose normalized marginals ${ \bar { r } } , { \bar { c } }$ are absolutely continuous with respect to $\mu , \nu$ . Then

$$
\mathsf { G a p } _ { \mathrm { m e a n } } ( \mu , \nu ) \le \sqrt { 2 \int c ( x , y ) \mathrm { d } \bar { \pi } ( x , y ) } + \sqrt { 2 \mathrm { K L } ( \bar { r } \| \mu ) } + \sqrt { 2 \mathrm { K L } ( \bar { c } \| \nu ) } .\tag{12}
$$

Proof sketch. Decompose $m _ { \mu } - m _ { \nu } = ( m _ { \mu } - m _ { \bar { r } } ) + ( m _ { \bar { r } } - m _ { \bar { c } } ) + ( m _ { \bar { c } } - m _ { \nu } )$ . The middle term is controlled by Theorem 1 applied to the normalized coupling π¯. The two marginal terms are bounded by total variation between the normalized marginals and the original measures, and Pinsker’s inequality converts those totalvariation terms into KL terms. Transport and marginal deviations therefore jointly certify the classical mean gap and ground the UOT residuals in a measurable geometric quantity.

Assumption 3 (Separated image-private mass). There exists $A \subset \mathbb { S } ^ { d - 1 }$ with $\mu ( A ) = \eta > 0$ such that $c ( x , y ) \geq \delta > 0$ for all $x \in A$ and $y \in \operatorname { s u p p } ( \nu )$ . We interpret A as image-private mass absent or underspecified in captions.

Theorem 4 (Missing-attribute lower bound). Under Assumption 3, every finite coupling π for the semi-unbalanced objective $\begin{array} { r } { \mathcal { I } _ { \rho } ^ { x } ( \pi ) = \int c \mathrm { d } \pi + \rho \mathrm { K L } ( \pi _ { X } \| \mu ) } \end{array}$ satisfies

$$
\begin{array} { r } { \mathcal { I } _ { \rho } ^ { x } ( \pi ) \geq \rho \eta \left( 1 - e ^ { - \delta / \rho } \right) . } \end{array}\tag{13}
$$

If the private attribute is uniformly distributed over k states and only one state is text-alignable, then $\eta = 1 - 1 / k = 1 - e ^ { - H ( U ) }$

Private visual mass therefore contributes a positive UOT residual through either transport cost or marginal KL.

Proof sketch. On the private set A, let $s ( x ) \in [ 0 , 1 ]$ denote the fraction of image mass retained by a coupling. Transporting retained mass costs at least $\delta s ( x )$ , while deleting mass pays $\rho ( s ( x )$ log $s ( x ) - s ( x ) + 1 )$ under generalized KL. Minimizing the pointwise function over s gives $s ^ { \star } = \stackrel { \cdot } { e } ^ { - \delta / \stackrel { \cdot } { \rho } }$ and value $\rho ( 1 - e ^ { - \delta / \rho } )$ . Integrating over $\mu ( A ) = \eta$ proves the lower bound. The entropy expression follows by setting $\eta = 1 - 1 / k$ for a uniformly distributed private factor with one alignable state.

Theorem 5 (First-order UOT correction descent). For each image embedding define $\begin{array} { r } { r _ { i } ^ { \star } = \sum _ { j } \pi _ { i j } ^ { \star } } \end{array}$ and $\begin{array} { r } { b _ { i } = ( \sum _ { j } \pi _ { i j } ^ { \star } y _ { j } ) / r _ { i } ^ { \star } } \end{array}$ . Update

$$
x _ { i } ( \alpha ) = \frac { x _ { i } + \alpha P _ { x _ { i } } ^ { \perp } b _ { i } } { \left\| x _ { i } + \alpha P _ { x _ { i } } ^ { \perp } b _ { i } \right\| _ { 2 } } .\tag{14}
$$

Let $\begin{array} { r } { \phi ( X , Y ) = \operatorname* { m i n } _ { \pi \geq 0 } \mathcal { I } _ { \varepsilon , \rho _ { x } , \rho _ { y } } ( \pi ; X , Y ) } \end{array}$ . For suficiently small $\alpha > 0$

$$
\varPhi ( X ( \alpha ) , Y ) \leq \varPhi ( X , Y ) - \alpha \sum _ { i : r _ { i } ^ { \star } > 0 } r _ { i } ^ { \star } \left\| P _ { x _ { i } } ^ { \perp } b _ { i } \right\| _ { 2 } ^ { 2 } + O ( \alpha ^ { 2 } ) .\tag{15}
$$

The guarantee concerns first-order descent of the UOT objective; retrieval response is evaluated empirically.

Proof sketch. For fixed $\pi ^ { \star }$ , only the cosine cost term changes to first order under the update. The derivative of $1 - \langle x _ { i } ( \alpha ) , y _ { j } \rangle$ at $\alpha = 0 \mathrm { \ e q u a l s } - \left. P _ { x _ { i } } ^ { \perp } b _ { i } , y _ { j } \right.$ Summing with weights $\pi _ { i j } ^ { \star }$ yields $- r _ { i } ^ { \star } \left\| P _ { x _ { i } } ^ { \perp } b _ { i } \right\| _ { 2 } ^ { 2 }$ for row i. The envelope theorem transfers this fixed-plan descent to the optimized value up to $O ( \alpha ^ { 2 } )$ terms, assuming local stability of the entropic UOT optimum.

## 5 Experiments

## 5.1 Setup

We evaluate frozen CLIP ViT-B/32 [22], OpenCLIP ViT-B/32 [11], and SigLIP B/16 [27] encoders. Flickr8K [10] and the COCO Karpathy split [18,14] are evaluated with $n = 1 0 0 0$ image–caption pairs, and five independent COCO subsets measure sampling stability. Caption conditions include full, half, quarter, nounonly, attribute-dropped, and random captions. The random condition cyclically shifts captions, preserving the text marginal distribution while breaking pair identity.

Unless otherwise stated, we use cosine cost, $\varepsilon = 0 . 0 8 , \rho = 0 . 8$ , generalized Sinkhorn iterations with threshold $1 0 ^ { - 7 }$ , and report Recall@1/5/10, modalityprobe AUC, logit entropy, $\mathsf { G a p } _ { \mathrm { m e a n } } , \mathsf { C o s t } _ { \mathrm { p a i r } } , \mathsf { G a p } _ { \mathrm { p a i r } }$ , and UOT-Gap. We compute metrics from cached frozen embeddings; no encoder is fine-tuned. The modality probe is a logistic classifier trained to distinguish image from text embeddings. The synthetic experiment samples normalized Gaussian sphere distributions with controlled mean separation and concentration imbalance, allowing the marginal and transport terms to be tested independently of caption semantics.

## 5.2 Retrieval Degradation

The main result is that pair-aware UOT statistics track retrieval degradation more reliably than the classical mean gap in this benchmark. On Flickr8K with CLIP ViT-B/32, Recall@1 drops from 0.559 for full captions to 0.148 for half captions, 0.043 for quarter captions, and 0.003 for random captions. The mean gap is unchanged by random permutation because the caption distribution is unchanged, while $\mathsf { C o s t } _ { \mathrm { p a i r } }$ increases from 0.681 to 0.824 and ${ \mathsf { G a p } } _ { \mathrm { p a i } }$ increases from −0.126 to 0.017. Across all fixed-parameter dataset–model conditions, ${ \mathsf { G a p } } _ { \mathrm { p a i r } }$ reaches best absolute Spearman 1.000 and mean absolute Spearman 0.973, compared with 0.841 best and 0.392 mean for ${ \sf G a p } _ { \mathrm { m e a n } }$

![](images/a8fa609b682630a7fdddb6b2def309fa6d85feda331ee09e102932335f3951d0.jpg)  
Fig. 2. Caption degradation on Flickr8K and COCO-1K. Pair-aware UOT residuals and the composite $\mathrm { U O T - G A P }$ score increase as observed caption pairs become less aligned, while Recall@1 decreases.

The COCO subset analysis shows stable behavior across five independent 1K samples with CLIP ViT-B/32. Full-caption Recall@1 is $0 . 5 1 4 \pm 0 . 0 0 9 .$ , quartercaption Recall@1 is $0 . 0 7 8 \pm 0 . 0 0 5$ , and random-caption Recall@1 is $0 . 0 0 1 \pm 0 . 0 0 1$ Across these seeds, ${ \mathsf { G a p } } _ { \mathrm { p a i r } }$ reaches mean Spearman $0 . 9 5 4 { \pm } 0 . 0 2 6$ with a minimum of 0.943, whereas ${ \mathsf { G a p } } _ { \mathrm { m e a n } } ^ { - }$ averages $0 . 2 6 9 \pm 0 . 1 4 2$ . The recurring association supports frozen-embedding subsets as a stable low-compute diagnostic setting.

## 5.3 Synthetic Data and Hyperparameter Sensitivity

On normalized Gaussian sphere samples, mean separation increases the mean gap, whereas concentration imbalance increases UOT marginal and transport terms. The mean gap therefore isolates global displacement, while UOT components respond to higher-order distributional mismatch. Changes in concentration can increase the UOT terms even when centroid separation remains modest.

We also sweep $\varepsilon \in \{ 0 . 0 2 , 0 . 0 5 , 0 . 0 8 , 0 . 1 2 , 0 . 2 0 \}$ and $\rho \in \{ 0 . 2 , 0 . 5 , 0 . 8 , 1 . 5 , 3 . 0 \}$ Across 150 parameter–dataset–model combinations, the pair-aware residual has mean absolute Spearman 0.947 and median 1.000 with retrieval degradation; 142 combinations reach $| \rho _ { s } | \geq 0 . 8$ . The pair-aware association is stable over this practical grid, while the raw decomposition values vary with ε and $\rho$ and are reported together with both parameters.

Table 1. Caption degradation on Flickr8K with CLIP ViT-B/32. The random condition preserves the caption distribution while breaking observed pairing, isolating the pair aware signal.
<table><tr><td>Caption</td><td>Mean</td><td>PairCost</td><td>Pair-UOT</td><td>UOT-GAP</td><td>R@1</td><td>Entropy</td></tr><tr><td>Full</td><td>0.825</td><td>0.681</td><td>-0.126</td><td>0.253</td><td>0.559</td><td>1.595</td></tr><tr><td>Half</td><td>0.922</td><td>0.739</td><td>-0.057</td><td>0.349</td><td>0.148</td><td>2.945</td></tr><tr><td>Quarter</td><td>0.981</td><td>0.773</td><td>-0.023</td><td>0.390</td><td>0.043</td><td>3.701</td></tr><tr><td>Noun-only</td><td>0.970</td><td>0.761</td><td>-0.038</td><td>0.374</td><td>0.072</td><td>4.143</td></tr><tr><td>Attribute-drop</td><td>0.863</td><td>0.691</td><td>-0.115</td><td>0.270</td><td>0.484</td><td>1.842</td></tr><tr><td>Random</td><td>0.825</td><td>0.824</td><td>0.017</td><td>0.455</td><td>0.003</td><td>1.595</td></tr></table>

Table 2. Spearman correlation between gap metrics and retrieval degradation across caption conditions. Higher absolute values indicate stronger monotone association with Recall@1 drop.
<table><tr><td>Dataset</td><td>Model</td><td>Mean</td><td>Paired</td><td>Pair-UOT</td><td>UOT-GAP</td><td>Entropy</td></tr><tr><td>COCO-1K</td><td>CLIP</td><td>0.429</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.290</td></tr><tr><td>COCO-1K</td><td>OpenCLIP</td><td>0.143</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.290</td></tr><tr><td>COCO-1K</td><td>SigLIP</td><td>0.841</td><td>0.464</td><td>0.841</td><td>0.464</td><td>0.058</td></tr><tr><td>Flickr8K</td><td>CLIP</td><td>0.143</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.232</td></tr><tr><td>Flickr8K</td><td>OpenCLIP</td><td>0.429</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.371</td></tr><tr><td>Flickr8K</td><td>SigLIP</td><td>0.371</td><td>0.657</td><td>1.000</td><td>0.943</td><td>-0.143</td></tr><tr><td>Mean absolute</td><td></td><td>0.392</td><td>0.853</td><td>0.973</td><td>0.901</td><td>0.231</td></tr></table>

## 5.4 Model Sweep

Fig. 5 demonstrates diagnostic behavior across three encoder families. CLIP and OpenCLIP show the clearest monotone relationship between pair-aware UOT residuals and caption degradation on Flickr8K and COCO-1K. SigLIP is more model-dependent: on COCO-1K, the mean gap has high correlation in one setting, while the composite UOT-Gap score is weaker because it combines marginal and coupling terms with the pair residual. This model dependence motivates reporting both the composite score and the raw pair-aware residual.

## 5.5 Post-hoc Correction

We compare raw embeddings with mean-shift removal, orthogonal Procrustes alignment, and UOT barycentric correction. Procrustes is a strong pair-preserving baseline based on global orthogonal alignment.

UOT barycentric correction follows Theorem 5: it decreases UOT-related objective terms, with paired cost dropping from 0.681 for raw CLIP embeddings to 0.324 at α = 0.6. Retrieval simultaneously decreases from R@1 0.559 to 0.349, whereas Procrustes increases R@1 to 0.829. This result separates objective descent from task performance and supports barycentric correction as a diagnostic intervention.

![](images/8d908f1b05f1a26c428c9fb19e386ebef4f3abb0368f86fdc3fdfca5b34a5342.jpg)

![](images/9aec284263b46b509749886e5280fd368ac18e3cc45da9abe230a7d7f806b95b.jpg)  
Fig. 3. UOT hyperparameter sensitivity. Across 150 parameter–dataset–model combinations, the pair-aware UOT residual has mean absolute Spearman 0.947 and median 1.000; 142/150 combinations reach $| \rho _ { s } | \geq 0 . 8 .$

Table 3. Post-hoc correction on Flickr8K with CLIP ViT-B/32. Lower AUC means less linear modality separability; higher retrieval is better.
<table><tr><td>Method</td><td>Mean↓</td><td>PairCost↓</td><td>Pair-UOT</td><td>AUC↓</td><td>R@1↑</td><td>R@5↑</td></tr><tr><td>Raw</td><td>0.825</td><td>0.681</td><td>-0.126</td><td>1.000</td><td>0.559</td><td>0.824</td></tr><tr><td>Mean-shift</td><td>0.009</td><td>0.341</td><td>-0.121</td><td>0.270</td><td>0.356</td><td>0.609</td></tr><tr><td>Procrustes</td><td>0.014</td><td>0.195</td><td>-0.219</td><td>0.317</td><td>0.829</td><td>0.947</td></tr><tr><td> $\mathrm { U O T } ~ \alpha = 0 . 2$ </td><td>0.713</td><td>0.574</td><td>-0.123</td><td>1.000</td><td>0.521</td><td>0.805</td></tr><tr><td> $\mathrm { U O T } ~ \alpha = 0 . 4$ </td><td>0.570</td><td>0.447</td><td>-0.114</td><td>1.000</td><td>0.430</td><td>0.708</td></tr><tr><td> $\mathrm { U O T } ~ \alpha = 0 . 6$ </td><td>0.419</td><td>0.324</td><td>-0.094</td><td>1.000</td><td>0.349</td><td>0.572</td></tr></table>

![](images/0574bd89593ded01e42754a551758878a1abce2fe2d92f87fa9520ab7b002780.jpg)

![](images/c0fc72e77392a5900eccf206ffa38d81c0ca6e22cb7aef396c4ffd9009772f9b.jpg)

![](images/2745705c8d8051ad176a7c02292e9f259e01dbd9d4f1547957e68c7bb09306dd.jpg)  
Fig. 4. Synthetic controlled data. Mean separation and concentration imbalance are varied on normalized Gaussian sphere samples. UOT transport and marginal terms respond to distributional shifts, while the mean gap isolates global displacement.

![](images/26ac712230a72f839dfd749f61bb66d549a4aa06e419fb8aaa1630343ffa5617.jpg)

![](images/3efaec0739ff2422d9e6f5289bae98b0ad92382e0d1850f3627c215cb059337d.jpg)

![](images/b445ee6184c33a94e6ed1fb740fd1cb9322967375386f5958cf1366c154add5d.jpg)

![](images/448768594edb43cd9c8353935a494edc00d3d295a27c305392982f9e1f9d8660.jpg)  
Fig. 5. Model sweep over frozen CLIP, OpenCLIP, and SigLIP checkpoints. Pair-aware UOT residuals are most informative for CLIP/OpenCLIP retrieval degradation settings and remain model-dependent for SigLIP.

![](images/06be644f542b8fa64147d44d7eaee2aadae8628010f7cc30b3121e90c2fe8df9.jpg)

![](images/49e33e3fe3ab6c260845e7256b582c693009d3d3cf5920b9d6eeac12c57eb38d.jpg)  
Fig. 6. Correction trade-of. UOT barycentric steps reduce paired cost and UOT residual as retrieval declines, while Procrustes provides the strongest retrieval correction in this frozen-embedding run.

## 6 Discussion and Limitations

The central finding is that the modality gap has two diagnostic levels. Distributional UOT components quantify transport, coupling complexity, and marginal mass variation, while the pair-aware residual measures how observed image– caption pairs difer from the UOT soft matching. The random-caption control isolates this distinction: it preserves the caption distribution, collapses retrieval, and increases the pair-aware residual.

Across the tested datasets and encoders, the pair-aware residual tracks captioninduced retrieval degradation more consistently than the mean-gap baseline. Distributional UOT terms answer a complementary question and vary more across encoders. We therefore recommend component-level reporting: transport and marginal terms for distribution comparison, and the pair-aware residual for paired retrieval diagnosis.

The correction experiment further separates geometric optimization from downstream utility. UOT barycentric steps reduce paired cost and UOT objective terms, as predicted by the descent theorem, while retrieval declines as the diagonal pair structure is smoothed. Procrustes preserves paired supervision and performs better as a retrieval correction. This comparison defines UOT-Gap as an analysis tool and its barycentric update as a diagnostic intervention.

The present scope covers frozen global embeddings and 1K retrieval subsets. Five independent COCO samples establish subset stability, while full COCO evaluation remains an important extension. The missing-attribute lower bound uses a stylized separation assumption, the pair-aware residual requires paired image–caption data, and UOT hyperparameters afect the decomposition values. Future evaluation should include larger retrieval benchmarks, additional VLM families, and distributional baselines such as MMD [8], energy distance [24], and Sinkhorn divergence [5]. Pair-aware comparisons with captioning metrics such as CLIPScore [9] will further define the diagnostic range.

## Reproducibility and Ethics

The experiments use frozen forward passes and UOT/Sinkhorn solvers. The implementation stores embeddings, caption-degradation metadata, random seeds, UOT hyperparameter grids, and plotting scripts. Task metrics accompany every geometric correction because semantic errors can persist after distributional alignment.

## 7 Conclusion

We introduced UOT-Gap, a variational framework that interprets the VLM modality gap through unbalanced entropic optimal transport. The theory connects transport and marginal variation to the mean gap and missing modality-specific mass, and establishes a first-order descent direction for UOT barycentric updates. Empirically, distributional components decompose modality imbalance, while the pair-aware residual tracks retrieval degradation across caption conditions, encoder families, hyperparameter settings, and random COCO subsets. The correction study further shows that geometric objective descent and retrieval improvement are distinct outcomes. Together, these results support UOT-Gap for screening caption quality, tracking modality alignment, and auditing frozen VLM retrieval robustness under low-compute constraints.

Declaration of Generative AI Use. During the preparation of this work, the authors used OpenAI GPT for language polishing and assistance with data analysis. All AI-assisted text and analytical outputs were reviewed and verified by the authors, who take full responsibility for the content of this publication.

## References

1. Chen, Y.C., Li, L., Yu, L., El Kholy, A., Ahmed, F., Gan, Z., Cheng, Y., Liu, J.: Uniter: Universal image-text representation learning. In: European conference on computer vision. pp. 104–120. Springer (2020)

2. Chizat, L., Peyré, G., Schmitzer, B., Vialard, F.X.: Scaling algorithms for unbalanced optimal transport problems. Mathematics of computation 87(314), 2563–2609 (2018)

3. Cuturi, M.: Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems 26 (2013)

4. Daunhawer, I., Bizeul, A., Palumbo, E., Marx, A., Vogt, J.E.: Identifiability results for multimodal contrastive learning. arXiv preprint arXiv:2303.09166 (2023)

5. Feydy, J., Séjourné, T., Vialard, F.X., Amari, S.i., Trouvé, A., Peyré, G.: Interpolating between optimal transport and mmd using sinkhorn divergences. In: The 22nd international conference on artificial intelligence and statistics. pp. 2681–2690. PMLR (2019)

6. Flamary, R., Courty, N., Gramfort, A., Alaya, M.Z., Boisbunon, A., Chambon, S., Chapel, L., Corenflos, A., Fatras, K., Fournier, N., et al.: Pot: Python optimal transport. Journal of Machine Learning Research 22(78), 1–8 (2021)

7. Genevay, A., Chizat, L., Bach, F., Cuturi, M., Peyré, G.: Sample complexity of sinkhorn divergences. In: The 22nd international conference on artificial intelligence and statistics. pp. 1574–1583. PMLR (2019)

8. Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B., Smola, A.: A kernel two-sample test. The journal of machine learning research 13(1), 723–773 (2012)

9. Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., Choi, Y.: Clipscore: A referencefree evaluation metric for image captioning. In: Proceedings of the 2021 conference on empirical methods in natural language processing. pp. 7514–7528 (2021)

10. Hodosh, M., Young, P., Hockenmaier, J.: Framing image description as a ranking task: Data, models and evaluation metrics. Journal of Artificial Intelligence Research 47, 853–899 (2013)

11. Ilharco, G., Wortsman, M., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., et al.: Openclip. Zenodo (2021)

12. Janati, H., Muzellec, B., Peyré, G., Cuturi, M.: Entropic optimal transport between unbalanced gaussian measures has a closed form. Advances in neural information processing systems 33, 10468–10479 (2020)

13. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: International conference on machine learning. pp. 4904–4916. PMLR (2021)

14. Karpathy, A., Fei-Fei, L.: Deep visual-semantic alignments for generating image descriptions. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3128–3137 (2015)

15. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: International conference on machine learning. pp. 19730–19742. PMLR (2023)

16. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: International conference on machine learning. pp. 12888–12900. PMLR (2022)

17. Liang, V.W., Zhang, Y., Kwon, Y., Yeung, S., Zou, J.Y.: Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. Advances in Neural Information Processing Systems 35, 17612–17625 (2022)

18. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014)

19. Lu, J., Batra, D., Parikh, D., Lee, S.: Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems 32 (2019)

20. Peyré, G., Cuturi, M.: Computational optimal transport: With applications to data science. Now Foundations and Trends (2019)

21. Pham, K., Le, K., Ho, N., Pham, T., Bui, H.: On unbalanced optimal transport: An analysis of sinkhorn algorithm. In: International Conference on Machine Learning. pp. 7673–7682. PMLR (2020)

22. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

23. Schrodi, S., Hofmann, D.T., Argus, M., Fischer, V., Brox, T.: Two efects, one trigger: On the modality gap, object bias, and information imbalance in contrastive vision-language models. In: International Conference on Learning Representations. vol. 2025, pp. 27836–27864 (2025)

24. Székely, G.J., Rizzo, M.L.: Energy statistics: A class of statistics based on distances. Journal of statistical planning and inference 143(8), 1249–1272 (2013)

25. Villani, C., et al.: Optimal transport: old and new, vol. 338. Springer (2009)

26. Yaras, C., Chen, S., Wang, P., Qu, Q.: Explaining and mitigating the modality gap in contrastive multimodal learning. arXiv preprint arXiv:2412.07909 (2024)

27. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pretraining. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11975–11986 (2023)