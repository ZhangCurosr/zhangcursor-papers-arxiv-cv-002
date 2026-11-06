# How Far Can Untrained Neural Networks Go in Anomaly Detection? From Tables to Pixels

Hyojun Yi, Geunyoung Kim, Hogeon Seo, and Seunghyoung Ryu, Member, IEEE

Abstract—Anomaly detection (AD) is often framed as a representationlearning problem, where strong performance is expected to require learned features or AD-specific optimization. However, this view can overlook the intrinsic separability of anomalies from normal references, even in the absence of learned representations. We therefore investigate how far untrained neural networks can go in anomaly detection. To this end, we propose PRISM, a framework combining fixed random nonlinear features with normal-reference Mahalanobis scoring for tabular detection and dense visual localization. Across 47 tabular datasets in ADBench, PRISM-T achieves the highest mean AUROC (86.4%) and the second-highest mean AUPRC (65.9%) among the compared methods. On MVTec AD and VisA, covering 27 visual categories, PRISM-V achieves mean image-level and pixel-level AUROCs of 90.4% and 96.4%, respectively. Notably, both models achieve these results without neural weight optimization or pretrained representations. A theoretical analysis further characterizes how nonlinear transformations affect the mean and covariance contributions to the Mahalanobis score gap, while empirical analyses examine the resulting score separability. These findings establish PRISM as both a competitive anomaly detector and a prior-to-learning baseline for assessing the added value of learned representations in anomaly detection.

Index Terms—Anomaly detection, random neural features, untrained neural networks, Mahalanobis scoring, visual anomaly localization.

## 1 INTRODUCTION

A <sup>NOMALY</sup> <sup>detection</sup> <sup>(AD)</sup> <sup>is</sup> <sup>fundamentally</sup> <sup>a</sup> <sup>problem</sup> <sup>of</sup>identifying deviation from normality. Unlike conventional predictive learning, anomalies need not be observed during model construction; a model may instead characterize normal data and determine how strongly a new observation departs from that reference. Modern AD methods nevertheless devote substantial modeling effort to constructing representations or scoring spaces in which deviations from normality become easier to expose. Depending on the formulation, this may involve learning compact representations of normal data through reconstruction or one-class objectives [1], [2], characterizing local neighborhoods or normal-data geometry [3], [4], modeling the normal distribution through generative or score-based objectives [5], [6], or leveraging pretrained representations in which semantic or structural deviations become more discernible [7]–[9]. Such approaches reflect a common emphasis on learned representations as a key source of anomaly separability.

Yet a simpler possibility remains. If abnormality is defined relative to normal data, some anomaly evidence may already be intrinsic to the data, even before representation learning. Intuitively, we often sense that “something looks wrong” before we can precisely identify which rule or regularity has been violated. We therefore ask a basic but easily overlooked question: how far can anomaly detection go before learning?

This prior-to-learning perspective was partially explored in an earlier study on untrained neural networks (UNNs), which showed that randomly initialized neural features combined with Mahalanobis scoring can detect anomalies without neural weight optimization [10]. However, the evidence remained limited in both breadth and granularity: the tabular evaluation covered only a small set of datasets, while the visual experiments were restricted to flattened MNIST and F-MNIST with image-level scoring. It therefore remained unclear whether prior-to-learning separability could support competitive detection across diverse tabular datasets or enable dense visual localization.

To investigate this question, we propose PRISM (Priorto-learning Random Intrinsic Scoring Model), a framework that exploits intrinsic anomaly evidence through frozen, randomly initialized neural features and normal-reference Mahalanobis scoring. At its core, PRISM is built upon a simple and lightweight PRISM block consisting of a randomly initialized linear operator, normalization, and kinked nonlinear activation. This design principle is instantiated in two modality-specific models: PRISM-T for global tabular anomaly detection and PRISM-V for dense visual anomaly localization. Neither model optimizes neural weights or relies on pretrained representations. For tabular AD, PRISM-T achieves competitive performance against recent state-ofthe-art detectors across 47 ADBench datasets. For visual AD, PRISM-V achieves strong image-level detection and extends prior-to-learning AD to spatially coherent pixel-level localization on industrial anomaly benchmarks. These results demonstrate the potential of prior-to-learning anomaly detection across tabular and visual domains, establishing PRISM as a competitive detector built on lightweight, untrained neural feature extractors and a reference for assessing the added value of learned representations.

Our main contributions are summarized as follows.

• To examine how far anomaly detection can go before learning, we introduce PRISM, a framework combining random neural features with Mahalanobis scoring. Through modality-specific implementations, PRISM extends untrained neural anomaly detection from tabular data to visual anomaly detection and localization.

• We conduct a comprehensive evaluation from tables to pixels, spanning 47 tabular datasets from ADBench and 27 visual categories from MVTec AD and VisA. For tabular AD, PRISM-T achieves the highest mean AUROC and the second-highest mean AUPRC among the compared detectors, while PRISM-V demonstrates strong image-level detection and pixel-level localization. Both models achieve these results using fully untrained neural networks with substantially fewer parameters than the compared stateof-the-art neural AD models.

• We investigate the mechanisms underlying prior-tolearning anomaly separability through theoretical and empirical analyses. The theory characterizes how nonlinear feature transformations affect the mean and covariance contributions to the expected Mahalanobis score gap. Experiments further examine the resulting score separability and its relationship with detection performance across both modalities.

The remainder of this paper is organized as follows. Section 2 reviews related work. Sections 3 and 4 introduce the PRISM framework and provide a theoretical analysis of anomaly score separation. Experimental results are presented in Section 5, with further analysis in Section 6. Finally, Section 7 concludes the paper.

## 2 RELATED WORK

## 2.1 Anomaly detection and tabular benchmarks

Tabular AD has long relied on diverse classical approaches, with methods such as Isolation forest, kNN, LOF, and OCSVM continuing to serve as competitive baselines [11], [12]. ADBench [12] further standardized large-scale tabular evaluation and highlighted the difficulty arising from substantial variation in data structure and anomaly characteristics. Alongside these classical approaches, neural methods have increasingly been explored to capture more complex nonlinear structures in tabular data. Recent neural tabular AD methods extend these classical principles by learning representations, nonlinear dependencies, and relational structures. Some methods learn representations through contrastive, masked, or disentangled objectives [13]–[15]; others refine outlier scoring through neighborhood- or graph-based modeling, as in LUNAR [3]. State-of-theart approaches have increasingly adopted perturbationbased objectives, modeling normality through responses to noise or time-conditioned transformations. This trend spans diffusion-time estimation, score matching, and flowmatching-inspired approaches, as exemplified by DTE [5], K-DSM [16], and TCCM [6]. Emerging work further explores LLM- or foundation-model-inspired tabular detectors [17]. Despite their methodological diversity, most recent highperforming approaches rely on neural modeling and taskspecific training to construct anomaly representations or scoring functions.

## 2.2 Visual anomaly detection and localization

Early deep visual AD methods often treated an image as a single observation, using reconstruction, generative modeling, or one-class representation learning to produce image-level anomaly scores or coarse residual-based maps [18]–[21]. Industrial visual AD later shifted its emphasis toward dense localization, modeling normality at the patch or spatial-location level rather than treating each image as a single observation. Representative methods such as SPADE [22], PaDiM [7], and PatchCore [4] established a pretrained feature paradigm based on nearest-neighbor retrieval, location-wise Gaussian modeling, or memory-bank comparison. This reliance on pretrained visual representations has more recently extended to foundation models, including vision-language models and self-supervised visual backbones, as exemplified by WinCLIP [23], AnomalyDINO [8], and SubspaceAD [9]. Although such methods are commonly described as training-free, they avoid taskspecific feature learning while still relying on representations acquired through large-scale pretraining. We use the term prior-to-learning to distinguish the setting considered in PRISM, where the neural feature extractor is randomly initialized and remains fixed without feature learning or pretraining. Accordingly, PRISM-V investigates visual anomaly detection and dense localization using only untrained visual representations.

## 2.3 Random features and untrained networks

Although random features may seem to lack learned semantic or task-specific information, fixed random transformations have long been studied for the useful geometric structure they can induce. For example, the Johnson– Lindenstrauss lemma shows that random linear projections can approximately preserve pairwise Euclidean distances [24]. Likewise, randomly weighted neural networks have been studied as nonlinear embeddings that reshape distances and angles [25]. This geometric perspective has also been extended to randomly initialized CNNs, including the effects of convolution and ReLU on angular preservation and contraction [26]. These studies establish that random neural mappings can preserve or reshape useful geometric structure, but do not directly address whether the resulting representations expose anomaly separability. In the tabular domain, UNN [10] provided empirical evidence that nonlinear features from a randomly initialized multilayer perceptron (MLP), combined with Mahalanobis scoring, can support competitive anomaly detection without training. DIF [27] employs an ensemble of randomly initialized MLPs to construct nonlinear feature spaces where isolation forests are applied for anomaly scoring. In visual AD, prior studies have noted that random weights can yield above-chance performance [28], [29], but only in auxiliary or ablation analyses rather than as systematic investigations of fully untrained detection. Despite these findings, whether prior-tolearning representations can support competitive detection across diverse tabular problems and dense visual anomaly localization remains underexplored. PRISM extends this line of inquiry through a unified investigation of large-scale tabular detection and dense visual localization, together with modality-specific design and a score-level analysis of prior-to-learning separability.

![](images/d368f5dfc12a2ca496d067166cbaa3d007d11681ef113ad9f15196732938f0d2.jpg)  
Fig. 1. PRISM framework. (a) Each PRISM block (PB) is randomly initialized once and kept fixed. (b) PRISM-T uses a single PB, and scores global Mahalanobis deviation from the closed-form reference statistics. (c) PRISM-V interleaves five PBs with BlurPool downsampling, aligns and concatenates PBs 2–5, and produces a dense anomaly map.

## 3 PRIOR-TO-LEARNING ANOMALY DETECTION 3.1 Problem Setting

We consider semi-supervised AD, where only normal samples are available during model construction. Let $\chi _ { \mathrm { r e f } } \ =$ $\mathbf { \dot { \{ } }  x _ { i } \} _ { i = 1 } ^ { N }$ denote the normal reference set. Given an observation x, the goal is to assign an anomaly score $A ( x ) \in \mathbb { R } ,$ where larger values indicate greater abnormality.

In tabular AD, each observation is a vector $x ~ \in ~ \mathbb { R } ^ { d }$ For an image I, let Ω denote a spatial scoring grid and $x _ { u } = \mathrm { v e c } ( \bar { \Pi _ { u } } ( I ) )$ the receptive-field patch associated with $\in \Omega$ , where $\Pi _ { u }$ extracts the patch centered at u. A feature extractor $g _ { \theta }$ maps an observation to $z ~ = ~ g _ { \boldsymbol { \theta } } ( x ) ~ \in ~ \mathbb { R } ^ { m }$ For images, the same mapping is shared across locations, yielding $z _ { u } = g _ { \theta } ( x _ { u } )$ . Applying the common anomaly score across Ω produces the location-wise score map

$$
A ( I ) = \{ A _ { u } ( I ) \} _ { u \in \Omega } , \qquad A _ { u } ( I ) = A ( x _ { u } ) .\tag{1}
$$

This provides a common formulation for tabular and visual AD. The score map can be mapped to the input resolution for pixel-level localization, while image-level anomaly scores can be obtained by aggregating spatial scores.

## 3.2 Formulation of PRISM

PRISM uses fixed random transformations to reveal intrinsic normal–anomalous separability without learning feature weights. This principle leads to two modality-specific instantiations: PRISM-T for tabular AD and PRISM-V for location-wise visual AD. In each instantiation, one or more PRISM blocks (PBs) induce a nonlinear random transformation, after which the resulting features are scored by their Mahalanobis distance from reference statistics. Figure 1 summarizes the shared PB design and the modality-specific pipelines of PRISM-T and PRISM-V.

For an input representation $h _ { \ell - 1 } ,$ the ℓ-th PB is defined as a composition of a randomly initialized linear operator, normalization, and a kinked nonlinearity:

$$
\begin{array} { r } { \mathrm { P B } _ { \ell } ( h _ { \ell - 1 } ) = \sigma ( \mathrm { N o r m } \left( \mathrm { L i n } _ { \theta _ { \ell } } ( h _ { \ell - 1 } ) \right) ) , } \end{array}\tag{2}
$$

where $\operatorname { L i n } _ { \theta _ { \ell } }$ denotes a linear operator with weights $\theta _ { \ell } ,$ Norm denotes a normalization layer without learnable affine parameters, and σ is the LeakyReLU activation function. All weights $\theta _ { \ell }$ are sampled once at initialization and kept fixed; no backpropagation, pretraining, or task-specific optimization is applied.

The feature extractor $g _ { \theta }$ is formed from one or more PBs. For a given scoring unit, the normal-reference features $\mathcal { Z } _ { \mathrm { r e f } } = \{ g _ { \theta } \breve { ( } x _ { i } ) \ | \ x _ { i } \in \breve { \mathscr { X } _ { \mathrm { r e f } } } \}$ define a mean $\mu$ and covariance Σ. The Mahalanobis anomaly score is

$$
A ( x ) = { \sqrt { \left( g _ { \theta } ( x ) - \mu \right) ^ { \top } ( \Sigma + \lambda I ) ^ { - 1 } \left( g _ { \theta } ( x ) - \mu \right) } } ,\tag{3}
$$

where λ regularizes covariance inversion. The reference statistics are global for PRISM-T and location-specific for PRISM-V. Thus, PRISM assigns a larger anomaly score when the randomly transformed representation deviates further from the normal-reference distribution.

PRISM for Tabular AD: The tabular version, denoted PRISM-T, applies a single-PB extractor to vector-valued samples and computes one Mahalanobis anomaly score for each instance. In this setting, Lin(·) in Eq. (2) is instantiated as a fully connected layer. Unlike bounded image pixel values, tabular variables may contain meaningful magnitude information. Instance Normalization (IN) [30] removes sample-specific mean and scale, emphasizing relative feature patterns but potentially suppressing anomalies expressed through unusual magnitudes. Accordingly, Norm(·) applies IN to one half of the feature dimensions and Batch Normalization (BN) [31] to the other half, preserving both pattern- and magnitude-level deviations. BN statistics are estimated once from the normal reference set and fixed during inference.

PRISM for Visual AD: PRISM-V extends the same PBbased random feature extraction to visual anomaly localization. For PRISM-V, Lin(·) in Eq. (2) is instantiated as a convolutional layer, preserving spatial correspondence while sharing the same fixed random filters across locations. Norm(·) uses IN. To capture visual anomalies at different spatial extents, PRISM-V interleaves PBs with anti-aliased BlurPool [32] downsampling to construct multi-scale random descriptors. Letting $H _ { 0 } = I .$ , the resulting hierarchy is

defined as

$$
F _ { \ell } = \mathrm { P B } _ { \ell } ( H _ { \ell - 1 } ) , \qquad H _ { \ell } = \mathrm { B l u r P o o l } ( F _ { \ell } ) .\tag{4}
$$

Selected PB outputs are bilinearly aligned to the common scoring grid Ω and concatenated into the location-wise descriptor $z _ { u } = [ \tilde { F } _ { \ell , u } ] _ { \ell \in \mathcal { T } } ,$ , where $\tilde { F } _ { \ell }$ is the aligned feature map from stage ℓ. At each $u \in \Omega$ , PRISM-V estimates $\left( \mu _ { u } , \Sigma _ { u } \right)$ and applies Eq. (3) to $z _ { u } = g _ { \theta } ( x _ { u } )$ , yielding the location-wise score $A _ { u } ( \bar { I ) = A ( x _ { u } ) }$ . The resulting score map $A ( I ) = \{ A _ { u } ( I ) \} _ { u \in \Omega }$ is upsampled to the input resolution $( 2 2 4 \times 2 2 4 )$ and Gaussian-smoothed to produce the final pixel-level anomaly map $A _ { \mathrm { p i x } } ( I )$ . The image-level anomaly score is obtained by applying empirical tail value-at-risk (TVaR) aggregation to $A _ { \mathrm { p i x } } ( I )$

## 4 ANALYSIS OF ANOMALY SCORE SEPARATION 4.1 Score-Level Separation

Let $X _ { \mathrm { N } } \sim P _ { \mathrm { N } }$ and $X _ { \mathrm { A } } \sim P _ { \mathrm { A } }$ denote normal and abnormal samples, with $c \in \{ \mathrm { N } , \mathrm { A } \}$ indexing the class. For a fixed PB extractor g and observation space $o ,$ define $X _ { c } ^ { o } ~ = ~ X _ { c }$ and $X _ { c } ^ { g } ~ = ~ g ( X _ { c } )$ . With space index $r ~ \in ~ \{ o , g \}$ , classconditional mean and covariance in space r are $\mu _ { c , r }$ <sub>r</sub> and $\Sigma _ { c , r } ,$ respectively. The normal-class index N is omitted when no ambiguity arises, so that $\mu _ { r } ~ = ~ \mu _ { \mathrm { N } , r }$ and $\begin{array} { r } { \Sigma _ { r } = \Sigma _ { \mathrm { N } , r } . } \end{array}$ For the population analysis, assume that $\Sigma _ { r }$ is nonsingular and set $\begin{array} { r } { \dot { \boldsymbol { M _ { r } } } = \boldsymbol { \Sigma } _ { \boldsymbol { r } } ^ { - 1 } } \end{array}$ . In the implementation, the mean and covariance are estimated from the reference set, and the ridge-regularized metric in Eq. (3) is retained for numerical stability.

The class-conditional squared Mahalanobis score is

$$
S _ { c } ^ { r } = ( X _ { c } ^ { r } - \mu _ { r } ) ^ { \top } M _ { r } ( X _ { c } ^ { r } - \mu _ { r } ) , \qquad c \in \{ \mathrm { N } , \mathrm { A } \} .\tag{5}
$$

To quantify normal–abnormal separation in space $r ,$ define the standardized score separation as

$$
G _ { r } = \frac { \Delta _ { r } } { \sqrt { V _ { r } } } = \frac { \mathbb { E } [ S _ { \mathrm { A } } ^ { r } ] - \mathbb { E } [ S _ { \mathrm { N } } ^ { r } ] } { \sqrt { \mathrm { V a r } ( S _ { \mathrm { A } } ^ { r } ) + \mathrm { V a r } ( S _ { \mathrm { N } } ^ { r } ) } } , \qquad V _ { r } > 0 .\tag{6}
$$

Here, $\Delta _ { r }$ is the expected score difference between abnormal and normal samples, and $V _ { r }$ is the sum of their within-class score variances. Thus, $G _ { r }$ measures the score gap relative to the combined score dispersion, $\sqrt { V _ { r } }$ . Its squared value corresponds to the Fisher discriminant ratio, while the sign of $\Delta _ { r }$ is retained since larger scores conventionally indicate greater abnormality. Note that $G _ { r }$ characterizes score separation through distributional moments rather than the pairwise rankings underlying AUROC.

Proposition 1 (Decomposition of the expected score gap). For the squared Mahalanobis score in Eq. (5),

$$
\begin{array} { r l } & { \Delta _ { r } = \underbrace { \mathrm { t r } \left[ M _ { r } ( \Sigma _ { \mathrm { A } , r } - \Sigma _ { r } ) \right] } _ { c o v a r i a n c e ~ c o n t r i b u t i o n , ~ \Delta _ { r } ^ { c o v } } } \\ & { ~ + \underbrace { ( \mu _ { \mathrm { A } , r } - \mu _ { r } ) ^ { \top } M _ { r } ( \mu _ { \mathrm { A } , r } - \mu _ { r } ) } _ { m e a n ~ c o n t r i b u t i o n , ~ \Delta _ { r } ^ { m e a n } } . } \end{array}\tag{7}
$$

The result follows from the quadratic-form identity

$$
\mathbb { E } [ S _ { c } ^ { r } ] = \mathrm { t r } ( M _ { r } \Sigma _ { c , r } ) + ( \mu _ { c , r } - \mu _ { r } ) ^ { \top } M _ { r } ( \mu _ { c , r } - \mu _ { r } ) .\tag{8}
$$

The covariance contribution measures the difference between abnormal and normal covariance weighted by $M _ { r }$

The mean contribution is the squared Mahalanobis distance between their means and is therefore nonnegative. Dividing Eq. (7) by $\sqrt { V _ { r } }$ gives an exact population-level decomposition,

$$
G _ { r } = G _ { r } ^ { \mathrm { c o v } } + G _ { r } ^ { \mathrm { m e a n } } = \frac { \Delta _ { r } ^ { \mathrm { c o v } } } { \sqrt { V _ { r } } } + \frac { \Delta _ { r } ^ { \mathrm { m e a n } } } { \sqrt { V _ { r } } } .\tag{9}
$$

These terms quantify covariance- and mean-based contributions relative to the combined score dispersion, $\sqrt { V _ { r } }$ . In particular, changes in $G _ { r } ^ { \mathrm { c o v } }$ need not correspond to changes of the same magnitude or sign in the unstandardized covariance gap.

Remark 1 (Invariance under invertible linear transformation). Under population-level Mahalanobis scoring, any fixed invertible linear transformation, including a randomly drawn one, preserves the sample-wise scores and hence $G _ { r } .$ This invariance does not generally extend to non-invertible projections, nonlinear transformations, or ridge-regularized finite-sample metrics.

## 4.2 Local Relation-Violating Anomalies

Assumption 1 (Local relation-violating anomaly model). A normal sample is represented as

$$
X _ { \mathrm { N } } = f ( \zeta ) + \varepsilon ,\tag{10}
$$

where $f ( \zeta )$ represents an underlying nominal relation and ε captures admissible normal variation. An abnormal counterpart is obtained through

$$
X _ { \mathrm { A } } = X _ { \mathrm { N } } + \rho , \qquad \mathbb { E } [ \rho \mid X _ { \mathrm { N } } ] = 0 ,\tag{11}
$$

where $\rho$ is a relation-violating residual with conditional covariance $C _ { \rho } ( x ) = \mathrm { C o v } ( \rho \mid X _ { \mathrm { N } } ^ { } = x )$ . Let $\bar { C } _ { \rho } = \mathbb { E } [ C _ { \rho } ( X _ { \mathrm { N } } ) ]$

This stylized model isolates anomalies that introduce additional conditional variation without a systematic conditional mean shift; it is not intended to characterize every anomaly in the benchmarks. In the observation space, the zero-conditional-mean property yields

$$
\mu _ { \mathrm { A } , o } = \mu _ { o } , \qquad \Sigma _ { \mathrm { A } , o } - \Sigma _ { o } = \bar { C } _ { \rho } \succeq 0 .\tag{12}
$$

Consequently, Proposition 1 implies

$$
\Delta _ { o } ^ { \mathrm { m e a n } } = 0 , \qquad \Delta _ { o } = \Delta _ { o } ^ { \mathrm { c o v } } = \mathrm { t r } ( M _ { o } \bar { C } _ { \rho } ) \geq 0 ,\tag{13}
$$

with a strict inequality when $\bar { C } _ { \rho } \neq 0$ . Hence, the raw Mahalanobis score already contains covariance-based anomaly evidence under this model. The question is how a fixed nonlinear transformation changes the same evidence and whether that change improves standardized separation.

## 4.3 Nonlinear Mechanisms of Score-Gap Variation

For the coupled normal–abnormal samples in Assumption 1, define their transformed displacement as

$$
D _ { g } = X _ { \mathrm { A } } ^ { g } - X _ { \mathrm { N } } ^ { g } = g ( X _ { \mathrm { N } } + \rho ) - g ( X _ { \mathrm { N } } ) ,\tag{14}
$$

so that $X _ { \mathrm { A } } ^ { g } = X _ { \mathrm { N } } ^ { g } + D _ { g }$

Proposition 2 (Covariance contribution under nonlinear transformation). Under the above displacement representation,

$$
\begin{array} { r l } & { \Sigma _ { \mathrm { A } , g } - \Sigma _ { g } = \mathrm { C o v } ( D _ { g } ) + \mathrm { C o v } ( X _ { \mathrm { N } } ^ { g } , D _ { g } ) } \\ & { \qquad + \mathrm { C o v } ( D _ { g } , X _ { \mathrm { N } } ^ { g } ) . } \end{array}\tag{15}
$$

The covariance contribution to the expected score gap is therefore

$$
\begin{array} { r } { \Delta _ { g } ^ { \mathrm { c o v } } = \underbrace { \mathrm { t r } \left[ M _ { g } \mathrm { C o v } ( D _ { g } ) \right] } _ { d i s p l a c e m e n t s p r e a d } + \underbrace { 2 \mathrm { t r } \left[ M _ { g } \mathrm { C o v } ( X _ { \mathrm { N } } ^ { g } , D _ { g } ) \right] } _ { i n t e r a c t i o n } . } \end{array}\tag{16}
$$

The displacement-spread term is nonnegative and measures additional variation in the Mahalanobis geometry. The interaction term captures the dependence between the transformed normal representation and its anomaly-induced displacement and can have either sign. Under Assumption 1, the corresponding observation-space interaction vanishes because $\mathrm { C o v } ( X _ { \mathrm { N } } , \bar { \rho } ) = 0$ , but the interaction need not vanish after nonlinear transformation. Specifically,

$$
\mathrm { t r } [ M _ { g } \mathrm { C o v } ( X _ { \mathrm { N } } ^ { g } , D _ { g } ) ] = \mathbb { E } \Big [ ( X _ { \mathrm { N } } ^ { g } - \mu _ { g } ) ^ { \top } M _ { g } D _ { g } \Big ] .\tag{17}
$$

A positive interaction indicates that transformed displacements tend to align with existing deviations from the normal mean in the Mahalanobis geometry.

The transformed displacement can also create a featuremean difference, $\mu _ { \mathrm { A } , g } \ - \ \mu _ { g } \ = \ \mathbb { E } [ D _ { g } ]$ , giving the mean contribution

$$
\Delta _ { g } ^ { \mathrm { m e a n } } = \mathbb { E } [ D _ { g } ] ^ { \top } M _ { g } \mathbb { E } [ D _ { g } ] \ge 0 .\tag{18}
$$

The following identity illustrates how an activation used in PRISM can produce a feature-mean difference even when pre-activation means are unchanged.

Proposition 3 (LeakyReLU-induced feature-mean gap). Let $U _ { \mathrm { N } }$ and $U _ { \mathrm { A } }$ denote normal and abnormal inputs to LeakyReLU activation $\sigma _ { \alpha } ,$ , where $0 \leq \alpha < 1$ . Then

$$
\begin{array} { r l } & { \mathbb { E } [ \sigma _ { \alpha } ( U _ { \mathrm { A } } ) ] - \mathbb { E } [ \sigma _ { \alpha } ( U _ { \mathrm { N } } ) ] } \\ & { \quad = \displaystyle \frac { 1 + \alpha } { 2 } \big ( \mathbb { E } [ U _ { \mathrm { A } } ] - \mathbb { E } [ U _ { \mathrm { N } } ] \big ) } \\ & { \quad \quad + \displaystyle \frac { 1 - \alpha } { 2 } \big ( \mathbb { E } | U _ { \mathrm { A } } | - \mathbb { E } | U _ { \mathrm { N } } | \big ) . } \end{array}\tag{19}
$$

This follows from $\begin{array} { r } { \sigma _ { \alpha } ( u ) = \frac { 1 + \alpha } { 2 } u + \frac { 1 - \alpha } { 2 } | u | } \end{array}$ . For example, let $U _ { \mathrm { N } } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ and $U _ { \mathrm { { A } } } = U _ { \mathrm { { N } } } + \eta$ with independent $\eta \sim \mathcal { N } ( 0 , \tau ^ { 2 } )$ . Their preactivation means coincide, but

$$
\mathbb { E } | U _ { \mathrm { A } } | - \mathbb { E } | U _ { \mathrm { N } } | = \sqrt { \frac { 2 } { \pi } } \left( \sqrt { \sigma ^ { 2 } + \tau ^ { 2 } } - \sigma \right) > 0 , \qquad \tau > 0 .\tag{20}
$$

Thus, LeakyReLU can convert a change in feature variation into a nonzero feature-mean difference and hence a nonnegative quadratic mean contribution when retained in the scored representation. Note that these mechanisms do not require weight optimization, allowing randomly initialized neural networks to modify the expected score gap.

## 4.4 Implications for Standardized Score Separation

The preceding results characterize how nonlinear transformations modify the covariance and mean contributions to the expected score gap. However, as shown in Eq. (9), the standardized separation $G _ { r }$ also depends on the combined score dispersion, $\sqrt { V _ { r } }$ . Since the behavior of $V _ { r }$ under nonlinear transformations depends on the underlying distributions, establishing a general improvement in $\breve { G _ { r } }$ remains challenging. The resulting separation is therefore examined empirically in the subsequent analysis section.

TABLE 1  
Architecture and configuration of PRISM-T and PRISM-V.
<table><tr><td>Setting</td><td>PRISM-T</td><td>PRISM-V</td></tr><tr><td>Linear operator</td><td>Fully connected</td><td>Convolution</td></tr><tr><td>Number of PBs</td><td>1</td><td>5</td></tr><tr><td>Feature dimension</td><td>512</td><td>512</td></tr><tr><td>Kernel size</td><td></td><td>[3,3,3,1,1]</td></tr><tr><td>Initialization</td><td>Uniform</td><td>He normal</td></tr><tr><td>Normalization</td><td>Instance + Batch</td><td>Instance</td></tr><tr><td>Activation</td><td>LeakyReLU (α = 0.01)</td><td></td></tr><tr><td>Mahalanobis λ</td><td> $1 0 ^ { - 3 ^ { \prime } }$ </td><td>10⁻2</td></tr><tr><td>Scoring scope</td><td>Global</td><td>Location-wise</td></tr></table>

## 5 EXPERIMENTS

## 5.1 Experimental Setup

Datasets and Evaluation: We evaluate PRISM on three widely adopted anomaly detection benchmarks: AD-Bench [12] for tabular AD, and MVTec AD [33] and VisA [34] for visual AD. Tabular experiments cover the 47 classical datasets in ADBench, excluding the 10 additional CV/NLP datasets represented by pretrained embeddings to maintain a prior-to-learning evaluation setting. Following recent normal-only protocols, 50% of the normal samples are randomly selected as the training set for learning-based baselines [6], [15] and as the reference set for PRISM-T. The remaining normal samples and all anomalous samples constitute the test set. AUROC and AUPRC are evaluated, with mean scores and average ranks reported across datasets.

Visual experiments cover 15 categories of industrial objects and textures in MVTec AD and 12 categories in VisA, including complex multi-instance scenes. For MVTec AD, the defect-free training split serves as the normal reference set, and evaluation uses the official test split. For VisA, we follow the official one-class protocol, using 90% of normal images for reference and the remaining normals plus all anomalies for testing. Image-level performance is evaluated using AUROC and AUPRC, and pixel-level localization using AUROC and area under the per-region overlap curve (AUPRO) [35]. All metrics range from 0 to 1, with higher values indicating better performance. For both tabular and visual evaluations, experimental results are averaged over five random seeds. Throughout the tables, the best and second-best results are highlighted in bold and underlined. For brevity, some tables use AUC, AUP, and PRO for AU-ROC, AUPRC, and AUPRO, respectively, with I- and Pindicating image- and pixel-level metrics.

Architecture and Hyperparameters: Table 1 summarizes the modality-specific architectural designs and the resulting hyperparameter settings. A single global configuration is selected for each modality using three preselected development datasets or categories (PRISM-T: Pima, Fault, and SpamBase; PRISM-V: cable, hazelnut, and carpet from MVTec AD) and then fixed across all evaluations without perdataset tuning.

Baseline Models: Tabular AD performance is compared against 14 detectors, including recent state-of-the-art methods such as DTE-NP [5], AnoDDAE [36], TCCM [6], and DenoiserAD [37]. The comparison also covers strong baselines from prior large-scale benchmarks, such as LUNAR [3] and KDE [38], as well as methods using untrained neural features, including DIF [27] and UNN [10]. Due to computational resource constraints, KDE, OCSVM, and DenoiserAD are fitted using 10,000 randomly selected normal training samples on the nine datasets with over 50,000 samples (backdoor, smtp, skin, celeba, census, cover, fraud, http, and donors). The test sets remain unchanged.

TABLE 2  
Summary of the compared anomaly detection methods, neural network requirements, and scoring mechanisms.
<table><tr><td>Method Description</td><td></td><td></td><td>NN Tr.</td></tr><tr><td colspan="2">Tabular baselines</td><td></td><td>Sc.</td></tr><tr><td>KDE</td><td>Kernel density estimation of normal data.</td><td></td><td>LIK</td></tr><tr><td>AE</td><td>Autoencoder trained on normal data.</td><td>√</td><td>T REC</td></tr><tr><td>OCSVM</td><td>Maximum-margin one-class decision boundary.</td><td></td><td>DIST</td></tr><tr><td>IForest</td><td>Random isolation trees with path-length scoring.</td><td></td><td>ISO</td></tr><tr><td>DSVDD</td><td>Distance to a learned hypersphere center.</td><td>√</td><td>T DIST</td></tr><tr><td>LUNAR</td><td>GNN-weighted nearest-neighbor distances.</td><td>√</td><td>T NN</td></tr><tr><td>DIF</td><td>Isolation forests on random neural features.</td><td>√</td><td>ISO</td></tr><tr><td>UNN</td><td>Mahalanobis distance on random neural features.</td><td>√</td><td>MD</td></tr><tr><td>DTE-NP</td><td>Diffusion-time estimation via kNN.</td><td>1</td><td>NN</td></tr><tr><td>Disent-AD</td><td>Disentangled correlations among attribute subsets.</td><td>√</td><td>T REC</td></tr><tr><td>AnoDDAE</td><td>Diffusion-scheduled denoising reconstruction.</td><td>√</td><td>T REC</td></tr><tr><td>TCCM</td><td>Time-conditioned contraction matching deviation.</td><td>L</td><td>T MS</td></tr><tr><td>K-DSM</td><td>Kurtosis-guided score matching and gradient- norm scoring.</td><td></td><td>T MS</td></tr><tr><td colspan="2">DenoiserAD Perturbation-based denoising deviation.</td><td>1</td><td>T MS</td></tr><tr><td colspan="2">Visual baselines</td><td></td><td></td><td></td></tr><tr><td>VAE</td><td>Variational autoencoder reconstruction.</td><td>1</td><td>T</td><td>REC</td></tr><tr><td>GANomaly</td><td>Encoder-decoder-encoder GAN with latent recon- struction discrepancy.</td><td></td><td>T</td><td>REC</td></tr><tr><td>SPADE</td><td>kNN correspondence over multi-scale pretrained ResNet feature pyramids.</td><td>√</td><td>P</td><td>NN</td></tr><tr><td>PaDiM</td><td>Patch-wise Gaussian modeling of pretrained ResNet features with Mahalanobis scoring.</td><td>√</td><td>P</td><td>MD</td></tr><tr><td>PatchCore</td><td>Coreset-subsampled memory bank of pretrained ResNet features with nearest-neighbor scoring.</td><td></td><td>P</td><td>NN</td></tr><tr><td>WinCLIP+</td><td>CLIP-based window-wise text-image alignment augmented with normal reference features.</td><td></td><td>P</td><td>MS</td></tr><tr><td>Anomaly DINO</td><td>Patch-level nearest-neighbor matching on frozen DINOv2 features.</td><td></td><td>P</td><td>NN</td></tr><tr><td>Subspace AD</td><td>PCA subspace fitted to frozen DINOv2 patch fea- √ tures, with reconstruction residual scoring.</td><td></td><td>P</td><td>REC</td></tr></table>

NN: neural network usage (✓: yes, –: no); Tr.: neural weight training (T: tasktrained, P: pretrained, –: none); Sc.: anomaly scoring function (NN: nearestneighbor, REC: reconstruction, DIST: distance or margin, MD: Mahalanobis distance, ISO: isolation, LIK: likelihood-based, MS: method-specific).

In the visual AD setting, evaluation is conducted against eight baselines, including generative models (VAE [39] and GANomaly [19]) and pretrained-feature methods. The latter include conventional CNN-based approaches (SPADE [22], PaDiM [7], and PatchCore [4]) and recent foundationmodel-based methods (WinCLIP+ [23], AnomalyDINO [8], and SubspaceAD [9]). For VAE, we adopt a CNN encoder and a symmetric decoder based on the PRISM-V architecture and generate pixel-level anomaly maps from reconstruction errors following prior studies [40], [41]. The hyperparameters of VAE and GANomaly are selected using the same three MVTec AD development categories used for PRISM-V and fixed across both visual benchmarks. All other baselines use publicly available implementations from PyOD, scikit-learn, or the original studies with their provided or default hyperparameter settings. Table 2 provides an overview of the baseline methods, their neural network training and anomaly scoring mechanisms.

TABLE 3  
Tabular AD results on ADBench datasets in AUROC and AUPRC (%).
<table><tr><td></td><td colspan="3">AUROC</td><td colspan="4">AUPRC</td></tr><tr><td>Method</td><td>Mean</td><td>Std pH</td><td></td><td>Top-2 Mean</td><td>Std</td><td>PH</td><td>Top-2</td></tr><tr><td>PRISM-T</td><td>86.4</td><td>±0.9</td><td>一</td><td>17</td><td>65.9 ±2.5</td><td>一</td><td>12</td></tr><tr><td>KDE†</td><td>84.2</td><td>±0.7</td><td>.315</td><td>10</td><td>62.3 ±2.5</td><td>1</td><td>9</td></tr><tr><td>AE</td><td>82.9</td><td>±1.2</td><td>&lt;.001</td><td>6</td><td>59.4 ±3.5</td><td>.004</td><td>4</td></tr><tr><td>OCSVM†</td><td>79.6</td><td>±0.7</td><td>.002</td><td>6</td><td>55.0 ±2.2</td><td>.003</td><td>3</td></tr><tr><td>IForest</td><td>79.8</td><td>±1.6</td><td>&lt;.001</td><td>5</td><td>49.3 ±3.4</td><td>&lt;.001</td><td>4</td></tr><tr><td>DSVDD</td><td>74.8</td><td>±4.8</td><td>&lt;.001</td><td>0</td><td>48.4 ±7.0</td><td>&lt;.001</td><td>2</td></tr><tr><td>LUNAR</td><td>85.9</td><td>±0.9</td><td>1</td><td>16</td><td>68.0 ±2.7</td><td>1</td><td>18</td></tr><tr><td>DIF</td><td>76.2</td><td>±1.7</td><td>&lt;.001</td><td>2</td><td>48.2 ±2.5</td><td>&lt;.001</td><td>3</td></tr><tr><td>UNN</td><td>83.5</td><td>±1.1</td><td>.015</td><td>8</td><td>61.0 ±2.4</td><td>.012</td><td>8</td></tr><tr><td>DTE-NP</td><td>86.3</td><td>±0.7</td><td>1</td><td>19</td><td>65.5 ±2.2</td><td>1</td><td>12</td></tr><tr><td>Disent-AD</td><td>76.1</td><td>±2.5</td><td>&lt;.001</td><td>2</td><td>51.3</td><td>±5.2 &lt;.001</td><td>3</td></tr><tr><td>AnoDDAE</td><td>85.9</td><td>±1.1</td><td>.315</td><td>10</td><td>65.6 ±3.0</td><td>1</td><td>5</td></tr><tr><td>TCCM</td><td>84.7</td><td>±2.4</td><td>.262</td><td>12</td><td>64.5 ±4.9</td><td>1</td><td>10</td></tr><tr><td>K-DSM</td><td>84.6</td><td>±1.1</td><td>.723</td><td>13</td><td>65.2 ±3.3</td><td>1</td><td>9</td></tr><tr><td>DenoiserAD†</td><td>86.3</td><td>±1.2</td><td>.774</td><td>10</td><td>64.5</td><td>±4.3 1</td><td>11</td></tr></table>

Mean and Std denote the averages of performance and per-dataset standard deviations across seeds, respectively. p<sub>H</sub> is the Holm-adjusted two-sided Wilcoxon signed-rank p-value against PRISM-T. Top-2 counts top-two rankings across datasets. † indicates a 10K training-sample limit on large datasets.

Implementation Details: For PRISM-T, input features are z-score standardized and passed through a single PB consisting of a frozen fully connected layer, mixed instance–batch normalization, and LeakyReLU. All datasetlevel statistics are estimated solely from the reference set. The random weights are initialized from $\mathcal { U } ( - 1 / \sqrt { d } , 1 / \sqrt { d } )$ For PRISM-V, input images are resized to 256 and centercropped to 224 × 224, a resolution commonly used in visual AD [4], [7]. Baseline methods retain their original preprocessing and input resolution settings. The feature extractor comprises five convolutional PBs with BlurPool downsampling, and the convolutional weights follow He normal initialization [42]. Feature taps are fixed at $\mathcal { T } = \{ 2 , 3 , 4 , 5 \}$ . The tapped feature maps are bilinearly upsampled to a 112×112 grid and concatenated for location-wise Mahalanobis scoring. The reference mean and covariance are estimated per grid location, with covariance matrices averaged over a 3 × 3 window before regularization and inversion. Gaussian smoothing uses σ = 4, and TVaR aggregation uses a tail ratio of $\rho = 1 \%$

## 5.2 Tabular Anomaly Detection

Table 3 summarizes the results on 47 ADBench datasets in terms of mean performance with standard deviations, statistical significance, and top-two counts. Despite using entirely untrained neural features, PRISM-T achieves the highest mean AUROC of 86.4% and the second-highest mean AUPRC of 65.9%. It also records the second-highest Top-2 counts, highlighting its competitive performance across both metrics. Among the top-performing baselines, DTE-NP and DenoiserAD achieve similar mean AUROC scores of 86.3%, but lower mean AUPRC, while LUNAR achieves the highest mean AUPRC of 68.0%. Compared with previous untrained neural feature-based methods, PRISM-T improves upon UNN by 3.5% and 8.0%, and DIF by 13.4% and 36.7% in AUROC and AUPRC, respectively. To assess statistical significance, Holm-adjusted Wilcoxon signed-rank test p-values (p<sub>H</sub>) are reported for comparisons between PRISM-T and each baseline. The results show no significant differences $\left( p _ { \mathrm { H } } \ge 0 . 0 5 \right)$ from leading baselines, including DTE-NP, LUNAR, AnoDDAE, TCCM, K-DSM, and DenoiserAD, on either metric.

<table><tr><td rowspan="12">Backdoor</td><td>PRISM-T ALOI</td><td></td><td></td><td>AnoDDAE</td><td>TCCM</td><td>K-DSM</td><td>QO Disent-</td><td>DTE-NP</td><td>LUNAR</td><td>UNN</td><td>DIF</td><td></td><td>KDE</td><td>AE</td><td></td><td>OCSVM</td><td>IForest</td><td></td></tr><tr><td>9.9 72.9</td><td></td><td>28.8</td><td>8.7</td><td>8.5</td><td>12.0</td><td>6.7</td><td>13.2</td><td>21.8</td><td></td><td></td><td>7.8</td><td>10.2</td><td></td><td></td><td>7.5</td><td></td><td>DSVDD</td></tr><tr><td>Annthy.</td><td></td><td>59.1</td><td>72.1</td><td>70.6</td><td>57.7</td><td>46.7</td><td>69.6</td><td>58.9</td><td>8.2 64.6</td><td>50.3</td><td></td><td>58.5</td><td>7.6</td><td></td><td></td><td>6.5 62.9</td><td>7.1</td></tr><tr><td>81.8</td><td></td><td>87.1</td><td>71.3</td><td>85.3</td><td>85.6</td><td>5.6</td><td>63.0</td><td>89.9</td><td>85.1</td><td>72.3</td><td></td><td>37.1</td><td>56.5</td><td></td><td>57.7</td><td></td><td>57.2</td></tr><tr><td></td><td></td><td>98.5</td><td>99.1</td><td>99.3</td><td>99.0</td><td>98.2</td><td></td><td>98.9</td><td></td><td>66.8</td><td></td><td></td><td>86.7</td><td>7.6</td><td></td><td>10.0</td><td>6.6</td></tr><tr><td>BreastW 99.1</td><td></td><td>33.3</td><td>51.6</td><td>49.0</td><td></td><td></td><td>99.2</td><td></td><td>99.0</td><td></td><td></td><td>99.1</td><td>98.6</td><td></td><td>99.2</td><td>99.5</td><td>98.7</td></tr><tr><td>Campaign 49.9</td><td></td><td>80.4</td><td>79.6</td><td>83.8</td><td>27.8</td><td>36.5</td><td>49.2</td><td>42.7</td><td>50.4</td><td></td><td>38.7</td><td>47.5</td><td>46.6</td><td></td><td>49.5</td><td>45.4</td><td>44.5</td></tr><tr><td>Cardio 80.0 CTG 64.2</td><td></td><td>68.7</td><td>62.6</td><td>71.5</td><td>83.8 68.7</td><td>83.5 68.2</td><td>79.2 66.2</td><td>84.7 72.8</td><td>86.1</td><td></td><td>84.5</td><td>82.4 68.4</td><td>70.9</td><td></td><td>83.2</td><td>80.3</td><td>75.8</td></tr><tr><td>CelebA</td><td>9.6</td><td>12.3</td><td>11.3</td><td>14.5</td><td>9.4</td><td>10.9</td><td>9.1</td><td></td><td>7.0</td><td>72.1 17.4</td><td>60.2 9.1</td><td>10.5</td><td>67.0</td><td>10.3</td><td>71.6 20.2</td><td>67.9 12.4</td><td>70.9 14.1</td></tr><tr><td>Census</td><td>21.8</td><td>19.7</td><td>18.1</td><td>22.0</td><td>20.3</td><td>14.7</td><td></td><td>21.8</td><td>16.5</td><td>21.2</td><td>13.0</td><td>20.7</td><td></td><td>21.9</td><td>20.5</td><td>14.1</td><td>19.5</td></tr><tr><td>Cover 57.7</td><td></td><td>66.8</td><td>80.1</td><td>74.3</td><td>78.4</td><td>5.6</td><td></td><td>80.8</td><td>89.7</td><td>43.1</td><td>47.4</td><td>33.5</td><td></td><td>50.5</td><td>22.0</td><td>8.9</td><td>13.6</td></tr><tr><td>Donors 95.7</td><td></td><td>36.4 79.5</td><td>97.8</td><td>95.0</td><td>72.8</td><td>63.5</td><td>98.2</td><td></td><td>99.9</td><td>77.1</td><td>32.1</td><td>70.1</td><td></td><td>54.1</td><td>42.8</td><td>39.2</td><td>31.7</td></tr><tr><td>Fault 80.7 Fraud 53.5</td><td></td><td>55.1</td><td>78.0 49.2</td><td>77.1 51.8</td><td>78.3 55.6</td><td>61.2 37.2</td><td></td><td>79.8 36.7</td><td>79.3</td><td>74.6 47.7</td><td>72.2 50.7</td><td>80.9 36.6</td><td>74.3</td><td>64.2</td><td>64.8 32.5</td><td>66.0 21.3</td><td>59.6</td></tr><tr><td>Glass 43.4</td><td></td><td>34.5</td><td>34.8</td><td>35.5</td><td>32.9</td><td>19.6</td><td></td><td>31.7</td><td>58.5 34.6</td><td>32.9</td><td>32.7</td><td>30.9</td><td></td><td>24.7</td><td>24.4</td><td>20.1</td><td>24.4 20.7</td></tr><tr><td>Hepat. HTTP 99.8</td><td>50.4</td><td>48.1 92.3 99.5</td><td>49.1</td><td>56.6</td><td>46.7</td><td>55.7</td><td></td><td>55.7</td><td>53.6</td><td>43.5</td><td>46.8</td><td>55.4</td><td></td><td>53.2</td><td>53.6</td><td>54.5</td><td>52.2</td></tr><tr><td>IAds lono. Landsat</td><td>85.0 97.5</td><td>82.5 97.6</td><td>84.6</td><td>83.5</td><td>96.0</td><td>99.9 85.6</td><td>66.7 49.4</td><td>99.8 69.1</td><td>72.5 74.4</td><td>99.1 70.7</td><td>44.9 35.9</td><td>99.9 78.7</td><td></td><td>92.7 85.4</td><td>99.8 60.0</td><td>47.9 29.8</td><td>82.3 51.2</td></tr><tr><td>Letter Lymph.</td><td>54.1 73.6</td><td>63.7</td><td>97.0 58.1</td><td></td><td>97.9 40.5</td><td>97.4 64.3</td><td>96.1 33.8</td><td>98.0 61.4</td><td>97.3 65.0</td><td>97.1 42.3</td><td>94.9 44.3</td><td>97.8 55.2</td><td></td><td>94.3 41.7</td><td>97.2 32.6</td><td>90.8 42.1</td><td>87.3</td></tr><tr><td>MAGIC</td><td>90.8</td><td>71.9 89.0</td><td>80.5 92.1</td><td></td><td>60.3</td><td>73.1</td><td>14.6</td><td>49.9</td><td>58.3</td><td>33.1</td><td>23.3</td><td>55.3</td><td></td><td>36.3</td><td>20.9</td><td>16.6</td><td>30.4 13.0</td></tr><tr><td>Mammo.</td><td>87.2</td><td>88.7</td><td>88.0</td><td></td><td>93.9 86.5</td><td>81.2 89.3</td><td>93.9</td><td>93.4</td><td>91.7</td><td>79.3 84.3</td><td>54.7 80.8</td><td>92.9 80.8</td><td></td><td>81.0 86.0</td><td>89.9 79.0</td><td>95.1 80.3</td><td>87.9</td></tr><tr><td>MNIST</td><td>36.5</td><td>40.5</td><td>37.3</td><td></td><td>35.9</td><td>48.6</td><td>72.0 13.5</td><td>86.5 42.7</td><td>87.7 46.2</td><td>47.3</td><td>33.8</td><td>42.6</td><td></td><td>46.2</td><td>40.6</td><td>37.3</td><td>72.9</td></tr><tr><td>Musk</td><td>78.0</td><td>75.1</td><td>77.3</td><td></td><td>75.5</td><td>86.0</td><td>61.6</td><td>77.6</td><td>77.4</td><td>72.5</td><td>61.4</td><td>77.5</td><td></td><td>72.9</td><td>68.2</td><td>53.9</td><td>34.6</td></tr><tr><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.0</td><td>100.0</td><td></td><td>100.0</td><td>100.0</td><td>47.8</td><td>54.1</td></tr><tr><td>OptDigits</td><td>36.8</td><td>70.2</td><td>37.8</td><td></td><td>22.0</td><td>65.9</td><td>22.6</td><td>36.6</td><td>91.4</td><td>11.0</td><td>6.4</td><td>45.2</td><td></td><td>20.3</td><td>6.6</td><td>17.5</td><td>87.4</td></tr><tr><td>PageBI.</td><td>88.7</td><td>78.7</td><td>86.3</td><td></td><td>86.5</td><td>78.7</td><td>72.3</td><td>86.3</td><td>83.1</td><td>88.4</td><td>77.8</td><td>83.9</td><td></td><td>82.7</td><td>79.4</td><td>68.2</td><td>6.4 73.7</td></tr><tr><td>PenDigits</td><td>80.5</td><td>88.3</td><td>91.0</td><td></td><td>79.4</td><td>93.7</td><td>79.3</td><td>97.1</td><td>97.7</td><td>85.2</td><td>62.3</td><td>96.1</td><td></td><td>57.1 49.2</td></table>

![](images/404c21a2373de2b9f667ef4061b91f2022f0fea2fca504c48778e626c2301578.jpg)  
Fig. 2. Per-dataset AUROC on ADBench. Cell values show AUROC (%), and colors indicate the gap from the best-performing method.  
Fig. 3. Per-dataset AUPRC on ADBench. Cell values show AUPRC (%), and colors indicate the gap from the best-performing method.

Figs. 2 and 3 illustrate per-dataset AUROC and AUPRC performance, respectively. Although the best-performing method varies across datasets, PRISM-T achieves performance close to the best on many datasets, particularly in AUROC. This observation is consistent with its leading mean performance and high Top-2 counts reported in Table 3.

To further examine statistical significance across all methods, Fig. 4 presents critical-difference (CD) diagrams. The CD diagrams show the average rank of each method, where lower ranks indicate better performance. Horizontal bars connect groups of methods with no statistically significant pairwise differences under Wilcoxon signed-rank tests with Holm correction over all model pairs $( \alpha = 0 . 0 5 )$ . As shown in Fig. 4, PRISM-T achieves the second-best average rank for AUROC (5.43) and the third-best for AUPRC (5.74). Its differences from the top-ranked methods are not statistically significant on either metric.

Beyond detection performance, Fig. 5 compares the computational efficiency of leading methods in terms of model-construction and inference times. Construction time measures the execution of the fit() procedure, while inference time measures the time required to score the test samples. Existing AD methods exhibit different computational trade-offs. Nearest-neighbor and kernel-based detectors (e.g., DTE-NP, KDE, and LUNAR) require comparisons with reference samples during inference, leading to increased inference time as the reference set grows. In contrast, deep learning-based detectors enable efficient inference but require costly weight optimization during model construction. LUNAR suffers from both reference-sample comparisons and neural weight optimization, resulting in higher costs in both stages. PRISM-T avoids both sources of computational overhead by eliminating neural weight optimization and repeated reference-sample comparisons during inference. As shown in Fig. 5, PRISM-T achieves low construction and inference costs. Its lightweight architecture uses only 41K neural parameters on average, approximately 0.16%–30.4% of those in the compared recent neural models. Based on average execution times across 47 datasets, PRISM-T on CPU achieves 18–156× faster inference than the nearest-neighbor and kernel-based detectors, and 38–6,824× faster model construction than the deep learning detectors. Moreover, its neural architecture readily supports GPU acceleration through existing deep learning frameworks, further speeding up model construction and inference by 7.4× and 21.6×, respectively, compared with its CPU implementation.

![](images/4f65ae3691238ba9c739bb88c2ba2ee8d4b45f2b5cabeb6b14d4b061fd5eb4ee.jpg)  
(a) AUROC

![](images/fdc03cb8aaf9b207d5ec18572322b5dfcbb34984e7610122c3873a24d9eb6b75.jpg)  
Fig. 4. Critical-difference diagrams of average ranks on ADBench: (a) AUROC and (b) AUPRC. Numbers indicate average rank and mean score (%) of each method. Horizontal bars connect methods with no significant difference (Wilcoxon signed-rank test with Holm correction).

![](images/580596d544b38140fe2cd873fea48ee61f8c1fc8cf307b6a2e7c92e416c4d145.jpg)  
Fig. 5. Construction–inference efficiency on ADBench. Diamonds denote means and dots denote per-dataset runtimes; labels show average AUROC and parameter count. All deep learning baselines ran on GPU, and p denotes the average number of neural network parameters.

Overall, these results demonstrate that PRISM-T combines performance comparable to state-of-the-art methods with low computational costs in both stages, establishing an efficient prior-to-learning baseline for tabular AD.

![](images/57f4c67996047667321a8be3a7210a938205c5c942bde07f898dc3f329ed6c4c.jpg)  
Fig. 6. Per-category image-level detection performance on MVTec AD and VisA. Cell values show image AUROC (AUC) and AUPRC (AUP) (%), and colors indicate the gap from the best-performing method.

## 5.3 Visual Anomaly Detection

Table 4 presents image-level detection and pixel-level localization results on MVTec AD and VisA. Despite relying entirely on randomly initialized networks, PRISM-V achieves image-level AUROC scores of 92.9 and 87.3, with corresponding AUPRC scores of 97.2 and 89.1, respectively. This capability extends to dense anomaly localization, achieving pixel-level AUROC scores of 96.2 and 96.7 and AUPRO scores of 85.9 and 73.1. PRISM-V substantially outperforms earlier deep learning methods (VAE and GANomaly) on the reported metrics, although a performance gap remains relative to recent pretrained-feature methods. Among the four evaluation metrics, PRISM-V is most competitive in pixel-level AUROC, showing the smallest gap to the bestperforming method on both benchmarks. To quantify its relative performance across categories and metrics, we use the performance retention ratio (PRR), defined as the average ratio of each category- and metric-specific score to the corresponding best score. PRISM-V achieves PRRs of 94.1% on MVTec AD and 87.8% on VisA, comparable to those of some pretrained-feature methods (SPADE and WinCLIP+). Notably, these results are achieved with entirely random neural weights and only 0.33M parameters, less than 0.5% of those in the pretrained neural networks used by SPADE and WinCLIP+. This suggests that substantial visual anomaly evidence can be exploited through random nonlinear features and Mahalanobis scoring alone, without learned visual representations.

To examine performance variations across object categories, Figs. 6 and 7 present the per-category results. At the image level, PRISM-V achieves AUROC scores above 90 in 12 of the 15 MVTec AD categories, including a perfect score on Toothbrush. It also outperforms PaDiM and WinCLIP+ on PCB1–PCB3 in VisA, although performance varies across categories, particularly on Screw, Capsules, and Macaroni2. At the pixel level, all 27 categories achieve AUROC scores above 90, with 20 categories scoring at least 95. PRISM-V also achieves higher pixel-level AUROC than multiple pretrained-feature methods in several categories, surpassing five of the six compared methods on Transistor and four on Cashew. At the region level, AUPRO provides a complementary assessment by measuring the coverage of individual anomaly regions. On Transistor, PRISM-V surpasses four pretrained-feature methods in AUPRO. In contrast, Cashew and Pipe fryum achieve pixel-level AUROC scores of 99.0 and 99.1, respectively, while their corresponding AUPRO scores are 73.1 and 78.1.

TABLE 4  
Image-level AD and pixel-level anomaly localization on MVTec AD and VisA benchmarks.
<table><tr><td></td><td></td><td colspan="5">MVTec AD</td><td colspan="5">VisA</td></tr><tr><td></td><td></td><td colspan="2">Image-level</td><td colspan="2">Pixel-level</td><td></td><td colspan="2">Image-level</td><td colspan="2">Pixel-level</td><td></td></tr><tr><td>Method</td><td>Parameters</td><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRO</td><td>PRR</td><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRO</td><td>PRR</td></tr><tr><td>PRISM-V</td><td>0.33 M</td><td> $9 2 . 9 \pm 0 . 2 $ </td><td> $9 7 . 2 \pm 0 . 1 $ </td><td> $9 6 . 2 \pm 0 . 1 $ </td><td>85.9±0.2</td><td>94.1</td><td>87.3±0.1</td><td> $8 9 . 1 \pm 0 . 1$ </td><td> $9 6 . 7 \pm 0 . 0 $ </td><td> $7 3 . 1 \pm 0 . 3$ </td><td>87.8</td></tr><tr><td>VAE</td><td>11.1 M</td><td> $6 9 . 1 \pm 0 . 6 $ </td><td> $8 6 . 1 \pm 0 . 4$ </td><td> $7 6 . 8 \pm 0 . 6$ </td><td>54.3±0.5</td><td>72.3</td><td> $7 8 . 0 \pm 2 . 1$ </td><td> $8 1 . 2 { \pm } 1 . 6 $ </td><td> $8 4 . 4 \pm 0 . 4$ </td><td> $5 9 . 1 \pm 0 . 8 $ </td><td>76.8</td></tr><tr><td>GANomaly</td><td>9.06 M</td><td> $5 7 . 9 \pm 3 . 3 $ </td><td> $8 1 . 2 { \pm } 1 . 5 $ </td><td>一</td><td></td><td>69.7</td><td> $6 1 . 7 \pm 4 . 2$ </td><td> $6 9 . 3 \pm 3 . 3$ </td><td></td><td></td><td>66.3</td></tr><tr><td>SPADE†</td><td>68.9 M</td><td>79.9</td><td>90.4</td><td>96.8</td><td>92.4</td><td>91.0</td><td>79.7</td><td>82.6</td><td>97.5</td><td>85.9</td><td>87.7</td></tr><tr><td>PaDiM</td><td>68.9 M</td><td> $9 6 . 7 \pm 0 . 1 $ </td><td> $9 8 . 5 { \pm } 0 . 1 $ </td><td> $9 7 . 5 { \pm } 0 . 0 $ </td><td>92.3±0.1</td><td>97.4</td><td> $9 0 . 6 \pm 0 . 1$ </td><td> $9 1 . 5 { \pm } 0 . 1 $ </td><td> $9 7 . 5 { \pm } 0 . 0 $ </td><td>83.5±0.1</td><td>92.2</td></tr><tr><td>PatchCore</td><td>68.9 M</td><td> $9 8 . 9 \pm 0 . 2 $ </td><td> ${ \bf 9 9 . 7 \pm 0 . 1 }$ </td><td> $9 7 . 9 { \pm } 0 . 0 $ </td><td> $9 2 . 3 { \pm } 0 . 1 $ </td><td>98.4</td><td> $9 4 . 7 \pm 0 . 2 $ </td><td> $9 6 . 1 \pm 0 . 1 $ </td><td> $9 7 . 9 \pm 0 . 1 $ </td><td> $8 7 . 7 \pm 0 . 4 $ </td><td>95.5</td></tr><tr><td>WinCLIP+†</td><td>208M</td><td>97.2</td><td>98.6</td><td>95.0</td><td>86.8</td><td>95.5</td><td>90.5</td><td>92.4</td><td>95.3</td><td>81.2</td><td>91.2</td></tr><tr><td>AnomalyDINO†</td><td>22.1 M</td><td>99.2</td><td>99.7</td><td>97.9</td><td>93.9</td><td>98.9</td><td>97.2</td><td>97.6</td><td>98.6</td><td>95.0</td><td>98.6</td></tr><tr><td>SubspaceAD†</td><td>1.14 B</td><td>99.2</td><td>99.6</td><td>98.2</td><td>95.7</td><td>99.3</td><td>98.2</td><td>98.3</td><td>99.1</td><td>96.9</td><td>99.7</td></tr></table>

For stochastic methods, we report the mean and standard deviation over five seeds; deterministic baselines are marked with † and evaluated once. Parameters denotes the parameter count of the neural network used by each method. Performance retention ratio (PRR) is the average ratio of each category- and metric-specific score to the corresponding best score.

![](images/74fc5fc35f726a73f823806ff32b1dac8e07f728f35290a160fc0b9d928767e8.jpg)  
Fig. 7. Per-category pixel-level localization performance on MVTec AD and VisA. Cell values show pixel AUROC (AUC) and AUPRO (PRO) (%), and colors indicate the gap from the best-performing method.

Effect of random transformation and nonlinear activation in PRISM.  
TABLE 5
<table><tr><td colspan="4">Tabular</td><td colspan="4">Visual</td></tr><tr><td>Variant</td><td>AUC</td><td>AUP</td><td>G</td><td>I-AUC</td><td>P-AUC</td><td>PRO</td><td> $G _ { i m g } / G _ { p i x }$ </td></tr><tr><td>Raw</td><td>82.4</td><td>56.8</td><td>0.78</td><td>77.7</td><td>84.1</td><td>62.4</td><td> $0 . 4 7 / 0 . 5 0$ </td></tr><tr><td>w/o Act.</td><td>83.2</td><td>59.3</td><td>0.77</td><td>87.9</td><td>95.0</td><td>75.4</td><td>0.58/0.71</td></tr><tr><td>PRISM</td><td>86.4</td><td>65.9</td><td>0.80</td><td>90.4</td><td>96.4</td><td>80.2</td><td> $\mathbf { 0 . 7 4 } / \mathbf { 0 . 7 8 }$ </td></tr></table>

“Raw” uses observation-space Mahalanobis scoring; “w/o Act.” removes nonlinear activation layer of PRISM.

Figure 8 presents CD diagrams comparing PRISM-V with the baselines using the same statistical procedure as in the tabular AD evaluation. Although PRISM-V ranks below the leading pretrained-feature methods, it achieves better average ranks than SPADE in image-level detection and WinCLIP+ in pixel-level AUROC. These findings demonstrate that prior-to-learning anomaly detection extends beyond tabular data to visual detection and pixel-level localization, while also suggesting that pretrained visual representations offer additional advantages.

## 6 FURTHER ANALYSIS

## 6.1 Random Transformation and Nonlinearity

The preceding results demonstrate the competitive performance of PRISM against existing AD methods. To investigate how random nonlinear transformations affect anomaly detection performance, Table 5 compares Mahalanobis scoring in the observation space (Raw), PRISM without nonlinear activation (w/o Act.), and the full model (PRISM). The Raw variant for visual AD uses pixel-unshuffled RGB inputs to match the scoring grid of PRISM-V, while retaining the same scoring pipeline.

![](images/535a4beeff62d02a54d43a143c915890f96191f0cd8c1f2dfd2ded46deba1e6c.jpg)  
Fig. 8. Critical-difference diagrams of average ranks across the 27 categories (MVTec AD and VisA combined) for each metric: (a) image-leve AUROC, (b) image-level AUPRC, (c) pixel-level AUROC, and (d) AUPRO. Numbers indicate average rank and mean score (%) of each method. Horizontal bars connect methods with no significant difference (Wilcoxon signed-rank test with Holm correction).

For tabular AD, random transformation improves AU-ROC and AUPRC over Raw. Introducing LeakyReLU nonlinearity further improves AUROC by 3.8% and AUPRC by 11.1%, with a more pronounced improvement in AUPRC. The score-level separability G also increases from 0.77 to 0.80. For visual AD, substantial improvements are observed even without nonlinear activation, with imagelevel AUROC, pixel-level AUROC, and AUPRO increasing by 13.1%, 13.0%, and 20.8% over Raw, respectively. LeakyReLU further enhances detection and localization, particularly AUPRO, which rises from 75.4% to 80.2%. This improvement is accompanied by increased score-level separability at both the image and pixel levels, reaching $G _ { \mathrm { i m g } } ^ { - } = 0 .$ 74 and $G _ { \mathrm { p i x } } = 0 . 7 8$

Overall, these results demonstrate that random nonlinear representations in PRISM improve anomaly detection over observation-space scoring in both modalities, consistent with the mechanisms discussed in Section 4. The associated changes in score-level separability are examined in the following analyses.

## 6.2 Analysis of PRISM-T

Feature Transformations and Scoring Rules: The PRISM framework consists of two components: untrained neural feature transformation and Mahalanobis scoring. To examine their respective contributions, we compare alternative nonlinear random features and anomaly scoring rules while keeping the other component fixed.

Table 6(a) compares PRISM-T with three existing random nonlinear feature transformations: Fourier [43], Maclaurin [44], and ReLU [45]. Note that the ReLU feature map differs from simply replacing the activation function in PRISM-T, as it uses a bias-free Gaussian random projection without normalization. All transformations produce 512- dimensional features and use the same scoring procedure. Under these conditions, PRISM-T achieves the highest mean AUROC and AUPRC, followed by ReLU. Table 6(b) compares scoring rules applied to the same PRISM-T features. Mahalanobis scoring achieves the highest mean AU-ROC and AUPRC, exceeding kNN by 1.0 and 2.0 percentage points (pp), respectively. While kNN achieves comparable performance, it requires nearest-neighbor search over the reference samples at inference. Mahalanobis scoring instead uses a precomputed mean and precision matrix, making its per-sample scoring cost independent of the referenceset size. Its substantial advantage over Euclidean scoring further highlights the value of accounting for covariance in the feature space.

TABLE 6  
Comparison of feature maps and scoring rules on ADBench datasets.
<table><tr><td colspan="3">(a) Feature map</td><td colspan="3">(b) Scoring rule</td></tr><tr><td>Map</td><td>AUROC</td><td>AUPRC</td><td>Scorer</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>Fourier</td><td> $8 5 . 9 \pm 0 . 7$ </td><td> $6 3 . 8 \pm 2 . 5$ </td><td>Euclidean</td><td> $7 8 . 6 \pm 1 . 5$ </td><td> $5 2 . 6 \pm 3 . 0$ </td></tr><tr><td>Maclaurin</td><td> $8 4 . 4 \pm 0 . 7$ </td><td> $6 1 . 1 \pm 2 . 5$ </td><td>IForest</td><td> $8 1 . 1 \pm 2 . 3$ </td><td> $5 2 . 4 \pm 4 . 8$ </td></tr><tr><td>ReLU</td><td> $8 6 . 0 \pm 0 . 7$ </td><td> $6 4 . 5 \pm 2 . 4$ </td><td>kNN</td><td> $8 5 . 4 \pm 0 . 9$ </td><td> $\underline { { 6 3 . 9 } } \pm 2 . 7$ </td></tr><tr><td>PRISM</td><td> ${ \bf 8 6 . 4 \pm 0 . 9 }$ </td><td> ${ \bf 6 5 . 9 \pm 2 . 5 }$ </td><td>Mahal.</td><td> ${ \bf 8 6 . 4 \pm 0 . 9 }$ </td><td> ${ \bf 6 5 . 9 \pm 2 . 5 }$ </td></tr></table>

(a) Random nonlinear feature transformations are compared under fixed Mahalanobis scoring. (b) Anomaly scoring methods are compared using fixed PRISM-T features.

Together, the two comparisons support the combination of random neural feature transformation and Mahalanobis scoring in PRISM-T. The architecture provides an effective nonlinear representation, while covariance-aware scoring translates deviations in that representation into anomaly scores.

Score-Level Separability: This analysis examines how the random neural representation in PRISM-T changes normal– abnormal score separation and how these changes relate to detection performance. Following Section 4, the covariance and mean contributions to the score gap are examined together with score dispersion. Tabular AD provides a direct setting for this analysis, as each sample receives a single anomaly score corresponding to the class-conditional score distributions considered in the theory.

Using anomaly scores from normal and anomalous test samples, Table 7 summarizes $G , \Delta ,$ , and $\sqrt { V }$ , including the mean and covariance contributions to $\Delta ,$ for Raw, $\mathbf { w } / \mathbf { o }$ $\operatorname { A c t . }$ , and PRISM-T. The unstandardized mean contribution $\Delta ^ { \mathrm { m e a n } }$ increases on 44 and 39 of the 47 datasets relative to Raw and w/o Act., respectively, with increases in both its mean and median. This pattern is consistent with the theoretical role of nonlinear transformation in enlarging the mean contribution to the score gap. After normalization by score dispersion, however, the covariance contribution shows more consistent gains: $G ^ { \mathrm { c o v } }$ increases on 40 and 39 datasets relative to Raw and w/o Act., whereas G<sup>mean</sup> increases on 19 and 25 datasets. The resulting changes in overall score-level separability G are illustrated across individual datasets in Fig. 9. PRISM-T improves separation across most datasets, although the magnitude of improvement varies.

TABLE 7  
Decomposition of score-level separability on ADBench datasets.
<table><tr><td></td><td colspan="3">Mean (median)</td><td colspan="2">Increase count</td></tr><tr><td>Quantity</td><td>Raw</td><td>w/o Act.</td><td>PRISM</td><td>vs. Raw</td><td>VS. w/o Act.</td></tr><tr><td> $G$ </td><td>0.78 (0.39)</td><td>0.77 (0.36)</td><td>0.80 (0.45)</td><td>34</td><td>35</td></tr><tr><td> $G ^ { \mathrm { c o v } }$ </td><td>0.24 (0.18)</td><td>0.28 (0.20)</td><td>0.49 (0.34)</td><td>40</td><td>39</td></tr><tr><td>Gmean</td><td>0.55 (0.15)</td><td>0.49 (0.09)</td><td>0.31 (0.11)</td><td>19</td><td>25</td></tr><tr><td> $\Delta$ </td><td>46.88 (0.06)</td><td>114.82 (0.42)</td><td>91.50 (4.11)</td><td>41</td><td>36</td></tr><tr><td> $\Delta ^ { \mathrm { c o v } }$ </td><td>45.18 (0.04)</td><td>110.56 (0.21)</td><td>60.58 (3.12)</td><td>38</td><td>35</td></tr><tr><td> $\Delta ^ { \mathrm { m e a n } }$ </td><td>1.69 (0.02)</td><td>4.17 (0.07)</td><td>29.33 (0.75)</td><td>44</td><td>39</td></tr><tr><td> $\sqrt { V _ { r } }$ </td><td>203.15 (0.13)</td><td>679.81 (0.64)</td><td>256.58 (12.45)</td><td>41</td><td>39</td></tr></table>

Entries report means, with medians in parentheses. Increase counts indicate the number of datasets with higher values than the baseline (out of 47). Gap and dispersion values (∆ and $\sqrt { V _ { r } } )$ are reported in units of $1 0 ^ { 3 }$

![](images/c4ba2dfe1898cf1e2c1709c1261a8a3e6ac9bcc49fd14e3676c25ecffb7b453a.jpg)  
Fig. 9. Score-level separability across 47 ADBench datasets. Crosses and circles indicate w/o Act. and PRISM-T, respectively; arrows show changes induced by nonlinear activation. The dashed line denotes $G _ { r } =$ $G _ { o } .$

Fig. 10 relates changes in score-level separability G to changes in AUROC and AUPRC relative to Raw. Without nonlinear activation, the random transformation produces little change in average separability $( \Delta G = - 0 . { \bar { 0 3 } } )$ , with modest improvements in AUROC (+0.9 pp) and AUPRC (+2.6 pp). In contrast, adding LeakyReLU shifts the average toward higher separability $( \Delta G = + 0 . 2 3 )$ and larger improvements in AUROC (+4.1 pp) and AUPRC (+9.1 pp). Across datasets, changes in G from Raw to PRISM-T are positively correlated with the corresponding changes in AUROC and AUPRC (Spearman’s $\rho \ = \ 0 . 4 9$ and 0.47, respectively).

The decomposition and performance analyses show that random nonlinear transformations modify both the score gap and its dispersion, with distinct effects on the mean and covariance contributions. The resulting improvements in score-level separability are associated with improved anomaly detection performance, supporting the role of random nonlinear representations in exposing intrinsic anomaly separability.

![](images/b9a697adf69c96e5658b062b3223d7d332c1b6fc35f455e68769e8760b65da17.jpg)

![](images/914a34dcf0ea87907b144efdbde8c227e9d6a484213e8d6eec2420a16c4e2b06.jpg)  
(a) AUROC  
(b) AUPRC  
Fig. 10. Changes in separability and detection performance relative to Raw on 47 datasets: (a) AUROC and (b) AUPRC. Large markers and ellipses indicate the mean and ± one standard deviation across datasets, excluding http due to its outlying ∆G.

TABLE 8  
Comparison with randomly initialized baselines.
<table><tr><td>Method</td><td>Backbone</td><td>I-AUROC</td><td>AUPRO</td></tr><tr><td>PRISM-V</td><td>Proposed Backbone</td><td> ${ \bf 9 0 . 4 \pm 0 . 1 }$ </td><td> ${ \bf 8 0 . 2 \pm 0 . 1 }$ </td></tr><tr><td>SPADE PaDiM PatchCore</td><td>WRN-50-2</td><td> $6 0 . 8 \pm 0 . 7$   $7 9 . 5 \pm 0 . 1$   $7 6 . 6 \pm 1 . 5$ </td><td> $5 2 . 0 \pm 1 . 0$   $5 5 . 5 \pm 0 . 5$   $5 1 . 3 \pm 1 . 2$ </td></tr><tr><td>WinCLIP+</td><td>V: CLIP ViT-B/16+ L: CLIP Transformer</td><td> $7 3 . 0 \pm 0 . 9$ </td><td> $4 3 . 5 \pm 4 . 5$ </td></tr><tr><td>AnomalyDINO SubspaceAD</td><td>DINOv2 ViT-S/14 DINOv2 ViT-g/14</td><td> $7 7 . 3 \pm 0 . 4$   $6 8 . 5 \pm 0 . 4$ </td><td> $6 0 . 2 \pm 0 . 6$   $5 4 . 4 \pm 0 . 4$ </td></tr></table>

WRN-50-2 denotes Wide ResNet-50-2. For WinCLIP+, V and L denote the vision and language encoders. ViT-S/14 and ViT-g/14 denote the small and giant Vision Transformers, respectively.

## 6.3 Analysis of PRISM-V

Feature Transformations and Scoring Rules: To further investigate the effectiveness of random feature representations in visual AD, we replace the pretrained backbone originally used by each baseline with a randomly initialized counterpart, while retaining its anomaly scoring procedure. These backbones include Wide ResNet [46], CLIP [47], and DINOv2 [48], as detailed in Table 8. Across the six randomly initialized baselines, image-level AUROC ranges from 60.8 to 79.5, while AUPRO ranges from 43.5 to 60.2. In contrast, PRISM-V achieves 90.4 and 80.2, exceeding the strongest baselines by 10.9 pp and 20.0 pp, respectively. Notably, PaDiM achieves the highest image-level AUROC among the baselines and employs a similar scoring principle based on location-specific Mahalanobis distance, yet remains substantially below PRISM-V. These results support the effectiveness of PRISM-V’s architectural design in exploiting random neural features for visual anomaly detection.

Table 9 compares four scoring rules (Euclidean, IForest, kNN, and Mahalanobis) applied to the same fixed

TABLE 9  
Comparison of scoring rules for PRISM-V.
<table><tr><td>Scorer</td><td>I-AUROC</td><td>I-AUPRC</td><td>P-AUROC</td><td>AUPRO</td></tr><tr><td>Euclidean</td><td> $7 1 . 2 \pm 0 . 3$ </td><td> $8 1 . 0 \pm 0 . 3$ </td><td> $8 9 . 0 \pm 0 . 1$ </td><td> $5 9 . 9 \pm 0 . 4$ </td></tr><tr><td>IForest</td><td> $7 9 . 3 \pm 0 . 2$ </td><td> $8 5 . 9 \pm 0 . 1$ </td><td> $8 9 . 4 \pm 0 . 2$ </td><td> $5 8 . 0 \pm 0 . 2$ </td></tr><tr><td>kNN</td><td> $8 1 . 2 \pm 0 . 3$ </td><td> $\underline { { 8 8 . 0 \pm 0 . 2 } }$ </td><td> $9 4 . 2 \pm 0 . 1$ </td><td> $\mathrm { 7 4 . 5 \pm 0 . 2 }$ </td></tr><tr><td>Mahalanobis</td><td> $\overline { { \mathbf { 9 0 . 4 } } } \pm 0 . 1$ </td><td> ${ \bf 9 3 . 6 \pm 0 . 0 }$ </td><td> ${ \bf 9 6 . 4 \pm 0 . 0 }$ </td><td> ${ \bf 8 0 . 2 \pm 0 . 1 }$ </td></tr></table>

TABLE 10

Score-level separability of PRISM-V on MVTec AD and VisA.
<table><tr><td rowspan="2"></td><td colspan="3">Mean (median)</td><td colspan="2">Increase count</td></tr><tr><td>Raw</td><td>w/o Act.</td><td>PRISM-V</td><td> $\mathbf { V S } .$  Raw</td><td>vS. w/o Act.</td></tr><tr><td>Quantity  $G _ { \mathrm { i m g } }$ </td><td>0.47 (0.44)</td><td>0.58 (0.52)</td><td>0.74 (0.63)</td><td>22</td><td>22</td></tr><tr><td> $G _ { \mathrm { p i x } }$ </td><td>0.50 (0.48)</td><td>0.71 (0.69)</td><td>0.78 (0.77)</td><td>22</td><td>16</td></tr><tr><td> $\Delta _ { \mathrm { i m g } }$ </td><td>0.05 (0.02)</td><td>2.26 (1.44)</td><td>6.63 (3.51)</td><td>27</td><td>26</td></tr><tr><td> $\Delta _ { \mathrm { p i x } }$ </td><td>0.04 (0.01)</td><td>1.79 (1.13)</td><td>4.68 (2.40)</td><td>27</td><td>25</td></tr><tr><td> $\sqrt { V _ { \mathrm { i m g } } }$ </td><td>0.14 (0.06)</td><td>3.70 (2.56)</td><td>8.23 (5.51)</td><td>27</td><td>24</td></tr><tr><td> $\sqrt { V _ { \mathrm { p i x } } }$ </td><td>0.09 (0.04)</td><td>2.73 (1.65)</td><td>6.17 (3.76)</td><td>27</td><td>25</td></tr></table>

Entries report means, with medians in parentheses. Increase counts indicate the number of categories with higher values than the baseline (out of 27).

512-dimensional PRISM-V features. All scorers are fitted location-wise. Consistent with PRISM-T, Mahalanobis scoring achieves the highest mean performance across all four metrics, followed by kNN. The advantage is more pronounced in PRISM-V, where Mahalanobis outperforms kNN by 2.2–9.2 pp across the four metrics. Compared with Euclidean scoring, it improves image-level AUROC and AUPRO by 19.2 pp and 20.3 pp, respectively. The larger performance gains in PRISM-V indicate a greater benefit from covariance-aware scoring when using random visual features.

Score-Level Separability: We extend the score-level separability analysis to visual AD across the 27 categories of MVTec AD and VisA. While the theoretical formulation defines G using squared Mahalanobis distances, visual AD involves spatial postprocessing and aggregation of Mahalanobis distances. Accordingly, we evaluate separability using the squared pixel-level anomaly scores $\begin{array} { r } { \hat { S } _ { \mathrm { p i x } } ( I ) ~ \dot { = } } \end{array}$ $( A _ { \mathrm { p i x } } ( I ) ) ^ { 2 }$ and its image-level aggregation $\begin{array} { r l } { S _ { \mathrm { i m g } } ( I ) } & { { } = } \end{array}$ $\mathrm { T V a R _ { 1 \% } } { \circ } ( S _ { \mathrm { p i x } } ( I ) )$ . Normal and abnormal score distributions are constructed by partitioning pixel-level scores using ground-truth masks and image-level scores using image labels. Since the resulting scores no longer retain the exact quadratic form of the original Mahalanobis score, we analyze $G , \Delta ,$ and $\sqrt { V }$ without decomposing the score gap into mean and covariance contributions.

Table 10 presents the image- and pixel-level separability results for Raw, w/o Act., and PRISM-V. As in tabular AD, PRISM-V improves score-level separability, but the gains are more pronounced, with G increasing at both stages of feature transformation. Mean $G _ { \mathrm { i m g } }$ increases from 0.47 for Raw and 0.58 for w/o Act. to 0.74, with improvements in 22 of 27 categories relative to each baseline. At the pixel level, mean $\breve { G } _ { \mathrm { { p i x } } }$ increases from 0.50 and 0.71 to 0.78. A notable change is observed in the score gap ∆. Its mean increases substantially from near-zero values in Raw (0.05 and 0.04) to 6.63 and 4.68 in PRISM-V at the image and pixel levels, respectively. These correspond to approximately 133- and 117-fold increases, while the mean score dispersion $\sqrt { V }$ increases by 59- and 69-fold. The category-wise changes in G are further illustrated in Fig. 11. Fig. 12 shows changes in G versus changes in detection and localization performance across categories. Consistent with the analysis of PRISM-T, positive correlations are observed (Spearman’s $\rho = 0 . 4 1$ 0.56, and 0.51 for image-level AUROC, pixel-level AUROC, and AUPRO, respectively).

![](images/b01efa76ef5af6fd44896a83f23f9102dcc51a0479b31ca16f0310100e3b0c77.jpg)

(a) Image-level separability per category  
![](images/11da60398a529babf7ba8eb44399e5e6344ce1cfaf6f56699a5146bc22baa3f8.jpg)  
(b) Pixel-level separability per category

Fig. 11. Score-level separability across 27 categories of MVTec AD and VisA. Crosses and circles indicate w/o Act. and PRISM-V, respectively; arrows show changes induced by nonlinear activation. The dashed line denotes $G _ { r } = G _ { o }$  
![](images/e8f79f8b697be79dad79c096f1a57045f526505da856e487086f837b3384be13.jpg)

![](images/3016340239ac83b566de1b8e6051e27016ee2c9b663a498c511144cccf1455ea.jpg)  
(a) Image-level AUROC  
(b) AUPRO  
Fig. 12. Changes in separability and detection performance relative to Raw on the 27 categories of MVTec AD and VisA: (a) image-level AUROC versus $\Delta G _ { \mathrm { i m g } }$ and (b) AUPRO versus $\Delta G _ { \mathrm { p i x } } .$ . Large markers and ellipses indicate the mean and ± one standard deviation over the 27 categories, and the arrow connects the two means.

![](images/90821463f4ed02cdc0895d49d241ad3f03e310e2970a18f799c5c8eba258d3c2.jpg)  
(b) Examples from the bottom-5 categories  
Fig. 13. Illustrative comparison of anomaly localization using PRISM-V, PaDiM, and SubspaceAD. Examples from the five highest- and lowestperforming categories ranked by PRISM-V’s AUPRO. GT denotes ground-truth anomaly regions. For each sample, values in the upper-left and upper-right corners indicate pixel AUROC and AUPRO (%), respectively.

Visual Inspection of Anomaly Localization: Figure 13 compares the anomaly localization results of PRISM-V, PaDiM, and SubspaceAD across the five highest- and five lowest-AUPRO categories for PRISM-V. In the top-5 categories, PRISM-V produces concentrated anomaly responses around the ground-truth defects, as illustrated by leather, screw, and zipper. Both pixel AUROC and AUPRO remain high, with localization performance generally comparable to that of the pretrained-feature baselines. These results demonstrate that randomly initialized CNN features can support precise anomaly localization without relying on learned visual representations. In contrast, the bottom-5 categories exhibit less selective anomaly responses, often extending into normal object boundaries or surrounding structures, as observed in fryum, pcb4, and chewinggum. In categories with multiple similar objects, such as capsules and macaroni2, anomaly responses also spread beyond the ground-truth defects. SubspaceAD generally provides more focused localization in these cases. Even in these challenging categories, PRISM-V exhibits elevated anomaly responses near some ground-truth defects, although these responses are less spatially selective and extend into normal regions.

These visual results complement the quantitative findings by showing that PRISM-V produces localized anomaly responses in favorable categories while retaining sensitivity to defects in more challenging cases. Together, they further demonstrate the potential of untrained neural features for anomaly localization.

TABLE 11  
Modality-specific component ablations for PRISM.
<table><tr><td colspan="3">Tabular</td><td colspan="3">Visual</td></tr><tr><td></td><td>AUROC</td><td>AUPRC</td><td></td><td>I-AUROC</td><td>AUPRO</td></tr><tr><td>Base</td><td> $8 6 . 2 _ { ( - ) }$ </td><td> $6 4 . 4 _ { ( - ) }$ </td><td>Base</td><td> $8 2 . 0 _ { ( - ) }$ </td><td> $^ { 6 9 . 5 } ( - )$ </td></tr><tr><td>+IN</td><td> $8 0 . 5 _ { ( - 5 . 7 ) }$ </td><td> $5 4 . 4 _ { ( - 1 0 ) }$ </td><td>+IN</td><td> $8 3 . 3 _ { ( + 1 . 3 ) }$ </td><td> $7 2 . 4 _ { ( + 2 . 9 ) }$ </td></tr><tr><td>+BN</td><td> $8 6 . 1 \dot { } ( - 0 . 1 )$ </td><td> $6 4 . 2 _ { ( - 0 . 2 ) }$ </td><td>+BP</td><td> $8 3 . 8 \dot { } _ { ( + 1 . 8 ) }$ </td><td> $7 5 . 8 \dot { ( } + 6 . 3 )$ </td></tr><tr><td>+IBN</td><td> $8 6 . 4 _ { ( + 0 . 2 ) }$ </td><td> ${ \bf 6 5 . 9 } _ { ( + 1 . 5 ) }$ </td><td>+Full</td><td> $9 0 . 4 _ { ( + 8 . 4 ) }$ </td><td> $8 0 . 2 _ { ( + 1 0 . 7 ) }$ </td></tr></table>

IN/BN/IBN denote instance/batch/instance-batch normalization; BP denotes BlurPool. Visual Base omits normalization and uses MaxPool in place of BP. Parentheses show differences from Base.

## 6.4 Ablation Studies

While the preceding analyses focused on score-level separability, this section examines how anomaly detection performance varies with model components and hyperparameter settings. Architectural configurations used in the main experiments are marked with ∗. Table 11 summarizes the modality-specific effects of normalization and pooling. For PRISM-T, IN alone degrades performance, whereas IBN achieves the best results. For PRISM-V, BlurPool and IN each improve performance, with their combination yielding the largest gains. Table 12 examines the effects of feature extractor width, depth, and activation functions. Increasing feature width improves performance in both variants, although the gains tend to diminish at larger widths, while covariance estimation and inversion become more costly. For PRISM-T, increasing depth provides no consistent improvement, suggesting limited benefits from stacking additional randomly initialized PBs. Regarding activation functions, ReLU and LeakyReLU yield nearly identical results, whereas Tanh and Sigmoid generally degrade performance, particularly in visual anomaly localization. In addition, spatially averaging location-wise covariance estimates over a $3 \times 3$ neighborhood improves performance in PRISM-V. Compared with no spatial covariance averaging, imagelevel AUROC increases by 1.8 pp and 1.3 pp on MVTec AD and VisA, respectively. The corresponding AUPRO gains are 0.2 pp and 0.5 pp.

## 7 CONCLUSION

In this work, we introduced PRISM, a prior-to-learning anomaly detection framework that combines fixed random nonlinear features with normal-reference Mahalanobis scoring. Its core component is a PRISM block that follows the typical linear transformation–normalization–activation pattern in neural networks, using a randomly initialized linear operator and a kinked nonlinearity. This design is instantiated in two modality-specific detectors: PRISM-T uses a single fully connected PRISM block with mixed instance and batch normalization for tabular anomaly scoring, while PRISM-V employs multi-scale convolutional features and location-wise scoring for visual anomaly localization.

Despite requiring neither pretraining nor neural weight optimization, both models achieve competitive detection performance. Across 47 tabular datasets in ADBench, PRISM-T achieves the highest mean AUROC (86.4%) and the second-highest mean AUPRC (65.9%) among the compared methods, while ranking first on the largest number of datasets in terms of AUROC. It also avoids stage-specific computational overhead, achieving up to 156× faster inference than nearest-neighbor and kernel-based detectors and up to 6,824× faster model construction than recent deep learning detectors. For visual AD, PRISM-V demonstrates effective detection and localization on MVTec AD and VisA, achieving mean image-level and pixel-level AUROCs of 90.4% and 96.4%, respectively. Across the evaluated metrics, it retains an average of 94.1% and 87.8% of the best categoryand metric-specific scores on the respective benchmarks, using a fully untrained network with only 0.33M parameters, approximately 0.03%–1.5% of those used by the compared pretrained-feature detectors. Our theoretical analysis further characterizes how nonlinear transformations affect the covariance and mean contributions to the expected Mahalanobis score gap. Complementary empirical analyses examine the resulting score separability and its relationship with detection performance, while ablation studies assess the effects of key architectural choices.

TABLE 12  
Ablations of feature extractor configurations in PRISM.
<table><tr><td colspan="4">PRISM-T</td></tr><tr><td>Factor</td><td>Configuration</td><td>AUROC</td><td>AUPRC</td></tr><tr><td rowspan="5">Width</td><td>128</td><td>85.61</td><td>62.83</td></tr><tr><td>256</td><td>86.01</td><td>64.83</td></tr><tr><td>512*</td><td>86.40</td><td>65.90</td></tr><tr><td>1024</td><td>86.60</td><td>66.64</td></tr><tr><td>1*</td><td>86.40</td><td>65.90</td></tr><tr><td rowspan="3">Depth</td><td>2</td><td>86.21</td><td>66.33</td></tr><tr><td>3</td><td>85.93</td><td>65.74</td></tr><tr><td>LeakyReLU*</td><td>86.40</td><td>65.90</td></tr><tr><td rowspan="4">Activation</td><td>ReLÚ</td><td>86.40</td><td>65.90</td></tr><tr><td>Tanh</td><td>86.30</td><td>64.63</td></tr><tr><td>Sigmoid</td><td>85.68</td><td>62.66</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Factor</td><td>PRISM-V</td><td></td><td></td></tr><tr><td rowspan="3">Width</td><td>Configuration</td><td>I-AUC I-AUP</td><td>P-AUC</td><td>PRO</td></tr><tr><td>32</td><td>87.77</td><td>91.95 95.32</td><td>76.60</td></tr><tr><td>64 128*</td><td>89.26 90.40</td><td>92.89 96.08 93.60 96.44</td><td>79.11 80.24</td></tr><tr><td rowspan="5">Activation</td><td></td><td></td><td>93.60</td><td></td></tr><tr><td>LeakyReLU* ReLU</td><td>90.40 90.40</td><td>96.44 93.60 96.43</td><td>80.24 80.25</td></tr><tr><td>Tanh</td><td>88.67</td><td>92.61 94.68</td><td>73.30</td></tr><tr><td></td><td></td><td>94.86</td><td>74.52</td></tr><tr><td>Sigmoid</td><td>86.74</td><td>91.13</td><td></td></tr></table>

∗ denotes the configuration used in the main experiments.

Together, these findings establish PRISM as a strong and computationally efficient prior-to-learning model across tabular and visual anomaly detection, providing a reference for assessing the added value of learned representations.

## ACKNOWLEDGEMENT

This work was supported in part by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2024-00454514, 50%), the Institute of Information & Communications Technology Planning & Evaluation (IITP) - ICAN (ICT Challenge and Advanced Network of HRD) grant funded by the Korea government (Ministry of Science and ICT) (IITP-2026-RS-2024-00436528, 40%), and the Institute Strategic Development Project (ISD)

in Korea Atomic Energy Research Institute (KAERI) funded by the Ministry of Science and ICT (No. 526140-26, 10%).

## REFERENCES

[1] C. Zhou and R. C. Paffenroth, “Anomaly detection with robust deep autoencoders,” in Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, 2017, pp. 665–674.

[2] L. Ruff, R. A. Vandermeulen, N. Gornitz, L. Deecke, S. A. Siddiqui,¨ A. Binder, E. Muller, and M. Kloft, “Deep one-class classification,”¨ in International Conference on Machine Learning (ICML), 2018, pp. 4393–4402.

[3] A. Goodge, B. Hooi, S.-K. Ng, and W. S. Ng, “LUNAR: Unifying local outlier detection methods via graph neural networks,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, 2022, pp. 6737–6745.

[4] K. Roth, L. Pemula, J. Zepeda, B. Scholkopf, T. Brox, and P. Gehler,¨ “Towards total recall in industrial anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 14 318–14 328.

[5] V. Livernoche, V. Jain, Y. Hezaveh, and S. Ravanbakhsh, “On diffusion modeling for anomaly detection,” in International Conference on Learning Representations (ICLR), vol. 2024, 2024, pp. 25 836– 25 866.

[6] Z. Li, Q. Huang, Y. Zhu, L. Yang, M. Mohammadi Amiri, N. van Stein, and M. van Leeuwen, “Scalable, explainable and provably robust anomaly detection with one-step flow matching,” Advances in Neural Information Processing Systems, vol. 38, pp. 87 834–87 899, 2025.

[7] T. Defard, A. Setkov, A. Loesch, and R. Audigier, “PaDiM: A patch distribution modeling framework for anomaly detection and localization,” in International Conference on Pattern Recognition. Springer, 2021, pp. 475–489.

[8] S. Damm, M. Laszkiewicz, J. Lederer, and A. Fischer, “Anomalydino: Boosting patch-based few-shot anomaly detection with dinov2,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 1319–1329.

[9] C. Lendering, E. Akdag, and E. Bondarau, “SubspaceAD: Trainingfree few-shot anomaly detection via subspace modeling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 28 557–28 566.

[10] S. Ryu, Y. Yu, and H. Seo, “Can untrained neural networks detect anomalies?” IEEE Transactions on Industrial Informatics, vol. 20, no. 4, pp. 6477–6488, 2024.

[11] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in IEEE International Conference on Data Mining (ICDM), 2008, pp. 413–422.

[12] S. Han, X. Hu, H. Huang, M. Jiang, and Y. Zhao, “ADBench: Anomaly detection benchmark,” in Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, vol. 35, 2022, pp. 32 142–32 159.

[13] T. Shenkar and L. Wolf, “Anomaly detection for tabular data with internal contrastive learning,” in International Conference on Learning Representations (ICLR), 2022.

[14] J. Yin, Y. Qiao, Z. Zhou, X. Wang, and J. Yang, “MCM: Masked cell modeling for anomaly detection in tabular data,” in International Conference on Learning Representations (ICLR), vol. 2024, 2024, pp. 4823–4845.

[15] J. Ye, Z. Tan, Y. Hu, X. Yang, G. Cheng, and K. Huang, “Disentangling tabular data towards better one-class anomaly detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, 2025, pp. 13 061–13 068.

[16] V. Livernoche, J. Zan, and R. Rabbany, “Kurtosis-guided denoising score matching for tabular anomaly detection,” arXiv preprint arXiv:2605.06955, 2026.

[17] C.-P. Tsai, G. Teng, P. Wallis, and W. Ding, “AnoLLM: Large language models for tabular anomaly detection,” in International Conference on Learning Representations (ICLR), vol. 2025, 2025, pp. 7562–7592.

[18] T. Schlegl, P. Seebock, S. M. Waldstein, G. Langs, and U. Schmidt-¨ Erfurth, “f-anogan: Fast unsupervised anomaly detection with generative adversarial networks,” Medical Image Analysis, vol. 54, pp. 30–44, 2019.

[19] S. Akc¸ay, A. Atapour-Abarghouei, and T. P. Breckon, “GANomaly: Semi-supervised anomaly detection via adversarial training,” in Proceedings of the Asian Conference on Computer Vision (ACCV), ser. LNCS, vol. 11363. Springer, 2018, pp. 622–637.

[20] S. Akc¸ay, A. Atapour-Abarghouei, and T. P. Breckon, “Skipganomaly: Skip connected and adversarially trained encoderdecoder anomaly detection,” in 2019 International Joint Conference on Neural Networks (IJCNN). IEEE, 2019, pp. 1–8.

[21] L. Ruff, R. A. Vandermeulen, N. Gornitz, A. Binder, E. M¨ uller, K.-R.¨ Muller, and M. Kloft, “Deep semi-supervised anomaly detection,”¨ in International Conference on Learning Representations (ICLR), 2020.

[22] N. Cohen and Y. Hoshen, “Sub-image anomaly detection with deep pyramid correspondences,” arXiv preprint arXiv:2005.02357, 2020.

[23] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer, “Winclip: Zero-/few-shot anomaly classification and segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 19 606–19 616.

[24] W. B. Johnson and J. Lindenstrauss, “Extensions of lipschitz mappings into a hilbert space,” Contemporary mathematics, vol. 26, pp. 189–206, 1984.

[25] R. Giryes, G. Sapiro, and A. M. Bronstein, “Deep neural networks with random gaussian weights: A universal classification strategy?” IEEE Transactions on Signal Processing, vol. 64, no. 13, pp. 3444–3457, 2016.

[26] I. Nachum, J. Hazla, M. C. Gastpar, and A. Khina, “A johnsonlindenstrauss framework for randomly initialized cnns,” in The Tenth International Conference on Learning Representations (ICLR 2022), 2022.

[27] H. Xu, G. Pang, Y. Wang, and Y. Wang, “Deep isolation forest for anomaly detection,” IEEE transactions on knowledge and data engineering, vol. 35, no. 12, pp. 12 591–12 604, 2023.

[28] B. N. Chiche, Y. Horikawa, and R. Fujita, “Pre-training vision models with mandelbulb variations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 22 062–22 071.

[29] J. Zhang, X. Chen, Y. Wang, C. Wang, Y. Liu, X. Li, M.-H. Yang, and D. Tao, “Exploring plain ViT features for multi-class unsupervised visual anomaly detection,” Computer Vision and Image Understanding, vol. 253, p. 104308, 2025.

[30] D. Ulyanov, A. Vedaldi, and V. Lempitsky, “Instance normalization: The missing ingredient for fast stylization,” arXiv preprint arXiv:1607.08022, 2016.

[31] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in International Conference on Machine Learning (ICML), 2015, pp. 448–456.

[32] R. Zhang, “Making convolutional networks shift-invariant again,” in International conference on machine learning. PMLR, 2019, pp. 7324–7334.

[33] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “MVTec AD—a comprehensive real-world dataset for unsupervised anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 9592–9600.

[34] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer, “SPot-the-Difference self-supervised pre-training for anomaly detection and segmentation,” in European Conference on Computer Vision (ECCV). Springer, 2022, pp. 392–408.

[35] P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, and C. Steger, “The MVTec anomaly detection dataset: A comprehensive realworld dataset for unsupervised anomaly detection,” International Journal of Computer Vision, vol. 129, no. 4, pp. 1038–1059, 2021.

[36] T. Sattarov, M. Schreyer, and D. Borth, “Diffusion-scheduled denoising autoencoders for anomaly detection in tabular data,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 2025, pp. 2478–2489.

[37] M. Hirth, L. Koberg, N. Jazdi, and E. Kasneci, “Denoising without diffusion: Fixed-noise denoiser anomaly detection in tabular data,” in Forty-third International Conference on Machine Learning, 2026.

[38] E. Parzen, “On estimation of a probability density function and mode,” The annals of mathematical statistics, vol. 33, no. 3, pp. 1065– 1076, 1962.

[39] D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” in International Conference on Learning Representations, 2014. [Online]. Available: https://mlanthology.org/iclr/2014/ kingma2014iclr-auto/

[40] C. Baur, S. Denner, B. Wiestler, N. Navab, and S. Albarqouni, “Autoencoders for unsupervised anomaly segmentation in brain mr images: a comparative study,” Medical image analysis, vol. 69, p. 101952, 2021.

[41] M. Miranda, L. Zabawa, A. Kicherer, L. Strothmann, U. Rascher, and R. Roscher, “Detection of anomalous grapevine berries using variational autoencoders,” Frontiers in Plant Science, vol. 13, p. 729097, 2022.

[42] K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers: Surpassing human-level performance on imagenet classification,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 1026–1034.

[43] A. Rahimi and B. Recht, “Random features for large-scale kernel machines,” in Advances in Neural Information Processing Systems, vol. 20, 2007.

[44] P. Kar and H. Karnick, “Random feature maps for dot product kernels,” in Artificial intelligence and statistics. PMLR, 2012, pp. 583–591.

[45] Y. Cho and L. K. Saul, “Kernel methods for deep learning,” in Advances in Neural Information Processing Systems, vol. 22, 2009.

[46] S. Zagoruyko and N. Komodakis, “Wide residual networks,” in Proceedings of the British Machine Vision Conference (BMVC), 2016, pp. 87.1–87.12.

[47] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning, vol. 139. PMLR, 2021, pp. 8748–8763. [Online]. Available: https://proceedings.mlr.press/v139/radford21a.html

[48] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou,´ J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research (TMLR), 2024.