# Investigating Cross-Spectral Feature Fusion and Targeted Structural Guidance for Multispectral Document Binarization

Yasas Savinda and Maheshi Buddhinee Dissanayake

Department of Electrical and Electronic Engineering, University of Peradeniya, Peradeniya, Sri Lanka.

\*Corresponding author(s). E-mail(s): e19356@eng.pdn.ac.lk; maheshid@eng.pdn.ac.lk;

## Abstract

Purpose: Historical document binarization remains challenging when faded ink, bleed-through, and background degradation exhibit wavelength-dependent behavior across visible and near-infrared observations. This study investigates whether adaptive cross-spectral feature fusion and targeted structural guidance can improve multispectral document binarization beyond fixed fusion and generic attention mechanisms.

Methods: Four registered SMADI bands at 500, 600, 700, and 1000 nm are processed using a multi-branch conditional generative adversarial network (cGAN). Visible, NIR, and structural feature streams are fused at encoder depth L = 3 using spatial- and channel-wise softmax gating, while auxiliary edge maps derived from selected VIS–NIR band pairs provide targeted structural guidance. Nine controlled fusion and structural configurations are evaluated using five-fold document-level crossvalidation, DIBCO-standard metrics, and Holm-corrected statistical testing.

Results: The F3 + NIR edge-guided configuration provides the strongest overall performance among the evaluated models, achieving mean F = 0.7916, p-FM = 0.8044, and IoU = 0.6654. Relative to the fixed Mid AVG baseline, it improves F<sub>1</sub> by 0.0269, p-FM by 0.0360, and IoU by 0.0321, with all three gains remaining statistically significant after Holm correction. Mean NRM and DRD are also reduced, although these reductions are not statistically significant after correction.

Conclusion: Targeted cross-spectral structural guidance provides an efective mechanism for regulating intermediate multispectral fusion, improving primary foreground-overlap performance while retaining the complete four-band observation.

Keywords: Historical document binarization, Multispectral imaging, Cross-spectral feature fusion, Targeted structural guidance, Conditional generative adversarial networks

## 1 Introduction

Historical documents are important linguistic, cultural, and historical sources, yet digital analysis is complicated by severe, spatially nonuniform degradation. Faded ink, bleed-through, stains, uneven illumination, aging, surface damage, and low foreground–background contrast can obscure strokes and reduce readability. Binarization therefore requires both background suppression and recovery of weak, thin, or discontinuous strokes without introducing false foreground structures. This has driven progression from global and adaptive thresholding toward supervised deeplearning and adversarial restoration methods [1, 2], although severe degradation remains dificult when text–background separation is ambiguous in visible imagery.

Multispectral imaging (MSI) records the same document at multiple wavelengths, providing complementary evidence because ink, paper, stains, and degradation can respond diferently across bands. Earlier work showed that hyperspectral and multispectral observations improve degradeddocument visualization and restoration, with near-infrared (NIR) particularly useful when ink and substrate exhibit diferent spectral behavior [3–5]. This motivated multispectral textextraction and binarization methods, including the ICDAR 2015 MS-TEx benchmark [6], targetdetection methods [7], CNN-based binarization [8], and MSdB-NMF [9], establishing that spectral diversity provides information unavailable from a single conventional image.

However, exploiting multiple spectral bands introduces a second problem: observations must be combined to preserve complementary information rather than treating all channels equally. Existing multispectral document methods have predominantly relied on spectral-band or subspace selection, analytical transformations, source decomposition, target detection, factorization, or direct learned mappings from spectral input to binary output [7–9]. These approaches demonstrate the value of multispectral information, but leave less explored how modality-specific visible and NIR features should be adaptively integrated within a deep restoration network. More generally, multimodal learning has shown that fusion location afects branch representations and that intermediate fusion preserves modality-specific processing before joint reasoning [10].

Related developments in visible–infrared image fusion demonstrate the advantages of replacing fixed fusion rules with learned integration. Deep fusion models preserve complementary modality information, while adversarial formulations incorporate detail- and edge-sensitive objectives for structural preservation [11]. These results are architecturally relevant, although generic visible–infrared fusion commonly considers thermal imagery, whereas historical-document MSI uses reflected NIR observations. In parallel, document-binarization research increasingly incorporates structural priors. For example, GDB combines Sobel-derived edge information with gated convolutions [2], while recent multispectral research shows that selected wavelength responses and ink-contour information can be highly discriminative for separating foreground from substrate [12]. Collectively, these studies establish the value of learned gating, structural guidance, and wavelength-specific evidence, but largely investigate them separately.

Despite these advancements, existing literature remains bifurcated: generic vision frameworks investigate multimodal fusion strategies without considering document-specific stroke topology, whereas document restoration methods incorporate structural priors but operate largely on single-channel or RGB images. Crucially, no prior work has systematically investigated how, where, and under what structural guidance visible and near-infrared (VIS–NIR) spectral bands should be integrated within a learned intermediate-fusion strategy for historical document restoration and downstream text legibility.

This gap motivates the central objective of this work: to establish an adaptive, edgeguided multispectral fusion paradigm that bridges cross-spectral physics with structural text priors. Rather than forcing hard channel selection or unguided feature blending, we investigate whether a targeted VIS–NIR structural cue can dynamically regulate intermediate feature maps at encoder depth L = 3, suppressing severe background artifacts such as stains, foxing, and bleed-through while recovering degraded or missing character-stroke geometry for binarisation and OCR readability.

This study substantially extends our preliminary multispectral-fusion work [5], which compared early, intermediate, and late fusion and identified normalized additive fusion at encoder depth L = 3 as the strongest baseline. That study established where to fuse spectral information but not how to adaptively control fusion. This journal extension replaces fixed averaging with learned modality and structural gating, adds targeted visible–NIR edge guidance and edgeaware supervision, shifts to full-page binarization, and expands evaluation to five-fold documentlevel cross-validation with statistical and computational profiling.

The principal contributions of this work are summarized as follows:

• Adaptive VIS–NIR Edge-Guided Fusion Architecture: We propose a multi-branch framework for degraded document binarization that retains the full multispectral observation while employing an auxiliary visible–NIR structural representation to dynamically gate intermediate features at encoder depth $L = 3$

• Controlled Architectural & Spectral Source Ablation: We conduct a systematic comparative study of nine fusion configurations, including fixed intermediate averaging, learned visible/NIR gating, edge-guided gating (with and without structural loss), generic spatial/channel attention (Squeeze-and-Excitation and CBAM), four-band structural guidance, and targeted spectral-source variants $( F _ { 2 } + \mathrm { N I R }$ $F _ { 3 } { \mathrm { + N I R } }$ , and $F _ { 4 } { + } \mathrm { N I R } )$ . This isolates whether gains stem from adaptive fusion, generic attention, explicit structural conditioning, or specific spectral pairings.

• Rigorous Page-Level Validation & Statistical Profiling: All architectures are evaluated using strict five-fold document-level crossvalidation (ensuring zero patch leakage prior to extraction), validation-only threshold tuning, and full-page image reconstruction. Performance is assessed using standard DIBCO metrics $( F _ { 1 }$ , pseudo-F<sub>1</sub>, IoU, NRM, and DRD), with pseudo-F<sub>1</sub> and DRD independently verified against oficial DIBCO reference tools, and backed by paired Wilcoxon signed-rank tests with Holm multiple-comparison correction.

• Computational & Reconstruction Profiling: We characterize the proposed framework in terms of parameters, multiply–accumulate operations (MACs), floating-point operations (FLOPs), GPU memory footprint, patch-level inference latency, throughput, and overlapsensitivity during full-page reconstruction.

Empirical evaluations confirm that structural guidance derived from the 600 nm visible band and 1000 nm NIR band $( F _ { 3 } { + } \mathrm { N I R } )$ provides the strongest overall binarization performance among the evaluated configurations. Gains over the fixed intermediate averaging (Mid AVG) baseline are statistically significant after Holm correction for $F _ { 1 } ,$ , pseudo-F<sub>1</sub>, and IoU $\left( p < 0 . 0 5 \right)$ . These findings demonstrate that targeted cross-spectral structural guidance efectively regulates intermediate multispectral fusion without stroke fragmentation or spectral artifact transfer.

## 2 Related Work

## 2.1 Historical Document Binarization and Enhancement

Document image binarization separates foreground text from background while preserving stroke geometry. Classical methods range from global Otsu thresholding [13] to locally adaptive Sauvola thresholding [14], which better handles varying illumination. These remain useful baselines, but combined faded ink, bleed-through, stains, texture variation, and localized degradation are dificult for a single handcrafted rule.

Deep-learning methods increasingly formulate binarization as supervised image-to-image prediction. Vo et al. [15] introduced a hierarchical deeply supervised network in which multiple representation levels contribute to the binary output, combining coarse degradation suppression with fine boundary retention. Calvo-Zaragoza and Gallego [16] proposed a selectional auto-encoder that estimates foreground confidence directly, demonstrating end-to-end binarization across heterogeneous collections. DeepOtsu [17] instead iteratively produces a cleaner representation for conventional thresholding.

Adversarial learning further extended this direction. Zhao et al. [18] used cascaded conditional GANs for degraded-document binarization, while DE-GAN [1] supported binarization, deblurring, and watermark removal. Khamekhem Jemni et al. [19] added a recognition-oriented objective, showing that visually plausible restoration is insuficient if thin or degraded strokes are lost. These studies motivate reconstruction and adversarial objectives but do not explicitly exploit wavelength-dependent multispectral information.

Recent methods add stronger adaptive and structural mechanisms. GDB [2] combines an Otsu-derived prior, Sobel gradients, an edge branch, and gated convolutions, while RA-GAN [20] uses gradient-gated convolution, guided attention, and a gated feature pyramid. Both show that structural mechanisms improve binarization, but derive their priors from the same conventional input rather than complementary spectral observations.

## 2.2 Multispectral Historical Document Analysis

Multispectral and hyperspectral imaging provide registered observations at multiple wavelengths, allowing text, substrate, stains, and degradation to exhibit distinct responses. Kim et al. [3] used hyperspectral information to enhance old documents, while Hedjam and Cheriet [4] exploited visible and infrared behavior for restoration. Qureshi et al. [21] further review hyperspectral imaging for historical information recovery, ink analysis, forensics, and cultural heritage.

The ICDAR 2015 MultiSpectral Text Extraction contest established a benchmark for multispectral text extraction [6]. Early binarization methods relied on explicit spectral modeling. Farrahi Moghaddam and Cheriet [22] used multiple experts over diferent multispectral subspaces, showing that spectral combinations carry diferent discriminative information. MSIO [7] combined target detection with spatial segmentation, while Hollaus et al. [23] modeled foreground and background using Gaussian mixtures.

Later work adopted learned and factorized representations. Hollaus et al. [8] applied a CNN directly to multispectral binarization, establishing a neural baseline. Rahiche et al. [24] used graph-regularized non-negative matrix factorization for blind source separation, while MSdB-NMF [9] reduced multispectral or hyperspectral observations before conventional binarization. These methods show that spectral dimensionality can be exploited without treating all channels identically.

Recent work published in the International Journal on Document Analysis and Recognition (IJDAR) [12] provides relevant evidence that selected wavelength responses and structural information can be more discriminative than indiscriminate spectral combination. The study analyzed twelve multispectral observations of Dead Sea Scroll fragments and exploited discriminative spectral diferences together with inkcontour information in an energy-minimization segmentation framework. This work is close to the present study in using selected spectral evidence and structural contours, but its processing remains based on spectral thresholding and graphcut optimization. In contrast, the present work investigates whether a selected visible–NIR pair can provide an auxiliary structural representation that regulates learned intermediate feature fusion while the complete multispectral input remains available. Beyond pixel-level segmentation, such unified spectral-structural enhancement can benefit downstream optical character recognition (OCR) by removing severe background interference while reconstructing fragmented stroke geometry to improve machine text legibility.

## 2.3 Adaptive Multimodal and Visible–Infrared Fusion

Multimodal representation learning broadly categorizes feature integration into early fusion at the input layer, intermediate fusion within latent feature spaces, and late fusion at the decision or output stage [10]. Intermediate fusion is advantageous when distinct modalities require domain-specific feature extraction before cross-modal interaction. For instance, FuseNet [25] demonstrated the efectiveness of internally combining modality-specific feature streams within a convolutional architecture for RGB-D semantic segmentation.

To overcome static feature concatenation, adaptive paradigms dynamically weight multisource contributions based on context. Gated Multimodal Units (GMUs) [26] introduced multiplicative gating for context-dependent multimodal integration, providing a basis for replacing handcrafted fusion rules with learned representations. Attention mechanisms similarly ofer spatial and channel-wise recalibration. Squeeze-and-Excitation (SE) networks [27] model inter-channel dependencies to emphasize informative feature maps, whereas CBAM [28] combines channel and spatial attention. While efective, these modules operate on domain-agnostic feature maps rather than document-specific stroke topographies, providing baselines for assessing explicit structural guidance.

Visible–infrared (VIS–IR) fusion provides another relevant precedent. DenseFuse [29] combines deep encoder features to preserve complementary visible and infrared information, while FusionGAN [30] uses adversarial learning to retain salient infrared targets and visible textures. Both demonstrate learned blending of complementary spectral inputs.

A physical distinction must be drawn between generic visible–infrared fusion and historical document analysis. Conventional VIS–IR fusion predominantly processes thermal infrared emissions (e.g., long-wave IR for heat signatures), whereas historical document multispectral imaging relies on reflected near-infrared (NIR, ∼1000 nm) light. While deep adaptive fusion principles remain relevant, the physical basis of spectral complementarity difers: in document analysis, spectral contrast emerges from wavelength-dependent light absorption and scattering across ink pigments, aged substrates, stains, and bleed-through artifacts rather than thermal radiation.

## 2.4 Structural Guidance for Fusion and Binarization

Preserving edges and fine structures is a recurring concern in both multimodal image fusion and document binarization. In visible–infrared fusion, Ma et al. [11] incorporated detail-preservation and target-edge-enhancement objectives into an adversarial framework to reduce boundary degradation and retain complementary source information. This shows that structural constraints can improve learned multimodal fusion, but the objective is generic fused-image generation rather than foreground/background separation in degraded historical documents.

In document binarization, GDB [2] provides an algorithmic precedent by combining explicit Sobel edge maps with gated convolutions to maintain stroke integrity. Similarly, RA-GAN [20] utilizes gradient-dependent gating and guided attention to handle spatially heterogeneous degradations. However, in both frameworks, the structural control signal is extracted solely from a single conventional document observation. These architectures therefore lack access to decoupled spectral channels where complementary, background-free structural cues can be isolated.

Conversely, multispectral document research confirms that specific spectral bands, inter-band diferences, and ink-contour regions can be highly discriminative for foreground-background separation [12, 22]. Nevertheless, existing techniques typically restrict these spectral cues to hard band selection, heuristic thresholding, matrix factorization, or localized graph-cut segmentation, rather than using them as dynamic control mechanisms within deep neural architectures.

Collectively, prior literature establishes four insights: (i) multispectral observations reveal text and degradation cues inaccessible to standard RGB imaging; (ii) intermediate learned fusion models cross-modal interactions; (iii) gradientaware mechanisms protect fine character-stroke topology during binarization; and (iv) targeted spectral bands ofer superior structural clarity. However, these paradigms have largely evolved in isolation. This work addresses the unresolved intersection by using a targeted cross-spectral structural representation to govern intermediate multispectral feature fusion while preserving the full-spectrum observation.

## 3 Proposed Method

## 3.1 Problem Formulation

Let a multispectral historical document page be represented by four spatially registered spectral observations,

$$
\mathbf { X } = [ X _ { F 2 } , X _ { F 3 } , X _ { F 4 } , X _ { F 7 } ] \in \mathbb { R } ^ { H \times W \times 4 } ,\tag{1}
$$

where $X _ { F 2 } , ~ X _ { F 3 }$ , and $X _ { F 4 }$ denote visible wavelengths at 500, 600, and 700 nm, respectively, and $X _ { F 7 }$ corresponds to the 1000 nm near-infrared (NIR) band. The input is partitioned into a visible stream $\mathbf { X } _ { \mathrm { v i s } } = [ X _ { F 2 } , \bar { X } _ { F 3 } , X _ { F 4 } ] \in \mathbb { R } ^ { H \times W }$ ×3 and an NIR stream $\mathbf { \dot { X } } _ { \mathrm { n i r } } = X _ { F 7 } \in \mathbb { R } ^ { \mathbf { \bar { H } } \times W \times 1 }$

The objective is to estimate a continuous document representation $\widehat { \mathbf { Y } } \in [ - 1 , 1 ] ^ { H \times W }$ that separates foreground text from degraded background. A generator $G _ { \theta }$ with learnable weights θ maps the input streams and auxiliary structural prior E to the output space,

$$
\begin{array} { r } { \widehat { \mathbf { Y } } = G _ { \theta } \left( \mathbf { X } _ { \mathrm { v i s } } , \mathbf { X } _ { \mathrm { n i r } } , \mathbf { E } \right) . } \end{array}\tag{2}
$$

The auxiliary structural representation $\begin{array} { r l } { \mathbf { E } } & { { } = } \end{array}$ $ { \mathcal { S } } ( X _ { F 3 } , X _ { F 7 } )$ is constructed from a selected visible–NIR pair via a deterministic edgeextraction operator $\boldsymbol { \mathcal { S } } ( \cdot )$ (Section 3.2). E provides targeted structural guidance without discarding or replacing any original spectral band in X.

Rather than imposing fixed arithmetic rules, $G _ { \theta }$ extracts modality-specific features and performs adaptive fusion at intermediate depth $L =$

3. Let $\mathbf { H } _ { \mathrm { v i s } } ^ { ( L ) } , \mathbf { H } _ { \mathrm { n i r } } ^ { ( L ) }$ , and $\mathbf { H } _ { \mathrm { e d g e } } ^ { ( L ) }$ denote intermediate features from the visible, NIR, and structural branches, respectively. The fused representation is:

$$
\mathbf { H } _ { \mathrm { f u s e } } ^ { ( L ) } = \Phi _ { \theta } \left( \mathbf { H } _ { \mathrm { v i s } } ^ { ( L ) } , \mathbf { H } _ { \mathrm { n i r } } ^ { ( L ) } , \mathbf { H } _ { \mathrm { e d g e } } ^ { ( L ) } \right) ,\tag{3}
$$

where $\Phi _ { \theta } ( \cdot )$ is a learnable spatial and channel-wise adaptive gating operator (Section 3.4).

Full-page predictions are converted to binary maps $\widehat { B } _ { \tau } ( p ) \in \{ 0 , 1 \}$ using a global threshold $\tau { : }$

$$
\widehat { B } _ { \tau } ( p ) = \left\{ \begin{array} { l l } { { 0 } } & { { \mathrm { i f ~ } \widehat { Y } ( p ) < \tau \quad \mathrm { ( f o r e g r o u n d ~ t e x t ) } , } } \\ { { 1 } } & { { \mathrm { i f ~ } \widehat { Y } ( p ) \geq \tau \quad \mathrm { ( b a c k g r o u n d ) } , } } \end{array} \right.\tag{4}
$$

where $p$ denotes pixel location. Threshold τ is optimized on reconstructed validation pages and fixed during test evaluation.

## 3.2 Multispectral Preprocessing and Structural Edge Construction

To prevent data leakage, document-level splitting precedes all statistical estimation and patch operations. Normalization parameters are derived only from training documents within each fold.

For each band $b \in \{ F 2 , F 3 , F 4 , F 7 \}$ , minimum and maximum intensities $( m _ { b } , M _ { b } )$ are computed from the training set. Inputs are rescaled to $[ - 1 , 1 ]$ via:

$$
\widetilde { X } _ { b } = 2 \left( \frac { X _ { b } - m _ { b } } { M _ { b } - m _ { b } + \epsilon } \right) - 1 ,\tag{5}
$$

where $\epsilon = 1 0 ^ { - 8 }$ ensures numerical stability.

Raw ground-truth maps, $Y _ { \mathrm { r a w } }$ , are thresholded at 128 to enforce binary targets $Y ( p ) \in \{ 0 , 2 5 5 \}$ (0 for text, 255 for background). For training, $Y ( p )$ is normalized to $\widetilde { Y } \in [ - 1 , 1 ]$

$$
\widetilde { Y } ( p ) = 2 \left( \frac { Y ( p ) } { 2 5 5 } \right) - 1 .\tag{6}
$$

The discrete {0, 255} format is retained for padding and evaluation, while $\widetilde { Y }$ is used for loss calculation.

To prevent boundary artifacts during slidingwindow inference, pages are symmetrically padded before patch extraction. Spectral inputs use reflection padding (BORDER REFLECT 101), while target maps use a constant white value (255) to avoid artificial foreground edges.

The auxiliary structural representation E is generated at full-page resolution before patch extraction to avoid artificial boundary gradients. Construction comprises four steps:

1. Intensity Normalization and Channel Averaging: The 600 nm visible band $\left( X _ { F 3 } \right)$ and the 1000 nm NIR band $( X _ { F 7 } )$ are used specifically to construct the auxiliary structural representation. These two bands are independently normalized to the 8-bit dynamic range ([0, 255]) and combined using equalweight averaging. Importantly, this spectralpair selection applies only to the structuralmap construction; the complete multispectral observation, comprising $X _ { F 2 } , X _ { F 3 } , X _ { F 4 }$ , and $X _ { F 7 }$ , remains available as input to the network:

$$
\begin{array} { r } { I _ { \mathrm { V N } } = \displaystyle \frac { 1 } { 2 } \left[ 2 5 5 \cdot \frac { X _ { F 3 } - \operatorname* { m i n } ( X _ { F 3 } ) } { \operatorname* { m a x } ( X _ { F 3 } ) - \operatorname* { m i n } ( X _ { F 3 } ) + \epsilon } \right. } \\ { \displaystyle \left. + 2 5 5 \cdot \frac { X _ { F 7 } - \operatorname* { m i n } ( X _ { F 7 } ) } { \operatorname* { m a x } ( X _ { F 7 } ) - \operatorname* { m i n } ( X _ { F 7 } ) + \epsilon } \right] . } \end{array}\tag{7}
$$

2. Local Contrast Enhancement: Contrast-Limited Adaptive Histogram Equalization (CLAHE) is applied to $I _ { \mathrm { V N } }$ with clip limit $c = 2 . 0$ over an $8 \times 8$ grid to enhance local stroke transitions:

$$
\begin{array} { r } { I _ { \mathrm { C } } = \mathrm { C L A H E } \left( I _ { \mathrm { V N } } ; c = 2 . 0 , g = 8 \times 8 \right) . } \end{array}\tag{8}
$$

3. Gradient Magnitude Extraction: Horizontal $\left( G _ { x } \right)$ and vertical $\left( G _ { y } \right)$ gradients are computed using $3 \times 3$ Sobel operators, yielding magnitude $G \mathrm { : }$

$$
G = \sqrt { \left( S _ { x } ( I _ { \mathrm { C } } ) \right) ^ { 2 } + \left( S _ { y } ( I _ { \mathrm { C } } ) \right) ^ { 2 } } .\tag{9}
$$

4. Range Normalization: G is min–max rescaled to 8-bit space $E ^ { ( 8 ) }$ and mapped to $[ - 1 , 1 ] ;$

$$
\begin{array} { r } { \widetilde { E } = 2 \left( \displaystyle \frac { E ^ { ( 8 ) } } { 2 5 5 } \right) - 1 , \qquad } \\ { { \mathrm { w h e r e ~ } } E ^ { ( 8 ) } = 2 5 5 \cdot \displaystyle \frac { G - \operatorname* { m i n } ( G ) } { \operatorname* { m a x } ( G ) - \operatorname* { m i n } ( G ) + \epsilon } . } \end{array}\tag{10}
$$

Aligned spectral inputs $\mathbf { X } ,$ structural prior $\widetilde { E }$ and ground truth $\widetilde { Y }$ are extracted as $2 5 6 ~ \times$ 256 patches with 204-pixel stride $( \sim 2 0 \%$ overlap), preserving registration across channels.

## 3.3 Fixed Intermediate Fusion Baseline

The fixed intermediate-fusion architecture is the principal baseline for the proposed adaptive mechanism. It uses the intermediate scheme identified in our prior work [5], where normalized additive fusion at encoder depth $L \ = \ 3$ outperformed early- and late-fusion alternatives. The baseline is retrained and evaluated under the fivefold document-level cross-validation protocol in Section 4.2.

The multispectral input is partitioned into visible and near-infrared (NIR) streams,

$$
\mathbf { X } _ { \mathrm { v i s } } = [ X _ { F 2 } , X _ { F 3 } , X _ { F 4 } ] , \qquad \mathbf { X } _ { \mathrm { n i r } } = [ X _ { F 7 } ] .\tag{11}
$$

The streams are encoded independently through the first three stages of the dual-branch network. Let $\mathbf { H } _ { \mathrm { v i s } } ^ { ( 3 ) }$ and $\mathbf { H } _ { \mathrm { n i r } } ^ { ( 3 ) }$ denote the representations at depth $L = 3 \mathrm { : }$

$$
\mathbf { H } _ { \mathrm { v i s } } ^ { ( 3 ) } = E _ { \mathrm { v i s } } ^ { ( 3 ) } ( \mathbf { X } _ { \mathrm { v i s } } ) , \qquad \mathbf { H } _ { \mathrm { n i r } } ^ { ( 3 ) } = E _ { \mathrm { n i r } } ^ { ( 3 ) } ( \mathbf { X } _ { \mathrm { n i r } } ) .\tag{12}
$$

Both feature maps have identical spatial and channel dimensions, permitting element-wise averaging:

$$
\mathbf { H } _ { \mathrm { a v g } } ^ { ( 3 ) } = \frac { 1 } { 2 } \left( \mathbf { H } _ { \mathrm { v i s } } ^ { ( 3 ) } + \mathbf { H } _ { \mathrm { n i r } } ^ { ( 3 ) } \right) .\tag{13}
$$

This fixes the spatial and channel-wise fusion weights to $\alpha _ { \mathrm { v i s } } = \alpha _ { \mathrm { n i r } } = 0 . 5$ across all samples and locations.

The unified representation $\mathbf { H } _ { \mathrm { a v g } } ^ { ( 3 ) }$ passes through the remaining shared encoder stages $E _ { \mathrm { s h a r e d } }$ and U-Net decoder $D _ { \theta }$ to produce $\widehat { \mathbf { Y } } _ { \mathrm { a v g } } \colon$

$$
\begin{array} { r } { \widehat { \mathbf { Y } } _ { \mathrm { a v g } } = D _ { \theta } \Big ( E _ { \mathrm { s h a r e d } } \Big ( \mathbf { H } _ { \mathrm { a v g } } ^ { ( 3 ) } \Big ) \Big ) . } \end{array}\tag{14}
$$

Because averaging has no learnable parameters, it assigns static weights regardless of local degradation or spectral contrast. This motivates the dynamic, content-adaptive architecture introduced next.

## 3.4 Adaptive Visible–NIR Edge-Guided Fusion

To overcome static averaging, the proposed generator replaces the two-stream baseline with learned three-stream dynamic gating and explicit structural guidance. The network accepts the complete multispectral tensor and auxiliary visible–NIR structural map $\widetilde { E }$ (Section 3.2):

$$
{ \bf X } _ { G } = \left[ X _ { F 2 } , X _ { F 3 } , X _ { F 4 } , X _ { F 7 } , \widetilde { E } \right] .\tag{15}
$$

These channels form three feature-extraction branches:

$$
\begin{array} { r l } & { \mathbf { X } _ { \mathrm { v i s } } = [ X _ { F 2 } , X _ { F 3 } , X _ { F 4 } ] , } \\ & { \mathbf { X } _ { \mathrm { n i r } } = [ X _ { F 7 } ] , \quad \mathrm { a n d } \quad \mathbf { X } _ { \mathrm { e d g e } } = [ \widetilde { E } ] . } \end{array}\tag{16}
$$

Each branch has three encoder stages, expanding channel capacity from 64 to 128 and 256. For a $2 5 6 \times 2 5 6$ patch, the representations at depth $L = 3$ are:

$$
\begin{array} { r } { \mathbf { H } _ { m } ^ { ( 3 ) } = E _ { m } ^ { ( 3 ) } ( \mathbf { X } _ { m } ) \in \mathbb { R } ^ { C \times H \times W } , } \\ { m \in \{ \mathrm { v i s } , \mathrm { n i r } , \mathrm { e d g e } \} , } \end{array}\tag{17}
$$

where $C = 2 5 6$ and $H = W = 3 2$

To compute context-dependent fusion weights, the feature tensors are concatenated along the channel dimension:

$$
\mathbf { H } _ { \mathrm { c a t } } = \mathrm { C o n c a t } \Big ( \mathbf { H } _ { \mathrm { v i s } } ^ { ( 3 ) } , \mathbf { H } _ { \mathrm { n i r } } ^ { ( 3 ) } , \mathbf { H } _ { \mathrm { e d g e } } ^ { ( 3 ) } \Big ) \in \mathbb { R } ^ { 3 C \times H \times W } .\tag{18}
$$

A $1 \times 1$ convolution maps the concatenated features to a logit tensor $\mathbf { Z } \in \mathbb { R } ^ { 3 C \times H \times W }$ <sup>W</sup> :

$$
{ \bf Z } = \mathrm { C o n v } _ { 1 \times 1 } ( { \bf H } _ { \mathrm { c a t } } ) .\tag{19}
$$

Reshaping Z yields modality-specific logits $[ \mathbf { Z } _ { \mathrm { v i s } } , \mathbf { \bar { Z } } _ { \mathrm { n i r } } , \mathbf { Z } _ { \mathrm { e d g e } } ] \quad \in \quad \mathbb { R } ^ { C \times H \times W }$ . Spatial- and channel-wise Softmax normalization is applied across $\mathcal { M } = \{ \mathrm { v i s } , \mathrm { n i r } , \mathrm { e d g e } \}$ :

$$
A _ { m } ( c , p ) = \frac { \exp { ( Z _ { m } ( c , p ) ) } } { \displaystyle \sum _ { k \in \mathcal { M } } \exp { ( Z _ { k } ( c , p ) ) } } , \quad m \in \mathcal { M } ,\tag{20}
$$

where $c \in \{ 1 , \ldots , C \}$ denotes channel and $p \in$ $\{ 1 , \ldots , H \times W \}$ spatial location. This guarantees unit-simplex normalization at each coordinate:

$$
A _ { \mathrm { v i s } } ( c , p ) + A _ { \mathrm { n i r } } ( c , p ) + A _ { \mathrm { e d g e } } ( c , p ) = 1 .\tag{21}
$$

The fused representation $\mathbf { H } _ { \mathrm { f u s e } } ^ { ( 3 ) }$ is formed using element-wise Hadamard products (⊙):

$$
\mathbf { H } _ { \mathrm { f u s e } } ^ { ( 3 ) } = \mathbf { A } _ { \mathrm { v i s } } \odot \mathbf { H } _ { \mathrm { v i s } } ^ { ( 3 ) } + \mathbf { A } _ { \mathrm { n i r } } \odot \mathbf { H } _ { \mathrm { n i r } } ^ { ( 3 ) } + \mathbf { A } _ { \mathrm { e d g e } } \odot \mathbf { H } _ { \mathrm { e c } } ^ { ( 3 ) }\tag{}
$$

(22) <sub>dge</sub>.

After fusion at depth $L = 3$ , the streams merge and pass through the remaining shared encoder stages. A U-Net decoder with skip connections restores spatial resolution and produces the final single-channel output using a hyperbolic tangent (tanh) activation:

$$
\begin{array} { r } { \widehat { \mathbf { Y } } = G _ { \theta } ( \mathbf { X } _ { \mathrm { v i s } } , \mathbf { X } _ { \mathrm { n i r } } , \mathbf { X } _ { \mathrm { e d g e } } ) . } \end{array}\tag{23}
$$

Unlike fixed averaging, this formulation allocates spatial and channel attention according to local signal quality, favoring visible features when strokes are clear, shifting toward NIR under severe visible-spectrum staining, or using $\mathbf { H } _ { \mathrm { e d g e } } ^ { ( 3 ) }$ near degraded boundaries to preserve stroke topology.

## 3.5 Conditional Adversarial Learning Objective

The framework is optimized as a supervised cGAN inspired by Pix2Pix [31, 32]. Generator $G _ { \theta }$ maps $\begin{array} { l l l } { { { \bf { X } } _ { G } } } & { { = } } & { { [ X _ { F 2 } , X _ { F 3 } , X _ { F 4 } , X _ { F 7 } , { \widetilde { E } } ] } } \end{array}$ to a restored single-channel representation, while PatchGAN discriminator $D _ { \phi }$ distinguishes generated outputs from paired ground truth conditioned on $\mathbf { X } _ { G }$

The conditional adversarial objective is:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { c G A N } } ( G , D ) = \mathbb { E } _ { \mathbf { X } _ { G } , \mathbf { Y } } \left[ \log D _ { \phi } ( \mathbf { X } _ { G } , \mathbf { Y } ) \right] } \\ & { \qquad + \mathbb { E } _ { \mathbf { X } _ { G } } \left[ \log \left( 1 - D _ { \phi } ( \mathbf { X } _ { G } , G _ { \theta } ( \mathbf { X } _ { G } ) ) \right) \right] . } \end{array}\tag{24}
$$

To enforce pixel-level alignment and preserve high-frequency stroke detail, the objective includes an $\ell _ { 1 }$ reconstruction loss:

$$
\begin{array} { r } { \mathcal { L } _ { 1 } ( G ) = \mathbb { E } _ { \mathbf { X } _ { G } , \mathbf { Y } } [ \| \mathbf { Y } - G _ { \theta } ( \mathbf { X } _ { G } ) \| _ { 1 } ] . } \end{array}\tag{25}
$$

The $\ell _ { 1 }$ norm provides a robust penalty that reduces blurring along thin text boundaries compared with squared error.

The total generator objective combines adversarial, pixel-level reconstruction, and edge-aware structural constraints:

$$
\mathcal { L } _ { G } = \mathcal { L } _ { \mathrm { a d v } } + \lambda _ { 1 } \mathcal { L } _ { 1 } + \lambda _ { \mathrm { e d g e } } \mathcal { L } _ { \mathrm { e d g e } } ,\tag{26}
$$

where $\lambda _ { 1 } = 1 0 0$ and $\lambda _ { \mathrm { e d g e } } = 5$ are fixed hyperparameters.

Binary cross-entropy (BCE) supervises adversarial updates. The discriminator loss $\mathcal { L } _ { D }$ and generator adversarial term ${ \mathcal { L } } _ { \mathrm { a d v } }$ are:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { D } = \frac { 1 } { 2 } \mathrm { B C E } ( D _ { \phi } ( \mathbf { X } _ { G } , \mathbf { Y } ) , \mathbf { 1 } ) } \\ { \displaystyle \phantom { \frac { 1 } { 2 } \mathrm { B C E } \big ( D _ { \phi } ( \mathbf { X } _ { G } , \widehat { \mathbf { Y } } ) , \mathbf { 0 } \big ) , } } \end{array}\tag{27}
$$

$$
\mathcal { L } _ { \mathrm { a d v } } = \mathrm { B C E } \Big ( D _ { \phi } ( \mathbf { X } _ { G } , \widehat { \mathbf { Y } } ) , \mathbf { 1 } \Big ) .\tag{28}
$$

where $\widehat { \mathbf { Y } } = G _ { \theta } ( \mathbf { X } _ { G } )$ . Optimization alternates between updating $D _ { \phi }$ with detached generated outputs and updating $G _ { \theta }$ by end-to-end backpropagation.

## 3.6 Edge-Aware Structural Loss

To penalize structural and boundary errors, an edge-consistency loss $\mathcal { L } _ { \mathrm { e d g e } }$ is added to the objective. Tensors normalized to $[ - 1 , 1 ]$ are rescaled to [0, 1] via $\begin{array} { r } { \mathbf { Y } _ { 0 1 } = \frac { 1 } { 2 } ( \mathbf { Y } + 1 ) } \end{array}$ and $\begin{array} { r } { \widehat { \mathbf { Y } } _ { 0 1 } = \frac { 1 } { 2 } ( \widehat { \mathbf { Y } } + 1 ) } \end{array}$ .

Gradient magnitudes $\mathcal { G } ( \mathbf { I } ) \in \mathbb { R } ^ { H \times \tilde { W } }$ are computed using Sobel operators $S _ { x }$ and $S _ { y } { \mathrm { : } }$

$$
\mathcal { G } ( \mathbf { I } ) = \sqrt { ( S _ { x } * \mathbf { I } ) ^ { 2 } + ( S _ { y } * \mathbf { I } ) ^ { 2 } + \epsilon } ,\tag{29}
$$

where ∗ denotes spatial convolution and $\epsilon = 1 0 ^ { - 8 }$ ensures numerical stability during backpropagation.

With $\mathbf { G } _ { \mathrm { p r e d } } = \mathcal { G } ( \widehat { \mathbf { Y } } _ { 0 1 } )$ and $\mathbf { G } _ { \mathrm { g t } } = \mathcal { G } ( \mathbf { Y } _ { 0 1 } )$ , the edge-aware loss measures mean absolute gradient diference:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { e d g e } } = \mathbb { E } \big [ \| \mathbf { G } _ { \mathrm { p r e d } } - \mathbf { G } _ { \mathrm { g t } } \| _ { 1 } \big ] . } \end{array}\tag{30}
$$

While $\mathcal { L } _ { 1 }$ enforces global intensity fidelity, $\mathcal { L } _ { \mathrm { e d g e } }$ penalizes character erosion, stroke fragmentation, and boundary degradation. The complete generator loss is:

$$
\mathcal { L } _ { G } = \mathcal { L } _ { \mathrm { a d v } } + 1 0 0 \mathcal { L } _ { 1 } + 5 \mathcal { L } _ { \mathrm { e d g e } } .\tag{31}
$$

The reconstruction weight $\lambda _ { 1 } ~ = ~ 1 0 0$ follows Pix2Pix [31] to retain pixel-level fidelity and baseline consistency. The fixed moderate weight $\lambda _ { \mathrm { e d g e } } ~ = ~ 5$ allows structural supervision to contribute without dominating adversarial and reconstruction objectives.

Structural guidance acts in two complementary ways: (i) input-side channel $\widetilde { E }$ drives adaptive gating (Section 3.4), and (ii) output-side constraint $\mathcal { L } _ { \mathrm { e d g e } }$ supervises boundary integrity.

## 3.7 Full-Page Reconstruction and Inference

To evaluate global topology and reduce patchboundary artifacts, inference uses tiled full-page soft reconstruction. Padded documents are partitioned into aligned 256 × 256 patches with stride s = 204 pixels (∼ 20% overlap).

Let $\mathcal { P } = \{ P _ { k } \} _ { k = 1 } ^ { K }$ denote the set of extracted patches for a document page. Each patch is processed independently by the generator to yield soft continuous predictions ${ \widehat { P } } _ { k } = { \cal \tilde { G } } _ { \theta } ( P _ { k } )$ . Soft outputs are accumulated into a page-level intensity matrix A and a spatial coverage counter matrix $\mathbf { C } \mathrm { : }$

$$
\widehat { \mathbf { Y } } _ { \mathrm { s o f t } } ( p ) = \frac { \mathbf { A } ( p ) } { \mathbf { C } ( p ) } = \frac { \sum _ { k \in \mathcal { K } ( p ) } \widehat { P } _ { k } ( p ) } { | \mathcal { K } ( p ) | } ,\tag{32}
$$

where $\mathcal { K } ( \boldsymbol { p } )$ denotes patches overlapping coordinate $p .$

Overlap averaging suppresses boundary seams and edge discontinuities. After cropping margins, the final binary map $\widehat { B } _ { \tau } ( p ) \in \{ 0 , 1 \}$ is obtained by global thresholding:

$$
\widehat { B } _ { \tau } ( p ) = \left\{ \begin{array} { l l } { \scriptstyle 0 ( \mathrm { f o r e g r o u n d } ) , } & { \mathrm { i f } \widehat { \mathbf { Y } } _ { \mathrm { s o f t } } ( p ) < \tau , } \\ { \scriptstyle 1 ( \mathrm { b a c k g r o u n d } ) , } & { \mathrm { i f } \widehat { \mathbf { Y } } _ { \mathrm { s o f t } } ( p ) \geq \tau . } \end{array} \right.\tag{33}
$$

The threshold $\tau$ is tuned only on reconstructed validation pages (Section 3.8) and applied unchanged to the test fold.

## 3.8 Validation-Based Model and Threshold Selection

To prevent data leakage, model and threshold selection use only the validation partition of each fold. Held-out test documents remain unseen during optimization and are evaluated once per fold.

Checkpoint screening uses validation epochs with soft outputs binarized at baseline threshold $\tau = 0 . 5$ . Let $\bar { F } _ { 1 } ^ { ( e ) }$ and $\overline { { \mathrm { S S I M } } } ^ { ( e ) }$ denote mean validation text $F _ { 1 }$ and structural similarity at epoch e. Optimal parameters $\theta ^ { * } = \theta ^ { ( e ^ { * } ) }$ are selected by lexicographic maximization:

$$
e ^ { * } = \arg \operatorname* { m a x } _ { e } \left( \overline { { F } } _ { 1 } ^ { \left( e \right) } , \overline { { \mathrm { S S I M } } } ^ { \left( e \right) } \right) ,\tag{34}
$$

where $F _ { 1 }$ is the primary criterion and SSIM resolves ties.

Following checkpoint freeze at $G _ { \theta ^ { * } }$ , validation pages are reconstructed following Section 3.7 and mapped to [0, 1] via $\begin{array} { r } { \widehat { Y } _ { 0 1 } = \frac { 1 } { 2 } ( \widehat { Y } + 1 ) } \end{array}$ . The binarization threshold $\tau ^ { * }$ is optimized over a discretized candidate set $\mathcal { T } = \{ 0 . 1 0 , 0 . 1 1 , \ldots , 0 . 9 0 \}$ with a step size of 0.01. Soft continuous predictions are thresholded to yield binary maps $\widehat { B } _ { \tau } ( p ) \in \{ 0 , 1 \}$ (0 = foreground, 1 = background):

$$
\widehat { B } _ { \tau } ( p ) = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ } \widehat { Y } _ { 0 1 } ( p ) < \tau , } \\ { 1 , } & { \mathrm { i f ~ } \widehat { Y } _ { 0 1 } ( p ) \geq \tau . } \end{array} \right.\tag{35}
$$

Threshold optimization prioritizes boundary and stroke-topology preservation. Let $\overline { { F } } _ { \mathrm { p s , v a l } } ( \tau )$ and $\overline { { F } } _ { \mathrm { 1 , v a l } } ( \tau )$ denote mean validation pseudo- $F _ { 1 }$ and text $F _ { 1 }$ at threshold $\tau .$ . The optimal $\tau ^ { * }$ is selected by lexicographic maximization:

$$
\tau ^ { * } = \arg \operatorname* { m a x } _ { \tau \in \mathcal { T } } \left( \overline { { F } } _ { \mathrm { p s , v a l } } ( \tau ) , \overline { { F } } _ { 1 , \mathrm { v a l } } ( \tau ) \right) .\tag{36}
$$

Here, pseudo- $F _ { 1 }$ acts as the primary driver, with standard text $F _ { 1 }$ serving as a secondary tiebreaker. (Note that this internal pseudo- $F _ { 1 }$ screening metric is restricted strictly to parameter tuning; final reported p-FM metrics are computed independently via the standardized DIBCO evaluation toolkit described in Section 4.5).

Once locked, $( G _ { \theta ^ { * } } , \tau ^ { * } )$ is applied directly to the unseen test set:

$$
\widehat { B } _ { \mathrm { t e s t } } =  { B _ { \tau ^ { * } } } \left( G _ { \theta ^ { * } } ( \mathbf { X } _ { \mathrm { t e s t } } ) \right) .\tag{37}
$$

This two-stage validation is repeated independently across all five folds, so test metrics use model and threshold choices based only on training and validation data.

## 4 Experimental Protocol

## 4.1 Dataset and Spectral Configurations

Experiments are conducted on the Synchromedia Multispectral Ancient Document Images Dataset (SMADI), associated with the ICDAR 2015 MultiSpectral Text Extraction (MS-TEx) benchmark [6]. The dataset provides spatially aligned multispectral observations of degraded historical pages with expert-annotated binary reference maps.

Following [5], four registered bands are retained for 30 paired pages: visible channels at 500 nm (F2), 600 nm (F3), and 700 nm (F4), plus a 1000 nm NIR channel (F7). The observation vector is ${ \bf X } = [ X _ { F 2 } , X _ { F 3 } , X _ { F 4 } , X _ { F 7 } ]$ , partitioned into visible $\mathbf { X } _ { \mathrm { v i s } } = [ X _ { F 2 } , X _ { F 3 } , X _ { F 4 } ]$ and NIR ${ \bf X } _ { \mathrm { n i r } } =$ $[ X _ { F 7 } ]$ streams. The $F 7$ channel is reflected NIR rather than thermal emission.

The four-band input X is retained across all edge-guided models. Band selection is restricted to constructing the auxiliary structural representation. In the primary architecture, the F3 (600 nm) and $F 7$ (1000 nm) channels generate the auxiliary edge map, while the full input X remains accessible to the network encoder.

## 4.2 Five-Fold Document-Level Cross-Validation

To prevent data leakage and evaluate full-page reconstruction, experiments use a strict five-fold cross-validation partitioned at the document level. The 30 SMADI pages are divided into five mutually exclusive 6-document folds: Fold 1 (z92, z60, z802, z65, z38, z43), Fold 2 (z95, z82, z582, z100, z34, z64), Fold 3 (z35, z59, z58, z80, z27, z30), Fold 4 (z822, z31, z76, z90, z67, z97), and Fold 5 (z70, z37, z48, z592, z68, z36).

For run $k \in \{ 1 , \ldots , 5 \}$ , Fold k is test $( N _ { \mathrm { t e s t } } =$ 6), the cyclic successor is validation $( N _ { \mathrm { v a l } } ~ =$ 6), and the remaining three folds are training $( N _ { \mathrm { t r a i n } } = 1 8 )$ . The partitions are disjoint: $\mathcal { D } _ { \mathrm { t r a i n } } \cap$ ${ \mathcal D } _ { \mathrm { v a l } } ~ = ~ { \mathcal D } _ { \mathrm { t r a i n } } \cap { \mathcal D } _ { \mathrm { t e s t } } ~ = ~ { \mathcal D } _ { \mathrm { v a l } } \cap { \mathcal D } _ { \mathrm { t e s t } } ~ = ~ { \mathcal D }$ Normalization and patch extraction occur only after assignment. Validation selects checkpoints and thresholds; test pages remain untouched until evaluation, yielding 30 independent reconstructed test pages per architecture.

## 4.3 Compared Fusion Architectures

To isolate the efects of adaptive fusion, explicit edge conditioning, and spectral pairing, nine structural configurations are evaluated with identical training schedules, fusion depths $( L = 3 )$ and validation pipelines (summarized in Table 1).

Configuration 1, Mid AVG, uses fixed equalweight fusion; Configuration 2 adds learned Visible/NIR gating without structural priors. Configurations 3–4 use four-channel edge maps to separate structural conditioning $( \lambda _ { \mathrm { e d g e } } = 0 )$ from edge supervision $( \lambda _ { \mathrm { e d g e } } = 5 )$ . Configurations $5 { - } 6$ use CBAM [28] and SE [27] as generic attention baselines. Configurations $_ { 7 - 9 }$ keep identical topology $( \lambda _ { \mathrm { e d g e } } = 5 )$ while pairing 1000 nm NIR with F2, F3, or F4.

## 4.4 Training Protocol

Networks are optimized independently in each fold using aligned 256 × 256 patches. Data augmentation (horizontal and vertical flips, $p = 0 . 5 )$ is applied synchronously to input bands, structural maps, and ground-truth targets during training only.

Models are trained for 100 epochs per fold with the Adam optimizer $( \eta = 2 \times 1 0 ^ { - 4 } , \beta _ { 1 } = 0 . 5$ $\beta _ { 2 } = 0 . 9 9 9$ , batch size $B = 8 ,$ random seed 42). The generative objective is defined as:

$$
\mathcal { L } _ { G } = \mathcal { L } _ { \mathrm { a d v } } + 1 0 0 \mathcal { L } _ { 1 } + \lambda _ { \mathrm { e d g e } } \mathcal { L } _ { \mathrm { e d g e } } ,\tag{38}
$$

where $\lambda _ { \mathrm { e d g e } } = 5$ for edge-loss models and 0 otherwise. Generator checkpoints are selected using validation text $F _ { 1 }$ (using validation SSIM as a tiebreaker). Binarization thresholds are tuned only on validation pages reconstructed via overlapping patches (stride 204 pixels, ∼20% overlap).

## 4.5 Evaluation Metrics

Evaluation uses full-page binarized outputs $( 0 =$ foreground text, 1 = background) with five complementary metrics:

• Text $F _ { 1 }$ Score: Geometric precision-recall balance, $\begin{array} { r } { F _ { 1 } = \frac { 2 T \prime P } { 2 T P + F P + F N } . } \end{array}$

• Intersection over Union (IoU): Direct spatial foreground overlap, $\begin{array} { r } { \mathrm { l o U } = \frac { \imath \mathit { \Omega } _ { 1 } } { T P + F P + F N } } \end{array}$ TP

• Pseudo-FMeasure (p-FM): Characterstructure-aware metric prioritizing stroke boundary fidelity [33]. Computed with an oficial DIBCO-compatible reference pipeline.

• Negative Rate Metric (NRM): Balanced error metric, $\begin{array} { r } { \mathrm { N R M } ~ = ~ \frac { 1 } { 2 } \left( \frac { F N } { T P + F N } + \frac { F P } { F P + T N } \right) } \end{array}$ Lower is better.

• Distance-Reciprocal Distortion (DRD): Local structural distortion metric weighted by

Table 1 Comparative architectural study (L = 3, evaluated under 5-fold document-level cross-validation).
<table><tr><td>No.</td><td>Configuration</td><td>Structural source</td><td> $\lambda _ { \mathrm { e d g e } }$ </td><td>Primary experimental objective</td></tr><tr><td>1</td><td>Mid AVG</td><td>None</td><td>0</td><td>Fixed intermediate fusion baseline (equal- weight feature averaging).</td></tr><tr><td>2</td><td>Visible/NIR gated fusion</td><td>None</td><td>0</td><td>Evaluates unguided, learned modality- dependent gating.</td></tr><tr><td>3</td><td>Four-band edge-guided gate</td><td> $F 2 + F 3 + F 4 + F 7$ </td><td>0</td><td>Tests structural input conditioning without edge-loss supervision.</td></tr><tr><td>4</td><td>Four-band edge-guided gate</td><td> $F 2 + F 3 + F 4 + F 7$ </td><td>5</td><td>Evaluates four-band structural guidance with explicit edge supervision.</td></tr><tr><td>5</td><td>CBAM fusion</td><td>None</td><td>0</td><td>Baseline for domain-agnostic spatial and channel attention [28].</td></tr><tr><td>6</td><td>SE fusion</td><td>None</td><td>0</td><td>Baseline for domain-agnostic channel recal-</td></tr><tr><td>7</td><td>F2+NIR edge-guided gate</td><td> $F 2 + F 7$ </td><td>5</td><td>ibration [27]. Evaluates structural guidance derived from</td></tr><tr><td>8</td><td>F3+NIR edge-guided gate</td><td> $F 3 + F 7$ </td><td>5</td><td>500 nm VIS + 1000 nm NIR. Primary proposed configuration (600 nm</td></tr><tr><td>9</td><td>F4+NIR edge-guided gate</td><td> $F 4 + F 7$ </td><td>5</td><td> $\mathrm { V I S } + 1 0 0 0 \mathrm { n m } \mathrm { N I R } ) .$  Evaluates structural guidance derived from 700 nm VIS + 1000 nm NIR.</td></tr></table>

$8 \times 8$ ground-truth stroke neighborhoods [34], $\begin{array} { r } { \mathrm { D R D } = \frac { \sum _ { k = 1 } ^ { S } \mathrm { D R D } _ { k } } { N _ { \mathrm { U B N } } } } \end{array}$ . Computed with an oficial DIBCO reference tool. Lower is better.

## 4.6 Statistical Analysis

Inferential statistics are calculated over the 30 paired full-page predictions. Non-parametric paired two-sided Wilcoxon signed-rank tests [35] compare each candidate architecture with the Mid AVG baseline.

Multiple testing is controlled using the Holm procedure [36] at family-wise significance level α = 0.05, applied independently to the eight candidate comparisons within each metric family. Efect sizes are quantified by matched rank-biserial correlation, with document win/tie/loss counts and 10,000-resample bootstrap 95% confidence intervals for the primary F3+NIR architecture.

## 4.7 Classical Binarization Baselines

Otsu [13] and Sauvola [14] are evaluated as nonlearning baselines on the 8-bit rescaled visible mean $\begin{array} { l c l } { { I _ { \mathrm { v i s } } } } & { { = } } & { { \frac { 1 } { 3 } ( X _ { F 2 } + X _ { F 3 } + X _ { F 4 } ) } } \end{array}$ . Sauvola parameters $( w , k )$ are selected per fold using validation pseudo-F over $w \in \{ 1 5 , 2 5 , 3 5 , 5 1 \}$ and $k \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 \}$ , yielding (15, 0.2), (25, 0.2), (15, 0.1), (15, 0.2), and (15, 0.2) for Folds 1–5.

Classical results report $F _ { 1 } { \mathrm { . } }$ , IoU, and NRM for consistency with non-reference validated outputs.

## 5 Results and Discussion

## 5.1 Overall Comparison of Fusion Architectures

Table 2 summarizes full-page performance over 30 independently held-out pages across five folds. Values are mean ± standard deviation; p-FM and DRD use the independently validated DIBCOcompatible implementations in Section 4.5. Higher F1, p-FM, and IoU and lower NRM and DRD are preferred.

The F3+NIR edge-guided configuration gives the strongest overall performance, with mean F1=0.7916, p-FM=0.8044, and IoU=0.6654. Relative to Mid AVG, the absolute gains are 0.0269, 0.0360, and 0.0321, respectively; mean NRM decreases from 0.1285 to 0.1115 and DRD from 5.9281 to 5.7007.

The performance pattern nevertheless shows that no single architecture is optimal for every error criterion. The F2+NIR configuration achieves the lowest NRM of 0.1064, indicating the most favorable balance between foreground and background classification errors according to this measure. In contrast, the four-band edge-guided model with $\lambda _ { \mathrm { e d g e } } = 5$ obtains the lowest DRD of

Table 2 Full-page binarization performance of the nine fusion architectures over 30 held-out SMADI documents. Values are reported as mean ± standard deviation. Best values for each metric are shown in bold.
<table><tr><td>Architecture</td><td>F1 ↑</td><td> $\mathbf { p } { \mathbf { - F M \uparrow } }$ </td><td>IoU ↑</td><td>NRM↓</td><td>DRD ↓</td></tr><tr><td>Mid AVG  $\left( L = 3 \right)$ </td><td> $0 . 7 6 4 7 \pm 0 . 1 2 3 2$ </td><td> $0 . 7 6 8 4 \pm 0 . 1 1 3 4$ </td><td> $0 . 6 3 3 3 \pm 0 . 1 4 8 6$ </td><td> $0 . 1 2 8 5 \pm 0 . 0 8 7 1$ </td><td> $5 . 9 2 8 1 \pm 3 . 1 9 0 1$ </td></tr><tr><td> $\mathrm { V i s i b l e / N I R ~ G a t e d ~ } ( L = 3 )$ </td><td> $0 . 7 6 5 1 \pm 0 . 1 6 9 4$ </td><td> $0 . 7 6 6 3 \pm 0 . 1 6 3 0$ </td><td> $0 . 6 4 1 4 \pm 0 . 1 6 7 9$ </td><td> $0 . 1 1 6 2 \pm 0 . 1 0 1 7$ </td><td> $6 . 0 2 8 4 \pm 3 . 7 6 9 9$ </td></tr><tr><td> $\mathrm { E d g e 4 B a n d ~ G a t e } ~ ( \lambda _ { \mathrm { e d g e } } = 0 )$ </td><td> $0 . 7 6 1 6 \pm 0 . 1 4 1 6$ </td><td> $0 . 7 6 9 4 \pm 0 . 1 3 1 5$ </td><td> $0 . 6 3 3 0 \pm 0 . 1 6 4 1$ </td><td> $0 . 1 3 0 7 \pm 0 . 0 9 4 6$ </td><td> $5 . 9 9 1 2 \pm 3 . 4 9 9 3$ </td></tr><tr><td>Edge4Band Gate  $( \lambda _ { \mathrm { e d g e } } = 5 )$ </td><td> $0 . 7 8 5 3 \pm 0 . 1 1 7 4$ </td><td> $0 . 8 0 2 1 \pm 0 . 1 0 2 9$ </td><td> $0 . 6 6 0 0 \pm 0 . 1 4 6 1$ </td><td> $0 . 1 2 2 7 \pm 0 . 0 8 5 5$ </td><td> ${ \bf 5 . 2 5 3 7 \pm 2 . 7 6 4 8 }$ </td></tr><tr><td>CBAM Fusion  $\left( L = 3 \right)$ </td><td> $0 . 7 5 0 2 \pm 0 . 1 7 5 0$ </td><td> $0 . 7 5 5 7 \pm 0 . 1 6 2 6$ </td><td> $0 . 6 2 4 7 \pm 0 . 1 8 2 6$ </td><td> $0 . 1 3 1 8 \pm 0 . 1 1 1 0$ </td><td> $6 . 0 3 2 4 \pm 3 . 3 9 2 4$ </td></tr><tr><td>SE Fusion  $\left( L = 3 \right)$ </td><td> $0 . 7 6 5 2 \pm 0 . 1 6 4 7$ </td><td> $0 . 7 7 3 7 \pm 0 . 1 5 3 5$ </td><td> $0 . 6 4 2 0 \pm 0 . 1 7 5 1$ </td><td> $0 . 1 2 6 0 \pm 0 . 1 0 2 9$ </td><td> $5 . 8 0 5 3 \pm 3 . 8 2 2 6$ </td></tr><tr><td>F2+NIR Edge Gate  $( \lambda _ { \mathrm { e d g e } } = 5 )$ </td><td> $0 . 7 8 4 2 \pm 0 . 1 0 9 0$ </td><td> $0 . 7 9 3 5 \pm 0 . 0 9 6 3$ </td><td> $0 . 6 5 6 8 \pm 0 . 1 3 6 0$ </td><td> $\mathbf { 0 . 1 0 6 4 \pm 0 . 0 7 0 4 }$ </td><td> $6 . 3 8 1 0 \pm 5 . 2 0 2 7$ </td></tr><tr><td>F3+NIR Edge Gate  $( \lambda _ { \mathrm { e d g e } } = 5 )$ </td><td> $\mathbf { 0 . 7 9 1 6 \pm 0 . 0 9 8 8 }$ </td><td> $\mathbf { 0 . 8 0 4 4 \pm 0 . 0 8 6 2 }$ </td><td> $\mathbf { 0 . 6 6 5 4 \pm 0 . 1 3 0 4 }$ </td><td> $0 . 1 1 1 5 \pm 0 . 0 6 9 7$ </td><td> $5 . 7 0 0 7 \pm 3 . 7 9 9 0$ </td></tr><tr><td>F4+NIR Edge Gate  $( \lambda _ { \mathrm { e d g e } } = 5 )$ </td><td> $0 . 7 7 4 6 \pm 0 . 1 3 7 1$ </td><td> $0 . 7 8 6 6 \pm 0 . 1 1 7 2$ </td><td> $0 . 6 4 9 5 \pm 0 . 1 6 1 4$ </td><td> $0 . 1 2 5 9 \pm 0 . 0 9 6 7$ </td><td> $5 . 4 3 1 3 \pm 3 . 0 3 3 8$ </td></tr></table>

5.2537, suggesting stronger suppression of locally weighted structural distortions. The $F 4 { + } \mathrm { N I R }$ model also achieves a relatively low DRD of 5.4313 despite ranking below the F3+NIR model in F1, p-FM, and IoU. These diferences reinforce the importance of assessing document binarization using complementary structure-sensitive metrics rather than a single aggregate score.

A clear efect is also observed when the four-band structural-input configuration is trained with and without the edge-aware loss. With $\lambda _ { \mathrm { e d g e } } = 0 _ { \mathrm { : } }$ the model obtains an F1 of 0.7616, p-FM of 0.7694, IoU of 0.6330, NRM of 0.1307, and DRD of $5 . 9 9 1 2$ . Introducing the edge-consistency term with $\lambda _ { \mathrm { e d g e } } ~ = ~ 5$ increases the corresponding F1, p-FM, and IoU values to 0.7853, 0.8021, and 0.6600, respectively, while reducing NRM to 0.1227 and DRD to 5.2537. Under the controlled architecture and training protocol, these results indicate that supplying structural information alone is less efective than jointly incorporating structural guidance and output-side edge consistency.

Generic adaptive mechanisms provide limited gains. Visible/NIR gating gives F1=0.7651 versus 0.7647 for Mid AVG, although NRM falls to 0.1162. SE reaches $\mathrm { F 1 / p  – F M / I o U }$ of 0.7652/0.7737/0.6420, while CBAM gives the lowest neural F1 (0.7502). Generic gating or attention therefore does not reproduce the gains from explicit cross-spectral structural guidance.

Among the three visible–NIR structural variants, the F3+NIR configuration yields the highest $F _ { 1 } { \mathrm { . } }$ , p-FM, and IoU scores, closely followed by F2 + NIR. Because all variants share the same network backbone, fusion depth, loss weighting $( \lambda _ { \mathrm { e d g e } } = 5 )$ , optimization protocol, and full multispectral input, these variations stem from the spectral pair used to construct the auxiliary structural map. This highlights the impact of targeted spectral guidance examined in Section 5.2.

Overall, the gains arise from adaptive intermediate fusion coupled with an explicit visible–NIR structural representation, not merely learnable weights or generic attention. NRM and DRD differences also show that F3 + NIR is strongest for this dataset rather than universally optimal.

## 5.2 Efect of Targeted Visible–NIR Structural Guidance

The overall comparison in Table 2 indicates that the source used to construct the structural representation has a measurable efect on binarization performance. To examine this behavior more directly, Table 3 compares the four structurally guided configurations trained with $\lambda _ { \mathrm { e d g e } } ~ = ~ 5 ,$ together with the four-band model trained without the edge-consistency loss.

Among the three targeted visible–NIR configurations, the $F 3 + F 7$ structural source provides the highest F1, p-FM, and IoU. Its mean F1 of 0.7916 exceeds the corresponding values of 0.7842 and 0.7746 obtained with $F 2 + F 7$ and $F 4 + F 7 .$ respectively. Similar behavior is observed for p-FM, where the $F 3 + F 7$ model achieves 0.8044 compared with 0.7935 for $F 2 + F 7$ and 0.7866 for $F 4 + F 7$ . The same ordering is retained for IoU, with values of 0.6654, 0.6568, and 0.6495, respectively.

Because the three models use the same encoder depth, gating mechanism, training objective, full multispectral input, and five-fold protocol, this comparison isolates the wavelength pair used to construct the auxiliary structural representation.

Table 3 Efect of structural-source selection and edge-aware supervision. All models retain the complete F2–F3–F4–F7 multispectral input, where $F 2 , F 3$ , and F4 denote the 500 nm, 600 nm, and 700 nm visible bands, respectively, and F7 denotes the 1000 nm near-infrared (NIR) band. Only the source of the auxiliary structural representation and the use of the edge-consistency loss are varied.
<table><tr><td>Structural configuration</td><td>F1 ↑</td><td> $\mathbf { p } \mathbf { - F M \uparrow }$ </td><td> $\mathbf { I o U \Sigma } \mathrm { \uparrow }$ </td><td>NRM↓</td><td> $ { \mathbf { D R D } } \downarrow$ </td></tr><tr><td>Four-band,  $\lambda _ { \mathrm { e d g e } } = 0$ </td><td>0.7616</td><td>0.7694</td><td>0.6330</td><td>0.1307</td><td>5.9912</td></tr><tr><td> $\mathrm { F o u r – b a n d } , \lambda _ { \mathrm { e d g e } } = 5$ </td><td>0.7853</td><td>0.8021</td><td>0.6600</td><td>0.1227</td><td>5.2537</td></tr><tr><td> $F 2 + F 7 , \lambda _ { \mathrm { e d g e } } { = } 5$ </td><td>0.7842</td><td>0.7935</td><td>0.6568</td><td>0.1064</td><td>6.3810</td></tr><tr><td> $F 3 + F 7 , \lambda _ { \mathrm { e d g e } } = 5$ </td><td>0.7916</td><td>0.8044</td><td>0.6654</td><td>0.1115</td><td>5.7007</td></tr><tr><td> $F 4 + F 7 , \lambda _ { \mathrm { e d g e } } = 5$ </td><td>0.7746</td><td>0.7866</td><td>0.6495</td><td>0.1259</td><td>5.4313</td></tr></table>

Relative to $F 2 + F 7 ,$ , the $F 3 + F 7$ configuration increases F1 by approximately 0.0074, p-FM by 0.0109, and IoU by 0.0086. Relative to $F 4 +$ $F 7 ,$ the corresponding improvements are approximately 0.0170, 0.0178, and 0.0158. These results suggest that the 600-nm and 1000-nm observations provide a particularly efective structural source for recovering foreground stroke regions in the evaluated SMADI documents.

The $F 3 + F 7$ advantage is not uniform across distortion measures. The $F 2 + F 7$ configuration obtains the lowest NRM, 0.1064, compared with 0.1115 for $F 3 { + } F 7$ . This indicates that the $F 2 { + } F 7$ structural source provides a slightly more favorable balance between foreground misses and background false positives according to NRM. Conversely, $F 4 + F 7$ obtains a lower DRD of 5.4313 than the 5.7007 achieved by $F 3 + F 7 $ , while the four-band structural representation achieves the lowest overall DRD of 5.2537. Hence, the spectral pair producing the strongest foreground-overlap measures is not necessarily the pair minimizing every form of local structural distortion.

The four-band results provide another insight. When the four-band structural map is supplied to the network without edge-aware output supervision, the model achieves an F1 of 0.7616, p-FM of 0.7694, IoU of 0.6330, NRM of 0.1307, and DRD of 5.9912. Retaining the same structurally guided architecture while introducing $\begin{array} { r l } { \lambda _ { \mathrm { e d g e } } } & { { } = } \end{array}$ 5, increases F1 by approximately 0.0237, p-FM by 0.0327, and IoU by 0.0270. NRM decreases by approximately 0.0080 and DRD decreases by approximately 0.7375.

This contrast indicates that the auxiliary structural branch alone is not suficient to account for the performance obtained by the complete edge-guided formulation. The combination of input-side structural conditioning and outputside gradient consistency provides substantially stronger results than the structurally guided architecture trained without the edge-consistency term. Because only the fixed values $\lambda _ { \mathrm { e d g e } } ~ = ~ 0$ and $\lambda _ { \mathrm { e d g e } } = 5$ are compared in this experiment, the results should not be interpreted as establishing 5 as an optimal edge-loss weight.

The four-band and targeted-pair results further indicate that simply aggregating all spectral observations when constructing the edge representation is not always preferable. The four-band model produces the lowest DRD, but the targeted $F 3 + F 7$ representation achieves higher F1, $\mathrm { p - F M } .$ and IoU. This supports the underlying hypothesis that the structural cue used to regulate fusion can benefit from deliberate wavelength selection rather than indiscriminate combination of all available bands.

Importantly, $F 3 + F 7$ does not reduce the generator input to two bands. The full X = $[ X _ { F 2 } , X _ { F 3 } , X _ { F 4 } , X _ { F 7 } ]$ remains available; $F 3 + F 7$ only derives the auxiliary structural representation. The gains therefore reflect targeted structural guidance, not exclusion of F2 or F4.

These findings identify the $F 3 \ + \ F 7$ (600/1000 nm) pair as the most efective structural source within our experimental setting. Although this does not imply universal optimality across degradation types, it shows that domain-targeted structural guidance outperforms generic attention and edge representations formed indiscriminately from all bands.

## 5.3 Statistical Significance

Document-level gains for F3 + NIR versus Mid AVG were tested with paired two-sided Wilcoxon signed-rank tests over all 30 held-out pages, with Holm adjustment across candidate architectures (Section 4.6). Table 4 reports statistics, win ratios, efect sizes $\left( r _ { \mathrm { r b } } \right)$ , and adjusted significance.

F3 + NIR significantly outperforms Mid AVG on all three primary overlap metrics after correction: $F _ { 1 }$ , p-FM, and IoU have $p _ { \mathrm { H o l m } } ~ < ~ 0 . 0 3$ rank-biserial efects $r _ { \mathrm { r b } } ~ \geq ~ 0 . 5 9 6$ , and win ratios $\ge ~ 2 3 / 3 0$ . Bootstrap 95% confidence intervals remain positive for $F _ { 1 }$ [0.0014, 0.0513], p-FM [0.0080, 0.0628], and IoU [0.0009, 0.0598].

F3+NIR also reduces mean NRM from 0.1285 to 0.1115 and DRD from 5.9281 to 5.7007, but neither remains significant after Holm adjustment $( p _ { \mathrm { H o l m } } = 0 . 1 5 4 6 9$ and 0.29148). The DRD bootstrap interval $\left[ - 1 . 0 4 2 0 , 1 . 3 1 5 2 \right]$ spans zero, so these reductions remain descriptive. Among other variants, only the four-band edge-guided model $( \lambda _ { \mathrm { e d g e } } = 5 )$ yields another Holm-significant gain over Mid $\mathrm { A V G } \ : \ : \mathrm { ( p . F M : } \ : \ : W \ : = \ : 9 8 , \ : r _ { \mathrm { r b } } \ : = \ : 0 . 5 7 8 ,$ $p _ { \mathrm { H o l m } } = 0 . 0 3 2 6 5 )$

In summary, non-parametric testing confirms that the advantages of targeted $F 3 \ +$ NIR structural guidance are consistent across pages and robust to multiple-comparison bias for spatial-overlap metrics, though non-significant for distance-based error criteria.

## 5.4 Comparison with Classical Binarization

The multispectral framework is also evaluated against standard non-learning binarization algorithms: Otsu [13] and Sauvola [14]. Both baselines operate on an 8-bit min–max rescaled visiblemean image:

$$
I _ { \mathrm { v i s } } = { \frac { 1 } { 3 } } \left( X _ { F 2 } + X _ { F 3 } + X _ { F 4 } \right) .\tag{39}
$$

Otsu thresholding is applied globally without tuning. For Sauvola thresholding $( R \ = \ 1 2 8 )$ , foldspecific window sizes $w ~ \in ~ \{ 1 5 , 2 5 , 3 5 , 5 1 \}$ and sensitivity parameters $k \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 \}$ are selected via validation pseudo- $F _ { 1 }$ optimization (with smaller $w$ , then smaller k resolving ties). Table 5 lists the fold-specific settings.

Table 6 reports mean performance over 30 held-out test pages. Global Otsu thresholding performs poorly $( F _ { 1 } ~ = ~ 0 . 5 4 0 0 )$ , especially when foreground strokes and background degradation overlap in the intensity histogram. Sauvola thresholding provides a stronger classical baseline $( F _ { 1 } =$

0.7436), confirming the value of spatially adaptive thresholding under non-uniform backgrounds.

The deep models nevertheless outperform the classical baselines. Fixed intermediate fusion (Mid AVG) improves upon Sauvola by +0.0211 $F _ { 1 } .$ , showing that learned multispectral integration extracts information beyond local intensity thresholding. The proposed F3+NIR edge-guided model achieves the highest performance $( F _ { 1 } ~ =$ $0 . 7 9 1 6 , \ \mathrm { I o U \ = \ 0 . 6 6 5 4 }$ $\mathrm { { N R M } \ = \ 0 . 1 1 1 5 ) }$ , with absolute gains of +0.0480 $F _ { 1 }$ and $+ 0 . 0 5 6 6$ IoU over Sauvola. Although Otsu achieves a seemingly competitive NRM (0.1175) due to metric balancing between false-negative and false-positive rates, its severe underperformance in $F _ { 1 }$ and IoU confirms that comprehensive multi-metric evaluation is essential.

## 5.5 Reconstruction-Overlap Sensitivity

A post hoc analysis evaluated the frozen F3+NIR model at 0% (s = 256), 10% $( s ~ = ~ 2 3 0 )$ , 20% $( s = 2 0 4 )$ , and 30% (s = 179) overlap. For each stride, validation thresholding was re-optimized before locked test evaluation without retraining. Re-evaluating the 20% baseline reproduced the original test binaries exactly. Table 7 summarizes all 30 held-out pages.

The largest drop occurs at $0 \%$ overlap $( F _ { 1 }$ = 0.7617, DRD=6.1510), indicating severe boundary artifacts. A 10% overlap restores $F _ { 1 }$ to 0.7885, confirming the value of soft averaging near patch borders. Gains then diminish through 20% and 30%, with 30% producing the best metrics $( F _ { 1 } =$ 0.7957, IoU=0.6717, DRD=5.5863).

Holm-corrected paired Wilcoxon tests against the 20% baseline show that 30% overlap remains significantly better for $F _ { 1 }$ , IoU, NRM, and DRD $( p _ { \mathrm { H o l m } } \ < \ 0 . 0 5 )$ . For a strict controlled comparison across architectures, the 20% setting is retained for primary comparisons, balancing eficiency against incremental gains.

## 5.6 Computational Complexity and Inference Eficiency

The selected F3+NIR model was profiled without further optimization or compression; operation counts refer to one $2 5 6 \times 2 5 6$ patch and runtime to an NVIDIA Tesla T4 GPU.

Table 4 Paired statistical comparison of the selected $F 3 + \mathrm { N I R }$ edge-guided model against the Mid AVG baseline across 30 held-out test documents. Diferences (∆) and rank-biserial correlations $( r _ { \mathrm { r b } } )$ are oriented such that positive values indicate improvement.
<table><tr><td>Metric</td><td>Mid AVG</td><td>F3+NIR</td><td>Δ</td><td>Wins</td><td>W</td><td> $r _ { \mathrm { r b } }$ </td><td>p</td><td>PHolm</td></tr><tr><td>F1</td><td>0.7647</td><td>0.7916</td><td>+0.0269</td><td>24/30</td><td>89</td><td>0.617</td><td>0.00237</td><td>0.01894</td></tr><tr><td>p-FM</td><td>0.7684</td><td>0.8044</td><td>+0.0360</td><td>23/30</td><td>94</td><td>0.596</td><td>0.00348</td><td>0.02780</td></tr><tr><td>IoU</td><td>0.6333</td><td>0.6654</td><td>+0.0321</td><td>24/30</td><td>89</td><td>0.617</td><td>0.00237</td><td>0.01894</td></tr><tr><td>NRM</td><td>0.1285</td><td>0.1115</td><td>+0.0170</td><td>21/30</td><td>122</td><td>0.475</td><td>0.02210</td><td>0.15469</td></tr><tr><td>DRD</td><td>5.9281</td><td>5.7007</td><td>+0.2274</td><td>20/30</td><td>132</td><td>0.432</td><td>0.03842</td><td>0.29148</td></tr></table>

Table 5 Fold-specific Sauvola parameters optimized strictly on validation partitions.
<table><tr><td>Fold</td><td>Window size (w)</td><td>k</td></tr><tr><td>1</td><td>15</td><td>0.2</td></tr><tr><td>2</td><td>25</td><td>0.2</td></tr><tr><td>3</td><td>15</td><td>0.1</td></tr><tr><td>4</td><td>15</td><td>0.2</td></tr><tr><td>5</td><td>15</td><td>0.2</td></tr></table>

Table 6 Performance comparison with classical binarization methods across 30 held-out test document pages.
<table><tr><td>Method</td><td>F1 ↑</td><td>IoU ↑</td><td>NRM↓</td></tr><tr><td>Otsu</td><td>0.5400</td><td>0.4076</td><td>0.1175</td></tr><tr><td>Sauvola</td><td>0.7436</td><td>0.6087</td><td>0.1178</td></tr><tr><td>Mid AVG  $\left( L = 3 \right)$  F3+NIR Edge Gate</td><td>0.7647</td><td>0.6333</td><td>0.1285</td></tr><tr><td>(Proposed)</td><td>0.7916</td><td>0.6654</td><td>0.1115</td></tr></table>

Table 7 Reconstruction-overlap sensitivity of the selected F3+NIR edge-guided model. No model retraining is performed.
<table><tr><td>Overlap F1 ↑</td><td>p-FM ↑</td><td>IoU ↑</td><td>NRM↓DRD↓</td></tr><tr><td>0%</td><td>0.7617</td><td>0.7783 0.6275</td><td>0.1382 6.1510</td></tr><tr><td>10%</td><td>0.7885</td><td>0.7995 0.6617</td><td>0.1141 5.7555</td></tr><tr><td>20%</td><td>0.7916</td><td>0.8044 0.6654</td><td>0.1115 5.7007</td></tr><tr><td>30%</td><td>0.7957</td><td>0.8053 0.6717</td><td>0.1078 5.5863</td></tr></table>

The generator contains 56.31 million trainable parameters, requiring approximately 214.82 MiB for FP32 parameters; the saved checkpoint occupies 214.90 MiB. The footprint primarily reflects the three modality-specific encoder branches before L = 3 fusion and the subsequent U-Net encoder–decoder.

Table 8 Computational characteristics of the selected F3+NIR edge-guided generator. Runtime measurements were obtained on an NVIDIA Tesla T4 GPU.
<table><tr><td>Measure</td><td>Value</td></tr><tr><td>Trainable parameters</td><td>56.31 M</td></tr><tr><td>FP32 parameter memory</td><td>214.82 MiB</td></tr><tr><td>Checkpoint size</td><td>214.90 MiB</td></tr><tr><td>Patch complexity (MACs)</td><td>20.68 GMACs</td></tr><tr><td>Patch complexity (FLOPs)</td><td>41.35 GFLOPs</td></tr><tr><td>Mean patch latency</td><td>8.73 ms</td></tr><tr><td>Median patch latency</td><td>8.68 ms</td></tr><tr><td>95th-percentile patch latency</td><td>8.87 ms</td></tr><tr><td>Patch throughput</td><td>114.51 patches/s</td></tr><tr><td>Peak GPU memory allocated</td><td>460.58 MiB</td></tr><tr><td>Peak GPU memory reserved</td><td>480.00 MiB</td></tr></table>

A 256 × 256 patch requires approximately 20.68 GMACs, corresponding to 41.35 GFLOPs under the two-FLOPs-per-MAC convention. These supported-operation profiling estimates are approximate complexity indicators rather than hardware-independent runtime predictions.

Table 8 summarizes the principal complexity and inference measurements.

Single-patch inference is stable under the tested GPU conditions, yielding a mean latency of $8 . 7 3 \pm 0 . 5 7$ ms (median: 8.68 ms; 95<sup>th</sup> percentile: 8.87 ms) with average throughput of approximately 114.5 patches/s. Peak allocated GPU memory during execution is 460.58 MiB (480 MiB reserved), accounting for runtime tensors and network parameters.

For document z92, nine 256 × 256 patches at 20% overlap required 0.0496±0.0056 s model-only and 0.1056 ± 0.0055 s end-to-end. Overhead varies with page dimensions.

Although 30% overlap yields peak binarization performance (Section 5.5), the 20% setting provides a favorable accuracy–cost trade-of. While practical for GPU deployment, the 56.31-millionparameter generator footprint motivates future work into model compression, knowledge distillation, and parameter-eficient fusion mechanisms.

## 5.7 Qualitative Analysis

Figures 1 and 2 compare nine architectures on held-out documents z27, z35, and z80, covering bleed-through, illumination drop-ofs, and faint strokes to assess foreground preservation, connectivity, and background suppression.

Mid AVG (L = 3) recovers primary text but shows stroke fragmentation and boundary degradation because unweighted averaging treats textbearing and noise-dominated channels equally. CBAM, SE, and Visible/NIR gating reduce background noise but introduce non-uniform artifacts.

The two four-band variants clarify the structural loss function’s role. An edge-derived channel without auxiliary supervision $( \lambda _ { \mathrm { { e d g e } } } ~ = ~ 0 )$ does not constrain feature representation space. Incorporating $\lambda _ { \mathrm { e d g e } } = 5$ improves stroke continuity and suppresses isolated false positives, matching the quantitative gains observed in $\mathrm { F _ { 1 } }$ and DRD.

Wavelength-specific guidance shows clear structural advantages. Among $F 2 + \mathrm { N I R }$ , F3 + NIR, and F4 + NIR, F3 + NIR gives stronger stroke reconstruction without boundary erosion or false-positive expansion. Deriving guidance from a targeted visible–NIR pair while retaining the full multispectral input preserves boundaries in severe degradation zones, supporting its $F _ { 1 }$ , p-FM, and IoU leads.

Residual errors persist where local background contrast approaches zero, aligning with metricspecific trades (e.g., F2 + NIR obtaining minimal NRM). Visual evidence nevertheless confirms that targeted multispectral structural guidance combined with adaptive intermediate fusion provides robust full-page binarization.

## 6 Limitations and Future Work

Several scope and computational limitations warrant further investigation.

Spectral Pair Selection and Dataset Scope: Evaluation uses 30 SMADI documents across four bands (500, 600, 700, and 1000 nm). Although five-fold document-level cross-validation ensures strict isolation, the dataset limits media, ink, and degradation diversity. Thus, $F 3 + \mathrm { N I R }$ (600/1000 nm) is not universally optimal; broader corpora and data-driven wavelength selection remain future work.

Loss Formulation and Structural Guidance Filters: The edge-aware loss weight $( \lambda _ { \mathrm { e d g e } } = 5 )$ and Sobel edge extraction were evaluated under controlled settings rather than exhaustive search. Sobel filters also respond to highgradient bleed-through, paper grain, and stains, occasionally preserving non-text artifacts. Future work could use learned text-aware descriptors, multiscale edges, or confidence-weighted losses to balance recovery and noise suppression.

Computational Complexity and Full-Page Inference: The generator contains 56.31 × $1 0 ^ { 6 }$ parameters and requires approximately 20.68 GMACs per 256×256 patch. GPU latency is feasible for ofline processing, but 20% reconstruction overlap increases page-level computation. Deployment-oriented work should explore branch sharing, distillation, lightweight fusion, and fully convolutional page inference.

Future work will therefore emphasize crossdataset generalization, adaptive spatial-spectral guidance, and architectural compression.

## 7 Conclusion

This study established a targeted multispectral intermediate-fusion framework for historicaldocument binarization, showing that physically guided regulation improves on simple averaging and generic attention. At feature depth $L  = 3$ , learned edge-guided fusion replaces uniform channel weighting. A dedicated F3 + NIR (600/1000 nm) pair supplies structural guidance while the full F2, F3, F4, F7 observation remains available, preserving character boundaries while suppressing complex degradation. Across 30 SMADI pages under five-fold document-level cross-validation, F3 + NIR achieved statistically significant gains in F<sub>1</sub>, p-FM, and IoU over fixed averaging and generic CBAM/SE attention. Patch overlap and runtime analyses confirmed that a 20% patch reconstruction overlap provides a practical deployment compromise for full-page GPU inference. The results confirm that physical wavelength selection governs distinct error profiles. While targeted F3 + NIR guidance delivers optimal overall foreground recovery and structural continuity, alternative pairings favor NRM or DRD. Targeted structural guidance therefore efectively regulates intermediate multispectral features and motivates future research into learned dynamic spectral selection and parameter-eficient document-scale architectures.

<table><tr><td rowspan="2">Mid AVG L = 3</td><td colspan="2">Lni chguhies aite s oil  dohlamais i</td><td>l.53上 也上 u ombriy 2</td><td></td></tr><tr><td rowspan="3"></td><td>manfedrleuell promest garserlesese ansuvent ovsre fatalabair stantor</td><td>261n91</td><td></td></tr><tr><td></td><td>q toailaffees diteff dohlama</td><td>13 Le  ombr </td><td></td></tr><tr><td></td><td>manfeau leqnelle prowest garser lesuss</td><td></td><td></td></tr><tr><td rowspan="3">Visible/NIR Gated</td><td></td><td>ansunerk ovsre, farteulabcip stanto useLi ni ant choique guiest シッン</td><td>1.53 xy 2 me_omb.riy 06</td><td></td></tr><tr><td>9</td><td>ls iteff  Tohlamais</td><td>0.g.x</td><td></td></tr><tr><td>Edge4Band, λedge = 0 anouuert osre fatalabeir stanTar</td><td>manfeaa lequelle promist garser lessass</td><td>IVad Se 040102</td><td>30</td></tr><tr><td rowspan="3">er, Li. ui ant ckotiquehguiest</td><td></td><td>srLi wantchoguehguiest 1</td><td>6.53 5v. 82上 ru  o mmy 1221 14.91</td><td></td></tr><tr><td>que</td><td>zilaiff itdolamai</td><td></td><td></td></tr><tr><td></td><td>manfealequele romest garserlesfus Edge4Band, λedge = 5 ansune ousre fautalalipe stanTar</td><td></td><td>l.sz Erd nvy uy omb ry </td></tr></table>

Fig. 1 Qualitative comparison of baseline, gated, and channel-attention architectures on held-out documents (z27, z35, and z80), using validation-selected thresholds.

## Statements and Declarations

Funding. This study is independent research carried out under the University of Peradeniya Multidisciplinary Research Grant Scheme (MRG 2025 06).

Competing interests. The authors declare no competing interests.

Author contributions. Yasas Savinda: Conceptualization, methodology, software, validation, formal analysis, investigation, visualization, and writing–original draft. Maheshi Buddhinee Dissanayake: Conceptualization, supervision, methodology, funding and grant management, formal analysis, and writing–review and editing. Both authors read and approved the final manuscript.

Data availability. The experiments use the SMADI multispectral historical-document dataset associated with the ICDAR 2015 MultiSpectral Text Extraction benchmark [6]. The original SMADI dataset and benchmark information are publicly available for non-commercial research from the oficial IAPR TC-11 SMADI repository. Information on the dataset organization and file structure required to reproduce the experiments is available in the supplementary dataset information folder. The experimental outputs and analysis results generated in this study are available in the supplementary results repository.

<table><tr><td rowspan="2">SE Fusion</td><td rowspan="2">win iqe riest que tor ailaiffe les diteff dohlamais manfedleyuele oomest garslessas ansuver ovsre fate alabair stantr usrunuintchoiquequiest zi loi s diteTo lawai</td><td rowspan="2">ど上 1.52k 2</td><td rowspan="2"></td></tr><tr><td></td></tr><tr><td>F2+NIR Edge-Guided</td><td>manfeau lequele rowest gaserlesfuss ansuvel ovsre fatalabairstantor uswPi iaokoqne guiest ailaiff es diteff dohlainas que合</td><td>rry</td><td></td></tr><tr><td>F3+NIR Edge-Guided (Proposed)</td><td>manfeaslequell promist gaselessas ansuver ovsre fatealabeir stantar ue Lin wi an  ckotque guiest ailie lsditeffe dohlamai</td><td>1.53K2 yombo.miyn 71.2</td><td></td></tr><tr><td>F4+NIR Edge-Guided</td><td>manfedv lequele promest garser lessass ansunel ovsre fateulabairstantar niw Rien uue au tre ckopequefequiest ques ton ai loiele diteff donlamais</td><td>1.53  2 omb (2b u 02</td><td></td></tr><tr><td>Ground Truth</td><td>manfeavrlequele promest gaser eshus ansuvek orsrefatoalabeipe stantor</td><td></td><td>12 基</td></tr></table>

Fig. 2 Qualitative comparison of channel-attention and wavelength-specific edge-guided models (F2–F4 + NIR). The proposed F3+NIR model is highlighted in bold.

Code availability. The training, evaluation, and analysis code supporting the findings of this study is available in the supplementary code repository.

Ethics approval. Not applicable.

Consent to participate. Not applicable.

Consent for publication. Not applicable.

## References

[1] Souibgui MA, Kessentini Y (2022) De-gan: A conditional generative adversarial network for document enhancement. IEEE Trans Pattern Anal Mach Intell 44(3):1180–1191. https: //doi.org/10.1109/TPAMI.2020.3022406

[2] Yang Z, Liu B, Xiong Y, et al (2024) Gdb: Gated convolutions-based document binarization. Pattern Recognit 146:109989. https: //doi.org/10.1016/j.patcog.2023.109989

[3] Kim SJ, Deng F, Brown MS (2011) Visual enhancement of old documents with hyperspectral imaging. Pattern Recognit

44(7):1461–1469. https://doi.org/10.1016/j. patcog.2010.12.019

[4] Hedjam R, Cheriet M (2013) Historical document image restoration using multispectral imaging system. Pattern Recognit 46(8):2297–2312. https://doi.org/10.1016/j. patcog.2012.12.015

[5] Savinda Y, Dissanayake MB (2026) Deep learning-based fusion for multispectral text enhancement using supervised conditional gan. Presented at the 12th Moratuwa Engineering Research Conference (MERCon 2026)

[6] Hedjam R, Ziaei Nafchi H, Farrahi Moghaddam R, et al (2015) Icdar 2015 contest on multispectral text extraction (ms-tex 2015). In: 2015 13th International Conference on Document Analysis and Recognition (ICDAR), pp 1181–1185, https://doi.org/10. 1109/ICDAR.2015.7333947

[7] Diem M, Hollaus F, Sablatnig R (2016) Msio: Multispectral document image binarization. In: 2016 12th IAPR Workshop on Document Analysis Systems (DAS), pp 84–89, https:// doi.org/10.1109/DAS.2016.39

[8] Hollaus F, Brenner S, Sablatnig R (2019) Cnn based binarization of multispectral document images. In: 2019 International Conference on Document Analysis and Recognition (ICDAR), pp 533–538, https://doi.org/10. 1109/ICDAR.2019.00091

[9] Esmaeili Salehani Y, Arabnejad E, Rahiche A, et al (2020) Msdb-nmf: Multispectral document image binarization framework via nonnegative matrix factorization approach. IEEE Trans Image Process 29:9099–9112. https:// doi.org/10.1109/TIP.2020.3023613

[10] Baltruˇsaitis T, Ahuja C, Morency LP (2019) Multimodal machine learning: A survey and taxonomy. IEEE Trans Pattern Anal Mach Intell 41(2):423–443. https://doi.org/ 10.1109/TPAMI.2018.2798607

[11] Ma J, Liang P, Yu W, et al (2020) Infrared and visible image fusion via detail preserving adversarial learning. Inf Fusion 54:85–98. https://doi.org/10.1016/j.infus.2019.07.005

[12] Kurar-Barakat B, Dershowitz N (2026) Segmentation of ink and parchment in dead sea scroll fragments. Int J Doc Anal Recognit 29:531–549. https://doi.org/ 10.1007/s10032-025-00564-4

[13] Otsu N (1979) A threshold selection method from gray-level histograms. IEEE Trans Syst Man Cybern 9(1):62–66. https://doi.org/10. 1109/TSMC.1979.4310076

[14] Sauvola J, Pietik¨ainen M (2000) Adaptive document image binarization. Pattern Recognit 33(2):225–236. https://doi.org/10.1016/ S0031-3203(99)00055-2

[15] Vo QN, Kim SH, Yang HJ, et al (2018) Binarization of degraded document images based on hierarchical deep supervised network. Pattern Recognit 74:568–586. https://doi.org/ 10.1016/j.patcog.2017.08.025

[16] Calvo-Zaragoza J, Gallego AJ (2019) A selectional auto-encoder approach for document image binarization. Pattern Recognit 86:37–47. https://doi.org/10.1016/j.patcog. 2018.08.011

[17] He S, Schomaker L (2019) Deepotsu: Document enhancement and binarization using iterative deep learning. Pattern Recognit 91:379–390. https://doi.org/10.1016/j. patcog.2019.01.025

[18] Zhao J, Shi C, Jia F, et al (2019) Document image binarization with cascaded generators of conditional generative adversarial networks. Pattern Recognit 96:106968. https: //doi.org/10.1016/j.patcog.2019.106968

[19] Khamekhem Jemni S, Souibgui MA, Kessentini Y, et al (2022) Enhance to read better: A multi-task adversarial network for handwritten document image enhancement. Pattern Recognit 123:108370. https://doi.org/ 10.1016/j.patcog.2021.108370

[20] Liu M, Yu L, Qin Y, et al (2026) Ragan: Region-adaptive gan for binarization on degraded document images. Knowl-Based Syst 342:115860. https://doi.org/10.1016/j. knosys.2026.115860

[21] Qureshi R, Uzair M, Khurshid K, et al (2019) Hyperspectral document image processing: Applications, challenges and future prospects. Pattern Recognit 90:12–22. https: //doi.org/10.1016/j.patcog.2019.01.026

[22] Farrahi Moghaddam R, Cheriet M (2015) A multiple-expert binarization framework for multispectral images. In: 2015 13th International Conference on Document Analysis and Recognition (ICDAR), pp 321–325, https:// doi.org/10.1109/ICDAR.2015.7333776

[23] Hollaus F, Diem M, Sablatnig R (2018) Multispectral image binarization using gmms. In: 2018 16th International Conference on Frontiers in Handwriting Recognition (ICFHR), pp 570–575, https://doi.org/10. 1109/ICFHR-2018.2018.00105

[24] Rahiche A, Bakhta A, Cheriet M (2019) Blind source separation based framework for multispectral document images binarization. In: 2019 International Conference on Document Analysis and Recognition (ICDAR), pp 1476– 1481, https://doi.org/10.1109/ICDAR.2019. 00237

[25] Hazirbas C, Ma L, Domokos C, et al (2017) Fusenet: Incorporating depth into semantic segmentation via fusion-based cnn architecture. In: Computer Vision – ACCV 2016, Lecture Notes in Computer Science, vol 10111. Springer, pp 213–228

[26] Arevalo Ovalle JE, Solorio T, Montes-y G´omez M, et al (2017) Gated multimodal units for information fusion. In: 5th International Conference on Learning Representations (ICLR 2017), Workshop Track Proceedings, URL https://mlanthology.org/iclr/ 2017/ovalle2017iclr-gated/

[27] Hu J, Shen L, Sun G (2018) Squeeze-andexcitation networks. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern

Recognition, pp 7132–7141

[28] Woo S, Park J, Lee JY, et al (2018) Cbam: Convolutional block attention module. In: European Conference on Computer Vision (ECCV), pp 3–19

[29] Li H, Wu XJ (2019) Densefuse: A fusion approach to infrared and visible images. IEEE Trans Image Process 28(5):2614–2623. https://doi.org/10.1109/TIP.2018.2887342

[30] Ma J, Yu W, Liang P, et al (2019) Fusiongan: A generative adversarial network for infrared and visible image fusion. Inf Fusion 48:11–26. https://doi.org/10.1016/j.infus.2018.09.004

[31] Isola P, Zhu JY, Zhou T, et al (2017) Imageto-image translation with conditional adversarial networks. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp 1125–1134, https://doi.org/10. 1109/CVPR.2017.632

[32] Dissanayake MB (2026) A pix2pix gan-based denoising system for historical stone writing restoration. In: 2025 IEEE 19th International Conference on Industrial and Information Systems (ICIIS), pp 431–435, https://doi. org/10.1109/ICIIS69028.2026.11450630

[33] Ntirogiannis K, Gatos B, Pratikakis I (2013) Performance evaluation methodology for historical document image binarization. IEEE Trans Image Process 22(2):595–609. https: //doi.org/10.1109/TIP.2012.2219550

[34] Lu H, Kot AC, Shi YQ (2004) Distancereciprocal distortion measure for binary document images. IEEE Signal Process Lett 11(2):228–231. https://doi.org/10.1109/LSP. 2003.821748

[35] Demˇsar J (2006) Statistical comparisons of classifiers over multiple data sets. J Mach Learn Res 7:1–30. URL https://www.jmlr. org/papers/v7/demsar06a.html

[36] Holm S (1979) A simple sequentially rejective multiple test procedure. Scand J Stat 6(2):65–70