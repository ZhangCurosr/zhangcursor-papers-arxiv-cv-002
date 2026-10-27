# Natural Jammers: When Non-Robust Features Become Antagonistic

Jiaming Zhang<sup>1</sup> Fuyao Zhang<sup>1</sup> Pengjun Xie<sup>2</sup> Wei Yang Bryan Lim<sup>1</sup>

<sup>1</sup>Nanyang Technological University <sup>2</sup>Alibaba Group

Project page: https://jiamingzhang.netlify.app/naturaljammers

## Abstract

Deep neural networks are known to rely on non-robust features that align poorly with human perception. Prior work has largely framed these features as opportunistic shortcuts that help standard models on clean data. We show instead that non-robust cues can be antagonistic: they can act as Natural Jammers that actively mislead standard models on clean, unmodified images. Using disagreement between a standard model and robust models as an analysis lens, we identify a Jammed Set on which robust models outperform standard models. We provide causal evidence for Spectral–spatial Locking: jamming arises only when high-frequency components are precisely aligned with specific spatial structures. Disrupting this lock through spectral filtering or small geometric shifts can restore correct predictions. Crucially, we uncover a Substitution Paradox: while individual jammers are fragile to geometric shifts, the aggregate prevalence of jamming remains approximately stable and is redistributed rather than eliminated under such transformations. Our results reframe natural non-robustness as a failure field over the empirical model–data geometry: one that redistributes rather than disappears under small transformations.

## 1 Introduction

Deep neural networks (DNNs) achieve remarkable success on visual tasks, yet their reasoning differs fundamentally from human perception. Rather than relying solely on semantic content, DNNs are highly sensitive to non-robust features—subtle, high-frequency patterns that predict labels but remain imperceptible to humans Ilyas et al. [2019], Gyevnar and Kasirzadeh [2025], Liu et al. [2025a]. The prevailing view treats these features as opportunistic shortcuts: brittle patterns that standard models exploit as a crutch to reach correct decisions Geirhos et al. [2020], Kauffmann et al. [2025], Eriksson et al. [2025]. This framing assumes non-robust signals and semantic labels work synergistically.

We challenge this synergistic view by uncovering natural feature antagonism. As Figure 1 illustrates, non-robust signals do not always aid classification—for a substantial fraction of natural images, they actively mislead standard inference. We term these signals Natural Jammers. This creates an empirical paradox: although the robustness-accuracy trade-off suggests standard models (f<sub>std</sub>) should upper-bound clean performance Tsipras et al. [2018], Suzuki et al. [2023], Zhang et al. [2024], Kafali et al. [2025], Li and Li [2025], we identify a distinct Jammed Set (J) where robustly trained models $( f _ { \mathrm { r o b } } )$ significantly outperform their standard counterparts on unmodified images. The standard model’s reliance on non-robust features becomes a liability—it is misled by brittle cues that contradict the label, while the robust model often succeeds because it is less sensitive to such cues. We employ $f _ { \mathrm { r o b } }$ as a diagnostic lens to disentangle robust semantic cues from non-robust correlations.

To explain this interference, we propose the Spectral-Spatial Locking mechanism through controlled experiments. Natural jammers arise from high-frequency patterns, but their disruptive influence depends critically on alignment with specific spatial configurations. This phase-locked coupling explains a striking duality: while these features mislead standard inference, they are also fragile. Breaking the lock—through spectral filtering (the Cleansing Paradox, Section 3.1) or minor geometric shifts (the Sensitivity Inversion, Section 3.2)—renders the jammer inert, allowing the model’s discriminative capacity to resurface.

![](images/6646614334169723075fe1731b02bec4bee15fb320b473580214dbf1b4e7e539.jpg)  
Figure 1: The Duality of Non-Robust Features. (Left) Spurious Shortcut (S): prediction relies on fragile non-robust cues and collapses when these cues are suppressed. (Right) Natural Jammer (J): non-robust cues interfere with semantics-aligned evidence, suppressing them partially restores correct predictions. This illustrates that non-robust cues can be antagonistic rather than merely supportive. Accuracies are measured with the primary lens (standard vs. robust ResNet-50, ℓ<sub>2</sub>, ϵ = 1).

This fragility raises a puzzle: if individual jammers are so easily disrupted, why does the model not achieve robust generalization? Our analysis reveals the Substitution Paradox: a 2% edge crop removes about one third of the images in the Jammed Set, yet its total size remains nearly unchanged. Natural non-robustness is not a defect of particular images but a persistent property of the model–data geometry. When one jammer’s spatial-spectral lock breaks, new jammers emerge elsewhere, keeping the overall vulnerability level nearly constant. Non-robustness is redistributed rather than eliminated by simple augmentations.

Our contributions are:

• Feature Antagonism: We formalize the Jammed Set (J), showing that a significant portion of standard model errors stem from non-robust signals that contradict semantic labels.

• Spectral-Spatial Locking: We provide causal evidence that non-robust features require precise alignment between high-frequency content and spatial structure to remain effective.

• Substitution Paradox: We demonstrate that jamming is a stable, systemic property—individual non-robust features are fragile and interchangeable, following a redistribution pattern under geometric transformations.

## 2 Landscape of Natural Non-Robustness

To move beyond a binary view of robust vs. non-robust features, we formalize a taxonomy based on the latent structure of the data distribution D.

Table 1: Distribution of Feature Pathologies on ImageNet. Data is reported for $\epsilon = 1 . 0$ constraints. The proportions of J and S remain relatively stable across ResNet and WideResNet architectures, suggesting that jamming is not confined to a single architecture.
<table><tr><td>MODEL</td><td>J (JAMMED)</td><td>S (SPURIOUS)</td><td>C (COMMON)</td><td>G (UNINFORMATIVE)</td></tr><tr><td rowspan="2">RESNET-18 (l2)</td><td>2,448</td><td>6,168</td><td>28,712</td><td>12,672</td></tr><tr><td>(4.90%)</td><td>(12.34%)</td><td>(57.42%)</td><td>(25.34%)</td></tr><tr><td rowspan="2">RESNET-50 (l2)</td><td>2,140</td><td>5,002</td><td>33,071</td><td>9,787</td></tr><tr><td>(4.28%)</td><td>(10.00%)</td><td>(66.14%)</td><td>(19.57%)</td></tr><tr><td rowspan="2">WRN-50-2 (l2)</td><td>2,167</td><td>4,698</td><td>34,543</td><td>8,592</td></tr><tr><td>(4.33%)</td><td>(9.40%)</td><td>(69.09%)</td><td>(17.18%)</td></tr><tr><td rowspan="2">RESNET-18 (l∞)</td><td>2,502</td><td>5,659</td><td>29,221</td><td>12,618</td></tr><tr><td>(5.00%)</td><td>(11.32%)</td><td>(58.44%)</td><td>(25.24%)</td></tr><tr><td rowspan="2"> $\mathsf { R E S N E T } { - } 5 0 \ ( \ell _ { \infty } )$ </td><td>2,252</td><td>4,306</td><td>33,767</td><td>9,675</td></tr><tr><td>(4.50%)</td><td>(8.61%)</td><td>(67.53%)</td><td>(19.35%)</td></tr><tr><td rowspan="2"> $\mathrm { W R N - } 5 0 { \mathrm { - } } 2 \ ( \ell _ { \infty } )$ </td><td>2,232</td><td>4,145</td><td>35,096</td><td>8,527</td></tr><tr><td>(4.46%)</td><td>(8.29%)</td><td>(70.19%)</td><td>(17.05%)</td></tr></table>

Assumption 2.1 (Manifold Decomposition). Following the latent manifold hypothesis Ilyas et al. [2019], we assume the input space $\chi$ maps to a latent representation Z that can be decomposed into two subspaces, $\mathcal { Z } = \bar { \mathcal { Z } } _ { r } \times \mathsf { \bar { Z } } _ { n r }$ , where ${ \mathcal { Z } } _ { r }$ and $\mathcal { Z } _ { n r }$ denote the robust and non-robust manifolds, respectively.

The interaction between these manifolds determines whether a model’s reliance on non-robust signals yields an opportunistic shortcut or a predictive failure. We utilize a standard classifier $f _ { \mathrm { s t d } }$ and an adversarially trained counterpart $f _ { \mathrm { r o b } }$ to analyze these divergent behaviors.

## 2.1 Formal Taxonomy: The Quad-Set Partition

Given a sample $( x , y ) \in \mathcal { D }$ and a chosen reference pair $( f _ { \mathrm { s t d } } , f _ { \mathrm { r o b } } )$ , we partition samples into four disjoint sets based on their predictions on the original clean input x:

• Common Set (C): $\{ x \mid f _ { \mathrm { s t d } } ( x ) = y \wedge f _ { \mathrm { r o b } } ( x ) = y \}$ . Samples where features in ${ \mathcal { Z } } _ { r }$ and $\mathcal { Z } _ { n r }$ are synergistic.

• Spurious Set (S): $\{ x \mid f _ { \mathrm { s t d } } ( x ) = y \wedge f _ { \mathrm { r o b } } ( x ) \neq y \}$ . The classic shortcut regime where $f _ { \mathrm { s t d } }$ succeeds via brittle patterns in $\mathcal { Z } _ { n r }$

• Jammed Set $( \mathcal { T } ) \colon \{ x \mid f _ { \mathrm { s t d } } ( x ) \neq y \land f _ { \mathrm { r o b } } ( x ) = y \}$ . Samples where $f _ { \mathrm { s t d } }$ is misguided by antagonistic signals in ${ \mathcal { Z } } _ { n r } .$

• Uninformative Set $( \mathcal { G } ) \colon \{ x \mid f _ { \mathrm { s t d } } ( x ) \neq y \land f _ { \mathrm { r o b } } ( x ) \neq y \}$ . Samples where signals in both manifolds are insufficient for classification.

While S represents the well-documented reliance on human-irrelevant shortcuts, $\mathcal { I }$ exposes a more pathological interaction where non-robust features actively contradict the ground-truth logic:

Definition 2.2 (Natural Jammer). A Natural Jammer is a non-robust signal ${ \mathbf z } _ { n r } \in \mathcal { Z } _ { n r }$ that exhibits feature antagonism, such that its predictive direction contradicts the semantic label y.

Operationally, for a standard model $f _ { \mathrm { s t d } } .$ , an input x is said to be jammed if there exists a semanticspreserving transformation $T \in { \mathcal { T } } ( { \mathrm { e . g . } }$ , spectral filtering or geometric cropping) such that:

$$
f _ { \mathrm { s t d } } ( x ) \neq y \quad \mathrm { a n d } \quad f _ { \mathrm { s t d } } ( T ( x ) ) = y .\tag{1}
$$

In this context, $T$ acts as a de-locking operator that suppresses the antagonistic non-robust signal, allowing the model’s latent discriminative capacity to resurface.

## 2.2 Empirical Prevalence and Stability

We conduct a representative audit on the ImageNet validation set Russakovsky et al. $[ 2 0 1 5 ] ( N =$ 50, 000) to evaluate the prevalence of these sets across different architectures. The membership of the set is defined on the original inputs and, therefore, on the unmodified images $\operatorname { A c c } _ { f _ { \mathrm { s t d } } } ( S ) = 1 0 0 \%$ and $\operatorname { A c c } _ { f _ { \mathrm { s t d } } } ( \mathcal { I } ) = 0 \%$ by construction (and analogously for $f _ { \mathrm { r o b } } )$ . As shown in Table 1, the Jammed Set $\mathcal { I }$ consistently accounts for a significant portion of the error manifold.

![](images/96af3281635493527705431a44edafaa253d711fd862de258f09b2d8a963fe8d.jpg)

![](images/8fc53e515c6690f7070fde6159e55fc94a0fb4203e86cb042c4cabd71b46379c.jpg)  
Figure 2: The Duality of Sensitivity. (a) In the spectral domain, a Cleansing Paradox is observed: removing high-frequencies restores accuracy for the Jammed Set $( \mathcal { T } )$ while destroying it for the Spurious Set (S). (b) In the spatial domain, zooming induces an Sensitivity Inversion transition, showing the reversal of dominance between $\mathcal { I }$ and S features. All curves use the primary lens.

The consistency of these proportions (4.3%–5.0% for $\mathcal { T } )$ across architectures suggests that natural jamming is not confined to a single evaluated model. However, this statistical stability does not necessarily imply that the “Jammed” status is an intrinsic property of the image. If the effect stems from universal data pathologies, the samples in $\mathcal { I }$ should overlap significantly across different models. Conversely, if it is a manifestation of specific inductive biases, these sets should diverge. We investigate this question by dissecting the spectral and spatial dependencies of these features in the next section.

## 3 Spectral-Spatial Locking: The Core Evidence

The divergent behaviors of $f _ { \mathrm { s t d } }$ and $f _ { \mathrm { r o b } }$ on the Jammed and Spurious sets suggest that these models prioritize different structural components within the same image. In this section, we conduct a series of diagnostic experiments to isolate whether these non-robust features are defined by their frequency content, their spatial location, or a combination of both. Our results lead to the discovery of a Spectral-Spatial Locking mechanism.

Evaluation convention. All sets ${ \mathcal { C } } , { \mathcal { S } } , { \mathcal { I } } , { \mathcal { G } }$ are defined using predictions on the original clean inputs. We then apply transformations and evaluate predictions on the same fixed subset of samples, without re-computing set membership after transformation. Unless stated otherwise, all experiments in this section use the primary lens (standard vs. robust ResNet-50, $\ell _ { 2 } , \epsilon = 1 ;$ checkpoints from Salman et al. [2020]).

## 3.1 The Cleansing Paradox: Spectral Sensitivity

We begin by testing whether the interference in the Jammed Set is an intrinsic property of highfrequency (HF) patterns. We apply a low-pass filter (LPF) to progressively remove HF components and observe the accuracy of the standard model $f _ { \mathrm { s t d } }$

Protocol. For each image x, we compute its 2D Fourier transform $\mathcal { F } ( x )$ , apply a circular hard-cutoff mask M with varying radius $r \in [ 1 0 , 1 1 2 ]$ (on a $2 2 4 \times 2 2 4$ spectrum), and reconstruct the filtered image $\tilde { x } _ { r } = \mathcal { F } ^ { - 1 } ( \mathbf { \bar { \boldsymbol { M } } } _ { r } \cdot \mathcal { F } ( \boldsymbol { x } ) )$ . A smaller r corresponds to more aggressive high-frequency removal.

Results. As shown in Figure 2(a), removing HF content yields opposite outcomes for the two sets: Spurious Set (S): Accuracy collapses from 100.0% to 23.7% at $r = 3 0$ . This confirms that these predictions rely entirely on high-frequency shortcuts. Jammed Set (J ): Accuracy recovers from

Table 2: Frequency Grafting: Toxicity Transfer Test. We graft the HF of Jammed images onto Common images. Native Toxicity represents the accuracy gain when HF is removed from the source host. Transferred Toxicity measures the rate at which the wrong label of the jammer infects a new semantic host.
<table><tr><td>TRANSFER SCENARIO</td><td>NATIVE TOXICITY (0% → X%)</td><td>TRANSFERRED TOXICITY (ON NEW HOST)</td></tr><tr><td> ${ \mathcal { I } } \to { \mathcal { C } }$ </td><td>31.8% (0%→31.8%)</td><td>0.17%</td></tr></table>

0.0% to 31.8% at $r = 5 0$ . Common Set (C): Accuracy degrades gracefully from 100.0% to 85.1% at $r = 5 0$ , demonstrating that robust semantic features are largely preserved in the low-frequency skeleton. These findings are diagnostic evidence of the spectralfragility of non-robust features: their predictive utility is tied to specific frequency supports (HF patterns).

## 3.2 The Sensitivity Inversion: Spatial Localization

To determine where these jammers are located, we examine model sensitivity to spatial transformations.

Protocol. We apply a center-crop zoom $T _ { \alpha }$ , where the image is cropped to α × 224 pixels and resized back to 224 × 224 with bilinear interpolation. We sweep the ratio α from 1.0 (original) to 0.5 (2× zoom).

Results. Figure 2(b) reveals a distinct Sensitivity Inversion: as we zoom in, accuracy on Jammed Set (J) increases while accuracy on Spurious Set (S) decreases. Since zooming removes peripheral regions, this pattern is consistent with the hypothesis that many interfering cues are spatially localized near image boundaries or object context, where they can conflict with central semantic evidence. The accuracy on Common Set (C) remains relatively stable, gracefully decreasing from 100% to 79.9%. Robust semantic features exhibit geometric invariance. This sensitivity highlights the spatial fragility of the underlying non-robust signals. This recovery is not specific to the resampling method: it appears with all six interpolation kernels we tested and when a 2% border is painted over instead of cropped.

## 3.3 Frequency Grafting: A Causality Test

The preceding experiments suggest a strong coupling between spectral content and spatial structure. To provide causal evidence, we test whether the interference in $\mathcal { I }$ is an intrinsic HF signal (analogous to a universal adversarial perturbation) or if its efficacy is strictly dependent on the host image’s geometry. We perform a frequency grafting experiment that transplants the HF components of a Jammed image onto the LF skeleton of a Common image.

Protocol. Let $x _ { j } \in { \mathcal { I } }$ be a Jammed image and $x _ { c } \in { \mathcal { C } }$ be a Common image. We construct a hybrid image $x _ { \mathrm { h y b r i d } }$ by combining their spectral components as follows:

$$
x _ { \mathrm { h y b r i d } } = \mathcal { F } ^ { - 1 } \left( M _ { r } \odot \mathcal { F } ( x _ { c } ) + ( 1 - M _ { r } ) \odot \mathcal { F } ( x _ { j } ) \right)\tag{2}
$$

where $\mathcal { F }$ denotes the 2D Discrete Fourier Transform, ⊙ represents element-wise multiplication, and $M _ { r }$ is a circular low-pass binary mask with radius $r = 5 0$ . This operation preserves the semantic skeleton of the Common image while grafting onto it the exact high-frequency textures of the Jammed image. We define transferred toxicity as the fraction of random pairs $( x _ { j } , x _ { c } )$ for which the hybrid inherits the source jammer’s predicted class, i.e., $f _ { \mathrm { s t d } } ( x _ { \mathrm { h y b r i d } } ) = f _ { \mathrm { s t d } } ( x _ { j } )$

Results. As shown in Table 2, the results demonstrate a near-complete collapse of predictive influence upon transplantation. While suppressing high-frequency components yields a 31.8 percentage-point recovery on the native jammed inputs (Section 3.1), the transferred toxicity to a foreign host is only 0.17% (averaged over five random pairings), whereas reassembling a jammer from its own spectral components preserves its jammed prediction in 99.9% of cases.

This result confirms that Natural Jammers are not independently “toxic” patterns. Their capacity to mislead the model is not an intrinsic property of the signal but emerges from a fragile resonance with the host image’s spatial structure. Once the spectral-spatial lock is broken by moving the HF signal to other host images, the feature becomes inefficient. This causal evidence shifts the understanding of non-robustness from a feature-level defect to a systemic interaction between spectrum and space.

## 3.4 Formal Definition: Spectral–Spatial Locking

The empirical evidence in the preceding sections reveals that natural non-robustness is neither a purely spectral defect nor a static spatial anomaly. Instead, it manifests as a structural dependency where the predictive utility of non-robust features is conditioned on the alignment of frequency and space. We formalize this mechanism as follows:

Definition 3.1 (Spectral–Spatial Locking). Let f be a classifier and $x \in \mathcal { X }$ denote an input sample. A non-robust predictive signal z is said to be spectral–spatially locked if its contribution to the model’s inference depends jointly on its frequency support and its spatial configuration. Formally, z is characterized by the following properties:

1. Spectral sensitivity. There exists a low-pass operator $\mathcal { L } _ { \sigma }$ that suppresses frequencies above σ such that removing these components changes the prediction, i.e., $f ( \mathcal { L } _ { \sigma } ( x ) ) \neq f ( x )$ for a non-trivial fraction of inputs exhibiting the signal.

2. Spatial Fragility. Let $\tau$ be a group of semantics-preserving geometric transformations. There exists ${ \bar { T } } \in { \mathcal { T } }$ such that $f ( \tau ( x ) ) \neq f ( x )$ , demonstrating that the efficacy of z is sensitive to precise spatial alignment.

3. Non-Transferability. Let $x _ { A } , x _ { B } \in \mathcal { X }$ be two distinct samples. Grafting the spectral components of $x _ { A }$ onto the spatial structure of $x _ { B }$ fails to reproduce the original prediction bias, as the non-robust signal becomes functionally inert when the original lock is broken.

Definition 3.1 establishes that individual jammers are fragile configurations that collapse when their spectral–spatial lock is disrupted. However, this fragility presents a paradox: if specific locks are so easily broken, why does the aggregate size of the Jammed Set remain approximately stable across models and augmentations? This indicates a redistribution of non-robustness across the data manifold. We will explore this in the next section.

## 4 Substitution Paradox: Systemic Redistribution of Natural Non-Robustness

Section 3 established that many clean-image failures of standard models are consistent with spectral– spatial locking: interfering cues are fragile under mild spectral or geometric perturbations. This fragility raises a natural question: if individual jammers are so easily disrupted, why does non-robust vulnerability persist at scale?

This section provides system-level evidence that natural non-robustness is not confined to a small set of “bad images” but is a persistent outcome of the model–data interaction. We focus on three questions: Q1: Is the prevalence of $\mathcal { I }$ and S stable across architectures and robustness recipes? Q2: Are jammed/spurious identities intrinsic to images, or model-dependent? Q3: Under tiny semantics-preserving transformations, do these pathologies disappear or get redistributed?

Assumption 4.1 (Robust Models as Analysis Lenses). Robust models are used as analysis lenses to induce the quad-set partition via disagreement with a standard model. We do not treat any single robust model as a ground-truth oracle; instead, we test whether conclusions persist across multiple robust oracles spanning architectures and robustness recipes.

Experimental Setup. All analyses are conducted on ImageNet validation $( N = 5 0 , 0 0 0 )$ . For each architecture family, we use its corresponding standard model as $f _ { \mathrm { s t d } }$ and a collection of robust models as candidate lenses $f _ { \mathrm { r o b } }$ that vary in threat model $( \ell _ { 2 } \ \mathrm { o r } \ \ell _ { \infty } )$ and budget ϵ. Within each architecture block, $f _ { \mathrm { s t d } }$ is fixed to that architecture’s standard model, and each robust model defines a quad-set partition ${ \mathcal { C } } , { \mathcal { S } } , { \mathcal { I } } , { \mathcal { G } }$ according to Section 2 on the original clean inputs.

## 4.1 Prevalence Across Architectures

We first establish a baseline fact that will serve as a foundation for the remainder of the section: natural jamming is not rare, nor specific to a single robustness recipe. Table 3 reports the quad-set partition induced by 17 robust lenses spanning CNNs and ViTs.

Table 3: Complete Cross-Model Landscape. Analysis on ImageNet validation set $( N = 5 0 , 0 0 0 )$ Within each architecture block, $f _ { \mathrm { s t d } }$ is fixed to that architecture’s standard model; each row uses a different robust model $f _ { \mathrm { r o b } }$ to define the quad-set partition on clean inputs.
<table><tr><td>Configuration</td><td>Jammed (J)</td><td>Spurious (S)</td><td>Common (C)</td><td>Uninformative (G)</td><td>Rob. Acc.</td></tr><tr><td>ResNet-18</td><td colspan="5">Standard Accuracy: 69.76%</td></tr><tr><td> $\ell _ { 2 } , \epsilon = 0 . 5$ </td><td>2,639 (5.28%)</td><td>4,786 (9.57%)</td><td>30,094 (60.19%)</td><td>12,481 (24.96%)</td><td>65.5%</td></tr><tr><td> $\ell _ { 2 } , \epsilon = 1 . 0$ </td><td>2,448 (4.90%)</td><td>6,168 (12.34%)</td><td>28,712 (57.42%)</td><td>12,672 (25.34%)</td><td>62.3%</td></tr><tr><td> $\ell _ { 2 } , \epsilon = 3 . 0$ </td><td>1,924 (3.85%)</td><td>10,240 (20.48%)</td><td>24,640 (49.28%)</td><td>13,196 (26.39%)</td><td>53.1%</td></tr><tr><td> $\ell _ { \infty } , \epsilon = 1 . 0$ </td><td>2,502 (5.00%)</td><td>5,659 (11.32%)</td><td>29,221 (58.44%)</td><td>12,618 (25.24%)</td><td>63.5%</td></tr><tr><td> $\ell _ { \infty } , \epsilon = 4 . 0$ </td><td>1,867 (3.73%)</td><td>10,497 (21.00%)</td><td>24,383 (48.77%)</td><td>13,253 (26.51%)</td><td>52.5%</td></tr><tr><td>ResNet-50</td><td colspan="5">Standard Accuracy: 76.15%</td></tr><tr><td> $\ell _ { 2 } , \epsilon = 0 . 5$ </td><td>2,317 (4.63%)</td><td>3,803 (7.61%)</td><td>34,270 (68.54%)</td><td>9,610 (19.22%)</td><td>73.2%</td></tr><tr><td> $\ell _ { 2 } , \epsilon = 1 . 0$ </td><td>2,140 (4.28%)</td><td>5,002 (10.00%)</td><td>33,071 (66.14%)</td><td>9,787 (19.57%)</td><td>70.4%</td></tr><tr><td> $\ell _ { 2 } , \epsilon = 3 . 0$ </td><td>1,807 (3.61%)</td><td>8,463 (16.93%)</td><td>29,610 (59.22%)</td><td>10,120 (20.24%)</td><td>62.8%</td></tr><tr><td> $\ell _ { \infty } , \epsilon = 1 . 0$ </td><td>2,252 (4.50%)</td><td>4,306 (8.61%)</td><td>33,767 (67.53%)</td><td>9,675 (19.35%)</td><td>72.0%</td></tr><tr><td> $\ell _ { \infty } , \epsilon = 4 . 0$ </td><td>1,812 (3.62%)</td><td>7,948 (15.90%)</td><td>30,125 (60.25%)</td><td>10,115 (20.23%)</td><td>63.9%</td></tr><tr><td>WideResNet-50-2</td><td colspan="5">Standard Accuracy: 78.48%</td></tr><tr><td> $\ell _ { 2 } , \epsilon = 0 . 5$ </td><td>2,202 (4.40%)</td><td>3,887 (7.77%)</td><td>35,354 (70.71%)</td><td>8,557 (17.11%)</td><td>75.1%</td></tr><tr><td> $\ell _ { 2 } , \epsilon = 1 . 0$ </td><td>2,167 (4.33%)</td><td>4,698 (9.40%)</td><td>34,543 (69.09%)</td><td>8,592 (17.18%)</td><td>73.4%</td></tr><tr><td> $\ell _ { 2 } , \epsilon = 3 . 0$ </td><td>1,910 (3.82%)</td><td>7,701 (15.40%)</td><td>31,540 (63.08%)</td><td>8,849 (17.70%)</td><td>66.9%</td></tr><tr><td> $\ell _ { \infty } , \epsilon = 1 . 0$ </td><td>2,232 (4.46%)</td><td>4,145 (8.29%)</td><td>35,096 (70.19%)</td><td>8,527 (17.05%)</td><td>74.7%</td></tr><tr><td> $\ell _ { \infty } , \epsilon = 4 . 0$ </td><td>2,000 (4.00%)</td><td>7,037 (14.07%)</td><td>32,204 (64.41%)</td><td>8,759 (17.52%)</td><td>68.4%</td></tr><tr><td>ViT-B/16</td><td colspan="5">Standard Accuracy: 81.07%</td></tr><tr><td> $\ell _ { \infty } , \epsilon = 1 . 0$ </td><td>1,569 (3.14%)</td><td>5,467 (10.93%)</td><td>35,067 (70.13%)</td><td>7,897 (15.79%)</td><td>73.3%</td></tr><tr><td>ViT-B/32</td><td colspan="5">Standard Accuracy: 75.91%</td></tr><tr><td> $\ell _ { \infty } , \epsilon = 1 . 0$ </td><td>1,828 (3.66%)</td><td>6,511 (13.02%)</td><td>31,443 (62.89%)</td><td>10,218 (20.44%)</td><td>66.5%</td></tr></table>

Across a wide range of architectures and robustness recipes, the Jammed Set consistently occupies a few percent of ImageNet validation set , and the Spurious Set occupies a larger but still substantial portion (roughly ∼7–20%, depending on ϵ). Natural jamming is thus not an isolated artifact of a particular model or robustness configuration; across these lenses, |J| is relatively stable compared to |S|.

## 4.2 Model-Specific Vulnerabilities

We next test whether jammed is an intrinsic property of the image or depends on the model used to define the partition. For two induced Jammed sets $\mathcal { I } _ { A }$ and $\mathcal { I } _ { B }$ , we quantify agreement via Intersection-over-Union (IoU):

$$
\mathrm { I o U } ( \mathcal { I } _ { A } , \mathcal { I } _ { B } ) = \frac { \vert \mathcal { I } _ { A } \cap \mathcal { I } _ { B } \vert } { \vert \mathcal { I } _ { A } \cup \mathcal { I } _ { B } \vert } ,
$$

## and analogously for S.

Agreement is modest within architecture and low across architectures. Figure 3 shows that even within the same architecture, changing only the robustness budget yields moderate agreement (IoU ≈ 0.40–0.47). Across architectures, IoU collapses (typically < 0.15), and can be even smaller between CNNs and ViTs. These results indicate that which images are jammed/spurious depends strongly on model inductive bias and robustness recipe, rather than being image-intrinsic.

While IoU captures identity, confidence helps assess risk: jammed errors are often high-confidence for standard models. Table 4 reports mean softmax confidence of $f _ { \mathrm { s t d } }$ and the corresponding robust lens $f _ { \mathrm { r o b } }$ on the induced sets. Across CNNs, $f _ { \mathrm { s t d } }$ is typically more confident than $f _ { \mathrm { r o b } }$ on ${ \mathcal { I } } ,$ , consistent with a brittle but dominant cue overwhelming semantic evidence. ViTs show a reversal (robust confidence $>$ standard confidence), which may reflect architecture-specific interfering cues; however, the ViT and CNN lenses also differ in training recipe, so this comparison cannot separate the two factors.

![](images/31b2377d461110ee68e20ab7e7e62f22efb5d318f07420f34b8ace0aa91b872a.jpg)  
Figure 3: Jammed Set IoU Analysis $\scriptstyle ( \ell _ { \infty } ) .$ . Top row: intra-architecture agreement when changing only robustness budget $( \epsilon = 1 \mathrm { v s } \epsilon = 4 )$ . Bottom: cross-architecture IoU matrix. Agreement is modest within the same architecture and collapses across architectures, indicating that jammed identity is model-dependent.

Table 4: Confidence Comparison: Standard vs Robust Models. Each cell reports $f _ { \mathrm { s t d } } / f _ { \mathrm { r o b } }$ mean confidence (max softmax on predicted class). ViT entries use $\ell _ { \infty } , \epsilon { = } 1 . 0 \mathrm { o n l y }$
<table><tr><td></td><td></td><td colspan="3">JAMMED  $( \mathcal { T } )$ </td><td colspan="3">SPURIOUS (S)</td></tr><tr><td>CONFIG</td><td>€</td><td>RN-18</td><td>RN-50</td><td>WRN-50-2</td><td>RN-18</td><td>RN-50</td><td>WRN-50-2</td></tr><tr><td> $\ell _ { 2 }$ </td><td>0.5</td><td>.457/.440</td><td>.511/.509</td><td>.545/.556</td><td>.571/.349</td><td>.624/.412</td><td>.684/.459</td></tr><tr><td> $\ell _ { 2 }$ </td><td>1.0</td><td>.470/.402</td><td>.517/.464</td><td>.555/.531</td><td>.601/.303</td><td>.661/.368</td><td>.710/.427</td></tr><tr><td> $\ell _ { 2 }$ </td><td>3.0</td><td>.491/.318</td><td>.544/.384</td><td>.574/.449</td><td>.678/.217</td><td>.726/.267</td><td>.770/.323</td></tr><tr><td> $\ell _ { \infty }$ </td><td>1.0</td><td>.468/.418</td><td>.513/.485</td><td>.554/.550</td><td>.588/.316</td><td>.643 /.382</td><td>.698 /.445</td></tr><tr><td> $\ell _ { \infty }$ </td><td>4.0</td><td>.490/.319</td><td>.538 /.394</td><td>.577/.464</td><td>.680/.206</td><td>.717/.270</td><td>.760/.321</td></tr><tr><td colspan="2"> $\mathrm { V I T - B } / 1 6 ( \ell _ { \infty } { = } 1 . 0 )$ </td><td></td><td>.504/.701</td><td></td><td></td><td>.640/.584</td><td></td></tr><tr><td colspan="2"> $\mathrm { V I T - B } / 3 2 ( \ell _ { \infty } { = } 1 . 0 )$ </td><td></td><td>.447 /.670</td><td></td><td></td><td>.591/.541</td><td></td></tr></table>

## 4.3 The Substitution Paradox

Section 3 demonstrated that individual jammed behaviors can be neutralized by mild spectral filtering or small geometric shifts, suggesting that jamming depends on fragile spectral–spatial alignment. We now formalize and test a stronger claim: tiny transformations can break many specific jammed identities, yet the overall prevalence ofjamming remains approximately conserved.

Definition 4.2 (Transformed Partition and Substitution Statistics). Fix an architecture and its standard model $f _ { \mathrm { s t d } }$ , together with a robust lens $f _ { \mathrm { r o b } }$ that induces the original sets ${ \mathcal { C } } , { \mathcal { S } } , { \mathcal { I } } , { \mathcal { G } }$ on clean inputs. For a transformation T, define the transformed Jammed set by evaluating both models on $T ( x )$

$$
\mathcal { I } _ { T } : = \{ x : f _ { \mathrm { s t d } } ( T ( x ) ) \neq y \ \land \ f _ { \mathrm { r o b } } ( T ( x ) ) = y \} ,
$$

Table 5: Structural Fragility + Substitution Paradox. Minimal edge crops substantially change the identity of jammed/spurious samples (IoU drops sharply), yet the prevalence (set size) remains approximately stable.
<table><tr><td rowspan="2">CROP</td><td colspan="2">SET SIZE</td><td colspan="2">IOU VS. ORIGINAL</td></tr><tr><td> $\vert \mathcal { I } _ { T } \vert$ </td><td> $| S _ { T } |$ </td><td>J</td><td>S</td></tr><tr><td>0% (ORIGINAL)</td><td>2,140</td><td>5,002</td><td>1.00</td><td>1.00</td></tr><tr><td>2%</td><td>2,289</td><td>5,193</td><td>0.46</td><td>0.58</td></tr><tr><td>4%</td><td>2,268</td><td>5,208</td><td>0.42</td><td>0.54</td></tr><tr><td>6%</td><td>2,276</td><td>5,127</td><td>0.38</td><td>0.51</td></tr><tr><td>8%</td><td>2,291</td><td>5,159</td><td>0.35</td><td>0.49</td></tr></table>

and analogously $s _ { T } , \mathcal { C } _ { T } , \mathcal { G } _ { T }$ . We quantify (i) identity stability via $\mathrm { I o U } ( \mathcal { I } , \mathcal { I } _ { T } )$ , and (ii) prevalence stability via the transformed set sizes $\vert \mathcal { I } _ { T } \vert$ and $| S _ { T } |$

Proposition 4.3 (Substitution Paradox (Empirical)). For small semantics-preserving geometric transformations $T ( e . g . , \leq 8 \%$ edge crop with resize), we observe simultaneously:

1. Identity instability: $\mathrm { I o U } ( \mathcal { I } , \mathcal { I } _ { T } )$ (and IoU $( S , S _ { T } ) )$ is low.

2. Prevalence stability: the set sizes $\vert \mathcal { I } _ { T } \vert$ and $| S _ { T } |$ remain approximately unchanged relative to $\left| { \mathcal { I } } \right| a n d \left| S \right|$

Thus, small geometric changes tend to redistribute jammed/spurious identities across samples rather than eliminate non-robust vulnerability.

We apply minimal center crops removing only 2%, 4%, 6%, or 8% of edge pixels and resize back to $2 2 4 \times 2 2 4$ . For each crop ratio, we compute the transformed partition (Definition 4.2) and report both set sizes and IoU against the original partition.

Identity instability (structural fragility): Even a 2% crop yields $\mathrm { I o U } ( \mathcal { I } , \mathcal { I } _ { T } ) = 0 . 4 6$ , implying that about one third of the original jammed images leave $\mathcal { I }$ , and a similar number of new ones enter, under an almost imperceptible geometric perturbation. This supports the view that the underlying cues depend on fragile spatial alignment, consistent with spectral–spatial locking.

Prevalence stability (substitution): Despite the large change in identity, the transformed set sizes remain stable: $| \mathcal { I } _ { T } | \approx 2 . 2 \mathrm { k }$ and $| S _ { T } | \approx 5 .$ .1k across crop ratios. This stability is approximate rather than exact: in a re-run that tracks individual images, a 2% crop moves 738 of the 2,140 jammers out of $\mathcal { I }$ and 873 new images in. Thus, breaking many specific locks does not reduce overall vulnerability; instead, new jammed/spurious identities emerge elsewhere. This resolves the tension raised by Section 3: individual jammers are fragile, yet aggregate non-robustness persists as a systemic property and is redistributed under small transformations. Similar turnover appears under mild low-pass filtering and on six further datasets; our claim is limited to these tested mild transformations. In this view, natural non-robustness is a systemic $\mathbf { \mu } ^ { 6 6 } \mathbf { t a x } ^ { 9 9 }$ that the model must pay to achieve high training accuracy.

We formalized and verified the Substitution Paradox: tiny semantics-preserving geometric changes drastically alter jammed/spurious identities while leaving their prevalence approximately unchanged.

## 5 Related Work

## 5.1 Adversarial Robustness and Non-robust Features

The vulnerability of deep neural networks to adversarial examples Szegedy et al. [2013], Goodfellow et al. [2014], Madry et al. [2017] has been increasingly understood through the lens of data-driven features. Ilyas et al. [2019] famously posited that adversarial examples arise from non-robust features—patterns that are highly predictive yet imperceptible to humans—and that these features are actively useful for standard classification. This framework has led to a prevailing view of non-robust features as “spurious shortcuts” Geirhos et al. [2020], Shah et al. [2020] that models exploit to achieve high standard accuracy, often resulting in an inherent trade-off between robustness and clean performance Tsipras et al. [2018], Zhang et al. [2019], Rebuffi et al. [2021].

This view has since been actively debated. Kim et al. [2021] show via information bottleneck that non-robust features are decomposable, with only a subset driving adversarial predictions. Li et al. [2023] argue that non-robust features are paradigm-specific shortcuts useful only under supervised objectives, not across learning paradigms. Crucially, however, both works study features under intentional adversarial perturbation—inputs explicitly crafted to expose model vulnerabilities.

We ask a fundamentally different question: can non-robust features mislead standard models on clean, unmodified natural images, without any perturbation? We identify a Jammed Set (J) where robust models outperform standard models on standard test images, and provide causal evidence for Spectral–Spatial Locking as the driving mechanism. The antagonism we document is not injected by an attacker but arises spontaneously from the interaction between standard training and the spectral-spatial structure of natural data. Our Substitution Paradox further reveals this to be a redistributing property of the model–data geometry: a model pair defines a failure field $j ( x ) = 1 [ f _ { \mathrm { s t d } } ( x ) \neq \bar { y } \land f _ { \mathrm { r o b } } ( x ) = y ]$ over natural images, which are commonly assumed to lie near a low-dimensional data manifold Fefferman et al. [2016], Stutz et al. [2019], and small transformations move the support of j while approximately preserving its mass. This provides a complementary framework to studies of natural adversarial or hard examples Hendrycks et al. [2021], Recht et al. [2019], Li et al. [2024], Liu et al. [2025b], Zhang et al. [2025], which rely on curated atypical subsets rather than identifying a structural mechanism within standard uncurated data.

## 5.2 Spectral and Geometric Bias in Generalization

The inductive biases of CNNs are often characterized by their sensitivity to specific input frequencies and spatial structures. From a Fourier perspective, models frequently exhibit a spectral bias, relying on high-frequency components that are critical for generalization but lack robustness Yin et al. [2019], Wang et al. [2020], Fang and Xu [2024], Krizhevsky et al. [2012], Xia et al. [2024]. Simultaneously, the lack of geometric invariance—often caused by aliasing in downsampling layers Zhang [2019]— renders models fragile to even minor image transformations Azulay and Weiss [2019], Michaeli and Soudry [2025]. This fragility is often linked to a "texture bias," where models prioritize local patterns over global shapes Geirhos et al. [2018], Hermann et al. [2020].

Our work synthesizes these views through the mechanism of Spectral–spatial Locking, providing causal evidence that natural jamming is contingent on the precise alignment of frequency and space. Furthermore, we address the stability of these vulnerabilities, i.e., the Substitution Paradox where the aggregate rate of jamming remains stable across the data manifold. This observation relates to the broader problem of underspecification D’Amour et al. [2022], where models with identical training objectives exhibit divergent and unpredictable failure modes under distribution shifts Taori et al. [2020], Arjovsky et al. [2019], Sagawa et al. [2019].

## 6 Conclusion

We show that non-robust features on clean natural images can be antagonistic: on a Jammed Set, robust models outperform standard models without modifying inputs. Using a standard-vs-robust partition of ImageNet-Val, we characterize Jammed and Spurious regimes and provide evidence for spectral–spatial locking via frequency filtering, geometric perturbations, and frequency grafting.

At the system level, jammed identities are highly model-dependent while prevalence remains stable, yielding the Substitution Paradox: small semantics-preserving transformations reshuffle jammed membership without reducing its overall rate.

## Acknowledgments and Disclosure of Funding

This research / project is supported by A\*STAR under its Japan-Singapore Joint Call: JST-A\*STAR 2024 (Project ID: R2416IR139). This research is also supported by the Ministry of Education, Singapore, under its Academic Research Fund Tier 2 (Award MOE-T2EP20125-0005). The authors declare no competing interests.

## References

M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

A. Azulay and Y. Weiss. Why do deep convolutional networks generalize so poorly to small image transformations? Journal ofMachine Learning Research, 20(184):1–25, 2019.

A. D’Amour, K. Heller, D. Moldovan, B. Adlam, B. Alipanahi, A. Beutel, C. Chen, J. Deaton, J. Eisenstein, M. D. Hoffman, et al. Underspecification presents challenges for credibility in modern machine learning. Journal ofMachine Learning Research, 23(226):1–61, 2022.

M. Eriksson, E. Purificato, A. Noroozian, J. Vinagre, G. Chaslot, E. Gomez, and D. Fernandez-Llorca. Can we trust ai benchmarks? an interdisciplinary review of current issues in ai evaluation. arXiv preprint arXiv:2502.06559, 2025.

R. Fang and Y. Xu. Addressing spectral bias of deep neural networks by multi-grade deep learning. Advances in Neural Information Processing Systems, 37:114122–114146, 2024.

C. Fefferman, S. Mitter, and H. Narayanan. Testing the manifold hypothesis. Journal of the American Mathematical Society, 29(4):983–1049, 2016. doi: 10.1090/jams/852.

R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and W. Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In International conference on learning representations, 2018.

R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

B. Gyevnar and A. Kasirzadeh. Ai safety for everyone. Nature Machine Intelligence, pages 1–12, 2025.

D. Hendrycks, K. Zhao, S. Basart, J. Steinhardt, and D. Song. Natural adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15262–15271, 2021.

K. Hermann, T. Chen, and S. Kornblith. The origins and prevalence of texture bias in convolutional neural networks. Advances in neural information processing systems, 33:19000–19015, 2020.

A. Ilyas, S. Santurkar, D. Tsipras, L. Engstrom, B. Tran, and A. Madry. Adversarial examples are not bugs, they are features. Advances in neural information processing systems, 32, 2019.

E. Kafali, T. Semertzidis, and P. Daras. Dynamic trade-offs in adversarial training: Exploring efficiency, robustness, forgetting, and interpretability. Neural Processing Letters, 57(3):47, 2025.

J. Kauffmann, J. Dippel, L. Ruff, W. Samek, K.-R. Müller, and G. Montavon. Explainable ai reveals clever hans effects in unsupervised learning models. Nature Machine Intelligence, pages 1–11, 2025.

J. Kim, B.-K. Lee, and Y. M. Ro. Distilling robust and non-robust features in adversarial examples by information bottleneck. Advances in Neural Information Processing Systems, 34:17148–17159, 2021.

A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

A. Li, Y. Wang, Y. Guo, and Y. Wang. Adversarial examples are not real features. Advances in Neural Information Processing Systems, 36:17222–17237, 2023.

B. Li, Z. Lin, W. Peng, J. d. D. Nyandwi, D. Jiang, Z. Ma, S. Khanuja, R. Krishna, G. Neubig, and D. Ramanan. Naturalbench: Evaluating vision-language models on natural adversarial samples. Advances in Neural Information Processing Systems, 37:17044–17068, 2024.

J. Li and G. Li. Triangular trade-off between robustness, accuracy, and fairness in deep neural networks: A survey. ACM Computing Surveys, 57(6):1–40, 2025.

C. Liu, Y. Dong, W. Xiang, X. Yang, H. Su, J. Zhu, Y. Chen, Y. He, H. Xue, and S. Zheng. A comprehensive study on robustness of image classification models: Benchmarking and rethinking. International Journal ofComputer Vision, 133(2):567–589, 2025a.

L. Liu, Y. Liang, X. Yan, L. Huangfu, S. Samtani, Z. Yu, Y. Zhang, and D. D. Zeng. Hard sample mining: A new paradigm of efficient and robust model training. IEEE Transactions on Neural Networks and Learning Systems, 2025b.

A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.

H. Michaeli and D. Soudry. Alias-free vit: Fractional shift invariance via linear attention. arXiv preprint arXiv:2510.22673, 2025.

S.-A. Rebuffi, S. Gowal, D. A. Calian, F. Stimberg, O. Wiles, and T. Mann. Fixing data augmentation to improve adversarial robustness. arXiv preprint arXiv:2103.01946, 2021.

B. Recht, R. Roelofs, L. Schmidt, and V. Shankar. Do imagenet classifiers generalize to imagenet? In International conference on machine learning, pages 5389–5400. PMLR, 2019.

O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015.

S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731, 2019.

H. Salman, A. Ilyas, L. Engstrom, A. Kapoor, and A. Madry. Do adversarially robust imagenet models transfer better? Advances in Neural Information Processing Systems, 33:3533–3545, 2020.

H. Shah, K. Tamuly, A. Raghunathan, P. Jain, and P. Netrapalli. The pitfalls of simplicity bias in neural networks. Advances in Neural Information Processing Systems, 33:9573–9585, 2020.

D. Stutz, M. Hein, and B. Schiele. Disentangling adversarial robustness and generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

S. Suzuki, S. Yamaguchi, S. Takeda, S. Kanai, N. Makishima, A. Ando, and R. Masumura. Adversarial finetuning with latent representation constraint to mitigate accuracy-robustness tradeoff. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4367–4378. IEEE, 2023.

C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.

R. Taori, A. Dave, V. Shankar, N. Carlini, B. Recht, and L. Schmidt. Measuring robustness to natural distribution shifts in image classification. Advances in Neural Information Processing Systems, 33: 18583–18599, 2020.

D. Tsipras, S. Santurkar, L. Engstrom, A. Turner, and A. Madry. Robustness may be at odds with accuracy. arXiv preprint arXiv:1805.12152, 2018.

H. Wang, X. Wu, Z. Huang, and E. P. Xing. High-frequency component helps explain the generalization of convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8684–8694, 2020.

J. Xia, Z. Yue, Y. Zhou, Z. Ling, Y. Shi, X. Wei, and M. Chen. Waveattack: Asymmetric frequency obfuscation-based backdoor attacks against deep neural networks. Advances in Neural Information Processing Systems, 37:43549–43570, 2024.

D. Yin, R. Gontijo Lopes, J. Shlens, E. D. Cubuk, and J. Gilmer. A fourier perspective on model robustness in computer vision. Advances in Neural Information Processing Systems, 32, 2019.

H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui, and M. Jordan. Theoretically principled tradeoff between robustness and accuracy. In International conference on machine learning, pages 7472–7482. PMLR, 2019.

J. Zhang, F. Liu, D. Zhou, J. Zhang, and T. Liu. Improving accuracy-robustness trade-off via pixel reweighted adversarial training. arXiv preprint arXiv:2406.00685, 2024.

R. Zhang. Making convolutional networks shift-invariant again. In International conference on machine learning, pages 7324–7334. PMLR, 2019.

W. Zhang, P. L. K. Ding, and H. Liu. Beyond imagenet: Understanding cross-dataset robustness of lightweight vision models. arXiv preprint arXiv:2511.00335, 2025.