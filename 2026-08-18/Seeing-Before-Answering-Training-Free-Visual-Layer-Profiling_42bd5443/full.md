# Seeing Before Answering: Training-Free Visual Layer Profiling for Vision-Language Models

Ruchen Liu1,\*0, Yi Yang1,\*0, Yiming Xu1,\*0, Michael Ying Yang20, Monika Sester1, and Bodo Rosenhahn1,\*\*

1 Leibniz Universität Hannover, Germany 2 University of Bath, UK

Abstract. LLaVA-style Vision-Language Models (VLMs) pass visual tokens from a fixed late layer of the vision backbone, typically the penultimate one, to the language model. We first show that this hidden convention is fragile: across 2 VLMs and 7 image and video benchmarks, the default layer is sub-optimal in 13 of 14 model-task pairs, and the best layer shifts with both task and visual backbone. Finding that layer by exhaustive layer-wise inference is prohibitively expensive, and no better fixed default exists. We therefore ask whether layer usefulness can instead be predicted from representation geometry. We study matrix-based entropy, introduced for unimodal layer analysis, which we compute over samplelevel visual embeddings as Visual Dataset Entropy (VDE); and Gromov-Wasserstein (GW) distance, introduced for encoder-level VLM model selection, which we repurpose as a layer-wise visual–language alignment signal. Transferring these to LLaVA-based models is not obvious a priori: the vision tower is frozen while the multimodal projector is trained, so we profile both sides of the projector. We find that VDE transfers, and GW does not. Computed from 100 unlabeled task samples without downstream inference, pre-projector VDE tracks layer-wise accuracy and its top-ranked layers cover the oracle best layer on every task for the SigLIP-based LLaVA-Video, while giving region-level guidance for the CLIP-based Video-LLaVA. Post-projector profiles show that the projector reshapes visual geometry but does not erase the performance-relevant trend, leaving $\mathrm { V D E _ { p r e } }$ the stronger signal. GW instead flattens after projection and is best read as an alignment diagnostic rather than a selector. VDE thus offers an interpretable, training-free policy that narrows the visual-layer search to a handful of candidates for limited downstream verification.

Keywords: Vision-Language Models · Visual Layer Selection · Representation Profiling

## 1 Introduction

Many Vision-Language Models (VLMs) use a single fixed layer of the vision backbone as the interface to the language model, and the widely-adopted LLaVA family [8, 11, 32] fixes this layer to a late depth, typically the penultimate one. This design is simple and widely adopted, but it makes the visual-layer choice a hidden architectural assumption: the same representation depth is expected to support different tasks, domains, and input modalities.

Table 1: Preliminary finding: the vision backbone's default layer is often sub-optimal.
<table><tr><td>Task</td><td>LLaVA-Video [32] / SigLIP [31] Default Layer = 26 Default Acc. Best Acc. Best Layer Default Acc. Best Acc. Best Layer</td><td></td><td>Video-LLaVA [11] / CLIP [21] Default Layer = 23</td></tr><tr><td>ScienceQA [16]</td><td>87.62</td><td>88.90 27</td><td>59.63 59.63</td></tr><tr><td>POPE-Adv. [10]</td><td>86.60</td><td>87.00 27</td><td>79.80 80.53</td></tr><tr><td>CV-Bench 2D [26]</td><td>68.50</td><td>70.20 27</td><td>22 49.86 53.20 20</td></tr><tr><td>CV-Bench 3D [26]</td><td>82.00 83.17</td><td>27</td><td>57.33 58.83 21</td></tr><tr><td>MMStar [5]</td><td>57.20 58.07</td><td>27</td><td>32.40 33.00</td></tr><tr><td>HD-EPIC Action [19]</td><td>59.50 61.00</td><td>25</td><td>21 32.00 33.00 22 36.40</td></tr><tr><td>HD-EPIC Gaze [19]</td><td>56.00 57.70</td><td>23</td><td>37.00 21</td></tr></table>

Layer-wise evaluation reveals that this assumption is fragile. Downstream performance varies systematically across visual depth, and the default layer is not always optimal, as shown in Table 1. This effect appears across both image and video benchmarks, indicating that visual-layer choice is not merely an implementation detail but a meaningful factor in VLM behavior. Exhaustive layer-wise inference [4, 13,24] can identify the best layer, but it requires repeated full evaluations and becomes costly for LLaVA-style large video VLMs.

This paper studies visual-layer choice as an interpretability problem. Rather than treating the best layer as a black-box empirical outcome, the analysis focuses on whether layer-wise representation geometry explains which visual layers are useful. The key finding is that sample-level visual diversity is strongly associated with downstream performance. Layers with richer and more discriminative sample geometry tend to form high-performing regions, and this signal is observable before answer generation. The analysis is instantiated with Visual Dataset Entropy (VDE) [24|, a matrix-entropy-based representation probe. Given a small set of task samples, VDE measures the diversity of sample-level visual embeddings at each candidate layer. VDE is computed both before and after the multimodal projector. The pre-projector profile reflects the intrinsic geometry of the vision backbone, while the post-projector profile captures how this geometry is reshaped at the vision-language interface.

Cross-modal structure is further examined with Gromov-Wasserstein (GW) [9] distance to compare visual and language-side sample geometries. GW provides a coordinate-free diagnostic of visual-language compatibility across layers. In the experiments, GW reveals alignment trends across depth, while VDE provides the more reliable practical signal for layer profiling.

Experiments on LLaVA-Video [32] with a SigLIP [31] vision tower and Video-LLaVA [11] with a CLIP [21] vision tower show that high-VDE regions largely overlap with strong downstream layers across diverse image and video benchmarks. For the SigLIP-based model, VDE retrieves the best-performing layer within a small candidate set across all evaluated tasks. For the CLIP-based model, VDE is less exact as a strict Top-3 selector, but still identifies compact high-performing regions and filters out clearly weak layers. These results support entropy-based profiling as a simple, training-free policy for narrowing visual-layer search and as an interpretable tool for understanding VLM visual representations.

The main findings and contributions are summarized as follows:

Fixed late-layer extraction is shown to be a fragile convention in LLaVAstyle VLMs: downstream performance changes systematically across visual depth, and the default layer is not always optimal.

Visual Dataset Entropy is validated as a simple training-free policy for narrowing visual-layer search, using only a small set of task samples and no answer generation.

The multimodal projector is shown to reshape layer-wise representation geometry, while Gromov-Wasserstein distance provides diagnostic evidence for visual-language alignment across depth.

## 2 Related Work

Layer-wise Representation Analysis in Unimodal Models. Layer depth strongly shapes what a network encodes. In vision transformers, representations evolve from local spatial structure in early layers to object-level and semantic abstractions in deeper ones [3, 7, 22]. Parallel studies in language models, using linear probes [1] and intrinsic-geometry analyses [6, 27], similarly find that intermediate rather than final layers often carry the most transferable or abstract information. To characterize such structure without task labels, informationtheoretic and spectral probes have been used, including representation rank, effective rank, and matrix-based entropy [23, 25, 28]. Matrix-based Rényi entropy in particular estimates information quantities directly from representation matrices and has been used to analyze information flow and layer-wise dynamics in deep networks [25, 29, 30], revealing structured representation dynamics across both language and vision models [24]. These works establish entropy as an analytical tool for understanding internal representations, but remain confined to unimodal settings.

Visual-Layer Selection and Cross-Modal Alignment in VLMs. Modern vision-language models (VLMs) connect a vision tower such as CLIP or SigLIP to a large language model through a multimodal projector [11, 21, 31, 32], and typically feed the language model visual tokens from a single fixed layer; the widely-used LLaVA family fixes this to a late layer, commonly the penultimate one [8, 11, 17, 32]. This convention provides a stable interface for instruction tuning but decouples visual-layer choice from task requirements. Recent work questions this default, showing that intermediate or shallower visual layers can be more effective for certain tasks [4] and motivating methods that fuse features from multiple visual layers [2, 12]. A related line studies the projector itself, showing that its design affects cross-modal alignment and instruction tuning [14, 15]; we instead ask how it reshapes layer-wise visual geometry at inference time. Since visual and language features live in different spaces, direct comparison is difficult. Interpretability work carried out by Liu et al. traces how visual information propagates inside the language model, finding that its layers augment but can also degrade the visual signal received from the encoder [13]. Gromov-Wasserstein (GW) distance quantitatively compares the relational geometry within each space [18, 20] and has recently been applied to VLM model selection [9]. We adopt GW as one alignment probe and find it to be a useful diagnostic across visual layers, complementing entropy-based profiling.

![](images/48f394940cbb38cd53e2522d1ae12c7c782bf5053e9cd915cf3399693b3b1b3a.jpg)  
Fig. 1: Overview of the proposed visual-layer profiling framework. Two training-free geometric signals are computed across candidate layers: VDE, measured from pre- and post-projector visual representations, and GW, measuring visual-language structural alignment. Both are profiled against downstream layer usefulness to assess how well each predicts strong visual layers.

Summary. While layer-wise analysis is well developed for unimodal models, whether its conclusions transfer to VLM visual layers is underexplored, and few existing methods select the best visual layer without running full downstream inference. We bridge this gap by extending representation profiling to VLMs with a training-free pipeline that predicts layer usefulness from sample-level visual geometry and presents it as a practical tool.

## 3 Method

We ask whether a small set of task samples can reveal which layer of a VLM vision tower yields the most effective visual representation, without answer generation, gradient updates, or exhaustive downstream inference. Figure 1 provides an overview of our visual-layer profiling framework. We focus on two trainingfree geometric signals from prior representation-analysis research: visual dataset entropy (VDE) [24], a matrix-based entropy; and Gromov-Wasserstein (GW)

distance between visual and language features [9, 20]. We investigate whether they transfer from the unimodal setting in which they were introduced to the frozen vision backbones of VLMs, and how they behave on either side of the multimodal projector. We first formalize the layer-profiling task (Section 3.1), then review the two geometric tools we adopt (Section 3.2), and finally describe how we instantiate them as layer-wise visual profiles (Section 3.3).

## 3.1 Problem Definition

Background. A modern VLM connects a vision tower, such as CLIP [21] or SigLIP [31], to a large language model (LLM) through a multimodal projector [11, 32]. The vision tower encodes an input image or video into token-level features at each of its L layers. In standard implementations, features from a single fixed layer, commonly the penultimate layer, are passed to the projector, which maps them into the LLM embedding space for downstream reasoning. Crucially, during multimodal instruction tuning the vision backbone is typically kept frozen while the projector is trained. As a result, the visual features before the projector are inherited directly from vision pretraining, whereas the features after the projector are additionally reshaped by multimodal training. This asymmetry motivates analyzing layer-wise representations separately on the two sides of the projector.

Task. Let the vision tower expose candidate layers $l \in \mathcal { L } = \{ 1 , . . . , L \}$ , and

$$
\mathcal { D } _ { s } = \{ ( x _ { i } , q _ { i } ) \} _ { i = 1 } ^ { N }\tag{1}
$$

be a small set of task samples, where $x _ { i }$ is an image or video and $q _ { i }$ is the corresponding question or task prompt. Let Acc(l) denote the downstream task performance obtained when the VLM reads visual features from layer l, and

$$
l _ { \mathrm { b e s t } } = \arg \operatorname* { m a x } _ { l \in \mathcal { L } } \mathrm { A c c } ( l )\tag{2}
$$

be the oracle layer identified by exhaustive layer-wise inference. Evaluating Acc(l) for every layer requires a full downstream pass per candidate and is prohibitively expensive as $\mathcal { L }$ and the benchmark grow.

The goal of layer-wise visual profling is to find a training-free score $s ( l )$ computed from $\mathcal { D } _ { s }$ alone, i.e., without answer generation, gradient updates, or downstream inference, whose top-ranked layers contain the oracle layer. We call such a score effective if, for a small budget k,

$$
l _ { \mathrm { b e s t } } \in \mathrm { T o p K } _ { k } \bigl ( s ( l ) \bigr ) ,\tag{3}
$$

so that profiling narrows the search from all L layers to a compact candidate set that can then be verified with far fewer full-inference runs. Whether any purely geometric, training-free score satisfies this criterion for VLM vision backbones is not known a priori. We therefore treat several candidate signals derived from representation geometry (introduced in Section 3.2) as competing hypotheses, and our profiling process (Section 3.3) decides which, if any, is a reliable selector.

## 3.2 Preliminaries

We build our profiles on two existing training-free geometric tools. Both operate on a representation matrix whose rows are sample-level embeddings, and neither requires labels or answer generation.

Matrix-based entropy. Matrix-based entropy quantifies the effective diversity of a set of representations through the spectrum of their Gram matrix [24, 25]. Given a representation matrix $\bar { Z } \in \mathbb { R } ^ { N \times d }$ with one embedding per sample, we center and normalize each row,

$$
\bar { z } ^ { i } = \frac { z ^ { i } - \mu } { \| z ^ { i } - \mu \| _ { 2 } + \epsilon } , \qquad \mu = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } z ^ { i } ,\tag{4}
$$

and form the trace-normalized Gram matrix

$$
A = { \frac { { \bar { Z } } { \bar { Z } } ^ { \top } } { \operatorname { t r } ( { \bar { Z } } { \bar { Z } } ^ { \top } ) } } ,\tag{5}
$$

where $\bar { Z }$ stacks the normalized rows $\bar { z } ^ { i }$ . With $\{ \lambda _ { j } ( A ) \}$ the eigenvalues of A, the entropy and its max-entropy-normalized form are

$$
\mathcal { H } ( Z ) = - \sum _ { j } \lambda _ { j } ( A ) \log \lambda _ { j } ( A ) , \qquad \widehat { \mathcal { H } } ( Z ) = \frac { \mathcal { H } ( Z ) } { \log \operatorname* { m i n } ( N , d ) } .\tag{6}
$$

A higher $\widehat { \mathcal { H } } ( Z )$ indicates that the sample embeddings spread across many directions, while a lower value indicates concentration in a few dominant directions. Prior work established this measure for unimodal language and vision representations; we adopt it unchanged and apply it to visual layers, referring to the resulting quantity as visual dataset entropy (VDE) in Section 3.3.

Gromov-Wasserstein distance. Matrix entropy characterizes a single representation space but says nothing about whether two spaces are structurally compatible. Gromov-Wasserstein (GW) distance compares the relational geometry of two representation sets without requiring them to share a coordinate system [9, 20]. Given a source matrix P and a target matrix $Q { \mathrm { . } }$ each with N rows, we form intra-set pairwise distance matrices

$$
D _ { P } ( i , j ) = d ( p _ { i } , p _ { j } ) , \qquad D _ { Q } ( i , j ) = d ( q _ { i } , q _ { j } ) ,\tag{7}
$$

where $d ( \cdot , \cdot )$ is the angular cosine distance. Following [9], we match the scales of the two spaces by median normalization,

$$
\widetilde { D } _ { P } = \frac { \mathrm { m e d i a n } ( D _ { Q } ) } { \mathrm { m e d i a n } ( D _ { P } ) } D _ { P } ,\tag{8}
$$

and define the GW distance as

$$
\mathrm { G W } ( P , Q ) = \operatorname* { m i n } _ { \pi \in \pi ( a , b ) } \sum _ { i , j , k , m } \left| \widetilde { D } _ { P } ( i , j ) - D _ { Q } ( k , m ) \right| \pi _ { i k } \pi _ { j m } ,\tag{9}
$$

where π is the transport plan and $^ { a , }$ b are uniform marginals. A lower ${ \mathrm { G W } } ( P , Q )$ indicates that the two sets share more similar internal geometry. This tool was introduced for encoder-level VLM model selection [9]; we repurpose it as a layerlevel cross-modal signal in Section 3.3, and assess its usefulness for layer selection empirically alongside VDE.

## 3.3 Layer-wise Visual Representation Profiling

We now instantiate the two tools above on the layer-wise representations of a frozen VLM to obtain a training-free profile for every candidate layer.

Layer-wise visual representations. Let $f _ { l }$ map an input to the hidden tokens of the l-th vision layer:

$$
H _ { l } ^ { i } = f _ { l } ( x _ { i } ) \in \mathbb R ^ { T _ { i } \times d _ { v } } ,\tag{10}
$$

where $T _ { i }$ is the number of visual tokens (including tokens from all sampled frames for video) and $d _ { v }$ is the vision hidden dimension. We reduce each sample to a pre-projector embedding by mean pooling and stack all samples:

$$
z _ { l , \mathrm { p r e } } ^ { i } = \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } H _ { l , t } ^ { i } , \qquad Z _ { l , \mathrm { p r e } } = [ z _ { l , \mathrm { p r e } } ^ { 1 } , \dots , z _ { l , \mathrm { p r e } } ^ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times d _ { v } } .\tag{11}
$$

The multimodal projector $g ( \cdot )$ maps visual tokens into the language hidden space; for the MLP projectors used in our models, $g ( H _ { l } ^ { i } ) = W _ { 2 } \sigma ( W _ { 1 } H _ { l } ^ { i } )$ with nonlinearity $\sigma$ . The corresponding post-projector representation is

$$
z _ { l , \mathrm { p o s t } } ^ { i } = \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } g ( H _ { l } ^ { i } ) _ { t } , \qquad Z _ { l , \mathrm { p o s t } } = [ z _ { l , \mathrm { p o s t } } ^ { 1 } , \dots , z _ { l , \mathrm { p o s t } } ^ { N } ] ^ { \top } .\tag{12}
$$

Comparing profiles computed from $\mathcal { Z } _ { l , \mathrm { p r e } }$ and $Z _ { l , \mathrm { p o s t } }$ lets us separate the intrinsic diversity of a vision layer from the diversity actually delivered to the LLM after projection.

Question-side language representations. For the GW signal, we also extract a language-side embedding for each sample, without answer generation. We feed the question, or the question with options when available, into the frozen LLM and take the last valid token representation from the penultimate layer:

$$
\begin{array} { r } { u _ { i } = h _ { - 2 , \mathrm { l a s t } } ( q _ { i } ) , \qquad U = [ u _ { 1 } , \dots , u _ { N } ] ^ { \top } . } \end{array}\tag{13}
$$

Layer-wise profiles. Applying matrix entropy to the visual representation matrices yields the pre- and post-projector visual dataset entropy,

$$
\mathrm { V D E } _ { \mathrm { p r e } } ( l ) = \widehat { \mathcal { H } } ( Z _ { l , \mathrm { p r e } } ) , \qquad \mathrm { V D E } _ { \mathrm { p o s t } } ( l ) = \widehat { \mathcal { H } } ( Z _ { l , \mathrm { p o s t } } ) ,\tag{14}
$$

where $\mathrm { V D E _ { p r e } }$ measures the intrinsic diversity of a vision layer and $\mathrm { V D E _ { p o s t } }$ characterizes how the projector reshapes it. Applying the GW distance between each visual representation matrix and the question-side matrix U yields the cross-modal alignment signals

$$
\mathrm { G W } _ { \mathrm { p r e } } ( l ) = \mathrm { G W } ( \boldsymbol { Z } _ { l , \mathrm { p r e } } , \boldsymbol { U } ) , \qquad \mathrm { G W } _ { \mathrm { p o s t } } ( l ) = \mathrm { G W } ( \boldsymbol { Z } _ { l , \mathrm { p o s t } } , \boldsymbol { U } ) ,\tag{15}
$$

where a lower value indicates that the visual sample geometry is more compatible with the question-side language geometry. The full training-free profile of layer l is

$$
\mathcal { P } ( l ) = \{ \mathrm { V D E } _ { \mathrm { p r e } } ( l ) , \mathrm { V D E } _ { \mathrm { p o s t } } ( l ) , \mathrm { G W } _ { \mathrm { p r e } } ( l ) , \mathrm { G W } _ { \mathrm { p o s t } } ( l ) \} .\tag{16}
$$

Profiling protocol. We treat the entries of $\mathcal { P } ( l )$ as candidate layer-selection signals, and evaluate them without any learned or task-specific combination of the four quantities. For a given signal, we rank all candidate layers by that signal, and take the Top-3 as the predicted candidates. Specifically, we rank in descending order for VDE and ascending order for GW, so that the most favorable layers come first. Following the task definition in Section 3.1, we count a hit when the oracle layer $l _ { \mathrm { b e s t } }$ from exhaustive layer-wise inference falls within these Top-3, and report per-signal hit rates. This protocol introduces no tunable scoring parameters and directly measures, for each signal, whether the geometry of a small sample set can narrow the visual-layer search space.

## 4 Experiments

## 4.1 Experimental Setup

Models and datasets. Experiments are conducted on two representative VLMs with different vision backbones: LLaVA-Video-7B-Qwen2 [32] with a 27-layer SigLIP vision tower, and Video-LLaVA-7B-hf [11] with a 24-layer CLIP vision tower. Both models connect the vision encoder to the language model through a multimodal projector and use the penultimate visual layer (-2) as the default feature layer. All candidate visual layers are evaluated using the corresponding frozen pretrained checkpoints.

For image-based evaluation, we use ScienceQA [16], POPE [10], MMStar [5], and CV-Bench [26]. ScienceQA is evaluated on the closed-choice validation subset, POPE on the adversarial split, and CV-Bench on both its 2D and 3D subsets. For video-based evaluation, we use HD-EPIC [19], including Action Recognition and Gaze Estimation. Gaze Estimation is evaluated on the full benchmark, while Action Recognition is evaluated on a randomly sampled subset of 500 examples for efficiency. A summary of the evaluated datasets is provided in Table 2.

Profiling setup. Each dataset is profiled using 100 randomly sampled task examples over five random seeds. For every candidate visual layer, we compute $\mathrm { V D E _ { p r e } }$ and $\mathrm { V D E _ { p o s t } }$ from pre- and post-projector visual representations, respectively. For cross-modal analysis, question-side representations are extracted following Section 3.3, and $\mathrm { G W } ( Z _ { \mathrm { p r e } } , U )$ and $\mathrm { G W } ( Z _ { \mathrm { p o s t } } , U )$ are computed to measure visual-language structural compatibility. The profiling stage is training-free and does not require answer generation or downstream inference, making it substantially cheaper than exhaustive layer-wise evaluation.

Evaluation metrics. Layer-wise profiling signals are compared with exhaustive layer-wise downstream accuracy. For VDE, we compare $\mathrm { V D E _ { p r e } }$ and $\mathrm { V D E _ { p o s t } }$ with exhaustive layer-wise accuracy to examine whether high-entropy regions correspond to high-performing visual layers, and whether the performance-related trends before projection are preserved after the multimodal projector.

Table 2: Datasets used in our experiments.
<table><tr><td>Domain Dataset</td><td></td><td>Samples</td><td>Task Type</td></tr><tr><td>Image</td><td>ScienceQA</td><td>2,004</td><td>Multi-choice VQA</td></tr><tr><td>Image</td><td>POPE (Adv.)</td><td>3,000</td><td>Hallucination Detection</td></tr><tr><td>Image</td><td>CV-Bench 2D</td><td>1,438</td><td>Count &amp; Relation Reasoning</td></tr><tr><td>Image</td><td>CV-Bench 3D</td><td>1,200</td><td>Depth &amp; Distance Reasoning</td></tr><tr><td>Image</td><td>MMStar</td><td>1,500</td><td>General Multimodal Reasoning</td></tr><tr><td>Video</td><td>HD-EPIC (Action)</td><td>500</td><td>Egocentric Action Recognition</td></tr><tr><td>Video</td><td>HD-EPIC (Gaze)</td><td>1,000</td><td>Egocentric Gaze Estimation</td></tr></table>

For cross-modal analysis, we compute $\mathrm { G W } ( Z _ { \mathrm { p r e } } , U )$ and $\mathrm { G W } ( Z _ { \mathrm { p o s t } } , U )$ , where U denotes question-side language representations extracted from the last valid token of the penultimate language-model layer. Lower GW indicates stronger structural compatibility between visual and language sample geometries, allowing us to assess whether cross-modal alignment is associated with stronger visual layers.

Quantitatively, we report the Pearson correlation coefficient (r), the Spearman rank correlation coefficient $( \rho )$ , and the coefficient of determination $( R ^ { 2 } )$ between each layer-wise profile and downstream accuracy. We also report Top-3 hit rates, where a hit is counted if the best-performing layer from exhaustive inference is included in the Top-3 layers ranked by a profiling signal.

The analysis is first conducted on ScienceQA, which supports both VDE and GW analysis, and then extended to additional image and video benchmarks.

## 4.2 Layer-wise Profiling on ScienceQA

ScienceQA provides semantically rich image-question pairs with multiple-choice options, making it particularly suitable for comparing VDE and GW with downstream accuracy within the same setting. Figure 2 reports their layer-wise profiles for both LLaVA-Video and Video-LLaVA.

VDE closely tracks downstream accuracy in both models. For LLaVA-Video, both $\mathrm { V D E _ { p r e } }$ and $\mathrm { V D E _ { p o s t } }$ peak at Layer $2 7 .$ which is also the best-performing layer and differs from the default Layer 26. For Video-LLaVA, the VDE peaks coincide with Layer 23, which is both the default and best-performing layer. Despite using only 100 profiling samples, $\mathrm { V D E _ { p r e } }$ shows strong correlations with accuracy (r = 0.85, ρ = 0.86 $R ^ { 2 } = 0 . 7 3$ for LLaVA-Video; $r = 0 . 6 3 , \rho = 0 . 7 4$ $R ^ { 2 } = 0 . 4 0$ for Video-LLaVA), indicating that sample-level representation diversity captures meaningful layer-wise performance trends before answer generation.

The projector reshapes, but does not erase, the entropy signal. While $\mathrm { V D E _ { p r e } }$ follows a smoother and nearly monotonic depth-wise trend as predicted in $[ 2 4 ] .$ $\mathrm { V D E _ { p o s t } }$ exhibits a less regular profile after multimodal projection. Nevertheless its high-value region remains aligned with the high-performing layers identified by $\mathrm { V D E _ { p r e } }$ and downstream accuracy. This suggests that the projector changes the detailed geometry of visual representations while preserving the performancerelevant layer-wise signal.

![](images/3a7c3a41a42fae49e1dd363b7ca636f9bf4787f4854b6d17c1bd80d8f5ea6e0d.jpg)  
Fig. 2: Layer-wise visual representation profiling on ScienceQA. VDE is estimated from 100 sampled examples over five seeds and compared with exhaustive layer-wise downstream accuracy. GW distances measure the structural compatibility between visual and question-side language representations.

GW provides a complementary alignment diagnostic rather than a reliable selector. Pre-projector GW generally decreases with depth, suggesting stronger structural compatibility between deeper visual representations and question-side language representations. After projection, GW becomes uniformly low across layers, reducing the contrast needed to localize the best visual layer. Thus, GW helps interpret visual-language alignment, whereas VDE provides the more useful signal for candidate-layer profiling.

## 4.3 Generalization Across Image Benchmarks

To evaluate the generality of this profiling behavior, we extend the analysis to four additional image benchmarks covering hallucination detection, spatial reasoning, and general multimodal reasoning. Following the profiling setup in Section 4.1, all image benchmarks use 100 randomly sampled examples over five random seeds. Figure 3 presents the layer-wise $\mathrm { V D E } _ { \mathrm { p r e } } , \ \mathrm { V D E } _ { \mathrm { p o s t } }$ , and downstream accuracy profiles across all image benchmarks.

For Video-LLaVA with the CLIP vision tower, $\mathrm { V D E _ { p r e } }$ increases from shallow to deeper layers and forms a broad plateau around Layers 18–23, followed by a mild decline at the final layer. Although $\mathrm { V D E _ { p o s t } }$ is less regular after projection, its high-value region remains concentrated in the same upper-layer interval. The downstream accuracy curves show a similar pattern: performance becomes relatively stable from around Layer 18, and the strongest layers are located within the high-VDE region. The final-layer drop in both VDE and accuracy further suggests that representation diversity captures layer-wise performance degradation.

![](images/e5c8ccaa59739cd16d0b0106c4803e2e70d3e688ce1509abb5681b1a50c5d1cd.jpg)  
Fig. 3: Layer-wise $\mathrm { V D E } _ { \mathrm { p r e } } , ~ \mathrm { V D E } _ { \mathrm { p o s t } } .$ and downstream accuracy across five image benchmarks for Video-LLaVA (CLIP) and LLaVA-Video (SigLIP). Curves are averaged over five random seeds using 100 samples per seed. The dashed vertical line indicates the default visual layer used by each model.

For LLaVA-Video with the SigLIP vision tower, the high-VDE and highaccuracy regions shift toward the deepest layers. $\mathrm { V D E _ { p r e } }$ increases progressively with depth, while $\mathrm { V D E _ { p o s t } }$ again has a less monotonic profile but peaks in the same late-layer region. The downstream accuracy curves show a stable highperformance region around Layers 22–27, with Layer 27 consistently outperforming the default Layer 26. This indicates that the preferred representation depth depends on the visual backbone, while the association between high VDE and strong downstream performance remains consistent.

Overall, these image-based results show that the relationship between VDE and downstream performance generalizes beyond ScienceQA and remains consistent across different task types and visual backbones. Across different image tasks and visual backbones, VDE identifies compact high-performing layer regions, while the post-projector profiles show that the multimodal projector reshapes but does not remove the performance-related entropy signal.

## 4.4 Extension to Video Tasks

We further examine whether the same profiling behavior extends to video-based VLM tasks. Because video inputs contain multiple frames and are more expensive to profile, we use reduced-budget validation protocols. For HD-EPIC Action Recognition, we perform a full-layer scan using 50 sampled examples with one seed, and additionally compute GW because the answer options provide rich video-text descriptions. For HD-EPIC Gaze Estimation, we use 100 sampled examples over five seeds but restrict the scan to mid-to-high visual layers, following the image results where strong layers concentrate in the upper part of the vision tower. Video-LLaVA uses 8 input frames, while LLaVA-Video uses 16 frames.

![](images/583636f042faaca07ca52f619bc1b328c716721b23beb417ca1bba0a1f7a9de7.jpg)

![](images/34eacbd7c44a8465c1702cda3232b36e38d0fda069df7c86f8f43f798ab85924.jpg)  
Fig. 4: Layer-wise profiling results on HD-EPIC Action Recognition. We perform a full-layer scan using 50 sampled examples with one random seed. The figure compares $\mathrm { V D E } _ { \mathrm { p r e } } , \mathrm { V D E } _ { \mathrm { p o s t } }$ , downstream accuracy, and GW-based cross-modal alignment across visual layers for the evaluated video VLMs.

![](images/cf79242912d0580975961237f25abb036127f8e32f2e975f7c2ca4092e58272e.jpg)

![](images/c8cc253e9994dfc52da5bb58abd870846c7583a6c0a94f49d1874713394ca18c.jpg)  
Fig. 5: Layer-wise profiling results on HD-EPIC Gaze Estimation. We use five random seeds with 100 sampled examples and restrict the scan to the mid-to-high visual layers. The figure compares $\mathrm { V D E } _ { \mathrm { p r e } } , \mathrm { V D E } _ { \mathrm { p o s t } }$ , and downstream accuracy across visual layers for the evaluated video VLMs.

Figure 4 shows that the image-based findings transfer to video action recognition under the lightweight 50-sample setting. For both video VLMs, $\mathrm { V D E _ { p r e } }$ generally increases toward the upper part of the vision tower, while $\mathrm { V D E _ { p o s t } }$ is less regular but still highlights the late-layer region. The downstream accuracy curves follow the same coarse structure: shallow and middle layers are substantially weaker, whereas the strongest layers appear in the upper-layer regions indicated by VDE. For LLaVA-Video, the best-performing layer shifts to around Layer 25, while the default Layer 26 remains competitive. For Video-LLaVA, the strongest layers also lie within the upper-layer plateau, followed by a clear degradation at the final layer. The GW profiles decrease rapidly with depth and stay low in the high-performing region, again supporting their role as an alignment diagnostic rather than a precise selector.

Figure 5 further evaluates VDE on HD-EPIC Gaze Estimation under the fiveseed setting. For LLaVA-Video, $\mathrm { V D E _ { p r e } }$ closely follows downstream accuracy, forming a high-value region around Layers 21–26 that overlaps with the highperformance region. Both VDE and accuracy drop at the final layer, indicating that the entropy profile captures this layer-wise degradation. For Video-LLaVA, the correspondence is weaker but still informative: high VDE concentrates in the mid-to-upper CLIP layers, while both VDE and accuracy decrease at the final layer. Together, the two video tasks show that VDE-based profiling remains practical under reduced search budgets, providing reliable high-performing regions for LLaVA-Video and useful region-level guidance for Video-LLaVA.

## 4.5 Top-3 Candidate Layer Retrieval

Finally, we summarize the practical utility of VDE-based profiling as a candidatelayer retrieval experiment. For each model-task pair, we rank all evaluated visual layers according to either $\mathrm { V D E _ { p r e } }$ or $\mathrm { V D E _ { p o s t } }$ and select the Top-3 layers as candidate layers. We then compare these candidates with the best-performing layer obtained from exhaustive layer-wise downstream inference. A hit is counted if the best-performing layer is included in the Top-3 VDE-ranked candidates. We also report the downstream accuracy of the Top-1 VDE-selected layer against the oracle best accuracy.

Table 3: Top-3 candidate layer retrieval. Hits denote whether the oracle best layer is covered by the Top-3 layers ranked by $\mathrm { V D E _ { p r e } }$ or $\mathrm { V D E } _ { \mathrm { p o s t } } .$ Pre/Post Top-1 vs. Best Acc. compares Top-1 VDE-selected accuracy with oracle best accuracy. Bold entries mark covered oracle layers or exact Top-1 matches.
<table><tr><td>Model</td><td>Task</td><td>VDE-pre Top-3</td><td>VDE-post Top-3</td><td>Pre Top-1/Best Acc. (%)</td><td></td><td>Post Top-1/Best Acc. (%)</td><td></td><td>Best Layer</td><td>Top-3 Hit: Pre/post</td></tr><tr><td rowspan="7"></td><td>ScienceQA</td><td>27, 26, 25</td><td>27, 26, 20</td><td>88.90</td><td>88.90</td><td>88.90</td><td>88.90</td><td>27</td><td>√/√</td></tr><tr><td>POPE-Adv.</td><td>27, 26, 25</td><td>27, 26, 25</td><td>87.00</td><td>87.00</td><td>87.00</td><td>87.00</td><td>27</td><td>√√</td></tr><tr><td>CV-Bench 2D</td><td>27, 26, 25</td><td>27, 26, 25</td><td>70.20</td><td>70.20</td><td>70.20</td><td>70.20</td><td>27</td><td>√√</td></tr><tr><td>LLaVA-Video CV-Bench 3D</td><td>27, 25, 26</td><td>19, 18, 20</td><td>83.17</td><td>83.17</td><td>75.92</td><td>83.17</td><td>27</td><td>√/x</td></tr><tr><td>MMStar</td><td>27, 26, 25</td><td>26, 24, 23</td><td>58.07</td><td>58.07</td><td>57.20</td><td>58.07</td><td>27</td><td>√/x</td></tr><tr><td>HD-EPIC Action</td><td>25, 24, 23</td><td>27, 18, 14</td><td>61.00</td><td>61.00</td><td>58.90</td><td>61.00</td><td>25</td><td>√/x</td></tr><tr><td>HD-EPIC Gaze</td><td>23, 24, 25</td><td>25, 24, 26</td><td>57.70</td><td>57.70</td><td>57.40</td><td>57.70</td><td>23</td><td>√/x</td></tr><tr><td rowspan="7"></td><td>ScienceQA</td><td>23, 24, 22</td><td>23, 21, 22</td><td>59.63</td><td>59.63</td><td>59.63</td><td>59.63</td><td>23</td><td>√/√</td></tr><tr><td>POPE-Adv.</td><td>19, 23, 20</td><td>23, 19, 17</td><td>76.80</td><td>80.53</td><td>79.80</td><td>80.53</td><td>22</td><td>x/x</td></tr><tr><td>CV-Bench 2D</td><td>23, 19, 21</td><td>23, 22, 19</td><td>49.86</td><td>53.20</td><td>49.86</td><td>53.20</td><td>20</td><td>x/x</td></tr><tr><td>Video-LLaVA CV-Bench 3D</td><td>20, 21, 19</td><td>21, 19, 22</td><td>58.08</td><td>58.83</td><td>58.83</td><td>58.83</td><td>21</td><td>√√</td></tr><tr><td>MMStar</td><td>23, 19, 24</td><td>23, 19, 22</td><td>32.40</td><td>33.00</td><td>32.40</td><td>33.00</td><td>21</td><td>x/x</td></tr><tr><td>HD-EPIC Action</td><td>20, 19, 21</td><td>19, 18, 17</td><td>31.50</td><td>33.00</td><td>30.00</td><td>33.00</td><td>22</td><td>x/x</td></tr><tr><td>HD-EPIC Gaze</td><td>19, 20, 18</td><td>19, 20, 21</td><td>26.50</td><td>37.00</td><td>26.50</td><td>37.00</td><td>21</td><td>x/√</td></tr></table>

Table 3 further summarizes the candidate-layer retrieval results. For LLaVA-Video with the SigLIP visual tower, the Top-3 candidates ranked by $\mathrm { V D E _ { p r e } }$ consistently cover the oracle best layer across both image and video tasks. This suggests that the VDE-ranked layers provide a useful approximation to the true high-performing layer region. In particular, the same behavior holds not only on image benchmarks, where the best layer is often the final visual layer, but also on the video Action Recognition task, where the best layer shifts to Layer 25 while still remaining within the high-VDE late-layer region.

For Video-LLaVA with the CLIP visual tower, strict Top-3 retrieval is less stable. The model often exhibits a broad upper-layer plateau, where adjacent layers have similar VDE values and comparable downstream accuracy. As a result, VDE is less reliable as an exact Top-1 or strict Top-3 selector, but still provides useful region-level guidance in several tasks by concentrating candidates around the stronger upper-layer region.

Overall, $\mathrm { V D E _ { p r e } }$ is the strongest practical signal among the tested profiles. It reliably retrieves high-performing layers for the SigLIP-based LLaVA-Video model and offers region-level guidance for the CLIP-based Video-LLaVA model. GW is better interpreted as a diagnostic of cross-modal alignment rather than a dependable stand-alone layer selector. These results support VDE, especially $\mathrm { V D E _ { p r e } }$ , as a lightweight training-free policy for narrowing visual-layer search.

Limitations. First, the experiments focus on two representative VLMs, LLaVA-Video with SigLIP and Video-LLaVA with CLIP. Although the benchmarks cover both image and video tasks, further validation on broader model families, larger vision towers, and different projector designs is needed. Second, VDE is a profiling signal rather than a guaranteed exact layer selector. It reliably retrieves high-performing layers for the SigLIP-based model, but is less stable as a strict Top-3 or Top-1 selector for the CLIP-based model. In such cases, VDE is better viewed as a region-level policy that narrows the search space and should be followed by limited downstream verification. Third, GW distance is only diagnostic in our framework. It provides evidence of visual-language alignment across depth, but does not by itself offer a dependable layer-selection rule.

## 5 Conclusion

This paper studies visual-layer choice in LLaVA-style VLMs from a training-free representation profiling perspective. Instead of treating the late visual layer used by these models as a fixed architectural convention, we analyze how downstream performance changes across vision depth and how this behavior relates to layerwise representation geometry. Across image and video benchmarks, the results show that the default layer is not always optimal, and that high-performing layers often form compact regions that can be identified before answer generation. Using Visual Dataset Entropy, we find that sample-level visual diversity provides a simple and effective signal for profiling candidate layers. The pre-projector entropy profile is especially reliable for the SigLIP-based LLaVA-Video model, while also offering useful region-level guidance for the CLIP-based Video-LLaVA model. The post-projector profile further shows that the multimodal projector reshapes visual geometry but does not fully remove performance-related layerwise trends. In contrast, Gromov-Wasserstein distance provides useful evidence of visual-language alignment across depth, but is better interpreted as a diagnostic rather than a standalone selector.

Overall, these findings suggest that representation geometry can provide insightful analysis and useful guidance for narrowing the visual-layer search space, reducing reliance on exhaustive inference. However, generalizability of entropybased profiling across more diverse visual backbones and to different families of vision-language models remains to be studied more systematically

## Acknowledgements

This work has been supported by the Centre for Spatial Intelligence (RCSI) at University of Bath, the European Union under grant agreement no. 101136006- XTREME, the European Innovation Council under grant agreement no. 1012575- 36-CEREBRIS, the MWK of Lower Saxony within Hybrint (VWZN4219) and LCIS (VWZN4704), the DFG under Germany's Excellence Strategy within the Cluster of Excellence PhoenixD (EXC2122) and Quantum Frontiers (EXC2123), and the German Research Foundation (DFG) as part of the Research Training Group i.c.sens (GRK2159).

## References

1. Alain, G., Bengio, Y.: Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644 (2016)

2. Cao, Y., Liu, Y., Chen, Z., Shi, G., Wang, W., Zhao, D., Lu, T.: Mmfuser: Multimodal multi-layer feature fuser for fine-grained vision-language understanding. arXiv preprint arXiv:2410.11829 (2024)

3. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)

4. Chen, H., Lin, J., Chen, X., Fan, Y., Dong, J., Jin, X., Su, H., Fu, J., Shen, X.: Multimodal language models see better when they look shallower. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp 6677–6695. Association for Computational Linguistics (Nov 2025)

5. Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al.: Are we on the right way for evaluating large visionlanguage models? Advances in Neural Information Processing Systems 37, 27056– 27087 (2024)

6. Cheng, E., Doimo, D., Kervadec, C., Macocco, I., Yu, L., Laio, A., Baroni, M.: Emergence of a high-dimensional abstraction phase in language transformers. In: The Thirteenth International Conference on Learning Representations (2025)

7. Dosovitskiy, A., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (ICLR) (2021)

8. Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., MA, Z., Li, C.: LLaVAneXT-interleave: Tackling multi-image, video, and 3d in large multimodal models. In: The Thirteenth International Conference on Learning Representations (2025)

9. Li, M., Liu, Y., Ma, J., Osborne, E., Han, B., Liu, T.: Rethinking model selection in vlm through the lens of gromov-wasserstein distance. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 17237–17247 (June 2026)

10. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, X., Wen, J.R.: Evaluating object hallucination in large vision-language models. In: Proceedings of the 2023 conference on empirical methods in natural language processing. pp. 292–305 (2023)

11. Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., Yuan, L.: Video-llava: Learning united visual representation by alignment before projection. In: Proceedings of the 2024 conference on empirical methods in natural language processing. pp. 5971– 5984 (2024)

12. Lin, J., Chen, H., Fan, Y., Fan, Y., Jin, X., Su, H., Fu, J., Shen, X.: Multi-layer visual feature fusion in multimodal llms: Methods, analysis, and best practices. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4156–4166 (June 2025)

13. Liu, B., Kamath, A., Grunde-McLaughlin, M., Han, W., Krishna, R.: Visual representations inside the language model. In: Second Conference on Language Modeling (2025)

14. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 26296–26306 (2024)

15. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 34892–34916 (2023)

16. Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.W., Zhu, S.C., Tafjord, O., Clark, P., Kalyan, A.: Learn to explain: Multimodal reasoning via thought chains for science question answering. In: The 36th Conference on Neural Information Processing Systems (NeurIPS) (2022)

17. Marafioti, A., Zohar, O., Farré, M., Noyan, M., Bakouch, E., Cuenca, P., Zakka, C., Allal, L.B., Lozhkov, A., Tazi, N., et al.: Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299 (2025)

18. Mémoli, F.: Gromov-wasserstein distances and the metric approach to object matching. Foundations of computational mathematics 11(4), 417–487 (2011)

19. Perrett, T., Darkhalil, A., Sinha, S., Emara, O., Pollard, S., Parida, K.K., Liu, K., Gatti, P., Bansal, S., Flanagan, K., Chalk, J., Zhu, Z., Guerrier, R., Abdelazim, F., Zhu, B., Moltisanti, D., Wray, M., Doughty, H., Damen, D.: Hd-epic: A highlydetailed egocentric video dataset. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 23901–23913 (June 2025)

20. Peyré, G., Cuturi, M., Solomon, J.: Gromov-wasserstein averaging of kernel and distance matrices. In: International conference on machine learning. pp. 2664–2672. PMLR (2016)

21. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

22. Raghu, M., et al.: Do vision transformers see like convolutional neural networks? In: Advances in Neural Information Processing Systems (NeurIPS) (2021

23. Roy, O., Vetterli, M.: The effective rank: A measure of effective dimensionality. In: 2007 15th European signal processing conference. pp. 606–610. IEEE (2007)

24. Skean, O., Arefin, M.R., Zhao, D., Patel, N.N., Naghiyev, J., Lecun, Y., Shwartz-Ziv, R.: Layer by layer: Uncovering hidden representations in language models. In: International Conference on Machine Learning. pp. 55854–55875. PMLR (2025)

25. Skean, O., Osorio, J.K.H., Brockmeier, A.J., Giraldo, L.G.S.: Dime: Maximizing mutual information by a difference of matrix-based entropies. arXiv preprint arXiv:2301.08164 (2023)

26. Tong, S., Brown, E., Wu, P., Woo, S., Middepogu, M., Akula, S.C., Yang, J., Yang, S., Iyer, A., Pan, X., et al.: Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems 37, 87310– 87356 (2024)

27. Valeriani, L., Doimo, D., Cuturello, F., Laio, A., Ansuini, A., Cazzaniga, A.: The geometry of hidden representations of large transformer models. Advances in Neural Information Processing Systems 36, 51234–51252 (2023)

28. Wei, L., Tan, Z., Li, C., Wang, J., Huang, W.: Diff-erank: A novel rank-based metric for evaluating large language models. Advances in Neural Information Processing Systems 37, 39501–39521 (2024)

29. Wickstrøm, K., Løkse, S., Kampffmeyer, M., Yu, S., Principe, J., Jenssen, R.: Information plane analysis of deep neural networks via matrix-based renyi's entropy and tensor kernels. arXiv preprint arXiv:1909.11396 (2019)

30. Yu, S., Wickstrøm, K., Jenssen, R., Principe, J.C.: Understanding convolutional neural networks with information theory: An initial exploration. IEEE transactions on neural networks and learning systems 32(1), 435–442 (2020)

31. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 11975–11986 (October 2023)

32. Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., Li, C.: Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713 (2024)