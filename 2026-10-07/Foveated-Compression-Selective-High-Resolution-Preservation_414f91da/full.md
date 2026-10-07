# Foveated Compression: Selective High-Resolution Preservation for Token-Efficient VLMs

Donghyun Han<sup>1∗†</sup>, Jangho Park<sup>2∗</sup>, Yuseok Bae<sup>1</sup> <sup>1</sup>ETRI, South Korea <sup>2</sup>Kyung Hee University, South Korea

## Abstract

Visual tokens are a major source of inference cost in vision-language models, yet simple image downsampling remains a surprisingly strong compression baseline. This raises a complementary question: under a fixed token budget, where should visual fidelity be preserved? We introduce Foveated Compression, which encodes a full-resolution image once and represents it with a mixture of nativeand compressed-resolution visual tokens. A behaviorally self-distilled Foveated Merger compresses local visual tokens while preserving compatibility with their native counterparts, and a lightweight Foveated Selector chooses one of nine spatial cells to retain at native resolution using exhaustive budget-matched intervention supervision. At 11.11% visual tokens, uniform Foveated Compression shows no significant paired difference from iso-token downsampling. At 20.99%, the learned selector significantly outperforms random and fixed allocation, but remains below strong whole-image resizing, showing that localized fidelity is not universally preferable. A budget-matched region-choice oracle reaches 82.73 macro accuracy versus 69.61 for the learned selector, revealing substantial headroom within the same spatial action space. Matched probing further shows that signals predicting when compression breaks the answer are substantially more accessible after language-model computation than to the lightweight prefill-free selector. These results expose complementary bottlenecks in region selection and compressed-region fidelity.

## 1 Introduction

Vision-language models (VLMs) increasingly rely on high-resolution visual inputs for fine-grained perception, document understanding, and visually grounded reasoning [2, 28]. Higher resolution, however, produces hundreds or thousands of visual tokens, often far more than the accompanying text, increasing language-model prefill cost and memory use [28, 39]. This has motivated visual-token pruning [23, 4, 39, 35, 6], merging [3, 24, 37], resampling [1], and learned compression [36, 40]. The central challenge is not reducing the sequence, but preserving the visual detail needed by high-resolution tasks.

VTC-Bench [13] complicates this picture by showing that simple image downsampling can match or outperform more elaborate compression methods. Downsampling preserves global image coverage while reducing spatial fidelity uniformly, suggesting that native resolution is rarely needed everywhere. This motivates a different question: rather than only asking how many visual tokens to retain [14], can we decide where a limited fidelity budget should be spent?

Our diagnostic experiments indicate substantial spatial variation in the value of high resolution. Many examples remain correct after aggressive resolution reduction, while others recover only when fine detail is restored. Exhaustive K=1 interventions over a 3 × 3 spatial grid show that the best region varies across examples, though many examples are insensitive to which region is preserved. When region choice does matter, a well-chosen single region can recover substantial performance lost under uniform compression. Pixel-space selective magnification [9, 33, 25, 26] offers one route to such allocation, but may revisit content already represented by the global view, require additional high-resolution visual processing, and fail sharply when an unhelpful region is selected. These observations motivate allocating fidelity directly within a single encoded visual representation.

![](images/54122644ed03e8ca03680a6445b0f72fcc7572b4de85f6122c88289d5dd4a2e7.jpg)  
Figure 1: Foveated Compression preserves one selected region at native visual fidelity while keeping the rest of the image compressed within a single full-resolution encoding. Unlike whole-image downsampling or selective cropping, it maintains global coverage and enables native-token restoration without additional high-resolution visual encoding.

We introduce Foveated Compression, which performs spatial fidelity allocation directly in feature space (Figure 1). A Foveated Merger compresses each 3 × 3 block of native visual tokens into one token and is trained by behavioral self-distillation from the frozen model’s full-resolution predictions. Unlike image downsampling, this feature-space representation retains direct access to the native tokens from the same full-resolution encoding, allowing selected compressed regions to be replaced with their native counterparts without re-encoding the image. Uniform Foveated Compression uses only 11.11% of the native visual-token count while retaining compressed coverage of the entire image.

A lightweight Foveated Selector ranks nine equal spatial cells and preserves the highest-scoring cell at native token resolution under a fixed K=1 budget. Supervision is obtained by exhaustively evaluating all nine budget-matched interventions and training the selector on the resulting winner sets. The selected cell replaces its compressed counterpart while the other eight remain compressed, yielding an exact 20.99% visual-token budget regardless of which region is chosen. The selector requires neither an additional language-model forward pass nor an additional high-resolution vision-encoder pass, and unselected regions remain represented in compressed form.

Our results reveal both the promise and the current limits of spatial fidelity allocation. At 11.11% visual tokens, uniform Foveated Compression matches iso-token downsampling within statistical uncertainty (95% CI [-0.586, 0.521]). At 20.99%, the learned selector significantly outperforms random and training-only fixed-region allocation, showing that useful spatial preferences can be learned under a fixed budget. However, strong whole-image resizing remains better on average, indicating that localized native preservation is not universally preferable to uniform resolution reduction. A budget-matched K=1 region-choice oracle reaches 82.73 macro accuracy compared with 69.61 for the learned selector, exposing substantial selection headroom. Matched probing further shows that signals predicting when compression breaks the answer are substantially more accessible after language-model computation than to the lightweight prefill-free selector. Together, these results point to complementary challenges in region selection and compressed-region fidelity.

Our contributions are:

• We introduce Foveated Compression, a mixed-resolution feature-space representation that combines aggressive global compression with native-resolution preservation in one selected region, while requiring only a single full-resolution visual encoding.

• We introduce a behaviorally self-distilled Foveated Merger that compresses $3 \times 3$ native visual-token blocks into one token while preserving direct compatibility with native tokens, enabling mixed-resolution replacement within the same encoded image.

• We formulate fixed-budget spatial fidelity allocation using exhaustive, budget-matched $K { = } 1$ interventions and tie-aware supervision, enabling controlled comparisons among learned, random, fixed, and oracle region allocation under identical token cost.

## 2 Related Work

## 2.1 Visual Token Compression

Visual-token reduction methods include pruning [23, 4, 39, 10, 35, 6], merging [3, 24, 34, 37], pooling or resampling [1], and learned or adaptive compression [36, 40, 31, 14]. VoCo-LLaMA [36] distills language-model attention into learned compression tokens, FocusLLaVA [40] assigns adaptive local visual scales before instruction-guided token selection, and DyMU [31] adapts token length to image content.

Recent work has also questioned whether increasingly elaborate compression consistently improves visual-token efficiency. VTC-Bench [13] shows that simple image downsampling can match or outperform learned compression methods, while benchmarks such as MMStar [5] emphasize genuine visual dependence. We extend this perspective from global token reduction to spatial fidelity allocation: rather than varying only the number of retained tokens, we use exhaustive, budget-matched interventions to study whether performance depends on where native visual fidelity is preserved.

Foveated Compression differs in its representation, supervision, and evaluation setting. We keep the vision-language backbone frozen, in contrast to end-to-end instruction tuning approaches such as FocusLLaVA [40], and train the Foveated Merger to match the frozen full-resolution model’s output distribution. The Foveated Selector is supervised directly by exhaustive region interventions, while a fixed-K=1 design holds visual-token cost constant across candidate regions and enables controlled comparisons among learned, random, fixed, and oracle allocations. Unlike image downsampling, feature-space compression also preserves direct access to native tokens from the same full-resolution encoding, allowing selected regions to be restored at native fidelity without re-encoding the image.

## 2.2 Selective High-Resolution Processing

A complementary line of work allocates visual fidelity non-uniformly by selectively revisiting or magnifying image regions [26, 38, 32, 11, 29]. V\* [33] performs iterative visual search to locate taskrelevant evidence, while ZoomEye [25] explores progressively magnified regions through hierarchical search. Dynamic tiling methods combine higher-resolution local crops with a coarse global view [15], and TEVA [9] proposes relevant regions before applying dynamic patch sampling at multiple scales.

These approaches demonstrate the value of selectively allocating high-resolution processing, but may require additional region proposal, repeated visual processing, or renewed access to image pixels. Foveated Reasoning [22] makes focusing an explicit action during decoding by re-encoding selected high-resolution evidence and injecting it into the reasoning trajectory. Foveated Compression instead performs allocation before decoding over a single full-resolution visual encoding: selected native tokens replace their compressed counterparts within the same sequence, while all unselected regions remain represented in compressed form.

## 3 Proposed Method

## 3.1 Overview

Foveated Compression represents a full-resolution image using a mixture of native- and compressedresolution visual tokens under a fixed budget. The frozen vision encoder is run once to obtain a native visual-token grid. A Foveated Merger reduces each local $3 \times 3$ block of native visual tokens to one compressed token, yielding 11.11% of the native sequence. Separately, the image is partitioned into a coarse $3 \times 3$ grid of nine spatial cells, and a Foveated Selector chooses exactly one cell to preserve at native resolution. Replacing the compressed units in that cell with their corresponding native-token blocks yields an exact $K { = } \bar { 1 }$ budget of 20.99%.

![](images/a9b636655ebee8633b4e61633297fa53d8ae025b47f7021bf6ceb3c106abdb2f.jpg)  
Figure 2: Overview of Foveated Compression. The image is encoded once by the frozen vision backbone, the Foveated Merger produces a globally compressed representation, and the Foveated Selector restores one selected spatial cell with native visual tokens through in-place replacement.

The vision-language backbone remains frozen throughout training. We first train and freeze the merger, then exhaustively evaluate the nine budget-matched $K { = } 1$ interventions to supervise the selector. Figure 2 summarizes the pipeline.

## 3.2 Foveated Merger

The Foveated Merger operates on the vision-encoder patch grid before Qwen’s frozen native visual merger. The encoder produces patch features $\mathbf { x } _ { i } \in \mathbb { R } ^ { C }$ with $\bar { C } { = } 1 2 8 0$ . For each $3 \times 3$ neighborhood, we combine mean pooling with a learned residual transformation:

$$
\tilde { \mathbf { x } } = \frac { 1 } { 9 } \sum _ { i = 1 } ^ { 9 } \mathbf { x } _ { i } + W _ { \mathrm { h e a d } } \phi ( W _ { \mathrm { f c } } [ \mathrm { R M S N o r m ( \mathbf { x } _ { 1 } ) } ; \dots ; \mathrm { R M S N o r m ( \mathbf { x } _ { 9 } ) } ] + \mathbf { b } _ { \mathrm { f c } } ) + \mathbf { b } _ { \mathrm { h e a d } } ,\tag{1}
$$

where $\phi$ denotes GELU, $W _ { \mathrm { f c } } : \mathbb { R } ^ { 9 C } \to \mathbb { R } ^ { 2 5 6 0 }$ , and $W _ { \mathrm { h e a d } } : \mathbb { R } ^ { 2 5 6 0 }  \mathbb { R } ^ { C }$ . The resulting features pass through Qwen’s frozen $2 \times 2$ visual merger to produce 3584-dimensional compressed tokens. Each compressed token corresponds spatially to a $\phantom { - } 1 3 \times 3$ block of native visual tokens and shares the same downstream interface. We randomly initialize $W _ { \mathrm { f c } }$ and zero-initialize the residual head, so the module starts exactly from mean pooling. The merger has 32.77M trainable parameters.

Behavioral self-distillation. Only the merger is trained; the vision encoder, native visual merger, and language model remain frozen. The native model generates a greedy teacher answer $y _ { 1 : L } .$ , and the compressed model is teacher-forced on the same sequence. Let $p _ { T } ^ { ( \ell ) }$ and $p _ { S } ^ { ( \ell ) }$ denote their next-token distributions at answer position ℓ. We minimize

$$
\mathcal { L } _ { \mathrm { m e r g e } } = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } D _ { \mathrm { K L } } \left( p _ { T } ^ { ( \ell ) } \Vert p _ { S } ^ { ( \ell ) } \right) ,\tag{2}
$$

thereby distilling the frozen model’s full-resolution predictive behavior without ground-truth answer supervision.

## 3.3 Foveated Selector

The Foveated Selector estimates which of the nine coarse cells benefits most from native-resolution preservation. Each compressed unit u corresponds to nine native visual tokens. Let $\mathbf { c } _ { u } \in \mathbb { R } ^ { 3 5 8 4 }$ denote the compressed token, $\mathcal { N } ( u )$ its native-token block, and

$$
\bar { \mathbf { h } } _ { u } = \frac { 1 } { 9 } \sum _ { i \in \mathcal { N } ( u ) } \mathbf { h } _ { i }
$$

their mean. Each unit also has a normalized 2D coordinate $\mathbf { p } _ { u } \in \mathbb { R } ^ { 2 }$

We construct a 512-dimensional visual descriptor

$$
\mathbf { v } _ { u } = P _ { n } \big ( \mathrm { L N } ( \bar { \mathbf { h } } _ { u } ) \big ) + P _ { c } ( \mathrm { L N } ( \mathbf { c } _ { u } ) ) + P _ { d } \big ( \mathrm { L N } ( \bar { \mathbf { h } } _ { u } ) - \mathrm { L N } ( \mathbf { c } _ { u } ) \big ) + P _ { x y } ( \mathbf { p } _ { u } ) ,\tag{3}
$$

where LN denotes LayerNorm, $P _ { n } , P _ { c } , P _ { d } : \mathbb { R } ^ { 3 5 8 4 }  \mathbb { R } ^ { 5 1 2 }$ are learned projections, and $P _ { x y }$ $\mathbb { R } ^ { 2 } \to \mathbb { R } ^ { 5 1 2 }$ projects spatial position. The four terms encode native content, compressed content, compression-induced discrepancy, and location.

Question tokens are obtained from the frozen language-model embedding table, truncated to 48 tokens, projected to 512 dimensions, and encoded by a lightweight self-attention/FFN block. Each visual descriptor then attends to the question through eight-head cross-attention, with visual units as queries. A depthwise $3 \times 3$ convolution adds local spatial context, followed by a scoring MLP that predicts one logit $\ell _ { u }$ per unit. The selector has 10.8M trainable parameters and requires no additional language-model forward pass. Although the architecture permits question-dependent scoring, Section 4.5 finds no measurable gain attributable to question content alone.

Let U<sub>c</sub> denote the compressed units contained in coarse cell c. We score each cell by

$$
s _ { c } = \frac { 1 } { | \mathcal { U } _ { c } | } \sum _ { u \in \mathcal { U } _ { c } } \ell _ { u } , \qquad c = 1 , \ldots , 9 ,
$$

and select the highest-scoring cell. Whole-cell selection matches the intervention granularity and keeps all candidate actions at identical token cost.

Intervention-based supervision. For each training example, we evaluate all nine deterministic $K { = } 1$ actions using the same generation and scoring protocol as evaluation. If $r _ { c }$ is the score obtained by preserving cell $c ,$ the winner set is $\mathcal { W } = \left. \boldsymbol { c } : r _ { c } = \operatorname* { m a x } _ { j } r _ { j } \right.$ . All-nine ties are discarded because they provide no spatial preference, while partial ties are retained rather than collapsed to an arbitrary one-hot target. We minimize

$$
\mathcal { L } _ { \mathrm { s e l e c t o r } } = - \log \frac { \sum _ { c \in \mathcal { W } } \exp ( s _ { c } ) } { \sum _ { j = 1 } ^ { 9 } \exp ( s _ { j } ) } .\tag{4}
$$

The merger and vision-language backbone remain frozen during selector training.

## 3.4 Mixed-Resolution Construction

After selection, each compressed unit inside the chosen cell is replaced by its corresponding $3 \times 3$ block of native visual tokens; all other units remain compressed. Because both representations share the same 3584-dimensional interface, replacement requires neither re-encoding nor an additional projection. Unlike pixel-space downsampling, feature-space compression retains direct access to native tokens from the same full-resolution encoding.

If the native sequence contains N visual tokens, restoring one of nine equal cells gives

$$
N _ { \mathrm { m i x } } = { \frac { N } { 9 } } + { \frac { 8 N } { 8 1 } } = { \frac { 1 7 N } { 8 1 } } ,\tag{5}
$$

or exactly 20.99% of the native token budget.

Native and compressed tokens share the native visual-grid coordinate system for M-RoPE [30]. Native tokens use unit-stride coordinates, while compressed tokens use the top-left coordinate of their corresponding $3 \times 3$ native block. Retaining absolute native-grid coordinates keeps other visual and subsequent text positions invariant across region choices.

## 4 Experiments

## 4.1 Experimental Setup

We use Qwen2.5-VL-7B-Instruct as a frozen backbone. The merger is trained on GQA [8], TextVQA [27], DocVQA [20], InfoVQA [21], ChartQA [19], and ScienceQA [18], then frozen before generating exhaustive K=1 selector labels. We evaluate on GQA, MMBench-en/cn [16], MME [7], POPE [12], MMStar [5], OCRBench [17], and ChartQA using official metrics. All conditions use geometry-matched preprocessing with image dimensions divisible by 252 pixels.

![](images/6fbc6e1b8ffe88c9dbdd57586143db1054a30983c361fea0f917d320ea5fc0bd.jpg)  
(a)

![](images/30380580d284ed1ec59da5236969c5c2e90ea87516118210fa0364ecab444113.jpg)  
(b)

![](images/b80ba1c8064d0d6f889ed6c6e804390be89b3fdc1fc1527e119e427cd4382a8d.jpg)  
(c)  
Figure 3: High-resolution diagnostics. (a) Native-correct retention under uniform compression. (b) Spatial informativeness and winner-region adjacency. (c) Performance headroom from the budgetmatched K=1 region-choice oracle.

Uniform Foveated Compression uses 11.11% of native visual tokens, while K=1 uses exactly 20.99%. We compare against iso-token and matched-budget bicubic downsampling, random and training-only fixed-cell allocation, and 2× whole-image downsampling. Native resolution and a test-time region-choice oracle serve as references. Matched-budget downsampling uses 19.82% tokens in practice due to spatial alignment. All conditions use the same short-answer prompt, greedy decoding, and at most 16 generated tokens. We report the unweighted benchmark macro average; further details are provided in Appendix A.

## 4.2 When and Where Does High Resolution Matter?

Across 37,610 evaluation examples, uniform 9:1 Foveated Compression retains 86.1% of predictions that are correct at native resolution despite using only 11.11% of the visual tokens. This nativecorrect retention ranges from 38.8% on ChartQA to 94.3% on POPE (Figure 3a), showing substantial task-dependent sensitivity to compression.

Exhaustive K=1 interventions show that all nine cells receive the same task score on 78.5% of examples, while the remaining 21.5% are spatially informative (Figure 3b). Among informative cases, 41.9% of winner pairs are horizontally or vertically adjacent, compared with $1 2 / \bar { ( } _ { 2 } ^ { 9 } ) = 1 / 3$ for a random pair, suggesting non-random spatial structure.

A budget-matched region-choice oracle reaches 82.73 macro accuracy at 20.99% tokens, versus 64.62 for uniform compression and 78.85 for the native reference (Figure 3c). Its 3.87-point advantage over native (95% CI [3.388, 4.366]) reflects a test-time maximum over nine interventions and is interpreted only as an upper bound. The fixed K=1 action space nevertheless contains substantial spatial-allocation headroom.

## 4.3 Foveated Compression Results

Table 1 reports the main results on the same 37,610 examples.

At 11.11% visual tokens, Uniform Foveated Compression and Iso-Token Downsample achieve nearly identical macro scores, 64.62 and 64.86, with no significant paired difference (95% CI [−0.586, 0.521]). Their task-level behavior differs: Foveated Compression is stronger on GQA, POPE, and ChartQA, whereas resizing is notably stronger on OCRBench and MME. The merger is therefore competitive with resizing on average while retaining the native-token compatibility required for mixed-resolution replacement.

At the fixed K=1 budget, the selector reaches 69.61, significantly outperforming random allocation at 67.98 by 1.62 points (95% CI [1.085, 2.161]) and training-only fixed allocation at 68.48 by 1.13 points (95% CI [0.610, 1.632]). Random allocation is nontrivial because informative selector-training labels contain 3.66 winning cells on average, corresponding to a 40.65% random winner-membership rate. These results provide direct evidence that useful spatial preferences are learnable.

<table><tr><td>Method</td><td>Tokens</td><td>GQA</td><td>MMB-E</td><td>MMB-C</td><td>MME</td><td>POPE</td><td>MMStar</td><td>OCR</td><td>ChartQA</td><td>Macro</td></tr><tr><td>Native</td><td>100.00%</td><td>60.97</td><td>86.86</td><td>85.61</td><td>2356.7</td><td>87.74</td><td>60.93</td><td>80.20</td><td>84.28</td><td>78.85</td></tr><tr><td>Uniform Foveated Compression</td><td>11.11%</td><td>58.28</td><td>82.51</td><td>82.01</td><td>2130.5</td><td>84.34</td><td>50.20</td><td>49.30</td><td>34.24</td><td>64.62</td></tr><tr><td>Iso-Token Downsample</td><td>11.11%</td><td>55.88</td><td>82.77</td><td>81.75</td><td>2240.8</td><td>82.69</td><td>51.07</td><td>57.30</td><td>27.44</td><td>64.86</td></tr><tr><td>Random K=1</td><td>20.99%</td><td>58.40</td><td>84.18</td><td>82.86</td><td>2190.7</td><td>84.34</td><td>52.33</td><td>55.10</td><td>48.36</td><td>67.98</td></tr><tr><td>Fixed K=1</td><td>20.99%</td><td>58.90</td><td>83.69</td><td>82.70</td><td>2170.3</td><td>84.54</td><td>51.87</td><td>53.70</td><td>54.92</td><td>68.48</td></tr><tr><td>Foveated Selector</td><td>20.99%</td><td>58.24</td><td>84.59</td><td>82.67</td><td>2186.7</td><td>84.92</td><td>52.13</td><td>58.40</td><td>57.84</td><td>69.61</td></tr><tr><td>Matched-Budget Downsample</td><td>19.82%</td><td>58.05</td><td>84.45</td><td>83.34</td><td>2303.0</td><td>84.77</td><td>54.53</td><td>67.00</td><td>53.48</td><td>70.99</td></tr><tr><td>2× Downsample</td><td>24.57%</td><td>58.82</td><td>85.12</td><td>84.15</td><td>2298.7</td><td>84.90</td><td>56.00</td><td>70.50</td><td>57.84</td><td>72.43</td></tr><tr><td>K=1 Region-Choice Oracle</td><td>20.99%</td><td>70.42</td><td>92.22</td><td>90.85</td><td>2500.8</td><td>91.82</td><td>70.67</td><td>77.60</td><td>78.92</td><td>82.73</td></tr></table>

Table 1: Main results on 37,610 examples across eight benchmarks. Bold and underlined values denote the best and second-best methods within the approximately 20% fixed-budget comparison group. MME is divided by 28 only for the macro average.

Strong whole-image resizing nevertheless remains better overall. Matched-Budget Downsample reaches 70.99 at 19.82% tokens, outperforming the selector by 1.38 points (95% CI [0.707, 2.062]), while 2× Downsample reaches 72.43 at 24.57%, a 2.82-point advantage (95% CI [2.164, 3.438]). The comparison is task dependent: the selector exceeds matched-budget downsampling on ChartQA (57.84 vs. 53.48), whereas resizing has a large advantage on OCRBench (67.00 vs. 58.40) and MME. Thus, localized fidelity can help when relevant evidence is concentrated, but is not universally preferable.

The non-deployable K=1 oracle reaches 82.73 at the same 20.99% budget, exceeding the learned selector by 13.12 points (95% CI [12.566, 13.691]). Although it selects using observed test outcomes, the large gap shows that substantially better region choices exist within the same action space. Full paired comparisons are provided in Appendix C.

## 4.4 Fidelity Allocation Analysis

The oracle gap indicates substantial remaining selection headroom. To examine what information is accessible before language-model computation, we construct a matched binary task on 3,416 examples, with positives defined as examples where native inference is correct but uniform compression is incorrect. This predicts whether compression fails rather than which region should be selected, and therefore is not a direct measure of nine-way selection quality.

The raw prefill-free selector signal achieves an AUC of 0.5239 (95% CI [0.4796, 0.5700]), compared with 0.7651 (95% CI [0.7347, 0.7937]) for first-token decoder entropy on the same examples and target. A same-fold five-fold out-of-fold comparison yields 0.5476 (95% CI [0.5012, 0.5929]) versus 0.8072 (95% CI [0.7738, 0.8391]). Compression-sensitive information is therefore weakly accessible before prefill but substantially more predictable after language-model computation. Decoder-side signals arise only after generation and cannot be directly used by our single-pass selector. Further details are provided in Appendix D.

Region prediction is not the only limitation. On OCRBench, Foveated Compression trails downsampling at both 11.11% tokens (49.30 vs. 57.30) and approximately 20% tokens (58.40 vs. 67.00), whereas the selector exceeds Matched-Budget Downsample on ChartQA (57.84 vs. 53.48). These contrasts expose complementary bottlenecks in region prediction and compressed-region fidelity. Additional qualitative examples are provided in Appendix E.

## 4.5 Ablations

Table 2 evaluates the main design choices using matched runs for controlled comparisons.

Behavioral training improves mean pooling by 5.57 macro points (95% CI [5.00, 6.20]), with gains across all eight benchmarks. Tie-aware supervision improves over arbitrary single-winner targets by 1.24 points (95% CI [0.768, 1.734]), supporting set-valued supervision when multiple regions are equally optimal.

Question content does not provide a measurable gain: removing it yields 69.80 versus 69.77 for the full selector, with a paired difference of −0.03 points (95% CI [−0.238, 0.180]). Removing both question and explicit spatial inputs yields 69.65, although this also changes model capacity. Together with Section 4.4, these results suggest that the limitation extends beyond question content alone to information accessible before language-model computation.

<table><tr><td>Variant</td><td>Tokens</td><td>GQA</td><td>MMB-E</td><td>MMB-C</td><td>MME</td><td>POPE</td><td>MMStar</td><td>OCR</td><td>ChartQA</td><td>Macro</td></tr><tr><td>Mean-Pooling Compression</td><td>11.11%</td><td>51.76</td><td>79.65</td><td>78.33</td><td>2013.4</td><td>80.47</td><td>44.73</td><td>41.40</td><td>24.12</td><td>59.05</td></tr><tr><td>Behaviorally Trained Merger</td><td>11.11%</td><td>58.28</td><td>82.51</td><td>82.01</td><td>2130.5</td><td>84.34</td><td>50.20</td><td>49.30</td><td>34.24</td><td>64.62</td></tr><tr><td>Single-Winner Supervision</td><td>20.99%</td><td>57.47</td><td>83.51</td><td>81.75</td><td>2203.5</td><td>83.80</td><td>52.47</td><td>56.20</td><td>54.40</td><td>68.54</td></tr><tr><td>Tie-Aware Supervision</td><td>20.99%</td><td>58.38</td><td>85.08</td><td>83.04</td><td>2189.5</td><td>84.86</td><td>52.40</td><td>59.20</td><td>57.04</td><td>69.77</td></tr><tr><td>Question Content Removed</td><td>20.99%</td><td>58.42</td><td>84.59</td><td>82.65</td><td>2205.7</td><td>85.01</td><td>53.00</td><td>58.90</td><td>57.08</td><td>69.80</td></tr><tr><td>Full Selector</td><td>20.99%</td><td>58.38</td><td>85.08</td><td>83.04</td><td>2189.5</td><td>84.86</td><td>52.40</td><td>59.20</td><td>57.04</td><td>69.77</td></tr><tr><td>Question and Spatial Inputs Removed</td><td>20.99%</td><td>58.20</td><td>84.50</td><td>82.95</td><td>2195.8</td><td>84.77</td><td>53.07</td><td>58.10</td><td>57.20</td><td>69.65</td></tr></table>

Table 2: Per-benchmark ablations. The merger block uses the 11.11% uniform setting; selector ablations use a matched full-selector rerun (69.77 macro), distinct from the 69.61 main checkpoint. MME is divided by 28 only for the macro average. The Tie-Aware Supervision and Full Selector rows report the same run, listed once in each ablation block.

## 5 Limitations

The current selector remains below strong whole-image resizing, and matched probing indicates that compression-sensitive signals are only weakly accessible before language-model computation. Moreover, preserving one native-resolution cell cannot recover information lost across the remaining compressed regions. Our token ratios measure visual tokens entering the language model, not end-to-end compute: Foveated Compression runs the full-resolution vision encoder once, whereas pixel-space downsampling also reduces vision-encoder cost. The comparison with downsampling is therefore favorable to the baseline in wall-clock terms, and we do not claim an end-to-end efficiency advantage. We evaluate a single frozen VLM backbone and a fixed K=1 budget; broader backbones and adaptive budgets remain future work.

## 6 Conclusion

We introduced Foveated Compression, a fixed-budget mixed-resolution representation that combines global compression with native-resolution preservation from a single full-resolution encoding. Learned spatial allocation significantly improves over random and fixed selection, although strong whole-image resizing remains better overall. A budget-matched oracle reaches 82.73 versus 69.61 for the learned selector, revealing substantial selection headroom, while matched probing shows that compression-failure signals become substantially more accessible after language-model computation. Future progress will require both stronger compressed representations and richer, efficient selection signals.

## 7 Acknowledgments

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2022-II220124, Development of Artificial Intelligence Technology for Self Improving Competency-Aware Learning Capabilities) and Electronics and Telecommunications Research Institute(ETRI) grant funded by the Korean government (26CB1200, Development and Application of Science-Specialized Multimodal Foundation Models).

## References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a Visual Language Model for Few-Shot Learning. 2022.

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025.

[3] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token Merging: Your ViT but Faster. In Int. Conf. Learn. Represent., 2023.

[4] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models. In Eur. Conf. Comput. Vis., 2024.

[5] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? In Adv. Neural Inform. Process. Syst., 2024.

[6] Mark Endo, Xiaohan Wang, and Serena Yeung-Levy. Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration. In Int. Conf. Comput. Vis., October 2025.

[7] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv preprint arXiv:2306.13394, 2023.

[8] Drew A. Hudson and Christopher D. Manning. GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. In IEEE Conf. Comput. Vis. Pattern Recog., June 2019.

[9] Yitong Jiang, Jinwei Gu, Tianfan Xue, Ka Chun Cheung, Pavlo Molchanov, Hongxu Yin, and Sifei Liu. Token-Efficient VLM: High-Resolution Image Understanding via Dynamic Region Proposal. In Int. Conf. Comput. Vis., pages 24147–24158, October 2025.

[10] Jaehoon Lee, Mingi Jung, Soohyuk Jang, Seungryong Yoo, Dahuin Jung, and Sungroh Yoon. Balancing Saliency and Coverage: Semantic Prominence-Aware Budgeting for Visual Token Compression in VLMs. 2026.

[11] Jewon Lee, Wooksu Shin, Seungmin Yang, Ki-Ung Song, DongUk Lim, Jaeyeon Kim, Tae-Ho Kim, and Bo-Kyeong Kim. ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models. In Int. Conf. Learn. Represent., 2026.

[12] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating Object Hallucination in Large Vision-Language Models. In Conf. Empir. Methods Nat. Lang. Process., pages 292–305, Dec 2023.

[13] Chenfei Liao, Wensong Wang, Zichen Wen, Xu Zheng, Yiyu Wang, Haocong He, Yuanhuiyi Lyu, Lutao Jiang, Xin Zou, Yuqian Fu, Bin Ren, Linfeng Zhang, and Xuming Hu. Are We Using the Right Benchmark: An Evaluation Framework for Visual Token Compression Methods. In Annu. Meeting Assoc. Comput. Linguistics, pages 4236–4253, Jul 2026.

[14] Zichuan Lin, Yicheng Liu, Yang Yang, Lvfang Tao, and Deheng Ye. AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition. In IEEE Conf. Comput. Vis. Pattern Recog., June 2026.

[15] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge, January 2024. URL https://llava-vl. github.io/blog/2024-01-30-llava-next/.

[16] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. MMBench: Is Your Multi-modal Model an All-around Player? In Eur. Conf. Comput. Vis., pages 216––233, 2024.

[17] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. OCRBench: on the hidden mystery of OCR in large multimodal models. Science China Information Sciences, 67(12), Dec 2024.

[18] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. In Adv. Neural Inform. Process. Syst., volume 35, pages 2507–2521, 2022.

[19] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In Findings Assoc. Comput. Linguistics, pages 2263–2279, May 2022.

[20] Minesh Mathew, Dimosthenis Karatzas, and C.V. Jawahar. DocVQA: A Dataset for VQA on Document Images. In IEEE Winter Conf. Appl. Comput. Vis., pages 2200–2209, January 2021.

[21] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. InfographicVQA. In IEEE Winter Conf. Appl. Comput. Vis., pages 2582–2591, January 2022.

[22] Juhong Min, Lazar Valkov, Vitali Petsiuk, Hossein Souri, and Deen Dayal Mohan. Foveated Reasoning: Stateful, Action-based Visual Focusing for Vision-Language Models. 2026.

[23] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. DynamicViT: efficient vision transformers with dynamic token sparsification. In Adv. Neural Inform. Process. Syst., 2021.

[24] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models. In Int. Conf. Comput. Vis., 2025.

[25] Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, Ruochen Xu, Zilun Zhang, Mingwei Zhu, and Jianwei Yin. ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration. In Conf. Empir. Methods Nat. Lang. Process., pages 6602–6618, Nov 2025.

[26] Yuheng Shi, Xiaohuan Pei, Linfeng Wen, Minjing Dong, and Chang Xu. Q-Zoom: Query-Aware Adaptive Perception for Efficient Multimodal Large Language Models, 2026.

[27] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA Models That Can Read. In IEEE Conf. Comput. Vis. Pattern Recog., June 2019.

[28] Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokula Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, and Hadi Pouransari. Fastvlm: Efficient vision encoding for vision language models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 19769–19780, June 2025.

[29] Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning. In Adv. Neural Inform. Process. Syst., 2025.

[30] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution, 2024.

[31] Zhenhailong Wang, Senthil Purushwalkam, Caiming Xiong, Silvio Savarese, Heng Ji, and Ran Xu. DyMU: Dynamic Merging and Virtual Unmerging for Efficient Variable-Length VLMs. In Adv. Neural Inform. Process. Syst., volume 38, Main Conference, pages 73020–73043, 2025.

[32] Lai Wei, Liangbo He, Jun Lan, Lingzhong Dong, Yutong Cai, Siyuan Li, Huijia Zhu, Weiqiang Wang, Linghe Kong, Yue Wang, Zhuosheng Zhang, and Weiran Huang. Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception. In Int. Conf. Mach. Learn., 2026.

[33] Penghao Wu and Saining Xie. V\*: Guided Visual Search as a Core Mechanism in Multimodal LLMs. In IEEE Conf. Comput. Vis. Pattern Recog., pages 13084–13094, June 2024.

[34] Longrong Yang, Dong Shen, Chaoxiang Cai, Kaibing Chen, Fan Yang, Tingting Gao, Di Zhang, and Xi Li. Libra-Merging: Importance-Redundancy and Pruning-Merging Trade-Off for Acceleration Plug-In in Large Vision-Language Model. In IEEE Conf. Comput. Vis. Pattern Recog., pages 9402–9412, 2025.

[35] Senqiao Yang, Junyi Li, Xin Lai, Jinming Wu, Wei Li, Zejun Ma, Bei Yu, Hengshuang Zhao, and Jiaya Jia. VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning. In Adv. Neural Inform. Process. Syst., 2025.

[36] Xubing Ye, Yukang Gan, Xiaoke Huang, Yixiao Ge, and Yansong Tang. VoCo-LLaMA: Towards Vision Compression with Large Language Models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 29836– 29846, June 2025.

[37] Hanxun Yu, Wentong Li, Xuan Qu, Song Wang, Junbo Chen, and Jianke Zhu. VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration. In Int. Conf. Learn. Represent., 2026.

[38] Qianhao Yuan, Jie Lou, Xing Yu, Hongyu Lin, Le Sun, Xianpei Han, and Yaojie Lu. Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation, 2026.

[39] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis A Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In Int. Conf. Mach. Learn., volume 267, pages 74840–74857, 2025.

[40] Yuke Zhu, Chi Xie, Shuang Liang, Bo Zheng, and Sheng Guo. FocusLLaVA: A Coarse-to-Fine Approach for Efficient and Effective Visual Token Compression. 2024.

## A Implementation and Evaluation Details

## A.1 Training Details

We train the Foveated Merger and Foveated Selector sequentially while keeping the Qwen2.5-VL-7B-Instruct backbone frozen. Table A1 summarizes the main optimization settings. The merger is trained for one epoch with 8-GPU distributed data parallelism, corresponding to $^ { 2 , 3 7 8 }$ optimization steps, and uses gradient accumulation of four. We use AdamW with a learning rate of $1 \dot { \times } 1 0 ^ { - 4 }$ , weight decay $1 \times \bar { 1 0 } ^ { - 2 }$ , 200 warmup steps followed by a constant learning rate, and gradient clipping at 1.0. The training visual-token cap is 1,600 tokens, reduced to 600 for GQA. Teacher predictions are cached for 79,723 examples using the top-256 output probabilities. Main training and full-split evaluation runs were performed on 8× NVIDIA RTX A5000 24GB GPUs. Training the Foveated Merger took approximately 6.9 GPU-hours. Generating exhaustive K=1 selector labels required executing all nine deterministic K=1 actions per training example and took approximately 18.1 GPU-hours in total.

The selector is trained only after the merger has been frozen. It is trained for four epochs with a learning rate of $2 \times 1 0 ^ { - 4 }$ and 100 warmup steps. Training and validation partitions are formed at the image-group level so that multiple questions associated with the same image do not cross splits. Question inputs are truncated to at most 48 tokens. All-nine-tie examples are excluded from selector training, while partial ties are retained through the set-valued objective in Equation (4).

## A.2 Evaluation Protocol

All reported conditions, including the native reference, use the same generation protocol. We append the common instruction

## Answer the question using a single word or phrase.

to the benchmark question and decode greedily with sampling disabled, a single beam, and at most 16 generated tokens. Benchmark predictions are scored using each benchmark’s official evaluation metric. MME is reported on its native 0–2800 scale in per-benchmark tables and divided by 28 only when computing the unweighted benchmark macro average.

The native reference uses the same 252-aligned source geometry as all Foveated Compression conditions. This differs from allowing each condition to independently choose its native preprocessing size and ensures that comparisons begin from the same visual input. Downsampling baselines use bicubic image resizing before visual encoding.

## A.3 Geometry and Token Accounting

Image dimensions are chosen as multiples of 252 pixels. This alignment follows

$$
2 5 2 = 1 4 \times 2 \times 3 \times 3 ,
$$

corresponding to the 14-pixel vision patch size, Qwen’s native $2 \times 2$ visual merge, the Foveated Merger’s $3 \times { \bar { 3 } }$ reduction, and the coarse $3 \times 3$ selection grid. Consequently, compression blocks and selection-cell boundaries align exactly.

Table A2 summarizes the visual-token budgets. Uniform Foveated Compression uses exactly $1 / 9 = { }$ 11.11% of the native visual tokens. Restoring one of nine cells gives $\mathrm { i 7 / 8 1 } = 2 0 . 9 9 \%$ as derived in Equation (5). The matched-budget downsampling baseline targets the same approximately 21% budget but uses 19.82% in practice because image dimensions are quantized to the model’s spatial alignment. Similarly, halving both image dimensions would nominally produce a 25% token budget, while the aligned 2× Downsample condition uses 24.57%.

These ratios measure the number of visual tokens entering the language model. They should not be interpreted as end-to-end compute ratios: Foveated Compression executes the full-resolution vision encoder once, whereas pixel-space downsampling also reduces vision-encoder computation.

## B Selector Supervision Statistics

Selector supervision is generated only after freezing the trained Foveated Merger. For each training example, we execute all nine deterministic $K { = } 1$ actions and record the set of cells attaining the maximum task score. Examples for which all nine cells tie are excluded because they contain no spatial preference; all other tied winners are retained.

<table><tr><td>Setting</td><td>Merger</td><td>Selector</td></tr><tr><td>Trainable parameters</td><td> $3 2 . 7 7 \mathrm { M }$ </td><td>10.8M</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Epochs</td><td>1</td><td>4</td></tr><tr><td>Warmup steps</td><td>200</td><td>100</td></tr><tr><td>Optimization steps</td><td>2,378</td><td></td></tr><tr><td>Gradient accumulation</td><td>4</td><td></td></tr><tr><td>Gradient clipping</td><td>1.0</td><td></td></tr><tr><td>Question length</td><td>一</td><td>48</td></tr></table>

Table A1: Main training settings for the Foveated Merger and Foveated Selector. Settings not shared across the two modules are marked with “–”.

<table><tr><td>Condition</td><td>Visual-token ratio</td></tr><tr><td>Native</td><td>100.00%</td></tr><tr><td>Uniform Foveated Compression</td><td>11.11%</td></tr><tr><td>Iso-Token Downsample</td><td>11.11%</td></tr><tr><td>Random / Fixed / Learned  $K { = } 1$ </td><td>20.99%</td></tr><tr><td>Matched-Budget Downsample</td><td>19.82%</td></tr><tr><td>2× Downsample</td><td>24.57%</td></tr><tr><td>K=1 Region-Choice Oracle</td><td>20.99%</td></tr></table>

Table A2: Visual-token budgets relative to the 252-aligned native input.

Table A3 summarizes the resulting supervision. Among 18,048 generated labels, 7,678 examples (42.54%) are spatially informative for selector training. Informative examples contain 3.66 tied winners on average. Consequently, a uniformly random cell belongs to the winner set in 40.65% of informative training examples, making random region selection a nontrivial baseline. The best single fixed cell chosen from training labels obtains 45.40% winner membership, while the held-out learned selector reaches 55.34%.

These statistics should not be confused with the 21.5% spatially informative rate in Section 4.2. The latter is measured over the full 37,610-example evaluation set, whereas Table A3 characterizes the separate data used to construct selector supervision.

## C Additional Diagnostic and Statistical Results

## C.1 Resolution and Spatial-Choice Diagnostics

Table A4 collects the aggregate diagnostics summarized in Section 4.2. Uniform Foveated Compression retains 86.1% of predictions that are correct at native resolution, but sensitivity varies substantially by task: native-correct retention ranges from 38.8% on ChartQA to 94.3% on POPE.

Across exhaustive K=1 interventions, all nine regions obtain the same task score for 78.5% of evaluation examples. The remaining 21.5% are spatially informative, meaning that at least two region choices produce different outcomes. For informative examples with winner sets containing between two and eight cells, 41.9% of winner pairs are horizontally or vertically adjacent. A uniformly random pair on a $3 \times 3$ grid is adjacent with probability $1 2 / { \binom { 9 } { 2 } } = 1 / 3$

The adjacency statistic is used as descriptive evidence of spatial structure rather than as a formal significance test.

## C.2 Paired Bootstrap Comparisons

We quantify uncertainty using 2,000 paired bootstrap resamples. Evaluation rows are resampled within benchmark so that predictions from the compared methods remain paired, and the benchmark macro average is recomputed for each resample using the same MME normalization as in the main results. Table A5 reports the principal comparisons.

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Total selector labels Spatially informative labels</td><td>18,048 7,678</td></tr><tr><td>Informative fraction Mean winner-set size</td><td>42.54% 3.66</td></tr><tr><td>Random winner membership Best training-only fixed cell</td><td>40.65% 45.40%</td></tr></table>

Table A3: Statistics of the intervention-derived selector supervision. Winner membership measures whether the selected cell belongs to the set of equally optimal K=1 actions.

<table><tr><td>Diagnostic</td><td>Value</td></tr><tr><td>Evaluation examples</td><td>37,610</td></tr><tr><td>Overall native-correct retention</td><td>86.1%</td></tr><tr><td>Lowest retention (ChartQA)</td><td>38.8%</td></tr><tr><td>Highest retention (POPE)</td><td>94.3%</td></tr><tr><td>All-nine-tie examples</td><td>78.5%</td></tr><tr><td>Spatially informative examples</td><td>21.5%</td></tr><tr><td>Adjacent winner pairs Random-pair adjacency</td><td>41.9% 33.3%</td></tr></table>

Table A4: Aggregate resolution-sensitivity and spatial-choice diagnostics.

Uniform Foveated Compression and Iso-Token Downsample show no significant paired difference at the 11.11% budget; we do not claim equivalence. At the K=1 operating point, the learned selector significantly improves over random and fixed allocation but remains below both matched-budget and 2× whole-image resizing. The large oracle gap therefore reflects unrealized region-choice headroom rather than an advantage achieved by the current selector.

## D Matched Compression-Failure Analysis

Section 4.4 compares the accessibility of compression-sensitive signals before and after languagemodel computation. To make this comparison directly interpretable, both feature families are evaluated on the same examples and the same binary target.

The matched set contains 3,416 examples, comprising 488 examples from each of seven benchmarks. We define

$$
y _ { \mathrm { r e s c u e } } = k ^ { 2 } \left[ \mathrm { n a t i v e ~ c o r r e c t } \wedge \mathrm { u n i f o r m ~ c o m p r e s s i o n ~ i n c o r r e c t } \right] .\tag{6}
$$

There are 178 positive examples, corresponding to 5.21% of the matched set. The target therefore measures whether aggressive compression changes a correct native prediction into an incorrect one. It does not identify which of the nine K=1 regions should be selected.

For the primary raw comparison, the prefill-free signal is the image-level mean of the selector-side keep logits, while the decoder-side signal is the first-token entropy from generation under uniform compression. Both are evaluated on the same rows. We additionally train same-fold five-fold out-offold probes for the prefill-free and decoder-side feature families. Confidence intervals and paired differences are estimated with 2,000 benchmark-stratified paired bootstrap resamples.

The raw decoder signal exceeds the raw prefill-free signal by 0.2411 AUC (95% CI [0.1845, 0.3015]). The out-of-fold comparison gives a similar advantage of 0.2596 (95% CI [0.2034, 0.3132]). Across individual decoder statistics, negative minimum-token log-probability achieves the highest observed AUC, 0.8105, whereas raw prefill-free selector statistics range from 0.4782 to 0.5239.

These results should not be interpreted as showing that decoder uncertainty directly solves region selection. First, the binary target asks whether native fidelity is needed rather than where it should be allocated. Second, decoder uncertainty becomes available only after language-model generation has begun and therefore cannot be used directly by the single-pass prefill-free selector studied in this work. The matched comparison instead shows that information associated with compression-induced failure is substantially more accessible after language-model computation.

<table><tr><td>Comparison</td><td>∆ Macro</td><td>95% CI</td></tr><tr><td>Uniform Foveated Compression — Iso-Token Downsample</td><td>-0.24</td><td>[-0.586, 0.521]</td></tr><tr><td>Foveated Selector — Native</td><td>-9.238</td><td>[-9.867, -8.625]</td></tr><tr><td>Foveated Selector — Random K=1</td><td>+1.622</td><td>[1.085, 2.161]</td></tr><tr><td>Foveated Selector — Fixed K=1</td><td>+1.131</td><td>[0.610, 1.632]</td></tr><tr><td>Foveated Selector — Matched-Budget Downsample</td><td>-1.378</td><td>[-2.062, -0.707]</td></tr><tr><td>2× Downsample — Foveated Selector</td><td>+2.802</td><td>[2.164, 3.438]</td></tr><tr><td>Oracle — Native</td><td>+3.873</td><td>[3.388, 4.366]</td></tr><tr><td>Oracle — Foveated Selector</td><td>+13.124</td><td>[12.566, 13.691]</td></tr><tr><td>Oracle - Random K=1</td><td>+14.753</td><td>[14.173, 15.350]</td></tr><tr><td>Oracle - Fixed K=1</td><td>+14.242</td><td>[13.688, 14.814]</td></tr><tr><td>Oracle — Matched-Budget Downsample</td><td>+11.747</td><td>[11.143, 12.350]</td></tr><tr><td>Behaviorally Trained Merger — Mean Pooling</td><td>+5.57</td><td>[5.00, 6.20]</td></tr><tr><td>Tie-Aware Supervision — Single-Winner Supervision</td><td>+1.236</td><td>[0.768, 1.734]</td></tr><tr><td>Full Selector — Question Content Removed</td><td>-0.031</td><td>[-0.238, 0.180]</td></tr></table>

Table A5: Paired bootstrap comparisons. Positive values favor the first method named in each comparison. The oracle is a non-deployable test-time upper bound.

<table><tr><td>Signal</td><td>AUC</td><td>95% CI</td></tr><tr><td>Prefill-free raw signal</td><td>0.5239</td><td>[0.4796, 0.5700]</td></tr><tr><td>First-token entropy</td><td>0.7651</td><td>[0.7347, 0.7937]</td></tr><tr><td>Prefill-free OOF probe</td><td>0.5476</td><td>[0.5012, 0.5929]</td></tr><tr><td>Decoder-side OOF probe</td><td>0.8072</td><td>[0.7738, 0.8391]</td></tr></table>

Table A6: Matched AUC comparison for predicting compression-induced answer failure.

## E Additional Qualitative Examples

We provide additional qualitative examples to complement the aggregate analysis in Section 4.4. Figure A1 presents nine examples organized into three outcome regimes: compression-robust (Figure A1a), selection-rescued (Figure A1b), and selection-miss (Figure A1c). Each regime contains one example from each of three representative visual reasoning categories—scene-text reading, object existence, and compositional reasoning—illustrating that these outcome patterns are not limited to a single type of visual evidence.

For each example, we show the question and ground-truth answer together with predictions from whole-image downsampling, the learned Foveated Selector, and the best K=1 oracle action. We additionally visualize the regions selected by the learned selector and the oracle when they differ. The oracle is included only as a diagnostic upper bound that identifies whether a successful action exists within the same fixed-budget K=1 action space; it should not be interpreted as a deployable prediction method.

In the compression-robust cases (Figure A1a), aggressive resolution reduction does not change the task outcome: whole-image downsampling, the learned selector, and the oracle all produce successful predictions. These examples are consistent with our aggregate observation that many inputs are insensitive to the precise allocation of native-resolution fidelity. In such cases, allocating additional native-resolution tokens to a particular region provides little benefit because the evidence required for the prediction remains sufficiently accessible under global resolution reduction.

In the selection-rescued cases (Figure A1b), whole-image downsampling fails, whereas the learned selector identifies a region whose native-resolution preservation enables the mixed-resolution representation to recover the correct answer. The examples span scene-text reading, object existence,

Question : "What does the man's hat say on it?' Ground Truth : canada

![](images/26604ec2729256c6ede0bc9dde15f2aba0e2fae46f6bd5c767257756e41052e2.jpg)  
0.5 Downsampling : “canada" Selector : "canada" Oracle: "canada"  
Ouestion : "What is the social media site listed at the bottom ol the white banner?"

![](images/598ff6d503a25a64061b3e471362198b4c35e3d3b6987963f74f4aa3bf06acb7.jpg)  
X0.5 Downsampling : "Facebook" Selector : "twitter Oracle: "twitter"  
Question : "Who is the author of the broken window?" Ground Truth : Jeffery deaver

![](images/8001860707eb7c6838939c2874fd74e9365c4e660000496b40874d0114977788.jpg)  
X0.5 Downsampling : “Jeffrey deaver" XSelector : "Jeffrery deaver" Oracle: “Jeffery deaver"

Question : "Is there a bowl in the image?" Ground Truth : no  
![](images/342e1cf1be8da20cae51772475a0a0d019596d39bb364b62a60a4ab598e6536a.jpg)  
Question : "Is there a person in the image?" Ground Truth : yes  
Question : "Is there a clock in the image?" Ground Truth : yes

![](images/61c628332fbc48a978a3ef4b22807a92f9b0852e0ab08c09568af9b62f64cd1b.jpg)

![](images/84c44505dda47730a01bec7811b2664007cf8cf4de01eb3d6a954b26a6439db6.jpg)  
0.5 Downsampling : "no" Selector : "no" Oracle: “no" Question : "What is hanging from the tree in front of the wall?" Ground Truth : ornaments

![](images/82ddb3a0cb305462ed6fd6913c5d58af590280ca090acce14498dfff3aa6ffe2.jpg)  
X0.5 Downsampling : “no" Selector : "yes" Oracle: "yes" Question : "Who is in front of the waman?" Ground Truth : man  
X0.5 Downsampling : “no" Xselector : "no" Oracle: "yes" Ouestion : "What color is the head band?" Ground Truth : green  
0.5 Downsampling : "ornaments Selector : "ornaments" Oracle: "ornaments

![](images/b70a811832a4bafba9fdcb047c5c74e813c002d8fd02061122b6d4b0e8ce2a15.jpg)  
(a) Compression-robust  
(b) Selection-rescued

![](images/d2e8a4a7553ed213e17112513ac44fcda151e86b30a059336a60c1c1ec25dcd1.jpg)  
(c) Selection-miss  
Figure A1: Additional qualitative examples across three outcome regimes. Each column contains examples of scene-text reading, object existence, and compositional reasoning. (a) Compressionrobust: downsampling, the learned selector, and the K=1 oracle all succeed. (b) Selection-rescued: downsampling fails, while the learned selector preserves a region that recovers the correct answer. (c) Selection-miss: both downsampling and the learned selector fail, but another K=1 oracle action succeeds. Oracle regions are shown only as a diagnostic upper bound and are not available to the deployable selector.

and compositional reasoning, showing that selective preservation can recover information lost by uniform resizing across qualitatively different forms of visual evidence. These cases illustrate the benefit that learned spatial fidelity allocation can provide under the same fixed-budget representation when task-relevant information is localized to a region that benefits from higher visual fidelity.

Finally, the selection-miss cases (Figure A1c) show examples for which both whole-image downsampling and the learned selector fail, while another K=1 action selected by the oracle produces the correct answer. Because the learned and oracle conditions use the same mixed-resolution representation and differ only in which spatial cell is preserved, these examples isolate region selection as the source of failure. They qualitatively illustrate the substantial selector–oracle gap reported in Table 1: the fixed K=1 action space can contain a successful allocation even when the learned selector fails to identify it.

Together, these examples expose three complementary regimes of fixed-budget fidelity allocation. Some examples remain robust to aggressive compression, some benefit from the region selected by the learned model, and others could be recovered within the same representation if region selection were more accurate. This qualitative progression is consistent with our quantitative findings that spatial fidelity allocation has meaningful potential, while improved region selection remains necessary to exploit the available headroom.