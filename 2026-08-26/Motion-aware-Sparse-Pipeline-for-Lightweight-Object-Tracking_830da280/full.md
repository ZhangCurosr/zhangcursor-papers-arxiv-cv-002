# Motion-aware Sparse Pipeline for Lightweight Object Tracking

Qingmao Wei<sup>1,2</sup> , Fagui Liu<sup>1,2</sup> <sup>B</sup>, Dengke Zhang<sup>1</sup> , Qingze He<sup>1</sup> , and Quan Tang<sup>2</sup> <sup>B</sup>

<sup>1</sup> South China University of Technology, Guangzhou, China <sup>2</sup> Pengcheng Laboratory, Shenzhen, China

Abstract. Transformer-based object trackers are renowned for their strong performance, yet dense token processing often leads to prohibitive computational cost, limiting real-time deployment on edge devices. While recent works explore token pruning to reduce computation, they often stop short of an end-to-end sparse pipeline, as early-layer token scores can be noisy without a motion prior, and many trackers ultimately fall back to dense reshaping to feed the dense prediction head that partially negates the savings. We introduce Motion-aware Sparse Tracker (MaST), a sparse tracking framework that makes sparsity efective from tokens to boxes. First, MaST injects a lightweight motion prior to refine cross-attentionbased importance scores, enabling earlier and more stable token reduction in the search region. Second, we introduce a natively sparse prediction head that operates directly on the retained unstructured tokens with a score-first, regress-once design, eliminating dense padding/reshaping and reducing redundant computation. Extensive experiments on multiple benchmarks demonstrate that MaST establishes new state of the art among lightweight trackers, where MaST-tiny attains 63.8 AUC on LaSOT and 80.1 SUC on TrackingNet, surpassing the prior best AsymTrack-S by +1.0 AUC and +2.2 SUC while running at 152 FPS on Jetson Nano, nearly twice as fast as AsymTrack-S at 88 FPS. Code is available at github.com/TsingWei/MaST.

Keywords: Motion Prior · Token Sparsification · Lightweight Object Tracking

## 1 Introduction

Visual object tracking is a fundamental task in computer vision with broad applications in autonomous driving [20], surveillance [50], augmented reality [56], and robotics [42]. Given the target location in an initial frame, the goal is to estimate its trajectory throughout a video sequence — a task made challenging by occlusions, background clutter, illumination changes, and appearance variations.

![](images/8754ee1d52f95382f4a2b8f2686146ca3c05caa3cc1ec1ff94a4de4ca2f4cdf7.jpg)

![](images/43ee97c56320a503d17c7ea4b596fdd6a76c1eac032afae89d4d8fbe4a1b59df.jpg)  
Fig. 1: Speed–accuracy trade-of on edge platforms. The left panel shows TrackingNet SUC vs. Orin Nano speed (FPS). The right panel shows LaSOT AUC vs. CPU speed (FPS). Bubble area is proportional to model parameter count. Our MaST-{small, tiny, nano} achieve Pareto-optimal trade-ofs, delivering competitive or superior accu racy at significantly higher speeds compared to prior lightweight trackers.

Recent one-stream Transformer-based trackers [47, 54] have achieved remarkable accuracy by jointly modeling template–search interactions through selfattention. However, their quadratic attention complexity over long token sequences incurs substantial computational cost, hindering deployment on resourceconstrained platforms such as UAVs and mobile robots. The field has pursued two lines of work to improve eficiency, including lightweight architectures [5, 27, 52] that employ scaled-down backbones at the expense of discriminative power, and adaptive computation methods [30, 32, 49] that reduce average latency but leave the worst-case computational cost largely unchanged.

Unlike CNNs that operate on dense feature maps, Vision Transformers represent images as discrete token sequences, naturally enabling dynamic token reduction — a strategy proven efective in general vision tasks [4, 33]. This makes token sparsification a particularly promising avenue for accelerating Transformerbased trackers. Several recent works have explored this direction. OSTrack [54] repurposes the cross-attention scores between search and template tokens as a pruning criterion, retaining a fixed ratio of search tokens at intermediate encoder layers; FARTrack [44] extends this by additionally leveraging attention to learnable location tokens; AVTrack [32] trains an auxiliary sub-network to predict per-token importance for early dropping decisions.

Despite these advances, existing token sparsification approaches for tracking still face systematic limitations. On the token-reduction side, pruning is typically applied too late, as illustrated in Fig. 2(a), early-layer attention maps are highly difuse, making attention-based importance scores unreliable and forcing most methods to defer pruning to intermediate layers (e.g., progressively at the 4th, 7th, and 11th layers), leaving the computationally expensive early layers to process the full token set. Moreover, common pruning criteria (cross-attention scores or auxiliary predictors) capture only appearance-level relevance within the current frame and ignore the strong temporal motion continuity in tracking, i.e., the prior knowledge of where the target was in the previous frame. On the output side, the prediction head remains a dense bottleneck. Convolutional heads assume a dense, spatially structured token grid, so aggressive sparsification requires padding and reshaping sparse tokens back to a full 2D feature map, wasting computation on empty positions; at high compression ratios, the target-center token may even be pruned, leading to severe mislocalization. This mismatch between a sparse backbone and a dense head caps the achievable speedup and degrades accuracy. Notably, the decoding head is not necessarily convolutional in modern trackers, and some methods decode targets via additional learnable tokens (e.g., MixFormerV2 [13] and the ARTrack series [44,47]), and LoRAT [34] replaces the convolutional head with MLPs. However, these predictor designs are typically developed under dense token/feature assumptions, and their behavior under aggressive token sparsification remains under-explored.

![](images/3b50269f0391ab1dd1b87a56ff88006c59855505dbb40ec579d9b64235bc058c.jpg)  
(a)

![](images/965bcf6ad52df4bc6003d67ad92dbadf9de119e062a0bb97e76f7c21e3834c36.jpg)  
Fig. 2: Why early-stage token pruning fails. (a) Attention map visualization across encoder layers on three tracking sequences. Early layers produce difuse, noisy attention that has not yet localized the target, explaining why existing methods conservatively defer token pruning to intermediate layers. (b) Efect of performing one-shot token reduction (33% retention align to OSTrack [54]) at diferent encoder layers on GOT-10k val. Without spatial guidance (w/o G.T.), accuracy drops sharply as pruning moves to earlier layers. With ground-truth target locations guiding selection $( w / G . T . )$ , accuracy remains nearly flat across all layers while GMACs decrease substantially with earlier pruning.

A natural question arises as to whether early-layer pruning is inherently harmful, or whether it merely lacks the right guidance. To answer this, we conduct a diagnostic experiment that disentangles the efect of when to prune from what to prune. As shown in Fig. 2(b), when token selection at 33% retention is guided by ground-truth target locations, accuracy remains remarkably stable regardless of the pruning layer, while computational cost (GMACs) decrease substantially with earlier pruning. In contrast, without such guidance, accuracy degrades sharply as pruning moves to earlier layers. This result indicates that the performance degradation associated with early pruning stems not from insuficient token representations, but from the inability to identify the correct tokens to retain at that stage. The challenge therefore reduces to two subproblems, (i) obtaining a reliable motion prior for token selection before the encoder has produced discriminative features, and (ii) designing a prediction head that natively operates on the resulting sparse, unstructured token set.

In this work, we propose MaST (Motion-aware Sparse Tracker), a sparse tracking framework that addresses these limitations with a cohesive design. As shown in Fig. 1, MaST variants achieve Pareto-optimal speed–accuracy tradeofs on edge platforms. MaST performs early, task-driven token reduction in the search region by injecting a history-based motion prior into the selection process. To avoid a dense output bottleneck, MaST also introduces a natively sparse prediction head that operates directly on unstructured tokens with a score-first, regress-once design, eliminating dense reshape and reducing redundant computation.

Our main contributions are summarized as follows.

– We propose MaST, a sparse tracking framework that performs early, task driven token reduction in the search region using temporal motion priors.

– We introduce a natively sparse prediction head with a score-first, regress-once design that operates directly on unstructured tokens, removing the dense head bottleneck.

– Extensive experiments on LaSOT, TrackingNet, GOT-10k, and more benchmarks demonstrate that MaST achieves competitive or superior accuracy while delivering substantially higher measured inference speeds on edge platforms such as Raspberry Pi and Jetson Nano.

## 2 Related Works

Lightweight Visual Tracking. Modern one-stream Transformer trackers [8, 12, 13, 47, 54] unify feature extraction and relation modeling into a single ViT backbone, achieving strong accuracy but sufering from quadratic self-attention complexity that hinders deployment on edge devices. Prior eforts to close this gap fall into three categories, each with notable limitations. Compact model design [3, 5, 10, 27, 32, 46] reduces parameter count but sacrifices representational capacity. Model compression via layer pruning [13,46] or knowledge distillation [13,24] risks non-trivial accuracy degradation. Conditional dynamic computation [30, 32, 49] improves average throughput but yields variable per-frame latency, complicating hardware scheduling. In contrast, our approach uses fixed top-K token selection guided by a motion prior, achieving both predictable latency and high accuracy.

Temporal Priors in Visual Tracking. Temporal priors, especially motion locality, are well-established in visual tracking, as target motion between consecutive frames is typically small relative to the search region. Classical approaches exploit this prior via a penalty window (Hanning/cosine) applied to the output score map. SiamFC [1] first applied a cosine window to the cross-correlation response to suppress high-response regions near the boundary of the search area. SiamRPN++ [29] further combines a cosine window penalty with a size-change penalty on the regression response to reduce large, abrupt displacement predictions. Discriminative correlation filters such as KCF [23] and ECO [15] apply a cosine window to the input features to reduce periodic boundary artefacts, which also implicitly penalises predictions near the edges. Many modern trackers, including DiMP [2], prDiMP [16], STARK [51], and OSTrack [54], likewise multiply a Hanning window onto the predicted score map as a post-processing step before selecting the final target location. Beyond hand-crafted windows, recent Transformer-based trackers also explore data-driven temporal modeling by encoding historical predictions as tokens and letting the network learn to use this information, e.g., SwinTrack [35] and the ARTrack series [44, 47]. However, these designs do not explicitly impose a spatial locality prior during computation. Despite their widespread adoption, penalty windows are applied only after the full forward pass and merely re-weight already-computed scores, leaving the heavy encoder computation unafected. In contrast, our approach injects temporal motion priors into the token reduction process itself, using them to guide which tokens are computed in subsequent layers and thereby reducing computation.

Token Sparsification in Vision Transformers. The tokenized processing paradigm of Vision Transformers enables novel sparsification strategies that are absent in CNN-based trackers. EViT [33] pioneered token pruning by selecting tokens with the highest attention to the class token. OSTrack [54] adapted this idea by measuring cross-attention between search and template tokens, retaining the top-K most correlated tokens. However, their reliance on early-layer attention scores introduces noise. As a result, many methods apply token pruning conservatively and progressively at intermediate layers, leading to limited speedup. DynamicViT [41] proposed learnable importance predictors using lightweight MLPs, and DToP [43] further explored this idea in semantic segmentation. Similarly, GRM [21] applied such techniques to guide token interactions in tracking tasks, while AVTrack [32] uses a sub-network for early token exit. SGL [57] likewise observes that partial early-layer attention is insuficient for reliable token pruning and introduces a small vision-language model as external guidance for a larger model. MaST shares the motivation of guided early pruning, but specializes it to single-object tracking by using a nearly cost-free temporal motion prior rather than an auxiliary model. Moreover, our fixed top-K budget avoids the variable per-frame computation of adaptive exits and provides predictable latency.

## 3 Method

## 3.1 Overview

The proposed framework shown in Fig. 3 operates in three stages, namely motion prediction, token sparsification, and target decoding. Given an input video frame and the target’s initial location, a pair of images is processed as input, consisting of a template image denoted as $\boldsymbol { \bar { Z } } \in \mathbb { R } ^ { 3 \times H _ { z } \times \smile }$ and a search image denoted as $\boldsymbol { X } \in \mathbb { R } ^ { 3 \times \overline { { H _ { \boldsymbol { x } } \times W _ { \boldsymbol { x } } } } }$ . These images are divided into non-overlapping patches of size $P \times P .$ , resulting in $\begin{array} { r } { P _ { z } = \frac { \boxtimes _ { z } \times W _ { z } } { P ^ { 2 } } } \end{array}$ patches for the template and $\begin{array} { r } { P _ { x } = \frac { H _ { x } \times W _ { x } } { P ^ { 2 } } } \end{array}$ patches for the search region. After patch embedding concatenated together, features are simultaneously extracted and fusion in the form of sequence using a Transformer encoder, in which the sequence will be first handled by our proposed sparsification block. In the end, the retained tokens will be further processed by the prediction head.

![](images/b1d076619a911441671e5c7487fc5c9643471b7e42435e5d1add1d73b410087e.jpg)  
Fig. 3: Overview of the MaST framework. The General Sparsification Module is inserted once at an early encoder layer, retaining the top-K search tokens; all subsequent Transformer blocks operate on this reduced set. Token selection fuses cross-attention relevance with a motion prior from the previous prediction for early and stable pruning. The retained sparse tokens are then decoded by a lightweight MLP sparse head with a score-first, regress-once pipeline, eliminating dense reshaping and redundant computation.

## 3.2 Motion-Aware Token Sparsification

Token sparsification plays a critical role in reducing the computational cost of Transformer-based visual object trackers by selectively focusing on the most relevant tokens. Typically, an importance score is assigned to each search token to determine whether it should be retained. Existing approaches often derive these scores either from an auxiliary prediction network or from cross-attention maps that measure the interaction between search and template tokens. Following OSTrack [54], we adopt the cross-attention map as the basis for token scoring. However, early-layer attention is often difuse and noisy, which makes naive early pruning unreliable. To enable earlier and more stable token reduction, we inject a lightweight temporal motion prior derived from the previous prediction.

Importance Score. In our framework, the importance score of search tokens is derived from the cross-attention map in the Transformer encoder. Let the search tokens and template tokens be represented as $\mathbf { Q } _ { x } \in \mathbb { R } ^ { P _ { x } \times d }$ (queries) and $\mathbf { K } _ { z } \in \mathbb { R } ^ { P _ { z } \times d } \ ( \mathrm { k e y s } )$ , where $P _ { x }$ and $P _ { z }$ are the numbers of tokens in the search and template regions, respectively, and d is the feature dimension. Following the approach explored in OSTrack, we simplify the aggregation process by only using the token from the center patch of the template as a representative feature. This reduces unnecessary complexity while retaining strong performance.

The cross-attention map $\mathbf { A } _ { x  z } \in \mathbb { R } ^ { P _ { x } \times P _ { z } }$ , which measures the interaction between search and template tokens, is defined as:

$$
{ \bf A } _ { x  z } = \mathrm { S o f t m a x } ( \frac { { \bf Q } _ { x } { \bf K } _ { z } ^ { \top } } { \sqrt { d } } ) ,\tag{1}
$$

where $\mathbf { A } _ { x  z } [ i , j ]$ represents the attention weight between the i-th search token and the j-th template token. Given that we only use the center token of the template, denoted as $\mathbf { k } _ { c } \in \mathbb { R } ^ { d }$ , the importance score ${ \bf s } _ { i }$ for the i-th search token is simplified as:

$$
{ \bf s } _ { i } = \frac { \exp \left( { \bf q } _ { i } ^ { \top } { \bf k } _ { c } / \sqrt { d } \right) } { \sum _ { k = 1 } ^ { P _ { x } } \exp \left( { \bf q } _ { k } ^ { \top } { \bf k } _ { c } / \sqrt { d } \right) } ,\tag{2}
$$

where $\mathbf { q } _ { i } \in \mathbb { R } ^ { d }$ is the query feature of the i-th search token. This formulation focuses the importance scoring process on the interaction between search tokens and the central template token, providing a lightweight yet efective means of measuring token relevance.

Motion Prior Injection. Directly using early-layer cross-attention scores for pruning is sub-optimal due to their difuse and noisy nature. Tracking, however, provides a strong temporal cue, as the target location changes smoothly across frames. We therefore build a simple spatial prior from the previous prediction and use it to guide token selection.

Specifically, as illustrated in the center panel of Fig. 3, we construct a Motion Window over the search region, centered at the previous predicted target location, and use it to softly re-weight the cross-attention importance scores. In practice, we find that a 2D Gaussian provides an efective and lightweight realization. Given the predicted bounding box from the previous frame $\mathbf { b } _ { t - 1 } = ( x _ { t - 1 } , y _ { t - 1 } , w _ { t - 1 } , h _ { t - 1 } )$ where $( x _ { t - 1 } , y _ { t - 1 } )$ is the box center and $w _ { t - 1 } , h _ { t - 1 }$ are its width and height, the Gaussian is defined as:

$$
\mathcal { G } _ { t } ( u , v ) = \exp \left( - \frac { ( u - x _ { t - 1 } ) ^ { 2 } } { 2 \sigma _ { x } ^ { 2 } } - \frac { ( v - y _ { t - 1 } ) ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } \right) ,\tag{3}
$$

where $( u , v )$ represents the spatial coordinates of a token in the search region, and $\sigma _ { x } = \gamma w _ { t - 1 } , \sigma _ { y } = \gamma h _ { t - 1 }$ are the standard deviations of the Gaussian in the x and y directions, respectively. The scaling factor $\gamma$ controls the size of the window, balancing the trade-of between including potentially relevant tokens and excluding irrelevant ones. We found that setting $\gamma = 0 . 5$ yields reasonable performance.

The Motion Window assigns higher weights to tokens closer to the previous predicted target center. We refine the importance scores by combining the crossattention importance scores $\mathbf { s } _ { i }$ with the Motion Window values $\mathcal { G } _ { t } ( u _ { i } , v _ { i } )$ for each token:

$$
\mathbf { w } _ { i } = { \mathcal G } _ { t } ( u _ { i } , v _ { i } ) \cdot \mathbf { s } _ { i } ,\tag{4}
$$

where $( u _ { i } , v _ { i } )$ is the spatial location of the i-th token. This combined score $\mathbf { w } _ { i }$ ensures that tokens within the likely target region (as constrained by the Motion Window) are prioritized, even if their raw attention scores are difuse. Then we only keep the top-K tokens $\mathcal { L } _ { K } \in \mathbb { R } ^ { N _ { K } \times d }$ as the output of Motion-Aware Token Sparsification block, attached by the standard Transformer blocks in the encoder.

## 3.3 Fully Sparse Prediction Head

Our goal is to avoid the dense-head bottleneck when the backbone operates on a sparsified, unstructured token set. Let the retained search tokens be $\mathcal { L } _ { K } = \{ \mathbf { f } _ { k } \} _ { k = 1 } ^ { N _ { K } }$ , where $N _ { K } \ll$ $H _ { x } W _ { x } / P ^ { 2 }$ and each token keeps its original 2D location on the patch grid, denoted as $\mathbf { p } _ { k } ~ = ~ ( u _ { k } , v _ { k } )$ . As illustrated in Fig. 4, unlike dense heads that must pad and reshape sparse tokens back to a full 2D grid before stacked convolutions, we design a natively sparse head that directly consumes $\mathcal { L } _ { K }$ without padding or reshaping back to a dense feature map. We parameterize both branches with lightweight MLPs, similar to prior MLP-based heads (e.g., LoRAT [34]). We further tailor the head for sparse token decoding via a score-first, regressonce computation path.

![](images/9a5acb5f1df3009e7325b3eddd5d2c9688e09961c838cec9c052bfdfe1ffbbaa.jpg)  
Fig. 4: Dense vs. sparse prediction head. (A) Dense head pads and reshapes sparse tokens into a full 2D grid, wasting computation on empty positions. (B) Our sparse head applies an MLP directly on retained tokens via sparse anchors, with no padding or reshape.

Score-first, regress-once. As illustrated in Fig. 3, the head contains two lightweight branches.

– Score branch $g _ { \mathrm { s } }$ predicts a scalar confidence $s _ { k } \in \mathbb { R }$ for each retained token. The target token is selected by

$$
k ^ { * } = \arg \operatorname* { m a x } _ { k } \ s _ { k } .\tag{5}
$$

This selection can be implemented by scattering $\{ s _ { k } \}$ to a sparse score map aligned with the original grid and applying an ArgMax, but no dense feature computation is required.

– Regression branch $g _ { \mathrm { r } }$ predicts box parameters only once for the selected token $\mathbf { f } _ { k ^ { * } }$ . Denoting the regression output as $\varDelta _ { k ^ { * } } = \left( \delta _ { x , k ^ { * } } , \delta _ { y , k ^ { * } } , w _ { k ^ { * } } , h _ { k ^ { * } } \right)$ the final box is decoded as

$$
\hat { \mathbf { b } } = \mathrm { D e c o d e } ( \mathbf { p } _ { k ^ { * } } , \mathbf { \xi } , \mathbf { \psi } _ { \lambda _ { k ^ { * } } } ) = ( u _ { k ^ { * } } + \delta _ { x , k ^ { * } } , v _ { k ^ { * } } + \delta _ { y , k ^ { * } } , w _ { k ^ { * } } , h _ { k ^ { * } } ) ,\tag{6}
$$

followed by conversion from patch-grid coordinates to search-image pixels using the patch stride.

This is the same local anchor-based box definition used by dense anchorbased heads, which decode the bounding box at every grid location after a 2D reshape. MaST instead retrieves $\mathbf { p } _ { k } \ast \mathbf { \sigma } _ { }$ ∗ directly from the retained token’s prepruning flattened-grid coordinate and evaluates the regression branch only at that location. Thus, only the dense reshape and redundant regressions are removed; the box parameterization is unchanged.

This design makes the head complexity scale with $N _ { K }$ for scoring and constanttime for regression, matching the sparsified backbone and eliminating redundant computation on empty spatial positions.

Training objective. We supervise the score branch with a classification loss over sparse tokens, and supervise the regression branch on a single token aligned with the ground-truth target. Let c be the ground-truth target center on the patch grid and define

$$
k ^ { \mathrm { g t } } = \arg \operatorname* { m i n } _ { k } \| \mathbf { p } _ { k } - \mathbf { c } \| _ { 2 } .\tag{7}
$$

The overall head loss is

$$
L _ { \mathrm { h e a d } } = L _ { \mathrm { c l s } } ( \{ s _ { k } \} ) + \lambda _ { \ell _ { 1 } } L _ { \ell _ { 1 } } ( \hat { \bf b } _ { k ^ { \mathrm { g t } } } , { \bf b } ) + \lambda _ { \mathrm { G I o U } } L _ { \mathrm { G I o U } } ( \hat { \bf b } _ { k ^ { \mathrm { g t } } } , { \bf b } ) ,\tag{8}
$$

where b is the ground-truth box, and we set $\lambda _ { \mathrm { G I o U } } = 2$ and $\lambda _ { \ell _ { 1 } } = 5$

## 4 Experiment

Device. Our MaST models are implemented using Python 3.10 and PyTorch 2.4.1. Training is conducted on a single NVIDIA RTX 3090 GPU. We evaluate inference speed on three representative edge platforms, namely Raspberry Pi 5, Apple M4, and NVIDIA Jetson Orin Nano. To ensure a consistent evaluation environment and better reflect real-world application deployment, we export ONNX files with OpSet 17 and benchmark with ONNXRuntime 1.20.1 to report practical on-device performance. For trackers whose source code is publicly available, we re-evaluate their speed under the same protocol to ensure a fair comparison.

Table 1: Details of our MaST model variants.
<table><tr><td rowspan="2">Model</td><td colspan="2">MaST</td></tr><tr><td></td><td>nano tiny small</td></tr><tr><td>Backbone</td><td>ViT-Tiny</td><td></td></tr><tr><td>Encoder Layers</td><td>8</td><td>12 12</td></tr><tr><td>Template Size</td><td>128</td><td>128 192</td></tr><tr><td>Search Size</td><td>256 256</td><td>384</td></tr><tr><td>Token Rate (%)</td><td>30 30</td><td>30</td></tr><tr><td>MACs (G)</td><td>0.59 0.84</td><td>1.82</td></tr><tr><td>Params (M)</td><td>3.85 5.63</td><td>5.63</td></tr></table>

## Model. To demonstrate our method on

the low-cost edge device, we use ViT-Tiny [18] as the encoder and the training are initialized with the MAE-lite [45]. We provide three model variants with diferent backbones, encoder depths, and token retention rates, as summarized in Tab. 1. MaST-nano uses the first 8 layers of ViT-Tiny for maximum eficiency; MaST-tiny employs the full 12-layer ViT-Tiny; and MaST-small increases the input resolution for higher accuracy. All three default variants retain 30% of the search tokens. A supplementary ViT-Base scaling study further shows that the speedup persists beyond lightweight backbones while reaching the lower accuracy range of heavier trackers.

Training. The training splits of COCO [36], LaSOT [19], GOT-10k [25] TrackingNet [39] and VastTrack [40] are used for training. Common data augmentations including horizontal flip and brightness jittering are used in training. The GPU holds 128 image pairs as the batch size. We train the model with AdamW optimizer [37], set the weight decay to $1 0 ^ { - 4 }$ , the initial learning rate for the backbone to $\dot { 4 } \times 1 0 ^ { - 5 }$ and other parameters to $4 \times 1 0 ^ { - 4 }$ , respectively. The total training epochs are set to 300 with 60k image pairs per epoch and we decrease the learning rate by a factor of 10 after 240 epochs. The sparsification rate is gradually warmed-up during training following OSTrack [54]. More details of training and hyper-parameters can be found in the supplementary material.

## 4.1 Comparison with State-of-the-arts

We conduct a comprehensive comparison on five widely used tracking benchmarks. Trackers are grouped into three tiers, namely ${ \sim } 5 0 0 0 , \ { \sim } 1 \mathrm { G }$ , and 2G MACs, based on their computational cost, and compared within the same tier.

LaSOT. LaSOT [19] is a large-scale long-term tracking benchmark comprising 1,400 video sequences with 280 sequences reserved for testing. As shown in Tab. 2, MaST-tiny achieves the highest AUC of 63.8 and $P _ { N o r m }$ of 72.2 in the 1G group, outperforming AsymTrack-S [59] by +1.0 AUC and HiT-Small [27] by +3.3 AUC, while running at 22.6 FPS on Raspberry Pi 5. MaST-small further <sup>pushes</sup> <sup>to</sup> <sup>65.8</sup> <sup>AUC</sup> <sup>in</sup> <sup>the</sup> ∼<sup>2G</sup> <sup>group,</sup> <sup>surpassing</sup> <sup>FERMT</sup> <sup>[58]</sup> <sup>by</sup> <sup>+0.7</sup> <sup>AUC.</sup>

TrackingNet. TrackingNet [39] is a large-scale short-term benchmark with 511 test sequences covering diverse real-world scenarios. MaST-tiny leads the 1G group with 80.1 SUC and $P _ { N o r m }$ of 85.3, surpassing AsymTrack-S by +2.2 SUC. In the 2G group, MaST-small achieves 82.3 SUC, outperforming all competing methods in its tier.

GOT-10k. GOT-10k [25] is a large-scale dataset with 180 test sequences featuring non-overlapping object categories between train and test splits, designed to measure generalization. MaST-tiny attains an AO of 66.6 and $\mathrm { S R } _ { 0 . 7 5 }$ of 62.5 in the 1G group, outperforming AsymTrack-S by +1.1 AO and $+ 3 . 6 ~ \mathrm { S R } _ { 0 . 7 5 }$ MaST-small achieves 70.0 AO and the highest $\mathrm { S R } _ { 0 . 7 5 }$ of 67.0 in the 2G group.

L $\mathbf { a S O T _ { e x t } } . \mathrm { L a S O T _ { e x t } }$ [19] extends LaSOT with 150 test sequences covering more diverse and challenging long-term scenarios. As reported in Tab. 3, MaST small achieves the highest AUC of 45.3, surpassing FARTrack<sub>tiny</sub> [44] by +0.3 and AsymTrack-B [59] by +0.7. MaST-tiny achieves 42.8 AUC at 22.6 FPS on Raspberry Pi 5, over 3 faster than AsymTrack-B (6.2 FPS) ,ofering a favorable speed–accuracy trade-of for long-term tracking.

Table 2: State-of-the-art comparisons of eficient tracking on three large-scale benchmarks, grouped by computational cost. Red and blue denote the best and second-best results within each group. Gray FPS values indicate speeds below VOT realtime setting (20 FPS).
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pub.</td><td colspan="3">LaSOT</td><td colspan="3">TrackingNet</td><td colspan="2">GOT-10k</td><td rowspan="2">MACs</td><td colspan="3">FPS</td></tr><tr><td colspan="2">AUC PNorm</td><td>P</td><td colspan="2">SUC PNorm</td><td>P AO</td><td colspan="2">SR0.5 SR0.75</td><td>(G)</td><td colspan="2">RPi5 CPU Nano</td></tr><tr><td>KCF [23]</td><td>|TPAMI&#x27;14|</td><td>21.1</td><td>22.8</td><td>18.4</td><td></td><td></td><td></td><td>27.926.3</td><td>9.9</td><td></td><td>142</td><td>946 2033</td><td></td></tr><tr><td colspan="10">~500M MACs</td><td></td><td></td><td></td><td></td></tr><tr><td>MaST-nano</td><td>Ours</td><td>58.6</td><td>67.7</td><td>59.4</td><td>77.2</td><td>82.5</td><td>72.4</td><td>61.6 71.8</td><td>53.7</td><td>0.585</td><td>30.1</td><td>101</td><td>230</td></tr><tr><td>AsymTrack-T [59]</td><td>AAAI&#x27;25</td><td>60.8</td><td>68.7</td><td>61.2</td><td>76.2 80.9</td><td></td><td>71.6 62.3</td><td>71.3</td><td>54.7</td><td>0.708</td><td>18.1</td><td>60</td><td>99</td></tr><tr><td>FEAR-XS [5]</td><td>ECCV&#x27;22</td><td>53.5</td><td>=</td><td>54.5</td><td>1</td><td>1</td><td>1</td><td>61.9 72.2</td><td>=</td><td>0.532</td><td>19.5</td><td>59</td><td>146</td></tr><tr><td>LightTrack [52]</td><td>CVPR’21</td><td>53.8</td><td>=</td><td>53.7</td><td>72.5</td><td>77.8</td><td>69.5</td><td>61.1 71.0</td><td>=</td><td>0.483</td><td>13.4</td><td>44</td><td>124</td></tr><tr><td colspan="10">~1G MACs</td><td></td><td></td><td></td><td></td></tr><tr><td>MaST-tiny</td><td>Ours</td><td>63.8</td><td>72.2</td><td>67.2</td><td>80.1</td><td>85.3</td><td>77.2</td><td>66.6 76.0</td><td>62.5</td><td>0.836</td><td>22.6</td><td>72</td><td>152</td></tr><tr><td>FARTrackpico [44]</td><td>ICLR’26</td><td>58.6</td><td>67.1</td><td>59.6</td><td>75.6</td><td>81.3</td><td>70.5</td><td>62.8 72.6</td><td>50.9</td><td>1.08</td><td>17.2</td><td>35</td><td>134</td></tr><tr><td>AsymTrack-S [59]</td><td>AAAI&#x27;25</td><td>62.8</td><td>71.2</td><td>64.8</td><td>77.9</td><td>82.2</td><td>74.0</td><td>65.5 74.8</td><td>58.9</td><td>0.806</td><td>15.6</td><td>51</td><td>88</td></tr><tr><td>ORTrack-D [48]</td><td>CVPR&#x27;25</td><td>54.6</td><td>62.6</td><td>54.3</td><td>73.7</td><td>79.1</td><td>68.2</td><td></td><td>=</td><td>1.54</td><td>18.5</td><td>52</td><td>138</td></tr><tr><td>HiT-Small [27]</td><td>ICCV’23</td><td>60.5</td><td>68.3</td><td>61.5</td><td>77.7</td><td>81.9</td><td>73.1</td><td>62.6 71.2</td><td>54.4</td><td>1.13</td><td>21.5</td><td>78</td><td>106</td></tr><tr><td>HiT-Tiny [27]</td><td>ICCV&#x27;23</td><td>54.8</td><td>60.5</td><td>52.9</td><td>74.6</td><td>78.1</td><td>68.8</td><td>52.6 59.3</td><td>42.7</td><td>0.995</td><td>27.3</td><td>97</td><td>118</td></tr><tr><td>E.T.Track [3]</td><td>WACV&#x27;23</td><td>59.1 1</td><td></td><td></td><td>75.0</td><td>80.3</td><td>70.6</td><td></td><td></td><td>1.56</td><td>8.8</td><td>24</td><td>71</td></tr><tr><td colspan="10">~2G MACs</td><td colspan="3"></td></tr><tr><td>MaST-small</td><td>Ours</td><td>65.8</td><td>74.7</td><td>70.4</td><td>82.3</td><td>87.1</td><td>80.5 70.0</td><td>80.4</td><td>67.0</td><td>1.82</td><td>7.5</td><td>30</td><td>98</td></tr><tr><td>FARTracknano [44]</td><td>ICLR&#x27;26</td><td>61.3</td><td>69.7</td><td>64.1</td><td>79.1</td><td>84.5</td><td>75.6 69.9</td><td>81.2</td><td>61.4</td><td>1.78</td><td>10.4</td><td>25</td><td>117</td></tr><tr><td>FARTracktiny [44]</td><td>ICLR&#x27;26</td><td>63.2</td><td>71.6</td><td>66.7</td><td>80.7</td><td>85.6</td><td>77.5 70.6</td><td>81.0</td><td>63.8</td><td>2.65</td><td>6.9</td><td>16</td><td>87</td></tr><tr><td>AsymTrack-B [59]</td><td>AAAI&#x27;25</td><td>64.7</td><td>73.0</td><td>67.8</td><td>80.0</td><td>84.5</td><td>77.4</td><td>67.77 76.6</td><td>61.4</td><td>1.81</td><td>6.2</td><td>25</td><td>57</td></tr><tr><td>FERMT [58]</td><td>ECCV&#x27;24</td><td>65.1 74.6</td><td></td><td>69.1</td><td>80.8 80.9</td><td></td><td>78.1 69.6</td><td>80.1</td><td>63.2</td><td>2.31</td><td>7.9</td><td>31</td><td>84</td></tr><tr><td>HiT-Base [27]</td><td>ICCV’23</td><td>64.6 73.3</td><td></td><td>68.1</td><td>80.0 84.4</td><td></td><td>77.3 64.0</td><td>72.1</td><td>58.1</td><td>4.34</td><td>6.7</td><td>22</td><td>79</td></tr><tr><td colspan="10">Heavy Trackers</td><td></td><td></td><td></td><td></td></tr><tr><td>SUTrack-T [9]</td><td>AAAI&#x27;25</td><td>69.6</td><td>79.3</td><td>75.4</td><td>82.7</td><td>87.2</td><td>80.8 72.7</td><td>82.1</td><td>70.5</td><td>5.52</td><td></td><td>2.1 10.9</td><td>23</td></tr><tr><td>MCITrack-T [26]</td><td>AAAI&#x27;25</td><td>71.7</td><td>81.5</td><td>78.2</td><td>84.8</td><td>89.4</td><td>83.7 74.0</td><td>83.9</td><td>72.1</td><td>12.6</td><td></td><td>4.3</td><td>9.1</td></tr><tr><td>LoRAT [34]</td><td>ECCV&#x27;24</td><td>71.7</td><td>80.9</td><td>77.3</td><td>83.5</td><td>87.9</td><td>82.1 72.1</td><td>81.8</td><td>70.7</td><td>19.1</td><td></td><td>5.4</td><td>12.1</td></tr><tr><td>OSTrack [54]</td><td>ECCV&#x27;22</td><td>69.1</td><td>78.7</td><td>75.2</td><td>83.1</td><td>87.8</td><td>82.0 71.0</td><td>80.4</td><td>68.2</td><td>18.6</td><td></td><td>6.5</td><td>13.7</td></tr></table>

UAV123. UAV123 [38] is an aerial tracking benchmark with 123 low-altitude UAV sequences featuring fast motion, small targets, and frequent occlusion. MaSTtiny achieves the highest AUC of 66.6 among all compared methods, outperforming LiteTrack-B4 [46] (66.4) and AsymTrack-B [59] (66.5) while running at 22.6 FPS on Raspberry Pi 5. MaST-nano attains 62.9 AUC at a leading 30.1 FPS, making it well-suited for resource-constrained aerial applications.

NFS. NFS [28] is a high-frame-rate benchmark with 100 sequences featuring fast-moving objects that challenge tracker responsiveness. Under 30fps setting, MaST-tiny achieves 66.2 AUC, second only to FARTrack<sub>tiny</sub> [44] (66.9) while running substantially faster on edge devices. MaST-small achieves 65.7 AUC, surpassing AsymTrack-B [59] (64.4) and HiT-Base [27] (63.6).

Table 3: Comparisons with lightweight trackers on more benchmarks.
<table><tr><td>Method</td><td>FPS RPi Nano</td><td> $\mathrm { \left| L a S O T _ { \mathrm { e x t } } \right. }$  AUC</td><td>UAV123 NFS AUC</td><td>AUC</td></tr><tr><td>LightTrack [52] FEAR-XS [5] E.T.Track [3] HiT-Base [27]</td><td>13.4 124| 19.5 146 8.8 71 6.7 79</td><td>一 一 44.1</td><td>62.5 61.4 59.0 65.6</td><td>55.3 61.4 59.0 63.6</td></tr><tr><td>MixformerV2-S [13] LiteTrack-B4 [46]</td><td>7.1 89 3.6 45</td><td>43.6</td><td>65.8 66.4</td><td>63.4</td></tr><tr><td>CompressTracker-2 [24]</td><td>6.1 97 6.2</td><td>40.4</td><td>62.5</td><td></td></tr><tr><td>AsymTrack-B [59]</td><td>57</td><td>44.6</td><td>66.5</td><td>64.4</td></tr><tr><td>FARTracktiny [44]</td><td>6.9 87</td><td>45.0</td><td>65.8</td><td>66.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MaST-nano</td><td>|30.1 230|</td><td>41.1</td><td>62.9</td><td>64.4</td></tr><tr><td>MaST-tiny</td><td>22.6 152</td><td>42.8</td><td>66.6</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>66.2</td></tr><tr><td>MaST-small</td><td>7.5 98</td><td>45.3</td><td>66.4</td><td>65.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 4: Comparison on the VastTrack [40] benchmark.
<table><tr><td>Method</td><td>VastTrack SUC PNorm Prec</td></tr><tr><td>ATOM [14] SiamRPN++ [29]</td><td>17.2 14.1 10.2</td></tr><tr><td>29.9</td><td>28.1 29.7 24.9</td></tr><tr><td>TransT [11]</td><td>31.4 25.4</td></tr><tr><td>STARK [51] 33.4</td><td>34.3 30.8</td></tr><tr><td>OSTrack [54] 33.6</td><td>34.5 31.5</td></tr><tr><td>ARTrack 35.6</td><td>35.8 32.4</td></tr><tr><td>MixFormerV2-B [13] 35.2</td><td>36.5 33.0</td></tr><tr><td>FARTrackpico [44] 30.3</td><td>33.0 25.7</td></tr><tr><td>FARTracknano [44] 33.9</td><td>35.1 30.3</td></tr><tr><td>FARTracktiny [44] 35.2</td><td>36.5 32.3</td></tr><tr><td>MaST-nano MaST-tiny</td><td>|30.6 37.8 30.1 33.4 40.7</td></tr><tr><td>MaST-small 35.6</td><td>26.2 43.2 33.0</td></tr></table>

Table 5: Comparison of performance on various sparisfication on encoder.
<table><tr><td>Sparsification By</td><td>LaSOT  $\mathrm { A U C } \operatorname* { P } _ { N o r m }$ </td><td>P</td><td>|Encoder| MACs</td><td>FPS RPi5 CPU Nano</td><td></td></tr><tr><td>None</td><td>64.0 74.2</td><td>68.3|</td><td>1752M</td><td>9.1</td><td>37 94</td></tr><tr><td>Attention</td><td>60.5 71.9</td><td>65.6</td><td>824M</td><td>22.9 73</td><td>157</td></tr><tr><td>Prediction</td><td>59.4 71.4</td><td>64.9</td><td>874M</td><td>21.5 66</td><td>138</td></tr><tr><td>Attention + Motion</td><td>63.8 73.6</td><td>67.9</td><td>824M</td><td>22.6 72</td><td>152</td></tr><tr><td>Motion</td><td>62.6 72.2</td><td>66.4</td><td>824M</td><td>23.5 78</td><td>169</td></tr></table>

VastTrack. VastTrack [40] is a large-vocabulary benchmark spanning 2,115 object categories with over 50,000 sequences, designed to evaluate tracking generalization across a vast range of target types. As shown in Tab. 4, MaST-small achieves a SUC of 35.6 and $\mathrm { P } _ { N o r m }$ of 43.2, surpassing heavier MixFormerV2- B [13](35.2 SUC) by +0.4. The $\mathrm { P } _ { N o r m }$ advantage is particularly striking at +6.7 over the next-best method, suggesting superior localization accuracy normalized by target scale.

## 4.2 Ablation Study

Impact of Diferent Sparsification Strategies. Tab. 5 compares various token sparsification strategies. Naive pruning based solely on attention scores (“Attention”) achieves significant compute reduction but sufers a severe accuracy drop, revealing that early cross-attention scores are too difuse to reliably identify important tokens. Replacing attention with an MLP predictor (“Prediction”) is even worse, as it lacks spatial grounding. The proposed motion prior is the key enabler, as fusing attention scores with the Gaussian spatial prior (“Attention +

Table 6: Comparison of prediction heads on LaSOT with ViT-Tiny backbone.
<table><tr><td>Prediction Head</td><td>AUC Dense Sparse</td><td></td><td>MACs (G) Dense Sparse</td><td></td><td>RPi FPS Dense Sparse</td></tr><tr><td colspan="6">Global Localization</td></tr><tr><td>Loc Tokens [13]</td><td>59.0</td><td>57.9</td><td>1.76 0.845</td><td>10.0</td><td>23.9</td></tr><tr><td>Trans. Decoder</td><td>61.4</td><td>60.1</td><td>1.88 0.833</td><td>9.4</td><td>22.7</td></tr><tr><td colspan="6">Anchor-based Localization</td></tr><tr><td>3×3 Conv [54]</td><td>65.0</td><td>64.1</td><td>2.39</td><td>1.463</td><td>7.7 13.8</td></tr><tr><td>MLP (Dense) [34]</td><td>64.0</td><td>63.8</td><td>1.81</td><td>0.872 9.1</td><td>21.3</td></tr><tr><td>MLP (Sparse)</td><td>64.0</td><td>63.8</td><td>1.75</td><td>0.836</td><td>9.7 23.2</td></tr></table>

Motion”) nearly closes the accuracy gap to the dense baseline while preserving the full compute savings. Using the motion window alone (“Motion”) yields a middle ground, confirming that the fusion of both signals is optimal. These results underscore that a lightweight temporal prior is critical to making aggressive early-layer sparsification practically viable.

![](images/8bf414123bc6334e451acde26e2fba3bd2073965ec544eb8946deef43941d2d0.jpg)  
Fig. 5: AUC–FPS trade-of at different token retention rates on GOT-10k val

Analysis of Diferent Prediction Heads. Tab. 6 compares prediction heads under two paradigms. Global localization heads (Loc Tokens, Trans. Decoder) are naturally sparse-compatible and achieve good speedups, but their accuracy is limited (57.9–60.1 AUC) due to insuficient spatial prior. These variants also converge substantially more slowly in image-pair IoU; extending training to 3 the default schedule, initializing from a trained head, and teacher distillation did not close the gap (see the Appendix). Anchor-based heads yield higher accuracy (63.8–64.1 AUC), yet the conventional 3 3 Conv-stacked head must reshape sparse tokens back to a dense grid, incurring redundant computation and capping

throughput at 13.8 FPS. Our sparse MLP head operates score-first, regress-once on retained tokens, improving throughput to 23.2 FPS with no accuracy loss over the dense MLP counterpart [34] (63.8 AUC in both cases), confirming the value of a natively sparse head design.

Impact of Token Retention Rate. Fig. 5 examines how the token retention rate afects the AUC–FPS trade-of at two input resolutions on GOT-10k val. For this diagnostic only, we warm up one checkpoint to a 10% retention rate during training and evaluate the same checkpoint at all plotted rates, rather than retraining a separate model for each operating point. At 384 resolution, retaining only 30–40% of tokens already matches the dense baseline in AUC while substan tially boosting throughput; further reduction to 20% causes only a moderate drop.

At 256 resolution, 50–60% retention achieves near-lossless accuracy, and 30% retention remains competitive.0

Table 7: Efect of sparsification layer on tracking accuracy and eficiency.
<table><tr><td>Sparsify at Layer</td><td>|LaSOT RPi MACs AUC FPS (M)</td></tr><tr><td>1</td><td>63.82 22.6 836</td></tr><tr><td>2</td><td>63.87 17.6 942</td></tr><tr><td>3</td><td>63.65 14.0 1004</td></tr><tr><td>4</td><td>63.83 12.1 1080</td></tr><tr><td>5 6</td><td>63.91 11.3 1154 63.96 10.1 1238</td></tr></table>

In both settings the motion prior guides early token selection reliably regardless of input scale, confirming that MaST’s accuracy–eficiency trade-of is robust across resolution choices and that a single checkpoint supports a smooth post-training speed–accuracy trade-of.

Layer-wise Sparsification Analysis. Tab. 7 evaluates the efect of applying token sparsification at diferent layers (all with motion window) of the ViT-Tiny encoder. Under the motion prior, the choice of sparsification layer has only a negligible impact on accuracy: AUC varies by at most 0.4 points across all configurations. In contrast, the speed penalty of delaying sparsification is substantial, throughput nearly halves from 22.6 to 10.1 FPS when

sparsification is pushed to later layers. Sparsifying at Layer 1 therefore ofers the best cost-efectiveness, as it matches the accuracy of later-layer alternatives while delivering the highest throughput (22.6 FPS) and the lowest compute (836M FLOPs), and we adopt it as our default.

Comparison with Alternative Compression Strategies. Tab. 8 compares MaST against three compression baselines under an identical compute budget ( 1G MACs, ViT-Tiny backbone): the full dense model (1752M MACs), progressive token pruning from OSTrack [54], resolution reduction from LoRe-Track [17], and layer pruning from CompressTracker [24]. Progressive token pruning retains accuracy (64.0 AUC) but provides negligible speedup (9.1 10.5 FPS on RPi5) due to its multi-stage, non-one-shot design. Resolution reduction and layer pruning achieve comparable FPS ( 24 FPS on RPi5) but at a severe accuracy cost of 5.5 and 4.9 AUC, respectively. Our one-shot token sparsification achieves the lowest MACs (836M) and the best AUC (63.8), within 0.2 of the dense baseline, while maintaining competitive speed (22.6 FPS on RPi5). This demonstrates that spatially-guided one-shot token sparsification ofers a clearly superior accuracy–eficiency trade-of compared to resolution reduction or model depth compression.

Visualization of the token sparsification process. As shown in Fig. 6, without a motion prior, raw attention scores are difuse at early layers, causing retained tokens to scatter across irrelevant background regions. Injecting the motion prior re-ranks scores toward the predicted target location, yielding compact, target-centered retention that directly reduces wasted computation on empty positions.

Table 8: Comparison with alternative model compression strategies under the same computational budget (≈1G MACs, ViT-Tiny backbone). “↓ Resolution” reduces the input image size to (48,96); “↓ Layer” compresses the encoder into 4 layers; our method applies motion-aware token sparsification at Layer 1 with 30% retention.
<table><tr><td>Compression Method</td><td>From</td><td>|MACs| (M)</td><td colspan="2">LaSOT AUC PNorm</td><td>P</td><td colspan="2">FPS RPi5 CPU Nano</td><td></td></tr><tr><td>Full model</td><td>I-</td><td>1752</td><td>64.0</td><td>74.2</td><td>68.3|</td><td>9.1</td><td>37</td><td>94</td></tr><tr><td>↓ Token (Progressive)</td><td>[OSTrack [54]</td><td>1293</td><td>64.0</td><td>74.1</td><td>68.1|</td><td>10.5</td><td>41</td><td>106</td></tr><tr><td>↓ Resolution</td><td>[LoReTrack [17]</td><td>990</td><td>58.5</td><td>67.7</td><td>59.4</td><td>23.6</td><td>83</td><td>176</td></tr><tr><td>↓ Layer</td><td>[CompressTracker [24]</td><td>937</td><td>59.1</td><td>68.5</td><td>60.1</td><td>24.1</td><td>89</td><td>177</td></tr><tr><td>↓ Token (One-shot)</td><td>Ours</td><td>836</td><td>63.8</td><td>73.6</td><td>67.9</td><td>22.6</td><td>76</td><td>152</td></tr></table>

![](images/7346b7f90f6652998788bac2b96d6695d5dfa464d81ba568b3085c2e202f9110.jpg)

![](images/1c9b72029003f6ed74cd92b450a3ef26e2ea7723d3b55f40adf37e2061869f6a.jpg)  
Fig. 6: Token sparsification visualization. Left (baseline) shows Search Region, Raw Importance, and Tokens Retained. Right (ours) shows Previous Prediction, Raw Importance, Motion Window, Re-ranked Importance, and Tokens Retained. Three tracking sequences are shown per row.

## 5 Conclusion

In this work, we present MaST, a sparse tracker that unifies motion-aware early token pruning with a natively sparse prediction head. Unlike prior methods that defer pruning to intermediate layers, a motion prior enables reliable one-shot reduction at the first encoder layer, letting all subsequent Transformer blocks operate on the compressed token set. A score-first, regress-once head decodes targets directly from retained tokens, eliminating dense reshape and redundant regression. Together, these designs achieve Pareto-optimal speed–accuracy trade ofs on edge platforms, showing that end-to-end sparsity is a practical path to real-time tracking. A remaining limitation is that high-resolution inputs still incur substantial attention computation before sparsification takes efect, motivating input-adaptive pruning as future work.

## Acknowledgements

This research is supported by the Major Key Project of Pengcheng Laboratory (PCL2025A13, PCL2025A08), the National Natural Science Foundation of China (U24B20151), and the Guangdong Major Project of Basic and Applied Basic Research (2019B030302002).

## References

1. Bertinetto, L., Valmadre, J., Henriques, J.F., Vedaldi, A., Torr, P.H.: Fullyconvolutional siamese networks for object tracking. In: European conference on computer vision. pp. 850–865. Springer (2016)

2. Bhat, G., Danelljan, M., Gool, L.V., Timofte, R.: Learning discriminative model prediction for tracking. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6182–6191 (2019)

3. Blatter, P., Kanakis, M., Danelljan, M., Van Gool, L.: Eficient Visual Tracking with Exemplar Transformers. In: WACV. pp. 1571–1581 (2023)

4. Bolya, D., Fu, C.Y., Dai, X., Zhang, P., Feichtenhofer, C., Hofman, J.: Token merging: Your ViT but faster. In: International Conference on Learning Representations (2023)

5. Borsuk, V., Vei, R., Kupyn, O., Martyniuk, T., Krashenyi, I., Matas, J.: FEAR: Fast, Eficient, Accurate and Robust Visual Tracker. In: ECCV. pp. 644–663 (2022)

6. Cao, Z., Fu, C., Ye, J., Li, B., Li, Y.: HiFT: Hierarchical feature transformer for aerial tracking. In: ICCV. pp. 15437–15446 (2021)

7. Cao, Z., Huang, Z., Pan, L., Zhang, S., Liu, Z., Fu, C.: Tctrack: Temporal contexts for aerial tracking. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 14778–14788 (2022). https://doi.org/10.1109/ CVPR52688.2022.01438

8. Chen, B., Li, P., Bai, L., Qiao, L., Shen, Q., Li, B., Gan, W., Wu, W., Ouyang, W.: Backbone is all your need: A simplified architecture for visual object tracking. In: European Conference on Computer Vision. pp. 375–392. Springer (2022)

9. Chen, X., Kang, B., Geng, W., Zhu, J., Liu, Y., Wang, D., Lu, H.: Sutrack: Towards simple and unified single object tracking. In: AAAI (2025)

10. Chen, X., Kang, B., Wang, D., Li, D., Lu, H.: Eficient Visual Tracking via Hierarchical Cross-Attention Transformer. In: ECCVW. pp. 461–477 (2022)

11. Chen, X., Yan, B., Zhu, J., Wang, D., Yang, X., Lu, H.: Transformer tracking. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8126–8135 (2021)

12. Cui, Y., Jiang, C., Wang, L., Wu, G.: Mixformer: End-to-end tracking with iterative mixed attention. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 13608–13618 (2022)

13. Cui, Y., Song, T., Wu, G., Wang, L.: Mixformerv2: Eficient fully transformer tracking. In: NeurIPS. pp. 58736–58751 (2023)

14. Danelljan, M., Bhat, G., Khan, F.S., Felsberg, M.: Atom: Accurate tracking by overlap maximization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4660–4669 (2019)

15. Danelljan, M., Bhat, G., Shahbaz Khan, F., Felsberg, M.: Eco: Eficient convolution operators for tracking. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6638–6646 (2017)

16. Danelljan, M., Gool, L.V., Timofte, R.: Probabilistic regression for visual tracking. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7183–7192 (2020)

17. Dong, S., Feng, Y., Liang, J., Yang, Q., Lin, Y., Fan, H.: Eficient and accurate low-resolution transformer tracking. In: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 20677–20684. IEEE (2025)

18. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2021)

19. Fan, H., Lin, L., Yang, F., Chu, P., Deng, G., Yu, S., Bai, H., Xu, Y., Liao, C., Ling, H.: Lasot: A high-quality benchmark for large-scale single object tracking. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019)

20. Gao, M., Jin, L., Jiang, Y., Guo, B.: Manifold siamese network: A novel visual tracking convnet for autonomous vehicles. IEEE Transactions on Intelligent Trans portation Systems 21(4), 1612–1623 (2020)

21. Gao, S., Zhou, C., Zhang, J.: Generalized relation modeling for transformer tracking. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18686–18695 (2023)

22. Gopal, G.Y., Amer, M.A.: Separable self and mixed attention transformers for eficient object tracking. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 6708–6717 (2024)

23. Henriques, J.F., Caseiro, R., Martins, P., Batista, J.: High-speed tracking with kernelized correlation filters. IEEE transactions on pattern analysis and machine intelligence 37(3), 583–596 (2014)

24. Hong, L., Li, J., Zhou, X., Yan, S., Guo, P., Jiang, K., Chen, Z., Gao, S., Li, R., Sheng, X., et al.: General compression framework for eficient transformer object tracking. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 13427–13437 (2025)

25. Huang, L., Zhao, X., Huang, K.: Got-10k: A large high-diversity benchmark for generic object tracking in the wild. IEEE Transactions on Pattern Analysis and Machine Intelligence 43(5), 1562–1577 (2021)

26. Kang, B., Chen, X., Lai, S., Liu, Y., Liu, Y., Wang, D.: Exploring enhanced contextual information for video-level object tracking. In: Proceedings of the AAAI conference on artificial intelligence. vol. 39, pp. 4194–4202 (2025)

27. Kang, B., Chen, X., Wang, D., Peng, H., Lu, H.: Exploring lightweight hierarchical vision transformers for eficient visual tracking. In: ICCV. pp. 9612–9621 (2023)

28. Kiani Galoogahi, H., Fagg, A., Huang, C., Ramanan, D., Lucey, S.: Need for speed: A benchmark for higher frame rate object tracking. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 1125–1134 (2017)

29. Li, B., Wu, W., Wang, Q., Zhang, F., Xing, J., Yan, J.: Siamrpn++: Evolution of siamese visual tracking with very deep networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4282–4291 (2019)

30. Li, S., Yang, Y., Zeng, D., Wang, X.: Adaptive and background-aware vision transformer for real-time uav tracking. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 13989–14000 (October 2023)

31. Li, S., Yeung, D.Y.: Visual object tracking for unmanned aerial vehicles: A benchmark and new motion models. In: AAAI (2017)

32. Li, Y., Liu, M., Wu, Y., Wang, X., Yang, X., Li, S.: Learning adaptive and viewinvariant vision transformer for real-time uav tracking. In: Forty-first International Conference on Machine Learning (2024)

33. Liang, Y., Ge, C., Tong, Z., Song, Y., Wang, J., Xie, P.: Not all patches are what you need: Expediting vision transformers via token reorganizations. arXiv preprint arXiv:2202.07800 (2022)

34. Lin, L., Fan, H., Zhang, Z., Wang, Y., Xu, Y., Ling, H.: Tracking meets lora: Faster training, larger model, stronger performance. In: ECCV. pp. 300–318. Springer (2024)

35. Lin, L., Fan, H., Zhang, Z., Xu, Y., Ling, H.: Swintrack: A simple and strong baseline for transformer tracking. NeurIPS 35, 16743–16754 (2022)

36. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., Zitnick, L.: Microsoft coco: Common objects in context. In: ECCV (September 2014)

37. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR. pp. 1–9 (2018)

38. Mueller, M., Smith, N., Ghanem, B.: A benchmark and simulator for uav tracking. In: ECCV. pp. 445–461. Springer (2016)

39. Muller, M., Bibi, A., Giancola, S., Alsubaihi, S., Ghanem, B.: Trackingnet: A large-scale dataset and benchmark for object tracking in the wild. In: Proceedings of the European Conference on Computer Vision (ECCV). pp. 300–317 (2018)

40. Peng, L., Gao, J., Liu, X., Li, W., Dong, S., Zhang, Z., Fan, H., Zhang, L.: Vasttrack: Vast category visual object tracking. Advances in Neural Information Processing Systems 37, 130797–130818 (2024)

41. Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., Hsieh, C.J.: Dynamicvit: Eficient vision transformers with dynamic token sparsification. Advances in neural information processing systems 34, 13937–13949 (2021)

42. Robin, C., Lacroix, S.: Multi-robot target detection and tracking: taxonomy and survey. Autonomous Robots 40(4), 729–760 (2016)

43. Tang, Q., Zhang, B., Liu, J., Liu, F., Liu, Y.: Dynamic token pruning in plain vision transformers for semantic segmentation. In: ICCV. pp. 777–786 (2023), https://doi.org/10.1109/ICCV51070.2023.00078

44. Wang, G., Lin, T., Bai, Y., Cao, A., Liang, S., Zhao, W., Wei, X.: Fartrack: Fast autoregressive visual tracking with high performance. ICLR (2026)

45. Wang, S., Gao, J., Li, Z., Zhang, X., Hu, W.: A closer look at self-supervised lightweight vision transformers. In: Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., Scarlett, J. (eds.) Proceedings of the 40th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 202, pp. 35624–35641. PMLR (23–29 Jul 2023), https://proceedings.mlr.press/v202/ wang23e.html

46. Wei, Q., Zeng, B., Liu, J., He, L., Zeng, G.: Litetrack: Layer pruning with asynchronous feature extraction for lightweight and eficient visual tracking. In: 2024 IEEE International Conference on Robotics and Automation (ICRA). pp. 4968–4975 (2024). https://doi.org/10.1109/ICRA57147.2024.10610022

47. Wei, X., Bai, Y., Zheng, Y., Shi, D., Gong, Y.: Autoregressive visual tracking. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9697–9706 (2023)

48. Wu, Y., Wang, X., Yang, X., Liu, M., Zeng, D., Ye, H., Li, S.: Learning occlusion robust vision transformers for real-time uav tracking. In: CVPR (2025)

49. Wu, Y., Wang, X., Zeng, D., Ye, H., Xie, X., Zhao, Q., Li, S.: Learning motion blur robust vision transformers with dynamic early exit for real-time uav tracking. arXiv preprint arXiv:2407.05383 (2024)

50. Xing, J., Ai, H., Lao, S.: Multiple human tracking based on multi-view upper-body detection and discriminative learning. In: 2010 20th International Conference on Pattern Recognition. pp. 1698–1701. IEEE (2010)

51. Yan, B., Peng, H., Fu, J., Wang, D., Lu, H.: Learning spatio-temporal transformer for visual tracking. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 10448–10457 (2021)

52. Yan, B., Peng, H., Wu, K., Wang, D., Fu, J., Lu, H.: Lighttrack: Finding lightweight neural networks for object tracking via one-shot architecture search. In: CVPR 2021 (June 2021)

53. Yao, L., Fu, C., Li, S., Zheng, G., Ye, J.: Sgdvit: Saliency-guided dynamic vision transformer for uav tracking. In: 2023 IEEE International Conference on Robotics and Automation (ICRA). pp. 3353–3359 (2023). https://doi.org/10. 1109/ICRA48891.2023.10161487

54. Ye, B., Chang, H., Ma, B., Shan, S., Chen, X.: Joint feature learning and relation modeling for tracking: A one-stream framework. In: European conference on computer vision. pp. 341–357. Springer (2022)

55. Yu, H., Li, G., Zhang, W., Huang, Q., Du, D., Tian, Q., Sebe, N.: The unmanned aerial vehicle benchmark: Object detection, tracking and baseline. Int. J. Comput. Vision 128(5), 1141–1159 (May 2020). https://doi.org/10.1007/s11263-019- 01266-1, https://doi.org/10.1007/s11263-019-01266-1

56. Zhang, G., Vela, P.A.: Good features to track for visual slam. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1373–1382 (2015)

57. Zhao, W., Han, Y., Tang, J., Li, Z., Song, Y., Wang, K., Wang, Z., You, Y.: A stitch in time saves nine: Small vlm is a precise guidance for accelerating large vlms. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19814–19824 (2025)

58. Zheng, J., Liang, M., Huang, S., Ning, J.: Exploring the feature extraction and relation modeling for light-weight transformer tracking. In: European Conference on Computer Vision. pp. 110–126. Springer (2025)

59. Zhu, J., Tang, H., Chen, X., Wang, X., Wang, D., Lu, H.: Two-stream beats onestream: Asymmetric siamese network for eficient visual tracking. In: AAAI. vol. 39, pp. 10959–10967 (2025)

## A Appendix

## A.1 Training Details.

Model Inputs. The tracker takes a pair of cropped images as input: a template patch Z and a search region patch X, both divided into non-overlapping 16 16 patches. Input sizes difer across model variants (see Tab. 1): MaST-nano and MaST-tiny use a template of 128  128 and a search region of 256  256, yielding 64 template tokens and 256 search tokens; MaST-small uses a larger template of 192  192 and a search region of 384  384, yielding 144 template tokens and 576 search tokens. The backbone is initialized with MAE-lite [45] pre-trained weights.

Augmentation. Each training sample is constructed as a template–search image pair sampled from the same video sequence. Standard data augmentations are applied, including random horizontal flipping and brightness jittering. The batch size is set to 128 image pairs.

Training Pipeline. Training proceeds in two stages. In Stage 1, we train a dense base model (no token sparsification) using the full ViT-Tiny backbone at $1 2 8 \times 1 2 8$ template and $2 5 6 \times 2 5 6$ search resolution. We use the AdamW optimizer [37] with a weight decay of $1 0 ^ { - 4 }$ , a backbone learning rate of $4 \times 1 0 ^ { - 5 }$ and $4 \times 1 0 ^ { - 4 }$ for all other parameters. Training runs for 300 epochs with 60,000 image pairs per epoch; the learning rate is decayed by a factor of 10 after epoch 240. This stage produces a single well-converged dense checkpoint shared by all variants.

In Stage 2, we fine-tune three variants from this checkpoint for 50 epochs each with sparsification enabled:

– MaST-tiny: fine-tuned directly from the Stage 1 checkpoint at the same 128/256 resolution.

MaST-small: the position encodings are bilinearly interpolated to accommodate the larger $1 9 2 \times 1 9 2$ template and 384 384 search resolution, then fine-tuned.

– MaST-nano: the last 4 encoder layers are removed and the remaining 8-layer backbone is fine-tuned at 128/256 resolution.

Sparsification Warmup. Across all Stage 2 variants, token sparsification is introduced with a gradual warmup. Starting from epoch 1, the token retention rate is linearly annealed from 100% down to the target rate (30% for all variants) over the first 10 epochs, and then held constant for the remainder of Stage 2 fine-tuning.

Motion Prior During Training. At test time, the motion window is centered on the bounding box predicted in the previous frame. During training, the search region is cropped with a random spatial jitter around the ground-truth target location, so the target is approximately centered in the search patch. We therefore simply set the motion window center to the center of the search image, which naturally approximates the previous-frame prediction without requiring any additional placeholder or annotation. The motion window is enabled throughout Stage 2 (including during the sparsification warmup epochs).

## A.2 Analysis of Motion Window.

Tab. 9 presents an ablation study of various windowing strategies used in the proposed framework. We compare two window types (Gaussian and Cosine), evaluate the efect of incorporating cross-attention importance scores, and analyze whether dynamic window resizing (based on the width and height of the previous frame’s bounding box) improves performance. Results show that combining importance scores with a Gaussian window achieves the best overall performance. The fiexd windowing approach yields sub-optimal perfrmance compared to dynamic window sizes. The last two rows highlight that using only motion windows for sparsification performs better than using only cross-attention scores, but their combination consistently outperforms either individually. This confirms that both the spatial prior and cross-attention scores are crucial for efective token selection.

Table 9: Comparison of diferent windowing strategies on GOT-10k val. We evaluate the impact of window type (Gaussian or Cosine), the use of importance scores, and whether the window dynamically adapts to the previous frame’s bounding box size and aspect ratio.
<table><tr><td>Window Type</td><td>|Importance Score</td><td>Dynamic</td><td>AUC P  $\mathrm { P } _ { N o r m }$ </td></tr><tr><td>Gaussian</td><td>√</td><td>√</td><td>79.971.7 91.5</td></tr><tr><td>Gaussian</td><td>√</td><td></td><td>79.5 70.5 90.2</td></tr><tr><td>Cosine</td><td>√</td><td>√</td><td>79.8 71.1 91.2</td></tr><tr><td>Cosine</td><td>√</td><td></td><td>79.1 70.2 89.9</td></tr><tr><td></td><td>√</td><td></td><td>76.9 68.2 86.5</td></tr><tr><td>Gaussian</td><td>一</td><td>了</td><td>78.2 69.1 88.1</td></tr></table>

Source of the appearance importance score. We additionally compared the center template token used by MaST with max-pooling over all template tokens, following the aggregation considered in GRM [21]. Template max-pooling was slightly less accurate in our experiments and introduced an extra reduction operation with marginal latency. This is consistent with the target-centered construction of tracking templates: the center token is usually a clean foreground cue, whereas pooling over the full template can mix target evidence with background context. We therefore use the center template token as the default appearance-score source.

Table 10: Results of scaling to a ViT-Base model and increasing input resolution. Accuracy (Acc) and FPS are reported.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>|Input</td><td rowspan=1 colspan=1>|%Tokens Retaine</td><td rowspan=1 colspan=1>d|LaSOT GOT-10kAUC    AO</td><td rowspan=1 colspan=1>FPS|(GPU)</td></tr><tr><td rowspan=3 colspan=1>ViT-base</td><td rowspan=1 colspan=1>2242</td><td rowspan=1 colspan=1>100%30%</td><td rowspan=1 colspan=1>71.3    71.969.8    70.2</td><td rowspan=1 colspan=1>116215</td></tr><tr><td rowspan=2 colspan=1>3782</td><td rowspan=2 colspan=1>100%30%20%10%</td><td rowspan=2 colspan=1>72.1    73.370.4    71.568.9    70.667.2    68.4</td><td rowspan=1 colspan=1>51104</td></tr><tr><td rowspan=1 colspan=1>119126</td></tr></table>

Table 11: State-of-the-art comparison on three UAV-oriented tracking benchmarks. Methods above the double rule are traditional correlation-filter trackers; those below are deep trackers. Red and blue indicate the best and second-best results overall.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pub.</td><td>DTB70</td><td>UAVDT</td><td>UAV123</td><td rowspan="2">[Param. MACs| (M)</td><td rowspan="2">(G)</td><td colspan="2" rowspan="2">FPS</td></tr><tr><td>SUC P</td><td>SUC P</td><td>SUC P</td><td>RPi Nano</td></tr><tr><td>HiFT [6]</td><td>ICCV&#x27;21</td><td>59.480.2</td><td>|47.5 65.2|</td><td>|59.0 78.7|</td><td>15.9</td><td>5.48</td><td>5.5</td><td>57</td></tr><tr><td>TCTrack [7]</td><td>CVPR’22</td><td>62.281.2</td><td>53.072.5</td><td>60.5 80.0</td><td>9.7</td><td>8.8</td><td>–</td><td>=</td></tr><tr><td>SGDViT [53]</td><td>ICRA&#x27;23</td><td>60.4 78.5</td><td>48.065.7</td><td>57.5 75.4</td><td>23.3</td><td>11.3</td><td>–</td><td></td></tr><tr><td>Aba-ViTrack-DeiT [30]</td><td>ICCV&#x27;23</td><td>66.4 85.9</td><td>59.983.4</td><td>66.4 86.4</td><td>7.98</td><td>2.39</td><td>7.9</td><td>83</td></tr><tr><td>SMAT [22]</td><td>WACV&#x27;24</td><td>63.881.9</td><td>58.7 80.8</td><td>64.681.8</td><td>3.76</td><td>2.49</td><td>7.7</td><td>77</td></tr><tr><td>AVTrack-DeiT [32]</td><td>ICML&#x27;24</td><td>65.084.3</td><td>58.7 82.1</td><td>66.8 84.8</td><td>7.98</td><td>2.39</td><td>7.9</td><td>83</td></tr><tr><td>ORTrack-DeiT [48]</td><td>CVPR’25</td><td>66.4 86.2</td><td>60.1 83.4</td><td>66.484.3</td><td>7.98</td><td>2.39</td><td>7.9</td><td>84</td></tr><tr><td>ORTrack-D-DeiT [48]</td><td>CVPR&#x27;25</td><td>65.1 83.7</td><td>59.7 82.5</td><td>66.184.0</td><td>5.31</td><td>1.54</td><td></td><td>18.5138</td></tr><tr><td>MaST-nano</td><td>Ours</td><td>61.8 79.9</td><td>55.1 75.3</td><td>62.9 82.4</td><td>3.85</td><td>0.585</td><td></td><td>30.1 230</td></tr><tr><td>MaST-tiny</td><td>Ours</td><td>63.3 81.9|</td><td>57.5 77.8</td><td>66.6 86.9</td><td>5.63</td><td>0.836</td><td></td><td>22.6152</td></tr><tr><td>MaST-small</td><td>Ours</td><td>66.2 85.6</td><td>63.0 85.9</td><td>66.4 86.2</td><td>5.63</td><td>1.82</td><td>7.5</td><td>98</td></tr></table>

## A.3 Scaling to Larger Models

Experiments shown in Tab. 10 examine the scalability of our motion-aware sparsification by applying it to a ViT-Base tracker (LoRAT [34]) and increasing the input resolution from $2 2 4 ^ { 2 }$ to 378<sup>2</sup>. At 224<sup>2</sup>, 30% retention nearly doubles throughput from 116 to 215 FPS while attaining 69.8 LaSOT AUC and 70.2 GOT-10k AO. At 378<sup>2</sup>, it more than doubles throughput from 51 to 104 FPS while attaining 70.4 AUC and 71.5 AO. These results place the sparse model in the lower accuracy range of heavier trackers and show that MaST’s speedup is not tied to tiny backbones.

## A.4 Additional Details on Global Box Decoding.

We explored two global localization alternatives that predict box coordinates directly from queries or tokens. Loc Tokens appends four box tokens to the encoder, following MixFormerV2 [13]; Trans. Decoder uses four queries that crossattend to the search tokens after the encoder and then decode box coordinates. Both designs are naturally compatible with sparse tokens, but they converged much more slowly in image-pair IoU. Training for 3 more epochs, replacing the head of a converged tracker, and teacher distillation did not resolve this optimization dificulty. Their LaSOT AUC remained at 57.9–60.1, compared with 63.8 for our local anchor-based sparse head. These attempts motivate preserving the established local anchor definition while removing only its dense implementation.

Table 12: State-of-the-art comparisons of eficient tracking on three large-scale bench marks, grouped by computational cost. Red and blue denote the best and second-best results within each group. Gray FPS values indicate speeds below VOT realtime setting (20 FPS).
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pub.</td><td colspan="3">LaSOT TrackingNet GOT-10k</td><td rowspan="2">[Params MACs| (M)</td><td rowspan="2">(G)</td><td colspan="5">FPS</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td colspan="2"></td><td colspan="3">RPi5 M4 14900 Nano 3090</td></tr><tr><td></td><td>21.1</td><td>27.9</td><td>=</td><td>142 =</td><td>946</td><td>679</td><td>2034</td><td></td></tr><tr><td>KCF [23] LightTrack [52]</td><td>TPAMI&#x27;14 CVPR&#x27;21</td><td>53.8</td><td>= 72.5</td><td>61.1</td><td>1.96</td><td>0.483</td><td>13.4</td><td>44</td><td>260</td><td>124</td><td>398</td></tr><tr><td>FEAR-XS [5]</td><td>ECCV&#x27;22</td><td>53.5</td><td>=</td><td>61.9</td><td>6.21</td><td>0.532</td><td>19.5</td><td>59</td><td>453</td><td>146</td><td>650</td></tr><tr><td>E.T.Track [3]</td><td>WACV&#x27;23</td><td>59.1</td><td>75.0</td><td>=</td><td>6.98</td><td>1.56</td><td>8.8</td><td>24</td><td>100</td><td>71</td><td>194</td></tr><tr><td>HiT-Tiny [27]</td><td>ICCV&#x27;23</td><td>54.8</td><td>74.6</td><td>52.6</td><td>9.59</td><td>0.995</td><td>27.3</td><td>97</td><td>176</td><td>118</td><td>253</td></tr><tr><td>HiT-Small [27]</td><td>ICCV&#x27;23</td><td>60.5</td><td>77.7</td><td>62.6</td><td>11.02</td><td>1.13</td><td>21.5</td><td>78</td><td>144</td><td>106</td><td>279</td></tr><tr><td>HiT-Base [27]</td><td>ICCV&#x27;23</td><td>64.6</td><td>80.0</td><td>64.0</td><td>42.14</td><td>4.34</td><td>6.7</td><td>22</td><td>64</td><td>79</td><td>337</td></tr><tr><td>FERMT [58]</td><td>ECCV&#x27;24</td><td>65.1</td><td>80.8</td><td>69.6</td><td>7.98</td><td>2.31</td><td>7.9</td><td>31</td><td>99</td><td>84</td><td>231</td></tr><tr><td>MixformerV2-S [13]</td><td>NeurIPS’23</td><td>60.6</td><td>75.8</td><td>61.9</td><td>16.05</td><td>4.40</td><td>7.1</td><td>21</td><td>72</td><td>89</td><td>415</td></tr><tr><td>LiteTrack-B4 [46]</td><td>ICRA&#x27;24</td><td>62.5</td><td>79.9</td><td>65.2</td><td>26.48</td><td>6.78</td><td>3.6</td><td>13</td><td>36</td><td>45</td><td>376</td></tr><tr><td>ORTrack-D [48]</td><td>CVPR’25</td><td>54.6</td><td>73.7</td><td>-</td><td>3.76</td><td>1.54</td><td>18.5</td><td>52</td><td>174</td><td>138</td><td>480</td></tr><tr><td>CompressTracker [24]</td><td>ICCV&#x27;25</td><td>60.4</td><td>78.2</td><td>=</td><td>21.24</td><td>6.38</td><td>3.9</td><td>17</td><td>56</td><td>97</td><td>605</td></tr><tr><td>AsymTrack-T [59]</td><td>AAAI&#x27;25</td><td>60.8</td><td>76.2</td><td>62.3</td><td>3.05</td><td>0.708</td><td>18.1</td><td>60</td><td>186</td><td>99</td><td>308</td></tr><tr><td>AsymTrack-S [59]</td><td>AAAI&#x27;25</td><td>62.8</td><td>77.9</td><td>65.5</td><td>3.36</td><td>0.806</td><td>15.6</td><td>51</td><td>165</td><td>88</td><td>269</td></tr><tr><td>AsymTrack-B [59]</td><td>AAAI&#x27;25</td><td>64.7</td><td>80.0</td><td>67.7</td><td>3.36</td><td>1.81</td><td>6.2</td><td>25</td><td>79</td><td>57</td><td>265</td></tr><tr><td>FARTrackpico [44]</td><td>ICLR&#x27;26</td><td>58.6</td><td>75.6</td><td>62.8</td><td>2.81</td><td>1.08</td><td></td><td>17.2 35.3</td><td>=</td><td>134.9</td><td></td></tr><tr><td>FARTracknano [44]</td><td>ICLR&#x27;26</td><td>61.3</td><td>79.1</td><td>69.9</td><td>4.59</td><td>1.78</td><td></td><td>10.4 25.6</td><td>=</td><td>117.3</td><td></td></tr><tr><td>FARTracktiny [44]</td><td>ICLR&#x27;26</td><td>63.2</td><td>80.7</td><td>70.6</td><td>6.82</td><td>2.65</td><td></td><td>6.9 16.1</td><td>=</td><td>87</td><td>=</td></tr><tr><td>MaST-nano</td><td>Ours</td><td>58.6</td><td>77.2</td><td>61.6</td><td>3.85</td><td>0.585</td><td></td><td>30.1 101</td><td>247</td><td>230</td><td>556</td></tr><tr><td>MaST-tiny</td><td>Ours</td><td>63.8</td><td>80.1</td><td>66.6</td><td>5.63</td><td>0.836</td><td>22.6</td><td>72</td><td>184</td><td>152</td><td>453</td></tr><tr><td>MaST-small</td><td>Ours</td><td>65.8</td><td>82.3</td><td>70.0</td><td>5.63</td><td>1.82</td><td>7.5</td><td>30</td><td>88</td><td>98</td><td>428</td></tr></table>

## A.5 Comparison on UAV Benchmarks.

Tab. 11 provides a comprehensive comparison on three UAV-specific benchmarks. On UAVDT [55], MaST-small achieves the highest success (63.0) and precision (85.9), surpassing ORTrack-DeiT (60.1 / 83.4) by +2.9 / +2.5 while consuming only 1.82 G MACs versus 2.39 G; on Jetson Orin Nano, MaST-small also runs faster at 98 FPS versus 84 FPS. On UAV123 [38], MaST-tiny attains the highest precision of 86.9, outperforming Aba-ViTrack-DeiT (86.4) and AVTrack-DeiT (84.8). It also achieves the second-highest success of 66.6, all with only 0.836 G MACs. On edge devices, MaST-tiny runs at 22.6 FPS on Raspberry Pi 5 and 152 FPS on Jetson Orin Nano. On DTB70 [31], MaST-small achieves the second-highest success of 66.2 (behind the 66.4 shared by Aba-ViTrack-DeiT and ORTrack-DeiT) and a precision of 85.6, demonstrating competitive accuracy on this challenging aerial benchmark. MaST-nano is the lightest variant at 0.585 G MACs. It achieves 30.1 FPS on RPi5 and 230 FPS on Jetson Nano, the highest edge throughput among all compared methods, while maintaining solid accuracy across all three benchmarks. These results confirm that MaST’s motion-aware sparse computation transfers favorably to UAV deployment scenarios, delivering a strong accuracy–eficiency trade-of on both general and edge hardware.

## A.6 Running Latency on More Platforms.

Our primary focus is on real-world deployment, particularly on edge devices where power consumption and thermal constraints are critical. High-power GPUs and desktop CPUs are convenient for ofline benchmarking but rarely represent embedded or mobile deployment conditions. Critically, high throughput on powerful hardware does not imply eficiency on low-power devices. Server-grade GPUs and desktop CPUs have large L2/L3 caches and abundant on-chip SRAM. These resources hide irregular memory-access latency and keep arithmetic units saturated. Edge platforms such as Raspberry Pi 5 and Jetson Orin Nano have no such bufer. Their small caches and limited memory bandwidth penalize operations that are cache-friendly on high-end hardware but bandwidth-bound on constrained devices. This disparity is evident in Tab. 12. CompressTracker reaches 605 FPS on an RTX 3090 but only 3.9 FPS on Raspberry Pi 5, well below the VOT realtime threshold. LiteTrack-B4 similarly reaches 376 FPS on GPU but drops to 3.6 FPS on-device. MaST avoids dense operations that scale poorly on cache-limited hardware. MaST-nano and MaST-tiny sustain 30.1 and 22.6 FPS on Raspberry Pi 5 respectively, while remaining competitive or superior in accuracy.

## A.7 NFS under Diferent Input Frame Rates.

Tab. 13 compares MaST-tiny with OSTrack-Tiny under the 30 FPS and 240 FPS evaluation protocols on NFS. At 30 fps, MaST-tiny already outperforms OSTrack-Tiny by $+ 1 . 1 \mathrm { ~ A U C } , + 0 . 6 \mathrm { ~ P } _ { N o r m } .$ , and +0.3 P. At 240 $f p s ,$ the margin widens to $+ 2 . 9 \mathrm { \ A U C } , + 4 . 0 \mathrm { \ P } _ { N o r m }$ , and +4.0 P. Higher frame rates yield smaller inter-frame displacement and a clearer motion prior, allowing MaST to guide token selection more accurately than appearance-only methods.

The comparison also highlights the importance of tracker latency. From 30 to 240 fps, OSTrack-Tiny improves only marginally (+0.7 AUC), while MaST-tiny gains +2.5 AUC. A motion-aware tracker can better exploit improved temporal continuity when its inference latency is low. MaST-tiny runs at 152 FPS on Jetson Nano versus 83 FPS for OSTrack-Tiny, making high-frame-rate online tracking more practical in real deployments.

## A.8 Robustness to Occlusion and Abrupt Motion.

MaST does not rely on the motion prior alone. As shown by the main-paper sparsification ablation, fusing appearance attention with motion reaches 63.8 LaSOT AUC, compared with 62.6 for motion alone and 60.5 for attention alone. The Gaussian prior is a soft re-weighting, so strong of-center appearance evidence can still retain tokens outside the most likely motion region.

Fig. 7 illustrates the view–token-budget trade-of. After the bus is occluded, MaST-tiny may retain mainly the visible front of the vehicle and localize only its head. A larger search view gives MaST-small a wider candidate field and improves the challenging-attribute results. This suggests that long occlusion and large displacement are chiefly constrained by the local search crop and its token budget, rather than by a hard dependence on the motion center.

Table 13: Comparison on NFS [28] under diferent input frame-rate settings. We compare MaST-tiny with OSTrack-Tiny under the standard 30 FPS and 240 FPS protocols. Higher input frame rates provide a more stable motion prior, while lower tracker latency improves the ability to exploit this temporal continuity in practice. Please fill in the numeric entries.
<table><tr><td>Method</td><td> $@ 3 0 \ f p s$   $\mathrm { A U C } \operatorname* { P } _ { N o r m }$ </td><td>P</td><td></td><td> $\textcircled { \mathrm { a } } 2 4 0 ~ f p s$   $\mathrm { A U C } \operatorname* { P } _ { N o r m }$ </td><td>P</td><td>|Nano FPS</td></tr><tr><td>OSTrack-Tiny [54]</td><td>65.1</td><td>81.2</td><td>78.4|</td><td>65.8 81.3</td><td>78.5|</td><td>83</td></tr><tr><td>MaST-tiny</td><td>66.2</td><td>81.8</td><td>78.7</td><td>68.7 85.3</td><td>82.5</td><td>152</td></tr></table>

![](images/7b2101cc1af901f33aa5fbbda45307748f9c1621d8bb906f64ee04ed33dac8b5.jpg)

![](images/a51867ab541d03de03930e41ff1b76d031ed79adf74531bc8adb4114dbed1c92.jpg)  
Fig. 7: Robustness analyses. Left: per-frame IoU on the LaSOT bus-2 sequence, with representative predictions around an occlusion. Right: AUC on challenging LaSOT attributes. Under the same 4× search factor and 256-pixel search input as FERMT, MaST-tiny is less robust to severe occlusion and fast motion; increasing the candidate field to a 5× factor and 384-pixel input allows MaST-small to surpass FERMT while using much less computation.

## A.9 More Visualizations.

1. Rapid Motion Scenario. The visualization of Fig. 9 demonstrates that MaST (30% tokens with motion window) can robustly track rapidly moving objects, closely matching the performance of the full-token model (100% tokens). In contrast, a naive sparsification strategy without the motion-aware component fails completely. This clearly illustrates the crucial role of our motion-aware sparsification in handling rapid motion cases.

2. Failure Cases under Extremely Challenging Conditions. We acknowledge there exist extreme conditions, e.g., low-quality UAV footage with significant appearance ambiguity in Fig. 10, that surpass the capability of even the full-token model. Such dificulties are fundamentally challenging for existing state-of-the-art trackers, not limited to MaST. Therefore, performance degradation under these conditions stems from inherent limitations common to the tracking domain rather than from sparsification or motion-aware strategies.

![](images/1e5ec202c65f2f0092bebdb87e742bec2c49afa797ed38b00013c315144fb885.jpg)  
Fig. 8: Token keep frequency on GOT-10k val. We align search crops to a common grid and accumulate binary keep/drop decisions. Warmer cells are retained more often. The product of appearance attention and the soft motion prior concentrates computation near the likely target while retaining of-center evidence.

## A.10 Limitations.

MaST inherits the local search region paradigm that is standard across modern one-stream and two-stream trackers. This design choice means that target loss under fast motion or prolonged occlusion remains an inherent challenge shared with the broader tracking community, rather than a problem introduced by our motion-aware sparsification. When the inter-frame displacement exceeds the search region radius, or when the target is occluded for an extended period, the tracker cannot relocate the target beyond the local window. This holds regardless of how tokens are selected within it. Looking forward, we see motion-aware sparse computation itself as a promising tool to address this limitation: by concentrating computation on a small set of informative tokens, the saved budget could be redistributed to extend the efective search area or to maintain a lightweight global context stream, enabling wider spatial coverage without increasing the overall computational cost. The retention rate is currently fixed for predictable latency. Our rate-sweep experiment shows mild adaptability from a single checkpoint; a stronger policy could vary the rate per frame using target confidence, motion uncertainty, or a device-level compute budget.

![](images/d1838f96d5d1e6f52fb9027faa9d49d7ecc622f347b4c3cb0cd23df0c59a3fd5.jpg)  
Fig. 9: Tracking a robot when it engage fast movement, leading to partially out-of-view. Sparsification without motion-aware fixing would quickly discard the features of the target, making the tracker focus on other objects in the view.

![](images/8d8d9d3e287a94c09b27c93ed9f068903afdee586822ad685f0ff85133a11b0b.jpg)  
Fig. 10: Handling small objects like UAVs under extremely bad situation. With or without sparsification or motion-aware sparsification, the tracker is dificult to keep tracking the target.