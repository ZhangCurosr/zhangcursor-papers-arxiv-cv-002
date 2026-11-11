# RS-JEPA: Remote Sensing Joint-Embedding Predictive Architecture with Scan-Path Views for Cross-Sensor Image Retrieval

Tanishq Rachamalla<sup>1</sup> Srishti Kaushik<sup>1</sup> Muhammad Attique Khan<sup>1,</sup> Gustau Camps-Valls<sup>2</sup> Swalpa Kumar Roy<sup>3</sup>

<sup>1</sup>Prince Mohammad Bin Fahd University, Al Khobar, Saudi Arabia <sup>2</sup>Universitat de Valencia, Valencia, Spain \` <sup>3</sup>Tezpur University, Assam, India

tanishqrachamalla12@gmail.com, kaushiksrishti108@gmail.com, attique.khan@ieee.org,

gustau.camps@uv.es, swalpa@tezu.ernet.in

## Abstract

Cross-sensor remote sensing image retrieval aims tofind scenes with the same content across heterogeneous sensors, such as SAR and optical or panchromatic (PAN) and multispectral (MS) imagery, without manual labels. Existing self-supervised methods either reconstruct sensor-specific pixel detail, align each image only with its co-registered twin, or rely on large transformers with 118–173M parameters (and up to 398M). We introduce RS-JEPA, a compact joint-embedding predictive framework built on a Vision Mamba encoder shared by both sensors. Since a state-space encoder reads an image as an ordered sequence of patches, we propose scan-path views, which treat two reading orders of the same image as two views with identical pixels and predict the hidden patches in one order from the visible patches in the other.To keep this prediction challenging, we propose Adversarial View Coupling (AVC), which learns the hardest split of the visible patches, and a neighbourhood consensus loss that aligns the scene similarities of both sensors. Extensive experiments on SAR– optical (BEN-14K, CBRSIR-VS) and PAN–MS (DSRSID) benchmarks demonstrate that RS-JEPA achieves the highest F1@5 among twelve re-implemented methods (72.92, 80.35 and 94.43) with a retrieval network of only 3.88M parameters (14.57M in training, 8–16× fewer than prior JEPA retrievers). Like prior methods, RS-JEPA trains on co-registered pairs. Our results highlight reading order as a source of self-supervised views. code https:// github.com/Tanishq251/RSJEPA.

## 1. Introduction

Synthetic aperture radar (SAR) images the ground through cloud and at night, whereas optical sensors provide detailed multispectral or panchromatic images only under clear skies [6, 14, 23]. Cross-sensor image retrieval ranks the images of one sensor by their content similarity to a query from another [20]. This is difficult because the same scene looks very different across sensors, labels are scarce at archive scale, and every archive image must be embedded, which favours small encoders. Masked image models [5, 12, 16] reconstruct pixels and thus also learn sensor noise such as SAR speckle. Cross-sensor alignment models [8,28] match each image with its co-registered counterpart but ignore how different scenes relate. Jointembedding predictive architectures (JEPAs) [1] predict features instead of pixels, and X-JEPA [4] and CR-JEPA [13] lead label-free retrieval on BEN-14K this way, but rely on large transformers that are costly to run over whole archives. State-space models (SSMs) [9, 31] are far smaller, yet they read an image as an ordered sequence, while existing selfsupervised views were designed for order-agnostic attention.

![](images/196a1147a2afd356d3255414ea103549084495c44948607a7f9e667cd23aa693.jpg)  
Figure 1. Accuracy against model size. Mean F1@5 over the three benchmarks against retrieval throughput (marker area: parameters).

Can we design a compact cross-sensor retrieval model that learns what a scene contains, regardless of the sensor, simply by reading it in two different orders?

A selective SSM summarises only the tokens it has already read, so its output for a patch depends on how the patch grid is flattened [9, 31]. Two traversals of one image therefore give two feature sequences from identical pixels, which is a natural pair of views. Only scene content stays predictable across paths; speckle and other sensor texture do not. Since prediction happens in feature space and the views come from the encoder itself, no image editing or momentum encoder is needed.

Existing vision SSMs fix the traversal or combine several directions [15, 18], discarding reading order before it can supervise learning. Self-supervised remote sensing methods build views by masking, augmentation or sensor pairing [1, 8, 28], assuming order-insensitive encoders. To our knowledge, no prior work has shown whether reading order can generate views for cross-sensor joint-embedding prediction, or how to train such views for retrieval.

Two complementary challenges must be addressed together. First, two scan paths over the same visible patches share much local context, so the predictor can exploit spatial continuity instead of modelling the scene, making the task too easy. Second, instance-level contrastive alignment pulls each scene towards its twin but leaves other scenes unconstrained, so the two sensors may induce different neighbourhoods, which directly harms retrieval.

To address these challenges, we introduce RS-JEPA, a remote sensing JEPA that learns cross-sensor features from the reading order of one Vision Mamba encoder shared by both sensors. Its scan-path views encode the visible patches along one traversal and predict the stop-gradient features of the hidden patches along another. For the first challenge, Adversarial View Coupling (AVC) learns, through gradient reversal, the split of visible patches that is hardest to predict and couples the two halves with cross-view attention. For the second, a neighbourhood consensus loss aligns the same- and cross-sensor similarity distributions of both sensors. RS-JEPA achieves the highest mean F1@5 among twelve re-implemented methods on three benchmarks with a retrieval network of only 3.88M parameters.

Our key contributions are as follows:

• We propose RS-JEPA, a compact joint-embedding predictive framework for label-free cross-sensor retrieval with one shared Vision Mamba encoder.

• We introduce scan-path views, which predict the hidden patches in one order from the visible patches in another, without image editing or a momentum encoder.

• We design Adversarial View Coupling (AVC), which learns the hardest split of the visible patches into two views, and a teacher-free neighbourhood consensus loss that aligns the scene similarities of both sensors.

• Extensive experiments on BEN-14K, CBRSIR-VS and DSRSID demonstrate state-of-the-art retrieval performance with a substantially smaller model.

## 2. Related Work

Self-Supervised Learning in Remote Sensing: Masked image modelling [12] has been adapted to satellite imagery with spectral and temporal grouping [5], multiple scales [16, 17, 22] and rotated windows [26]. For radar–optical pairs, CROMA [8] combines redrawing with cross-sensor matching, DeCUR [28] separates shared from sensor-specific features, and others use self-distillation [29] or fusion [2]; SkySense [10] and RoMA [27] scale this to foundation models. They target fine-tuning for classification or segmentation, not small retrieval encoders.

Cross-Sensor Retrieval: Early methods learned binary hash codes for radar and optical images from labels [21,30] on paired benchmarks such as BigEarthNet-MM [19, 20]. CSMAE [11] removed the labels with cross-sensor masked autoencoders, and X-JEPA [4] and CR-JEPA [13] replaced pixel redrawing with cross-sensor feature prediction in large transformers; RS-JEPA uses a compact state-space encoder. While REJEPA [3] predicts masked patch embeddings within a single image for same-sensor retrieval, RS-JEPA uses adversarially split scan views and symmetric neighbourhood consensus to align scene similarities across radar and optical embeddings.

Views and Relations in Self-Supervised Learning: Rather than relying on heuristic image augmentations or asymmetric teacher-student models [1, 7], RS-JEPA couples adversarial view splitting (AVC) with symmetric neighbourhood consensus to align cross-sensor similarity geometry without external guidance.

State-Space Models for Vision: Vision Mamba (Vim) [31] adapts selective state-space models [9] to vision by traversing patch sequences bidirectionally, while VMamba [15] incorporates four-way scanning directions, and autoregressive models [18] enforce sequential ordering. In prior works, the scan trajectory is treated as a fixed architectural choice; RS-JEPA instead leverages multi-directional scan paths as an intrinsic source of multiview self-supervision.

## 3. Proposed Method

In this section, we present RS-JEPA, which learns crosssensor retrieval features without labels by predicting the features of hidden patches, rather than their pixels, with a single Vision Mamba encoder shared by both sensors.

## 3.1. Overview

Given M unlabeled, co-registered pairs $\{ ( x _ { i } ^ { ( 1 ) } , x _ { i } ^ { ( 2 ) } ) \} _ { i = 1 } ^ { M }$ from sensors S1 (e.g., SAR or panchromatic) and S2 (e.g., optical or multispectral), indexed by $m \in \{ 1 , 2 \}$ , we seek an encoder whose unit-norm embeddings rank semantically related scenes first under cosine similarity in the same-sensor (S1→S1, S2→S2) and cross-sensor (S1→S2, S2→S1) directions. A sensorspecific stem (strided convolution plus learned position embeddings $\mathbf { p } _ { j } )$ maps each image $x ^ { ( m ) }$ to N patch tokens $\mathbf { Z } ^ { ( m ) } \in \bar { \mathbb { R } } ^ { \bar { N } \times \bar { D } }$ , and a random mask splits them into a visible set V and a hidden set H with $| \mathcal { H } | ~ \approx ~ r N$ for masking ratio $r ,$ giving $\mathbf { Z } _ { \nu }$ and $\mathbf { Z } _ { \mathcal { H } }$ . As shown in Fig. 2, a shared 12-block Vision Mamba (Vim) encoder $f _ { \theta }$ [31] of width $D$ reads the visible tokens along one scan path, and a per-sensor predictor predicts the hidden tokens as the same encoder reads them along that path and along another (Sec. 3.2). AVC makes this prediction harder (Sec. 3.3), and retrieval heads on the pooled features align the two sensors (Secs. 3.4 and 3.5). The encoder and heads are shared by both sensors, whereas the stems, AVC modules and predictors are per sensor; sg[·] denotes stop-gradient.

## 3.2. Scan-Path Views

Since a state-space encoder reads an image as an ordered sequence, two traversals of the same image yield two views with identical pixels, which we use as self-supervised views without image editing.

Order dependence of selective scans. A selective statespace layer processes a token sequence $( s _ { 1 } , \ldots , s _ { L } )$ with the input-dependent recurrence

$$
h _ { t } = \bar { A } _ { t } h _ { t - 1 } + \bar { B } _ { t } s _ { t } , \qquad y _ { t } = \Gamma _ { t } h _ { t } ,\tag{1}
$$

where $h _ { t }$ is the hidden state, and the discretised $\bar { A } _ { t } , \bar { B } _ { t }$ and the readout $\Gamma _ { t }$ are functions of $s _ { t }$ . Because $h _ { t }$ summarises exactly the tokens that precede t, the output for a patch depends on the order in which the 2-D grid is flattened, and running the recurrence in both directions, as Vim does, still depends on the chosen traversal.

Path-ordered encoding (Fig. 3a). Let G be a set of four traversals of the patch grid: row-major, column-major, and a mirrored version of each (row-major mirrored left to right, column-major mirrored top to bottom). Because Vim also reads each traversal in reverse, they give four distinct scan dynamics. Each $g \in { \mathcal { G } }$ assigns every patch $j$ a rank $\rho _ { g } ( j )$ Encoding a token subset along $^ { g , }$ written $f _ { \theta } ( \cdot ; g )$ , sorts the tokens by $\rho _ { g }$ of their patch indices, applies the trunk, and restores the original order. Position embeddings are added before sorting, so only the scan order changes. Per image, the context path $g _ { 1 }$ is drawn uniformly from ${ \mathcal { G } } ,$ and

$$
\mathbf { C } = f _ { \theta } ( \mathbf { Z } _ { \mathcal { V } } ; g _ { 1 } ) , \qquad \mathbf { T } ^ { g } = \mathrm { s g } \big [ f _ { \theta } ( \mathbf { Z } _ { \mathcal { H } } ; g ) \big ] \ \forall g \in \mathcal { G } .\tag{2}
$$

The targets are the hidden tokens encoded on their own by the same trunk, without a momentum encoder; this costs one gradient-free trunk pass over the hidden tokens per path during training and nothing at inference. The second path is the one whose targets disagree most with those of $g _ { 1 } \colon$

$$
g _ { 2 } = \underset { g \in \mathcal { G } \backslash \{ g _ { 1 } \} } { \arg \operatorname* { m a x } } \ \frac { 1 } { | \mathcal { H } | D } \left. \mathbf { T } ^ { g } - \mathbf { T } ^ { g _ { 1 } } \right. _ { F } ^ { 2 } .\tag{3}
$$

In an earlier configuration (50% masking, without AVC), a random $g _ { 2 } \ \ne \ g _ { 1 }$ performed comparably on BEN-14K (72.63 vs. 72.56 F1@5), so we do not count this rule as a contribution.

Path-conditioned prediction (Fig. 3b). The predictor $P ^ { ( m ) }$ of sensor m is a transformer [25] whose queries for the hidden patches are ${ \bf q } _ { j } = { \bf z } _ { \mathrm { m a s k } } + { \bf p } _ { j } + { \bf e } _ { g }$ for $j \in \mathcal { H }$ where $\mathbf { z } _ { \mathrm { m a s k } }$ is a learned mask token and $\mathbf { e } _ { g } ~ \in ~ \mathbb { R } ^ { D }$ is a learned path code that tells the predictor which traversal to predict. Each layer applies self-attention among the queries, cross-attention to a conditioning sequence and an MLP. With $P ^ { ( m ) } ( { \bf S } ; g )$ denoting the prediction from conditioning S for path g, the context loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c t x } } ^ { ( m ) } = \frac { 1 } { | \mathcal { H } | D } \left\| P ^ { ( m ) } ( { \bf C } ; g _ { 1 } ) - { \bf T } ^ { g _ { 1 } } \right\| _ { F } ^ { 2 } . } \end{array}\tag{4}
$$

The view loss $\mathcal { L } _ { \mathrm { v i e w } } ^ { ( m ) } \left( \mathrm { E q . } \left( 7 \right) \right)$ instead predicts the $g _ { 2 }$ targets from the AVC streams, so only information that survives a change of path, namely the content of the scene, can be predicted reliably.

## 3.3. Adversarial View Coupling

Two scan paths over the same visible patches share much local context, so their prediction can become trivial; AVC prevents this by learning the split of the visible patches from whose two halves the hidden-patch targets are hardest to predict.

Adversarial view partition (Fig. 4). A per-sensor linear scorer gives every visible stem token a score $a _ { j } = \mathbf { w } ^ { \top } \mathbf { z } _ { j } +$ $b , j \in \mathcal { V } .$ . The top-scoring half of the visible tokens forms view $\nu _ { A }$ and the rest form $\gamma _ { B }$ , a hard, disjoint split of equal size. As the split itself is not differentiable, the scorer learns through a straight-through weight:

$$
\begin{array} { r } { \tilde { \mathbf { z } } _ { j } = \mathbf { z } _ { j } \left( 1 + \omega _ { j } - \mathrm { s g } [ \omega _ { j } ] \right) , \quad \omega _ { j } = \left\{ \begin{array} { l l } { \sigma ( \mathcal { R } ( a _ { j } ) ) , } & { j \in \mathcal { V } _ { A } , } \\ { 1 - \sigma ( \mathcal { R } ( a _ { j } ) ) , } & { j \in \mathcal { V } _ { B } , } \end{array} \right. } \end{array}\tag{5}
$$

whose forward value is exactly one. Here σ is the logistic sigmoid and R is a gradient-reversal layer, which acts as the identity in the forward pass and negates the gradient in the backward pass. The predictor and encoder minimise ${ \mathcal { L } } _ { \mathrm { v i e w } } .$ while the scorer receives the negated gradient and passes it to the stem tokens it reads, in addition to their ordinary gradient, so the scorer learns the partition that makes this prediction hardest.

![](images/c1185607d04802cb0edea9cd9628580f98ef4bc8ac8b2137b8f5f9d3ccbc619c.jpg)  
Figure 2. Overview of RS-JEPA. Top: training. Per-sensor stems feed one shared Vim encoder; AVC (Fig. 4) and a path-conditioned predictor (Fig. 3b) predict stop-gradient targets $\mathbf { T } ^ { g }$ , and the heads are aligned by InfoNCE and neighbourhood consensus (full objective in Eq. (10)). Bottom: inference, a real S1→S2 top-5 on BEN-14K test (query with F1@5 closest to the mean, not hand-picked).

![](images/f24dedd4ba91dd6a131b1b04fb68cb8726fd37f2e1eeb0b58eedce4795ea6047.jpg)  
Figure 3. Encoder and predictor. (a) One of the 12 Vim blocks: bidirectional selective scans along path g. (b) The six-layer query predictor; queries carry position and path code $\mathbf { e } _ { g }$

Shared encoding and Coupled View Attention. Both views pass through the same trunk along the context path, $\mathbf { H } _ { A } = f _ { \theta } ( \tilde { \mathbf { Z } } _ { \mathcal { V } _ { A } } ; g _ { 1 } )$ and ${ \bf H } _ { B } = f _ { \theta } ( \tilde { \bf Z } \nu _ { B } ; g _ { 1 } )$ , so the coupling gradient trains the retrieval backbone itself. The two views are then coupled by a stack of four Coupled View Attention (CVA) blocks (Fig. 4) whose projections and MLP are shared by both streams. The streams $\mathbf { H } _ { A } , \mathbf { H } _ { B }$ are refined by self-attention, and their queries, keys and values $( \mathbf { Q } _ { v } , \mathbf { K } _ { v } , \mathbf { V } _ { v } ) , v \in \{ A , B \}$ , enter a single coupled atten-

tion call

$$
\begin{array} { r } { \left[ \mathbf { O } _ { A B } \right] = \mathrm { A t t n } \left( \left[ \mathbf { Q } _ { A } \right] , \left[ \mathbf { K } _ { B } \right] , \left[ \mathbf { V } _ { B } \right] \right) , } \\ { \left[ \mathbf { O } _ { B A } \right] = \left[ \left[ \mathbf { Q } _ { B } \right] , \left[ \mathbf { K } _ { A } \right] , \left[ \mathbf { V } _ { A } \right] \right) , } \end{array}\tag{6}
$$

where rows are stacked along the batch axis, so view-A queries attend only to view-B keys and values, and vice versa. Each output is added residually and passed through a shared MLP, giving the crossed streams $\mathbf { X } _ { A B }$ (initialised from $\mathbf { H } _ { B }$ and carried across blocks) and $\mathbf { X } _ { B A }$ (recomputed from $\mathbf { H } _ { A }$ in every block), so only $\mathbf { X } _ { A B }$ accumulates context over depth. The predictor then predicts the targets of the other path from the crossed streams $[ { \bf X } _ { A B } ; { \bf X } _ { B A } ]$ , concatenated along the token axis:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { v i e w } } ^ { ( m ) } = \frac { 1 } { | \mathcal { H } | D } \left\| P ^ { ( m ) } ( [ \mathbf { X } _ { A B } ; \mathbf { X } _ { B A } ] ; g _ { 2 } ) - \mathbf { T } ^ { g _ { 2 } } \right\| _ { F } ^ { 2 } . } \end{array}\tag{7}
$$

The encoder must thus produce features that remain predictive across adversarially chosen subsets of the scene.

## 3.4. Cross-Sensor Neighbourhood Consensus

Instance-level contrastive alignment pulls each scene towards its co-registered twin but leaves the arrangement of the other scenes unconstrained; the consensus loss instead requires both sensors to agree on which scenes are neighbours. For a batch of K pairs, let $\mathbf { u } _ { i } ^ { ( m ) }$ and $\mathbf { c } _ { i } ^ { ( m ) }$ be the unit-norm uni- and cross-head embeddings of scene i from sensor m (Sec. 3.5). The same-sensor neighbourhood distribution of scene i over the other scenes is

![](images/287f6ceee0367e79fb4cf315ad344effa0dd837c3324375e94ccf76e8fa598d8.jpg)  
Figure 4. Adversarial View Coupling. A scorer with gradient reversal splits the visible tokens into views A and $B ,$ which the shared encoder reads along g . In each of four CVA blocks, one view’s queries attend to the other view’s keys and values, giving $\mathbf { X } _ { A B }$ and $\mathbf { X } _ { B A }$ for the predictor.

$$
Q _ { m } ( j | i ) = \frac { \exp ( \mathbf { u } _ { i } ^ { ( m ) \top } \mathbf { u } _ { j } ^ { ( m ) } / \tau _ { r } ) } { \sum _ { k \neq i } \exp ( \mathbf { u } _ { i } ^ { ( m ) \top } \mathbf { u } _ { k } ^ { ( m ) } / \tau _ { r } ) } , \quad j \neq i ,\tag{8}
$$

where $\tau _ { r }$ is a relational temperature. The cross-sensor distribution $Q _ { m  m ^ { \prime } } ( j | i )$ is defined in the same way, with query $\mathbf { c } _ { i } ^ { ( m ) }$ and gallery $\{ \mathbf { c } _ { j } ^ { ( m ^ { \prime } ) } \} _ { j \neq i }$ for $m ^ { \prime } \neq$ m. The target is the consensus of the two sensors, $\begin{array} { r } { \bar { Q } _ { i } = \mathrm { s g } \bigl [ \frac 1 2 \sum _ { m } \bar { Q } _ { m } ( \cdot | i ) \bigr ] } \end{array}$ , and all four distributions are pulled towards it:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r e l } } = \cfrac { 1 } { 4 K } \underset { i = 1 } { \overset { K } { \sum } } \bigg [ \underset { m } { \sum } \mathrm { K L } \big ( \bar { Q } _ { i } \| Q _ { m } ( \cdot | i ) \big ) } \\ & { \qquad + \underset { m \neq m ^ { \prime } } { \sum } \mathrm { K L } \big ( \bar { Q } _ { i } \| Q _ { m  m ^ { \prime } } ( \cdot | i ) \big ) \bigg ] . } \end{array}\tag{9}
$$

The twin $j = i$ is excluded from every distribution, and the loss vanishes only when all four distributions coincide.

## 3.5. Training Objective and Inference

Retrieval heads and inference. The context $\mathbf { C } ^ { ( m ) }$ is mean-pooled and passed through two MLP heads shared by both sensors, uni and cross, whose $\ell _ { 2 } \cdot$ -normalised outputs are $\mathbf { u } _ { i } ^ { ( m ) }$ and $\mathbf { c } _ { i } ^ { ( m ) }$ . Both heads are trained with the symmetric InfoNCE loss [24] $\mathcal { L } _ { \mathrm { n c e } }$ with temperature τ and in-batch negatives, and the uni head also with a cosine term $\begin{array} { r } { \mathcal { L } _ { \mathrm { c o s } } = 1 - \frac { 1 } { K } \sum _ { i } \mathbf { u } _ { i } ^ { ( 1 ) \top } \mathbf { u } _ { i } ^ { ( 2 ) } } \end{array}$ . At test time, the encoder reads all N patches along the row-major path without masking, and gallery items are ranked by cosine similarity of the uni (same-sensor) or cross (cross-sensor) embeddings. The deployed model consists only of the two stems, the shared trunk and the two heads.

Full objective. The total loss is

$$
\begin{array} { r l } & { \mathcal { L } = \sum _ { m } \mathcal { L } _ { \mathrm { c t x } } ^ { ( m ) } + \frac { 1 } { 2 } \sum _ { m } \mathcal { L } _ { \mathrm { v i e w } } ^ { ( m ) } + \mathcal { L } _ { \mathrm { n c e } } ( \mathbf { c } ^ { ( 1 ) } , \mathbf { c } ^ { ( 2 ) } ) } \\ & { ~ + \mathcal { L } _ { \mathrm { n c e } } ( \mathbf { u } ^ { ( 1 ) } , \mathbf { u } ^ { ( 2 ) } ) + \mathcal { L } _ { \mathrm { c o s } } + \lambda _ { \mathrm { r e l } } \mathcal { L } _ { \mathrm { r e l } } , } \end{array}\tag{10}
$$

where each loss term is averaged over the batch and $\lambda _ { \mathrm { { r e l } } }$ weights the consensus term. All parameters, including the scorers through the reversed gradient of Eq. (5), are optimised jointly in a single stage. The in-batch negatives of ${ \mathcal { L } } _ { \mathrm { n c e } }$ prevent collapse.

## 4. Experiments

## 4.1. Datasets and Data Splits

We use three dual-sensor benchmarks; training is labelfree and train and test scenes do not overlap. BEN-14K [19], a BigEarthNet subset over ten European countries, has 14,832 Sentinel-1 (VV/VH, S1) and Sentinel-2 (12 bands, S2) pairs with 19 CORINE land-cover multilabels; 11,310 train / 3,522 test. CBRSIR-VS [21] pairs SAR (S1) and RGB (S2) views of the same scene, which differ strongly in appearance: 26,901 pairs, 10 classes, perclass 70/30 split (18,831 / 8,070). DSRSID [14] pairs Gaofen-1 high-resolution panchromatic (S1) and lowerresolution 4-band multispectral (S2) images: 80,000 pairs, 8 classes, per-class 70/30 split (56,000 / 24,000). All images are resized to 120 × 120 and normalised per channel; every test sample is both query and gallery item.

## 4.2. Implementation Details

RS-JEPA uses a 12-block Vim encoder (width 192) on $1 0 \times 1 0$ patches, a fixed 60% masking ratio [1,4], a six-layer predictor and four CVA blocks, with $\tau = 0 . 0 7 , \tau _ { r } = 0 . 1$ and $\lambda _ { \mathrm { r e l } } = 1$ . We train for 100 epochs with AdamW (batch 290, learning rate warmed up to $\mathrm { \bar { 1 0 } ^ { - 3 } }$ over 15 epochs, then cosine decay) and identical random flips and $9 0 °$ rotations on both sensors, and report results across independent runs. All experiments run in PyTorch on four NVIDIA V100 GPUs with 32 GB RAM and an Intel Xeon CPU.

Top-5 Retrieved  
Table 1. Cross-sensor retrieval on BEN-14K, CBRSIR-VS and DSRSID. F1@5 (%); i→j: query from sensor Si, gallery from sensor Sj. Best in bold, second best underlined. All methods are re-implemented and trained by us under one protocol. Params: training model; GFLOPs: one S1–S2 pair at 120×120 (2 FLOPs per multiply-accumulate; selective scan included for RS-JEPA), BEN-14K configuration.
<table><tr><td rowspan="2">Method</td><td colspan="2">Params GFLOPs</td><td colspan="4">BEN-14K</td><td colspan="4">CBRSIR-VS</td><td colspan="4">DSRSID</td></tr><tr><td>(M) ↓</td><td>↓</td><td>1→1 2→2</td><td>1→2</td><td>2→1</td><td>Avg.</td><td>1→1 2→2</td><td>1→2</td><td>2→1</td><td>Avg.</td><td>1→1 2→2</td><td>1→2</td><td>2→1</td><td>Avg.</td></tr><tr><td colspan="2">Masked image modelling</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MAE-RVSA [26]</td><td>172.23</td><td>50.33</td><td>65.27 68.87</td><td>68.28 67.88</td><td>67.58</td><td></td><td>64.2873.35</td><td>63.41 64.79</td><td>66.46</td><td></td><td>74.9473.1077.8677.4575.84</td><td></td><td></td><td></td></tr><tr><td>ScaleMAE [17]</td><td>215.07</td><td>51.12</td><td>63.88 68.93</td><td>67.86</td><td>68.93</td><td>67.40 64.99</td><td>75.29</td><td>64.59</td><td>66.10</td><td>67.74</td><td>73.47 71.84</td><td>76.12</td><td>75.49</td><td>74.23</td></tr><tr><td>SatMAE++ [16]</td><td>223.57</td><td>106.18</td><td>64.85 69.08</td><td>67.69</td><td>68.27</td><td>67.47</td><td>64.93 74.55</td><td>65.47</td><td>65.23</td><td>67.55</td><td>77.07 77.00</td><td>78.29</td><td>78.80</td><td>77.79</td></tr><tr><td colspan="2">Multi-sensor self-supervised</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CROMA [8]</td><td>200.27</td><td>38.17</td><td>68.57 71.09</td><td>68.92</td><td>69.97</td><td>69.64</td><td>72.1680.13</td><td>72.08</td><td>73.65</td><td>74.50</td><td>87.53 88.14</td><td>87.28</td><td>86.74 87.42</td><td></td></tr><tr><td>DeCUR [28]</td><td>198.07</td><td>5.22</td><td>66.06 66.43</td><td>62.39</td><td>61.48</td><td>64.09</td><td>75.99 85.92</td><td>72.83</td><td>77.55</td><td>78.07</td><td>73.67 73.31</td><td>62.49</td><td>61.05</td><td>67.63</td></tr><tr><td>SkySense [10]</td><td>173.24</td><td>51.13</td><td>66.65 71.01</td><td>67.97</td><td>68.02</td><td>68.41</td><td>66.23 74.28</td><td>65.70</td><td>65.04</td><td>67.81</td><td>87.94 88.02</td><td>88.96</td><td>88.85</td><td>88.44</td></tr><tr><td colspan="2">Joint-embedding predictive</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CR-JEPA [13]</td><td>115.18</td><td>14.74</td><td>67.93 71.64</td><td>69.27</td><td>70.41</td><td>69.81</td><td>60.7774.72</td><td>61.64 64.73</td><td></td><td>65.46</td><td>71.55 73.65</td><td></td><td>74.7075.2573.79</td><td></td></tr><tr><td>X-JEPA [4]</td><td>230.47</td><td>32.98</td><td>66.27 70.08</td><td>66.81</td><td>68.01</td><td>67.79</td><td>67.94 81.50</td><td>71.78</td><td>71.52</td><td>73.18</td><td>85.84 85.91</td><td>87.81</td><td>87.89</td><td>86.86</td></tr><tr><td>RS-JEPA (Ours)</td><td>14.57</td><td>2.23</td><td>70.93 72.55</td><td>73.86</td><td>74.33</td><td>72.92</td><td>76.39 86.54</td><td>78.72</td><td>79.75</td><td>580.35</td><td>93.77 93.73</td><td>95.10</td><td>95.14 94.43</td><td></td></tr></table>

![](images/c407401ad935f74b7d6398e8910978c27761290d4685307a5aa72c4a2ee452f9.jpg)

Query  
Top-5 Retrieved  
![](images/70c8842f35eef061f81812f31211a77a13ef903ffedab716c0a116b28ec6a79a.jpg)

![](images/f12fa1e7c903aa81c64b10b3a95ede2464a9d05a5509301b55df8265d394f403.jpg)  
Figure 5. Qualitative retrieval. Comparison of the top-5 results of each method for one S1→S2 and one S2→S1 query per benchmark. ✓: relevant (F1 ≥ 0.5 on BEN-14K; same class otherwise), ✗: irrelevant. Queries are selected by a fixed rule as the most representative of each method’s F1@5, not hand-picked.

Baselines: DeCUR uses a ResNet-50 backbone, as in its original design [28], and the other seven use a ViT-Base/16 backbone; all eight are re-implemented following the X-JEPA protocol [4] and trained from scratch by us for 100 epochs on the same pairs at 120 × 120, with shared retrieval code; single-sensor MAE variants get one encoder per sensor and an InfoNCE head. Tab. 1 thus compares methods under one training budget, not their best published results (e.g., CR-JEPA reports 75.82/75.40 cross-sensor F1@5 on BEN-14K after 400 epochs at 224 × 224 [13]); at their original 224 × 224 setting, X-JEPA and CR-JEPA still trail RS-JEPA (Fig. 6c).

Table 2. Ablations. Mean F1@5 and mAP (%) over the four directions. Top: components added cumulatively; bottom: masking ratio. Best in bold, second underlined.
<table><tr><td></td><td colspan="2">BEN-14K</td><td colspan="2">CBRSIR-VS</td><td colspan="2">DSRSID</td></tr><tr><td>Configuration</td><td>F1@5</td><td>mAP</td><td>F1@5</td><td>mAP</td><td>F1@5</td><td>mAP</td></tr><tr><td>Components</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td> $7 1 . 4 7 _ { \pm 0 . 1 8 }$ </td><td>83.85±0.22</td><td> $7 8 . 2 5 _ { \pm 0 . 2 1 }$ </td><td> $5 1 . 4 0 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $9 2 . 8 4 _ { \pm 0 . 1 6 }$ </td><td> $3 8 . 7 3 { \scriptstyle \pm 0 . 3 }$  2</td></tr><tr><td>+ Consensus</td><td> $7 1 . 8 2 { \scriptstyle \pm 0 . . }$  15</td><td>84.42±0.19</td><td> $7 9 . 1 8 _ { \pm 0 . 1 9 }$ </td><td>53.19±0.25</td><td> $9 3 . 5 8 _ { \pm 0 . 1 4 }$ </td><td>42.93±0.29</td></tr><tr><td>+ Scan views</td><td>72.64±0.14</td><td>84.85±0.20</td><td> $7 9 . 8 2 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $5 4 . 4 3 _ { \pm 0 . 2 2 }$ </td><td></td><td>94.06±0.12 45.83±0.25</td></tr><tr><td>+ AVC (Ours)</td><td> $7 2 . 9 2 _ { \pm 0 . 1 2 }$  21</td><td> $\mathbf { 8 5 . 1 6 _ { \pm 0 . 1 8 } }$ </td><td> $\mathbf { 8 0 . 3 5 _ { \pm 0 . 1 5 } }$ </td><td> ${ \bar { \bf 5 } } 5 . { \bf 0 5 } _ { \pm 0 . 2 0 }$ </td><td> $\mathbf { 9 4 . 4 3 _ { \pm 0 . 1 1 } }$ </td><td> ${ \bf 4 7 . 0 9 _ { \pm 0 . 2 2 } }$ </td></tr><tr><td colspan="2">Masking ratio (full model)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>50%</td><td> $7 2 . 7 5 \substack { \pm 0 . 1 6 }$ </td><td>84.33±0.21</td><td> $\underline { { 8 0 . 1 2 } } \pm \mathbf { 0 . 1 8 }$ </td><td> ${ \bar { 5 } } 5 . 3 3 _ { \pm 0 . 2 4 }$ </td><td> $9 4 . 0 6 _ { \pm 0 . 1 5 }$ </td><td> $4 6 . 8 3 _ { \pm 0 . 2 7 }$ </td></tr><tr><td>60%</td><td> $7 2 . 9 2 _ { \pm 0 . 1 }$  2</td><td>85.16±0.18</td><td> $\mathbf { 8 0 . 3 5 _ { \pm 0 . 1 5 } }$ </td><td> $5 5 . 0 5 { \scriptstyle \pm 0 . 2 0 } 9 4 . 4 3 { \scriptstyle \pm 0 . 1 1 }$ </td><td></td><td> $4 7 . 0 9 _ { \pm 0 . 2 2 }$ </td></tr><tr><td>70%</td><td> $7 2 . 4 9 _ { \pm 0 . 1 7 }$ </td><td>86.40±0.24 78.73±0.23</td><td></td><td> $5 4 . 7 8 { \scriptstyle \pm 0 . 2 6 } 9 4 . 2 2 { \scriptstyle \pm 0 . 1 3 }$ </td><td></td><td> ${ \bf 4 7 . 4 4 _ { \pm 0 . 2 4 } }$ </td></tr></table>

## 4.3. Evaluation Protocol

Following X-JEPA [4], we evaluate four retrieval directions (S1→S1, S2→S2, S1→S2, S2→S1) and their mean. The main metric is F1@5 between the label sets of the query and its top-5 results, which equals precision@5 on the single-label CBRSIR-VS and DSRSID. For mAP, a result is relevant if it shares a label with the query. Same-sensor search excludes the query, while the cross-sensor gallery keeps its co-registered partner [4, 13]; Sec. 5 also reports results without it.

## 5. Results and Discussion

Quantitative Analysis. As shown in Tab. 1, RS-JEPA achieves the best F1@5 in every direction on all three benchmarks, improving the mean over the strongest baseline by 3.11 points on BEN-14K (CR-JEPA), 2.28 on CBRSIR-VS (DeCUR) and 5.99 on DSRSID (SkySense). The baselines rank inconsistently across datasets: DeCUR, for instance, leads on CBRSIR-VS but is among the weakest on DSRSID. RS-JEPA ranks first throughout. Its advantage is consistently larger in cross-sensor than in samesensor retrieval, in line with an objective that targets content shared by both sensors. When each query’s co-registered partner is removed from the cross-sensor gallery, RS-JEPA still leads on all benchmarks (70.46, 78.82 and 93.55 vs. 67.82, 75.07 and 85.81 for the strongest baselines), so its margin does not rely on instance matching. Over the depths in Tab. 4, RS-JEPA is best in 59 of the 60 direction and average columns, the exception being S2→S2 F1@20 on CBRSIR-VS, where DeCUR leads by 0.29 points. On DSRSID, its lead in average F1 widens from 2.88 points at k = 1 to 7.36 at $k = 2 0$ , showing that its ranking stays reliable deeper into the retrieved list.

Qualitative Analysis. Figure 5 shows the top-5 results for one query per direction and benchmark, selected by a fixed rule rather than by hand. Over the two queries per benchmark, RS-JEPA retrieves at least as many relevant scenes as every other method, and all ten of its results on DSRSID match the query class. A few queries cannot by themselves

![](images/a3927319344caa3163ba834949d2ff7e72fd709d15880a07c72112c3fa7fb181.jpg)

(c) Original vs. our input setting
<table><tr><td rowspan="2">Method</td><td colspan="2">224×224,16×16</td><td colspan="2">120×120,10×10</td></tr><tr><td>F1@5</td><td>mAP</td><td>F1@5</td><td>mAP</td></tr><tr><td>X-JEPA [4]</td><td>68.47</td><td>81.49</td><td>67.79</td><td>81.30</td></tr><tr><td>CR-JEPA [13]</td><td>69.86</td><td>80.26</td><td>69.81</td><td>81.08</td></tr><tr><td>RS-JEPA (Ours)</td><td>72.38</td><td>85.40</td><td>72.92</td><td>85.16</td></tr></table>

Figure 6. Resolution (BEN-14K). (a) Patch size at $1 2 0 \times 1 2 0 .$ (b) input size at 10 × 10 patches: change from our default (star), with token counts below. (c) Mean F1@5 and mAP (%) at each method’s original and our setting.

Table 3. Cross-sensor alignment. mAP (%) for same- and crosssensor retrieval; Asym.: |S1→S2 − S2→S1|. Best in bold.
<table><tr><td rowspan="2">Model</td><td colspan="3">BEN-14K</td><td colspan="3">CBRSIR-VS</td><td colspan="3">DSRSID</td></tr><tr><td>Same</td><td>Cross</td><td>Asym.↓</td><td>Same</td><td>Cross</td><td>Asym.↓</td><td>Same</td><td></td><td>Cross Asym.↓</td></tr><tr><td>CROMA</td><td></td><td>87.02 86.84</td><td>0.16</td><td>49.82 48.81</td><td></td><td>0.50</td><td>38.87 38.50</td><td></td><td>0.04</td></tr><tr><td>DeCUR</td><td></td><td>83.21 83.06</td><td>1.03</td><td>47.83</td><td>44.71</td><td>1.58</td><td>29.56</td><td>27.24</td><td>0.14</td></tr><tr><td>SkySense</td><td>81.45</td><td>80.40</td><td>0.16</td><td>34.99</td><td>31.78</td><td>0.10</td><td>46.17</td><td>45.60</td><td>0.10</td></tr><tr><td>X-JEPA</td><td></td><td>81.3681.23</td><td>0.89</td><td>48.52</td><td>47.89</td><td>2.50</td><td>33.5433.57</td><td></td><td>0.04</td></tr><tr><td>CR-JEPA</td><td></td><td>81.8780.28</td><td>0.28</td><td>43.97</td><td>47.73</td><td>0.50</td><td>30.23 28.19</td><td></td><td>0.09</td></tr><tr><td>RS-JEPA (ours)</td><td>85.26 85.06</td><td></td><td>0.12</td><td>55.42 54.68</td><td></td><td>0.08</td><td>47.35 46.82</td><td></td><td>0.03</td></tr></table>

support general conclusions.

Computational Efficiency Analysis. RS-JEPA uses 14.57M parameters during training, of which only 3.88M are retained for retrieval, since the AVC modules, predictors and target pass serve training alone. Embedding one S1–S2 pair requires 2.23 GFLOPs, against 230.47M parameters and 32.98 GFLOPs for X-JEPA (Tab. 1), corresponding to roughly 16× fewer parameters and 15× fewer operations. We also timed every model of Tab. 1 on one V100 with FP16. At batch size 64, RS-JEPA embeds 1,918 pairs per second, faster than all baselines except DeCUR (3,235), and needs the least GPU memory at batch 256 (0.60 GiB against 1.34–3.90 GiB). For a single pair, however, it takes 33.1 ms against 13.7–21.1 ms for most baselines, as its many small scan kernels leave the GPU idle, so its advantage lies in batched embedding.

Table 4. Retrieval depth. F1@k (%) for $k \in \{ 1 , 1 0 , 2 0 \}$ in every direction (F1@5 in Tab. 1); i→j: query from sensor ${ \mathrm { S } } i ,$ gallery from sensor Sj. Best in bold, second best underlined, per column among the models at the same k. Cross-sensor F1@1 is high because th gallery contains the query’s co-registered partner (Sec. 4.3).
<table><tr><td rowspan="3">Method</td><td rowspan="3">k</td><td colspan="5">BEN-14K</td><td colspan="5">CBRSIR-VS</td><td colspan="5">DSRSID</td></tr><tr><td>1→1</td><td>2→2</td><td>1→2</td><td>2→1</td><td>Avg.</td><td>1→1</td><td>2→2</td><td>1→2</td><td>2→1</td><td>Avg.</td><td>1→1</td><td>2→2</td><td>1→2</td><td>2→1</td><td>Avg.</td></tr><tr><td rowspan="3">MAE-RVSA [26]</td><td>1 10</td><td>66.48 64.26</td><td>70.70 67.81</td><td>74.28 66.62</td><td>74.43 65.66</td><td>71.47 66.09</td><td>65.70 63.35</td><td>75.39 71.91</td><td>65.18 62.58</td><td>66.56 63.70</td><td>68.21 65.39</td><td>77.38 73.46</td><td>75.89 71.49</td><td>98.43 73.73</td><td>98.20 73.23</td><td>87.48 72.98</td></tr><tr><td>20</td><td>63.31</td><td>66.60</td><td>64.67</td><td>63.85</td><td>64.61</td><td>62.01</td><td>70.29</td><td>61.19</td><td>62.23</td><td></td><td>71.67</td><td>69.57</td><td>70.65</td><td>70.10</td><td>70.50</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>63.93</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">ScaleMAE [17]</td><td>1</td><td>64.94</td><td>70.14</td><td>76.08</td><td>77.70</td><td>72.22</td><td>66.77</td><td>77.21</td><td>66.25</td><td>67.43</td><td>69.41</td><td>75.96</td><td>74.53</td><td>97.41</td><td>96.92</td><td>86.20</td></tr><tr><td>10</td><td>63.10</td><td>68.00</td><td>65.67</td><td>66.47</td><td>65.81</td><td>63.94</td><td>73.89</td><td>63.53</td><td>64.94</td><td>66.58</td><td>72.03</td><td>70.34</td><td>71.74</td><td>71.10</td><td>71.30</td></tr><tr><td>20</td><td>62.25</td><td>66.83</td><td>63.87</td><td>64.44</td><td>64.35</td><td>62.57</td><td>72.10</td><td>62.06</td><td>63.44</td><td>65.04</td><td>70.22</td><td>68.48</td><td>68.40</td><td>67.63</td><td>68.68</td></tr><tr><td rowspan="3">SatMAE++ [16]</td><td>1</td><td>66.89</td><td>70.36</td><td>73.12</td><td>74.44</td><td>71.20</td><td>66.73</td><td>76.90</td><td>66.85</td><td></td><td></td><td>79.42</td><td>79.57</td><td>93.95</td><td>95.00</td><td>86.99</td></tr><tr><td>10</td><td>63.73</td><td>68.15</td><td>65.58</td><td>66.08</td><td>65.88</td><td>63.78</td><td>73.03</td><td>64.47</td><td>66.37 64.57</td><td>69.21 66.46</td><td></td><td>75.53</td><td>74.76</td><td>75.00</td><td>75.23</td></tr><tr><td>20</td><td>62.58</td><td>66.95</td><td>63.98</td><td>64.15</td><td>64.42</td><td>62.36</td><td>71.17</td><td>62.99</td><td>63.11</td><td>64.91</td><td>75.63 73.97</td><td>73.71</td><td>72.01</td><td>72.18</td><td>72.97</td></tr><tr><td rowspan="3">CROMA [8]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1 10</td><td>69.45</td><td>72.30 70.54</td><td>69.91 68.18</td><td>71.34</td><td>70.75 68.92</td><td>73.68</td><td>81.82</td><td>72.32</td><td>74.19</td><td>75.50</td><td>93.03</td><td>93.37</td><td>89.86</td><td>89.45</td><td>91.42</td></tr><tr><td>20</td><td>67.86 66.95</td><td>69.61</td><td>67.29</td><td>69.10 68.04</td><td>67.97</td><td>71.42</td><td>79.08</td><td>71.72</td><td>72.76</td><td>73.74</td><td>85.64</td><td>86.44</td><td>85.61</td><td>85.10</td><td>85.70</td></tr><tr><td rowspan="3">DeCUR [28]</td><td></td><td></td><td></td><td></td><td></td><td></td><td>70.37</td><td>77.73</td><td>70.89</td><td>71.73</td><td>72.68</td><td>83.97</td><td>84.71</td><td>83.79</td><td>83.28</td><td>83.93</td></tr><tr><td>1</td><td>66.89</td><td>67.74</td><td>64.24</td><td>62.92</td><td>65.45</td><td>77.05</td><td>86.77</td><td>72.86</td><td>77.37</td><td>78.51</td><td>80.70</td><td>78.18</td><td>63.43</td><td>61.27</td><td>70.90</td></tr><tr><td>10</td><td>65.52</td><td>65.88</td><td>61.93</td><td>61.29</td><td>63.66</td><td>75.48</td><td>85.29</td><td>72.49</td><td>76.97</td><td>77.56</td><td>71.96</td><td>71.92</td><td>61.92</td><td>61.16</td><td>66.74</td></tr><tr><td rowspan="3"></td><td>20</td><td>64.86</td><td>65.27</td><td>61.40</td><td>60.99</td><td>63.13</td><td>74.77</td><td>84.45</td><td>72.08</td><td>75.82</td><td>76.78</td><td>70.65</td><td>70.64</td><td>61.29</td><td>60.83</td><td>65.85</td></tr><tr><td>1</td><td></td><td>72.34</td><td>70.34</td><td>70.83</td><td>70.29</td><td>67.40</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>95.07</td></tr><tr><td>10</td><td>67.63</td><td>70.10</td><td></td><td></td><td></td><td></td><td>76.72</td><td>67.00</td><td>66.18</td><td>69.32</td><td>93.35</td><td>93.23</td><td>96.82</td><td>96.88</td><td></td></tr><tr><td rowspan="3">SkySense [10]</td><td></td><td>65.84</td><td>68.88</td><td>66.87 65.54</td><td>66.76 65.47</td><td>67.39 66.18</td><td>65.43</td><td>72.93</td><td>64.56</td><td>63.92</td><td>66.71</td><td>85.82</td><td>85.99</td><td>85.95</td><td>85.70</td><td>85.86 83.43</td></tr><tr><td>20</td><td>64.85</td><td></td><td></td><td></td><td></td><td>64.20</td><td>71.01</td><td>63.17</td><td>62.82</td><td>65.30</td><td>83.52</td><td>83.84</td><td>83.30</td><td>83.04</td><td></td></tr><tr><td>1</td><td>67.15</td><td>70.81</td><td>68.32</td><td>69.89</td><td>69.04</td><td>68.27</td><td>83.10</td><td>72.59</td><td>71.92</td><td>73.97</td><td>94.38</td><td>93.94</td><td>94.11</td><td></td><td>94.08</td></tr><tr><td rowspan="3">X-JEPA [4]</td><td>10</td><td>65.42</td><td>69.10</td><td>66.22</td><td>67.13</td><td>66.97</td><td>67.40</td><td>80.68</td><td>71.36</td><td>71.30</td><td>72.68</td><td>83.28</td><td>83.49</td><td>84.66</td><td>93.89 84.72</td><td>84.03</td></tr><tr><td>20</td><td>64.57</td><td>67.99</td><td>65.42</td><td>66.12</td><td>66.02</td><td>67.18</td><td>79.71</td><td>70.92</td><td>70.77</td><td>72.14</td><td>81.23</td><td>81.47</td><td>82.11</td><td>82.18</td><td>81.75</td></tr><tr><td>1</td><td>68.76</td><td>73.42</td><td>72.39</td><td>73.25</td><td>71.95 68.70</td><td>61.52</td><td>77.10</td><td>61.91</td><td>65.63</td><td>66.54</td><td>72.79 75.37</td></table>

## 6. Ablation Studies

All defaults (60% masking, $1 0 \times 1 0$ patches, $1 2 0 \times 1 2 0$ input) were fixed a priori from prior JEPA work; the ablations below are post-hoc sensitivity analyses.

Components: Starting from a single-view base (rowmajor scan; $\mathcal { L } _ { \mathrm { c t x } } , \mathcal { L } _ { \mathrm { n c e } } , \mathcal { L } _ { \mathrm { c o s } } )$ , Tab. 2 (top) adds consensus $( \mathcal { L } _ { \mathrm { r e l } } )$ , then scan views (random paths and ${ \mathcal { L } } _ { \mathrm { v i e w } }$ from C, no $\mathrm { A V C } )$ , then AVC. Each step improves F1@5 and mAP on all three benchmarks, and the full model is best (72.92/85.16 on BEN-14K, 80.35/55.05 on CBRSIR-VS, 94.43/47.09 on DSRSID).

Component Variants: On BEN-14K, removing consensus changes F1@5 by only 0.10 but lowers mAP by 3.20; a symmetric cyclic loss in its place lowers F1@5 by 0.61 and raises mAP by 1.57. A random second path matches Eq. (3). We did not run random splits, AVC without coupled attention, or a fixed-path control with the same ${ \mathcal { L } } _ { \mathrm { v i e w } } ,$ so Tab. 2 cannot separate adversarial partitioning from coupled attention, nor scan order from the extra prediction loss.

Masking Ratio: From 50% to 70% masking (Tab. 2, bottom), F1@5 varies by at most 0.43 on BEN-14K and 0.37 on DSRSID, but drops by 1.62 on CBRSIR-VS at 70%, with lower mAP, suggesting heavy masking suits its large SAR–RGB appearance gap poorly.

Patch and Input Size: On BEN-14K (Fig. 6a,b; larger inputs upsampled), F1@5 peaks at the default (72.92) and stays within 1.22 points elsewhere, while mAP gains at most 0.50 with $1 2 \times 1 2$ or 15 × 15 patches or $1 6 0 \times 1 6 0$ inputs. Smaller patches and larger inputs need 1.6–4× more tokens without improving F1@5.

Cross-Sensor Alignment: In full-ranking mAP (Tab. 3), RS-JEPA is best for same- and cross-sensor retrieval on CBRSIR-VS (55.42/54.68) and DSRSID (47.35/46.82) and second to CROMA on BEN-14K. Its asymmetry between the two cross-sensor directions is the lowest on all benchmarks (0.12, 0.08, 0.03), so retrieval is equally reliable whichever sensor issues the query.

## 7. Conclusion

We proposed RS-JEPA, a compact JEPA for label-free cross-sensor retrieval that turns the reading order of a shared Vision Mamba encoder into scan-path views, makes their prediction harder with Adversarial View Coupling, and aligns the sensors through neighbourhood consensus. With a 3.88M-parameter retrieval network, RS-JEPA achieves the highest F1@5 of twelve re-implemented methods across all three benchmarks, while CROMA remains ahead in fullranking mAP on BEN-14K. Limitations. RS-JEPA still needs co-registered training pairs, limiting scale. We test three benchmarks at $1 2 0 \times 1 2 0$ pixels, with no unseen regions or other sensors, one fixed scan-path set and a shared baseline budget. Ablations are scored on the test splits, so they show trends only. Training adds 10.7M parameters; its time cost is unmeasured. Broader archives and transfer to new regions are future work. Reading order is thus an effective source of self-supervised views.

## References

[1] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15619–15629, 2023. 1, 2, 5

[2] Yuxing Chen and Lorenzo Bruzzone. Self-supervised SARoptical data fusion of Sentinel-1/-2 images. IEEE Transactions on Geoscience and Remote Sensing, 60:1–11, 2022. 2

[3] Shabnam Choudhury, Yash Salunkhe, Sarthak Mehrotra, and Biplab Banerjee. REJEPA: A novel joint-embedding predictive architecture for efficient remote sensing image retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 2398–2407, June 2025. 2

[4] Shabnam Choudhury, Yash Salunkhe, Vaibhav Rajan, Subhasis Chaudhuri, and Biplab Banerjee. X-JEPA: A novel joint learning cross-modal predictive alignment framework for remote sensing image retrieval. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 4355–4364, 2026. 1, 2, 5, 6, 7, 8

[5] Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David Lobell, and Stefano Ermon. SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 197–211, 2022. 1, 2

[6] M. Drusch, U. Del Bello, S. Carlier, O. Colin, V. Fernandez, F. Gascon, B. Hoersch, C. Isola, P. Laberinti, P. Martimort, A. Meygret, F. Spoto, O. Sy, F. Marchese, and P. Bargellini. Sentinel-2: ESA’s optical high-resolution mission for GMES operational services. Remote Sensing of Environment, 120:25–36, 2012. 1

[7] Fabio Ferreira, Ivo Rapant, Jorg K. H. Franke, and Frank¨ Hutter. Beyond random augmentations: Pretraining with hard views. In International Conference on Learning Representations (ICLR), 2025. 2

[8] Anthony Fuller, Koreen Millard, and James R. Green. CROMA: Remote sensing representations with contrastive radar-optical masked autoencoders. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pages 5506–5538, 2023. 1, 2, 6, 8

[9] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In Proceedings of the First Conference on Language Modeling (COLM), 2024. 1, 2

[10] Xin Guo, Jiangwei Lao, Bo Dang, Yingying Zhang, Lei Yu, Lixiang Ru, Liheng Zhong, Ziyuan Huang, Kang Wu, Dingxiang Hu, Huimei He, Jian Wang, Jingdong Chen, Ming Yang, Yongjun Zhang, and Yansheng Li. SkySense: A multimodal remote sensing foundation model towards universal interpretation for earth observation imagery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27662–27673, 2024. 2, 6, 8

[11] Jakob Hackstein, Gencer Sumbul, Kai Norman Clasen, and Begum Demir. Exploring masked autoencoders for sensor-¨

agnostic image retrieval in remote sensing. IEEE Transactions on Geoscience and Remote Sensing, 63:1–14, 2025. 2

[12] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable´ vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15979–15988, 2022. 1, 2

[13] Md Aminur Hossain, Ayush V. Patel, Nitant Dube, and Biplab Banerjee. CR-JEPA: Cross-modal joint-embedding predictive learning for remote sensing image retrieval. arXiv preprint arXiv:2606.00706, 2026. 1, 2, 6, 7, 8

[14] Yansheng Li, Yongjun Zhang, Xin Huang, and Jiayi Ma. Learning source-invariant deep hashing convolutional neural networks for cross-source remote sensing image retrieval. IEEE Transactions on Geoscience and Remote Sensing, 56(11):6521–6536, 2018. 1, 5

[15] Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Jianbin Jiao, and Yunfan Liu. VMamba: Visual state space model. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pages 103031–103063, 2024. 2

[16] Mubashir Noman, Muzammal Naseer, Hisham Cholakkal, Rao Muhammad Anwar, Salman Khan, and Fahad Shahbaz Khan. Rethinking transformers pre-training for multispectral satellite imagery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27811–27819, 2024. 1, 2, 6, 8

[17] Colorado J. Reed, Ritwik Gupta, Shufan Li, Sarah Brockman, Christopher Funk, Brian Clipp, Kurt Keutzer, Salvatore Candido, Matt Uyttendaele, and Trevor Darrell. Scale-MAE: A scale-aware masked autoencoder for multiscale geospatial representation learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4065–4076, 2023. 2, 6, 8

[18] Sucheng Ren, Xianhang Li, Haoqin Tu, Feng Wang, Fangxun Shu, Lei Zhang, Jieru Mei, Linjie Yang, Peng Wang, Heng Wang, Alan Yuille, and Cihang Xie. Autoregressive pretraining with Mamba in vision. In International Conference on Learning Representations (ICLR), pages 81451–81465, 2025. 2

[19] Gencer Sumbul, Marcela Charfuelan, Begum Demir, and¨ Volker Markl. BigEarthNet: A large-scale benchmark archive for remote sensing image understanding. In Proceedings of the IEEE International Geoscience and Remote Sensing Symposium (IGARSS), pages 5901–5904, 2019. 2, 5

[20] Gencer Sumbul, Arne de Wall, Tristan Kreuziger, Filipe Marcelino, Hugo Costa, Pedro Benevides, Mario Caetano,´ Begum Demir, and Volker Markl. BigEarthNet-MM: A¨ large-scale, multimodal, multilabel benchmark archive for remote sensing image classification and retrieval. IEEE Geoscience and Remote Sensing Magazine, 9(3):174–180, 2021. 1, 2

[21] Yuxi Sun, Shanshan Feng, Yunming Ye, Xutao Li, Jian Kang, Zhichao Huang, and Chuyao Luo. Multisensor fusion and explicit semantic preserving-based deep hashing for cross-modal remote sensing image retrieval. IEEE Transactions on Geoscience and Remote Sensing, 60:1–14, 2022. 2, 5

[22] Maofeng Tang, Andrei Cozma, Konstantinos Georgiou, and Hairong Qi. Cross-scale MAE: A tale of multiscale exploitation in remote sensing. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pages 20054– 20066, 2023. 2

[23] Ramon Torres, Paul Snoeij, Dirk Geudtner, David Bibby, Malcolm Davidson, Evert Attema, Pierre Potin, Bjorn Rom-¨ men, Nicolas Floury, Mike Brown, Ignacio Navas Traver, Patrick Deghaye, Berthyl Duesmann, Betlem Rosich, Nuno Miranda, Claudio Bruno, Michelangelo L’Abbate, Renato Croci, Andrea Pietropaolo, Markus Huchler, and Friedhelm Rostan. GMES Sentinel-1 mission. Remote Sensing ofEnvironment, 120:9–24, 2012. 1

[24] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 5

[25] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, pages 5998–6008, 2017. 3

[26] Di Wang, Qiming Zhang, Yufei Xu, Jing Zhang, Bo Du, Dacheng Tao, and Liangpei Zhang. Advancing plain vision transformer toward remote sensing foundation model. IEEE Transactions on Geoscience and Remote Sensing, 61:1–15, 2023. 2, 6, 8

[27] Fengxiang Wang, Yulin Wang, Mingshuo Chen, Haotian Wang, Hongzhen Wang, Haiyan Zhao, Yangang Sun, Shuo Wang, Di Wang, Long Lan, Wenjing Yang, and Jing Zhang. RoMA: Scaling up Mamba-based foundation models for remote sensing. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pages 4669–4690, 2025. 2

[28] Yi Wang, Conrad M. Albrecht, Nassim Ait Ali Braham, Chenying Liu, Zhitong Xiong, and Xiao Xiang Zhu. Decoupling common and unique representations for multimodal self-supervised learning. In Proceedings of the European Conference on Computer Vision (ECCV), Lecture Notes in Computer Science, pages 286–303. Springer, 2024. 1, 2, 6, 8

[29] Yi Wang, Conrad M. Albrecht, and Xiao Xiang Zhu. Selfsupervised vision transformers for joint SAR-optical representation learning. In Proceedings of the IEEE International Geoscience and Remote Sensing Symposium (IGARSS), pages 139–142, 2022. 2

[30] Wei Xiong, Zhenyu Xiong, Yang Zhang, Yaqi Cui, and Xiangqi Gu. A deep cross-modality hashing network for SAR and optical remote sensing images retrieval. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 13:5284–5296, 2020. 2

[31] Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision Mamba: Efficient visual representation learning with bidirectional state space model. In Proceedings of the 41st International Conference on Machine Learning (ICML), volume 235 of Proceedings ofMachine Learning Research, pages 62429–62442. PMLR, 2024. 1, 2, 3