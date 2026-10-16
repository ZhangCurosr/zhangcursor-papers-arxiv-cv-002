# LAMP: Layerwise Adaptive Mixing of Predictors for Training-Free Few-Shot Vision–Language Adaptation

Zhipeng Ye<sup>a</sup>, Feng Jiang<sup>a,∗</sup>, Qiufeng Wang<sup>b</sup>, Hao Li<sup>c</sup>

<sup>a</sup>Taizhou Institute of Science and Technology, Nanjing University of Science and Technology, Taizhou 225300, Jiangsu, China

<sup>b</sup>Department of Intelligence Science, Xi’an Jiaotong-Liverpool University, Suzhou 215123, Jiangsu, China <sup>c</sup>School ofComputer Science and Technology, University ofArizona, Tucson 85705, AZ, USA

## Abstract

Few-shot adaptation often relies on an encoder’s final global representation, leaving complementary information in intermediate layers underused. Combining these features is challenging because their dimensions can difer and their usefulness varies across tasks. We propose LAMP, Layerwise Adaptive Mixing of Predictors, a training-free framework that combines global and regional information from multiple encoder layers. A single forward pass per view supplies all descriptors. Global and support-calibrated regional kernel predictors independently fit the same label residuals. Labeled validation data select their mixture in class-score space, preserving native feature dimensions without learned projections. LAMP-F further adapts text-anchored class prototypes, using a cached response to update the fixed-kernel residual predictions exactly. Experiments cover eleven datasets and ViT and ResNet backbones. LAMP achieves state-of-the-art mean accuracy among the compared published training-free ResNet-50 methods at 4, 8, and 16 shots, reaching 78.73% at 16 shots. Averaged across five support budgets, LAMP improves over a paired global-feature-only baseline by 1.14 and 1.59 percentage points on ViT-B/16 and ResNet-50; LAMP-F raises these gains to 1.96 and 2.54 points. LAMP-F reaches 16-shot accuracies of 83.57% and 79.10%, respectively. Controlled experiments identify multi-layer integration as the main source of improvement and examine pooling, calibration, validation labels, and computational cost. These findings show that combining layer-specific predictions can exploit complementary features from frozen encoders to improve few-shot classification.

Keywords: Few-shot classification, Vision–language models, Training-free adaptation, Kernel residual learning, Layerwise predictor mixing

## 1. Introduction

Vision–language models such as CLIP recognize images by matching visual representations to category descriptions [1]. Few-shot adaptation uses a small labeled support set to improve this pretrained predictor for a target task. Many adaptation methods rely on the encoder’s final global descriptor. This compact representation supports image– text matching, but intermediate token grids and convolutional maps also retain visual information at diferent levels of abstraction. These features are already available during the encoder’s forward pass. Using them ofers an opportunity to improve adaptation while keeping the encoder frozen.

Existing approaches use diferent sources of information to adapt the predictor. Prompt and prototype methods refine the class representations [2–4]. Tip-Adapter retrieves labeled instances, while ProKeR fits a regularized support-label residual; both use final global descriptors to match images [5, 6]. SSP instead uses local features to construct subspaces for image–text alignment [7]. These methods motivate the use of complementary semantic and visual information. Integrating features from several encoder layers, however, requires addressing two related challenges: their native dimensions can difer, and the most useful layers can vary with the target task.

Our central idea is to combine predictions from diferent layers in a shared classscore space. Diferent layers provide diferent measures of image similarity, but each predictor outputs a score for the same set of classes. Their predictions can therefore be combined while preserving the original feature dimensions. Fitting these predictors separately also allows their outputs to be reused when selecting the mixture, so each candidate mixture does not require a new fit.

Based on this idea, we introduce training-free LAMP, Layerwise Adaptive Mixing of Predictors. A single encoder pass per view yields a global descriptor and regional descriptors from three layers of a Vision Transformer (ViT) or ResNet [8, 9]. The global predictor and three regional predictors independently fit the same label residual: the diference between the support labels and CLIP’s initial scores. Each regional kernel is calibrated against the support set, and labeled validation data select the mixture of global and regional corrections. LAMP thus uses complementary information from several layers without gradient optimization or learned feature projections. Its extension, LAMP-F, adapts text-anchored class prototypes while retaining the selected kernels and mixture weights. A cached linear response accounts exactly for the resulting changes to both the initial scores and residual corrections, avoiding repeated matrix factorization during prototype evaluation.

We evaluate LAMP on eleven datasets, five support budgets, and CLIP backbones from the ViT and ResNet families. In the comparison with published training-free ResNet-50 (RN50) methods, LAMP achieves state-of-the-art mean accuracy at 4, 8, and 16 shots, reaching 78.73% at 16 shots. LAMP-F reaches 83.57% and 79.10% at 16 shots on ViT-B/16 and RN50, respectively. Across the eleven datasets and five support budgets, LAMP improves over our paired global-feature-only baseline by an average of 1.14 and 1.59 percentage points; LAMP-F raises these gains to 1.96 and 2.54 points. Controlled comparisons identify the integration of multiple encoder layers as the main source of improvement and examine the efects of regional pooling and calibration. Experiments with RN101 and ViT-B/32, together with transfer to ImageNet-V2 and ImageNet-Sketch, assess generalization across architectures and image distributions [10, 11].

Our contributions are:

• We introduce LAMP, a training-free method that combines global and regional information from multiple encoder layers using independently fitted kernel predictors. Regional kernels are calibrated against the support set. Score-level fusion preserves native feature dimensions, and separating branch fitting from mixture selection makes predictions reusable across candidates.

• We develop LAMP-F, which combines prototype adaptation with the same global and regional predictors. For fixed kernels and mixture weights, its cached response exactly updates both the initial class scores and residual corrections as the prototypes change.

• We demonstrate state-of-the-art mean accuracy in the published training-free RN50 comparison at 4, 8, and 16 shots across eleven datasets. Paired comparisons on ViT and ResNet backbones establish the accuracy gains, and controlled experiments explain the contribution of multi-layer integration.

## 2. Related work

## 2.1. Class-level semantic adaptation

Semantic adaptation adjusts a vision–language model’s class representations to a target task. CoOp and CoCoOp learn contextual prompts, while KgCoOp and ProGrad constrain prompt updates to preserve pretrained knowledge [2, 12–14]. MaPLe couples visual and textual prompts; PLOT and CAPEL use multiple prompts or class representations to capture variation within a category [15–17]. TaskRes learns residual updates to class prototypes, while CLAP adapts prototypes with a text-anchoring penalty [3, 4]. These methods improve the class representations through optimization. LAMP keeps the class text vectors fixed and uses visual similarities to correct their prediction errors on the support set. Its trainable extension, LAMP-F, combines CLAP’s text anchoring with global and regional residual predictors. Cached responses account for the efects of prototype updates on both the initial query scores and the support residuals, allowing prototype adaptation to work with the selected visual predictors.

## 2.2. Instance-level global adaptation

Instance-level adaptation draws on visual information from labeled support images. CLIP-Adapter learns transformations of global embeddings, and CLIP-LoRA updates internal attention weights through low-rank parameters [18, 19]. Both adapt visual representations through gradient optimization. Training-free methods instead adjust how frozen features are used for prediction: Tip-Adapter retrieves support labels, APE refines feature channels and afinities, and GDA estimates class statistics [5, 20, 21]. IDEA combines image similarities with similarities between image captions, and T-IDEA further learns alignment and bias correction [22]. LAMP obtains complementary features from intermediate encoder layers and combines their predictions. This allows it to use information beyond the final global descriptor while preserving each layer’s original feature space.

LAMP uses ProKeR’s residual kernel formulation, which transfers corrections from support images to queries according to global-feature similarities [6]. It fits a global predictor and separate regional predictors whose kernels are calibrated against the support set. Combining their outputs in class-score space preserves each branch’s feature dimensions and similarity structure; labeled validation data determine their relative contributions. Since the branches are fitted independently, their predictions can be reused when evaluating diferent mixtures. LAMP-F also exploits their linear dependence on the support residual to cache the combined response to prototype updates.

## 2.3. Patch- and region-level adaptation

Selective Vision-Language Subspace Projection (SSP) uses local features to construct visual and class-specific language subspaces for image–text alignment without gradient optimization [7]. Both SSP and LAMP use information beyond a single global descriptor, but they integrate it diferently. SSP improves cross-modal alignment through subspace projection. LAMP uses regional features from several encoder layers to fit the same support-label residual and then combines the resulting class scores. Because all branches predict the same classes, their outputs can be combined even when their feature dimensions difer. A single encoder pass per view provides all regional descriptors. LAMP uses generalized-mean pooling for convolutional activations [23] and kernel-mean comparisons of descriptor sets [24]. Comparisons with simpler summaries of the same layers distinguish the benefit of using multiple layers from that of retaining regional descriptor sets.

## 3. Method

LAMP adapts CLIP by using labeled support images to correct its initial class scores. The support labels indicate which scores should increase or decrease, and visual similarity determines how these corrections transfer to a query. Global descriptors and regional features from intermediate layers capture diferent relationships between images, providing complementary ways to correct a prediction. LAMP combines the resulting corrections in class-score space: every branch produces one score per class, even when the branches have diferent feature dimensions.

We first introduce the residual target and global predictor using ProKeR’s residual kernel formulation [6]. We then explain how intermediate layers provide regional descriptors, how their similarities are calibrated, and how the resulting predictors fit the same target. Validation determines the contribution of each predictor. LAMP-F further adapts the class prototypes while keeping the selected visual kernels and mixture weights fixed. Figure 1 illustrates the method with a ViT backbone.

## 3.1. Residual prediction around the CLIP prior

Consider C classes with K labeled images each. For support image $i \in \{ 1 , . . . , C K \}$ with class label $y _ { i } \in \{ 1 , \ldots , C \}$ , frozen CLIP [1] encodes $V = 1 0$ augmented views $x _ { i , \nu } ,$ $\nu \in \{ 1 , \ldots , V \}$ . These views form the ordered support bank $\boldsymbol { S } = ( s _ { 1 } , \ldots , s _ { N } )$ , where

LAMP: Layerwise Adaptive Mixing of Predictors  
![](images/75ad66ad46fd18f11749e363119e6cb467187db2437d4d1bb040131dd2c3df9b.jpg)  
Figure 1: Training-free LAMP with ViT-B/16 for one query q. (a) The frozen encoder provides global descriptors, query regional matrices $\mathbf { Q } _ { \ell } ( q ) .$ and support tensors $\mathbf { s } _ { \ell } ;$ branches $\ell = 1 , 2 , 3$ correspond to blocks 4, 8, and 12. Row labels $1 , 2 , 3 , \ldots , C$ denote class indices. (b) Independent predictors fit the shared support-label residual R . The efective kernel $\mathcal { K } _ { b }$ is $K _ { g }$ for the global branch and $\widehat { K } _ { \ell }$ for a calibrated regional branch. (c) Their corrections $\Delta _ { b } ( q )$ are weighted and added to the CLIP prior. Here $N = V C K$ , V = 10, J = 4, D = 768, and $N _ { p } = 1 9 6$ . Prediction uses the selected parameters; Section 3.5 describes LAMP-F. Asterisks mark the displayed support subset and illustrative prediction.

$N = V C K ;$ each entry inherits its source image’s label. The support budget therefore counts CK labeled images, and the kernel bank contains N augmented views. The ordered query collection $Q = ( q _ { 1 } , \dots , q _ { M } )$ contains one deterministic view per query.

Vectors are column vectors unless stated otherwise; feature matrices contain their transposes as rows. In particular, $\mathbf { S } _ { g } \in \mathbb { R } ^ { N \times d _ { g } }$ and $\mathbf { Q } _ { g } \in \mathbb { R } ^ { M \times d _ { g } }$ contain unit global descriptors $\mathbf { g } ( x ) \in \mathbb { R } ^ { d _ { g } }$ . The matrix $\mathbf { Y } \in \{ 0 , 1 \} ^ { N \times C }$ contains one-hot support-view labels, and $\mathbf { T } \in \mathbb { R } ^ { C \times d _ { g } }$ contains unit class text vectors from fixed prompts as rows. For one query q, $\mathbf { q } _ { g } = \mathbf { g } ( q ) ^ { \top } \in \mathbb { R } ^ { 1 \times d _ { g } }$ is its row in $\mathbf { Q } _ { g }$ . We write norm for L2 normalization of a vector and row-wise L2 normalization of a matrix. Bold symbols denote vector-, matrix-, or tensor-valued quantities; calligraphic S and Q denote ordered collections.

The text matrix T defines both the initial query scores and the support scores used to form the residual target. Each CLIP score is the dot product of an image descriptor and a class text vector. For a support view, subtracting these scores from its one-hot label gives the desired correction for each class. All branches fit this same residual:

$$
\mathbf { F } _ { 0 } ( { \boldsymbol { Q } } ) = \mathbf { Q } _ { g } \mathbf { T } ^ { \top } , \qquad \mathbf { R _ { T } } = \mathbf { Y } - \mathbf { S } _ { g } \mathbf { T } ^ { \top } .\tag{1}
$$

The global branch transfers these corrections using similarities between global descriptors [6]. The following kernel gives higher similarity to descriptors with more closely aligned directions. For unit vectors, the bandwidth $\beta$ controls how rapidly similarity decreases with angular separation:

$$
k _ { \beta } ( \mathbf { a } , \mathbf { b } ) = \exp [ - \beta ( 1 - \mathbf { a } ^ { \top } \mathbf { b } ) ] , \qquad \beta > 0 .\tag{2}
$$

Throughout, x and $z$ denote arbitrary image views; the arguments of a kernel determine whether each view comes from the support or query collection. Let $K _ { g } ( x , z ) =$ $k _ { \beta _ { g } } ( \mathbf { g } ( \boldsymbol { x } ) , \mathbf { g } ( z ) )$ . Evaluating a kernel on two ordered collections gives the matrix of all pairwise similarities. The matrix $\mathbf { G } _ { g } = K _ { g } ( S , S )$ records similarities among support entries, while $\mathbf { H } _ { g } = K _ { g } ( Q , S )$ links queries to support entries. We solve a linear system for the coeficient matrix $\alpha _ { g } ,$ , which has one coeficient per support entry and class. The matrix $\mathbf { G } _ { g }$ accounts for relationships among the support entries, so their coeficients are determined jointly. Multiplication by $\mathbf { H } _ { g }$ transfers the fitted corrections to each query. With regularization parameter $\lambda > 0$ and identity matrix ${ \bf I } _ { N } \in \mathbb { R } ^ { N \times N }$

$$
\alpha _ { g } = ( { \bf I } _ { N } + { \bf G } _ { g } / \lambda ) ^ { - 1 } { \bf R _ { T } } , \qquad { \bf F } _ { \mathrm { g l o b a l } } ( Q ) = { \bf F } _ { 0 } ( Q ) + { \bf H } _ { g } \alpha _ { g } .\tag{3}
$$

Thus $\mathbf { H } _ { g } \pmb { \alpha } _ { g }$ supplies the class-score adjustment added to the CLIP prior. The solve has an equivalent quadratic characterization: it keeps the coeficients close to the support residual target while penalizing the squared kernel-space norm of the correction. This penalty couples the coeficients through $\mathbf { G } _ { g } ,$ yielding the linear system in Eq. (3). Since $( \mathbf { I } _ { N } + \mathbf { G } _ { g } / \lambda ) ^ { - 1 } = \lambda ( \mathbf { G } _ { g } + \lambda \mathbf { I } _ { N } ) ^ { - 1 }$ , λ controls both regularization and correction amplitude. All branches use this parameterization with the unscaled CLIP prior.

## 3.2. Extracting native-layer regional descriptors

Intermediate feature grids retain regional information from diferent stages of the encoder. A single encoder pass per view yields $J = 4$ regional descriptors at each of three selected layers, indexed by $\ell \in \{ 1 , 2 , 3 \}$ . A fixed $2 \times 2$ pooling grid summarizes the content of four spatial regions. The pooling operation follows the structure of each backbone: ViT averages normalized token directions, while RN50 retains activation strength during pooling. Each descriptor $\mathbf { v } _ { \ell , a } ( x )$ retains the original channel width $D _ { \ell }$ of its layer.

ViT-B/16. Branches $\ell = 1 , 2 , 3$ use blocks 4, 8, and 12, respectively. For input view $x , \mathbf { t } _ { \ell , u } ( x ) \in \mathbb { R } ^ { D _ { \ell } }$ denotes the patch token at spatial position u in the selected block. At each layer, ${ \cal T } _ { a }$ contains the token positions in spatial bin $a \in \{ 1 , \ldots , J \}$ , and $| { \cal T } _ { a } |$ is the number of positions in that bin. We apply parameter-free LayerNorm, denoted by LN, across each token’s channels, followed by L2 normalization and regional averaging:

$$
\mathbf { v } _ { \ell , a } ( x ) = \mathrm { n o r m } _ { 2 } \left( \frac { 1 } { | \mathcal { T } _ { a } | } \sum _ { u \in \mathcal { I } _ { a } } \mathrm { n o r m } _ { 2 } ( \mathrm { L N } ( \mathbf { t } _ { \ell , u } ( x ) ) ) \right) ,\tag{4}
$$

Each norm operation divides its input vector by its L2 norm. The inner operation gives each token direction equal weight before averaging; the outer operation makes the resulting regional descriptor a unit vector. Thus Eq. (4) first normalizes individual tokens, averages them within each region, and then normalizes the regional average. All four regional vectors are retained, together with the projected CLS descriptor used by the global branch.

RN50. Branches $\ell = 1 , 2 , 3$ use C3, C4, and C5, respectively. These stages provide nonnegative post-ReLU feature maps. Generalized-mean (GeM) pooling [23] aggregates each of the four bins in an adaptive $2 \times 2$ grid:

$$
u _ { \ell , a , d } ( x ) = \left( \frac { 1 } { | { \cal T } _ { a } | } \sum _ { u \in { \cal T } _ { a } } h _ { \ell , d , u } ( x ) ^ { 3 } \right) ^ { 1 / 3 } , \qquad { \bf v } _ { \ell , a } ( x ) = \mathrm { n o r m } _ { 2 } ( { \bf u } _ { \ell , a } ( x ) ) .\tag{5}
$$

Here $h _ { \ell , d , u }$ is the scalar activation of channel $d \in \{ 1 , \ldots , D _ { \ell } \}$ at position u, and $\mathbf { u } _ { \ell , a }$ stacks the pooled channel values $u _ { \ell , a , d }$ . The fixed exponent of three emphasizes strong responses. Normalization is applied after pooling so that activation magnitudes contribute to the regional summary. RN50 batch-normalization (BatchNorm) statistics remain frozen. The global branch uses the original attention-pooled feature.

For a single query $q ,$ the regional descriptor matrix $\mathbf { Q } _ { \ell } ( q ) \in \mathbb { R } ^ { J \times D _ { \ell } }$ has row a equal to $\mathbf { v } _ { \ell , a } ( q ) ^ { \intercal }$ . Applying the same notation to each support view $s _ { i } .$ the support tensor $\mathbf { S } _ { \ell } \in \mathbb { R } ^ { N \times J \times D _ { \ell } }$ has ith slice $\mathbf { Q } _ { \ell } ( s _ { i } )$ . Thus $\mathbf { Q } _ { \ell } ( q )$ describes one image, while $\boldsymbol { Q }$ denotes the collection of queries.

## 3.3. Calibrating regional kernels

Augmentation can move the same visual content into diferent regions of two image views. To match this content across positions, we compare every region in one image with every region in the other and average their kernel similarities. We then normalize this average by the images’ self-similarities so that the resulting set embeddings have a common norm. At each depth, the shared bandwidth is $\beta _ { f } = r \beta _ { g }$ , where r is the regional-to-global bandwidth ratio. Define

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { \ell } ( x , z ) = \frac { 1 } { J ^ { 2 } } \sum _ { a = 1 } ^ { J } \sum _ { a ^ { \prime } = 1 } ^ { J } k _ { \beta _ { f } } ( { \bf v } _ { \ell , a } ( x ) , { \bf v } _ { \ell , a ^ { \prime } } ( z ) ) , } } \\ { { \displaystyle K _ { \ell } ( x , z ) = \frac { { \cal A } _ { \ell } ( x , z ) } { \sqrt { { \cal A } _ { \ell } ( x , x ) { \cal A } _ { \ell } ( z , z ) } } . } } \end{array}\tag{6}
$$

The average $A _ { \ell }$ is a standard kernel-mean construction [24]: each image is represented by the mean of its regional embeddings in kernel space. This comparison captures similarity between sets of regional descriptors and is invariant to their ordering. Dividing by the geometric mean of self-similarities removes variation in set-embedding norms and gives $K _ { \ell } ( x , x ) = 1$

We next measure similarity relative to a common reference defined by the support bank. Let $\psi _ { \ell } ( x )$ be the unit-norm kernel-space embedding associated with $K _ { \ell } ,$ so that $K _ { \ell } ( x , z ) = \langle \pmb { \psi } _ { \ell } ( x ) , \pmb { \psi } _ { \ell } ( z ) \rangle$ ⟩. Its support mean is $\begin{array} { r } { \bar { \pmb { \psi } } _ { \ell } = N ^ { - 1 } \sum _ { i } \pmb { \psi } _ { \ell } ( s _ { i } ) } \end{array}$ . Centering subtracts this same mean from every support and query embedding. The required statistics are the mean similarity $m _ { \ell } ( x )$ between a view and the support bank, and the mean similarity $b _ { \ell }$ among support entries:

$$
m _ { \ell } ( x ) = { \frac { 1 } { N } } \sum _ { i } K _ { \ell } ( x , s _ { i } ) , \qquad b _ { \ell } = { \frac { 1 } { N ^ { 2 } } } \sum _ { i , j } K _ { \ell } ( s _ { i } , s _ { j } ) .\tag{7}
$$

The same definition gives $m _ { \ell } ( z )$ for the other view. These statistics allow centered inner products to be computed directly from kernel values:

$$
\begin{array} { c } { { \langle \psi _ { \ell } ( x ) - \bar { \psi } _ { \ell } , \psi _ { \ell } ( z ) - \bar { \psi } _ { \ell } \rangle } } \\ { { = K _ { \ell } ( x , z ) - m _ { \ell } ( x ) - m _ { \ell } ( z ) + b _ { \ell } . } } \end{array}
$$

Thus both views are compared relative to the support mean in kernel space. Since $K _ { \ell } ( x , x ) = 1$ , the squared length after centering is $1 - 2 m _ { \ell } ( x ) + b _ { \ell }$ . We then normalize the centered embeddings to control variation in their lengths. With $d _ { \ell } ^ { + } ( x ) = \operatorname* { m a x } \{ 1 -$ $2 m _ { \ell } ( x ) + b _ { \ell } , 1 0 ^ { - 1 0 } \}$

$$
\widehat { K } _ { \ell } ( x , z ) = \frac { K _ { \ell } ( x , z ) - m _ { \ell } ( x ) - m _ { \ell } ( z ) + b _ { \ell } } { \sqrt { d _ { \ell } ^ { + } ( x ) d _ { \ell } ^ { + } ( z ) } } .\tag{8}
$$

The numerator measures similarity between the centered embeddings; the denominator normalizes their magnitudes, with a floor to ensure numerical stability. Support and query images use the same support reference, so each query is calibrated independently of other queries in its batch. The global branch retains its uncentered kernel.

## 3.4. Fitting and selecting the residual mixture

Diferent layers can assign diferent similarities to the same pair of images. Each branch therefore provides a distinct way to transfer the shared support residual to queries. Let $\mathcal { B } = \{ g , 1 , 2 , 3 \}$ index all branches, with $\mathcal { K } _ { g } = K _ { g }$ and $\mathcal { K } _ { \ell } = \widehat { K } _ { \ell }$ for $\ell = 1 , 2 , 3$ . This notation covers both the uncentered global kernel and the calibrated regional kernels. Each branch $b \in { \mathcal { B } }$ uses its own kernel matrices and is fitted independently:

$$
\mathbf { G } _ { b } = \mathcal { K } _ { b } ( S , S ) , \quad \mathbf { H } _ { b } = \mathcal { K } _ { b } ( Q , S ) , \quad \alpha _ { b } = ( \mathbf { I } _ { N } + \mathbf { G } _ { b } / \lambda ) ^ { - 1 } \mathbf { R } _ { \mathbf { T } } .\tag{9}
$$

Each branch fits the same $\mathbf { R } _ { \mathbf { T } }$ through its own $\mathbf { G } _ { b } ,$ , then produces the correction $\Delta _ { b } ( Q ) =$ $\mathbf { H } _ { b } \pmb { \alpha } _ { b } \in \mathbb { R } ^ { M \times C }$ . For a single query, write $\mathbf { H } _ { b } ( q ) = \mathcal { K } _ { b } ( q , S ) \in \mathbb { R } ^ { 1 \times N }$ and $\Delta _ { b } ( q ) =$ $\mathbf { H } _ { b } ( q ) \pmb { \alpha } _ { b } \in \mathbb { R } ^ { 1 \times C }$ . Diferent depths therefore use their own image relationships to predict corrections to the same class scores. Independent fitting preserves these distinct relationships, while the common C-dimensional output allows the predictions to be fused with each layer’s native feature dimension retained. The weight vector $\mathbf { w } = ( w _ { 1 } , w _ { 2 } , w _ { 3 } ) ^ { \top }$ sets the relative contributions of the regional branches, and $\rho$ balances their combined correction against the global correction. For $\begin{array} { r } { w _ { \ell } \geq 0 , \sum _ { \ell = 1 } ^ { 3 } w _ { \ell } = 1 } \end{array}$ , and $\rho \in [ 0 , 1 ]$ , predict

$$
\mathbf { F } _ { \mathbf { T } } ( { \boldsymbol { Q } } ) = \mathbf { F } _ { 0 } ( { \boldsymbol { Q } } ) + ( 1 - \rho ) \mathbf { H } _ { g } \alpha _ { g } + \rho \sum _ { \ell = 1 } ^ { 3 } w _ { \ell } \mathbf { H } _ { \ell } \alpha _ { \ell } .\tag{10}
$$

The single-query prediction $\mathbf { F } _ { \mathbf { T } } ( q ) \in \mathbb { R } ^ { 1 \times C }$ is the corresponding row of $\mathbf { F } _ { \mathbf { T } } ( \boldsymbol { Q } )$ . Classification assigns the class with the largest score. The predictor starts with the CLIP prior and adds the weighted global and regional corrections. The independently fitted predictions are reused across candidate mixture weights. The setting $\rho = 0$ recovers the global predictor.

Validation first selects the global bandwidth and regularization $\lambda ,$ then the regional bandwidth and mixture weights. All regional branches use the selected λ. These parameters remain fixed during subsequent prototype adaptation and evaluation.

## 3.5. Prototype adaptation with fixed kernels

Each row of T is a text-derived representative vector, or prototype, for one class. LAMP-F adapts these vectors to the target task using the labeled support images. A prototype update changes both the initial query scores and the support scores used to form the residual target. We account for both efects while keeping the selected visual kernels and mixture weights fixed. The trainable matrix $\mathbf { W } \in \mathbb { R } ^ { C \times d _ { g } }$ is initialized to T:

$$
\mathbf { \Gamma } \mathbf { \Gamma } ( \mathbf { W } ) = \operatorname { n o r m } _ { 2 } ( \mathbf { W } ) - \operatorname { n o r m } _ { 2 } ( \mathbf { T } ) , \qquad \mathbf { \Gamma } \mathbf { P } ( \mathbf { W } ) = \mathbf { T } + \mathbf { \Gamma } \mathbf { \Gamma } ( \mathbf { W } ) ,\tag{11}
$$

Here Γ(W) is the change in class directions after row-wise normalization, and $\mathbf { P } ( \mathbf { W } )$ is the adapted prototype matrix. With unit rows in T, P(W) equals norm<sub>2</sub>(W) in exact arithmetic. Writing the update as a diference also makes $\begin{array} { r } { \Gamma ( \mathbf { T } ) = \mathbf { 0 } . } \end{array}$ , so initialization recovers LAMP exactly. Prototype training uses one mean descriptor per support image, $\begin{array} { r } { \bar { \bf g } _ { i } = V ^ { - 1 } \sum _ { \nu = 1 } ^ { V } { \bf g } ( x _ { i , \nu } ) } \end{array}$ . This mean is used directly, so its linear class scores equal the average scores across views. Following CLAP [4], we optimize

$$
\mathcal { L } ( \mathbf { W } ) = \frac { 1 } { C K } \sum _ { i = 1 } ^ { C K } \mathrm { C E } ( 1 0 0 \mathbf { P } ( \mathbf { W } ) \bar { \mathbf { g } } _ { i } , y _ { i } ) + \frac { 1 } { C } \sum _ { c = 1 } ^ { C } a _ { c } \| \mathbf { W } _ { c } - \mathbf { T } _ { c } \| _ { 2 } ^ { 2 } ,\tag{12}
$$

Here CE is cross-entropy, and $\mathbf { W } _ { c } , \mathbf { T } _ { c } \in \mathbb { R } ^ { 1 \times d _ { g } }$ are the cth rows of their respective matrices. The first term adapts the prior scores to the support labels; the second anchors each class parameter to its original text vector. The fixed weight $a _ { c }$ is the mean softmax confidence assigned to the correct class across the original views of class $^ { c , }$ computed from unscaled CLIP scores. A larger weight anchors the class prototype more strongly to its text vector.

AdamW updates the prototypes using the support-set classification loss and textprototype anchor in $\operatorname { E q . }$ (12). Checkpoint selection evaluates the complete residual predictor on labeled validation data. The cached response derived below accelerates these repeated full-predictor evaluations as the prototypes change. The factor 100 scales training logits. The kernel bank continues to use ten separate views per image.

The prototype change adds $\mathbf { Q } _ { g } \mathbf { \Gamma } \Gamma ( \mathbf { W } ) ^ { \top }$ to the query prior and changes the support residual to $\mathbf R _ { \mathbf T } - \mathbf S _ { g } \mathbf { \Gamma } \Gamma ( \mathbf { W } ) ^ { \top }$ . An improvement in the support prior therefore changes the correction that the kernel branches need to supply. For $b \in { \mathcal { B } } ,$ , let ${ \bf L } _ { b } = ( { \bf I } _ { N } + { \bf G } _ { b } / \lambda ) ^ { - 1 }$ and write $\alpha _ { b } ( { \bf P } )$ for the coeficients associated with an adapted prototype matrix $\mathbf { P } = \mathbf { P } ( \mathbf { W } )$ , with $\alpha _ { b } ( \mathbf { T } ) = \alpha _ { b }$ . Linearity in the residual gives

$$
\begin{array} { r l } & { \alpha _ { b } ( \mathbf { P } ) = \mathbf { L } _ { b } [ \mathbf { R } _ { \mathbf { T } } - \mathbf { S } _ { g } \boldsymbol { \Gamma } ( \mathbf { W } ) ^ { \top } ] } \\ & { \quad \quad \quad = \alpha _ { b } ( \mathbf { T } ) - \underbrace { \mathbf { L } _ { b } \mathbf { S } _ { g } } _ { \mathbf { B } _ { b } } \boldsymbol { \Gamma } ( \mathbf { W } ) ^ { \top } . } \end{array}
$$

The fixed matrix $\mathbf { B } _ { b }$ therefore converts a prototype change into a coeficient update, allowing repeated updates to reuse the same linear-system response. With $\pi _ { g } = 1 - \rho$ and $\pi _ { \ell } = \rho w _ { \ell }$ , we precompute the support responses and combine their efects on a query collection:

$$
\mathbf { B } _ { b } = \mathbf { L } _ { b } \mathbf { S } _ { g } , \qquad \mathbf { E } ( \pmb { Q } ) = \sum _ { b \in \mathcal { B } } \pi _ { b } \mathbf { H } _ { b } \mathbf { B } _ { b }\tag{13}
$$

The matrix $\mathbf { B } _ { b } \in \mathbb { R } ^ { N \times d _ { g } }$ is obtained by solving the linear system for branch b with the global support descriptors as its right-hand side. It is shared across query collections. Multiplying it by $\mathbf { H } _ { b }$ transfers the response to the queries, and the weighted sum gives $\mathbf { E } ( Q ) \in \mathbb { R } ^ { M \times d _ { g } }$ . This query-specific response can be reused across prototype checkpoints for the same collection. The adapted prediction is then exactly

$$
\mathbf { F } _ { \mathbf { P } ( \mathbf { W } ) } ( { \boldsymbol { Q } } ) = \mathbf { F } _ { \mathbf { T } } ( { \boldsymbol { Q } } ) + [ \mathbf { Q } _ { g } - \mathbf { E } ( { \boldsymbol { Q } } ) ] \Gamma ( \mathbf { W } ) ^ { \top } .\tag{14}
$$

The term $\mathbf { Q } _ { g } \mathbf { \mathbf { \mathbf { { T } } } } ( \mathbf { W } ) ^ { \top }$ updates the prior, while $- \mathbf { E } ( Q ) \Gamma ( \mathbf { W } ) ^ { \top }$ updates the fitted corrections for the changed support residual. Together, these terms account for both efects of prototype adaptation. Once the response is cached, evaluating a new prototype matrix only requires matrix products with $\mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \mathbf { \Gamma } } \mathbf { \Sigma } \mathbf { \mathbf { \Gamma } } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf \Sigma \mathbf { \Sigma } \mathbf { \Sigma } \mathbf \Sigma \Sigma \mathbf { } \Sigma \mathbf { \Sigma } \mathbf \Sigma \Sigma \mathbf { } \mathbf \Sigma \Sigma \Sigma \mathbf { } \mathbf \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \Sigma \Sigma \mathbf \Sigma \Sigma $ . LAMP-F optimizes only $C d _ { g }$ prototype parameters; the encoders, regional descriptors, and fusion weights remain frozen.

## 4. Experiments

We evaluate recognition accuracy at diferent support budgets and examine how class-prototype adaptation and native-layer integration contribute to performance. We also test transfer to new domains without fitting on target data and measure deployment costs. Comparisons with published methods establish overall performance, while paired controls assess individual design choices.

## 4.1. Evaluation protocol

We evaluate training-free LAMP and its prototype-adapted extension, LAMP-F, with CLIP encoders from the ViT and ResNet families. The main comparison covers eleven datasets using ViT-B/16 and RN50, with 1, 2, 4, 8, and 16 support images per class and three support-sampling seeds. Additional experiments use RN101 and ViT-B/32. We report top-1 accuracy and give each dataset equal weight when computing the mean.

We follow the established benchmark partitions and evaluation conventions used in previous studies [2, 5, 6]. Labeled validation images are not counted toward the support budget. Paired controls use the same support images, augmented views, prompts, and query images. Results taken from published baselines retain their original protocols. Confidence intervals for our experiments are conditional on the selected models.

Ten datasets have distinct validation and test lists. For ImageNet, support images come from the training partition, while the same oficial 50,000-image validation partition is used for both parameter selection and benchmark scoring, following the released Tip-Adapter and ProKeR implementations. The eleven-dataset aggregate therefore combines these evaluation protocols. Published comparisons retain their diferent selection procedures: CLIP-LoRA shares hyperparameters across tasks, whereas LAMP uses dataset-specific validation.

Paired analyses of the three-seed benchmark and its controls use 5,000 paired crossed bootstrap replicates, with shared support-seed and class-stratified query-identity draws. Selected models remain fixed during resampling. The benchmark tasks were used during method development; these intervals do not cover repeated-selection or development uncertainty.

## 4.2. Comparison with state-of-the-art methods

We compare LAMP and LAMP-F with representative published methods for prompt learning, feature adaptation, and support matching across five support budgets. Tables 1 and 2 report mean accuracy over eleven datasets for ViT-B/16 and RN50. The methods included in each comparison depend on their backbone requirements and the availability of published results. For example, MaPLe [15] inserts deep visual prompts into Transformer blocks and is evaluated with ViT-B/16. SSP [7] reports eleven-dataset mean accuracy at all five support budgets for RN50, but does not report the corresponding means for ViT-B/16. We therefore use the ViT-B/16 results reported by CLIP-LoRA and CAPEL [17, 19] to compare with methods that require training, and the RN50 results reported by ProKeR and SSP [6, 7] to compare with training-free methods.

LAMP achieves state-of-the-art mean accuracy among the training-free RN50 methods compared at 4, 8, and 16 shots. At 16 shots, it reaches 78.73%, exceeding the best published training-free result by 1.98 percentage points without gradient optimization.

The ViT-B/16 comparison includes published methods that require training. LAMP exceeds CLIP-LoRA, the strongest published baseline at 8 and 16 shots, by 0.35 and 0.30 points, respectively, without gradient-based training. LAMP-F further raises mean accuracy to 80.94% and 83.57%. CLIP-LoRA updates encoder weights, while LAMP keeps both the encoders and class prototypes fixed; LAMP-F optimizes only the class prototypes. These results show that combining residual corrections from multiple depths can achieve competitive accuracy with a frozen encoder and fixed text prototypes.

On RN50, training-free LAMP outperforms the strongest published method by 0.77, 1.18, and 1.98 points at 4, 8, and 16 shots, respectively. APE+SSP [7] achieves the highest training-free mean accuracy at 1 and 2 shots, while LAMP’s advantage grows with the number of support images. LAMP-F further improves the means to 65.91, 68.83, 72.65, 75.50, and 79.10%. Across the two backbones, native-layer residual integration is competitive with both training-free support matching and methods that require training, and prototype optimization provides further gains. Section 4.6 presents a paired comparison with independently trained ProKeR+CLAP prototypes.

Results by dataset. Table 3 compares representative published methods at 16 shots on all eleven datasets using ViT-B/16. LAMP-F achieves the highest accuracy among the listed methods on five datasets: ImageNet, EuroSAT, Food101, Flowers102, and DTD. It exceeds the best published scores shown for these datasets by 0.66, 1.58, 0.14, 0.90, and 2.74 points, respectively. The large gains on texture and satellite-image datasets motivate the analysis of native-layer features below. The methods have diferent strengths across datasets: CLIP-LoRA leads on Aircraft, Cars, and UCF101; CAPEL on SUN397 and Caltech101; and PLOT++ on Pets. Table 7 extends the comparison of training-free methods to ImageNet with four backbones.

Table 1: ViT-B/16 comparison: mean top-1 accuracy (%) over eleven datasets. Published scores are taken from CLIP-LoRA [19] and CAPEL [17]; results for LAMP and LAMP-F are averaged over three seeds. Boldface and underlining indicate the highest and second-highest means in each column. The numbers in CoOp (4)/(16) denote context length. Venue/year identifies each method’s publication; PLOT++ is the enhanced PLOT implementation used in CLIP-LoRA. PR denotes Pattern Recognition, and CVPRW denotes CVPR Workshops. Evaluation protocols are described in Section 4.1.
<table><tr><td>Method</td><td>Venue/year</td><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td></tr><tr><td>CoOp (4) [2]</td><td>IJCV, 2022</td><td>67.2</td><td>70.5</td><td>73.0</td><td>76.4</td><td>79.0</td></tr><tr><td>CoOp (16) [2]</td><td>IJCV, 2022</td><td>67.6</td><td>70.8</td><td>74.0</td><td>76.8</td><td>80.0</td></tr><tr><td>CoCoOp [12]</td><td>CVPR, 2022</td><td>68.8</td><td>70.4</td><td>71.7</td><td>73.8</td><td>75.4</td></tr><tr><td>Tip-Adapter-F [5]</td><td>ECCV, 2022</td><td>70.9</td><td>73.7</td><td>75.6</td><td>78.3</td><td>80.7</td></tr><tr><td>CLIP-Adapter [18]</td><td>IJCV, 2024</td><td>65.7</td><td>66.8</td><td>67.7</td><td>71.5</td><td>75.5</td></tr><tr><td>PLOT++ [16]</td><td>ICLR, 2023</td><td>70.7</td><td>74.0</td><td>76.9</td><td>79.6</td><td>82.1</td></tr><tr><td>KgCoOp [13]</td><td>CVPR, 2023</td><td>69.6</td><td>71.6</td><td>73.9</td><td>76.0</td><td>77.3</td></tr><tr><td>TaskRes [3]</td><td>CVPR, 2023</td><td>70.8</td><td>72.9</td><td>74.7</td><td>78.4</td><td>80.8</td></tr><tr><td>MaPLe [15]</td><td>CVPR, 2023</td><td>66.4</td><td>70.5</td><td>73.5</td><td>76.4</td><td>78.6</td></tr><tr><td>ProGrad [14]</td><td>ICCV, 2023</td><td>69.5</td><td>72.4</td><td>74.7</td><td>77.4</td><td>79.9</td></tr><tr><td>CLIP-LoRA [19]</td><td>CVPRW, 2024</td><td>72.5</td><td>75.5</td><td>77.4</td><td>80.3</td><td>83.0</td></tr><tr><td>CAPEL [17]</td><td>PR, 2026</td><td>72.97</td><td>74.90</td><td>77.64</td><td>80.14</td><td>82.45</td></tr><tr><td>LAMP</td><td>Ours</td><td>72.18</td><td>74.27</td><td>77.70</td><td>80.65</td><td>83.30</td></tr><tr><td>LAMP-F</td><td>Ours</td><td>73.28</td><td>75.62</td><td>78.80</td><td>80.94</td><td>83.57</td></tr></table>

Table 2: RN50 comparison: mean top-1 accuracy (%) over eleven datasets. Published scores are taken from Table 2 of ProKeR [6], including variants using random Fourier features (RFF), and Table 3 of SSP [7]. Results for LAMP and LAMP-F are averaged over three seeds. The published methods and LAMP are training-free; LAMP-F optimizes class prototypes. Boldface and underlining indicate the highest and second-highest values in each column across all rows. APE+SSP uses the CuPL prompts specified in the source publication. Venue/year identifies the publication reporting each method or RFF variant. Evaluation protocols are described in Section 4.1.
<table><tr><td>Method</td><td>Venue/year</td><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td></tr><tr><td>GDA [21]</td><td>ICLR, 2024</td><td>62.19</td><td>66.19</td><td>69.77</td><td>73.30</td><td>76.04</td></tr><tr><td>Tip-Adapter [5]</td><td>ECCV, 2022</td><td>62.83</td><td>64.84</td><td>66.50</td><td>68.64</td><td>70.02</td></tr><tr><td>Tip-Adapter with RFF [5, 6]</td><td>CVPR, 2025</td><td>62.77</td><td>64.58</td><td>66.29</td><td>68.15</td><td>69.60</td></tr><tr><td>APE [20]</td><td>ICCV, 2023</td><td>64.43</td><td>66.48</td><td>68.66</td><td>70.75</td><td>72.81</td></tr><tr><td>APE with RFF [6, 20]</td><td>CVPR, 2025</td><td>64.16</td><td>66.11</td><td>68.76</td><td>70.37</td><td>72.02</td></tr><tr><td>APE+SSP [7]</td><td>ACM MM, 2024</td><td>66.03</td><td>68.04</td><td>69.90</td><td>71.63</td><td>73.72</td></tr><tr><td>ProKeR (RBF) [6]</td><td>CVPR, 2025</td><td>64.01</td><td>67.31</td><td>70.42</td><td>73.85</td><td>76.75</td></tr><tr><td>ProKeR with RFF [6]</td><td>CVPR, 2025</td><td>64.01</td><td>67.12</td><td>70.28</td><td>73.61</td><td>76.44</td></tr><tr><td>LAMP</td><td>Ours</td><td>64.79</td><td>67.50</td><td>71.19</td><td>75.03</td><td>78.73</td></tr><tr><td>LAMP-F</td><td>Ours</td><td>65.91</td><td>68.83</td><td>72.65</td><td>75.50</td><td>79.10</td></tr></table>

Table 3: ViT-B/16 16-shot top-1 accuracy (%) for representative published methods. Boldface and underlining indicate the highest and second-highest scores in each row. We retain the numerical precision and mean accuracies reported in the source publications. Evaluation protocols are described in Section 4.1.
<table><tr><td>Dataset</td><td>Tip-Adapter-F [5]</td><td>PLOT++ [16]</td><td>CLIP-LoRA [19]</td><td>CAPEL [17]</td><td>LAMP</td><td>LAMP-F</td></tr><tr><td>ImageNet [25]</td><td>73.4</td><td>72.6</td><td>73.6</td><td>73.73</td><td>74.10</td><td>74.39</td></tr><tr><td>SUN397 [26]</td><td>76.0</td><td>76.0</td><td>76.1</td><td>77.11</td><td>76.35</td><td>76.36</td></tr><tr><td>FGVC Aircraft [27]</td><td>44.6</td><td>46.7</td><td>54.7</td><td>48.09</td><td>49.83</td><td>50.58</td></tr><tr><td>EuroSAT [28]</td><td>85.9</td><td>92.0</td><td>92.1</td><td>88.73</td><td>93.16</td><td>93.68</td></tr><tr><td>Stanford Cars [29]</td><td>82.3</td><td>84.6</td><td>86.3</td><td>84.40</td><td>84.66</td><td>84.85</td></tr><tr><td>Food101 [30]</td><td>86.8</td><td>87.1</td><td>84.2</td><td>87.81</td><td>87.85</td><td>87.95</td></tr><tr><td>Oxford-IIIT Pets [31]</td><td>92.6</td><td>93.6</td><td>92.4</td><td>93.57</td><td>93.38</td><td>93.38</td></tr><tr><td>Flowers102 [32]</td><td>96.2</td><td>97.6</td><td>98.0</td><td>97.97</td><td>98.85</td><td>98.90</td></tr><tr><td>Caltech101 [33]</td><td>95.7</td><td>96.0</td><td>96.4</td><td>96.55</td><td>96.36</td><td>96.48</td></tr><tr><td>DTD [34]</td><td>70.8</td><td>71.4</td><td>72.0</td><td>73.46</td><td>75.73</td><td>76.20</td></tr><tr><td>UCF101 [35]</td><td>83.9</td><td>85.3</td><td>86.7</td><td>85.51</td><td>85.96</td><td>86.50</td></tr><tr><td>Avg.</td><td>80.7</td><td>82.1</td><td>83.0</td><td>82.45</td><td>83.30</td><td>83.57</td></tr></table>

## 4.3. Paired gains over the global baseline

Our paired global baseline reproduces ProKeR’s residual predictor in Eq. (3), using the final descriptor from frozen CLIP. Table 4 compares this predictor with LAMP and LAMP-F across all eleven datasets. The three methods share support banks, prompts, and selected global kernel parameters, so the comparison measures the gains from regional integration and the subsequent prototype adaptation. At 16-shot, the global baseline reaches 82.30% with ViT-B/16 and 76.62% with RN50. LAMP raises these means to 83.30% and 78.73%, and LAMP-F to 83.57% and 79.10%, respectively.

Averaged equally across datasets and the five support budgets, LAMP improves accuracy by 1.14 percentage points with ViT and 1.59 with RN50; LAMP-F improves it by 1.96 and 2.54 points. Both methods improve mean accuracy at every support budget on both backbones. Section 4.4 examines the gains by dataset.

## 4.4. Dataset-wise performance gains

At four shots, integrating native-layer features improves mean accuracy across the eleven datasets by 1.446 points with ViT and 1.568 with RN50 (Table 5). The largest gains occur on EuroSAT: 5.860 and 6.695 points, respectively. Mean accuracy increases on all eleven datasets with ViT and on ten with RN50; the RN50 change on SUN397 is −0 032 points. Native-layer integration is particularly efective for satellite imagery and, with ViT, texture recognition.

Table 4: Benchmark accuracy on eleven datasets, including ImageNet. Accuracy is averaged equally across datasets for each support-sampling seed; entries report the mean ± sample SD across seeds 1–3. The final row also averages the five shot counts. Gains are paired diferences from our paired reproduction of the ProKeR global predictor, in percentage points.
<table><tr><td>K</td><td>Global control</td><td>LAMP</td><td>LAMP-F</td><td>∆ LAMP</td><td>∆ LAMP-F</td></tr><tr><td colspan="6">ViT-B/16</td></tr><tr><td>1</td><td> $7 1 . 1 7 \pm 0 . 4 7$ </td><td> $7 2 . 1 8 \pm 0 . 4 6$ </td><td> $7 3 . 2 8 \pm 0 . 4 0$ </td><td> $1 . 0 0 \pm 0 . 0 6$ </td><td> $2 . 1 1 \pm 0 . 2 0$ </td></tr><tr><td>2</td><td> $7 3 . 3 2 \pm 0 . 4 9$ </td><td> $7 4 . 2 7 \pm 0 . 4 3$ </td><td> $7 5 . 6 2 \pm 0 . 4 5$ </td><td> $0 . 9 6 \pm 0 . 1 0$ </td><td> $2 . 3 0 \pm 0 . 1 0$ </td></tr><tr><td>4</td><td> $7 6 . 2 6 \pm 0 . 1 2$ </td><td> $7 7 . 7 0 \pm 0 . 2 3 $ </td><td> $7 8 . 8 0 \pm 0 . 2 0$ </td><td> $1 . 4 5 \pm 0 . 2 6$ </td><td> $2 . 5 4 \pm 0 . 0 9$ </td></tr><tr><td>8</td><td> $7 9 . 3 4 \pm 0 . 0 9$ </td><td> $8 0 . 6 5 \pm 0 . 1 1$ </td><td> $8 0 . 9 4 \pm 0 . 1 6$ </td><td> $1 . 3 0 \pm 0 . 0 9$ </td><td> $1 . 5 9 \pm 0 . 1 9$ </td></tr><tr><td>16</td><td> $8 2 . 3 0 \pm 0 . 0 9$ </td><td> $8 3 . 3 0 \pm 0 . 1 5$ </td><td> $8 3 . 5 7 \pm 0 . 1 3$ </td><td> $0 . 9 9 \pm 0 . 0 8$ </td><td> $1 . 2 7 \pm 0 . 0 8$ </td></tr><tr><td>Mean</td><td> $7 6 . 4 8 \pm 0 . 0 9$ </td><td> $7 7 . 6 2 \pm 0 . 1 2$ </td><td> $7 8 . 4 4 \pm 0 . 1 2$ </td><td> $1 . 1 4 \pm 0 . 0 3$ </td><td> $1 . 9 6 \pm 0 . 0 3$ </td></tr><tr><td colspan="6">RN50</td></tr><tr><td>1</td><td> $6 3 . 6 9 \pm 0 . 2 3$ </td><td> $6 4 . 7 9 \pm 0 . 3 2$ </td><td> $6 5 . 9 1 \pm 0 . 1 5$ </td><td> $1 . 1 1 \pm 0 . 2 3$ </td><td> $2 . 2 2 \pm 0 . 3 5$ </td></tr><tr><td>2</td><td> $6 6 . 4 2 \pm 0 . 1 3$ </td><td> $6 7 . 5 0 \pm 0 . 4 7$ </td><td> $6 8 . 8 3 \pm 0 . 6 1$ </td><td> $1 . 0 7 \pm 0 . 3 5$ </td><td> $2 . 4 1 \pm 0 . 4 8$ </td></tr><tr><td>4</td><td> $6 9 . 6 2 \pm 0 . 4 6$ </td><td> $7 1 . 1 9 \pm 0 . 2 2$ </td><td> $7 2 . 6 5 \pm 0 . 0 5$ </td><td> $1 . 5 7 \pm 0 . 2 4$ </td><td> $3 . 0 3 \pm 0 . 4 1$ </td></tr><tr><td>8</td><td> $7 2 . 9 3 \pm 0 . 1 0$ </td><td> $7 5 . 0 3 \pm 0 . 0 7$ </td><td> $7 5 . 5 0 \pm 0 . 0 9$ </td><td> $2 . 0 9 \pm 0 . 1 0$ </td><td> $2 . 5 7 \pm 0 . 1 9$ </td></tr><tr><td>16</td><td> $7 6 . 6 2 \pm 0 . 0 5$ </td><td> $7 8 . 7 3 \pm 0 . 2 0$ </td><td> $7 9 . 1 0 \pm 0 . 1 8$ </td><td> $2 . 1 1 \pm 0 . 1 4$ </td><td> $2 . 4 8 \pm 0 . 1 2$ </td></tr><tr><td>Mean</td><td> $6 9 . 8 6 \pm 0 . 0 2$ </td><td> $7 1 . 4 5 \pm 0 . 1 2$ </td><td> $7 2 . 4 0 \pm 0 . 1 3$ </td><td> $1 . 5 9 \pm 0 . 1 4$ </td><td> $2 . 5 4 \pm 0 . 1 4$ </td></tr></table>

Table 5: Paired accuracy gains at 4 shots on all eleven benchmark datasets, in percentage points. LAMP minus global measures the gain from training-free multi-layer integration; LAMP-F minus LAMP measures the additional gain from prototype adaptation. Entries report the mean ± sample SD across three support-sampling seeds. For the mean row, dataset gains are first averaged within each seed.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">LAMP minus global</td><td colspan="2">LAMP-F minus LAMP</td></tr><tr><td>ViT-B/16</td><td>RN50</td><td>ViT-B/16</td><td>RN50</td></tr><tr><td>ImageNet</td><td> $+ 0 . 1 6 8 \pm 0 . 1 3 7$ </td><td> $+ 0 . 1 2 7 \pm 0 . 0 3 0$ </td><td> $+ 0 . 5 7 5 \pm 0 . 0 6 6$ </td><td> $+ 0 . 8 3 5 \pm 0 . 0 3 5$ </td></tr><tr><td>Caltech101</td><td> $+ 0 . 4 1 9 \pm 0 . 3 2 8$ </td><td> $+ 0 . 8 5 2 \pm 0 . 2 2 6$ </td><td> $+ 0 . 2 0 3 \pm 0 . 3 6 1$ </td><td> $+ 1 . 0 4 1 \pm 0 . 3 6 6$ </td></tr><tr><td>Pets</td><td> $+ 0 . 1 9 1 \pm 0 . 3 3 0$ </td><td> $+ 0 . 1 4 5 \pm 0 . 1 8 6$ </td><td> $+ 0 . 3 7 2 \pm 0 . 5 7 6$ </td><td> $+ 0 . 0 7 3 \pm 0 . 1 2 6$ </td></tr><tr><td>Cars</td><td> $+ 0 . 8 5 8 \pm 0 . 4 0 2$ </td><td> $+ 1 . 7 2 9 \pm 1 . 2 9 6$ </td><td> $+ 1 . 2 0 6 \pm 0 . 9 1 4$ </td><td> $+ 1 . 1 9 0 \pm 0 . 5 4 9$ </td></tr><tr><td>Flowers102</td><td> $+ 1 . 1 6 4 \pm 0 . 3 9 4$ </td><td> $+ 2 . 3 2 8 \pm 0 . 1 3 1$ </td><td> $+ 0 . 3 9 2 \pm 0 . 6 8 0$ </td><td> $+ 0 . 7 5 8 \pm 0 . 5 8 7$ </td></tr><tr><td>Food101</td><td> $+ 0 . 2 3 2 \pm 0 . 0 7 4$ </td><td> $+ 0 . 0 3 2 \pm 0 . 0 4 6$ </td><td> $+ 0 . 2 7 0 \pm 0 . 0 1 0$ </td><td> $+ 0 . 9 0 1 \pm 0 . 0 5 3$ </td></tr><tr><td>Aircraft</td><td> $+ 0 . 1 6 0 \pm 0 . 5 6 8$ </td><td> $+ 2 . 3 8 0 \pm 1 . 2 9 1$ </td><td> $+ 0 . 9 1 0 \pm 0 . 2 7 2$ </td><td> $+ 1 . 3 3 0 \pm 0 . 1 9 1$ </td></tr><tr><td>SUN397</td><td> $+ 1 . 5 4 7 \pm 2 . 8 9 3$ </td><td> $- 0 . 0 3 2 \pm 0 . 0 2 3$ </td><td> $+ 5 . 6 2 0 \pm 4 . 5 9 8$ </td><td> $+ 6 . 7 1 5 \pm 0 . 3 7 6$ </td></tr><tr><td>DTD</td><td> $+ 3 . 8 6 1 \pm 0 . 0 3 4$ </td><td> $+ 1 . 4 1 8 \pm 0 . 3 6 9$ </td><td> $+ 0 . 3 5 5 \pm 0 . 6 8 2$ </td><td> $+ 0 . 9 8 5 \pm 1 . 7 3 1$ </td></tr><tr><td>EuroSAT</td><td> $+ 5 . 8 6 0 \pm 0 . 7 6 6$ </td><td> $+ 6 . 6 9 5 \pm 1 . 7 9 6$ </td><td> $+ 1 . 4 6 9 \pm 0 . 4 6 0$ </td><td> $+ 0 . 9 9 6 \pm 0 . 4 7 6$ </td></tr><tr><td>UCF101</td><td> $+ 1 . 4 4 5 \pm 0 . 2 9 1$ </td><td> $+ 1 . 5 6 8 \pm 0 . 9 1 6$ </td><td> $+ 0 . 6 9 6 \pm 0 . 3 1 3$ </td><td> $+ 1 . 2 6 9 \pm 0 . 8 2 8$ </td></tr><tr><td>Mean</td><td> $+ 1 . 4 4 6 \pm 0 . 2 5 5$ </td><td> $+ 1 . 5 6 8 \pm 0 . 2 3 7$ </td><td> $+ 1 . 0 9 7 \pm 0 . 2 8 9$ </td><td> $+ 1 . 4 6 3 \pm 0 . 1 7 6$ </td></tr></table>

Prototype adaptation benefits a diferent set of tasks. Its largest 4-shot gains occur on SUN397, at 5.620 points with ViT and 6.715 with RN50. Prototype adaptation and native-layer integration thus benefit diferent tasks: the former adjusts class directions, while the latter changes how support evidence is used to correct the initial CLIP scores.

Table 6: Representation controls at 4-shot with 113 mixture candidates per family. Top: mean gains in percentage points over ten non-ImageNet datasets. Bottom: mean accuracy (%) over Pets, Food101, DTD, and EuroSAT, averaged equally across datasets and three support seeds. MULTIBAND uses the final CLIP global descriptor; FINAL uses the deepest native feature map.
<table><tr><td>Comparison or representation</td><td>ViT-B/16</td><td>RN50</td></tr><tr><td>Native-layer versus global information (gain) REGION – MULTIBAND</td><td>+1.286</td><td>+1.635</td></tr><tr><td>Depth and pooling (accuracy)</td><td></td><td></td></tr><tr><td>REGION: four descriptors per depth</td><td>82.993</td><td>77.060</td></tr><tr><td>CENTROID: mean of regional descriptors</td><td>82.991</td><td>77.067</td></tr><tr><td>WHOLE: one descriptor per depth</td><td>83.021</td><td>76.988</td></tr><tr><td>FINAL: deepest native map only</td><td>81.828</td><td>76.083</td></tr></table>

## 4.5. Layer depth, pooling, and calibration

We examine the source of LAMP’s gains through three groups of controls: the information used by the kernel branches, the representation retained at each depth, and the calibration and weighting of their corrections. All comparisons use paired support banks at 4-shot.

Information from diferent depths. MULTIBAND constructs three kernels from the same final CLIP global descriptor, using bandwidths $0 . 5 \beta _ { g } , \beta _ { g }$ , and $2 \beta _ { g }$ . Each calibrated branch fits the shared support residual independently, and their corrections are mixed with the original global predictor. This control tests the benefit of changing the similarity scale while keeping the visual representation fixed. Across the ten non-ImageNet datasets, native-region kernels outperform MULTIBAND by 1.286 points with ViT and 1.635 with RN50 under matched 113-candidate mixture searches (Table 6, top). The family-adjusted 98.75% intervals are positive for both backbones, supporting the value of information from multiple encoder depths.

Depth and spatial pooling. We compare four representations on Pets, Food101, DTD, and EuroSAT. REGION retains four regional descriptors at each of three depths. CEN-TROID averages the four normalized descriptors at each depth and normalizes the result; WHOLE pools each feature map directly into one descriptor. FINAL uses only the deepest native feature map, with three bandwidths. Thus WHOLE versus FINAL tests the use of multiple depths, while REGION versus CENTROID or WHOLE tests the efect of retaining separate regional descriptors. All four families evaluate 113 mixtures; REGION, CENTROID, and WHOLE use fixed regional bandwidth ratio r = 1.

WHOLE exceeds FINAL by 1.193 [0.648, 1.806] points with ViT and 0.905 [0.499, 1.384] with RN50, with pointwise 95% intervals in brackets. Combining depths therefore improves accuracy even when each map is pooled into a single descriptor. REGION and WHOLE difer by −0 028 [−0 162, 0.129] and 0.072 [−0 083, 0.220] points, respectively. The similar mean accuracies of REGION, CENTROID, and WHOLE indicate that compact descriptors retain much of the observed gain from native-layer information.

Calibration and layer weighting. We next restore bandwidth selection and simplify the full LAMP model on the same four datasets. S1 replaces regional sets with whole-map descriptors; S2 additionally removes support centering and diagonal normalization; S3 further fixes the three layer weights to 1/3. LAMP, S1, and S2 each search 337 configurations, while S3 searches 22. S1 thus uses the WHOLE representation with a larger bandwidth search.

The clearest calibration efect occurs with RN50: removing both operations reduces mean accuracy by 0.627 points, with an adjusted 99.1667% interval of [−0 979, −0 277] for S2 minus S1. On EuroSAT, accuracy falls from 82.802% to 80.811%. The ViT change is +0 215 [−0 075, +0 514], showing a less consistent benefit from calibration on this backbone. Replacing regional sets with whole-map descriptors also has taskdependent efects: ViT accuracy on DTD falls from 66.923% to 66.056%, while Pets and Food101 improve slightly.

Equal layer weights reduce the means by a further 0.331 points with ViT and 0.180 with RN50. Their adjusted intervals include zero; this comparison assesses the complete selected-weight and equal-weight strategies, including their search budgets. Even after all three simplifications, S3 exceeds the paired global baseline by 2.179 and 1.117 points. These results locate the main gain in native-layer information and the strongest calibration benefit in RN50.

## 4.6. Performance gains ofLAMP-F over LAMP

This experiment isolates prototype adaptation by retaining LAMP’s selected kernels and mixture weights. Across eleven datasets and five support budgets, LAMP-F improves the three-seed mean in 53 of 55 settings with ViT and 52 of 55 with RN50. With ViT, there is one tie (Pets, 16-shot) and one decrease (Caltech101, 8-shot; −0 068 points); the three RN50 ties all occur on Pets, at 1, 8, and 16 shots. Mean gains decrease from 1.097 and 1.463 points at 4-shot to 0.274 and 0.366 at 16-shot for ViT and RN50, respectively. SUN397 benefits most, with gains concentrated at 1–4 shots.

Comparison with an independently adapted global predictor. We also train a separate ProKeR+CLAP model [4, 6] on the ten non-ImageNet datasets at 4-shot. It shares the support data, prompts, validation labels, and checkpoint-search budget with LAMP-F. Both methods evaluate 33 learning-rate/checkpoint candidates; LAMP-F additionally inherits the 337-candidate regional-mixture selection. LAMP-F improves mean accuracy by 1.840 points with ViT and 1.949 with RN50; the family-adjusted intervals are [1.451, 2.283] and [1.483, 2.417], respectively. This comparison assesses the complete LAMP-F adaptation procedure against a global predictor with independently optimized prototypes.

## 4.7. Support-draw robustness and validation-label budget

To assess the annotation requirements of model selection, we sample ten new 4-shot support sets on DTD and EuroSAT. For each backbone and support set, we repeat the complete parameter search using one validation image per class, four images per class, or the full validation split. The nested validation subsets are fixed across support draws and backbones. Confidence intervals for this study use paired Student-t statistics over the ten support draws, conditional on the fixed validation subset and test set. Figure 2 summarizes all 120 configurations. With the full validation split, LAMP improves on the global baseline by 2.494 and 2.252 points on DTD with ViT and RN50, respectively; the corresponding gains on EuroSAT are 5.253 and 6.738 points. All four adjusted confidence intervals lie above zero, showing that the gains persist across additional support draws.

![](images/f84d03d36775b335f55722c509676bd2229c8941be80f7ba125c5d36f77fd944.jpg)  
Figure 2: Efect of the validation-label budget at four shots on DTD and EuroSAT. Points show mean test accuracy over ten new support draws, and error bars indicate one sample SD. The full validation split contains 24 images per class on DTD and 540 on EuroSAT. Methods and support draws use the same nested validation subsets; the complete parameter search is repeated for each setting.

The validation budget strongly afects model selection. With one validation image per class on EuroSAT, the procedure selects the global-only candidate in 19 of 20 combinations of backbone and support set. Increasing the budget to four images per class yields mean gains of 3.267 and 4.857 points with ViT and RN50, respectively. Labeled validation data therefore play an important role in selecting efective nativelayer corrections.

## 4.8. Additional backbones and frozen-target transfer

To examine adaptation across encoder architectures, depths, and patch sizes, we compare training-free methods on ImageNet at 16 shots using four CLIP backbones (Table 7). LAMP improves over the published ProKeR scores by 1.28, 1.13, 1.00, and 0.85 percentage points for RN50, RN101, ViT-B/32, and ViT-B/16, respectively. LAMP achieves the highest accuracy among the methods listed in this table on each backbone.

Paired comparisons assess the contribution of native-layer corrections with RN101 and ViT-B/32. With RN101, LAMP improves mean accuracy across the ten non-

Table 7: Training-free ImageNet 16-shot accuracy (%) across four CLIP backbones. External scores come from ProKeR Table 6 [6] and IDEA Table 2 [22]. LAMP reports means over seeds 1–3. Bold marks the highest accuracy among the listed methods in each column; gains over ProKeR are in percentage points.
<table><tr><td>Models</td><td>ResNet-50</td><td>ResNet-101</td><td>ViT-B/32</td><td>ViT-B/16</td></tr><tr><td>Zero-shot CLIP</td><td>60.33</td><td>62.53</td><td>63.80</td><td>68.73</td></tr><tr><td>Tip-Adapter</td><td>61.43</td><td>64.08</td><td>65.18</td><td>70.25</td></tr><tr><td>APE</td><td>62.60</td><td>65.61</td><td>66.31</td><td>71.37</td></tr><tr><td>IDEA</td><td>62.58</td><td>65.51</td><td>65.93</td><td>71.07</td></tr><tr><td>ProKeR</td><td>64.45</td><td>67.39</td><td>68.12</td><td>73.25</td></tr><tr><td>LAMP (Ours)</td><td>65.73</td><td>68.52</td><td>69.12</td><td>74.10</td></tr><tr><td>Gain over ProKeR (pp)</td><td>+1.28</td><td>+1.13</td><td>+1.00</td><td>+0.85</td></tr></table>

Table 8: Training-free RN101 accuracy averaged equally over the ten non-ImageNet datasets. LAMP and the global-feature-only baseline use the same support banks. Accuracy is in percent; gains and pointwise 95% paired-bootstrap intervals are in percentage points.
<table><tr><td>Shot</td><td>Global-only</td><td>LAMP</td><td>Gain</td><td>95% interval</td></tr><tr><td>1</td><td>65.750</td><td>66.946</td><td>+1.196</td><td>[+0.873, +1.523]</td></tr><tr><td>2</td><td>68.491</td><td>70.030</td><td>+1.539</td><td>[+1.240, +1.832]</td></tr><tr><td>4</td><td>72.333</td><td>74.146</td><td>+1.814</td><td>[+1.536, +2.104]</td></tr><tr><td>8</td><td>75.873</td><td>78.206</td><td>+2.333</td><td>[+1.989, +2.685]</td></tr><tr><td>16</td><td>79.790</td><td>81.927</td><td>+2.137</td><td>[+1.893, +2.384]</td></tr></table>

ImageNet datasets at every support budget by 1.196–2.333 points; all pointwise 95% confidence intervals lie above zero (Table 8). Across all eleven datasets and five support budgets, five of the 55 changes are negative, including Pets at 2-shot (−0 645 [−1 663, −0 018]). On ImageNet at 16-shot, LAMP achieves 68.523% with RN101 and 69.125% with ViT-B/32, with the latter exceeding its paired baseline by 0.465 points. These results demonstrate the use of the same training-free residual formulation with diferent encoder depths and patch sizes.

To evaluate robustness to distribution shift, we apply the ViT-B/16 and RN50 models fitted on ImageNet at 16-shot to ImageNet-V2 and ImageNet-Sketch [10, 11]. All fitted components remain fixed, with no further fitting or parameter selection on either target dataset.

LAMP improves accuracy in all twelve combinations of seed, target dataset, and backbone. The pointwise 95% confidence intervals for all four aggregate gains in Table 9 lie above zero, showing that native-layer corrections remain useful under both distribution shifts. With ViT, LAMP achieves 66.297% on ImageNet-V2 and 49.910% on ImageNet-Sketch, compared with 74.105% on the source ImageNet benchmark.

Table 9: Transfer of frozen 16-shot models: mean top-1 accuracy (%) over three seeds and LAMP’s paired gain over the global-feature-only baseline. Gains are in percentage points, with pointwise 95% confidence intervals from 5,000 paired bootstrap draws.
<table><tr><td>Backbone</td><td>Target</td><td>Global-only</td><td>LAMP</td><td>Gain [95% interval]</td></tr><tr><td>ViT-B/16</td><td>ImageNet-V2</td><td>65.843</td><td>66.297</td><td>+0.453 [+0.110, +0.837]</td></tr><tr><td>ViT-B/16</td><td>ImageNet-Sketch</td><td>49.258</td><td>49.910</td><td>+0.652 [+0.363, +0.914]</td></tr><tr><td>RN50</td><td>ImageNet-V2</td><td>56.220</td><td>57.077</td><td>+0.857 [+0.537, +1.177]</td></tr><tr><td>RN50</td><td>ImageNet-Sketch</td><td>35.771</td><td>36.631</td><td>+0.860 [+0.689, +1.059]</td></tr></table>

![](images/a25903dac6b41c2fa1aad025b403a3dc6975e2e97a2861aacfba9a8c3fa67635.jpg)  
Figure 3: Regional parameter profiles at 4-shot, averaged equally over eleven datasets and three support seeds. Each point is the validation gain over the global baseline after reselecting the other regional parameters. Left: mixture strength $\rho$ in $\operatorname { E q . }$ (10), with an inset showing the small $- \rho$ values. Right: bandwidth ratio $r = \beta _ { f } / \beta _ { g }$ for the regional kernel in Eq. (6).

## 4.9. Hyperparameter behavior and layer contributions

We examine how the parameters in the residual mixture afect validation performance. The mixture strength $\rho$ and layer weights w appear in Eq. (10); the ratio $r = \beta _ { f } / \beta _ { g }$ sets the regional bandwidth in Eq. (6), using the kernel in Eq. (2). Figure 3 summarizes 4-shot results over eleven datasets and three support-sampling seeds. At each plotted value, the remaining regional parameters are reselected for each support bank.

Global–regional balance. The mean validation profile peaks at $\rho = 0 . 5$ for ViT-B/16 and $\rho = 0 . 7 5$ for RN50, with gains of 1.437 and 1.678 points over the global baseline. $\mathbf { A t } \rho = 1$ , the gains are 0.470 and 1.394 points. Combining the global and native-layer corrections therefore gives the highest mean validation accuracy among the tested mixture strengths.

Regional bandwidth. For r ∈ {0 5 1 2}, the mean validation gains range from 1.575 to 1.604 points with ViT and from 1.746 to 1.754 with RN50. Selecting the mixture separately at each ratio yields similar mean performance across these three similarity scales.

![](images/9d018d6e81cbf8016189dcad145e5a9099f543242fb79fd0a4c7e82ede6e5a6d.jpg)  
Figure 4: Efective weights of the global and regional corrections at 4 shots, averaged over seeds 1–3. The four coeficients in Eq. (10) are $( 1 - \rho , \rho w _ { 1 } , \rho w _ { 2 } , \rho w _ { 3 } )$ ; their weighted corrections are added to the shared CLIP prior. Both panels use the same 0–1 scale.

Contributions of diferent depths. Figure 4 displays the four coeficients in Eq. (10): $( 1 - \rho , \rho w _ { 1 } , \rho w _ { 2 } , \rho w _ { 3 } )$ ). The factor ${ \bf \Phi } _ { \mathcal { P } }$ determines the total regional contribution, and w distributes it among the three depths. At 4-shot, the selected mixtures favor the global branch on Pets and assign a larger regional contribution on DTD and EuroSAT than on Pets. With ViT, Flowers102 places its largest regional weight on the late layer, while DTD favors the middle layer. These patterns show how the selected mixture varies with the task. Section 4.5 measures the corresponding accuracy efects of combining depths.

## 4.10. Computational cost

Table 10 measures image-to-label inference on one RTX 4090 at 16 shots. On ImageNet with ViT-B/16, LAMP increases single-image latency from 14.22 to 64.83 ms, while batch-128 throughput changes from 113.38 to 107.89 images/s. With RN50, the corresponding values are 14.81 to 65.86 ms and 118.64 to 112.28 images/s. Regional matching therefore has a larger relative efect on single-image latency than on batched throughput. Peak allocated GPU memory at batch 128 increases from 2.71 to 7.64 GiB with ViT and from 3.02 to 11.55 GiB with RN50. LAMP-F adds the prototype response storage and reaches 8.56 and 13.39 GiB, respectively.

Table 10: Deployment cost at 16 shots, measured on one RTX 4090 with seed 1. Latency is the median time from image decoding to classification (ms) at batch size one. Throughput (images/s) and peak allocated GPU memory (GiB) are measured at batch size 128, separately for each method. Timing includes decoding, preprocessing, encoding, and classification; disk retrieval is excluded.
<table><tr><td>Dataset</td><td>Backbone</td><td>Method</td><td>Latency</td><td>Throughput</td><td>GPU peak</td></tr><tr><td>DTD</td><td>ViT-B/16</td><td>Global-only</td><td>12.52</td><td>128.71</td><td>1.82</td></tr><tr><td></td><td></td><td>LAMP</td><td>42.38</td><td>124.15</td><td>2.23</td></tr><tr><td></td><td></td><td>LAMP-F</td><td>42.70</td><td>127.02</td><td>2.29</td></tr><tr><td></td><td>RN50</td><td>Global-only</td><td>12.67</td><td>138.47</td><td>1.84</td></tr><tr><td></td><td></td><td>LAMP</td><td>23.40</td><td>137.79</td><td>2.02</td></tr><tr><td></td><td></td><td>LAMP-F</td><td>23.64</td><td>137.43</td><td>2.11</td></tr><tr><td>ImageNet</td><td>ViT-B/16</td><td>Global-only</td><td>14.22</td><td>113.38</td><td>2.71</td></tr><tr><td></td><td></td><td>LAMP</td><td>64.83</td><td>107.89</td><td>7.64</td></tr><tr><td></td><td></td><td>LAMP-F</td><td>65.68</td><td>108.05</td><td>8.56</td></tr><tr><td></td><td>RN50</td><td>Global-only</td><td>14.81</td><td>118.64</td><td>3.02</td></tr><tr><td></td><td></td><td>LAMP</td><td>65.86</td><td>112.28</td><td>11.55</td></tr><tr><td></td><td></td><td>LAMP-F</td><td>68.04</td><td>109.82</td><td>13.39</td></tr></table>

Compact regional representations ofer a storage–accuracy trade-of. On the DTD 4-shot diagnostic bank, CENTROID matches REGION’s accuracy on both backbones using one quarter of the native descriptor storage.

## 5. Discussion

## 5.1. Implications for training-free adaptation

LAMP improves adaptation by combining complementary features from diferent encoder layers. The final global descriptor provides image–text alignment, while intermediate feature maps supply additional visual information for the target task. Controls with matched search budgets show that descriptors from multiple layers outperform an ensemble built from the final global feature. Likewise, pooling several feature maps outperforms pooling only the deepest map. In the pooling study, regional centroids and whole-map descriptors retain most of the accuracy gain. These compact representations ofer useful options for applications with limited storage or strict latency requirements.

Calibration and task-specific mixing determine how each layer contributes to prediction. Calibration is particularly efective for RN50 on EuroSAT, and the selected layer weights vary across datasets and backbones. LAMP-F adds prototype adaptation, with the largest gains when few support images are available. Caching the response to prototype changes makes their evaluation inexpensive relative to constructing the regional predictors. The results highlight the value of the visual information already present in a frozen encoder and show how to use it through complementary residual corrections.

## 5.2. Limitations

Using features from multiple layers increases storage requirements and kernel computation costs. On ImageNet at 16 shots, LAMP increases latency at batch size one from 14.22 to 64.83 ms on ViT-B/16 and from 14.81 to 65.86 ms on RN50 (Table 10). Dense fitting requires $O ( N ^ { 2 } )$ storage and $O ( N ^ { 3 } )$ factorization per branch; LAMP-F also stores prototype responses.

## 6. Conclusion

We presented LAMP, a training-free method that combines global and regional predictions from multiple layers of a frozen vision–language encoder. Each branch retains its native feature dimension and independently fits the same support-label residual; their corrections are then combined in class-score space. LAMP achieves state-ofthe-art mean accuracy among the compared training-free RN50 methods at 4, 8, and 16 shots, while matching or exceeding representative methods that require training at higher support budgets. LAMP-F adds text-anchored prototype adaptation and uses a cached response to update the predictions exactly. Beyond these accuracy gains, LAMP provides a way to combine features of diferent dimensions without learning a shared projection. Its separation of branch fitting, score fusion, and prototype updates ofers a foundation for studying task-specific layer selection and developing more scalable residual prediction methods.

## Acknowledgements

This work was supported by the Young Scientific and Technological Talent Support Program under the Taizhou Fengcheng Talent Plan. The funder provided financial support only and had no role in the study design, conduct of the experiments, data analysis, manuscript preparation, or decision to submit the article for publication.

## Declaration of generative AI and AI-assisted technologies

During manuscript preparation, the authors used OpenAI ChatGPT/Codex to assist with language editing, document organization, experimental-code preparation, and consistency checks against the recorded experiments. The authors reviewed the material and take responsibility for the article.

## References

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever, Learning transferable visual models from natural language supervision, in: Proceedings of the 38th International Conference on Machine Learning, Vol. 139 of Proceedings of Machine Learning Research, 2021, pp. 8748–8763.

[2] K. Zhou, J. Yang, C. C. Loy, Z. Liu, Learning to prompt for vision-language models, International Journal of Computer Vision 130 (9) (2022) 2337–2348. doi:10.1007/s11263-022-01653-1.

[3] T. Yu, Z. Lu, X. Jin, Z. Chen, X. Wang, Task Residual for Tuning Vision-Language Models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10899–10909.

[4] J. Silva-Rodríguez, S. Hajimiri, I. Ben Ayed, J. Dolz, A closer look at the few-shot adaptation of large vision-language models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 23681–23690.

[5] R. Zhang, W. Zhang, R. Fang, P. Gao, K. Li, J. Dai, Y. Qiao, H. Li, Tip-Adapter: Training-free adaption of CLIP for few-shot classification, in: Computer Vision – ECCV 2022, 2022, pp. 493–510. doi:10.1007/978-3-031-19833-5\_29.

[6] Y. Bendou, A. Ouasfi, V. Gripon, A. Boukhayma, ProKeR: A kernel perspective on few-shot adaptation of large vision-language models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 25092–25102.

[7] X. Zhu, B. Zhu, Y. Tan, S. Wang, Y. Hao, H. Zhang, Selective vision-language subspace projection for few-shot CLIP, in: Proceedings of the ACM International Conference on Multimedia, 2024. doi:10.1145/3664647.3680885.

[8] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, An image is worth 16x16 words: Transformers for image recognition at scale, in: International Conference on Learning Representations, 2021.

[9] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778. doi:10.1109/CVPR.2016.90.

[10] B. Recht, R. Roelofs, L. Schmidt, V. Shankar, Do ImageNet classifiers generalize to ImageNet?, in: Proceedings of the 36th International Conference on Machine Learning, Vol. 97 of Proceedings of Machine Learning Research, 2019, pp. 5389– 5400.

[11] H. Wang, S. Ge, Z. Lipton, E. P. Xing, Learning robust global representations by penalizing local predictive power, in: Advances in Neural Information Processing Systems, Vol. 32, 2019.

[12] K. Zhou, J. Yang, C. C. Loy, Z. Liu, Conditional Prompt Learning for Vision-Language Models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16816–16825.

[13] H. Yao, R. Zhang, C. Xu, Visual-Language Prompt Tuning With Knowledge-Guided Context Optimization, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6757–6767.

[14] B. Zhu, Y. Niu, Y. Han, Y. Wu, H. Zhang, Prompt-aligned Gradient for Prompt Tuning, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15659–15669.

[15] M. U. Khattak, H. Rasheed, M. Maaz, S. Khan, F. S. Khan, MaPLe: Multi-Modal Prompt Learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 19113–19122.

[16] G. Chen, W. Yao, X. Song, X. Li, Y. Rao, K. Zhang, PLOT: Prompt Learning with Optimal Transport for Vision-Language Models, in: International Conference on Learning Representations, 2023.

[17] Z. Chen, X. Yu, X. Tao, Y. Li, Z. Huang, Cluster-aware prompt ensemble learning for few-shot vision-language model adaptation, Pattern Recognition 172 (2026) 112596. doi:10.1016/j.patcog.2025.112596.

[18] P. Gao, S. Geng, R. Zhang, T. Ma, R. Fang, Y. Zhang, H. Li, Y. Qiao, CLIP-Adapter: Better vision-language models with feature adapters, International Journal of Computer Vision 132 (2) (2024) 581–595. doi:10.1007/s11263-023-01891-x.

[19] M. Zanella, I. Ben Ayed, Low-Rank Few-Shot Adaptation of Vision-Language Models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2024, pp. 1593–1603.

[20] X. Zhu, R. Zhang, B. He, A. Zhou, D. Wang, B. Zhao, P. Gao, Not All Features Matter: Enhancing Few-shot CLIP with Adaptive Prior Refinement, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 2605–2615.

[21] Z. Wang, J. Liang, L. Sheng, R. He, Z. Wang, T. Tan, A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation, in: International Conference on Learning Representations, 2024.

[22] Z. Ye, F. Jiang, Q. Wang, K. Huang, J. Huang, IDEA: Image description enhanced CLIP-adapter for image classification, Pattern Recognition 171 (2026) 112224. doi:10.1016/j.patcog.2025.112224.

[23] F. Radenovic, G. Tolias, O. Chum, Fine-tuning CNN image retrieval with no human´ annotation, IEEE Transactions on Pattern Analysis and Machine Intelligence 41 (7) (2019) 1655–1668. doi:10.1109/TPAMI.2018.2846566.

[24] K. Muandet, K. Fukumizu, B. K. Sriperumbudur, B. Schölkopf, Kernel mean embedding of distributions: A review and beyond, Foundations and Trends in Machine Learning 10 (1–2) (2017) 1–141. doi:10.1561/2200000060.

[25] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei, ImageNet: A large-scale hierarchical image database, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255. doi:10.1109/cvpr.2009. 5206848.

[26] J. Xiao, J. Hays, K. A. Ehinger, A. Oliva, A. Torralba, SUN database: Large-scale scene recognition from abbey to zoo, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2010, pp. 3485–3492. doi:10.1109/ cvpr.2010.5539970.

[27] S. Maji, E. Rahtu, J. Kannala, M. Blaschko, A. Vedaldi, Fine-grained visual classification of aircraft, arXiv preprint arXiv:1306.5151 (2013).

[28] P. Helber, B. Bischke, A. Dengel, D. Borth, EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 12 (7) (2019) 2217–2226. doi:10.1109/jstars.2019.2918242.

[29] J. Krause, M. Stark, J. Deng, L. Fei-Fei, 3D object representations for fine-grained categorization, in: IEEE International Conference on Computer Vision Workshops, Workshop on 3D Representation and Recognition, 2013, pp. 554–561. doi: 10.1109/iccvw.2013.77.

[30] L. Bossard, M. Guillaumin, L. V. Gool, Food-101—mining discriminative components with random forests, in: European Conference on Computer Vision, Vol. 8694 of Lecture Notes in Computer Science, 2014, pp. 446–461. doi:10.1007/978-3-319-10599-4\_29.

[31] O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, Cats and dogs, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2012, pp. 3498–3505. doi:10.1109/cvpr.2012.6248092.

[32] M.-E. Nilsback, A. Zisserman, Automated flower classification over a large number of classes, in: Indian Conference on Computer Vision, Graphics and Image Processing, 2008, pp. 722–729. doi:10.1109/icvgip.2008.47.

[33] L. Fei-Fei, R. Fergus, P. Perona, Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories, in: IEEE Conference on Computer Vision and Pattern Recognition Workshops, Workshop on Generative-Model Based Vision, 2004.

[34] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, A. Vedaldi, Describing textures in the wild, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 3606–3613. doi:10.1109/cvpr.2014.461.

[35] K. Soomro, A. R. Zamir, M. Shah, UCF101: A dataset of 101 human actions classes from videos in the wild, arXiv preprint arXiv:1212.0402 (2012).