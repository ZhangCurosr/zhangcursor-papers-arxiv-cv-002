# LiSA: Linear Stochastic Attention for Long-Range Visual Modeling

Sahar Dastani<sup>1,2</sup>, Fereshteh Shakeri<sup>1</sup>, Samuel Barbeau<sup>1</sup>, Kunal Mahatha<sup>1</sup>, Ismail Ben Ayed<sup>1</sup>, Herve Lombaert<sup>2,3</sup>, Christian Desrosiers<sup>1</sup>

<sup>1</sup>LIVIA, ETS Montreal, Canada <sup>´ 2</sup>Mila – Quebec AI Institute, Canada <sup>3</sup>Polytechnique Montreal, Canada

## Abstract

Self-attention has become the standard for modern sequence modeling, enabling Transformers to model longrange dependencies across a wide range ofmodalities. Despite these advances, its quadratic cost with respect to sequence length remains a significant computational bottleneck. Recent work addresses this limitation by introducing linear variants ofself-attention, namely with kernel approximations ofthe softmax operator. However, theirfirst-order aggregation restricts expressivity and struggles to capture higher-order structure. We propose Linear Stochastic Attention (LiSA), a linear attention plug-in module that enables higher-order token interactions via a stochastic diffusion process based on truncated random walks. LiSA incorporates adaptive locality, favoring semantically similar neighbors to preserve fine-grained structures, while maintaining stable and efficient training. We conduct extensive experiments on classification, semantic segmentation, and object detection tasks and show improved results over stateof-the-art methods. The code is available here.

## 1. Introduction

Transformers [34] have become a central paradigm in visual representation learning, with self-attention [24] serving as a key mechanism for capturing long-range dependencies. Since the advent of Vision Transformers (ViTs) [10, 33], self-attention has delivered strong performance across image classification [33, 38], object detection [3, 22], segmentation [4, 43], and multi-modal tasks [30], highlighting the value of global context in modern vision pipelines. However, self-attention scales quadratically with token length, which limits applicability to high-resolution images.

A number of recent works seek to reduce the quadratic complexity of standard ViTs. One common approach restricts attention to a limited set of tokens, either through fixed-sized attention windows [25, 32], introducing sparsity [38, 41] or representing the attention queries and keys with low rank matrices [37]. By reducing the number of pairwise token interactions, these methods lower computational cost, but often at the expense of global context modeling.

![](images/5e21cd73d39853b96c99cfb8f71eefa9e03ef85aeabc575878df8ad828e03811.jpg)  
Figure 1. Softmax vs. Linear vs. Higher-order attention. Given $Q , K , V \in \mathbb { R } ^ { n \times d } ,$ : softmax-attention forms $Q K ^ { \top }$ then applies to V . Linear-attention computes $K ^ { \top } V$ then applies to Q. Higherorder attention (ours) uses a truncation length T, retaining multiple diffusion terms while remaining linear in n and expanding the effective receptive field.

Other approaches linearize self-attention using kernelbased approximations [1, 20, 29, 31], replacing the softmax with a positive feature map ϕ(·). This enables an associative reordering of the Q, K, and V computation that avoids explicit pairwise interactions and reduces the complexity to linear in the sequence length. These methods retain a global receptive field without relying on downsampling.

In practice, however, an accuracy gap between kernelized linear-attention and standard self-attention persists. Recent works have reduced this gap by linearizing dot-product attention [2], enriching the kernel with angular [45] or focused parameterizations, and adding depthwise-convolutional components [14] or auxiliary softmax branches [45]. However, these variants are almost always built on the same core backbone, vanilla kernelized linear-attention [20]. Our analysis points to a structural bottleneck in this baseline: its single-pass bilinear aggregation.

Vanilla linear-attention mixes values for each query once through direct query-key affinities. With the affinity matrix defined as $A = \phi ( Q ) \phi ( K ) ^ { \top }$ , the output $Y = A V$ can be interpreted as a first-order message-passing step along the edges encoded by A. Crucially, indirect dependencies (i.e., using intermediate tokens) would appear only in higher powers $( \mathrm { e . g . , } A ^ { 2 } V , A ^ { 3 } V )$ , but vanilla linear-attention never materializes higher-order terms. As a result, when two regions are semantically related yet have weak first-order similarity (e.g., appearance changes due to lighting, occlusion, or viewpoint), their connection is under-weighted and the signal is diluted by many competing matches. In contrast, softmax-attention introduces competitive normalization that can concentrate probability mass on high-scoring connections, which help preserve long-range links.

Our approach targets this limitation and proposes a novel plug-in Linear Stochastic Attention (LiSA) module, which achieves efficient higher-order information propagation without sacrificing stability or locality. LiSA models attention as a linear stochastic process, where image patches are represented as graph nodes and edges encode transition probabilities among them during a truncated random walk. This formulation allows information to traverse intermediate tokens, enabling transitive reasoning. The random walk is truncated to a small number of steps to maintain linear complexity and numerical stability, providing a controllable balance between expressiveness and efficiency.

Overall, LiSA serves as a stronger drop-in module: it integrates cleanly with standard linear-attention refinements and often amplifies their benefits, delivering higher accuracy than vanilla linear-attention while preserving lineartime complexity and training stability. Our contributions are summarized as follows:

• We introduce LiSA, a plug-in linear-attention module that enables efficient higher-order token interactions.

• We formulate attention as a truncated random walk, enabling transitive information flow while preserving lineartime complexity and stable training.

• We evaluate our method across classification, segmentation, and detection, consistently outperforming both selfattention and linear-attention baselines.

## 2. Related Work

## 2.1. Efficient Vision Transformer

Despite their success in computer vision, attention-based models are computationally expensive due to the quadratic scaling of self-attention with the number of tokens. To reduce this cost, ViT, VOLO, and T2T-ViT [10, 46, 47] use patchification or progressive tokenization, with similar approaches applied to downstream tasks [22]. Other methods reduce attention costs by restricting token interactions: PVT/PVTv2 [38, 39] reduce key/value resolution, Swin [25] uses local windows, NAT [18] uses neighborhood attention, and BiFormer [52] adopts bi-level routing. However, the cost of softmax-attention remains a challenge for high-resolution images.

## 2.2. Linear Attention

An alternative to quadratic self-attention is Linear attention (linear-attention) [20], which reduces the quadratic cost of self-attention by factorizing query–key interactions and computing $K ^ { \top } V$ , reducing complexity from $\mathcal { O } ( n ^ { 2 } d )$ to $\mathcal { O } ( n d ^ { 2 } )$ while preserving the global context.

To improve the efficiency and expressiveness of linearattention, Performer uses random features [6], CosFormer uses cosine re-weighting [29], SOFT adopts low-rank approximations [27, 44], and Efficient Attention separate query/key normalization [31]. FLatten token selectivity [14], and Hydra and Castling-ViT explore alternative similarity functions [1, 45]. More recently, MILA adopts Mamba-inspired designs [16], InLine addresses noninjectivity and locality [15], RALA mitigates the low-rank bottleneck [12], and PolaFormer models polarity-aware interactions [28]. Despite these advances, most linearattention methods primarily model direct token interactions. Our LiSA captures higher-order, transitive interactions through truncated random walks while preserving linear complexity and achieving competitive accuracy compared to softmax-attention.

## 2.3. Higher-Order Attention

Several works extend attention beyond direct interactions through diffusion or graph filtering. MAGNA [36] diffuses attention over multiple hops in graph neural networks, while Diffuser [13] extends sparse Transformer attention with multi-hop diffusion. From a graph-signal-processing perspective, GFSA [5] interprets self-attention as a firstorder graph filter and generalizes it through polynomial filtering. More recently, AGF [40] extends this perspective to linear Transformers by learning graph filters directly in the singular-value domain while retaining linear complexity. Attention Chains [11] instead interprets existing softmax attention matrices as Markov chains, using repeated transitions to characterize indirect attention and stationary token importance.

Our LiSA also models higher-order interactions, but differs in how they are computed. Unlike MAGNA and Diffuser, which diffuse attention over graph or sparse-attention structures, LiSA constructs a stochastic transition matrix directly from kernelized query–key affinities. While this can also be viewed as polynomial graph filtering, unlike AGF’s spectral formulation, LiSA uses non-negative, normalized step weights to control multi-hop propagation. This allows information to flow through intermediate tokens while maintaining a probabilistic interpretation, with a local attention branch to preserve spatial details. Unlike Attention Chains, LiSA learns these interactions end-to-end to produce layer representations. Its low-rank factorization computes higher-order terms in the $d _ { h } \times d _ { h }$ space, preserving linear complexity in the number of tokens.

## 3. Method

In Section 3.1 we give a brief definition of the self-attention mechanism and the linear version. We then proceed with the

description and formulation of our proposed LiSA module in Section 3.2.

## 3.1. Preliminaries

Self-attention (SA). We start by presenting the selfattention model of transformers. Given a set of n tokens $\boldsymbol { X } \in \mathbb { R } ^ { n \times d } .$ , each represented by a d-dimensional vector, self-attention maps X to a set of queries $Q = X W _ { Q } \in \mathbb { R } ^ { n \times d _ { h } }$ , keys $K = X W _ { K } \in \mathbb { R } ^ { n \times d _ { h } }$ and values $V = \dot { X } W _ { V } \in \mathbb { R } ^ { n \times d }$ , where $W _ { Q } , W _ { K }$ and $W _ { V }$ are parameter matrices learned in training. The output of the selfattention layer is obtained as:

$$
Y = \mathrm { s o f t m a x } \Big ( \frac { Q K ^ { \top } } { \sqrt { d } } \Big ) V\tag{1}
$$

The output can also be expressed for a specific token i as:

$$
y _ { i } = \sum _ { j = 1 } ^ { N } a _ { i j } v _ { j } , \quad { \mathrm { w h e r e } } \quad a _ { i j } = { \frac { \exp \left( q _ { i } ^ { \top } k _ { j } / { \sqrt { d } } \right) } { \sum _ { j ^ { \prime } = 1 } ^ { N } \exp \left( q _ { i } ^ { \top } k _ { j ^ { \prime } } / { \sqrt { d } } \right) } } ,\tag{2}
$$

and $y _ { i } , v _ { j } \in \mathbb { R } ^ { d }$ . Self-attention requires computing the attention score $a _ { i j }$ between each pair of tokens before the softmax, leading to $\mathcal { O } ( n ^ { 2 } )$ time and memory complexity. This quadratic scaling makes high-resolution inputs and long sequences prohibitively expensive.

In this work, we seek an alternative to self-attention with the following properties: (1) Scalability: The complexity of the model in terms of memory and computation should increase linearly with respect to the number of tokens, during training as well as during inference. (2) Expressiveness: The model should also have expressiveness similar to that of self-attention while maintaining a comparable parameter budget. In particular, it should capture global and non-linear relations between input tokens.

Linear Attention (LA). Linear-attention [20] offers an effective alternative to softmax-attention by reducing the complexity from $\mathcal { O } ( n ^ { 2 } )$ to ${ \mathcal { O } } ( n )$ in the sequence length N. The key idea is to replace the softmax similarity with a positive feature map $\phi ( \cdot )$ so that the pairwise similarity can be written in a kernelized form $\phi ( Q ) \phi ( K ) ^ { \top }$ . With this factorization, the output for token i becomes:

$$
y _ { i } = { \frac { \sum _ { j = 1 } ^ { N } { \Big ( } \phi ( q _ { i } ) \phi ( k _ { j } ) ^ { \top } { \Big ) } v _ { j } } { \sum _ { j = 1 } ^ { N } { \Big ( } \phi ( q _ { i } ) \phi ( k _ { j } ) ^ { \top } { \Big ) } } } = { \frac { \phi ( q _ { i } ) { \Big ( } \sum _ { j = 1 } ^ { N } \phi ( k _ { j } ) ^ { \top } v _ { j } { \Big ) } } { \phi ( q _ { i } ) { \Big ( } \sum _ { j = 1 } ^ { N } \phi ( k _ { j } ) ^ { \top } { \Big ) } } }\tag{3}
$$

thereby avoiding the $n \times n$ attention matrix and yielding linear-time computation.

Despite their efficiency, current linear-attention approaches primarily perform a single-pass aggregation: $\bar { Y } = \phi ( Q ) \bar { ( } \phi ( K ) ^ { \top } \bar { V } )$ mixes each query with values exactly once via first-order (one-hop) affinities. This design fails to materialize transitive dependencies involving intermediate tokens, which would arise from higher powers of $A = \phi ( Q ) \phi ( K ) ^ { \top }$

## 3.2. Linear Stochastic Attention (LiSA)

Our LiSA approach is based on discrete random walks. This stochastic process models the probability of moving among the nodes V of a graph $\mathcal { G } = ( \nu , \mathcal { E } )$ with a time-invariant transition probability matrix $S \in \mathbb { R } ^ { | \nu | \times | \nu | }$ where each $s _ { i j }$ for $( i , j ) \in \mathcal { E }$ defines the probability of moving from node i to node j at any given time. Matrix S is row-stochastic as $\textstyle \sum _ { j } s _ { i j } = 1 , \forall i \in \mathcal { V }$

At each step, the random walk can either continue by moving to another node with probability $0 \leq \beta < 1$ , in which case the next node is selected based on the transition probability matrix S, or stop with probability $( 1 - \beta )$ Denote by $p _ { i j } ^ { ( T ) }$ the probability of ending the walk at node j when starting at node i under a truncation length T. Here, $T$ denotes the truncation length, i.e., the number of terms retained in the diffusion expansion, corresponding to powers $S ^ { 0 } , \ldots , S ^ { T - 1 }$ . The random walk probabilities can be expressed as a geometric series:

$$
P ^ { ( T ) } = \left( \frac { 1 - \beta } { 1 - \beta ^ { T } } \right) \sum _ { k = 0 } ^ { T - 1 } \beta ^ { k } S ^ { k } .\tag{4}
$$

Proposition 1. For walks of infinite length, probabilities can be computed analytically as:

$$
P ^ { ( \infty ) } = ( 1 - \beta ) \bigl ( I - \beta S \bigr ) ^ { - 1 } .\tag{5}
$$

The proofis provided in the supplementary material.

To model global relationships between tokens, we follow the self-attention approach based on matrices representing the K, Q and V of tokens. However, we consider a more general formulation $Q = \phi ( X W _ { Q } ) \in \mathbb { R } ^ { n \times d _ { h } }$ ， $K = \phi ( X W _ { K } ) \in \mathbb { R } ^ { n \times d _ { h } } , V = X W _ { V } \in \mathbb { R } ^ { n \times d }$ , where $\phi$ is a point-wise activation function.

Considering tokens as nodes in the graph, the transition probability from a token i to a token j is then computed as:

$$
s _ { i j } = \frac { q _ { i } k _ { j } ^ { \top } } { \sum _ { j ^ { \prime } } q _ { i } k _ { j ^ { \prime } } ^ { \top } } .\tag{6}
$$

This can be expressed in matrix form as: ${ \cal { S } } \quad = \quad$ $( D ^ { - 1 } Q ) K ^ { \top } \ = \ \mathsf { \bar { Q } } K ^ { \top }$ with $D \ = \ \operatorname { D i a g } ( Q ( K ^ { \top } { \bf 1 } ) )$ . To avoid division by zero, we add a small constant $\epsilon = 1 0 ^ { - 6 }$ to the normalization denominator. Intuitively, the transition probability $s _ { i j }$ encodes information similar to the attention score $a _ { i j }$ of self-attention, where a higher value means a stronger relationship between tokens i and j.

To obtain the transformed features of a token i, we imagine a generative process where a random walk is initiated at node i and, once the walk stops at a node $j ,$ the value vector $v _ { j }$ is generated. For a potentially infinite random walk with continuation probability $\beta ,$ the expected features are then given by:

![](images/f3a6c7ee5874087a01e443a772248bf5959c2f683e17220361a37f19bba5a192.jpg)  
Figure 2. Overview of LiSA. We patchify the input image into tokens and treat them as nodes in a graph $G = ( V , E )$ , where edge weights define a row-stochastic transition matrix S. Aggregating the truncated terms on $S$ yields a global attention map that captures transitive (higher-order) dependencies. In parallel, we compute content-based similarity within each token’s local neighborhood to form a local attention map. The layer output is a convex combination of the two, blending sharp local context with long-range structure.

$$
\begin{array} { r } { Y _ { \mathrm { g l o b a l } } = A _ { \mathrm { g l o b a l } } V = ( 1 - \beta ) \left( \overbrace { \sum _ { k = 0 } ^ { \infty } \beta ^ { k } ( \tilde { Q } K ^ { \top } ) ^ { k } } ^ { \left( I - \beta \tilde { Q } K ^ { \top } \right) ^ { - 1 } } \right) V . } \end{array}\tag{7}
$$

The formulation in Eq. (7) satisfies the desired property of expressiveness, as the function models an infinite-order relation between the input X and output Y with the same number of trained parameters as self-attention. Moreover, it can be computed effectively, with a complexity linear in the number of tokens n, using the property that $\dot { S } = \tilde { Q } K ^ { \top }$ is low rank. Nevertheless, matrix inversion, and in particular computing the derivative of this operation during backpropagation, remains expensive. In the next section, we present a more efficient solution based on truncated random walks.

Truncated Random Walk Approximation. To avoid matrix inversions, we retain the first T terms of the randomwalk expansion. In this case, the computation of output fea-

tures becomes:

$$
\begin{array} { l } { { \displaystyle Y _ { \mathrm { g l o b a l } } = \left( \frac { 1 - \beta } { 1 - \beta ^ { T } } \right) \left( \sum _ { k = 0 } ^ { T - 1 } \beta ^ { k } S ^ { k } \right) V } } \\ { { = \displaystyle \left( \frac { 1 - \beta } { 1 - \beta ^ { T } } \right) \left( I + \sum _ { k = 1 } ^ { T - 1 } \beta ^ { k } \tilde { Q } ( K ^ { \top } \tilde { Q } ) ^ { k - 1 } K ^ { \top } \right) V } } \\ { { = \displaystyle \left( \frac { 1 - \beta } { 1 - \beta ^ { T } } \right) \left( V + \beta \tilde { Q } \left( \sum _ { \underbrace { k = 0 } _ { d _ { \mathrm { h } } \times d _ { h } } } ^ { T - 2 } \beta ^ { k } ( K ^ { \top } \tilde { Q } ) ^ { k } \right) K ^ { \top } V \right) } } \end{array}\tag{8}
$$

This gives rise to the efficient computational process in Algorithm 1. Each loop of this algorithm multiples two $d _ { h } \times d _ { h }$ matrices, which has a complexity $\mathcal { O } ( d _ { h } ^ { 3 } )$ . Therefore, the total complexity of the algorithm is $\mathcal { O } ( T \cdot d _ { h } ^ { 3 } ) +$ $O ( n \cdot d \cdot d _ { h } )$ Due to the added $\bar { T }$ factor, the truncation strategy has the important advantage of requiring only matrix multiplications that can be easily parallelized on GPU. However, since this approach limits the random walk to a finite length, it can produce results that differ from those of the original infinite random walk model. Nevertheless, as shown in the following proposition, this discrepancy rapidly diminishes as the walk length increases.

Proposition 2. Let $R _ { T }$ be the truncation residual obtained after retaining the first T terms in $P ^ { ( \infty ) }$ . The $L _ { 1 }$ norm of $R _ { T }$ decays exponentially as the truncation depth T increases.

$$
R _ { T } = ( 1 - \beta ) \left( \sum _ { k = T } ^ { \infty } \beta ^ { k } S ^ { k } \right) .\tag{9}
$$

The proof is provided in the supplementary material.

Learnable Step Weights. To further increase the expressiveness of the model, we allow the step weights in Eq. (8) to be learnable and input-dependent. Specifically, we replace the fixed geometric weights $\left( { \frac { 1 - \beta } { 1 - \beta ^ { T } } } \right) \beta ^ { k }$ with nonnegative weights $\beta _ { k }$ predicted from the input and normalized across the $T$ retained terms. This yields:

$$
{ \boldsymbol { Y } } _ { \mathrm { g l o b a l } } = \beta _ { 0 }  { \boldsymbol { V } } + \sum _ { k = 0 } ^ { T - 2 } \beta _ { k + 1 } \tilde {  { \boldsymbol { Q } } } (  { \boldsymbol { K } } ^ { \top } \tilde {  { \boldsymbol { Q } } } ) ^ { k }  { \boldsymbol { K } } ^ { \top }  { \boldsymbol { V } } ,\tag{10}
$$

where $\beta _ { k } \geq 0$ and $\textstyle \sum _ { k = 0 } ^ { T - 1 } \beta _ { k } = 1$ . Further details on the generation of the step weights are provided in the supplementary material.

Link to softmax in SA. Softmax attention adds nonlinearity through the pointwise exponential function: $\scriptstyle \exp ( x ) = \sum _ { k = 0 } ^ { \infty } { \frac { x ^ { k } } { k ! } }$ . In contrast, the random-walk (RW) diffusion probabilities are defined by a matrix power series over the transition matrix S: ${ \cal P } ^ { ( \infty ) } = ( 1 { - } \beta ) \bar { \sum _ { k = 0 } ^ { \infty } \beta ^ { k } } S ^ { k }$ Because each power $S ^ { k }$ captures k-step transitions, RW diffusion models higher-order, non-pointwise relationships across the graph.

## Adaptive Local Attention.

In LiSA, increasing the truncation length T retains higher-order powers of the row-stochastic transition operator, progressively extending information propagation across patches. Since each step updates $\mathbf { x } _ { i }$ as a weighted average of other tokens, $\| \mathbf { x } _ { i } - \mathbf { x } _ { j } \|$ shrinks over steps, making distinct regions (e.g., foreground and background) harder to separate (over-smoothing). We add a local branch that restricts mixing to a $3 \times 3$ neighborhood to preserve spatial detail, while the global branch provides long-range context. This module models local interactions within a small neighborhood window around each token. For each patch token, affinities are computed only among its immediate spatial neighbors, with higher transition probabilities assigned to semantically similar regions. Let the spatial position of token i be $( h _ { i } , w _ { i } ) \in \mathbb { Z } ^ { 2 }$ and define:

$$
\begin{array} { c } { { \mathcal { N } ( i ) = \big \{ j : ( h _ { j } , w _ { j } ) = ( h _ { i } , w _ { i } ) + ( \Delta _ { h } , \Delta _ { w } ) , } } \\ { { ( \Delta _ { h } , \Delta _ { w } ) \in \{ - 1 , 0 , 1 \} ^ { 2 } \big \} . } } \end{array}\tag{11}
$$

We then form a sparse local affinity matrix $A ^ { \mathrm { l o c a l } } \in \mathbb { R } ^ { N \times N }$ whose non-zero entries connect a node i only to its spatial neighbors:

```latex
Algorithm 1 Truncation-based computation
Require: Truncation length T
$M \gets I$
$A  \beta K ^ { \top } \tilde { Q }$
$Y  V$
$B  K ^ { \top } V$
for $k = 1 , \ldots , T { - } 1$ do
$Y \gets Y + \beta \tilde { Q } ( M B )$
$M \gets A \cdot M$
end for
$\begin{array} { r } { Y \gets \left( \frac { 1 - \beta } { 1 - \beta ^ { T } } \right) Y } \end{array}$
return Y
```

$$
{ \cal A } _ { \mathrm { l o c a l } } = { \cal M } \odot \exp \left( \frac { Q K ^ { \top } } { \sqrt { d } } \right)\tag{12}
$$

where $M _ { i j } = \mathbf { 1 } \{ j \in N ( i ) \}$ . We row-normalize this affinity matrix to obtain the local attention transition. The neighborhood includes a self-transition through the (0, 0) offset, which helps stabilize local aggregation. Without this selfloop, each token would be forced to replace itself with a weighted average of its neighboring tokens. All entries outside $\mathcal { N } ( i )$ are zero and are never explicitly computed, so the local branch has linear complexity in the number of tokens.

Mixture of Operators. We combine the outputs of the global and local branches, allowing the model to trade off long-range higher-order aggregation with local neighborhood refinement using a learnable gate:

$$
Y = ( 1 - \alpha ) \underbrace { A _ { \mathrm { g l o b a l } } V } _ { Y _ { \mathrm { g l o b a l } } } + \alpha \underbrace { A _ { \mathrm { l o c a l } } V } _ { Y _ { \mathrm { l o c a l } } } ,\tag{13}
$$

where $\alpha ~ \in ~ [ 0 , 1 ]$ is a learnable parameter that balances global and local information.

## 4. Experiments

We validate the effectiveness of our method on ImageNet-1K classification [9], ADE20K semantic segmentation [51], and COCO object detection [23], and include extensive ablations to assess each design component.

## 4.1. Image Classification

Datasets. We use the ImageNet-1K dataset [9] for image classification. This dataset comprises 1.28M training images and 50K validation images across 1,000 categories.

Settings. We integrate our module into four Vision Transformer backbones, DeiT [33], PVT [38], Swin Transformer [25], and BiFormer [52] and report Top-1 accuracy on the validation split. For a fair comparison, we adopt the same training recipe as the baseline. All models are trained from scratch for 300 epochs with AdamW [26], a cosine-decayed learning rate and a 20-epoch linear warmup. We initialize the LR at $1 \times 1 0 ^ { - 3 }$ and use a weight decay of 0.05. Data augmentation and regularization include RandAugment [8], Mixup [49], CutMix [48], and Random Erasing [50]. The experiments were conducted on eight NVIDIA A100/H100 GPUs. We replace every self-attention layer with our LiSA module and use T = 3 for all main experiments. Since our method is a plug-in, it can be integrated into existing linear-attention architectures without other modifications. In these experiments, we use FLatten as the baseline linear attention, which combines a focusing function with a lightweight depthwise-convolution branch. In the ablation, we also evaluate additional baselines.

Table 1. Results of Classification. Comparison of LiSA across diverse vision backbones. Red indicates gains over the self-attention baseline (in all tables).
<table><tr><td>Model</td><td></td><td>| Input | Params (M) FLOPs (G) | Top1-acc(%)</td><td></td><td></td></tr><tr><td colspan="5">DeiT Family</td></tr><tr><td>DeiT-T [33]</td><td>224</td><td>5.72</td><td>1.26</td><td>72.2</td></tr><tr><td>FLatten-DeiT-T [14]</td><td>224</td><td>6.19</td><td>1.15</td><td>74.1</td></tr><tr><td>LiSA-DeiT-T</td><td>224</td><td>6.30</td><td>1.29</td><td>74.54+2.34</td></tr><tr><td>DeiT-T [33]</td><td>384</td><td>5.79</td><td>4.70</td><td>74.9</td></tr><tr><td>FLatten-DeiT-T [14]</td><td>384</td><td>7.14</td><td>3.37</td><td>76.53</td></tr><tr><td>LiSA-DeiT-T</td><td>384</td><td>7.24</td><td>3.75</td><td>77.54 +2.64</td></tr><tr><td>DeiT-S [33]</td><td>224</td><td>22.05</td><td>4.61</td><td>79.8</td></tr><tr><td>FLatten-DeiT-S [14]</td><td>224</td><td>22.98</td><td>4.39</td><td>80.1</td></tr><tr><td>LiSA-DeiT-S</td><td>224</td><td>23.08</td><td>4.66</td><td>80.78 +0.98</td></tr><tr><td colspan="5">PVT Family</td></tr><tr><td>PVT-T [38]</td><td>224</td><td>13.23</td><td>1.94</td><td>75.1</td></tr><tr><td>FLatten-PVT-T [14]</td><td>224</td><td>12.14</td><td>2.00</td><td>77.8</td></tr><tr><td>LiSA-PVT-T</td><td>224</td><td>12.16</td><td>2.12</td><td>78.53 +3.43</td></tr><tr><td>PVT-S [38]</td><td>224</td><td>24.48</td><td>3.82</td><td>79.8</td></tr><tr><td>FLatten-PVT-S [14]</td><td>224</td><td>21.66</td><td>3.95</td><td>81.7</td></tr><tr><td>LiSA-PVT-S</td><td>224</td><td>21.69</td><td>4.18</td><td>81.96 +2.16</td></tr><tr><td colspan="5">Swin Family</td></tr><tr><td>Swin-T [25]</td><td>224</td><td>28.29</td><td>4.49</td><td>81.3</td></tr><tr><td>FLatten-Swin-T [14]</td><td>224</td><td>29.71</td><td>4.49</td><td>82.1</td></tr><tr><td>LiSA-Swin-T</td><td>224</td><td>29.74</td><td>4.72</td><td>82.25 +0.95</td></tr></table>

Results. Table 1 compares ImageNet Top-1 accuracy across DeiT, PVT, and Swin backbones. With comparable parameter counts and FLOPs, LiSA consistently outperforms the baseline and FLatten. On DeiT at 224 input resolution, LiSA improves DeiT-T/S by +2.34%/+0.98% and further surpasses FLatten-DeiT-T/S by +0.44%/+0.68%. On PVT, LiSA yields large gains over the baselines (+3.43% on PVT-T and +2.16% on PVT-S) and improves over FLatten by +0.73%/+0.26%. On Swin, LiSA also brings +0.95% improvement compare to selfattention. At a higher resolution (384×384) on DeiT-T, LiSA reaches 77.54%, improving over FLatten by +1.01 points (76.53%), while using fewer FLOPs than the original DeiT-T at the same input size. Table 2 shows LiSA transfers to BiFormer as well, improving classification by +1.0%/+0.3%/+0.7% for T/S/B.

Table 2. Four-stage pyramid architecture: classification and segmentation results. Segmentation is with S-FPN. Dashes indicate values not reported.
<table><tr><td>Model</td><td>Par. (M)</td><td>Top1-acc(%)</td><td>mIoU</td><td>mAcc</td></tr><tr><td>BiFormer-T [52]</td><td>13.1</td><td>81.4</td><td></td><td></td></tr><tr><td>LiSA-BiFormer-T</td><td>13.2</td><td>82.4+1.0</td><td></td><td></td></tr><tr><td>BiFormer-S [52]</td><td>25.5</td><td>83.8</td><td>48.9</td><td></td></tr><tr><td>LiSA-BiFormer-S</td><td>25.6</td><td>84.1 +0.3</td><td>49.5 +0.6</td><td>61.5</td></tr><tr><td>BiFormer-B [52]</td><td>56.8</td><td>84.3</td><td></td><td></td></tr><tr><td>LiSA-BiFormer-B</td><td>56.8</td><td>85.0 +0.7</td><td></td><td></td></tr></table>

## 4.2. Semantic Segmentation

Datasets. For semantic segmentation we use ADE20K [51], the standard semantic-segmentation benchmark, which includes 20K training images, 2K validation images, and 150 categories.

Settings. For a fair comparison, we use the same baseline training setup. We plug our method into two representative segmentation frameworks, Semantic FPN [21] and UPerNet [42], implemented in MMSegmentation [7]. Backbones are initialized from ImageNet-1K. We train Semantic FPN following the PVT recipe [38] and UPerNet using the default Swin configuration [25].

Table 3. Results of semantic segmentation. FLOPs are computed with an input of 512 × 2048. S-FPN denotes SemanticFPN [21]. Dashes indicate values not reported.
<table><tr><td colspan="5">Semantic Segmentation on ADE20K</td></tr><tr><td>Backbone</td><td></td><td>|Method | FLOPs (G) Par. (M) |</td><td>mIoU</td><td>mAcc</td></tr><tr><td>PVT-T</td><td>S-FPN</td><td>158 17</td><td>36.57</td><td>46.72</td></tr><tr><td>FLatten-PVT-T</td><td>S-FPN</td><td>169 16</td><td>37.21</td><td>48.95</td></tr><tr><td>LiSA-PVT-T</td><td>S-FPN</td><td>170 16</td><td></td><td>40.85 +4.28  51.75 +5.03</td></tr><tr><td>Swin-T</td><td>|UperNet |</td><td>945 60</td><td>44.51</td><td>55.61</td></tr><tr><td>FLatten-Swin-T</td><td>UperNet</td><td>946 60</td><td>44.82</td><td>57.01</td></tr><tr><td>LiSA-Swin-T</td><td>UperNet</td><td>947 60</td><td>45.03 +0.52</td><td>56.4-0.61</td></tr></table>

Results. As shown in Table 3, our method delivers consistent gains across all settings. On PVT, it improves mIoU by more than 4.0 points and mAcc by more than 5.0 points over the baselines, under comparable FLOPs and parameter counts. On Swin, it improves mIoU of the baseline by 0.5 points respectively. LiSA-BiFormer-S (Table 2) also improves the corresponding mIoU baseline by 0.6 points. These results show that local attention effectively captures informative features for segmentation.

## 4.3. Object Detection

Datasets. For object detection we use the COCO object detection and instance segmentation benchmark [23], which includes 118K training images and 5K validation images.

Settings. We assess our method within Mask R-CNN [19], initializing backbones with ImageNet pretraining. Experiments use 1× schedules with varying detection heads.

Results. As shown in Table 4, our method achieves the best accuracy with near-zero compute overhead. On PVT: box AP and mask AP improve by 3.8 and 2.5 points, respectively, compared with the PVT baseline. On Swin: box AP and mask AP improve by 0.8 and 0.5 points, respectively, compared with the Swin baseline

Table 4. Results of object detection. The FLOPs are computed over backbone, FPN and detection head with input resolution of $1 2 8 0 \times 8 0 0 .$
<table><tr><td colspan="8">Mask R-CNN Object Detection on COCO</td></tr><tr><td>Method</td><td>FLOPs</td><td>Sch.</td><td>APb</td><td>AP50 AP75</td><td>APm</td><td>AP50</td><td>AP75</td></tr><tr><td>PVT-T</td><td>240G</td><td>1x</td><td>36.7</td><td>59.2 39.3</td><td>35.1</td><td>56.7</td><td>37.3</td></tr><tr><td>FLatten-PVT-T</td><td>244G</td><td>1x</td><td>38.2 61.6</td><td>41.9</td><td>37.0</td><td>57.6</td><td>39.0</td></tr><tr><td>LiSA-PVT-T</td><td>245G</td><td>1x</td><td>40.5 +3.8</td><td>62.9 +3.8 44.1 +4.8</td><td>37.6 +2.5</td><td>59.7 +3.0</td><td>40.2 +2.9</td></tr><tr><td>Swin-T</td><td>267G</td><td>1x</td><td>43.7</td><td>66.6 47.7</td><td>39.8</td><td>63.3</td><td>42.7</td></tr><tr><td>FLatten-Swin-T</td><td>268G</td><td>1x</td><td>44.2</td><td>67.3 48.5</td><td>40.2</td><td>63.8</td><td>43.0</td></tr><tr><td>LiSA-Swin-T</td><td>269G</td><td>1x</td><td>44.5+0.8 67.4+0.8 48.7 +1.0</td><td></td><td></td><td>40.3+0.5 63.9 +0.6 43.1 +0.4</td><td></td></tr></table>

## 4.4. Ablation Study

We conduct ablations to analyze the impact of individual components in our method. Unless stated otherwise, all ablation results are reported on the mini-ImageNet dataset [35] $( T = 5 )$ for computational efficiency.

Comparison with other linear-attention methods. For a fair comparison with prior linear-attention modules, we compare against six designs: Hydra Attention [1], Vanilla linear-attention [20], Efficient Attention [31], Linear Angular Attention [45], Enhanced linear-attention [2], and Focused linear-attention (FLatten) [14]. Table 5 shows that LiSA outperforms all alternatives, including softmaxattention.

Table 5. Comparison of linear-attention variants on DeiT-S (ImageNet).
<table><tr><td>Linear Attention</td><td>|Top1-acc(%)</td></tr><tr><td>DeiT-S [33] Hydra Attn [1]</td><td>79.8 75.38</td></tr><tr><td>Vanilla Linear Attn [20]</td><td>76.47</td></tr><tr><td>Efficient Attn [31]</td><td>76.81</td></tr><tr><td>Linear Angular Attn [45]</td><td>79.21</td></tr><tr><td>Enhanced Linear Attn [2]</td><td>77.17</td></tr><tr><td>Focused Linear Attn [14]</td><td>80.1</td></tr><tr><td>LiSA</td><td>80.78+0.98</td></tr></table>

Ablation for α. Table 6 studies α, which balances LiSA’s global and local branches. We compare global-only (α=0, F+D+G), local-only (α=1, F+D+L), and learned α. On both DeiT tiny and small, learning α performs best (76.3% and 80.7%), outperforming both individual branches. Moreover, global-only exceeds local-only (73.7 vs. 73.4 on DeiT-T; 78.3 vs. 78.1 on DeiT-S), suggesting that long-range transitive aggregation provides a stronger standalone signal than local neighborhood matching.

Table 6. Component ablation on DeiT-T/DeiT-S classification.
<table><tr><td>Backbone</td><td>F+D+G (α=0)</td><td>F+D+L (α=1)</td><td>Learned α</td></tr><tr><td>DeiT-T</td><td>73.7</td><td>73.4</td><td> ${ \bf 7 6 . 3 _ { \mathrm { ~ + 2 . 6 } } }$ </td></tr><tr><td>DeiT-S</td><td>78.3</td><td>78.1</td><td> $\mathbf { 8 0 . 7 \ + 2 . 4 }$ </td></tr></table>

Number of truncations. We vary the truncation length T in Eq. 4. All LiSA components and settings are kept fixed; only the truncation length T is varied. Figure 3 shows that accuracy peaks at a moderate truncation length $( T = 5 )$ . Across three seeds, $T > 1$ consistently outperforms $T = 1$ , isolating the benefit of higher-order propagation. Among the higher-order settings, $T = 3$ achieves the second-highest mean accuracy $( 7 5 . 6 3 \pm 0 . 3 9 )$ , closely approaching the best-performing $T = 5 \ : ( 7 5 . 7 3 \pm 0 . 4 7 )$ . We use $T = 5$ for the ablation studies due to its highest mean accuracy, while adopting $T = 3$ for the main ImageNet experiments to reduce computational cost by requiring fewer propagation terms with minimal loss in accuracy.

![](images/fd9dfe4af1ee7c3b3df82f430d7f16230e1242b516ae1de8fe55ab9acecfdf69.jpg)  
Figure 3. Effect of truncation length T on DeiT-T accuracy, reported as mean ± standard deviation across three runs.

Statistical variability. In Figure 3, we report results across three runs for truncation depths $T = 1$ to $T = 5 ,$ including the mean and standard deviation. The results confirm that higher-order propagation consistently improves over $T = 1$ (e.g., $7 5 . 6 3 \pm 0 . 3 9$ for $T = 3$ and $7 5 . 7 3 \pm 0 . 4 7$ for $T = 5 ,$ versus $7 3 . 3 4 \pm 0 . 6 4$ for $T ~ = ~ 1 )$ , with $T \ = \ 5$ achieving the highest mean accuracy. The relatively small differences among the stronger higher-order settings, together with their consistent gains over $T = 1$ , indicate that the benefit is robust to run-to-run variability rather than driven by a single run.

Source of improvement. GL is our proposed attention core, a direct plug-in replacement for LA. F and D are additions borrowed from FLatten and are applied identically on top of both LA and GL. Our contribution is to replace the vanilla linear attention core (LA) with a higherorder global-local stochastic diffusion operator (GL). Table 7 isolates the contribution of our core: First, GL > LA (69.2 vs. 65.7), showing that our core already improves over vanilla linear attention before introducing F or D. Second, the same trend holds when each auxiliary component is added independently: GL+F > LA+F (69.3 vs. 65.9) and GL+D > LA+D (75.2 vs. 73.3). Notably, GL+D even surpasses the full FLatten baseline LA+F+D (75.2 vs. 74.4). Finally, when both F and D are applied equally to both cores, GL+F+D > LA+F+D (76.3 vs. 74.4), showing that LiSA outperforms FLatten under the same auxiliary design. These consistent gains indicate that the improvement comes from our proposed GL mechanism rather than from the borrowed auxiliary components.

![](images/e43081a6b7cb58913a7c4b04cf333d9cdd9c5fab4e7dd9d816288b98ea61dfc4.jpg)

(a) DeiT-T attention map. The red square marks the query patch.  
![](images/601e9c7ed83860d6b8b14cede747094ded1ee172ae51b950eb3a888b0d5dd2d5.jpg)  
(b) Transitive reasoning visualized in DeiT-T attention maps.  
Figure 4. Visualization of attention behavior in DeiT-T.

Table 7. Component ablation on DeiT-T classification.
<table><tr><td>LA</td><td>GL</td><td>LA+F</td><td>GL+F</td><td>LA+D</td><td>GL+D</td><td>FLatten LA+F+D</td><td>LiSA GL+F+D</td></tr><tr><td>65.7</td><td>69.2</td><td>65.9</td><td>69.3</td><td>73.3</td><td>75.2</td><td>74.4</td><td>76.3</td></tr></table>

Different linear-attention baseline. This experiment highlights the plug-in nature of our method and evaluates whether LiSA generalizes beyond FLatten. On DeiT-T, with linear angular attention [45], the baseline reaches 66.0% Top-1, while adding LiSA improves it to 73.4%. Furthermore, integrating LiSA into Agent Attention [17] improves DeiT-T from 73.40 to 75.77 (+2.37) and DeiT-S from 78.83 to 80.84 (+2.01), demonstrating that LiSA transfers effectively to other linear-attention variants.

Distribution of linear-attention. As shown in Figure 4a, vanilla linear-attention produces diffuse maps: the weights are spread across many locations, so each query effectively aggregates an almost averaged representation of the whole image. In contrast, LiSA yields more structured, non-uniform patterns, where larger weights are assigned to patches on the same object as the query, while most background areas receive only small contributions.

Visualization of Transitive Reasoning. In Figure 4b, we start from a query patch on the bird (red box). Local attention remains restricted to nearby patches, so its influence stays spatially confined. In contrast, LiSA expands its receptive field with hop depth: as we increase the truncation length (T=3, 5, 7), attention progressively propagates along semantically related regions, reaching distant yet correlated parts of the bird and eventually covering the tail and most of the object. Please zoom in for a clearer view.

Practical efficiency. We provide latency and peak GPU memory plots across sequence lengths. The efficiency advantage of LiSA over softmax attention becomes more pronounced as sequence length increases. DeiT-T scales quadratically in both latency and memory, becoming prohibitively expensive at long sequences, while FLatten and LiSA maintain linear scaling. LiSA incurs a modest latency overhead compared to FLatten, which is expected given its higher-order random-walk modeling and adaptive local branch. This overhead reflects a trade-off between efficiency and expressiveness. Memory-wise, LiSA preserves linear scaling and remains substantially more efficient than softmax attention, with some additional overhead over FLatten due to its higher-order modeling.

![](images/356e5ba660702159d47cf0b250db07ff18b66f1ab00abdbdcd3e960f2c51e4e0.jpg)  
(a) Latency

![](images/956f2b123b5fc7990121b7d8f26b94697df41560c5016e2cd887551d669690a1.jpg)  
(b) GPU memory  
Figure 5. Efficiency comparison across sequence lengths.

## 5. Conclusion

We introduced LiSA, a linear-attention plug-in module that restores higher-order token interactions via a truncated random-walk formulation, while preserving stability and locality through an adaptive 3×3 neighborhood. This design directly overcomes the first-order limitation of vanilla linear-attention, enabling richer transitive reasoning without sacrificing linear complexity. Across image classification, semantic segmentation, and object detection, LiSA yields consistent gains at near-identical compute budgets and transfers cleanly to diverse backbones (DeiT, PVT, Swin, BiFormer). Ablations confirm that each component contributes meaningfully and that a short truncation length already offers a strong accuracy–latency trade-off, positioning LiSA as a more powerful drop-in alternative to standard kernelized linear-attention. As future work, we plan to integrate LiSA into additional state-of-the-art linearattention variants and study its behavior under diverse training recipes and data regimes.

# LiSA: Linear Stochastic Attention for Long-Range Visual Modeling

Supplementary Material

## 6. Proofs of the Propositions

This section provides proofs for the propositions introduced in the methodology.

Proposition 3. For walks of infinite length, probabilities can be computed analytically as

$$
P ^ { ( \infty ) } = ( 1 - \beta ) \big ( I - \beta S \big ) ^ { - 1 }\tag{14}
$$

Proof. Consider the sum over the first T terms in Eq. (4). Pre-multiplying this sum by $( I - \beta S )$ gives

$$
\begin{array} { l } { { ( I - \beta S ) P ^ { ( T ) } = \left( \displaystyle \frac { 1 - \beta } { 1 - \beta ^ { T } } \right) \left( \displaystyle \sum _ { k = 0 } ^ { T - 1 } \beta ^ { k } S ^ { k } - \displaystyle \sum _ { k = 1 } ^ { T } \beta ^ { k } S ^ { k } \right) } } \\ { { = \displaystyle \left( \displaystyle \frac { 1 - \beta } { 1 - \beta ^ { T } } \right) \left( I - ( \beta S ) ^ { T } \right) . \qquad ( 1 5 } } \end{array}
$$

Since $\beta < 1$ , we have that $( \beta S ) ^ { T } \to 0$ and $\beta ^ { T } \to 0$ when $T \to \infty$ , hence

$$
\begin{array} { c } { { ( I - \beta S ) P ^ { ( \infty ) } = ( 1 - \beta ) } } \\ { { P ^ { ( \infty ) } = ( 1 - \beta ) \bigl ( I - \beta S \bigr ) ^ { - 1 } . } } \end{array}\tag{16}
$$

□

Proposition 4. Let $R _ { T }$ be the truncation residual obtained after retaining the first T terms in $P ^ { ( \infty ) }$ . The $L _ { 1 }$ norm of $R _ { T }$ decays exponentially as the truncation depth T increases.

$$
R _ { T } = ( 1 - \beta ) \left( \sum _ { k = T } ^ { \infty } \beta ^ { k } S ^ { k } \right) .\tag{17}
$$

Proof. We use the entrywise matrix $L _ { 1 }$ norm, defined as $\begin{array} { r } { \| A \| _ { 1 } = \sum _ { i , j } | A _ { i j } | } \end{array}$ . Since S is stochastic, we have that $S ^ { k }$ is also stochastic, hence $\| S ^ { k } \| _ { 1 } = n .$ . Therefore,

$$
\begin{array} { l } { \displaystyle \| R _ { T } \| _ { 1 } = ( 1 - \beta ) \sum _ { k = T } ^ { \infty } \beta ^ { k } \| S ^ { k } \| _ { 1 } } \\ { \displaystyle = n ( 1 - \beta ) \sum _ { k = T } ^ { \infty } \beta ^ { k } } \\ { \displaystyle = n ( 1 - \beta ) \left( \sum _ { k = 0 } ^ { \infty } \beta ^ { k } - \sum _ { k = 0 } ^ { T - 1 } \beta ^ { k } \right) } \\ { \displaystyle = n ( 1 - \beta ) \left( \frac { 1 } { 1 - \beta } - \frac { 1 - \beta ^ { T } } { 1 - \beta } \right) } \\ { \displaystyle = n \beta ^ { T } . } \end{array}\tag{18}
$$

For $\beta < 1 , \| R _ { T } \| _ { 1 }$ therefore decays exponentially to zero as the truncation length T increases. □

## 7. Additional Ablation Study

Unless stated otherwise, all ablation results are reported on the mini-ImageNet dataset [35] (T = 5) for computational efficiency.

Impact of kernel functions. In Table 8, we ablate the feature-map non-linearity used in linear-attention. ReLU achieves the best performance (76.3% Top-1), slightly outperforming ELU (76.0%). We therefore use ReLU by default in all experiments.

Table 8. Impact of different kernel functions (DeiT-T).
<table><tr><td>Kernel Function</td><td>ELU</td><td>ReLU</td></tr><tr><td>Accuracy (%)</td><td>76.0</td><td>76.3</td></tr></table>

Ablation for local window size. In Table 9 we vary the window size in local attention on DeiT-T. A compact $3 \times 3$ window achieves the best Top-1 (76.3%) while being less computationally expensive, whereas larger windows slightly reduce accuracy (75.7% at $5 \times 5 ,$ , 76.1% at $7 \times 7 )$ and increase cost.

Table 9. Ablation study on local window size using DeiT-T.
<table><tr><td>Local Window Size</td><td> $3 \times 3$  </td><td> $5 \times 5$ </td><td> $7 \times 7$ </td></tr><tr><td>Accuracy (%)</td><td>76.3</td><td>75.7</td><td>76.1</td></tr></table>

## 8. Learnable Step Weights Details

The step weights in Eq. (10) are predicted per token and per head, allowing each token to adaptively weight the retained diffusion terms through a lightweight gate applied to its query and key:

$$
{ \beta } _ { i } ^ { ( j ) } = \mathrm { s o f t m a x } \Big ( W _ { 2 } \mathrm { R e L U } \Big ( W _ { 1 } \Big [ { q } _ { i } ^ { ( j ) } ; k _ { i } ^ { ( j ) } \Big ] \Big ) \Big ) \in \mathbb { R } ^ { T } ,\tag{19}
$$

where $W _ { 1 } ~ \in ~ \mathbb { R } ^ { d _ { h } \times 2 d _ { h } }$ and $W _ { 2 } ~ \in ~ \mathbb { R } ^ { T \times d _ { h } }$ . The softmax guarantees $\beta _ { k } \geq 0$ and $\textstyle \sum _ { k = 0 } ^ { T - 1 } \beta _ { k } \ = \ 1$ , so the output is a convex combination of the truncation length T. The gate parameters are shared across heads and instantiated independently in each layer, while the weights are predicted separately for each token and head. The gate adds $\mathcal { O } ( n d d _ { h } + n T d )$ operations, preserving linear complexity in the number of tokens and leaving the asymptotic complexity of Algorithm 1 unchanged.

## 9. Failure Cases

Excessive higher-order diffusion can be detrimental due to over-smoothing: since S is row-stochastic, deep walks

converge toward a query-independent stationary distribution, making token representations increasingly similar and weakening discriminative local structure.

## 10. Effective Regimes

LiSA provides the largest gains precisely where long-range modeling is hardest: plain columnar backbones with weak first-order cores (+3.43 on PVT-T; +7.4 when added to linear angular attention), high-resolution inputs (+2.64 at $3 8 4 ^ { 2 } )$ , and dense prediction tasks that require spatial reasoning (+4.28 mIoU on ADE20K, +3.8 box AP on COCO). Gains remain consistent but naturally smaller for backbones that already encode hierarchy and locality (e.g. +0.95 on Swin-T), where part of the long-range signal is available by design. Its additional cost is therefore least justified at short sequence lengths, where the $\mathcal { O } ( T d _ { h } ^ { 3 } )$ term is not amortized.

## 11. Model Architecture

We summarize the architectures of the three Transformer models used in the main paper, DeiT, PVT, and Swin Transformer, in Table 10, 11, and 12. In all cases, we replace each original self-attention block with our proposed LiSA block.

Table 10. Architectures of LiSA-DeiT models. All original DeiT transformer blocks are removed (#DeiT blocks = 0) and replaced with LiSA blocks.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>Resolution</td><td rowspan=1 colspan=1>LiSA blocks(res, dim, heads, #layers)</td><td rowspan=1 colspan=1>#DeiT blocks</td></tr><tr><td rowspan=1 colspan=1>LiSA-DeiT-T</td><td rowspan=1 colspan=1>Encoder</td><td rowspan=1 colspan=1> $1 4 \times 1 4$ 一</td><td rowspan=1 colspan=1> $( 1 4 \times 1 4 , 1 9 2 , 3 ) \times 1 2$ 1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>LiSA-DeiT-S</td><td rowspan=1 colspan=1>Encoder</td><td rowspan=1 colspan=1> $1 4 \times 1 4$ </td><td rowspan=1 colspan=1> $( 1 4 \times 1 4 , 3 8 4 , 6 ) \times 1 2$ </td><td rowspan=1 colspan=1>0</td></tr></table>

Table 11. Architectures of LiSA-PVT models. At all stages, the original PVT transformer blocks are removed (#PVT blocks = 0) and replaced with LiSA blocks.
<table><tr><td>Model</td><td>Stage</td><td>Output</td><td>LiSA blocks (res, dim, heads, #layers)</td><td>#PVT blocks</td></tr><tr><td rowspan="4">LiSA-PVT-T</td><td>res1</td><td> $5 6 \times 5 6$ </td><td> $( 5 6 \times 5 6 , 6 4 , 2 ) \times 2$ </td><td>0</td></tr><tr><td>res2</td><td> $2 8 \times 2 8$ </td><td> $( 2 8 \times 2 8 , 1 2 8 , 4 ) \times 2$ </td><td>0</td></tr><tr><td>res3</td><td> $1 4 \times 1 4$ </td><td> $( 1 4 \times 1 4 , 3 2 0 , 1 0 ) \times 2$ </td><td>0</td></tr><tr><td>res4</td><td> $7 \times 7$ </td><td> $( 7 \times 7 , 5 1 2 , 1 6 ) \times 2$ </td><td>0</td></tr><tr><td rowspan="4">LiSA-PVT-S</td><td>res1</td><td> $5 6 \times 5 6$ </td><td> $( 5 6 \times 5 6 , 6 4 , 2 ) \times 3$ </td><td>0</td></tr><tr><td>res2</td><td> $2 8 \times 2 8$ </td><td> $( 2 8 \times 2 8 , 1 2 8 , 4 ) \times 4$ </td><td>0</td></tr><tr><td>res3</td><td> $1 4 \times 1 4$ </td><td> $( 1 4 \times 1 4 , 3 2 0 , 1 0 ) \times 6$ </td><td>0</td></tr><tr><td>res4</td><td> $7 \times 7$ </td><td> $( 7 \times 7 , 5 1 2 , 1 6 ) \times 3$ </td><td>0</td></tr></table>

## 12. Detailed Efficiency Measurements

To complement the efficiency plots in the main paper, we additionally report the exact latency, throughput, and peak

Table 12. Architecture of the LiSA-Swin model. At all stages, the original Swin transformer blocks are removed (#Swin blocks = 0) and replaced with LiSA blocks.
<table><tr><td rowspan="2">Stage</td><td rowspan="2">Output</td><td colspan="2">LiSA-Swin-T</td></tr><tr><td>LiSA blocks (win, dim, heads, #layers)</td><td>#Swin blocks</td></tr><tr><td>res1</td><td> $5 6 \times 5 6$ </td><td> $( 7 \times 7 , 9 6 , 3 ) \times 2$ </td><td>0</td></tr><tr><td>res2</td><td> $2 8 \times 2 8$ </td><td> $( 7 \times 7 , 1 9 2 , 6 ) \times 2$ </td><td>0</td></tr><tr><td>res3</td><td> $1 4 \times 1 4$ </td><td> $( 7 \times 7 , 3 8 4 , 1 2 ) \times 6$ </td><td>0</td></tr><tr><td>res4</td><td> $7 \times 7$ </td><td> $( 7 \times 7 , 7 6 8 , 2 4 ) \times 2$ </td><td>0</td></tr></table>

GPU memory values in Table 13. The measurements span input resolutions from $2 2 4 ^ { 2 }$ to $2 0 4 8 ^ { 2 }$ , corresponding to sequence lengths from 197 to 16, 385 tokens, and compare DeiT-T, FLatten, and LiSA under the same setting. Consistent with Fig. 5, DeiT-T exhibits rapidly increasing latency and memory as the sequence length grows, whereas FLatten and LiSA scale much more favorably. LiSA incurs additional latency over FLatten due to its higher-order propagation and adaptive local branch, while retaining substantially lower latency and memory than softmax attention at long sequences.

Table 13. Latency, throughput, and peak GPU memory across input resolutions.
<table><tr><td colspan="2">Input</td><td colspan="3">DeiT-T</td><td colspan="3">Flatten</td><td colspan="3">LiSA</td></tr><tr><td></td><td>Res. Seq.</td><td>Latency (ms)</td><td>Throughput GPU Memory (img/s)</td><td>(MB)</td><td>Latency (ms)</td><td>Throughput GPU Memory (img/s)</td><td>(MB)</td><td>|Latency (ms)</td><td>Throughput GPU Memory (img/s)</td><td>(MB)</td></tr><tr><td>224</td><td>197</td><td>3.50</td><td>2288.00</td><td>74.0</td><td>7.15</td><td>1118.91</td><td>73.2</td><td>13.53</td><td>591.49</td><td>95.4</td></tr><tr><td>384</td><td>577</td><td>4.07</td><td>1967.60</td><td>152.7</td><td>7.34</td><td>1090.32</td><td>110.2</td><td>13.52</td><td>591.60</td><td>172.1</td></tr><tr><td>512</td><td>1025</td><td>8.06</td><td>993.06</td><td>312.9</td><td>7.40</td><td>1080.62</td><td>151.1</td><td>13.48</td><td>593.68</td><td>263.4</td></tr><tr><td>640</td><td>1601</td><td>15.56</td><td>514.19</td><td>627.9</td><td>11.11</td><td>720.42</td><td>206.6</td><td>19.47</td><td>410.90</td><td>382.0</td></tr><tr><td>768</td><td>2305</td><td>27.69</td><td>288.93</td><td>1178.7</td><td>14.74</td><td>542.73</td><td>275.7</td><td>26.15</td><td>305.90</td><td>529.2</td></tr><tr><td>896</td><td>3137</td><td>43.80</td><td>182.66</td><td>2061.4</td><td>19.96</td><td>400.87</td><td>352.4</td><td>34.86</td><td>229.49</td><td>697.4</td></tr><tr><td>1024</td><td>4097</td><td>66.51</td><td>120.29</td><td>3394.2</td><td>25.59</td><td>312.58</td><td>441.9</td><td>44.89</td><td>178.21</td><td>890.9</td></tr><tr><td>1152</td><td>5185</td><td>98.78</td><td>80.99</td><td>5317.0</td><td>31.79</td><td>251.65</td><td>546.2</td><td>55.58</td><td>143.95</td><td>1115.9</td></tr><tr><td>1280</td><td>6401</td><td>146.13</td><td>54.74</td><td>7976.9</td><td>39.05</td><td>204.88</td><td>663.9</td><td>67.58</td><td>118.38</td><td>1367.2</td></tr><tr><td>1408</td><td>7745</td><td>205.52</td><td>38.93</td><td>11545.5</td><td>46.43</td><td>172.29</td><td>791.3</td><td>80.73</td><td>99.10</td><td>1642.5</td></tr><tr><td>1536</td><td>9217</td><td>279.54</td><td>28.62</td><td>16210.5</td><td>54.77</td><td>146.07</td><td>926.4</td><td>95.24</td><td>83.99</td><td>1937.2</td></tr><tr><td>179212545</td><td></td><td>502.82</td><td>15.91</td><td>29690.0</td><td>71.92</td><td>111.24</td><td>1244.8</td><td>126.33</td><td>63.33</td><td>2623.8</td></tr><tr><td>204816385</td><td></td><td>898.92</td><td>8.90</td><td>50281.7</td><td>91.84</td><td>87.11</td><td>1605.2</td><td>161.71</td><td>49.47</td><td>3409.1</td></tr></table>

## References

[1] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, and Judy Hoffman. Hydra attention: Efficient attention with many heads. arXiv preprint arXiv:2209.07484, 2022. 1, 2, 7

[2] Han Cai, Chuang Gan, and Song Han. Efficientvit: Enhanced linear attention for high-resolution low-computation visual recognition. arXiv preprint arXiv:2205.14756, 2022. 1, 7

[3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 213–229. Springer, 2020. 1

[4] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Perpixel classification is not all you need for semantic segmentation. Advances in Neural Information Processing Systems, 34:17864–17875, 2021. 1

[5] Jeongwhan Choi, Hyowon Wi, Jayoung Kim, Yehjin Shin, Kookjin Lee, Nathaniel Trask, and Noseong Park. Graph convolutions enrich the self-attention in transformers! Advances in Neural Information Processing Systems, 37: 52891–52936, 2024. 2

[6] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. In International Conference on Learning Representations, 2021. 2

[7] MMSegmentation Contributors. Mmsegmentation, an open source semantic segmentation toolbox, 2020. 6

[8] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition workshops, pages 702–703, 2020. 6

[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,

and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on Computer Vision and Pattern Recognition, pages 248–255. Ieee, 2009. 5

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 1, 2

[11] Yotam Erel, Olaf Dunkel, Rishabh Dabral, Vladislav¨ Golyanik, Christian Theobalt, and Amit Bermano. Attention (as discrete-time markov) chains. Advances in Neural Information Processing Systems, 38:54658–54690, 2026. 2

[12] Qihang Fan, Huaibo Huang, and Ran He. Breaking the lowrank dilemma of linear attention. In 2025 IEEE/CVF Confer ence on Computer Vision and Pattern Recognition (CVPR), pages 25271–25280. IEEE, 2025. 2

[13] Aosong Feng, Irene Li, Yuang Jiang, and Rex Ying. Diffuser: efficient transformers with multi-hop attention diffusion for long sequences. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12772–12780, 2023. 2

[14] Dongchen Han, Xuran Pan, Yizeng Han, Shiji Song, and Gao Huang. Flatten transformer: Vision transformer using focused linear attention. In ICCV, 2023. 1, 2, 6, 7

[15] Dongchen Han, Yifan Pu, Zhuofan Xia, Yizeng Han, Xuran Pan, Xiu Li, Jiwen Lu, Shiji Song, and Gao Huang. Bridging the divide: Reconsidering softmax and linear attention. In NeurIPS, 2024. 2

[16] Dongchen Han, Ziyi Wang, Zhuofan Xia, Yizeng Han, Yifan Pu, Chunjiang Ge, Jun Song, Shiji Song, Bo Zheng, and Gao Huang. Demystify mamba in vision: A linear attention perspective. In NeurIPS, 2024. 2

[17] Dongchen Han, Tianzhu Ye, Yizeng Han, Zhuofan Xia, Shiji Song, and Gao Huang. Agent attention: On the integration of softmax and linear attention. In ECCV, 2024. 8

[18] Ali Hassani, Steven Walton, Jiachen Li, Shen Li, and Humphrey Shi. Neighborhood attention transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2

[19] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Gir-´ shick. Mask r-cnn. In International Conference on Computer Vision, 2017. 7

[20] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, pages 5156–5165. PMLR, 2020. 1, 2, 3, 7

[21] Alexander Kirillov, Ross Girshick, Kaiming He, and Piotr Dollar. Panoptic feature pyramid networks. In ´ CVPR, 2019. 6

[22] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part IX, pages 280–296. Springer, 2022. 1, 2

[23] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence´ Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, 2014. 5, 6

[24] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding, 2017. 1

[25] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021. 1, 2, 5, 6

[26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2018. 5

[27] Jiachen Lu, Jinghan Yao, Junge Zhang, Xiatian Zhu, Hang Xu, Weiguo Gao, Chunjing Xu, Tao Xiang, and Li Zhang. Soft: softmax-free transformer with linear complexity. Advances in Neural Information Processing Systems, 34: 21297–21309, 2021. 2

[28] W Meng, Y Luo, X Li, D Jiang, and Z Zhang. Polaformer: Polarity-aware linear attention for vision transformers. arxiv 2025. arXiv preprint arXiv:2501.15061. 2

[29] Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, and Yiran Zhong. cosformer: Rethinking softmax in attention. In International Conference on Learning Representations, 2022. 1, 2

[30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 1

[31] Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, and Hongsheng Li. Efficient attention: Attention with linear complexities. In WACV, 2021. 1, 2, 7

[32] Shitao Tang, Jiahui Zhang, Siyu Zhu, and Ping Tan. Quadtree attention for vision transformers. In International Conference on Learning Representations, 2022. 1

[33] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve J´ egou. Training´

data-efficient image transformers & distillation through attention. In International Conference on Machine Learning, pages 10347–10357. PMLR, 2021. 1, 5, 6, 7

[34] Ashish Vaswani, Noam Shazeer, Niki Parmar, et al. Atten tion is all you need. In NeurIPS, 2017. 1

[35] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. Advances in neural information processing systems, 29, 2016. 7, 1

[36] Guangtao Wang, Rex Ying, Jing Huang, and Jure Leskovec. Multi-hop attention graph neural network. arXiv preprint arXiv:2009.14332, 2020. 2

[37] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity, 2020. 1

[38] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 568–578, 2021. 1, 2, 5, 6

[39] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pvtv2: Improved baselines with pyramid vision transformer. Computational Visual Media, 8(3):1–10, 2022. 2

[40] Hyowon Wi, Jeongwhan Choi, and Noseong Park. Learning advanced self-attention for linear transformers in the singular value domain. arXiv preprint arXiv:2505.08516, 2025. 2

[41] Zhuofan Xia, Xuran Pan, Shiji Song, Li Erran Li, and Gao Huang. Vision transformer with deformable attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4794–4803, 2022. 1

[42] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In ECCV, 2018. 6

[43] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in Neural Information Processing Systems, 34:12077–12090, 2021. 1

[44] Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas Singh. Nystromformer: A nystr ¨ om-based algorithm for approximat-¨ ing self-attention. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 14138–14148, 2021. 2

[45] Haoran You, Yunyang Xiong, Xiaoliang Dai, Bichen Wu, Peizhao Zhang, Haoqi Fan, Peter Vajda, and Yingyan Lin. Castling-vit: Compressing self-attention via switching towards linear-angular attention during vision transformer inference. In CVPR, 2023. 1, 2, 7, 8

[46] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In Proceedings of the IEEE/CVF international conference on computer vision, pages 558–567, 2021. 2

[47] Li Yuan, Qibin Hou, Zihang Jiang, Jiashi Feng, and Shuicheng Yan. Volo: Vision outlooker for visual recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. 2

[48] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6023–6032, 2019. 6

[49] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017. 6

[50] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 13001–13008, 2020. 6

[51] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. In International Journal ofComputer Vision. Springer, 2019. 5, 6

[52] Lei Zhu, Xinjiang Wang, Zhanghan Ke, Wayne Zhang, and Rynson Lau. Biformer: Vision transformer with bi-level routing attention. In CVPR, 2023. 2, 5, 6