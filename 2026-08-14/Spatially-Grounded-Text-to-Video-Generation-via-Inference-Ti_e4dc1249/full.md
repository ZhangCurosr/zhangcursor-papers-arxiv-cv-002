# Spatially-Grounded Text-to-Video Generation via Inference-Time Gradient-Free Optimization

Guillaume Jeanneret<sup>1</sup> , Mathis Koroglu<sup>1,2</sup> , Hugo Caselles-Dupré<sup>2</sup> , Arnaud Dapogny<sup>1</sup>, and Matthieu Cord<sup>1,3</sup>

<sup>1</sup> ISIR - Sorbonne Université, France <sup>2</sup> Obvious Research, France <sup>3</sup> Valeo.ai, France

Abstract. Difusion Transformer Text-to-Video models have achieved remarkable synthesis quality, yet fine-grained spatial controllability remains a significant challenge. While existing training-free methods produce solid overall results in spatially grounded generation, i.e., placing a specific object in a designated location, they rely on gradient-based optimization techniques that incur prohibitive computational overhead, a bottleneck amplified in modern large-scale architectures. To address this limitation, we present Gradient-free Analytical Trajectory Optimization Video Generation (GATO-Vid), a novel training-free and gradient-free approach for precise spatial guidance. Rather than relying on costly backward passes, we introduce an alternative cross-attention score and solve it analytically to obtain an exact, closed-form solution. To use our analytical solution, we propose an on-the-fly injection mechanism tailored to the topological manifold of the transformer’s latent space. Our experiments demonstrate that GATO-Vid significantly outperforms existing baselines in localization accuracy while introducing minimal computational overhead.

Keywords: Gradient- and training-free controllable text-to-video · Spatially grounded video generation · Analytical Trajectory Optimization

## 1 Introduction

Text-to-Video (T2V) difusion models have made significant progress, producing high-quality videos that are coherent, realistic, and well-aligned with textual prompts. These improvements have been driven by advances in Difusion Transformer (DiT) architectures [30], large-scale video-text training data, and the scaling of generative models to billions of parameters [39, 51]. However, text remains a limited interface for controlling video generation. A prompt can describe the presence of objects, their appearance, and their general motion, but cannot reliably specify spatio-temporal localization.

To go beyond this controllability limitation, we address the task of spatially localized text-to-video generation. Given a text prompt and spatial constraints, such as bounding boxes, the goal of this a task is to generate a video that both faithfully follows the prompt and respects the prescribed object locations. In the literature, existing approaches can be divided into training-based and training-free methods. Training-based approaches learn spatial control from annotated data [15], either by training dedicated modules or by fine-tuning existing generative models. However, this comes at a significant cost, requiring substantial computational resources and large-scale annotated data given the scale of recent text-to-video models. For this reason, training-free methods are attractive alternatives as they only control pretrained generators at inference time, without modifying their weights or collecting new data. Despite that, current training-free methods also face limitations. The most efective approaches often rely on gradient-based guidance [52], which requires backpropagating through the generator during sampling. This is particularly costly when applied to modern DiT-based T2V models.

To add objects in specific locations in videos, most methods [2, 8, 11, 23, 25, 31, 46], across both image and video generation domains, leverage the observation that these cross-attention layers are responsible for the objects’ localization [34]. Thus, these strategies optimize for this similarity with gradients. However, we noted two primary drawbacks in computing gradients: (i) computing the mean cross-attention map explicitly is memory-intensive. In fact, modern neural network engines often use FlashAttention [12] to avoid this explicit calculation; (ii) computing the gradient with respect to the input requires backpropagation through the entire network (billions of parameters) with very large spatio-temporal feature maps, necessitating VRAM-reducing techniques such as gradient checkpointing, multi-GPU sharding [54], or other engineering complexities, thereby significantly increasing inference time. For instance, in enormous architectures like Wan2.2 [39], computing the backward pass requires more than a single 80GB GPU. In contrast, users with consumer-grade GPUs, which have less than 32GB available VRAM, would not be able to utilize these methods. This motivates our setup, where we constrain T2V grounded generation to operate only in a training- and gradient-free environment. While existing gradient-free alternatives [21, 38, 48] avoid this overhead, they usually ofer weaker or less reliable spatial control, especially in DiT architectures.

In this paper, we connect gradient-based methods with cross-attention manipulation techniques, proposing a training-free and gradient-free strategy that simulates the guiding behavior of their gradient-based counterparts. Our method, dubbed Gradient-free Analytical Trajectory Optimization Video Generation, in short GATO-Vid, aims to optimize the global cross-attention maps between the target words and the spatio-temporal masks. To perform this guidance on-the-fly, we first analyze the standard cross-attention control loss and propose an eficient surrogate score. We then derive the analytical optimum of this objective, which provides explicit directions for increasing the alignment between target words and target regions. Finally, we introduce an injection mechanism that applies these directions directly within the transformer blocks, while respecting the geometry induced by query normalization.

Our contributions are as follows:

– We introduce GATO-Vid, a training-free and gradient-free framework for spatially grounded text-to-video generation that avoids backpropagation during inference in modern DiT-based video generators.

– We reformulate cross-attention localization as an analytically solvable surrogate objective, deriving a closed-form solution that provides optimal directions without explicitly computing attention maps or gradients.

– We propose a geometry-aware query injection mechanism that applies these analytical steering directions within transformer cross-attention modules while respecting the RMS normalization manifold, enabling stable and eficient inference-time control.

We demonstrate that GATO-Vid significantly improves quantitative localization metrics, whereas the tested baselines barely outperform the base model, highlighting the challenging nature of this task. To foster reproducibility of gradient-free and training-free approaches for grounded video generation, our code is available at https://gato-vid.github.io/.

## 2 Related Work

## 2.1 Text-to-Video Generation

Early Text-to-Video (T2V) generators [3, 7, 17] typically inflate existing text-toimage UNet models [33], such as Stable Difusion 1.5 [32], and incorporate temporal layers to transfer feature information across the time dimension. While effective, these early models remained constrained along the temporal axis, which made them capable of generating videos of a few frames. More recently, with the introduction of DiTs [30], large-scale datasets [26, 43], and massive training schemes [14], DiT-based architectures have become the dominant paradigm. State-of-the-art models, including CogVideoX [51], LTX-Video [18], Hunyuan-Video 1.5 [37], and Wan2.2 [39], leverage these architectures to generate highquality, realistic videos from textual prompts. Nevertheless, their controllability remains restricted, as users are limited to guiding the generation using pure text.

Beyond text-only prompting, controllable video generation has also been studied through additional conditioning signals. A first line of work extends structural control to video, using frame-wise signals such as edge and depth maps or combining text, sketches, reference videos, and motion vectors to guide both spatial layout and temporal dynamics [10, 42]. A second line focuses on motion-centric control, where users specify camera poses, camera movements, or object trajectories instead of relying solely on textual descriptions [16,19,44,50]. Rather than introducing new trainable layers, we achieve spatially grounded video generation by directly manipulating the T2V generator in a gradient-free manner.

## 2.2 Spatially-Grounded Text-to-Image Control

Instead of controlling the generation with mere text, users might want to generate content with objects in specific locations within a scene. In the text-to-image (T2I) domain, trainable approaches append modules on top of a frozen generator; for instance, GLIGEN [22], IFAdapters [45], SpaText [1], and BlobGEN [27] train new cross-attention layers to inject object descriptions into target locations. Conversely, training-free methods eliminate the need for parameter updates. Most methods rely on gradient-based optimization [2, 8, 9, 11, 25, 31, 46, 47, 49], maximizing the similarity between cross-attention maps and target object locations via backpropagation. This arises from the observation that these cross-attention layers are responsible for object localization [34]. While these methods do not require as much compute as video generators, backpropagating gradients through the network layers still incurs a notable computational overhead. Finally, the gradient-free literature is less extensive and typically can be categorized into two strategies: some methods [6, 28, 41] directly modify cross-attention layers to inject object segments or descriptions into their corresponding regions, while others [4, 35] utilize multi-inference pipelines to generate separate object estimates and subsequently merge them.

## 2.3 Spatially-Grounded Text-to-Video Control

Compared to the T2I domain, spatially grounded video generation remains less thoroughly explored. Mirroring text-to-image trends, existing approaches can be classified into the same three categories: training-based methods, gradient-based training-free strategies, and training- and gradient-free pipelines. Training-based strategies add new trainable modules on top of existing T2V models [13, 15]. Among gradient-based training-free methods, Lian et al. [23] optimize crossattention maps between target words and input bounding boxes via backpropagation within the ModelScope [40] framework. Within the gradient-free literature, attention manipulation is the dominant approach. For instance, Peekaboo [21] applies hard masking in both self- and cross-attention layers: in crossattention, object text tokens attend strictly to their designated regions while the background attends elsewhere, whereas in self-attention, background and object video tokens are mutually isolated. Direct-a-Video [50] follows a similar trend by constraining cross-attention maps within the ModelScope [40] architecture. VideoTetris [38] operates akin to Regional Prompting [6] by inserting distinct object descriptions solely into their respective spatial regions. These previous methods have only been applied to 3D-UNet T2V models and have not been translated to modern DiT architectures. Closest to our work, SwitchCraft [48] tackles multi-event generation in Wan2.1 [39] by projecting source queries into a target subspace to enhance similarity before reintegrating them. Unlike these methods, our approach is built on a theoretical foundation that optimizes a surrogate score on the fly in a completely gradient-free manner, while explicitly preserving the T2V model’s geometric properties.

## 3 Methodology

In this section, we first define the problem and review standard gradient-based approaches. Next, we introduce our proposed surrogate objective and derive

its analytical closed-form solution. Lastly, we explain how to incorporate this solution into each block of the DiT while maintaining its geometric properties.

## 3.1 Preliminaries

We begin by formally defining the main framework. Let $f _ { \theta }$ be the T2V latent flow matching generator [24, 32], and $\tau = \{ e _ { 1 } , \ldots , e _ { | \tau | } , e _ { j } \in \mathbb { R } ^ { d } \}$ , be the input prompt tokenized and processed by the text encoder, consisting of $| \tau |$ vectors of dimension $d ,$ where d is the hidden channel dimension of $f _ { \theta }$ . Let $\boldsymbol { x } _ { t } \in \dot { \mathbb { R } } ^ { \mathrm { C } \times \mathrm { T } \times \mathrm { H } \times \mathrm { W } }$ be the input spatio-temporal video latent at time $t ,$ where C, T, H, and W are the channel, time, height, and width dimensions, respectively. To synthesize a video using $f _ { \theta }$ , we iteratively refine $x _ { t }$ according to the following equations

$$
\begin{array} { c } { v _ { t } = f _ { \theta } ( x _ { t } , \tau , t ) } \\ { x _ { t + \Delta t } = \mathrm { F } 1 \circ \mathrm { w } \mathbf { S } \mathbf { c h } ( x _ { t } , v _ { t } , t ) . } \end{array}\tag{1}
$$

Here, we initialize with Gaussian noise $x _ { 0 } \sim \mathcal { N } ( 0 , 1 )$ , ∆t is the step size of the iterative process, and FlowSch is the flow matching scheduler. At the final timestep, $i . e . , t = 1$ , the latent $x _ { 1 }$ is decoded into pixel space using a pretrained variational autoencoder.

The training-free, gradient-free, spatially grounded T2V generation task aims to generate a video in which the object described in the prompt follows a given trajectory. Crucially, this control must be exerted during inference without altering the parameters of $f _ { \theta }$ or computing gradients. To spatially ground the generative model, let M denote the spatio-temporal indices tracking the target object’s trajectory within the video volume. Finally, let $T$ be the set of textual indices pointing to the object in the tokenized prompt, τ .

Difusion transformers leverages native spatio-temporal transformer blocks [30] to process the video data [39, 51]. Particularly, these architectures first concatenate text and video tokens and then process them through a sequence of blocks consisting of self-attention layers, followed by a feed-forward network. As noted by Shen et al. [34], a sub-operation within self-attention is the cross-attention function, where text data is injected into video tokens. Thus, regardless of the architecture, whether it is a UNet [17] or a DiT [30], the fundamental component for grounding visual features with textual information is the cross-attention mechanism:

$$
\begin{array} { l } { { A ( Q _ { i } , K _ { i } ) = \mathsf { s o f t m a x } \left( \displaystyle \frac { Q _ { i } K _ { i } ^ { \top } } { \sqrt { d } } \right) } } \\ { { \mathsf { c r o s s - a t t n } ( Q _ { i } , K _ { i } , V _ { i } ) = A ( Q _ { i } , K _ { i } ) V _ { i } , } } \end{array}\tag{2}
$$

where $Q _ { i } \in \mathbb { R } ^ { \mathrm { T H W } \times d } , K _ { i } \in \mathbb { R } ^ { | \tau | \times d } ;$ , and $V _ { i } \in \mathbb { R } ^ { | \tau | \times d }$ represent the query (video), key (text), and value (text) sequences, respectively, for the i-th block of the network. In modern DiT architectures, to generate the queries $Q _ { i }$ , the sequence features are mapped via a linear projection layer. Then, the RMS Normalization [53] (Eq. (3)) projects all individual vectors $q \in Q _ { i } , q \in \mathbb { R } ^ { d }$ , onto a hyper-ellipsoid surface:

$$
q  \sqrt { d } \frac { q \odot \gamma } { \lVert q \rVert } ,\tag{3}
$$

where $\odot$ denotes the element-wise multiplication and $\gamma \in \mathbb { R } ^ { d }$ is a learnable vector.

Gradient-based algorithms generate an object in a location M by first passing $x _ { t }$ and $\tau$ through $f _ { \theta }$ to extract the vision queries and text keys:

$$
( v _ { t } , \{ Q _ { i } , K _ { i } \} _ { i = 1 } ^ { B } ) = f _ { \theta } ( x _ { t } , \tau , t ) ,\tag{4}
$$

where $v _ { t }$ is the model output, and $B$ is the total number of blocks in $f _ { \theta }$ . Second, they compute the mean attention map over the B blocks:

$$
\bar { A } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } A ( Q _ { i } , K _ { i } ) .\tag{5}
$$

Third, they compute the gradients of a loss function, ℓ, between the mean attention ${ \bar { A } } .$ , the target words, $T$ , and the target location M with respect to the input $x _ { t }$ . This function ℓ is typically a Dice or $\ell _ { 2 }$ loss between A<sup>¯</sup>, which is averaged over each target text token, and M. Then, the latent $x _ { t }$ is updated with:

$$
x _ { t } ^ { \prime } = x _ { t } - \mu \nabla _ { x _ { t } } \ell ( \bar { A } , M , T ) ,\tag{6}
$$

where $\mu$ is the learning rate. Finally, the subsequent denoising step is computed using $ { \boldsymbol { { x } } } _ { t } ^ { \prime }$ instead of $x _ { t } ,$ or it is refined iteratively with more gradient steps. In this paper, our goal is to derive an eficient surrogate score that analytically captures an optimal direction for decreasing ℓ instead of optimizing this loss directly with gradients.

## 3.2 Building a Simpler Score

Our first contribution is an analytically solvable alternative score function to approximate the behavior of the loss function. Essential to the cross-attention mechanism is the softmax operation, a highly non-linear function that normalizes the cross-attention scores. To derive an exact analytical solution for the alternative loss function, we need a proxy for the softmax to circumvent the non-linearity. Formally, if $q _ { j } \in Q _ { i }$ and $k _ { l } \in K _ { i }$ are a specific query token and a key token, respectively, then, the cross-attention between $q _ { j }$ and $k _ { l }$ is:

$$
A ( q _ { j } , K _ { i } ) = \left[ \ldots , { \frac { \exp \left( q _ { j } \cdot k _ { l } / { \sqrt { d } } \right) } { \sum _ { k \in K _ { i } } \exp \left( q _ { j } \cdot k / { \sqrt { d } } \right) } } , \ldots \right] .\tag{7}
$$

To maximize the attention between the l-th text token $\left( k _ { l } \right)$ and the j-th query $( q _ { j } )$ , we can either maximize $q _ { j } \cdot k _ { l }$ or minimize $\begin{array} { r } { \sum _ { k \in { \cal K } , k \neq k _ { l } } \exp \left( q _ { j } \cdot k / \sqrt { d } \right) } \end{array}$ . Fortunately, to minimize this second term, we can directly work with each $q _ { j } \cdot k _ { l }$

individually as $\exp ( { q _ { j } \cdot k / \sqrt { d } } )$ is strictly increasing with respect to $q _ { j } \cdot k _ { l }$ . Thus, we can directly utilize the pre-softmax operation (i.e., the logits),

$$
A ^ { \prime } ( Q , K ) = Q K ^ { \top } ,\tag{8}
$$

and use it as a proxy for optimizing the attention. Consequently, we build our score function directly on top of $\operatorname { E q . } \left( 8 \right)$ . To increase attention in $M .$ , we maximize the following score:

$$
s = \underbrace { \sum _ { j \in M } \sum _ { l \in T } q _ { j } \cdot k _ { l } } _ { ( i ) } - \underbrace { \sum _ { j \in M } \sum _ { l \in T ^ { c } } q _ { j } \cdot k _ { l } } _ { ( i ) } - \underbrace { \sum _ { j \in M } \sum _ { l \in T ^ { c } } q _ { j } \cdot k _ { l } } _ { ( i i ) } - \underbrace { \sum _ { j \in M ^ { c } } \sum _ { l \in T } q _ { j } \cdot k _ { l } } _ { ( i i i ) } ,\tag{9}
$$

where $M ^ { c }$ and $T ^ { c }$ are the complement sets of indices of M and $T ,$ respectively. This score consists of three terms (from left to right): (i) a factor to increase the inner product, and thus the attention, between target regions and target words, (ii) a component reducing the influence of negative keys in target regions, and (iii) an element reducing the influence of positive keys outside target regions. In Fig. 1a we visually represent all regions (i), (ii), and (iii). Analogous to the original loss function $\ell ,$ maximizing s implicitly minimizes terms (ii) and (iii) while maximizing term (i), but operates completely in the logit space.

Since the double sum is separable, we can factorize Eq. (9) into:

$$
s = \mathcal { Q } ( M ) \cdot ( \boldsymbol { K } ( T ) - \boldsymbol { K } ( T ^ { c } ) ) - \mathcal { Q } ( M ^ { c } ) \cdot \boldsymbol { K } ( T ) ,\tag{10}
$$

where $\mathcal { Q } ( M ) \in \mathbb { R } ^ { d }$ and $\mathcal { K } ( T ) \in \mathbb { R } ^ { d }$ are defined as:

$$
\mathcal { Q } ( M ) = \frac { \sum _ { j \in M } q _ { j } } { | M | } , \quad K ( T ) = \frac { \sum _ { l \in T } k _ { l } } { | T | } .\tag{11}
$$

This factorization avoids the explicit attention map calculation, reducing the complexity from $\mathcal { O } ( | Q | \cdot | K | )$ to three vector dot products. In Fig. 1b, we provide a toy example illustrating how maximizing s acts as a proxy for minimizing a standard $\ell _ { 2 }$ localization loss - see Sec. 3.3 for more details.

## 3.3 On-the-fly Score Optimization

To avoid any gradient-based optimization, we first analytically maximize our proposed score in Eq. (10) by finding the vectors $\mathcal { Q } ( M )$ and $\mathcal { Q } ( M ^ { c } )$ that maximize s. Suppose $b ^ { + }$ and $b ^ { - }$ are unit vectors replacing $\mathcal { Q } ( M )$ and $\mathcal { Q } ( M ^ { c } )$ , respectively. The score s, which is now a function of $( b ^ { + } , b ^ { - } )$ , becomes:

$$
s ( b ^ { + } , b ^ { - } ) = b ^ { + } \cdot ( \mathcal { K } ( T ) - \mathcal { K } ( T ^ { c } ) ) - b ^ { - } \cdot \mathcal { K } ( T ) .\tag{12}
$$

Consequently, s is maximized when $b ^ { + }$ is collinear with $( \mathcal { K } ( T ) - \mathcal { K } ( T ^ { c } ) )$ and $b ^ { - }$ is parallel to $\left( - \kappa ( T ) \right)$ . Thus, we can maximize the attention map by simply

![](images/0e2c268fc1ebd30fdf78d99638dcbc43e80c5471913b3b2489902e29a88ff3ae.jpg)  
(a)

![](images/50cd2cd10a6c121f39e44f46b9d433918f7feba652ea2ed1c7dd8a36db0b7b83.jpg)  
(b)  
Fig. 1: (a) Target Regions: Cross-attention scores targeted with our score on Eq. (9). s searches to: (i) increase the logits of the target spatio-temporal regions and selected text tokens, (ii) decrease the logits of the target region and complementary text tokens, and (iii) reduce the logits of the regions outside the mask and the chosen text tokens. (b) Loss Minimization: Toy example exemplifying that our on-the-fly score optimization minimizes an $\ell _ { 2 }$ loss. The plot illustrates how increasing the $\lambda ^ { \pm }$ parameter gradually decreases the $\ell _ { 2 }$ error over diferent runs. The $^ { \mathrm { . 4 } } \mathrm { Q 2 5 - Q 7 5 ^ { \mathrm { 7 } } }$ shaded area represents the 25th–75th quantile range.

replacing these vectors (now referred to as biases) with the query sequence. Accordingly, we set $b ^ { + }$ and $b ^ { - }$ as

$$
b ^ { + } = \frac { \ K ( T ) - K ( T ^ { c } ) } { \| K ( T ) - K ( T ^ { c } ) \| } \quad \mathrm { a n d } \quad b ^ { - } = - \frac { K ( T ) } { \| K ( T ) \| } .\tag{13}
$$

Now, for each cross-attention block, we have an optimal direction vector. ${ \mathrm { Y e t } } ,$ directly replacing the queries $q _ { j }$ with the optimal biases $( q _ { j }  b ^ { \pm } )$ or naively adding the biases $( q _ { j }  b ^ { \pm } + q _ { j } )$ disrupts the network’s internal representations due to magnitude mismatches. Here, $b ^ { \pm }$ is either $b ^ { + }$ or $b ^ { - }$ . As the query vectors lie on the RMSNorm ellipsoid $\left( \mathrm { E q . \ ( 3 ) } \right)$ , we need to take into account this geometric property to remain closer to the distribution. Thus, we modify the queries as follows:

$$
q _ { j }  \{ \begin{array} { l l } { \frac { \sqrt { d } } { \lVert ( \lambda ^ { + } b ^ { + } \rVert q _ { j } \lVert + q _ { j } ) \oslash \gamma \rVert } ( \lambda ^ { + } b ^ { + } \lVert q _ { j } \rVert + q _ { j } ) } & { \mathrm { i f ~ } j \in M } \\ { \frac { \sqrt { d } } { \lVert ( \lambda ^ { - } b ^ { - } \rVert q _ { j } \lVert + q _ { j } ) \oslash \gamma \rVert } ( \lambda ^ { - } b ^ { - } \lVert q _ { j } \rVert + q _ { j } ) } & { \mathrm { i f ~ } j \in M ^ { c } } \end{array}  ,\tag{14}
$$

where $\oslash$ is the element-wise division. This renormalization consists of three main components: (i) We multiply $b ^ { \pm }$ by $\| q _ { j } \|$ to make the impact relative to $q _ { j }$ . (ii) We scale the bias with a hyperparameter $\lambda ^ { \pm }$ to control the impact of the bias $b ^ { \pm }$ Here, $\lambda ^ { \pm }$ plays a similar role to the learning rate $\mu$ in gradient-based approaches $\left( \mathrm { E q . ~ } \left( 6 \right) \right)$ . (iii) The constant $\frac { \surd d } { \lVert ( \lambda ^ { \pm } b ^ { \pm } \lVert q _ { j } \rVert + q _ { j } ) \oslash \gamma \rVert }$ projects the vector back to the ellipsoid surface of the RMS Normalization in Eq. (3).

For the background tokens $( j \in M ^ { c } )$ , adding the negative bias $b ^ { - }$ can distort the existing background features if the bias aligns too closely with the original query $q _ { j }$ . To prevent this distortion, we isolate the component of the bias that is strictly orthogonal to the query, ensuring the injection acts purely as a directional adjustment without overwriting the original feature content.

Formally, we define the orthogonal projection of a vector a onto the orthogonal complement of b as:

$$
P ( a , b ) = a - \left( a \cdot { \frac { b } { \| b \| } } \right) { \frac { b } { \| b \| } } .\tag{15}
$$

Using this projection, our final formulation for the query modification becomes:

$$
\begin{array} { r } { q _ { j } \gets \left\{ \begin{array} { l l } { \frac { \sqrt { d } } { \| ( \lambda ^ { + } b ^ { + } \| q _ { j } \| + q _ { j } ) \oslash \gamma \| } \left( \lambda ^ { + } b ^ { + } \| q _ { j } \| + q _ { j } \right) } & { \mathrm { i f ~ } j \in M , } \\ { \frac { \sqrt { d } } { \| ( \lambda ^ { - } P ( b ^ { - } \| q _ { j } \| , q _ { j } ) + q _ { j } ) Q \gamma \| } \left( \lambda ^ { - } P ( b ^ { - } \| q _ { j } \| , q _ { j } ) + q _ { j } \right) } & { \mathrm { i f ~ } j \in M ^ { c } } \end{array} . \right. } \end{array}\tag{16}
$$

As a final adjustment, we note that the cross-attention activations typically peak at the semantic center of an object and attenuate toward its boundaries. While we correctly insert the positive bias into each query, uniformly applying it yields an unnaturally homogeneous attention distribution across the target region. To circumvent this problem, we modulate the guidance scale spatially by fitting a normalized 2D Gaussian distribution to the bounding box of each frame. Subsequently, we multiply it by $\lambda ^ { + }$ . Hence, $\lambda ^ { + }$ varies spatially now than remaining fixed for every query inside the mask.

To validate empirically that maximizing our gradient-free surrogate score s with our injection mechanism in Eq. (14) efectively acts as a proxy for minimizing a standard $\ell _ { 2 }$ localization loss, we conduct a controlled toy experiment. Specifically, we instantiate a three-layer cross-attention model with randomly initialized weights and apply our directional query steering mechanism across a range of guidance intensities $\lambda ^ { \pm }$ . We conduct this process 100 times, storing the $( \lambda ^ { \pm } , \ell _ { 2 } , s )$ values and plotting both $( \lambda ^ { \pm } , s )$ and $( \lambda ^ { \pm } , \ell _ { 2 } )$ sequences. As illustrated in Fig. 1b, increasing $\lambda ^ { \pm }$ monotonically reduces the $\ell _ { 2 }$ loss between the resulting attention maps and the given target spatial layout M. Our analytical approximation yields highly localized attention maps, ofering an eficient trade-of by bypassing the heavy backpropagation during inference.

In a nutshell, to generate a spatially grounded video with GATO-Vid, we first sample the noisy latent variables with Gaussian noise $x _ { 0 } .$ . Next, for the first few iterations and the first blocks of the generative model, we extract the target biases $( b ^ { + } , b ^ { - } )$ and inject them to the queries following Eq. (14), summarized in Alg. 1. For the remaining iterations, we adhere to the original iterative process in Eq. (1).

## 4 Experimentation

## 4.1 Implementation Details

We implemented GATO-Vid on Wan2.2 [39], the best open-sourced T2V generative model. We used our injection mechanism in the first 20 blocks of the

Algorithm 1 Bias Injection: First, the positive and negative directions are computed using the selected text tokens. Then, the biases are injected back into the queries, which are then reprojected onto the RMSNorm ellipsoid. Finally, the newly modified tokens through the cross-attention.

Require: Generator $f _ { \theta } , x$ video features, t current step, prompt features $\tau \in \mathbb { R } ^ { | \tau | \times n }$   
target map M, selected words T, S<sub>B</sub> set of blocks where we inject the bias.   
$x \gets f$ .tokenize(x) ∈ R<sup>THW×n</sup>   
for b in f<sub>θ</sub>.blocks do   
$x  x + b . \mathbf { S } \mathbf { e } \mathbf { 1 } \mathbf { f } .$ -Attention(x)   
q, $k , v  b$ .norm\_q(b.to\_q(x)), b.norm\_k(b.to\_k(τ)), b.to\_v(τ)   
if $b \in S _ { B }$ then   
$b ^ { + } =$ Normalize mean(k[T]) − mean $\left( k [ T ^ { c } ] \right) )$   
$q ^ { \prime } \gets q [ M ]$ ▷ Get queries inside the mask   
$\bar { q } ^ { \prime }  \bar { q } ^ { \prime } + \bar { \lambda } ^ { + } \| q ^ { \prime } \| b ^ { + }$ ▷ Add positive bias   
$\begin{array} { r } { q ^ { \prime } \gets \frac { \sqrt { n } } { \lVert q ^ { \prime } \oslash b . \tt n o r m \tt _ { - } q . \gamma \rVert } q ^ { \prime } } \end{array}$ ▷ Normalization to hyper-ellipsoid   
$q [ M ]  q ^ { \prime }$   
$b ^ { - } =$ Normalize(mean $\left( k [ T ^ { c } ] \right) )$   
$q ^ { \prime } \gets q [ M ^ { c } ]$ ▷ Get queries outside the mask   
$q ^ { \prime }  q ^ { \prime } - \lambda ^ { + } \underline { { \| q ^ { \prime } \| } } b ^ { - }$ ▷ Add negative bias   
$\begin{array} { r } { q ^ { \prime } \gets \frac { \sqrt { n } } { \lVert q ^ { \prime } \oslash b . \tt n o r m \tt _ { - } q . \gamma \rVert } q ^ { \prime } } \end{array}$ ▷ Normalization to hyper-ellipsoid   
$q [ M ^ { c } ] \gets q ^ { \prime }$   
x $ x + b$ .cross-attention $( q , k , v )$   
x $ x + b . \mathrm { F F N } ( x )$   
return f.detokenize(x)

transformer architecture during the first 15% of the iterations. Furthermore, we set $\lambda ^ { \pm } = 1 . 5$ with a linear decay. As our baseline choices, we selected the most representative methods available in the literature for gradient-free and trainingfree grounded T2V generation. These are Peekaboo [21], VideoTetris [38], and SwitchCraft [48]. To mitigate the efect of using other architectures, we adapted them all to Wan2.2 for a fair comparison. Moreover, we performed a hyperparameter search in all baselines to adjust for the diferent backbone. Additionally, we evaluated Wan2.2 without grounding to establish a quality reference. All generated videos have 81 frames with a spatial resolution of 480×832 pixels using 30 steps flow matching steps. Finally, all experiments were performed on a single NVIDIA H100 (80GB) GPU.

## 4.2 Evaluation Setup

Dataset To benchmark GATO-Vid against concurrent approaches, we constructed two dedicated evaluation datasets. On the one hand, we leveraged Gemini [36] to generate 25 diverse prompts covering a variety of objects. For each prompt, we generated four ground-truth videos using Wan2.2 across diferent random seeds and visually verified the results. We then applied SAM 3 [5] to extract the corresponding reference bounding boxes for the target objects. During evaluation, we executed each prompt and bounding box configuration using four independent random seeds. In total, our benchmark comprises 400 generated videos. We name this set Set 1. On the other hand, we leveraged ChatGPT [29] to generate 100 distinct prompts, creating synthetic bounding boxes of various sizes, shapes, and movement patterns (linear, spiral, circular, Z/N-shaped movements) for each prompt. This set is inherently more challenging yet closer to our application, as the target locations conflict with the model’s typical movement patterns. Consistent with Set 1, we ran each prompt across four diferent seeds, yielding a total of 400 videos. We refer to this collection as Set 2.

Evaluation Metrics We evaluate our approach along two dimensions: spatial alignment and overall video quality. Spatial alignment metrics assess whether the target object was accurately generated within the specified bounding box while respecting its dimensions. To achieve this, we follow an evaluation protocol similar to that in [21]. Given a set of trajectory boxes, the textual description of the object, and the generated video, we extract object masks using SAM 3 [5] prompted with the target object’s prompt. Then, we compute two metrics to measure the similarity with the input bounding boxes: (i) the Intersection over Union (IoU) between the bounding box of the generated object and the target box (higher is better) and (ii) the Euclidean distance between the center of mass of the generated object and the center of the target bounding box, divided by the maximum possible distance, named Center Distance (CD); lower is better. Both CD and IoU are computed exclusively for videos with successful SAM 3 object detections. To complement this, we introduce a Success Rate (SR) metric to track the proportion of generated videos that contain the target object according to SAM 3. This metric is especially useful for identifying methods that fail to generate the target object altogether.

To evaluate overall video generation quality (e.g., temporal consistency, visual quality), we adopt VBench [20] metric suite. Specifically, we evaluate five core dimensions: Subject Consistency (SC), Background Consistency (BC), Dynamic Degree (DD), Aesthetic Quality (AQ), and Imaging Quality (IQ). SC and BC measure the temporal stability of objects and backgrounds across video frames, ensuring they do not drastically shift or warp. DD quantifies the amount of motion and dynamic variation within the video. Finally, AQ assesses the stylistic and compositional appeal of each frame, while IQ penalizes visual artifacts such as noise, blur, or distortions. For all metrics, higher values indicate better performance. Ideally, each method yields scores comparable to the vanilla model; a substantial change signals a noticeable shift from the base behavior.

Quantitative Result We report the quantitative comparison between GATO-Vid and various baselines in Tab. 1. Importantly, no baseline demonstrates robust localization capabilities. Peekaboo leads with a modest IoU on both datasets, while all other configurations perform comparably to the standard T2V model. In contrast, GATO-Vid outperforms all existing baselines by a large margin across all spatial alignment metrics, while being on par in the SR metric. Several key findings from these results merit closer inspection. First, we observe that directly adapting training-free, gradient-free strategies [21, 38] originally designed for 3D-UNet architectures to modern DiT models yields poor results. Specifically, VideoTetris [38] performs comparably to the vanilla T2V model, completely failing to ground the target objects in their specified locations. Conversely, Peekaboo [21] achieves the highest alignment scores among the baselines; however, it introduces visual artifacts, which we analyze qualitatively in Sec. 4.3. Second, SwitchCraft [48] - originally built for Wan2.1 - exhibits a similar limitation to VideoTetris, matching its performance to the unconstrained Wan2.2 baseline. We attribute this to SwitchCraft’s inability to suppress the influence of target text tokens outside the spatial mask; our ablation study (Sec. 4.4) shows that a negative component is necessary to improve alignment metrics. Third, the VBench evaluation indicates that all methods perform similarly across most dimensions. However, a decline in GATO-Vid’s DD and AQ scores suggests that our spatial constraints can lead to more static compositions. This stems from the background degradation and reduced scene dynamics of certain videos. Peekaboo exhibits a similar trade-of, albeit less severely. The diference in the DD metric to the T2V vanilla model suggests that Peekaboo creates excessive movement, and the decreased AQ indicates lower-quality videos. This final discussion pinpoints that better localization comes at the cost of quality. Finally, we underscore the inherent dificulty of precise training-free gradient-free spatial control.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="3">Localization</td><td colspan="5">VBench</td></tr><tr><td>CD (↓)</td><td>IoU (↑)</td><td>SR (↑)</td><td>SC (↑)</td><td>BC (↑)</td><td>DD (↑)</td><td>AQ (↑)</td><td>IQ (↑)</td></tr><tr><td rowspan="5">S1</td><td>|Wan2.2 T2V</td><td>0.138</td><td>0.154</td><td>89.3%</td><td>0.974</td><td>0.971</td><td>0.248</td><td>0.631</td><td>0.666</td></tr><tr><td>|VideoTetris*</td><td>0.132</td><td>0.160</td><td>85.7%</td><td>0.973</td><td>0.972</td><td>0.260</td><td>0.630</td><td>0.673</td></tr><tr><td>[SwitchCraft</td><td>0.138</td><td>0.152</td><td>88.7%</td><td>0.973</td><td>0.972</td><td>0.248</td><td>0.629</td><td>0.664</td></tr><tr><td>Peekaboo*</td><td>0.103</td><td>0.249</td><td>85.5%</td><td>0.970</td><td>0.966</td><td>0.310</td><td>0.597</td><td>0.677</td></tr><tr><td>|GATO-Vid (Ours)</td><td>0.059</td><td>0.363</td><td>89.7%</td><td>0.975</td><td>0.964</td><td>0.113</td><td>0.551</td><td>0.670</td></tr><tr><td rowspan="5">2 Set</td><td>|Wan2.2 T2V</td><td>0.198</td><td>0.124</td><td>74.2%</td><td>0.957</td><td>0.960</td><td>0.395</td><td>0.613</td><td>0.682</td></tr><tr><td>|VideoTetris*</td><td>0.195</td><td>0.136</td><td>69.7%</td><td>0.960</td><td>0.962</td><td>0.402</td><td>0.612</td><td>0.698</td></tr><tr><td>SwitchCraft</td><td>0.196</td><td>0.128</td><td>75.5%</td><td>0.956</td><td>0.960</td><td>0.393</td><td>0.611</td><td>0.678</td></tr><tr><td>Peekaboo*</td><td>0.193</td><td>0.171</td><td>49.2%</td><td>0.929</td><td>0.937</td><td>0.750</td><td>0.510</td><td>0.679</td></tr><tr><td>|GATO-Vid (Ours)</td><td>0.121</td><td>0.324</td><td>74.7%</td><td>0.952</td><td>0.951</td><td>0.360</td><td>0.557</td><td>0.690</td></tr></table>

Table 1: Quantitative Localization and Quality Metrics: GATO-Vid outperforms every baseline on the localization metrics with a trade-of. Its superior alignment comes at the cost of generation quality, as shown by the VBench metrics. All methods marked with <sup>∗</sup> were re-implemented on Wan2.2.

Next, we measure the computational eficiency of GATO-Vid by recording the average inference time across 50 runs. GATO-Vid introduces minimal runtime overhead of 0.4% compared to Vanilla Wan2.2. By comparison, VideoTetris adds 6.20%, SwitchCraft adds 32.80%, and Peekaboo adds 92.13%. However, since these methods (including ours) are only applied during the initial sampling steps, their impact on the total generation time is modest to negligible. In contrast, performing a single backpropagation step on one of Wan2.2’s two transformers increases inference time by 300.31% and requires an additional 26 GB of VRAM (totaling 59 GB) with only one transformer loaded in memory and without computing an attention map loss. When computing the explicit attention maps and the attention loss on an 80GB GPU, we received out-of-memory errors. Thus, these methods require CPU ofloading strategies, multiple GPUs, and gradient checkpointing. This makes gradient-based approaches unattractive for users with limited compute budgets.

Frame 1 Frame 81Frame 41  
Frame 1 Frame 41 Frame 81  
![](images/4c9be1b7181171896fe8449193fc9dfa924b0d154be10ed080be5d9c6f34d5ee.jpg)  
Fig. 2: Qualitative Comparison: From top to bottom, rows display results for GATO-Vid, Peekaboo, VideoTetris, SwitchCraft, and the vanilla T2V model, all sharing identical random seeds. Prompts are provided on top of each video, with the target object and bounding boxes highlighted in green.

## 4.3 Qualitative Results

To complement our study, we provide visual examples in Fig. 2. First, Vanilla Wan2.2, VideoTetris, and SwitchCraft produce similar results. They preserve the original quality because these algorithms are not strong enough to steer the generation toward the goal. Next, Peekaboo results are of good quality; however, the hard masking introduces visual artifacts in certain regions of the outputs. In contrast, our method succeeds where the baselines fail by successfully generating and localizing the target object within the specified region at the expense of background quality. For more visualization, we refer to our webpage.

<table><tr><td rowspan="2">Ablation</td><td colspan="3">Localization</td><td colspan="5">VBench</td></tr><tr><td>CD (↓)</td><td>IoU (↑)</td><td>SR (↑)</td><td>SC (↑)</td><td>BC (↑)</td><td>DD (↑)</td><td>AQ (↑)</td><td>IQ (↑)</td></tr><tr><td>Wan2.2 T2V</td><td>0.138</td><td>0.154</td><td>89.3%</td><td>0.974</td><td>0.971</td><td>0.248</td><td>0.631</td><td>0.666</td></tr><tr><td>(i) No Projection</td><td>0.059</td><td>0.355</td><td>87.5%</td><td>0.965</td><td>0.957</td><td>0.090</td><td>0.541</td><td>0.646</td></tr><tr><td>(ii) No Gaussian Fitting</td><td>0.059</td><td>0.375</td><td>72.0%</td><td>0.969</td><td>0.960</td><td>0.093</td><td>0.535</td><td>0.662</td></tr><tr><td>(iii) No Foreground</td><td>0.104</td><td>0.252</td><td>89.7%</td><td>0.977</td><td>0.968</td><td>0.158</td><td>0.558</td><td>0.660</td></tr><tr><td>(iv) No Background</td><td>0.105</td><td>0.241</td><td>89.7%</td><td>0.972</td><td>0.970</td><td>0.258</td><td>0.628</td><td>0.679</td></tr><tr><td>GATO-Vid (Ours)</td><td>0.059</td><td>0.363</td><td>89.7%</td><td>0.975</td><td>0.964</td><td>0.113</td><td>0.551</td><td>0.670</td></tr></table>

Table 2: Ablation Study: Efect of systematically removing each proposed component to validate its advantages. Eliminating any part of GATO-Vid degrades localization accuracy, which empirically validates all of our proposed components.

GATO-Vid  
(i) Projection  
(ii) Gaussian  
(iii) Foreground (iv) Background  
![](images/5ab4155ac46749c0b684c3ba9ae32c3e56977a68a2a64c3181a11f5e2af6932e.jpg)  
Fig. 3: Ablation Visualization: Representative frames from three videos illustrating the visual degradation under diferent ablation settings. Removing either the reprojection mechanism (i) or the Gaussian filtering (ii) introduces noticeable visual artifacts. Conversely, omitting the positive (iii) or negative biases (iv) leads to localization errors.

## 4.4 Ablation Studies

We conducted an ablation study to demonstrate the advantages of each proposed component. To do so, we performed the following four ablations: (i) We removed the projection to the ellipsoid, found in Eq. (14). Thus, we replaced the injection with a naive sum $( q _ { j }  q _ { j } + \lambda ^ { \pm } \| q _ { j } \| b ^ { \pm } )$ . (ii) We removed the Gaussian filtering, used to mimic the smooth surfaces of the cross-attention maps. Therefore, we set $\lambda ^ { + }$ equal for all queries inside the mask. Finally, (iii & iv) we removed the positive and negative biases $( b ^ { + }$ and $b ^ { - } ;$ ; refer to Eq. (13)). This is equivalent to removing the positive or negative influences of Eq. (10).

We present the ablation results in Tab. 2 along with some visualization in Fig. 3 (see our webpage for full videos). The performance diferences indicate that each component contributes positively to the localization metrics. Despite this, we observe some qualitative trade-ofs. Ablating the projection (i) and Gaussian (ii) mechanisms has the least efect on the localization metrics, but has a higher impact on the SR and VBench metrics. The reduction in DD, AQ, and IQ suggests that excluding these components negatively afects dynamics and visuals. Next, the positive and negative components of $\operatorname { E q }$ . (10) have the most significant impact. These two ablations demonstrate the most pronounced impact: a significant improvement in alignment comes at the expense of lower quality. In addition, and counterintuitively, removing the negative bias yielded the most severe degradation in localization performance.

![](images/60c04de2c72744a36a654b54d6876ed7ed52822ad0b0ad3d455ce8f6473edd28.jpg)  
(a) Iteration ablation

![](images/6ed2f67b9ce3cc7a1cc147cf49eb1b29eb3a454d87286933e99836a5f99c611a.jpg)  
(b) Block ablation

![](images/776e0562d4a8212393b390721effec8cc32c9e7783bf58ca4d98070a8d14575f.jpg)  
(c) Attention analysis  
Fig. 4: (a): More iterations induce higher IoU but reduce AQ. (b): Using the first 20 blocks is as efective as using all blocks, whereas using the last 20 blocks yields no improvement. (c): The early blocks produce higher attention scores than later blocks or steps, suggesting that initial layers specialize in the video’s general layout.

Finally, we analyze the efect of varying the injection scope across denoising steps and transformer blocks (Fig. 4). As shown in Fig. 4a, applying our mechanism across a larger fraction of sampling iterations increases the IoU but gradually degrades AQ. Applying bias injection during the first 15% of iterations strikes an optimal trade-of between spatial control and visual quality. Regarding layer depth, injecting bias into the first 20 blocks achieves an IoU and AQ performance similar to applying it across all 40 blocks, whereas targeting only the last 20 blocks yields results close to the Wan2.2 (Fig. 4b). This behavior is explained by the cross-attention score distribution in Fig. 4c, where early blocks show strong attention values, while later layers exhibit homogeneous scores, suggesting that late layers are less critical for structural layout.

## 5 Conclusions

We present GATO-Vid, a training- and gradient-free approach for spatially grounded text-to-video generation. By reformulating attention localization as an analytically solvable surrogate objective, our method replaces backpropagationbased approaches with a closed-form query steering mechanism. While our experiments suggest that while stronger spatial control can reduce scene dynamics and aesthetic appeal in some cases, they highlight the potential of analytical attention manipulation as an eficient alternative for controlling DiT-based architectures. We hope this work motivates future research into more accurate and scalable controllable video generation methods that better balance precise spatial control with visual fidelity.

Acknowledgements This work has been supported by chair VISA DEEP (ANR-20-CHIA-0022), Cluster PostGenAI@Paris (ANR-23-IACL-0007, FRANCE 2030), and funded by the French National Research Agency (ANR) under the Renaissance project (grant ANR-23-CE23-0023; France 2030). This work was performed using HPC resources from GENCI-IDRIS (Grant 2025-AD011017053).

## References

1. Avrahami, O., Hayes, T., Gafni, O., Gupta, S., Taigman, Y., Parikh, D., Lischinski, D., Fried, O., Yin, X.: Spatext: Spatio-textual representation for controllable image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 18370–18380 (June 2023)

2. Bansal, A., Chu, H.M., Schwarzschild, A., Sengupta, S., Goldblum, M., Geiping, J., Goldstein, T.: Universal guidance for difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 843–852 (June 2023)

3. Bar-Tal, O., Chefer, H., Tov, O., Herrmann, C., Paiss, R., Zada, S., Ephrat, A., Hur, J., Liu, G., Raj, A., Li, Y., Rubinstein, M., Michaeli, T., Wang, O., Sun, D., Dekel, T., Mosseri, I.: Lumiere: A space-time difusion model for video generation. In: SIGGRAPH Asia 2024 Conference Papers. SA ’24, Association for Computing Machinery, New York, NY, USA (2024). https://doi.org/10.1145/3680528. 3687614, https://doi.org/10.1145/3680528.3687614

4. Bar-Tal, O., Yariv, L., Lipman, Y., Dekel, T.: MultiDifusion: Fusing difusion paths for controlled image generation. In: Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., Scarlett, J. (eds.) Proceedings of the 40th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 202, pp. 1737–1752. PMLR (23–29 Jul 2023), https://proceedings.mlr.press/v202/ bar-tal23a.html

5. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Coll-Vinent, D.S., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., Lei, J., Ma, T., Guo, B., Kalla, A., Marks, M., Greer, J., Wang, M., Sun, P., Rädle, R., Afouras, T., Mavroudi, E., Xu, K., Wu, T.H., Zhou, Y., Momeni, L., HAZRA, R., Ding, S., Vaze, S., Porcher, F., Li, F., Li, S., Kamath, A., Cheng, H.K., Dollar, P., Ravi, N., Saenko, K., Zhang, P., Feichtenhofer, C.: SAM 3: Segment anything with concepts. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview.net/forum?id=r35clVtGzw

6. Chen, A., Xu, J., Zheng, W., Dai, G., Wang, Y., Zhang, R., Wang, H., Zhang, S.: Training-free regional prompting for difusion transformers. arXiv preprint arXiv:2411.02395 (2024)

7. Chen, H., Zhang, Y., Cun, X., Xia, M., Wang, X., Weng, C., Shan, Y.: Videocrafter2: Overcoming data limitations for high-quality video difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 7310–7320 (June 2024)

8. Chen, H., Li, J., Zhuang, W., Vikalo, H., Lyu, L.: Training-free layout-to-image generation with marginal attention constraints. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 3410–3419 (June 2026)

9. Chen, M., Laina, I., Vedaldi, A.: Training-free layout control with cross-attention guidance. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 5343–5353 (January 2024)

10. Chen, W., Ji, Y., Wu, J., Wu, H., Xie, P., Li, J., Xia, X., Xiao, X., Lin, L.: Control-A-Video: Controllable text-to-video difusion models with motion prior and reward feedback learning (2023), https://arxiv.org/abs/2305.13840

11. Couairon, G., Careil, M., Cord, M., Lathuilière, S., Verbeek, J.: Zero-shot spatial layout conditioning for text-to-image difusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 2174–2183 (October 2023)

12. Dao, T., Fu, D.Y., Ermon, S., Rudra, A., Ré, C.: FlashAttention: Fast and memoryeficient exact attention with IO-awareness. In: Advances in Neural Information Processing Systems (NeurIPS) (2022)

13. Dou, H., Li, R., Su, W., Li, X.: Gvdif: Grounded text-to-video generation with difusion models (2024), https://arxiv.org/abs/2407.01921

14. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Rombach, R.: Scaling rectified flow transformers for high-resolution image synthesis. In: Proceedings of the 41st International Conference on Machine Learning. ICML’24, JMLR.org (2024)

15. Feng, W., Liu, C., Liu, S., Wang, W.Y., Vahdat, A., Nie, W.: Blobgen-vid: Compositional text-to-video generation with blob video representations. In: Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR). pp. 12989– 12998 (June 2025)

16. Geng, D., Herrmann, C., Hur, J., Cole, F., Zhang, S., Pfaf, T., Lopez-Guevara, T., Doersch, C., Aytar, Y., Rubinstein, M., Sun, C., Wang, O., Owens, A., Sun, D.: Motion prompting: Controlling video generation with motion trajectories. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1–12 (2025), https://openaccess.thecvf.com/content/CVPR2025/ html/Geng\_Motion\_Prompting\_Controlling\_Video\_Generation\_with\_Motion\_ Trajectories\_CVPR\_2025\_paper.html

17. Guo, Y., Yang, C., Rao, A., Liang, Z., Wang, Y., Qiao, Y., Agrawala, M., Lin, D., Dai, B.: Animatedif: Animate your personalized text-to-image difusion models without specific tuning. International Conference on Learning Representations (2024)

18. HaCohen, Y., Brazowski, B., Chiprut, N., Bitterman, Y., Kvochko, A., Berkowitz, A., Shalem, D., Lifschitz, D., Moshe, D., Porat, E., Richardson, E., Shiran, G., Chachy, I., Chetboun, J., Finkelson, M., Kupchick, M., Zabari, N., Guetta, N.B., Kotler, N.G., Bibi, O., Gordon, O., Panet, P., Benita, R., Armon, S., Kulikov, V.M., Inger, Y., Shiftan, Y., Melumian, Z., Farbman, Z.: Ltx-2: Eficient joint audio-visual foundation model. ArXiv abs/2601.03233 (2026), https://api. semanticscholar.org/CorpusID:284512618

19. He, H., Xu, Y., Guo, Y., Wetzstein, G., Dai, B., Li, H., Yang, C.: CameraCtrl: Enabling camera control for text-to-video generation (2024), https://arxiv.org/ abs/2404.02101

20. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

21. Jain, Y., Nasery, A., Vineet, V., Behl, H.: Peekaboo: Interactive video generation via masked-difusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8079–8088 (June 2024)

22. Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., Lee, Y.J.: Gligen: Open-set grounded text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 22511–22521 (June 2023)

23. Lian, L., Shi, B., Yala, A., darrell, t., Li, B.: Llm-grounded video difusion models. In: Kim, B., Yue, Y., Chaudhuri, S., Fragkiadaki, K., Khan, M., Sun, Y. (eds.) International Conference on Learning Representations. vol. 2024, pp. 50207– 50227 (2024), https://proceedings.iclr.cc/paper\_files/paper/2024/file/ d9f8b5abc8e0926539ecbb492af7b2f1-Paper-Conference.pdf

24. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: The Eleventh International Conference on Learning Representations (2023), https://openreview.net/forum?id=PqvMRDCJT9t

25. Liu, J., Huang, T., Xu, C.: Training-free composite scene generation for layout-toimage synthesis. In: Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler, T., Varol, G. (eds.) Computer Vision – ECCV 2024. pp. 37–53. Springer Nature Switzerland, Cham (2025)

26. Nan, K., Xie, R., Zhou, P., Fan, T., Yang, Z., Chen, Z., Li, X., Yang, J., Tai, Y.: Openvid-1m: A large-scale high-quality dataset for text-to-video generation. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id=j7kdXSrISM

27. Nie, W., Liu, S., Mardani, M., Liu, C., Eckart, B., Vahdat, A.: Compositional textto-image generation with dense blob representations. In: Proceedings of the 41st International Conference on Machine Learning. ICML’24, JMLR.org (2024)

28. Ohanyan, M., Manukyan, H., Wang, Z., Navasardyan, S., Shi, H.: Zero-painter: Training-free layout control for text-to-image synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8764–8774 (June 2024)

29. OpenAI: Gpt-4 technical report (2024), https://arxiv.org/abs/2303.08774

30. Peebles, W., Xie, S.: Scalable difusion models with transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 4195– 4205 (October 2023)

31. Phung, Q., Ge, S., Huang, J.B.: Grounded text-to-image synthesis with attention refocusing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 7932–7942 (June 2024)

32. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10684– 10695 (June 2022)

33. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Navab, N., Hornegger, J., Wells, W.M., Frangi, A.F. (eds.) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. pp. 234–241. Springer International Publishing, Cham (2015)

34. Shen, T., Huang, Z., Li, X., Lin, Z., Liu, J., Wang, Y., Feng, J., Yang, M.H., Liew, J.H.: Qk-edit: Revisiting attention-based injection in mm-dit for image and video editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 19043–19053 (October 2025)

35. Shirakawa, T., Uchida, S.: Noisecollage: A layout-aware text-to-image difusion model based on noise cropping and merging. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

36. Team, G.: Gemini: A family of highly capable multimodal models (2025), https: //arxiv.org/abs/2312.11805

37. Team, T.H.F.M.: Hunyuanvideo 1.5 technical report (2025), https://arxiv.org/ abs/2511.18870

38. Tian, Y., Yang, L., Yang, H., Gao, Y., Deng, Y., Chen, J., Wang, X., Yu, Z., Tao, X., Wan, P., Zhang, D., Cui, B.: Videotetris: Towards compositional textto-video generation. In: Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., Zhang, C. (eds.) Advances in Neural Information Processing Systems. vol. 37, pp. 29489–29513. Curran Associates, Inc. (2024). https://doi. org/10.52202/079017-0928, https://proceedings.neurips.cc/paper\_files/ paper/2024/file/345208bdbbb6104616311dfc1d093fe7-Paper-Conference.pdf

39. Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Meng, X., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W.C., Wang, W., Shen, W., Yu, W., Shi, X., Huang, X., Xu, X., Kou, Y., Lv, Y.M., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li, Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z., Liu, Z.: Wan: Open and advanced large-scale video generative models. ArXiv abs/2503.20314 (2025), https://api.semanticscholar.org/CorpusID: 277321639

40. Wang, J., Yuan, H., Chen, D., Zhang, Y., Wang, X., Zhang, S.: Modelscope textto-video technical report (2023), https://arxiv.org/abs/2308.06571

41. Wang, R., Chen, Z., Chen, C., Ma, J., Lu, H., Lin, X.: Compositional textto-image synthesis with attention map control of difusion models. In: Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence. AAAI’24/IAAI’24/EAAI’24, AAAI Press (2024). https://doi.org/10.1609/ aaai.v38i6.28364, https://doi.org/10.1609/aaai.v38i6.28364

42. Wang, X., Yuan, H., Zhang, S., Chen, D., Wang, J., Zhang, Y., Shen, Y., Zhao, D., Zhou, J.: VideoComposer: Compositional video synthesis with motion controllability. In: Advances in Neural Information Processing Systems. vol. 36 (2023), https://papers.nips.cc/paper\_files/paper/2023/hash/ 180f6184a3458fa19c28c5483bc61877-Abstract-Conference.html

43. Wang, Y., He, Y., Li, Y., Li, K., Yu, J., Ma, X., Li, X., Chen, G., Chen, X., Wang, Y., Luo, P., Liu, Z., Wang, Y., Wang, L., Qiao, Y.: Internvid: A largescale video-text dataset for multimodal understanding and generation. In: The Twelfth International Conference on Learning Representations (2024), https:// openreview.net/forum?id=MLBdiWu4Fw

44. Wang, Z., Yuan, Z., Wang, X., Li, Y., Chen, T., Xia, M., Luo, P., Shan, Y.: MotionCtrl: A unified and flexible motion controller for video generation. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1–11 (2024). https://doi.org/10.1145/ 3641519.3657518

45. Wu, Y., Zhou, X., Ma, B., Su, X., Ma, K., Wang, X.: Ifadapter: Instance feature control for grounded text-to-image generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 15949–15959 (October 2025)

46. Xiao, J., Lv, H., Li, L., Wang, S., Huang, Q.: R&amp;b: Region and boundary aware zero-shot grounded text-to-image generation. In: Kim, B., Yue, Y., Chaudhuri, S., Fragkiadaki, K., Khan, M., Sun, Y. (eds.) International Conference on Learning Representations. vol. 2024, pp. 26088– 26116 (2024), https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 6f61f25abf3197bb58cc7422761186e0-Paper-Conference.pdf

47. Xie, J., Li, Y., Huang, Y., Liu, H., Zhang, W., Zheng, Y., Shou, M.Z.: Boxdif: Textto-image synthesis with training-free box-constrained difusion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 7452– 7461 (2023)

48. Xu, Q., Song, C., Cai, Y., Zhang, C.: Switchcraft: Training-free multi-event video generation with attention controls. In: Proceedings of the IEEE/CVF Conference

on Computer Vision and Pattern Recognition (CVPR). pp. 29136–29145 (June 2026)

49. Yang, S.: Mftf: Mask-free training-free object level layout control difusion model (2024)

50. Yang, S., Hou, L., Huang, H., Ma, C., Wan, P., Zhang, D., Chen, X., Liao, J.: Direct-a-video: Customized video generation with user-directed camera movement and object motion. In: ACM SIGGRAPH 2024 Conference Papers. SIGGRAPH ’24, Association for Computing Machinery, New York, NY, USA (2024). https: //doi.org/10.1145/3641519.3657481, https://doi.org/10.1145/3641519. 3657481

51. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., Yin, D., Yuxuan.Zhang, Wang, W., Cheng, Y., Xu, B., Gu, X., Dong, Y., Tang, J.: Cogvideox: Text-to-video difusion models with an expert transformer. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id=LQzN6TRFg9

52. Yi, L., Minyi, L., Bozheng, L., Jiawang, C., Wenbo, Z.: Zerotrail: Zero-shot trajectory control framework for video difusion models. In: Advances in Neural Information Processing Systems Workshops (2025)

53. Zhang, B., Sennrich, R.: Root mean square layer normalization. In: Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 32. Curran Associates, Inc. (2019), https://proceedings.neurips.cc/paper\_files/paper/2019/file/ 1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf

54. Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.C., Xu, M., Wright, L., Shojanazeri, H., Ott, M., Shleifer, S., Desmaison, A., Balioglu, C., Damania, P., Nguyen, B., Chauhan, G., Hao, Y., Mathews, A., Li, S.: Pytorch fsdp: Experiences on scaling fully sharded data parallel. Proc. VLDB Endow. 16(12), 3848–3860 (Aug 2023). https://doi.org/10.14778/3611540.3611569, https: //doi.org/10.14778/3611540.3611569