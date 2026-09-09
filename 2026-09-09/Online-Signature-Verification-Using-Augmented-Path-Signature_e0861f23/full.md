# Online Signature Verification Using Augmented Path Signature and T-Mamba

Ruiling Li and Danyu Yang<sup>(Q)</sup>

College of Mathematics and Statistics, Chongqing University, China 202406021058T@stu.cqu.edu.cn, danyuyang@cqu.edu.cn

Abstract. Handwritten signature verification is vital for personal authentication across commercial and financial applications. Although deep learning methods are widely adopted for online signature verification (OSV), they often struggle with capturing highly discriminative features and modelling long-range dependencies. To address these issues, we propose a novel framework that integrates the augmented path signature (APS) descriptor with the T-Mamba model. The APS descriptor first applies time and basepoint augmentations, then computes slidingwindow path signatures. The path signature is a non-parametric feature map from rough path theory that efectively captures geometric structures and nonlinear inter-channel interactions. Inspired by the eficacy of state space models (SSMs) in sequence modelling, our T-Mamba model employs a hybrid design combining two temporal convolutional network (TCN) blocks with a time-scanning Mamba. This design enables the model to learn both local temporal patterns and global longrange dependencies, substantially improving verification accuracy. Our framework achieves state-of-the-art EERs on three public benchmark datasets (MCYT-100, SVC-2004 Task 2, DeepSignDB), validating its efectiveness and robustness, especially when the training data is limited. Our code is publicly available at https://github.com/DLRL04/ OSV-using-APS-and-T-Mamba.

Keywords: Online signature verification · Mamba · Temporal convolutional network · Path signature · Dynamic time warping.

## 1 Introduction

In the past decades, handwritten signatures have served as one of the most widely accepted forms of personal authentication, playing a vital role in administrative, commercial, and financial contexts such as contract signing and payment authorization. Signature verification is typically categorized as ofline and online approaches based on the data acquisition process. While ofline signature verification systems analyse static images, online signature verification (OSV) systems capture dynamic trajectories using specialised digital devices such as smartphones and pressure-sensitive tablets, making them more reliable than ofline approaches [29,14]. However, efectively modelling the complex and multidimensional time series for OSV remains challenging. This dificulty arises from the substantial intra-writer variability inherent in human handwriting, as well as the presence of skilled and random forgery attacks, making OSV considerably more challenging than many other biometric modalities [33].

Traditional parameter- or function-based feature extraction for OSV lacks explicit mechanisms to address intra-writer variability. To overcome this, we introduce the augmented path signature (APS) descriptor as a robust, discriminative representation. Specifically, raw signature trajectories are augmented with timestamps and basepoints prior to sliding-window path signature computation [6]. Grounded in rough path theory [22], this principled non-parametric descriptor efectively captures local geometric structures and nonlinear inter-channel interactions, thereby significantly boosting verification performance.

With the advent of deep learning, models such as RNNs [21,20], CNNs [4,37], and Transformers [36,24] have achieved promising results in OSV. Nevertheless, RNNs are capable of capturing temporal dependencies but struggle with long sequences and typically require a substantial amount of training data. CNNs eficiently extract local and spatial patterns but are limited by their receptive fields. While Transformers excel at modelling long-range dependencies, they incur high computational costs and may overfit when training data is limited. Although great advances have been made, signature verification remains an open challenge. In light of this, we introduce T-Mamba, a hybrid model that integrates temporal convolutional networks (TCN) with time-scanning Mamba, a selective state-space model based on Mamba [10]. By employing TCNs with dilated convolutions, we first expand the receptive field to capture fine-grained local patterns. Then the time-scanning Mamba processes the sequence in both forward and backward temporal directions with shared parameters, enabling globally selective integration of relevant information with linear complexity. This combination keeps T-Mamba computationally eficient and robust to the variable lengths typical of handwritten signature sequences.

Alongside representation learning, deep metric learning has emerged as a cornerstone of signature verification by optimizing feature spaces to maximize inter-writer separability and minimize intra-writer variation. While traditional Dynamic Time Warping (DTW) [17] excels at temporal alignment, its nondiferentiable nature precludes direct end-to-end integration. To circumvent this, we employ soft-DTW [7] during training to compute diferentiable pairwise alignment costs within a triplet loss. During inference, standard DTW is utilized to calculate the optimal alignment and similarity scores between query and reference signatures for authentication.

The overall framework of our proposed method is illustrated in Fig. 1. Specifically, we first extract the APS features to obtain representations of the local geometric structure. These features are then fed into T-Mamba, optimised during training with a soft-DTW triplet loss, and evaluated during testing using standard DTW. The primary contributions of this paper are as follows:

– We introduce the APS descriptor, a robust representation designed to capture the local geometric structure of signatures. By integrating path augmentations with sliding-window path signatures from rough path theory, it enables the extraction of high-order discriminative features.

![](images/03d062c0cabaae29f33fd20a4f79263b689d01263481b1a75d8603baf43ea947.jpg)  
Fig. 1. Overall framework of the proposed method.

– To the best of our knowledge, this work is the first to apply a state space model, specifically Mamba, to online signature verification and to demonstrate its efectiveness.

– We propose T-Mamba, a novel backbone that integrates TCN blocks with a time-scanning Mamba through max pooling. This design strikes an efective balance between local patterns and global contextual information.

– We achieve state-of-the-art performance on three public benchmarks (MCYT-100, SVC-2004 Task 2, DeepSignDB), demonstrating the efectiveness and robustness of our framework across diverse datasets and acquisition devices.

## 2 Related Work

Over the past three decades, signature verification has remained an active research area, with comprehensive surveys [29,14] reviewing its developments. In this pipeline, preprocessing and robust feature representation are essential to mitigate noise and handle variability. For instance, Li et al. [21] proposed a stroke segmentation approach to isolate individual writing behaviors, while Lai et al. [20] introduced the LNPS descriptor, achieving scale and rotation invariance to significantly enhance feature robustness. The path signature was originally introduced by Chen [5] to study the geometry of paths, and later further developed by Lyons in the rough path theory [22]. Graham introduced path signature features to machine learning [9] and won the ICDAR 2013 Online Isolated Chinese Character Recognition Competition.

DTW is a cornerstone for aligning signature sequences by minimizing cumulative alignment costs. To improve precision, SM-DTW [28] assigns higher weights to stable regions while penalizing distortions. Alternatively, to mitigate template sensitivity, Okawa [25] introduced a single-template DTW approach that computes an Euclidean barycenter to summarize multiple genuine signatures into a representative reference. Furthermore, the development of diferentiable soft-DTW [7,15] has successfully bridged the gap between DTW and deep learning. By incorporating soft-DTW into the loss function, Jiang et al. [15] directly optimised the alignment process during training, leading to significant performance gains.

To model the temporal dynamics, Lai and Jin [18] proposed a recurrent adaption network as a neural filter to learn discriminative representations from signature sequences. To further exploit local features, researchers have proposed models based on CNNs [4,37], achieving lower EERs than earlier sequence models. In [37], Vorugunti et al. introduced a lightweight framework employing depthwise separable convolutions which efectively reduces parameters while maintaining high accuracy. To better capture the global context, Transformer-based architectures have attracted growing attention. However, despite their dominance in natural language processing, their eficacy in OSV remains limited [36,24]. TSOSVNet [36] reported a relatively high EER of 3.07% on MCYT-100. Melzi et al. [24] demonstrated that a vanilla Transformer encoder can marginally surpass RNN-based baselines, while the added architectural complexity often results in a performance plateau that lags behind state-of-the-art models.

To overcome these limitations, state-space models (SSMs) [11,10] have emerged as a promising alternative, ofering linear computational complexity while maintaining long-range dependency modelling capabilities. In 2024, Gu and Dao [10] introduced Mamba, a selective SSM that filters noise and retains task-relevant information in long sequences. Its superior balance between performance and complexity has established it as a potent sequence modelling backbone [1]. Building upon this, this paper presents the first attempt to apply Mamba to signature verification, achieving state-of-the-art results.

## 3 Preprocessing with Augmented Path Signature

Our preprocessing workflow is illustrated in Fig. 2. APS is a geometric descriptor constructed by first applying time and basepoint augmentations, then extracting sliding-window path signatures.

![](images/fa092f108bf49ae88c4a3f8e3f8ef436c545f456ef6d4348af2babefed71eff0.jpg)  
Fig. 2. Data preprocessing workflow.

## 3.1 Time Functions

We select 12 time functions derived from coordinates and pressure, as shown in Table 1, and then apply z-score normalization to each feature. Previous studies [15,23] have employed these time functions, demonstrating strong empirical performance for both stylus- and finger-written signatures.

Table 1. Time functions for dynamic feature extraction.  
Number Time Functions   
$1 \sim 3$ Velocity features: $v _ { x } , v _ { y } , v = \sqrt { v _ { x } ^ { 2 } + v _ { y } ^ { 2 } }$   
$4 \sim 6$ Path orientation features: θ = arctan $( v _ { y } / v _ { x } )$ , cos(θ), sin(θ)   
$7 \sim 8$ First-order derivatives of v and $\theta \colon \dot { v } , \dot { \theta }$   
$9 \sim 1 0$ Log curvature radius and centripetal acceleration: $\rho = \log ( v / \dot { \theta } ) , c = v \cdot \dot { \theta }$   
$1 1 \sim 1 2$ Total acceleration and pressure: $a = { \sqrt { \dot { v } ^ { 2 } + c ^ { 2 } } } , p$

## 3.2 Augmented Path Signature (APS)

Path Augmentations Since path signature features are invariant to translations and time reparametrizations, these properties may sometimes reduce their discriminative ability. Here we add two path augmentations to improve sensitivity to timestamps and the absolute spatial position respectively [6].

The time augmentation transforms a d-dimensional path X into a $( d + 1 )$ dimensional path by incorporating the increasing timestamps:

$$
\phi _ { t } ( X ) = { \big ( } ( t _ { 0 } , X _ { 0 } ) , ( t _ { 1 } , X _ { 1 } ) , \ldots , ( t _ { l } , X _ { l } ) { \big ) } .
$$

This augmentation encodes the writing speed, and can ensure the uniqueness of the path signature [13].

The basepoint augmentation adds a zero at the beginning of the series:

$$
\phi _ { b } ( X ) = { \big ( } 0 , X _ { 1 } , \ldots , X _ { l } { \big ) } .
$$

It encodes the absolute position and removes the translation invariance introduced by the path signature features.

The efect of these augmentations, both individually and in combination, will be discussed in Section 5.

Sliding-Window Path Signatures Let $X : [ a , b ]  \mathbb { R } ^ { d }$ be a continuous path of bounded variation. For any positive integer N, the order-N truncated path signature is defined as:

$$
S _ { N } ( \boldsymbol { X } ) = \Big ( \{ S ( \boldsymbol { X } ) ^ { ( j _ { 1 } ) } \} _ { j _ { 1 } = 1 } ^ { d } , \{ S ( \boldsymbol { X } ) ^ { ( j _ { 1 } , j _ { 2 } ) } \} _ { j _ { 1 } , j _ { 2 } = 1 } ^ { d } , \ldots , \{ S ( \boldsymbol { X } ) ^ { ( j _ { 1 } , \ldots , j _ { N } ) } \} _ { j _ { 1 } , \ldots , j _ { N } = 1 } ^ { d } \Big ) ,
$$

where

$$
S ( X ) ^ { ( j _ { 1 } , \ldots , j _ { k } ) } = \int _ { a \leq t _ { 1 } < \cdots < t _ { k } \leq b } \mathrm { d } X _ { t _ { 1 } } ^ { j _ { 1 } } \cdot \cdot \cdot \mathrm { d } X _ { t _ { k } } ^ { j _ { k } } ,
$$

which is a feature map of dimension $\textstyle \sum _ { k = 1 } ^ { N } d ^ { k }$ . In the experiments, the continuous path is constructed by linearly interpolating between consecutive time points.

An illustration of the order-2 truncated path signature of a planar path is shown in Fig. 3. The oriented increments $\varDelta X ^ { 1 }$ and $\bar { \varDelta } X ^ { 2 }$ correspond to the firstlevel iterated integrals, while the signed area $A ^ { + } - A ^ { - }$ is a linear combination of second-level iterated integrals based on Green’s theorem.

![](images/c497d0bdb38006b1a9209860dd7b7d19de9bbc6ad0c9f1d42b5221d148d632d7.jpg)  
Fig. 3. Geometric interpretation of the order-2 truncated path signature of a planar path (shown in red) that starts at the bottom-left and ends at the top-right.

## Benefits of the Path Signature

– The path signature collapses the temporal dimension into a fixed-dimensional representation, enabling direct comparison of varying-length time series.   
– It ofers a top-down geometric description, with the truncation improving computational eficiency and suppressing noisy local variations.

– Iterated integrals encode nonlinear inter-channel interactions, thereby capturing fine-grained dynamic features, which are not accessible to linear transformations.

To encode contextual information, a sliding window of size w is applied to the time series at each time step t before extracting path signature features. Define the sliding window $W _ { t }$ :

$$
W _ { t } = ( X _ { t } , X _ { t + 1 } , \ldots , X _ { t + w - 1 } ) .
$$

The APS descriptor of order N is then defined as the sequence of path signatures computed over each sliding window of the augmented path:

$$
\mathrm { A P S } : = \left( S _ { N } ( W _ { 1 } ) , S _ { N } ( W _ { 2 } ) , S _ { N } ( W _ { 3 } ) , \ldots , S _ { N } ( W _ { l } ) \right) ,
$$

where l is the total number of windows.

## 4 T-Mamba with Soft-DTW

## 4.1 TCN Block

The temporal convolutional network (TCN) [3] is depicted in Fig. 4a. This block is a residual module composed of two layers of dilated causal 1D convolutions. Each Conv1D layer is followed by weight normalization, a ReLU activation, and a spatial dropout for regularization. The receptive field of the network is jointly determined by the network depth $m = 2$ , filter size $k = 2$ and dilation factor $d = 2$ , where each filter tap contributes $( k - 1 ) d$ steps of historical context that accumulate across stacked layers.

![](images/864e75f6a58c8130c6ae257cce016d2acc18682a16100a40384759296827763b.jpg)  
(a) TCN block

![](images/631e885982917be13cf25dcfecb740f955c6c8ae689d7353c1f6db2b5339a260.jpg)  
(b) Mamba  
Fig. 4. The structure of (a) TCN block and (b) Mamba. The symbol ⊕ indicates an element-wise addition and ⊗ represents matrix multiplication. Here σ is SiLU.

## 4.2 Time-Scanning Mamba

The structure of Mamba is illustrated in Fig. 4b. Continuous state space models define the evolution of a latent state $h ( t )$ and output y(t) as:

$$
\frac { d h ( t ) } { d t } = A h ( t ) + B x ( t ) , \qquad y ( t ) = C h ( t ) ,
$$

Here A, B, C are learnable and time-invariant, and the system is inherently linear time-invariant (LTI) [11]. To execute the continuous models in practice, they are mapped to a discrete recurrence $h _ { k } = \tilde { A } h _ { k - 1 } + \tilde { B } x _ { k }$ through a fixed discretization rule $\varDelta .$ , where the exact forms of A<sup>˜</sup> and B<sup>˜</sup> depend on the numerical scheme used.

Mamba’s key innovation is to remove the LTI constraint by making the SSM parameters $B , C ,$ , and the discretization rule ∆ input-dependent and timevarying, introducing selectivity into the model. For an input token $x _ { k }$ , these parameters are computed through linear maps and activation functions:

$$
B _ { k } = W _ { D } x _ { k } , C _ { k } = W _ { D } x _ { k } , \varDelta _ { k } = \mathrm { s o f t p l u s } \left( \mathrm { b i a s } + W _ { D } ( W _ { 1 } x _ { k } ) \right) ,
$$

where $W _ { d }$ is a linear map to a d-dimensional space and the bias is learnable. Specifically, based on [10, Theorem 1], when $D = 1 , A = - 1 , B = 1$ , the selective recurrence takes the form:

$$
g _ { k } = \mathrm { s i g m o i d } \big ( \mathrm { L i n e a r } ( x _ { k } ) \big ) , \qquad h _ { k } = \left( 1 - g _ { k } \right) h _ { k - 1 } + g _ { k } x _ { k } .
$$

This gating mechanism enables precise state control: as $g _ { k } \ \to \ 1$ , the update focuses on the current input $x _ { k }$ , efectively resetting the state. To circumvent the convolutional constraints imposed by this selective recurrence, Mamba leverages a hardware-aware parallel algorithm to maintain high throughput [10].

However, since this selective recurrence is strictly causal, Mamba cannot exploit future context, which limits its eficacy when the prior context is uninformative. Inspired by the reversal-based scanning in [1], we propose a time-scanning Mamba that captures complementary bidirectional contexts via a parametershared dual temporal scan.

Given a sequence $a = [ a _ { 1 } , \dots , a _ { L } ] \in \mathbb { R } ^ { L \times D }$ , its temporal reversal is denoted as $\operatorname { R } ( a ) = [ a _ { L } , a _ { L - 1 } , \dotsc , a _ { 1 } ]$ . Let Mamba<sub>θ</sub>(a) represent the output of a vanilla Mamba parameterized by θ. To capture bidirectional temporal context using parameter-shared scanning, the sequence is processed in both directions and aggregated via element-wise addition ⊕:

$$
\begin{array} { r l } & { \mathrm { o u t p u t } ^ { ( 1 ) } = a \ \oplus \ \mathrm { M a m b a } _ { \theta } ( a ) , } \\ & { \mathrm { o u t p u t } ^ { ( 2 ) } = \mathrm { R } ( a ) \ \oplus \ \mathrm { M a m b a } _ { \theta } ( \mathrm { R } ( a ) ) , } \\ & { \mathrm { o u t p u t } = \mathrm { o u t p u t } ^ { ( 1 ) } \ \oplus \ \mathrm { R } ( \mathrm { o u t p u t } ^ { ( 2 ) } ) . } \end{array}
$$

Here, the first two branches execute forward and backward scanning with shared weights θ, maintaining original information via residual paths. Aggregating these streams allows each token to leverage both past and future contexts for enhanced sequence modelling.

## 4.3 Soft-DTW with a Triplet Loss

γ-Soft-DTW Alignment Cost Let $X \ = \ [ x _ { 1 } , \ldots , x _ { l _ { 1 } } ] \ \in \ \mathbb { R } ^ { l _ { 1 } \times d }$ and $Y =$ $[ y _ { 1 } , \dots , y _ { l _ { 2 } } ] \in \mathbb { R } ^ { l _ { 2 } \times d }$ be two variable-length sequences that are produced by T-Mamba. Let $B _ { \ell _ { 1 } , \ell _ { 2 } } \subset \{ 0 , 1 \} ^ { \ell _ { 1 } \times \ell _ { 2 } }$ denote the set of admissible binary alignment matrices satisfying continuity, monotonicity and boundary conditions [30]. The DTW alignment cost is defined as

$$
d ( X , Y ) = \operatorname { D T W } ( X , Y ) \ = \ \operatorname* { m i n } _ { B \in \mathcal { B } _ { \ell _ { 1 } , \ell _ { 2 } } } \langle B , \Delta ( X , Y ) \rangle ,\tag{1}
$$

where $\varDelta ( X , Y )$ is an $\ell _ { 1 } \times \ell _ { 2 }$ matrix given by $[ \varDelta ( X , Y ) ] _ { i , j } : = \| x _ { i } - y _ { j } \| _ { 2 } ^ { 2 }$ and $\langle \cdot , \cdot \rangle$ denotes the inner product.

However, DTW cannot be directly optimized via backpropagation due to its non-diferentiable nature. To address this limitation, Cuturi and Blondel [7] introduced a diferentiable alternative based on a smoothed minimum controlled by a non-negative parameter γ:

$$
\begin{array} { r } { \operatorname* { m i n } _ { \gamma } \{ b _ { 1 } , \dotsc , b _ { n } \} = \left\{ \begin{array} { l l } { \operatorname* { m i n } _ { i } b _ { i } , } & { \gamma = 0 , } \\ { - \gamma \log \sum _ { i = 1 } ^ { n } e ^ { - b _ { i } / \gamma } , } & { \gamma > 0 , } \end{array} \right. } \end{array}
$$

and the γ-soft-DTW alignment cost is defined as:

$$
d _ { \gamma } ( X , Y ) = { \mathrm { D T W } } _ { \gamma } ( X , Y ) \ = \ \operatorname* { m i n } _ { \gamma } \Big \{ \langle B , \Delta ( X , Y ) \rangle : \ B \in \mathcal { B } _ { \ell _ { 1 } , \ell _ { 2 } } \Big \} .
$$

The parameter $\gamma$ controls the smoothness of the cost. $\mathrm { A s } \ \gamma \to 0$ , it recovers the original DTW. For $\gamma > 0$ , the cost becomes diferentiable, allowing gradients to propagate across all warping paths.

Triplet Loss with Alignment Cost We formulate OSV as an alignment-based learning task, employing a triplet loss to enforce intra-writer similarity and maximize inter-writer separability. For each training triplet $( X _ { a } , X _ { g } , X _ { f } ) \in \mathcal { S }$ comprising an anchor, a genuine sample, and a forgery—the objective minimizes the internal alignment costs. Specifically, to enhance intra-writer compactness, we incorporate an intra-writer contraction term that penalizes the average alignment cost over the anchor-genuine subset $\boldsymbol { \mathcal { S } } _ { g }$ , defined as:

$$
\mathcal { V } _ { \mathrm { i n t r a } } = \frac { 1 } { | \mathcal { S } _ { g } | } \sum _ { ( X _ { a } , X _ { g } ) \in \mathcal { S } _ { g } } d _ { \gamma } ( X _ { a } , X _ { g } ) .
$$

Then the final loss function is given by:

$$
\mathcal { L } = \frac { 1 } { | \mathcal { S } | } \sum _ { ( X _ { a } , X _ { g } , X _ { f } ) \in \mathcal { S } } \Big ( \Big [ d _ { \gamma } ( X _ { a } , X _ { g } ) + \xi - d _ { \gamma } ( X _ { a } , X _ { f } ) \Big ] _ { + } + \ \lambda \mathcal { V } _ { \mathrm { i n t r a } } \Big ) ,\tag{2}
$$

where the first term enforces a positive margin ξ between the anchor–genuine and anchor–forgery alignment costs. Concurrently, the intra-writer term, regulated by λ, minimizes variance among genuine samples to regularize the embedding space. This dual constraint yields compact writer-specific clusters and establishes a robust discriminative boundary.

Verifier Based on DTW During the test stage, given m reference signatures $X ^ { 1 } , \ldots , X ^ { m }$ and a test signature $Y$ , we compute the DTW cost $d ( X _ { i } , Y )$ for $i = 1 , \cdots , m$ based on Eq. 1. Let D denote the average DTW cost over all pairs of reference signatures. For each writer, we compute the following scores:

$$
s _ { \mathrm { a v e } } ( Y ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } d ( X _ { i } , Y ) / \sqrt { D } , \quad s _ { \mathrm { m i n } } ( Y ) = \operatorname* { m i n } _ { i = 1 , \dots , m } d ( X _ { i } , Y ) / \sqrt { D } .
$$

Define the final score $s ( Y ) = s _ { \mathrm { a v e } } ( Y ) + s _ { \mathrm { m i n } } ( Y ) . \mathrm { I f } ~ s ( Y ) \leq \tau$ where τ is a writerdependent threshold, the test signature $Y$ is classified as a genuine; otherwise, it is considered a forgery. The performance of the OSV system is evaluated using the writer-specific equal error rate (EER), a standard criterion for OSV, obtained by sweeping τ to the point where the false acceptance rate equals the false rejection rate.

## 5 Experiments

## 5.1 Datasets and Implementation Details

Experiments were conducted on three benchmark datasets: MCYT-100 [27] (100 writers, 25 genuine and 25 forged signatures per writer), SVC-2004 Task 2 [40] (40 writers, 20 genuine and 20 forged signatures per writer), and the large-scale DeepSignDB [34] (69,972 signatures from 1,526 writers). DeepSignDB integrates multiple subsets (e.g., MCYT, BiosecurID, Biosecure DS2, e-BioSign DS1, and e-BioSign DS2) to provide diverse stylus- and finger-written conditions.

Following [38], we evaluate the system under both skilled forgery (S\_N) and random forgery (R\_N) protocols, where $N \in \{ 5 , 1 0 , 1 5 \}$ denotes the number of genuine training samples per writer. For enrollment, 5 genuine signatures serve as references, with the remaining samples reserved for verification.

The T-Mamba framework comprises two TCN blocks (hidden dimensions: 256 and 128) followed by a time-scanning Mamba module (state expansion: 256). All models were implemented in PyTorch and trained on a single NVIDIA A100 GPU (80 GB) using SGD for 20 epochs. The initial learning rate was set to 0.001 with an exponential decay factor of 0.9 per epoch, utilizing a batch size of 40. Following [15], the margin factor ξ in Eq. 2 and the smoothing parameter γ were fixed at 1 and 5, respectively. The configuration was empirically optimized via grid search, with all results averaged over five random seeds.

## 5.2 Efectiveness of APS

To evaluate the contribution of the APS descriptor, we conducted a comparative experiment on MCYT-100 under S\_05. We assessed the performance of APS across various sliding-window sizes w using four augmentation configurations: (i) the path signature (PS), (ii) the time augmentation (TA) then PS, (iii) the basepoint augmentation (BA) then PS, and (iv) APS.

Table 2 shows that a moderate window size $( w = 9$ to 13) achieves optimal performance by encoding a balanced temporal context that captures suficient local dynamics while avoiding excessive smoothing or fragmentation. Beyond window scale, the results confirm the eficacy of the proposed APS descriptor, where both time and basepoint augmentations outperform vanilla PS in most cases. Although APS does not always deliver the best performance across all window sizes, it attains the lowest overall EER at the optimal window size of $w = 1 1$ , suggesting a strong synergistic efect at this particular scale.

Table 2. Performance comparison across diferent augmentation configurations and sliding-window sizes w.
<table><tr><td>EER%</td><td> $w = 5$ </td><td> $w = 7$ </td><td> $w = 9$ </td><td> $w = 1 1$ </td><td> $w = 1 3$ </td><td> $w = 1 5$ </td></tr><tr><td>PS</td><td>0.700</td><td>0.688</td><td>0.656</td><td>0.638</td><td>0.604</td><td>0.662</td></tr><tr><td> $\mathrm { T A } { + } \mathrm { P S }$ </td><td>0.696</td><td>0.676</td><td>0.628</td><td>0.631</td><td>0.619</td><td>0.616</td></tr><tr><td> $\mathrm { B A } { + } \mathrm { P S }$ </td><td>0.705</td><td>0.673</td><td>0.608</td><td>0.635</td><td>0.606</td><td>0.640</td></tr><tr><td>APS</td><td>0.677</td><td>0.661</td><td>0.626</td><td>0.555</td><td>0.625</td><td>0.625</td></tr></table>

We also compared APS performance under diferent truncation orders of the path signature, as illustrated in Fig. 5. The results show that the second-order path signature achieves the best performance across all window sizes, followed by the third-order, while the first-order performs the worst. This trend confirms that higher order terms can capture inter-channel interactions that are crucial for efective signature verification. However, increasing the order beyond N = 2 adds unnecessary noisy details that degrade performance.

![](images/dc8e2bc648034084fd86800642ec8660b19a3de0fb861674e124a1c4c1a15f61.jpg)  
Fig. 5. Performance comparison across diferent orders of the path signature in the APS descriptor.

## 5.3 Evaluation of the Proposed Method

To evaluate computational eficiency, we compared the proposed time-scanning Mamba with the standard Mamba and Transformer architectures on MCYT-100 under S\_05, as shown in Table 3. Compared with the Transformer, the Mamba variants reduce the number of parameters, FLOPs, and runtime GPU memory consumption by more than 60%, while maintaining comparable training speed. In addition, the proposed time-scanning mechanism reduces the EER from 0.87% to 0.56%, demonstrating its enhanced modelling capability.

Table 3. Comparison of computational eficiency and verification performance on MCYT-100 under S\_05.
<table><tr><td>Model</td><td>Parameters (M)</td><td>FLOPs (G)</td><td>GPU Memory Usage (MB)</td><td>Time per Epoch (s)</td><td>EER (%)</td></tr><tr><td>Time-scanning Mamba</td><td>0.33</td><td>0.29</td><td>3,902</td><td>21.6</td><td>0.56</td></tr><tr><td>Mamba</td><td>0.33</td><td>0.29</td><td>3,452</td><td>18.5</td><td>0.87</td></tr><tr><td>Transformer</td><td>0.83</td><td>0.73</td><td>11,250</td><td>19.0</td><td>0.83</td></tr></table>

We conducted a cross-ablation study to validate the efectiveness of the APS descriptor and the T-Mamba model on MCYT-100 under S\_05, SVC-2004 Task 2 under S\_05, and DeepSignDB. The path signature truncation order is 2 and the sliding-window size is 11. We adopt DsDTW with CRAN [15] as a strong baseline, as it achieved first place in the ICDAR 2021 Competition on On-Line Signature Verification [35]. Table 4 shows that T-Mamba consistently outperforms CRAN on all datasets, with or without APS. Incorporating APS further improves performance and stabilises results for both models, as reflected in the lower standard deviations. These findings confirm that the APS descriptor provides robust feature representations, and T-Mamba serves as a strong discriminative backbone for OSV. The performance of CRAN on DeepSignDB suggests that APS may be sensitive to the noise inherent in finger-written signatures, which CRAN does not suficiently suppress. By contrast, T-Mamba’s selective state modelling exploits path signature features more efectively, resulting in improved performance.

Table 4. Ablation study on the efect of APS and a comparison between T-Mamba and CRAN (EER%). ‘w’ and $\mathbf { \bar { \tau } } _ { \mathbf { W } } / \mathbf { o } ^ { \prime }$ denote ‘with’ and ‘without’.
<table><tr><td rowspan="2">Method</td><td rowspan="2">APS</td><td rowspan="2">MCYT-100</td><td rowspan="2">SVC-2004 Task 2</td><td colspan="2">DeepSignDB</td></tr><tr><td>Stylus</td><td>Finger</td></tr><tr><td rowspan="2">T-Mamba</td><td> $\mathrm { \bf w }$ </td><td> ${ \bf 0 . 5 6 \pm 0 . 0 4 }$ </td><td> ${ \bf 1 . 3 9 \pm 0 . 2 3 }$ </td><td> $\mathbf { 0 . 7 6 \pm 0 . 0 5 }$ </td><td> ${ \bf 3 . 2 0 \pm 0 . 1 5 }$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o }$ </td><td> $0 . 9 3 \pm 0 . 0 8$ </td><td> $2 . 3 0 \pm 0 . 2 5$ </td><td> $0 . 9 2 \pm 0 . 0 8$ </td><td> $3 . 6 7 \pm 0 . 1 8$ </td></tr><tr><td rowspan="2">CRAN</td><td> $\mathrm { \bf w }$ </td><td> $0 . 6 7 \pm 0 . 0 6$ </td><td> $2 . 2 5 \pm 0 . 3 5$ </td><td> $0 . 8 8 \pm 0 . 0 8$ </td><td> $4 . 2 3 \pm 0 . 3 2$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o }$ </td><td> $1 . 1 5 \pm 0 . 0 7$ </td><td> $3 . 1 1 \pm 0 . 4 2$ </td><td> $1 . 0 0 \pm 0 . 1 1$ </td><td> $3 . 4 6 \pm 0 . 3 9$ </td></tr></table>

We compared our framework against other state-of-the-art methods on MCYT 100 across skilled (S\_05, S\_10, S\_15) and random (R\_05, R\_10) forgery configurations. As summarized in Table 5, our method achieves state-of-the-art results across all skilled forgery protocols, yielding EERs of 0.56%, 0.45%, and 0.43%, respectively. This performance scales robustly with training sample volume, indicating enhanced generalization. For random forgeries, the framework attains competitive EERs of 0.04% (R\_05) and 0.03% (R\_10). While Probabilistic-DTW [2] reports 0.01% on R\_05, its lack of evaluation on other protocols limits comprehensive comparison. In particular, Few-shot learning [37] performs poorly under skilled forgeries despite reasonable random-forgery EERs, suggesting that skilled forgery is the key challenge on MCYT-100 and the proposed method addresses it efectively.

Similar experiments were conducted on SVC-2004 Task 2 under S\_05, S\_10, R\_05, R\_10. As shown in Table 6, the proposed method delivers the best or nearbest performance across the four settings. It achieves the lowest EER on S\_10 and attains perfect separation in both random-forgery cases. On the more challenging skilled-forgery setting S\_05 with fewer references, the proposed method ranks the second, surpassed only by Few-shot learning [37], which is specifically designed for low-shot training. Notably, increasing the number of genuine training samples from 5 to 10 leads to a substantial improvement in the skilled scenario (1.39% to 0.25%), indicating that the model efectively leverages additional training data.

## 5.4 Experiments with DeepSignDB

To further assess the efectiveness and generalization capability of the proposed method, we conducted experiments on DeepSignDB, currently the largest public dataset for OSV, following the standard protocol [34]. We adopt DsDTW [15] as a strong baseline, which won first place in the ICDAR 2021 Competition on On-Line Signature Verification [35]. The $\mathrm { E E R } _ { \mathrm { g l o b a l } }$ is computed based on a uniform threshold for all writers in the subset. The overall $\mathrm { E E R } _ { \mathrm { g l o b a l } }$ is computed using a single global threshold shared by all writers within the Stylus and Finger subsets, respectively. In Table 7, the $\mathrm { E E R _ { g l o b a l } s }$ for DsDTW are taken directly from [15]. The writer-specific EERs for DsDTW are not reported in [15] and were obtained by running their publicly available code. Here we tune the window size w on each subset for optimal performance.

Table 5. Comparison between the proposed method and other state-of-the-art methods on MCYT-100 (EER%).
<table><tr><td>Method</td><td>S_05 S_10</td><td></td><td>S_15</td><td>R_05</td><td>R_10</td></tr><tr><td>Proposed Method</td><td>0.56</td><td>0.45</td><td>0.43</td><td>0.04</td><td>0.03</td></tr><tr><td>Time-series averaging + DTW [26]</td><td>0.72</td><td></td><td></td><td>0.07</td><td></td></tr><tr><td>SynSig2Vec [19]</td><td>0.93</td><td></td><td></td><td></td><td></td></tr><tr><td>DTW and warping path-based features [32]</td><td>1.15</td><td>1.08</td><td>0.84</td><td>0.13</td><td></td></tr><tr><td>Mean templates and multiple DTW distances [25]</td><td>1.28</td><td></td><td></td><td></td><td></td></tr><tr><td>Enhanced contextual DTW [31]</td><td>1.55</td><td></td><td></td><td></td><td></td></tr><tr><td>DsDTW with CRAN [15]</td><td>1.01</td><td></td><td></td><td></td><td></td></tr><tr><td>Modified DTW with signature curve constraint [39]</td><td>2.17</td><td></td><td></td><td></td><td></td></tr><tr><td>Interval valued symbolic [12]</td><td>2.20</td><td></td><td></td><td>1.00</td><td></td></tr><tr><td>TSOSVNet [36]</td><td>3.07</td><td>1.23</td><td>0.45</td><td></td><td></td></tr><tr><td>SM-DTW using distance normalization [28]</td><td>3.09</td><td>2.25</td><td></td><td>1.30</td><td></td></tr><tr><td>Feature fusion based on deep learning [38]</td><td>3.02</td><td>1.83</td><td>1.25</td><td>0.42</td><td>0.10</td></tr><tr><td>DTW and sigma-lognormal analysis [8]</td><td>3.56</td><td></td><td></td><td>1.01</td><td></td></tr><tr><td>Few-shot learning [37]</td><td>7.03</td><td>5.70</td><td>3.95</td><td>0.05</td><td>0.06</td></tr><tr><td>Probabilistic-DTW [2]</td><td></td><td></td><td></td><td>0.01</td><td></td></tr></table>

Table 6. Comparison between the proposed method and other state-of-the-art methods on SVC-2004 Task 2 (EER%).
<table><tr><td>Method S_05 S_10 R_05 R_10</td></tr><tr><td></td></tr><tr><td>Proposed Method 1.39 0.25 0.00 0.00 0.87 0.35 1.40 0.15</td></tr><tr><td>Few-shot learning [37] DsDTW with CRAN [15] 1.62</td></tr><tr><td>1.70 0.75</td></tr><tr><td>TSOSVNet [36]</td></tr><tr><td>Time-series averaging + DTW [26] 2.08 1.53 0.11 0.03</td></tr><tr><td>DTW and warping path-based features [32] 2.53 2.79 0.00</td></tr><tr><td>Mean templates and multiple DTW distances [25] 2.98 1.80</td></tr><tr><td>Enhanced contextual DTW [31] 2.73</td></tr><tr><td>Modified DTW with signature curve constraint [39] 2.60</td></tr><tr><td>SynSig2Vec [19] 2.63</td></tr><tr><td>Feature fusion based on deep learning [38] 3.93 2.98 0.45 0.39</td></tr><tr><td>Stroke point warping [16] 1.00</td></tr><tr><td>Probabilistic-DTW [2] 0.00</td></tr></table>

As shown in Table 7, the proposed method either outperforms or remains comparable to DsDTW on most acquisition devices, achieving lower overall $\mathrm { E E R _ { g l o b a l } s }$ and EERs. Notably, in the Stylus setting, our method attains an EER of 0.76%, yielding a 24% relative error reduction over DsDTW. Despite minor fluctuations in a few configurations, the aggregate gains across diverse subsets demonstrate strong cross-dataset and cross-device generalization against pronounced device variability. These improvements are primarily driven by the synergy between APS, which captures robust local geometric structures, and T-Mamba, which models bidirectional long-range dependencies. Furthermore, unlike the sequential, recurrence-based CRAN in DsDTW [15], T-Mamba leverages a hardware-friendly parallel execution mechanism, significantly enhancing backbone eficiency.

Table 7. Comparison of DsDTW and the proposed method on DeepSignDB in terms of EER<sub>global</sub>% and EER%.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Dataset</td><td rowspan="2">Device</td><td rowspan="2">w</td><td colspan="2"> $\mathrm { E E R _ { g l o b a l } }$ </td><td colspan="2">EER</td></tr><tr><td></td><td>DsDTW Ours</td><td></td><td>DsDTW Ours</td></tr><tr><td rowspan="13">Stylus</td><td>MCYT</td><td>Wacom Intuos A6</td><td>11</td><td>1.86</td><td>1.99</td><td>1.01</td><td>0.80</td></tr><tr><td>BiosecurID</td><td>Wacom Intuos 3</td><td>13</td><td>0.95</td><td>0.98</td><td>0.33</td><td>0.21</td></tr><tr><td>Biosecure DS2 Wacom Intuos 3</td><td></td><td>5</td><td>2.66</td><td>2.89</td><td>1.59</td><td>1.50</td></tr><tr><td>e-BioSign DS2 Wacom STU-530</td><td></td><td>11</td><td>0.71</td><td>0.71</td><td>0.25</td><td>0.19</td></tr><tr><td></td><td>Wacom STU-500</td><td>10</td><td>3.54</td><td>1.79</td><td>0.14</td><td>0.71</td></tr><tr><td rowspan="4"></td><td>Wacom STU-530</td><td>10</td><td>3.24</td><td>1.67</td><td>0.90</td><td>0.00</td></tr><tr><td>e-BioSign DS1 Wacom DTU-1031</td><td>10</td><td>4.44</td><td>3.67</td><td>1.33</td><td>0.90</td></tr><tr><td>Samsung ATIV 7</td><td>10</td><td>4.14</td><td>2.19</td><td>2.20</td><td>0.38</td></tr><tr><td>Samsung Galaxy Note 10.110</td><td></td><td>5.08</td><td>4.67</td><td>1.11</td><td>1.38</td></tr><tr><td rowspan="4">Finger</td><td rowspan="2">Overall EER e-BioSign DS1</td><td></td><td></td><td>2.54</td><td>2.44</td><td>1.00</td><td>0.76</td></tr><tr><td>Samsung ATIV 7</td><td>17 11.45</td><td></td><td>11.32</td><td>7.45</td><td>7.19</td></tr><tr><td rowspan="2"></td><td>Samsung Galaxy Note 10.117</td><td>9.57</td><td>6.89</td><td>5.11</td><td></td><td>4.71</td></tr><tr><td>Samsung Galaxy Note 10.1 5</td><td>5</td><td>2.53</td><td>0.36</td><td>0.92</td><td>0.25</td></tr><tr><td rowspan="2">e-BioSign DS2 Overall EER</td><td rowspan="2"></td><td rowspan="2">Samsung Galaxy S3</td><td rowspan="2">4.29</td><td></td><td>3.76</td><td>0.36</td><td>0.64</td></tr><tr><td>6.99</td><td>6.10</td><td>3.46</td><td>3.20</td></tr></table>

## 6 Conclusion

We developed a novel OSV framework that integrates the augmented path signature (APS) descriptor with the T-Mamba model. Experimental results demonstrate that, with an appropriate window size and truncation order, APS captures complex inter-channel interactions and produces discriminative features. Furthermore, the T-Mamba model serves as a new backbone for OSV by integrating TCN blocks with a time-scanning Mamba to model local and global dependencies and leverage past and future contexts. Overall, our framework achieves state-ofthe-art performance on MCYT-100, SVC-2004 Task 2, and the largest dynamic signature database to date DeepSignDB. The strong performance observed on smaller datasets (e.g. e-BioSign) warrants further investigation, particularly in the context of few-shot learning.

## Acknowledgements

DY gratefully acknowledges the support of the National Natural Science Foundation of China (Young Scholar Grant 12201081), Chongqing University (Starting Grant 02080011044104, School of Mathematics and Statistics Basic Discipline Development Fund 0208005406001) and the Engineering and Physical Sciences Research Council (Programme Grant EP/S026347/1).

## References

1. Ahamed, M.A., Cheng, Q.: TSCMamba: Mamba meets multi-view learning for time series classification. Inf. Fusion 120, 103079 (2025). https://doi.org/10.1016/ j.infus.2025.103079

2. Al-Hmouz, R., Pedrycz, W., Daqrouq, K., Morfeq, A., Al-Hmouz, A.: Quantifying dynamic time warping distance using probabilistic model in verification of dynamic signatures. Soft Comput. 23(2), 407–418 (2019). https://doi.org/10.1007/ s00500-017-2782-5

3. Bai, S., Kolter, J.Z., Koltun, V.: An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271 (2018), https: //arxiv.org/abs/1803.01271

4. Bhowal, P., Banerjee, D., Malakar, S., Sarkar, R.: A two-tier ensemble approach for writer dependent online signature verification. J. Ambient Intell. Humaniz. Comput. 13(1), 21–40 (2022). https://doi.org/10.1007/s12652-020-02872-5

5. Chen, K.T.: Integration of paths, geometric invariants and a generalized Baker– Hausdorf formula. Ann. Math. 65(1), 163–178 (1957). https://doi.org/10.2307/ 1969671

6. Chevyrev, I., Kormilitzin, A.: A primer on the signature method in machine learning. In: Signature Methods in Finance: An Introduction with Computational Applications, pp. 3–64. Springer Nature Switzerland, Cham (2026). https: //doi.org/10.1007/978-3-031-97239-3\_1

7. Cuturi, M., Blondel, M.: Soft-DTW: A diferentiable loss function for time-series. In: Proceedings of the 34th International Conference on Machine Learning. pp. 894–903. PMLR (2017), https://proceedings.mlr.press/v70/cuturi17a.html

8. Fischer, A., Plamondon, R.: Signature verification based on the kinematic theory of rapid human movements. IEEE Trans. Human-Mach. Syst. 47(2), 169–180 (2017). https://doi.org/10.1109/THMS.2016.2634922

9. Graham, B.: Sparse arrays of signatures for online character recognition. arXiv:1308.0371 (2013), https://arxiv.org/abs/1308.0371

10. Gu, A., Dao, T.: Mamba: linear-time sequence modeling with selective state spaces. In: First Conference on Language Modeling (2024), https://openreview.net/forum? id=tEYskw1VY2

11. Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., Ré, C.: Combining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in Neural Information Processing Systems 34, 572–585 (2021), https://proceedings.neurips.cc/paper/2021/hash/ 05546b0e38ab9175cd905eebcc6ebb76-Abstract.htm

12. Guru, D.S., Manjunatha, K.S., Manjunath, S., Somashekara, M.T.: Interval valued symbolic representation of writer dependent features for online signature verification. Expert Syst. Appl. 80, 232–243 (2017). https://doi.org/10.1016/j.eswa.2017. 03.024

13. Hambly, B., Lyons, T.: Uniqueness for the signature of a path of bounded variation and the reduced path group. Ann. Math. 171(1), 109–167 (2010). https://doi.org/ 10.4007/annals.2010.171.109

14. Impedovo, D., Pirlo, G., Plamondon, R.: Handwritten signature verification: new advancements and open issues. In: 2012 International Conference on Frontiers in Handwriting Recognition. pp. 367–372. IEEE (2012). https://doi.org/10.1109/ ICFHR.2012.211

15. Jiang, J., Lai, S., Jin, L., Zhu, Y.: DsDTW: local representation learning with deep soft-DTW for dynamic signature verification. IEEE Trans. Inf. Forensics Security 17, 2198–2212 (2022). https://doi.org/10.1109/TIFS.2022.3180219

16. Kar, B., Mukherjee, A., Dutta, P.K.: Stroke point warping-based reference selection and verification of online signature. IEEE Trans. Instrum. Meas. 67(1), 2–11 (2018). https://doi.org/10.1109/TIM.2017.2755898

17. Kholmatov, A., Yanikoglu, B.: Identity authentication using improved online signature verification method. Pattern Recognit. Lett. 26(15), 2400–2408 (2005). https://doi.org/10.1016/j.patrec.2005.04.017

18. Lai, S., Jin, L.: Recurrent adaptation networks for online signature verification. IEEE Trans. Inf. Forensics Security 14(6), 1624–1637 (2019). https://doi.org/10. 1109/TIFS.2018.2883152

19. Lai, S., Jin, L., Lin, L., Zhu, Y., Mao, H.: SynSig2Vec: learning representations from synthetic dynamic signatures for real-world verification. In: Proceedings of the AAAI Conference on Artificial Intelligence. pp. 735–742. AAAI Press (2020). https://doi.org/10.1609/aaai.v34i01.5416

20. Lai, S., Jin, L., Yang, W.: Online signature verification using recurrent neural network and length-normalized path signature descriptor. In: 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR). vol. 1, pp. 400–405. IEEE (2017). https://doi.org/10.1109/ICDAR.2017.73

21. Li, C., Zhang, X., Lin, F., Wang, Z., Liu, J., Zhang, R., Wang, H.: A stroke-based RNN for writer-independent online signature verification. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 526–532. IEEE (2019). https://doi.org/10.1109/ICDAR.2019.00090

22. Lyons, T.J.: Diferential equations driven by rough signals. Rev. Mat. Iberoam. 14(2), 215–310 (1998). https://doi.org/10.4171/RMI/240

23. Martinez-Diaz, M., Fierrez, J., Krish, R.P., Galbally, J.: Mobile signature verification: feature robustness and performance comparison. IET Biom. 3(4), 267–277 (2014). https://doi.org/10.1049/iet-bmt.2013.0081

24. Melzi, P., Tolosana, R., Vera Rodríguez, R., Delgado Santos, P., Stragapede, G., Fiérrez, J., Ortega Garcia, J.: Exploring transformers for on-line handwritten signature verification. In: CEUR Workshop Proceedings. vol. 3517, pp. 58–64 (2023), https://ceur-ws.org/Vol-3517/paper5.pdf

25. Okawa, M.: Online signature verification using single-template matching with timeseries averaging and gradient boosting. Pattern Recognit. 102, 107227 (2020). https://doi.org/10.1016/j.patcog.2020.107227

26. Okawa, M.: Time-series averaging and local stability-weighted dynamic time warping for online signature verification. Pattern Recognit. 112, 107699 (2021). https: //doi.org/10.1016/j.patcog.2020.107699

27. Ortega-Garcia, J., Fierrez-Aguilar, J., Simon, D., Gonzalez, J., Faundez-Zanuy, M., Espinosa, V., Satue, A., Hernaez, I., Igarza, J.J., Vivaracho, C., Escudero, D., Moro, Q.I.: MCYT baseline corpus: A bimodal biometric database. IEE Proc. Vis. Image Signal Process. 150(6), 395–401 (2003), https://digital-library.theiet.org/ doi/10.1049/ip-vis:20031078

28. Parziale, A., Diaz, M., Ferrer, M.A., Marcelli, A.: SM-DTW: stability modulated dynamic time warping for signature verification. Pattern Recognit. Lett. 121, 113– 122 (2019). https://doi.org/10.1016/j.patrec.2018.07.029

29. Plamondon, R., Srihari, S.N.: Online and of-line handwriting recognition: a comprehensive survey. IEEE Trans. Pattern Anal. Mach. Intell. 22(1), 63–84 (2000). https://doi.org/10.1109/34.824821

30. Sakoe, H., Chiba, S.: Dynamic programming algorithm optimization for spoken word recognition. IEEE Trans. Acoust. Speech Signal Process. 26(1), 43–49 (1978). https://doi.org/10.1109/TASSP.1978.1163055

31. Sharma, A., Sundaram, S.: An enhanced contextual DTW based system for online signature verification using vector quantization. Pattern Recognit. Lett. 84, 22–28 (2016). https://doi.org/10.1016/j.patrec.2016.07.015

32. Sharma, A., Sundaram, S.: On the exploration of information from the DTW cost matrix for online signature verification. IEEE Trans. Cybern. 48(2), 611–624 (2018). https://doi.org/10.1109/TCYB.2017.2647826

33. Sundararajan, K., Woodard, D.L.: Deep learning for biometrics: a survey. ACM Comput. Surv. 51(3), 1–34 (2018). https://doi.org/10.1145/3190618

34. Tolosana, R., Vera-Rodriguez, R., Fierrez, J., Ortega-Garcia, J.: DeepSign: deep on-line signature verification. IEEE Trans. Biom. Behav. Identity Sci. 3(2), 229– 239 (2021). https://doi.org/10.1109/TBIOM.2021.3054533

35. Tolosana, R., Vera-Rodriguez, R., Gonzalez-Garcia, C., Fierrez, J., Rengifo, S., Morales, A., et al.: ICDAR 2021 competition on on-line signature verification. In: Document Analysis and Recognition – ICDAR 2021. Lecture Notes in Computer Science, vol. 12824, pp. 723–737. Springer, Cham (2021). https://doi.org/10.1007/ 978-3-030-86337-1\_48

36. V, C.S., Gautam, A., P, V., Sreeja, S., G, R.K.S.: TSOSVNet: Teacher-student collaborative knowledge distillation for online signature verification. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 742–751 (October 2023). https://doi.org/10.1109/ICCVW60793.2023.00082

37. Vorugunti, C.S., Gorthi, R.K.S., Pulabaigari, V.: Online signature verification by few-shot separable convolution based deep learning. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 1125–1130. IEEE (2019). https://doi.org/10.1109/ICDAR.2019.00182

38. Vorugunti, C.S., Pulabaigari, V., Gorthi, R.K.S.S., Mukherjee, P.: OSVFuseNet: online signature verification by feature fusion and depth-wise separable convolution based deep learning. Neurocomputing 409, 157–172 (2020). https://doi.org/10. 1016/j.neucom.2020.05.072

39. Xia, X., Song, X., Luan, F., Zheng, J., Chen, Z., Ma, X.: Discriminative feature selection for on-line signature verification. Pattern Recognit. 74, 422–433 (2018). https://doi.org/10.1016/j.patcog.2017.09.033

40. Yeung, D.Y., Chang, H., Xiong, Y., George, S., Kashi, R., Matsumoto, T., Rigoll, G.: SVC2004: first international signature verification competition. In: International Conference on Biometric Authentication. pp. 16–22. Springer (2004). https://doi.org/10.1007/978-3-540-25948-0\_3