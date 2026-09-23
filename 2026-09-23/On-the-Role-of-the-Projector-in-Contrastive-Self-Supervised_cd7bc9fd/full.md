# On the Role of the Projector in Contrastive Self-Supervised Learning: Last-Layer Rank Dynamics Drive Representation Quality ∗

Siladittya Manna Hong Kong Baptist University, Hong Kong, China <sup>†</sup> Indian Institute of Science, Bengaluru, India

Priyangshu Mandal Indian Institute of Technology Kharagpur, India

Umapada Pal

Indian Statistical Institute, Kolkata, India

Saumik Bhattacharya Indian Institute of Technology Kharagpur, India

smanna@hkbu.edu.hk, siladittyam@iisc.ac.in

priyangshu.mandal@kgpian.iitkgp.ac.in

umapada@isical.ac.in

saumik@ece.iitkgp.ac.in

## Abstract

The dimensional collapse of representations in self-supervised contrastive learning is an ever-present issue. One notable technique to prevent such a collapse of representations is using a multi-layered perceptron network called Projector. In several works, the projector has been found to heavily influence the quality of representations learned in a self-supervised contrastive pre-training task. However, the question still lingers. What role does the projector play? Assuming the projector mitigates dimensional collapse, what prevents the terminal layer of the base encoder from functioning as the projector in the absence of an explicit multi-layer perceptron (MLP) head? In this work, we intend to study what happens inside the projector by examining the rank dynamics of the same and the encoder through empirical study and analysis. Through mathematical analysis, we observe that the efect of rank reduction predominantly occurs in the last layer. Motivated by this insight, we propose a weight regularization strategy applied specifically to the last layer. We demonstrate that this targeted approach yields better performance than applying orthogonal weight regularization across the entire network (WeRank), both with and without a projector. Our method improves Top-1 accuracy by more than 1% on SimCLR on the ImageNet100 dataset and consistently outperforms baseline SimCLR variants on CIFAR datasets, supporting our interpretation of the projector’s role.

## 1 Introduction

Self-supervised learning aims to learn representations without any human annotations. Recent works like SimCLR (Chen et al., 2020a), MoCov2 (Chen et al., 2020b), DCL (Yeh et al., 2022), BYOL (Grill et al., 2020), Barlow Twins (Zbontar et al., 2021), etc. present frameworks which allow learning of representations which are similar for semantically similar samples. However, this objective may lead to a complete collapse of representations when the representations of all samples get mapped trivially to a single point in the representation space.

Various techniques, such as using a large batch size (Chen et al., 2020a), momentum encoder (Chen et al., 2020b; Grill et al., 2020), stop gradient (Chen & He, 2020), feature whitening (Bardes et al., 2022; Zbontar et al., 2021) and clustering (Caron et al., 2020), have been used to prevent the complete collapse of representations. However, contrastive self-supervised learning still sufers from dimensional collapse, where the embedding vectors only span a lower-dimensional subspace. Dimensional collapse occurs when the variance of information along some dimensions becomes insignificant. We avoid saying that variance will be zero because information content along any dimension can never be entirely zero in practical terms.

In Hua et al. (2021), the author discusses that dimensional collapse is mainly related to a strong correlation between information flowing through diferent dimensions. This challenging issue of dimensional collapse has also been addressed in works like Balestriero & LeCun (2022), RankMe (Garrido et al., 2023), DirectCLR (Jing et al., 2022) and WeRank (Pasand et al., 2024). These works also stress the importance of full rank representations for better performance on downstream tasks. However, WeRank does not provide any mathematical insight into the dimensional collapse of representation. In DirectCLR, the attempt at investigating the causes of the dimensional collapse is limited to toy examples, and only uses a truncated vector for training, leaving the last few dimensions non-trainable. This, however, is not fully capable of preventing dimensional collapse. We instead use the full output vector for both training and evaluation as well. Additionally, WeRank restricts the set of possible learnable functions to a subset satisfying the orthogonality condition of the weight matrices, which causes an over-regularization efect and prevents learning of expressive features. We find several pieces of evidence in the literature, not limited to self-supervised learning, which state that anisotropy is essential for capturing non-uniform data structures, with neural rendering models using it to resolve directional ambiguity and disambiguate geometry from complex appearances for highfidelity 3D reconstructions (Wang et al., 2025; Gao et al., 2025). In parallel, research on Large Language Models demonstrates that decreasing isotropy via the I-STAR regularizer improves semantic performance by facilitating data clustering and reducing the intrinsic dimensionality of representations (Rudman & Eickhof, 2024). Recently, LeJePA (Balestriero & LeCun, 2025) also stresses that embeddings should be isotropic and that anisotropy amplifies both bias and variance. We show that not enforcing the feature maps to be isotropic via orthogonal weight regularisation, while only regularising the last layer, improves the weight spectrum and reduces the dimensional collapse efect in self-supervised contrastive learning when trained without a projector. Furthermore, we show that, unlike WeRank (Pasand et al., 2024), it is not necessary to apply the weight regularisation on the whole network, thereby reducing the computation overhead from $\mathcal { O } ( L \cdot n ^ { 3 } )$ to O(n<sup>3</sup>), where L is the scalar factor that comes naturally, as shown in the later section Sec. 3.6.

Even though the embedding covariance matrix remains ill-conditioned when a projector is used, we understand that the projector plays an important role in reducing dimensional collapse. The role of the projector has been studied previously in works like Gupta et al. (2022); Song et al. (2023); Xue et al. (2024). However, none of the above works explores what the fundamental causes are behind the efect of dimensional collapse.

In this work, we first empirically verify the decorrelating efect of InfoNCE loss. Next, we try to determine what happens inside the projector and its role in self-supervised contrastive learning. We further investigate the phenomenon occurring in the encoder layer that causes degradation in performance in the case of dimensional collapse resulting from negligible eigenvalues of the embedding covariance matrix, both with or without a projector. Finally, we employ a simple strategy for self-supervised learning, both with and without using a projector, which verifies our mathematical conclusion about the role of weight norm. We summarize our contributions as follows:

• We investigate the role of the projector in self-supervised contrastive learning in the light of dimensional collapse. To our knowledge, this is one of the first works to do so.

• We further investigate the phenomenon in the encoder when not using a projector for contrastive self-supervised pre-training, giving us more insight into the phenomenon of dimensional collapse.

• Based on our findings, we propose a simple strategy to improve performance in contrastive selfsupervised pre-training by applying weight regularization only on the last layer.

• The proposed regularization strategy outperforms the contemporary regularization strategy WeRank on benchmark datasets CIFAR10, CIFAR100 and ImageNet100.

## 2 Related Works

SSL methods take diferent approaches to prevent a complete collapse of representations. Instance discrimination methods like SimCLR (Chen et al., 2020a), MoCov2 (Chen et al., 2020b), and DCL (Yeh et al., 2022) use repulsion between negative samples to prevent complete collapse. However, in addition to the negative repulsion in the InfoNCE loss, they also use a projector which projects the encoder output representations into a lower-dimensional space before computing the InfoNCE loss. Methods like DeepCluster (Caron et al., 2018) and SwAV (Caron et al., 2020) use a clustering-based instance-group discrimination approach. However, dimensional collapse persists according to Garrido et al. (2023) and Jing et al. (2022).

Architectures similar to the above are also seen in dimension contrastive methods like BYOL (Grill et al., 2020), where the extra predictor for predicting the output of the projector from the momentum updated target encoder and l2-normalization prevents complete collapse. SimSiam (Chen & He, 2020), on the other hand, uses a stop-gradient method to prevent the same. WMSE (Ermolov et al., 2021), ZeroCL (Zhang et al., 2022b) uses feature whitening to prevent collapse.

Non-contrastive methods like Barlow Twins (Zbontar et al., 2021) aim to decorrelate the feature dimensions to reduce redundancy in the output embeddings, thereby preventing dimensional collapse. However, Barlow Twins fails to perform without the projector, as we will see in the later subsections. VICReg (Bardes et al., 2022) uses a covariance term in the loss to do feature decorrelation like Barlow Twins. However, according to Garrido et al. (2023), even these methods are not free from dimensional collapse.

Hua et al. (2021) discusses that the strong correlation between dimensions of the representation vector is the primary cause of dimensional collapse, and uses feature decorrelation to prevent it and improve performance. Balestriero & LeCun (2022) also uses a decorrelation loss like VICReg as a method to prevent dimensional collapse and learn optimal representations. Gupta et al. (2022) shows that a projector prevents low-rank backbone features, thereby preventing dimensional collapse. However, it does not explore the reason behind it. This work primarily discusses that a learnable projection head is a way of mitigating the shortcomings of contrastive loss and helps in learning generalizable representations. A detailed discussion of the relationship between downstream performance and embedding rank is also presented in Garrido et al. (2023). WeRank (Pasand et al., 2024) uses the same feature decorrelation strategy to deduce that the weight norm of each layer should be as close to the identity matrix as possible to prevent dimensional collapse.

DirectCLR (Jing et al., 2022) achieved considerable success in preventing collapse. This work mainly proposed two findings as the possible causes of dimensional collapse: (1) implicit regularization due to over-parametrization of networks, and (2) strong augmentations. However, in terms of performance (linear evaluation accuracy), it falls short of SimCLR with a non-linear projector.

DINO (Caron et al., 2021) introduced self-distillation without labels, where a student network learns from a momentum-updated teacher using normalized feature matching, enabling Vision Transformers to learn semantic features without supervision. iBOT (Zhou et al., 2022) extended DINO by combining selfdistillation with masked image modelling, allowing simultaneous global and local representation learning through patch-level prediction. DINOv2 (Oquab et al., 2024) further refined the framework with largescale curated data, improved regularization, and stronger transferability, yielding universal visual features competitive with supervised models. I-JEPA (Assran et al., 2023) departed from contrastive and pixel-leve objectives by predicting high-level latent representations of masked regions, emphasizing abstraction and context understanding. Together, these methods progressively evolve self-supervised learning from instance discrimination toward semantically rich, transferable, and predictive representations.

Other related works, such as VCReg (Zhu et al., 2023), extend the idea of weight and feature regularization by encouraging high-variance and low-covariance representations to enhance feature diversity and prevent neural collapse. This approach aligns conceptually with our method and WeRank (Pasand et al., 2024), as both aim to improve representation quality and transferability through more balanced and decorrelated feature learning.

## 3 Methodology

## 3.1 Preliminaries

In this work, we consider SSL pre-training with SimCLR as the baseline. Let us denote $f$ and g as the encoder and the projector, respectively. The encoder output and the projector output embeddings are denoted by $h = f ( x )$ and $z = g ( f ( x ) )$ ), respectively, where x denotes the input sample. The total number of dimensions in the encoder output embedding is given by D. $d _ { 0 }$ and $d _ { r }$ denote the number of dimensions of the encoder output embedding, which are trainable and non-trainable or fixed to a constant value. To learn the representations, the InfoNCE loss is given by,

$$
\mathcal { L } _ { i n f o n c e } = - \underset { i } { \mathbb { E } } \left[ \frac { e x p ( s _ { i i + } ) } { e x p ( s _ { i i + } ) + \sum _ { j = 1 } ^ { B } e x p ( s _ { i j } ) } \right]\tag{1}
$$

where $s _ { i i + }$ and $s _ { i j }$ are the cosine similarity between the projector output embeddings of the samples of positive pair $( x _ { i } , x _ { i + } )$ obtained by augmentations applied on the sample $x _ { i }$ , and the samples of negative pair $( x _ { i } , x _ { j } )$ , respectively, and B denotes the batch size.

In the later subsections, we divide the output embeddings into 2 parts, which we refer to as trainable and non-trainable dimensions. We define the trainable part of an embedding to consist of those dimensions through which the gradient propagation is allowed to flow. At the same time, the non-trainable part of the embedding means the opposite.

## 3.2 Definitions

(D1) Dimensional Collapse: Dimensional collapse occurs when representation vectors $\boldsymbol { h } \in \mathbb { R } ^ { D }$ are efectively constrained to an r-dimensional linear subspace $\mathbb { R } ^ { r } \subset \mathbb { R } ^ { D }$ with $r < \operatorname* { m i n } ( N , D )$ (N denotes the sample count), characterized by eigenvalues of the representation covariance matrix $\Sigma _ { h } = \operatorname { C o v } ( h )$ decaying to near-zero magnitudes $\left( \lambda _ { i } ( \Sigma _ { h } ) \leq \epsilon \right)$ for $i \in \{ r + 1 , \ldots , D \} ,$ ). Unlike complete collapse (where all embeddings map to a single point), dimensional collapse selectively suppresses variance along specific orthogonal directions, restricting representations to a low-rank subspace.

(D2) Information Bottleneck: The information bottleneck (IB) is a principle for representation learning that aims to extract a compressed representation of a variable that is as predictive as possible of a target variable. The IB framework is based on finding a compressed representation, T, of an input variable, X, that preserves the maximum relevant information about a target variable, Y. This is expressed through a constrained optimization problem. The goal is to find the representation T that maximizes the relevance $\mathcal { T } ( T ; Y )$ for a given compression level $\mathcal { I } ( T ; X )$ . This can be expressed as a Lagrangian optimization problem: m $\begin{array} { r } { \operatorname* { i n } _ { p ( t | x ) } \mathcal { I } ( T ; X ) - \beta \mathcal { I } ( T ; Y ) } \end{array}$ , where $\beta$ is the Lagrangian multiplier.

(D3) High-level representations: High-level representations refer to feature spaces that encode abstract, class-discriminative, and view-invariant semantic attributes of the input data while discarding task-irrelevant, low-level spatial variations. In deep architectures, feature extraction is hierarchical; earlier layers capture local spatial primitives, whereas the terminal layers (such as layer $L )$ map these primitives into high-level semantic representations. Consequently, weight degradation or rank collapse at layer L directly impairs the network’s capacity to output well-conditioned high-level features suitable for downstream tasks.

Relevance to Downstream Tasks: The principle of information bottleneck is utilised in the downstream task to discard irrelevant information while retaining useful information related to the downstream task. The high-level representations, that is, the task-specific representations in the deeper layers, are also relevant to the downstream task. Finally, the dimensional collapse, which can occur for both the self-supervised pre-training and supervised training stages, causes the learning to occur in a lower-dimensional subspace rather than in the high-dimensional embedding space. A larger utilisation of the learning subspace results in better performance in the downstream task.

## 3.3 Theoretical Setup and Scope

In this subsection, we formalize the modelling assumptions under which the theoretical analysis in the subsequent sections is carried out. This setup is not a repetition of the preliminaries in Sec. 3.1, but rather a restriction of scope that specifies which components of the network and training objective are explicitly modelled and which are abstracted away.

Architectural assumptions. We consider an InfoNCE-based contrastive SSL framework with a convolutional neural network (CNN) encoder f, optionally followed by a projection head $^ { g , }$ as introduced in Sec. 3.1. While a projector is not strictly required for all self-supervised learning paradigms (e.g., DirectCLR (Jing et al., 2022)), it is a standard architectural component in InfoNCE-based contrastive frameworks such as SimCLR and has been shown to play an important role in stabilising training and mitigating representation collapse. Accordingly, the presence of a projector is assumed in the general setup, while its absence is treated as a special case that is analyzed explicitly in later sections.

Layer-wise scope of analysis. Although the encoder f consists of multiple convolutional layers, our analysis is intentionally restricted to the final two convolutional layers of the encoder and the layers of the projector. All earlier layers are treated as a black box that produces intermediate feature representations with well-defined second-order statistics (i.e., finite covariance).

This restriction is motivated by empirical observations showing that dimensional collapse and rank degradation emerge most prominently at the layer whose output is directly optimized by the contrastive loss (shown in later sections). Consequently, all propositions in this paper concern the behaviour of: (i) the final encoder layer when no projector is used, or (ii) the projector layers when a projection head is present.

Throughout the paper, we denote by layer L the layer whose output embedding is used to compute the InfoNCE loss. In this work, layer l − 1 is termed a shallower layer with respect to l, while the layer l is deeper with respect to the layer l − 1. While there is no specific threshold in the literature to specify which layers are deep or shallow, by the term “deep” we will consider layers which are close to the final layer L, while “shallow” layers indicate those closer to the input layer.

Linearization and analytical simplifications. For analytical tractability, the layers under consideration (i.e., the last two encoder layers and the projector layers) are modelled as linear transformations parameterized by weight matrices. Non-linear activations, batch normalization, and skip connections are omitted from the theoretical analysis. This simplification allows us to isolate the structural relationship between embedding covariance, weight norms, and dimensional collapse.

A convolutional layer without non-linear activation defines a linear operator. Let $X \in \mathbb { R } ^ { C _ { \mathrm { i n } } \times H \times W }$ and $Y \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times H ^ { \prime } \times W ^ { \prime } }$ be the input and output feature maps, respectively. After vectorization, there exists a structured sparse matrix $\hat { W _ { \mathrm { c o n v } } } \in \mathbb { R } ^ { ( C _ { \mathrm { o u t } } \hat { H } ^ { \prime } W ^ { \prime } ) \times ( C _ { \mathrm { i n } } H W ) }$ such that vec $( Y ) = W _ { \mathrm { c o n v } } \mathrm { v e c } ( X )$ . The sparsity and block Toeplitz structure of $W _ { \mathrm { c o n v } }$ (Appendix Sec. C) arise from local connectivity and weight sharing, but do not afect the linear-algebraic arguments used in this work.

Importantly, this linearization is used only for theoretical reasoning; all empirical results in this paper are obtained using full CNN architectures with non-linearities and normalization layers intact.

Under this setup, the theoretical results in the following Sections should be interpreted as local, layerwise characterizations of dimensional collapse in InfoNCE-based contrastive learning, conditioned on the architectural and modelling assumptions stated above.

## 3.4 Motivation

In this work, the main motivation is to study the phenomenon occurring inside the projector in the selfsupervised contrastive learning scenario and what happens in the absence of it. In DirectCLR (Jing et al., 2022), it is stated that in instance discrimination-based contrastive learning, even though the presence of positive and negative samples should prevent the dimensional collapse of representations intuitively, it still occurs.

![](images/b3898de66b588f79612f7b68c85c84ab7e22037568568ae8cc33ef47c584176b.jpg)  
(a)

![](images/5273fd2b8be7ca9b65b98ddeb18bc61130a6d8ae39d8ca445202404750d09f7d.jpg)  
(b)  
Figure 1: (a) Singular value plots of the covariance matrix of ResNet50 encoder output embeddings pre-trained on ImageNet100 using SimCLR with and without a projector. ‘Layer4’ indicates the last layer in the ResNet50 encoder. ‘blue’: without (wo) projector, ‘red’:with projector. (b) Singular value plots of Barlow Twins and SimCLR encoders compared with DirectCLR. The plot (a) exhibits that without the projector, the singular values of the covariance matrix drop sharply. A similar observation is also found in Barlow Twins (b), while the vanilla SimCLR and DirectCLR method prevents the sharp decline.

![](images/a36c11fe89bd09cc7de4869bb577740ae3eb3228eea1c25c29be52c799e20f3b.jpg)  
(a) CIFAR100

![](images/ae34c6fb5f9219c113145a32be79096f9f3cb42b12cd1bc99019d5e3ed46aee8.jpg)  
(b) ImageNet100

![](images/45442b7c22b98506d9489dc6895e47ac6241157d65e8802020fe44f036d33d16.jpg)  
(c) CIFAR100

![](images/5db69d8cbffb587bd90cbb65db5fef2a009d49e2857ec5698d8c131aeb95be51.jpg)  
(d) ImageNet100  
Figure 2: Covariance matrices of output embedding of the projector for SimCLR trained on (a) CIFAR100 and (b) ImageNet100. Covariance matrices of embeddings from the SimCLR encoder trained on (c) CIFAR100 and (d) ImageNet100. On both CIFAR100 and ImageNet100, we can observe the projector output embedding exhibiting low covariance in the of-diagonal components, which points towards the decorrelation efects of the InfoNCE loss (a and b). Furthermore, the decorrelation efect is propagated partially to the encoder output embeddings, as evident from the uniform nature of the covariance matrix values (c and d). Best viewed at 300%.

We find this to be true empirically as shown in Fig. 1a, where we observe that the magnitudes of the sorted eigenvalue spectrum dip considerably when the encoder is trained without a non-linear projector than when trained with one. Similar findings are also reported in Gupta et al. (2022). Furthermore, methods using feature decorrelation to prevent dimensional collapse, like Balestriero & LeCun (2022) or Hua et al. (2021), still sufer from dimensional collapse. This is primarily due to the low-rank embeddings of shallower layers, that is, from the encoder (Pasand et al., 2024). To determine the role of the projector, we empirically study whether the InfoNCE loss has a decorrelating efect. Then we try to analyze the dynamics of the projector through rank decomposition of the covariance matrix and how it prevents dimensional collapse.

## 3.5 Does InfoNCE have a decorrelating efect?

According to Zhang et al. (2022a), InfoNCE also acts as a decorrelating loss, similar to Barlow Twins (Zbontar et al., 2021) or Balestriero & LeCun (2022). In Fig. 2a and 2b, we show the covariance matrix of the output feature dimensions. From the covariance matrix of the embeddings of the CIFAR100 and ImageNet100 datasets, we can see that the magnitudes of the diagonal elements of the covariance matrix are much higher than the non-diagonal ones. This shows that the InfoNCE loss has a decorrelating efect, as shown in Zhang et al. (2022a). However, from Fig. 2c and 2d, we see that the diagonal nature of the covariance matrix of the encoder output embeddings is not present. This proves that even if the loss enforces feature decorrelation on the projector output embeddings, it is possible to obtain low-rank output embeddings from the encoder.

## 3.6 Understanding the events in Projector in case of Dimensional Collapse

It is empirically observed in DirectCLR (Jing et al., 2022) that InfoNCE loss fails to properly optimize the parameters of a network without a projector and results in dimensional collapse. Barlow Twins (BT) (Zbontar et al., 2021) performs better than most contrastive learning frameworks on benchmark datasets, but not when implemented without a projector, even though a decorrelation loss is applied. The singular value spectrum in Fig. 1b plots the singular values of BT $( \mathrm { w } / \mathrm { o }$ projector), SimCLR $( \mathrm { w } / \mathrm { o }$ projector), and DirectCLR. We can see that the dimensional collapse efect in BT is greater than in SimCLR $\mathrm { w / o }$ projector, even though it is trained directly using a decorrelation-based loss. According to Zhang et al. (2022a), the projector is essential for a decorrelation-based framework too, even though both InfoNCE and the loss used in Zbontar et al. (2021) have a decorrelating component. So, the question arises, what exactly happens after the addition of a non-linear projector towards the prevention of dimensional collapse?.

A Linear Algebraic perspective: In RankMe (Garrido et al., 2023) and DirectCLR (Jing et al., 2022), the authors have shown that without a projector, the embeddings from the pre-trained encoder have a low rank. Does a low-rank embedding indicate that the useful information can be approximated using fewer dimensions? But then, why does it lead to worse performance, if that is the case?. How is it diferent from the information bottleneck theory of the projector? (Ouyang et al., 2025).

## 3.6.1 Weight Norm Analysis

It is important to note that in DirectCLR (Jing et al., 2022), a part of the output vector z is left unchanged; that is, the kernels leading to the unchanged part of z still have the randomly initialized weights at the end of pre-training. Now, these randomly initialized weights have non-zero variance. However, when the rank of the encoder output embeddings is reduced, it practically means that the variance of the information along those dimensions is very low. A very low variance means there is almost no useful information available in that dimension. Thus, when not using a projector, the reduction in rank in the last layer embedding covariance matrix indicates that there is little variance of information along some of the embedding dimensions. Consequently, assuming that the input to the last layer is full rank and well-conditioned, it indicates that the norm of the weights in the last layer $\| W _ { l } ^ { i } \| ^ { 2 } < \epsilon ,$ where ϵ is very small, and $\lambda _ { i } ( C o v ( h ) )  0$ , where $W _ { l } ^ { i }$ is the layer weights corresponding to the i-th output embedding dimension of layer l.

Proposition 1 (Collapsed dimensions imply vanishing weight norms). Let $\boldsymbol { x } \in \mathbb { R } ^ { D _ { i } }$ be a random vector with covariance $\Sigma _ { x } : = \operatorname * { C o v } ( x ) \succ 0$ . Let $W \in \bar { \mathbb { R } ^ { D _ { o } \times D _ { i } } }$ be a linear map and define $h = W x ,$ with $h _ { i } = W _ { i } x$ , where $W _ { i }$ denotes the i-th row of W. Furthermore, let $\mathrm { C o v } ( h ) = \Sigma _ { h } = P \Lambda P ^ { T }$ be the spectral decomposition of the representation covariance matrix, where $P = [ p _ { 1 } , \dots , p _ { D _ { o } } ] \in \mathbb { R } ^ { D _ { o } \times D _ { o } }$ is an orthogonal matrix of eigenvectors $( \bar { P ^ { T } } P = P P ^ { T } = I _ { D _ { o } } )$ and $\Lambda = \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { D _ { o } } )$ is the diagonal matrix of eigenvalues. Then, an eigenvalue $\lambda _ { i } ( \Sigma _ { h } ) = 0 \ i j$ and only if $W ^ { T } p _ { i } = 0$ , where $p _ { i }$ is the i-th eigenvector (column) of P. Consequently, dimensional collapse $\left( \lambda _ { i } ( \Sigma _ { h } ) = 0 \right)$ cannot be induced by X and is driven solely by W.

Proof: We can prove the above proposition by a simple deduction. Let $x \in \mathbb { R } ^ { D _ { i } }$ be the embeddings with dimensions $N \times D _ { i }$ , and $W \in \mathbb { R } ^ { \hat { D _ { o } } \times \hat { D _ { i } } }$ be the weight matrix with dimensions $D _ { o } \times D _ { i }$ . To consider only a single dimension $i ,$ we take the i-th row of the weight matrix as $W ^ { i } \in \mathbb { R } ^ { 1 \times D _ { i } }$ . Let the covariance of x be

defined as

$$
\Sigma _ { x } = C o v ( x ) = \underset { x } { \mathbb { E } } \left[ \left( x - \underset { x } { \mathbb { E } } [ x ] ) \right) \left( x - \underset { x } { \mathbb { E } } [ x ] ) \right) ^ { T } \right]\tag{2}
$$

It is to be noted that, since we are dealing with the last layer only, we assume that the input to the last layer does not have any collapsed dimension, and $\Sigma _ { x }$ is positive definite with the smallest eigenvalue $\lambda _ { m i n } > 0$ and is well conditioned, that is $\lambda _ { m i n } \approx \lambda _ { m a x } > 0 .$

$$
h _ { i } = W _ { l } ^ { i } x _ { l } = W _ { l } ^ { i } . \left( W _ { l - 1 } x _ { l - 1 } \right) = \left( W _ { l } ^ { i } W _ { l - 1 } \right) x _ { l - 1 }\tag{3}
$$

Because $\Sigma _ { x } = \operatorname { C o v } ( X ) \in \mathbb { R } ^ { D _ { i } \times D _ { i } }$ is real, symmetric, and Positive Definite $( \Sigma _ { x } \succ 0 )$ , its eigenvalues are strictly positive:

$$
\lambda _ { \operatorname* { m a x } } ( \Sigma _ { x } ) \geq \cdots \geq \lambda _ { i } ( \Sigma _ { x } ) \geq \cdots \geq \lambda _ { \operatorname* { m i n } } ( \Sigma _ { x } ) > 0\tag{4}
$$

The covariance of the representation $h = W X$ is given by:

$$
\operatorname { C o v } ( h ) = \operatorname { C o v } ( W x ) = W \operatorname { C o v } ( x ) W ^ { \top } = W \Sigma _ { x } W ^ { \top }\tag{5}
$$

Expressing $\operatorname { C o v } ( h )$ via its spectral decomposition $\Sigma _ { h } = \mathrm { C o v } ( h ) = P \Lambda P ^ { \intercal }$ , we establish the matrix identity:

$$
\boldsymbol { P } \boldsymbol { \Lambda } \boldsymbol { P } ^ { \intercal } = \boldsymbol { W } \boldsymbol { \Sigma } _ { x } \boldsymbol { W } ^ { \intercal }\tag{6}
$$

Premultiplying both sides by $P ^ { \top }$ and postmultiplying by $P ,$ and using the orthogonality property $P ^ { \top } P =$ $P P ^ { \top } = \bar { I _ { d } }$ , we isolate the diagonal eigenvalue matrix $\Lambda ^ { \prime }$

$$
\Lambda = P ^ { \top } W \Sigma _ { x } W ^ { \top } P\tag{7}
$$

Since $\Lambda = P ^ { \top } \operatorname { C o v } ( h ) P$ is diagonal with entries $\Lambda _ { i i } = \lambda _ { i } ( \Sigma _ { h } )$ , and the i-th column of $P$ is the eigenvector $\mathbf { p } _ { i } .$ taking the (i, i)-th entry of both sides gives:

$$
\lambda _ { i } ( \Sigma _ { h } ) = \mathbf { p } _ { i } ^ { \top } W \Sigma _ { x } W ^ { \top } \mathbf { p } _ { i } = ( W ^ { \top } \mathbf { p } _ { i } ) ^ { \top } \Sigma _ { x } ( W ^ { \top } \mathbf { p } _ { i } )\tag{8}
$$

Define the projection vector $\mathbf { q } _ { i } \in \mathbb { R } ^ { n }$ as:

$$
\mathbf { q } _ { i } = W ^ { \top } \mathbf { p } _ { i }\tag{9}
$$

Substituting $\mathbf { q } _ { i }$ back into Equation 8 gives the quadratic form:

$$
\lambda _ { i } ( \Sigma _ { h } ) = \mathbf { q } _ { i } ^ { \top } \Sigma _ { x } \mathbf { q } _ { i }\tag{10}
$$

By the Rayleigh-Ritz Theorem (Trefethen & Bau, 2022), for any vector $\mathbf { q } _ { i } \in \mathbb { R } ^ { n }$ and positive-definite matrix $\Sigma _ { x } \succ 0 ;$

$$
\lambda _ { \operatorname* { m i n } } ( \Sigma _ { x } ) \| \mathbf { q } _ { i } \| _ { 2 } ^ { 2 } \leq \mathbf { q } _ { i } ^ { \top } \Sigma _ { x } \mathbf { q } i \leq \lambda _ { \operatorname* { m a x } } ( \Sigma _ { x } ) \| \mathbf { q } _ { i } \| _ { 2 } ^ { 2 }\tag{11}
$$

Combining Equations 10 and 11 yields the sandwich inequality:

$$
\lambda _ { \operatorname* { m i n } } ( \Sigma _ { x } ) \| \mathbf { q } _ { i } \| _ { 2 } ^ { 2 } \leq \lambda _ { i } ( \Sigma _ { h } ) \leq \lambda _ { \operatorname* { m a x } } ( \Sigma _ { x } ) \| \mathbf { q } _ { i } \| _ { 2 } ^ { 2 }\tag{12}
$$

We evaluate both directions of the logical equivalence $\lambda _ { i } ( \Sigma _ { h } ) = 0 \iff \mathbf { q } _ { i } = \mathbf { 0 }$

(⇐) Suficiency: If $\mathbf q _ { i } = \mathbf 0$ , then $\lambda _ { i } ( \Sigma _ { h } ) = \mathbf { 0 } ^ { \top } \Sigma _ { x } \mathbf { 0 } = 0 .$

(⇒) Necessity: Suppose $\lambda _ { i } ( \Sigma _ { h } ) = 0$ . From the lower bound in equation (12):

$$
\lambda _ { \operatorname* { m i n } } ( \Sigma _ { x } ) \lVert \mathbf { q } _ { i } \rVert _ { 2 } ^ { 2 } \leq 0\tag{13}
$$

Because $\Sigma _ { x } \succ 0$ , we have $\lambda _ { \operatorname* { m i n } } \bigl ( \Sigma _ { x } \bigr ) > 0$ . Since $\| \mathbf { q } _ { i } \| _ { 2 } ^ { 2 } \geq 0$ , inequality (13) forces:

$$
\| \mathbf { q } _ { i } \| _ { 2 } ^ { 2 } = 0 \implies \mathbf { q } _ { i } = \mathbf { 0 }\tag{14}
$$

Therefore,

$$
\lambda _ { i } ( \Sigma _ { h } ) = 0 \quad \Longleftrightarrow \quad { \bf q } _ { i } = { \bf 0 } \quad \Longleftrightarrow \quad W ^ { \top } { \bf p } _ { i } = { \bf 0 }\tag{15}
$$

Since $P$ is an orthogonal matrix, its i-th column $\mathbf { p } _ { i }$ is an orthonormal eigenvector satisfying $\| \mathbf { p } _ { i } \| _ { 2 } = 1 \neq 0$ The condition $W ^ { \top } { \bf p } _ { i } = { \bf 0 }$ requires the non-zero eigenvector $\mathbf { p } _ { i } \in \mathbb { R } ^ { D _ { o } }$ to lie in the nullspace of $W ^ { \top }$

$$
\mathbf { p } _ { i } \in \mathrm { n u l l } ( W ^ { \top } )\tag{16}
$$

By the Rank-Nullity Theorem, the dimension of this nullspace is:

$$
\dim ( \mathrm { n u l l } ( W ^ { \top } ) ) = d - \operatorname { r a n k } ( W )\tag{17}
$$

We analyze two distinct cases for $W { : }$

• Full Row Rank $( \mathrm { r a n k } ( W ) = d )$

The nullspace nul $\mathbf { l } ( W ^ { \top } ) = \{ \mathbf { 0 } \}$ . Since $\| \mathbf { p } _ { i } \| _ { 2 } = 1$ , no eigenvector can belong to nul $1 ( W ^ { \top } )$ . Thus, $\mathbf { q } _ { i } = W ^ { \top } \mathbf { p } _ { i } \neq \mathbf { 0 }$ for all i, which guarantees that $\lambda _ { i } ( A ) > 0$ for all $i = 1 , \ldots , d .$

• Rank-Deficient $( \operatorname { r a n k } ( W ) = r < d )$ :

The nullspace dimension is dim $( \mathrm { n u l l } ( W ^ { \top } ) ) = d - r > 0$ . Consequently, exactly $d - r$ orthogonal eigenvectors $\mathbf { p } _ { i }$ span this nullspace and satisfy $W ^ { \top } { \bf p } _ { i } = { \bf 0 }$ , forcing exactly $d - r$ eigenvalues to collapse $\left( \lambda _ { i } ( \Sigma _ { h } ) = 0 \right)$

Because $\lambda _ { \operatorname* { m i n } } ( \Sigma _ { x } ) > 0$ strictly prevents the positive-definite input distribution X from collapsing $\mathbf { q } _ { i } ^ { \top } \Sigma _ { x } \mathbf { q } _ { i } = 0$ for any non-zero $\mathbf { q } _ { i }$ , output eigenvalue collapse $\left( \lambda _ { i } ( \Sigma _ { h } ) = 0 \right)$ occurs if and only if $W ^ { \top } { \bf p } _ { i } = \dot { \bf 0 }$ . Hence, dimensional collapse is solely driven by W.

While the above proposition establishes that exact collapse $\left( \lambda _ { i } ( \Sigma _ { h } ) \ : = \ : 0 \right)$ on well-conditioned data is strictly weight-induced $\left( \mathrm { r a n k } ( W ) < D _ { o } \right)$ , real-world feature distributions frequently lie on low-dimensional submanifolds, yielding ill-conditioned input covariance matrices where $\lambda _ { \operatorname* { m i n } } ( \Sigma _ { x } ) = \epsilon \approx 0$

In this regime, the lower bound in Equation 11 collapses $( \lambda _ { \operatorname* { m i n } } ( \Sigma _ { x } ) \to 0 )$ , removing the guarantee that full-rank weights $( \mathrm { r a n k } ( W ) = D _ { o } )$ protect against representation collapse. To investigate how input ill-conditioning interacts with W to induce practical representation collapse $\left( \lambda _ { i } ( \Sigma _ { h } ) \leq \epsilon \right)$ , we relax the well-conditioned assumption on $\Sigma _ { x }$ and analyze the joint spectral alignment between W and the eigenspaces of $\Sigma _ { x }$

Key Takeaway 1: Weights with near-zero norm prevent learning of high-level representations, provided input is well-conditioned Thus, if the norm and the variance of the weights corresponding to the collapsed dimensions become close to zero, it prevents the kernels in the deeper layers in the backbone encoder from learning class-specific high-level representations, consequently hampering the downstream performance, as the presence of lower values in the weight variances reduces their representation learning capacity.

Empirical Justification: We look at the distribution of the channel-wise weight norms of the convolutional layers in the penultimate and last layer of the encoder (Fig. 3), trained using the SimCLR framework, but without a projector. We can observe that there are several samples in the distributions which have values very close to zero, indicating the presence of collapsed dimensions. Additionally, we also present the efect of applying a weight regularization in the last layer only when trained without a projector in the same figure.

1st Block - 2nd Conv  
1st Block - 2nd Conv  
2nd Block - 2nd Conv  
![](images/696111e46bcaa9eb5cb884edbcf0e5e7fcabcb6ceea12c7649964d4090be677e.jpg)

![](images/e7e1936189e516b10446205c70adca26b0644188d39eb81fd9d8bbaa000ae76c.jpg)

![](images/a33242c1270d690de12684e450b9fe0026ed0aec0a4dacc77adb93d19769e429.jpg)

![](images/baa5b9c3cb0e62d6e1549b263a948082571a33b762e0939e8147c4c7f5c5f46f.jpg)  
(a) CIFAR10 - Last Layer - (b) CIFAR10 - Last Layer - (c) CIFAR100 - Last Layer - (d) CIFAR100 - Last Layer - 1st Block - 2nd Conv 2nd Block - 2nd Conv 1st Block - 2nd Conv 2nd Block - 2nd Conv

Figure 3: Comparison of channel-wise weight norm distribution of SimCLR without projector trained without (blue) and with (red) our proposed weight regularization in the last layer for the penultimate and final layers of the projector. The distribution of weight norms of SimCLR without the projector shows that the weights of the second convolutional layer of the second ResNet block in the last layer go very close to 0.0. Whereas the proposed method can raise the minimum away from 0.0 despite having similar mean values. Best viewed at 200%.  
![](images/dc4e4023bdeed8ef92a6149ba6aabf4f506d97c04f8987fe1ffff2a8d40484fe.jpg)  
(a) CIFAR10 - Last Layer -

![](images/41165037f367067b7e97c1ba10af110ffeec8ff1e6b25f2f7384e1573412d06e.jpg)  
(b) CIFAR10 - Last Layer

![](images/b9303bd740ab7a89495dacf7464973e97b8890fa88f0be52e9bdf7528d2225f8.jpg)  
(c) CIFAR100 - Last Layer -

![](images/5eb81c2797ca2981c4cee355da61e817d7286ed9057e3a393048287c9e05c44a.jpg)  
(d) CIFAR100 - Last Layer - 2nd Block - 2nd Conv

Figure 4: Comparison of channel-wise unbiased variance of weight of SimCLR without projector trained without (blue) and with (red) our proposed weight regularization in the last layer for the penultimate and final layers of the projector. The distribution of unbiased variances of the weight of the SimCLR without the projector shows that the weights of the second convolutional layer of the second resnet block in the last layer go very close to 0.0. Whereas the proposed method can raise the minimum away from 0.0 despite having similar mean values. Best viewed at 200%.

The importance of these empirical findings lies in the fact that they directly corroborate the “Key Takeaway 1” mentioned previously. By definition, the last layer corresponds to learning the high-level and more complex representations. Having a negligible weight norm and a negligible variance reduces the representation learning capacity of the layers in concern. From Figs. 3 and 5b, we can combine the empirical results to infer that in the absence of a projector, both the covariance of the embedding dimensions and the weight norm become negligible along a few dimensions. Hence, considering that the concerned weights are from the last layer, we can safely state that the kernels in the deeper layers are incapable of learning useful information in the absence of a projector.

## 3.6.2 Propagation of Low-Rank Representations

In the case of a perfect decorrelating efect of the InfoNCE loss as discussed in the previous subsection, the flow of information would be maximized, resulting in better semantic representation learning and consequently better downstream performance. But the decorrelation efect of the InfoNCE on the encoder output embeddings does not maximize the flow of information through the encoder output dimensions. A similar approach to prevent dimensional collapse was also presented in WeRank (Pasand et al., 2024), where the authors used weight regularization by whitening the weight covariance matrix.

![](images/82e572e16f62e92920574e1b9dfaaf1a313e08fb1b78f76b9c5ad6b4d3778b3f.jpg)  
(a) CIFAR10

![](images/4f4d26e87feee4c2fc3e396a3a652331a9c7f4ec25e51991087b3bf17e572ffd.jpg)  
(b) CIFAR100

![](images/84f7d05f2e1b4e9cdbac66a6a46b6361d7dbcca6e159e5179107b556fb2fe49f.jpg)  
(c) ImageNet100  
Figure 5: Singular values plots of the penultimate and final layers of the encoder with and without the projector. The plots above show the efect of the presence and absence of the projector on the last and the penultimate layer of the encoder. The last layer (Layer5) shows a clear dimensional collapse without the projector in all three datasets, CIFAR10, CIFAR100 and ImageNet100. However, there is a negligible change in the singular value plot for the penultimate layer (Layer4), which agrees with our previous observation that the decorrelation efect is not fully propagated towards the shallower layers. Best viewed at 300%.

Empirical justification and observations : We study the singular value plots of the penultimate and final layer of the encoder backbone in Fig. 5. We see an increase in the number of high-valued singular values in the final layer of the backbone when using a projector. However, the eigenvalue spectrum is almost unchanged with or without a projector in the penultimate layer of the ResNet backbone encoder. Thus, we can say that the efect of rank reduction is observed only on the final layer, when not using a separate projector, and the final layer of the backbone acts as the make-shift or pseudo projector. However, according to our observation, it is wise to say that the rank reduction efect is observed on the layer from which the final embedding is taken for contrastive loss computation, while the eigenspectrum of the previous layer output embeddings shows almost no change. This is confirmed from the plots of eigenvalues in Fig. 6, where we observe that the eigenvalues for the last layer of the projector are significantly low in magnitude.

![](images/6abeabd1c32687a84485863d1bbeecfaec8c0f098cb515e70df24ab495781b5f.jpg)  
(a) CIFAR10

![](images/3c88b7a376db63bf87e5277bc457de4d728c0392d23b32bb69c93457b6f35508.jpg)  
(b) CIFAR100

![](images/b75885434db03d1b663b3a8e6d0205bfa1728019645b37a06912600530f706dc.jpg)  
(c) ImageNet100  
Figure 6: Singular values plots of the penultimate and final layers of the projector. Best viewed at 300%.

Why are full-rank embeddings better than low-rank embeddings? The trick used in DirectCLR is to leave a part of the output vector z randomly initialized, theoretically making the variance of those dimensions non-zero. This causes the encoder to learn representations that are efectively higher-dimensional. Therefore, separability is also better according to Cover’s theorem (Cover, 1965). Whereas, when the rank gets reduced, the representations are mapped to a low-dimensional subspace which should reduce the probability that the mapped instances are linearly separable while still being embedded in a high-dimensional space.

A common premise in the literature, often motivated by frameworks like RankMe (Garrido et al., 2023) is that higher-rank embeddings directly correlate with superior downstream performance. RankMe evaluates a within-method, cross-hyperparameter relationship between representation rank and linear evaluation performance within a single, consistent feature space. In contrast, evaluating pre-projector features $h \in \mathbb { R } ^ { D _ { h } }$ versus post-projector features $z \in \mathbb { R } ^ { D _ { z } }$ within the same trained network involves projections across spaces with fundamentally diferent ambient dimensions $( D _ { h } \neq D _ { z } )$ and distinct functional roles. The projection head intentionally maps h into z to isolate loss-specific decorrelation, whereas h retains a richer, multi-dimensional semantic hierarchy optimized for general linear separability. Because h and z reside in distinct ambient spaces, comparing their absolute ranks directly does not imply that z is a superior representation overall. Instead, rank dynamics across the pre- and post-projector boundary reflect geometric transformations between distinct feature spaces rather than a naive "higher rank implies better representation" rule.

Proposition 2. Let $x _ { l - 1 } \in \mathbb { R } ^ { D _ { l - 1 } }$ be a random vector with covariance $\Sigma _ { l - 1 } = \mathrm { C o v } ( x _ { l - 1 } ) \succ 0$ . Let $x _ { l } =$ $W _ { l - 1 } x _ { l - 1 } \in \mathbb { R } ^ { D _ { l } }$ have covariance $\Sigma _ { x _ { l } } = { W _ { l - 1 } \Sigma _ { l - 1 } W _ { l - 1 } ^ { \top } }$ with spectral decomposition $\begin{array} { r } { \Sigma _ { x _ { l } } = \sum _ { k = 1 } ^ { D _ { l } } \lambda _ { k } \mathbf { v } _ { k } \mathbf { v } _ { k } ^ { \top } } \end{array}$ where $\lambda _ { 1 } \ge \cdots \ge \lambda _ { D _ { l } } > 0$ . Let $h = W _ { l } x _ { l } \in \mathbb { R } ^ { D _ { o } }$ with output covariance $\Sigma _ { h } = \mathrm { C o v } ( h ) = W _ { l } \Sigma _ { x _ { l } } W _ { l } ^ { \top } \in \mathbb { R } ^ { D _ { o } \times D _ { o } }$ Suppose the i-th eigenvalue of $\Sigma _ { h }$ exhibits practical collapse, $\mathrm { i . e . , } \lambda _ { i } ( \Sigma _ { h } ) \leq \varepsilon$ . Then:

$\lambda _ { i } ( \Sigma _ { h } ) \leq \varepsilon$ does not imply that rank $\left( W _ { l - 1 } \right) < D _ { l }$ or that $\Sigma _ { l - 1 }$ is ill-conditioned.

• In particular, for any well-conditioned input $\Sigma _ { l - 1 } \succ 0$ and full-rank matrix $W _ { l - 1 } , \lambda _ { i } ( \Sigma _ { h } ) \leq \varepsilon$ occurs if and only if the projected eigenvector $\mathbf { w } _ { i } = W _ { l } ^ { \top } \mathbf { u } _ { i }$ satisfies:

$$
\sum _ { k = 1 } ^ { D _ { l } } \lambda _ { k } \left( \mathbf { u } _ { i } ^ { \top } W _ { l } \mathbf { v } _ { k } \right) ^ { 2 } \leq \varepsilon\tag{18}
$$

where $\mathbf { u } _ { i }$ is the i-th orthonormal eigenvector of $\Sigma _ { h }$

Proof: We assume that the network under consideration is consisted of two consecutive layers in a linear network without skip connections or activation functions:

$$
x _ { l } = W _ { l - 1 } x _ { l - 1 } , \quad h = W _ { l } x _ { l } = W _ { l } W _ { l - 1 } x _ { l - 1 }\tag{19}
$$

where $x _ { l - 1 } \in \mathrm { ~ \mathbb { R } ^ { \delta _ { l - 1 } } , ~ } x _ { l } \in \mathrm { ~ \mathbb { R } ^ { \delta _ { l } } ~ }$ and $\begin{array} { r l r } { h } & { { } \in } & { \mathbb { R } ^ { D _ { o } } } \end{array}$ are column vectors. Let $\begin{array} { r l } { \Sigma _ { l - 1 } } & { { } = } \end{array}$ $\mathbb { E } \left[ ( x _ { l - 1 } - \mathbb { E } [ x _ { l - 1 } ] ) ( x _ { l - 1 } - \mathbb { E } [ x _ { l - 1 } ] ) ^ { T } \right] \in \mathbb { R } ^ { D }$ l<sup>−</sup>1<sup>×D</sup>l<sup>−</sup>1 denote the covariance matrix of $x _ { l - 1 }$ . The representation covariance matrix $\Sigma _ { h } = \mathrm { C o v } ( h ) \in \mathbb R ^ { \vec { D } _ { o } \times D _ { o } }$ is given by:

$$
\Sigma _ { h } = W _ { l } W _ { l - 1 } \Sigma _ { l - 1 } W _ { l - 1 } ^ { T } W _ { l } ^ { T }\tag{20}
$$

Let $\Sigma _ { h } = U \Lambda U ^ { T }$ be the spectral decomposition of $\Sigma _ { h }$ , where $U = [ \mathbf { u } _ { 1 } , \dots , \mathbf { u } _ { D _ { o } } ] \in \mathbb { R } ^ { D _ { o } \times D _ { o } }$ is an orthogonal matrix of eigenvectors $( U ^ { T } U = I )$ and $\boldsymbol { \Lambda } = \operatorname { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { D _ { o } } )$ contains the output eigenvalues ordered as $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdot \cdot \cdot \geq \lambda _ { D _ { o } } \geq 0 .$

By Rayleigh-Ritz (Trefethen & Bau, 2022), the i-th eigenvalue $\lambda _ { i } ( \Sigma _ { h } )$ corresponds to the variance along the i-th principal component direction $\mathbf { u } _ { i } \mathbf { : }$

$$
\begin{array} { r } { \boldsymbol { \lambda } _ { i } ( \Sigma _ { h } ) = \mathbf { u } _ { i } ^ { T } \Sigma _ { h } \mathbf { u } _ { i } = \mathbf { u } _ { i } ^ { T } \left( W _ { l } W _ { l - 1 } \Sigma _ { l - 1 } W _ { l - 1 } ^ { T } W _ { l } ^ { T } \right) \mathbf { u } _ { i } = \left. \mathbf { u } _ { i } ^ { T } W _ { l } W _ { l - 1 } \Sigma _ { l - 1 } ^ { \frac { 1 } { 2 } } \right. _ { 2 } ^ { 2 } } \end{array}\tag{21}
$$

We can safely state that, practical dimensional collapse along the i-th principal component occurs when $\lambda _ { i } ( \Sigma _ { h } ) \leq \epsilon$ for a small threshold $\epsilon > 0$ . This yields:

$$
\left\| \mathbf { u } _ { i } ^ { T } W _ { l } W _ { l - 1 } \Sigma _ { l - 1 } ^ { \frac { 1 } { 2 } } \right\| _ { 2 } \leq \sqrt { \epsilon } = \epsilon ^ { \prime }\tag{22}
$$

The above Eqn. 22 reveals two distinct structural mechanisms through which collapse occurs:

Case 1 (Spectral Misalignment with Feature Covariance): Let $\begin{array} { r } { \Sigma _ { l - 1 } = V \Lambda _ { l - 1 } V ^ { T } = \sum _ { k = 1 } ^ { D _ { l - 1 } } \mu _ { k } \mathbf { v } _ { k } \mathbf { v } _ { k } ^ { T } } \end{array}$ be the spectral decomposition of $\Sigma _ { l - 1 }$ , where $\mu _ { k } \geq 0$ are the input eigenvalues and $\mathbf { v } _ { k }$ are the orthonormal eigenvectors. From Eqn. 21, we expand the expression for $\lambda _ { i } ( \Sigma _ { h } )$ yielding:

$$
\lambda _ { i } \bigl ( \Sigma _ { h } \bigr ) = \sum _ { k = 1 } ^ { D _ { l - 1 } } \mu _ { k } \left( \mathbf { u } _ { i } ^ { T } W _ { l } W _ { l - 1 } \mathbf { v } _ { k } \right) ^ { 2 } \leq \epsilon\tag{23}
$$

If $\lambda _ { i } ( \Sigma _ { h } ) \leq \epsilon .$ , the projected vector $\mathbf { u } _ { i } ^ { T } W _ { l } W _ { l - 1 }$ lies in the approximate left nullspace of $\Sigma _ { l - 1 } ^ { \frac { 1 } { 2 } } .$ denoted as ${ \mathbf { u } } _ { i } ^ { T } W _ { l } W _ { l - 1 } \ \in \ \mathrm { N u l l } _ { \epsilon ^ { \prime } } ( \Sigma _ { l - 1 } ^ { \frac { 1 } { 2 } } )$ . This occurs when $\mathbf { u } _ { i } ^ { T } W _ { l } W _ { l - 1 }$ is orthogonally misaligned with all dominant eigenvectors $\mathbf { v } _ { k }$ corresponding to large eigenvalues $\mu _ { k } \gg \epsilon$

Case 2 (Inter-Layer Weight Misalignment or Shrinkage): Applying the Rayleigh-Ritz bounds (Trefethen & Bau, 2022) with respect to the spectrum of $\Sigma _ { l - 1 } $

$$
\begin{array} { r } { \mu _ { \mathrm { m i n } } \big ( \Sigma _ { l - 1 } \big ) \left\| \mathbf { u } _ { i } ^ { T } W _ { l } W _ { l - 1 } \right\| _ { 2 } ^ { 2 } \leq \lambda _ { i } \big ( \Sigma _ { h } \big ) \leq \mu _ { \mathrm { m a x } } \big ( \Sigma _ { l - 1 } \big ) \left\| \mathbf { u } _ { i } ^ { T } W _ { l } W _ { l - 1 } \right\| _ { 2 } ^ { 2 } } \end{array}\tag{24}
$$

When $\Sigma _ { l - 1 }$ is well-conditioned $( \mu _ { \mathrm { m i n } } ( \Sigma _ { l - 1 } ) > 0 ) , \lambda _ { i } ( \Sigma _ { h } ) \leq \epsilon$ forces the efective weight norm to be bounded:

$$
{ \left\| { \bf { u } } _ { i } ^ { T } W _ { l } W _ { l - 1 } \right\| } _ { 2 } \le \sqrt { \frac { \epsilon } { \mu _ { \mathrm { { m i n } } } ( \Sigma _ { l - 1 } ) } } = \epsilon _ { l }\tag{25}
$$

This establishes that $\mathbf { u } _ { i } ^ { T } W _ { l }$ lies in the approximate left nullspace of $W _ { l - 1 } .$ , i.e., $\mathbf { u } _ { i } ^ { T } W _ { l } \in \mathrm { N u l l } _ { \epsilon _ { l } } ( W _ { l - 1 } )$

From these spectral derivations, dimensional collapse $( \lambda _ { i } ( \Sigma _ { h } ) \leq \epsilon )$ originates from three non-unique mechanisms:

1. Inter-Layer Weight Shrinkage: $\Vert \mathbf { u } _ { i } ^ { T } W _ { l } \Vert _ { 2 }$ is small, that is, vanishing weight norm along output eigenvector $\mathbf { u } _ { i }$

2. Inter-Layer Spectral Misalignment: $\mathbf { u } _ { i } ^ { T } W _ { l }$ aligns with the left approximate nullspace of $W _ { l - 1 }$

3. Feature Space Misalignment: $\mathbf { u } _ { i } ^ { T } W _ { l } W _ { l - 1 }$ aligns with the low-variance eigenspace of $\Sigma _ { l - 1 }$

Key Takeaway 2: Dimensional collapse does not necessarily imply rank-deficient representations or weights in shallower layers. A collapsed principal component mode of the representation at layer $l ,$ characterized by a near-zero output eigenvalue $( \lambda _ { i } ( \Sigma _ { h } ) \leq \epsilon )$ , arises from multiple, non-exclusive spectral mechanisms rather than a single structural bottleneck. First, even when the input covariance $\Sigma _ { l - 1 }$ and intermediate weights $W _ { l - 1 }$ are strictly full-rank and well-conditioned, collapse along an output principal direction $\mathbf { u } _ { i }$ can occur due to inter-layer spectral misalignment, where the projected weight vector $\mathbf { u } _ { i } ^ { T } W _ { l }$ falls into the approximate left nullspace of $W _ { l - 1 }$ . Second, collapse occurs when the composite transformation $\mathbf { u } _ { i } ^ { T } W _ { l } W _ { l - 1 }$ is spectrally misaligned with the dominant eigenspaces of the input covariance $\Sigma _ { l - 1 }$ , or when $W _ { l }$ undergoes directional weight shrinkage along $\mathbf { u } _ { i } .$ Consequently, observing a low-rank or collapsed representation spectrum at layer l does not automatically imply rank deficiency in shallower weight matrices $\left( W _ { l - 1 } \right)$ or input embeddings. This finding refutes the premise in WeRank (Pasand et al., 2024), which attributes collapse primarily to low-rank weights in earlier layers, demonstrating instead that representation collapse is an unidentifiable phenomenon co-driven by inter-layer spectral misalignment and feature covariance interactions.

Mechanisms of Representation Collapse. Prior work, like WeRank (Pasand et al., 2024), attribute output representation collapse primarily to rank-deficient or low-rank weight matrices in earlier layers. However, our 4-cause taxonomy demonstrates that observing a collapsed representation spectrum $( \lambda _ { i } ( \Sigma _ { h } ) \leq \epsilon )$ at layer l is fundamentally an unidentifiable phenomenon that cannot be uniquely mapped to shallow-layer rank deficiency. Specifically, while output collapse can indeed stem from weight magnitude shrinkage (Cause 1) or data-inherited degeneracy (Cause 4), it also routinely occurs when earlier weight matrices $W _ { l - 1 }$ and input feature covariances $\Sigma _ { l - 1 }$ are strictly full-rank and well-conditioned. In this regime, collapse is independently driven by geometric alignment mechanisms: inter-layer spectral misalignment, where the active layer projects onto the approximate left nullspace of $W _ { l - 1 }$ (Cause 2), or feature-space misalignment, where the composite transformation rotates orthogonally to the dominant eigenspace of $\Sigma _ { l - 1 }$ (Cause 3). By failing to disentangle geometric directional alignment from matrix rank, prior frameworks oversimplify the diagnostics of collapse; our theoretical results prove that output spectral collapse provides no necessary condition on the rank or conditioning of earlier-layer weights.

## 3.6.3 Does stopping information flow along some dimensions of the encoder output result in a better information bottleneck than using a Projector?

We attempt to substantiate our aforementioned statements more concretely and propose a straightforward approach to enhance performance on self-supervised contrastive learning tasks without requiring a projector. According to Property 5 described in Fang et al. (2024), a subvector of fixed value is the same as dimensional collapse along the dimensions of the subvector. We intend to stop the information flow, resulting in an enforced dimensional collapse to cause an information bottleneck without the projector by fixing the output of the embeddings along a few dimensions to a constant k. First, let us $_ \mathrm { g o }$ through the notations. We only keep $d _ { 0 }$ out of $\mathrm { a }$ total of D dimensions as dynamic, while making the rest $( D - d _ { 0 } = d _ { r } )$ static by assigning a constant output value to those dimensions of the embedding. Making the i-th dimension of the encoder output $h ,$ that is, $h _ { i } .$ , static with a constant $k = 0$ , stops the gradient flow through all paths connected directly or indirectly to $h _ { i } ,$ and the rank of the covariance matrix $\mathcal { C }$ follows Eqn. 26. However, the weights $W ^ { i }$ , which result in $h _ { i } .$ are still randomly initialized. Whereas, in DirectCLR, due to the randomized subvector, the rank of the covariance matrix does not collapse drastically (Eqn. 27).

$$
{ \mathrm { r a n k } } ( { \mathcal { C } } ) \leq d _ { 0 }\tag{26}
$$

$$
d _ { r } \le \mathrm { r a n k } ( \mathcal { C } ) \le d _ { r } + d _ { 0 }\tag{27}
$$

Thus, when a constant value is not assigned to the static $d _ { r }$ dimensions, the rank of the encoder output embedding h or, consequently, the random variable H is less than when the static $d _ { r }$ dimensions are left untouched.

Increasing the value of $d _ { 0 }$ reduces the explicit dimensional collapse enforced on the representation space by allowing the rank of the embedding covariance matrix to increase according to Eqn. 26. Let us denote the new value of $d _ { 0 }$ and $d _ { r }$ be indicated by $d _ { 0 } ^ { \prime }$ and $d _ { r } ^ { \prime }$ , where $d _ { 0 } ^ { \prime } > d _ { 0 }$ and $d _ { r } ^ { \prime } < d _ { r }$ . The rank of the new embedding covariance matrix also increases. So, we may think that we are efectively increasing the shattering capacity by (a) increasing the rank of the embedding covariance matrix, (b) decreasing the degree of dimensional collapse (Fang et al., 2024), and also (c) increasing the dimensionality of the representation learning subspace (Cover, 1965). However, this is not the case as observed from Table 1. The gradient of the InfoNCE loss $\mathcal { L }$ with respect to an embedding $z _ { i }$ is given by Eqn. 28.

$$
\frac { \partial \mathcal { L } } { \partial z _ { i } } = \left[ - \frac { z _ { i } + } { \tau } + \frac { \frac { z _ { i } + } { \tau } \cdot e ^ { s _ { i i } + } + \sum _ { j \neq i } ^ { N } \frac { z _ { j } } { \tau } \cdot e ^ { s _ { i j } } } { e ^ { s _ { i i } + } + \sum _ { j \neq i } ^ { N } e ^ { s _ { i j } } } + \sum _ { j = i } ^ { N } \frac { z _ { j } } { e ^ { s _ { j j } + } } \cdot e ^ { \frac { s _ { j i } } { \tau } }  \right] = - \left[ \frac { z _ { i } + } { \tau } \left( 1 - p ^ { i + } \right) - \sum _ { j = i \atop j \neq i } ^ { N } \frac { z _ { j } } { \tau } \left( p ^ { i j } + p ^ { i } \right) \right]\tag{28}
$$

The quantity $p ^ { i j }$ is the probability of the pair $( x _ { i } , x _ { j } )$ being predicted as a positive pair with the sample $x _ { i }$ as the anchor. The gradient along each dimension can be written as follows,

$$
\frac { \partial \mathcal { L } } { \partial z _ { i } ^ { d } } = \left\{ \begin{array} { l l } { - \frac { k } { \tau } \left[ \left( 1 - p ^ { i i ^ { + } } \right) - \sum _ { j = 1 } ^ { N } \left( p ^ { i j } + p ^ { j i } \right) \right] } & { \mathrm { f o r } ~ d _ { 0 } < d \leq d _ { 0 } + d _ { r } } \\ { - \left[ \frac { z _ { i + } ^ { d } } { \tau } \left( 1 - p ^ { i i ^ { + } } \right) - \sum _ { j = 1 } ^ { N } \frac { z _ { j } ^ { d } } { \tau } \left( p ^ { i j } + p ^ { j i } \right) \right] } & { \mathrm { f o r } ~ d \leq d _ { 0 } } \end{array} \right.\tag{29}
$$

Therefore, $\begin{array} { r } { \frac { \partial \mathcal { L } } { \partial z _ { i } ^ { d } } = 0 \mathrm { ~ i f ~ } k = 0 } \end{array}$ for the sub-vector $h [ d _ { 0 } : d _ { 0 } + d _ { r } ]$ , whereas the gradient flows normally through the rest of the dimensions. Using a constant other than 0 causes a small gradient to flow through all $d _ { r }$ dimensions. This gradient disrupts proper training of the kernel weights, since the gradient of the $d _ { r }$ sub-vector $( h [ d _ { 0 } : d _ { 0 } + d _ { r } ] )$ points toward $\mathbb { 1 } _ { d _ { r } }$ , which will eventually lead to dimensional collapse or may lead all points to lie within an open ball of finite radius along each of the $d _ { r }$ dimensions. The performance, in this case, is worse than training with zero value in the $d _ { r }$ dimensions and did not contribute towards maximizing the flow of information through the $d _ { 0 }$ dimensions, as in DirectCLR (Jing et al., 2022) or our framework with $k = 0$ , due to the injection of a non-converging gradient. Empirical results for the CIFAR datasets are provided in Table 1.

Table 1: 200-NN Accuracy for diferent values of $d _ { 0 }$ and $d _ { r }$ on CIFAR10 and CIFAR100 dataset.
<table><tr><td rowspan=2 colspan=1> $d _ { 0 }$ </td><td rowspan=2 colspan=1> $d _ { r }$ </td><td rowspan=2 colspan=1>Fixed Value</td><td rowspan=1 colspan=2>CIFAR10  CIFAR100</td></tr><tr><td rowspan=1 colspan=2>200-NNAcc.</td></tr><tr><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>312</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>83.20</td><td rowspan=1 colspan=1>49.4</td></tr><tr><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>312</td><td rowspan=1 colspan=1> $\scriptstyle { \frac { 1 } { \sqrt { 5 1 2 } } }$ </td><td rowspan=1 colspan=1>82.5</td><td rowspan=1 colspan=1>48.6</td></tr><tr><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>312</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>78.9</td><td rowspan=1 colspan=1>41.1</td></tr><tr><td rowspan=1 colspan=2>400 112</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>84.2</td><td rowspan=1 colspan=1>52.2</td></tr><tr><td rowspan=1 colspan=2>400 112</td><td rowspan=1 colspan=1> $\scriptstyle { \frac { 1 } { \sqrt { 5 1 2 } } }$ </td><td rowspan=1 colspan=1>84.0</td><td rowspan=1 colspan=1>50.9</td></tr><tr><td rowspan=1 colspan=3>400 112        1</td><td rowspan=1 colspan=1>79.8</td><td rowspan=1 colspan=1>44.3</td></tr><tr><td rowspan=1 colspan=1>480</td><td rowspan=1 colspan=1>321</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>84.7</td><td rowspan=1 colspan=1>52.5</td></tr><tr><td rowspan=1 colspan=1>480</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1> $\scriptstyle { \frac { 1 } { \sqrt { 5 1 2 } } }$ </td><td rowspan=1 colspan=1>84.4</td><td rowspan=1 colspan=1>51.8</td></tr><tr><td rowspan=1 colspan=1>480</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>80.9</td><td rowspan=1 colspan=1>45.9</td></tr></table>

![](images/1e70e6e649c779469183933ccc8638eae3277e2b4b0224608be2ba1dad2b6206.jpg)  
(a) CIFAR10 $( d _ { 0 } = 2 0 0 )$

![](images/0f81a985c1409fc984762985303410a43b7b622548d8f92b30c64546f8267bec.jpg)  
(b) CIFAR10 $( d _ { 0 } = 4 0 0 )$

![](images/a9f1dc00d75acdd0556bbde88a62983e62005c6025e6de7de7faab125a0bbd5d.jpg)  
(c) CIFAR100(d<sub>0</sub> = 200)

![](images/a43bc2e147f59a7be16444ca1471b77e8eefb6f380f4e8e80e8d7a1827870899.jpg)  
(d) CIFAR100(d<sub>0</sub> = 400)  
Figure 7: Singular values plots of encoder outputs embeddings when using a constant subvector in the output embeddings for diferent values of $d _ { 0 }$ on CIFAR10 and CIFAR100 datasets. The above figures show the efect on the singular values of the encoder output embeddings, and it is evident that a forced collapse does not have the same efect as a natural dimensional collapse . Best viewed at 300%.

Key Takeaway 3: Forced Collapse does not enforce an Information Bottleneck The empirical results provided in Tables 1 combined with Fig. 7 indicate that we cannot observe the same information bottleneck efect without a projector. From the discussion in this subsection, we can safely say that the role of the projector is not only that of an information bottleneck, which is observed from the inefectiveness of the forced collapse on the encoder output. The driving factor behind the efectiveness of the projector is that it allows the encoder to learn more high-level representations and, consequently, better separability with a higher rank of the embedding covariance matrix. Whereas a collapse in the last layer of the encoder prevents it from learning useful representations, which are essential for the efective classification of the input samples, as the norm of several kernels will be reduced to zero. With a constant subvector in the output, the weights are not updated to learn useful representations. Furthermore, being unable to learn essential representations also diminishes the mutual information between the input and the output and the generalization error bound, which we prove in the next subsection.

## 3.7 Proposed Method: Remedy based on the Takeaways

To address weight-induced dimensional collapse without distorting the learned feature geometry of shallower layers, we introduce an orthogonal weight regularization restricted exclusively to the final layer $( W _ { l } )$ . Guided by Proposition 1 and Proposition 2, applying this regularization strictly to the projection head eliminates weight magnitude shrinkage (Cause 1) by enforcing row-orthogonality $( W _ { l } W _ { l } ^ { \top } = I )$ and preserving the full-rank capacity $( \mathrm { r a n k } ( W _ { l } ) = D _ { o } )$ of $W _ { l }$

Empirical observations in contrastive SSL reveal that the penultimate feature covariance $\Sigma _ { l - 1 } = \mathrm { C o v } ( x _ { l - 1 } )$ maintains an intrinsically ill-conditioned eigenspectrum $\left( \kappa ( \Sigma _ { l - 1 } ) \gg 1 \right)$ , reflecting the natural variance hierarchy of learned semantic representations. In this ill-conditioned regime (Cause 4), the output representation is hyper-sensitive to weight decay and rank degradation in the projection head $W _ { l }$ . Applying orthogonal regularization $( W _ { l } W _ { l } ^ { \top } = I )$ exclusively to the last layer eliminates weight-induced collapse (Cause 1) and guarantees that $W _ { l }$ maintains full rank $( \mathrm { r a n k } ( W _ { l } ) = D _ { o } )$ , without distorting the functional feature geometry or semantic hierarchy of the backbone features $x _ { l - 1 }$ . Additionally, preventing the weight norm from collapsing preserves the representational capacity of the layer, thereby preventing artificial information loss between the penultimate and output representations.

Proposition 3: Weight regularization and mutual information: Let $Z = W X \in \mathbb { R } ^ { D }$ be the output embedding used in the InfoNCE objective, where x is a random vector with $\Sigma _ { X } = \mathrm { C o v } ( X ) \succ 0 , Z$ is also a random vector such that $Z = [ z _ { 1 } , z _ { 2 } , \dots , z _ { D _ { o } } ]$ and $W \in \mathbb { R } ^ { D _ { o } \times D _ { i } }$ is the weight matrix of the final layer. Assume that $Z _ { k }$ and $Z _ { l }$ form a positive pair generated from two augmentations of the same input $X _ { k } , X _ { l } \in \mathbb { R } ^ { D _ { i } }$ , and let

$$
Z _ { k } = W X _ { k } , \qquad Z _ { l } = W X _ { l } ,\tag{30}
$$

Assume that $X _ { k }$ and $X _ { l }$ have finite second-order moments. We compare two cases: (i) an orthogonalityconstrained representation satisfying $W W ^ { \top } = I$ , and (ii) an unconstrained (denoted by “unc.” from here onwards) representation for which weight regularization may reduce the singular values of W and consequently reduce its efective rank. Then the orthogonality constraint preserves the mutual information between the two augmented views, whereas an unconstrained W can reduce the mutual information when it suppresses directions carrying information shared between the two views, that is,

$$
{ \mathcal { T } } ( Z _ { k } ; Z _ { l } \mid W W ^ { \top } = I ) \ \geq \ { \mathcal { T } } ( Z _ { k } ; Z _ { l } \mid W _ { \mathrm { u n c . } } )\tag{31}
$$

Proof: Consider the final representation layer $Z = W X \in \mathbb { R } ^ { D _ { a } }$ , where $W \in \mathbb { R } ^ { D _ { o } \times D _ { i } } \left( D _ { o } \leq D _ { i } \right)$ and $\Sigma _ { X } = \operatorname { C o v } ( X ) \succ 0$ . The output covariance is given by:

$$
\Sigma _ { Z } = \operatorname { C o v } ( Z ) = W \Sigma _ { X } W ^ { \top } \in \mathbb { R } ^ { D _ { o } \times D _ { o } }\tag{32}
$$

Let $\mathbf { u } _ { i } \in \mathbb { R } ^ { D _ { o } }$ be an orthonormal eigenvector of $\Sigma _ { Z }$ corresponding to eigenvalue $\lambda _ { i } ( \Sigma _ { Z } )$ , with $\| \mathbf { u } _ { i } \| _ { 2 } = 1$ . By the Rayleigh-Ritz theorem (Trefethen & Bau, 2022):

$$
\lambda _ { i } ( \Sigma _ { Z } ) = \mathbf { u } _ { i } ^ { \top } \Sigma _ { Z } \mathbf { u } _ { i } = \mathbf { u } _ { i } ^ { \top } \left( W \Sigma _ { X } W ^ { \top } \right) \mathbf { u } _ { i } = ( W ^ { \top } \mathbf { u } _ { i } ) ^ { \top } \Sigma _ { X } ( W ^ { \top } \mathbf { u } _ { i } )\tag{33}
$$

Since $\Sigma _ { X } \succ 0$ is symmetric positive-definite, bounding the quadratic form by the extremal eigenvalues of $\Sigma _ { X }$ yields:

$$
\lambda _ { \operatorname* { m i n } } ( \Sigma _ { \boldsymbol { X } } ) \| \boldsymbol { W } ^ { \top } \mathbf { u } _ { i } \| _ { 2 } ^ { 2 } \leq \lambda _ { i } ( \Sigma _ { \boldsymbol { Z } } ) \leq \lambda _ { \operatorname* { m a x } } ( \Sigma _ { \boldsymbol { X } } ) \| \boldsymbol { W } ^ { \top } \mathbf { u } _ { i } \| _ { 2 } ^ { 2 }\tag{34}
$$

where $\| \boldsymbol { W } ^ { \top } \mathbf { u } _ { i } \| _ { 2 } ^ { 2 } = \mathbf { u } _ { i } ^ { \top } \boldsymbol { W } \boldsymbol { W } ^ { \top } \mathbf { u } _ { i }$ . Equation 34 establishes the exact connection between $W W ^ { \top }$ and output spectrum collapse:

1. Unconstrained Case $( W W ^ { \top } \ne I _ { D _ { o } } )$ : Without orthogonal constraints, weight decay or rank deficiency permits $\| \boldsymbol { W } ^ { \top } \mathbf { u } _ { i } \| _ { 2 } ^ { 2 }  0$ along trailing modes $i \in \{ r + 1 , \ldots , D _ { o } \}$ . Using the upper bound in Eqn. 34, this directly forces output collapse:

$$
\lambda _ { i } ( \Sigma _ { Z } ) \leq \lambda _ { \operatorname* { m a x } } ( \Sigma _ { X } ) \lVert W ^ { \top } \mathbf { u } _ { i } \rVert _ { 2 } ^ { 2 } \to 0 ,\tag{35}
$$

The above equation indicates that the unconstrained case concentrates representations $Z$ onto an r-dimensional afine subspace $\mathcal { A } _ { Z } \subset \mathbb { R } ^ { D _ { o } }$ by the Eckart-Young-Mirsky theorem (Humpherys et al., 2017).

2. Orthogonally Constrained Case $( W W ^ { \top } = I _ { D _ { o } } )$ : Enforcing $W W ^ { \top } = I _ { D _ { c } }$ guarantees that for any unit vector $\mathbf { u } _ { i } , \| W ^ { \top } \mathbf { u } _ { i } \| _ { 2 } ^ { 2 } = \mathbf { u } _ { i } ^ { \top } I _ { D _ { o } } \mathbf { u } _ { i } = 1$ . Utilizing the lower bound in Eqn. 34, the output spectrum is strictly bounded away from zero:

$$
\lambda _ { i } ( \Sigma _ { Z } ) \geq \lambda _ { \operatorname* { m i n } } ( \Sigma _ { X } ) > 0 \quad \forall i \in \{ 1 , \dots , D _ { o } \} ,\tag{36}
$$

The above equation guarantees rank $\begin{array} { r } { \mathbf { \nabla } : ( \Sigma _ { Z } ) = D _ { c } } \end{array}$ and preventing weight-induced dimensional collapse.

Preservation of Mutual Information via Orthogonal Regularization. Next, we establish that enforcing $W$ to be orthogonal results in preservation of mutual information, compared to when the weights are unconstrained. We arrive at the above statement by progressing step-by-step through the proof as follows:

Step 1: Constructing the Matrix Factorization. Consider the output representations under the orthogonal reference matrix $W _ { \mathrm { o r t h o } } \mathrm { : }$

$$
Z _ { k } ^ { \mathrm { o r t h o } } = W _ { \mathrm { o r t h o } } X _ { k } , \qquad Z _ { l } ^ { \mathrm { o r t h o } } = W _ { \mathrm { o r t h o } } X _ { l } \quad \in \mathbb { R } ^ { D _ { o } }\tag{37}
$$

Also assume, there exists an orthogonally constrained final-layer weight matrix $W _ { \mathrm { o r t h o } } \in \mathbb { R } ^ { D _ { o } \times D _ { i } } \left( D _ { o } \leq D _ { i } \right)$ satisfying strict row-orthogonality: $W _ { \mathrm { o r t h o } } W _ { \mathrm { o r t h o } } ^ { \top } = I _ { D _ { o } }$ . This guarantees that $W _ { \mathrm { o r t h o } }$ is full row rank $\left( \mathrm { r a n k } ( W _ { \mathrm { o r t h o } } ) = D _ { o } \right)$ and the matrix product $P _ { \mathrm { o r t h o } } ^ { > \overline { { \mathbf { \phi } } } > } = W _ { \mathrm { o r t h o } } ^ { \top } W _ { \mathrm { o r t h o } } \in \mathbb { R } ^ { D _ { i } \times D _ { i } }$ forms an orthogonal projection operator onto row $( W _ { \mathrm { o r t h o } } )$

Furthermore assume that, for any unconstrained weight matrix $W _ { \mathrm { u n c . } } \in \mathbb { R } ^ { D _ { o } \times D _ { i } }$ , its row space is contained within the $D _ { o }$ -dimensional row space of $W _ { \mathrm { o r t h o } } \colon \mathrm { r o w } ( W _ { \mathrm { u n c . } } ) \subseteq \mathrm { r o w } ( W _ { \mathrm { o r t h o } } )$

Consequently, the matrix $P _ { \mathrm { o r t h o } } = W _ { \mathrm { o r t h o } } ^ { \top } W _ { \mathrm { o r t h o } } \in \mathbb { R } ^ { D _ { i } \times D _ { i } }$ acts as an identity operator on the row space of $W _ { \mathrm { u n c . } }$ , yielding $W _ { \mathrm { u n c . } } P _ { \mathrm { o r t h o } } = W _ { \mathrm { u n c . } }$ , which implies $W _ { \mathrm { u n c . } } P _ { \mathrm { o r t h o } } = W _ { \mathrm { u n c . } }$ . Next, we can define a transition matrix $M \in \mathbb { R } ^ { D _ { o } \times D _ { c } }$ as:

$$
M = W _ { \mathrm { u n c . } } W _ { \mathrm { o r t h o } } ^ { \top }\tag{38}
$$

Next, if we multiply M by $W _ { \mathrm { o r t h o } }$ we get the following:

$$
\begin{array} { r l } & { M W _ { \mathrm { o r t h o } } = \left( W _ { \mathrm { u n c . } } W _ { \mathrm { o r t h o } } ^ { \top } \right) W _ { \mathrm { o r t h o } } } \\ & { ~ = W _ { \mathrm { u n c . } } \left( W _ { \mathrm { o r t h o } } ^ { \top } W _ { \mathrm { o r t h o } } \right) = W _ { \mathrm { u n c . } } P _ { \mathrm { o r t h o } } = W _ { \mathrm { u n c . } } } \end{array}\tag{39}
$$

Thus, $W _ { \mathrm { u n c . } }$ <sub>.</sub> cleanly decomposes into a linear transformation M acting directly on $W _ { \mathrm { o r t h o } }$

$$
W _ { \mathrm { u n c . } } = M W _ { \mathrm { o r t h o } }\tag{40}
$$

In the next step, we map the representations obtained using the unconstrained and orthogonal weights and formulate using a Markov chain to find the relationship between the mutual information of the two.

Step 2: Representation Mapping and Markov Chain Formulation. Using the factorization $W _ { \mathrm { u n c . } } =$ $M W _ { \mathrm { o r t h o } } ,$ the unconstrained output representations $Z _ { k } ^ { \mathrm { u n c . } } = W _ { \mathrm { u n c . } } X _ { k }$ and $Z _ { l } ^ { \mathrm { u n c . } } = W _ { \mathrm { u n c . } } X _ { l }$ can be written as deterministic functions of the orthogonal representations:

$$
Z _ { k } ^ { \mathrm { u n c . } } = M \left( W _ { \mathrm { o r t h o } } X _ { k } \right) = M Z _ { k } ^ { \mathrm { o r t h o } } , \qquad Z _ { l } ^ { \mathrm { u n c . } } = M \left( W _ { \mathrm { o r t h o } } X _ { l } \right) = M Z _ { l } ^ { \mathrm { o r t h o } }\tag{41}
$$

Since $Z _ { k } ^ { \mathrm { u n c . } }$ depends on $X _ { k }$ solely through $Z _ { k } ^ { \mathrm { o r t h o } }$ , and $Z _ { l } ^ { \mathrm { u n c . } }$ <sup>.</sup> depends on $X _ { l }$ solely through $Z _ { l } ^ { \mathrm { o r t h o } }$ , the four random vectors satisfy a symmetric 4-node Markov chain:

$$
Z _ { k } ^ { \mathrm { u n c . } } \longrightarrow Z _ { k } ^ { \mathrm { o r t h o } } \longrightarrow Z _ { l } ^ { \mathrm { o r t h o } } \longrightarrow Z _ { l } ^ { \mathrm { u n c . } }\tag{42}
$$

Next, we apply the data processing inequality (Cover & Thomas, 2006) on the Markov chain to deduce that the mutual information cannot increase.

Step 3: Application of the Data Processing Inequality (DPI). By the Data Processing Inequality (Cover & Thomas, 2006) for Markov chains, passing a random vector through a deterministic mapping cannot increase its mutual information with any other random variable.

First, applying DPI to the transition $Z _ { k } ^ { \mathrm { u n c . } }  Z _ { k } ^ { \mathrm { o r t h o } }  Z _ { l } ^ { \mathrm { o r t h o } }$

$$
{ \mathscr T } \left( Z _ { k } ^ { \mathrm { u n c . } } ; Z _ { l } ^ { \mathrm { o r t h o } } \right) \ \leq \ { \mathscr T } \left( Z _ { k } ^ { \mathrm { o r t h o } } ; Z _ { l } ^ { \mathrm { o r t h o } } \right)\tag{43}
$$

Second, applying DPI to the right-hand transition $Z _ { k } ^ { \mathrm { u n c . } }  Z _ { l } ^ { \mathrm { o r t h o } }  Z _ { l } ^ { \mathrm { u n c . } }$

$$
{ \mathcal { T } } \left( Z _ { k } ^ { \mathrm { u n c . } } ; Z _ { l } ^ { \mathrm { u n c . } } \right) \ \leq \ { \mathcal { T } } \left( Z _ { k } ^ { \mathrm { u n c . } } ; Z _ { l } ^ { \mathrm { o r t h o } } \right)\tag{44}
$$

Combining these two inequalities yields the general weak inequality:

$$
{ \mathcal { T } } \left( Z _ { k } ^ { \mathrm { u n c . } } ; Z _ { l } ^ { \mathrm { u n c . } } \right) \ \leq \ { \mathcal { T } } \left( Z _ { k } ^ { \mathrm { o r t h o } } ; Z _ { l } ^ { \mathrm { o r t h o } } \right)\tag{45}
$$

Step 4: Establishing Strict Inequality via Null-space Information Loss. From Proposition 1 and $2 ,$ we can state that, under unconstrained training or weight decay, $W _ { \mathrm { u n c } }$ undergoes dimensional collapse to an efective rank $r < D _ { o }$ . Because rank $( M ) \leq \mathrm { r a n k } ( W _ { \mathrm { u n c . } } ) = r ,$ , the operator $\boldsymbol { M } \in \mathbb { R } ^ { D _ { o } \times D _ { o } }$ is non-invertible and possesses a non-trivial null-space $\operatorname { N u l l } ( M ) \subset \mathbb { R } ^ { D _ { o } }$ of dimension $D _ { o } - r > 0$

Next, we decompose the orthogonal representation $Z _ { k } ^ { \mathrm { o r t h o } }$ into two orthogonal components: $Z _ { k } ^ { \parallel } \in \mathrm { R a n g e } ( M ^ { \top } )$ (the preserved subspace) and $Z _ { k } ^ { \perp } \in \operatorname { N u l l } ( M )$ (the collapsed subspace). We similarly decompose $Z _ { l } ^ { \mathrm { o r t h o } }$ as $Z _ { l } ^ { \mathrm { o r t h o } } = Z _ { l } ^ { \parallel } + Z _ { l } ^ { \perp }$

The deterministic transformation M acts as a bijection on $\mathrm { R a n g e } ( M ^ { \top } )$ but maps all variance in $\mathrm { N u l l } ( M )$ to zero (Detailed description in Appendix E):

$$
Z _ { k } ^ { \mathrm { u n c . } } = M Z _ { k } ^ { \| } + 0 = M Z _ { k } ^ { \| } , \qquad Z _ { l } ^ { \mathrm { u n c . } } = M Z _ { l } ^ { \| }\tag{46}
$$

By the chain rule for mutual information:

$$
\mathcal { T } \left( Z _ { k } ^ { \mathrm { o r t h o } } ; Z _ { l } ^ { \mathrm { o r t h o } } \right) = \mathcal { Z } \left( Z _ { k } ^ { \parallel } ; Z _ { l } ^ { \parallel } \right) + \mathcal { Z } \left( Z _ { k } ^ { \perp } ; Z _ { l } ^ { \perp } \mid Z _ { k } ^ { \parallel } , Z _ { l } ^ { \parallel } \right)\tag{47}
$$

Since M is a bijection on Ra $\begin{array} { r } { \mathrm { 1 g e } ( M ^ { \top } ) , \mathcal { T } ( Z _ { k } ^ { \mathrm { u n c . } } ; Z _ { l } ^ { \mathrm { u n c . } } ) = \mathcal { T } \left( Z _ { k } ^ { \| } ; Z _ { l } ^ { \| } \right) } \end{array}$ . Therefore, the exact information lost due to rank collapse is:

$$
\mathcal { T } \left( Z _ { k } ^ { \mathrm { o r t h o } } ; Z _ { l } ^ { \mathrm { o r t h o } } \right) - \mathcal { T } \left( Z _ { k } ^ { \mathrm { u n c . } } ; Z _ { l } ^ { \mathrm { u n c . } } \right) = \mathcal { Z } \left( Z _ { k } ^ { \perp } ; Z _ { l } ^ { \perp } \ | \ Z _ { k } ^ { \parallel } , Z _ { l } ^ { \parallel } \right)\tag{48}
$$

If we assume that, the feature subspace eliminated by the rank collapse of $W _ { \mathrm { u n c . } }$ carries a non-zero amount of shared statistical information between the two augmented views $X _ { k }$ and $X _ { l }$ , making $\mathcal { T } \left( Z _ { k } ^ { \perp } ; Z _ { l } ^ { \perp } \mid Z _ { k } ^ { \parallel } , Z _ { l } ^ { \parallel } \right) > 0$ this establishes the strict inequality:

$$
{ \mathcal { T } } \left( Z _ { k } ; Z _ { l } \mid W W ^ { \top } = I _ { D _ { o } } \right) ~ > ~ { \mathcal { T } } \left( Z _ { k } ; Z _ { l } \mid W _ { \mathrm { u n c . } } \right)\tag{49}
$$

which proves that orthogonal regularization strictly preserves mutual information against weight-induced dimensional collapse.

Preventing small variance using weight regularization: Considering the above deductions, we intend to apply weight regularization to the last-layer weight matrix, thereby maintaining full-rank covariance and preserving information across all dimensions, thereby maximising mutual information between embeddings.

Regularizing the weight matrix via minimizing $\| W W ^ { \top } - I \| _ { F } ^ { 2 }$ , as follows,

$$
\boldsymbol { W } = \arg \operatorname* { m i n } _ { \boldsymbol { W } } \| \boldsymbol { W } \boldsymbol { W } ^ { \intercal } - \boldsymbol { I } \| _ { F } ^ { 2 }\tag{50}
$$

This objective encourages the singular values of W to remain bounded away from zero, thereby discouraging degenerate projection directions in the representation space. Consequently, the representation covariance is less likely to collapse onto a lower-dimensional subspace.

## 3.8 Optimization Objective

To prove that our interpretation of the role of the projector in self-supervised contrastive learning is correct, we regularize the weight matrix W of the last layer only by minimizing the regularization loss $\mathcal { L } _ { r e g } = \| W W ^ { T } - I \| ^ { 2 }$ in addition to the conservative loss in Eqn. 1. Thus, the final loss is described as,

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { i n f o n c e } + \alpha \mathcal { L } _ { r e g }\tag{51}
$$

Based on the above objective function, we optimize the parameters of the neural network following the implementation details outlined in Sec. 5. The results obtained are presented in the section below. We present the results on the datasets CIFAR10, CIFAR100, and ImageNet100 and also present eigenvalue plots for the CIFAR datasets to compare the efect of the proposed regularization towards the prevention of dimensional collapse. For all our experiments, we use α = 0.1 for optimal performance following WeRank.

## 4 Results and Analyses

## 4.1 Comparison with state-of-the-art Contrastive learning frameworks

Table 2: Comparison of results obtained by applying WeRank variations to SimCLR without Projector, and our proposed strategy on ImageNet100 datasets. Here, ‘LL’ refers to ‘Last Layer’. $( + / - \cdot \cdot ) \colon$ change from the previous model variation. We can observe that the proposed methods outperform the baseline SimCLR (vanilla) and WeRank on almost all cases (except one). This proves the efectiveness of the proposed regularization strategy on preventing degradation of performance due to dimensional collapse.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>ImageNet100</td></tr><tr><td rowspan=1 colspan=1>SimCLR (vanilla) w/o Projector</td><td rowspan=1 colspan=1>45.82</td></tr><tr><td rowspan=1 colspan=1>SimCLR w/o Proj. + WeRank (Full Enc.)</td><td rowspan=1 colspan=1>44.68 (-1.14)</td></tr><tr><td rowspan=1 colspan=1>SimCLR w/o Proj. + Wt. Reg. LL (Ours)</td><td rowspan=1 colspan=1>46.82 (+1.0)</td></tr></table>

In this subsection, we analyse the eficacy of the proposed solution on diferent self-supervised frameworks, both with and without a projector. From Table 2 and 3, we can observe that the proposed solution successfully improves the kNN accuracy of all the SSL frameworks used and also reduces the dimensional collapse issue due to the low variance of feature dimensions in Fig. 8 (in Section 4.2). We also conduct experiments on the ImageNet100 dataset, and compare our proposed strategy with vanilla SimCLR, and SimCLR added with WeRank, all trained without a projector. We observe from Table 2 that the proposed strategy outperforms the two baselines on ImageNet100.

Enforcing $W . W ^ { T } = I$ at the last layer improves SimCLR by conditioning the embedding geometry seen by the contrastive objective and preventing projector-induced anisotropy. However, imposing orthogonality constraints throughout the backbone (as in WeRank) over-regularizes the representation function class to a subspace in the function space with orthogonal weight matrices, restricting anisotropic feature shaping that is crucial for semantic abstraction (Wang et al., 2025; Gao et al., 2025; Rudman & Eickhof, 2024), and thereby degrades downstream performance. This suggests that spectral conditioning is most efective when applied locally to the contrastive embedding space rather than globally to all feature transformations.

## 4.2 Eigenvalue plots comparing SimCLR Encoder and our method

From Fig.8 we can see that when weight regularization is performed on the last layer of the encoder network, the singular values of the output embeddings improve across all dimensions. Thus, our method can reduce the efect of dimensional collapse due to the low variance of feature dimensions and provide better performance.

Table 3: Comparison of results obtained by applying WeRank variations to SimCLR with and without Projector, DirectCLR and our proposed strategy on CIFAR10 and CIFAR100 datasets. Here, ‘LL’ refers to ‘Last Layer’. ‘CS’ refers to ‘Constant Subvector’. $( + / - \cdot \cdot ) \colon$ change from the baseline model. We can observe that the proposed methods outperform the baseline SimCLR (vanilla) and WeRank on almost all cases (except one). This proves the efectiveness of the proposed regularization strategy on preventing degradation of performance due to dimensional collapse.
<table><tr><td>Method</td><td>CIFAR10</td><td>CIFAR100</td></tr><tr><td>SimCLR (vanilla)</td><td>86.1</td><td>56.3</td></tr><tr><td>SimCLR + WeRank (Pasand et al., 2024)</td><td>86.5 (+0.4)</td><td>56.8 (+0.5)</td></tr><tr><td>SimCLR + Wt. Reg. LL (Ours)</td><td>86.3 (+0.2)</td><td>57.5 (+1.2)</td></tr><tr><td>SimCLR (vanilla) w/o Projector</td><td>84.5</td><td>52.8</td></tr><tr><td>SimCLR w/o Proj. + WeRank (Full Enc.)</td><td>84.9 (+0.4)</td><td>52.6 (-0.2)</td></tr><tr><td>SimCLR w/o Proj. + Wt. Reg. LL (Ours)</td><td>85.1 (+0.6) | 53.1 (+0.3)</td><td></td></tr><tr><td>DirectCLR (vanilla)</td><td>85.2</td><td>53.2</td></tr><tr><td>DirectCLR+ WeRank (Full Enc.)</td><td>85.4(+0.2)|</td><td>53.0 (-0.2)</td></tr><tr><td>DirectCLR+ Wt. Reg. LL (Ours)</td><td>85.5 (+0.3)</td><td>53.2 (+0.0)</td></tr><tr><td>SimCLR w/o Proj (w/ CS)</td><td>84.7</td><td>52.5</td></tr><tr><td>SimCLR w/o Proj (w/ CS) + WeRank (Full Enc.)</td><td>85.0 (+0.3)</td><td>53.0 (+0.5)</td></tr><tr><td>SimCLR w/o Proj (w/ CS) + Wt. Reg. LL (Ours) | </td><td>85.5 (+0.8)</td><td>53.1 (+0.6)</td></tr></table>

![](images/bfe79f740632437fe94b02463c40b9bd7e5d51577627f5b95f191b74a6e014d2.jpg)  
(a) CIFAR10

![](images/a009931b1853efefc4bd8f2a638338754df52cf934c93c686d6986be398210b2.jpg)  
(b) CIFAR100  
Figure 8: Singular values plots of encoder outputs, embeddings of SimCLR without a projector, and our method (last layer weight regularization) on CIFAR10 and CIFAR100 datasets. The plots clearly exhibit an improvement in the singular values, which indicates that the proposed regularization strategy prevents the dimensional collapse in the encoder output embeddings, when used without an additional projector.

## 5 Implementation Details

Datasets: We primarily used three datasets for our study: CIFAR10, CIFAR100 (Krizhevsky, 2009), and ImageNet100 (Tian et al., 2020). CIFAR10 and CIFAR100 datasets consist of 10 and 100 classes, respectively, with 50K samples in the training set. ImageNet100 contains 1300 images in each of the 100 classes.

Pre-training Details: For experiments on CIFAR (Krizhevsky, 2009) and ImageNet, we used ResNet18 and ResNet50 as a backbone with the same modifications as done in SimCLR (Chen et al., 2020a) for small-scale datasets (CIFAR). We used a batch size of 128 and 256 for CIFAR and ImageNet, respectively. For the CIFAR and ImageNet datasets, we used SGD and LARS optimizer, respectively. All the implementations were done using lightly-ai (Susmelj et al., 2020) library. The value of the temperature hyper-parameter was set to 0.2 for all experiments. The value of α was set to 0.1 for all experiments, as using a higher value degrades performance. For the CIFAR dataset, the initial learning rate was set to 0.06 and decayed following a cosine schedule, whereas for the ImageNet dataset, an initial learning rate of 0.3 was used and decayed following the same schedule as the CIFAR dataset. For all the datasets, the training was conducted for 200 epochs only.

For the computation of the SVD decomposition, we simply used the svd function from the numpy library, following DirectCLR (Jing et al., 2022).

## 6 Limitations and Future Work

In this work, we presented a layer-wise linear algebraic framework to characterize the multiple possible mechanisms of dimensional collapse, specifically isolating weight norm shrinkage, inter-layer spectral misalignment, and feature covariance misalignment. While our proposed targeted weight regularization $( W W ^ { \top } = I )$ efectively mitigates weight-induced rank collapse in the last layer, it primarily addresses weight norm decay without explicitly constraining inter-layer or feature-space directional alignments. In future work, we plan to develop a unified optimization framework that jointly addresses spectral and feature-space misalignment, ofering a more comprehensive solution to dimensional collapse. Additionally, while our theoretical setup and empirical validations focus specifically on InfoNCE-based contrastive learning architectures with CNN backbones, extending this last-layer rank analysis to non-contrastive self-supervised paradigms (e.g., feature decorrelation (Bardes et al., 2022) or predictive architectures (Assran et al., 2023; Balestriero & LeCun, 2025)) and Transformer-based backbones ((Dosovitskiy et al., 2021) presents an exciting direction for future research.

## 7 Conclusion

In this work, we investigate the main reason behind the efectiveness of the projector in preventing dimensional collapse. We analyze mathematically the phenomenon that happens inside the projector and the encoder when trained without a projector. We find that the projector not only creates an information bottleneck but also facilitates the learning of high-level representations in the encoder; without it, a dimensional collapse at the encoder output prevents such learning. We also devise a solution to improve performance by applying weight regularization only to the last layer, with or without a projector, achieving performance better than WeRank, which uses weight regularization across the whole network. We leave the study of the cause of dimensional collapse for our future work.

## Broader Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## References

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15619–15629, 2023.

Randall Balestriero and Yann LeCun. Contrastive and non-contrastive self-supervised learning recover global and local spectral embedding methods. In Advances in Neural Information Processing Systems (NIPS), volume 35, pp. 26671–26685. Curran Associates, Inc., 2022.

Randall Balestriero and Yann LeCun. Lejepa: Provable and scalable self-supervised learning without the heuristics. CoRR, abs/2511.08544, 2025. doi: 10.48550/ARXIV.2511.08544. URL https://doi.org/10. 48550/arXiv.2511.08544.

Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id= xm6YD62D1Ub.

Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In Computer Vision – ECCV 2018, pp. 139–156, Cham, 2018. Springer International Publishing. ISBN 978-3-030-01264-9.

Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In Proceedings of the 34th Internationa Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geofrey Hinton. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, ICML’20. JMLR.org, 2020a.

Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15745–15753, 2020. URL https://api. semanticscholar.org/CorpusID:227118869.

Xinlei Chen, Haoqi Fan, Ross B. Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. CoRR, abs/2003.04297, 2020b. URL https://arxiv.org/abs/2003.04297.

Thomas M. Cover. Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. IEEE Transactions on Electronic Computers, EC-14(3):326–334, 1965. doi: 10.1109/PGEC.1965.264137.

Thomas M. Cover and Joy A. Thomas. Elements of Information Theory 2nd Edition (Wiley Series in Telecommunications and Signal Processing). Wiley-Interscience, July 2006. ISBN 0471241954.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto, and Nicu Sebe. Whitening for self-supervised representation learning. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 3015–3024. PMLR, 2021. URL http://proceedings.mlr.press/v139/ermolov21a.html.

Xianghong Fang, Jian Li, Qiang Sun, and Benyou Wang. Rethinking the uniformity metric in self-supervised learning. In The Twelfth International Conference on Learning Representations, ICLR, 2024. URL https://openreview.net/forum?id=3pf2hEdu8B.

Jingnan Gao, Zhuo Chen, Xiaokang Yang, and Yichao Yan. Anisdf: Fused-granularity neural surfaces with anisotropic encoding for high-fidelity 3d reconstruction. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=v1f6c7wVBm.

Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann LeCun. Rankme: assessing the downstream performance of pretrained self-supervised representations by their rank. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent a new approach to self-supervised learning. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

Kartik Gupta, Thalaiyasingam Ajanthan, Anton van den Hengel, and Stephen Gould. Understanding and improving the role of projection head in self-supervised learning, 2022.

Tianyu Hua, Wenxiao Wang, Zihui Xue, Yue Wang, Sucheng Ren, and Hang Zhao. On feature decorrelation in self-supervised learning. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9578–9588, 2021. URL https://api.semanticscholar.org/CorpusID:233481690.

Jefrey Humpherys, Tyler J. Jarvis, and Emily J. Evans. Foundations of Applied Mathematics, Volume 1: Mathematical Analysis. Society for Industrial and Applied Mathematics, Philadelphia, PA, 2017. doi: 10.1137/1.9781611974904. URL https://epubs.siam.org/doi/abs/10.1137/1.9781611974904.

Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in contrastive self-supervised learning. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id= YevsQ05DEN7.

Alex Krizhevsky. Learning multiple layers of features from tiny images. University of Toronto, pp. 32–33, 2009. URL https://www.cs.toronto.edu/\~kriz/learning-features-2009-TR.pdf.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabrie Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. Trans. Mach. Learn. Res., 2024, 2024. URL https://openreview.net/forum?id=a68SUt6zFt.

Zhuo Ouyang, Kaiwen Hu, Qi Zhang, Yifei Wang, and Yisen Wang. Projection head is secretly an information bottleneck. In The Thirteenth International Conference on Learning Representations, ICLR, 2025. URL https://openreview.net/forum?id=L0evcuybH5.

Ali Saheb Pasand, Reza Moravej, Mahdi Biparva, and Ali Ghodsi. Werank: Towards rank degradation prevention for self-supervised learning using weight regularization, 2024.

William Rudman and Carsten Eickhof. Stable anisotropic regularization. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=dbQH9AOVd5.

Zeen Song, Xingzhe Su, Jingyao Wang, Wenwen Qiang, Changwen Zheng, and Fuchun Sun. Towards the sparseness of projection head in self-supervised learning, 2023.

Igor Susmelj, Matthias Heller, Philipp Wirth, Jeremy Prescott, and Malte Ebner et al. Lightly. GitHub. Note: https://github.com/lightly-ai/lightly, 2020.

Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In Computer Vision – ECCV 2020, pp. 776–794, Cham, 2020. Springer International Publishing. ISBN 978-3-030-58621-8.

Lloyd N Trefethen and David Bau. Numerical linear algebra. Other Titles in Applied Mathematics. Society for Industrial & Applied Mathematics, New York, NY, 25 edition, July 2022.

Yifan Wang, Jun Xu, Yuan Zeng, and Yi Gong. Improving neural volume rendering via learning viewdependent integral approximation. IEEE Transactions on Visualization and Computer Graphics, 31(10): 7684–7695, October 2025. ISSN 2160-9306. doi: 10.1109/tvcg.2025.3554692. URL http://dx.doi.org/ 10.1109/TVCG.2025.3554692.

Yihao Xue, Eric Gan, Jiayi Ni, Siddharth Joshi, and Baharan Mirzasoleiman. Investigating the benefits of projection head for representation learning. In The Twelfth International Conference on Learning Representations, ICLR, 2024. URL https://openreview.net/forum?id=GgEAdqYPNA.

Chun-Hsiao Yeh, Cheng-Yao Hong, Yen-Chi Hsu, Tyng-Luh Liu, Yubei Chen, and Yann LeCun. Decoupled contrastive learning. In Computer Vision – ECCV 2022, pp. 668–684, Cham, 2022. Springer Nature Switzerland. ISBN 978-3-031-19809-0.

Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised learning via redundancy reduction. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 12310–12320. PMLR, 2021. URL http://proceedings.mlr.press/v139/zbontar21a.html.

Chaoning Zhang, Kang Zhang, Chenshuang Zhang, Trung X. Pham, Chang D. Yoo, and In So Kweon. How does simsiam avoid collapse without negative samples? A unified understanding with self-supervised contrastive learning. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022a. URL https://openreview.net/forum?id=bwq6O4Cwdl.

Shaofeng Zhang, Feng Zhu, Junchi Yan, Rui Zhao, and Xiaokang Yang. Zero-cl: Instance and feature decorrelation for negative-free symmetric contrastive learning. In The Tenth International Conference on Learning Representations, ICLR 2022, 2022b. URL https://openreview.net/forum?id=RAW9tCdVxLj.

Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. Image BERT pre-training with online tokenizer. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=ydopy-e6Dg.

Jiachen Zhu, Katrina Evtimova, Yubei Chen, Ravid Shwartz-Ziv, and Yann LeCun. Variance-covariance regularization improves representation learning. arXiv preprint arXiv:2306.13292, 2023.

## A Notations

In this subsection, we discuss the notations followed in the rest of the paper for mathematical derivations and discussions.

Table 4: Table for Notations  
ine Symbols What it means   
ine L Loss function   
$x$ Input   
$f$ Encoder   
h output of Encoder f   
g Projector   
z Output of Projector   
$s _ { i j }$ Cosine similarity between the projector output embeddings of the samples of $x _ { i }$ and $x _ { j }$   
$B$ Batch Size   
D Number of dimensions in the encoder output embedding   
$d _ { 0 }$ Number of trainable dimensions in the encoder output embedding   
$d _ { r }$ Number of non-trainable dimensions in the encoder output embedding   
$W _ { l }$ Weight matrix of l-th layer with dimensions $D _ { o } \times D _ { i }$   
$D _ { o }$ Output dimensions   
$D _ { i }$ Input dimensions   
$W _ { l } ^ { i }$ i-th row of the weight matrix $W _ { l }$   
$\mathcal { T }$ Mutual information   
ine

## B Bounding the cross-covariance term.

For any two rows $W ^ { i } , W ^ { j } \in \mathbb { R } ^ { 1 \times D }$ and a symmetric positive semi-definite matrix $\Sigma \in \mathbb { R } ^ { D \times D }$ , the following holds:

$$
| W ^ { j } \Sigma ( W ^ { i } ) ^ { T } | \leq \lambda _ { \operatorname* { m a x } } ( \Sigma ) \| W ^ { j } \| _ { 2 } \| W ^ { i } \| _ { 2 } .\tag{52}
$$

Proof. Let $w _ { i } = ( W ^ { i } ) ^ { T }$ and $w _ { j } = ( W ^ { j } ) ^ { T }$ . Then, the scalar quantity can be rewritten as

$$
| W ^ { j } \Sigma ( W ^ { i } ) ^ { T } | = | w _ { j } ^ { T } \Sigma w _ { i } | .\tag{53}
$$

Since Σ is symmetric and positive semi-definite, it admits a spectral factorization $\Sigma = \Sigma ^ { 1 / 2 } \Sigma ^ { 1 / 2 }$ . Hence,

$$
| w _ { j } ^ { T } \Sigma w _ { i } | = | w _ { j } ^ { T } \Sigma ^ { 1 / 2 } \Sigma ^ { 1 / 2 } w _ { i } | = | \langle \Sigma ^ { 1 / 2 } w _ { j } , \Sigma ^ { 1 / 2 } w _ { i } \rangle | .\tag{54}
$$

Applying the Cauchy–Schwarz inequality yields

$$
| \langle \Sigma ^ { 1 / 2 } w _ { j } , \Sigma ^ { 1 / 2 } w _ { i } \rangle | \leq \| \Sigma ^ { 1 / 2 } w _ { j } \| _ { 2 } \| \Sigma ^ { 1 / 2 } w _ { i } \| _ { 2 } .\tag{55}
$$

Using the sub-multiplicative property of the operator (spectral) norm,

$$
\begin{array} { r } { \| \Sigma ^ { 1 / 2 } w _ { i } \| _ { 2 } \leq \| \Sigma ^ { 1 / 2 } \| _ { o p } \| w _ { i } \| _ { 2 } , } \end{array}\tag{56}
$$

and similarly for $w _ { j }$ . Therefore,

$$
| w _ { j } ^ { T } \Sigma w _ { i } | \leq \| \Sigma ^ { 1 / 2 } \| _ { o p } ^ { 2 } \| w _ { j } \| _ { 2 } \| w _ { i } \| _ { 2 } .\tag{57}
$$

Since $\| \Sigma ^ { 1 / 2 } \| _ { o p } ^ { 2 } = \| \Sigma \| _ { o p } = \lambda _ { \mathrm { m a x } } ( \Sigma )$ for symmetric $\Sigma ,$ , we obtain

$$
| W ^ { j } \Sigma ( W ^ { i } ) ^ { T } | \leq \lambda _ { \operatorname* { m a x } } ( \Sigma ) \| W ^ { j } \| _ { 2 } \| W ^ { i } \| _ { 2 } ,\tag{58}
$$

which completes the proof.

Companion lower bounds and alignment condition. When Σ is symmetric positive definite (so $\lambda _ { \operatorname* { m i n } } ( \Sigma ) > 0 )$ , the variance of an output coordinate admits the immediate lower bound

$$
\mathrm { V a r } ( h _ { i } ) = W ^ { i } \Sigma ( W ^ { i } ) ^ { T } \geq \lambda _ { \operatorname* { m i n } } ( \Sigma ) \| W ^ { i } \| _ { 2 } ^ { 2 } .\tag{59}
$$

Thus, if $\mathrm { V a r } ( h _ { i } )$ is small then $\| W ^ { i } \|$ must be small (cf. Eq. ??).

For the cross-covariance term one can write an exact decomposition by using the square root of $\Sigma \ i$

$$
\begin{array} { r } { W ^ { j } \Sigma ( W ^ { i } ) ^ { T } = \big \langle \Sigma ^ { 1 / 2 } w _ { j } , \Sigma ^ { 1 / 2 } w _ { i } \big \rangle = \| \Sigma ^ { 1 / 2 } w _ { j } \| _ { 2 } \| \Sigma ^ { 1 / 2 } w _ { i } \| _ { 2 } \cos \theta , } \end{array}\tag{60}
$$

where $w _ { i } = ( W ^ { i } ) ^ { T } , w _ { j } = ( W ^ { j } ) ^ { T }$ , and θ is the angle between the vectors $\Sigma ^ { 1 / 2 } w _ { j }$ and $\Sigma ^ { 1 / 2 } w _ { i }$ in $\mathbb { R } ^ { D }$ . Using $\| \Sigma ^ { 1 / 2 } w \| _ { 2 } \geq \sqrt { \lambda _ { \operatorname* { m i n } } ( \Sigma ) } \| w \| _ { 2 }$ and $\| \Sigma ^ { 1 / 2 } w \| _ { 2 } \leq \sqrt { \lambda _ { \operatorname* { m a x } } ( \Sigma ) } \| w \| _ { 2 }$ gives the two-sided inequalities

$$
\begin{array} { r } { | W ^ { j } \Sigma ( W ^ { i } ) ^ { T } | = \| \Sigma ^ { 1 / 2 } w _ { j } \| _ { 2 } \| \Sigma ^ { 1 / 2 } w _ { i } \| _ { 2 } | \cos \theta | \le \lambda _ { \operatorname* { m a x } } ( \Sigma ) \| W ^ { j } \| _ { 2 } \| W ^ { i } \| _ { 2 } , } \end{array}\tag{61}
$$

and, if desired,

$$
\begin{array} { r } { | W ^ { j } \Sigma ( W ^ { i } ) ^ { T } | \ge \lambda _ { \operatorname* { m i n } } ( \Sigma ) \left\| W ^ { j } \right\| _ { 2 } \left\| W ^ { i } \right\| _ { 2 } | \cos \theta | . } \end{array}\tag{62}
$$

Equation 62 shows why an alignment or non-orthogonality assumption is necessary to deduce that a small cross-covariance forces small weight norms: even if $\| \Sigma ^ { 1 / 2 } w _ { i } \| _ { 2 }$ and $\| \Sigma ^ { 1 / 2 } w _ { j } \| _ { 2 }$ are large, the inner product can vanish when cos $\theta = 0$ (i.e., the two vectors are orthogonal in the $\Sigma ^ { 1 / 2 }$ -weighted space). Therefore, to conclude that $| W ^ { j } \Sigma ( W ^ { i } ) ^ { T } |$ being small implies either $\| W ^ { i } \|$ or $\| W ^ { j } \|$ is small, one must additionally assume a lower bound on | cos θ| (for instance, $| \cos \theta | \ge c > 0 )$ , or otherwise restrict the class of admissible weight pairs so that they are not $\sum ^ { 1 / 2 }$ -orthogonal.

## C Linear Operator Representation of Convolution

This appendix shows how a convolutional layer can be expressed as a structured linear operator by progressively moving from the 1D case to the 2D and multi-channel settings. This derivation justifies the use of linearalgebraic arguments in the main text.

## C.1 1D Convolution and Toeplitz Structure

Consider a 1D convolution without non-linearity. Let the input be $x \in \mathbb { R } ^ { n }$ and the kernel be $k \mathbf { \Omega } =$ $[ k _ { 0 } , \ldots , k _ { m - 1 } ] ^ { \top } \in \mathbb { R } ^ { m }$ . Assuming stride 1 and no padding, the output $y \in \mathbb { R } ^ { n - m + 1 }$ is

$$
y _ { i } = \sum _ { j = 0 } ^ { m - 1 } k _ { j } x _ { i + j } , \quad i = 0 , \ldots , n - m .\tag{63}
$$

Define the matrix $W _ { \mathrm { 1 D } } \in \mathbb { R } ^ { ( n - m + 1 ) \times n }$ as

$$
W _ { \mathrm { 1 D } } = \left[ \begin{array} { c c c c c c c c } { k _ { 0 } } & { k _ { 1 } } & { \cdots } & { k _ { m - 1 } } & { 0 } & { \cdots } & { 0 } \\ { 0 } & { k _ { 0 } } & { k _ { 1 } } & { \cdots } & { k _ { m - 2 } } & { k _ { m - 1 } } & { \cdots } \\ { \vdots } & { } & { \ddots } & { } & { } & { } & { \vdots } \\ { 0 } & { \cdots } & { 0 } & { k _ { 0 } } & { k _ { 1 } } & { \cdots } & { k _ { m - 1 } } \end{array} \right] .
$$

Then equation 63 can be written compactly as

$$
y = W _ { \mathrm { 1 D } } x .\tag{64}
$$

The matrix $W _ { \mathrm { 1 D } }$ is a Toeplitz matrix, i.e., its entries are constant along each diagonal.

## C.2 2D Convolution and Block Toeplitz Structure

Now consider a 2D convolution with a single input and output channel. Let $X \in \mathbb { R } ^ { H \times W }$ be the input feature map and $K \in \mathbb { R } ^ { k _ { h } \times k _ { w } }$ be the convolution kernel. The output $Y \in \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } }$ is given by

$$
Y _ { i , j } = \sum _ { u = 0 } ^ { k _ { h } - 1 } \sum _ { v = 0 } ^ { k _ { w } - 1 } K _ { u , v } X _ { i + u , j + v } .\tag{65}
$$

Let $x = \operatorname { v e c } ( X )$ and $y = \operatorname { v e c } ( Y )$ . Then there exists a matrix $W _ { \mathrm { 2 D } } \in \mathbb { R } ^ { ( H ^ { \prime } W ^ { \prime } ) \times ( H W ) }$ such that

$$
y = W _ { \mathrm { 2 D } } x .\tag{66}
$$

The matrix $W _ { \mathrm { 2 D } }$ has a block Toeplitz with Toeplitz blocks (BTTB) structure:

$$
W _ { \mathrm { 2 D } } = \left[ \begin{array} { c c c c c c c c } { T _ { 0 } } & { T _ { 1 } } & { \cdots } & { T _ { k _ { h } - 1 } } & { 0 } & { \cdots } & { 0 } \\ { 0 } & { T _ { 0 } } & { T _ { 1 } } & { \cdots } & { T _ { k _ { h } - 2 } } & { T _ { k _ { h } - 1 } } & { \cdots } \\ { \vdots } & & { \ddots } & & & & { \vdots } \\ { 0 } & { \cdots } & { 0 } & { T _ { 0 } } & { T _ { 1 } } & { \cdots } & { T _ { k _ { h } - 1 } } \end{array} \right] ,
$$

where each block $T _ { u } \in \mathbb { R } ^ { W ^ { \prime } \times W }$ is a Toeplitz matrix formed from the u-th row of the kernel:

$$
T _ { u } = \left[ \begin{array} { c c c c c c } { K _ { u , 0 } } & { K _ { u , 1 } } & { \ldots } & { K _ { u , k _ { w } - 1 } } & { 0 } & { \ldots } \\ { 0 } & { K _ { u , 0 } } & { K _ { u , 1 } } & { \ldots } & { K _ { u , k _ { w } - 1 } } & { \ldots } \\ { \vdots } & { } & { \ddots } & { } & { } & { \vdots } \end{array} \right] .
$$

Thus, 2D convolution corresponds to a BTTB linear operator.

## C.3 Multi-Channel Convolution

Finally, consider a multi-channel convolutional layer. Let

$$
\begin{array} { r } { X \in \mathbb { R } ^ { C _ { \mathrm { i n } } \times H \times W } , \quad Y \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times H ^ { \prime } \times W ^ { \prime } } , } \end{array}
$$

with kernel

$$
K \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times k _ { h } \times k _ { w } } .
$$

For output channel $^ { c , }$

$$
Y _ { c , i , j } = \sum _ { c ^ { \prime } = 1 } ^ { C _ { \mathrm { i n } } } \sum _ { u = 0 } ^ { k _ { h } - 1 } \sum _ { v = 0 } ^ { k _ { w } - 1 } K _ { c , c ^ { \prime } , u , v } X _ { c ^ { \prime } , i + u , j + v } .\tag{67}
$$

Let $x = \operatorname { v e c } ( X )$ and $y = \operatorname { v e c } ( Y )$ . Then

$$
y = W _ { \mathrm { c o n v } } x ,\tag{68}
$$

where

$$
W _ { \mathrm { c o n v } } = \left[ \begin{array} { c c c } { W _ { 1 , 1 } } & { \cdot \cdot } & { W _ { 1 , C _ { \mathrm { i n } } } } \\ { \vdots } & { \ddots } & { \vdots } \\ { W _ { C _ { \mathrm { o u t } } , 1 } } & { \cdot \cdot \cdot } & { W _ { C _ { \mathrm { o u t } } , C _ { \mathrm { i n } } } } \end{array} \right] .
$$

Each block $W _ { c , c ^ { \prime } }$ is a BTTB matrix constructed from the kernel slice $K _ { c , c ^ { \prime } }$ as in the single-channel 2D case. Each row of $W _ { \mathrm { c o n v } }$ corresponds to a specific output channel and spatial location and contains exactly $C _ { \mathrm { i n } } k _ { h } k _ { w }$ non-zero entries given by the kernel weights. Consequently,

$$
\| w _ { i } \| _ { 2 } ^ { 2 } = \sum _ { c ^ { \prime } = 1 } ^ { C _ { \mathrm { i n } } } \sum _ { u = 0 } ^ { k _ { h } - 1 } \sum _ { v = 0 } ^ { k _ { w } - 1 } K _ { c , c ^ { \prime } , u , v } ^ { 2 } .\tag{69}
$$

## C.4 Covariance and Variance Propagation

Let $\Sigma _ { x } = \operatorname { C o v } ( x )$ . From equation 68,

$$
\Sigma _ { y } = \mathrm { C o v } ( y ) = W _ { \mathrm { c o n v } } \Sigma _ { x } W _ { \mathrm { c o n v } } ^ { \top } .\tag{70}
$$

For the i-th output unit,

$$
\begin{array} { r } { \mathrm { V a r } ( y _ { i } ) = w _ { i } ^ { \top } \Sigma _ { x } w _ { i } , } \end{array}\tag{71}
$$

where $\boldsymbol { w _ { i } ^ { \intercal } }$ denotes the i-th row of $W _ { \mathrm { c o n v } }$ . These expressions justify the application of linear-algebraic arguments (e.g., rank, nullspace, and norm-based reasoning) to convolutional layers.

## C.5 Remark on Tensor Representation

The vectorization of X and Y is introduced solely for analytical purposes. The tensors X and Y themselves are not modified, and all empirical results and interpretations in the main text are expressed in terms of the original multi-dimensional feature maps.

## D Decomposition of Entropy and Mutual Information

Entropy of a random vector. Let $\mathbf { Z } = ( z _ { 1 } , z _ { 2 } , \dots , z _ { D } )$ be a continuous random vector with joint density $p ( z _ { 1 } , \dots , z _ { D } )$ . By definition, the (diferential) entropy of Z is

$$
H ( \mathbf { Z } ) = - \int p ( z _ { 1 } , \dots , z _ { D } ) \log p ( z _ { 1 } , \dots , z _ { D } ) d z _ { 1 } \cdot \cdot \cdot d z _ { D } .
$$

Using the chain rule of probabilities,

$$
p ( z _ { 1 } , \dots , z _ { D } ) = p ( z _ { 1 } ) \prod _ { i = 2 } ^ { D } p ( z _ { i } \mid z _ { 1 : i - 1 } ) ,
$$

and therefore

$$
\log p ( z _ { 1 } , \dots , z _ { D } ) = \log p ( z _ { 1 } ) + \sum _ { i = 2 } ^ { D } \log p ( z _ { i } \mid z _ { 1 : i - 1 } ) .
$$

Substituting into the entropy definition,

$$
\begin{array} { l } { { \displaystyle { \cal H } ( { \bf Z } ) = - \int p ( z _ { 1 } , \dots , z _ { D } ) \left[ \log p ( z _ { 1 } ) + \sum _ { i = 2 } ^ { D } \log p ( z _ { i } \mid z _ { 1 : i - 1 } ) \right] d z _ { 1 } \cdots d z _ { D } } } \\ { ~ } \\ { { \displaystyle ~ = - \int p ( z _ { 1 } ) \log p ( z _ { 1 } ) d z _ { 1 } - \sum _ { i = 2 } ^ { D } \int p ( z _ { 1 } , \dots , z _ { i } ) \log p ( z _ { i } \mid z _ { 1 : i - 1 } ) d z _ { 1 } \cdots d z _ { i } } } \\ { ~ } \\ { { \displaystyle ~ = { \cal H } ( z _ { 1 } ) + \sum _ { i = 2 } ^ { D } { \cal H } ( z _ { i } \mid z _ { 1 : i - 1 } ) } . } \end{array}
$$

Thus, the entropy of the vector admits the chain-rule decomposition

$$
\boxed { H ( \mathbf { Z } ) = \sum _ { i = 1 } ^ { D } H ( z _ { i } \mid z _ { 1 : i - 1 } ) , }
$$

with the convention $H ( z _ { 1 } \mid z _ { 1 : 0 } ) \equiv H ( z _ { 1 } )$

Setup for Mutual Information Derivation in terms of decomposed Entropy Let

$$
\mathbf { Z } _ { 1 } = ( z _ { 1 1 } , z _ { 1 2 } , \ldots , z _ { 1 D } ) , \qquad \mathbf { Z } _ { 2 } = ( z _ { 2 1 } , z _ { 2 2 } , \ldots , z _ { 2 D } )
$$

be two continuous random vectors with a joint density $p ( \mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } )$ . No independence assumptions are made across dimensions.

Joint Entropy of the two vectors $\mathbf { Z } _ { 1 }$ and $\mathbf { Z } _ { 2 }$ . By definition,

$$
H ( \mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } ) = - \int p ( \mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } ) \log p ( \mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } ) d \mathbf { Z } _ { 1 } d \mathbf { Z } _ { 2 } .
$$

Using the chain rule of probabilities,

$$
\begin{array} { r } { p ( \mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } ) = p ( \mathbf { Z } _ { 1 } ) p ( \mathbf { Z } _ { 2 } \mid \mathbf { Z } _ { 1 } ) , } \end{array}
$$

hence

$$
\log p ( \mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } ) = \log p ( \mathbf { Z } _ { 1 } ) + \log p ( \mathbf { Z } _ { 2 } \mid \mathbf { Z } _ { 1 } ) .
$$

Substituting into the entropy definition,

$$
\begin{array} { l } { { \displaystyle { \cal H } ( { \bf Z } _ { 1 } , { \bf Z } _ { 2 } ) = - \int p ( { \bf Z } _ { 1 } , { \bf Z } _ { 2 } ) \big [ \log p ( { \bf Z } _ { 1 } ) + \log p ( { \bf Z } _ { 2 } \mid { \bf Z } _ { 1 } ) \big ] d { \bf Z } _ { 1 } d { \bf Z } _ { 2 } } \ ~ } \\ { { \displaystyle ~ = - \int p ( { \bf Z } _ { 1 } ) \log p ( { \bf Z } _ { 1 } ) d { \bf Z } _ { 1 } - \int p ( { \bf Z } _ { 1 } , { \bf Z } _ { 2 } ) \log p ( { \bf Z } _ { 2 } \mid { \bf Z } _ { 1 } ) d { \bf Z } _ { 1 } d { \bf Z } _ { 2 } } \ ~ } \\ { { \displaystyle ~ = { \cal H } ( { \bf Z } _ { 1 } ) + { \cal H } ( { \bf Z } _ { 2 } \mid { \bf Z } _ { 1 } ) } . } \end{array}
$$

Entropy of each vector. From the vector entropy derivation,

$$
H ( \mathbf { Z } _ { 1 } ) = \sum _ { i = 1 } ^ { D } H ( z _ { 1 i } \mid z _ { 1 , 1 : i - 1 } ) , \qquad H ( \mathbf { Z } _ { 2 } ) = \sum _ { j = 1 } ^ { D } H ( z _ { 2 j } \mid z _ { 2 , 1 : j - 1 } ) .
$$

Definition of mutual information. The mutual information between the two vectors is

$$
\boldsymbol { \mathcal { T } } ( \mathbf { Z } _ { 1 } ; \mathbf { Z } _ { 2 } ) = \boldsymbol { H } ( \mathbf { Z } _ { 1 } ) + \boldsymbol { H } ( \mathbf { Z } _ { 2 } ) - \boldsymbol { H } ( \mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } ) .
$$

Substituting the joint entropy decomposition,

$$
{ \displaystyle \mathbb { Z } ( \mathbf { Z } _ { 1 } ; \mathbf { Z } _ { 2 } ) = H ( \mathbf { Z } _ { 1 } ) + H ( \mathbf { Z } _ { 2 } ) - \left[ H ( \mathbf { Z } _ { 1 } ) + H ( \mathbf { Z } _ { 2 } \mid \mathbf { Z } _ { 1 } ) \right] = H ( \mathbf { Z } _ { 2 } ) - H ( \mathbf { Z } _ { 2 } \mid \mathbf { Z } _ { 1 } ) } .
$$

Step 5: Decomposition over dimensions of $\mathbf { Z } _ { 2 }$ . Using the entropy chain rule for $\mathbf { Z } _ { 2 }$

$$
H ( \mathbf { Z } _ { 2 } \mid \mathbf { Z } _ { 1 } ) = \sum _ { j = 1 } ^ { D } H ( z _ { 2 j } \mid \mathbf { Z } _ { 1 } , z _ { 2 , 1 : j - 1 } ) ,
$$

hence

$$
\mathcal { Z } ( \mathbf { Z } _ { 1 } ; \mathbf { Z } _ { 2 } ) = \sum _ { j = 1 } ^ { D } \left[ H ( z _ { 2 j } \mid z _ { 2 , 1 : j - 1 } ) - H ( z _ { 2 j } \mid \mathbf { Z } _ { 1 } , z _ { 2 , 1 : j - 1 } ) \right] = \left[ \sum _ { j = 1 } ^ { D } \mathcal { Z } \left( z _ { 2 j } ; \mathbf { Z } _ { 1 } \mid z _ { 2 , 1 : j - 1 } \right) \right] .
$$

Symmetric form. By symmetry,

$$
\boxed { \mathcal { T } ( \mathbf { Z } _ { 1 } ; \mathbf { Z } _ { 2 } ) = \sum _ { i = 1 } ^ { D } \mathcal { T } \left( z _ { 1 i } ; \mathbf { Z } _ { 2 } \mid z _ { 1 , 1 : i - 1 } \right) . }
$$

Step 6: Chain rule for conditional mutual information. Applying the chain rule to $\mathbf { Z } _ { 1 } ~ =$ $\big ( z _ { 1 1 } , \dots , z _ { 1 D } \big )$

$$
\mathcal { T } ( z _ { 2 j } ; \mathbf { Z } _ { 1 } \mid z _ { 2 , 1 : j - 1 } ) = \sum _ { i = 1 } ^ { D } \mathcal { T } \left( z _ { 2 j } ; z _ { 1 i } \mid z _ { 2 , 1 : j - 1 } , z _ { 1 , 1 : i - 1 } \right) .
$$

Final decomposition.

$$
\boxed { \mathcal { T } ( \mathbf { Z } _ { 1 } ; \mathbf { Z } _ { 2 } ) = \sum _ { j = 1 } ^ { D } \sum _ { i = 1 } ^ { D } \mathcal { T } \left( z _ { 2 j } ; z _ { 1 i } ~ | ~ z _ { 2 , 1 : j - 1 } , ~ z _ { 1 , 1 : i - 1 } \right) }
$$

## E Why M Maps All Variance in Null(M) to Zero and Acts as a Bijection on Range(M<sup>⊤</sup>)

This comes directly from the Fundamental Theorem of Linear Algebra and the orthogonal decomposition of vector spaces.

1. Why M Maps All Variance in Null(M) to Zero By definition, the nullspace (or kernel) of a matrix $M \in \mathbb { R } ^ { \tilde { D } _ { o } \times D _ { o } }$ consists of all vectors that M evaluates to zero:

$$
\mathrm { N u l l } ( M ) = \{ \mathbf { v } \in \mathbb { R } ^ { D _ { o } } \mid M \mathbf { v } = \mathbf { 0 } \}
$$

If a random vector $Z ^ { \perp }$ lies entirely in $\mathrm { N u l l } ( M )$ , then every realization of $Z ^ { \perp }$ satisfies $M Z ^ { \perp } = \mathbf { 0 .  S i n c e } ~ M Z ^ { \perp }$ is identically the constant zero vector 0:Its mean is $\mathbb { E } [ M \dot { Z } ^ { \bot } ] = \mathbf { 0 }$

Its covariance matrix is $\mathrm { C o v } ( M Z ^ { \perp } ) = M \mathrm { C o v } ( Z ^ { \perp } ) { \cal M } ^ { \top } = { \bf 0 }$ . Thus, M completely extinguishes all variance along these directions—collapsing those dimensions into a single point at the origin.

2. Why M Acts as a Bijection on $\mathrm { R a n g e } ( M ^ { \top } )$ By the Fundamental Theorem of Linear Algebra, the space $\mathbb { R } ^ { D _ { o } }$ decomposes into two mutually orthogonal, complementary subspaces:

$$
\mathbb { R } ^ { D _ { o } } = \mathrm { R a n g e } ( M ^ { \top } ) \oplus \mathrm { N u l l } ( M )
$$

Now consider the linear mapping M restricted exclusively to vectors in $\mathrm { R a n g e } ( M ^ { \top } )$ , i.e., $M | _ { \mathrm { R a n g e } ( M ^ { \top } ) }$ $\mathrm { R a n g e } ( M ^ { \top } )  \mathrm { R a n g e } ( M )$

To prove it is a bijection (both injective and surjective):

## • A. Injectivity (One-to-One)

Suppose two vectors $\mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } \in \mathrm { R a n g e } ( M ^ { \top } )$ produce the same output under M:

$$
M \mathbf { v } _ { 1 } = M \mathbf { v } _ { 2 } \implies M ( \mathbf { v } _ { 1 } - \mathbf { v } _ { 2 } ) = \mathbf { 0 }
$$

This implies that the diference vector $\left( \mathbf { v } _ { 1 } - \mathbf { v } _ { 2 } \right)$ lies in $\mathrm { N u l l } ( M )$ . However, since $\mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } \in \mathrm { R a n g e } ( M ^ { \top } )$ their diference $\left( \mathbf { v } _ { 1 } - \mathbf { v } _ { 2 } \right)$ must also belong to $\mathrm { R a n g e } ( M ^ { \dagger } )$ ).Because Range(M<sup>⊤</sup>) and $\mathrm { N u l l } ( M )$ are orthogonal complements, their intersection contains only the zero vector:

$$
\mathrm { R a n g e } ( M ^ { \top } ) \cap \mathrm { N u l l } ( M ) = \{ { \bf 0 } \} \implies { \bf v } _ { 1 } - { \bf v } _ { 2 } = { \bf 0 } \implies { \bf v } _ { 1 } = { \bf v } _ { 2 }
$$

This proves that M is strictly injective on Range $( M ^ { \top } )$ (no two distinct inputs in $\mathrm { R a n g e } ( M ^ { \top } )$ map to the same output).

## • B. Surjectivity (Onto)

By the Rank-Nullity Theorem, the dimension of the row space equals the dimension of the column space (the rank r):

$$
\dim ( \operatorname { R a n g e } ( M ^ { \top } ) ) = \operatorname { r a n k } ( M ) = \dim ( \operatorname { R a n g e } ( M ) ) = r
$$

An injective linear map between two finite-dimensional vector spaces of identical dimension r is automatically surjective (onto ${ \mathrm { R a n g e } } ( M ) $ ).

This is important for evaluating Mutual Information because M is a bijection on $\mathrm { R a n g e } ( M ^ { \top } )$ , passing $Z ^ { \parallel } \in \operatorname { R a n g e } ( M ^ { \top } )$ through M is a lossless, invertible transformation. Invertible linear mappings preserve mutual information perfectly:

$$
\mathcal { T } ( M Z _ { k } ^ { \parallel } ; M Z _ { l } ^ { \parallel } ) = \mathcal { T } ( Z _ { k } ^ { \parallel } ; Z _ { l } ^ { \parallel } )
$$

Meanwhile, because M maps $\mathrm { N u l l } ( M )$ to 0, all information contained in $Z ^ { \perp } \in \operatorname { N u l l } ( M )$ is permanently wiped out. This isolates the exact source of information loss to $\mathcal { T } ( Z _ { k } ^ { \perp } ; Z _ { l } ^ { \perp } \mid Z _ { k } ^ { \parallel } , Z _ { l } ^ { \parallel } )$