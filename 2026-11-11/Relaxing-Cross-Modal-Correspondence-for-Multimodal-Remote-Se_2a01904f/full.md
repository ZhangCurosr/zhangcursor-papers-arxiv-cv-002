# Relaxing Cross-Modal Correspondence for Multimodal Remote Sensing Pretraining

Xianping Ma, Yu Zhang, Mingzhe Jiang, Wanxin Xie, Qing Zhu, Nan Xu, Xiaokang Zhang, and Yuanxin Ye<sup>∗</sup>

Abstract—Multimodal remote sensing pretraining commonly relies on exact instance-level correspondence across modalities, implicitly treating strictly paired observations as a prerequisite for learning transferable representations. In this work, we revisit this assumption by relaxing cross-modal correspondence from the instance level to the semantic level. We introduce Semantic Pairing (SemPair), a pairing-relaxed pretraining strategy that replaces exact instance pairs with class-consistent cross-instance pairs. To characterize the effect of correspondence relaxation, we formulate a latent-factor model that separates cross-modal dependence into shared semantic structure and instance-specific correlation, showing that SemPair preserves the former while marginalizing instance-specific cross-modal correlations in expectation. We further introduce Mixed Pairing (MixPair) as a controlled interpolation between semantic and exact pairing to study correspondence granularity along a continuous spectrum. Extensive experiments consistently show that SemPair achieves transfer performance comparable to exact pairing (ExaPair), while MixPair reveals a non-monotonic relationship be-<sup>.</sup> tween downstream transfer and the instance-pairing ratio. These results demonstrate that exact instance correspondence is not a prerequisite<sup>[</sup> for effective multimodal remote sensing pretraining and establish crossmodal correspondence granularity as an important design dimension for transferable multimodal representation learning. More broadly, this finding suggests a practical route toward scaling multimodal pretraining with reduced dependence on costly instance-aligned data.

Index Terms—Multimodal pretraining, remote sensing, cross-modal correspondence, semantic pairing, transferable representation learning

## 1 INTRODUCTION

Earth observation increasingly relies on multiple sensing modalities that provide complementary spectral, geometric, structural, and scattering information [1], [2]. Remote sensing research is also moving from task-specific learning toward reusable pretraining and transfer, making general representation learning an increasingly important problem [3]. Recent multimodal foundation models have shown that large-scale heterogeneous observations can support more general Earth observation representations [4]. This trend has continued toward larger multimodal corpora and broader downstream applicability [5]. A common assumption in these systems, however, is that observations from different modalities should correspond to the same physical instance or spatial location during pretraining.

Exact instance pairing provides a clear cross-modal relation, but the information captured by such correspondence is not necessarily identical to the information required for downstream transfer. The goal of pretraining is to learn representations that remain useful when tasks, regions, and sensing conditions change, rather than to reproduce the source domain pairing relation itself [6], [7]. Paired observations contain both semantic information that can recur across different instances and details that are specific to individual samples. Recent multimodal representation learning has also shown that shared and modality-specific information can play different roles in learned representations [8]. These observations raise a central question: Is exact instance correspondence necessary for learning transferable multimodal representations?

Remote sensing makes this question particularly important. Large-scale Earth observation data vary substantially in sensor type, spatial resolution, acquisition time, geographic coverage, and available supervision. Recent pretraining efforts have increasingly focused on learning from heterogeneous sensors, resolutions, and spatial scales [9]. Large multimodal models are also beginning to integrate a wider range of Earth observation signals within a common representation space [10]. At the data level, however, multimodal supervision is still commonly constructed from observations matched by spatial location and acquisition time [11]. Requiring all samples to form strict one-to-one multimodal pairs can nevertheless restrict the amount and diversity of data that can be readily organized for pretraining.

Existing multimodal learning mainly studies how heterogeneous information should be represented, aligned, fused, and transferred across modalities [12], [13]. Remote sensing research has similarly explored complementary information across heterogeneous observations and knowledge transfer between sensing domains [14], [15]. More recent multisensor and multimodal foundation models have extended this direction to broader sensor combinations and unified representation learning [4], [16]. Despite this progress, the correspondence relation used during pretraining is still largely determined by the available data rather than treated as a controllable learning variable. Whether exact instance correspondence is necessary, and how transferability changes when it is relaxed, therefore remain insufficiently understood.

This study addresses this gap by considering cross-modal correspondence granularity as an explicit design dimension of multimodal pretraining. Instead of treating exact pairing (ExaPair) as a fixed requirement, we explore how much correspondence should be preserved between modalities to learn transferable representations. This view places different levels of cross-modal correspondence within a common space, allowing the effect of correspondence level to be studied independently of changes in model architecture or training objective. Figure 1 illustrates this perspective by contrasting exact instance pairing with semantic pairing, where semantic consistency is preserved while exact instance identity is relaxed.

![](images/9ef17fba682c757252c6142b90e0293a0d38a2f1669a82dcb395e8daef2440bc.jpg)  
Fig. 1. Motivation for relaxing exact cross-modal correspondence in multimodal pretraining. ExaPair enforces same-instance correspondence, coupling semantic consistency with instance-specific dependence. Different instances from the same semantic class can still share common semantic structure despite variations in appearance and geometry. This motivates SemPair, which preserves semantic consistency while relaxing exact instance identity.

We investigate this problem through a controlled pairing-relaxed pretraining framework, using Semantic Pairing (SemPair) and Mixed Pairing (MixPair) to study different levels of cross-modal correspondence. We further combine theoretical analysis with extensive transfer experiments to examine the role of correspondence granularity in multimodal pretraining. The main contributions of this work are summarized as follows:

• We formulate cross-modal correspondence granularity as an explicit design dimension for multimodal remote sensing pretraining. Instead of assuming exact instancealigned pairs as a necessary form of supervision, we investigate how much cross-modal correspondence should be preserved to learn transferable multimodal representations.

• We introduce SemPair, a pairing-relaxed pretraining strategy that replaces exact instance correspondence with class-consistent cross-instance correspondence, without modifying the network architecture, training objective, or optimization budget. We further generalize ExaPair and SemPair into a continuous correspondence spectrum through MixPair, and provide a latent-factor analysis showing how correspondence granularity regulates the balance between shared semantic structure and instance-specific dependence.

• Extensive transfer experiments show that relaxing exact instance correspondence causes little loss in downstream performance. SemPair remains highly competitive with ExaPair across four multimodal semantic segmentation benchmarks, while the continuous pairingratio study further reveals that transferability does not improve monotonically with the instance-pairing ratio. These results demonstrate that ExaPair is neither indispensable nor consistently optimal for transferable multimodal pretraining.

• We systematically evaluate pairing-relaxed pretraining across two pretraining domains, four downstream multimodal segmentation benchmarks, different supervision and adaptation regimes, and multiple model scales. The results consistently demonstrate its effectiveness across diverse source-target conditions, establishing the robustness and generality of the proposed correspondence-relaxation paradigm.

This work provides a practical perspective for constructing larger multimodal pretraining datasets. By reducing the dependence on strictly paired observations, it offers a more flexible way to organize and utilize heterogeneous remote sensing data for pretraining. The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 presents the proposed pairing-relaxed pretraining framework and its theoretical analysis. Section 4 provides the experimental evaluation, analyses the effect of correspondence granularity under different transfer settings, and discusses the scope and limitations of the study. Finally, Section 5 concludes the paper.

## 2 RELATED WORK

## 2.1 Multimodal Representation Learning in Remote Sensing

Multimodal remote sensing combines observations from different sensing mechanisms to exploit complementary information [17]–[19]. Early deep-learning approaches mainly focused on how heterogeneous information should be represented and fused. Optical-elevation fusion demonstrated the value of combining appearance and geometric cues for very-high-resolution scene understanding [14]. Subsequent studies explored joint representation learning across heterogeneous observations [20], knowledge transfer between sensing domains [15], and the decomposition of shared and modality-specific information [21]. Manifold-based alignment has also been used to relate observations with substantially different sensing characteristics [22]. These studies established the benefit of complementary sensing, but generally operated on observations that were already matched at the instance or spatial level.

Recent architectures have further strengthened crossmodal interaction through more structured feature exchange. CMGFNet [23] introduces gated interaction between optical and elevation features, while Transformerbased FTransUNet [24] captures cross-modal dependencies across multiple representation levels. Feature compensation and reconstruction have also been introduced to improve the use of complementary information under modality differences [25]. Large pretrained encoders can further be adapted to optical-DSM segmentation through multimodal fine-tuning [26], [27]. Although these methods differ considerably in architecture and fusion mechanism, their main focus is how corresponding multimodal observations should interact. The correspondence relation between the inputs is usually inherited from the dataset and remains outside the model design.

## 2.2 Multimodal Remote Sensing Pretraining

A parallel line of research aims to learn reusable representations from large-scale remote sensing data. CMID [28] combines contrastive learning with masked image modeling, while RingMo [29] develops large-scale masked pretraining for remote sensing imagery. SpectralGPT [30] further extends foundation modeling to spatial-spectral representations. These studies demonstrate the effectiveness of pretraining for transferring knowledge across downstream tasks, but primarily focus on improving representation objectives or expanding the scale and diversity of the pretraining corpus.

Multisensor pretraining further exploits complementary observations across sensing modalities. SSL4EO-S12 [31] provides large-scale SAR-optical observations for selfsupervised representation learning, and multimodal selfsupervision has been used to reduce dependence on downstream annotations [32]. More recent foundation models extend this direction toward increasingly heterogeneous sensors, resolutions, and acquisition conditions [4], [9]. Large-scale multimodal frameworks further seek unified representations across diverse Earth observation signals [10], [16]. These developments indicate a clear trend toward broader multimodal pretraining and increasingly heterogeneous data sources.

Despite this progress, cross-modal supervision in such frameworks is still commonly constructed from observations that are spatially or temporally associated. When exact multimodal pairs are available, they are typically used directly as the basic correspondence unit for pretraining. As a result, improvements in architecture, objective, data scale, and sensor diversity have received substantial attention, whereas the correspondence relation itself has rarely been isolated as an independent variable. It therefore remains difficult to determine whether the benefits of multimodal pretraining arise from semantic information shared across modalities, from exact instance-level correspondence, or from both. This distinction becomes increasingly important when pretraining is expected to scale to heterogeneous data for which strict one-to-one pairing may be costly or unavailable.

## 2.3 Relaxed Cross-Modal Correspondence Learning

Broader multimodal learning has begun to consider incomplete, unaligned, and pair-free data, although these settings span substantially different modality combinations and learning objectives [33]. For image-image learning, several recent studies are more closely related to the problem considered here. MMGL [34] introduces self-supervised RGBinfrared pretraining with cross-modal cycle consistency for person re-identification under unpaired settings, while MCL [35] constructs pseudo cross-modal identity relations to support visible-infrared ReID without paired supervision. More recently, APTP and UPTP in Beyond Strict Pairing [36] explicitly challenge strictly paired infrared-visible image fusion and show that arbitrary or unpaired training can substantially relax the need for rigidly aligned inputs. These studies demonstrate that exact image-level pairing is not always indispensable, but they mainly address task-specific alignment, identity association, or fusion objectives rather than transferable multimodal pretraining.

The necessity of exact correspondence depends strongly on the learning objective. Multimodal registration and image matching explicitly require geometric or spatial correspondence, and relaxing this relation would change the task definition [37]. Semantic representation learning is different, because useful information can be shared across instances even when their local appearance, geometry, or identity is not exactly matched. Existing image-image studies have shown that strict pairing can be relaxed under specific tasks, but the broader role of cross-modal correspondence in learning transferable representations remains insufficiently understood. In particular, it is still unclear how much correspondence should be preserved between modalities and how different correspondence levels affect downstream transfer. This gap motivates our study of cross-modal correspondence granularity in multimodal remote sensing pretraining.

## 3 METHODOLOGY

## 3.1 Problem Formulation and Overview

We formulate cross-modal correspondence granularity as a controllable factor in multimodal pretraining. Specifically, we vary the correspondence between the two input modalities while keeping the multimodal architecture and learning objective unchanged. Let the labeled pretraining set be

$$
\begin{array} { r } { \mathbf { \mathcal { D } } = \left\{ \left( \mathbf { x } _ { i } ^ { a } , \mathbf { x } _ { i } ^ { b } , y _ { i } \right) \right\} _ { i = 1 } ^ { N } , } \end{array}\tag{1}
$$

where $\mathbf { x } _ { i } ^ { a }$ and $\mathbf { x } _ { i } ^ { b }$ are observations of two modalities associated with instance $i ,$ and $y _ { i } \in \{ 1 , \ldots , C \}$ is the corresponding class label. In our controlled pretraining experiments, modality a is RGB and modality b is pseudo-depth, while downstream transfer uses optical imagery and DSM. The source corpus contains exact pairs, allowing cross-modal correspondence to be deliberately changed without introducing additional samples or altering the semantic labels and training budget. Importantly, SemPair does not use the known exact associations during pretraining.

For compactness, we denote the fused prediction of the multimodal network by

$$
\begin{array} { r } { \widehat { \mathbf { p } } _ { i j } = g _ { \phi } \left( f _ { \theta } ( \mathbf { x } _ { i } ^ { a } , \mathbf { x } _ { j } ^ { b } ) \right) , } \end{array}\tag{2}
$$

where $f _ { \theta }$ denotes the jointly interactive multimodal encoder and $g _ { \phi }$ denotes the prediction head. Branch-specific predictions involved in the pretraining objective are omitted here for clarity. Conventional ExaPair fixes $j = i ,$ whereas our intervention changes only how the partner index $j$ is selected. Cross-modal correspondence is therefore treated as a property of the training distribution rather than a hard architectural requirement.

After pretraining, the classification head $g _ { \phi }$ is discarded and the pretrained multimodal encoder $f _ { \theta }$ is transferred to downstream multimodal semantic segmentation. The downstream inputs consist of naturally corresponding optical imagery and DSM, and a segmentation decoder is attached to the transferred encoder for dense prediction. No re-pairing is performed at this stage, as correspondence relaxation is applied only during pretraining.

![](images/56232e452480ade4835e7ee541f330ca21fd907c6bd9ef483f2c15207ff5f202.jpg)  
Fig. 2. Overview of pairing-relaxed multimodal pretraining and downstream transfer. Modality-a anchors are paired with modality-b samples according to the correspondence strategy. Only the cross-modal association varies among ExaPair, SemPair, and MixPair, while the multimodal encoder, pretraining objective, optimization budget, and downstream transfer protocol remain fixed.

The overall pipeline is shown in Figure 2. ExaPair, SemPair, and MixPair share the same network architecture, pretraining objective, and downstream transfer protocol. The only variable during pretraining is the pairing sampler, which controls the association between the two modality streams.

## 3.2 Pairing-Relaxed Multimodal Pretraining

## 3.2.1 ExaPair and SemPair

For each semantic class $c ,$ we first define the class-indexed training set

$$
{ \mathcal { T } } _ { c } = \{ k \mid y _ { k } = c \} .\tag{3}
$$

For an anchor instance $i ,$ its same-class cross-instance candidate set is

$$
{ \mathcal { C } } ( i ) = \{ j \mid y _ { j } = y _ { i } , j \neq i \} .\tag{4}
$$

ExaPair sets $j = i$ and therefore preserves exact instance correspondence across modalities. SemPair instead samples the modality-b partner according to

$$
p _ { \mathrm { S } } ( j \mid i ) = \frac { \mathbb { I } [ j \in \mathcal { C } ( i ) ] } { | \mathcal { C } ( i ) | } , \qquad j \sim p _ { \mathrm { S } } ( \cdot \mid i ) ,\tag{5}
$$

where $\mathbb { I } [ \cdot ]$ denotes the indicator function. Thus, all candidates in $\mathcal { C } ( i )$ are sampled with equal probability, ensuring that the two inputs share the same semantic label while corresponding to different instances. If $\mathcal { C } ( i )$ is empty, Sem-Pair falls back to the exact pair for that anchor. This case does not occur in the two pretraining datasets used in our experiments because every class contains multiple training samples. All candidate sets are constructed exclusively from the training partition, and the modality-a anchor $\mathbf { x } _ { i } ^ { a }$ remains unchanged.

SemPair is therefore different from random unpairing. An arbitrary cross-class partner could make the multimodal input semantically inconsistent with its supervision label. It is also different from geometric perturbation or modality dropout: both inputs remain complete and valid observations, while only their exact instance correspondence is relaxed. In our implementation, the pseudo-depth input for partner j is the precomputed depth map belonging to sample j and is never regenerated from, or copied from, anchor i.

We denote the training distributions induced by ExaPair and SemPair as $\mathbb { P } _ { \mathrm { E } }$ and $\mathbb { P } _ { \mathrm { S } } ,$ respectively. They preserve the same anchor and semantic-label marginals and differ only in the conditional association of modality-b given the anchor.

## 3.2.2 MixPair and Correspondence Ratio

MixPair connects ExaPair and SemPair through a continuous correspondence spectrum. We define $\rho \in [ 0 , 1 ]$ as the instance-pairing ratio, where $\rho = 1$ corresponds to ExaPair and $\rho = 0$ corresponds to SemPair.

For a requested ratio $\rho ,$ let $N _ { \mathrm { E } } ~ = ~ \lfloor \rho N \rfloor$ denote the number of anchors assigned to ExaPair in each epoch. The realized ratio $N _ { \mathrm { E } } / N$ differs from the requested value by less than 1/N. For simplicity, we use $\rho$ to denote the pairing ratio hereafter. At the beginning of epoch $e ,$ let $\pi _ { e }$ be a random permutation of all $\check { N }$ training indices and define

$$
{ \mathcal { T } } _ { e } ^ { \mathrm { E } } = \{ \pi _ { e } ( 1 ) , \ldots , \pi _ { e } ( N _ { \mathrm { E } } ) \} .\tag{6}
$$

The modality-b partner for anchor i is then selected as

$$
j _ { e } ( i ) = \left\{ \begin{array} { l l } { i , } & { i \in \mathcal { I } _ { e } ^ { \mathrm { E } } , } \\ { j , } & { j \sim p _ { \mathrm { S } } ( \cdot \mid i ) , } \\ { i , } & { \vert \mathcal { C } ( i ) \vert = 0 . } \end{array} \right.\tag{7}
$$

![](images/3d62d61965bf1da513d1b87bdf9ded0dbdf0c7dd838aca5140baaa7f27d4fad5.jpg)  
Fig. 3. ExaPair, SemPair, and MixPair for controlling cross-modal correspondence during pretraining. ExaPair preserves exact instance correspondence. SemPair keeps the modality-a anchor fixed and replaces modality b with a different sample from the same semantic class. MixPair assigns a fraction $\rho$ of training anchors to ExaPair and the remaining fraction $1 - \rho$ to SemPair, with the assignment reshuffled globally at every epoch.

The global permutation is independent of class identity. MixPair therefore does not impose a class-stratified allocation between ExaPair and SemPair, as it controls only the global epoch-level proportion. The assignment is regenerated at every epoch, allowing the same anchor to experience different correspondence modes during training. This epoch-wise reshuffling prevents the pairing mode from being permanently tied to particular samples and makes $\rho$ reflect the overall exposure to exact instance correspondence during training rather than a fixed partition of the dataset. Unless otherwise stated, MixPair uses $\rho = 0 . 5$

Figure 3 summarizes the three pairing strategies. ExaPair and SemPair form the two endpoints of the correspondence spectrum, while MixPair provides intermediate correspondence levels through $\rho .$ Because anchor sampling is unchanged and $y _ { j } ~ = ~ y _ { i }$ under SemPair, varying $\rho$ does not alter the number of anchors, the anchor-label distribution, or the number of optimizer steps. It changes only which modality-b observation accompanies each modality-a anchor.

## 3.2.3 Pretraining Procedure and Controlled Comparison

Algorithm 1 summarizes the training procedure. Classindexed candidate sets are constructed once from the train ing partition. At each epoch, MixPair first shuffles all training anchors and assigns the configured fraction to ExaPair. Mini-batches are then assembled according to Eq. (7) and optimized using the original pretraining objective.

The class index requires O(N) preprocessing, while partner lookup introduces negligible overhead relative to network optimization. Pair construction is used only during pretraining, and downstream inference uses the naturally corresponding modalities of each target sample.

For the class-supervised pretraining considered in this work, the original modality-specific and fused classification terms are retained without modification. Denoting them by $\mathcal { L } _ { a } , \mathcal { L } _ { b } ,$ , and $\mathcal { L } _ { f }$ , respectively, the pretraining objective is

$$
\mathcal { L } _ { \mathrm { c l s } } = \mathcal { L } _ { f } + \lambda _ { \mathrm { a u x } } ( \mathcal { L } _ { a } + \mathcal { L } _ { b } ) .\tag{8}
$$

The loss weighting emphasizes the fused prediction as the primary supervision, while the two modality-specific branches are assigned equal lower weights as auxiliary terms. Because $y _ { j } ~ = ~ y _ { i }$ under SemPair, the supervision remains valid for both modality-specific branches and the fused prediction.

More generally, correspondence relaxation is compatible with objectives whose supervision remains valid after within-class instance replacement. At the population level, MixPair can be represented by the mixture distribution

$$
\mathbb { P } _ { \rho } = \rho \mathbb { P } _ { \mathrm { E } } + ( 1 - \rho ) \mathbb { P } _ { \mathrm { S } } .\tag{9}
$$

This formulation makes $\rho \textbf { a }$ direct control over the relative exposure to ExaPair and SemPair during pretraining.

All partner indices are drawn strictly from the training partition, and validation samples never enter a semantic candidate set. The number of anchors per epoch, data augmentations, input normalization, spatial resolution, optimizer steps, and checkpoint criterion are kept fixed across pairing strategies. SemPair enlarges the set of possible crossmodal associations but does not expose the optimizer to additional anchors or a larger training budget. These controls isolate correspondence granularity from changes in data volume, augmentation strength, or model capacity.

## 3.3 Theoretical Analysis of Correspondence Granularity

Cross-modal correspondence also acts as an inductive bias because it changes which shared factors are repeatedly presented to a multimodal learner. We therefore develop a simple latent-factor model to characterize how exact and relaxed correspondence modify the cross-modal statistical structure observed during pretraining. The analysis is intended as a structural interpretation rather than an exact description of a nonlinear neural network.

```latex
Algorithm 1: Multimodal Pretraining with Controlled Cross-Modal Correspondence
Input: Pretraining set D, instance-pairing ratio $\rho ,$ network $\left( f _ { \theta } , g _ { \phi } \right)$
Output: Pretrained multimodal encoder $f _ { \theta ^ { * } }$
1 Build class-indexed sets $\{ \mathcal { T } _ { c } \} _ { c = 1 } ^ { C }$ from the training split
2 Set $N _ { \mathrm { E } } \gets \lfloor \rho N \rfloor$ $/ / \rho = 1$ : ExaPair; $\rho = 0 ;$ : SemPair; otherwise: MixPair
3 foreach epoch e do
4 Randomly permute all training indices and construct $\mathcal { T } _ { e } ^ { \mathrm { E } }$ with $N _ { \mathrm { E } }$ anchors
5 foreach mini-batch B do
6 foreach $i \in \boldsymbol { B }$ do
7 if $i \in \mathcal { T } _ { e } ^ { \mathrm { E } }$ then
8 $j ( i )  i$ $/ /$ ExaPair path
9 else
10 Sample $j ( i )$ uniformly from $\mathcal { C } ( i )$ $/ /$ SemPair path
11 Form multimodal pairs $\{ ( \mathbf { x } _ { i } ^ { a } , \mathbf { x } _ { j ( i ) } ^ { b } , y _ { i } ) : i \in B \}$
12 Compute $\mathcal { L } _ { \mathrm { c l s } }$ and update $\theta ,$ ϕ
13 return $f _ { \theta ^ { \ast } }$
```

## 3.3.1 Semantic and Instance Factors

Consider modalities a and b. For instance i with semantic label $y _ { i } ,$ we decompose each observation into a classdependent semantic factor, an instance-dependent factor, and a modality-specific residual:

$$
\begin{array} { r } { \mathbf { x } _ { i } ^ { a } = \mathbf { A } _ { a } \mathbf { s } _ { y _ { i } } + \mathbf { B } _ { a } \mathbf { u } _ { i } + \pmb { \epsilon } _ { i } ^ { a } , } \\ { \mathbf { x } _ { i } ^ { b } = \mathbf { A } _ { b } \mathbf { s } _ { y _ { i } } + \mathbf { B } _ { b } \mathbf { u } _ { i } + \pmb { \epsilon } _ { i } ^ { b } . } \end{array}\tag{10}
$$

Here, $\mathbf { s } _ { y _ { i } }$ denotes semantic structure shared by samples of class y<sub>i</sub>, u<sub>i</sub> represents instance-specific properties, and $\epsilon _ { i } ^ { a }$ and $\epsilon _ { i } ^ { b }$ capture modality-specific residual variations. The matrices $\bar { \mathbf { A } } _ { a } , \mathbf { A } _ { b } , \mathbf { B } _ { a } ,$ and $\bar { \mathbf { B } } _ { b }$ describe how the two modalities observe the corresponding latent factors. This additive model is deliberately minimal and does not imply that the nonlinear encoder itself performs such a decomposition. It provides a tractable description of the second-order crossmodal structure induced by different pairing distributions.

We assume that all variables are globally centered and that

$$
\mathbb { E } [ { \mathbf { u } } _ { i } \mid y _ { i } ] = \mathbf { 0 } .\tag{11}
$$

Distinct instances are assumed conditionally independent given their semantic class. The modality-specific residuals are assumed zero-mean and uncorrelated with the semantic and instance factors. Their cross-modal covariance is also assumed zero. These assumptions define $\mathbf { s } _ { y }$ as the classconsistent component and place residual within-class variability in $\mathbf { u } _ { i } .$ Systematic class-related structure is therefore absorbed into $\mathbf { s } _ { y } ,$ whereas pose, local geometry, texture realization, object configuration, or acquisition-specific details may remain in $\mathbf { u } _ { i }$

Under ExaPair, the two observations share both the semantic factor $\mathbf { s } _ { y _ { i } }$ and the instance factor $\mathbf { u } _ { i }$ . Under SemPair, $( \mathbf { x } _ { i } ^ { a } , \mathbf { x } _ { j } ^ { b } )$ satisfies $i \neq j$ and $y _ { i } = y _ { j }$ , so the semantic factor remains common while the instance factors differ.

## 3.3.2 SemPair Marginalizes Instance Correlation

Define the semantic and within-class instance covariances as

$$
\pmb { \Sigma } _ { s } = \mathbb { E } [ \mathbf { s } _ { y } \mathbf { s } _ { y } ^ { \top } ] , \qquad \pmb { \Sigma } _ { u } = \mathbb { E } _ { y } \left[ \mathbb { E } [ \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \mid y ] \right] .\tag{12}
$$

Proposition 1: Cross-modal covariance decomposition. Under Eq. (10) and the assumptions above, ExaPair induces

$$
\begin{array} { r } { \pmb { \Sigma } _ { a b } ^ { \mathrm { E } } = \underbrace { \mathbf { A } _ { a } \pmb { \Sigma } _ { s } \mathbf { A } _ { b } ^ { \top } } _ { \mathbf { s } } + \underbrace { \mathbf { B } _ { a } \pmb { \Sigma } _ { u } \mathbf { B } _ { b } ^ { \top } } _ { \mathbf { U } } , } \end{array}\tag{13}
$$

whereas SemPair yields

$$
\begin{array} { r } { \pmb { \Sigma } _ { a b } ^ { \mathrm { S } } = \mathbf { S } . } \end{array}\tag{14}
$$

Proof: Under ExaPair, the two modalities contain the same $\mathbf { s } _ { y _ { i } }$ and $\mathbf { u } _ { i } .$ . Expanding $\mathbb { E } [ { \mathbf { x } } _ { i } ^ { a } ( { \mathbf { x } } _ { i } ^ { b } ) ^ { \top } ]$ therefore produces the semantic term S and the shared instance term ${ \bf { \bar { U } } } ,$ while the remaining cross terms vanish under the stated assumptions.

Under SemPair, distinct instance factors $\mathbf { u } _ { i }$ and $\mathbf { u } _ { j }$ are conditionally independent given y. Hence,

$$
\begin{array} { r } { \mathbb { E } [ \mathbf { u } _ { i } \mathbf { u } _ { j } ^ { \top } \mid y ] = \mathbb { E } [ \mathbf { u } _ { i } \mid y ] \mathbb { E } [ \mathbf { u } _ { j } \mid y ] ^ { \top } = \mathbf { 0 } , } \end{array}\tag{15}
$$

which removes the shared instance term while leaving the class-dependent semantic covariance unchanged.

Proposition 1 captures the central statistical distinction between the two pairing strategies. ExaPair repeatedly presents both class-consistent semantic structure and within-instance cross-modal correlation. SemPair preserves the former while marginalizing, in expectation, the correlation supported only by exact instance identity. This statement concerns the pairing distribution presented during pretraining, and it does not imply that a trained nonlinear representation contains no instance-specific information.

The same distinction can be expressed from an information-theoretic perspective under a stronger conditional-independence version of the latent model. Under SemPair, the selected instances and their modalityspecific residuals are conditionally independent given the semantic class, and this gives

$$
\operatorname { p } _ { \mathrm { S } } ( \mathbf { X } ^ { a } , \mathbf { X } ^ { b } \mid Y ) = p _ { \mathrm { S } } ( \mathbf { X } ^ { a } \mid Y ) p _ { \mathrm { S } } ( \mathbf { X } ^ { b } \mid Y ) ,\tag{16}
$$

and consequently

$$
I _ { \mathrm { S } } ( \mathbf { X } ^ { a } ; \mathbf { X } ^ { b } \mid Y ) = 0 .\tag{17}
$$

In contrast, ExaPair shares the same instance variable $\mathbf { u } _ { i }$ across modalities. When this shared instance factor induces non-degenerate dependence in both modalities, the conditional mutual information is positive,

$$
I _ { \mathrm { E } } ( \mathbf { X } ^ { a } ; \mathbf { X } ^ { b } \mid Y ) > 0 .\tag{18}
$$

SemPair therefore removes within-class cross-modal instance dependence under the assumed model while preserving the shared semantic relation through Y .

## 3.3.3 A Continuum ofCorrespondence Granularity

MixPair induces the population correspondence distribution in Eq. (9). Combining this mixture with Proposition 1 gives the following result.

Corollary 1. The cross-modal covariance under instancepairing ratio $\rho$ is

$$
\begin{array} { r } { \sum _ { a b } ( \rho ) = \mathbf { S } + \rho \mathbf { U } . } \end{array}\tag{19}
$$

Equation (19) gives $\rho$ a direct structural interpretation: it controls the contribution of shared instance-specific correlation presented during pretraining. The two endpoints satisfy

$$
\rho = 0 \Longleftrightarrow \mathrm { S e m a n t i c ~ P a i r i n g } , \qquad \rho = 1 \Longleftrightarrow \mathrm { E x a c t ~ P a i r i n g } .\tag{20}
$$

Intermediate values retain corresponding fractions of the exact-pair component in expectation. The central question is therefore not simply whether cross-modal correspondence is present, but how much exact instance correspondence should be retained.

The epoch-wise global assignment described in Section 3.2 provides a finite-sample realization of this population mixture. Averaging over the random assignments recovers the covariance form in Eq. (19).

## 3.3.4 Transferability and Optimal Correspondence Granularity

To relate correspondence granularity to downstream transfer, let $\mathbf { M } _ { t }$ represent the cross-modal structure relevant to target task $t ,$ expressed in the same surrogate comparison space as S and U. We define the structural mismatch as

$$
\mathcal { D } _ { t } ( \rho ) = \| \mathbf { S } + \rho \mathbf { U } - \mathbf { M } _ { t } \| _ { F } ^ { 2 } .\tag{21}
$$

Equation (21) is not intended to equal the downstream risk of a nonlinear model. Instead, it provides a surrogate measure of compatibility between the cross-modal structure emphasized during pretraining and the structure relevant to the target task.

Proposition 2: Optimal correspondence level. For $\mathbf { U } \neq \mathbf { 0 } ,$ the correspondence level minimizing Eq. (21) is

$$
\rho ^ { * } = \Pi _ { [ 0 , 1 ] } \left( \frac { \langle \mathbf { U } , \mathbf { M } _ { t } - \mathbf { S } \rangle _ { F } } { \| \mathbf { U } \| _ { F } ^ { 2 } } \right) ,\tag{22}
$$

where $\Pi _ { [ 0 , 1 ] }$ denotes projection onto [0, 1]. If $\textbf { U } = \textbf { 0 }$ , the structural mismatch is independent of $\rho .$

Proof: Expanding Eq. (21) with respect to $\rho$ yields a convex quadratic. Setting its derivative to zero gives the unconstrained optimum

$$
\frac { \langle \mathbf { U } , \mathbf { M } _ { t } - \mathbf { S } \rangle _ { F } } { \Vert \mathbf { U } \Vert _ { F } ^ { 2 } } ,\tag{23}
$$

which is then projected onto the feasible interval [0, 1]. Proposition 2 indicates that the preferred correspondence level depends on how strongly the instance-specific structure presented during pretraining matches the structure relevant to the target task. SemPair is favored when this component contributes little to the target-relevant structure, whereas ExaPair is favored when it remains strongly relevant. Intermediate correspondence can be preferable when only part of the instance-specific structure remains relevant. An intuitive special case is

$$
{ \bf M } _ { t } = { \bf S } + \gamma { \bf U } , 0 \leq \gamma \leq 1 ,\tag{24}
$$

for which Eq. (22) reduces to

$$
\rho ^ { * } = \gamma .\tag{25}
$$

![](images/98af60adb830a30bd2a55ce22d2cc87b6e520fc81205fa83adbd2501eeb87686.jpg)  
Fig. 4. Representative samples from the two pretraining domains. ImageNet-RS90 provides an object-centric natural-image source, whereas NWPU-RGBD-45 provides overhead remote-sensing scenes. Both are organized as RGB-pseudo-depth inputs for controlled correspondence experiments.

The preferred correspondence level therefore tracks how much of the instance-specific cross-modal structure remains relevant to the target task. Within this surrogate analysis, increasing the instance-pairing ratio is not guaranteed to reduce the structural mismatch monotonically. This motivates the empirical study in the next section of how downstream transfer changes across different levels of exact instance correspondence.

## 4 EXPERIMENTS

The proposed pairing-relaxed pretraining framework is evaluated by examining how cross-modal correspondence granularity affects downstream transfer. The experiments first establish overall transferability on four multimodal semantic-segmentation benchmarks, then analyze continuous correspondence granularity and representation structure, and finally test the consistency of the observations across pretraining domains, supervision regimes, adaptation strengths, and model scales. These experiments assess whether relaxed cross-modal correspondence can provide consistently transferable multimodal representations under diverse pretraining and downstream conditions.

## 4.1 Experimental Setup

## 4.1.1 Pretraining Datasets

We use two source domains with markedly different image statistics as summarized in Table 1. ImageNet-RS90 is a 90- class subset of ImageNet [38] selected to provide semantically meaningful object-centric categories. After associating each RGB image with a precomputed pseudo-depth map, the corpus contains 116300 samples, of which 110483 are used for training and 5817 for validation. The number of samples per class ranges from 1152 to 1300. This source is intentionally different from overhead remote sensing and therefore provides a useful test of transfer across both task and image domain.

NWPU-RGBD-45 is constructed from NWPU-RESISC45 [39]. It contains 45 overhead scene categories with 700 images per category, giving 31500 RGB-pseudo-depth samples. We use 29925 for training and 1575 for validation under the same class-wise validation ratio. Compared with ImageNet-RS90, this source is substantially closer to the downstream remote-sensing imagery in viewpoint and scene composition.

For both sources, pseudo-depth is generated once before pretraining by Depth Anything V2 [40] and then treated as the second modality. Figure 4 presents representative samples from the two pretraining datasets. ExaPair uses the RGB image and the pseudo-depth map of the same instance. SemPair keeps the RGB anchor but draws the pseudo-depth map from another training sample of the same class. MixPair uses the epoch-wise global ratio defined in Section 3.2, with $\rho = 0 . 5$ by default. Semantic repairing is performed only within the training split. RGBonly retains the structure-matched dual-stream backbone while providing RGB information to both streams, following the corresponding unimodal setting of the backbone.

TABLE 1  
Summary of the pretraining datasets. Pseudo-depth maps are generated from RGB images using the same monocular depth estimation pipeline.
<table><tr><td>Dataset</td><td>Source domain</td><td>Classes</td><td>Pairs</td><td>Train</td><td>Val.</td><td>Input</td></tr><tr><td> $\mathrm { I m a g e N e t – R S 9 0 }$ </td><td>ImageNet subset</td><td>90</td><td>116300</td><td>110483</td><td>5817</td><td> $2 2 4 \times 2 2 4$ </td></tr><tr><td> $N W { \mathrm { P U } } \mathrm { - R G B D } { - } 4 5$ </td><td>Remote-sensing scenes</td><td>45</td><td>31500</td><td>29925</td><td>1575</td><td> $2 2 4 \times 2 2 4$ </td></tr></table>

TABLE 2

Summary of the downstream multimodal segmentation benchmarks. GeoNRW uses the Gelsenkirchen subset.
<table><tr><td>Dataset</td><td>Resolution (m)</td><td>Classes</td><td>Train</td><td>Validation</td><td>Test</td><td>Tile size</td><td>Crop size</td></tr><tr><td>Potsdam</td><td>0.05</td><td>6</td><td>18</td><td>6</td><td>14</td><td> $6 0 0 0 \times 6 0 0 0$ </td><td> $5 1 2 \times 5 1 2$ </td></tr><tr><td>Vaihingen</td><td>0.09</td><td>6</td><td>12</td><td>4</td><td>17</td><td> $\mathrm { a b o u t 2 5 0 0 \times 2 0 0 0 }$ </td><td> $5 1 2 \times 5 1 2$ </td></tr><tr><td>MMHunan</td><td>10</td><td>7</td><td>300</td><td>50</td><td>150</td><td> $2 5 6 \times 2 5 6$ </td><td> $2 5 6 \times 2 5 6$ </td></tr><tr><td>GeoNRW</td><td>1</td><td>6</td><td>82</td><td>13</td><td>42</td><td> $1 0 0 0 \times 1 0 0 0$ </td><td> $5 1 2 \times 5 1 2$ </td></tr></table>

## 4.1.2 Downstream Benchmarks

The pretrained encoders are transferred to four multimodal semantic-segmentation benchmarks of equal status: Potsdam, Vaihingen, MMHunan [41], and GeoNRW [42]. Table 2 summarizes the key information of the four downstream benchmarks. They cover different geographic regions, spatial resolutions, scene compositions, and landcover taxonomies. Optical imagery and DSM are used as the two downstream modalities. Since monocular pseudodepth and elevation DSM have opposite geometric conventions in our preprocessing pipeline, the DSM is inverted before entering the depth stream to reduce this orientation mismatch. This operation does not make pseudo-depth and DSM physically equivalent.

Potsdam and Vaihingen additionally support controlled low-label and adaptation analyses. For these two datasets, clutter is excluded from the reported mean IoU and the five foreground categories are used for class-wise comparison. Representative samples from all four datasets are shown in Figure 5.

## 4.1.3 Implementation Details

Pretraining: We adopt DFormer-Small as the default mul timodal encoder unless otherwise stated [43]. Its explicit cross-modal interaction provides a representative backbone for studying correspondence granularity. Pretraining uses $2 2 4 \times 2 2 4$ inputs for 50 epochs with AdamW, a learning rate of $1 0 ^ { - 3 }$ , and weight decay of 0.05. The learning rate follows a cosine schedule with a five-epoch linear warm-up and a minimum value of $1 0 ^ { - 6 }$ . The default configuration uses two GPUs with 64 samples per GPU, giving a global batch size of 128. We apply Mixup and CutMix with parameters 0.8 and 1.0, respectively, label smoothing of 0.1, and stochastic depth with a drop-path rate of 0.1. ExaPair, MixPair, and SemPair use identical settings across all pretraining configurations, with only the cross-modal correspondence strategy varying.

![](images/af3c8b892b371638c46a9b8eb8e5041c109f5e9715455218d4afb41a70d15fd4.jpg)  
Fig. 5. Representative multimodal samples and semantic annotations from the downstream benchmarks: (a) Potsdam, (b) Vaihingen, (c) MMHunan, and (d) GeoNRW. Each example shows the optical image, the corresponding elevation representation, and the ground-truth semantic mask.

Downstream transfer: The pretraining classification heads are discarded, and the encoder is coupled with the same lightweight segmentation decoder for all strategies. Downstream training uses a batch size of 4 for 50 epochs. We use SGD with momentum 0.9 and weight decay $5 \times 1 0 ^ { - 4 } .$ Standard fine-tuning uses encoder and decoder learning rates of $1 0 ^ { - 3 }$ and $1 0 ^ { - 2 }$ , respectively. Both learning rates are reduced by a factor of 0.1 at epochs 25, 35, and 45. Frozen transfer fixes all encoder parameters and trains only the decoder, whereas the scratch baseline trains both components from random initialization with a learning rate of $1 \dot { 0 } ^ { - 2 }$ . The downstream protocol is held fixed across pairing strategies within each benchmark and adaptation setting.

## 4.1.4 Evaluation Protocol

For semantic segmentation, we report overall accuracy (OA), mean F1 score (mF1), Cohen’s kappa, class-wise intersection over union (IoU), and mean IoU (mIoU). Each configuration is independently run twice with the same protocol, and the better of the two runs is reported uniformly for all methods. All quantitative downstream results are reported on the validation split. Within each run, the checkpoint with the highest validation mIoU is selected, and all accompanying metrics are computed on the same validation split using that checkpoint.

For representation-level retrieval, we use the fixed validation split of each pretraining dataset and keep the original full DFormer forward path unchanged. RGB-stream features are used as queries and depth-stream features as the gallery. Instance R@1 uses the complete gallery and asks whether the exact depth counterpart is ranked first. For semantic retrieval, the exact same-instance depth sample is removed from the gallery before computing semantic R@1 and R@5. These metrics therefore measure whether a different instance of the same class can be retrieved. As DFormer is a jointly interactive encoder, these results are interpreted as properties of the complete learned multimodal representation rather than as measurements of isolated unimodal encoders.

TABLE 3  
Pretraining validation results. All metrics in a row are taken from the epoch with the highest fused-branch Acc@1.
<table><tr><td>Dataset</td><td>Pretraining</td><td>Val. loss ↓</td><td>Acc@1 ↑</td><td>Acc@5 ↑</td><td>RGB Acc@1 ↑</td><td>Depth Acc@1 ↑</td></tr><tr><td rowspan="4">ImageNet-RS90</td><td>RGB-only</td><td>0.8060</td><td>82.50</td><td>97.04</td><td>82.12</td><td>/</td></tr><tr><td>ExaPair</td><td>0.8398</td><td>81.80</td><td>96.80</td><td>81.78</td><td>80.85</td></tr><tr><td>MixPair</td><td>0.8509</td><td>81.08</td><td>96.42</td><td>81.20</td><td>79.84</td></tr><tr><td>SemPair</td><td>0.9086</td><td>79.25</td><td>95.65</td><td>79.13</td><td>78.53</td></tr><tr><td rowspan="4">NWPU-RGBD-45</td><td>RGB-only</td><td>0.5563</td><td>90.04</td><td>99.18</td><td>89.85</td><td>/</td></tr><tr><td>ExaPair</td><td>0.4783</td><td>91.88</td><td>99.18</td><td>91.62</td><td>91.37</td></tr><tr><td>MixPair</td><td>0.5060</td><td>90.80</td><td>99.30</td><td>90.93</td><td>90.29</td></tr><tr><td>SemPair</td><td>0.5676</td><td>88.32</td><td>98.98</td><td>87.94</td><td>88.07</td></tr></table>

TABLE 4

Primary transfer results on Potsdam. Imp., Build., and Low veg. denote impervious surfaces, buildings, and low vegetation, respectively.
<table><tr><td rowspan="2">Adaptation</td><td rowspan="2">Pretraining</td><td rowspan="2">OA</td><td rowspan="2">mF1</td><td rowspan="2">Kappa</td><td colspan="5">Class-wise IoU</td><td rowspan="2">mIoU</td></tr><tr><td>Imp.</td><td>Build.</td><td>Low veg.</td><td>Tree</td><td>Car</td></tr><tr><td>Scratch</td><td>None</td><td>78.74</td><td>76.83</td><td>71.79</td><td>72.78</td><td>87.19</td><td>54.53</td><td>43.58</td><td>60.61</td><td>63.74</td></tr><tr><td rowspan="4">Frozen</td><td>RGB-only</td><td>84.36</td><td>85.39</td><td>79.31</td><td>76.41</td><td>90.31</td><td>63.62</td><td>64.55</td><td>80.49</td><td>75.07</td></tr><tr><td>ExaPair</td><td>85.68</td><td>86.58</td><td>81.15</td><td>79.41</td><td>91.08</td><td>65.21</td><td>65.41</td><td>83.53</td><td>76.93</td></tr><tr><td>MixPair</td><td>85.81</td><td>86.77</td><td>81.27</td><td>79.58</td><td>90.94</td><td>65.68</td><td>65.63</td><td>84.26</td><td>77.22</td></tr><tr><td>SemPair</td><td>85.72</td><td>86.67</td><td>81.16</td><td>79.72</td><td>91.03</td><td>64.69</td><td>66.06</td><td>83.79</td><td>77.06</td></tr><tr><td rowspan="4">Fine-tuning</td><td>RGB-only</td><td>87.90</td><td>88.84</td><td>84.08</td><td>82.79</td><td>91.94</td><td>69.25</td><td>70.89</td><td>86.98</td><td>80.37</td></tr><tr><td>ExaPair</td><td>88.04</td><td>88.97</td><td>84.28</td><td>82.89</td><td>91.98</td><td>70.04</td><td>70.49</td><td>87.42</td><td>80.57</td></tr><tr><td>MixPair</td><td>88.26</td><td>89.05</td><td>84.53</td><td>83.28</td><td>91.89</td><td>70.12</td><td>70.94</td><td>87.16</td><td>80.68</td></tr><tr><td>SemPair</td><td>88.18</td><td>89.03</td><td>84.44</td><td>83.13</td><td>91.88</td><td>70.18</td><td>71.11</td><td>86.89</td><td>80.64</td></tr></table>

## 4.2 Overall Transferability on Multimodal Benchmarks

We first test the central claim directly: does a multimodal encoder remain transferable when exact instance pairing is removed during pretraining? Table 3 reports source validation performance during pretraining, and Tables 4-7 report downstream segmentation under frozen transfer and standard fine-tuning.

## 4.2.1 Source Fitting Versus Downstream Transfer

The pretraining validation results in Table 3 show a consistent advantage of ExaPair over MixPair and SemPair in source-domain classification. The results on ImageNet-RS90 indicate that ExaPair achieves a fused Acc@1 of 81.80%, compared with 81.08% for MixPair and 79.25% for SemPair. A similar trend is observed on NWPU-RGBD-45, where the corresponding accuracies are 91.88%, 90.80%, and 88.32%. These results indicate that preserving exact instance correspondence facilitates source-domain fitting, as expected from the additional cross-modal instance information available to ExaPair. Nevertheless, relaxing correspondence does not lead to optimization failure or representation collapse: both MixPair and SemPair retain strong classification performance on the two pretraining datasets. This suggests that exact instance correspondence is beneficial, but not indispensable, for learning meaningful multimodal representations during pretraining.

4.2.2 Primary Results on Four Multimodal Benchmarks

Tables 4-7 report the primary transfer results on four multimodal semantic-segmentation benchmarks under both frozen-encoder evaluation and standard end-to-end finetuning. The frozen setting provides a direct assessment of the pretrained representation because the downstream decoder must operate on a fixed encoder, whereas finetuning allows the pretrained representation to adapt to the target task.

Before comparing the three cross-modal pairing strategies, the baseline results provide two useful sanity checks. First, RGB-only pretraining consistently improves over training from scratch, confirming the effectiveness of the pretraining-and-transfer protocol itself. Second, the three multimodal strategies generally provide further gains over the RGB-only initialization, showing that the pseudo-depth modality contributes useful complementary information rather than merely increasing the input complexity. These observations establish a meaningful multimodal transfer setting in which the effect of cross-modal correspondence can be examined independently.

Potsdam: As shown in Table 4, ExaPair, MixPair, and SemPair achieve 76.93%, 77.22%, and 77.06% mIoU, respectively, compared with 75.07% for RGB-only pretraining. Notably, removing exact instance correspondence in Sem-Pair does not reduce frozen transfer performance. SemPair’s mIoU is slightly higher than that of ExaPair, and MixPair gives the best overall result. After fine-tuning, the three pairing strategies become even closer, reaching 80.57%, 80.68%, and 80.64% mIoU. Their spread is only 0.11%, indicating that the transferability of the pretrained representation is largely preserved across substantially different correspondence levels. The class-wise results show a similar pattern. SemPair achieves the highest IoUs for low vegetation and trees, while MixPair obtains the highest overall mIoU, suggesting that the effect is not driven by a single semantic category.

Vaihingen: The results on Vaihingen in Table 5 provide an even clearer comparison under frozen transfer. RGBonly pretraining reaches 67.14% mIoU, whereas ExaPair, MixPair, and SemPair improve it to 70.16%, 71.53%, and 70.61%, respectively. SemPair therefore exceeds ExaPair by 0.45% despite never observing exact cross-modal instance pairs during pretraining, while MixPair achieves the highest frozen performance. After fine-tuning, the differences remain small: ExaPair obtains 74.13%, MixPair 74.52%, and SemPair 73.94% mIoU. Although SemPair is 0.19% below ExaPair in this setting, both remain above the RGB-only baseline of 72.49%. The class-wise results also show no systematic dominance of ExaPair: ExaPair performs best on impervious surfaces and buildings, MixPair on low vegetation and cars, and SemPair on trees. This distribution of gains suggests that correspondence relaxation does not simply favor or suppress a particular land-cover category.

TABLE 5  
Primary transfer results on Vaihingen. Imp., Build., and Low veg. denote impervious surfaces, buildings, and low vegetation, respectively.
<table><tr><td rowspan="2">Adaptation</td><td rowspan="2">Pretraining</td><td rowspan="2">OA</td><td rowspan="2">mF1</td><td rowspan="2">Kappa</td><td colspan="5">Class-wise IoU</td><td rowspan="2">mIoU</td></tr><tr><td>Imp.</td><td>Build.</td><td>Low veg.</td><td>Tree</td><td>Car</td></tr><tr><td>Scratch</td><td>None</td><td>82.67</td><td>72.19</td><td>76.69</td><td>71.81</td><td>81.12</td><td>50.02</td><td>74.51</td><td>21.71</td><td>59.83</td></tr><tr><td rowspan="4">Frozen</td><td>RGB-only</td><td>84.03</td><td>79.61</td><td>78.50</td><td>71.22</td><td>81.72</td><td>52.66</td><td>78.90</td><td>51.23</td><td>67.14</td></tr><tr><td>ExaPair</td><td>85.80</td><td>81.93</td><td>80.89</td><td>74.35</td><td>82.74</td><td>58.13</td><td>80.33</td><td>55.25</td><td>70.16</td></tr><tr><td>MixPair</td><td>86.63</td><td>82.88</td><td>82.01</td><td>75.89</td><td>84.67</td><td>59.00</td><td>80.96</td><td>57.12</td><td>71.53</td></tr><tr><td>SemPair</td><td>86.20</td><td>82.22</td><td>81.42</td><td>74.83</td><td>83.49</td><td>59.40</td><td>80.80</td><td>54.51</td><td>70.61</td></tr><tr><td rowspan="4">Fine-tuning</td><td>RGB-only</td><td>87.38</td><td>83.48</td><td>83.02</td><td>78.11</td><td>87.44</td><td>59.09</td><td>80.29</td><td>57.56</td><td>72.49</td></tr><tr><td>ExaPair</td><td>88.83</td><td>84.52</td><td>84.94</td><td>81.47</td><td>89.60</td><td>60.56</td><td>81.38</td><td>57.66</td><td>74.13</td></tr><tr><td>MixPair</td><td>88.31</td><td>84.95</td><td>84.28</td><td>79.71</td><td>87.88</td><td>62.08</td><td>81.50</td><td>61.41</td><td>74.52</td></tr><tr><td>SemPair</td><td>88.36</td><td>84.49</td><td>84.32</td><td>79.94</td><td>88.13</td><td>61.31</td><td>81.57</td><td>58.74</td><td>73.94</td></tr></table>

TABLE 6

Primary transfer results on MMHunan. Crop., Grass., Wet., and Unused denote cropland, grassland, wetland, and unused land, respectively.
<table><tr><td rowspan="2">Adaptation</td><td rowspan="2">Pretraining</td><td rowspan="2">OA</td><td rowspan="2">mF1</td><td rowspan="2">Kappa</td><td colspan="7">Class-wise IoU</td><td rowspan="2">mIoU</td></tr><tr><td>Crop.</td><td>Forest</td><td>Grass.</td><td>Wet.</td><td>Water</td><td>Unused</td><td>Built-up</td></tr><tr><td>Scratch</td><td>None</td><td>68.04</td><td>43.90</td><td>55.01</td><td>45.69</td><td>68.35</td><td>10.68</td><td>0.00</td><td>60.70</td><td>22.33</td><td>19.07</td><td>32.40</td></tr><tr><td rowspan="4">Frozen</td><td>RGB-only</td><td>72.96</td><td>50.72</td><td>61.72</td><td>47.79</td><td>71.91</td><td>13.24</td><td>0.32</td><td>65.36</td><td>51.77</td><td>21.52</td><td>38.84</td></tr><tr><td>ExaPair</td><td>74.14</td><td>53.46</td><td>63.54</td><td>51.48</td><td>71.97</td><td>14.06</td><td>4.91</td><td>70.54</td><td>51.65</td><td>23.24</td><td>41.12</td></tr><tr><td>MixPair</td><td>74.64</td><td>53.13</td><td>64.27</td><td>52.04</td><td>72.47</td><td>13.33</td><td>4.62</td><td>71.03</td><td>52.54</td><td>21.30</td><td>41.05</td></tr><tr><td>SemPair</td><td>74.36</td><td>52.95</td><td>63.80</td><td>51.46</td><td>72.19</td><td>13.38</td><td>3.60</td><td>71.00</td><td>53.14</td><td>21.84</td><td>40.94</td></tr><tr><td rowspan="4">Fine-tuning</td><td>RGB-only</td><td>74.51</td><td>53.70</td><td>64.28</td><td>51.16</td><td>74.01</td><td>17.11</td><td>1.91</td><td>66.80</td><td>53.28</td><td>25.45</td><td>41.39</td></tr><tr><td>ExaPair</td><td>74.23</td><td>54.28</td><td>63.91</td><td>51.81</td><td>72.40</td><td>15.59</td><td>5.94</td><td>69.55</td><td>53.77</td><td>23.09</td><td>41.74</td></tr><tr><td>MixPair</td><td>75.22</td><td>55.46</td><td>65.27</td><td>52.85</td><td>73.04</td><td>17.32</td><td>5.83</td><td>73.67</td><td>53.02</td><td>24.98</td><td>42.96</td></tr><tr><td>SemPair</td><td>75.24</td><td>54.69</td><td>65.22</td><td>53.10</td><td>73.32</td><td>15.00</td><td>5.91</td><td>72.59</td><td>53.90</td><td>23.02</td><td>42.41</td></tr></table>

TABLE 7

Primary transfer results on GeoNRW. Agric. and Grass. denote agricultural areas and grassland, respectively.
<table><tr><td rowspan="2">Adaptation</td><td rowspan="2">Pretraining</td><td rowspan="2">OA</td><td rowspan="2">mF1</td><td rowspan="2">Kappa</td><td colspan="6">Class-wise IoU</td><td rowspan="2">mIoU</td></tr><tr><td>Forest</td><td>Agric.</td><td>Built-up</td><td>Grass.</td><td>Road</td><td>Building</td></tr><tr><td>Scratch</td><td>None</td><td>67.39</td><td>54.01</td><td>52.15</td><td>31.07</td><td>70.26</td><td>60.89</td><td>12.21</td><td>14.92</td><td>54.72</td><td>40.68</td></tr><tr><td rowspan="4">Frozen</td><td>RGB-only</td><td>69.15</td><td>57.94</td><td>54.10</td><td>38.25</td><td>74.65</td><td>62.69</td><td>19.41</td><td>18.00</td><td>50.10</td><td>43.85</td></tr><tr><td>ExaPair</td><td>70.68</td><td>61.22</td><td>57.44</td><td>39.04</td><td>76.52</td><td>63.96</td><td>20.62</td><td>26.02</td><td>55.01</td><td>46.86</td></tr><tr><td>MixPair</td><td>71.69</td><td>61.80</td><td>58.76</td><td>39.86</td><td>77.72</td><td>65.05</td><td>20.99</td><td>25.54</td><td>56.40</td><td>47.60</td></tr><tr><td>SemPair</td><td>71.42</td><td>62.14</td><td>58.41</td><td>40.51</td><td>76.72</td><td>64.49</td><td>21.71</td><td>25.91</td><td>57.59</td><td>47.82</td></tr><tr><td rowspan="4">Fine-tuning</td><td>RGB-only</td><td>72.36</td><td>63.34</td><td>59.99</td><td>42.18</td><td>78.29</td><td>65.83</td><td>26.58</td><td>24.03</td><td>57.18</td><td>49.01</td></tr><tr><td>ExaPair</td><td>73.04</td><td>63.67</td><td>60.73</td><td>41.39</td><td>78.79</td><td>66.83</td><td>22.53</td><td>28.50</td><td>58.84</td><td>49.48</td></tr><tr><td>MixPair</td><td>73.42</td><td>63.88</td><td>61.27</td><td>41.68</td><td>79.66</td><td>67.39</td><td>23.50</td><td>26.96</td><td>59.60</td><td>49.80</td></tr><tr><td>SemPair</td><td>73.20</td><td>64.43</td><td>61.27</td><td>42.68</td><td>76.69</td><td>67.05</td><td>24.81</td><td>28.61</td><td>60.51</td><td>50.06</td></tr></table>

MMHunan: Table 6 reports transfer results with mIoU and mF1 averaged over all seven land-cover classes. Under frozen transfer, ExaPair, MixPair, and SemPair obtain 41.12%, 41.05%, and 40.94% mIoU, respectively, compared with 38.84% for RGB-only. Their spread is only 0.18%, indicating that SemPair preserves performance close to ExaPair. After fine-tuning, the corresponding mIoUs are 41.74%, 42.96%, and 42.41%. These differences remain modest and do not establish a universal advantage for any pairing strategy. At the selected fine-tuning epochs, ExaPair has the highest wetland IoU among the three multimodal strategies, MixPair has the highest grassland, water, and built-up IoUs, and SemPair has the highest cropland, forest, and unusedland IoUs.

GeoNRW: On GeoNRW (Table 7), SemPair reaches 47.82% mIoU under frozen transfer, compared with 46.86% for ExaPair, 47.60% for MixPair, and 43.85% for RGBonly. After fine-tuning, SemPair obtains 50.06%, MixPair 49.80%, and ExaPair 49.48%. The differences between the multimodal strategies are modest, with SemPair remaining competitive with ExaPair. Among the three multimodal strategies, SemPair has the highest fine-tuned IoUs for forest, grassland, road, and building, while MixPair has the highest agricultural and built-up IoUs.

Figure 6 provides qualitative comparisons on Potsdam and GeoNRW. The pretrained models generally recover more coherent semantic regions than Scratch, while ExaPair, MixPair, and SemPair exhibit broadly similar spatial predictions with local differences in boundaries and category assignments. These examples complement the quantitative evidence that relaxing exact instance pairing can preserve downstream segmentation quality.

![](images/56dad0477a6f1b20eccc28dd85051c671d5883aeff7e103fd8ff98f5a46bcff5.jpg)  
Fig. 6. Qualitative segmentation results on test set from Potsdam (2048 × 2048) and GeoNRW (512 × 512). Each row shows the optical image, DSM, ground truth, and predictions from the end-to-end trained Scratch baseline and four pretrained models under frozen transfer.

Overall, the four benchmarks show a consistent pattern. Across the eight frozen and fine-tuned evaluation settings, SemPair remains within 0.20% of ExaPair in the two cases where it is lower, while giving slightly higher mIoU in the remaining six settings. This includes frozen transfer, where encoder adaptation cannot compensate for differences introduced during pretraining. These observations support comparable transfer performance from class-consistent crossinstance pretraining. MixPair is also competitive, indicating that preserving more exact instance correspondence does not systematically improve downstream transfer.

These downstream results also provide an informative contrast to the source-domain classification results in Table 3, where ExaPair consistently achieves higher Acc@1 than SemPair. The advantage of exact correspondence for source fitting is therefore not preserved as a comparable advantage after transfer. This observation is consistent with the central premise of our analysis: exact instance correspondence can provide additional source-domain crossmodal dependence, but retaining that dependence is not a prerequisite for effective downstream transfer.

## 4.3 Understanding Cross-Modal Correspondence Granularity

The overall results establish that ExaPair can be removed without significantly degrading transfer. We next ask a more general question: does transfer improve monotonically as more exact correspondence is restored?

## 4.3.1 Continuous Correspondence Granularity

We vary the epoch-level instance-pairing ratio over $\rho ~ \in$ $\{ 0 , 0 . 2 5 , 0 . 5 , 0 . \hat { 7 } 5 , 1 \}$ . Here $\rho = 0$ corresponds to SemPair, $\rho = 1$ to ExaPair, and $\rho = 0 . 5$ to the default MixPair. At every epoch, $N _ { \mathrm { E } } = \lfloor \rho N \rfloor$ anchors are assigned to ExaPair after a random permutation of the training set, while the remaining anchors use SemPair. No class-wise stratification is used, and all other conditions remain fixed.

Table 8 and Figure 7 examine how the instance-pairing ratio $\rho$ affects downstream transfer. The Potsdam results exhibit limited sensitivity to correspondence granularity, with mIoU varying by only 0.29% and 0.11% under frozen transfer and fine-tuning, respectively. In contrast, Vaihingen shows more pronounced variations, particularly under frozen transfer, where the mIoU increases from 70.61% at $\rho = 0$ to 71.53% at $\rho = 0 . 5$ , before decreasing to 69.92% at $\rho ~ = ~ 0 . 7 5$ . A similar non-monotonic trend is observed after fine-tuning. This difference in sensitivity may be partly attributed to the larger downstream training set of Potsdam, which provides more supervision for adapting pretrained representations, whereas the more limited supervision in Vaihingen may make the inherited pretraining characteristics more influential.

Despite these differences, $\rho = 0 . 5$ consistently achieves the highest mIoU among the evaluated ratios across both datasets and adaptation settings. This observation suggests that an equal mixture of ExaPair and SemPair may provide a favorable balance between preserving instance-specific cross-modal information and encouraging shared semantic structure. Such a balance is consistent with the latent-factor analysis in Eq. (19), where $\rho$ controls the contribution of instance-specific cross-modal correlation. Nevertheless, the overall performance differences among the pairing strategies are generally modest, particularly on Potsdam. Rather than establishing a universally optimal ratio, these results indicate that increasing exact instance correspondence does not necessarily improve downstream transfer, and that moderate correspondence relaxation can maintain competitive performance.

![](images/96e056d4e5226946c91f373f2f17fe48dba397bf0f798b03a64b0fab6fac9df5.jpg)

TABLE 8  
Effect of continuous correspondence granularity on downstream mF1 and mIoU (%).
<table><tr><td rowspan="3">ρ</td><td rowspan="3">Strategy</td><td colspan="4">Potsdam</td><td colspan="4">Vaihingen</td></tr><tr><td colspan="2">Frozen</td><td colspan="2">Fine-tuning</td><td colspan="2">Frozen</td><td colspan="2">Fine-tuning</td></tr><tr><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td></tr><tr><td>0</td><td>SemPair</td><td>86.67</td><td>77.06</td><td>89.03</td><td>80.64</td><td>82.22</td><td>70.61</td><td>84.49</td><td>73.94</td></tr><tr><td>0.25</td><td>MixPair</td><td>86.68</td><td>77.10</td><td>88.98</td><td>80.59</td><td>82.32</td><td>70.76</td><td>84.77</td><td>74.40</td></tr><tr><td>0.50</td><td>MixPair</td><td>86.77</td><td>77.22</td><td>89.05</td><td>80.68</td><td>82.88</td><td>71.53</td><td>84.95</td><td>74.52</td></tr><tr><td>0.75</td><td>MixPair</td><td>86.58</td><td>76.94</td><td>88.98</td><td>80.58</td><td>81.75</td><td>69.92</td><td>84.09</td><td>73.40</td></tr><tr><td>1</td><td>ExaPair</td><td>86.58</td><td>76.93</td><td>88.97</td><td>80.57</td><td>81.93</td><td>70.16</td><td>84.52</td><td>74.13</td></tr></table>

Fig. 7. Effect of the instance-pairing ratio ρ on downstream mIoU for Potsdam and Vaihingen under frozen transfer and end-to-end finetuning.

## 4.3.2 Representation-Level Cross-Modal Retrieval

This analysis examines whether SemPair preserves crossmodal information despite relaxing exact instance correspondence during pretraining. Exact-instance retrieval and instance-excluded semantic retrieval are evaluated between the RGB and depth branches of the pretrained encoder. Features are extracted from paired RGB-depth inputs with cross-modal interaction active, so the evaluation measures cross-branch correspondence under multimodal condition ing rather than retrieval between independently encoded modalities.

As presented in Table 9, all three strategies achieve high exact-instance R@1 with the complete pretrained DFormer. SemPair reaches 97.59% on ImageNet-RS90, only 0.30% below ExaPair, and 96.13% on NWPU-RGBD-45, compared with 97.40% for ExaPair. Instance-excluded retrieval provides a complementary assessment of semantic organization by removing the exact counterpart from the candidate gallery. SemPair achieves semantic R@1 and R@5 of 72.80% and 88.95% on ImageNet-RS90, trailing ExaPair by only 1.40% and 1.46%, respectively. Comparable gaps of 1.40% and 1.27% are observed on NWPU-RGBD-45, where MixPair also achieves the highest semantic R@1. These results indicate that SemPair retains much of the cross-modal structure measured by retrieval, even without exact instance pairs during pretraining. Together with the downstream segmentation results, they provide complementary evidence for the transferability of representations learned through semantic-level correspondence.

TABLE 9  
Representation-level RGB-to-depth retrieval on the fixed pretraining validation split. Inst. R@1 uses the full depth gallery. Excl-inst. Sem. R@1 and R@5 exclude the same-instance depth sample from the gallery. All metrics are reported in percentages.
<table><tr><td>Pretraining domain</td><td>Strategy</td><td>Inst. R@1</td><td>Excl-inst. Sem. R@1</td><td>Excl-inst. Sem. R@5</td></tr><tr><td rowspan="3">ImageNet-RS90</td><td>ExaPair</td><td>97.89</td><td>74.20</td><td>90.41</td></tr><tr><td>MixPair</td><td>97.09</td><td>72.98</td><td>89.77</td></tr><tr><td>SemPair</td><td>97.59</td><td>72.80</td><td>88.95</td></tr><tr><td rowspan="3">NWPU-RGBD- 45</td><td>ExaPair</td><td>97.40</td><td>86.86</td><td>96.06</td></tr><tr><td>MixPair</td><td>96.44</td><td>87.75</td><td>95.81</td></tr><tr><td>SemPair</td><td>96.13</td><td>85.46</td><td>94.79</td></tr></table>

Figure 8 presents representative instance-excluded retrieval examples from both pretraining domains. All three strategies retrieve semantically consistent samples for categories such as airliner and sea ice, whereas more challenging categories, including dock and intersection, exhibit differences in retrieval rankings and category confusions. These results further demonstrate that SemPair can learn meaningful cross-modal semantic associations without exact instance correspondence during pretraining.

## 4.4 Generalization Across Pretraining Domains

The preceding representation analysis uses both pretraining sources, whereas the primary transfer experiments use ImageNet-RS90. We therefore repeat downstream transfer after pretraining on NWPU-RGBD-45 to examine whether the observations depend on a particular source domain.

As shown in Table 10, changing the pretraining domain affects the relative performance of the correspondence strategies, but does not alter the overall findings. With NWPU-RGBD-45 pretraining, SemPair achieves 77.87% mIoU on Potsdam, slightly outperforming Exa-Pair at 77.61%. In contrast, ExaPair performs better on Vaihingen, reaching 73.12% compared with 72.11% for SemPair. Similar variations are observed on the other two benchmarks. For MMHunan, ExaPair, MixPair, and SemPair achieve 41.05%, 40.91%, and 40.20% mIoU, respectively, all exceeding the RGB-only baseline of 38.15%. GeoNRW exhibits a different ranking, with MixPair achieving 47.96%, followed by SemPair at 47.38% and ExaPair at 47.08%.

Across the four downstream benchmarks, all three multimodal strategies outperform RGB-only pretraining with the alternative source dataset, while SemPair remains competitive with ExaPair. The varying rankings suggest that the effect of correspondence granularity depends on the sourcetarget setting.

TABLE 10  
Generalization across pretraining domains measured by downstream mF1 and mIoU (%).
<table><tr><td rowspan="2">Pretraining domain</td><td rowspan="2">Strategy</td><td colspan="2">Potsdam</td><td colspan="2">Vaihingen</td><td colspan="2">MMHunan</td><td colspan="2">GeoNRW</td></tr><tr><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td></tr><tr><td rowspan="4">ImageNet-RS90</td><td>RGB-only</td><td>88.84</td><td>80.37</td><td>83.48</td><td>72.49</td><td>53.70</td><td>41.39</td><td>63.34</td><td>49.01</td></tr><tr><td>ExaPair</td><td>88.97</td><td>80.57</td><td>84.52</td><td>74.13</td><td>54.28</td><td>41.74</td><td>63.67</td><td>49.48</td></tr><tr><td>MixPair</td><td>89.05</td><td>80.68</td><td>84.95</td><td>74.52</td><td>55.46</td><td>42.96</td><td>63.88</td><td>49.80</td></tr><tr><td>SemPair</td><td>89.03</td><td>80.64</td><td>84.49</td><td>73.94</td><td>54.69</td><td>42.41</td><td>64.43</td><td>50.06</td></tr><tr><td rowspan="4">NWPU-RGBD-45</td><td>RGB-only</td><td>86.20</td><td>76.36</td><td>79.09</td><td>67.52</td><td>49.51</td><td>38.15</td><td>59.08</td><td>45.07</td></tr><tr><td>ExaPair</td><td>87.04</td><td>77.61</td><td>83.66</td><td>73.12</td><td>52.82</td><td>41.05</td><td>61.02</td><td>47.08</td></tr><tr><td>MixPair</td><td>87.16</td><td>77.80</td><td>82.88</td><td>71.99</td><td>53.83</td><td>40.91</td><td>61.95</td><td>47.96</td></tr><tr><td>SemPair</td><td>87.21</td><td>77.87</td><td>83.04</td><td>72.11</td><td>52.11</td><td>40.20</td><td>61.29</td><td>47.38</td></tr></table>

![](images/e47d4fe3f9916efbfddd10688d8de7115e1ca0f330bd48f548b9a3874d10114c.jpg)  
Fig. 8. Qualitative cross-branch RGB-to-depth retrieval on the fixed validation splits of (a) ImageNet-RS90 and (b) NWPU-RGBD-45. Each example shows the RGB query and its same-instance depth reference once, followed by the top-three retrieved depth samples for ExaPair, MixPair, and SemPair. The reference depth is excluded from the retrieval gallery.

## 4.5 Transfer under Different Downstream Conditions

We next examine conditions that control how strongly the pretrained representation influences the final model. Two factors are varied: the amount of labeled downstream data and the degree to which the encoder is allowed to adapt.

## 4.5.1 Label-Efficient Transfer

The results of label-efficient transfer are summarized in Table 11. Pretraining provides substantial benefits under limited downstream supervision. With only one labeled training image from Potsdam, Scratch achieves 56.85% mIoU, whereas RGB-only pretraining improves the result to 71.33%. Multimodal pretraining brings further gains, with ExaPair, MixPair, and SemPair achieving 72.15%, 72.53%, and 71.64%, respectively. A similar pattern emerges on Vaihingen with three labeled tiles, where SemPair achieves 68.43% mIoU, substantially outperforming Scratch at 54.85% and slightly exceeding ExaPair at 67.67%. These results indicate that SemPair preserves the benefits of multimodal pretraining even when downstream annotations are severely limited.

![](images/fb0672f5c1c33f02866b567355c7922bb23b1a3696d8a0b4a7eb0534f114ae71.jpg)  
Fig. 9. Label-efficient transfer on Potsdam and Vaihingen.

The ranking varies with the label budget, arguing against a universal preference for exact pairs. As illustrated in Figure 9, the performance trends remain broadly comparable across the three multimodal strategies, although their relative rankings change with the amount of supervision. MixPair performs best on Potsdam with one and six training tiles and with the full split, while ExaPair is slightly better with three tiles. For Vaihingen, SemPair performs best in the low-label settings, exceeding ExaPair by 0.76% with three tiles and 1.09% with six. With full supervision, MixPair achieves the highest mIoU. Overall, SemPair remains competitive across different label budgets despite the absence of exact RGB-depth pairs during pretraining, indicating that effective transfer under pairing relaxation does not depend on abundant downstream annotations.

## 4.5.2 Effect of Downstream Adaptation Strength

Table 12 examines the effect of downstream adaptation strength, comparing frozen-encoder transfer with finetuning at encoder learning rates of $1 0 ^ { - 3 }$ and $1 0 ^ { - 2 }$ . For Potsdam, SemPair slightly outperforms ExaPair under frozen transfer and conservative fine-tuning, by 0.13% and 0.07%, respectively, while trailing by only 0.12% under stronger adaptation. A similar pattern emerges on Vaihingen, where SemPair leads by 0.45% with a frozen encoder, whereas ExaPair holds small advantages of 0.19% and 0.14% under the two fine-tuning settings. Overall, SemPair remains competitive across different adaptation strengths, with no consistent advantage for exact instance correspondence. The smaller performance gaps after fine-tuning suggest that downstream adaptation can partially mitigate the influence of correspondence granularity during pretraining.

TABLE 11  
Label-efficient transfer measured by mF1 and mIoU (%). P1, P3, and P6 denote 1, 3, and 6 labeled Potsdam training tiles. V3 and V6 denote 3 and 6 labeled Vaihingen training tiles. Full denotes the complete training split.
<table><tr><td rowspan="3">Initialization</td><td colspan="8">Potsdam</td><td colspan="6">Vaihingen</td></tr><tr><td colspan="2">P1</td><td colspan="2">P3</td><td colspan="2">P6</td><td colspan="2">Full</td><td colspan="2">V3</td><td colspan="2">V6</td><td colspan="2">Full</td></tr><tr><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td></tr><tr><td>Scratch</td><td>70.91</td><td>56.85</td><td>72.83</td><td>58.82</td><td>75.60</td><td>62.03</td><td>76.83</td><td>63.74</td><td>67.84</td><td>54.85</td><td>71.23</td><td>58.48</td><td>72.19</td><td>59.83</td></tr><tr><td>RGB-only</td><td>82.89</td><td>71.33</td><td>85.07</td><td>74.46</td><td>86.52</td><td>76.69</td><td>88.84</td><td>80.37</td><td>79.44</td><td>67.04</td><td>82.67</td><td>71.51</td><td>83.48</td><td>72.49</td></tr><tr><td>ExaPair</td><td>83.42</td><td>72.15</td><td>85.54</td><td>75.24</td><td>86.86</td><td>77.24</td><td>88.97</td><td>80.57</td><td>79.89</td><td>67.67</td><td>82.83</td><td>71.53</td><td>84.52</td><td>74.13</td></tr><tr><td>MixPair</td><td>83.66</td><td>72.53</td><td>85.32</td><td>74.90</td><td>87.19</td><td>77.75</td><td>89.05</td><td>80.68</td><td>80.29</td><td>68.12</td><td>82.96</td><td>71.80</td><td>84.95</td><td>74.52</td></tr><tr><td>SemPair</td><td>83.09</td><td>71.64</td><td>85.20</td><td>74.71</td><td>87.12</td><td>77.61</td><td>89.03</td><td>80.64</td><td>80.49</td><td>68.43</td><td>83.56</td><td>72.62</td><td>84.49</td><td>73.94</td></tr></table>

TABLE 12

Effect of downstream adaptation strength on mF1 and mIoU (%). The same Scratch baseline is used across all three adaptation settings, with the entire network trained from random initialization at an initial learning rate of 10<sup>−2</sup>.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Init.</td><td colspan="2">Frozen</td><td colspan="2">Fine-tuning</td><td colspan="2">FT(10−2)</td></tr><tr><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td></tr><tr><td rowspan="3">Potsdam</td><td>Scratch</td><td>76.83</td><td>63.74</td><td>76.83</td><td>63.74</td><td>76.83</td><td>63.74</td></tr><tr><td>ExaPair</td><td>86.58</td><td>76.93</td><td>88.97</td><td>80.57</td><td>89.65</td><td>81.61</td></tr><tr><td>SemPair</td><td>86.67</td><td>77.06</td><td>89.03</td><td>80.64</td><td>89.56</td><td>81.49</td></tr><tr><td rowspan="3">Vaihingen</td><td>Scratch</td><td>72.19</td><td>59.83</td><td>72.19</td><td>59.83</td><td>72.19</td><td>59.83</td></tr><tr><td>ExaPair</td><td>81.93</td><td>70.16</td><td>84.52</td><td>74.13</td><td>86.01</td><td>76.29</td></tr><tr><td>SemPair</td><td>82.22</td><td>70.61</td><td>84.49</td><td>73.94</td><td>85.98</td><td>76.15</td></tr></table>

## 4.6 Generalization Across Model Scales

Finally, we evaluate DFormer-Tiny, DFormer-Small, and DFormer-Base under the same standard fine-tuning protocol. As presented in Table 13, no single pairing strategy consistently dominates across model scales. For Potsdam, both MixPair and SemPair outperform ExaPair with DFormer-Tiny, while all three strategies achieve comparable performance with DFormer-Small. SemPair achieves the highest mIoU of 81.08% with DFormer-Base. A different pattern emerges on Vaihingen, where ExaPair performs best with DFormer-Tiny, whereas MixPair leads with DFormer-Small and DFormer-Base. With DFormer-Base, MixPair and SemPair achieve 75.34% and 75.26% mIoU, respectively, compared with 74.26% for ExaPair. These results extend the evidence for correspondence relaxation beyond the default DFormer-Small encoder, covering model capacities from 5.64M to 29.95M parameters.

## 4.7 Limitations and Discussion

Our results show that exact instance correspondence is not a prerequisite for effective multimodal pretraining in the remote-sensing settings considered in this work. Nevertheless, the present study is based on controlled re-pairing: the original RGB-pseudo-depth correspondence is known, while SemPair deliberately replaces exact pairs with classconsistent cross-instance pairs during pretraining. This design allows correspondence granularity to be isolated from other factors, but it does not yet establish the same conclusion for independently collected multimodal archives in real scenes.

SemPair also assumes that semantic consistency provides a meaningful cross-modal relation. This assumption is well aligned with semantic recognition and segmentation, but exact instance correspondence may remain important for tasks that intrinsically depend on local geometric or temporal consistency, such as registration, reconstruction, or fine-grained change localization. SemPair should therefore be viewed as a relaxed alternative to ExaPair rather than a universal replacement for paired supervision.

Although all downstream experiments in this work focus on remote sensing, the formulation itself is not tied to a specific sensor combination or backbone. Its essential requirement is a reliable semantic relation across modalities rather than exact instance alignment. This makes the same principle potentially applicable to other multimodal settings where large-scale paired data are difficult to obtain. Such generalization, however, remains to be empirically validated beyond the remote-sensing scenarios studied here.

## 5 CONCLUSION

This work challenges the assumption that effective multimodal pretraining requires exact instance correspondence. By introducing SemPair and MixPair, we demonstrate that relaxing cross-modal pairing can preserve transferable representations without modifying the network architecture or learning objective. The latent-factor analysis characterizes how pairing granularity changes cross-modal correlations, while extensive experiments across pretraining domains, downstream benchmarks, and adaptation settings show that SemPair remains competitive with ExaPair. These findings suggest that exact instance correspondence is not a prerequisite for effective multimodal remote sensing pretraining, and that semantic-level correspondence provides a viable alternative. Future work will extend this controlled study to independently collected multimodal archives with incomplete cross-modal correspondence.

## REFERENCES

[1] X. X. Zhu, D. Tuia, L. Mou, G.-S. Xia, L. Zhang, F. Xu, and F. Fraundorfer, “Deep learning in remote sensing: A comprehensive review and list of resources,” IEEE Geoscience and Remote Sensing Magazine, vol. 5, no. 4, pp. 8–36, 2017.

[2] L. Ma, Y. Liu, X. Zhang, Y. Ye, G. Yin, and B. A. Johnson, “Deep learning in remote sensing applications: A meta-analysis and review,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 152, pp. 166–177, 2019.

[3] S. Lu, J. Guo, J. R. Zimmer-Dauphinee, J. M. Nieusma, X. Wang, P. VanValkenburgh, S. A. Wernke, and Y. Huo, “Vision foundation models in remote sensing: A survey,” IEEE Geoscience and Remote Sensing Magazine, vol. 13, no. 3, pp. 190–215, 2025.

[4] X. Guo, J. Lao, B. Dang, Y. Zhang, L. Yu, L. Ru, L. Zhong, Z. Huang, K. Wu, D. Hu, H. He, J. Wang, J. Chen, M. Yang, Y. Zhang, and Y. Li, “SkySense: A multi-modal remote sensing foundation model towards universal interpretation for earth observation imagery,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 27 672–27 683.

TABLE 13  
Generalization across model scales measured by mF1 and mIoU (%). The best result for each metric within each dataset and model scale is highlighted in bold.
<table><tr><td rowspan="3">Architecture</td><td rowspan="3">Params.</td><td colspan="6">Potsdam</td><td colspan="6">Vaihingen</td></tr><tr><td colspan="2">ExaPair</td><td colspan="2">MixPair</td><td colspan="2">SemPair</td><td colspan="2">ExaPair</td><td colspan="2">MixPair</td><td colspan="2">SemPair</td></tr><tr><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td></td><td>mF1 mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td><td>mF1</td><td>mIoU</td></tr><tr><td>DFormer-Tiny</td><td>5.64M</td><td>88.91</td><td>80.47</td><td>89.09</td><td>80.77</td><td>89.06</td><td>80.71</td><td>83.99</td><td>73.44</td><td>83.60</td><td>72.66</td><td>83.97</td><td>73.20</td></tr><tr><td>DFormer-Small</td><td>19.05M</td><td>88.97</td><td>80.57</td><td>89.05</td><td>80.68</td><td>89.03</td><td>80.64</td><td>84.52</td><td>74.13</td><td>84.95</td><td>74.52</td><td>84.49</td><td>73.94</td></tr><tr><td>DFormer-Base</td><td>29.95M</td><td>89.25</td><td>81.02</td><td>89.16</td><td>80.86</td><td>89.30</td><td>81.08</td><td>84.72</td><td>74.26</td><td>85.42</td><td>75.34</td><td>85.37</td><td>75.26</td></tr></table>

[5] K. Wu, Y. Zhang, L. Ru, B. Dang, J. Lao, L. Yu, J. Luo, Z. Zhu, Y. Sun, J. Zhang, Q. Zhu, J. Wang, M. Yang, J. Chen, Y. Zhang, and Y. Li, “A semantic-enhanced multi-modal remote sensing foundation model for earth observation,” Nature Machine Intelligence, vol. 7, pp. 1235–1249, 2025.

[6] Y. Bengio, A. Courville, and P. Vincent, “Representation learning: A review and new perspectives,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, no. 8, pp. 1798–1828, 2013.

[7] L. Jing and Y. Tian, “Self-supervised visual feature learning with deep neural networks: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 11, pp. 4037–4058, 2021.

[8] Y. Wang, C. M. Albrecht, N. A. A. Braham, C. Liu, Z. Xiong, and X. X. Zhu, “Decoupling common and unique representations for multimodal self-supervised learning,” in European Conference on Computer Vision, 2024, pp. 286–303.

[9] G. Astruc, N. Gonthier, C. Mallet, and L. Landrieu, “AnySat: One earth observation model for many resolutions, scales, and modalities,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 19 530–19 540.

[10] J. Jakubik, F. Yang, B. Blumenstiel, E. Scheurer, R. Sedona, S. Maurogiovanni, J. Bosmans, N. Dionelis, V. Marsocci, N. Kopp, R. Ramachandran, P. Fraccaro, T. Brunschwiler, G. Cavallaro, J. Bernabe-Moreno, and N. Longep´ e, “TerraMind: Large-scale´ generative multimodality for earth observation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 7383–7394.

[11] V. Nedungadi, A. Kariryaa, S. Oehmcke, S. Belongie, C. Igel, and N. Lang, “MMEarth: Exploring multi-modal pretext tasks for geospatial representation learning,” in European Conference on Computer Vision, 2024, pp. 164–182.

[12] D. Ramachandram and G. W. Taylor, “Deep multimodal learning: A survey on recent advances and trends,” IEEE Signal Processing Magazine, vol. 34, no. 6, pp. 96–108, 2017.

[13] T. Baltrusaitis, C. Ahuja, and L.-P. Morency, “Multimodal machineˇ learning: A survey and taxonomy,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 2, pp. 423–443, 2019.

[14] N. Audebert, B. Le Saux, and S. Lefevre, “Beyond RGB: Very\` high resolution urban remote sensing with multimodal deep networks,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 140, pp. 20–32, 2018.

[15] D. Hong, N. Yokoya, G.-S. Xia, J. Chanussot, and X. X. Zhu, “X-ModalNet: A semi-supervised deep cross-modal network for classification of remote sensing data,” ISPRS Journal ofPhotogrammetry and Remote Sensing, vol. 167, pp. 12–23, 2020.

[16] Y. Zhang, L. Ru, K. Wu, L. Yu, L. Liang, Y. Li, and J. Chen, “Sky-Sense V2: A unified foundation model for multi-modal remote sensing,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 9136–9146.

[17] B. Khaleghi, A. Khamis, F. O. Karray, and S. N. Razavi, “Multisensor data fusion: A review of the state-of-the-art,” Information Fusion, vol. 14, no. 1, pp. 28–44, 2013.

[18] Y. Zhang, D. Sidibe, O. Morel, and F. M ´ eriaudeau, “Deep multi-´ modal fusion for semantic image segmentation: A survey,” Image and Vision Computing, vol. 105, p. 104042, 2021.

[19] J. Li, D. Hong, L. Gao, J. Yao, K. Zheng, B. Zhang, and J. Chanussot, “Deep learning in multimodal remote sensing data fusion: A comprehensive review,” International Journal of Applied Earth Observation and Geoinformation, vol. 112, p. 102926, 2022.

[20] D. Hong, L. Gao, N. Yokoya, J. Yao, J. Chanussot, Q. Du, and B. Zhang, “More diverse means better: Multimodal deep learning meets remote-sensing imagery classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 5, pp. 4340–4354, 2021.

[21] D. Hong, J. Hu, J. Yao, J. Chanussot, and X. X. Zhu, “Multimodal remote sensing benchmark datasets for land cover classification with a shared and specific feature learning model,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 178, pp. 68–80, 2021.

[22] J. Hu, D. Hong, and X. X. Zhu, “MIMA: Mapper-induced manifold alignment for semi-supervised fusion of optical image and

polarimetric SAR data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 11, pp. 9025–9040, 2019.

[23] H. Hosseinpour, F. Samadzadegan, and F. Dadrass Javan, “CMGFNet: A deep cross-modal gated fusion network for building extraction from very high-resolution remote sensing images,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 184, pp. 96–115, 2022.

[24] X. Ma, X. Zhang, M.-O. Pun, and M. Liu, “A multilevel multimodal fusion transformer for remote sensing semantic segmentation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1– 15, 2024.

[25] Z. Wang, J. Li, N. Xu, and Z. You, “Combining feature compensation and GCN-based reconstruction for multimodal remote sensing image semantic segmentation,” Information Fusion, vol. 122, p. 103207, 2025.

[26] X. Ma, X. Zhang, M.-O. Pun, and B. Huang, “A unified framework with multimodal fine-tuning for remote sensing semantic segmentation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–15, 2025.

[27] Z. Qi, Q. Guo, L. Wang, and Y. Zhou, “MultiMoE: Multimodal mixture of experts adapter for semantic segmentation of remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, 2026.

[28] D. Muhtar, X. Zhang, P. Xiao, Z. Li, and F. Gu, “CMID: A unified self-supervised learning framework for remote sensing image understanding,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, p. 5607817, 2023.

[29] X. Sun, P. Wang, W. Lu, Z. Zhu, X. Lu, Q. He, J. Li, X. Rong, Z. Yang, H. Chang, Q. He, G. Yang, R.-P. Wang, J. Lu, and K. Fu, “RingMo: A remote sensing foundation model with masked image modeling,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, p. 5612822, 2023.

[30] D. Hong, B. Zhang, X. Li, Y. Li, C. Li, J. Yao, N. Yokoya, H. Li, P. Ghamisi, X. Jia, A. Plaza, P. Gamba, J. A. Benediktsson, and J. Chanussot, “SpectralGPT: Spectral remote sensing foundation model,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5227–5244, 2024.

[31] Y. Wang, N. Ait Ali Braham, Z. Xiong, C. Liu, C. M. Albrecht, and X. X. Zhu, “SSL4EO-S12: A large-scale multimodal, multitemporal dataset for self-supervised learning in earth observation,” IEEE Geoscience and Remote Sensing Magazine, vol. 11, no. 3, pp. 98–106, 2023.

[32] Z. Xue, G. Yang, X. Yu, A. Yu, Y. Guo, B. Liu, and J. Zhou, “Multimodal self-supervised learning for remote sensing data land cover classification,” Pattern Recognition, vol. 157, p. 110959, 2025.

[33] Y. Zong, O. Mac Aodha, and T. M. Hospedales, “Self-supervised multimodal learning: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 7, pp. 5299–5318, 2025.

[34] L. Wan, Q. Jing, Z. Sun, C. Zhang, Z. Li, and Y. Chen, “Selfsupervised modality-aware multiple granularity pre-training for RGB-infrared person re-identification,” IEEE Transactions on Information Forensics and Security, vol. 18, pp. 3044–3057, 2023.

[35] H. Yao, B. Yang, W. Huang, B. Du, and M. Ye, “Unsupervised visible-infrared person re-identification under unpaired settings,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 11 916–11 926.

[36] Y. Deng, T. Xu, C. Cheng, H. Li, X. Wu, and J. Kittler, “Beyond strict pairing: Arbitrarily paired training for high-performance infrared and visible image fusion,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 12 334–12 343.

[37] Z. Geng, H. Liu, P. Duan, X. Wei, and S. Li, “Feature-based multimodal remote sensing image matching: Benchmark and state-ofthe-art,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 229, pp. 285–302, 2025.

[38] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009, pp. 248–255.

## PREPRINT

[39] G. Cheng, J. Han, and X. Lu, “Remote sensing image scene classification: Benchmark and state of the art,” Proceedings of the IEEE, vol. 105, no. 10, pp. 1865–1883, 2017.

[40] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” Advances in neural information processing systems, vol. 37, pp. 21 875–21 911, 2024.

[41] Y. Li, Y. Zhou, Y. Zhang, L. Zhong, J. Wang, and J. Chen, “DKDFN: Domain knowledge-guided deep collaborative fusion network for multimodal unitemporal remote sensing land cover classification,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 186, pp. 170–189, 2022.

[42] G. Baier, A. Deschemps, M. Schmitt, and N. Yokoya, “Geonrw,” 2020. [Online]. Available: https://dx.doi.org/10.21227/s5xq-b822

[43] B.-W. Yin, J.-L. Cao, D. Xu, M.-M. Cheng, and Q. Hou, “Dformer++: Improving rgbd representation learning for semantic segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.