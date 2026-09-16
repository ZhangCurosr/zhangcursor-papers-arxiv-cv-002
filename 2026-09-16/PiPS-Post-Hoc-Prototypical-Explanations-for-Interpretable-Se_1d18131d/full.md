# PiPS: Post-Hoc Prototypical Explanations for Interpretable Semantic Segmentation

Miłosz Adamczyk<sup>1</sup>, Tymoteusz Zapala <sup>2</sup>, Piotr Borycki<sup>1,</sup> <sup>3</sup>, Przemysław Spurek<sup>1,</sup> <sup>3</sup>,

<sup>1</sup> Jagiellonian University

<sup>2</sup> Wrocław University of Science and Technology <sup>3</sup> IDEAS Research Institute

## Abstract

With the increasing deployment of deep neural networks in critical systems, such as medical diagnostics and autonomous vehicles, ensuring their interpretability is crucial to building trust in decision-making systems. In the field of explainable artificial intelligence, prototype-based reasoning has gained particular popularity, as it mimics human cognitive processes by explaining model decisions based on visual similarity under the looks like this paradigm. While this paradigm has been thoroughly investigated in the context of global image classification, the interpretability of dense predictions, particularly semantic segmentation, remains largely unexplored despite its immense importance in tasks requiring precise object localization. Existing prototype-based interpretable segmentation models rely on ante-hoc architectures, which entails significant limitations because they require costly training from scratch and modifications to the network structure, ultimately leading to a noticeable drop in predictive performance compared to standard black-box models. To address this issue, we propose PiPS (Post-hoc interpretable Prototypical Segmentation), the first fully post-hoc solution for generating prototypical explanations for semantic segmentation models. Our method enables the extraction of intuitive, spatially localized explanations from any pre-trained network without modification or fine-tuning, thereby preserving 100% of the model’s original predictive performance. This approach opens a new avenue for the safe and cost-efective deployment of transparent systems in advanced computer vision tasks. Codebase available at https://github.com/gmum/PIPS

## Introduction

As deep neural networks continue to achieve unprecedented performance across a multitude of computer vision tasks, their inherent black-box nature poses a significant barrier to their deployment in real-world scenarios. Ensuring model interpretability has thus become a fundamental requirement for building trust, accountability, and safety in artificial intelligence systems. Within the domain of Explainable Artificial Intelligence (XAI), prototype-based reasoning has emerged as a leading and highly intuitive concept. By adopting the looks like this paradigm (Chen et al. 2019), prototypical methods mimic human cognitive processes, explaining a model’s decision by pointing out visual similarities between parts of an input image and a set of learned, representative prototypes.

While prototypical explainability has been extensively studied and successfully implemented in the context ofglobal image classification (Chen et al. 2019), its application to dense prediction tasks remains largely unexplored. This constitutes a critical gap in the literature, given that semantic segmentation provides the fine-grained, pixel-level scene understanding indispensable for numerous high-stakes applications. In domains such as autonomous driving, medical image analysis, and robotic navigation, merely knowing that a specific class is present in an image is insuficient. In these fields, precise spatial localization and local interpretability are paramount, making the ability to explain exactly why a specific group of pixels was assigned to a given class a crucial safety requirement.

Despite the pressing need for interpretable dense predictions, existing attempts to bring prototypical reasoning to semantic segmentation rely almost exclusively on ante-hoc architectures (Sacha et al. 2023; Porta et al. 2025a). These approaches embed custom prototype layers directly into the network design, which imposes severe limitations. Primarily, they require computationally expensive training from scratch, making it exceptionally dificult to adapt them to modern, massive foundation models. Furthermore, they introduce a problematic trade-of: forcing the network to learn a constrained, prototype-friendly latent space typically leads to a noticeable degradation in predictive performance, resulting in a lower mean Intersection over Union (mIoU) compared to state-of-the-art black-box baselines. On the other hand, while post-hoc methods have been adapted for segmentation, they are largely limited to attribution maps or adversarial frameworks (Gipiškis 2025; Selvaraju et al. 2017), which inherently lack the human-understandable visual reasoning that prototypes provide.

To overcome these fundamental limitations, we propose PiPS (Post-hoc interpretable Prototypical Segmentation), the first fully post-hoc solution designed specifically for prototypical semantic segmentation. Instead of modifying and retraining the network, PiPS operates on top of existing, pre-trained models. Our method works by extracting dense, pixel-wise feature maps from the deep layers of a frozen base model and mathematically projecting them onto a set of representative spatial prototypes. By decoupling the explanation mechanism from the primary prediction pipeline, PiPS successfully translates the complex latent representations of dense tasks into intuitive, region-based visual similarities without interfering with the model’s original weights (see Figure 1 for a visual comparison of our explanations against existing baseline methods and Figure 2 for comparsion using heatmaps).

![](images/57a4fdf5eee112253567bd40b408605a1b6c72710e78c0e699332a26f5303e22.jpg)  
Figure 1: Comparison of explanation methods on a multi-object scene (bus and person) from PASCAL VOC 2012. PiPS (Ours) delivers a disentangled, part-by-part decomposition of the object, explicitly visualizing the local features that influenced the model’s decision by comparing them against training data. Every two rows correspond to a specific prototypical part (showing the top-2 channels for each predicted class). Green bounding boxes highlight the activation of a given part within the reference training images, while red boxes in the leftmost column pinpoint the corresponding parts in the original target test image (the second column displays segmentation masks).ScaledProtoSeg yields similarly looking results and outputs a fixed numbe of prototypical parts arbitrarily chosen regardless of the actual number of classes present. Similarly, SegGradCam produce general-purpose saliency maps rather than explicit part activations, but provides exactly one attribution map per detected class.

The main contributions of our work can be summarized as follows:

• We introduce PiPS, the first fully post hoc framework for prototype-based explainability in semantic segmentation, which eliminates the need for architectural modifications and costly retraining from scratch.

• We achieve complete preservation of the base model’s predictive performance, efectively breaking the accuracy-versus-interpretability trade-of that severely limits existing ante hoc prototypical segmentation methods.

• We demonstrate the broad flexibility and universal applicability of our approach by successfully integrating it with various state-of-the-art pre-trained segmentation networks, providing highly localized, human-understandable visual explanations across complex datasets.

## Related Works

With the rapid development and increasingly widespread deployment of deep learning models in key areas such as healthcare and autonomous systems, the issue of explainability has become a fundamental research challenge. In the scholarly literature on explainable artificial intelligence (XAI), two principal paradigms can be distinguished: post hoc explanation methods and inherently interpretable (ante hoc) models.

Post hoc methods focus on analyzing already-trained models, providing explanations without altering their architecture. Classic examples include SHAP (Lundberg and Lee 2017) and LIME (Ribeiro, Singh, and Guestrin 2016), which assign importance to individual features, as well as Grad-CAM (Selvaraju et al. 2017), which generates attention maps highlighting critical input regions. Recently, the post-hoc approach has also been extensively explored in the context of semantic image segmentation, ofering various attribution maps and even providing frameworks for adversarial attacks (Gipiškis 2025). However, despite their flexibility, post-hoc attribution methods are often criticized for the instability of the generated explanations (Adebayo et al. 2018) and their inability to provide intuitive, human-like visual

![](images/7d55721049200bd74e80f2a7abf7bf166d3178a4c32fa4b57cfb2edd0dcd3d2f.jpg)  
Figure 2: Comparison of PiPS and ScaledProtoSeg with heatmaps. Visual evaluation using heatmaps illustrating the spatial activation patterns and localization performance of prototypical parts generated by PiPS and ScaledProtoSeg on semantic segmentation tasks. Notice the variations in activation in compared approaches.

reasoning.

To address the limitations of post-hoc attribution maps, ante-hoc models integrate interpretability mechanisms directly into the architecture. A prominent development in this area is the ProtoPNet algorithm (Chen et al. 2019), which introduces class prototypes, enabling interpretation of model decisions through the looks like this paradigm. While originally designed for global image classification, this prototypical reasoning has recently been adapted to dense prediction tasks, where spatial localization is paramount. For instance, ProtoSeg (Sacha et al. 2023) introduced interpretable semantic segmentation using prototypical parts, enabling models to explain pixel-wise decisions based on local visual similarities. This concept was further advanced by introducing multi-scale grouped prototypes (Porta et al. 2025a), which allow the network to capture and explain objects at various spatial granularities.

Despite providing highly intuitive and localized explanations, prototypical segmentation models face a critical bottleneck, they are strictly ante-hoc. Methods such as ProtoSeg (Sacha et al. 2023) and multi-scale prototype networks (Porta et al. 2025a) require specialized architectures and computationally expensive training from scratch. This makes them dificult to apply to modern, large-scale foundation models. Furthermore, forcing the network to learn a constrained, prototype-based latent space typically leads to a noticeable degradation in predictive performance (e.g., lower mean Intersection over Union) compared to state-of-the-art black-box models.

While hybrid methods that combine post-hoc flexibility with concept-based explanations, such as ACE (Ghorbani et al. 2019) or Concept Whitening (Chen et al. 2020), have been proposed for classification, they fail to provide the exact, visually verifiable prototypical patches that are highly desired in dense tasks. Recently, EPIC (Borycki et al. 2026) introduced a post-hoc prototype framework for global image classification. However, EPIC is inherently designed for whole-image classification and lacks the spatially bounded attribution mechanism required to explain dense, pixel-level prediction tasks. Thus, a clear gap exists: the literature lacks a method that provides the deep interpretability of prototypical parts combined with the flexibility of post-hoc application specifically tailored for semantic segmentation.

Our proposed method, PiPS, addresses this exact gap by enabling prototype-based explanations on top of already trained segmentation networks. It successfully combines the scalability ofered by post hoc techniques (Gipiškis 2025) with the localized interpretability characteristic of ante hoc prototypical segmentation approaches (Sacha et al. 2023; Porta et al. 2025a). Crucially, PiPS achieves this without requiring any architectural modifications or retraining, thereby preserving 100% of the original model’s performance.

## PiPS

In this section, we present our post-hoc interpretability framework for explaining pre-trained semantic segmentation networks. While global model-attribution methods are typically designed to identify whole-image-level concepts for classification tasks, explaining dense pixel-level predictions requires strict spatial localization. To address this challenge, our framework leverages a post-hoc orthogonal coordinate rotation, optimizes a spatially-formulated purity objective to isolate localized concepts, and utilizes a segment-constrained attribution mechanism to ensure that explanations are structurally bound to the corresponding predicted class boundaries (an overview of our training pipeline is illustrated in Figure 3).

Orthogonal Transformation To interpret the network’s latent representations without modifying its underlying feature-extraction capabilities, we freeze all backbone parameters of the pre-trained segmentation network and modify only the final $\bar { 1 \times 1 }$ convolutional classifier layer of the segmentation head. Let $W \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times 1 \times 1 }$ and $b \in \mathbb { R } ^ { C _ { \mathrm { o u t } } }$ denote the pre-trained weights and bias of the classifier, where $C _ { \mathrm { i n } }$ and $C _ { \mathrm { { o u t } } }$ represent the input channel dimension and the number of target semantic classes, respectively. We introduce a trainable, raw parameter matrix $A \in \breve { \mathbb { R } } ^ { C _ { \mathrm { i n } } \times C _ { \mathrm { i n } } }$ . An orthogonal transformation matrix M $\in \mathbb { R } ^ { C _ { \mathrm { i n } } \times C _ { \mathrm { i n } } }$ is parameterized via the matrix exponential of the skew-symmetric matrix derived from A:

$$
A = A - A ^ { T } , \quad M = \exp ( A )
$$

The skew-symmetry of A guarantees the mathematical orthogonality of M. For a latent activation tensor $X \in \mathbf { \Sigma }$ $\mathbb { R } ^ { B \times C _ { \mathrm { i n } } \times H \times \bar { W } }$ prior to the classifier, the disentangled representations $Z \stackrel { \bullet } { \in } \mathbb { R } ^ { B \times C _ { \mathrm { i n } } \times H \times W }$ are computed by applying the coordinate rotation along the channel dimension:

$$
Z = X M
$$

![](images/12ade320b45af8ef1e017112db1c0c4d5994fb6a5014aa0fa19843591d34d80c.jpg)  
Figure 3: Overview of the PiPS training framework. The pre-trained backbone features are passed through a trainable invertible transformation matrix. We optimize a purity objective to isolate localized concepts. During training, we dynamically subset the dataset based on exemplar scores to progressively focus on the most representative prototypes.

To maintain exact mathematical equivalence with the original classifier mapping, the pre-trained weights are rotated dually:

$$
W _ { \mathrm { n e w } } = W M
$$

The final classification logits are subsequently computed as $\hat { Y } = \mathrm { C o n v } 2 \mathrm { d } ( Z , W _ { \mathrm { n e w } } , b )$ , ensuring that the transformation remains strictly post-hoc and preserves the original predictions of the network.

Optimization Objective To align the rotated latent coordinates in Z with discrete, human-interpretable semantic concepts, we train the parameter A using a spatially-formulated purity loss. For each channel $c \in \{ \bar { 1 } , \ldots , C _ { \mathrm { i n } } \}$ of a sample b in a batch, the spatial activation map $z _ { b , c } \in \mathbb { R } ^ { N }$ (where $N = H \times W )$ is converted to a spatial probability distribution using the softmax operator:

$$
p _ { b , c , i } = \frac { \exp ( z _ { b , c , i } ) } { \sum _ { j = 1 } ^ { N } \exp ( z _ { b , c , j } ) }
$$

We define the spatial entropy $H ( p _ { b , c } )$ to measure the dispersion of the channel’s activation:

$$
H ( p _ { b , c } ) = - \sum _ { i = 1 } ^ { N } p _ { b , c , i } \log ( p _ { b , c , i } + \epsilon )
$$

where $\epsilon = 1 0 ^ { - 8 }$ . To focus the optimization on highly active channels while discounting dead or noisy features, we weight each channel’s entropy by its normalized peak absolute activation:

$$
a _ { b , c } = \operatorname* { m a x } _ { i } | z _ { b , c , i } | , \quad w _ { b , c } = \frac { a _ { b , c } } { \sum _ { k = 1 } ^ { C _ { \mathrm { i n } } } a _ { b , k } + \epsilon }
$$

The objective function is then defined as the expectation of the weighted spatial entropies:

$$
\mathcal { L } _ { \mathrm { p u r i t y } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \sum _ { c = 1 } ^ { C _ { \mathrm { i n } } } w _ { b , c } H ( p _ { b , c } )
$$

Minimizing ${ \mathcal { L } } _ { \mathrm { p u r i t y } }$ guides M to align the coordinate space such that individual latent channels capture highly localized, compact spatial regions. The qualitative results obtained before and after this optimization phase can be observed in Figure 4.

Training Strategy We train the disentanglement parameters for a total of $E = 2 0$ epochs. To systematically filter out noisy spatial activations and accelerate convergence, every two epochs, the training dataset is dynamically subset. For every image, we calculate a purity-based exemplar score $S ( b , c )$ for each channel:

$$
S ( b , c ) = { \frac { \operatorname* { m a x } _ { i } | z _ { b , c , i } | } { H ( p _ { b , c } ) + \epsilon } }
$$

For each channel c, we identify the top K training samples that yield the highest exemplar scores $S ( b , c )$ . To progressively narrow the training distribution to the most representative prototypes, we linearly decay K from 100 to 5 over the course of training.

Segment Visualization During inference, we generate visual explanations by identifying the training prototypes that correspond to active latent channels. While original imagelevel attribution models evaluate features globally, our semantic segmentation setting demands that explanations be spatially bounded to the exact predicted class boundaries to avoid background contamination. Given a test image, let $\hat { Y } _ { \mathrm { m a s k } }$ be the model’s categorical segmentation output. For a target class $c _ { \mathrm { p r e d } }$ within the prediction, we define a binary spatial mask $M _ { c _ { \mathrm { p r e d } } } = \mathbb { I } ( \hat { Y } _ { \mathrm { m a s k } } = c _ { \mathrm { p r e d } } )$ . We resize $M _ { c _ { \mathrm { p r e d } } }$ to match the spatial dimensions of the latent features using nearest-neighbor interpolation, yielding $M _ { c _ { \mathrm { p r e d } } } ^ { \mathrm { r e s i z e d } }$ . For each active channel c, we restrict the feature evaluation strictly to this predicted segment:

$$
z _ { c , i } ^ { \prime } = { \left\{ \begin{array} { l l } { | z _ { c , i } | } & { { \mathrm { i f ~ } } i \in M _ { c _ { \mathrm { p r e d } } } ^ { \mathrm { r e s i z e d } } } \\ { - 1 . 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

We then identify the peak spatial activation coordinate within the segment:

$$
i ^ { * } = \arg \operatorname* { m a x } _ { i } z _ { c , i } ^ { \prime }
$$

A visual bounding patch of size $P \times P$ is centered around the projected coordinate of $i ^ { * }$ on the input image. This step ensures that the attribution is strictly localized within the segment boundary of the predicted class. The resulting test patches are compared with the top-scoring training exemplars to provide localized, human-interpretable visual evidence for the model’s pixel-level decisions.

![](images/014034431867dea780cbec3f70cd7edd5fbec9eac3193b3dbb2b4df1c6646ea0.jpg)  
Figure 4: Impact of the optimization phase on the generated explanations. Visual comparison of the extracted prototypical parts before (left) and after (right) the purity-driven optimization process on the pre-trained segmentation backbone. As observed, prototypes without the additional tuning correspond to random, uninterpretable image patches with limited explanatory properties. After the optimization, these prototypes become highly consistent and accurately correspond with the localized semantic features of the input image.

Adaptation to Point Clouds Although our framework is presented in the context of image semantic segmentation, it is not restricted to the image domain. The only requirement imposed by PiPS is that the underlying segmentation architecture produces localized latent representations that correspond to spatially coherent regions and are subsequently used for dense prediction. This property is naturally satisfied by transformer-based point cloud segmentation models, such as the Point Transformer architecture (Zhao et al. 2021), where the input point cloud is first partitioned into local groups. Given an input point cloud $\boldsymbol { P } \in \mathbb { R } ^ { N \times 3 }$ , Farthest Point Sampling (FPS) selects a set of G representative group centers,

$$
C = \operatorname { F P S } ( P , G ) ,
$$

after which each center is assigned its K nearest neighboring points using the K-Nearest Neighbors (KNN) algorithm,

$$
\mathcal { G } _ { g } = \mathrm { K N N } ( c _ { g } , K ) , \qquad g = 1 , \ldots , G .
$$

The resulting local groups are processed by the Point Transformer backbone, yielding patch-level latent representations

$$
Z = f _ { \mathrm { P T } } \big ( \{ \mathcal { G } _ { g } \} _ { g = 1 } ^ { G } \big ) .
$$

Following Point-BERT (Yu et al. 2022) and Point-MAE (Pang et al. 2022), the representations from the final transformer layer are propagated back to the original point cloud using the PointNet++ (Qi et al. 2017b) feature propagation module, where each point feature is reconstructed by interpolating the embeddings of its nearest patch centers before point-wise classification. Analogously to the image setting, we insert the invertible orthogonal transformation immediately after the final patch representations and compensate it by rotating the subsequent classifier weights, thereby preserving the original logits exactly. Consequently, the same purity objective, training strategy, and prototype retrieval procedure described above can be applied without modification, with the only distinction being that prototypes correspond to localized point groups instead of image patches.

## Experiments and Results

In the experimental section, we evaluate our PiPS framework across several scenarios. First, we provide a qualitative comparison, showcasing example predictions and comparing our results against the post-hoc method SegGradCam and the ante-hoc prototypical model ScaledProtoSeg. Then, we demonstrate that our method acts strictly as a post-hoc plugin, perfectly preserving the network’s predictive performance without altering its original output. Finally, we present the structure and results of extensive user studies assessing the human-understandability of our approach.

Explanation of model decision This section outlines the experimental results of PiPS explanations and its comparison to other XAI methods. Figure 1, illustrates the interpretability diferences between PiPS and the classical post-hoc Seg-GradCam and the ante-hoc ScaledProtoSeg on multi-object scenes from the PASCAL VOC 2012 dataset. While Seg-GradCam produces difused attribution maps that highlight general areas of importance, it falls short of capturing visually meaningful concepts such as textures or distinctive object parts. Similarly, ScaledProtoSeg struggles to provide clear, disentangled explanations without degrading the underlying predictions. In contrast, PiPS delivers a disentangled, part-by-part decomposition of the object. It not only highlights critical spatial regions but also explicitly pairs them with semantically rich training prototypes that represent these crucial visual features. Additional visual examples and comparisons can be found in the Appendix A6.

Table 1: Segmentation accuracy (mIoU %) on the PAS-CAL VOC 2012 validation dataset. We compare our posthoc method (PiPS) and SegGradCam against ante-hoc prototypical models (ProtoSeg (Sacha et al. 2023), ScaleProto-Seg (Porta et al. 2025b)). Notice that while ante-hoc modifications noticeably degrade the predictive performance of their DeepLabV2 baseline, post-hoc methods preserve 100% of the original DeepLabV3 black-box accuracy, successfully breaking the accuracy-versus-interpretability trade-of.
<table><tr><td>Method</td><td>Backbone</td><td>mIoU</td></tr><tr><td>Original Baseline</td><td>DeepLabV2</td><td>77.69% (Porta et al. 2025b)</td></tr><tr><td>ProtoSeg</td><td>DeepLabV2</td><td>71.98% (Porta et al. 2025b)</td></tr><tr><td>ScaleProtoSeg</td><td>DeepLabV2</td><td>71.80% (Porta et al. 2025b)</td></tr><tr><td>Original Baseline</td><td>DeepLabV3</td><td>85.70% (Chen et al. 2017)</td></tr><tr><td>SegGradCam</td><td>DeepLabV3</td><td>85.70%</td></tr><tr><td>PiPS (ours)</td><td>DeepLabV3</td><td>85.70%</td></tr></table>

Segmentation Performance As previously mentioned, the construction of PiPS preserves the predictive ability of the pre-trained backbone. This means that integrating the invertible transformation matrix does not change the model’s categorical output. However, since we apply additional operations, minor numerical errors might theoretically arise. To demonstrate that this situation does not occur, we present the mean Intersection over Union (mIoU) numerical accuracy on the PASCAL VOC 2012 validation dataset in Table 3. The results clearly show that while ante-hoc modifications (ProtoSeg, ScaledProtoSeg) noticeably degrade the predictive performance of their DeepLabV2 baseline, our post-hoc method preserves 100% of the original DeepLabV3 blackbox accuracy, successfully breaking the accuracy-versusinterpretability trade-of.

Point Cloud Segmentation To demonstrate that PiPS is not limited to image-based semantic segmentation, we additionally evaluate its point cloud adaptation on the ShapeNet Parts dataset (Qi et al. 2017a), which contains objects from 16 semantic categories, each composed of multiple part classes. The training procedure follows exactly the same optimization strategy as described for the image domain, with hyperparameters adjusted only to the structure of point cloud representations (see Appendix for implementation details). During inference, for every predicted part segment, we identify the latent channel exhibiting the strongest activation within the corresponding point group and retrieve the training object whose representation maximally activates the same channel. Using the hyperparameter configuration proposed by the original Point-BERT authors, the underlying segmentation model achieves an mIoU of 84.1 and an mIoU of 85.6, matching the performance reported for the original Point-BERT model. This confirms that the proposed posthoc disentanglement preserves the model logits exactly and therefore does not alter the segmentation performance. Example of the generated point cloud prototypes are presented in Figure 5.

![](images/131772d46432bd61335a5d66bc575d37e22455bf857f4d4744b4ec7290962f4b.jpg)  
Figure 5: Examples of a PiPS explanation for point cloud data. For each predicted part class (first column) of the explained object, we select the patch with the most active channel (second column) that belongs to the corresponding segment. We then select prototypes from other objects for which the same channel is the most active (third column).

User Study Results To comprehensively evaluate the human-understandability of our prototypical explanations, we conducted a user study (N = 26 participants, aged 20- 50 with no specific background) utilizing dense multi-object scenes from the PASCAL VOC 2012 dataset. Participants evaluated the generated explanations on a 5-point Likert scale (1-5) across three core criteria: (i) Visual Similarity (extent to which the prototype matches visual elements in the input image), (ii) Visual Coherence (consistency of visual concepts across prototypes in a given row), and (iii) Feature Presence (verifiability of the prototype feature within the target image).

Table 2: User study evaluation metrics $( N = 2 6 )$ . Overall ratings on a $5 \mathrm { - }$ point Likert scale (1-5) are reported alongside onesample Wilcoxon signed-rank tests against the neutral midpoint (3.0) and Friedman tests for item variability.
<table><tr><td>Metric</td><td> $\mathbf { M e a n } \pm \mathbf { S D }$ </td><td>Median</td><td>Wilcoxon  $V \left( p \mathbf { \ v s . 3 . 0 } \right)$ </td><td> $\mathbf { F r i e d m a n } \chi ^ { 2 } \left( p \right)$ </td></tr><tr><td>Visual Similarity (Q1)</td><td> $3 . 1 5 \pm 1 . 0 8$ </td><td>3.0</td><td> $3 4 1 0 . 5 ( p = 0 . 0 4 2 9 )$ </td><td> $3 4 . 3 6 ( p = 2 \cdot 1 0 ^ { - 6 } )$ </td></tr><tr><td>Visual Coherence (Q2)</td><td> $3 . 5 6 \pm 1 . 0 6$ </td><td>4.0</td><td> $4 9 1 7 . 5 ( p = 3 \cdot 1 0 ^ { - 9 } )$ </td><td> $2 7 . 4 6 ( p = 5 \cdot 1 0 ^ { - 5 } )$ </td></tr><tr><td>Feature Presence (Q3)</td><td> $3 . 6 0 \pm 1 . 0 1$ </td><td>4.0</td><td> $5 9 2 3 . 0 ( p = 1 \cdot 1 0 ^ { - 1 0 } )$ </td><td> $1 7 . 9 6 ( p = 0 . 0 0 3 0 )$ </td></tr></table>

![](images/4cb31834b560dbe2baf9ce9f911ca335f12814e7f4ceb9eb2d62a02a1f865d2c.jpg)  
Figure 6: Percentage distribution of Likert scale ratings (1-5) for Visual Similarity, Visual Coherence, and Feature Presence $( N = 2 6 )$ .

![](images/97f9c3dcd0dd9c18eb84d6d66721b149989fb380bbfee50a2637e656417e09e2.jpg)  
Figure 7: Boxplots showing Likert score distributions across question types. The white dot represents the mean score, and the dashed line denotes the neutral midpoint (3.0).

The quantitative results are summarized in Table 2, with response distributions illustrated in Figures 6 and 7. Overall, participants rated Feature Presence the highest (Mean = $3 . 6 0 \pm 1 . 0 1$ , Median = 4.0), closely followed by Visual Coherence (Mean = 3.56 ± 1.06, Median = 4.0). Onesample Wilcoxon signed-rank tests confirmed that both Feature Presence $( V = 5 9 2 3 . 0 , p = 1 \cdot 1 0 ^ { - 1 0 } )$ and Visual Coherence $( V = 4 9 1 7 . 5 , p = 3 \cdot 1 0 ^ { - 9 } )$ were rated statistically significantly higher than the neutral midpoint (3.0). Visual Similarity achieved a positive score $( \mathrm { M e a n } = 3 . 1 5 \pm 1 . 0 8$ Median = 3.0), also significantly exceeding the neutral baseline $( V ~ = ~ 3 4 1 0 . 5 , p ~ \stackrel { - } { = } ~ 0 . 0 4 \dot { 2 } 9 )$ . Statistically significant Friedman tests across all three metrics $( p \le 0 . 0 0 3 )$ indicate that user evaluations vary depending on the visual feature complexity of the underlying target images.

Limitations Since the post-hoc coordinate rotation requires exact mathematical equivalence between the transformed features and the rotated classifier weights, it is primarily applicable to architectures with a linear or $1 \times 1$ convolutional classification head, making direct adaptation to non-linear or query-based decoders less straightforward.

## Conclusions

In this work, we introduced PiPS, a fully post-hoc framework for prototype-based interpretability in semantic segmentation. By optimizing a spatially-formulated purity objective using an orthogonal coordinate transformation on frozen pre-trained backbones, PiPS extracts intuitive, regionlocalized, and human-understandable visual explanations without modifying network architectures or fine-tuning backbone weights. As a result, our method achieves 100% preservation of the base model’s original predictive performance, efectively resolving the performance-interpretability tradeof inherent to ante-hoc prototypical segmentation methods. Furthermore, we demonstrated the broad versatility of PiPS by successfully adapting it to 3D point cloud segmentation. Comprehensive qualitative evaluations, quantitative experiments, and user studies confirm the fidelity, clarity, and human-understandability of our prototypical explanations.

## References

Adebayo, J.; Gilmer, J.; Muelly, M.; Goodfellow, I.; Hardt, M.; and Kim, B. 2018. Sanity checks for saliency maps. Advances in Neural Information Processing Systems, 31.

Borycki, P.; Trędowicz, M.; Janusz, S.; Tabor, J.; Spurek, P.; Lewicki, A.; and Struski, Ł. 2026. EPIC: Explanation of Pretrained Image Classification Networks via Prototypes. Proceedings of the AAAI Conference on Artificial Intelligence, 40(21).

Chen, C.; Barnett, A.; Su, J.; Rudin, C.; and Venkatasubramanian, S. 2020. Concept whitening for interpretable image recognition. Nature Machine Intelligence, 2: 772–782.

Chen, C.; Li, O.; Tao, D.; Barnett, A.; Rudin, C.; and Su, J. K. 2019. This looks like that: Deep learning for interpretable image recognition. Advances in Neural Information Processing Systems, 32.

Chen, L.-C.; Papandreou, G.; Schrof, F.; and Adam, H. 2017. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587.

Everingham, M.; Van Gool, L.; Williams, C. K. I.; Winn, J.; and Zisserman, A. 2012. The PASCAL Visual Object Classes

Challenge 2012 (VOC2012) Results. http://www.pascalnetwork.org/challenges/VOC/voc2012/workshop/index.html.

Ghorbani, A.; Wexler, J.; Zou, J. Y.; and Kim, B. 2019. Towards automatic concept-based explanations. In Advances in Neural Information Processing Systems, volume 32.

Gipiškis, R. 2025. Post-hoc explainable semantic image segmentation: applications for interpretability and adversarial attacks. Ph.D. thesis, Vilniaus universitetas.

Lundberg, S. M.; and Lee, S.-I. 2017. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.

Pang, Y.; Wang, W.; Tay, F. E. H.; Liu, W.; Tian, Y.; and Yuan, L. 2022. Masked Autoencoders for Point Cloud Selfsupervised Learning. arXiv:2203.06604.

Porta, H.; Dalsasso, E.; Marcos, D.; and Tuia, D. 2025a. Multi-Scale Grouped Prototypes for Interpretable Semantic Segmentation. In 2025 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2869–2880. IEEE.

Porta, H.; Dalsasso, E.; Marcos, D.; and Tuia, D. 2025b. Multi-Scale Grouped Prototypes for Interpretable Semantic Segmentation. In 2025 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2869–2880. IEEE.

Qi, C. R.; Su, H.; Mo, K.; and Guibas, L. J. 2017a. Point-Net: Deep Learning on Point Sets for 3D Classification and Segmentation. arXiv:1612.00593.

Qi, C. R.; Yi, L.; Su, H.; and Guibas, L. J. 2017b. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. arXiv:1706.02413.

Ribeiro, M. T.; Singh, S.; and Guestrin, C. 2016. "Why should I trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135–1144.

Sacha, M.; Rymarczyk, D.; Struski, Ł.; Tabor, J.; and Zieliński, B. 2023. Protoseg: Interpretable semantic segmentation with prototypical parts. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 1481–1492.

Selvaraju, R. R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; and Batra, D. 2017. Grad-CAM: Visual Explanations from Deep Networks via Gradient-b ased Localization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 618–626.

Yu, X.; Tang, L.; Rao, Y.; Huang, T.; Zhou, J.; and Lu, J. 2022. Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling. arXiv:2111.14819.

Zhao, H.; Jiang, L.; Jia, J.; Torr, P.; and Koltun, V. 2021. Point Transformer. arXiv:2012.09164.

## A Appendix / Supplemental Material

## A.1 Explanations of model decision

In this section, we provide additional results of experiments regarding the explanations of model decisions made by PiPS, comparing it to both the post-hoc approach SegGradCam and the ante-hoc method ScaledProtoSeg. The experimental results, which highlight our method’s ability to provide disentangled, part-by-part decomposition of objects across dense predictions, are presented on multi-object scenes from the PASCAL VOC 2012 in Fig. 3.

## A.2 More details on user study

To evaluate the human-understandability of our prototypical explanations, we conducted a user study via the Google Forms platform. To ensure a diverse assessment, the survey questions were randomized, meaning each participant received a subset of unique queries. Data quality was strictly maintained by filtering out unengaged submissions— specifically, responses where participants selected the identical answer across all questions. In total, we collected 26 valid responses. Prior to the evaluation, participants were provided with comprehensive instructions and visual examples to familiarize themselves with the interpretation of our method’s results.

Figures 28, 29 illustrate example questions used in the user study, where users were asked to evaluate:

• The quality of the generated explanations.

• Which method generates better segmentations (only PiPS and ScaledProtoSeg were compared, since SegGradCam generates the same results as PiPS because of their posthoc nature).

## A.3 Segmentation Performance

As previously discussed, the strictly post-hoc design of PiPS maintains 100% of the predictive performance of the pretrained base model. In other words, integrating the invertible transformation matrix yields the exact same categorical segmentation output for any image as the original model. While additional operations could potentially introduce minor numerical errors, we demonstrate that this is not the case. The mIoU metric values are presented in table 3.

Table 3: Segmentation accuracy (mIoU %) on the PAS-CAL VOC 2012 validation dataset. We compare our posthoc method (PiPS) and SegGradCam against ante-hoc prototypical models (ProtoSeg (Sacha et al. 2023), ScaleProto-Seg (Porta et al. 2025b)). Notice that while ante-hoc modifications noticeably degrade the predictive performance of their DeepLabV2 baseline, post-hoc methods preserve 100% of the original DeepLabV3 black-box accuracy, successfully breaking the accuracy-versus-interpretability trade-of.
<table><tr><td>Method</td><td>Backbone</td><td>mIoU</td></tr><tr><td>Original Baseline</td><td>DeepLabV2</td><td>77.69% (Porta et al. 2025b)</td></tr><tr><td>ProtoSeg</td><td>DeepLabV2</td><td>71.98% (Porta et al. 2025b)</td></tr><tr><td>ScaleProtoSeg</td><td>DeepLabV2</td><td>71.80% (Porta et al. 2025b)</td></tr><tr><td>Original Baseline</td><td>DeepLabV3</td><td>85.70% (Chen et al. 2017)</td></tr><tr><td>SegGradCam</td><td>DeepLabV3</td><td>85.70%</td></tr><tr><td>PiPS (ours)</td><td>DeepLabV3</td><td>85.70%</td></tr></table>

## A.4 Dataset

In our experiments, we utilized the PASCAL VOC 2012 (Everingham et al. 2012) dataset, which is frequently employed in dense prediction and segmentation model evaluations. Its dense multi-object scenes, severe occlusions, and high intra-class spatial variations pose significant challenges for prototype-based models. It is worth noting that only a few of the previous prototypical parts-based methods, namely ProtoSeg, and than ScaledProtoSeg, have been successfully evaluated on such complex scenes.

## A.5 PACAL VOC 2012 Experiments

All experiments were conducted using the Athena GPU cluster. It is important to note that since PiPS operates entirely post-hoc, the underlying segmentation network remains strictly frozen and is not retrained. The computational overhead is therefore dedicated exclusively to a lightweight optimization phase: training the post-hoc orthogonal transformation matrix via the spatial purity objective, and algorithmically mining a set of representative spatial prototypes from the training data. The duration of this process is highly dependent on the size of the training set and the latent resolution of the base model. For the PASCAL VOC 2012 dataset, optimizing the transformation parameters and dynamically subsetting the exemplars takes approximately one hour on a single GPU node for the DeepLabV3 model.

## A.6 PASCAL VOC 2012 examplec

Figures 8 through 16 present qualitative examples demonstrating how our method explains predictions for various object classes from the PASCAL VOC 2012 dataset. All visualizations share a consistent layout, where the top row displays the original image, the ground truth, and the model-generated segmentation. Furthermore, the first column presents the original image, the second shows the target segment to be explained, and the subsequent columns illustrate the generated prototypes (explanations).

## A.7 Point Cloud Experiments

The point cloud adaptation of PiPS was implemented on top of the Point Transformer V1 segmentation architecture provided by the Point-BERT framework. The backbone consists of 12 transformer layers with a latent embedding dimension of 384 channels. To ensure a fair comparison with the original model, all experiments were initialized using the publicly available pre-trained Point-BERT weights, while the backbone parameters remained completely frozen throughout training.

The disentanglement module was optimized using the same procedure as described and is attached to the last layer (output layer) of point transformer. The optimization was performed for 40 epochs. Following the dynamic exemplar mining strategy, each latent channel was initially associated with the top 40 representative prototypes, with this number linearly reduced to 5 prototypes per channel by the end of training. The prototype assignments were recomputed every two epochs based on the current exemplar scores.

The semantic segmentation head follows the original Point-BERT implementation. Specifically, the patch-level representations produced by the final Point Transformer layer are propagated back to the original point cloud using Point-Net++ feature propagation layers before point-wise classification. Since PiPS is inserted only as an invertible transformation of the latent representation and the subsequent classifier weights are rotated accordingly, no architectural modifications or retraining of the segmentation model are required.

Following the experimental protocol of Point-BERT, each point cloud was uniformly subsampled to 2,048 points prior to processing, and the disentanglement module was trained using a batch size of 32.

## A.8 Point Cloud examples

Figures from 17 to 27 present qualitative examples demonstrating how our method explains predictions for diferent object classes from the ShapeNetPart dataset. All visualizations follow the same layout. The top row shows the ground-truth part segmentation together with the predicted segmentation. The first column presents the explained object, highlighting the selected patch corresponding to the most active channel associated with a particular part class. The remaining columns display the most similar prototypes retrieved from diferent objects, which also exhibit high activation for the corresponding channels.

## A.9 Explaining model prediction

After completing the optimization of the parameter matrix A and aligning the orthogonal coordinate space M, the next step is to explain the model’s pixel-level predictions for a given input image. Unlike image-level classification methods that rely on global pooling, dense semantic segmentation demands that explanations be spatially bounded to the exact predicted class boundaries to avoid background contamination.

For an input image I, we restrict the feature evaluation strictly to the spatial mask of the predicted class $c _ { p r e d }$ . We then identify the peak spatial activation coordinates within the segment to extract the most relevant localized visual patches.

![](images/673213df4018914ab838e49a61b6f3bb220cf05c618ac5124d778f2f832e02b1.jpg)  
Figure 8: Visual explanation for the ’bird’ object identified during semantic segmentation

![](images/f2ea5958295addc77e0ebc868bc47150cee198bbaab9a87a4e24784aaeea102e.jpg)  
Figure 9: Visual explanation for the ’bus’ object identified during semantic segmentation

![](images/1426f583fef68af2a6b3f047ee7ed17f3378bebf8f97e974b92e987c8dfb1d63.jpg)  
Figure 10: Visual explanation for the ’pottedplant’ object identified during semantic segmentation

![](images/7a5853b9bb697b335d56f2217ace35a3d14241684d8194ba60d259f1b2d01446.jpg)  
Figure 11: Visual explanation for the ’dog’ and ’pearson’ objects identified during semantic segmentation

![](images/6b744f5a767600548502db7cb66f97d8cab75094f978857dda8469e7c34eaf66.jpg)  
Figure 12: Visual explanation for the ’chair’ and ’tvmonitor’ objects identified during semantic segmentation

![](images/a1ad52379fb6fd267932aa0fe02790d609d60a31f4225e6c807a2bf933169ec8.jpg)  
Figure 13: Visual explanation for the ’bus’ and ’car’ objects identified during semantic segmentation

![](images/0e672c49177d25fd9af54300b01a610d8eb0a8b6a916845ff3d5804c3cce02cc.jpg)  
Figure 14: Visual explanation for the ’chair’, ’sofa’ and ’tvmonitor’ objects identified during semantic segmentation

![](images/27f290aec56a70ef4716a25a239696536467020bfd7d2aab479c91527d6f9fa7.jpg)  
Figure 15: Visual explanation for the ’dog’, ’person’ and ’sofa’ objects identified during semantic segmentation

![](images/0cd0985aa10adab3cb3d6cb40cb412fc4068e0f422daba36e2bf9f81f3708492.jpg)  
Figure 16: Visual explanation for the ’chair’, ’diningtable’ and ’tvmonitor’ objects identified during semantic segmentation

![](images/77f99de6f04c4192e05be0afc046f0f87c3421616d86b5a844f2cf8fea12c808.jpg)  
Figure 17: Explanation of the guitar object from the ShapeNetPart dataset

![](images/4365842ad0f96be60fde2070c79567644736e430c182873132b3041b44a213fc.jpg)  
Figure 18: Explanation of the motor object from the ShapeNetPart dataset

![](images/58def5ac92dfef09270fab9ae6b05a2f63ba2fd947e0dda2ae18e42b77a40e58.jpg)  
Figure 19: Explanation of the rocket object from the ShapeNetPart dataset

![](images/0439445991a0f1d76c2972fb2f4647eda349ae510ca7a23bcca33d8383bfc6da.jpg)  
Figure 20: Explanation of the mug object from the ShapeNetPart dataset

![](images/ffbb010e1e1cc046fe9bb463720e243d4233a143ef1c70546d0e74e9e5c8a39c.jpg)  
Figure 21: Explanation of the earphone object from the ShapeNetPart dataset

![](images/f9771323f42e5c40aff53455bbbd26116e9e429fec815b727d8be82e13d5fe31.jpg)  
Figure 22: Explanation of the chair object from the ShapeNetPart dataset

![](images/d33c7e3c1004efc5acdac0092f4a8d5ccc78dd5ca15523592d6ba15eb577ec64.jpg)  
Figure 23: Explanation of the table object from the ShapeNetPart dataset

![](images/62cab9de50e156a6f36efaaf5e17cdf6aa663da23e5122459e69e61284dd6d6d.jpg)  
Figure 24: Explanation of the skateboard object from the ShapeNetPart dataset

![](images/24d2c68e53aca07f019724e291a09ac92ed268e23c5bf20a6d2a6f95fd54919d.jpg)  
Figure 25: Explanation of the knife object from the ShapeNetPart dataset

![](images/c03a06b6241c9fa1ef37d0ed5e6c9d569ad0e59c5f5d943bc7ebca13a897239e.jpg)  
Figure 26: Explanation of the car object from the ShapeNetPart dataset

![](images/008d3983f6490d2eb4e9c67f55010543c06d503c8a6579910d74d84dec075722.jpg)  
Figure 27: Explanation of the cap object from the ShapeNetPart dataset

![](images/6727d769ea54ec8520f02b7730989d06a5ca21a952c1d1fc6965a27635613409.jpg)  
Figure 28: An exemplary interface from the user study. While the specific images varied across trials, all three question types shared a consistent layout featuring six visual examples. Participants evaluated the generated explanations using a 5-point Likert scale based on three distinct criteria: (1) Visual Similarity: “To what extent is the presented prototype similar to the elements visible in the input image?” (1 – completely dissimilar, 2 – rather dissimilar, 3 – partially similar, 4 – similar, 5 – very similar). (2) Visual Coherence: “To what extent do the prototypes in the same row represent a visually coherent idea or theme?” (1 – completely inconsistent, 2 – rather inconsistent, 3 – partially consistent, 4 – consistent, 5 – very consistent). (3) Feature Presence: “Does the prototype represent a feature that can actually be observed in the input image?” (1 – the feature is not present, 2 – rather not visible, 3 – hard to tell, 4 – rather visible, 5 – clearly visible).

Option 1  
![](images/90d5f1a65619896f094435224aa147feec41c93c5b90ddf5c4d380c2933e591a.jpg)  
Option 2  
Figure 29: Segmentation quality comparison interface. In this task of the user study, participants were asked a question: “Which method generates a better segmentation?”. Users compared the spatial segmentation masks produced by PiPS against those generated by the ante-hoc ScaledProtoSeg method. For this comparison, participants could explicitly choose their preferred segmentation or select a neutral “Neither is better” option. Notably, SegGradCam was deliberately excluded from this specific task; because both PiPS and SegGradCam are strictly post-hoc methods operating on the identical frozen DeepLabV3 backbone, they inherently yield the exact same categorical segmentation outputs. Thus, this task efectively evaluated the visual quality of our perfectly preserved baseline predictions against the altered masks produced by the ante-hoc architecture.