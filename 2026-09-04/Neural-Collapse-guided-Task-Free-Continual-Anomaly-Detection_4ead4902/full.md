# Neural-Collapse-guided Task-Free Continual Anomaly Detection

Xiaotong Kong, Chaoyang Song, Ziai Zhou, Jinxia Zhang\*, Kanjian Zhang, Haikun Wei

Abstract—Recent years have witnessed growing interest in continual anomaly detection for industrial visual inspection. However, real-world manufacturing environments exhibit unpredictable shifts in data distributions, rendering task-dependent continual learning assumptions impractical. To address this limitation, we formulate industrial anomaly detection as a taskfree continual learning problem and propose NC-TFAD, a neuralcollapse-inspired, geometry-driven framework for learning from non-stationary data streams without task boundaries. NC-TFAD freezes a pretrained backbone and aligns streaming features to a simplex Equiangular Tight Frame (ETF) prototype space to stabilize representation geometry under non-stationary streams. To satisfy the NC-inspired geometric construction in the absence of real anomalies, we generate synthetic anomaly samples as auxiliary anchors during training. Building on this geometry, we further introduce inter- and intra-class regularization together with a Focal Neural Collapse Contrastive (FNCC) loss to suppress representation drift and improve normal–anomaly separability. Finally, a normal-patch-prototype-guided localization branch constructs calibrated patch-wise deviation maps from normal training samples and fuses them with a weak self-attention prior, producing anomaly heatmaps without pixel-level annotations. Extensive experiments on MVTec AD and VisA show that NC-TFAD consistently outperforms representative task-free continual learning methods adapted from general vision, as well as unified anomaly detection baselines, in both image-level detection and pixel-level localization under the task-free continual learning protocol. These results highlight that geometry-driven modeling offers an effective and robust solution for task-free continual anomaly detection in real-world industrial applications.

Index Terms—Anomaly detection, Task-free, Continual learning, Neural collapse

## I. INTRODUCTION

NOMALY detection (AD) is a fundamental problem in industrial visual inspection and intelligent manufacturing systems. It is critical for ensuring product quality, reducing production risks, and enabling automated workflows [1], [2]. In practice, anomalies exhibit diverse appearances, wide scale variation, and severe class imbalance. These factors make manual, expertise-driven inspection inefficient and inconsistent in accuracy and repeatability. As industrial data volumes grow, manual inspection can no longer meet the requirements of high precision, low latency, and long-term reliability, accelerating the adoption of deep-learning-based AD in industrial settings.

In recent years, deep learning has significantly advanced the development of anomaly detection methods [3], [4]. Compared with rule-based or shallow-feature approaches, deep learning methods can learn representations directly from data. They can exploit large collections of normal samples to acquire discriminative features. As a result, they typically generalize better under complex backgrounds, diverse defect types, and cross-category scenarios. Existing deep anomaly detection methods have evolved from the paradigm of one-model-onecategory toward one-model-for-all [5], [6]. Earlier categoryspecific models perform well in controlled settings but require frequent retraining as products or production lines change, leading to high maintenance cost and poor scalability. Unified frameworks such as UniAD [7] alleviate these issues by consolidating multiple categories into a single model and reducing deployment and storage overhead.

Since most existing methods assume that all training data are jointly available during model training [1], [8], [9], [10], a substantial gap remains between unified AD models and real industrial deployment. In practical manufacturing environments, new product categories and inspection objects are introduced incrementally. In this context, continual learning (CL) [11], [12] has been adopted in anomaly detection to address distributional shifts over time. Nevertheless, conventional CL methods generally rely on explicit task boundaries or task identifiers. Such information is rarely available in industrial AD scenarios, where task transitions are inherently ambiguous and task labels are unavailable during inference [13]. Consequently, task-free continual learning (TFCL) is regarded as a more realistic and deployment-oriented learning paradigm for industrial anomaly detection.

While TFCL has achieved promising progress in general vision tasks [14], [15], its application to industrial anomaly detection remains limited. Existing task-free frameworks are not specifically designed for anomaly detection and struggle to reconcile long-term knowledge retention with the detection of subtle anomalies. The problem is further exacerbated by the lack of anomaly supervision.

To bridge this gap, we formulate industrial AD as a task-free continual learning problem and propose Neural-Collapse-guided Task-Free Anomaly Detection (NC-TFAD), a geometry-driven framework specifically tailored for this demanding setting. Unlike existing continual or unified anomaly detection approaches that rely on explicit task boundaries, replay buffers, or joint training assumptions, NC-TFAD enables continual adaptation under unknown task transitions using only the currently observed data stream. Specifically, NC-TFAD freezes a pretrained backbone to extract stable visual features and employs a lightweight linear projection layer to align streaming features to a Simplex Equiangular Tight Frame (ETF) prototype space induced by neural collapse [16].To instantiate the NC-inspired normal–anomaly geometry in the absence of real anomaly supervision, we introduce a synthetic anomaly generation strategy that provides auxiliary geometric anchors for the otherwise unobserved anomalous direction. The resulting synthetic anomalies provide weak image-level supervision for establishing a stable normal–anomaly geometric reference, rather than explicitly modeling the full appearance distribution of real industrial defects. Building on this geometric structure, we introduce NC-guided inter- and intra-class regularization to preserve the ETF-induced geometry by promoting intra-class compactness and inter-class separation, thereby stabilizing the embedding space under distribution shifts. We further propose a Focal Neural Collapse Contrastive (FNCC) loss that emphasizes hard sample–sample and sample–prototype pairs, improving sample-level discriminability and anomaly separability under non-stationary streams. In addition, NC-TFAD includes a normal-patch-prototype-guided localization branch. It builds a compact prototype set from normal training patches only, scores each test patch by its calibrated deviation from the normal patch manifold, and uses backbone self-attention only as a weak spatial prior.

Extensive experiments on the MVTec AD and VisA datasets demonstrate that NC-TFAD significantly outperforms existing state-of-the-art approaches under the task-free continual anomaly detection setting.

The main contributions of this work are summarized as follows:

1) We formulate industrial anomaly detection under a task-free continual learning protocol in which normal samples arrive in a single-pass non-stationary stream without task boundaries, category identifiers, or access to historical data, better reflecting real-world industrial inspection with continuously evolving data distributions.

2) We propose NC-TFAD, which reformulates neuralcollapse geometry for task-free continual anomaly detection in a category-agnostic manner. Unlike standard multiclass ETF formulations tied to semantic classes, NC-TFAD employs two fixed ETF directions to represent normal and anomalous states shared across all categories. Since real anomalies are unavailable during training, synthetic anomalies are introduced as auxiliary geometric anchors to instantiate the missing anomalous direction without anomaly supervision.

3) We develop an NC-guided continual optimization mechanism that integrates inter-class alignment and intraclass compactness with a Focal Neural Collapse Contrastive (FNCC) loss. FNCC jointly models sample– sample and sample–prototype relations within a unified contrastive objective and applies focal weighting to hard positive pairs and ambiguous prototype assignments, improving robustness under non-stationary distributions.

4) We further introduce a normal-patch-prototype-guided localization branch that uses only normal training patches to construct a compact localization memory. During inference, patch-wise prototype deviations are calibrated by the normal distance distribution and fused with a weak self-attention prior, improving pixel-level localization without requiring pixel-level supervision.

## II. RELATED WORKS

## A. Continual Anomaly Detection

With the rapid progress of embodied intelligence and intelligent manufacturing, learning from non-stationary data streams has attracted increasing attention. In industrial visual inspection, continual anomaly detection (CAD) has emerged as an important application of continual learning.

Most existing CAD methods focus on mitigating catastrophic forgetting while introducing new inspection objects [17]. For example, UCAD [18] enhances feature discriminability via contrastive learning and incorporates prompt mechanisms together with knowledge memory structures to alleviate performance degradation. IUF [19] improves the adaptability to new objects by employing object-aware attention and semantic compression loss. More recently, diffusion-based CAD methods have also attracted growing interest. CDAD [20] reduces gradient interference between old and new tasks via gradient projection, but it still relies on explicit task delineation during updates.

Overall, existing CAD approaches often depend on task identifiers, stage boundaries, or external memory. These assumptions are misaligned with industrial settings, where task transitions are ambiguous, data arrive continuously, and historical samples are difficult to revisit.

## B. Task-Free Continual Learning

Task-free continual learning (TFCL) [13] closely matches real-world streaming scenarios, where data arrive continuously and neither training nor inference has access to task boundaries or identifiers. Models must therefore update online under unknown distribution shifts.

To address TFCL, one line of research dynamically models distributional changes and adaptively extends the model structure [21]. For instance, Dynamic Cluster Memory (DCM) [22] uses knowledge-difference-driven clustering to identify shifts and dynamically grows memory under unsupervised learning. With the advancement of large pretrained models, another line of work freezes the backbone network and introduces lightweight tuning modules to adapt to TFCL scenarios [23]. MVP [24] leverages the general representation capability of pretrained vision models and proposes instance-level logit masking and contrastive visual prompting, achieving notable performance gains in task-free settings. Similarly, Online-LoRA [25] integrates low-rank adaptation modules into Vision Transformers. It automatically detects distribution shifts based on loss-landscape variations, enabling online and task-free parameter-efficient continual learning.

Despite these advances, TFCL remains underexplored for anomaly detection. The lack of anomaly supervision and the need for stable, well-separated feature spaces make existing TFCL methods difficult to transfer directly to this domain.

## C. Neural Collapse

Neural collapse (NC) is an important theoretical finding in deep learning, revealing that deep classification networks tend to form a highly symmetric geometric structure in their feature space during the terminal phase of training. Extensive theoretical studies [26], [27] demonstrate that, under simplified models or appropriate regularization, NC corresponds to the global optimum of cross-entropy [28] or mean-squared-error [29] objectives.

Beyond theoretical analysis [30], [31], researchers have begun to investigate how NC structures may be actively induced in different learning scenarios. In2NeCT [32] addresses classimbalance issues by introducing intra-class and inter-class collapse calibration strategies. These strategies jointly promote feature compactness and separability, mitigating geometric distortion caused by imbalanced data distribution. In recent years, NC has also been explored in continual and online learning settings. FCA [26] alleviates catastrophic forgetting in fewshot incremental learning by aligning features with fixed ETFbased class prototypes. Similarly, DYSON [33] estimates and updates the optimal geometric structure in an online manner. It guides feature representations toward an ideal collapsed state under task-free incremental learning and achieves competitive performance on natural image classification benchmarks.

These studies collectively indicate that neural collapse is not merely a passive phenomenon emerging during training. Instead, it can serve as an active geometric inductive bias that guides representation learning under non-stationary environments.

## III. METHODS

## A. Problem Formulation and Preliminary

1) Task-Free Continual Anomaly Detection: This work investigates the problem of task-free continual anomaly detection (TF-CAD), where training follows a single-pass online continual learning protocol without task boundaries. Given a chronologically ordered stream $\textit { \textbf { D } } = \ \{ B _ { 1 } , B _ { 2 } , . . . , B _ { n } \}$ each incoming batch $\boldsymbol { B } _ { n } ~ = ~ \{ x _ { i } \} _ { i = 1 } ^ { N _ { n } }$ contains only normal training samples. No real anomalous image or anomaly annotation is available during training. For optimization, synthetic anomalous counterparts are generated from the current normal samples to form an augmented batch, where $y _ { i } = 0$ denotes an original normal sample and $y _ { i } ~ = ~ 1$ denotes its synthetically perturbed counterpart. Hence, $y _ { i }$ is a synthetic training label rather than a ground-truth anomaly label. The data are sequentially fed into the model in a batch-wise manner, while no task boundary or task identifier is provided, and the model is unaware of when or how the underlying distribution changes. Parameter updates use only the current normal batch and its synthetically generated counterparts, without access to historical samples. After training, a unified evaluation is performed over all categories that appear in the data stream. The evaluation assesses whether the learned feature space remains stable while preserving adaptability to continuously evolving distributions. Here, “task-free” refers to the information available to the learner. Although category information is used offline to construct the non-stationary stream, no category label, task identifier, or transition boundary is provided during training. The model therefore performs continual updates without explicit task information.

2) Neural Collapse and Equiangular Tight Frame: Neural Collapse (NC) refers to a highly symmetric geometric phenomenon that emerges in deep classification networks during the terminal phase of training. When the training error approaches zero and optimization proceeds further, the features of the final layer, class means, and classifier weights spontaneously converge to an analytically characterizable structured configuration. This configuration can typically be represented by a simplex Equiangular Tight Frame (ETF).

Consider a K-class classification problem with $K \ \geq \ 2$ Let $u _ { k }$ denote the mean feature of class k, and assume it is centered such that $\textstyle \sum _ { k = 1 } ^ { K } u _ { k } = 0$ . The corresponding simplex ETF can be constructed as

$$
M = \sqrt { \frac { K } { K - 1 } } U \left( I _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \top } \right) ,\tag{1}
$$

where $M = [ \mathbf { m } ^ { 1 } , \dots , \mathbf { m } ^ { K } ] \in \mathbb { R } ^ { d \times K }$ is the matrix of class prototypes, $U \in \mathbb { R } ^ { d \times K }$ has orthonormal columns satisfying $U ^ { \top } U = I _ { K }$ , and $\mathbf { 1 } _ { K } \in \mathbb { R } ^ { K }$ denotes the K-dimensional allones vector. This structure ensures that all prototypes have equal norm and that the cosine of the angle between any pair of classes equals $- \frac { 1 } { K - 1 }$ . As a result, the prototypes achieve maximal angular separation and minimal coherence on the unit hypersphere.

The Neural Collapse phenomenon can be summarized as:

• NC1: Features of samples from the same class converge to their class mean;

• NC2: Class means converge to the vertices of a simplex ETF, with equal norm and equal inter-class angles;

• NC3: Classifier weight vectors align directionally with their corresponding class means;

• NC4: The classifier decision rule becomes equivalent to a nearest-class-mean decision in Euclidean space.

In the binary normal–anomaly setting considered in this work, the simplex ETF yields two antipodal prototypes that define a fixed and maximally separated geometric reference for the two anomaly states. This property is well suited to task-free continual anomaly detection, where non-stationary data streams can progressively distort the representation space and shift the decision boundary. Motivated by this observation, we adopt the NC geometry as a category-agnostic structural prior to maintain a consistent normal–anomaly organization throughout continual adaptation. Specifically, the fixed ETF prototypes provide stable reference directions, while NCinspired class-mean alignment and intra-class compactness further constrain the evolving embeddings to remain discriminative under distribution shifts.

## B. Framework Overview

Fig.1 illustrates the overall architecture of NC-TFAD. The framework consists of a training process and an inference-time localization process.

During training, each batch contains both normal and synthetic anomalous samples with labels $y _ { i } \in \{ 0 , 1 \}$ . An input image $x _ { i }$ is first processed by the frozen backbone $\mathcal F ( \cdot )$ to obtain the final-layer token sequence. For the image-level training branch, we use the LayerNorm-normalized [CLS] token as the global image representation, rather than pooling all patch tokens:

![](images/7e1262ef32e18e7c4cc1b88fde28691164fb01d3d49c4aadcfaf5169dbd58d5f.jpg)  
Fig. 1. The overview of NC-TFAD. During training, normal images and their synthetically perturbed counterparts form a task-free data stream that is fed into a frozen backbone and a learnable linear projection layer. Specifically, the LayerNorm-normalized [CLS] token from the last Transformer block is projected and ℓ<sub>2</sub>-normalized to form the embedding aligned with fixed ETF prototypes under NC alignment loss (NC Loss), NC-guided regularization (Reg Loss), and Focal Neural Collapse Contrastive loss (FNCC Loss). During inference, both the backbone and the trained linear projection layer are frozen, and the same [CLS]-based pathway is used with the fixed ETF prototype classifier to obtain the image-level anomaly score. In parallel, the localization branch constructs a normal patch prototype bank from normal training samples, computes calibrated patch-to-prototype deviation maps for test images, and fuses them with pre-projection CLS-to-patch self-attention as a weak spatial prior to generate the final anomaly map.

$$
f _ { i } ^ { \mathrm { c l s } } = \mathrm { L N } ( \mathcal { F } _ { \mathrm { c l s } } ( x _ { i } ) ) ,\tag{2}
$$

where $\mathcal { F } _ { \mathrm { c l s } } ( x _ { i } )$ denotes the [CLS] token extracted from the last Transformer block. This [CLS] feature is then projected into a learnable embedding space through a linear projection layer g(·):

$$
z _ { i } = \frac { g ( f _ { i } ^ { \mathrm { c l s } } ) } { \| g ( f _ { i } ^ { \mathrm { c l s } } ) \| _ { 2 } } , \quad z _ { i } \in \mathbb { R } ^ { d } , \quad \| z _ { i } \| _ { 2 } = 1 .\tag{3}
$$

Here, d denotes the embedding dimension and the resulting embeddings form the set $Z ~ = ~ \{ z _ { 1 } , z _ { 2 } , . ~ . ~ . , z _ { n } \}$ . The ETF prototypes are also unit-normalized, i.e., $\| p _ { k } \| _ { 2 } = 1$ . Hence, all subsequent sample–sample and sample–prototype dot products are evaluated in the normalized embedding space and correspond to cosine similarities. This embedding space serves as a unified domain for geometric alignment, regularization, and contrastive learning.

Within this space, two fixed ETF prototypes $\mathcal { P } = \{ p _ { 0 } , p _ { 1 } \}$ are pre-constructed according to Eq. (1), corresponding to the normal and anomalous classes. Here, the prototype index represents the anomaly state rather than the industrial object category. Specifically, $p _ { 0 }$ and $p _ { 1 }$ are shared by normal and anomalous samples, respectively, across all object categories.

Category identities are not provided to the model and are not used to construct category-specific prototypes. This categoryagnostic design provides a fixed normal–anomalous geometric reference that does not require task or category identification as the data distribution evolves. These prototypes remain fixed during training and provide stable geometric references throughout the stream. The learnable projection layer is optimized with three complementary objectives: NC Loss aligns individual embeddings with their ETF prototypes; Reg Loss preserves the global normal–anomaly geometry through classmean alignment and intra-class compactness; and FNCC Loss improves sample-level discriminability by modeling sample– sample and sample–prototype relations with emphasis on hard pairs. Together, they maintain a structured and discriminative embedding space under non-stationary distributions. The fixed ETF prototypes also serve as geometric classifier weights, yielding predictions via:

$$
{ \hat { y } } _ { i } = \arg \operatorname* { m a x } _ { k \in \{ 0 , 1 \} } z _ { i } ^ { \top } p _ { k } .\tag{4}
$$

During inference, both the pretrained backbone and the learned linear projection layer are frozen. Given a test image $x ,$ the representation used for image-level anomaly detection is obtained by extracting the LayerNorm-normalized [CLS] token from the last Transformer block, followed by the same 384-to-384 linear projection and $\ell _ { 2 }$ normalization used during training. The resulting embedding is then compared with the fixed ETF prototypes to obtain the image-level anomaly prediction and anomaly confidence. For pixel-level localization, NC-TFAD uses the patch tokens extracted by the frozen backbone to construct a normal patch prototype bank from normal training samples. At inference time, each test patch is scored by its calibrated deviation from the nearest normal prototype, yielding a prototype-based anomaly response. This response is further fused with the original backbone CLS-topatch self-attention as a weak spatial prior. We use the preprojection [CLS] attention because it directly reflects tokenlevel interactions in the frozen vision backbone, whereas the projected [CLS] embedding is optimized for image-level ETF discrimination and may discard spatial correspondence useful for localization.

## C. Synthetic Anomaly Generation

In industrial anomaly detection, training data typically contain only normal samples, whereas the proposed NC-based normal–anomaly geometry requires reference directions for both states. To instantiate the otherwise unavailable anomalous direction, we generate synthetic anomalies from normal images and use them exclusively as auxiliary geometric anchors during training. These synthetic samples provide weak imagelevel abnormality cues that enable the projection layer to establish a discriminative normal–anomaly geometry in the absence of real anomaly supervision.

As shown in Fig.2, following prior work [6], [35], [36], the data are divided into object categories and texture categories. For object categories, a coarse foreground is obtained by grayscale thresholding (Otsu or triangle, with inversion when necessary) and refined via morphological operations. For texture data, the entire image is treated as foreground, as defects may appear anywhere on the surface.

To determine candidate anomaly regions, we employ Perlin noise to generate irregular spatial patterns with natural and diverse shapes. These candidate anomaly regions are intersected with the foreground to ensure that anomalies are injected only within semantically valid areas, while regions outside the foreground are discarded. The resulting intersection defines the final anomaly mask. Within this mask, we then apply one of three local perturbations with equal probability: appearance perturbation that alters color and intensity statistics, local blurring that disrupts fine-grained texture patterns, or local rearrangement that breaks spatial continuity by shuffling subregions. For texture categories, we restrict the operation set to local blurring and local rearrangement. For object categories, we further impose constraints on the number and size of connected anomalous regions to avoid overly fragmented artifacts, and apply smooth alpha blending along mask boundaries to reduce visual discontinuities.

The resulting perturbed images serve as synthetic anomalous samples for image-level geometric learning. The spatial mask is used only to constrain local perturbations and is discarded after synthesis, without being provided to the backbone or used as pixel-level supervision. Since the ViT [CLS] token aggregates information from patch representations through self-attention, localized perturbations can still influence the global representation, enabling the projection layer to learn image-level normal–anomaly separation without explicit anomaly-location supervision.

![](images/d96824b3e9645cfe56fb4147de826456a8471a7357721a63d25dc9587a591ddd.jpg)  
Fig. 2. Pipeline for synthetic anomaly generation.

## D. NC-Guided Regularization

In task-free continual learning, models must adapt to nonstationary streams without degrading previously learned decision structures. We therefore propose an NC-guided geometric regularization that enforces inter-class alignment and intraclass compactness in the embedding space. This regularization stabilizes the embedding geometry and mitigates catastrophic forgetting and representation drift.

Inter-Class Alignment Regularization: NC theory indicates that, in the terminal phase of training, class means converge to a simplex ETF. Inspired by this property, we introduce a set of fixed directional priors in the embedding space:

$$
\mathbf { W } ^ { * } = \left[ \mathbf { w } _ { 0 } ^ { * \top } \right] \in \mathbb { R } ^ { 2 \times d } , \quad \| \mathbf { w } _ { k } ^ { * } \| _ { 2 } = 1 .\tag{5}
$$

where $\mathbf { w } _ { k } ^ { * }$ denotes the target alignment direction of class $k \in$ $\{ 0 , 1 \}$ , obtained from the ETF prototype $\mathcal { P } ~ = ~ \{ p _ { 0 } , p _ { 1 } \}$ in Eq. (1). Let $S \subseteq \{ 0 , 1 \}$ denote the set of classes appearing in the current batch. For any $k \in S$ , the batch mean embedding is computed as

$$
\mathbf { u } _ { k } = \frac { 1 } { | \boldsymbol { \mathcal { B } } _ { k } | } \sum _ { i \in \boldsymbol { \mathcal { B } } _ { k } } \mathbf { z } _ { i } ,\tag{6}
$$

where $\boldsymbol { B } _ { k }$ is the index set of samples belonging to class k. The similarity between the class mean and directional prior is defined as

$$
u _ { k  l } = \mathbf { u } _ { k } ^ { \top } \mathbf { w } _ { l } ^ { * } , \quad k , l \in S .\tag{7}
$$

Based on the similarity matrix, the inter-class alignment loss is formulated as a row-wise softmax cross-entropy:

$$
L _ { \mathrm { i n t e r } } = \frac { 1 } { | S | } \sum _ { k \in S } \left( - \log \frac { \exp ( u _ { k \to k } ) } { \sum _ { l \in S } \exp ( u _ { k \to l } ) } \right) .\tag{8}
$$

This loss encourages each class mean to align with its corresponding prior direction. When $| S | \ < \ 2$ or the prior is unavailable, this term is omitted to ensure optimization stability.

Intra-Class Compactness Regularization: Beyond interclass alignment, NC also reveals a trend of within-class feature aggregation and variance collapse. To enhance intra-class consistency and improve robustness to distribution shift and noise, we introduce an intra-class compactness constraint.

The pairwise similarity between two distinct samples within the same class is defined as

$$
d _ { i j } ^ { ( k ) } = z _ { i } ^ { \top } z _ { j } , \quad i \ne j , i , j \in \mathcal { B } _ { k } .\tag{9}
$$

For each sample $z _ { i }$ , similarities to all other samples from the same class are aggregated as

$$
a _ { i } = - \log \left( \frac { 1 } { n _ { k } - 1 } \sum _ { j \neq i } \exp { d _ { i j } ^ { ( k ) } } \right) .\tag{10}
$$

Based on this aggregation, the intra-class compactness loss is given by

$$
L _ { \mathrm { i n t r a } } = \sum _ { k \in S } { \frac { 1 } { n _ { k } ^ { \zeta } } } { \frac { 1 } { n _ { k } } } \sum _ { i \in \mathcal { B } _ { k } } a _ { i } .\tag{11}
$$

Here, $\zeta = 1$ controls inverse-frequency re-weighting to mitigate within-batch class imbalance and increase sensitivity to minority (anomalous) classes. This constraint increases the overall similarity among samples from the same class, encouraging tighter intra-class clusters and suppressing representation scattering caused by drift and noise.

Regularization Objective: The overall NC-guided regularization objective is defined as

$$
L _ { \mathrm { { r e g } } } = \lambda _ { \mathrm { { i n t e r } } } L _ { \mathrm { { i n t e r } } } + \lambda _ { \mathrm { { i n t r a } } } L _ { \mathrm { { i n t r a } } } ,\tag{12}
$$

where $\lambda _ { \mathrm { { i n t e r } } }$ and $\lambda _ { \mathrm { i n t r a } }$ balance the contributions of the two terms. The ETF directions are used as a fixed geometric inductive bias for organizing the streaming embeddings, and the formulation does not require the embeddings to attain an exact neural-collapse configuration.

## E. Focal NC Contrastive Loss

In task-free continual learning, some samples may still remain close to the decision boundary or deviate substantially from their target prototypes. To address this issue, we introduce a Focal Neural Collapse Contrastive loss (FNCC Loss), which enhances sample-level discriminability via contrastive learning while emphasizing hard-to-align samples within the NC-induced geometry.

Similarity Measurement and Unified Partition Function: We first define similarity metrics for both sample–sample and sample–prototype relations:

$$
s _ { i j } = \frac { \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { j } } { \tau } , \quad i , j \in B , i \neq j ,\tag{13}
$$

$$
t _ { i k } = \frac { \mathbf { z } _ { i } ^ { \top } p _ { k } } { \tau } , \quad i \in \boldsymbol { B } , k \in \boldsymbol { S } ,\tag{14}
$$

where $\tau > 0$ is the temperature parameter and $p _ { k }$ denotes the prototype of class k. To unify both relations within a single normalized probability space, the log partition function for sample i is given by

$$
G _ { i } = \left\{ \begin{array} { l l } { \displaystyle { \sum _ { j = 1 } ^ { n } \exp { s _ { i j } } + \sum _ { k = 1 } ^ { | \mathcal { P } | } \exp { t _ { i k } } } , } & { \mathcal { P } \neq \emptyset , } \\ { \displaystyle { \sum _ { j = 1 } ^ { n } \exp { s _ { i j } } } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{15}
$$

Here, $\mathcal { P }$ represents the set of class prototypes involved in the current batch; when no prototype is available, $\mathcal { P } = \emptyset$ . The conditional probability of sample j given anchor sample i is then defined as

$$
P ( j \mid i ) = \exp \left( s _ { i j } - \log G _ { i } \right) .\tag{16}
$$

Positive-Pair Term: Positive pairs are defined as samples belonging to the same class but not identical: $\Omega ^ { + } = \{ ( i , j ) ~ \}$ $y _ { i } ~ = ~ y _ { j } , i ~ \ne ~ j \}$ . To focus optimization on hard positive pairs, we introduce a focal modulation. The sample-pair focal contrastive loss is formulated as

$$
L _ { \mathrm { p a i r } } = - \frac { 1 } { | \Omega ^ { + } | } \sum _ { ( i , j ) \in \Omega ^ { + } } ( 1 - P ( j  { | } i ) ) ^ { \gamma } \log \left( \operatorname* { m a x } ( P ( j  { | } i ) , \varepsilon ) \right) ,\tag{17}
$$

where $\gamma \ \geq \ 0$ is the focusing parameter and $\varepsilon ~ = ~ 1 0 ^ { - 8 }$ ensures numerical stability. If $| \Omega ^ { + } | = 0$ , we set $L _ { \mathrm { p a i r } } = 0$ This term assigns higher weights to low-confidence positive pairs, adaptively emphasizing samples that lie near decision boundaries or are affected by distribution shift.

Prototype-Assignment Term: Let $\pi ( \cdot )$ denote the mapping from class labels to prototypes. For a sample i with $\pi ( y _ { i } ) =$ $p _ { k }$ , the probability of being assigned to its correct prototype is

$$
R ( i ) = \exp \left( t _ { i , \pi ( y _ { i } ) } - \log G _ { i } \right) .\tag{18}
$$

Based on this probability, the prototype-guided focal term is given by

$$
L _ { \mathrm { p r o t o } } = - \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \left( 1 - R ( i ) \right) ^ { \gamma } \log R ( i ) ,\tag{19}
$$

where $\mathcal { H } = \{ i \in B \ | \ \pi ( y _ { i } ) \neq \emptyset \}$ . If H is empty, we set $L _ { \mathrm { p r o t o } } = 0$ . This term strengthens sample–prototype alignment and encourages clearer prototype-centered cluster structures.

Final Contrastive Objective: The proposed Focal NC Contrastive Loss is defined as

$$
L _ { \mathrm { F N C C } } = L _ { \mathrm { p a i r } } + L _ { \mathrm { p r o t o } } .\tag{20}
$$

By re-weighting probabilities via the focal factor $\gamma ,$ FNCC adaptively emphasizes hard positive pairs and ambiguous prototype assignments. This design enhances the discriminability and robustness of the embedding space under continual distribution shifts.

## F. Anomaly Localization

Normal Patch Prototype Bank: The localization branch is used only for pixel-level heatmap generation and does not change the image-level training or evaluation pathway. To obtain a more precise localization signal than attention alone, we construct a compact normal patch prototype bank using only normal samples from the training stream. Specifically, for a normal training image $x ,$ the frozen backbone extracts patch tokens $\{ \mathbf { r } _ { i } \} _ { i = 1 } ^ { N }$ from the last Transformer block, where $N$ is the number of spatial patches. The patch tokens are ℓ -normalized and summarized into a compact set of normal patch prototypes:

$$
\begin{array} { r } { \mathcal { C } = \{ \mathbf { c } _ { m } \} _ { m = 1 } ^ { N _ { c } } , \qquad \| \mathbf { c } _ { m } \| _ { 2 } = 1 , } \end{array}\tag{21}
$$

where $N _ { c }$ denotes the number of normal patch prototypes. The prototype bank C is constructed by online sampling followed by lightweight k-means clustering over normal patch tokens. Only normal training patches are used in this process; synthetic anomalies, test images, and pixel-level masks are excluded from prototype construction. This design preserves the normalonly supervision setting of industrial anomaly detection while remaining compatible with the task-free continual learning protocol.

Prototype-Calibrated Patch Deviation Map: For a test image, each normalized patch token $\mathbf { r } _ { i }$ is compared with the normal patch prototype bank. The patch-level deviation score is defined as the cosine distance to the nearest normal prototype:

$$
d _ { i } = 1 - \operatorname* { m a x } _ { 1 \leq m \leq N _ { c } } \mathbf { r } _ { i } ^ { \top } \mathbf { c } _ { m } .\tag{22}
$$

Rather than applying per-image min–max normalization, which may amplify small score variations and yield spuriously high responses even for normal images, we calibrate patchlevel deviations using statistics estimated from normal training data. Specifically, after constructing the prototype bank ${ \mathcal { C } } ,$ we compute the nearest-prototype cosine distance for every normal training patch using Eq. (22). Let $q _ { 9 5 }$ denote the 95th percentile of these normal nearest-prototype distances over the training stream. A test-patch distance $d _ { i }$ is then calibrated relative to this normal reference distribution using a localization temperature τ<sub>l</sub>:

$$
{ \bf S } _ { i } ^ { \mathrm { p r o t o } } = \sigma \left( \frac { d _ { i } - q _ { 9 5 } } { \pi } \right) .\tag{23}
$$

The patch responses are rearranged into a spatial grid and bilinearly upsampled to the input resolution, yielding $\mathbf { S } ^ { \mathrm { p r o t o } } \in [ 0 , \dot { 1 } ] ^ { H \times }$ . This training-distribution calibration improves pixel-level ranking stability under severe foreground– background imbalance.

Attention-Prior Hybrid Fusion: Although prototype deviations provide discriminative local anomaly evidence, they may remain sensitive to benign appearance variations. We therefore retain backbone self-attention as a weak spatial prior. CLS-to-patch attention is extracted from the frozen backbone, fused across attention heads, rearranged into a spatial map, and upsampled to obtain $\mathbf { A } \in [ 0 , 1 ] ^ { H \times W }$ . The resulting highconfidence anomaly map is defined as

$$
{ \bf H } ^ { \mathrm { p r e c } } = { \bf S } ^ { \mathrm { p r o t o } } \odot ( \epsilon + { \bf A } ) ^ { \gamma _ { l } } ,\tag{24}
$$

where ϵ prevents excessive suppression and $\gamma _ { l }$ controls the strength of the attention prior. To extend spatial support while limiting the influence of smoothing on high-confidence anomaly responses, we further construct a low-confidence support map

$$
\mathbf { H } ^ { \mathrm { c o v } } = \mathrm { A v g P o o l } ( \mathbf { H } ^ { \mathrm { p r e c } } ) \odot \mathbf { A } ,\tag{25}
$$

and obtain the final localization map as

$$
{ \bf H } = \mathrm { m a x } ( { \bf H } ^ { \mathrm { p r e c } } , \rho { \bf H } ^ { \mathrm { c o v } } ) ,\tag{26}
$$

where $\rho \in ( 0 , 1 )$ assigns lower confidence to the expanded support regions. This dual-map fusion preserves highconfidence anomaly responses while extending spatial support for region-level localization.

Connected-Region Filtering: Finally, isolated anomaly responses are further suppressed using Top-k connected-region filtering. For each image, high- and low-response thresholds are defined as

$$
\theta _ { h } = \rho _ { h } \operatorname* { m a x } ( \mathbf { H } ) , \quad \theta _ { l } = \rho _ { l } \operatorname* { m a x } ( \mathbf { H } ) , \quad 0 < \rho _ { l } < \rho _ { h } < 1 .\tag{27}
$$

Connected components above $\theta _ { h }$ are used as confident seeds, and low-threshold components are retained only when they overlap with these seeds. Responses outside the retained support are attenuated by η. This post-processing operates exclusively on the pixel-level heatmap and does not affect image-level prediction.

## IV. EXPERIMENTS

## A. Datasets

MVTec AD: MVTec AD [37] is one of the most widely used benchmarks for industrial anomaly detection. It contains 15 categories in total, including 10 object categories and 5 texture categories. The dataset consists of 4,096 normal images and 1,258 anomalous images. These anomalous samples cover 73 types of realistic defects, such as scratches, dents, structural deformations, and surface damages.

VisA: VisA [38] comprises 12 subsets corresponding to 12 industrial object categories. In total, it contains 10,821 images, including 9,621 normal samples and 1,200 anomalous samples. The anomalies span a broad range of defect types, including scratches, dents, stains, cracks, and other surface defects, as well as structural anomalies such as misalignment and missing components.

## B. Evaluation Metrics

To evaluate anomaly detection under the task-free continual learning setting, we follow a unified evaluation protocol at the end of the data stream [7], [18]. Specifically, Image-level Area Under the Receiver Operating Characteristic curve (I-AUROC) is used to assess the ability to distinguish between normal and anomalous images, while Pixel-level AUROC (P-AUROC) is used to evaluate anomaly localization performance. Given the severe class imbalance in industrial scenarios, AUROC alone can be insufficient, so we additionally report image-level Average Precision (I-AP) and pixel-level Average Precision (P-AP) to more accurately assess detection and localization under highly imbalanced conditions. For pixel-level localization, we further report the Area Under the Per-Region Overlap curve (P-AUPRO), which measures the overlap between predicted anomaly regions and individual ground-truth defect regions across different thresholds, providing a complementary assessment of region-level localization quality.

## C. Implementation Details

We adopt a DINO ViT-S/8 pretrained on ImageNet [39], [40] as the frozen backbone and resize all images to 224×224. For image-level anomaly detection, only the final LayerNorm output of the [CLS] token is used as the backbone representation. The 384-dimensional feature is passed through a biasfree linear projection layer (384 → 384) without activation, BatchNorm, or Dropout, introducing 147,456 trainable parameters. The projected feature is ℓ -normalized before NC-guided optimization and classification with fixed ETF prototypes. Training follows the Online Task-Free Continual Learning (OTFCL) protocol, where each sample is accessed once. We use a batch size of 64 and Adam optimizer with an initial learning rate of $2 \times 1 0 ^ { - 5 }$ and weight decay of $5 \times 1 0 ^ { - 6 }$ . The loss weights are set to $\gamma = 2 . 0 , \lambda _ { \mathrm { { i n t e r } } } = 0 . 9 ,$ and $\lambda _ { \mathrm { { i n t r a } } } = 0 . 5$

The localization branch is independent of the image-level linear projection layer and introduces no trainable parameters. Normalized patch tokens from the frozen backbone are summarized into a normal patch prototype bank with $N _ { c } = 6 4$ prototypes using 5,000 normal patch tokens. Pixel-level scores are computed by nearest-prototype cosine distance and calibrated with the 95-th percentile normal distance $( \tau _ { l } ~ = ~ 0 . 0 2 )$ . The attention-prior fusion uses $\epsilon = 0 . 2 , \gamma _ { l } = 0 . 5 , \rho = 0 . 3 5$ , and a $5 \times 5$ average-pooling window. Connected-region filtering uses $\rho _ { h } = 0 . 7 , \rho _ { l } = 0 . 2 , k = 4 .$ , and $\eta = 0 . 2 .$ . The localization branch only affects pixel-level heatmaps, while image-level scores are obtained exclusively from the [CLS]-based ETF classifier.

For stream construction, samples are shuffled within each category and arranged into category-contiguous segments according to a fixed random category order. Consecutive segments are directly concatenated to create abrupt distribution shifts, while category labels remain unavailable to the model during training. Specifically, the predefined category order for MVTec AD is Tile, Transistor, Toothbrush, Leather, Carpet, Wood, Pill, Hazelnut, Capsule, Cable, Bottle, Zipper, Grid, Screw, and Metal Nut. For VisA, the order is Chewing Gum, Macaroni1, Macaroni2, PCB4, Cashew, Capsules, PCB2, Candle, Pipe Fryum, Fryum, PCB3, and PCB1. All methods are evaluated using the same predefined category orders. Experiments are implemented in PyTorch on a single NVIDIA RTX 4090 GPU.

## D. Baselines

We compare NC-TFAD with representative methods from both continual learning and industrial anomaly detection to provide a comprehensive evaluation under the considered taskfree continual anomaly detection setting. The continual learning baselines include ER [41], DualPrompt [42], L2P [43],

MVP [24], FCA [26], DCM [22], DYSON [33], and Online-LoRA [25], which represent a range of replay-based, promptbased, geometry-guided, dynamic-memory, and parameterefficient continual learning strategies. The industrial anomaly detection baselines include UniAD [7], UCAD [18], IUF [19], CDAD [20], and IB-IUMAD [45], covering representative unified and continual anomaly detection frameworks. Together, these methods enable evaluation against both general continual learning strategies and approaches specifically designed for industrial anomaly detection.

Recent vision–language-based anomaly detection methods, such as WinCLIP [44] and its variants, primarily operate under zero-shot or few-shot protocols, where the target domain requires no continual training or only a small number of reference samples. This setting differs from OTFCL, in which normal samples from evolving target-domain distributions are processed sequentially for single-pass continual updates without task boundaries or category identifiers. Owing to this protocol difference, vision–language-based methods are considered related approaches rather than direct quantitative baselines under OTFCL. For ER, the replay memory is fixed at 500 samples. All applicable methods are evaluated under the same OTFCL stream and evaluation protocol.

## E. Experimental Results

Detection Results on the MVTec AD Dataset: As shown in Table I, NC-TFAD achieves the best average image-level detection performance on MVTec AD and maintains strong results across diverse object and texture categories. Its advantage is consistently observed over both general continual learning methods and approaches specifically developed for industrial anomaly detection, indicating that directly adapting either paradigm is insufficient to handle the normal-only, taskfree, and non-stationary setting considered in this work.

DYSON is an NC-based method originally developed for online multiclass classification, where observed semantic labels directly define class-dependent geometric structures. Under the task-free anomaly detection setting considered here, the training stream contains only normal samples and therefore provides no observed anomalous counterpart for constructing a normal–anomaly geometry. Introducing synthetic anomalies enables DYSON\* to operate under a two-class training formulation, but its original optimization remains designed for online multiclass classification. In contrast, NC-TFAD constructs a category-agnostic normal–anomaly ETF reference and jointly incorporates class-mean alignment, intra-class compactness, and FNCC-based sample–sample and sample–prototype optimization. As shown in Table I, NC-TFAD achieves substantially higher average detection performance than DYSON\*, indicating the effectiveness of adapting NC-based geometry and continual optimization specifically to industrial anomaly detection under non-stationary streams.

Detection Results on the VisA Dataset: The image-level detection results on the VisA dataset are summarized in Table II. Compared with MVTec AD, VisA contains more complex backgrounds and more diverse anomaly patterns, making task-free continual anomaly detection more challenging. NC-TFAD achieves the highest average I-AUROC of

TABLE I  
IMAGE-LEVEL ANOMALY DETECTION PERFORMANCE (%, I-AUROC / I-AP) ON MVTEC AD UNDER THE OTFCL SETTING.
<table><tr><td>Method</td><td>Bottle</td><td>Cable</td><td>Capsule</td><td>Carpet</td><td>Grid</td><td>Hazelnut</td><td>Leather</td><td>Metal_nut</td><td>Pill</td><td>Screw</td><td>Tile</td><td>Toothbrush</td><td>Transistor</td><td>Wood</td><td>Zipper</td><td>Average</td></tr><tr><td>ER</td><td>28.7/70.7</td><td>36.2/58.3</td><td>70.4/92.1</td><td>69.4/90.2</td><td>50.0/76.4</td><td>29.1/52.2</td><td>65.0/86.2</td><td>34.1/73.2</td><td>45.4/85.3</td><td>29.7/61.4</td><td>67.3/86.0</td><td>65.6/85.5</td><td>53.3/45.6</td><td>67.2/86.9</td><td>42.1/77.9</td><td>50.2/76.3</td></tr><tr><td>DualPrompt</td><td>79.2/93.6</td><td>73.2/78.9</td><td>75.8/94.0</td><td>67.4/88.8</td><td>62.7/79.6</td><td>43.9/63.3</td><td>34.9/69.3</td><td>47.2/79.9</td><td>59.0/89.3</td><td>58.3/80.4</td><td>68.8/86.0</td><td>47.2/70.3</td><td>55.0/52.0</td><td>68.3/90.1</td><td>66.5/88.6</td><td>60.5/80.3</td></tr><tr><td>UniAD</td><td>53.9/80.8</td><td>41.9/58.0</td><td>50.6/84.5</td><td>92.7/97.0</td><td>76.4/89.4</td><td>83.3/89.5</td><td>86.6/95.2</td><td>42.5/82.0</td><td>53.4/86.1</td><td>51.9/75.6</td><td>64.8/84.7</td><td>56.1/77.0</td><td>52.4/49.6</td><td>88.7/96.4</td><td>50.8/85.0</td><td>63.1/82.0</td></tr><tr><td>L2P</td><td>89.1/96.8</td><td>59.5/73.3</td><td>43.2/82.0</td><td>56.7/81.8</td><td>56.7/77.2</td><td>67.6/83.3</td><td>38.9/74.9</td><td>59.7/96.6</td><td>50.1/87.3</td><td>48.8/72.4</td><td>73.2/85.3</td><td>28.9/60.3</td><td>55.2/55.1</td><td>56.5/78.1</td><td>67.8/88.7</td><td>56.8/78.9</td></tr><tr><td>MVP</td><td>66.4/87.5</td><td>63.0/72.3</td><td>70.9/92.6</td><td>32.2/71.3</td><td>57.9/80.1</td><td>40.5/63.8</td><td>41.5/76.7</td><td>61.7/89.4</td><td>50.6/88.1</td><td>33.2/64.3</td><td>37.5/67.9</td><td>55.6/78.1</td><td>67.0/61.3</td><td>45.5/78.8</td><td>56.0/85.7</td><td>52.0/77.2</td></tr><tr><td>FCA*</td><td>63.6/86.3</td><td>43.6/43.6</td><td>65.5/87.9</td><td>51.9/79.6</td><td>64.9/81.0</td><td>30.9/52.6</td><td>30.0/66.0</td><td>41.3/77.4</td><td>66.4/91.0</td><td>98.6/99.6</td><td>36.9/67.7</td><td>48.9/74.9</td><td>66.7/65.4</td><td>45.7/80.8</td><td>55.1/83.1</td><td>54.0/77.0</td></tr><tr><td>DCM</td><td>49.2/77.6</td><td>49.8/64.3</td><td>18.6/84.4</td><td>59.4/61.7</td><td>83.1/91.7</td><td>64.5/78.5</td><td>72.0/89.7</td><td>34.0/75.3</td><td>49.9/82.8</td><td>99.9/99.9</td><td>39.7/64.4</td><td>49.4/79.3</td><td>34.3/33.3</td><td>66.8/86.4</td><td>24.7/68.1</td><td>53.0/75.8</td></tr><tr><td></td><td></td><td>50.0/61.3</td><td>50.0/82.6</td><td>50.0/76.1</td><td>50.0/73.1</td><td>50.0/63.6</td><td>50.0/74.2</td><td>50.0/80.9</td><td>50.0/84.4</td><td>50.0/74.4</td><td>50.0/71.8</td><td></td><td></td><td></td><td></td><td>50.0/72.3</td></tr><tr><td>DYSON</td><td>50.0/75.9</td><td>58.6/75.5</td><td>38.7/82.4</td><td>76.8/93.1</td><td>78.6/91.6</td><td>74.2/85.3</td><td>100.0/100.0</td><td>55.4/85.1</td><td>63.6/91.5</td><td>61.0/78.9</td><td>80.6/93.6</td><td>50.0/71.4</td><td>50.0/40.0</td><td>50.0/75.9</td><td>50.0/78.8</td><td>70.1/86.3</td></tr><tr><td>DYSON*</td><td>95.6/98.7</td><td>65.5/78.4</td><td>52.0/84.7</td><td></td><td>54.0/79.5</td><td>32.4/56.3</td><td></td><td></td><td></td><td></td><td></td><td>86.7/95.8</td><td>55.3/48.0</td><td>70.4/88.6</td><td>55.3/86.2</td><td>61.4/81.3</td></tr><tr><td>Online-LoRA</td><td>80.6/94.2</td><td>55.49/59.97</td><td></td><td>83.8/95.1</td><td></td><td>88.75/94.35</td><td>74.8/90.8</td><td>55.1/82.7</td><td>40.9/80.7</td><td>9.2/55.3</td><td>77.9/90.6</td><td>65.6/86.2</td><td>58.3/53.9</td><td>87.4/96.0</td><td>83.3/94.9</td><td></td></tr><tr><td>IUF</td><td>59.60/85.02</td><td>54.16/67.61</td><td>70.08/76.19</td><td>97.07/99.20</td><td>70.76/87.57</td><td></td><td>97.89/99.30</td><td>64.71/78.57</td><td>55.29/86.83</td><td>52.61/72.27</td><td>84.05/93.71</td><td>50.00/77.48</td><td>56.46/43.00</td><td>95.35/98.73</td><td>64.97/89.65</td><td>70.87/82.79</td></tr><tr><td>CDAD</td><td>57.86/83.25</td><td>78.71/83.86</td><td>52.29/82.78</td><td>61.60/80.28</td><td>51.55/77.72</td><td>70.75/78.31</td><td>76.63/90.24</td><td>51.22/82.35</td><td>53.93/86.07</td><td>62.16/67.52</td><td>60.53/80.33</td><td>55.28/80.51</td><td>51.46/43.22</td><td>86.23/94.65</td><td>60.27/87.01</td><td>60.39/78.79</td></tr><tr><td>UCAD</td><td>65.40/88.22</td><td></td><td>52.05/85.13</td><td>94.62/98.41</td><td>79.28/92.36</td><td>91.29/94.21</td><td>99.66/99.88</td><td>50.10/82.32</td><td>53.22/88.99</td><td>56.45/80.64</td><td>79.69/91.03</td><td>73.89/90.57</td><td>50.75/49.29</td><td>92.28/97.00</td><td>63.71/88.75</td><td>72.07/87.38</td></tr><tr><td>IB-IUMAD</td><td>58.97/84.98</td><td>56.54/59.23</td><td>68.45/76.84</td><td>97.27/99.24</td><td>69.09/87.32</td><td>87.39/93.67</td><td>98.27/99.43</td><td>66.62/77.66</td><td>55.67/87.28</td><td>51.44/73.10</td><td>86.83/94.91</td><td>50.56/77.94</td><td>58.67/42.24</td><td>95.35/98.73</td><td>66.44/90.10</td><td>71.17/82.84</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>78.8/94.7</td><td>84.6/96.9</td><td>68.1/87.7</td><td>96.1/98.7</td><td>91.1/96.8</td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td></td><td>65.8/77.2</td><td>69.4/92.7</td><td></td><td></td><td>93.3/96.7</td><td>100.0/100.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>88.4/96.9</td><td>87.8/95.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>99.8/99.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>82.1/91.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>57.9/56.8</td><td>81.3/94.7</td><td>69.6/90.9</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></table>

<sup>∗</sup>FCA\* and DYSON\* are NC-based continual learning methods; synthetic anomaly samples were used to meet their two-class training requirement.

TABLE II  
IMAGE-LEVEL ANOMALY DETECTION PERFORMANCE (%, I-AUROC / I-AP) ON VISA UNDER THE OTFCL SETTING.
<table><tr><td>Method</td><td>Candle</td><td>Capsules</td><td>Cashew</td><td>Chewinggum</td><td>Fryum</td><td>Macaroni1</td><td>Macaroni2</td><td>Pcb1</td><td>Pcb2</td><td>Pcb3</td><td>Pcb4</td><td>Pipe_fryum</td><td>Average</td></tr><tr><td>ER</td><td>52.1/9.4</td><td>64.5/24.0</td><td>57.0/22.9</td><td>42.2/14.4</td><td>77.1/48.6</td><td>52.6/12.0</td><td>53.4/10.8</td><td>67.9/18.1</td><td>71.5/18.6</td><td>63.4/21.7</td><td>48.9/8.5</td><td>53.5/24.1</td><td>58.7/19.4</td></tr><tr><td>DualPrompt</td><td>61.8/12.6</td><td>48.3/14.3</td><td>55.1/24.9</td><td>74.6/58.4</td><td>80.8/49.5</td><td>50.7/11.9</td><td>63.3/16.0</td><td>57.9/13.5</td><td>52.0/14.3</td><td>46.3/9.2</td><td>70.7/25.6</td><td>41.9/18.7</td><td>58.6/21.7</td></tr><tr><td>UniAD</td><td>26.7/5.8</td><td>56.9/18.7</td><td>40.4/13.7</td><td>46.7/15.9</td><td>48.2/17.3</td><td>47.3/8.6</td><td>41.2/7.5</td><td>41.6/7.3</td><td>45.4/8.3</td><td>36.3/6.8</td><td>64.2/19.4</td><td>38.7/15.5</td><td>44.5/12.1</td></tr><tr><td>L2P</td><td>79.3/39.8</td><td>57.0/21.9</td><td>62.4/30.7</td><td>83.1/64.9</td><td>65.0/30.5</td><td>75.5/25.8</td><td>58.7/13.0</td><td>50.9/13.9</td><td>56.8/26.0</td><td>43.5/8.1</td><td>71.8/21.5</td><td>40.0/16.6</td><td>62.0/26.0</td></tr><tr><td>MVP</td><td>83.9/39.2</td><td>60.4/19.5</td><td>28.2/14.5</td><td>76.0/49.8</td><td>64.4/29.4</td><td>67.5/22.2</td><td>58.6/33.9</td><td>80.4/33.9</td><td>55.4/14.8</td><td>43.3/7.6</td><td>63.0/24.6</td><td>68.1/35.8</td><td>62.4/25.2</td></tr><tr><td>FCA*</td><td>18.5/5.3</td><td>30.1/9.7</td><td>82.7/57.2</td><td>29.6/17.8</td><td>27.3/11.0</td><td>80.7/31.9</td><td>43.9/7.7</td><td>46.6/8.9</td><td>78.1/56.4</td><td>61.3/16.6</td><td>18.6/5.3</td><td>56.3/24.4</td><td>50.4/22.0</td></tr><tr><td>DCM</td><td>73.0/27.0</td><td>51.2/18.5</td><td>90.9/71.3</td><td>60.3/22.7</td><td>21.2/10.2</td><td>19.0/5.3</td><td>67.5/16.7</td><td>69.7/19.9</td><td>79.7/49.0</td><td>38.2/7.0</td><td>12.1/5.0</td><td>34.2/12.6</td><td>51.4/22.1</td></tr><tr><td>DYSON</td><td>50.0/9.1</td><td>50.0/14.3</td><td>50.0/16.7</td><td>50.0/16.6</td><td>50.0/16.7</td><td>50.0/9.1</td><td>50.0/9.1</td><td>50.0/9.1</td><td>50.0/9.1</td><td>50.0/9.0</td><td>50.0/9.1</td><td>50.0/16.7</td><td>50.0/12.0</td></tr><tr><td>DYSON*</td><td>69.9/23.3</td><td>59.8/20.3</td><td>87.3/74.0</td><td>91.5/88.0</td><td>74.5/56.6</td><td>61.3/20.2</td><td>69.1/21.5</td><td>58.9/18.4</td><td>55.6/11.7</td><td>66.0/17.9</td><td>83.9/49.1</td><td>91.0/80.0</td><td>72.4/40.1</td></tr><tr><td>Online-LoRA</td><td>54.3/11.9</td><td>31.8/9.8</td><td>66.7/34.1</td><td>25.4/11.3</td><td>62.6/32.6</td><td>73.6/21.3</td><td>55.8/18.0</td><td>80.9/15.9</td><td>64.5/16.5</td><td>57.7/7.8</td><td>43.8/15.9</td><td>39.5/15.9</td><td>54.7/18.4</td></tr><tr><td>IUF</td><td>69.46/40.92</td><td>54.37/64.33</td><td>60.84/60.12</td><td>58.28/60.91</td><td>51.82/67.45</td><td>50.69/51.31</td><td>58.42/45.37</td><td>62.27/43.17</td><td>53.73/50.14</td><td>50.24/48.38</td><td>51.79/51.42</td><td>60.10/76.85</td><td>56.83/55.03</td></tr><tr><td>CDAD</td><td>63.96/67.76</td><td>57.45/59.20</td><td>64.66/77.39</td><td>51.38/66.08</td><td>55.56/73.78</td><td>54.83/50.40</td><td>52.30/52.77</td><td>59.08/58.59</td><td>57.96/49.30</td><td>57.04/60.49</td><td>52.25/54.62</td><td>73.90/84.81</td><td>58.36/62.93</td></tr><tr><td>UCAD</td><td>57.44/58.29</td><td>67.53/75.64</td><td>51.50/67.51</td><td>61.98/78.35</td><td>54.74/75.84</td><td>42.01/53.93</td><td>58.32/61.96</td><td>63.10/66.15</td><td>51.67/54.88</td><td>52.36/54.93</td><td>48.32/50.35</td><td>47.70/70.95</td><td>54.72/64.06</td></tr><tr><td>IB-IUMAD</td><td>67.61/42.01</td><td>56.08/66.73</td><td>55.80/63.25</td><td>57.86/62.06</td><td>50.66/68.26</td><td>55.11/54.74</td><td>58.56/44.81</td><td>54.99/47.18</td><td>51.40/48.48</td><td>52.09/50.91</td><td>60.35/60.77</td><td>60.64/76.77</td><td>56.76/57.16</td></tr><tr><td>Ours</td><td>75.6/35.0</td><td>62.4/22.4</td><td>85.7/71.4</td><td>90.2/86.6</td><td>81.1/67.8</td><td>67.4/23.3</td><td>74.2/27.9</td><td>85.4/37.2</td><td>66.4/18.0</td><td>66.4/21.0</td><td>84.7/51.5</td><td>92.6/81.1</td><td>77.7/45.3</td></tr></table>

<sup>∗</sup>FCA<sup>∗</sup> and DYSON<sup>∗</sup> are NC-based continual learning methods; synthetic anomaly samples were used to meet their two-class training requirement.

77.7%, outperforming the second-best DYSON\* by 5.3 percentage points, and shows strong performance across several challenging categories, including Fryum, Macaroni2, Pcb1, Pcb3, Pcb4, and Pipe fryum. However, its average I-AP of 45.3% is lower than that of several industrial anomaly detection baselines, with UCAD achieving the highest average I-AP of 64.06%. This discrepancy is likely related to the severe class imbalance in VisA: while AUROC is relatively insensitive to class prevalence and primarily reflects overall ranking discrimination, AP is more sensitive to false positives among high-confidence predictions. Therefore, the VisA results demonstrate that NC-TFAD provides strong overall normal–anomaly discrimination, but its precision–recall behavior under highly imbalanced streams remains a limitation. Improving anomaly-score calibration and suppressing highconfidence false positives constitute promising directions for future work.

Localization Results: Most task-free and incremental learning methods considered in the image-level evaluation are designed for image-level prediction and do not natively produce dense anomaly maps. Introducing additional localization modules would alter their original architectures and make the resulting comparison dependent on the specific localization design. Therefore, the pixel-level evaluation is conducted on methods that inherently support anomaly localization, including UniAD, IUF, UCAD, CDAD, and IB-IUMAD.

Table III reports the pixel-level localization performance on MVTec AD and VisA in terms of P-AUROC, P-AUPRO, and P-AP. On MVTec AD, NC-TFAD achieves the best average performance across all three metrics, reaching 87.54% P-AUROC, 73.49% P-AUPRO, and 34.93% P-AP. These results demonstrate that the proposed prototype-calibrated localization branch effectively identifies anomalous pixels while preserving coherent defect regions. On VisA, NC-TFAD further achieves the highest P-AUROC and P-AUPRO, with 88.91% and 69.83%, respectively, demonstrating strong pixel-level discrimination and region-level localization under more complex anomaly patterns. Its P-AP remains comparatively low, which is mainly attributable to the small and sparse defects in VisA, where a limited number of false-positive pixels can substantially affect precision under severe pixel-level class imbalance. This limitation and potential improvements in finegrained anomaly-score calibration are further discussed in the Limitations section.

The qualitative results in Fig. 3 are consistent with these quantitative observations. The prototype-calibrated deviation responses are generally concentrated around defect-related regions, while the attention prior and connected-region filtering reduce isolated background responses and preserve spatially coherent anomaly regions. Nevertheless, localization remains challenging for small or weakly contrasted defects, particularly on VisA. These results suggest that pixel-level localization remains a challenging auxiliary task under the OTFCL setting.

## F. Ablation Study

Module Effectiveness: Ablation results for different module configurations are presented in Table IV. When both NC regularization and FNCC loss are disabled, the model performance is substantially limited. This effect is particularly pronounced on the VisA dataset, indicating that basic feature learning alone is insufficient to handle complex distribution shifts in a taskfree data stream.

![](images/178b9cdeed1ca452f423e447e868d0a7e5cb133e284615086b4abf9802c19551.jpg)  
Fig. 3. Qualitative visualizations of anomaly localization on MVTec AD and VisA. The first six columns show results on MVTec AD, and the last six columns show results on VisA.

TABLE III  
PIXEL-LEVEL ANOMALY LOCALIZATION PERFORMANCE (%) ON MVTEC AD AND VISA UNDER THE CONTINUAL LEARNING SETTING. RESULTS ARE REPORTED IN TERMS OF P-AUROC, P-AUPRO, AND P-AP.
<table><tr><td rowspan="2">Method</td><td colspan="3">MVTec AD</td><td colspan="3">VisA</td></tr><tr><td>P-AUROC ↑</td><td>P-AUPRO ↑</td><td>P-AP ↑</td><td>P-AUROC ↑</td><td>P-AUPRO ↑</td><td>P-AP ↑</td></tr><tr><td>IUF</td><td>77.38</td><td>55.02</td><td>14.76</td><td>85.35</td><td>53.30</td><td>3.34</td></tr><tr><td>CDAD</td><td>75.53</td><td>47.97</td><td>8.79</td><td>85.98</td><td>56.04</td><td>5.54</td></tr><tr><td>UCAD</td><td>73.56</td><td>51.92</td><td>22.38</td><td>42.59</td><td>6.16</td><td>1.13</td></tr><tr><td>IB-IUMAD</td><td>77.45</td><td>56.04</td><td>14.78</td><td>85.27</td><td>53.62</td><td>3.49</td></tr><tr><td>UniAD</td><td>81.70</td><td>50.70</td><td>17.20</td><td>84.90</td><td>53.80</td><td>0.60</td></tr><tr><td>NC-TFAD</td><td>87.54</td><td>73.49</td><td>34.93</td><td>88.91</td><td>69.83</td><td>2.85</td></tr></table>

When only NC-guided regularization is applied, the performance on MVTec AD improves noticeably. Specifically, I-AUROC increases from 75.5% to 81.2%, while I-AP rises to 91.5%. This result confirms that NC-based regularization effectively alleviates representation drift and reinforces the geometric decision structure. When only the FNCC loss is introduced, performance gains on VisA are more prominent, highlighting the importance of hard-sample-aware contrastive optimization in challenging continual settings.

TABLE IV  
ABLATION STUDY OF MODEL COMPONENTS
<table><tr><td rowspan="2">Regularizer</td><td rowspan="2">FNCC Loss</td><td colspan="2">MvTec</td><td colspan="2">VisA</td></tr><tr><td>I-AUROC</td><td>I-AP</td><td>I-AUROC</td><td>I-AP</td></tr><tr><td>X</td><td>X</td><td>75.5</td><td>88.6</td><td>72.0</td><td>39.7</td></tr><tr><td>√</td><td>×</td><td>81.2</td><td>91.5</td><td>74.0</td><td>41.4</td></tr><tr><td>X</td><td>√</td><td>80.2</td><td>90.7</td><td>77.4</td><td>44.7</td></tr><tr><td>√</td><td>√</td><td>82.1</td><td>91.8</td><td>77.7</td><td>45.3</td></tr></table>

TABLE V  
ABLATION STUDY ON INTER- AND INTRA-CLASS REGULARIZATION
<table><tr><td colspan="2">Regularizer</td><td colspan="2">MVTec</td><td colspan="2">VisA</td></tr><tr><td> $\lambda _ { \mathrm { i n t e r } }$ </td><td> $\lambda _ { \mathrm { i n t r a } }$ </td><td>I-AUROC</td><td>I-AP</td><td>I-AUROC</td><td>I-AP</td></tr><tr><td>0.0</td><td>0.0</td><td>75.5</td><td>88.6</td><td>72.0</td><td>39.7</td></tr><tr><td>0.1</td><td>0.05</td><td>79.2</td><td>89.7</td><td>75.6</td><td>43.2</td></tr><tr><td>0.1</td><td>0.2</td><td>79.3</td><td>87.3</td><td>75.4</td><td>43.2</td></tr><tr><td>0.2</td><td>0.1</td><td>77.6</td><td>89.5</td><td>74.0</td><td>41.4</td></tr><tr><td>0.3</td><td>0.1</td><td>78.4</td><td>89.6</td><td>76.2</td><td>43.9</td></tr><tr><td>0.4</td><td>0.15</td><td>78.1</td><td>87.1</td><td>75.8</td><td>43.2</td></tr><tr><td>0.5</td><td>0.2</td><td>78.9</td><td>90.0</td><td>75.9</td><td>43.1</td></tr><tr><td>0.9</td><td>0.5</td><td>81.2</td><td>91.5</td><td>76.1</td><td>43.9</td></tr></table>

When both components are enabled simultaneously, the model achieves the best overall performance on both datasets. This observation validates the strong complementarity between geometric regularization and focal contrastive learning.

Effect of Inter- and Intra-Class Regularization: The impact of the inter-class and intra-class regularization terms on NC-TFAD is examined in Table V. When both constraints are removed, the model performance drops markedly on both datasets, indicating that the decision boundary becomes unstable under continual distribution shifts.

As the strengths of the inter-class and intra-class regularizers are gradually increased, the performance exhibits a consistent upward trend. On MVTec AD, the model reaches its best performance when the regularization strength is further increased. With $\lambda _ { \mathrm { i n t e r } } ~ = ~ 0 . 9$ and $\lambda _ { \mathrm { { i n t r a } } } ~ = ~ 0 . 5$ , I-AUROC and I-AP are improved to 81.2% and 91.5%, respectively. On the VisA dataset, the best results are obtained in a medium-tohigh regularization regime, where I-AUROC reaches up to 76.2% $( \lambda _ { \mathrm { { i n t e r } } } = 0 . 3 )$ and I-AP 43.9% $( \lambda _ { \mathrm { { i n t e r } } } = 0 . 3 \ \mathrm { { o r } \ 0 . 9 ) }$ Compared with MVTec AD, the more complex backgrounds and diverse anomaly patterns in VisA make the performance more sensitive to the choice of regularization weights.

Sensitivity to the Contrastive Temperature Parameter: Table VI reports the sensitivity analysis of the FNCC loss with respect to the temperature parameter τ . The temperature controls the smoothness of the similarity distribution and thus affects the sharpness of the decision boundaries between samples.

Without NC-guided regularization, the model is highly sensitive to variations in τ. As τ increases from 0.05 to 0.20, the performance on both datasets improves significantly. For example, on VisA, I-AUROC rises from 72.0% to 77.4% and I-AP from 39.7% to 44.7%. However, when τ continues to increase $( \mathbf { e . g . } , \tau \geq 0 . 3 0 )$ , the performance begins to degrade. This behavior indicates that an excessively large temperature weakens discriminative learning on hard samples.

After jointly incorporating the NC regularization and FNCC, the model becomes substantially more stable across a wide range of temperature values. For $0 . 1 \leq \tau \leq 0 . 3 ,$ performance remains consistently high on both datasets and exceeds that of the setting without regularization. Among these configurations, $\tau = 0 . 1$ yields the best results, achieving 82.1% / 91.8% on MVTec AD and 77.7% / 45.3% on VisA in terms of I-AUROC / I-AP.

TABLE VI  
ABLATION STUDY ON THE SENSITIVITY OF THE CONTRASTIVETEMPERATURE τ
<table><tr><td rowspan="2">FNCC loss</td><td rowspan="2">Reg.</td><td rowspan="2">Temp.(τ)</td><td colspan="2">MVTec</td><td colspan="2">VisA</td></tr><tr><td>I-AUROC</td><td>I-AP</td><td>I-AUROC</td><td>I-AP</td></tr><tr><td>√</td><td>X</td><td>0.05</td><td>75.5</td><td>88.6</td><td>72.0</td><td>39.7</td></tr><tr><td>√</td><td>×</td><td>0.1</td><td>80.2</td><td>90.7</td><td>77.1</td><td>44.4</td></tr><tr><td>√</td><td>X</td><td>0.2</td><td>80.4</td><td>90.5</td><td>77.4</td><td>44.7</td></tr><tr><td>√</td><td>X</td><td>0.3</td><td>79.9</td><td>90.4</td><td>76.4</td><td>43.2</td></tr><tr><td>√</td><td>X</td><td>0.5</td><td>79.4</td><td>90.1</td><td>75.9</td><td>42.7</td></tr><tr><td>√</td><td>√</td><td>0.05</td><td>80.8</td><td>90.9</td><td>75.9</td><td>43.3</td></tr><tr><td>√</td><td>√</td><td>0.1</td><td>82.1</td><td>91.8</td><td>77.7</td><td>45.3</td></tr><tr><td>V</td><td>V</td><td>0.2</td><td>82.0</td><td>91.6</td><td>77.2</td><td>44.5</td></tr><tr><td>√</td><td>√</td><td>0.3</td><td>81.8</td><td>91.4</td><td>77.1</td><td>44.5</td></tr><tr><td>√</td><td>√</td><td>0.5</td><td>81.4</td><td>91.2</td><td>76.8</td><td>44.1</td></tr></table>

Effect of Minibatch Composition: To assess sensitivity to minibatch composition, we consider three stream arrangements: Random, which globally shuffles samples; Homogeneous, which groups samples by product category; and Balanced, which uses round-robin sampling to increase category diversity within each batch. All settings share the same samples, batch size, optimization configuration, and singlepass protocol. Category identities are used only to construct this diagnostic ablation and are never provided to NC-TFAD during training or inference.

As shown in Table VII, NC-TFAD is generally robust to different minibatch compositions. The three settings yield comparable results on MVTec AD, while on VisA the Homogeneous setting is consistently weaker than Random and Balanced. This suggests that the shared normal–anomaly geometry does not depend on category-homogeneous batches, although greater cross-category exposure can benefit more composition-sensitive streams.

Temporal Retention Analysis: We compare four variants under the same task-free continual stream: base retains the fixed ETF alignment but excludes NC-guided regularization and FNCC, + Reg. and + FNCC introduce the two objectives individually, and NC-TFAD combines both. Final I-AUROC and Final I-AP evaluate overall image-level detection at the end of the stream, while Old-category I-AUROC evaluates the final model only on categories observed before the last stream segment to measure retained detection capability. Category identities are used only for offline evaluation.

As shown in Table VIII, both NC-guided regularization and FNCC improve final and old-category performance over the base model, while their combination achieves the strongest results across both datasets. This indicates that the proposed objectives improve retention of earlier distributions while maintaining strong end-of-stream detection performance.

Representation Geometry Dynamics: To examine how the representation geometry evolves during continual learning, we save read-only checkpoints at 20%, 40%, 60%, 80%, and 100% of the single-pass stream and evaluate them offline after training. This procedure neither alters optimization nor revisits any training sample. At each checkpoint, we evaluate three quantities shown in Fig. 4. Feature-to-ETF Alignment is the mean cosine similarity between each projected embedding and the fixed ETF prototype associated with its anomaly state. Within-State Scatter measures the cosine-distance dispersion around the corresponding mean embedding, computed separately for each product category and anomaly state and then macro-averaged. Real-Anomaly Scatter uses the same measure but is restricted to real anomalous samples. Product-category identities and real anomaly labels are used only for these offline statistics and are never provided to the model during continual optimization.

TABLE VII  
SENSITIVITY ANALYSIS OF MINIBATCH COMPOSITION ON MVTEC AD AND VISA. RESULTS ARE REPORTED AS MEAN ± STANDARD DEVIATION OVER THREE RANDOM SEEDS.
<table><tr><td rowspan="2">Setting</td><td colspan="4">MVTec AD</td><td colspan="4">VisA</td></tr><tr><td> $\mathrm { I { - } A U R O C \uparrow }$ </td><td>I-AP↑</td><td>P-AUROC↑</td><td>P-AP↑</td><td> $\mathrm { I { - } A U R O C \uparrow }$ </td><td>I-AP↑</td><td>P-AUROC↑</td><td>P-AP↑</td></tr><tr><td>Random</td><td> $\mathbf { 8 0 . 0 1 } 2 \mathbf { 0 . 7 4 }$ </td><td> $9 0 . 5 7 { \pm } 0 . 5 3 $ </td><td> $8 7 . 3 6 { \pm } 0 . 2 7 $ </td><td> $3 0 . 7 7 { \pm } 0 . 1 8$ </td><td> $7 7 . 0 4 { \pm } 0 . 9 2 $ </td><td> $4 7 . 5 2 { \pm } 1 . 3 5 $ </td><td> $\mathbf { 8 9 . 0 2 \pm 0 . 1 7 }$ </td><td> $\mathbf { 2 . 1 4 } \pm \mathbf { 0 . 0 9 }$ </td></tr><tr><td>Homogeneous</td><td> $7 9 . 5 4 \pm 1 . 6 1$ </td><td> $9 0 . 1 1 { \pm } 0 . 9 4$ </td><td> $8 7 . 2 5 { \pm } 0 . 2 1 $ </td><td> $3 0 . 2 7 { \pm } 0 . 0 9$ </td><td> $7 6 . 2 6 { \pm } 1 . 0 5$ </td><td> $4 4 . 1 3 { \pm } 1 . 7 4 $ </td><td> $8 8 . 6 9 { \pm } 0 . 1 8 $ </td><td> $1 . 8 2 { \pm } 0 . 1 5 $ </td></tr><tr><td>Balanced</td><td> $7 9 . 6 2 { \pm } 0 . 9 5 $ </td><td> $\mathbf { 9 0 . 6 9 } \pm \mathbf { 0 . 6 4 }$ </td><td> $\mathbf { 8 7 . 6 0 { \pm 0 . 1 4 } }$ </td><td> $\mathbf { 3 0 . 8 5 \pm 0 . 3 3 }$ </td><td> ${ \bf 7 7 . 8 9 { \pm } 1 . 2 7 }$ </td><td> $\mathbf { 4 8 . 0 3 { \pm } 1 . 7 0 }$ </td><td> $8 8 . 9 3 { \pm } 0 . 1 5 $ </td><td> $2 . 1 2 { \pm } 0 . 1 5$ </td></tr></table>

![](images/e71d775df5d0ebf08ad4b122a90426478ad65a0801c6b783abc11833d593cc80.jpg)

![](images/e61d52f811eb63b78fb8888f04c0748ed109c5fdc29654d211e374461d44aafb.jpg)

![](images/fabf0016e16b2cb29d4c2747d3193bad2debd538033b9265a132a311913ce196.jpg)

![](images/e4efc6b0af53938d179e10c34fd2b5baca39cb6673f3c2b1d421583959ca7881.jpg)

![](images/52172b8b44ebbd8d8dca8ff475669147897e01550f2f730461b882864765bb1f.jpg)

![](images/18d6f8e29868ea9f7d47e63cd04bde700433b7f853e8120248835c0866fe3ee5.jpg)  
Fig. 4. Representation-geometry dynamics over the continual stream. Curves and shaded regions denote the mean and one standard deviation over three random seeds, respectively. Feature-to-ETF Alignment measures the cosine similarity between projected features and their assigned fixed ETF prototypes. Within-State Scatter measures the dispersion of normal and anomalous representations around their corresponding mean embeddings, whereas Real-Anomaly Scatter evaluates the compactness of real anomalous representations only. Lower scatter indicates more compact representations.

TABLE VIII  
TEMPORAL RETENTION ANALYSIS ON MVTEC AD AND VISA. RESULTS ARE REPORTED AS MEAN±STANDARD DEVIATION OVER THREE RANDOM SEEDS.
<table><tr><td>Dataset</td><td>Variant</td><td>Final I-AUROC↑</td><td>Final I-AP↑</td><td>Old-category I-AUROC↑</td></tr><tr><td rowspan="4">MVTec AD</td><td>base</td><td>76.11±0.40</td><td> $8 8 . 9 0 { \pm } 0 . 1 3 $ </td><td>77.41±0.09</td></tr><tr><td>+ Reg.</td><td>79.22±1.92</td><td> $9 0 . 2 5 { \pm } 0 . 8 1 $ </td><td> $8 0 . 2 5 { \pm } 1 . 6 2 $ </td></tr><tr><td> $+ \mathrm { F N C C }$ </td><td>79.12±0.78</td><td> $9 0 . 0 1 { \pm } 0 . 4 4$ </td><td> $7 9 . 3 8 { \pm } 0 . 7 2 $ </td></tr><tr><td>NC-TFAD</td><td>79.76±1.40</td><td>90.34±0.72</td><td>80.60±1.18</td></tr><tr><td rowspan="4">VisA</td><td>base</td><td>71.16±0.52</td><td> $3 8 . 4 1 { \pm } 0 . 7 6 $ </td><td> $7 2 . 2 8 { \pm } 0 . 3 8 $ </td></tr><tr><td>+ Reg.</td><td> $7 4 . 4 2 { \pm } 1 . 2 5 $ </td><td> $4 1 . 7 6 { \pm } 1 . 5 9$ </td><td> $7 4 . 5 8 { \pm } 1 . 4 3 $ </td></tr><tr><td> $+ \mathrm { F N C C }$ </td><td> $7 5 . 6 9 { \pm } 0 . 2 6 $ </td><td> $4 2 . 6 6 { \pm } 0 . 3 5 $ </td><td> $7 5 . 4 4 { \pm } 0 . 4 1$ </td></tr><tr><td> $_ { \mathrm { N C - T F A D } }$ </td><td> $\mathbf { 7 6 . 3 3 { \scriptstyle \pm 0 . 0 8 } }$ </td><td> $\mathbf { 4 3 . 7 0 { \scriptstyle \pm 0 . 2 6 } }$ </td><td> $\mathbf { 7 7 . 0 3 { \pm } 0 . 3 9 }$ </td></tr></table>

As shown in Fig. 4, NC-TFAD consistently strengthens feature-to-ETF alignment while producing more compact representations than the base model, with particularly pronounced improvements on VisA and for real anomalous samples. These trends support the intended role of NC-guided regularization and FNCC in preserving prototype anchoring and representation compactness as the data distribution evolves.

Effect of Different Backbones: To assess the sensitivity of NC-TFAD to the pretrained feature extractor, we evaluate the framework with DINO ViT-S/8, ViT-S/16, ResNet-50, and EfficientNet-B4 under the same OTFCL protocol, with all backbone parameters kept frozen throughout training. As shown in Table IX, NC-TFAD remains effective across both Transformer- and convolution-based architectures. Moreover, comparisons under matched backbone settings consistently show stronger I-AUROC than the corresponding representative baselines. These results demonstrate that the effectiveness of the proposed NC-guided framework is not restricted to the default DINO ViT-S/8 backbone and generalizes across heterogeneous feature extractors.

Effect of Synthetic Anomaly Strategies: To examine the influence of synthetic anomaly construction, we compare several synthesis variants on MVTec AD under the same task-free continual learning protocol and fixed seed. As summarized in Table X, all variants are trained with synthetic anomalies but evaluated exclusively on real anomalous test images.

As shown in Table X, local structural perturbations are more effective than simple appearance or blur transformations, while global perturbations substantially degrade image-level detection. The full strategy consistently provides the strongest overall performance, indicating that combining spatially plausible local regions with diverse perturbation types yields more effective auxiliary anchors for both image-level geometric learning and anomaly localization.

TABLE IX  
EFFECT OF DIFFERENT FROZEN BACKBONES ON NC-TFAD AND ARCHITECTURE-RELATED COMPARISON WITH REPRESENTATIVE METHODS UNDER THE OTFCL PROTOCOL. RESULTS ARE REPORTED AS I-AUROC / I-AP (%).
<table><tr><td>Backbone</td><td>Method</td><td>MVTec AD</td><td>VisA</td></tr><tr><td colspan="4">NC-TFAD with different frozen backbones</td></tr><tr><td>ViT-S/16</td><td>NC-TFAD</td><td>72.18 / 87.04</td><td>63.83 / 32.09</td></tr><tr><td>ResNet-50</td><td>NC-TFAD</td><td>76.45 / 87.81</td><td>63.54 / 28.17</td></tr><tr><td>EfficientNet-B4</td><td>NC-TFAD</td><td>77.74 / 89.53</td><td>69.10 / 35.90</td></tr><tr><td>DINO ViT-S/8</td><td>NC-TFAD</td><td>82.10 / 91.80</td><td>77.70 / 45.30</td></tr><tr><td colspan="4">Architecture-related comparison with representative methods</td></tr><tr><td>DINO ViT-S/8</td><td>DYSON* NC-TFAD</td><td>70.10 / 86.30 82.10 / 91.80</td><td>72.40 / 40.10 77.70 / 45.30</td></tr><tr><td>ViT-S/16</td><td>Online-LoRA</td><td>61.40 / 81.30</td><td>54.70 /  18.40</td></tr><tr><td></td><td>NC-TFAD</td><td>72.18 / 87.04</td><td>63.83 / 32.09</td></tr><tr><td>ResNet-family</td><td>CDAD (ResNet-50)</td><td>60.39 /  78.79</td><td>58.36 / 62.93</td></tr><tr><td></td><td>UCAD (WideResNet-50)</td><td>72.07  / 87.38</td><td>54.72 / 64.07</td></tr><tr><td></td><td>NC-TFAD (ResNet-50)</td><td>76.45 / 87.81</td><td>63.54 /  28.17</td></tr><tr><td></td><td>UniAD</td><td>63.10 / 82.00</td><td>44.50 /  12.10</td></tr><tr><td></td><td>IUF</td><td>70.87  / 82.79</td><td>56.83 / 55.03</td></tr><tr><td>EfficientNet-B4</td><td></td><td>71.17 / 82.85</td><td>56.76 / 57.16</td></tr><tr><td></td><td>IB-IUMAD NC-TFAD</td><td>77.74 / 89.53</td><td>69.10 / 35.90</td></tr></table>

<sup>∗</sup>DYSON<sup>∗</sup> denotes the variant trained with synthetic anomaly samples to satisfy its two-class training requirement. CDAD, UCAD, IUF, and IB-IUMAD are reproduced under the same OTFCL stream used in the SOTA comparison.

TABLE X  
ABLATION STUDY OF SYNTHETIC ANOMALY STRATEGIES ON MVTEC AD. ALL VARIANTS FOLLOW THE SAME TASK-FREE CONTINUAL LEARNING PROTOCOL AND ARE EVALUATED ON REAL ANOMALOUS TEST IMAGES. THE FULL STRATEGY DENOTES THE FOREGROUND-CONSTRAINED LOCAL MIXED SYNTHESIS USED IN NC-TFAD.
<table><tr><td>Synthetic Strategy</td><td>I-AUROC</td><td>I-AP</td><td>P-AUROC</td><td>P-AP</td><td>P-AUPRO</td></tr><tr><td>Foreground + appearance only</td><td>63.77</td><td>83.96</td><td>87.07</td><td>28.57</td><td>70.96</td></tr><tr><td>Foreground + local blur only</td><td>56.61</td><td>80.16</td><td>87.31</td><td>28.62</td><td>70.45</td></tr><tr><td>Foreground + local rearrangement only</td><td>79.18</td><td>90.43</td><td>86.95</td><td>29.29</td><td>72.20</td></tr><tr><td>Global perturbation + mixed</td><td>53.01</td><td>76.75</td><td>86.45</td><td>27.30</td><td>70.70</td></tr><tr><td>Random local mask + mixed</td><td>77.25</td><td>89.82</td><td>87.47</td><td>29.00</td><td>71.63</td></tr><tr><td>Full strategy</td><td>82.10</td><td>91.80</td><td>87.54</td><td>34.93</td><td>73.49</td></tr></table>

## G. Limitations

Despite the encouraging results, NC-TFAD has several limitations. First, synthetic anomalies are used as auxiliary geometric anchors because real defects are unavailable during training, and their effectiveness therefore depends on whether the generated perturbations provide sufficiently informative normal–anomaly cues; they cannot fully capture the appearance and structural diversity of real industrial defects. Second, although the proposed shared geometry is generally robust to different stream compositions, performance can still vary under more challenging data arrangements, particularly on VisA. Third, the current evaluation is limited to MVTec AD and VisA, which are constructed benchmark datasets rather than naturally collected long-term industrial streams, leaving the behavior of NC-TFAD under gradual, recurrent, or irregular real-world distribution shifts to be further investigated. Finally, the pixel-level comparison is restricted to methods that natively support anomaly localization, and the relatively low AP observed under severe class imbalance indicates that finer anomaly-score calibration remains an open issue. These limitations motivate future research on more adaptive anomaly synthesis, robust continual adaptation under realistic temporal shifts, and improved calibration and localization for highly imbalanced industrial data.

## V. CONCLUSION

This paper formulates industrial visual inspection as a task-free continual anomaly detection problem, where data streams evolve over time without observable task boundaries. To address this setting, we propose NC-TFAD, a neuralcollapse-inspired geometry-driven framework that establishes a category-agnostic normal–anomaly reference using fixed ETF prototypes. Building on this geometry, NC-guided interand intra-class regularization and the Focal Neural Collapse Contrastive loss jointly promote prototype alignment, representation compactness, and sample-level discriminability under non-stationary streams. A normal-patch-prototype-guided localization branch further combines calibrated patch deviations with a weak attention prior to enable pixel-level anomaly localization without pixel-level supervision.

Extensive experiments and ablation studies on MVTec AD and VisA demonstrate that NC-TFAD achieves strong and competitive image-level anomaly detection performance and competitive pixel-level localization performance against representative localization-capable baselines under the taskfree continual learning protocol. The results further show that the proposed geometric constraints improve representation stability and preserve discriminative capability as the data distribution evolves. Future work will investigate more adaptive synthetic anomaly generation, anomaly-score calibration under severe class imbalance, continual adaptation on naturally collected long-term industrial streams, and more general localization mechanisms across broader industrial scenarios.

## REFERENCES

[1] S. Qu, X. Tao, X. Gong, Z. Qu, M. Prasad, F. Shen, Z. Zhang, and G. Ding, “Lscad: A large-small model collaboration framework for unsupervised industrial anomaly detection,” IEEE Trans. Instrum. Meas., 2025.

[2] S. Liu, X. Luan, and Y. Li, “Multimodal Industrial Anomaly Detection via Attention-Enhanced Memory-Guided Network,” IEEE Trans. Multimedia, 2025.

[3] F. Yang, P. Jing, W. Wang, F. L. Wang, and Y. Su, “PADNet: Progressive-Difference-Aware Feature Reconstruction Mechanism for Anomaly Detection,” IEEE Trans. Multimedia, 2025.

[4] D. Chen, K. Pan, G. Dai, G. Wang, Y. Zhuang, S. Tang, and M. Xu, “Improving vision anomaly detection with the guidance of language modality,” IEEE Trans. Multimedia, 2024.

[5] Y. Lin et al., “A survey on RGB, 3D, and multimodal approaches for unsupervised industrial image anomaly detection,” Inf. Fusion, p. 103139, 2025.

[6] W. Lv, Q. Su, and W. Xu, “One-for-all few-shot anomaly detection via instance-induced prompt learning,” in Proc. 13th Int. Conf. Learn. Represent. (ICLR), 2025.

[7] Z. You, L. Cui, Y. Shen, K. Yang, X. Lu, Y. Zheng, and X. Le, “A unified model for multi-class anomaly detection,” in Advances in Neural Information Processing Systems, vol. 35, pp. 4571–4584, 2022.

[8] H. He et al., “Learning transferable representations for image anomaly localization using dense pretraining,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2024.

[9] H. Zhang, Z. Wang, D. Zeng, Z. Wu, and Y.-G. Jiang, “DiffusionAD: Norm-guided one-step denoising diffusion for anomaly detection,” IEEE Trans. Pattern Anal. Mach. Intell., 2025.

[10] C. Huang, Q. Xu, Y. Wang, Y. Wang, and Y. Zhang, “Self-supervised masking for unsupervised anomaly detection and localization,” IEEE Trans. Multimedia, vol. 25, pp. 4426–4438, 2022.

[11] E. Yang, L. Shen, Z. Wang, S. Liu, G. Guo, X. Wang, and D. Tao, “Revisiting flatness-aware optimization in continual learning with orthogonal gradient projection,” IEEE Trans. Pattern Anal. Mach. Intell., 2025.

[12] T. Fukuda, H. Kera, and K. Kawamoto, “Adapter merging with centroid prototype mapping for scalable class-incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025.

[13] R. Aljundi, K. Kelchtermans, and T. Tuytelaars, “Task-free continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 11254–11263.

[14] F. Ye and A. G. Bors, “Task-free continual learning via online discrepancy distance learning,” in Advances in Neural Information Processing Systems, vol. 35, pp. 23675–23688, 2022.

[15] S. A. Bidaki, A. Mohammadkhah, K. Rezaee, F. Hassani, S. Eskandari, M. Salahi, and M. M. Ghassemi, “Online continual learning: A systematic literature review of approaches, challenges, and benchmarks,” arXiv preprint arXiv:2501.04897, 2025.

[16] V. Papyan, X. Han, and D. L. Donoho, “Prevalence of neural collapse during the terminal phase of deep learning training,” Proc. Nat. Acad. Sci. USA, vol. 117, no. 40, pp. 24652–24663, 2020.

[17] F. Ning, Y. Shi, X. Tong, M. Cai, and W. Xu, “A review and assessment of 3D CAD model retrieval in machine-part design,” Int. J. Comput. Integr. Manuf., vol. 38, no. 6, pp. 752–774, 2025.

[18] J. Liu, K. Wu, Q. Nie, Y. Chen, B. B. Gao, Y. Liu, and F. Zheng, “Unsupervised continual anomaly detection with contrastively-learned prompt,” in Proc. AAAI Conf. Artif. Intell., vol. 38, no. 4, 2024, pp. 3639–3647.

[19] J. Tang, H. Lu, X. Xu, R. Wu, S. Hu, T. Zhang, and F. Tsung, “An incremental unified framework for small defect inspection,” in Proc. Eur. Conf. Comput. Vis. (ECCV), Cham, Switzerland: Springer, 2024, pp. 307–324.

[20] X. Li, X. Tan, Z. Chen, Z. Zhang, R. Zhang, R. Guo, and Y. Xie, “One-for-more: Continual diffusion model for anomaly detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 4766–4775.

[21] F. Ye and A. G. Bors, “Online task-free continual learning via dynamic expandable memory distribution,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 20512–20522.

[22] F. Ye and A. G. Bors, “Online task-free continual generative and discriminative learning via dynamic cluster memory,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 26202–26212.

[23] F. Ye and A. G. Bors, “Online task-free continual learning via expansible vision transformer,” Pattern Recognit., vol. 169, 2026, Art. no. 111730.

[24] J. Y. Moon, K. H. Park, J. U. Kim, and G. M. Park, “Online class incremental learning on stochastic blurry task boundary via mask and visual prompt tuning,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 11731–11741.

[25] X. Wei, G. Li, and R. Marculescu, “Online-LoRA: Task-free online continual learning via low rank adaptation,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2025.

[26] Y. Yang, H. Yuan, X. Li, Z. Lin, P. Torr, and D. Tao, “Neural collapse inspired feature-classifier alignment for few-shot class incremental learning,” arXiv preprint arXiv:2302.03004, 2023.

[27] W. E and S. Wojtowytsch, “On the emergence of simplex symmetry in the final and penultimate layers of neural network classifiers,” in Math. Sci. Mach. Learn. (MSML), 2022, pp. 270–290, PMLR.

[28] Z. Zhang and M. Sabuncu, “Generalized cross entropy loss for training deep neural networks with noisy labels,” in Advances in Neural Information Processing Systems, vol. 31, 2018.

[29] U. Sara, M. Akter, and M. S. Uddin, “Image quality assessment through FSIM, SSIM, MSE and PSNR—a comparative study,” J. Comput. Commun., vol. 7, no. 3, pp. 8–18, 2019.

[30] Z. Zhong, J. Cui, Y. Yang, X. Wu, X. Qi, X. Zhang, and J. Jia, “Understanding imbalanced semantic segmentation through neural collapse,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 19550–19560.

[31] T. A. Dang, V. Nguyen, N. S. Vu, and C. Vrain, “Memory-efficient continual learning with neural collapse contrastive,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2025, pp. 7950–7959.

[32] J. Shen, Q. Hu, T. Feng, X. Wang, H. Cui, S. Wu, and W. Zhang, “In2nect: Inter-class and intra-class neural collapse tuning for semantic segmentation of imbalanced remote sensing images,” in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 7, 2025, pp. 6814–6822.

[33] Y. He, Y. Chen, Y. Jin, S. Dong, X. Wei, and Y. Gong, “Dyson: Dynamic feature space self-organization for online task-free class incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 23741–23751.

[34] J. Pourcel, N. S. Vu, and R. M. French, “Online task-free continual learning with dynamic sparse distributed memory,” in Proc. Eur. Conf. Comput. Vis. (ECCV), Cham, Switzerland: Springer, 2022, pp. 739–756.

[35] Q. Chen, H. Luo, C. Lv, and Z. Zhang, “A unified anomaly synthesis strategy with gradient ascent for industrial anomaly detection and localization,” in Proc. Eur. Conf. Comput. Vis. (ECCV), Cham, Switzerland: Springer, 2024, pp. 37–54.

[36] Z. Zuo, J. Dong, Y. Wu, Y. Qu, and Z. Wu, “CLIP3D-AD: Extending CLIP for 3D few-shot anomaly detection with multi-view images generation,” arXiv preprint arXiv:2406.18941, 2024.

[37] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “MVTec AD—A comprehensive real-world dataset for unsupervised anomaly detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 9592–9600.

[38] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer, “Spot-thedifference self-supervised pre-training for anomaly detection and segmentation,” in Proc. Eur. Conf. Comput. Vis. (ECCV), Cham, Switzerland: Springer, 2022, pp. 392–408.

[39] H. Deng and X. Li, “Anomaly detection via reverse distillation from one-class embedding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 9737–9746.

[40] H. Zhang, F. Li, S. Liu, L. Zhang, H. Su, J. Zhu, L. M. Ni, and H.- Y. Shum, “DINO: DETR with improved denoising anchor boxes for end-to-end object detection,” arXiv preprint arXiv:2203.03605, 2022.

[41] D. Rolnick, A. Ahuja, J. Schwarz, T. Lillicrap, and G. Wayne, “Experience replay for continual learning,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[42] Z. Wang, Z. Zhang, S. Ebrahimi, R. Sun, H. Zhang, C.-Y. Lee, X. Ren, G. Su, V. Perot, and J. Dy, “DualPrompt: Complementary prompting for rehearsal-free continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 631–648.

[43] Z. Wang, Z. Zhang, C.-Y. Lee, H. Zhang, R. Sun, X. Ren, and T. Pfister, “Learning to prompt for continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 139–149.

[44] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer, “WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[45] K. Long, L. Ma, J. Liu, L. Liu, and G. Xie, “Towards an Incremental Unified Multimodal Anomaly Detection: Augmenting Multimodal Denoising From an Information Bottleneck Perspective,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026, pp. 14116–14125.