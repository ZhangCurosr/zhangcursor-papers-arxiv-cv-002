# DiMoP: Diffusion-Driven Motion Representation Learning With Frame-Level Pseudo-Classification for Skeleton-Based Action Recognition

Shanaka Ramesh Gunasekara, Student, IEEE, Wanqing Li<sup>∗</sup>, Senior Member, IEEE, Nikalal Kaldera, Student, IEEE, Philip Ogunbona, Senior Member, IEEE, Jack Yang, Senior Member, IEEE.

Abstract—Robust skeleton-based action recognition requires representations that capture a wide spectrum of motions, from subtle to moderate and strong ones. Existing methods often focus on strong motions. This paper introduces DiMoP, a maskingand diffusion-driven motion representation learning method with frame-level pseudo-classification to explicitly learn the distribution of joint motions rather than regressing deterministic coordinates, as existing methods often do. By diffusing masked joints with progressive noise and denoising them conditioned on visible joints, DiMoP learns through controllable noising and denoising processes, enabling uniform learning of weak, moderate, and strong dynamics. To enable the masking-based generative diffusion learning with a discriminative capability, a pseudo-frame classifier is proposed that enforces the learning towards sequence-consistent and temporally coherent pseudo-labels without manual annotations. Together, these strategies provide a principled mechanism for joint generative and discriminative motion modeling. DiMoP achieves state-of-the-art performance across NTU RGB+D 60/120, and PKUMMD, including a 1.1 percentage point gain over prior works on NTU RGB+D 120 with the cross-subject protocol. The code is available at DiMoP.

Index Terms—Self-supervised learning, Skeleton-based action recognition, Motion distribution, Diffusion

## I. INTRODUCTION

Human action recognition is required in many diverse applications, including video surveillance, autonomous driving, robotics, healthcare monitoring, and industrial safety. It enables the detection of hazardous events, abnormal behaviors, and work-efficiency patterns. Compared with RGB-based methods [1, 2], skeleton data, obtainable from CCTV systems or commodity 3D cameras, has emerged as an efficient and robust alternative due to its lower computational cost and reduced sensitivity to environmental variations [3].

There has been significant progress in supervised action recognition [4, 5, 6, 7, 8, 9]. Despite this progress, performance remains highly dependent on large amounts of annotated training data. Collecting and annotating extensive datasets in the real-world is labor- and time-intensive. Consequently, unsupervised and self-supervised methods have emerged as promising alternatives [10, 11, 12, 13, 14, 15].

In self-supervised learning (SSL), learned representations are required to capture the intrinsic structure and distribution of the data without annotations to enhance downstream performance. To this end, various pre-training approaches are employed to learn effective representations. These approaches can be broadly categorized into two groups: contrastive methods [13, 14, 10, 16] and generative methods [12, 17, 18].

Contrastive approaches employ various transformations to approximate the data distribution in the real world. Typically, handcrafted transformations, such as random cropping, scaling, rotations, and temporal jittering, are used to generate multiple views of the same instance, thereby enhancing data diversity [14, 16]. However, these operations provide only limited stochasticity and fail to simulate the true probabilistic distribution of motion dynamics, ultimately oversimplifying the complex and uncertain behaviors inherent in skeletal movements, particularly the subtle movements.

Recent generative methods typically adopt the Masked Autoencoder (MAE) [19] framework for representation learning by reconstructing masked joints or body parts from unmasked or visible joints [12, 17, 18]. In these methods, random [11, 12] or sample-dependent masking [17] is often used. Although this design effectively models the relationship between unmasked and masked parts and captures their dynamics, the direct prediction of masked joints from unmasked joints often fails to learn subtle dynamics due to the simultaneous coexistence of weak and strong dynamics, leading to ineffective modeling of subtle motion in the presence of highly complex and dynamic motion patterns in the real world and, hence, compromising the generalization of the learned representations.

To improve the learning of subtle motion and the discriminative power of the learned representation, this paper proposes a framework for Diffusion-Driven Motion Representation Learning with Frame-Level Pseudo-classification, referred to as DiMoP. DiMoP is a joint diffusion-based generative and unsupervised discriminative learning model. First, DiMoP randomly selects joints and gradually diffuses them with controlled Gaussian noise, followed by denoising conditioned on the representation of visible joints. This progressive diffusion trajectory compels the model to learn subtle motion variations in the early steps and high-energy dynamics in the later steps, providing a principled mechanism that treats all motion scales equally: subtle, moderate, and strong. Second, a framebased pseudo-classification module is introduced to enhance the discriminativeness of the learned motion representation. The pseudo-classification module transforms a sequence of uniformly distributed random numbers of the same length as the action sequence into a frame-based pseudo-action label, conditioned upon the learned motion representation of the unmasked joints or parts. Note that we refer to the module as a pseudo-frame-based classifier or a pseudo-classifier because the actual action label is unknown, and the classifier aims to enforce the learned representation to achieve consistent and temporally coherent frame-based labels for the entire sequence.

The key contributions of this work are summarized as follows:

1) A generative-discriminative framework, DiMoP, is proposed. DiMoP replaces deterministic masking with a progressive noise–denoise process, enabling the model to fully capture the joint motion distribution and effectively learn the subtle, moderate, and strong dynamics. These capabilities are often lacking in existing maskingbased methods.

2) The introduction and integration of a pseudo-framebased classifier module into DiMoP enhance the discriminative power of the learned motion representation. This mechanism effectively enforces explicitly discriminative learning in a diffusion-based generative model.

3) Extensive experiments were conducted on four benchmark datasets to demonstrate that the proposed generative–discriminative design consistently yields state-ofthe-art or competitive performance, thereby validating its effectiveness for skeleton representation learning.

## II. RELATED WORK

This section briefly reviews recent advances in supervised and self-supervised skeleton-based action recognition.

## A. Supervised Methods

In early-stage works, CNN [20, 21, 22] and RNN [23, 24] architectures were often employed for action recognition, while more recent studies have been dominated by GCN [4, 25, 7, 6] and transformer-based architectures [26, 8] due to their promising performance.

Graph convolutional networks (GCNs) for skeleton-based action recognition were introduced by Yan et al. in the ST-GCN framework [27]. Subsequently, various techniques have been explored to improve model accuracy and robustness, including the use of adaptive data-driven topologies [4, 28], the expansion of spatial and temporal receptive fields [29, 30], adaptive feature aggregation [5, 6, 31], and prompt-based learning strategies [32].

In ST-TR [33], vision transformers are employed for skeleton action recognition through the introduction of distinct spatial and temporal attention modules. Subsequent works have explored hybrid transformers by integrating transformers with GCNs [34, 35] or temporal convolutions [36], along with attention modules and developing parameter-efficient models [26]. Recent studies have demonstrated that mask modeling is an effective approach for training vanilla transformer networks. The proposed DiMoP adapts a vanilla ViT [37] as the encoder and extends transformers to function as both the denoising decoder and the pseudo-frame-based classifier.

## B. Unsupervised Methods

Recent self-supervised approaches have employed contrastive [13, 14, 38, 10] and generative methods [11, 12, 18, 39] to learn meaningful representations from unlabeled data, thereby improving performance in multiple downstream tasks.

In contrastive learning, representations are learned by distinguishing between positive and negative pairs, which are typically generated using transformation functions that enhance the diversity of training samples. For instance, SkeletonCLR [38] employs shear and crop transformations. Recent studies demonstrate that stronger transformations, those that incorporate rich semantic information, can significantly improve generalizability and reduce the gap with fully supervised methods. Building on this idea, AimCLR [14] introduces extreme transformations to generate novel movement patterns, employing four spatial transformations (shear, spatial flip, rotation, axis mask), two temporal transformations (crop, temporal flip), and two spatio-temporal transformations (Gaussian noise, Gaussian blur). In contrast, ActCLR [13] applies extreme transformations only to static body parts to preserve critical motion-related information. Despite the introduction of various data transformations, many fail to effectively simulate the complex motion dynamics, limiting their ability to capture the full range of movement variations.

Masked Autoencoders (MAEs) [19] have recently been actively adopted in 3D action representation learning. Transformer-based MAEs were reported in SkeletonMAE [12] to reconstruct randomly masked joints. Specifically, visible joints are mapped to latent representations by the encoder, after which masked joints are replaced with fixed embeddings referred to as learnable tokens. These tokens, along with the latent representations of visible joints, are then fed into the reconstruction decoder to recover the masked joint coordinates. Subsequent works explored different masking operations or reconstruction targets to improve representation learning. For instance, in MAMP [17], highly mobile joints were masked with a higher probability, and the original motion was reconstructed starting from learnable tokens. However, because every masked joint is represented by the same learnable token, the model’s ability to capture the variability inherent in human motion is limited. In contrast, in DiMoP, diffused joints are used to replace the conventional masked patches, thereby encouraging the network to learn a probabilistic mapping from noisy to clean joint configurations.

Recently proposed MacDiff [11] employs a masked conditional diffusion framework where the denoiser is conditioned via global Adaptive Layer Normalization (AdaLN). The global representation, obtained by pooling the local features of visible joints from the semantic encoder, is used to uniformly scale and shift all feature channels in each denoiser layer. This global conditioning drives the denoising of the entire diffused skeleton, updating both visible and masked joints under the same modulation. Such global feature modulation provides coarse, sequence-level conditioning but lacks jointor spatial-specific distinction. In contrast, DiMoP introduces a structured partial-state diffusion that diffuses only masked joints while keeping visible joints as clean anchors throughout the denoising. Instead of global AdaLN modulation, DiMoP performs token-level cross-conditioning via cross-attention; queries from noisy (masked) joints attend to keys/values from visible-joint embeddings, enabling localized, motion-aware guidance that MacDiff’s global feature modulation cannot achieve.

## III. PROPOSED METHOD

## A. Overview

Figure 1 depicts the network architecture of the proposed method, DiMoP, which takes a raw skeleton sequence S as input, where $S \in \mathbb { R } ^ { T _ { i n } \times V \times C _ { i n } } , T _ { i n } , V$ and $C _ { i n }$ indicate the number of input frames, the number of joints, and the number of input channels, respectively. Following common practice in vision transformers [37], a linear embedding layer is utilized to map the input joint coordinates into joint embeddings.

The $\mathrm { D i M o P }$ network consists of four modules: an encoder $E ( \cdot )$ , a denoising decoder $D _ { \mathrm { { D e n o i s e } } } ( \cdot )$ , a mapping layer $g ( \cdot )$ and a pseudo-classifier $G _ { \mathrm { p s u } } ( \cdot )$ , where all except the mapping layer follow the standard transformer architecture [40].

Given a skeleton sample in the embedding space $x _ { 0 } \sim q ( x _ { 0 } )$ (where the subscript 0 denotes the original sample), a masking operation M first decomposes $x _ { 0 }$ into two parts: the masked joints $x _ { 0 } ^ { m }$ and the visible joints $x _ { 0 } ^ { v }$ . In DiMoP, the term masking refers to an indexing operation that separates the input joints into visible and masked subsets rather than replacing the masked joints with zero or a fixed value. The visible joints $x _ { 0 } ^ { v }$ are passed through the encoder $E ( \cdot )$ to produce latent representations, while the masked joints $x _ { 0 } ^ { m }$ retain their original values and go through the forward diffusion process. Specifically, the masked subset is progressively noised according to Equation 1, generating $\boldsymbol { x } _ { t } ^ { m }$ at diffusion step t. The denoising decoder $D _ { \mathrm { { D e n o i s e } } } ( \cdot )$ then takes the diffused masked joints and the latent representation of the visible joints as inputs, ultimately recovering the motion of the original sample x<sub>0</sub>.

$$
x _ { t } = \sqrt { \bar { \alpha } } x _ { 0 } + \sqrt { ( 1 - \bar { \alpha } _ { t } ) } \epsilon ,\tag{1}
$$

where, $x _ { t }$ denotes the diffused sample at t, $\alpha _ { t } = 1 - \beta _ { t }$ , and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } ~ ( \beta _ { 1 } , . . . , \beta _ { T } } \end{array}$ are predefined variance schedules for $T$ steps), $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ and I is an identity matrix.

Simultaneously, the encoder output u is converted by a mapping layer into a conditioning signal for $G _ { p s u } ( \cdot )$ , which transforms a randomly generated sequence into a homogeneous sequence via the pseudo-classifier. The denoising of masked joints and the pseudo-label generated by the classifier are used jointly to guide the encoder in enhancing its representational and discriminative capabilities.

After pre-training, only the joint embedding layer, encoder, and mapping layer are retained, with encoder and mapping outputs concatenated along the channel dimension for downstream inference.

## B. DiMoP

1) Encoder: An input sequence $\begin{array} { r c l } { S } & { \in } & { \mathbb { R } ^ { T _ { i n } \times V \times C _ { i n } } } \end{array}$ is divided into $\tau = T _ { i n } / l$ non-overlapping temporal segments and embedded by $g _ { 1 } ( \cdot )$ , resulting in $\boldsymbol { x } \in \mathbb { R } ^ { \tau \times V \times C }$ . Spatial and temporal positional embeddings are then added following the transformer design [40]: $x _ { 0 } ~ = ~ x + p e ^ { s } + p e ^ { t }$ , where $p e ^ { s } \ \in \ \mathbb { R } ^ { 1 \times V \times C }$ and $p e ^ { t } \in \mathbb { R } ^ { \tau \times 1 \times C }$ . The resulting $\tau \times V$ tokens are treated as the clean sample x<sub>0</sub>.

A random mask with a ratio r separates $x _ { 0 }$ into visible joints $x _ { 0 } ^ { v }$ and masked joints $x _ { 0 } ^ { m }$ , where $\Gamma = ( 1 - r ) \tau V$ joints remain visible. A high masking ratio $r = 0 . 9$ was set to form a strong representation for learning and to reduce encoding time. The random mask is used as an unbiased denoising learning strategy rather than as a sensor-occlusion simulation. Although real skeleton noise can be joint-dependent, an explicit occlusion prior may over-emphasize frequently corrupted or highly mobile joints. Uniform masking instead enables all joints, including low-motion but discriminative ones, to contribute to conditional denoising, as supported by the masking ablation in Section IV-D1.

The visible joints map to the latent space via the encoder.

$$
u = E \left( x _ { 0 } ^ { v } \right) \in \mathbb { R } ^ { \Gamma \times C } ,\tag{2}
$$

where u is the latent representation of $x _ { 0 } ^ { v }$ . This representation u is employed as the conditioning signal for both the denoising decoder and the pseudo-classifier.

After pre-training, only the encoder $E ( \cdot )$ and the mapping layer $g ( \cdot )$ are fine-tuned for downstream tasks.

2) Forward diffusion: In the forward diffusion process, only the masked joints are progressively diffused by repeatedly adding small amounts of Gaussian noise over $T$ steps, as specified by Equation 1.

The objective is to generate the masked joints through sampling, which is approximated by recursively drawing samples from $q ( x _ { 0 } ^ { m } \mid x _ { 0 } ^ { v } )$ , beginning with $x _ { T } ^ { m } \sim \mathcal { N } ( 0 , I )$ . When the variance of the noise $\beta _ { t }$ at each step t is sufficiently small, the distribution $q \big ( x _ { t - 1 } ^ { m } \ \big | \ x _ { t } ^ { m } , x _ { 0 } ^ { v } \big )$ can be treated as Gaussian [41]. Therefore, it can be approximated by a deep network.

3) Denoising Decoder: The denoising decoder takes noisy masked joints $\boldsymbol { x } _ { t } ^ { m }$ as input and utilizes the latent representations of the visible joints u as conditioning information. The noise level of each diffused patch is indicated by the timestep t, which is uniformly sampled from [1, T] during training.

Three decoder configurations are examined, differing in how attention is applied between visible latents and noisy tokens.

Joint Decoder: This configuration takes the full set of tokens (joints), $\tau \times V$ , as input, which consists of (i) encoded visible joints and (ii) diffused mask joints placed at their corresponding indices (visible indices $i d x _ { v i s i b l e }$ and mask indices $i d x _ { m a s k } )$ . Queries, keys, and values are obtained through linear projections of this combined token set. However, by mixing visible and masked joints without explicit separation, there is a risk of interference between the clean conditional data and the noisy inputs.

Cross Decoder V1: In this variant, each decoder block employs a cross-attention layer where noisy joints first attend to the latent representation of visible joints. Specifically, queries are computed through a linear projection of the noisy joints, while keys and values are derived solely from the linear projections of the latent visible joints. This iterative process denoises each noise patch conditioned on the constant visible information. One drawback is that each noisy joint attends independently to the visible joints, without considering the interactions among other noisy joints.

![](images/69b83bdb3536523b9b5ad440be1647ca759eccf03ebc75bd54a96f157ae1ea6c.jpg)  
Fig. 1: Overview of the proposed DiMoP framework. A given skeleton sequence is embedded into tokens, and random masking is applied. The encoder $E ( . )$ maps visible joints to the latent space. The decoder $D _ { D e n o i s e }$ uses the latent representation of the visible joints as a condition to denoise the diffused masked joints and recover the original motion. The $G _ { p s u }$ transforms a random sequence of distinct numbers into a homogeneous sequence to assign frame-wise pseudo-labels to enforce the discriminative power of the encoder.

Cross Decoder V2: This configuration takes the complete joint set as input but treats visible joints as a separate conditioning branch. Queries are computed through a linear projection of the entire joint set, while keys and values are derived solely from the linear projections of the latent visible joints. This design preserves the global context in the queries by allowing each joint to benefit from interactions with the complete set, while the exclusive conditioning on visible joints ensures a stable and reliable guidance signal during denoising.

Combining u and $x _ { t } ^ { m }$ allows the model to learn the conditional distribution $p ( x _ { 0 } ^ { m } \mid x _ { t } ^ { m } , x _ { 0 } ^ { v } )$ , where u acts as a learned prior guiding structure-aware denoising. This formulation aligns noisy joints with plausible motion manifolds, producing stable and discriminative representations.

Three decoder architectures are ablated in Section IV-D. In all scenarios, positional embeddings are added to the tokens (joints) before they are fed into the decoder. The model linearly projects the decoder output to generate the final prediction, which has the same shape as $x _ { 0 } .$ . The $D _ { D e n o i s e }$ is only used to denoise the diffused masked joints during training and will not be used for downstream tasks.

4) Pseudo-classifier: The pseudo-frame classifier serves as an auxiliary head that imposes a frame-level discriminative constraint. In particular, the pseudo-classifier enforces temporal coherence by mapping all frames of a sequence toward a consistent label space, thereby suppressing frame-level noise and compacting sequence-level representations. Although not tied to true action labels, these homogeneous pseudo-labels act as latent surrogates that enhance feature separability and strengthen discriminative power for downstream recognition.

While diffusion learning primarily reconstructs motion dynamics, it may yield overly smooth representations; the pseudo-classifier mitigates this by enforcing temporal consistency within actions and separation across distinct motion phases, effectively introducing an implicit temporal clustering signal that enhances inter-class separability and intra-action coherence without real labels. Specifically, the pseudo-framebased classifier $G _ { p s u } ( \cdot )$ is driven by a random sequence and generates a homogeneous sequence conditioned on the encoder output u.

The random sequence consists of distinct random integers $Z _ { 0 } = \{ z _ { 1 } , z _ { 2 } , z _ { 3 } , . . . , z _ { i } , . . . , z _ { \tau } \} \in \mathcal { R } ^ { \tau \times 1 }$ , where each element $z _ { i } \in [ 1 , \tau ]$ . Based on the encoder output u, the classifier produces a homogeneous sequence $Z _ { H } = \{ z _ { k } , z _ { k } , z _ { k } , . . . , z _ { k } \} \in$ $\mathbb { R } ^ { \tau \times 1 }$ , where $z _ { k } \in \{ z _ { 1 } , z _ { 2 } , z _ { 3 } , . . . , z _ { \tau } \}$ . The classifier is formally expressed as: $Z _ { H } = G _ { p s u } ( Z _ { 0 } , u )$

5) Training Objective: The model is pre-trained using a combination of denoising loss and the homogeneity of the label sequences produced by the pseudo-classifier. The homogeneity loss is assessed by measuring the temporal smoothness and consistency of the pseudo-labels across the frames.

Denoising loss: Unlike the standard denoising Diffusion Probabilistic Models (DDPM) approach [41], which often employs ϵ-prediction for noise estimation, the proposed Di-MoP model directly reconstructs the clean sample through a denoising process. Both formulations are widely adopted in diffusion models [41]. In particular, DiMoP is trained to predict the motions of masked joints by denoising the noisy input, thus recovering the underlying motion distributions.

The denoising objective is defined as:

$$
\mathcal { L } _ { D e n o i s e } = \mathbb { E } _ { t , x _ { 0 } , \epsilon } \Big \| \dot { x } _ { 0 } ^ { m } - D _ { \mathrm { D e n o i s e } } \big ( x _ { t } ^ { m } , t , E ( x _ { 0 } ^ { v } ) \big ) \Big \| ^ { 2 } ,\tag{3}
$$

where $\dot { x } _ { 0 } ^ { m }$ is the motion of the original sequence $x _ { 0 } .$

Temporal smoothness: The temporal smoothness loss regularizes local frame-to-frame variations in the pseudo-classifier output. The output of $G _ { p s u } ( \cdot )$ is first passed through a softmax layer, and the frame-level pseudo-label sequence is obtained as $Z _ { H } = \mathrm { a r g m a x } ( \mathrm { s o f t m a x } ( G _ { p s u } ( \cdot ) ) )$ . Since $Z _ { H }$ represents frame-wise pseudo-class assignments within the same action instance, abrupt transitions between adjacent frames are discouraged. To this end, a binary sequence $Z _ { B } \in \{ 0 , 1 \} ^ { \tau - 1 }$ is generated by comparing consecutive predictions in $Z _ { H }$ , where $Z _ { B } [ i ] = 1$ if the predicted value changes between frames i and $i + 1$ , and $Z _ { B } [ i ] = 0$ otherwise. The smoothness loss is then defined as,

$$
\mathcal { L } _ { s m o o t h } = \frac { 1 } { \tau - 1 } \sum _ { i = 1 } ^ { \tau - 1 } Z _ { B } [ i ] .\tag{4}
$$

Therefore, $\mathcal { L } _ { s m o o t h }$ penalizes local temporal discontinuities and encourages smooth pseudo-label evolution. However, since it only measures adjacent transitions, it does not explicitly enforce sequence-level agreement among all frames.

Value consistency loss: To impose a global sequencelevel constraint, a value consistency loss was introduced that encourages all frames to support the dominant pseudo-label estimated over the entire sequence. Unlike $\mathcal { L } _ { s m o o t h }$ , which only penalizes local transitions, this loss directly promotes global agreement across all frame-level predictions. Specifically, for each candidate value $z _ { k }$ , its sequence-level vote is computed as

$$
v o t e _ { k } = \sum _ { r = 1 } ^ { \tau } p _ { r , k } ,\tag{5}
$$

where $p _ { r , k }$ denotes the probability of assigning frame r to value $z _ { k }$ . The dominant pseudo-label is then selected as

$$
z ^ { m a j } = \underset { k } { \mathrm { a r g m a x } } ( v o t e _ { k } ) .\tag{6}
$$

Finally, each frame is encouraged to agree with this dominant value:

$$
\mathcal { L } _ { c o n s i s t e n c y } = 1 - \frac { 1 } { \tau } \sum _ { r = 1 } ^ { \tau } p _ { r } ^ { m a j } ,\tag{7}
$$

where $p _ { r } ^ { m a j }$ is the probability assigned to the majority value at frame r. Thus, $\mathcal { L } _ { s m o o t h }$ and $\mathcal { L } _ { { c o n s i s t e n c y } }$ are complementary: the former regularizes local temporal continuity, while the latter enforces global sequence-level consensus.

The total loss function for the proposed network is expressed as a combination of $\mathcal { L } _ { D e n o i s e } ,$ the temporal smoothness loss $\mathcal { L } _ { s m o o t h }$ , and the consistency loss $\mathcal { L } _ { { c o n s i s t e n c y } } ,$

$$
\mathcal { L } = \alpha \mathcal { L } _ { D e n o i s e } + \gamma \left( \mathcal { L } _ { c o n s i s t e n c y } + \frac { \lambda } { \gamma } \mathcal { L } _ { s m o o t h } \right) ,\tag{8}
$$

where $\alpha$ and $\gamma$ balance the generative denoising and discriminative pseudo-classification objectives, respectively, while $\lambda / \gamma$ controls the contribution of local temporal smoothness relative to global sequence consistency.

## IV. EXPERIMENTS

## A. Datasets

The proposed DiMoP is evaluated on three commonly used large-scale datasets.

1) NTU RGB+D 60 [42]: NTU RGB+D 60 is a large-scale dataset for human action recognition that contains 60 action categories performed by 40 subjects, totaling 56,880 skeleton sequences. The dataset is evaluated using two standard protocols: cross-subject (X-Sub) and cross-view (X-View). In the X-Sub protocol, action sequences from 20 subjects are used for training, while sequences from the remaining subjects are reserved for testing. In the X-View protocol, training samples are captured from cameras 2 and 3, while testing is conducted on samples from camera 1.

2) NTU RGB+D 120 [43]: NTU RGB+D 120 extends NTU RGB+D 60 by doubling the action categories from 60 to 120. The dataset comprises 114,480 skeleton sequences and is evaluated using cross-subject (X-Sub) and cross-setup (X-Set) protocols. In X-Set, sequences are divided into 32 setups based on camera distance and background, with half allocated for training. The unique camera viewpoints ensure diversity in positions, angles, and backgrounds. The dataset covers a broad range of actions, including in-office activities, object interactions, fine-grained single-person actions (which are subject to occlusions and noise), and mutual interactions.

3) PKUMMD [44]: PKUMMD is a large-scale dataset for 3D human action recognition, divided into two parts. Part 1 contains over 1,000 action instances across 51 classes performed by 66 subjects, while Part 2 includes approximately 20,000 instances under more challenging conditions, characterized by increased variability, occlusion, and overlap.

These datasets provide complementary evaluation settings with distinct degradation factors and purposes. NTU RGB+D 60 serves as a standard benchmark for comparison with existing self-supervised skeleton representation learning methods, covering viewpoint changes, inter-subject variation, intra-class diversity, and skeleton tracking noise. NTU RGB+D 120 evaluates scalability and robustness in a larger and more diverse setting, with 120 action classes, 106 subjects from 15 countries, wide age and height ranges, multiple camera setups, fine-grained actions, object interactions, and diverse skeleton variations. PKUMMD assesses generalization under more realistic sequence-level challenges, including long continuous actions, action transitions, temporal ambiguity, subject overlap, and occlusion. Collectively, these datasets evaluate DiMoP across standard benchmark comparison, large-scale recognition, and challenging settings, while covering action classes with subtle, strong, and mixed subtle–strong motion patterns.

## B. Implementation

The encoder was adapted from MAMP [17]. The model takes a 300-frame-long skeleton sequence as input, which is then cropped and interpolated to 60 frames. The patch size l is set to 2 to generate non-overlapping temporal segments, which facilitates $\tau = 3 0$

The embedding dimension of 256 is assigned to all three networks, and each multi-head self-attention (MSHA) block in $E ( \cdot )$ and $D _ { \mathrm { { D e n o i s e } } } ( \cdot )$ is configured with 8 heads. The encoder consists of eight layers, while the denoising decoder comprises five layers. The pseudo-classifier consists of cross-attention blocks. Before the first layer of each module, separate learnable spatial and temporal positional embeddings are added to the input embeddings. A masking ratio of 0.9 is applied. For the forward diffusion process, a total of $T = 3 0 0$ sampling steps are used, and a linear noise scheduling scheme [45] is adopted.

A pre-training phase consisting of 400 epochs is conducted with a batch size of 32 on nine NVIDIA P100 GPUs. The first 20 epochs serve as a warm-up stage to ensure stable training. During this phase, the learning rate is elevated from 0 to $1 \times$ $1 0 ^ { - 3 }$ , and then gradually lowered to $5 \times 1 0 ^ { - 4 }$ using a cosine decay schedule. The Adam optimizer is employed to update the model parameters.

## C. Comparison with State-of-the-art Methods

1) Linear evaluation: During the linear evaluation, a linear classifier $\phi ( . )$ is placed on top of the pre-trained encoder $E ( . )$ and the mapping layer $g ( . )$ . Both $E ( . )$ and $g ( . )$ are kept frozen, while only $\phi ( . )$ is optimized for action classification. The model was trained for 150 epochs with a batch size of 128 and an initial learning rate of 0.1. Table I compares the performance of DiMoP with state-of-the-art (SOTA) approaches. DiMoP outperformed the baseline MAMP [17] by 2.8 and 1.9 percentage points on the NTU RGB+D 60 dataset using X-view and X-sub evaluations, respectively. Additionally, it outperformed MacDiff [11] by 1.6 percentage points on the NTU RGB+D 120 X-set evaluation.

The statistical reliability of DiMoP was further assessed using bootstrap confidence estimation on NTU RGB+D 60 X-sub and X-view. Test samples were resampled with replacement for 50 iterations, and 95% confidence intervals were computed from the resulting accuracy distributions. On X-sub, 86.80% accuracy was achieved by DiMoP with a 95% CI of [86.53, 87.01], exceeding MAMP [84.27, 85.13] and MacDiff [85.93, 86.56]. On X-view, 91.90% accuracy was obtained with a 95% CI of [91.62, 92.13], also exceeding MAMP [88.79, 89.42] and MacDiff [90.61, 91.18]. These results indicate that the observed improvements are statistically significant.

2) Supervised fine-tune evaluation: A classification head $\phi ( \cdot )$ is attached to the pre-trained encoder and mapping layer, and the entire network is fine-tuned. The trained model was evaluated on both the NTU RGB+D 60 and NTU RGB+D 120 datasets. The model was trained for 150 epochs with a batch size of 128 and an initial learning rate of 0.1. Table II compares the performance of DiMoP with the SOTA approaches. DiMoP achieved results comparable to those of the recent generative SOTA method. For instance, DiMoP increased accuracy from 97.3% to 97.7% in the NTU RGB+D 60 X-view evaluation. DiMoP achieved comparable results to MAMP [17] on the NTU RGB+D 120 dataset. Additionally, DiMoP outperformed the fully supervised STGCN [27] and BlockGCN [4] across all evaluations on the NTU RGB+D 60 dataset while achieving comparable results on the NTU RGB+D 120 dataset.

3) Transfer Learning Evaluation: The effectiveness of the proposed method is evaluated in a transfer-learning setup to assess its generalization capability. The encoder and mapping layer are pre-trained on the NTU RGB+D 60 and NTU RGB+D 120 datasets separately using X-sub settings and are evaluated on the PKU-MMD II dataset using the supervised fine-tuning evaluation setting. The results are presented in Table III. DiMoP achieved state-of-the-art (SOTA) performance on NTU RGB+D 60, surpassing MacDiff by 0.2 percentage points. These improvements confirm that DiMoP generalizes well beyond the datasets and highlight its robustness under transfer learning.

4) Semi-supervised evaluation: The post-attached classification layer and pre-trained encoder are fine-tuned using 1% and 10% of the training data, following the supervised settings in Section IV-C2. Table IV compares our method with stateof-the-art approaches. The proposed DiMoP achieved a 2.6 percentage point improvement in accuracy over S-JEPA and yielded the same state-of-the-art (SOTA) result as S-JEPA on the NTU RGB+D 60 X-sub evaluation, utilizing only 1% and 10% of the training data, respectively.

Although MacDiff [11] reports higher semi-supervised results in its original paper, its exact semi-supervised training configuration was not provided with the released code. Hence, rather than directly comparing against potentially different settings, we re-trained MacDiff under the same semi-supervised protocol used for DiMoP, including identical label ratios, training settings, and evaluation protocols. The lower reproduced MacDiff accuracy in Table IV than that of originally reported may be attributed to the different settings. Nevertheless, under this controlled comparison, DiMoP improves the reproduced MacDiff results by 0.3 and 0.4 percentage points on the Xview protocol with 1% and 10% labeled data, respectively.

## D. Ablation studies

The joint decoder configuration is used for the first four ablation studies. Unless an ablation explicitly varied a factor (e.g., masking ratio, number of diffusion steps, decoder design, or loss weights), all other hyperparameters were fixed to the linear evaluation settings.

1) Analysis ofmasking strategy and masking ratio: Table V compares simple random masking with motion-aware random masking, implemented following [17]. Random masking achieves better performance because it uniformly exposes all joints to the denoising objective, whereas motion-aware masking may over-emphasize large-motion joints and under-sample subtle yet discriminative motions. Nevertheless, uniform masking is not intended to explicitly model structured occlusion. In scenarios with persistent joint occlusion, such as lowerbody occlusion behind a desk, joint-confidence scores from the pose estimator could be incorporated into both the masking probabilities and denoising loss to avoid treating unreliable or unavailable joints as reconstruction targets. Table VI further shows that a high masking ratio of 90% yields the best performance.

2) Hyperparameter tuning of the loss function and contribution analysis of each loss term: Table VII reports the linear evaluation results under different loss-weight settings. The denoising loss is the primary generative objective, while the consistency and smoothness losses impose global and local constraints on the pseudo-classifier. We set $\alpha = \gamma = 1$ to balance the generative and discriminative objectives. Since $\mathcal { L } _ { s m o o t h }$ is auxiliary to sequence-level consistency, $\lambda / \gamma$ should generally remain below 1. Our experiments showed that α = 1, $\lambda = 0 . 3 ,$ and γ = 1 yielded the best performance across all datasets and evaluation protocols considered in this study.

TABLE I: Linear evaluation performance of DiMoP versus state-of-the-art methods on the NTU RGB+D 60, NTU RGB+D 120, and PKUMMD datasets. The term “3s-” denotes the ensemble results from the joint (J), bone (B), and motion (M) streams. Bold and underlined entries indicate the best and second-best performances, respectively.
<table><tr><td rowspan=1 colspan=5>Models</td><td rowspan=1 colspan=1>Venue</td><td rowspan=1 colspan=1>Stream</td><td rowspan=1 colspan=2>NTU 60</td><td rowspan=1 colspan=2>NTU 120</td><td rowspan=1 colspan=1>PKU MMD</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X-sub</td><td rowspan=1 colspan=1>X-view</td><td rowspan=1 colspan=1>X-sub</td><td rowspan=1 colspan=1>X-set</td><td rowspan=1 colspan=1>Part I</td></tr><tr><td rowspan=1 colspan=5>Contrastive MethodsISC [46]</td><td rowspan=1 colspan=1>MM&#x27;21</td><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>76.3</td><td rowspan=1 colspan=1>78.6</td><td rowspan=1 colspan=1>67.9</td><td rowspan=1 colspan=1>67.1</td><td rowspan=1 colspan=1>80.9</td></tr><tr><td rowspan=2 colspan=5>GL-Transformer [47]PSTL [48]</td><td rowspan=1 colspan=1>ansformer [47]</td><td rowspan=1 colspan=1>ECCV’22</td><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>76.3</td><td rowspan=1 colspan=1>83.8</td><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>68.7</td></tr><tr><td rowspan=1 colspan=1>AAAI&#x27;23</td><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>77.3</td><td rowspan=1 colspan=1>81.8</td><td rowspan=1 colspan=1>69.2</td><td rowspan=1 colspan=1>67.7</td><td rowspan=1 colspan=1>88.4</td></tr><tr><td rowspan=1 colspan=5>CPM [49]</td><td rowspan=1 colspan=1>ECCV’22</td><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>78.7</td><td rowspan=1 colspan=1>84.9</td><td rowspan=1 colspan=1>68.7</td><td rowspan=1 colspan=1>69.6</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=5>USDRL [50]</td><td rowspan=1 colspan=1>AAAI&#x27;25</td><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>85.2</td><td rowspan=1 colspan=1>91.7</td><td rowspan=1 colspan=1>76.6</td><td rowspan=1 colspan=1>78.1</td><td rowspan=3 colspan=1>92.9</td></tr><tr><td rowspan=1 colspan=2>U-FEFP [</td><td rowspan=1 colspan=3>51]</td><td rowspan=1 colspan=1>TCSVT’25</td><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>91.2</td><td rowspan=1 colspan=1>78.23</td><td rowspan=1 colspan=1>79.6</td></tr><tr><td rowspan=1 colspan=5>3s-SkeletonCLR [38]</td><td rowspan=1 colspan=1>CVPR&#x27;21</td><td rowspan=1 colspan=1>J+B+M</td><td rowspan=1 colspan=1>75.0</td><td rowspan=1 colspan=1>79.8</td><td rowspan=1 colspan=1>60.7</td><td rowspan=1 colspan=1>62.6</td></tr><tr><td rowspan=1 colspan=5>3s-CrosSCLR [38]</td><td rowspan=1 colspan=1>CVPR&#x27;21</td><td rowspan=1 colspan=1>J+B+M</td><td rowspan=1 colspan=1>77.8</td><td rowspan=1 colspan=1>83.4</td><td rowspan=1 colspan=1>67.9</td><td rowspan=1 colspan=1>66.7</td><td rowspan=1 colspan=1>84.9</td></tr><tr><td rowspan=1 colspan=5>3s-AimCLR [14]</td><td rowspan=1 colspan=1>AAAI&#x27;22</td><td rowspan=1 colspan=1>J+B+M</td><td rowspan=1 colspan=1>78.9</td><td rowspan=1 colspan=1>83.8</td><td rowspan=1 colspan=1>68.2</td><td rowspan=1 colspan=1>68.8</td><td rowspan=1 colspan=1>87.8</td></tr><tr><td rowspan=1 colspan=5>3s-PSTL [48]</td><td rowspan=1 colspan=1>AAAI&#x27;23</td><td rowspan=1 colspan=1>J+B+M</td><td rowspan=1 colspan=1>79.1</td><td rowspan=1 colspan=1>83.8</td><td rowspan=1 colspan=1>62.2</td><td rowspan=1 colspan=1>70.3</td><td rowspan=4 colspan=1>89.2</td></tr><tr><td rowspan=1 colspan=5>3s-CPM [49]</td><td rowspan=1 colspan=1>ECCV’22</td><td rowspan=1 colspan=1>J+B+M</td><td rowspan=1 colspan=1>83.2</td><td rowspan=1 colspan=1>87.0</td><td rowspan=1 colspan=1>73.0</td><td rowspan=3 colspan=1>74.075.779.3</td></tr><tr><td rowspan=1 colspan=5>3s-ActCLR [13]</td><td rowspan=2 colspan=1>CVPR&#x27;23TBIOM&#x27;25</td><td rowspan=2 colspan=1>J+B+MJ+B+M</td><td rowspan=2 colspan=1>84.385.9</td><td rowspan=2 colspan=1>88.890.0</td><td rowspan=2 colspan=1>74.377.1</td></tr><tr><td rowspan=1 colspan=5>3s-STJD-CL [10]</td></tr><tr><td rowspan=2 colspan=5>Generative Methods:SkeletonMAE [12]MAMP [17]</td><td rowspan=1 colspan=1>ICMEW&#x27;23</td><td rowspan=2 colspan=1>JJ</td><td rowspan=2 colspan=1>74.884.9</td><td rowspan=3 colspan=1>77.789.1</td><td rowspan=2 colspan=1>72.578.6</td><td rowspan=5 colspan=1>73.579.179.980.280.4</td><td rowspan=5 colspan=1>82.892.292.292.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ICCV&#x27;23</td></tr><tr><td rowspan=1 colspan=4>S-JEPA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ECCV&#x27;24</td><td rowspan=3 colspan=1>JJJ</td><td rowspan=3 colspan=1>85.386.485.4</td><td rowspan=2 colspan=1>91.0</td><td rowspan=1 colspan=1>79.6</td></tr><tr><td rowspan=2 colspan=5>MacDiff [11]STJD-MP [10]</td><td rowspan=2 colspan=1>ECCV’24TBIOM&#x27;25</td><td rowspan=2 colspan=1>79.1</td></tr><tr><td rowspan=1 colspan=1>90.2</td></tr><tr><td rowspan=1 colspan=5>DiMoP(Ours)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>86.8</td><td rowspan=1 colspan=1>91.9</td><td rowspan=1 colspan=1>80.7</td><td rowspan=1 colspan=1>81.8</td><td rowspan=1 colspan=1>93.2</td></tr></table>

TABLE II: Supervised fine-tune evaluation results on the NTU RGB+D 60, and NTU RGB+D 120 datasets.
<table><tr><td rowspan=1 colspan=1>Models</td><td rowspan=1 colspan=1>NTU 60 (%)X-sub  X-view</td><td rowspan=1 colspan=1>NTU 120 (%)X-sub  X-set</td></tr><tr><td rowspan=3 colspan=1>Contrastive Methods3s-CrosSCLR3s-AimCLR3s-PSTL3s-ActCLR3s-STJD-CL</td><td rowspan=3 colspan=1>86.2    92.586.9    92.887.1    93.988.2    93.989.3    94.8</td><td rowspan=1 colspan=1>80.5   80.4</td></tr><tr><td rowspan=1 colspan=1>80.1   80.9</td></tr><tr><td rowspan=1 colspan=1>81.3   82.681.6   81.283.5   86.8</td></tr><tr><td rowspan=1 colspan=1>Generative MethodsSkeletonMAEMAMPMacDiffS-JEPA</td><td rowspan=1 colspan=1>88.5    94.793.1    97.592.7    97.393.1    97.6</td><td rowspan=1 colspan=1>87.0   88.990.0   91.390.3   91.3</td></tr><tr><td rowspan=1 colspan=1>Supervised Methods3s-STGCN [27]CTR-GCN [7]BlockGCN [4]</td><td rowspan=1 colspan=1>85.2    91.492.4    96.893.1    97.0</td><td rowspan=1 colspan=1>77.2   77.188.9   90.690.3   91.5</td></tr><tr><td rowspan=1 colspan=1>DiMoP (ours)</td><td rowspan=1 colspan=1>93.2    97.7</td><td rowspan=1 colspan=1>90.3   91.1</td></tr></table>

TABLE IV: Semi-supervised performance of DiMoP and its comparison with the SOTA methods. <sup>∗</sup> results were obtained by the provided codes.
<table><tr><td rowspan="3">Models</td><td colspan="3">NTU RGB+D 60 (%)</td></tr><tr><td>1% labels</td><td>10% labels</td><td></td></tr><tr><td>X-sub X-view</td><td>X-sub</td><td>X-view</td></tr><tr><td>3s-AimtCLR</td><td>54.8</td><td>54.3</td><td>78.2 81.6</td></tr><tr><td>3s-ActCLR</td><td>64.8 65.6</td><td>81.7</td><td>85.8</td></tr><tr><td>3s-STJD-CL</td><td>67.5 68.4</td><td>82.5</td><td>88.0</td></tr><tr><td>SkeletonMAE</td><td>54.4 54.6</td><td>80.6</td><td>83.5</td></tr><tr><td>MAMP</td><td>66.0 68.7</td><td>88.0</td><td>91.5</td></tr><tr><td>MacDiff</td><td>65.6 77.3</td><td>88.2</td><td>92.5</td></tr><tr><td>MacDiff *</td><td>64.4 70.9</td><td>87.5</td><td>91.9</td></tr><tr><td>S-JEPA</td><td>67.5 69.1</td><td>88.4</td><td>91.4</td></tr><tr><td>DiMoP (ours)</td><td>70.1 71.2</td><td>88.4</td><td>92.3</td></tr></table>

TABLE V: DiMoP performance under varying masking strategies. The linear evaluation results are reported on the NTU RGB+D 60 Xsub dataset. The best one is in bold.

TABLE III: Transfer learning evaluation of DiMoP on the PKU-MMD Part II dataset: Fine-tuning the pre-trained encoder from NTU RGB+D 60 and NTU RGB+D 120.
<table><tr><td>Models</td><td>To PKUMMD II (%) NTU 60 NTU 120</td></tr><tr><td>SkeletonMAE MAMP MacDiff</td><td>58.4 61.0 70.6 73.2 72.2 73.4</td></tr><tr><td>S-JEPA</td><td>71.4 74.2</td></tr><tr><td>DiMoP (ours)</td><td>72.4 73.9</td></tr></table>

<table><tr><td>Masking strategy</td><td>Acc(%)</td></tr><tr><td>Random Masking</td><td>86.48</td></tr><tr><td>Motion Aware masking</td><td>86.13</td></tr></table>

TABLE VI: DiMoP performance under varying masking ratios. The linear evaluation results are reported on the NTU RGB+D 60 Xsub dataset. The best one is in bold.

<table><tr><td>Masking ratio</td><td>Acc(%)</td></tr><tr><td>60%</td><td>70.21</td></tr><tr><td>65%</td><td>74.44</td></tr><tr><td>70%</td><td>75.65</td></tr><tr><td>75%</td><td>78.38</td></tr><tr><td>80%</td><td>82.59</td></tr><tr><td>85%</td><td>85.64</td></tr><tr><td>90%</td><td>86.48</td></tr><tr><td>95%</td><td>81.23</td></tr></table>

As a practical guideline, we recommend fixing the other loss weights and setting λ to a relatively small value for cases involving complex and diverse actions, while increasing them for cases involving less diverse actions with lower temporal variation.

TABLE VII: Hyperparameter tuning for DiMoP on the NTU-RGB+D 60 dataset under the X-Sub setting in linear evaluation. The best one is in bold.
<table><tr><td>α</td><td>λ</td><td>γ</td><td>Acc %</td></tr><tr><td>1</td><td>1</td><td>1</td><td>85.85</td></tr><tr><td>1</td><td>0.1</td><td>0.1</td><td>85.34</td></tr><tr><td>1</td><td>1</td><td>0.1</td><td>85.81</td></tr><tr><td>1</td><td>0.1</td><td>1</td><td>86.24</td></tr><tr><td>1</td><td>0.3</td><td>1</td><td>86.48</td></tr><tr><td>1</td><td>0.5</td><td>0.5</td><td>86.02</td></tr><tr><td>0.9</td><td>0.1</td><td>0.9</td><td>86.35</td></tr><tr><td>0.5</td><td>0.2</td><td>0.2</td><td>83.23</td></tr></table>

3) Analysis of the number of diffusion steps and Noise scheduling: The impact of noise level on learning discriminative features is studied by varying the number of diffusion steps T from $T \ : = \ : 5 0$ to $T = 7 0 0$ in 50-step increments. A linear schedule is used by default, following DDPM [45], where the variances $\beta [ 1 : T ]$ increase linearly. Figure 2 presents the linear evaluation accuracy on the NTU $\mathrm { R G B + D }$ X-sub evaluation with varying diffusion steps. It is observed that the model performed well with 250 to 350 diffusion steps compared to those with either low or high diffusion steps. When the number of diffusion steps is high, the diffused samples approach random noise, which prevents the denoising networks from learning meaningful and discriminative feature representations for downstream classification tasks.

The selection of noise schedules, such as linear and cosine schedules [45], for forward diffusion has been validated. Table IX presents the linear evaluation accuracy on NTU RGB+D X-sub evaluation with varying noise schedules. For this experiment, a noise level $T = 3 0 0$ was used, and the joint decoder was employed. The linear schedule yields the best performance.

4) Analysis of denoising prediction target:: In the denoising process, various prediction targets were evaluated, including noise, ϵ prediction, the prediction of the original joint coordinates, $x _ { 0 } ,$ , and the prediction of the motion dynamics of the original $x _ { 0 }$ . This work focuses on the impact of prediction targets on representation learning, rather than on the quality of generation. As demonstrated in Table X, predicting the motion dynamics of the original sample $x _ { 0 }$ yielded better performance than noise prediction.

TABLE VIII: Module Contribution Analysis in DiMoP
<table><tr><td>LDenoise</td><td>Lsmooth</td><td>Lconsistency</td><td>Acc (%)</td></tr><tr><td>√</td><td></td><td></td><td>85.31</td></tr><tr><td></td><td>√</td><td>√</td><td>34.73</td></tr><tr><td>√</td><td>√</td><td></td><td>85.81</td></tr><tr><td>√</td><td></td><td>√</td><td>86.09</td></tr><tr><td>√</td><td>√</td><td>√</td><td>86.48</td></tr></table>

TABLE IX: DiMoP performance under varying noise schedules. The linear evaluation results are reported on the NTU RGB+D 60 Xsub dataset. The best one is in bold.
<table><tr><td>Noise Schedule</td><td>Acc(%)</td></tr><tr><td>Linear</td><td>86.48</td></tr><tr><td>Cosine</td><td>85.61</td></tr></table>

![](images/4f9456fbb32e2db490c5a2e9344ddcb0353dbd5a9a87107a169fd2c49c22d20a.jpg)  
Fig. 2: Linear evaluation performance variation against the number of diffusion steps on the NTU RGB+D X-sub.

TABLE X: Ablation study on the prediction target. The linear evaluation results are reported on the NTU RGB+D 60 Xsub dataset. The best one is in bold.
<table><tr><td>Prediction target</td><td>Acc(%)</td></tr><tr><td>Noise €</td><td>49.03</td></tr><tr><td>Coordinates of the original  $x _ { 0 }$ </td><td>85.94</td></tr><tr><td>Motion dynamics of the original  $x _ { 0 }$ </td><td>86.48</td></tr></table>

5) Denoising decoder configurations: The three design configurations described in Section III-B3 were experimentally validated. During these experiments, $T = 3 0 0$ was set, and a linear noise schedule was employed. The linear evaluation performance on the NTU RGB+D 60 X-Sub dataset is presented in Table XI. The best performance was obtained with Cross Decoder V2, thereby validating the theoretical design configuration.

TABLE XI: Comparison of Denoising Decoder Configurations: Linear Evaluation Results on the NTU RGB+D 60 Xsub Dataset with the Best Configuration in Bold.
<table><tr><td>Configuration</td><td>Acc(%)</td></tr><tr><td>Joint decoder</td><td>86.48</td></tr><tr><td>Cross decoder V1</td><td>85.73</td></tr><tr><td>Cross decoder V2</td><td>86.76</td></tr></table>

6) Class-wise performance comparison on X-sub evaluation: Class-wise accuracies for the proposed DiMoP are compared with those of the baseline method, MAMP [17], whose results were reproduced using publicly available code. On the NTU RGB+D 60 dataset, 47 of the 60 action classes show improved accuracy in the X-sub. Figure 4 presents a detailed comparison, where pink indicates drops and green indicates gains.

As shown, substantial improvements were observed for the majority of actions. Performance declines were noted for some actions, many of which involve interactions between two subjects. Actions in which the discriminative cue is subtle and short-lived, while most frames are dominated by neutral or overlapping poses, remain challenging. In these cases, sequence-level homogeneity tends to overshadow the brief but critical cues, resulting in misclassifications and reduced accuracy. These results underscore the effectiveness of DiMoP in modeling the distribution of joint dynamics.

![](images/29b98da57ea59b30eea37380464150cf0ab35e4a6a92583cc4b30235bafb483d.jpg)  
(a) MAMP

![](images/ac5b7b60479c06dd80b85b5ddc9a06be0d59c3b3ff7b41f8fa95bfb319465d42.jpg)  
(b) MacDiff

![](images/1abba239c26859b4e58af741ed1a0a6d24cd782516d86a52dd63a38b0af2d45c.jpg)  
(c) DiMoP  
■ A2, ■ A3, ■ A11, ■ A12, ■ A14, ■ A16, ■ A17, ■ A29, ■ A30, ■ A31, ■ A33, ■ A34, ■ A36, ■ A41, ■ A44, ■ A45, and ■ A46.

Fig. 3: The t-SNE visualization of embeddings on the NTU RGB+D 60 X-view benchmark. The same set of actions with subtle motion is used. (The MAMP [17] and MacDiff [11] results are obtained by regenerating results with the provided code)  
![](images/d0f6ebd5119e3875838500a87df733aa49d47835f4de7cedd868560c93410c01.jpg)  
Fig. 4: Class-wise linear evaluation accuracy difference between DiMoP and MAMP on NTU RGB+D 60 X-sub.

7) Performance Analysis Among Action Groups: To provide a more fine-grained analysis, the 60 action classes are grouped according to their motion characteristics: fine-grained single-person actions (G1), large body-motion actions (G2), hand-object interactions (G3), two-person interactions (G4)

TABLE XII: Group-wise comparison between DiMoP and MAMP on NTU RGB+D 60 X-sub under linear evaluation.
<table><tr><td>Group</td><td>Imp./Drop</td><td>% imp</td><td>Mean ∆Acc.</td></tr><tr><td>G1: Fine-grained single-person</td><td>17/6</td><td>74%</td><td>+1.88</td></tr><tr><td>G2: Large body-motion</td><td>11/1</td><td>92%</td><td>+2.19</td></tr><tr><td>G3: Hand-object interaction</td><td>9/5</td><td>64%</td><td>+2.59</td></tr><tr><td>G4: Two-person interaction</td><td>10/1</td><td>91%</td><td>+3.57</td></tr><tr><td>G5: Brief-discriminative-frame†</td><td>10/6</td><td>63%</td><td>+2.16</td></tr><tr><td>Overall</td><td>47/13</td><td>78%</td><td>+1.90</td></tr></table>

Action IDs: G1: A3, A4, A18–A21, A28, A31, A33–A41, A44–A49; G2: A5–A9, A22–A24, A26, A27, A42, A43; G3: A1, A2, A10–A17, A25, A29, A30, A32; G4: A50–A60. G5: A3, A18–A21, A25, A28, A33, A37, A41, A44–A48, A57;

<sup>†</sup>G5 overlaps with other groups and is used only for temporal failure analysis; the extended table is provided in the supplementary material.

and weak-discriminative-frame actions (G5). As summarized in Table XII, DiMoP improves 47 out of 60 classes, corresponding to 78.3% of the actions, with an average gain of 1.90 percentage points over MAMP. Consistent improvements are observed across all groups, particularly for large body-motion actions and two-person interactions, where 91.7% and 90.9% of the classes are improved, respectively. These results indicate that progressive denoising facilitates the learning of both localized motion patterns and interaction-related dynamics. The relatively lower improvement ratios in hand-object and weak-discriminative-frame actions suggest that skeleton-only representations remain limited when object contact cues or short-lived discriminative frames are critical.

Figure 3 presents a t-SNE visualization of the learned embeddings for actions with subtle motion on the NTU RGB+D 60 dataset (X-Sub protocol). Compared with MAMP [17] and MacDiff [11], DiMoP produces compact intra-class clusters and stronger inter-class separability. The embedding distribution shows that DiMoP captures the joint motion distributions more coherently, modeling subtle motion variations while suppressing cross-class overlap.

## V. DISCUSSION AND CONCLUSION

A diffusion-based framework, DiMoP, was introduced for skeleton representation learning to model motion dynamics by progressively adding noise and denoising skeleton joints. By replacing static masking and handcrafted transformations with a stochastic, iterative denoising process, DiMoP effectively captures the uncertainty inherent in skeletal motion and enhances generalization. In addition, a pseudo-frame-based classifier was proposed to inject a discriminative structure without requiring manual labels. The proposed method achieved stateof-the-art performance on three standard benchmarks in action classification.

However, DiMoP might be limited when discriminative cues are brief, highly localized, or dependent on object/contact information. For instance, actions such as headache and neck pain share similar hand-to-head/neck movements, where only a few frames carry discriminative evidence. The pseudo-frame classifier promotes sequence-level temporal consistency, which stabilizes representations but may suppress such short-lived cues. Moreover, object-centric actions such as wearing shoes, typing on a keyboard, or playing with a phone may remain challenging because skeleton-only input does not explicitly encode object appearance or precise hand-object contact. It should be noted that these issues are not unique to DiMoP; they also exist in existing methods such as MAMP and MacDiff.

Future work will explore motion-adaptive diffusion scheduling and interaction between subjects. For instance, subtle moving joints will receive less noise or undergo slower denoising, while highly moving joints will retain full diffusion depth. This direction is expected to further improve the model’s ability to distinguish between complex and diverse motion patterns in real-world scenarios.

## REFERENCES

[1] Shengqin Jiang, Haokui Zhang, Yuankai Qi, and Qingshan Liu. Spatial-temporal interleaved network for efficient action recognition. IEEE Transactions on Industrial Informatics, 21(1):178–187, 2025.

[2] Shengqin Jiang, Yuankai Qi, Haokui Zhang, Zongwen Bai, Xiaobo Lu, and Peng Wang. D3d: Dual 3- d convolutional network for real-time action recognition. IEEE Transactions on Industrial Informatics, 17(7):4584–4593, 2021.

[3] Pichao Wang, Wanqing Li, Philip Ogunbona, Jun Wan, and Sergio Escalera. Rgb-d-based human motion recognition with deep learning: A survey. Computer Vision and Image Understanding, 171:118–139, 2018.

[4] Yuxuan Zhou, Xudong Yan, Zhi-Qi Cheng, Yan Yan, Qi Dai, and Xian-Sheng Hua. Blockgcn: Redefining topology awareness for skeleton-based action recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[5] Shanaka Ramesh Gunasekara, Wanqing Li, Jack Yang, and Philip. Ogunbona. Joint temporal pooling for improving skeleton-based action recognition. In 2023 International Conference on Digital Image Computing: Techniques and Applications (DICTA), 2023.

[6] Shanaka Ramesh Gunasekara, Wanqing Li, Jack Yang, and Philip Ogunbona. Asynchronous joint-based temporal pooling for skeleton-based action recognition. IEEE

Transactions on Circuits and Systems for Video Technology, pages 1–1, 2024.

[7] Yuxin Chen, Ziqi Zhang, Chunfeng Yuan, Bing Li, Ying Deng, and Weiming Hu. Channel-wise topology refinement graph convolution for skeleton-based action recognition. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 13339–13348, 2021.

[8] Lei Wang and Piotr Koniusz. 3mformer: Multi-order multi-mode transformer for skeletal action recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5620–5631, June 2023.

[9] Bruno Degardin, Vasco Lopes, and Hugo Proenc¸a. Fake it till you recognize it: Quality assessment for human action generative models. IEEE Transactions on Biometrics, Behavior, and Identity Science, 6(2):261–271, 2024.

[10] Shanaka Ramesh Gunasekara, Wanqing Li, Philip Ogunbona, and Jack Yang. Spatio-temporal joint density driven learning for skeleton-based action recognition. IEEE Transactions on Biometrics, Behavior, and Identity Science, pages 1–1, 2025.

[11] Lehong Wu, Lilang Lin, Jiahang Zhang, Yiyang Ma, and Jiaying Liu. Macdiff: Unified skeleton modeling with masked conditional diffusion. In Ales Leonardis,ˇ Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gul Varol, editors,¨ Computer Vision – ECCV 2024, pages 110–128, Cham, 2025. Springer Nature Switzerland.

[12] Wenhan Wu, Yilei Hua, Ce Zheng, Shiqian Wu, Chen Chen, and Aidong Lu. Skeletonmae: Spatial-temporal masked autoencoders for self-supervised skeleton action recognition. In 2023 IEEE International Conference on Multimedia and Expo Workshops (ICMEW), pages 224– 229, 2023.

[13] Lilang Lin, Jiahang Zhang, and Jiaying Liu. Actionlet-Dependent Contrastive Learning for Unsupervised Skeleton-Based Action Recognition . In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2363–2372, Los Alamitos, CA, USA, June 2023. IEEE Computer Society.

[14] Tianyu Guo, Hong Liu, Zhan Chen, Mengyuan Liu, Tao Wang, and Runwei Ding. Contrastive learning from extremely augmented skeleton sequences for selfsupervised action recognition. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pages 762–770, 2022.

[15] Bulat Khaertdinov, Stylianos Asteriadis, and Esam Ghaleb. Dynamic temperature scaling in contrastive self-supervised learning for sensor-based human activity recognition. IEEE Transactions on Biometrics, Behavior, and Identity Science, 4(4):498–507, 2022.

[16] Chen Zhan, Liu Hong, Guo Tianyu, Chen Zhengyan, Song Pinhao, and Tang Hao. Contrastive learning from spatio-temporal mixed skeleton sequences for selfsupervised skeleton-based action recognition. In arXiv, 2022.

[17] Yunyao Mao, Jiajun Deng, Wengang Zhou, Yao Fang, Wanli Ouyang, and Houqiang Li. Masked motion pre-

dictors are strong 3d action representation learners. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10147–10157. IEEE, 2023.

[18] Hong Yan, Yang Liu, Yushen Wei, Zhen Li, Guanbin Li, and Liang Lin. Skeletonmae: Graph-based masked autoencoder for skeleton sequence pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 5606–5618, October 2023.

[19] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are´ scalable vision learners. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16000–16009, June 2022.

[20] Pichao Wang, Zhaoyang Li, Yonghong Hou, and Wanqing Li. Action recognition based on joint trajectory maps using convolutional neural networks. In Proceedings of the 24th ACM international conference on Multimedia, pages 102–106, 2016.

[21] Pichao Wang, Wanqing Li, Chuankun Li, and Yonghong Hou. Action recognition based on joint trajectory maps with convolutional neural networks. Knowledge-Based Systems, 158:43–53, 2018.

[22] Yong Du, Yun Fu, and Liang Wang. Skeleton based action recognition with convolutional neural network. In 2015 3rd IAPR Asian Conference on Pattern Recognition (ACPR), pages 579–583, 2015.

[23] Yong Du, Wei Wang, and Liang Wang. Hierarchical recurrent neural network for skeleton based action recognition. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1110–1118, 2015.

[24] Yong Du, Yun Fu, and Liang Wang. Representation learning of temporal dynamics for skeleton-based action recognition. IEEE Transactions on Image Processing, 25(7):3010–3022, 2016.

[25] Woomin Myung, Nan Su, Jing-Hao Xue, and Guijin Wang. Degcn: Deformable graph convolutional networks for skeleton-based action recognition. IEEE Transactions on Image Processing, 33:2477–2490, 2024.

[26] Jeonghyeok Do and Munchurl Kim. Skateformer: skeletal-temporal transformer for human action recognition. In European Conference on Computer Vision, pages 401–420. Springer, 2025.

[27] Sijie Yan, Yuanjun Xiong, and Dahua Lin. Spatial temporal graph convolutional networks for skeletonbased action recognition. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications of Artificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence, AAAI’18/IAAI’18/EAAI’18. AAAI, 2018.

[28] Tianchen Li, Pei Geng, Xuequan Lu, Wanqing Li, and Lei Lyu. Skeleton-based action recognition through attention guided heterogeneous graph neural network. Knowledge-Based Systems, 309:112868, 2025.

[29] Ke Cheng, Yifan Zhang, Xiangyu He, Weihan Chen, Jian Cheng, and Hanqing Lu. Skeleton-based action recognition with shift graph convolutional network. In

2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 180–189, 2020.

[30] Tianchen Li, Pei Geng, Guohui Cai, Xinran Hou, Xuequan Lu, and Lei Lyu. Variation-aware directed graph convolutional networks for skeleton-based action recognition. Knowledge-Based Systems, 302:112319, 2024.

[31] Ugur Kilic, Ozge Oztimur Karadag, and Gulsah Tumuklu Ozyer. Agms-gcn: Attention-guided multi-scale graph convolutional networks for skeleton-based action recognition. Knowledge-Based Systems, 311:113045, 2025.

[32] Wangmeng Xiang, Chao Li, Yuxuan Zhou, Biao Wang, and Lei Zhang. Generative action description prompts for skeleton-based action recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10276–10285, 2023.

[33] Chiara Plizzari, Marco Cannici, and Matteo Matteucci. Skeleton-based action recognition via spatial and temporal transformer networks. Computer Vision and Image Understanding, 208:103219, 2021.

[34] Yaolin Zheng, Hongbo Huang, Xiuying Wang, Xiaoxu Yan, and Longfei Xu. Spatio-temporal fusion for human action recognition via joint trajectory graph. Proceedings of the AAAI Conference on Artificial Intelligence, 38(7):7579–7587, Mar. 2024.

[35] Yuxuan Zhou, Zhi-Qi Cheng, Chao Li, Yanwen Fang, Yifeng Geng, Xuansong Xie, and Margret Keuper. Hypergraph transformer for skeleton-based action recognition. arXiv preprint arXiv:2211.09590, 2022.

[36] Helei Qiu, Biao Hou, Bo Ren, and Xiaohua Zhang. Spatio-temporal tuples transformer for skeleton-based action recognition. ArXiv, abs/2201.02849, 2022.

[37] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[38] Linguo Li, Minsi Wang, Bingbing Ni, Hang Wang, Jiancheng Yang, and Wenjun Zhang. 3d human action representation learning via cross-view consistency pursuit. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4741– 4750, 2021.

[39] Wentao Zhu, Xiaoxuan Ma, Zhaoyang Liu, Libin Liu, Wayne Wu, and Yizhou Wang. Motionbert: A unified perspective on learning human motion representations. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15085–15099, October 2023.

[40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[41] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Pro-

ceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, ICML’15, page 2256–2265. JMLR.org, 2015.

[42] Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang. Ntu rgb+ d: A large scale dataset for 3d human activity analysis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1010– 1019, 2016.

[43] Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, Alex Kot, • Liu, and L.-Y Duan. Ntu rgb+d 120: A large-scale benchmark for 3d human activity understanding. In IEEE Transactions on Pattern Analysis and Machine Intelligence. Institute of Electrical and Electronics Engineers (IEEE), 2020.

[44] Chunhui Liu, Yueyu Hu, Yanghao Li, Sijie Song, and Jiaying Liu. Pku-mmd: A large scale benchmark for skeleton-based human action understanding. In Proceedings of the workshop on visual analysis in smart and connected communities, pages 1–8, 2017.

[45] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 6840–6851. Curran Associates, Inc., 2020.

[46] Fida Mohammad Thoker, Hazel Doughty, and Cees GM Snoek. Skeleton-contrastive 3d action representation learning. In Proceedings of the 29th ACM international conference on multimedia, pages 1655–1663, 2021.

[47] Boeun Kim, Hyung Jin Chang, Jungho Kim, and Jin Young Choi. Global-local motion transformer for unsupervised skeleton-based action learning. In Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part IV, page 209–225, Berlin, Heidelberg, 2022. Springer-Verlag.

[48] Yujie Zhou, Haodong Duan, Anyi Rao, Bing Su, and Jiaqi Wang. Self-supervised action representation learning from partial spatio-temporal skeleton sequences. In Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’23/IAAI’23/EAAI’23. AAAI Press, 2023.

[49] Haoyuan Zhang, Yonghong Hou, Wenjing Zhang, and Wanqing Li. Contrastive positive mining for unsupervised 3d action representation learning. In Shai Avidan, Gabriel Brostow, Moustapha Cisse, Giovanni Maria´ Farinella, and Tal Hassner, editors, Computer Vision – ECCV 2022, pages 36–51, Cham, 2022. Springer Nature Switzerland.

[50] Wanjiang Weng, Hongsong Wang, Junbo Wang, Lei He, and Guosen Xie. Usdrl: Unified skeleton-based dense representation learning with multi-grained feature decorrelation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

[51] Chuankun Li, Shuai Li, Yanbo Gao, Xingyu Gao, Ping

Chen, Jian Li, and Wanqing Li. Unsupervised feature enrichment and fidelity preservation learning framework for skeleton-based action recognition. IEEE Transactions on Circuits and Systemsfor Video Technology, pages 1–1, 2025.

[52] Jingyu Wang, Yin Nie, Tianyang Xia, Ying Wu, and Song-Chun Zhu. Cross-view action modeling, learning, and recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2649–2656, 2014.

[53] David J. Lerch, Zeyun Zhong, Manuel Martin, Michael Voit, and Jurgen Beyerer. Unsupervised 3d skeleton-¨ based action recognition using cross-attention with conditioned generation capabilities. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW), pages 202–211, 2024.

![](images/41c23b933f8d1e87cd9bb0d32c70d27e486f8ee9ffb9f30b9b95cd1df63904c9.jpg)

Shanaka Ramesh Gunasekara (Member, IEEE) received his PhD in computer science and engineering with the Advanced Multimedia Research Lab (AMRL), University of Wollongong, Australia, with a focus on human action recognition. He received the B.Sc. (hons) degree in electrical and electronic engineering from the University of Peradeniya, Sri Lanka. He is currently working as a postdoc research fellow at RMIT. His research interests include 3D computer vision, human motion analysis, signal processing, medical image analysis, and robotics.

![](images/560ccbf75f5154b9205e4a7bafb3ff2c3d6ee3dfa335984ea2fba4c98148181c.jpg)

Wanqing Li (M’97-SM’05) received his PhD in electronic engineering from the University of Western Australia. He was a Senior Researcher and later a Principal Researcher at the Motorola Research Lab in Sydney from 1998 to 2003, and a visiting researcher at Microsoft Research, USA, in 2008, 2010, and 2013. He is currently a Professor and Co-Director of the Advanced Multimedia Research Lab (AMRL), University of Wollongong, Australia. His research areas include machine learning, 3D computer vision, 3D multimedia signal processing, medical image analysis, natural language processing, and their applications. Dr. Li served as a Technical Program Co-Chair for IEEE ICME 2021 and has served as Co-Chair for many IEEE Workshops. He is an Associate Editor for IEEE Transactions on Image Processing and IEEE Transactions on Multimedia. He served as an Associate Editor for IEEE Transactions on Circuits and Systems for Video Technology from 2018 to 2021 and for the Journal of Visual Communication and Image Representation from 2016 to 2019.

![](images/b721a4b935fb1aa990c8e274a8556e4afe399ba460d3ca6975d76fa7489d44e9.jpg)

Nikalal Kaldera holds a Bachelor’s degree in Electrical and Electronic Engineering from the University of Peradeniya, Sri Lanka. He is currently pursuing a Ph.D. at the Advanced Multimedia Research Laboratory (AMRL), University of Wollongong, Australia, focusing on human action recognition. His research interests include computer vision, machine learning, data science, and signal/image processing..

![](images/f76fd665129eeb8b1d5402d74e068da69adcc4ddfce533e8d79c8e5fc6baa6fa.jpg)

Philip O. Ogunbona received the B.Sc. degree (with first class honours) in electronics and electrical engineering from the University of Ife, Nigeria, and the Ph.D. degree in electrical engineering from Imperial College London, U.K. He is a professor in computer science at the University of Wollongong, Australia. His research interests include signal and image processing, machine learning, computer vision and natural language processing. Professor Ogunbona is Fellow of the Australian Computer Society and a Life Senior Member of IEEE.

![](images/ca8986c4c04cf38fa02354b4cc76cb5f6cf874686cfff998533825bcabc0c8dd.jpg)

Jie (Jack) Yang is a lecturer of Big Data Analytics in the School of Computing and Information Technology at the University of Wollongong. His major research interests include Data Mining and Natural Language Processing. Dr. Yang has been the Chief Investigator for three research grants of Discovery/Linkage Project themes from the prestigious Australian Research Council (ARC) and has published over 70 articles.

# Supplementary Materials

## APPENDIX A THEORETICAL ANALYSIS

We provide a theoretical analysis showing that diffusion enables noise-dependent learning of motion modes, allowing simultaneous modeling of subtle and strong motion.

In masked autoencoders (MAE) [19], masked coordinates are directly regressed from visible ones by minimizing an MSE loss:

$$
\mathcal { L } _ { \mathrm { M A E } } = \mathbb { E } \big [ \| x _ { 0 } ^ { m } - f _ { \theta } ( x _ { 0 } ^ { v } ) \| _ { 2 } ^ { 2 } \big ] = \sum _ { i \in \mathrm { m a s k e d } } ( x _ { 0 , i } - \hat { x } _ { 0 , i } ) ^ { 2 } ,\tag{9}
$$

where $x _ { 0 } ^ { m }$ and $x _ { 0 } ^ { v }$ denote the masked and visible coordinates, $f _ { \theta }$ is the prediction model, and $\hat { x } _ { 0 , i }$ is the reconstructed value of the ith coordinate.

Even when $f _ { \theta }$ is optimal, the gradient scale driving learning is proportional to the reconstruction residual:

$$
\left\| \nabla _ { \theta } ( x _ { 0 , i } - \hat { x } _ { 0 , i } ) ^ { 2 } \right\| = 2 | x _ { 0 , i } - \hat { x } _ { 0 , i } | \left| \frac { \partial \hat { x } _ { 0 , i } } { \partial \theta } \right| .\tag{10}
$$

Thus, coordinates associated with strong motions yield larger gradients and dominate optimization, while subtle motions contribute less without explicit reweighting.

In contrast, the proposed framework performs diffusion in the latent embedding space while supervising denoising in coordinate-space motion. Let $S ~ \in ~ \mathbb { R } ^ { \mathbf { \bar { \ b { T } } } _ { i n } \times V \times \mathbf { \breve { C } } _ { i n } }$ denote the input skeleton sequence. After temporal segmentation and embedding via $g _ { 1 } ( \cdot )$ , the latent representation is obtained as

$$
\begin{array} { r } { x = g _ { 1 } ( S ) , \qquad x \in \mathbb { R } ^ { \tau \times V \times C } . } \end{array}\tag{11}
$$

With spatial and temporal positional encodings, the clean latent sample is defined as

$$
x _ { 0 } = x + p e ^ { s } + p e ^ { t } ,\tag{12}
$$

where $p e ^ { s } \in \mathbb { R } ^ { 1 \times V \times C }$ and $p e ^ { t } \in \mathbb { R } ^ { \tau \times 1 \times C }$ . A masking operation partitions $x _ { 0 }$ into visible and masked subsets, denoted by $x _ { 0 } ^ { v }$ and $x _ { 0 } ^ { m }$ , respectively.

In the forward diffusion process, only the masked tokens, $x _ { 0 } ^ { m }$ , are corrupted by Gaussian noise. However, for notational clarity, we denote the clean latent sample as x<sub>0</sub>

$$
\begin{array} { r } { x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \epsilon \sim \mathcal { N } ( 0 , I ) , } \end{array}\tag{13}
$$

where $\bar { \alpha } _ { t }$ controls the noise level at timestep t. The corresponding signal-to-noise ratio (SNR) is

$$
\mathrm { S N R } _ { t } = \frac { \bar { \alpha } _ { t } } { 1 - \bar { \alpha } _ { t } } .\tag{14}
$$

Early timesteps correspond to high SNR (low noise), while later timesteps correspond to low SNR (high noise).

The denoising decoder learns the conditional distribution $p ( x _ { 0 } ^ { m } \mid x _ { t } ^ { m } , x _ { 0 } ^ { v } )$ using u as conditioning. The full denoising function is expressed as

$$
\widehat { \dot { x } } _ { 0 } = D _ { \mathrm { D e n o i s e } } ( x _ { t } ^ { m } , t , u ) ,\tag{15}
$$

which predicts the motion of the clean sequence. Let the coordinate motion be defined by the temporal finite-difference operator

$$
\dot { x } _ { 0 } = D x _ { 0 } , \qquad D x = [ x ^ { \tau + 1 } - x ^ { \tau } ] .\tag{16}
$$

The training objective supervises motion reconstruction:

$$
\mathcal { L } _ { \mathrm { d e n o i s e } } = \mathbb { E } \left[ \| \dot { x } _ { 0 } - \widehat { \dot { x } } _ { 0 } \| _ { 2 } ^ { 2 } \right] .\tag{17}
$$

To analyze the denoising behavior, assume the clean latent tokens follow a Gaussian distribution:

$$
x _ { 0 } \sim { \mathcal { N } } ( \mu _ { x } , \Sigma _ { x } ) .\tag{18}
$$

The posterior mean of the clean sample given $x _ { t }$ is

$$
\operatorname { \mathbb { E } } [ x _ { 0 } \mid x _ { t } ] = \mu _ { x } + K _ { t } ( x _ { t } - { \sqrt { { \bar { \alpha } } _ { t } } } \mu _ { x } ) ,\tag{19}
$$

where

$$
K _ { t } = \sqrt { \bar { \alpha } _ { t } } \Sigma _ { x } \big ( \bar { \alpha } _ { t } \Sigma _ { x } + ( 1 - \bar { \alpha } _ { t } ) I \big ) ^ { - 1 } .\tag{20}
$$

To align the analysis with the motion-based supervision, we define the latent motion variable

$$
m _ { 0 } = D x _ { 0 } .\tag{21}
$$

Since $D$ is linear, $m _ { 0 }$ is Gaussian:

$$
m _ { 0 } \sim \mathcal { N } ( D \mu _ { x } , \Sigma _ { m } ) ,\tag{22}
$$

with motion covariance

$$
\Sigma _ { m } = D \Sigma _ { x } D ^ { \top } .\tag{23}
$$

The eigenvalues of $\Sigma _ { m }$ quantify motion strength: large eigenvalues correspond to strong motion modes, while small eigenvalues correspond to subtle motion modes.

Because $D$ is linear, the posterior mean of latent motion satisfies

$$
\mathbb { E } [ m _ { 0 } \mid x _ { t } ] = D \mathbb { E } [ x _ { 0 } \mid x _ { t } ] = D \mu _ { x } + D K _ { t } ( x _ { t } - \sqrt { \bar { \alpha } _ { t } } \mu _ { x } ) .\tag{24}
$$

The diffusion gain along a motion mode with eigenvalue λ is governed by

$$
k _ { t } ( \lambda ) = \frac { \sqrt { \bar { \alpha } _ { t } } \lambda } { \bar { \alpha } _ { t } \lambda + ( 1 - \bar { \alpha } _ { t } ) } .\tag{25}
$$

For strong motion modes satisfying

$$
\lambda \gg \frac { 1 - \bar { \alpha } _ { t } } { \bar { \alpha } _ { t } } ,\tag{26}
$$

the gain simplifies to

$$
k _ { t } ( \lambda ) \approx \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } ,\tag{27}
$$

indicating that strong motion modes remain identifiable even at high noise levels.

For subtle motion modes satisfying

$$
\lambda \ll \frac { 1 - \bar { \alpha } _ { t } } { \bar { \alpha } _ { t } } ,\tag{28}
$$

the gain becomes

$$
k _ { t } ( \lambda ) \approx { \frac { \sqrt { { \bar { \alpha } } _ { t } } \lambda } { 1 - { \bar { \alpha } } _ { t } } } , \qquad \mathrm { a s } { \bar { \alpha } } _ { t }  1 .\tag{29}
$$

Thus, subtle motion modes are most recoverable at early timesteps with high SNR.

Finally, we relate latent motion denoising to coordinatespace supervision. Using a first-order approximation of the decoder around a reference point x¯,

$$
h _ { \psi } ( x ) \approx h _ { \psi } ( \bar { x } ) + J _ { h } ( x - \bar { x } ) ,\tag{30}
$$

where $J _ { h }$ is the Jacobian of the decoder. The predicted motion can be locally written as

$$
\widehat { \dot { x } } _ { 0 } \approx c + J _ { h } x _ { 0 } ,\tag{31}
$$

where c is a constant. Substituting the posterior mean yields

$$
\mathbb { E } [ \dot { x } _ { 0 } \mid x _ { t } ] \approx c + J _ { h } \mathbb { E } [ m _ { 0 } \mid x _ { t } ] .\tag{32}
$$

This shows that coordinate motion prediction inherits the same SNR-dependent structure as latent motion denoising. Therefore, provided that the decoder preserves motion-relevant directions, the model learns different motion scales depending on the noise level.

Consequently, early diffusion steps with high SNR emphasize subtle motion modes, while later steps increasingly capture strong motion dynamics. Unlike direct masked regression, which is biased toward large residuals, latent diffusion with motion supervision distributes learning across the full spectrum of motion dynamics in a principled manner.

## APPENDIX B ADDITIONAL RESULTS

More experimental results are provided in this document.

## A. Dataset

1) NW-UCLA [52]: dataset consists of 1,494 action sequences performed by 10 subjects, recorded from three different viewpoints, and spanning 10 distinct action categories.

## B. More Comparison Results

1) Transfer Learning: The encoder and mapping layer are pre-trained on the NTU RGB+D [42] 60 and NTU RGB+D 120 [43] datasets separately using X-sub settings and are evaluated on the UCLA dataset using the supervised fine-tuning evaluation setting. The results are presented in Table XIII. DiMoP outperforms TAHAR [53] by 1.1 and 1.5 percentage points on the UCLA with pretraining on NTU RGB+D 60 and 120 datasets respectively.

TABLE XIII: Transfer learning evaluation of DiMoP on the UCLA dataset: Fine-tuning the pre-trained encoder from NTU RGB+D 60 and NTU RGB+D 120.
<table><tr><td rowspan=1 colspan=1>Models</td><td rowspan=1 colspan=1>To UCLA (%)NTU 60  NTU 120</td></tr><tr><td rowspan=1 colspan=1>TAHAR [53]</td><td rowspan=1 colspan=1>93.1       95.2</td></tr><tr><td rowspan=1 colspan=1>DiMoP (ours)</td><td rowspan=1 colspan=1>94.2      96.7</td></tr></table>

## C. More Ablation Results

1) Analysis of decoder configuration:: To further verify the source of DiMoP’s improvement, MacDiff’s [11] AdaLNbased conditional U-Net decoder was substituted into DiMoP while keeping the encoder, noise schedule, and training settings identical $( T = 3 0 0$ , linear schedule). The linear evaluation results are summarized in Table XIV. No performance gain was observed; accuracy slightly decreased under the same noise-prediction objective, and remained nearly unchanged when the motion-prediction target was applied. These outcomes indicate that DiMoP’s advantage arises primarily from its partial-state diffusion mechanism and motion-prediction objective, rather than from architectural or capacity differences in the decoder.

TABLE XIV: Linear Evaluation Results on the NTU RGB+D 60 X-Sub Dataset After Replacing DiMoP’s Decoder with MacDiff’s AdaLN-Based Decoder.
<table><tr><td>Model</td><td>Acc(%)</td></tr><tr><td>MacDiff (reproduced)</td><td>86.03</td></tr><tr><td>DiMoP (ours) DiMoP + MacDiff decoder (Noise)</td><td>86.76</td></tr><tr><td>DiMoP + MacDiff decoder (Motion)</td><td>85.83 86.61</td></tr></table>

2) Analysis of training stability:: Training stability was evaluated under varying diffusion hyperparameter settings, including prediction targets, noise schedules, and diffusion step counts. During pre-training, validation was performed every 50 epochs on a held-out set, and both training and validation losses were recorded. The corresponding loss curves are shown in Figure 5. In all cases, convergence of both training and validation losses was observed, demonstrating that stable optimization was achieved and robustness to different diffusion configurations was maintained.

3) Analysis of computational complexity:: We further analyze the computational overhead of DiMoP compared to the baseline MAMP [17]. During pre-training on NTU RGB+D 60 (X-View) with four NVIDIA P100 GPUs, DiMoP incurs slightly higher complexity due to the additional mapping MLP, pseudo-classifier, and denoising decoder. As shown in Table XV, the parameter count increases from 8.79M to 9.69M and FLOPs from 5.45G to 6.78G. Importantly, these components are only used in pre-training and are not involved during downstream inference.

For downstream tasks, only the encoder and mapping layer are required. Table XV reports the parameter count and inference time per iteration for linear evaluation on two P100 GPUs. DiMoP requires 0.38M parameters and 0.413s per iteration, compared to 0.27M and 0.342s for MAMP. The overhead is marginal and remains practically manageable, while offering consistent accuracy improvements.

![](images/ce60569e4b7bb8113b2367ed0ecae2275e3176036d360953dd096c1e9acb15d8.jpg)  
(a) Varying the prediction target.

![](images/6945a6401c5d9158afaea4b2eeb147a3d1e6aa8a8a71e99dc118ea17912bdfcb.jpg)

(b) Varying the noise scheduling.  
![](images/909e971755bb3ab25097bdad5aa4bf660bd7aa5b797367596a05d81fdae86cd9.jpg)  
(c) Varying the Diffusion steps T.  
Fig. 5: Training and validation loss curves obtained by varying the Diffusion hyperparameters.

TABLE XV: Computational cost comparison of MAMP and DiMoP during pre-training and linear evaluation.
<table><tr><td></td><td colspan="2">Pre-training</td><td colspan="2">Linear Evaluation</td></tr><tr><td>Model</td><td>Params</td><td>FLOPs (G)</td><td>Params</td><td>Time (s/iter)</td></tr><tr><td>MAMP [17] (baseline)</td><td>8.79M</td><td>5.45</td><td>0.27M</td><td>0.342</td></tr><tr><td>DiMoP</td><td>9.69M</td><td>6.78</td><td>0.38M</td><td>0.413</td></tr></table>

4) Statistical Significance Analysis:: To further assess statistical significance, the linear evaluation was repeated five times for both MAMP and DiMoP on NTU RGB+D 60 Xsub and X-view. In each trial, a linear classifier was trained on top of the frozen encoder with random initialization. A paired two-tailed t-test was then applied at α = 0.05. The null hypothesis was defined as no significant difference between the two methods, i.e., $H _ { 0 } : \mu _ { \mathrm { D i M o P } } = \mu _ { \mathrm { M A M P } }$ , while the alternative hypothesis was defined as a significant difference, i.e., $H _ { 1 } : \mu _ { \mathrm { D i M o P } } \neq \mu _ { \mathrm { M A M P } } .$

TABLE XVI: Paired two-tailed t-test results between MAMP and DiMoP on NTU RGB+D 60.
<table><tr><td>Protocol</td><td>μMAMP</td><td>µDiMoP</td><td>t-value</td><td> $\overline { { t _ { \mathrm { c r i t } } } }$ </td><td>Decision</td></tr><tr><td>X-sub</td><td>84.95</td><td>86.79</td><td>5.636</td><td>2.776</td><td>Reject  $\overline { { H _ { 0 } } }$ </td></tr><tr><td>X-view</td><td>89.14</td><td>91.91</td><td>8.154</td><td>2.776</td><td>Reject  $H _ { 0 }$ </td></tr></table>

With four degrees of freedom, the computed t-values exceed the critical value for both protocols. Therefore, $H _ { 0 }$ is rejected, indicating that the performance difference between DiMoP and MAMP is statistically significant. Since DiMoP obtains higher mean accuracy in both protocols, the significant difference supports the superiority of DiMoP over MAMP under this evaluation.

5) Reproducibility of Baseline Results:: The two most closely related publicly available baselines, MAMP [17] and MacDiff [11], were reproduced using their released implementations. As shown in Table XVII, the reproduced linear evaluation results are highly consistent with the originally reported values, with only minor deviations caused by training stochasticity.

TABLE XVII: Linear evaluation performance of reproduced baselines. <sup>∗</sup> denotes reproduced results.
<table><tr><td rowspan="2">Model</td><td colspan="2">NTU 60</td><td colspan="2">NTU 120</td><td>PKU</td></tr><tr><td>X-sub</td><td>X-view</td><td>X-sub</td><td>X-set</td><td>Part I</td></tr><tr><td>MAMP</td><td>84.90</td><td>89.10</td><td>78.60</td><td>79.10</td><td>92.20</td></tr><tr><td>MAMP*</td><td>84.84</td><td>89.13</td><td>78.49</td><td>78.98</td><td>92.24</td></tr><tr><td>MacDiff</td><td>86.40</td><td>91.00</td><td>79.40</td><td>80.20</td><td>92.80</td></tr><tr><td>MacDiff*</td><td>86.43</td><td>90.95</td><td>79.34</td><td>80.13</td><td>92.74</td></tr></table>

For semi-supervised evaluation, the exact MacDiff config-

TABLE XVIII: Group-wise accuracy improvement of DiMoP over MAMP on NTU RGB+D 60 X-sub.
<table><tr><td rowspan=1 colspan=1>Group</td><td rowspan=1 colspan=1>Dominant motion cue</td><td rowspan=1 colspan=1>Average % Gain</td></tr><tr><td rowspan=1 colspan=1>Subtle motion</td><td rowspan=1 colspan=1>Localized   low-amplitudehand, arm, head, and object-related motion</td><td rowspan=1 colspan=1>+5.77</td></tr><tr><td rowspan=1 colspan=1>Strong motion</td><td rowspan=1 colspan=1>Global displacement, pos-ture transition, and interac-tion dynamics</td><td rowspan=1 colspan=1>+4.31</td></tr></table>

Subtle: A2, A3, A11, A12, A14, A16, A17, A29, A30, A31, A33, A34, A36, A41, A44, A45, A46. Strong: A6, A7, A8, A9, A15, A20, A22, A24, A26, A27, A40, A50, A51, A52, A55, A57, A58.

uration was not provided with the released code. Therefore, MacDiff was re-evaluated under the same data split, label ratio, training setting, and evaluation protocol used for DiMoP to ensure a controlled comparison.

6) Performance on actions with subtle vs strong Motion:: Table XVIII provides a group-level interpretation of the classwise results. The larger gain on subtle-motion actions suggests that DiMoP better preserves localized, low-amplitude cues that are easily dominated by stronger body movements in direct masked reconstruction. This indicates a key limitation of MAE-style objectives: reconstruction can be biased toward high-energy motion components, whereas progressive denoising exposes the model to fine deviations around the clean motion manifold at high-SNR stages. The gain on strongmotion actions further shows that DiMoP does not improve subtle actions at the expense of large-scale dynamics, but also retains global displacement, posture transitions, and interaction patterns through later denoising stages. Therefore, the observed improvements are not merely class-wise fluctuations; they indicate that DiMoP learns motion representations across multiple spatial and temporal scales.

To further examine whether the improvements are biased toward high-motion classes, class-wise motion magnitude was computed as the average frame-wise joint displacement normalized by body size. The body size was estimated using the spine-to-neck bone length, which provides a stable torsobased scale factor and reduces the influence of inter-subject size variation. Fig. 6 plots the normalized motion magnitude against the relative percentage improvement of DiMoP over MAMP.

The scatter distribution shows that DiMoP improves classes across a broad range of normalized motion magnitudes, including several low-motion subtle actions. This indicates that the proposed diffusion-driven learning does not merely benefit high-amplitude actions, but also enhances weak and finegrained motion representations. Together with the group-wise analysis, these results suggest that DiMoP functions as a multiscale motion learner: high-SNR denoising stages preserve subtle local deviations, while later stages maintain robustness to large-scale temporal and spatial dynamics.

7) Group-wise and Motion-magnitude Analysis:: We analyze how DiMoP improves performance on actions with different characteristics in comparison with the baseline MAMP. In particular, the 60 actions are divided into 5 groups (nonexclusively): G1 - Fine-grained single-person actions; G2 - Large body-motion actions; G3 - Hand-object interaction actions; G4 - Two-person interaction actions; G5 - Weakdiscriminative-frame actions. Please note that the G5 overlaps with other groups, as some actions share multiple characteristics.

![](images/d31597978a4283c3ecf622601a448ca10d0a5055fc62741b190b4ae5fda08209.jpg)  
Fig. 6: Class-wise normalized motion magnitude versus relative percentage improvement of the actions Subtle and strong motions with DiMoP over MAMP on NTU RGB+D 60 X-sub.

![](images/b4eff5be119bcc9c0e50054a9ef3840372272dc37c3d2068966c8acdfd882231.jpg)  
Fig. 7: Group level analysis of normalized class-wise motion magnitude against the relative percentage improvement of DiMoP over MAMP on NTU60 X-sub dataset under linear evaluation.

As shown in Table XIX, positive average gains were obtained across all motion groups, indicating that the improvements were not restricted to isolated classes. Overall, DiMoP improves 47/60 (78%) classes and achieves positive average gains (1.9 percentage points) across all groups of actions. The gains are more evident in fine-grained actions, handobject interaction actions, and two-person interaction actions, with 74%, 92%, and 91% of actions improved, respectively. This supports the claim that progressive denoising captures both localized and interaction-related motion dynamics. As expected, the two types of actions, hand-object interaction and weak discriminative frame, have the lowest percentage of the actions being improved. Furthermore, the Figure 7 presents the normalized class-wise motion magnitude against the relative percentage improvement of DiMoP over MAMP.

TABLE XIX: Performance improvement on different groups of actions under linear evaluation between DiMoP and MAMP on the NTU RGB+D 60 X-sub dataset and the analysis of probable contributing factors provided by the proposed method.
<table><tr><td>Group</td><td>Action IDs</td><td>Improved/Dropped actions (% Improved)</td><td>Mean ∆Acc.</td><td>Main Insight</td></tr><tr><td>G1: Fine-grained single- person actions</td><td>A3, A4, A18–A21, A28, A31, A33– A41, A44–A49</td><td>17/6(74%)</td><td>+1.88</td><td>Localized hand, arm, head, and object- related motions were better captured, while drops mainly occurred when cues were confined to similar body regions.</td></tr><tr><td>G2: Large body-motion ac- tions</td><td>A5–A9, A22–A24, A26, A27, A42, A43</td><td>11/1 (92%)</td><td>+2.19</td><td>Global displacement, posture transitions, and strong temporal dynamics were con- sistently preserved.</td></tr><tr><td>G3: Hand-object interac- tion actions</td><td>A1, A2, A10–A17, A25, A29, A30, A32</td><td>9/5(64%)</td><td>+2.59</td><td>Local-to-mid-level hand motion was im- proved, although skeleton-only input re- mained limited when object/contact cues</td></tr><tr><td>G4: Two-person interac- tion actions</td><td>A50-A60</td><td>10/1 (91%)</td><td>+3.57</td><td>were essential. Interaction-related motion cues were effectively captured, although explicit inter-subject relation modeling may fur-</td></tr><tr><td>G5: Brief-discriminative- frame actions†</td><td>A3, A18–A21, A25, A28, A33, A37. A41, A44–A48, A57</td><td>10/6(63%)</td><td>+2.16</td><td>ther improve performance. Actions with short-lived discriminative cues remained challenging when most frames contained neutral or visually sim-</td></tr><tr><td>Overall</td><td>A1-A60</td><td>47/13 (78%)</td><td>+1.90</td><td>ilar poses. Most classes were improved, supporting the effectiveness of diffusion-based mo- tion distribution learning.</td></tr></table>

<sup>†</sup>The brief-discriminative-frame group overlaps with other semantic groups and is used only to analyze temporal failure cases.

8) Effect of Increasing the Temporal Smoothness Weight:: To examine whether the value consistency loss can be replaced by a larger temporal smoothness weight, we conducted an additional ablation on NTU RGB+D 60 X-sub under linear evaluation. The denoising loss was kept unchanged, while λ was increased without using $\mathcal { L } _ { \mathit { c o n s i s t e n c y } } .$

TABLE XX: Effect of increasing λ without L on NTU RGB+D 60 X-sub under linear evaluation.
<table><tr><td>α</td><td> $\bar { \lambda }$ </td><td> $\gamma$  Acc. (%)</td></tr><tr><td>1</td><td>0.3 0</td><td>85.33</td></tr><tr><td>1</td><td>1.0 0</td><td>84.92</td></tr><tr><td>1</td><td>3.0 0</td><td>84.59</td></tr><tr><td>1</td><td>5.0 0</td><td>83.46</td></tr><tr><td>1</td><td>0.3 1.0</td><td>86.48</td></tr></table>

As shown in Table XX, increasing λ without $\mathcal { L } _ { \mathit { c o n s i s t e n c y } }$ does not recover the performance of the full objective. Instead, large λ values progressively degrade accuracy, indicating that excessive temporal smoothing over-regularizes the pseudo-classifier and weakens representation learning. This confirms that $\mathcal { L } _ { s m o o t h }$ and $\mathcal { L } _ { { c o n s i s t e n c y } }$ are complementary: $\mathcal { L } _ { s m o o t h }$ regularizes local frame-to-frame transitions, whereas $\mathcal { L } _ { \mathit { c o n s i s t e n c y } }$ imposes global sequence-level agreement.