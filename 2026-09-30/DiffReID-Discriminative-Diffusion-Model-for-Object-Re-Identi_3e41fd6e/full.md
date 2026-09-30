# DiffReID: Discriminative Diffusion Model for Object Re-Identification

Yingquan Wang, Pingping Zhang<sup>∗</sup>, IEEE Member, Dong Wang, IEEE Member, Huchuan Lu, IEEE Fellow

Abstract—As a fundamental image processing task, object Re-Identification (ReID) aims to retrieve objects across nonoverlapping cameras. Recently, with the development of deep learning, significant advancements have been made in object ReID. However, most existing methods suffer from generalization due to the limited size and diversity of ReID datasets. Meanwhile, current models tend to focus on extracting semantic patterns rather than learning identity-aware feature distributions. To address these issues, we propose a novel feature learning framework named DiffReID for object ReID. It leverages a discriminative diffusion model to gradually learn identity-aware distributions and generate identity-invariant features. More specifically, with the Contrastive Language-Image Pre-training (CLIP) model, we first obtain identity-aware text features by prompt tuning. Then, we propose a Vision-guided Noise Generator (VNG) to initialize probabilistic noises and gradually corrupt identity-aware text features. Afterwards, we take visual features as conditions and propose a Light Weight Denoiser (LWD) to denoise the corrupted text features step-by-step for identity-aware distribution learning. To obtain discriminative features, we further generate identityinvariant guided features from randomly sampling visual-guided noises. Finally, we propose a Mutual Enhancement Constraint (MEC) to facilitate mutual learning between visual features and guided features to enhance the representation robustness and discrimination. Extensive experiments on five object ReID benchmarks demonstrate that our method shows better results than most state-of-the-art methods. The source code is available at https://github.com/AWangYQ/DiffReID.

Index Terms—Object Re-identification, Diffusion Model, Contrastive Language-Image Pre-training, Domain Generalization.

## I. INTRODUCTION

O <sup>BJECT</sup> <sup>Re-Identification</sup> <sup>(ReID)</sup> <sup>aims</sup> <sup>to</sup> <sup>match</sup> <sup>objects</sup>across non-overlapping cameras. As a fundamental im- across non-overlapping cameras. As a fundamental image processing task, it plays an important role in many real-world applications, including societal security, intelligent surveillance, mobile robotics and human-computer interaction. The key of ReID models is to extract identity-invariant features in complex visual variations, such as illumination changes, cross-scale resolutions and different occlusions. Although achieving great successes, most existing methods [1]–[5] suffer from generalization due to the limited size and diversity of ReID datasets. Meanwhile, current methods tend to focus on extracting semantic patterns rather than learning identity-aware feature distributions.

Recently, diffusion models [6] have gained more attention for their superior ability to model complex data distributions in many image generation tasks [7]. As a result, diffusion models can learn discriminative visual representations. Thus, researchers have begun to introduce diffusion models into object ReID [8]–[10]. Currently, two types of diffusion modelbased frameworks have been proposed for object ReID. As shown in Fig. 1(a), one type directly unifies feature extraction and denoising within a backbone pipeline [8]. However, the layer-wise injection of Gaussian noises may weaken the representational ability of hierarchical features. On the other hand, as shown in Fig. 1(b), several works [11]–[13] have demonstrated that intermediate features of pre-trained diffusion models contain rich semantic information, making them well-suited for downstream discriminative tasks. However, pre-trained diffusion models typically rely on complex network architectures and large timesteps, significantly increasing inference time and computational cost.

![](images/82ec04a68fd2906e030da32091e5a3a9d5aecc9a9b03016df8245de82cac6414.jpg)  
Fig. 1: Different diffusion model frameworks for object ReID. (a) The refinement diffusion model framework unifies feature extraction and denoising within a backbone pipeline. (b) The restoration diffusion model framework utilizes intermediate features from diffusion models for object representation. (c) Our discriminative diffusion model framework generates identity-invariant guided features and facilitates mutual learning between guided features and visual features.

To address the above issues, we propose a novel feature learning framework named DiffReID for high-performance object ReID. Different from previous diffusion model-based ReID methods, our framework constructs a discriminative feature distribution learning process in a compact latent space. As shown in Fig. 1(c), it leverages diffusion models to learn identity-aware feature distributions and generate identityinvariant guided features for object representation learning. Specifically, we introduce three key modules: the Visionguided Noise Generator (VNG) for noise initialization, the

Light Weight Denoiser (LWD) for feature generation, and the Mutual Enhancement Constraint (MEC) for feature refinement. They are jointly designed to support more discriminative representation learning. Technically, inspired by the strong vision-text understanding ability of diffusion models, we first obtain identity-aware text features via prompt tuning. As demonstrated in [14], noise initialization plays a crucial role in diffusion models. Thus, we utilize VNG to take visual features as priors and sample vision-aware noises to gradually corrupt identity-aware text features. Afterwards, we propose LWD to model the identity-aware distribution. The LWD allows the model to predict noises and generate identity-invariant guided features. Finally, we propose MEC to facilitate mutual learning between visual features and guided features to enhance the representation robustness and discrimination. Experiments on five object ReID benchmarks demonstrate that our method shows better results than most state-of-the-art methods.

In summary, our main contributions are as follows:

• We propose a novel feature learning framework named DiffReID for high-performance object ReID. It leverages the underlying ability of diffusion models to obtain discriminative features while maintaining efficient inference.

• We introduce three key modules to facilitate discriminative feature extraction: a Vision-guided Noise Generator (VNG) for noise initialization, a Light Weight Denoiser (LWD) for feature generation, and a Mutual Enhancement Constraint (MEC) for feature refinement.

• Experiments on five object ReID benchmarks demonstrate that our method achieves outstanding performance.

## II. RELATED WORK

## A. Diffusion Models for Representation Learning

Diffusion models [6] belong to a class of generative models. They have emerged as mainstream generative methods due to their exceptional ability in modeling complex data distributions. Recent studies have explored how diffusion models can be adapted for discriminative tasks [14], [15]. In fact, current works can be broadly grouped into two categories. One category of methods directly leverages pre-trained diffusion models (e.g., Stable Diffusion [16] and Imagen [17]) as feature extractors. For example, Li et al. [18] show that diffusion models can be leveraged to perform zero-shot classification without any additional training. Soumik et al. [12] and Dahye et al. [13] further extract discriminative intermediate features from diffusion models for robust segmentation and classification. While pre-trained diffusion models show great potential for discriminative tasks, their complex denoisers and large timestep requirements lead to high computational cost and long inference time. Another category of methods focuses on designing task-specific denoising models that learn dataset-specific distributions to generate both diverse and discriminative features. For instance, Du et al. [19] employ diffusion models for class-wise prototype feature generation, significantly improving few-shot classification performance. Li et al. [14] introduce diffusion models for continuous distribution transformation across object, image, and text latent spaces. These methods utilize denoisers to model task-specific data distributions, effectively balancing feature generation and computational efficiency. However, most of them naively utilize diffusion models for diverse image generation or general feature extraction. In contrast, our method constructs a discriminative feature distribution learning process for diffusion models. It can effectively utilize fine-grained conditions to generate more discriminative features.

## B. Image-based Object Re-Identification

Early object ReID methods predominantly leverage Con volutional Neural Networks (CNNs) to extract fine-grained feature representations [20], [21]. However, the inherent locality of CNNs limits their ability to capture long-range dependencies. To this end, recent studies have integrated at tention mechanisms [22] for robust object ReID. For example, Zhang et al. [23] construct a global pixel attention map to obtain more robust object representations. He et al. [1] introduce a Transformer-based backbone to capture global de pendencies, achieving impressive performance in object ReID. Afterwards, many Transformer-based ReID models [24]–[28] are proposed. However, image-only methods easily overfit to salient regions while neglecting essential semantic informa tion. Therefore, many researchers [2], [3], [29] have incorporated large vision-language models into object ReID. Besides, recent studies have also explored the model generalization. For example, Li et al. [30] investigate domain generalization object ReID by leveraging cross-camera unpaired samples. Zhang et al. [31] propose heterogeneous expert collaborative learning for weakly supervised visible-infrared person ReID. Zhang et al. [32] utilize a dual-granularity cross-modal identity association for text-based object ReID. Li et al. [33] improve clustering-based unsupervised object ReID via feature calibration. However, object ReID remains constrained by the limited availability of large-scale training datasets. To mitigate this issue, recent works [10], [34] leverage diffusion models to enrich training data. For example, Si et al. [34] introduce a diffusion model-based inpainting method to generate clothingchange samples. Similarly, Niu et al. [10] leverage diffusion models to construct large-scale person ReID datasets. However, simple image generation inevitably causes the loss of identity information. To this end, some researchers leverage diffusion models for both feature extraction and image generation. For example, Wang et al. [8] integrate feature extraction and denoising into a single image encoder. Additionally, Jia et al. [35] utilize diffusion models to learn the distribution of bounding boxes and ReID features, conditioned on the provided visual features. Unlike existing works that solely utilize generated features for objects, we explore the mutual learning between generated features and real visual features for more robust and discriminative representations.

## III. OUR PROPOSED METHOD

As illustrated in Fig. 2, our framework leverages a discriminative diffusion model to gradually learn identity-aware distributions and generate identity-invariant features. More specifically, with the CLIP model [36], we first obtain identityaware text features by prompt tuning. Then, we propose a

![](images/1aa0cd059a74e8e1e99d37058c828c53331fa91d69789dfd9dd4467081076ce3.jpg)  
Fig. 2: Illustration of our proposed framework. With the CLIP model, the framework first generates identity-aware text features by prompt tuning. Then, the Vision-guided Noise Generator (VNG) is used to initialize probabilistic noises and gradually corrupt identity-aware text features. Afterwards, the Light Weight Denoiser (LWD) is used to denoise the corrupted text feature for identity-aware distribution learning. To train the framework, the Mutual Enhancement Constraint (MEC) is performed to facilitate mutual learning between visual features and guided features.

Vision-guided Noise Generator (VNG) to initialize probabilistic noises and gradually corrupt identity-aware text features. Afterwards, we propose a Light Weight Denoiser (LWD) to denoise the corrupted text features for identity-aware distribution learning. Finally, we propose a Mutual Enhancement Constraint (MEC) to facilitate mutual learning between visual features and guided features to enhance the representation robustness and discrimination. We describe the above modules in the following sections.

## A. Overview of Our Framework

The overall framework adopts a two-stage training procedure. In the first stage, we aim to obtain identity-aware text features by prompt tuning. As shown in Fig. 2(a), with the CLIP model, we freeze the pre-trained text encoder and image encoder, only optimizing prompt tokens. Inspired by [2], these prompt tokens are passed through the text encoder with the template “A photo of a [x] [x] [x] [x] person/vehicle”. To obtain identity-aware text features, we employ the following bidirectional contrastive losses:

$$
L _ { t 2 i } ( y ) = - \frac { 1 } { | P ( y ) | } \sum _ { p \in P ( y ) } \log \frac { e x p ( f _ { v } ^ { p } , f _ { d } ^ { y } ) } { \sum _ { b = 1 } ^ { B } e x p ( f _ { v } ^ { a } , f _ { d } ^ { y } ) } ,\tag{1}
$$

$$
L _ { i 2 t } ( y ) = - \frac { 1 } { | P ( y ) | } \sum _ { p \in P ( y ) } \log \frac { e x p ( f _ { d } ^ { p } , f _ { v } ^ { y } ) } { \sum _ { b = 1 } ^ { B } e x p ( f _ { d } ^ { a } , f _ { v } ^ { y } ) } ,\tag{2}
$$

where $P ( y )$ is the image set of the same identity y. B is the batch size. $f _ { d }$ and $f _ { v }$ are text and visual features, respectively.

In the second stage, we feed person images into the learnable image encoder to extract visual tokens $\mathbf { \bar { \mathcal { F } } } \in \mathbb { R } ^ { N \times D }$ . They are sent to the VNG and LWD for noise initialization and feature generation, respectively. It involves two key processes: a forward noising process and a backward denoising process. Unlike most diffusion models [35], we integrate textual semantics into the denoising process. Specifically, as shown in the red line of Fig. 2, the forward noising process gradually adds visual-guided noises ε to the text feature $f _ { d }$ over $T$ timesteps. It should be noted that VNG generates visual-guided noises based on visual priors. At each timestep, visual-guided noises are injected with a timestep-dependent variance $\alpha _ { t } \mathrm { : }$

$$
\begin{array} { r } { q ( \bar { f } _ { t } | f _ { d } ) = \mathcal N ( \bar { f } _ { t } ; \sqrt { \bar { \alpha } _ { t } } f _ { d } , ( 1 - \bar { \alpha } _ { t } ) I ) , } \end{array}\tag{3}
$$

$$
\bar { f } _ { t } = \sqrt { \bar { \alpha } _ { t } } f _ { d } + \sqrt { 1 - \bar { \alpha } _ { t } } \varepsilon ,\tag{4}
$$

where $\textstyle { \bar { \alpha } } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i }$ . This process ultimately transforms $f _ { d }$ into the final state $\bar { f } _ { T }$ , which closely approximates its visualaware Gaussian distribution ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ . Noting that, $\mu$ and $\sigma ^ { 2 }$ are obtained by the VNG. As shown in the blue line of Fig. 2, we then treat the visual feature as a condition to the backward denoising process. It aims at gradually denoising $p ( { \bar { f } } _ { T } )$ to $q ( { \bar { f } } _ { 0 } )$ as follows:

$$
p _ { \theta } ( \bar { f } _ { t - 1 } | \bar { f } _ { t } , \hat { c } _ { t } ) = \mathcal { N } \Big ( \bar { f } _ { t - 1 } ; \mu _ { \theta } ( \bar { f } _ { t } , \hat { c } _ { t } , t ) , \Sigma _ { \theta } ( \bar { f } _ { t } , \hat { c } _ { t } , t ) \Big ) ,\tag{5}
$$

where $p _ { \theta } ( \bar { f } _ { t - 1 } \ | \ \bar { f } _ { t } , \hat { c } _ { t } )$ is the backward denoising distribution. $f _ { d } ~ \sim ~ q ( \bar { f } _ { 0 } )$ , θ denotes learnable parameters, and $\hat { c } _ { t }$ is a timestep-specific condition. To obtain guided features, our framework first generates an initial noise state ${ \bar { f } } _ { T } ,$ and then progressively samples $\bar { f } _ { t - 1 }$ from $p _ { \theta } ( \bar { f } _ { t - 1 } \ \mid \ \bar { f } _ { t } , \hat { c } _ { t } )$ at each denoising timestep $h \in \{ 1 , \ldots , T \}$ . After all denoising timesteps, the final denoised state $\bar { f } _ { 0 }$ is taken as the guided feature g. Here, t and h are the numbers of noising and denoising steps, respectively. They satisfy $T = t + h$ . Note that, the above diffusion process is performed in the latent feature space rather than the image space. It avoids reconstructing high-dimensional pixel-level details and provides a more compact semantic guidance for identity-aware distribution learning. Moreover, we perform the diffusion process on identity-aware text features rather than visual features. In fact, visual features may contain appearance-biased cues, such as clothing colors, bags, or backgrounds, while identityaware text features provide a compact semantic distribution aligned with identity labels. Therefore, with visual features as conditions and identity-aware text features as diffusion targets, our framework can generate identity-aware guided features that preserve visual discrimination while benefiting from semantic guidance.

## B. Vision-guided Noise Generator

As demonstrated in [14], the noise initialization significantly affects the generation process of diffusion models. In our framework, the initialized noise is used to corrupt identityaware text features and determines the starting state of the denoising process. Therefore, the quality of initial noises directly influences whether the guided features can preserve identity-related information. However, most existing methods [6], [19] sample noises directly from a standard Gaussian distribution. Such randomly sampled noises are sampleagnostic and contain no visual prior related to the input image. They may introduce perturbations inconsistent with the object identity and degrade the discrimination of generated features. To address this issue, we introduce the Vision-guided Noise Generator (VNG) to estimate the mean and variance of noise distributions, as shown in Fig. 2(b). Instead of using a fixed Gaussian prior, our VNG estimates the mean and variance of the noise distribution $p _ { \phi } ( { \bar { f } } _ { T } | { \hat { f } } )$ based on the visual feature ${ \hat { f } } .$ The noise is then sampled from this learned distribution, making it more adaptive for feature generation. In particular, we use two fully connected layers to transform the visual feature $\hat { f }$ into the mean $\mu$ and variance $\sigma ^ { 2 }$

$$
\mu = W _ { \mu } \hat { f } + b _ { \mu } ,\tag{6}
$$

$$
\sigma ^ { 2 } = W _ { \sigma } \widehat { f } + b _ { \sigma } ,\tag{7}
$$

where $W _ { \mu } , ~ W _ { \sigma } , ~ b _ { \mu }$ and $b _ { \sigma }$ are learnable parameters. Then, we sample initial visual-guided noises from the distribution $p _ { \phi } ( { \bar { f } _ { T } } | \hat { f } )$ using the reparameterization trick [37]:

$$
\varepsilon = \mu + \sigma \otimes \epsilon ,\tag{8}
$$

where $\epsilon \sim \mathcal { N } ( 0 , I )$ and ⊗ denotes the element-wise multiplication. To align with the distribution assumption in diffusion models [6], we impose a Kullback-Leibler (KL) divergence regularization to constrain $p _ { \phi } ( { \bar { f } } _ { T } | { \hat { f } } )$ toward a standard Gaussian distribution. With the proposed VNG, our framework can take visual features as priors and gradually corrupt identityaware text features.

## C. Light Weight Denoiser

Many works [12], [13], [38] have demonstrated that diffusion models can learn comprehensive data distributions, improving the robustness of generated features. However, applying complex diffusion models to discriminative tasks presents two major challenges: (a) the inconsistency between training and inference processes; (b) the high computational cost during inference. To address these issues, we propose the Light Weight Denoiser (LWD) to denoise identity-aware text features, while maintaining computational efficiency. As shown in Fig. 2(c), LWD utilizes several Multilayer Perceptron (MLP) to transform identity-aware text features instead of complex networks. To enhance the modeling capacity, we stack M denoisers for LWD. Note that, LWD is lightweight because our diffusion process is performed in a compact latent feature space for identity-aware distribution learning [14], [19]. Unlike image-based diffusion models that require complex denoising networks and long denoising trajectories, LWD models the transformation between semantic feature representations. Therefore, an MLP-based lightweight denoiser with a small number of timesteps is sufficient to generate discriminative features while maintaining computational efficiency. To obtain the condition of LWD, we first aggregate the intermediate features from the image encoder and then pass them through a linear layer to obtain the condition representation ${ \check { \mathcal { F } } } _ { : }$

$$
\hat { \mathcal { F } } = \sum _ { l } f _ { l } ,\tag{9}
$$

$$
\check { \mathcal { F } } = W _ { 1 } \hat { \mathcal { F } } ,\tag{10}
$$

where $f _ { l }$ represents the output from the l-th Transformer layer and $W _ { 1 }$ is the learnable parameter of a linear layer.

During the denoising process, the features change progressively across timesteps, and different stages require different types of visual guidance. At early timesteps, the corrupted features contain stronger noise, so the denoiser mainly needs coarse and global contextual information to recover the overall identity-related structure. At later timesteps, the features are gradually refined, and the denoiser requires more fine-grained and identity-discriminative cues to further improve feature discrimination. Therefore, using the same condition for all timesteps may limit the ability of LWD to adapt to the evolving feature state. To address this issue, we introduce a set of learnable Temporal Condition Queries (TCQ) to obtain timestep-specific conditions for LWD. Specifically, at each timestep t, we initialize a set of learnable queries $\mathcal { Q } = \{ q _ { t } ^ { 1 } , q _ { t } ^ { 2 } , \ldots , q _ { t } ^ { N } \}$ , where N denotes the number of queries per timestep. Inspired by [39], we introduce the noise $\bar { f } _ { t }$ as a prior into the corresponding condition queries to stabilize training and accelerate convergence:

$$
{ \check { q } } _ { t } ^ { n } = q _ { t } ^ { n } + W _ { 2 } { \bar { f } } _ { t } ,\tag{11}
$$

where $t \in \{ 1 , 2 \cdot \cdot \cdot , T \} , n \in \{ 1 , 2 \cdot \cdot \cdot , N \}$ and $W _ { 2 }$ is the learnable parameter. Finally, these temporal queries $\check { q } _ { t } ^ { n }$ interact with the condition representation $\check { \mathcal { F } }$ through a cross-attention to obtain the condition features at each timestep:

$$
\hat { q } _ { t } ^ { n } = \mathrm { S o f t m a x } \left( \frac { \check { q } _ { t } ^ { n } W _ { q } ( \check { \mathcal { F } } W _ { k } ) ^ { T } } { \sqrt { D } } \right) ( \check { \mathcal { F } } W _ { v } ) ,\tag{12}
$$

where $W _ { q } , W _ { k }$ and $W _ { v }$ are learnable parameters. $D$ is the feature dimension for scaling. As shown in Fig. 2(c), we obtain the timestep-specific condition $\hat { c } _ { t }$ by averaging the condition features as follows:

$$
\hat { c } _ { t } = \mathrm { A v g } ( \hat { q } _ { t } ^ { 1 } , \hat { q } _ { t } ^ { 2 } , \cdot \cdot \cdot , \hat { q } _ { t } ^ { N } ) .\tag{13}
$$

Then, the noised feature ${ \bar { f } } _ { t } ,$ the time embedding $v _ { t } ,$ , and the timestep-specific condition $\hat { c } _ { t }$ are jointly processed to denoise the corrupted identity-aware text features:

$$
\bar { f } _ { t - 1 } = W _ { 3 } ( W _ { 4 } \bar { f } _ { t } + W _ { 5 } v _ { t } + \hat { c } _ { t } ) + W _ { 6 } \bar { f } _ { t } ,\tag{14}
$$

where $W _ { 3 } , \ W _ { 4 } , \ W _ { 5 }$ and $W _ { 6 }$ are learnable parameters. With LWD, we can take visual features as conditions and denoise the corrupted text features step-by-step for identity-aware distribution learning.

## D. Mutual Enhancement Constraint

Technically, most of diffusion model-based feature learning methods [8], [19] directly adopt generated features as the final representations. However, they overlook the complementary between features extracted by general visual encoders and those generated by diffusion models. In fact, features extracted by general visual encoders capture discriminative information but tend to overfit to salient appearance patterns [40]. In contrast, generated features of diffusion models are encouraged to follow the identity-aware feature distribution [38], making them more robust and semantic. To better integrate the advantages of both features, we introduce the Mutual Enhancement Constraint (MEC), which encourages visual features and generated features to learn from each other. Specifically, generated features provide identity-aware semantic cues to improve the generalization of visual features. Visual features provide image-specific evidence to constrain the generated features and suppress unreliable semantic deviations.

In each iteration, the diffusion model first generates guided features that follow the identity-aware distribution. Then, these guided features and visual features mutually refine each other. Specifically, for each input image $I _ { b } ,$ we obtain a feature set as follows:

$$
\mathcal { V } _ { b } = \{ \hat { f } _ { b } , g _ { b } ^ { 1 } , \cdot \cdot \cdot , g _ { b } ^ { S } \} ,\tag{15}
$$

where $b \in \{ 1 , 2 , \cdots , B \} , \hat { f }$ is the visual feature and $g ^ { s }$ is the guided feature. $S$ is the total number of guided features. Then, we can construct the feature set within each batch:

$$
\mathcal { V } = \{ \mathcal { V } _ { 1 } , \mathcal { V } _ { 2 } \cdot \cdot \cdot , \mathcal { V } _ { B } \} .\tag{16}
$$

To achieve mutual learning, we introduce the MEC:

$$
L _ { \mathrm { m e c } } = \sum _ { \stackrel { a , p , n } { y _ { a } = y _ { p } \neq y _ { n } } } [ D _ { a , p } - D _ { a , n } + m _ { 1 } ] _ { + } ,\tag{17}
$$

where $D _ { a , p }$ and $D _ { a , n }$ are the distances between the anchor–positive and anchor–negative pairs within V, respectively. $m _ { 1 }$ is the margin parameter. As observed, MEC combines the visual and guided features into a joint feature set and constructs cross-feature triplets. The triplet loss pulls each guided feature closer to the same-identity visual feature and pushes it away from different identity features. Thus, the visual feature acts as a stable identity reference that reduces stochastic shifts of the guided feature toward incorrect identities. Meanwhile, the guided feature introduces distributional variations that encourage the visual feature to retain identity-consistent cues. Through this bidirectional interaction, the advantages of visual and guided features are jointly amplified, enabling the framework to learn more discriminative representations.

## E. Training and Inference

Model Training. In the first stage, we follow previous works [2], [41] and apply the bidirectional contrastive losses for prompt tuning:

$$
L _ { \mathrm { s t a g e 1 } } = L _ { t 2 i } + L _ { i 2 t } .\tag{18}
$$

In the second stage, we employ multiple losses to supervise the framework. To optimize the visual feature ${ \hat { f } } ,$ we adopt the following combined loss:

$$
L _ { \mathrm { v i s } } = \gamma _ { 1 } L _ { c e } + \gamma _ { 2 } L _ { t r i } + L _ { t 2 i } ,\tag{19}
$$

where $L _ { c e }$ is the identity loss. $L _ { t r i }$ is the triplet loss. $\gamma _ { 1 }$ and γ<sub>2</sub> are hyper-parameters to balance the loss terms.

As for the diffusion procedure, we supervise the noise prediction at each timestep by the following loss:

$$
L _ { \mathrm { d i f f u s i o n } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } | | \bar { f } _ { t } - f _ { d } | | _ { 2 } .\tag{20}
$$

As for guided features, we further ensure that they have the same identity with the following loss:

$$
L _ { \mathrm { d i s } } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } [ L _ { t r i } ( g ^ { s } ) + L _ { c e } ( g ^ { s } ) ] .\tag{21}
$$

To align with the distribution assumption, the Kullback-Leibler (KL) divergence loss is reformulated as:

$$
L _ { k l } = D _ { \mathrm { k l } } \big ( p _ { \phi } ( \bar { f } _ { T } | \hat { f } ) | | \mathcal { N } ( 0 , I ) \big ) .\tag{22}
$$

Thus, the overall loss for the second stage is formulated as:

$$
L _ { \mathrm { s t a g e } 2 } = L _ { \mathrm { v i s } } + L _ { \mathrm { d i f f u s i o n } } + L _ { \mathrm { m e c } } + L _ { \mathrm { d i s } } + L _ { \mathrm { k l } } .\tag{23}
$$

TABLE I: Comparison of state-of-the-art methods on three person ReID datasets and one vehicle ReID dataset. The best and second results are marked in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">Market1501</td><td colspan="2">MSMT17</td><td colspan="2">DukeMTMC</td><td rowspan="2">Method</td><td colspan="2">VeRi-776</td></tr><tr><td>mAP</td><td>Rank-1</td><td>mAP</td><td>Rank-1</td><td>mAP</td><td>Rank-1</td><td>mAP</td><td>Rank-1</td></tr><tr><td>Nformer [42]</td><td>91.1</td><td>94.7</td><td>59.8</td><td>77.3</td><td>83.5</td><td>89.4</td><td>CFVMNet [43]</td><td>77.1</td><td>95.3</td></tr><tr><td>CMT [27]</td><td>87.7</td><td>95.4</td><td>62.6</td><td>83.1</td><td>80.0</td><td>90.1</td><td>DCAL [44]</td><td>80.2</td><td>96.9</td></tr><tr><td>TransReID [1]</td><td>88.9</td><td>95.2</td><td>67.4</td><td>85.3</td><td>82.0</td><td>90.7</td><td>GLTrans [4]</td><td>82.9</td><td>97.5</td></tr><tr><td>GLTrans [4]</td><td>90.0</td><td>95.6</td><td>69.0</td><td>85.8</td><td>82.4</td><td>90.7</td><td>HPGN [45]</td><td>80.2</td><td>96.7</td></tr><tr><td>HAT [24]</td><td>89.8</td><td>95.8</td><td>61.2</td><td>82.3</td><td>81.4</td><td>90.4</td><td>PGAN [46]</td><td>79.3</td><td>96.5</td></tr><tr><td>ADSO [47]</td><td>87.7</td><td>94.8</td><td></td><td></td><td>74.9</td><td>87.4</td><td>PVEN [48]</td><td>79.5</td><td>95.6</td></tr><tr><td>PFD [49]</td><td>89.6</td><td>95.5</td><td>65.1</td><td>82.7</td><td>82.2</td><td>90.6</td><td>SAVER [50]</td><td>79.6</td><td>96.4</td></tr><tr><td>DCAL [44]</td><td>87.5</td><td>94.7</td><td>64.0</td><td>83.1</td><td>80.1</td><td>89.0</td><td>SOFCT [51]</td><td>80.7</td><td>96.6</td></tr><tr><td>SAP [52]</td><td>90.5</td><td>96.0</td><td>67.8</td><td>85.7</td><td></td><td></td><td>GLAMOR [53]</td><td>80.3</td><td>96.5</td></tr><tr><td>DC-Former [25]</td><td>90.4</td><td>96.0</td><td>68.8</td><td>86.2</td><td></td><td></td><td>MPC [54]</td><td>80.9</td><td>96.2</td></tr><tr><td>RGANet [55]</td><td>89.8</td><td>95.5</td><td>72.3</td><td>88.1</td><td></td><td></td><td>MsKAT [56]</td><td>82.0</td><td>97.1</td></tr><tr><td>PHA [57]</td><td>90.2</td><td>96.1</td><td>68.9</td><td>86.1</td><td></td><td></td><td>TransReID [1]</td><td>82.0</td><td>97.1</td></tr><tr><td>PCL-CLIP [58]</td><td>91.4</td><td>95.9</td><td>76.1</td><td>89.8</td><td></td><td></td><td>PCL-CLIP [58]</td><td>82.5</td><td>97.1</td></tr><tr><td>CLIP-ReID [2]</td><td>90.5</td><td>95.4</td><td>75.8</td><td>89.7</td><td>83.1</td><td>90.8</td><td>Vehicle-Diff [59]</td><td>83.8</td><td>97.7</td></tr><tr><td>TF-CLIP [3]</td><td>90.4</td><td>95.7</td><td>73.9</td><td>88.5</td><td></td><td></td><td>MDPDTrans [60]</td><td>83.7</td><td>97.7</td></tr><tr><td>DenoiseRep [8]</td><td>91.1</td><td>95.8</td><td>76.3</td><td>90.6</td><td>83.7</td><td>91.6</td><td>ADPRP-Net [61]</td><td>82.8</td><td>95.6</td></tr><tr><td>CLIMB-ReID [29]</td><td>92.6</td><td>96.8</td><td>77.8</td><td>90.5</td><td></td><td></td><td>CLIP-ReID [2]</td><td>84.5</td><td>97.3</td></tr><tr><td>DiffReID (Ours)</td><td>91.5</td><td>96.1</td><td>79.0</td><td>91.2</td><td>85.6</td><td>92.4</td><td>DiffReID (Ours)</td><td>84.7</td><td>97.4</td></tr></table>

Model Inference. We extract the visual feature $\hat { f }$ and use LWD to generate guided features from visual-guided noises. The final feature $f _ { r }$ is formed by averaging the guided features:

$$
f _ { r } = \mathrm { A v g } ( g ^ { 1 } , g ^ { 2 } , \dots , g ^ { S } ) .\tag{24}
$$

Since each guided feature is sampled from the learned identityaware distribution, a single guided feature may only reflect one possible representation of this distribution. Thus, we average multiple guided features to obtain a more comprehensive and stable representation.

## IV. EXPERIMENTS

## A. Datasets and Evaluation Metrics

To comprehensively evaluate the effectiveness of our proposed framework, we conduct experiments on five large-scale object ReID datasets, i.e., Market1501 [62], DukeMTMC [63], MSMT17 [64], CUHK03 [65] and VeRi-776 [66]. The details of these datasets can be found in corresponding references. To assess the generalization ability, we employ two domain generalization settings [67]. The first setting involves training on one dataset and testing on another unseen dataset. The second setting utilizes the training sets of multiple datasets for training and evaluates on the test set of an unseen dataset. Following previous works [1], [20], we use mean Average Precision (mAP) and Cumulative Matching Characteristics (CMC) at Rank-1 as our evaluation metrics.

## B. Implementation Details

We implement our framework using the PyTorch toolbox. All experiments are conducted on a single NVIDIA RTX 4090 GPU with 24GB memory. We adopt the pre-trained CLIP [36] as the feature extraction backbone for both images and texts. Person images are resized to 256×128 and vehicle images are resized to $2 5 6 \times 2 5 6$ . In the first stage, we employ the Adam optimizer [68] with an initial learning rate of $3 . 5 \times 1 0 ^ { - 4 }$ and cosine decay. Training is performed with a batch size of 64, without augmentation, only optimizing the learnable prompt tokens. In the second stage, a mini-batch of 64 images is sampled, which contains 16 identities and each with 4 images. Data augmentation consists of random cropping, horizontal flipping, and random erasing [69]. We train the model for 60 epochs using Adam, setting the initial learning rate to $5 \times 1 0 ^ { - 6 }$ for the image encoder and $5 \times 1 0 ^ { - 4 }$ for the discriminative diffusion model. The model is warmed up for 10 epochs, with the learning rate reduced by a factor of 0.1 at the 30th and 50th epochs. The hyper-parameters $\gamma _ { 1 } , \gamma _ { 2 } , m _ { 1 } , T$ and M are set to 0.25, 1.0, 0.3, 10 and 3, respectively.

## C. Comparison with State-of-the-Arts Methods

In Tab. I and Tab. II, our method is compared with other state-of-the-art methods on five benchmarks under both single domain and domain generalization settings. Note that no postprocessing techniques are employed in these experiments.

Single Domain Comparison. As shown in Tab. I, our method achieves state-of-the-art results on Market1501, MSMT17, DukeMTMC and VeRi-776. More remarkably, our method achieves 79.0% in mAP on MSMT17, outperforming most of the compared methods, $e . g . ,$ , DenoiseRep [8] and CLIP-ReID [2] by 2.7% and 3.2%, respectively. In fact, DenoiseRep [8] also introduces diffusion models for person ReID. However, its complex denoising process and long diffusion steps may hinder model convergence, ultimately limiting performance improvements. CLIP-ReID [2] introduces the large-scale vision-language models into object ReID. However, due to the lack of class-wise descriptions, CLIP-ReID tends to focus on semantic patterns rather than identity-aware feature distributions. Unlike these methods, our method introduces a diffusion model to learn identity-aware feature distributions and further generate diverse guided features, leading to more generalized and robust representations.

Domain Generalization Comparison. As shown in Tab. II, we evaluate the generalization of different methods in different domain generalization settings. The results show that our method achieves competitive performance compared with other domain generalization methods. Especially, our method achieves the best performance in the setting of Market1501+DukeMTMC+MSMT17→CUHK03, reaching 50.7% mAP, which surpasses OGNorm by 10.4%. These results indicate that our method effectively learns identity-aware feature distributions and prevents the model from overfitting to semantic regions.

TABLE II: Comparison of state-of-the-art methods on domain generalization settings. The best and second results are marked in bold and underlined, respectively. The superscript \* indicates that the model resizes the input images to 384×128.
<table><tr><td rowspan=1 colspan=2>Method</td><td rowspan=1 colspan=5>Training</td><td rowspan=1 colspan=1>Market1501mAP   R1</td><td rowspan=1 colspan=1>MSMT17mAP   R1</td><td rowspan=1 colspan=1>CUHK03-NPmAP   R1</td><td rowspan=1 colspan=1>DukeMTMCmAP   R1</td></tr><tr><td rowspan=9 colspan=2>QAConv [70]TransMatcher [71]QAConv+Gs [72]]MDA [67]PAT [73]OGNorm* [74]CLIP-ReID [2]DiffReID (Ours)</td><td rowspan=3 colspan=5></td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>7.0   22.6</td><td rowspan=1 colspan=1>8.6    9.9</td><td rowspan=1 colspan=1>33.6   54.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>18.4   47.3</td><td rowspan=1 colspan=1>21.4   22.2</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[72]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>17.2  45.9</td><td rowspan=1 colspan=1>18.1   19.1</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=6 colspan=5>Market1501</td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>11.8   33.518.2   42.8</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>34.4   56.7</td></tr><tr><td rowspan=2 colspan=2>501</td></tr><tr><td rowspan=1 colspan=1>26.0   25.4</td><td rowspan=1 colspan=1>48.9   67.9</td></tr><tr><td rowspan=3 colspan=1>一一一</td><td rowspan=1 colspan=1>19.9   49.7</td><td rowspan=1 colspan=1>24.9   26.6</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>23.0  48.9</td><td rowspan=1 colspan=1>38.5   39.9</td><td rowspan=1 colspan=1>51.7   69.6</td></tr><tr><td rowspan=1 colspan=1>24.6  51.5</td><td rowspan=1 colspan=1>41.2   43.5</td><td rowspan=1 colspan=1>51.9  71.8</td></tr><tr><td rowspan=8 colspan=2>QAConv [70]TransMatcher [71]QAConv+Gs [72]MDA [67]PAT [73]OGNorm* [74]CLIP-ReID [2]DiffReID (Ours)</td><td rowspan=3 colspan=5></td><td rowspan=1 colspan=1>43.1   72.6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>22.6   25.3</td><td rowspan=1 colspan=1>53.4   72.2</td></tr><tr><td rowspan=1 colspan=1>52.0   80.1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>22.5   23.7</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>49.5   79.1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>20.6   20.9</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=5 colspan=5>MSMT17</td><td rowspan=1 colspan=1>53.0  79.7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>52.4   71.7</td></tr><tr><td rowspan=1 colspan=1>47.3   72.2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>25.1   24.2</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>54.5  83.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>28.5   31.0</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>51.5   76.2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>38.9   40.2</td><td rowspan=1 colspan=1>58.0   74.9</td></tr><tr><td rowspan=1 colspan=1>54.2  82.5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>40.7   42.7</td><td rowspan=1 colspan=1>60.3  76.0</td></tr><tr><td rowspan=7 colspan=2>QAConv [70]RaMoE [75]M³L [76]CINorm [77]PAT [73]OGNorm* [74]DiffReID (Ours)</td><td rowspan=4 colspan=5>Multi-source</td><td rowspan=1 colspan=1>39.5  68.6</td><td rowspan=1 colspan=1>10.0   29.9</td><td rowspan=1 colspan=1>19.2   22.9</td><td rowspan=1 colspan=1>43.4   64.9</td></tr><tr><td rowspan=1 colspan=1>56.5   82.0</td><td rowspan=1 colspan=1>13.5   34.1</td><td rowspan=1 colspan=1>35.5   36.6</td><td rowspan=1 colspan=1>56.9  73.6</td></tr><tr><td rowspan=1 colspan=1>50.2   75.9</td><td rowspan=1 colspan=1>14.7   36.9</td><td rowspan=1 colspan=1>32.1   33.1</td><td rowspan=1 colspan=1>51.1   69.2</td></tr><tr><td rowspan=1 colspan=1>57.8  82.3</td><td rowspan=1 colspan=1>21.1   49.7</td><td rowspan=1 colspan=1>31.1   30.3</td><td rowspan=1 colspan=1>52.4  71.3</td></tr><tr><td rowspan=3 colspan=5></td><td rowspan=1 colspan=1>51.7   75.2</td><td rowspan=1 colspan=1>21.6  45.6</td><td rowspan=1 colspan=1>31.5   31.1</td><td rowspan=1 colspan=1>56.5   71.8</td></tr><tr><td rowspan=1 colspan=1>65.2  87.1</td><td rowspan=1 colspan=1>25.9  57.7</td><td rowspan=1 colspan=1>40.3   44.0</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>61.6   82.3</td><td rowspan=1 colspan=1>26.8  52.6</td><td rowspan=1 colspan=1>50.7   51.4</td><td rowspan=1 colspan=1>61.5  76.8</td></tr></table>

## D. Ablation Studies

In this section, we perform ablation studies on MSMT17 and DukeMTMC datasets to investigate the effect of key components and hyper-parameters.

TABLE III: Ablation results with different components.
<table><tr><td></td><td>Baseline</td><td>LWD</td><td>VNG</td><td>MEC</td><td colspan="2">MSMT17</td><td colspan="2">DukeMTMC</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td></tr><tr><td>(a)</td><td>√</td><td>X</td><td>×</td><td>X</td><td>75.4</td><td>89.3</td><td>82.8</td><td>90.7</td></tr><tr><td>(b)</td><td>√</td><td>√</td><td>X</td><td>X</td><td>76.6</td><td>90.4</td><td>83.9</td><td>91.1</td></tr><tr><td>(c)</td><td>√</td><td>√</td><td>√</td><td>X</td><td>77.7</td><td>90.8</td><td>84.5</td><td>91.7</td></tr><tr><td>(d)</td><td>√</td><td>√</td><td>×</td><td>√</td><td>77.2</td><td>90.9</td><td>84.6</td><td>92.0</td></tr><tr><td>(e)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>79.0</td><td>91.2</td><td>85.6</td><td>92.4</td></tr></table>

Effect of Key Components. Tab. III presents the ablation results with key components. The results show that using the LWD achieves performance gains on the MSMT17 and DukeMTMC, consistently. Furthermore, with VNG, the model achieves an additional improvement of 1.1% mAP on MSMT17 and 0.6% mAP on DukeMTMC. It confirms the effectiveness of adaptive noise initialization in enhancing feature discrimination. With MEC, the model further improves mAP by 0.6% on MSMT17 and 0.7% on DukeMTMC. These results demonstrate that mutually learning visual features and guided features allows them to complement each other. Finally, incorporating all components achieves the best results: 79.0% mAP on MSMT17 and 85.6% mAP on DukeMTMC. The consistent improvements across all configurations validate the effectiveness of our proposed components.

TABLE IV: Comparison with different diffusion steps.
<table><tr><td rowspan="2">No.</td><td colspan="2">MSMT17</td><td colspan="2">DukeMTMC</td></tr><tr><td>mAP</td><td>Rank-1</td><td>mAP</td><td>Rank-1</td></tr><tr><td>Baseline</td><td>75.4</td><td>89.3</td><td>82.8</td><td>90.7</td></tr><tr><td>2</td><td>77.0</td><td>90.2</td><td>84.4</td><td>92.1</td></tr><tr><td>4</td><td>77.5</td><td>90.5</td><td>84.6</td><td>92.3</td></tr><tr><td>6</td><td>78.0</td><td>90.9</td><td>85.0</td><td>92.0</td></tr><tr><td>8</td><td>78.6</td><td>91.0</td><td>85.3</td><td>92.2</td></tr><tr><td>10</td><td>79.0</td><td>91.2</td><td>85.6</td><td>92.4</td></tr><tr><td>12</td><td>78.9</td><td>91.3</td><td>85.3</td><td>92.1</td></tr></table>

Effect of the Total Number of Diffusion Steps. To analyze the effect of the total number of diffusion steps T, we conduct additional experiments in Tab. IV. As observed, increasing T to 2 significantly boosts performance, reaching 77.0% mAP on MSMT17 and 84.4% mAP on DukeMTMC. From T = 4 to T = 8, our model maintains competitive performance. There is a trend that the discrimination of guided features gradually improves as T increases. With $T = 1 0$ , our model yields the best performance. This result indicates that even a small total number of diffusion steps can enhance feature representation.

Effect of Mutual Enhancement in MEC. In Tab. V, we show the effect of MEC with visual and guided features on MSMT17. The MEC consistently improves the performance with both visual and guided features. We further evaluate the intra-class and inter-class cosine similarities. For each input image, three guided features are generated using independently sampled noises, and their average pairwise cosine similarity is used to measure the generation consistency. With MEC, the intra-class similarity increases, while the inter-class similarity decreases. The generation consistency also increases. These results indicate that MEC produces more discriminative and stable guided features. These results demonstrate that MEC benefits both features and support their mutual enhancement.

TABLE V: Effect of MEC with visual and guided features.
<table><tr><td rowspan="2">MEC</td><td rowspan="2">Feature</td><td colspan="2">Performance</td><td colspan="2">Cosine Similarity</td><td rowspan="2">Generation Consistency ↑</td></tr><tr><td>mAP</td><td>Rank-1</td><td>Intra-class ↑</td><td>Inter-class ↓</td></tr><tr><td rowspan="2">w/o</td><td>Visual</td><td>77.0</td><td>89.5</td><td>0.72</td><td>0.35</td><td></td></tr><tr><td>Guided</td><td>77.7</td><td>90.8</td><td>0.76</td><td>0.29</td><td>0.75</td></tr><tr><td rowspan="2">w/</td><td>Visual</td><td>77.8</td><td>90.3</td><td>0.78</td><td>0.29</td><td></td></tr><tr><td>Guided</td><td>79.0</td><td>91.2</td><td>0.83</td><td>0.24</td><td>0.89</td></tr></table>

TABLE VI: Comparison with different guided features.
<table><tr><td rowspan="2"> $\mathrm { N o . }$ </td><td colspan="2">MSMT17</td><td colspan="2">DukeMTMC</td></tr><tr><td>mAP</td><td>Rank-1</td><td>mAP</td><td>Rank-1</td></tr><tr><td>Baseline</td><td>75.4</td><td>89.3</td><td>82.8</td><td>90.7</td></tr><tr><td>1</td><td>76.7</td><td>90.5</td><td>84.1</td><td>91.7</td></tr><tr><td>2</td><td>78.1</td><td>90.9</td><td>84.8</td><td>92.0</td></tr><tr><td>3</td><td>79.0</td><td>91.2</td><td>85.6</td><td>92.4</td></tr><tr><td>4</td><td>79.0</td><td>91.3</td><td>85.4</td><td>92.2</td></tr></table>

TABLE VII: Effect of different number of temporal queries.
<table><tr><td rowspan="2"> $\mathrm { N o . }$ </td><td colspan="2">MSMT17</td><td colspan="2">DukeMTMC</td></tr><tr><td>mAP</td><td>Rank-1</td><td>mAP</td><td>Rank-1</td></tr><tr><td>2</td><td>76.7</td><td>89.9</td><td>83.9</td><td>91.4</td></tr><tr><td>4</td><td>77.1</td><td>90.1</td><td>84.1</td><td>91.6</td></tr><tr><td>8</td><td>77.5</td><td>90.6</td><td>84.3</td><td>92.1</td></tr><tr><td>16</td><td>79.0</td><td>91.2</td><td>85.6</td><td>92.4</td></tr><tr><td>32</td><td>78.7</td><td>91.0</td><td>85.4</td><td>92.4</td></tr></table>

Effect of Guided Features. In Tab. VI, we evaluate the effect of guided features. As observed, using one guided feature can improve the performance over the baseline. This suggests that introducing guided features enhances the robustness of representation learning. Increasing the number of guided features to two leads to further improvements, indicating that diverse guided features contribute to a more comprehensive feature. When using three or four guided features, the model consistently achieves better performance. To balance the performance and efficiency, we adopt three guided features as our default setting.

Effect of Different Numbers of TCQ. In Tab. VII, we evaluate the effect of using different numbers of TCQ. As the number of TCQ increases from 2 to 16, the performance consistently improves on both MSMT17 and DukeMTMC. This trend indicates that richer temporal conditions help the denoiser better model the evolving feature distribution across diffusion steps. In particular, using 16 queries achieves the best results. These results confirm the effectiveness of TCQ in enhancing the discrimination of guided features.

Effect of Intermediate Features for the Generation. We investigate the impact of incorporating intermediate-layer features from the backbone as conditions for the denoiser. In Tab. VIII, $f _ { 1 2 }$ refers to using only the output from the 12-th layer as the condition, while $f _ { 1 2 } + f _ { 1 1 }$ indicates the addition of the outputs from the 11-th and 12-th layers as the condition. These results demonstrate that features from earlier layers consistently improve the performance. This suggests that intermediate-layer features provide additional information that enhances the generation. Meanwhile, the results from the last line of Tab. VIII also indicate that incorporating too many intermediate-layer features leads to a noticeable performance drop. A plausible explanation is that early-layer features mainly capture low-level patterns and lack semantic discrimination, which may reduce the representational ability.

TABLE VIII: Effect of intermediate features for the generation.
<table><tr><td>Setting</td><td>MSMT17 mAP Rank-1</td><td>mAP</td><td>DukeMTMC Rank-1</td></tr><tr><td>f12</td><td>77.8</td><td>90.8</td><td>84.4 91.0</td></tr><tr><td> $f _ { 1 2 } + f _ { 1 1 }$ </td><td>78.0</td><td>91.2</td><td>84.5 91.2</td></tr><tr><td> $f _ { 1 2 } + f _ { 1 1 } + f _ { 1 0 }$ </td><td>78.6 91.4</td><td>84.6</td><td>91.5</td></tr><tr><td> $f _ { 1 2 } + f _ { 1 1 } + f _ { 1 0 } + f _ { 9 }$ </td><td>78.8</td><td>91.3 84.9</td><td>91.7</td></tr><tr><td> $f _ { 1 2 } + f _ { 1 1 } + f _ { 1 0 } + f _ { 9 } + f _ { 8 }$ </td><td>78.8</td><td>91.0 85.1</td><td>92.0</td></tr><tr><td> $f _ { 1 2 } + f _ { 1 1 } + f _ { 1 0 } + f _ { 9 } + f _ { 8 } + f _ { 7 }$ </td><td>79.0</td><td>91.2</td><td>85.6 92.4</td></tr><tr><td> $f _ { 1 2 } + f _ { 1 1 } + f _ { 1 0 } + f _ { 9 } + f _ { 8 } + f _ { 7 } + f _ { 6 }$ </td><td>77.9</td><td>90.8</td><td>84.2 91.8</td></tr></table>

TABLE IX: Effect of different random seeds.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Seed</td><td colspan="2">MSMT17</td><td colspan="2">DukeMTMC</td></tr><tr><td>mAP</td><td>Rank-1</td><td>mAP</td><td>Rank-1</td></tr><tr><td>Baseline</td><td>1234</td><td>75.4</td><td>89.3</td><td>82.8</td><td>90.7</td></tr><tr><td>DiffReID</td><td>1</td><td>78.8</td><td>91.3</td><td>85.8</td><td>92.3</td></tr><tr><td>DiffReID</td><td>12</td><td>78.4</td><td>91.4</td><td>85.5</td><td>92.0</td></tr><tr><td>DiffReID</td><td>123</td><td>78.7</td><td>90.7</td><td>85.4</td><td>92.2</td></tr><tr><td>DiffReID</td><td>1234</td><td>79.0</td><td>91.2</td><td>85.6</td><td>92.4</td></tr></table>

Effect of Different Random Seeds. To evaluate the training robustness, we conduct additional experiments with four different random seeds. As shown in Tab. IX, our method consistently outperforms the baseline under all random seeds. Specifically, the mAP ranges from 78.4% to 79.0%, and the Rank-1 accuracy ranges from 90.7% to 91.4%. The performance fluctuation is very small, indicating that our method is not sensitive to the random seed initialization. These results demonstrate that the improvement of our method is stable and does not come from a specific random seed initialization.

Effect of Different Diffusion Features. As shown in Tab. X, performing diffusion on visual features improves the baseline from 75.4% to 77.8% mAP and from 89.3% to 90.5% Rank-1 on MSMT17. This indicates that diffusionbased feature distribution learning is beneficial for discriminative representation learning. Furthermore, performing diffusion on identity-aware text features achieves better performance, reaching 79.0% mAP and 91.2% Rank-1. These results demonstrate that identity-aware text features provide a more suitable semantic space for diffusion modeling. As a result, the model can learn more discriminative and generalizable identity-aware feature distributions rather than appearance-biased visual feature distributions.

TABLE X: Effect of different diffusion features.
<table><tr><td rowspan="2">Setting</td><td colspan="2">MSMT17</td><td colspan="2">DukeMTMC</td></tr><tr><td>mAP</td><td>Rank-1</td><td>mAP</td><td>Rank-1</td></tr><tr><td>Baseline</td><td>75.4</td><td>89.3</td><td>82.8</td><td>90.7</td></tr><tr><td>On visual features</td><td>77.8</td><td>90.5</td><td>84.5</td><td>91.8</td></tr><tr><td>On identity-aware text features</td><td>79.0</td><td>91.2</td><td>85.6</td><td>92.4</td></tr></table>

TABLE XI: Effect of the KL regularization in VNG.
<table><tr><td rowspan="2">Model</td><td colspan="2">MSMT17</td><td colspan="2">DukeMTMC</td></tr><tr><td>mAP</td><td>Rank-1</td><td>mAP</td><td>Rank-1</td></tr><tr><td>Baseline</td><td>75.4</td><td>89.3</td><td>82.8</td><td>90.7</td></tr><tr><td>DiffReID w/o KL</td><td>77.6</td><td>90.8</td><td>83.7</td><td>91.1</td></tr><tr><td>DiffReID</td><td>79.0</td><td>91.2</td><td>85.6</td><td>92.4</td></tr></table>

Effect of the KL Regularization. As shown in Tab. XI, removing the KL regularization still achieves 77.6% mAP and 90.8% Rank-1, outperforming the baseline by 2.2% mAP and 1.5% Rank-1. This indicates that the visual-guided noise initialization in VNG is beneficial for discriminative feature generation. With the KL regularization, the performance is further improved to 79.0% mAP and 91.2% Rank-1. These results demonstrate that the KL regularization helps constrain the learned noise distribution and stabilizes the diffusion process, leading to more effective guided-feature generation.

## E. Computational Cost Analysis

Tab. XII compares the computational cost of different methods. Note that, our LWD adopts an MLP-based structure instead of the complex U-Net. Thus, it is more efficient than previous diffusion-based methods, such as SD-ReID [78]. Compared with other methods, our method has a comparable computational overhead but achieves superior ReID performance. These results demonstrate that our method provides a trade-off between computational cost and performance.

![](images/e492d14b427660813dbc09405516a3d33872c5c053b53f957f5ff353834112cc.jpg)

![](images/0e89d8e4a229399aa27a0e556f6caba78eb5b7d0642220fc0d39211a7c06af11.jpg)  
Fig. 3: Retrieval results on MSMT17. Black, green and red boxes indicate query images, correct matches and incorrect matches, respectively.

TABLE XII: Cost comparison with different methods.
<table><tr><td>Method</td><td>Trainable Params (M)</td><td>Memory (GB)</td><td>GFLOPs</td><td>Time (s/batch)</td><td>mAP (MSMT17)</td></tr><tr><td>SD-ReID [78]</td><td>103.36</td><td>11.05</td><td>677.67</td><td>1.66</td><td>一</td></tr><tr><td>FusionReID [28]</td><td>153.80</td><td>1.62</td><td>28.10</td><td>0.14</td><td>69.5</td></tr><tr><td>TF-CLIP [3]</td><td>104.26</td><td>1.53</td><td>24.24</td><td>0.26</td><td>73.9</td></tr><tr><td>DiffReID</td><td>104.81</td><td>1.32</td><td>43.40</td><td>0.22</td><td>79.0</td></tr></table>

## F. Qualitative Analysis

To better understand our proposed framework, we present comprehensive qualitative results in this subsection.

Top-5 Retrieval Results. Fig. 3 shows retrieval results with CLIP-ReID [2] and our method. In general, our method captures more fine-grained information while reducing the influence of local salient features. In Fig. 3(a) and (b), CLIP-ReID is distracted by visually prominent attributes, such as similar black clothing and white backpack, leading to incorrect matches. In contrast, our method focuses on more comprehensive information. It avoids over-reliance on a few specific attributes and successfully identifies the correct match among visually similar samples. A similar pattern is observed in Fig. 3(c) and (d). In Fig. 3(d), even though the person in the query image carries a white backpack, our method does not get misled by this prominent detail. In Fig. 3(c), despite significant distortions in the query image, our method still accurately retrieves the correct match.

Visualization of Feature Distribution. We further analyze the feature distributions on the MSMT17 datasets. Specifically, as shown in Fig. 5, we randomly sample twenty identities from the MSMT17 test set and visualize their feature distributions using t-SNE [79]. Each point represents a sample and different colors represent different identities. Compared with the baseline, our DiffReID can better reduce intra-class distances and increase inter-class distances. The main reason is that our DiffReID directly captures the identity-aware distribution rather than the salient semantic regions.

Furthermore, we visualize the cosine similarity distributions on the MSMT17 test set to analyze the discrimination of different feature representations. As shown in Fig. 6(a), the baseline exhibits a small distribution gap between positive and negative pairs. It indicates that the baseline struggles to distinguish samples with similar visual patterns. When our DiffReID is introduced with random Gaussian noises, as shown in Fig. 6(b), the distribution gap becomes much larger. It demonstrates that diffusion-based guided feature generation improves feature discrimination. Moreover, when visual-guided noises are adopted in Fig. 6(c), the negativepair distribution further shifts to a lower similarity range while the positive-pair distribution remains in a high-similarity range. As a result, the distribution gap increases from 0.5 to 0.598. This indicates that VNG provides a more suitable noise initialization for generating discriminative features. It improves the inter-identity separation while preserving the intra-identity consistency. Overall, these results show that our DiffReID captures identity-aware distributions and obtains more discriminative representations.

Visualization of Guided Features Across the Diffusion Process. To investigate how the model gradually captures the identity-aware feature distribution, we visualize guided features at successive denoising timesteps using t-SNE, as shown in Fig. 4. Specifically, we randomly sample ten identities from the MSMT17 test set. Each point represents a sample and different colors denote different identities. At the early denoising timesteps, the guided features exhibit large variance and strong overlap across identities, indicating that they primarily encode coarse and noisy structural information. As the diffusion denoising process progresses, the features begin to form clearer local structures, and samples belonging to the same identity become increasingly aggregated. In the later denoising timesteps, the guided features evolve into wellseparated and compact clusters for each identity. It demonstrates that the denoiser effectively captures discriminative semantic cues and refines the representation toward the underlying identity-aware distribution. These observations confirm that our model progressively improves the discrimination of guided features through step-wise denoising. Noted that, the feature shift between $h \ = \ 6$ and $h \ = \ 7$ arises from the frequency-dependent reconstruction behavior in the diffusion model [80]. Specifically, early denoising steps mainly recover low-frequency components that encode global structures. Later denoising steps progressively restore high-frequency details, introducing fine-grained identity cues. Thus, it reflects the shift from low-frequency structure to high-frequency identity, rather than instability in the denoising process.

![](images/d6d21e2e408b437d06654e0994695d61b970992de785feedc492c3a0cbaf7c6d.jpg)

![](images/819a883ba8c9c6fb0d4956fd69303d62aa24612f707d9ff8885f34543724516b.jpg)

![](images/1d996c0380614006cbe0b957f6a5b15d601b2cc4d56a82624c09ecbad8984551.jpg)

![](images/2e2e979c0eb1ef8c8780512abe0adf39feed93f960fcd7b2d1136c548bbb50b4.jpg)

![](images/c7393bf3edffc386e220c7a538ea2b9e45771798ed87e5af1f7661759a081059.jpg)

![](images/96110c6c8b235e4bf3754f3d1a6a77a8250772b56b268c87b68eba2a2acad649.jpg)

![](images/7680e0d0f1a38b31f532f15976ef12e2c168003cf4c111d8596d64e497dcc513.jpg)

![](images/559acff6d20b367a3de493af91874f3fbf30111286b139b87c194df2146edd07.jpg)

![](images/9cf364922f61e403871205a16f6891d26b95e393522ebae60ca6eaaa244ebc89.jpg)

![](images/4f8b35229efe86b40205170d76be504ea889f0307241e090f1803d99a92f3215.jpg)  
Fig. 4: t-SNE visualization of guided features at successive denoising timesteps.

![](images/b792b57846e457851f7fd4454e121446ebe705f469a86ab75c6c794dd3685efb.jpg)  
Fig. 5: Feature distributions visualized by t-SNE. Different colors represent different identities.

![](images/5b98d5e087f2c2d32af3f8727517dfd8e9ff85efddb1dfd5b9e513c43527ffe5.jpg)

![](images/d6948bc0334ca1fe2f05186d842b016e062ca80ab41ff0d73c9650deec6c3226.jpg)

Fig. 6: Visualization of the cosine similarity distribution.  
![](images/61eb1e6e167b9e0f7908cd8916354a126e695971ef1d27ac911beef98bf1843e.jpg)

![](images/9dbb524dc8c34191590f3a4041cf33ae51dc7fd643cc01e17f5be4b394a2ef06.jpg)  
Fig. 7: Visualization of attention maps for different temporal queries. Darker red indicates higher attention weights.

Visual Effect of TCQ. To better understand the effect of TCQ, we visualize attention maps of learned queries at different timesteps. As shown in Fig. 7, our method attends to different information at different timesteps. In early denoising steps, it focuses more on the global structure of persons. In later denoising steps, it gradually shifts attention to discriminative regions highlighted by the red boxes. Moreover, as shown in Fig.7, our method can effectively suppress irrelevant distractions. Notably, in the last row of Fig. 7, even when an irrelevant person appears, our method still focuses on the correct target. In summary, these visualization results indicate that our method progressively generates discriminative features while avoiding the influence of irrelevant contents.

## V. CONCLUSION

In this paper, we propose DiffReID, an advanced generative framework for image-based object ReID. It learns identity-aware distributions and generates diverse guided features to complement visual features for robust representation. The framework consists of a Vision-guided Noise Generator (VNG) to sample visual-guided noises and a Light Weight Denoiser (LWD) to model identity-aware distributions. In addition, a Mutual Enhancement Constraint (MEC) is introduced to facilitate mutual learning between visual features and guided features, thereby improving both generalization and discrimination. Extensive experiments validate that our proposed method performs better than existing works on several widely used object ReID benchmarks.

## REFERENCES

[1] S. He, H. Luo, P. Wang, F. Wang, H. Li, and W. Jiang, “Transreid: Transformer-based object re-identification,” in ICCV, 2021, pp. 15 013– 15 022.

[2] S. Li, L. Sun, and Q. Li, “Clip-reid: Exploiting vision-language model for image re-identification without concrete text labels,” in AAAI, 2023, pp. 1405–1413.

[3] C. Yu, X. Liu, Y. Wang, P. Zhang, and H. Lu, “Tf-clip: Learning textfree clip for video-based person re-identification,” in AAAI, 2024, pp. 6764–6772.

[4] Y. Wang, P. Zhang, D. Wang, and H. Lu, “Other tokens matter: Exploring global and local features of vision transformers for object re-identification,” CVIU, vol. 244, p. 104030, 2024.

[5] M. Cao, Y. Lu, Z. Zeng, D. Yi, J. Wang, and M. Ye, “An empirical study of validating synthetic data for text-based person retrieval,” IEEE TIFS, vol. 21, pp. 6832–6844, 2026.

[6] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” NIPS, vol. 33, pp. 6840–6851, 2020.

[7] H. Cao, C. Tan, Z. Gao, Y. Xu, G. Chen, P.-A. Heng, and S. Z. Li, “A survey on generative diffusion models,” TKDE, vol. 36, no. 7, pp. 2814–2830, 2024.

[8] G. Wang, X. Huang, J. Sang et al., “Denoiserep: Denoising model for representation learning,” NeurIPS, vol. 37, pp. 40 032–40 056, 2024.

[9] X. Tao, J. Kong, M. Jiang, M. Lu, and A. Mian, “Unsupervised learning of intrinsic semantics with diffusion model for person re-identification,” TIP, vol. 33, pp. 6705–6719, 2024.

[10] K. Niu, H. Yu, X. Qian, T. Fu, B. Li, and X. Xue, “Synthesizing efficient data with diffusion models for person re-identification pre-training,” ML, vol. 114, no. 3, pp. 1–25, 2025.

[11] K. Feng, M. Ni, J. Jiang, Z. Zhang, and W. Zuo, “Multi-attentional distance for zero-shot classification with text-to-image diffusion model,” in ICME, 2024, pp. 1–6.

[12] S. Mukhopadhyay, M. Gwilliam, Y. Yamaguchi, V. Agarwal, N. Padmanabhan, A. Swaminathan, T. Zhou, J. Ohya, and A. Shrivastava, “Do text-free diffusion models learn discriminative visual representations?” in ECCV, 2024, pp. 253–272.

[13] D. Kim, X. Thomas, and D. Ghadiyaram, “Revelio: Interpreting and leveraging semantic information in diffusion models,” in ICCV, 2025, pp. 4659–4669.

[14] W. Li, X. Liu, J. Ma, and Y. Yuan, “Cliff: Continual latent diffusion for open-vocabulary object detection,” in ECCV, 2024, pp. 255–273.

[15] P. Han, C. Ye, J. Zhou, J. Zhang, J. Hong, and X. Li, “Latent-based diffusion model for long-tailed recognition,” in CVPR, 2024, pp. 2639– 2648.

[16] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in CVPR, 2022, pp. 10 684–10 695.

[17] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic text-to-image diffusion models with deep language understanding,” NIPS, vol. 35, pp. 36 479–36 494, 2022.

[18] A. C. Li, M. Prabhudesai, S. Duggal, E. Brown, and D. Pathak, “Your diffusion model is secretly a zero-shot classifier,” in ICCV, 2023, pp. 2206–2217.

[19] Y. Du, Z. Xiao, S. Liao, and C. Snoek, “Protodiff: Learning to learn prototypical networks by task-guided diffusion,” NeurIPS, vol. 36, pp. 46 304–46 322, 2023.

[20] Y. Sun, L. Zheng, Y. Yang, Q. Tian, and S. Wang, “Beyond part models: Person retrieval with refined part pooling (and a strong convolutional baseline),” in ECCV, 2018, pp. 480–496.

[21] G. Wang, Y. Yuan, X. Chen, J. Li, and X. Zhou, “Learning discriminative features with multiple granularities for person re-identification,” in ACMMM, 2018, pp. 274–282.

[22] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” NIPS, vol. 30, 2017.

[23] Z. Zhang, C. Lan, W. Zeng, X. Jin, and Z. Chen, “Relation-aware global attention for person re-identification,” in CVPR, 2020, pp. 3186–3195.

[24] G. Zhang, P. Zhang, J. Qi, and H. Lu, “Hat: Hierarchical aggregation transformers for person re-identification,” in ACMMM, 2021, pp. 516– 525.

[25] W. Li, C. Zou, M. Wang, F. Xu, J. Zhao, R. Zheng, Y. Cheng, and W. Chu, “Dc-former: Diverse and compact transformer for person reidentification,” in AAAI, 2023, pp. 1415–1423.

[26] H. Lu, X. Zou, and P. Zhang, “Learning progressive modality-shared transformers for effective visible-infrared person re-identification,” in AAAI, vol. 37, no. 2, 2023, pp. 1835–1843.

[27] P. Yan, X. Liu, P. Zhang, and H. Lu, “Learning convolutional multilevel transformers for image-based person re-identification,” Visual Intelligence, vol. 1, no. 1, p. 24, 2023.

[28] Y. Wang, P. Zhang, X. Liu, Z. Tu, and H. Lu, “Unity is strength: Unifying convolutional and transformeral features for better person reidentification,” IEEE TITS, vol. 26, no. 3, pp. 3713–3723, 2025.

[29] C. Yu, X. Liu, J. Zhu, Y. Wang, P. Zhang, and H. Lu, “Climb-reid: A hybrid clip-mamba framework for person re-identification,” in AAAI, vol. 39, no. 9, 2025, pp. 9589–9597.

[30] H. Li, Y. Liu, Y. Zhang, J. Li, and Z. Yu, “Breaking the paired sample barrier in person re-identification: Leveraging unpaired samples for domain generalization,” IEEE TITS, vol. 20, pp. 2357–2371, 2025.

[31] Y. Zhang, L. Kong, H. Li, and J. Wen, “Weakly supervised visibleinfrared person re-identification via heterogeneous expert collaborative consistency learning,” in ICCV, 2025, pp. 12 659–12 669.

[32] Y. Zhang, Y. Shang, and H. Li, “Dual-granularity cross-modal identity association for weakly-supervised text-to-person image matching,” in ACM MM, 2025, pp. 5247–5256.

[33] H. Li, Q. Hu, and Z. Hu, “Catalyst for clustering-based unsupervised object re-identification: Feature calibration,” in AAAI, vol. 38, no. 4, 2024, pp. 3091–3099.

[34] N. Siddiqui, F. A. Croitoru, G. K. Nayak, R. T. Ionescu, and M. Shah, “Dlcr: A generative data expansion framework via diffusion for clotheschanging person re-id,” in WACV, 2025, pp. 1608–1617.

[35] C. Jia, M. Luo, Z. Dang, G. Dai, X. Chang, and J. Wang, “Psdiff: Diffusion model for person search with iterative and collaborative refinement,” TCSVT, vol. 35, no. 6, pp. 5153–5165, 2024.

[36] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021, pp. 8748–8763.

[37] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013.

[38] A. Han, W. Huang, Y. Cao, and D. Zou, “On the feature learning in diffusion models,” in ICLR, 2025.

[39] S. Huang, Z. Lu, X. Cun, Y. Yu, X. Zhou, and X. Shen, “Deim: Detr with improved matching for fast convergence,” in CVPR, 2025, pp. 15 162– 15 171.

[40] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, “Shortcut learning in deep neural networks,” NMI, vol. 2, no. 11, pp. 665–673, 2020.

[41] Y. Wang, P. Zhang, C. Sun, D. Wang, and H. Lu, “What makes you unique? attribute prompt composition for object re-identification,” IEEE TCSVT, vol. 36, no. 3, pp. 3173–3184, 2026.

[42] H. Wang, J. Shen, Y. Liu, Y. Gao, and E. Gavves, “Nformer: Robust person re-identification with neighbor transformer,” in CVPR, 2022, pp. 7297–7307.

[43] Z. Sun, X. Nie, X. Xi, and Y. Yin, “Cfvmnet: A multi-branch network for vehicle re-identification based on common field of view,” in ACMMM, 2020, pp. 3523–3531.

[44] H. Zhu, W. Ke, D. Li, J. Liu, L. Tian, and Y. Shan, “Dual crossattention learning for fine-grained visual categorization and object reidentification,” in CVPR, 2022, pp. 4692–4702.

[45] F. Shen, J. Zhu, X. Zhu, Y. Xie, and J. Huang, “Exploring spatial significance via hybrid pyramidal graph network for vehicle re-identification,” TITS, vol. 23, no. 7, pp. 8793–8804, 2022.

[46] X. Zhang, R. Zhang, J. Cao, D. Gong, M. You, and C. Shen, “Partguided attention learning for vehicle instance retrieval,” TITS, vol. 23, no. 4, pp. 3048–3060, 2020.

[47] A. Zhang, Y. Gao, Y. Niu, W. Liu, and Y. Zhou, “Coarse-to-fine person re-identification with auxiliary-domain classification and second-order information bottleneck,” in CVPR, 2021, pp. 598–607.

[48] D. Meng, L. Li, X. Liu, Y. Li, S. Yang, Z.-J. Zha, X. Gao, S. Wang, and Q. Huang, “Parsing-based view-aware embedding network for vehicle re-identification,” in CVPR, 2020, pp. 7103–7112.

[49] T. Wang, H. Liu, P. Song, T. Guo, and W. Shi, “Pose-guided feature disentangling for occluded person re-identification based on transformer,” in AAAI, 2022, pp. 2540–2549.

[50] P. Khorramshahi, N. Peri, J.-c. Chen, and R. Chellappa, “The devil is in the details: Self-supervised attention for vehicle re-identification,” in ECCV, 2020, pp. 369–386.

[51] Z. Yu, Z. Huang, J. Pei, L. Tahsin, and D. Sun, “Semantic-oriented feature coupling transformer for vehicle re-identification in intelligent transportation system,” TITS, vol. 25, no. 3, pp. 2803–2813, 2024.

[52] M. Jia, Y. Sun, Y. Zhai, X. Cheng, Y. Yang, and Y. Li, “Semi-attention partition for occluded person re-identification,” in AAAI, 2023, pp. 998– 1006.

[53] A. Suprem and C. Pu, “Looking glamorous: Vehicle re-id in heterogeneous cameras networks with global and local attention,” arXiv preprint arXiv:2002.02256, 2020.

[54] M. Li, J. Liu, C. Zheng, X. Huang, and Z. Zhang, “Exploiting multiview part-wise correlation via an efficient transformer for vehicle reidentification,” TMM, vol. 25, pp. 919–929, 2021.

[55] S. He, W. Chen, K. Wang, H. Luo, F. Wang, W. Jiang, and H. Ding, “Region generation and assessment network for occluded person reidentification,” TIFS, vol. 19, pp. 120–132, 2023.

[56] H. Li, C. Li, A. Zheng, J. Tang, and B. Luo, “Mskat: Multiscale knowledge-aware transformer for vehicle re-identification,” TITS, vol. 23, no. 10, pp. 19 557–19 568, 2022.

[57] G. Zhang, Y. Zhang, T. Zhang, B. Li, and S. Pu, “Pha: Patch-wise highfrequency augmentation for transformer-based person re-identification,” in CVPR, 2023, pp. 14 133–14 142.

[58] J. Li and X. Gong, “Prototypical contrastive learning-based clip finetuning for object re-identification,” arXiv preprint arXiv:2310.17218, 2023.

[59] L. Jin, W. Ji, T.-S. Chua, and Z. Zheng, “Coarse-to-fine cross-modality generation for enhancing vehicle re-identification with high-fidelity synthetic data,” in ICRA, 2025, pp. 7319–7326.

[60] Z. Yu, Z. Huang, M. Hou, Y. Yan, Y. Liu, D. Sun, and H. Gregersen, “A multi-domain patch-differentiated transformer for vehicle reidentification,” EAAI, vol. 162, p. 112711, 2025.

[61] X. Zhou, X. Li, H. Zhou, X. Pang, J. Tian, X. Nie, C. Wang, and Y. Yin, “Adaptive division and priori reinforcement part learning network for vehicle re-identification,” PR, vol. 163, p. 111453, 2025.

[62] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian, “Scalable person re-identification: A benchmark,” in ICCV, 2015, pp. 1116–1124.

[63] E. Ristani, F. Solera, R. Zou, R. Cucchiara, and C. Tomasi, “Performance measures and a data set for multi-target, multi-camera tracking,” in ECCV, 2016, pp. 17–35.

[64] L. Wei, S. Zhang, W. Gao, and Q. Tian, “Person transfer gan to bridge domain gap for person re-identification,” in ICCV, 2018, pp. 79–88.

[65] W. Li, R. Zhao, T. Xiao, and X. Wang, “Deepreid: Deep filter pairing neural network for person re-identification,” in CVPR, 2014, pp. 152– 159.

[66] X. Liu, W. Liu, H. Ma, and H. Fu, “Large-scale vehicle re-identification in urban surveillance videos,” in ICME, 2016, pp. 1–6.

[67] H. Ni, J. Song, X. Luo, F. Zheng, W. Li, and H. T. Shen, “Meta distribution alignment for generalizable person re-identification,” in CVPR, 2022, pp. 2487–2496.

[68] D. Kinga, J. B. Adam et al., “A method for stochastic optimization,” in ICLR, vol. 5, no. 6, 2015.

[69] Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, “Random erasing data augmentation,” in AAAI, 2020, pp. 13 001–13 008.

[70] S. Liao and L. Shao, “Interpretable and generalizable person reidentification with query-adaptive convolution and temporal lifting,” in ECCV, 2020, pp. 456–474.

[71] ——, “Transmatcher: Deep image matching through transformers for generalizable person re-identification,” NIPS, vol. 34, pp. 1992–2003, 2021.

[72] ——, “Graph sampling based deep metric learning for generalizable person re-identification,” in CVPR, 2022, pp. 7359–7368.

[73] H. Ni, Y. Li, L. Gao, H. T. Shen, and J. Song, “Part-aware transformer for generalizable person re-identification,” in ICCV, 2023, pp. 11 280– 11 289.

[74] K. Chen, P. Fang, Z. Ye, and L. Zhang, “Multi-scale explicit matching and mutual subject teacher learning for generalizable person reidentification,” TCSVT, vol. 34, no. 9, pp. 8881–8895, 2024.

[75] Y. Dai, X. Li, J. Liu, Z. Tong, and L.-Y. Duan, “Generalizable person reidentification with relevance-aware mixture of experts,” in CVPR, 2021, pp. 16 145–16 154.

[76] Y. Zhao, Z. Zhong, F. Yang, Z. Luo, Y. Lin, S. Li, and S. Nicu, “Learning to generalize unseen domains via memory-based multi-source metalearning for person re-identification,” in CVPR, 2021, pp. 6277–6286.

[77] Z. Chen, W. Wang, Z. Zhao, F. Su, A. Men, and Y. Dong, “Clusterinstance normalization: A statistical relation-aware normalization for generalizable person re-identification,” TMM, vol. 26, pp. 3554–3566, 2023.

[78] Y. Wang, X. Hu, L. Wang, P. Zhang, and H. Lu, “Sd-reid: view-aware stable diffusion for aerial-ground person re-identification,” IEEE TIP, vol. 35, pp. 5686–5697, 2026.

[79] L. Van der Maaten and G. Hinton, “Visualizing data using t-sne.” JMLR, vol. 9, no. 11, 2008.

[80] M. Yi, A. Li, Y. Xin, and Z. Li, “Towards understanding the working mechanism of text-to-image diffusion model,” in NeurIPS, vol. 37, 2024, pp. 55 342–55 369.