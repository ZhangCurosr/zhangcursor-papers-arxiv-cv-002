# Steering the Flow: Inverting Face Recognition Models via Gradient-Guided Flow Matching

Ye Lu, Shen Wang, Zhaoyang Zhang, Yihan Yan, Li Liu, Runze Liu, and Fanghui Sun

Abstract—Model Inversion Attacks (MIAs) aim to reconstruct representative training samples of target identities from face recognition models, exposing critical security vulnerabilities. Existing methods typically rely on indirect guidance or highly stochastic guidance, making it difficult to stably optimize generation trajectories toward target facial images. In this paper, we propose Steering Flow Model Inversion (SFMI), a novel two-stage white-box model inversion method that reformulates inversion as a trajectory-steering task. Specifically, Step I, Learning a Generic Flow Matching Prior, pre-trains a generic unconditional Flow Matching model to encode the manifold of human faces as a robust prior. Step II, Attacking with Progressive Guidance Scheduler (PGS), injects time-dependent target-specific gradients during sampling. By backpropagating through the target model to obtain gradients from intermediate generated states, PGS progressively injects adaptive guidance signals into the vector field. This process effectively steers the current generative flow from random noise toward the high-density regions of the target class. Under an identity-disjoint cross-evaluation setting using the CelebA dataset, SFMI achieves an ACC of 0.9248, an FID of 22.61, and an LPIPS of 0.3874 on the ArcFace target. Extensive experiments on multiple target models demonstrate that SFMI achieves competitive state-of-the-art performance in attack success and visual fidelity under the evaluated white-box protocol.

Index Terms—Model inversion attack, face recognition, privacy attack, deep learning.

## I. INTRODUCTION

Deep Neural Networks (DNNs) have emerged as the backbone of modern biometric authentication, achieving superhuman accuracy in tasks such as Face Recognition (FR) [1]. The efficacy of these models is predicated on their capacity to extract high-level semantic features from massive-scale datasets. However, this reliance on extensive training data engenders a fundamental tension between utility and privacy. Theoretical and empirical evidence suggests that DNNs tend to unintentionally “memorize” specific training samples rather than merely learning generalized patterns. This phenomenon, coupled with the pervasive deployment of FR systems in critical infrastructure, poses severe privacy risks. In light of stringent regulations such as the GDPR, auditing the potential leakage of sensitive biometric information has become a paramount research imperative.

![](images/75c02ed5022b04d8c7abad7b29ad17c911790038ed6d6e69a45c228ea30d20af.jpg)  
Fig. 1. Basic concept of the Model Inversion Attack (MIA).

Among the emerging privacy threats, Model Inversion Attacks (MIAs) reveal that DNNs—especially face recognition models—are inherently vulnerable to identity leakage. In an MIA, the adversary reconstructs a representative facial image of a target identity by exploiting the information encoded in a trained model. Generally, MIAs are categorized into black-box and white-box settings based on the adversary’s knowledge. While the black-box setting restricts access to model queries, the white-box setting assumes full access to the model’s parameters and architecture. In this paper, we focus on the white-box scenario, as illustrated in Fig. 1. Here, the adversary leverages full access to the model parameters to backpropagate gradients for a specific label (e.g., “ID: 999”). This feedback iteratively guides random noise to evolve into a facial image that semantically aligns with the target identity. In IoT deployments, such access can arise when an adversary extracts an on-device FR model from a physically compromised smart camera, access-control terminal, or mobile device through firmware inspection or exposed local storage. The extracted model can then be analyzed offline using external computing resources, so inversion need not run on the resource-constrained device itself. We therefore treat whitebox inversion as a realistic high-access, worst-case privacy audit rather than a universal adversarial capability.

Despite the well-defined threat model, recovering highfidelity images remains a formidable challenge due to the high dimensionality of the search space. Early approaches framed inversion as a direct optimization problem in the pixel space, aiming to maximize the likelihood of the target class. However, the optimization landscape of pixel values is highly non-convex, rendering these methods prone to entrapment in local minima. Consequently, the reconstructed images often suffer from pronounced visual artifacts and fail to preserve semantic coherence. To mitigate this, subsequent works leveraged Generative Adversarial Networks (GANs) [2], [3] as prior knowledge, optimizing a latent code within the generator’s manifold. While GAN priors improve visual plausibility, they introduce a structural mismatch: the mapping from the latent space to the image space is inherently non-linear and nonsmooth. This results in unstable gradient backpropagation, impeding the optimization from effectively traversing the manifold toward the precise features of the target identity.

In recent years, Diffusion Models (DMs) [4]–[6] and their continuous-time generalizations, specifically Flow Matching (FM) [7], have superseded GANs as the state-of-the-art in generative modeling. While these models offer superior capabilities in modeling complex data distributions, adapting them for adversarial inversion remains non-trivial. Most diffusion-based inversion methods rely on Stochastic Differential Equation (SDE)-based sampling. However, the stochastic noise injected along the SDE trajectory can mislead, or even interfere with, the sampling process toward the target class. Crucially, these methods lack a mathematically grounded mechanism to effectively rectify the generation path based on the target model’s response. Consequently, they struggle to explicitly navigate the generative flow toward the specific identity regions defined by the target classifier.

In this paper, we bridge these gaps by proposing Steering Flow Model Inversion (SFMI), a Flow Matching-based whitebox model inversion method that reformulates inversion as a trajectory-steering task. Unlike diffusion models that rely on complex stochastic differential equations, Flow Matching models data generation through an Ordinary Differential Equation (ODE), providing relatively smooth and stable velocity fields. We leverage this property to treat the inversion process not as a static optimization, but as guiding a flow from a simple prior distribution to the target class manifold. Specifically, SFMI operates in two stages: Step I, Learning a Generic Flow Matching Prior, which pre-trains an unconditional Flow Matching model to encapsulate a robust human face prior; and Step II, Attacking with Progressive Guidance Scheduler (PGS), which injects target-specific gradients during integration. By backpropagating through the target model at intermediate integration steps, PGS injects adaptive gradients into the velocity field, effectively “steering” the flow toward the high-density regions of the target identity.

We evaluate SFMI under an identity-disjoint protocol across diverse face recognition targets, including ResNet-based backbones, commonly used face recognition models such as Arc-Face and CosFace, the modern transformer-based ViT model, and MobileFaceNet, which is widely adopted in IoT and edgeface-recognition scenarios. SFMI achieves an average attack accuracy above 90% with favorable perceptual similarity, demonstrating strong and competitive performance relative to state-of-the-art methods under the evaluated white-box protocol.

Our contributions can be summarized as follows:

• We propose SFMI, the first white-box model inversion method that leverages Flow Matching as the generative prior and formulates inversion as ODE velocity-field steering, thereby mitigating the optimization instability of previous GAN-based and stochastic diffusion-based attacks.

• We design the Progressive Guidance Scheduler (PGS) that seamlessly integrates the target model’s feedback into the ODE solver. By dynamically modulating the vector field with adaptive gradient signals, we achieve fine-grained control over the identity recovery process.

• Under an identity-disjoint cross-evaluation protocol, extensive experiments on diverse face recognition targets demonstrate that SFMI delivers strong and competitive performance relative to state-of-the-art optimizationbased and generative MIA methods, particularly in attack accuracy and visual fidelity.

The remainder of this paper is organized as follows. Section II reviews related work on optimization-based and generative-prior model inversion. Section III introduces the proposed SFMI method, including Learning a Generic Flow Matching Prior and Attacking with Progressive Guidance Scheduler (PGS). Section IV presents experimental settings and quantitative and qualitative evaluations. Finally, Section VI concludes the paper and discusses future directions.

## II. RELATED WORK

## A. Deep Face Recognition

Deep learning architectures have substantially reshaped visual representation learning. Convolutional Neural Networks (CNNs) established a dominant paradigm for extracting hierarchical spatial features, Residual Networks (ResNet) [8] enabled stable training of deeper models, and Vision Transformers (ViT) [9] further introduced self-attention for global contextual modeling.

Driven by these advances, face recognition has shifted from handcrafted-feature pipelines to deep learning-based paradigms [1], [10]–[12], where models learn identitydiscriminative embeddings in an end-to-end manner. Training objectives have also evolved from metric-learning formulations, such as FaceNet [13], to margin-based softmax losses that enhance intra-class compactness and inter-class separabil ity. In this work, we evaluate SFMI on representative targets spanning these developments: Face.evoLVe and IR-152 are conventional CNN-based FR backbones; CosFace [14] and ArcFace [15] represent widely used margin-based recognition objectives; MobileFaceNet [16] is a lightweight model designed for accurate real-time face verification on mobile, IoT, and edge devices; and the ViT model represents modern transformer-based FR designs. These heterogeneous targets provide different model capacities, embedding geometries, and deployment characteristics for evaluating inversion difficulty.

## B. Model Inversion Attacks

Model Inversion Attacks (MIAs) pose a severe privacy threat by aiming to reconstruct sensitive training data or infer private attributes from trained machine learning models [17], [18]. The concept was initially formalized by exploiting prediction confidence information from simple classifiers to recover recognizable images [19]. Subsequent research rapidly expanded MIAs into rigorous white-box settings [20], [21], formulating the attack as an explicit optimization problem in continuous pixel space. By utilizing gradient descent [22], attackers attempted to synthesize random noise into inputs that maximize the target class probability. However, because this optimization is performed directly in the extremely highdimensional and highly non-convex natural image pixel space, traditional methods are prone to being trapped in poor local optima, often resulting in failure modes such as semantically meaningless reconstructions dominated by abstract highfrequency noise.

Related privacy and face-recognition studies in IoT contexts have investigated distinct but complementary directions, including secure face identification under fog computing [23], sparse low-rank representation for IoT face recognition [24], and practical feature inference attacks in vertical federated prediction [25], collectively highlighting privacy-related risks in IoT scenarios.

To mitigate the mathematical ill-posedness of direct pixel optimization, a profound paradigm shift occurred with the introduction of auxiliary generative priors. Zhang et al. [26] pioneered the Generative Model Inversion (GMI) attack, which leverages a pre-trained Generative Adversarial Network (GAN) to constrain the adversarial search space strictly to a natural image manifold. Following this breakthrough, a proliferation of generative MIAs emerged to enhance reconstruction fidelity and attack transferability. Notable advancements include modeling private data distributions (KEDMI) [27], decoupling target models from priors via Plug & Play Attacks (PPA) [28], and utilizing pseudo-label-guided conditional GANs (PLG) to disentangle search spaces [29]. More recently, researchers have focused on dissecting the prior architecture itself by exploiting intermediate GAN features (IFGMI) [30]. Diffusion-based MIA methods have also emerged as a recent direction: DiffMI [31] fine-tunes a target-specific conditional diffusion prior to improve consistency with the target identity, while FGMIA [32] first recovers feature encodings through a surrogate model and then performs one-shot feature-guided generation without iterative correction from the target model. Several other recent works have also explored related research directions [33]–[35]. Other explorations span variational inversion formulations [36], optimization objective refinements [37], and adversarial prior exploitations [38], [39].

To clarify the methodological differences among representative MIAs, Table I summarizes their publication years, priors, evaluation data, image resolutions, attack knowledge, and gradient dependence.

As for defense methods, existing model-inversion defenses [17] mainly fall into two categories. The first category strengthens model training or pre-training, using robust representation learning, mutual-information regularization [40], transfer-learning-based robustness [41], trainingdata deduplication [42], trapdoor-based defense [43], or label smoothing [44] to reduce sensitive memorization. The second category processes model outputs, using noise injection, prediction purification [45], adversarial perturbation [46], or other post-processing strategies to weaken exploitable signals. For face recognition models, however, extracting identitydiscriminative features is necessary for utility, while model inversion attacks exploit this feature extraction process, revealing the trade-off between recognition performance and privacy security.

Despite improving visual plausibility, existing generative MIAs still suffer from fundamental limitations when reconstructing identity-specific targets. On the one hand, GANbased approaches rely on latent-space optimization, where the latent-to-image mapping is highly non-linear and often non-smooth, leading to unstable optimization dynamics and frequent convergence to suboptimal solutions. On the other hand, diffusion-based approaches typically adopt Stochastic Differential Equation (SDE)-based sampling, whose injected stochastic perturbations can continuously deflect the generation trajectory away from the target class signal. Although DiffMI introduces target-specific conditional diffusion finetuning, its DDPM-based sampling still faces potential trajectory deviation and requires multi-step cascaded backpropagation, which can mislead the generative prior during inversion. FGMIA reduces iterative optimization by using recovered feature encodings, but it does not use target-model feedback to progressively correct the generation trajectory. As a result, both paradigms struggle to consistently recover precise, identity-specific target face images, especially against modern margin-based FR models (e.g., CosFace and ArcFace in Sec. II-A).

This fundamental optimization instability directly motivates our proposed Steering Flow Model Inversion (SFMI). SFMI uses Flow Matching to construct a deterministic ODE trajectory and progressively injects target-model feedback during sampling. By reformulating the generative prior as a deterministic ODE with smooth trajectories and introducing a progressive guidance mechanism, SFMI provides a trajectorysteering backbone. This mechanism effectively bridges the semantic gap, enabling target-specific adversarial gradients to stably and precisely steer the generative trajectory toward the target identity and thereby overcome the limitations of both highly non-convex GANs and stochastic diffusion models.

## III. METHOD

## A. Preliminaries

Problem Definition. Let $f _ { \theta } : \mathcal { X }  \mathcal { Y }$ denote a pre-trained target face recognition model parameterized by $\theta ,$ which maps the continuous image space $\dot { \mathcal { X } } \subseteq \mathbb { R } ^ { C \times H \times W }$ to the probability simplex Y over $K$ identities. Given a target label $y _ { \mathrm { t a r g e t } } \in$ $\{ 1 , \ldots , K \}$ , the goal of a Model Inversion Attack (MIA) is to recover an image that captures the private visual characteristics of that identity from the information encoded in the trained model. We formulate this objective as an optimization problem over $\mathcal { X } \colon$ the adversary seeks an input $x ^ { * }$ whose prediction is aligned with $y _ { \mathrm { { t a r g e t } } } ,$ , while a regularization term enforces natural image priors. Mathematically, the adversary solves the following minimization problem:

$$
\boldsymbol { x } ^ { * } = \underset { \boldsymbol { x } \in \mathcal { X } } { \arg \operatorname* { m i n } } ~ \mathcal { L } _ { \mathrm { i n v } } \big ( f _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) , \boldsymbol { y } _ { \mathrm { t a r g e t } } \big ) + \lambda \mathcal { R } ( \boldsymbol { x } ) ,\tag{1}
$$

where $\mathcal { L } _ { \mathrm { i n v } }$ represents the classification objective that penalizes deviations from the target class, $\mathcal { R } ( x )$ serves as a prior constraint to ensure the semantic plausibility of the reconstructed image, and λ is a hyperparameter balancing the two terms.

TABLE I  
COMPARISON OF REPRESENTATIVE MODEL INVERSION ATTACK METHODS IN RELATED WORK, INCLUDING PUBLICATION YEAR, PRIOR TYPE, DATASET,
<table><tr><td>Method</td><td>Year</td><td>Prior</td><td>Dataset</td><td>Resolution</td><td>Scenario</td><td>Gradients</td></tr><tr><td>MIA [19]</td><td>2015</td><td>None</td><td>MNIST</td><td>Various</td><td>Black/White-box</td><td>Optional</td></tr><tr><td>YMIA [22]</td><td>2019</td><td>None</td><td>AT&amp;T Faces</td><td>Various</td><td>Black/White-box</td><td>Optional</td></tr><tr><td>GMI [26]</td><td>2020</td><td>Unconditional GAN</td><td>CelebA, FaceScrub</td><td>64</td><td>White-box</td><td>Yes</td></tr><tr><td>KEDMI [27]</td><td>2021</td><td>Knowledge-enriched GAN</td><td>CelebA, FaceScrub</td><td>64</td><td>White-box</td><td>Yes</td></tr><tr><td>FMI [39]</td><td>2022</td><td>DCGAN</td><td>FaceScrub, CelebA</td><td>64</td><td>White-box</td><td>Yes</td></tr><tr><td>PPA [28]</td><td>2022</td><td>StyleGAN2</td><td>CelebA, FFHQ</td><td>112/224</td><td>White-box</td><td>Yes</td></tr><tr><td>PLGMI [29]</td><td>2023</td><td>Pseudo-label cGAN</td><td>CelebA, FaceScrub, PubFig</td><td>64/112</td><td>White-box</td><td>Yes</td></tr><tr><td>DiffMI [31]</td><td>2024</td><td>Conditional DDPM</td><td>CelebA, FFHQ</td><td>64/112</td><td>White-box</td><td>Yes</td></tr><tr><td>IFGMI [30]</td><td>2024</td><td>StyleGAN2</td><td>CelebA, FaceScrub</td><td>64/112/224</td><td>White-box</td><td>Yes</td></tr><tr><td>FGMIA [32]</td><td>2025</td><td>DDPM</td><td>CelebA, FFHQ</td><td>64/112/224</td><td>Black/White-box</td><td>No</td></tr><tr><td>SFMI (Ours)</td><td>2026</td><td>Flow Matching</td><td>CelebA, FFHQ</td><td>64/112/224</td><td>White-box</td><td>Yes</td></tr></table>

TABLE II  
NOTATION USED FOR DATASETS AND KEY VARIABLES.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $\mathcal { D } _ { \mathrm { p r i v } }$ </td><td>Private dataset used to train the target face recognition model.</td></tr><tr><td> $\mathcal { D } _ { \mathrm { p u b } }$ </td><td>Public auxiliary dataset available to the adversary and</td></tr><tr><td>Ytarget</td><td>strictly identity-disjoint from  $\begin{array} { r } { \mathcal { D } _ { \mathrm { p r i v } } . } \end{array}$  Target identity label to be reconstructed by the attack</td></tr><tr><td> $x _ { 0 }$ </td><td>Initial Gaussian noise sample drawn from  $p _ { 0 } = \mathcal { N } ( 0 , { \bf { I } } )$ </td></tr><tr><td> $x _ { 1 }$ </td><td>Clean face sample or the final endpoint of the generative</td></tr><tr><td> $x t$ </td><td>trajectory. Intermediate state along the flow trajectory at time  $t \in$ </td></tr><tr><td> $u _ { t } ( \cdot )$ </td><td>[0, 1]. Ground-truth conditional vector field defined by the OT</td></tr><tr><td> $v _ { \phi } ( \cdot )$ </td><td>path. Learned velocity field induced by the Flow Matching prior</td></tr><tr><td> $\mathcal { M } _ { \phi }$ </td><td> $\mathcal { M } _ { \phi } .$  Flow Matching prior model parameterized by φ.</td></tr><tr><td> $\mathcal { L } _ { \mathrm { F M } }$ </td><td>Flow Matching training loss for learning  $v _ { \phi } .$ </td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { i d } }$ </td><td>Identity supervision objective used to guide inversion to-</td></tr><tr><td> $g ( x _ { t } )$ </td><td>ward  $y _ { \mathrm { t a r g e t } } .$  Raw identity-gradient signal computed with respect to the</td></tr><tr><td> $g _ { \mathrm { n o r m } } ( x _ { t } )$ </td><td>current state xt. Normalized guidance signal scaled to the velocity magni-</td></tr><tr><td> $\mathbb { E }$ </td><td>tude. Expectation over sampled time steps, noise samples, and public data samples.</td></tr></table>

In SFMI, R(x) is not evaluated or optimized as an explicit regularization loss. Instead, the Flow Matching model trained on public face data implicitly imposes the facial prior, while progressive target-model gradients steer the sampling trajectory toward the target class within the learned face manifold.

Adversary Knowledge. We operate under a white-box threat model, which grants the adversary comprehensive access to the target system. In this setting, the adversary is assumed to possess full knowledge of the model architecture and its trained parameters $\theta ,$ thereby enabling the explicit computation of gradients via backpropagation. Furthermore, to facilitate the generation of realistic facial structures, the adversary is assumed to have access to a generic public face dataset $\mathcal { D } _ { \mathrm { p u b } }$ that remains strictly identity-disjoint from the private training set $\mathcal { D } _ { \mathrm { p r i v } }$ used for $f _ { \theta } .$ . In practical terms, the public dataset shares neither identities nor samples with the private data. This independent data separation allows the adversary to learn transferable facial priors while fully preserving the assumption that private data are inaccessible.

## B. Steering Flow Model Inversion (SFMI)

1) Method Overview: In this section, we first provide an overview of Steering Flow Model Inversion (SFMI). As illustrated in Fig. 2, the method consists of two tightly coupled stages: Stage I provides a smooth generative prior over human faces, while Stage II injects target-aware supervision to steer this prior toward identity-specific reconstruction.

In the upper part, Stage I trains a Flow Matching model through velocity-field matching so that samples from a Gaussian prior can be transported to the human-face manifold. Specifically, the model learns a smooth vector field that maps samples from $p _ { 0 }$ to samples from $p _ { 1 }$ , yielding a stable transport trajectory and a generic facial prior for inversion. This prior captures broad facial structure and appearance statistics, which helps constrain subsequent attack trajectories to remain on realistic face manifolds.

In the lower part, Stage II starts from Gaussian noise and performs guided sampling toward a target identity. During sampling, clean estimates predicted from intermediate states are fed into the target model to compute identity-related gradients, and these gradients are injected back into the update direction to progressively move intermediate states toward the target face class. Meanwhile, PGS dynamically modulates the guidance strength over time (e.g., warm-up, sustain, and gradual decay), balancing identity alignment and visual fidelity, and thus enabling stable trajectory correction throughout generation.

2) Learning a Generic Flow Matching Prior: In this stage, we adopt Flow Matching (FM [7], [47]) because model inversion requires a generative prior that is both smooth and controllable when transporting samples from noise to realistic face images. FM directly learns a time-dependent vector field, which provides a stable and differentiable trajectory for optimization and avoids the instability introduced by highly irregular mappings or stochastic perturbations. The role of this stage is therefore to learn an unconditional facial prior that maps a simple Gaussian distribution to the humanface manifold, providing a reliable generative backbone for subsequent identity-specific steering.

![](images/9eede8b0ebe3de843a37db54f2e37885b4f24184264ff6e2ef39a24efab59643.jpg)  
Fig. 2. Overview of SFMI, which consists of two stages: learning a generic Flow Matching prior and performing progressive gradient-guided attack.

Let $\mathcal { D } _ { \mathrm { p u b } }$ denote a public face dataset accessible to the adversary, which is disjoint from the private dataset $\mathcal { D } _ { \mathrm { p r i v } }$ used to train the target model. We define $p _ { 1 } ( x )$ as the empirical data distribution supported by $\mathcal { D } _ { \mathrm { p u b } }$ . Conversely, let $p _ { 0 } ( x )$ denote a simple prior noise distribution, which we specify as a standard d-dimensional Gaussian distribution, i.e., $p _ { 0 } = \mathcal { N } ( 0 , \mathbf { I } )$ . Our objective is to learn a time-dependent vector field that pushes samples from $p _ { 0 }$ to $p _ { 1 }$

We define this generative process via an Optimal Transport (OT [7]) displacement map. This map constructs the simplest possible trajectory—a straight line—between a noise sample $x _ { 0 } \sim p _ { 0 }$ and a data sample $x _ { 1 } \sim p _ { 1 }$ . The state $x _ { t }$ at any time $t \in [ 0 , 1 ]$ is given by linear interpolation:

$$
x _ { t } = \psi _ { t } ( x _ { 0 } , x _ { 1 } ) = ( 1 - t ) x _ { 0 } + t x _ { 1 } .\tag{2}
$$

Differentiating Eq. (2) with respect to time yields the groundtruth conditional vector field, denoted as $u _ { t }$ , which represents the target velocity for this specific path:

$$
u _ { t } ( x \mid x _ { 0 } , x _ { 1 } ) = x _ { 1 } - x _ { 0 } .\tag{3}
$$

While standard Flow Matching directly approximates $u _ { t }$ with a velocity-predicting network, we opt for an x-prediction parameterization following prior practice [48]. We train a neural network $\mathcal { M } _ { \phi } ( x _ { t } , t )$ to estimate the clean original image $\hat { x } _ { 1 }$ from the noisy state $x _ { t } .$ . Based on the geometry of the OT path, the relationship between the current state $x _ { t } ,$ the destination $x _ { 1 } .$ , and the velocity v is derived as $v = ( x _ { 1 } - x _ { t } ) / ( 1 - t )$

Consequently, the vector field $v _ { \phi }$ induced by our network is formulated as:

$$
v _ { \phi } ( x _ { t } , t ) = \frac { \mathcal { M } _ { \phi } ( x _ { t } , t ) - x _ { t } } { 1 - t } .\tag{4}
$$

Following the design choice in [48], we optimize the model with a velocity-matching loss (v-loss). To prioritize training on the most critical temporal regions, we sample the time step t from a Logit-Normal distribution parameterized by mean m and standard deviation $\sigma ,$ rather than a uniform distribution. The loss is computed by minimizing the discrepancy between the induced velocity field $v _ { \phi }$ (Eq. 4) and the ground-truth target velocity $u _ { t } \ ( { \mathrm { E q . } } \ 3 ) { \mathrm { : } }$

$$
\mathcal { L } _ { \mathrm { F M } } ( \phi ) = \mathbb { E } _ { t , x _ { 0 } , x _ { 1 } } \left[ \Vert v _ { \phi } ( x _ { t } , t ) - u _ { t } \Vert _ { 2 } ^ { 2 } \right] .\tag{5}
$$

The complete training process for the generic Flow Matching prior is detailed in Algorithm 1. At each iteration, we first sample a mini-batch of clean faces $x _ { 1 } \sim \mathcal { D } _ { \mathrm { p u b } }$ and a mini-batch of Gaussian noise samples $x _ { 0 } \sim \mathcal { N } ( 0 , \mathbf { I } )$ , then draw time steps t from a Logit-Normal distribution with l $\mathrm { o g i t } ( t ) \sim \mathcal { N } ( m , \sigma ^ { 2 } )$ to emphasize informative temporal regions. Next, we construct intermediate states along the OT path via linear interpolation and compute the corresponding target velocities $u _ { t } = x _ { 1 } - x _ { 0 }$ Given each pair $( x _ { t } , t )$ , the network $\mathcal { M } _ { \phi }$ predicts ${ \hat { x } } _ { 1 }$ , which is converted to the induced velocity $v _ { \phi }$ using $\mathrm { E q . ~ 4 }$ . We then evaluate the v-loss in Eq. 5 as the batch-wise discrepancy between $v _ { \phi }$ and $u _ { t } ,$ , and update parameters $\phi$ by gradient descent. By repeating this procedure, the model learns a stable, smooth vector field that reliably transports Gaussian noise toward realistic face samples.

Algorithm 1 Training of the Generic Flow Matching Prior   
Input: Public dataset ${ \mathcal { D } } _ { \mathrm { p u b } } ,$ max iterations ${ \overline { { N } } } ,$ batch size $B ,$   
learning rate $\eta ,$ sampling hyperparameters $m , \sigma .$   
Output: Trained model $\mathcal { M } _ { \phi } .$   
1: Initialize network parameters $\phi .$   
2: for iter = 1 to $N$ do   
3: Sample data batch $\{ x _ { 1 } ^ { ( i ) } \} _ { i = 1 } ^ { B } \sim \mathcal { D } _ { \mathrm { p u b } } .$   
4: Sample noise batch $\{ x _ { 0 . } ^ { ( i ) } \} _ { i = 1 } ^ { B } \sim \mathcal { N } ( 0 , \mathbf { I } ) .$   
5: Sample time steps $\{ t ^ { ( i ) } \} _ { i = 1 } ^ { B } \sim$ Logit-Norma $( m , \sigma )$   
$\{ \mathrm { l o g i t } ( t ) \sim \mathcal { N } ( m , \bar { \sigma ^ { 2 } } ) \}$   
6: for $i = 1$ to $B$ do   
7: Construct intermediate state via OT path:   
8: $x _ { t } ^ { ( i ) } \gets ( 1 - t ^ { ( i ) } ) x _ { 0 } ^ { ( i ) } + t ^ { ( i ) } x _ { 1 } ^ { ( i ) }$   
9: Forward pass to predict clean data:   
10: $\hat { x } _ { 1 } ^ { ( i ) }  \dot { \mathcal { M } } _ { \phi } ( x _ { t } ^ { ( i ) } , t ^ { ( i ) } )$   
11: Compute induced velocity (Eq. 4):   
12: $v _ { \phi } ^ { ( i ) } \stackrel { \cdot } {  } ( \hat { x } _ { 1 } ^ { ( i ) } - x _ { t } ^ { ( i ) } ) / ( 1 - \stackrel { \cdot } { t ^ { ( i ) } } ) ^ { \cdot }$   
13: Compute target velocity (Eq. 3):   
14: $u _ { t } ^ { ( i ) } \dot {  } x _ { 1 } ^ { ( i ) } - x _ { 0 } ^ { ( i ) }$   
15: end for   
16: Compute Loss: $\begin{array} { r } { \mathcal { L }  \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \| v _ { \phi } ^ { ( i ) } - u _ { t } ^ { ( i ) } \| _ { 2 } ^ { 2 } } \end{array}$   
17: Update parameters: $\phi  \phi - \eta \nabla _ { \phi } \mathcal { L }$   
18: end for   
19: return Trained model $\mathcal { M } _ { \phi } .$

3) Progressive Gradient-Guided Attack: With the generic flow matching prior $\mathcal { M } _ { \phi }$ trained, the second stage of SFMI focuses on recovering the specific target identity. We formulate this as a trajectory steering problem within the ODE formulation. The objective is to navigate the generative flow such that the final sample minimizes the identity mismatch with respect to the target label $y _ { \mathrm { t a r g e t } }$

To achieve this, we intervene in the numerical integration process. In a standard unconditional generation, the trajectory is governed solely by the learned velocity field $v _ { \phi } ( x _ { t } , t )$ . In our adversarial setting, we introduce a guidance term derived from the target classifier $f _ { \theta } .$ . Crucially, since the target classifier is trained on clean images, directly feeding the noisy state $x _ { t }$ into $f _ { \theta }$ would yield uninformative gradients. Instead, we leverage the x-prediction capability of our pre-trained prior.

At each time step t, we first pass the current noisy state $x _ { t }$ through the Flow Matching model to obtain a clean image estimate $\hat { x } _ { 1 } = \mathcal { M } _ { \phi } ( x _ { t } , t )$ . This estimated clean image is then fed into the target classifier to compute the inversion loss. Formally, the loss function is defined as the composition of the classifier and the prior model:

$$
\mathcal { L } _ { \mathrm { i n v } } ( x _ { t } ) = \mathcal { L } _ { \mathrm { i d } } \big ( f _ { \theta } ( \mathcal { M } _ { \phi } ( x _ { t } , t ) ) , y _ { \mathrm { t a r g e t } } \big ) .\tag{6}
$$

where ${ \mathcal { L } } _ { \mathrm { i d } }$ denotes the identity supervision objective. In practice, we instantiate ${ \mathcal { L } } _ { \mathrm { i d } }$ with the max-margin loss (MMLoss) adopted by PLGMI [29]. Let $z _ { \theta } ( \hat { x } _ { 1 } ) ~ \in ~ \mathbb { R } ^ { K }$ be the logit vector of the target classifier for the estimated clean image $\hat { x } _ { 1 } = \mathcal { M } _ { \phi } ( x _ { t } , t )$ . MMLoss is defined as

$$
\mathcal { L } _ { \mathrm { M M } } ( z _ { \theta } ( \hat { x } _ { 1 } ) , y _ { \mathrm { t a r g e t } } ) = \operatorname* { m a x } _ { k \neq y _ { \mathrm { t a r g e t } } } z _ { \theta , k } ( \hat { x } _ { 1 } ) - z _ { \theta , y _ { \mathrm { t a r g e t } } } ( \hat { x } _ { 1 } ) .\tag{7}
$$

Minimizing Eq. 7 enlarges the target logit relative to the strongest non-target logit, which encourages identitydiscriminative reconstruction.

To steer the flow, we require the gradient with respect to the current state $x _ { t } ,$ not the estimated outcome $\hat { x } _ { 1 }$ . We obtain this by backpropagating the error signal through the frozen target classifier and, significantly, through the frozen Flow Matching prior itself:

$$
\begin{array} { r } { g ( x _ { t } ) = \nabla _ { x _ { t } } \mathcal { L } _ { \mathrm { i n v } } ( x _ { t } ) = \nabla _ { x _ { t } } \left[ \mathcal { L } _ { \mathrm { i d } } \big ( f _ { \theta } ( \mathcal { M } _ { \phi } ( x _ { t } , t ) ) , y _ { \mathrm { t a r g e t } } \big ) \right] . } \end{array}\tag{8}
$$

This end-to-end backpropagation ensures that the guidance signal $g ( x _ { t } )$ accurately reflects how an infinitesimal change in the current noisy state $x _ { t }$ influences the final identity objective, accounting for the manifold projection learned by $\mathcal { M } _ { \phi }$ . Before injection, we normalize the raw gradient to a velocity-proportional length:

$$
g _ { \mathrm { n o r m } } ( x _ { t } ) = \frac { g ( x _ { t } ) } { \| g ( x _ { t } ) \| _ { 2 } + \epsilon } \cdot \| v _ { \phi } ( x _ { t } , t ) \| _ { 2 } ,\tag{9}
$$

where ϵ is a small constant for numerical stability. This preserves the adversarial direction while matching its scale to the FM velocity.

We then modify the original velocity field by injecting this gradient signal. The rectified velocity field $\tilde { v } ( x _ { t } , t )$ is formulated as:

$$
\tilde { v } ( x _ { t } , t ) = v _ { \phi } ( x _ { t } , t ) - \gamma ( t ) \cdot g _ { \mathrm { n o r m } } ( x _ { t } ) .\tag{10}
$$

As illustrated in Fig. 2, this correction preserves the natural FM trajectory while steering sampling toward a face recognized as the target identity.

Progressive Guidance Schedule. Instead of a constant guidance scale, which often leads to saturation artifacts or optimization instability, we design a time-dependent schedule $\gamma ( t )$ that modulates the steering strength dynamically. The schedule consists of four phases: warm-up, sustain, decay, and relaxation. Mathematically, it is defined as:

$$
\gamma ( t ) = M \cdot \left\{ \begin{array} { l l } { V _ { \operatorname* { m a x } } \cdot \frac { t } { t _ { 0 } } , } & { 0 \leq t \leq t _ { 0 } , } \\ { V _ { \operatorname* { m a x } } , } & { t _ { 0 } < t \leq t _ { 1 } , } \\ { \frac { V _ { \operatorname* { m a x } } } { 2 } \left[ 1 + \cos \left( \pi \frac { t - t _ { 1 } } { t _ { 2 } - t _ { 1 } } \right) \right] , } & { t _ { 1 } < t \leq t _ { 2 } , } \\ { 0 , } & { t _ { 2 } < t \leq 1 . } \end{array} \right.\tag{11}
$$

where M represents the global magnitude scale, and $V _ { \mathrm { m a x } }$ defines the peak intensity. The thresholds $t _ { 0 } , t _ { 1 } , t _ { 2 }$ delineate the phases. The linear warm-up $\left( t \leq t _ { 0 } \right)$ prevents abrupt trajectory shifts when noise is high. The cosine decay $( t _ { 1 } < t \leq t _ { 2 } )$ ensures a smooth transition, and the final zero-guidance phase $\left( t \right) > t _ { 2 } )$ allows the Flow Matching prior to refine the image without external interference, thereby reinforcing adherence to the learned face manifold, stabilizing the final refinement trajectory, and preserving high visual fidelity. The PGS diagram in Fig. 2 visualizes this schedule as four consecutive warmup, sustain, decay, and relaxation stages, providing an intuitive counterpart to the piecewise definition above.

To mitigate discretization errors and ensure the steered trajectory adheres smoothly to the face manifold, we adopt a second-order predictor-corrector scheme inspired by the Heun sampler [49]. As detailed in Algorithm 2, this mechanism involves a two-stage evaluation at each step: a predictor step to estimate a provisional next state using the gradient at the current position, and a corrector step to refine the update using the gradient at the predicted position. This approach effectively corrects the trajectory curvature induced by the adversarial guidance.

Algorithm 2 SFMI: Progressive Gradient-Guided Attack   
Input: Target model $f _ { \theta } ,$ Prior model ${ \mathcal { M } } _ { \phi } ,$ Target label $y _ { \mathrm { t a r g e t } } ,$   
Steps $N .$   
Output: Reconstructed image $x _ { 1 }$   
1: Sample initial noise $x _ { 0 } \sim \mathcal { N } ( 0 , \mathbf { I } ) .$   
2: Set initial state $x _ { t _ { 0 } } \gets x _ { 0 }$ and step size $\Delta t = 1 / N$   
3: for $i = 0$ to $N - 1$ do   
4: $t _ { i } \gets i / N ; \quad t _ { i + 1 } \gets ( i + 1 ) / N .$   
5: Compute $\gamma ( t _ { i } )$ and $\gamma ( t _ { i + 1 } )$ using Eq. 11.   
6: // Stage 1: Predictor   
7: $\hat { x } _ { 1 } \gets \mathcal { M } _ { \phi } ( x _ { t _ { i } } , t _ { i } ) .$   
8: $v _ { \mathrm { c u r r } } \gets ( \hat { x } _ { 1 } - x _ { t _ { i } } ) / ( 1 - t _ { i } ) .$   
9: $g _ { \mathrm { c u r r } } \gets \nabla _ { x _ { t _ { i } } } \mathcal { L } _ { \mathrm { i d } } \big ( f _ { \theta } ( \mathcal { M } _ { \phi } ( x _ { t _ { i } } , t _ { i } ) ) , y _ { \mathrm { t a r g e t } } \big ) .$   
10: $\begin{array} { r } { g _ { \mathrm { c u r r } }  \frac { g _ { \mathrm { c u r r } } ^ { \mathrm { s } } } { \| g _ { \mathrm { c u r r } } \| _ { 2 } + \epsilon } \cdot \| v _ { \mathrm { c u r r } } \| _ { 2 } . } \end{array}$   
11: $\tilde { v } _ { 1 }  v _ { \mathrm { c u r r } } - \gamma ( t _ { i } ) \cdot g _ { \mathrm { c u r r } } .$   
12: $\hat { x } _ { t _ { i + 1 } } \gets x _ { t _ { i } } + \tilde { v } _ { 1 } \cdot \Delta t .$   
13: // Stage 2: Corrector   
14: $\hat { x } _ { 1 } ^ { \prime } \gets \mathcal { M } _ { \phi } ( \hat { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) .$   
15: $v _ { \mathrm { p r e d } }  ( \hat { x } _ { 1 } ^ { \prime } - \hat { x } _ { t _ { i + 1 } } ) / ( 1 - t _ { i + 1 } ) .$   
16: $\dot { g _ { \mathrm { p r e d } } }  \nabla _ { \hat { x } _ { t _ { i + 1 } } } \mathcal { L } _ { \mathrm { i d } } \big ( f _ { \theta } ( \mathcal { M } _ { \phi } ( \hat { x } _ { t _ { i + 1 } } , t _ { i + 1 } ) ) , y _ { \mathrm { t a r g e t } } \big )$   
17: $\begin{array} { r } { g _ { \mathrm { p r e d } }  \frac { g _ { \mathrm { p r e d } } } { \| g _ { \mathrm { p r e d } } \| _ { 2 } + \epsilon } \cdot \| v _ { \mathrm { p r e d } } \| _ { 2 } . } \end{array}$   
18: $\tilde { v } _ { 2 }  v _ { \mathrm { p r e d } } - \gamma ( t _ { i + 1 } ) \cdot g _ { \mathrm { p r e d } } .$   
19: // Update State   
20: $\begin{array} { r } { x _ { t _ { i + 1 } } \gets x _ { t _ { i } } + \frac { \Delta t } { 2 } ( \tilde { v } _ { 1 } + \tilde { v } _ { 2 } ) } \end{array}$   
21: end for   
22: return $x _ { 1 } = x _ { t _ { N } } .$

## IV. EXPERIMENTS

In this section, we evaluate the proposed Steering Flow Model Inversion (SFMI) method from four complementary perspectives. We first describe the experimental protocol, including disjoint dataset construction, target face recognition models, and cross-evaluation metrics. We then compare SFMI with representative white-box model inversion baselines through both quantitative results and qualitative visualizations. Next, we conduct component-level ablations to isolate the contributions of the prior model, guidance scheduler, and loss design. Finally, we provide a detailed analysis of progressive guidance variants to explain the trajectory-steering behavior behind SFMI’s performance.

## A. Experimental Setup

Datasets. We enforce an identity-disjoint protocol throughout all experiments: identities in the adversary’s public prior data never overlap with those used to train the target models. Consistent with Sec. III, $\mathcal { D } _ { \mathrm { p r i v } }$ denotes the private targettraining dataset, and $\mathcal { D } _ { \mathrm { p u b } }$ denotes the public auxiliary dataset available to the adversary. From CelebA [50], we construct

CelebA-priv as $\mathcal { D } _ { \mathrm { p r i v } }$ with 1000 identities and 30,000 images for target-model training, and build a non-overlapping 30,000- image CelebA-pub split as the default $\mathcal { D } _ { \mathrm { p u b } }$ for learning the unconditional face prior. Since prior learning is unconditional, $\mathcal { D } _ { \mathrm { p u b } }$ requires no identity annotations. To evaluate crossdistribution transferability, we also construct FFHQ-pub from FFHQ [51] as another identity-disjoint 30,000-image auxiliary prior dataset. Thus, CelebA-pub and FFHQ-pub use the same public-data scale while preventing overlap with CelebA-priv. All images follow the same face-recognition preprocessing pipeline, including face detection, five-point landmark localization, 2D partial affine alignment, interpolation, cropping, and resizing to the target-model resolution.

Target Models. We evaluate SFMI against six face recognition targets covering different architectures, objectives, and input resolutions. Face.evoLVe [52] and IR-152 [8] are conventional CNN-based face recognition models evaluated at $6 4 \times 6 4$ resolution. CosFace [14] and ArcFace [15] are two widely used margin-based face recognition models and are evaluated at $1 1 2 \times 1 1 2$ resolution. MobileFaceNet [16] is a lightweight face recognition architecture commonly used in mobile, IoT, and edge-device scenarios, and ViT [9] represents a modern transformer-based face recognition model; both are evaluated at $1 1 2 \times 1 1 2$ resolution in the main comparison. These resolutions refer to the detected, aligned, cropped, and resized face inputs consumed by the FR models rather than raw high-resolution images. In the ablation study, we further include $2 2 4 \times 2 2 4$ ArcFace and MobileFaceNet targets to examine higher-resolution behavior. For each target, we use an Internet-public pretrained model as the backbone and train only the final classification layer on $\mathcal { D } _ { \mathrm { p r i v } }$ for 10 epochs. To ensure a rigorous and unbiased assessment of the recovered identities and avoid self-evaluation, we strictly adopt a crossevaluation protocol, where the quantitative performance of an attack against a specific target model is reported as the average score evaluated by the other models.

Evaluation Metrics. We evaluate model inversion performance using three complementary metrics: Attack Accuracy (ACC), Frechet Inception Distance (´ FID), and Learned Perceptual Image Patch Similarity (LPIPS). Together, they characterize identity recovery success, global visual realism, and fine-grained perceptual consistency. ACC quantifies semantic identity recovery under the cross-evaluation protocol by computing the average proportion of reconstructed images that are recognized as the target identity by non-target evaluators, thus directly reflecting attack effectiveness. During the attack, the adversary only accesses the target model and has no access to the evaluation model; therefore, ACC measures whether the reconstruction is recognized as the target identity by independent evaluators rather than merely fitting the attacked model. FID [53] measures distribution-level similarity and overall image fidelity via the Wasserstein-2 distance between reconstructed samples and real private training images in feature space, where lower values indicate fewer artifacts and better photorealism. LPIPS [54] further evaluates structural and textural alignment between reconstructions and groundtruth faces using deep perceptual features, and lower LPIPS indicates closer perceptual resemblance to the target identity.

Implementation Details. We implement $\mathcal { M } _ { \phi }$ using the Flow Matching architecture in [48], which is a ViT/DiTlike Transformer with a $1 6 \times 1 6$ patch size and 16 attention heads. The FM prior is trained for 500k optimization steps using AdamW, a per-batch learning rate of $2 \times 1 0 ^ { - 5 }$ and a learning-rate schedule with linear warmup followed by cosine decay. Instead of uniformly sampling t, we use $\mathrm { l o g i t } ( t ) \ \sim \ \mathcal { N } ( - 0 . 8 , 0 . 8 ^ { 2 } )$ . During the attack, we use 50 sampling steps by default, with $M = 0 . 3 , t _ { 0 } = 0 . 1 , t _ { 1 } = 0 . 3 ,$ $t _ { 2 } = 0 . 7$ , and $V _ { \mathrm { m a x } } = 1 . 0$ . To balance the facial prior and identity guidance, SFMI directly predicts the original image with the FM prior, injects normalized guidance only through the selected PGS stages, and strictly controls the guidance magnitude; in practice, we recommend keeping $M ~ \leq ~ 0 . 3 5$ because larger values can push the trajectory too far and cause facial distortion or artifacts. At the 112×112 resolution setting, training one FM prior takes about 18 hours on a single RTX 5090 GPU with 32 GB of memory; at the 224×224 resolution setting, it takes about 50 hours on the same GPU. Once trained, the generic FM prior can be reused for different target models. The 50-step attack takes about 0.23 seconds per image at 112×112 resolution and 0.47 seconds per image at $2 2 4 \times 2 2 4$ resolution.

## B. Comparison with State-of-the-Art Methods

To verify the effectiveness of SFMI, we conduct a comprehensive comparison with existing white-box model inversion methods. Specifically, we evaluate six target face recognition models (Face.evoLVe, IR-152, CosFace, ArcFace, MobileFaceNet, and ViT), following prior settings [29], [32] to ensure consistency with standard white-box MIA evaluation practice. We compare SFMI against five representative baselines: GMI [26], PPA [28], PLGMI [29], IFGMI [30], and FGMIA [32]. For fairness, all methods are implemented from official codebases and evaluated under the same protocol in Sec. IV-A, including strict identity-disjoint data splits, model-aligned preprocessing, 30,000-image public auxiliary data at comparable scale, and the same cross-evaluation rule. Specifically, SFMI uses CelebA-pub as the default $\mathcal { D } _ { \mathrm { p u b } }$ for training the Flow Matching prior. For the baselines, GMI, PLGMI, and FGMIA use priors trained following their original procedures with the same-scale public auxiliary data, while PPA and IFGMI reuse the prior models released by the original papers because their official protocols rely on these pretrained generative priors. All baseline attacks use the parameters recommended by the corresponding papers or official implementations.

Table III jointly summarizes the prior architecture, perattack computational budget, prior-training time, and attack time. PPA and IFGMI use released StyleGAN2 checkpoints rather than priors retrained on our platform. For context, the official StyleGAN2 implementation reports 9 days and 18 hours to train its FFHQ model at $1 0 2 4 \times 1 0 2 4$ resolution on eight Tesla V100 GPUs; because this upstream cost differs in hardware, resolution, and training setup from our RTX 5090 measurements, we exclude it from the direct timing comparison. Since SFMI uses second-order Heun integration, its 50-step attack corresponds to 100 model evaluations (NFE). For SFMI, we repeat the full process three times, from Flow Matching prior training to attacking each target class; for each target class, eight images are sampled and used to compute the reported mean and standard deviation.

Table IV reports the main quantitative comparison between SFMI and representative white-box MIA baselines on six target face recognition models, using ACC, FID, and LPIPS as complementary evaluation metrics. SFMI achieves the highest ACC on all evaluated target models and the lowest LPIPS across all targets, indicating strong identity recovery together with superior perceptual consistency. For FID, SFMI attains the best score on Face.evoLVe and remains highly competitive on the other target models, suggesting that the gain in attack success is not obtained at the cost of noticeable realism degradation. Taken together, these results demonstrate that SFMI offers a favorable trade-off between semantic fidelity and visual quality, and that this advantage generalizes across target models with different architectural and loss-design characteristics.

Figure 3 further provides visual comparisons. The first row shows real private images, and rows 2–7 correspond to GMI, PPA, PLGMI, IFGMI, FGMIA, and SFMI, respectively. In most cases, SFMI preserves facial geometry and identityrelated local traits more clearly, while reducing common artifacts such as blurry textures and unstable high-frequency noise observed in baseline reconstructions. We also observe that several previous methods are, to some extent, misled during optimization, which causes their reconstruction trajectories to drift away from target identities and leads to substantial inconsistencies with the corresponding private faces. These qualitative observations are well aligned with the quantitative trends in Table IV, further confirming the robustness and visual fidelity of SFMI across diverse identities.

## C. Ablation Studies

To rigorously validate the necessity and individual contributions of the core components within the proposed SFMI method, we conduct a comprehensive ablation study. We systematically isolate the effects of the generative prior model, the auxiliary dataset distribution, the Progressive Guidance Scheduler (PGS), the optimization loss design, the input resolution, a simple output-perturbation defense, and a training-phase BiDO defense. Specifically, the variants include: replacing the FM prior with a DDPM prior paired with a DDIM sampler (DDPM prior + DDIM sampler); training the prior on FFHQpub instead of CelebA-pub (w/ FFHQ Prior); replacing PGS with a constant multiplier (w/ Constant Guidance); replacing the MM loss with cross-entropy (w/ CE Loss (replace MM)); increasing the target input resolution (w/ 224 Resolution); injecting noise into the target model output before comput ing attack guidance (w/ Output Perturbation); and training the target model with Bilateral Dependency Optimization (BiDO) [57] (w/ BiDO Defense). All ablation experiments are evaluated on the primary private dataset $( \mathcal { D } _ { \mathrm { p r i v } } )$ , with ArcFace as the main robust margin-based target and MobileFaceNet as an additional lightweight target for checking whether the component-level conclusions generalize across architectures. Following the standard MIA evaluation protocol, each target identity is attacked by sampling eight reconstructed images, and the reported metrics are averaged over all target identities rather than over a single example. To further verify that the component-level conclusions are not specific to ArcFace, we additionally repeat the same key ablations on MobileFaceNet, as summarized in Table VI.

TABLE III  
CONFIGURATION AND COMPUTATIONAL-COST SUMMARY UNDER THE 112 × 112 RESOLUTION COSFACE SETTING ON AN RTX 5090 GPU WITH 32GB MEMORY. CANDIDATE SCREENING IS LISTED SEPARATELY FROM GRADIENT UPDATES, AND LOCALLY MEASURED PRIOR-TRAINING TIMES EXCLUDE RELEASED PRETRAINED CHECKPOINTS.
<table><tr><td>Method</td><td>Prior type</td><td>Attack budget</td><td>Prior training</td><td>Attack time</td></tr><tr><td>GMI [26]</td><td>WGAN-GP [55]</td><td>1500 latent-optimization steps</td><td>15 h</td><td>0.16 s/pic</td></tr><tr><td>PPA [28]</td><td>StyleGAN2 [56]</td><td>5000-candidate screening + 70 latent-optimization steps</td><td>– (pretrained)</td><td>0.42 s/pic</td></tr><tr><td>PLGMI [29]</td><td>Pseudo-label cGAN</td><td>600 latent-optimization steps</td><td>17 h</td><td>0.20 s/pic</td></tr><tr><td>IFGMI [30]</td><td>StyleGAN2 [56]</td><td>5000-candidate screening + 170 staged updates  $( 7 0 + 4 \times 2 5 )$ </td><td>– (pretrained)</td><td>0.27 s/pic</td></tr><tr><td>FGMIA [32]</td><td>Feature-guided DDPM</td><td>50 denoising steps</td><td>13 h</td><td>0.40 s/pic</td></tr><tr><td>SFMI (Ours)</td><td>Flow Matching</td><td>50 Heun steps (100 NFE)</td><td>18 h</td><td>0.23 s/pic</td></tr></table>

TABLE IV

COMPREHENSIVE WHITE-BOX MIA COMPARISON AMONG GMI [26], PPA [28], PLGMI [29], IFGMI [30], FGMIA [32], AND SFMI ON SIX TARGET MODELS (FACE.EVOLVE, IR-152, COSFACE, ARCFACE, MOBILEFACENET, AND VIT). METRICS INCLUDE ACC (↑), FID (↓), AND LPIPS (↓) UNDER THE CROSS-EVALUATION PROTOCOL. BEST RESULTS ARE HIGHLIGHTED IN BOLD. ALL METHODS ARE REPORTED AS MEAN ± STANDARD DEVIATION OVER THREE REPEATED RUNS.
<table><tr><td rowspan="2">Method</td><td colspan="3">Target: Face.evoLVe (64 × 64)</td><td colspan="3">Target: IR-152 (64 × 64)</td><td colspan="3">Target: CosFace (112 × 112)</td></tr><tr><td>ACC ↑</td><td>FID ↓</td><td>LPIPS ↓</td><td>ACC ↑</td><td>FID ↓</td><td>LPIPS ↓</td><td>ACC ↑</td><td>FID ↓</td><td>LPIPS ↓</td></tr><tr><td>GMI [26]</td><td> $0 . 2 5 3 7 \pm 0 . 0 1 5 6$ </td><td> $2 7 . 9 7 \pm 0 . 9 0$ </td><td> $0 . 5 3 1 9 \pm 0 . 0 0 8 7$ </td><td> $0 . 1 3 8 2 \pm 0 . 0 1 7 6$ </td><td> $\mathbf { 2 7 . 6 1 \pm 0 . 4 0 }$ </td><td> $0 . 5 5 9 5 \pm 0 . 0 0 8 1$ </td><td> $0 . 0 6 0 8 \pm 0 . 0 1 9 0$ </td><td> $2 6 . 1 4 \pm 0 . 7 1$ </td><td> $0 . 5 3 5 0 \pm 0 . 0 1 4 7$ </td></tr><tr><td>PPA [28]</td><td> $0 . 7 7 6 0 \pm 0 . 0 1 2 0$ </td><td> $2 6 . 6 1 \pm 0 . 3 1$ </td><td> $0 . 5 0 8 0 \pm 0 . 0 1 0 0$ </td><td> $0 . 6 7 0 8 \pm 0 . 0 1 2 0$ </td><td> $2 9 . 8 4 \pm 0 . 6 1$ </td><td> $0 . 5 2 6 0 \pm 0 . 0 0 4 2$ </td><td>0.7480 ± 0.0099</td><td> $2 2 . 7 2 \pm 0 . 5 5$ </td><td> $0 . 5 4 4 8 \pm 0 . 0 1 8 0$ </td></tr><tr><td>PLGMI [29]</td><td> $0 . 3 2 2 4 \pm 0 . 0 2 2 4$ </td><td> $5 0 . 2 6 \pm 2 . 2 4$ </td><td> $0 . 4 8 3 5 \pm 0 . 0 0 5 6$ </td><td> $0 . 4 4 8 9 \pm 0 . 0 2 5 1$ </td><td> $4 9 . 6 2 \pm 0 . 3 3$ </td><td> $0 . 4 8 8 8 \pm 0 . 0 1 0 6$ </td><td> $0 . 5 6 2 5 \pm 0 . 0 0 5 5$ </td><td> $5 7 . 5 2 \pm 1 . 3 1$ </td><td> $0 . 4 8 0 4 \pm 0 . 0 1 1 6$ </td></tr><tr><td>IFGMI [30]</td><td> $0 . 8 2 4 1 \pm 0 . 0 1 0 2$ </td><td> $3 2 . 5 3 \pm 0 . 4 6$ </td><td> $0 . 4 8 7 3 \pm 0 . 0 0 7 7$ </td><td> $0 . 8 0 6 2 \pm 0 . 0 0 5 7$ </td><td> $3 0 . 6 1 \pm 0 . 5 2$ </td><td> $0 . 4 7 5 3 \pm 0 . 0 0 7 5$ </td><td> $0 . 7 1 5 7 \pm 0 . 0 1 2 9$ </td><td> $3 1 . 2 4 \pm 0 . 2 7$ </td><td> $0 . 4 5 0 7 \pm 0 . 0 0 6 4$ </td></tr><tr><td>FGMIA [32]</td><td> $0 . 8 7 4 9 \pm 0 . 0 0 4 8$ </td><td> $3 4 . 8 8 \pm 0 . 6 2$ </td><td> $0 . 4 3 0 6 \pm 0 . 0 0 4 8$ </td><td> $0 . 8 7 1 9 \pm 0 . 0 0 5 4$ </td><td> $3 1 . 3 2 \pm 0 . 9 9$ </td><td> $0 . 4 6 3 1 \pm 0 . 0 0 3 3$ </td><td> $0 . 9 1 1 6 \pm 0 . 0 0 7 4$ </td><td> $2 2 . 4 0 \pm 0 . 8 4$ </td><td> $0 . 4 0 4 6 \pm 0 . 0 0 2 9$ </td></tr><tr><td>SFMI (Ours)</td><td> $\mathbf { 0 . 8 9 8 0 \pm 0 . 0 0 8 6 }$ </td><td> $\mathbf { 2 5 . 8 6 \pm 0 . 2 1 }$ </td><td> $\mathbf { 0 . 4 2 4 8 \pm 0 . 0 0 8 1 }$ </td><td>0.9103 ± 0.0073</td><td> ${ \bf 2 7 . 6 1 \pm 0 . 3 1 }$ </td><td> $\mathbf { 0 . 4 4 1 9 \pm 0 . 0 0 7 7 }$ </td><td> $\mathbf { 0 . 9 1 5 0 \pm 0 . 0 0 6 6 }$ </td><td> ${ \bf 2 2 . 3 7 \pm 0 . 7 4 }$ </td><td>0.3966 ± 0.0041</td></tr><tr><td>Method</td><td colspan="3">Target: ArcFace (112 × 112)</td><td colspan="3">Target: MobileFaceNet (112 × 112)</td><td colspan="3">Target: ViT (112 × 112)</td></tr><tr><td></td><td>ACC ↑</td><td>FID ↓</td><td>LPIPS ↓</td><td>ACC ↑</td><td>FID ↓</td><td>LPIPS ↓</td><td>ACC ↑</td><td>FID ↓</td><td>LPIPS ↓</td></tr><tr><td>GMI [26]</td><td> $0 . 0 3 7 9 \pm 0 . 0 2 5 8$ </td><td> $2 9 . 8 9 \pm 0 . 5 8$ </td><td>0.5161 ± 0.0093</td><td> $0 . 1 4 4 0 \pm 0 . 0 1 5 8$ </td><td> $2 6 . 5 0 \pm 0 . 1 4$ </td><td> $0 . 5 0 5 9 \pm 0 . 0 0 1 0$ </td><td>0.0418 ± 0.0065</td><td> $2 9 . 7 1 \pm 1 . 0 9$ </td><td> $0 . 5 4 0 0 \pm 0 . 0 0 6 8$ </td></tr><tr><td>PPA [28]</td><td> $0 . 4 1 9 1 \pm 0 . 0 1 0 1$ </td><td> $2 2 . 8 4 \pm 0 . 8 7$ </td><td> $0 . 5 5 9 4 \pm 0 . 0 1 4 3$ </td><td> $0 . 8 0 3 6 \pm 0 . 0 0 7 0$ </td><td> $2 7 . 4 9 \pm 0 . 0 7$ </td><td> $0 . 4 5 4 9 \pm 0 . 0 0 5 4$ </td><td> $0 . 3 6 9 8 \pm 0 . 0 1 8 2$ </td><td> ${ \bf 2 4 . 8 2 \pm 0 . 3 8 }$ </td><td> $0 . 5 5 0 2 \pm 0 . 0 0 8 2$ </td></tr><tr><td>PLGMI [29]</td><td> $0 . 2 3 5 7 \pm 0 . 0 2 9 2$ </td><td> $4 2 . 1 9 \pm 1 . 2 8$ </td><td> $0 . 4 5 8 7 \pm 0 . 0 1 0 1$ </td><td> $0 . 8 6 4 8 \pm 0 . 0 0 8 1$ </td><td> $3 9 . 2 5 \pm 0 . 3 6$ </td><td> $0 . 4 3 8 3 \pm 0 . 0 1 2 1$ </td><td> $0 . 2 5 5 5 \pm 0 . 0 0 9 6$ </td><td> $4 4 . 0 9 \pm 0 . 4 6$ </td><td> $0 . 4 5 1 1 \pm 0 . 0 0 7 9$ </td></tr><tr><td>IFGMI [30]</td><td> $0 . 7 5 4 4 \pm 0 . 0 2 6 3$ </td><td> $3 5 . 7 9 \pm 0 . 5 1$ </td><td> $0 . 4 4 2 4 \pm 0 . 0 0 1 7$ </td><td> $0 . 8 0 2 8 \pm 0 . 0 0 7 3$ </td><td> $3 4 . 8 4 \pm 0 . 3 7$ </td><td> $0 . 4 0 9 9 \pm 0 . 0 0 4 5$ </td><td> $0 . 7 4 7 1 \pm 0 . 0 1 1 5$ </td><td> $3 3 . 5 7 \pm 0 . 4 9$ </td><td> $0 . 4 5 4 8 \pm 0 . 0 0 4 7$ </td></tr><tr><td>FGMIA [32]</td><td> $0 . 8 8 8 0 \pm 0 . 0 0 8 7$ </td><td> ${ \bf 2 1 . 7 2 \pm 0 . 0 8 }$ </td><td> $0 . 4 0 8 8 \pm 0 . 0 0 1 2$ </td><td> $0 . 9 0 0 9 \pm 0 . 0 0 6 4$ </td><td> $2 2 . 9 8 \pm 0 . 3 4$ </td><td> $0 . 3 8 8 4 \pm 0 . 0 0 9 0$ </td><td> $0 . 8 0 7 4 \pm 0 . 0 5 8 0$ </td><td> $2 5 . 6 9 \pm 0 . 1 9$ </td><td> $0 . 3 9 5 2 \pm 0 . 0 0 5 0$ </td></tr><tr><td>SFMI (Ours)</td><td>0.9248 ± 0.0089</td><td> $2 2 . 6 1 \pm 0 . 3 0$ </td><td> $\mathbf { 0 . 3 8 7 4 } \pm 0 . 0 0 5 9$ </td><td> $\mathbf { 0 . 9 3 3 5 \pm 0 . 0 1 1 6 }$ </td><td> ${ \bf 2 2 . 6 7 \pm 0 . 7 4 }$ </td><td> $\mathbf { 0 . 3 7 8 6 \pm 0 . 0 0 8 8 }$ </td><td>0.8735 ± 0.0147</td><td> $2 5 . 6 1 \pm 0 . 6 5$ </td><td> $\mathbf { 0 . 3 9 1 8 \pm 0 . 0 1 3 7 }$ </td></tr></table>

TABLE VI  
TABLE V  
DETAILED COMPONENT ABLATION OF SFMI ON THE MOBILEFACENET TARGET: DDPM prior + DDIM sampler, w/ FFHQ Prior, w/ Constant Guidance, w/ CE Loss (replace MM), w/ 224 Resolution, w/ Output Perturbation, AND w/ BiDO Defense, WITH SFMI (FULL) AS THE REFERENCE. EVALUATION METRICS ARE ACC (↑), FID (↓), AND LPIPS (↓).  
DETAILED COMPONENT ABLATION OF SFMI ON THE ARCFACE TARGET: DDPM prior + DDIM sampler, w/ FFHQ Prior, w/ Constant Guidance, w/ CE Loss (replace MM), w/ 224 Resolution, w/ Output Perturbation, AND w/ BiDO Defense, SFMI (FULL) . E METRICS ARE ACC (↑), FID (↓), AND LPIPS (↓).
<table><tr><td>Variant</td><td>ACC ↑</td><td>FID ↓</td><td>LPIPS ↓</td></tr><tr><td>DDPM prior + DDIM sampler</td><td>0.1086</td><td>76.50</td><td>0.6084</td></tr><tr><td>w/ FFHQ Prior</td><td>0.8995</td><td>24.67</td><td>0.4564</td></tr><tr><td>w/ Constant Guidance</td><td>0.1658</td><td>80.66</td><td>0.5556</td></tr><tr><td>w/ CE Loss (replace MM)</td><td>0.8842</td><td>23.99</td><td>0.4200</td></tr><tr><td>w/ 224 Resolution</td><td>0.8599</td><td>25.21</td><td>0.4593</td></tr><tr><td>w/ Output Perturbation</td><td>0.8858</td><td>21.77</td><td>0.4320</td></tr><tr><td>w/ BiDO Defense</td><td>0.8611</td><td>24.63</td><td>0.4291</td></tr><tr><td>SFMI (Full)</td><td>0.9248</td><td>22.61</td><td>0.3874</td></tr></table>

As observed in Table V, replacing the FM prior with a DDPM-trained prior and sampling it with DDIM (DDPM prior + DDIM sampler) still results in a substantial degradation in semantic recovery. Following the original DDIM formulation, this baseline uses the DDPM noise-prediction model to estimate the denoised image $\scriptstyle { \hat { x } } _ { 0 }$ at each step, computes the targetmodel gradient through the predicted denoising path, and uses this gradient as guidance for DDIM sampling. Although DDIM removes step-wise sampling randomness, we observe that its guided trajectory is less robust and can more easily drift away from the natural image manifold. We attribute this to two factors: first, SFMI directly predicts the original image rather than relying on noise estimation, which is consistent with the denoising-oriented observation in [48]; second, the DDPM prior induces a more irregular noise-to-face trajectory than Flow Matching, making target-class optimization more difficult. In contrast, the FM velocity field provides a smoother backbone for progressive identity-gradient steering.

<table><tr><td>Variant</td><td>ACC ↑</td><td>FID ↓</td><td>LPIPS↓</td></tr><tr><td>DDPM prior + DDIM sampler</td><td>0.0879</td><td>62.70</td><td>0.5848</td></tr><tr><td>w/ FFHQ Prior</td><td>0.8183</td><td>21.78</td><td>0.4390</td></tr><tr><td>w/ Constant Guidance</td><td>0.0387</td><td>56.71</td><td>0.5958</td></tr><tr><td>w/ CE Loss (replace MM)</td><td>0.8983</td><td>23.80</td><td>0.4291</td></tr><tr><td>w/ 224 Resolution</td><td>0.8382</td><td>26.00</td><td>0.4237</td></tr><tr><td>w/ Output Perturbation</td><td>0.9012</td><td>23.01</td><td>0.4110</td></tr><tr><td>w/ BiDO Defense</td><td>0.8572</td><td>26.39</td><td>0.3978</td></tr><tr><td>SFMI (Full)</td><td>0.9335</td><td>22.67</td><td>0.3786</td></tr></table>

![](images/a44895df90bc5ab38317726a301917ee823831ddcc19f543a900eea40adf4b15.jpg)  
Fig. 3. Qualitative visual comparison across MIA methods. The first row shows private ground-truth images, and rows 2–7 correspond to GMI, PPA, PLGMI, IFGMI, FGMIA, and SFMI, respectively.

Furthermore, to evaluate our method’s sensitivity to distribution shifts, we train the prior on the identity-disjoint FFHQpub subset (w/ FFHQ Prior) rather than the distributionally aligned CelebA-pub subset. Both public auxiliary subsets contain 30,000 images, so this comparison controls the auxiliarydata scale while changing only the public-prior distribution. The results indicate that while this dataset discrepancy introduces a minor degradation in performance, the overall attack remains highly effective. This demonstrates that SFMI does not strictly rely on matching the target’s data distribution, exhibiting strong cross-distribution robustness.

Moreover, ablating the progressive schedule in favor of a static multiplier (w/ Constant Guidance) severely compromises the practical usability of the method. Applying a rigid, constant gradient force uniformly across all integration steps irrecoverably corrupts the highly noisy initial states. This abrupt intervention forces the generative trajectory completely off the natural face manifold, leading to catastrophic optimization collapse. Consequently, the ACC plummets and the generated outputs suffer from severe, unrecognizable visual artifacts (evidenced by a spike in FID). This validates that the carefully modulated temporal dynamics of our progressive guidance are absolutely critical for maintaining the structural integrity of the generation while simultaneously injecting identity-specific features.

Regarding the loss design, replacing the margin-based MM loss with a standard cross-entropy objective (w/ CE Loss (replace MM)) leads to a slight but consistent performance drop compared with the full setting. This observation highlights the superiority of MM for identity-level supervision and is broadly consistent with the findings reported in PLGMI [29]. Beyond validating our current design, it also suggests a promising direction for future work: developing more effective discriminative loss functions tailored to white-box model inversion.

To further examine higher-resolution behavior, we include a 224-resolution setting in the ablation tables. Scaling SFMI to a higher face-recognition input resolution requires a resolutionmatched FM prior and the same detected/aligned/cropped FR preprocessing pipeline; thus, the main practical constraints are the target model’s standardized input resolution and the computational cost of training and sampling the prior, rather than raw image size alone. Although the metrics moderately degrade at this higher resolution, the attack remains effective, indicating that SFMI is not restricted to low-resolution targets.

To provide an initial defense-oriented evaluation, we add the w/ Output Perturbation setting in the ablation tables. In this setting, we inject Gaussian noise with standard deviation 0.03 into the target model output before computing attack guidance, representing a lightweight post-processing defense that can be applied without retraining the face recognition model. As shown in the ablation results, SFMI remains effective under this perturbation. We attribute this to the observation that the perturbation-induced interference with the guidance direction is negligible compared with the guidance signal required for face recognition model utility; consequently, the attack still performs well.

To further evaluate SFMI against a training-phase defense, we adopt BiDO [57] and train defended ArcFace and Mobile-FaceNet models on the same private-data split before attacking them with the unchanged SFMI protocol. The corresponding defense results are reported in Tables V and VI. For ArcFace and MobileFaceNet, BiDO lowers attack ACC by 6.37% and 7.63%, respectively, at the cost of 6.86% and 4.45% reductions in target-model recognition accuracy. These results demonstrate that BiDO provides measurable protection but does not eliminate the inversion risk, revealing a clear trade-off between defense effectiveness and model utility.

## D. Detailed Parameter Analysis of Progressive Guidance

Having established that a meticulously designed progressive guidance schedule is paramount to the method’s overall usability, we now provide an in-depth parameter analysis of trajectory steering dynamics. We systematically evaluate seven variants of the guidance mechanism to dissect the functional necessity of both spatial direction alignment and the temporal warm-up, sustain, and annealing phases.

To intuitively illustrate the impact of each hyperparameter and structural choice, Fig. 4 presents a comprehensive visual and quantitative matrix on the CosFace target model. The first column shows the mathematical curve of the guidance function γ(t), and columns 2 through 8 visualize intermediate generation states from initial noise to a fully formed image. The final three columns report the reconstructed attack image, the corresponding real ground-truth face, and the ACC trajectory over guidance sampling steps, respectively.

A row-wise reading of Fig. 4 reveals several important behaviors. In Row 1, we remove PGS and apply a constant guidance coefficient of 1.0 without normalization; this seemingly simple guidance is too weak to effectively redirect the generation trajectory toward the target classifier. In Row 2, we again remove PGS and use a linear increase from 0.0 to 1.0 (also without normalization), but the guidance still fails to produce reliable identity steering. In Row 3, we use a linear decay from 1.0 to 0.0 without normalization and observe similarly weak control over facial trajectory evolution. In Row 4, we retain the PGS shape but remove normalization; although the schedule is more structured, the steering strength remains insufficient and reconstruction quality is still unsatisfactory. This weakness is also reflected by the last-column ACC curves, where both the blue target-model line and the yellow validation-model line remain relatively low or fluctuate unstably, indicating limited optimization effectiveness and poor transfer consistency.

Rows 5–7 further clarify how schedule design and loss design interact. In Row 5, based on standard PGS, we prolong the high-guidance stage and find that excessive late-stage forcing degrades performance on validation models, likely due to artifact accumulation in the refinement phase; correspondingly, the yellow curve drops in later steps and the blue-yellow gap widens, suggesting overfitting to short-term target gradients. In Row 6, standard PGS with CE loss already yields reasonably effective guidance and improved identity consistency, with both curves showing a clearer upward trend. In Row 7, standard PGS with our recommended MM loss achieves the best overall behavior: both target-model optimization and validation-model accuracy improve more stably, leading to the most favorable final reconstructions and the most consistent dual-curve progression.

The standard PGS configuration used in Eq. (11) is $M =$ 0.3, $t _ { 0 } = 0 . 1 , t _ { 1 } = 0 . 3 , t _ { 2 } = 0 . 7 .$ and $V _ { \mathrm { m a x } } ~ = ~ 1 . 0$ . The scheduler progressively introduces identity guidance and then attenuates it before final refinement, thereby balancing effective steering toward the target identity with stable evolution along the learned face manifold. The PGS hyperparameters do not need to be reselected for every target model: by normalizing the identity gradient and scaling it relative to the current FM velocity, SFMI reduces sensitivity to variations in target-model gradient magnitude and yields relatively stable hyperparameter choices across architectures. In practice, we retain the standard relative phase allocation defined by $t _ { 0 } , t _ { 1 } ,$ and $t _ { 2 } ,$ together with $V _ { \mathrm { m a x } } .$ , and primarily adjust the global scale M, which is typically kept at or below 0.35 to preserve manifold stability.

## V. ETHICAL CONSIDERATIONS

This work studies model inversion attacks to audit privacy leakage in face recognition systems under a white-box, strong-access threat model. All experiments are conducted on publicly available research datasets, including CelebA and FFHQ, which are used only for academic evaluation in accordance with their dataset terms and without redistributing the original data. The qualitative reconstructions shown in the paper are included to demonstrate potential privacy leakage risks rather than to identify, profile, or target any individual. We treat publication as bounded academic disclosure of this privacy risk, acknowledge the dual-use nature of identityreconstruction attacks, and frame SFMI as a privacy-auditing tool for researchers, developers, and system owners to motivate stronger privacy-preserving face recognition defenses, rather than as a practical guide for misuse.

## VI. CONCLUSION

In this paper, we introduced Steering Flow Model Inversion (SFMI), a method that reformulates white-box model inversion attacks as a deterministic trajectory-steering problem. By leveraging the continuous ODE formulation of Flow Matching,

![](images/0b41a5a44702ea744930cf5d6982c163d603309be80bf736358145d4820600b2.jpg)  
Fig. 4. Detailed row-wise analysis of progressive guidance variants on the CosFace target model. Rows 1–7 correspond to constant guidance (1.0), linear increase (0.0→1.0), linear decay (1.0→0.0), unnormalized PGS, prolonged-peak PGS, standard PGS+CE loss, and standard PGS+MM loss (ours). For each row, the first column shows the guidance curve γ(t), middle columns show intermediate trajectory states, and the final three columns report the reconstructed image, ground-truth face, and the ACC evolution curve over sampling steps. In the ACC curve, the blue line denotes ACC on the target model, and the yellow line denotes ACC on validation models.

SFMI helps mitigate the optimization instability in highly nonconvex GAN latent spaces and the stochastic disruption typical of standard diffusion models. Furthermore, we designed a progressive gradient guidance mechanism that dynamically modulates adversarial force across integration steps, ensuring precise temporal and spatial alignment with the generative flow. Extensive experiments demonstrate that SFMI delivers strong and competitive performance relative to state-of-theart methods, particularly in attack accuracy and visual fidelity, while maintaining competitive distribution-level fidelity. SFMI also exhibits strong attack capability against robust marginbased face recognition architectures such as CosFace and ArcFace and maintains cross-distribution generalization.

SFMI still has limitations. First, it involves relatively many hyperparameters and tuning can be difficult, and effective settings depend on a clear understanding of guidance dynamics. Second, SFMI is a white-box method that requires full access to the target model including gradients, which corresponds to a relatively high-access setting in practical scenarios. Third, SFMI depends on access to relevant public auxiliary face data, and its performance may degrade under larger domain shifts between the public and private data. Fourth, SFMI is not intended to attack directly on IoT devices; instead, the deployed target model must first be extracted from the device and then attacked using external computing resources. Fifth, cross-model ACC evaluation and LPIPS provide complementary evidence of identity recovery and perceptual similarity. Nevertheless, ACC remains a cross-model recognition proxy, and satisfying face recognition evaluators does not necessarily imply alignment with human identity perception. Overall,

SFMI provides a rigorous and effective tool for auditing privacy leakage in deep biometric systems. In future work, we plan to investigate privacy-preserving face recognition models, weaker-access and black-box model inversion scenarios, more robust attack methods that reduce reliance on many hyperparameters, stronger defensive paradigms against model inversion attacks, and model inversion methods that better align with human identity judgments.

## REFERENCES

[1] X. Wang, J. Peng, S. Zhang, B. Chen, Y. Wang, and Y. Guo, “A survey of face recognition,” 2022.

[2] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” Commun. ACM(CACM), vol. 63, no. 11, pp. 139–144, 2020.

[3] J. Gui, Z. Sun, Y. Wen, D. Tao, and J. Ye, “A review on generative adversarial networks: Algorithms, theory, and applications,” IEEE Trans. Knowl. Data Eng.(TKDE), vol. 35, no. 4, pp. 3313–3332, 2021.

[4] C.-H. Lai, Y. Song, D. Kim, Y. Mitsufuji, and S. Ermon, “The principles of diffusion models,” arXiv preprint arXiv:2510.21890, 2025.

[5] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proc. IEEE/CVF Int. Conf. Comput. Vis.(ICCV), 2023, pp. 4195–4205.

[6] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in Proc. Int. Conf. Mach. Learn.(ICML), ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 32 211–32 252.

[7] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in Proc. Int. Conf. Learn. Represent.(ICLR), 2023.

[8] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn.(CVPR), 2016, pp. 770–778.

[9] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. Int. Conf. Learn. Represent.(ICLR), 2021.

[10] M. Wang and W. Deng, “Deep face recognition: A survey,” Neurocomputing, vol. 429, pp. 215–244, 2021.

[11] H. Du, H. Shi, D. Zeng, X.-P. Zhang, and T. Mei, “The elements of end-to-end deep face recognition: A survey of recent advances,” 2021.

[12] M. Kim, A. Jain, and X. Liu, “50 years of automated face recognition,” 2025.

[13] F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet: A unified embedding for face recognition and clustering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn.(CVPR), 2015, pp. 815–823.

[14] H. Wang, Y. Wang, Z. Zhou, X. Ji, D. Gong, J. Zhou, Z. Li, and W. Liu, “Cosface: Large margin cosine loss for deep face recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn.(CVPR), 2018, pp. 5265– 5274.

[15] J. Deng, J. Guo, J. Yang, N. Xue, I. Kotsia, and S. Zafeiriou, “ArcFace: Additive angular margin loss for deep face recognition,” IEEE Trans. Pattern Anal. Mach. Intell.(TPAMI), vol. 44, no. 10, pp. 5962–5979, 2022.

[16] S. Chen, Y. Liu, X. Gao, and Z. Han, “Mobilefacenets: Efficient cnns for accurate real-time face verification on mobile devices,” in Proc. Chinese Conf. Biometric Recognit.(CCBR), 2018, pp. 428–438.

[17] H. Fang, Y. Qiu, H. Yu, W. Yu, J. Kong, B. Chong, B. Chen, X. Wang, S.-T. Xia, and K. Xu, “Privacy leakage on dnns: A survey of model inversion attacks and defenses,” arXiv preprint arXiv:2402.04013, 2024.

[18] W. Yang, S. Wang, D. Wu, T. Cai, Y. Zhu, S. Wei, Y. Zhang, X. Yang, Z. Tang, and Y. Li, “Deep learning model inversion attacks and defenses: a comprehensive survey,” Artif. Intell. Rev., vol. 58, no. 8, p. 242, 2025.

[19] M. Fredrikson, S. Jha, and T. Ristenpart, “Model inversion attacks that exploit confidence information and basic countermeasures,” in Proc. ACM SIGSAC Conf. Comput. Commun. Secur.(CCS), 2015, pp. 1322– 1333.

[20] M. Nasr, R. Shokri, and A. Houmansadr, “Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning,” in Proc. IEEE Symp. Secur. Privacy(SP), 2019, pp. 739–753.

[21] C. Song, T. Ristenpart, and V. Shmatikov, “Machine learning models that remember too much,” in Proc. ACM SIGSAC Conf. Comput. Commun. Secur.(CCS), 2017, pp. 587–601.

[22] Z. Yang, J. Zhang, E.-C. Chang, and Z. Liang, “Neural network inversion in adversarial setting via background knowledge alignment,” in Proc. ACM SIGSAC Conf. Comput. Commun. Secur.(CCS), 2019, pp. 225– 240.

[23] P. Hu, H. Ning, T. Qiu, H. Song, Y. Wang, and X. Yao, “Security and privacy preservation scheme of face identification and resolution framework using fog computing in internet of things,” IEEE Internet Things J.(IoT-J), vol. 4, no. 5, pp. 1143–1155, 2017.

[24] S. Yang, Y. Wen, L. He, M. Zhou, and A. Abusorrah, “Sparse individual low-rank component representation for face recognition in the iot-based system,” IEEE Internet Things J.(IoT-J), vol. 8, no. 24, pp. 17 320– 17 332, 2021.

[25] R. Yang, J. Ma, J. Zhang, S. Kumari, S. Kumar, and J. J. Rodrigues, “Practical feature inference attack in vertical federated learning during prediction in artificial internet of things,” IEEE Internet Things J.(IoT-J), vol. 11, no. 1, pp. 5–16, 2023.

[26] Y. Zhang, R. Jia, H. Pei, W. Wang, B. Li, and D. Song, “The secret revealer: Generative model-inversion attacks against deep neural networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn.(CVPR), 2020, pp. 253–261.

[27] S. Chen, M. Kahla, R. Jia, and G.-J. Qi, “Knowledge-enriched distributional model inversion attacks,” Proc. IEEE/CVF Int. Conf. Comput. Vis.(ICCV), pp. 16 178–16 187, 2021.

[28] L. Struppek, D. Hintersdorf, A. D. A. Correia, A. Adler, and K. Kersting, “Plug & play attacks: Towards robust and flexible model inversion attacks,” Proc. Int. Conf. Mach. Learn.(ICML), vol. 162, pp. 20 522– 20 545, 2022.

[29] X. Yuan, K. Chen, J. Zhang, W. Zhang, N. Yu, and Y. Zhang, “Pseudo label-guided model inversion attack via conditional generative adversarial network,” in Proc. AAAI Conf. Artif. Intell.(AAAI), vol. 37, no. 3, 2023, pp. 3349–3357.

[30] Y. Qiu, H. Fang, H. Yu, B. Chen, M. Qiu, and S.-T. Xia, “A closer look at gan priors: Exploiting intermediate features for enhanced model inversion attacks,” in Proc. Eur. Conf. Comput. Vis.(ECCV), 2024, pp. 109–126.

[31] O. Li, Y. Hao, Z. Wang, B. Zhu, S. Wang, Z. Zhang, and F. Feng, “Model inversion attacks through target-specific conditional diffusion models,” arXiv preprint arXiv:2407.11424, 2024.

[32] Y. Lu, S. Wang, G. Zhu, Z. Zhang, and J. Huang, “FGMIA: Featureguided model inversion attacks against face recognition models,” IEEE Trans. Inf. Forensics Secur.(TIFS), vol. 20, pp. 8465–8480, 2025.

[33] L. Zhou, Y. Zhu, and R. Liu, “Model inversion attack against federated unlearning,” IEEE Trans. Inf. Forensics Secur.(TIFS), 2026.

[34] Z. Shen, Z. Xia, K. Gan, P. Yu, and X. Zhou, “Swfti: Facial template inversion via styleswin mapping,” Pattern Recognit.(PR), p. 113190, 2026.

[35] X. Wang, H. Sun, and Y. He, “Gidd: Gradient inversion using diffusion model for denoising in federated learning,” Neurocomputing, p. 132682, 2026.

[36] K.-C. Wang, Y. Fu, K. Li, A. Khisti, R. Zemel, and A. Makhzani, “Variational model inversion attacks,” Adv. Neural Inf. Process. Syst.(NeurIPS), vol. 34, pp. 31 686–31 698, 2021.

[37] N.-B. Nguyen, K. Chandrasegaran, M. Abdollahzadeh, and N.-M. Cheung, “Re-thinking model inversion attacks against deep neural networks,” Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn.(CVPR), pp. 16 384–16 393, 2023.

[38] D. Usynin, D. Rueckert, and G. Kaissis, “Beyond gradients: Exploiting adversarial priors in model inversion attacks,” ACM Trans. Priv. Secur.(TOPS), vol. 26, no. 3, pp. 1–30, 2023.

[39] M. Khosravy, K. Nakamura, Y. Hirose, N. Nitta, and N. Babaguchi, “Model inversion attack by integration of deep generative models: Privacy-sensitive face generation from a face recognition system,” IEEE Trans. Inf. Forensics Secur.(TIFS), vol. 17, pp. 357–372, 2022.

[40] T. Wang, Y. Zhang, and R. Jia, “Improving robustness to model inversion attacks via mutual information regularization,” in Proc. AAAI Conf. Artif. Intell.(AAAI), vol. 35, no. 13, 2021, pp. 11 666–11 673.

[41] S.-T. Ho, K. J. Hao, K. Chandrasegaran, N.-B. Nguyen, and N.-M. Cheung, “Model inversion robustness: Can transfer learning help?” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn.(CVPR), 2024, pp. 12 183–12 193.

[42] N. Kandpal, E. Wallace, and C. Raffel, “Deduplicating training data mitigates privacy risks in language models,” in Proc. Int. Conf. Mach. Learn.(ICML), 2022, pp. 10 697–10 707.

[43] Z.-T. Liu and S.-T. Chen, “Trap-MID: Trapdoor-based defense against model inversion attacks,” in Adv. Neural Inf. Process. Syst.(NeurIPS), vol. 37, 2024, pp. 88 486–88 526.

[44] L. Struppek, D. Hintersdorf, and K. Kersting, “Be careful what you smooth for: Label smoothing can be a privacy shield but also a catalyst for model inversion attacks,” in Proc. Int. Conf. Learn. Represent.(ICLR), 2024.

[45] Z. Yang, B. Shao, B. Xuan, E.-C. Chang, and F. Zhang, “Defending model inversion and membership inference attacks via prediction purification,” arXiv preprint arXiv:2005.03915, 2020.

[46] J. Wen, S.-M. Yiu, and L. C. K. Hui, “Defending against model inversion attack by adversarial examples,” in Proc. IEEE Int. Conf. Cyber Secur. Resilience(CSR), 2021, pp. 551–556.

[47] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller, H. Saini,¨ Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in Proc. Int. Conf. Mach. Learn.(ICML), 2024.

[48] T. Li and K. He, “Back to basics: Let denoising generative models denoise,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn.(CVPR), 2026, pp. 36 115–36 125.

[49] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of diffusion-based generative models,” in Adv. Neural Inf. Process. Syst.(NeurIPS), 2022.

[50] Z. Liu, P. Luo, X. Wang, and X. Tang, “Deep learning face attributes in the wild,” in Proc. IEEE Int. Conf. Comput. Vis.(ICCV), 2015, pp. 3730–3738.

[51] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn.(CVPR), 2019, pp. 4401–4410.

[52] Y. Cheng, J. Zhao, Z. Wang, Y. Xu, J. Karlekar, S. Shen, and J. Feng, “Know you at one glance: A compact vector representation for low-shot learning,” in Proc. IEEE Int. Conf. Comput. Vis. Workshops(ICCVW), 2017, pp. 1924–1932.

[53] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Adv. Neural Inf. Process. Syst.(NeurIPS), vol. 30, 2017.

[54] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn.(CVPR), 2018, pp. 586–595.

[55] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, “Improved training of wasserstein gans,” in Adv. Neural Inf. Process. Syst.(NeurIPS), vol. 30, 2017, pp. 5767–5777.

[56] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, “Analyzing and improving the image quality of stylegan,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn.(CVPR), 2020, pp. 8110– 8119.

[57] X. Peng, F. Liu, J. Zhang, L. Lan, J. Ye, T. Liu, and B. Han, “Bilateral dependency optimization: Defending against model-inversion attacks,” in Proc. ACM SIGKDD Conf. Knowl. Discov. Data Min.(KDD), 2022, pp. 1358–1367.