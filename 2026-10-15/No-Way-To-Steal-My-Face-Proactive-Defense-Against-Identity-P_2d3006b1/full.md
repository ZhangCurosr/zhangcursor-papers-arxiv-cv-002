# No Way To Steal My Face: Proactive Defense Against Identity-Preserving Personalized Generation

Lizhi Xiong<sup>1</sup>, Jun Li<sup>1</sup>, Ziqiang Li<sup>1\*</sup>, Weiwei Jiang<sup>1</sup>, Zhangjie Fu<sup>1</sup>

<sup>1</sup>Nanjing University of Information Science and Technology

lzxiong16@163.com, lijuun@yeah.net, iceli@mail.ustc.edu.cn weiwei.jiang@nuist.edu.cn, fzj@nuist.edu.cn

## Abstract

Recent advances in diffusion models have enabled highfidelity, identity-preserving image generation for personalized applications such as digital avatars and virtual try-on systems. However, their reliance on sensitive facial reference images raises growing privacy concerns. Existing defense mechanisms are primarily designed for trainingbased personalization and struggle to generalize to emerging training-free approaches, due to fundamental differences in their identity integration paradigms. To bridge this gap, we propose IDGuardian—the first generalizable and model-agnostic identity protection framework capable of defending against both training-based and training-free personalization methods. IDGuardian abstracts the personalization process into two critical stages: identity extraction and identity injection. It then introduces crafted adversarial perturbations to simultaneously disrupt both stages. Specifically, it degrades the identityfeatures extracted by external encoders and establishes an adversarial conceptual bridge that misdirects the generative trajectory away from the target identity. Extensive experiments show that IDGuardian effectively protects identity across various personalization pipelines and model architectures, while remaining robust to post-processing, adaptive attacks, and cross-dataset generalization. Our code is available at https://github. com/OpenAscent-L/IDGuardian.git.

## 1. Introduction

Recent advances in diffusion models have significantly improved high-fidelity, identity-preserving image generation [17, 29]. By conditioning on facial reference images, these models can produce photorealistic portraits that accurately retain individual identities [15, 30, 42]. Existing identitypreserving personalization methods generally fall into two categories: training-based and training-free approaches.

![](images/46cab64ddb4e9f58f77922effc37cad3f7271922e092a61d889db6a91030ca23.jpg)  
Figure 1. Existing defense methods become ineffective in trainingfree personalization due to paradigm discrepancies and fusion diversity.

Training-based methods [11, 14, 31] fine-tune pre-trained diffusion models on a small set of user-specific images, effectively capturing detailed identity features. However, they require significant computational resources, careful hyperparameter tuning, and GPU support, which limits their scalability in practice. In contrast, training-free methods [16, 47, 48] leverage identity encoders pre-trained on largescale datasets to extract semantic facial representations in a zero-shot manner during inference. These methods enable fast, scalable, and user-friendly personalization, making them popular in real-world applications due to their efficiency and low deployment cost.

Despite the impressive progress in personalized image generation enabled by diffusion models, growing privacy concerns have arisen due to their reliance on sensitive facial reference images, which can be misused for identity reconstruction or impersonation. To address these risks, recent proactive defenses have explored perturbing input images to disrupt the generation of personalized content [45]. These methods [12, 33, 41, 43] typically adopt trainingbased personalization as a proxy task, optimizing adversarial perturbations to hinder the model’s ability to encode identity-specific features. However, their effectiveness degrades markedly when applied to training-free personalization, as illustrated in Figure 1, due to two core challenges arising from differences in personalization paradigms: i) Personalized Pipeline Discrepancy. Training-based approaches internalize identity features through model finetuning, whereas training-free methods inject identity embeddings via pre-trained encoders without modifying model parameters. This structural distinction renders perturbations optimized under the training-based setting ineffective in training-free scenarios. ii) Information Fusion Diversity. Training-free pipelines exhibit considerable heterogeneity in encoder architectures and identity-semantics fusion strategies, lacking a unified framework. This diversity poses a significant obstacle to developing generalizable defenses, leaving critical privacy risks unresolved.

To bridge this gap, we propose IDGuardian, a generalizable and model-agnostic defense mechanism designed to protect identity privacy across both training-based and training-free personalization paradigms.<sup>1</sup> Despite the structural differences among existing personalization pipelines, IDGuardian mitigates identity leakage by abstracting the generation process into two fundamental stages: identity extraction and identity injection. It then formulates a dualperspective adversarial strategy that disrupts identity learning through the following mechanisms: i) Cross-Encoder Identity Field Confounding. IDGuardian enforces intentional dissimilarity between identity features extracted from the protected and original images, hindering malicious exploitation of the reference identity in downstream generation. ii) Guidance-Flow Identity Deflection. By interpreting identity-preserving personalization as a conceptual bridge from an unconditional to an identity-conditioned distribution, IDGuardian establishes an adversarial conceptual bridge that redirects the diffusion sampling trajectory away from the clean identity manifold, thereby impairing identity consistency during synthesis.

To evaluate the effectiveness and robustness of ID-Guardian, we conduct extensive experiments across diverse personalization pipelines, including training-based and training-free methods, as well as diffusion- and fluxbased generators, and observe strong transfer from a single white-box training setup to unseen black-box systems. The protection remains effective under post-processing, adaptive attacks, and cross-dataset transfer. Ablation studies verify the contribution of each component, while both human and perceptual evaluations confirm reliable identity protection without compromising usability. Our contributions can be summarized into three-fold:

• We introduce the first proactive identity protection framework that defends against both training-free and trainingbased personalization methods, effectively mitigating malicious identity misuse. Our approach fills a critical gap in existing defenses by demonstrating dual adaptability across personalization paradigms.

• We propose IDGuardian, which abstracts identitypreserving generation into identity extraction and injection. Through Cross-Encoder Identity Field Confounding and Guidance-Flow Identity Deflection, it disrupts feature alignment and redirects diffusion trajectories, mitigating identity preservation across personalization pipelines.

• Extensive experiments show that IDGuardian effectively protects identity across diverse personalization methods, including generators based on diffusion and flux. Trained on a single white-box diffusion model, it generalizes to unseen black-box pipelines and datasets, and remains robust against post-processing and adaptive attacks.

## 2. Related Work

## 2.1. Identity-Preserving Personalized Generation

Identity-preserving personalized generation aims to synthesize images that faithfully retain the unique identity of a given subject [3, 34]. Recent advances in diffusion-based models have significantly enhanced the fidelity and identity consistency of generated images [7, 27]. Existing methods can be broadly categorized into two classes: trainingbased and training-free. Training-based methods—such as DreamBooth [31], CustomDiffusion [14], Textual Inversion [5], and LoRA-based fine-tuning [11]—adapt pre-trained diffusion models to new identities by fine-tuning them on a small set of reference images. These approaches often yield high-quality personalization and effectively preserve detailed identity cues, but they are computationally intensive and sensitive to hyperparameter settings, limiting their scalability and usability in practical scenarios.

To address these challenges, training-free approaches have emerged as lightweight and efficient alternatives [4, 6]. These methods typically use external identity encoders to extract semantic facial representations zero-shot, injecting them into the diffusion process via various fusion strategies without modifying model parameters [51]. The key difference among recent training-free methods lies in how identity is integrated into generation [17]. For example, PhotoMaker [16] uses a Direct Stack strategy to encode multiple references into a unified identity embedding replacing class-specific tokens. IP-Adapter [48] and InstantID [46] employ Cross-Attention to inject identity features alongside text-based attention. Infinite-ID [47] extends this with a Mixed Attention scheme combining cross- and selfattention. In contrast, KV-Edit [52] relies solely on Self-Attention, caching background key-value pairs and merging them with foreground queries for identity-aware editing.

## 2.2. Proactive Defense for Personalized Generation

With the rapid proliferation of personalized generative models, concerns about privacy leakage and the unauthorized use of personal images have spurred growing interest in developing effective defense mechanisms [18, 20, 38, 49]. Among these, active defense strategies aim to proactively perturb user-provided reference images to prevent downstream models from accurately capturing and reproducing identity features. Recent advances have introduced a variety of active defense techniques designed to disrupt the fine-tuning processes of generative models [1, 8, 24]. For instance, Anti-DreamBooth [43] applies PGD-based perturbations [24] to interfere with the fine-tuning pipeline of models such as DreamBooth. AdvDM [19] leverages Monte Carlo estimation to optimize perturbations in the latent space of diffusion models, preventing the extraction of features from protected artworks. SimAC [44] enhances perturbation effectiveness through greedy timestep selection and a feature-level interference loss tailored to diffusion dynamics. In contrast, PersGuard [22] implants backdoor triggers into pre-trained models to selectively disable personalization for protected inputs. Additionally, ACE [50] proposes an unified adversarial framework that jointly optimizes encoder error, reverse process error, and finetuning error, employing target-guided optimization to balance conflicts among these objectives.

However, the effectiveness of existing defenses significantly diminishes in training-free settings, primarily due to Personalized Pipeline Discrepancy and Information Fusion Diversity inherent in personalization methods. As trainingfree pipelines gain traction for their scalability and ease of deployment, there is an urgent need for universal defense strategies that can robustly counteract both training-based and training-free personalization approaches. In contrast to training-based pipelines, effective defense against trainingfree personalization remains insufficiently explored, particularly in ways that can generalize across both paradigms. Concurrent with our work, IDProtector [39] addresses identity leakage by aligning features across identity encoders used in various personalization pipelines. Although effective in training-free scenarios, it relies on multiple encoderspecific features tied to specific models, limiting its generalization to training-based pipelines and introducing considerable architectural complexity. These limitations underscore the need for more efficient, generalizable, and modelagnostic defense mechanisms.

## 3. Methods

## 3.1. Threat Model

We consider a real-world identity abuse scenario involving three roles: the identity holder (victim), the generative user (attacker), and the protection mechanism (defender)

deployed by the identity holder.

Identity Holder (Victim). The identity holder shares facial images on public platforms. These images may be harvested and exploited to synthesize novel identity-consistent content without consent. To mitigate such privacy risks, the identity holder proactively applies subtle, imperceptible perturbations that preserve visual realism while reducing extractability of identity-relevant features.

Generative User (Attacker). The attacker uses personalized generation techniques, often diffusion-based, to synthesize images that preserve the victim’s identity, enabling scalable and high-fidelity reproduction.

Protection Mechanism (Defender). To prevent unauthorized identity preservation, the defender perturbs the input image. Operating in a restricted access setting, the defender has no knowledge of the specific personalized generation techniques employed by the attacker and is limited to modifying the input image alone. Given any personalized identity-preserving generator $G ( \cdot )$ and input image x, the defender seeks to optimize a perturbation δ such that $x + \delta$ reduces identity consistency in the generated output. The objective is to ensure that $G ( x + \delta )$ deviates in identity from x, thereby diminishing the effectiveness of personalized generation in preserving the victim’s identity. The optimization problem is formulated as:

$$
\operatorname* { m i n } _ { \delta } \mathrm { S I M } \big ( \mathrm { I D } ( \boldsymbol { x } ) , \mathrm { I D } ( G ( \boldsymbol { x } + \delta ) ) \big ) \quad \mathrm { s . t . } \quad \| \delta \| _ { \infty } < \epsilon ,
$$

where $\operatorname { I D } ( \cdot )$ denotes the identity feature extractor, $\operatorname { S I M } ( \cdot , \cdot )$ quantifies the similarity between identities, and ϵ constrains the magnitude of the perturbation to ensure imperceptibility.

## 3.2. Overall Method

Identity-preserving personalized generation methods typically consist of two critical stages: identity extraction and identity injection. In the first stage, an identity extractor derives identity features from a reference image. In the second, these features are fused with semantic content via identity injection techniques to produce personalized outputs. As depicted in the upper part of Figure 2, this pipeline can be abstracted into a two-stage process. Leveraging this structure, we propose IDGuardian, a defense framework that protects a reference identity by independently disrupting both the identity extraction and injection stages.

• Cross-Encoder Identity Field Confoundin: ID-Guardian enforces the identity features extracted from the protected image to be intentionally dissimilar from those of the original image, thereby hindering the malicious use of the reference identity in subsequent generative processes.

• Guidance-Flow Identity Deflection: IDGuardian establishes an adversarial conceptual bridge by interpreting identity-preserving personalization as a conceptual bridge from an unconditional to a target identity-conditioned distribution. This redirects the diffusion sampling trajectory away from the clean identity manifold, impairing identity consistency during synthesis.

![](images/1234657bb57ab1a068768fb7ee786e20d6715413253ebd52b4227297a56efc56.jpg)  
Figure 2. Overview of IDGuardian. The framework centers on two complementary stages of personalization: identity extraction and identity injection. To disrupt identity extraction, Cross-Encoder Identity Field Confounding decorrelates visual identity embeddings across feature spaces. To weaken identity injection, Guidance-Flow Identity Deflection redirects the denoising trajectory toward identity neutral regions. Identity loss is computed using embeddings from the original $\mathrm { ( I _ { F a c e N e t / C L I P } ) }$ and protected $( \mathbf { P } _ { \mathrm { F a c e N e t / C L I P } } )$ images, while distributional deviation is induced via adversarial conceptual bridge during the denoising process.

By jointly optimizing perturbations at both stages, ID-Guardian produces a final adversarial perturbation δ that remains imperceptible to human observers while substantially degrading identity consistency in the generated outputs.

## 3.3. Cross-Encoder Identity Field Confounding

To disrupt identity extraction, we explicitly minimize the identity similarity between the clean image x and its adversarially perturbed counterpart x + δ within the embedding spaces of two complementary identity encoders. Employing dual encoders enables the capture of richer and more comprehensive identity representations, mitigating the limitations of using a single encoder, which may struggle to generalize across diverse image domains and identity variations. Specifically, we adopt FaceNet [32] and CLIP [28] as complementary identity extractors to enhance robustness of the identity loss. FaceNet is tailored for extracting facial identity features, while CLIP’s image encoder captures high-level semantic information from visual data, providing a broader and more nuanced representation of identity.

The identity loss is defined as a weighted sum of cosine similarities computed in the respective embedding spaces:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { I D } } = \mathcal { L } _ { \mathrm { F a c e N e t } } + \mathcal { L } _ { \mathrm { C L I P } } , } \end{array}\tag{1}
$$

where each term measures the cosine similarity between

features extracted from the clean and perturbed images:

$$
{ \mathcal { L } } _ { \mathrm { F a c e N e t } } = \cos { \big ( } f _ { \mathrm { F a c e N e t } } ( x ) , f _ { \mathrm { F a c e N e t } } ( x + \delta ) { \big ) } ,\tag{2}
$$

$$
\mathcal { L } _ { \mathrm { C L I P } } = \cos \big ( f _ { \mathrm { C L I P } } ( x ) , f _ { \mathrm { C L I P } } ( x + \delta ) \big ) ,\tag{3}
$$

where $f _ { \mathrm { F a c e N e t } } ( \cdot )$ and $f _ { \mathrm { C L I P } } ( \cdot )$ denote the feature extraction functions that map input images to their respective identity embedding spaces.

## 3.4. Guidance-Flow Identity Deflection

To weaken identity injection during diffusion generation, we interpret identity-preserving personalization as a conceptual bridge represented by a structured shift in the score function toward the target identity manifold. Building on this, we establish an adversarial conceptual bridge that diverts the generative trajectory from the original manifold, disrupting identity preservation. The subsequent sections detail score-based foundations, the conceptual bridge perspective, and our adversarial conceptual bridge design.

Score-Based Foundations in Diffusion. Diffusion models synthesize data by iteratively denoising a Gaussian noise sample through a learned reverse Markov process. During each timestep t, the model learns to approximate the gradient of the data log-probability—referred to as the score function, denoted by $\nabla _ { \mathbf { x } _ { t } } \log p _ { \theta } ( \mathbf { x } _ { t } )$ —which steers the noisy input $\mathbf { x } _ { t }$ toward regions of higher data likelihood. As the exact score function is intractable to compute, diffusion models instead learn to predict the noise component $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t )$ introduced during the forward process. This leads to the following widely used approximation [10, 21, 25, 35, 37]:

$$
\nabla _ { \mathbf { x } _ { t } } \log p _ { \boldsymbol \theta } ( \mathbf { x } _ { t } ) \approx - \frac { 1 } { \sqrt { 1 - \bar { \alpha } _ { t } } } \cdot \epsilon _ { \boldsymbol \theta } ( \mathbf { x } _ { t } , t ) ,\tag{4}
$$

where $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$ denotes the cumulative noise schedule up to timestep t. In conditional generation settings, such as identity- or text-conditioned synthesis, the score function incorporates the conditioning variable y, yielding the approximation:

$$
\nabla _ { \mathbf { x } _ { t } } \log p _ { \theta } ( \mathbf { x } _ { t } \mid \mathbf { y } ) \approx - \frac { 1 } { \sqrt { 1 - \bar { \alpha } _ { t } } } \cdot \mathbf { \varepsilon } _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { y } ) ,\tag{5}
$$

where $\epsilon _ { \theta } ( \mathbf x _ { t } , t , \mathbf y )$ denotes the predicted noise under the conditioning of y.

Analyzing Identity-Preserving Personalization through the Lens of the Score Function as a Conceptual Bridge. Inspired by distributional transport frameworks such as Dual Diffusion Implicit Bridges [40], we view identitypreserving personalization through the lens of a conceptual bridge. Grounded in the score-based foundation of diffusion models, this bridge can be interpreted as linking an unconditional to a conditional generative process, guiding generation toward the target identity $y .$ This behavior can be modeled by maximizing the conditional likelihood log $p ( y \mid \mathbf { x } _ { t } )$ with its gradient reflecting how the identity y influences the generative trajectory. Applying Bayes’ rule, the gradient can be decomposed as:

$$
\nabla _ { \mathbf { x } _ { t } } \log p ( y \mid \mathbf { x } _ { t } ) = \nabla _ { \mathbf { x } _ { t } } \log p ( \mathbf { x } _ { t } \mid y ) - \nabla _ { \mathbf { x } _ { t } } \log p ( \mathbf { x } _ { t } ) ,\tag{6}
$$

where the first term corresponds to the conditional likelihood, and the second term reflects the unconditional prior.

Following prior study [36], this gradient difference $\nabla _ { \mathbf { x } _ { t } } \log p ( y \mid \mathbf { x } _ { t } )$ can be approximated using the predicted noise of the model as:

$$
\nabla _ { \mathbf { x } _ { t } } \log p ( y \mid \mathbf { x } _ { t } ) \approx - \frac { 1 } { \sqrt { 1 - { \bar { \alpha } } _ { t } } } \left( \epsilon _ { \theta } ( \mathbf { x } _ { t } , t , y ) - \epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \emptyset ) \right) ,\tag{7}
$$

where $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , y )$ represents predicted noise conditioned on the target identity, corresponding to the blue arrow $\epsilon _ { \mathrm { c l e a n i d } }$ in Figure 3. In contrast, $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \emptyset )$ denotes the unconditional noise prediction, illustrated by the orange arrow $\epsilon _ { \mathrm { s r c } } .$ . The difference between these two predictions (indicated by the light cyan dashed arrow) captures how identity conditioning alters the denoising direction, effectively steering the generative trajectory toward the target identity manifold [9, 25].

This identity-induced distributional shift can be interpreted as a structured perturbation of the score function. Inspired by the analogy to distributional transport, we refer to this shift as a conceptual bridge, highlighting its role in connecting the source and identity-conditioned distributions in the generative process.

![](images/a41c6a2b018200c095fa65bb0b902f3a1389166dcf1a1d88555aa4cf3770c441.jpg)  
Figure 3. Visualization of Conceptual Bridges Constructed via Noise Prediction Differences in Diffusion Sampling.

Score-Based Adversarial Conceptual Bridge. Building on the conceptual bridge perspective of identity-preserving personalization, we introduce a parallel adversarial conceptual bridge that reorients the generative trajectory away from the clean identity manifold and toward a strategically designed adversarial identity distribution. Unlike its identity-preserving counterpart, this bridge deliberately disrupts identity consistency rather than reinforcing alignment.

Formally, this adversarial conceptual bridge is constructed through the difference between the conditional gradients corresponding to adversarial and clean identity:

$$
S ^ { * } = \nabla _ { \mathbf { x } _ { t } } \log p ( y _ { \mathrm { a d v } } \mid \mathbf { x } _ { t } ) - \nabla _ { \mathbf { x } _ { t } } \log p ( y _ { \mathrm { c l e a n } } \mid \mathbf { x } _ { t } ) ,\tag{8}
$$

where $\nabla _ { \mathbf { x } _ { t } } \log p ( y _ { \mathrm { a d v } } \ \mid \ \mathbf { x } _ { t } )$ can be approximated by the difference $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , y _ { \mathrm { a d v } } ) - \epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \emptyset )$ , shown as the purple dashed arrow in Figure 3, with $y _ { \mathrm { a d v } }$ extracted from the protected image, steering generation toward the adversarial identity manifold. Concretely,

$$
\begin{array} { l } { S ^ { * } \approx - \displaystyle \frac { 1 } { \sqrt { 1 - \bar { \alpha } _ { t } } } \cdot \left( \epsilon _ { \theta } ( { \bf x } _ { t } , t , y _ { \mathrm { a d v } } ) - \epsilon _ { \theta } ( { \bf x } _ { t } , t , \emptyset ) \right) } \\ { \displaystyle \quad + \frac { 1 } { \sqrt { 1 - \bar { \alpha } _ { t } } } \cdot \left( \epsilon _ { \theta } ( { \bf x } _ { t } , t , y _ { \mathrm { c l e a n } } ) - \epsilon _ { \theta } ( { \bf x } _ { t } , t , \emptyset ) \right) } \\ { \displaystyle = - \frac { 1 } { \sqrt { 1 - \bar { \alpha } _ { t } } } \cdot \left( \epsilon _ { \theta } ( { \bf x } _ { t } , t , y _ { \mathrm { a d v } } ) - \epsilon _ { \theta } ( { \bf x } _ { t } , t , y _ { \mathrm { c l e a n } } ) \right) , } \end{array}\tag{9}
$$

where $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , y _ { \mathrm { a d v } } )$ and $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , y _ { \mathrm { c l e a n } } )$ denote predicted noise under adversarial and clean identity conditions, corresponding to the red and blue arrows $\epsilon _ { \mathrm { a d v i d } }$ and ϵ<sub>clean id</sub> in Figure 3. This adversarial conceptual bridge, shown as the green dashed arrow, suppresses identity preservation by steering the sample path to lower-likelihood regions under y<sub>clean</sub>. It integrates into the model’s scoring mechanism, ensuring temporal consistency and underpinning our defense.

## 3.5. Perturbation Optimization

We update the adversarial perturbation by integrating gradients derived from both the identity loss $\nabla _ { \delta } \mathcal { L } _ { \mathrm { I D } }$ and the score-based adversarial conceptual bridge $S ^ { * }$ . To ensure dimensional consistency, S<sup>∗</sup> (originally computed in the latent score space) is first upsampled and channel-aligned to match the pixel domain of $\delta ,$ resulting in $S _ { \mathrm { u p } } ^ { * } .$ . Both gradients are then $\ell _ { 2 }$ -normalized to balance their relative contributions. The combined gradient is computed as:

$$
\mathrm { \ t o t a l \mathrm { - } g r a d } \gets \frac { \nabla _ { \delta } \mathcal { L } _ { \mathrm { I D } } } { \| \nabla _ { \delta } \mathcal { L } _ { \mathrm { I D } } \| _ { 2 } } - \frac { S _ { \mathrm { u p } } ^ { * } } { \| S _ { \mathrm { u p } } ^ { * } \| _ { 2 } } .\tag{10}
$$

The perturbation is updated via a projected gradient step:

$$
\delta  \delta - \alpha \cdot \mathrm { s i g n ( t o t a l \mathrm { { \_ g r a d } } ) } ,\tag{11}
$$

where α controls the update magnitude. For details of the optimization process, please refer to Appendix A.1.

## 4. Experiments

## 4.1. Implementation Details and Metrics

Datasets. We evaluate our method on two widely used face datasets: VGGFace2 [26] and CelebA-HQ [23]. Unless specified, primary experiments are conducted on VG-GFace2. We exclude images smaller than 250 × 250 pixels, perform alignment and cropping for consistent facial positioning, and select 50 identities, resizing images to 224 × 224. To assess cross-dataset generalization, we further evaluate on CelebA-HQ by applying adversarial perturbations and identity-conditioned generation to 50 highresolution images (512 × 512).

Training Details. We adopt IP-Adapter SDXL Plus Face [48] as the surrogate model, built on the pretrained SDXL backbone [27] with a CLIP-based image encoder to extract identity embeddings for conditioning the UNet. Unless specified, all primary experiments use this SDXL-based surrogate. We apply Projected Gradient Descent (PGD) with a fixed text prompt $y \colon$ “A photo of a person”, perturbation budget $\epsilon = 8 / 2 5 5 .$ , learning rate $\alpha ~ = ~ 0 . 0 0 5$ , and maximum steps $N _ { \mathrm { i t e r } } = 2 0 0$ . Experiments are conducted on an NVIDIA RTX A6000 GPU (48G). To evaluate surrogatelevel generalization, we also test a variant based on Stable Diffusion 1.5 (SD 1.5).

Evaluation Metrics. We adopt three categories of evaluation metrics to comprehensively assess identity protection performance: i) Identity Similarity Metrics (ISM): ISM is the average similarity between generated and source images computed from ArcFace, FaceNet, and VGG-Face features, indicating identity removal effectiveness. ii) Visual Quality: PSNR and SSIM assess the pixel-level fidelity of protected images, reflecting the visual quality after perturbation. iii) Recognition Utility: FaceQNet and MagFace assess the utility of identity-personalized face images for face recognition models. Details specific to DreamBooth evaluation are provided in Appendix A.2.

Reproducibility Resources. For reproducibility, more details on the public resources used in our implementation and evaluation are provided in Appendix A.3.

## 4.2. Overall Performance

IDGuardian Enhances Identity Protection Effectiveness. We evaluate IDGuardian against five representative active defense methods across seven mainstream identitypersonalization pipelines. These pipelines encompass both training-free methods (IP-Adapter [48], IP-Adapter SDXL Plus Face [48], Blip-Diffusion [15], InfiniteID [47], and PhotoMaker [16]) and training-based methods (Dream-Booth [31] and InfiniteYou [13]), leveraging diverse generative backbones with Stable Diffusion 1.5, SDXL, and FLUX.1-dev (Transformer-based) for Infinite You.

As summarized in Table 1, IDGuardian consistently achieves the lowest average identity similarity scores across ArcFace, FaceNet, and VGGFace metrics, alongside substantial reductions in image quality scores measured by FaceQNet. Detailed per-metric results and comprehensive statistical analysis are provided in Appendix B.1. Qualitative examples in Figure 4 further confirm that IDGuardian effectively suppresses identity cues across a wide range of personalization scenarios, outperforming competing defense methods. These findings highlight the generalizability of IDGuardian in protecting identity across diverse generative models and personalization pipelines.

![](images/b74ce8ff51892795583ba3c52d7fbc22739a40accceea399ee0d35ef020a14d6.jpg)  
Figure 4. Visualization of identity suppression effectiveness for IDGuardian and comparison methods across different personalization settings.

IDGuardian Preserves Perceptual Quality While Suppressing Identity. Table 2 reports PSNR, SSIM, and time cost scores evaluating visual fidelity and computational efficiency of IDGuardian-processed images. Results show that, compared with other defense methods, IDGuardian more effectively suppresses identity cues while better preserving perceptual quality and maintaining competitive time efficiency. The low time cost of IDGuardian arises from its training-free inference design. By contrast, IDP requires training a noise generator that can generalize to most images, incurring substantial computational resources. This is reflected in IDP-P, where performing PGD optimization for each image clearly results in much higher time cost.

Table 1. Comparison of identity protection against baseline methods. ISM is the average identity similarity (ArcFace, FaceNet, VGGFace); FQ is the FaceQNet quality score. “No Protected” denotes generation w/o defense. Best results are in bold.
<table><tr><td rowspan="2">Method</td><td colspan="2">Ip-Adapter</td><td colspan="2">Ip-Ada + XL</td><td colspan="2">Blip-Diffu</td><td colspan="2">InfiniteID</td><td colspan="2">Photomaker</td><td colspan="2">Dreambooth</td><td colspan="2">InfiniteYou</td></tr><tr><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td></tr><tr><td>No Protected</td><td>0.382</td><td>0.334</td><td>0.582</td><td>0.456</td><td>0.336</td><td>0.331</td><td>0.675</td><td>0.495</td><td>0.327</td><td>0.456</td><td>0.525</td><td>0.332</td><td>0.673</td><td>0.493</td></tr><tr><td>Anti-DB [43]</td><td>0.307</td><td>0.305</td><td>0.548</td><td>0.455</td><td>0.271</td><td>0.357</td><td>0.620</td><td>0.496</td><td>0.272</td><td>0.288</td><td>0.433</td><td>0.319</td><td>0.594</td><td>0.467</td></tr><tr><td>SimAC [44]</td><td>0.295</td><td>0.312</td><td>0.401</td><td>0.451</td><td>0.305</td><td>0.363</td><td>0.591</td><td>0.484</td><td>0.248</td><td>0.296</td><td>0.390</td><td>0.343</td><td>0.571</td><td>0.459</td></tr><tr><td>AdvDM [19]</td><td>0.286</td><td>0.349</td><td>0.398</td><td>0.438</td><td>0.260</td><td>0.355</td><td>0.542</td><td>0.490</td><td>0.227</td><td>0.293</td><td>0.337</td><td>0.192</td><td>0.563</td><td>0.459</td></tr><tr><td>ACE [50]</td><td>0.327</td><td>0.363</td><td>0.536</td><td>0.453</td><td>0.275</td><td>0.363</td><td>0.576</td><td>0.499</td><td>0.252</td><td>0.287</td><td>0.420</td><td>0.215</td><td>0.635</td><td>0.461</td></tr><tr><td>IDProtector [39]</td><td>0.291</td><td>0.337</td><td>0.425</td><td>0.432</td><td>0.263</td><td>0.368</td><td>0.603</td><td>0.489</td><td>0.240</td><td>0.312</td><td>0.402</td><td>0.225</td><td>0.541</td><td>0.454</td></tr><tr><td>IDGuar (Ours)</td><td>0.036</td><td>0.282</td><td>0.126</td><td>0.450</td><td>0.220</td><td>0.351</td><td>0.444</td><td>0.494</td><td>0.142</td><td>0.273</td><td>0.336</td><td>0.220</td><td>0.301</td><td>0.452</td></tr></table>

Table 2. Comparison of protected image quality (PSNR (dB), SSIM) and time cost (s) across different methods. At-DB denotes Anti-DreamBooth, SAC denotes SimAC, ADM denotes AdvDM, ACE denotes ACE, IDP-P denotes the PGD-optimized variant of IDProtector, IDP denotes IDProtector, and IDGuar denotes our proposed IDGuardian.
<table><tr><td>At-DB SAC ADM ACE IDP-P IDP IDGuar</td></tr><tr><td>PSNR 31.63 32.15 30.7631.61 32.49 32.10 32.19</td></tr><tr><td>SSIM 0.701 0.772 0.735 0.753 0.840 0.805 0.842</td></tr></table>

IDGuardian Is Robust To Post-Processing and Adaptive Threats. We assess the robustness of IDGuardian against common image degradations—including JPEG compression, Gaussian blur, and Gaussian noise—as well as against adaptive threats such as Impress [2], a recent method designed to bypass active defenses. As shown in Table 3, ID-Guardian consistently maintains effective identity suppression under these challenging conditions. Interestingly, for some baselines, mild post-processing such as JPEG compression may slightly improve defense performance. While compression generally disrupts adversarial perturbations, in some cases it also perturbs the model’s recognition of the original image’s identity features, which can inadvertently make identity suppression appear stronger. Qualitative examples in Figure 8 of the Appendix further support these results. Please see Appendix B.3 and B.4 for details.

IDGuardian Generalizes to Commercial APIs. We eval uate IDGuardian on commercial face personalization APIs (Seedream 4.0, Qwen-Image, Kling-AI, Ihuiwa). As shown in Appendix D.1, it consistently suppresses identity similarity, demonstrating robustness under black-box conditions.

IDGuardian Generalizes Across Surrogate Models. To assess the generalization of IDGuardian, we replace the default SDXL-based IP-Adapter with a variant based on Stable Diffusion 1.5. As shown in Table 2 and Figure 6 of Appendix, IDGuardian consistently achieves strong identity suppression across both generative backbones, demonstrating robustness to architectural variations in the surrogate model. Further details are provided in Appendix D.2.

IDGuardian Generalizes Across Datasets. Beyond VG-GFace2, we further evaluate IDGuardian on the CelebA-HQ dataset to assess generalization across data distributions. As shown in Table 9, Table 10, and Figure 10 of Appendix, ID-Guardian consistently suppresses identity similarity across various metrics. Consistent results on the LFW dataset using the identity similarity metric are further reported in Table 11 of the Appendix. This performance holds true across different image resolutions and demographics, reinforcing the robustness of our method in diverse real-world scenarios. For details, please refer to Appendix D.3.

Human and GPT-Based Evaluation of Identity Protection Efficacy. To further validate the effectiveness of identity protection, we conduct both a user study and a GPTbased evaluation using a large multimodal model to assess perceived identity similarity. Results from both evaluations consistently indicate that IDGuardian significantly reduces identity similarity compared to baseline methods. For details on the evaluation protocols and results, please refer to Appendix E.1, E.2 and Figure 7 of Appendix.

## 4.3. Ablation Study

Roles of Individual Components. We conduct ablations on three identity supervision components: i) CLIP identity loss alone, ii) combined CLIP and FaceNet losses, and iii) CLIP + FaceNet losses with additional Score-Based Adversarial Conceptual Bridge. As shown in Table 4, the dualencoder setup improves identity suppression by leveraging both semantic and low-level features. Incorporating adversarial conceptual bridge further enhances suppression effectiveness significantly, demonstrating the key role of guided optimization. More details are provided in Appendix B.2.

Table 3. Robustness evaluation against common image post-processing operations, including JPEG compression, Gaussian blur, and Gaussian noise, and attacks targeting active defenses such as Impress.
<table><tr><td rowspan="2">Method</td><td colspan="2">Ip-Adapter</td><td colspan="2"> $\mathbf { \overline { { I p - A d a + X L } } }$ </td><td colspan="2">Blip-Diffu</td><td colspan="2">InfiniteID</td><td colspan="2">Photomaker</td><td colspan="2">Dreambooth</td><td colspan="2">InfiniteYou</td></tr><tr><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td></tr><tr><td>No Protected</td><td>0.382</td><td>0.334</td><td>0.582</td><td>0.456</td><td>0.336</td><td>0.331</td><td>0.675</td><td>0.495</td><td>0.327</td><td>0.456</td><td>0.525</td><td>0.332</td><td>0.673</td><td>0.493</td></tr><tr><td>JPEG</td><td>0.141</td><td>0.221</td><td>0.340</td><td>0.454</td><td>0.215</td><td>0.323</td><td>0.384</td><td>0.492</td><td>0.135</td><td>0.274</td><td>0.331</td><td>0.254</td><td>0.322</td><td>0.454</td></tr><tr><td>Blur</td><td>0.220</td><td>0.321</td><td>0.376</td><td>0.467</td><td>0.247</td><td>0.290</td><td>0.412</td><td>0.503</td><td>0.147</td><td>0.275</td><td>0.405</td><td>0.281</td><td>0.316</td><td>0.448</td></tr><tr><td>Noise</td><td>0.149</td><td>0.250</td><td>0.339</td><td>0.450</td><td>0.185</td><td>0.346</td><td>0.482</td><td>0.502</td><td>0.148</td><td>0.281</td><td>0.400</td><td>0.358</td><td>0.356</td><td>0.453</td></tr><tr><td>Impress</td><td>0.132</td><td>0.250</td><td>0.329</td><td>0.427</td><td>0.240</td><td>0.335</td><td>0.462</td><td>0.494</td><td>0.183</td><td>0.276</td><td>0.174</td><td>0.248</td><td>0.293</td><td>0.454</td></tr><tr><td>IDGuar (Ours)</td><td>0.036</td><td>0.282</td><td>0.126</td><td>0.450</td><td>0.220</td><td>0.351</td><td>0.444</td><td>0.494</td><td>0.142</td><td>0.273</td><td>0.336</td><td>0.220</td><td>0.301</td><td>0.452</td></tr></table>

Table 4. Ablation study of identity loss combinations and guidance directions, where G adv denotes guidance solely toward the adver sarial identity distribution and G clean denotes guidance solely away from the clean identity distribution.
<table><tr><td rowspan="2">Method</td><td colspan="2">Ip-Adapter</td><td colspan="2"> $\mathbf { \overline { { I p - A d a + X L } } }$ </td><td colspan="2">Blip-Diffu</td><td colspan="2">InfiniteID</td><td colspan="2">Photomaker</td><td colspan="2">Dreambooth</td><td colspan="2">InfiniteYou</td></tr><tr><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td><td>ISM↓</td><td>FQ↓</td></tr><tr><td>No Protected</td><td>0.382</td><td>0.334</td><td>0.582</td><td>0.456</td><td>0.336</td><td>0.331</td><td>0.675</td><td>0.495</td><td>0.327</td><td>0.456</td><td>0.525</td><td>0.332</td><td>0.673</td><td>0.493</td></tr><tr><td>CLIP</td><td>0.243</td><td>0.313</td><td>0.217</td><td>0.473</td><td>0.287</td><td>0.368</td><td>0.577</td><td>0.514</td><td>0.264</td><td>0.301</td><td>0.504</td><td>0.331</td><td>0.531</td><td>0.473</td></tr><tr><td>Facenet</td><td>0.246</td><td>0.319</td><td>0.222</td><td>0.479</td><td>0.288</td><td>0.372</td><td>0.583</td><td>0.508</td><td>0.268</td><td>0.308</td><td>0.503</td><td>0.330</td><td>0.531</td><td>0.466</td></tr><tr><td>CLIP + Facenet</td><td>0.175</td><td>0.271</td><td>0.167</td><td>0.469</td><td>0.255</td><td>0.378</td><td>0.536</td><td>0.513</td><td>0.213</td><td>0.292</td><td>0.497</td><td>0.298</td><td>0.477</td><td>0.468</td></tr><tr><td>IDGuar (Ours)</td><td>0.036</td><td>0.282</td><td>0.126</td><td>0.450</td><td>0.220</td><td>0.351</td><td>0.444</td><td>0.494</td><td>0.142</td><td>0.273</td><td>0.336</td><td>0.220</td><td>0.301</td><td>0.452</td></tr><tr><td>G adv</td><td>0.044</td><td>0.290</td><td>0.144</td><td>0.460</td><td>0.285</td><td>0.357</td><td>0.468</td><td>0.490</td><td>0.175</td><td>0.281</td><td>0.347</td><td>0.275</td><td>0.307</td><td>0.454</td></tr><tr><td>G clean</td><td>0.049</td><td>0.269</td><td>0.138</td><td>0.453</td><td>0.239</td><td>0.356</td><td>0.453</td><td>0.497</td><td>0.196</td><td>0.279</td><td>0.376</td><td>0.227</td><td>0.314</td><td>0.452</td></tr><tr><td>IDGuar (Ours)</td><td>0.036</td><td>0.282</td><td>0.126</td><td>0.450</td><td>0.220</td><td>0.351</td><td>0.444</td><td>0.494</td><td>0.142</td><td>0.273</td><td>0.336</td><td>0.220</td><td>0.301</td><td>0.452</td></tr></table>

Adversarial Guidance Direction. We study how the direction of adversarial score guidance influences optimization. We compare i) gradient toward the adversarial identity distribution $( \nabla _ { \mathbf { x } _ { t } } \log p ( \mathbf { y } _ { \mathrm { a d v } } \ | \ \mathbf { x } _ { t } ) )$ , and ii) gradient away from the clean identity distribution $( \nabla _ { \mathbf { x } _ { t } } \log p ( \mathbf { y } _ { \mathrm { c l e a n } } \mid \mathbf { x } _ { t } ) )$ Both are one-sided and often unstable due to distributional ambiguity, while our direction explicitly models the shift and achieves more stable identity suppression. Results are shown in Table 4. Qualitative visualizations are provided in Figure 9 of the Appendix. See Appendix B.2 for details.

Perturbation Budget and Learning Rate. We perform a grid search over perturbation magnitude ϵ and step size α to explore their effects on identity suppression and perceptual quality. As shown in Appendix Figure 3, the combination $\alpha = 0 . 0 0 5$ and $\epsilon = 8 / 2 5 5$ achieves the best balance, effectively reducing identity similarity while preserving visual fidelity. Smaller perturbations yield insufficient suppression, while larger ones introduce artifacts and degrade quality. Please refer to Appendix C.2 for more details.

Number Of Optimization Steps. We analyze convergence by tracking identity loss over 1000 iterations. As shown in Figure 4 of the Appendix, loss decreases rapidly in the first 100 steps and stabilizes after 200, with only marginal gains afterward. Thus, we set optimization steps to 200 for all experiments. Further details are in Appendix C.3.

## 5. Conclusion

We propose IDGuardian, an adversarial defense framework for proactive identity protection in diffusion- and flux-based personalized image generation. By learning imperceptible perturbations, it prevents generated outputs from preserving the original identity under similarity-based evaluation. IDGuardian integrates two complementary strategies: i) Cross-Encoder Identity Field Confounding, which reduces identity similarity across multiple encoders, and ii) Guidance-Flow Identity Deflection, which steers the diffusion trajectory away from the target identity manifold. It remains imperceptible while effectively suppressing identity information, and is robust to common postprocessing operations and adaptive removal attacks such as Impress. Experiments show that IDGuardian reduces identity consistency in generated outputs, mitigating identity leakage in personalized image synthesis.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China under Grants 62472231, 62172233, and 62502215; the Jiangsu Provincial Science and Technology Major Project under Grant BG2024042; the Natural Science Foundation of Jiangsu under Grant BK20250735; and the Postgraduate Research & Practice Innovation Program of Jiangsu Province under Grant SJCX25 0523.

## References

[1] Jinyang An, Wanqian Zhang, Dayan Wu, Zheng Lin, Jingzi Gu, and Weiping Wang. Sd4privacy: exploiting stable diffusion for protecting facial privacy. In 2024 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE, 2024. 3

[2] Bochuan Cao, Changjiang Li, Ting Wang, Jinyuan Jia, Bo Li, and Jinghui Chen. Impress: Evaluating the resilience of imperceptible perturbations against unauthorized data usage in diffusion-based generative ai. Advances in Neural Information Processing Systems, 36:10657–10677, 2023. 7

[3] Wen Chen, Pipei Huang, Jiaming Xu, Xin Guo, Cheng Guo, Fei Sun, Chao Li, Andreas Pfadler, Huan Zhao, and Binqiang Zhao. Pog: personalized outfit generation for fashion recommendation at alibaba ifashion. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2662–2670, 2019. 2

[4] Siying Cui, Jia Guo, Xiang An, Jiankang Deng, Yongle Zhao, Xinyu Wei, and Ziyong Feng. Idadapter: Learning mixed features for tuning-free personalization of text-toimage models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 950– 959, 2024. 2

[5] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2

[6] Zinan Guo, Yanze Wu, Chen Zhuowei, Peng Zhang, Qian He, et al. Pulid: Pure and lightning id customization via contrastive alignment. Advances in neural information processing systems, 37:36777–36804, 2024. 2

[7] Junjie He, Yifeng Geng, and Liefeng Bo. Uniportrait: A unified framework for identity-preserving singleand multi-human image personalization. arXiv preprint arXiv:2408.05939, 2024. 2

[8] Xiao He, Mingrui Zhu, Dongxin Chen, Nannan Wang, and Xinbo Gao. Diff-privacy: Diffusion-based face privacy protection. IEEE Transactions on Circuits and Systems for Video Technology, 2024. 3

[9] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 2328–2337, 2023. 5

[10] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5

[11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 1, 2

[12] Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, and Yisen Wang. Unlearnable examples: Making personal data unexploitable. arXiv preprint arXiv:2101.04898, 2021. 1

[13] Liming Jiang, Qing Yan, Yumin Jia, Zichuan Liu, Hao Kang, and Xin Lu. Infiniteyou: Flexible photo recrafting while preserving your identity. arXiv preprint arXiv:2503.16418, 2025. 6

[14] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1931–1941, 2023. 1, 2

[15] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36:30146–30166, 2023. 1, 6

[16] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8640–8650, 2024. 1, 2, 6

[17] Ziqiang Li, Jun Li, Lizhi Xiong, Zhangjie Fu, and Zechao Li. A comprehensive survey on visual concept mining in text-toimage diffusion models. arXiv preprint arXiv:2503.13576, 2025. 1, 2

[18] Ziqiang Li, Jiazhen Yan, Ziwen He, Kai Zeng, Weiwei Jiang, Lizhi Xiong, and Zhangjie Fu. Is artificial intelligence gen erated image detection a solved problem? arXiv preprint arXiv:2505.12335, 2025. 3

[19] Chumeng Liang, Xiaoyu Wu, Yang Hua, Jiaru Zhang, Yiming Xue, Tao Song, Zhengui Xue, Ruhui Ma, and Haibing Guan. Adversarial example does good: Preventing painting imitation from diffusion models via adversarial exam ples. arXiv preprint arXiv:2302.04578, 2023. 3, 7

[20] Jiang Liu, Chun Pong Lau, and Rama Chellappa. Diffprotect: Generate adversarial examples with diffusion models for facial privacy protection. arXiv preprint arXiv:2305.13625, 2023. 3

[21] Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, and Trevor Darrell. More control for free! im age synthesis with semantic diffusion guidance. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 289–299, 2023. 5

[22] Xinwei Liu, Xiaojun Jia, Yuan Xun, Hua Zhang, and Xi aochun Cao. Persguard: Preventing malicious personalization via backdoor attacks on pre-trained text-to-image diffu sion models. arXiv preprint arXiv:2502.16167, 2025. 3

[23] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Large-scale celebfaces attributes (celeba) dataset. Retrieved August, 15(2018):11, 2018. 6

[24] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017. 3

[25] David McAllister, Songwei Ge, Jia-Bin Huang, David Jacobs, Alexei Efros, Aleksander Holynski, and Angjoo Kanazawa. Rethinking score distillation as a bridge between image distributions. Advances in Neural Information Processing Systems, 37:33779–33804, 2024. 5

[26] Omkar Parkhi, Andrea Vedaldi, and Andrew Zisserman. Deep face recognition. In BMVC 2015-Proceedings of the British Machine Vision Conference 2015. British Machine Vision Association, 2015. 6

[27] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and¨ Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 6

[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 4

[29] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1

[30] Litu Rout, Yujia Chen, Nataniel Ruiz, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Rb-modulation: Training-free personalization of diffusion models using stochastic optimal control. arXiv preprint arXiv:2405.17401, 2024. 1

[31] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 1, 2, 6

[32] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 815–823, 2015. 4

[33] Shawn Shan, Jenna Cryan, Emily Wenger, Haitao Zheng, Rana Hanocka, and Ben Y Zhao. Glaze: Protecting artists from style mimicry by text-to-image models. In 32nd USENIX Security Symposium (USENIX Security 23), pages 2187–2204, 2023. 1

[34] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8543–8552, 2024. 2

[35] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015. 5

[36] Yang Song and Stefano Ermon. Generative modeling by esti mating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 5

[37] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Ab hishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 5

[38] Yiren Song, Shengtao Lou, Xiaokang Liu, Hai Ci, Pei Yang, Jiaming Liu, and Mike Zheng Shou. Anti-reference: Universal and immediate defense against reference-based generation. arXiv preprint arXiv:2412.05980, 2024. 3

[39] Yiren Song, Pei Yang, Hai Ci, and Mike Zheng Shou. Idprotector: An adversarial noise encoder to protect against id-preserving image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 3019–3028, 2025. 2, 3, 7

[40] Xuan Su, Jiaming Song, Chenlin Meng, and Stefano Ermon. Dual diffusion implicit bridges for image-to-image transla tion. arXiv preprint arXiv:2203.08382, 2022. 5

[41] Zekun Sun, Zijian Liu, Shouling Ji, Chenhao Lin, and Na Ruan. Pretender: Universal active defense against diffusion finetuning attacks. In 34nd USENIX Security Symposium (USENIX Security 25), 2025. 1

[42] Dani Valevski, Danny Lumen, Yossi Matias, and Yaniv Leviathan. Face0: Instantaneously conditioning a text-toimage model on a face. In SIGGRAPHAsia 2023 Conference Papers, pages 1–10, 2023. 1

[43] Thanh Van Le, Hao Phung, Thuan Hoang Nguyen, Quan Dao, Ngoc N Tran, and Anh Tran. Anti-dreambooth: Protecting users from personalized text-to-image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2116–2127, 2023. 1, 3, 7

[44] Feifei Wang, Zhentao Tan, Tianyi Wei, Yue Wu, and Qidong Huang. Simac: A simple anti-customization method for protecting face privacy against text-to-image synthesis of diffu sion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12047– 12056, 2024. 3, 7

[45] Hanhui Wang, Yihua Zhang, Ruizheng Bai, Yue Zhao, Sijia Liu, and Zhengzhong Tu. Edit away and my face will not stay: Personal biometric defense against malicious generative editing. arXiv preprint arXiv:2411.16832, 2024. 1

[46] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 2

[47] Yi Wu, Ziqiang Li, Heliang Zheng, Chaoyue Wang, and Bin Li. Infinite-id: Identity-preserved personalization via idsemantics decoupling paradigm. In European Conference on Computer Vision, pages 279–296. Springer, 2024. 1, 2, 6

[48] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721, 2023. 1, 2, 6

[49] Xiaoyu Ye, Hao Huang, Jiaqi An, and Yongtao Wang. Duaw: Data-free universal adversarial watermark against stable dif fusion customization. arXiv preprint arXiv:2308.09889, 2023. 3

[50] Boyang Zheng, Chumeng Liang, Xiaoyu Wu, and Yan Liu. Understanding and improving adversarial attacks on latent diffusion model. 2023. 3, 7

[51] Zhengguang Zhou, Jing Li, Huaxia Li, Nemo Chen, and Xu Tang. Storymaker: Towards holistic consistent characters in text-to-image generation. arXiv preprint arXiv:2409.12576, 2024. 2

[52] Tianrui Zhu, Shiyi Zhang, Jiawei Shao, and Yansong Tang. Kv-edit: Training-free image editing for precise background preservation. arXiv preprint arXiv:2502.17363, 2025. 2

# No Way To Steal My Face: Proactive Defense Against Identity-Preserving Personalized Generation

Supplementary Material

## A. Methodological and Evaluation Details

While the main paper introduces the two-stage defense principle of IDGuardian, this section provides the complete algorithmic details of its optimization process. Specifically, IDGuardian jointly optimizes adversarial perturbations through two complementary strategies—Cross-Encoder Identity Field Confounding and Guidance-Flow Identity Deflection—which correspond to the disruption of identity extraction and injection, respectively. For clarity and reproducibility, we detail the optimization objectives, loss composition, and implementation settings that were omitted for brevity in the main text.

## A.1. Optimization Process of IDGuardian

Our defense operates in the latent denoising space of the pretrained SDXL backbone within the IP-Adapter SDXL Plus Face pipeline, and optimizes an adversarial perturbation δ on the input image $x ,$ such that the final protected image $x _ { \mathrm { a d v } } = x + \delta$ preserves perceptual quality while significantly reducing identity consistency in downstream personalized generation.

The optimization is carried out iteratively over $N _ { \mathrm { i t e r } }$ steps. At each step:

• The input clean image is encoded into the latent space using the VAE encoder.

• A random diffusion timestep t is sampled, and corresponding noise is added to simulate a forward noisy latent $x _ { t } .$

• Identity embeddings are computed from the clean and perturbed versions of the input, and serve as conditioning inputs for the UNet during denoising.

• Two complementary gradients are computed: i) the identity loss gradient $\nabla _ { \delta } \mathcal { L } _ { \mathrm { I D } }$ based on FaceNet and CLIP similarity; ii) a distributional score gradient $S ^ { * }$ , which measures the semantic divergence between adversarial and clean identity trajectories in diffusion space.

• To match the spatial resolution and RGB channels of the perturbation, $S ^ { * }$ is upsampled via bilinear interpolation (PyTorch: F.interpolate(..., $m o d e { = } ^ { \prime } b i l i n e a r ^ { \prime } $ align corners=False)) and only the first three channels (RGB) are retained to align with the perturbation tensor.

• These gradients are normalized and combined to update δ via a projected gradient descent step, bounded by the perturbation constraint $\| \delta \| _ { \infty } < \epsilon .$

Algorithm 1 outlines the full optimization pipeline. This instantiation reflects a synergy between semantic featurespace objectives and internal generative guidance dynamics, ensuring that the optimization is both perceptually aware and diffusion-aligned.

Algorithm 1 IDGuardian   
1: Input: original image x, perturbation budget ϵ, learning rate   
α, text prompt y, max steps $N _ { \mathrm { i t e r } }$   
2: Output: optimized adversarial image $x _ { \mathrm { a d v } }$   
3: Load pretrained IP-Adapter SDXL Plus Face pipeline com  
ponents: VAE Encoder, UNet, and CLIP image encoder   
4: Encode image into latent space: $x _ { 0 } \gets \mathrm { V A E } .$ Encoder(x)   
5: Initialize perturbation $\delta \gets 0$   
6: for $i = 1$ to $N _ { \mathrm { i t e r } }$ do   
7: Sample random timestep $t ,$ corresponding noise $\epsilon _ { t }$   
8: Compute noisy latent: $x _ { t } \gets \mathrm { A d d N o i s e } ( x _ { 0 } , \epsilon _ { t } , t )$   
9: x<sub>adv</sub> $\gets \mathrm { c l i p } ( x + \delta , 0 , 1 )$   
10: Compute embeddings: $\mathrm { C L I P } ( x ) , \mathrm { C L I P } ( x _ { \mathrm { a d v } } )$   
11: Predict denoising outputs:   
12: $\hat { \epsilon } _ { \mathrm { c l e a n } } \gets \mathrm { U N e t } ( x _ { t } , t , \mathrm { C L I P } ( x ) , y )$   
13: $\hat { \epsilon } _ { \mathrm { a d v } } \gets \mathrm { U N e t } ( x _ { t } , t , \mathrm { C L I P } ( x _ { \mathrm { a d v } } ) , y )$   
14: Compute score gradient:   
15: $\begin{array} { r } { S ^ { * } \gets - \frac { 1 } { \sqrt { 1 - \bar { \alpha } _ { t } } } \cdot ( \hat { \epsilon } _ { \mathrm { a d v } } - \hat { \epsilon } _ { \mathrm { c l e a n } } ) } \end{array}$   
16: Upsample $S ^ { * }$ to image resolution using bilinear interpola  
tion and align channels with RGB:   
17: $S _ { \mathrm { u p } } ^ { * } \gets \mathrm { B i l i n e a r U p } ( S ^ { * } , \mathrm { s i z e } = x . \mathrm { s h a p e } ) [ : , : 3 , : , : ]$   
18: Compute identity loss gradient: $\nabla _ { \delta } \mathcal { L } _ { \mathrm { I D } }$   
19: Combine gradients:   
20: total grad ← Normalize $\left( \nabla _ { \delta } \mathcal { L } _ { \mathrm { I D } } \right)$ − Normalize $( S _ { \mathrm { u p } } ^ { * } )$   
21: Update perturbation: $\delta  \delta - \alpha$ · sign(total grad)   
22: Clip: $\delta \gets \mathrm { c l i p } ( \delta , - \epsilon , \epsilon )$   
23: end for   
24: Return $x _ { \mathrm { a d v } } = x + \delta$

## A.2. DreamBooth Fine-Tuning and Similarity Evaluation

For DreamBooth and other training-based methods, which typically require multiple images per identity for finetuning, we provide four distinct images per identity. For each generated image of that identity, we compute its similarity to each of the four original images, yielding four similarity scores. We then average these four scores to obtain a single similarity metric for that generated image. This procedure is repeated for all generated images of the identity, and the final reported similarity is obtained by averaging across all generated images. This ensures that the evaluation reflects the model’s performance using all available reference images rather than relying on a single image.

## A.3. Reproducibility Resources

For reproducibility, we list the main public resources used in this work:

• FaceNet implementation: We adopt the official FaceNet implementation from the GitHub repository davidsandberg / facenet, using the Inception-ResNet-v1 checkpoint with identifier 20180402-114759 in our loss computation.

• Evaluation toolkit: We use the public DeepFace toolkit from the GitHub repository serengil/deepface for identity similarity evaluation.

• Datasets: The VGGFace2 dataset is accessed from the HuggingFace repository datasets / ProgramComputer / VGGFace2. The CelebA-HQ dataset is obtained from the GitHub repository IIGROUP/MM-CelebA-HQ-Dataset.

![](images/265e5b8c773a6b034f3adb9040ef3ab9c824e16a78d03fcfa42f8d40d5e3c78c.jpg)  
Figure 5. Statistical analysis of identity similarity (ISM) under IP-Adapter. Error bars denote the variability across repeated runs.

## B. Detailed Quantitative Results

## B.1. Comparative Evaluation with Baselines

In the main paper, we summarized the performance of ID-Guardian against baseline methods using average identity similarity scores and a representative perceptual quality metric. Here, we provide comprehensive quantitative results for all individual identity similarity metrics (ArcFace, FaceNet, VGGFace) and perceptual quality metrics (Face-QNet, MagFace) across the tested methods. Table 7 and Table 8 present detailed comparisons highlighting the consistent superiority of IDGuardian in identity suppression and its effectiveness in preserving perceptual quality.

We further include statistical analysis results, demonstrating that the observed performance gains remain stable and consistent across repeated runs. Taking IP-Adapter as an example, as shown in Figure 5, IDGuardian exhibits consistently lower identity similarity with small variance compared to baseline methods.

![](images/9bd81d81bd6117ec30b3a2f593027a79d37ea032265105c86319225463e31559.jpg)  
Figure 6. Visual results under three levels of JPEG compression, Gaussian blur, and Gaussian noise, showing that our protection mechanism remains effective even under stronger distortions.

## B.2. Ablation Study on Identity Similarity and Perceptual Quality Metrics

We further expand the ablation study presented in the main paper by showing the full set of identity similarity metrics and perceptual quality metrics across various ablation configurations. Tables 9 and 10 provide detailed insights into the contributions of different identity encoder combinations and adversarial conceptual bridge. These results validate the complementary roles of identity supervision and conceptual bridge in enhancing identity suppression, where visual degradation is part of the intended defense effect.

## B.3. Robustness Evaluation Settings.

To evaluate the robustness of our protection mechanism, we applied three types of distortions to the protected images: JPEG compression, Gaussian blur, and Gaussian noise. Specifically, we used JPEG compression with a quality factor of 85, Gaussian blur with a kernel size of $5 \times 5$ and $\sigma ~ = ~ 1 . 5$ , and noise perturbation with a variance of 100. These parameters were selected to simulate common post-processing or transmission degradations while maintaining perceptual plausibility. As illustrated in Figure 5 of the main text, the protection performance under these distortions remains comparable to that on undistorted inputs, demonstrating that our method exhibits strong robustness against typical image degradations. To further evaluate the robustness, we conducted additional experiments with stronger distortions. Specifically, we tested two additiona levels of perturbations: JPEG compression with quality factors of 70 and 50, Gaussian blur with kernel sizes of $7 \times 7$ and $1 1 \times 1 1$ (both with $\sigma = 1 . 5 )$ , and Gaussian noise with variances of 400 and 900. Table 5 summarizes the results, demonstrating that our method maintains protection performance even under these more severe image degradations.

Table 5. Robustness evaluation under different distortion levels. The results are reported on Ip-Adapter Plus XL as an example.
<table><tr><td colspan="3">No Protected</td><td colspan="3">JPEG (85)</td><td colspan="3">Blur (5 × 5, 1.5)</td><td colspan="3">Noise (100)</td></tr><tr><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>0.4985</td><td>0.6899</td><td>0.5573</td><td>0.3209</td><td>0.3983</td><td>0.3006</td><td>0.3317</td><td>0.4443</td><td>0.3520</td><td>0.3078</td><td>0.3970</td><td>0.3113</td></tr><tr><td colspan="3">No Protected</td><td colspan="3">JPEG (70)</td><td colspan="3">Blur (7 × 7, 1.5)</td><td colspan="3">Noise (400)</td></tr><tr><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>0.4985</td><td>0.6899</td><td>0.5573</td><td>0.3520</td><td>0.4863</td><td>0.3730</td><td>0.3303</td><td>0.4473</td><td>0.3338</td><td>0.3955</td><td>0.4778</td><td>0.3746</td></tr><tr><td colspan="3">No Protected</td><td></td><td>JPEG (50)</td><td></td><td>Blur (11 × 11, 1.5)</td><td></td><td></td><td></td><td>Noise (900)</td><td></td></tr><tr><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>0.4985</td><td>0.6899</td><td>0.5573</td><td>0.3426</td><td>0.4606</td><td>0.3382</td><td>0.3416</td><td>0.4728</td><td>0.3632</td><td>0.3094</td><td>0.4495</td><td>0.3604</td></tr></table>

As illustrated in Figure 6, we also provide visual comparisons under three levels of post-processing strength, showing that the perceptual quality of the protected images remains largely preserved despite the increased distortions.

## B.4. Robustness Evaluation on Identity Similarity and Perceptual Quality Metrics

Robustness experiments are conducted under both common image degradations (JPEG compression, Gaussian blur, and additive noise) and adaptive defense-aware attacks such as Impress, which is explicitly designed to circumvent identity protection mechanisms. As shown in Tables 11 and 12, IDGuardian maintains low identity similarity scores across all perturbation types, while perceptual quality metrics naturally decline under heavy input distortion. These results demonstrate that our method retains strong protection capability even under adverse conditions. Qualitative examples in Figure 12 further support these findings by illustrating that identity-relevant features remain suppressed even under various post-processing transformations. These detailed robustness results complement the summary metrics presented in the main paper and underscore IDGuardian’s practical effectiveness in real-world usage scenarios.

## C. Additional Ablation Study

## C.1. Visualization of the Main Ablation Results

To further validate the effectiveness of our proposed identity guidance strategy and loss function design, we provide an additional visualization of the ablation results in Figure 13. The figure clearly demonstrates the influence of different identity loss combinations and guidance directions on the overall performance. Consistent with the quantitative results in Table 4 of the main text, IDGuardian achieves superior identity protection while maintaining high visual quality.

## C.2. Perturbation Budget and Learning Rate

We perform a grid search over the perturbation magnitude ϵ and learning rate α to determine an optimal trade-off between identity consistency suppression and perceptual quality. Specifically, we evaluate the following five configurations:

$$
{ \begin{array} { l } { \bullet \ \alpha = 0 . 0 0 5 , \epsilon = 4 / 2 5 5 } \\ { \bullet \ \alpha = 0 . 0 0 5 , \epsilon = 8 / 2 5 5 } \\ { \bullet \ \alpha = 0 . 0 0 5 , \epsilon = 1 6 / 2 5 5 } \\ { \bullet \ \alpha = 0 . 0 0 1 , \epsilon = 8 / 2 5 5 } \\ { \bullet \ \alpha = 0 . 0 1 , \epsilon = 8 / 2 5 5 } \end{array} }
$$

As depicted in Figure 7, the configuration α = 0.005, ϵ = 8/255 achieves the best balance:

• It consistently reduces identity consistency below 0.15 across ArcFace, FaceNet, and VGGFace.

• It maintains high perceptual quality, with SSIM exceeding 0.8419 and PSNR surpassing 32.186 dB.

Smaller perturbations (e.g., ϵ = 4/255) yield insufficient identity suppression, while larger ones (e.g., ϵ = 16/255) compromise visual fidelity. Similarly, extremely small learning rates (α = 0.001) lead to under-optimization, whereas large values (α = 0.01) introduce instability and visual artifacts. These observations underscore the importance of jointly tuning the perturbation magnitude and step size, as a delicate balance is required to ensure both identity confusion and natural-looking outputs.

## C.3. Number of Optimization Steps

We investigate the convergence behavior of identity loss over a range of optimization steps, up to 1000 iterations. As shown in Figure 8, the identity loss declines rapidly during the initial 100 steps, indicating early-stage effectiveness in suppressing identity-related features. To illustrate optimization dynamics more clearly, we apply a shaded band using $\mathtt { f i l 1 \_ b e t w e e n } ^ { 2 }$ to visualize artificial fluctuation bounds around the loss curve. This is not a confidence interval but a heuristic indicator of stability and oscillation patterns.

![](images/d6f765b5cc9bcc161e3ce12d79d88eb01583d013fe038b1b6fd8f4e3d140a1f4.jpg)

![](images/e174b945f2ac6b037394d6d03a603742dd82019fc0c6de8cd1184f7416a579ac.jpg)  
Figure 7. Grid search results for perturbation budget ϵ and learning rate α. The configuration $\alpha = 0 . 0 0 5 , \epsilon = 8 / 2 5 5$ achieves the best trade-off between identity suppression (measured by ArcFace, FaceNet, and VGGFace similarity) and perceptual quality (SSIM and PSNR).

Table 6. Identity Suppression Performance Across Surrogate Generative Models. The metrics Arc, Face, and VGG correspond to identity similarity scores computed by the face recognition models ArcFace, FaceNet, and VGGFace, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">Ip-Adapter</td><td colspan="2">Ip-Adapter Plus XL</td><td colspan="2">Blip-Diffusion</td><td colspan="3">InfiniteID</td></tr><tr><td>Arc ↓ Face ↓</td><td>VGG↓</td><td>Arc ↓ Face↓</td><td>VGG↓</td><td>Arc ↓ Face ↓</td><td>VGG↓</td><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>No Protected</td><td>0.3369</td><td>0.4774 0.3308</td><td>0.4985 0.6899</td><td>0.5573</td><td>0.3305 0.3918</td><td>0.2848</td><td>0.6345</td><td>0.7627</td><td>0.6286</td></tr><tr><td>IDGuardian-SD1.5</td><td>0.0425</td><td>0.0119 0.0650</td><td>0.1589 0.1413</td><td>0.1403</td><td>0.2670 0.2550</td><td>0.2204</td><td>0.4338</td><td>0.4989</td><td>0.4113</td></tr><tr><td rowspan="2">Method</td><td>Photomaker</td><td></td><td>Dreambooth</td><td></td><td>InfiniteYou</td><td></td><td></td><td>Mean</td><td></td></tr><tr><td>Arc↓</td><td>Face ↓ VGG↓</td><td>Arc↓ Face↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face↓ VGG↓</td><td>Arc↓</td><td>Face↓</td><td>VGG↓</td></tr><tr><td>No Protected</td><td>0.3488 0.3513</td><td>0.2806</td><td>0.4681 0.6580</td><td>0.4477</td><td>0.6788</td><td>0.7554 0.5847</td><td>0.4708</td><td>0.5838</td><td>0.4450</td></tr><tr><td>IDGuardian-SD1.5</td><td>0.1353 0.1556</td><td>0.2233</td><td>0.2907 0.3940</td><td>0.3455</td><td>0.3328</td><td>0.3573 0.2809</td><td>0.2373</td><td>0.2591</td><td>0.2409</td></tr></table>

The loss saturates around 200 iterations, beyond which further gains become marginal. Based on this trend, we adopt $N _ { \mathrm { i t e r } } ~ = ~ 2 0 0$ as the default number of steps in our experiments, ensuring adequate convergence without excessive computation.

## D. Generalization Analysis

## D.1. Evaluation on Commercial Face Personalization APIs

To further validate the generalization ability of our protection method, we conducted additional evaluations on several commercial face personalization APIs, including Seedream 4.0, Owen-Image, Kling-AI, and Ihuiwa. As shown

![](images/5bf6c786b3b67af279c91b96b388013ce61b9ef472c3feef7e5681f0b0b3ec2c.jpg)  
Figure 8. Identity loss convergence over optimization steps with shaded band indicating fluctuation bounds (±0.1) around the mean curve. The identity loss stabilizes after approximately 200 steps, guiding the selection of the default iteration count.

in Figure 9, our method still provides a certain degree of defense under these black-box conditions, demonstrating its robustness and transferability beyond the training environ-

![](images/98b9fc8d7e3763c306f100c3572ebc2b2053ba735abb46d5517d69b323a365d4.jpg)  
Figure 9. Visualization of generalization results on commercial face personalization APIs.

ment.

## D.2. Flexibility in Surrogate Selection

To evaluate the generalizability of our approach at the model level, we replace the surrogate model used for adversarial optimization with a significantly different generative backbone. Our main experiments utilize the IP-Adapter SDXL Plus Face model based on Stable Diffusion XL (SDXL). Additionally, we assess our method on the earlier Stable Diffusion 1.5 (SD1.5) backbone along with its corresponding IP-Adapter variant. All attack configurations, including text prompts and optimization parameters, are held constant to ensure comparability.

The results demonstrate that the proposed method consistently achieves strong identity suppression across distinct surrogate models, confirming its robustness and applicability beyond specific model architectures, as summarized in Table 6. Representative visual results further illustrate the consistency and effectiveness of identity suppression across models, as shown in Figure 10.

## D.3. Dataset Generalization

In addition to the VGGFace2 dataset results presented in the main paper, we further evaluate the method on the highresolution CelebA-HQ dataset. We employ multiple identity similarity metrics, namely ArcFace, FaceNet, and VG-GFace, to measure the extent of identity removal across diverse personalized generation models. Perceptual quality is concurrently assessed through FaceQnet and MagFace to verify preservation of visual fidelity.

The quantitative results summarized in Table 13 and Table 14 indicate that the method maintains effective identity suppression and reduced perceptual quality on CelebA-HQ, thus demonstrating robustness across datasets varying in resolution and demographic composition. As illustrated in Figure 14, the visual results confirm that the method successfully removes identity-specific features while preserving photorealism, supporting the effectiveness of the proposed defense. We further provide results on the LFW dataset using the identity similarity metric in Table 15. The results remain consistent with those on VGGFace2 and CelebA-HQ, further demonstrating the robustness and cross-dataset generalization ability of IDGuardian. Collectively, these experiments verify that IDGuardian consistently preserves identity suppression capability and visual quality across different surrogate model architectures and datasets, highlighting its strong generalization capability in diffusion-based personalized face generation.

## E. Human and GPT-Based Evaluation of Identity Protection Effectiveness

As illustrated in Figure 11, the upper part of the figure presents the results of the human perceptual study, while the lower part corresponds to the GPT-based evaluation. This joint visualization highlights the consistency between subjective human judgments and automated assessments, with our method favored in 92% of human responses and 94% of GPT comparisons across over 50 test cases, further reinforcing the effectiveness of our approach.

## E.1. Human Study

To further assess the effectiveness of identity protection from a human perceptual perspective, we conducted a comprehensive subjective evaluation involving human participants. These participants were tasked with rating the perceived identity similarity between the original identity images and the personalized images generated by our method as well as several state-of-the-art baseline methods. The results consistently demonstrate that images produced by our approach exhibit significantly lower perceived identity similarity compared to those generated by other methods, thereby indicating superior identity obfuscation. This extensive subjective evaluation complements the quantitative metrics by capturing nuanced perceptual differences and providing insights into human visual cognition that automated measures alone may not fully capture.

## E.2. GPT-Based Evaluation

In addition to the human evaluation, we also employed a state-of-the-art large multimodal language model, GPT, to automatically assess the identity protection performance. Given pairs of original and generated images, GPT was prompted to analyze and identify which generated image exhibits the greatest discrepancy in identity features relative to the original input. Leveraging GPT’s advanced visuallanguage reasoning capabilities, this automated evaluation approach offers a scalable and efficient complement to human judgment. The findings from the GPT-based evaluation are consistent with the results obtained from the user study, thereby further validating the robustness and effectiveness of our method in substantially reducing identity similarity across different evaluation paradigms.

![](images/f1a774857519400e0e11bd495ba8afbf194e03b32f852a09afef84bcc0ac3a4d.jpg)  
Figure 10. Visual comparison of identity suppression results across different surrogate generative models.

![](images/e63477685a012492e79d94c93fb19330510634aa104e4612a6f9ccc685f2afd9.jpg)  
Figure 11. Comparison of Identity Protection Effectiveness Based on Human Perception and GPT-Based Evaluation.

## E.3. Ethics Statement

Our user study involved voluntary participation of adult subjects recruited within our institution. Participants were asked to provide subjective ratings of visual similarity without disclosing or providing any personally identifiable or sensitive information. The study used publicly available face datasets and collected no new identifiable personal data. All user evaluations were performed on synthetic or anonymized data, and no sensitive or personally identifiable information was recorded. The study posed no potential harm or risk to participants. While formal Institutional Review Board (IRB) approval was not required under our institution’s policy for this low-risk, anonymized perceptual evaluation, all participants were informed about the study’s purpose and provided consent prior to participation. The study was conducted in accordance with recognized ethical research principles.

Table 7. Comparison of identity protection performance against baseline methods. Evaluation is conducted using three different identity similarity metrics: ArcFace, FaceNet, and VGGFace. “No Protected” denotes generation without defense.
<table><tr><td rowspan="2">Method</td><td colspan="3">Ip-Adapter</td><td colspan="3">Ip-Adapter Plus XL</td><td colspan="3">Blip-Diffusion</td><td colspan="3">InfiniteID</td></tr><tr><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>No Protected</td><td>0.3369</td><td>0.4774</td><td>0.3308</td><td>0.4985</td><td>0.6899</td><td>0.5573</td><td>0.3305</td><td>0.3918</td><td>0.2848</td><td>0.6345</td><td>0.7627</td><td>0.6286</td></tr><tr><td>Anti-DB</td><td>0.2525</td><td>0.3964</td><td>0.2715</td><td>0.4815</td><td>0.6419</td><td>0.5219</td><td>0.2685</td><td>0.3153</td><td>0.2286</td><td>0.5897</td><td>0.6859</td><td>0.5848</td></tr><tr><td>SimAC</td><td>0.2048</td><td>0.4005</td><td>0.2792</td><td>0.3454</td><td>0.4634</td><td>0.3956</td><td>0.3238</td><td>0.3344</td><td>0.2563</td><td>0.5472</td><td>0.6670</td><td>0.5599</td></tr><tr><td>AdvDM</td><td>0.2409</td><td>0.3624</td><td>0.2535</td><td>0.3427</td><td>0.4810</td><td>0.3700</td><td>0.2461</td><td>0.2989</td><td>0.2363</td><td>0.5069</td><td>0.6107</td><td>0.5087</td></tr><tr><td>ACE</td><td>0.2573</td><td>0.4249</td><td>0.2996</td><td>0.4615</td><td>0.6377</td><td>0.5079</td><td>0.2738</td><td>0.3160</td><td>0.2336</td><td>0.5419</td><td>0.6489</td><td>0.5383</td></tr><tr><td>IDProtector</td><td>0.2059</td><td>0.4167</td><td>0.2657</td><td>0.3324</td><td>0.5297</td><td>0.4120</td><td>0.2565</td><td>0.2967</td><td>0.2346</td><td>0.4972</td><td>0.7009</td><td>0.6112</td></tr><tr><td>IDGuar (Ours)</td><td>0.0673</td><td>-0.0226</td><td>0.0640</td><td>0.1296</td><td>0.1231</td><td>0.1260</td><td>0.2255</td><td>0.2376</td><td>0.1964</td><td>0.4121</td><td>0.5215</td><td>0.3990</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="3">Photomaker</td><td colspan="3">Dreambooth</td><td colspan="3">InfiniteYou</td><td colspan="3">Mean</td></tr><tr><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>No Protected</td><td>0.3488</td><td>0.3513</td><td>0.2806</td><td>0.4681</td><td>0.6580</td><td>0.4477</td><td>0.6788</td><td>0.7554</td><td>0.5847</td><td>0.4709</td><td>0.5838</td><td>0.4449</td></tr><tr><td>Anti-DB</td><td>0.2331</td><td>0.3104</td><td>0.2716</td><td>0.3590</td><td>0.5420</td><td>0.3976</td><td>0.5867</td><td>0.6569</td><td>0.5343</td><td>0.3959</td><td>0.5070</td><td>0.4014</td></tr><tr><td>SimAC</td><td>0.2267</td><td>0.2614</td><td>0.2564</td><td>0.2987</td><td>0.4760</td><td>0.3942</td><td>0.5604</td><td>0.6472</td><td>0.5039</td><td>0.3581</td><td>0.4643</td><td>0.3780</td></tr><tr><td>AdvDM</td><td>0.2083</td><td>0.2189</td><td>0.2477</td><td>0.2191</td><td>0.4507</td><td>0.3327</td><td>0.5546</td><td>0.6309</td><td>0.4905</td><td>0.3312</td><td>0.4362</td><td>0.3485</td></tr><tr><td>ACE</td><td>0.2246</td><td>0.2545</td><td>0.2756</td><td>0.3252 </td><td>0.5328</td><td>0.4033</td><td>0.6450</td><td>0.7207</td><td>0.5504</td><td>0.3900</td><td>0.5051</td><td>0.4012</td></tr><tr><td>IDProtector</td><td>0.2170</td><td>0.2212</td><td>0.2807</td><td>0.3372</td><td>0.4768</td><td>0.3932</td><td>0.5047</td><td>0.6067</td><td>0.5105</td><td>0.3358</td><td>0.4641</td><td>0.3868</td></tr><tr><td>IDGuar (Ours)</td><td>0.0747</td><td>0.1284</td><td>0.2239</td><td>0.2792</td><td>0.3893</td><td>0.3404</td><td>0.3230</td><td>0.3265</td><td>0.2522</td><td>0.2159</td><td>0.2434</td><td>0.2288</td></tr></table>

Table 8. Comparison of identity protection performance against baseline methods. Evaluation is conducted using FaceQNet and MagFace quality scores. “No Protected” denotes generation without defense.
<table><tr><td rowspan="2">Method</td><td colspan="2">Ip-Adapter</td><td colspan="2">Ip-Adapter Plus XL</td><td colspan="2">Blip-Diffusion</td><td colspan="2">InfiniteID</td></tr><tr><td>FaceQNet↓ MagFace ↓</td><td></td><td>FaceQNet↓ MagFace ↓</td><td></td><td>FaceQNet ↓ MagFace ↓</td><td></td><td>FaceQNet ↓ MagFace ↓</td><td></td></tr><tr><td>No Protected</td><td>0.3339</td><td>21.3847</td><td>0.4564</td><td>19.3085</td><td>0.3305</td><td>21.8112</td><td>0.4946</td><td>19.0747</td></tr><tr><td>Anti-DB</td><td>0.3045</td><td>21.1209</td><td>0.4553</td><td>18.4376</td><td>0.3565</td><td>20.8991</td><td>0.4957</td><td>18.9697</td></tr><tr><td>SimAC</td><td>0.3124</td><td>20.8217</td><td>0.4492</td><td>18.9702</td><td>0.3628</td><td>21.4539</td><td>0.4844</td><td>19.0755</td></tr><tr><td>AdvDM</td><td>0.3486</td><td>20.6235</td><td>0.4379</td><td>18.8501</td><td>0.3547</td><td>21.6632</td><td>0.4896</td><td>18.9385</td></tr><tr><td>ACE</td><td>0.3627</td><td>20.6235</td><td>0.4528</td><td>18.5293</td><td>0.3630</td><td>21.7078</td><td>0.4994</td><td>19.0961</td></tr><tr><td>IDProtector</td><td>0.3367</td><td>19.9330</td><td>0.4318</td><td>18.8869</td><td>0.3680</td><td>21.3936</td><td>0.4885</td><td>18.9833</td></tr><tr><td>IDGuar (Ours)</td><td>0.2820</td><td>20.8712</td><td>0.4503</td><td>18.6099</td><td>0.3511</td><td>20.8498</td><td>0.4938</td><td>18.9031</td></tr><tr><td rowspan="2">Method</td><td colspan="2">Photomaker</td><td colspan="2">Dreambooth</td><td colspan="2">InfiniteYou</td><td colspan="2">Mean</td></tr><tr><td>FaceQNet ↓ MagFace ↓</td><td></td><td>FaceQNet↓</td><td>MagFace ↓</td><td>FaceQNet↓ MagFace ↓</td><td></td><td>FaceQNet ↓ MagFace ↓</td><td></td></tr><tr><td>No Protected</td><td>0.4564</td><td>20.0541</td><td>0.3316</td><td>20.2766</td><td>0.4927</td><td>20.7298</td><td>0.4137</td><td>20.3771</td></tr><tr><td>Anti-DB</td><td>0.2883</td><td>20.0427</td><td>0.3186</td><td>20.2498</td><td>0.4666</td><td>19.9501</td><td>0.3836</td><td>19.9529</td></tr><tr><td>SimAC</td><td>0.2959</td><td>19.9726</td><td>0.3427</td><td>20.1656</td><td>0.4590</td><td>19.9077</td><td>0.3866</td><td>20.0525</td></tr><tr><td>AdvDM</td><td>0.2934</td><td>20.1307</td><td>0.1921</td><td>19.7116</td><td>0.4586</td><td>19.9321</td><td>0.3678</td><td>19.9785</td></tr><tr><td>ACE</td><td>0.2866</td><td>20.1201</td><td>0.2147</td><td>20.1827</td><td>0.4610</td><td>19.8904</td><td>0.3772</td><td>19.8786</td></tr><tr><td>IDProtector</td><td>0.3119</td><td>28.5475</td><td>0.2248</td><td>20.6773</td><td>0.4535</td><td>19.5957</td><td>0.3736</td><td>21.1453</td></tr><tr><td>IDGuar (Ours)</td><td>0.2733</td><td>19.8202</td><td>0.2203</td><td>20.2199</td><td>0.4517</td><td>19.8111</td><td>0.3604</td><td>19.8693</td></tr></table>

Table 9. Ablation study of identity loss combinations and guidance directions, where G adv denotes guidance solely toward the adversarial identity distribution and G clean denotes guidance solely away from the clean identity distribution.
<table><tr><td rowspan="2">Method</td><td colspan="3">Ip-Adapter</td><td colspan="3">Ip-Adapter Plus XL</td><td colspan="3">Blip-Diffusion</td><td colspan="3">InfiniteID</td></tr><tr><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>No Protected</td><td>0.3369</td><td>0.4774</td><td>0.3308</td><td>0.4985</td><td>0.6899</td><td>0.5573</td><td>0.3305</td><td>0.3918</td><td>0.2848</td><td>0.6345</td><td>0.7627</td><td>0.6286</td></tr><tr><td>CLIP</td><td>0.2289</td><td>0.3058</td><td>0.1949</td><td>0.2032</td><td>0.2328</td><td>0.2147</td><td>0.2864</td><td>0.3078</td><td>0.2664</td><td>0.5048</td><td>0.6775</td><td>0.5478</td></tr><tr><td>Facenet</td><td>0.2412</td><td>0.2897</td><td>0.2058</td><td>0.2145</td><td>0.2281</td><td>0.2223</td><td>0.2948</td><td>0.2989</td><td>0.2716</td><td>0.5126</td><td>0.6843</td><td>0.5534</td></tr><tr><td>CLIP + Facenet</td><td>0.1578</td><td>0.2347</td><td>0.1326</td><td>0.1665</td><td>0.1707</td><td>0.1638</td><td>0.2534</td><td>0.2764</td><td>0.2355</td><td>0.4932</td><td>0.6047</td><td>0.5089</td></tr><tr><td>IDGuar (Ours)</td><td>0.0673</td><td>-0.0226</td><td>0.0640</td><td>0.1296</td><td>0.1231</td><td>0.1260</td><td>0.2255</td><td>0.2376</td><td>0.1964</td><td>0.4121</td><td>0.5215</td><td>0.3990</td></tr><tr><td>G adv</td><td>0.0688</td><td>-0.0006</td><td>0.0650</td><td>0.1318</td><td>0.1476</td><td>0.1536</td><td>0.2971</td><td>0.3072</td><td>0.2509</td><td>0.4501</td><td>0.5369</td><td>0.4181</td></tr><tr><td>G clean</td><td>0.0869</td><td>-0.0061</td><td>0.0668</td><td>0.1442</td><td>0.1274</td><td>0.1432</td><td>0.2400</td><td>0.2751</td><td>0.2029</td><td>0.4265</td><td>0.5261</td><td>0.4073</td></tr><tr><td>IDGuar (Ours)</td><td>0.0673</td><td>-0.0226</td><td>0.0640</td><td>0.1296</td><td>0.1231</td><td>0.1260</td><td>0.2255</td><td>0.2376</td><td>0.1964</td><td>0.4121</td><td>0.5215</td><td>0.3990</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="3">Photomaker</td><td colspan="3">Dreambooth</td><td colspan="3">InfiniteYou</td><td colspan="3">Mean</td></tr><tr><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc↓</td><td>Face↓</td><td>VGG↓</td><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>No Protected</td><td>0.3488</td><td>0.3513</td><td>0.2806</td><td>0.4681</td><td>0.6580</td><td>0.4477</td><td>0.6788</td><td>0.7554</td><td>0.5847</td><td>0.4709</td><td>0.5838</td><td>0.4449</td></tr><tr><td>CLIP</td><td>0.2465</td><td>0.2696</td><td>0.2757</td><td>0.4634</td><td>0.5890</td><td>0.4584</td><td>0.5273</td><td>0.6043</td><td>0.4599</td><td>0.3458</td><td>0.4210</td><td>0.3397</td></tr><tr><td>Facenet</td><td>0.2593</td><td>0.2634</td><td>0.2810</td><td>0.4721</td><td>0.5743</td><td>0.4628</td><td>0.5345</td><td>0.5926</td><td>0.4658</td><td>0.3612</td><td>0.4188</td><td>0.3518</td></tr><tr><td>CLIP + Facenet</td><td>0.2056</td><td>0.2142</td><td>0.2202</td><td>0.4577</td><td>0.5606</td><td>0.4726</td><td>0.5032</td><td>0.5357</td><td>0.3932</td><td>0.3063</td><td>0.3577</td><td>0.2905</td></tr><tr><td>IDGuar (Ours)</td><td>0.0747</td><td>0.1284</td><td>0.2239</td><td>0.2792</td><td>0.3893</td><td>0.3404</td><td>0.3230</td><td>0.3265</td><td>0.2522</td><td>0.2159</td><td>0.2434</td><td>0.2288</td></tr><tr><td>G adv</td><td>0.1437</td><td>0.1477</td><td>0.2324</td><td>0.2796</td><td>0.4187</td><td>0.3432</td><td>0.3279</td><td>0.3405</td><td>0.2526</td><td>0.2427</td><td>0.2711</td><td>0.2451</td></tr><tr><td>G clean</td><td>0.1562</td><td>0.1974</td><td>0.2333</td><td>0.3046</td><td>0.4813</td><td>0.3410</td><td>0.3379</td><td>0.3472</td><td>0.2565</td><td>0.2423</td><td>0.2783</td><td>0.2358</td></tr><tr><td>IDGuar (Ours)</td><td>0.0747</td><td>0.1284</td><td>0.2239</td><td>0.2792</td><td>0.3893</td><td>0.3404</td><td>0.3230</td><td>0.3265</td><td>0.2522</td><td>0.2159</td><td>0.2434</td><td>0.2288</td></tr></table>

Table 10. Ablation study of different identity loss combinations and guidance directions. Evaluation is conducted using FaceQNet and MagFace to measure the perceptual quality of generated images.
<table><tr><td rowspan="2">Method</td><td colspan="2">Ip-Adapter</td><td colspan="2">Ip-Adapter Plus XL</td><td colspan="2">Blip-Diffusion</td><td colspan="2">InfiniteID</td></tr><tr><td></td><td></td><td></td><td>FaceQNet↓ MagFace ↓ FaceQNet↓ MagFace ↓ FaceQNet↓ MagFace↓</td><td></td><td></td><td></td><td>FaceQNet↓ MagFace ↓</td></tr><tr><td>No Protected</td><td>0.3339</td><td>21.3847</td><td>0.4564</td><td>19.3085</td><td>0.3305</td><td>21.8112</td><td>0.4946</td><td>19.0747</td></tr><tr><td>CLIP</td><td>0.3129</td><td>21.7832</td><td>0.4726</td><td>20.4175</td><td>0.3683</td><td>20.9487</td><td>0.5142</td><td>19.6254</td></tr><tr><td>Facenet</td><td>0.3185</td><td>21.6910</td><td>0.4791</td><td>20.5083</td><td>0.3724</td><td>20.0284</td><td>0.5076</td><td>19.4939</td></tr><tr><td>CLIP + Facenet</td><td>0.2714</td><td>21.8426</td><td>0.4689</td><td>19.5268</td><td>0.3776</td><td>20.4123</td><td>0.5128</td><td>20.1884</td></tr><tr><td>IDGuar (Ours)</td><td>0.2820</td><td>20.8712</td><td>0.4503</td><td>18.6099</td><td>0.3511</td><td>20.8498</td><td>0.4938</td><td>18.8031</td></tr><tr><td>G adv</td><td>0.2900</td><td>20.8653</td><td>0.4603</td><td>19.2300</td><td>0.3569</td><td>20.7490</td><td>0.4899</td><td>19.0814</td></tr><tr><td>G clean</td><td>0.2692</td><td>21.6448</td><td>0.4526</td><td>19.2488</td><td>0.3557</td><td>19.3413</td><td>0.4968</td><td>18.8645</td></tr><tr><td>IDGuar (Ours)</td><td>0.2820</td><td>20.8712</td><td>0.4503</td><td>18.6099</td><td>0.3511</td><td>20.8498</td><td>0.4938</td><td>18.8031</td></tr><tr><td rowspan="2">Method</td><td colspan="2">Photomaker</td><td colspan="2">Dreambooth</td><td colspan="2">InfiniteYou</td><td colspan="2">Mean</td></tr><tr><td>FaceQNet↓ MagFace ↓ FaceQNet↓ MagFace ↓</td><td></td><td></td><td></td><td>FaceQNet↓ MagFace ↓ FaceQNet↓ MagFace ↓</td><td></td><td></td><td></td></tr><tr><td>No Protected</td><td>0.4564</td><td>20.0541</td><td>0.3316</td><td>20.2766</td><td>0.4527</td><td>19.7298</td><td>0.4083</td><td>20.2342</td></tr><tr><td>CLIP</td><td>0.3012</td><td>20.2741</td><td>0.3310</td><td>19.5135</td><td>0.4729</td><td>19.1276</td><td>0.3962</td><td>20.2414</td></tr><tr><td>Facenet</td><td>0.3075</td><td>20.3417</td><td>0.3298</td><td>19.8824</td><td>0.4660</td><td>19.4059</td><td>0.3972</td><td>20.1930</td></tr><tr><td>CLIP + Facenet</td><td>0.2916</td><td>20.1042</td><td>0.2983</td><td>19.8631</td><td>0.4679</td><td>20.0893</td><td>0.3768</td><td>20.2895</td></tr><tr><td>IDGuar (Ours)</td><td>0.2733</td><td>19.8202</td><td>0.2203</td><td>20.2199</td><td>0.4517</td><td>19.8111</td><td>0.3604</td><td>19.8693</td></tr><tr><td>G adv</td><td>0.2813</td><td>19.7368</td><td>0.2754</td><td>19.9528</td><td>0.4536</td><td>20.2709</td><td>0.3725</td><td>19.9837</td></tr><tr><td>G clean</td><td>0.2789</td><td>19.8856</td><td>0.2270</td><td>20.5887</td><td>0.4519</td><td>20.3764</td><td>0.3617</td><td>19.9929</td></tr><tr><td>IDGuar (Ours)</td><td>0.2733</td><td>19.8202</td><td>0.2203</td><td>20.2199</td><td>0.4517</td><td>19.8111</td><td>0.3604</td><td>19.8693</td></tr></table>

Table 11. Robustness evaluation against common image post-processing operations, including JPEG compression, Gaussian blur, and Gaussian noise, and attacks targeting active defenses such as Impress.
<table><tr><td rowspan="2">Method</td><td colspan="3">Ip-Adapter</td><td colspan="3">Ip-Adapter Plus XL</td><td colspan="3">Blip-Diffusion</td><td colspan="3">InfiniteID</td></tr><tr><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face↓</td><td>VGG↓</td><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>No Protected</td><td>0.3369</td><td>0.4774</td><td>0.3308</td><td>0.4985</td><td>0.6899</td><td>0.5573</td><td>0.3305</td><td>0.3918</td><td>0.2848</td><td>0.6345</td><td>0.7627</td><td>0.6286</td></tr><tr><td>JPEG</td><td>0.1068</td><td>0.1900</td><td>0.1263</td><td>0.3209</td><td>0.3983</td><td>0.3006</td><td>0.2236</td><td>0.2196</td><td>0.2026</td><td>0.4286</td><td>0.5158</td><td>0.4062</td></tr><tr><td>Blur</td><td>0.1678</td><td>0.2948</td><td>0.1967</td><td>0.3317</td><td>0.4443</td><td>0.3520</td><td>0.2663</td><td>0.2663</td><td>0.2077</td><td>0.3854</td><td>0.4780</td><td>0.3734</td></tr><tr><td>Noise</td><td>0.1365</td><td>0.1794</td><td>0.1295</td><td>0.3078</td><td>0.3970</td><td>0.3113</td><td>0.1916</td><td>0.1910</td><td>0.1716</td><td>0.4482</td><td>0.5581</td><td>0.4391</td></tr><tr><td>Impress</td><td>0.1018</td><td>0.1696</td><td>0.1230</td><td>0.3184</td><td>0.3735</td><td>0.2956</td><td>0.2261</td><td>0.2653</td><td>0.2277</td><td>0.4478</td><td>0.5309</td><td>0.4074</td></tr><tr><td>IDGuar (Ours)</td><td>0.0673</td><td>-0.0226</td><td>0.0640</td><td>0.1296</td><td>0.1231</td><td>0.1260</td><td>0.2255</td><td>0.2376</td><td>0.1964</td><td>0.4121</td><td>0.5215</td><td>0.3990</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="3">Photomaker</td><td colspan="3">Dreambooth</td><td colspan="3">InfiniteYou</td><td colspan="3">Mean</td></tr><tr><td>Arc ↓</td><td>Face↓</td><td>VGG↓</td><td>Arc↓</td><td>Face↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face↓</td><td>VGG↓</td><td>Arc↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>No Protected</td><td>0.3488</td><td>0.3513</td><td>0.2806</td><td>0.4681</td><td>0.6580</td><td>0.4477</td><td>0.6788</td><td>0.7554</td><td>0.5847</td><td>0.4709</td><td>0.5838</td><td>0.4449</td></tr><tr><td>JPEG</td><td>0.1199</td><td>0.0754</td><td>0.2082</td><td>0.2723</td><td>0.3963</td><td>0.3267</td><td>0.3344</td><td>0.3728</td><td>0.2590</td><td>0.2581</td><td>0.3097</td><td>0.2614</td></tr><tr><td>Blur</td><td>0.0709</td><td>0.1591</td><td>0.2102</td><td>0.2899</td><td>0.5136</td><td>0.4110</td><td>0.3371</td><td>0.3510</td><td>0.2587</td><td>0.2642</td><td>0.3582</td><td>0.2871</td></tr><tr><td>Noise</td><td>0.1785</td><td>0.1618</td><td>0.1971</td><td>0.3418</td><td>0.5177</td><td>0.3844</td><td>0.3410</td><td>0.3625</td><td>0.2623</td><td>0.2780</td><td>0.3382</td><td>0.2708</td></tr><tr><td>Impress</td><td>0.1910</td><td>0.1494</td><td>0.2097</td><td>0.2516</td><td>0.4479</td><td>0.3386</td><td>0.3166</td><td>0.3233</td><td>0.2387</td><td>0.2648</td><td>0.3229</td><td>0.2629</td></tr><tr><td>IDGuar (Ours)</td><td>0.0747</td><td>0.1284</td><td>0.2239</td><td>0.2792</td><td>0.3893</td><td>0.3404</td><td>0.3230</td><td>0.3265</td><td>0.2522</td><td>0.2159</td><td>0.2434</td><td>0.2288</td></tr></table>

Table 12. Robustness evaluation against common image post-processing operations. Evaluation is conducted using FaceQNet and MagFace to measure the perceptual quality of generated images.
<table><tr><td rowspan="2">Method</td><td colspan="2">Ip-Adapter</td><td colspan="2">Ip-Adapter Plus XL</td><td colspan="2">Blip-Diffusion</td><td colspan="2">InfiniteID</td></tr><tr><td>FaceQNet↓ MagFace ↓</td><td></td><td>FaceQNet↓</td><td>MagFace ↓</td><td>FaceQNet↓ MagFace ↓</td><td></td><td>FaceQNet↓ MagFace ↓</td><td></td></tr><tr><td>No Protected</td><td>0.3339</td><td>21.3847</td><td>0.4564</td><td>19.3085</td><td>0.3305</td><td>21.8112</td><td>0.4946</td><td>19.0747</td></tr><tr><td>JPEG</td><td>0.2211</td><td>21.3679</td><td>0.4539</td><td>18.5163</td><td>0.3233</td><td>21.7881</td><td>0.4918</td><td>18.8227</td></tr><tr><td>Blur</td><td>0.3211</td><td>21.3901</td><td>0.4667</td><td>19.7750</td><td>0.2895</td><td>19.8607</td><td>0.5031</td><td>18.8229</td></tr><tr><td>Noise</td><td>0.2499</td><td>20.6822</td><td>0.4497</td><td>18.4623</td><td>0.3460</td><td>20.6286</td><td>0.5016</td><td>19.0230</td></tr><tr><td>Impress</td><td>0.2501</td><td>20.7604</td><td>0.4270</td><td>19.6015</td><td>0.3353</td><td>20.9642</td><td>0.4935</td><td>19.1813</td></tr><tr><td>IDGuar (Ours)</td><td>0.2820</td><td>20.8712</td><td>0.4503</td><td>18.5099</td><td>0.3511</td><td>20.8498</td><td>0.4908</td><td>18.9031</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="2">Photomaker</td><td colspan="2">Dreambooth</td><td colspan="2">InfiniteYou</td><td colspan="2">Mean</td></tr><tr><td>FaceQNet↓ MagFace ↓</td><td></td><td>FaceQNet↓ MagFace↓</td><td></td><td>FaceQNet↓</td><td>MagFace ↓</td><td></td><td>FaceQNet↓ MagFace ↓</td></tr><tr><td>No Protected</td><td>0.4564</td><td>20.0541</td><td>0.3316</td><td>20.2766</td><td>0.4527</td><td>19.7298</td><td>0.4080</td><td>20.2342</td></tr><tr><td>JPEG</td><td>0.2743</td><td>19.6998</td><td>0.2535</td><td>20.5918</td><td>0.4544</td><td>20.0803</td><td>0.3532</td><td>19.9809</td></tr><tr><td>Blur</td><td>0.2754</td><td>19.8549</td><td>0.2814</td><td>19.6256</td><td>0.4483</td><td>20.0016</td><td>0.3694</td><td>19.9044</td></tr><tr><td>Noise</td><td>0.2809</td><td>19.8303</td><td>0.3583</td><td>19.2192</td><td>0.4528</td><td>19.8165</td><td>0.3770</td><td>19.6660</td></tr><tr><td>Impress</td><td>0.2762</td><td>20.0743</td><td>0.2475</td><td>21.0550</td><td>0.4536</td><td>19.6948</td><td>0.3547</td><td>20.1902</td></tr><tr><td>IDGuar (Ours)</td><td>0.2733</td><td>19.8202</td><td>0.2203</td><td>20.2199</td><td>0.4517</td><td>19.8111</td><td>0.3604</td><td>19.8693</td></tr></table>

Table 13. Comparison of identity protection performance against baseline methods on the CelebA-HQ dataset. Evaluation is conducted using three widely adopted identity similarity metrics: ArcFace, FaceNet, and VGGFace.
<table><tr><td rowspan="2">Method</td><td colspan="3">Ip-Adapter</td><td colspan="3">Ip-Adapter Plus XL</td><td colspan="3">Blip-Diffusion</td><td colspan="3">InfiniteID</td></tr><tr><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>No Protected</td><td>0.4104</td><td>0.4714</td><td>0.3764</td><td>0.5706</td><td>0.7152</td><td>0.6062</td><td>0.2862</td><td>0.3495</td><td>0.3061</td><td>0.6352</td><td>0.6886</td><td>0.5642</td></tr><tr><td>Anti-DB</td><td>0.3061</td><td>0.3647</td><td>0.3193</td><td>0.4940</td><td>0.6321</td><td>0.5260</td><td>0.2257</td><td>0.3369</td><td>0.2894</td><td>0.5344</td><td>0.5946</td><td>0.5031</td></tr><tr><td>SimAC</td><td>0.2398</td><td>0.3390</td><td>0.3021</td><td>0.4049</td><td>0.5131</td><td>0.4339</td><td>0.2499</td><td>0.3240</td><td>0.2849</td><td>0.4483</td><td>0.5262</td><td>0.4455</td></tr><tr><td>AdvDM</td><td>0.2827</td><td>0.3907</td><td>0.3125</td><td>0.4158</td><td>0.5007 </td><td>0.4464</td><td>0.2222</td><td>0.3484</td><td>0.2680</td><td>0.5056</td><td>0.5435</td><td>0.4669</td></tr><tr><td>ACE</td><td>0.3462</td><td>0.4354</td><td>0.3505</td><td>0.5412 </td><td>0.6673</td><td>0.5834</td><td>0.2418</td><td>0.3422</td><td>0.2880</td><td>0.4813</td><td>0.5480</td><td>0.4839</td></tr><tr><td>IDProtector</td><td>0.2543</td><td>0.3071</td><td>0.2922</td><td>0.4363</td><td>0.5421</td><td>0.4486</td><td>0.1881</td><td>0.3003</td><td>0.2256</td><td>0.5065</td><td>0.6161</td><td>0.5611</td></tr><tr><td>IDGuar (Ours)</td><td>0.1003</td><td>0.0515</td><td>0.0940</td><td>0.2686</td><td>0.2905</td><td>0.2385</td><td>0.1794</td><td>0.3061</td><td>0.2431</td><td>0.3469</td><td>0.4198</td><td>0.3489</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="3">Photomaker</td><td colspan="3">Dreambooth</td><td colspan="3">InfiniteYou</td><td colspan="3">Mean</td></tr><tr><td>Arc ↓</td><td>Face↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td><td>Arc ↓</td><td>Face ↓</td><td>VGG↓</td></tr><tr><td>No Protected</td><td>0.6934</td><td>0.4219</td><td>0.2612</td><td>0.5421</td><td>0.6555</td><td>0.4335</td><td>0.6614</td><td>0.7485</td><td>0.5640</td><td>0.5428</td><td>0.5787</td><td>0.4445</td></tr><tr><td>Anti-DB</td><td>0.6504</td><td>0.4065</td><td>0.2560</td><td>0.1031</td><td>0.0666</td><td>0.0761</td><td>0.6035</td><td>0.6869</td><td>0.5312</td><td>0.4167</td><td>0.4412</td><td>0.3573</td></tr><tr><td>SimAC</td><td>0.6251</td><td>0.3564</td><td>0.2321</td><td>0.0831</td><td>0.0852</td><td>0.0665</td><td>0.5854</td><td>0.6846</td><td>0.4654</td><td>0.3766</td><td>0.4041</td><td>0.3186</td></tr><tr><td>AdvDM</td><td>0.6313</td><td>0.3718</td><td>0.2672</td><td>0.0868</td><td>0.0365</td><td>0.1100</td><td>0.5207</td><td>0.6293</td><td>0.4654</td><td>0.3807</td><td>0.4030</td><td>0.3338</td></tr><tr><td>ACE</td><td>0.6422</td><td>0.3880</td><td>0.2384</td><td>0.1401</td><td>0.2271</td><td>0.0944</td><td>0.6358</td><td>0.7283</td><td>0.5560</td><td>0.4327</td><td>0.4766</td><td>0.3707</td></tr><tr><td>IDProtector</td><td>0.6151</td><td>0.3013</td><td>0.2022</td><td>0.3298</td><td>0.3727</td><td>0.2578</td><td>0.5617</td><td>0.6609</td><td>0.4943</td><td>0.4131</td><td>0.4429</td><td>0.3545</td></tr><tr><td>IDGuar (Ours)</td><td>0.3320</td><td>0.2654</td><td>0.1996</td><td>0.0838</td><td>0.0320</td><td>0.0610</td><td>0.3331</td><td>0.3871</td><td>0.2428</td><td>0.2349</td><td>0.2503</td><td>0.2040</td></tr></table>

Table 14. Comparison of identity protection performance against baseline methods on the CelebA-HQ dataset. Evaluation is conducted using FaceQNet and MagFace to measure the perceptual quality of generated images.
<table><tr><td rowspan="2">Method</td><td colspan="2">Ip-Adapter</td><td colspan="2">Ip-Adapter Plus XL</td><td colspan="2">Blip-Diffusion</td><td colspan="2">InfiniteID</td></tr><tr><td></td><td>FaceQNet↓ MagFace ↓</td><td>FaceQNet↓</td><td>MagFace↓</td><td>FaceQNet↓ MagFace↓</td><td></td><td></td><td>FaceQNet ↓ MagFace ↓</td></tr><tr><td>No Protected</td><td>0.3031</td><td>20.6785</td><td>0.4490</td><td>18.8023</td><td>0.3344</td><td>20.9884</td><td>0.4527</td><td>19.6607</td></tr><tr><td>Anti-DB</td><td>0.2599</td><td>19.5696</td><td>0.4515</td><td>18.5281</td><td>0.3170</td><td>21.4418</td><td>0.4731</td><td>18.9686</td></tr><tr><td>SimAC</td><td>0.2670</td><td>19.9800</td><td>0.4493</td><td>19.2084</td><td>0.3433</td><td>19.4777</td><td>0.4695</td><td>19.0426</td></tr><tr><td>AdvDM</td><td>0.2842</td><td>19.9800</td><td>0.4505</td><td>19.2084</td><td>0.2941</td><td>20.0386</td><td>0.4645</td><td>19.2935</td></tr><tr><td>ACE</td><td>0.2928</td><td>20.0025</td><td>0.4540</td><td>19.3925</td><td>0.3363</td><td>20.7060</td><td>0.4539</td><td>19.1724</td></tr><tr><td>IDProtector</td><td>0.2945</td><td>20.0540</td><td>0.4544</td><td>19.0985</td><td>0.3039</td><td>20.4320</td><td>0.4204</td><td>19.7844</td></tr><tr><td>IDGuar (Ours)</td><td>0.2556</td><td>19.9783</td><td>0.4490</td><td>18.7536</td><td>0.3098</td><td>20.6180</td><td>0.4574</td><td>18.7834</td></tr><tr><td rowspan="2">Method</td><td colspan="2">Photomaker</td><td colspan="2">Dreambooth</td><td colspan="2">InfiniteYou</td><td colspan="2">Mean</td></tr><tr><td>FaceQNet↓ MagFace ↓</td><td></td><td>FaceQNet↓</td><td>MagFace ↓</td><td>FaceQNet↓ MagFace↓</td><td></td><td></td><td>FaceQNet↓ MagFace ↓</td></tr><tr><td>No Protected</td><td>0.2674</td><td>20.5900</td><td>0.3222</td><td>21.1998</td><td>0.4463</td><td>19.4956</td><td>0.3679</td><td>20.2022</td></tr><tr><td>Anti-DB</td><td>0.2693</td><td>20.4309</td><td>0.2802</td><td>21.8453</td><td>0.4451</td><td>19.6502</td><td>0.3566</td><td>20.0621</td></tr><tr><td>SimAC</td><td>0.2693</td><td>20.5074</td><td>0.2650</td><td>21.6122</td><td>0.4489</td><td>19.7678</td><td>0.3589</td><td>19.9423</td></tr><tr><td>AdvDM</td><td>0.2672</td><td>20.5074</td><td>0.3072</td><td>21.3755</td><td>0.4418</td><td>19.7678</td><td>0.3585</td><td>20.0245</td></tr><tr><td>ACE</td><td>0.2677</td><td>20.7290</td><td>0.2501</td><td>21.5659</td><td>0.4463</td><td>19.4555</td><td>0.3573</td><td>20.1463</td></tr><tr><td>IDProtector</td><td>0.2907</td><td>28.8511</td><td>0.3740</td><td>20.0474</td><td>0.4565</td><td>21.7575</td><td>0.3706</td><td>21.4321</td></tr><tr><td>IDGuar (Ours)</td><td>0.2653</td><td>20.5549</td><td>0.3073</td><td>21.2664</td><td>0.4352</td><td>19.3972</td><td>0.3542</td><td>19.9074</td></tr></table>

Table 15. ArcFace-only identity similarity on LFW. Method names are abbreviated for brevity.
<table><tr><td>Method</td><td></td><td>IP-Adapter↓ IP-Ada. + XL↓ 1</td><td>BLIP-D.↓</td><td>InfiniteID↓</td><td></td><td></td><td>PhotoM.↓ DreamB.↓ Infinite You↓ Mean↓</td><td></td></tr><tr><td>No Protected</td><td>0.3580</td><td>0.5210</td><td>0.3010</td><td>0.6150</td><td>0.3420</td><td>0.4550</td><td>0.6670</td><td>0.4656</td></tr><tr><td>Anti-DB</td><td>0.2550</td><td>0.4890</td><td>0.2490</td><td>0.5720</td><td>0.2300</td><td>0.3380</td><td>0.5980</td><td>0.3901</td></tr><tr><td>SimAC</td><td>0.2070</td><td>0.3620</td><td>0.2760</td><td>0.5490</td><td>0.2200</td><td>0.2870</td><td>0.5570</td><td>0.3511</td></tr><tr><td>AdvDM</td><td>0.2320</td><td>0.3480</td><td>0.2410</td><td>0.5030</td><td>0.2050</td><td>0.2540</td><td>0.5410</td><td>0.3320</td></tr><tr><td>ACE</td><td>0.2480</td><td>0.4620</td><td>0.2610</td><td>0.5330</td><td>0.2270</td><td>0.3210</td><td>0.6400</td><td>0.3846</td></tr><tr><td>IDProtector</td><td>0.2130</td><td>0.3310</td><td>0.2380</td><td>0.4920</td><td>0.2180</td><td>0.3160</td><td>0.5090</td><td>0.3310</td></tr><tr><td>IDGuar (Ours)</td><td>0.0790</td><td>0.1360</td><td>0.1980</td><td>0.4050</td><td>0.0910</td><td>0.2520</td><td>0.3180</td><td>0.2113</td></tr></table>

![](images/873c831d3c11f40140f8388356ddfd4129b5db29dbb16382ad264a141f4e8326.jpg)  
Figure 12. Robustness Evaluation Against Image Post-Processing and Defense Attacks.

![](images/83606286085c2c9307114fd7f99c732d42bbfba51f8505d9a5ba4c5d8085d0c1.jpg)  
Figure 13. Visualization of the ablation study on identity loss combinations and guidance directions.

![](images/6cd921cb6ef83e4fb40c670861e6db3a5bb1a2f2b7a56b09b300dc11aa49392e.jpg)  
Figure 14. Visualization of Identity Suppression Effectiveness on CelebA-HQ Dataset for IDGuardian and Comparison Methods.