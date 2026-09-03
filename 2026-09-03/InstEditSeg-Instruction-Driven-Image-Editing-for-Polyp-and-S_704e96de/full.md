# InstEditSeg: Instruction-Driven Image Editing for Polyp and Skin Lesion Segmentation

Ziquan Liu , Zhewei Zhu , and Xuyang Shi<sup>B</sup>

School of Information and Control Engineering, Southwest University of Science and Technology, Mianyang 621010, China

{ziquanliu, zhuzw}@mails.swust.edu.cn, xuyangshi@swust.edu.cn

Abstract. Accurate segmentation of polyps and skin lesions is pivotal for clinical diagnosis, yet existing methods struggle with low contrast, ambiguous boundaries, and cross-domain distribution discrepancies. Discriminative networks and most difusion-based segmentation approaches predict standalone binary masks, leaving the visual priors of large-scale pretrained generative models largely unexploited. We propose InstEditSeg, a unified generative framework that reformulates medical segmentation as an instruction-driven image editing problem. Instead of emitting a mask, the model renders a color-coded overlay on the original image, conditioned on a textual instruction, so that the edited output aligns with the natural image distribution learned by latent difusion models and mitigates the domain gap between natural and medical imagery. To recover fine anatomical structures, we introduce DINOv3 as an auxiliary visual encoder and a DINO Feature Guidance Block that builds a multi-scale feature pyramid. The pyramid is fused into the difusion U-Net by channel concatenation and zero-initialized convolution so that hierarchical discriminative priors can be injected without perturbing the pretrained weights. A dual-branch classifier-free guidance strategy requiring only two forward passes per denoising step reduces inference cost. On polyp and skin lesion benchmarks the framework achieves accuracy competitive with strong discriminative baselines, and it further demonstrates concrete advantages of the generative formulation: notably better cross-domain generalization on unseen data, more complete multi-lesion segmentation, instruction-conditioned task control, and sampling flexibility. We also analyze the strengths and limitations of the paradigm, including its color sensitivity and unsupported attribute-conditioned selection. Code is available at: https://github.com/wincharm001/InstEditSeg

## 1 Introduction

Colorectal cancer (CRC) is a leading cause of cancer-related mortality worldwide, and most cases originate from adenomatous polyps. Accurate polyp segmentation during colonoscopy is therefore crucial for early diagnosis, but it remains challenging due to low contrast, ambiguous boundaries, diverse morphology, and imaging artifacts such as uneven illumination and specular reflections, which hinder both accuracy and generalization.

Similarly, accurate segmentation of skin lesions in dermoscopic images is essential for the early diagnosis of melanoma [1], yet it faces comparable challenges, including fuzzy lesion borders, low contrast, and high inter-patient variability.

Deep learning has improved medical segmentation substantially. U-Net [2] became the de facto standard, while foundation models such as the Segment Anything Model (SAM) [3] show strong zeroshot capabilities. Nevertheless, SAM is trained on natural images and sufers substantial domain gaps on medical data, especially for small lesions and blurry boundaries.

Difusion probabilistic models [4] ofer an alternative paradigm. SegDif [5] casts segmentation as iterative denoising, and MedSegDif and MedSegDif-V2 [6, 7] extend this to medical tasks. These approaches benefit from generative capabilities but still predict task-specific binary masks, making them specialized difusion models rather than a unified generative framework.

Instruction-driven image editing [8] and unified visual generation models [9,10] show that detection, segmentation, and depth can be formulated as conditional generation. Using textual instructions, these models unify tasks within a single difusion network and can produce color-coded segmentation directly, bridging visual understanding and generation. Applying them to medical images, however, faces two limitations: the substantial distribution gap between natural and medical images, and the limited spatial guidance of text alone for fine anatomical structures.

![](images/f837b4d5a01846fbdefbd8e96dd72369ecf4e88693e7efec9e4edbe681bd00ac.jpg)  
Fig. 1. Conceptual illustration of the proposed framework. Traditional methods generate binary masks. Our method reformulates segmentation as an image editing task. It renders color-coded regions on the original image to align with generative priors and mitigate domain shift.

We propose InstEditSeg, a unified generative framework that reformulates segmentation as instructiondriven image editing, producing an edited image that keeps the original content and adds color-coded regions; this aligns the output with the natural image prior of latent difusion models and reduces the mismatch with medical images (Fig. 1). To compensate for the limited spatial guidance of denoising alone, we introduce DINOv3 as an auxiliary visual encoder and inject hierarchical features from a multi-scale pyramid into the U-Net, providing discriminative priors from local boundaries to global structure.

The editing formulation provides three advantages over discriminative baselines. The color-coded overlay keeps the original image content and aligns with the generative prior, so the model generalizes better to unseen settings, attaining the best Dice on the unseen PolypGen and ISIC2017 sets. The generative formulation also lets the model retain low-contrast secondary lesions that discriminative decoders drop, and a single architecture conditioned only on a text instruction covers multiple categories without task-specific heads or manual prompting. In-domain, the strongest discriminative baselines remain competitive, so the advantage is concentrated in generalization, segmentation completeness, and task control.

The main contributions of this work are summarized as follows:

– We propose InstEditSeg, a unified generative framework that recasts medical image segmentation as instruction-driven color-coded image editing, eliminating task-specific heads and aligning the output with latent difusion priors.

We develop an instruction-based data pipeline that converts segmentation annotations into unified image-text samples via textual instructions and randomized colors.

We design a DINOv3-guided multi-scale feature pyramid that injects hierarchical discriminative priors into the difusion U-Net, improving lesion localization and boundary delineation.

– Experiments on polyp and skin lesion benchmarks show competitive accuracy, notably better cross-domain generalization on unseen data, and concrete advantages of the editing formulation in multi-lesion completeness and instruction-conditioned task control.

## 2 Related Work

## 2.1 Deep Learning for Medical Image Segmentation

Deep learning has established the dominant paradigms for medical segmentation. U-Net [2] set the encoder-decoder standard with skip connections, which Attention U-Net [11] and U-Net++ [12] refined with attention and dense connections. Specialized architectures [13,14] further targeted polyp and skin lesion segmentation. Transformer-based designs now lead, with Polyp-PVT [15] using pyramid vision transformers and EMCAD [16] an eficient multi-scale attention decoder. Nevertheless, discriminative models remain sensitive to domain shifts and require many pixel-level annotations, so our work explores a generative paradigm that leverages pretrained difusion priors for better cross-domain generalization.

## 2.2 Foundation Models in Medical Imaging

Foundation models have reshaped visual understanding. SAM [3] achieved strong zero-shot segmentation on natural images, prompting medical adaptation such as MedSAM [17], Medical SAM Adapter [18], SAM-Med2D [19], Polyp-SAM [20], and MedSAM2 [21] based on SAM 2 [22]. However, systematic evaluations [23] show that SAM-family models are unstable on medical data, and their point or box prompts are cumbersome for low-contrast lesions with ambiguous boundaries. In contrast, we generate results directly from a textual instruction without manual prompting.

## 2.3 Difusion Models for Segmentation

Difusion probabilistic models [4] are now widely applied to segmentation. SegDif [5] introduced iterative denoising for segmentation, Wolleb et al. [24] obtained uncertainty ensembles, and MedSegDif [6] and MedSegDif-V2 [7] generalized this across medical tasks, while SDSeg [25] and TSLDseg [26] operate in the latent space of Stable Difusion. Beyond mask prediction, segmentation has been cast as generation through Pix2Seq-D [27], SegGen [28], LDMSeg [29], and VPD [30], and, in medical imaging, through TextDif [31], DifDGSS [32], and GenMed [33]; see [34] for a survey. Most of these methods still generate standalone binary masks with task-specific objectives, underutilizing the generative priors of latent difusion models. By contrast, we reformulate segmentation as instruction-driven editing that produces color-coded overlays, aligning with the learned distribution of latent difusion models [35] and reducing the domain gap.

## 2.4 Instruction-Driven Editing and Unified Vision Generation

Instruction-following editing has become a powerful interface for visual generation. InstructPix2Pix [8] conditions editing on instructions, and ControlNet [36] and T2I-Adapter [37] inject spatial controls into frozen difusion models. Unified visual generation models formulate perception as conditional generation, from Pix2Seq [38] to difusion generators such as Vision Banana [9] and SenseNova Vision [10], which solve segmentation, detection, and depth from textual instructions without task-specific heads, and the CLIP-driven universal model [39] shows similar potential in medical imaging. Directly applying these to medical images faces a distribution gap and limited spatial guidance from text alone. Our framework addresses both by injecting a DINOv3-guided multi-scale feature pyramid that complements the instruction-driven formulation.

## 3 Methodology

The overall architecture of the proposed InstEditSeg framework is illustrated in Fig. 2. Unlike conventional segmentation networks that directly predict pixel-wise semantic labels, our method reformulates medical image segmentation as an instruction-driven conditional image editing task. Given an input medical image together with a textual instruction, the difusion model generates an edited image in which the target region is highlighted using color-coded annotations, thereby achieving segmentation within a unified generative framework.

The framework consists of three components. A latent difusion model based on Stable Difusion [35] serves as the generative backbone for conditional generation in the latent space. A frozen DINOv3 [40] encoder extracts discriminative visual representations, and a multi-scale feature projection module transforms them into hierarchical visual priors spatially aligned with the decoding stages of the difusion U-Net. These priors are injected into the denoising process to jointly model global semantics, loca anatomy, and fine boundaries, which preserves the generative capability of latent difusion models while improving segmentation accuracy for medical images.

![](images/62d0777aece7ac97de615a4c824b08ee40555da1e3232e8edf1f57857cc382e7.jpg)  
Fig. 2. Overall architecture of the proposed framework. It integrates a Stable Difusion U-Net with a DINOv3 guidance branch. The input image is VAE-encoded and concatenated with the noisy latent as the conditioning input. Multi-scale DINOv3 features are projected via DFG blocks and injected into the U-Net. A lightweight auxiliary decoder supervises intermediate features with cross-entropy and Dice losses to ensure anatomical consistency.

## 3.1 Instruction-Based Segmentation Reformulation

Given an input image X and its corresponding ground-truth segmentation mask M, a random color mapping function $\varPhi ( \cdot )$ is first applied to obtain the color-coded segmentation mask C:

$$
\mathbf { C } = \varPhi ( \mathbf { M } ) .\tag{1}
$$

The target edited image Y is subsequently constructed by overlaying the color-coded mask C onto the original image X via a composite function f(·, ·):

$$
\mathbf { Y } = f ( \mathbf { X } , \mathbf { C } ) .\tag{2}
$$

Specifically, the overlay renders the target region with the assigned color while preserving the original content outside the target region, so that the underlying anatomical context is retained in the edited image. Meanwhile, a textual instruction T is randomly sampled from a predefined template set (e.g., “Segment the polyp region using red.”). Both the color assignments and the instruction templates are randomly sampled during training. This strategy provides multiple linguistic and visual representations for the same semantic category, thereby enhancing the model’s robustness in instruction comprehension. During training, the difusion model learns to reconstruct the edited image Y conditioned on the input image X and textual instruction T. During inference, only the original image along with the specified instruction is required to generate the final color-coded segmentation output.

## 3.2 Difusion-Based Segmentation Framework

We adopt Stable Difusion as our unified generative backbone. Given the target edited image Y, a frozen VAE encoder $\mathcal { E } ( \cdot )$ first maps it into the latent space:

$$
\begin{array} { r } { \mathbf { z } _ { y } = \mathcal { E } ( \mathbf { Y } ) . } \end{array}\tag{3}
$$

Following the standard forward difusion process, Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ is added at timestep t:

$$
\begin{array} { r } { \mathbf { z } _ { t } = \sqrt { \bar { \alpha } _ { t } } \mathbf { z } _ { y } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , } \end{array}\tag{4}
$$

where $\bar { \alpha } _ { t }$ denotes the cumulative product of the noise schedule coeficients up to step $t ,$ and I represents the identity matrix. Similarly, the input image X is encoded into its latent representation:

$$
\mathbf { z } _ { I } = { \mathcal { E } } ( \mathbf { X } ) .\tag{5}
$$

Following the image editing paradigm $[ 8 ] ,$ , the latent code $\mathbf { z } _ { I }$ of the input image is concatenated with the noisy latent $\mathbf { z } _ { t }$ along the channel dimension, and this concatenated pair serves as the image conditioning input of the U-Net. The conditional difusion U-Net parameterizes the noise prediction function $\epsilon _ { \theta } .$ Taking the noisy latent $\mathbf { z } _ { t } ,$ image condition latent $\mathbf { z } _ { I } .$ , timestep $t ,$ text prompt embedding $\mathbf { c } _ { T }$ (encoded from instruction $T )$ , and DINO-guided visual priors $\mathcal { P }$ as inputs, the model predicts the added noise ˆϵ:

$$
\hat { \mathbf { \epsilon } } = \epsilon _ { \theta } ( \mathbf { z } _ { t } , \mathbf { z } _ { I } , t , \mathbf { c } _ { T } , \mathcal { P } ) .\tag{6}
$$

## 3.3 DINO-Guided Multi-Scale Feature Projection

Although difusion models possess strong generative capabilities, relying solely on standard image and text conditions is often insuficient for accurately capturing subtle anatomical structures and ambiguous lesion boundaries in medical images. To address this limitation, we introduce a frozen DINOv3 encoder as an auxiliary visual backbone to supply discriminative semantic priors throughout the denoising process. Let $\mathbf { F } _ { D } \in \mathbb { R } ^ { N \times D }$ denote the sequence of patch token representations extracted by DINOv3, where N is the number of tokens and D is the feature dimension. Because these vision transformer tokens fundamentally difer from the spatial feature maps utilized by the difusion U-Net, they cannot be directly injected into the denoising network. To bridge this representation gap, we design a DINO Feature Guidance (DFG) Block. This module progressively transforms transformer tokens into hierarchical convolutional feature maps via token projection, spatial reconstruction, and multi-scale feature transformation, with its detailed architecture illustrated in Fig. 2. The resulting visual priors are expressed as:

$$
\mathcal { P } = \{ { \bf P } _ { 1 } , { \bf P } _ { 2 } , { \bf P } _ { 3 } , { \bf P } _ { 4 } \} ,\tag{7}
$$

where the spatial resolutions of $\mathbf { P } _ { i } ~ ( i \in \{ 1 , 2 , 3 , 4 \} )$ match the four decoding stages of the difusion U-Net, establishing spatially aligned visual priors for multi-scale feature fusion; the active DINOv3 feature layers are grouped into four levels corresponding to these stages.

## 3.4 DINO Feature Injection

After extracting the multi-scale visual priors ${ \mathcal { P } } _ { : }$ they are injected into their corresponding stages within the difusion U-Net. Existing conditional difusion models $( \mathrm { e . g . }$ , ControlNet [36], T2I-Adapter [37]) incorporate external conditions via element-wise addition, which implicitly assumes that the conditioning and network features share an aligned embedding space. In our framework, difusion features encode generative representations for noise prediction, whereas DINOv3 provides discriminative semantic representations, so these heterogeneous features exhibit distinct statistical properties. To address this domain mismatch, we adopt a Concatenation and Zero-Convolution fusion strategy. For the i-th feature level, the U-Net feature map $\mathbf { U } _ { i }$ and the corresponding visual prior $\mathbf { P } _ { i }$ are concatenated along the channel dimension:

$$
\begin{array} { r } { \tilde { \mathbf { U } } _ { i } = \mathrm { C o n c a t } ( \mathbf { U } _ { i } , \mathbf { P } _ { i } ) . } \end{array}\tag{8}
$$

The concatenated representation is then processed by a zero-initialized convolutional layer $\mathrm { C o n v } _ { \mathrm { z e r o } } ( \cdot ) { : }$

$$
\begin{array} { r } { \hat { \mathbf { U } } _ { i } = \mathrm { C o n v } _ { \mathrm { z e r o } } ( \tilde { \mathbf { U } } _ { i } ) . } \end{array}\tag{9}
$$

Compared with additive fusion, concatenation preserves the full information from both branches, enabling the network to learn complex non-linear interactions automatically. The zero initialization of the convolution ensures that the initial forward pass matches the pretrained difusion model, and because the fused output is added back to the U-Net feature through a residual connection $( \mathbf { U } _ { i } \gets$ $\mathbf { U } _ { i } + \hat { \mathbf { U } } _ { i } )$ , the pretrained behavior is preserved until the guidance branch takes efect.

## 3.5 Training Objective

The proposed framework is optimized end-to-end using a joint objective. The primary noise prediction loss $\mathcal { L } _ { \mathrm { n o i s e } }$ is defined as:

$$
\mathcal { L } _ { \mathrm { n o i s e } } = \mathbb { E } _ { \mathbf { z } _ { y } , \epsilon , t } \left[ | \epsilon - \epsilon _ { \theta } ( \mathbf { z } _ { t } , \mathbf { z } _ { I } , t , \mathbf { c } _ { T } , \mathcal { P } ) | _ { 2 } ^ { 2 } \right] .\tag{10}
$$

To further enforce anatomical structural consistency, a lightweight auxiliary decoder (shown in Fig. 2) is attached to the visual guidance branch. We apply a combination of Dice loss $( \mathcal { L } _ { \mathrm { D i c e } } )$ and Cross-Entropy loss $( \mathcal { L } _ { \mathrm { C E } } )$ to supervise the intermediate features:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { D i c e } } + \mathcal { L } _ { \mathrm { C E } } . } \end{array}\tag{11}
$$

The overall optimization objective is given by:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { n o i s e } } + \lambda \mathcal { L } _ { \mathrm { s e g } } , } \end{array}\tag{12}
$$

where $\lambda > 0$ is a balancing hyperparameter that weights the auxiliary segmentation supervision relative to the difusion denoising loss.

## 3.6 Inference Strategy

![](images/536fb74e907fd6901cc67faa4ac14d368e37e72911bf521fa03927cea2037cc4.jpg)  
Fig. 3. Inference sampling pipeline. A dual-branch CFG strategy is adopted to reduce latency. The U-Net takes the instruction text and an empty text to predict the conditional and unconditional noises. The guided noise is computed as Eq. (15). DINO feature injection is omitted from the figure for the sake of brevity.

During inference, classifier-free guidance (CFG) is used to control conditional sampling. Unlike general multi-modal generation models that require three forward passes per step (unconditioned, textconditioned, and joint image-text conditioned), we propose a simplified dual-branch guidance strategy tailored specifically for medical image segmentation. Medical image segmentation is driven by visua anatomy, so spatial boundaries are determined by the input image while the text instruction only selects the target category; the image condition alone is therefore a strong structural constraint, and a separate text-only branch adds computation without benefit and can induce semantic drift. Accordingly, our method requires only two U-Net forward evaluations per denoising step: Noise prediction under imageonly conditioning (with an empty text embedding ∅):

$$
\begin{array} { r } { \epsilon _ { I } = \epsilon _ { \theta } ( \mathbf { z } _ { t } , \mathbf { z } _ { I } , t , \boldsymbol { \mathcal { O } } , \mathcal { P } ) . } \end{array}\tag{13}
$$

Noise prediction under joint image-text conditioning:

$$
\begin{array} { r } { \epsilon _ { I , T } = \epsilon _ { \theta } ( \mathbf { z } _ { t } , \mathbf { z } _ { I } , t , \mathbf { c } _ { T } , \mathcal { P } ) . } \end{array}\tag{14}
$$

The final guided noise prediction ˆϵ is formulated as:

$$
\hat { \pmb { \epsilon } } = \pmb { \epsilon } _ { I } + s ( \pmb { \epsilon } _ { I , T } - \pmb { \epsilon } _ { I } ) ,\tag{15}
$$

where s denotes the guidance scale factor for text control. By restricting text guidance strictly to task specification while relying on image features for spatial alignment, this dual-branch scheme preserves high image fidelity and reduces inference latency compared to standard three-branch CFG sampling.

## 4 Experiments

## 4.1 Experimental Setup

Datasets To comprehensively validate the efectiveness of the proposed method, we conducted experiments on two medical image segmentation tasks: polyp segmentation and skin lesion segmentation. Polyp segmentation evaluates the model’s capability for target localization and boundary delineation within complex endoscopic scenes, while skin lesion segmentation verifies the generalization ability of the unified generative framework across a diferent medical imaging modality.

Table 1. Statistics of the datasets used in the experiments. “Unseen Test” indicates that the dataset is not involved in any training stage.
<table><tr><td>Dataset Task</td><td>Train Test Unseen Test</td></tr><tr><td>Kvasir-SEG Polyp</td><td>900 100 X</td></tr><tr><td>CVC-ClinicDB Polyp</td><td>550 62 X</td></tr><tr><td>CVC-ColonDB Polyp</td><td>341 38 X</td></tr><tr><td>ETIS-LaribPolypDB Polyp</td><td>176 20 X</td></tr><tr><td>PolypGen Polyp</td><td>1411 V</td></tr><tr><td>ISIC2016 Skin</td><td>900 379 X</td></tr><tr><td>ISIC2017 Skin</td><td>600 V</td></tr></table>

Polyp Segmentation. We utilized four public datasets: Kvasir-SEG [41], CVC-ClinicDB [42], CVC-ColonDB [43], and ETIS-LaribPolypDB [44]; Table 1 summarizes their sample sizes. Following common practice, the training sets of these four datasets are combined, with independent evaluations on each test set. To rigorously assess cross-dataset generalization, PolypGen [45] is employed as an unseen test set: it was excluded from training, and its images originate from diferent medical institutions and endoscopic equipment, presenting significant domain shifts that provide an objective benchmark for generalization.

Table 2. Quantitative comparison of polyp segmentation performance. Bold denotes the best result among all compared methods.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Prompt</td><td colspan="2">Kvasir-SEG</td><td colspan="2">ETIS</td><td colspan="2">|CVC-ClinicDB|CVC-ColonDB|PolypGen (Unseen)</td><td colspan="2"></td><td></td></tr><tr><td>Dice</td><td>IoU Dice</td><td>IoU</td><td>Dice</td><td>IoU</td><td>Dice</td><td>IoU</td><td>Dice</td><td>IoU</td></tr><tr><td>U-Net</td><td>None</td><td>81.99 73.81</td><td>69.64</td><td>61.84</td><td>82.03</td><td>74.92</td><td>78.76</td><td>63.04</td><td>69.57</td><td>56.21</td></tr><tr><td>Polyp-PVT</td><td>None</td><td>91.86 86.52</td><td>88.70</td><td>79.33</td><td>93.36</td><td>88.59</td><td>88.31</td><td>71.25</td><td>80.84</td><td>75.88</td></tr><tr><td>EMCAD</td><td>None</td><td>93.74 89.21</td><td>88.46</td><td>82.62</td><td>93.22</td><td>88.04</td><td>89.48</td><td>83.63</td><td>80.78</td><td>74.25</td></tr><tr><td>SAM</td><td>bbox</td><td>89.25 82.54</td><td>77.01</td><td>68.88</td><td>89.72</td><td>83.57</td><td>75.32</td><td>66.69</td><td>80.94</td><td>72.20</td></tr><tr><td>MedSAM</td><td>bbox</td><td>92.75 88.14</td><td></td><td>|92.61 87.28|</td><td>92.31</td><td>86.24</td><td>87.78</td><td>79.01</td><td>83.51</td><td>74.59</td></tr><tr><td>MedSegDiff</td><td>None</td><td>87.98</td><td>80.19 84.81</td><td>75.38</td><td>87.26</td><td>79.95</td><td>87.05</td><td>79.78</td><td>75.25</td><td>66.17</td></tr><tr><td>MedSegDiff-V2</td><td>None</td><td>92.12 87.43</td><td>89.36</td><td>81.35</td><td>92.02</td><td>86.79</td><td>91.22</td><td>84.41</td><td>80.74</td><td>75.82</td></tr><tr><td>SDSeg</td><td>None</td><td>90.67 87.56</td><td>90.86</td><td>84.51</td><td>89.73</td><td>81.57</td><td>90.17</td><td>85.44</td><td>82.46</td><td>76.03</td></tr><tr><td>TSLDseg</td><td>None</td><td>91.98 87.01</td><td>91.44</td><td>85.27</td><td>90.25</td><td>85.09</td><td>91.82</td><td>86.76</td><td>79.56</td><td>71.64</td></tr><tr><td>InstEditSeg</td><td>Text</td><td>|92.1087.05</td><td>92.01</td><td>85.93</td><td>92.83</td><td>87.32</td><td>|92.95</td><td>87.46</td><td>|83.92</td><td>77.50</td></tr></table>

their public implementations and oficial recommended configurations. To ensure a strictly controlled comparison, the discriminative baseline EMCAD was retrained from scratch on the same training data (the combined polyp training sets and the ISIC2016 training set, respectively) following its oficial implementation, including multi-scale test-time augmentation, and all of its reported numbers are from this controlled re-run.

Implementation Details All experiments were implemented in PyTorch and trained on a single NVIDIA A100 (40 GB) GPU.

We adopted Stable Difusion as the generative backbone. All images were resized to $5 1 2 \times 5 1 2$ . The VAE encoder and the DINOv3 backbone remained frozen throughout training, while only the difusion U-Net and the CLIP Text Encoder were fine-tuned. The initial learning rate was set to $1 \times 1 0 ^ { - 4 }$ for the U-Net and $5 \times 1 0 ^ { - 7 }$ for the CLIP Text Encoder. We employed the AdamW optimizer with a batch size of 8 and a weight decay of $1 \times 1 0 ^ { - 3 }$ . The total training steps were set to 20,000. The DINOv3 ViT-S/16 model was adopted as the default visual encoder, with 6 active feature layers used to construct the multi-scale guidance pyramid, and the auxiliary loss weight λ was set to 0.3.

The model was supervised by the constructed image editing target, jointly optimizing the difusion noise prediction and auxiliary segmentation losses. Unless otherwise specified, all experiments adopted identical training configurations.

During inference, we utilized the DDIM scheduler [47] with 25 sampling steps and the proposed dual-branch classifier-free guidance (CFG) strategy described in Section 3.6, with the guidance scale s set to 7.5. The Dice Similarity Coeficient (Dice) and Intersection over Union (IoU) were adopted as the primary evaluation metrics.

## 4.2 Comparison with State-of-the-art Methods

We compared the proposed method against several state-of-the-art segmentation approaches, including traditional CNN-based methods (U-Net [2]), transformer-based methods (Polyp-PVT [15], EM-CAD [16]), foundation models (SAM [3], MedSAM [17]), and difusion-based methods (MedSegDif [6], MedSegDif-V2 [7], SDSeg [25], TSLDseg [26]). Table 2 reports the polyp segmentation results. Among the baselines, the discriminative EMCAD attains the highest Dice and IoU on Kvasir-SEG and CVC-ClinicDB (93.74% and 93.22%, respectively); the foundation model MedSAM is strongest on ETIS, with a Dice of 92.61%; and the best difusion baseline, TSLDseg, reaches a Dice of 91.82% on CVC-ColonDB.

Our proposed method achieves the best overall results on CVC-ColonDB, with a Dice of 92.95% and an IoU of 87.46%. On ETIS it attains a Dice of 92.01%, ranking second only behind MedSAM, and on Kvasir-SEG and CVC-ClinicDB it delivers highly competitive results, with 92.10% and 92.83% Dice. Most importantly, on the unseen PolypGen dataset it achieves the best Dice of 83.92% and IoU of 77.50%, outperforming all compared methods and demonstrating strong cross-domain generalization. Paired Wilcoxon tests against the strongest baseline confirm that this advantage is significant on CVC-ColonDB with $p < 0 . 0 5$ and on PolypGen with $p < 0 . 0 0 1$ , that it is statistically tied $( p > 0 . 0 5 )$ on ETIS and CVC-ClinicDB, and that EMCAD remains ahead on Kvasir-SEG with $p < 0 . 0 5$

Table 3. Quantitative comparison of skin lesion segmentation performance. Bold denotes the best result among all compared methods.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>ISIC2016Dice IoU</td><td rowspan=1 colspan=1>|ISIC2017 (Unseen)Dice     IoU</td></tr><tr><td rowspan=1 colspan=1>U-Net</td><td rowspan=1 colspan=1>85.4575.52</td><td rowspan=1 colspan=1>77.35    68.97</td></tr><tr><td rowspan=1 colspan=1>Polyp-PVT</td><td rowspan=1 colspan=1>88.75 81.10</td><td rowspan=1 colspan=1>81.76    72.58</td></tr><tr><td rowspan=1 colspan=1>EMCAD</td><td rowspan=1 colspan=1>92.6587.40</td><td rowspan=1 colspan=1>82.39    74.88</td></tr><tr><td rowspan=1 colspan=1>SAM</td><td rowspan=1 colspan=1>88.5080.86</td><td rowspan=1 colspan=1>78.08    69.12</td></tr><tr><td rowspan=1 colspan=1>MedSAM</td><td rowspan=1 colspan=1>90.2482.74</td><td rowspan=1 colspan=1>82.83   76.55</td></tr><tr><td rowspan=1 colspan=1>MedSegDiff</td><td rowspan=1 colspan=1>88.1375.17</td><td rowspan=1 colspan=1>72.34   67.24</td></tr><tr><td rowspan=1 colspan=1>MedSegDiff-V2</td><td rowspan=1 colspan=1>92.3587.06</td><td rowspan=1 colspan=1>80.87   74.41</td></tr><tr><td rowspan=1 colspan=1>SDSeg</td><td rowspan=1 colspan=1>92.73 88.68</td><td rowspan=2 colspan=1>76.32    68.0380.51    71.69</td></tr><tr><td rowspan=1 colspan=1>TSLDseg</td><td rowspan=1 colspan=1>92.0888.22</td></tr><tr><td rowspan=1 colspan=1>InstEditSeg</td><td rowspan=1 colspan=1>92.5886.91</td><td rowspan=1 colspan=1>83.14   75.62</td></tr></table>

![](images/0de0928e89919251e1d50d987ac8c6b55fb2448e025caf5d2bbd6abac52348cd.jpg)  
Fig. 4. Qualitative comparison of representative methods on challenging polyp and skin lesion cases. Our method delineates lesion boundaries more precisely in low-contrast, irregular, or ambiguous cases, where other methods often produce fragmented predictions.

Table 3 presents the skin lesion segmentation results. On the in-domain ISIC2016 test set, the difusion-based SDSeg leads with a Dice of 92.73% and an IoU of 88.68%, closely followed by the discriminative EMCAD (92.65%) and our method (92.58%), both above the remaining difusion baselines and the foundation model MedSAM. This confirms that the instruction-driven editing paradigm captures lesion boundaries in dermoscopic images while remaining competitive with the best discriminative baseline.

On the unseen ISIC2017 dataset, all methods degrade under domain shift. Our framework attains the best Dice of 83.14% and a competitive IoU of 75.62%, outperforming the strongest discriminative baseline EMCAD at 82.39% and the discriminative transformer Polyp-PVT at 81.76%, with a significant paired Wilcoxon lead over EMCAD at $p < 0 . 0 5$ , while MedSAM holds the highest IoU of 76.55%. Our framework also degrades less than EMCAD from the in-domain result. These results show that the instruction-driven editing paradigm is robust to domain shift on an unseen skin set, even though the strongest discriminative baselines remain competitive.

Fig. 4 provides qualitative comparisons on challenging cases from both datasets. For polyp segmentation, our method delineates lesion boundaries with higher precision in cases with low contrast, irregular shapes, or small flat polyps, where other approaches often produce fragmented predictions or miss subtle regions. In skin lesion segmentation, the proposed framework accurately captures lesions with fuzzy borders, while baseline difusion models tend to over-segment or include background artifacts. These visual results corroborate the quantitative improvements in Tables 2 and 3, highlighting the benefit of DINOv3 discriminative priors and the instruction-driven editing formulation.

Table 4. Segmentation performance on the 119 PolypGen test images containing multiple polyps (all lesions evaluated).
<table><tr><td>Method</td><td>Dice IoU</td></tr><tr><td>EMCAD</td><td>63.18 53.00</td></tr><tr><td>InstEditSeg</td><td>75.93 64.93</td></tr></table>

## 4.3 Advantages over Discriminative Segmentation

To answer why a generative editing formulation is preferable to conventional discriminative segmentation, we compare the proposed method against the strongest discriminative baseline, EMCAD, on the axes where the two paradigms materially difer. We focus on cross-domain generalization, segmentation completeness on multi-lesion scenes, instruction-conditioned task control, and sampling flexibility, and we report both the strengths and the regimes in which discriminative methods remain competitive.

Cross-Domain Generalization The clearest advantage of the editing formulation is robustness to domain shift. On the unseen PolypGen set, the framework attains the best Dice and IoU among all compared methods, 83.92% and 77.50%, exceeding both the discriminative EMCAD and the medical foundation model MedSAM, and on the unseen ISIC2017 set it attains the best Dice of 83.14%, outperforming EMCAD and Polyp-PVT and led in IoU only by MedSAM. This generalization advantage is consistent with the design-level ablations, where the overlay representation and the DINOv3 guidance branch provide the largest gains, and it is most pronounced on the unseen PolypGen and ISIC2017 sets. We attribute the behavior to two properties of the editing formulation. First, rendering a color-coded overlay on the original image keeps the generated output close to the natural image distribution, so the pretrained prior remains efective on medical inputs. Second, the DINOv3 feature injection anchors the generator to genuine anatomical boundaries, which reduces the sensitivity of the output to a specific imaging protocol.

Multi-Lesion Completeness Polyps often appear in multiples, a scenario where small discriminative decoders tend to miss secondary lesions. To quantify this, we select the 119 PolypGen test images whose ground truth contains at least two lesion components (second-largest component ≥ 200 px) and evaluate both models on all lesions. As reported in Table 4, InstEditSeg attains a Dice of 75.93%, outperforming EMCAD by 12.75 Dice points; qualitative examples are shown in Fig. 5. We attribute this primarily to the position-agnostic formulation of our editing task. Discriminative decoders learn an implicit prior about where a lesion is likely to appear, biasing them toward the dominant lesion and suppressing secondary or atypically placed ones. Our framework never anchors the model to a location: the editing target is defined only by the color-coded region and the instruction, and the generator is supervised over the whole image without any position term. It therefore reasons over all candidate regions and preserves multiple lesions regardless of position. DINOv3 guidance helps with low-contrast boundaries, but the absence of an explicit location prior is the principal reason secondary lesions are retained. This position-agnostic behavior is two-sided, since the model also does not parse spatial or size modifiers, such as “segment only the largest polyp”, or “segment the rightmost lesion”.

Unified Instruction-Conditioned Task Control A discriminative network is specialized to a single task and output space, whereas InstEditSeg is a single generative architecture that produces a segmentation whenever a textual instruction is provided. The same architecture, without any taskspecific head, handles both polyp and skin lesion segmentation, with the instruction selecting the target category; each task is trained separately on its own data. This difers from promptable foundation models, which need a point or box prompt, and from discriminative networks, which require a dedicated decoder per task. The current model does not yet parse spatial or size modifiers, and we treat attributeconditioned selection as open future work.

![](images/73ff70dc1e555e761658148aa6669fd8618c3243a3d23ced11acbd93d43301f9.jpg)  
Fig. 5. Multi-lesion segmentation on PolypGen. Columns: input image, ground truth, InstEditSeg prediction, EMCAD prediction. EMCAD tends to drop secondary lesions, whereas the proposed method recovers multiple lesions.

Sampling Flexibility and CFG Analysis Difusion sampling is typically the main computational cost of generative segmentation. We analyze the trade-of between DDIM steps and accuracy on the Kvasir-SEG test set. As shown in Table 5 and Fig. 6, five DDIM steps attain a Dice of 91.78% at 711 ms per image. For comparison, the discriminative baseline EMCAD reaches a Dice of 93.74% at only 218 ms per image, so the generative pipeline is inherently more costly at inference. Furthermore, we compare inference variants at 25 steps: the proposed dual-branch CFG attains 92.10% Dice at 3.58 s, a no-CFG variant that performs a single image-and-text conditioned forward per step attains 89.46% at 1.84 s, and an image-only variant with empty text collapses to 72.75% at 1.80 s. The no-CFG variant therefore cuts latency by roughly half at a cost of about 2.6 Dice points relative to dual-branch CFG, whereas the image-only variant collapses, which confirms that textual instructions are indispensable for task specification. These results substantiate the eficiency claim of the dual-branch scheme and indicate that, in latency-critical deployments, the step count can be reduced to five with a small accuracy loss, although dropping CFG altogether incurs a more noticeable drop. Increasing the step count to 50 yields 91.94%, slightly below the 25-step result, so 25 steps was adopted as the default.

## 4.4 Ablation Studies

To verify the contribution of each component, we conducted ablation studies on the Kvasir-SEG and ISIC2016 datasets. The results are summarized in Table 6.

Efectiveness of Components. The baseline Stable Difusion model fine-tuned with the editing target attains 85.47% Dice on Kvasir-SEG. Injecting the DINO Feature Guidance (DFG) block alone raises it to 87.73%, validating that multi-scale discriminative priors are essential for recovering fine lesion structures. Further incorporating the auxiliary segmentation loss yields the full model performance of 92.10% Dice, a substantial improvement of +4.37% over the DFG-only variant. This progressive enhancement demonstrates that both discriminative visual priors and explicit structural supervision are critical in a difusion framework.

Output Representation. A core design choice of InstEditSeg is predicting a color-coded overlay rendered on the original image rather than a standalone binary mask. We ablate this choice by training two variants under an otherwise identical protocol: (i) a grayscale binary mask output ([0, 1]), whose instruction removes all color specification, and (ii) a color-coded mask on a pure black background, which keeps the color instruction but discards the image context. The quantitative comparison is reported in Table 7.

Table 5. Inference eficiency on Kvasir-SEG (100 test images, A100). Latency includes the full generation pipeline per image.
<table><tr><td>Configuration</td><td colspan="2">Dice (%) Latency (ms)</td></tr><tr><td>EMCAD</td><td>93.74</td><td>218</td></tr><tr><td>DDIM 5 steps (dual CFG)</td><td>91.78</td><td>711</td></tr><tr><td>DDIM 10 steps (dual CFG)</td><td>92.02</td><td>1428</td></tr><tr><td>DDIM 25 steps (dual CFG)</td><td>92.10</td><td>3575</td></tr><tr><td>DDIM 50 steps (dual CFG)</td><td>91.94</td><td>6461</td></tr><tr><td>25 steps, no CFG (text+image)</td><td>89.46</td><td>1842</td></tr><tr><td>25 steps, image only (empty text)</td><td>72.75</td><td>1801</td></tr></table>

![](images/2a884d123e6963e337cf4802cf3339c22a36b0c118c0d6259a0ed236ef766223.jpg)  
Fig. 6. Dice accuracy and per-image latency as functions of the number of DDIM sampling steps on Kvasir-SEG.

As shown in Table 7, the overlay representation consistently outperforms both alternatives. The binary-mask variant attains only 82.35% Dice on Kvasir-SEG and 52.60% on ISIC2016, with a paired Wilcoxon test giving $p < 0 . 0 0 1$ against the overlay. This indicates that direct [0, 1] mask prediction discards the color-coded structure that latent difusion models are naturally biased to generate, and that color words in the instruction serve as an additional task-anchoring signal. The black-background variant also degrades substantially even after isolating color-mapping artifacts, reaching 65.31% corrected Dice on Kvasir-SEG and 53.40% on ISIC2016, which shows that rendering the mask over the original content provides essential visual context for the denoising process, as illustrated in Fig. 7.

Fusion Strategy. We compare the proposed concatenation-based fusion against the commonly used element-wise addition. As reported in Table 8, with the DFG guidance branch and the auxiliary loss kept identical, concatenation outperforms addition by 0.79% in Dice on Kvasir-SEG and by 0.81% on ISIC2016, suggesting that preserving the full feature information from heterogeneous sources allows more efective non-linear interactions, whereas additive fusion forces an overly restrictive alignment between generative and discriminative feature spaces. As shown in Fig. 8, the training loss curve of concatenation fusion remains consistently lower throughout optimization, demonstrating more efective convergence.

Alternative Auxiliary Backbones. To verify that the advantage of the DFG branch stems from the DINOv3 representation itself rather than the presence of an auxiliary branch per se, we replace the DINOv3 encoder with alternative visual backbones under an otherwise identical training protocol (same pyramid adapters and auxiliary segmentation head; the last 6 blocks of each alternative backbone are unfrozen, except for ResNet-50, which is fully frozen): a fully supervised ImageNet ResNet-50, a MedSAM image encoder (ViT-B), a CLIP ViT-B/16 image encoder, and a self-supervised DINOv2 gain from 6 to 12 layers is small relative to the added computational cost, so we retain 6 layers as the default. Regarding the loss weight, the model exhibits stable performance for λ in {0.1, 0.3, 0.5, 0.7, 0.9}, with only marginal degradation at extreme values, indicating that the framework is not highly sensitive to this parameter. Considering the trade-of between accuracy and computational cost, we adopt $\lambda = 0 . 3$ with 6 active layers as the default configuration for the main experiments and all ablation studies, demonstrating that a moderate balance between the difusion denoising objective and the auxiliary segmentation supervision yields good convergence.

![](images/2e8863b97b284f801c1e066db65fddc0e629ffdb6a13e3d8b51a40b925c4f00b.jpg)  
Fig. 7. Output-representation comparison on two Kvasir-SEG samples. Rows: binary [0, 1] mask, color mask on black background, and the proposed overlay. Columns: input image, editing target, model prediction.

![](images/915f9ff849f2779c5c70516218a19d38c1b5ef034ff2694ec6c830db55022ab5.jpg)  
Fig. 8. Training loss curves of the two fusion strategies.

Table 6. Ablation study on key components. “DFG” denotes the DINO Feature Guidance Block. “Aux. Loss” represents the auxiliary segmentation loss.
<table><tr><td rowspan=1 colspan=1>Baseline DFG Aux. Loss</td><td rowspan=1 colspan=1>Kvasir-SEGDice  IoU</td><td rowspan=1 colspan=1>ISIC2016Dice  IoU</td></tr><tr><td rowspan=3 colspan=1>VV           V了            V              了</td><td rowspan=1 colspan=1>85.4778.82</td><td rowspan=1 colspan=1>87.64 80.08</td></tr><tr><td rowspan=1 colspan=1>87.73 79.59</td><td rowspan=1 colspan=1>89.12 82.33</td></tr><tr><td rowspan=1 colspan=1>92.10 87.05|</td><td rowspan=1 colspan=1>92.58 86.91</td></tr></table>

Table 7. Ablation study on the output representation. “Overlay” is the proposed color-coded mask rendered on the original image. “Black-bg mask” is the color-coded mask on a black background. “Binary mask” is the grayscale [0, 1] mask without color instructions. For the black-background variant, the evaluation color map is remapped to the dominant color actually painted by the model, so its reported scores are corrected Dice.
<table><tr><td>Output Representation</td><td>Kvasir-SEG Dice IoU</td><td>ISIC2016 Dice IoU</td></tr><tr><td>Binary mask</td><td>82.3574.22</td><td>52.6043.22</td></tr><tr><td>Black-bg color mask</td><td>65.31 56.99</td><td>53.4044.88</td></tr><tr><td>Overlay (Ours)</td><td></td><td>92.10 87.05|92.58 86.91</td></tr></table>

Table 8. Ablation study on feature fusion strategy.
<table><tr><td rowspan=1 colspan=1>Fusion Method</td><td rowspan=1 colspan=1>Kvasir-SEGDice IoU</td><td rowspan=1 colspan=1>ISIC2016Dice  IoU</td></tr><tr><td rowspan=2 colspan=1>Element-wise AdditionConcatenation</td><td rowspan=1 colspan=1>91.31 85.26</td><td rowspan=2 colspan=1>91.77 84.6592.58 86.91</td></tr><tr><td rowspan=1 colspan=1>92.10 87.05|</td></tr></table>

![](images/f8088c9202fbda59ab102abe14e5de2120889cb9242b7503ad55db87f5346de6.jpg)  
Fig. 9. Sensitivity analysis of the auxiliary loss weight λ and the number of active DINOv3 feature layers.

## 4.5 Color Sensitivity Analysis

Because the instruction-driven formulation relies on color-coded editing targets, we assess the impact of color diversity during training. We build five training configurations with $N _ { c } ~ \in ~ \{ 3 , 5 , 8 , 1 2 , 1 6 \}$ } colors, drawn in order from the fixed palette in Table 10; the main experiments adopt the $N _ { c } = 5$ configuration. All models are trained on the Kvasir-SEG dataset and evaluated with the instruction fixed to “Segment the polyp region using red,” regardless of the color set used during training. Table 11 reports the Dice and IoU on the Kvasir test set for each $N _ { c } .$ The results indicate that performance remains relatively stable across $N _ { c }$ , with a mild peak at $N _ { c } = 1 2$ . Increasing color diversity therefore

Table 9. Ablation study on the auxiliary backbone of the DFG branch. All backbones share the same pyramid adapter and auxiliary segmentation head. The last 6 blocks of each alternative backbone are unfrozen during training (ResNet-50 is fully frozen), while DINOv3 remains frozen as in the main configuration.
<table><tr><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>Kvasir-SEGDice  IoU</td><td rowspan=1 colspan=1>ISIC2016Dice   IoU</td></tr><tr><td rowspan=3 colspan=1>NoneResNet-50MedSAM ViT-B</td><td rowspan=1 colspan=1>86.2379.36</td><td rowspan=4 colspan=1>91.6285.4491.04 85.0892.2286.3591.7785.86</td></tr><tr><td rowspan=1 colspan=1>85.59 78.58</td></tr><tr><td rowspan=1 colspan=1>90.6084.65</td></tr><tr><td rowspan=1 colspan=1>CLIP ViT-B/16</td><td rowspan=1 colspan=1>87.5181.08</td></tr><tr><td rowspan=1 colspan=1>DINOv2 ViT-B/14</td><td rowspan=1 colspan=1>91.0085.49</td><td rowspan=1 colspan=1>91.4885.46</td></tr><tr><td rowspan=3 colspan=1>DINOv3 ViT-S/16DINOv3 ViT-B/16DINOv3 ViT-L/16</td><td rowspan=1 colspan=1>92.1087.05</td><td rowspan=1 colspan=1>92.58 86.91</td></tr><tr><td rowspan=1 colspan=1>92.36 87.87</td><td rowspan=1 colspan=1>92.74 87.30</td></tr><tr><td rowspan=1 colspan=1>92.54 88.1193.06</td><td rowspan=1 colspan=1>87.62</td></tr></table>

does not harm the editing capability when a requested color has been seen, and a moderate number of colors is suficient for learning the instruction-to-color mapping.

Unseen Colors: Raw versus Corrected Evaluation Fig. 10 shows the raw segmentation performance on three unseen colors, brown, silver, and lavender, none of which belongs to the 16-color training palette. The raw Dice conflates two distinct failure modes: the model may fail to segment the lesion at all, or it may segment correctly but paint with a wrong color, in which case the fixed color-map evaluation assigns every pixel to the wrong class and collapses the score. To separate these modes, we extract the dominant color actually painted by each model, remap the evaluation color map to this color, and recompute a corrected Dice. Table 12 reports raw and corrected Dice for all models and unseen colors.

Three findings follow from Table 12. First, the low raw Dice on lavender is almost entirely a colormapping artifact, since the corrected Dice is above 90% for all $N _ { c } \geq 5 ,$ , meaning the model segments correctly but renders purple, the nearest color in its training pool, instead of lavender. Second, genuine segmentation failures on unseen colors persist at small $N _ { c } ,$ with corrected Dice of 14.70% and 12.12% for brown and silver at $N _ { c } = 5 .$ but largely disappear at $N _ { c } = 1 6 .$ , where brown reaches 92.55%; training with a richer color pool decouples the color token from the task. Third, for lavender the apparent failure is essentially an evaluation artifact, whereas for brown and silver genuine segmentation failures coexist with color binding at small $N _ { c } ;$ both issues largely disappear once the color pool is rich enough.

Fig. 11 visualizes the output on two test samples. When the target color is unseen, models either reconstruct the input without an overlay or segment the polyp using a training color. With $N _ { c } = 1 6$ the model produces an accurate brown mask, and for the harder silver and lavender requests it renders an incorrect but consistent color while preserving the segmentation, which suggests that a richer color set encourages more robust visual reasoning.

![](images/b7aef5c9468916bb01667c9220633cd02f081c0d04a33b7113d737be836c10cc.jpg)  
Fig. 10. Boxplots of Dice scores for three unseen colors.

GT  
N<sub>C</sub>=3  
N<sub>C</sub>=5  
N<sub>C</sub>=8  
N<sub>C</sub>=12  
N<sub>C</sub>=16  
![](images/00194d97398cc29a62f9e1ff3a5b7bd1714b618848fc0b09dc523294018f4f91.jpg)  
Fig. 11. Segmentation outputs on two test samples from models trained with diferent $N _ { c } .$ When the target color is unseen, models either reconstruct the original input directly or segment the polyp using a color that appeared during their training.

Table 10. Color palette used for instruction-driven training. The colors are arranged in the order used for constructing the training subsets with increasing $N _ { c } .$
<table><tr><td colspan="2">Index Color Name RGB Values</td></tr><tr><td>1 Red</td><td>(255, 0, 0)</td></tr><tr><td>2 Green</td><td>(0, 255, 0)</td></tr><tr><td>3 Blue</td><td>(0, 0, 255)</td></tr><tr><td>4 Yellow</td><td>(255, 255, 0)</td></tr><tr><td>5 Purple</td><td>(128, 0, 128)</td></tr><tr><td>6 Pink</td><td>(255, 192, 203)</td></tr><tr><td>7 Cyan</td><td>(0, 255, 255)</td></tr><tr><td>8 Teal</td><td>(0, 128, 128)</td></tr><tr><td>9 Magenta</td><td>(255, 0, 255)</td></tr><tr><td>10 Orange</td><td>(255, 165, 0)</td></tr><tr><td>11 Lime</td><td>(50, 205, 50)</td></tr><tr><td>12 Coral</td><td>(255, 127, 80)</td></tr><tr><td>13 Violet</td><td>(238, 130, 238)</td></tr><tr><td>14 Navy</td><td>(0, 0, 128)</td></tr><tr><td>15 Olive</td><td>(128, 128, 0)</td></tr><tr><td>16 Maroon</td><td>(128, 0, 0)</td></tr></table>

Table 11. Segmentation performance on Kvasir-SEG under diferent numbers of training colors $( N _ { c } )$ . The inference instruction always specifies the color “red”.
<table><tr><td colspan="3">Dice (%) IoU (%)  $N _ { c }$ </td></tr><tr><td>3</td><td>91.25</td><td>86.43</td></tr><tr><td>5</td><td>92.10</td><td>87.05</td></tr><tr><td>8</td><td>91.28</td><td>86.38</td></tr><tr><td>12</td><td>92.42</td><td>87.41</td></tr><tr><td>16</td><td>91.49</td><td>86.46</td></tr></table>

## 5 Discussion

Our results support the editing formulation as a practical alternative to both discriminative models and binary-mask difusion segmentation. Because the color-coded overlay retains the original image content while superimposing the target annotation, a pretrained latent difusion model can operate on medical imagery with a much smaller efective domain shift. This is reflected in the unseen-domain results, where the framework attains the best Dice on both PolypGen and ISIC2017, and it explains why the overlay outperforms a standalone mask or a black-background mask in the output-representation ablation.

Two analyses bound the framework. The color-coded interface is robust to design choices, since performance stays within a narrow band across palettes and the unseen-color analysis shows that the model tends to render the nearest training color whenever segmentation succeeds; both color binding and genuine segmentation failures under unseen colors largely vanish with a suficiently rich palette. In contrast, instruction control is limited to category selection, since the model does not parse spatia or size modifiers and attribute-conditioned selection is not supported.

These strengths come with trade-ofs. In-domain, the strongest discriminative baselines remain competitive or superior on Kvasir-SEG, CVC-ClinicDB, and ISIC2016, and the editing formulation is more expensive at inference than a single discriminative pass, so its value is concentrated in crossdomain generalization, multi-lesion completeness, and instruction-conditioned control.

Table 12. Raw versus corrected Dice (%) on unseen colors. Corrected Dice remaps the evaluation color map to the color actually painted by the model, isolating color-mapping artifacts from true segmentation failures.
<table><tr><td colspan="3"> $N _ { c }$  Unseen color Raw Dice Corrected Dice</td></tr><tr><td>3 brown</td><td>26.66</td><td>35.13</td></tr><tr><td>3 silver</td><td>3.81</td><td>60.23</td></tr><tr><td>3 lavender</td><td>14.48</td><td>79.43</td></tr><tr><td>5 brown</td><td>0.59</td><td>14.70</td></tr><tr><td>5 silver</td><td>8.61</td><td>12.12</td></tr><tr><td>5 lavender</td><td>2.20</td><td>92.72</td></tr><tr><td>8 brown</td><td>17.46</td><td>32.28</td></tr><tr><td>8 silver</td><td>26.28</td><td>51.33</td></tr><tr><td>8 lavender</td><td>2.18</td><td>91.83</td></tr><tr><td>12 brown</td><td>39.18</td><td>70.39</td></tr><tr><td>12 silver</td><td>17.22</td><td>51.58</td></tr><tr><td>12 lavender</td><td>2.03</td><td>92.37</td></tr><tr><td>16 brown</td><td>92.13</td><td>92.55</td></tr><tr><td>16 silver</td><td>26.96</td><td>68.93</td></tr><tr><td>16 lavender</td><td>28.26</td><td>92.71</td></tr></table>

## 6 Conclusion

In this paper, we proposed InstEditSeg, a unified generative framework that recasts medical image segmentation as an instruction-driven difusion editing task. By producing a color-coded edited image instead of a standalone binary mask, it difers from both binary-mask difusion segmentation and discriminative decoders and aligns the output with the priors of latent difusion models, reducing the domain gap between natural and medical images. A DINOv3-guided multi-scale feature pyramid, integrated through concatenation and zero-initialized convolution, injects hierarchical discriminative priors into the denoising process and improves lesion localization and boundary delineation. Experiments on polyp and skin lesion benchmarks show that the framework is competitive with strong discriminative baselines and attains the best Dice on the unseen PolypGen and ISIC2017 sets. The editing formulation further gives concrete advantages in multi-lesion completeness (12.75 Dice points over EMCAD), instruction-conditioned task control, and sampling flexibility, retaining about 99.7% of the 25-step accuracy at five steps. The framework also has clear limitations: attribute-conditioned selection is unsupported, and the strongest discriminative baselines remain competitive or superior in-domain on Kvasir-SEG, CVC-ClinicDB, and ISIC2016. We plan to extend the framework to attribute-conditioned and interactive editing, few-step distillation, and 3D volumetric segmentation.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used DeepSeek in order to assist with LaTeX formatting. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Acknowledgment

This work was supported by the National Natural Science Foundation of China under Grant 62572406, the Science and Technology Department of Sichuan Province under Grant 2024NSFSC2040.

## References

1. D. Gutman, N. C. Codella, E. Celebi, B. Helba, M. Marchetti, N. Mishra, and A. Halpern, “Skin lesion analysis toward melanoma detection: A challenge at the international symposium on biomedical imaging (isbi) 2016, hosted by the international skin imaging collaboration (isic),” 2016.

2. O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention, pp. 234–241, Springer, 2015.

3. A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Doll´ar, and R. Girshick, “Segment anything,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 5, pp. 3412–3426, 2023.

4. J. Ho, A. Jain, and P. Abbeel, “Denoising difusion probabilistic models,” in Advances in Neural Information Processing Systems, vol. 33, pp. 6840–6851, 2020.

5. T. Amit, T. Shaharbany, E. Nachmani, and L. Wolf, “Segdif: Image segmentation with difusion probabilistic models,” 2021.

6. J. Wu, R. Fu, H. Fang, Y. Zhang, Y. Yang, H. Xiong, H. Liu, and Y. Xu, “Medsegdif: Medical image segmentation with difusion probabilistic model,” in Medical imaging with deep learning, vol. 227 of Proceedings of Machine Learning Research, pp. 1623–1639, PMLR, 2023.

7. J. Wu, W. Ji, H. Fu, M. Xu, Y. Jin, and Y. Xu, “Medsegdif-v2: Difusion-based medical image segmentation with transformer,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, pp. 6030–6038, 2024.

8. T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to follow image editing instructions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18392– 18402, 2023.

9. V. Gabeur, S. Long, S. Peng, P. Voigtlaender, S. Sun, Y. Bao, K. Truong, Z. Wang, W. Zhou, J. T. Barron, K. Genova, N. Kannen, S. Ben, Y. Li, M. Guo, S. Yogin, Y. Gu, H. Chen, O. Wang, S. Xie, H. Zhou, K. He, T. Funkhouser, J.-B. Alayrac, and R. Soricut, “Image generators are generalist vision learners,” 2026.

10. X. Han, J. Li, K. Deng, Z. Chen, X. Shi, S. Wang, B. Li, L. Wang, S. Xie, X. You, J. Quan, Z. Cai, H. Diao, Z. Liu, L. Yang, D. Lin, and Q. Wang, “Vision as unified multimodal generation,” 2026.

11. O. Oktay, J. Schlemper, L. L. Folgoc, M. Lee, M. Hein, K. Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz, et al., “Attention u-net: Learning where to look for the pancreas,” in Medical imaging with deep learning, pp. 1–9, 2018.

12. Z. Zhou, M. M. Rahman Siddiquee, N. Tajbakhsh, and J. Liang, “Unet++: A nested u-net architecture for medical image segmentation,” in Deep learning in medical image analysis and multimodal learning for clinical decision support, pp. 3–11, Springer, 2018.

13. D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. d. Lange, H. D. Johansen, D. Johansen, and J. Rittscher, “Resunet++: An advanced architecture for medical image segmentation,” IEEE Transactions on Emerging Topics in Computational Intelligence, 2020.

14. L. Yu, H. Chen, Q. Dou, J. Qin, and P. A. Heng, “Automated polyp segmentation in colonoscopy images using deep learning,” International Journal of Computer Assisted Radiology and Surgery, 2019.

15. B. Dong, W. Wang, D.-P. Fan, J. Li, H. Fu, and Y.-G. Shao, “Polyp-pvt: Polyp segmentation with pyramid vision transformers,” CAAI Artificial Intelligence Research, vol. 2, p. 9150015, 2023.

16. M. M. Rahman, M. Munir, and R. Marculescu, “Emcad: Eficient multi-scale convolutional attention decoding for medical image segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

17. J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, “Segment anything in medical images,” Nature Communications, vol. 15, no. 1, p. 654, 2024.

18. J. Wu, R. Ye, L. Zhang, Y. Wang, J. Qin, and Y. Huang, “Medical sam adapter: Adapting segment anything model for medical image segmentation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2023.

19. J. Cheng, J. Ye, Z. Deng, J. Chen, T. Li, H. Wang, Y. Su, Z. Huang, J. Chen, L. Jiang, H. Sun, J. He, S. Zhang, M. Zhu, and Y. Qiao, “Sam-med2d,” 2023.

20. Y. Li, M. Hu, and X. Yang, “Polyp-sam: Transfer SAM for polyp segmentation,” in Medical Imaging 2024: Computer-Aided Diagnosis, vol. 12931, p. 117, SPIE, 2024.

21. J. Ma, Z. Yang, S. Kim, B. Chen, M. Baharoon, A. Fallahpour, R. Asakereh, H. Lyu, and B. Wang, “Medsam2: Segment anything in 3d medical images and videos,” 2025.

22. N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. R¨adle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Doll´ar, and C. Feichtenhofer “Sam 2: Segment anything in images and videos,” in Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2408.00714.

23. Y. Huang, X. Yang, L. Liu, H. Zhou, A. Chang, X. Zhou, R. Chen, J. Yu, J. Chen, C. Chen, S. Liu, H. Chi, X. Hu, K. Yue, L. Li, V. Grau, D.-P. Fan, F. Dong, and D. Ni, “Segment anything model for medical images?,” Medical Image Analysis, vol. 92, p. 103061, 2024.

24. J. Wolleb, R. Sandk¨uhler, F. Bieder, P. Valmaggia, and P. C. Cattin, “Difusion models for implicit image segmentation ensembles,” in Proceedings of the 5th International Conference on Medical Imaging with Deep Learning, vol. 172 of Proceedings of Machine Learning Research, pp. 1336–1348, PMLR, 2022.

25. T. Lin, Z. Chen, Z. Yan, W. Yu, and F. Zheng, “Stable difusion segmentation for biomedical images with single-step reverse process,” in International conference on medical image computing and computer-assisted intervention, pp. 656–666, Springer, 2024.

26. Z. Yang, C. Li, and J. Ma, “Tsldseg: A texture-aware and semantic-enhanced latent difusion model for medical image segmentation,” Pattern Recognition, vol. 173, p. 112795, 2026.

27. T. Chen, L. Li, S. Saxena, G. Hinton, and D. J. Fleet, “A generalist framework for panoptic segmentation of images and videos,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 909–919, 2023.

28. H. Ye, J. Kuen, Q. Liu, Z. Lin, B. Price, and D. Xu, “Seggen: Supercharging segmentation models with text2mask and mask2img synthesis,” in Computer Vision – ECCV 2024, Springer, 2024. arXiv:2311.03355.

29. W. Van Gansbeke and B. De Brabandere, “A simple latent difusion approach for panoptic segmentation and mask inpainting,” 2024.

30. W. Zhao, Y. Rao, Z. Liu, B. Liu, J. Zhou, and J. Lu, “Unleashing text-to-image difusion models for visual perception,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5729–5739, 2023.

31. C.-M. Feng, “Enhancing label-eficient medical image segmentation with text-guided difusion models,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, vol. 15008 of Lecture Notes in Computer Science, Springer, 2024.

32. Y. Xie, J. Qu, H. Xie, T. Wang, and B. Lei, “Difdgss: Generalizable retinal image segmentation with deterministic representation from difusion models,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, Lecture Notes in Computer Science, pp. 166–176, Springer, 2024.

33. H. Zhang, W. Guo, Y. Liu, J. Yang, S. Bhagavan, D. Shi, M. Xu, and P. Fua, “Genmed: A pairwise generative reformulation of medical diagnostic tasks,” 2026.

34. A. Kazerouni, E. K. Aghdam, M. Heidari, R. Azad, M. Fayyaz, I. Hacihaliloglu, and D. Merhof, “Difusion models in medical imaging: A comprehensive survey,” Medical Image Analysis, vol. 88, p. 102846, 2023.

35. R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent difusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

36. L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image difusion models,” in Proceedings of the IEEE/CVF international conference on computer vision, pp. 3836–3847, 2023.

37. C. Mou, X. Wang, L. Xie, Y. Wu, J. Zhang, Z. Qi, and Y. Shan, “T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image difusion models,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, pp. 4296–4304, 2024.

38. T. Chen, S. Saxena, L. Li, D. J. Fleet, and G. Hinton, “Pix2seq: A language modeling framework for object detection,” 2021.

39. J. Liu, Y. Zhang, J.-N. Chen, J. Xiao, Y. Lu, B. A. Landman, Y. Yuan, A. Yuille, Y. Tang, and Z. Zhou, “Clip-driven universal model for organ segmentation and tumor detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

40. O. Sim´eoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. J´egou, P. Labatut, and P. Bojanowski, “DINOv3,” 2025.

41. D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. De Lange, D. Johansen, and H. D. Johansen, “Kvasir-seg: A segmented polyp dataset,” in International conference on multimedia modeling, pp. 451–462, Springer, 2019.

42. J. Bernal, F. J. S´anchez, G. Fern´andez-Esparrach, D. Gil, C. Rodr´ıguez, and F. Vilari˜no, “Wm-dova maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians,” Computerized medical imaging and graphics, vol. 43, pp. 99–111, 2015.

43. N. Tajbakhsh, S. R. Gurudu, and J. Liang, “Automated polyp detection in colonoscopy videos using shape and context information,” IEEE transactions on medical imaging, vol. 35, no. 2, pp. 630–644, 2015.

44. J. Silva, A. Histace, O. Romain, X. Dray, and B. Granado, “Toward embedded detection of polyps in wce images for early diagnosis of colorectal cancer,” International journal of computer assisted radiology and surgery, vol. 9, no. 2, pp. 283–293, 2014.

45. S. Ali, D. Jha, N. Ghatwary, S. Realdon, R. Cannizzaro, O. E. Salem, D. Lamarque, C. Daul, M. A. Riegler, K. V. Anonsen, et al., “A multi-centre polyp detection and segmentation dataset for generalisability assessment,” Scientific Data, vol. 10, no. 1, p. 75, 2023.

46. N. C. Codella, D. Gutman, M. E. Celebi, B. Helba, M. A. Marchetti, S. W. Dusza, A. Kalloo, K. Liopyris, N. Mishra, H. Kittler, et al., “Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collabora tion (isic),” in 2018 IEEE 15th international symposium on biomedical imaging (ISBI 2018), pp. 168–172, IEEE, 2018.

47. J. Song, C. Meng, and S. Ermon, “Denoising difusion implicit models,” in International Conference on Learning Representations (ICLR), 2021.