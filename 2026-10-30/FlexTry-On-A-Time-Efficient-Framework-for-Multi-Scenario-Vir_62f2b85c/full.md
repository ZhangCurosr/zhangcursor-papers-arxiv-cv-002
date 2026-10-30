# FlexTry-On: A Time-Efficient Framework for Multi-Scenario Virtual Try-on

Phuong Dam Quyen Nguyen Duc Son Hoang Minh Daeyoung Kim Korea Advanced Institute of Science and Technology (KAIST) {hoangphuong1211,nguyenducquyen2311,sonhm2910,kimd}@kaist.ac.kr

![](images/9305618bdfe5074ec1f39c72545ce9bbe9df29cdca30f1196a23bdf72dce08f9.jpg)  
Figure 1. Qualitative results of FlexTry-On across multiple IVTON scenarios, including Cloth-to-Person, Person-to-Person on the DressCode dataset, In-the-Wild settings, and Cross-Domain IVTON. Please zoom in for a clearer visualization of fine details.

## Abstract

Virtual try-on has achieved remarkable progress in recent years, yet most existing methods are designed for a specific scenario, such as cloth-to-person or person-to-person try-on, and often require separate conditioning strategies and processing pipelines for different settings. We present FlexTry-On, a unified and time-efficient diffusion-based framework for diverse virtual try-on scenarios, including cloth-to-person, person-to-person, cross-domain, and inthe-wild settings. FlexTry-On adopts a common conditioning formulation, where a text-aware condition extraction strategy converts heterogeneous garment sources into a unified representation, and a dual-conditioning diffusion transformer jointly models garment appearance and person identity. Extensive experiments demonstrate competitive performance on both cloth-to-person and person-toperson benchmarks while maintainingfast inference. Moreover, the same framework can be directly applied to unseen cross-domain scenarios without requiring separate architectures or scenario-specific processing pipelines. These results highlight the potential ofunified and efficient virtual try-on frameworks as a practical step toward more flexible and widely applicable virtual try-on systems.

## 1. Introduction

Image-based virtual try-on (IVTON) aims to synthesize realistic images of a target person wearing a desired garment while preserving both garment appearance and human identity [5]. Recent advances in diffusion models [18] have significantly improved virtual try-on quality, leading to highly realistic results on standard benchmarks [3, 8, 19, 23, 33]. However, most existing methods are developed for a specific setting, such as cloth-to-person (C2P) or person-toperson (P2P) virtual try-on. In practice, garments may originate from clean product images, reference persons, or unconstrained real-world photographs. Existing virtual try-on systems, however, are typically tailored to a particular input format and scenario. Extending them to cross-domain or in-the-wild settings often requires additional assumptions, preprocessing steps, or task-specific adaptations, which also introduce additional computational overhead. As a result, virtual try-on remains fragmented across scenarios despite sharing the same underlying objective of garment transfer. A key challenge is how to convert garments from heterogeneous sources into a unified conditioning representation while preserving garment appearance and person identity.

## Can a single virtual try-on framework effectively handle diverse virtual try-on scenarios?

In this paper, we present FlexTry-On, a unified and time-efficient diffusion-based framework for cloth-toperson, person-to-person, cross-domain, and in-the-wild virtual try-on. To accommodate heterogeneous garment sources, FlexTry-On first employs a Text-Aware Segmentation Module (TASM) that extracts task-aware garment conditions from either product images or reference persons.

These conditions are converted into a unified conditioning representation and processed by a Dual-DiT Tryon Module (DDTM), where a reference branch captures garment appearance and a try-on branch synthesizes the final image while preserving person identity. In addition, we employ a Style-GRAM loss to encourage appearance consistency between the generated garment region and the reference garment, improving garment preservation under noisy supervision. Together, these components enable the same framework to support diverse virtual try-on scenarios through a shared conditioning and generation pipeline.

By formulating diverse virtual try-on tasks under a common conditional generation framework, FlexTry-On eliminates the need for scenario-specific architectures and maintains a consistent inference pipeline across different settings. Furthermore, the same framework supports multiple garment categories within a unified conditioning formulation, enabling a unified formulation across different virtual try-on scenarios.

Extensive experiments demonstrate that FlexTry-On achieves competitive visual quality and strong FID [17] performance on both C2P and P2P benchmarks while maintaining efficient inference. In addition, the same framework can be directly applied to challenging cross-domain and in-the-wild scenarios. As shown in Fig. 2, FlexTry-On achieves a favorable balance between synthesis quality, inference efficiency, and scenario flexibility, providing a practical step toward more flexible and widely applicable virtual try-on systems. Our contributions are summarized as:

1. We propose FlexTry-On, a unified and time-efficient diffusion-based framework that bridges cloth-to-person, person-to-person, cross-domain, and in-the-wild virtual try-on scenarios within a single framework.

2. We introduce a unified virtual try-on framework consisting of a Text-Aware Segmentation Module (TASM) and a Dual-DiT Tryon Module (DDTM), enabling conditioning from heterogeneous garment sources while jointly modeling garment appearance and person identity. We further employ a Style-GRAM loss for garment appearance preservation under noisy supervision.

3. Extensive evaluations on both standard benchmarks and challenging real-world settings demonstrate competitive visual quality, strong FID performance, and efficient inference within a unified framework.

## 2. Related Works

IVTON [5] has evolved from warping-based pipelines to diffusion-based generative frameworks. Early methods [2, 8, 16, 26, 36, 41] rely on explicit garment alignment and image synthesis, making them sensitive to pose variations, occlusions, and inaccurate geometric estimation. Recent diffusion-based approaches [3, 4, 13, 15, 22, 23, 25, 29, 33,

![](images/12553b8dd28c2d9b8446d335e943666d45c64ffd8a8244a04fa451cbb5c956e2.jpg)  
Figure 2. Comparison of FlexTry-On with other diffusion-based methods using the VITON-HD dataset (C2P). Lower values indicate better performance for both FID and inference speed. FlexTry-On supports all evaluated virtual try-on scenarios within a unified framework, as summarized in Tab. 1.

45, 48] significantly improve visual quality through large pre-trained models and attention-based mechanisms.

Despite these advances, most existing methods are designed around specific conditioning assumptions. Many diffusion-based IVTON approaches assume a particular garment representation, such as clean product images, extracted garment regions, or reference-person-based garment cues, and design their conditioning mechanisms accordingly [3, 4, 13, 15, 22, 23, 25, 29, 33, 45, 48]. Consequently, their conditioning mechanisms are often tailored to specific input formats, limiting both flexibility and efficiency across different virtual try-on scenarios.

Furthermore, most existing methods [3, 4, 13, 22, 23, 25, 29, 33, 45, 48] implicitly assume category-consistent garment transfer, where the source and target garments belong to the same clothing category (e.g., upper-to-upper or dressto-dress). Under this assumption, editable regions can be determined using predefined masks, human parsing labels, or garment extraction pipelines. However, transfers across different garment categories introduce ambiguous editable regions that cannot be uniquely determined using conventional category-based masking strategies.

Recent methods, including Mask Does Not Matter [9], All Parts Matter [10], OmniTry [11], OmniVTON [44], and M&M VTO [49], extend virtual try-on beyond the conventional cloth-to-person setting toward more diverse and realistic applications. Nevertheless, they are primarily designed for specific input assumptions and do not explicitly address category-crossing transfers. These observations suggest the key challenge is not only generating realistic try-on images but also designing a framework that can accommodate heterogeneous garment sources, identify task-aware editable regions, and support diverse virtual try-on scenarios without sacrificing inference efficiency.

## 3. Methodology

To support diverse virtual try-on scenarios within a single framework, we propose FlexTry-On, a diffusion-based approach that accommodates different input modalities and garment categories under a unified formulation. FlexTry-

![](images/3e600ea7ed19f57b1a5c198822def4f12be8bf88d66dcf66763c9268436e653b.jpg)  
Figure 3. Overview of the $\operatorname { F l e x T r y - O n }$ framework for IVTON. Multiple reference inputs, including pose keypoints $J _ { p }$ and DensePose maps $I _ { d p } ,$ are encoded into latent representations and concatenated as inputs to the diffusion process. The TASM module receives a latent image with text conditions embedded via the CLIP-Text Encoder [39] from SD 2.0. The TASM outputs both the latent representations of the target RGB regions and their corresponding masks. In TASM, we illustrate the latent representation of the RGB input cloth $P _ { i c }$ and its associated mask $M _ { i c } .$ The garment-agnostic mask $M _ { g a }$ is obtained by merging masks under different text conditions according to Eq. 2

On consists of two key components: a Text-Aware Segmentation Module (TASM), which converts heterogeneous garment inputs into task-aware garment conditions and editable regions, and a Dual-DiT TryOn Module (DDTM), which synthesizes the final try-on image by jointly modeling garment appearance and person identity.

Built on a DiT-based latent diffusion framework [35, 39], FlexTry-On encodes inputs into a shared latent space using a pre-trained VAE. DDTM employs a dual-DiT design, where a reference DiT extracts garment representations only once and a try-on DiT performs image synthesis through denoising. This design enables a unified and computationally efficient pipeline for cloth-to-person, personto-person, and cross-domain virtual try-on scenarios.

## 3.1. Text-Aware Segmentation Module (TASM)

To support diverse virtual try-on scenarios, we introduce a Text-Aware Segmentation Module (TASM) that converts heterogeneous inputs into a unified conditioning representation. Unlike conventional human parsing models [28] or generic segmentation frameworks such as SAM [24] and Grounded-SAM [30, 37] that are designed for pixel-level segmentation, TASM is designed specifically for virtual tryon conditioning and operates directly in the latent space. Given an image and predefined text conditions, TASM predicts garment-specific masks for downstream conditioning.

As illustrated in Fig. 3, TASM adopts a DiT-based architecture [35] and injects text conditions through multi-head cross-attention. Given a latent image representation, the module predicts region-specific masks corresponding to different garment categories. Specifically, only the predicted masks are retained for subsequent processing, while associated latent appearance features are discarded. An ablation of this design is provided in Appendix A.1. The objective function is a combination of reconstruction $\mathcal { L } _ { 1 } , \mathcal { L } _ { 2 }$ and adversarial objectives:

$$
\mathcal { L } _ { \mathrm { t e x t s e g } } = \mathcal { L } _ { 1 } + \mathcal { L } _ { 2 } + \alpha _ { \mathrm { G A N } } \mathcal { L } _ { \mathrm { G A N } } ,\tag{1}
$$

where $\alpha _ { \mathrm { G A N } }$ controls the contribution of the adversarial loss [20]. Ground-truth masks follow the protocol of [47], supplemented with additional annotations for facial, hair, and accessory regions from [28].

To construct the garment-agnostic mask used in the tryon stage, TASM predicts region-specific masks according to predefined text conditions. These masks are merged as:

$$
{ \cal M } _ { g a } = \mathrm { R E C T } \left[ M _ { i c } \vee M _ { i s k } \right] \backslash { \cal M } _ { i f h } ,\tag{2}
$$

where $M _ { i c } , M _ { i s k } ,$ , and $M _ { i f h }$ denote the clothing, skin, and face-with-hands masks, respectively, and RECT[·] represents the enclosing rectangular region. This formulation enables flexible region control across cross-domain try-on settings (e.g., dress-to-upper or upper–lower-to-dress), as illustrated in Fig. 6. Unlike conventional human parsing [21, 28], TASM generates task-specific masks that explicitly identify the regions involved in the try-on process. By decoupling editable-region construction from predefined garment categories, TASM directly supports cross-domain virtual try-on scenarios where conventional category-dependent masking strategies become ambiguous. Additional discussion is provided in Appendix G.1.

![](images/9938b12314379425072d97b5c3f11e00a708ba360596b0f8577fdf1260b0785e.jpg)  
Figure 4. Overview of the Dual-DiT Tryon Module (DDTM). The DDTM follows a DiT-based architecture [35] with cross-attention to integrate person and garment information. During training, different conditioning strategies are adopted for C2P and P2P settings. For clarity, patchification and positional embeddings are omitted.

## 3.2. Dual-DiT Tryon Module (DDTM)

The Dual-DiT Tryon Module consists of two diffusion transformer branches [35]: a Reference Net and a Main Net, as illustrated in Fig. 4. The Reference Net extracts garmentrelated features, while the Main Net performs try-on synthesis. This dual-stream design decouples garment appearance modeling from the generation process, providing structured guidance throughout denoising.

Given a garment image $I _ { g }$ and a person image $I _ { p } ,$ , TASM constructs the garment condition $z _ { g } = \mathcal { E } ( I _ { g } ) \odot M _ { g } .$ , where $M _ { g }$ denotes the garment mask. Similarly, the garmentagnostic person condition is constructed as $z _ { g a } = \mathcal { E } ( I _ { p } )$ ⊙ $M _ { g a } ,$ , where $M _ { g a }$ is predicted by TASM. For P2P, TASM extracts $M _ { g }$ from the reference person to isolate the source garment, whereas for C2P the clean product garment is directly retained. Thus, the same interface accommodates both product-garment and person-reference inputs.

The Reference Net processes $z _ { g }$ once at a fixed diffusion timestep [3], producing multi-level garment features that are reused throughout denoising. The Main Net receives the noisy latent $z _ { t } ,$ garment-agnostic latent $z _ { g a } ,$ , mask $M _ { g a } ,$ pose keypoints $\mathcal { E } ( J _ { p } )$ , and DensePose map $\mathcal { E } ( I _ { d p } )$ . These spatial conditions are patch-embedded and concatenated to form the global conditioning input. Intermediate Reference-Net features are combined with garment-type text embeddings and injected into the corresponding Main-Net layers through cross-attention. This design provides multi-level garment guidance while separating garment feature extraction from iterative image synthesis.

## 3.2.1. Denoising Objective

The training objective follows the standard latent diffusion formulation [39]:

$$
\begin{array} { r } { \mathcal { L } _ { d e n o i s e } = \mathbb { E } _ { y , c , \epsilon \sim \mathcal { N } ( 0 , I ) , t } \left[ \left\| \epsilon - \epsilon _ { \theta } ( y _ { t } , c , t ) \right\| ^ { 2 } \right] , } \end{array}\tag{3}
$$

where $y _ { t }$ denotes the noisy latent at timestep t, c represents the conditioning inputs including $z _ { g } , z _ { g a } , M _ { g a } , \mathcal { E } ( I _ { d p } )$ , and $\mathcal { E } ( J _ { p } )$ , and $\epsilon _ { \theta }$ is the predicted noise.

## 3.2.2. GRAM-based Style Objective

A fundamental requirement of IVTON is that the target garment must appear in the final synthesized result. However, the diffusion denoising objective primarily supervises image reconstruction and does not explicitly optimize garment fidelity, particularly on large-scale datasets with noisy garment-person correspondences [40]. Specifically, samples may share the same garment category while differing substantially in color, texture, or fine-grained appearance. Under such conditions, the model can satisfy the denoising objective while gradually reducing its reliance on garmentspecific cues, leading to degraded garment fidelity. An illustrative mis-pairing example is provided in Fig. 5.

To bridge this gap, we introduce a GRAM-based Style Loss as a task-alignment constraint that explicitly encourages preservation of garment-specific appearance cues. Although GRAM-based style losses are not new [12], our contribution lies in revisiting them as an optimization objective for virtual try-on rather than image stylization. Unlike pixel-wise reconstruction losses, which are incompatible with the geometric variations inherent in virtual try-on, the proposed loss operates on appearance statistics and remains invariant to spatial layout, making it particularly effective for preserving garment color and texture during synthesis. The style loss is defined as:

$$
\mathcal { L } _ { s t y l e } = \mathrm { M S E } \Big ( G ( \boldsymbol { z } _ { g } ) , G ( \boldsymbol { z } _ { o u t } \odot M _ { i c } ) \Big ) ,\tag{4}
$$

where $G ( \cdot )$ denotes the Gram matrix, $z _ { g }$ represents the latent garment condition, and $z _ { o u t } \odot M _ { i c }$ extracts the predicted garment region from the denoised latent.

Table 1. Quantitative comparison on VITON-HD and DressCode Upper in C2P. Inf. Time denotes method-specific inference time; shared pose/DensePose preprocessing is excluded, with exceptions detailed in Appendix E. FlexTry-On is trained only on VITON-HD and Dress-Code, whereas FlexTry-On\* uses the full uncurated C2P–P2P corpus. Best/second-best results are bold/underlined. Cross C2P/P2P denote cross-domain variants. DressCode Lower/Dresses results are in Appendix B. †: paper-reported.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Inf. Time (s) ↓</td><td colspan="3">VITON-HD</td><td rowspan="2"></td><td colspan="3">Dress Code - Upper</td><td rowspan="2"></td><td colspan="3">Support IVTON Tasks</td></tr><tr><td>FID (pair/unp.) ↓</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID (pair/unp.) ↓</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>C2P P2P</td><td>Cross C2P</td><td>Cross P2P</td></tr><tr><td>MDNM †[IJCAI 2025] [9]</td><td></td><td>-/9.38</td><td>0.872</td><td>0.086</td><td></td><td></td><td></td><td></td><td></td><td>x</td><td>X</td><td>x</td></tr><tr><td>APM †[ICCV 2025] [10]</td><td></td><td>-19.38</td><td>0.901</td><td>0.079</td><td></td><td></td><td></td><td></td><td></td><td>X</td><td>x</td><td>x</td></tr><tr><td>M&amp;MVTO [CVPR 2024] [49]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>OmniVTON † [ICCV 2025] [44]</td><td>140</td><td>7.76/9.62</td><td>0.858</td><td>0.098</td><td></td><td></td><td></td><td></td><td></td><td></td><td>x</td><td>x</td></tr><tr><td>LaDI-VTON [ACM 2023][33]</td><td>2.1</td><td>9.63/12.31</td><td>0.858</td><td>0.098</td><td>10.45/14.05</td><td>0.927</td><td></td><td>0.062</td><td></td><td>X</td><td>x</td><td>x</td></tr><tr><td>DCI-VTON [ACM 2023][13]</td><td>7.2</td><td>5.56/9.03</td><td>0.896</td><td>0.087</td><td>10.50/12.49</td><td>0.918</td><td></td><td>0.068</td><td></td><td>X</td><td>x</td><td>x</td></tr><tr><td>StableVITON [CVPR 2024][22]</td><td>6.8</td><td>7.04/9.71</td><td>0.872</td><td>0.086</td><td>11.57/13.67</td><td>0.919</td><td></td><td>0.066</td><td></td><td></td><td>x</td><td>x</td></tr><tr><td>CAT-DM [CVPR 2024] [45]</td><td>3.5</td><td>7.70/10.26</td><td>0.855</td><td>0.104</td><td>10.26/14.04</td><td></td><td>0.922</td><td>0.058</td><td></td><td></td><td>X</td><td>x</td></tr><tr><td>TPD [CVPR 2024] [43]</td><td>20.1</td><td>10.16/13.07</td><td>0.885</td><td>0.097</td><td>9.50/12.91</td><td>0.956</td><td></td><td>0.048</td><td></td><td></td><td>x</td><td>x</td></tr><tr><td>IDM-VTON [ECCV 2024] [3]</td><td>6.1</td><td>6.00/9.16</td><td>0.881</td><td>0.102</td><td>7.47/11.92</td><td>0.904</td><td></td><td>0.062</td><td></td><td></td><td>X</td><td>x</td></tr><tr><td>ITA-MDT [CVPR 2025][19]</td><td>3.2</td><td>7.66/10.66</td><td>0.882</td><td>0.082</td><td>7.21/11.71</td><td>0.943</td><td></td><td>0.038</td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>CatVITON [ICLR 2025] [4]</td><td>4.4</td><td>8.85/11.36</td><td>0.834</td><td>0.108</td><td>7.37/11.77</td><td>0.933</td><td></td><td>0.051</td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>PromptDresser [ICCV 2025][23]</td><td>3.3</td><td>11.15/13.06</td><td>0.819</td><td>0.151</td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td>x</td></tr><tr><td>Any2Any [ICCV 2025] [15]</td><td>17</td><td>10.04/11.09</td><td>0.839</td><td>0.090</td><td></td><td></td><td></td><td></td><td></td><td></td><td>x</td><td>x</td></tr><tr><td>OmniTry [NeurIPS 2025] [11]</td><td>30</td><td>17.08/19.92</td><td>0.799</td><td>0.202</td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td>x</td></tr><tr><td>FlexTry-On (Ours)</td><td>1.2</td><td>5.51/8.97</td><td>0.861</td><td>0.086</td><td>7.15/11.07</td><td></td><td>0.927</td><td>0.042</td><td></td><td>X</td><td></td><td>x</td></tr><tr><td>FlexTry-On* (Ours)</td><td>1.2</td><td>6.34/9.22</td><td>0.859</td><td>0.093</td><td>7.71/11.58</td><td></td><td>0.923</td><td>0.046</td><td></td><td></td><td></td><td></td></tr></table>

![](images/bd1d3d12970b4c81ba7dd653a331cb2a97e9d2d2c4595839b63aabbef77f6a56.jpg)  
Figure 5. An example of mislabeled pairs in the IMAGDress dataset [40]. The black arrow indicates the correct pairing, whereas the dataset incorrectly associates each person’s image with all garment images, as shown by red dashed arrows.

This loss explicitly enforces the diffusion model to preserve garment-related appearance statistics throughout the synthesis process, ensuring that the generated output remains consistent with the reference garment in terms of color and texture. Unlike reconstruction-oriented objectives used in prior work [6, 38], the proposed GRAM-based loss does not aim to recover exact pixel values; instead, it serves as a task-aligned constraint that directly reflects the essential requirement of IVTON. As discussed in Sec. 4.3.2, this design improves garment appearance preservation under both curated and noisy training, with substantially larger gains under noisy supervision, and helps maintain garment appearance under pose and identity variations. The total loss is formulated as:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { d e n o i s e } + \alpha _ { s t y l e } \mathcal { L } _ { s t y l e } ,\tag{5}
$$

where $\alpha _ { s t y l e }$ is the balance coefficient for the style loss.

## 4. Experiments

## 4.1. Experimental Setting

## 4.1.1. Implementation details

The TASM and DDTM are trained in separate stages. Detailed configurations are provided in Appendix E

## 4.1.2. Dataset and Evaluation Metrics

For multi-scenario training, we combine VITON-HD [2], DressCode [32], DeepFashion [31], IMAGDress [40], and a subset of LRVS-Fashion [27]. We denote the model trained on this full corpus as FlexTry-On\*, while FlexTry-On in Tab. 1 is trained only on VITON-HD and DressCode for fair C2P comparison.

The full corpus contains noisy or weakly aligned garment–person pairs, motivating the task-aligned constraint in Sec. 3.2.2. Details are provided in Appendix F.

Following LaDI-VTON [33], we report FID [17], SSIM [42], and LPIPS [46]. We additionally evaluate inference-seed sensitivity using seeds {0, 1, 2} with the same C2P checkpoint; seed-wise results are reported in Appendix Tab. B(b).

## 4.2. Quantitative and Qualitative Evaluation

## 4.2.1. Baseline Comparison

Tab. 1 compares FlexTry-On with prior diffusion-based IV-TON methods. Publicly released checkpoints are used whenever available. For runtime evaluation, FlexTry-On uses 30 denoising steps, whereas each baseline follows its default inference configuration. All directly reproducible methods are measured with batch size 1 on a single RTX 4090 (24GB VRAM); exceptions and hardware normalization are detailed in Appendix E. All evaluations are conducted at 512×512 resolution. For methods without official

![](images/2b7ba3c57c207e2f49e1a906712b0f28c36813edde379af1b0500221b61cf43c.jpg)  
Human Parsing (Existing methods)  
TASM - Text\_aware Segmentation (Ours)

Figure 6. Cross-domain IVTON results on DressCode [32]. Existing methods [3, 4, 8, 16, 19, 22, 23, 33] typically rely on category-specific human parsing [28] or human-centric segmentation models such as Sapiens [21] (e.g., Voost [25]), which become ambiguous when defining editable regions in cross-domain scenarios. In contrast, our framework employs TASM for text-conditioned segmentation, enabling task-aware mask generation across different garment categories and supporting cross-domain IVTON. The TASM module operates in latent space conditioned on text; masks are interpolated for visualization purposes.  
![](images/7af89374e074a060b654e0df73d8fd8239b9042a1822eab55035407ab4a22a4e.jpg)  
Figure 7. Qualitative comparison with state-of-the-art methods on VITON-HD [2]. All methods are evaluated on the same test inputs using their publicly released checkpoints when available. Conventional human parsing [28] may introduce inaccurate garment boundaries and incomplete editable regions, which can affect downstream virtual try-on performance. In contrast, TASM generates task-aware masks specifically designed for virtual try-on conditioning, resulting in improved garment preservation and reduced visua artifacts. Note that the TASM module operates in latent space conditioned on text; masks are interpolated for visualization purposes.

![](images/65fd33e48e774dcb348dbb27616e0bda705d4a6f986e1abe6e98b2d01c3bcbee.jpg)  
Cloth-to-person IVTON (C2P)

![](images/ecb27cc3df05ff4af27b2d34a648fce57d7d515e5504e996cd21907c187324af.jpg)  
Person-to-Person IVTON (P2P)

Figure 8. Qualitative results in C2P and P2P IVTON setting. The person and clothes images are collected from the Internet. FlexTry-On\* is our proposed method trained on merged uncurated datasets. ITA-MDT\* denotes the fine-tuned ITA-MDT with segmented clothing, trained on our dataset (the full, uncurated dataset).

implementations, we report paper-reported results. For OmniVTON [44], we attempted to reproduce the official implementation but encountered unresolved issues and therefore retain its published quantitative results.

FlexTry-On achieves the best paired/unpaired FID on both VITON-HD and DressCode-Upper among the evaluated methods while maintaining fast inference. Its efficiency comes from a single Reference-Net pass for garment feature extraction, followed by iterative denoising in the Main Net, avoiding repeated garment-side computation across denoising steps. Qualitative results in Fig. 7 further show more consistent garment texture and structure preservation, with fewer arm-region artifacts and clothing inconsistencies.

## 4.2.2. Full Dataset Training

To support multiple IVTON scenarios within a single model, we train FlexTry-On\* on the full combined C2P– P2P corpus. As shown in Tab. 1, FlexTry-On\* supports C2P, P2P, and cross-domain C2P/P2P, providing broader scenario coverage than the compared methods. We evaluate its standard C2P performance on the VITON-HD [2] and DressCode [32] Upper test sets.

Table 2. Ablation study quantitative results on VITON-HD [2] and DressCode [32] Upper in C2P. DDTM denotes the DDTM trained only on VITON-HD and DressCode in C2P, whereas DDTM\* denotes training on the full uncurated combined corpus. Best/second-best results are bold/underlined. Times are measured for one batch-size-1 try-on. For tha image, each segmenter evaluates all 10 segmentation-class conditions listed in Appendix D. TASM batches them in 0.1 s, whereas G-SAM runs GroundingDINO followed by SAM sequentially for each condition (1.9 s per class; 19.0 s total). Denoising takes 1.1 s per try-on.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Seg Param (M Param) ↓</td><td colspan="2">Time / try-on</td><td colspan="3">VITON-HD</td><td colspan="3">DressCode - Upper</td></tr><tr><td></td><td>Seg. (s) ↓ Denoising (s) ↓</td><td>FID (pair/unp.) ↓</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID (pair/unp.) ↓</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>DDTM (Baseline)</td><td>65</td><td>0.1</td><td>1.1</td><td>6.21/9.64</td><td>0.857</td><td>0.091</td><td>8.89/12.84</td><td>0.911</td><td>0.061</td></tr><tr><td>Baseline + TASM</td><td>130</td><td>0.1</td><td>1.1</td><td>5.83/9.02</td><td>0.861</td><td>0.085</td><td>7.12/11.11</td><td>0.925</td><td>0.042</td></tr><tr><td> $\mathbf { \underline { { B a s e l i n e + T A S M } } } + \mathcal { L } _ { s t y l e }$ </td><td>130</td><td>0.1</td><td>1.1</td><td>5.51/8.97</td><td>0.861</td><td>0.086</td><td>7.15/11.07</td><td>0.927</td><td>0.042</td></tr><tr><td>Baseline + G-SAM</td><td>136</td><td>19.0</td><td>1.1</td><td>7.15/10.27</td><td>0.841</td><td>0.107</td><td></td><td></td><td></td></tr><tr><td>DDTM* + TASM</td><td>130</td><td>0.1</td><td>1.1</td><td>6.80/9.42</td><td>0.857</td><td>0.093</td><td>8.22/11.92</td><td>0.923</td><td>0.048</td></tr><tr><td> $\mathrm { D D T M ^ { * } } + \mathrm { T A S M } + \mathcal { L } _ { s t y l e }$ </td><td>130</td><td>0.1</td><td>1.1</td><td>6.34/9.22</td><td>0.859</td><td>0.093</td><td>7.71/11.58</td><td>0.923</td><td>0.046</td></tr></table>

![](images/13040e283162c9317da236006068e0c51247989244435fef0e1f495d808c9ee9.jpg)

![](images/fefe6d4dc250bd2e2ed5e7c18c30ba02c33c23cf8c6edeb52ce54a58a67e104b.jpg)  
Figure 9. Qualitative StreetTryOn results [7]. \* marks our unsuccessful reproduction

Table 3. Results on Street Try-On. †: OmniVTON [44] reported the result
<table><tr><td>Method</td><td>Shop2Street FID↓</td><td>Model2Model FID↓</td><td>Model2Street FID↓</td><td>Street2Street FID↓</td></tr><tr><td>CAT-DM †[CVPR 2024] [45]</td><td>37.48</td><td>-</td><td>–</td><td>–</td></tr><tr><td>IDM-VTON ↑[ECCV 2024] [3]</td><td>42.28</td><td></td><td></td><td></td></tr><tr><td>OmniVTON †[ICCV 2025] [44]</td><td>33.92</td><td>8.98</td><td>33.45</td><td>23.47</td></tr><tr><td>FlexTry-On (Ours)</td><td>32.51</td><td>10.06</td><td>33.07</td><td>32.6</td></tr></table>

Compared with FlexTry-On trained only on VITON-HD and DressCode, FlexTry-On\* shows a modest reduction in standard C2P performance (5.51/8.97 vs. 6.34/9.22 paired/unpaired FID on VITON-HD), while enabling P2P and cross-domain scenarios within the same model. This illustrates the trade-off between optimizing for a single benchmark setting and supporting broader scenarios.

As shown in Fig. 8, FlexTry-On\* supports P2P in both same-domain and zero-shot cross-domain settings. We further compare it with CatVTON and ITA-MDT\*, a retrained variant of ITA-MDT. For cross-domain dress transfer, these methods require manually defined full-garment masks to specify the editable region because their original masking strategies are designed for category-consistent try-on. Even with these manually adapted masks, preserving garment structure and appearance across categories remains challenging. Meanwhile, TASM automatically constructs taskaware editable regions, allowing FlexTry-On\* to produce more coherent results with better garment texture and structure preservation. Additional cross-domain examples, including dress-to-lower, dress-to-upper, and upper-lower-todress transfers in C2P/P2P, are provided in Appendix C.

## 4.2.3. In-the-wild Evaluation

On Street Try-On [7], both FlexTry-On and OmniV-TON are evaluated zero-shot, without training on Street-TryOn. As shown in Tab. 3, FlexTry-On performs better on Shop2Street and Model2Street, whereas the paperreported OmniVTON results are better on Model2Model and Street2Street. Street2Street remains challenging due to uncontrolled pose, illumination, background, and occlusion, especially when the source garment is small or partially visible (Appendix H).

For OmniVTON [44], we retain the published Street-TryOn scores for comparison. However, our evaluation using the official repository did not reproduce the reported performance and produced degraded outputs (Fig. 9). This discrepancy suggests that the results may depend on implementation details or evaluation conditions not captured by the released setup. Thus, these results are paper-reported rather than independently verified, and the reported inference times are not directly comparable across environments.

## 4.3. Ablation Studies

## 4.3.1. Effectiveness of TASM

We compare TASM with conventional human parsing and Grounded-SAM (G-SAM) [30, 37]. Because reliable ground-truth masks are unavailable, particularly for crossdomain cases, and mask accuracy does not directly measure conditioning quality, we evaluate downstream try-on performance rather than Boundary IoU [1].

Replacing TASM degrades quantitative and visual quality (Tab. 2, Fig. 10(A)(D)), producing incomplete garment preservation and retained body regions. Although trained from G-SAM pseudo-labels, TASM provides better taskaware conditioning than G-SAM. Its latent-space design also supports category-aware cross-domain masking without additional supervision; pixel-space alternatives are discussed in Sec. A.2.

## 4.3.2. Effectiveness of the Style-GRAM Loss

Tab. 2 isolates $\mathcal { L } _ { s t y l e }$ under matched training settings. On curated VITON-HD and DressCode, it maintains comparable image quality while slightly improving unpaired FID. Notably, on DressCode-Upper, paired FID changes only from 7.12 to 7.15, whereas unpaired FID improves from 11.11 to 11.07.

![](images/24c295e94fb6944f3540ec5084f518254a2d8db182fa112c6683048a663dc0a1.jpg)  
Figure 10. Qualitative ablation on VITON-HD [2] under curated and full noisy-data training. DDTM and DDTM\* denote the two settings, respectively; “Use G-SAM” replaces TASM with Grounded-SAM under matched-text conditions.

![](images/e2585dce5c8d094e70dfb1ce196ff651125cb8fc9f49c0dd915696c7d3dbe6a8.jpg)  
Figure 11. Qualitative result on VITON-HD, P2P setting. Model\*: retrained model.

Table 4. P2P and C2P results on VITON-HD. <sup>∗</sup> denotes retraining on our full C2P– P2P corpus. Best and second-best results are bold and underlined.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Inf. Time (s) ↓</td><td colspan="3">VITON-HD P2P</td><td colspan="3">VITON-HD C2P</td></tr><tr><td>FID (pair/unp.) ↓</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID (pair/unp.) ↓</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>Stable VITON* [CVPR 2024] [22]</td><td>6.8</td><td>7.01/10.35</td><td>0.876</td><td>0.098</td><td>6.93/9.53</td><td>0.856</td><td>0.091</td></tr><tr><td>CAT-DM* [CVPR 2024] [45]</td><td>3.5</td><td>10.71/12.44</td><td>0.832</td><td>0.153</td><td>8.72/11.32</td><td>0.843</td><td>0.124</td></tr><tr><td>ITA-MDT* [CVPR 2025] [19]</td><td>3.2</td><td>8.02/11.91</td><td>0.881</td><td>0.078</td><td>7.99/11.34</td><td>0.876</td><td>0.094</td></tr><tr><td>FlexTry-On* (Ours)</td><td>1.2</td><td>4.01/10.06</td><td>0.905</td><td>0.044</td><td>6.34/9.22</td><td>0.859</td><td>0.093</td></tr></table>

The benefit becomes clearer under noisy full-data training. On VITON-HD, $\mathcal { L } _ { s t y l \epsilon }$ improves paired/unpaired FID from 6.80/9.42 to 6.34/9.22; on DressCode-Upper, it improves from 8.22/11.92 to 7.71/11.58, while maintaining comparable SSIM/LPIPS. Together with the qualitative improvements in Fig. 10(B,C), these results support $\mathcal { L } _ { s t y l e }$ as a garment-preservation constraint, particularly under noisy supervision.

## 4.3.3. Effect ofExtended Training Data on Prior Methods

To separate model design from data scale, we retrain StableVITON, CAT-DM, and ITA-MDT on the same C2P– P2P corpus, adapting their released checkpoints with segmented person garments while following the original protocols. Methods requiring additional conditioning are excluded. Tab. 4 and Fig. 11 evaluate the resulting checkpoints in both settings.

With matched training data, FlexTry-On achieves the best paired/unpaired FID in both C2P and P2P.Qualitative results in Fig. 11 further support this result, showing more consistent garment appearance and fewer distortions under pose and identity changes. Taken together, these controlled comparisons indicate that the observed gains cannot be explained by the broader training corpus alone, but also reflect the effectiveness of the proposed unified conditioning and generation framework.

## 5. Limitations

FlexTry-On remains challenged by severe occlusion, unusual or oversized garments, and Street-to-Street cases involving uncontrolled pose, illumination, background, and occlusion. The current pipeline assumes a single selected visible person and does not automatically handle crowded or multi-person scenes. The combined training corpus may also inherit demographic and social-media biases.

In addition, FlexTry-On does not consistently outperform all baselines on SSIM/LPIPS, despite achieving stronger FID in several settings. The OmniVTON comparison relies on paper-reported quantitative results because our local reproduction was unsuccessful. Additional discussion and failure analysis are provided in Appendices G and H.

The full training corpus also contains weakly aligned garment–person pairs; $\mathcal { L } _ { s t y l e }$ mitigates their effect on garment preservation but does not eliminate supervision noise.

In practice, FlexTry-On targets single-person try-on applications where a target person and garment are specified before inference; automatic person selection in crowded scenes is outside the current scope.

## 6. Conclusion

We present FlexTry-On, a unified framework for C2P, P2P, cross-domain, and in-the-wild virtual try-on. TASM provides task-aware editable regions, DDTM enables efficient synthesis with a single garment-reference pass, and $\mathcal { L } _ { s t y l e }$ improves garment preservation under noisy supervision. FlexTry-On achieves competitive quality with 1.2-s inference, while matched retraining and qualitative results support its effectiveness across diverse try-on settings.

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (RS-2025-00573160) and by the “Advanced GPU Utilization Support Program” funded by the Government of the Republic of Korea (Ministry of Science and ICT).

## References

[1] Bowen Cheng, Ross Girshick, Piotr Dollar, Alexander C´ Berg, and Alexander Kirillov. Boundary iou: Improving object-centric image segmentation evaluation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15334–15342, 2021. 7

[2] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In Proc. of the IEEE conference on computer vision and pattern recognition (CVPR), 2021. 2, 5, 6, 7, 8, 1, 3

[3] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for authentic virtual try-on in the wild. In European Conference on Computer Vision, pages 206–235. Springer, 2024. 1, 2, 4, 5, 6, 7, 9, 10, 11, 12

[4] Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, Dongmei Jiang, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual try-on with diffusion models. arXiv preprint arXiv:2407.15886, 2024. 2, 5, 6, 9, 10, 11, 12

[5] Fred´ eric Cordier, WonSook Lee, HyeWon Seo, and Nadia´ Magnenat-Thalmann. From 2d photos of yourself to virtual try-on dress on the web. In People and computers XV—interaction without frontiers: Joint proceedings of HCI 2001 and IHM 2001, pages 31–46. Springer, 2001. 1, 2

[6] Aiyu Cui, Daniel McKee, and Svetlana Lazebnik. Dressing in order: Recurrent person image generation for pose transfer, virtual try-on and outfit editing. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14638–14647, 2021. 5

[7] Aiyu Cui, Jay Mahajan, Viraj Shah, Preeti Gomathinayagam, Chang Liu, and Svetlana Lazebnik. Street tryon: Learning in-the-wild virtual try-on from unpaired person images. In Proceedings of the Winter Conference on Applications of Computer Vision, pages 1414–1423, 2025. 7, 4

[8] Phuong Dam, Jihoon Jeong, Anh Tran, and Daeyoung Kim. Time-efficient and identity-consistent virtual try-on using a variant of altered diffusion models. In European Conference on Computer Vision, pages 35–51. Springer, 2024. 1, 2, 6, 9, 10, 11, 12

[9] Chenghu Du, Junyin Wang, Kai Liu, Shengwu Xiong, and Yi Rong. Mask does not matter: a unified latent diffusionenhanced framework for mask-free virtual try-on. In Proceedings ofthe Thirty-Fourth International Joint Conference on Artificial Intelligence, pages 936–944, 2025. 2, 5

[10] Chenghu Du, Shengwu Xiong, and Yi Rong. All parts matter: A unified mask-free virtual try-on framework. In

Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19525–19534, 2025. 2, 5

[11] Yutong Feng, Linlin Zhang, Hengyuan Cao, Yiming Chen, Xiaoduan Feng, Jian Cao, Yuxiong Wu, and Bin Wang. Omnitry: Virtual try-on anything without masks. Advances in Neural Information Processing Systems, 38:132640– 132667, 2026. 2, 5

[12] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576, 2015. 4

[13] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In Proceedings ofthe 31st ACM International Conference on Multimedia, pages 7599–7607, 2023. 2, 5

[14] Rıza Alp Guler, Natalia Neverova, and Iasonas Kokkinos.¨ Densepose: Dense human pose estimation in the wild. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 7297–7306, 2018. 3

[15] Hailong Guo, Bohan Zeng, Yiren Song, Wentao Zhang, Jiaming Liu, and Chuang Zhang. Any2anytryon: Leveraging adaptive position embeddings for versatile virtual clothing tasks. In Proceedings of the IEEE/CVF International Con ference on Computer Vision, pages 19085–19096, 2025. 2, 5

[16] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S Davis. Viton: An image-based virtual try-on network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7543–7552, 2018. 2, 6, 9, 10, 11, 12

[17] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 2, 5, 4

[18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1

[19] Ji Woo Hong, Tri Ton, Trung X Pham, Gwanhyeong Koo, Sunjae Yoon, and Chang D Yoo. Ita-mdt: Image-timestep adaptive masked diffusion transformer framework for imagebased virtual try-on. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28284–28294, 2025. 1, 5, 6, 8, 2, 9, 10, 11, 12

[20] Alexia Jolicoeur-Martineau. The relativistic discriminator: a key element missing from standard gan. arXiv preprint arXiv:1807.00734, 2018. 3

[21] Rawal Khirodkar, Timur Bagautdinov, Julieta Martinez, Su Zhaoen, Austin James, Peter Selednik, Stuart Anderson, and Shunsuke Saito. Sapiens: Foundation for human vision mod els. In European Conference on Computer Vision, pages 206–228. Springer, 2024. 3, 6

[22] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspon dence with latent diffusion model for virtual try-on. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8176–8185, 2024. 2, 5, 6, 8, 9, 10, 11, 12

[23] Jeongho Kim, Hoiyeong Jin, Sunghyun Park, and Jaegul Choo. Promptdresser: Improving the quality and controllability of virtual try-on via generative textual prompt and prompt-aware mask. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16026– 16036, 2025. 1, 2, 5, 6, 9, 10, 11, 12

[24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 3

[25] Seungyong Lee and Jeong-gi Kwak. Voost: A unified and scalable diffusion transformer for bidirectional virtual try-on and try-off. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–11, 2025. 2, 6

[26] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In European Conference on Computer Vision, pages 204–219. Springer, 2022. 2

[27] Simon Lepage, Jer´ emie Mary, and David Picard. Lrvs-´ fashion: Extending visual search with referring instructions. arXiv:2306.02928, 2023. 5, 3

[28] Peike Li, Yunqiu Xu, Yunchao Wei, and Yi Yang. Selfcorrection for human parsing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020. 3, 6, 2, 9, 10, 11, 12

[29] Qi Li, Shuwen Qiu, Julien Han, Xingzi Xu, Mehmet Saygin Seyfioglu, Kee Kiat Koo, and Karim Bouyarmane. Dit-vton: Diffusion transformer framework for unified multi-category virtual try-on and virtual try-all with integrated image editing. arXiv preprint arXiv:2510.04797, 2025. 2

[30] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 3, 7

[31] Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 5, 3

[32] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress Code: High-Resolution Multi-Category Virtual Try-On. In Proceedings ofthe European Conference on Computer Vision, 2022. 5, 6, 7, 1, 2, 3, 8, 9, 10, 11, 12

[33] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In Proceedings of the 31st ACM international conference on multimedia, pages 8580–8589, 2023. 1, 2, 5, 6, 9, 10, 11, 12

[34] Amit Moryossef. Optimizing hand region detection in mediapipe holistic full-body pose estimation to improve accuracy and avoid downstream errors. arXiv preprint arXiv:2405.03545, 2024. 3

[35] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023. 3, 4

[36] Amit Raj, Patsorn Sangkloy, Huiwen Chang, Jingwan Lu, Duygu Ceylan, and James Hays. Swapnet: Garment transfer in single view images. In Proceedings of the European conference on computer vision (ECCV), pages 666–682, 2018. 2

[37] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024. 3, 7

[38] Yurui Ren, Xiaoming Yu, Junming Chen, Thomas H Li, and Ge Li. Deep image spatial transformation for person image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7690–7699, 2020. 5

[39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3, 4, 1

[40] Fei Shen, Xin Jiang, Xin He, Hu Ye, Cong Wang, Xiaoyu Du, Zechao Li, and Jinhui Tang. Imagdressing-v1: Customizable virtual dressing. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6795–6804, 2025. 4, 5, 3

[41] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristicpreserving image-based virtual try-on network. In Proceedings of the European conference on computer vision (ECCV), pages 589–604, 2018. 2

[42] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 5, 4

[43] Xu Yang, Changxing Ding, Zhibin Hong, Junhao Huang, Jin Tao, and Xiangmin Xu. Texture-preserving diffusion models for high-fidelity virtual try-on. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7017–7026, 2024. 5

[44] Zhaotong Yang, Yuhui Li, Shengfeng He, Xinzhe Li, Yangyang Xu, Junyu Dong, and Yong Du. Omnivton: Training-free universal virtual try-on. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16702–16711, 2025. 2, 5, 6, 7

[45] Jianhao Zeng, Dan Song, Weizhi Nie, Hongshuo Tian, Tongtong Wang, and An-An Liu. Cat-dm: Controllable accelerated virtual try-on with diffusion model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8372–8382, 2024. 2, 5, 7, 8

[46] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recogni tion, pages 586–595, 2018. 5, 4

[47] Xujie Zhang, Ente Lin, Xiu Li, Yuxuan Luo, Michael Kampffmeyer, Xin Dong, and Xiaodan Liang. Mmtryon: Multi-modal multi-reference control for high-quality fashion generation. arXiv preprint arXiv:2405.00448, 2024. 3

[48] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4606–4615, 2023. 2

[49] Luyang Zhu, Yingwei Li, Nan Liu, Hao Peng, Dawei Yang, and Ira Kemelmacher-Shlizerman. M&m vto: Multigarment virtual try-on and editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1346–1356, 2024. 2, 5

## Supplementary Material

Table A. Quantitative results of diffusion-based virtual try-on methods on VITON-HD and DressCode upper-body test sets. We report both paired and unpaired FID. TASM-MaskOnly denotes the TASM with output that is the mask only. DDTM refers to the training of the Dual-DiT Tryon Module with only VITON-HD [2] and Dresscode [32] in the C2P tasks. The better performance for each metric is bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">VITON-HD</td><td colspan="3">Dress Code - Upper</td></tr><tr><td>FID (pair/unp.) ↓</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID (pair/unp.)↓</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>DDTM + TASM-MaskOnly + Lstyle</td><td>5.98/9.08</td><td>0.860</td><td>0.086</td><td>7.20/11.08</td><td>0.924</td><td>0.043</td></tr><tr><td>DDTM + TASM + Lstyle</td><td>5.51/8.97</td><td>0.861</td><td>0.086</td><td>7.15/11.07</td><td>0.927</td><td>0.042</td></tr></table>

## A. Additional Ablation Study

## A.1. Empirical Justification of the TASM Design

This ablation investigates a key design choice of TASM. Although TASM predicts both latent appearance representations and segmentation masks, only the masks are used during try-on inference. The latent representations serve solely as an auxiliary supervision signal during training, raising the question about this branch’s necessary.

To answer this question, we construct a simplified variant, denoted as TASM-MaskOnly, which predicts segmentation masks only while retaining the same architecture and training objective. An overview of both designs is shown in Fig. A. As described in the main paper, TASM injects text conditions through multi-head cross-attention and scale-and-shift modulation. Both variants are integrated into the same Dual-DiT framework with the Style-GRAM loss, trained on VITON-HD and DressCode in the C2P setting and evaluated on their respective Upper test sets.

We hypothesize that latent prediction acts as an auxiliary representation-learning task. While mask supervision mainly focuses on region boundaries, reconstructing latent garment representations requires the model to capture garment semantics, appearance, and spatial extent simultaneously. Consequently, the learned features become more informative for segmentation, potentially leading to more accurate mask prediction even though the latent outputs are discarded during inference.

Quantitative results in Tab. A show that TASM consistently outperforms TASM-MaskOnly across nearly all metrics. Qualitative comparisons in Fig. B further reveal that TASM-MaskOnly frequently fails to accurately localize garment regions, particularly for lower-body garments, leading to degraded try-on results. These findings support our hypothesis that auxiliary latent prediction improves region understanding and contributes to higher-quality garment masks, despite not being used during inference.

## A.2. Further discussion about TASM

”Why not light-weight pixel segmentation?” FastSAM, a lightweight SAM, is designed by sacrificing segmentation accuracy for speed relative to SAM. Our empirical advantage over the SAM-based approach further supports TASM’s effectiveness. These results motivate the latentspace formulation of TASM. By focusing on semantically consistent garment regions rather than exact contours, TASM improves robustness, computational efficiency, and compatibility with diffusion-based synthesis, while enabling more flexible scenarios such as cross-domain try-on.

![](images/0676f65ce25cfa2733dbebb63cc664083295b71570b6e728f269ce193e0b7849.jpg)  
Figure A. Overview of the Text-aware segmentation Module for FlexTry-On Framework and its variant. The TASM module receives a latent image with text conditions embedded via the CLIP-Text Encoder from Stable Diffusion 2.0 [39]. The module outputs both the latent representations of the target RGB regions and their corre sponding masks. In TASM, we illustrate the latent representation of the RGB input cloth $\bar { P _ { i c } }$ and its associated mask $M _ { i c } .$ While in TASM-MaskOnly, the output is only the mask $M _ { i c } .$

## B. DressCode Results

This section presents additional results on the DressCode benchmark. Tab. B extends Tab. 1 in the main paper by reporting results on the Lower and Dresses subsets. FlexTry-On achieves the best unpaired FID on Lower and the second-best on Dresses. All baselines are evaluated using their official checkpoints under identical settings. Qualitative comparisons are shown in Fig. E, and Fig. F.

Table B. Additional quantitative results. (a) DressCode lower-body and dresses results; inference time denotes method-specific inference time for one image. FlexTry-On is trained on VITON-HD [2] and DressCode [32] for C2P. Best and second-best results are bold and underlined. (b) Seed-wise VITON-HD FID using the same FlexTry-On checkpoint; the last row reports mean ± standard deviation.  
(a) DressCode lower-body and dresses results  
(b) Seed-wise VITON-HD FID
<table><tr><td></td><td></td><td colspan="3">DressCode – Lower</td><td colspan="3">DressCode – Dresses</td></tr><tr><td>Method</td><td></td><td>Time (s) ↓ FID (pair/unp.) ↓</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID (pair/unp.) ↓</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>LaDI-VTON [ACM 2023] [33]</td><td>2.1</td><td>10.97/16.31</td><td>0.903</td><td>0.061</td><td>12.75/15.80</td><td>0.852</td><td>0.108</td></tr><tr><td>ITA-MDT [CVPR 2025] [19]</td><td>3.2</td><td>8.22/14.03</td><td>0.921</td><td>0.054</td><td>7.79/11.51</td><td>0.878</td><td>0.081</td></tr><tr><td>CatVTON [ICLR 2025] [4]</td><td>4.4</td><td>8.26/13.41</td><td>0.920</td><td>0.037</td><td>7.66/10.80</td><td>0.862</td><td>0.055</td></tr><tr><td>FlexTry-On (Ours)</td><td>1.2</td><td>7.63/13.31</td><td>0.906</td><td>0.056</td><td>7.05/10.89</td><td>0.880</td><td>0.075</td></tr></table>

<table><tr><td>Seed</td><td>Paired FID ↓</td><td>Unpaired FID ↓</td></tr><tr><td>0</td><td>5.51</td><td>8.97</td></tr><tr><td>1</td><td>5.54</td><td>8.99</td></tr><tr><td>2</td><td>5.50</td><td>8.98</td></tr><tr><td>Mean ± std.</td><td>5.52 ± 0.02</td><td>8.98 ± 0.01</td></tr></table>

For lower-body try-on, our approach generates higherfidelity outputs and better preserves surrounding regions such as shoes, socks, and upper garments (examples 1–5 in Fig. E). In comparison, CatVTON still alters clothing regions, while ITA-MDT struggles to maintain consistent structure and realism in the preserved areas.

For dress try-on, prior methods often misplace the dress or alter its style, and may produce artifacts around the neck and hands (examples 1, 2, 4, 5, and 6 in Fig. F). In contrast, our method maintains more accurate garment structure and appearance while reducing such artifacts.

## C. Cross-Domain Qualitative Results

This section presents additional qualitative results on crossdomain virtual try-on, where the garment category of the reference input differs from that of the target person. Such settings are particularly challenging because they violate the category-consistency assumption commonly adopted by existing virtual try-on methods.

One representative scenario involves transferring a dress or full-body garment onto a person wearing separate upper and lower garments. As shown in Fig. G, FlexTry-On produces realistic results in both C2P and P2P settings while preserving fine-grained garment details, including complex patterns, transparency effects, and structural characteristics.

A second scenario considers transferring upper or lower garments onto a person wearing a dress. This requires determining which regions should be preserved and which should participate in the try-on process. By leveraging textaware garment masks, FlexTry-On can effectively handle these ambiguous boundaries and generate coherent results across different category transformations. Examples are shown in Fig. H, Fig. I, Fig. J, and Fig. K.

In contrast, methods relying on conventional human parsing [28] do not natively define task-specific editable regions for cross-domain configurations, particularly dressto-upper and dress-to-lower try-on. For visualization purposes, when evaluating ITA-MDT\* and CatVTON in these settings, we manually mask all clothing regions on the input person to define the inpainting area. Even with this manually adapted masking strategy, FlexTry-On produces more coherent and visually consistent results. These examples demonstrate how the proposed unified framework extends beyond conventional C2P and P2P settings to challenging cross-domain virtual try-on scenarios.

![](images/ae54a34fec889568b574715b5484c68f9570027c03f8b579f78dc2fbe3534e59.jpg)  
Figure B. Ablation study results on VITON-HD [2] to evaluate the effectiveness of the TASM module architecture design in the clean datasets. The DDTM denotes the Dual-DiT Tryon Module, which is trained only on the clean dataset (VITON-HD [2] and DressCode [32]).

## D. Text-aware Segmentation Module Training Configuration

In this section, we provide the handcrafted text conditions used to train TASM and discuss the rationale behind their design. Unlike conventional human parsing, which assigns each pixel to a fixed semantic category, TASM aims to produce task-oriented masks that can support diverse virtual try-on scenarios, including cross-domain settings. Therefore, the predefined text conditions are organized according to functional garment regions rather than dataset-specific semantic labels.

The classes and corresponding text conditions are listed below:

1. Upper body Clothes: ’inner shirt or inner t-shirt or top, or bra’

2. Lower body Clothes: ’shorts, or pants, or skirt, or trousers, or underwear

3. Full body Clothes: ’dresses, or jumpsuits, or overalls’

4. Outer Clothes: ’jackets, or blazers, or outer shirt, or long padding jackets

5. Face and hand: ’face and hand’

6. Shoes: ’shoes’

7. Bag: ’bags’

8. Upper Body Skin (neck and arms): ’upper body skin’

9. Lower Body Skin (legs): ’lower body skin

10. Full Body Skin (neck, arms, and legs): ’full body skin

The region-preservation strategy is designed according to the target garment category. Intuitively, regions unrelated to the target garment should be preserved whenever possible to maintain visual consistency. For example, during upperbody try-on, lower garments and lower-body skin are retained, whereas during lower-body try-on, upper garments and upper-body skin are preserved. For full-body and outerwear try-on, only identity-related and accessory regions, such as faces, hands, shoes, and bags, are retained.

This design is particularly important for cross-domain virtual try-on. Consider the case of transferring an upper garment onto a person wearing a dress. Conventional human parsing typically treats the entire dress as a single semantic region, making it difficult to determine which parts should be preserved and which should participate in the tryon process. In contrast, the proposed text-conditioned decomposition enables TASM to generate task-specific masks, providing more flexible region control across diverse garment categories and cross-domain scenarios.

## E. Implementation Details

This section provides additional implementation details of the proposed framework. For preprocessing, we use MediaPipe [34] for pose keypoint extraction and DensePose [14] for dense body correspondence estimation.

The proposed Dual-DiT Tryon Module (DDTM) is built upon DiT-XL/2 [35], with 28 transformer layers, including 4 decoding layers, a hidden dimension of 1152, and 16 attention heads. All images are resized to 512 × 512. Following prior work, we employ the SD 2.0 VAE [39], which encodes images into latent representations of size 64 × 64 × 4.

During training, we use 1000 diffusion timesteps and perform inference with 30 denoising steps. The model is optimized using a learning rate of $1 \times 1 0 ^ { - 4 }$ and an EMA decay of 0.9999. All remaining optimization settings, including the optimizer and learning-rate scheduler, follow the original DiT configuration [35]. The classifier-free guidance scale is set to 2.0. Unless otherwise specified, reported results correspond to the model trained for 1.5M iterations.

Table C. IVTON Data Statistic
<table><tr><td>Dataset</td><td>Resolutions</td><td>C2P</td><td>P2P</td><td># of images</td></tr><tr><td>VITON-HD [2]</td><td>1024 x 768</td><td></td><td>X</td><td>11 647</td></tr><tr><td>DressCode [32]</td><td>1024 x 768</td><td></td><td>X</td><td>53 792</td></tr><tr><td>DeepFashion [31]</td><td>Varies</td><td>X</td><td></td><td>52 712</td></tr><tr><td>IMAGDress [40]</td><td>Varies</td><td></td><td></td><td>324 857</td></tr><tr><td>LRVS-Fashion [27]</td><td>Varies</td><td></td><td></td><td>500 000</td></tr></table>

Training is conducted on either 8 NVIDIA A6000 GPUs (48GB each) or 4 NVIDIA H200 GPUs (141GB each), depending on hardware availability. Within every ablation pair, the variants with and without $\mathcal { L } _ { s t y l e }$ use the same batch size, optimizer, learning-rate schedule, and number of training iterations; only the loss term is changed.

The Text-Aware Segmentation Module (TASM) is built upon DiT-B/2, consisting of 12 transformer layers, a hidden dimension of 768, and 12 attention heads. TASM is trained for 800K iterations on a single RTX 4090 GPU (24GB) with a batch size of 7.

For inference-time measurement, all methods are benchmarked on an RTX 4090 GPU with 24GB memory whenever possible. Since OmniTry-On and Any2Any Try-On cannot be directly evaluated on this platform, we first measure their runtime on an RTX A6000 GPU with 48GB memory and estimate the corresponding RTX 4090 runtime using the runtime scaling factor of our method measured on both GPUs. The controlled module comparison reports online segmentation and denoising; pose, DensePose, and other preprocessing are identical across the compared variants and therefore excluded.

For the G-SAM baseline [30, 37], we use GroundingDINO-Tiny followed by SAM-B with their publicly released checkpoints, without any additiona fine-tuning. Runtime is measured for one batch-size-1 try-on. Each segmentation method must evaluate all 10 segmentation-class conditions listed above for the same input image. TASM evaluates these conditions together as one batch in 0.1 s. In contrast, the GroundingDINO–SAM pipeline processes the 10 class conditions sequentially, requiring approximately 1.9 s per class, or 19.0 s per try-on. The subsequent DDTM denoising stage takes 1.1 s per try-on for either segmentation method.

## F. Dataset and Evaluation

This section describes the datasets used for training and evaluation. To support diverse virtual try-on scenarios, we construct a unified training corpus from VITON-HD [2], DressCode [32], DeepFashion [31], IMAGDress [40], and a filtered subset of LRVS-Fashion [27]. These datasets provide both C2P and P2P samples across diverse garment categories, poses, identities, and backgrounds.

Cloth

Person

Ours

VITON-HD provides 11,647 training and 2,032 test images at 1024×768. DressCode contains over 50,000 images covering upper-body, lower-body, and dresses. DeepFashion contributes 52,712 images for P2P training. IMAG-Dress contains 324,857 paired samples, while approximately 500,000 downloadable LRVS-Fashion images are retained after filtering. All datasets are standardized into four garment categories, yielding approximately 930,000 training samples in total. Detailed statistics are reported in Tab. C. We report LPIPS [46], SSIM [42], and FID [17]. Lower LPIPS/FID and higher SSIM indicate better performance. To evaluate inference-seed sensitivity, we repeat inference on VITON-HD with seeds {0, 1, 2} using the same checkpoint. Seed-wise paired/unpaired FID is reported in Tab. B(b).

We report both paired and unpaired FID. Paired FID evaluates reconstruction against corresponding ground truth, whereas unpaired FID evaluates unseen garment– person combinations and serves as the primary metric for the standard unpaired IVTON setting.

## G. Discussion

## G.1. Inpainting Mask Strategy

In most prior IVTON methods, the inpainting mask is determined by the try-on garment category. For example, when the target garment is an upper-body item (e.g., a T-shirt or shirt), the mask is typically constructed from the corresponding upper-body garment region of the person together with exposed skin regions, while preserving the hands, face, and hair. A similar strategy is applied to lower-body garments. For full-body garments such as dresses, the editable region generally covers all clothing regions and relevant body areas while preserving identity-related regions such as the face, hair, and hands.

However, such category-dependent masking becomes ambiguous in cross-domain try-on settings. For instance, when a person is wearing a dress, conventional human parsing does not directly determine which part of the dress should be edited when the target garment is an upper- or lower-body item. Consequently, existing category-based masking strategies cannot naturally define the corresponding editable region. TASM addresses this issue by producing task-aware, text-conditioned masks according to the target garment category.

For qualitative comparison in dress-to-upper and dressto-lower settings, we adapt prior methods by treating the editable clothing area as a full-body region, allowing them to generate outputs under their existing masking pipelines. The resulting comparisons are shown in Fig. 6 in the main paper and Figs. H, I, J, and K.

![](images/d71103b0100cd3ebb4a79c47a43700d30de796d9028135a5cceda0c46a9f92df.jpg)  
Figure C. Representative Street2Street failure cases on Street Try-On [7]. FlexTry On may preserve the garment’s appearance while failing to fully reproduce its fine grained structure.

## G.2. Evaluation on Diverse Subjects

To further examine generalization beyond curated fashionstyle examples, we provide additional qualitative results on heterogeneous subjects with varied appearances, body shapes, poses, framing, and background conditions in Fig. D. All examples are generated using the same FlexTry-On\* checkpoint without additional adaptation. These results broaden the qualitative evaluation beyond conventional benchmark-style images, although they do not establish demographic robustness.

## G.3. Evaluation Scope and Reproducibility

Our quantitative evaluation covers standard C2P benchmarks and matched-data C2P–P2P evaluation. Specifically, C2P is evaluated using paired and unpaired VITON-HD and DressCode protocols, while Tab. 4 compares methods retrained on the same C2P–P2P corpus. Because no standardized paired benchmark exists for the category-changing transitions considered in our cross-domain experiments, we restrict this capability to zero-shot qualitative evaluation.

Paired metrics evaluate reconstruction against corresponding targets, whereas unpaired FID evaluates unseen garment–person combinations and remains the primary metric for standard unpaired IVTON. Using the same checkpoint, three inference seeds yield paired/unpaired FID standard deviations of 0.02/0.01, indicating low sensitivity to sampling randomness.

![](images/6b89f758c6f5dd86e2834248ae08b5a4815fe4a0abed016c8c3fae221c039fba.jpg)  
Figure D. Additional qualitative results on heterogeneous subjects with varied ap pearances, body shapes, poses, framing, and backgrounds. All outputs are generated using the same FlexTry-On\* checkpoint without additional adaptation.

## G.4. Efficiency and Computational Footprint

The controlled TASM–G-SAM comparison isolates segmentation as the only differing component. TASM evaluates all 10 text conditions jointly in 0.1 s, whereas G-SAM processes them sequentially in approximately 19.0 s. Both are followed by the same 1.1 s DDTM inference. Shared pose and DensePose preprocessing is excluded from this controlled comparison. Precomputing G-SAM masks would remove this online segmentation cost, but would not affect the downstream conditioning-quality comparison in Tab. 2.

FlexTry-On contains 1.634B parameters across its three DiT modules and 2.058B parameters when including the frozen VAE and CLIP text encoder. At 512 × 512 resolution, batch size 1, and BF16 precision, inference requires 5.15 GB peak allocated GPU memory.

## H. Limitations

Our framework is built upon an inpainting-based formulation and therefore inherits limitations common to diffusionbased virtual try-on methods. In particular, extremely oversized or unconventional garments remain challenging when their desired geometry extends substantially beyond the editable region.

Performance may also degrade under severe occlusion or for highly complex garment structures that are underrepresented in the training data. Street-to-Street transfer is particularly challenging because both the source garment and target person may exhibit uncontrolled pose, illumination, background, garment appearance, and occlusion. Moreover, when the reference garment occupies only a small region of the source image, fine-grained garment cues become less informative, making garment-identity preservation more difficult and increasing the tendency to hallucinate garment details.

Representative failure cases are shown in Fig. C. In these examples, FlexTry-On captures the overall garment appearance but may fail to faithfully preserve fine-grained details and geometry, such as patterns, necklines, sleeve shapes, and silhouettes. These failures are more pronounced when the reference garment is small or partially occluded, suggesting that limited garment evidence remains a key challenge in unconstrained Street-to-Street settings.

The current preprocessing pipeline assumes that a single visible target person has been selected before inference. It does not automatically handle multiple people, person-toperson occlusion, or ambiguous target selection in crowded scenes. In addition, the combined training corpus inherits demographic and social-media collection biases; our current evaluation does not establish uniform performance across demographic groups.

These limitations motivate future work on more robust garment representations under limited visual evidence, automatic multi-person target selection, and broader, demographically balanced virtual try-on datasets.

Person

Cloth

LaDI-VTON

Cat-VTON

ITA-MDT

Ours

![](images/1b5555471551b13ee8d9af07e5abd878c0f89001acfa0bec0f81bfab014d538a.jpg)  
Figure E. Qualitative results on DressCode [32] Lower. Our model (Ours) here is trained only with VITON-HD [2] and DressCode [32].

![](images/7394e41305995bd27ce0ddbbf1b994403615171404497053a35f5f1a13dd0e7d.jpg)  
Figure F. Qualitative results on DressCode [32] Dresses. Our model (Ours) here is trained only with VITON-HD [2] and DressCode [32]

FlexTry-On\*

![](images/1ddbb58ae3908846f9be8745a8e17cab261f5a174526f98db4953937f8da9641.jpg)  
Figure G. Qualitative results on Cross Domain (Upper-Lower to Dress) with C2P and P2P Setting. The dresses are taken from DressCode [32], and the person’s images are taken from the Internet. FlexTry-On\* refers to our training on the combined uncurated datasets. ITA-MDT\* [19] denotes the fine-tuned ITA-MDT with segmented clothing, trained on our dataset.

![](images/2b0ed7e3baff3ab86d63b5900490095aa8a605800ca8fdef495391ad00499de3.jpg)  
Figure H. Qualitative results on Cross Domain (Dress to Upper) with C2P Setting. The Upper Garment and person’s image are taken from DressCode [32]. ITA-MDT\* [19] denotes the fine-tuned ITA-MDT with segmented clothing, trained on our dataset. FlexTry-On\* refers to training on the combined uncurated datasets. Existing methods [3, 4, 8, 16, 19, 22, 23, 33] based on conventional human parsing [28] are limited in supporting flexible masking for cross-domain try-on. In this experiment, we either masked all clothing or used full-body clothing masks for ITA-MDT and CatVTON.

![](images/9323313218eba90552f6fee06609fbd4d6d24d256f359892f534960ab1f363d7.jpg)  
Figure I. Qualitative results on Cross Domain (Dress to Upper) with P2P Setting. The Upper Garment and person’s image are taken from DressCode [32]. ITA-MDT\* [19] denotes the fine-tuned ITA-MDT with segmented clothing, trained on our dataset. FlexTry-On\* refers to training on the combined uncurated datasets. Existing methods [3, 4, 8, 16, 19, 22, 23, 33] typically rely on category-specific masks derived from conventional human parsing pipelines [28]. In cross-domain scenarios, however, the editable regions cannot be uniquely determined from predefined garment categories. For ITA-MDT and CatVTON, we therefore evaluate two representative masking strategies: masking all clothing regions and using full-body clothing masks. In contrast, TASM generates task-aware masks conditioned on the target garment description.

![](images/c4d069713c35fbf9d7a50420da3bc6b5815db56f52f9213cf5f08da7c414ce75.jpg)  
Figure J. Qualitative results on Cross Domain (Dress to Lower) with C2P Setting. The Lower Garment and person’s image are taken from DressCode [32]. ITA-MDT\* [19] denotes the fine-tuned ITA-MDT with segmented clothing, trained on our dataset. FlexTry-On\* refers to training on the combined uncurated datasets. Existing methods [3, 4, 8, 16, 19, 22, 23, 33] typically rely on category-specific masks derived from conventional human parsing pipelines [28]. In cross-domain scenarios, however, the editable regions cannot be uniquely determined from predefined garment categories. For ITA-MDT and CatVTON, we therefore evaluate two representative masking strategies: masking all clothing regions and using full-body clothing masks. In contrast, TASM generates task-aware masks conditioned on the target garment description.

![](images/e03a25285f2f20b4431a79de9b2020eea2e99f1759ed85d7f5724944396340dc.jpg)  
Figure K. Qualitative results on Cross Domain (Dress to Lower) with P2P Setting. The Lower Garment and person’s image are taken from DressCode [32]. ITA-MDT\* [19] denotes the fine-tuned ITA-MDT with segmented clothing, trained on our dataset. FlexTry-On\* refers to training on the combined uncurated datasets. Existing methods [3, 4, 8, 16, 19, 22, 23, 33] typically rely on category-specific masks derived from conventional human parsing pipelines [28]. In cross-domain scenarios, however, the editable regions cannot be uniquely determined from predefined garment categories. For ITA-MDT and CatVTON, we therefore evaluate two representative masking strategies: masking all clothing regions and using full-body clothing masks. In contrast, TASM generates task-aware masks conditioned on the target garment description.