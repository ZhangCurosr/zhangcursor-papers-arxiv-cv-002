# JewelTry: Mask-Free Scale Aware Jewelry Virtual Try-On

Xinlei Niu<sup>1∗</sup>, Peixia Li<sup>2</sup>, Jun Wang<sup>2</sup>, Chenchen Xu<sup>2</sup>, Jiayu Yang<sup>2</sup>, Jing Zhang<sup>1</sup>, Pulak Purkait<sup>2</sup>, Hongdong Li<sup>1,2</sup>

<sup>1</sup>Australian National University, Canberra, Australia;

<sup>2</sup>Amazon, Melbourne, Australia

xinlei.niu@anu.edu.au

## Abstract

Virtual try-on (VTON) enables customers to visualize how fashion products appear when worn and has become an important technology for online shopping. While recent advances have substantially improved garment VTON, jewelry remains a challenging and underexplored category due to its small size, rigid structure, and sensitivity to fine-grained visual details. Realistic jewelry VTON requires not only faithful appearance transfer but also accurate scale and placement relative to the wearer. Existing jewelry VTON methods typically rely on mask guidance, whereas mask-free approaches lack explicit guidance for modeling the product scale. To bridge this gap, we introduce JVTO-Bench, a benchmark dataset for scalefaithful jewelry VTON, providing reference–source–target triplets with real-world product-scale annotations across four major jewelry categories. Building upon this benchmark, we propose JewelTry, a mask-free difusion framework for scaleaware jewelry VTON. JewelTry incorporates a scale adapter that encodes product dimensions into a scale token, enabling the model to learn scale relationships between jewelry items and surrounding human anatomy in-context. To further improve jewelry consistency, we introduce a single-directional condition attention mechanism and an attention refinement loss that preserve both coarse geometry and fine-grained structural details of the reference jewelry. Extensive experiments show that JewelTry achieves a balance among visual fidelity, background preservation, object consistency and scale accuracy, establishing a strong baseline for mask-free, scale-aware jewelry virtual try-on.

## Introduction

Visual understanding plays a critical role in online fashion shopping, where customers rely heavily on product image to evaluate appearance, fit, and purchasing suitability. Virtual try-on (VTON) have emerged as an efective solution for bridging the gap between product presentation and realworld appearance, enabling customers to visualize how fashion items look when worn. While substantial progress has been made in garment VTON, jewelry remains a considerably more challenging category due to its small size, rigid structure, and sensitivity to fine-grained visual details (Miao et al. 2025). Beyond its design, the appeal of jewelry depends on subtle factors such as its scale, placement, and harmony with the wearer. Without realistic try-on imagery, customers must infer these factors from standalone product photos and textual metadata such as dimensions and materials, which can lead to misleading expectations and reduced purchase confidence. Therefore, scale-faithful jewelry VTON is a practically important and technically challenging problem: it requires preserving the detailed appearance of small objects while rendering them at physically plausible sizes and locations on diverse human models.

![](images/572d7f7846d6d0a7ce92c96fa9e0e61112e5e266972e2e0284d7ba4cea4dbc8d.jpg)  
Figure 1: Mask-free scale aware jewelry VTON provides more realistic visualization of product size and appearance.

Recent advances in image generation and editing have substantially improved virtual try-on for garments, accessories, and jewelries (Wu et al. 2025; Li et al. 2025; Zhu et al. 2023; Xu et al. 2025a; Choi et al. 2024; Feng et al. 2025; Miao et al. 2025). In garment VTON, recent methods have moved toward parser-free or mask-free approaches, reducing the need for dense human parsing or manually specified spatial guidance (Zhang et al. 2025a; Feng et al. 2025; Du, Xiong, and Rong 2025). However, mask-free jewelry VTON remains comparatively underexplored. This setting is especially challenging, as the small, rigid, and detailed nature of jewelry demands accurate scale, precise placement, and faithful geometry preservation. Errors in size, shape, or location can make a try-on image visually misleading to customers. Existing jewelry VTON methods typically rely on mask guidance to constrain jewelry position and scale (Miao et al. 2025), while more general try-on frameworks do not explicitly model real-world product scale and instead require the model to infer it implicitly from visual appearance (Feng et al. 2025). These limitations motivate the need for a maskfree framework that can preserve product appearance, placement, and physical scale without relying on manually specified masks or pixel-level spatial guidance.

A major obstacle to advancingjewelry VTON is the lack of suitable benchmarks. Existing VTON datasets primarily target garments or general accessories, and they rarely provide jewelry-specific annotations or reliable real-world productscale information (Hu et al. 2026). As illustrated in Figure 1, scale is particularly critical in jewelry VTON: the perceived realism of a jewelry depends not only on transferring its visual appearance, but also on rendering it at a plausible size relative to the wearer’s anatomy. To bridge this gap, we introduce JVTO-Bench, a benchmark dataset designed specifically for scale-faithful jewelry VTON. JVTO-Bench covers four major jewelry categories and represents each sample to a reference-source-target triplet with accurate real-world product scale. JVTO-Bench provides both training and test splits, enabling model development as well as evaluation. By pairing triplet-based try-on supervision with product-scale information, JVTO-Bench supports training and assessment of jewelry VTON models under mask-free conditions.

Building on JVTO-Bench, we propose JewelTry, a maskfree and scale-aware framework for jewelry VTON. Jewel-Try is designed to address two fundamental challenges in jewelry VTON: (1) rendering jewelry at a physically plausible scale and (2) preserving the fine-grained details of the jewelry. To model real-world scale without relying on manually specified masks or explicit geometric supervision, we introduce a scale adapter that encodes product dimensions, measured in inches, into a scale token. The scale token enables the model to learn relative scale relationships between jewelry items and surrounding anatomical regions, such as ears, fingers, wrists, and necks, resulting in more realistic and scale-consistent try-on results. To improve jewelry fidelity, we further introduce a single-directional condition attention mechanism and an attention refinement loss. The single-directional condition attention prevents reference condition tokens from being contaminated by noisy latent features during the difusion process, thereby preserving the coarse structure of the reference jewelry. Building upon this, the attention refinement loss explicitly supervises the interaction between jewelry condition tokens and the target try-on region, encouraging the model to focus on relevant object areas and improving fine-grained appearance consistency. Together, these components enable JewelTry to faithfully preserve both the physical scale and visual structure of jewelry in a fully mask-free setting. In summary, our main contributions are threefold:

• JVTO-Bench Dataset. A benchmark dataset for the mask-free jewelry VTON task, featuring high-quality reference-source-target image triplets as well as productscale annotations across four major jewelry categories.

• JewelTry. A difusion framework designed for jewelry consistency and scale awareness in a mask-free manner.

• We conduct extensive experiments demonstrating that JewelTry establishes a state-of-the-art baseline for scaleaware and mask-free jewelry VTON.

## Related Work

Object-guided Image Editing and Generation. Objectguided image editing and generation aims to modify or synthesize an image according to a given reference object, while preserving scene context and visual realism. The generated object is expected to match the reference in appearance, structure, and semantics, and be naturally integrated into the target image (Tan et al. 2025; Zhang et al. 2025b; Chen et al. 2026, 2025; Xu et al. 2026a; She et al. 2025; Cheng et al. 2025; Zhang et al. 2026; Shin et al. 2025). Virtual tryon is a specialized task of object-guided editing, where the reference object is a wearable item such as clothing and accessories, and the target image is a person image. Compared with general object-guided editing, VTON imposes stronger constraints on geometric alignment, scale consistency, and interaction with human body regions, since the generated object must be both realistic and correctly positioned with proper proportions.

Garment Virtual Try-On. Garment VTON aims to generate a realistic image of a person wearing a target garment by transferring clothing appearance from a reference image to the person image. It has evolved from early U-Net-based reconstruction methods (Issenhuth, Mary, and Calauzènes 2019) to difusion-based frameworks with substantially improved generation quality and garment realism. Recent methods enhance garment authenticity through specialized adapters and semantic feature injection (Wang et al. 2024; Choi et al. 2024), while difusion transformer-based approaches further improve scalability and the modeling of complex physical deformations (Lee and Kwak 2025; Mei and Ni 2026) with inpainting-based masking strategies for paired data generation (Jiang et al. 2024). More recently, garment VTON has shifted from parser-based pipelines to mask-free paradigms to enable realistic try-on without explicit segmentation masks (Niu et al. 2024; Du, Xiong, and Rong 2025; Zhang et al. 2025a; Du et al. 2025; Li et al. 2025; Kwon et al. 2026). Beyond garments, omni-style VTON further extends virtual try-on to accessories and diverse object categories through unified mask-free frameworks (Feng et al. 2025; Wang et al. 2025; Zeng et al. 2026).

Jewelry and Ornament Virtual Try-On. Despite the progress in garment VTON, high-end jewelry remain challenging due to their rigid shapes, details, and complex topologies (Chang and Lekena 2024). ShiningYourself (Miao et al. 2025) pioneers mask-guided ornament VTON, where a target mask is required to explicitly specify the placement region and object scale during generation. SparklingTogether (Xu et al. 2026b) further extends the mask-guided single jewelry VTON to a mask-guided multi-accessory VTON framework. Recent omni-style methods (Feng et al. 2025; Wu et al. 2025) support mask-free jewelry VTON, but they are not specifically designed for enhancing jewelry scale-faithful, where accurate scale control and structural consistency are critical.

## JVTO-Bench Dataset

Accurate product-scale information is essential for the maskfree jewelry virtual try-on task, as it directly afects size realism and visual plausibility. However, existing jewelry VTON research lacks publicly available datasets with reliable scale annotations. To address this gap, we introduce JVTO-Bench, a benchmark dataset for scale-faithful jewelry VTON. JVTO-Bench covers four major categories: rings, earrings, necklaces, and bracelets, and contains over 23K high quality and diverse jewelry fashion images, ranging from well-posed shop images to unconstrained consumer photos. Each sample is organized as a triplet J, P, T with product scale in inches, where J is the reference jewelry image, P is the source try-of image, and T is the target try-on image. We position JVTO-Bench as a key contribution of this work and a comprehensive resource for futurejewelry VTON research. As illustrated in Figure 2, our dataset construction pipeline consists of four stages:

![](images/f56199f6fe3b07311c353521c5c25c8675dea4d48655cb2719f0b81c1557c4d8.jpg)  
Figure 2: Overview of the JVTO-Bench dataset preparation pipeline.

Stage 1: Data Collection. To construct a diverse dataset, we collect jewelry data from a representative online shopping platform across major markets, including the UK, the US, and India, ensuring broad coverage ofjewelry styles and regional preferences. Each item contains one main display image and several auxiliary images. These auxiliary images may include valid human try-on images as well as product-only, detail, lifestyle, or scale-informed images, yielding approximately 3M items in the raw pool.

Stage 2: Product Scale Retrieval. Although sellers often provide product scale information, it is often noisy or inaccurate. We use Claude-Sonnet-4 to extract scale annotation directly from scale-informed images followed by human validation. See our supplementary material for more details.

Stage 3: Data Cleaning. We apply strict filtering to obtain high-quality product-person pairs. For each reference image, we require exactly one jewelry product on a clean white background. For each try-on image, the jewelry must be clearly visible and correctly matched to the reference. We use Claude-Sonnet-4 and Qwen-VL-3 for automatic filtering, followed by expert-level manual verification to further clean the mismatched samples and ensure quality.

Stage 4: Annotation and Post-processing. We convert filtered product-person pairs into training-ready triplets. The reference jewelry image is tightly cropped around the product region with an additional margin 25% using Grounding DINO (Liu et al. 2024). We also use Claude-Sonnet-4 to generate captions for try-on images, following the template: “The person from image 1 is wearing/holding {jewelry type} from image 2 {optional position descriptor}.” These captions describe the interaction between the person and jewelry and can serve as text prompts for training. Since source images without the jewelry are typically unavailable, we synthesize try-of source images by removing the jewelry. Existing segmentation-and-inpainting pipelines (Feng et al. 2025) are less efective for jewelry removal, as residual shadows often remain, causing visible artifacts and degraded realism. We further refine the try-of images using the output of Qwen-Object-Remover<sup>1</sup> and the SSIM-based diference map between the object-removed result and the original target image, which helps localize residual jewelry and shadow artifacts for inpainting. See our supplement for more details.

## Method

We now present the technical details ofour method, JewelTry. Figure 3 provides the framework overview, which consists of three key components: (1) a scale adapter that projects numerical scale prompts into scale tokens; (2) a single-directional condition attention mechanism that prevents reference feature collapse and preservesjewelry structure consistency; and (3) an attention refinement loss that explicitly encourages the model to focus on fine-grained details during training.

## Scale Adapter

The first challenge is enabling the model to perceive product scale, which is essential for generating realistic try-on results with accurate size and proportion. As in Figure 3 (right), we first incorporate scale information into the text prompt of a vision-language model (VLM), leveraging its reasoning capability to interpret physical measurements and translate them into semantic scale-aware guidance. To further make the model aware of scale inputs, we involve a scale adapter that encodes product’s physical measurement in inches into a scale token (Chen et al. 2026; Xu et al. 2025b). The scale token is subsequently concatenated with the text token produced by the VLM. The scale adapter is motivated by the need to model the contextual relationship between real-world jewelry dimensions and human anatomy. Since reference product images and target person images are typically captured under diferent camera settings and viewpoints, recovering exact camera parameters or performing explicit geometric alignment is non-trivial. Instead of enforcing direct absolute scale supervision, the scale adapter provides an implicit condition that encourages the model to learn relative scale relationships between jewelry size and nearby anatomical regions.

MMDiT Block with single direction condition attention  
![](images/f25cb168f63c54c0cef2a7de07bce9b3a699bb7042bf7305694897aeb179626e.jpg)

![](images/7160ecb540bb338bfc67a53c33450140590cb05951125a8bafa950c9a8c50c0c.jpg)  
Figure 3: Overview of the JewelTry framework. Snowflake and fire icons denote frozen and trainable parameters, respectively.

![](images/27b959700604688ff6c4a9e05b6afbeddea6c98989afd1e3f9af815ce3d3d174.jpg)  
Figure 4: Examples of condition collapse problem, where the results fail to preserve the rigid structure in reference images.

## Single Directional Condition Attention

Jewelry items typically have rigid structures and fixed topology, making structural consistency a critical challenge in VTON (Miao et al. 2025). The generated result must faithfully preserve the jewelry structure and fine-grained details from the reference image. In bidirectional self-attention, noisy latent tokens and condition tokens attend to each other symmetrically. While this facilitates information exchange, it can also destabilize the conditioning representation: finegrained jewelry cues encoded from the reference image may be corrupted by noisy latent features. We refer to this as condition collapse (Figure 4), where the model fails to consistently preserve the structural details of the reference jewelry.

To mitigate condition collapse, we introduce single direction condition attention in MMDiT to preserve a stable jewelry conditioning. Standard bidirectional self-attention allows noisy latent tokens and condition tokens to update each other symmetrically. While efective for general information exchange, this design is suboptimal for jewelry VTON: the reference jewelry tokens encode rigid topology and finegrained product details, and should remain a reliable source of structural guidance rather than being updated by noisy latent features. We therefore block the reverse attention path from noisy latent tokens to jewelry tokens, while preserving the forward guidance from jewelry tokens to the noisy latent representation. In this way, latent tokens can still attend to the reference jewelry and receive structural guidance, whereas jewelry tokens are protected from noise-dependent interference. This asymmetric design maintains the stability of the jewelry representation during training, reducing condition collapse issue and improving structural fidelity. We provide additional discussion in our supplementary material.

![](images/3cb67b18e7f315bea6a65f601588fed086caef744b0e2083ec25687487f07a3b.jpg)  
Figure 5: Attention mask from later MMDiT blocks are supervised by the GT mask to preserve fine-grained detail.

Let M denote the binary attention mask that controls the allowable interactions among condition and latent branches. We concatenate the query, key, and value tokens as

$$
\begin{array} { c } { Q = [ Q _ { \mathrm { s } , \mathrm { t } } ; Q _ { \mathrm { n o i s e } } ; Q _ { \mathrm { P } } ; Q _ { \mathrm { J } } ] , \quad K = [ K _ { \mathrm { s } , \mathrm { t } } ; K _ { \mathrm { n o i s e } } ; K _ { \mathrm { P } } ; K _ { \mathrm { J } } ] , } \\ { V = [ V _ { \mathrm { s } , \mathrm { t } } ; V _ { \mathrm { n o i s e } } ; V _ { \mathrm { P } } ; V _ { \mathrm { J } } ] , } \end{array}
$$

where s,t denotes scale and text tokens, P denotes personimage tokens, and J denotes jewelry-image tokens. For token positions i and j, the masked attention score is computed as $\begin{array} { r } { A _ { i j } = \frac { Q _ { i } K _ { j } ^ { \intercal } } { \sqrt { d _ { k } } } + M _ { i j } } \end{array}$ . As illustrated in Figure 3 (right), our single direction condition attention mask is defined as

$$
M _ { i j } = { \left\{ \begin{array} { l l } { - \infty , } & { i \in \mathbf { J } , \backslash j \in { \mathrm { n o i s e } } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{1}
$$

Although attention masking in MMDiT has been explored for conditional generation and editing (Cai et al. 2025; Wang et al. 2025; Shen et al. 2025; Zhang et al. 2025b; Chen et al. 2026), existing methods mainly use it for spatial control, condition injection, or controllable generation. In contrast, our masking strategy targets condition collapse in jewelry VTON. Specifically, it prevents noisy latent tokens from interfering with fine-grained jewelry condition tokens while preserving the guidance from jewelry tokens to the latent branch, thereby improving jewelry structural consistency.

## Training Loss

Attention refinement loss. To further improve the structural and fine-grained jewelry consistency in the try-on results, we introduce an attention refinement loss. We extract a predicted jewelry soft mask from the attention map between the reference query $Q _ { \mathrm { r e f } }$ and the noisy latent key $\bar { K } _ { \mathrm { n o i s e } }$ , and supervise it with the ground-truth jewelry mask. We observe that, in the later MMDiT blocks, this attention map naturally highlights the target jewelry region and is strongly correlated with the final generated result (see supplementary for more details). By explicitly aligning this attention-derived soft mask with the ground-truth object region, the proposed loss encourages more accurate spatial correspondence between the reference jewelry and the generated try-on image, leading to improved object placement, scale fidelity, and fine-grained structural preservation. As illustrated in Figure 5, the attention refinement loss ${ \mathcal { L } } _ { \mathrm { a t t n } }$ is defined as

$$
\mathcal { L } _ { \mathrm { a t t n } } = \mathcal { L } _ { \mathrm { B C E } } ( A _ { i } ^ { ( k ) } , G ) + \mathcal { L } _ { \mathrm { D i c e } } ( A _ { i } ^ { ( k ) } , G )\tag{2}
$$

Where $A _ { i } ^ { ( k ) }$ denotes the soft attention mask extracted from the k-th attention block at the denoising step $i . A _ { i } ^ { ( k ) }$ is computed by averaging the cross-attention weights across all attention heads, aggregating them over the conditioning token dimension, and finally applying min–max normalization to obtain a continuous mask with values in [0, 1]. G denotes the ground-truth mask.

$$
{ \mathcal { L } } _ { \mathrm { B C E } } ( A , G ) = - \frac { 1 } { N } \sum _ { p = 1 } ^ { N } [ G _ { p } l o g A _ { p } + ( 1 - G _ { p } ) l o g ( 1 - A _ { p } ) ]
$$

$$
\mathcal { L } _ { \mathrm { D i c e } } ( A , G ) = 1 - \frac { 2 \sum _ { p = 1 } ^ { N } A _ { p } G _ { p } + c } { \sum _ { p = 1 } ^ { N } A _ { p } + \sum _ { p = 1 } ^ { N } G _ { p } + c }
$$

N represents the number of pixel and c is a small constant.

Object consistency loss. We also adopt the velocity prediction loss (Lipman et al. 2023) and introduce an objectregion loss that emphasizes the prediction accuracy within the target jewelry region:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { o b j } } = \mathbb { E } _ { \boldsymbol { x } _ { 0 } \sim \mathcal { D } , \boldsymbol { x } _ { 1 } , t } \left[ \left\| \boldsymbol { G } \odot \left( \boldsymbol { v } _ { \theta } ( \boldsymbol { x } _ { t } , t , c ) - \boldsymbol { v } _ { t } \right) \right\| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{3}
$$

where G denotes the ground-truth jewelry mask, $v _ { \theta } ( x _ { t } , t , c )$ is the predicted velocity, and $v _ { t }$ is the ground-truth velocity.

Overall objective. In contrast to previous jewelry-specific objectives, which explicitly improve product consistency, we also employ the standard velocity prediction loss $\mathcal { L } _ { \mathrm { M S E } }$ as in Wu et al. (2025). This loss provides global supervision over the entire try-on image and encourages the model to reconstruct the overall target distribution, including the person identity, skin tone, clothing, and background context. While ${ \mathcal { L } } _ { \mathrm { a t t n } }$ and $\mathcal { L } _ { \mathrm { o b j } }$ focus on preserving the structure and details of jewelry, L<sub>MSE</sub> serves as the primary constraint at the imagelevel to maintain the consistency of the person and the overall scene. The overall training objective is defined as:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { M S E } } + \mathcal { L } _ { \mathrm { o b j } } + \mathcal { L } _ { \mathrm { a t t n } }\tag{4}
$$

## Experiment and Results

## Evaluation metrics

We evaluate jewelry virtual try-on performance from three perspectives: (1) image fidelity; (2) background preservation;

and (3) object consistency. We measure the fidelity of generated images using Fréchet Inception Distance (FID), computed between the generated try-on images and the groundtruth target images. FID evaluates the distributional similarity between generated and real images and serves as a measure ofimage realism. Following (Feng et al. 2025), we assess how well the non-jewelry regions are preserved after try-on. We compute $\mathrm { D I N O _ { p } }$ (Zhang et al. 2022), $\mathrm { L P I P S } _ { \mathrm { p } }$ (Zhang et al. 2018), and $\mathrm { S S i m _ { p } }$ (Wang et al. 2004) on the maskedout non-jewelry regions between the generated image and the target image. To evaluate whether the generated jewelry faithfully matches the reference jewelry, we compute $\mathrm { D I N O } _ { \mathrm { R e f } }$ and $\mathrm { C L I P _ { R e f } }$ between the cropped generated jewelry region and the reference jewelry image. In addition, we compute $\mathrm { D I N O } _ { \mathrm { T a r } }$ and $\mathrm { C L I P _ { T a r } }$ between the cropped generated jewelry region and target jewelry region. These metrics capture both structural and semantic consistency of the generated jewelry. To assess scale and placement accuracy, we compute the IoU between the predicted jewelry region and the corresponding ground-truth jewelry region in the target image. A higher IoU indicates better alignment in both object scale and spatial placement. We also report scale error rate by calculating the log ratio between predict jewelry and ground-truth jewelry via bounding box, the lower scale error rate indicating the better scale faithfulness.

## Comparison

As we are the first to focus on mask-free scale-aware jewelry VTON, no prior method shares our exact setting. To enable a comprehensive and fair comparison, we evaluate against five baselines spanning two groups. Zero-shot baselines use of-the-shelf models without any adaptations: (1) Qwen-Image-Edit (Wu et al. 2025), an image-editing model supporting multi-image prompts; (2) OmniTry (Feng et al. 2025), a mask-free VTON model supporting jewelry categories; (3) Any-to-any TryOn (Guo et al. 2025), a mask-free garment VTON model with free-text prompts; and (4) InsertAnything (Song et al. 2026), a mask-guided object-insertion model. Trained-on-bench baselines are fine-tuned on the JVTO-Bench training split under the same dataset as Jewel-Try: (5) Qwen-JVTON, a Qwen-Image-Edit model LoRAfine-tuned on JVTO-Bench with scale information embedded into the text prompt, which serves as our most directly comparable baseline. We exclude ShiningYourself (Miao et al. 2025) and SparklingTogether (Xu et al. 2026b), due to the lack of publicly available training data and implementation details, which would preclude a fair comparison.

Dataset. We evaluate JewelTry on two datasets: the JVTO-Bench test split and the jewelry subset of OmniTry-Bench (Feng et al. 2025). The JVTO-Bench contains 375 samples for evaluation, each consisting of a person image, a reference jewelry image, and a ground truth, with scale annotation and caption. It covers four jewelry categories, including rings, earrings, necklaces, and bracelets, with 59, 104, 106, and 106 product-scale annotations, respectively. In addition, we construct a jewelry subset from OmniTry-Bench, comprising 300 paired object-person samples in total without ground-truths and scale annotations. This subset includes 15 independent person images and 5 clean-background reference images for each jewelry category.

![](images/8abeb1923c0b34354cab24f42ee119211eaf878bb88e5e821a7cbf30b76de310.jpg)

Figure 6: Qualitative comparison on JVTO-Bench test split cross four jewelry categories.
<table><tr><td></td><td colspan="2"></td><td>|Fidelity</td><td colspan="3">Background preservation</td><td colspan="6">Object consistency</td></tr><tr><td>Method</td><td colspan="2">|Mask-free|Dataset</td><td>FID↓</td><td colspan="6"> $\mathrm { | D I N O _ { p } \uparrow L P I P S _ { p } \downarrow S S I M _ { p } }$  1  $\cdot | \mathrm { D I N O } _ { \mathrm { R e f } } \uparrow \mathrm { D I N O } _ { \mathrm { T a r } }$  ←  ${ \mathrm { C L I P } } _ { \mathrm { R e f } }$ </td><td>↑ CLIPTar ↑ IoU↑ ScaleErr↓</td><td></td></tr><tr><td>Ground Truth</td><td></td><td>|JVTO-Bench</td><td></td><td></td><td></td><td></td><td>0.567</td><td></td><td>0.799</td><td></td><td></td><td></td></tr><tr><td>Qwen-Image-Edit</td><td></td><td>JVTO-Bench</td><td>39.72</td><td>0.926</td><td>0.249</td><td>0.805</td><td>0.617</td><td>0.749</td><td>0.820</td><td>0.891</td><td>0.469</td><td>0.263</td></tr><tr><td>ÖmniTry</td><td></td><td>JVTO-Bench</td><td>31.23</td><td>0.953</td><td>0.126</td><td>0.879</td><td>0.563</td><td>0.761</td><td>0.773</td><td>0.899</td><td>0.538</td><td>0.256</td></tr><tr><td>Any2AnyTryOn</td><td>√</td><td>JVTO-Bench</td><td>73.64</td><td>0.368</td><td>0.755</td><td>0.544</td><td>0.511</td><td>0.457</td><td>0.771</td><td>0.785</td><td>0.136</td><td>1.158</td></tr><tr><td>InsertAnything</td><td>X</td><td>JVTO-Bench</td><td>54.47</td><td>0.919</td><td>0.156</td><td>0.869</td><td>0.579</td><td>0.660</td><td>0.805</td><td>0.868</td><td>0.409</td><td>0.397</td></tr><tr><td>Qwen-JVTON</td><td>√</td><td>JVTO-Bench</td><td>32.62</td><td>0.950</td><td>0.128</td><td>0.867</td><td>0.525</td><td>0.759</td><td>0.774</td><td>0.898</td><td>0.591</td><td>0.208</td></tr><tr><td>JewelTry (Ours)</td><td>√</td><td>JVTO-Bench</td><td>30.49</td><td>0.959</td><td>0.121</td><td>0.873</td><td>0.565</td><td>0.792</td><td>0.801</td><td>0.914</td><td>0.658</td><td>0.169</td></tr><tr><td>Qwen-Image-Edit</td><td>√</td><td>[OmniTry-Bench</td><td></td><td>0.926</td><td>0.223</td><td>0.685</td><td>0.556</td><td></td><td>0.781</td><td></td><td></td><td></td></tr><tr><td>OmniTry</td><td></td><td>OmniTry-Bench</td><td></td><td>0.992</td><td>0.017</td><td>0.960</td><td>0.517</td><td></td><td>0.761</td><td></td><td></td><td></td></tr><tr><td>Any2AnyTryOn</td><td></td><td>OmniTry-Bench</td><td></td><td>0.539</td><td>0.636</td><td>0.429</td><td>0.345</td><td></td><td>0.708</td><td></td><td></td><td></td></tr><tr><td>InsertAnything</td><td>× √</td><td>OmniTry-Bench</td><td></td><td>0.993</td><td>0.012</td><td>0.987</td><td>0.502</td><td></td><td>0.760</td><td></td><td></td><td></td></tr><tr><td>Qwen-JVTON</td><td></td><td>OmniTry-Bench</td><td></td><td>0.996</td><td>0.034</td><td>0.928</td><td>0.471</td><td></td><td>0.717</td><td></td><td></td><td></td></tr><tr><td>JewelTry (Ours)</td><td></td><td>OmniTry-Bench</td><td></td><td>0.997</td><td>0.033</td><td>0.929</td><td>0.540</td><td></td><td>0.762</td><td></td><td></td><td></td></tr></table>

Table 1: Quantitative evaluations between our method and other baselines. First , Second, and Third indicate the best, second best, and third best performance respectively.

Qualitative Comparison. Figure 6 compares try-on results across jewelry categories with diferent structures and wearing regions. The general image-editing model Qwen-Image-Edit fails to preserve the source image consistency, demonstrating the dificulty of directly applying general editing models to jewelry VTON. In contrast, OmniTry and Qwen-JVTON better preserve the source image, but tend to lose jewelry details and structural consistency. Any2AnyTryOn is primarily designed for garment VTON and does not generalize well to of-the-shelf jewelry objects. InsertAnything, as a mask-guided object insertion method, achieves relatively strong object preservation, but produces weaker integration with the human body. In contrast, our method has better performance on preserving jewelry scale and structural details, which produces more realistic and visually coherent jewelry virtual try-on results.

Quantitative Comparison. Table 1 reports the results of JVTO-Bench and OmniTry-Bench. In JVTO-Bench, Jewel-

Try achieves the best score among all baselines compared on FID, $\mathrm { D I N O _ { p } , L P I P S _ { p } , D I N O _ { T a r } , C L I P _ { T a r } , }$ IoU and ScaleErr, which indicate JewelTry achieves better fidelity, background preservation, jewelry object consistency, and scale accuracy. Although JewelTry does not achieve the highest $\mathrm { D I N O } _ { \mathrm { R e f } }$ and $\mathrm { C L I P _ { R e f } }$ scores, its scores remain close to the ground truth. Notably, these scores approaching 1 are not necessarily desirable, as they may indicate near-direct copying of the reference image rather than realistic re-rendering (See the bracelet and ring examples in Figure 6). Targetsupervised metrics and jewelry scale are inapplicable on OmniTry-Bench dataset. We drop the scale tokens for Jewel-Try during inference and obtains the best $\mathrm { D I N O _ { p } }$ and thirdbest $\mathrm { L P I P S _ { p } / S S I M _ { p } }$ . Qwen-Image-Edit leads the referencebased object-consistency score there, but at the cost of person preservation. Overall, JewelTry achieves a balance among visual fidelity, background preservation, object consistency and scale accuracy. We provide more experimental results and implementation details in our supplementary material.

<table><tr><td>Method</td><td> $| \mathrm { D I N O } _ { \mathrm { p } } \uparrow$   $\mathrm { L P I P S } _ { \mathrm { p } } \downarrow$ </td><td>_  $\mathrm { D I N O } _ { \mathrm { r e f } }$ </td><td>↑  $\mathrm { D I N O } _ { \mathrm { t a r } }$ </td><td>↑IoU↑</td></tr><tr><td>Qwen-JVTON</td><td>|0.950</td><td>0.128</td><td>|0.525</td><td>0.759 0.591</td></tr><tr><td>+SDAttn</td><td>0.953</td><td>0.121 0.559</td><td>0.760</td><td>0.589</td></tr><tr><td>+SDAttn &amp; SA</td><td>0.958</td><td>0.124</td><td>0.545 0.772</td><td>0.634</td></tr><tr><td> $+ \mathrm { S D A t t n } \& \mathrm { S A } \& \mathcal { L } _ { \mathrm { a t t n } }$ </td><td>|0.957</td><td>0.122</td><td>|0.559 0.788</td><td>0.635</td></tr><tr><td>JewelTry</td><td>|0.959</td><td>0.121</td><td>|0.565</td><td>0.792 0.658</td></tr></table>

Table 2: Ablation study for JewelTry on JVTO-Bench, where SA stands for scale adapter and SDAttn stands for single direction condition attention.  
![](images/f50f418433ae13920642fccbaaba02e02835d680cbf9c295b9eeaee2dd8f25ad.jpg)  
Figure 7: The visual comparisons of our models with diferent module configurations for jewelry consistency.

## Ablation study

We conduct ablation studies by progressively adding each proposed component to the baseline, Qwen-JVTON (Table 2), and assess whether each component improves the specific aspect it targets. Introducing single-directional condition attention (SDAttn) primarily targets reference consistency and yields the largest single gain in $\mathrm { D I N O } _ { \mathrm { r e f } } ,$ indicating better preservation of reference jewelry structure. Since Qwen-JVTON is trained with scale information embedded in the VLM’s inputs, it exhibits a degree of scale awareness. Adding the scale adapter further enhances scale-awareness in the model, and accordingly increases IoU; We observe a small decrease in $\mathrm { D I N O } _ { \mathrm { r e f } }$ and small increases in $\mathrm { D I N O } _ { \mathrm { t a r } } ,$ which is expected as inference variance. The ${ \mathcal { L } } _ { \mathrm { a t t n } }$ further improves target-region consistency by sharpening alignment between jewelry condition tokens and the target try-on region. Finally, adding $\mathcal { L } _ { \mathrm { o b j } }$ yields the full JewelTry model, which attains the best on object-consistency metric while maintaining background preservation. In addition, $\mathcal { L } _ { \mathrm { o b j } }$ enhances the correctness ofjewelry placement with an increased IoU. Figure 7 provides a qualitative comparison which provides a better interpretation. Without the attention refinement loss, the model fails to fully preserve fine-grained jewelry details, leading to slight structural changes in the earring. Removing the singledirectional condition attention further degrades coarse-level object consistency, producing jewelry with distorted overall structure. Similarly, removing the object consistency loss weakens reference preservation and results in noticeable deviations from the earring structure. In contrast, the full JewelTry model better maintains both the coarse geometry and fine-grained details of the reference jewelry, demonstrating the efectiveness of these components for object consistency.

## Discussion and limitation

As the first mask-free scale-aware jewelry VTON method, JewelTry uses a scale token as implicit condition to model the relationship between jewelry and human anatomy. While this improves scale consistency, it does not guarantee precise absolute scale control. Repeated inference with the same jewelry image, person image, and scale input may produce slight variations in results (Figure 8). Moreover, JewelTry mainly provides relative rather than absolute scale control: increasing the scale token consistently enlarges the generated jewelry, but the visual size may not exactly match the extreme large or small dimension, e.g., a 6-inch earring may not be rendered at a perceptually exact 6-inch scale (Figure 9). This limitation may arise from data bias, as extreme scales are rare in real-world jewelry distributions, and from the inherent dificulty of mapping physical dimensions to image-space geometry without explicit geometric supervision.

Result 1  
Result 2  
Result 3  
![](images/d732470fc21b9d2a97964a12fbdd4d599055db0f3bd808f96a61119f61520c76.jpg)  
Figure 8: Despite implicit size control in JewelTry, repeated generations converge to similar relative jewelry scales.

![](images/1baba81789dfd64e60e75c4b911fee6edd9eeddc563ba55ea8047869354c73ad.jpg)  
Figure 9: JewelTry enables relative size control but may fail at extreme scales due to its reliance on implicit in-context learning and the scale bias inherent in real-world jewelries.

## Conclusion

In this paper, we address mask-free scale-aware jewelry virtual try-on, a challenging setting where visual realism depends critically on object scale and fine-grained structural preservation. We introduce JVTO-Bench, a dataset with triplet samples and product-scale annotations across four major jewelry categories. Building on this, we propose Jewel-Try, a mask-free framework that incorporates product scale through a scale adapter and improves jewelry fidelity via single-directional condition attention and an attention refinement loss. These designs preserve jewelry appearance and structure while rendering it at scale-aware without relying on masks. Extensive experiments show that JewelTry achieves a balance among visual fidelity, background preservation, object consistency, and scale accuracy, establishing a strong baseline for mask-free, scale-aware jewelry virtual try-on for future research.

Cai, M.; Cun, X.; Li, X.; Liu, W.; Zhang, Z.; Zhang, Y.; Shan, Y.; and Yue, X. 2025. Ditctrl: Exploring attention control in multi-modal difusion transformer for tuning-free multi-prompt longer video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 7763–7772.

Chang, T.-Y.; and Lekena, S. K. 2024. GlamTry: Advancing Virtual Try-On for High-End Accessories. arXiv preprint arXiv:2409.14553.

Chen, B.; Zhao, M.; Sun, H.; Chen, L.; Wang, X.; Du, K.; and Wu, X. 2025. Xverse: Consistent multi-subject control of identity and semantic attributes via dit modulation. arXiv preprint arXiv:2506.21416.

Chen, D.; Duan, Z.; Li, Z.; Chen, C.; Chen, D.; Li, Y.; and Chen, Y. 2026. AttriCtrl: A Generalizable Framework for Controlling Semantic Attribute Intensity in Difusion Models. In The Fourteenth International Conference on Learning Representations.

Cheng, Y.; Wu, W.; Wu, S.; Huang, M.; Ding, F.; and He, Q. 2025. UMO: Scaling Multi-Identity Consistency for Image Customization via Matching Reward. arXiv preprint arXiv:2509.06818.

Choi, Y.; Kwak, S.; Lee, K.; Choi, H.; and Shin, J. 2024. Improving difusion models for authentic virtual try-on in the wild. In European Conference on Computer Vision, 206– 235. Springer.

Du, C.; Xiong, S.; and Rong, Y. 2025. All Parts Matter: A Unified Mask-Free Virtual Try-On Framework. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 19525–19534.

Du, C.; Xiong, S.; Wang, J.; Rong, Y.; and Xiong, S. 2025. Mitigating Occlusions in Virtual Try-On via A Simple-Yet-Efective Mask-Free Framework. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Feng, Y.; Zhang, L.; Cao, H.; Chen, Y.; Feng, X.; Cao, J.; Wu, Y.; and Wang, B. 2025. Omnitry: Virtual try-on anything without masks. arXiv preprint arXiv:2508.13632.

Guo, H.; Zeng, B.; Song, Y.; Zhang, W.; Liu, J.; and Zhang, C. 2025. Any2anytryon: Leveraging adaptive position embeddings for versatile virtual clothing tasks. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 19085–19096.

Hu, J.; Cheng, Z.; Wong, W.; and Zou, X. 2026. Garments2Look: A Multi-Reference Dataset for High-Fidelity Outfit-Level Virtual Try-On with Clothing and Accessories. arXiv preprint arXiv:2603.14153.

Issenhuth, T.; Mary, J.; and Calauzènes, C. 2019. End-toend learning of geometric deformations of feature maps for virtual try-on. arXiv preprint arXiv:1906.01347.

Jiang, Y.; Zhao, N.; Liu, Q.; Singh, K. K.; Yang, S.; Loy, C. C.; and Liu, Z. 2024. Groupdif: Difusion-based group portrait editing. In European Conference on Computer Vision, 221–239. Springer.

Kwon, S.; Lee, K.; Jung, D.; and Lee, J. 2026. FEAT: Fashion Editing and Try-On from Any Design. In Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 22080–22089.

Lee, S.; and Kwak, J.-g. 2025. Voost: A Unified and Scalable Difusion Transformer for Bidirectional Virtual Try-On and Try-Of. arXiv preprint arXiv:2508.04825.

Li, L.; Gong, Y.; Liu, S.; Cheng, B.; Ma, Y.; Wu, L.; Jiang, D.; Wang, Z.; Leng, D.; and Yin, Y. 2025. RefVTON: person-toperson Try on with Additional Unpaired Visual Reference. arXiv preprint arXiv:2511.00956.

Lipman, Y.; Chen, R. T. Q.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2023. Flow Matching for Generative Modeling. arXiv:2210.02747.

Liu, S.; Zeng, Z.; Ren, T.; Li, F.; Zhang, H.; Yang, J.; Jiang, Q.; Li, C.; Yang, J.; Su, H.; et al. 2024. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, 38– 55. Springer.

Mei, S.; and Ni, B. 2026. PhysDif-VTON: Cross-Domain Physics Modeling and Trajectory Optimization for Virtual Try-On. Advances in Neural Information Processing Systems, 38: 9551–9572.

Miao, Y.; Huang, Z.; Han, R.; Wang, Z.; Lin, C.; and Shen, C. 2025. Shining yourself: High-fidelity ornaments virtual tryon with difusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 359–368.

Niu, Y.; Yi, D.; Wu, L.; Liu, Z.; Cai, P.; and Wang, J. 2024. PFDM: Parser-Free Virtual Try-On via Difusion Model. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 3780– 3784. IEEE.

She, D.; Fu, S.; Liu, M.; Jin, Q.; Wang, H.; Liu, M.; and Jiang, J. 2025. MOSAIC: Multi-Subject Personalized Generation via Correspondence-Aware Alignment and Disentanglement. arXiv preprint arXiv:2509.01977.

Shen, T.; Huang, Z.; Li, X.; Lin, Z.; Liu, J.; Wang, Y.; Feng, J.; Yang, M.-H.; and Liew, J. H. 2025. Qk-edit: Revisiting attention-based injection in mm-dit for image and video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 19043–19053.

Shin, J.; Hwang, A.; Kim, Y.; Kim, D.; and Park, J. 2025. Exploring multimodal difusion transformers for enhanced prompt-based image editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 19492–19502.

Song, W.; Jiang, H.; Yang, Z.; Cheng, Z.; Quan, R.; and Yang, Y. 2026. Insert anything: Image insertion via in-context editing in dit. In Proceedings of the AAAI Conference on Artificial Intelligence, 11, 9097–9105.

Tan, Z.; Liu, S.; Yang, X.; Xue, Q.; and Wang, X. 2025. Ominicontrol: Minimal and universal control for difusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 14940–14950.

Wang, A.; Li, W.; Luo, H.; Ao, M.; Zhu, C.; Li, X.; and Wang, F. 2025. JCo-MVTON: Jointly Controllable Multi-Modal Difusion Transformer for Mask-Free Virtual Try-on. arXiv preprint arXiv:2508.17614.

Wang, C.; Chen, T.; Chen, Z.; Huang, Z.; Jiang, T.; Wang, Q.; and Shan, H. 2024. FLDM-VTON: Faithful latent difusion model for virtual try-on. arXiv preprint arXiv:2404.14162.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612.

Wu, C.; Li, J.; Zhou, J.; Lin, J.; Gao, K.; Yan, K.; Yin, S.-m.; Bai, S.; Xu, X.; Chen, Y.; et al. 2025. Qwen-image technical report. arXiv preprint arXiv:2508.02324.

Xu, P.; Tang, P.; Luo, D.; Hu, X.; Cui, W.; He, Q.; Chen, Z.; Zhang, J.; Ling, C.; and Wang, B. 2026a. Towards Generalized Multi-Image Editing for Unified Multimodal Models. arXiv preprint arXiv:2601.05572.

Xu, Y.; Gu, T.; Chen, W.; and Chen, A. 2025a. Ootdifusion: Outfitting fusion based latent difusion for controllable virtual try-on. In Proceedings of the AAAI Conference on Artificial Intelligence, 9, 8996–9004.

Xu, Z.; Li, X.; Zhang, J.; Wan, J.; Chen, C.; and Wu, J. 2026b. Sparkling Together: Joint Editing for Multi-Accessory Virtual Try-On. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 8892–8896. IEEE.

Xu, Z.; Shen, X.; Nan, H.; and Zhang, X. 2025b. NumeriKontrol: Adding Numeric Control to Difusion Transformers for Instruction-based Image Editing. arXiv preprint arXiv:2511.23105.

Zeng, W.; Wei, P.; Wang, H.; Zhang, B.; Sun, J.; Fan, D.; HE, L.; Chen, L.; Gan, Q.; Yang, F.; et al. 2026. OmniDiT: Extending Difusion Transformer to Omni-VTON Framework. arXiv preprint arXiv:2603.19643.

Zhang, H.; Li, F.; Liu, S.; Zhang, L.; Su, H.; Zhu, J.; Ni, L. M.; and Shum, H.-Y. 2022. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The Unreasonable Efectiveness of Deep Features as a Perceptual Metric. In CVPR.

Zhang, X.; Song, D.; Zhan, P.; Chang, T.; Zeng, J.; Chen, Q.; Luo, W.; and Liu, A.-A. 2025a. Boow-vton: Boosting inthe-wild virtual try-on via mask-free pseudo data training. In Proceedings of the Computer Vision and Pattern Recognition Conference, 26399–26408.

Zhang, X.; Southwell, B. J.; Pan, S.; Niu, X.; Ahmed, B.; and Epps, J. 2026. Why your tokenizer fails in information fusion: A timing-aware pre-quantization fusion for video-enhanced audio tokenization. arXiv preprint arXiv:2604.12145.

Zhang, Y.; Yuan, Y.; Song, Y.; Wang, H.; and Liu, J. 2025b. Easycontrol: Adding eficient and flexible control for difusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 19513–19524.

Zhu, L.; Yang, D.; Zhu, T.; Reda, F.; Chan, W.; Saharia, C.; Norouzi, M.; and Kemelmacher-Shlizerman, I. 2023. Tryondifusion: A tale of two unets. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 4606–4615.

# [Supplementary Material] JewelTry: Mask-Free Scale Aware Jewelry Virtual Try-On

Xinlei Niu<sup>1∗</sup>, Peixia Li<sup>2</sup>, Jun Wang<sup>2</sup>, Chenchen Xu<sup>2</sup>, Jiayu Yang<sup>2</sup>, Jing Zhang<sup>1</sup>, Pulak Purkait<sup>2</sup>, Hongdong Li<sup>1,2</sup>

<sup>1</sup>Australian National University, Canberra, Australia; <sup>2</sup>Amazon, Melbourne, Australia xinlei.niu@anu.edu.au

## Contents

1 JVTO-Bench 1   
2 Implementation details 2   
2.1 Training 2   
2.2 Inference 3   
2.3 Seeds and randomness statement . 3   
2.4 Evaluation metrics 3   
3 Quantitative comparison across jewelry cate  
gories 3   
4 Result variance and significance test 3   
Attention Refinement Loss 4   
5.1 Attention refinement loss and block-wise   
cross attention . 4   
5.2 Timestep binary gate 4   
6 Person-conditioned Classifier-Free Guidance 4

7 More study on single direction condition attention mechanism

## 1 JVTO-Bench

JVTO-Bench is a real-world jewelry virtual try-on dataset covering four major jewelry categories: rings, earrings, necklaces, and bracelets. The dataset contains approximately 23k samples with diverse jewelry items and pose. JVTO-Bench provides triplet samples including ground-truth try-on image, try-of person image, and reference jewelry image. JVTO-Bench also provides real-world product 2D scale in inches, product type, how jewelry object interact with the person image caption, and target object bounding box in try-on image annotations.

Each data sample includes a reference product image captured on a white background, a try-on image showing the jewelry worn by a model, a corresponding try-of image without the jewelry, an object mask indicating the jewelry region in the try-on image, and product type, H × W scale (in inches), and caption annotations. JVTO-Bench is designed to support the development and evaluation of jewelry virtual try-on methods across diverse product types, visual appearances, and market sources.

Table 2 summarizes the statistics of JVTO-Bench. To prevent product-level overlap between the training and test sets, we randomly reserve 150 products from each of the ring, earrings, necklace, and bracelet categories before constructing the training split. After data filtering, the final test set contains 59 ring products, 104 earring products, 106 necklace products, and 106 bracelet products. The ring category is treated separately because its scale annotations are more heterogeneous: many ring listings either omit explicit dimension information or provide multiple size variants. We therefore retain the 59 valid ring samples as a separate, potentially biased test subset, while keeping the other three categories approximately balanced at around 100 samples each. Ground-truth try-on images in JVTO-Bench have resolutions ranging from 500 to 2,000 pixels, and images with a minimum spatial resolution below 500 pixels are discarded. Representative samples are provided in the dataset supplementary material.

<table><tr><td></td><td>Ring</td><td>Earrings</td><td>Necklace</td><td>Bracelet</td><td>Total</td></tr><tr><td>Train split</td><td>2908</td><td>12149</td><td>4586</td><td>4019</td><td>23662</td></tr><tr><td>Test split</td><td>59</td><td>104</td><td>106</td><td>106</td><td>375</td></tr></table>

Table 1: JVTO-Bench Statistics.

Since jewelry is a rigid, solid object with fixed geometry, its proportions therefore remain constant, so a single dominant dimension is suficient to characterize its overall scale. As illustrated in Figure 1, we define category-specific scale annotations according to the structural characteristics of each jewelry type. For earrings, H denotes the overall vertical height of the earring; for necklaces, H corresponds to the pendant height. For bracelets and rings, whose worn appearance is mainly determined by band thickness, we define H as the maximum width of the bracelet or ring band. This unified scale representation enables consistent modeling of jewelry size across categories. Since jewelry items are rigid and their perceived size is largely determined by a single dominant dimension, we adopt a unified one-dimensional scale representation and use only the H dimension during training. In JVTO-Bench, however, we provide 2D scale for future research, where W dimension is perpendicular to H dimension indicated in Figure 1.

![](images/4be3136accf8665a85a3ec7bfffcc883b7f24985a7e92bf8aeb52a3850a74fea.jpg)

Figure 1: Product-scale annotations across diferent jewelry categories. Although jewelry items are 3D objects, we annotate category-specific 2D physical dimensions that are commonly provided in online shopping, such as length, width, diameter, or pendant size. This design follows real-world ecommerce practice, where sellers typically describe the most visually relevant dimensions for each jewelry category.  
![](images/ed56a48df39dbb283e58a18867090e018d5f55c98c817118c91aaad13a5306fd.jpg)  
Figure 2: Our refined try-of image processing pipeline. We further refine try-of image for those low quality Qwen-Object-Remover outputs.

Obtaining high-quality try-of images is particularly challenging for jewelry VTON. Since jewelry items often cast shadows or introduce subtle local appearance changes, simply combining object detection with an inpainting model (Feng et al. 2025) may fail to fully remove the jewelry-related regions. Object removal models such as Qwen-Object-Remover can more efectively eliminate both the jewelry and its shadow, producing more consistent tryof images; however, they may also introduce color shifts or unintended changes in some cases. To improve try-of quality, we further refine the object removal results by identifying changed regions between the original image and the object-removed image using SSIM. We then use the resulting diference region as a refined mask and apply an additional inpainting step to better preserve the surrounding appearance (see Figure 2). Figure 3 shows a qualitative comparison of try-of image generation using diferent removal strategies.

![](images/86f52ecb6b352f13cba4c174d6ec5e13a10c67e946d06be69b20c96b65a75dbf.jpg)  
Figure 3: Comparison of diferent object removal methods. Our refined try-of images exhibit more natural visual quality and better suppress residual shadows and artifacts caused by the removed jewelry.

However, we observe that the SSIM-based mask is not robust across all images. Therefore, we manually apply this refinement step only to Qwen object-removal results that exhibit obvious quality degradation.

We provide an example sample of JVTO-Bench in Figure 4.
<table><tr><td></td><td>Ring</td><td>Earrings</td><td>Necklace</td><td>Bracelet</td><td>Total</td></tr><tr><td>Train split</td><td>2908</td><td>12149</td><td>4586</td><td>4019</td><td>23662</td></tr><tr><td>Test split</td><td>59</td><td>104</td><td>106</td><td>106</td><td>375</td></tr></table>

Table 2: JVTO-Bench Statistics.

## 2 Implementation details

## 2.1 Training

JewelTry is built upon Qwen-Image-Edit-2511 (Wu et al. 2025). We fine-tune the model using LoRA adapters with rank 16 and LoRA alpha 16 inserted into the self-attention blocks (q, k, v, out). Training is performed on the JVTO-Bench training split using 8 NVIDIA H200 GPUs for 200k iterations with a batch size of 8 and a learning rate o $: 5 \times 1 0 ^ { - 5 }$

During training, the person image and target try-on image are resized to 1024 × 1024, while the reference jewelry image is resized to 768 × 768. The proposed scale adapter first projects the numerical scale value into a 256-dimensional embedding through a linear layer followed by a SiLU activation. The resulting embedding is then combined with a learnable jewelry-category embedding and further projected to the 3584-dimensional text-token space using another SiLUactivated projection layer. To preserve the model’s ability to perform inference without scale annotations, we randomly drop the scale token with a probability of 25% during training.

For the attention refinement loss, object attention maps are extracted from the $5 1 ^ { \mathrm { s t } }$ to $6 0 ^ { \mathrm { { t h } } }$ MMDiT transformer blocks, where the attention responses exhibit the strongest correlation with the target jewelry region (see examples in Figure 6 and Figure 7). To obtain an attention-based object soft mask, we extract the cross-attention maps, average them across multiple attention heads and feature dimensions, and then normalize the resulting map to the range [0, 1]. We also involve a timestep binary gate in attention refinement loss, see Section 5.2 for more details.

Since we follow qwen-image-edit and use Qwen-VL-2.5 as the VLM to encode text prompt in JewelTry, which has reasoning ability. We use caption extracted by Claude (annotations provided by JVTO-Bench) as text prompt with additional information of product scale, $\mathrm { e . g . }$ , “The person from image 1 is wearing earrings from image 2. Earring height: 3.8 inches. ” for both training JewelTry and Qwen-JVTON model. To fine-tune the Qwen-image-edit model for Qwen-JVTON on JVTO-Bench dataset, we inject LoRA parameter into the self-attention block with rank 16, and we follow the exact same setting as used in JewelTry on 8 NVIDIA H200 GPUs for 200k iterations with a batch size of 8 and a learning rate of $5 \times 1 0 ^ { - 5 }$ for 200k iterations.

## 2.2 Inference

For Qwen-Image-Edit-2511, Qwen-JVTON, and JewelTry, we set the classifier-free guidance (CFG) scale to 3.0 and use 50 denoising steps. For OmniTry, Any2AnyTryOn, and InsertAnything, we follow their original inference configurations.

For InsertAnything, we generate reference object masks and target inpainting masks using Grounded DINO and SAM. Since InsertAnything requires an explicit insertion mask and the mask provides direct guidance on object scale and placement. Therefore, this make mask-guided methods not directly comparable to mask-free methods. To reduce such mask-induced scale guidance while retaining a valid insertion region on JVTO-Bench, we convert the corresponding person-region mask into a rectangular bounding box and randomly enlarge the target bounding box by 20%. For OmniTry-Bench, we directly use the target masks provided by the dataset for InsertAnything inference.

For Qwen-Image-Edit-2511 inference on JVTO-Bench, we use the same text prompts as JewelTry and Qwen-JVTON which with the scale information to ensure a fair comparison.

For inference results on OmniTry-Bench, we found that the given captions is not strongly match to the target results, which further caused Qwen-JVTON, Qwen-Image-Edit, and JewelTry to fail to preserve the person image. Therefore, we instead adopt a simplified prompt in the format: ”The person is wearing the given {jewelry class}“. Since OmniTry-Bench dataset has no product scale information, we drop the scale token from the scale adapter in JewelTry and also exclude scale information in the text prompt for JewelTry, Qwen-JVTON and Qwen-image-edit model.

## 2.3 Seeds and randomness statement

To ensure a fair comparison under realistic stochastic inference conditions, we generate all inference results using random seeds rather than fixing a specific seed.

## 2.4 Evaluation metrics

Since jewelry is a rigid object with fixed geometry, its proportions remain constant, so a single dominant dimension is suficient to characterize its overall scale. Motivated by this, we design the ScaleErr metric to evaluate scale faithfulness. Specifically, we detect the jewelry bounding box in both the generated and ground-truth try-on images, and compute the absolute log-ratio of their dominant dimension H, defined as

$$
\mathrm { S c a l e E r r } = | l o g ( H _ { \mathrm { p r e d } } / H _ { \mathrm { g t } } ) |\tag{1}
$$

where $H _ { \mathrm { p r e d } }$ and $H _ { \mathrm { g t } }$ denote the dominant dimension defined in Figure 1 of the predicted and ground-truth jewelry, respectively. A value of 0 indicates a perfect size match, and larger values indicate greater scale mismatch. The log-ratio is symmetric, penalizing equally an object rendered too large or too small by the same factor.

We use Gounded DINO (Liu et al. 2024) and SAM (Kirillov et al. 2023) to detect the jewelry region, which further use the output to mask the jewelry region or calculate metrics.

## 3 Quantitative comparison across jewelry categories

Figure 5 provides a category-wise quantitative comparison on JVTO-Bench. We report $\mathrm { D I N O _ { p } }$ and $\mathrm { C L I P _ { T a r } }$ , which evaluate person/background preservation and object consistency with respect to the ground-truth try-on images, respectively. Across the four major jewelry categories, JewelTry achieves the best performance on rings, necklaces, and bracelets, and ranks second on earrings, with only a small gap to the best method: −0.0309 on $\mathrm { D I N O _ { p } }$

These results demonstrate that JewelTry generalizes well across jewelry types with diferent spatial scales, body attachment regions, and structural characteristics. The strong performance on rings and bracelets indicates that our method can handle small and localized try-on regions, while the gains on necklaces suggest its efectiveness for larger accessories that require accurate placement around complex neck and upper-body regions. Although earrings remain challenging due to their tiny size, complex structure and high sensitivity to scale, and frequent occlusion by hair or face contours, JewelTry remains highly competitive. Overall, the categorywise results further validate the robustness of our scale-aware conditioning and conditional fidelity design across diverse jewelry VTON scenarios.

## 4 Result variance and significance test

We conduct significance testing (t-test) compared with each baseline on JVTO-Bench test split. Compared to the zeroshot baselines, JewelTry obtains a more balanced performance and significantly outperforms OmniTry, Qwen-imageedit, InsertAnything and Any2AnyTryOn regarding object consistency, background preservation, and scale faithfulness on metrics such as $\mathrm { D I N O } _ { \mathrm { T a r } }$ (with p-values $2 . 8 1 \times 1 0 ^ { - 3 }$ $1 . 1 \times 1 0 ^ { - 2 } , \ 3 . 1 0 \times 1 0 ^ { - 2 \overleftarrow { 2 } }$ , and $\phantom { + } 3 . 8 7 \times 1 0 ^ { - 4 7 }$ , respectively), IoU (with p-values $1 . 1 9 \times 1 0 ^ { - 1 3 } , 1 . 4 6 \times 1 0 ^ { - 1 5 }$ $1 . 4 9 \times 1 0 ^ { - 2 3 }$ , and $\mathrm { i . 5 2 \times 1 0 ^ { - 4 9 } }$ , respectively), and LPIPS<sub>p</sub> (with p-values $8 . 1 6 \times 1 0 ^ { - 3 2 } , 7 . 7 6 \times 1 0 ^ { - 2 1 } , 9 . 6 1 \times 1 0 ^ { - 3 1 }$ and $3 . { \overset { \cdot } { 8 } } 7 \times 1 0 ^ { - 4 7 }$ , respectively). Compared to the trainedon-bench baselines, Qwen-JVTON, JewelTry also significantly outperform on object consistency, background preservation, and scale-faithfulness with p-values $\mathbf { \bar { 8 . 6 4 } \times 1 0 ^ { - 3 } }$ $4 . 7 8 \times 1 0 ^ { - 7 }$ , and $2 . 1 9 \times 1 0 ^ { - 2 }$ on $\mathrm { D I N O } _ { \mathrm { R e f } } ,$ LPIPS , and IoU, respectively. In terms ofjewelry scale faithfulness, JewelTry significantly outperforms than all the baselines with p-values $1 . 2 4 \times \mathrm { i } 0 ^ { - 4 }$ (Qwen-JVTON), $2 . 0 3 \times 1 0 ^ { - 1 2 }$ (OmniTry), $9 . 7 \times 1 0 ^ { - 7 }$ (Qwen-image-edit), $3 . 0 5 \times 1 0 ^ { - 1 0 }$ (InsertAnything), and $1 . 6 1 \times 1 0 ^ { - 5 0 } \left( \mathrm { \bar { A } n y { 2 A } n y T r y O n } \right)$ . Significant tests further indicate JewelTry obtains more balanced performance on object consistency and background preservation, while also reach to a better scale-aware ability.

## 5 Attention Refinement Loss

In this section, we provide more technical details for the proposed attention refinement loss.

## 5.1 Attention refinement loss and block-wise cross attention

We provide further discussion and insights of the proposed attention refinement loss. Inspired by Shin et al. (2025), attention refinement loss is motivated by the empirical finding of the efectiveness of block-wise attention between conditions and latent in MMDiT transformer. Figure 6 and Figure 7 visualize the attention maps across MMDiT transformer blocks for $Q _ { \mathrm { n o i s e } } \cdot K _ { \mathrm { p e r s o n } } ^ { T }$ and ${ \dot { Q } } _ { \mathrm { n o i s e } } \cdot K _ { \mathrm { r e f e r e n c e } } ^ { T } ,$ averaged over diffusion time steps. We observe that attention maps from later transformer blocks (e.g., the last two rows in $Q _ { \mathrm { n o i s e } } \cdot K _ { \mathrm { r e } } ^ { T }$ eference figures) more clearly highlight the target jewelry region in the noisy latent space, with substantially reduced background noise. This observation motivates our attention refinement loss: by extracting attention maps from later transformer blocks and supervising them with the ground-truth jewelry mask, we explicitly encourage the model to focus on finegrained jewelry regions during training, thereby improving object localization and structural consistency.

## 5.2 Timestep binary gate

In the attention refinement loss, we supervise the crossattention maps with the ground-truth object mask via a combined BCE and Dice objective. Since spatial structure is unreliable at high noise levels, we restrict this supervision to low-noise timesteps through a binary gate $1 [ \sigma _ { b } < \tau ]$ , which activates the loss only when the sample’s noise level $\sigma _ { b }$ falls below a threshold τ (we use $\tau = 0 . 5$ , i.e. the lower half of the noise schedule). Samples with $\sigma _ { b } \geq \tau$ contribute zero attention loss. The timestep binary gate is defined as

$$
\mathbf { 1 } [ \sigma _ { b } < \tau ] = \left\{ \begin{array} { l l } { 1 } & { \sigma _ { b } < \tau ; } \\ { 0 } & { \sigma _ { b } \geq \tau ; } \end{array} \right.\tag{2}
$$

## 6 Person-conditioned Classifier-Free Guidance

In the inference stage of JewelTry, we treat the scale token, text tokens, and reference jewelry tokens as a unified conditioning signal and apply classifier-free guidance (CFG) with a shared guidance value. Specifically, the guided noise prediction is formulated as:

$$
\begin{array} { r } { \tilde { \epsilon } ( z _ { t } , c _ { s } , c _ { \mathrm { t x t } } , c _ { P } , c _ { J } ) = \epsilon _ { \theta } ( z _ { t } , c _ { s } , c _ { \mathrm { t x t } } , c _ { P } , c _ { J } ) + \qquad } \\ { w \left[ \epsilon _ { \theta } ( z _ { t } , c _ { s } , c _ { \mathrm { t x t } } , c _ { P } , c _ { J } ) - \epsilon _ { \theta } ( z _ { t } , c _ { P } ) \right] , } \end{array}\tag{3}
$$

where $c _ { s } , c _ { \mathrm { t x t } } , c _ { P }$ , and $c _ { J }$ denote the scale, text, person image, and reference jewelry image conditions, respectively. w is the guidance strength. In person-conditioned CFG, the person condition $c _ { P }$ is retained in the negative sample, while the scale, text, and reference jewelry conditions are dropped, encouraging the model to follow the jewelry-related conditions while preserving the target person structure.

Figure 8 presents a qualitative comparison of JewelTry inference results under diferent CFG values. The results show that a low CFG value, e.g., CFG =1, provides insuficient conditional guidance, making the generated try-on results less responsive to the specified product scale. In contrast, an overly large CFG value, e.g., CFG =10 or 15, can overamplify the scale, text, and reference jewelry conditions, leading to unrealistic try-on results or structural collapse of the jewelry. These observations suggest that a moderate CFG value is necessary to balance scale controllability, jewelry fidelity, and visual realism.

## 7 More study on single direction condition attention mechanism

<table><tr><td>Method</td><td>|Mask type</td><td> $| \mathrm { D I N O } _ { \mathrm { p } } \uparrow$   $\mathrm { L P I P S } _ { \mathrm { p } } \downarrow$ </td><td> $| \mathrm { D I N O } _ { \mathrm { r e f } } \uparrow$ </td><td> $\mathrm { D I N O } _ { \mathrm { t a r } }$  ↑ IoU↑</td></tr><tr><td>Qwen-JVTON|-</td><td></td><td>|0.950 0.128</td><td>|0.525</td><td>0.759 0.591</td></tr><tr><td>Qwen-JVTON|M JewelTry</td><td>M</td><td>|0.953 0.121 0.959 0.121</td><td>|0.559 0.565</td><td>0.760 0.589 0.792 0.658</td></tr><tr><td>Qwen-JVTON JewelTry</td><td> $M ^ { \mathrm { P \& J } }$   $M ^ { \mathrm { P \& J } }$ </td><td>|0.952 0.151 0.957 0.164</td><td>|0.555 |0.568</td><td>0.756 0.504 0.793 0.660</td></tr></table>

Table 3: Ablation study for single direction condition attention mask for block-wise attention. Where M is the single direction condition attention mask defined in our main text.

In JewelTry, we apply single-direction conditional attention to block the information flow from noisy latent tokens to jewelry condition tokens during training. Our ablation studies show that this mechanism improves coarse-level jewelry consistency during inference. In this section, we further investigate the efect of single-direction conditional attention under diferent block-wise masking strategies.

In the VTON task, the generated try-on result is expected to preserve both the identity and appearance of the person image while maintaining consistency with the reference jewelry image. Based on the hypothesis that information flow from noisy latent tokens to conditional tokens may cause condition collapse, we further examine whether blocking such information flow to both person and jewelry tokens can improve background preservation and object consistency. Following the notation in the main text, we define a mask $M ^ { \mathrm { P \& J } }$ that blocks the reverse attention paths from noisy latent tokens to both person and reference jewelry tokens as follows:

$$
M _ { i j } ^ { \mathrm { P } \& \mathrm { J } } = \left\{ \begin{array} { c l } { - \infty } & { ( i \in \mathbf { P } , \ j \in \mathrm { n o i s e } ) \cup ( i \in \mathbf { J } , \ j \in \mathrm { n o i s e } ) ; } \\ { 0 } & { \mathrm { o t h e r w i s e } ; } \end{array} \right.\tag{4}
$$

The mask $M ^ { \mathrm { P \& J } }$ blocks the reverse attention paths from latent noisy tokens to person and referencejewelry tokens, with quantitative results reported in Table 3. In our ablation study, we integrate a single direction condition attention mask to both Qwen-JVTON and JewelTry. The comparison shows that blocking person and jewelry tokens further improves object consistency; however, it slightly degrades background preservation with higher $\mathrm { L P I P S } _ { \mathrm { p } }$ . Figure 9 provides a visual comparison between the two masking strategies. Although $M ^ { \mathrm { P \& J } }$ achieves better object consistency, it introduces a noticeable color-shift issue, where the skin tone of the source image is slightly changed. This may be because suppressing attention to person tokens weakens the model’s access to person-specific appearance cues, making it less efective in preserving local color and illumination consistency.

## 8 More visualization

Many jewelry items, such as earrings and necklaces, require the model to preserve not only complex topology but also fine-grained textual details, similar to garments. As shown in Figure 10, JewelTry efectively preserves these textual details in the generated jewelry.

We provide additional qualitative comparisons on both JVTO-Bench and OmniTry-Bench in Figure 11 to Figure 13. For clarity, we compare JewelTry against the four most competitive baselines: OmniTry, Qwen-JVTON, Qwen-imageedit, and InsertAnything, because Any2AnyTryOn is not initially designed for jewelry VTON task. Although OmniTry-Bench does not provide product-scale annotations, JewelTry consistently generates jewelry with better structural fidelity, texture preservation, and fine-grained detail consistency. In particular, our method more faithfully preserves the geometry and appearance of the reference jewelry while maintaining natural integration with the wearer, demonstrating the efectiveness of the proposed object-consistency components.

![](images/63aab082daedcc9bd49e46e4ca6b527e67d99768c78bdc1abcf09d1de5669e36.jpg)  
(c) Ring sample.

![](images/a46dcb121f24d3ed02c232ae4862ec205e6519de26239613897a391fe984bef8.jpg)  
(d) Necklace sample.  
Figure 4: Examples from JVTO-Bench across four jewelry categories: earrings, bracelet, ring, and necklace.

DINO-P  
![](images/08341278867af3fca547e7ab226ef6849c209de373ff2a96ae25a8a9eccf772b.jpg)  
Figure 5: Quantitative comparison across four jewelry categories in JVTO-Bench. JewelTry present superior performance on maintaining object consistency and balanced background preservation cross the four jewelry categories.

![](images/f3a3ee79079607c0104bc3018eed705ca59cb27b0cc3c5ee43a1d993283c613c.jpg)  
Figure 6: A visualization example for attention maps cross diferent block for earring. The left hand side visualizes $Q _ { \mathrm { n o i s e } } \cdot K _ { \mathrm { p e r s } } ^ { T }$ on attention map cross the whole attention blocks averaged time steps during inference; While the right hand size visualizes $Q _ { \mathrm { n o i s e } } \cdot K _ { \mathrm { r e f e r e n c e } } ^ { T }$ attention map cross the whole attention blocks averaged time steps during inference.

![](images/c855b12db1b9fbabf43c8ff1f947dd6488c10dcb289d45295ebebd7ac1e3e1bc.jpg)  
Figure 7: A visualization example for attention maps cross diferent block for necklace. The left hand side visualizes $Q _ { \mathrm { n o i s e } } { \cdot } K _ { \mathrm { p e r s } } ^ { T }$ son attention map cross the whole attention blocks averaged time steps during inference; While the right hand size visualizes $Q _ { \mathrm { n o i s e } } \cdot K _ { \mathrm { r e f e r e n c e } } ^ { T }$ attention map cross the whole attention blocks averaged time steps during inference.

![](images/b6064f96b4de9689ac067a6f2c358ac3e53f16f9c21ee45d3a5a4d6ff8c29aaf.jpg)  
Figure 8: JewelTry inference result using diferent values of the classifier-free guidance scale.

Source image

Reference

Ground truth

JewelTry w $M ^ { P \& J }$

JewelTry w �

![](images/d7932a72aff56e778bd6e9e859bbe3f5faec90bb1c657a3549f536c7114aaf2b.jpg)  
Figure 9: Qualitative comparison of diferent single direction condition attention mechanism on JewelTry. Although JewelTry with $M ^ { P \& \tilde { J } }$ has better object consistency; however, it sufers bad background preservation, skin tone color shift, and loss fine details on input source images.

Reference

Ground truth

JewelTry

Reference

Ground truth  
JewelTry  
![](images/d67036e37712729b7aadc969d1a604d04bbdaf9a782cc0d11b794732f95874c5.jpg)  
Figure 10: Additional JewelTry results on JVTO-Bench. These examples demonstrates JewelTry’s ability to preserve fine-grained textual details in jewelry.

![](images/6e814912c0c6198b3f760111129f4f88034658418a811d0f1af4c121ef378ae1.jpg)  
Figure 11: More visual comparison on JVTO-Bench cross four jewelry categories.

Source image

Reference

JewelTry

OmniTry

Qwen-JVTON

InsertAnything

Qwen-image-edit

![](images/a4332e911bfaa614e43fbe076bbbd6103b1b908a393fa5af451667eb744bf84a.jpg)  
Figure 12: More visual comparison on OmniTry-Bench for bracelet and ring categories.

![](images/d064b74f2a1bfe42725a0367d06c5e5040b2384314264b321694979a4867b52a.jpg)  
Figure 13: More visual comparison on OmniTry-Bench for earrings and necklace categories.

## References

Feng, Y.; Zhang, L.; Cao, H.; Chen, Y.; Feng, X.; Cao, J.; Wu, Y.; and Wang, B. 2025. Omnitry: Virtual try-on anything without masks. arXiv preprint arXiv:2508.13632.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.- Y.; Dollár, P.; and Girshick, R. 2023. Segment Anything. arXiv:2304.02643.

Liu, S.; Zeng, Z.; Ren, T.; Li, F.; Zhang, H.; Yang, J.; Jiang, Q.; Li, C.; Yang, J.; Su, H.; et al. 2024. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, 38– 55. Springer.

Shin, J.; Hwang, A.; Kim, Y.; Kim, D.; and Park, J. 2025. Exploring multimodal difusion transformers for enhanced prompt-based image editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 19492–19502.

Wu, C.; Li, J.; Zhou, J.; Lin, J.; Gao, K.; Yan, K.; Yin, S.-m.; Bai, S.; Xu, X.; Chen, Y.; et al. 2025. Qwen-image technical report. arXiv preprint arXiv:2508.02324.