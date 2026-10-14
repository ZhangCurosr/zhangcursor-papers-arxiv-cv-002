# LensErase: A Foundation Model for Erase Anything

Yunuo Chen<sup>1,∗</sup> Xiuyu Wu<sup>2,∗</sup> Qinhong Yang<sup>3</sup> Jinjing Zhao<sup>4</sup> Yanchen Dong<sup>5</sup> Sirui Zhang<sup>3</sup> Dong Chen<sup>6,‡</sup> Fangyun Wei<sup>6,†</sup> Lei Shi<sup>6</sup> Ji Li<sup>6</sup> Dongdong Chen<sup>6</sup> Guo Lu<sup>1,†</sup> Yan Lu<sup>6</sup> Baining Guo<sup>6</sup>

<sup>1</sup> Shanghai Jiao Tong University <sup>2</sup> Nankai University <sup>3</sup> USTC <sup>4</sup> University of Sydney <sup>5</sup> Peking University <sup>6</sup> Microsoft

∗ Equal contribution. <sup>†</sup> Corresponding authors. <sup>‡</sup> Project lead.

![](images/61b3a105e68f3464af61ddf6383ced9b809418ba96fc0ec6727d4e13390e815c.jpg)  
Figure 1 LensErase across diverse removal scenarios. Each pair shows a source image with highlighted removal targets and the corresponding LensErase result. Examples cover general objects, visual text, watermarks, multiple objects, coupled efects, and high-fidelity inpainting.

We introduce LensErase, a general-purpose mask-based Erase Anything foundation model designed to cover a broad spectrum of real-world erase scenarios. LensErase supports diverse objects, visual text, watermarks, associated visual efects, multiple-object removal, and a wide range of object scales, aspect ratios, and image resolutions. First, to enable such broad coverage, we construct LensErase-100M, a large-scale dataset containing 101.07 million paired source–mask–target examples spanning diverse erase scenarios. Second, we pretrain LensErase from the text-to-image model Lens, adapting its strong generative prior to mask-based erasing. To eficiently incorporate the source image and erase mask, we propose Mask-Aware RoPE, a 3D rotary positional encoding that directly injects mask information into the source-image tokens without introducing a separate mask token sequence. Finally, we refine the model through supervised fine-tuning and reinforcement learning with contract-guided, region-aware rewards, and apply TraceFusion at inference to preserve high-fidelity content outside the efective edit region. Across six public benchmarks and the proposed LensErase-Bench, LensErase achieves substantial improvements under both reference-based objective metrics and reference-free VLM evaluation. We hope LensErase can serve as a foundational model for erase-related applications and systems.

Project Page: https://lensfamily.github.io/LensErase/

## 1 Introduction

Object erasing [1–3] is a fundamental image-editing capability with broad applications in photography, content creation, graphic design, and automated visual-production pipelines. Given an input image and a user-specified region, an ideal erase model should understand what the user intends to remove, eliminate all visual evidence associated with the target, and plausibly reconstruct content that is geometrically, semantically, and visually consistent with the surrounding scene, while preserving the remainder of the image.

However, building a single erase model that works reliably across the diverse requests encountered in real-world applications remains challenging. Handling targets across a wide range of spatial scales, from tiny objects occupying less than 1% of an image to dominant foreground objects covering nearly half of the scene, requires both precise local editing and reconstruction of large missing regions. Erasing diverse target types poses distinct reconstruction challenges: text and logos require removing fine strokes without damaging nearby details, while translucent watermarks require recovering underlying content from blended pixels. Multi-object removal requires clearing every selected instance while maintaining scene consistency across spatially separated edits. Removing associated visual efects, such as reflections and shadows, requires recognizing and removing target-dependent changes that may extend beyond the selection mask [1, 3].

We introduce LensErase, a general-purpose Erase Anything model conditioned only on a source image and an erase mask. To enable broad coverage, we construct LensErase-100M, a large-scale dataset containing 101.07 million paired training examples, each consisting of a source image, an erase mask, and a target image. LensErase-100M is two to three orders of magnitude larger than representative object-removal datasets, which contain approximately 100K–1M training examples [2, 4, 5]. Beyond scale, LensErase-100M is systematically constructed to cover diverse object categories, visual text, watermark types, object scales, image resolutions, and multi-object settings. We use domain-specific construction and validation pipelines to ensure reliable source– target supervision. General Data is constructed by inserting scene-compatible objects into clean real-world images and reversing the pairs, preserving the original images as removal targets. Specialized pipelines use alpha compositing for watermarks, layer rendering for posters, verified teacher edits for visual text, and foreground compositing for large objects.

LensErase builds upon a pretrained text-to-image model, Lens [6], leveraging its strong generative prior to reconstruct realistic and contextually coherent content in erased regions. Since both the condition and output are visual, we initialize both the image branch and condition branch from the pretrained Lens image branch. The condition branch receives the intact source image, allowing the model to identify the target and its associated efects, such as shadows and reflections. Mask-Aware RoPE injects region identity directly into the source-image tokens without introducing a separate mask token sequence. Region-weighted supervision further emphasizes the intended edit during pretraining, producing LensErase-Base. We then refine the model through supervised fine-tuning on curated high-quality, high-resolution data.

While large-scale supervised training equips the model with broad erase capabilities, removal quality can still vary across stochastic samples for the same input. Some outputs achieve complete and clean removal, while others may leave target fragments or residual efects, or alter neighboring content. We therefore further optimize the supervised model with reinforcement learning (RL) using DifusionNFT [7]. Region-aware rewards encourage complete removal, plausible reconstruction, and preservation of non-target content. At inference, TraceFusion estimates the efective edit region from the selection mask and source–output residuals, and selectively restores the source image outside this region to improve fidelity. Together, RL and TraceFusion yield LensErase, combining reliable removal with high-fidelity preservation.

We also introduce LensErase-Bench, a benchmark of 500 removal cases spanning natural images, posters, watermarks, and visual text. Its reference-free visual evaluation assesses removal correctness, reconstruction quality, content preservation and visual cleanliness, complementing reference-based metrics when multiple plausible reconstructions exist. We evaluate LensErase against 12 existing methods on LensErase-Bench and six public benchmarks. LensErase achieves substantial gains under both reference-based objective metrics and reference-free VLM evaluation, demonstrating strong and consistent performance across diverse erase scenarios. We aim for LensErase to serve as a general-purpose erase model for diverse visual-generation and editing systems.

## 2 Data

## 2.1 Pre-Training Data: LensErase-100M

We construct LensErase-100M, a large-scale pre-training dataset comprising 101.07M data triplets, where each triplet is represented as $( I _ { \mathrm { i n p u t } } , I _ { \mathrm { o u t p u t } } , M )$ , denoting the input image, erased target image, and corresponding erase mask, respectively. LensErase-100M covers five major data categories: General Data (77.6M), Watermark Data (9.99M), Large-Mask Data (4.02M), Visual Text Data (1.44M), and Poster Data (8.02M). Figure 2 shows the distributions of mask area ratios and mask counts in LensErase-100M. Below, we briefly introduce the construction pipeline for each data category, with additional details and filtering criteria provided in Appendix B.

General Data. We construct general objectremoval pairs by adding an object to a real image and reversing the pair. We observe that general-purpose editing models perform addition more reliably than removal, which often leaves target remnants, changes unrelated backgrounds, or produces blurry fills and artifacts. Using the original real image

![](images/6177cc2e2c6be8a25df1c9db51e25419d05e40e5541a9148900533e249e9fe15.jpg)

![](images/0dc5dd503b2f9f1e0e30295d87af55fdf002f23a6b09d42cf42d208728a4c51e.jpg)  
Figure 2 Left: Distribution of mask area ratios. Right: Distribution of mask counts.

as the clean target grounds supervision in the real-image distribution and encourages realistic outputs. The construction pipeline is as follows.

1. Object Taxonomy. We define a hierarchical taxonomy with 15 coarse categories (e.g., Animal, Furniture, and Infrastructure), each further divided into multiple fine-grained object classes.

2. Instruction Generation. We first build an image-caption pool $\mathcal { D } = ( I , T )$ . For each sample, we sample three distinct coarse categories. Given (I, T) and the three sampled categories, GPT-4.1 [8] is instructed to select an object that is absent from I, belongs to one of these categories, and can be naturally inserted into the scene. It outputs the corresponding coarse category, object class, and an insertion instruction P (e.g., “add a dog on the grass”).

3. Dynamic Balancing. Sampling is dynamically adjusted according to accumulated category/object statistics. Over-represented categories or object classes are excluded from subsequent sampling to maintain data diversity and balance.

4. Object Insertion. Given the original image I and insertion instruction P, we use an of-the-shelf image editing model $G _ { \mathrm { a d d } }$ to synthesize an image with the target object inserted. We use the inserted image as the input to the erase model, $I _ { \mathrm { i n p u t } } = G _ { \mathrm { a d d } } ( I , P )$ , and the original image as the desired erase output, $I _ { \mathrm { o u t p u t } } = I$

5. Coarse-to-Fine Mask Generation. We threshold low-frequency DCT and DINOv3 [9] feature diferences between $I _ { \mathrm { i n p u t } }$ and $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ to obtain complementary binary change maps. Their union defines a high-recall change region, while the interior of their overlap provides reliable positive-point candidates. Negative points are sampled outside the union. SAM2 [10] uses these prompts to produce the final object-selection mask M.

Watermark Data. We construct watermark-removal data through the following pipeline.

1. Watermark Collection. We collect a diverse set of commonly seen watermark patterns in real-world images, including logo watermarks, text watermarks, repeated watermarks, diagonal cross-image watermarks, stripe-style watermarks, and small corner marks.

2. Watermark Synthesis. For each clean image, we randomly select a watermark asset or text. We then synthesize the watermark with randomized font, color, position, scale, rotation, opacity, stroke, and overall layout to cover diverse real-world watermark appearances.

3. Triplet Construction. Since the watermark is rendered programmatically, we do not estimate the mask from image diferences. We treat the watermarked image as the erase-model input, the clean image as the erase target, and the rendered region as the mask, i.e., $I _ { \mathrm { i n p u t } }$ is the synthesized watermarked image, $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ is the original clean image, and M is the watermark mask.

Large-Mask Data. We observe that large-mask samples are relatively under-represented in the General Data, since current general-purpose editing models struggle to insert very large objects naturally into a scene through prompting alone without occluding existing objects. We therefore construct an additional large-mask subset using a controllable copy-paste pipeline.

1. Foreground Generation. We define a foreground-object taxonomy containing 25 coarse categories and approximately 2.5K fine-grained classes. We randomly select a fine-grained class and generate a matching object on a white background using an image generation model. We then apply BiRefNet [11] to extract the foreground object and its alpha mask.

2. Copy-Paste. We sample a clean background image I and apply YOLOE [12] to detect existing objects in the scene. The generated foreground is then scaled to occupy a relatively large image region and pasted preferentially into areas that do not overlap with detected objects.

3. Triplet Construction. We use the composite image after copy-paste as the erase-model input $I _ { \mathrm { i n p u t } }$ , the original background image as the clean target $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ , and the transformed foreground alpha mask as the erase mask M.

Visual Text Data. We construct visual-text removal data through the following pipeline.

1. Text Detection. Given an image I containing visual text, we first apply PaddleOCR [13] to detect text regions and their bounding boxes.

2. Text Editing. We use an image editing model with a prompt that instructs it to erase text while preserving its carriers and other scene content, producing a teacher-edited image <sup>˜</sup>I.

3. Candidate Removal Identification. We run PaddleOCR again on <sup>˜</sup>I and compare the OCR results before and after editing. Text regions detected in I but no longer matched in $\tilde { I }$ are treated as candidate removal regions. We further use SSIM to exclude candidates that exhibit insuficient structural change despite the OCR mismatch.

4. Poisson Blending. We randomly select a subset of eligible text groups and blend the corresponding regions from <sup>˜</sup>I into the original image I, preserving unselected text and surrounding content. Poisson blending [14] reduces color seams introduced by the editing model.

5. Triplet Construction. We use the original image with text as the erase-model input, i.e., $I _ { \mathrm { i n p u t } } = I .$ The Poisson-blended image is used as the target $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ , and the union of the selected text regions, with a small dilation, forms the mask M. The resulting triplets are accepted as valid removal pairs after the quality filtering detailed in Appendix B.1.4.

Poster Data. We construct poster-element removal data by modeling a poster as a structured set of layers with explicit z-order.

1. Canvas and Layout Sampling. We first sample a canvas from either a natural image or a design background, and randomly choose a canvas aspect ratio and a layout template.

2. Element Generation. We populate each poster with approximately 4–9 elements. The element types include title, subtitle, paragraph, label, price, date, URL, basic shape, decorative strip, logo, QR code, and chart. For each element, we randomly sample its content and attributes, including font, language, color, opacity, and other style parameters.

3. Structured Layer Composition. All elements are organized as a structured layer set with explicit z-order, which determines their spatial arrangement and mutual occlusion. Under a fixed background, layout, and layer configuration, we render the complete poster.

4. Element Deletion. We randomly remove a subset of elements while keeping the same background, layout, and remaining layer states unchanged, and render the edited poster after deletion.

5. Triplet Construction. We use the complete poster as the erase-model input $I _ { \mathrm { i n p u t } }$ , the poster after deleting the selected elements as the target image $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ , and the visible regions of the removed elements as the mask M. The mask is determined according to the z-order of the layered poster, so that only the actually visible pixels of the removed elements are included.

## 2.2 Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) Data

SFT Data. We follow the pretraining pipelines in Section 2.1, prioritizing high-resolution images. All candidate triplets undergo GPT-5.6 Sol [15] screening for mask validity, edit correctness, nontarget preservation, and reconstruction quality (Appendix B.2). We do not include Large-Mask Data in ${ \mathrm { S F T } } ,$ as these examples are synthetically generated and are mainly used to improve the coverage of large-mask cases during pre-training. For Poster Data, instead of using the synthetic generation pipeline adopted during pre-training, we collect high-quality poster examples designed by professional designers, providing more realistic layouts, typography, and visual compositions. The SFT set contains 140K examples: 80K General Data, 20K Watermark Data, 20K Visual Text Data, and 20K Poster Data.

RL Data. We curate 19K image–mask tasks using source images disjoint from SFT. General Data contains 8K dificulty-filtered Hard/Extreme cases and 5K targeted supplements. Manual analysis of LensErase-SFT failures guides collection of new images with similar challenges. For annotated General Data tasks, GPT-5.6 Sol [15] localizes objects, including those with associated efects. SAM2 converts their bounding boxes into masks. GPT then assesses dificulty from the validated source–mask pairs without observing erase-model outputs. Poster, Watermark, and Visual Text contribute 2K tasks each through high-resolution layered removal, controlled synthesis, and screened text removal with text-centered cropping, respectively. The model receives only $I _ { \mathrm { i n p u t } }$ and M. General Data requires no paired target. Poster and Watermark use deterministic clean references, whereas Visual Text uses verified weak references for rewards. Appendix B.3 details the pipelines.

## 2.3 Evaluation Dataset: LensErase-Bench

LensErase-Bench. Most existing object-removal benchmarks evaluate outputs against paired reference images obtained through synthesis or real-world capture. Synthesized references may contain artifacts, while separately captured references may difer in lighting or scene details. Since object removal also admits multiple valid outcomes, agreement with a single reference may not fully reflect removal quality. To complement reference-based metrics, we introduce LensErase-Bench, a benchmark of 500 carefully curated removal cases evaluated without paired ground-truth target images. A VLM evaluates each output against the source image and selection mask. The Poster subset uses layered designs from Crello [16]. We also include image–mask pairs selected from existing segmentation datasets [17–19]. We normalize the selected images to a shorter side above 512 pixels and a longer side below 2048 pixels while preserving aspect ratio. We carefully curate the benchmark to provide balanced coverage across multiple dimensions, including application scenarios, mask area ratio, image resolution, selected-target count, and erase dificulty. These factors are jointly considered to ensure that LensErase-Bench is both diverse and representative of practical applications. Each evaluation sample consists of an input image $I _ { \mathrm { i n p u t } }$ and an erase mask M. Appendix C details the selection constraints and distributions.

Evaluation Metrics. We use GPT-5.6 Sol [15] as the VLM judge to evaluate each output against the input image and selection mask along four dimensions: Removal, the completeness of removing selected targets and their associated efects; Inpainting, plausible completion consistent with the surrounding context; Preservation, retention of non-target content; and Cleanliness, absence of artifacts and unrelated additions. Each dimension receives an integer rating from 0 to 4, rescaled to 0–100 for reporting. Overall weights these dimensions by 40%, 20%, 20%, and 20%, respectively. We additionally report the High-Fidelity Removal Rate (HiFi Removal, %), the percentage of all valid cases that achieve complete removal while meeting the quality requirements for inpainting, preservation, and visual cleanliness. Specifically, a case is successful when Removal = 4 and Inpainting, Preservation, and Cleanliness are all ≥ 3 on the original 0–4 scale. Mean scores are computed over all valid cases with equal weight per image. Detailed scoring rules and the evaluation protocol are provided in Appendix C.2.

![](images/7698149edf284de893606f308ae4e2987c9c04cafe8a7665a53ea78cfbbbca74.jpg)  
Figure 3 Architecture of LensErase. LensErase follows a two-branch MMDiT architecture, where the image branch processes the noisy target latent and the condition branch processes the source latent. The proposed Mask-Aware RoPE incorporates the pooled binary mask into the condition branch by encoding masked and unmasked source regions along an additional modality dimension.

## 3 Method

Given a source image $I _ { \mathrm { i n p u t } }$ and a binary selection mask M, LensErase produces an output image $\hat { I } _ { \mathrm { o u t p u t } }$ that removes the selected objects and their associated visual efects, while preserving the remaining scene and plausibly completing the removed regions. Figure 3 shows the architecture.

## 3.1 Pretraining: Adapting Generation to Erasure

Architecture and Initialization. We initialize LensErase from Lens [6], a text-to-image model whose MMDiT backbone contains a condition branch for the text modality and an image branch for the visual modality. LensErase follows the same backbone architecture, but both branches are initialized from the pretrained image branch of Lens. This is because, in the erase task, the condition input, including the mask M and source image features, is also visual. Figure 3 illustrates the architecture of LensErase: the image branch processes the noisy target latent, while the condition branch processes the source latent. The erase mask M, with 1 denoting the masked region and 0 the unmasked region, is incorporated into the condition branch through the proposed Mask-Aware RoPE with negligible computational overhead.

We retain the complete source image, including the selected region. The selected object’s appearance and location help identify associated shadows and reflections, while semi-transparent overlays retain useful information about the underlying background. Masking out the selected region would therefore discard information useful for both efect removal and reconstruction.

Mask-Aware RoPE. We apply RoPE [20] to both noisy target tokens and source tokens. To encode the erase mask in the source tokens, we extend conventional 2D spatial RoPE to a 3D formulation by introducing an additional modality dimension. Specifically, we first downsample the binary mask M to the source-token grid via max pooling, yielding $m _ { i } \in \{ 0 , 1 \}$ for each source token. We then define the positional coordinate for source tokens as

$$
\mathbf { p } _ { i } ^ { \mathrm { c o n d } } = ( m _ { i } , h _ { i } , w _ { i } ) ,\tag{1}
$$

where $( h _ { i } , w _ { i } )$ are spatial coordinates and $m _ { i } = 1 ~ ( m _ { i } = 0 )$ denotes a masked (unmasked) source token. This directly encodes the mask into the positional representation without introducing additional mask tokens or embedding layers. To represent the noisy target tokens within the same 3D RoPE formulation, we assign them a proxy modality coordinate of 2:

$$
\mathbf { p } _ { i } ^ { \mathrm { i m g } } = ( 2 , h _ { i } , w _ { i } ) .\tag{2}
$$

Region-Weighted Flow Matching. We pretrain LensErase on LensErase-100M using a regionweighted flow-matching objective. Since object removal often afects only a small image region, a uniform objective can be dominated by unchanged background. Let E denote the VAE encoder, with target latent $z ^ { \star } = E ( I _ { \mathrm { o u t p u t } } )$ and source latent $z _ { \mathrm { s r c } } = E ( I _ { \mathrm { i n p u t } } )$ . Following flow matching [21], we construct $z _ { t } = ( 1 - t ) z ^ { \star } + t$ ϵ with target velocity $v ^ { \star } = \epsilon - z ^ { \star }$ , where $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ . Conditioned on $z _ { \mathrm { s r c } }$ and the erase mask M, the model predicts the velocity $v _ { \theta } ( z _ { t } , t , z _ { \mathrm { s r c } } ; M )$ . To emphasize the intended edit, we assign larger loss weights to tokens within the masked region:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { I _ { \mathrm { i n p u t } } , I _ { \mathrm { o u t p u t } } , M , t , \epsilon } \left[ \frac { \sum _ { i = 1 } ^ { N } \omega _ { i } \| v _ { \theta , i } ( z _ { t } , t , z _ { \mathrm { s r c } } ; M ) - v _ { i } ^ { \star } \| _ { 2 } ^ { 2 } } { \sum _ { i = 1 } ^ { N } \omega _ { i } } \right] .\tag{3}
$$

Here, i indexes the N target tokens and $\omega _ { i } = 1 + ( \lambda - 1 ) \tilde { M } _ { i }$ , where $\tilde { M }$ is the dilated mask downsampled to the token grid and $\lambda = 5$ . The normalization reduces sensitivity of the overall loss scale to mask size. Further pretraining details and notation are provided in Appendices D and E.6.

Supervised Fine-Tuning (SFT). After pretraining, we perform SFT on the curated SFT set using LoRA [22], while retaining the same region-weighted flow-matching objective. Training details are provided in Appendix E.1.

## 3.2 RL: Contract-Guided Erasure

The SFT model already produces strong removals, but its success can vary across stochastic samples for the same image–mask input. We therefore use RL to increase the likelihood of complete removal while preserving non-target content. Figure 6 in Appendix E.2 summarizes the reward pipeline.

1. Constructing the Erase Contract. For each General Data source image $I _ { \mathrm { i n p u t } }$ and selection mask M, we first construct a source-grounded, candidate-independent erase contract that specifies the intended edit. GPT-5.6 Sol [15] examines the source image, mask, and mask-centered context views to produce $\mathcal { K } = ( \mathcal { T } , \mathcal { E } , \mathcal { O } )$ . The removal targets T specify the selected entities and key visual components whose presence would indicate incomplete removal. The associated efects E describe target-dependent efects, such as shadows or reflections, that should also disappear. The protected objects O identify nearby independent content that should remain unchanged. Thus, T and E specify what should disappear, while O specifies what should be preserved. The same contract is shared across all rollouts for a given image–mask condition, ensuring consistent reward evaluation.

2. Policy Rollouts and Reward Computation. The SFT-initialized policy generates multiple stochastic candidates per image–mask pair, which are evaluated against the shared erase contract. GPT-5.6 Sol provides four semantic scores: Removal assesses whether targets in T and efects in E are fully removed; Reconstruction evaluates the plausibility and contextual consistency of synthesized content; Preservation measures retention of protected objects in O and other non-target content; and Cleanliness captures residual fragments, unwanted additions, and artifacts.

We additionally compute Outside Fidelity as source–candidate PSNR over expected unchanged regions. Reference-guided adaptations for other domains are detailed in Appendix E.2.

3. Removal-Aware Reward Gating. A weighted average can assign a high reward to an edit that performs well on reconstruction or preservation but still exhibits clear removal failures. We therefore first aggregate the five reward components, namely Removal, Reconstruction, Preservation, Cleanliness, and Outside Fidelity, into a base reward:

$$
b = \sum _ { p \in \mathcal { P } } \lambda _ { p } P _ { p } , \qquad \sum _ { p \in \mathcal { P } } \lambda _ { p } = 1 ,\tag{4}
$$

where $\mathcal { P } =$ {Removal, Reconstruction, Preservation, Cleanliness, OutsideFidelity}, and $P _ { p } \in [ 0 , 1 ]$ denotes the normalized score of reward component $p .$ We suppress the candidate index j. Component and subcriterion weights are renormalized over applicable terms, as detailed in Appendix E.2. The Removal component is further decomposed into multiple subcriteria. Each zero-valued applicable Removal subcriterion incurs an additional penalty factor of 0.5. Specifically, let

$$
n _ { 0 } = \sum _ { q \in \mathcal { Q } _ { \mathrm { R e m o v a l } } } \mathbb { 1 } [ s _ { q } = 0 ] ,\tag{5}
$$

where $\scriptstyle \mathcal { Q } _ { \mathrm { R e m o v a l } }$ denotes the set of applicable Removal subcriteria and $s _ { q } \in [ 0 , 1 ]$ is the normalized score of subcriterion $q .$ The final reward is

$$
r = 0 . 5 ^ { n _ { 0 } } b .\tag{6}
$$

This progressively penalizes candidates with multiple removal failures, reducing the extent to which strong performance on other criteria can compensate for them.

4. Optimization. We optimize the SFT-initialized LensErase model with DifusionNFT [7]. Implementation details are provided in Appendix E.3.

## 3.3 Post-Processing with TraceFusion

Although LensErase primarily modifies the selected target and associated efects, VAE reconstruction may introduce small unintended changes in otherwise unchanged regions. TraceFusion estimates the efective edit region from source–output residuals, preserves generated content there, and restores the source image elsewhere. This improves the fidelity of unchanged content while retaining intended edits that may extend beyond the input mask, such as removed shadows or reflections.

We compute pixel-wise residuals between the generated and source images after compensating for reconstruction bias in unchanged regions. On the 0–255 scale, pixels above 4 form candidate change components, of which only those containing suficient pixels above 10 are retained to suppress small or fragmented artifacts. This allows the estimated edit region to extend beyond the input mask to associated efects such as shadows or reflections. Detailed criteria are provided in Appendix E.5.

We then combine the detected residual support with a dilated input mask to form the primary edit support. Because changes in dark regions can produce weak residuals in the original intensity space, we additionally apply the same detector after a shared $\gamma = 0 . 7 5$ transform to both images. This gamma branch only supplements the primary support: newly detected components are retained only when they connect to the primary edit support and predominantly lie in dark regions of both images.

Finally, we feather the boundary of the resulting edit support to obtain a blending weight $\alpha \in [ 0 , 1 ]$ . The generated image is preserved inside the estimated edit region, while the source

Table 1 Comparison with state-of-the-art object removal methods on six public benchmarks. For each benchmark, we report the three baselines with the lowest mean rank across the eight metrics together with LensErase. Complete results for all 12 baselines are provided in Tables 10-15 of the Appendix. Cell backgrounds mark the best, second-best, and third-best scores among these four methods.
<table><tr><td rowspan=1 colspan=12>Method                                           PSNR↑ FID↓ CMMD↓ LPIPS↓ SSIM↑ DISTS↓CLIP-I↑DINO↑</td></tr><tr><td rowspan=5 colspan=6>OBER-Test [3]ObjectClear [3]                                     33.05   19.16OSOR [23]                                          30.14   27.82OmniPaint [24]                                     28.79  23.68LensErase                                          34.96   16.97</td><td rowspan=1 colspan=6></td></tr><tr><td rowspan=1 colspan=1>33.05</td><td rowspan=1 colspan=1>19.16</td><td rowspan=1 colspan=1>0.0132</td><td rowspan=1 colspan=1>0.0338</td><td rowspan=1 colspan=1>0.9353</td><td rowspan=1 colspan=1>0.0217</td><td rowspan=1 colspan=1>0.9894</td><td rowspan=1 colspan=1>0.9839</td></tr><tr><td rowspan=1 colspan=1>30.14</td><td rowspan=1 colspan=1>27.82</td><td rowspan=1 colspan=1>0.0674</td><td rowspan=1 colspan=1>0.0524</td><td rowspan=1 colspan=1>0.8748</td><td rowspan=1 colspan=1>0.0377</td><td rowspan=1 colspan=1>0.9827</td><td rowspan=1 colspan=1>0.9693</td></tr><tr><td rowspan=1 colspan=1>28.79</td><td rowspan=1 colspan=1>23.68</td><td rowspan=1 colspan=1>0.0829</td><td rowspan=1 colspan=1>0.0549</td><td rowspan=1 colspan=1>0.8575</td><td rowspan=1 colspan=1>0.0427</td><td rowspan=1 colspan=1>0.9836</td><td rowspan=1 colspan=1>0.9803</td></tr><tr><td rowspan=1 colspan=1>34.96</td><td rowspan=1 colspan=1>16.97</td><td rowspan=1 colspan=1>0.0073</td><td rowspan=1 colspan=1>0.0256</td><td rowspan=1 colspan=1>0.9543</td><td rowspan=1 colspan=1>0.0176</td><td rowspan=1 colspan=1>0.9930</td><td rowspan=1 colspan=1>0.9839</td></tr><tr><td rowspan=5 colspan=4>ObjectClear [3]OSOR [23]OmniPaint [24]LensErase</td><td rowspan=1 colspan=8>RORD-Val-343 [25]</td></tr><tr><td rowspan=1 colspan=1>26.21</td><td rowspan=1 colspan=1>25.42</td><td rowspan=1 colspan=1>0.0511</td><td rowspan=1 colspan=1>0.1163</td><td rowspan=1 colspan=1>0.8577</td><td rowspan=1 colspan=1>0.0657</td><td rowspan=1 colspan=1>0.9704</td><td rowspan=1 colspan=1>0.9566</td></tr><tr><td rowspan=2 colspan=3></td><td rowspan=1 colspan=1>25.67</td><td rowspan=1 colspan=1>26.33</td><td rowspan=1 colspan=1>0.0390</td><td rowspan=1 colspan=1>0.1037</td><td rowspan=1 colspan=1>0.8138</td><td rowspan=1 colspan=1>0.0542</td><td rowspan=1 colspan=1>0.9663</td><td rowspan=1 colspan=1>0.9475</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>23.64</td><td rowspan=1 colspan=1>27.45</td><td rowspan=1 colspan=1>0.0334</td><td rowspan=1 colspan=1>0.1119</td><td rowspan=1 colspan=1>0.7823</td><td rowspan=1 colspan=1>0.0623</td><td rowspan=1 colspan=1>0.9672</td><td rowspan=1 colspan=1>0.9541</td></tr><tr><td rowspan=1 colspan=1>28.23</td><td rowspan=1 colspan=1>20.53</td><td rowspan=1 colspan=1>0.0272</td><td rowspan=1 colspan=1>0.0601</td><td rowspan=1 colspan=1>0.9053</td><td rowspan=1 colspan=1>0.0357</td><td rowspan=1 colspan=1>0.9715</td><td rowspan=1 colspan=1>0.9568</td></tr><tr><td rowspan=5 colspan=4>OSOR [23]OmniPaint [24]ObjectClear [3]LensErase</td><td rowspan=1 colspan=2>RemovalBench [</td><td rowspan=1 colspan=1>4]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>25.80</td><td rowspan=1 colspan=1>43.89</td><td rowspan=1 colspan=1>0.0275</td><td rowspan=1 colspan=1>0.1935</td><td rowspan=1 colspan=1>0.7656</td><td rowspan=1 colspan=1>0.0882</td><td rowspan=1 colspan=1>0.9838</td><td rowspan=1 colspan=1>0.9597</td></tr><tr><td rowspan=1 colspan=1>24.57</td><td rowspan=1 colspan=1>43.25</td><td rowspan=1 colspan=1>0.0265</td><td rowspan=1 colspan=1>0.2103</td><td rowspan=1 colspan=1>0.7535</td><td rowspan=1 colspan=1>0.0998</td><td rowspan=1 colspan=1>0.9821</td><td rowspan=1 colspan=1>0.9663</td></tr><tr><td rowspan=1 colspan=1>25.99</td><td rowspan=1 colspan=1>46.61</td><td rowspan=1 colspan=1>0.0248</td><td rowspan=1 colspan=1>0.2853</td><td rowspan=1 colspan=1>0.7794</td><td rowspan=1 colspan=1>0.1428</td><td rowspan=1 colspan=1>0.9853</td><td rowspan=1 colspan=1>0.9718</td></tr><tr><td rowspan=1 colspan=1>28.16</td><td rowspan=1 colspan=1>33.67</td><td rowspan=1 colspan=1>0.0116</td><td rowspan=1 colspan=1>0.1125</td><td rowspan=1 colspan=1>0.8306</td><td rowspan=1 colspan=1>0.0586</td><td rowspan=1 colspan=1>0.9874</td><td rowspan=1 colspan=1>0.9737</td></tr><tr><td rowspan=5 colspan=4>OmniPaiOSOR [23]OmniPaint [24]GeoRemover [26]LensErase</td><td rowspan=1 colspan=1>nt-Bench</td><td rowspan=1 colspan=1>(captured</td><td rowspan=1 colspan=6>subset) [24]</td></tr><tr><td rowspan=1 colspan=1>26.02</td><td rowspan=1 colspan=1>37.52</td><td rowspan=1 colspan=1>0.0687</td><td rowspan=1 colspan=1>0.1336</td><td rowspan=1 colspan=1>0.8421</td><td rowspan=1 colspan=1>0.0566</td><td rowspan=1 colspan=1>0.9617</td><td rowspan=1 colspan=1>0.9506</td></tr><tr><td rowspan=1 colspan=1>23.42</td><td rowspan=1 colspan=1>40.65</td><td rowspan=1 colspan=1>0.0743</td><td rowspan=1 colspan=1>0.1484</td><td rowspan=1 colspan=1>0.8193</td><td rowspan=1 colspan=1>0.0644</td><td rowspan=1 colspan=1>0.9562</td><td rowspan=1 colspan=1>0.9423</td></tr><tr><td rowspan=1 colspan=1>24.87</td><td rowspan=1 colspan=1>38.54</td><td rowspan=1 colspan=1>0.1376</td><td rowspan=1 colspan=1>0.1575</td><td rowspan=1 colspan=1>0.8173</td><td rowspan=1 colspan=1>0.0704</td><td rowspan=1 colspan=1>0.9524</td><td rowspan=1 colspan=1>0.9494</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>26.22</td><td rowspan=1 colspan=1>33.29</td><td rowspan=1 colspan=1>0.0245</td><td rowspan=1 colspan=1>0.1257</td><td rowspan=1 colspan=1>0.8521</td><td rowspan=1 colspan=1>0.0544</td><td rowspan=1 colspan=1>0.9660</td><td rowspan=1 colspan=1>0.9595</td></tr><tr><td rowspan=5 colspan=4>OSOR [23]OmniPaint [24]ObjectClear [3]LensErase</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>AnimeEr</td><td rowspan=1 colspan=1>aseBench1</td><td rowspan=1 colspan=1>[23]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>27.96</td><td rowspan=1 colspan=1>24.45</td><td rowspan=1 colspan=1>0.0094</td><td rowspan=1 colspan=1>0.0967</td><td rowspan=1 colspan=1>0.8870</td><td rowspan=1 colspan=1>0.0475</td><td rowspan=1 colspan=1>0.9900</td><td rowspan=1 colspan=1>0.9751</td></tr><tr><td rowspan=1 colspan=1>25.58</td><td rowspan=1 colspan=1>27.76</td><td rowspan=1 colspan=1>0.0138</td><td rowspan=1 colspan=1>0.1195</td><td rowspan=1 colspan=1>0.8581</td><td rowspan=1 colspan=1>0.0636</td><td rowspan=1 colspan=1>0.9874</td><td rowspan=1 colspan=1>0.9764</td></tr><tr><td rowspan=1 colspan=1>25.42</td><td rowspan=1 colspan=1>36.76</td><td rowspan=1 colspan=1>0.0121</td><td rowspan=1 colspan=1>0.1791</td><td rowspan=1 colspan=1>0.8417</td><td rowspan=1 colspan=1>0.0978</td><td rowspan=1 colspan=1>0.9914</td><td rowspan=1 colspan=1>0.9821</td></tr><tr><td rowspan=1 colspan=1>30.50</td><td rowspan=1 colspan=1>17.20</td><td rowspan=1 colspan=1>0.0094</td><td rowspan=1 colspan=1>0.0527</td><td rowspan=1 colspan=1>0.9398</td><td rowspan=1 colspan=1>0.0159</td><td rowspan=1 colspan=1>0.9920</td><td rowspan=1 colspan=1>0.9798</td></tr><tr><td rowspan=5 colspan=4>OSOR [23]OmniEraser [4]RORem-mixed [5]LensErase</td><td rowspan=1 colspan=2>TextEraseBench [</td><td rowspan=1 colspan=3>23]</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>32.04</td><td rowspan=1 colspan=1>18.52</td><td rowspan=1 colspan=1>0.0093</td><td rowspan=1 colspan=1>0.0728</td><td rowspan=1 colspan=1>0.9126</td><td rowspan=1 colspan=1>0.0412</td><td rowspan=1 colspan=1>0.9843</td><td rowspan=1 colspan=1>0.9771</td></tr><tr><td rowspan=1 colspan=1>30.70</td><td rowspan=1 colspan=1>21.97</td><td rowspan=1 colspan=1>0.0195</td><td rowspan=1 colspan=1>0.0763</td><td rowspan=1 colspan=1>0.9005</td><td rowspan=1 colspan=1>0.0435</td><td rowspan=1 colspan=1>0.9812</td><td rowspan=1 colspan=1>0.9758</td></tr><tr><td rowspan=1 colspan=1>29.50</td><td rowspan=1 colspan=1>24.83</td><td rowspan=1 colspan=1>0.0178</td><td rowspan=1 colspan=1>0.1034</td><td rowspan=1 colspan=1>0.8292</td><td rowspan=1 colspan=1>0.0584</td><td rowspan=1 colspan=1>0.9815</td><td rowspan=1 colspan=1>0.9744</td></tr><tr><td rowspan=1 colspan=1>36.88</td><td rowspan=1 colspan=1>13.51</td><td rowspan=1 colspan=1>0.0091</td><td rowspan=1 colspan=1>0.0339</td><td rowspan=1 colspan=1>0.9654</td><td rowspan=1 colspan=1>0.0135</td><td rowspan=1 colspan=1>0.9906</td><td rowspan=1 colspan=1>0.9822</td></tr></table>

image is restored elsewhere. Blending is performed in linear RGB:

$$
\hat { I } _ { \mathrm { o u t p u t } } ^ { \mathrm { T F } } = \mathcal { C } ^ { - 1 } \Big [ \alpha \odot \mathcal { C } \big ( \hat { I } _ { \mathrm { o u t p u t } } \big ) + ( 1 - \alpha ) \odot \mathcal { C } ( I _ { \mathrm { i n p u t } } ) \Big ] ,\tag{7}
$$

where C converts sRGB values to linear RGB.

## 4 Experiments

Benchmarks. We evaluate on seven benchmarks, including six public benchmarks: OBER-Test [3], RORD-Val-343 [25], RemovalBench [4], OmniPaint-Bench [24], AnimeEraseBench, and TextEraseBench [23], together with our proposed LensErase-Bench (Section 2.3). For OmniPaint-Bench, we use only its 300 captured cases, excluding the RORD-derived subset to avoid overlap with RORD-Val-343.

Baselines. We compare against 12 publicly released methods: OSOR [23], ObjectClear [3], DORS [27], PANDORA [28], GeoRemover [26], SmartEraser [2], RORem-mixed [5], EntityErasure [29], Attentive Eraser [30], DesignEdit [31], OmniEraser [4], and OmniPaint [24]. Related methods are reviewed in Appendix A, and detailed baseline configurations are provided in Appendix F.2.

Implementation Details. Pretraining takes approximately two weeks on 96 NVIDIA A100 GPUs. SFT and RL are each conducted on 16 GPUs, taking approximately three and five days, respectively. Additional implementation details are provided in Appendix E.

Metrics. For the six public benchmarks with ground-truth target images, we report PSNR, SSIM, LPIPS, DISTS, CLIP-I, DINO, FID, and CMMD. For our LensErase-Bench, we adopt the evaluation metrics described in Section 2.3.

Table 2 Comparison on LensErase-Bench. We report the six representative baselines together with LensErase. Complete results for all 12 baselines are provided in Table 16 of the Appendix.
<table><tr><td>Method</td><td>Removal ↑</td><td>Inpainting ↑</td><td>Preservation ↑</td><td>Cleanliness ↑</td><td>Overall↑</td><td>HiFi Removal (%) ↑</td></tr><tr><td>OSOR [23]</td><td>73.85</td><td>60.45</td><td>75.55</td><td>73.55</td><td>71.45</td><td>32.60</td></tr><tr><td>OmniPaint [24]</td><td>69.20</td><td>66.15</td><td>80.20</td><td>83.20</td><td>73.59</td><td>42.80</td></tr><tr><td>ObjectClear [3]</td><td>73.95</td><td>61.45</td><td>73.80</td><td>75.20</td><td>71.67</td><td>28.40</td></tr><tr><td>GeoRemover [26]</td><td>54.30</td><td>49.85</td><td>77.05</td><td>78.00</td><td>62.70</td><td>25.60</td></tr><tr><td>OmniEraser [4]</td><td>55.65</td><td>47.95</td><td>77.55</td><td>57.65</td><td>58.89</td><td>22.60</td></tr><tr><td>RORem-mixed [5]</td><td>73.95</td><td>59.70</td><td>74.05</td><td>67.10</td><td>69.75</td><td>27.40</td></tr><tr><td>LensErase</td><td>95.15</td><td>92.40</td><td>96.35</td><td>97.05</td><td>95.22</td><td>82.80</td></tr></table>

Table 3 Stage-wise ablation of LensErase. We report performance after pre-training, SFT, RL, and TraceFusion. Results on LensErase-Bench are shown with all evaluation metrics, while results on public benchmarks report the average PSNR and LPIPS across the six public benchmarks.
<table><tr><td rowspan="2">Model</td><td colspan="6">LensErase-Bench</td><td colspan="2">Public Benchmarks</td></tr><tr><td>Removal ↑</td><td>Inpainting ↑</td><td>Preservation ↑</td><td>Cleanliness ↑</td><td>Overall ↑</td><td>HiFi Removal ↑</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>Pre-Training</td><td>82.35</td><td>84.65</td><td>91.05</td><td>92.15</td><td>86.51</td><td>65.80</td><td>28.03</td><td>0.0855</td></tr><tr><td>+SFT</td><td>88.65</td><td>87.15</td><td>92.05</td><td>93.05</td><td>89.91</td><td>70.80</td><td>28.14</td><td>0.0826</td></tr><tr><td>+RL</td><td>94.95</td><td>92.65</td><td>93.35</td><td>96.15</td><td>94.41</td><td>81.60</td><td>28.70</td><td>0.0816</td></tr><tr><td>+TraceFusion</td><td>95.15</td><td>92.40</td><td>96.35</td><td>97.05</td><td>95.22</td><td>82.80</td><td>30.82</td><td>0.0684</td></tr></table>

Table 4 Comparison of mask injection strategies. Mask-Aware RoPE is compared with channel- and token-concatenation baselines, all evaluated at the pre-training stage. Both baselines obtain mask features via pixel unshufle of 16×16 patches. Channel Concat concatenates source and mask features along channels and fuses them with an MLP before condition projection. Token Concat places source tokens alongside noisy tokens in the image branch and projects mask features to the condition branch. Both use positional indices independent of mask values.
<table><tr><td></td><td colspan="6">LensErase-Bench</td><td colspan="2">Public Benchmarks</td></tr><tr><td>Model</td><td>Removal ↑</td><td>Inpainting ↑</td><td>Preservation ↑</td><td>Cleanliness ↑</td><td>Overall ↑</td><td>HiFi Removal ↑</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>Mask-Aware RoPE</td><td>82.35</td><td>84.65</td><td>91.05</td><td>92.15</td><td>86.51</td><td>65.80</td><td>28.03</td><td>0.0855</td></tr><tr><td>Channel Concat</td><td>81.50</td><td>84.55</td><td>82.90</td><td>91.15</td><td>84.32</td><td>58.80</td><td>27.84</td><td>0.0869</td></tr><tr><td>Token Concat</td><td>77.50</td><td>76.85</td><td>84.90</td><td>87.15</td><td>80.78</td><td>51.40</td><td>27.41</td><td>0.0965</td></tr></table>

State-of-the-Art Performance Across Six Public Benchmarks. As shown in Table 1, LensErase achieves state-of-the-art performance across six public benchmarks. For clarity, we report only the three baselines with the lowest mean rank across the eight metrics per benchmark. Full comparisons with all 12 baselines are provided in Tables 10–15 of the Appendix.

State-of-the-Art Performance on LensErase-Bench. We further compare LensErase with 12 publicly released baselines on the proposed LensErase-Bench, which contains more challenging and diverse removal cases. As shown in Table 2, LensErase achieves the best overall performance. For clarity, we report six representative baselines together with LensErase in the main table. Complete comparisons with all 12 baselines are provided in Table 16 of the Appendix.

Stage-wise Performance. Table 3 shows that HiFi Removal increases from 65.80% after pretraining to 70.80% after SFT and 81.60% after RL. TraceFusion further improves mean PSNR from 28.70 to 30.82 dB and LPIPS from 0.0816 to 0.0684, while reaching 82.80% HiFi Removal. Appendix G reports reward-component ablations.

Efective Design for Mask Injection. Section 3.1 introduces our Mask-Aware RoPE for injecting mask information. In Table 4, we compare it with two variants, Channel Concat and Token Concat. Appendix D provides further pretraining ablations.

Visualizations. Appendix Figures 7–17 compare LensErase with closed-source commercial models (mask-based GPT-Image 2.5-Sunburst [32] and Flux2-Erase [33]) and open-source baselines. Appendix J shows further examples using hand-drawn and SAM masks.

## References

[1] Daniel Winter, Matan Cohen, Shlomi Fruchter, Yael Pritch, Alex Rav-Acha, and Yedid Hoshen. Objectdrop: Bootstrapping counterfactuals for photorealistic object removal and insertion. In European Conference on Computer Vision, pages 112–129. Springer, 2024.

[2] Longtao Jiang, Zhendong Wang, Jianmin Bao, Wengang Zhou, Dongdong Chen, Lei Shi, Dong Chen, and Houqiang Li. Smarteraser: Remove anything from images using masked-region guidance. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24452–24462. IEEE, 2025.

[3] Jixin Zhao, Zhouxia Wang, Peiqing Yang, and Shangchen Zhou. Precise object and efect removal with adaptive target-aware attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19370–19379, 2026.

[4] Runpu Wei, Zijin Yin, Shuo Zhang, Lanxiang Zhou, Xueyi Wang, Chao Ban, Tianwei Cao, Hao Sun, Zhongjiang He, Kongming Liang, et al. Omnieraser: Remove objects and their efects in images with paired video-frame data. arXiv preprint arXiv:2501.07397, 2025.

[5] Ruibin Li, Tao Yang, Song Guo, and Lei Zhang. Rorem: Training a robust object remover with human-in-the-loop. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14024–14035. IEEE, 2025.

[6] Dong Chen, Fangyun Wei, Ziyu Wan, Dongdong Chen, Jiawei Zhang, Jinjing Zhao, Sirui Zhang, Yang Yue, Zhiyang Liang, Baining Guo, et al. Lens: Rethinking training eficiency for foundational text-to-image models. arXiv preprint arXiv:2605.21573, 2026.

[7] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Difusionnft: Online difusion reinforcement with forward process. In International Conference on Learning Representations, volume 2026, pages 134129–134150, 2026.

[8] OpenAI. Introducing GPT-4.1 in the API. https://openai.com/index/gpt-4-1/, 2025.

[9] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

[10] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. URL https://arxiv.org/abs/2408.00714.

[11] Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation. CAAI Artificial Intelligence Research, 3:9150038, 2024.

[12] Ao Wang, Lihao Liu, Hui Chen, Zijia Lin, Jungong Han, and Guiguang Ding. Yoloe: Real-time seeing anything. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 24591–24602. IEEE, 2025.

[13] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang, Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, and Yanjun Ma. Paddleocr 3.0 technical report, 2025. URL https://arxiv.org/abs/2507.05595.

[14] Patrick Pérez, Michel Gangnet, and Andrew Blake. Poisson image editing. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 577–582. 2023.

[15] OpenAI. GPT-5.6 Sol. https://developers.openai.com/api/docs/models/gpt-5.6-sol, 2026. Accessed: 2026-09-01.

[16] Kota Yamaguchi. Canvasvae: Learning to generate vector graphic documents. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 5461–5469. IEEE, 2021.

[17] Lu Qi, Jason Kuen, Tiancheng Shen, Jiuxiang Gu, Weidong Guo, Jiaya Jia, Zhe Lin, and Ming-Hsuan Yang. High-quality entity segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

[18] Xinhao Deng, Pingping Zhang, Wei Liu, and Huchuan Lu. Recurrent multi-scale transformer for high-resolution salient object detection. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7413–7423, 2023. doi: 10.1145/3581783.3611983.

[19] Chenxi Xie, Changqun Xia, Mingcan Ma, Zhirui Zhao, Xiaowu Chen, and Jia Li. Pyramid grafting network for one-stage high resolution saliency detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[20] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[21] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[23] Qinming Zhou, Chenxi Sun, Deyang Kong, Junhao He, Xiangheng Tang, Peike Yu, Haotian Wu, Leilei Cao, and Linfeng Zhang. Osor: One-step difusion inpainting for efect-aware object removal. In European Conference on Computer Vision, pages 367–384. Springer, 2026.

[24] Yongsheng Yu, Ziyun Zeng, Haitian Zheng, and Jiebo Luo. Omnipaint: Mastering objectoriented editing via disentangled insertion-removal inpainting. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 17324–17334. IEEE, 2025.

[25] Min-Cheol Sagong, Yoon-Jae Yeo, Seung-Won Jung, and Sung-Jea Ko. Rord: A real-world object removal dataset. In Proceedings of the British Machine Vision Conference (BMVC), 2022.

[26] Zixin Zhu, Haoxiang Li, Xuelu Feng, He Wu, Chunming Qiao, and Junsong Yuan. Georemover: Removing objects and their causal visual artifacts. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum? id=RnfyqrkOxD.

[27] Haitong Tang, Haipeng Liu, and Yang Wang. Dors: Dynamic attention routing for difusionbased object removal in dense scenes. arXiv preprint arXiv:2607.16656, 2026.

[28] Dinh-Khoi Vo, Van-Loc Nguyen, Tam V Nguyen, Minh-Triet Tran, and Trung-Nghia Le. Pandora: Pixel-wise attention dissolution and latent guidance for zero-shot object removal. arXiv preprint arXiv:2603.27555, 2026.

[29] Yixing Zhu, Qing Zhang, Yitong Wang, Yongwei Nie, and Wei-Shi Zheng. Entityerasure: Erasing entity cleanly via amodal entity segmentation and completion. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28274–28283. IEEE, 2025.

[30] Wenhao Sun, Xue-Mei Dong, Benlei Cui, and Jingqun Tang. Attentive eraser: Unleashing diffusion model’s object removal potential via self-attention redirection guidance. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 20734–20742, 2025.

[31] Yueru Jia, Aosong Cheng, Yuhui Yuan, Chuke Wang, Ji Li, Huizhu Jia, and Shanghang Zhang. Designedit: unify spatial-aware image editing via training-free inpainting with a multi-layered latent difusion framework. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 3958–3966, 2025.

[32] OpenAI. GPT-Image-2.5 Sunburst. https://developers.openai.com/api/docs/models/ gpt-image-2.5-sunburst, 2026. Accessed: 2026-09-21.

[33] Black Forest Labs. FLUX Erase. https://docs.bfl.ai/flux\_tools/flux\_erase, 2026. Accessed: 2026-09-21.

[34] Marcelo Bertalmio, Guillermo Sapiro, Vincent Caselles, and Coloma Ballester. Image inpainting. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pages 417–424, 2000.

[35] Antonio Criminisi, Patrick Pérez, and Kentaro Toyama. Region filling and object removal by exemplar-based image inpainting. IEEE Transactions on image processing, 13(9):1200–1212, 2004.

[36] Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B Goldman. Patchmatch: A randomized correspondence algorithm for structural image editing. ACM Trans. Graph., 28 (3):24, 2009.

[37] Deepak Pathak, Philipp Krahenbuhl, Jef Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2536–2544, 2016.

[38] Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa. Globally and locally consistent image completion. ACM Transactions on Graphics (ToG), 36(4):1–14, 2017.

[39] Guilin Liu, Fitsum A Reda, Kevin J Shih, Ting-Chun Wang, Andrew Tao, and Bryan Catanzaro. Image inpainting for irregular holes using partial convolutions. In European conference on computer vision, pages 89–105. Springer, 2018.

[40] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Generative image inpainting with contextual attention. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pages 5505–5514. IEEE, 2018.

[41] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Free-form image inpainting with gated convolution. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4471–4480, 2019.

[42] Kamyar Nazeri, Eric Ng, Tony Joseph, Faisal Qureshi, and Mehran Ebrahimi. Edgeconnect: Structure guided image inpainting using edge prediction. In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pages 3265–3274. IEEE, 2019.

[43] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 3172–3182. IEEE, 2022.

[44] Wenbo Li, Zhe Lin, Kun Zhou, Lu Qi, Yi Wang, and Jiaya Jia. Mat: Mask-aware transformer for large hole image inpainting. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10748–10758. IEEE, 2022.

[45] Qiaole Dong, Chenjie Cao, and Yanwei Fu. Incremental transformer structure enhanced image inpainting with masking positional encoding. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11348–11358. IEEE, 2022.

[46] Chenjie Cao, Qiaole Dong, and Yanwei Fu. Zits++: Image inpainting by improving the incremental transformer on structural priors. IEEE transactions on pattern analysis and machine intelligence, 45(10):12667–12684, 2023.

[47] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[48] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising difusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 10674–10685. ieee, 2022.

[50] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent difusion models for high-resolution image synthesis. In International Conference on Learning Representations, volume 2024, pages 1862–1874, 2024.

[51] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising difusion probabilistic models. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 11451– 11461. IEEE, 2022.

[52] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended difusion for text-driven editing of natural images. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18187–18197. IEEE, 2022.

[53] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent difusion. ACM transactions on graphics (TOG), 42(4):1–11, 2023.

[54] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image difusion models. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022.

[55] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In European Conference on Computer Vision, pages 195–211. Springer, 2024.

[56] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch difusion. In European Conference on Computer Vision, pages 150–168. Springer, 2024.

[57] Ahmet Burak Yildirim, Vedat Baday, Erkut Erdem, Aykut Erdem, and Aysegul Dundar. Instinpaint: Instructing to remove objects with difusion models. arXiv preprint arXiv:2304.03246, 2023.

[58] Yiğit Ekin, Ahmet B Yildirim, Erdem E Caglar, Aykut Erdem, Erkut Erdem, and Aysegul Dundar. Clipaway: Harmonizing focused embeddings for removing objects via difusion models. Advances in Neural Information Processing Systems, 37:17572–17601, 2024.

[59] Fan Li, Zixiao Zhang, Yi Huang, Jianzhuang Liu, Renjing Pei, Bin Shao, and Songcen Xu. Magiceraser: Erasing any objects via semantics-aware control. In European Conference on Computer Vision, pages 215–231. Springer, 2024.

[60] Yi Liu, Hao Zhou, Benlei Cui, Wenxiang Shang, and Ran Lin. Erase difusion: Empowering object removal through calibrating difusion pathways. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2418–2427. IEEE, 2025.

[61] Siyuan Yang, Lu Zhang, Liqian Ma, Yu Liu, JingJing Fu, and You He. Magicremover: Tuningfree text-guided image inpainting with difusion models. arXiv preprint arXiv:2310.02848, 2023.

[62] Ledun Zhang, Yatu Ji, Xufei Zhuang, and Xinying Yao. Clickremoval: An interactive open-source tool for object removal in difusion models. arXiv preprint arXiv:2605.14461, 2026.

[63] Sanghyun Jo, Donghwan Lee, Eunji Jung, Seong Je Oh, and Kyungsu Kim. Eraselora: Mllm-driven foreground exclusion and background subtype aggregation for dataset-free object removal. arXiv preprint arXiv:2512.21545, 2025.

[64] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in neural information processing systems, 36:31428–31449, 2023.

[65] Delong Liu, Haotian Hou, Zhaohui Hou, Shihao Han, Zhiyuan Huang, Mingjie Zhan, Fei Su, and Zhicheng Zhao. Raa: Achieving interactive remove/add anything via fully synthetic data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 7123–7131, 2026.

[66] Maksim Kuprashevich, Grigorii Alekseenko, Irina Tolstykh, Georgii Fedorov, Bulat Suleimanov, Vladimir Dokholyan, and Aleksandr Gordeev. Nohumansrequired: Autonomous high-quality image editing triplet mining. In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 6059–6068. IEEE, 2026.

[67] Zhekai Chen, Wen Wang, Zhen Yang, Zeqing Yuan, Hao Chen, and Chunhua Shen. Freecompose: Generic zero-shot image composition with difusion prior. In European Conference on Computer Vision, pages 70–87. Springer, 2024.

[68] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6593–6602. IEEE, 2024.

[69] Jinjing Zhao, Fangyun Wei, Yitong Wang, Xiuyu Wu, Yunuo Chen, Yang Yue, Sirui Zhang, Wenbo Wang, Hongyang Zhang, Dong Chen, et al. Vibeedit: Image editing with canvas instructions. arXiv preprint arXiv:2610.12229, 2026.

[70] Jinghong Lan, Wei Cheng, Yunuo Chen, Ziqi Ye, Peng Xing, Yixiao Fang, Rui Wang, Yufeng Yang, Xuanyang Zhang, Xianfang Zeng, et al. Freestyle: Free control of style-content dual-reference generation from community lora mining. arXiv preprint arXiv:2606.20506, 2026.

[71] Zhoujie Fu, Xianfang Zeng, Jinghong Lan, Xinyao Liao, Cheng Chen, Junyi Chen, Jiacheng Wei, Wei Cheng, Shiyu Liu, Yunuo Chen, et al. imontage: Unified, versatile, highly dynamic many-to-many image generation. arXiv preprint arXiv:2511.20635, 2025.

[72] Dinh-Khoi Vo, Thanh-Toan Do, Tam V Nguyen, Minh-Triet Tran, and Trung-Nghia Le. Cpam: Context-preserving adaptive manipulation for zero-shot real image editing. arXiv preprint arXiv:2506.18438, 2025.

[73] Clément Chadebec, Onur Tasar, Sanjeev Sreetharan, and Benjamin Aubin. Lbm: Latent bridge matching for fast image-to-image translation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 29086–29098. IEEE, 2025.

[74] Waikit Xiu, Qiang Lu, Junbiao Chen, and Xiying Li. Prederase: Training-free object-and-efect removal with predictive latent guidance. arXiv preprint arXiv:2609.00956, 2026.

[75] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training difusion models with reinforcement learning. In International Conference on Learning Representations, volume 2024, pages 4965–4987, 2024.

[76] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image difusion models. Advances in neural information processing systems, 36:79858–79885, 2023.

[77] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image difusion models with reward backpropagation. 2023.

[78] Kevin Clark, Paul Vicol, Kevin Swersky, and David Fleet. Directly fine-tuning difusion models on diferentiable rewards. In International Conference on Learning Representations, volume 2024, pages 4793–4822, 2024.

[79] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di ZHANG, and Wanli Ouyang. Flow-GRPO: Training flow matching models via online RL. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=oCBKGw5HNf.

[80] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612, 2004.

[81] Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar. Rethinking fid: Towards a better evaluation metric for image generation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9307–9315. IEEE, 2024.

## A Related Work

## A.1 Image Inpainting and Object Removal

Classical image inpainting propagates structure from hole boundaries or retrieves compatible patches from visible regions [34–36]. Learning-based methods introduce semantic completion, global–local adversarial supervision, partial or gated convolutions, contextual feature transfer, and explicit structural priors [37–42]. Fourier-based models and mask- or structure-aware Transformers further expand the efective context for completing large and irregular missing regions [43–46].

Difusion models provide strong generative priors for reconstructing missing content [47–50]. Inpainting methods adapt these priors through constrained resampling, spatial blending, conditional image translation, task prompting, or dedicated conditioning branches [51–56]. Object-removal methods further introduce language or semantic control [57–59], masked-region guidance [2], amodal completion [29], and calibrated difusion pathways [60]. A complementary line avoids task-specific training by redirecting attention or modifying latent trajectories [27, 28, 30, 61]. ClickRemoval provides interactive click-based removal through attention manipulation [62], while EraseLoRA combines multimodal scene interpretation with test-time optimization for dataset-free removal [63].

## A.2 Data and Models for Image Editing

Editing supervision ranges from manually annotated source–instruction–target triplets in MagicBrush [64] to fully synthetic pairs in RAA [65] and automated triplet mining in NoHumansRequired [66]. RORem introduces human-in-the-loop refinement to improve removal supervision [5]. Broader image-editing systems study image composition, object customization, and spatial control [31, 67, 68]. VibeEdit specifies edits through on-image canvas instructions [69]. Reference-guided generation includes style–content control in FreeStyle [70] and many-to-many image generation in iMontage [71]. CPAM emphasizes context preservation during real-image editing [72], while LBM formulates image translation through latent bridge matching [73].

LensErase targets a unified source–mask interface spanning natural objects, visual text, translucent watermarks, and layered designs. Its coverage is supported by large-scale, domain-specific data construction, while visual initialization and mask-aware conditioning adapt a pretrained text-to-image prior to erasing. Supervised fine-tuning and reward-based post-training further refine removal quality and content preservation.

## A.3 Object Efects and Background Preservation

Object erasing often requires removing associated efects, such as shadows and reflections, that may extend beyond the selection mask. ObjectDrop explicitly models such efects through counterfactual supervision [1]. OmniEraser mines paired video frames [4], OmniPaint couples removal with insertion [24], and GeoRemover separates geometry removal from appearance rendering [26]. ObjectClear reuses target-aware attention to localize associated efects and guide inference-time fusion [3], whereas OSOR combines alpha-region prediction with single-step, maskrobust generation [23]. PredErase provides a training-free alternative based on predictive latent guidance [74].

Spatial difusion blending and gradient-domain editing provide established mechanisms for combining edited content with preserved surroundings [14, 52]. LensErase instead retains the intact source image as a visual condition, preserving information about the selected target, its associated efects, and content visible through translucent overlays. At inference, TraceFusion estimates the efective edit support from the selection mask and source–output residuals, then combines generated content with the original image outside this support.

## A.4 Reward-Based Post-Training

Reward optimization adapts generative models to criteria that paired reconstruction objectives may not fully capture. DDPO and DPOK formulate denoising as a sequential decision process and optimize terminal rewards with policy gradients, with DPOK additionally regularizing updates toward a pretrained reference [75, 76]. For diferentiable rewards, AlignProp and DRaFT backpropagate through the sampling trajectory [77, 78]. Flow-GRPO extends online policy optimization to flow-matching models through stochastic sampling [79]. DifusionNFT [7] instead learns from scored final rollouts using a forward-process flow-matching objective, with relative rewards defining positive and negative implicit objectives without backpropagating through the sampling trajectory.

LensErase adopts DifusionNFT with removal-specific rewards that combine semantic assessment of the intended edit with deterministic fidelity measures. The reward separately evaluates removal, region reconstruction, semantic preservation, visual cleanliness, and fidelity outside the edit region, encouraging successful erasure while protecting unrelated content.

## B Data Construction and Filtering Details

All supervised data pipelines produce the aligned triplet $( I _ { \mathrm { i n p u t } } , I _ { \mathrm { o u t p u t } } , M )$ . Language is used for data construction and screening, but is never provided as an input to the student model.

Figure 4 summarizes the composition of the 101.07M pretraining examples, 140K SFT examples, and 19K RL tasks. Pretraining and SFT use aligned input–target–mask triplets, whereas RL uses curated image–mask tasks with domain-specific references for reward computation. The following sections detail the construction and filtering procedures for pretraining, SFT, and RL data.

## B.1 Pretraining Data Pipelines

## B.1.1 General Data: addition-to-removal counterfactuals

Directly asking a generative teacher to remove an object makes the clean target a hallucinated inpainting. We instead begin with a real image $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ and ask an editing teacher $G _ { \mathrm { a d d } }$ to add an object that was absent from the original scene:

$$
I _ { \mathrm { i n p u t } } = G _ { \mathrm { a d d } } ( I _ { \mathrm { o u t p u t } } , P ) , \qquad M = \mathrm { R e c o v e r M a s k } ( I _ { \mathrm { i n p u t } } , I _ { \mathrm { o u t p u t } } ) .\tag{8}
$$

The instruction P is used only by the teacher. Reversing the pair therefore provides removal supervision whose target is the untouched real image instead of a generated background.

Scene-aware proposals. Candidate additions are organized into 15 categories: human, animal, apparel and accessories, vehicles, furniture, household items, electronic devices, food and beverages, containers and tableware, sports and leisure, plants, buildings, infrastructure, tools and hardware, and natural elements. Each category contains a fine-grained object catalog. For every image, we sample three distinct coarse categories and provide them, the image, and its caption to GPT-4.1 [8]. The model proposes one scene-compatible object absent from the image and belonging to one of the sampled categories, returning its coarse category, object class, and insertion instruction. Restricting the proposal to one object reduces ambiguous additions, while mask recovery can still handle occasional multi-instance outputs.

Sampling is dynamically adjusted using accumulated category and object statistics. Overrepresented categories or object classes are excluded from subsequent sampling, while proposals remain constrained by scene compatibility.

<table><tr><td>Pretraining</td><td>101.07M</td><td>SFT</td><td>140K</td><td>RL</td><td>19K</td></tr><tr><td>General Data</td><td>77.60M</td><td>General Data</td><td>80K</td><td>General Data</td><td>13K</td></tr><tr><td></td><td></td><td>Watermark Data</td><td></td><td>Watermark Data</td><td></td></tr><tr><td>Watermark Data</td><td>9.99M</td><td></td><td>20K</td><td></td><td>2K</td></tr><tr><td>Visual Text Data</td><td>1.44M</td><td>Visual Text Data</td><td>20K</td><td>Visual Text Data</td><td>2K</td></tr><tr><td>Poster Data</td><td>8.02M</td><td>Poster Data</td><td>20K</td><td>Poster Data</td><td>2K</td></tr><tr><td>Large-Mask Data</td><td>4.02M</td><td></td><td></td><td></td><td></td></tr></table>

Figure 4 Data composition across training stages. Dataset sizes and category counts for pretraining, SFT, and RL. Bands connect shared categories across stages.

Addition teacher. We instantiate $G _ { \mathrm { a d d } }$ with an of-the-shelf image editing model. It receives the real image and a short $^ { 6 6 } { \mathrm { A d d ~ . ~ . ~ } } ^ { 3 }$ instruction and produces only $I _ { \mathrm { i n p u t } }$ , while the original is preserved exactly as $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ . We retain only spatially aligned pairs. Because the teacher does not provide an accurate object mask, we recover it from the aligned before/after pair.

DCT–DINOv3–SAM2 mask recovery. A raw RGB diference is sensitive to texture noise and incidental appearance changes. We instead form two complementary maps. First, we retain only low-frequency DCT coeficients in each color channel, reconstruct both images, and compute their absolute diference $D _ { \mathrm { D C T } }$ . This suppresses high-frequency generation noise while retaining object-scale pixel changes. Second, we compare aligned DINOv3 [9] dense features using cosine distance,

$$
D _ { \mathrm { { D I N O } } } ( i , j ) = 1 - \frac { f _ { \mathrm { { i n p u t } } } ( i , j ) ^ { \top } f _ { \mathrm { { o u t p u t } } } ( i , j ) } { \Vert f _ { \mathrm { { i n p u t } } } ( i , j ) \Vert _ { 2 } \Vert f _ { \mathrm { { o u t p u t } } } ( i , j ) \Vert _ { 2 } } .\tag{9}
$$

Here $f _ { \mathrm { i n p u t } }$ and $f _ { \mathrm { o u t p u t } }$ denote the dense features of $I _ { \mathrm { i n p u t } }$ and $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ , respectively. Thresholding and morphological opening/closing give binary maps $B _ { \mathrm { D C T } }$ and $B _ { \mathrm { D I N O } }$ . When both are nonempty, we use

$$
B _ { \mathrm { c o a r s e } } = B _ { \mathrm { D C T } } \cup B _ { \mathrm { D I N O } } , \qquad B _ { \mathrm { p o s } } = B _ { \mathrm { D C T } } \cap B _ { \mathrm { D I N O } } .\tag{10}
$$

The union maximizes recall for the possible edit extent, while the intersection provides conservative positive-point candidates. Cross-signal IoU is retained as a filtering indicator.

SAM2-based mask refinement. To recover object boundaries from the coarse change maps, we apply SAM2 [10] to the object-added image $I _ { \mathrm { i n p u t } }$ using sparse point prompts. Positive candidates are preferentially selected from the interior of $B _ { \mathrm { p o s } }$ using a distance transform, while negative candidates are sought outside $B _ { \mathrm { c o a r s e } }$ , preferably in a nearby exterior band. Farthest-point sampling selects up to eight positive and four negative points per image to obtain spatially distributed prompts.

The sampled positive points are then grouped by the 8-connected components of $B _ { \mathrm { p o s } }$ . Each component receiving a positive point is processed separately with its own positive prompts and the shared negative prompts. We retain the highest-scoring SAM2 prediction from each group and take their union as the final mask M. The coarse change maps guide prompt placement but are not directly merged into the final mask.

Object-focused masks and associated efects. Our mask recovery combines low-frequency DCT diferences for photometric change detection, DINOv3 feature diferences for semantic localization, and SAM2 for object-boundary refinement. The DCT branch can capture spatially extended appearance changes, such as cast shadows, while DINOv3 provides complementary semantic cues for localizing the added object. An associated efect may therefore enter the high-recall region $B _ { \mathrm { c o a r s e } }$ through the DCT branch even when the shared positive region $B _ { \mathrm { p o s } }$ remains concentrated on the object body. SAM2 can then produce an object-focused selection mask without requiring it to cover the entire detected change region.

Because $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ is the original image before insertion, the full input–target pair still contains the corresponding associated-efect changes, including those outside M. Such pairs provide supervision for removing the selected object together with its associated efects, without requiring every efect pixel to be explicitly included in the selection mask.

General Data pair filtering. We screen General Data for unreliable localization, inefective additions, and unintended scene changes. The remaining pairs are assessed using the following measures:

• Cross-signal agreement. The intersection-over-union between $B _ { \mathrm { D C T } }$ and $B _ { \mathrm { D I N O } }$ measures agreement between photometric and semantic change localization. Pairs flagged for low agreement are rejected to reduce supervision based on inconsistent estimates of the edited region.

• Mask coverage. The recorded mask-to-image area ratio identifies excessively broad selections. We reject masks with excessive coverage to limit cases in which the recovered selection extends over a large fraction of the scene.

• Within-mask change. We compute input–target PSNR within M and reject pairs with unusually high similarity. Such cases exhibit insuficient change in the selected region, as can occur when an insertion fails or the mask includes substantial unchanged background.

• Outside-mask consistency. Input–target PSNR over the complement of M measures aggregate changes outside the selection. Low values trigger rejection, targeting outpainting-induced shifts, global appearance drift, and broad background regeneration. This check bounds aggregate discrepancy without requiring every exterior pixel to remain identical.

• Feature-level neighborhood change. We compare SAM2 features in an exterior band separated from M by a guard margin to reduce contamination from receptive fields overlapping the inserted object. One minus the mean cosine similarity of the least-similar patches provides a score for suspected replacement or damage to neighboring content.

• Pixel-level neighborhood change. A complementary score combines the fraction of changed pixels in an adjacent exterior ring with the strength of the largest RGB diferences. It measures the spatial extent and intensity of nearby appearance changes, providing pixel-level evidence for the replacement check.

The feature- and pixel-level neighborhood scores form a joint replacement criterion: this criterion triggers only when both scores indicate substantial change. The other rejection criteria are checked separately.

## B.1.2 Alpha-supervised watermark synthesis

Watermark-removal pairs are generated by compositing rendered overlays onto clean images, providing known watermark support without image-diference-based mask recovery.

Assets and typography. We combine a pool of approximately 2K logo assets, in raster or SVG format, with template-based multi-lingual watermark text. The text pool includes copyright and ownership notices, preview and sample labels, usage restrictions, photographic attributions, website addresses, and identifiers. Templates are instantiated with sampled company names, domains, years, and numeric IDs. Font faces and sizes are randomized.

Layout and appearance. The renderer samples six synthesis modes: individual logos, standalone text, tiled logos or text, large diagonal text, stripe patterns with optional embedded text, and corner logos. Mode-specific randomization controls position, scale, orientation, opacity, color, and repetition spacing. Additional appearance variations include text outlines or ofset shadows, slight logo blur, and translucent backdrops for corner marks. These combinations produce both localized overlays and repeated patterns extending across the image.

Triplet construction. Let $( W , \alpha )$ denote the rendered watermark layer in image coordinates, with $\alpha \in [ 0 , 1 ]$ incorporating the sampled opacity and any rendered decorations. Alpha compositing gives

$$
{ \cal I } _ { \mathrm { i n p u t } } = \alpha { \cal W } + ( 1 - \alpha ) { \cal I } _ { \mathrm { o u t p u t } } , \qquad { \cal M } = { \bf 1 } [ \alpha > 0 ] .\tag{11}
$$

The clean image is retained as $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ , while M is derived directly from the rendered alpha support, including disconnected glyphs, repeated elements, semi-transparent marks, and any rendered outlines, shadows, or backdrops.

## B.1.3 Generated large-foreground compositing

Large removal regions are relatively under-represented in General Data. We complement this branch by generating isolated foregrounds and compositing them onto real images. Separating foreground generation from placement allows us to vary object appearance, location, and visible area while retaining the pre-compositing image as the clean removal target.

Foreground diversity. We sample from a catalog of 25 coarse categories and approximately 2.5K object descriptions, spanning people, animals, vehicles, furniture, plants, and everyday objects. Generation prompts combine the sampled object description with viewpoint or pose descriptors, including frontal, side, three-quarter, and top-down views, using varied studio-photography templates that encourage isolated objects on white backgrounds. BiRefNet [11] predicts the foreground alpha matte. We discard degenerate extractions with negligible or nearly frame-filling foreground support, then apply random in-plane rotation and crop to the foreground bounding box. Viewpoint variation and in-plane transformations diversify both object appearance and the geometry of the resulting masks.

Background-aware placement. To vary image position while limiting interference with existing content, we apply YOLOE [12] to estimate foreground occupancy in each background image. Candidate anchors are sampled across the central image region, with preference given to locations whose neighborhoods contain less occupied area. Placement aligns the foreground’s alpha-weighted centroid with the selected anchor, positioning irregular objects according to their foreground mass rather than the center of their bounding box.

Visible-area and occlusion control. We sample a desired mask-area ratio uniformly from 20%–60% and adjust foreground scale to approach this ratio. Coverage is measured from the foreground actually visible after placement and clipping to the image. We constrain occlusion at two levels: the covered fraction of each suficiently large detected background instance, and the covered fraction of the union of all detected instances. Placements causing excessive occlusion are retried or discarded. These constraints allow partial overlap while limiting severe loss of visible context. The realized mask area is checked again after the final compositing steps, and samples outside the designated large-mask range are discarded.

Appearance adjustment and triplet construction. We partially adjust foreground color statistics toward those of the local background and feather the alpha boundary to reduce appearance and boundary discontinuities. Let F and α denote the adjusted foreground and its alpha matte in final image coordinates, with $\alpha \in [ 0 , 1 ]$ . We construct

$$
I _ { \mathrm { i n p u t } } = \mathrm { C o m p o s i t e } ( I _ { \mathrm { o u t p u t } } , F , \alpha ) , \qquad M = { \bf 1 } [ \alpha > 0 . 5 ] .\tag{12}
$$

The pre-compositing image remains $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ , so the background behind the inserted foreground is known rather than hallucinated by a removal teacher. The selection mask follows the transformed foreground silhouette, providing object-shaped, large-area removal supervision with a fully specified target.

## B.1.4 OCR-verified partial visual text removal

A full-image text-removal teacher may leave residual lettering or alter unrelated scene content. Rather than adopting its output directly as the training target, we use it to propose local reconstructions and construct selection-specific removal pairs through OCR-guided region selection and local blending.

Text localization and teacher editing. Given the original text-containing image $I _ { \mathrm { i n p u t } }$ , we apply PaddleOCR [13] with the PP-OCRv6 medium detector and recognizer to obtain text polygons. An image editing teacher receives the source image and a global instruction to erase text while preserving objects, signs, boards, and surfaces, producing a teacher-edited image <sup>˜</sup>I. We then run PaddleOCR on <sup>˜</sup>I to obtain post-edit text detections. We use a higher OCR-confidence threshold on the source image to prioritize reliable text localization and a lower threshold after editing to increase sensitivity to weak residual text. The two OCR passes provide the source locations and residual-text evidence used by the eligibility assessment below.

Grouped selection and mask construction. We group overlapping or nearby text polygons in the source image into joint selection units. A nonempty subset of eligible groups is randomly selected for removal, with eligibility determined by the OCR-based criterion described below. Group-level selection couples the treatment of neighboring or overlapping text regions, while subset sampling varies the removal scope without requiring a separately prompted teacher edit. The union of the selected text polygons, with a small dilation, defines the binary condition mask M. This retains the selected regions’ spatial geometry without replacing each group with its enclosing rectangle.

Region-restricted target construction. Starting from $I _ { \mathrm { i n p u t } }$ , we blend the selected groups’ teacher edited regions back into the source through arbitrary-mask Poisson blending [14]. Blending is performed group by group using the corresponding dilated polygon masks to reduce color and exposure seams from direct patch replacement. The resulting composite is $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ : the source image supplies unselected text and surrounding content, while the teacher supplies candidate reconstructions for the selected removal regions. This local reuse of teacher edits constructs partial-removal targets without adopting the teacher’s full-image appearance changes. Removal efectiveness and blending quality are assessed using the criteria below.

Visual Text Data pair filtering. We assess text-removal efectiveness separately from artifacts introduced by local paste-back. Failed construction records and missing, undecodable, or empty masks are excluded. Let <sup>˜</sup>I denote the teacher-edited image before paste-back. The filtering criteria are:

• OCR-based removal eligibility. Original text detections are matched to post-edit detections by bounding-box overlap. A group is selectable only when none of its members has a post-edit match. Candidates without a selectable group are discarded. This prevents regions with detected residual text from entering the partial-removal target.

• Structural change in selected text. We compute RGB SSIM [80] between $I _ { \mathrm { i n p u t } }$ and <sup>˜</sup>I within each selected group’s dilated polygon support. High similarity indicates insuficient structural change despite the absence of an OCR match. We reject candidates with a high maximum SSIM across groups, so inefective removal in one group cannot be averaged away by successful edits elsewhere.

• Boundary gradient discrepancy. From the paste-back residual $R = I _ { \mathrm { o u t p u t } } - \tilde { I }$ , we compute per-channel Sobel gradient magnitudes and take their maximum across RGB channels at each pixel. For each selected group, we average this magnitude in an inner boundary band. A high maximum group mean triggers rejection to screen for boundary leakage introduced by blending.

• Regional gradient discrepancy. Using the same residual-gradient map, we compute the 90th-percentile magnitude over each group’s full selected region. A high maximum across groups triggers rejection to screen for strong internal discrepancies relative to the teacher reconstruction, complementing the boundary-focused check.

The two gradient criteria are applied independently. They measure discrepancies introduced by paste-back rather than penalizing image gradients that are also present in the teacher reconstruction.

## B.1.5 Layer-aware poster rendering

We construct poster-removal pairs from explicit color–alpha layers with known z-order. The same rendered design defines both the complete poster and a selected-layer removal target, allowing visual diversity without independently regenerating the two images.

Canvas and layout sampling. Background images are fitted to canvases with varied resolutions and square, portrait, landscape, and widescreen aspect ratios. Layout templates guide initial element placement through header–body–footer, centered, grid, sidebar, full-bleed, and diagonal arrangements, alongside unrestricted layouts. Sample plans balance canvas types, layout templates, element types, and element and removal counts, with an initial budget of 4–9 elements. Element sizes are also varied to encourage coverage of diferent removal extents.

Element and typographic diversity. Elements span text, geometric shapes, decorative graphics, synthetic logos and icons, and presentation-style components such as bullet lists, tables, and chart placeholders. Text pools distinguish titles, subtitles, paragraphs, labels, prices, dates, URLs, and mixed-script content, with multi-lingual examples. Fonts are sampled according to typographic role and, for CJK strings, character coverage. Font size and line wrapping are adapted to the allocated region. We further vary color, opacity, alignment, filled versus outlined shapes, and optional drop shadows or outer glows.

Layer-consistent pair construction. Each element is rendered independently and placed in z-order. Layout constraints prevent overlap between text-element bodies, limit excessive occlusion and overlap depth, and check text contrast against the content beneath it. Retained text receives additional protection against severe occlusion, including checks at the character-region level. The planned removal set is refined by excluding candidates overlapped by higher layers designated for retention. We then reuse the identical background and rendered layers to produce $I _ { \mathrm { i n p u t } }$ with all accepted elements and $\boldsymbol { I _ { \mathrm { o u t p u t } } }$ with the selected layers omitted. The content, appearance, position, and order of the remaining layers are held fixed.

Body-focused mask construction. For mask construction, we use the element-body layers before shadow or glow is added. A front-to-back traversal assigns each pixel to the topmost body layer with nonzero alpha. Pixels assigned to selected layers are unioned and slightly dilated to obtain M. This defines a body-focused selection instead of a bounding-box mask. The paired images nevertheless remove the full selected layers, including their rendered efects, so supervision for efect removal can extend beyond M without requiring an exhaustive efect mask.

## B.2 SFT Data Curation and Quality Screening

We curate SFT examples from General, Watermark, Visual Text, and Poster Data, emphasizing high-resolution candidates. Construction follows the domain-specific procedures summarized in Section 2.2, including professionally designed posters rather than the procedural layouts used for pretraining. GPT-5.6 Sol [15] examines each aligned triplet $( I _ { \mathrm { i n p u t } } , I _ { \mathrm { o u t p u t } } , M )$ using the source image, a mask-overlay visualization, and the target image. When the selected region is dificult to inspect at full-image scale, aligned mask-centered crops of the source and target supplement the full-image views to reveal fine residuals and reconstruction artifacts.

• Mask validity. The selection must identify one or more intended targets without substantial leakage into unrelated regions. Coarse boundaries and limited inclusion of nearby background or target-associated efects are permitted when the erase request remains clear.

• Edit correctness and preservation. The source–target pair must realize the intended removal completely while preserving unrelated content. Target remnants, unintended deletions, and spatial misalignment lead to rejection.

• Reconstruction quality. The revealed region must have plausible geometry and texture without conspicuous editing artifacts. We reject seams, smearing, ghosting, abnormal blur, and implausible local reconstruction.

For General Data only, we additionally require natural photographs or credible photorealistic renderings, excluding illustrations, documents, and poster-like graphics. A triplet is accepted only when its mask is valid, all pair-quality criteria pass, and the overall judgment has medium or high confidence without contradictory failure evidence. Preservation does not require pixel-wise equality outside the intended edit. Screening focuses on correctly realizing the erase request without imposing additional scene-level constraints unrelated to it.

## B.3 RL Data Pipeline

Task definition and composition. The dataset contains 19K tasks: 13K General Data examples and 2K examples each from Poster Data, Watermark Data, and Visual Text Data. General Data comprises 8K dificulty-filtered tasks and 5K supplementary tasks. Each task conditions LensErase only on the source image $( I _ { \mathrm { i n p u t } } )$ and selection mask (M). General Data requires no paired target, whereas Poster Data and Watermark Data retain deterministic clean references and Visual Text Data retains verified weak references. Available reference images and structured annotations support data curation and reward evaluation but are not additional inputs to the erase model. During $\mathrm { R L , }$ domains are sampled at a ratio of General:Poster:Watermark:Visual $\mathrm { T e x t } = 6 { : } 2 { : } 1 { : } 1$ , independently of their dataset cardinalities; the corresponding reward ablation protocol is reported in Section G.

General Data: failure-mode-guided collection and efect mining. The 5K supplementary tasks augment the dificulty-filtered subset through failure-mode-guided collection and targeted associatedefect mining. During extensive internal evaluation of LensErase-SFT, we manually analyze recurring failures and identify challenging scenarios. We then collect images presenting similar erasure challenges.

For associated-efect mining, GPT-5.6 Sol [15] examines candidate images and selects those containing visibly attributable object-associated efects. These efects are already present in the images and are not synthetically added. They include cast shadows, reflections, water ripples and wakes, splashes, smoke or steam, emitted light, refraction or caustics, contact traces, and motion trails when their causal relation to a selected object is visually supported.

General Data: object localization and mask construction. For newly annotated tasks, GPT-5.6 Sol identifies the target objects and returns their physical-object bounding boxes. For efect-focused tasks, these boxes localize the objects responsible for the efects rather than the efects themselves. Source images and bounding-box overlays are used to verify object localization and, where present, the visibility and causal attribution of associated efects. We then provide the image and verified object boxes to SAM 2.1 [10] to segment the object bodies, taking the union of masks when multiple objects are selected. Further VLM checks assess object-body completeness, instance identity, the natural-image domain, and leakage into non-target content.

The mask is an object-selection cue, not an exhaustive annotation of all pixels that should change. We do not automatically dilate or close it to absorb shadows or reflections. Accurately and compactly captured efects may be included, but do not justify covering large unrelated regions. Only after mask construction and validation do we analyze the finalized source–mask task and assess its dificulty using the procedure below. Thus, dificulty is evaluated for the actual selection, not for the source image or proposed bounding boxes alone.

General Data: quality filtering and dificulty screening. We separate source–mask validity from task dificulty. After mask construction, we exclude invalid conditions, including unidentifiable targets and degenerate masks, and suppress exact and near-duplicate source–mask pairs. The strict pool additionally excludes ambiguous or highly irregular selections, including masks with substantial overlap with independent non-target objects. Thus, strict filtering establishes usable erase conditions, while task dificulty is assessed separately.

We then apply a two-stage ofline analysis using GPT-5.6 Sol [15]. First, the VLM establishes a structured understanding of the selected targets, their constituent parts, associated efects and efect bounding boxes, and nearby independent content to preserve. Second, it estimates the intrinsic dificulty of the same source–mask task without observing an erase-model output. Five integer scores $s _ { k } \in \{ 0 , \ldots , 4 \}$ assess, in order, removal scope and associated efects; reconstruction and fine-detail recovery; occlusion and object interactions; preservation and locality risk; and the risk of introducing unwanted entities. With respective weights $( w _ { 1 } , \dots , w _ { 5 } ) = ( 0 . 2 5 , 0 . 3 0 , 0 . 2 0 , 0 . 2 0 , 0 . 0 5 )$ overall dificulty is

$$
D = 1 0 \sum _ { k = 1 } ^ { 5 } w _ { k } { \frac { s _ { k } } { 4 } } .\tag{13}
$$

Tasks are categorized as Easy $( D \le 2 )$ , Moderate $( 2 < D \leq 4 )$ , Challenging $( 4 < D \le 6 )$ , Hard $( 6 < D \le 8 )$ , or Extreme $( 8 < D \le 1 0 )$ . High ratings must reflect multiple substantive challenges, not mask area alone. From the strict pool, we retain 6,991 Hard and 1,009 Extreme tasks, yielding the 8K dificulty-filtered subset. This score is used for task selection, not as the training reward.

Efects outside the selection remain explicitly described in the source understanding for reward evaluation. Final masks are stored as single-channel binary PNGs with values 0 and 255 at the source-image resolution, and the associated understanding is bound to hashes of the finalized source and mask.

Poster Data: high-resolution layered removal. The RL Poster Data subset contains 2,000 unique high-quality poster examples designed by professional designers. We rank candidate designs using their native canvas area, side lengths, and layer count, and construct the subset from an ordered pool of 2,500 candidates, replacing ineligible designs until 2,000 valid tasks are obtained. The source canvases are not downscaled during pair construction. Each design contributes one task that removes a single visible non-background layer, which may contain text, graphics, icons, color blocks, or an image. The selected layer may lie behind other foreground layers. Rendering the same layer stack with and without the selected layer gives the source image and an exact removal reference $\left( I _ { \mathrm { r e f } } \right)$ , following the layer-aware construction principle described in Appendix B.1.5.

We derive the mask from the final rendered RGB diference,

$$
M ( x ) = \mathbf { 1 } [ \exists c \in \{ R , G , B \} : I _ { \mathrm { i n p u t } } ( x , c ) \neq I _ { \mathrm { r e f } } ( x , c ) ] ,\tag{14}
$$

rather than directly using the selected layer’s alpha support. This excludes occluded or otherwise unchanged pixels from the selection. Unselected foreground content that remains visible over the selected layer’s spatial extent is part of the correct reference, not evidence of incomplete removal. Candidate selection encourages diversity in removal area, element type, and layer depth, while prioritizing high resolution without enforcing equal counts across these attributes.

Watermark Data: controlled synthesis and mask validation. For the RL watermark subset, we start with a curated pool of 25,000 high-resolution natural images. We first retain images whose original width and height are both at least 1,440 pixels, yielding 24,688 eligible backgrounds. From this eligible pool, we select 2,000 distinct images and use each as the clean background for one watermark-removal task, producing 2,000 tasks in total. Before watermark synthesis, we resize each selected image as needed to limit its long side to 2,048 pixels while preserving its aspect ratio. The upstream asset library combines logo resources with procedurally generated stamps, signatures, patterns, machine-readable codes, and multilingual text. Asset provenance describes the candidate library; the assets actually used by accepted samples are recorded in the sample manifest. Following the synthesis formulation in Appendix B.1.2, we composite sampled assets onto the clean canvas while varying layout, color, opacity, scale, and rotation, retaining the pre-compositing image as the exact clean reference.

Precise change regions are computed from the final source–reference RGB diferences. Half of the tasks use precise masks, while the other half use user-style coarse local selections, such as rectangular or rounded regions, that cover all actual changes without substituting an indiscriminate whole-image box. We validate image–reference–mask dimensions, mask binarity, and change coverage. The final subset enforces marginal quotas over six dimensions: ten asset families with 200 tasks each; four layouts (corner, central, diagonal band, and tiled) with 500 each; synthesized watermark-area bins of 0.1–1%, 1–5%, 5–15%, and 15–45% with 500 each; four opacity bins with 500 each; five requested color families with 400 each; and precise versus coarse masks with 1,000 each. These are marginal quotas, not a requirement to balance every joint attribute combination.

Visual Text Data: curated weak references. The 2,000 Visual Text tasks use the OCR-verified partial-removal procedure in Appendix B.1.4. Pre- and post-edit OCR identify candidate removals, and selected teacher-edited regions are blended back into the source. RL curation further applies VLM screening and text-centered cropping. When selected text occupies only a small part of the image, we crop around the target text, applying the same crop to the source image, selection mask, and weak reference. This increases the selected text’s relative extent in the resulting task while maintaining alignment. Selection also considers efective image resolution and mask coverage. The target remains a verified weak reference, since the hidden text-free appearance is not observed directly. Reference-guided rewards therefore assess removal and reconstruction without requiring every plausible completion to match the teacher reconstruction exactly, as detailed in Appendix E.2.

Interface to reward evaluation. Ofline source understanding supports task curation and validation; reward evaluation follows the candidate-independent contract construction in Section 3.2 and Appendix E.2. For General Data, the erase contract and aligned detail-view geometry are defined after the training geometry transform and fixed across candidates for the same condition. Poster Data, Watermark Data, and Visual Text Data use their domain-specific references as described in that appendix. These reward-side specifications do not change the model’s sourceimage–mask conditioning interface. Training resolution buckets and RL optimization are described in Appendices E.1 and E.3, respectively.

## C LensErase-Bench: Construction and Evaluation

LensErase-Bench contains 500 cases spanning four erase domains: 200 General cases and 100 each for Poster, Watermark, and Visual Text. The benchmark is curated to cover diverse mask scales, target counts, image resolutions, and removal dificulty. Evaluation follows the reference-free protocol in Appendix C.2.

## C.1 Selection and Coverage

Mask-area coverage. We stratify each domain by mask area to ensure that the benchmark covers both highly localized edits and large-region removal. Table 5 reports the resulting domain-specific distribution. General images are evenly distributed across five mask-area ranges, while Poster, Watermark, and Visual Text use bins adapted to their characteristic edit scales.

Table 5 Domain-specific mask-area distribution in LensErase-Bench.
<table><tr><td>Domain</td><td>Mask-area bins (%)</td></tr><tr><td>General</td><td>&lt; 1, [1, 5), [5, 15), [15, 30), ≥ 30</td></tr><tr><td>Poster</td><td>&lt; 5, [5, 15), ≥ 15</td></tr><tr><td>Watermark</td><td> &lt; 1, [1, 5), [5, 15), ≥ 15</td></tr><tr><td>Visual Text</td><td>&lt; 0.5, [0.5, 1), [1, 2), ≥ 2</td></tr></table>

Attribute coverage. Beyond mask area, we curate complementary attributes that characterize practical erase requests. The General subset is evenly split between cases with and without associated visual efects, and between single-target and multi-target removal. Poster cases are approximately balanced across graphic, image, and text layers, as well as back, middle, and front layer depths. Watermark cases evenly cover ten asset families, four spatial layouts, four opacity ranges, and precise versus user-style coarse masks. Visual Text includes diverse languages, layouts, and background types.

Dificulty and target count. GPT-5.6 Sol [15] rates the filling dificulty of each case on a 0–4 scale. Table 6 reports the distribution of filling dificulty together with the number of selected semantic targets. Across the full benchmark, 59.2% of cases contain a single selected target and 40.8% contain multiple targets, with Watermark providing particularly dense multi-target cases.

Table 6 Distribution of filling dificulty and number of selected targets in LensErase-Bench.
<table><tr><td></td><td colspan="4">Filling difficulty</td><td colspan="3">Selected targets</td></tr><tr><td>Domain</td><td>0-1</td><td>2</td><td>3</td><td>4</td><td>1</td><td>2-3</td><td>≥4</td></tr><tr><td>General</td><td>9</td><td>50</td><td>87</td><td>54</td><td>100</td><td>87</td><td>13</td></tr><tr><td>Poster</td><td>34</td><td>25</td><td>25</td><td>16</td><td>93</td><td>3</td><td>4</td></tr><tr><td>Watermark</td><td>0</td><td>9</td><td>64</td><td>27</td><td>40</td><td>16</td><td>44</td></tr><tr><td>Visual Text</td><td>25</td><td>45</td><td>30</td><td>0</td><td>63</td><td>25</td><td>12</td></tr></table>

Resolution coverage. All cases preserve their original aspect ratio and are normalized without cropping or padding. The benchmark spans four long-side resolution ranges, $\leq 1 0 2 4$ , 1025–1280, 1281–1536, and 1537–2047 pixels. Each range contains 125 cases: 50 General cases and 25 from each of Poster, Watermark, and Visual Text. This produces balanced resolution coverage across domains while retaining the original selection geometry.

## C.2 Evaluation Protocol

LensErase-Bench uses the four evaluation dimensions defined in Section 2.3: Removal, Inpainting, Preservation, and Cleanliness. GPT-5.6 Sol evaluates each candidate using a shared rubric across all four domains, with domain-specific interpretation only where the semantics of the selected target difer.

Evaluation inputs. For each case, the judge receives four aligned full-image views: the source image, the source image with the selection overlaid, the candidate output, and the binary selection mask. No clean reference is provided.

Scoring rubric. Each case receives an integer score from 0 to 4 for each dimension, where 4 denotes full satisfaction of the criterion, 3 a minor defect, 2 a clear localized failure, 1 a major failure, and 0 a fundamentally unsuccessful result. Removal measures whether all selected targets and associated efects disappear; Inpainting measures the plausibility and local consistency of newly revealed content; Preservation measures fidelity of non-target content; and Cleanliness measures artifacts and unrelated additions. Multi-target cases are judged jointly, so a missed selected instance remains a Removal failure even if the remaining targets are successfully erased. To avoid double counting, a single visual defect is attributed to the most directly afected dimension.

Scores and aggregation. Let $r _ { i } ^ { \mathrm { r e m } } , r _ { i } ^ { \mathrm { i n p } } , r _ { i } ^ { \mathrm { p r e } } , r _ { i } ^ { \mathrm { c l n } } \in \{ 0 , 1 , 2 , 3 , 4 \}$ denote the Removal, Inpainting, Preservation, and Cleanliness ratings for case i. Each dimension is mapped to 0–100 by multiplying its rating by 25. The case-level Overall Score is

$$
o _ { i } = 2 5 \left[ 0 . 4 r _ { i } ^ { \mathrm { r e m } } + 0 . 2 \left( r _ { i } ^ { \mathrm { i n p } } + r _ { i } ^ { \mathrm { p r e } } + r _ { i } ^ { \mathrm { c l n } } \right) \right] .\tag{15}
$$

For each method, we report the mean of each dimension and Overall across its N valid judged cases, with equal weight assigned to every case. Here, N = 500 for all methods except PANDORA, for which N = 499 because one output was consistently rejected by the judge’s safety filter; see Table 16.

HiFi Removal. We additionally report the percentage of cases that achieve complete removal while maintaining high editing quality:

$$
\mathrm { H i F i R e m o v a l } = \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } \mathbb { 1 } \left[ r _ { i } ^ { \mathrm { r e m } } = 4 ~ \land ~ \operatorname* { m i n } \{ r _ { i } ^ { \mathrm { i n p } } , r _ { i } ^ { \mathrm { p r e } } , r _ { i } ^ { \mathrm { c l n } } \} \ge 3 \right] ,\tag{16}
$$

where 1 is the indicator function. Overall summarizes average quality across the four dimensions, whereas HiFi Removal measures the fraction of cases that satisfy all removal and editing-quality requirements simultaneously.

## D Pretraining Design and Ablations

This section provides additional implementation details and ablations for the initialization, mask conditioning, and two-stage adaptation strategy introduced in Section 3.1.

## D.1 Visual Initialization and Mask Conditioning

Both branches of LensErase process visual inputs. We retain the pretrained Lens image branch and initialize the condition branch from the same image-branch weights, while discarding the pretrained text-specific weights. The source image remains intact, preserving visual evidence of the selected target and its associated efects. Mask-Aware RoPE injects the selection mask by assigning diferent modality coordinates to masked and unmasked source tokens, as defined in Equation 1.

Architecture and positional encoding. The image and condition streams operate on spatial token grids with efective stride 16 relative to the input image. The backbone contains 48 MMDiT blocks, with 24 attention heads per block and 64 channels per head. The three-dimensional RoPE allocates these channels across the modality, height, and width axes as (8, 28, 28). Both streams use the centered spatial grid of the pretrained positional encoder. The binary selection mask is adaptively max-pooled to the source-token grid and thresholded at 0.5. The resulting value selects between the two condition-stream modality indices in Equation 1. The mask is therefore encoded only through RoPE and is not embedded into the latent channels. RoPE is applied to queries and keys in every block, while values are unchanged. The condition stream is initialized from the image stream’s input projection and corresponding attention, feed-forward, normalization, and modulation parameters.

Alternative mask representations. For the two feature-based alternatives in Table 4, we form mask features with $\mathrm { P i x e l U n s h u f f e } _ { 1 6 } ( M )$ , yielding a 256-dimensional vector aligned with each source-image token. Channel Concat concatenates this vector with the corresponding source feature and fuses them with a two-layer MLP before the condition projection. Token Concat places source and noisy-image tokens together in the image branch and projects the mask features into the condition branch. Both alternatives use positional indices independent of mask values.

Text-branch initialization control. This control retains the pretrained text-Transformer weights for the condition branch while keeping the same visual inputs, branch routing, and Mask-Aware RoPE as LensErase. Its visual input projection is still initialized from the Lens image branch, and no text prompt is used. This isolates the efect of the pretrained prior used to initialize the condition Transformer.

## D.2 Region-Weighted Objective Details

Training follows the flow-matching path and region-weighted objective in Section 3.1. In implementation, the per-token squared velocity error is averaged over channels. Equation 3 omits this fixed normalization factor for clarity. The conditioning mask uses the original selection, whereas the loss support uses a separate 31 × 31 pixel-space dilation, followed by adaptive max-pooling to the token grid and thresholding at 0.5.

## D.3 Two-Stage Pretraining Schedule

To adapt the newly repurposed visual condition stream without immediately disrupting the pretrained image-generation prior, we use a two-stage schedule. Stage 1 trains the condition stream while freezing the image-specific backbone. Stage 2 then unfreezes the image backbone and jointly optimizes both streams using the same region-weighted flow-matching objective (Figure 5).

![](images/744f358e1afdd90ecef8a82d9c80f6576dd74f65bd416aac94a463c0647ed69b.jpg)  
Figure 5 Initialization and two-stage pretraining. Both visual branches are initialized from the pretrained Lens image branch. Stage 1 adapts the condition branch while freezing the image-specific backbone. Stage 2 jointly adapts both branches for removal and reconstruction.

In Stage 1, all image-specific parameters are frozen, including the image input projection, image-side attention projections, query/key normalization, modulation, layer normalization, and feed-forward layers. Condition-side parameters remain trainable, while the shared timestep embeddings and output normalization and projection are not frozen. Stage 2 jointly optimizes both streams. Pretraining uses nominal resolution buckets of 512, 768, 1024, 1280, and 1440, each with nine aspect ratios spanning 1:2 to 2:1.

## D.4 Initialization and Training Ablations

Table 7 compares alternative conditioning and initialization designs together with diferent pretraining schedules. The shared reference is the Stage 2 checkpoint obtained with Mask-Aware RoPE, image-branch initialization, and the two-stage schedule. Stage 1 denotes the endpoint of the initial frozen phase. No freeze optimizes both streams from the outset, whereas Always freeze continues the Stage 1 checkpoint for the second training phase while keeping the image-specific backbone frozen.

Visual conditioning and initialization. Mask-Aware RoPE achieves 65.80% HiFi Removal, compared with 58.80% for Channel Concat and 51.40% for Token Concat. Text-branch initialization retains a high Removal score (89.50) but collapses Inpainting to 1.30 and HiFi Removal to 0.00%, indicating that, in this setup, visual initialization is substantially more suitable for the condition stream than text-branch initialization.

Two-stage adaptation. The freeze–unfreeze schedule achieves 65.80% HiFi Removal, compared with 55.80% when both streams are optimized from the outset and 58.60% when the image backbone remains frozen throughout. This supports first adapting the visual condition stream while preserving the pretrained image prior, followed by joint adaptation for removal and reconstruction.

Table 7 Pretraining design ablations. The shared reference uses Mask-Aware RoPE, image-branch initialization, and two-stage freeze–unfreeze training. All rows use raw outputs. LensErase-Bench metrics follow Section 2.3; PSNR and LPIPS are equal-weight means over the six public benchmarks.
<table><tr><td rowspan="2">Model</td><td colspan="6">LensErase-Bench</td><td colspan="2">Public Benchmarks</td></tr><tr><td>Removal↑</td><td>Inpainting ↑</td><td>Preservation ↑</td><td>Cleanliness ↑</td><td>Overall ↑</td><td>HiFi Removal ↑</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>Mask-Aware RoPE (Stage 2)</td><td>82.35</td><td>84.65</td><td>91.05</td><td>92.15</td><td>86.51</td><td>65.80</td><td>28.03</td><td>0.0855</td></tr><tr><td colspan="9">Conditioning and initialization</td></tr><tr><td>Channel Concat</td><td>81.50</td><td>84.55</td><td>82.90</td><td>91.15</td><td>84.32</td><td>58.80</td><td>27.84</td><td>0.0869</td></tr><tr><td>Token Concat</td><td>77.50</td><td>76.85</td><td>84.90</td><td>87.15</td><td>80.78</td><td>51.40</td><td>27.41</td><td>0.0965</td></tr><tr><td>Text initialization</td><td>89.50</td><td>1.30</td><td>0.05</td><td>1.75</td><td>36.42</td><td>0.00</td><td>10.71</td><td>0.7420</td></tr><tr><td colspan="9">Training schedule</td></tr><tr><td>Stage 1</td><td>79.10</td><td>78.95</td><td>89.40</td><td>90.05</td><td>83.32</td><td>60.40</td><td>26.74</td><td>0.1089</td></tr><tr><td>No freeze</td><td>75.35</td><td>76.05</td><td>89.75</td><td>89.85</td><td>81.27</td><td>55.80</td><td>26.76</td><td>0.1020</td></tr><tr><td>Always freeze</td><td>80.75</td><td>80.25</td><td>88.00</td><td>88.20</td><td>83.59</td><td>58.60</td><td>27.66</td><td>0.0864</td></tr></table>

## E Post-Training and Inference Details

## E.1 Supervised Fine-Tuning

Appendix D describes the initialization and pretraining schedule. SFT uses rank-64 LoRA with scaling parameter 64 and zero dropout, while keeping the VAE and backbone weights fixed. The adapters are applied to the joint query/key/value and attention-output projections of both streams. We retain the same image–mask conditioning interface and region-weighted flow-matching objective as in pretraining. SFT is performed for $5 { , } 0 0 0$ optimization steps using a learning rate of $2 \times 1 0 ^ { - 5 }$ with a cosine schedule. SFT and RL use nominal resolution buckets of 512, 768, 1024, 1280, 1536, and 1792, with nine aspect ratios spanning 1:2 to 2:1.

## E.2 Contract-Guided Reward Construction

Following Section 3.2, reward construction for General Data first establishes a candidateindependent erase contract, then evaluates each candidate against this contract and applies the removal-aware reward gate (Figure 6). GPT-5.6 Sol serves as the VLM evaluator. Reference-guided adaptations for Poster, Watermark, and Visual Text are described below.

![](images/4083f6f1a1173964d1ccf30ea51488fe9ecb983dc8a57215d54e6ba81a6b9b49.jpg)  
Figure 6 Contract-guided reward construction for General Data. Source analysis establishes an erase contract shared across rollouts. Each candidate is evaluated against this fixed contract using semantic judgments and pixel fidelity. The five reward components are combined and passed through the removalaware gate in Equation 6. The resulting rewards define group advantages for DifusionNFT.

Establishing the erase contract. Before observing any candidate, the VLM analyzes the source image, selection mask, and mask-centered context to construct an erase contract $\mathcal { K } = ( \mathcal { T } , \mathcal { E } , \mathcal { O } )$ Here, T denotes the removal targets, E their clearly attributable visual efects, and O nearby independent content that should be preserved. The same contract and detail-view geometry are shared by all candidates of the same image–mask condition.

Evaluating candidate edits. Each candidate is evaluated against the fixed contract using the source image, selection mask, candidate output, and aligned detail views. Removal assesses whether T and E have been fully removed; Reconstruction measures the plausibility and local consistency of newly revealed content; Preservation assesses O and other non-target content; and Cleanliness captures residual fragments, unwanted additions, and technical artifacts. Outside Fidelity is computed separately from source–candidate PSNR over regions expected to remain unchanged.

Reward aggregation. VLM criteria are scored on a 0–4 scale and normalized to [0, 1], while computed fidelity scores are already normalized. For candidate $j ,$ let $s _ { j , p , k }$ denote a normalized subscore and let $A _ { j , p }$ contain the enabled, applicable subcriteria for component p. We aggregate

them as

$$
P _ { j , p } = \sum _ { k \in \mathcal { A } _ { j , p } } \widetilde { w } _ { j , p , k } s _ { j , p , k } ,\tag{17}
$$

where the weights are renormalized over applicable subcriteria. If an entire component is inapplicable, the remaining component weights are likewise renormalized. With the resulting weights $\widetilde { \lambda } _ { j , p } ,$ , the base reward is $\begin{array} { r } { b _ { j } = \sum _ { p } \widetilde { \lambda } _ { j , p } P _ { j , p } } \end{array}$ . The final reward follows the removal-aware gate in Equation 6.

Table 8 Reward weights before applicability renormalization and removal-aware gating. The last column summarizes the General Data criteria. Poster, Watermark, and Visual Text share component weights, with reference-specific subcriteria described below.
<table><tr><td>Component</td><td>General Data</td><td>Poster / Watermark/ Visual Text</td><td>General Data criteria</td></tr><tr><td>Removal</td><td>0.30</td><td>0.20</td><td>Target body absence (0.35), parts/identity absence (0.30), associated-effect removal (0.35)</td></tr><tr><td>Reconstruction</td><td>0.15</td><td>0.20</td><td>Semantic plausibility, structural continuity, and appearance/material consistency</td></tr><tr><td>Preservation</td><td>0.15</td><td>0.15</td><td>Independent-object preservation (0.60), background/scene-structure preservation (0.40)</td></tr><tr><td>Outside Fidelity</td><td>0.30</td><td>0.25</td><td>Outside-region PSNR score (1.00)</td></tr><tr><td>Cleanliness</td><td>0.10</td><td>0.20</td><td>Absence of residual fragments, unwanted additions (objects, duplicates, text, logos, or watermarks), and technical artifacts</td></tr></table>

Outside Fidelity. For General Data, the valid pixel region for candidate $j$ is

$$
\Omega _ { j } = \mathcal { V } \setminus \left( M _ { \mathrm { d i l } } \cup B _ { \mathcal { E } } \cup M _ { \Delta , j } \right) ,\tag{18}
$$

where V excludes padding, $M _ { \mathrm { d i l } }$ is a mildly dilated selection, and $B _ { \mathcal { E } }$ is the union of expanded source-stage efect boxes. $M _ { \Delta , j }$ further excludes size- and strength-qualified source–candidate residual components, including a gamma-0.75 pass for weak changes in dark regions. These residual components may be spatially detached from the mask, as associated reflections or shadows can extend beyond it. If this supplemental exclusion would cover more than 45% of the valid canvas, it is discarded while the selection and declared efect boxes are retained. Outside Fidelity is used only when at least 1,024 pixels and 8% of the valid canvas remain. These exclusions define the support of the pixel-fidelity metric, while semantic Preservation is still evaluated over the full image.

For an applicable PSNR measurement $p _ { j }$ in decibels, we map the interval 20–40 dB to [0, 1] with a clipped smoothstep:

$$
u _ { j } = \mathrm { c l i p } \bigg ( { \frac { p _ { j } - 2 0 } { 2 0 } } , 0 , 1 \bigg ) , \qquad s _ { j } ^ { \mathrm { a b s } } = u _ { j } ^ { 2 } ( 3 - 2 u _ { j } ) .\tag{19}
$$

We combine this absolute quality anchor with relative diferences among valid candidates of the same condition:

$$
s _ { j } ^ { \mathrm { P S N R } } = \frac { 1 } { 2 } s _ { j } ^ { \mathrm { a b s } } + \frac { 1 } { 2 } \frac { p _ { j } - p _ { \mathrm { m i n } } } { p _ { \mathrm { m a x } } - p _ { \mathrm { m i n } } } .\tag{20}
$$

The relative term is used only when at least two valid measurements are available and $p _ { \operatorname* { m a x } } { - p _ { \operatorname* { m i n } } } >$ $1 0 ^ { - 6 }$ dB. Otherwise, the score remains $s _ { j } ^ { \mathrm { a b s } }$

Reference-guided domains. Poster, Watermark, and Visual Text additionally use their clean or weak references. Their component weights are 0.20 for Removal, 0.20 for Reconstruction, 0.15 for Preservation, 0.25 for Outside Fidelity, and 0.20 for Cleanliness. Removal combines a deterministic reference-based score (0.60) with the VLM target-absence score (0.40). The deterministic score measures both progress toward the reference and suppression of the original source–reference residual over changed pixels. For Watermark, a spatial low-tail term additionally checks whether repeated overlay elements remain.

For Poster and Watermark, Reconstruction combines masked reference fidelity and visual recovery quality with weights 0.70 and 0.30. For Visual Text, whose weak reference may not represent a unique valid reconstruction, the corresponding weights are 0.40 and 0.60. Visual Text Preservation additionally balances retention of unselected text and scene content with the absence of newly introduced or replacement text. Cleanliness assesses residual fragments, unwanted additions, and technical artifacts in all three reference-guided domains. The PSNR blending in Equation 20 applies to Outside Fidelity in every domain and to masked reference fidelity for Poster and Watermark; the inside text-reference score remains absolute.

Removal-aware reward gating. After component aggregation, the candidate-specific form of the removal-aware gate in Equation 6 is

$$
n _ { 0 , j } = \sum _ { k \in A _ { j , \mathrm { R e m o v a l } } } \mathbb { 1 } [ s _ { j , \mathrm { R e m o v a l } , k } = 0 ] , \qquad r _ { j } = 0 . 5 ^ { n _ { 0 , j } } b _ { j } .
$$

Each zero-valued applicable Removal subcriterion therefore contributes a factor of 0.5, so multiple removal failures compound without hard-rejecting a candidate solely because one subcriterion is zero. The same rule is applied across all data domains.

## E.2.1 Reward Prompt Excerpts

The following excerpts summarize the principal semantic instructions used by the training reward evaluator. They illustrate the edit semantics and domain-specific judgments underlying Section 3.2; serialization details and output schemas are omitted. LensErase-Bench uses the separate referencefree evaluation protocol in Section 2.3.

General Data: source analysis. The mask is treated as object-selection evidence rather than an exact deletion boundary. If most of a recognizable object’s visible main body is selected, the complete object—including its parts, attachments, and visible extensions—is considered a removal target. Clearly attributable efects such as cast shadows, reflections, wakes, smoke, glow, or contact traces are included in the intended removal.

General Data: candidate assessment. All selected targets are assessed jointly, and a clear failure on one target cannot be compensated by successful removal of the others. Independent non-target objects are evaluated for both presence and structural integrity, including objects not explicitly listed in the source-stage contract.

Poster. Evaluation is guided by the selected layer semantics and the corresponding removed reference, not by generic inpainting expectations. Foreground content may therefore remain visible inside the selected region after a back or middle layer is removed.

Watermark. All selected repeated watermark instances belong to the removal target, while text, logos, symbols, signatures, QR codes, and graphics present in the clean reference are treated as protected content.

Visual Text. Because the weak reference may not be unique, a plausible clean reconstruction is not penalized solely for difering from it. Evaluation instead focuses on residual target text, reconstruction defects, newly introduced text, and damage to content that should remain.

## E.3 DifusionNFT Implementation

Group-relative advantages. For condition group $^ { g , }$ let $\bar { r } _ { g }$ and $\sigma _ { g }$ denote the mean and standard deviation of its valid candidate rewards, and let $\sigma _ { B }$ denote the standard deviation over all valid rewards in the current distributed collection. We compute

$$
\rho _ { g } = \operatorname * { m i n } \left( 1 , \frac { \sigma _ { g } } { 0 . 0 3 } \right) , \qquad A _ { j } = \mathrm { c l i p } \left( \rho _ { g } \frac { r _ { j } - \bar { r } _ { g } } { \sigma _ { B } + 1 0 ^ { - 4 } } , - 5 , 5 \right) .\tag{21}
$$

A group is used for training only if at least 12 of its 16 candidates receive valid rewards and the reward spread is nonzero. Collections with $\sigma { s } < 1 0 ^ { - 3 }$ are skipped. The factor $\rho _ { g }$ suppresses updates from groups with insuficient within-condition reward variation.

DifusionNFT objective. We convert the clipped advantage into $q _ { j } = ( 1 + A _ { j } / A _ { \operatorname* { m a x } } ) / 2$ . Let $v _ { \theta }$ denote the current policy, $v _ { \mathrm { o l d } }$ the rollout policy, and $v _ { \mathrm { r e f } }$ the SFT reference. Following DifusionNFT, we define

$$
v _ { \theta } ^ { \pm } = v _ { \mathrm { o l d } } \pm \beta \big ( v _ { \theta } - v _ { \mathrm { o l d } } \big ) .\tag{22}
$$

For a generated latent $\hat { z } _ { j }$ , we sample fresh Gaussian noise at each nonterminal scheduler time and form $z _ { j , t } = ( 1 - t ) \hat { z } _ { j } + t \epsilon$ , followed by $\hat { z } _ { j , 0 } ^ { \pm } = z _ { j , t } - t v _ { \theta } ^ { \pm }$ . Defining $\delta _ { j } ^ { \pm } = \hat { z } _ { j , 0 } ^ { \pm } - \hat { z } _ { j }$ , we use

$$
d _ { j } ^ { \pm } = \mathrm { s g } \Big [ \mathrm { m a x } \Big ( \mathrm { m e a n } | \delta _ { j } ^ { \pm } | , 1 0 ^ { - 5 } \Big ) \Big ] , \qquad \ell _ { j } ^ { \pm } = \frac { \mathrm { m e a n } [ ( \delta _ { j } ^ { \pm } ) ^ { 2 } ] } { d _ { j } ^ { \pm } } ,\tag{23}
$$

where sg denotes stop-gradient. The RL objective is

$$
\mathcal { L } _ { \mathrm { R L } } = \mathbb { E } _ { j , t , \epsilon } \bigg [ \frac { A _ { \mathrm { m a x } } } { \beta } \big ( q _ { j } \ell _ { j } ^ { + } + ( 1 - q _ { j } ) \ell _ { j } ^ { - } \big ) + \lambda _ { \mathrm { r e f } } \operatorname* { m e a n } \Big [ \big ( v _ { \theta } - v _ { \mathrm { r e f } } ) ^ { 2 } \Big ] \bigg ] .\tag{24}
$$

We use $\beta = 0 . 5 , A _ { \mathrm { m a x } } = 5$ , and $\lambda _ { \mathrm { r e f } } = 1 0 ^ { - 4 }$ . Gradients are propagated only through the current policy; sampled latents and the old and reference policies are treated as fixed targets. The RL reconstruction terms are averaged over the full latent and do not reuse the supervised mask weighting in Equation 3.

Policy updates. Each RL collection, corresponding to one policy update, samples conditions from a single data domain; diferent domains are not mixed within the same collection. We merge the SFT adapters into the backbone and use the resulting model as the fixed RL base and SFT reference. Fresh rank-64 LoRA adapters, with scaling parameter 64 and zero dropout, are attached to the same attention modules used during SFT for the current and rollout policies. We optimize the current adapters with AdamW at a learning rate of $5 \times 1 0 ^ { - 5 }$ and clip the gradient norm to 1. We perform 100 successful policy updates in total. After the kth successful update, the rollout policy is updated as

$$
\theta _ { \mathrm { o l d } }  \eta _ { k } \theta _ { \mathrm { o l d } } + ( 1 - \eta _ { k } ) \theta , \qquad \eta _ { k } = \mathrm { m i n } ( 0 . 0 0 1 k , 0 . 5 ) .\tag{25}
$$

The SFT reference remains fixed throughout RL training.

## E.4 Inference and Sampling

All methods receive the same source images and masks and produce one output per case, with fixed per-case seeds. LensErase uses a 20-step Euler sampler throughout, pads input dimensions to multiples of 16, and crops predictions back to their original size. All outputs are restored to the source dimensions for scoring.

Sampling. At inference, we initialize $z _ { 1 } \sim \mathcal { N } ( 0 , \mathbf { I } )$ and integrate from $t = 1$ to $t = 0$ with deterministic Euler updates:

$$
z _ { t + \Delta t } = z _ { t } + \Delta t v _ { \theta } ( z _ { t } , t , z _ { \mathrm { s r c } } ; M ) , \qquad \Delta t < 0 .\tag{26}
$$

Each step recomputes the velocity at the updated latent, while $z _ { \mathrm { s r c } } = E ( I _ { \mathrm { i n p u t } } )$ and M remain fixed. The Euler update has no independent learned parameters. A resolution-dependent timestep shift changes the sequence of step sizes but leaves this update rule unchanged. We use 20 Euler steps throughout. The VAE decodes only the terminal latent.

## E.5 TraceFusion Parameters

Let $M _ { \mathrm { d i l } }$ denote a dilation of the input selection mask. To suppress small reconstruction biases when estimating the efective edit region, we first Gaussian-smooth the signed residual between the generated output and source image and subtract its per-channel median outside $M _ { \mathrm { d i l } }$

$$
\begin{array} { r } { R = G _ { \sigma } \ast ( \widehat { I } _ { \mathrm { o u t p u t } } - I _ { \mathrm { i n p u t } } ) , \qquad d ( x ) = \left. R ( x ) - \mathrm { m e d i a n } _ { u \notin M _ { \mathrm { d i l } } } R ( u ) \right. _ { \infty } . } \end{array}\tag{27}
$$

Here, $G _ { \sigma }$ is a Gaussian kernel, ∗ denotes convolution, and the infinity norm takes the largest absolute response across color channels.

The detector in Section 3.3 operates on sRGB values in the 0–255 range. For an image of height H and width W, we define the resolution scale $s = \operatorname* { m i n } ( H , W ) / 5 1 2$ . The Gaussian standard deviation is max(0.5, 5s) pixels with reflected boundary handling, and $M _ { \mathrm { d i l } }$ uses a dilation radius of 12s pixels. If $M _ { \mathrm { d i l } }$ covers the full image, the median ofset in Equation 27 is set to zero.

We threshold $d ( x )$ at 4 and 10 to obtain low- and high-confidence change regions. The lowthreshold mask is opened with a disk of radius s, after which 8-connected components are extracted. A component is retained if it contains at least 48 $s ^ { 2 }$ pixels, including at least $8 s ^ { 2 }$ pixels above the high threshold. Components that do not intersect $M _ { \mathrm { d i l } }$ additionally require at least 1% of their pixels to exceed the high threshold. The retained components are closed with a disk of radius 3s and dilated by 5s, yielding the residual support $S _ { \mathrm { r e s } }$ . The primary edit support is then

$$
S _ { 0 } = M _ { \mathrm { d i l } } \cup S _ { \mathrm { r e s } } .\tag{28}
$$

To recover weak changes in dark regions, we apply the same detector after a shared gamma transform,

$$
I _ { \mathrm { i n p u t } , \gamma } = 2 5 5 \left( \frac { I _ { \mathrm { i n p u t } } } { 2 5 5 } \right) ^ { 0 . 7 5 } ,\tag{29}
$$

with the same transform applied to $\hat { I } _ { \mathrm { o u t p u t } }$ . Only changes not already covered by $S _ { 0 }$ are considered. We extract 8-connected components from this supplemental support and retain a component only if it contains at least $8 s ^ { 2 }$ pixels, intersects a dilation of $S _ { 0 }$ by radius s, and lies predominantly in dark regions. Specifically, at least half of its pixels must have luma at most 96 in both the untransformed source and generated images, where luma is computed as $0 . 2 1 2 6 R + 0 . 7 1 5 2 G + 0 . 0 7 2 2 B$ . Let $S _ { \gamma }$ denote the union of accepted supplemental components. The final fusion support is

$$
S = S _ { 0 } \cup S _ { \gamma } .\tag{30}
$$

All integer radii and area thresholds above are rounded and clamped to at least one.

Finally, we feather the boundary of S using its normalized Euclidean distance transform. Let $d _ { S } ( x )$ be zero inside S and the distance to $S$ outside it, with vertical and horizontal pixel spacings

1/H and $1 / W$ . The blending weight is

$$
\alpha ( x ) = \operatorname* { m a x } \left( 0 , 1 - \frac { d _ { S } ( x ) } { 0 . 1 } \right) .\tag{31}
$$

If S is empty, $\alpha = 0$ everywhere. The source and generated images are converted from sRGB to linear RGB before blending and converted back to sRGB for the final output.

We use the same TraceFusion parameters for all reported experiments.

## E.6 Notation Summary

Table 9 summarizes the principal notation used in Section 3 and Appendices D and E.

Table 9 Principal notation for LensErase.
<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td colspan="2">Images, masks, and positional coordinates</td></tr><tr><td> $I _ { \mathrm { i n p u t } }$ </td><td>Source image containing the selected targets.</td></tr><tr><td> $I _ { \mathrm { o u t p u t } }$ </td><td>Paired erase target used for supervised training.</td></tr><tr><td> $\hat { I } _ { \mathrm { o u t p u t } } , \ : \hat { I } _ { \mathrm { o u t p u t } } ^ { \mathrm { T F } }$ </td><td>Generated image before and after TraceFusion, respectively.</td></tr><tr><td> $M , m _ { i }$ </td><td>Binary selection mask and its max-pooled value for source token  $i ,$  with 1 denoting a selected region.</td></tr><tr><td> $h _ { i } , \ w _ { i }$ </td><td>Height and width coordinates of token i.</td></tr><tr><td> $\mathbf { p } _ { i } ^ { \mathrm { c o n d } } , \mathbf { p } _ { i } ^ { \mathrm { i m g } }$ </td><td>Source and noisy-target RoPE coordinates:  $( m _ { i } , h _ { i } , w _ { i } )$  and  $( 2 , h _ { i } , w _ { i } )$  2 respectively.</td></tr><tr><td>Latents and flow matching</td><td></td></tr><tr><td colspan="2"> $E$ </td></tr><tr><td></td><td>VAE encoder.</td></tr><tr><td> $z _ { \mathrm { s r c } } , z ^ { \star }$ </td><td>Source latent  $E ( I _ { \mathrm { i n p u t } } )$  and clean target latent  $E ( \boldsymbol { I _ { \mathrm { o u t p u t } } } )$ </td></tr><tr><td> $t , \epsilon , \mathbf { I }$ </td><td>Noise level  $t \in [ 0 , 1 ]$  , Gaussian noise  $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$  , and identity covariance.</td></tr><tr><td> $z _ { t }$   $v _ { \theta } , v ^ { \star }$ </td><td>Noisy target latent; during supervised training,  $z _ { t } = ( 1 - t ) z ^ { \star } + t \epsilon .$ </td></tr><tr><td></td><td>Predicted velocity  $v _ { \theta } ( z _ { t } , t , z _ { \mathrm { s r c } } ; M )$  with model parameters  $\theta ,$  and target velocity  $v ^ { \star } = \epsilon - z ^ { \star }$ </td></tr><tr><td> $N , i$ </td><td>Number of target tokens and token index, respectively.</td></tr><tr><td> $\tilde { M }$ </td><td>Binary loss mask obtained by dilating  $M$  and downsampling to the token grid.</td></tr><tr><td> $\lambda , \omega _ { i }$ </td><td>Masked-region loss multiplier and token weight  $\omega _ { i } = 1 + ( \lambda - 1 ) \tilde { M } _ { i }$ </td></tr><tr><td colspan="2">Contract-guided rewards</td></tr><tr><td> $\mathcal { K } = ( \mathcal { T } , \mathcal { E } , \mathcal { O } )$ </td><td>Erase contract containing removal targets, associated effects, and protected objects.</td></tr><tr><td> $\mathcal { P } , P _ { p } , \lambda _ { p }$ </td><td>Reward-component set, normalized component score, and component weight.</td></tr><tr><td> $\mathcal { A } _ { j , p } , s _ { j , p , k } , \widetilde { w } _ { j , p , k }$ </td><td>Applicable subcriteria, normalized subscore, and renormalized subcriterion weight for candidate j and component  $p .$ </td></tr><tr><td> $b , n _ { 0 } , r$ </td><td>Base reward, number of zero-valued Removal subcriteria, and gated reward  $r = 0 . 5 ^ { n _ { 0 } } b .$ </td></tr><tr><td> $j$ </td><td>Candidate index, suppressed in Section 3.2 and explicit in quantities such as  $P _ { j , p } { \mathrm { , } }$   $\widetilde { \lambda } _ { j , p } , b _ { j } , n _ { 0 , j }$  and  $\boldsymbol { r } _ { j } .$ </td></tr><tr><td colspan="2">TraceFusion</td></tr><tr><td> $M _ { \mathrm { d i l } }$   $R , d ( x )$ </td><td>Dilated selection mask used by the residual detector.</td></tr><tr><td></td><td>Smoothed signed source—output residual and its median-centered magnitude at pixel  $x ,$  respectively.</td></tr><tr><td> $S _ { \mathrm { r e s } }$ </td><td>Residual support retained after component filtering and morphology.</td></tr><tr><td> $S _ { 0 }$ </td><td>Initial support:  $M _ { \mathrm { d i l } } \cup S _ { \mathrm { r e s } } .$ </td></tr><tr><td> $\gamma , S _ { \gamma }$ </td><td>Shared power-transform exponent and the union of accepted dark-region supplemental components.</td></tr><tr><td> $S$ </td><td>Final fusion support:  $S _ { 0 } \cup S _ { \gamma } .$ </td></tr><tr><td> $d _ { S } ( x ) , \alpha ( x )$ </td><td>Distance to S (zero inside), using normalized pixel spacings, and blending weight  $\operatorname* { m a x } ( 0 , 1 - d _ { S } ( \stackrel { . } { x } ) / 0 . 1 )$ </td></tr><tr><td> $\mathcal { C } , \odot$ </td><td>sRGB-to-linear-RGB conversion and pixelwise multiplication.</td></tr></table>

## F Complete Benchmark Results

## F.1 Evaluation Protocol

All methods receive the same source images and masks and produce one output per case, with fixed per-case seeds. Clean targets are used only for scoring. Outputs are restored to the evaluation image dimensions.

CMMD (full-image resize). For CMMD [81], we bicubically resize full predictions and references to 336 × 336 without center cropping, to retain of-center removal regions and associated efects. Within each benchmark, all methods use the same samples, preprocessing, CLIP encoder, and MMD parameters.

## F.1.1 OmniPaint-Bench Mask Corrections

Among the 300 captured cases of OmniPaint-Bench [24], we correct 32 masks whose orientation is misaligned with the corresponding input image. We rotate 29 masks 90<sup>◦</sup> counterclockwise and flip three horizontally; the other 268 masks are unchanged. These transformations preserve mask pixel values and foreground area without interpolation. All input images and ground-truth targets remain unchanged, and ground-truth targets are not used to generate or redraw the masks. All evaluated methods receive the same corrected masks.

## F.2 Baseline Configurations

We use released checkpoints with the variants specified below. Code repositories and weights are linked for reproducibility.

ObjectClear. We use the released ObjectClear pipeline [3]<sup>1</sup> with attention-guided fusion.

SmartEraser. We run the released SmartEraser controller [2]<sup>2</sup>.

OmniPaint. We use OmniPaint’s FLUX.1-dev removal LoRA and fixed prompt embeddings [24]<sup>3</sup>.

OmniEraser. We evaluate OmniEraser-ControlNet [4]<sup>4</sup> using FLUX.1-dev, the Alimama inpainting ControlNet, and the released LoRA.

OSOR. We use OSOR-FLUX-Fill Phase II [23]<sup>5</sup>. Alpha-head latent blending is retained.

Attentive Eraser. We evaluate the SDXL+SIP configuration [30]<sup>6</sup>.

PANDORA. We use the SDXL configuration of PANDORA [28]<sup>7</sup>.

DesignEdit. We use DesignEdit’s SDXL object-removal mode [31]<sup>8</sup> with DDIM inversion and LayerFusion.

DORS. We use DORS with SDXL-Inpainting and SAM3 [27]<sup>9</sup>.

GeoRemover. We run both GeoRemover stages [26]<sup>10</sup> on FLUX.1-Fill-dev with Video Depth Anything Large.

EntityErasure. We use EntityErasure’s full amodal segmentation and completion pipeline [29]<sup>11</sup>.

RORem-mixed. We use RORem’s nine-channel mixed UNet [5]<sup>12</sup> with the released fixed prompts.

## F.3 Results on Public Benchmarks

Tables 10–15 expand Table 1 with all twelve baselines on each benchmark, using the same evaluation protocol and eight metrics. Red, orange, and yellow cell backgrounds mark the best, second-best, and third-best values among all thirteen methods within each benchmark. Best values are also bold; ties at the displayed precision share a color.

Table 10 Complete comparison on OBER-Test [3].
<table><tr><td>Method</td><td>PSNR↑</td><td>FID↓</td><td>CMMD↓</td><td>LPIPS↓</td><td>SSIM↑</td><td>DISTS↓</td><td>CLIP-I↑</td><td>DINO↑</td></tr><tr><td>SmartEraser [2]</td><td>26.67</td><td>94.58</td><td>0.4178</td><td>0.0762</td><td>0.9446</td><td>0.0792</td><td>0.9393</td><td>0.8423</td></tr><tr><td>RORem-mixed [5]</td><td>25.93</td><td>61.32</td><td>0.1832</td><td>0.1122</td><td>0.8549</td><td>0.0836</td><td>0.9593</td><td>0.9186</td></tr><tr><td>EntityErasure [29]</td><td>22.75</td><td>82.02</td><td>0.2933</td><td>0.1330</td><td>0.7263</td><td>0.1037</td><td>0.9376</td><td>0.8866</td></tr><tr><td>Attentive Eraser [30]</td><td>26.22</td><td>59.96</td><td>0.2048</td><td>0.1123</td><td>0.8497</td><td>0.0889</td><td>0.9605</td><td>0.9228</td></tr><tr><td>DesignEdit [31]</td><td>26.39</td><td>71.25</td><td>0.2052</td><td>0.1302</td><td>0.8585</td><td>0.0955</td><td>0.9485</td><td>0.8997</td></tr><tr><td>OmniEraser [4]</td><td>25.52</td><td>85.73</td><td>0.2131</td><td>0.0996</td><td>0.9141</td><td>0.0896</td><td>0.9457</td><td>0.8698</td></tr><tr><td>OmniPaint [24]</td><td>28.79</td><td>23.68</td><td>0.0829</td><td>0.0549</td><td>0.8575</td><td>0.0427</td><td>0.9836</td><td>0.9803</td></tr><tr><td>GeoRemover [26]</td><td>24.66</td><td>56.24</td><td>0.4277</td><td>0.1423</td><td>0.7622</td><td>0.1128</td><td>0.9305</td><td>0.9292</td></tr><tr><td>PANDORA [28]</td><td>25.63</td><td>79.21</td><td>0.3854</td><td>0.1482</td><td>0.8507</td><td>0.1056</td><td>0.9397</td><td>0.8723</td></tr><tr><td>DORS [27]</td><td>24.13</td><td>88.09</td><td>0.4874</td><td>0.1250</td><td>0.7492</td><td>0.0966</td><td>0.9371</td><td>0.8714</td></tr><tr><td>ObjectClear [3]</td><td>33.05</td><td>19.16</td><td>0.0132</td><td>0.0338</td><td>0.9353</td><td>0.0217</td><td>0.9894</td><td>0.9839</td></tr><tr><td>OSOR [23]</td><td>30.14</td><td>27.82</td><td>0.0674</td><td>0.0524</td><td>0.8748</td><td>0.0377</td><td>0.9827</td><td>0.9693</td></tr><tr><td>LensErase</td><td>34.96</td><td>16.97</td><td>0.0073</td><td>0.0256</td><td>0.9543</td><td>0.0176</td><td>0.9930</td><td>0.9839</td></tr></table>

Table 11 Complete comparison on RORD-Val-343 [25].
<table><tr><td>Method</td><td>PSNR↑</td><td>FID↓</td><td>CMMD↓</td><td>LPIPS↓</td><td>SSIM↑</td><td>DISTS↓</td><td>CLIP-I↑</td><td>DINO↑</td></tr><tr><td>SmartEraser [2]</td><td>24.89</td><td>35.41</td><td>0.1104</td><td>0.1658</td><td>0.8472</td><td>0.0950</td><td>0.9644</td><td>0.9349</td></tr><tr><td>RORem-mixed [5]</td><td>23.16</td><td>35.91</td><td>0.0983</td><td>0.1906</td><td>0.6893</td><td>0.0990</td><td>0.9482</td><td>0.9316</td></tr><tr><td>EntityErasure [29]</td><td>21.25</td><td>50.71</td><td>0.2829</td><td>0.3340</td><td>0.6142</td><td>0.1461</td><td>0.9400</td><td>0.9082</td></tr><tr><td>Attentive Eraser [30]</td><td>22.57</td><td>39.19</td><td>0.0948</td><td>0.1743</td><td>0.6896</td><td>0.0879</td><td>0.9501</td><td>0.9189</td></tr><tr><td>DesignEdit [31]</td><td>23.28</td><td>49.55</td><td>0.1603</td><td>0.1938</td><td>0.7268</td><td>0.1046</td><td>0.9304</td><td>0.8837</td></tr><tr><td>OmniEraser [4]</td><td>24.46</td><td>35.78</td><td>0.0426</td><td>0.1189</td><td>0.8380</td><td>0.0690</td><td>0.9580</td><td>0.9262</td></tr><tr><td>OmniPaint [24]</td><td>23.64</td><td>27.45</td><td>0.0334</td><td>0.1119</td><td>0.7823</td><td>0.0623</td><td>0.9672</td><td>0.9541</td></tr><tr><td>GeoRemover [26]</td><td>24.62</td><td>28.13</td><td>0.1716</td><td>0.1185</td><td>0.7659</td><td>0.0708</td><td>0.9640</td><td>0.9536</td></tr><tr><td>PANDORA [28]</td><td>21.61</td><td>68.85</td><td>0.5348</td><td>0.2274</td><td>0.7106</td><td>0.1211</td><td>0.8886</td><td>0.7928</td></tr><tr><td>DORS [27]</td><td>21.84</td><td>50.22</td><td>0.3032</td><td>0.2902</td><td>0.6253</td><td>0.1412</td><td>0.9330</td><td>0.8862</td></tr><tr><td>ObjectClear [3]</td><td>26.21</td><td>25.42</td><td>0.0511</td><td>0.1163</td><td>0.8577</td><td>0.0657</td><td>0.9704</td><td>0.9566</td></tr><tr><td>OSOR [23]</td><td>25.67</td><td>26.33</td><td>0.0390</td><td>0.1037</td><td>0.8138</td><td>0.0542</td><td>0.9663</td><td>0.9475</td></tr><tr><td>LensErase</td><td>28.23</td><td>20.53</td><td>0.0272</td><td>0.0601</td><td>0.9053</td><td>0.0357</td><td>0.9715</td><td>0.9568</td></tr></table>

Table 12 Complete comparison on RemovalBench [4].
<table><tr><td>Method</td><td>PSNR↑</td><td>FID↓</td><td>CMMD↓</td><td>LPIPS↓</td><td>SSIM↑</td><td>DISTS↓</td><td>CLIP-I↑</td><td>DINO↑</td></tr><tr><td>SmartEraser [2]</td><td>23.49</td><td>151.88</td><td>0.4359</td><td>0.3308</td><td>0.7695</td><td>0.1890</td><td>0.9339</td><td>0.8239</td></tr><tr><td>RORem-mixed [5]</td><td>23.51</td><td>90.95</td><td>0.2044</td><td>0.2262</td><td>0.7149</td><td>0.1175</td><td>0.9575</td><td>0.9098</td></tr><tr><td>EntityErasure [29]</td><td>20.36</td><td>161.80</td><td>0.3525</td><td>0.4287</td><td>0.6664</td><td>0.2288</td><td>0.9116</td><td>0.7918</td></tr><tr><td>Attentive Eraser [30]</td><td>23.78</td><td>80.49</td><td>0.1666</td><td>0.1941</td><td>0.7095</td><td>0.0969</td><td>0.9646</td><td>0.9263</td></tr><tr><td>DesignEdit [31]</td><td>23.96</td><td>108.46</td><td>0.2758</td><td>0.2343</td><td>0.7200</td><td>0.1357</td><td>0.9457</td><td>0.8676</td></tr><tr><td>OmniEraser [4]</td><td>22.33</td><td>147.90</td><td>0.3822</td><td>0.2150</td><td>0.7539</td><td>0.1285</td><td>0.9213</td><td>0.7990</td></tr><tr><td>OmniPaint [24]</td><td>24.57</td><td>43.25</td><td>0.0265</td><td>0.2103</td><td>0.7535</td><td>0.0998</td><td>0.9821</td><td>0.9663</td></tr><tr><td>GeoRemover [26]</td><td>25.35</td><td>42.05</td><td>0.2310</td><td>0.2316</td><td>0.7457</td><td>0.1181</td><td>0.9777</td><td>0.9698</td></tr><tr><td>PANDORA [28]</td><td>23.84</td><td>98.21</td><td>0.4140</td><td>0.2389</td><td>0.7167</td><td>0.1413</td><td>0.9359</td><td>0.8427</td></tr><tr><td>DORS [27]</td><td>23.07</td><td>141.61</td><td>0.4622</td><td>0.3755</td><td>0.6987</td><td>0.1935</td><td>0.9419</td><td>0.8629</td></tr><tr><td>ObjectClear [3]</td><td>25.99</td><td>46.61</td><td>0.0248</td><td>0.2853</td><td>0.7794</td><td>0.1428</td><td>0.9853</td><td>0.9718</td></tr><tr><td>OSOR [23]</td><td>25.80</td><td>43.89</td><td>0.0275</td><td>0.1935</td><td>0.7656</td><td>0.0882</td><td>0.9838</td><td>0.9597</td></tr><tr><td>LensErase</td><td>28.16</td><td>33.67</td><td>0.0116</td><td>0.1125</td><td>0.8306</td><td>0.0586</td><td>0.9874</td><td>0.9737</td></tr></table>

Table 13 Complete comparison on OmniPaint-Bench [24] (captured subset), excluding RORD-derived cases.
<table><tr><td>Method</td><td>PSNR↑</td><td>FID↓</td><td>CMMD↓</td><td>LPIPS↓</td><td>SSIM↑</td><td>DISTS↓</td><td>CLIP-I↑</td><td>DINO↑</td></tr><tr><td>SmartEraser [2]</td><td>22.05</td><td>94.79</td><td>0.2854</td><td>0.2787</td><td>0.8121</td><td>0.1460</td><td>0.9189</td><td>0.8610</td></tr><tr><td>RORem-mixed [5]</td><td>23.14</td><td>54.66</td><td>0.1398</td><td>0.1994</td><td>0.7730</td><td>0.1072</td><td>0.9433</td><td>0.9226</td></tr><tr><td>EntityErasure [29]</td><td>20.44</td><td>96.41</td><td>0.3472</td><td>0.3033</td><td>0.7262</td><td>0.1583</td><td>0.9068</td><td>0.8579</td></tr><tr><td>Attentive Eraser [30]</td><td>23.31</td><td>50.60</td><td>0.1324</td><td>0.1642</td><td>0.7865</td><td>0.0719</td><td>0.9452</td><td>0.9322</td></tr><tr><td>DesignEdit [31]</td><td>23.30</td><td>68.54</td><td>0.2245</td><td>0.1838</td><td>0.7859</td><td>0.1033</td><td>0.9261</td><td>0.8838</td></tr><tr><td>OmniEraser [4]</td><td>21.73</td><td>93.71</td><td>0.2645</td><td>0.1767</td><td>0.8123</td><td>0.0904</td><td>0.9059</td><td>0.8212</td></tr><tr><td>OmniPaint [24]</td><td>23.42</td><td>40.65</td><td>0.0743</td><td>0.1484</td><td>0.8193</td><td>0.0644</td><td>0.9562</td><td>0.9423</td></tr><tr><td>GeoRemover [26]</td><td>24.87</td><td>38.54</td><td>0.1376</td><td>0.1575</td><td>0.8173</td><td>0.0704</td><td>0.9524</td><td>0.9494</td></tr><tr><td>PANDORA [28]</td><td>23.26</td><td>83.41</td><td>0.3606</td><td>0.1940</td><td>0.7835</td><td>0.1122</td><td>0.9130</td><td>0.8538</td></tr><tr><td>DORS [27]</td><td>22.20</td><td>79.22</td><td>0.3511</td><td>0.2799</td><td>0.7479</td><td>0.1389</td><td>0.9246</td><td>0.8949</td></tr><tr><td>ObjectClear [3]</td><td>24.93</td><td>48.91</td><td>0.0981</td><td>0.2233</td><td>0.8178</td><td>0.1054</td><td>0.9613</td><td>0.9549</td></tr><tr><td>OSOR [23]</td><td>26.02</td><td>37.52</td><td>0.0687</td><td>0.1336</td><td>0.8421</td><td>0.0566</td><td>0.9617</td><td>0.9506</td></tr><tr><td>LensErase</td><td>26.22</td><td>33.29</td><td>0.0245</td><td>0.1257</td><td>0.8521</td><td>0.0544</td><td>0.9660</td><td>0.9595</td></tr></table>

Table 14 Complete comparison on AnimeEraseBench [23].
<table><tr><td rowspan=1 colspan=9>Method                                         PSNR↑ FID↓ CMMD↓LPIPS↓SSIM↑DISTS↓CLIP-I↑DINO↑</td></tr><tr><td rowspan=1 colspan=2>SmartEraser [2]                                  23.90</td><td rowspan=1 colspan=1>55.01</td><td rowspan=1 colspan=1>0.0451</td><td rowspan=1 colspan=1>0.2116</td><td rowspan=1 colspan=1>0.8252</td><td rowspan=1 colspan=1>0.1174</td><td rowspan=1 colspan=1>0.9837</td><td rowspan=1 colspan=1>0.9577</td></tr><tr><td rowspan=1 colspan=2>RORem-mixed [5]                                24.42</td><td rowspan=1 colspan=1>33.68</td><td rowspan=1 colspan=1>0.0261</td><td rowspan=1 colspan=1>0.1305</td><td rowspan=1 colspan=1>0.7926</td><td rowspan=1 colspan=1>0.0654</td><td rowspan=1 colspan=1>0.9875</td><td rowspan=1 colspan=1>0.9756</td></tr><tr><td rowspan=1 colspan=2>EntityErasure [29]                                20.26</td><td rowspan=1 colspan=1>83.39</td><td rowspan=1 colspan=1>0.0906</td><td rowspan=1 colspan=1>0.3574</td><td rowspan=1 colspan=1>0.6143</td><td rowspan=1 colspan=1>0.1776</td><td rowspan=1 colspan=1>0.9711</td><td rowspan=1 colspan=1>0.9282</td></tr><tr><td rowspan=1 colspan=1>Attentive Eraser [30]</td><td rowspan=1 colspan=1>22.60</td><td rowspan=1 colspan=1>37.77</td><td rowspan=1 colspan=1>0.0315</td><td rowspan=1 colspan=1>0.1329</td><td rowspan=1 colspan=1>0.7321</td><td rowspan=1 colspan=1>0.0629</td><td rowspan=1 colspan=1>0.9852</td><td rowspan=1 colspan=1>0.9699</td></tr><tr><td rowspan=1 colspan=1>DesignEdit [31]</td><td rowspan=1 colspan=1>24.10</td><td rowspan=1 colspan=1>49.33</td><td rowspan=1 colspan=1>0.0666</td><td rowspan=1 colspan=1>0.1582</td><td rowspan=1 colspan=1>0.7861</td><td rowspan=1 colspan=1>0.0846</td><td rowspan=1 colspan=1>0.9800</td><td rowspan=1 colspan=1>0.9519</td></tr><tr><td rowspan=1 colspan=1>OmniEraser [4]</td><td rowspan=1 colspan=1>24.05</td><td rowspan=1 colspan=1>49.39</td><td rowspan=1 colspan=1>0.0383</td><td rowspan=1 colspan=1>0.1155</td><td rowspan=1 colspan=1>0.8569</td><td rowspan=1 colspan=1>0.0617</td><td rowspan=1 colspan=1>0.9772</td><td rowspan=1 colspan=1>0.9273</td></tr><tr><td rowspan=1 colspan=1>OmniPaint [24]</td><td rowspan=1 colspan=1>25.58</td><td rowspan=1 colspan=1>27.76</td><td rowspan=1 colspan=1>0.0138</td><td rowspan=1 colspan=1>0.1195</td><td rowspan=1 colspan=1>0.8581</td><td rowspan=1 colspan=1>0.0636</td><td rowspan=1 colspan=1>0.9874</td><td rowspan=1 colspan=1>0.9764</td></tr><tr><td rowspan=1 colspan=1>GeoRemover [26]</td><td rowspan=1 colspan=1>24.80</td><td rowspan=1 colspan=1>30.72</td><td rowspan=1 colspan=1>0.0533</td><td rowspan=1 colspan=1>0.1361</td><td rowspan=1 colspan=1>0.8247</td><td rowspan=1 colspan=1>0.0734</td><td rowspan=1 colspan=1>0.9828</td><td rowspan=1 colspan=1>0.9766</td></tr><tr><td rowspan=1 colspan=1>PANDORA [28]</td><td rowspan=1 colspan=1>23.38</td><td rowspan=1 colspan=1>54.81</td><td rowspan=1 colspan=1>0.0968</td><td rowspan=1 colspan=1>0.1621</td><td rowspan=1 colspan=1>0.7787</td><td rowspan=1 colspan=1>0.0911</td><td rowspan=1 colspan=1>0.9716</td><td rowspan=1 colspan=1>0.9274</td></tr><tr><td rowspan=1 colspan=1>DORS [27]</td><td rowspan=1 colspan=1>21.59</td><td rowspan=1 colspan=1>73.28</td><td rowspan=1 colspan=1>0.0925</td><td rowspan=1 colspan=1>0.3123</td><td rowspan=1 colspan=1>0.6459</td><td rowspan=1 colspan=1>0.1548</td><td rowspan=1 colspan=1>0.9742</td><td rowspan=1 colspan=1>0.9460</td></tr><tr><td rowspan=1 colspan=1>ObjectClear [3]</td><td rowspan=1 colspan=1>25.42</td><td rowspan=1 colspan=1>36.76</td><td rowspan=1 colspan=1>0.0121</td><td rowspan=1 colspan=1>0.1791</td><td rowspan=1 colspan=1>0.8417</td><td rowspan=1 colspan=1>0.0978</td><td rowspan=1 colspan=1>0.9914</td><td rowspan=1 colspan=1>0.9821</td></tr><tr><td rowspan=1 colspan=1>OSOR [23]</td><td rowspan=1 colspan=1>27.96</td><td rowspan=1 colspan=1>24.45</td><td rowspan=1 colspan=1>0.0094</td><td rowspan=1 colspan=1>0.0967</td><td rowspan=1 colspan=1>0.8870</td><td rowspan=1 colspan=1>0.0475</td><td rowspan=1 colspan=1>0.9900</td><td rowspan=1 colspan=1>0.9751</td></tr><tr><td rowspan=2 colspan=3>LensErase                                        30.50   17.20</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.0094</td><td rowspan=1 colspan=1>0.0527</td><td rowspan=1 colspan=1>0.9398</td><td rowspan=1 colspan=1>0.0159</td><td rowspan=1 colspan=1>0.9920</td><td rowspan=1 colspan=1>0.9798</td></tr></table>

Table 15 Complete comparison on TextEraseBench [23].
<table><tr><td>Method</td><td>PSNR↑</td><td>FID↓</td><td>CMMD↓</td><td>LPIPS↓</td><td>SSIM↑</td><td>DISTS↓</td><td>CLIP-I↑</td><td>DINO↑</td></tr><tr><td>SmartEraser [2]</td><td>28.85</td><td>47.96</td><td>0.0608</td><td>0.1686</td><td>0.8885</td><td>0.1049</td><td>0.9430</td><td>0.9471</td></tr><tr><td>RORem-mixed [5]</td><td>29.50</td><td>24.83</td><td>0.0178</td><td>0.1034</td><td>0.8292</td><td>0.0584</td><td>0.9815</td><td>0.9744</td></tr><tr><td>EntityErasure [29]</td><td>24.98</td><td>53.79</td><td>0.1175</td><td>0.2653</td><td>0.7307</td><td>0.1502</td><td>0.9427</td><td>0.9414</td></tr><tr><td>Attentive Eraser [30]</td><td>26.31</td><td>34.70</td><td>0.0228</td><td>0.1134</td><td>0.7956</td><td>0.0563</td><td>0.9738</td><td>0.9613</td></tr><tr><td>DesignEdit [31]</td><td>27.61</td><td>49.23</td><td>0.1002</td><td>0.1391</td><td>0.8289</td><td>0.0883</td><td>0.9578</td><td>0.9247</td></tr><tr><td>OmniEraser [4]</td><td>30.70</td><td>21.97</td><td>0.0195</td><td>0.0763</td><td>0.9005</td><td>0.0435</td><td>0.9812</td><td>0.9758</td></tr><tr><td>OmniPaint [24]</td><td>28.14</td><td>27.78</td><td>0.0188</td><td>0.1085</td><td>0.8798</td><td>0.0607</td><td>0.9772</td><td>0.9622</td></tr><tr><td>GeoRemover [26]</td><td>26.26</td><td>60.36</td><td>0.1997</td><td>0.1346</td><td>0.8613</td><td>0.0804</td><td>0.8907</td><td>0.9169</td></tr><tr><td>PANDORA [28]</td><td>26.39</td><td>53.93</td><td>0.1185</td><td>0.1412</td><td>0.8202</td><td>0.0914</td><td>0.9463</td><td>0.9106</td></tr><tr><td>DORS [27]</td><td>25.75</td><td>53.04</td><td>0.0676</td><td>0.2467</td><td>0.7484</td><td>0.1332</td><td>0.9612</td><td>0.9343</td></tr><tr><td>ObjectClear [3]</td><td>30.37</td><td>27.81</td><td>0.0193</td><td>0.1329</td><td>0.8976</td><td>0.0822</td><td>0.9774</td><td>0.9749</td></tr><tr><td>OSOR [23]</td><td>32.04</td><td>18.52</td><td>0.0093</td><td>0.0728</td><td>0.9126</td><td>0.0412</td><td>0.9843</td><td>0.9771</td></tr><tr><td>LensErase</td><td>36.88</td><td>13.51</td><td>0.0091</td><td>0.0339</td><td>0.9654</td><td>0.0135</td><td>0.9906</td><td>0.9822</td></tr></table>

## F.4 Results on LensErase-Bench

Table 16 reports mean scores over LensErase-Bench. PANDORA uses 499 cases because one output was consistently rejected by GPT-5.6 Sol’s safety filter; all other methods use 500. HiFi Removal denotes the percentage of cases achieving complete removal (Removal = 4) while maintaining Inpainting, Preservation, and Cleanliness scores of at least 3.

Table 16 VLM comparison on LensErase-Bench.
<table><tr><td>Method</td><td>Removal↑</td><td>Inpainting ↑</td><td>Preservation ↑</td><td>Cleanliness ↑</td><td>Overall ↑</td><td>HiFi Removal (%) ↑</td></tr><tr><td>SmartEraser [2]</td><td>23.40</td><td>24.15</td><td>66.25</td><td>57.75</td><td>38.99</td><td>3.60</td></tr><tr><td>RORem-mixed [5]</td><td>73.95</td><td>59.70</td><td>74.05</td><td>67.10</td><td>69.75</td><td>27.40</td></tr><tr><td>EntityErasure [29]</td><td>59.65</td><td>48.50</td><td>48.45</td><td>60.20</td><td>55.29</td><td>4.80</td></tr><tr><td>Attentive Eraser [30]</td><td>78.40</td><td>52.25</td><td>61.50</td><td>55.05</td><td>65.12</td><td>20.60</td></tr><tr><td>DesignEdit [31]</td><td>76.95</td><td>60.25</td><td>63.85</td><td>62.95</td><td>68.19</td><td>25.00</td></tr><tr><td>OmniEraser [4]</td><td>55.65</td><td>47.95</td><td>77.55</td><td>57.65</td><td>58.89</td><td>22.60</td></tr><tr><td>OmniPaint [24]</td><td>69.20</td><td>66.15</td><td>80.20</td><td>83.20</td><td>73.59</td><td>42.80</td></tr><tr><td>GeoRemover [26]</td><td>54.30</td><td>49.85</td><td>77.05</td><td>78.00</td><td>62.70</td><td>25.60</td></tr><tr><td>PANDORA [28]</td><td>41.93</td><td>39.28</td><td>58.17</td><td>49.50</td><td>46.16</td><td>7.21</td></tr><tr><td>DORS [27]</td><td>55.20</td><td>27.40</td><td>39.10</td><td>32.60</td><td>41.90</td><td>0.60</td></tr><tr><td>ObjectClear [3]</td><td>73.95</td><td>61.45</td><td>73.80</td><td>75.20</td><td>71.67</td><td>28.40</td></tr><tr><td>OSOR [23]</td><td>73.85</td><td>60.45</td><td>75.55</td><td>73.55</td><td>71.45</td><td>32.60</td></tr><tr><td>LensErase</td><td>95.15</td><td>92.40</td><td>96.35</td><td>97.05</td><td>95.22</td><td>82.80</td></tr></table>

## G Reward Component Ablations

We examine five variants of the reward in Section 3.2, each omitting one component: Removal, Reconstruction, Preservation, Outside Fidelity, or Cleanliness. The remaining component weights are renormalized. The variant without Removal also disables the removal-aware gate, whereas the other four retain it. All variants start from the same SFT checkpoint, with optimization details given in Appendix E.3. Table 17 reports the VLM metrics together with PSNR and LPIPS.

Removal component and gate. Jointly removing the Removal component and the removal-aware gate causes the largest degradation, with Removal dropping from 94.95 to 43.35 and HiFi Removal from 81.60% to 38.80%.

Table 17 Reward-component ablations. PSNR and LPIPS are equal-weight means over the six public benchmarks.
<table><tr><td rowspan="2">Model</td><td colspan="6">LensErase-Bench</td><td colspan="2">Public Benchmarks</td></tr><tr><td>Removal ↑</td><td>Inpainting ↑</td><td>Preservation ↑</td><td>Cleanliness ↑</td><td>Overall ↑</td><td>HiFi Removal ↑</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>RL-Raw (full reward)</td><td>94.95</td><td>92.65</td><td>93.35</td><td>96.15</td><td>94.41</td><td>81.60</td><td>28.70</td><td>0.0816</td></tr><tr><td>w/o Removal + gate</td><td>43.35</td><td>46.10</td><td>98.70</td><td>98.10</td><td>65.92</td><td>38.80</td><td>21.85</td><td>0.1446</td></tr><tr><td>w/o Reconstruction</td><td>90.85</td><td>91.40</td><td>96.10</td><td>95.70</td><td>92.98</td><td>78.00</td><td>28.32</td><td>0.0791</td></tr><tr><td>w/o Preservation</td><td>94.15</td><td>91.50</td><td>89.20</td><td>96.00</td><td>93.00</td><td>74.60</td><td>28.01</td><td>0.0817</td></tr><tr><td>w/o Outside Fidelity</td><td>92.90</td><td>91.70</td><td>94.40</td><td>96.40</td><td>93.66</td><td>78.00</td><td>26.69</td><td>0.0888</td></tr><tr><td>w/o Cleanliness</td><td>90.90</td><td>90.95</td><td>96.25</td><td>95.95</td><td>92.99</td><td>78.40</td><td>28.48</td><td>0.0803</td></tr></table>

Complementary reward components. The remaining components provide complementary benefits. Removing Preservation yields the lowest HiFi Removal among these variants (74.60%), while removing Outside Fidelity causes the largest PSNR drop, from 28.70 to 26.69 dB. Removing Reconstruction or Cleanliness also reduces HiFi Removal to 78.00% and 78.40%, respectively. Overall, the full reward achieves the best HiFi Removal.

## H Limitations and Conclusion

Despite its broad coverage, LensErase still has several limitations. First, scaling training to ultra-high resolutions remains challenging. Our large-scale data construction relies in part on existing general-purpose image editing models, which remain less reliable for ultra-high-resolution editing, particularly at resolutions such as 4K. This makes it dificult to construct suficiently large and consistent 4K-scale training data. Future work could explore more reliable high-resolution data construction and more challenging erase scenarios. Overall, LensErase establishes a scalable framework for general-purpose mask-based erasing by combining large-scale data, generative-prior adaptation, reward-based post-training, and high-fidelity post-processing. We hope it provides a useful foundation for future erase-related models and visual editing systems.

## I Qualitative Comparisons

## I.1 Comparisons with Mask-based GPT-Image-2.5-Sunburst [32] and Flux2-Erase [33]

![](images/752a1c0f7c87045cb7145ccecbd06f99cd945ef047d478e2ba2ac387abc78adc.jpg)  
Figure 7 Qualitative comparisons with GPT-Image 2.5 and Flux2-Erase.

![](images/270263fcdaeef1d201cb31df61659f1e52fc1f54ce2c79231aebbe435afcf159.jpg)  
Figure 8 Qualitative comparisons with GPT-Image 2.5 and Flux2-Erase.

![](images/91a9bba091b58edf98041ba6d0065a9b33e53ef080b33e9353b4344a11ecdb00.jpg)  
Figure 9 Qualitative comparisons with GPT-Image 2.5 and Flux2-Erase.

![](images/facae6716e42224ee5144c78e231e717cc1fe833fd89b81a4a920e67924fa132.jpg)  
Figure 10 Qualitative comparisons with GPT-Image 2.5 and Flux2-Erase.

## I.2 Comparisons with Object-Removal Baselines

![](images/576196b6b69e3c95d85a6121a5059a21894ac7e67bfd5ffc480eb1d77e9bd920.jpg)  
Figure 11 Qualitative comparisons on the General subset.

![](images/d264de19c31649aa17d397041895754f6b0b0265d8fc1df28aeff1d9af44fcd3.jpg)  
Figure 12 Qualitative comparisons on General images and posters.

![](images/e1a0dd8e1d5562987dbb36c3ad40f6e265bf5e3d945b1306c197e5ca4cb3f98e.jpg)  
Figure 13 Qualitative comparisons on posters.

![](images/8b1cfeab33c207e0c7a0c2e3dcc338ffc27dc56a7bc50b63310395c1a0fd5ea0.jpg)  
Figure 14 Qualitative comparisons on posters.

![](images/97285caf79922adf3cd8f40c269775f036b0fef6c6f0c08c8b05b970df467160.jpg)  
Figure 15 Qualitative comparisons on watermark removal.

![](images/0a718e1278c250d079d8bcaba8710ec176a5c30d85c847e5e12055994a42b040.jpg)  
Figure 16 Qualitative comparisons on watermark and visual text removal.

![](images/c31bef9f6ae5a1cbd6ed1d649597bb892d853c90cb8c7c130ad0876e2711c6fe.jpg)  
Figure 17 Qualitative comparisons on visual text removal.

## J Removal with Hand-Drawn and SAM Masks

Figures 18–21 present additional LensErase results using hand-drawn masks and SAM 2 segmentations [10]. The examples include text, transparent objects, multiple targets, and associated shadows or reflections. Each row shows the input image, mask, and LensErase output.

![](images/376252cec10c7424393829374973d7c799e77f70fb90913319bd67fa1990837f.jpg)  
Figure 18 LensErase results with user-provided masks.

![](images/ef70cc2e0a6c7acd0ed70f824fbcfbb31abe34a3ed06431dedbbac6fbcf40cbb.jpg)  
Figure 19 LensErase results with user-provided masks.

![](images/5774015a200ae48410719fda9de5f325d1f7ddc6aa5af75609787d9be7850d0f.jpg)  
Figure 20 LensErase results with user-provided masks.

![](images/12f299f57d11c3a744dc5575006f6a6435bfd2d0f9d64a5a4a760e4948230b57.jpg)  
Figure 21 LensErase results with user-provided masks.