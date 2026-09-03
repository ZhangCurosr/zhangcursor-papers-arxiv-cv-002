# Structured-Prior-Guided Difusion Inpainting with Physical Consistency for Trafic Sign Augmentation

Luo Li   
liluo.ll@alibaba-inc.com   
AMAP, Alibaba Group   
Beijing, China   
Qiang Gao   
longmo.gq@autonavi.com   
AMAP, Alibaba Group   
Beijing, China   
Chongchong Huang   
huangchongchong.hc@alibaba  
inc.com   
AMAP, Alibaba Group   
Beijing, China

Xinlong Liu wendiao.lxl@alibaba-inc.com AMAP, Alibaba Group Beijing, China

Liang Cao<sup>∗</sup>   
xuanhe.xh@autonavi.com   
AMAP, Alibaba Group   
Beijing, China   
Jun Jia   
tianchen.jj@alibaba-inc.com   
AMAP, Alibaba Group   
Beijing, China

Gui Yang yanggui.yg@alibaba-inc.com AMAP, Alibaba Group Beijing, China

## Abstract

Trafic sign detection faces a long-tailed data distribution. Many rare signs matter as much as common ones from a regulatory standpoint, yet they have very few samples. Generative data augmentation is one way out. General-purpose inpainting models, however, distort digits, deform geometry and perspective, and shift colours when applied directly to sign regions. We trace this to a single gap: the conditioning signal is too abstract for the physical composition of a sign. We propose a structured-prior-guided difusion inpainting framework with physical consistency. It injects the semantic, appearance and geometric priors of a sign through three orthogonal pathways: a JSON-formatted text prompt, a front-view vector template rendered with measured dominant colours (via IP-Adapter), and an afine-aligned vector template (via ControlNet). Two physical consistency losses constrain colour with a CIELAB chromaticity � term and edge structure with a Sobel gradient term. We train by self-supervised reconstruction on a large set of images collected in-house at AMAP, then evaluate zero-shot on the public TT100K-2021 dataset, a diferent source. Our method uses a Stable Difusion 1.5 backbone of about 1.4B parameters. It beats seven representative competitors on every metric of reconstruction fidelity, physical consistency and semantic controllability. Its OCR exact-match rate reaches 91.1%, against 44.2% for the 12B industrial model FLUX.1 Fill [dev], and it needs only 1/14 of that model’s inference time. Leave-one-out ablations confirm that each of the three prior pathways and both loss terms contribute on their own. In downstream detection, the synthetic data raises the group-pooled AP50 of rare classes by 1.23× to 7.40× over a realdata-only baseline. Code and pre-trained models are available at https://github.com/52hz-whale/TraficSignInpaint.

## 1 Introduction

Trafic sign detection and recognition is a core part of autonomous driving perception and of automated high-definition map production. Large datasets such as TT100K [1] and GTSRB [2] have appeared, and the YOLO [3–6] and DETR [7, 8] detector families have matured. Accuracy on common signs is close to saturated. What still limits usability is the small set of categories in the tail. This tail reflects real-world frequency, not sampling bias. More collection mileage buys very little, and tail categories are usually small objects as well.

Conventional data augmentation has a clear ceiling. Geometric and photometric transforms and Mosaic [9] leave semantic content untouched, so they cannot create new categories. Copy-Paste [10, 11] and its Poisson-blending [12] variants move objects around, but the pasted content still comes from existing instances. Resampling and class-balanced losses [13] only reweight the optimisation. They add no information. Difusion models [14–17] and the controllable-generation ecosystem around them [18–20] open another route. Suppose we can replace a sign in a real scene with a sign of any chosen category and content, and the new sign keeps the illumination, occlusion, background and viewpoint of the original location. Real scenes then become carriers for long-tailed samples, and the synthetic data inherits the scene diversity of real data. Recent work synthesises trafic visual elements with generative models [21] and detection training data with difusion models [22– 26]. A general-purpose inpainting model applied directly to signs, however, fails in predictable ways.

Three failure modes appear, and they are entangled. The first is distortion of the text on the sign face. The semantic core of a sign is often two or three characters. “Speed limit 100” and “speed limit 10” are entirely diferent regulatory constraints. Difusion denoising guarantees nothing about character-level topological correctness [27–29]. A wrong digit is label noise. It does more harm than generating nothing at all. The second is destruction of geometry and perspective. The shape of a sign carries semantics and obeys perspective projection. A circular sign appears as an ellipse whose axis ratio and tilt follow the viewing angle. The three vertices of a triangular sign satisfy a specific perspective relation. Natural language cannot convey these continuous quantities, and generated results come out with incomplete edges and skewed shapes. The third is colour inconsistency with the background. The measured colour of one sign difers enormously under harsh noon sunlight, low-angle dusk light and rainy difuse light. A model that paints the standard colours produces a result detached from the surrounding illumination. For the downstream detector that is a domain shift.

All three have one cause. The conditioning information sits at the wrong level of abstraction for the target. A trafic sign is a standardised artefact. National standards prescribe its semantics, colour scheme and geometry in detail. It carries little information but is strongly structured. Natural language is a low-precision, ambiguous condition. Driving generation with it asks the model to guess quantities that are already known.

We define the task as follows. A real road image comes with a mask ofthe sign region and a target category label. Structured priors guide a difusion model to repaint that region, and the new sign must follow the standard in semantics, colour and geometry. Our idea is to split the known priors of a sign along physical dimensions and give each one the conditioning form it fits. Semantics becomes a structured JSON text prompt. Appearance colour becomes a frontview vector template infused with the measured dominant colours, injected through IP-Adapter [20]. Geometry becomes an afinealigned vector template, injected through ControlNet [18]. The three pathways are orthogonal, and none substitutes for another. Together they let the model read the priors instead of guessing them.

Our main contributions are as follows.

• A structured-prior-guided difusion inpainting framework. We split sign priors into semantics, appearance and geometry, and inject them through separate pathways. An automated pipeline builds all three conditions from raw annotations, using SAM 2 [30] segmentation, analytic geometric fitting and colour clustering. Training is self-supervised reconstruction. On top of the difusion objective we add a CIELAB chromaticity $L _ { 1 }$ loss for colour and a Sobel gra dient $L _ { 1 }$ loss for edge structure. At inference, editing the conditions edits category, digits, colour and geometry.

• Comprehensive leadership in generation quality and controllability. Under a unified TT100K evaluation protocol, our method ranks first on all nine metrics. Structured prior modelling on a vertical task beats merely scaling up a general-purpose model.

• Substantial gains from synthetic data on rare-class detection. We verify this with YOLO26 on TT100K rare classes. Our synthetic data raises pooled AP50 by 1.23× to 7.40×, and the gain reaches 7.40× when only 10% of the real data is available.

## 2 Related Work

Image inpainting. Early inpainting used GAN and Transformer backbones. LaMa [31] enlarges the receptive field with fast Fourier convolutions. MAT [32] improves texture realism with long-range modelling. This generation of methods is text-free. It can only erase a region and continue the surrounding texture plausibly. Its edge scores come from that texture continuation, not from correct reconstruction of the target structure. On the difusion side, DDPM [14], DDIM [15] and guidance mechanisms [16, 33] established the conditional generation framework. Latent difusion [17] moved denoising into the VAE latent space. Several text-conditioned inpainting methods followed. Imagen Editor [34] conditions jointly on text and mask shape and proposes the EditBench benchmark. PowerPaint [35] unifies multiple inpainting modes with learnable task prompts. BrushNet [36] injects masked-image features through a parallel UNet branch. ASUKA [37] replaces text conditioning with a visual prior to suppress hallucinated insertion and colour inconsistency. FLUX.1 Fill [dev] [38] reaches a very strong level on general inpainting at 12B scale. All of them carry the conditioning in natural language and reference photographs. None ofers a precise geometric and chromatic constraint channel for standardised artefacts. Their evaluation targets general perceptual quality and skips task-critical questions such as whether the digit on the sign face is correct.

Conditional injection and reference-driven generation. ControlNet [18] injects spatially aligned conditions such as edge and depth maps. It duplicates the encoder of the difusion backbone and connects it with zero-initialised convolutions, and it is the de facto standard for spatially controllable generation. T2I-Adapter [19] reaches a similar goal with a lighter adapter. IP-Adapter [20] targets appearance conditions that are not spatially aligned. It injects CLIP [39] image embeddings into the UNet through decoupled cross-attention. This split is the design rationale of our framework. Geometric shape needs a strong spatially aligned constraint. Colour scheme and style need a soft non-spatial one. Reference-driven object insertion takes another line. Paint-by-Example [40] replaces masked content with a reference image. AnyDoor [41] performs zero-shot object teleportation with dual identity and detail pathways. These methods use a real photograph as the reference. We use a standard vector template instead, and two diferences follow. The template is a noise-free, illumination-free canonical representation, so it imports no illumination or background from a reference photograph. The template is also generated programmatically from a category label, so it covers zero-shot categories just as well. FLUX.1 Kontext [42] and Qwen-Image-Edit [43] represent the instructionediting paradigm. They edit with natural language and need no mask. The model itself decides the editing extent, so confining the edit strictly to the sign region is hard to guarantee.

Visual text generation. The digits on the sign face must be correct, which ties this task to visual text generation. TextDifuser [27] guides text rendering with character-level segmentation masks. It also established the convention of scoring generated text with OCR metrics. GlyphControl [29] injects glyph images as a spatial condition. AnyText [28] combines glyph, position and mask for multilingual text generation and editing. We share the idea of using glyphs or templates as a spatial condition. Our conditioning form difers in two respects. First, regulation enumerates a tiny character set for trafic signs, so general glyph-rendering capacity is not what we need. Second, the text on a sign and its carrier are inseparable. Text, colour and geometry therefore live in one and the same vector template, not in a separate text layer. We follow this field in adopting character accuracy and exact-match rate, with PaddleOCR [44] as the evaluator.

To the best of our knowledge, no prior work injects semantic, appearance and geometric priors explicitly and adds loss-level physical consistency constraints on trafic signs. Nor does any prior work trace a complete evidence chain from generation quality to downstream detection.

## 3 Method

## 3.1 Task Definition and Overall Framework

Formalisation. Given a road image $I \in \mathbb { R } ^ { H \times W \times 3 }$ , a binary mask $M \in \{ 0 , 1 \} ^ { H \times W }$ of the sign region, and a structured description $y = ( \mathrm { t y p e } , \mathrm { i n f o } )$ of the target sign (for example type = maximum speed limit, info = 100), the task is to generate an image <sup>ˆ</sup>� satisfying

$$
\hat { I } \odot ( 1 - M ) = I \odot ( 1 - M ) , \qquad \hat { I } \odot M \sim p ( \mathrm { s i g n } \mid y , C ) ,\tag{1}
$$

Pixels outside the mask stay unchanged. Pixels inside the mask follow the sign distribution, conditioned on the structured descrip tion � and a derived condition set C. We split the condition set into three orthogonal components,

$$
C = \{ c _ { \mathrm { s e m } } , \ c _ { \mathrm { a p p } } , \ c _ { \mathrm { g e o } } \} ,\tag{2}
$$

where $c _ { \mathrm { { s e m } } }$ is the JSON-formatted semantic prompt text, $c _ { \mathrm { a p p } }$ is the front-view vector template infused with the measured dominant colours, and $c _ { \mathrm { g e o } }$ is the vector template afine-aligned to the image pose. Each enters the denoising network by its own route. . Csem $c _ { \mathrm { { s e m } } }$ goes through cross-attention after the text encoder. $c _ { \mathrm { a p p } }$ goes through IP-Adapter decoupled cross-attention after the CLIP image encoder. $c _ { \mathrm { g e o } }$ goes through residual injection by the ControlNet branch.

Training paradigm. During training we extract all conditions automatically from the original sign region, and the reconstruction target is the original sign itself. No manually paired data is needed. Any road image with a sign annotation becomes a training sample.

Inference paradigm. At inference, changing the conditions changes the generated content. Replacing type/info in � edits category and digits, which is how we supplement long-tail categories and generate rare classes. Replacing the clustered colours edits hue and lightness. Replacing the geometric parameters edits rotation and scale. The three pathways are decoupled, so these edits apply on their own or together.

Overall framework. Figure 1 shows the three steps. The mask and condition construction module takes the original image and produces the repaint mask, the shape mask and the three conditions. The three conditions then drive SD1.5 Inpainting together with ControlNet to repaint the region. The repainted result is pasted back into the original-resolution image according to the crop box, ready for downstream detection training. Training and inference share the same pathway. They difer in two places only. Training draws the conditions from the original sign region and applies the two physical consistency losses to �ˆ<sub>0</sub> at low-noise steps. Inference rebuilds the conditions from the target category.

## 3.2 Data Construction Pipeline and Mask Strategy

Figure 2 shows the data construction pipeline. It has five sub-steps, all automated, with no manual annotation involved.

(1) Local cropping. We crop a square region from the original image, centred near the sign annotation box. Its side length is 3× the long side of the annotation box. The crop position is random, subject to the annotation box keeping a margin from the crop boundary. All crops are resampled to 512×512.

(2) SAM 2 segmentation. The segmentation model is SAM 2 [30], the second generation of SAM [45], prompted with the annotation box to obtain a pixel-level mask of the sign.

(3) Analytic geometric fitting. We fit per geometric family. For circular signs we fit an ellipse on the largest contour, which gives the centre, the two semi-axes and the tilt angle. For triangular signs we run probabilistic Hough line detection on the mask edge, then enumerate triples of lines and intersect them to obtain candidate triangles. We score each fit by its IoU with the SAM mask and discard the samples that fail.

(4) Dual mask construction. The repaint mask $M _ { \mathrm { { i n p a i n t } } }$ is a rectangle, obtained by dilating each of the four sides of the annotation box by 5 pixels. It is the inpainting input of the difusion model. The shape mask $M _ { \mathrm { s h a p e } }$ fills the analytic fitting result. It bounds the colour clustering statistics and the domain of the physical consistency losses.

(5) Lab dominant colour clustering. We run �-means in Lab colour space over the pixels inside the shape mask. � is the number of colours in the national-standard colour scheme of that category. The standard colour names must match the cluster centres one to one, otherwise we discard the sample. For speed-limit signs we keep the pixel coordinates of the cluster holding the digits. The tilt angle of their minimum bounding rectangle is the sign-face rotation angle �.

## 3.3 Structured Prior Injection Modules

JSON-formatted semantic prompt. Instead of free-form natural language descriptions, we express sign semantics with a fixed JSON structure:

There are three reasons behind structured text. First, it removes ambiguity and redundancy. In natural language “limit one hundred”, “100 km/h speed limit sign” and “maximum speed $1 0 0 ^ { \mathfrak { n } }$ carry the same meaning, yet the text encoder maps them to diferent embeddings. Second, it is enumerable. The Cartesian product of the category table and the value table lists every legal condition, so we pre-compute and cache all text embeddings ofline and read them at zero cost during training. Third, it is easy to edit. Changing one field at inference completes a category or digit edit, with no sentence to rewrite.

IP-Adapter appearance guidance. The appearance prior rides on a front-view vector template infused with the measured dominant colours. Figure 3 shows the construction. The category label indexes the national-standard vector graphic (SVG). The measured dominant colours replace the standard colour definitions in that

![](images/837a8db18316f825726d867d67c5aa981b122165f5136363c0f56accde3895b0.jpg)  
Figure 1: Overview of the framework. The semantic (JSON prompt), appearance (IP-Adapter) and geometric (ControlNet) conditions are injected into the SD1.5 inpainting UNet.

![](images/47859c701a8a1c63092ca50a2111e6a283e58274103ab7dde5474a3ad81ff01a.jpg)  
Figure 2: Data construction pipeline: local cropping, SAM 2 segmentation, analytic geometric fitting, dual mask construction and Lab dominant colour clustering, all fully automatic.

SVG. The result is rasterised into a 512×512 reference image. Measured colour schemes for one category difer markedly across illumination conditions, so the reference image follows the scene instead of being a fixed per-category template.

The CLIP image encoder [39] turns this reference image into an image embedding, and the decoupled cross-attention of IP-Adapter [20] injects it into the UNet. Each UNet layer gains an image pathway alongside the text pathway. The two share the query and keep independent keys and values,

$$
\mathrm { A t t n } ( Q , K , V ) + \lambda _ { \mathrm { i p } } \mathrm { A t t n } ( Q , K _ { \mathrm { a p p } } , V _ { \mathrm { a p p } } ) ,\tag{3}
$$

where $K _ { \mathrm { a p p } } , V _ { \mathrm { a p p } }$ are obtained from the CLIP image embedding of the reference image $c _ { \mathrm { a p p } }$ through independent projection matrices, and $\lambda _ { \mathrm { i p } }$ is the IP-Adapter scale in the inference configuration. We choose IP-Adapter over a spatial conditioning pathway because appearance is a global attribute. Colour scheme, style and material feel are not aligned to pixel positions and should not be bound to them.

ControlNet geometric control. The geometric prior rides on the vector template, afine-transformed into alignment with the image pose. Figure 4 shows guidance images for one category under a near-front-view pose and a strongly perspective pose. Both aligned results fit their pose precisely. The transformed template image is the conditional input of ControlNet. It enters the features of UNet layer � as

$$
h _ { l } ^ { \prime } = h _ { l } + Z _ { l } \big ( \mathcal { F } _ { \mathrm { c n } } ( c _ { \mathrm { g e o } } , z _ { t } , t , c _ { \mathrm { s e m } } ) \big ) ,\tag{4}
$$

where $\mathcal { F } _ { \mathrm { c n } }$ is the ControlNet branch and $Z _ { l }$ is a zero-initialised convolution whose scaling factor is the ControlNet conditioning scale in the inference configuration. One choice here departs from common usage. We feed the colour vector template image directly as the condition, not its edge map. The ControlNet branch starts from a Canny-edge ControlNet and is then fine-tuned end-to-end with the UNet to adapt to this new conditioning modality. In practice this works better than feeding edge maps. The template image carries edge positions, region filling relations and the colour scheme at once, so it holds strictly more information than an edge map. Endto-end fine-tuning is enough for the branch to exploit that extra information.

## 3.4 Physical Consistency Loss Design

Base difusion loss. We adopt the �-prediction MSE loss with min-SNR-� weighting [46]:

$$
\mathcal { L } _ { \mathrm { b a s e } } = \mathbb { E } _ { t , \epsilon } \left[ \frac { \operatorname* { m i n } ( \mathrm { S N R } ( t ) , \gamma ) } { \mathrm { S N R } ( t ) } \left. \epsilon _ { \theta } ( \tilde { z } _ { t } , t , C ) - \epsilon \right. ^ { 2 } \right] , \quad \gamma = 5 ,\tag{5}
$$

where $z _ { t }$ is the 4-channel noisy latent at timestep $t . \ \tilde { z } _ { t }$ is the $9 \mathrm { - }$ channel UNet input, formed by concatenating $z _ { t }$ with the downsampled mask and the masked-image latent. The ControlNet branch and the inversion to $\hat { x } _ { 0 }$ below both operate on $z _ { t }$ . Min-SNR weighting caps the oversized gradient weights at low-noise steps and balances the efective learning rate across timesteps. It is known to speed up convergence and improve detail quality in inpainting fine-tuning.

Lab chromaticity consistency loss. The colour constraint is imposed on the chromaticity channels of CIELAB space. Let $\hat { x } _ { 0 }$ be the predicted image obtained by inverting the model-predicted noise into a clean latent and decoding it with the VAE, � the groundtruth image, $M _ { \mathrm { s h a p e } }$ the shape mask, and $\operatorname { L a b } ( \cdot ) _ { a b }$ the operation taking the �, � chromaticity channels of Lab. Then

$$
{ \mathcal { L } } _ { \mathrm { l a b } } = { \frac { \sum \left| \mathrm { L a b } ( { \hat { x } } _ { 0 } ) _ { a b } - \mathrm { L a b } ( x ) _ { a b } \right| \odot M _ { \mathrm { s h a p e } } } { \sum M _ { \mathrm { s h a p e } } + \varepsilon } } .\tag{6}
$$

Dropping the lightness channel � is deliberate. Lightness carries illumination, shadow and highlight. Much of its per-pixel diference is a legitimate variation of scene illumination, and constraining it would push the model to overfit the illumination of the training samples. Chromaticity (�, �) stays closer to the intrinsic colour of the object itself, and that is what the sign standard prescribes. The split also matches the original intent of Lab space, which separates lightness from chromaticity.

Sobel structural consistency loss. The structural constraint is imposed on the lightness channel of Lab as the $L _ { 1 }$ distance between Sobel gradient magnitudes:

$$
\mathcal { L } _ { \mathrm { s o b e l } } = \frac { \sum \big | \operatorname { S o b e l } \bigl ( \mathrm { L a b } ( \hat { x } _ { 0 } ) _ { L } \bigr ) - \operatorname { S o b e l } \bigl ( \mathrm { L a b } ( x ) _ { L } \bigr ) \big | \odot M _ { \mathrm { s h a p e } } } { \sum M _ { \mathrm { s h a p e } } + \varepsilon } .\tag{7}
$$

The two losses are complementary. The colour constraint uses chromaticity, the structural constraint uses lightness. They occupy diferent channels and do not interfere.

The total loss is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { b a s e } } + \lambda _ { \mathrm { l a b } } \mathcal { L } _ { \mathrm { l a b } } + \lambda _ { \mathrm { s o b e l } } \mathcal { L } _ { \mathrm { s o b e l } } .\tag{8}
$$

## 3.5 Training Strategy and Implementation Details

Backbone and trainable modules. The backbone is the SD1.5 inpainting checkpoint (stable-diffusion-v1-5/stablediffusion-inpainting, a UNet with 9-channel input). The ControlNet branch starts from lllyasviel/sd-controlnetcanny. The trainable parameters are all weights of the UNet and of ControlNet. The VAE, the CLIP text encoder and the CLIP image encoder stay frozen throughout. IP-Adapter uses the SD1.5 version of h94/IP-Adapter, and we fine-tune its image projection layer and decoupled cross-attention weights together with the UNet.

Optimisation configuration. 8-bit AdamW [47, 48] $( \beta ~ = ~ ( 0 . 9 , 0 . 9 9 9 )$ , weight decay $1 0 ^ { - 2 } )$ , constant learning rate $1 \times 1 0 ^ { - 5 }$ , gradient clipping threshold 1.0, fp16 mixed precision (the two-loss path is forced to fp32), 4×V100 data parallelism, efective batch size 16, and 50,000 training steps (about 800k samples). All ablation variants use exactly the same protocol and number of steps, changing only the module being ablated.

Loss weights. The weights are calibrated by magnitude matching. At low-noise steps the base term is about 0.01–0.05 while the two auxiliary losses are about 2.7 and 1.1. A weight of $3 \times 1 0 ^ { - 3 }$ therefore puts their contribution on the same order of magnitude as the base term without overwhelming it. The two physical losses are evaluated only at low-noise timesteps (� < 200 out of 1000 total steps). There the $\hat { x } _ { 0 }$ inverted from $\epsilon _ { \theta }$ has low variance, so the pixel-level constraints act on a meaningful signal. This also caps the extra VAE decoding cost at roughly 20% of the samples.

Inference configuration. Sampler UniPC [49] with 20 steps; classifier-free guidance scale 3.0; IP-Adapter scale 1.0; ControlNet conditioning scale 0.75; resolution 512×512; random seed fixed to 42. The hyper-parameter sensitivity analysis in Section 4.2 selected this configuration on the Pareto frontier. It sits at a balanced position with no obvious weakness.

Convergence. The training loss enters a plateau after 10,000 steps. The interval mean of the total loss falls from 0.0173 over steps 10,000–15,000 to 0.0163 over steps 45,000–50,000, a 5.8% reduction. Over the same period the Lab and Sobel terms fall by 3.1% and 2.9%. Training for 50,000 steps has thus entered the convergence regime, and further steps buy little.

![](images/3d181b9e3b5bdca918b17786555fc2ad1ea546771f15ed5de2dcdbf8068bb2a8.jpg)  
Standard Template  
Crop → Colour-injected Reference  
IP-Adapter Injection

Figure 3: Construction of the IP-Adapter appearance prior. Left: the national-standard vector templates. Middle: the local crops and the reference images after infusing the measured dominant colours. Right: the path by which the reference image is injected into the UNet.  
![](images/422a3f6a97e245807ddcafee3f64b3c1529856716cdc69061af04d53f75895ee.jpg)  
Crop → Affine-aligned Guide  
ControlNet Injection

Figure 4: Construction of the ControlNet geometric prior. Left: the local crops and the colour template guidance images afine-aligned according to the analytic geometric parameters. Right: the path by which the ControlNet branch is added to the UNet.

## 4 Experiments

## 4.1 Datasets, Evaluation Protocol and Implementation

Training set. The training data is road images collected in-house, paired with their sign annotations (category + annotation box + digits). There are 1,673,112 pairs. They cover 41 templates: 19 triangular warning classes and 22 circular speed-limit classes with diferent digits.

Table 1: The seven competing models and the role each plays in the comparison.
<table><tr><td>Model (venue)</td><td>Role in our argument</td></tr><tr><td>LaMa (WACV&#x27;22) [31]</td><td>Non-diffusion (FFC-GAN) genera- tional anchor</td></tr><tr><td>MAT (CVPR’22) [32]</td><td>Non-diffusion (Transformer) large- hole anchor</td></tr><tr><td>SD1.5 Inpainting (2022) [17]</td><td>Bare lower bound of our own back- bone</td></tr><tr><td>PowerPaint v2 (ECCV&#x27;24) [35]</td><td> Task-prompt style multi-task in-</td></tr><tr><td>BrushNet (ECCV&#x27;24) [36]</td><td>painting Academic benchmark of the controllable-injection school</td></tr><tr><td>ASUKA-SD1.5 (CVPR’25) [37]</td><td>Fairest rival: same backbone, colour-</td></tr><tr><td>FLUX.1 Fill [dev] (2024) [38]</td><td>consistency theme 12B industrial SOTA, parameter- count upper bound</td></tr></table>

Evaluation set. We evaluate on TT100K-2021. Its collection source difers entirely from that of the training data. We read all TT100K annotations and keep the targets in the 41 supported categories, which gives 7,898 candidates. These pass three automatic gates: SAM 2 segmentation, analytic geometric fitting and one to-one colour-scheme matching. 5,182 sign regions remain as the evaluation set, and 4,601 of them are speed-limit classes bearing digits. This dataset is also heavily long-tailed and dense in small objects. The 9,050 images left after excluding the other split hold 25,010 target instances across 184 categories. 83 of those categories have a single-digit number of instances, and 46.3% of targets have a short side below 32 pixels.

Evaluation metrics. Metrics are chosen along three dimensions. Reconstruction fidelity: PSNR/SSIM [50], LPIPS [51] and FID [52] (Inception-v3 [53] features). Physical consistency: CIEDE2000 colour diference [54] and in-mask IoU/F1 on Canny [55] edge maps. CIEDE2000 is a perceptually uniform industrial standard with a discrimination threshold of ΔE≈5, and it beats a simple RGB distance. Semantic controllability: OCR exact-match rate (EM), character accuracy (CharAcc) and normalised edit distance (NED), with PaddleOCR [44] as the evaluator.

Competing models. There are seven, all run zero-shot with oficial pretrained weights. Table 1 lists the role each plays in our argument.

Two instruction-editing methods, FLUX.1 Kontext [42] and Qwen-Image-Edit [43], stay out of the quantitative comparison. They take no mask input, the model itself decides the editing extent, and they cannot guarantee that pixels outside the mask remain unchanged, which data augmentation requires.

Fairness protocol. (i) The same 5,182 (image, mask) inputs; (ii) the same natural-language prompt transcribed from one set of JSON annotations (except LaMa/MAT, which have no text interface); (iii) fixed random seed 42; (iv) each method uses its oficially recommended inference configuration; (v) all metrics are computed only inside the mask region, by one and the same evaluation code.

One diference deserves note. Our method is domain fine-tuned, the competing models are zero-shot. The same-backbone ablation in Section 4.3 covers this diference.

## 4.2 Generation Quality Evaluation

(1) First place on all nine metrics. As Table 2 shows, our method leads on reconstruction fidelity, physical consistency and semantic controllability at once. On the semantic dimension the lead is of a diferent order (OCR-EM 91.1% vs. 44.2%, +46.9 pt).

(2) The three classes of competitors fail in diferent ways, all pointing to a deficiency in the conditioning form. The non-difusion methods (LaMa/MAT) do respectably on PSNR/SSIM/EdgeIoU, yet their OCR exact-match rate is near zero. They erase the sign and continue the background texture plausibly, and their edge scores come from that continuity rather than from correct reconstruction of the sign structure. The general text-conditioned difusion methods (SD1.5/PowerPaint/BrushNet) do generate sign-like content, with OCR-EM of 13.9%–23.2%. But their colour diference reaches 22–26 and their PSNR falls below the non-difusion methods, because the generated result does not match the illumination and geometry of the scene. ASUKA is the most instructive case. Its LPIPS (0.080) and ΔE (17.12) are the best in the SD1.5 family and second only to the 12B FLUX.1 Fill, so its colour-consistency design works. Its OCR-EM is nonetheless 0.8%. The MAE visual prior suppresses hallucinated object insertion, and it costs the model almost all of its text generation ability.

(3) A 1B-scale task-specialised model surpasses a 12B general-purpose model. FLUX.1 Fill [dev] is the strongest competitor (FID 9.05, OCR-EM 44.2%). Its 12B parameters give it far better general generation ability than the SD1.5-family baselines. Our method still passes it on all nine metrics with about 1/9 of the parameters: +46.9 percentage points on OCR-EM and 53% lower FID. Our single-sample inference time on one V100 is 8.81 s, only 1/14 of the 123.55 s of FLUX.1 Fill. The judgement is clear. On vertical tasks with standardised semantics and geometry, feeding the prior to the model in structured form beats scaling the model up and letting it guess.

(4) OCR confidence separation. OCR confidence separates sharply. Correct samples average 0.988, incorrect ones 0.328. This gives a reference-free signal for screening generation quality.

Cell-by-cell comparison with the baselines. Figure 5 places our method beside the seven competing models over ten representative scenes, and the trend is clear. The non-difusion methods (LaMa/MAT) smooth the sign over with background texture and lose the sign-face semantics almost entirely. The general textconditioned difusion methods (SD1.5/PowerPaint/BrushNet) produce sign-like content, but often distort the sign-face geometry and the digit glyphs. FLUX.1 Fill, the strongest baseline, keeps glyphs far more intact. It still skews the structure and sticks characters together in strongly perspective and three-digit scenes. Our method holds the sign outline, the perspective pose and the digit glyphs in all ten rows.

Reconstruction fidelity against the ground truth. Figure 6 leaves the baselines aside and examines our reconstruction fidelity against the ground truth alone. Each tile gives our reconstruction with the original crop thumbnail overlaid in a free corner as the reference, and a red box marks the same repaint region on both. Reading the pairs cell by cell, the sign-face colour scheme, outline and digit glyphs all agree closely with the ground truth. Nighttime low-light and distant small-scale samples show no obvious degradation either.

Table 2: Zero-shot comparison against the seven competing models on the 5,182-region TT100K evaluation set. Best value per column in bold.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td><td>∆E↓</td><td>EdgeIoU↑</td><td>EdgeF1↑</td><td>OCR-EM↑</td><td>NED↑</td></tr><tr><td>LaMa (WACV&#x27;22) [31]</td><td>13.59</td><td>0.508</td><td>0.112</td><td>31.70</td><td>17.40</td><td>0.086</td><td>0.127</td><td>0.2%</td><td>0.021</td></tr><tr><td>MAT (CVPR’22) [32]</td><td>13.03</td><td>0.459</td><td>0.109</td><td>29.48</td><td>18.29</td><td>0.070</td><td>0.107</td><td>0.0%</td><td>0.019</td></tr><tr><td>SD1.5 Inpainting [17]</td><td>10.51</td><td>0.346</td><td>0.123</td><td>21.25</td><td>23.36</td><td>0.051</td><td>0.095</td><td>20.7%</td><td>0.425</td></tr><tr><td>PowerPaint v2 (ECCV&#x27;24) [35]</td><td>9.52</td><td>0.292</td><td>0.136</td><td>24.35</td><td>26.11</td><td>0.046</td><td>0.086</td><td>23.2%</td><td>0.455</td></tr><tr><td>BrushNet (ECCV&#x27;24) [36]</td><td>11.16</td><td>0.361</td><td>0.125</td><td>19.06</td><td>22.05</td><td>0.048</td><td>0.090</td><td>13.9%</td><td>0.315</td></tr><tr><td>ASUKA-SD1.5 (CVPR’25) [37]</td><td>13.54</td><td>0.437</td><td>0.080</td><td>18.73</td><td>17.12</td><td>0.085</td><td>0.138</td><td>0.8%</td><td>0.074</td></tr><tr><td>FLUX.1 Fill [dev] (12B) [38]</td><td>13.90</td><td>0.435</td><td>0.074</td><td>9.05</td><td>16.79</td><td>0.070</td><td>0.123</td><td>44.2%</td><td>0.650</td></tr><tr><td>Ours (SD1.5, ~1.4B)</td><td>15.53</td><td>0.534</td><td>0.039</td><td>4.22</td><td>12.40</td><td>0.087</td><td>0.144</td><td>91.1%</td><td>0.927</td></tr></table>

Independent controllability of the condition pathways. Here we test the editability of the condition pathways. We keep the scene and mask fixed, change one condition component at a time, and check whether the result changes as intended while the other attributes hold. Figure 7 shows sequences for six kinds of edits: changing the speed-limit value, changing the warning sign category, changing the sign-face colour scheme, and continuous interpolation over lightness, geometric rotation and scale. In every row the target attribute varies monotonically and smoothly with the condition. The background pixels and the unedited attributes stay stable, with no cross-talk between conditions. The text, appearance and geometry pathways are thus decoupled and individually efective, which is exactly what a controllable data augmentation engine needs.

Sensitivity of inference hyper-parameters. We selected the sampler and the three guidance-strength parameters (gs/ip/cn) on a 1000-sample subset with 20 steps. The sampler came first, chosen among UniPC, DPM++ 2M [56] and PNDM. UniPC leads stably by 1.1–2.3 pt on OCR-EM, and the LPIPS/FID diferences among the three stay below 2%, so we adopt it throughout. The guidance parameter grid covers $\mathrm { g s } \in \{ 2 . 0 , \ldots , 6 . 0 \} \times \mathrm { i p } \in \{ 0 . 8 , 1 . 0 \}$ × cn ∈ {0.5, 0.75, 1.0}. The Pareto frontier holds 11 non-dominated solutions, and Table 3 lists the five core ones.

Three points follow. The gs knee lies on the 2.5–3.0 plateau. Below it the text condition is too weak, above it the guidance is too strong. cn is the core trade-of axis. Moving from 0.5 to 1.0 buys a structural gain of SSIM +0.027 / EdgeF1 +0.007 at the price of a slight OCR loss. ip has a mild influence, yet all 11 non-dominated solutions sit at ip = 1.0. The optimum of the inference hyper-parameters depends on the training configuration and does not carry over across configurations. The main table therefore uses the middle of the Pareto frontier: gs 3.0 / ip 1.0 / cn 0.75 / UniPC / 20 steps.

## 4.3 Ablation Study

Variant design. All ablations run on the SD1.5 backbone with a strict leave-one-out single-variable design. Taking Full as the reference, we remove exactly one prior module or loss term at a time, which gives 5 ablation variants plus Full. Table 4 lists the configurations.

One clarification is necessary. The measured colours are injected into the vector template, and both the IP-Adapter reference image and the ControlNet guidance image derive from that template, so “colour injection” is a switch that spans two condition pathways. V1 removes only the IP-Adapter branch, and the ControlNet guidance image still carries the measured colours. V2 keeps both pathways but uses the original template colours throughout. V6 vs. V1 therefore isolates the contribution of the appearance/layout condition pathway, and V6 vs. V2 isolates the contribution of measured colour injection. The two do not overlap.

The following conclusions can be drawn from Table 5.

(1) The IP-Adapter appearance prior is the decisive factor for digit legibility and distributional realism. Removing it drops OCR-EM from 91.1% to 66.1%, pushes FID from 4.22 to 16.94, and worsens LPIPS by 51%. The vector template supplies the colour scheme and the complete layout structure of the sign face: the glyph shape and size of the digits, their relative position within the face, and the white space. Text conditioning cannot convey any of that.

(2) Colour injection is the decisive factor for distributional realism. V2 keeps IP-Adapter but switches to the original template colours. FID rises from 4.22 to 9.65, LPIPS worsens by 38% and OCR-EM falls by 6.0 pt. The idealised saturated template colours sit too far from real imaging colours. The model has to bridge the gap itself, and its modelling of the other attributes sufers.

(3) The ControlNet geometric prior buys structural and fidelity gains at a slight OCR/FID cost. Removing ControlNet raises OCR-EM by 1.5 pt and lowers FID by 0.70. EdgeIoU drops by 17%, EdgeF1 by 10% and SSIM by 0.021. The trade-of is plain. A strong spatial constraint limits the model’s freedom inside the sign face, and writing the digits correctly needs some freedom. We keep ControlNet. Geometric correctness matters for downstream detection, for example the consistency of box shape and pose, and OCR cannot stand in for it.

(4) The core benefit of the two physical losses is text accuracy. Relative to V3 (no physical losses), OCR-EM is +2.2 pt while

![](images/b22c3e38cdc2f3382666540fcd058a5e80eb7a5312493e0448f5bd07fdc76d23.jpg)  
Figure 5: Qualitative comparison over ten representative scenes (rows). Each row shows the input followed by the seven competing models and our result.

Table 3: The five core groups among the 11 non-dominated configurations on the Pareto frontier of the inference hyperparameter grid. The first row is the adopted configuration; best value per column in bold.
<table><tr><td>Configuration</td><td>PSNR↑</td><td>SSIM↑</td><td>∆E↓</td><td>EdgeIoU↑</td><td>EdgeF1↑</td><td>OCR-EM↑</td><td>Positioning</td></tr><tr><td>gs3.0·ip1.0·cn0.75</td><td>15.66</td><td>0.541</td><td>12.06</td><td>0.081</td><td>0.138</td><td>90.0%</td><td>Balanced, no weakness</td></tr><tr><td>gs2.5·ip1.0·cn0.5</td><td>15.74</td><td>0.525</td><td>11.98</td><td>0.077</td><td>0.132</td><td>90.6%</td><td>Triple crown: OCR / fidelity / colour</td></tr><tr><td>gs3.0·ip1.0·cn0.5</td><td>15.69</td><td>0.525</td><td>12.07</td><td>0.079</td><td>0.135</td><td>90.6%</td><td>OCR peak</td></tr><tr><td>gs2.5·ip1.0·cn1.0</td><td>15.53</td><td>0.552</td><td>12.17</td><td>0.085</td><td>0.143</td><td>89.4%</td><td>SSIM champion</td></tr><tr><td>gs4.0·ip1.0·cn1.0</td><td>15.29</td><td>0.549</td><td>12.56</td><td>0.086</td><td>0.145</td><td>90.2%</td><td>Structure first, OCR retained</td></tr></table>

![](images/45719501681d6ae26c4518be6c04f61978d62fb631e2a8f4c5fb4aac7e1d4e3d.jpg)  
Figure 6: Reconstruction quality wall grouped by sign family. Each tile is our reconstruction with the ground-truth crop inset; the red box marks the repaint region in both.

Table 4: Leave-one-out ablation variants. ✓: module kept; ✗: module removed.
<table><tr><td>Variant</td><td>SOonN Pmpt</td><td>I-dad-ter</td><td>Colr n.</td><td>ConoNet</td><td>Iabb ss</td><td>Sobl os</td></tr><tr><td colspan="2">V1 Full – IP-Adapter V2 Full – colour injection</td><td>√ X √</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">V3 Full – both phys. losses</td><td>√ √</td><td>√</td><td>X</td><td>√</td><td>√</td><td></td></tr><tr><td colspan="2"></td><td>√</td><td></td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td colspan="2">V4 Full – Sobel Loss</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td></tr><tr><td colspan="2">V5 Full – ControlNet</td><td></td><td>√</td><td>√</td><td>X</td><td>√</td><td></td></tr><tr><td colspan="2">V6 Full (Ours)</td><td></td><td></td><td>L</td><td>√</td><td></td><td></td></tr></table>

ΔE/FID/LPIPS are essentially unchanged; the cost is SSIM −0.010 and EdgeF1 −4.6%.

(5) The independent contribution of the Sobel structural loss is a structural gain at no distributional cost. Relative to V4 (Lab present, Sobel absent): EdgeIoU +7.4%, EdgeF1 +5.1%, SSIM +0.017, ΔE −0.73 and OCR-EM +2.2 pt, while FID is essentially unchanged. The Sobel loss is thus an almost free gain term.

(6) The cost of the two losses on purely structural metrics. V3 (no physical losses) keeps a slight advantage over Full on SSIM (+1.9%), EdgeF1 (+4.9%), EdgeIoU (+3.4%) and FID (3.96 vs. 4.22). The benefit of the two losses concentrates in colour fidelity (the ΔE −0.73 isolated by point (5)) and text accuracy (+2.2 pt). The cost on purely structural metrics is real. Point (5) already showed by independent leave-one-out evidence that the Sobel term brings a structural gain, so the cost here belongs to the indirect efect of the Lab chromaticity term on lightness structure. Paying 0.010 of SSIM for 2.2 pt of OCR is a reasonable deal: for data augmentation, a sample with a wrong digit is a harmful sample, while a sample with 0.010 lower SSIM is still a usable one.

(7) The failure of PSNR and the “conservative generation” trap. V1 and V2 reach a PSNR of 16.44 and 16.47, well above the

![](images/8f4b751828278d3f0681361633376fb9fe7d547e901134d4ab0c157c6ec97fff.jpg)  
Figure 7: Controllability grid. Leftmost column: the input (red box: the repaint region). Each row edits one condition only: speed-limit value, warning category, colour scheme, lightness, rotation, scale.

Table 5: Leave-one-out ablation results, V6 being the full model. Best value per column in bold.
<table><tr><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td><td>∆E↓</td><td>EdgeIoU↑</td><td>EdgeF1↑</td><td>OCR-EM↑</td><td>CharAcc↑</td><td>NED↑</td></tr><tr><td>V6 Full (Ours)</td><td>15.53</td><td>0.534</td><td>0.039</td><td>4.22</td><td>12.40</td><td>0.087</td><td>0.144</td><td>91.1%</td><td>92.7%</td><td>0.927</td></tr><tr><td>V1 – IP-Adapter</td><td>16.44</td><td>0.509</td><td>0.059</td><td>16.94</td><td>11.98</td><td>0.078</td><td>0.140</td><td>66.1%</td><td>69.6%</td><td>0.697</td></tr><tr><td>V2 – colour injection</td><td>16.47</td><td>0.516</td><td>0.054</td><td>9.65</td><td>11.65</td><td>0.079</td><td>0.142</td><td>85.1%</td><td>87.9%</td><td>0.880</td></tr><tr><td>V3 – both phys. losses</td><td>15.45</td><td>0.544</td><td>0.038</td><td>3.96</td><td>12.44</td><td>0.090</td><td>0.151</td><td>88.9%</td><td>90.7%</td><td>0.907</td></tr><tr><td>V4 – Sobel Loss</td><td>15.01</td><td>0.517</td><td>0.042</td><td>4.27</td><td>13.13</td><td>0.081</td><td>0.137</td><td>88.9%</td><td>90.5%</td><td>0.905</td></tr><tr><td>V5 – ControlNet</td><td>15.54</td><td>0.513</td><td>0.040</td><td>3.52</td><td>12.06</td><td>0.072</td><td>0.130</td><td>92.6%</td><td>93.7%</td><td>0.937</td></tr></table>

15.53 of Full. Their FID is 4.0× and 2.3× that of Full, and their OCR-EM is 25.0 pt and 6.0 pt lower. Without the appearance/colour prior the model generates conservative content with low contrast, close to the mean of the mask surroundings. Such content wins on perpixel squared error and fails completely in perceptual and semantic terms. The methodological conclusion is clear. On semantically critical tasks such as sign repainting, PSNR alone cannot evaluate generation quality. It must be read together with FID/LPIPS/OCR and the physical consistency metrics.

Full wins no single column in Table 5. It stays balanced and near the front on the key dimensions: third on FID, second on OCR behind V5, second on the Edge metrics behind V3. That is the reasonable landing point of a multi-objective trade-of.

## 4.4 Downstream Detection Verification

Detector and data strategies. The detector is YOLO26s-p2 [5] with a P2 detection head (stride 4) for the small objects of this task. It is only a fixed measuring instrument here. All 16 runs share an identical training protocol, item by item. Horizontal flipping is disabled: the category set contains three pairs of left–right mirrored classes, and flipping would manufacture exact mislabels. The four data strategies are: (i) real data only (baseline); (ii) real + Copy-Paste hard paste; (iii) real + Copy-Paste Poisson blending [12]; (iv) real + our difusion synthesis.

Category protocol and main metric. The 34 classes are 14 Tier-1 classes, 19 Tier-2 classes and an other\_sign catch-all. The capabilities the difusion model already has determined this set in reverse. Data come from the oficial TT100K-2021 train split, divided into train/val, and we evaluate on test exactly once. The other split is excluded entirely because its annotation is incomplete: missing labels would count as false positives and systematically depress AP. The main metric is Tier-2 in-group pooled AP50. We merge the 19 classes into one virtual class and rematch, which gives 84 test instances. A single class holds only 1–13 test instances, so per-class AP means nothing statistically. Tier-2 macro AP50, the per-class mean, supplements it and is more sensitive to the hardest classes.

Overall conclusion. As the left panel ofTable 6 shows, difusion is best on the main metric at the 20%/50%/100% fractions. It improves over the real-data-only baseline by 1.23×–7.40×, and its lead over the best Copy-Paste peaks at the 20% fraction (+0.096, 1.53×). Under the Tier-2 macro AP50 criterion it is best at all four fractions (right panel). On the 14 head Tier-1 classes at the 100% fraction the four groups reach 0.966 / 0.954 / 0.970 / 0.963, a spread within ±0.008. The synthetic data does not harm the already saturated head classes. The trend under mAP50-95 is the same. The last 8 epochs of val flatten out in every run, so under-training does not explain the efect.

Three advantages over the strongest Copy-Paste baseline. First, it works better on the hardest rare classes. Macro AP50 is best at all four fractions (1.08–1.24×), and the absolute lead at the 100% fraction reaches +0.122 against only +0.022 on pooled AP at the same fraction. Macro weights every rare class equally, while pooling is dominated by instance-rich classes. The gap between the two says the benefit concentrates on the hardest classes. Second, it is best at the medium scale across all four fractions, reaching 0.942 / 0.667 / 0.386 / 0.208 at the 100%/50%/20%/10% fractions. Third, direct template pasting is not enough on complex symbol classes. Difusion is markedly better on the four classes w34/w42/w24/w18 (0.497–0.995 vs. 0.000–0.507), and its mean AP50 over the 12 target classes is also the highest (0.770 vs. 0.606 for hard paste and 0.481 for Poisson). A pasted patch is flat and adapts to no illumination, so its deviation from real imaging grows with symbol complexity. Difusion-generated symbols instead get harmonised illumination, blur and noise.

Two boundaries of applicability. First, the pooled-AP advantage does not widen monotonically as data shrinks. It peaks at the 20% fraction. At the 10% fraction the three strategies converge under this criterion: difusion 0.107 vs. CP hard paste 0.111, a 0.003 gap below the resolution that only 84 test instances aford. The training data is simply too scarce there, and all three strategies sit close to the floor of this criterion. Even at that fraction, difusion is still best on macro AP and at the small/medium scales. Second, small-object AP is not consistently superior. Difusion is best at the 100%/10% fractions, CP at the 50%/20% fractions. Neither CP variant is consistently stronger than the other: hard paste wins at 100%/50%, Poisson at 20%, and they tie at 10%. The comparisons above always take the stronger of the two per fraction.

![](images/e3c0e2abd604eebaefd774d4c95f6808da238ad5cc78648ad1ea8f0f3b9a9508.jpg)  
Figure 8: Failure case grid. Columns: input, local crop (red box: the repaint region), ControlNet guidance image, and our result. Rows: pseudo-text generation, geometry–semantics conflict, and failure of colour clustering.

Difusion thus brings a 1.23×–7.40× pooled AP50 improvement over the real-data-only baseline. Copy-Paste reaches most of that improvement too. The case for difusion rests on the three advantages above, and the two boundaries above bound it.

## 5 Discussion and Conclusion

## 5.1 Failure Case Analysis

The OCR confidence distribution separates cleanly, so we screen out the low-confidence samples and summarise three typical failure modes (Figure 8).

(1) Pseudo-text generation. Real roads carry signs that do not fully comply with the specification, for example a speed-limit sign with a line of small annotation text beneath the digits. Once such samples enter the training set, the model picks up the mistaken prior that a sign face may carry additional text. It then adds pseudo-text of its own accord when generating a compliant sign face.

(2) Geometry–semantics tension from conflicting conditions. Sometimes the geometry of the ControlNet guidance image cannot host the layout the target digits require, for instance a threedigit value on an extremely flattened elliptical mask. The geometric constraint and the text generation then squeeze each other. The digits come out flattened and deformed, sometimes with one digit dropped altogether.

(3) Failure of colour clustering. A sign may be half covered by shadow, or severely faded or soiled. The dominant colour from

Table 6: Downstream detection results on the Tier-2 rare classes at four real-data fractions. Left: in-group pooled AP50 (main metric). Right: macro AP50. Diferences and ratios are computed from the unrounded values.
<table><tr><td></td><td colspan="6">In-group pooled AP50 (main metric)</td><td colspan="5">Macro AP50</td></tr><tr><td>Fraction</td><td>(i) Real</td><td>(ii) CP hard</td><td>(iii) CP Poisson</td><td>(iv) Diff.</td><td>(iv)−best CP</td><td>(iv)/(i)</td><td>(i) Real</td><td>(ii) CP hard</td><td>(iii) CP Poisson</td><td>(iv) Diff.</td><td>(iv)-best CP</td></tr><tr><td>10%</td><td>0.0145</td><td>0.1108</td><td>0.1098</td><td>0.1073</td><td>-0.003 (0.97×)</td><td>7.40×</td><td>0.007</td><td>0.042</td><td>0.029</td><td>0.045</td><td>+0.003 (1.08×)</td></tr><tr><td>20%</td><td>0.0578</td><td>0.1528</td><td>0.1817</td><td>0.2775</td><td>+0.096 (1.53×)</td><td>4.80×</td><td>0.047</td><td>0.140</td><td>0.109</td><td>0.173</td><td>+0.034 (1.24×)</td></tr><tr><td>50%</td><td>0.1442</td><td>0.4390</td><td>0.3418</td><td>0.4738</td><td>+0.035 (1.08×)</td><td>3.29×</td><td>0.116</td><td>0.321</td><td>0.276</td><td>0.393</td><td>+0.071 (1.22×)</td></tr><tr><td>100%</td><td>0.6722</td><td>0.8076</td><td>0.7923</td><td>0.8291</td><td>+0.022 (1.03×)</td><td>1.23×</td><td>0.483</td><td>0.702</td><td>0.655</td><td>0.825</td><td>+0.122 (1.17×)</td></tr></table>

K-Means clustering then departs from the colour scheme prescribed for that class, and the injected reference colour guides the model towards a sign with a global colour cast.

## 5.2 Diversity and Potential Bias of the Synthetic Data

Sources and ceiling of diversity. The diversity of our synthetic data comes from three levels: the scene level (background, illumination, occlusion and weather supplied by the original photograph), the pose level (imaged scale, tilt angle and perspective supplied by the source instance), and the content level (re-specification of the target category and digits). The first two levels are inherited entirely from real data. The number and the distribution of the source instances therefore set the ceiling of diversity.

Three identified potential biases. (1) A deliberate shift of the scale distribution. Small objects make up 42% ofthe generated set, above the 24–34% ofthe natural distribution. We introduced this on purpose to address the near-zero AP of small-object rare classes under a real-data-only baseline. It does leave the scale distribution of the synthetic data inconsistent with that of the test set. (2) An inherited bias in edit location. Instances of a target class can only appear at locations where a source sign once appeared. Some classes of sign follow systematic placement regularities in the real world, merge signs at ramp entrances for instance. If the source instances come from the locations of other classes, the synthetic data introduces an erroneous “location–category” prior. (3) Risk of distribution shift induced by the quality gate. Strict OCR filtering biases systematically towards large-scale, high-contrast samples. We relax the criterion to avoid this shift, intercepting only the case where the value is read as another legal value. A small number of retained samples may thus still carry an incorrect symbol, and warning-class samples are not automatically verified at all.

## 5.3 Conclusion

This paper addresses the long-tailed data predicament in trafic sign detection with a structured-prior-guided difusion inpainting framework with physical consistency. A trafic sign is a regulated artefact, defined precisely by a national standard. We split its semantic, colour and geometric priors along physical dimensions and inject each one in the conditioning form that matches it. Semantics becomes structured JSON text. Appearance becomes a front-view vector template painted with the measured dominant colour, injected via IP-Adapter. Geometry becomes an afine-aligned template, injected via ControlNet. A CIELAB chromaticity loss and a Sobel structure loss add two physical consistency terms. Together with the leave-one-out protocol they attribute each module’s contribution to performance.

The experimental evidence covers three levels. On generation quality, our method ranks first on all nine metrics in a cross-source zero-shot evaluation over the full 5,182-sample TT100K set. Its OCR exact-match rate reaches 91.1%, 46.9 percentage points above the 12B FLUX.1 Fill, at 1/14 of that model’s inference time. On module contribution, leave-one-out ablations quantify the decisive role of the appearance prior for digit legibility and of colour injection for distributional realism. They also show that the Sobel loss brings a structural gain at no distributional cost, and they report the geometry–text trade-of of ControlNet openly. On the downstream task, 16 controlled training runs confirm that the synthetic data raises the pooled AP50 of rare classes by 1.23×–7.40×.

Three limits remain. Template coverage does not extend to noncircular and non-triangular signs. Geometric fitting fails under large-angle occlusion. The single-target editing framework runs serially. The downstream conclusions apply to speed-limit classes and some warning signs, and the marginal advantage over Copy-Paste tends to vanish at the 10% data fraction. Future work includes migrating to larger backbones such as FLUX, cross-validating with a second detector, and extending laterally to inpainting tasks on other regulated targets such as licence plates and road markings.

The framework assumes one thing: a specification defines the semantics, appearance and geometry of the target precisely. Licence plates, road markings, industrial nameplates and package labels satisfy that assumption too. Our route therefore transfers to inpainting tasks on those regulated targets, and to broader applications such as autonomous driving, street-view understanding and automated high-definition map production.

## References

[1] Zhe Zhu, Dun Liang, Songhai Zhang, Xiaolei Huang, Baoli Li, and Shimin Hu. 2016. Trafic-Sign Detection and Classification in the Wild. In CVPR. 2110–2118.

[2] Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. 2012. Man vs. Computer: Benchmarking Machine Learning Algorithms for Trafic Sign Recognition. Neural Networks 32 (2012), 323–332.

[3] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. 2016. You Only Look Once: Unified, Real-Time Object Detection. In CVPR. 779–788.

[4] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. 2023. Ultralytics YOLOv8. https: //github.com/ultralytics/ultralytics.

[5] Glenn Jocher, Jing Qiu, Muhammad Liu, et al. 2026. Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. arXiv preprint arXiv:2606.03748.

[6] Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, and Guiguang Ding. 2024. YOLOv10: Real-Time End-to-End Object Detection. In NeurIPS.

[7] Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, and Jie Chen. 2024. DETRs Beat YOLOs on Real-time Object Detection. In CVPR. 16965–16974.

[8] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. 2020. End-to-End Object Detection with Transformers. In ECCV. 213–229.

[9] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. 2020. YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934.

[10] Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D. Cubuk, Quoc V. Le, and Barret Zoph. 2021. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. In CVPR. 2918–2928.

[11] Debidatta Dwibedi, Ishan Misra, and Martial Hebert. 2017. Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection. In ICCV. 1301–1310.

[12] Patrick Pérez, Michel Gangnet, and Andrew Blake. 2003. Poisson Image Editing. ACM Transactions on Graphics 22, 3 (2003), 313–318.

[13] Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. 2019. Class-Balanced Loss Based on Efective Number of Samples. In CVPR. 9268–9277.

[14] Jonathan Ho, AjayJain, and Pieter Abbeel. 2020. Denoising Difusion Probabilistic Models. In NeurIPS.

[15] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2021. Denoising Difusion Implicit Models. In ICLR.

[16] Prafulla Dhariwal and Alexander Nichol. 2021. Difusion Models Beat GANs on Image Synthesis. In NeurIPS.

[17] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis with Latent Difusion Models. In CVPR. 10684–10695

[18] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding Conditional Control to Text-to-Image Difusion Models. In ICCV. 3836–3847.

[19] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. 2024. T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability for Text-to-Image Difusion Models. In AAAI. 4296–4304.

[20] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Difusion Models. arXiv preprint arXiv:2308.06721.

[21] Vlad Shakhuro, Boris Faizov, and Anton Konushin. 2019. Rare Trafic Sign Recognition using Synthetic Training Data. In ICVIP. 23–26.

[22] Hanqing Zhao, Dianmo Sheng, Jianmin Bao, Dongdong Chen, Dong Chen, Fang Wen, Lu Yuan, Ce Liu, Wenbo Zhou, Qi Chu, Weiming Zhang, and Nenghai Yu. 2023. X-Paste: Revisiting Scalable Copy-Paste for Instance Segmentation using CLIP and StableDifusion. In ICML.

[23] Saksham Suri, Fanyi Xiao, Animesh Sinha, Sean Chang Culatana, Raghuraman Krishnamoorthi, Chenchen Zhu, and Abhinav Shrivastava. 2023. Gen2Det: Gen erate to Detect. arXiv preprint arXiv:2312.04566.

[24] Ruifei He, Shuyang Sun, Xin Yu, Chuhui Xue, Wenqing Zhang, Philip Torr, Song Bai, and Xiaojuan Qi. 2023. Is Synthetic Data from Generative Models Ready for Image Recognition? In ICLR.

[25] Brandon Trabucco, Kyle Doherty, Max Gurinas, and Ruslan Salakhutdinov. 2024. Efective Data Augmentation with Difusion Models. In ICLR.

[26] Weijia Wu, Yuzhong Zhao, Hao Chen, Yuchao Gu, Rui Zhao, Yefei He, Hong Zhou, Mike Zheng Shou, and Chunhua Shen. 2023. DatasetDM: Synthesizing Data with Perception Annotations Using Difusion Models. In NeurIPS.

[27] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. 2023. TextDifuser: Difusion Models as Text Painters. In NeurIPS.

[28] Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. 2024. AnyText: Multilingual Visual Text Generation and Editing. In ICLR.

[29] Yukang Yang, Dongnan Gui, Yuhui Yuan, Weicong Liang, Haisong Ding, Han Hu, and Kai Chen. 2023. GlyphControl: Glyph Conditional Control for Visual Text Generation. In NeurIPS.

[30] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, et al. 2024. SAM 2: Segment Anything in Images and Videos. arXiv preprint arXiv:2408.00714.

[31] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. 2022. Resolution-Robust Large Mask Inpainting with Fourier Convolutions. In WACV. 2149–2159.

[32] Wenbo Li, Zhe Lin, Kun Zhou, Lu Qi, Yi Wang, and Jiaya Jia. 2022. MAT: Mask-Aware Transformer for Large Hole Image Inpainting. In CVPR. 10758–10768.

[33] Jonathan Ho and Tim Salimans. 2021. Classifier-Free Difusion Guidance. In NeurIPS Workshop on Deep Generative Models.

[34] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J. Fleet, Radu Soricut, Jason Baldridge, Mohammad Norouzi, Peter Anderson, and William Chan. 2023. Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting. In CVPR. 18359–18369.

[35] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. 2024. A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting. In ECCV. 195–211.

[36] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. 2024. BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Difusion. In ECCV. 150–168.

[37] Yikai Wang, Chenjie Cao,Junqiu Yu, Ke Fan, Xiangyang Xue, and Yanwei Fu. 2025. Towards Enhanced Image Inpainting: Mitigating Unwanted Object Insertion and

Preserving Color Consistency. In CVPR. 23237–23248.

[38] Black Forest Labs. 2024. Introducing FLUX.1 Tools: FLUX.1 Fill [dev]. https: //bfl.ai/blog/24-11-21-tools.

[39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In ICML. 8748–8763.

[40] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. 2023. Paint by Example: Exemplar-based Image Editing with Difusion Models. In CVPR. 18381–18391.

[41] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. 2024. AnyDoor: Zero-shot Object-level Image Customization. In CVPR. 6593–6602.

[42] Black Forest Labs. 2025. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space. arXiv preprint arXiv:2506.15742.

[43] Qwen Team, Alibaba Group. 2025. Qwen-Image Technical Report. arXiv preprint arXiv:2508.02324.

[44] Yuning Du, Chenxia Li, Ruoyu Guo, Xiaoting Yin, Weiwei Liu, Jun Zhou, Yifan Bai, Zilin Yu, Yehua Yang, Qingqing Dang, and Haoshuang Wang. 2020. PP-OCR: A Practical Ultra Lightweight OCR System. arXiv preprint arXiv:2009.09941.

[45] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. 2023. Segment Anything. In ICCV. 4015–4026

[46] Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. 2023. Eficient Difusion Training via Min-SNR Weighting Strategy. In ICCV. 7441–7451.

[47] Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 2022. 8-bit Optimizers via Block-wise Quantization. In ICLR.

[48] Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization. In ICLR.

[49] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. 2023. UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Difusion Models. In NeurIPS.

[50] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. 2004. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing 13, 4 (2004), 600–612

[51] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Efectiveness of Deep Features as a Perceptual Metric. In CVPR. 586–595.

[52] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In NeurIPS.

[53] Christian Szegedy, Vincent Vanhoucke, Sergey Iofe, Jon Shlens, and Zbigniew Wojna. 2016. Rethinking the Inception Architecture for Computer Vision. In CVPR. 2818–2826.

[54] Gaurav Sharma, Wencheng Wu, and Edul N. Dalal. 2005. The CIEDE2000 Color-Diference Formula: Implementation Notes, Supplementary Test Data, and Math ematical Observations. Color Research & Application 30, 1 (2005), 21–30.

[55] John Canny. 1986. A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence 8, 6 (1986), 679–698.

[56] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022. DPM-Solver++: Fast Solver for Guided Sampling of Difusion Probabilistic Models. arXiv preprint arXiv:2211.01095.