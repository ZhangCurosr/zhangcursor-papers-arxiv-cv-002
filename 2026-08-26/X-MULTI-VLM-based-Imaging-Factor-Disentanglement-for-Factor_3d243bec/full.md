# X-MULTI: VLM-based Imaging Factor Disentanglement for Factor-Aware Image Synthesis

Sonali Godavarthy<sup>1\*</sup>, Matthias Neuwirth-Trapp<sup>2,3B</sup>, Tim-Felix Faasch<sup>3</sup>, Maarten Bieshaar<sup>3</sup>, Michael Moeller<sup>1</sup> Kristof Van Laerhoven<sup>1</sup>, and Danda Pani Paudel<sup>4</sup>

<sup>1</sup> University of Siegen, Siegen, Germany

2 ETH Zürich, Zürich, Switzerland

<sup>3</sup> Bosch Research, Hildesheim, Germany

INSAIT, Sofia University “St. Kliment Ohridski”, Sofia, Bulgaria

Abstract. Imaging factor disentanglement in text-to-image generation aims to independently control image acquisition properties such as types of camera lenses, sensor types, viewpoints, and domains to enable combinatorial generalization. This should let the model synthesize novel factor combinations unobserved in the training data, such as pairing a fisheye lens with an event sensor never observed in training data. Recent work, MULTI, introduced learnable, factor-specific embeddings to disentangle imaging factors, along with the Factor Alignment Accuracy (FAA) metric to evaluate disentanglement quality. We identify and address two independent limitations. First, MULTI’s pixel-level reconstruction objective supervises the model only on observed imaging factor combinations, providing no direct training signal for novel combinations. We therefore propose X-MULTI, which uses a pretrained vision-language model (VLM) to supervise novel factor combinations synthesized during training. Second, we show the FAA metric exhibits severe cross-factor correlation leakage, misrepresenting true disentanglement quality. We therefore propose Improved-FAA (I-FAA), which employs factor-specific augmentation strategies to break these correlations and enables more rigorous evaluation. Experiments demonstrate that X-MULTI achieves improved factor alignment on novel combinations compared to MULTI. Moreover, we show that correlation leakage in FAA distorts the evaluation of true factor disentanglement and I-FAA reduces this leakage and therefore provides a more robust assessment of factor alignment.

Keywords: Text-to-Image · Difusion Models · Factor Disentanglement · Textual Inversion · Vision-Language Model

## 1 Introduction

Text-to-image (T2I) difusion models have achieved remarkable visual fidelity in synthesizing images from natural language descriptions [9, 29–31, 33]. However, precise control over low-level image acquisition properties such as types of camera lens, sensor type, viewpoint, and domain remains challenging [13]. These factors define diferent aspects of the image formation process, but in real datasets they often appear only in limited and correlated combinations. Therefore, a model can be considered factor-disentangled only if it can modify each factor independently and synthesize valid novel combinations that were not observed during training. This problem, termed imaging factor disentanglement, requires learning representations that capture the semantic and physical identity of each factor while allowing them to be modified independently [13] to generate images with novel factor combinations that are not present during training. Controlling these imaging factors is important because collecting real data for all acquisition conditions is expensive, while vision systems still need to remain robust across sensor types, lenses, viewpoints, and domains.

Recent work, MULTI [13], introduced learnable factor-specific text embeddings optimized using Textual Inversion (TI) [11]. These embeddings are inserted into the text prompt to represent imaging factors, and are optimized with the standard difusion reconstruction loss while keeping the T2I difusion model frozen. However, MULTI’s reconstruction-based training objective is applied only for factor combinations that are represented in the training data, so the training does not receive any signal for novel, unseen combinations. This leaves the model unable to reliably distinguish semantically distinct factors (e.g., thermal sensor from rgb sensor), resulting in factor confusion at inference time. To encourage disentanglement and thereby allow the synthesis of images for novel factor combinations, a supervisory signal is required that can provide meaningful signal during training. We therefore propose X-MULTI (eXtended MULTI) which introduces supervision from a zero-shot Vision-Language Model (VLM) [2, 27]. During training, images with novel factor combinations are generated and the VLM acts as external semantic classifier, independently predicting factors for these generated images and thereby providing factor-level supervisory signals.

To evaluate whether generated images contain the specified factors, authors of MULTI [13] proposed Factor Alignment Accuracy (FAA), a novel classifierbased metric measuring factor correctness. We demonstrate that this metric is contaminated by severe correlations between the factor classifiers, enabling the metric to exploit these correlations rather than independently taking each one into account. This evaluation misrepresents the true quality of imaging factor disentanglement. To address this critical limitation, we introduce I-FAA (Improved Factor Alignment Accuracy), a novel evaluation metric, which employs class-balanced under-sampling and diverse factor-specific augmentation strategies during training of the imaging factor classifiers to break correlations between factors and mitigate shortcut learning of dataset-specific cues. This design provides more accurate factor disentanglement evaluation by eliminating the reliance on inter-factor patterns.

To study imaging factor disentanglement, we conduct experiments on DF-RICO [13] benchmark. First, we show improvements of our method over MULTI and other baselines, particularly for novel factor combinations. Second, we show cross-factor leakage in FAA, and I-FAA through correlation analysis.

Our main contributions are:

– X-MULTI Approach: Introduces zero-shot VLM-based supervision for synthetically generated unseen imaging-factor combinations, providing explicit factor-level supervision beyond the reconstruction-only training.

– I-FAA Metric: Introduces an improved factor-alignment evaluation metric that reduces cross-factor shortcut learning through class-balanced training and factor-specific augmentations, enabling more robust assessment of imaging factor disentanglement.

## 2 Related Work

Disentanglement in Generative Models: T2I models such as SDXL [30] provide strong generative backbones, while DreamBooth [32] adapts these models to specific visual concepts. Disentanglement in T2I models typically focuses on separating objects from their attributes, such as color or texture. Early approaches utilized Low-Rank Adaptation (LoRA) blocks to decouple style from content [10, 23]. Textual Inversion (TI) [11] optimizes a new token embedding for a target concept while keeping the model weights frozen, which is then inserted into prompts to generate that concept. This idea has been applied to learn embeddings for visual concepts such as color, shape, style, texture and object appearance [3, 28, 38, 39, 43]. To prevent attribute leakage, several methods steer cross-attention maps by constraining concept tokens to attend to their corresponding image regions [5,15,18,36,47] or employ explicit masks to isolate foreground objects from their backgrounds [1,19,22]. More recently, transformerbased architectures have leveraged the modulation space for multi-concept disentanglement [12, 49]. However, these methods mainly disentangle semantic or personalized concepts rather than imaging acquisition factors, making direct comparison with X-MULTI dificult.

Imaging-Factor Disentanglement: The image synthesis conditioned on imaging factors has received less attention. MULTI [13] addressed this by utilizing TI to optimize imaging factor embeddings. While MULTI improved over baselines for novel factor combinations, its factor alignment remained limited. We improve upon MULTI by introducing VLM-guided supervision to encourage disentanglement of imaging factors.

Discriminator-Based Guidance and VLM Supervision: Large-Language Models (LLMs) and Vision-Language Models (VLMs) have emerged as versatile zero-shot discriminators [27]. LLMs and VLMs are increasingly repurposed for complex tasks such as object detection [2], optical character recognition [20], and more [7, 41]. Recent frameworks have integrated these models as frozen discriminators to provide guidance, ensuring that generative models follow high-level conceptual constraints [2, 4, 42]. X-MULTI leverages this paradigm by utilizing a VLM as an external classifier to enforce factor identity, improving image synthesis of novel factor combinations.

Disentanglement Evaluation Metrics: Evaluating factor disentanglement requires metrics that assess each factor in isolation. [13] introduced FAA, which evaluates imaging-factor disentanglement using factor classifiers trained on the DF-RICO benchmark [13]. While classifier-based evaluation metrics like FAA often rely on highly correlated, co-occurring dataset shortcuts, they fail to isolate individual target attributes. To address this, I-FAA introduces factor-specific augmentations that break these correlations to ensure reliable disentanglement assessment.

## 3 Background

Imaging Factor Disentanglement: Following MULTI [13], imaging factor disentanglement aims to decompose image formation into K factors such as lens, sensor, viewpoint, and domain. We consider a collection of datasets $\mathcal { D } =$ $D ^ { ( 1 ) } , \ldots , D ^ { ( N ) }$ with shared factor annotations, where each image is associated with a factor tuple $\mathbf { f } = ( f ^ { ( k ) } ) _ { k \in \mathcal { K } }$ . Each factor k takes values from a discrete category $\mathcal { F } ^ { ( k ) }$ , defining the full factor space $\mathcal { F } = \mathcal { F } ^ { ( 1 ) } \times \cdots \times \mathcal { F } ^ { ( K ) }$ . In practice, training data usually only covers a sparse subset ${ \mathcal { F } } _ { \mathrm { o b s } } \subset { \mathcal { F } }$ . The challenge is to learn factor representations that produce images which reflect any target tuple f, including combinations absent from training.

Learnable Embeddings: Each factor value $f ^ { ( k ) }$ is represented by a learnable token $t ^ { ( k ) }$ , implemented as a sequence of n embedding vectors, $\mathcal { V } ( t ^ { ( k ) } ) =$ $\{ v _ { 1 } ^ { ( k ) } , \ldots , v _ { n } ^ { ( k ) } \} \in \mathbb { R } ^ { n \times d }$ Following MULTI [13], these tokens are inserted into a structured prompt together with an image caption and optimized using the standard difusion loss,

$$
\mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } _ { z _ { t } , x , \epsilon , t } [ | | \epsilon - \mathcal { U } ( z _ { t } , t , \mathbf { c } ) | | _ { 2 } ^ { 2 } ]\tag{1}
$$

where ϵ is the sampled noise, $z _ { t }$ is the noisy latent at timestep t, U is the denoising network, and c is the text-conditioning embedding from the structured prompt. The difusion backbone remains frozen.

Factor Alignment Accuracy (FAA): To evaluate whether the generated images faithfully represent the intended factors, MULTI introduced Factor Alignment Accuracy (FAA) [13], where for each factor category, a classifier is trained. During evaluation, a generated image is passed through these classifiers, and each predicted factor is compared against the corresponding factor label used to condition the generation. FAA measures the fraction of predictions that match the target factor labels, thereby estimating whether the generated image visually contains the requested imaging factors.

![](images/ab0b7e6d63f747dadc63bc12d3777a11149046f2cd2d9bf01acbb0c7eb90f1d9.jpg)  
Fig. 1: X-MULTI Architecture. Overview of our framework, which augments MULTI (components enclosed within the dark grey dashed box) with zero-shot VLM supervision to optimize factor embeddings via joint difusion and VLM-based factor alignment loss.

## 4 Methodology

## 4.1 X-MULTI: VLM-based Factor Disentanglement

To strengthen factor disentanglement and synthesize images with novel factor combinations, we introduce X-MULTI, which augments MULTI with zero-shot VLM-based supervision. MULTI learns factor embeddings in two stages: (1), it learns general embeddings $\Theta _ { \mathrm { g e n } }$ for broad imaging factors such as lens, sensor, viewpoint, and domain (e.g. fisheye lens); (2), it adapts these embeddings to the specific factor values of a target dataset $\Theta _ { \mathrm { s p e c } }$ (e.g., fisheye lens of Woodscape dataset). X-MULTI retains this two-stage structure, and adds VLM-based supervision only at Stage-1. Fig. 1 provides an overview of the proposed pipeline. VLM-based Supervision: Let $x _ { \mathrm { g n r } }$ be an image generated with the target factor tuple $\mathbf { f } _ { \mathrm { g n r } } = ( f _ { \mathrm { g n r } } ^ { ( k ) } ) _ { k \in \mathcal { K } }$ . We employ a frozen zero-shot VLM Z and factorspecific prompts $\mathcal { P } = \{ \mathcal { P } _ { 1 } , \ldots , \mathcal { P } _ { | \mathcal { K } | } \}$ . For each factor category $k ,$ the VLM predicts $f _ { \mathrm { p r e d } } ^ { ( k ) } = \mathcal { Z } ( x _ { \mathrm { g n r } } , \mathcal { P } _ { k } )$ The VLM-based factor alignment loss is defined as

$$
\mathcal { L } _ { \mathrm { v l m } } = \sum _ { k \in \mathcal { K } } \ell _ { \mathrm { C E } } \left( f _ { \mathrm { p r e d } } ^ { ( k ) } , f _ { \mathrm { g n r } } ^ { ( k ) } \right) ,\tag{2}
$$

where $\ell _ { \mathrm { C E } }$ denotes cross-entropy loss. Factor-wise VLM reliability is determined by evaluating $\mathcal { Z }$ on ground-truth images and comparing its predictions against true labels using the factor-specific prompts.

Training Objective: Following MULTI [13], we optimize the general factor embeddings $\Theta _ { \mathrm { g e n } }$ with the difusion objective. After an initial difusion-only warm-up, in each training step, a real batch $\boldsymbol { B } _ { \mathrm { r e a l } }$ is sampled from the training datasets. From the factor values present in $B _ { \mathrm { r e a l } } .$ we additionally construct a synthetic factor tuple $\mathbf { f } _ { \mathrm { g n r } }$ by resampling factor values across categories. The current difusion model G then produces an image $x _ { \mathrm { g n r } } = \mathcal { G } ( \mathbf { f } _ { \mathrm { g n r } } )$ , which is used only for VLM-based supervision. The training objective is

$$
\begin{array} { r } { \hat { \theta } _ { \mathrm { g e n } } = \arg \underset { \theta _ { \mathrm { g e n } } } { \operatorname* { m i n } } \mathbb E \left[ \mathcal L _ { \mathrm { d i f f } } + \lambda \mathcal L _ { \mathrm { v l m } } \right] , } \end{array}\tag{3}
$$

where λ controls the strength of VLM-based supervision.

## 4.2 Improved Factor Alignment Accuracy (I-FAA)

To evaluate imaging factor disentanglement, we propose Improved Factor Alignment Accuracy (I-FAA) as a new metric. Given a set of N generated images $\{ x _ { i } \} _ { i = 1 } ^ { N } .$ , where each $x _ { i }$ is conditioned on the factor tuple $\mathbf { f } _ { i } = ( f _ { i } ^ { ( 1 ) } , \dots , f _ { i } ^ { ( K ) } )$ 2 we use a set of factor classifiers $\mathcal { S } = \{ \boldsymbol { S } _ { 1 } , \boldsymbol { S } _ { 2 } , \ldots , \boldsymbol { S } _ { K } \}$ , where $\boldsymbol { S _ { k } }$ predicts factor values in $\mathcal { F } ^ { ( k ) }$ . The per-category alignment accuracy is defined as

$$
\mathrm { I } \mathrm { - F A A } _ { k } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \Big [ S _ { k } ( x _ { i } ) = = f _ { i } ^ { ( k ) } \Big ] .\tag{4}
$$

The overall I-FAA score is obtained by averaging over all factor categories

$$
\operatorname { I - F A A } _ { \operatorname { a v g } } = { \frac { 1 } { \mathcal { K } } } \sum _ { k = 1 } ^ { \kappa } \operatorname { I - F A A } _ { k } .\tag{5}
$$

Classifier Architecture: Each factor classifier $\boldsymbol { S _ { k } }$ consists of the shared frozen visual backbone ϕ with parameters $\Theta _ { \mathrm { b b } }$ , followed by a factor-specific trainable linear head $h _ { k } .$ with parameters $\Theta _ { h _ { k } }$ . Thus, $S _ { k }$ is parameterized by the shared frozen backbone parameters $\Theta _ { \mathrm { b b } }$ and the factor-specific head parameters $\Theta _ { h _ { k } }$ During training, only $\Theta _ { h _ { k } }$ is optimized, while $\Theta _ { \mathrm { b b } }$ remains fixed.

Class-Balanced Sampling: To mitigate bias in highly skewed data, we adopt class-balanced under-sampling during training of the classifiers. Let $D _ { k } = \{ ( x _ { i } ,$ $f _ { i } ^ { ( k ) } ) \}$ } denote the training dataset for factor category k, where $f _ { i } ^ { ( k ) } \in \mathcal { F } ^ { ( k ) }$ is the factor label associated with image $x _ { i }$ . Let $n _ { \mathrm { m i n } } ^ { ( k ) }$ denote the number of the smallest class in category k. Samples from every class $c \in \mathcal { F } ^ { ( k ) }$ are randomly under-sampled to $n _ { \mathrm { m i n } } ^ { ( k ) }$ , ensuring balanced class representation.

Factor-Specific Augmentation: Beyond class imbalance, classifiers may acquire shortcuts by learning dataset-level features that co-occur with a target factor. To reduce such shortcut learning, we employ each classifier $S _ { k }$ with dedicated augmentations. Formally, for category k we define $\mathcal { A } ^ { ( k ) } = \{ a _ { 1 } ^ { ( k ) } , a _ { 2 } ^ { ( k ) } , \dots , a _ { M _ { k } } ^ { ( k ) } \}$ 2 where $M _ { k }$ denotes the number of augmentation operations assigned to category k. During training, augmentations are sampled uniformly from $\mathcal { A } ^ { ( k ) }$ and applied to the image x to produce the augmented image $\tilde { x } ^ { ( k ) }$

Table 1: Factor-specific Prompts for VLM-based supervision. Our method uses structured prompts and JSON output constraints to independently query the zero-shot VLM for each imaging factor.  
Factor Prompt   
Sensor Classify the SENSOR type of this image (0-4):   
0 = RGB: normal color camera   
1 = Thermal: grayscale images with bright/warm colors/white   
2 = Gated: high contrast black and white, harsh lighting, foggy/smoky   
3 = Event: motion changes, red/blue motion capture   
4 = RGB Thermal (Fusion): Color images containing an “overlay” or “glow” of   
thermal data. Look for heat signatures (bright orange/yellow/white highlights)   
mapped onto real-world objects and textures.   
Return only JSON: {"sensor": <int>}   
Lens Classify the LENS type of this image (0-1):   
0 = Normal: standard camera lens   
1 = Fisheye: distorted edges, wide field of view   
Return only JSON: {"lens": <int>}   
Viewpoint Classify the VIEWPOINT of this image (0-4):   
0 = Drone: camera placed on a flying drone   
1 = Pole: camera mounted on a pole   
2 = Front: camera faces forward   
3 = Back: camera faces backward   
4 = Side: camera placed on left/right (typically on car)   
Return only JSON: {"viewpoint": <int>}   
Domain Classify the DOMAIN of this image (0-2):   
0 = Real: real-world images   
1 = Simulation: rendered or simulated images   
2 = Video Game: captured from video-game   
Return only JSON: {"domain": <int>}

Optimization: Given augmented inputs and balanced mini-batches, each classifier $\scriptstyle { S _ { k } }$ is optimized via the standard cross-entropy objective

$$
\hat { \theta } _ { h _ { k } } = \arg \operatorname* { m i n } _ { \theta _ { h _ { k } } } \mathbb { E } _ { ( x , f ^ { ( k ) } ) \sim D _ { k } } \left[ \ell _ { \mathrm { C E } } \left( S _ { k } \left( { \tilde { x } } ^ { ( k ) } \right) , f ^ { ( k ) } \right) \right] .\tag{6}
$$

## 5 Experimental Results

## 5.1 Implementation Details

We utilize the DF-RICO benchmark from MULTI [13], spanning 15 autonomous driving and surveillance datasets annotated with four factor categories: (1) camera lens (normal, fisheye), (2) source/domain (real, simulation, video-game), (3) viewpoint (front, back, side, drone, pole), and (4) sensor modality (rgb, thermal, rgb-thermal, gated, event).

X-MULTI: We follow the implementation setup of MULTI [13]. We use Stable Difusion XL (SDXL) [30] as our generative backbone, where each factor is represented by n = 15 vectors optimized via AdamW [26] (weight decay: $1 0 ^ { - 2 } )$ using a linear warmup and cosine annealing learning rate schedule (maximum: $1 0 ^ { - 4 } )$ . Training is conducted for 10 epochs with a total batch size of 4.

Table 2: Data augmentations for I-FAA classifier training. Category-specific, non-overlapping image augmentations applied during I-FAA (ours) training.
<table><tr><td>Augmentation Category &amp; Type</td><td>Lens</td><td>Sensor</td><td>Domain</td><td>Viewpoint</td></tr><tr><td>Target Classes</td><td>normal, fisheye</td><td>rgb, thermal, rgb-thermal, event, gated</td><td>real, simulation, video-game</td><td>front, back, side, pole, drone</td></tr><tr><td>Geometric</td><td></td><td></td><td></td><td></td></tr><tr><td>Horizontal flip</td><td>√</td><td>√</td><td>√</td><td>Mild</td></tr><tr><td>Vertical flip</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>Random rotation &amp; translation</td><td>Mild</td><td>Moderate</td><td>Moderate</td><td>Mild</td></tr><tr><td>Color / Appearance</td><td></td><td></td><td></td><td></td></tr><tr><td>Color jitter, Grayscale, &amp; Noise</td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td>Gaussian blur</td><td>Moderate</td><td>Moderate</td><td>Strong</td><td>Mild</td></tr><tr><td>Occlusion / Region</td><td></td><td></td><td></td><td></td></tr><tr><td>Cutout, Random erasing, &amp; Hide-and-seek</td><td>Moderate</td><td>Mild</td><td>Moderate</td><td></td></tr><tr><td>Additional</td><td></td><td></td><td></td><td></td></tr><tr><td>Radial distortion</td><td></td><td>Moderate</td><td>Moderate</td><td>Mild</td></tr><tr><td>Perspective warp</td><td></td><td>√</td><td>√</td><td></td></tr></table>

![](images/a8235f8c46232c67491d3f8f19d244585543acce905b22ca4119960aa1466a23.jpg)  
Fig. 2: Cramer’s V correlations. Pairwise correlations between predicted factors for FAA (left) and I-FAA (ours) (right), showing that I-FAA reduces cross-factor information leakage. High values show strong correlations.

For the VLM-based supervision branch, we employ Qwen2-VL-7B-Instruct [40] as a zero-shot VLM classifier starting from epoch 3 with supervision strength of $1 0 ^ { - 6 }$ . Each mini-batch consists of 3 real samples and 1 synthetic sample. We formulate structured factor-specific prompts as shown in Table 1 that independently query the VLM for each imaging factor: sensor, lens, domain, and viewpoint. To prevent incorrect supervision of unreliable VLM predictions which is discussed in Section 5.2, rgb-thermal sensor and non-front viewpoints are masked to 0.

I-FAA: We employ DINOv3 [37] as the vision backbone for all factor classifiers. We train the linear prediction heads using the AdamW optimizer [26] with a learning rate of $1 0 ^ { - 4 }$ and a cosine annealing learning rate schedule [25], and a batch size of 32. Table 2 shows data augmentations carried out during training of each factor classifier.

Table 3: Qualitative comparison of novel combinations. Image generation with unseen factor combinations. Our method, X-MULTI shows better factor adherence.
<table><tr><td>Domain</td><td>real</td><td>video-game</td><td>real</td><td>real</td><td>video-game</td><td>simulation</td><td>video-game</td><td>simulation(shift)</td></tr><tr><td>Sensor</td><td>thermal</td><td>rgb</td><td>gated</td><td>rgb-thermal</td><td>rgb</td><td>thermal</td><td>thermal</td><td>thermal(timo)</td></tr><tr><td>Viewpoint</td><td>fisheye(loaf)</td><td>fisheye</td><td>normal</td><td>fisheye</td><td>normal</td><td>normal</td><td>fisheye</td><td>normal</td></tr><tr><td>Lens</td><td>front</td><td>drone(visdrone)</td><td>pole</td><td>front</td><td>pole</td><td>drone(visdrone)</td><td>pole</td><td>pole(fisheye8k)</td></tr></table>

![](images/11d2e1394b622eee1bc89d01bcc245033db4d056e3522d278d291dfee1f2aa5b.jpg)

![](images/ae16c7aa5f974e7f6618399d3651785a8373a58ba5c10ff838dfb92c40b0d976.jpg)  
(a) Lens

![](images/a1d65272570b56b49bf00bf0decd6872c957f8fbd4a7fe06d2b68b4de0ffc89a.jpg)  
(b) Sensor

![](images/6531b06b4a498b2a955fd6d98d01755c5eb862b08996c77e66d927a82296bdaa.jpg)  
(c) Domain

![](images/e80ad6eee4176bc7956cc97005a6f978b3c88717155ebc026bad2623f444f4f4.jpg)  
(d) Viewpoint  
Fig. 3: Confusion matrices of the zero-shot VLM for factor classification: (a) Lens, (b) Sensor, (c) Domain, and (d) Viewpoint. Rows correspond to ground-truth labels and columns to predicted labels.

Evaluation Metrics: We employ standard image generation metrics: Fréchet Inception Distance (FID) [17] to measure distributional similarity between real and generated images, CLIP Score [16] to assess semantic alignment with factorbased prompts (evaluated on three components: factors only, content description only, and full prompt), Inception Score (IS) [34] for image quality, and Diversity Score (DS) [46] for perceptual diversity. To measure factor disentanglement, we employ our introduced metric, I-FAA, and analyze its reliability in Sec. 5.3.

Baselines: We compare against: (1) SDXL Zeroshot [30], the pre-trained model without adaptation; (2) DreamBooth [32], fine-tuning-based personalization; (3) Inspiration Tree [39], hierarchical disentanglement of stylistic components (evaluated only on existing combinations, as its learned embeddings do not correspond to independent factor categories that can be recombined into novel combinations); and (4) MULTI [13], a prior factor disentanglement method.

Table 4: Qualitative comparison of novel factor combinations with ControlNets. Image-to-image generation using ControlNet guidance with altered factors, showing X-MULTI (ours) achieves best factor adherence.  
![](images/2a83438dc2ec0782817bb89dea6e21f44f2e2d16663828d35e7bb407afa7ec96.jpg)

## 5.2 Imaging Factor Disentanglement Results

The primary goal is to evaluate the quality of the synthesis of images with unseen novel factor combinations. We show both qualitative and quantitative results along with the analysis of the choice of VLM and prompt designs for VLM-based supervision.

Novel Factor Combinations: Table 3 shows qualitative inspection of images generated with novel factor combinations. SDXL Zeroshot fails to reflect target factors, while MULTI, DreamBooth, and X-MULTI produce more plausible combinations. X-MULTI shows clearer factor adherence particularly for viewpoint, lens, and domain than other baselines. X-MULTI shows less proficiency with thermal sensor but handled other sensors competently.

Table 4 shows the usage of ControlNets [45] for guiding the image generation process, and altering a factor (to respect control maps we use sensor or domain factors only). In this setting, our approach demonstrates good precision compared to MULTI and DreamBooth, which outperforms SDXL Zeroshot. Specifically, when substituting the event sensor for a gated one, DreamBooth retained event artifacts while our method executed the transformation cleanly. Similarly, shifting domains from real to video-game, our method completely eliminated real-world traces while MULTI and DreamBooth could not achieve this.

Table 5: Quantitative comparison on novel factor combinations. Performance metrics across methods with and without ControlNet structural guidance, with best values highlighted in bold. X-MULTI (ours) achieves best overall disentanglement.
<table><tr><td rowspan="3">Method</td><td colspan="6">Image Generation Metrics</td><td colspan="5">Improved-Factor Alignment Accuracy (I-FAA)</td></tr><tr><td rowspan="2">IS [34] ↑</td><td colspan="4">CLIP Score [16] ↑</td><td rowspan="2">DS [46] ↑</td><td rowspan="2">Lens ↑</td><td rowspan="2">Sensor ↑</td><td rowspan="2">Domain ↑</td><td rowspan="2">View ↑</td><td rowspan="2">Avg. ↑</td></tr><tr><td>factor</td><td>context</td><td>average</td><td>full-prompt</td></tr><tr><td>DreamBooth [32]</td><td>3.68</td><td>23.96</td><td>23.35</td><td>23.66</td><td>24.67</td><td>0.55</td><td>0.78</td><td>0.24</td><td>0.48</td><td>0.27</td><td>0.44</td></tr><tr><td>SDXL Zeroshot [30]</td><td>3.11</td><td>22.31</td><td>25.59</td><td>23.95</td><td>27.80</td><td>0.59</td><td>0.72</td><td>0.25</td><td>0.29</td><td>0.29</td><td>0.39</td></tr><tr><td>MULTI [13]</td><td>3.28</td><td>24.20</td><td>23.43</td><td>23.82</td><td>27.32</td><td>0.65</td><td>0.63</td><td>0.29</td><td>0.64</td><td>0.34</td><td>0.47</td></tr><tr><td>X-MULTI (ours)</td><td>3.23</td><td>24.62</td><td>23.57</td><td>24.10</td><td>28.22</td><td>0.59</td><td>0.76</td><td>0.30</td><td>0.66</td><td>0.40</td><td>0.53</td></tr><tr><td>DreamBooth-Canny</td><td>5.78</td><td>23.67</td><td>25.79</td><td>25.23</td><td>29.12</td><td>0.50</td><td>0.99</td><td>0.71</td><td>0.89</td><td>0.80</td><td>0.84</td></tr><tr><td>SDXL Zeroshot-Canny</td><td>5.74</td><td>23.88</td><td>26.75</td><td>25.31</td><td>28.95</td><td>0.51</td><td>0.99</td><td>0.63</td><td>0.84</td><td>0.75</td><td>0.78</td></tr><tr><td>MULTI-Canny</td><td>4.88</td><td>22.82</td><td>23.08</td><td>22.95</td><td>26.72</td><td>0.55</td><td>0.99</td><td>0.73</td><td>0.84</td><td>0.86</td><td>0.85</td></tr><tr><td>X-MULTI-Canny (ours)</td><td>4.64</td><td>23.91</td><td>23.12</td><td>23.51</td><td>29.60</td><td>0.56</td><td>0.99</td><td>0.81</td><td>0.89</td><td>0.87</td><td>0.89</td></tr><tr><td>DreamBooth-Depth</td><td>5.56</td><td>23.57</td><td>25.40</td><td>24.98</td><td>27.95</td><td>0.53</td><td>1.00</td><td>0.69</td><td>0.88</td><td>0.77</td><td>0.84</td></tr><tr><td>SDXL Zeroshot-Depth</td><td>5.49</td><td>23.62</td><td>26.57</td><td>25.10</td><td>29.82</td><td>0.52</td><td>1.00</td><td>0.66</td><td>0.81</td><td>0.71</td><td>0.77</td></tr><tr><td>MULTI-Depth</td><td>4.36</td><td>22.12</td><td>23.45</td><td>22.78</td><td>26.82</td><td>0.52</td><td>0.99</td><td>0.76</td><td>0.84</td><td>0.87</td><td>0.86</td></tr><tr><td>X-MULTI-Depth (ours)</td><td>4.87</td><td>23.63</td><td>24.79</td><td>23.21</td><td>29.81</td><td>0.53</td><td>1.00</td><td>0.71</td><td>0.88</td><td>0.88</td><td>0.87</td></tr></table>

Table 5 reports quantitative metrics for novel factor combinations with and without using ControlNets [45]. Under I-FAA, X-MULTI achieves the best overall factor disentanglement, CLIP alignment with factors, and DS while other metrics remain mixed across methods.

Efect of VLM-based Supervision: We evaluate the contribution of VLMbased supervision by comparing MULTI (without VLM-based supervision) and X-MULTI (with supervision) on novel factor combinations. MULTI achieves an average I-FAA of 0.47, while X-MULTI achieves 0.53, representing a 0.06 (11% relative) improvement. X-MULTI shows a best improvements in lens and viewpoint factors with 21%, 18% relative improvement respectively, while both sensor and domain show a relative improvement of 0.01. This demonstrates that supervision enhances factor disentanglement, especially for lens, and viewpoint.

VLM Analysis: Since VLM predictions provide the supervision in X-MULTI, we evaluate their reliability in a zero-shot setting using the prompts in Table 1. Figure 3 presents the confusion matrices for Lens, Sensor, Domain, and Viewpoint classification. Across all factors, the VLM achieves an average classification accuracy of ∼ 0.78, demonstrating that the model captures meaningful semantic representations of imaging factors. However, the analysis also reveals that certain factor categories are less reliably recognized. In particular, the rgb-thermal sensor exhibits strong confusion with rgb images, and several viewpoint categories other than front show inconsistent predictions. Since these inaccurate classification signals can negatively afect disentanglement learning in X-MULTI, we disable the VLM supervision loss for these categories during training. Specifically, the supervision strength is set to zero for the rgb-thermal sensor and for all viewpoints except front.

Table 6: Comparison of attention maps for FAA and I-FAA. Grad-CAM visualizations showing that I-FAA (ours) learns more semantically meaningful attention regions.  
![](images/712b230736dbc56c688ad87b2364de1f50d0601a80ea228b4e6b050f8df46260.jpg)

Table 7: FAA and I-FAA on ground-truth DF-RICO images. Comparison of FAA and I-FAA (ours) metric on ground-truth images.
<table><tr><td>Metric</td><td>Lens</td><td>Sensor</td><td>Domain</td><td>View</td><td> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td>FAA [13]</td><td>0.23</td><td>0.83</td><td>0.50</td><td>0.17</td><td>0.43</td></tr><tr><td>I-FAÁ</td><td>1.00</td><td>0.97</td><td>0.98</td><td>0.84</td><td>0.95</td></tr></table>

Ablation Studies: We carry out diferent ablation studies to evaluate the design of core components of X-MULTI.

(1) VLM Selection and Prompt Design: Since factor disentanglement depends on VLM feedback quality, we evaluate the zero-shot classification performance of VLM on ground-truth data. First, comparing VLM backbones shows that Qwen2-VL-7B-Instruct [40] gives 0.78 average accuracy and 0.51 for LLaVA-1.6 [24], indicating Qwen provides much stronger visual grounding. Second, comparing prompt designs using Qwen2-VL shows that detailed factor-specific prompts with per-class descriptions achieve 0.78 accuracy, simple prompts without descriptions achieve 0.70, and a single unified prompt classifying all factors jointly drops to 0.63, demonstrating that detailed, isolated prompts provide the best signal for X-MULTI.

– (2) Supervision strength: We vary the supervision strength (λ) and evaluate I-FAA on novel factor combinations. Moderate $( 1 0 ^ { - 6 } )$ and weak $( 1 0 ^ { - 8 } )$ supervision perform best, both reaching an I-FAA of 0.53. In contrast, overly

Table 8: Factor prediction comparison. Per-factor classification outputs compared against ground-truth labels for ground truth images. I-FAA (ours) predicts factor correctly.
<table><tr><td>Image</td><td>Factor</td><td>GT</td><td>FAA [13]</td><td>I-FAA</td></tr><tr><td></td><td>Lens Sensor Domain View</td><td>normal rgb real front</td><td>fisheye rgb simulation front</td><td>normal rgb real front</td></tr><tr><td></td><td>Lens Sensor Domain View</td><td>normal event real front</td><td>fisheye rgb real front</td><td>normal event real front</td></tr><tr><td></td><td>Lens Sensor Domain View</td><td>fisheye rgb real pole</td><td>fisheye rgb real drone</td><td>fisheye rgb real pole</td></tr><tr><td></td><td>Lens Sensor Domain View</td><td>normal thermal real front</td><td>normal thermal real pole</td><td>normal thermal real front</td></tr></table>

<table><tr><td>Image</td><td>Factor</td><td>GT</td><td>FAA [13]</td><td>I-FAA</td></tr><tr><td rowspan="4"></td><td>Lens</td><td>normal</td><td>normal</td><td>normal</td></tr><tr><td>Sensor</td><td>rgb</td><td>rgb</td><td>rgb</td></tr><tr><td>Domain</td><td>real</td><td>real</td><td>real</td></tr><tr><td>View</td><td>back</td><td>back</td><td>back</td></tr><tr><td rowspan="2"></td><td>Lens Sensor</td><td>normal rgb</td><td>normal rgb</td><td>normal rgb</td></tr><tr><td>Domain View</td><td>simulation front</td><td>simulation back</td><td>simulation front</td></tr><tr><td rowspan="2"></td><td>Lens Sensor</td><td>normal rgb</td><td>normal rgb</td><td>normal rgb</td></tr><tr><td>View</td><td>front</td><td>Domain video-game simulation video-game front</td><td>front</td></tr><tr><td rowspan="2"></td><td>Lens</td><td>normal</td><td>normal</td><td>normal</td></tr><tr><td>Sensor Domain</td><td>rgb real</td><td>rgb simulation</td><td>rgb real</td></tr></table>

strong supervision $( 1 0 ^ { - 4 } )$ degrades performance to 0.47, indicating that excessive supervision over-constrains factor adaptation.

## 5.3 Analysis of Metrics

To validate the reliability of FAA and I-FAA, we test whether their underlying classifiers isolate imaging factors independently rather than exploiting datasetspecific shortcuts. Specifically, we analyze cross-factor alignment using Grad-CAM attention maps to visually verify semantic localization, Cramer’s V correlation [8] to statistically detect information leakage, and metric scores alongside factor predictions on ground-truth DF-RICO [13] images to establish performance.

Attention Map Visualization: Table 6 visualizes Grad-CAM [35] attention maps for FAA and I-FAA across imaging factors. It shows that I-FAA learns more localized and semantically meaningful attention regions for each factor category, whereas FAA often exhibits entangled attention patterns. For example, in the fisheye lens samples, I-FAA strongly attends to image boundary curvature and geometric distortions, while FAA focuses on unrelated scene regions. Similarly, for viewpoint prediction, I-FAA concentrates on perspective-specific regions, whereas FAA frequently attends to the image center irrespective of viewpoint semantics. For sensor-related factors, I-FAA captures modality-specific characteristics scattered across regions more distinctly than FAA. Domain attention maps also demonstrate that I-FAA focuses on scene appearance cues by attending on objects, while FAA exhibits more inconsistent activations.

Cramér’s V Correlation Analysis: Figure 2 presents the pairwise Cramér’s V correlations [8] between predicted imaging factors for FAA and I-FAA. Ideally, a factor-isolated evaluation metric should produce low inter-factor correlations, since each factor prediction should depend primarily on the target factor rather than on co-occurring factors. FAA exhibits strong coupling between multiple factor pairs, particularly lens-viewpoint, domain-viewpoint, and lens-domain. In contrast, I-FAA reduces most of these correlations, indicating lower cross-factor leakage. However, residual correlations remain, especially for factor pairs that are strongly coupled or sparsely represented in DF-RICO. This highlights that augmentation in I-FAA classifier training mitigates shortcut learning but cannot fully compensate for correlations in the underlying data.

Metric and Factor Prediction Comparison: Since FAA was originally proposed as an evaluation metric without independently validating the accuracy of its underlying classifiers on ground-truth images, we first evaluate both FAA and I-FAA classifiers on DF-RICO. Table 7 shows that I-FAA improves the average classification accuracy from 0.43 to 0.95. This improvement stems from the classbalanced training and factor-specific augmentations introduced in I-FAA, which reduce shortcut learning from correlated imaging factors. The largest gains are observed for viewpoint and domain, where FAA frequently confuses factor predictions. Table 8 further illustrates that I-FAA produces more reliable factor-wise predictions, while FAA is more afected by inter-factor interference.

## 6 Discussion

Our numerical results lead to the following main findings:

VLM-Guided Supervision: Zero-shot VLM supervision provides additional training signals by predicting factors from synthetically generated images during training. This significantly enhances factor alignment, particularly for geometric attributes. ▶ Zero-shot supervision enhances disentanglement.

VLM Reliability and Prompt Context: X-MULTI relies on a VLM to provide supervision. Our VLM analysis shows that some factor values are less reliably recognized, so we mask unreliable predictions during training. This indicates that performance depends on both the VLM choice and the design of factor-specific prompts. Providing richer visual context in the prompts may further improve supervision. ▶ VLM-based supervision is useful, but its reliability depends on prompt design and VLM factor recognition.

Reliability of I-FAA: FAA exhibits cross-factor correlations consistent with shared representations entangling factor predictions, while I-FAA reduces this leakage through factor-specific augmentations. However, residual correlations remain and could be addressed with stronger augmentations or correlationreduction methods [6, 14, 21, 44, 48]. ▶ I-FAA reduces correlations for imaging factor disentanglement, but they could be reduced further.

## 7 Conclusion

In this work, we addressed imaging factor disentanglement for controllable image generation of unseen novel combinations of imaging factors. We first identify that MULTI’s pixel-level reconstruction focuses only on available training data leading to no factor-level supervision for novel combinations. We therefore proposed X-MULTI, which augments MULTI with zero-shot VLM supervision for unseen factor combinations to enforce semantic factor identity. We also identified a limitation in the FAA metric, i.e., it produces strongly entangled factor predictions, evidenced by high cross-factor correlation, and attention maps. To address this, we introduced I-FAA, an evaluation framework with targeted augmentation strategies that reduces these correlations and provides improved disentanglement assessment. Experiments demonstrated that X-MULTI achieves stronger factor disentanglement than baselines, particularly for novel factor combinations, and I-FAA substantially reduces correlations and establishes a more robust metric for factor disentanglement evaluation. Despite promising results, imaging factor disentanglement remains challenging.

## References

1. Avrahami, O., Aberman, K., Fried, O., Cohen-Or, D., Lischinski, D.: Break-a-scene: Extracting multiple concepts from a single image. In: SIGGRAPH Asia. pp. 1–12 (2023)

2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

3. Butt, M.A., Wang, K., Vazquez-Corral, J., van de Weijer, J.: Colorpeel: Color prompt learning with difusion models via color and shape disentanglement. In: ECCV. pp. 456–472. Springer (2024)

4. Cai, M., Liu, H., Mustikovela, S.K., Meyer, G.P., Chai, Y., Park, D., Lee, Y.J.: Vip-llava: Making large multimodal models understand arbitrary visual prompts. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12914–12923 (2024)

5. Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., Cohen-Or, D.: Attend-and-excite: Attention-based semantic guidance for text-to-image difusion models. ACM transactions on Graphics (TOG) 42(4), 1–10 (2023)

6. Chew, O., Lin, H.T., Chang, K.W., Huang, K.H.: Understanding and mitigating spurious correlations in text classification with neighborhood analysis. In: Findings of the association for computational linguistics (EACL). pp. 1013–1025 (2024)

7. Cho, J., Zala, A., Bansal, M.: Dall-eval: Probing the reasoning skills and social biases of text-to-image generation models. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3043–3054 (2023)

8. Cramér, H.: Mathematical methods of statistics, vol. 9. Princeton university press (1999)

9. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: ICML. pp. 12606–12633. PMLR (2024)

10. Frenkel, Y., Vinker, Y., Shamir, A., Cohen-Or, D.: Implicit style-content separation using b-lora. In: ECCV. pp. 181–198. Springer (2024)

11. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. In: ICLR (2022)

12. Garibi, D., Yadin, S., Paiss, R., Tov, O., Zada, S., Ephrat, A., Michaeli, T., Mosseri, I., Dekel, T.: Tokenverse: Versatile multi-concept personalization in token modulation space. ACM Transactions On Graphics (TOG) 44(4), 1–11 (2025)

13. Godavarthy, S., Neuwirth-Trapp, M., Faasch, T.F., Bieshaar, M., Moeller, M., Paudel, D.P.: Multi: Disentangling camera lens, sensor, view, and domain for novel image generation. arXiv preprint arXiv:2605.12134 (2026)

14. Gui, S., Ji, S.: Mitigating spurious correlations in llms via causality-aware posttraining. arXiv preprint arXiv:2506.09433 (2025)

15. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-or, D.: Prompt-to-prompt image editing with cross-attention control. In: ICLR (2022)

16. Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., Choi, Y.: Clipscore: A referencefree evaluation metric for image captioning. In: Proceedings of the 2021 conference on empirical methods in natural language processing. pp. 7514–7528 (2021)

17. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)

18. Jin, C., Tanno, R., Saseendran, A., Diethe, T., Teare, P.A.: An image is worth multiple words: Discovering object level concepts using multi-concept prompt learning. In: International Conference on Machine Learning (ICML). PMLR (2024)

19. Kwon, G., Jenni, S., Li, D., Lee, J.Y., Ye, J.C., Heilbron, F.C.: Concept weaver: Enabling multi-concept fusion in text-to-image models. In: CVPR. pp. 8880–8889 (2024)

20. Laurençon, H., Tronchon, L., Cord, M., Sanh, V.: What matters when building vision-language models? Advances in Neural Information Processing Systems 37, 87874–87907 (2024)

21. Lee, S., Payani, A., Chau, D.H.P.: Towards mitigating spurious correlations in image classifiers with simple yes-no feedback. In: AI and HCI Workshop at the International Conference on Machine Learning (ICML Workshop) (2023)

22. Li, P., Huang, Q., Ding, Y., Li, Z.: Layerdifusion: Layered controlled image editing with difusion models. In: SIGGRAPH Asia 2023 Technical Communications (SIGGRAPH Asia), pp. 1–4 (2023)

23. Liu, C., Shah, V., Cui, A., Lazebnik, S.: Unziplora: Separating content and style from a single image. In: ICCV. pp. 16776–16785 (2025)

24. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 26296–26306 (2024)

25. Liu, Z.: Super convergence cosine annealing with warm-up learning rate. In: CAIBDA 2022; 2nd International Conference on Artificial Intelligence, Big Data and Algorithms. pp. 1–7. VDE (2022)

26. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2017)

27. Luo, G., Granskog, J., Holynski, A., Darrell, T.: Dual-process image generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17972–17983 (2025)

28. Motamed, S., Paudel, D.P., Van Gool, L.: Lego: Learning to disentangle and invert personalized concepts beyond object appearance in text-to-image difusion models. In: European Conference on Computer Vision (ECCV). pp. 116–133. Springer (2024)

29. Nichol, A.Q., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., Mcgrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided difusion models. In: ICML. pp. 16784–16804. PMLR (2022)

30. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent difusion models for high-resolution image synthesis. In: ICLR (2023)

31. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: CVPR. pp. 10684–10695 (2022)

32. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image difusion models for subject-driven generation. In: CVPR. pp. 22500–22510 (2023)

33. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image difusion models with deep language understanding. Advances in neural information processing systems 35, 36479–36494 (2022)

34. Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X.: Improved techniques for training gans. Advances in neural information processing systems 29 (2016)

35. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Gradcam: Visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE international conference on computer vision. pp. 618–626 (2017)

36. Shentu, J., Watson, M., Al Moubayed, N.: Attencraft: Attention-guided disentanglement of multiple concepts for text-to-image customization. CoRR (2024)

37. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)

38. Sohn, K., Ruiz, N., Lee, K., Chin, D.C., Blok, I., Chang, H., Barber, J., Jiang, L., Entis, G., Li, Y., et al.: Styledrop: text-to-image generation in any style. In: Proceedings of the 37th International Conference on Neural Information Processing Systems. pp. 66860–66889 (2023)

39. Vinker, Y., Voynov, A., Cohen-Or, D., Shamir, A.: Concept decomposition for visual exploration and inspiration. ACM Transactions on Graphics (TOG) 42(6), 1–13 (2023)

40. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du, M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., Lin, J.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)

41. Wen, S., Fang, G., Zhang, R., Gao, P., Dong, H., Metaxas, D.: Improving compositional text-to-image generation with large vision-language models. arXiv preprint arXiv:2310.06311 (2023)

42. Wu, P., Xie, S.: V?: Guided visual search as a core mechanism in multimodal llms. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13084–13094 (2024)

43. Xu, C., Xu, Y., Zhang, H., Xu, X., He, S.: Dreamanime: Learning style-identity textual disentanglement for anime and beyond. IEEE Transactions on Visualization and Computer Graphics (2024)

44. Yang, Y., Nushi, B., Palangi, H., Mirzasoleiman, B.: Mitigating spurious correlations in multi-modal models during fine-tuning. In: International Conference on Machine Learning (ICML). pp. 39365–39379. PMLR (2023)

45. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image difusion models. In: ICCV. pp. 3836–3847 (2023)

46. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)

47. Zhang, Y., Yang, M., Zhou, Q., Wang, Z.: Attention calibration for disentangled text-to-image personalization. In: CVPR. pp. 4764–4774 (2024)

48. Zheng, G., Ye, W., Zhang, A.: Learning robust classifiers with self-guided spurious correlation mitigation. In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI). pp. 5599–5607 (2024)

49. Zhong, W., Yang, H., Liu, Z., He, H., He, Z., Niu, X., Zhang, D., Li, G.: Modadapter: Tuning-free and versatile multi-concept personalization via modulation adapter. arXiv preprint arXiv:2505.18612 (2025)

## S.1 Experimental Results on X-MULTI: Existing Factor Combinations

Having evaluated X-MULTI on the core objective of novel factor combinations, this section provides an extended evaluation of X-MULTI on existing factor combinations that are combinations present in the training data to demonstrate that our semantic supervision does not compromise performance on the training distributions.

Table S.1 demonstrates that when testing existing factor combinations, SDXL Zeroshot struggled to represent imaging factors accurately. While MULTI, Dream-Booth, and Inspiration Tree showed competence, our method excelled at generating images that most faithfully matched their source datasets. Notably, X-MULTI captured specialized sensors like gated, rgb-thermal, and event well, yet our approach’s representations were superior. Table S.2 compares baseline methods across image generation metrics for existing combinations. The results show that our method has best I-FAA, indicating that it achieves good factor disentanglement, while it performs comparable across other metrics.

## S.2 Augmentations of Factor Classifiers for I-FAA

To prevent the factor classifiers from relying on spurious correlation shortcuts inherent in the DF-RICO benchmark, we enforce strict, non-overlapping boundaries for the data transformations across each individual head. Table S.3 details the exact operational parameter bounds utilized during training. For instance, color deviations are completely restricted from the Sensor classifier to ensure it focuses strictly on physical noise distributions and exposure characteristics, while geometric radial limits are maximized on the Lens classifier to highlight fisheye edge-distortions without confounding viewpoint perspectives.

## S.3 Additional Correlation Analysis on FAA and I-FAA

To provide a deeper diagnostic evaluation of the structural limitations of the baseline Factor Alignment Accuracy (FAA) and validate the independence of our proposed Improved-FAA (I-FAA) metric, we conduct an extended correlation analysis across two distinct paradigms. Specifically, we examine the resilience of the classifiers against structural domain changes via cross-factor accuracy variance, and inspect the independence of prediction failures through error cooccurrence patterns.

Table S.1: Qualitative comparison of existing combinations. X-MULTI (ours) faithfully reproduces all target sensor, lens, and domain modalities present in the training distribution.  
![](images/2d102ebc143244bdb5d035e755956be3ca85d1c3695b31d8f9be8751b8e4db42.jpg)

## S.3.1 Cross-factor accuracy variance

Figure S.1 shows prediction accuracy of one factor conditioned on the groundtruth class of another. I-FAA maintains near-100% accuracy regardless of conditioning class, while FAA degrades sharply. Lens accuracy conditioned on viewpoint ranges from ∼0% to 75% in FAA, yet stays at 100% under I-FAA. Similar instability appears for viewpoint and sensor, where FAA drops below 5% for certain classes. FAA shows standard deviations as high as 37.0%.

## S.3.2 Error co-occurrence

Figure S.2 shows the conditional probability P(factor B wrong | factor A wrong), revealing whether prediction failures are shared or isolated. FAA prediction failures are almost never isolated: P(view wrong | lens wrong) = 0.98, P(sensor wrong | lens wrong) = 0.91, and P(lens wrong | domain wrong) = 0.99, indicating that factor predictions are not independent. I-FAA errors are nearly fully isolated, with all probabilities at or near 0.

Table S.2: Quantitative comparison on existing factor combinations. Best in bold. X-MULTI (ours) achieves the highest I-FAA metric while maintaining highly competitive underlying image generation quality scores.
<table><tr><td rowspan="3">Method</td><td colspan="6">Image Generation Metrics</td><td rowspan="2"></td><td colspan="5">Improved-Factor Alignment Accuracy (I-FAA)</td></tr><tr><td rowspan="2">FID [17] ↓ IS [34] ↑</td><td rowspan="2"></td><td colspan="4">CLIP Score [16] ↑</td><td rowspan="2">DS [46] ↑</td><td rowspan="2">Lens ↑ Sensor ↑</td><td rowspan="2"></td><td rowspan="2">Domain ↑ View ↑</td><td rowspan="2">Avg. ↑</td></tr><tr><td>factor</td><td>context</td><td>average</td><td>full-prompt</td></tr><tr><td>DreamBooth [32]</td><td>58.27</td><td>5.02</td><td>24.52</td><td>23.75</td><td>24.64</td><td>24.86</td><td>0.57</td><td>1.00</td><td>0.91</td><td>0.96</td><td>0.80</td><td>0.91</td></tr><tr><td>SDXL Zeroshot [30]</td><td>60.19</td><td>4.36</td><td>20.70</td><td>24.68</td><td>21.19</td><td>25.13</td><td>0.59</td><td>0.65</td><td>0.71</td><td>0.65</td><td>0.36</td><td>0.59</td></tr><tr><td>Inspiration Tree [39]</td><td>55.35</td><td>5.08</td><td>19.78</td><td>21.89</td><td>20.84</td><td>23.78</td><td>0.59</td><td>0.60</td><td>0.64</td><td>0.68</td><td>0.42</td><td>0.58</td></tr><tr><td>MULTI [13]</td><td>60.13</td><td>4.68</td><td>23.82</td><td>24.77</td><td>24.29</td><td>27.30</td><td>0.56</td><td>0.98</td><td>0.90</td><td>0.94</td><td>0.82</td><td>0.91</td></tr><tr><td>X-MULTI (ours)</td><td>54.48</td><td>4.90</td><td>24.37</td><td>24.97</td><td>24.66</td><td>28.08</td><td>0.59</td><td>1.00</td><td>0.91</td><td>0.96</td><td>0.82</td><td>0.92</td></tr><tr><td>DreamBooth-Canny</td><td>58.39</td><td>4.54</td><td>24.13</td><td>25.64</td><td>25.39</td><td>28.27</td><td>0.57</td><td>0.99</td><td>0.92</td><td>0.98</td><td>0.84</td><td>0.93</td></tr><tr><td>SDXL Zeroshot-Canny</td><td>63.24</td><td>4.76</td><td>23.96</td><td>27.34</td><td>25.65</td><td>28.92</td><td>0.59</td><td>1.00</td><td>0.73</td><td>0.93</td><td>0.76</td><td>0.85</td></tr><tr><td>Inspiration Tree-Canny</td><td>48.45</td><td>5.21</td><td>20.81</td><td>26.80</td><td>23.80</td><td>29.53</td><td>0.59</td><td>0.95</td><td>0.78</td><td>0.81</td><td>0.75</td><td>0.82</td></tr><tr><td>MULTI-Canny</td><td>45.39</td><td>4.69</td><td>24.64</td><td>25.90</td><td>25.27</td><td>28.33</td><td>0.55</td><td>0.99</td><td>0.91</td><td>0.97</td><td>0.85</td><td>0.93</td></tr><tr><td>X-MULTI-Canny (ours)</td><td>47.72</td><td>4.67</td><td>24.81</td><td>26.52</td><td>25.67</td><td>29.68</td><td>0.57</td><td>1.00</td><td>0.91</td><td>0.98</td><td>0.85</td><td>0.94</td></tr><tr><td>DreamBooth-Depth</td><td>59.35</td><td>4.87</td><td>24.06</td><td>25.17</td><td>25.11</td><td>27.18</td><td>0.57</td><td>1.00</td><td>0.92</td><td>0.97</td><td>0.82</td><td>0.92</td></tr><tr><td>SDXL Zeroshot-Depth</td><td>65.39</td><td>4.75</td><td>23.59</td><td>27.42</td><td>25.50</td><td>28.23</td><td>0.59</td><td>1.00</td><td>0.75</td><td>0.92</td><td>0.73</td><td>0.85</td></tr><tr><td>Inspiration Tree-Depth</td><td>55.15</td><td>5.15</td><td>20.62</td><td>26.38</td><td>23.50</td><td>28.20</td><td>0.59</td><td>0.95</td><td>0.79</td><td>0.81</td><td>0.75</td><td>0.82</td></tr><tr><td>MULTI-Depth</td><td>60.60</td><td>4.79</td><td>24.38</td><td>25.75</td><td>25.06</td><td>27.93</td><td>0.55</td><td>0.99</td><td>0.92</td><td>0.97</td><td>0.85</td><td>0.93</td></tr><tr><td>X-MULTI-Depth (ours)</td><td>55.58</td><td>4.66</td><td>24.63</td><td>26.13</td><td>25.38</td><td>28.81</td><td>0.53</td><td>1.00</td><td>0.92</td><td>0.97</td><td>0.85</td><td>0.93</td></tr></table>

Table S.3: Detailed operational parameter ranges for factor classifiers of I-FAA (ours).
<table><tr><td>Factor</td><td></td><td>Augmentation Operator Exact Transformation</td></tr><tr><td rowspan="6"></td><td>Random Rotation</td><td>Max ±5°(Probability  $p = 0 . 5 )$ </td></tr><tr><td>tion</td><td>Random Affine Transla- Max ±3% horizontally/vertically  $( p = 0 . 5 )$ </td></tr><tr><td>Color Jittering</td><td>Brightness/Contrast/Saturation:  $\pm 0 . 2 ,$  Hue:  $\pm 0 . 1 \ ( p = 0 . 5 )$ </td></tr><tr><td>Grayscale Conversion</td><td>Probability  $p = 0 . 1$ </td></tr><tr><td>Gaussian Blur</td><td>Kernel Size: 5 × 5, Sigma Range: [0.1, 2.0] (p = 0.5)</td></tr><tr><td>Noise &amp; Occlusions</td><td>Gaussian Noise (Std: 0.05), Random Erasing  $( p = 0 . 1 ) ,$  Cutout HideAndSeek (  $\mathrm {  ~ \dot { \mathfrak { j } } ~ } \times \mathrm {  ~ 6 ~ } \mathrm { g r i d } , \mathrm {  ~ \dot { \mathfrak { p } } ~ } = 0 . 1 )$ </td></tr><tr><td rowspan="6">Sensor</td><td>Random Perspective</td><td>Distortion Scale: 0.35 (Probability  $p = 0 . 4 )$ </td></tr><tr><td>Optical Distortion</td><td>Distortion Limit: ±0.12, Shift Limit: ±0.03  $( p = 0 . 3 ,$  applied inside Albu-</td></tr><tr><td></td><td>mentations transform with  $p = 0 . 5 )$   $( p = 0 . 5 )$ </td></tr><tr><td>Spatial Adjustments</td><td>Rotation Max: ±5°, Affine Translation Max: ±10%</td></tr><tr><td>Spatial Cropping Occlusions &amp; Erasures</td><td>Cropped to square via RandomCropToSquare Cutout Length: 6 px  $( p = 0 . 0 5 )$ </td></tr><tr><td></td><td>, Random Erasing Scale: [0.005, 0.03] (p = 0.05), HideAndSeek (8 × 8 grid,  $p = 0 . 0 5 )$ </td></tr><tr><td rowspan="6">Domain</td><td>Color Modification</td><td>Brightness/Contrast/Saturation: ±0.2, Hue: ±0.1  $( p = 0 . 5 )$ </td></tr><tr><td>Grayscale Conversion</td><td>Probability p = 0.1</td></tr><tr><td>Gaussian Blur</td><td>Heavy Blur (Kernel Size: 7 × 7, Sigma Range: [1.0, 3.0],  $p = 0 . 5 )$  Perspective Scale: 0.35  $( p \ = \ 0 . 4 )$  Albumentations Optical Distortion:</td></tr><tr><td>Geometric Distortions</td><td>±0.12 (p = 0.3)</td></tr><tr><td>Spatial Scaling &amp; Noise</td><td>Affine Translation: ±10%, Gaussian Noise Std: 0.05  $( p \ = \ 0 . 5 ) ,$  Random Erasing (p = 0.1), Cutout (p = 0.1), HideAndSeek (p = 0.1)</td></tr><tr><td>Horizontal Flipping</td><td>Restricted to conservative probability  $p = 0 . 1$ </td></tr><tr><td rowspan="4">Viewpoint</td><td></td><td>Random Rotation &amp; Affine Rotation Max: ±5°, Affine Translation Max: ±3%  $( p = 0 . 5 )$ </td></tr><tr><td>Color Modification</td><td>Brightness/Contrast/Saturation: ±0.2, Hue:  $\pm 0 . 1 ( p \ = \ 0 . 5 )$ </td></tr><tr><td></td><td>(p = 0.1) Spatial Blurring &amp; Distor- Gaussian Blur (p = 0.2, Sigma: [0.1, 1.0]), Albumentations Optical Dis-</td></tr><tr><td>tion</td><td>tortion: ±0.03, Shift Limit: ±0.01  $( p \ = \ \mathbf { \bar { 0 } } . 1 5 ) ,$  Gaussian Noise Std: 0.05 (p = 0.5)</td></tr></table>

![](images/2202148904c3ad9007822b0292029bfd5ed82c1cf61dbc4d764feb564e6ae34e.jpg)  
Fig. S.1: Cross-factor accuracy variance analysis. Individual factor prediction accuracies evaluated under ground-truth class conditioning. I-FAA (ours) has less crossfactor variance.

![](images/73e6041a2be74228920ce5b33b4f94072f1e3524a65e442effadc3e5d96940f1.jpg)

![](images/183f02e1798eb339b5eaa27909b0a8716681eeb14823c45b974110e7d946c25b.jpg)  
Fig. S.2: Error co-occurrence patterns. Conditional joint error probabilities map the independence of classification failures, showing I-FAA (ours) has low co-occurrence.

Table S.4: Extended qualitative comparison of attention maps between FAA and I-FAA.
<table><tr><td>Image</td><td>Metric</td><td>Lens</td><td>Sensor</td><td>Domain</td><td>View</td></tr><tr><td></td><td>FAA [13]</td><td>a</td><td></td><td>B</td><td>C</td></tr><tr><td>fisheye/rgb/real/pole</td><td>I-FAA</td><td>0</td><td>D</td><td>0</td><td>9</td></tr><tr><td>fisheye/rgb/real/front</td><td>FAA [13]</td><td>🌌 </td><td></td><td>3</td><td>2 </td></tr><tr><td></td><td>I-FAA</td><td>区</td><td>医</td><td>B R</td><td>A</td></tr><tr><td>normal/rgb/real/front</td><td>FAA [13] I-FAA</td><td>一</td><td>四</td><td>R</td><td>二</td></tr><tr><td></td><td>FAA [13]</td><td>2</td><td>2</td><td>國</td><td>M</td></tr><tr><td>fisheye/rgb/real/back</td><td>I-FAA</td><td>A</td><td>A</td><td>🌌</td><td></td></tr><tr><td></td><td>FAA [13]</td><td>M</td><td>S</td><td></td><td>区</td></tr><tr><td>normal/thermal/real/front</td><td>I-FAA</td><td>國</td><td>國</td><td>一</td><td>国</td></tr></table>