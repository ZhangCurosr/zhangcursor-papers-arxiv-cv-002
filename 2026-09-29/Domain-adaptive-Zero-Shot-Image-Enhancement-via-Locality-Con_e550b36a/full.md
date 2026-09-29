# Domain-adaptive Zero-Shot Image Enhancement via Locality-Constrained Difusion Guidance

Theresa Neubauer<sup>a</sup>, Dimitrios Lenis<sup>a</sup>, Astrid Berg<sup>a</sup>, Maria Wimmer<sup>a</sup>, Gaia Romana De Paolis<sup>a</sup>, Philip Matthias Winter<sup>a</sup>, David Major<sup>a</sup>, Johannes Novotny<sup>a</sup>, Ariharasudhan Muthusami<sup>a</sup>, Katja Bühler<sup>a</sup>

<sup>a</sup>VRVis GmbH, Vienna, Austria

## Abstract

Denoising Difusion Probabilistic Models have shown remarkable performance in unconditional image generation. In order to generate images with desired semantics, recent works have restricted the solution space by using guidance constraints in the difusion sampling process.

However, for image enhancement across diferent domains, these methods struggle to balance two main requirements: looking realistic in the target domain (photorealistic images) and preserving relevant features of the source domain, e.g., low-quality renderings or art paintings. Here, small local changes can alter the fidelity of the image completely, while large changes in other regions might be insignificant.

We introduce LocDif, a locality-constrained guidance method for image enhancement, which serves as a zero-shot extension to pre-trained difusion models, ensuring the preservation of critical features during domain adaptation. In this way, we retain important local features, while allowing less critical regions to remain unconstrained and not interfere with the guidance process for relevant regions. We evaluate our method on two diferent domain-shift tasks: For art-to-photo translation, we apply the method in a fully zero-shot setting, preserving facial identity from paintings while generating photorealistic details. For enhancing low-quality fetal ultrasound renderings, we demonstrate zero-shot inference with auxiliary prior alignment. Here, the objective is to artificially add high-resolution characteristics

and produce photorealistic ultrasound renderings, a target domain for which no ground truth distribution exists. Our experimental results demonstrate that LocDif achieves favorable realism-faithfulness trade-ofs compared to state-of-the-art methods, enabling controllable cross-domain enhancement.

Keywords: Controllable Image Generation, Face Image Enhancement, Low-Quality Image Restoration, Difusion Models, Generative AI, Fetal Ultrasound, Art Painting

![](images/84335507f171d3c4a656b0d285ab448452f2758195ff1007fa43d86b2f112689.jpg)  
Figure 1: Our proposed locality-constrained guidance method named LocDif controls the generation process of difusion models with region-adaptive conditions, allowing the user fine-grained control of image enhancement, even under domain shifts. Left: We enhance pre-trained difusion models with LocDif to preserve facial characteristics of low-quality fetal ultrasound renderings. Right: Locality-constrained guidance to enhance face parts of art paintings with diferent strengths (constrained face parts in brackets).

## 1. Introduction

Deep learning techniques have revolutionized image generation, opening up new ways of creative expression to the general public without the need for technical expertise (Stable Difusion [1], DALL-E 2 [2]). While these tools often generate content from text prompts describing the desired outcome, there is also a strong need for enhancement of existing images, made possible through techniques from inpainting and editing to super-resolution [3, 4]. Current Generative Adversarial Networks (GAN) and difusion-based meth ods are producing images that appear increasingly realistic (realism) [1]. However, for successful image enhancement of a source image, it is essential that resulting images preserve the distinctive features of the source (faithfulness) [4], like structural and color details. Existing approaches for photorealistic image enhancement often struggle to meet these requirements simultaneously [5, 6], with methods trading of between generating realistic details and preserving source identity. GAN-based methods such as conditional GANs and GAN inversion methods ofer limited control to preserve sourcedomain specific semantics [7, 8]. Additionally, they lack robustness when exposed to domain shifts, necessitating retraining on the new domain [9, 10, 11].

Recent advances in difusion-based zero-shot image generation leverage the generative prior of pre-trained difusion models with additional guidance, enabling finer control of the generation process without the need for retraining [12, 13]. However, these methods primarily prove efective for inputs that align with the domain of the trained difusion prior, such as denoising blurred face images originating from the high-resolution portrait domain [12], while neglecting input images exhibiting significant domain shifts.

For image enhancement, a low-quality input image is provided, with the objective to enhance it with high-quality visual features. In contrast to the regular difusion process, starting from random noise, the low-quality image is introduced at an intermediate sampling step with noise levels corresponding to this step. This process makes models robust to various types of degradation and to some extent acts as domain adaptation [14]. However, images that exhibit significant distribution shifts from the difusion prior, which is in most cases the photo-realistic image domain, typically require an increased number of difusion steps to achieve high-resolution features, as higher noise levels are necessary to align the input image with the learned distribution. These higher noise levels can compromise faithfulness, leading to weaker correspondence of fine details and diminished preservation of crucial structures, such as faces [13]. Therefore, it is essential to implement guidance mechanisms that preserve the underlying structure as accurately as possible while producing a realistic natural image.

We propose a difusion guidance mechanism for domain-adaptive zeroshot image enhancement on unseen target domains. Following [15, 16], we define zero-shot learning as inference on target domains that remain entirely unobserved during training, crucially without any access to domain-specific samples. We use domain-adaptive to describe adjusting a model’s implicit prior under this zero-shot constraint. While the proposed guidance operates without any retraining of the difusion model, we show that under strong semantic mismatch (fetal ultrasound renderings), performance can be substantially improved through an optional prior-alignment step using auxiliary real-image datasets. Importantly, this alignment step does not involve any target-domain data, and inference on the target domain remains zero-shot.

In this paper, we propose LocDif, a method to balance realism and faithfulness requirements of image generation methods, particularly under domain shifts. Unlike existing zero-shot image enhancement techniques that rely on guidance applied uniformly across the entire image, we address domain shift tasks by introducing locality-constrained guidance. This approach enables region-specific flexibility, allowing for diverse conditioning methods and varying guidance strengths to be applied across diferent image areas. In this way, we can ensure that important local features are preserved, while allowing less critical regions, such as the background, to remain unconstrained and not interfere with the guidance process for other relevant regions.

## Contributions:

• We introduce LocDif, a novel locality-constrained guidance method. Through utilizing flexible conditioning regions and an adaptive sampling schedule for individual image regions, our method acts as a zeroshot extension to the reverse difusion process, able to selectively preserve critical details during domain adaptation from the input to the difusion prior domain.

• Our ablation results indicate that we can precisely target specific image regions by independently modulating the guidance for this region in the reversed difusion process.

• We demonstrate the efectiveness of LocDif across distinct domains: 1) art-to-photo and 2) rendering-to-photo. We evaluate LocDif using image restoration, where the aim is to preserve the facial identity and relevant features of the source domain, while at the same time adding high-frequency details of the photorealistic domain. Our experimental results demonstrate that LocDif performs competitively with state-of-the-art methods on art-to-photo and rendering-to-photo tasks, achieving efective realism-faithfulness trade-ofs on both art painting and ultrasound rendering datasets.

• LocDif supports both fully zero-shot scenarios (art-to-photo without fine-tuning) and zero-shot inference with auxiliary prior alignment (ultrasound task with semantic space alignment), ensuring flexibility across domains without requiring target-domain training data.

## 2. Related Work

Guidance Strategies for Difusion Models. Denoising Difusion Probabilistic Models [17] are known for generating impressive photorealistic images from random noise. In the forward difusion process, Gaussian noise is iteratively added to an input image. In the reverse sampling process, the difusion model learns to reverse the forward process and thereby iteratively denoises the data, aiming to reconstruct the original image. Previous studies indicate that difusion models are inherently adept at learning the data manifold [18, 19]. For their guidance strategies, these works utilize the assumption that high-dimensional data points (such as images) form an (assumed linear) low-dimensional manifold [20]. Typically, this results in a two-step approach [21, 22, 23]: a denoising step, which performs an orthogonal projection onto the data manifold, followed by a guidance step that advances tangentially along the current manifold. Recent works have focused on developing guidance strategies for pre-trained difusion models to adapt them to diferent downstream tasks, using various guidance concepts such as text, styles, sketches, or facial attributes [24, 25, 26, 27, 23].

To selectively guide specific regions of an image, some methods employ binary masks, such as in inpainting [28] or text-to-image tasks [29]. However, these approaches ofer limited control over the enhancement process. For example, MultiDifusion [29] uses binary masks to guide image generation with text prompts, starting from random noise, thus can’t be used for enhancing existing images. RePaint [28] applies binary masks during the reverse difusion process to inpaint the selected regions with new content, while leaving the rest of the image unchanged. Critically, these binary masking approaches operate in an all-or-nothing manner, regions are either fully constrained or fully unconstrained, and lack mechanisms to preserve structural consistency within guided regions. Our approach difers by enhancing the entire image with adaptive, region-specific guidance strength that enables fine-grained control over the realism-faithfulness trade-of, preserving both structure and visual coherence across all regions.

Difusion Prior for Image Enhancement with Domain Shifts. Existing methods for image enhancement [6, 30, 31] or image editing [3, 4] show that low quality input images (from a diferent domain) can be mapped to the high-quality image domain of a pre-trained unconditioned difusion model by adding various conditions to the difusion sampling process. Various approaches have been proposed: (1) Classifier guidance adjusts the intermediate steps in the sampling process, by incorporating the gradient of the log-likelihood derived from an auxiliary classifier [32], which has demonstrated efectiveness in tasks such as image enhancement [33, 27]. Nevertheless, classifier guidance lacks generalizability to unseen classes and introduces considerable training overhead, limiting its suitability for domain adaptation tasks. (2) Other methods [21, 14, 13, 12, 34] use the low-quality input image to guide pre-trained difusion models by adjusting the intermediate steps in the sampling process, aligning the intermediate output more closely to the guide. Recent approaches have also explored frequency-domain decomposition, combining low-frequency and high-frequency priors through wavelet transforms [35]. (3) An alternative approach to constrain the solution space of the difusion model involves starting the reverse sampling process at an intermediate step, rather than beginning with random noise. Specifically, the process starts with the low-quality image and adds noise corresponding to the intermediate step [4, 6]. Consequently, images exhibiting mild degradation will require fewer difusion steps, whereas images with stronger degradation or significant domain shifts will necessitate more steps [36].

These guidance methods apply their strategies uniformly across the entire image. However, for domain adaptation, not all regions of the image carry the same importance. Therefore, we propose adapting constraints for each region based on its relevance to ensure faithfulness in the enhanced image. This approach preserves even small local features while allowing less critical areas to remain unconstrained, ensuring better adaptation to the difusion prior’s domain.

Image Enhancement for Faces. A variety of face enhancement techniques employ face-specific constraints to direct the generation process, thereby facilitating the generation of enhanced facial features, such as generative priors [5, 37, 34], facial landmarks or parsing maps [38, 39], or codebooks with mappings from low quality and high quality images as reference priors [40, 41, 42], or image editing with text-prompts [43, 44]. However, GAN-based restoration models often perform poorly when the input image or degradation type difers from what they were trained on. The work by Kuai et al. [45] enhances the model’s robustness by training directly on real-world degraded images. However, if the target domain deviates from photorealistic imagery, retraining is necessary to maintain the enhancement quality. In contrast, recent studies [6, 46, 36, 14, 34] leveraging difusion priors demonstrate higher robustness to small domain shifts, yielding visually realistic results. Degraded or stylistically diferent images, which exhibit significant distribution shifts from the photorealistic face domain, typically require higher noise levels (i.e., more difusion steps) to produce satisfactory results within the target distribution [36]. However, increased noise can result in weaker anatomical correspondence. Given the ability of humans to discern slight diferences in facial features, even small diferences can result in a diferent face [47]. To preserve facial identity, it is crucial to maintain fine-grained structural information in key regions, such as eyes, nose, and mouth. Blind face restoration methods, as discussed in [48, 33, 49], employ various techniques to preserve facial identities. However, these methods have been exclusively designed for the photorealistic domain. In contrast, our proposed method is designed to handle domain shifts in the context of image enhancement.

## 3. Method

We address the challenge of enhancing low-quality images from a shifted domain (e.g., artistic paintings, ultrasound renderings) by leveraging pretrained difusion models trained on high-quality photorealistic images. Our method, LocDif (Locality-Constrained Difusion), introduces region-specific guidance during the reverse difusion process. Instead of applying uniform constraints across the entire image, which either over-preserve the input (losing realism) or over-enhance it (losing identity), we apply constraints with diferent guidance strengths to diferent facial regions. This is especially relevant for translation between domains, where diferent regions of an image require diferent levels of enhancement: critical facial features should be preserved to maintain faithfulness, while less important areas can be freely enhanced to achieve photorealism. Through disjoint binary masks, our method makes it possible to apply strong preservation constraints to regions critical for identity like eyes, while background regions receive weak constraints.

We now formalize this approach and provide the technical details.

Problem Setup. We assume possession of a small set of low-quality images $\mathcal { V } : = \{ y \} _ { i = 1 } ^ { N }$ , that we aim to enhance. As is common for real-world scenarios, no corresponding high-quality ground-truth is available (cf. Fig. 1). We propose to leverage a pre-trained difusion model trained on a high-quality target distribution $p _ { \mathrm { d a t a } } ( x )$ , assuming semantic continuity despite a clear distribution shift between $\mathcal { V } \sim p ( y )$ and $p _ { \mathrm { d a t a } }$ . For example, ultrasound renderings of fetuses and photorealistic baby photographs show similar subjects but distinct visual characteristics like skin texture. Our goal is to enhance the quality of a sample $y \sim p ( y )$ , by mapping it towards $p _ { \mathrm { d a t a } }$ . This transformations aims to preserve the original characteristics of $y$ (the low frequency features) ensuring faithfulness, while enriching it with high-frequency characteristic of $p _ { \mathrm { d a t a } }$ , hence improving realism. For this, we propose locality-constrained guidance to bridge the shift between $p ( y )$ and $p _ { \mathrm { d a t a } } ,$ overcoming the unconstrained transform of the difusion prior (cf. Section 4.4 and Figs. 6 and 7).

Background on Difusion Models. For a given dataset $\mathcal { T } = \{ x _ { i } \} _ { i = 1 } ^ { N ^ { \prime } } \sim$ $p _ { \mathrm { d a t a } } ( x )$ , a difusion model can capture the implicit prior of the underlying data distribution $p _ { \mathrm { d a t a } }$ by aligning with the gradient of the log density (known as the score function $\nabla _ { \boldsymbol { x } } \log p _ { t } ( \boldsymbol { x } _ { t } ) )$ [21]. Conceptually, these models consist of a forward process (the difusion) of $T$ steps, and its reverse. Let $y \in \mathcal { V }$ be a sample of interest. For $x _ { i } \in \mathcal { I }$ we abbreviate $x : = x _ { i }$ , such that its iterations $x _ { i , t }$ simplify to $x _ { t } , t \in [ 0 , T ]$ , similarly for $y$ . In the forward process, $\{ p _ { t } \} _ { 0 \leqslant t \leqslant T }$ , the model progressively adds noise following the transition kernel $p ( x _ { t } \mid x _ { 0 } ) = \mathcal { N } ( x _ { 0 } \alpha _ { t } , \sigma _ { t } ^ { 2 } \mathbb { I } )$ (with $\alpha _ { t }$ and $\sigma _ { t }$ being scalar functions characterizing the noise schedule). Starting from data distribution $p _ { 0 } ( x ) = p _ { \mathrm { d a t a } }$ , this process gradually erodes the underlying data structure, eventually approaching the standard Gaussian distribution $p _ { T } ( x ) \approx \mathcal { N } ( 0 ; \mathbb { I } )$ as $T \to \infty$ The reverse process learns to denoise by fitting a neural network $s _ { \theta } ( x _ { t } , t )$ that approximates the score function, resulting in samples sharing the characteristics of the data-distribution $p _ { \mathrm { d a t a } }$ . In SOTA denoising difusion models, this is characterized by a stochastic diferential equation, which can serve as a generative model once the score function $\nabla _ { \boldsymbol { x } } \log { p _ { t } ( \boldsymbol { x } _ { t } ) }$ is estimated tractably [21]. This again is done by fitting a neural network $s _ { \theta } ( x _ { t } , t )$ via stochastic regression, leveraging $\mathbb { E } _ { t , x , p _ { t , x _ { 0 } } } \left[ \| p ( x _ { t } \mid x _ { 0 } - s _ { \theta } ( x _ { t } , t ) ) \| _ { 2 } \right] \left[ 2 3 \right]$ . A classic result building upon Tweedie’s formula [50] shows that an estimate for the clean data $x _ { 0 | t }$ can be derived through $x _ { 0 \vert t } ~ = ~ ( \sqrt { \alpha _ { t } } ) ^ { - 1 } ( x _ { t } - \sqrt { 1 - \alpha _ { t } } s _ { \theta } ( x _ { t } , t ) )$

## 3.1. Locality-Constrained Guidance

In this section, we introduce LocDif, a local manifold guidance method for controlling the reverse sampling process of an unconditional pre-trained difusion model. Our method extends the reverse difusion sampling process with a locality-constrained guidance step comprising two parts: (1) sampling the posterior of the pre-trained difusion model (ensuring realism), and (2)

![](images/e6266568e9603d9e74874e0b4d3e7e4759a40ef1660583e2c1c838cafa52f07f.jpg)  
Figure 2: Visualization of the locality-constrained guidance step. For each conditioning region $M _ { k }$ (user-defined or model generated binary masks), the distance function $d _ { k }$ measures the distance between the input $y$ and the predicted $x _ { 0 \mid t }$ restricted to $M _ { k } .$ . For each region $M _ { k }$ , the gradient of $d _ { k }$ with respect to $x _ { t }$ is computed independently and updates the current $x _ { t - 1 }$ exclusively within the region $M _ { k }$

refining it with a localized gradient to constrain the solution space to local manifolds that align with the input image (maintaining faithfulness). Figure 2 illustrates our method.

Posterior Sampling for Realism. To generate samples conditioned on input y, we rewrite the conditional score $\nabla _ { x } \log p _ { t } ( x _ { t } | y ) = \nabla _ { x _ { t } } { \log p ( x _ { t } ) } +$ $\nabla _ { \boldsymbol { x } _ { t } } \log { p ( \boldsymbol { y } | \boldsymbol { x } _ { t } ) }$ (Bayes). The first term is directly approximated by our pretrained difusion model $s _ { \theta } ( x _ { t } , t )$ by sampling from the posterior $p _ { \mathrm { d a t a } } ,$ through $\begin{array} { r } { x _ { t - 1 } = \left( - \frac { 1 } { 2 } \alpha _ { t } x _ { t - 1 } - \alpha _ { t } ( \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } ) + \nabla _ { x _ { t } } \log p _ { t } ( y \mid x _ { t } ) ) \right) + \sqrt { \alpha _ { t } } \sigma _ { t } \left[ 2 1 \right] } \end{array}$ , where α and σ are components of the noise schedule (cf. above), while the second term (the likelihood) guides the sample toward compatibility with y.

The noisy likelihood gradient serves as an optimization step through gradient descent to minimize the guidance loss in the neighborhood of the denoised sample $x _ { t }$ , pointing towards solutions compatible with y [21]. Following difusion posterior sampling, we approximate the noisy likelihood using the clean data estimate $\nabla _ { x _ { t } } \log p ( y \mid x _ { t } ) \simeq \nabla _ { x _ { t } } \log p ( y \mid x _ { 0 | t } )$ with a controllable upper bound [21]. In combination, this provides a per-timestep guidance given by $\nabla _ { x _ { t } } \log p _ { t } ( x _ { t } \mid y ) \approx s _ { \theta } ^ { * } ( x _ { t } , t ) - \zeta _ { t } \nabla _ { x _ { t } } \log p ( y \mid x _ { 0 \mid t } )$ , where $\zeta _ { t }$ scales the step size.

While this guides sampling towards realism within the difusion model’s distribution, uniform image-level conditioning allows the learned prior to dominate fine-grained details, potentially distorting critical structures and diminishing semantic correspondence.

As full image conditioning fails in this regard (cf. Fig. 3), we propose to

combine manifold constraining with locality-constrained guidance.

Locality-Constrained Guidance for Faithfulness.

The local guidance, LocDif, consists of three components: (i) a set of attribution maps M, (ii) a diferentiable distance measure $d ,$ and (iii) a sampling scheme for guidance timesteps $t _ { g } .$ . An attribution map $\mathcal { M } : = \dot { \{ M _ { k } \} } _ { k = 0 } ^ { N _ { m } }$ identifies a set of regions to constrain toward either the source or target domain. By selecting which timesteps $t _ { g }$ apply local guidance and choosing the distance function d, we control constraint strength from strong preservation (faithfulness) to weak constraints (realism).

Inspired by the product of experts formulations [51] where the target density is modeled as a product of individual components, we interpret our desired distribution as favoring configurations with high probability across all regions. Since $\begin{array} { r } { \nabla _ { x } \log \pi ( x ) = \sum _ { e } \nabla _ { x } \log \pi _ { e } ( x ) } \end{array}$ , which is compatible with standard sampling strategies [51], we refine the constrained denoised estimate $x _ { t - 1 }$ through the sum of region-specific guidance steps:

$\begin{array} { r } { x _ { t - 1 } - \sum _ { k = 0 } ^ { N _ { m } } \zeta _ { t } \nabla _ { x _ { t } } d ( y , x _ { 0 | t } , M _ { k } ) * M _ { k } } \end{array}$ , where $\zeta _ { t }$ is a time dependent dampening factor.

Adaptive Timestep Selection. The sampling scheme $t _ { g }$ is based on the observation that at some timestep $\tau ,$ the noisy input $y _ { \tau }$ aligns with noisy samples from $p _ { \mathrm { d a t a } } \colon \ x _ { \tau } \approx y _ { \tau }$ . Depending on the observed gap between the data manifolds of the input data and the training data for the difusion model, we carefully design our sampling scheme to only apply local guidance at selected timesteps $t _ { g }$ . Consequently, input images that are similar to the training distribution require fewer difusion steps (less noise) to bridge the gap between the two data manifolds, whereas images with significant domain shifts necessitate more steps [36]. As the forward process advances, highfrequency details are progressively eliminated, causing neighboring samples from distinct manifolds to converge in appearance. Adapting semantic features (low-frequency features) to the input y should therefore be performed at later timesteps, specifically $t \geq \tau$ . If the gap between the two data manifolds is too large (high τ), fine-tuning the difusion prior on a small subset of data that more closely aligns with the input domain can significantly improve image enhancement. Algorithm 1 describes our steps, where masks M in this algorithm are binary, disjoint, and zero on non-relevant regions.

Algorithm 1 Locality-Constrained Sampling Process   
Require: original image y, score network $s _ { \theta } ( \cdot , t )$   
distance function $d ( y , \cdot , M )$   
pre-defined parameters $\beta _ { t } , \bar { \alpha } _ { t } , \zeta _ { t } , \tilde { \sigma } _ { t }$   
list of conditioning regions $L = [ ( M , d , t _ { g } ) , \ldots ]$   
Constraint: masks M are binary and pairwise disjoint   
starting timestep $T _ { s t a r t }$ for $1 \leq T _ { s t a r t } < T$   
1: Sample $x _ { T _ { s t a r t } }$ from $q ( x _ { T _ { s t a r t } } | y )$   
2: for $t = T _ { s t a r t }$ to 1 do   
3: $\epsilon _ { t } \sim \mathcal { N } ( 0 , I )$   
4: $\begin{array} { r } { x _ { 0 | t } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } ( x _ { t } + ( 1 - \bar { \alpha } _ { t } ) s _ { \theta } ( x _ { t } , t ) ) } \end{array}$   
5: $\begin{array} { r } { x _ { t - 1 } ^ { \prime } = \frac { \sqrt { \bar { \alpha } _ { t } } \left( 1 - \bar { \alpha } _ { t - 1 } \right) } { 1 - \bar { \alpha } _ { t } } x _ { t } + \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } x _ { 0 \mid t } + \tilde { \sigma } _ { t } \epsilon _ { t } } \end{array}$   
6: for $( M , d , t _ { g } )$ in $L$ do   
7: if t in $t _ { g }$ then   
8: $g = \bar { \nabla } _ { x _ { t } } d ( y , x _ { 0 | t } , M ) * M$   
9: $x _ { t - 1 } = x _ { t - 1 } ^ { \prime } - \zeta _ { t } g * M$   
10: return $x _ { 0 }$

## 4. Experiments

This study evaluates the impact of our locality-constrained guidance on specific image regions. We demonstrate its efectiveness in art-to-photo and rendering-to-photo translation for face images, aiming to convert low-quality (art/ultrasound) images into photorealistic ones while preserving facial identity. We focus on faces since it is a challenging task and due to the availability of precise evaluation metrics to measure facial preservation.

## 4.1. Difusion Prior for Domain-Adaptive Zero-Shot Tasks

For artistic face portraits, we directly apply a difusion model pre-trained on high-quality face images (FFHQ dataset [11]), without any fine-tuning. This constitutes a fully zero-shot setting under strong domain shift.

For rendered fetal face enhancement, we observe that the FFHQ prior contains adult-specific modes that dominate the conditional generation and negatively impact the resulting baby face semantics. We therefore perform an auxiliary prior-alignment step using a small collected dataset (100 samples) of real high-resolution portrait images of babies, cf. Appendix A.1 for dataset details. We fine-tune $s _ { \theta }$ for 20,000 epochs following the training strategy of [6]. Importantly, this fine-tuning does not introduce target-domain information, as the envisioned target domain (enhanced ultrasound images with clearer facial details) does not exist in the real world and is never observed during training. In this case, the target domain is defined implicitly through a conditioning process and desired output constraints at inference time. In such settings, training on the target domain is infeasible, and inference is necessarily zero-shot with respect to the target. Due to the prior alignment, we label the task as “zero-shot inference with auxiliary prior alignment".

We utilize the framework for conditional difusion models from Dhariwal et al. [32]. To accelerate the difusion process, we employ the DDIM sampling strategy [52], reducing the number of difusion steps from 1000 to 250. Additionally, we leverage the pre-trained difusion model from DifFace [6] as score estimator $s _ { \theta }$ , trained on the FFHQ dataset [11] for image denoising tasks, and use the linear noise schedule from [6] for $\beta$

## 4.2. Datasets, Implementation Details, Metrics

Datasets for Evaluation. MetFaces [53] comprises 1,366 portrait images of human faces extracted from various artworks, including paintings and sculptures.

WikiArt-Faces consists of images of human faces extracted from paintings available on WikiArt.org. We curated all face portraits within the realism category, resulting in a dataset of 1,373 portrait images, cf. Appendix A.1.

The fetal ultrasound rendering dataset comprises 437 2D images, projected from 3D-rendered ultrasound volumes, showing fetuses in the second and third trimester. The dataset was acquired by our medical partner in accordance with ethical standards, and we have obtained consent to use it for research purposes in this study.

Conditioning Regions. Conditioning regions can be derived from suitable segmentation algorithms or manually specified by the user. For the art-tophoto task, we define five conditioning regions: (1) eyes, (2) eye brows, (3) mouth and nose, (4) skin, and (5) background (remaining image region). For the rendering-to-photo task, we define five conditioning regions: 1) eyes, (2) mouth, (3) nose, (4) skin, and (5) background. In this study, the attribution maps $M _ { 1 \ldots 5 }$ are generated using a face parsing model [54] (trained on the LaPa dataset [55]) that predicts binary masks for individual face parts. Each mask, excluding the background and skin, is extended by 20 pixels without overlap. However, the eye mask used for the rendering-to-photo task is extended by 40 pixels. The ablation study presented in Tab. 1 evaluates individual conditioning regions to examine their contributions to the guidance mechanism in the art-to-photo task. For the comparison to SOTA, we use all five conditioning regions (multi-region) for guidance.

Distance Function. For the image enhancement task, we design the distance function d similar to Chung et al. [21], but restrict the image region for the corresponding conditioning region with an attribution map M: $\mathbf { d } ( y , x _ { 0 \mid n } , M ) : = \| y * M - { \mathcal { A } } ( x _ { 0 \mid n } * M ) \| _ { 2 } ^ { 2 }$ . The operator A is defined as a Gaussian blur kernel with a size of $5 \times 5$ and standard deviation of 4. Using a low-pass filter is common for guidance methods with low-quality reference images to focus on coarser structures for distance calculation [27, 14, 21]. The distance function was selected empirically, as justified in Tab. 2.

Guidance Strength. To reduce the guidance strength, there are two strategies: (1) Set a lower value for the hyperparameter $\zeta _ { t }$ to decrease the influence of the guidance gradient, or (2) restrict the guidance to certain steps $t _ { g }$ in the reverse sampling process. We use $\zeta _ { t } = 1 . 0$ for all experiments and reduce the number of guidance steps to minimize the costly gradient calculations. We observe that image quality improves when guidance for certain time steps is skipped. Our observation aligns with the findings of Yu et al. [27] and Gu et al. [42], who noted that guidance in the last sampling steps (refinement stage) harms the development of high-frequency details. We define the time step schedule $t _ { g }$ for guidance of one conditioning region as: $t _ { g } = [ i \in \mathbb { N } \mid t _ { m i n } \le i \le T _ { s t a r t }$ and i ≡ 0 (mod m)]. Details regarding the art-to-photo and the rendering-to-photo task are described in Appendix A.4 and Appendix A.5.

Evaluation Metrics. To evaluate faithfulness for faces, we use the identity preservation metrics Identity Score (IDS) [56] and Landmark Distance (LMD) [6]. We adapt the SSIM metric to evaluate image similarity for facespecific regions, naming it SSIM-F. To assess realism we use the Fréchet Inception Distance (FID) [57] to measure the similarity between the feature distributions of our predicted images and the FFHQ dataset [11], a dataset consisting of high-quality face images. For the rendering-to-photo task, we use the collected baby portrait dataset for calculating the FID metric. We also evaluate image similarity using PSNR and LPIPS, which are widely employed for image enhancement. LMD was omitted for rendering-to-photo, as no reliable landmark detection model exists for fetal ultrasound data. To select the best hyperparameters for our locality-constrained guidance method, we define a metric to quantify the balance between realism and faithfulness: $\mathrm { R F } = \left( 1 - \mathrm { S S I M } \mathrm { - F } \right)$ · FID, where FID quantifies image quality and (1 − SSIM-F) measures content preservation. See Appendix A.3 for implementation details.

Hardware Setup. All experiments were conducted on a Linux Debian server equipped using an Intel Xeon Gold 5118 CPU and an Nvidia A100 GPU (40GB VRAM).

## 4.3. Ablation on Locality-Constrained Guidance

Conditioning regions. To demonstrate the efectiveness of the localityconstrained guidance, we perform an ablation study on the art datasets, in which conditioning is applied to diferent image regions. The following experiments were performed: (I) no locality-constrained guidance, (II) joint region for mouth and nose (single condition), (III) separate conditions for eye and eyebrow region (two conditions), (IV) full-image region (single condition) using the same $t _ { g }$ as in (II), (V) the proposed multi-region guidance: eye, eyebrow, mouth and nose, skin, and background regions. Individual LMD scores are calculated for eye, mouth, and nose separately, presented in Tab. 1. As expected, face-specific LMD scores increase for unconstrained face parts.

In Tab. 1 we observe the realism-faithfulness trade-of. The model without locality-constrained guidance achieves a high image quality score (FID), but low scores in terms of identity preservation (IDS, LMD, SSIM-F) and image similarity (LPIPS, PSNR). More extensive guidance enhances the model’s ability to maintain the original identity and similarity of the images, albeit at the expense of lower FID.

More importantly, our findings indicate that our proposed multi-region guidance (V) leads to higher identity preservation, while still maintaining enhanced image quality over the full-image guidance (IV). Figure 3 shows that the locality-constrained guidance excels in preserving the natural eye color of the original painting, while the reduced guidance (less guidance steps in $t _ { g } )$ in the background region enhances high-frequency details for hair. To visualize the influence of locality-constrained guidance during the reverse sampling process, we present the intermediate results of $x _ { 0 | t }$ in Fig. 4. Guidance via conditioning regions influences only the specified areas (Fig. 4b), allowing the remaining regions to evolve similarly to the unguided process (Fig. 4a). Notably, the final image output exhibits no visible boundary artifacts around the conditioned regions, but instead demonstrates a smooth and coherent transition between guided and unguided areas. Figure 5 contains further qualitative comparisons that explore how diferent conditioning regions impact the final result.

<table><tr><td colspan="5"></td><td colspan="2">faithfulness</td><td>realism</td><td colspan="2">traditional metrics</td></tr><tr><td>Conditioning</td><td>LMD↓</td><td>LMD eye↓</td><td>LMD mouth↓</td><td>LMD nose↓</td><td>SSIM-F↑</td><td>IDS↓</td><td>FID↓</td><td>LPIPS↓</td><td>PSNR↑</td></tr><tr><td colspan="10">WikiArt-Faces Dataset</td></tr><tr><td>(I) w/o guidance 8.28</td><td></td><td>7.78</td><td>8.61</td><td>6.80</td><td>0.41</td><td>65.98</td><td>56.22</td><td>0.36</td><td>22.33</td></tr><tr><td>(II) mouth &amp; nose</td><td>7.15</td><td>6.14</td><td>4.57</td><td>3.64</td><td>0.49</td><td>55.28</td><td>65.97</td><td>0.33</td><td>24.08</td></tr><tr><td>(III) eyes</td><td>7.58</td><td>3.76</td><td>8.13</td><td>5.61</td><td>0.50</td><td>57.20</td><td>64.82</td><td>0.33</td><td>23.68</td></tr><tr><td>(IV) full image</td><td>4.76</td><td>4.39</td><td>4.95</td><td>3.75</td><td>0.55</td><td>47.23</td><td>72.92</td><td>0.30</td><td>26.61</td></tr><tr><td>(V) multi-region</td><td>3.91</td><td>3.39</td><td>4.17</td><td>3.10</td><td>0.61</td><td>43.52</td><td>70.88</td><td>0.32</td><td>25.21</td></tr><tr><td colspan="10">MetFaces Dataset</td></tr><tr><td>(I) w/o guidance</td><td>8.72</td><td>8.77</td><td>8.67</td><td>6.90</td><td>0.38</td><td>68.59</td><td>49.19</td><td>0.31</td><td>21.54</td></tr><tr><td>(II) mouth &amp; nose</td><td>7.44</td><td>7.08</td><td>4.06</td><td>3.09</td><td>0.46</td><td>56.58</td><td>60.09</td><td>0.29</td><td>23.39</td></tr><tr><td>(III) eyes</td><td>7.71</td><td>3.32</td><td>8.61</td><td>5.91</td><td>0.48</td><td>58.00</td><td>57.73</td><td>0.29</td><td>23.02</td></tr><tr><td>(IV) full image</td><td>4.21</td><td>3.91</td><td>4.39</td><td>3.11</td><td>0.53</td><td>46.61</td><td>66.44</td><td>0.26</td><td>25.95</td></tr><tr><td>(V) multi-region</td><td>3.21</td><td>2.72</td><td>3.44</td><td>2.47</td><td>0.60</td><td>40.95</td><td>64.76</td><td>0.27</td><td>24.58</td></tr></table>

Table 1: Ablation study on the efectiveness of locality-constrained guidance for diferent image regions on WikiArt-Faces and MetFaces. Our proposed approach, localityconstrained guidance for multiple regions, demonstrates superior performance in terms of identity preservation, as evidenced by the LMD and IDS metrics. Furthermore, it exhibits an improvement in image quality (FID) compared to full image guidance. The best results are highlighted in bold.

Additional hyperparameters. Table 2 shows the conducted ablation study on the number of difusion steps $T _ { s t a r t }$ , the distance function d, and the guidance strength m for the time step guidance schedule $t _ { g }$ (lower m means stronger guidance) using the proposed multi-region conditioning from Section 4.3. For distance function d, we tested: $( \mathbf { d } _ { 1 } ) ~ 5 { \times } 5$ blur, (d<sub>2</sub>) 9×9 blur, (d<sub>3</sub>) no blur, and resizing function r with factors 0.25 (d<sub>4</sub>), 0.125 (d<sub>5</sub>), defined as $\| r ( y * M ) - r ( x _ { 0 | n } * M ) \|$

We provide qualitative results on varying starting time steps $T _ { s t a r t }$ presented in Fig. 6. As $T _ { s t a r t }$ increases, the generated images exhibit enhanced photorealism; however, this improvement is accompanied by a degradation in structural fidelity. For instance, at lower values of $T _ { s t a r t }$ , the model fails to accurately reconstruct high-resolution details for e.g. the left eye. Conversely, at $T _ { s t a r t } = 1 0 0$ , the left eye appears realistic, yet other structural elements, such as the tongue, are omitted. This phenomenon may be attributed to the scarcity of tongue-containing images in the training dataset. As $T _ { s t a r t }$ increases, granting the model greater generative freedom, the difusion prior tends to shift semantic content toward the dominant learned distribution. To mitigate this efect, our proposed multi-region guidance framework enables region-specific conditioning, allowing targeted control over individual image areas. In Fig. 6, we apply stronger guidance to preserve structural fidelity in the mouth region, while employing weaker guidance for the eye region to promote photorealism.

![](images/c696743dd9ab9b12e777c3ef74bcb2d249df2cdbdf34838e28774accddb69ab2.jpg)  
Figure 3: Compared to the full-image guidance (c), the locality-constrained guidance (b) preserves the natural eye colour and lip structure of the original painting (a), while the reduced guidance in the background region enhances high-frequency details for hair.

## 4.4. Comparison with State-of-the-Art Methods

We compare our proposed method LocDif with SOTA methods for the artto-photo and rendering-to-photo tasks. We employ DifFace [6] and PGDif [33] with the same model weights (FFHQ prior for art-to-image, fine-tuned baby face prior for rendering-to-photo) as LocDif to facilitate a fair comparison and to evaluate the benefit of locality-constrained guidance over classifier guidance and guidance without flexible region constraints. For the art-tophoto task, we further compare our approach with recent blind face restoration methods, including DifBIR [34] and DT-BFR [45], as well as with models specifically designed for art painting enhancement, such as Art2Real [58] and ILVR [14]. Additionally, we evaluate ControlNet[59], a controllable textto-image difusion method with structure-preserving guidance using the tilebased control method.

![](images/69d6a313b53fac8e462941939d0709028563066a9bb55c4dfe285906b51fa8e9.jpg)

![](images/35c12cf3a20c0535722e8fda4596036ec2a40aa84343b274b4bfa311bcb0e2cf.jpg)  
Figure 4: Demonstration of the locality-constrained guidance for face restoration of y. Images $x _ { 0 \mid t }$ are generated during the reverse sampling process using no locality-constrained guidance (a), locality-constrained guidance for eyes only (b), locality-constrained guidance for eye, eyebrow, mouth and nose, and skin (c).

![](images/3e6ecf22b563a49b2ef7dbb04c1ababb66422965e686f1deb32b80381d139255.jpg)

Figure 5: Art-to-photo: Qualitative results of our ablation study for single and multi condition guidance of specific face regions using our locality-constrained guidance. Each color in the attribution map indicates a separate image region that is individually condi tioned, where black regions are not guided by our locality-constrained guidance method.
<table><tr><td colspan="3">hyperparameter</td><td colspan="5">faithfulness</td><td colspan="2">realism</td><td colspan="2">overall</td></tr><tr><td> $T _ { s t a r t }$ </td><td>m</td><td>d</td><td>LMD↓</td><td></td><td>SSIM-F↑</td><td>IDS↓</td><td></td><td>FID↓</td><td></td><td>RF↓</td><td></td></tr><tr><td>60</td><td>2</td><td>d1</td><td>2.96</td><td>3.29 0.61</td><td>/0.63</td><td>35.2</td><td>36.7</td><td>72.3</td><td>81.3</td><td>28.2</td><td>30.0</td></tr><tr><td>100</td><td>2</td><td>d1</td><td>2.96</td><td>3.52 0.61</td><td>/0.62</td><td>36.5</td><td>39.1</td><td>67.1</td><td>72.3</td><td>26.5</td><td>27.4</td></tr><tr><td>140</td><td></td><td>d1</td><td>3.21</td><td>3.91 0.60</td><td>/0.61</td><td>41.0</td><td>43.5</td><td>64.8</td><td>70.9</td><td>26.1</td><td>27.4</td></tr><tr><td>180</td><td>2-2</td><td>d1</td><td>3.39</td><td>3.97 0.59</td><td>0.61</td><td>42.0</td><td>44.8</td><td>69.8</td><td>76.0</td><td>28.4</td><td>29.8</td></tr><tr><td>220</td><td>2</td><td> $\mathbf { d _ { 1 } }$ </td><td>3.29</td><td>3.91 0.60</td><td>0.61</td><td>42.5</td><td>44.9</td><td>66.1</td><td>74.0</td><td>26.6</td><td>28.7</td></tr><tr><td>140</td><td>1</td><td> $\mathbf { d _ { 1 } }$ </td><td>2.97</td><td>3.60 0.62</td><td>0.63</td><td>39.1</td><td>41.8</td><td>69.1</td><td>75.5</td><td>26.2</td><td>27.6</td></tr><tr><td>140</td><td>4</td><td> $\mathbf { d _ { 1 } }$ </td><td>3.97</td><td>4.37 0.56</td><td>/0.57</td><td>45.2</td><td>46.6</td><td>62.9</td><td>68.8</td><td>27.9</td><td>29.4</td></tr><tr><td>140</td><td>8</td><td> $\mathbf { d _ { 1 } }$ </td><td>5.07</td><td>5.20 0.52</td><td>0.54</td><td>52.3</td><td>52.3</td><td>61.4</td><td>67.9</td><td>29.6</td><td>30.9</td></tr><tr><td>140</td><td>2</td><td> $\mathbf { d _ { 2 } }$ </td><td>3.57</td><td>3.95 0.56</td><td>0.58</td><td>45.1</td><td>44.4</td><td>63.1</td><td>70.9</td><td>27.6</td><td>30.0</td></tr><tr><td>140</td><td>2</td><td> $\mathbf { d _ { 3 } }$ </td><td>3.15</td><td>3.74 0.62</td><td>0.64</td><td>40.6</td><td>43.0</td><td>68.4/</td><td>75.4</td><td>26.3</td><td>27.5</td></tr><tr><td>140</td><td>2</td><td> $\mathbf { d } _ { 4 }$ </td><td>5.23</td><td>5.28 0.51</td><td>/0.53</td><td>54.8</td><td>54.1</td><td>56.2</td><td>61.9</td><td>27.5</td><td>28.9</td></tr><tr><td>140</td><td>2</td><td> $\mathbf { d } _ { 5 }$ </td><td>6.64</td><td>6.31 0.45</td><td>0.48</td><td>61.0</td><td>59.0</td><td>53.0</td><td>58.8</td><td>28.9</td><td>30.7</td></tr><tr><td>140</td><td></td><td>no guidance</td><td>8.72</td><td>8.28 0.38</td><td>0.41</td><td>68.6</td><td>66.0</td><td>49.2</td><td>56.2</td><td>30.9</td><td>33.0</td></tr></table>

Table 2: Ablation study on the number of difusion steps $T _ { s t a r t } .$ , guidance strength $m ,$ and distance function d for the locality-constrained guidance, evaluated on MetFaces/WikiArt datasets. Stronger guidance (lower m indicates stronger guidance) improves faithfulness but harms realism. The best setting is underlined.

For SOTA implementations, we adopt the recommended settings and use the validation split for hyperparameter tuning (see Appendix A.1).

![](images/7b310d45cfc60d2d25a55f601a82be55e055c69d755f44cf0508afb2dd38f646.jpg)  
Figure 6: Comparison of reconstructed results without locality-constrained guidance across diferent starting time steps $( T _ { s t a r t } )$ illustrating the faithfulness-realism tradeof. At $T _ { s t a r t } = 6 0$ , eye appearance lacks realism, while at $T _ { s t a r t } = 1 0 0$ , mouth appearance lacks faithfulness. Our proposed locality-constrained guidance method, LocDif, preserves facial details and maintains realism.

Rendering-to-photo. Table 3 presents quantitative results on the ultrasound dataset, with visual comparisons in Fig. 7. Quantitative and qualitative results demonstrate that our method achieves superior facial detail preservation while maintaining quality (FID) on par with baselines.

Art-to-image. Table 4 presents quantitative comparisons on the Met-Faces and WikiArt-Faces datasets. Figures 8 and 9 show qualitative comparisons. While DifFace and PGDif yield photorealistic outputs with favorable FID scores, they fail to preserve identity, as illustrated in Fig. 8. Conversely, DifBIR [34] achieves high identity-preserving metrics, demonstrating reliable performance in image denoising of faces. However, DifBIR produces less photorealistic images, retaining fine brushstroke artifacts and thus exhibiting high FID, see Fig. 9 for further visual examples. While ControlNet adds photo-realistic details, it underperforms on identity-preservation compared to face-specific methods, reflecting the challenges of adapting text-to-image control methods to identity-preserving enhancement tasks. We observed that text-guided spatial control requires careful prompt engineering and may produce overly smooth outputs that lack the fine-grained detail preservation needed for face enhancement. As the objective of LocDif is to preserve salient source domain features, it is not feasible to achieve complete alignment with the target domain, resulting in a slightly higher FID score. Our LocDif model demonstrates a favourable trade-of of identity preservation and image quality, reflected by the second best metrics for LMD, SSIM-F, and IDS.

<table><tr><td colspan="3">faithfulness</td><td rowspan="2">realism</td><td colspan="2">traditional metrics</td></tr><tr><td>Methods</td><td>SSIM-F↑</td><td>IDS↓</td><td>FID↓  $\overline { { \mathrm { L P I P S } \downarrow } }$ </td><td> $\overline { { \mathrm { P S N R \uparrow } } }$ </td></tr><tr><td>DifFace [6]</td><td>0.73±0.06</td><td>51.0±7.9</td><td>74.8</td><td> $\overline { { 0 . 1 9 4 2 0 . 0 4 } }$ </td><td> $\overline { { 2 8 . 2 7 { \pm } 1 . 2 } }$ </td></tr><tr><td>PGDiff [33]</td><td>0.69±0.06</td><td>54.9±8.1</td><td>76.5</td><td> $\overline { { 0 . 2 4 0 { \pm } 0 . 0 5 } }$ </td><td> $\overline { { 2 6 . 6 1 \pm 1 . 1 } }$ </td></tr><tr><td>LocDiff (Ours)</td><td> $\mathbf { 0 . 7 8 { \overset { . } { \bot } } 0 . 0 5 }$ </td><td> ${ \bf 4 5 . 4 } \pm { \bf 7 . 7 }$ </td><td>75.2</td><td> $\mathbf { 0 . 1 8 8 } { \pm } \mathbf { 0 . 0 4 }$ </td><td> $\mathbf { 2 8 . 7 5 { \pm } 1 . 2 }$ </td></tr></table>

Table 3: Quantitative comparison on the fetal ultrasound rendering dataset. Best results in bold; second-best underlined.

<table><tr><td></td><td></td><td>faithfulness</td><td>realism</td><td>traditional metrics</td><td colspan="2"></td></tr><tr><td>Methods</td><td>LMD↓ SSIM-F↑</td><td>IDS↓</td><td>FID↓</td><td></td><td>LPIPS↓</td><td>PSNR↑</td></tr><tr><td colspan="7">WikiArt-Faces dataset</td></tr><tr><td>Art2Real [58] ILVR [14] DifFace [6]</td><td>6.6±8.1 4.5±5.7 5.1±7.2</td><td>0.50±0.1 0.56±0.1 0.53±0.1</td><td>51.5±11.3 46.0±7.1 50.8±7.6</td><td>130.4 108.6 66.3</td><td> $0 . 2 7 { \pm } 0 . 0 7$  0.26±0.06 0.30±0.05</td><td> $1 8 . 7 { \pm } 3 . 1 $   $2 6 . 7 { \pm } 2 . 0 \ $   $2 5 . 3 { \pm } 1 . 8 $ </td></tr><tr><td>PGDiff [33] DiffBIR [34]</td><td>4.7±7.0 2.3±4.8</td><td>0.51±0.1 0.59±0.1</td><td>47.7±7.7 27.9±8.4</td><td>68.2 95.6</td><td>0.31±0.06 0.39±0.10</td><td> $2 4 . 9 { \pm } 2 . 0 \ $  28.0±2.8</td></tr><tr><td>DT-BFR [45]</td><td>4.6±7.1</td><td>0.48±0.1</td><td>47.1±7.6</td><td>74.3</td><td>0.31±0.07</td><td> $2 5 . 3 { \pm } 2 . 0 $ </td></tr><tr><td>ControlNet [59]</td><td>5.1±7.1 3.9±7.3</td><td>0.42±0.0 0.61±0.1</td><td>52.2±7.7</td><td>92.7</td><td>0.40±0.08</td><td> $2 2 . 7 { \pm } 2 . 0 $ </td></tr><tr><td>LocDiff (Ours)</td><td></td><td></td><td>43.5±7.2</td><td>70.9</td><td>0.32±0.06</td><td> $2 5 . 2 { \pm } 2 . 3 $ </td></tr><tr><td></td><td> $5 . 1 { \pm } 3 . 5 $ </td><td>0.45±0.2</td><td>MetFaces dataset</td><td></td><td></td><td></td></tr><tr><td>Art2Real [58]</td><td></td><td></td><td>49.4±15.2</td><td>114.5 74.5</td><td>0.34±0.07 0.31±0.08</td><td> $1 7 . 5 { \pm } 3 . 9 $   $2 6 . 0 { \pm } 2 . 1 $ </td></tr><tr><td>ILVR [14] DifFace [6]</td><td> $4 . 4 { \pm } 1 . 8 $  5.3±2.7</td><td>0.53±0.1 0.50±0.1</td><td>46.7±7.9 52.2±8.9</td><td>59.1</td><td>0.27±0.06</td><td> $\overline { { 2 4 . 6 { \pm } 1 . 9 } }$ </td></tr><tr><td>PGDiff [33]</td><td>4.4±1.9</td><td>0.53±0.1</td><td> $4 7 . 4 \pm 7 . 9$ </td><td>66.1</td><td>0.26±0.05</td><td> $2 4 . 3 { \pm } 2 . 1 $ </td></tr><tr><td>DiffBIR [34]</td><td> ${ \bf 1 . 6 { \pm } 1 . 0 }$ </td><td>0.62±0.1</td><td> $\mathbf { 1 8 . 3 \pm 5 . 3 }$ </td><td>99.3</td><td>0.26±0.06</td><td> $\mathbf { 2 7 . 8 \pm 3 . 1 }$ </td></tr><tr><td>DT-BFR [45]</td><td>4.0±1.8</td><td>0.49±0.1</td><td>42.0±7.2</td><td>73.6</td><td>0.22±0.04</td><td> $2 5 . 0 { \pm } 2 . 2 $ </td></tr><tr><td>ControlNet [59]</td><td>4.4±1.6</td><td>0.42±0.1</td><td>48.4±7.3</td><td>97.9</td><td>0.36±0.06</td><td> $2 3 . 0 { \pm } 2 . 0 $ </td></tr><tr><td></td><td>3.2±1.6</td><td>0.60±0.1</td><td>41.0±7.6</td><td>64.8</td><td>0.27±0.06</td><td></td></tr><tr><td>LocDiff (Ours)</td><td></td><td></td><td></td><td></td><td></td><td> $2 4 . 6 { \pm } 2 . 3 $ </td></tr></table>

Table 4: Quantitative comparison to SOTA methods on WikiArt-Faces and MetFaces. Best results in bold; second-best underlined.

Original  
LocDiff (Ours)  
DifFace  
PGDiff  
![](images/c5e05277ced5d8d333da1054be3ddd44dec85a27b6902142f7101c5e7aac17f6.jpg)  
Figure 7: Rendering-to-photo: Qualitative comparison to baseline methods. LocDif preserves facial characteristics better, as evidenced by the highlighted regions. Zoom in for best view.

Computational Requirements. The locality-constrained guidance introduces a 1.4-1.7× overhead compared to baseline difusion sampling (Tab. 5), which we consider acceptable for ofline enhancement tasks prioritizing quality over speed. Total inference time (7.96-16.84s per image) remains competitive with other gradient-based methods.

<table><tr><td>Methods</td><td>Time (s)</td><td>Peak GPU Memory (GB)</td></tr><tr><td colspan="3">Art-to-photo task (WikiArt/MetFaces)</td></tr><tr><td>Art2Real</td><td> $0 . 1 4 \pm 0 . 0 1$ </td><td>8.3GB</td></tr><tr><td>ILVR</td><td> $0 . 6 4 \pm 0 . 0 2$ </td><td>2.7GB</td></tr><tr><td>DifFace</td><td> $7 . 2 0 \pm 0 . 1 7$ </td><td>5.8GB</td></tr><tr><td>PGDiff</td><td> $1 0 3 . 4 7 \pm 1 4 . 2 8$ </td><td>4.8GB</td></tr><tr><td>DifBIR</td><td> $1 5 . 8 2 \pm 0 . 5 9$ </td><td>13.9GB</td></tr><tr><td>DT-BFR</td><td> $0 . 1 7 \pm 0 . 0 1$ </td><td>1.0GB</td></tr><tr><td>ControlNet</td><td> $3 . 4 1 \pm 0 . 2 5$ </td><td>2.8GB</td></tr><tr><td>LocDiff (Ours)</td><td> $\overline { { 1 6 . 8 4 \pm 0 . 2 8 } }$ </td><td>6.2GB</td></tr><tr><td>Baseline diffusion (no LocDiff guidance)</td><td> $9 . 7 \pm 0 . 1 9$ </td><td>5.8GB</td></tr><tr><td colspan="3">Rendering-to-photo task (Ultrasound)</td></tr><tr><td>DifFace</td><td> $7 . 2 9 \pm 0 . 1 6$ </td><td>5.8GB</td></tr><tr><td>PGDiff</td><td> $1 0 1 . 7 4 \pm 1 4 . 5 0$ </td><td>4.8GB</td></tr><tr><td>LocDiff (Ours)</td><td> $\overline { { 7 . 9 6 \pm 0 . 1 8 } }$ </td><td>8.2GB</td></tr><tr><td>Baseline diffusion (no LocDiff guidance)</td><td> $5 . 5 \pm 0 . 1 5$ </td><td>5.8GB</td></tr></table>

Table 5: Computational requirements of baseline methods. Time: average inference time per sample.

Balancing Realism and Faithfulness. While the similarity metrics LPIPS and PSNR are useful for measuring overall image similarity, our primary interest lies in preserving facial identity. To this end, we are particularly concerned with the preservation of facial regions that are crucial for facial recognition. In the regions of eyes, nose, and mouth, even minor alterations can result in significant changes to the face, leading to unfaithful results. In other regions, such as hair or background, moderate diferences may even be advantageous and result in a faithful outcome, see Fig. 3. For instance, in the pursuit of a smooth skin efect, it may be preferable to prioritize a more seamless appearance over the preservation of minute structural details, such as brush strokes. Figure 5 visualizes that the LocDif model ensures a high structural similarity for eyes, nose, and mouth, but provides less guidance strength for the background and hair region. This results in a smoother background and high-frequency details for hairs, thereby adapting to the photorealistic domain. As previously stated, the realism-faithfulness tradeof in the context of domain shifts implies that the more realistic the results become in the target domain, the less we can preserve the source domain.

![](images/b4b0655a1519fcdec273d6aa7975a53e0b9a5f9ca10d653b9d4f18fe4e0888e7.jpg)  
Figure 8: Art-to-photo: Qualitative comparison with state-of-the-art methods. Our proposed LocDif enhances the low-quality art painting with photo-realistic details, while preserving facial identity.

To achieve faithful and realistic image enhancement, it is essential to achieve a balance between these two. The results demonstrate that our proposed locality-constraint guidance demonstrates a favourable trade-of of realism and identity preservation, adapting to user preferences and improving image enhancement under domain shifts.

## 4.5. Robustness to Segmentation Mask Quality

We conduct a comprehensive ablation study examining both the performance of state-of-the-art face parsing models on our domain-shifted datasets and the sensitivity of LocDif to mask inaccuracies.

Face Parsing Model Performance on Domain-Shifted Data. We evaluate three state-of-the-art face parsing models: FaRL [54] trained on LaPa [55], FaRL trained on CelebAMask-HQ [60], and SegFace [61] trained on LaPa. We extract six semantic classes (eyes, eyebrows, nose, mouth, skin) following our standard preprocessing pipeline (Section 4.2), with remaining classes treated as background.

Table 6 reports the frequency of missing semantic classes across models and datasets. On MetFaces and WikiArt, all LaPa-trained models achieve highly accurate detection, while the CelebAMask-HQ model struggles with closed eyes (14/8 missing eye detections on MetFaces). This may be explained by the fact that closed eyes are underrepresented in the CelebAMask-HQ dataset. The ultrasound dataset is the most challenging due to the fact that most fetuses have their eyes closed. However, FaRL (LaPa) maintains robust performance with only 1/3 missing eye detections compared to substantial failures of the CelebAMask-HQ model.

![](images/20c4fb40911c1abf983e636414bcdc5daff5c0068403eba90bb8faf4341bb93f.jpg)  
Figure 9: Visual comparison with baseline methods. Although DifBIR excels at denoising and preserving structure, it lacks photorealistic details in the mouth (bottom) and eye (top) regions and retains a painting-like appearance. In contrast, our method (LocDif) achieves photorealistic domain transfer while maintaining competitive identity preservation.

To quantify parsing consistency, we measure pairwise mean IoU (mIoU) between models across semantic classes (Tab. 6, bottom). On MetFaces and WikiArt, models show strong agreement (average mIoU: 0.849 and 0.866), indicating reliable parsing on artistic images. On ultrasound data, overall agreement drops (average mIoU: 0.522) due to eye detection failures. However, when excluding eye classes, agreement remains high (average mIoU: 0.846), demonstrating that parsing quality for other facial regions is maintained despite significant domain shift.

FaRL (LaPa) demonstrates the best generalization across all datasets and semantic classes, which is also used for all other experiments in this paper. We illustrate how parsing variations/failures afect enhancement quality on samples from the MetFaces and Ultrasound datasets with low cross-model mIOU to, see Figs. A.10 and A.11.

<table><tr><td>Dataset</td><td>Model</td><td>R.Eye</td><td>L.Eye</td><td>Nose</td><td>Mouth</td><td>Brows</td></tr><tr><td rowspan="3">MetFaces</td><td>SegFace (LaPa)</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>FaRL (LaPa)</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>FaRL (CelebA-HQ)</td><td>14</td><td>8</td><td>0</td><td>0</td><td>0</td></tr><tr><td rowspan="3">WikiArt</td><td>SegFace (LaPa)</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr><tr><td>FaRL (LaPa)</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>FaRL (CelebA-HQ)</td><td>10</td><td>9</td><td>0</td><td>0</td><td>0</td></tr><tr><td rowspan="3">Ultrasound</td><td>SegFace (LaPa)</td><td>8</td><td>13</td><td>0</td><td>0</td><td>1</td></tr><tr><td>FaRL (LaPa)</td><td>1</td><td>3</td><td>0</td><td>2</td><td>1</td></tr><tr><td>FaRL (CelebA-HQ)</td><td>330</td><td>339</td><td>0</td><td>2</td><td>10</td></tr><tr><td colspan="7">Dataset</td></tr><tr><td rowspan="2"></td><td>FaRL (C)</td><td rowspan="2"></td><td rowspan="2">FaRL (L) vs. SegFace (L)</td><td rowspan="2">FaRL (C) vs. SegFace (L)</td><td rowspan="2"></td><td rowspan="2">Avg mIoU</td></tr><tr><td>0.818</td></tr><tr><td colspan="2">MetFaces</td><td colspan="2">0.876</td><td colspan="2">0.854</td></tr><tr><td colspan="2">WikiArt</td><td colspan="2">0.846 0.473</td><td colspan="2">0.869</td></tr><tr><td colspan="2"></td><td colspan="2">0.883 0.574</td><td colspan="2"></td></tr><tr><td colspan="2">Ultrasound</td><td colspan="2"></td><td colspan="2">0.518</td></tr><tr><td colspan="2">Ultrasound (w/o eyes)</td><td colspan="2">0.848</td><td colspan="2">0.818</td></tr></table>

Table 6: Top: Missing semantic classes across face parsing models. Bottom: Pairwise cross-model agreement (mIoU). L=LaPa, C=CelebAMask-HQ. Dataset sizes: MetFaces (1366), WikiArt (1373), Ultrasound (437). High mIoU indicates reliable parsing despite domain shift.

Sensitivity to Mask Perturbations. To systematically analyze robustness to mask inaccuracies, we evaluate LocDif under controlled mask degradation scenarios on MetFaces and Ultrasound datasets:

• Spatial misalignment: 10-pixel shifts in $\mathrm { { x } / \mathrm { { y } } }$ directions

• Noise: 30% random pixels reassigned to background

• Morphological errors: Dilation (10 pixels), erosion (5/10 pixels)

• Missing features: Eye classes assigned to skin or background class

Table 7 and Figs. A.12 and A.13 present quantitative and qualitative results. LocDif maintains robust performance under spatial misalignment, moderate noise, and small morphological perturbations, with metrics degrading by less than 10% in most cases. The most significant degradation occurs when eyes are entirely missing from the face parsing map, though enhancement quality remains acceptable as shown in the diference maps. These results demonstrate that LocDif is resilient to common parsing errors encountered in practice and moderate inaccuracies of face parsing maps do not lead to artifacts.

<table><tr><td>Mask Perturbation</td><td>LMD↓</td><td>SSIM-F ↑</td><td>IDS ↓</td><td>FID ↓</td></tr><tr><td></td><td>MetFaces dataset</td><td></td><td></td><td></td></tr><tr><td>None</td><td> $3 . 2 \pm 1 . 6$ </td><td> $0 . 6 0 \pm 0 . 1 1$ </td><td> $4 1 . 0 \pm 7 . 6$ </td><td>64.8</td></tr><tr><td>XY-shift (10 pixels)</td><td> $3 . 3 \pm 1 . 5$ </td><td> $0 . 5 9 \pm 0 . 1 0$ </td><td> $4 1 . 5 \pm 7 . 6$ </td><td>65.4</td></tr><tr><td>Noise (30%)</td><td> $3 . 6 \pm 1 . 6$ </td><td> $0 . 5 7 \pm 0 . 1 0$ </td><td> $4 2 . 8 \pm 7 . 8$ </td><td>63.6</td></tr><tr><td>Dilation (10 pixels)</td><td> $3 . 0 \pm 1 . 4$ </td><td> $0 . 5 9 \pm 0 . 1 0$ </td><td> $3 8 . 2 \pm 7 . 3$ </td><td>66.6</td></tr><tr><td>Erosion (5 pixels)</td><td> $4 . 1 \pm 1 . 7$ </td><td> $0 . 5 4 \pm 0 . 1 0$ </td><td> $4 6 . 9 \pm 8 . 0$ </td><td>63.6</td></tr><tr><td>Eyes to background</td><td> $4 . 3 \pm 1 . 8$ </td><td> $0 . 5 3 \pm 0 . 1 1$ </td><td> $4 5 . 0 \pm 7 . 8$ </td><td>64.0</td></tr><tr><td>Eyes to skin</td><td> $3 . 7 \pm 1 . 6$ </td><td> $0 . 5 6 \pm 0 . 1 0$ </td><td> $4 3 . 1 \pm 7 . 8$ </td><td>63.8</td></tr><tr><td colspan="5">Ultrasound Rendering dataset</td></tr><tr><td>None</td><td></td><td> $0 . 7 8 \pm 0 . 0 5$ </td><td> $4 5 . 4 \pm 7 . 7$ </td><td>75.2</td></tr><tr><td>XY-shift (10 pixels)</td><td></td><td> $0 . 7 7 \pm 0 . 0 6$ </td><td> $4 4 . 3 \pm 8 . 0$ </td><td>74.6</td></tr><tr><td>Noise (30%)</td><td></td><td> $0 . 7 9 \pm 0 . 0 5$ </td><td> $4 1 . 8 \pm 7 . 6$ </td><td>75.3</td></tr><tr><td>Dilation (10 pixels)</td><td></td><td> $0 . 7 5 \pm 0 . 0 6$ </td><td> $4 6 . 4 \pm 8 . 2$ </td><td>73.3</td></tr><tr><td>Erosion (10 pixels)</td><td></td><td> $0 . 7 7 \pm 0 . 0 6$ </td><td> $4 4 . 2 \pm 8 . 1$ </td><td>75.6</td></tr><tr><td>Eyes to background</td><td></td><td> $0 . 8 1 \pm 0 . 0 5$ </td><td> $4 0 . 7 \pm 7 . 2$ </td><td>76.3</td></tr><tr><td>Eyes to skin</td><td></td><td> $0 . 7 7 \pm 0 . 0 6$ </td><td> $4 4 . 4 \pm 8 . 0$ </td><td>74.6</td></tr></table>

Table 7: Sensitivity to mask perturbations on MetFaces and Ultrasound datasets. Metrics (mean ± std) show LocDif is robust to common parsing errors, with performance degrading less than 10% in most cases. Metric changes reflect how perturbations redistribute pixels between regions with diferent conditioning strengths (MetFaces: strong eyes/skin, weak background; Ultrasound: strong background, weak eyes). Perturbations reducing strongly-conditioned regions trade faithfulness for photorealism, while the reverse occurs when strongly-conditioned regions expand. Visual examples in Figs. A.12 and A.13.

## 4.6. Limitations

The degree of realism depends on the power of the pre-trained difusion model. Consequently, strong domain shifts between training and source images will result in an unrealistic appearance. From our observations, performance quality is limited for unnatural face shapes, as seen in surrealistic paintings or statues, as well as images characterized by a high degree of noise or distortion, see Fig. A.14.

## 5. Conclusion and Future Work

Motivated by the challenge of balancing realism and faithfulness in crossdomain image enhancement, we presented a locality-constrained guidance approach named LocDif. Our method enables region-adaptive control of pre-trained difusion models, ofering flexible integration that supports both fully zero-shot scenarios and scenarios with lightweight prior alignment.

To enable difusion models trained on photos to handle input images with domain shifts, such as art paintings or renderings, we extend the sam pling step of the reversed difusion process with locality-constrained guidance. This is crucial to preserve selected local source-domain features while enabling global adaptation of the remaining features to the target domain. Our method enables precise control over the image enhancement process and facilitates user adaptation, e.g., adjusting of the local image region by modifying the attribution map and adapting the guidance strength.

Our experimental results demonstrate that LocDif performs competitively with state-of-the-art methods, achieving efective realism-faithfulness trade-ofs on both art painting and ultrasound rendering datasets, addressing the distinct challenge of cross-domain enhancement rather than in-domain restoration.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used DeepL to refine the language and improve readability. After using this tool, the author reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Acknowledgements

The VRVis GmbH is funded by BMIMI, BMWET, Tyrol, Vorarlberg and Vienna Business Agency in the scope of COMET - Competence Centers for Excellent Technologies (911654) which is managed by FFG.

## Appendix A. Appendix

## Appendix A.1. Datasets

All images in the used datasets are aligned to the FFHQ [11] template face and resized to 512 × 512 pixels using an alignment method<sup>1</sup>. We used the provided code<sup>2</sup> to curate the WikiArt-Faces dataset.

Baby Portrait Dataset (100 images): In order to fine-tune the difusion prior for the rendering-to-photo task, we collected a small dataset comprising 100 high-resolution face portrait images of babies. The images were sourced from Pexels.com using the search terms “baby” and “newborn.” Selection was restricted to images under a CC0 license with a single clearly visible face, only minimal facial occlusion (no objects or hairs covering important facial features), a balanced representation of head poses with 50% frontal and 50% non-frontal views, and high image quality. Images matching the search terms were manually reviewed, and the first 100 that satisfied the selection criteria were downloaded. All images were then preprocessed using the same alignment protocol described above.

## Appendix A.2. Hyperparameter & Baseline Implementation

For each dataset, we randomly selected 20% of the data to perform hyperparameter search. The performance metrics for the ablation study and the baselines were subsequently reported on the remaining 80% of the data. We evaluated the metrics of ControlNet [59] with the tile-based control method combined with the Realistic Vision 5.1 backbone, using the positive prompt "best quality, highly detailed face, photorealistic portrait" and negative prompt "blurry, low quality, distorted"), control scale 1.0, and guidance scale 5. We tested multiple configurations including various control types (tile, canny, soft boundaries) and backbones (SD 1.5, Realistic Vision 5.1) and report results for the best setting above.

## Appendix A.3. Metrics

IDS leverages ArcFace [56] embeddings to measure the angular distance between the features of the ground truth and reference images. The implementation from [41] is employed for this computation. To compute the Landmark Distance (LMD) score, which quantifies the distance between facial landmarks, we use SPIGA [62] to predict the facial landmarks and choose only the landmarks for eyebrows, eyes, nose, and mouth, resulting in a total of 65 landmarks. We use the torchmetrics framework<sup>3</sup> to calculate LPIPS, PSNR, and SSIM. To calculate the FID distance we use the implementation of [63]. We compute the SSIM-F as the mean SSIM over the eye, nose, and mouth regions. Peak GPU memory during inference (batch size = 1, 512×512 input) was measured on an Nvidia A100 using torch.cuda.max\_memory\_allocated.

## Appendix A.4. Implementation Details: Art-to-photo

For the art-to-photo task, Fig. 4 illustrates that at time step $t = 6 0$ the predicted image x<sub>0|60</sub> exhibits comparable image semantics to the final predicted image x<sub>0|0</sub>. We terminate the guidance at step 60 for our experiments, because the remaining steps only have minimal impact on the image semantics, as they are designed to develop high-frequency details.

• Eye, eyebrow, mouth, nose regions: $( t _ { m i n } = 6 0 , m = 2 )$

• Skin region: $( t _ { m i n } = 6 0 , m = 4 )$

• Background region: $( t _ { m i n } = 9 0 , m = 8 )$

We found that metrics remain stable beyond 140 steps, see Tab. 2, so we set the starting time step $T _ { s t a r t } = 1 4 0$

## Appendix A.5. Implementation Details: Rendering-to-photo

For the rendering-to-photo task, the goal is to preserve the ultrasound appearance while incorporating photorealistic details with moderate enhancement. This contrasts with the art-to-photo task, which aims for a result closely aligned with the photorealistic domain. To achieve this, we set $T _ { s t a r t } = 8 0$ and stop guidance at $t _ { m i n } = 4 0$ . In the ultrasound domain, eye details are often not visible. Therefore, we apply a weak guidance strength $( m = 8 )$ for the eyes to avoid compromising the generation of high-resolution details. For these barely visible features, the model is encouraged to generate plausible eye details. We apply strong constraints to the background, ensuring it closely matches the input image to prevent the generation of artifacts in the surrounding tissue.

• Eye and skin regions: $( t _ { m i n } = 4 0 , m = 8 )$

• Mouth and nose regions: $( t _ { m i n } = 4 0 , m = 4 )$

• Background region: $( t _ { m i n } = 4 0 , m = 2 )$

## Appendix A.6. Additional Visual Results

Figures A.10, A.11, A.12, A.13 support the mask quality ablation study presented in Section 4.5. Figure A.14 presents failure cases corresponding to the limitations analyzed in Section 4.6.

![](images/68ae16ac4991f6eb4ebc1ed140b09d45cabbb5cff7292db67541b5bc411f345a.jpg)

![](images/1bdb6bfb894c61239921f4390fc6d3926d428b6dc2488137f25dba2c6b87f27e.jpg)

![](images/90d81516ab3a95b3d3f20f8bf084cc217c5327d8f61f4e25d851ca5629d15326.jpg)

![](images/14256bf4234ce1cdb730e3d0eae026ae2458188968ed388ccb3e30884bcdfb4c.jpg)  
Figure A.10: Impact of parsing disagreement on challenging case. MetFaces sample with lowest cross-model mIoU (0.709) in the test set. Parsing disagreement around the mouth region (mouth vs. skin vs. background) produces visible diferences in enhancement: stronger conditioning on mouth/skin regions preserves the moustache better than weak background conditioning. Despite incorrect parsing, no artifacts are introduced in the enhanced images.

## References

[1] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, High Resolution Image Synthesis with Latent Difusion Models, in: CVPR, IEEE, 2022, pp. 10674–10685. doi:10.1109/CVPR52688.2022.01042. 2

[2] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, M. Chen, Hierarchical Text-Conditional Image Generation with CLIP Latents (2022). doi: 10.48550/ARXIV.2204.06125. 2

![](images/2c4e7569a3006f546a720f66c9b7404679dc47600de096f998eb0aba81b00d7e.jpg)

![](images/d900fdbdda3fe3825ad9b26a72ae5e2e8a2a4cc3f7f9c303ad9a50d287803516.jpg)

![](images/093fd2b31d0c0be57282aa4957b98ff77fe4a89405347b2d2acea50ea67843ae.jpg)

![](images/bda77b813177d64ea8ed5571b31eb6360aa67d8b54c36b3005dd5048583a155b.jpg)  
Figure A.11: Impact of parsing disagreement on challenging ultrasound sample. Ultrasound rendering with lowest cross-model mIoU (0.424) in the test set. FaRL (CelebM-HQ) incorrectly assigns eye and background regions to skin. Given the ultrasound-specific conditioning (strong background, weak skin/eyes), misclassified background regions are slightly more enhanced and the eye region is enhanced diferently. Despite these mask errors, enhancement quality degrades moderately with only localized diferences.

[3] C. Saharia, W. Chan, H. Chang, C. Lee, J. Ho, T. Salimans, D. Fleet, M. Norouzi, Palette: Image-to-Image Difusion Models, in: Special Interest Group on Computer Graphics and Interactive Techniques Conference Proceedings, ACM, 2022, pp. 1–10. doi:10.1145/3528233. 3530757. 2, 5

[4] C. Meng, Y. He, Y. Song, J. Song, J. Wu, J.-Y. Zhu, S. Ermon, SDEdit:image synthesis and editing with stochastic diferential equations, in: ICLR, 2022. 2, 3, 5, 6

[5] X. Wang, Y. Li, H. Zhang, Y. Shan, Towards Real-World Blind Face Restoration with Generative Facial Prior, in: CVPR, 2021, pp. 9164– 9174. doi:10.1109/CVPR46437.2021.00905. URL https://ieeexplore.ieee.org/document/9578533/ 3, 6

[6] Z. Yue, C. C. Loy, DifFace: Blind Face Restoration with Difused Error Contraction, IEEE TPAMI (2024) 1–15doi:10.1109/TPAMI.2024. 3432651. 3, 5, 6, 12, 13, 16, 20

![](images/576b324b1166638540b983fd4693fa6467b1d443abfe12ec1644687083ec788e.jpg)  
Figure A.12: Mask perturbation robustness analysis on ultrasound data. Top row: input rendering, unperturbed mask, and baseline enhancement. Rows 2-4 show mask perturbation pairs: perturbed mask, resulting enhancement, and RGB diference magnitude from baseline enhancement. Diferences are most visible in regions where perturbations alter conditioning strength. For the ultrasound task, background receives strong conditioning while skin/eyes receive weak conditioning. Diference maps reveal that most perturbations cause subtle, localized changes only, with missing eye classes and severe morphological errors producing the most visible deviations.

[7] P. Isola, J.-Y. Zhu, T. Zhou, A. A. Efros, Image-to-Image Translation with Conditional Adversarial Networks, in: CVPR, 2017, pp. 5967–5976. doi:10.1109/CVPR.2017.632. 3

[8] Y. Shen, J. Gu, X. Tang, B. Zhou, Interpreting the Latent Space of GANs for Semantic Face Editing, in: CVPR, 2020, pp. 9240–9249. doi: 10.1109/CVPR42600.2020.00926. 3

![](images/dcb4fa943aca97b9ca66c1418d79225b33f20cb7e3a5c6dc804acfc4e8b4a69f.jpg)  
Figure A.13: Mask perturbation robustness analysis on MetFaces. Top row: input, baseline mask, and baseline enhancement. Subsequent rows show pairs of perturbed masks, corresponding enhancements, and diference maps (RGB magnitude from baseline). Strongest diferences occur in perturbed regions where conditioning strength changes, particularly when critical facial features like eyes are misclassified, though overall enhancement remains stable. The dilation perturbation better preserves hair structure, as the dilated skin class (more strongly conditioned than the background) partially covers the hair regions.

[9] J.-Y. Zhu, T. Park, P. Isola, A. A. Efros, Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks, in: ICCV, 2017, pp. 2242–2251. doi:10.1109/ICCV.2017.244. 3

[10] Y. Choi, M. Choi, M. Kim, J.-W. Ha, S. Kim, J. Choo, StarGAN: Unified Generative Adversarial Networks for Multi-domain Image-to-Image Translation, in: CVPR, 2018, pp. 8789–8797. doi:10.1109/ CVPR.2018.00916. 3

[11] T. Karras, S. Laine, T. Aila, A Style-Based Generator Architecture

![](images/4fae6c2537c03203c97f75383ee64df7500fd820c2060fc11ecbafb0ead202e0.jpg)

![](images/08a9e82a6989d8f950e9aa472fa6fbdd25f398e0c0a07bbfe16a94610bdcd1d5.jpg)

![](images/5402bfdb1e49031651a37e0519928d0bfe07b89ac4ab6e3fd9070008bdf74c68.jpg)

![](images/a8345e8e145321c31c333bbda6889319b1b676c5e69fce2f86f0ffdf7d190587.jpg)  
Figure A.14: Failure cases of LocDif. Top row: original images. Bottom row: enhanced results. The method struggles when: (1) extreme anatomical deviations (elongated ear) are normalized to common shapes present in the training data, (2) heavy artistic brush strokes cause structural misinterpretation (nose deformation), (3) highly abstract face semantics produce only denoising without photorealistic detail generation, and (4) outof-distribution elements like hands are distorted due to absence from the FFHQ training prior.

for Generative Adversarial Networks, in: CVPR, 2019, pp. 4396–4405. doi:10.1109/CVPR.2019.00453. 3, 11, 12, 13, 27

[12] Y. Wang, J. Yu, J. Zhang, Zero-Shot Image Restoration Using Denoising Difusion Null-Space Model, in: ICLR, 2023. 3, 6

[13] B. Fei, Z. Lyu, L. Pan, J. Zhang, W. Yang, T. Luo, B. Zhang, B. Dai, Generative Difusion Prior for Unified Image Restoration and Enhancement, in: CVPR, 2023, pp. 9935–9946. doi:10.1109/CVPR52729.2023. 00958. 3, 6

[14] J. Choi, S. Kim, Y. Jeong, Y. Gwon, S. Yoon, ILVR: Conditioning Method for Denoising Difusion Probabilistic Models, in: ICCV, 2021, pp. 14347–14356. doi:10.1109/ICCV48922.2021.01410. 3, 6, 13, 16, 20

[15] E. Kodirov, T. Xiang, Z. Fu, S. Gong, Unsupervised Domain Adaptation

for Zero-Shot Learning, in: ICCV, IEEE, Santiago, Chile, 2015, pp. 2452–2460. doi:10.1109/ICCV.2015.282. 3

[16] S. Changpinyo, W.-L. Chao, B. Gong, F. Sha, Synthesized Classifiers for Zero-Shot Learning, in: CVPR, IEEE, Las Vegas, NV, USA, 2016, pp. 5327–5336. doi:10.1109/CVPR.2016.575. 3

[17] J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, S. Ganguli, Deep unsupervised learning using nonequilibrium thermodynamics, in: ICML, 2015, p. 2256–2265. 5

[18] J. Pidstrigach, Score-Based Generative Models Detect Manifolds, in: NeurIPS, Vol. 35, 2022, pp. 35852–35865. 5

[19] J. P. Stanczuk, G. Batzolis, T. Deveney, C.-B. Schönlieb, Difusion Models Encode the Intrinsic Dimension of Data Manifolds, in: ICML, 2022. 5

[20] Y. Bengio, A. Courville, P. Vincent, Representation Learning: A Review and New Perspectives, IEEE TPAMI 35 (8) (2013) 1798–1828. doi: 10.1109/TPAMI.2013.50. 5

[21] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, J. C. Ye, Difusion posterior sampling for general noisy inverse problems, in: ICLR, 2023. 5, 6, 8, 9, 13

[22] H. Chung, B. Sim, D. Ryu, J. C. Ye, Improving Difusion Models for Inverse Problems using Manifold Constraints, in: NeurIPS, 36, 2022, pp. 25683–25696. 5

[23] Y. He, N. Murata, C.-H. Lai, Y. Takida, T. Uesaka, D. Kim, W.-H. Liao, Y. Mitsufuji, J. Z. Kolter, R. Salakhutdinov, S. Ermon, Manifold Preserving Guided Difusion, in: ICLR, 2024. 5, 8

[24] T. Zhang, H. Xie, Sketch-Guided Text-to-Image Generation with Spatial Control, in: 2024 2nd International Conference on Computer Graphics and Image Processing (CGIP), 2024, pp. 153–159. doi:10.1109/ CGIP62525.2024.00035. 5

[25] N. Tumanyan, M. Geyer, S. Bagon, T. Dekel, Plug-and-Play Difusion Features for Text-Driven Image-to-Image Translation, in: CVPR, IEEE, 2023, pp. 1921–1930. doi:10.1109/CVPR52729.2023.00191. 5

[26] Z. Gu, E. Yang, A. Davis, Filter-Guided Difusion for Controllable Image Generation, in: ACM SIGGRAPH 2024 Conference Papers, SIG-GRAPH ’24, 2024, pp. 1–10. doi:10.1145/3641519.3657489. 5

[27] J. Yu, Y. Wang, C. Zhao, B. Ghanem, J. Zhang, FreeDoM: Training-Free Energy-Guided Conditional Difusion Model, in: ICCV, 2023, pp. 23117–23127. doi:10.1109/ICCV51070.2023.02118. 5, 6, 13

[28] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, L. Van Gool, RePaint: Inpainting using Denoising Difusion Probabilistic Models, CVPR (2022) 11451–11461doi:10.1109/CVPR52688.2022.01117. 5

[29] O. Bar-Tal, L. Yariv, Y. Lipman, T. Dekel, Multidifusion: Fusing diffusion paths for controlled image generation, in: ICML, PMLR, 2023. 5

[30] Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, L. V. Gool, Denoising Difusion Models for Plug-and-Play Image Restoration, in: CVPRW, 2023, pp. 1219–1229. doi:10.1109/CVPRW59228.2023.00129. 5

[31] Q. Cui, Y. Liu, X. Zhang, Q. Bao, Q. Liao, L. Wang, T. Lu, Z. Liu, Z. Wang, E. Barsoum, Taming difusion prior for image super-resolution with domain shift sdes, in: NeurIPS, 2024. 5

[32] P. Dhariwal, A. Nichol, Difusion models beat gans on image synthesis, in: NeurIPS, 2021. 6, 12

[33] P. Yang, S. Zhou, Q. Tao, C. C. Loy, PGDif: Guiding Difusion Models for Versatile Face Restoration via Partial Guidance, in: NeurIPS, 2023. 6, 7, 16, 20

[34] X. Lin, J. He, Z. Chen, Z. Lyu, B. Dai, F. Yu, W. Ouyang, Y. Qiao, C. Dong, Difbir: Towards blind image restoration with generative difusion prior, in: ECCV, 2024, pp. 430–448. doi:10.1007/ 978-3-031-73202-7\_25. 6, 16, 19, 20

[35] K. Shang, M. Shao, C. Wang, Y. Cheng, S. Wang, Multi-domain multiscale difusion model for low-light image enhancement, Proceedings of the AAAI Conference on Artificial Intelligence 38 (5) (2024) 4722–4730. doi:10.1609/aaai.v38i5.28273. 6

[36] Z. Fabian, B. Tinaz, M. Soltanolkotabi, Adapt and difuse: Sampleadaptive reconstruction via latent difusion models, in: ICML, 2024, pp. 12723–12753. 6, 7, 10

[37] S. Menon, A. Damian, S. Hu, N. Ravi, C. Rudin, PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models, in: CVPR, IEEE, 2020, pp. 2434–2442. doi:10.1109/ CVPR42600.2020.00251. 6

[38] C. Chen, X. Li, L. Yang, X. Lin, L. Zhang, K.-Y. K. Wong, Progressive Semantic-Aware Style Transformation for Blind Face Restoration, in: CVPR, 2021, pp. 11891–11900. doi:10.1109/CVPR46437.2021.01172. URL https://ieeexplore.ieee.org/document/9577487/ 6

[39] Y. Chen, Y. Tai, X. Liu, C. Shen, J. Yang, FSRNet: End-to-End Learning Face Super-Resolution with Facial Priors, in: CVPR, IEEE, 2018, pp. 2492–2501. doi:10.1109/CVPR.2018.00264. 6

[40] S. Zhou, K. C. Chan, C. Li, C. C. Loy, Towards robust blind face restoration with codebook lookup transformer, in: NeurIPS, 2022. 6

[41] Z. Wang, J. Zhang, T. Chen, W. Wang, P. Luo, RestoreFormer++: Towards Real-World Blind Face Restoration From Undegraded Key-Value Pairs, IEEE TPAMI 45 (12) (2023) 15462–15476. doi:10.1109/TPAMI. 2023.3315753. 6, 28

[42] Y. Gu, X. Wang, L. Xie, C. Dong, G. Li, Y. Shan, M.-M. Cheng, VQFR: Blind Face Restoration with Vector-Quantized Dictionary and Parallel Decoder, in: ECCV, Vol. 13678, 2022, pp. 126–143. doi:10.1007/ 978-3-031-19797-0\_8. 6, 13

[43] T. Brooks, A. Holynski, A. A. Efros, Instructpix2pix: Learning to follow image editing instructions, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 18392–18402. 6

[44] R. Mokady, A. Hertz, K. Aberman, Y. Pritch, D. Cohen-Or, Null-text inversion for editing real images using guided difusion models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 6038–6047. doi:10.1109/CVPR52729. 2023.00585. 6

[45] T. Kuai, S. Honari, I. Gilitschenski, A. Levinshtein, Towards unsupervised blind face restoration using difusion prior, in: Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV), 2025, pp. 1839–1849. doi:10.48550/arXiv.2410.04618. 6, 16, 20

[46] Y. Miao, J. Deng, J. Han, WaveFace: Authentic Face Restoration with Eficient Frequency Recovery, in: CVPR, 2024, pp. 6583–6592. doi: 10.1109/CVPR52733.2024.00629. 6

[47] D. Maurer, R. L. Grand, C. J. Mondloch, The many faces of configural processing, Trends in Cognitive Sciences 6 (6) (2002) 255–260. doi: 10.1016/s1364-6613(02)01903-4. 7

[48] X. Qiu, C. Han, Z. Zhang, B. Li, T. Guo, X. Nie, DifBFR: Bootstrapping Difusion Model for Blind Face Restoration, in: Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 7785– 7795. doi:10.1145/3581783.3611731. 7

[49] Y. Zhao, T. Hou, Y.-C. Su, X. Jia, Y. Li, M. Grundmann, Towards Authentic Face Restoration with Iterative Difusion Models and Beyond, in: ICCV, 2023, pp. 7278–7288. doi:10.1109/ICCV51070.2023.00672. 7

[50] J. Song, A. Vahdat, M. Mardani, J. Kautz, Pseudoinverse-guided difusion models for inverse problems, in: ICLR, 2023. 8

[51] L. Kong, Y. Du, W. Mu, K. Neklyudov, V. D. Bortoli, H. Wang, D. Wu, A. Ferber, Y.-A. Ma, C. P. Gomes, C. Zhang, Difusion Models as Constrained Samplers for Optimization with Unknown Constraints, AIS-TATS (2025). 10

[52] J. Song, C. Meng, S. Ermon, Denoising difusion implicit models, in: ICLR, 2021. 12

[53] T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, T. Aila, Training Generative Adversarial Networks with Limited Data, in: NeurIPS, Vol. 33, 2020, pp. 12104–12114. 12

[54] Y. Zheng, H. Yang, T. Zhang, J. Bao, D. Chen, Y. Huang, L. Yuan, D. Chen, M. Zeng, F. Wen, General facial representation learning in a visual-linguistic manner, in: CVPR, 2022, pp. 18697–18709. 12, 23

[55] Y. Liu, H. Shi, H. Shen, Y. Si, X. Wang, T. Mei, A new dataset and boundary-attention semantic segmentation for face parsing., in: AAAI, 2020, pp. 11637–11644. 12, 23

[56] J. Deng, J. Guo, N. Xue, S. Zafeiriou, ArcFace: Additive Angular Margin Loss for Deep Face Recognition, in: CVPR, 2019, pp. 4685–4694. doi:10.1109/CVPR.2019.00482. 13, 28

[57] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, S. Hochreiter, Gans trained by a two time-scale update rule converge to a local nash equilibrium, in: NeurIPS, 2017, p. 6629–6640. 13

[58] M. Tomei, M. Cornia, L. Baraldi, R. Cucchiara, Art2Real: Unfolding the Reality of Artworks via Semantically-Aware Image-To-Image Translation, in: CVPR, 2019, pp. 5842–5852. doi:10.1109/CVPR.2019.00600. 16, 20

[59] L. Zhang, A. Rao, M. Agrawala, Adding Conditional Control to Textto-Image Difusion Models, in: ICCV, IEEE, Paris, France, 2023, pp. 3813–3824. doi:10.1109/ICCV51070.2023.00355. 16, 20, 28

[60] T. Karras, T. Aila, S. Laine, J. Lehtinen, Progressive growing of GANs for improved quality, stability, and variation, in: International Conference on Learning Representations (ICLR), 2018. 23

[61] K. Narayan, V. Vs, V. M. Patel, Segface: Face segmentation of long-tail classes, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 6182–6190. 23

[62] A. Prados-Torreblanca, J. M. Buenaposada, L. Baumela, Shape Preserving Facial Landmarks with Graph Attention Networks, in: BMVC, 2022. 28

[63] G. Stein, J. Cresswell, R. Hosseinzadeh, Y. Sui, B. Ross, V. Villecroze, Z. Liu, A. L. Caterini, E. Taylor, G. Loaiza-Ganem, Exposing flaws of generative model evaluation metrics and their unfair treatment of difusion models, in: NeurIPS, Vol. 36, 2023. 29