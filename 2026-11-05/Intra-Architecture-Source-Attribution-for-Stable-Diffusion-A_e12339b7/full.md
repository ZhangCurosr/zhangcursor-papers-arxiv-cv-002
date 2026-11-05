# Intra-Architecture Source Attribution for Stable Diffusion: A New Task with Benchmark Dataset and Baseline

Ashutosh Dayal, Mallika Garg, and Debashis Sen Indian Institute of Technology Kharagpur, INDIA

ashutoshdayal@kgpian.iitkgp.ac.in, mallika@ec.iitr.ac.in, dsen@ece.iitkgp.ac.in

## Abstract

Designing systems that attribute a generated image to its source model has lately become a major area of AI research. Most existing datasets and systems related to image source attribution are targeted towards the detection of architectural signature of the source models. Distinct from this,focusing on stable diffusion, we introduce a novel task, IntrAA-SD, Intra-Architecture source Attribution for Stable Diffusion, where the source models to be differentiated share the same architectural backbone and differ only due to fine-tuning. The proposed task aligns with the emergence offine-tuned generators having the same architecture, like Juggernaut XL and RealVis XL which are finetuned versions ofStable Diffusion XL (SDXL). We provide a first-of-its-kind 140K-image benchmark dataset containing images generated from different fine-tunes of SDXL, along with in-domain and out-of-distribution prompts. We also provide a baseline framework for IntrAA-SD, which leverages a style-conditioned Mixture-of-Expert network. It uses a frozen CLIP backbone to capture the image style and gate the MoE, which processes fine-tunes’ trace features from a high-resolution LoRA-adapted DINOv2 network. The finetuned models used, the prompting strategies considered, and theformation details employed in the novel dataset are elaborated. Our baseline intra-architecture source attribution method is evaluated in terms ofattribution accuracy on our dataset, and ablation studies are also performed. Code and dataset is available here.

## 1. Introduction

Recent developments in generative AI has led to the availability of many open-weight text-to-image foundational models and their numerous fine-tuned versions. Therefore, attribution of image generations, not only to a specific architectural model but also to its particular fine-tuned version, is required for proper crediting and to identify misuse [31]. Although source attribution systems and related datasets are available that aim to link a generated image to an architectural model, the advent of fine-tuned generators has made reliable source attribution to specific fine-tunes difficult under the existing frameworks [1].

Branching off from the task of detecting whether an image is fake or not [45], the field of source attribution emerged for identifying the specific models that generated the given images [49] with the goal to facilitate intellectual property protection [52]. Recently, diffusion-based generators such as DALL-E [4], Midjourney [27], Imagen [37] and the Stable Diffusion [34] family have made high-quality synthetic image generation widely accessible, and hence, source attribution for diffusion-based generative models have since been investigated. Such architecture-level source attribution considers the classification of images from generator models that have significantly different architectures and generation pipelines [6, 13, 14, 16, 39, 48, 49, 51], like DALL-E 3 and Midjourney v6. However, models like Stable Diffusion XL (SDXL) [30] have a multitude of finetuned versions, which have been effectively used to generate images of various kinds, ranging from anime art to photorealistic portraits. The existing frameworks of designing systems for source attribution by detecting architectural signatures with the help of training on datasets of images created using different diffusion-based generators are neither intended nor suited for attributing the generation to such specific fine-tunes.

Therefore, “intra-architecture” source attribution is required, where the target would be to identify fine-tunes of the same architectural base model, which is a fundamentally different task from architecture-level source attribution. An intra-architecture source attribution system should be designed to be capable of classifying images from differently fine-tuned models having the same architecture and pipeline, like the Juggernaut XL [36] and RealVis XL [50] fine-tunes of SDXL. But, as [3, 39, 49] indicate, such an intra-architecture attribution using generic deep learningbased classifiers would be harder, as fine-tuning does not significantly alter the weights involved [35], and obviously, the backbone remains the same. Classifiers trained on images generated from such fine-tunes are prone to the influence of content and style, making the learning to distinguish between the subtle traces of the fine-tunes weak [1]. When tested, they may fail, particularly when out-of-distribution prompts are used for generation, where the image style may no longer match those used during training [3].

![](images/aa1bdd54d3684c1c163c836a43e20156cdf5260fc3709d20bbdee7ed31660738.jpg)  
Figure 1. Sample images from our IntrAA-SD benchmark dataset generated using identical prompts by SDXL fine-tunes represented by M1: SDXL Base, M2: Juggernaut, M3: RealVis, M4: DreamShaper, M5: Animagine, M6: Counterfeit, M7: ZavyChroma, M8: Copax, M9: Proteus, M10: AAM Anime. P1: “volcanic landscape with lava flow”, P2: “electric car charging at a station”. Despite sharing the same backbone, each fine-tune produces outputs with subtle visual distinctness with some pairs (like M2 & M3) more similar than others. Note: We use M1-M10 to denote the corresponding fine-tuned models throughout the paper.

In this paper, we introduce a novel task, IntrAA-SD, intra-architecture source attribution for stable diffusion, where the source of the images generated from different fine-tunes of stable diffusion are to be identified. To this end, we compile a novel 140K-image benchmark dataset containing images generated from 10 different finetunes of the SDXL, along with in-domain (ID) and out-ofdistribution (OOD) prompts. To the best of our knowledge, this is the first benchmark dataset for attribution within a single architecture family of fine-tuned models, and we focus on SDXL fine-tunes given their popularity. Fig. 1 displays representative samples from our benchmark dataset, showing the subtle visual differences across different finetunes of a base model given identical prompts. We discuss the fine-tuned models used, elaborate the prompting strategies considered, and provide details about the image collection to form the dataset.

We also propose an initial baseline framework of intraarchitecture attribution for the SDXL fine-tunes, which is designed to utilize a style-conditioned Mixture-of-Experts (SC-MoE) network. It uses a frozen CLIP backbone [32] to capture image style that is used to gate a set of expert networks, where each expert processes model trace features of the fine-tunes from a LoRA [18]-adapted DINOv2 [29] operating at its native resolution. While DINOv2 is selfsupervised with no use of text, making its features naturally suited for image structure representation [5], LoRA refines the features for the source attribution task with the potential to emphasize the fine-tune traces. The gating of mixtureof-experts using style information from the CLIP backbone allows the trace features to be different for differently styled images. At the end, a classification is performed based on a concatenated feature representing both style and fine tune traces to achieve the intra-architecture source attribution.

We rigorously evaluate the proposed baseline framework on our novel benchmark dataset in terms of attribution accuracy, and ablation studies are performed to investigate the effectiveness of its components.

To summarize, our key contributions are:

• We introduce a novel Intra-Architecture Attribution for Stable Diffusion (IntrAA-SD) task of identifying the specific fine-tuned model that generated a given image, among fine-tunes that share the same model architecture.

• To facilitate the IntrAA-SD task, we provide the novel IntrAA-SD benchmark (ISDB) dataset of images generated using different fine-tunes of Stable Diffusion XL and various categories of prompts.

• We propose an initial baseline, the IntrAA-SD framework, utilizing style-conditioned MoE, CLIP and LoRAadapted DINOv2 for performing the IntrAA-SD task.

## 2. Related Work

Although, to the best of our knowledge, there is no existing contribution specifically related to intra-architecture source attribution, we discuss several modern architecturelevel source attribution techniques briefly below.

## 2.1. Fake Image Detection and Attribution

Early work on fake image detection and source attribution includes detection of CNN-based generation, which shows that different architectures, datasets, and training methods leave detectable fingerprints in the generated images [45]. A similar observation has been reported in the context of attributing images generated by generative adversarial networks (GANs), where the presence of rigid checkerboard artifacts created by the up-sampling layers (transposed convolutional layers) [14] was leveraged. Up-scaling operations alter the high-frequency spectrum of images [19] and generative models often produce spectral patterns that do not match real data [9], making the detection effective in the frequency domain. Thus, GAN based generation was found to be detectable by leveraging frequency domain spectral distortions [10].

Table 1. The Stable Diffusion XL (SDXL) fine-tunes used to generate our IntrAA-SD benchmark dataset.
<table><tr><td>Styles</td><td>Models</td></tr><tr><td>Photorealistic</td><td>SDXL Base 1.0 [30], Juggernaut XL [36], RealVis XL [38], DreamShaper XL [26]</td></tr><tr><td>Anime</td><td>Animagine XL [21], CounterfeitXL [15], AAM XL Anime [25]</td></tr><tr><td>Stylized</td><td>ZavyChromaXL [42], CopaxTimelessXL [41], ProteusXL [8]</td></tr></table>

Later, in [16], along with fake images from GANs, images generated from diffusion models were also identified and discriminated. A CLIP-based detector was used in [7] as it supports high generalizability, which is a key requirement for synthetic image detectors. Further, efforts have been made to minimize semantic bias by utilizing the frequency domain for the detection of diffusion-based image generations [2, 33]. It was demonstrated in [33] that unexpected convolutional traces remained in the high-frequency spectrum of images from diffusion models that use U-Net architectures.

## 2.2. Source Attribution of Diffusion Models

In the era of diffusion models, the focus has shifted from merely detecting fake images or attributing them to a broad class of generative models (like GAN, Diffusion, etc.), to examining the residual traces left by the diffusion process itself, enabling architecture-level source attribution [11]. It was shown in [6] that diffusion models that uses different backbones (like Latent Diffusion, GLIDE, etc.) produce different spectral signatures based on the particulars of the noise scheduler and the up-sampler utilized in the generative process.

The current leading methods for diffusion source attribution can be broadly organized into three categories, semantic analysis-based, spatial extraction-based and reconstruction-based.

Semantic Based: Standard methods [7, 28] in this category primarily use frozen CLIP embeddings [28] to identify and delineate semantic inconsistencies often found in synthetic images generated from different diffusion models. However, these methods have been found to be prone to overfitting, which in certain cases make CLIP-based classification detect styles rather than model fingerprints [3], making them ill-suited to detect fine-tuned sources.

Spatial Based: Methods in this category like DE-FAKE [39], use standard CNN backbones and crossattention maps for diffusion source attribution and work well for classifying different generators such as Stable Diffusion, Midjourney, etc. These methods are also susceptible to false attribution when fine-tuned diffusion models are encountered, as they usually do not possess the required capacity to capture the subtle differences [3].

Reconstruction Based: This recent category of methods like the Diffusion Reconstruction Error (DIRE) method [46] attempts diffusion model attribution by measuring the difference between the image at hand and its reconstructed counterpart, which is obtained by running forward diffusion using the image and then executing the pretrained reverse diffusion generation. A down-side to approaches like DIRE as an attribution technique is their resource-intensive nature, as rerun of the diffusion process is required. Use of DIRE for distinguishing images from fine-tuned diffusion models remains unexplored.

Until now, investigations on source attribution of diffusion models have been limited to architecture-level, like distinguishing Stable Diffusion (SD) v1.4 from Stable Diffusion (SD) v2.0. As mentioned earlier, such attribution techniques neither intend to distinguish fine-tunes of architecturally identical diffusion models nor many of them are suited for the same. Thus, the rapid growth of openweight diffusion models naturally leads to the emergence of the intra-architecture source attribution problem, where the goal is to detect the nearly indistinguishable structural traces to delineate a fine-tune of the same architectura model from the other fine-tunes.

## 3. Novel Dataset: The IntrAA-SD Benchmark

We build IntrAA-SD Benchmark (ISDB) dataset in order to facilitate the attribution of a generated image to one of the fine-tunes of open-weight Stable Diffusion XL (SDXL) having the same architectural backbone, and to subsequently design such an intra-architecture source attribution baseline and evaluate it. We consider in-domain (ID) and out-of-distribution (OOD) prompts for the testing.

## 3.1. Generative Models Considered

For intra-architecture source attribution, we generate images using 10 SDXL fine-tunes from Hugging Face [47] spanning photorealistic, anime, and stylized categories as given in Table 1. This selection is done to include both challenging pairs in terms of attribution whose outputs are nearly indistinguishable to humans like Juggernaut XL vs. RealVis XL (refer to Fig. 1), and stylistically divergent pairs that primarily differ in visual style like Animagine XL vs. RealVis XL.

## 3.2. In & Out-of-Domain Prompting Strategies

We created two prompt sets, in-domain (ID) and out-ofdistribution (OOD) for testing, and different prompts from the ID set are used to train the SDXL fine-tunes.

The ID set contains 300 prompts spread across 15 categories as shown in Table 2. Eight of these categories cover standard real-world content such as portraits, landscapes, architecture, animals and food. The remaining seven are particularly challenging in terms of generation, which involves image generation with repetitive textures and geometric patterns, extreme lighting, noise, etc. [6]. Some such categories under the type of generative stress test listed in Table 2 are elaborated in the supplementary.

Table 2. In-Domain (ID) prompt categories in our ISDB dataset with example prompts.
<table><tr><td># Category</td><td>Example Prompt</td></tr><tr><td colspan="2">Real-World Content (8 categories)</td></tr><tr><td>1 Portraits &amp; Faces 2</td><td>“close up portrait of a young woman with freckles, natural light&quot;</td></tr><tr><td>Landscapes &amp; Nature</td><td>“mountain range at sunrise with fog in the valley, golden light</td></tr><tr><td>3 Architecture &amp; Interiors</td><td>“gothic cathedral interior with stained glass windows, light beams&quot;</td></tr><tr><td>4 Animals &amp; Wildlife</td><td>“tiger walking through shallow water, jungle background, 8k detail&quot;</td></tr><tr><td>5 Food &amp; Objects</td><td>“bowl of ramen with egg, nori, chopsticks, steam rising&#x27;</td></tr><tr><td>6</td><td>Vehicles &amp; Scenes “red sports car on a mountain road, motion blur&quot;</td></tr><tr><td>7</td><td>Fantasy &amp; Sci-Fi “dragon flying over a medieval castle, fire breath, dark clouds&quot;</td></tr><tr><td>8 Abstract &amp; Artistic</td><td>“oil painting of a stormy sea, thick impasto brushstrokes&quot;</td></tr><tr><td colspan="2">Generative Stress Tests (7 categories)</td></tr><tr><td>9 Texture</td><td>“woven fabric extreme close up, individual threads visible&quot;</td></tr><tr><td>10 Geometric &amp; Repetitive</td><td>“infinite mirror tunnel, repeating reflections’</td></tr><tr><td>11 Infeasible &amp; Indeterminate</td><td>“Penrose triangle rendered as a real 3D object’</td></tr><tr><td>12 Fluids &amp; Particles</td><td>“milk drop splash crown forming, high speed macro&quot;</td></tr><tr><td>13 Lighting Extremes</td><td>“single candle flame in complete darkness, warm glow&quot;</td></tr><tr><td>14 Text &amp; Signals</td><td>“chalkboard filled with mathematical equations&quot;</td></tr><tr><td>15 Noisy &amp; Malicious</td><td>“television static, random pixels, no coherent pattern’”</td></tr></table>

Table 3. Out-of-Distribution (OOD) prompt categories in our ISDB dataset with example prompts.
<table><tr><td>#</td><td>Category</td><td>Example Prompt</td></tr><tr><td colspan="3">Adversarial Strategies (7 categories)</td></tr><tr><td>1</td><td>Cross-Domain</td><td>&quot;studio portrait, soft lighting, 85mm lens&quot; → fed to Animagine XL; &quot;flat anime char-</td></tr><tr><td>2</td><td>Void Prompts</td><td>acter, cel shading&quot; → fed to RealVis XL “pure white background, nothing else&quot;; “complete darkness, empty frame&quot;</td></tr><tr><td>3</td><td>Nested Representation</td><td>“a photograph of a charcoal sketch pinned to a cork board&quot;; “a watercolor painting lying on a wooden desk&quot;</td></tr><tr><td>4</td><td>Text-Heavy</td><td>“a handwritten letter on aged parchment, cursive text filling the page&quot;</td></tr><tr><td>5</td><td>Hands &amp; Anatomy</td><td>“a pianist playing a grand piano, fingers on keys, concert hal</td></tr><tr><td>6</td><td>Reflections &amp; Glass</td><td>“raindrops on a car window with blurred city lights, bokeh, night</td></tr><tr><td>7</td><td>High-Frequency Patterns</td><td>&quot;brick wall with every individual brick visible&quot;; “woven fabric macro, individual threads&quot;</td></tr><tr><td colspan="3">Domain-Shift (11 categories)</td></tr><tr><td>8</td><td>Diverse Cultural</td><td>&quot;a colorful Indian Holi festival, powder clouds, joyful people&quot;</td></tr><tr><td>9</td><td></td><td>“a microscope view of cells dividing, stained tissue sample&quot;</td></tr><tr><td>10</td><td>Medical &amp; Scientific</td><td>“a daguerreotype style portrait, early photography aesthetic&quot;</td></tr><tr><td>11</td><td>Historical &amp; Vintage Underwater &amp; Aquatic</td><td>“a diver exploring a sunken shipwreck, flashlight beam, deep blue&quot;</td></tr><tr><td>12</td><td>Sports &amp; Action</td><td>“a basketball player dunking, arena lights, freeze frame”</td></tr><tr><td>13</td><td>Fashion &amp; Textile</td><td>“haute couture dress on a runway, dramatic lighting”</td></tr><tr><td>14</td><td>Mythology &amp; Folklore</td><td>“a Japanese kitsune spirit in a misty bamboo forest”</td></tr><tr><td>15</td><td>Industrial &amp; Decay</td><td>“a derelict shopping mall, broken escalators, overgrown plants&quot;</td></tr><tr><td>16</td><td>Unusual Perspectives</td><td>“isheye lens view of a skate park, extreme wide angle distortion’</td></tr><tr><td>17</td><td>Weather &amp; Atmospheric</td><td>“a tornado funnel cloud touching down on farmland, debris&quot;</td></tr><tr><td>18</td><td>Edge Cases</td><td>“a completely white image with a barely visible gray shape&quot;</td></tr><tr><td colspan="3">Subtle Domain-Shift (2 categories)</td></tr><tr><td>19</td><td></td><td></td></tr><tr><td>20</td><td>Lighting Condition</td><td>“harsh midday sun, deep black shadows on white architecture&quot;</td></tr></table>

![](images/e364c6662f82815ccf30949f338a12dd47a2b2c050c1188ec794ad7101686ca6.jpg)  
Figure 2. Our IntrAA-SD Baseline Framework. A frozen CLIP backbone captures the image style (Sec. 4.1) to compute gating weights across N experts. A LoRA-adapted DINOv2 backbone extracts the fine-tunes’ related patterns fed into experts (Sec. 4.2). The expert out puts are combined via style-conditioned Mixture-of-Experts (Sec. 4.3) and concatenated with the style context for the attribution (Sec. 4.4).

The OOD set contains 200 prompts across 20 categories as listed in Table 3. These categories fall under adversarial strategies and domain shift ones. In the latter, a couple of categories represent subtle domain shift, where the prompts are similar to a few in the ID set but demand quite different generations. Adversarial strategies contain categories such as cross-domain prompting, void prompts, nested representations, etc. Due to its adversarial nature, the crossdomain prompting category is mentioned here and not under domain shift strategy. In cross-domain prompting category, photorealistic prompts are fed to anime-focused models like Animagine XL, and conversely, cartoon and illustration prompts to photorealistic models like RealVis XL. This makes it difficult for a intra-architecture source attribution technique to classify, as the generated images could be quite different than those on which the attribution model is trained.

A couple more categories under adversarial strategies listed in Table 3 are elaborated in the supplementary. The other OOD prompt categories like cultural scenes, medical and scientific imagery, historical styles, underwater environments, sports and fashion are mentioned under domainshift, as they are in general not adversarial in nature. Their use to generate images different from those generated using ID prompts is straightforward.

## 3.3. Dataset Details

Our ISDB dataset consists of 140K images, evenly split across 10 classes (12K ID & 2K OOD images per class), with each class representing generation from an SDXL finetune. Every image is generated with 1024 × 1024 resolution using the Hugging Face Diffusers library [44]. We have used the DPM++ SDE Karras scheduler [20, 24] with 25 denoising steps, a classifier-free guidance (CFG) scale of 7.0 [17], and a fixed negative prompt to suppress common artifacts. All outputs are saved at 95% JPEG quality to simulate real-world dissemination conditions, since most images encountered in the wild have gone through at least one round of lossy compression [45]. The ID set comprises 300 (prompts) × 40 (seeds) = 12000 images per model of 15 prompt categories, while the OOD set adds 200 (prompts) × 10 (seeds) = 2000 images per model from 20 prompt categories. We split the ID set into 80/20 train and test partitions using stratified sampling with a fixed random state, ensuring no prompt overlap between splits. The OOD set is entirely used only for evaluation. ID and OOD set images are rendered with 40 and 10 deterministic seeds (to random number generator), respectively, to ensure diversity while maintaining reproducibility.

## 4. Initial Intra-architecture Source Attribution Baseline: The IntrAA-SD Framework

We propose the first baseline, our IntrAA-SD framework, for intra-architecture source attribution of Stable Diffusion XL (SDXL) fine-tunes. The key idea in our framework is to decouple the style understanding from the fine-tune trace analysis, and utilize both judiciously. As illustrated in Fig. 2, the framework has four components, a frozen CLIP backbone [32] that captures the visual style of the input, a LoRA-adapted DINOv2 backbone [29] that extracts fine-tune trace related features at its native pixel resolution, a style-conditioned Mixture-of-Experts (MoE) [12] where CLIP-derived style information gates the trace features through specialized experts, and finally, a classifier that acts on a fused feature content to give the final logits. Our framework takes generated images as inputs at $5 1 8 \times 5 1 8$ resolution, with random crop employed during training and five crop test-time augmentation during testing.

## 4.1. Visual Style Extraction

A common aspect that may hinder intra-architecture attribution is that classifiers would learn the predominant image style rather than the subtle traces left by the generator [3]. For instance, a model trained on our dataset might learn to associate anime-style output with Animagine XL, as the training images generated by it would mostly be anime. As evident from Table 1, while style can help distinguish some fine-tunes from others, it alone may not be sufficient to pinpoint a fine-tuned generator in general. Dependence on style alone could result in failure, particularly in OOD cases such as when we feed a photorealistic prompt to Animagine XL. To address this, we intend to ensure that subtle fine-tune traces are leveraged along with style for intraarchitecture source attribution.

For extracting the style features $( \bf x _ { \mathrm { s t y l e } } )$ from a given image, we use a frozen CLIP ViT-L/14 [32]. As CLIP was trained on a large-scale dataset of image-text pairs, it has learned rich representations of visual style and composition [32]. So, it would capture the same from the input images. More details on this are shared in the supplementary.

## 4.2. Fine-tune Trace Feature Extraction

For extracting features representing fine-tune traces in image structure, we choose a frozen DINOv2 ViT-L/14 [29] followed by a trainable Low-Rank Adaptation (LoRA) [18]. Unlike CLIP, DINOv2 is self-supervised and do not involve text. So, the pretrained DINOv2 has learned rich representations of visual structure, texture and content [5]. Working at its native high resolution [29], DINOv2 has the capability to represent image details [6, 33]. Then, to consider only the features relevant to intra-architecture source attribution using our framework, trainable LoRA is adopted. LoRA therefore emphasizes only fine-tuned generator relevant features distilling them from the image structure and content representations in its low-rank representation. We represent the fine-tune trace features obtain from LoRA-adapted DI-NOv2 as a vector $\mathbf { x } _ { \mathrm { t r a c e } } \in \mathbb { R } ^ { d _ { \mathrm { t r a c e } } }$ , where $d _ { \mathrm { t r a c e } } = 1 0 2 4$

## 4.3. Style-Conditioned Mixture of Experts

Differently styled generated images may contain structurally different generator signatures [14, 45]. Therefore, rather than relying on a single classifier to implicitly handle this variation, we consider a style-conditioned gating driven Mixture-of-Experts (SC-MoE) architecture before classification. We adopt an MoE architecture [12, 40] with a crossmodal gating of model trace features from LoRA-adapted DINOv2. The gating decision, which dictates how much each expert contributes to the final output, is determined by a gating network working on the style feature vector x<sub>style</sub> from CLIP. The gating weights are computed in the gating network as $g = \operatorname { S o f t m a x } ( W _ { g } \mathbf { x } _ { \operatorname { s t y l e } } )$ , where $g \in \mathbb { R } ^ { N }$ [40] with N being the number of experts. Every expert $E _ { i }$ is a two-layer MLP with output dimension, $d _ { \mathrm { { e x p e r t } } } \mathrm { { = } } 1 0 2 4$ , and their outputs are combined using the weights, $g _ { i } ,$ to get the fused vector $\mathbf { F } _ { \mathrm { f u s e d } } \in \mathbb { R } ^ { d _ { \mathrm { c x p e r } } }$

## 4.4. Classification for Intra-architecture Source Attribution

To leverage both the fine-tune trace and style information from the image, we concatenate the clearly separated SC-MoE fused trace features $\mathbf { F } _ { \mathrm { f u s e d } }$ and style features $\mathbf { x } _ { \mathrm { s t y l e } }$ to form a joint representation $\mathbf { V } _ { \mathrm { f i n a l } }$ that carries both fine-tuned model signature and image style context.

Specifically, the fused SC-MoE output $\mathbf { F } _ { \mathrm { f u s e d } } \in \mathbb { R } ^ { d _ { \mathrm { e x p e r t } } }$ $( d _ { \mathrm { { e x p e r t } } } = 1 0 2 4 )$ is concatenated with the CLIP style vector $\mathbf { x } _ { \mathrm { s t y l e } } \in \mathbb { R } ^ { d _ { \mathrm { s t y l e } } } \ ( d _ { \mathrm { s t y l e } } = 1 0 2 4 )$ as:

$$
\mathbf { V } _ { \mathrm { f i n a l } } = [ \mathbf { F } _ { \mathrm { f u s e d } } \ \lVert \ \mathbf { x } _ { \mathrm { s t y l e } } ] \in \mathbb { R } ^ { d _ { \mathrm { e x p e r t } } + d _ { \mathrm { s t y l e } } } ( \mathbb { R } ^ { 2 0 4 8 } ) .
$$

This 2048-dimensional vector is then passed through a linear layer with the weight matrix $W _ { c } \ \in \ \mathbb { R } ^ { C \times 2 0 4 8 }$ and bias $b _ { c } \in \dot { \mathbb { R } } ^ { C }$ . The training of the entire system is done predominantly using the standard cross-entropy loss $\mathcal { L } _ { C E }$ over C classes (number of fine-tuned generators considered). The overall training objective is:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { C E } + \lambda _ { \mathrm { b a l } } \mathcal { L } _ { \mathrm { b a l } } ,\tag{1}
$$

where $\lambda _ { b a l }$ is a hyperparameter to adjust the weightage of load-balancing loss, $\mathcal { L } _ { \mathrm { b a l } }$ , which is designed to make the model use all the experts in the MoE more equally [40]. When our ISDB dataset is considered, we have $C = 1 0$ Thus, by providing the classifier both the model trace information and the style context, we enable it to learn generator separation based on different model trace signatures for different styles.

## 5. Experiments

As we have introduced the new task of intra-architecture source attribution in this paper with a novel benchmark dataset and a baseline technique, we evaluate the baseline technique on the ISDB dataset in this section and perform its empirical ablation study. Due to space constraints here, we report additional experiments, and investigate the applicability of existing architecture-level attribution models for intra-architecture attribution in the supplementary.

## 5.1. Training Strategy

The IntrAA-SD baseline framework is trained for 30 epochs on the in-domain (ID) training set of our ISDB dataset employing AdamW [23] and cosine annealing with warm restarts [22]. The LoRA parameters use a learning rate of $1 \times 1 0 ^ { - 4 }$ , while the MoE and classification heads use $5 \times 1 0 ^ { - 4 }$ . We also use label smoothing (ϵ = 0.1) [43]. For augmentation, we apply MixUp $( \alpha = 0 . 2 )$ [54] and Cut-Mix (α = 1.0) [53] with a probability $p = 0 . 5 .$ Random cropping is used for training, and FiveCrop Test Time Augmentation is applied at inference.

Table 4. Intra-architecture source attribution accuracies (%) of our IntrAA-SD baseline framework on our ISDB dataset, for the test images generated with In-Domain (ID) and Out-of-Distribution (OOD) prompts.
<table><tr><td>Source Model</td><td>ID</td><td>OOD</td></tr><tr><td>SDXL Base (M1) Juggernaut XL (M2) RealVis XL (M3) DreamShaper XL (M4) Animagine XL (M5)</td><td>99.42 99.79 98.67 99.83 99.92</td><td>91.50 70.50 82.50 99.00 98.00</td></tr><tr><td>CounterfeitXL (M6) ZavyChromaXL (M7) CopaxTimelessXL (M8) ProteusXL (M9)</td><td>98.29 99.67 97.88</td><td>98.00 74.00 90.00</td></tr><tr><td>AAM XL Anime (M10) Overall</td><td>98.58 98.21 99.03</td><td>90.50 91.50 88.55</td></tr></table>

## 5.2. Attribution Results for ID and OOD Prompts

Table 4 presents the results of our IntrAA-SD framework considering images generated in our ISDB dataset from all the 10 fine-tuned source models using in-domain (ID) and out-of-distribution (OOD) testing prompts, showing the per-class (source) accuracies. Our method achieves an average of 99.03% accuracy on the ID test set case and 88.55% on OOD case. We refer to the results in Table 4 further in the supplementary for more discussion.

## 5.3. Ablation Studies on IntrAA-SD framework

## 5.3.1. Core Modules

In Table 5, we report an ablation study of our baseline intraarchitecture attribution method using our ISDB dataset’s ID test set, to note the significance of its components.

Gating strategy. As seen in Table 5, the equal weighting case (A1), where the gating network is entirely removed and equal-weighted average of all the experts is considered, yields 98.40% accuracy. The DINOv2 driven variant (A2), where features from frozen DINOv2 (not trained with text) are used in the gating network to determine routing weights, produces almost a similar accuracy of 98.42%. However, when use consider the use of style features from frozen CLIP in our IntrAA-SD framework to drive the gating network, the highest accuracy of 99.03% is achieved. This observation demonstrates that marginal benefit is achieved by

Table 5. IntrAA-SD Ablation: significance of components. All variants use LoRA r=16, 518px, 4 experts (unless removed).
<table><tr><td rowspan=1 colspan=1>Components</td><td rowspan=1 colspan=1>Variants</td><td rowspan=1 colspan=1>Acc %</td></tr><tr><td rowspan=1 colspan=1>Gatingstrategy</td><td rowspan=1 colspan=1>A1: Equal weighting (no gat-ing)A2: Structure-based gating (DI-NOv2 → gate)</td><td rowspan=1 colspan=1>98.4098.42</td></tr><tr><td rowspan=1 colspan=1>Style concat</td><td rowspan=1 colspan=1>A3: No style concatenation (af-ter MoE) before classification</td><td rowspan=1 colspan=1>98.21</td></tr><tr><td rowspan=1 colspan=1>Style ortrace feature</td><td rowspan=1 colspan=1>A4: Image Style-only (FrozenCLIP + linear head)A5: Source trace-only (LoRA $\mathrm { D I N O v 2 } + \mathrm { l i n e a r h e a d } )$ </td><td rowspan=1 colspan=1>87.5998.52</td></tr><tr><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>IntrAA-SD framework</td><td rowspan=1 colspan=1>99.03</td></tr></table>

specifically designing the extraction of style features and using it for gating the Mixture-of-Experts (MoE) network.

Style feature concatenation. From Table 5, we infer that the concatenation of style features $\mathbf { x } _ { \mathrm { s t y l e } }$ with fused finetuned model trace features $\mathbf { F } _ { \mathrm { f u s e d } }$ from the MoE helps our IntrAA-SD framework to achieve 99.03% accuracy, while if the concatenation is not performed (A3) and only $\mathbf { F } _ { \mathrm { f u s e d } }$ is fed into the classifier, the accuracy drops to 98.21%.

Style & source trace. We attempt to classify the generated input images for detecting the fine-tuned source model using only the style features from the frozen CLIP without using any MoE network (A4). This yields a poor 87.59% accuracy as seen in Table 5. This is expected, as from our discussion in Section 4.1, we know that style features may not be sufficient on its own for intra-architecture attribution. In contrast, when we attempt the same using only the fine-tune trace features from the frozen DINOv2 and LoRA removing the MoE network (A5), a much improved 98.52% accuracy is obtained. This shows that the frozen DINOv2 with LoRA indeed provides features relevant to intra-architecture attribution. When both the source trace features to drive the MoE and the style features primarily to gate the MoE are used in our IntrAA-SD framework, we achieve the best performance with an accuracy of 99.03%.

## 5.3.2. Hyperparameter Choices

We also study the impact of hyperparameter and inference choices, and the related performance is listed in Table 6.

Fine-tune model trace feature extractor. In Table 6, we see that if a completely frozen DINOv2 alone is used to extract intra-architecture attribution relevant features, only 93.01% accuracy is obtained. However, injecting trainable LoRA matrices improves the performance substantially achieving 98.5% and 99.03% accuracies for ranks (r) 4 and 16, respectively. This suggests that LoRA indeed effectively distills model trace features from the image structure represented by DINOv2 as discussed in Section 4.2.

Number of experts in MoE. We experimented by changing the number of experts in MoE in our framework. As seen from Table 6, a slight increase in accuracy from 98.77% to 99.03% is observed when four experts are used instead of one, which represents the case of no MoE. Although the effect of MoE on the performance may not be that significant, we stick with the four experts case to keep the fraction of gain attained.

Table 6. Ablation on Hyperparameters. Ours: IntrAA-SD baseline intra-architecture source attribution framework
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Variants</td><td rowspan=1 colspan=1>Acc (%)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Variants</td><td rowspan=1 colspan=1>Acc (%)</td></tr><tr><td rowspan=1 colspan=1>Model trace extractor</td><td rowspan=1 colspan=1>Frozen DINOv2 $\mathrm { L o R A } \ r { = } 4$  $\operatorname { L o R A } r { = } 1 6 ( \operatorname { O u r s } )$ </td><td rowspan=1 colspan=1>93.0198.5099.03</td><td rowspan=1 colspan=1>Input resolution</td><td rowspan=1 colspan=1> $\overline { { 2 2 4 \times 2 2 4 } }$  $3 7 8 \times 3 7 8$  $5 1 8 \times 5 1 8 ( \mathrm { O u r s } )$ </td><td rowspan=1 colspan=1>96.7598.4199.03</td></tr><tr><td rowspan=1 colspan=1>Number of experts</td><td rowspan=1 colspan=1> $\overline { { 1 \ : ( \mathrm { n o } \ : \mathrm { M o E } ) } }$  $2$  $4 \ : ( \mathrm { O u r s } )$ 8</td><td rowspan=1 colspan=1>98.7798.8499.0398.88</td><td rowspan=1 colspan=1>Inference input</td><td rowspan=1 colspan=1>Center CropFive Crop TTA (Ours)</td><td rowspan=1 colspan=1>98.2399.03</td></tr></table>

Table 7. Ablation on the loss function $( \mathcal { L } _ { t o t a l } )$ parameter $\lambda _ { b a l }$
<table><tr><td rowspan=1 colspan=1> $\lambda _ { b a l }$ </td><td rowspan=1 colspan=1>0.0 (No balancing)</td><td rowspan=1 colspan=1>0.001</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=1>Acc (%)</td><td rowspan=1 colspan=1>98.78</td><td rowspan=1 colspan=1>99.03</td><td rowspan=1 colspan=1>98.72</td><td rowspan=1 colspan=1>98.70</td></tr></table>

Resolution and inference. In Table 6, we see that the use of $5 1 8 \times 5 1 8$ resolution inputs gives the best accuracy of 99.03% among the listed resolutions. The higher accuracy at higher resolution is expected as model trace signatures embedded in spatial details would be more accurately represented. Further, we see that five crop test-time augmentation (TTA) provides an improved accuracy over single center crop during inference.

Load-balancing loss weightage. In Table 7, when a load-balancing loss is employed in the loss function of Equation (1) with a small weight of .001, the best accuracy of 99.03% is attained, and hence, it is used in our baseline.

## 5.4. Fine-tunes’ Trace Feature Space Visualization

We project the 1024-d $\mathbf { F } _ { \mathrm { f u s e d } }$ model trace features onto two dimensions using t-SNE, as shown in Fig. 3. As can be seen, images from the fine-tuned SDXL models such as Animagine, Counterfeit, AAM and DreamShaper are represented as compact clusters well-separated from the others. Some fine-tunes like Proteus appear reasonably separable, but the few others do not appear so in the 2-d plot. However, as we have seen earlier, the fine-tune trace feature representation in our baseline model actually achieves more than 99% source attribution accuracy, which suggests that most of the fine-tunes must be reasonably well-separated in the 1024 dimensional feature space.

## 6. Conclusion

Identifying the precise source of images produced by finetuned generators, which share the same architectural backbone with other fine-tunes, is non-trivial and essentially a separate task than the existing architecture-level source attribution. In this regard, we introduced a novel task, IntrAA-SD, Intra-Architecture source Attribution for Stable Diffusion, provided a first-of-its-kind 140K-image benchmark dataset (IntrAA-SD Benchmark - ISDB), and proposed the initial baseline framework (IntrAA-SD framework) to perform the Intra-architecture attribution of fine-tunes. Our ISDB dataset is formed considering 10 Stable Diffusion XL fine-tunes, and both in-domain (ID) and out-of-distribution (OOD) prompts. Our baseline approach leverages styleconditioned Mixture-of-Experts (MoE), pretrained CLIP and LoRA-adapted DINOv2.

![](images/80a1f32b324dd3d6ee672f6bf3b2064911320a1f7dec854c8feddbea8b5ef0a7.jpg)  
Figure 3. t-SNE visualization of model trace feature embeddings.

The baseline IntrAA-SD framework is trained and tested on our ISDB dataset. Elaborate ablation studies are also performed on the framework components, hyperparameters and the loss function, which reveal the significance of certain design choices. While our baseline approach achieves a high attribution accuracy of 99.03% accuracy for the ID test images of our dataset, it does reasonably well with 88.55% accuracy for the OOD test images. These performances are reasonably higher than those achieved by training the existing architecture-level source attribution models in our ISDB dataset, as those models are not designed for intraarchitecture source attribution. These findings suggest that there is immense scope for future investigations targeting development of intra-architecture source attribution models and compilation of more relevant datasets, picking up from our contribution.

## References

[1] Irene Amerini, Mauro Barni, Sebastiano Battiato, Paolo Bestagini, Giulia Boato, Vittoria Bruni, Roberto Caldelli, Francesco De Natale, Rocco De Nicola, Luca Guarnera, et al. Deepfake media forensics: Status and future challenges. Journal ofImaging, 11(3):73, 2025. 1, 2

[2] Quentin Bammey. Synthbuster: Towards detection of diffusion model generated images. IEEE Open Journal of Signal Processing, 5:1–9, 2024. 3

[3] Giulia Bertazzini, Daniele Baracchi, Dasara Shullani, Isao Echizen, and Alessandro Piva. DRAGON: A large-scale dataset of realistic images generated by diffusion models. arXiv preprint arXiv:2505.11257, 2025. 1, 2, 3, 6

[4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 1

[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In IEEE/CVF International Conference on Computer Vision, pages 9650–9660, 2021. 2, 6

[6] Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, and Luisa Verdoliva. On the detection of synthetic images generated by diffusion models. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 1, 3, 5, 6

[7] Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Nießner, and Luisa Verdoliva. Raising the bar of AI-generated image detection with CLIP. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4356–4366, 2024. 3

[8] dataautogpt3. Proteus v0.4. https://huggingface. co/dataautogpt3/ProteusV0.4, 2024. 3

[9] Ricard Durall, Margret Keuper, and Janis Keuper. Watch your up-convolution: CNN based generative deep neural networks are failing to reproduce spectral distributions. In IEEE/CVF conference on computer vision and pattern recognition, pages 7890–7899, 2020. 2

[10] Tarik Dzanic, Karan Shah, and Freddie Witherden. Fourier spectrum discrepancies in deep network generated images. Advances in neural information processing systems, 33: 3022–3032, 2020. 2

[11] David C Epstein, Ishan Jain, Oliver Wang, and Richard Zhang. Online detection of AI-generated images. In IEEE/CVF international conference on computer vision, pages 382–392, 2023. 3

[12] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022. 6

[13] Jianwei Fei, Yunshu Dai, Xiaoyu Zhou, Zhihua Xia, and Alessandro Piva. Enabling supervised learning of generative signatures for generalized AI-generated images detec-

tion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14084–14094, 2026. 1

[14] Joel Frank, Thorsten Eisenhofer, Lea Schonherr, Asja Fis-¨ cher, Dorothea Kolossa, and Thorsten Holz. Leveraging frequency analysis for deep fake image recognition. In International conference on machine learning, pages 3247–3258. PMLR, 2020. 1, 2, 6

[15] gsdf. Counterfeitxl. https://huggingface.co/ gsdf/CounterfeitXL, 2024. 3

[16] Luca Guarnera, Oliver Giudice, and Sebastiano Battiato. Mastering deepfake detection: A cutting-edge approach to distinguish gan and diffusion-model images. ACM Transactions on Multimedia Computing, Communications and Ap plications, 20(11):1–24, 2024. 1, 3

[17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS Workshop on Deep Generative Models and Downstream Applications, 2021. 5

[18] Edward J Hu, Yelong Shen, Phil Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In In ternational Conference on Learning Representations, 2022. 2, 6

[19] Anil K Jain. Fundamentals of digital image processing. Prentice-Hall, Inc., 1989. 2

[20] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35, 2022. 5

[21] Cagliostro Lab. Animagine xl 3.1. https : / / huggingface.co/cagliostrolab/animaginexl-3.1, 2024. 3

[22] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations, 2017. 7

[23] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 7

[24] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022. 5

[25] Lykon. Aam xl animemix. https://huggingface. co/Lykon/AAM\_XL\_AnimeMix, 2024. 3

[26] Lykon. Dreamshaper xl v2 turbo. https : / / huggingface.co/Lykon/DreamShaper-xl-v2- turbo, 2024. 3

[27] Midjourney, Inc. Midjourney. https://github.com/ midjourney, 2026. GitHub repository, accessed Jun. 26, 2026. 1

[28] Utkarsh Ojha, Yuheng Li, and Yong Jae Lee. Towards uni versal fake image detectors that generalize across generative models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24480–24489, 2023. 3

[29] Maxime Oquab, Timothee Darcet, et al. DINOv2: Learning´ robust visual features without supervision. Transactions on Machine Learning Research, 2024. 2, 5, 6

[30] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and¨ Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024. 1, 3

[31] Yiting Qu, Xinyue Shen, Xinlei He, Michael Backes, Savvas Zannettou, and Yang Zhang. Unsafe diffusion: On the generation of unsafe images and hateful memes from textto-image models. In ACM CCS, 2023. 1

[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 5, 6

[33] Jonas Ricker, Simon Damm, Thorsten Holz, and Asja Fischer. Towards the detection of diffusion model deepfakes. In International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, VISI-GRAPP, pages 446–457, 2024. 3, 6

[34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn-¨ thesis with latent diffusion models. In IEEE/CVF conference on computer vision and pattern recognition, pages 10684– 10695, 2022. 1

[35] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023. 1

[36] RunDiffusion. Juggernaut xl v9. https : / / huggingface.co/RunDiffusion/Juggernaut-XL-v9, 2024. 1, 3

[37] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 1

[38] SG161222. Realvisxl v4.0. https://huggingface. co/SG161222/RealVisXL\_V4.0, 2024. 3

[39] Zeyang Sha, Zheng Li, Ning Yu, and Yang Zhang. DE-FAKE: Detection and attribution of fake images generated by text-to-image generation models. In ACM SIGSAC conference on computer and communications security, pages 3418–3432, 2023. 1, 3

[40] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixtureof-experts layer. In International Conference on Learning Representations, 2017. 6

[41] stablediffusionapi. Copax timelessxl sdxl 1.0. https:// huggingface.co/stablediffusionapi/copaxtimelessxl-sdxl10, 2024. 3

[42] stablediffusionapi. Zavychromaxl. https : / / huggingface . co / stablediffusionapi / zavychromaxl, 2024. 3

[43] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016. 7

[44] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022. 5

[45] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnn-generated images are surprisingly easy to spot... for now. In IEEE/CVF conference on computer vision and pattern recognition, pages 8695–8704, 2020. 1, 2, 5, 6

[46] Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. DIRE for diffusion-generated image detection. In IEEE/CVF International Conference on Computer Vision, pages 22445–22455, 2023. 3

[47] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In 2020 Con ference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, 2020. 3

[48] Zhiyuan Yan, Jiangming Wang, Peng Jin, Ke-Yue Zhang, Chengchun Liu, Shen Chen, Taiping Yao, Shouhong Ding, Baoyuan Wu, and Li Yuan. Orthogonal subspace decomposition for generalizable AI-generated image detection. In International Conference on Machine Learning, pages 70268– 70288. PMLR, 2025. 1

[49] Tianyun Yang, Ziyao Huang, Juan Cao, Lei Li, and Xirong Li. Deepfake network architecture attribution. In AAAI Con ference on Artificial Intelligence, pages 4662–4670, 2022. 1

[50] Yntec. Realistic stock photo 3. https : / / huggingface . co Yntec / realisticStockPhoto3, 2024. Stable Diffusion model. Accessed: Feb. 28, 2026. 1

[51] Ning Yu, Larry S Davis, and Mario Fritz. Attributing fake images to GANS: Learning and analyzing GAN fingerprints. In IEEE/CVF international conference on computer vision, pages 7556–7566, 2019. 1

[52] Ning Yu, Vladislav Skripniuk, Sahar Abdelnabi, and Mario Fritz. Artificial fingerprinting for generative models: Rooting deepfake attribution in training data. In IEEE/CVF International conference on computer vision, pages 14448– 14457, 2021. 1

[53] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. CutMix: Regularization strategy to train strong classifiers with localizable features. In IEEE/CVF international conference on computer vision, pages 6023–6032, 2019. 7

[54] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations, 2018. 7

# Intra-Architecture Source Attribution for Stable Diffusion: A New Task with Benchmark Dataset and Baseline — Supplementary Document —

Ashutosh Dayal, Mallika Garg, and Debashis Sen Indian Institute of Technology Kharagpur, INDIA

ashutoshdayal@kgpian.iitkgp.ac.in, mallika@ec.iitr.ac.in, dsen@ece.iitkgp.ac.in

We provide the following important details about our contribution in this supplementary document:

• Continuing from Section 3.2 in the main paper, we describe a few more IntrAA-SD benchmark (ISDB) dataset In-domain (ID) and out-of-distribution (OOD) prompt categories in Section 1 here.

• Continuing from Section 4.1 in the main paper, we provide a visualization to analyze the style feature extraction using CLIP in Section 2 here.

• Continuing from Section 5.2 in the main paper, in Section 3 here, we discuss the intra-architecture source attribution results of some existing architecture-level source attribution methods when trained and tested in our ISDB dataset.

• In addition to the above, we dive into more experimental analysis of our IntrAA-SD intra-architecture source attribution baseline framework in Section 4 here.

## 1. Certain Prompt Categories of Our ISDB Dataset

## 1.1. In-Domain (ID) Generative Stress Test Prompts

We have considered 7 prompt categories in the ID set of our ISDB dataset that are designed to push the fine-tuned diffusion generators to their limits, both during training and inference. As in such situations the generative models are more likely to introduce pronounced generator-specific signatures in the images generated [1], they represent favorable cases for attribution methods. We discuss a few such generative stress test categories below.

• Texture and Geometric & Repetitive: Generating textures or repetitive patterns, which often represent high spatial frequencies, challenges generator decoders, especially their spatial up-sampling layers [8]. Different generator fine-tuning processes alter how the fine-tuned models handle aliasing, causing them to produce distinct frequency-domain model-related signatures within the generated image regions [3].

• Text & Signals: Rendering coherent text or a signal plot often leads to failure for the Stable Diffusion XL (SDXL) base model [6, 9]. Fine-tunes try to overcome this weakness by aggressively introducing structural corrections particularly to improve synthetic text generation. However, such attempts often leave model-specific traces in the generated areas containing characters, symbols or plots.

• Noise & Malicious: Such prompts provide minimal scope for semantic conditioning. So, without a meaningful object to render, the generator usually relies heavily on its underlying structural prior, which leaves its imprint in the generated image that aids source isolation without much interference from any visual content [13].

## 1.2. Out-Of-Distribution (OOD) Adversarial Prompts

We have considered 7 prompt categories in the OOD set of our ISDB dataset that are adversarial in nature to the finetuned generative models. The images thus generated tests the source attribution systems, as they must posses the capability of ignoring the uncharacteristic image style generated to a large extent in favor of model signature.

• Void Prompts: Prompts such as “Pure white background, nothing else” or “Complete darkness, empty frame” demand generation of images with no content. The models are usually not well trained for such generation, and hence, can produce arbitrarily-styled images with model signature camouflaged in them.

• Nested Representations: Prompts such as “A photograph of an oil painting displayed on a modern gallery wall” requires the generator to transition abruptly between distinct image types often producing anomalous /unseen distortions at boundaries, where model signature may be contained [5].

Table 1. Intra-architecture source attribution accuracy (%) on our ISDB test set. Models - M1: SDXL Base, M2: Juggernaut, M3: RealVis, M4: DreamShaper, M5: Animagine, M6: Counterfeit, M7: ZavyChroma, M8: Copax, M9: Proteus, M10: AAM Anime. Ours is the only algorithm (& the first) in the list which is designed for intra-architecture source attribution. All others listed are existing architecturelevel source attribution methods
<table><tr><td></td><td>Method</td><td>M1</td><td>M2</td><td>M3</td><td>M4</td><td>M5</td><td>M6</td><td>M7</td><td>M8</td><td>M9</td><td>M10</td><td>Overall</td></tr><tr><td rowspan="6">m</td><td>Spec [3]</td><td>18.13</td><td>12.96</td><td>11.86</td><td>46.45</td><td>40.83</td><td>36.51</td><td>9.31</td><td>20.49</td><td>32.01</td><td>21.32</td><td>24.99</td></tr><tr><td>UnivFD [5]</td><td>24.46</td><td>27.79</td><td>30.32</td><td>70.52</td><td>76.47</td><td>29.73</td><td>33.85</td><td>35.62</td><td>56.71</td><td>61.41</td><td>44.69</td></tr><tr><td>DE-FAKE [10]</td><td>62.96</td><td>65.37</td><td>61.32</td><td>89.58</td><td>93.91</td><td>76.24</td><td>62.28</td><td>67.97</td><td>87.87</td><td>86.22</td><td>75.37</td></tr><tr><td>NPR [11]</td><td>88.27</td><td>85.53</td><td>86.78</td><td>97.97</td><td>96.56</td><td>94.03</td><td>86.51</td><td>89.81</td><td>94.42</td><td>95.08</td><td>91.50</td></tr><tr><td>CNNDetect [12]</td><td>87.26</td><td>83.93</td><td>88.96</td><td>98.32</td><td>98.40</td><td>95.38</td><td>87.01</td><td>90.27</td><td>96.34</td><td>96.28</td><td>92.21</td></tr><tr><td>Ours*</td><td>99.42</td><td>99.79</td><td>98.67</td><td>99.83</td><td>99.92</td><td>98.29</td><td>99.67</td><td>97.88</td><td>98.58</td><td>98.21</td><td>99.03</td></tr><tr><td rowspan="6">000</td><td>Spec [3]</td><td>11.50</td><td>11.70</td><td>12.40</td><td>55.00</td><td>36.90</td><td>31.10</td><td>11.50</td><td>16.20</td><td>31.40</td><td>14.40</td><td>23.21</td></tr><tr><td>UnivFD [5]</td><td>20.60</td><td>22.00</td><td>26.50</td><td>78.10</td><td>74.80</td><td>25.60</td><td>27.40</td><td>32.20</td><td>61.70</td><td>59.00</td><td>42.79</td></tr><tr><td>DE-FAKE [10]</td><td>41.20</td><td>34.00</td><td>34.90</td><td>78.00</td><td>83.80</td><td>36.10</td><td>31.80</td><td>45.40</td><td>68.30</td><td>65.30</td><td>51.88</td></tr><tr><td>CNNDetect [12]</td><td>70.30</td><td>47.30</td><td>61.90</td><td>94.40</td><td>92.10</td><td>76.30</td><td>45.70</td><td>66.60</td><td>86.60</td><td>78.20</td><td>71.94</td></tr><tr><td>NPR [11]</td><td>75.20</td><td>53.60</td><td>61.90</td><td>95.10</td><td>92.00</td><td>79.60</td><td>54.20</td><td>71.10</td><td>83.80</td><td>81.90</td><td>74.84</td></tr><tr><td>Ours*</td><td>91.50</td><td>70.50</td><td>82.50</td><td>99.00</td><td>98.00</td><td>98.00</td><td>74.00</td><td>90.00</td><td>90.50</td><td>91.50</td><td>88.55</td></tr></table>

![](images/57e9c20f64d7539450c4a6521f41987b4e1b069a8a7030c1d7818ebb4a338049.jpg)  
Figure 1. t-SNE projection of frozen CLIP ViT-L/14 features of input images generated using a suitable shared prompt across all 10 SDXL fine-tunes.

## 2. Style Capture using Pretrained CLIP

In Section 4.1 of the main paper, we state that frozen CLIP [7] would be able to capture visual style representative features from the input image. To empirically verify this property on our ISDB dataset, we visualize the t-SNE projection of the frozen CLIP features extracted from all the 10 SDXL fine-tunes using the same single prompt suitable for all in Fig. 1. So, for an input image I, we take a 518 × 518 crop and resize it to 224 × 224 for CLIP, which produces a style representation feature $\mathbf { x } _ { \mathrm { s t y l e } } \in \mathbb { R } ^ { d _ { \mathrm { c l i p } } }$ where $d _ { \mathrm { c l i p } } = 1 0 2 4$ . The t-SNE plot in the figure shows the projection of $\mathbf { x } _ { \mathrm { s t y l e } }$ for the input image as a point. As can be seen, the anime-styled images generated from fine-tuned SDXL anime image generators Animagine (M5), Counterfeit (M6) and AAM Anime (M10) are reasonably separated from the images generated by stylized (M7, M8, M9) and photorealistic (M1, M2, M3, M4) fine-tuned generators derived from SDXL (see Table 1 in the main paper for the style categories of the SDXL fine tunes considered). The stylized and photorealitic generated images are somewhat separated with former mostly occupying the bottom & middle, and the latter occupying the middle & top-right parts of the plot. Therefore, although we visualize the 1024-d vectors in an optimized 2-d view, we still find some impression of separation between them in terms of image style.

## 3. Intra-architecture Attribution Results

Table 1 lists the intra-architecture source attribution results obtained for our IntrAA-SD baseline framework on our ISDB dataset, which is also reported in Table 4 of the main paper. However, in Table 1 here, the performance of five other well-known recent methods, which are designed to perform architecture-level source attribution including diffusion-based generators, are reported. As shown in the table, when existing architectural-level source attribution models are trained on our training set, the best average accuracies achieved on our test set are 92.21% and 74.84% for ID and OOD cases, respectively. Their performance remains substantially below our intra-architecture source attribution baseline, which is specifically designed for intraarchitecture attribution. This suggests that the development of dedicated intra-architecture attribution techniques is necessary continuing from our baseline framework, particularly to improve generalization on the OOD test set of the ISDB dataset.

Table 2. Mean Mixture of Expert (MoE) gating weights (%) per model, grouped & averaged by style type.
<table><tr><td rowspan="2">Experts</td><td colspan="5">Photorealistic</td><td colspan="4">Stylized</td><td colspan="4">Anime</td></tr><tr><td>M4</td><td>M2</td><td>M1</td><td>M3</td><td>avg.</td><td>M8</td><td>M7</td><td>M9</td><td>avg.</td><td>M6</td><td>M10</td><td>M5</td><td>avg.</td></tr><tr><td>Expert 1</td><td>35.5</td><td>36.7</td><td>37.1</td><td>37.6</td><td>36.7</td><td>37.2</td><td>36.6</td><td>34.9</td><td>36.2</td><td>33.5</td><td>28.1</td><td>15.8</td><td>25.8</td></tr><tr><td>Expert 2</td><td>24.9</td><td>24.8</td><td>23.0</td><td>25.2</td><td>24.5</td><td>26.3</td><td>26.8</td><td>27.2</td><td>26.8</td><td>27.2</td><td>29.2</td><td>27.4</td><td>27.9</td></tr><tr><td>Expert 3</td><td>23.7</td><td>21.6</td><td>23.6</td><td>21.2</td><td>22.5</td><td>18.8</td><td>19.5</td><td>19.6</td><td>19.3</td><td>17.8</td><td>17.4</td><td>13.3</td><td>16.2</td></tr><tr><td>Expert 4</td><td>15.9</td><td>16.9</td><td>16.2</td><td>16.1</td><td>16.3</td><td>17.7</td><td>17.1</td><td>18.3</td><td>17.7</td><td>21.6</td><td>25.3</td><td>43.6</td><td>30.2</td></tr></table>

Table 3. Cross-Dataset Performance of Our IntrAA-SD baseline
<table><tr><td rowspan="2">Training Set</td><td colspan="3">Test Set Accuracy %</td></tr><tr><td>ISDB-ID</td><td>ImageNet</td><td>COCO</td></tr><tr><td>ImageNet</td><td>80.59</td><td>98.35</td><td>96.47</td></tr><tr><td>COCO</td><td>76.16</td><td>90.31</td><td>98.91</td></tr><tr><td>ISDB-ID (Ours)</td><td>99.03</td><td>84.58</td><td>85.95</td></tr></table>

## 4. Additional Experiments

## 4.1. Mixture of Experts (MoE) Gating Weights

Table 2 reports the mean MoE gating weights for each expert per fine-tuned model, and the average over the models for a particular style-type is also provided. As can be seen, the Experts 1-4 respectively contribute on an average around 37%, 25%, 23%, and 16% for the photorealistic models, whereas they in that order contribute on an average 36%, 27%, 19%, & 18%, and 26%, 28%, 16%, & 30% for the stylized and Anime models, respectively. Therefore, the distribution of the average contributions by the four experts change with the style-type of the models. This signifies that differently-styled generated images might have structurally different source model signatures as stated in Section 4.3 of the main paper, which can be captured by learning different distribution of weighting multiple experts for computing model trace features.

## 4.2. Cross-dataset Intra-Architecture Attribution Performance

Datasets like ImageNet [2] and COCO [4], which contain captions along with images, have primarily been used for object /image recognition purpose. Here, we study the appropriateness of using such datasets to train intraarchitecture source attribution models instead of using specially curated image & prompt set like the ID set in ISDB dataset (ISDB-ID), and also perform cross-dataset performance analys of our IntrAA-SD baseline approach for intraarchitecture attribution.

So, we first prepare two datasets from the 10 SDXL finetuned models using the image-caption pairs from ImageNet and COCO datasets for training two variants of our IntrAA-SD baseline approach. The first dataset includes 12,000 images per fine-tuned model generated using ImageNet captions as prompts, and the second has 12,000 images per fine-tuned model generated using the COCO captions. We split the each set into 80/20 train and test partitions using stratified sampling with a fixed random state, ensuring no prompt overlap between splits. We test our ISDB-ID trained IntrAA-SD baseline on these datasets. We also test our IntrAA-SD baseline variants, which are trained from scratch on the two training sets formed from ImageNet and COCO datasets, on our ISDB-ID test set.

The results of the cross-dataset experiment is shown in Table 3. As can be seen, when the training and testing of our IntrAA-SD baseline are done on the same dataset, the performance achieved is quite high, irrespective of the datasets. However, the performance drops to different extents in the different cases when the the training and testing are done on different datasets. If the training and testing are done on ImageNet and COCO, one or the other, the performance achieved is still above 90%, which signifies the similarity of two datasets derived from them. If the training is done on ImageNet or COCO derived datasets, and the testing on our ISDB-ID test set, the performance barely touches 80%. Whereas when the training is done on our ISDB-ID train set, and the testing on ImageNet or COCO, the performance manages to reach around 85%.

These observations suggests that the cross-dataset performance of our IntrAA-SD baseline is just satisfactory leaving scope for future improvement. Further, another important finding is that training on our ISDB-ID train set imbibes our baseline approach with slightly better generalization capability achieving around 5% accuracy boost. This highlights the importance of such specially curated datasets for intra-architecture source attribution.

## 4.3. Confusion Matrix ofIntrAA-SD baseline on OOD set in ISDB dataset

From Table 4 in the main paper, we see that our IntrAA-SD baseline trained on our ISDB-ID training set achieves 88.55% attribution accuracy on the OOD test set of the ISDB dataset (ISDB-OOD). We further examine here the related confusion matrix in Table 4. A salient observation from the matrix is that images generated from a majority of the photorealistic and stylized image generators are prone to be misclassified as from the SDXL-Base source. This is understandable as SDXL-Base is categorized under photorealistic generator producing images with styles closer to other photorealistic and stylized category fine-tunes rather than the anime category ones. In fact, images from the fine-tune ZavyChroma (M7) in the stylized category has got misclassified as obtained from three of the four photorealistic fine-tunes. Another key observation is that slightly higher (around 8−15%) misclassifications has occurred just between photorealistic fine-tunes Juggernaut (M2) and RealVis (M3), as they produce images with very similar visual styles. The above findings point to paths for designing improved intra-architecture attribution methods in future.

Table 4. Confusion Matrix related to performance of our IntrAA-Diff on ISDB-ODD set. Photorealistic: M1-4, Stylized: M7-9, Anime: M5-6, M10.
<table><tr><td>True ↓ / Pred →</td><td>M1</td><td>M2</td><td>M3</td><td>M4</td><td>M5</td><td>M6</td><td>M7</td><td>M8</td><td>M9</td><td>M10</td></tr><tr><td>M1 (SDXL Base)</td><td>923</td><td>3</td><td>2</td><td>0</td><td>1</td><td>5</td><td>11</td><td>46</td><td>5</td><td>4</td></tr><tr><td>M2 (Juggernaut)</td><td>22</td><td>708</td><td>149</td><td>8</td><td>0</td><td>14</td><td>49</td><td>28</td><td>16</td><td>6</td></tr><tr><td>M3 (RealVis)</td><td>16</td><td>87</td><td>816</td><td>4</td><td>0</td><td>12</td><td>35</td><td>18</td><td>9</td><td>3</td></tr><tr><td>M4 (DreamShaper)</td><td>0</td><td>5</td><td>0</td><td>985</td><td>0</td><td>0</td><td>4</td><td>0</td><td>1</td><td>5</td></tr><tr><td>M5 (Animagine)</td><td>4</td><td>0</td><td>0</td><td>0</td><td>984</td><td>1</td><td>3</td><td>0</td><td>0</td><td>8</td></tr><tr><td>M6 (Counterfeit)</td><td>1</td><td>3</td><td>0</td><td>1</td><td>2</td><td>978</td><td>5</td><td>0</td><td>3</td><td>7</td></tr><tr><td>M7 (ZavyChroma)</td><td>57</td><td>45</td><td>43</td><td>4</td><td>1</td><td>15</td><td>742</td><td>50</td><td>25</td><td>18</td></tr><tr><td>M8 (Copax)</td><td>51</td><td>3</td><td>4</td><td>2</td><td>0</td><td>6</td><td>20</td><td>908</td><td>4</td><td>2</td></tr><tr><td>M9 (Proteus)</td><td>8</td><td>16</td><td>12</td><td>3</td><td>5</td><td>1</td><td>7</td><td>8</td><td>940</td><td>0</td></tr><tr><td>M10 (AAM Anime)</td><td>6</td><td>1</td><td>0</td><td>5</td><td>8</td><td>7</td><td>24</td><td>10</td><td>0</td><td>939</td></tr></table>

![](images/684d7df11b7292527a6da6ba1292fe5a8f1be40c6f2aa1d9885c3e3e67c410b6.jpg)  
Figure 2. Significant changes (Red: increase, Blue: decrease) at patch-level feature norm after LoRA adaptation of DINOv2 for images from four different fine-tuned models.

## 4.4. Effect of LoRA on Feature Representations

As mentioned in Section 4.2 of the main paper, our IntAA-SD baseline model contains a LoRA-adapted DinoV2 to extract fine-tuned model’s trace features from the input image. To visualize the DINOv2’s internal representation modification by LoRA adaptation, we compute the $L _ { 2 }$ norm of each patch token from the final transformer layer for both frozen DINOv2 and LoRA-adapted DINOv2. For each of the $3 7 \times 3 7$ spatial patches (h vector), we measure the relative change $\Delta _ { i } ~ = ~ ( \| \mathbf { h } _ { i } ^ { \mathrm { L o R A + D I N O } } \| _ { 2 } ~ { - } ~ \| \mathbf { h } _ { i } ^ { \mathrm { f r o z e n . D I N O } } \| _ { 2 } ) / \| \mathbf { h } _ { i } ^ { \mathrm { f r o z e n . D I N O } } \| _ { 2 }$ and mark only the patches in the top $2 5 ^ { \mathrm { t h } }$ percentile of $| \Delta _ { i } |$ for visualization in Fig. 2. Among those, positive $\Delta _ { i }$ patches are shown in red, which can be seen is mostly concentrated on smooth and less-textured regions. The negative $\Delta _ { i }$ patches are shown in blue, which we see appear in all regions including salient regions like faces, knots, and edges &corners. Such observations are found to be consistent across images from the different fine-tuned generators. Therefore, LoRA seems to increase (red patches) the magnitude of the features in the smooth, low-texture patches, indicating more likelihood of discernible model signature presence in these regions. Analogously, it also suppresses (blue patches) the magnitude of features at a variety of regions which include salient ones, where model signature might be well-camouflaged in general.

## References

[1] Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Gio vanni Poggi, Koki Nagano, and Luisa Verdoliva. On the detection of synthetic images generated by diffusion mod els. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 1

[2] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 3

[3] Joel Frank, Thorsten Eisenhofer, Lea Schonherr, Asja Fis-¨

cher, Dorothea Kolossa, and Thorsten Holz. Leveraging frequency analysis for deep fake image recognition. In International conference on machine learning, pages 3247–3258. PMLR, 2020. 1, 2

[4] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence´ Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 3

[5] Utkarsh Ojha, Yuheng Li, and Yong Jae Lee. Towards universal fake image detectors that generalize across generative models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24480–24489, 2023. 1, 2

[6] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and¨ Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024. 1

[7] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2

[8] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn-¨ thesis with latent diffusion models. In IEEE/CVF conference on computer vision and pattern recognition, pages 10684– 10695, 2022. 1

[9] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 1

[10] Zeyang Sha, Zheng Li, Ning Yu, and Yang Zhang. DE-FAKE: Detection and attribution of fake images generated by text-to-image generation models. In ACM SIGSAC conference on computer and communications security, pages 3418–3432, 2023. 2

[11] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, and Yunchao Wei. Learning on gradients: Generalized artifacts representation for gan-generated images detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12105–12114, 2023. 2

[12] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnn-generated images are surprisingly easy to spot... for now. In IEEE/CVF conference on computer vision and pattern recognition, pages 8695–8704, 2020. 2

[13] Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. DIRE for diffusion-generated image detection. In IEEE/CVF International Conference on Computer Vision, pages 22445–22455, 2023. 1