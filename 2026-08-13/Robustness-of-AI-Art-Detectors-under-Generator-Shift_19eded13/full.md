# Robustness of AI-Art Detectors under Generator Shift

Shivank Singh Thakur<sup>∗</sup> Meien Li<sup>∗</sup> Mark Stamp<sup>∗†</sup>

August 13, 2026

## Abstract

Text-to-image generative models have advanced rapidly, with modern Difusion Transformer architectures producing images that are increasingly dificult to distinguish from human-created artwork. This development has raised significant concerns regarding copyright protection, misinformation, fraud, impersonation, and the authenticity of digital content. Most AI-art detectors are trained and evaluated on the same generator family, leaving robustness to newer architectures underexplored. In this chapter, we analyze generator shift based on a Stable Difusion 3.5 Medium (SD3.5m) artwork dataset spanning ten art styles through reverse prompting of held-out human artwork samples. Five detectors are trained on U-Net-based latent difusion artwork and evaluated in a zero-shot cross-generator setting on the SD3.5m dataset. Deep learning models perform strongly in-distribution but degrade under generator shift, misclassifying many SD3.5m images as human while human false positives remain low. The CLIP ViT-L/14 model performs best overall, while Grad-CAM analysis reveals weaker and more difuse activation on false negatives. These findings highlight a generalization gap in current AI-art detectors and motivate the development of detectors as one component of a layered defense that remains reliable across rapidly evolving generative architectures.

Keywords: AI-Art Detection · Stable Difusion · CLIP ViT · ResNet · EficientNet · ConvNeXt · Grad-CAM

## 1 Introduction

Text-to-image generative models have greatly improved in their ability to synthesize high-quality artwork from natural language descriptions. Given a text prompt, these models iteratively denoise random noise into a high-resolution image that matches the requested content and style. As architectures, training datasets, and compute budgets have scaled up, modern systems such as DALL·E, Midjourney, and Stable Difusion now produce images that are often dificult to distinguish from humancreated artwork [42, 44]. Newer versions, such as Stable Difusion 3.5 and its variants including Medium, Large, and Turbo, further reduce visible artifacts and generate more coherent textures and fine details than earlier models [13].

This rapid improvement creates a security and trust problem. AI-generated art can be used for misinformation, fraud, and impersonation, especially when there is no reliable metadata about how the image was created. In many real-world scenarios (social media, messaging apps, reposted or edited content), cryptographic signatures or provenance tags may be missing or unreliable. This motivates continued interest in image-based detection as an important layer of defense for estimating whether an artwork is human-created or AI-generated.

A central challenge in this setting is generator shift. Most existing AI-image detection studies train and evaluate detectors on images from the same generator family or from closely related generators (for example, Latent Difusion models or GANs) [15, 44, 52]. As a result, detectors may perform well on controlled benchmark datasets, but their reliability in real deployment is far more uncertain. The images encountered after deployment may come from newer generative models that were not included during training. This chapter considers such a setting by asking whether detectors trained on artwork from earlier U-Net-based difusion models remain reliable when evaluated on Stable Difusion 3.5 Medium (SD3.5m), a newer Difusion Transformer-based model [13].

We model this deployment setting as an emerging-generator threat model. We assume that the defender operates an image-based detector as a screening component at a social media platform, a news verification desk, or a content-moderation pipeline. The detector uses a fixed operating threshold and has no access to labeled images from future generators. The opposing role is a synthetic-media producer who generates images using tools that were not available during detector training. Importantly, this setting does not require a sophisticated adversary who crafts adversarial perturbations or optimizes directly against the detector. The producer can simply adopt a newer publicly available generator as stronger generative models continue to emerge and become widely accessible. This creates a dynamic similar to malware detection, where new malware families may appear after a classifier has already been deployed. The threat distribution changes whereas the defensive model remains fixed and the resulting reliability gap can silently grow over time. The main operational risk is therefore a false negative, where AI-generated content is passed as human-created. This leads to an evasion-like failure mode in which the detector appears reliable on known generators but misses a substantial fraction of synthetic images from the newer generator. The formal threat model and evaluation scope are defined in Section 3.1.

In this chapter, we investigate detector robustness under generator shift using a prompt-aligned SD3.5m dataset. The dataset is constructed across 10 art styles from held-out human artwork samples via reverse prompting. Five deep detector backbones are considered (two ResNet models [17], EficientNet-B0 [51], ConvNeXt-Base [29], and CLIP ViT-L/14 [39]). These models are trained on in-distribution (ID) artwork from Latent Difusion (LDM) and Stable Difusion 2.1 (SD2.1), then evaluated under a shared protocol on both ID and out-of-distribution (OOD) data. This design makes it possible to assess overall OOD degradation while also examining how the performance varies across styles and source generators.

The main contributions of this work are the following.

1. A prompt-aligned Stable Difusion 3.5 Medium dataset of 10,000 images across ten art styles, constructed through reverse prompting of human artwork.

2. A systematic evaluation of five deep learning detectors under generator shift, measuring both in-distribution and out-of-distribution performance.

3. A qualitative failure analysis of the case where detection fails on the newer generator, with an emphasis on how such failure varies across art styles.

4. A security-oriented analysis of generator shift as an emerging synthetic-media threat, including the implications of false-negative failures for defensive contentscreening and authenticity workflows.

The remainder of this chapter is organized as follows. Section 2 reviews background on image generation architectures, prompt engineering, detection methodologies, and relevant datasets. Section 3 describes dataset construction and detector implementation. Section 4 presents our experimental results and analysis. Finally, Section 5 concludes this study and outlines possible directions for future work.

## 2 Background and Related Work

Artificial intelligence has become a major force in visual art and image synthesis. What began as experimental algorithmic graphics has turned into an industrial-scale capability where generative models can produce images that are often indistinguishable from human-created artwork. This shift has driven parallel advances in forensic methods and computational models of aesthetics. Modern AI systems can generate, analyze, critique, and imitate artistic styles. This raises practical questions about originality, content provenance, and even the reliability of visual evidence.

## 2.1 Computational Aesthetics: Can AI Understand Art?

In addition to producing pixels, modern systems can evaluate images, describe them, and optimize toward aesthetic preferences. Early computer vision systems focused on low-level features like color histograms, edges and textures. Modern Multimodal Large Language Models (MLLMs) go much further, as they can perform structured art analysis (composition, style, mood) and often agree with human judgments about what looks “good” or “beautiful” [21].

By mapping images and text into a shared representation space, models such as CLIP (Contrastive Language-Image Pre-training) can retrieve and classify artwork by abstract concepts like “melancholy” or “sublime” and also detect concrete objects. This enables aesthetic reward models that steer generation toward humanpreferred styles [21].

However, such model “understanding” is statistical and not experiential. The model learns correlations in a high-dimensional latent space. For example, a model can recognize Impressionist brushwork or Renaissance composition patterns, but it does not experience emotion or meaning the way human viewers do [8]. Recent work adding chain-of-thought-style reasoning to MLLMs narrows this gap by having the model explicitly explain its judgments, which makes it appear to be more like an art critic [21].

## 2.2 Evolution of Image Generation Architectures

Modern AI art has gone through three main phases: early visualization methods, generative adversarial networks (GANs) and difusion models. Each phase improved fidelity, diversity, and controllability, and each left diferent forensic traces.

## 2.2.1 Pre-Generative Era: DeepDream and Style Transfer (2015- 2017)

Google’s DeepDream (2015) is often cited as a starting point for modern AI art [32]. DeepDream used convolutional neural networks (CNNs) trained for image classification and inverted them: instead of predicting labels from an image, it optimized the input image to strongly activate specific neurons or layers. This produced surreal, hallucination-like images dominated by patterns the network had learned (e.g., dog faces, eyes). DeepDream was deterministic and limited by the features learned for classification.

Neural Style Transfer (NST) separated content (the structure of a scene) from style (texture and color statistics). It allowed users to apply the style of artists such as Van Gogh or Picasso to a photograph, making AI-generated art widely accessible as a filter. However, since NST is an image-to-image translation method rather than a fully generative model, it stylizes existing content instead of creating new content from scratch [14].

## 2.2.2 Adversarial Era: GANs (2017-2021)

Generative Adversarial Networks (GANs) introduced adversarial training as a powerful framework for image generation [15]. In this setup, a generator network creates synthetic images from latent noise, while a discriminator tries to distinguish generated images from real images. Training proceeds as a two-player game in which the generator improves by learning to fool the discriminator.

Key developments include the following.

• StyleGAN and BigGAN delivered high-fidelity, high-resolution images. Style-GAN in particular allowed control over the style at diferent spatial scales (coarse, medium, fine), enabling realistic control over attributes like pose, lighting, or facial features [6, 24].

• GAN-based models trained on artwork datasets such as WikiArt [33] could produce novel paintings that resembled established art movements. The auction of the GAN-generated Portrait of Edmond de Belamy symbolized the growing cultural and commercial visibility of AI-generated art [9].

Despite their success, GANs have several limitations. One common problem is mode collapse, where the generator produces a limited diversity of images that still fool the discriminator. GAN-generated images may also contain characteristic spatial or high-frequency artifacts, including checkerboard-like patterns often associated with upsampling operations. These artifacts can provide useful signals for forensic detection [58].

## 2.2.3 Difusion Era: Midjourney, DALL·E and Stable Difusion (2021-Present)

Difusion models use a probabilistic iterative denoising approach. They are trained to reverse a gradual noising process. Starting from a noisy image, they learn to denoise it step by step. At inference time, generation begins from random noise and proceeds through multiple denoising steps.

Latent Difusion Models (LDM) achieve a balance between computational eficiency and high-fidelity generation by operating in a compressed latent space rather than pixel space [56]. Advantages of difusion models include better training stability and wider coverage of the data distribution as compared to GANs, as well as strong controllability when conditioned on text embeddings (for example, from models like CLIP or T5). This enables more detailed and precise prompt adherence [40].

## 2.3 Misuse of AI-Generated Artwork

Generative artificial intelligence (GenAI) has become increasingly widespread, substantially lowering the technical barriers to digital art creation and making creative tools more accessible to the general public. For example, text-to-image generative models such as Stable Difusion became publicly available through online APIs and open-source repositories in mid-to-late 2022. By 2024, these models had attracted tens of millions of users [59]. Many researchers have emphasized the eficiency and potential of this technology for educational and academic applications. However, alongside these benefits, the misuse of GenAI has raised growing concerns, particularly regarding the unethical use and exploitation of copyrighted materials.

The harms associated with GenAI models have not yet been efectively addressed from a regulatory perspective. This is due to a combination of policy-related challenges, including inadequate regulation of social media platforms, unresolved ethical concerns, legal uncertainty, and the rapid pace of technological development.

For example, many artists have adopted various methods to prevent their work from being incorporated into GenAI training datasets, such as applying watermarks or using other protective measures [48]. However, these approaches do not fully deter malicious actors who continue to develop increasingly sophisticated techniques for reproducing or imitating distinctive artistic styles.

At the same time, high-quality AI-generated artwork can be used to make disinformation more persuasive and dificult to detect. Consequently, preserving the authenticity of visual content has become increasingly challenging. This erosion of trust in digital media may undermine public confidence and make it more dificult for individuals to distinguish genuine content from fabricated information.

## 2.3.1 Copyright Concerns

With the widespread adoption of the Internet, digitization of traditional artwork, online publication, and sharing of digital art have become common practices. Digital platforms provide artists with practical ways to preserve copies of their work, promote their creations, and reach broader audiences. In sharing their work online, artists often rely on copyright protection policies and the expectation that their work will not be used without authorization.

Over many years, the collective creative output of artists has contributed to the development of vast online image collections. These collections have subsequently provided the foundation for the large datasets required to train generative AI models. However, artists often have limited opportunities to refuse or prevent the unauthorized inclusion of their work in such datasets. Moreover, AI-generated artwork can blur the boundaries of ownership and authorship, particularly because relevant legal frameworks remain incomplete or insuficiently developed [5, 45]. This legal ambiguity poses significant risks to the protection of artists’ intellectual property rights.

## 2.3.2 Misinformation

The rapid development of GenAI, particularly difusion-based image-generation models, has enabled the fast and convincing imitation of artists’ styles. It has also increased the realism and persuasiveness of fabricated identities and visual content, potentially contributing to social disruption [45].

An artist’s body of work and distinctive style are integral components of their artistic identity. However, these characteristics can be easily exploited through generative technologies. For example, an artist’s style may be imitated to appropriate its commercial value, while the artist’s identity may be misrepresented to disseminate misinformation, rumors, or harmful content. Such misuse may damage the artist’s reputation and weaken public trust in visual media. More broadly, the increasing prevalence of convincing fabricated content may contribute to widespread distrust and make reliable information more dificult to identify [35].

## 2.4 Prompt Engineering and Inversion

The emergence of GenAI systems has altered the landscape of digital art. Unlike the traditional image generation models, current GenAI systems are capable of producing high-quality images that closely mimic the styles of human artists, which has introduced a critical threat to the intellectual property rights of human artists. A particular type of threat involves an adversary passing an artist’s original artwork through a vision-language model to obtain a natural language description, to later feed into a GenAI model to produce a stylistically similar image. Because no pixel is directly reproduced, such attacks are challenging and dificult to detect. Since these attacks rely on textual descriptions to transfer stylistic information from an original artwork to a generative model, understanding the role of prompts becomes essential.

## 2.4.1 Image Captioning as Prompt

Prompt engineering is the practice of crafting text inputs that steer the model toward a desired style, composition, or mood. This often relies on modifiers i.e., keywords that the model has learned to associate with specific visual attributes, such as “unreal engine”, “octane render”, or “volumetric lighting”.

Image captioning models like BLIP [25] or LLaVA [28] can generate prompts from reference images helping automate prompt creation. However, generic captions usually describe objects and actions and often miss fine-grained stylistic details (brushwork, camera model, rendering engine) needed to reproduce an artwork’s look.

## 2.4.2 Reverse Prompt Engineering and Prompt Inversion

Reverse prompt engineering (RPE) aims to reconstruct a prompt that could possibly have generated a given image. Common methods include the following.

• PEZ (Hard Prompts made EaZy) uses gradient-based optimization on discrete tokens to minimize CLIP loss between the prompt and target image. It often discovers trigger tokens that work numerically but do not have a humanreadable meaning (for example, “horse cat 8k!!”) [54].

• CLIP Interrogator combines a captioning model with a large database of style tags, then searches for tags that maximize CLIP similarity. This produces human-readable prompts like “digital art by [Artist], trending on ArtStation” but performance is limited by the tag library [38].

• ARPO (Automatic Reverse Prompt Optimization) iteratively generates images from candidate prompts, compares them to the target and updates the prompt. It can find high-fidelity prompts that are easy to edit, but is computationally expensive because it requires many generation steps [43].

These methods converge on prompts that are semantically close to the original, efectively reconstructing a textual specification of the generation process that can be used for attribution, reproducibility or further analysis.

## 2.5 Methodologies for Detecting AI-Generated Artwork

As generative models have improved, detection has moved from obvious visual glitches to more subtle statistical, semantic, and model-based cues. Current methods can be grouped into artifact analysis, generative fingerprinting and semantic reasoning.

## 2.5.1 Artifact Analysis: Frequency and Spatial Domains

Frequency-domain analysis uses techniques like the Fast Fourier Transform (FFT) to study the spectrum of an image. GANs with transposed convolutions can leave periodic patterns that show up as spikes or checkerboard structures in the frequency domain. Similar but more subtle artifacts can sometimes be found in difusion images due to their deconvolution modules [10].

Spatial-domain analysis focuses directly on pixel patterns, such as texture inconsistencies or unusual local noise structures. These signals are fragile and common post-processing operations (like JPEG compression, blurring and resizing) can weaken or remove them, which makes pure artifact-based detectors vulnerable [7].

## 2.5.2 Generative Fingerprinting: Difusion Reconstruction Error

DIRE (Difusion Reconstruction Error) uses a difusion model as a detector instead of only as a generator [53]. The method assumes that if an image was created by a given difusion model, it lies close to that model’s image manifold. In practice, the image is first inverted into a noisy latent using the same model, then reconstructed by running the difusion process forward and finally an error is computed between the original and reconstructed images. Synthetic images that match the model tend to reconstruct with low error, while real images or out-of-distribution images generally reconstruct with higher error because important details are lost in the inversion–reconstruction cycle. DIRE is efective even for unseen difusion models because it exploits properties of the difusion process itself rather than specific visual artifacts. Its main drawback is computational cost due to repeated denoising steps.

## 2.5.3 Human Perception and Semantics

Human judgment is still important even with strong automatic detectors. Recent work that compares human and machine performance on real versus AI-generated images shows that people are unreliable at this task, as they tend to rely more on semantic and physical cues than on low-level artifacts. Typical signals include lighting and shadows (e.g., inconsistent shadow directions, incorrect reflections, or specular highlights that do not match the scene’s light sources) and anatomy and geometry (e.g., subtle deformations in hands, faces, or object boundaries, or physically impossible arrangements such as floating objects) [23]. Benchmarks such as AnomReason [50] explicitly test whether models can detect these kinds of semantic and physics-based anomalies rather than only exploiting texture statistics.

Empirical studies report that human participants misclassify a large fraction of images (real versus AI), while the best automatic models achieve substantially lower error rates on clean, in-distribution data [16]. However, automatic detectors can be fragile under adversarial perturbations, meaning small and carefully designed changes to pixels can drastically reduce their accuracy, even when the images still look the same to humans. For artwork, large-scale studies with crowdworkers and expert artists show a similar pattern. Experts tend to outperform non-experts and are better at noticing semantic and physical inconsistencies, but they also exhibit a “skepticism bias”, i.e., they label unusual or abstract human art as AI-generated, which can lead to a high false-positive rate [16].

Overall, these results suggest that humans and models make diferent kinds of errors. Humans are more robust to small pixel changes and better at reasoning about semantics and physical errors, but are biased and overly-suspicious especially for atypical styles. Models are more consistent and sensitive to subtle statistical patterns, but can be fooled by low-level perturbations and shifts in distribution and often struggle with semantic anomaly detection.

## 2.6 Deep Learning for Detection

Deep learning architectures for visual recognition have evolved rapidly, progressing from early convolutional networks through residual learning, eficient scaling, and modernized CNN designs, to transformer-based models that capture global spatial dependencies. This section reviews the key architectural developments most relevant to image classification and representation learning. Table 1 provides a summary comparison of the models used in this study.

Table 1: Summary of deep learning backbone architectures
<table><tr><td>Model</td><td>Architecture</td><td>Parameters</td><td>Year</td><td>Pretraining</td></tr><tr><td>ResNet-18 [17]</td><td>Residual CNN</td><td>11.7M</td><td>2016</td><td>ImageNet-1K</td></tr><tr><td>ResNet-50 [17]</td><td>Residual CNN</td><td>25.6M</td><td>2016</td><td>ImageNet-1K</td></tr><tr><td>EfficientNet-B0 [51]</td><td>Compound-scaled CNN</td><td>5.3M</td><td>2019</td><td>ImageNet-1K</td></tr><tr><td>CLIP ViT-L/14 [39]</td><td>Vision Transformer</td><td>307.4M</td><td>2021</td><td>400M image-text pairs</td></tr><tr><td>ConvNeXt-Base [29]</td><td>Modernized CNN</td><td>88.6M</td><td>2022</td><td>ImageNet-1K</td></tr></table>

## 2.6.1 Residual Networks

Training very deep convolutional networks is hindered by the degradation problem, where adding more layers increases training error, suggesting an optimization dificulty rather than overfitting. Residual learning addresses this by introducing skip connections that add the block input directly to its output, so the network learns residual mappings rather than full unreferenced mappings, allowing gradients to propagate efectively through very deep architectures. The resulting ResNet family, spanning 18 to over 150 layers, achieved state-of-the-art results on ImageNet and became a foundational backbone architecture widely adopted across detection, segmentation, and representation learning tasks [17, 46].

## 2.6.2 EficientNet

Scaling network depth, width, and input resolution independently yields suboptimal results, as these dimensions are interdependent. Compound scaling addresses this by uniformly scaling all three dimensions simultaneously using a fixed coeficient derived from a constrained grid search, producing models that achieve significantly better accuracy while using fewer parameters and less computation than many earlier models. The EficientNet family showed that scaling strategy is just as important as choosing the model design itself [51].

## 2.6.3 Vision Transformers and CLIP

Transformer architectures were adapted to image recognition by splitting images into fixed-size patches, projecting each patch into a token embedding, and applying multi-head self-attention across the full sequence [11]. Unlike convolutional networks, this approach captures global spatial relationships from the first layer without locality constraints, and achieves strong performance when pretrained on suficiently large datasets. This was extended further by training a Vision Transformer (ViT) through contrastive learning on 400 million image-text pairs, aligning visual and textual representations in a shared embedding space [39]. This Contrastive Language-Image Pretraining (CLIP) objective produces semantic, transferable representations that generalize broadly across tasks without task-specific fine-tuning, and has become one of the most widely used backbones for zero-shot and transfer learning applications.

## 2.6.4 ConvNeXt

As ViT gained traction, the convolutional design space was revisited by systematically applying transformer-inspired choices (larger depthwise kernels, inverted bottleneck blocks, layer normalization, and GELU activations [18]) to a standard ResNet backbone. The resulting ConvNeXt family matches ViT performance on standard benchmarks while retaining the computational eficiency of convolutional architectures, demonstrating that the CNN-transformer performance gap was largely attributable to training procedures and design choices rather than a fundamental limitation of convolutions [29].

## 2.7 Public Datasets for AI Artwork Detection

A variety of public datasets are available for AI-image and AI-artwork detection covering diferent domains, generators, and threat models. Some are large, generalpurpose benchmarks designed to learn broad, model-agnostic forensic features, while others focus specifically on artistic styles or robustness to real-world degradations and adversarial perturbations.

Table 2 summarizes key properties of representative datasets, including total size, real/fake composition, real-image sources, underlying generators, and primary use cases. Robust AI-artwork detectors are usually trained and evaluated using a combination of resources, including large general datasets for broad coverage, artfocused datasets for style awareness, and robustness-oriented datasets to stress-test models under realistic noise, compression, and attack settings.

Recent research shows a shift from GAN-based image generation to modern difusion and transformer-based systems. Detection methods have expanded from low-level artifact analysis to feature-based and semantically informed deep learning approaches. Artwork-specific detection studies have also explored human-versus-AI classification using classical machine learning and deep learning models, showing that strong performance is possible under controlled style and generator settings [26]. However, current benchmarks still provide limited coverage of promptaligned, art-specific generator shift, especially for newer models such as Stable Difusion 3.5. This gap motivates the present study, which evaluates five deep backbone detectors under a common experimental protocol and tests their ability to generalize from earlier generators to a new prompt-aligned Stable Difusion 3.5 Medium dataset.

Table 2: Summary of public AI-generated image and art detection datasets
<table><tr><td>Dataset</td><td>Total images</td><td>Real/Fake split</td><td>Real sources</td><td>Generators</td><td>Primary use case</td></tr><tr><td>ArtiFact [41]</td><td>~2.5M</td><td>Balanced</td><td>COCO, LSUN, FFHQ, AFHQ</td><td>GANs, diffusion</td><td>Social-media robustness</td></tr><tr><td>GenImage [60]</td><td>~2.7M</td><td>ImageNet</td><td>1000 classes</td><td>Midjourney, SD, GLIDE, etc.</td><td>Forensics, cross-generation</td></tr><tr><td>ArtBench-10 [27]</td><td>60k</td><td>All real</td><td>WikiArt (10 styles)</td><td></td><td>Style-aware benchmarks</td></tr><tr><td>AI-ArtBench [49]</td><td>~185k</td><td>60k real 125k fake</td><td>ArtBench-10 (real)</td><td>Latent Diffusion, SD</td><td>Human vs. AI art styles</td></tr><tr><td>Human-Art [22]</td><td>50k</td><td>Mixed real and stylized</td><td>Natural photos plus WikiArt</td><td>Various</td><td>Detection across styles</td></tr><tr><td>CIFAKE [4]</td><td>120k</td><td>60k real 60k fake</td><td>CIFAR-10</td><td>SD v1.4</td><td>Edge benchmarking</td></tr><tr><td>COCO-Fake [1]</td><td>~1.2M</td><td>Paired fakes</td><td>MS-COCO</td><td>SD v1.4, v2.0</td><td>Caption-image consistency</td></tr><tr><td>RAID [12]</td><td>~72k</td><td>Balanced</td><td>Subset of LAION-400M</td><td>SD variants</td><td>Adversarial robustness</td></tr></table>

## 3 Dataset and Implementation

This section describes the construction of the datasets used for training and out-ofdistribution (OOD) evaluation, and the design and training of the binary detectors that we evaluate as defensive screening components. We also describe our detector architecture, training procedure, threshold selection strategy, and evaluation protocol.

## 3.1 Threat Model and Evaluation Scope

We consider a post-deployment generator-shift scenario in which an image-based detector is used as a defensive screening component to distinguish human-created artwork from AI-generated artwork. The defender may represent a content-moderation platform, media-verification service, digital-art marketplace, or another organization that reviews the claimed origin of visual content. We assume that the detector has been trained using images from known generator families, deployed with a fixed decision threshold, and has no access to labeled samples from generators that emerge after deployment.

The opposing role is a synthetic-media producer who uses a newer, publicly available image generator that was not represented during detector training. The producer is not assumed to have access to the detector’s parameters, gradients, training data, or internal confidence scores, and does not construct adversarial pixel perturbations. Instead, the relevant evasion mechanism is the adoption of a generator whose architecture and learned visual characteristics difer from those represented in the training data. The primary failure of interest is therefore a false negative, in which an AI-generated image is classified as human-created to pass the screening mechanism and evade the detector. False positives are also important because incorrectly labeling human-created artwork as AI-generated can cause attribution errors and reputational harm to artists, while also undermining public trust in detection systems.

This threat model is operationalized by training detectors on artwork generated by Latent Difusion and Stable Difusion 2.1 and evaluating them, without retraining or threshold adjustment, on artwork generated by Stable Difusion 3.5 Medium. The SD3.5m images are produced from prompts derived from held-out human artwork so that semantic content and artistic-style coverage remain comparable while the underlying generator changes. The scope of this study is limited to image-level human-versus-AI artwork detection. It does not directly evaluate metadata manipulation, watermark removal, account compromise, adversarial perturbations or the intent of the content producer. Detector predictions should therefore be interpreted as screening signals for further verification rather than as definitive proof of authenticity or malicious use.

## 3.2 Experimental Setup

Table 3 lists the hardware configuration used for the experiments. All deep learning workloads used PyTorch and torchvision, with CUDA 12.8 as the compute backend [36]. Image generation was performed with the HuggingFace difusers library, and pretrained CLIP and BLIP model weights are loaded via HuggingFace transformers [55].

Table 3: Hardware configuration used for all experiments
<table><tr><td>Component</td><td>Details</td></tr><tr><td>Compute platform</td><td>Google Colab Pro</td></tr><tr><td>GPU</td><td>NVIDIA A100-SXM4-80GB</td></tr><tr><td>CPU</td><td>Intel(R) Xeon(R) CPU @ 2.20GHz</td></tr><tr><td>System RAM</td><td>167 GB (high-memory instance)</td></tr><tr><td>Storage</td><td>Google Drive, Colab Disk</td></tr></table>

All large image directories are archived as zip files on Google Drive and extracted to the Colab instance’s local disk at the start of each session using the zipfile module. This reduced dataset loading time from several hours of cloud-based file access to a single bulk extraction step completed in only a few minutes.

## 3.3 Artwork ID Dataset

AI-ArtBench is used as the primary in-distribution (ID) dataset in this study [49]. It is an art-focused benchmark for detecting and attributing AI-generated artwork, containing approximately 185,000 images across 10 artistic styles: Art Nouveau, Baroque, Expressionism, Impressionism, Post-Impressionism, Realism, Renaissance, Romanticism, Surrealism, and Ukiyo-e. It includes 60,000 human artwork samples and 125,015 AI-generated images produced by Latent Difusion (LDM) and Stable Difusion 2.1 (SD2.1) models. Its human subset is derived from ArtBench-10, a standardized and class-balanced dataset of 60,000 artwork images across the ten styles [27].

AI-ArtBench includes three source categories namely human, LDM, and SD2.1. In this chapter, we consider a binary human-versus-AI setting by merging the two synthetic classes. The final in-distribution splits used in the experiments consist of 105,000 training samples, 15,000 validation samples, and 30,000 test samples. This benchmark provides the core data foundation for training and evaluating the detector in a style-diverse artwork setting.

![](images/2d4a59d4ceee04250f3c8ebd818e853312426d34dbd5fe439b2fcae6715561ef.jpg)  
Figure 1: Samples from the AI-ArtBench dataset: human artwork (top row), LDMgenerated artwork (middle row), and SD2.1-generated artwork (bottom row) across 10 art styles, shown left to right: Art Nouveau, Baroque, Expressionism, Impressionism, Post-Impressionism, Realism, Renaissance, Romanticism, Surrealism, and Ukiyo-e

## 3.4 Stable Difusion OOD Dataset

The main challenge that we address is whether a detector trained on images from LDM and SD2.1 can remain reliable when an adversary switches to a newer and architecturally diferent generator as defined in Section 3.1. To study this, a dataset of 10,000 images was constructed using Stable Difusion 3.5 Medium (SD3.5m) [13].

SD3.5m was selected because it is a newer member of the Stable Difusion family which Stability AI describes as ofering improved prompt adherence, image quality, and practical use on accessible hardware. Unlike the training U-Net generators, SD3.5m is built on the Difusion Transformer (DiT) architecture rather than a convolutional U-Net. DiT supports richer text-to-image conditioning through a multi-encoder text pipeline [37]. These architectural diferences make SD3.5m a challenging and realistic test of cross-generator generalization.

In this study, we use the dataset for out-of-distribution evaluation. All SD3.5m images are generated using prompts derived from human artwork samples through a reverse prompting pipeline. This helps make the OOD images semantically comparable to the human artwork samples and reduces content mismatch. Essentially this prompt-aligned construction means that any decrease in detector performance is more likely to be caused by the change in generator rather than by diferences in image content or style distribution.

The pipeline illustrated in Figure 2 consists of four stages: human artwork selection, reverse prompting via CLIP Interrogator, prompt augmentation and controlled image generation with SD3.5m.

![](images/126574a02e58d8d5f2be24896c07d010b67e60381649aa60093984c89a8b4716.jpg)  
Figure 2: Overview of the prompt-aligned SD3.5m dataset construction pipeline

## 3.4.1 Held-Out Human Reference Set

The source images for the reverse prompting pipeline are a separate set of 10,000 human artwork samples selected from the broader artwork corpus and reserved as a held-out subset. These images were sampled across the 10 art styles used in this study, with 1,000 artwork samples per style, so that the extension would preserve stylistic coverage consistent with the rest of the dataset. The held-out artwork samples were excluded from the in-distribution training, validation, and test splits to prevent data leakage. These images serve as input to the reverse prompting pipeline and as the human-class reference set in the final OOD test set.

## 3.4.2 Image-to-Text via Image Captioning

We used CLIP Interrogator to generate content-aware and style-consistent text prompts from human artwork [38]. This is a reverse prompt engineering tool that combines a visual captioning model with a large database of style, artist, and aesthetic tags ranked by CLIP similarity to the input image [39].

The system was configured with the BLIP-Large captioning model and the CLIP ViT-L/14 vision encoder [25]. The chunk size and flavor intermediate count were both set to 2,048, which is the recommended value for the ViT-L/14 backbone to accommodate its higher-dimensional embedding space. In classic mode, CLIP Interrogator first generates a BLIP caption describing the image content, then appends the highest-scoring style, artist, and flavor tags from its curated databases. This produces human-readable prompts such as

a painting of a family with a bird on their shoulder, a painting by Jacob

Jordaens, baroque, oil on canvas, Flemish baroque, Dutch golden age

Each artwork image was processed individually, producing one prompt per image and saved to a CSV file along with the source filename. The CLIP Interrogator runs on the NVIDIA A100 GPU (see Section 3.2) at approximately 1.5 images per second.

## 3.4.3 Prompt Augmentation

A prompt augmentation step is applied to captions before image generation to ensure the SD3.5m output reflects the target art style. The final prompt for each artwork is made by combining three components namely the target art style, the painting title parsed from the source image filename and the CLIP-based caption from the previous stage. This enriches the caption with explicit style information while preserving the semantic content extracted by CLIP Interrogator. The prompt template used for augmentation is

$$
\left\{ \mathrm { a r t \_ s t y l e } \right\} \mathrm { s t y l e ~ a r t ~ t i t 1 e d ~ } \left\{ \mathrm { p a i n t i n g \_ n a m e } \right\} , \left\{ \mathrm { C L I P \_ c a p t i o n } \right\}
$$

## 3.4.4 Text-to-Image Generation with Stable Difusion

For our OOD dataset, image generation was performed using the HuggingFace diffusers library with the stabilityai/stable-diffusion-3.5-medium checkpoint. Images were generated at a fixed resolution of 768 × 768 pixels using 28 denoising steps and a classifier-free guidance scale of 4.5. This configuration provides good adherence to the prompts without overly constraining the output. The FlowMatchEulerDiscreteScheduler is used as recommended for SD3.x models with bfloat16 precision to balance image quality and memory usage [13]. Memoryeficient attention (xFormers) was enabled to lower memory usage and support faster image generation (see Section 3.2 for environment specification). These configuration parameters are summarized in Table 4.

The randomness is controlled through a combination of a global seed and a per-image seed derived deterministically from the input filename using a SHA-256 hash, ensuring that each human artwork always maps to the same generated image. Output files are named to encode style, subject, and seed, for example

$$
\mathrm { a i \mathrm { - } s d 3 5 m \mathrm { - } \mathrm { < s t y l e > \_ < p a i n t i n g > \_ < s e e d > . \mathrm { j p g } } }
$$

An audit log is saved for every image and includes the original image filename, injected style, parsed painting title, final prompt, seed, configuration parameters, output path, runtime, timestamp and status message. GPU telemetry (utilization, memory, power, temperature) was sampled at 500 ms intervals during each generation call using NVIDIA NVML for resource monitoring.

Table 4: Stable Difusion 3.5 Medium generation configuration used to construct the OOD test set
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Model checkpoint</td><td>stabilityai/stable-diffusion-3.5-medium</td></tr><tr><td>Output resolution</td><td>768 × 768 pixels</td></tr><tr><td>Number of denoising steps</td><td>28</td></tr><tr><td>Classifier-free guidance scale</td><td>4.5</td></tr><tr><td>Scheduler</td><td>FlowMatchEulerDiscreteScheduler</td></tr><tr><td>Precision Seeding strategy</td><td>bfloat16 SHA-256 hash of source filename</td></tr><tr><td>Average generation time</td><td>3.2 seconds per image</td></tr><tr><td>Average GPU utilization</td><td>86%</td></tr><tr><td>Average GPU power</td><td>333 W</td></tr><tr><td>Average GPU energy per image</td><td></td></tr><tr><td></td><td>0.29 Wh</td></tr></table>

The complete SD3.5m AI dataset contains 10,000 images consisting of 1,000 prompt-aligned SD3.5m images per style. Sample generated images alongside their source human artwork samples are shown in Figure 3. This dataset can test detector behavior under a realistic change in the generative model, and the pipeline is scalable for future experiments.

![](images/7e134eb025551458f8f17b12bb33b81c4299708df6c7b980a17ca9703022cbf5.jpg)  
Figure 3: Samples of human artwork (top row) and prompt-aligned SD3.5mgenerated artwork samples (bottom row) across 10 art styles, shown left to right: Art Nouveau, Baroque, Expressionism, Impressionism, Post-Impressionism, Realism, Renaissance, Romanticism, Surrealism, and Ukiyo-e

## 3.5 Dataset Quality Evaluation

The quality of the SD3.5m dataset extension was evaluated using a set of complementary metrics covering prompt alignment, distributional similarity, duplicate detection and visual diversity. A subset of the data was used for the evaluation which consists of 5,000 generated images (500 per style) and 5,000 corresponding human artwork samples. Four metrics are computed: CLIPScore [19], Fr´echet Inception Distance (FID) [20], Kernel Inception Distance (KID) [3] and Learned Perceptual Image Patch Similarity (LPIPS) perceptual diversity [57].

CLIPScore measures how well each generated image matches its generation prompt. All images and their corresponding prompts were passed through the CLIP ViT-L/14 model and CLIPScore was computed as the cosine similarity between the projected image and text embeddings. A shufled-prompt baseline was also computed by randomly reassigning prompts to images (five repetitions), providing a lower bound that reflects what CLIP similarity looks like when image-prompt pairing is broken.

FID and KID measure the statistical distance between the distribution of generated images and the distribution of real human artwork samples for the same style, both computed in Inception feature space at $2 5 6 \times 2 5 6$ resolution. A human-versushuman (HvH) baseline was also computed by repeatedly splitting the human set into two random halves and measuring FID/KID between those halves, providing a lower bound that reflects the natural within-style variation among human artwork samples.

LPIPS diversity measures the average perceptual distance between randomly sampled pairs of generated images within the same style, using an AlexNet backbone. Higher values indicate that the generated images are visually diverse rather than collapsing into a small set of visual patterns. We compute the LPIPS diversity score for the generated dataset as the mean perceptual distance across 1,000 randomly sampled intra-style image pairs.

Duplicate analysis uses perceptual hashing (pHash) to detect duplicate candidates with identical pHash values and near-duplicates. For this metric, we consider Hamming distance less than or equal to 4.

Table 5: Dataset quality metrics for the SD3.5m OOD dataset (5,000 images, 500 per style; CLIP and $\mathrm { C L I P _ { s h u f } }$ shown in units of $\times 1 0 ^ { - 2 }$ ; KID and HvH KID shown in units of ×10<sup>−4</sup>)
<table><tr><td>Style</td><td>CLIP</td><td> $\mathbf { C L I P _ { s h u f } }$ </td><td>FID</td><td>HvH FID</td><td>KID</td><td>HvH KID</td><td>LPIPS</td></tr><tr><td>Art Nouveau</td><td>28.52</td><td>11.70</td><td>119.67</td><td>126.21</td><td>214.0</td><td>-0.46</td><td>0.712</td></tr><tr><td>Baroque</td><td>28.14</td><td>12.87</td><td>110.10</td><td>116.30</td><td>226.0</td><td>0.70</td><td>0.660</td></tr><tr><td>Expressionism</td><td>27.32</td><td>9.33</td><td>130.60</td><td>137.77</td><td>215.0</td><td>1.92</td><td>0.740</td></tr><tr><td>Impressionism</td><td>27.91</td><td>10.19</td><td>120.43</td><td>126.47</td><td>256.0</td><td>-3.72</td><td>0.712</td></tr><tr><td>Post-Impressionism</td><td>27.63</td><td>9.73</td><td>124.95</td><td>124.06</td><td>266.0</td><td>-0.66</td><td>0.725</td></tr><tr><td>Realism</td><td>28.81</td><td>10.31</td><td>118.20</td><td>130.44</td><td>219.0</td><td>-3.98</td><td>0.706</td></tr><tr><td>Renaissance</td><td>26.37</td><td>12.98</td><td>104.76</td><td>109.60</td><td>213.0</td><td>-2.63</td><td>0.673</td></tr><tr><td>Romanticism</td><td>28.49</td><td>10.79</td><td>103.66</td><td>118.24</td><td>175.0</td><td>0.17</td><td>0.690</td></tr><tr><td>Surrealism</td><td>28.82</td><td>11.86</td><td>128.19</td><td>148.35</td><td>187.0</td><td>-2.16</td><td>0.736</td></tr><tr><td>Ukiyo-e</td><td>30.93</td><td>22.55</td><td>139.37</td><td>83.14</td><td>674.0</td><td>-3.67</td><td>0.681</td></tr><tr><td>Overall</td><td>28.29</td><td>7.53</td><td>43.84</td><td>24.16</td><td>180.0</td><td>0.38</td><td>0.724</td></tr></table>

Table 5 reports the per-style and overall results. The overall FID, KID and shufled-prompt CLIP baseline are computed on the pooled multi-style dataset and not as averages of the per-style rows. For the pooled dataset spanning all ten styles, the reported overall CLIPScore is 28.29, compared to a shufled-prompt baseline of 7.53, yielding a margin of 20.76 in the reported table units. This large gap suggests that the generated images remain meaningfully aligned with their source prompts.

The overall FID is 43.84, compared to a human-versus-human baseline FID of 24.16, indicating that the SD3.5m images remain distributionally distinct from the human reference set. The reported overall KID is 180.0, corresponding to a raw KID value of 0.0180, whereas the corresponding human-versus-human KID baseline is 0.38 in the table units and therefore remains near zero. The trend is consistent with the FID scores and supports distributional separation when measured under the second metric. The average diversity by the LPIPS criterion is 0.72 and the style-wise LPIPS scores are all in the mid to high ranges, suggesting a good amount of visual diversity across the images. There were no duplicate images found by the pHash metric and no near duplicate pairs within each style.

Ukiyo-e has the highest raw CLIPScore (30.93), but also a high shufled baseline (22.55). This suggests the score partly reflects shared style features, not only exact image–prompt alignment. Ukiyo-e also has the largest FID score (139.37) and KID (674.0), making it a clear outlier. By comparison, Romanticism (103.66) and Renaissance (104.76) are associated with the smallest FID values among the ten styles.

In general, we see that the SD3.5m dataset retains the style coverage and prompt relevance while remaining diferent from the human reference dataset. This makes it suitable for style-aware out-of-distribution evaluation under generator shift.

## 3.6 Detector Implementation

Image detection is organized as a modular framework for evaluating multiple deep learning models on the task of binary human-versus-AI artwork detection. We consider several detector families, including ResNet, EficientNet, ConvNeXt, and CLIP-ViT, all studied under a shared in-distribution and out-of-distribution evaluation protocol.

## 3.6.1 Architecture Overview

All detectors follow a frozen-backbone, linear-probe design. A pretrained deep network serves as a fixed feature extractor, the final classification layer is removed and the penultimate feature vector is extracted for every input image. A lightweight linear classification head is then trained on top of these frozen features to perform binary human-versus-AI classification. This approach has three main benefits. First, it reduces the risk of catastrophically fine-tuning a large pretrained model on a relatively small dataset. Second, feature extraction can be precomputed once for the entire dataset, making the head-training loop extremely fast. Third, the frozen representations provide a controlled comparison across backbone families without confounding efects from diferences in fine-tuning schedules. This also models a realistic deployment scenario where the OOD performance reflects cross-generator transfer without adaptation to SD3.5m.

The following five models are evaluated. Note that these models include both convolutional and transformer-based architectures.

1. ResNet-18 [17] is a lightweight 18-layer residual network, providing a lowcapacity CNN baseline. Feature dimension: 512; pretrained on ImageNet-1K.

2. ResNet-50 [17] is the 50-layer variant residual network, providing a mediumcapacity CNN baseline. Feature dimension: 2,048; pretrained on ImageNet-1K.

3. EficientNet-B0 [51] is a compound-scaled CNN, optimized for eficiency. Feature dimension: 1,280; pretrained on ImageNet-1K.

4. ConvNeXt-Base [29] is a modern CNN architecture incorporating design principles from Vision Transformers, serving as a bridge between pure CNN and transformer families. Feature dimension: 1,024; pretrained on ImageNet-1K.

5. CLIP ViT-L/14 [39] is a large Vision Transformer trained on 400M imagetext pairs with a contrastive language-image objective, and a feature dimension of 768. This backbone is particularly relevant because its representations are sensitive to semantic and stylistic content rather than low-level texture statistics.

## 3.6.2 Dataset Splits, Pruning, and Preprocessing

The AI-ArtBench dataset [49] is built on top of ArtBench-10 [27], which contains 6,000 images per art style (60,000 human artwork samples in total). AI-ArtBench splits the human portion into a train split of 5,000 images per style (50,000 total) and a test split of 1,000 images per style (10,000 total), adding AIgenerated images from LDM and SD2.1 to each split. Table 6 reproduces the full dataset composition from the original paper [49]. The 30,000-image test split is used directly as the in-distribution (ID) test set in this study and is not used during training or validation.

Table 6: Original AI-ArtBench dataset composition
<table><tr><td>Source</td><td>Styles</td><td>Train</td><td>Test</td></tr><tr><td>Latent Diffusion (LDM)</td><td>10</td><td>52,092</td><td>10,000</td></tr><tr><td>Stable Diffusion 2.1 (SD2.1)</td><td>10</td><td>52,923</td><td>10,000</td></tr><tr><td>Human (WikiArt / ArtBench-10)</td><td>10</td><td>50,000</td><td>10,000</td></tr><tr><td>Total</td><td>30</td><td>155,015</td><td>30,000</td></tr></table>

The prompt-aligned SD3.5m OOD evaluation requires a held-out set of human artwork samples that share no overlap with training data. These images serve as both the source for reverse prompting (Section 3.4.2) and the human-class reference in the final OOD test set. To obtain this set, 1,000 human images per style (10,000 in total) were extracted from the 50,000-image human training pool. The remaining human training pool contains exactly 4,000 images per style (40,000 total).

After OOD extraction, the training pool consists of 40,000 human images, 52,092 LDM images, and 52,923 SD2.1 images. The AI generator counts are substantially larger than the human count and vary unevenly across styles: LDM per-style counts range from 4,844 to 5,504 and SD2.1 per-style counts range from 5,052 to 5,455. Table 7 shows the full per-style breakdown before and after pruning. Without pruning, the combined training set would contain 105,015 AI images against 40,000 human images, a 2.6:1 binary-class imbalance that would bias detectors toward the majority AI class and conflate style-specific detection dificulty with class frequency efects.

To remove the source-level and style-level frequency diferences, exactly 4,000 images per style were retained for each AI source class by random sampling with a fixed seed. This removed 12,092 excess LDM images and 12,923 excess SD2.1 images, totaling 25,015 pruned images. The resulting corpus contains 3 × 4,000 × 10 = 120,000 images with equal representation across all style-source combinations.

Table 7: Per-style image counts in the training pool before and after pruning
<table><tr><td rowspan="2">Style</td><td colspan="4">Before pruning</td><td colspan="4">After pruning</td></tr><tr><td>Human</td><td>LDM</td><td>SD2.1</td><td>Total</td><td>Human</td><td>LDM</td><td>SD2.1</td><td>Total</td></tr><tr><td>Art Nouveau</td><td>4,000</td><td>4,992</td><td>5,384</td><td>14,376</td><td>4,000</td><td>4,000</td><td>4,000</td><td>12,000</td></tr><tr><td>Baroque</td><td>4,000</td><td>4,960</td><td>5,052</td><td>14,012</td><td>4,000</td><td>4,000</td><td>4,000</td><td>12,000</td></tr><tr><td>Expressionism</td><td>4,000</td><td>5,116</td><td>5,304</td><td>14,420</td><td>4,000</td><td>4,000</td><td>4,000</td><td>12,000</td></tr><tr><td>Impressionism</td><td>4,000</td><td>5,208</td><td>5,388</td><td>14,596</td><td>4,000</td><td>4,000</td><td>4,000</td><td>12,000</td></tr><tr><td>Post-Impressionism</td><td>4,000</td><td>4,844</td><td>5,360</td><td>14,204</td><td>4,000</td><td>4,000</td><td>4,000</td><td>12,000</td></tr><tr><td>Realism</td><td>4,000</td><td>5,232</td><td>5,248</td><td>14,480</td><td>4,000</td><td>4,000</td><td>4,000</td><td>12,000</td></tr><tr><td>Renaissance</td><td>4,000</td><td>5,416</td><td>5,060</td><td>14,476</td><td>4,000</td><td>4,000</td><td>4,000</td><td>12,000</td></tr><tr><td>Romanticism</td><td>4,000</td><td>5,436</td><td>5,455</td><td>14,891</td><td>4,000</td><td>4,000</td><td>4,000</td><td>12,000</td></tr><tr><td>Surrealism</td><td>4,000</td><td>5,384</td><td>5,364</td><td>14,748</td><td>4,000</td><td>4,000</td><td>4,000</td><td>12,000</td></tr><tr><td>Ukiyo-e</td><td>4,000</td><td>5,504</td><td>5,308</td><td>14,812</td><td>4,000</td><td>4,000</td><td>4,000</td><td>12,000</td></tr><tr><td>Total</td><td>40,000</td><td>52,092</td><td>52,923</td><td>145,015</td><td>40,000</td><td>40,000</td><td>40,000</td><td>120,000</td></tr></table>

The before-pruning column in Table 7 makes the imbalance concrete: the combined pre-pruning pool contains 105,015 AI images versus 40,000 human images. Within the AI classes, per-style variation is as high as 660 images for LDM (4,844 in Post-Impressionism versus 5,504 in Ukiyo-e) and 403 images for SD2.1 (5,052 in Baroque versus 5,455 in Romanticism). Pruning equalizes the three source categories and removes within-source style skew, ensuring that no style-source combination contributes disproportionately to detector training.

The 120,000-image balanced corpus was divided into a training split of 105,000 images (3,500 per style per source class) and a validation split of 15,000 images (500 per style per source class), maintaining the source balance within each split. Images were assigned to each split using a fixed random seed. The validation split is used exclusively for epoch selection and threshold optimization and is not used for final evaluation. The complete dataset composition across all splits is summarized in Tables 8 and 9. Overall, the final in-distribution dataset was split into 70% training, 10% validation, and 20% ID test.

All preprocessing performed is deterministic and no data augmentation is applied at any stage. Each image is loaded using the Pillow Python imaging library (PIL), converted to RGB, and passed through the same deterministic backbonespecific preprocessing pipeline before feature extraction. For the four CNN backbones (ResNet-18, ResNet-50, EficientNet-B0 and ConvNeXt-Base), images are resized to $2 2 4 \times 2 2 4$ pixels using bicubic interpolation, converted to tensors, and normalized using the ImageNet training set mean $( \mu = [ 0 . 4 8 5 , 0 . 4 5 6 , 0 . 4 0 6 ] )$ and standard deviation $( \sigma = [ 0 . 2 2 9 , 0 . 2 2 4 , 0 . 2 2 5 ] )$ ) [46]. For CLIP ViT-L/14, the HuggingFace processor associated with

Table 8: In-distribution (ID) dataset composition across all splits
<table><tr><td>Split</td><td>Human</td><td>LDM</td><td>SD2.1</td><td>Total</td></tr><tr><td>Training</td><td>35,000</td><td>35,000</td><td>35,000</td><td>105,000</td></tr><tr><td>Validation</td><td>5,000</td><td>5,000</td><td>5,000</td><td>15,000</td></tr><tr><td>ID Test</td><td>10,000</td><td>10,000</td><td>10,000</td><td>30,000</td></tr><tr><td>Total</td><td>50,000</td><td>50,000</td><td>50,000</td><td>150,000</td></tr></table>

Table 9: Out-of-distribution (OOD) evaluation set composition
<table><tr><td>Source Images</td></tr><tr><td>Human (held-out) 10,000</td></tr><tr><td>AI (SD3.5m, generated) 10,000</td></tr><tr><td>Total 20,000</td></tr></table>

$$
\mathsf { o p e n a i / c l i p - v i t - l a r g e \mathrm { - } p a t c h 1 4 }
$$

is used to prepare images for feature extraction, applying the checkpoint’s default resize/crop and normalization pipeline. The same backbone-specific deterministic preprocessing is used for all splits including training, validation, ID test and OOD test.

## 3.6.3 Feature Extraction

Each backbone is used as a frozen feature extractor: the final classification layer is removed so that a forward pass returns a fixed-length feature vector for each image. For CLIP ${ \mathrm { V i T - L } } / 1 4$ , the projected image embedding is used, consistent with the space in which CLIP was pretrained. Table 10 lists the pretrained weights and output feature dimension for each backbone. CNN backbones are loaded using torchvision and CLIP $\mathrm { V i T - L } / 1 4$ is loaded using HuggingFace Transformers.

Features are extracted once per backbone per split and saved to disk. All subsequent head training reads directly from these cached feature files, reducing the per-epoch training cost to a single linear layer pass and allowing rapid iteration across training epochs.

## 3.6.4 Classification Head Training

A single linear layer maps the frozen backbone feature vector $\mathbf { x } \in \mathbb { R } ^ { D }$ to a scalar logit according to

$$
z = \mathbf { w } ^ { \top } \mathbf { x } + b , { \mathrm { ~ w h e r e ~ } } \mathbf { w } \in \mathbb { R } ^ { D } { \mathrm { ~ a n d ~ } } b \in \mathbb { R }\tag{1}
$$

Table 10: Pretrained weights and output feature dimension for each backbone
<table><tr><td>Model</td><td>Pretrained weights</td><td>Feature dimension</td></tr><tr><td>ResNet-18</td><td>ImageNet-1K (V1)</td><td>512</td></tr><tr><td>ResNet-50</td><td>ImageNet-1K (V2)</td><td>2,048</td></tr><tr><td>EfficientNet-B0</td><td>ImageNet-1K (V1)</td><td>1,280</td></tr><tr><td>ConvNeXt-Base</td><td>ImageNet-1K (V1)</td><td>1,024</td></tr><tr><td>CLIP ViT-L/14</td><td>openai/clip-vit-large-patch14</td><td>768</td></tr></table>

At inference, the logit is converted to an AI probability score via the sigmoid function

$$
p _ { \mathrm { A I } } = \sigma ( z ) = \frac { 1 } { 1 + e ^ { - z } } \in ( 0 , 1 )
$$

The head contains no hidden layers and no dropout. This linear probe design is intentional because the backbone is frozen, the trainable component is restricted to a lightweight classifier operating on a high-dimensional pretrained feature space. This keeps the training setup simple, reduces the risk of overfitting in the head, and makes comparisons across backbone families easier to interpret, since most of the discriminative capacity comes from the pretrained representation rather than from a more expressive task-specific head.

The head is optimized using weighted binary cross-entropy with logits. For a mini-batch of � samples with logits $\left\{ z _ { i } \right\}$ and binary labels $\{ y _ { i } \} \in \{ 0 , 1 \}$ , the loss is

$$
\mathcal { L } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } w _ { y _ { i } } \Big [ y _ { i } \log \sigma ( z _ { i } ) + ( 1 - y _ { i } ) \log ( 1 - \sigma ( z _ { i } ) ) \Big ]\tag{2}
$$

where $w _ { 0 }$ and $w _ { 1 }$ are per-class weights computed from the training split as

$$
w _ { c } = \frac { N } { 2 \cdot N _ { c } } , \ \mathrm { f o r } \ c \in \{ 0 , 1 \}
$$

where $N _ { c }$ denotes the number of training samples in class �. This inverse-frequency weighting increases the contribution of the minority class and reduces bias toward the majority class. In the completed binary training split, the class counts are 35,000 human and 70,000 AI samples, which correspond to weights of 1.5 for the human class and 0.75 for the AI class. The loss is computed directly from logits using F.binary cross entropy with logits, which combines the sigmoid and cross-entropy terms in a single numerically stable operation.

AdamW [30] is used with a fixed learning rate of $1 \times 1 0 ^ { - 3 }$ and weight decay of 1× $1 0 ^ { - 4 }$ . AdamW is preferred over standard Adam because it decouples weight decay from the gradient update, yielding more consistent regularization in the presence of adaptive learning rates. In this setting, weight decay acts as L2-style regularization on the head weights and helps limit overfitting in the linear classifier.

Model training runs for a maximum of ten epochs. At the end of each epoch, the head is evaluated on the validation split using a fixed threshold of 0.5. The checkpoint with the highest validation balanced accuracy is retained. Balanced accuracy is computed as

$$
\mathrm { B a l a n c e d ~ A c c u r a c y } = { \frac { 1 } { 2 } } \left( { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } } + { \frac { \mathrm { T N } } { \mathrm { T N } + \mathrm { F P } } } \right)
$$

and is used as the model-selection criterion rather than standard accuracy because it gives equal weight to the true positive rate and true negative rate. This is more informative than raw accuracy under class imbalance and better reflects the need to control both missed AI images and misclassified human artwork samples. The training configuration also includes patience-based early stopping with a patience of three epochs, although in runs that continued to improve, training proceeded to the full ten-epoch budget. After the best checkpoint is selected, a separate threshold sweep is performed on the validation predictions. These configuration parameters are summarized in Table 11.

Table 11: Shared linear head training configuration
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Maximum epochs</td><td>10</td></tr><tr><td>Loss function</td><td>Weighted binary cross-entropy with logits: equation (2)</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W }$ </td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>Epoch selection criterion Best validation balanced accuracy</td></tr><tr><td>Image resolution</td><td> $2 2 4 \times 2 2 4$ </td></tr><tr><td>Interpolation</td><td>Bicubic</td></tr><tr><td>Normalization</td><td>ImageNet mean and standard deviation</td></tr><tr><td>Backbone weights</td><td>Frozen (no fine-tuning)</td></tr><tr><td>Classification head</td><td>Single linear layer, logit output: equation (1)</td></tr><tr><td>Task</td><td>Binary: human  $( y = 0 )$  versus AI (y = 1)</td></tr></table>

## 3.6.5 Threshold Selection

The sigmoid output $p _ { \mathrm { A I } } \in [ 0 , 1 ]$ must be mapped to a binary prediction using a threshold. While the standard choice of a threshold is 0.5, such a threshold would be suboptimal when the score distribution is not centered around 0.5 or when the relative impact of false positives and false negatives difers. To select an operating threshold for each model, a sweep over 99 linearly spaced candidate values from 0.01 to 0.99 was performed on the validation set. For each candidate threshold, binary predictions were derived and validation balanced accuracy was computed, with F1 score used as a tie-breaker. The threshold that maximized validation balanced accuracy was selected and locked for all subsequent evaluations on the ID test and OOD sets. This threshold is not adjusted after selection, preventing threshold overfitting and ensuring that the reported test and OOD metrics reflect realistic deployment conditions.

Selecting thresholds using validation balanced accuracy provides a consistent basis for comparing the models under the assumption that AI-image recall and humanimage specificity are equally important. However, the most appropriate operating threshold in practice may depend on the defensive context and the relative costs of false positives and false negatives. For example, a news-verification or high-risk fraud-screening workflow may prioritize recall and tolerate a higher false-positive rate so that fewer synthetic images escape review. In contrast, an art-attribution or authenticity-assessment system may require a very low false-positive rate to reduce the risk of incorrectly labeling human-created artwork as synthetic. The thresholds used in this study are therefore intended for controlled cross-model and crossgenerator comparison rather than as universally optimal deployment thresholds.

## 3.6.6 Evaluation Protocol

Detectors were evaluated on three disjoint sets: the validation set (used for epoch selection and threshold selection), the ID test set (drawn from the same generator distribution as training), and the OOD evaluation set (unseen SD3.5m dataset). For each model on each split, the following metrics were computed at the validationselected threshold, with AI treated as the positive class: balanced accuracy, precision, recall, F1 score, Matthews correlation coeficient (MCC) [2, 31], false positive rate (FPR), false negative rate (FNR), ROC-AUC, and PR-AUC. Here, recall corresponds to the detector’s coverage of AI-generated images, FNR corresponds to the AI-image miss rate or evasion-like failure rate, and FPR corresponds to the false-alarm rate on human artwork. Style-wise breakdowns were also computed to identify which artistic styles contributed most strongly to model failures. In addition, source-wise analyses were performed for the in-distribution setting.

The main result of interest is detector performance on the OOD set, especially the false negative rate on SD3.5m images. An OOD AI image is classified as AI if $p _ { \mathrm { A I } } \geq \tau _ { \mathrm { v a l } }$ , where $\mathcal { T } _ { \mathrm { v a l } }$ is the threshold selected on the validation set. The drop in balanced accuracy between the ID test set and OOD set is used to calculate how much each detector degrades under generator shift from the training generators (LDM and SD2.1) to SD3.5m. All experiments were run on the hardware described in Section 3.2, and the pipeline was designed to support reproducible evaluation through fixed random seeds, frozen backbone weights, deterministic preprocessing, and cached feature bundles.

## 4 Experimental Results and Analysis

This section reports the experimental outcomes for AI-art detection and provides analysis of the results. Section 4.1 describes training dynamics and the selected classification thresholds. Section 4.2 reports in-distribution performance on the validation and ID test splits. Then in Section 4.3, we present cross-generator OOD evaluation on the prompt-aligned SD3.5m test set, including the ID-to-OOD performance gap. Section 4.4 provides a per-style breakdown of OOD performance,

Section 4.5 reports source-wise accuracy on the ID test set, and Section 4.6 analyzes the results using Grad-CAM spatial attribution maps.

Throughout our experiments, detector models are identified by their backbone name: ResNet-18, ResNet-50, EficientNet-B0, ConvNeXt-Base, and CLIP ViT-$\mathrm { L } / 1 4 .$ . The primary evaluation metric is balanced accuracy, which weights the true positive rate and true negative rate equally. When a single ranking is needed, balanced accuracy is used as the primary criterion, with F1 score, recall, and ROC-AUC provided for various experiments as secondary criteria.

## 4.1 Model Training and Threshold Selection

The detectors were trained for ten epochs using the frozen-backbone, linear-head setup described in Section 3.6.4. Table 12 reports the validation F1 score and validation balanced accuracy at the end of each epoch for every model. The best-epoch checkpoint for each model was selected based on the highest validation balanced accuracy achieved across all ten epochs.

Table 12: Validation F1 score and balanced accuracy by epoch
<table><tr><td rowspan="2">Epoch</td><td colspan="2">ResNet-18</td><td colspan="2">ResNet-50</td><td colspan="2">EffNet-B0</td><td colspan="2">ConvNeXt-Base</td><td colspan="2">CLIP ViT-L/14</td></tr><tr><td>F1</td><td>BalAcc</td><td>F1</td><td>BalAcc</td><td>F1</td><td>BalAcc</td><td>F1</td><td>BalAcc</td><td>F1</td><td>BalAcc</td></tr><tr><td>1</td><td>0.871</td><td>0.835</td><td>0.924</td><td>0.897</td><td>0.927</td><td>0.903</td><td>0.929</td><td>0.904</td><td>0.979</td><td>0.973</td></tr><tr><td>2</td><td>0.907</td><td>0.874</td><td>0.948</td><td>0.927</td><td>0.945</td><td>0.923</td><td>0.956</td><td>0.935</td><td>0.991</td><td>0.988</td></tr><tr><td>3</td><td>0.922</td><td>0.892</td><td>0.957</td><td>0.940</td><td>0.951</td><td>0.932</td><td>0.964</td><td>0.948</td><td>0.994</td><td>0.992</td></tr><tr><td>4</td><td>0.927</td><td>0.902</td><td>0.962</td><td>0.948</td><td>0.956</td><td>0.938</td><td>0.968</td><td>0.955</td><td>0.996</td><td>0.995</td></tr><tr><td>5</td><td>0.931</td><td>0.908</td><td>0.966</td><td>0.954</td><td>0.958</td><td>0.943</td><td>0.972</td><td>0.960</td><td>0.997</td><td>0.996</td></tr><tr><td>6</td><td>0.936</td><td>0.913</td><td>0.970</td><td>0.958</td><td>0.961</td><td>0.946</td><td>0.975</td><td>0.963</td><td>0.997</td><td>0.997</td></tr><tr><td>7</td><td>0.939</td><td>0.917</td><td>0.972</td><td>0.961</td><td>0.961</td><td>0.948</td><td>0.977</td><td>0.967</td><td>0.998</td><td>0.997</td></tr><tr><td>8</td><td>0.940</td><td>0.920</td><td>0.973</td><td>0.964</td><td>0.962</td><td>0.950</td><td>0.978</td><td>0.969</td><td>0.998</td><td>0.997</td></tr><tr><td>9</td><td>0.942</td><td>0.922</td><td>0.974</td><td>0.965</td><td>0.964</td><td>0.952</td><td>0.980</td><td>0.972</td><td>0.998</td><td>0.998</td></tr><tr><td>10</td><td>0.945</td><td>0.924</td><td>0.976</td><td>0.966</td><td>0.966</td><td>0.953</td><td>0.981</td><td>0.973</td><td>0.998</td><td>0.998</td></tr></table>

CLIP ViT-L/14 converges faster than all CNN-based models by epoch 1 it already achieves a validation F1 of 0.979 and a balanced accuracy of 0.973, whereas ResNet-18 does not reach these values by epoch 10. This suggests that the semantic representations learned by CLIP through contrastive language-image pretraining transfer more eficiently to the binary AI-art detection task than convolutional features pretrained on image classification alone. Among the CNN backbones, ConvNeXt-Base converges faster and to a higher plateau than the other three, reflecting its larger capacity and modern architectural design. ResNet-18 shows the slowest convergence among the detectors. Figure 4 shows the validation balanced accuracy trajectory for the detector models across ten epochs.

Following training, the optimal decision threshold was selected for each model as described in Section 3.6.5. Table 13 reports the selected threshold for each model. These values range from 0.50 for CLIP ViT-L/14 to 0.58 for EficientNet-B0. The CLIP threshold of 0.50 indicates the default midpoint is already optimal. In contrast, EficientNet-B0’s higher threshold of 0.58 suggests its raw probabilities are slightly compressed near the center, requiring a shift toward the AI side to maximize the balanced accuracy.

![](images/6ea5cc8c6f24a5bc50a501dcee215f0479a9836114dca6edc6f2dafa99b26e6f.jpg)  
Figure 4: Validation balanced accuracy as a function of training epoch

Table 13: Decision threshold selected and corresponding validation and ID test balanced accuracy at the threshold
<table><tr><td rowspan="2">Model</td><td rowspan="2">Threshold</td><td colspan="2">Balanced accuracy</td></tr><tr><td>Validation</td><td>ID Test</td></tr><tr><td>CLIP ViT-L/14</td><td>0.50</td><td>0.9980</td><td>0.9969</td></tr><tr><td>ConvNeXt-Base</td><td>0.53</td><td>0.9745</td><td>0.9727</td></tr><tr><td>ResNet-50</td><td>0.55</td><td>0.9672</td><td>0.9637</td></tr><tr><td>EfficientNet-B0</td><td>0.58</td><td>0.9538</td><td>0.9524</td></tr><tr><td>ResNet-18</td><td>0.53</td><td>0.9246</td><td>0.9260</td></tr></table>

Figure 5 illustrates the threshold sweep for ConvNeXt-Base, showing how validation balanced accuracy varies across the 99 candidate threshold values and confirming that the selected value of 0.53 is indeed at the peak.

## 4.2 In-Distribution Performance

Table 14 reports evaluation results for the detectors on the validation and ID test splits, using the thresholds selected from the validation set. In-distribution performance measures how well each detector identifies images from the same generator families used during training.

All models perform well on the in-distribution test set, the ranking generally follows the strength and capacity of the various models. CLIP ViT-L/14 is the best in-distribution detector, as it achieves an ID test balanced accuracy of 0.9969, an F1 score of 0.9977, and a ROC-AUC of 0.9999. Its false positive rate and false negative rate are both 0.003, which implies that CLIP ViT-L/14 makes fewer than 100 errors across the 30,000 ID test images. ConvNeXt-Base is the second best model with an ID test balanced accuracy of 0.9727. The other CNN models perform close to each other: ResNet-50 reaches 0.9637, EficientNet-B0 achieves 0.9524, and ResNet-18 is at 0.9260. The validation and ID test results are also very consistent for all models, indicating that the threshold selection did not overfit to the validation split.

![](images/30062cf4e059c57fd60457eb0d9f1ebb43f3b38c68ef2647289c8ec3f1f7c30f.jpg)  
Figure 5: Validation threshold sweep for ConvNeXt-Base

Table 14: In-distribution performance on the validation split (Val) and ID test split (IDT); here Bal = balanced accuracy, Prec = precision, Rec = recall, F1 = F1 score, FNR = false negative rate, FPR = false positive rate, MCC = Matthews correlation coeficient, AUC = ROC-AUC, PR = PR-AUC, and all metrics computed at the validation-selected threshold as per Table 13
<table><tr><td>Model</td><td>Split</td><td>Bal</td><td>Prec</td><td>Rec</td><td>F1</td><td>FNR</td><td>FPR</td><td>MCC</td><td>AUC</td><td>PR</td></tr><tr><td rowspan="2">CLIP ViT-L/14</td><td>Val</td><td>0.9980</td><td>0.9992</td><td>0.9975</td><td>0.9983</td><td>0.0025</td><td>0.0016</td><td>0.9951</td><td>1.0000</td><td>1.0000</td></tr><tr><td>IDT</td><td>0.9969</td><td>0.9984</td><td>0.9971</td><td>0.9977</td><td>0.0030</td><td>0.0032</td><td>0.9932</td><td>0.9999</td><td>1.0000</td></tr><tr><td rowspan="2">ConvNeXt-Base</td><td>Val</td><td>0.9745</td><td>0.9870</td><td>0.9745</td><td>0.9807</td><td>0.0255</td><td>0.0256</td><td>0.9431</td><td>0.9962</td><td>0.9980</td></tr><tr><td>IDT</td><td>0.9727</td><td>0.9863</td><td>0.9726</td><td>0.9794</td><td>0.0275</td><td>0.0271</td><td>0.9391</td><td>0.9962</td><td>0.9980</td></tr><tr><td rowspan="2">ResNet-50</td><td>Val</td><td>0.9672</td><td>0.9847</td><td>0.9645</td><td>0.9745</td><td>0.0355</td><td>0.0300</td><td>0.9254</td><td>0.9940</td><td>0.9968</td></tr><tr><td>IDT</td><td>0.9637</td><td>0.9821</td><td>0.9624</td><td>0.9721</td><td>0.0377</td><td>0.0350</td><td>0.9185</td><td>0.9939</td><td>0.9968</td></tr><tr><td rowspan="2">EfficientNet-B0</td><td>Val</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>IDT</td><td>0.9538 0.9524</td><td>0.9795 0.9776</td><td>0.9472 0.9482</td><td>0.9631 0.9627</td><td>0.0528 0.0518</td><td>0.0396 0.0435</td><td>0.8939 0.8922</td><td>0.9901 0.9900</td><td>0.9946 0.9949</td></tr><tr><td rowspan="2">ResNet-18</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Val IDT</td><td>0.9246 0.9260</td><td>0.9595 0.9603</td><td>0.9274 0.9289</td><td>0.9432 0.9443</td><td>0.0726 0.0712</td><td>0.0782 0.0768</td><td>0.8362 0.8393</td><td>0.9767 0.9777</td><td>0.9868 0.9874</td></tr></table>

Figure 6 shows the ID test confusion matrices for CLIP ViT-L/14 and ConvNeXt-Base. Figure 7 overlays the ID test ROC curves for each of the five models.

Figure 8 shows the ID test probability histograms for CLIP ViT-L/14 and ConvNeXt-Base. For both models, human images are mostly grouped near $p _ { \mathrm { A I } } = 0 .$ while AI images are mostly grouped near $p _ { \mathrm { A I } } = 1$ , and there is little overlap around the threshold selected from the validation set. This clear separation indicates that both models can distinguish human and AI images very well on the in-distribution test set.

Table 15 reports per-style ID test balanced accuracy for CLIP ViT-L/14 and ConvNeXt-Base. All per-style accuracies are above 0.94 for ConvNeXt-Base, and above 0.994 for CLIP ViT-L/14. Surrealism is the most challenging style for both models. Ukiyo-e is the easiest style for ConvNeXt-Base, while all styles are nearuniformly easy for CLIP ViT-L/14.

![](images/e0b9a12783a1a8a839a99fcd3041f1646d7eb85d9cde02f33a60a2ddae28d0a9.jpg)  
(a) CLIP ViT-L/14

![](images/25f4bf4620db66524a1b33033571f0fa181c085ea384fa9d589a191c94d2e235.jpg)  
(b) ConvNeXt-Base

Figure 6: ID test confusion matrices  
![](images/4d4f4c510e45ea0a5b3e1908b2605f8d4a6225643cec5d7c84970e401cad0f0d.jpg)  
Figure 7: ID test ROC curves for all detectors

![](images/efde6ea07f7060b9be0865420791efa0a499b6c0dadfa01badb0faaecaaabb9e.jpg)  
(a) CLIP ViT-L/14

![](images/efdee386dfdc2ee86293749b2657acc2268f71b3f1518d447c22676dda5fbe3a.jpg)  
(b) ConvNeXt-Base  
Figure 8: ID test predicted-probability histograms

Table 15: Per-style ID test balanced accuracy for CLIP ViT-L/14 and ConvNeXt-Base
<table><tr><td>Style</td><td>CLIP ViT-L/14</td><td>ConvNeXt-Base</td></tr><tr><td>Art Nouveau</td><td>0.9978</td><td>0.9860</td></tr><tr><td>Baroque</td><td>0.9978</td><td>0.9818</td></tr><tr><td>Expressionism</td><td>0.9955</td><td>0.9725</td></tr><tr><td>Impressionism</td><td>1.0000</td><td>0.9870</td></tr><tr><td>Post-Impressionism</td><td>0.9953</td><td>0.9770</td></tr><tr><td>Realism</td><td>0.9968</td><td>0.9523</td></tr><tr><td>Renaissance</td><td>0.9958</td><td>0.9713</td></tr><tr><td>Romanticism</td><td>0.9968</td><td>0.9658</td></tr><tr><td>Surrealism</td><td>0.9948</td><td>0.9420</td></tr><tr><td>Ukiyo-e</td><td>0.9995</td><td>0.9918</td></tr></table>

## 4.3 OOD Cross-Generator Evaluation

The main question in our OOD experiment is whether detectors trained on LDM and SD2.1 artwork can still work reliably on prompt-aligned SD3.5m images that were not seen during training. Table 16 reports the full OOD evaluation results for all models on the 20,000-image SD3.5m test set.

Table 16: OOD performance on the SD3.5m test set (column definitions are the same as Table 14, and all metrics are computed at the validation-selected threshold as per Table 13)
<table><tr><td>Model</td><td>Bal</td><td>Prec</td><td>Rec</td><td>F1</td><td>FNR</td><td>FPR</td><td>MCC</td><td>AUC</td><td>PR</td></tr><tr><td>CLIP ViT-L/14</td><td>0.7829</td><td>0.9967</td><td>0.5676</td><td>0.7233</td><td>0.4324</td><td>0.0019</td><td>0.6268</td><td>0.9658</td><td>0.9703</td></tr><tr><td>ConvNeXt-Base</td><td>0.7643</td><td>0.9537</td><td>0.5556</td><td>0.7021</td><td>0.4444</td><td>0.0270</td><td>0.5817</td><td>0.9256</td><td>0.9252</td></tr><tr><td>EfficientNet-B0</td><td>0.7156</td><td>0.9164</td><td>0.4745</td><td>0.6252</td><td>0.5255</td><td>0.0433</td><td>0.4922</td><td>0.8799</td><td>0.8782</td></tr><tr><td>ResNet-50</td><td>0.7125</td><td>0.9258</td><td>0.4619</td><td>0.6163</td><td>0.5381</td><td>0.0370</td><td>0.4910</td><td>0.8779</td><td>0.8811</td></tr><tr><td>ResNet-18</td><td>0.6688</td><td>0.8369</td><td>0.4193</td><td>0.5587</td><td>0.5807</td><td>0.0817</td><td>0.3896</td><td>0.7955</td><td>0.7927</td></tr></table>

Table 17 quantifies the performance drop in terms of the absolute balanced accuracy gap $\left( \Delta _ { \mathrm { B a l } } \right)$ and absolute recall gap $( \Delta _ { \mathrm { R e c } } )$ between the ID test set and the OOD set. Figures 9 and 10 visualize the OOD balanced accuracy ranking and the ID-to-OOD drop across all models.

Every model sufers a substantial performance drop on the OOD test set. ResNet-18 experiences the largest decrease, losing 25.7 percentage points of balanced accuracy and 51.0 percentage points of AI recall. Even the strongest model, CLIP ViT-L/14, loses 21.4 percentage points of balanced accuracy.

Critically, the failure mode is asymmetric. False negative rates increase sharply on the OOD set while false positive rates remain comparatively low: CLIP ViT-L/14 maintains an OOD FPR of just 0.0019, and ConvNeXt-Base maintains an OOD FPR of 0.0270. This implies that the detectors remain conservative in assigning the AI label. They rarely misclassify human paintings as AI-generated, but fail to recognize a large fraction of SD3.5m images. At the validation-selected thresholds, the five detectors miss approximately 4,300 to 5,800 of the 10,000 SD3.5m images. From a defensive perspective, these false negatives represent synthetic images that would pass the screening mechanism as human-created. In an operational workflow, such failures could allow synthetic content to bypass checks used for misinformation triage, marketplace fraud review, impersonation assessment or content-origin verification.

Table 17: ID-to-OOD performance gap $( \Delta _ { \mathrm { B a l } }$ is OOD minus ID balanced accuracy; $\Delta _ { \mathrm { R e c } }$ is OOD minus ID recall)
<table><tr><td rowspan="2">Model</td><td colspan="2">Balanced accuracy</td><td rowspan="2"> $\Delta _ { \mathrm { B a l } }$ </td><td rowspan="2"> $\Delta _ { \mathrm { R e c } }$ </td></tr><tr><td>ID</td><td>OOD</td></tr><tr><td>ResNet-18</td><td>0.9260</td><td>0.6688</td><td>-0.257</td><td>-0.510</td></tr><tr><td>ResNet-50</td><td>0.9637</td><td>0.7125</td><td>-0.251</td><td>-0.501</td></tr><tr><td>EfficientNet-B0</td><td>0.9524</td><td>0.7156</td><td>-0.237</td><td>-0.474</td></tr><tr><td>CLIP  $\mathrm { V i T - L } / 1 4$ </td><td>0.9969</td><td>0.7829</td><td>-0.214</td><td>-0.430</td></tr><tr><td>ConvNeXt-Base</td><td>0.9727</td><td>0.7643</td><td>-0.208</td><td>-0.417</td></tr></table>

![](images/6a00026faa132c709ac8c97732a74fe22cceca99038f76fbcca6d6ab9cdd081a.jpg)  
Figure 9: OOD balanced accuracy across detectors

![](images/8f57cecb23d60d0e8d748d729dcf70ad95e2d3cab2c75d7562c84b5ecc0f11a0.jpg)  
Figure 10: ID-test and OOD balanced accuracy

Among the five models, ConvNeXt-Base is the strongest CNN-based detector on the OOD test set, achieving a balanced accuracy of 0.764 and a ROC-AUC of 0.926, while CLIP ViT-L/14 performs best overall with a balanced accuracy of 0.783 and a ROC-AUC of 0.966. ResNet-50 and EficientNet-B0 show very similar OOD performance (0.712 and 0.716 balanced accuracy, respectively), although their in-distribution results difer more noticeably. This suggests that improvements in standard in-distribution performance do not necessarily translate into proportional gains in cross-generator generalization. More broadly, the results show that strong performance on known generators may create a misleading impression of security when a deployed detector is not regularly evaluated against newly emerging generation systems [34].

Figure 11 presents the OOD confusion matrices for CLIP ViT-L/14 and ConvNeXt-Base. The matrices show that OOD degradation is driven primarily by false negatives, with many SD3.5m AI images predicted as human, while false positives on human artwork samples remain relatively uncommon. This directly illustrates the asymmetric error pattern underlying the OOD performance drop.

![](images/7432beb30503e67bc84bf0c68532866c00ccb6a229c95e9e7c9d10ef4718deb1.jpg)  
(a) CLIP ViT-L/14

![](images/b83437d016b8ff82f128d6607b2debd01fcd62e69cf16240b0666c91cb3fced4.jpg)  
(b) ConvNeXt-Base  
Figure 11: OOD confusion matrices

Figure 12 shows the OOD ROC curves for all five models overlaid. Unlike the tightly clustered ID ROC curves in Figure 7, the OOD curves spread out visibly, reflecting the divergence in model generalization.

The probability histograms in Figure 13 emphasize the OOD failure mode. On the ID test set, human and AI scores are strongly separated, with human images concentrated near $p _ { \mathrm { A I } } = 0$ and AI images near $p _ { \mathrm { A I } } = 1$ . In contrast, for the OOD SD3.5m dataset, this separation weakens substantially, as the AI-score distribution becomes much more spread out and shifts toward lower values, with many SD3.5m images falling below the selected threshold. As a result, a large number of OOD AI images are misclassified as human.

Next, we consider t-SNE plots, which provide qualitative visualizations of the feature-space structure. For these visualizations, a random subset of 2,000 images was sampled from each evaluated split. Frozen backbone features were first reduced to 50 dimensions using PCA. Then t-SNE was run with perplexity 30, PCA initialization, and automatic learning rate selection. Figure 14 shows the resulting t-SNE projections of the OOD feature space for CLIP ViT-L/14 and ConvNeXt-Base, with human and SD3.5m AI images colored diferently. The substantial overlap between the two classes suggests that the decision boundary learned from the ID data does not transfer cleanly to the OOD distribution.

![](images/ffae69c3962ec510fe73b65c9c2681ea1d03383a45cfaf021937f1f97a048321.jpg)  
Figure 12: OOD ROC curves across detectors

![](images/7378b2afe72e7e1dde6aa8fe49ea5c97a85698e720b538438f3500ab617a6faf.jpg)  
(a) CLIP ViT-L/14

![](images/d7fa62965d3948c5d0e1f36a792caad13fda0ed91b8a115978b2683a648871b5.jpg)  
(b) ConvNeXt-Base  
Figure 13: OOD predicted probability histograms

## 4.4 Style-wise OOD Performance

The overall OOD metrics reported in Section 4.3 provide a useful summary of detector performance across all ten art styles. A style-wise analysis complements this view by showing how generator shift manifests within individual artistic movements. Diferent styles have distinct visual vocabularies: for example, Ukiyo-e is characterized by flat color areas and bold outlines, whereas Realism emphasizes naturalistic textures and photographic plausibility. SD3.5m may preserve these stylistic properties to diferent degrees across styles. Examining per-style OOD performance helps identify which styles remain more robust under generator shift, which styles are more challenging, and whether these patterns are consistent across all five backbones or vary by architecture.

![](images/853abca253ee075e745dc89b6ce9785d1aa672efdf51778b727358a2f6df7c56.jpg)  
(a) CLIP ViT-L/14

![](images/7e098a2e1cb669f39cc14e2d50d381cc54415c66ee879c1e37c074f99cfafbdc.jpg)  
(b) ConvNeXt-Base  
Figure 14: t-SNE projections of the OOD feature space

Table 18 reports style-wise OOD balanced accuracy for all five detectors. Each cell corresponds to the 2,000-image OOD subset for that style (1,000 human and 1,000 SD3.5m AI), and the rightmost column gives the mean across all five models. Figure 15 visualizes the same style-by-model matrix as a heatmap, making the overall dificulty pattern across detector backbones easier to compare. Ukiyo-e (bottom row) is consistently the easiest style for all models, whereas Realism (top row) is the hardest, with no model exceeding a balanced accuracy of 0.694 for Realism.

Table 18: OOD balanced accuracy per style for all five detectors
<table><tr><td>Style</td><td>RN18</td><td>RN50</td><td>ENB0</td><td>CNX</td><td>CLIP</td><td>Mean</td></tr><tr><td>Realism</td><td>0.631</td><td>0.647</td><td>0.661</td><td>0.666</td><td>0.694</td><td>0.660</td></tr><tr><td>Romanticism</td><td>0.636</td><td>0.673</td><td>0.675</td><td>0.702</td><td>0.740</td><td>0.685</td></tr><tr><td>Impressionism</td><td>0.659</td><td>0.693</td><td>0.686</td><td>0.708</td><td>0.752</td><td>0.700</td></tr><tr><td>Post-Impressionism</td><td>0.655</td><td>0.684</td><td>0.681</td><td>0.743</td><td>0.754</td><td>0.703</td></tr><tr><td>Expressionism</td><td>0.655</td><td>0.678</td><td>0.707</td><td>0.746</td><td>0.751</td><td>0.707</td></tr><tr><td>Art Nouveau</td><td>0.692</td><td>0.723</td><td>0.717</td><td>0.840</td><td>0.709</td><td>0.736</td></tr><tr><td>Surrealism</td><td>0.654</td><td>0.687</td><td>0.725</td><td>0.782</td><td>0.837</td><td>0.737</td></tr><tr><td>Baroque</td><td>0.668</td><td>0.737</td><td>0.715</td><td>0.738</td><td>0.833</td><td>0.738</td></tr><tr><td>Renaissance</td><td>0.708</td><td>0.743</td><td>0.726</td><td>0.776</td><td>0.795</td><td>0.750</td></tr><tr><td>Ukiyo-e</td><td>0.733</td><td>0.862</td><td>0.865</td><td>0.943</td><td>0.966</td><td>0.874</td></tr><tr><td>Average</td><td>0.669</td><td>0.712</td><td>0.716</td><td>0.764</td><td>0.783</td><td>0.729</td></tr></table>

Recall is particularly important in the OOD setting because it measures the detector’s ability to correctly identify SD3.5m AI images as AI. Style-wise, recall provides a useful complement to balanced accuracy by showing which styles most often lead to missed AI detections. Figure 16 presents this recall-based view and highlights which styles are prone to false negatives under generator shift.

![](images/ab9d971d18184f0a9f46bca0a9d283e0c844b8fdecca9036a9d43aabacfa507a.jpg)

Figure 15: Heatmap of OOD balanced accuracy across art styles and detectors  
![](images/bbffbc675b1ef1e4b614213e51d504f52432faf276a98b3b174f22abb3f421b0.jpg)  
Figure 16: Mean OOD AI recall across all detectors per style

The styles are ordered from hardest to easiest by mean OOD AI recall, giving a model-agnostic view of style-level dificulty under generator shift. Realism is the hardest style at 0.337. Ukiyo-e is the easiest at 0.758, while all other styles fall below 0.57. Ukiyo-e stands out because its visual structure (flat color regions, bold outlines and clear linework) remains distinctive in the SD3.5m output, making these images easier for detectors to separate from human artwork samples. Realism is the most dificult style as the detectors miss about two thirds of SD3.5m images in this category. Romanticism, Impressionism, and Post-Impressionism form the next hardest group, each with recall below 0.44, suggesting that SD3.5m images in these styles are more visually similar to authentic human paintings in color and composition.

## 4.5 Source-wise In-Distribution Performance

The in-distribution data consists of images from three source classes: human artwork samples, LDM-generated images, and SD2.1-generated images. Although the binary task merges LDM and SD2.1 into a single AI class, examining performance by source reveals whether detectors treat both generators similarly or whether one is easier to identify than the other. This also provides a useful baseline for understanding how each model distributes its errors between false positives on human images and false negatives on each AI generator, which in turn helps contextualize the OOD results.

Table 19 reports source-wise performance for all models. These results show how errors are distributed between false positives on human artwork samples and false negatives for both AI generators.

Table 19: Source-wise performance on ID test set
<table><tr><td>Model</td><td> $\mathbf { F P R _ { h u m a n } }$ </td><td> $\mathbf { R e c a l l _ { L D M } }$ </td><td>RecallsD2.1</td></tr><tr><td>CLIP ViT-L/14</td><td>0.003</td><td>0.995</td><td>0.999</td></tr><tr><td>ConvNeXt-Base</td><td>0.027</td><td>0.960</td><td>0.985</td></tr><tr><td>ResNet-50</td><td>0.035</td><td>0.949</td><td>0.976</td></tr><tr><td>EfficientNet-B0</td><td>0.044</td><td>0.928</td><td>0.968</td></tr><tr><td>ResNet-18</td><td>0.077</td><td>0.918</td><td>0.939</td></tr></table>

Across all five models, SD2.1 images are consistently detected at a higher rate than LDM images, with SD2.1 recall exceeding LDM recall by up to four percentage points, depending on the backbone. One possible explanation is that SD2.1, with its default sampling configuration, introduces slightly more distinctive statistical patterns at the resolutions used here, though a definitive attribution would require further investigation. The distribution of errors on human images also varies with model capacity: ResNet-18 produces a human false positive rate of 0.077, whereas CLIP ViT-L/14 reduces this to 0.003, consistent with higher-capacity backbones learning more selective decision boundaries. On the ID test set, every model achieves recall above 0.91 for both LDM and SD2.1, whereas on the OOD set, every model falls below 0.57 recall on SD3.5m images. This contrast suggests that the performance drop is linked to the specific characteristics of the new generator rather than a general degradation of detector performance.

Figure 17 shows the ID test t-SNE colored by source class. These results show that LDM and SD2.1 images form distinguishable sub-clusters within the broader AI region.

![](images/1380dcb9ab1d4baa9089c0fba3d6e5cdf86ec08c8dc28243971cb541a1f46e94.jpg)  
(a) CLIP ViT-L/14

![](images/deba01a44a76e126e71c6be9d78a5e21a3ed4a6e5ab2cdd6afb8240e7776e5e4.jpg)  
(b) ConvNeXt-Base  
Figure 17: t-SNE projections of ID test feature space colored by source class

## 4.6 Explainability and Failure Analysis

Previous sections measured the OOD performance drop caused by generator shift. This section adds a qualitative perspective by examining where the detector focuses when making correct and incorrect predictions. We apply Grad-CAM to ConvNeXt-Base, which is the best performing of the CNN-based detectors considered in this study. The final feature map of the ConvNeXt-Base backbone was used as the target layer and the AI class was used as the target output [47]. Since the classifier produces a single AI logit �, the human-target attribution was computed using the complementary score −�, while the AI-target attribution was computed using �.

Three cases are examined including correctly detected ID AI images, OOD false negatives, and OOD false positives. These attribution maps help show which image regions support successful detection and how the model’s focus changes when it fails under generator shift. The Grad-CAM maps are used only as qualitative visualizations of model behavior and not as complete explanations of the model’s decision process.

## 4.6.1 Detection Success: ID True Positives

Figure 18 shows Grad-CAM overlays for one confidently detected AI (LDM or SD2.1) image per style. Specifically, the image selected is that which is assigned the highest AI probability for each style on the ID test set.

For ID true positives, the activation maps are usually focused on specific parts of the image rather than spread across the full canvas. In portrait-like styles such as Baroque, Realism, Renaissance and Romanticism, the strongest responses appear around faces, neck regions, upper body outlines and sometimes near frame boundaries. In Expressionism, Impressionism and Post-Impressionism, the activation is concentrated near the main subject and nearby high contrast or strongly textured regions. Surrealism and Ukiyo-e show a similar pattern, but the activated regions are spread across distinctive parts of the composition rather than one small object. Overall, these maps suggest that the detector uses local visual cues to identify in-distribution AI-generated images instead of looking at the image holistically.

![](images/87ea479e51a5864848b1fbd31c7202581d5c4475ef7546fbf24d39127aebe31f.jpg)  
Figure 18: Grad-CAM attribution maps for ID true positives with ConvNeXt-Base

## 4.6.2 Detection Fails: OOD False Negatives

Figure 19 shows Grad-CAM overlays for the OOD false negatives, one SD3.5m image per style selected as the most convincingly human-like miss (i.e., lowest predicted AI probability) for each style. The contrast between Figures 18 and 19 ofers a visual perspective on the diference between successful ID detection and OOD failure.

![](images/eaf57997d1181d6ae9590ec54dace0e2c8fd1589a72f6f8132f84fe98bb67f34.jpg)  
Figure 19: Grad-CAM attribution maps for OOD false negatives with ConvNeXt-Base

For OOD false negatives, the Grad-CAM maps are weaker and less clear than the maps for correct ID AI detections. The model does not focus strongly on the main subject or other important parts of the image. Instead, it looks at small areas near the image edges, frame-like regions or isolated texture details. In some cases, the activation appears in only one or two small spots. These weak and peripheral attribution patterns are associated with a lower predicted AI scores. These observations are consistent with the OOD results and indicate that SD3.5m does not produce the same AI-related cues that the detector learned from LDM and SD2.1 images.

Figure 20 shows a representative OOD false negative analyzed with both AItarget and human-target Grad-CAM. In this example, the AI-target map is weak and concentrated near a peripheral lower-image region, while the human-target map is stronger and more focused on the dog and surrounding foreground. This suggests that the highlighted regions provide stronger evidence against the AI classification than evidence supporting it, pushing the model toward a human prediction.

![](images/c0ac29e1ecfc91b4fb6426e3042070f3ff3b286dbcd6a8dc5b1daffa37b6e2e5.jpg)  
Figure 20: Grad-CAM for an OOD false negative sample with ConvNeXt-Base

## 4.6.3 False Alarms: OOD False Positives

Figure 21 shows Grad-CAM overlays for OOD false positives. It includes one human artwork from each style that was incorrectly classified as AI. For each style, the selected example is the human artwork with the highest predicted AI probability.

![](images/9523781bc3f2bf6f2a22c82ab873b741de6131d8c694ea7a36d83aee0dc7c0f8.jpg)  
Figure 21: Grad-CAM attribution maps for OOD false positives with ConvNeXt-Base

For OOD false positives, the Grad-CAM maps appear more organized than the maps for OOD false negatives. In many cases, the model focuses on faces, facial outlines, frame boundaries or other clear local regions instead of spreading attention across the whole image. This suggests that false positives are caused by specific visual patterns that the detector connects with the AI class, although the image is actually human-made. These mistakes are not very common, since the overall OOD false positive rate is low, at roughly 0.2% to 8.2% across models. But when these errors do occur, the Grad-CAM maps show that the model is usually reacting to certain parts of the image rather than the full image as a whole.

## 4.6.4 Qualitative Summary

Overall, the Grad-CAM results provide qualitative evidence consistent with the OOD performance drop. For in-distribution images, ConvNeXt-Base focuses on clear local regions that appear useful for detecting LDM and SD2.1 images. These cues allow the linear classifier to separate AI-generated images from human artwork samples.

For the SD3.5m OOD set, these cues are weaker and less consistent. Many false negatives show only small or poorly placed activation regions that suggest the detector did not find the same AI-related evidence in SD3.5m images. This supports the quantitative results, where the main OOD failure comes from missed AI detections.

The style-wise patterns also match the Grad-CAM results. Realism and Romanticism show weaker activation which is consistent with their lower OOD recall. In contrast, Ukiyo-e shows more structured activation around distinctive visual elements such as flat color regions, bold outlines and calligraphic markings. This is consistent with Ukiyo-e being the easiest OOD style.

CLIP ViT-L/14 shows the same general failure pattern with low false positives but many missed SD3.5m images. Nevertheless, it performs the best overall, suggesting that its broader semantic features are more robust under generator shift.

## 5 Conclusion

In this chapter, we focused on whether AI-art detectors trained on earlier difusion generators remain viable when applied to images produced by a newer, architecturally diferent model. To study this question, a prompt-aligned Stable Difusion 3.5 Medium (SD3.5m) dataset was constructed across 10 art styles and used for out-of-distribution evaluation alongside a held-out human reference set. Five frozenbackbone detectors, namely, ResNet-18, ResNet-50, EficientNet-B0, ConvNeXt-Base, and CLIP ViT-L/14, were then trained on in-distribution artwork from LDM and SD2.1 and evaluated under a common experimental setup.

The results reveal a consistent pattern. All five detectors achieve strong indistribution performance, indicating that the frozen-backbone, linear-probe design is efective when training and test data come from the same generator family. However, under generator shift to SD3.5m, every model experiences a substantial drop in performance. The degradation is strongly asymmetric with false positive rates on human artwork samples remaining low while recall on SD3.5m images falls sharply. This means the primary failure is not an increase in false alarms on human artwork, but a substantial increase in missed detections of images produced by the newer generator. Among the five models, CLIP ViT-L/14 achieves the strongest OOD performance overall, while ConvNeXt-Base is the best performing CNN-based model. Style-wise analysis further shows that generator shift does not afect all artistic movements equally, with Ukiyo-e remaining comparatively easy and Realism being the most dificult style under OOD evaluation.

The qualitative analysis also supports these findings. Grad-CAM maps for indistribution true positives show more focused and structured activation, whereas OOD false negatives often produce weaker and more fragmented responses. This suggests that the visual cues learned from LDM and SD2.1 do not transfer cleanly to SD3.5m. Overall, the results indicate that strong in-distribution accuracy is not suficient evidence of detector robustness when the underlying generator changes. More broadly, they highlight a practical limitation of current AI-art detectors: systems that perform well on known generators may generalize poorly to newer models with diferent synthesis mechanisms.

This limitation has direct cybersecurity implications and demonstrates why AIart detectors should not be treated as static or standalone mechanisms for contentauthenticity verification. A detector that performs almost perfectly on known generators may still allow a substantial fraction of images from an emerging generator to pass as human-created. Such false negatives could weaken screening workflows that support misinformation review, the investigation of fraudulent authenticity claims, impersonation assessment, and content-origin verification. The risk is particularly important because it does not require a sophisticated or adaptive adversary. Simply switching to a newer generator may be suficient to reduce detector reliability, creating a low barrier evasion pathway that does not require direct access or manipulation of the detector. Image-based detection should therefore be used as one signal within a layered defensive process that also considers provenance metadata, watermark or content-credential verification, source analysis, drift monitoring, periodic evaluation against newly released generators and human review.

The findings of this study point to several directions for future work. First, the detectors are evaluated in a frozen-backbone, linear-probe setting, which enables controlled comparison across architectures. However, future work could test whether backbone fine-tuning or parameter-eficient adaptation can improve crossgenerator robustness. Second, the OOD evaluation uses only Stable Difusion 3.5 Medium as the new generator, so testing across additional recent image generation models would help determine whether the same failure pattern appears more broadly. Also, even though the SD3.5m dataset is prompt aligned with human artworks, there may still be some prompt distribution shift. This is because the SD3.5m images are generated using reverse prompted and title augmented prompts rather than the original AI-ArtBench prompt pipeline. Third, all detectors operate on resized inputs, which may suppress fine-scale forensic cues available at higher resolutions, suggesting that multi-scale or frequency-aware approaches could be valuable extensions. At the same time, extending the framework from binary human-versus-AI detection to multi-generator attribution would provide a more detailed view of how diferent synthesis pipelines leave detectable traces.

Overall, the prompt-aligned dataset construction and OOD evaluation performed in this chapter provide a practical foundation for future studies and broader research on robustness in AI-art detection and its application in cybersecurity workflows.

## References

[1] Roberto Amoroso, Davide Morelli, Marcella Cornia, et al. Parents and children: Distinguishing multimodal deepfakes from natural images. ACM Transactions on Multimedia Computing, Communications and Applications, 21(1):1– 23, 2024.

[2] Pierre Baldi, Søren Brunak, Yves Chauvin, et al. Assessing the accuracy of prediction algorithms for classification: an overview. Bioinformatics, 16(5):412– 424, 2000.

[3] M Bi´nkowski, Danica J Sutherland, Michael Arbel, et al. Demystifying MMD GANs. https://arxiv.org/abs/1801.01401, 2018.

[4] Jordan J. Bird and Ahmad Lotfi. CIFAKE: Image classification and explainable identification of AI-generated synthetic images. IEEE Access, 12:15642–15650, 2024.

[5] Aras Bozkurt. GenAI et al. cocreation, authorship, ownership, academic ethics and integrity in a time of generative ai. Open Praxis, 16(1):1–10, 2024.

[6] Andrew Brock, Jef Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. https://arxiv.org/abs/1809. 11096, 2018.

[7] Hongfei Cai, Chi Liu, Sheng Shen, et al. Robust AI-synthesized image detection via multi-feature frequency-aware learning. In International Conference on Knowledge Science, Engineering and Management, KSEM, pages 157–171, 2025.

[8] Eva Cetinic and James She. Understanding and creating art with AI: Review and outlook. ACM Transactions on Multimedia Computing, Communications, and Applications, 18(2):1–22, 2022.

[9] Christie’s. Obvious and the interface between art and artificial intelligence. https://www.christies.com/en/stories/acollaboration-between-two-artists-one-human-one-a-machine-0cd01f4e232f4279a525a446d60d4cd1.

[10] Vito Nicola Convertini, Donato Impedovo, Ugo Lopez, et al. Discrete Fourier transform in unmasking deepfake images: A comparative study of StyleGAN creations. Information, 15(11):711, 2024.

[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. An image is worth 16 × 16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, ICLR, 2021.

[12] Hicham Eddoubi, Jonas Ricker, Federico Cocchi, et al. RAID: A dataset for testing the adversarial robustness of AI-generated image detectors. https: //arxiv.org/abs/2506.03988, 2025.

[13] Patrick Esser, Sumith Kulal, Andreas Blattmann, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-First International Conference on Machine Learning, ICML, 2024.

[14] Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR, pages 2414–2423, 2016.

[15] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, et al. Generative adversarial nets. Advances in Neural Information Processing Systems, pages 2672– 2680, 2014.

[16] Anna Yoo Jeong Ha, Josephine Passananti, Ronik Bhaskar, et al. Organic or difused: Can we distinguish human art from AI-generated images? In Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security, pages 4822–4836, 2024.

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, pages 770–778, 2016.

[18] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (GELUs). https://arxiv.org/abs/1606.08415, 2016.

[19] Jack Hessel, Ari Holtzman, Maxwell Forbes, et al. CLIPscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP, pages 7514– 7528, 2021.

[20] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, et al. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Advances in Neural Information Processing Systems, NeurIPS, pages 6627–6638, 2017.

[21] Ruixiang Jiang and Chang Wen Chen. Multimodal LLMs can reason about aesthetics in zero-shot. In Proceedings of the 33rd ACM International Conference on Multimedia, MM ’25, pages 6634–6643, 2025.

[22] Xuan Ju, Ailing Zeng, Jianan Wang, et al. Human-art: A versatile humancentric dataset bridging natural and artificial scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CPVR, pages 618–629, 2023.

[23] Negar Kamali, Karyn Nakamura, Aakriti Kumar, et al. Characterizing photorealism and artifacts in difusion model-generated images. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, CHI ’25, pages 1–26, 2025.

[24] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, pages 4401–4410, 2019.

[25] Junnan Li, Dongxu Li, Caiming Xiong, et al. BLIP: Bootstrapping languageimage pre-training for unified vision-language understanding and generation.

In International Conference on Machine Learning, ICML, pages 12888–12900, 2022.

[26] Meien Li and Mark Stamp. Detecting AI-generated artwork. In Proceedings of the Fifth Annual Computer Science Conference for CSU Undergraduates, CSCSU, 2025. https://arxiv.org/abs/2504.07078.

[27] Peiyuan Liao, Xiuyu Li, Xihui Liu, et al. The artbench dataset: Benchmarking generative models with artworks. https://arxiv.org/abs/2206.11404, 2022.

[28] Haotian Liu, Chunyuan Li, Qingyang Wu, et al. Visual instruction tuning. Advances in Neural Information Processing Systems, 36:34892–34916, 2023.

[29] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, et al. A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, pages 11976–11986, 2022.

[30] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR, 2019. https: //arxiv.org/abs/1711.05101.

[31] Brian W. Matthews. Comparison of the predicted and observed secondary structure of T4 phage lysozyme. Biochimica et Biophysica Acta (BBA)-Protein Structure, 405(2):442–451, 1975.

[32] Alexander Mordvintsev, Christopher Olah, and Mike Tyka. Inceptionism: Going deeper into neural networks. https://research.google/blog/ inceptionism-going-deeper-into-neural-networks/.

[33] Stefano Morelli. WikiArt. https://www.kaggle.com/datasets/steubk/ wikiart.

[34] NIST. Artificial intelligence risk management framework (AI RMF 1.0). https: //nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf, 2023.

[35] Seyeon Park and Xiaoli Nan. Generative ai and misinformation: a scoping review of the role of generative AI in the generation, detection, mitigation, and impact of misinformation. AI & SOCIETY, 41(2):1501–1515, 2026.

[36] Adam Paszke, Sam Gross, Francisco Massa, et al. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, NeurIPS, 2019.

[37] William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, pages 4195–4205, 2023.

[38] pharmapsychotic et al. pharmapsychotic/clip-interrogator. https://github. com/pharmapsychotic/clip-interrogator, 2023.

[39] Alec Radford, Jong Wook Kim, Chris Hallacy, et al. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML, pages 8748–8763, 2021.

[40] Colin Rafel, Noam Shazeer, Adam Roberts, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.

[41] Md Awsafur Rahman, Bishmoy Paul, Najibul Haque Sarker, et al. Artifact: A large-scale dataset with artificial and factual images for generalizable and robust synthetic image detection. In 2023 IEEE International Conference on Image Processing, ICIP, pages 2200–2204, 2023.

[42] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, et al. Zero-shot text-to-image generation. In International Conference on Machine Learning, ICML, pages 8821–8831, 2021.

[43] Zhiyao Ren, Yibing Zhan, Baosheng Yu, et al. Reverse prompt: Cracking the recipe inside text-to-image generation. https://arxiv.org/abs/2503.19937, 2025.

[44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, et al. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, pages 10684– 10695, 2022.

[45] Sudhir K Routray. Ethical considerations and implications of generative AI in computer graphics. IEEE Computer Graphics and Applications, 45:78–89, 2025.

[46] Olga Russakovsky, Jia Deng, Hao Su, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211–252, 2015.

[47] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision, ICCV, pages 618–626, 2017.

[48] Shawn Shan, Jenna Cryan, Emily Wenger, et al. Glaze: Protecting artists from style mimicry by {Text-to-Image} models. In 32nd USENIX Security Symposium, USENIX Security, pages 2187–2204, 2023.

[49] Ravidu Suien Rammuni Silva, Ahmad Lotfi, Isibor Kennedy Ihianle, et al. ArtBrain: An explainable end-to-end toolkit for classification and attribution of AI-generated art and style. https://arxiv.org/abs/2412.01512, 2024.

[50] Chuangchuang Tan, Xiang Ming, Jinglu Wang, et al. Semantic visual anomaly detection and reasoning in AI-generated images. In International Conference on Learning Representations, ICLR, 2026. https://openreview.net/forum? id=0iN4UKZwgn.

[51] Mingxing Tan and Quoc Le. EficientNet: Rethinking model scaling for convolutional neural networks. In Proceedings of the 36th International Conference on Machine Learning, ICML, pages 6105–6114, 2019.

[52] Sheng-Yu Wang, Oliver Wang, Richard Zhang, et al. CNN-generated images are surprisingly easy to spot . . . for now. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, CVPR, pages 8695– 8704, 2020.

[53] Zhendong Wang, Jianmin Bao, Wengang Zhou, et al. DIRE for difusiongenerated image detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, pages 22445–22455, 2023.

[54] Yuxin Wen, Neel Jain, John Kirchenbauer, et al. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. In Advances in Neural Information Processing Systems, NeurIPS, pages 51008– 51025, 2023.

[55] Thomas Wolf, Lysandre Debut, Victor Sanh, et al. Transformers: State-ofthe-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, EMNLP, pages 38–45, 2020.

[56] Ling Yang, Zhilong Zhang, Yang Song, et al. Difusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4):1–39, 2023.

[57] Richard Zhang, Phillip Isola, Alexei A Efros, et al. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, pages 586– 595, 2018.

[58] Xu Zhang, Svebor Karaman, and Shih-Fu Chang. Detecting and Simulating Artifacts in GAN Fake Images. In 2019 IEEE International Workshop on Information Forensics and Security, WIFS, pages 1–6, 2019.

[59] Eric Zhou and Dokyun Lee. Generative artificial intelligence, human creativity, and art. PNAS Nexus, 3(3):pgae052, 2024.

[60] Mingjian Zhu, Hanting Chen, Qiangyu Yan, et al. Genimage: A million-scale benchmark for detecting AI-generated image. Advances in Neural Information Processing Systems, pages 77771–77782, 2023.