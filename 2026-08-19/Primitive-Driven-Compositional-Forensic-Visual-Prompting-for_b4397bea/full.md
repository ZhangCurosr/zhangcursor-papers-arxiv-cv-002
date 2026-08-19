# Primitive-Driven Compositional Forensic Visual Prompting for Open-World Face Anti-Spoofing

Fangling Jiang, Qi Li, Bing Liu, Weining Wang, Quilin Huang, Zhenan Sun, Ming-Hsuan Yang

Abstract—Open-world face anti-spoofing must address both covariate and semantic shifts: source and target domains differ in imaging conditions, while target domains contain diverse attack types absent from training. Existing prompt-based approaches often express spoofing through category semantics or language guidance, which is effective for modeling high-level concepts but is less suited to explicitly capturing the evolving fine-grained and spatially heterogeneous forensic evidence of unseen attacks. Motivated by the hypothesis that many unseen attacks can be characterized by new combinations of recurring visual cues, we propose a compositional forensic visual prompt learning framework that operates entirely in the visual feature space. Built on a frozen ViTbased vision foundation model, the framework employs patchaware attention to refine a shared set of learnable micro-forensic primitives into localized forensic evidence units derived from image patches. Class-specific global contextual prompts then provide input-dependent routing weights that adaptively select and compose these primitives into compositional forensic visual prompts for real/spoof discrimination. The primitives are not assigned predefined semantic meanings; instead, their specialization and reuse emerge from shared parameterization and joint optimization across categories. Extensive experiments on nine open-world protocols demonstrate state-of-the-art performance, strong cross-domain generalization, and robust adaptation to unseen attacks. Further analyses in the frequency, spatial, feature, and primitive domains show that the learned primitives capture transferable and composable local forensic evidence, while global-context-guided routing produces input-dependent and discriminative primitive compositions. These results support visual primitive composition as an effective paradigm for detecting heterogeneous attacks in open-world face anti-spoofing.

Index Terms—Face anti-spoofing, Face Presentation Attack Detection, Unseen attack detection, Cross-scenario testing.

## I. INTRODUCTION

tation attacks and is essential to securing face recognition systems. Existing studies [1] have primarily investigated crossscenario generalization, typically assuming that source and target domains share the same attack types (e.g., print and replay attacks) but differ in acquisition conditions, such as cameras, illumination, and backgrounds. Under this assumption, the dominant challenge is covariate shift, which is commonly addressed through domain adaptation [2]–[5] and domain generalization [6]–[9]. Real-world face anti-spoofing, however, is inherently open-world [10]–[12]. The asymmetric adversarial dynamics between attackers and defenders continually produce unseen attacks with distinct materials, geometric structures, and visual appearances, as illustrated in Fig. 1. Open-world continually evolving spoofing attacks exhibit substantial heterogeneity in both visual patterns and spatial distributions. Their forensic evidence spans multiple scales, ranging from global physical anomalies, such as three-dimensional geometric distortions, to local fine-grained artifacts. Such evidence varies markedly both across and within attack types, and may occur over the entire face or be confined to different facial regions. These attacks introduce semantic shifts beyond conventional covariate shift, causing models trained on seen attack types to generalize poorly [10]–[12].

Recent studies exploit the transferable knowledge of large scale vision-language models through textual or multimodal prompt learning [13]–[25], achieving promising performance under both covariate and semantic shifts. However, they rely on language semantics as an intermediary for visual knowledge representation, making them inherently limited in modeling heterogeneous unseen attacks. Text encoders emphasize category-level abstraction and may suppress low-level and high-frequency forensic evidence that is critical for face anti-spoofing during cross-modal alignment. Moreover, many spoof-specific cues, such as irregular specular reflections, material texture anomalies, and subtle frequency distortions, are difficult to describe precisely in natural language, while textual prompts typically provide global semantic supervision with limited spatial sensitivity. More importantly, fixed textual prompts used at inference time are difficult to adapt to continually emerging unseen attacks with diverse physical properties and forensic manifestations.

Motivated by this gap, we investigate a compositional visual diagnosis paradigm. Our working hypothesis is that many unseen attacks can be characterized by previously observed forensic cues appearing in new combinations rather than by an entirely new physical mechanism. For example, a siliconemask attack can contain a combination of geometric rigidity, unusual reflection, and altered skin texture, while related cues may also occur individually in other seen presentation attacks, such as print and replay attacks. We therefore represent spoofing evidence with a shared set of learnable visual primitives and combine them adaptively for each input. We use the term compositional in an operational sense: the class-specific representation is explicitly constructed as an input-dependent weighted combination of refined evidence primitives; the primitives are not assumed to correspond to manually defined semantic factors.

![](images/96dd68b326a473b03878aa88efbf36d9229a1318c93af939aed41e52672a8c2d.jpg)  
Fig. 1. (a) Open-world face anti-spoofing exhibits severe covariate and semantic shifts, resulting in continuously evolving heterogeneous spoof cues in both pattern and spatial distribution. (b) Our framework learns reusable micro-forensic primitives and dynamically composes them using image-level physical priors for unified detection of heterogeneous attacks. (c) Motivating hypothesis: heterogeneous unseen attacks may be represented by new combinations of recurring visual forensic cues.

Accordingly, we propose a purely visual prompt learning framework based on micro-forensic primitive composition, as illustrated in Fig. 1. Unlike language-mediated prompting, the proposed framework learns prompts directly in the continuous visual feature space, enabling joint modeling of low-frequency global structures and high-frequency local artifacts that are difficult to describe linguistically. We initialize micro-forensic primitives as learnable vectors and refine them through patchaware attention, allowing each primitive to specialize in a distinct type of reusable fine-grained evidence, such as geometric rigidity, abnormal reflections, and missing skin texture. In this way, heterogeneous spoofing traces are decomposed into elementary evidence units and organized within a shared primitive library, providing a structured basis for compositional diagnosis. We further introduce a dynamic routing mechanism in which global contextual prompts encode image-level physical priors and serve as conditioning signals for primitive composition. Guided by the holistic physical characteristics of each input, the framework adaptively selects and combines taskrelevant primitives to generate input-specific compositional forensic visual prompts for real/spoof discrimination.

The proposed framework avoids reliance on predefined attack categories or textual descriptions and establishes a unified reasoning paradigm for both seen and unseen attacks directly in the visual feature space. By learning compact real representations and organizing heterogeneous spoofing patterns within a shared primitive space, the framework provides an explicit intermediate representation in which local forensic evidence can be reused across attack types, thereby enabling compositional reasoning over unseen heterogeneous attacks and improving zero-shot generalization in open-world scenarios.

## The main contributions of this work are:

• We formulate open-world face anti-spoofing from a compositional visual evidence perspective, representing heterogeneous attacks with input-dependent combinations of shared forensic evidence units rather than fixed attackcategory descriptions.

• We propose a purely visual prompt learning framework that organizes spoofing artifacts into reusable microforensic primitives and dynamically routes and composes them into input-adaptive compositional forensic visual prompts for unified inference across diverse seen and unseen attacks.

• Extensive experiments on nine open-world protocols demonstrate that the proposed method achieves stateof-the-art performance under concurrent covariate and semantic shifts, together exhibiting strong zero-shot generalization to heterogeneous unseen attacks.

## II. RELATED WORK

## A. Cross-Domain Face Anti-Spoofing

Cross-domain face anti-spoofing aims to improve generalization ability under unseen acquisition devices, illumination conditions, background environments, and attack types. Existing studies mainly follow four directions: domain adaptation, domain generalization, multimodal generalization, and openset/one-class generalization.

Domain adaptation methods exploit target domain data under unsupervised, semi-supervised, source-free, or test-time settings to reduce source and target discrepancies. Representative techniques include adversarial alignment [2], [26], pseudolabel learning [3], [27], optimal transport [4], [5], and selfsupervised exploration [28]. In contrast, domain generalization methods learn transferable representations without accessing target domain data. Existing approaches employ domainadversarial learning [6], [7], [9], knowledge distillation [29], gradient or consistency regularization [30], [31], meta-learning [32], continual learning [33], and liveness-irrelevant feature disentanglement [8], [34]–[36]. Other studies improve generalization through data-centric or frequency-domain strategies, including physics-driven synthesis [37], diffusion-based generation [38], and frequency-shortcut suppression [39]. Multimodal methods further exploit cross-modal alignment [40], [41] and fine-grained textual guidance [42], [43] to enhance cross-scenario robustness. Beyond covariate shift, open-set [10], [44] and one-class [11], [12] methods explicitly address attacks absent from training. These approaches model real face distributions [12], synthesize or extract spoofing cues [10], [11], or learn unknown spoof prompts [45]. More recently, multimodal large language models have been introduced to improve both interpretability and generalization [46]–[50].

Overall, existing methods have made substantial progress in mitigating cross-domain covariate shift. However, most of them still rely on domain-invariant representations, seen attack distributions, or language-semantic guidance, leaving the continually evolving heterogeneous attacks and their finegrained forensic evidence insufficiently modeled in openworld scenarios.

## B. Prompt Learning in Face Anti-Spoofing

Prompt learning-based face anti-spoofing methods exploit prior knowledge encoded in vision-language models [51] to improve generalization under cross-domain and unseen-attack scenarios. Existing approaches can be broadly divided into textual and multimodal prompting.

Text prompt methods are pioneered by FLIP [13], which injects face anti-spoofing-related semantic information into visual representation learning through language guidance, thereby enhancing the generalization capability of real/spoof discriminative features. Building on this paradigm, subsequent studies have further designed fine-grained text prompts [14]–[19], [52], instance and category prompts [20], domain prompts [21], [22], and style-conditioned prompts [23] to improve the descriptive capacity of textual prompts from the perspectives of attack attributes, domain discrepancies, style variations, and local semantics. Multimodal prompt [24] methods further integrate visual, textual, and cross-modal alignment information [25], improving the consistency and generalizability of vision-language representations. In addition, several works exploit text-to-image generation to expand the training distribution [53], or unknown spoof prompt learning [54] to improve generalization to unseen attacks.

Overall, prompt learning-based methods demonstrate the value of foundation-model knowledge for generalizable face anti-spoofing. Nevertheless, most approaches still rely on textual descriptions, category semantics, or cross-modal alignment to represent spoofing patterns. Our approach is complementary: it discards the text encoder and treats learnable visual queries, refined directly against image patches, as candidate local evidence units. These evidence units are then combined by class-conditioned routing rather than being assigned fixed semantic labels. The distinction is therefore not simply text versus visual tokens; the goal is to expose an input-adaptive intermediate evidence space that can reuse local visual patterns across heterogeneous attack types.

## III. PROPOSED METHOD

## A. Problem Formulation and Overall Architecture

Open-world face anti-spoofing aims to learn a model that generalizes under concurrent covariate and semantic shifts. We consider a general setting with a single source domain $\mathcal { D } ^ { s } = \{ ( \mathbf { x } _ { i } ^ { s } , \mathbf { y } _ { i } ^ { s } ) \} _ { i = 1 } ^ { N ^ { s } }$ for training and multiple unseen target domains $\lbrace \mathcal { D } _ { m } ^ { u } \rbrace _ { m = 1 } ^ { M }$ for evaluation, where $\mathbf { x } _ { i } ^ { s }$ and $\mathbf { y } _ { i } ^ { s }$ denote the i-th training sample and its label, respectively. Let $\mathcal { Z } ^ { s }$ and $\mathcal { Z } _ { m } ^ { u }$ denote the acquisition-condition sets of the source domain and the m-th target domain, and let $\mathcal { A } ^ { s }$ and $\mathcal { A } _ { m } ^ { u }$ denote their corresponding attack-type sets. The open-world setting is characterized by

$$
\mathcal Z _ { m } ^ { u } \neq \mathcal Z ^ { s } , \qquad \mathcal A _ { m } ^ { u } \setminus \mathcal A ^ { s } \neq \emptyset , \qquad m \in \{ 1 , \dots , M \} ,\tag{1}
$$

where the first condition represents acquisition-induced covariate shift, while the second indicates that the target domain contains attack types absent during training.

Unlike methods that represent class concepts through textual prompts, we learn purely visual prompts directly in the visual feature space to capture forensic evidence across multiple granularities, from image-level physical context to localized artifacts. We formulate open-world face anti-spoofing as a compositional forensic diagnosis problem, in which discriminative evidence is organized into a shared library of learnable visual primitives and dynamically composed into inputspecific prompts. This formulation promotes cross-category reuse of local forensic evidence and enables unified detection of both seen and unseen attacks.

As illustrated in Fig. 2, the proposed framework performs visual prompt learning on a frozen ViT-based vision foundation model. We simply retain the visual encoder of pretrained CLIP as the vision foundation model, discard its text encoder, and freeze all backbone parameters during training. Since shallow layers primarily capture local textures, boundaries, and high-frequency artifacts, whereas deeper layers encode structural, geometric, and higher-level physical discrepancies, we partition the ViT backbone into L hierarchical stages and inject task-specific forensic knowledge through stage-wise compositional forensic visual prompts.

At each stage l, a compositional forensic visual prompt $\mathbf { V } ^ { l }$ is constructed through three successive operations. First, patch-aware refinement adapts learnable micro-forensic primitives to the input patches, enabling them to capture reusable and composable fine-grained evidence units. Second, classspecific global contextual prompts provide routing signals to dynamically select and compose the refined primitives into real and spoof-specific primitive evidence prompts. Finally, the resulting primitive evidence prompts are augmented with the corresponding global contextual prompts to form $\mathbf { V } ^ { l }$ . The following subsections detail the patch-aware primitive refinement and globally guided primitive composition mechanisms.

## B. Patch-Aware Micro-Forensic Primitive Refinement

Spoofing cues in open-world face anti-spoofing are highly heterogeneous in both appearance and spatial distribution. Across and within attack types, variations in materials, structures, and fabrication processes produce diverse geometric, reflective, textural, and boundary artifacts. These cues may be global or localized, salient or subtle, making attack-specific templates prone to overfitting seen categories. Nevertheless, unseen attacks often represent novel combinations of recurring forensic evidence rather than entirely new physical mechanisms. This motivates learning reusable forensic primitives and adaptively composing them for each input, offering a more generalizable alternative to attack-type modeling.

![](images/d44e586f98741f3b1f913cbef3b266f6a8187306ad05a9a281735f4ca647e363.jpg)  
Fig. 2. Overview of the proposed compositional forensic diagnosis framework for open-world face anti-spoofing. Built on a frozen ViT-based vision foundation model, the framework learns hierarchical purely visual prompts without language encoding. At each stage, micro-forensic primitives are first refined through patch-aware interactions to summarize reusable and composable fine-grained forensic units. These forensic units are then dynamically composed into classspecific primitive evidence prompts according to global priors and augmented with global contextual prompts to generate stage-wise compositional forensic visual prompts. Multi-stage prompts are finally aggregated for real/spoof discrimination. By representing heterogeneous attacks as dynamic compositions of reusable forensic evidence, the framework enables unified reasoning over diverse unseen attack types.

Accordingly, we introduce micro-forensic primitives to organize heterogeneous spoofing patterns into reusable and composable evidence units. At stage l, the learnable primitive set is defined as

$$
\mathbf { P } ^ { l } = \left\{ p _ { i } ^ { l } \right\} _ { i = 1 } ^ { N _ { P } ^ { l } } \in \mathbb { R } ^ { N _ { P } ^ { l } \times d } ,
$$

where $N _ { P } ^ { l }$ is the number of primitives and d is the token dimension. Each primitive serves as an elementary evidence probe for localized, fine-grained, and high-frequency forensic cues, such as specular anomalies, boundary discontinuities, and local texture absence.

Because such evidence is often sparsely distributed over local patches, we introduce a patch-aware attention mechanism to refine each primitive using input-dependent local evidence. Given an input image x, the patch embedding layer converts it into an initial token sequence: $\mathbf { T } ^ { 0 } = [ \mathbf { x } _ { c l s } ^ { 0 } , \mathbf { \bar { x } } _ { 1 } ^ { 0 } , \mathbf { \bar { x } } _ { 2 } ^ { 0 } , \cdot \cdot \cdot , \mathbf { x } _ { N _ { t } } ^ { 0 } ] ,$ where $\mathbf { x } _ { c l s } ^ { 0 }$ is the classification token, and $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N _ { t } }$ are the $N _ { t }$ patch tokens. Let $\mathbf { T } ^ { l } = [ \mathbf { x } _ { 1 } ^ { l } , \mathbf { x } _ { 2 } ^ { l } , \ldots , \mathbf { x } _ { N _ { t } } ^ { l } ]$ denote the patch tokens at stage l. Treating the primitives as queries and the patch tokens as keys and values, we compute the attention weight between the i-th primitive and the j-th patch as

$$
a _ { i , j } ^ { l } = \frac { \exp { \left( ( q _ { i } ^ { l } ) ^ { \top } k _ { j } ^ { l } / \sqrt { d } \right) } } { \sum _ { t = 1 } ^ { N _ { t } } \exp { \left( ( q _ { i } ^ { l } ) ^ { \top } k _ { t } ^ { l } / \sqrt { d } \right) } } ,\tag{2}
$$

where $q _ { i } ^ { l } = p _ { i } ^ { l } \mathbf { W } _ { q } , \quad k _ { j } ^ { l } = \mathbf { x } _ { j } ^ { l } \mathbf { W } _ { k } , \quad v _ { j } ^ { l } = \mathbf { x } _ { j } ^ { l } \mathbf { W } _ { v }$ . Here, $\mathbf { W } _ { q } .$ $\mathbf { W } _ { k } .$ , and $\mathbf { W } _ { v }$ are learnable projection matrices. The coefficient $a _ { i , j } ^ { l }$ measures how strongly the i-th primitive responds to the $j \mathrm { - t h }$ patch, allowing each primitive to selectively capture regions containing relevant forensic evidence.

The refined primitive is then obtained by aggregating the corresponding local evidence as

$$
\hat { p } _ { i } ^ { l } = \sum _ { j = 1 } ^ { N _ { t } } a _ { i , j } ^ { l } v _ { j } ^ { l } .\tag{3}
$$

Micro-forensic primitives are implemented as learnable vectors shared across all training samples and presentation categories. Through patch-aware attention, each primitive acts as a query to selectively retrieve and aggregate relevant evidence from local patches, encouraging it to capture localized physical patterns. During joint optimization over diverse categories and samples, recurring local patterns provide more consistent learning signals than incidental factors such as background variation and sample-specific noise. This shared learning process encourages individual primitives to develop preferences for different types of local evidence and promote complementary functional roles among them. Since the primitives represent shared candidate evidence rather than categoryspecific templates, a primitive can potentially contribute to both real and heterogeneous spoof representations, facilitating reuse across presentation categories. Such specialization and reusability are not explicitly enforced by the architecture, but are instead encouraged by the inductive biases introduced through shared parameterization, patch-aware evidence aggregation, and joint optimization across categories.

Because different stages of the vision foundation model encode distinct abstraction levels, we assign an independent primitive set to each stage. This design aligns primitive learning with the hierarchical feature space and enables multi-scale forensic modeling. Overall, patch-aware refinement avoids reliance on attack-specific semantics or fixed category templates and yields a shared library of input-refined visual evidence units for subsequent class-conditioned composition.

C. Globally Guided Primitive Routing and Compositional Inference

A presentation attack is typically characterized by multiple forensic cues, while different attacks exhibit distinct combinations of such cues. Since the primitive pool is shared across samples and presentation categories, the relevance of each primitive varies with the input: the same primitive may provide useful evidence for multiple categories, whereas different inputs require different combinations and weighting of primitives. Therefore, treating all primitives equally may dilute discriminative cues and introduce irrelevant responses, motivating input-adaptive primitive composition conditioned on global context.

We therefore use global context to condition primitive routing and composition. The global contextual prompts interact with the image tokens through the frozen backbone and are intended to summarize image-level context relevant to the real/spoof decision, including acquisition and presentation characteristics. They then provide sample-specific and classspecific conditions for weighting local evidence. At stage l, the global contextual prompt is defined as

$$
\mathbf { G } ^ { l } = [ \mathbf { g } _ { \mathrm { r e a l } } ^ { l } , \mathbf { g } _ { \mathrm { s p o o f } } ^ { l } ] ,\tag{4}
$$

where the two tokens correspond to the real and spoof classes, respectively. The global contextual prompt is initialized at the beginning of training, inserted after the CLS token and before the patch tokens, and propagated through successive backbone stages. Through self-attention, it interacts with both global and local image representations. Accordingly, $\mathbf { G } ^ { l }$ denotes its representation at stage l, rather than an independently initialized stage-specific prompt.

A lightweight routing network maps each class-specific contextual token to a distribution over the refined primitives as

$$
\begin{array} { r } { \beta ^ { l } = [ \beta _ { \mathrm { r e a l } } ^ { l } ; \beta _ { \mathrm { s p o o f } } ^ { l } ] \in \mathbb { R } ^ { 2 \times N _ { P } ^ { l } } , } \end{array}\tag{5}
$$

where

$$
\begin{array} { r l } & { \beta _ { c } ^ { l } = \mathrm { S o f t m a x } \left( \mathrm { M L P } ( \mathbf { g } _ { c } ^ { l } ) \right) , \qquad c \in \{ \mathrm { r e a l } , \mathrm { s p o o f } \} . } \end{array}\tag{6}
$$

The softmax operation is applied along the primitive dimension

$$
\boldsymbol { \beta } _ { c } ^ { l } = [ \beta _ { c , 1 } ^ { l } , \beta _ { c , 2 } ^ { l } , \ldots , \beta _ { c , N _ { P } ^ { l } } ^ { l } ] , \qquad \sum _ { i = 1 } ^ { N _ { P } ^ { l } } \boldsymbol { \beta } _ { c , i } ^ { l } = 1 ,\tag{7}
$$

where $\beta _ { c , i } ^ { l }$ quantifies the contribution of the i-th primitive to the evidence representation of class c. Each global contextual token therefore acts as a class-specific router that modulates local forensic evidence according to the holistic physical characteristics of the input. Given the refined primitive set $\hat { \mathbf P } ^ { l } = \{ \hat { p } _ { i } ^ { l } \} _ { i = 1 } ^ { N _ { P } ^ { l } }$ , the class-specific primitive evidence prompt is constructed as

$$
\mathbf { c } _ { c } ^ { l } = \sum _ { i = 1 } ^ { N _ { P } ^ { l } } \beta _ { c , i } ^ { l } \hat { p } _ { i } ^ { l } , \qquad c \in \{ \mathrm { r e a l } , \mathrm { s p o o f } \} .\tag{8}
$$

The resulting prompts are represented as

$$
\mathbf { C } ^ { l } = [ \mathbf { c } _ { \mathrm { r e a l } } ^ { l } , \mathbf { c } _ { \mathrm { s p o o f } } ^ { l } ] .\tag{9}
$$

The refined primitives summarize candidate local evidence, while the global contextual prompts determine their classspecific routing weights for the current input. Class-specific routing therefore forms distinct real- and spoof-oriented combinations from the same primitive pool. This global-to-local conditioning provides an explicit mechanism for suppressing weakly weighted responses and constructing input-adaptive evidence combinations for both seen and unseen attack types.

Finally, the dynamically composed primitive evidence prompt is further augmented with the corresponding global contextual prompt to obtain the compositional forensic visual prompt $\mathbf { V } ^ { l }$ for the stage l as

$$
\mathbf { V } ^ { l } = \mathbf { G } ^ { l } + \mathbf { C } ^ { l } .\tag{10}
$$

The resulting stage-specific prompt preserves the composition of localized forensic evidence while incorporating complementary image-level context.

## D. Compositional Prompt Aggregation and Optimization

The stage-specific compositional forensic visual prompts are aggregated across representation depths to obtain the final prompt as

$$
\mathbf { V } ^ { * } = \sum _ { l = 1 } ^ { L } \mathbf { V } ^ { l } ,\tag{11}
$$

where $\mathbf { V } ^ { * } = [ \mathbf { v } _ { \mathrm { r e a l } } ^ { * } , \mathbf { v } _ { \mathrm { s p o o f } } ^ { * } ]$ contains the final class-specific prompt representations. For each class $c ,$ the classification logit is computed by measuring the similarity between its prompt representation and the final CLS embedding $\mathbf { z } _ { \mathrm { c l s } }$ as

$$
o _ { c } = \mathrm { s i m } ( \mathbf { z } _ { \mathrm { c l s } } , \mathbf { v } _ { c } ^ { * } ) , \qquad c \in \{ \mathrm { r e a l } , \mathrm { s p o o f } \} ,\tag{12}
$$

where sim $( \cdot , \cdot )$ denotes the similarity function. To support unified discrimination across heterogeneous attacks, all presentation attack types share the binary label space $\begin{array} { r l } { \mathcal { C } } & { { } = } \end{array}$ {real, spoof}, where both seen and unseen attacks are assigned to the spoof class. The model is optimized using the cross-entropy loss as

$$
\mathcal { L } _ { \mathrm { c e } } = - \sum _ { c \in \mathcal { C } } y _ { c } \log \frac { \exp ( o _ { c } ) } { \sum _ { k \in \mathcal { C } } \exp ( o _ { k } ) } ,\tag{13}
$$

where $y _ { c }$ represents the one-hot ground-truth label.

During training, the vision foundation model remains frozen, and only the global contextual prompts, micro-forensic primitives, projection layers, and lightweight routing modules are optimized. This parameter-efficient learning strategy preserves the general visual knowledge of the pretrained backbone while adapting it to compositional forensic diagnosis. During inference, given an input image x, the framework first constructs the corresponding compositional forensic visual prompts, computes the class logits using Eq. 12, and applies softmax normalization to obtain the prediction probabilities.

Overall, the proposed framework performs prompt learning entirely in the continuous visual feature space. By dynamically combining refined primitive queries across hierarchical representation levels, it provides an input-adaptive representation that spans image-level context and localized evidence without requiring predefined attack-category descriptions. In this sense, the method implements compositional inference through explicit weighted combinations of shared visual evidence units for both seen and unseen attack types.

## IV. EXPERIMENTS

## A. Experimental Setups

Datasets. We conduct experiments on eight public benchmarks: CASIA-MFSD [55] (C), Replay-Attack [56] (I), MSU-MFSD [57] (M), OULU-NPU [58] (O), HQ-WMCA [59] (H), SiW-Mv2 [60] (W), CASIA-SURF [61] (S), and CASIA-SURF CeFA [28] (F). These datasets cover substantial variations in acquisition devices, illumination, backgrounds, attack types, providing a comprehensive evaluation of cross-domain and unseen-attack generalization in open-world scenarios.

CASIA-MFSD, Replay-Attack, MSU-MFSD, and OULU-NPU primarily contain conventional print and replay attacks. CASIA-MFSD includes warped-photo, cut-eye, and videoreplay attacks captured under different imaging qualities. Replay-Attack contains photo and video attacks presented using printed media and electronic displays under both controlled and adverse illumination. MSU-MFSD introduces variations in acquisition devices and presentation media through printed photos and replay attacks generated by mobile devices. OULU-NPU targets mobile authentication scenarios and systematically varies smartphone cameras, illumination, backgrounds, printers, and display devices. Together, these datasets provide diverse yet relatively constrained sourcedomain attack distributions.

HQ-WMCA and SiW-Mv2 contain substantially more diverse attack types and are therefore used to evaluate generalization to heterogeneous unseen attacks. HQ-WMCA is a multimodal benchmark covering visible, depth, thermal infrared, near-infrared, and short-wave infrared data, with attacks including print, replay, rigid mask, paper mask, flexible mask, mannequin, glasses, makeup, tattoo, and wig. SiW-Mv2 contains 14 attack types, spanning conventional print and replay attacks, as well as half mask, paper mask, transparent mask, silicone mask, mannequin, partial eye, funny eye glasses, partial mouth, paper glasses, cosmetic makeup, impersonation makeup, obfuscation makeup. Both datasets exhibit pronounced heterogeneity in attack appearance, physical properties, and spatial distributions.

CASIA-SURF is a large-scale multimodal dataset providing visible, depth, and near-infrared observations for each sample, with print-based presentation attacks. CASIA-SURF CeFA further introduces cross-ethnicity, multimodal, and attack-type variations, covering Asian, African, and Central Asian subjects as well as print, replay, 3D-printed mask, and silicone-mask attacks. The CASIA-SURF to CeFA setting therefore provides a large-scale benchmark involving both covariate and semantic shifts. Since this work focuses on visible-spectrum face antispoofing, only the visible modality is used for all multimodal datasets.

Evaluation protocols and metrics. We construct nine crossdataset open-world protocols. Eight protocols are formed by selecting one source domain from CASIA-MFSD, Replay-Attack, MSU-MFSD, and OULU-NPU and one target domain from HQ-WMCA and SiW-Mv2. The source datasets contain print and replay attacks, whereas the target datasets include substantially more diverse attack categories. We additionally evaluate the protocol from CASIA-SURF to CASIA-SURF CeFA to assess open-world generalization at a larger scale. These protocols introduce simultaneous discrepancies in acquisition devices, imaging conditions, and attack semantics, thereby jointly evaluating robustness to covariate and semantic shifts. Half Total Error Rate (HTER) and Area Under the ROC Curve (AUC) are adopted as the evaluation metrics.

Implementation details. The proposed method is implemented in PyTorch. We employ the pretrained CLIP ViT-L/14@336px image encoder as a frozen backbone and insert the proposed visual prompt modules at layers 6, 12, 18, and 24. Input images are resized to $2 5 9 \times 2 5 9$ , and the pretrained positional embeddings are bilinearly interpolated to match the resulting token grid. Eight micro-forensic primitives are used at each stage, with a batch size of 32. The model is optimized using AdamW with $\beta _ { 1 } ~ = ~ 0 . 9 , ~ \beta _ { 2 } ~ = ~ 0 . 9 9 9$ , a weight decay of 0.01, and an initial learning rate of $1 \times 1 0 ^ { - 3 }$ which is decayed using a cosine annealing schedule over 30 training epochs. For preprocessing, face regions are detected and cropped from video frames using the method in [62] and stored at a resolution of $1 2 8 \times 1 2 8$ . During training, random horizontal flipping with a probability of 0.5 is applied for data augmentation.

## B. Visualization Analysis of Compositional Forensic Visual Prompts

To comprehensively examine the learned compositional forensic visual prompts, we conduct visualization analyses from four perspectives. We first compare visual and textual prompts in the frequency domain to evaluate their ability to capture fine-grained spoofing artifacts, particularly highfrequency forensic cues. We then visualize the spatial responses of compositional forensic visual prompts to investigate their localization of discriminative forensic evidence. Next, layer-wise attention maps are analyzed to reveal how forensic evidence evolves across different representation levels. Finally, t-SNE visualization is employed to examine the distribution and discriminability of different prompt components in the feature space.

1) Frequency-Domain Comparison with Text Prompts: To investigate whether the proposed visual prompts capture richer high-frequency forensic evidence than text prompts, we conduct Fourier analysis under the C to W protocol. CoOp [51] is adopted as the text-prompt baseline and trained on the same source domain. For each target-domain sample, we compute the similarity between the learned prompts and the last-layer patch tokens to obtain a spatial activation map. We then calculate its two-dimensional (2D) Fourier spectrum and the corresponding one-dimensional (1D) radial energy curve, where the radius represents the spatial frequency.

![](images/3a5b8da979639f38cbe11f1d8bebf5e2de5376ee4afeda909824ed58748088cc.jpg)

![](images/09511c7ce62c3b2e3bb792548e8e8e23bd843d8101583008575e96e19e7e14b4.jpg)  
Fig. 3. Fourier analysis results of the text-prompt baseline and the proposed visual prompts under the C to W protocol. The 2D spectrum show that bot prompt types produce strong low-frequency responses, whereas the proposed visual prompts exhibit more pronounced energy expansion in the high-frequenc regions. The 1D radial energy curves further demonstrate that visual prompts maintain stronger and more stable responses in the mid- and high-frequency ranges, indicating their superior ability to capture fine-grained forensic cues and thereby enhance cross-domain unseen-attack detection.

As shown in Fig. 3, both prompt types exhibit strong responses near the spectral center, indicating comparable sensitivity to low-frequency semantics and global structures. However, the visual prompts show a broader energy distribution toward the spectral periphery, suggesting that they activate richer high-frequency responses. The radial energy curves further reveal that, although both prompts retain strong low-frequency responses, the energy of text prompts decays rapidly with increasing frequency, whereas visual prompts preserve substantially stronger mid- and high-frequency responses. These results demonstrate that the proposed visual prompts are more sensitive to fine-grained high-frequency spoofing artifacts. They also validate the effectiveness of compositional forensic visual prompts in jointly modeling global low-frequency structures and local high-frequency forensic evidence.

2) Spatial Response Visualization of Compositional Forensic Visual Prompts: To examine the spatial evidence captured by the learned prompts, we visualize their activation maps under the C to W protocol. For each sample, the activation map is generated from the similarity between the learned prompts and the last-layer patch tokens. As shown in Fig. 4, real samples mainly activate structurally informative regions, including the eyes, nose, and mouth. In contrast, spoof samples exhibit responses aligned with their attack-specific artifacts. Makeup attacks primarily activate regions with abnormal color, texture, wrinkles, or painted patterns; full-face masks produce broadly distributed responses over the facial area; and partial attacks concentrate on localized spoof regions, such as the eyes, funny-eye regions, or mouth. The consistent responses observed across samples of the same attack type qualitatively suggest that the learned prompts attend to spatially relevant evidence across heterogeneous attacks.

3) Layer-Wise Visualization of Prompt Attention: To examine the hierarchical diagnostic behavior of the proposed framework, we visualize the attention regions of visual prompts at different layers under the C to W protocol, using partialeye and partial-mouth attacks with clearly localized spoofing cues as example. For each sample, we compute the similarity between the layer-specific prompt representations and the corresponding patch tokens to generate spatial activation maps.

![](images/334c984a2b81da5ff4269594e651d0dea54f3ba208d06b4d6535850965cbef89.jpg)  
Fig. 4. Visualization of attention regions activated by the compositional forensic visual prompts under the C to W protocol. Real faces mainly activate structurally informative regions, such as the eyes, nose, and mouth, while spoof samples activate attack-specific forensic regions, including makeup artifacts, full-face mask areas, and localized eye- or mouth-level spoof traces. These results show that the learned prompts can effectively localize discriminative evidence across heterogeneous attacks.

As shown in Fig. 5, the prompt responses progressively evolve from global to local and from coarse- to fine-grained patterns. At shallow layers, the prompts exhibit broad responses over the facial region and surrounding context, primarily capturing low-level appearance and global facial structure. At intermediate layers, the activations gradually concentrate on facial boundaries and geometrically informative regions, such as the forehead and nose, indicating increased sensitivity to structural and physical inconsistencies. At deeper layers, the responses further converge on the manipulated eye or mouth regions, enabling precise localization of fine-grained spoofing artifacts.

RGB  
Layer1  
Layer2  
Layer3  
Layer4  
![](images/5500c605d66a59093aff9d1a42778141ba6320923d635c44b463ac665b34917a.jpg)  
Fig. 5. Visualization of attention regions across visual prompt layers, using partial eye and partial mouth attacks under the C to W protocol. The results show a progressive shift from global facial context to localized spoofed regions, indicating that the learned prompts guide the frozen vision foundation model toward forensic-level evidence localization.

This hierarchical evolution indicates that the learned prompts progressively guide the frozen vision foundation model from generic visual perception toward forensic-level cues localization. By progressively refining attention across layers, the model gradually suppresses task-unrelated interference and locks onto local discriminative evidence associated with unseen attacks, thereby enhancing generalization in openworld scenarios.

4) t-SNE Analysis of Different Prompt Components: To investigate the discriminative structures learned by different prompt components, we visualize the compositional forensic visual prompts, global contextual prompts, and primitive evidence prompts using t-SNE under the C to W protocol. For clarity, the 14 target-domain attacks are grouped into four categories according to their physical characteristics: 2D attacks, including print and replay; 3D attacks, including half mask, paper mask, transparent mask, silicone mask, and mannequin; partial attacks, including partial eye, funny-eye glasses, partial mouth, and paper glasses; and makeup attacks, including cosmetic, impersonation, and obfuscation makeup. We randomly sample 1,000 target-domain instances, extract their class-specific prompt embeddings, and use category prototypes to denote the corresponding feature centroids.

As shown in Fig. 6, all three types of prompts exhibit a certain degree of real/spoof discriminability. Real samples are generally closer to the real prompts, whereas spoof samples of different types are closer to the spoof prompts, indicating that the learned prompts can form discriminative directions in the feature space that are consistent with the classification objective. A further comparison shows that the real global contextual prompts of real samples are relatively dispersed, with evident overlap between real samples and 2D, 3D, and makeup attacks. Moreover, the spoof global contextual prompts of real samples surround those of makeup attacks. This suggests that relying solely on image-level global semantics remains vulnerable to interference from identity, imaging conditions, and appearance similarity.

![](images/3506c9446d44e3218d810c9f3925ffac52ed0b43b2a2c8981da7f69d7160ade9.jpg)  
(a) Global Contextual Prompts

![](images/141340e813fc46c17734483370e8c196759a41f916f2c4166ecf3c13a188c99a.jpg)  
(b) Primitive Evidence Prompts

![](images/f1f6467918eebefbe2175a2095d92cf5828ac08e2c8eb59d3d02c43d83389e71.jpg)  
(c) Compositional Forensic Visual Prompts  
Fig. 6. t-SNE visualization analysis of different prompt components.

![](images/70f2b5b0c3cf1eceaaf598fb9e4bfdc976063bb728186d227bcdcb30060e0484.jpg)  
Fig. 7. Category activation visualization of micro-forensic primitives under the C to W protocol. Different primitives exhibit clear category selectivity and produce distinct responses to both real and spoof evidence. Meanwhile, different attack types share a subset of primitives, whereas a single primitive cannot cover all spoofing patterns, indicating that heterogeneous attacks should be jointly represented by multiple reusable primitives. These results demonstrate that micro-forensic primitives provide an effective foundation for dynamic compositional modeling of unseen attacks in open-world scenarios.

In contrast, the primitive evidence prompts produce a clearer separation between real and spoof samples, indicating that micro-forensic primitives capture transferable local evidence, and the primitive evidence prompts achieve effective evidence composition. The compositional forensic visual prompts form a more compact distribution of real prompts and a clearer real/spoof decision boundary, while effectively alleviating the surrounding and entanglement phenomena observed in the global contextual prompts. These results demonstrate that the proposed composition mechanism effectively integrates local forensic evidence and global physical priors, thereby improving prompt discriminability and generalization capability in open-world scenarios with heterogeneous attacks.

## C. Visualization Analysis of Micro-Forensic Primitives

To further investigate the learned micro-forensic primitives, we conduct visualization analyses from three perspectives. We first analyze primitive activations across different presentation categories to reveal their category-dependent response patterns. We then examine how these activation patterns evolve across layers to characterize the progressive formation of forensic representations. Finally, we visualize the image patches most strongly associated with each primitive to investigate the localized visual evidence captured by different primitives.

1) Primitive Category Activation Analysis: To analyze the category selectivity of the learned primitives, we compute the average similarity between each primitive and all patch tokens from samples of a given category under the C to W protocol. The responses of each primitive are then normalized across categories using z-score normalization, producing the activation map shown in Fig. 7.

![](images/5c175a4225a6397e5615d847380b0fed68d34f1c00b9e4fda0c71cbe8f74741a.jpg)  
Fig. 8. Evolution of category activations of primitive P5 across different layers under the C to W protocol. Taking real faces, replay attacks, and silicone mask attacks as examples, the responses of P5 gradually evolve from cross-category similarity in shallow layers to clear category separation in deeper layers, indicating that micro-forensic primitives progressively develop stronger category selectivity and forensic discriminative capability as the network depth increases.

Different primitives exhibit distinct category preferences. Some primitives show strong activation for print, replay, and partial mouth attacks, suggesting that they may capture shared evidence such as 2D medium artifacts, local texture degradation, and fine-grained spoofing traces. Some primitives respond more strongly to attacks such as paper glasses, funny eye glasses, half mask, paper mask, mannequin, and makeup, indicating greater sensitivity to forensic cues related to local occlusion, mask materials, geometric discontinuities, and appearance disguise.

Several primitives also exhibit strong responses to real faces, showing that the primitive library models not only spoofing artifacts but also normal structural and textural evidence. Moreover, different attacks share partially overlapping highresponse primitives, while each attack is typically represented by multiple primitives. This observation supports the hypothesis that heterogeneous attacks can be represented by distinct combinations of shared evidence units. It also validates the effectiveness of the micro-forensic primitive library and further demonstrate the necessity of globally guided dynamic primitive composition for modeling unseen attacks in openworld scenarios.

2) Hierarchical Evolution ofPrimitive Category Activation: To examine how primitive responses evolve across representation depths, we visualize the activation of primitive P5 for real faces, replay attacks, and silicone mask attacks under the C to W protocol. As shown in Fig. 8, the three categories exhibit similar activations at Layer 1, indicating that the shallow representation of P5 mainly captures low-level patterns shared across categories and exhibits weak category selectivity. As the network depth increases, the activation differences become progressively more pronounced. At Layer 2, P5 begins to show clearer category-specific response differences. At Layer 3, the activation values of replay attacks and silicone mask attacks are relatively close to each other, yet both are clearly separated from real faces, suggesting that at this stage P5 has acquired a certain capability to distinguish real from spoof samples, although its ability to differentiate between spoof types remains limited. At Layer 4, all three categories are more distinctly separated, indicating stronger category selectivity and improved sensitivity to heterogeneous forensic patterns. These results show that micro-forensic primitives progressively evolve from generic visual responses into categoryrelevant forensic cues, validating the effectiveness of multistage primitive learning for open-world face anti-spoofing.

![](images/547557cbeed46234e34c973d5b55146febab05e897755818ab6801a081008c53.jpg)  
Fig. 9. Visualization of the top-10 similar patches for micro-forensic primitives under the C to W protocol. Red boxes mark the matched patch regions, and surrounding crops provide local context. The retrieved examples show recurring local patterns for individual primitives and different preferences across primitives, providing qualitative evidence of specialization and reuse.

3) Visualization of Primitive-Related Patches: To further understand the local evidence represented by each microforensic primitive, we retrieve and visualize its top-10 most similar patches under the C to W protocol. Patch features are extracted from the final layer of the image encoder. To prevent the retrieval results from being dominated by a single sample, the top-10 patches for each primitive are constrained to originate from different images. Because the original patches are small, we additionally display their local context, with red boxes marking the matched regions.

As shown in Fig. 9, the same primitive consistently responds to similar local patterns across different samples, such as skin textures, eye and mouth structures, boundary transitions, and reflections. Different primitives exhibit distinct visual preferences, demonstrating functional specialization and complementarity within the primitive library. These results indicate that the primitives do not encode attack templates; instead, they represent transferable local forensic units shared across samples and categories, providing the basis for compositional modeling of heterogeneous attacks.

## D. Quantitative Analysis of Primitive Routing

1) Primitive Similarity Analysis: To quantify the representational diversity of the learned micro-forensic primitives, we compute the pairwise cosine similarity among the primitives in the final layer under the C to W protocol, as shown in Fig. 10. Most primitive pairs exhibit relatively low similarity, and none shows an excessively high correlation. This suggests that the primitive library avoids evident representational collapse and severe redundancy. Instead, the primitives occupy differentiated directions in the feature space, indicating that joint optimization encourages them to capture distinct aspects of local forensic evidence. Such diversity provides a rich candidate evidence pool for subsequent adaptive routing and compositional inference.

![](images/0071b049b5ef1647f5554936f40d0309b38a71885f2dc4c07abfe24e3347d737.jpg)

Fig. 10. Pairwise cosine similarity matrix of the learned micro-forensic primitives under the C to W protocol. Most primitive pairs exhibit low similarity, indicating that the learned primitives capture diverse aspects of localized forensic evidence.  
![](images/cb298045637a80aec93bda85029cc2274afc358392a32dcf7f3aab894477a304.jpg)  
Fig. 11. Utilization of the micro-forensic primitives under the C to W protocol. All primitives remain actively utilized, while their different average routing weights reflect differentiated contributions to compositional forensic representation.

2) Primitive Utilization Analysis: To investigate how the learned micro-forensic primitives are actually utilized during test-time compositional inference, we analyze the routing weights of the last layer conditioned on the global contextual prompt associated with the predicted class of each sample under the C to W protocol. Specifically, for each test sample, we select the routing distribution corresponding to its predicted class and average the routing weights of each primitive over the entire target domain to obtain its utilization. As shown in Fig. 11, all eight primitives receive non-negligible utilization, with average weights ranging from 0.0906 to 0.1444. In particular, $P _ { 1 }$ exhibits the highest utilization, whereas $P _ { 8 }$ receives the lowest but remains actively involved in the compositional representation. The remaining primitives show relatively balanced yet differentiated utilization around the uniform reference of $1 / 8 = 0 . 1 2 5$ . These results indicate that the routing mechanism does not collapse onto a small subset of primitives; instead, the learned primitive library is broadly utilized while different primitives contribute with different strengths to forensic evidence composition.

TABLE I  
COMPARISON WITH STATE-OF-THE-ART METHODS USING SIW-MV2 (W FOR SHORT) DATASET AS THE TARGET DOMAIN, AND ONE OF CASIA-MFSD (C), REPLAY-ATTACK (I), MSU-MFSD (M), AND OULU-NPU (O) DATASET AS THE SOURCE DOMAIN. AVG REPRESENTS THE AVERAGE VALUE ACROSS THE FOUR PROTOCOLS. A LOWER HTER INDICATES BETTER PERFORMANCE, WHILE A HIGHER AUC SIGNIFIES BETTER PERFORMANCE.
<table><tr><td rowspan="2">Method</td><td colspan="2">C to W</td><td colspan="2">I to W</td><td colspan="2">M to W</td><td colspan="2">O to W</td><td colspan="2">AVG</td></tr><tr><td>HTER</td><td>AUC</td><td>HTER</td><td>AUC</td><td>HTER</td><td>AUC</td><td>HTER</td><td>AUC</td><td>HTER</td><td>AUC</td></tr><tr><td></td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td></tr><tr><td>MS-LBP [63]</td><td>43.84</td><td>52.71</td><td>54.12</td><td>44.42</td><td>54.19</td><td>46.15</td><td>42.53</td><td>54.86</td><td>48.67</td><td>49.54</td></tr><tr><td>Color texture [64]</td><td>45.10</td><td>54.80</td><td>57.47</td><td>41.80</td><td>46.84</td><td>53.97</td><td>47.17</td><td>53.55</td><td>49.15</td><td>51.03</td></tr><tr><td>CNN [65]</td><td>41.92</td><td>59.64</td><td>43.29</td><td>53.84</td><td>43.56</td><td>57.22</td><td>42.06</td><td>57.88</td><td>42.71</td><td>57.15</td></tr><tr><td>DTN [66]</td><td>45.39</td><td>55.13</td><td>47.61</td><td>52.90</td><td>38.62</td><td>65.61</td><td>42.78</td><td>61.49</td><td>43.60</td><td>58.78</td></tr><tr><td>SDTN [36]</td><td>40.73</td><td>61.89</td><td>42.91</td><td>58.21</td><td>38.14</td><td>63.97</td><td>46.39</td><td>54.33</td><td>42.04</td><td>59.60</td></tr><tr><td>DR-UDA [26]</td><td>45.08</td><td>57.43</td><td>50.22</td><td>49.77</td><td>49.18</td><td>51.25</td><td>44.35</td><td>58.93</td><td>47.21</td><td>54.35</td></tr><tr><td>USDAN [2]</td><td>35.05</td><td>69.57</td><td>38.64</td><td>64.97</td><td>37.76</td><td>64.39</td><td>35.52</td><td>69.19</td><td>36.74</td><td>67.03</td></tr><tr><td>PatchCNN [67]</td><td>35.53</td><td>67.93</td><td>46.52</td><td>53.86</td><td>34.71</td><td>68.21</td><td>35.74</td><td>69.13</td><td>38.13</td><td>64.78</td></tr><tr><td>PDEN [68]</td><td>34.41</td><td>70.15</td><td>45.62</td><td>55.45</td><td>44.61</td><td>56.65</td><td>44.28</td><td>57.43</td><td>42.23</td><td>59.92</td></tr><tr><td>LD [69]</td><td>33.96</td><td>71.01</td><td>41.37</td><td>62.11</td><td>39.07</td><td>62.96</td><td>36.08</td><td>64.92</td><td>37.62</td><td>65.25</td></tr><tr><td>IADG [7]</td><td>49.11</td><td>50.76</td><td>45.82</td><td>52.82</td><td>48.52</td><td>49.19</td><td>46.72</td><td>56.27</td><td>47.54</td><td>52.26</td></tr><tr><td>OSDG [10]</td><td>31.96</td><td>72.28</td><td>37.50</td><td>65.51</td><td>33.24</td><td>72.10</td><td>35.20</td><td>69.28</td><td>34.48</td><td>69.79</td></tr><tr><td>FoundPAD [70]</td><td>45.08</td><td>56.28</td><td>51.12</td><td>48.86</td><td>51.55</td><td>48.37</td><td>48.38</td><td>52.65</td><td>49.03</td><td>51.54</td></tr><tr><td>MEFAS [24]</td><td>13.65</td><td>94.90</td><td>41.16</td><td>63.69</td><td>35.15</td><td>70.92</td><td>36.35</td><td>68.33</td><td>31.58</td><td>74.46</td></tr><tr><td>MVPFAS [14]</td><td>27.24</td><td>81.72</td><td>32.41</td><td>74.97</td><td>54.63</td><td>43.77</td><td>26.65</td><td>79.92</td><td>35.23</td><td>70.10</td></tr><tr><td>Ours</td><td>10.14</td><td>95.26</td><td>29.40</td><td>75.99</td><td>23.86</td><td>84.21</td><td>25.63</td><td>78.31</td><td>22.26</td><td>83.44</td></tr></table>

TABLE II

COMPARISON WITH STATE-OF-THE-ART METHODS USING HQ-WMCA (H FOR SHORT) DATASET AS THE TARGET DOMAIN, AND ONE OF CASIA-MFSD (C), REPLAY-ATTACK (I), MSU-MFSD (M), AND OULU-NPU (O) DATASET AS THE SOURCE DOMAIN. AVG REPRESENTS THE AVERAGE VALUE ACROSS THE FOUR PROTOCOLS. A LOWER HTER INDICATES BETTER PERFORMANCE, WHILE A HIGHER AUC SIGNIFIES BETTER PERFORMANCE.
<table><tr><td rowspan="2">Method</td><td colspan="2">C to H</td><td colspan="2">I to H</td><td colspan="2">M to H</td><td colspan="2">O to H</td><td colspan="2">AVG</td></tr><tr><td>HTER</td><td>AUC</td><td>HTER</td><td>AUC</td><td>HTER</td><td>AUC</td><td>HTER</td><td>AUC</td><td>HTER</td><td>AUC</td></tr><tr><td></td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td><td>(%)</td></tr><tr><td>MS-LBP [63]</td><td>46.78</td><td>56.63</td><td>53.85</td><td>44.26</td><td>50.52</td><td>51.45</td><td>43.24</td><td>59.42</td><td>48.60</td><td>52.94</td></tr><tr><td>Color texture [64]</td><td>43.66</td><td>61.70</td><td>49.63</td><td>50.00</td><td>50.00</td><td>50.00</td><td>42.20</td><td>60.07</td><td>46.37</td><td>55.44</td></tr><tr><td>CNN [65]</td><td>40.95</td><td>64.67</td><td>40.33</td><td>66.27</td><td>42.31</td><td>63.87</td><td>39.04</td><td>65.36</td><td>40.66</td><td>65.04</td></tr><tr><td>DTN [66]</td><td>39.04</td><td>62.86</td><td>36.59</td><td>67.33</td><td>40.54</td><td>65.22</td><td>45.64</td><td>53.70</td><td>40.45</td><td>62.28</td></tr><tr><td>SDTN [36]</td><td>51.16</td><td>48.33</td><td>45.27</td><td>56.35</td><td>50.76</td><td>47.22</td><td>41.92</td><td>62.07</td><td>47.28</td><td>53.49</td></tr><tr><td>DR-UDA [26]</td><td>40.43</td><td>62.23</td><td>43.84</td><td>57.19</td><td>42.70</td><td>59.40</td><td>39.82</td><td>64.09</td><td>41.70</td><td>60.73</td></tr><tr><td>USDAN [2]</td><td>38.96</td><td>66.30</td><td>38.30</td><td>66.40</td><td>40.64</td><td>62.80</td><td>29.09</td><td>75.60</td><td>36.75</td><td>67.78</td></tr><tr><td>PatchCNN [67]</td><td>39.54</td><td>64.54</td><td>35.03</td><td>73.24</td><td>38.21</td><td>67.28</td><td>34.24</td><td>65.77</td><td>36.76</td><td>67.71</td></tr><tr><td>PDEN [68]</td><td>35.76</td><td>69.10</td><td>31.03</td><td>74.35</td><td>51.56</td><td>48.48</td><td>42.41</td><td>59.00</td><td>40.19</td><td>62.73</td></tr><tr><td>LD [69]</td><td>38.33</td><td>65.12</td><td>32.43</td><td>73.57</td><td>42.87</td><td>60.13</td><td>29.73</td><td>76.38</td><td>35.84</td><td>68.80</td></tr><tr><td>IADG [7]</td><td>45.91</td><td>54.19</td><td>48.75</td><td>51.08</td><td>45.30</td><td>57.02</td><td>42.95</td><td>59.90</td><td>45.73</td><td>55.55</td></tr><tr><td>OSDG [10]</td><td>27.24</td><td>78.81</td><td>28.03</td><td>79.37</td><td>34.79</td><td>71.26</td><td>27.23</td><td>80.75</td><td>29.32</td><td>77.55</td></tr><tr><td>FoundPAD [70]</td><td>47.07</td><td>53.02</td><td>45.71</td><td>53.95</td><td>52.76</td><td>46.12</td><td>48.24</td><td>51.94</td><td>48.45</td><td>51.26</td></tr><tr><td>MEFAS [24]</td><td>16.38</td><td>90.43</td><td>32.14</td><td>73.58</td><td>43.18</td><td>61.25</td><td>32.12</td><td>76.38</td><td>30.96</td><td>75.41</td></tr><tr><td>MVPFAS [14]</td><td>17.88</td><td>90.21</td><td>24.56</td><td>85.16</td><td>40.13</td><td>64.10</td><td>23.95</td><td>84.31</td><td>26.63</td><td>80.95</td></tr><tr><td>Ours</td><td>15.11</td><td>90.06</td><td>23.63</td><td>82.67</td><td>28.97</td><td>76.61</td><td>22.13</td><td>84.84</td><td>22.46</td><td>83.55</td></tr></table>

## E. Comparison with State-of-the-Art Methods

We compare the proposed method with representative face anti-spoofing approaches on nine open-world protocols, including handcrafted-feature methods, conventional deep models, domain adaptation, domain generalization methods, and foundation-model adaptation approaches. The results are reported in Tables I, II, and III. Handcrafted descriptors, such as MS-LBP and Color Texture, and conventional convolutional models exhibit substantial performance degradation when covariate and semantic shifts occur simultaneously. Domain adaptation and generalization methods, including DTN, US-DAN, PatchCNN, and OSDG, partially mitigate this degradation by learning more transferable domain-invariant representations. Methods that adapt pretrained foundation models through LoRA or prompt learning, such as FoundPAD, MEFAS, and MVPFAS, generally achieve stronger performance, highlighting the value of transferable priors learned from large-scale pretraining.

TABLE III  
COMPARISON WITH STATE-OF-THE-ART METHODS USING CASIA-SURF (S) DATASET AS THE SOURCE DOMAIN AND CEFA (F) DATASET AS THE TARGET DOMAIN. A LOWER HTER INDICATES BETTER PERFORMANCE, WHILE A HIGHER AUC SIGNIFIES BETTER PERFORMANCE.
<table><tr><td colspan="2">Method S to F</td></tr><tr><td>HTER (%)</td><td>AUC (%)</td></tr><tr><td></td><td>46.52</td></tr><tr><td>MS-LBP [63]</td><td>53.80</td></tr><tr><td>Color texture [64]</td><td>55.85 45.55</td></tr><tr><td>CNN [65]</td><td>43.74 61.07</td></tr><tr><td>DTN [66]</td><td>41.37 64.45</td></tr><tr><td>SDTN [36]</td><td>36.03 66.47</td></tr><tr><td>DR-UDA [26]</td><td>45.58 57.34</td></tr><tr><td>USDAN [2]</td><td>43.04 58.26</td></tr><tr><td>PatchCNN [67]</td><td>37.92 63.83</td></tr><tr><td>PDEN [68]</td><td>55.43 42.32</td></tr><tr><td>LD [69]</td><td>44.45 58.56</td></tr><tr><td>IADG [7]</td><td>47.70 50.11</td></tr><tr><td>OSDG [10]</td><td>33.91 71.42</td></tr><tr><td>FoundPAD [70]</td><td>44.90 58.11</td></tr><tr><td>MEFAS [24]</td><td>29.42 76.95</td></tr><tr><td>MVPFAS [14] Ours</td><td>39.90 64.98 25.31 79.44</td></tr></table>

The proposed method delivers the best or highly competitive performance across the reported protocols. In Table I, it achieves the lowest average HTER of 22.26% and the highest average AUC of 83.44%, corresponding to a 29.51% relative reduction in HTER over the second-best method, MEFAS. It likewise obtains the lowest average HTER and highest average AUC in Tables II and III. These results demonstrate that the learned compositional forensic visual prompts generalize well to simultaneous covariate and semantic shifts induced by cross-domain heterogeneous unseen attacks.

The proposed method also exhibits stronger cross-protocol consistency. In contrast, several competing methods perform well under specific settings but degrade markedly when the source distribution changes. For example, MEFAS achieves a high AUC on C to H but performs substantially worse on M to H, suggesting that prompts learned from source datasets with limited attack diversity may not adequately cover the heterogeneous attacks in the target domain. Overall, the consistent gains across protocols indicate that the proposed method better suppresses domain-specific interference and captures transferable attack-discriminative evidence, resulting in improved robustness in open-world scenarios.

## F. Ablation Study and Hyperparameter Analysis

1) Component Contribution Analysis: We evaluate the contribution of each component through the ablations reported in Table IV. The variant w/o global augmentation removes the augmentation between the global contextual and primitive evidence prompts in Equation 10, while retaining global guidance for primitive composition. The variant w/o global guide directly aggregates all refined primitives without class-specific global routing. The variant w/o all global contextual prompts removes the global contextual prompts entirely and performs classification using only the aggregated primitive evidence.

TABLE IV  
COMPONENT CONTRIBUTION ANALYSIS RESULTS UNDER THE C to W PROTOCOL.
<table><tr><td>Component</td><td>HTER (%)</td><td>AUC (%)</td></tr><tr><td>w/o primitive evidence prompts</td><td>26.72</td><td>80.72</td></tr><tr><td>w/o global augmentation</td><td>10.89</td><td>95.08</td></tr><tr><td>w/o global guild</td><td>17.84</td><td>90.29</td></tr><tr><td>w/o all global contextual prompts</td><td>50.00</td><td>50.00</td></tr><tr><td>Ours</td><td>10.14</td><td>95.26</td></tr></table>

TABLE V

COMPARISON OF THE NUMBER OF PRIMITIVES UNDER THE C to W PROTOCOL.
<table><tr><td>Number of Primitives</td><td>HTER (%)</td><td>AUC (%)</td></tr><tr><td>4</td><td>16.78</td><td>90.16</td></tr><tr><td>6</td><td>12.51</td><td>93.86</td></tr><tr><td>8</td><td>10.14</td><td>95.26</td></tr><tr><td>10</td><td>12.57</td><td>92.74</td></tr><tr><td>12</td><td>15.56</td><td>91.02</td></tr></table>

Removing the final global-context augmentation increases HTER by less than one percentage point, indicating that this residual augmentation provides additional improvement but is not the primary source of performance gains. In contrast, removing global guidance increases HTER by approximately seven percentage points. This demonstrates that different samples and attack types require adaptive selection and composition of relevant primitives, and simple aggregation is insufficient to characterize the diverse forensic evidence in heterogeneous attacks. When all global contextual prompts are removed, the model nearly loses its discriminative capability, suggesting that the primitives require class-specific global priors to form meaningful real and spoof representations. This also suggests that the primitives are inherently shared and reusable across categories rather than being directly aligned with specific classes. Removing the primitive evidence prompts increases HTER by approximately 16 percentage points, confirming that localized primitive evidence is the principal source of discriminative information. Overall, these results reveal a clear division of roles: micro-forensic primitives provide fine-grained and transferable evidence, whereas global contextual prompts supply class-specific physical priors for dynamic routing and composition.

2) Effect of the Number of Primitives: To examine the effect of primitive-library size, we vary the number of microforensic primitives from 4 to 12. As shown in Table V, performance first improves and then declines. An insufficient number of primitives limits the diversity of forensic evidence that can be represented, whereas an excessively large primitive set increases composition complexity and introduces redundant or irrelevant responses. We therefore use eight primitives, which provides the best trade-off between evidence coverage and composition stability.

3) Ablation of Layer-Specific Visual Prompts: We assess the contribution of visual prompts inserted at different stages by removing them individually. As shown in Table VI, removing the prompts at Layers 6 and 18 causes an approximately five percentage point performance drop, whereas removing those at Layers 12 and 24 results in a smaller degradation of about two percentage points. This indicates that the prompts at Layers 6 and 18 contribute more substantially to the final compositional representation. Prompts at relatively shallow layers are more effective at capturing local textures, boundaries, and high-frequency artifacts, whereas prompts at intermediate and deeper layers emphasize structural and semantic-level cues. Although the prompts at Layers 12 and 24 provide complementary information, part of their evidence may overlap with that captured at other depths. These results confirm that multi-stage prompting integrates complementary forensic evidence across the feature hierarchy.

TABLE VI  
ABLATION ANALYSIS OF VISUAL PROMPTS AT DIFFERENT LAYERS UNDER THE C to W PROTOCOL.
<table><tr><td>Component</td><td>HTER (%)</td><td>AUC (%)</td></tr><tr><td>w/o layer6</td><td>15.45</td><td>92.32</td></tr><tr><td>w/o layer12</td><td>11.89</td><td>92.93</td></tr><tr><td>w/o layer18</td><td>15.00</td><td>91.97</td></tr><tr><td>w/o layer24</td><td>11.13</td><td>94.37</td></tr><tr><td>Ours</td><td>10.14</td><td>95.26</td></tr></table>

TABLE VII

COMPARISON OF VISION FOUNDATION MODEL BACKBONES UNDER THE C to W PROTOCOL.
<table><tr><td>Backbone</td><td>HTER (%)</td><td>AUC (%)</td></tr><tr><td>ViT-B/16</td><td>23.40</td><td>83.09</td></tr><tr><td>ViT-B/32</td><td>24.69</td><td>82.59</td></tr><tr><td>ViT-L/14</td><td>13.61</td><td>92.91</td></tr><tr><td>ViT-L/14@336px</td><td>10.14</td><td>95.26</td></tr></table>

4) Effect of Different Vision Foundation Model Backbones: We evaluate the impact of different vision foundation model backbones under the C to W protocol. When using ViT-B as the backbone, the four selected layers are set to 3, 6, 9, and 12. As shown in Table VII, the proposed method consistently benefits from stronger visual representations. ViT-B/16 slightly outperforms ViT-B/32, indicating that finer patch granularity facilitates the capture of localized spoofing artifacts. Replacing the base-scale backbone with ViT-L/14 substantially reduces HTER from 23.40% to 13.61% and improves AUC from 83.09% to 92.91%, highlighting the advantage of increased model capacity for transferable forensic representation learning. ViT-L/14@336px further improves performance, achieving the lowest HTER of 10.14% and the highest AUC of 95.26%. These results show that the proposed compositional visual prompting framework can effectively leverage stronger and higher-resolution vision backbones to capture fine-grained and spatially localized forensic evidence.

5) Category-Wise Error Rate Analysis: We analyze the category-wise error rates under the C to W protocol to explore how well our method generalizes to different types of attacks, as shown in Fig. 12. Errors are concentrated in a few challenging categories. In particular, makeup obfuscation and makeup cosmetic exhibit the highest error rates, while partial paper glasses, real faces, and partial funnyeye glasses show moderate errors. These categories largely preserve natural facial appearance and introduce only subtle or spatially localized artifacts, making their forensic cues more difficult to distinguish. In contrast, most remaining attack types achieve error rates below 3%, as they exhibit more pronounced discrepancies in texture, geometry, reflectance, or presentation medium. These results demonstrate that the learned compositional forensic visual prompts provide robust discrimination across diverse presentation attacks.

![](images/ed629dd5f1ac1dc23523bfc478eba5fa10d1da24b3caaa68d40fd357252466eb.jpg)  
Fig. 12. Category-wise error rates under the C to W protocol. Errors are mainly concentrated in makeup and partial attacks with subtle or spatially localized spoofing cues, while most other attack types achieve error rates below 3%.

## V. CONCLUSION

This paper addresses the concurrent covariate and semantic shifts in open-world face anti-spoofing through a compositional forensic visual prompt learning framework. Rather than representing heterogeneous attacks with closed-set textual categories, the proposed method learns reusable forensic primitives directly in the visual feature space and dynamically routes and composes them under global-prior guidance to construct input-adaptive compositional forensic visual prompts. Extensive experiments on nine open-world protocols demonstrate that the proposed method achieves state-of-theart performance, with strong cross-domain generalization and adaptability to unseen attacks. Ablation studies and analyses in the frequency, spatial, and feature domains further show that micro-forensic primitives capture transferable and composable local evidence, while global contextual prompts provide classspecific physical priors for input-adaptive primitive composition. The resulting compositional forensic visual prompts enable unified representation and discrimination of heterogeneous seen and unseen attacks. Future work will investigate more explicit semantic interpretations of the learned primitives and extend the framework to multimodal face anti-spoofing.

## REFERENCES

[1] Z. Yu, Y. Qin, X. Li, C. Zhao, Z. Lei, and G. Zhao, “Deep learning for face anti-spoofing: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 05, pp. 5609–5631, 2023. 1

[2] Y. Jia, J. Zhang, S. Shan, and X. Chen, “Unified unsupervised and semi-supervised domain adaptation network for cross-scenario face antispoofing,” Pattern Recognition, vol. 115, p. 107888, 2021. 1, 2, 11, 12

[3] J. Yang, X. Cui, Y. Chen, S. Du, B. Chen, and Y. Liu, “Fine-grained domain alignment for face anti-spoofing with asymmetric pseudo-labels,” IEEE Transactions on Information Forensics and Security, 2026. 1, 2

[4] Z. Li, T. Zhao, X. Xu, Z. Zhang, Z. Li, X. Chen, Q. Zhang, A. Bergamo, A. K. Jain, and Y. Xing, “Optimal transport-guided source-free adaptation for face anti-spoofing,” in Computer Vision and Pattern Recognition Conference, 2025, pp. 24 351–24 363. 1, 2

[5] S. Mao, R. Chen, and H. Li, “Weighted joint distribution optimal transport based domain adaptation for cross-scenario face anti-spoofing,” International Journal of Computer Vision, vol. 133, no. 2, pp. 590–610, 2025. 1, 2

[6] Y. Jia, J. Zhang, S. Shan, and X. Chen, “Single-side domain generalization for face anti-spoofing,” in IEEE Conference on Computer Vision and Pattern Recognition, 2020, pp. 8484–8493. 1, 2

[7] Q. Zhou, K.-Y. Zhang, T. Yao, X. Lu, R. Yi, S. Ding, and L. Ma, “Instance-aware domain generalization for face anti-spoofing,” in Conference on Computer Vision and Pattern Recognition, 2023, pp. 20 453– 20 463. 1, 2, 11, 12

[8] S. Jung, K. Lee, Y. Jeong, H. Noh, J. Lee, and J. Choi, “Group-wise scaling and orthogonal decomposition for domain-invariant feature extraction in face anti-spoofing,” in International Conference on Computer Vision, 2025, pp. 13 372–13 381. 1, 2

[9] F. Jiang, Q. Li, P. Liu, X.-D. Zhou, and Z. Sun, “Adversarial learning domain-invariant conditional features for robust face anti-spoofing,” International Journal of Computer Vision, vol. 131, pp. 1680–1703, 2023. 1, 2

[10] F. Jiang, Q. Li, W. Wang, M. Ren, W. Shen, B. Liu, and Z. Sun, “Open-set single-domain generalization for robust face anti-spoofing,” International Journal of Computer Vision, vol. 132, no. 11, pp. 5151– 5172, 2024. 1, 3, 11, 12

[11] P.-K. Huang, C.-H. Chiang, T.-H. Chen, J.-X. Chong, T.-L. Liu, and C.-T. Hsu, “One-class face anti-spoofing via spoof cue map-guided feature learning,” in IEEE Conference on Computer Vision and Pattern Recognition, 2024, pp. 277–286. 1, 3

[12] K. Narayan and V. M. Patel, “Hyp-oc: Hyperbolic one class classification for face anti-spoofing,” in IEEE International Conference on Automatic Face and Gesture Recognition, 2024, pp. 1–10. 1, 3

[13] K. Srivatsan, M. Naseer, and K. Nandakumar, “Flip: Cross-domain face anti-spoofing with language guidance,” in International Conference on Computer Vision, 2023, pp. 19 685–19 696. 1, 3

[14] J. Yu, S. Kim, K. Lee, T. Kwon, W.-Y. Shin, and H. Y. Kim, “Multiview slot attention using paraphrased texts for face anti-spoofing,” in International Conference on Computer Vision, 2025, pp. 21 117–21 128. 1, 3, 11, 12

[15] X. Wang, K.-Y. Zhang, T. Yao, Q. Zhou, S. Ding, P. Dai, and R. Ji, “Tffas: twofold-element fine-grained semantic guidance for generalizable face anti-spoofing,” in European Conference on Computer Vision, 2024, pp. 148–168. 1, 3

[16] X. Hu, H. Liu, H. Yuan, Z. Fu, Y. Luo, N. Zhang, H. Zou, J. Gan, and Y. Zhang, “Fine-grained prompt learning for face anti-spoofing,” in ACM International Conference on Multimedia, 2024, pp. 7619–7628. 1, 3

[17] J. Guo, A. Liu, Y. Diao, J. Zhang, H. Ma, B. Zhao, R. Hong, and M. Wang, “Domain generalization for face anti-spoofing via contentaware composite prompt engineering,” IEEE Transactions on Multimedia, 2025. 1, 3

[18] J. Zhang, J. Zhang, D. Yang, R. Li, and Z. Li, “Long-fas: Crossdomain face anti-spoofing with long text guidance,” Image and Vision Computing, p. 105901, 2026. 1, 3

[19] H. Zhang, X. Zhu, L. Gao, A. Liu, S. Peng, and Z. Lei, “Cpg-pad: Concept-informed prompts guided presentation attack detection,” arXiv preprint arXiv:2607.01303, 2026. 1, 3

[20] A. Liu, X. Lin, H. Ma, X. Yu, J. Guo, Z. Yu, J. Wan, Z. Cai, Z. Lei, and Y. Liang, “Icpe-fas: Instance and category prompts engineering for generalizable face anti-spoofing: A. liu et al.” International Journal of Computer Vision, vol. 134, no. 6, p. 292, 2026. 1, 3

[21] A. Liu, X. Lin, R. Zhi, Y. Liang, X. Zhu, Z. Cai, J. Wan, S. Escalera, and Z. Lei, “Dgpdl: Domain-guided prompt distribution learning for generalizable face anti-spoofing,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026. 1, 3

[22] S.-Q. Liu, Q. Wang, and P. C. Yuen, “Bottom-up domain prompt tuning for generalized face anti-spoofing,” in European Conference on Computer Vision, 2024, pp. 170–187. 1, 3

[23] J. Guo, H. Liu, Y. Luo, X. Hu, H. Zou, Y. Zhang, H. Liu, and B. Zhao, “Style-conditional prompt token learning for generalizable face antispoofing,” in ACM International Conference on Multimedia, 2024, pp. 994–1003. 1, 3

[24] L. Cai, H. Wang, J. Ji, X. Sun, L. Cao, and R. Ji, “Me-fas: Multimodal text enhancement for cross-domain face anti-spoofing,” IEEE Transactions on Information Forensics and Security, 2025. 1, 3, 11, 12

[25] H. Ma, A. Liu, S. Xue, Y. Diao, D. Liang, H. Liang, Y. Liang, J. Wan, X. Yang, Z. Hu et al., “Clip-sa: Clip-guided semantic alignment for generalizable face anti-spoofing,” IEEE Transactions on Multimedia, 2026. 1, 3

[26] G. Wang, H. Han, S. Shan, and X. Chen, “Unsupervised adversarial domain adaptation for cross-domain face presentation attack detection,” IEEE Transactions on Information Forensics and Security, vol. 16, pp. 56–69, 2020. 2, 11, 12

[27] P.-K. Huang, C.-Y. Lu, S.-J. Chang, J.-X. Chong, and C.-T. Hsu, “Testtime adaptation for robust face anti-spoofing.” in British Machine Vision Conference, 2023, pp. 379–380. 2

[28] A. Liu, Z. Tan, J. Wan, S. Escalera, G. Guo, and S. Z. Li, “Casia-surf cefa: A benchmark for multi-modal cross-ethnicity face anti-spoofing,” in IEEE Winter Conference on Applications of Computer Vision, 2021, pp. 1179–1187. 2, 6

[29] Z. Kong, W. Zhang, T. Wang, K. Zhang, Y. Li, X. Tang, and W. Luo, “Dual teacher knowledge distillation with domain alignment for face anti-spoofing,” IEEE Transactions on Circuits and Systems for Video Technology, 2024. 2

[30] Y. Liu, Z. Li, and L. Wu, “Dual consistency regularization for generalized face anti-spoofing,” IEEE Transactions on Information Forensics and Security, 2025. 2

[31] B. M. Le and S. S. Woo, “Gradient alignment for cross-domain face anti-spoofing,” in IEEE Conference on Computer Vision and Pattern Recognition, 2024, pp. 188–199. 2

[32] Y. Jia, J. Zhang, and S. Shan, “Dual-branch meta-learning network with distribution alignment for face anti-spoofing,” IEEE Transactions on Information Forensics and Security, vol. 17, pp. 138–151, 2021. 2

[33] R. Cai, Y. Cui, Z. Yu, X. Lin, C. Chen, and A. Kot, “Rehearsal-free and efficient continual learning for cross-domain face anti-spoofing,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 2

[34] J. Yang, Z. Yu, X. Ni, J. He, and H. Li, “Generalized face anti-spoofing via finer domain partition and disentangling liveness-irrelevant factors,” in European Conference on Artificial Intelligence, 2024, pp. 274–281. 2

[35] Y. Ma, J. Qian, J. Li, and J. Yang, “Dual feature disentanglement for face anti-spoofing,” Pattern Recognition, vol. 155, p. 110656, 2024. 2

[36] Y. Liu, J. Stehouwer, and X. Liu, “On disentangling spoof trace for generic face anti-spoofing,” in European Conference on Computer Vision, 2020, pp. 406–422. 2, 11, 12

[37] R. Cai, C. Soh, Z. Yu, H. Li, W. Yang, and A. C. Kot, “Towards datacentric face anti-spoofing: Improving cross-domain generalization via physics-based data synthesis,” International Journal of Computer Vision, pp. 1–22, 2024. 2

[38] X. Ge, X. Liu, Z. Yu, J. Shi, C. Qi, J. Li, and H. Kalvi¨ ainen, “Diff-¨ fas: face anti-spoofing via generative diffusion models,” in European Conference on Computer Vision, 2024, pp. 144–161. 2

[39] J. Cao and C. Ma, “Towards generalized face anti-spoofing from a frequency shortcut view,” in Winter Conference on Applications of Computer Vision, 2025, pp. 1005–1015. 2

[40] Y. Ma, X. Lin, Z. Yu, H. Wang, R. Zhang, S. Ding, X. Liu, X. Yuan, W. Xie, and L. Shen, “Purify then guide: Rethinking domain generalization for multimodal face anti-spoofing,” arXiv preprint arXiv:2505.09484, 2026. 2

[41] X. Lin, A. Liu, Z. Yu, R. Cai, S. Wang, Y. Yu, J. Wan, Z. Lei, X. Cao, and A. Kot, “Reliable and balanced transfer learning for generalized multimodal face anti-spoofing,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 2

[42] X. Chen, Y. Xie, H. Ma, N. Li, Y. Liang, and J. Guo, “Crossgap: Unified face anti-spoofing via cross-modal global-aware prompting,” in International Conference on Computer Vision, 2025, pp. 3208–3215. 2

[43] D. Li, Z. Yu, J. Hu, G. Chen, J. Zeng, and M. Tan, “Fine-grained textual guidance for generalized multi-modal face anti-spoofing,” IEEE Transactions on Information Forensics and Security, 2025. 2

[44] X. Dong, H. Liu, W. Cai, P. Lv, and Z. Yu, “Open set face anti-spoofing in unseen attacks,” in ACM International Conference on Multimedia, 2021, pp. 4082–4090. 3

[45] F. Jiang, Q. Li, W. Wang, W. Shen, B. Liu, and Z. Sun, “Learning unknown spoof prompts for generalized face anti-spoofing using only real face images: F. jiang et al.” International Journal of Computer Vision, vol. 134, no. 5, p. 250, 2026. 3

[46] H. Zhang, K. Wang, G. Zhang, H. Yue, Z. Tan, S. Peng, T. Zhang, X. Tan, K. Chen, W. He et al., “From intuition to investigation: A

tool-augmented reasoning mllm framework for generalizable face antispoofing,” in Conference on Computer Vision and Pattern Recognition, 2026, pp. 40 855–40 865. 3

[47] H. Zhang, Z. Fang, N. Zhao, S. Hou, L. Ma, R. Pei, and Z. He, “Harnessing chain-of-thought reasoning in multimodal large language models for face anti-spoofing,” in Conference on Computer Vision and Pattern Recognition, 2026, pp. 33 566–33 576. 3

[48] G. Zhang, K. Wang, H. Yue, A. Liu, G. Zhang, K. Yao, E. Ding, and J. Wang, “Interpretable face anti-spoofing: Enhancing generalization with multimodal large language models,” in AAAI Conference on Artificial Intelligence, vol. 39, no. 9, 2025, pp. 9896–9904. 3

[49] Y. Ma, X. Lin, Y. Xu, W. Xie, and Z. Yu, “Pa-fas: Towards interpretable and generalizable multimodal face anti-spoofing via path-augmented reinforcement learning,” in AAAI Conference on Artificial Intelligence, vol. 40, no. 10, 2026, pp. 7856–7864. 3

[50] H. Wang, Y. Shi, Z. Tao, Y. Gao, L. Zhang, X. Lin, J. Feng, X. Yuan, Z. Yu, and X. Cao, “Faceshield: Explainable face anti-spoofing with multimodal large language models,” in AAAI Conference on Artificial Intelligence, vol. 40, no. 12, 2026, pp. 9811–9819. 3

[51] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Learning to prompt for visionlanguage models,” International Journal of Computer Vision, vol. 130, no. 9, pp. 2337–2348, 2022. 3, 6

[52] F. Jiang, Q. Li, B. Liu, W. Wang, C. Shan, Z. Sun, and M.-H. Yang, “Learning knowledge-based prompts for robust 3d mask presentation attack detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 2, pp. 1321–1338, 2026. 3

[53] N. Ko, Y. Jeong, and J. C. Ye, “Text-to-image synthesis for domain generalization in face anti-spoofing,” in Winter Conference on Applications of Computer Vision, 2025, pp. 1850–1860. 3

[54] P.-K. Huang, J.-X. Chong, C.-H. Chiang, T.-H. Chen, T.-L. Liu, and C.- T. Hsu, “Slip: Spoof-aware one-class face anti-spoofing with language image pretraining,” in Association for the Advancement of Artificial Intelligence, vol. 39, 2025, pp. 3697–3706. 3

[55] Z. Zhang, J. Yan, S. Liu, Z. Lei, D. Yi, and S. Z. Li, “A face antispoofing database with diverse attacks,” in International Conference on Biometrics, 2012, pp. 26–31. 6

[56] I. Chingovska, A. Anjos, and S. Marcel, “On the effectiveness of local binary patterns in face anti-spoofing,” in International Conference of Biometrics Special Interest Group, 2012, pp. 1–7. 6

[57] D. Wen, H. Han, and A. K. Jain, “Face spoof detection with image distortion analysis,” IEEE Transactions on Information Forensics and Security, vol. 10, no. 4, pp. 746–761, 2015. 6

[58] Z. Boulkenafet, J. Komulainen, L. Li, X. Feng, and A. Hadid, “Oulu-npu: A mobile face presentation attack database with real-world variations,” in IEEE International Conference on Automatic Face and Gesture Recognition, 2017, pp. 612–618. 6

[59] G. Heusch, A. George, D. Geissbuhler, Z. Mostaani, and S. Marcel,¨ “Deep models and shortwave infrared information to detect face presentation attacks,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 2, no. 4, pp. 399–409, 2020. 6

[60] X. Guo, Y. Liu, A. Jain, and X. Liu, “Multi-domain learning for updating face anti-spoofing models,” in European Conference on Computer Vision, 2022, pp. 230–249. 6

[61] S. Zhang, A. Liu, J. Wan, Y. Liang, G. Guo, S. Escalera, H. J. Escalante, and S. Z. Li, “Casia-surf: A large-scale multi-modal benchmark for face anti-spoofing,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 2, no. 2, pp. 182–193, 2020. 6

[62] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint face detection and alignment using multitask cascaded convolutional networks,” IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499–1503, 2016. 6

[63] J. Ma¨att¨ a, A. Hadid, and M. Pietik¨ ainen, “Face spoofing detection from¨ single images using micro-texture analysis,” in IEEE International Joint Conference on Biometrics, 2011, pp. 1–7. 11, 12

[64] Z. Boulkenafet, J. Komulainen, and A. Hadid, “Face spoofing detection using colour texture analysis,” IEEE Transactions on Information Forensics and Security, vol. 11, no. 8, pp. 1818–1830, 2016. 11, 12

[65] J. Yang, Z. Lei, and S. Z. Li, “Learn convolutional neural network for face anti-spoofing,” arXiv preprint arXiv:1408.5601, 2014. 11, 12

[66] Y. Liu, J. Stehouwer, A. Jourabloo, and X. Liu, “Deep tree learning for zero-shot face anti-spoofing,” in IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 4680–4689. 11, 12

[67] C.-Y. Wang, Y.-D. Lu, S.-T. Yang, and S.-H. Lai, “Patchnet: A simple face anti-spoofing framework via fine-grained patch recognition,” in Conference on Computer Vision and Pattern Recognition, 2022, pp. 20 281–20 290. 11, 12

[68] L. Li, K. Gao, J. Cao, Z. Huang, Y. Weng, X. Mi, Z. Yu, X. Li, and B. Xia, “Progressive domain expansion network for single domain generalization,” in Conference on Computer Vision and Pattern Recognition, 2021, pp. 224–233. 11, 12

[69] Z. Wang, Y. Luo, R. Qiu, Z. Huang, and M. Baktashmotlagh, “Learning to diversify for single domain generalization,” in International Conference on Computer Vision, 2021, pp. 834–843. 11, 12

[70] G. Ozgur, E. Caldeira, T. Chettaoui, F. Boutros, R. Ramachandra, and N. Damer, “Foundpad: Foundation models reloaded for face presentation attack detection,” in Winter Conference on Applications of Computer Vision, 2025, pp. 745–755. 11, 12

![](images/f7e06776ab996424ef3424c330d8841a6fdb8ddefee4e31d376dab6a2c54947c.jpg)

Fangling Jiang received the B.E. and M.S. degrees from Tianjin University in 2009 and 2012, respectively, and the Ph.D. degree from University of Chinese Academy of Sciences in 2021. She is an Associate Professor with The First Affiliated Hospital and the School of Computer Science, University of South China. Her research interests include face anti-spoofing, transfer learning, and computer vision.

![](images/b29c2a5e5a430c84047e1dacdc76909c214d1499ab227dd7a04f4b8a2f9a3b1e.jpg)

Qi Li received the B.E. degree from China University of Petroleum in 2011, the Ph.D. degree from the Institute of Automation, Chinese Academy of Sciences (CASIA) in 2016. He is an Associate Professor with the New Laboratory of Pattern Recognition (NLPR), State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS), CASIA. His research interests include face recognition, computer vision, and machine learning.

![](images/1d17967edda859b685a73165cc9baf2f063747843922525cde1451e025478a8c.jpg)

Bing Liu is a Professor of Computer Science and Technology at University of South China. Liu serves as a project head of Industry Academia Collaborative Education of the Ministry of Education in 2024. He is the executive director of the Hunan Provincial University Network Association. His research interests include computer vision and machine learning.

![](images/2624dda5da41117bbaf986f229db14e6452bb36ae1a4fd246739121490a5961c.jpg)

Weining Wang received her B.E. degree from North China Electric Power University in 2015 and the Ph.D. degree from University of Chinese Academy of Sciences (UCAS) in 2020. She is now an Associate Professor at the Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences (CA-SIA). Her research interests include pattern recognition and computer vision.

![](images/0495d09c3ac4f617bd7809ddddaee42da1f8918b15385904e26066fbd2215014.jpg)

Qiulin Huang received the B.E. degree from Hengyang Medical College in 1996 ,and the Ph.D. degree from Central South University in 2004. He is an Professor of The First Affiliated Hospital, University of South China. His research interests include pattern recognition and computer vision.

![](images/136bd868c1262d14abef01491cfc2be173bd31ca03365de1931468c3e512db12.jpg)

Zhenan Sun received the Ph.D. degree from the Institute of Automation, Chinese Academy of Sciences (CASIA) in 2006. He is a professor at New Laboratory of Pattern Recognition (NLPR), State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS), CASIA. His current research interests include biometrics, pattern recognition, and computer vision. He is a fellow of the IAPR, and an Associate Editor of the IEEE Transactions on Biometrics, Behavior, and Identity Science.

![](images/88c15b40fba68932d706c3fd6c49badc11c43a77f5fe72c367311b92b2c43124.jpg)

Ming-Hsuan Yang is a Professor of Electrical Engineering and Computer Science at University of California, Merced. Yang serves as a program cochair of IEEE International Conference on Computer Vision (ICCV) in 2019. He received the Best Paper Award at ICML 2024, Longuet-Higgins Prize at CVPR 2023, NSF CAREER award in 2012 and Google Faculty Award in 2009. He is a Fellow of the IEEE and ACM.