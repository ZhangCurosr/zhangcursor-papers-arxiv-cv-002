# Towards Text-Only Remote Sensing Scene Classification via Semantic Proxy

Xianping Ma, Kunlin Wu, Xiaofeng Liu, Xiaokang Zhang, Yuanxin Ye<sup>∗</sup>

Abstract—Remote sensing scene classification (RSSC) is traditionally formulated as a vision-centric problem, where inference relies on rich visual representations extracted from images. In this work, we revisit this assumption by introducing text-only RSSC, a new problem setting that investigates whether remote sensing scenes can be recognized using only textual semantic representations. To enable this study, we propose SeProRS, which reformulates RSSC as a text-only classification task through semantic proxy generation. Specifically, a vision-language model (VLM) is first employed to convert remote sensing images into textual descriptions, termed semantic proxies, which are then encoded by a text embedding model for scene classification. This decoupled pipeline provides a unified and reproducible benchmark for evaluating text-only scene understanding. Furthermore, we investigate representative prompting strategies and proxy generators, and analyze their impact on classification performance, text embedding behavior, cross-generator transferability, and efficiency. Extensive experiments on three wellknown RSSC datasets show that semantic proxies can preserve substantial discriminative information, while also revealing inherent limitations of purely text-based scene understanding. This paper does not aim to replace image-based RSSC, but instead establishes a new research direction by quantifying the feasibility and boundaries of semantic proxy-based scene recognition. The proposed benchmark and analysis provide a foundation for future research on efficient and modality-reduced remote sensing scene understanding. The source code and datasets will be accessible at https://github.com/sstary/SSRS.

Index Terms—Remote Sensing Scene Classification, Text-Only, Semantic Proxy, Systematic Benchmark

## I. INTRODUCTION

Remote sensing scene classification (RSSC) aims to assign semantic land-use or land-cover labels to aerial or satellite image patches, and has long served as a fundamental task in remote sensing image understanding. Representative benchmarks, such as UCM [1] and AID [2], have promoted early progress by providing diverse scene categories and high-resolution image samples. Later datasets, such as NWPU-RESISC45 [3], further increased the number of categories and image variations, making RSSC a more challenging benchmark for evaluating scene-level visual representation learning [4]. Driven by these benchmarks, RSSC has evolved from handcrafted descriptors and CNN-based models to Transformer-based architectures that better capture long-range spatial dependencies and contextual scene patterns [5]. Despite continuous architectural advances, most existing RSSC methods share the same basic assumption: the target scene must be represented by visual features extracted from the input image.

![](images/1f477846eed44f52b70abad7e18bde2235ce3403425eb53ac33ddd860508108c.jpg)  
Fig. 1. Comparison of three RSSC paradigms. (a) Conventional RSSC relies on image-derived visual features for scene classification, while (b) visionlanguage RSSC uses textual cues to assist visual recognition. (c) The proposed text-only RSSC instead represents remote sensing scenes with textual semantic proxies and predicts scene categories from text representations alone. Representative application scenarios are included to illustrate the practical relevance of this setting.

This vision-centric paradigm has been further strengthened by the emergence of foundation models in remote sensing. Large-scale visual pretraining has enabled remote sensing models to learn transferable representations from massive unlabeled or weakly labeled imagery, improving their adapt ability across sensors, resolutions, and downstream tasks [6, 7]. Domain-specific foundation models, such as RingMo [8] and DOFA [9], further demonstrate the potential of largescale pretraining for capturing generic remote sensing visual patterns. Recent surveys also indicate that foundation models are becoming a central paradigm in remote sensing image interpretation, covering visual, multimodal, and promptable model families [10, 11]. However, these models still primarily improve the quality of visual representations, while the final prediction remains dependent on image pixels or imagederived features.

Recently, vision-language learning has introduced textual semantics into remote sensing interpretation. RS-CLIP [12] and RemoteCLIP [13] adapt contrastive vision-language learning to remote sensing and show promising performance in zero-shot scene classification and image-text retrieval. Large-scale image-text datasets and aligned VLMs, such as RS5M/GeoRSCLIP [14] further expand the semantic supervision available for remote sensing representation learning. More recent studies explore prompt learning and transductive inference to improve zero-shot RSSC under vision-language settings [15, 16]. Meanwhile, remote sensing multimodal large language models, such as RS-LLaVA [17] and SkyEyeGPT [18], extend vision-language interpretation toward captioning, visual question answering, grounding, and instructionfollowing analysis. These developments show that language can provide rich semantic cues for remote sensing scenes, but they still generally use text to assist visual recognition rather than to replace visual inputs.

This observation raises an underexplored question: Can textual semantic representations serve as partial substitutes for visual inputs in remote sensing scene understanding? As a first step, this paper instantiates this question on RSSC and examines whether scene categories can be inferred from textual semantic proxies alone. Unlike conventional visionlanguage RSSC, where text typically serves as a class prompt, auxiliary description, or alignment target while visual features remain indispensable [12, 15], our setting treats text as the only classification input. For RSSC, this question is particularly meaningful because scene categories are often defined not only by object presence, but also by object co-occurrence, spatial layout, density, scale, and functional context. For example, airport, commercial area, dense residential area, and industrial area can share common objects such as buildings and roads, but differ substantially in spatial organization and land-use semantics [2, 3]. Therefore, text-only RSSC requires more than class-name prompting or overly generic descriptions. The textual representation should preserve task-relevant visual evidence as much as possible, enabling downstream classification to infer scene categories from sufficient semantic cues. Such semantic proxies provide compact and interpretable scenelevel information, making this paradigm potentially useful when direct image access, transmission, or deployment is constrained. Fig. 1 compares conventional RSSC, vision-language RSSC, and the proposed text-only RSSC setting, while also highlighting several representative application scenarios.

Building upon this motivation, we take a first step toward this unexplored direction by introducing text-only RSSC, a new problem setting that investigates whether remote sensing scenes can be recognized from textual semantic representations alone. Rather than extending existing vision-language RSSC methods that still rely on visual inputs for final prediction, this work reformulates RSSC as a text-only semantic reasoning problem, where the visual scene is first converted into a textual semantic proxy and the final prediction is made solely from text representations. To establish a reproducible baseline for this new task, we propose SeProRS, a foundation model-driven semantic proxy framework consisting of two key stages: semantic proxy generation and text-based scene classification. Specifically, a VLM is employed to generate scene-level textual proxies from remote sensing images, while a text embedding model is used to encode these proxies into discriminative semantic representations for classification. This pipeline enables the feasibility of text-only RSSC to be systematically examined under a unified framework. Furthermore, we investigate two representative prompting strategies for semantic proxy generation, aiming to reveal how different forms of textual descriptions affect the behavior of text-only RSSC. Through this work, we do not claim to replace imagebased RSSC, but instead establish a new research direction and provide a foundational benchmark for understanding how far textual semantic proxies can support RSSC when classification relies only on textual information. The main contributions of this work are summarized as follows:

• We introduce text-only remote sensing scene classification, a new RSSC setting that explores scene recognition using textual semantic representations as the only classification input;

• We propose SeProRS, a foundation model-driven baseline that performs text-only RSSC through VLM-based semantic proxy generation and text embedding-based classification;

• We establish a systematic benchmark for text-only RSSC using three standard datasets, three representative VLMs, and two typical prompting strategies, revealing the feasibility and limitations of semantic proxy-based scene recognition.

To facilitate reproducible research and future comparison, we will release the generated semantic proxies on three standard RSSC datasets, establishing a reusable text-only resource for semantic proxy-based remote sensing recognition.

## II. RELATED WORK

## A. Remote Sensing Scene Classification

RSSC methods mainly focus on learning discriminative visual representations from remote sensing images. Since scene categories are usually determined by complex combinations of objects, textures, spatial layouts, and contextual patterns, effective RSSC requires representations that can capture both local visual details and global scene structures. Early deep learning methods addressed this problem by transferring or fine-tuning CNNs pretrained on natural images, showing that hierarchical visual features are effective for remote sensing scene recognition [19]. Subsequent studies further improved visual representation learning to enhance the separability of visually similar scene categories [4]. Therefore, the mainstream development of RSSC has been centered on strengthening image-derived visual features.

CNN-based methods first dominated the visual representation-driven paradigm in RSSC. By learning hierarchical features from image pixels, deep convolutional models provide stronger scene representations than handcrafted descriptors and improve the recognition of complex land-use categories [20]. Later studies further enhanced CNN representations through multi-scale feature extraction, attention mechanisms, and contextual modeling, aiming to capture local texture, object composition, and category-specific spatial patterns more effectively [21, 22]. Transformer-based methods then introduced global self-attention into RSSC, enabling long-range dependency modeling and more flexible scene-structure representation [5, 23]. Hybrid CNN-Transformer designs further combine local inductive bias with global modeling ability, reflecting a continued effort to improve image-derived visual representations for scene classification [24].

Foundation models have pushed this visual paradigm further by transferring large-scale pretrained representations to remote sensing tasks [10, 25]. More recent efforts, including DINOv3 with satellite pretraining [26], SkySense++ [27], and AlphaEarth Foundations [28], further demonstrate the trend toward larger-scale visual or geospatial representation learning for Earth observation. These developments significantly strengthen the visual backbone of RSSC. However, existing RSSC methods remain fundamentally vision-centric. The role of non-visual semantic representations as independent carriers for scene classification has not been sufficiently explored.

## B. Vision-Language Learning in Remote Sensing

Vision-language learning has become an important direction in remote sensing interpretation. Early remote sensing image captioning datasets, such as RSICD [29] and RSITMD [30], demonstrate that aerial images can be associated with natural language descriptions containing objects, attributes, and contextual relations. More recent large-scale image-text datasets further expand remote sensing vision-language learning by collecting or generating large amounts of paired satellite images and textual descriptions [31]. These datasets provide a foundation for learning cross-modal alignment between remote sensing imagery and language.

These image-text resources have supported the rapid evolution of remote sensing vision-language learning [32, 33]. Recently, LRSCLIP [34] extends long-text vision-language modeling to remote sensing and improves cross-modal retrieval and zero-shot recognition under more descriptive textual supervision. LHRS-Bot-Nova [35] enhances vision-language alignment through improved visual encoding, visual compression, and remote sensing-oriented instruction data. These efforts further explore reliable instruction generation and hierarchical task modeling for remote sensing VLMs, such as DDFAV [36] and GeoRSMLLM [37].

Although these studies demonstrate the growing importance of language in remote sensing, their primary goal is still to enhance visual interpretation. Text is typically used as supervision, prompt, instruction, or response, while visual encoders and image features remain indispensable. Therefore, existing remote sensing vision-language studies have not systematically examined whether textual semantic representations, once generated from remote sensing images, can serve as independent inputs for remote sensing scene understanding.

## III. TEXT-ONLY REMOTE SENSING SCENE CLASSIFICATION

## A. Problem Formulation

Conventional RSSC is typically formulated as an imagebased recognition problem. Given a remote sensing image $x \in \mathcal { X }$ and its corresponding scene label $y ~ \in ~ \mathcal { V }$ , where $y = 1 , \ldots , C$ denotes a predefined set of C scene categories, the goal is to learn a visual classifier:

$$
f _ { v } : \mathcal { X } \to \mathcal { Y } ,\tag{1}
$$

which predicts the scene category from image pixels or imagederived visual features. Under this formulation, inference depends on visual evidence extracted from x.

![](images/ec1af5abc66d4c848731a68445ca4769d80c2812770c4d753416f84197c47b9f.jpg)  
Fig. 2. Overall framework of SeProRS. Remote sensing images are first converted into textual semantic proxies by VLMs under task-agnostic descriptive prompts, and the downstream model performs scene recognition using only the generated proxy. The generated proxies are stored and reused for all textonly training and testing, which decouples semantic abstraction from classifier optimization.

In contrast, this work considers text-only remote sensing scene classification, which represents a fundamentally different setting. Instead of directly classifying an image, this setting assumes that each scene is represented by a textual semantic proxy $s \in S ,$ , where S denotes the space of naturallanguage scene descriptions. The classifier then learns a textbased mapping:

$$
f _ { t } : S  \mathcal { V } ,\tag{2}
$$

and predicts the scene category using only s as input. This setting does not imply that images are absent from the entire pipeline; rather, it specifies that the final prediction is made from textual representations without accessing image pixels or visual features.

More formally, text-only RSSC can be viewed as a two-step abstraction-and-recognition problem. A semantic abstraction function $\phi$ converts an image into a textual proxy:

$$
s = \phi ( x ) ,\tag{3}
$$

and a text-based classifier $f _ { t }$ predicts the label from this proxy:

$$
\hat { y } = f _ { t } ( s ) = f _ { t } ( \phi ( x ) ) .\tag{4}
$$

The key distinction from conventional RSSC lies in the information available to the classifier: while $f _ { v }$ directly accesses visual evidence from $x , f _ { t }$ only receives the textual abstraction s.

## B. Semantic Proxy

In the text-only RSSC setting, the semantic proxy serves as the intermediate representation that connects the original visual scene to text-based classification. It is generated for a specific remote sensing image and used as the classification input. Importantly, the term semantic proxy does not refer to a fixed linguistic format. It may be expressed as a freeform scene description or as a structured textual description, depending on the proxy-generation strategy. What makes it a proxy is its functional role: it carries scene-relevant semantic evidence in textual form and replaces direct visual inputs during classification. As summarized in Table I, the semantic proxy is therefore distinguished by both its granularity and its role in the task.

TABLE I  
COMPARISON OF DIFFERENT TEXTUAL REPRESENTATIONS RELATED TO RSSC.
<table><tr><td>Representation</td><td>Granularity</td><td>Image-specific</td><td>Main Function</td></tr><tr><td>Class label</td><td>Category-level</td><td>No</td><td>Defines the prediction target</td></tr><tr><td>Class prompt</td><td>Category-level</td><td>No</td><td>Provides textual class prior</td></tr><tr><td>Generic description</td><td>Instance-level</td><td>Yes</td><td>Describes visible scene content</td></tr><tr><td>Semantic proxy</td><td>Instance-level</td><td>Yes</td><td>Preserves task-relevant evidence for text-only classification</td></tr></table>

For example, rather than simply associating an image with the word “airport”, a proxy may describe evidence such as runways, large paved areas, terminal-like buildings, and surrounding transportation structures. Such textual evidence provides a basis for predicting the scene category from semantic cues rather than from raw visual features. The quality of a semantic proxy is therefore determined by its usefulness for downstream scene recognition, not by whether it follows a specific textual style. A fluent caption-like description, a concise summary, or a structured semantic description can all serve as semantic proxies if they preserve discriminative scene information for classification.

## C. Scope and Implications

Text-only RSSC should be interpreted as an initial exploration of semantic proxy-based remote sensing understanding, rather than as a direct replacement for image-based RSSC. In this work, RSSC is selected as a representative scene-level task because its prediction depends on high-level land-use semantics, object co-occurrence, spatial organization, and contextual cues. These properties make RSSC a suitable testbed for examining whether textual semantic representations can partially substitute visual inputs for scene-level recognition.

This setting also defines a different evaluation perspective from conventional visual recognition. Since the classifier only accesses textual proxies, its performance reflects the information-preserving capacity of semantic abstraction and the discriminability of text-based scene representations. Therefore, the results should not be interpreted as a direct upper bound of image-based RSSC, where richer visual evidence is available. Instead, they provide evidence about the feasibility and limitations of performing remote sensing recognition under a modality-reduced condition.

Beyond RSSC, this formulation points to a broader research direction. Many remote sensing applications involve semanticlevel decisions, where compact textual representations may support tasks such as efficient transmission, privacy-sensitive sharing, and lightweight deployment. In this sense, text-only RSSC provides a concrete benchmark for studying how far semantic proxies can support modality-reduced remote sensing understanding.

## IV. SEPRORS: A SEMANTIC PROXY BASELINE

## A. Overall Pipeline

SeProRS is designed as a simple and reproducible baseline framework for text-only RSSC, rather than a task-specific architecture with complex modules. The purpose of this work is to establish a standard pipeline for studying semantic proxybased recognition. Therefore, each component in SeProRS is intentionally kept simple, so that the resulting performance mainly reflects the effectiveness and limitations of textual semantic proxies, instead of the capacity of a specialized classifier. The overall framework of SeProRS is illustrated in Fig. 2.

Given a remote sensing image $x _ { i } ,$ , SeProRS first generates a textual semantic proxy $s _ { i }$ through a VLM. The proxy is then encoded into a text representation and used for scene prediction. The overall process can be written as:

$$
s _ { i } = \Phi _ { \mathrm { v l m } } ( x _ { i } , p ) ,\tag{5}
$$

$$
\mathbf { z } _ { i } = E _ { \mathrm { t e x t } } ( s _ { i } ) ,\tag{6}
$$

$$
{ \hat { y } } _ { i } = g ( \mathbf { z } _ { i } ) ,\tag{7}
$$

where $\Phi _ { \mathrm { v l m } }$ denotes the VLM-based proxy generator, $p$ is the prompting instruction, $E _ { \mathrm { t e x t } }$ is a text embedding model, $\mathbf { z } _ { i }$ is the text representation, and $g$ is a lightweight prediction function. In this pipeline, the image is used only for proxy generation. Once $s _ { i }$ is obtained, both training and testing of the classifier are conducted entirely in the textual space without accessing image pixels or visual features.

This design distinguishes SeProRS from direct VLM-based classification. Instead of asking a VLM to directly output a scene label, SeProRS uses the VLM to produce an intermediate semantic proxy and then evaluates whether this proxy can support downstream classification. This separation makes the benchmark more interpretable, because the generated text can be inspected as the modality-reduced representation that mediates between the original image and the final prediction.

## B. VLM-based Semantic Proxy Generation

The first stage of SeProRS converts each remote sensing image into a textual semantic proxy. Since the proxy serves as the only input for subsequent classification, the VLM is treated as a semantic abstraction module rather than as a classifier. Its role is to translate visual observations into natural language while preserving scene-relevant evidence useful for RSSC.

For a dataset $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , each image $x _ { i }$ is processed by a VLM under a given prompt p. This produces a text-only dataset:

$$
\mathcal { D } _ { t } ^ { ( p ) } = \{ ( s _ { i } ^ { ( p ) } , y _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{8}
$$

where $s _ { i } ^ { ( p ) } \ = \ \Phi _ { \mathrm { v l m } } ( x _ { i } , p )$ denotes the generated proxy. When multiple VLMs and prompting strategies are used, the corresponding proxy sets form different text-only views of the same RSSC dataset. This allows the text-only setting to be evaluated under different semantic abstraction behaviors without changing the downstream classifier.

In our protocol, proxy generation is deliberately label-free: no class name, ground-truth label, or candidate label list is provided to the VLM. Each proxy is generated once and stored before classifier training, so downstream experiments can be repeated without calling the VLM again. This protocol prevents the VLM from acting as a label selector and makes the generated text a reusable benchmark resource.

## C. Prompting Strategies

To examine how proxy formulation affects text-only RSSC, SeProRS adopts two representative prompting strategies: Flat Prompt and Structured Prompt. These two strategies are not designed as sophisticated prompt engineering schemes, but as simple and reproducible ways to produce different forms of semantic proxies.

The Flat Prompt asks the VLM to describe the remote sensing scene in a concise natural-language paragraph. This strategy encourages a fluent scene-level description that integrates overall scene type, key objects, and functional context when possible. It resembles a caption-like proxy, but its role is different from conventional captioning: the generated text is directly used as the classification input rather than being evaluated for linguistic quality.

The Structured Prompt asks the VLM to describe the scene through multiple semantic aspects in a fixed order. In our implementation, the prompt guides the model to produce descriptions related to the global scene, object-level attributes, and functional cues. This strategy encourages the proxy to expose different types of scene evidence in a more organized form. However, the use of a structured prompt does not imply that structured text is necessarily superior. Instead, the comparison between Flat and Structured prompts is intended to reveal how different proxy formats influence the feasibility and behavior of text-only RSSC.

## D. Text-Only Scene Classification

After semantic proxy generation, SeProRS performs scene classification using only textual representations. Each proxy $s _ { i }$ is encoded by the text embedding model $E _ { \mathrm { t e x t } }$

To avoid introducing unnecessary architectural complexity, we adopt a lightweight prediction module on top of the text embeddings. Before classification, $\mathbf { z } _ { i }$ is passed through a small projection adapter consisting of layer normalization, a linear layer, GELU activation, and dropout. This adapter slightly adjusts the general-purpose text embedding space to the target RSSC label space without introducing a task-specific architecture. The adapted representation $\tilde { \mathbf { z } } _ { i }$ is then fed into a linear classifier:

$$
\hat { \mathbf { q } } _ { i } = \mathrm { s o f t m a x } ( \mathbf { W } \tilde { \mathbf { z } } _ { i } + \mathbf { b } ) ,\tag{9}
$$

where $\mathbf { W } \in \mathbb { R } ^ { C \times h }$ and $\mathbf { b } \in \mathbb { R } ^ { C }$ are learnable parameters, h is the projected feature dimension, and C denotes the number of scene categories. The training objective is the standard crossentropy loss.

![](images/7317ac7e099893351f5ac12f998a60452ce6c029d08e57da00bcb52517057c82.jpg)  
Fig. 3. Representative samples from UCM, AID, and NWPU-RESISC45 datasets, with 6 samples selected from each.

The use of a simple text embedding model and lightweight classifier is deliberate. Since the focus of this work is the textonly setting and the semantic proxy representation, a simple classification network helps prevent the benchmark from being dominated by task-specific model design. Consequently, the resulting performance provides a clearer indication of how much discriminative scene information is preserved in the generated textual proxies and how effectively such information can support RSSC under a modality-reduced inference protocol.

## V. EXPERIMENTS

## A. Datasets and Settings

SeProRS is evaluated on three widely used RSSC benchmarks, including UCM [1], AID [2] and NWPU-RESISC45 [3]. These datasets cover different spatial resolutions, category scales, and scene complexities, providing a controlled basis for analyzing whether semantic-only representations preserve discriminative scene information. The detailed dataset statistics and training/testing splits are summarized in Table II, and representative samples are shown in Fig. 3.

UCM contains 2,100 images from 21 land-use categories, with 100 images per category. Each image is of size 256 × 256 with a spatial resolution of 0.3 m. In this work, UCM is included as an explicit benchmark under 50%/50% and 80%/20% splits, enabling a compact yet informative evaluation of semantic-only classification under relatively small-scale data settings.

AID contains 10,000 aerial scene images from 30 categories, with a spatial size of 600 × 600 and spatial resolutions ranging from 0.5 m to 8 m. Following common RSSC protocols, 20% and 50% of images are used for training, while the remaining samples are used for testing.

NWPU-RESISC45 contains 31,500 images evenly distributed across 45 scene categories. Each image has a size of 256 × 256, and the spatial resolution varies from 0.2 m to 30 m. We adopt the 10%/90% and 20%/80% training/testing splits, which are widely used in RSSC evaluation and provide a challenging testbed for semantic scene discrimination.

As summarized in Table III, we consider one commercial VLM, Gemini-2.5-Flash, one efficient open VLM,

TABLE II  
SUMMARY OF THE RSSC DATASETS AND EXPERIMENTAL SPLITS USED IN SEPRORS.
<table><tr><td>Dataset</td><td>Classes</td><td>Total images</td><td>Spatial resolution</td><td>Training/Test ratio</td><td>Train/Test images</td></tr><tr><td>UCM</td><td>21</td><td>2,100</td><td>0.3 m</td><td>50% / 50% 80% / 20%</td><td>1,050 / 1,050 1,680 / 420</td></tr><tr><td>AID</td><td>30</td><td>10,000</td><td>0.5-8 m</td><td>20% / 80% 50% / 50%</td><td>2,000 / 8,000 5,000 / 5,000</td></tr><tr><td>NWPU-RESISC45</td><td>45</td><td>31,500</td><td>0.2-30 m</td><td>10% / 90% 20% /  80%</td><td>3,150 / 28,350 6,300 / 25,200</td></tr></table>

TABLE III  
VISUAL-LANGUAGE MODELS USED FOR SEMANTIC PROXY GENERATION IN SEPRORS.
<table><tr><td>ID</td><td>Generator</td><td>Access</td><td>Model Scale</td><td>Parameters</td><td>Role</td></tr><tr><td>G1</td><td>LLaVA-OneVision-Qwen2-0.5B [38]</td><td>local</td><td>Small</td><td>0.5B</td><td>lightweight open VLM</td></tr><tr><td>G2</td><td>Qwen2.5-VL-3B [39]</td><td>local</td><td>Medium</td><td>3B</td><td>efficient open VLM</td></tr><tr><td>G3</td><td>Gemini-2.5-Flash [40]</td><td>API</td><td>Commercial</td><td>Not disclosed</td><td>closed-source strong VLM</td></tr></table>

TABLE IV  
HYPERPARAMETERS FOR SEMANTIC PROXY GENERATION.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Input</td><td>Remote sensing image and task-agnostic descriptive prompt.</td></tr><tr><td>Label control</td><td>No class name, ground-truth label, or candidate label list is provided to the</td></tr><tr><td>Generators</td><td>VLM. LLaVA-OneVision-Qwen2-0.5B, Qwen2.5-VL-3B, Gemini-2.5-Flash.</td></tr><tr><td>Prompt types</td><td>Flat Prompt and Structured Prompt.</td></tr><tr><td>Flat Prompt</td><td>Generates one concise paragraph de- scribing scene-level semantics.</td></tr><tr><td>Structured Prompt</td><td>Generates global description, object- level attributes, and functional cues.</td></tr><tr><td>Storage</td><td>Semantic proxies are generated once and then reused for all classifier training and testing.</td></tr></table>

Qwen2.5-VL-3B, and one lightweight LLaVA-OneVision model. Gemini-2.5-Flash is used as a strong API-based generator, whereas Qwen2.5-VL-3B and LLaVA-OneVision-Qwen2- 0.5B are deployed locally to examine the effect of access mode, model scale, and architecture on semantic proxy quality. The dataset class names are used only for defining the supervised training labels of the downstream classifier; they are not included in the proxy-generation prompts.

## B. Implementation Details

All semantic-only classifiers are trained under the same data splits as the image-based RSSC protocols. Unless otherwise specified, the text encoder is frozen and only the projection adapter and classification head are optimized. The maximum text length is fixed to 192 tokens, and generated proxies longer than this limit are truncated. Each experiment is repeated five times with independent seeds, and the mean and standard deviation of overall accuracy (OA) are reported.

The experiments are designed to answer three questions: 1) how well can semantic-only proxies support RSSC; 2) whether structured proxies are more effective than flat paragraph-style proxies; and 3) how sensitive the classification performance is to the VLM used for semantic proxy generation. The same text-only classifier and training protocol are used across all settings.

The semantic proxy generation settings are reported in Table IV, which specifies the task-agnostic descriptive prompts and the exclusion of class labels or candidate label lists. Additional implementation settings are reported in Tables X and XI of Appendix A. Table X summarizes the text-only classifier configuration, including the frozen text encoder, projection adapter, optimization settings, and repeated runs. Table XI lists the training settings for the vision-based reference models to make the visual comparison reproducible. For both text-only and vision-based training, the base learning rate of $5 \times 1 0 ^ { - 4 }$ is linearly scaled by the global batch size divided by 512, yielding an effective learning rate of $1 . 2 5 \times 1 0 ^ { - 4 }$ at a global batch size of 128. In addition, Tables XII-XIV in Appendix B provide representative proxies from the three datasets under both Flat and Structured prompts for all three VLMs, illustrating the actual textual inputs used for downstream classification.

## C. Main Results

To provide a visual-performance reference, which can be regarded as an upper bound under full image access, comparisons are conducted against representative vision-based RSSC models, including HFAM [41], CGINet [42], UPetu [43], MSCN [44], UniRepLKNet [45] and DINOv3-Adapter. In particular, DINOv3-Adapter applies the standard adapter tuning strategy [46] to the recent vision foundation model DINOv3 [26], and therefore serves as a strong and up-to-date backbone baseline in our experiments.

Tables V-VII report the main results on UCM, AID, and NWPU-RESISC45. The vision-based methods are included as full-image reference results, while SeProRS performs classification only from textual semantic proxies. Therefore, these two groups should not be interpreted as input-matched competitors. Instead, the comparison provides a reference for understanding how much scene-discriminative information can be retained after remote sensing images are abstracted into text.

TABLE V  
MAIN RESULTS ON THE UCM DATASET. VISION-BASED METHODS USE FULL IMAGE INPUTS, WHILE SEPRORS PERFORMS CLASSIFICATION FROM TEXTUAL SEMANTIC PROXIES.
<table><tr><td>Method</td><td>50% training</td><td>80% training</td></tr><tr><td></td><td>Vision-based Methods</td><td></td></tr><tr><td>HFAM</td><td> $9 8 . 3 5 \pm 0 . 1 4$ </td><td> $9 9 . 4 3 \pm 0 . 1 3$ </td></tr><tr><td>CGINet</td><td> $9 9 . 1 4 \pm 0 . 1 6$ </td><td> $9 9 . 8 4 \pm 0 . 1 6$   $9 9 . 0 5 \pm 0 . 2 8$ </td></tr><tr><td>UPetu MSCN</td><td> $9 8 . 9 5 \pm 0 . 1 2$   $9 9 . 2 6 \pm 0 . 0 7$ </td><td> $9 9 . 9 5 \pm 0 . 1 0$ </td></tr><tr><td>UniRepLKNet</td><td> $9 9 . 3 7 \pm 0 . 0 8$ </td><td> $9 9 . 2 4 \pm 0 . 1 0$ </td></tr><tr><td>DINOv3-Adapter</td><td> $9 9 . 1 2 \pm 0 . 0 7$ </td><td> $9 9 . 5 2 \pm 0 . 0 5$ </td></tr><tr><td></td><td>SeProRS (Text-based Methods)</td><td></td></tr><tr><td>G1 + Structured</td><td> $8 8 . 4 2 \pm 0 . 3 0$ </td><td> $8 9 . 4 8 \pm 0 . 5 2$ </td></tr><tr><td>G1 + Flat</td><td> $9 0 . 2 7 \pm 0 . 1 4$ </td><td> $9 1 . 2 4 \pm 0 . 3 1$ </td></tr><tr><td>G2 + Structured</td><td> $7 7 . 3 1 \pm 0 . 2 1$ </td><td> $8 0 . 9 0 \pm 0 . 2 0$ </td></tr><tr><td> $\mathrm { G } 2 + \mathrm { F l a t }$ </td><td> $8 1 . 4 1 \pm 0 . 3 6$ </td><td> $8 3 . 8 1 \pm 0 . 4 8$ </td></tr><tr><td></td><td></td><td> $9 2 . 1 6 \pm 0 . 0 7$ </td></tr><tr><td>G3 + Structured  $\mathrm { G } 3 + \mathrm { F l a t }$ </td><td> $9 3 . 6 0 \pm 0 . 1 4$   $9 6 . 1 9 \pm 0 . 1 3$ </td><td> $9 5 . 3 9 \pm 0 . 2 7$ </td></tr></table>

TABLE VI

MAIN RESULTS ON THE AID DATASET. VISION-BASED METHODS USE FULL IMAGE INPUTS, WHILE SEPRORS PERFORMS CLASSIFICATION FROM TEXTUAL SEMANTIC PROXIES.
<table><tr><td>Method</td><td>20% training</td><td>50% training</td></tr><tr><td colspan="3">Vision-based Methods</td></tr><tr><td>HFAM</td><td> $9 4 . 7 1 \pm 0 . 0 6$ </td><td> $9 6 . 7 3 \pm 0 . 0 4$ </td></tr><tr><td>CGINet</td><td> $9 5 . 3 4 \pm 0 . 1 4$ </td><td> $9 7 . 1 0 \pm 0 . 2 4$ </td></tr><tr><td>UPetu</td><td> $9 5 . 3 3 \pm 0 . 1 0$ </td><td> $9 7 . 0 4 \pm 0 . 2 1$ </td></tr><tr><td>MSCN</td><td> $9 5 . 8 6 \pm 0 . 1 6$ </td><td> $9 7 . 4 6 \pm 0 . 1 2$ </td></tr><tr><td>UniRepLKNet DINOv3-Adapter</td><td> $9 6 . 4 6 \pm 0 . 0 7$   ${ \bf 9 7 . 5 8 \pm 0 . 1 1 }$ </td><td> $9 7 . 8 2 \pm 0 . 1 0$   ${ \bf 9 8 . 6 2 \pm 0 . 0 5 }$ </td></tr><tr><td colspan="3">SeProRS (Text-based Methods)</td></tr><tr><td>G1 + Structured</td><td> $8 6 . 8 5 \pm 0 . 0 9$ </td><td> $8 9 . 2 8 \pm 0 . 1 4$ </td></tr><tr><td>G1 + Flat</td><td> $9 0 . 4 4 \pm 0 . 1 5$ </td><td> $9 2 . 0 1 \pm 0 . 0 7$ </td></tr><tr><td>G2 + Structured</td><td> $8 3 . 8 8 \pm 0 . 0 4$ </td><td> $8 5 . 9 6 \pm 0 . 1 2$ </td></tr><tr><td> $\mathrm { G } 2 + \mathrm { F l a t }$ </td><td> $8 7 . 2 8 \pm 0 . 1 0$ </td><td> $8 8 . 5 1 \pm 0 . 1 3$ </td></tr><tr><td>G3 + Structured</td><td> $9 3 . 4 6 \pm 0 . 1 1$ </td><td> $9 4 . 7 6 \pm 0 . 0 6$ </td></tr><tr><td> $\mathrm { G } 3 + \mathrm { F l a t }$ </td><td> $9 3 . 9 5 \pm 0 . 1 3$ </td><td> $9 5 . 1 8 \pm 0 . 0 4$ </td></tr></table>

The results demonstrate that text-only RSSC achieves nontrivial performance across all three datasets, showing that semantic proxies preserve substantial scene-level information for classification. With the strongest proxy generator and the Flat Prompt, SeProRS reaches 96.19% and 95.39% on UCM, 93.95% and 95.18% on AID, and 89.13% and 90.08% on NWPU-RESISC45 under the two training ratios, respectively. Although these results are still lower than full-image visual references, they demonstrate that textual semantic proxies can support meaningful scene recognition without using image features during classification.

The performance gap between text-only and vision-based recognition varies across datasets and should be interpreted together with dataset complexity and the saturation level of visual baselines. On AID, G3+Flat reaches 93.95% and 95.18%, approaching some vision-based references, which shows that semantic proxies can preserve substantial highlevel land-use information. On UCM, SeProRS also achieves high accuracy, but visual methods are nearly saturated, mostly above 98%, making the remaining gap more visible despite the strong text-only performance. On NWPU-RESISC45, the gap is larger, suggesting that textual proxies are less able to preserve fine-grained visual cues, spatial layouts, and subtle category differences when the dataset contains more classes and stronger intra-class diversity.

TABLE VII  
MAIN RESULTS ON THE NWPU-RESISC45 DATASET. VISION-BASED METHODS USE FULL IMAGE INPUTS, WHILE SEPRORS PERFORMS CLASSIFICATION FROM TEXTUAL SEMANTIC PROXIES.
<table><tr><td>Method</td><td>10% training</td><td>20% training</td></tr><tr><td colspan="3">Vision-based Methods</td></tr><tr><td>HFAM</td><td> $9 0 . 8 1 \pm 0 . 1 1$ </td><td> $9 2 . 9 3 \pm 0 . 1 7$ </td></tr><tr><td>CGINet</td><td>92.28 ± 0.17</td><td> $9 4 . 3 8 \pm 0 . 1 3$ </td></tr><tr><td>UPetu</td><td> $9 2 . 2 0 \pm 0 . 0 7$ </td><td> $9 3 . 7 0 \pm 0 . 1 4$ </td></tr><tr><td>MSCN</td><td> $9 2 . 6 4 \pm 0 . 0 9$   $9 3 . 6 1 \pm 0 . 0 6$ </td><td> $9 4 . 5 9 \pm 0 . 1 1$   $9 5 . 2 6 \pm 0 . 1 1$ </td></tr><tr><td>UniRepLKNet  $\mathrm { D I N O v } 3 { \mathrm { - A d a p t e r } }$ </td><td> ${ \bf 9 4 . 6 1 \pm 0 . 1 0 }$ </td><td> ${ \bf 9 5 . 8 9 \pm 0 . 0 4 }$ </td></tr><tr><td colspan="3">SeProRS (Text-based Methods)</td></tr><tr><td>G1 + Structured</td><td> $7 6 . 1 0 \pm 0 . 1 0$ </td><td> $7 8 . 1 5 \pm 0 . 1 0$ </td></tr><tr><td> $\mathrm { G 1 } + \mathrm { F l a t }$ </td><td> $8 3 . 7 9 \pm 0 . 0 6$ </td><td> $8 5 . 5 1 \pm 0 . 0 5$ </td></tr><tr><td>G2 + Structured</td><td> $6 7 . 6 6 \pm 0 . 1 1$ </td><td> $6 9 . 6 7 \pm 0 . 0 8$ </td></tr><tr><td> $\mathrm { G } 2 + \mathrm { F l a t }$ </td><td> $7 3 . 3 9 \pm 0 . 1 6$ </td><td> $7 5 . 2 6 \pm 0 . 1 3$ </td></tr><tr><td> $\mathrm { G 3 } + \mathrm { S t r u c t u r e d }$ </td><td> $8 7 . 6 4 \pm 0 . 0 6$ </td><td> $8 8 . 6 4 \pm 0 . 0 7$ </td></tr><tr><td> $\mathrm { G } 3 + \mathrm { F l a t }$ </td><td> $8 9 . 1 3 \pm 0 . 0 8$ </td><td> $9 0 . 0 8 \pm 0 . 0 6$ </td></tr></table>

## D. Effect of VLMs

The performance of SeProRS is clearly affected by the proxy generator, yet it does not increase monotonically with model scale. While G3 generally provides the strongest results, G1 outperforms G2 in all matched configurations, indicating that larger VLMs do not always generate more discriminative proxies for text-only RSSC. This result suggests that proxy quality should be evaluated by its downstream semantic utility, including faithfulness, scene specificity, and compatibility with the text embedding space, rather than by generator size alone.

## E. Effect of Prompting Strategies

The Flat Prompt consistently outperforms the Structured Prompt across all VLMs and datasets. This result suggests that a more explicitly organized description does not necessarily lead to better text-only classification. A possible explanation is that flat descriptions tend to produce more coherent scenelevel semantics, which better match the representation space of general-purpose text embedding models. In contrast, structured descriptions may introduce rigid formatting or uneven information across semantic fields, which can weaken the overall discriminability of the resulting text representation.

## F. Text Embedding Behavior

Fig. 4 visualizes the text embeddings of semantic proxies using t-SNE on AID. First, many categories form recognizable clusters, indicating that the generated proxies preserve class-discriminative semantic information rather than acting as generic descriptions. Second, the visualization further shows that proxy generation quality is reflected in the text feature space. The separability of different generators is broadly consistent with the quantitative trend in Tables V-VII, where G3 exhibits the clearest class structures, followed by G1, while G2 shows more noticeable overlaps. Finally, the difference between the two rows demonstrates that the lightweight projection layer makes the class structures more compact and separable, indicating its role as a simple text-feature adapter for aligning general-purpose text embeddings with the RSSC label space.

![](images/ba67a7baa0c6b4c661fe85770bda1129d5c372ffb8cfe8f6237477ab6250354c.jpg)  
Fig. 4. t-SNE visualization of semantic proxy features on AID under the 50% training setting with Flat Prompt. Columns correspond to G1, G2, and G3, respectively. The first and second rows show the text embeddings before and after the lightweight projection layer, which serves as a text-feature adapter for RSSC.

TABLE VIII  
CROSS-GENERATOR TRANSFER ACCURACY ON AID WITH THE 50% TRAINING SPLIT AND FLAT PROMPT.
<table><tr><td>Train/Test</td><td>G1</td><td>G2</td><td>G3</td></tr><tr><td>G1</td><td>92.08</td><td>79.66</td><td>80.56</td></tr><tr><td>G2</td><td>82.16</td><td>88.54</td><td>82.10</td></tr><tr><td>G3</td><td>78.78</td><td>78.40</td><td>95.16</td></tr></table>

## G. Cross-Generator Transfer

Table VIII presents the cross-generator transfer results on AID under the 50% training setting with Flat Prompt. The diagonal entries correspond to the standard same-generator setting, while the off-diagonal entries evaluate whether a classifier trained on proxies from one VLM can generalize to proxies generated by another VLM. A clear observation is that the diagonal results are consistently higher than most off-diagonal results, indicating that semantic proxies exhibit generator-specific distributions. This suggests that different VLMs may describe the same remote sensing scene with different language styles, semantic granularity, or descriptive preferences.

Interestingly, G3 achieves the highest same-generator accuracy, reaching 95.16%, but its proxies transfer poorly to G1 and G2, with accuracies of 78.78% and 78.40%, respectively. This indicates that a stronger proxy generator can produce highly discriminative proxies within its own semantic space, but these proxies may also introduce a more generator-specific representation style. In contrast, G2 shows more balanced cross-generator behavior: although its same-generator accuracy is lower than G3, training on G2 proxies transfers better to both G1 and G3. These results reveal that proxy quality is not determined solely by same-generator classification accuracy; transferability across VLMs is another important property of semantic proxies.

Overall, the off-diagonal results remain substantially above chance, showing that different VLMs still preserve shared scene-level semantic evidence. However, the gap between diagonal and off-diagonal performance reveals a proxy distribution shift across generators. This finding suggests that future text-only RSSC methods may benefit from generator-invariant proxy learning or proxy normalization strategies, especially when semantic proxies are collected from heterogeneous VLM sources.

## H. Efficiency and Scalability

Table IX reports the efficiency comparison on AID under the 50%-50% setting. SeProRS uses only 0.54M trainable parameters, much fewer than DINOv3-Adapter (7.12M) and UniRepLKNet-T (31.07M). It also achieves substantially lower peak memory consumption, requiring only 1266 MB compared with 8543 MB and 15793 MB for the two visionbased methods. In terms of speed, SeProRS needs 0.0701 s/it for training and 0.0935 s/it for testing, making it clearly faster than the visual references.

These results show that, after semantic proxies are generated, the downstream text-only classifier is lightweight and scalable. Although the offline VLM-based proxy generation cost is not included, the generated proxies can be reused across different classifiers or analyses. Thus, SeProRS provides an efficient baseline for modality-reduced RSSC, trading a moderate accuracy gap for significantly lower trainable parameters, memory usage, and computational cost.

## I. Limitations

The experimental results clarify the potential of text-only RSSC while also exposing the limitations introduced by semantic abstraction.

First, textual semantic proxies can serve as partial substitutes for remote sensing images. Although SeProRS does not access image pixels or visual features during classification, it achieves meaningful performance across three standard datasets. This indicates that high-level scene evidence, such as dominant objects, land-use functions, spatial context, and object co-occurrence, can be transferred from images to text. However, this substitution is incomplete, since fine-grained visual cues such as texture, scale, geometry, and subtle spatial layouts are difficult to fully preserve in natural language.

TABLE IX  
EFFICIENCY COMPARISON ON AID UNDER THE 50%-50% SETTING.
<table><tr><td>Type</td><td>Method</td><td>Max OA (%)</td><td>Params (M)</td><td>Train Time (s/it)</td><td>Test Time (s/it)</td><td>Max Mem. (MB)</td></tr><tr><td>Vision</td><td>DINOv3-ViT-B/16 Adapter</td><td>98.58</td><td>7.12</td><td>0.3025</td><td>0.2861</td><td>8543</td></tr><tr><td>Vision</td><td>UniRepLKNet-T</td><td>97.88</td><td>31.07</td><td>0.4557</td><td>0.3030</td><td>15793</td></tr><tr><td>Text</td><td>SeProRS</td><td>95.20</td><td>0.54</td><td>0.0701</td><td>0.0935</td><td>1266</td></tr></table>

Second, the cross-generator transfer results suggest the potential of more flexible image-free inference. A classifier trained on proxies generated by one VLM can still obtain nontrivial performance on proxies generated by another VLM, indicating that different generators preserve shared scenelevel semantics. This means that, once a semantic proxy is available, the final classification stage does not depend on the original image or a fixed VLM generator. Meanwhile, the performance drop in cross-generator settings reveals generatorspecific proxy shifts, suggesting that generator-invariant proxy learning is a promising future direction.

Third, SeProRS is intended as a basic and reproducible baseline rather than a fully optimized solution. Its components are deliberately simple, including prompt-based proxy generation, general-purpose text encoding, a lightweight projection adapter, and a linear classifier. This design helps isolate the role of semantic proxies, but also leaves considerable room for improvement. Future work may further narrow the gap to vision-based methods by improving proxy generation, designing remote sensing-oriented text encoders, refining projection adapters, or introducing proxy quality estimation.

## VI. CONCLUSION

This paper introduced text-only RSSC, a new setting that studies whether remote sensing scenes can be classified from textual semantic proxies alone. We proposed SeProRS as a reproducible semantic proxy baseline and established a benchmark using three datasets, three VLMs, and two prompting strategies. Experiments show that semantic proxies can preserve substantial discriminative information, while also exposing limitations caused by information loss and proxy generation quality. Additional implementation details and proxy examples further indicate that the benchmark is reproducible and inspectable: the VLMs generate label-free textual proxies, and the downstream classifier operates entirely in the textual space. We hope this work will stimulate future research on semantic proxy-based and modality-reduced remote sensing understanding.

## REFERENCES

[1] Y. Yang and S. Newsam, “Bag-of-visual-words and spatial extensions for land-use classification,” in Proceedings of the 18th SIGSPATIAL International Conference on Advances in Geographic Information Systems. ACM, 2010, pp. 270–279.

[2] G.-S. Xia, J. Hu, F. Hu, B. Shi, X. Bai, Y. Zhong, L. Zhang, and X. Lu, “AID: A benchmark data set for performance evaluation of aerial scene classification,” IEEE transactions on geoscience and remote sensing, vol. 55, no. 7, pp. 3965–3981, 2017.

[3] G. Cheng, J. Han, and X. Lu, “Remote sensing image scene classification: Benchmark and state of the art,” Proceedings of the IEEE, vol. 105, no. 10, pp. 1865–1883, 2017.

[4] G. Cheng, X. Xie, J. Han, L. Guo, and G. Xia, “Remote sensing image scene classification meets deep learning: Challenges, methods, benchmarks, and opportunities,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 13, pp. 3735–3756, 2020.

[5] Y. Bazi, L. Bashmal, M. M. Al Rahhal, R. Al Dayil, and N. Al Ajlan, “Vision transformers for remote sensing image classification,” Remote Sensing, vol. 13, no. 3, p. 516, 2021.

[6] Y. Cong, S. Khanna, C. Meng, P. Liu, E. Rozi, Y. He, M. Burke, D. B. Lobell, and S. Ermon, “SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 197–211.

[7] C. J. Reed, R. Gupta, S. Li, S. Brockman, C. Funk, B. Clipp, K. Keutzer, S. Candido, M. Uyttendaele, and T. Darrell, “Scale-MAE: A scale-aware masked autoencoder for multiscale geospatial representation learning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4088–4099.

[8] X. Sun, P. Wang, W. Lu, Z. Zhu, X. Lu, Q. He, J. Li, X. Rong, Z. Yang, H. Chang et al., “RingMo: A remote sensing foundation model with masked image modeling,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–22, 2022.

[9] Z. Xiong, Y. Wang, F. Zhang, A. J. Stewart, J. Hanna, D. Borth, I. Papoutsis, B. L. Saux, G. Camps-Valls, and X. X. Zhu, “Neural plasticity-inspired foundation model for observing the earth crossing modalities,” arXiv preprint arXiv:2403.15356, 2024.

[10] S. Lu, J. Guo, J. R. Zimmer-Dauphinee, J. M. Nieusma, X. Wang, P. VanValkenburgh, S. A. Wernke, and Y. Huo, “Vision foundation models in remote sensing: A survey,” IEEE Geoscience and Remote Sensing Magazine, vol. 13, no. 3, pp. 190–215, 2025.

[11] C. Huo, K. Chen, S. Zhang, Z. Wang, H. Yan, J. Shen, Y. Hong, G. Qi, H. Fang, and Z. Wang, “When remote sensing meets foundation model: A survey and beyond,” remote sensing, vol. 17, no. 2, p. 179, 2025.

[12] X. Li, C. Wen, Y. Hu, and N. Zhou, “RS-CLIP: Zero shot remote sensing scene classification via contrastive vision-language supervision,” International journal of applied earth observation and geoinformation, vol. 124, p. 103497, 2023.

[13] F. Liu, D. Chen, Z. Guan, X. Zhou, J. Zhu, Q. Ye, L. Fu, and J. Zhou, “Remoteclip: A vision language foundation model for remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–16, 2024.

[14] Z. Zhang, T. Zhao, Y. Guo, and J. Yin, “RS5M and GeoRSCLIP: A large-scale vision-language dataset and a large visionlanguage model for remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–23, 2024.

[15] K. El Khoury, M. Zanella, B. Gerin, T. Godelaine, B. Macq,´ S. Mahmoudi, C. De Vleeschouwer, and I. B. Ayed, “Enhancing remote sensing vision-language models for zero-shot scene classification,” in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[16] I. Dimitrovski, V. Spasev, and I. Kitanovski, “Few-shot remote sensing image scene classification with clip and prompt learning,” Journal of Intelligent Information Systems, vol. 64, no. 2, pp. 497–532, 2026.

[17] Y. Bazi, L. Bashmal, M. M. Al Rahhal, R. Ricci, and F. Melgani, “RS-LLaVA: A large vision-language model for joint captioning and question answering in remote sensing imagery,” Remote Sensing, vol. 16, no. 9, p. 1477, 2024.

[18] Y. Zhan, Z. Xiong, and Y. Yuan, “SkyEyeGPT: Unifying remote sensing vision-language tasks via instruction tuning with large language model,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 221, pp. 64–77, 2025.

[19] F. Hu, G.-S. Xia, J. Hu, and L. Zhang, “Transferring deep convolutional neural networks for the scene classification of highresolution remote sensing imagery,” Remote Sensing, vol. 7, no. 11, pp. 14 680–14 707, 2015.

[20] K. Nogueira, O. A. Penatti, and J. A. Dos Santos, “Towards better exploiting convolutional neural networks for remote sensing scene classification,” Pattern Recognition, vol. 61, pp. 539–556, 2017.

[21] W. Wang, Y. Chen, and P. Ghamisi, “Transferring cnn with adaptive learning for remote sensing scene classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–18, 2022.

[22] C. Yang, C. Wei, Y. Zhao, L. Wang, P. Xu, K. Qi, Y. Shao, and H. Wu, “MSALNet: a multi-scale adaptive learning network for high-resolution remote sensing scene classification,” Geospatial Information Science, vol. 29, no. 1, pp. 143–167, 2026.

[23] L. Tong, J. Liu, and B. Du, “SceneFormer: Neural architecture search of transformers for remote sensing scene classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–15, 2025.

[24] A. Sivasubramanian, P. VR, S. V, and V. Ravi, “Transformer based ensemble deep learning approach for remote sensing natural scene classification,” International Journal of Remote Sensing, vol. 45, no. 10, pp. 3289–3309, 2024.

[25] X. Li, C. Wen, Y. Hu, Z. Yuan, and X. X. Zhu, “Vision-language models in remote sensing: Current progress and future trends,” IEEE Geoscience and Remote Sensing Magazine, vol. 12, no. 2, pp. 32–66, 2024.

[26] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab,´ C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.

[27] K. Wu, Y. Zhang, L. Ru, B. Dang, J. Lao, L. Yu, J. Luo, Z. Zhu, Y. Sun, J. Zhang et al., “A semantic-enhanced multi-modal remote sensing foundation model for earth observation,” Nature Machine Intelligence, vol. 7, no. 8, pp. 1235–1249, 2025.

[28] C. F. Brown, M. R. Kazmierski, V. J. Pasquarella, W. J. Rucklidge, M. Samsikova, C. Zhang, E. Shelhamer, E. Lahera, O. Wiles, S. Ilyushchenko et al., “Alphaearth foundations: An embedding field model for accurate and efficient global mapping from sparse label data,” arXiv preprint arXiv:2507.22291, 2025.

[29] X. Lu, B. Wang, X. Zheng, and X. Li, “Exploring models and data for remote sensing image caption generation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 4, pp. 2183–2195, 2018.

[30] Z. Yuan, W. Zhang, K. Fu, X. Li, C. Deng, H. Wang, and X. Sun, “Exploring a fine-grained multiscale method for crossmodal remote sensing image retrieval,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–19, 2022.

[31] Z. Yuan, Z. Xiong, L. Mou, and X. Zhu, “ChatEarthNet: a global-scale image–text dataset empowering vision–language geo-foundation models,” Earth System Science Data, vol. 17, no. 3, pp. 1245–1263, 2025.

[32] X. Weng, C. Pang, and G.-S. Xia, “Vision-language modeling meets remote sensing: Models, datasets, and perspectives,” IEEE Geoscience and Remote Sensing Magazine, vol. 13, no. 3, pp. 276–323, 2025.

[33] L. Tao, H. Zhang, H. Jing, Y. Liu, D. Yan, G. Wei, and X. Xue,

“Advancements in vision–language models for remote sensing: Datasets, capabilities, and enhancement techniques,” Remote Sensing, vol. 17, no. 1, p. 162, 2025.

[34] W. Chen, J. Chen, Y. Deng, J. Chen, Y. Feng, Z. Xi, D. Liu, K. Li, and Y. Meng, “LRSCLIP: A vision-language foundation model for aligning remote sensing image with longer text,” arXiv preprint arXiv:2503.19311, 2025.

[35] Z. Li, D. Muhtar, F. Gu, Y. He, X. Zhang, P. Xiao, G. He, and X. Zhu, “Lhrs-bot-nova: Improved multimodal large language model for remote sensing vision-language interpretation,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 227, pp. 539–550, 2025.

[36] H. Li, X. Zhang, and H. Qu, “Ddfav: Remote sensing large vision language models dataset and evaluation benchmark,” Remote Sensing, vol. 17, no. 4, p. 719, 2025.

[37] Z. Zhang, H. Shen, T. Zhao, B. Chen, Z. Guan, Y. Wang, X. Jia, Y. Cai, Y. Shang, and J. Yin, “GeoRSMLLM: A multimodal large language model for vision-language tasks in geoscience and remote sensing,” arXiv preprint arXiv:2503.12490, 2025.

[38] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu et al., “Llava-onevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.

[39] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin, “Qwen2.5-vl technical report,” 2025.

[40] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025.

[41] Q. Wan, Z. Xiao, Y. Yu, Z. Liu, K. Wang, and D. Li, “A hyperparameter-free attention module based on feature map mathematical calculation for remote-sensing image scene classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–18, 2023.

[42] Y. Zhao, Y. Chen, S. Xiong, X. Lu, X. X. Zhu, and L. Mou, “Co-enhanced global-part integration for remote-sensing scene classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–14, 2024.

[43] Z. Dong, Y. Gu, and T. Liu, “UPetu: A unified parameterefficient fine-tuning framework for remote sensing foundation model,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–13, 2024.

[44] J. Ma, W. Jiang, X. Tang, X. Zhang, F. Liu, and L. Jiao, “Multiscale sparse cross-attention network for remote sensing scene classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–16, 2025.

[45] Y. Zhang, X. Ding, and X. Yue, “Scaling up your kernels: Large kernel design in convnets towards universal representations,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[46] J. Wu, Z. Wang, M. Hong, W. Ji, H. Fu, Y. Xu, M. Xu, and Y. Jin, “Medical sam adapter: Adapting segment anything model for medical image segmentation,” Medical image analysis, vol. 102, p. 103547, 2025.

TABLE X  
HYPERPARAMETERS FOR TEXT-ONLY CLASSIFICATION.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Text encoder</td><td>bge-base-en-v1.5</td></tr><tr><td>Encoder update</td><td>Frozen</td></tr><tr><td>Maximum text length</td><td>192 tokens</td></tr><tr><td>Projection adapter</td><td>LayerNorm, Linear 768→512, GELU, Dropout</td></tr><tr><td>Classifier head</td><td>LayerNorm, Linear 512→256, GELU, Dropout, Linear</td></tr><tr><td>Loss</td><td>Cross-entropy</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Base learning rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Effective learning rate</td><td> $1 . 2 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Warm-up learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Minimum learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Schedule</td><td>Cosine schedule with 5 warm-up epochs</td></tr><tr><td>Epochs</td><td>200</td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Runs</td><td>5 independent seeds</td></tr></table>

TABLE XI

HYPERPARAMETERS FOR VISION-BASED REFERENCE METHODS.
<table><tr><td>Item</td><td>DINOv3 Adapter</td><td>UniRepLKNet-T</td></tr><tr><td>Input size</td><td> $2 2 4 \times 2 2 4$ </td><td> $2 2 4 \times 2 2 4$ </td></tr><tr><td>Pretraining</td><td>DINOv3 ViT-B/16</td><td>ImageNet</td></tr><tr><td>Training epochs</td><td>200</td><td>200</td></tr><tr><td>Batch size</td><td>128</td><td>128</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Base learning rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Effective learning rate</td><td> $1 . 2 5 \times 1 0 ^ { - 4 }$ </td><td> $1 . 2 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Warm-up learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Minimum learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Weight decay</td><td>0.05</td><td>0.05</td></tr><tr><td>Schedule</td><td>Cosine</td><td>Cosine</td></tr><tr><td>Warm-up epochs</td><td>5</td><td>5</td></tr></table>

These are the original generated descriptions, without manual corrections or abridgment. The full stored proxies are shown; downstream encoding still uses the 192-token limit reported in Table X. Any image identifiers or class names shown for reader interpretation are not included in the generation prompts. The examples illustrate differences in semantic detail and factual reliability; for instance, the $G _ { 2 }$ Flat proxy for airplane 001.jpg describes aircraft as being in flight, whereas the other descriptions characterize them as parked. Such generator-dependent statements are retained to document the inputs available to the text-only classifier, rather than presented as verified scene annotations.

## APPENDIX A

## ADDITIONAL IMPLEMENTATION DETAILS

This appendix reports the classifier training settings and complete representative semantic proxies. Table X summarizes the text-only classifier configuration, and Table XI lists the settings of the vision-based reference models. The semantic proxy generation protocol is reported in Table IV of the main text.

The training implementation scales the base learning rate linearly with the global batch size, i.e., $\begin{array} { r l } { \eta _ { \mathrm { e f f } } } & { { } = } \end{array}$ $\eta _ { \mathrm { b a s e } } B _ { \mathrm { g l o b a l } } / 5 1 2$ . With $\eta _ { \mathrm { b a s e } } = 5 \times 1 0 ^ { - 4 }$ and $B _ { \mathrm { g l o b a l } } = 1 2 8 .$ the effective learning rate is $1 . 2 5 \times 1 0 ^ { - 4 }$ for both text-only and vision-based training, as confirmed by the training logs. The scheduler uses five warm-up epochs starting from $1 \times 1 0 ^ { - 6 }$ followed by cosine decay with a minimum learning rate of $1 \times 1 0 ^ { - 6 }$

## APPENDIX B

## REPRESENTATIVE SEMANTIC PROXY EXAMPLES

Tables XII–XIV provide complete representative proxies for one image from each of the three datasets. For each image, both Flat and Structured outputs are shown for all three generators: $G _ { 1 }$ (LLaVA-OneVision-Qwen2-0.5B), $G _ { 2 }$ (Qwen2.5-VL-3B), and $G _ { 3 }$ (Gemini-2.5-Flash). Dataset and image identifiers are retained to support comparison across generators and prompting strategies.

TABLE XII  
REPRESENTATIVE SEMANTIC PROXIES ON UCM FOR THE SAME IMAGE UNDER FLAT AND STRUCTURED PROMPTS. $G _ { 1 } , G _ { 2 } ,$ , AND $G _ { 3 }$ DENOTE LLAVA-ONEVISION-QWEN2-0.5B, QWEN2.5-VL-3B, AND GEMINI-2.5-FLASH, RESPECTIVELY.
<table><tr><td>Dataset</td><td>Image</td><td>VLM</td><td>Prompt</td><td>Semantic proxy</td></tr><tr><td>UCM</td><td>agricultural00.tif</td><td> $G _ { 1 }$ </td><td>Flat</td><td>The image depicts a large, well-organized agricultural field with rows of crops. The field is divided into multiple sections by straight lines that run parallel to each other, indicating the use of precision farming techniques. The crops appear to be in good health and are likely being cultivated for their potential yield or market value. There are no visible structures or people within the field, suggesting it may be an open space</td></tr><tr><td>UCM</td><td>agricultural00.tif</td><td> $G _ { 1 }$ </td><td>Structured</td><td>used for crop cultivation rather than a farmstead. The image shows a large, well-maintained agricultural field with neatly arranged rows of crops. The field is bordered by a dirt road and there are trees visible in the background. Buildings can be seen lining the perimeter of the field, indicating residential or commercial structures. Roads run parallel to the fields, providing access for vehicles and farming equipment. Vegetation includes grasses and other plants that cover the soil, contributing to the overall health of the crop. The scene appears to be an active agricultural area, likely used for growing crops such as grains or vegetables. The presence of roads suggests easy transportation for farm</td></tr><tr><td>UCM</td><td>agricultural00.tif</td><td> $G _ { 2 }$ </td><td>Flat</td><td>farm. The remote sensing image depicts a large agricultural field with rows of crops, likely wheat or corn, stretching across the landscape. The field is bordered by a dense forest on the left side, indicating a transition between agricultural land and natural vegetation. The functional context suggests this area is used for crop cultivation,</td></tr><tr><td>UCM</td><td>agricultural00.tif</td><td> $G _ { 2 }$ </td><td>Structured</td><td>possibly for commercial purposes such as grain production. The image depicts a large agricultural field with rows of crops, likely wheat or corn, stretching across the landscape. The field is surrounded by a mix of trees and open areas, indicating a rural setting. The functional use of this area is primarily agricultural, designed for crop cultivation and harvesting</td></tr><tr><td>UCM</td><td>agricultural00.tif</td><td> $G _ { 3 }$ </td><td>Flat</td><td>An expansive agricultural scene characterized by numerous, precisely aligned green crop rows covering the majority of the area, hinting at large-scale cultivation or a nursery, with a small patch of denser, irregular vegetation in the upper left corner and a bright, rectilinear artificial structure at the bottom right, collectively</td></tr><tr><td>UCM</td><td>agricultural00.tif</td><td> $G _ { 3 }$ </td><td>Structured</td><td>depicting an organized rural or peri-urban agricultural landscape. The scene depicts a predominantly agricultural landscape characterized by extensive cultivation. It features numerous parallel crop rows, a small patch of natural vegetation, and a potential artificial structure at the periphery. This area is primarily used for agricultural production, indicating a farming context.</td></tr></table>

REPRESENTATIVE SEMANTIC PROXIES ON AID FOR THE SAME IMAGE UNDER FLAT AND STRUCTURED PROMPTS. $G _ { 1 } , G _ { 2 }$ , AND $G _ { 3 }$ DENOTE LLAVA-ONEVISION-QWEN2-0.5B, QWEN2.5-VL-3B, AND GEMINI-2.5-FLASH, RESPECTIVELY.  
TABLE XIII
<table><tr><td>Dataset</td><td>Image</td><td>VLM</td><td>Prompt</td><td>Semantic proxy</td></tr><tr><td>AID</td><td>airport_1.jpg</td><td> $G _ { 1 }$ </td><td>Flat</td><td>The image is an aerial view of a busy airport tarmac, showcasing multiple airplanes parked at gates. The tarmac is marked with various lines and symbols indicating taxiways and other operational areas. In the center of the image, there&#x27;s a large jet bridge connecting one terminal to another, facilitating passenger boarding and deplaning. Adjacent to this jet bridge, several smaller aircraft are stationed on the ground, likely awaiting their turn for departure or having just arrived. The overall scene represents a typical day at an airport, with</td></tr><tr><td>AID</td><td>airport_1.jpg</td><td> $G _ { 1 }$ </td><td>Structured</td><td>planes ready for takeoff and passengers waiting in anticipation. The image shows an aerial view of a busy airport terminal with multiple jet bridges connecting the terminals. There are several airplanes parked at gates, and various buildings and roads can be seen surrounding the airport. Buildings include large hangars, terminals, and control towers, while roads are visible as well as parking areas for vehicles. Vegetation is present in the form of grassy patches near the terminal buildings. The scene represents transportation-related activities, with airplanes being the primary mode of transport. The presence of jet bridges indicates that passengers can board or disembark from aircraft directly to the terminal.</td></tr><tr><td>AID</td><td>airport_1.jpg</td><td> $G _ { 2 }$ </td><td>Flat</td><td>The image depicts an expansive airport terminal with multiple runways and taxiways, featuring several aircraft parked at gates. The terminal is surrounded by various ground support equipment and vehicles, indicating active operations. The surrounding area includes parking lots and possibly maintenance facilities, suggesting a well-equipped airport infrastructure.</td></tr><tr><td>AID</td><td>airport_1.jpg</td><td> $G _ { 2 }$ </td><td>Structured</td><td>The image depicts an expansive airport terminal with multiple runways and taxiways, surrounded by various infrastructure elements including parking lots and service areas. The terminal is characterized by its long, linear structure with numerous windows and doors, indicative of passenger boarding and disembarking points. The functional aspect of this scene is primarily transportation, serving as a hub for air travel with clear demarcations for aircraft movement and ground operations.</td></tr><tr><td></td><td></td><td> $G _ { 3 }$ </td><td>Flat</td><td>An expansive airport facility is depicted, featuring a prominent, elongated passenger terminal building with numerous aircraft parked at gates along its length, surrounded by vast concrete aprons, intricate taxiways, and portions of runways, alongside a distinct landscaped circular structure near the main terminal, all indicative of a busy aviation hub designed for air travel and transport.</td></tr><tr><td>AID</td><td>airport_1.jpg</td><td> $G _ { 3 }$ </td><td>Structured</td><td>The scene depicts a large-scale developed area characterized by extensive paved surfaces and structures, indicative of an airport. Numerous aircraft are visible parked alongside a long, multi-sectioned terminal building, surrounded by an intricate network of runways, taxiways, and aprons. This area serves as a critical hub for air transportation, facilitating the movement and servicing of aircraft and passengers.</td></tr></table>

TABLE XIV  
REPRESENTATIVE SEMANTIC PROXIES ON NWPU-RESISC45 FOR THE SAME IMAGE UNDER FLAT AND STRUCTURED PROMPTS. $G _ { 1 } , G _ { 2 } ,$ , AND $G _ { 3 }$ DENOTE LLAVA-ONEVISION-QWEN2-0.5B, QWEN2.5-VL-3B, AND GEMINI-2.5-FLASH, RESPECTIVELY.
<table><tr><td>Dataset</td><td>Image</td><td>VLM</td><td>Prompt</td><td>Semantic proxy</td></tr><tr><td>NWPU- RESISC45</td><td>airplane_001.jpg</td><td> $G _ { 1 }$ </td><td>Flat</td><td>The image shows a scene of three airplanes parked on an airport tarmac. The planes are positioned in a staggered formation, with the middle plane slightly ahead of the other two. Each airplane is white and appears to be a commercial jet airliner. The tarmac surface is a light grey color, typical for such environments. There are no visible markings or identifiers that would indicate the specific location or airline. The overall scene type is remote sensing, as it captures aerial imagery rather than ground-level observations.</td></tr><tr><td>NWPU- RESISC45</td><td>airplane_001.jpg</td><td> $G _ { 1 }$ </td><td>Structured</td><td>The image shows a large, open area with three airplanes parked on the tarmac. The planes are white and appear to be commercial airliners. There is also some vegetation visible in the background. The airplanes are likely used for commercial purposes such as air travel or cargo transport. The tarmac appears to be made of concrete and there are no other objects or structures visible in the immediate vicinity. The scene suggests that it could be an airport or a transportation hub where aircraft are parked before takeoff or after landing. The presence of multiple airplanes indicates that this location serves a significant number of passengers or</td></tr><tr><td>NWPU- RESISC45</td><td>airplane_001.jpg</td><td> $G _ { 2 }$ </td><td>Flat</td><td>cargo operations. The image depicts an aerial view of an airport runway with three airplanes in flight. Two aircraft are positioned closer to the camera, appearing larger due to their proximity, while the third is further away, smaller in size. The planes are likely commercial jets, given their size and design, and they are flying at a low altitude, suggesting they might be preparing for landing or taking off. The runway is clearly marked, indicating its functional use for aircraft operations.</td></tr><tr><td>NWPU- RESISC45 NWPU-</td><td>airplane_001.jpg</td><td> $G _ { 2 }$ </td><td>Structured</td><td>The image depicts an airport with multiple airplanes on the tarmac. The ground is paved with asphalt, indicating a well-maintained runway area. Airplanes are present, suggesting active air traffic at this location.</td></tr><tr><td>RESISC45</td><td>airplane_001.jpg</td><td> $G _ { 3 }$ </td><td>Flat</td><td>An airbase or large airport scene is presented, characterized by multiple large, propeller-driven transport aircraft prominently parked on a paved tarmac or apron, strongly indicating a military or heavy logistics operational facility.</td></tr><tr><td>NWPU- RESISC45</td><td>airplane_001.jpg</td><td> $G _ { 3 }$ </td><td>Structured</td><td>This is an aerial scene dominated by an airport or airfield environment. Multiple propeller aircraft are visible parked on large, untextured concrete or asphalt surfaces. The scene clearly indicates a transportation function, specifically for air travel and aircraft operations.</td></tr></table>