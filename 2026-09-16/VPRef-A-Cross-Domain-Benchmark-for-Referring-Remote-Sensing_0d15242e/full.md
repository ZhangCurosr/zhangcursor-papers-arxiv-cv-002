# VPRef: A Cross-Domain Benchmark for Referring Remote Sensing

Image Segmentation

Quanwei Liu, Student Member, IEEE, Tao Huang, Senior Member, IEEE, Jiaqi Yang, Member, IEEE, Wei Xiang, Senior Member, IEEE

Abstract—Rapid advancements in vision-language models have propelled Referring Remote Sensing Image Segmentation (RRSIS) to the forefront of Earth observation. However, practical deployments suffer severe performance degradation under a coupled dual-drift paradigm: visual domain drift from crossspatial-resolution mismatches and spectral variations, alongside textual logic drift from unconstrained, variable user-input granularities. To mitigate these bottlenecks, this paper establishes the first cross-domain RRSIS benchmark, designated as the Vaihingen-Potsdam Referring (VPRef) dataset, comprising 46,972 language-image-annotation triplets organized into a three-tier linguistic hierarchy. Building upon this benchmark, we develop a tailored parameter-efficient domain adaptation baseline anchored on the Segment Anything Model (SAM3) via Low-Rank Adaptation (LoRA). Our framework counteracts visual distribution discrepancies through pseudo-label-driven self-training and addresses textual logic drift via random multi-granularity text prompt mixing. Crucially, the distribution of empirical metrics across ablative variants suggests a potential decoupling between cross-modal semantic robustification and visual domain alignment, demonstrating that linguistic variance drives finegrained semantic invariance while pseudo-label propagation governs macro-scale spatial grid alignment. Extensive benchmarks demonstrate the proposed framework achieves superior cross-domain segmentation boundaries while modifying merely 1.08% of the foundational parameter footprint, establishing a robust baseline for future multi-modal remote sensing domain adaptation research. The dataset and code will be available at https://github.com/quanweiliu/VPRef.

Index Terms—Referring segmentation, domain adaptation, vision and language, remote sensing.

## I. INTRODUCTION

APID advancements in high-resolution remote sensing language models have propelled Referring Remote Sensing Image Segmentation (RRSIS) to the forefront of Earth observation research [1]. Compared with traditional semantic segmentation or object detection frameworks that are restricted to predefined, closed-set categories, RRSIS offers an inherently user-friendly interaction paradigm. In practical deployment scenarios, such as disaster response [2], urban planning [3] and smart agriculture [4], non-expert operators can bypass tedious professional labels and manual boundary tracing. Instead, by providing intuitive natural language instructions, such as “the blue-roofed factory building on the north side of the road and partially occluded by trees”, users enable the model to achieve pixel-level localization within large-scale imagery [5]. This natural-language-as-an-interface paradigm substantially unlocks the practical utility of remote sensing big data.

Transitioning RRSIS models from controlled laboratory environments to unconstrained real-world applications exposes severe generalization vulnerabilities [6]. The primary bottleneck stems from inherent visual domain drift across remote sensing datasets. Current state-of-the-art architectures, including LAVT [7] and RMSIN [8], are evaluated almost exclusively within a closed, single-domain setting where the training and testing distributions share identical geographic regions and sensor configurations. Actual deployments, however, require models to generalize across heterogeneous cities and variable acquisition conditions. Pronounced variations in regional landscapes, sensor spectral characteristics, and particularly the non-linear scaling induced by cross-spatial resolution mismatches, such as migrating from a 9 cm ground sampling distance (GSD) to a 5 cm GSD, disrupt the learned cross-modal semantic alignment, leading to catastrophic performance degradation on unseen target domains [9].

A more insidious challenge arises from the unconstrained, non-standard nature of real-world user queries. Human linguistic habits exhibit substantial variation. Under urgent operational constraints, users favor brief, macro-level descriptions restricted to basic categories and scales. For example, the client will use “the small cars” to find all cars scattered in the remote sensing image. Conversely, when conditions are loose, users articulate highly intricate sentences featuring multiple topological relations to disambiguate dense clusters of similar objects. This severe fluctuation in linguistic granularity triggers a drift in text logic across domains. Existing RRSIS benchmarks fail to capture this vulnerability because they rely on rigid, template-based generation or provide only a single textual level, leaving models ill-equipped to handle the dual drift in visual and linguistic features [8], [10], [11].

To address these limitations, we introduce the first crossdomain RRSIS benchmark, the Vaihingen-Potsdam Referring (VPRef) dataset. Transcending single-domain constraints, VPRef bridges two classic remote sensing domains, ISPRS Vaihingen and ISPRS Potsdam [12], which exhibit distinct spectral profiles and a pronounced shift in spatial resolution.

The dataset comprises 10,377 high-resolution image patches paired with 46,972 high-quality language annotations. To replicate authentic, non-standard user inputs, VPRef replaces monotonous template configurations with a metadata-driven pipeline assisted by advanced large language models (LLMs). This approach yields a three-tier hierarchy of linguistic granularity: Basic, Moderate, and Complex. The Basic tier omits relative positions and focuses on object attributes and scale; the Moderate tier introduces first-order spatial relationships; the Complex tier integrates multi-layered geometric constraints, target occlusion states, and chained topological reasoning. As analyzed in Table I, VPRef provides a rigorous diagnostic tool to evaluate model resilience against multi-dimensional domain shifts.

Utilizing the VPRef benchmark, we execute an extensive empirical evaluation across prominent RRSIS architectures, revealing a performance gap of up to 19.73% under spatiallogic transitions and 47.88% under textual-logic transitions. To address this performance drop, we present a tailored domain adaptation baseline designed to counteract both visual and textual logic shifts. For visual domain drift, we first introduce and evaluate a strong semantic baseline based on a fully fine-tuned Segment Anything Model (SAM3). Then, we integrate a selftraining mechanism that propagates high-confidence pseudolabels from the target domain back into the source training loop. To address textual domain drift, we concatenate multigranularity text prompts and use a random sampling strategy during training, simulating variance in heterogeneous user inputs. Experimental results show that these joint strategies effectively mitigate dual drift, establishing a robust baseline for future cross-domain RRSIS research.

In summary, the primary contributions of this work are threefold:

1) We establish the VPRef dataset, the first cross-domain benchmark for RRSIS. The dataset provides 46,972 high-quality language-image-annotation triplets structured across a three-tier linguistic hierarchy to accurately reflect unconstrained user commands.

2) We develop a domain adaptation baseline to mitigate visual and textual drifts. The framework counteracts visual drift through self-training and addresses textual drift via random sampling from concatenated, multigranularity text prompts.

3) We provide the first systematic analysis showing a decoupling between the semantic and visual domains and highlight limitations of current models under crossdomain scenarios.

The remainder of this paper is organized as follows. Section II reviews related work on referring image segmentation datasets and methods. Section III details the construction, properties, and hierarchical annotation pipeline of the proposed VPRef dataset. Section IV describes the proposed domain adaptation framework, detailing the self-training mechanism and the multi-granularity text sampling strategy. Section V presents the experimental setup and quantitative and qualitative benchmark results. Finally, Section VI concludes this paper.

## II. RELATED WORK

## A. Referring Image Segmentation

Referring image segmentation originated in the natural image domain, where the objective is to segment a specific target given an arbitrary natural language expression. Early paradigms established benchmarks such as RefCOCO, RefCOCO+ [13], and RefCOCOg [14], utilizing recurrent neural networks [15] or cross-modal attention mechanisms [16], [17] to fuse visual and textual features. The field shifted significantly with the advent of the Vision Transformer (ViT). For instance, Language-Aware Vision Transformer (LAVT) [7] showed that early-stage multi-modal fusion within the vision backbone yields superior cross-modal alignment compared to late-fusion architectures. Other developments, including scale-aware attention and query-based detectors [18], further enhanced the suppression of linguistic noise and improved boundary localization.

Transitioning this paradigm to Earth observation led to the development of RRSIS. Remote sensing imagery introduces unique challenges, including overhead perspectives, extreme scale variations, and dense target distributions. To capture these characteristics, early benchmarks like RefSegRS [5] and RRSIS-D [8] introduced template-based and manual textual descriptions for high-resolution imagery. To handle the arbitrary orientations of geographical features, architectures such as the Rotated Multi-Scale Interaction Network (RMSIN) [8] incorporate oriented bounding mechanisms and multiscale attention to align text with rotated targets. More recent large-scale benchmarks, such as RISBench [10] and NWPU-Refer [11], expand the diversity of geographic scenes and feature granularities. However, existing RRSIS frameworks assume that the training and testing datasets originate from the same domain and possess consistent referring expressions. Consequently, these models lack the mechanisms required to handle structural and spectral discrepancies when deployed across heterogeneous geographic regions.

## B. Domain Adaptation in Remote Sensing Imagery

Domain Adaptation (DA) addresses the performance degradation caused by domain shift, which occurs when a model trained on a source dataset is then deployed on an unannotated target dataset [19]. In remote sensing semantic segmentation, domain shifts arise from variations in sensor specifications, illumination conditions, atmospheric effects, and regional architectural styles [20]. Early remote sensing DA methods relied heavily on adversarial learning to align global feature distributions between domains, either at the input pixel level or within intermediate latent spaces [21]. Albeit effective for coarse style transfer, generic domain-minimax games frequently disrupt fine-grained class boundaries and exhibit volatile gradient optimization when coupled with foundation models [22], [23]. Consequently, the research paradigm has fundamentally transitioned toward target-domain pseudo-annotation mining via iterative self-training loops.

To counteract pseudo-label noise under severe domain shifts, recent self-training frameworks incorporate advanced distribution alignment and denoising mechanisms. For instance, Liang et al. [24] utilize probability-consistent selftraining paired with batch nuclear-norm maximization to align features while preserving prediction diversity. Similarly, Liang et al. [25] integrate adversarial learning with self-training, leveraging a conditional adversarial loss to filter target-domain prediction noise and stabilize classifier adaptation. These methods iteratively select high-confidence model predictions on unannotated target imagery to optimize the network via supervised objectives. To mitigate the accumulation of confirmation bias and label noise caused by spectral or resolution mismatches, researchers employ uncertainty estimation, classbalanced thresholds, and teacher-student consistency regularizations [6], [26].

In addition, the LoveDA [6] dataset established a benchmark for evaluating these frameworks in remote sensing across distinct urban and rural landscapes, while “Cross-city matters” [9] highlighted visual domain gaps across different metropolitan areas. Despite these advancements in pure computer vision tasks, conventional remote sensing DA methods and datasets remain limited to single-modality settings. They are unequipped to handle multi-modal RRSIS tasks, where visual domain drift couples with textual logic drift to disrupt crossmodal semantic alignment.

C. Efficiency and Parameter Scaling in Multimodal Vision Models

The rapid scaling of vision-language foundation models has introduced a trade-off between reasoning capability and computational efficiency. Autoregressive vision-language models and modern reasoning segmentation frameworks leverage multi-billion-parameter backbones to interpret complex, implicit queries. Although these large-scale models exhibit remarkable zero-shot generalization, their parameter scaling introduces massive computational overhead [27], [28]. This requirement creates a substantial barrier for remote sensing applications, which often demand rapid processing, onboard satellite execution, or deployment on edge devices with constrained memory and power resources. Furthermore, executing unsupervised domain adaptation loops on models with billions of parameters is structurally unstable and computationally restrictive. To balance performance with deployability, parameter-efficient fine-tuning (PEFT) and specialized foundational backbones have emerged as viable alternatives [29].

The SAM series demonstrates strong zero-shot localization capabilities by utilizing promptable visual features [30]. The recent SAM3 architecture integrates explicit concept representations, allowing the model to bridge low-level geometry with high-level semantic abstractions [31]. By utilizing parameter-efficient transfer learning techniques such as Low-Rank Adaptation (LoRA), researchers can adapt these foundation models to specific downstream tasks without updating the entire parameter space [32]. This study focuses on referring segmentation rather than generative-reasoning segmentation to ensure our framework remains highly parameter-efficient. This structural choice enables the integration of multi-tiered linguistic consistency constraints to stabilize cross-domain adaptation while maintaining a lightweight computational footprint suitable for Earth observation applications.

## III. THE VPREF DATASET

To systematically investigate the dual drift of visual and textual logic in Earth observation, we introduce the VPRef dataset. Transcending the single-domain boundaries of prior RRSIS benchmarks, VPRef provides a paired, multi-tiered linguistic evaluation platform spanning two distinct geographic domains. As summarized in Table I, the dataset comprises 46,972 high-quality language-image-annotation triplets, establishing a standardized baseline for cross-domain RRSIS research.

## A. Data Source and Domain Specifications

The visual components of VPRef are derived from the classic ISPRS Vaihingen and ISPRS Potsdam aerial frameworks. Both domains utilize near-infrared, red, and green (IRRG) spectral band configurations. For the source domain, designated as VaiRef, we harvest 3,125 image patches of uniform size 480 × 480 pixels from the Vaihingen tiles, using the official high-quality semantic labels to ensure absolute mask fidelity. For the target domain, we construct PotsRef using 7,252 image patches, each cropped to 800 × 800 pixels, extracted from the Potsdam tiles. Crucially, the two domains exhibit distinct geometric and spatial layout properties. VaiRef features a GSD of 9 cm, while PotsRef has a finer GSD of 5 cm. This structural difference creates an explicit cross-spatial resolution mismatch, forcing models to adapt across different pixel scales, object-density profiles, and sensory footprints.

## B. Metadata Extraction and Hierarchical Text Generation

To build a high-quality multimodal dataset without manual text-writing fatigue or template monotony, we implement a metadata-driven pipeline powered by advanced generative models. The text generation process executes in two sequential stages: structural attribute mining and language synthesis. First, we develop a geometric extraction routine to parse the ground-truth semantic masks of each image patch. This script isolates connected components for five target categories: impervious surfaces, buildings, cars, trees, and low vegetation. For each isolated instance or collective cluster, the routine computes explicit spatial attributes, including normalized bounding box coordinates, global centroids, aspect ratios, area ratios, and boundary truncation states. Because the underlying imagery relies on IRRG band configurations rather than standard RGB channels, we explicitly omit color attributes from the metadata extraction to prevent cross-modal spectral mapping confusion. Second, we design a rule-based mapping system that projects these spatial attributes into natural language, such as determining building scale and the relative positions of land objects. The proposed semantic lifting pipeline decouples spatial masks from rigid numerical values, transforming shape and density metrics into granular, scale-invariant natural language primitives that effectively eliminate data leakage during cross-domain deployment. Third, the extracted structural metadata is passed to a specialized text generation script

(m) Target Domain Moderate

(n) Target Domain Complex

(k) Source Domain Complex

TABLE I  
COMPARATIVE ANALYSIS OF VPREF AND PRIOR RRSIS DATASETS.
<table><tr><td>Dataset</td><td>Text Levels</td><td>Domains</td><td>GSD (m)</td><td>Images</td><td>Annotations</td><td>Resolution</td><td>Mask Generation</td><td>Text Generation</td></tr><tr><td>RefSegRS [5]</td><td>1</td><td>1</td><td>0.13m</td><td>4420</td><td>4420</td><td>512</td><td>Manual</td><td>Template</td></tr><tr><td>RRSIS-D [8]</td><td>1</td><td>1</td><td>0.5m-30m</td><td>17402</td><td>17402</td><td>800</td><td>Semi-auto</td><td>Template</td></tr><tr><td>RISBench [10]</td><td>1</td><td>1</td><td>0.1m-30m</td><td>52472</td><td>52472</td><td>512</td><td>Semi-auto</td><td>Semi-auto(GPT-4V)</td></tr><tr><td>NWPU-Refer [11]</td><td>23</td><td>1</td><td>0.12m-0.5m</td><td>15003</td><td>49745</td><td>1024-2048</td><td>Manual</td><td>Manual</td></tr><tr><td>VPRef (Ours)</td><td></td><td>2</td><td>0.05-0.09</td><td>10377</td><td>46972</td><td>480-800</td><td>Manual</td><td>Semi-auto(Gemini 3.5)</td></tr></table>

![](images/041a86ef719f86e9d050692d45344cc7b64c830bf42383d0d69c05ffb9693101.jpg)  
Fig. 1. Illustration of the VPREF dataset, which contains three levels: Basic, Moderate, and Complex. The L1, L2, and L3 denote the Basic, Moderate, and Complex tiers of linguistic description.

![](images/f081243c5c18c1e37a432b272cc040ec228964a1d54dcaac54238f4bac0f55a9.jpg)  
(e) Source Domain Category Distribution

![](images/e84e8618595e02c1706fbc34b4b9b0721cafe08f35fb3f9236ee637252d190f5.jpg)  
(b)  
(f) Target Domain Category Distribution

![](images/51e71580c038019f3fd3aefa27fe5408edcdbcb75d60e311ca3ccef442ddbc36.jpg)

![](images/91d95a36023d71b814678ff8ddf1aef0a993d5de04680702af544ab7e1c61f46.jpg)

![](images/412f5d02be42ed0c8e501d65a7c17b6dd47d8af77f3ebe502480e91b8bb52f06.jpg)

![](images/d94d4f70faeb027e16b08e289e7b8c8985f68db661f06f8084bf2a548a551f2b.jpg)

![](images/fbd4e0747a34caaafb87307ab4c47d55d854072c4821d87952466b32871a7acf.jpg)  
(h) Target Domain Pixel Distribution

![](images/0ff6a918b8ea5613840eaece400f506d5795789619fe4eecdd3d6583a77fcec9.jpg)

![](images/14cc5f13d2599f7099d9df01d8b796f9892dd0e54ec71a136ae506209ccd8ce7.jpg)

![](images/758294197813f504dc2bdc58ad9711996d59d8b4228db710b1e3ae73f03b9b0c.jpg)

(c)  
![](images/b6b562b9ab2213e83a4d5074b73f9daf0b7ecddd8ccdfc33193b6676463306a6.jpg)

![](images/9cdad5186eda64fc3ea677a03c7057ac9ad2a8c0283586285fc4c4ae724b5123.jpg)

![](images/19e4d8f14e1a8c3f6e07bc67a6354312d923a270eebd62e023ea395e797b4e91.jpg)  
(r) Target Domain Basic

![](images/99f1d6577f4d31c8e3e1e9b0d156c7a3e8c043dee0279cd219348892b3020d64.jpg)

![](images/b230092acb4a448f5c6a02ec389387b2fddba10a6e212c9544b20d0bd877a07d.jpg)

![](images/37e27d5ab3b5bf0b93e4eef04f7b1f5db771543e27d6998354e47587e63bab4a.jpg)  
(s) Target Domain Moderate

![](images/f39a4e238cef756ecbb67e0f2aed4ec9a300058665aeb44ac49e88209ba00734.jpg)  
(t) Target Domain Complex

Fig. 2. Statistical analysis of the constructed VPRef dataset across the source and target domains. (a) Category and pixel proportion of five objects. (b) Distribution of word lengths of referring expressions. (c) Mean and standard deviation of spectral values. (d) Word clouds for referring expressions.

that interfaces with the Gemini 3.5 architecture. Rather than relying on rigid string concatenation, we implement a promptbased synthesis mechanism to ensure grammatical fluidity and stylistic variation. The script processes macro attributes to construct basic descriptions. Concurrently, it ingests advanced spatial relations and chained topological structures to generate more sophisticated expressions.

## C. Linguistic Granularity Tiers

To replicate the diverse, unconstrained inputs of real-world end-users, VPRef establishes a three-tier hierarchy of linguistic granularity. Each tier represents a distinct level of complexity in cross-modal reasoning, as illustrated by the visual examples in Fig. 1.

• Basic Tier: Designed to simulate direct, low-overhead user inputs. This level completely excludes relative positioning or environmental context. It focuses exclusively on target scale and the core category name, yielding lean expressions such as “a cluster of vehicles” or “minor lawn”.

• Moderate Tier: Represents conventional operational queries. This tier incorporates first-order absolute locations and primary contextual relations. It describes targets using a direct subject-first syntax, typically constraining the description length to 12–20 words, such as “a cluster of building sections occupying a major portion of the patch alongside the road intersection”.

• Complex Tier: Modelled after detailed, highly descriptive queries required for dense target disambiguation. This level integrates multiple chained topological relations, fine-grained shape geometries, and explicit boundary truncation states. The syntax varies dynamically across 22–32 words, forcing models to resolve multi-layered geometric constraints, such as “occupying a major portion of the patch, this cluster of typical building sections is partially shown alongside the road intersection and adjacent to trees”.

## D. Statistical Analysis and Dataset Characteristics

Deep cross-domain statistics reveal the severe domain gaps embedded within VPRef, as visualized across the comprehensive analytics in Fig. 2. The spectral value distributions, illustrated in Fig. 2(c), show pronounced shifts in pixel intensities between the Vaihingen and Potsdam environments, driven by variable seasonal flight times and sensor calibrations. The word clouds presented in Fig. 2(d) illustrate the linguistic shift across the granularity scale. The Basic tier relies heavily on simple quantifiers, while the Moderate and Complex tiers introduce a dense vocabulary of spatial prepositions, geometric descriptors, and directional terms. This linguistic progression, coupled with the resolution shift from 9 cm to 5 cm, makes VPRef a highly diagnostic benchmark for quantifying model resilience to simultaneous visual and text-logic drifts.

Besides, the image and language in the source and target domains also show a consistent trend. Figs. 2(e)-(f) highlight the land cover categories. The five objects are relatively evenly distributed in the dataset. In contrast, Figs. 2(g)-(h) demonstrate that the Car accounts for only a very small proportion of pixels of the dataset. This shows that, compared with other categories, the car is much smaller in scale, which may challenge the model’s ability to recognise small objects. Linguistically, the three tiers demonstrate clear structural separation. As depicted in Fig. 2(b), the sentence length of the Basic tier ranges from 2–10 words, the Moderate tier from 12–20 words, and the Complex tier from 22–32 words.

![](images/4d87f87edbe7733300eb51c60fda11e5eac8a0dffc2dfc594e461f9eda22a6b1.jpg)  
Fig. 3. Illustration of full LoRA Adaptation SAM 3 Architecture with selftraining and multi-granularity text prompt mixing.

## IV. METHODOLOGY

This section details the unified framework designed to counteract the dual drift of visual appearance and text logic in cross-domain RRSIS. We first present the overall framework overview, followed by the PEFT mechanism based on SAM3- LoRA. We then describe the generalized domain adaptation components, comprising pseudo-label-driven self-training and a multi-granularity text prompt mixing strategy, which are extended symmetrically across all evaluated baseline architectures to establish a universal adaptation benchmark. Finally, we formulate the cross-modal matching logic and joint loss optimization objective.

## A. Overall Framework Overview

The core objective of our methodology is to enable an RRSIS model trained on the source domain (VaiRef) to generalize seamlessly to an unannotated target domain (PotsRef). To address the simultaneous shifts in spatial-spectral visual features and in unconstrained user-command granularities, we establish a dual-drift adaptation pipeline, as illustrated in Fig. 3.

The overall forward pass operates on a multi-modal interactive pipeline. Given an input remote sensing image patch paired with a natural language query, the framework extracts concurrent feature streams through a visual backbone and a textual encoder. These disparate modalities are projected into a shared embedding space and fed into a multi-modal interaction transformer, which coordinates fine-grained cross-modal attention layers to align linguistic tokens with pixel-level spatial representations. A specialized segmentation head then decodes these aligned tokens to output coordinate bounding boxes and dense pixel masks.

Rather than treating domain adaptation as an isolated vision task, our framework views the structural variance of text prompts as a regulatory mechanism. The self-training pipeline handles visual distribution alignment by propagating target domain predictions, while the text prompt mixing strategy stabilizes cross-modal text alignment. Crucially, these two adaptation components are designed as generalized modules. They operate independently of the underlying network architecture, enabling uniform deployment across conventional RRSIS baselines such as LAVT [7] and RMSIN [8], as well as our advanced foundational baseline.

## B. Parameter-Efficient Fine-Tuning via SAM3-LoRA

To establish a strong semantic baseline that handles complex Earth observation categories across domains, we anchor our main architecture on SAM3. Due to the massive scale of foundational parameters, full-weight backpropagation risks catastrophic forgetting and training divergence under unsupervised domain adaptation loops. We mitigate this instability by implementing LoRA, restricting gradient updates to a lightweight set of intrinsic rank-decomposition matrices while freezing the global foundational weights.

As illustrated in the architectural breakdown in Fig. 3, we systematically insert LoRA layers into the network’s core components to achieve balanced multimodal adaptation. Within the visual backbone, we embed LoRA layers into the query, key, and value projection weights of the vision transformer attention blocks to capture regional landscape variations. Symmetrically, the text encoder incorporates LoRA layers within its multi-head attention layers to handle domainspecific geographic terminology. To align these adapted feature streams, we embed low-rank matrices into both the encoder and decoder structures of the multi-modal interaction transformer, forcing the cross-modal queries to adapt to scale transitions. Finally, the prediction and segmentation heads are augmented with LoRA to refine boundary localization.

Formally, for any targeted weight layer $W _ { 0 } \in \mathbb { R } ^ { d \times k }$ , the adapted forward pass bypasses full matrix modification by computing a parallel low-rank update. The modified weight matrix $W$ is expressed as:

$$
W = W _ { 0 } + \Delta W = W _ { 0 } + \frac { \alpha } { r } ( B \cdot A )\tag{1}
$$

where $B \in \mathbb { R } ^ { d \times r }$ and $A \in \mathbb { R } ^ { r \times k }$ represent the trainable lowrank matrices, r ≪ min $( d , k )$ denotes the intrinsic rank, and α is a constant scaling hyperparameter. During training, $W _ { 0 }$ remains strictly frozen, and only A and B receive gradient updates. This design reduces the trainable parameter footprint to a fraction of the total model volume, ensuring computational efficiency during cross-domain adaptation.

## C. Generalized Pseudo-Label Driven Self-Training

To bridge the visual domain gap driven by variable flight altitudes, sensor specifications, and GSD scaling, we implement an unsupervised self-training routine. This routine leverages the model optimized on the source domain to harvest latent spatial knowledge directly from the unannotated target imagery. The pipeline executes via a decoupled pseudolabel generation and joint training loop. In the generation stage, target domain image patches from PotsRef are passed through the initialized source-domain model along with their corresponding text descriptions. The model outputs predicted probability maps, which we convert into hard binary masks using a fixed confidence threshold. To ensure structural control during data routing and mask filtering, our data loader incorporates an absolute control interface that maps low-confidence or non-reference target scenes to valid all-zero control masks, preventing corrupted background noise from updating the segmentation head. During the joint-training stage, we append the target image, the generated target pseudo-mask, and the target text description directly to the source training batches. The corresponding target text query is synthesized via the offline metadata lifting pipeline without exposing target masks to the optimization loop. The data pipelines mix these streams into a unified training loop. The model minimizes a supervised objective using true ground truths for source samples and pseudo-labels for target samples, iteratively aligning the latent visual distributions without requiring manual target annotations.

## D. Multi-Granularity Text Prompt Mixing Strategy

While self-training addresses visual distribution drift, it remains vulnerable to text logic drift caused by unconstrained, variable user command granularities. To force the crossmodal alignment layer to build scale-invariant topological representations, we introduce a text prompt mixing strategy within the training loop. Our data pipeline exploits the multitiered linguistic hierarchy established in the VPRef dataset. As implemented in the dataset loading classes, each target groundtruth mask is mapped to a synchronized pool containing Basic, Moderate, and Complex textual variations. During training, rather than fixing the text input to a single level of detail, we activate a dynamic mixing variant. For each training instance within an epoch, the text merger module randomly samples a single description from the multi-granularity text pool. The selection follows a uniform discrete distribution:

$$
\mathcal { T } _ { \mathrm { i n p u t } } \sim \mathcal { T } _ { \mathrm { B a s i c } } , \mathcal { T } _ { \mathrm { M o d e r a t e } } , \mathcal { T } _ { \mathrm { C o m p l e x } }\tag{2}
$$

This epoch-level resampling acts as a text-space data augmentation. By decoupling a fixed visual mask from a singular sentence structure, the model cannot rely on specific keyword patterns or fixed phrase lengths. This forces the model to accurately locate the target object, regardless of whether the description is short or long, simple or complex, successfully mitigating textual logic drift under domain transitions.

## E. Cross-Modal Matching and Joint Loss Optimization

To optimize the network across both bounding box localization and dense pixel segmentation, we utilize a multi-task objective regulated by an assignment mechanism. Because our segmentation head generates a fixed set of prediction queries, we employ an optimized bipartite matching strategy [33]. This matcher computes an optimal one-to-one assignment between the predicted queries and the target reference labels by minimizing a joint matching cost containing classification confidence, bounding box coordinates, and Generalized Intersection over Union (GIoU) metrics. Once the optimal query assignments are established, the framework minimizes a composite joint loss function containing three distinct taskspecific constraints:

$$
L _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { b o x } } L _ { \mathrm { b o x } } + \lambda _ { \mathrm { c e } } L _ { \mathrm { c e } } + \lambda _ { \mathrm { m a s k } } L _ { \mathrm { m a s k } }\tag{3}
$$

where $\lambda _ { \mathrm { b o x } } , ~ \lambda _ { \mathrm { c e } }$ , and $\lambda _ { \mathrm { m a s k } }$ represent balancing hyperparameters. The bounding box localization loss, $L _ { \mathrm { { b o x } } }$ , combines a standard Smooth L1 regression objective with a GIoU loss to optimize coordinate boundaries against scale variations. The existence loss, $L _ { \mathrm { c e } }$ , evaluates query presence and classification confidence through a focal-style cross-entropy objective, penalizing false positives in background regions. The dense mask loss, $L _ { \mathrm { m a s k } }$ , integrates a binary focal loss with a dice loss to handle target pixel imbalance, ensuring crisp boundary definitions for both tiny objects like vehicles and extensive regions like residential buildings. This joint optimization loop forces the adapted multi-modal components to preserve finegrained spatial structures under severe cross-domain shifts.

TABLE II  
QUANTITATIVE EVALUATION OF MODEL RESILIENCE AGAINST TEXTUAL GRANULARITY SHIFTS AND CROSS-PROMPT MISMATCHES.
<table><tr><td></td><td>mIoU</td><td>oIoU</td><td>mIoU</td><td>oIoU</td><td>mIoU</td><td>oIoU</td><td>mIoU</td><td>oIoU</td></tr><tr><td></td><td></td><td>Basic→ Basic</td><td></td><td>Basic→ Moderate</td><td>Basic→</td><td>Complex</td><td>Mix→ Basic</td><td></td></tr><tr><td>LAVT [7]</td><td>78.81</td><td>73.08</td><td>40.81</td><td>43.05</td><td>40.04</td><td>41.3</td><td>78.16</td><td>72.35</td></tr><tr><td>RRSIS [5]</td><td>78.6</td><td>73.1</td><td>38.98</td><td>40.67</td><td>45.59</td><td>48.32</td><td>78.15</td><td>72.08</td></tr><tr><td>RMSIN [8]</td><td>78.84</td><td>73.48</td><td>41.43</td><td>47.48</td><td>43.45</td><td>47.61</td><td>78.32</td><td>72.41</td></tr><tr><td>SAM3 [31]</td><td>47.38</td><td>50.35</td><td>29.43</td><td>34.87</td><td>25.92</td><td>29.52</td><td>47.38</td><td>50.35</td></tr><tr><td>SAM3-ft</td><td>79.64</td><td>74.64</td><td>51.37</td><td>58.54</td><td>31.76</td><td>40.57</td><td>79.33</td><td>74.33</td></tr><tr><td></td><td></td><td>Moderate→ Moderate</td><td></td><td>Moderate→ Complex</td><td></td><td>Moderate→ Basic</td><td>Mix→ Moderate</td><td></td></tr><tr><td>LAVT [7]</td><td>78.68</td><td>73.2</td><td>76.03</td><td>71.28</td><td>65.25</td><td>61.26</td><td>78.27</td><td>72.38</td></tr><tr><td>RRSIS [5]</td><td>78.72</td><td>73.19</td><td>76.47</td><td>71.46</td><td>68.68</td><td>64.34</td><td>78.39</td><td>72.25</td></tr><tr><td>RMSIN [8]</td><td>78.9</td><td>73.11</td><td>77.31</td><td>71.72</td><td>74.91</td><td>67.53</td><td></td><td>72.52</td></tr><tr><td>SAM3 [31]</td><td>29.43</td><td>34.87</td><td>25.92</td><td>29.52</td><td>47.38</td><td>50.35</td><td>78.46</td><td></td></tr><tr><td>SAM3-ft</td><td>79.66</td><td>74.88</td><td>68.93</td><td>68.17</td><td>78</td><td></td><td>29.43</td><td>34.87</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>73.62</td><td>79.46</td><td>74.55</td></tr><tr><td>LAVT [7]</td><td>77.66</td><td>Complex→ Complex</td><td></td><td>Complex→ Moderate</td><td></td><td>Complex→ Basic</td><td>Mix→</td><td>Complex</td></tr><tr><td>RRSIS [5]</td><td></td><td>72.31</td><td>78.32</td><td>72.68</td><td>73.32</td><td>68.43</td><td>77.31</td><td>71.9</td></tr><tr><td></td><td>77.6</td><td>72.49</td><td>78.52</td><td>73.01</td><td>73.62</td><td>68.01</td><td>77.09</td><td>71.5</td></tr><tr><td>RMSIN [8]</td><td>77.4</td><td>72.34</td><td>78.45</td><td>72.87</td><td>76.2</td><td>70.45</td><td>77.63</td><td>71.95</td></tr><tr><td>SAM3 [31]</td><td>25.92</td><td>29.52</td><td>29.43</td><td>34.87</td><td>47.38</td><td>50.35</td><td>25.92</td><td>29.52</td></tr><tr><td>SAM3-ft</td><td>79.41</td><td>74.33</td><td>79.1</td><td>74.32</td><td>78.02</td><td>73.53</td><td>79.34</td><td>74.37</td></tr></table>

## V. EXPERIMENTS

## A. Experimental Settings

1) Evaluation Metrics: To evaluate model performance in cross-domain RRSIS, we define two primary pixel-level mask metrics: mean Intersection over Union (mIoU) and overall Intersection over Union (oIoU). Let N denote the total number of test samples. For the i-th sample, let $P _ { i }$ represent the predicted binary pixel mask and $G _ { i }$ represent the corresponding ground-truth reference mask. The mIoU metric computes the arithmetic mean of the intersection-overunion ratios calculated individually for each sample, assigning equal weight to each geographical target regardless of its pixel footprint:

$$
\mathrm { m I o U } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \frac { \left| P _ { i } \cap G _ { i } \right| } { \left| P _ { i } \cup G _ { i } \right| } }\tag{4}
$$

Concurrently, the oIoU metric aggregates the total intersecting pixels across the entire test distribution and divides them by the cumulative union area, effectively prioritizing large-scale geomorphic features such as extended industrial buildings or runways:

$$
{ \mathrm { o I o U } } = { \frac { \sum _ { i = 1 } ^ { N } \left| P _ { i } \cap G _ { i } \right| } { \sum _ { i = 1 } ^ { N } \left| P _ { i } \cup G _ { i } \right| } }\tag{5}
$$

2) Baselines and Reference Models: Our empirical framework assesses three prominent RRSIS architectures originally designed for single-domain scenarios: LAVT [7], RRSIS [5], and RMSIN [8]. To analyze the impact of large-scale parameters, we incorporate the SAM3 [31] under two distinct operating modes. First, we test the unmodified SAM3 configuration to establish an open-vocabulary reference baseline. Second, we integrate our low-rank parameter-efficient adaptation layout to yield the fine-tuned baseline, designated as SAM3-ft.

3) Implementation Details: We restrict adaptation gradients using a low-rank allocation with intrinsic rank $r \ = \ 8$ and scaling coefficient $\alpha ~ = ~ 1 6 ,$ , and apply a dropout rate of 0.5 to prevent feature co-adaptation [34]. Optimization is conducted using the AdamW solver with an initial learning rate of $5 \times 1 0 ^ { - 5 }$ , balanced by a weight decay parameter of 0.01. Training executes across a uniform batch size of 16, distributed symmetrically across two NVIDIA RTX 5000 GPUs. For the conventional baseline networks, we preserve the optimization schedulers, learning rates, and tokenization steps native to their original publications. We apply data augmentation uniformly across all architectures, including random horizontal and vertical coordinate reflections. To neutralize baseline spatial grid mismatches, all source domain tiles from VaiRef and target domain tiles from PotsRef are dynamically resized to a standardized input resolution of 480 × 480 pixels during batch loading.

## B. Main Quantitative Results and Cross-Domain Comparisons

1) Textual Granularity Shift and Robustness Analysis: Table II evaluates model sensitivity against discrepancies in user query structures by monitoring performance under textdomain transitions within the source dataset. Columns 2 and 3 document the in-distribution baseline scores, where models are trained and tested on matching textual tiers. Under these idealized conditions, the LAVT, RRSIS, and RMSIN demonstrate highly consistent and competitive performance, with mIoU scores clustering tightly around 77% to 78%. This stability stems from their architectural reliance on symmetrical crossmodal backbones that pair BERT textual representations with

Swin Transformer visual features, ensuring uniform crossmodal grounding when linguistic boundaries remain fixed. All models achieved the lowest segmentation performance on the Complex textual tier, indicating difficulty capturing complex semantic relationships among objects in remote sensing images. The untuned open-vocabulary SAM3 framework struggles significantly in this domain, likely due to spectral discrepancies between RGB and IRRG images. Conversely, our adapted SAM3-ft baseline establishes the highest performance ceiling, reaching an in-distribution peak of 79.66% mIoU on the Moderate tier.

Crucially, when subject to cross-textual granularity mismatches where training and testing complexities diverge, conventional models experience severe performance drops. As documented in the intermediate columns of Table II, initializing a model on the Basic tier and deploying it against higherorder spatial descriptors results in a substantial degradation in cross-modal alignment. For instance, when transitioning from Basic commands to Complex topological descriptions, the SAM3-ft architecture experiences a substantial performance decline of 47.88% mIoU, decreasing to 31.76%. This vulnerability confirms that models optimized on brief keyword configurations fail to resolve multi-layered geometric constraints or chained topological logic.

Our proposed multi-granularity text prompt mixing strategy effectively counteracts this structural vulnerability. The final columns in Table II illustrate performance when models are trained via random epoch-level mixing and evaluated against individual static text streams. This joint optimization strategy recovers the lost performance, securing metrics that sit remarkably close to the absolute in-distribution upper bounds. Specifically, under the mixed training paradigm, evaluation on the Basic tier yields 79.33% mIoU, only a negligible 0.31% deficit relative to the 79.64% upper bound. These results show that introducing dynamic linguistic boundaries during training forces the cross-modal interaction layers to learn scale-invariant spatial representations, insulating the model from variations in user input granularity.

TABLE III  
QUANTITATIVE GENERALIZATION ASSESSMENT AND PERFORMANCE GAPS UNDER CROSS-VISUAL DOMAINS.
<table><tr><td colspan="2"></td><td colspan="2">VaiRef test</td><td colspan="2">PotsRef val</td><td colspan="2">PotsRef test</td></tr><tr><td colspan="2"></td><td>mIoU</td><td>oIoU</td><td>mIoU</td><td>oIoU</td><td>mIoU</td><td>oIoU</td></tr><tr><td rowspan="5">Basic</td><td>LAVT</td><td>78.81</td><td>73.08</td><td>61.19</td><td>52.68</td><td>61.55</td><td>54.38</td></tr><tr><td>RRSIS</td><td>78.6</td><td>73.1</td><td>60.26</td><td>53.34</td><td>61.26</td><td>55.54</td></tr><tr><td>RMSIN</td><td>78.84</td><td>73.48</td><td>60.21</td><td>52.94</td><td>60.96</td><td>55.2</td></tr><tr><td>SAM3</td><td>47.38</td><td>50.35</td><td>43.9</td><td>48.15</td><td>45.43</td><td>49.38</td></tr><tr><td>SAM3-ft</td><td>79.64</td><td>74.64</td><td>70.45</td><td>65.64</td><td>71.53</td><td>67.72</td></tr><tr><td rowspan="5">Moderate</td><td>LAVT</td><td>78.68</td><td>73.2</td><td>62.43</td><td>54.34</td><td>59.23</td><td>52.82</td></tr><tr><td>RRSIS</td><td>78.72</td><td>73.19</td><td>60.53</td><td>52.98</td><td>58.99</td><td>52.71</td></tr><tr><td>RMSIN</td><td>78.9</td><td>73.11</td><td>60.93</td><td>52.31</td><td>59.07</td><td>51.33</td></tr><tr><td>SAM3</td><td>29.43</td><td>34.87</td><td>30</td><td>36.34</td><td>28.26</td><td>34.77</td></tr><tr><td>SAM3-ft</td><td>79.66</td><td>74.88</td><td>70.48</td><td>65.57</td><td>65.78</td><td>64.56</td></tr><tr><td rowspan="5">Complex</td><td>LAVT</td><td>77.66</td><td>72.31</td><td>60.3</td><td>51.16</td><td>59.17</td><td>52.35</td></tr><tr><td>RRSIS</td><td>77.6</td><td>72.49</td><td>60.73</td><td>53.49</td><td>61.22</td><td>55.08</td></tr><tr><td>RMSIN</td><td>77.4</td><td>72.34</td><td>59.81</td><td>52.28</td><td>60.01</td><td>53.99</td></tr><tr><td>SAM3</td><td>25.92</td><td>29.52</td><td>26.35</td><td>30.36</td><td>25.94</td><td>29.44</td></tr><tr><td>SAM3-ft</td><td>79.41</td><td>74.33</td><td>70.11</td><td>65.14</td><td>70.16</td><td>66.32</td></tr></table>

2) Visual Cross-Domain Generalization Gaps: Table III establishes a baseline diagnostic assessment of visual domain drift by evaluating source-trained models directly on the validation and testing partitions across all three text granularities. The empirical shifts quantify a pervasive performance gap across all evaluated architectures, revealing a severe generalization barrier when moving across geographic configurations. On the cross-scene task, conventional models show a steep performance decline. Under the Basic text configuration, LAVT degrades by 17.26% mIoU, dropping from a source test score of 78.81% to 61.55% on the target test set. This vulnerability worsens in the Moderate tier, where RRSIS target test performance drops to 58.99%, an absolute cross-domain gap of 19.73% mIoU. The untuned SAM3 baseline fails across all cross-domain configurations because it cannot resolve finegrained ground targets under shifting sensory footprints. Our SAM3-ft framework retains the highest absolute resilience to generalization across domains. SAM-ft hits 71.53% mIoU on the PotsRef test set of the Basic tier. It still exhibits an absolute performance drop of 8.11% mIoU relative to its source baseline.

This uniform decline confirms that variations in visual features and spatial-scale discrepancies distort learned textual alignment coordinates, necessitating dedicated domain adaptation mechanisms.

3) Unsupervised Domain Adaptation via Self-Training: Table IV isolates and quantifies the performance gains from integrating our generalised pseudo-label-driven self-training framework across different model architectures. The evaluation distinguishes between the baseline models and the self-training pipelines utilizing target-domain predictions (-ssv). We found that some predictions detect nothing, so we filter them out as the cleaner pseudo-labels variant (-ssvf). The empirical trends demonstrate that while self-training yields minor performance fluctuations on the source domain, it produces substantial and consistent improvements on the target domain. For example, on the PotsRef testing set, the self-training plugin elevates LAVT’s baseline performance from 59.23% to 60.89% mIoU, while the variant LAVT-ssvf drives the target metric to a peak of 63.09% mIoU, representing a total absolute recovery of 3.86% mIoU. Our parameter-efficient foundation model baseline capitalizes on this self-training loop. The SAM3-ftssv pipeline increases the PotsRef testing metric from 65.78% to 67.27% mIoU. By filtering out-of-distribution label noise, the framework achieves an absolute target peak of 67.46% mIoU.

4) Ablative Cross-Analysis of Foundation Model: Table V summarises the empirical results for the SAM3-ft baseline across various combinations of the multi-granularity text mixing strategy and pseudo-label-driven self-training. Rather than a single optimal configuration, the distribution of peak metrics across variants reveals a structural decoupling between crossmodal semantic robustification and visual domain alignment.

Enriching the textual prompt space independently yields a profound regularizing effect on target-domain generalization. As shown in row 5 (SAM3-ft-mix) of Table V, deploying the multi-granularity mixing strategy without self-training enables the framework to achieve the highest instance-level segmentation accuracy on the unseen target test partition, reaching an absolute peak of 69.76% mIoU. Because the dynamic mixing mechanism decouples a fixed spatial mask from a singular sentence length or syntactic template during training, it prevents the cross-modal transformer from exploiting localized linguistic shortcuts native to the source domain. The model is forced to distill invariant geometric and topological concepts, successfully insulating instance-level localization against text logic drift.

TABLE IV  
PERFORMANCE GAINS OF GENERALIZED UNSUPERVISED SELF-TRAINING FRAMEWORKS.
<table><tr><td colspan="2">VaiRef test</td><td colspan="2">PotsRef val</td><td colspan="2">PotsRef test</td></tr><tr><td></td><td>mIoU oIoU</td><td>mIoU</td><td>oIoU</td><td>mIoU</td><td>oIoU</td></tr><tr><td>LAVT</td><td>78.68</td><td>73.2</td><td>62.43</td><td>54.34</td><td>59.23 52.82</td></tr><tr><td>LAVT-ssv</td><td>78.67</td><td>72.78 64.14</td><td>56.25</td><td>60.89</td><td>54.5</td></tr><tr><td>LAVT-ssvf</td><td>78.57</td><td>72.6 63.78</td><td>56.24 52.98</td><td>63.09</td><td>56.81</td></tr><tr><td>RRSIS</td><td>78.72</td><td>73.19</td><td>60.53</td><td>58.99</td><td>52.71</td></tr><tr><td>RRSIS-ssV</td><td>78.7</td><td>73.01</td><td>61.91</td><td>54.77 63.1</td><td>56.58</td></tr><tr><td>RRSIS-ssvf</td><td>78.75</td><td>73.19</td><td>62.02</td><td>54.67 60.15</td><td>53.64</td></tr><tr><td>RMSIN</td><td>78.9</td><td>73.11</td><td>60.93</td><td>52.31 59.07</td><td>51.33</td></tr><tr><td>RMSIN-ssv</td><td>78.8</td><td>72.86</td><td>61.96</td><td>53.75 61.03</td><td>54.02</td></tr><tr><td>RMSIN-ssvf</td><td>78.65</td><td>72.66</td><td>62.3</td><td>54.23 63.54</td><td>56.26</td></tr><tr><td>SAM3</td><td>29.43</td><td>34.87</td><td>30</td><td>36.34</td><td>28.26 34.77</td></tr><tr><td>SAM3-ft</td><td>79.66</td><td>74.88</td><td>70.48</td><td>65.57 65.78</td><td>64.56</td></tr><tr><td>SAM3-ft-ssv</td><td>79.55</td><td>74.63</td><td>70.85</td><td>66.11 67.27</td><td>65.54</td></tr><tr><td>SAM3-ft-ssvf</td><td>80.01</td><td>75.15</td><td>71.15</td><td>66.22 67.46</td><td>65.98</td></tr></table>

TABLE V

ABLATION STUDY OF THE FOUNDATION MODEL FRAMEWORK INTEGRATING MULTI-GRANULARITY TEXT MIXING AND PSEUDO-LABEL SELF-TRAINING.
<table><tr><td rowspan="2"></td><td colspan="2">VaiRef test</td><td colspan="2">PotsRef val</td><td colspan="2">PotsRef test</td></tr><tr><td>mIoU</td><td>oIoU</td><td>mIoU</td><td>oIoU</td><td>mIoU</td><td>oIoU</td></tr><tr><td>SAM3-ft</td><td>79.66</td><td>74.88</td><td>70.48</td><td>65.57</td><td>65.78</td><td>64.56</td></tr><tr><td>SAM3-ft-ssv</td><td>79.55</td><td>74.63</td><td>70.85</td><td>66.11</td><td>67.27</td><td>65.54</td></tr><tr><td>SAM3-ft-ssvf</td><td>80.01</td><td>75.15</td><td>71.15</td><td>66.22</td><td>67.46</td><td>65.98</td></tr><tr><td>SAM3-ft-mix</td><td>79.46</td><td>74.55</td><td>70.04</td><td>65.31</td><td>69.76</td><td>66.45</td></tr><tr><td>SAM3-ft-mix-ssv</td><td>79.31</td><td>74.3</td><td>70.45</td><td>65.94</td><td>68.54</td><td>66.59</td></tr><tr><td>SAM3-ft-mix-ssvf</td><td>79.52</td><td>74.45</td><td>70.41</td><td>66.15</td><td>68.08</td><td>65.95</td></tr></table>

The self-training variant SAM3-ft-ssv introduces minor feature corruption on the source dataset due to the propagation of uncalibrated out-of-distribution predictions. With the pseudolabel filter, SAM3-ft-ssvf reverses this degradation, driving the source partition to an absolute maximum of 80.01% mIoU and 75.15% oIoU. This denoising mechanism also exhibits strong optimization stability on the target validation split.

When visual self-training couples with text-space randomization, the optimization trajectory shifts focus toward dominant geomorphic structures. The self-training variant, combined with text mixing (i.e., SAM3-ft-mix-ssv), achieves the highest cumulative pixel alignment on the target test partition, peaking at 66.59% oIoU. Unfiltered self-training loops naturally gravitate toward majority land-cover classes, such as expansive impervious surfaces and building footprints, which command the largest pixel areas. Simultaneously, the randomized text mixing layer acts as a data amplifier that robustifies the cross-modal boundaries of these dominant targets. This dual alignment forces the segmentation head to maximize overall pixel intersections for large-scale geographic elements, even if small-scale objects like vehicles experience marginal suppression. The resulting dispersion of peak metrics across -mix, -ssvf, and -mix-ssv underscores a fundamental tradeoff: linguistic variation drives fine-grained semantic invariance, whereas pseudo-label propagation governs macro-scale spatial grid alignment.

TABLE VI  
PARAMETER EFFICIENCY ANALYSIS ACROSS EVALUATED RRSIS FRAMEWORKS.
<table><tr><td></td><td>Visual Encoder</td><td>Text Encoder</td><td>Total parameters</td><td>Trainable parameters</td></tr><tr><td>LAVT</td><td>Swin-B</td><td>BERT</td><td>227.74M</td><td>227.74M</td></tr><tr><td>RRSIS</td><td>Swin-B</td><td>BERT</td><td>276.26M</td><td>276.26M</td></tr><tr><td>RMSIN</td><td>Swin-B</td><td>BERT</td><td>240.04M</td><td>240.04M</td></tr><tr><td>SAM3</td><td>ViT-H/16</td><td>CLIP-ViT-H-Text</td><td>849.76M</td><td>0M</td></tr><tr><td>SAM3-ft</td><td>ViT-H/16</td><td>CLIP-ViT-H-Text</td><td>849.76M</td><td>9.25M</td></tr></table>

5) Computational Complexity and Parameter Efficiency Analysis: Table VI provides a diagnostic assessment of the structural configurations and computational attributes of the evaluated frameworks on the VaiRef dataset, isolating the trade-offs between total capacity and parameter-efficient optimization scalability.

Conventional RRSIS models rely on identical sub-module combinations comprising a Swin-B vision backbone [35] coupled with a BERT text encoder [36]. While these networks maintain a moderately memory footprint, with total parameter counts ranging from 227.74M to 276.26M, they leverage full-parameter training pathways. Full-weight backpropagation across these dense cross-modal pipelines not only demands intensive hardware execution budgets but also risks triggering catastrophic forgetting of foundational alignment priors due to unrestricted gradient contamination across model weights.

Concurrently, the unmodified foundational SAM3 dramatically expands the semantic ceiling by hosting a high-capacity ViT-H/16 visual encoder [37] and a CLIP-ViT-H-Text [38] embedding layer, scaling the collective parameter count to 849.76M. Although freezing the entire configuration yields a zero trainable footprint, this static operational mode remains decoupled from downstream remote sensing distributions, leaving its open-vocabulary spaces under-utilized without local spatial calibration. SAM3-ft successfully bridges this operational divide. By strategically embedding low-rank decomposition matrices within the foundational streams, SAM3-ft retains the massive 849.76M representational capacity while restricting active gradient pathways to a lightweight fraction of just 9.25M parameters. This design limits the trainable parameter footprint to approximately 1.08% of the total model volume. Compared to conventional RRSIS baselines that require fullweight parameter backpropagation, SAM3-ft achieves superior segmentation performance while consuming a fraction of the trainable gradient budget. This balance confirms the efficiency and practical scalability of our framework for referring segmentation tasks.

## C. Qualitative Analysis and Visualizations

1) Comparative Visual Evaluation Across RRSIS Baselines: Fig. 4 and Fig. 5 present qualitative comparisons between our SAM3-ft baseline and conventional architectures across the VaiRef and PotsRef test sets. In the in-distribution VaiRef

![](images/b272b938933f395a33c8fc4bf3b0f6104b78775c1ddec842a0674cc93e4072e3.jpg)  
An extensive road intersection diagonally traversing from top-left to bottom-right, bordering a large building section

Fig. 4. Qualitative segmentation comparisons on the in-distribution VaiRef test set. The red bounding boxes highlight the superior multi-modal grounding capability of SAM3-ft in preserving continuous tree topologies (row 1), crisp structural margins (row 2), and complete small-object details (rows 3–4) relative to full-parameter baselines.  
![](images/26b3eda511e8eab03a89c0dd7d31ff62210ba1cc91560337ac7dd77acc15719f.jpg)  
A substantial road intersection traverses diagonally from the top-left to the bottom-right, bordering the buildings.

Fig. 5. Qualitative segmentation comparisons under cross-domain deployment on the PotsRef test set against spatial-spectral drifts.

domain, the qualitative sequences in Fig. 4 confirm SAM3- ft’s superior multimodal alignment capability across diverse land-cover classes. In the first row, highlighted by the red bounding box, SAM3-ft isolates the continuous, intact topology of the targeted paved street, whereas the masks from LAVT, RRSIS, and RMSIN are visibly fragmented. For the residential structure in the second row, conventional RRSIS models yield blurred, imprecise boundaries, whereas SAM3-ft precisely traces the building’s margins to match the ground truth. Furthermore, SAM3-ft resolves target instances with higher completeness in micro-scale contexts, such as the vehicles in the third row, and accurately distills intricate finegrained details in complex mixed structures shown in the fourth row.

When deployed under cross-domain conditions on the PotsRef test set in Fig. 5, all networks show noticeable visual degradation due to cross-spatial-resolution mismatches and spectral variations. Conventional frameworks display significant spatial misalignment, with RRSIS failing to segment large, contiguous target regions. Symmetrically, LAVT and RMSIN suffer from severe target omissions and boundary erosion. In contrast, SAM3-ft exhibits the highest visual resilience, consistently generating the most stable, complete, and accurate mask structures among all evaluated models.

2) Linguistic Granularity Analysis: We evaluate the baseline SAM3-ft architecture on the VaiRef test set under an explicit cross-prompt granularity mismatch, keeping the model weights optimized on the Basic tier while systematically elevating the test prompts to Moderate and Complex descriptions.

As illustrated in Fig. 6, this configuration reveals sensitivity to semantic interference and keyword distraction in long-tailed clauses. In the first row, where the target reference is an impervious paved surface, the model successfully segments the target under the Moderate prompt. However, under the Complex prompt, which introduces a long description with multiple secondary geomorphic objects, the cross-modal alignment mechanism becomes less effective. Instead of isolating the paved surface, the segmentation head completely shifts its focus and mistakenly segments the “detached house” mentioned in the relative clause.

A parallel breakdown occurs in the second target sequence, which involves impervious surface extraction. When transitioning to the Complex textual query, the cross-modal transformer fails to maintain semantic hierarchy and misinterprets the structural description, incorrectly outputting the adjacent building structure instead of the road footprint. This error pattern confirms that training an RRSIS model solely on brief keyword strings restricts its cross-modal attention field. Without dynamic text-space regularization during training, the network fails to construct an operational hierarchy for multilayered sentences.

3) Error Profile Diagnostics and Annotation Noise Resilience: Fig. 7 visually diagnoses the boundary conditions and inherent limits of pure spectral-textual cross-grounding, while highlighting our framework’s resilience to dataset annotation noise. The qualitative sequences reveal distinct insights regarding elevation ambiguity and label corruption within the VaiRef test partition.

The first two rows illustrate a structural constraint stemming from the absence of three-dimensional geometric data. When tasked with separating tall tree canopies from low vegetation variants under identical spectral zones, the crossmodal interaction layers exhibit localized height ambiguity. As highlighted by the red bounding box in the first row, the network misidentifies shrubbery as low-lying lawn due to their highly overlapping multi-spectral signatures on the two-dimensional grid. Symmetrically, in the second row, the model omits a lower-center tree cluster. These persistent discrepancies demonstrate that relying exclusively on twodimensional appearance cues limits the network’s capacity to resolve height-dependent vegetation tiers, confirming that incorporating auxiliary elevation models remains essential for complete structural disambiguation.

![](images/875f498983d6281a19b795e67551b0210b3eab95a8073f93d8ccb324cf3b461a.jpg)  
Moderate : Extensive partially shown paved surface occupies a major portion of the patch adjacent to the wooded patch. Complex: A typical extensive partially shown paved surface occupies a major portion of the patch, bordering the detached house and adjacent to the wooded patch.

![](images/7da03fe6fb0cf31c369bf194c95892da5b7d01117bf1e9df92b4275ff487e89c.jpg)

![](images/f0232a0a7df48a7b7c3bb321f38ee93e61bd7aaf4a5975dd363cd518a551580d.jpg)  
road intersection. Moderate : An extensive road intersection occupying a major portion of the patch adjacent to the lawn. Complex : This extensive, typical road intersection is partially shown occupying most of the patch, bordering the detached house while adjacent to the lawn.  
Basic : A single detached house. Moderate : A single partially shown detached house in the middle and bottom zones is adjacent to the wooded patch. Complex : Extending to the left edge, this typical single detached house is partially shown across the middle and bottom zones, adjacent to the paved surface and wooded patch.

![](images/32c684863b3d117b7da2b312294295107c2e26be0e34fee623b4f72dfc0099f3.jpg)  
Basic : A small group of detached house Moderate : A small group of partially shown detached houses scattered across the tile alongside the road intersection. Complex : Scattered across all zones and sectors, this small group of typical partially shown detached houses sits alongside the road intersection and adjacent to the lawn.

Fig. 6. Qualitative breakdown of cross-linguistic granularity inference on the VaiRef dataset at high levels of textual complexity.  
![](images/16d647bca6222e235aef02d7f7b1348c83b1b112699d4f5fa0507e6d53fb93a8.jpg)  
Minor partially shown lawn scattered across all sectors alongside the road intersection.

Fig. 7. Diagnostic error profiles and resilience to annotation-noise evaluations within the VaiRef test partition.

Conversely, the third and fourth rows demonstrate a compelling instance of resilience to annotation noise, in which our fine-tuned baseline corrects systematic errors in the manual ground-truth annotations. In the third row, the true reference target under deep shadow comprises shrubbery, yet the original dataset label shows potential annotation ambiguity and categorizes this zone as tree canopy. Rather than blindly overfitting to this corrupted label distribution, these models suppress this localized label noise. In the fourth row, the baseline models correctly isolate the shrubbery boundaries and align perfectly with the structural query. This capacity to successfully bypass annotation artifacts demonstrates that the proposed framework does not merely memorize localized pixel-text associations, but instead distills transferable, structurally sound geomorphic

concepts.

## VI. CONCLUSION

In this paper, we introduce VPRef, the first cross-domain benchmark dataset for referring remote sensing image segmentation, specifically designed to assess model resilience to concurrent visual domain drift and textual logic drift. Symmetrically, we developed a generalized parameter-efficient domain adaptation baseline that integrates pseudo-label-driven self-training with a multi-granularity text prompt mixing strategy within a lightweight SAM3-LoRA optimization loop. Quantitative and qualitative cross-evaluations demonstrate that our framework successfully recovers from severe cross-city performance drops, verifying that PEFT can stabilize multimodal alignment coordinates under unknown target distributions. Despite these achievements, a key structural limitation of our current framework stems from the exclusive reliance on two-dimensional appearance cues; without auxiliary threedimensional elevation information, the cross-modal interaction layers exhibit localized height ambiguity when separating tall tree canopies from spectrally overlapping low-lying vegetation. Future investigations will incorporate multi-source elevation data to resolve vertical geometric structures and explore source-free domain adaptation protocols to further enhance multi-modal deployment security in unconstrained Earth observation scenarios.

## REFERENCES

[1] Q. Liu, T. Huang, Y. Dong, J. Yang, and W. Xiang, “From pixels to images: Deep learning advances in remote sensing image semantic segmentation,” arXiv preprint arXiv:2505.15147, 2025.

[2] J. Yuan, X. Ma, Z. Zhang, Q. Xu, G. Han, S. Li, W. Gong, F. Liu, and X. Cai, “Effc-net: lightweight fully convolutional neural networks in remote sensing disaster images,” Geo-Spatial Information Science, vol. 28, no. 1, pp. 212–223, 2025.

[3] Q. Liu, Y. Dong, Y. Zhang, and H. Luo, “A fast dynamic graph convolutional network and cnn parallel network for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–15, 2022.

[4] Y. Dong, Q. Liu, B. Du, and L. Zhang, “Weighted feature fusion of convolutional neural network and graph attention network for hyperspectral image classification,” IEEE Transactions on Image Processing, vol. 31, pp. 1559–1572, 2022.

[5] Z. Yuan, L. Mou, Y. Hua, and X. X. Zhu, “Rrsis: Referring remote sensing image segmentation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–12, 2024.

[6] J. Wang, Z. Zheng, A. Ma, X. Lu, and Y. Zhong, “Loveda: A remote sensing land-cover dataset for domain adaptive semantic segmentation,” arXiv preprint arXiv:2110.08733, 2021.

[7] Z. Yang, J. Wang, Y. Tang, K. Chen, H. Zhao, and P. H. Torr, “Lavt: Language-aware vision transformer for referring image segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 18 155–18 165.

[8] S. Liu, Y. Ma, X. Zhang, H. Wang, J. Ji, X. Sun, and R. Ji, “Rotated multi-scale interaction network for referring remote sensing image segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 658–26 668.

[9] D. Hong, B. Zhang, H. Li, Y. Li, J. Yao, C. Li, M. Werner, J. Chanussot, A. Zipf, and X. X. Zhu, “Cross-city matters: A multimodal remote sensing benchmark dataset for cross-city semantic segmentation using high-resolution domain adaptation networks,” Remote Sensing of Environment, vol. 299, p. 113856, 2023.

[10] Z. Dong, Y. Sun, T. Liu, W. Zuo, and Y. Gu, “Cross-modal bidirectional interaction model for referring remote sensing image segmentation,” arXiv preprint arXiv:2410.08613, 2024.

[11] Z. Yang, H. Yao, L. Tian, X. Zhao, Q. Li, and Q. Wang, “A large-scale referring remote sensing image segmentation dataset and benchmark,” arXiv preprint arXiv:2506.03583, 2025.

[12] F. Rottensteiner, G. Sohn, M. Gerke, J. D. Wegner, U. Breitkopf, and J. Jung, “Results of the isprs benchmark on urban object detection and 3d building reconstruction,” ISPRS journal ofphotogrammetry and remote sensing, vol. 93, pp. 256–271, 2014.

[13] S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg, “ReferItGame: Referring to objects in photographs of natural scenes,” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), A. Moschitti, B. Pang, and W. Daelemans, Eds. Doha, Qatar: Association for Computational Linguistics, Oct. 2014, pp. 787–798. [Online]. Available: https://aclanthology.org/D14-1086/

[14] J. Mao, J. Huang, A. Toshev, O. Camburu, A. L. Yuille, and K. Murphy, “Generation and comprehension of unambiguous object descriptions,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

[15] R. Li, K. Li, Y.-C. Kuo, M. Shu, X. Qi, X. Shen, and J. Jia, “Referring image segmentation via recurrent refinement networks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[16] Z. Hu, G. Feng, J. Sun, L. Zhang, and H. Lu, “Bi-directional relationship inferring network for referring image segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 4424–4433.

[17] S. Liu, T. Hui, S. Huang, Y. Wei, B. Li, and G. Li, “Cross-modal progressive comprehension for referring segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 9, pp. 4761– 4775, 2021.

[18] J. Liu, H. Ding, Z. Cai, Y. Zhang, R. K. Satzoda, V. Mahadevan, and R. Manmatha, “Polyformer: Referring image segmentation as sequential polygon generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 18 653–18 663.

[19] J. Li, Z. Yu, Z. Du, L. Zhu, and H. T. Shen, “A comprehensive survey on source-free domain adaptation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5743–5762, 2024.

[20] J. Peng, Y. Huang, W. Sun, N. Chen, Y. Ning, and Q. Du, “Domain adaptation in remote sensing image classification: A survey,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 9842–9859, 2022.

[21] O. Tasar, Y. Tarabalka, A. Giros, P. Alliez, and S. Clerc, “Standardgan: Multi-source domain adaptation for semantic segmentation of very high resolution satellite images by data standardization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2020, pp. 192–193.

[22] Y.-H. Tsai, W.-C. Hung, S. Schulter, K. Sohn, M.-H. Yang, and M. Chandraker, “Learning to adapt structured output space for semantic segmentation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 7472–7481.

[23] Y. Zou, Z. Yu, B. Kumar, and J. Wang, “Unsupervised domain adaptation for semantic segmentation via class-balanced self-training,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 289–305.

[24] J. Liang, J. Yang, R. Liu, Q. Liu, and P. Zhu, “Spectral structureaware initialization and probability-consistent self-training for crossscene hyperspectral image classification,” IEEE Geoscience and Remote Sensing Letters, 2025.

[25] C. Liang, B. Cheng, B. Xiao, and Y. Dong, “Unsupervised domain adaptation for remote sensing image segmentation based on adversarial learning and self-training,” IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1–5, 2023.

[26] X. Luo, W. Chen, Z. Liang, L. Yang, S. Wang, and C. Li, “Crots: Cross-domain teacher–student learning for source-free domain adaptive semantic segmentation,” International Journal of Computer Vision, vol. 132, no. 1, pp. 20–39, 2024.

[27] J. Quenum, W.-H. Hsieh, T.-H. P. Wu, R. Gupta, T. Darrell, and D. Chan, “Lisat: Language-instructed segmentation assistant for satellite imagery,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[28] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, “Lisa: Reasoning segmentation via large language model,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 9579–9589.

[29] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for nlp,” in International conference on machine learning. PMLR, 2019, pp. 2790–2799.

[30] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.

[31] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, J. Lei, T. Ma, B. Guo, A. Kalla, M. Marks, J. Greer, M. Wang, P. Sun, R. Radle, T. Afouras,¨ E. Mavroudi, K. Xu, T.-H. Wu, Y. Zhou, L. Momeni, R. Hazra, S. Ding, S. Vaze, F. Porcher, F. Li, S. Li, A. Kamath, H. K. Cheng, P. Dollar, N. Ravi, K. Saenko, P. Zhang, and C. Feichtenhofer,´ “Sam 3: Segment anything with concepts,” 2026. [Online]. Available: https://arxiv.org/abs/2511.16719

[32] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” Iclr, vol. 1, no. 2, p. 3, 2022.

[33] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in European conference on computer vision. Springer, 2020, pp. 213– 229.

[34] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting,” The journal of machine learning research, vol. 15, no. 1, pp. 1929–1958, 2014.

[35] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10 012–10 022.

[36] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[37] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[38] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.