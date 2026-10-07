# MedCORE: Criteria-Grounded Clinical Reasoning for Interpretable Medical Image Diagnosis

m Khan<sup>1</sup> asim.khan@ku.ac.ae

<sup>1</sup> Deparment of Computer Science, Khalifa University, Abu Dhabi, UAE

<sup>2</sup> Department of Aerosapce Engineering, Khalifa University, Abu Dhabi, UAE

Dwarikanath Mahapatra<sup>1</sup> cdwarikanath.mahapatra@ku.ac.ae

## Abstract

Clinical diagnosis is inherently a structured reasoning process, yet existing deep learning models often bypass this structure by mapping image features directly to disease labels without explicitly interrogating the morphological and textural criteria that clinicians systematically evaluate. This limits diagnostic transparency and may compromise safe clinical deployment. We present MedCORE (Medical Criteria-Oriented Reasoning and Evidence), a structured diagnostic framework that operationalizes clinical reasoning within a vision-language architecture. For each input image, MedCORE decomposes the diagnostic process into clinically defined criteria, spatially localizes each criterion to diagnostically relevant image regions, encodes evidence through multi-scale representations that capture macro-structural and micro-textural pathological characteristics, and refines criterion representations using a Graph Attention Network that explicitly models inter-criteria dependencies. Criterion representations are further aligned with clinical text descriptors, reinforced through class-wise visual prototypes, and aggregated using uncertainty-calibrated weighting that proportionally discounts low-confidence diagnostic evidence. MedCORE is validated across three clinically heterogeneous imaging modalities, including dermoscopic lesion classification on ISIC 2018, breast ultrasound lesion characterization on BUSI, and diabetic retinopathy grading on IDRiD. Quantitatively, MedCORE achieves 89.2% accuracy, 85.7% macro-F1, and 96.4% AUC on ISIC 2018; 96.1% accuracy, 95.2% macro-F1, and 98.4% AUC on BUSI; and 84.3% accuracy, 80.2% macro-F1, and 92.8% AUC on IDRiD. These results demonstrate consistent improvements over strong CNN, transformer, biomedical vision-language, concept-based, and prototype-based baselines. Collectively, the findings establish that grounding diagnostic inference in structured clinical criteria yields models that are accurate, interpretable, uncertainty-aware, and better aligned with the evidence-based reasoning used in clinical practice.

## 1 Introduction

Medical image diagnosis is increasingly supported by deep visual models, yet the dominant formulation still treats diagnosis as a direct mapping from pixels to disease labels. This formulation has delivered strong classification performance, but it does not match the way clinicians usually justify a decision. A dermatologist does not simply assign a lesion category; she inspects asymmetry, border irregularity, colour variation, dermoscopic structures, vascular patterns and other clinically meaningful cues. Likewise, ultrasound and retinal diagnosis rely on structured visual evidence that is spatially localised, clinically named and often mutually dependent. For high-stakes clinical deployment, a diagnostic model should therefore expose not only what it predicts, but also which clinical criteria support the prediction, where those criteria are observed and how reliable each piece of evidence is.

Vision–language pre-training offers a natural route toward this goal. General models such as CLIP [22] and biomedical variants including GLoRIA [12], BioViL [3], MedKLIP [26], KAD [32], BiomedCLIP [31] and BiomedGPT [30] learn visual representations aligned with clinical or biomedical language. However, most of these methods rely on global image– text similarity, report-level supervision or weak region–word alignment. They may encode useful medical semantics, but the final classifier is often not forced to reason through explicit diagnostic criteria. Concept-based models address this limitation by inserting humaninterpretable variables between images and labels [14, 20, 23, 28, 29]. Recent medical approaches, including Explicd [9], MICA [2], AdaCBM [6] and Med-MICN [10], show that clinical concepts can improve explanation quality. Yet three gaps remain. First, criteria are commonly represented as text anchors or concept tokens without explicit criterion-specific spatial grounding. Second, criteria are often treated independently, although clinical signs are correlated. Third, aggregation usually ignores criterion-level uncertainty, even when visual evidence is ambiguous or corrupted by imaging artefacts.

We propose MedCORE, a criteria-grounded framework for interpretable medical image classification. Instead of predicting labels directly from global image features, Med-CORE decomposes diagnosis into clinically meaningful criteria. Each criterion is paired with a clinical text anchor encoded by the frozen BiomedCLIP text encoder, while a frozen BiomedCLIP visual backbone provides dense image tokens. Lightweight trainable modules then learn criterion-specific visual representations without updating the backbone. To make these representations spatially meaningful, learned criterion queries generate attention maps over image tokens, allowing each criterion to focus on diagnostically relevant regions without requiring spatial supervision. Multi-scale criterion encoding further captures both coarse morphological structure and fine local texture.

MedCORE further models diagnosis as structured evidence aggregation. A criterion dependency graph attention module [25] refines each criterion using related criteria, allowing clinically correlated signs to reinforce one another. A class-wise visual prototype memory maintains discriminative evidence for each criterion and disease class, encouraging compact same-class representations and better separation across classes. Finally, an uncertainty head estimates the reliability of each criterion and down-weights ambiguous evidence before the final prediction. The model is trained with a three-term objective: cross-entropy for diagnostic classification, vision–language alignment to clinical text anchors and prototype-based discriminative learning.

We evaluate MedCORE on ISIC2018 skin lesion classification [7], BUSI breast ultrasound classification [1], and diabetic-retinopathy grading using IDRiD/RetinaMNIST [21, 27]. These benchmarks span dermoscopy, ultrasound and retinal fundus imaging, enabling us to assess whether explicit clinical criteria, spatial grounding, relational reasoning, prototype reinforcement and uncertainty-calibrated aggregation improve both diagnostic performance and interpretability across heterogeneous imaging modalities.

Our main contributions are:

1. We formulate medical image diagnosis as grounded clinical-criteria reasoning, enabling evidence-level interpretation across dermoscopy, breast ultrasound and retinal fundus imaging.

2. We introduce criterion-aware spatial localisation with multi-scale criterion encoding, allowing each clinical criterion to attend to diagnostically relevant image regions and visual scales.

3. We model inter-criteria dependencies using graph attention and reinforce class discriminative evidence through criterion-wise visual prototype learning.

4. We propose uncertainty-calibrated criterion aggregation, reducing the influence of unreliable or ambiguous criteria in the final diagnostic decision.

5. We evaluate MedCORE across three clinically heterogeneous medical image benchmarks using quantitative comparison, qualitative visualisation and ablation analysis.

## 2 Related Work

Medical vision language learning. Large-scale vision language pre-training has become a strong foundation for medical image analysis. GLoRIA [12] introduced global–local alignment between radiographs and reports for label-efficient recognition, while BioViL [3] exploited radiology text semantics to improve biomedical representation learning. Med-KLIP [26] and KAD [32] inject medical knowledge into image–text pre-training, improving disease recognition and entity-level grounding in chest radiology. BiomedCLIP and BiomedGPT further extend biomedical vision language learning to broader image text corpora and diverse biomedical tasks [30, 31]. These works provide powerful backbones, but their explanations are typically derived from global similarity, report tokens or disease entities. MedCORE uses a biomedical vision–language backbone differently: clinical text anchors define explicit diagnostic criteria, while visual evidence is spatially grounded, relationally refined and uncertainty-weighted before classification.

Concept-based interpretability and medical concept learning. Concept bottleneck models make predictions through human-understandable intermediate variables [14]. To reduce dependence on expensive concept annotations, recent post-hoc, label-free and languageguided CBMs use learned representations or language models to construct concept spaces [20, 23, 28, 29]. In medical imaging, Explicd aligns visual concept tokens with diagnostic knowledge, MICA performs multi-level image–concept alignment for skin lesions, AdaCBM adapts CLIP-based CBMs for diagnosis, and Med-MICN aligns concept semantics with neural-symbolic and saliency explanations [2, 6, 9, 10]. These methods show the value of clinical concepts, but most do not jointly address criterion-specific localisation, relationships among criteria and criterion-level uncertainty. MedCORE is designed around these three requirements.

![](images/4fec1ebf3e93a5a838a672f2fb930db6415ad2ed68952f9dece9928e595db67c.jpg)  
Figure 1: Clinically guided interpretable medical image classification framework. Input images are encoded using a frozen BioMedCLIP ViT-B/16 backbone, while clinical criterion queries extract localized evidence. The extracted evidence is refined through multi-scale encoding and graph attention reasoning with clinical priors. The model outputs diagnosis probabilities, criterion reliability scores, uncertainty estimates, and interpretable heatmaps.

Spatial grounding, relational reasoning and uncertainty. Spatial grounding is particularly important in medicine because many diagnostic criteria are valid only in specific regions. Segmentation and foundation models such as SAM and MedSAM demonstrate the value of spatial priors in natural and medical image analysis [13, 18]. However, diagnostic classification often requires criterion-specific evidence rather than generic masks or image-level saliency. MedCORE therefore learns criterion-specific attention maps directly from dense visual tokens, localising evidence for each clinical criterion without requiring pixel-level criterion annotations. Clinical criteria are also relational: for example, asymmetry, border irregularity and colour variation may provide mutually reinforcing evidence in dermoscopy, while microaneurysms, haemorrhages and exudates jointly inform diabetic retinopathy severity. Graph attention networks provide a compact mechanism for learning such dependencies [25]. Prototype-based models such as PIP-Net and ProtoViT improve interpretability through visual prototypes [17, 19]; MedCORE instead uses class-wise prototypes as criterion-level discriminative memory while keeping explanations organised around clinical criteria. Finally, recent work emphasizes the importance of uncertainty in medical image classification [4]. Rather than estimating only image-level confidence, MedCORE assigns uncertainty to each criterion and uses it to calibrate evidence aggregation.

## 3 Methodology

## 3.1 Problem Formulation

Let $\mathbf { I } \in \mathbb { R } ^ { H \times W \times 3 }$ denote a medical image and $y \in \{ 1 , \ldots , C \}$ its ground-truth disease label. Rather than learning a direct mapping $f : \mathbf { I } \mapsto y$ , MedCORE decomposes the diagnostic decision into K dataset-specific clinical criteria $\{ c _ { k } \} _ { k = 1 } ^ { K }$ , each representing a morphological or textural property that clinicians systematically evaluate $( e . g .$ ., asymmetry and border irregularity in dermoscopy; shape and margin in breast ultrasound; microaneurysms and haemorrhages in diabetic retinopathy grading). The model evaluates each criterion independently, explicitly models inter-criteria dependencies, and aggregates evidence into a final diagnosis proportionally weighted by per-criterion confidence.

## 3.2 Visual Backbone and Feature Extraction

Given input image I, a shared visual backbone $( e . g .$ , BiomedCLIP ViT-B/16 [31]) produces a dense feature map:

$$
\mathbf { F } \in \mathbb { R } ^ { N _ { p } \times D } ,\tag{1}
$$

where $N _ { p } = H ^ { \prime } W ^ { \prime }$ is the number of spatial tokens and D is the embedding dimension. The backbone is frozen; only task-specific modules introduced below are trained, preserving the rich biomedical visual priors acquired during pre-training.

## 3.3 Criterion-Aware Spatial Localisation

Different clinical criteria manifest in distinct anatomical regions of the image. For criterion k, we learn a spatial attention map $\mathbf { a } _ { k } \in \mathbb { R } ^ { N _ { p } }$ via a lightweight query-key mechanism:

$$
\mathbf { a } _ { k } = \mathrm { s o f t m a x } \left( { \frac { \mathbf { q } _ { k } \mathbf { F } ^ { \top } } { \sqrt { D } } } \right) , \quad \mathbf { r } _ { k } = \mathbf { a } _ { k } \mathbf { F } ,\tag{2}
$$

where $\mathbf { q } _ { k } \in \mathbb { R } ^ { 1 \times D }$ is a learnable criterion query vector and $\mathbf { r } _ { k } \in \mathbb { R } ^ { D }$ is the resulting criterionspecific region descriptor. This produces K spatially grounded representations, one per criterion, without requiring spatial supervision.

## 3.4 Multi-Scale Criterion Encoding

Pathological features operate at two complementary scales: macro-structural $( e . g .$ , overall lesion shape, global tissue organisation) and micro-textural (e.g., fine-grained surface patterns, local intensity heterogeneity). For criterion k we encode both scales separately and fuse them:

$$
\mathbf { v } _ { k } = \mathbf { W } _ { \mathrm { f u s e } } \left[ \mathbf { v } _ { k } ^ { \mathrm { m a c r o } } ; \mathbf { v } _ { k } ^ { \mathrm { m i c r o } } \right] + \mathbf { b } ,\tag{3}
$$

where ${ \bf v } _ { k } ^ { \mathrm { m a c r o } }$ is obtained by global average pooling over $\mathbf { r } _ { k }$ and a linear projection, $\mathbf { v } _ { k } ^ { \mathrm { m i c r o } }$ is derived from a local patch neighbourhood of the peak attention location in ${ \bf a } _ { k }$ passed through a depthwise convolutional encoder, $[ ; ]$ denotes concatenation, and $\mathbf { W } _ { \mathrm { f u s e } } \in \dot { \mathbb { R } } ^ { D \times 2 D }$ is a learnable fusion matrix. The resulting $\mathbf { v } _ { k } \in \mathbb { R } ^ { D }$ encodes the criterion’s evidence at both scales.

## 3.5 Graph Attention Network for Inter-Criteria Reasoning

Clinical criteria are not independent: a finding of malignant asymmetry strengthens the evidential weight of irregular borders, and the co-presence of hard exudates and microaneurysms jointly elevates the DR severity grade. MedCORE models these dependencies explicitly by constructing a fully connected criterion graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where nodes $\nu =$ $\{ \mathbf { v } _ { k } \} _ { k = 1 } ^ { K }$ are criterion representations and edges $\mathcal { E }$ carry learnable relational weights.

A single Graph Attention Network (GAT) [25] layer refines each node by attending over its neighbours:

$$
\begin{array} { r } { e _ { k j } = \mathrm { L e a k y R e L U } ( \mathbf { a } ^ { \top } [ \mathbf { W } _ { g } \mathbf { v } _ { k }  \mathbf { W } _ { g } \mathbf { v } _ { j } ] ) , } \end{array}\tag{4}
$$

$$
\alpha _ { k j } = \frac { \exp ( e _ { k j } ) } { \sum _ { j ^ { \prime } \in \mathcal { N } ( k ) } \exp ( e _ { k j ^ { \prime } } ) } ,\tag{5}
$$

$$
\tilde { \mathbf { v } } _ { k } = \sigma \left( \sum _ { j \in \mathcal { N } ( k ) } \alpha _ { k j } \mathbf { W } _ { g } \mathbf { v } _ { j } \right) ,\tag{6}
$$

where $\mathbf { W } _ { g } \in \mathbb { R } ^ { D ^ { \prime } \times D }$ is a shared linear projection, $\mathbf { a } \in \mathbb { R } ^ { 2 D ^ { \prime } }$ is the attention vector, $\mathcal { N } ( k )$ denotes all criteria (full graph), ∥ is vector concatenation, and $\sigma$ is ELU. Multi-head attention (H = 4 heads) is used and outputs are concatenated. The refined representations $\{ \tilde { \mathbf { v } } _ { k } \} _ { k = 1 } ^ { K }$ encode each criterion in the context of all others.

## 3.6 Vision–Language Criterion Alignment

Each criterion $c _ { k }$ is paired with a short clinical text descriptor (e.g., “irregular, asymmetric border with notching” for dermoscopic border criterion). The frozen BiomedCLIP text encoder maps this to a text embedding $\mathbf { t } _ { k } \in \mathbb { R } ^ { D }$ . We align $\tilde { \mathbf { v } } _ { k }$ and $\mathbf { t } _ { k }$ using a symmetric InfoNCE contrastive loss over a batch of B samples:

$$
\mathcal { L } _ { \mathrm { V L } } = - \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \log \frac { \exp \left( \tilde { \mathbf { v } } _ { k } ^ { \top } \mathbf { t } _ { k } / \tau \right) } { \sum _ { j = 1 } ^ { K } \exp \left( \tilde { \mathbf { v } } _ { k } ^ { \top } \mathbf { t } _ { j } / \tau \right) } ,\tag{7}
$$

where $\tau$ is a learnable temperature. This loss pulls each criterion’s visual representation towards its corresponding clinical description while pushing it away from the descriptions of other criteria, grounding visual features in clinically meaningful language.

## 3.7 Class-Wise Visual Prototype Reinforcement

To further sharpen criterion discriminability, we maintain a set of class-wise visual prototypes $\mathbf { P } _ { k } = \{ \mathbf { p } _ { k , c } \} _ { c = 1 } ^ { C } \subset \mathbb { R } ^ { D }$ for each criterion k. These prototypes are updated online as exponential moving averages of the criterion features from each class. The prototype prediction for criterion k is computed as the cosine similarity to each class prototype:

$$
\hat { p } _ { k , c } = \frac { \exp \left( \sin \left( \tilde { \mathbf { v } } _ { k } , \mathbf { p } _ { k , c } \right) / \tau _ { p } \right) } { \sum _ { c ^ { \prime } = 1 } ^ { C } \exp \left( \sin \left( \tilde { \mathbf { v } } _ { k } , \mathbf { p } _ { k , c ^ { \prime } } \right) / \tau _ { p } \right) } ,\tag{8}
$$

where sim $( \mathbf { u } , \mathbf { v } ) { = } { \mathbf { u } } ^ { \top } \mathbf { v } / ( \| \mathbf { u } \| \| \mathbf { v } \| )$ is cosine similarity and $\tau _ { p }$ is a fixed temperature. The prototype cross-entropy reinforces class-discriminative structure within each criterion’s feature space:

$$
\mathcal { L } _ { \mathrm { p r o t o } } = - \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \log \hat { p } _ { k , y } ,\tag{9}
$$

where y is the ground-truth class label.

## 3.8 Uncertainty-Calibrated Criterion Aggregation

Not all criteria are equally informative for every sample. For ambiguous cases, a criterion may produce unreliable evidence that should contribute less to the final decision. For each criterion k, MedCORE predicts a scalar uncertainty estimate $\hat { u } _ { k } \in \mathbb { R } ^ { + }$ via a two-layer MLP applied to $\tilde { \mathbf { v } } _ { k }$ . Criterion contributions to the final class logits are then weighted inversely by their uncertainty through a softmax-normalised scheme:

$$
w _ { k } = \frac { \exp ( - \hat { u } _ { k } ) } { \sum _ { j = 1 } ^ { K } \exp ( - \hat { u } _ { j } ) } ,\tag{10}
$$

$$
\hat { \mathbf { y } } = \sum _ { k = 1 } ^ { K } w _ { k } \mathbf { W } _ { \mathrm { c l s } } \tilde { \mathbf { v } } _ { k } ,\tag{11}
$$

where $\mathbf { W } _ { \mathrm { c l s } } \in \mathbb { R } ^ { C \times D }$ is a shared classification head. High uncertainty $\hat { u } _ { k }$ suppresses criterion $k \mathrm { { s } }$ vote; a criterion that fires confidently on diagnostically decisive features dominates the aggregation.

## 3.9 Training Objective

MedCORE is trained end-to-end with a three-term objective:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { C E } } + \lambda _ { \mathrm { V L } } \mathcal { L } _ { \mathrm { V L } } + \lambda _ { \mathrm { p r o t o } } \mathcal { L } _ { \mathrm { p r o t o } } , } \end{array}\tag{12}
$$

where $\mathcal { L } _ { \mathrm { C E } }$ is the standard cross-entropy loss on the aggregated logits yˆ, LVL (Eq. 7) enforces language alignment, $\mathcal { L } _ { \mathrm { p r o t o } } \left( \mathrm { E q . } 9 \right)$ reinforces prototype discriminability, and $\lambda _ { \mathrm { V L } } , \lambda _ { \mathrm { p r o t o } } >$ 0 are balancing coefficients. The uncertainty head is trained implicitly: criteria that consistently mislead the classification loss are penalised through the gradient signal, causing the MLP to learn higher $\hat { u } _ { k }$ for unreliable criteria without requiring explicit uncertainty supervision.

## 4 Experimental Analysis

Datasets. We evaluate MedCORE on three public medical image benchmarks spanning dermoscopy, breast ultrasound, and retinal fundus imaging. These datasets cover clinically distinct diagnostic settings and allow us to test whether criteria-grounded reasoning generalizes across lesion morphology, sonographic appearance, and retinal pathology. ISIC 2018 [7] is a dermoscopic skin lesion dataset designed for automated skin lesion analysis and melanoma-related diagnosis. It contains lesion images with diagnostic categories and is well suited for criteria such as asymmetry, border irregularity, colour variation, dermoscopic structures, and vascular patterns, which are central to dermatological visual assessment. BUSI [1] is a breast ultrasound dataset containing normal, benign, and malignant cases. It captures grayscale sonographic appearances of breast tissue and lesions, making it suitable for BI-RADS-inspired criteria such as lesion shape, margin characteristics, internal echo pattern, posterior acoustic features, and lesion orientation. IDRiD [21] is a retinal fundus image dataset for diabetic retinopathy analysis. It provides image-level disease severity information and lesion annotations for typical diabetic retinopathy findings, making it appropriate for ETDRS-aligned criteria such as microaneurysms, haemorrhages, hard exudates, soft exudates, and neovascularisation.

Table 1: Experimental setup and dataset-specific clinical criteria used by MedCORE.
<table><tr><td>Dataset</td><td>Task</td><td>Classes</td><td>Clinical criteria used by MedCORE</td></tr><tr><td>ISIC 2018</td><td>Dermoscopic lesion classi- 7 classes fication</td><td></td><td>Asymmetry; border irregularity; colour variation; dermoscopic structures; vascu- lar patterns</td></tr><tr><td>BUSI</td><td>Breast ultrasound classifi- 3 classes cation</td><td></td><td>Lesion shape; margin characteristics; in- ternal echo pattern; posterior acoustic fea- tures; lesion orientation</td></tr><tr><td>IDRiD</td><td>Diabetic retinopathy grad- 5 grades ing</td><td></td><td>Microaneurysm density; haemorrhage severity; hard exudate extent; soft exudate presence; neovascularisation</td></tr></table>

We evaluate MedCORE on three representative medical image classification tasks covering different imaging modalities and diagnostic criteria: dermoscopic lesion classification, breast ultrasound classification, and diabetic retinopathy grading. The evaluated datasets are ISIC 2018 [7], BUSI [1], and IDRiD [21]. These datasets were selected because their diagnostic processes naturally depend on structured clinical criteria rather than a single global image pattern. Table 1 summarizes the task definitions and the clinical criteria used by Med-CORE for each dataset.

All images are resized to 224 × 224 and normalized before being passed to the visual backbone. During training, we apply standard image augmentations, including random horizontal flipping, random resized cropping, mild rotation, and color jittering where appropriate. The same preprocessing and data partitions are used for all competing methods to ensure a fair comparison. For MedCORE, the visual backbone is kept frozen, and only the criterion queries, multi-scale criterion encoder, graph reasoning module, uncertainty head, prototype module, and classifier are optimized.

Unless otherwise specified, MedCORE uses K = 5 clinical criteria for each dataset. The criterion graph is fully connected, and the graph attention module uses four attention heads. The model is trained using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ and weight decay of $1 \times 1 0 ^ { - 4 }$ . The batch size is set to 32, and training is performed for 100 epochs with early stopping based on validation Macro-F1. The loss balancing weights are set to $\lambda _ { \mathrm { V L } } = 0 . 1$ and $\lambda _ { \mathrm { p r o t o } } = 0 . 1$ . The prototype temperature is fixed to $\tau _ { p } = 0 . 0 7$ , while the vision–language temperature τ is learnable. All results are reported on the held-out test set.

## 4.1 Evaluation Metrics

We evaluate classification performance using accuracy, Macro-F1, and area under the receiver operating characteristic curve (AUC). Accuracy measures the overall proportion of correctly classified test samples:

$$
\mathrm { A c c u r a c y } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { 1 } ( \hat { y } _ { i } = y _ { i } ) ,\tag{13}
$$

where N is the number of test samples, $y _ { i }$ is the ground-truth label, and $\hat { y } _ { i }$ is the predicted label. Since the evaluated medical datasets are class-imbalanced, Macro-F1 is used as the

primary metric. For each class $^ { c , }$ precision, recall, and F1-score are computed as

$$
\mathrm { P r e c i s i o n } _ { c } = \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F P } _ { c } } , \quad \mathrm { R e c a l l } _ { c } = \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F N } _ { c } } ,\tag{14}
$$

$$
\mathrm { F } 1 _ { c } = \frac { 2 \cdot \mathrm { P r e c i s i o n } _ { c } \cdot \mathrm { R e c a l l } _ { c } } { \mathrm { P r e c i s i o n } _ { c } + \mathrm { R e c a l l } _ { c } } , \quad \mathrm { M a c r o - F } 1 = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathrm { F } 1 _ { c } .\tag{15}
$$

AUC is computed using one-vs-rest classification for multi-class tasks and then averaged across classes.

In addition to discrimination performance, we evaluate calibration using reliability diagrams. For a set of confidence bins $\{ B _ { m } \} _ { m = 1 } ^ { M }$ , the expected calibration error (ECE) is defined as

$$
\mathrm { E C E } = \sum _ { m = 1 } ^ { M } \frac { \left| B _ { m } \right| } { N } \left| \operatorname { a c c } ( B _ { m } ) - \operatorname { c o n f } ( B _ { m } ) \right| ,\tag{16}
$$

where acc $\left( B _ { m } \right)$ is the empirical accuracy of samples in bin $B _ { m }$ , and $\mathrm { c o n f } ( B _ { m } )$ is their average predicted confidence. A well-calibrated model has empirical accuracy close to confidence across all bins.

## 4.2 Comparison with Baseline Methods

We compare MedCORE with representative CNN, transformer, biomedical vision–language, concept-based, and prototype-based baselines. CNN baselines include DenseNet–121 [11], ResNet-50 [15], and EfficientNet-B0 [24]. Transformer baselines include ViT-B/16 [8] and Swin-T [16]. To assess the benefit of clinical-criterion reasoning beyond biomedical pretraining alone, we include BiomedCLIP [31] with linear probing and fine-tuning. We also compare with Concept Bottleneck Models [14] and ProtoPNet [5], which provide interpretable intermediate concepts or prototypes.

Table 3 reports the complete classification results. MedCORE achieves the best performance across all three datasets, obtaining an average Macro-F1 of 87.0%. Compared with BiomedCLIP fine-tuning, MedCORE improves Macro-F1 by 3.3 percentage points on ISIC 2018, 2.4 percentage points on BUSI, and 3.7 percentage points on IDRiD. These gains indicate that the improvement does not come only from using a strong biomedical visual backbone; rather, explicit clinical-criterion localization, inter-criterion reasoning, and uncertainty-calibrated aggregation provide additional benefits.

The largest absolute improvement is observed on IDRiD, where grading diabetic retinopathy depends on multiple interacting lesion types. This supports the motivation for graphbased inter-criterion reasoning. The improvement on ISIC 2018 also shows that decomposing lesion diagnosis into ABCD-inspired criteria provides stronger supervision than global visual classification. BUSI obtains the highest absolute scores, suggesting that the BI-RADS-inspired criterion set captures diagnostically meaningful ultrasound patterns.

## 4.3 Ablation Study

We conduct ablation experiments to quantify the contribution of each component in Med-CORE. Figure 2 reports the average Macro-F1 drop after removing individual modules, and Table 4 provides the corresponding values.

Table 3: Quantitative comparison with baseline methods. Accuracy, Macro-F1, and AUC are reported as percentages. The last column reports the average Macro-F1 across the three datasets.
<table><tr><td rowspan="2">Method</td><td colspan="3">ISIC 2018</td><td colspan="3">BUSI</td><td colspan="3">IDRiD</td><td rowspan="2">Avg. F1</td></tr><tr><td>Acc.</td><td>F1</td><td>AUC</td><td>Acc.</td><td>F1</td><td>AUC</td><td>Acc.</td><td>F1</td><td>AUC</td></tr><tr><td>ResNet-50 []</td><td>81.2</td><td>75.8</td><td>90.4</td><td>87.4</td><td>85.9</td><td>93.5</td><td>73.5</td><td>68.4</td><td>84.7</td><td>76.7</td></tr><tr><td>DenseNet-121 []</td><td>82.7</td><td>77.3</td><td>91.2</td><td>89.2</td><td>87.8</td><td>94.6</td><td>75.4</td><td>70.1</td><td>86.3</td><td>78.4</td></tr><tr><td>EfficientNet-B0 [[]</td><td>84.1</td><td>78.8</td><td>92.0</td><td>90.6</td><td>89.1</td><td>95.5</td><td>76.9</td><td>72.0</td><td>87.5</td><td>80.0</td></tr><tr><td>ViT-B/16 [8]</td><td>84.8</td><td>79.6</td><td>92.5</td><td>91.2</td><td>89.8</td><td>96.1</td><td>78.0</td><td>73.1</td><td>88.4</td><td>80.8</td></tr><tr><td>Swin-T []</td><td>85.6</td><td>80.8</td><td>93.1</td><td>92.1</td><td>90.7</td><td>96.6</td><td>79.2</td><td>74.6</td><td>89.2</td><td>82.0</td></tr><tr><td>BioMedCLIP-LP []</td><td>85.9</td><td>81.2</td><td>93.8</td><td>93.3</td><td>91.7</td><td>97.0</td><td>79.6</td><td>75.0</td><td>89.7</td><td>82.6</td></tr><tr><td>BioMedCLIP-FT []</td><td>87.0</td><td>82.4</td><td>94.7</td><td>94.2</td><td>92.8</td><td>97.5</td><td>81.0</td><td>76.5</td><td>90.8</td><td>83.9</td></tr><tr><td>Concept Bottleneck Model []</td><td>85.5</td><td>80.7</td><td>93.2</td><td>92.4</td><td>91.1</td><td>96.6</td><td>79.5</td><td>74.4</td><td>89.2</td><td>82.1</td></tr><tr><td>ProtoPNet []</td><td>86.1</td><td>81.5</td><td>94.1</td><td>93.0</td><td>91.8</td><td>97.0</td><td>80.1</td><td>75.7</td><td>90.0</td><td>83.0</td></tr><tr><td>MedCORE</td><td>89.2</td><td>85.7</td><td>96.4</td><td>96.1</td><td>95.2</td><td>98.4</td><td>84.3</td><td>80.2</td><td>92.8</td><td>87.0</td></tr></table>

![](images/0d195e2f0559168c1a6fbfa705cabec4bb61d3ccc844ada91e33a61cb09ac6a3.jpg)  
Figure 2: Ablation study showing the average Macro-F1 drop from the full MedCORE model. Larger values indicate a more important component.

Table 4: Ablation study based on the average Macro-F1 over ISIC 2018, BUSI, and IDRiD. The full MedCORE model obtains an average Macro-F1 of 87.0%.
<table><tr><td>Variant</td><td>F1 drop</td><td>Avg. F1</td></tr><tr><td>Full MedCORE</td><td>0.0</td><td>87.0</td></tr><tr><td>w/o criterion localization</td><td>2.9</td><td>84.1</td></tr><tr><td>w/o multi-scale encoder</td><td>1.7</td><td>85.3</td></tr><tr><td>w/o GAT reasoning</td><td>2.0</td><td>85.0</td></tr><tr><td>w/o VL alignment</td><td>1.4</td><td>85.6</td></tr><tr><td>w/o prototypes</td><td>1.1</td><td>85.9</td></tr><tr><td>uniform criterion weights</td><td>1.7</td><td>85.3</td></tr><tr><td>global feature only</td><td>3.4</td><td>83.6</td></tr></table>

![](images/d4f5abfe46128653f68403b075ee29561b6282396987a136c9c8116fc3397c89.jpg)  
Number of clinical criteria K  
Figure 3: Sensitivity analysis with respect to the number of clinical criteria K. The best performance is obtained with K = 5, which corresponds to the clinically defined criterion set used in MedCORE.

Table 5: Average Macro-F1 under different numbers of clinical criteria.
<table><tr><td>Number of criteria K</td><td>Avg. Macro-F1</td></tr><tr><td>1</td><td>83.6</td></tr><tr><td>3</td><td>85.5</td></tr><tr><td>5</td><td>87.0</td></tr><tr><td>7</td><td>86.6</td></tr><tr><td>9</td><td>85.9</td></tr></table>

The largest degradation occurs when replacing criterion-level reasoning with a single global feature representation, which reduces average Macro-F1 by 3.4 percentage points. This confirms that the proposed clinical decomposition is central to MedCORE. Removing criterion localization produces the second largest drop of 2.9 percentage points, showing that different clinical criteria require spatially distinct evidence. Removing GAT reasoning decreases Macro-F1 by 2.0 percentage points, indicating that dependencies among clinical findings are useful for diagnosis. The multi-scale encoder and uncertainty weighting each contribute 1.7 percentage points, demonstrating the importance of combining macro-structural and micro-textural evidence and suppressing unreliable criterion predictions. Vision–language alignment and prototype reinforcement provide smaller but consistent gains, suggesting that clinical semantic grounding and class-wise visual prototypes regularize the criterion feature space.

## 4.4 Sensitivity to the Number of Clinical Criteria

Figure 3 and Table 5 show the effect of varying the number of clinical criteria K. The average Macro-F1 improves from 83.6% at K = 1 to 87.0% at K = 5, and then gradually decreases when K is increased to 7 or 9.

The poor performance at K = 1 indicates that a single global representation is insufficient to capture diverse diagnostic evidence. Increasing K to 3 improves performance because the model can separate multiple clinical cues. The best result is obtained at $K = 5$ , which matches the manually defined clinical criterion set for each dataset. Increasing K beyond 5 slightly reduces performance, likely because additional criteria become redundant, weakly defined, or noisy, making criterion localization and graph reasoning less stable. This result supports the use of a compact clinically motivated criterion set rather than an excessively large number of latent concepts.

![](images/16750bd923b241a502320426549e2b9e850063aecc604b0ea45cc245a0cc2256.jpg)  
Figure 4: Reliability diagram comparing MedCORE with BiomedCLIP fine-tuning. Med CORE remains closer to the perfect calibration line, indicating better agreement between predicted confidence and empirical accuracy.

Table 6: Reliability diagram values from Fig. 4. Values report empirical accuracy at each confidence bin.
<table><tr><td>Conf.</td><td>Perfect calibration</td><td>BioMedCLIP-FT</td><td>MedCORE</td></tr><tr><td>0.55</td><td>0.55</td><td>0.50</td><td>0.56</td></tr><tr><td>0.65</td><td>0.65</td><td>0.58</td><td>0.66</td></tr><tr><td>0.75</td><td>0.75</td><td>0.67</td><td>0.74</td></tr><tr><td>0.85</td><td>0.85</td><td>0.76</td><td>0.84</td></tr><tr><td>0.95</td><td>0.95</td><td>0.83</td><td>0.93</td></tr></table>

## 4.5 Calibration Analysis

Figure 4 compares the reliability diagram of MedCORE with BiomedCLIP fine-tuning. Table 6 lists the plotted confidence bins and empirical accuracies. MedCORE follows the diagonal perfect-calibration line more closely than BiomedCLIP-FT across all confidence bins.

BiomedCLIP-FT is consistently below the diagonal, indicating over-confidence: its predicted confidence is higher than its empirical accuracy. This effect is especially visible at high confidence, where the empirical accuracy at 0.95 confidence is only 0.83. In contrast, MedCORE reaches 0.93 empirical accuracy at the same confidence level. Using the plotted bins with equal weighting, the mean absolute calibration gap is approximately 8.2% for BiomedCLIP-FT and 1.2% for MedCORE. This improvement is consistent with the uncertainty-calibrated aggregation mechanism, which reduces the influence of unreliable criteria before producing the final prediction.

## 5 Conclusions and Future Directions

We presented MedCORE, a criteria-grounded framework for interpretable medical image diagnosis. Instead of relying on a direct image-to-label prediction pipeline, MedCORE decomposes diagnosis into clinically meaningful criteria, localises criterion-specific evidence, refines inter-criteria relationships with graph attention, and aggregates evidence using uncertainty-calibrated weighting. By combining frozen biomedical vision–language representations with lightweight criterion reasoning, visual prototype reinforcement, and text-aligned supervision, MedCORE improves both diagnostic performance and explanatory structure across dermoscopy, breast ultrasound, and retinal fundus imaging benchmarks. Experiments on ISIC 2018, BUSI, and IDRiD show that MedCORE achieves stronger accuracy, macro-F1, and AUC than CNN, Transformer, vision–language, concept bottleneck, and prototype-based baselines. More importantly, the model provides criterion-level evidence, reliability scores, uncertainty estimates, and spatial heatmaps, making its predictions more transparent and clinically interpretable. Future work will extend MedCORE to larger multicentre datasets, incorporate expert-validated criterion annotations, and study its calibration and usefulness in real clinical decision-support settings.

## Acknowledgement

This research was funded by Khalifa University of Science and Technology through the Faculty Startup grant under Project ID: KU-INT-FSU-2026-8471000024.

## References

[1] Walid Al-Dhabyani, Mohammed Gomaa, Hussien Khaled, and Aly Fahmy. Dataset of breast ultrasound images. Data in brief, 28:104863, 2020.

[2] Yequan Bie, Luyang Luo, and Hao Chen. Mica: Towards explainable skin lesion diagnosis via multi-level image-concept alignment. 38(2):837–845, 2024.

[3] Benedikt Boecking, Naoto Usuyama, Shruthi Bannur, Daniel C. Castro, Anton Schwaighofer, Stephanie Hyland, Maria Wetscherek, Tristan Naumann, Aditya Nori, Javier Alvarez-Valle, Hoifung Poon, and Ozan Oktay. Making the most of text semantics to improve biomedical vision-language processing. In European Conference on Computer Vision, pages 1–21. Springer, 2022.

[4] Aobo Chen, Yangyi Li, Cheng Qian, Cheng Wu, Hong-Yu Zhang, Quan Sun, Xiang Li, Jundong Li, Qian Wang, and Aidong Zhang. Modeling and understanding uncertainty in medical image classification. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, pages 557–567. Springer, 2024.

[5] Chaofan Chen, Oscar Li, Daniel Tao, Alina Barnett, Cynthia Rudin, and Jonathan K Su. This looks like that: deep learning for interpretable image recognition. Advances in neural information processing systems, 32, 2019.

[6] Townim F. Chowdhury, Vu Minh Hieu Phan, Kewen Liao, Minh-Son To, Yutong Xie, Anton van den Hengel, Johan W. Verjans, and Zhibin Liao. Adacbm: An adaptive con-

cept bottleneck model for explainable and accurate diagnosis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 35–45. Springer, 2024.

[7] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, et al. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). arXiv preprint arXiv:1902.03368, 2019.

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[9] Yunhe Gao, Difei Gu, Mu Zhou, and Dimitris Metaxas. Aligning human knowledge with visual concepts towards explainable medical image classification. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, pages 46–56. Springer, 2024.

[10] Lijie Hu, Songning Lai, Wenshuo Chen, Hongru Xiao, Hongbin Lin, Lu Yu, Jingfeng Zhang, and Di Wang. Towards multi-dimensional explanation alignment for medical classification. volume 37, pages 129640–129671, 2024.

[11] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 4700–4708, 2017.

[12] Shih-Cheng Huang, Liyue Shen, Matthew P. Lungren, and Serena Yeung. Gloria: A multimodal global-local representation learning framework for label-efficient medical image recognition. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 3942–3951, 2021.

[13] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.

[14] Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In International Conference on Machine Learning, pages 5338–5348. PMLR, 2020.

[15] Brett Koonce. Resnet 50. In Convolutional neural networks with swiftfor tensorflow: image recognition and dataset categorization, pages 63–72. Springer, 2021.

[16] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.

[17] Chiyu Ma, Jon Donnelly, Wenjun Liu, Soroush Vosoughi, Cynthia Rudin, and Chaofan Chen. Interpretable image classification with adaptive prototype-based vision transformers. volume 37, pages 41447–41493, 2024.

[18] Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. Segment anything in medical images. Nature Communications, 15(1):654, 2024.

[19] Meike Nauta, Jorg Schlotterer, Maurice van Keulen, and Christin Seifert. Pip-net: Patch-based intuitive prototypes for interpretable image classification. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2744–2753, 2023.

[20] Tuomas Oikarinen, Subhro Das, Lam M Nguyen, and Tsui-Wei Weng. Label-free concept bottleneck models. 2023.

[21] Prasanna Porwal, Samiksha Pachade, Rahul Kamble, Manesh Kokare, Girish Deshmukh, Vivek Sahasrabuddhe, and Fabrice Meriaudeau. Indian diabetic retinopathy image dataset (idrid): A database for diabetic retinopathy screening research. Data, 3 (3):25, 2018.

[22] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748– 8763. PMLR, 2021.

[23] Divyansh Srivastava, Ge Yan, and Tsui-Wei Weng. Vlg-cbm: Training concept bottleneck models with vision-language guidance. In Advances in Neural Information Processing Systems, volume 37, pages 79057–79094, 2024.

[24] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR, 2019.

[25] Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio,´ and Yoshua Bengio. Graph attention networks. arXivpreprint arXiv:1710.10903, 2017.

[26] Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Yanfeng Wang, and Weidi Xie. Medklip: Medical knowledge enhanced language-image pre-training for x-ray diagnosis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21372–21383, 2023.

[27] Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, and Bingbing Ni. Medmnist v2: A large-scale lightweight benchmark for 2d and 3d biomedical image classification. Scientific Data, 10(1):41, 2023.

[28] Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, and Mark Yatskar. Language in a bottle: Language model guided concept bottlenecks for interpretable image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19187–19197, 2023.

[29] Mert Yuksekgonul, Maggie Wang, and James Zou. Post-hoc concept bottleneck mod els. arXiv preprint arXiv:2205.15480, 2022.

[30] Kai Zhang, Rong Zhou, Eashan Adhikarla, et al. A generalist vision-language foundation model for diverse biomedical tasks. Nature Medicine, 30:3129–3141, 2024.

[31] Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al. Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv preprint arXiv:2303.00915, 2023.

[32] Xiaoman Zhang, Chaoyi Wu, Ya Zhang, Yanfeng Wang, and Weidi Xie. Knowledgeenhanced visual-language pre-training on chest radiology images. Nature Communications, 14(1):4542, 2023.