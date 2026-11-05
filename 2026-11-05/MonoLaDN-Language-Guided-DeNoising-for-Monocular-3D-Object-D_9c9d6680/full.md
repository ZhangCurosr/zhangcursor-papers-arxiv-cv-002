# MonoLaDN: Language-Guided DeNoising for Monocular 3D Object Detection

Kiet Dang Vu<sup>1,2</sup> , Cao Le Minh Khoa<sup>1,2</sup> , Nguyen Viet Tri Pham<sup>1,2</sup> , An Nguyen Gia<sup>1,2</sup> , Tam Le Thuc Ho<sup>1,2</sup> , and Duc Dung Nguyen<sup>1,2⋆</sup>

<sup>1</sup> AITech Lab., Ho Chi Minh City University of Technology (HCMUT) <sup>2</sup> Vietnam National University Ho Chi Minh City (VNUHCM)

Abstract. Monocular 3D object detection ofers a low-cost route to 3D perception, but it remains fundamentally ill-posed, since depth must be inferred from a single view. Recent transformer-based detectors learn geometry purely from image features and 3D box annotations, lacking the high-level semantic priors that vision-language models (VLMs) acquire from large-scale pretraining. VLMs ofer such priors, but deploying one directly as a detector is impractical due to its heavy inference cost. We propose MonoLaDN, a language-guided denoising framework that transfers the geometric reasoning of a VLM into a monocular 3D detector while keeping the deployed model purely vision-based. During training, a frozen VLM produces structured reasoning about each object’s occlusion, depth, dimension, and angle from the image and its ground-truth boxes. We encode this reasoning into queries, perturb them with noise, and train the detector to reconstruct the clean reasoning. Reconstructing the corrupted reasoning forces the detector’s depth-guided features to internalize the VLM’s geometric priors, so this knowledge remains available even after the reasoning path is discarded at inference, adding no latency or parameters to the deployed model. Experiments on KITTI 3D and nuScenes show that MonoLaDN consistently outperforms stateof-the-art detectors and generalizes well to new domains.

Keywords: Mono3D · VLM

## 1 Introduction

Monocular 3D object detection aims to recover the location, dimension, and orientation of objects in 3D space from a single RGB image. Relying only on a standard camera, it avoids the cost, calibration burden, and hardware complexity of LiDAR [6,30,42] and multi-view camera [16,18,20] setups, ofering a low-cost, easily deployable route to 3D perception for autonomous driving and robotics. The task is fundamentally ill-posed, however, since depth lost during projection must be inferred from a single view, where small distance errors translate into large 3D localization errors. These dificulties are compounded by occlusion and truncation, which remove the visual cues needed for reliable estimation.

![](images/20c9dff5759367887622164e02e4d72112f1e8906444938f3895f55a2fc603f3.jpg)  
Fig. 1: Comparison of paradigms for monocular 3D detection. (a) VLM-based methods carry rich semantic and geometric priors but are too costly for real-time inference. (b) Traditional vision-only detectors are eficient yet lack such priors, learning geometry only from image features and box annotations. (c) Our MonoLaDN injects VLM-derived 3D reasoning into a depth-guided detector during training, so it gains these priors while keeping a vision-only, eficient model at inference.

Recent transformer-based detectors have substantially advanced the field [13, 26, 40, 45]. MonoDETR [45] introduced the first DETR [2] framework, guiding object queries with contextual depth cues rather than purely local features. MonoDGP [26] improves geometry by replacing direct height regression with perspective-invariant geometry errors. MonoCLUE [40] strengthens appearance modeling via feature clustering and a generalized scene memory for robustness under occlusion, while MonoDLGD [13] improves supervision with a dificultyaware label-guided denoising scheme. Despite this progress, these methods learn geometry purely from image features and 3D box annotations, without higherlevel semantic knowledge about how objects are arranged and occluded in a scene.

In parallel, vision-language models (VLMs) have emerged as general-purpose perception systems with broad world knowledge and strong semantic reasoning. Their 3D competence, however, lags behind their 2D abilities, as benchmarks such as UniQA-3D [46] show that state-of-the-art VLMs still trail specialized detectors on fundamental 3D tasks. A growing body of work therefore adapts VLMs toward explicit 3D understanding [3, 22, 23, 28], through explicit spatial reasoning [22], fine-tuning on synthetic reasoning traces [28], end-to-end geometric perception [3], or casting detection as next-token prediction [23]. Yet these adapted models remain too large to run within a real-time detection pipeline, limiting their direct use as detectors.

These two lines of work occupy opposite extremes: VLMs encode rich semantic and geometric priors about real-world objects but are too computationally heavy for real-time deployment, whereas monocular detectors are eficient yet learn geometry solely from image features and box annotations, lacking such priors. The two are seldom reconciled, as a VLM’s knowledge is presumed useful only when the VLM itself is retained at inference. This raises a natural question:

## Can the geometric reasoning of a VLM benefit a monocular 3D detector without incurring its inference cost?

We answer afirmatively with MonoLaDN, a monocular 3D detector that exploits a VLM’s geometric reasoning only during training and discards it entirely at inference. As illustrated in Fig. 1, MonoLaDN bridges the two extremes: it keeps the lightweight, vision-only architecture of a depth-guided detector, while transferring the VLM’s spatial knowledge into it through a training-only reasoning branch that is removed before deployment.

Concretely, during training, we pair each image with its ground-truth 3D boxes and feed them, through a prompt generator, to a frozen VLM that produces structured natural-language reasoning about each object’s occlusion, depth, dimension, and angle. A text encoder maps this reasoning into attribute-specific embeddings, which are then perturbed with noise to form the reasoning queries. Inspired by the query denoising of DN-DETR [14], these noised queries are passed alongside the standard learnable detection queries into the depth-guided decoder, and the model is trained to reconstruct the clean object attributes from their corrupted reasoning. To prevent information leakage and control how reasoning is exchanged, we introduce a language-guided attention mask with two components. Following DN-DETR [14], a leakage-prevention mask blocks the learnable detection queries from attending to the reasoning queries, so they cannot copy the noised ground-truth as a shortcut that would collapse at inference. A separated-reasoning mask then restricts each reasoning query to three groups: all reasoning queries of the same object, those of the same attribute type across objects, and the learnable object queries, so each attribute is reconstructed from informative context without interference from unrelated attributes. At inference, the VLM, the reasoning branch, and the reasoning queries are all removed, leaving a vision-only detector that runs at the same speed as its MonoDGP [26] baseline while benefiting from the knowledge absorbed during training.

## Our contributions are summarized as follows:

We propose MonoLaDN, the first framework to transfer the geometric reasoning of a vision-language model into a monocular 3D detector, using the VLM purely as training-time supervision and discarding it at inference, adding no latency or parameters to the deployed model.

– We introduce a language-guided denoising scheme that encodes VLM’s reasoning into attribute-specific queries and trains the detector to reconstruct the clean reasoning from their noised form, regulated by a language-guided attention mask that blocks ground-truth shortcuts and confines each attribute to its informative context.

– Extensive experiments on KITTI 3D and nuScenes show that MonoLaDN consistently outperforms state-of-the-art monocular 3D detectors and generalizes to new domains, while adding no inference-time cost.

## 2 Related Work

3D - Vision Language Model. Vision-language models such as Qwen2.5- VL [1] and InternVL [36] excel at object recognition, 2D grounding, and visual reasoning. Their 3D competence lags behind, however, as Zuo et al. [46] show through the UniQA-3D benchmark, where state-of-the-art VLMs perform poorly on fundamental 3D tasks. Recent work therefore adapts VLMs toward explicit 3D understanding: SpatialReasoner [22] shares explicit 3D representations across perception, computation, and reasoning, while SpaceThinker [28] fine-tunes Qwen2.5-VL for quantitative spatial reasoning. Moving to geometric prediction, VLM-3D [3] enables end-to-end 3D perception via a joint semanticgeometric loss, and LocateAnything3D [23] casts 3D detection as next-token prediction through a Chain-of-Sight sequence that grounds objects in 2D before inferring distance, size, and pose. Unlike these approaches, we use a VLM only during training to produce structured geometric reasoning about occlusion, depth, dimension, and angle, supervising a conventional monocular detector through language-guided denoising while keeping it VLM-free at inference.

Monocular 3D Object Detection. Transformer-based detectors now dominate monocular 3D object detection. MonoDETR [45] pioneered this as the first DETR [2] framework for the task, using a depth-guided transformer whose decoder lets each object query draw features from depth-informative regions rather than only local cues. MonoDGP [26] refines the geometric side, replacing direct height regression with perspective-invariant geometry errors and adding a decoupled visual decoder and a Region Segmentation Head. MonoCLUE [40] instead emphasizes appearance cues via K-means clustering and a generalized scene memory for robustness under occlusion, while MonoDLGD [13] addresses supervision with a dificulty-aware label-guided denoising scheme, built on DN-DETR [14], that perturbs and reconstructs ground-truth labels. Built on MonoDGP, our method applies language-guided denoising that, inspired by DN-DETR, perturbs and reconstructs VLM-derived geometric reasoning rather than ground-truth labels, learning geometry-aware representations grounded in the VLM’s understanding while remaining VLM-free at inference.

Vision-Language-to-Detector Distillation. Distilling knowledge from pretrained VLM into detectors transfers their visual-semantic priors to lightweight models. ViLD [8] established the paradigm, aligning a two-stage detector’s box embeddings with a vision-language teacher’s embeddings, and OADP [35] refined this with an object-aware distillation pyramid that preserves proposal features while transferring contextual knowledge. DK-DETR [15] adapted distillation to query-based detectors via explicit semantic and implicit relational transfer. The idea has since reached 3D: CLIP-Fo3D [44] distills dense 2D CLIP features into point clouds through projection, and ImOV3D [41] transfers vision-language knowledge into a point-cloud detector by combining it with point-cloud geometry. In contrast to these methods, which transfer semantic and recognition priors into 2D or point-cloud detectors, we leverage a VLM’s explicit geometric reasoning to guide a monocular 3D detector through language-guided denoising over occlusion, depth, dimension, and angle, adding no inference-time cost.

## 3 Method

## 3.1 Overview

![](images/834d0225ebb2d0bf2bf347f37e65be9ba496ac461878fc48d388a61ebc63987d.jpg)  
Fig. 2: Overview of MonoLaDN. During training, a 3D Reasoning Generation stage queries a frozen VLM with the image and ground-truth 3D boxes to produce structured reasoning about each object’s occlusion, depth, dimension, and angle. A text encoder and noise injector turn this reasoning into perturbed queries, decoded jointly with the N learnable object queries under a Language-Guided Mask and trained to reconstruct the clean attributes, alongside the detection loss. At inference, the reasoning path is discarded, leaving a purely vision-based detector with no added cost.

Fig. 2 illustrates the overall architecture of MonoLaDN. The framework follows a dual-path design that shares a single depth-guided detector but difers between training and inference. We build on MonoDGP [26] as the detection path, where a backbone extracts multi-scale features for a visual and a depth encoder, and N learnable object queries are decoded into 3D boxes. This path is the only component retained at inference, incurring no cost over the baseline. Throughout, we index the four geometric attributes by a ∈ {occ, depth, dim, angle} and denote by M the number of ground-truth objects in an image.

During training, an auxiliary reasoning path supplies high-level supervision. A prompt generator pairs the image with its ground-truth 3D boxes and queries a frozen VLM to produce structured reasoning about each object’s occlusion, depth, dimension, and angle (Sec. 3.2). A text encoder and noise injector then encode this reasoning into queries and perturb them with noise (Sec. 3.3). Decoded jointly with the learnable queries under a language-guided attention mask, these perturbed queries are trained to reconstruct the clean attributes via a reasoning loss $\mathcal { L } _ { r e s }$ alongside the detection loss $\mathcal { L } _ { d e t }$ . This objective forces the detector’s depth-guided features to internalize the VLM’s reasoning, so the knowledge remains implicitly available once the reasoning path is discarded at inference.

## 3.2 3D Reasoning Generation

![](images/95effc6daac469f24c1a9889269865113c9432328f85ccd6dbd48caeeed12879.jpg)  
Fig. 3: 3D Reasoning Generation. The object-centric visual input and a geometryto-language mapping are fed to a frozen VLM. The mapping discretizes each attribute into an ordinal label: depth by distance binning, angle by viewpoint sectoring, dimension by shape-archetype matching, and occlusion by combining truncation and occlusion flags. The VLM then produces structured reasoning per attribute.

For each ground-truth object, we generate structured language reasoning about four geometric attributes, namely occlusion, depth, dimension, and angle, which serves as the supervision signal for the denoising branch. As illustrated in Fig. 3, this process comprises three stages: an object-centric visual input, a geometry-to-language mapping, and reasoning generation by a frozen VLM, all performed ofline during training.

Object-centric visual input. Since a scene image typically contains multiple objects, prompting the VLM with the full image alone leaves the target ambiguous. For each object, we therefore construct three complementary views. The full image preserves global scene context such as road layout and inter-object overlap, the clean crop exposes the object’s fine-grained appearance, and the object-attended view dims the background while preserving the original intensity of the object region. Together, these views focus the VLM on the target object while retaining the surrounding context required to reason about occlusion and depth.

Geometry-to-language mapping. As language models are known to handle discrete language far more reliably than precise continuous values [32, 34], and this limitation persists for VLMs in 3D geometric estimation [46], we do not ask the VLM to regress geometry directly. Instead, we derive a discrete ordinal label (detailed in the supplementary material) for each attribute from the groundtruth 3D annotation: the occlusion and truncation flags are combined into a visibility label, the camera-frame depth is binned by distance, the box dimensions are matched to shape archetypes, and the observation angle is sectored into a viewpoint label. This procedure keeps the supervision grounded in the true geometry while expressing it in a form the VLM can reason about efectively.

Reasoning generation. Conditioned on the object-centric views and the discretized label, the frozen VLM produces one concise evidence sentence that justifies each attribute label from the visible cues. Pairing each label with its evidence yields the per-object reasoning set $\{ R _ { o c c } , R _ { d e p t h } , R _ { d i m } , R _ { a n g l e } \}$ , which is subsequently consumed by the Language-Guided DeNoising described in Sec. 3.3.

## 3.3 Language-Guided DeNoising

Given the object reasoning set $\{ R _ { o c c } , R _ { d e p t h } , R _ { d i m } , R _ { a n g l e } \}$ , we first convert it into a reasoning embedding set $\{ E _ { o c c } , E _ { d e p t h } , E _ { d i m } , E _ { a n g l e } \}$ using a pretrained text encoder [27, 43]. To force the detector to internalize this reasoning rather than merely read it of at inference, we corrupt the reasoning embeddings with learned noise and train the model to reconstruct them, following the querydenoising paradigm of DN-DETR [14]. Concretely, we inject noise by predicting a diagonal Gaussian ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ with a two-layer MLP and apply the reparameterization trick to obtain the perturbed query:

$$
q _ { P } = \mu + \sigma \odot \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I )\tag{1}
$$

where $\mu$ and σ are the predicted mean and standard deviation and ⊙ is elementwise multiplication.

These perturbed queries are decoded jointly with the learnable object queries, so we must carefully design an attention mask to control how the two types of queries interact. Let q denote the set of N learnable object queries, and let q<sub>P</sub> denote the set of perturbed reasoning embeddings, where each of the M groundtruth objects contributes four attribute-specific embeddings for occlusion, depth, dimension, and angle:

$$
\begin{array} { r l } & { \mathbf { q } = \{ q _ { 0 } , q _ { 1 } , \dots , q _ { N - 1 } \} , } \\ & { \mathbf { q } _ { P } = \{ q _ { 0 } ^ { o c c } , q _ { 0 } ^ { d e p t h } , q _ { 0 } ^ { d i m } , q _ { 0 } ^ { a n g l e } , \dots , q _ { M - 1 } ^ { o c c } , q _ { M - 1 } ^ { d e p t h } , q _ { M - 1 } ^ { d i m } , q _ { M - 1 } ^ { a n g l e } \} . } \end{array}\tag{2}
$$

We represent the attention mask as a binary matrix $A \in \{ 0 , 1 \} ^ { W \times W }$ with $W =$ $4 M + N$ , where $A _ { i j } = 1$ allows query i to attend to query j and $A _ { i j } = 0$ blocks it. The W queries are ordered as the 4M reasoning queries $\mathbf { q } _ { P }$ followed by the N learnable queries $\mathbf { q } .$ We design this mask according to two rules, each addressing a diferent requirement, which we describe below before combining them into the final mask.

Leakage Prevention Mask. The reasoning queries are derived from the groundtruth 3D boxes. If the learnable queries could attend to them, the detector would simply copy this ground-truth information as a shortcut, and this dependency would break at inference, where the reasoning path is removed. Following DN-DETR [14], we therefore prevent every learnable query from attending to any reasoning query:

$$
A _ { i j } ^ { l p } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ } i \in { \bf q } \mathrm { ~ a n d ~ } j \in { \bf q } _ { P } , } \\ { 1 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{3}
$$

where $i \in { \textbf { q } }$ denotes a learnable query and $j \in \{ \mathbf { q } _ { P }$ a reasoning query, so a learnable query cannot attend to any reasoning query while all other interactions remain open.

Separated Reasoning Mask. We further control how the reasoning queries exchange information, so that each attribute is reconstructed from related context rather than unrelated cues. Specifically, each reasoning query attends only to all reasoning queries of the same object, the reasoning queries of the same attribute across diferent objects, and the learnable object queries. Denoting a reasoning query by its object index o and attribute $a \in \{ o c c , d e p t h , d i m , a n g l e \}$ we define:

$$
A _ { i j } ^ { s r } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ } i , j \in { \bf q } _ { P } , ~ o _ { i } \neq o _ { j } , ~ \mathrm { a n d ~ } a _ { i } \neq a _ { j } , } \\ { 1 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{4}
$$

so two reasoning queries are disconnected only when they belong to diferent objects and diferent attributes, while any interaction involving a learnable query stays open.

Language-Guided Mask. To enforce both rules at the same time, we combine the two masks by element-wise multiplication, so that an interaction is allowed only when both masks permit it:

$$
{ \cal A } _ { i j } = { \cal A } _ { i j } ^ { l p } \cdot { \cal A } _ { i j } ^ { s r } .\tag{5}
$$

## 3.4 Loss Function

Detection Loss. We adopt the detection objective of MonoDGP [26], which aggregates four terms: a region segmentation loss ${ \mathcal { L } } _ { \mathrm { r e g i o n } } ,$ a depth map regression loss ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ , a 2D detection loss $ { \mathcal { L } } _ { \mathrm { 2 D } }$ , and a 3D estimation loss $ { \mathcal { L } } _ { \mathrm { 3 D } }$ . The 2D term $ { \mathcal { L } } _ { \mathrm { 2 D } }$ groups the classification, 2D box regression, GIoU, and projected-center objectives, whereas the 3D term $ { \mathcal { L } } _ { \mathrm { 3 D } }$ supervises object size, angle, and center depth. The overall detection loss is given by

$$
\mathcal { L } _ { \mathrm { d e t } } = \mathcal { L } _ { \mathrm { 2 D } } + \mathcal { L } _ { \mathrm { 3 D } } + \lambda \mathcal { L } _ { \mathrm { d e p t h } } + \lambda \sum _ { i = 0 } ^ { 4 } \mathcal { L } _ { \mathrm { r e g i o n } } ^ { i } ,\tag{6}
$$

where λ weights the depth and region segmentation terms.

Reconstruction Loss. The reconstruction loss consists of two components. The first is an embedding reconstruction loss, which encourages the decoder output of each perturbed query to recover its original reasoning embedding produced by the text encoder, measured by a cosine similarity loss. Since the noise is sampled from a learnable diagonal Gaussian ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ , we additionally regularize this distribution with a KL divergence term against a standard normal prior, following the variational autoencoder formulation [11]. This component is defined as

$$
\mathcal { L } _ { \mathrm { e m b } } = \sum _ { a } \left( 1 - \cos \Bigl ( \hat { E } _ { a } , E _ { a } \Bigr ) \right) + \beta \operatorname { K L } \bigl ( \mathcal { N } ( \mu , \sigma ^ { 2 } ) \| \mathcal { N } ( 0 , I ) \bigr ) ,\tag{7}
$$

where $E _ { a }$ is the reasoning embedding of attribute $a \in \{ o c c , d e p t h , d i m , a n g l e \}$ 2 $\hat { E } _ { a }$ is the corresponding decoder output, and $\beta$ weights the KL term.

![](images/b2f27e9a15a8e581f7983f70087870ab17e99bd58cce62b26995a5d116febff3.jpg)  
Fig. 4: Object reconstruction heads. Each attribute output $\hat { E } _ { a }$ is decoded into the box component it reasons about. The occlusion output $\hat { E } _ { o c c }$ has no box target of its own and is broadcast as shared visibility context (dashed) to the dimension, depth, and angle heads. When a head takes more than one embedding, the inputs are averaged before the head; the 2D center head uses the mean of all four outputs. The reconstructed components are supervised by the same 2D and 3D objectives as the detection loss.

The second component is an object reconstruction loss, which requires the reasoning queries to reconstruct the 3D box of their object so that the learned representations carry genuine geometric meaning rather than merely matching the text embeddings. As illustrated in Fig. 4, each attribute output $\hat { E } _ { a }$ is decoded into the box component it reasons about, aligning every reasoning query with a concrete geometric target and reinforcing attribute-specific specialization. The occlusion output $\hat { E } _ { o c c }$ has no box target of its own; since visibility governs how reliable the visual evidence is for the other attributes, it is instead broadcast as shared context to the dimension, depth, and angle heads. When a head receives more than one embedding, the inputs are averaged before being passed to the head, each attribute head takes the mean of its own output and $\hat { E } _ { o c c } .$ and the 2D center head takes the mean of all four outputs $\begin{array} { r } { \bar { E } = \frac { 1 } { 4 } \sum _ { a } \hat { E } _ { a } , } \end{array}$ since the projected center depends on the object as a whole rather than any single attribute. We supervise the resulting boxes with the same 2D and 3D objectives as the detection loss, giving $\mathcal { L } _ { \mathrm { o b j } } = \mathcal { L } _ { \mathrm { 2 D } } + \mathcal { L } _ { \mathrm { 3 D } }$ . The total reconstruction loss combines both components:

$$
\mathcal { L } _ { \mathrm { r e s } } = \mathcal { L } _ { \mathrm { e m b } } + \mathcal { L } _ { \mathrm { o b j } } .\tag{8}
$$

## 4 Experiments

## 4.1 Settings

Dataset. We evaluated our model on the widely-used KITTI 3D object detection benchmark [7], comprising 7481 training and 7518 testing images. Following Chen et al. [5], we further split the training data into 3712 training and 3769 validation images, allowing ablation studies on the contribution of each component in our MonoLaDN model.

Evaluation metrics. We assessed detection performance across three dificulty levels: easy, moderate, and hard. Two principal metrics were employed: $A P _ { 3 D }$ and $A P _ { B E V }$ , which respectively quantify the accuracy of 3D bounding box predictions and the accuracy of their two-dimensional projections onto the bird’s-eye view. Both metrics were computed at 40 recall positions [31].

Implementation Details. We build MonoLaDN upon MonoDGP [26] with a ResNet-50 backbone, keeping its detection loss $\mathcal { L } _ { \mathrm { d e t } }$ unchanged. The reasoning branch uses SpaceThinker-Qwen2.5-VL-3B [28] to generate the geometric reasoning text and SigLIP (base-patch16-224) [43] to embed it, and a β-VAE with fixed $\beta = 4 . 0$ injects noise into these embeddings to form the perturbed queries. We train with AdamW for 250 epochs, using weight decay $1 0 ^ { - 4 }$ , an initial learning rate of $2 \times 1 0 ^ { - 4 }$ halved at epochs 85, 125, 165, and 225, and batch size $\begin{array} { r } { 8 , } \end{array}$ on a single NVIDIA RTX 3090. At inference, the VLM and reasoning branch are removed, and predictions below 0.2 category confidence are filtered out.

## 4.2 Main Results

Experiment on the KITTI 3D test set. As shown in Tab. 1, MonoLaDN establishes state-of-the-art performance on the KITTI test set for the car category. Without leveraging any auxiliary data, it attains 29.67, 20.45, and 18.02 $A P _ { 3 D | R 4 0 }$ under the easy, moderate, and hard settings, surpassing all prior monocular approaches, including those using auxiliary LiDAR or depth supervision. Relative to our baseline MonoDGP [26], MonoLaDN yields improvements of $+ 3 . 3 2 , \ + 1 . 7 3 .$ and $+ 2 . 0 5 \ A P _ { 3 D | R 4 0 }$ across the three dificulty levels, and it consistently exceeds the strongest competing methods, MonoDLGD [13] and MonoCLUE [40], on all test metrics. The corresponding gains in $A P _ { B E V | R 4 0 }$ further indicate that the improvement stems from more accurate 3D localization rather than image-plane alignment.

Experiment on the KITTI 3D val set. A consistent trend is observed on the validation set, as presented in Tab. 1. MonoLaDN achieves 35.21, 25.03, and 21.60 $A P _ { 3 D | R 4 0 }$ under the easy, moderate, and hard settings, improving upon MonoDGP [26] by +4.45, +2.69, and +2.58, respectively. It ranks first on the easy level and second on the moderate and hard levels, remaining highly competitive with MonoDLGD [13]. The consistent margins over the baseline on both $A P _ { 3 D | R 4 0 }$ and $A P _ { B E V | R 4 0 }$ demonstrate that transferring the geometric reasoning of the VLM during training yields more accurate 3D detection without incurring any inference cost.

Table 1: Comparisons with state-of-the-art monocular methods on the KITTI test and val sets for the car category. We bold the best results and underline the second-best results. The blue refers to the gain and the red is the decrease of our method.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Extra data</td><td rowspan="2">Venue</td><td colspan="3">Test, AP3D|R40</td><td colspan="3">Test, APBEV|R40</td><td colspan="3">Val, AP3D|R40</td><td colspan="3">Val, APBEV|R40</td></tr><tr><td>Easy</td><td>Mod.</td><td>Hard</td><td>Easy</td><td>Mod.</td><td>Hard</td><td>Easy</td><td>Mod.</td><td>Hard</td><td>Easy</td><td>Mod.</td><td>Hard</td></tr><tr><td>MonoDTR [9]</td><td rowspan="3">LiDAR</td><td>CVPR 2022</td><td>21.99</td><td>15.39</td><td>12.73</td><td>28.59</td><td>20.38</td><td>17.14</td><td>24.52</td><td>18.57</td><td>15.51</td><td>33.33</td><td>25.35</td><td>21.68</td></tr><tr><td>DID-M3D [24]</td><td>ECCV 2022</td><td>24.40</td><td>16.29</td><td>13.75</td><td>32.95</td><td>22.76</td><td>19.83</td><td>22.98</td><td>16.12</td><td>14.03</td><td>31.10</td><td>22.76</td><td>19.50</td></tr><tr><td>OccupancyM3D [25]</td><td>CVPR 2024</td><td>25.55</td><td>17.02</td><td>14.79</td><td>35.38</td><td>24.18</td><td>21.37</td><td>26.87</td><td>19.96</td><td>17.15</td><td>35.72</td><td>26.60</td><td>23.68</td></tr><tr><td>MonoPGC [37]</td><td rowspan="2">Depth</td><td>ICRA 2023</td><td>24.68</td><td>17.17</td><td>14.14</td><td>32.50</td><td>23.14</td><td>20.30</td><td>25.67</td><td>18.63</td><td>15.65</td><td>34.06</td><td>24.26</td><td>20.78</td></tr><tr><td>OPA-3D [33]</td><td>RAL 2023</td><td>24.60</td><td>17.05</td><td>14.25</td><td>33.54</td><td>22.53</td><td>19.22</td><td>24.97</td><td>19.40</td><td>16.59</td><td>33.80</td><td>25.51</td><td>22.13</td></tr><tr><td>MonoCon [19]</td><td rowspan="7"></td><td>AAAI 2022</td><td>22.50</td><td>16.46</td><td>13.95</td><td>31.12</td><td>22.10</td><td>19.00</td><td>26.33</td><td>19.01</td><td>15.98</td><td></td><td></td><td></td></tr><tr><td>DEVIANT [12]</td><td>ECCV 2022</td><td>21.88</td><td>14.46</td><td>11.89</td><td>29.65</td><td>20.44</td><td>17.43</td><td>24.63</td><td>16.54</td><td>14.52</td><td>32.60</td><td>23.04</td><td>19.99</td></tr><tr><td>MonoDDE [17]</td><td>CVPR 2022</td><td></td><td>24.93 17.1415.10</td><td></td><td>33.58</td><td>23.46</td><td>20.37</td><td>26.66</td><td>19.75</td><td>16.72</td><td>35.51</td><td>26.48</td><td>23.07</td></tr><tr><td>MonoUNI [10]</td><td>NeurIPS 2023</td><td>24.75</td><td>16.73</td><td>13.49</td><td></td><td></td><td></td><td></td><td>24.51 17.18</td><td>14.01</td><td></td><td></td><td></td></tr><tr><td>MonoDETR [45]</td><td>ICCV 2023</td><td>25.00</td><td>16.47</td><td>13.58</td><td>33.60</td><td>22.11 18.60</td><td></td><td>28.84</td><td>20.61</td><td>16.38</td><td>37.86 26.95 22.80</td><td></td><td></td></tr><tr><td>MonoCD [39]</td><td>CVPR 2024</td><td>25.53</td><td>16.59</td><td>14.53</td><td>33.41</td><td>22.81</td><td>19.57</td><td>26.45</td><td>19.37</td><td>16.38</td><td>34.60</td><td>24.96</td><td>21.51</td></tr><tr><td>FD3D [38]</td><td>AAAI 2024</td><td>25.38</td><td>17.12</td><td>14.50</td><td>34.20</td><td>23.72</td><td>20.76</td><td>28.22</td><td>20.23</td><td>17.04</td><td>36.98</td><td>26.77</td><td>23.16</td></tr><tr><td>MonoDGP [26]</td><td></td><td>CVPR 2025</td><td>26.35</td><td>18.72</td><td>15.97</td><td>35.24 25.23</td><td>22.02</td><td></td><td>30.76</td><td>22.34</td><td>19.02</td><td>39.40 28.20</td><td>24.42</td><td></td></tr><tr><td>MonoDLGD [13]</td><td></td><td>AAAI 2026</td><td>29.11</td><td>19.87</td><td>17.74</td><td>36.63</td><td>25.30</td><td>23.13</td><td>34.89</td><td></td><td>25.19 21.78</td><td>41.68</td><td>30.53</td><td>27.76</td></tr><tr><td>MonoCLUE [40]</td><td></td><td>AAAI 2026</td><td>27.94</td><td>19.70</td><td>16.69</td><td>36.15</td><td>26.15</td><td>22.81</td><td>33.74</td><td>24.10</td><td>20.58</td><td>41.79</td><td>29.91</td><td>26.00</td></tr><tr><td>MonoLaDN (Ours)</td><td>None</td><td></td><td></td><td>29.67 20.45 18.02</td><td></td><td>36.94 26.81 23.73</td><td></td><td></td><td>35.21 25.03</td><td></td><td>21.60</td><td></td><td>42.09 30.75 27.89</td><td></td></tr><tr><td>Improvement vs second-best</td><td></td><td></td><td></td><td>+0.56 +0.58 +0.28</td><td></td><td>+0.31 +0.66 +0.60</td><td></td><td></td><td></td><td>+0.32 -0.16</td><td>-0.18</td><td></td><td>+0.30 +0.22 +0.13</td><td></td></tr><tr><td>Improvement vs MonoDGP</td><td></td><td></td><td></td><td>+3.32 +1.73 +2.05</td><td></td><td>+1.70 +1.58 +1.71</td><td></td><td></td><td></td><td></td><td>+4.45 +2.69 +2.58</td><td></td><td>+2.69 +2.55 +3.47</td><td></td></tr></table>

Cross-Dataset Evaluation. To assess generalization, we follow DEVIANT [12] and evaluate the KITTI-trained model on the car category of the nuScenes frontal validation set with zero fine-tuning. As shown in Tab. 2a, MonoLaDN attains the lowest depth MAE across all ranges on KITTI, and the lowest or second-lowest error on nuScenes despite the large domain shift. This indicates that the transferred geometric reasoning improves robustness to domain shift rather than overfitting to the source dataset. DEVIANT [12] occasionally yields lower far-range error, owing to its equivariance to depth translations.

Eficiency. As shown in Tab. 2b, our proposed MonoLaDN maintains the same computational budget as its baseline MonoDGP [26] while achieving a substantial boost in $A P _ { 3 D }$ . This is because the VLM and the entire language-guided reasoning branch are used only during training and fully discarded at inference, leaving the deployed detector identical to the baseline in parameters, FLOPs, and runtime. Furthermore, compared to the strongest competing methods MonoCLUE [40] and MonoDLGD [13], MonoLaDN demonstrates clear dominance, outperforming both in 3D detection accuracy and inference eficiency.

Table 2: Cross-dataset depth accuracy and computational complexity. (a) Depth prediction MAE (↓) on KITTI and nuScenes frontal validation. (b) Complexity on a single RTX 3090 GPU with batch size 1. Best in bold, second best underlined.

(a) Cross-dataset depth prediction MAE (↓).
<table><tr><td rowspan="2">Method</td><td colspan="4">KITTI Val</td><td colspan="4">nuScenes frontal Val</td></tr><tr><td>0-20</td><td>20-40</td><td>40-∞</td><td>All</td><td>0-20</td><td>20-40</td><td>40-∞</td><td>All</td></tr><tr><td>GUPNet [21]</td><td>0.45</td><td>1.10</td><td>1.85</td><td>0.89</td><td>0.82</td><td>1.70</td><td>6.20</td><td>1.45</td></tr><tr><td>DEVIANT [12]</td><td>0.40</td><td>1.09</td><td>1.80</td><td>0.87</td><td>0.76</td><td>1.60</td><td>4.50</td><td>1.26</td></tr><tr><td>MonoCon [19]</td><td>0.40</td><td>1.08</td><td>1.78</td><td>0.85</td><td>0.78</td><td>1.65</td><td>6.02</td><td>1.40</td></tr><tr><td>MonoUNI [10]</td><td>0.38</td><td>0.92</td><td>1.79</td><td>0.87</td><td>0.72</td><td>1.79</td><td>4.98</td><td>1.43</td></tr><tr><td>MonoCD [39]</td><td>0.37</td><td>1.04</td><td>1.72</td><td>0.83</td><td>0.73</td><td>1.59</td><td>5.78</td><td>1.33</td></tr><tr><td>MonoLaDN (ours)</td><td>0.32</td><td>0.87</td><td>1.54</td><td>0.81</td><td>0.55</td><td>1.62</td><td>4.76</td><td>1.28</td></tr></table>

(b) Computational complexity.
<table><tr><td>Method</td><td>Params (M)</td><td>FLOPs ↓ (G)</td><td> $\mathrm { A P } _ { 3 D | R 4 0 } \uparrow$  (Moderate)</td><td>Runtime ↓ (ms)</td></tr><tr><td>MonoDGP [26]</td><td>42.16</td><td>68.99</td><td>18.72</td><td>42.4</td></tr><tr><td>MonoCLUE [40]</td><td>44.17</td><td>72.71</td><td>19.70</td><td>52.5</td></tr><tr><td>MonoDLGD [13]</td><td>42.16</td><td>69.21</td><td>19.87</td><td>42.8</td></tr><tr><td>MonoLaDN (ours)</td><td>42.16</td><td>68.99</td><td>20.45</td><td>42.4</td></tr></table>

## 4.3 Ablation Study

We verify the efectiveness of each of our components in Tab. 3 (more ablation can be found in the supplementary material), reporting $\mathbf { A P _ { 3 D } }$ for the Car category on the KITTI val set.

Efectiveness of the reasoning branch. Starting from the MonoDGP [26] baseline (a), we introduce the reasoning branch with the embedding loss ${ \mathcal { L } } _ { \mathrm { e m b } }$ (b). Since the perturbed queries are derived from the ground-truth boxes, the leakage-prevention mask $A ^ { l p }$ is enabled alongside the reasoning branch, preventing the learnable queries from accessing ground-truth reasoning as a shortcut, and is kept on in all subsequent settings. Moderate $A P _ { 3 D }$ improves from 22.34 to 22.55, with consistent gains on easy and hard, showing that reconstructing the VLM reasoning embeddings from their noised form is a useful training signal for the detector’s 3D geometry.

Choice of text encoder. We next compare encoders for embedding the reasoning text. Replacing CLIP (ViT-B/32) [27] (b) with SigLIP (base-patch16- 224) [43] (c) raises moderate $A P _ { 3 D }$ from 22.55 to 22.72 and improves both other settings, indicating that SigLIP yields representations better aligned with our geometric reasoning. We therefore adopt SigLIP in all remaining experiments.

Efectiveness of the reconstruction objectives. We then study the two reconstruction losses in isolation. The object loss $\mathcal { L } _ { \mathrm { o b j } }$ (d) alone reaches 23.15 moderate $A P _ { 3 D }$ , outperforming the embedding loss ${ \mathcal { L } } _ { \mathrm { e m b } }$ (c) at 22.72 and showing that reconstructing object-level targets is the stronger supervisory signal. Combining both (e) further improves the result to 23.61, confirming that the embedding and object objectives are complementary rather than redundant.

Efectiveness of the Separated Reasoning mask. Adding the Separated Reasoning mask $A ^ { s r } \mathrm { ~ ( f ) ~ }$ raises moderate $A P _ { 3 D }$ to 24.52. By routing each reasoning query to its related context, the reasoning queries of the same object and those of the same attribute across objects, $A ^ { s r }$ lets each attribute be reconstructed from informative rather than unrelated cues, improving fidelity.

Choice of reasoning VLM. Finally, we vary the model that generates the reasoning text. Replacing Qwen2.5-VL-7B (f) with InternVL3.5-8B (g) improves moderate $A P _ { 3 D }$ from 24.52 to 24.83, with consistent gains on easy and hard, indicating that stronger VLM reasoning translates into more accurate detection. Using SpaceThinker-Qwen2.5-VL-3B (h), a variant explicitly fine-tuned for 3D spatial reasoning, gives the best result at 35.21, 25.03, and 21.60 $A P _ { 3 D }$ , despite being the smallest of the three models. We therefore adopt it as our reasoning VLM. This trend suggests that our method will benefit directly from future VLMs with stronger 3D reasoning, gaining accuracy without any change to the detector. Across all variants, the detector remains VLM-free at inference, so improved reasoning quality comes at no deployment cost.

Table 3: Ablation of each component on the KITTI val set. $\mathcal { L } _ { \mathrm { e m b } } / \mathcal { L } _ { \mathrm { o b j } }$ are the embedding and box reconstruction losses. $A ^ { l p } / A ^ { s r }$ the leakage-prevention and separated-reasoning masks. Text Encoder embeds the reasoning text, and VLM is the model that generates it. The best results are in bold.
<table><tr><td></td><td colspan="7">Ablation</td><td colspan="3">Val,  $A P _ { 3 D | R 4 0 }$ </td></tr><tr><td></td><td> $\mathcal { L } _ { \mathrm { e m b } }$ </td><td> $\mathcal { L } _ { \mathrm { o b j } }$ </td><td> $A ^ { l p }$ </td><td> $A ^ { s r }$ </td><td>Text Encoder</td><td>VLM</td><td>Easy</td><td>Mod.</td><td></td><td>Hard</td></tr><tr><td>(a)</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>30.76</td><td></td><td>22.34</td><td>19.02</td></tr><tr><td>(b)</td><td>√</td><td>X</td><td>√</td><td>X</td><td>CLIP [27]</td><td></td><td>Qwen2.5-VL [1]</td><td>31.41</td><td>22.55</td><td>19.34</td></tr><tr><td>(c)</td><td>√</td><td>X</td><td>√</td><td>X</td><td>SigLIP [43]</td><td>Qwen2.5-VL [1]</td><td></td><td>31.63</td><td>22.72</td><td>19.53</td></tr><tr><td>(d)</td><td>X</td><td>√</td><td>√</td><td>X</td><td>SigLIP [43]</td><td>Qwen2.5-VL [1]</td><td></td><td>32.07</td><td>23.15</td><td>19.98</td></tr><tr><td>(e)</td><td>√</td><td>√</td><td>√</td><td>X</td><td>SigLIP [43]</td><td>Qwen2.5-VL [1]</td><td></td><td>33.46</td><td>23.61</td><td>20.57</td></tr><tr><td>(f)</td><td>√</td><td>√</td><td>V</td><td>√</td><td>SigLIP [43]</td><td>Qwen2.5-VL [1]</td><td></td><td>34.78</td><td>24.52</td><td>21.13</td></tr><tr><td>(g)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>SigLIP [43]</td><td>InternVL3.5 [36]</td><td></td><td>35.04</td><td>24.83</td><td>21.47</td></tr><tr><td>(h)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>SigLIP [43]</td><td>SpaceThinker [28]</td><td></td><td>35.21</td><td>25.03</td><td>21.60</td></tr></table>

## 4.4 Qualitative Results

Fig. 5 presents qualitative comparisons on the KITTI val set against MonoDETR [45] and MonoDGP [26]. The 3D boxes predicted by MonoLaDN align more closely with the ground truth in both the image and bird’s-eye views, particularly in localization along the depth axis. The improvement is most evident for distant and partially occluded objects, where the baselines tend to misestimate depth or object extent, producing boxes that are shifted or poorly oriented in the bird’s-eye view, whereas our predictions stay better aligned. These results are consistent with our quantitative gains and indicate that the geometric reasoning transferred from the VLM during training helps the detector recover more accurate 3D geometry.

## 5 Conclusion

In this work, we presented MonoLaDN, a framework that transfers geometric reasoning from a vision-language model into a monocular 3D detector while keeping inference purely vision-based. During training, a frozen VLM provides structured reasoning about occlusion, depth, dimension, and angle, which is encoded into noisy queries for reconstruction. A language-guided attention mask prevents ground-truth shortcuts and restricts each attribute to relevant context. The reasoning branch is removed at inference, introducing no additional cost. Experiments on KITTI 3D and nuScenes show consistent improvements over state-of-the-art detectors and strong cross-domain generalization.

![](images/f41445a0788b0e330409f6c47d06b11b81b3bb8beb61488758a0d3161ff03704.jpg)  
(a) MonoDETR  
(b) MonoDGP  
(c) MonoLaDN (Ours)  
Fig. 5: Qualitative results on KITTI val set. (a) MonoDETR [45]. (b) MonoDGP [26]. (c) MonoLaDN (Ours). For each image set, the top row presents the camera-view visualization, while the bottom row ofers the corresponding bird’s-eye view. Ground-truth bounding boxes are rendered in green, and predictions are shown in order: red, yellow, blue. We also circle some objects to highlight the diference between other state-of-the-art methods and our method.

Limitations and future work. Our approach currently targets single-image detection. Extending the Language-Guided DeNoising to multi-view and temporal settings is left for future work.

## Acknowledgement

We acknowledge Ho Chi Minh City University of Technology (HCMUT), VNU-HCM for supporting this study.

## References

1. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)

2. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: Endto-end object detection with transformers. In: European conference on computer vision. pp. 213–229. Springer (2020)

3. Chang, F., Li, S., Li, Y., He, L.: Vlm-3d: end-to-end vision-language models for open-world 3d perception. arXiv preprint arXiv:2508.09061 (2025)

4. Chen, H., Dou, Z., Mao, K., Liu, J., Zhao, Z.: Generalizing conversational dense retrieval via llm-cognition data augmentation. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 2700–2718 (2024)

5. Chen, X., Kundu, K., Zhu, Y., Berneshawi, A.G., Ma, H., Fidler, S., Urtasun, R.: 3d object proposals for accurate object class detection. In: Advances in Neural Information Processing Systems. vol. 28. Curran Associates, Inc. (2015)

6. Dong, Z., Ji, H., Huang, X., Zhang, W., Zhan, X., Chen, J.: Pep: a point enhanced painting method for unified point cloud tasks. CoRR (2023)

7. Geiger, A., Lenz, P., Urtasun, R.: Are we ready for autonomous driving? the kitti vision benchmark suite. In: 2012 IEEE Conference on Computer Vision and Pattern Recognition. pp. 3354–3361 (2012)

8. Gu, X., Lin, T.Y., Kuo, W., Cui, Y.: Open-vocabulary object detection via vision and language knowledge distillation. In: International Conference on Learning Representations (2021)

9. Huang, K.C., Wu, T.H., Su, H.T., Hsu, W.H.: Monodtr: Monocular 3d object detection with depth-aware transformer. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 4002–4011 (2022)

10. Jia, J., Li, Z., Shi, Y.: Monouni: A unified vehicle and infrastructure-side monocular 3d object detection network with suficient depth clues. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)

11. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)

12. Kumar, A., Brazil, G., Corona, E., Parchami, A., Liu, X.: Deviant: Depth equivariant network for monocular 3d object detection. In: European Conference on Computer Vision. pp. 664–683. Springer (2022)

13. Lee, S., Baek, S., Min, D.: Dificulty-aware label-guided denoising for monocular 3d object detection. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 5918–5926 (2026)

14. Li, F., Zhang, H., Liu, S., Guo, J., Ni, L.M., Zhang, L.: Dn-detr: Accelerate detr training by introducing query denoising. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 13619–13627 (2022)

15. Li, L., Miao, J., Shi, D., Tan, W., Ren, Y., Yang, Y., Pu, S.: Distilling detr with visual-linguistic knowledge for open-vocabulary object detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6501–6510 (2023)

16. Li, Z., Wang, W., Li, H., Xie, E., Sima, C., Lu, T., Yu, Q., Dai, J.: Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. ArXiv (2022)

17. Li, Z., Qu, Z., Zhou, Y., Liu, J., Wang, H., Jiang, L.: Diversity matters: Fully exploiting depth clues for reliable monocular 3d object detection. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 2781–2790 (2022)

18. Liu, F., Huang, T., Zhang, Q., Yao, H., Zhang, C., Wan, F., Ye, Q., Zhou, Y.: Ray denoising: Depth-aware hard negative sampling for multi-view 3d object detection. In: European Conference on Computer Vision. pp. 200–217. Springer (2024)

19. Liu, X., Xue, N., Wu, T.: Learning auxiliary monocular contexts helps monocular 3d object detection. In: AAAI Conference on Artificial Intelligence (2021)

20. Liu, Y., Wang, T., Zhang, X., Sun, J.: Petr: Position embedding transformation for multi-view 3d object detection. ArXiv (2022)

21. Lu, Y., Ma, X., Yang, L., Zhang, T., Liu, Y., Chu, Q., Yan, J., Ouyang, W.: Geometry uncertainty projection network for monocular 3d object detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3111–3121 (2021)

22. Ma, W., Chou, Y.C., Liu, Q., Wang, X., de Melo, C., Xie, J., Yuille, A.: Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning. Advances in Neural Information Processing Systems 38, 140751–140774 (2026)

23. Man, Y., Wang, S., Zhang, G., Bjorck, J., Gui, L.Y., Fan, J., Kautz, J., Wang, Y.X., Yu, Z.: Locateanything3d: Vision-language 3d detection with chain-of-sight. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 31089–31102 (2026)

24. Peng, L., Wu, X., Yang, Z., Liu, H., Cai, D.: Did-m3d: Decoupling instance depth for monocular 3d object detection. In: Computer Vision – ECCV 2022. pp. 71–88 (2022)

25. Peng, L., Xu, J., Cheng, H., Yang, Z., Wu, X., Qian, W., Wang, W., Wu, B., Cai, D.: Learning occupancy for monocular 3d object detection. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 10281–10292 (2023)

26. Pu, F., Wang, Y., Deng, J., Yang, W.: Monodgp: Monocular 3d object detection with decoupled-query and geometry-error priors. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6520–6530 (2025)

27. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

28. Remyx AI: SpaceThinker-Qwen2.5VL-3B. https://huggingface.co/remyxai/ SpaceThinker-Qwen2.5VL-3B (2025), hugging Face model repository. Accessed 2026-06-20

29. Sennrich, R., Haddow, B., Birch, A.: Improving neural machine translation models with monolingual data. In: Proceedings of the 54th annual meeting of the association for computational linguistics (volume 1: long papers). pp. 86–96 (2016)

30. Shi, S., Guo, C., Jiang, L., Wang, Z., Shi, J., Wang, X., Li, H.: Pv-rcnn: Point-voxel feature set abstraction for 3d object detection. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10526–10535 (2020)

31. Simonelli, A., Bulò, S.R., Porzi, L., López-Antequera, M., Kontschieder, P.: Disentangling monocular 3d object detection. 2019 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 1991–1999 (2019)

32. Spithourakis, G., Riedel, S.: Numeracy for language models: Evaluating and improving their ability to predict numbers. In: Proceedings of the 56th Annual Meet-

ing of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 2104–2115 (2018)

33. Su, Y., Di, Y., Manhardt, F., Zhai, G., Rambach, J.R., Busam, B., Stricker, D., Tombari, F.: Opa-3d: Occlusion-aware pixel-wise aggregation for monocular 3d object detection. IEEE Robotics and Automation Letters 8, 1327–1334 (2022)

34. Sundararaman, D., Si, S., Subramanian, V., Wang, G., Hazarika, D., Carin, L.: Methods for numeracy-preserving word embeddings. In: Conference on Empirical Methods in Natural Language Processing (2020)

35. Wang, L., Liu, Y., Du, P., Ding, Z., Liao, Y., Qi, Q., Chen, B., Liu, S.: Objectaware distillation pyramid for open-vocabulary object detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11186–11196 (2023)

36. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265 (2025)

37. Wu, Z., Gan, Y.Z., Wang, L., Chen, G., Pu, J.: Monopgc: Monocular 3d object detection with pixel geometry contexts. 2023 IEEE International Conference on Robotics and Automation (ICRA) pp. 4842–4849 (2023)

38. Wu, Z., Gan, Y., Wu, Y., Wang, R., Wang, X., Pu, J.: Fd3d: Exploiting foreground depth map for feature-supervised monocular 3d object detection. In: AAAI Conference on Artificial Intelligence (2024)

39. Yan, L., Yan, P., Xiong, S., Xiang, X., Tan, Y.: Monocd: Monocular 3d object detection with complementary depths. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 10248–10257 (2024)

40. Yang, S., Lee, M., Lee, J., Lee, S.: Monoclue: Object-aware clustering enhances monocular 3d object detection. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 11721–11729 (2026)

41. Yang, T., Ju, Y., Yi, L.: Imov3d: Learning open vocabulary point clouds 3d object detection from only 2d images. Advances in Neural Information Processing Systems 37, 141261–141291 (2024)

42. Yin, T., Zhou, X., Krähenbühl, P.: Center-based 3d object detection and tracking. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 11779–11788 (2020)

43. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11975–11986 (2023)

44. Zhang, J., Dong, R., Ma, K.: Clip-fo3d: Learning free open-world 3d scene representations from 2d dense clip. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 2048–2059 (2023)

45. Zhang, R., Qiu, H., Wang, T., Guo, Z., Cui, Z., Xu, X., Qiao, Y.J., Gao, P., Li, H.: Monodetr: Depth-guided transformer for monocular 3d object detection. 2023 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 9121–9132 (2022)

46. Zuo, Y., Kayan, K., Wang, M., Jeon, K., Deng, J., Grifiths, T.L.: Towards foundation models for 3d vision: How close are we? In: 2025 International Conference on 3D Vision (3DV). pp. 1285–1296. IEEE (2025)

# MonoLaDN: Language-Guided DeNoising for Monocular 3D Object Detection

Supplementary Material

Kiet Dang Vu<sup>1,2</sup> , Cao Le Minh Khoa<sup>1,2</sup> , Nguyen Viet Tri Pham<sup>1,2</sup> , An Nguyen Gia<sup>1,2</sup> , Tam Le Thuc Ho<sup>1,2</sup> , and Duc Dung Nguyen<sup>1,2⋆</sup>

<sup>1</sup> AITech Lab., Ho Chi Minh City University of Technology (HCMUT) <sup>2</sup> Vietnam National University Ho Chi Minh City (VNUHCM)

A Addition Explanation of MonoLaDN

## A.1 Where Do the Reasoning Queries Attend?

To understand why the transferred reasoning improves detection, we visualize where the reasoning queries attend in the image. For each ground-truth object, we extract the deformable sampling locations and attention weights of its four attribute queries at the final decoder layer, and accumulate them into per-attribute Gaussian heatmaps over the input image (Fig. 1).

The four attributes attend to clearly distinct and semantically meaningful regions, rather than collapsing onto a single object blob. The occlusion query spreads over the object boundary and the surrounding context, where evidence of truncation or overlap with neighboring objects appears; for truncated cases it concentrates toward the image border that cuts the object. The depth query attends to the ground-contact region and the scene context around the object, which carry the perspective cues needed to localize distance. The dimension query covers the full extent of the object body, consistent with reasoning about overall shape and size. The angle query focuses on orientation-revealing parts such as the visible side profile, front, or rear faces, matching the viewpoint stated in its reasoning.

Crucially, these attention patterns are consistent with the natural-language reasoning generated for each attribute. For instance, when the reasoning describes a left-facing side view, the angle query concentrates on the side profile of the vehicle, and when it reports truncation by the border, the occlusion query attends to the cut-of region. This alignment shows that the reasoning queries are spatially grounded in the image evidence rather than memorizing the groundtruth annotation, so reconstructing them forces the detector’s depth-guided features to encode genuine, attribute-specific geometric cues that remain useful after the reasoning path is removed at inference.

![](images/add1ea186038d12eaadf156a4182907e3a5ad832bb9d6daf32aac8ddee958e79.jpg)  
Fig. 1: Deformable cross-attention of the reasoning queries on the KITTI val set. Each row is one object, and the four columns show the occlusion, depth, dimension, and angle reasoning queries. For each query, we accumulate its deformable sampling locations at the final decoder layer, weighted by the corresponding attention weights, into a Gaussian heatmap overlaid on the image (red high, blue low). The target object is marked in green, and the generated reasoning for each attribute is shown beneath its column.

## A.2 How Reasoning Queries Exchange Information?

We next examine how information flows among the queries inside the decoder, by visualizing the self-attention matrix together with the structure imposed by the Language-Guided Mask (Fig. 2). The queries are ordered as the 4M reasoning queries followed by the N = 50 learnable queries, matching the mask defined.

![](images/cc11c5b67abcd515ccb34b85475e85b82a01e0f2d47cc352cb2eb518e7805666.jpg)  
The car's front surface is primarily facing the camera, with the windshield and grille clearly visible.  
Fig. 2: Decoder self-attention under the Language-Guided Mask on the KITTI val set. Each row is one scene. The left column shows the image with al projected ground-truth 3D boxes (the target object in orange, others in green) and the generated reasoning for its four attributes. The right column shows the decoder self-attention matrix over the $W = 4 M + N$ queries, ordered as the 4M reasoning queries followed by the N = 50 learnable queries; the red lines separate the two groups and the thin gray lines delimit the four-attribute block of each object.

The learned attention respects the leakage-prevention mask exactly. The block formed by the learnable-query rows and the reasoning-query columns is empty, confirming that no learnable detection query attends to any reasoning query. The learnable queries therefore cannot copy the ground-truth-derived reasoning as a shortcut, which is essential since this path is removed at inference; without it, the detector would rely on information that is unavailable at test time and collapse, as shown in our mask ablation.

Within the reasoning block, the attention follows the separated-reasoning structure rather than spreading uniformly. Each object forms a dense $4 \times 4$ block on the diagonal, showing that the four attributes of the same object exchange information with one another, while a regular of-diagonal pattern links queries of the same attribute across diferent objects. This confirms that intraobject cross-attribute communication carries real weight, which is precisely why our separated-reasoning mask outperforms the stricter attribute-isolated variant that severs these links. Meanwhile, the reasoning queries attend to the learnable queries, so the geometric reasoning is injected into the detection queries during training. Together, these patterns show that the Language-Guided Mask routes information as designed: it blocks ground-truth shortcuts while letting each attribute reconstruct itself from informative context, allowing the detector’s features to absorb the VLM’s geometric reasoning that remains available after the reasoning path is discarded.

## B Detail of 3D Reasoning Generation

This section details the ofline reasoning-generation pipeline used to supervise the denoising branch. For each ground-truth object, we (i) derive a discrete ordinal label for each of the four attributes directly from the 3D annotation, (ii) assemble a prompt from these labels and object metadata, and (iii) query a frozen VLM to produce one evidence sentence per attribute. The entire process is run once before training and cached.

## B.1 Geometry-to-Language Labels

Rather than asking the VLM to regress continuous geometry, we map each annotation to a discrete label using fixed, deterministic rules. Tab. 1 summarizes the rules for occlusion, depth, and angle, and the dimension rules are described below.

Occlusion. The KITTI truncation and occlusion flags are thresholded (truncation > 0.15, occlusion > 0) and combined into one of four visibility labels, capturing whether the object is fully visible, occluded by another object, truncated by the image border, or both.

Depth. The camera-frame depth z is binned by the boundaries [10, 20, 35, 50] m into five distance labels, from very near to very far.

Angle. The observation angle α is mapped by its magnitude |α| into cardinal views (front/rear/left/right) and oblique views, with the sign of α selecting the left or right variant, giving eight viewpoint labels in total.

Table 1: Geometry-to-language label rules. Each attribute is mapped from the ground-truth annotation to a discrete ordinal label. α is the observation angle, z the camera-frame depth, and $h / w / l$ the box height/width/length.
<table><tr><td rowspan=1 colspan=1>Attribute</td><td rowspan=1 colspan=1>Rule</td><td rowspan=1 colspan=1>Label</td></tr><tr><td rowspan=1 colspan=1>Occlusion</td><td rowspan=1 colspan=1> $\mathrm { t r u n c } \leq 0 . 1 5 , \mathrm { o c c } = 0$  $\mathrm { o c c } > 0$  $\mathrm { t r u n c } > 0 . 1 5$  $\mathrm { t r u n c } > 0 . 1 5 , \mathrm { o c c } > 0$ </td><td rowspan=1 colspan=1>Fully visiblePartially occludedTruncated by borderTruncated and occluded</td></tr><tr><td rowspan=1 colspan=1>Depth (z)</td><td rowspan=1 colspan=1> $z \leq 1 0$  $1 0 < z \le 2 0$  $2 0 < z \le 3 5$  $3 5 < z \le 5 0$  $z > 5 0$ </td><td rowspan=1 colspan=1>Very nearNearMid-distanceFarVery far</td></tr><tr><td rowspan=1 colspan=1>Angle (|α|)</td><td rowspan=1 colspan=1> $| \alpha | \leq 2 5 ^ { \circ }$  $| \alpha | \geq 1 5 5 ^ { \circ }$  $6 5 ^ { \circ } \leq | \alpha | \leq 1 1 5 ^ { \circ }$ otherwise</td><td rowspan=1 colspan=1>Rear-facingFront-facingLeft / right side viewOblique views</td></tr><tr><td rowspan=1 colspan=1>Dimension</td><td rowspan=1 colspan=1> $h \geq 2 . 2$  $w \geq 2 . 2$  $l / h \ge 3 . 6$  $l \leq 3 . 2$ otherwise</td><td rowspan=1 colspan=1>TallWideElongatedCompactModerate-length</td></tr></table>

Dimension. The dimension label is class-dependent. Pedestrians are labeled Upright or Compact, cyclists Elongated, and large vehicles (van, truck, tram) Long or Boxy. For cars, the label is chosen from the height $h ,$ width $w ,$ and length $l \colon h \geq 2 . 2$ m gives Tall, $w \geq 2 . 2$ m gives Wide, $l / h \ge 3 . 6$ gives Elongated, $l \leq 3 . 2$ m gives Compact, and the remaining cars are labeled Moderate-length. This grounds the shape descriptor in the true box proportions while remaining a single discrete token.

## B.2 Prompt Construction

Each attribute is prompted independently. For a given attribute, the frozen VLM is first queried to select one label from that attribute’s ontology, using a system prompt that fixes the task and restricts the reasoning to visible object cues, and a user prompt containing the object metadata (class, camera context, range, region, scale, and border contact) together with the candidate labels and their definitions. Conditioned on the selected label, a second prompt then asks the VLM to write one concise evidence sentence justifying that label from the visible cues. The system prompt and object metadata are shared across the four attributes, while the candidate labels, their definitions, and the resulting evidence difer per attribute. To keep the evidence grounded in the object rather than the input construction, generated sentences are filtered to remove references to prompt artifacts such as crops or overlays.

## B.3 Output Format

For each attribute, the VLM returns a {label, evidence} pair, and the four pairs are concatenated into the per-object reasoning set $\{ R _ { o c c } , R _ { d e p t h } , R _ { d i m } , R _ { a n g l e } \}$ Each $R _ { a }$ therefore consists of a discrete ordinal label grounded in the annotation and a short natural-language justification grounded in the image, which is subsequently embedded and denoised as described in the main paper.

## C More Results

## C.1 Other Categories on KITTI

Beyond the car category reported in the main paper, we evaluate MonoLaDN on the more challenging pedestrian and cyclist categories of the KITTI test set, as shown in Tab. 2. These classes are substantially harder, since their smaller size and greater shape variation provide weaker geometric cues. Despite this, Mono-LaDN attains the best results across all six metrics, surpassing the strongest competing methods MonoDLGD [13] and MonoCLUE [40]. The consistent gains on both categories indicate that the geometric reasoning transferred from the VLM benefits detection beyond the car category and generalizes across object types.

Table 2: Comparison on the KITTI test set for the pedestrian and cyclist categories at $\mathrm { I o U } = 0 . 5$ , reporting $A P _ { 3 D | R 4 0 }$ . Best in bold.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Pedestrian</td><td colspan="3">Cyclist</td></tr><tr><td>Easy</td><td>Mod.</td><td>Hard</td><td>Easy</td><td>Mod.</td><td>Hard</td></tr><tr><td>GUPNet [21]</td><td>14.72</td><td>9.53</td><td>7.87</td><td>4.18</td><td>2.65</td><td>2.09</td></tr><tr><td>MonoCon [19]</td><td>13.10</td><td>8.41</td><td>6.94</td><td>2.80</td><td>1.92</td><td>1.55</td></tr><tr><td>DEVIANT [12]</td><td>13.43</td><td>8.65</td><td>7.69</td><td>5.05</td><td>3.13</td><td>2.59</td></tr><tr><td>MonoDDE [17]</td><td>11.13</td><td>7.32</td><td>6.67</td><td>5.94</td><td>3.78</td><td>3.33</td></tr><tr><td>MonoDETR [45]</td><td>12.65</td><td>7.19</td><td>6.72</td><td>5.12</td><td>2.74</td><td>2.02</td></tr><tr><td>MonoDGP [26]</td><td>15.04</td><td>9.89</td><td>8.38</td><td>5.28</td><td>2.82</td><td>2.65</td></tr><tr><td>MonoDLGD [13]</td><td>15.99</td><td>10.44</td><td>8.82</td><td>6.62</td><td>4.39</td><td>3.35</td></tr><tr><td>MonoCLUE [40]</td><td>16.18</td><td>10.45</td><td>8.75</td><td>5.93</td><td>3.20</td><td>2.94</td></tr><tr><td>MonoLaDN (Ours)</td><td>16.57</td><td>10.81</td><td>9.03</td><td>6.64</td><td>4.45</td><td>3.42</td></tr></table>

## C.2 Training Complexity

While MonoLaDN adds no cost at inference, the auxiliary reasoning path introduces a modest overhead during training. Tab. 3 compares our method against the MonoDGP [26] baseline in terms of training time and peak memory usage. Adding the reasoning branch increases the training time from 4.68 to 5.23 minutes per epoch and the peak memory from 14.35 to 20.45 GB, while raising moderate $A P _ { 3 D }$ from 22.34 to 25.03. This overhead is incurred only during training, and since the VLM reasoning text is generated ofline beforehand, it is not repeated each epoch. At inference, the reasoning path is fully discarded, so the deployed detector retains the exact cost of the baseline.

Table 3: Training Complexity.
<table><tr><td rowspan="2">Method</td><td colspan="3">Val,  $A P _ { 3 D | R 4 0 }$ </td><td rowspan="2">Training time (min/epoch)↓</td><td rowspan="2">Memory Usage (GB) ↓</td></tr><tr><td>Easy</td><td>Mod.</td><td>Hard</td></tr><tr><td>MonoDGP</td><td>30.76</td><td>22.34</td><td>19.02</td><td>4.68</td><td>14.35</td></tr><tr><td>MonoDGP + Ours</td><td>35.21</td><td>25.03</td><td>21.60</td><td>5.23</td><td>20.45</td></tr></table>

## C.3 3D Reasoning Generation Complexity

Since the reasoning text is generated by a frozen VLM, we report the ofline cost of this stage for the three VLMs in Tab. 4, alongside their val $A P _ { 3 D }$ . This generation is performed once before training and cached, so it adds no cost at inference and is not repeated across epochs. SpaceThinker-3B attains the best accuracy at a memory footprint of 9.35 GB, the lowest of the three, as its smaller size ofsets the longer generation time. InternVL3.5-8B is the fastest to generate but requires the most memory, while Qwen2.5-VL-7B sits in between. In all cases, the cost is incurred only once and entirely ofline, so the choice of reasoning VLM afects preprocessing time and memory but never the deployed detector.

## D Additional Ablation Study

## D.1 Ablation on the Language-Guided Mask Design

We study how the connectivity among reasoning queries afects denoising, comparing three mask rules of decreasing connectivity. Beyond the leakage-prevention mask $A ^ { l p }$ and our separated-reasoning mask $A ^ { s r }$ , we introduce an attributeisolated mask $A ^ { a i }$ , a stricter rule that forbids diferent attributes of the same object from attending to one another. Denoting a reasoning query by its object index o and attribute $^ { a , }$ it is defined as

$$
A _ { i j } ^ { a i } = \left\{ \begin{array} { l l } { { 0 , } } & { { \mathrm { i f ~ } i , j \in { \bf q } _ { P } \mathrm { ~ a n d ~ } a _ { i } \ne a _ { j } , } } \\ { { 1 , } } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right.\tag{1}
$$

Table 4: 3D Reasoning Generation Complexity. Comparison of the VLMs used to generate the reasoning text, reporting val $A P _ { 3 D | R 4 0 }$ for the Car category alongside the total generation time over the dataset and the memory cost. The reasoning is generated ofline during training only, so it adds no inference cost.
<table><tr><td rowspan="2">VLM</td><td colspan="3"> $\mathrm { V a l } , A P _ { 3 D | R 4 0 }$ </td><td rowspan="2">Total time (h) ↓</td><td rowspan="2">Memory Usage (GB) ↓</td></tr><tr><td>Easy</td><td>Mod.</td><td>Hard</td></tr><tr><td>Qwen2.5-VL-7B [1]</td><td>34.78</td><td>24.52</td><td>21.13</td><td>220.7</td><td>9.35</td></tr><tr><td>InternVL3.5-8B [36]</td><td>35.04</td><td>24.83</td><td>21.47</td><td>73.6</td><td>17.45</td></tr><tr><td>SpaceThinker-3B [28]</td><td>35.21</td><td>25.03</td><td>21.60</td><td>230.6</td><td>9.35</td></tr></table>

![](images/c57359e6ebaee759412aeb713a47e0dbd4dafaa3d43ad41a27eb1f7570dcdc34.jpg)  
Fig. 3: Designs of the language-guided attention mask. Black entries are blocked and white entries are permitted. The leakage-prevention mask $A ^ { l p }$ stops the learnable queries from attending to any reasoning query. The separated-reasoning mask $A ^ { s r }$ lets each reasoning query attend to all reasoning queries of the same object and to those of the same attribute across objects. The attribute-isolated mask $A ^ { a i }$ is a stricter variant that additionally blocks diferent attributes of the same object from attending to each other.

so two reasoning queries are disconnected whenever they belong to diferent attributes, regardless of whether they share the same object. All experiments here use SigLIP [43] as the text encoder and Qwen2.5-VL-7B [1] as the reasoning VLM, varying only the mask. Tab. 5 reports $A P _ { 3 D }$ for the Car category on the KITTI val set.

Removing $A ^ { l p }$ causes a severe collapse, dropping moderate $A P _ { 3 D }$ from 23.61 to 10.92, since the learnable queries copy the ground-truth reasoning as a shortcut that is unavailable at inference [14]. With $A ^ { l p }$ in place, adding our separatedreasoning mask $A ^ { s r }$ raises moderate $A P _ { 3 D }$ from 23.61 to 24.52, confirming that routing each attribute to its informative context improves reconstruction. Replacing $A ^ { s r }$ with the stricter $A ^ { a i }$ instead yields only 23.81, and combining both further degrades performance to 23.42. These results show that exchanging information across the diferent attributes of the same object is beneficial, whereas isolating each attribute entirely is over-restrictive. Our separated-reasoning design therefore strikes the right balance and is adopted in all main experiments.

Table 5: Ablation on the language-guided mask design on the KITTI val set, reporting $A P _ { 3 D | R 4 0 }$ for the Car category. All rows use SigLIP as the text encoder and Qwen2.5-VL-7B as the reasoning VLM. $A ^ { l p }$ $A ^ { s r }$ , and $A ^ { a i }$ denote the leakageprevention, separated-reasoning, and attribute-isolated masks. The best results are in bold.
<table><tr><td> $A ^ { l p }$ </td><td> $A ^ { s r }$ </td><td> $A ^ { a i }$ </td><td>Easy</td><td>Mod.</td><td>Hard</td></tr><tr><td>X</td><td>X</td><td>X</td><td>17.25</td><td>10.92</td><td>9.87</td></tr><tr><td>√</td><td>X</td><td>X</td><td>33.46</td><td>23.61</td><td>20.57</td></tr><tr><td>√</td><td>√</td><td>X</td><td>34.78</td><td>24.52</td><td>21.13</td></tr><tr><td>√</td><td>X</td><td>√</td><td>33.64</td><td>23.81</td><td>20.72</td></tr><tr><td>√</td><td>√</td><td>V</td><td>33.05</td><td>23.42</td><td>20.11</td></tr></table>

## D.2 Ablation on the Noise Injection Strategy

We compare three strategies for injecting the noise that forms the perturbed queries, difering in the stage at which corruption is applied, as illustrated in Fig. 4. BBox perturbation perturbs the ground-truth 3D boxes directly and then generates reasoning from the corrupted geometry. Reasoning perturbation generates reasoning from the clean boxes and applies text augmentation [4, 29] to the reasoning text before encoding. Embedding perturbation, our main approach, encodes the clean reasoning with the text encoder and injects learned Gaussian noise in the embedding space via the β-VAE [11]. All experiments here use SigLIP [43] as the text encoder and Qwen2.5-VL-7B [1] as the reasoning VLM, varying only the noise strategy, and report $A P _ { 3 D }$ for the Car category on the KITTI val set.

As shown in Tab. 6, the stage at which noise is injected has a clear efect. BBox perturbation performs worst (22.83 moderate $A P _ { 3 D } )$ : since the VLM is prompted with the perturbed boxes, the corrupted geometry no longer matches the visual evidence, so the VLM cannot reason reliably and produces inaccurate reasoning that provides a noisy, inconsistent supervision signal. Reasoning perturbation improves to 23.69 by keeping the geometry intact, but discrete text augmentation gives only coarse, non-diferentiable control over the noise. Our embedding perturbation achieves the best result (24.52), since injecting continuous learned noise directly in the embedding space provides smooth, controllable corruption that is well matched to the reconstruction objective. We therefore adopt embedding perturbation in all main experiments.

## E Additional Qualitative Results

We provide further qualitative comparisons on the KITTI val set in Fig. $5 ,$ covering additional scenes with varying object density, distance, and occlusion. Consistent with the main paper, MonoLaDN produces 3D boxes that align more

![](images/6e6f1ade59a7b6a567ad6da45ddc613919041fa2f17a1f01d851e9c2462a174e.jpg)  
Fig. 4: Strategies for forming the perturbed queries. All three start from the ground-truth 3D boxes and end with perturbed queries, but inject noise at diferent stages. BBox perturbation perturbs the boxes before the VLM, so the reasoning is generated from corrupted geometry. Reasoning perturbation generates clean reasoning and then applies text augmentation to the reasoning text before encoding. Embedding perturbation (ours) encodes the clean reasoning and injects learned noise in the embedding space.

Table 6: Ablation on the noise injection strategy on the KITTI val set, reporting $A P _ { 3 D | R 4 0 }$ for the Car category. All rows use SigLIP as the text encoder and Qwen2.5- VL-7B as the reasoning VLM. The best results are in bold.
<table><tr><td>Noise Injection Strategy</td><td>Easy</td><td>Mod.</td><td>Hard</td></tr><tr><td>BBox perturbation</td><td>33.15</td><td>22.83</td><td>19.65</td></tr><tr><td>Reasoning perturbation</td><td>34.05</td><td>23.69</td><td>20.34</td></tr><tr><td>Embedding perturbation (ours)</td><td>34.78</td><td>24.52</td><td>21.13</td></tr></table>

closely with the ground truth than MonoDETR [45] and MonoDGP [26], particularly for distant and partially occluded objects.

![](images/660856c71d1379222c93be4993eb26d3628f0560021d56c24ef37ca58ca795f0.jpg)  
(a) MonoDETR  
(b) MonoDGP  
(c) MonoLaDN (Ours)  
Fig. 5: Additional qualitative results on KITTI val set. (a) MonoDETR [45]. (b) MonoDGP [26]. (c) MonoLaDN (Ours). For each image set, the top row presents the camera-view visualization, while the bottom row ofers the corresponding bird’s-eye view. Ground-truth bounding boxes are rendered in green, and predictions are shown in order: red, yellow, blue. We also circle some objects to highlight the diference between other state-of-the-art methods and our method.

## Acknowledgement

We acknowledge Ho Chi Minh City University of Technology (HCMUT), VNU-HCM for supporting this study.