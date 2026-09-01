# OPUS: A Simple yet Effective Unified Framework for Open-Vocabulary Detection

Xiaoyan Wei<sup>1</sup>, Zhimin Yao<sup>1</sup>, Ruilin Yang<sup>1</sup>, Wei Zhang<sup>1</sup>, Yong Dai<sup>2</sup>, Yi Zhang<sup>2</sup>, Wei Ge<sup>1†</sup>

<sup>1</sup> Megvii Technology Inc. <sup>2</sup> X-Humanoid Correspondence: gewei@megvii.com

## Abstract

Recent unified open-vocabulary detection (OVD) supports heterogeneous prompts, including text queries, visual exemplars, and their combinations, but often rely on increasingly complex designs such as heavy cross-modal fu sion, staged training, and iterative annotation pipelines. We revisit whether such complexity is necessary in the era of stronger foundation models. Our finding is that unified OVD can be made substantially simpler with semantic-rich visual representations and scal able grounding supervision. We present OPUS (Open-vocabulary, Prompt-Unified, Simple), a unified detector supporting text, interactive visual, generic visual, and mixed prompting within one framework. OPUS adopts a simple three-part design. Its model architecture combines a semantic-rich visual encoder, built on a DINOv3-ConvNeXt-B backbone with efficient hybrid encoding, with a prompt-aware decoder that avoids prompt-specific branches for uni fied prompt reasoning. OPUS is trained with a one-stage text-visual training strategy with Instance-level Contrastive Alignment (ICA), and is supported by a SAM3-based singlepass data engine for heterogeneous grounding supervision. Experiments on COCO, LVIS-minival, and ODinW35 show that OPUS achieves state-of-the-art Visual-I performance, reaching 68.1/69.2/54.7 AP, while maintaining balanced Text and Visual-G accuracy. OPUS also turns mixed prompting from interference into complementarity, improving over text or visual prompt alone. These results show that simplicity and strong unified prompting capability can be achieved together.

## 1 Introduction

Open-vocabulary detection (OVD) aims to localize and recognize objects beyond a fixed label set by aligning visual regions with semantic embeddings. Driven by vision-language pretraining and promptable architectures, OVD has evolved from text-prompted detection toward unified prompting, where detectors also accept visual prompts such as reference images, boxes, or points. Compared with text prompts, visual prompts provide complementary appearance-level cues and enable finer control in long-tailed or textually ambiguous scenarios.

However, this flexibility has often been achieved through increasingly complex system designs. Architecturally, recent unified OVD methods (Chen et al., 2025; Qian et al., 2026), built upon Grounding DINO (Liu et al., 2024), rely on heavy early cross-modal fusion and additional decoder-side prompt interaction modules. In training strategy, T-Rex2 (Jiang et al., 2024) uses staged text-visual alternating training, while PET-DINO (Fu et al., 2026) further introduces more prompt-enriched training schemes. In data construction, methods such as T-Rex2 (Jiang et al., 2024) depend on iterative re-annotation pipelines that interleave model training and data refinement. These advances substantially improve unified OVD. At the same time, strong results are often specialized to particular prompt modalities or system designs, making it unclear how to achieve more balanced performance across text and visual prompting.

Modern foundation models suggest a simpler alternative. Strong visual encoders such as DINOv3 (Siméoni et al., 2025) provide semantic-rich and transferable image representations, while promptable segmentation models such as SAM3 (Carion et al., 2025) enable scalable grounding annotation without iterative relabeling. Together, these advances motivate us to revisit whether heavy cross-modal fusion, staged training, and iterative relabeling are necessary for unified OVD. Our results suggest that they are not: with strong visual representations and scalable singlepass grounding supervision, unified OVD can be made substantially simpler.

We instantiate this principle as OPUS (Openvocabulary, Prompt-Unified, Simple), a simple unified OVD framework supporting text, interactive visual, generic visual, and mixed prompting. OPUS adopts a simple three-part design: (1) a model architecture that combines a semantic-rich visual encoder, built on a DINOv3-ConvNeXt-B backbone with efficient hybrid encoding, with a promptaware decoder that avoids prompt-specific branches for unified prompt reasoning; (2) a one-stage textvisual training strategy with an Instance-level Contrastive Alignment (ICA) objective for fine-grained instance coordination; and (3) a SAM3-based single-pass data pipeline that converts text-image, classification, and segmentation sources into unified grounding supervision.

![](images/840b278a2c5ad3d6d9aab87dca52a108f35a22c47fb07a06a8288554a48b262c.jpg)  
Figure 1: Overview of OPUS. Top: OPUS simplifies unified open-vocabulary detection through a DINOv3-based semantic-rich encoder with a prompt-aware decoder, one-stage text-visual training with Instance-level Contrastive Alignment (ICA), and a SAM3-based single-pass data engine. Bottom left: OPUS achieves a strong accuracyefficiency trade-off across prompting settings. Bottom right: OPUS turns mixed prompting from interference into complementarity, improving over text or visual prompt alone.

Despite its simple design, OPUS achieves stateof-the-art Visual-I performance on COCO (Lin et al., 2014), LVIS-minival (Gupta et al., 2019), and ODinW35 (Li et al., 2022a), while maintaining competitive Text and Visual-G performance. On Visual-I of ODinW35 , OPUS outperforms PET-DINO-T (Fu et al., 2026) by +6.4 AP and DETR-ViP-T (Qian et al., 2026) by +7.9 AP.

More importantly, OPUS achieves genuine mixed-prompt complementarity: combining Text and Visual-G prompts improves performance from 49.6 to 49.9 on COCO and from 43.0 to 45.2 on LVIS-minival, whereas T-Rex2 drops from 45.8 to 42.4 on COCO. Further analysis reveals a role shift in how the two modalities contribute to mixed prompting: as grounding supervision expands, the Mixed-over-Text gain decreases from 8.7 to 2.2, while the Mixed-over-Visual-G gain increases from 1.6 to 6.6. This suggests that Visual-G becomes less effective as a standalone prompt and increasingly relies on textual semantics in the mixed setting, making stronger Visual-G modeling a promising future direction.

Our contributions are summarized as follows:

• We present OPUS, a simple unified OVD framework that supports text, visual, and mixed prompting without heavy cross-modal fusion, staged training, or iterative relabeling.

• Experiments on COCO, LVIS-minival, and ODinW35 show that OPUS achieves state-ofthe-art Visual-I performance while maintaining strong Text and Visual-G accuracy within a single unified model.

• We demonstrate genuine mixed-prompt complementarity and reveal an evolving role shift between text and visual prompts, offering new insights into unified prompt interaction.

## 2 Related Work

From Text-Prompted Detection to Visual Prompting. Open-vocabulary detection was first dominated by text-prompted paradigms, where category names or noun phrases serve as detection queries. Representative methods such as OWL-ViT (Minderer et al., 2022), GLIP (Li et al., 2022b), and Grounding DINO (Liu et al., 2024) establish this paradigm through large-scale vision-language pretraining and region-text alignment, while later approaches including OV-DINO (Wang et al., 2024), DetCLIPv3 (Yao et al., 2024), LLMDet (Fu et al., 2025b), and YOLO-World (Cheng et al., 2024) further improve open-world generalization and inference efficiency. OpenSeeD (Zhang et al., 2023) studies joint open-vocabulary segmentation and detection under text prompts. However, text prompts provide limited appearance-level guidance, especially for fine-grained, long-tailed, or textually ambiguous categories. Visual prompting methods such as DINOv (Li et al., 2024), T-Rex (Jiang et al., 2023), MQDet (Xu et al., 2023), CountGD (Amini Naieni et al., 2024), and Prompt-DINO (Guan et al., 2025) address this limitation by using exemplars, boxes, points, or multimodal queries to provide instance-level appearance cues. These works show that visual prompts are complementary to text, but they often focus on specific prompting settings rather than a fully unified text-visual prompting framework.

Unified Prompting and System Complexity. Recent work has therefore moved toward unified detectors that support both text and visual prompts. T-Rex2 (Jiang et al., 2024) introduces a unified text-visual prompting paradigm through staged text-to-visual optimization and alternating training. YOLOE (Wang et al., 2025) extends the YOLOstyle detection framework to support unified text and visual prompting through a lightweight design, providing a real-time and cost-efficient solution. PET-DINO (Fu et al., 2026) further adopts prompt-enriched multi-stage optimization built upon Grounding DINO (Liu et al., 2024). In parallel, methods such as CP-DETR (Chen et al., 2025) and DETR-ViP (Qian et al., 2026) strengthen prompt interaction through additional encoder or decoder-side cross-modal fusion modules. Beyond model designs and training strategies, recent works also scale grounding supervision in different ways. Some expand image-text pairs via detector-generated pseudo boxes, as in GLIP (Li et al., 2022b), OWLv2 (Minderer et al., 2023), and YOLO-World (Cheng et al., 2024). Others build generative annotation engines with vision-language models, such as DetCLIPv3 (Yao et al., 2024) and WeDetect (Fu et al., 2025a). Still others assemble large multi-source corpora, including GLEE (Wu et al., 2024) and DINO-X (Ren et al., 2024), while T-Rex2 (Jiang et al., 2024) designs separate text and visual data engines with iterative relabeling. These approaches broaden open-vocabulary coverage, yet collecting and cleaning such data remain costly and often hard to reproduce.

Foundation Models for Simpler Unified OVD. Modern foundation models offer a different route. Strong self-supervised visual encoders such as DI-NOv3 (Siméoni et al., 2025) provide transferable and semantically rich image representations, reducing the need for heavy early cross-modal fusion. Meanwhile, promptable segmentation models such as SAM3 (Carion et al., 2025) make it possible to generate large-scale grounding supervision from heterogeneous sources without iterative relabeling. Rather than further increasing architectural, optimization, or data-engine complexity, our work revisits unified OVD under these stronger foundationmodel priors. We show that a simple framework can support text, interactive visual, generic visual, and mixed prompting while maintaining strong performance across prompting settings.

## 3 Method

OPUS is a unified open-vocabulary detector designed around three simple components: a semantic-rich image encoder with a prompt-aware decoder, a one-stage text-visual training strategy, and a single-pass grounding data engine. Given an image and a set of heterogeneous prompts, OPUS maps text and visual prompts into a shared prompt space and decodes object queries conditioned on both image features and prompt features. Figure 1 provides an overview of our framework.

## 3.1 Model Architecture

Semantic-Rich Image Encoder. We use a DI-NOv3 (Siméoni et al., 2025) pretrained ConvNeXt (Liu et al., 2022) backbone to extract multiscale image features $\{ C _ { 3 } , C _ { 4 } , C _ { 5 } \}$ . Instead of performing heavy early cross-modal fusion, OPUS first processes these visual features with an efficient hybrid encoder (Zhao et al., 2024), which consists of attention-based intra-scale feature interaction and CNN-based cross-scale feature fusion:

$$
\mathbf { F } = \mathrm { H y b r i d E n c } ( C _ { 3 } , C _ { 4 } , C _ { 5 } ) \in \mathbb { R } ^ { N _ { f } \times D } ,\tag{1}
$$

where $\begin{array} { r } { N _ { f } \ = \ \sum _ { l \in \{ 3 , 4 , 5 \} } H _ { l } W _ { l } } \end{array}$ denotes the number of visual tokens across all feature scales, and $D$ is the hidden dimension. The strong semantic transferability of DINOv3 allows OPUS to rely on this lightweight visual encoding pipeline without encoder-side image-text fusion.

Unified Prompt Generator. Following T-Rex2 (Jiang et al., 2024), OPUS supports text and visual prompts under a unified prompt representation. For text prompts, category names or noun phrases are encoded by a CLIP (Radford et al., 2021) text encoder and projected into the detector hidden space, producing text prompt features $\mathbf { P } ^ { t } \in \mathbb { R } ^ { M \times D }$ , where M is the number of category prompts in the current input. For visual prompts, a Visual Prompt Encoder (VPE) encodes user-provided boxes or points. Given $K _ { m }$ visual exemplars for category $m \in \{ 1 , \ldots , M \}$ , VPE stacks three Transformer layers with multi-scale deformable cross-attention to extract features from multi-scale image feature maps conditioned on these exemplars, yielding instance-level content prompt features $\mathbf { P } _ { m , \mathrm { { c o n } } } ^ { v }$ and an aggregated class-level prompt feature $\mathbf { P } _ { m , \mathrm { c l s } } ^ { v }$ via a learnable class token:

$$
\mathbf { P } _ { m } ^ { v } = [ \mathbf { P } _ { m , \mathrm { c o n } } ^ { v } ; \mathbf { P } _ { m , \mathrm { c l s } } ^ { v } ] \in \mathbb { R } ^ { ( K _ { m } + 1 ) \times D } .\tag{2}
$$

Class-level prompt features of all M categories are then stacked into $\mathbf { P } _ { \mathrm { c l s } } ^ { v } \colon$

$$
\mathbf { P } _ { \mathrm { c l s } } ^ { v } = [ \mathbf { P } _ { 1 , \mathrm { c l s } } ^ { v } ; \mathbf { P } _ { 2 , \mathrm { c l s } } ^ { v } ; \cdot \cdot \cdot ; \mathbf { P } _ { M , \mathrm { c l s } } ^ { v } ] \in \mathbb { R } ^ { M \times D } .\tag{3}
$$

Depending on the prompting setting, the final prompt representation $\mathbf { P } \in \bar { \mathbb { R } ^ { M \times D } }$ can be constructed from textual prompts, visual prompts, or their combination:

$$
\mathbf { P } = \left\{ \begin{array} { l l } { \mathbf { P } ^ { t } , } & { \mathrm { T e x t } , } \\ { \mathbf { P } _ { \mathrm { c l s } } ^ { v } , } & { \mathrm { V i s u a l } , } \\ { ( \mathbf { P } ^ { t } + \mathbf { P } _ { \mathrm { c l s } } ^ { v } ) / 2 , } & { \mathrm { M i x e d } . } \end{array} \right.\tag{4}
$$

Prompt-Aware Decoder. To support unified reasoning over heterogeneous prompts, we adopt a prompt-aware decoder based on the Grounding

DINO’s (Liu et al., 2024) decoder. Given image features $\mathbf { F } \in \mathbb { R } ^ { N _ { f } \times D }$ and prompt features $\textbf { P } \in \ \mathbb { R } ^ { M \times D }$ , we initialize $N _ { q }$ object queries $\mathbf { Q } \in \mathbb { R } ^ { N _ { q } \times D }$ through prompt-guided proposal selection. Each decoder layer l iteratively refines the queries via query self-attention, prompt crossattention over P, and image cross-attention over F:

$$
\mathbf { Q } ^ { ( l ) } = \mathrm { D e c L a y e r } \left( \mathbf { Q } ^ { ( l - 1 ) } , \mathbf { F } , \mathbf { P } \right) .\tag{5}
$$

The resulting decoder naturally supports text, interactive visual, generic visual, and mixed prompting within a unified decoding process, without introducing prompt-specific architectural branches.

## 3.2 Training Strategy and Objective

One-Stage Text-Visual Training. OPUS is trained with a one-stage alternating strategy. For text-prompt batches, we use object detection and visual grounding data, where category names or noun phrases serve as text prompts. For visualprompt batches, we sample ground-truth boxes or points as visual prompts and train the model to detect objects conditioned on visual exemplars. Text and visual batches are alternated with a fixed ratio during training.

Detection and Alignment Losses. The detection objective follows DETR-style detection training and includes classification, box regression, IoU, and denoising losses:

$$
\mathcal { L } _ { \mathrm { d e t } } = \mathcal { L } _ { \mathrm { c l s } } + \mathcal { L } _ { \mathrm { b b o x } } + \mathcal { L } _ { \mathrm { i o u } } + \mathcal { L } _ { \mathrm { d n } } .\tag{6}
$$

For visual-prompt batches, we additionally use the global prompt alignment loss from T-Rex2 (Jiang et al., 2024), which aligns each class-level visual prompt $\mathbf { P } _ { \mathrm { c l s } } ^ { v }$ with its matched text prompt $\mathbf { P } ^ { t }$ . The base objective is therefore

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { b a s e } } = \mathcal { L } _ { \mathrm { d e t } } + \mathcal { L } _ { \mathrm { a l i g n } } . } \end{array}\tag{7}
$$

While this global alignment encourages text-visual category consistency, it does not explicitly supervise instance-level visual cues, which are important for visual and mixed prompting.

Instance-level Contrastive Alignment (ICA). To strengthen instance-level coordination between visual prompts and object queries, we introduce ICA objective. Let $\{ ( Q _ { i } , \bar { P ^ { v } } _ { \mathrm { c o n } , i } ) \} _ { i = 1 } ^ { N ^ { + } }$ denote the query-content prompt pairs associated through Hungarian matching between predicted queries and ground-truth instances. ICA treats each matched pair $\left( Q _ { i } , P _ { \mathrm { c o n } , i } ^ { v } \right)$ as positive and the remaining content prompts in the same mini-batch as negatives. The ICA loss is then defined as:

![](images/868a6768d682aad60ccc2879ae5f63a7809ac4ffc4e5576043b797f61c340f94.jpg)  
Figure 2: Composition of the OPUS training corpus. OPUS uses 7.98M training images from heterogeneous detection, grounding, classification, image-text, and segmentation sources. Datasets marked with <sup>†</sup> are generated by our SAM3-based single-pass annotation pipeline: Bamboo-CLS, CC3M, and SA-1B. Together, these generated annotations contribute 4.38M images, accounting for 54.9% of the corpus, showing that OPUS scales grounding supervision without iterative relabeling.

$$
\mathcal { L } _ { \mathrm { I C A } } = - \frac { 1 } { N ^ { + } } \sum _ { i = 1 } ^ { N ^ { + } } \log \frac { \exp ( \sin ( Q _ { i } , P _ { \mathrm { c o n } , i } ^ { v } ) / \tau ) } { \displaystyle \sum _ { j = 1 } ^ { N ^ { + } } \exp ( \sin ( Q _ { i } , P _ { \mathrm { c o n } , j } ^ { v } ) / \tau ) }\tag{8}
$$

where sim $. ( \cdot , \cdot )$ denotes cosine similarity and τ is a fixed temperature hyperparameter. The final objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { b a s e } } + \mathcal { L } _ { \mathrm { I C A } } , } \end{array}\tag{9}
$$

where ${ \mathcal { L } } _ { \mathrm { I C A } }$ is also applied only to visual-prompt batches and set to zero otherwise. ICA encourages visual prompt features to preserve fine-grained instance information and improves their coordination with object queries.

## 3.3 Data Engine

To construct heterogeneous grounding supervision, we build a SAM3-based single-pass annotation pipeline. The training corpus combines existing detection and grounding datasets with single-pass annotated data from text-image, classification, and segmentation sources, as summarized in Figure 2.

For text-image data, we extract noun phrases from captions and use them as text prompts for SAM3 (Carion et al., 2025) to generate corresponding object regions. For classification data, we use image-level category labels as prompts and convert the resulting SAM3 regions into box annotations.

For segmentation data, we treat existing masks or regions as geometric prompts to recover additional object instances and convert them into bounding boxes, and apply TAP (Pan et al., 2024) to assign unified semantic labels to the resulting regions.

This pipeline produces Bamboo-CLS (Zhang et al., 2025), CC3M (Sharma et al., 2018), and annotated SA-1B (Kirillov et al., 2023) supervision for text and visual prompting. To control annotation noise, we apply a unified postprocessing pipeline, including confidence thresholding, category-wise and image-wise NMS, removal of near full-image boxes, and filtering of oversized crowd boxes. Process details and representative success and failure cases are provided in Appendix A.4.

## 4 Experiments

## 4.1 Experimental Setup

Model and Training Details. Unless otherwise specified, OPUS refers our final setting with a DI-NOv3 pretrained ConvNeXt-B backbone, a CLIP-B text encoder, and the Visual Prompt Encoder (VPE) following T-Rex2 (Jiang et al., 2024). The hidden dimension of all FFN layers is 1024. We optimize with AdamW (Loshchilov and Hutter, 2017), using a learning rate of $4 \times 1 0 ^ { - 5 }$ for the image backbone and text encoder, and $4 \times 1 0 ^ { - 4 }$ for the remaining modules. Models are trained on 16 A100 GPUs with a global batch size of 128. Vsual-prompt and text-prompt batches are alternated at a ratio of 1:8. For the O365 (Shao et al., 2019)+GoldG (Kamath et al., 2021) ablation, we train on 8 A100 GPUs with a batch size of 64 for 120k iterations. Full hyperparameters are provided in Appendix B.1.

Evaluation Protocol. Following T-Rex2, we evaluate zero-shot detection on COCO (Lin et al., 2014), LVIS-minival (Gupta et al., 2019), and ODinW35 (Li et al., 2022a), reporting standard AP metrics as in prior work. AP numbers in comparison tables are taken from the original papers. We consider four prompting settings. Text uses category names directly as text prompts. Visual-G constructs category-level visual prototypes by randomly sampling $N { = } 1 6$ reference images per category from the training set. Visual-I simulates interactive detection by randomly sampling one ground-truth instance per category from each test image as the visual prompt. Mixed averages Text and Visual-G embeddings. For efficiency, we report $\mathrm { F P S _ { g e n } }$ for generic detection and $\mathrm { F P S } _ { \mathrm { i n t e r } }$ for interactive detection on COCO, using our reproductions of publicly available codebases or model architectures from prior work. Measurement details are provided in Appendix B.3.

<table><tr><td colspan="4"></td><td colspan="3">Visual-I</td><td colspan="3">Visual-G</td><td colspan="4">Text</td></tr><tr><td>Model</td><td>Backbone</td><td>Input Size</td><td>COCO</td><td>LVIS-mv</td><td>ODinW</td><td> $\mathbf { F P S _ { i n t e r } }$ </td><td>COCO</td><td>LVIS-mv</td><td>ODinW</td><td>COCO</td><td>LVIS-mv</td><td>ODinW</td><td>FPSgen</td></tr><tr><td>T-Rex2-T</td><td>Swin-T</td><td>800×1333</td><td>56.6</td><td>59.3</td><td>37.7</td><td>48.8</td><td>38.8</td><td>37.4</td><td>23.6</td><td>45.8</td><td>42.8</td><td>18.0</td><td>15.4</td></tr><tr><td>CP-DETR-T</td><td>Swin-T</td><td>800×1333</td><td>61.8</td><td>64.1</td><td>41.0</td><td>一</td><td>一</td><td>–</td><td>–</td><td>52.0</td><td>47.6</td><td>27.3</td><td>一</td></tr><tr><td>DETR-ViP-T</td><td>Swin-T</td><td>800×1333</td><td>65.4</td><td>66.1</td><td>46.8</td><td>41.5</td><td>43.2</td><td>41.1</td><td>31.2</td><td>一</td><td>†25.0</td><td>–</td><td>10.3</td></tr><tr><td>PET-DINO-T</td><td>Swin-T</td><td>800×1333</td><td>64.3</td><td>64.5</td><td>48.3</td><td>38.8</td><td>38.4</td><td>31.5</td><td>25.5</td><td>49.8</td><td>37.8</td><td>20.6</td><td>8.2</td></tr><tr><td>YOLOE-v8-L</td><td>YOLOv8-L</td><td>640×640</td><td>一</td><td>-</td><td>–</td><td>-</td><td>一</td><td>34.2</td><td>一</td><td>一</td><td>35.9</td><td>-</td><td>一</td></tr><tr><td>OPUS (Swin-T)</td><td>Swin-T</td><td>800×1333</td><td>68.1</td><td>70.6</td><td>53.7</td><td>42.1</td><td>35.8</td><td>35.8</td><td>24.8</td><td>47.1</td><td>44.6</td><td>22.0</td><td>14.5</td></tr><tr><td>OPUS (Ours)</td><td>ConvNeXt-B</td><td>640×640</td><td>68.1</td><td>69.2</td><td>54.7</td><td>43.5</td><td>43.4</td><td>38.6</td><td>25.9</td><td>49.6</td><td>43.0</td><td>22.1</td><td>21.3</td></tr></table>

Table 1: Main zero-shot detection results under Visual-I, Visual-G, and Text prompting. AP is reported on COCO, LVIS-minival (LVIS-mv), and ODinW35, abbreviated as ODinW in the table. $\mathrm { F P S } _ { \mathrm { i n t e r } }$ measures Visual-I inference with cached image features, and $\mathrm { F P S _ { g e n } }$ measures Text/Visual-G inference with cached prompt features. OPUS achieves the best Visual-I performance while maintaining balanced accuracy across prompting modes. Bold: best; underline: second best. <sup>†</sup>: result from supplementary materials.

## 4.2 Main Results

Accuracy and Efficiency. Table 1 compares OPUS with T-Rex2-T, CP-DETR-T, DETR-ViP-T, PET-DINO-T, and YOLOE-v8-L under Visual-I, Visual-G, and Text prompting. OPUS (Ours) achieves the best Visual-I performance among these baselines on COCO, LVIS-minival, and ODinW35, exceeding DETR-ViP-T by 2.7 on COCO and 3.1 AP on LVIS-minival and surpassing PET-DINO-T and DETR-ViP-T on ODinW35 by 6.4 and 7.9 AP, respectively. Prior methods often specialize in one prompt mode (e.g., DETR-ViP-T has strong Visual-G but weak Text), whereas OPUS remains more balanced across prompting modes.

For a controlled comparison, we also report OPUS (Swin-T) under the same Swin-T (Liu et al., 2021) backbone and 800×1333 input as T-Rex2- T. It improves Visual-I LVIS-minival by 11.3 AP over T-Rex-T, with only a minor Visual-G trade-off. OPUS (Ours) is our final practical setting with a DINOv3 pretrained ConvNeXt-B backbone, hybrid encoding and a reduced 640×640 input. It further recovers Visual-G relative to OPUS (Swin-T) and yields a more balanced profile across generic and interactive detection. Despite the lower resolution, it remains competitive in accuracy while running faster than the Swin-T baselines in Table 1, achieving the highest $\mathrm { F P S _ { g e n } }$ of 21.3.

As a cross-scale reference, Table 2 further compares Visual-I against larger Swin-L models. Although DETR-ViP-L remains stronger on COCO and LVIS-minival, OPUS (Ours) stays within about 3 AP of it on these two benchmarks and leads

<table><tr><td>Model</td><td>Backbone</td><td>COCO</td><td>LVIS-mv</td><td>ODinW</td></tr><tr><td>T-Rex2-L</td><td>Swin-L</td><td>58.5</td><td>62.5</td><td>39.7</td></tr><tr><td>CP-DETR-L</td><td>Swin-L</td><td>68.4</td><td>71.6</td><td>50.6</td></tr><tr><td>DETR-ViP-L</td><td>Swin-L</td><td>71.1</td><td>71.9</td><td>51.2</td></tr><tr><td>PET-DINO-L</td><td>Swin-L</td><td>66.5</td><td>65.8</td><td>49.7</td></tr><tr><td>OPUS(Ours)</td><td>ConvNeXt-B</td><td>68.1</td><td>69.2</td><td>54.7</td></tr></table>

Table 2: Visual-I comparison against Swin-L baselines. Gray numbers indicate settings where the model used the corresponding benchmark’s training split during training or pre-training; these numbers are reported for reference but excluded from ranking. Bold and underline mark the best and second best among ranked results.

<table><tr><td>Model</td><td>COCO Text/Visual-G/Mixed Text/Visual-G/Mixed Text/Visual-G/Mixed</td><td>LVIS-mv</td><td>LVIS</td></tr><tr><td>T-Rex2-T</td><td>45.8/38.8/42.4</td><td>42.8/37.4/-</td><td>34.8/34.9/37.0</td></tr><tr><td>OPUS(Ours)</td><td>49.6/43.4/49.9</td><td>43.0/38.6/45.2</td><td>34.8/31.2/37.6</td></tr></table>

Table 3: Mixed prompting comparison on COCO, LVISminival (LVIS-mv), and LVIS. Bold indicates the best or tied best. − indicates that the result is not reported.

ODinW35 by 3.5 AP, despite using a smaller ConvNeXt-B backbone and lower input resolution.

Mixed Prompting Complementarity. Table 3 evaluates Mixed prompting that combines Text and Visual-G at inference. On COCO, T-Rex2-T drops from 45.8 (Text) to 42.4 (Mixed), showing interference. In contrast, OPUS improves from 49.6 to 49.9 on COCO and from 43.0 to 45.2 on LVISminival. On full LVIS, T-Rex2-T achieves stronger Visual-G than OPUS, yet its Mixed score remains lower. These results suggest that OPUS does not simply rely on stronger individual prompt modalities, but more effectively combines text semantics and visual appearance cues. Section 4.5 further analyzes how this complementarity evolves as grounding supervision expands.

<table><tr><td>ID</td><td>Evolution Step</td><td>Visual-I</td><td>Visual-G</td><td>Text</td><td>Mixed</td><td> $\pmb { \Delta } _ { \mathbf { T } }$ </td><td> $\Delta _ { \mathbf { G } }$ </td><td> $\mathbf { \overline { { F P S _ { g e n } } } }$ </td></tr><tr><td>A</td><td>Baseline (Swin-T)</td><td>56.3</td><td>33.6</td><td>26.5</td><td>35.2</td><td>+8.7</td><td>+1.6</td><td>15.4</td></tr><tr><td>B</td><td>+ Semantic-rich enc.</td><td>57.5</td><td>35.3</td><td>29.5</td><td>36.2</td><td>+6.7</td><td>+0.9</td><td>22.0</td></tr><tr><td>C</td><td>+ Prompt-aware dec.</td><td>63.0</td><td>33.3</td><td>32.7</td><td>36.5</td><td>+3.8</td><td>+3.2</td><td>21.3</td></tr><tr><td>D</td><td>+ICA</td><td>63.2</td><td>33.5</td><td>33.2</td><td>38.0</td><td>+4.8</td><td>+4.5</td><td>21.3</td></tr><tr><td>E</td><td>+ Data engine</td><td>69.2</td><td>38.6</td><td>43.0</td><td>45.2</td><td>+2.2</td><td>+6.6</td><td>21.3</td></tr></table>

Table 4: Component-wise ablation and prompt-complementarity analysis on LVIS-minival. Mixed denotes Text+Visual-G prompting. $\Delta _ { T }$ and $\Delta _ { G }$ measure the gains of Mixed prompting over Text-only and Visual-Gonly prompting, respectively.

## 4.3 OPUS Design Analysis

Component-wise Performance. We study how each design step contributes to the final OPUS configuration. Since T-Rex2 has not released its code or model weights, we re-implement its architecture based on the published paper and train it on O365 and GoldG as our baseline (step A). Table 4 reports the progression on LVIS-minival as we add each OPUS component.

Semantic-rich enc. Replacing Swin-T with a DINOv3-based ConvNeXt-B backbone, reducing the input from 800×1333 to 640×640, and adopting a hybrid encoder improves all prompting modes and raises $\mathrm { F P S _ { g e n } }$ from 15.4 to 22.0 (+43%).

Prompt-aware dec. Adding the prompt-aware decoder improves Visual-I from 57.5 to 63.0 and Text from 29.5 to 32.7, but reduces Visual-G from 35.3 to 33.3. Visual-I already knows which categories are present in the image, so prompt and image cross-attention helps find similar instances. Text also remains effective under supervised training with both positive and negative category names. In contrast, Visual-G uses cross-image category prototypes at inference, which can mismatch the Visual-I-style prompts seen during training. This mismatch is most severe for rare categories, whose prototypes are aggregated from fewer and more diverse instances.

ICA. Table 5 isolates ICA under O365+GoldG after Step C. Visual prompting is effective on the long tail. Even without ICA, Visual-G already outperforms Text on rare categories (AP<sub>r</sub> 26.1 vs. 20.5). Adding ICA raises Visual-G $\mathsf { A P } _ { r }$ to 30.0 (Text AP<sub>r</sub> remains 20.3), while overall Visual-I, Visual-G, and Text AP change by less than 0.5. Mixed AP also increases by 1.5 (36.5→38.0). Section 4.5 further analyzes why ICA helps rarecategory Visual-G beyond class-level alignment $\mathcal { L } _ { \mathrm { a l i g n } }$

Data engine. ICA and the data engine address complementary long-tailed bottlenecks. ICA strengthens long-tailed visual prompting, while the data engine expands grounding supervision and mainly improves text semantic coverage, lifting Text from 33.2 to 43.0 and Mixed from 38.0 to 45.2 on LVIS-minival, with Text $\mathsf { A P } _ { r }$ rising from 20.3 to 36.6.

<table><tr><td rowspan="2">ICA</td><td colspan="2">Visual-I</td><td colspan="2">Visual-G</td><td colspan="2">Text</td><td colspan="2">Mixed</td></tr><tr><td>AP</td><td>APr</td><td>AP</td><td>APr</td><td>AP</td><td>APr</td><td>AP</td><td>APr</td></tr><tr><td>w/o</td><td>63.0</td><td>73.9</td><td>33.3</td><td>26.1</td><td>32.7</td><td>20.5</td><td>36.5</td><td>30.8</td></tr><tr><td>w/</td><td>63.2</td><td>74.7</td><td>33.5</td><td>30.0</td><td>33.2</td><td>20.3</td><td>38.0</td><td>31.1</td></tr></table>

Table 5: ICA under the O365+GoldG setting on LVISminival (steps C→D, ConvNeXt-B @ 640×640).

<table><tr><td>ID</td><td>Evolution Step</td><td>Params</td><td>Input</td><td>GFLOPs</td><td>FPSgen</td></tr><tr><td>A</td><td>Baseline (Swin-T)</td><td>115M</td><td> $\overline { { 8 0 0 \times 1 3 3 3 } }$ </td><td>257</td><td>15.4</td></tr><tr><td>B1</td><td>+ DINOv3-ConvNeXt-B</td><td>173M</td><td> $8 0 0 \times 1 3 3 3$ </td><td>439</td><td>11.6</td></tr><tr><td>B2</td><td>+ Reduce to 640×640</td><td>173M</td><td> $6 4 0 \times 6 4 0$ </td><td>208</td><td>19.4</td></tr><tr><td>B3</td><td>+ Hybrid encoder</td><td>172M</td><td> $6 4 0 \times 6 4 0$ </td><td>164</td><td>22.0</td></tr><tr><td>C</td><td>+ Prompt-aware dec.</td><td>174M</td><td> $6 4 0 \times 6 4 0$ </td><td>165</td><td>21.3</td></tr><tr><td>D</td><td>+ ICA</td><td>174M</td><td> $6 4 0 \times 6 4 0$ </td><td>165</td><td>21.3</td></tr><tr><td>E</td><td>+ Data engine</td><td>174M</td><td> $6 4 0 \times 6 4 0$ </td><td>165</td><td>21.3</td></tr></table>

Table 6: Component-wise Params, GFLOPs, and $\mathrm { F P S } _ { \mathrm { g e n } } . \mathrm { \bf ~ B } _ { 1 }$ to ${ \bf B } _ { 3 }$ factorize step B in Table 4; C to E match the same table. Accuracy under each prompting mode is reported in Table 4.

Component-wise Efficiency. Table 6 decomposes step B into ${ \bf B } _ { 1 }$ to $\mathrm { B _ { 3 } }$ (C to E follow Table 4). Switching to DINOv3-ConvNeXt-B at 800×1333 $( \mathbf { B } _ { 1 } )$ increases Params/GFLOPs and slows inference (FPS 15.4→11.6). Reducing the input to 640×640 (B<sub>2</sub>) cuts GFLOPs from 439 to 208 and recovers FPS to 19.4. The hybrid encoder $\left( \mathbf { B } _ { 3 } \right)$ further reduces GFLOPs to 164 and reaches FPS 22.0. Later steps add little cost. The prompt-aware decoder adds 2M parameters and 1 GFLOPs, while ICA and the data engine leave inference unchanged. Appendix C.2 repoprts the same IDs with per-step accuracy under all prompting modes.

## 4.4 Grounding Supervision and Data Scaling

Data Engine Annotation Quality. The SAM3- based data engine contributes 4.38M of the 7.98M training images (54.9%), so annotation quality matters for the scaling results below. We randomly sample 200 images from each of Bamboo-CLS, CC3M, and SA-1B and manually check every pseudo box–phrase pair after filtering. A box is marked incorrect if either its localization or label is wrong. Box accuracies are 91.9%, 89.5% and 91.2% on the three sources (Table 9). Residual errors are mainly nested or redundant background boxes on Bamboo-CLS and CC3M, and incorrect box–label assignments on SA-1B. Appendix A.4 details the filtering pipeline and shows qualitative success and failure cases.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Training data</td><td colspan="3">COCO</td><td colspan="3">LVIS-mv</td></tr><tr><td>Visual-I</td><td>Visual-G</td><td>Text</td><td>Visual-I</td><td>Visual-G</td><td>Text</td></tr><tr><td>T-Rex2-T</td><td>official</td><td>56.6</td><td>38.8</td><td>45.8</td><td>59.3</td><td>37.4</td><td>42.8</td></tr><tr><td rowspan="3">T-Rex2-T (re-impl.)</td><td>Base</td><td>57.2</td><td>41.3</td><td>45.8</td><td>56.3</td><td>33.6</td><td>26.5</td></tr><tr><td>Mid</td><td>56.6</td><td>42.1</td><td>47.3</td><td>57.2</td><td>36.6</td><td>39.0</td></tr><tr><td>Full</td><td>56.0</td><td>41.4</td><td>47.1</td><td>59.4</td><td>40.5</td><td>45.2</td></tr><tr><td rowspan="2">OPUS (Swin-T)</td><td>Base</td><td>61.2</td><td>37.6</td><td>45.2</td><td>64.1</td><td>34.4</td><td>33.9</td></tr><tr><td>Full</td><td>68.1</td><td>35.8</td><td>47.1</td><td>70.6</td><td>35.8</td><td>44.6</td></tr><tr><td rowspan="2">OPUS (Ours)</td><td>Base</td><td>64.6</td><td>43.0</td><td>48.2</td><td>63.2</td><td>33.5</td><td>33.2</td></tr><tr><td>Full</td><td>68.1</td><td>43.4</td><td>49.6</td><td>69.2</td><td>38.6</td><td>43.0</td></tr></table>

Table 7: Data scaling on COCO and LVIS-minival. Base: O365<sup>v</sup> (Shao et al., 2019)+GoldG (Kamath et al., 2021). Mid: Base+OI<sup>v</sup> (Kuznetsova et al., 2020)+V3Det (Wang et al., 2023)+HT<sup>v</sup> (Long et al., 2022)+CH<sup>v</sup> (Shao et al., 2018)+Bamboo<sup>v†</sup> (Zhang et al., 2025). Full: Mid+CC3M<sup>†</sup> (Sharma et al., 2018)+SA-1B<sup>v†</sup> (Kirillov et al., 2023) (7.98M images). <sup>v</sup>: visual-prompt training (all sources used for text); <sup>†</sup>: SAM3-based data-engine annotations. Top: progressive scaling on T-Rex2-T (re-impl.); official T-Rex2-T is shown for reference. Bottom: cross-framework scaling under the same Base and Full sets.

Progressive Data Scaling. Table 7 first reports progressive data scaling on T-Rex2-T (re-impl.), with official T-Rex2-T as a reference. Each row lists all datasets used in that run. The impact is much stronger on LVIS-minival than on COCO. On COCO, Visual-I, Visual-G, and Text change only sightly from O365+GoldG to the largest set, moving from 57.2, 41.3, and 45.8 to 56.0, 41.4, and 47.1. On LVIS-minival, Visual-G rises from 33.6 to 40.5 and Text from 26.5 to 45.2, indicating that large-scale grounding mainly improves longtailed coverage. Visual-I is already strong with O365 visual training and further improves on LVISminival after adding SA-1B. Visual-G stays stable on COCO and improves on LVIS-minival, unlike official T-Rex2, whose Visual-G drops after adding visual data (Appendix C.3).

Data Scaling Across Frameworks. Table 7 compares T-Rex2-T (re-impl.), OPUS (Swin-T) and OPUS (Ours) under the same baseline of O365+GoldG and the same largest training set. On LVIS-minival Visual-I, scaling to the full set improves T-Rex2-T (re-impl.) by 3.1 AP, while OPUS improves by 6.5 AP with Swin-T and by 6.0 AP with ConvNeXt-B. On the full set, OPUS (Swin-T) and OPUS (Ours) reach 70.6 and 69.2 Visual-I AP on LVIS-minival and 68.1 on COCO, compared with 59.4 and 56.0 for T-Rex2-T (re-impl.). Visual-G shows a different scaling pattern within OPUS. OPUS (Swin-T) gains only 1.4 AP on LVISminival and slightly drops on COCO, whereas OPUS (Ours) gains 5.1 AP on LVIS-minival. On the full set, OPUS (Ours) leads COCO Visual-G at 43.4, while T-Rex2-T (re-impl.) remains higher on LVIS-minival at 40.5. For Text, OPUS (Ours) is strongest on COCO at 49.6, while T-Rex2-T (reimpl.) remains slightly higher on LVIS-minival at 45.2. Overall, OPUS converts expanded grounding data more effectively into interactive visual prompting gains, and data scaling works best together with the full OPUS design for more balanced improvements across prompting modes.

## 4.5 Further Analysis

Fine-grained Category Analysis Across Component Evolution. Figure 3 breaks down the A→E component evolution into frequent, common, and rare categories on LVIS-minival. Overall AP and category-group AP mostly improve throughout the progression, confirming that the gains are not limited to a specific frequency group.

The main exception appears when adding the prompt-aware decoder. While this decoder improves Text and Visual-I, it temporarily hurts Visual-G, especially on rare categories. As discussed in Table 4, this likely stems from a training-inference mismatch: the decoder is trained with Visual-I-style prompts from the same image, whereas Visual-G inference uses cross-image category prototypes. Rare categories are more sensitive to this mismatch because their prototypes are aggregated from fewer and more diverse instances.

![](images/9e03d8c0bf3fc355006a6d5d257e587420bec187adddfd112ac9bf2bd8d2a377.jpg)  
Figure 3: Fine-grained category analysis across component evolution on LVIS-minival. We report overall AP and AP on frequent, common, and rare categories under Text, Visual-G, Visual-I, and Mixed prompting.

ICA partially recovers this drop, with the most notable improvement on rare categories. This suggests that the issue is not only the distribution mismatch of Visual-G prototypes, but also the limited granularity of class-level prompt alignment. The base alignment objective $\mathcal { L } _ { \mathrm { a l i g n } }$ pulls the aggregated visual prototype $\mathbf { P } _ { \mathrm { c l s } } ^ { v }$ toward the text prompt feature $\mathbf { P } ^ { t }$ . While useful for category-level alignment, this objective may suppress fine-grained visual details, especially for rare categories whose text representations and visual prototypes are both less reliable. By contrast, ICA aligns instance-level visual prompt features with their matched decoder queries. This provides a finer-grained learning signal that is less dependent on class-level text anchors, helping recover rare-category Visual-G performance.

Evolving Complementarity Between Text and Visual Prompting. Table 4 also reveals how textvisual complementarity evolves across the A→E progression. Mixed prompting consistently outperforms both Text and Visual-G inference, showing that the two prompt types provide complementary cues. However, the source of this complementarity changes over time: $\Delta _ { T }$ decreases from 8.7 to 2.2, while $\Delta _ { G }$ increases from 1.6 to 6.6.

This trend reflects a progressive role shift between the two prompting modalities. At early stages, Text prompting suffers from limited longtail semantic coverage, so Visual-G provides useful appearance-level guidance. As the data engine expands grounding supervision, Text prompting becomes much stronger and eventually surpasses Visual-G. Mixed prompting then shifts from using visual prompts to compensate for weak text semantics toward using text prompts to stabilize noisier cross-image visual prototypes. This evolving interaction explains why mixed prompting remains consistently beneficial, and also exposes a limitation of current generic visual prompting, suggesting that the Visual-G paradigm itself deserves further study, including reference selection, prototype aggregation, and prototype denoising.

## 5 Conclusion

We presented OPUS, a simple unified framework for open-vocabulary detection. OPUS simplifies unified OVD along three axes: a semanticrich DINOv3 backbone with efficient hybrid encoding and a prompt-aware decoder that avoids prompt-specific branches for unified prompt reasoning, a one-stage text-visual training strategy with Instance-level Contrastive Alignment, and a SAM3- based single-pass data engine. Across COCO, LVIS-minival, and ODinW35, OPUS achieves state-of-the-art Visual-I performance while maintaining strong Text and Visual-G accuracy, and further turns mixed prompting from interference into genuine complementarity.

Our analysis reveals that this complementarity evolves with grounding supervision. As Text prompting recovers long-tailed semantic coverage, Visual-G increasingly relies on textual semantics in the mixed setting, exposing the limitation of noisy cross-image category prototypes. This suggests that improving generic visual prompting through better reference selection, prototype aggregation, and prototype denoising, is an important direction for future unified OVD systems. Overall, OPUS demonstrates that simplicity and strong unified prompting capability can be achieved together in open-vocabulary detection.

## Limitations

Although OPUS achieves strong overall performance, generic visual prompting remains less improved than text and interactive visual prompting. Our training uses Visual-I-style prompts sampled from ground-truth instances, whereas Visual-G inference relies on cross-image category prototypes aggregated from reference images. This traininginference mismatch makes Visual-G more sensitive to reference quality and prototype noise, especially for rare categories. Our analysis suggests that current generic visual prompting still requires more targeted modeling, such as better reference image selection, prototype aggregation, and prototype denoising.

In addition, our SAM3-based data engine is designed to scale heterogeneous grounding supervision in a single pass, but its effectiveness depends on the quality and domain coverage of the underlying promptable model. We validate OPUS on standard natural-image detection benchmarks, including COCO, LVIS-minival, and ODinW35, but have not evaluated the annotation pipeline on specialized domains such as medical, industrial, or remote-sensing imagery. Extending and validating the pipeline in such domains remains future work.

## References

Niki Amini-Naieni, Tengda Han, and Andrew Zisserman. 2024. Countgd: Multi-modal open-world counting. Advances in Neural Information Processing Systems, 37:48810–48837.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, and 19 others. 2025. Sam 3: Segment anything with concepts. Preprint, arXiv:2511.16719.

Qibo Chen, Weizhong Jin, Jianyue Ge, Mengdi Liu, Yuchao Yan, Jian Jiang, Li Yu, Xuanjiang Guo, Shuchang Li, and Jianzhong Chen. 2025. Cp-detr: Concept prompt guide detr toward stronger universal object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 2141–2149.

Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. 2024. Yolo-world: Real-time open-vocabulary object detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 16901–16911.

Shenghao Fu, Yukun Su, Fengyun Rao, Jing Lyu, Xiaohua Xie, and Wei-Shi Zheng. 2025a. Wedetect: Fast open-vocabulary object detection as retrieval. arXiv preprint arXiv:2512.12309.

Shenghao Fu, Qize Yang, Qijie Mo, Junkai Yan, Xihan Wei, Jingke Meng, Xiaohua Xie, and Wei-Shi Zheng.

2025b. LLMDet: Learning strong open-vocabulary object detectors under the supervision of large language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14987–14997.

Weifu Fu, Jinyang Li, Bin-Bin Gao, Jialin Li, Yuhuan Lin, Hanqiu Deng, Wenbing Tao, Yong Liu, and Chengjie Wang. 2026. Pet-dino: Unifying visual cues into grounding dino with prompt-enriched training. arXiv preprint arXiv:2604.00503.

Yuchen Guan, Chong Sun, Canmiao Fu, Zhipeng Huang, Chun Yuan, and Chen Li. 2025. Text-guided visual prompt dino for generic segmentation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 21288–21298.

Agrim Gupta, Piotr Dollar, and Ross Girshick. 2019. Lvis: A dataset for large vocabulary instance segmentation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 5356–5364.

Qing Jiang, Feng Li, Tianhe Ren, Shilong Liu, Zhaoyang Zeng, Kent Yu, and Lei Zhang. 2023. Trex: Counting by visual prompting. arXiv preprint arXiv:2311.13596.

Qing Jiang, Feng Li, Zhaoyang Zeng, Tianhe Ren, Shilong Liu, and Lei Zhang. 2024. T-rex2: Towards generic object detection via text-visual prompt synergy. In European Conference on Computer Vision, pages 38–57. Springer.

Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. 2021. Mdetr-modulated detection for end-toend multi-modal understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1780–1790.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. 2023. Segment anything. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 4015–4026.

Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. 2020. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal ofComputer Vision, 128:1956– 1981.

Chunyuan Li, Haotian Liu, Liunian Harold Li, Pengchuan Zhang, Jyoti Aneja, Jianwei Yang, Ping Jin, Houdong Hu, Zicheng Liu, Yong Jae Lee, and Jianfeng Gao. 2022a. Elevater: A benchmark and toolkit for evaluating language-augmented visual models. Advances in Neural Information Processing Systems, 35:9287–9301.

Feng Li, Qing Jiang, Hao Zhang, Tianhe Ren, Shilong Liu, Xueyan Zou, Huaizhe Xu, Hongyang Li, Jianwei Yang, Chunyuan Li, Lei Zhang, and Jianfeng Gao. 2024. Visual in-context prompting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12861–12871.

Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. 2022b. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965–10975.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. 2024. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. 2021. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. 2022. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986.

Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii, and Michalis Raptis. 2022. Towards end-to-end unified scene text detection and layout analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1049–1059.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. 2023. Scaling open-vocabulary object detection. In Advances in Neural Information Processing Systems.

Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. 2022. Simple open-vocabulary object detection with vision transformers. In European Conference on Computer Vision, pages 728–755. Springer.

Ting Pan, Lulu Tang, Xinlong Wang, and Shiguang Shan. 2024. Tokenize anything via prompting. In European Conference on Computer Vision, pages 330–348. Springer.

Bo Qian, Dahu Shi, and Xing Wei. 2026. Detr-vip: Detection transformer with robust discriminative visual prompts. arXiv preprint arXiv:2604.14684.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR.

Tianhe Ren, Yihao Chen, Qing Jiang, Zhaoyang Zeng, Yuda Xiong, Wenlong Liu, Zhengyu Ma, Junyi Shen, Yuan Gao, Xiaoke Jiang, Xingyu Chen, Zhuheng Song, Yuhong Zhang, Hongjie Huang, Han Gao, Shilong Liu, Hao Zhang, Feng Li, Kent Yu, and Lei Zhang. 2024. DINO-X: A unified vision model for open-world object detection and understanding. arXiv preprint arXiv:2411.14347.

Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. 2019. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439.

Shuai Shao, Zijian Zhao, Boxun Li, Tete Xiao, Gang Yu, Xiangyu Zhang, and Jian Sun. 2018. Crowdhuman: A benchmark for detecting human in a crowd. arXiv preprint arXiv:1805.00123.

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. 2018. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565.

Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, and 7 others. 2025. DINOv3. Preprint, arXiv:2508.10104.

Ao Wang, Lihao Liu, Hui Chen, Zijia Lin, Jungong Han, and Guiguang Ding. 2025. Yoloe: Real-time seeing anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 24591–24602.

Hao Wang, Pengzhen Ren, Zequn Jie, Xiao Dong, Chengjian Feng, Yinlong Qian, Lin Ma, Dongmei Jiang, Yaowei Wang, Xiangyuan Lan, and Xiaodan Liang. 2024. Ov-dino: Unified openvocabulary detection with language-aware selective fusion. Preprint, arXiv:2407.07844.

Jiaqi Wang, Pan Zhang, Tao Chu, Yuhang Cao, Yujie Zhou, Tong Wu, Bin Wang, Conghui He, and Dahua Lin. 2023. V3det: Vast vocabulary visual detection dataset. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19844–19854.

Junfeng Wu, Yi Jiang, Qihao Liu, Zehuan Yuan, Xiang Bai, and Song Bai. 2024. General object foundation model for images and videos at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3783–3795.

Yifan Xu, Mengdan Zhang, Chaoyou Fu, Peixian Chen, Xiaoshan Yang, Ke Li, and Changsheng Xu. 2023. Multi-modal queried object detection in the wild. Advances in Neural Information Processing Systems, 36:4452–4469.

Lewei Yao, Renjie Pi, Jianhua Han, Xiaodan Liang, Hang Xu, Wei Zhang, Zhenguo Li, and Dan Xu. 2024. Detclipv3: Towards versatile generative openvocabulary object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 27391–27401.

Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianwei Yang, and Lei Zhang. 2023. A simple framework for open-vocabulary segmentation and detection. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 1020–1031.

Yuanhan Zhang, Qinghong Sun, Yichun Zhou, Zexin He, Zhenfei Yin, Kun Wang, Lu Sheng, Yu Qiao, Jing Shao, and Ziwei Liu. 2025. Bamboo: Building mega-scale vision dataset continually with human– machine synergy. International Journal ofComputer Vision, 133(8):5806–5821.

Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, and Jie Chen. 2024. Detrs beat yolos on real-time object detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 16965–16974.

Algorithm 1 Unified Forward and Training Objec  
tive   
Require: Image I, annotations Y, text prompts $P ^ { t } .$ , visual   
prompts V, mode m ∈ {TEXT, VISUAL}   
Ensure: $\dot { \boldsymbol { \mathcal { L } } } = \mathcal { L } _ { \mathrm { d e t } } + \mathcal { L } _ { \mathrm { a l i g n } } \dot { + } \mathcal { L } _ { \mathrm { i c a } }$   
1: $\{ F _ { \ell } \} _ { \ell \in \{ 3 , 4 , 5 \} }  \mathrm { \bf B A C K B O N E } ( I )$   
2: $\{ F _ { \ell } \} \gets \mathrm { C H A N N E L M A P P E R } ( \{ \dot { F } _ { \ell } \} )$   
3: $\dot { F } \gets$ HYBRIDENCODER $( \{ F _ { \ell } \} )$   
4: $\mathbf { i f } m = \mathrm { V I S U A L }$ then   
5: $( P _ { \mathrm { c l s } } ^ { v } , P _ { \mathrm { c o n } } ^ { v } ) \gets \mathsf { V P E } ( F , V )$   
6: $( P , \mathcal { A } ) \gets \mathbf { M E R G E V I S U A L T O T E X T } ( P ^ { t } , P _ { \mathrm { c l s } } ^ { v } )$   
7: else   
$8 \colon \mathrm { ~  ~ \textit ~ { ~ P ~ } ~ } \{ \mathrm { ~  ~ \Gamma ~ } \}$   
9: end if   
10: $( Q _ { 0 } , R _ { 0 } ) \gets \mathrm { T o p K P R O P O S A L S } ( F , P , N _ { q } )$   
11: $\dot { Q } _ { 0 } \gets \dot { \mathbf { C } }$ ONCAT(BUILDDNQUERIE $ { \langle \mathcal { V } , \dot { P } ) , Q _ { 0 } \rangle }$   
12: for l = 1 to 6 do   
13: $Q _ { l } \gets \mathrm { S E L F A T T N } ( Q _ { l - 1 } )$   
14: Q ← CROSSATTNPROMPT $( Q _ { l } , P )$   
15: $\bar { Q } _ { l } \gets \mathrm { C R O S S A T T N I M A G E } ( \bar { Q } _ { l } , F , \bar { R } _ { l - 1 } )$   
16: Q<sub>l</sub> ← FFN(Q<sub>l</sub>)   
17: R<sub>l</sub> ← REFINEREFERENCEPOINTS $\left( Q _ { l } , R _ { l - 1 } \right)$   
18: end for   
19: Split $Q _ { l }$ into $Q ^ { \mathrm { m a t c h } } , Q ^ { \mathrm { d n } }$   
20: $( \hat { \boldsymbol { y } } , \hat { b } ) , ( \hat { \boldsymbol { y } } _ { \mathrm { d n } } , \hat { b } _ { \mathrm { d n } } ) \gets \mathrm { D E T H E A D } ( \boldsymbol { Q } ^ { \mathrm { m a t c h } } , \boldsymbol { Q } ^ { \mathrm { d n } } , P )$   
21: Π ← HUNGARIANMATCH(ˆy, <sup>ˆ</sup>b, Y)   
22: $\mathcal { L } _ { \mathrm { d e t } }  \mathrm { D E T L o s s } ( \hat { y } , \hat { b } , \hat { y } _ { \mathrm { d n } } , \hat { b } _ { \mathrm { d n } } , \mathcal { V } , \Pi )$   
23: if $m =  { \mathrm { V I S U A L } }$ then   
24: $\mathcal { L } _ { \mathrm { a l i g n } }  \mathrm { A L I G N L O s s } ( P _ { . } ^ { t } , P _ { \mathrm { c l s } } ^ { v } ; \mathcal { A } )$   
25: $\mathcal { L } _ { \mathrm { i c a } }  \mathrm { I N F O N C E } ( Q ^ { \mathrm { m a t c h } } , P _ { \mathrm { c o n } } ^ { v } , \Pi )$   
26: else   
27: $\mathcal { L } _ { \mathrm { a l i g n } } , \mathcal { L } _ { \mathrm { i c a } }  0$   
28: end if   
29: return ${ \mathcal { L } } _ { \mathrm { d e t } } + { \mathcal { L } } _ { \mathrm { a l i g n } } + { \mathcal { L } } _ { \mathrm { i c a } }$

<table><tr><td>Dataset</td><td>Prompt Type</td><td>Images</td><td>Ratio</td></tr><tr><td>Objects365</td><td>Text / Visual</td><td>0.6M</td><td>7.5%</td></tr><tr><td>OpenImages</td><td>Text / Visual</td><td>1.67M</td><td>20.9%</td></tr><tr><td>GoldG</td><td>Text</td><td>76K</td><td>1.0%</td></tr><tr><td>V3Det</td><td>Text</td><td>0.15M</td><td>1.9%</td></tr><tr><td>CrowdHuman</td><td>Visual</td><td>15K</td><td>0.2%</td></tr><tr><td>HierText</td><td>Visual</td><td>8K</td><td>0.1%</td></tr><tr><td>Bamboo-Det</td><td>Text / Visual</td><td>1.08M</td><td>13.5%</td></tr><tr><td>Bamboo-CLS†</td><td>Text</td><td>1.56M</td><td>19.6%</td></tr><tr><td>CC3M†</td><td>Text</td><td>1.32M</td><td>16.5%</td></tr><tr><td> $\mathbf { S A - } 1 \mathbf { B } ^ { \dagger }$ </td><td>Visual</td><td>1.5M</td><td>18.8%</td></tr><tr><td>Total</td><td>Text / Visual</td><td>7.98M</td><td>100.0%</td></tr></table>

Table 8: Training datasets used in OPUS. Datasets marked with † have annotations generated by our SAM3- based data engine pipeline.

## A Method Details

## A.1 Model Architecture and Forward

Algorithm 1 summarizes the forward propagation and loss computation of our unified detector during training. Given an input image, the model first encodes category prompts using the text encoder and extracts multi-scale visual features through the backbone and hybrid encoder. Under the VI-SUAL mode, the Visual Prompt Encoder (VPE) further produces category-aware visual prompt embeddings and merges them into the text prompt space through visual-to-text prompt interaction. The decoder adopts alternating self-attention, text crossattention, and image cross-attention for iterative query refinement. Following the two-stage query initialization and denoising strategy, the model computes both matching and denoising detection losses. Additional alignment and instance-level contrastive objectives are introduced only for visual prompt training.

Algorithm 2 Alternating Training   
Require: text dataloader $\mathcal { D } _ { \mathrm { t e x t } } ,$ , visual dataloader $\mathcal { D } _ { \mathrm { v i s } } ,$ total   
iterations $T _ { \mathrm { m a x } } ,$ alternating ratio 1:8 (VISUAL : TEXT)   
Ensure: Parameters Θ   
1: for t = 0 to $T _ { \mathrm { m a x } } - 1$ do   
2: if t mod $9 = 8$ then   
3: $( I , \mathcal { V } ) \sim \mathcal { D } _ { \mathrm { v i s } }$   
4: mode ← VISUAL   
5: else   
6: $( I , \mathcal { V } ) \sim \mathcal { D } _ { \mathrm { t e x t } }$   
7: mode ← TEXT   
8: end if   
9: $\mathcal { L } $ FORWARDANDLOSS $( I , { \mathcal { Y } } ,$ mode; Θ) {Algo  
rithm 1}   
10: Θ ← ADAMWUPDATE $\left( \Theta , \nabla _ { \Theta } \mathcal { L } \right)$   
11: end for

Algorithm 3 Open-Vocabulary Inference   
Require: input image $I ;$ mode m; text prompts $P ^ { t } ;$ visual   
prompts $\overset { \star } { V }$   
Ensure: detections $\{ ( b _ { j } , s _ { j } , c _ { j } ) \} _ { j = 1 } ^ { N }$   
1: $F $ ENCODER(I)   
2: if $m = \mathrm { T E X T }$ then   
$3 \colon \mathrm { ~ \quad ~ } P \gets P ^ { t }$   
4: else   
5: $( P _ { \mathrm { c l s } } ^ { v } , P _ { \mathrm { c o n } } ^ { v } ) \gets \mathsf { V P E } ( F , V )$ or precomputed $P _ { \mathrm { c l s } } ^ { v }$   
$6 { \div } \mathrm { ~  ~ { ~ \hat { ~ } { ~ P ~ } { ~  ~ M E R G E V I S U A L T O T E X T } ^ { i n f e r } ( \vec { P } ^ { t } , \vec { P } _ { c l s } ^ { v } , m ) } }$   
7: end if   
8: $( Q _ { 0 } , R _ { 0 } ) \gets \mathrm { T o p K P R O P O S A L S } ( F , P , N _ { q } )$   
9: $Q , R $ DECODER $( Q _ { 0 } , P , F , \tilde { R _ { 0 } } )$   
10: $\{ ( b _ { j } , s _ { j } , c _ { j } ) \} \gets \mathrm { D E T H E A D . P R E D I C T } ( Q , P )$   
11: return $\{ ( b _ { j } , s _ { j } , c _ { j } ) \}$

## A.2 Training and Optimization

Algorithm 2 presents the alternating optimization strategy used in training. To jointly support text prompting and visual prompting learning under a unified detector, we alternate between visual batches and text prompt batches with a ratio of 1:8. Text batches optimize conventional openvocabulary detection objectives using enhanced category prompts and sampled negative categories, while visual-prompt batches additionally enable visual-text prompt interaction and auxiliary visual prompt losses. All parameters are optimized endto-end using AdamW.

<table><tr><td>Source</td><td>#Images</td><td>#Boxes</td><td>Box Acc. (%)</td></tr><tr><td>Bamboo-CLS</td><td>200</td><td>653</td><td>91.9</td></tr><tr><td>CC3M</td><td>200</td><td>1291</td><td>89.5</td></tr><tr><td>SA-1B</td><td>200</td><td>7897</td><td>91.2</td></tr></table>

Table 9: Human verification of SAM3 pseudo-labels after filtering. A box is counted as incorrect if localization or the assigned label is wrong.

## A.3 Inference Workflows

Algorithm 3 describes the unified inference workflow supporting four inference modes: text prompting, interactive visual prompting, generic visual prompting, mixed prompting. Depending on the inference mode, the detector dynamically constructs textual prompts, user-provided interactive prompts, or cached visual embeddings. The encoded visual prompts are merged into the text embedding space through the same visual-to-text interaction used during training. For Mixed prompting, there is no learned fusion module: text and class-level visual embeddings are averaged at matched class slots, identical to the training-time merge. The decoder then performs iterative object query refinement to produce final open-vocabulary detections. For generic visual prompting, visual embeddings can be precomputed and cached to improve inference efficiency.

## A.4 Data Engine

Section 3 describes our SAM3-based singlepass pipeline that converts text-image, classification, and segmentation sources into Bamboo-CLS, CC3M, and annotated SA-1B supervision. Here we detail the filtering stages and show qualitative cases. Dataset statistics are summarized in Table 8.

Human verification. We randomly sample 200 images from each generated source and manually verify every pseudo box–phrase pair after filtering. A box counts as incorrect if localization or the assigned label is wrong. Table 9 reports box accuracies near 90% on Bamboo-CLS, CC3M, and SA-1B.

Source-specific preprocess filtering. Different sources introduce different noise. For CC3M, noun phrases extracted from captions are used only as candidate text prompts and must be visually grounded by SAM3, which removes many ungroundable phrases. For Bamboo-CLS, imagelevel labels can be coarser than object regions or cover only part of the visible content; subsequent grounding and filtering mitigate this mismatch while retaining scalable supervision. For SA-1B, semantic labels come from TAP over a vocabulary of 2560 common categories; we discard unreliable tags with category-dependent confidence thresholds.

Unified post-processing. All generated boxes then pass through a shared post-processing stage: confidence thresholding, category-wise and imagewise NMS, removal of near full-image boxes, and filtering of large boxes that enclose many small same-category instances.

Qualitative cases. Figure 4 and Figure 5 show representative success and failure cases after filtering, with one column each for Bamboo-CLS, CC3M, and SA-1B. Successful examples typically produce tight object-level boxes with consistent labels. Residual failures include mislabeled boxes, nested boxes, and redundant background boxes on regions without clear object features (e.g., sky). On Bamboo-CLS and CC3M, nested or redundant boxes are more common; on SA-1B, mislabels appear more often.

## B Experimental Setup Details

## B.1 Training Hyperparameter Configuration

Our training hyperparameter configuration is summarized in Table 10. Unless otherwise specified, all models are trained under the same optimization setting, including input resolution, learningrate schedule, alternating training ratio, and loss weighting strategy, to ensure fair comparison under a fixed efficiency budget.

## B.2 Sensitivity Analysis

We study sensitivity to Visual-G reference sampling (N), the visual:text alternating ratio, and ICA temperature τ.

## B.2.1 Visual-G Reference Sampling

Table 11 varies N ∈ {1, 4, 8, 16, 32, 64} to verify the default N=16 choice. Visual-G needs enough references for stable category-level prototypes, rising from 25.0 AP at N=1 to 38.6 AP at N=16 on LVIS-minival, and then saturates beyond N=16. For rare categories, the training set may contain fewer usable images than N, so larger N does not always mean more distinct references. Under Mixed prompting, text provides an additional anchor, so performance varies less with N and already exceeds Text-only from N=4 on LVISminival and from N=16 on COCO. Five random seeds at the default N=16 confirm that this choice is stable: Visual-G is 43.3±0.1 on COCO and 38.6±0.3 on LVIS-minival, and Mixed is 49.9±0.1 and $4 5 . 5 { \pm } 0 . 2 \ $

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Backbone</td><td>DINOv3-ConvNeXt-B</td></tr><tr><td>Text encoder</td><td>CLIP ViT-B/32</td></tr><tr><td>Image size</td><td> $6 4 0 \times 6 4 0$ </td></tr><tr><td>Batch size per GPU</td><td>8</td></tr><tr><td>GPUs</td><td>16</td></tr><tr><td>Global batch size</td><td>128</td></tr><tr><td>Training iterations</td><td>400K</td></tr><tr><td>Alternating ratio</td><td>1:8 (VISUAL:TEXT)</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Base learning rate</td><td> $4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Backbone / text encoder LR</td><td> $4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Gradient clipping</td><td>0.1</td></tr><tr><td>Warmup</td><td>1K iterations, start factor 0.1</td></tr><tr><td>LR scheduler</td><td>MultiStep (50%, 90%;  $\gamma = 0 . 1 )$ </td></tr><tr><td>Precision</td><td>FP16 (AMP)</td></tr><tr><td>Detection queries</td><td>900</td></tr><tr><td>DN queries</td><td>100</td></tr><tr><td>Max categories per image</td><td>256</td></tr><tr><td>Loss weights</td><td>cls / bbox / iou / align / ica  $1 . 0 / 5 . 0 / 2 . 0 / 1 . 0 / 1 . 0$ </td></tr><tr><td>Align loss</td><td>Cross-entropy w/ learnable log-scale</td></tr><tr><td>ICA temperature</td><td> $\mathrm { I n f o N C E } \left( \tau = 0 . 0 7 \right)$ </td></tr><tr><td>Validation interval</td><td>20K iterations</td></tr></table>

Table 10: Training configuration.

<table><tr><td rowspan="2">N</td><td rowspan="2">Prompt COCO AP</td><td colspan="3">LVIS-mv</td></tr><tr><td>AP 25.0</td><td>APf APc APr 25.6 25.4 20.0</td></tr><tr><td>1 4 8 16</td><td>Visual-G Visual-G Visual-G Visual-G</td><td>27.6 38.6 41.6 43.4</td><td>34.5 35.1 37.1 38.0 38.6 39.5</td><td>34.8 28.8 36.9 33.6 38.8 32.8</td></tr><tr><td>32 64 1 4</td><td>Visual-G Visual-G Mixed Mixed</td><td>44.3 44.7 46.2 49.1</td><td>39.7 40.1 39.7 40.4 40.5 42.6 44.7 46.1 45.1 46.7 45.2</td><td>40.7 32.3 40.3 33.0 39.4 34.2 44.5 38.6 44.2 39.9</td></tr></table>

Table 11: Visual-G and Mixed prompting vs. number of references (N) on the final OPUS model. Text-only: 49.6 AP (COCO) / 43.0 AP (LVIS-minival). Bold marks the default N=16.

<table><tr><td>visual:text</td><td>Visual-I</td><td>Visual-G</td><td>Text</td><td>Mixed</td></tr><tr><td>1:4</td><td>64.6</td><td>31.5</td><td>31.6</td><td>36.8</td></tr><tr><td>1:8</td><td>63.2</td><td>33.5</td><td>33.2</td><td>38.0</td></tr><tr><td>1:16</td><td>58.8</td><td>34.6</td><td>33.7</td><td>38.5</td></tr></table>

Table 12: Sensitivity to the visual:text alternating ratio on LVIS-minival.

## B.2.2 Visual:Text Alternating Ratio

Table 12 ablates the visual:text alternating ratio on the ICA-equipped model under the O365+GoldG setting (main-text Table 4, step D). A larger visual share (1:4) favors Visual-I, while a larger text share (1:16) favors Text, Visual-G, and Mixed. We adopt 1:8 as a balanced default across all four prompt modes, matching the main experiments.

## B.2.3 ICA Temperature

Table 13 varies the ICA temperature τ under the same setting as Table 12, with $w _ { \mathrm { I C A } } { = } 1 . 0$ . Across

<table><tr><td>τ</td><td>Visual-I</td><td>Visual-G</td><td>Text</td><td>Mixed</td></tr><tr><td>0.05</td><td>63.3</td><td>33.7</td><td>32.7</td><td>38.4</td></tr><tr><td>0.07</td><td>63.2</td><td>33.5</td><td>33.2</td><td>38.0</td></tr><tr><td>0.10</td><td>63.1</td><td>33.4</td><td>33.2</td><td>38.5</td></tr></table>

Table 13: ICA temperature sensitivity (τ) on LVISminival.

$\tau \in \{ 0 . 0 5 , 0 . 0 7 , 0 . 1 0 \}$ , Visual-I, Text, and Mixed change only slightly, and Visual-G stays within 0.3 AP. We therefore keep $\tau { = } 0 . 0 7$ as the default.

## B.3 Evaluation Protocol

## B.3.1 Accuracy Evaluation Protocols

Text Protocol. Under the text prompt protocol, category names are used as textual inputs and encoded by a pretrained text encoder. The resulting text embeddings are projected into the shared prompt space and used as category prompts for open-vocabulary detection. This protocol evaluates the detector’s ability to recognize categories from language descriptions alone.

Visual-I Protocol. Under the interactive visual prompt protocol, one ground-truth instance is randomly selected for each category in a test image and used as the visual prompt. The model then detects all instances of the corresponding categories in the same image. This setting simulates interactive applications such as annotation and object counting, where users provide instance-level guidance at inference time.

Visual-G Protocol. Under the generic visual prompt protocol, category-level visual prompts are constructed offline from reference images. For each category, we randomly sample a fixed number of training images, encode their visual prompts, and aggregate them into a category-level visual prototype. The resulting prototypes are reused across all test images during inference. This protocol evaluates whether visual prompts can serve as generic category descriptions without per-image interaction.

Mixed Protocol (Text + Visual-G). Under the mixed protocol, text prompt embeddings and Visual-G prototypes are independently constructed and jointly used as category prompts during inference. The two embeddings are averaged at matched class slots without a learned fusion module. This setting evaluates whether semantic text cues and appearance-level visual cues can complement each other in a unified detector.

## B.3.2 Efficiency Evaluation Protocols

Interactive Speed $( \mathrm { F P S _ { i n t e r } } )$ . This metric measures inference throughput in the interactive visual prompting setting. Following prior work, image features are cached for each test image, and the measured runtime includes visual prompt encoding and decoder inference for each user interaction. This reflects the latency-sensitive use case where users iteratively provide visual prompts on the same image.

Generic Speed $\mathrm { ( F P S _ { g e n } ) }$ This metric measures inference throughput for Text and Visual-G prompting. In this setting, category prompt embeddings are precomputed offline and reused across test images. The measured runtime therefore includes image feature extraction and decoder inference, but excludes repeated prompt encoding. This reflects standard generic detection, where a fixed set of category prompts is applied to many images.

## C More Experimental Analysis

## C.1 Breakdown of Component-wise Ablation

Table 14 keeps the same A→E steps as Table 4, and breaks each step into $\mathbf { A P } _ { f } / \mathbf { A P } _ { c } / \mathbf { A P } _ { r }$ . Overall, the progression from the baseline to the full OPUS model improves performance across most category groups and prompting modes, showing that the gains are not limited to frequent categories.

A notable exception appears when adding the prompt-aware decoder in Row C. While the decoder improves Text and Visual-I prompting, it temporarily reduces Visual-G performance, with the largest drop on rare categories. This supports our interpretation that Visual-G is more sensitive to the mismatch between Visual-I-style training prompts and cross-image category prototypes used during Visual-G inference. Rare categories are particularly affected because their prototypes are aggregated from fewer and more diverse examples, making them less stable than frequent-category prototypes.

Adding ICA in Row D produces a selective recovery pattern under Visual-G. Although the overall Visual-G AP recovers only modestly, AP<sub>r</sub> improves substantially from 26.1 to 30.0, while $\mathsf { A P } _ { f }$ gains slightly and $\mathbf { A P } _ { c }$ remains below the predecoder level. This pattern suggests that ICA is particularly useful when category-level visual prototypes are unreliable, especially for rare categories.

To understand this effect, recall that the base alignment objective $\mathcal { L } _ { \mathrm { a l i g n } }$ operates at the class level. It trains the aggregated visual prompt feature $\mathbf { P } _ { \mathrm { c l s } } ^ { v }$ by pulling it toward the corresponding text prompt feature $\mathbf { P } ^ { t }$ . While this alignment encourages text-visual category consistency, it also makes Visual-G prototypes depend on a global semantic anchor. Such an anchor is coarse by design: text features mainly encode category-level semantics rather than instance-level appearance variations. This limitation becomes more pronounced for rare categories, where both the text representations and the cross-image visual prototypes receive less reliable supervision. As a result, aligning Visual-G features primarily through $\mathcal { L } _ { \mathrm { a l i g n } }$ may weaken finegrained visual details and make rare-category prototypes inherit the limitations of the text modality.

<table><tr><td rowspan="2">ID</td><td colspan="4">Visual-I</td><td colspan="4">Visual-G</td><td colspan="4">Text</td><td colspan="4">Mixed</td></tr><tr><td>AP</td><td> $\mathbf { A P } _ { f }$ </td><td> $\mathbf { A P } _ { c }$ </td><td> $\mathbf { A P } _ { r }$ </td><td>AP</td><td> $\mathbf { A P } _ { f }$ </td><td> $\mathbf { A P } _ { c }$ </td><td> $\mathbf { A P } _ { r }$ </td><td>AP</td><td> $\mathbf { A P } _ { f }$ </td><td> $\mathbf { A P } _ { c }$ </td><td> $\mathbf { A P } _ { r }$ </td><td>AP</td><td> $\mathbf { A P } _ { f }$ </td><td> $\mathbf { A P } _ { c }$ </td><td> $\mathbf { A P } _ { r }$ </td></tr><tr><td>A</td><td>56.3</td><td>48.4</td><td>63.7</td><td>63.8</td><td>33.6</td><td>34.8</td><td>33.4</td><td>28.0</td><td>26.5</td><td>34.5</td><td>19.7</td><td>16.0</td><td>35.2</td><td>40.0</td><td>32.3</td><td>22.8</td></tr><tr><td>B</td><td>57.5</td><td>49.3</td><td>64.9</td><td>66.8</td><td>35.3</td><td>36.6</td><td>35.1</td><td>29.1</td><td>29.5</td><td>34.9</td><td>25.2</td><td>20.4</td><td>36.2</td><td>38.9↓</td><td>33.8</td><td>33.1</td></tr><tr><td>C</td><td>63.0</td><td>54.1</td><td>70.8</td><td>73.9</td><td>33.3↓</td><td>34.7↓</td><td>33.1↓</td><td>26.1.↓</td><td>32.7</td><td>38.7</td><td>28.5</td><td>20.5</td><td>36.5</td><td>39.7</td><td>34.2</td><td>30.8↓</td></tr><tr><td>D</td><td>63.2</td><td>54.6</td><td>70.5↓</td><td>74.7</td><td>33.5</td><td>35.5</td><td>32.1↓</td><td>30.0</td><td>33.2</td><td>38.9</td><td>29.4</td><td>20.3↓</td><td>38.0</td><td>40.4</td><td>36.8</td><td>31.1</td></tr><tr><td>E</td><td>69.2</td><td>61.7</td><td>75.7</td><td>78.3</td><td>38.6</td><td>39.5</td><td>38.8</td><td>32.8</td><td>43.0</td><td>45.5</td><td>41.3</td><td>36.6</td><td>45.2</td><td>47.1</td><td>44.4</td><td>38.7</td></tr></table>

Table 14: Detailed $A P _ { f } / A P _ { c } / A P _ { r }$ breakdown of component-wise ablation on LVIS-minival across four prompting modes (steps A→E as in Table 4).
<table><tr><td>ID</td><td>Evolution Step</td><td>Params</td><td>Input</td><td>GFLOPs</td><td> $\overline { { \mathbf { F P S _ { g e n } } } }$ </td><td>Visual-I</td><td>Visual-G</td><td>Text</td><td>Mixed</td></tr><tr><td> $\mathbf { A }$ </td><td>Baseline (Swin-T)</td><td>115M</td><td> $\overline { { 8 0 0 \times 1 3 3 3 } }$ </td><td>257</td><td>15.4</td><td>56.3</td><td>33.6</td><td>26.5</td><td>35.2</td></tr><tr><td></td><td> $\begin{array} { r l } { \mathrm { B } _ { 1 } } & { { } + \mathrm { D I N O v 3 - C o n v N e X t - B } } \end{array}$ </td><td>173M</td><td> $8 0 0 \times 1 3 3 3$ </td><td>439</td><td>11.6</td><td>63.2</td><td>38.0</td><td>36.4</td><td>39.3</td></tr><tr><td></td><td> $\begin{array} { r l } { \mathbf { B } _ { 2 } } & { { } + \mathbf { R e d u c e \ t o \ 6 4 0 \times 6 4 0 } } \end{array}$ </td><td>173M</td><td> $6 4 0 \times 6 4 0$ </td><td>208</td><td>19.4</td><td>59.1</td><td>33.4</td><td>28.6</td><td>34.4</td></tr><tr><td></td><td>B3 + Hybrid encoder</td><td>172M</td><td> $6 4 0 \times 6 4 0$ </td><td>164</td><td>22.0</td><td>57.5</td><td>35.3</td><td>29.5</td><td>36.2</td></tr><tr><td></td><td> $\begin{array} { r l } { \mathrm { C } } & { { } + \mathrm { P r o m p t - a w a r e ~ d e c } . } \end{array}$ </td><td>174M</td><td> $6 4 0 \times 6 4 0$ </td><td>165</td><td>21.3</td><td>63.0</td><td>33.3</td><td>32.7</td><td>36.5</td></tr><tr><td>D</td><td>+ICA</td><td>174M</td><td> $6 4 0 \times 6 4 0$ </td><td>165</td><td>21.3</td><td>63.2</td><td>33.5</td><td>33.2</td><td>38.0</td></tr><tr><td>E</td><td>+ Data engine</td><td>174M</td><td> $6 4 0 \times 6 4 0$ </td><td>165</td><td>21.3</td><td>69.2</td><td>38.6</td><td>43.0</td><td>45.2</td></tr></table>

Table 15: Joint efficiency and LVIS-minival accuracy view of the component-wise progression. IDs match Table 6.

ICA addresses this issue from a different direction. Instead of aligning only the aggregated classlevel visual prototype to text, ICA aligns instancelevel visual prompt features $\mathbf { P } _ { \mathrm { c o n } } ^ { v }$ directly with their matched decoder queries. These decoder queries are grounded in image regions through detection supervision, and therefore provide a more localized and appearance-sensitive target than text embeddings. This instance-query alignment helps preserve fine-grained visual information without being constrained solely by the semantic coarseness of the text anchor. It also provides a stronger learning signal for rare categories, where class-level textvisual alignment is least reliable. This explains why ICA yields its clearest Visual-G recovery on $\mathsf { A P } _ { r }$ rather than uniformly improving all frequency groups.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Training data</td><td colspan="3">COCO</td><td colspan="3">LVIS-mv</td></tr><tr><td>Visual-I Visual-G</td><td></td><td></td><td>Text Visual-I Visual-G</td><td></td><td>Text</td></tr><tr><td rowspan="2"></td><td>T-Rex2-T O365, OI, HT, CH</td><td>41.1</td><td>41.1</td><td>一</td><td>40.6</td><td>38.1</td><td>一</td></tr><tr><td>O365, OI, HT, CH, SA-1B</td><td>56.6</td><td>38.8</td><td>45.8</td><td>59.3</td><td>37.4</td><td>42.8</td></tr></table>

Table 16: Official T-Rex2-T second-stage visual-prompt training results from T-Rex2 Table 6. Compare with our one-stage progressive scaling in Table 7.

## C.2 Component-wise Efficiency and Accuracy

Table 15 lists Params, input size, GFLOPs, and FPS together with Visual-I/Visual-G/Text/Mixed AP for each intermediate configuration. It complements the main-text efficiency table, which omits AP to avoid duplicating Table 4, and makes the accuracy/efficiency trade-off along step B easier to inspect in one place.

## C.3 Comparison with Staged Visual-Prompt Training

Table 16 recalls official T-Rex2-T second-stage visual-prompt results from T-Rex2 Table 6. These numbers are a reference comparison against reported results, not a controlled re-training under the same implementation.

With only O365 for visual-prompt training, our one-stage re-impl. already reaches Visual-I 57.2/56.3 on COCO/LVIS-minival (Table 7), while official staged training reports 41.1/40.6 before SA-1B and 56.6/59.3 after. Under Visual-G, adding

SA-1B lowers official T-Rex2-T from 41.1 to 38.8 on COCO and from 38.1 to 37.4 on LVISminival. The one-stage re-impl. instead stays flat on COCO (41.3→41.4) and rises on LVIS-minival (33.6→40.5).

## C.4 Visualizations

We provide qualitative results for interactive visual prompting, generic visual prompting, and text prompting. These examples complement the quantitative results by showing how OPUS behaves across dense scenes, cross-image exemplar transfer, and open-vocabulary text-guided detection.

Figure 6 compares OPUS with T-Rex2 and PET-DINO under interactive visual prompting. OPUS demonstrates stronger visual prompt understanding for both single-category and multi-category targets in diverse and dense scenes.

Figure 7 further compares OPUS with T-Rex2 on challenging interactive visual prompting cases. OPUS produces fewer background false positives and more accurate instance localization.

Figure 8 shows generic visual prompting results, where exemplar prompts from reference images are transferred to different test images. OPUS effectively preserves category-level visual semantics across changes in scene layout, object scale, and instance appearance.

Figure 9 presents text-prompted zero-shot detection examples.

Together, these visualizations show that OPUS provides robust prompt understanding across text, interactive visual, and generic visual prompting settings.

![](images/ec000ac4319b809f93cbf7bac6415374d93090d374ec2796673d5b0973335b69.jpg)  
Figure 4: Representative SAM3 pseudo-label success cases after filtering on Bamboo-CLS, CC3M, and SA-1B.

![](images/82765c5794cc4d67a2305783bccbdc508180cd62548ce448fcef18979af00812.jpg)  
Figure 5: Representative SAM3 pseudo-label failure cases after filtering on Bamboo-CLS, CC3M, and SA-1B (mislabel, nested, and redundant background boxes).

![](images/c653346e517c126bd71671af5987151f5917a18434ce6daa9512e5c09c8001a8.jpg)  
Figure 6: Interactive visual prompt detection results of OPUS compared with T-Rex2 and PET-DINO. OPUS demonstrates stronger visual prompt understanding for both single-category and multi-category targets across diverse and dense scenes.

![](images/20e51f4a128c2cc439dcdd1064d9aa0a7c92fb732f873282d9c30d2f8e56d3a3.jpg)  
Figure 7: Comparison with T-Rex2 on challenging interactive visual detection cases. OPUS produces more accurate localization results with fewer FPs.

![](images/eab1c1aa80bdf211c1c53ba54b0a108560c21e6a46f74a7259625ca9c99ae677.jpg)  
Figure 8: Zero-shot detection visualizations of OPUS using cross-image exemplar visual prompts. The top row shows exemplar prompts, and the bottom row presents the corresponding detection results.

![](images/d5f7bc6557b88a25cb0c8810ead5f93cdba3bbc164349274aa6bfa00da09d34f.jpg)  
Figure 9: Zero-shot detection visualizations of OPUS using text prompts.