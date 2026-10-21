# FocusOVD: Geometric and Contextual Feature Refinement for Open-Vocabulary Small-Object Detection

Hanlin Kou<sup>1</sup> <sup>[0009−0000−4502−3626]</sup> , Beiqi Bai<sup>1</sup> <sup>[0009−0000−5587−3843]</sup> , Li Chen<sup>1</sup>, Youjia Jing<sup>2</sup>, Shupan Li<sup>1</sup>, and Mingyuan Jiu<sup>1</sup>

<sup>1</sup> School of Computer Science and Artificial Intelligence, Zhengzhou University, Zhengzhou,China

https://www7.zzu.edu.cn/csai/

<sup>2</sup> Sinosteel Zhengzhou Research Institute of Steel Wire Products Co., Ltd., China

Abstract. Open-vocabulary object detection aims to detect both seen and unseen categories by leveraging the semantic knowledge of vision-language models. However, existing methods often struggle with small objects due to insuficient preservation of high-resolution details, inefective information aggregation across feature levels, and limited region-level discriminability. To address these challenges, we propose FocusOVD, an open-vocabulary object detector that enhances small-object representations without modifying the frozen VLM backbone. FocusOVD introduces three complementary components: a Reweighted Cross-scale Feature Fusion module, which learns normalized level-wise weights to adaptively balance shallow appearance details and deep semantic cues during feature pyramid aggregation; a Residual Pinwheel-shaped Gated Convolution module, which strengthens local geometric responses in high-resolution features; and a Residual Convolutional Gated Linear Unit module, which suppresses backgrounddominated activations and enhances RoI discriminability before prediction. Extensive experiments on COCO-OVD and LVIS-OVD demonstrate the efectiveness of FocusOVD. On COCO-OVD, FocusOVD achieves a novel-category AP of 36.3%, outperforming CORA, UniDetector, and SAMP by 1.3%, 1.1%, and 1.5%, respectively; on LVIS-OVD, it achieves an overall AP of 33.7%, surpassing the strongest competing method by 0.6%. These results demonstrate the strong generalization ability of FocusOVD for open-vocabulary small-object detection.

Keywords: Open-Vocabulary Detection · Small-Object Detection · Vision-Language Models.

## 1 Introduction

Conventional closed-set detectors perform well on predefined categories but assume that the target vocabulary is known during training, limiting deployment in evolving environments [1]. Open-vocabulary detection (OVD) addresses this limitation by leveraging pretrained vision-language models (VLMs) to recognize categories unseen during detector training [3]. However, open-vocabulary small object detection (OVSD) remains challenging. Small objects occupy only a few pixels in the image, resulting in weak foreground information, and their detection demands finer-grained detail cues, more effective cross-scale interaction mechanisms, and more precise local representations [2]. Moreover, general OVD methods are not specifically tailored to small object detection: high-resolution details tend to decay during feature propagation, and sparse RoI activation signals are easily overwhelmed by background responses [1, 2], which renders small objects prone to missed detection or false positives.

Despite rapid progress in OVD, small-object recognition in open-vocabulary settings remains underexplored. To fill this research gap, we propose FocusOVD, a framework specifically designed for OVSD, targeting three bottlenecks: detail erosion along highresolution pathways, insuficient level-wise reweighting between shallow spatial cues and deep semantic context, and weak region-level discrimination under sparse foreground evidence. Specifically, (1) Reweighted Cross-scale Feature Fusion (RCFF) is proposed to balance shallow appearance details and deep semantic context during feature pyramid aggregation by learning adaptive weights; (2) Residual Pinwheel-shaped Gated Convolution (RPGConv) is designed, which uses directional depthwise processing and a lightweight gate to enhance local geometric responses while preserving the original feature representation; (3) Residual Convolutional Gated Linear Unit (RCGLU) is introduced to refine RoI features through gated spatial modeling and a residual update, improving foreground-background discrimination before prediction.

The remainder of this paper is organized as follows: Section 2 reviews related work, Section 3 presents FocusOVD, Section 4 reports the experimental results, and Section 5 concludes the paper.

## 2 Related Work

## 2.1 Open-Vocabulary Object Detection

Open-vocabulary object detection (OVD) targets categories absent from detector training using large-scale vision-language models (VLMs). Existing approaches mainly fall into three categories: knowledge-distillation methods, region-text alignment methods, and frozen-VLM methods. Knowledge-distillation methods transfer VLM knowledge to a detector: ViLD distills CLIP image and text embeddings into region features, while HierKD [21] transfers global and instance-level semantics through hierarchical distillation. These methods depend on auxiliary distillation objectives and teacher quality, adding training complexity. Region-text alignment methods learn region-language correspondence from grounding or caption data. RegionCLIP [5] constructs region-text pairs with CLIP pseudo-labels for contrastive pretraining, whereas GLIP [6] unifies detection and phrase grounding through joint pretraining. They require large-scale weakly labeled data and substantial pretraining. Frozen-VLM methods retain the pretrained backbone and train lightweight detection components. F-VLM [7] directly exploits frozen CLIP features for localization and open-vocabulary classification, while CORA [8] uses region prompting and anchor pre-matching to bridge image-level and regionlevel recognition. These methods preserve VLM generalization while reducing training cost. Recent work further improves region-language alignment and contextual modeling. DenseVLM [27] decouples foreground and background semantics to reduce foreground bias in region-language alignment, while CODet [28] incorporates visual and textual object co-occurrence knowledge to guide open-vocabulary detection.

## 2.2 Small-Object Detection

Small-object detection focuses on objects occupying few pixels, such as instances smaller than $3 2 \times 3 2$ pixels. Limited spatial resolution and feature degradation caused by downsampling make small objects dificult to detect. Existing methods address this problem primarily through enhanced multi-scale representation, contextual reasoning, or data and optimization strategies. FPN [2], PANet [22], and BiFPN [15], for instance, enhance multi-scale feature aggregation; SNIP [24] and TridentNet [25] adopt scale-aware training or feature processing, and PyramidBox [26] exploits contextual information.

However, most small-object detectors assume closed-vocabulary recognition. In OVD, the frozen VLM backbone preserves semantic generalization but restricts adaptation: small-object features are easily diluted during pyramid aggregation and remain dificult to align with textual embeddings. Improving these representations without modifying the VLM backbone remains underexplored. FocusOVD addresses this gap without modifying the frozen VLM backbone by enhancing the detector-side FPN with cross-scale fusion and high-resolution local modeling, and refining RoI features before the Fast R-CNN head.

## 3 Methodology

## 3.1 Overall Architecture

FocusOVD is a two-stage open-vocabulary detector [1] built on a frozen CLIP-based vision-language model (VLM) [3], as shown in Fig. 1. Given an input image, the frozen image encoder extracts multi-level features $C _ { 2 ^ { - } } C _ { 5 }$ , while the text encoder maps category names into the shared vision-language embedding space. The text encoder is used to generate category embeddings for visual-text alignment. Then, these visual features are processed by a modified Feature Pyramid Network (FPN). In this network, RPGConv is inserted between high-resolution feature $C _ { 3 }$ and 1 × 1 convolution to enhance local geometric details for small objects. RCFF replaces the conventional top-down fusion at the $P _ { 2 } , P _ { 3 }$ and $P _ { 4 }$ levels, enabling adaptive aggregation of fine-grained and semantic information. The remaining layers follow the standard FPN design, producing multiscale features $P _ { 2 } – P _ { 6 }$

The RPN generates candidate boxes from the feature pyramid, and RoIAlign extracts fixed-size region features. RCGLU further refines these features before they are passed to the box head for classification and bounding-box regression. FocusOVD integrates frozen VLM semantics with enhanced high-resolution geometric modeling, cross-scale fusion, and RoI feature refinement to produce open-vocabulary detections.

## 3.2 Residual Pinwheel-Shaped Gated Convolution

To enhance local geometry while limiting changes to the frozen VLM representation, RPGConv is applied only to the high-resolution $C _ { 3 }$ feature before its lateral projection,

![](images/fd811820e1c12601a3e8f98e2526ca54c51ecaf1da6f2ec757a6cfaf189f0afe.jpg)  
Fig. 1. Overview of FocusOVD.

as shown in Fig. 1. Let � denote the channel dimension of the input feature $C _ { 3 }$ . The compressed channel dimension is defined as

$$
C _ { m } = \operatorname* { m a x } ( \lfloor 0 . 5 C \rfloor , 1 ) .\tag{1}
$$

The input feature is first projected from � to $C _ { m }$ channels using a $1 \times 1$ convolution, followed by Group Normalization (GN) and the Sigmoid Linear Unit (SiLU) activation:

$$
X = \operatorname { S i L U } \left( \operatorname { G N } \left( \operatorname { C o n v } _ { 1 \times 1 } ( C _ { 3 } ) \right) \right) \in \mathbb { R } ^ { H \times W \times C _ { m } } .\tag{2}
$$

GN normalizes feature channels within groups and does not depend on batch-level statistics, making it suitable for detection training with small batch sizes. The number of GN groups is determined by

$$
N _ { \mathrm { G N } } ( C _ { m } ) = \operatorname* { m a x } \left\{ g \in \mathbb { N } \mid g \leq 3 2 , C _ { m } \ \mathrm { m o d } \ g = 0 \right\} .\tag{3}
$$

The compressed feature � is then processed by four asymmetric depthwise convolution (DWConv) branches arranged along the four cardinal directions. The left and right branches are horizontal branches using $1 \times 3$ kernels, while the up and down branches

are vertical branches using $3 \times 1$ kernels. The four branches are defined as

$$
\begin{array} { r l } & { X _ { l } = \mathrm { D W C o n v } _ { 1 \times 3 } \left( \mathrm { P a d } _ { l e f t } ( X ) \right) , } \\ & { X _ { r } = \mathrm { D W C o n v } _ { 1 \times 3 } \left( \mathrm { P a d } _ { r i g h t } ( X ) \right) , } \\ & { X _ { u } = \mathrm { D W C o n v } _ { 3 \times 1 } \left( \mathrm { P a d } _ { u p } ( X ) \right) , } \\ & { X _ { d } = \mathrm { D W C o n v } _ { 3 \times 1 } \left( \mathrm { P a d } _ { d o w n } ( X ) \right) . } \end{array}
$$

Where, $\mathrm { P a d } _ { l e f t } , \mathrm { P a d } _ { r i g h t } , \mathrm { P a d } _ { u p } .$ , and $\mathrm { P a d } _ { d o w n }$ denote zero padding applied exclusively to the left, right, top, and bottom spatial boundaries, respectively.

The four directional responses are concatenated along the channel dimension:

$$
X _ { d i r } = \operatorname { C o n c a t } ( X _ { l } , X _ { r } , X _ { u } , X _ { d } ) \in \mathbb { R } ^ { H \times W \times 4 C _ { m } } .\tag{4}
$$

The horizontal branches capture directional responses along the left-right axis, whereas the vertical branches capture responses along the up-down axis. Their asymmetric receptive fields provide direction-aware local information for modeling object boundaries and corner structures.

To suppress background amplification, RPGConv uses a lightweight spatial-channel gate:

$$
G = \sigma \left( \mathrm { D W C o n v _ { 3 \times 3 } } ( \mathbf { C o n v _ { 1 \times 1 } } ( X ) ) + \mathbf { C o n v _ { 1 \times 1 } } ( X ) \right) \in ( 0 , 1 ) ^ { H \times W \times C } ,\tag{5}
$$

where the convolutional projections produce � channels. The enhanced response is

$$
E = G \odot { \mathrm { G N } } \left( { \mathrm { C o n v } } _ { 1 \times 1 } ( X _ { d i r } ) \right) \in \mathbb { R } ^ { H \times W \times C } ,\tag{6}
$$

where $\sigma ( \cdot )$ is the sigmoid function and ⊙ denotes element-wise multiplication. Finally, a learnable channel-wise scale provides a residual update:

$$
C _ { 3 } ^ { o u t } = C _ { 3 } + \beta \odot E , \quad \beta \in \mathbb { R } ^ { C } .\tag{7}
$$

The parameter $\beta$ is initialized to $1 0 ^ { - 3 }$ . During training, it is updated through backpropagation and the optimizer together with the other trainable parameters. The output preserves the spatial resolution and channel dimension of $C _ { 3 }$ before entering the FPN lateral projection.

## 3.3 Reweighted Cross-Scale Feature Fusion

In a standard FPN, each top-down feature is obtained by adding a lateral feature to the upsampled feature from the adjacent level. Equal-weight summation can allow deep semantic responses to dominate shallow spatial details, weakening the boundary cues needed for small-object detection.To address this imbalance, we introduce Reweighted Cross-scale Feature Fusion (RCFF), which learns the relative contribution of the two inputs. Given the lateral feature $L _ { i }$ and the upsampled feature $P _ { i + 1 }$ , the fusion weights are computed as

$$
[ \alpha _ { 0 } , \alpha _ { 1 } ] = \operatorname { s o f t m a x } ( [ w _ { 0 } , w _ { 1 } ] ) ,\tag{8}
$$

where $w _ { 0 }$ and $w _ { 1 }$ are learnable parameters initialized to 0.6 and 0.4, respectively. The softmax operation produces normalized weights that are positive and sum to one. The weighted features are concatenated along the channel dimension:

$$
Z _ { i } = \mathrm { C o n c a t } \left( \alpha _ { 0 } L _ { i } , \alpha _ { 1 } P _ { i + 1 } \right) .\tag{9}
$$

RCFF processes the concatenated feature $Z _ { i }$ through a main branch and a lightweight residual branch. The main branch first projects $Z _ { i }$ to the target channel width using a $1 \times 1$ convolution, Batch Normalization, and SiLU activation. Denoting this sequence as $\mathrm { C B } \mathbf { S } _ { \mathrm { 1 \times 1 } }$ , the projected feature is then processed by three sequential re-parameterizable convolution, denoted as RepConv :

$$
B _ { i } = \mathrm { { C B S } } _ { 1 \times 1 } ( Z _ { i } ) , \quad Z _ { i } ^ { m a i n } = \mathrm { { R e p C o n v } } _ { 3 } ( B _ { i } ) .\tag{10}
$$

![](images/ddb32fa737f2dbc8960d3eba565e497422793fbc6725776dc5d700c8aa7d9af0.jpg)  
Fig. 2. Structure of the re-parameterizable convolution used in RCFF.

The re-parameterizable convolution used in RCFF follows the multi-branch training and single-branch inference. The structure of the block is illustrated in Fig. 2. During training, each RepConv contains parallel $3 \times 3$ and $1 \times 1$ convolutional branches, each followed by Batch Normalization. If $\mathrm { C B } _ { k \times k }$ denotes a $k \times k$ convolution followed by Batch Normalization, one re-parameterizable convolution is written as

$$
\begin{array} { r } { \mathrm { R e p C o n v } ( B _ { i } ) = \mathrm { S i L U } \left( \mathrm { C B } _ { 3 \times 3 } ( B _ { i } ) + \mathrm { C B } _ { 1 \times 1 } ( B _ { i } ) \right) . } \end{array}\tag{11}
$$

During inference, the parallel convolution-Batch Normalization branches can be reparameterized into an equivalent $3 \times 3$ convolution followed by SiLU activation.

The residual branch applies an independent $1 \times 1$ convolution, Batch Normalization, and SiLU activation:

$$
Z _ { i } ^ { r e s } = \mathbf { C B S } _ { 1 \times 1 } ( Z _ { i } ) .\tag{12}
$$

The final RCFF output is obtained by summing the two branches:

$$
P _ { i } = Z _ { i } ^ { m a i n } + Z _ { i } ^ { r e s } .\tag{13}
$$

To control computational overhead, RCFF is applied at the $P _ { 2 } , P _ { 3 }$ and $P _ { 4 }$ levels, where the imbalance between semantic context and fine-grained spatial details has the greatest impact on small-object localization. The remaining FPN levels retain the standard fusion operation.

## 3.4 Residual Convolutional Gated Linear Unit

After RoIAlign, each proposal is represented by a fixed-resolution feature map. For small objects, this representation contains limited spatial evidence and may lose the local structure needed to distinguish foreground from clutter. We therefore insert a Residual Convolutional Gated Linear Unit (RCGLU) between RoIAlign and the Fast R-CNN head.

Let $F ~ \in ~ \mathbb { R } ^ { N \times S \times S \times C }$ denote the pooled RoI features. RCGLU first expands the channel dimension to $2 C _ { h }$ , where

$$
C _ { h } = \operatorname* { m a x } ( \lfloor 2 C \rfloor , 1 ) .\tag{14}
$$

The expanded feature is obtained using a $1 \times 1$ convolution followed by Group Normalization:

$$
\ b { F } _ { z } = \ b { \mathrm { G N } } \left( \ b { \mathrm { C o n v } } _ { 1 \times 1 } ( F ) \right) \in \mathbb { R } ^ { N \times S \times S \times 2 C _ { h } } .\tag{15}
$$

The expanded feature is split evenly along the channel dimension into a gate branch and a value branch:

$$
\begin{array} { r } { ( F _ { g } , F _ { v } ) = \mathrm { S p l i t } _ { c h a n n e l } ( F _ { z } ) , \quad F _ { g } , F _ { v } \in \mathbb { R } ^ { N \times S \times S \times C _ { h } } . } \end{array}\tag{16}
$$

To make the gate spatially aware, the gate branch applies a depthwise $3 \times 3$ convolution, Group Normalization, and GELU activation:

$$
\widetilde { \cal F } _ { g } = \mathrm { G E L U } \left( \mathrm { G N } \left( \mathrm { D W C o n v } _ { 3 \times 3 } ( { \cal F } _ { g } ) \right) \right) .\tag{17}
$$

Same padding is used to preserve the spatial resolution of the RoI feature map. The gated feature is then obtained by element-wise modulation:

$$
F _ { u } = { \widetilde { F } } _ { g } \odot F _ { v } .\tag{18}
$$

The $1 \times 1$ convolution and Group Normalization project the modulated feature back to � channels:

$$
Y = \mathrm { G N } \left( \mathrm { C o n v } _ { 1 \times 1 } ( F _ { u } ) \right) \in \mathbb { R } ^ { N \times S \times S \times C } .\tag{19}
$$

Finally, RCGLU applies a channel-wise residual update:

$$
F _ { o u t } = F + \gamma \odot Y , \quad \gamma \in \mathbb { R } ^ { 1 \times 1 \times 1 \times C } ,\tag{20}
$$

where $\gamma$ is a learnable channel-wise scaling vector initialized to $1 0 ^ { - 2 }$ and broadcast over all proposals and spatial locations. The output $F _ { o u t }$ retains the spatial and channel dimensions of $F$ and is forwarded to the Fast R-CNN head.

## 3.5 Training and Inference

During training, the CLIP backbone remains frozen, while the feature pyramid, RPN, and detection head are jointly optimized with the standard detection objective:

$$
L = L _ { \mathrm { r p n } } ^ { \mathrm { c l s } } + L _ { \mathrm { r p n } } ^ { \mathrm { b b o x } } + L _ { \mathrm { r o i } } ^ { \mathrm { c l s } } + L _ { \mathrm { r o i } } ^ { \mathrm { b b o x } } ,\tag{21}
$$

where $L _ { \mathrm { r p n } } ^ { \mathrm { c l s } }$ and $L _ { \mathrm { r p n } } ^ { \mathrm { b b o x } }$ measure proposal-stage objectness classification and bounding-box regression, respectively, while $L _ { \mathrm { r o i } } ^ { \mathrm { c l s } }$ and $L _ { \mathrm { r o i } } ^ { \mathrm { b b o x } }$ measure RoI-level category classification and bounding-box regression. The RPN classification loss $L _ { \mathrm { r p n } } ^ { \mathrm { c l s } }$ uses binary cross-entropy over positive and negative anchors from all pyramid levels. The RPN regression loss $L _ { \mathrm { r p n } } ^ { \mathrm { b b o x } }$ applies the Huber loss to the coordinate diferences between predicted and target boxes only for positive anchors, with transition threshold $\delta = 0 . 1 1 1$ . For the detection head, the classification loss $L _ { \mathrm { r o i } } ^ { \mathrm { c l s } }$ uses softmax cross-entropy over sampled RoIs, with the background class assigned a weight of 0.9 and foreground classes a weight of 1.0. The box regression loss $\bar { L } _ { \mathrm { r o i } } ^ { \mathrm { b b o x } }$ is class-agnostic and applies the Huber loss only to foreground RoIs; background RoIs are excluded from the regression loss. We apply weight decay of $1 0 ^ { - 3 }$ to trainable parameters, excluding normalization layers and the learnable temperature parameter.

The detection head does not learn a fixed closed-set classifier. Instead, each region feature is L2-normalized and compared with category text embeddings produced by the CLIP text encoder. The cosine similarities are divided by a learnable temperature to produce pre-softmax class scores, allowing the vocabulary to change by substituting the corresponding text embeddings at inference time.

During inference, we additionally use the frozen CLIP branch as an auxiliary classifier. RoI features are projected into the shared vision-language space through the fixed attention-pooling layer and matched against ofline-generated category embeddings. The resulting similarities use a fixed temperature of 0.01. Let $p _ { \mathrm { d e t } , c }$ and $p _ { \mathrm { c l i p } , c }$ denote the detector and CLIP probabilities for category �, respectively. For each foreground category, we compute an unnormalized fused score as

$$
\widetilde { p _ { c } } = p _ { \mathrm { d e t } , c } ^ { 1 - \lambda } \cdot p _ { \mathrm { c l i p } , c } ^ { \lambda } ,\tag{22}
$$

where $\lambda = 0 . 3 5$ for base classes and $\lambda = 0 . 6 5$ for novel classes. Thus, the detector dominates base-category predictions, while the frozen CLIP branch contributes more strongly to novel categories. The background score is taken directly from the detector branch. After renormalization, score thresholding and NMS produce the final detections.

## 4 Experiments

## 4.1 Experimental Setup

Images are resized to 512 pixels with scale jittering in the range [0.1, 2.0]. The feature pyramid uses levels $P _ { 2 } \mathrm { - } P _ { 6 } ,$ , and the RoI output size is $7 \times 7 .$ . The anchor size is 8, and at most 100 detections are retained per image. Category text embeddings are generated ofline from category names and kept fixed throughout training and inference [12–14].

Models are trained in float32 with batch size 4, while evaluation uses batch size 1. We use SGD with momentum and weight decay $1 0 ^ { - 3 }$ , excluding normalization layers and the learnable temperature parameter. The learning rate is linearly warmed up from 0.0032 to 0.02 over the first 100 steps, followed by stepwise decay with milestones at 80%, 90%, and 95% of training and a decay factor of 0.1. The background and foreground classification weights are 0.9 and 1.0, respectively. The fixed CLIP similarity temperature is 0.01, and the number of pre-NMS detections is fixed at 1,000.

## 4.2 Evaluation on Open-Vocabulary Benchmarks

We first evaluate FocusOVD on the COCO [13] and LVIS [12] open-vocabulary benchmarks. For COCO [13], we follow the standard split with 48 base and 17 novel categories. The model is trained using only base-category annotations and evaluated on all 65 categories. We report AP at an IoU threshold of 0.50 for novel categories $( \mathrm { A P _ { n o v e l } } )$ , base categories $( \mathrm { A P _ { b a s e } ) }$ , and all categories $\left( \mathrm { A P _ { a l l } } \right)$ . To further evaluate scale-specific performance, we report $\mathrm { A P _ { S } ^ { n o v e l } , A P _ { M } ^ { n o v e l } }$ , and $\mathsf { A P } _ { \mathrm { L } } ^ { \mathrm { n o v e l } }$ for novel categories, corresponding to object areas smaller than $3 2 ^ { 2 }$ , between $3 2 ^ { 2 }$ and $9 6 ^ { 2 }$ , and larger than $9 6 ^ { 2 }$ , respectively. Table 1 summarizes the open-vocabulary detection results on COCO-OVD.

Table 1. Open-vocabulary detection results on COCO-OVD, including size-specific AP for novel categories.
<table><tr><td>Model</td><td> $\mathsf { A P } _ { \mathrm { n o v e l } }$ </td><td> $\mathrm { \sf A P _ { b a s e } }$ </td><td> $\mathrm { A P _ { a l l } }$ </td><td> $\overline { { \mathrm { A P _ { S } ^ { n o v e l } } } }$ </td><td> $\overline { { \mathbf { A P _ { M } ^ { n o v e l } } } }$ </td><td> $\overline { { \mathrm { A P _ { L } ^ { n o v e l } } } }$ </td></tr><tr><td>ViLD [4]</td><td>27.6</td><td>58.7</td><td>51.3</td><td>14.5</td><td>29.1</td><td>39.3</td></tr><tr><td>RegionCLIP [5]</td><td>31.4 29.4</td><td>57.1</td><td>50.4</td><td>18.2</td><td>33.5</td><td>44.4</td></tr><tr><td>OV-DETR [17]</td><td></td><td>61.0</td><td>52.7</td><td>17.8</td><td>32.6</td><td>41.0</td></tr><tr><td>Detic [18]</td><td>27.8</td><td>47.1</td><td>45.0</td><td>15.7</td><td>30.0</td><td>39.9</td></tr><tr><td>CoDet [10]</td><td>30.2</td><td>52.4</td><td>46.7</td><td>17.5</td><td>32.8</td><td>41.1</td></tr><tr><td>BARON [20]</td><td>34.0</td><td>60.4</td><td>53.5</td><td>20.3</td><td>36.5</td><td>46.3</td></tr><tr><td>CORA [8]</td><td>35.0</td><td>44.2</td><td>43.5</td><td>21.9</td><td>38.0</td><td>47.7</td></tr><tr><td>UniDetector [9]</td><td>35.2</td><td>56.8</td><td>51.2</td><td>22.4</td><td>38.3</td><td>48.5</td></tr><tr><td>SAMP [23]</td><td>34.8</td><td>61.4</td><td>54.2</td><td>21.9</td><td>38.2</td><td>47.1</td></tr><tr><td>FocusOVD</td><td>36.3</td><td>60.9</td><td>54.5</td><td>24.3</td><td>38.9</td><td>48.3</td></tr></table>

As shown in Table 1, FocusOVD achieves 36.3% $\mathsf { A P } _ { \mathrm { n o v e l } }$ <sub>l</sub>, outperforming CORA, UniDetector, and SAMP by 1.3%, 1.1%, and 1.5%, respectively. It also reaches 54.0% $\mathrm { A P _ { a l l } } .$ , ranking second among the compared methods and remaining competitive on base categories. For novel objects across diferent scales, FocusOVD obtains 24.3% $\mathrm { A P _ { S } ^ { n o v e l } }$ 38.9% $\mathrm { { A P _ { M } ^ { n o v e l } } }$ , and 48.3% $\mathsf { A P } _ { \mathrm { L } } ^ { \mathrm { n o v e l } }$ . Compared with UniDetector, FocusOVD improves $\mathsf { A P } _ { \mathbb S } ^ { \mathrm { n o v e l } }$ and $\mathrm { { A P _ { M } ^ { n o v e l } } }$ by 1.9% and $0 . 6 \%$ , respectively, while obtaining a comparable result on large objects. Fig. 3 shows that the largest improvement over UniDetector occurs on small objects, where FocusOVD achieves a gain of 1.9%.

To complement the quantitative comparison, Fig. 4 presents qualitative detection results of CORA, RegionCLIP, UniDetector, and FocusOVD on COCO validation images. The zoomed regions show that FocusOVD can recover several small objects that are missed or poorly localized by the competing methods.

![](images/1c1dac5b0f75bbd320ce4ef280997596272f9801f877ccc94cd9d6e176271b48.jpg)

![](images/2d2784c280f2270d629d48aec5dd2b53797f9d97a189f0c789195b00eb979f6f.jpg)  
Fig. 3. Novel-category AP on COCO-OVD for UniDetector and FocusOVD across diferent object scales.

LVIS [12] provides a more challenging long-tailed setting. We report overall AP together with $\mathbf { A P }$ for rare $( \mathrm { A P _ { r } } ,$ , fewer than 10 training instances), common $( \mathrm { A P _ { c } , }$ 10–100 instances), and frequent $( \mathrm { A P _ { f } } ,$ , more than 100 instances) categories. The rare-category score is particularly informative for evaluating open-vocabulary generalization.

Table 2. Open-vocabulary detection results on LVIS-OVD.
<table><tr><td>Model  $\mathsf { A P } _ { \mathrm { r } }$   $\mathsf { A P _ { c } }$   $\mathrm { A P _ { f } }$  AP</td></tr><tr><td>ViLD [4] 16.1 23.9 32.5 24.9 RegionCLIP [5] 21.3 30.8 36.0 30.7</td></tr><tr><td>OV-DETR [17]17.4 25.0 32.5 26.6</td></tr><tr><td>Detic [18] 17.8 26.3 31.6 26.8</td></tr><tr><td>CoDet [10] 21.8 30.2 34.9 30.6 MEDet [11] 22.2 32.736.9 33.1</td></tr></table>

As reported in Table 2, FocusOVD obtains 33.7% overall AP and the best $\mathsf { A P } _ { \mathrm { r } }$ and $\mathsf { A P _ { f } }$ among the compared methods, reaching 23.9% and 37.1%, respectively. Its $\mathsf { A P _ { c } }$ of 32.6% is within 0.1% of the best result. The consistent performance across category frequencies demonstrates that FocusOVD improves recognition of rare categories without sacrificing detection accuracy on common and frequent categories.

## 4.3 Cross-Dataset Zero-Shot Transfer

We train FocusOVD on the LVIS base categories and evaluate it on the COCO and Objects365 v2 [14] validation sets to assess cross-dataset zero-shot transfer. FocusOVD uses the CLIP ResNet-50 backbone in Table 3. We report AP, $\mathsf { A P 5 0 }$ , and $\mathsf { A P 7 5 }$ , where AP averages precision over IoU thresholds from 0.50 to 0.95, while $\mathsf { A P } _ { 5 0 }$ and $\mathsf { A P 7 5 }$ measure performance at IoU thresholds of 0.50 and 0.75, respectively.

![](images/1192e6caf93325ab4a1967e95d7b5e45f2dace0608b6aff445c11b737a7ce412.jpg)  
Fig. 4. Qualitative comparison of CORA, RegionCLIP, UniDetector, and FocusOVD on COCO validation images.

As shown in Table 3, FocusOVD achieves the best performance on COCO, with 37.0% AP, 56.0% $\mathsf { A P } _ { 5 0 }$ , and 40.1% $\mathsf { A P } _ { 7 5 }$ . Its AP exceeds ViLD, F-VLM, DetPro, BARON, and UniDetector by 0.4%, 4.5%, 2.1%, 0.8%, and 0.6%, respectively. On Objects365 v2, FocusOVD obtains 13.8% AP, and achieves the highest $\mathsf { A P 7 5 }$ of 15.0%. Although its $\mathsf { A P 5 0 }$ of 21.1% is slightly below the best result, the overall results demonstrate strong transfer across datasets with diferent category distributions and visual characteristics.

Table 3. Cross-dataset zero-shot transfer results on COCO and Objects365 v2.
<table><tr><td rowspan="2">Model</td><td>COCO Objects365</td></tr><tr><td> $\mathbf { v } 2$  AP AP50 AP75 AP AP50  $\overline { { \mathrm { A P } _ { 7 5 } } }$ </td></tr><tr><td>ViLD [4]</td><td>36.6 55.6 39.8 11.8 18.2 12.6</td></tr><tr><td>F-VLM [7]</td><td>32.5 53.1 34.6 11.9 19.2 12.6</td></tr><tr><td>DetPro [19]</td><td>34.9 53.8 34.7 12.1 18.8 12.9</td></tr><tr><td>BARON [20]</td><td>36.2 55.7 38.1 13.6 21.0 14.5</td></tr><tr><td>UniDetector [9] 36.4 54.9 FocusOVD</td><td>38.5 13.3 21.4 14.7 37.0 56.0 40.1 13.8 21.1 15.0</td></tr></table>

## 4.4 Ablation Study

We train FocusOVD on the LVIS base categories and evaluate it on the COCO validation set to isolate the individual and combined efects of RCFF, RPGConv, and RCGLU. All ablation configurations use the CLIP ResNet-50×4 backbone. Table 4 reports the results of the component ablation study; values in parentheses denote absolute gains over the baseline without the proposed modules.

Table 4. Component ablation results.
<table><tr><td>#</td><td>RCFF</td><td>RPG</td><td>RCGLU</td><td>AP</td><td> $\mathrm { A P } _ { 5 0 }$ </td><td> $\mathsf { A P } _ { 7 5 }$ </td><td> $\mathsf { A P } _ { S }$ </td><td> $\mathsf { A P } _ { M }$ </td><td> $\mathrm { A P } _ { L }$ </td></tr><tr><td>1</td><td>一</td><td>一</td><td></td><td>35.5</td><td>56.9</td><td>38.2</td><td>21.6</td><td>39.0</td><td>52.8</td></tr><tr><td>2</td><td>√</td><td></td><td></td><td>38.8 (+3.3)</td><td>60.4 (+3.5)</td><td>41.2 (+3.0)</td><td>23.2 (+1.6)</td><td>42.3 (+3.3)</td><td>56.9 (+4.1)</td></tr><tr><td>3</td><td>√</td><td>√</td><td></td><td>39.5 (+4.0)</td><td>60.8 (+3.9)</td><td>42.8 (+4.6)</td><td>24.7 (+3.1)</td><td>43.4 (+4.4)</td><td>57.5 (+4.7)</td></tr><tr><td>4</td><td>√</td><td>一</td><td>√</td><td>39.2 (+3.7)</td><td>61.6 (+4.7)</td><td>42.4 (+4.2)</td><td>25.1 (+3.5)</td><td>43.7 (+4.7)</td><td>57.2 (+4.4)</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>39.9 (+4.4)</td><td>62.1 (+5.2)</td><td>44.6 (+6.4)</td><td>26.4 (+4.8)</td><td>44.5 (+5.5)</td><td>57.9 (+5.1)</td></tr></table>

The baseline achieves 35.5% AP and 21.6% AP<sub>�</sub>. Adding RCFF increases these scores to 38.8% and 23.2%, corresponding to gains of 3.3% and 1.6%. Adding RPGConv to RCFF yields 39.5% AP and 24.7% $\mathsf { A P } _ { S }$ , corresponding to gains of 4.0% and 3.1% over the baseline. Replacing RPGConv with RCGLU in the RCFF-based configuration yields 39.2% AP and 25.1% $\mathsf { A P } _ { S }$ , corresponding to gains of 3.7% and 3.5% over the baseline, respectively. With all three modules enabled, FocusOVD achieves 39.9% AP and 26.4% $\mathsf { A P } _ { S }$ , improving over the baseline by 4.4% and 4.8%. It also gains 5.2%, 6.4%, 5.5%, and 5.1% in $\mathrm { A P } _ { 5 0 } , \mathrm { A P } _ { 7 5 } , \mathrm { A P } _ { M }$ , and $\mathsf { A P } _ { L }$ , respectively. The progressive improvements indicate that the three modules provide complementary benefits across cross-scale fusion, local geometric modeling, and RoI feature refinement.

## 5 Conclusion

In this paper we introduced FocusOVD, an open-vocabulary detector designed for small objects. It enhances frozen vision-language features through RPGConv for local geometric modeling, RCFF for cross-scale fusion, and RCGLU for region-level refinement. Experiments on COCO, LVIS, and Objects365 demonstrate competitive performance in open-vocabulary detection and cross-dataset transfer, particularly for small objects. Future work will extend this feature-refinement framework to open-vocabulary instance segmentation and other dense prediction tasks.

## References

1. Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks. TPAMI (2017).

2. Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: CVPR (2017).

3. Radford, A., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021).

4. Gu, X., Lin, T.-Y., Kuo, W., Cui, Y.: Open-vocabulary object detection via vision and language knowledge distillation. In: ICLR (2022).

5. Zhong, Y., et al.: RegionCLIP: Region-based language-image pretraining. In: CVPR (2022).

6. Li, L.H., et al.: Grounded language-image pre-training. In: CVPR (2022).

7. Kuo, W., et al.: F-VLM: Open-vocabulary object detection upon frozen vision and language models. In: ICLR (2023).

8. Wu, X., Zhu, F., Zhao, R., Li, H.: CORA: Adapting CLIP for open-vocabulary detection with region prompting and anchor pre-matching. In: CVPR (2023).

9. Wang, Z., et al.: Detecting everything in the open world: Towards universal object detection. In: CVPR (2023).

10. Ma, C., et al.: CoDet: Co-occurrence guided region-word alignment for open-vocabulary object detection. In: NeurIPS (2023).

11. Chen, P., et al.: Open-vocabulary object detection with proposal mining and prediction equalization. arXiv:2206.11134 (2022).

12. Gupta, A., Dollár, P., Girshick, R.: LVIS: A dataset for large vocabulary instance segmentation. In: CVPR (2019).

13. Lin, T.-Y., et al.: Microsoft COCO: Common objects in context. In: ECCV (2014).

14. Shao, S., et al.: Objects365: A large-scale, high-quality dataset for object detection. In: ICCV (2019).

15. Tan, M., Pang, R., Le, Q.V.: EficientDet: Scalable and eficient object detection. In: CVPR (2020).

16. Girshick, R.: Fast R-CNN. In: ICCV (2015).

17. Zang, Y., Li, W., Zhou, K., Huang, C., Loy, C.C.: Open-vocabulary DETR with conditional matching. In: Proc. ECCV, pp. 106–122 (2022).

18. Zhou, X., Girdhar, R., Joulin, A., Krähenbühl, P., Misra, I.: Detecting twenty-thousand classes using image-level supervision. In: Proc. ECCV, pp. 350–368 (2022).

19. Du, Y., Wei, F., Zhang, Z., Shi, M., Gao, Y., Li, G.: Learning to prompt for open-vocabulary object detection with vision-language model. In: Proc. CVPR, pp. 14084–14093 (2022).

20. Wu, S., Zhang, W., Jin, S., Liu, W., Loy, C.C.: Aligning bag of regions for open-vocabulary object detection. In: Proc. CVPR, pp. 15481–15490 (2023).

21. Ma, Z., Luo, G., Gao, J., Li, L., Chen, Y., Wang, S., Zhang, C., Hu, W.: Open-vocabulary one-stage detection with hierarchical visual-language knowledge distillation. In: Proc. CVPR, pp. 14054–14063 (2022).

22. Liu, S., Qi, L., Qin, H., Shi, J., Jia, J.: Path aggregation network for instance segmentation. In: Proc. CVPR (2018).

23. Zhao, X., Liu, X., Wang, D., Gao, Y., Liu, Z.: Scene-adaptive and region-aware multi-modal prompt for open vocabulary object detection. In: Proc. CVPR, pp. 16741–16750 (2024).

24. Singh, B., Davis, L.S.: An analysis of scale invariance in object detection—SNIP. In: CVPR (2018).

25. Li, Y., Chen, Y., Wang, N., Zhang, Z.: Scale-aware trident networks for object detection. In: ICCV (2019).

26. Zhang, S., et al.: PyramidBox: A context-assisted single shot face detector. In: ECCV (2018).

27. Li, Y., Li, Y., Zeng, Q.-S., Wang, W., Hou, Q., Cheng, M.-M.: Unbiased region-language alignment for open-vocabulary dense prediction. In: ICCV, pp. 23795–23805 (2025).

28. Li, Y., Niu, J., Ren, T.: Benefit from seen: Enhancing open-vocabulary object detection by bridging visual and textual co-occurrence knowledge. In: ICCV, pp. 22110–22119 (2025).