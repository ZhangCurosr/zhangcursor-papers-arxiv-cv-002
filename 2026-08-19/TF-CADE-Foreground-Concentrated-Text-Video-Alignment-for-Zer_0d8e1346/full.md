# TF-CADE: Foreground-Concentrated Text-Video Alignment for Zero-Shot Temporal Action Detection

Yearang Lee Ho-Joong Kim Seong-Whan Lee<sup>∗</sup>

Dept. of Artificial Intelligence, Korea University, Seoul, Korea {yr lee, hojoong kim, sw.lee}@korea.ac.kr

## Abstract

Zero-Shot Temporal Action Detection (ZSTAD) aims to localize and recognize action instances from unseen action categories in untrimmed videos. Although existing methods have shown effectiveness by advancing architectural text-video alignment, they still struggle with capturing semantic distinctions between action classes, resulting in textirrelevant predictions. To address this issue, we propose a Text-Foreground Concentrated Alignment for zero-shot temporal action DEtector (TF-CADE) that explicitly aligns textual information with action-relevant foreground regions. Specifically, we introduce Action Concentrate Aggregation (ACA), which extracts action concentrate scores to aggregate temporally informative video segments into a foregroundweighted video embedding. This foreground concentrated alignment enhances the semantic consistency between text and video features and improves inter-class discriminability. In addition, a Certainty-based Confidence Re-weighting (CCR) strategy refines per-snippet confidence scores by leveraging foreground-aware similarity, effectively suppressing irrelevant action classes during inference. Extensive evaluations show that our TF-CADE not only achieves state-of-theart performance under in-distribution settings but also excels in cross-dataset generalization to unseen action classes.

## 1. Introduction

Zero-Shot Temporal Action Detection (ZSTAD) aims to temporally localize and classify action instances from novel categories that are not seen during training time in untrimmed videos. With the advances of large-scale pre-trained visuallanguage models (e.g., CLIP [23] and ALIGN [9]), existing ZSTAD methods [6, 7, 13, 15, 16, 20, 22, 24] have leveraged their generalization capabilities by aligning the text features with the relevant temporal regions in the video. The existing ZSTAD methods are categorized into two types: foreground-based and foreground-free methods.

![](images/46bce714a33efcabfb0c686845db857a9eccc3ca72e85b482b450df5f5910524.jpg)

(a) Ti-FAD [13]  
![](images/4c61d64bc997e864566306df4ea1c233a8665daec7c3cbb80a995bef6f9ed7ec.jpg)  
(b) TF-CADE (Ours)  
Figure 1. Class confidence scores under different textual inputs. (a) Ti-FAD [13] produces nearly identical confidence distributions whether the input text is the correct action label (“ThrowDiscus”) or an unrelated word (“XYZ”). In contrast, our (b) TF-CADE generates clearly differentiated confidence scores.

The foreground-based methods [7, 10, 15, 20, 22] first extract foreground proposals to identify action candidate segments, and subsequently align these foreground proposals with text features. The pre-extracted foreground proposals are obtained independently without considering the text features, limiting the integration of textual representations. To overcome this limitation, existing foreground-free methods [6, 13, 16, 24] incorporate text features directly into video features throughout the entire detection process. Instead of relying on pre-extracted foreground proposals, the foreground-free methods leverage bi-directional crossattention between text and video features, generating mutually enhanced representations. These mutually enhanced representations are subsequently utilized to produce class confidence scores during detection process. Among them, Ti-FAD [13] introduces a cross-modal adaptation, in which text-infused bi-directional cross attention module aligns text with video features containing both foreground and background regions, producing text-aware video and video-aware text representations. However, despite such architectural improvements, the existing foreground-free methods such as Ti-FAD [13] still struggle to incorporate text features into the detection process, producing text-irrelevant predictions.

To demonstrate this issue, we analyze how the detector responds when conditioned on two distinct types of text inputs: a ground-truth action label and a meaningless word that has no semantic relation to any action category. Fig. 1 compares the resulting class confidence scores computed via text-video similarity between the foreground-free method Ti-FAD [13] and our method. As shown in Fig. 1 (a), Ti-FAD [13] produces almost identical confidence distributions across all time steps, regardless of whether the input is the correct action name “ThrowDiscus” or an irrelevant word such as “XYZ”. This invariance of the confidence scores indicates that the existing foreground-free method fails to incorporate meaningful textual information, and struggles to distinguish between semantically valid and invalid text inputs, demonstrating that the prediction is predominantly driven by video features. This text-irrelevant prediction indicates that the foreground-free method does not incorporate meaningful guidance from textual inputs.

To understand why textual cues fail to influence the prediction, we assume that the cross-modal adaptation in foreground-free method aligns text with video features that contain both action-relevant foreground and text-irrelevant background regions. Since background regions often dominate the visual representation in untrimmed videos, this joint alignment causes the updated text features to drift toward background-biased visual patterns. To examine this hypothesis, we analyze how the video-aware text features produced by the cross-modal adaptation in Ti-FAD [13] represent when background influence is removed. Fig. 2 visualizes the cosine similarity matrices of the resulting text features under two controlled conditions. As shown in Fig. 2 (a), the first condition reflects the standard foregroundfree setting where all video regions are used for cross-modal adaptation. The resulting text features form uniformly high similarities across different action names, indicating that the model fails to encode class-specific textual cues. In contrast, Fig. 2 (b) shows the outcome when we construct a controlled variant that feeds only the ground-truth foreground regions, removing all background segments while keeping the overall detector architecture unchanged. This setup serves purely as a diagnostic experiment that removes the role of background during the alignment step. Under this foreground-only condition, the resulting text features become distinctly more separable across classes compared to Fig. 2(a). This comparison shows that background information interferes with the alignment between text features and action-related visual patterns. Consequently, these observations motivate the need for explicitly aligning text with action-relevant foreground regions, preventing background-dominated drift.

![](images/a8d66fac7d17eee13d59b2ce2a6d4a8c31e3cb4b70668e1c264ffb3324ca03b7.jpg)  
(a) w/ foreground & background

![](images/3655cf135579f5046090309d3d39f2c8cefc1ffd0830405c916998c9100f60a6.jpg)  
(b) w/ foreground only  
Figure 2. Cosine similarity heatmap of the visual-aware text features produced by the cross-modal adaptation in Ti-FAD [13] across various action classes on THUMOS14 [8]. (a) With video features including both foreground and background regions, and (b) with only foreground regions used for cross-modal adaptation.

In this paper, we propose a novel Text-Foreground Concentrated Alignment for zero-shot temporal action DEtector (TF-CADE), which explicitly aligns textual information with foreground regions that are semantically relevant to the actions. To achieve this, we first introduce an Action Concentrate Aggregation (ACA) method that extracts foregroundweighted video embedding by aggregating action concentrate scores over time, focusing on the text-relevant video regions. ACA employs a Gaussian smoothing mechanism to filter continuous action segments, thereby generating foreground-weighted video embedding that emphasizes temporally consistent action regions. This foreground-weighted embedding is then used to produce a foreground-based videolevel similarity score, which represents the overall action confidence of the video. During training, this foregroundweighted video embedding is aligned with its corresponding ground-truth text embedding. By selectively aggregating action-relevant regions, this alignment process enables the model to align text with action-relevant foreground regions. Additionally, we introduce a Certainty-based Confidence Reweighting (CCR) strategy, which serves as a video-level prior during inference. It applies the foreground-based video-level similarity scores to refine per-snippet classification confidence scores and suppress irrelevant action classes. Extensive experiments on standard in-distribution ZSTAD results show that TF-CADE outperforms state-of-the-art methods on THUMOS14 and ActivityNet v1.3. Furthermore, our TF-CADE achieves superior performance in cross-dataset evaluation, demonstrating its strong generalization and zeroshot capability.

In summary, our main contributions are as follows:

• We identify the key limitation of existing foreground-free ZSTAD models, which produce text-irrelevant predictions due to misalignment between text and video features.

• We propose TF-CADE with ACA to enhance video-text alignment between action-relevant segments and texts, and CCR to suppress confusing classes during inference.

• Extensive experiments across in-distribution and crossdataset settings show TF-CADE achieves state-of-the-art performance on diverse datasets.

## 2. Related Work

Vision-language Models. CoCa [31], CLIP [23], and ALIGN [9] are large-scale vision-language models trained with contrastive learning on massive image-text pairs, aiming to learn generalizable representations for open-set recognition tasks. These models align visual and textual modalities in a shared embedding space, enabling zero-shot transfer to a wide range of downstream tasks. Recent studies [11, 21, 30] have extended these models to video domains, demonstrating their strong potential for zero-shot temporal action understanding. In this work, we employ the visual and text encoders from CoCa, as well as the text encoder from CLIP, to extract the text features.

Temporal Action Detection. Existing temporal action detection methods [12, 17, 25, 32] aim to localize and classify actions in untrimmed videos. Temporal action detection is categorized into anchor-based and anchor-free approaches. Anchor-based methods rely on predefined temporal anchors or proposals to predict action boundaries and categories, which limits flexibility and increases computational cost due to dense anchor sampling. In contrast, anchor-free detectors directly predict action boundaries and confidence scores from time steps. Among them, ActionFormer [32] introduce anchor-free detector, which directly predict action boundaries and classification scores from temporal feature sequences without the predefined temporal anchors. In this work, following the previous works [13, 15], we adopt the anchor-free detector as our base architecture.

Zero-shot Temporal Action Detection. ZSTAD aims to detect and classify previously unseen action classes in untrimmed videos. Existing ZSTAD approaches can be categorized into two types: foreground-based [7, 10, 15, 20, 22] and foreground-free methods [1, 6, 13, 16, 24]. The foreground-based methods first identify foreground proposals without incorporating text features, and subsequently align these pre-extracted proposals with text embeddings. STALE [20] adopts a masking strategy to extract foreground proposals but does not incorporate text features in the extraction process. Ti-FAD [13] introduces a foreground-free framework that jointly aligns video and text features via textinfused attention. Although Ti-FAD [13] achieves strong performance through architectural improvements, they still struggle to integrate text features into the detection process. Recently, T3AL [16] proposes a training-free method that uses a text-guided region suppression mechanism to align text inputs with meaningful video features. However, T3AL relies on an additional captioning model (e.g., CoCa [31]) to generate sentence-level descriptions, making the pipeline dependent on external language models. In contrast, our approach focuses on explicitly aligning meaningful visual features with text embeddings without relying on any captioning model or auxiliary classification scores (e.g., from a video-level classifier such as UntrimmedNets [28]).

## 3. Preliminary

## 3.1. Problem Setting

Let $X = \{ x _ { t } \} _ { t = 1 } ^ { T _ { 0 } }$ denote the sequence of snippet-level features from an untrimmed video, where $T _ { 0 }$ is the number of snippets that corresponds to a few sequence of consecutive frames. These snippets are processed using a pretrained visual backbone (e.g., I3D [4], InternVideo2 [29]). Each video is annotated with $N _ { a }$ action instances ${ \mathcal { A } } =$ $\{ ( s _ { a } , e _ { a } , y _ { a } ) \} _ { a = 1 } ^ { N _ { a } }$ , where $s _ { a }$ and $e _ { a }$ denote the start and end times of the a -th action instance, and $y _ { a } \in \mathcal { C }$ denotes its class name from the set of all action names \mathcal {C} , with $| { \mathcal { C } } | = N _ { c }$ . In the zero-shot setting, the training and test action categories, denoted as $S _ { \mathrm { t r a i n } }$ and $S _ { \mathrm { t e s t } } .$ , are disjoint, $\mathrm { i . e . , } S _ { \mathrm { t r a i n } } \cap S _ { \mathrm { t e s t } } = \emptyset$ ZSTAD addresses the problem of detecting actions from unseen categories that do not appear in the training data.

## 3.2. Cross-modal Adaptation Baseline

Ti-FAD [13] introduces a cross-modal adaptation baseline designed to enhance the alignment between text and video features. The snippet-level features X are projected into initial video embeddings $v ^ { ( 0 ) } = \{ v _ { t } ^ { ( 0 ) } \} _ { t = 1 } ^ { T _ { 0 } } \in \mathbb { R } ^ { T _ { 0 } \times D }$ via 1D convolution layers, where D is the embedding dimension of the detector. The initial text embeddings $c ^ { ( 0 ) } =$ $\{ c _ { n } ^ { ( 0 ) } \} _ { n = 1 } ^ { N _ { c } } \in \mathbb { R } ^ { N _ { c } \times D }$ are obtained using a pre-trained text encoder such as CLIP [23]. These initial video $v ^ { ( 0 ) }$ and text features $c ^ { ( 0 ) }$ are updated through an encoder module $\mathrm { E n c o d e r } ^ { ( l ) } ( \cdot , \cdot )$ , which consists of self-attention layers for video and text, cross-attention between the two modalities, and feed-forward networks for each modality, applied at each layer l. After the self-attention, the video features are temporally downsampled to produce multi-scale representations $v ^ { ( l ) } = \{ v _ { t } ^ { ( l ) } \} _ { t = 1 } ^ { T _ { l } }$ , where $T _ { l } = T _ { l - 1 } / 2$ is the temporal length at l. Meanwhile, the text features are progressively updated at each level as $c ^ { ( l ) } = \{ c _ { n } ^ { ( l ) } \} _ { n = 1 } ^ { N _ { c } }$ . At each level, cross-attention is performed with video features as queries to attend to the text features, and the outputs are refined via feed-forward layers. The l-th encoded features are represented as follows:

$$
\boldsymbol v ^ { ( l ) } , \boldsymbol c ^ { ( l ) } = \mathrm { E n c o d e r } ^ { ( l ) } ( \boldsymbol v ^ { ( l - 1 ) } , \boldsymbol c ^ { ( l - 1 ) } ) \quad l = 1 , \dots , L ,\tag{1}
$$

where L denotes the total number of pyramid levels.

## 4. TF-CADE

## 4.1. Overview

In this section, we describe our TF-CADE, which aligns text features with foreground-concentrated visual features to mitigate the influence of background regions containing action-irrelevant visual patterns. Built upon a cross-modal adaptation baseline following Ti-FAD [13], our TF-CADE is composed of two main components: (1) Action Concentrate

![](images/62f4f8629bfc28adb8568f65584c8e340bdeb9a94dcd2631142cf38b4091b048.jpg)  
Figure 3. Overview of TF-CADE. (a) Overall framework of the proposed TF-CADE, which builds upon the cross-modal adaptation baseline and introduces text-foreground concentrated alignment to explicitly associate textual information with action-relevant visual regions. (b) Illustration of the text-foreground concentrated alignment method at the l -th layer, which consists of ACA and CCR. ACA computes temporal action certainty scores and generates a foreground-weighted video embedding for foreground-text alignment during training, while CCR refines per-snippet classification scores using the foreground-based video-level similarity score during inference.

Aggregation (ACA), which computes temporal action certainty scores and aggregates video features into a foregroundweighted video embedding for text-foreground alignment; and (2) Certainty-based Confidence Re-weighting (CCR), which refines snippet-level classification logits during inference by incorporating the foreground-based video-level similarity scores, thereby suppressing irrelevant action classes. An overview of TF-CADE is illustrated in Fig. 3.

## 4.2. Action Concentrate Aggregation

To align text with foreground visual features, we introduce ACA: (1) obtaining a temporal action certainty map applying smoothing strategy with Gaussian filter; (2) producing a video-level similarity score aggregating video features weighted by foreground information and comparing them with candidate class embeddings.

Temporal Action Certainty Map. To obtain action-relevant foreground visual features, we construct temporal action certainty map. We utilize the multi-level video features $v ^ { ( l ) }$ and text features $c ^ { ( l ) }$ at each level $l = 1 , \ldots , L ,$ of the Encoder(·) output. We obtain a video-text confidence score map $P _ { \mathrm { c l s } }$ via similarity between video and text features:

$$
P _ { \mathrm { c l s } } = v ^ { ( l ) } \cdot { c ^ { ( l ) } } ^ { \top } \in \mathbb { R } ^ { T _ { l } \times N _ { c } } .\tag{2}
$$

To obtain an action certainty score, we first take the maximum similarity across the class dimension and apply a softmax function as follows:

$$
m _ { \mathrm { m a x } } ^ { ( l ) } = \mathrm { s o f t m a x } \left( \operatorname* { m a x } _ { N _ { c } } \left( P _ { \mathrm { c l s } } \right) \right) \in \mathbb { R } ^ { T _ { l } } ,\tag{3}
$$

where $m _ { \mathrm { m a x } } ^ { ( l ) }$ represents the initial action certainty at level l. $m _ { \operatorname* { m a x } } ^ { ( l ) }$ is sharply concentrated on the most salient action frames via the softmax function, representing the peak occurrence of the action.

To aggregate the full temporal context over the entire foreground action segment, we apply a smoothing method to $m _ { \operatorname* { m a x } } ^ { ( l ) }$ . Specifically, we generate a smoothed action certainty map $m _ { \mathrm { { f i l t e r } } } ^ { ( l ) }$ by applying Gaussian kernel $G ( \sigma )$ along the temporal dimension $T _ { l }$ as follows:

$$
m _ { \mathrm { f i l t e r } } ^ { ( l ) } = m _ { \mathrm { m a x } } ^ { ( l ) } \circledast G ( \sigma ) \in \mathbb { R } ^ { T _ { l } } ,\tag{4}
$$

where ⊛ represents the 1D convolution operation, and σ denotes the standard deviation of the Gaussian kernel that controls the degree of temporal smoothing. This smoothing process reduces impact of noise and overly sharp peaks, promoting more stable temporal pooling. The final temporal action certainty map $m ^ { ( \bar { l } ) }$ is obtained by combining the initial sharp saliency and the smoothed contextual saliency:

$$
m ^ { ( l ) } = m _ { \mathrm { m a x } } ^ { ( l ) } + m _ { \mathrm { f i l t e r } } ^ { ( l ) } .\tag{5}
$$

We then normalize $m ^ { ( l ) }$ along the temporal dimension by dividing it by the sum of all time-step values, ensuring a consistent scale across the sequence.

Foreground-weighted Video Embedding. Utilizing the action certainty $\mathbf { \Omega } _ { m } ^ { \bigcup } ( l )$ , we compute the foreground-weighted video embedding $v _ { \mathrm { f g } } ^ { ( l ) }$ from the video features via soft aggregation as follows:

$$
v _ { \mathrm { f g } } ^ { ( l ) } = \sum _ { t = 1 } ^ { T _ { l } } m _ { t } ^ { ( l ) } \odot v _ { t } ^ { ( l ) } \in \mathbb { R } ^ { D } ,\tag{6}
$$

where ⊙ represents the element-wise product and $v ^ { ( l ) }$ denotes l-th video features produced by cross-modal adaptation module. We then calculate the foreground-based video-level score using the foreground-weighted video embedding $v _ { \mathrm { f g } } ^ { ( l ) }$ and text embedding $c _ { n } ^ { ( l ) }$ . The temporal foreground-based video-level score $S _ { \mathrm { f g } } ^ { ( n ) }$ for the n -th class is calculated by averaging the similarities across all L levels:

$$
S _ { \mathrm { f g } } ^ { ( n ) } = \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \sin ( v _ { \mathrm { f g } } ^ { ( l ) } , c _ { n } ^ { ( l ) } ) \quad n = 1 , \dots , N _ { c } ,\tag{7}
$$

where sim(·, ·) denotes the cosine similarity. We denote the final similarity vector as $S _ { \mathrm { f g } } = \{ S _ { \mathrm { f g } } ^ { ( n ) } \} _ { n = 1 } ^ { N _ { c } }$ , which represents the foreground-based video-level similarity scores used for text-foreground concentrated alignment.

## 4.3. Certainty-based Confidence Re-weighting

In the standard ZSTAD inference pipeline, the video-text confidence score map defined in Eq. (2) is passed through a sigmoid function to produce per-snippet classification scores. However, this snippet-level prediction often leads to overactivation of visually similar but semantically irrelevant classes. To suppress confusing classes that are semantically similar to the target class, we introduce CCR that refines the confidence distribution using a video-level prior. CCR leverages the foreground-based video-level similarity score $S _ { \mathrm { f g } } .$ obtained via action concentrate aggregation in Eq. (7). We apply a softmax function to $S _ { \mathrm { f g } }$ to estimate the likelihood of each class being present in the video. We then re-weight the snippet-wise class confidence map by performing elementwise multiplication with $S _ { \mathrm { f g } }$ as prior:

$$
\tilde { P } _ { \mathrm { c l s } } = \sqrt { \mathrm { s i g m o i d } ( P _ { \mathrm { c l s } } ) \odot \mathrm { s o f t m a x } ( S _ { \mathrm { f g } } ) } \in \mathbb { R } ^ { T _ { l } \times N _ { c } } ,\tag{8}
$$

where $\tilde { P } _ { \mathrm { c l s } }$ represents final re-weighted class confidence score map and ⊙ represents the element-wise product. This re-weighting emphasizes action-relevant classes while suppressing irrelevant ones.

## 4.4. Training and Inference

Training. The classification loss is defined as:

$$
\mathcal { L } _ { c l s } = \mathcal { L } _ { s n i p p e t } + \mathcal { L } _ { v i d e o } ,\tag{9}
$$

where $\mathcal { L } _ { s n i p p e t }$ supervises snippet-level classification based on the class confidence map in Eq. (2), and $\mathcal { L } _ { v i d e o }$ aligns the foreground-based video-level similarity score $S _ { \mathrm { f g } }$ with the corresponding action class. Both losses are computed using the focal loss [18]. Following the previous methods [13, 32], we define the total objective function described as: ${ \mathcal { L } } =$ $\mathcal { L } _ { c l s } + \mathcal { L } _ { l o c } + \mathcal { L } _ { a n }$ , where localization loss $\mathcal { L } _ { l o c }$ regresses action boundaries using the DIoU loss [34], and $\mathcal { L } _ { a n }$ is actioness loss [32] computed using the focal loss [18].

Inference. For each time step t , the model predicts $( \hat { s } , \hat { e } , \hat { p } _ { c l s } )$ , where sˆ and eˆ represent the estimated start and end points of action instance, and $\hat { p } _ { c l s }$ denotes the corresponding action confidence score by applying the argmax over the final filtered class confidence score map $\tilde { P } _ { \mathrm { c l s } }$ in Eq. (8). Subsequently, we apply Soft-NMS [3] to suppress redundant proposals and retain the most confident detections.

## 5. Experiments

## 5.1. Experiment Settings

Datasets. We use three datasets: THUMOS14 [8], ActivityNet v1.3 [5], and HACS-Segment [33]. THUMOS14 contains 20 action classes related to sports across 200 training and 213 test videos. ActivityNet v1.3 includes 200 action classes covering daily actions and contains 19,994 videos. HACS-Segment is a large-scale human activities dataset consisting of 50,000 videos across 200 action classes. We consider split settings for zero-shot scenario: 50%-50% setting (training on 50% of the classes and testing on the remaining 50%) and 75%-25% setting (training on 75% of the classes and testing on the remaining 25%) following EffPrompt [10].

Evaluation Protocol and Comparison Methods. To ensure a strictly zero-shot evaluation for ZSTAD, we focus our comparison on models that do not rely on external classification scores. The external classification scores are commonly used as a post-processing step, obtained from a supervised classifier such as UntrimmedNet [28], but cannot address novel classes because they are trained in closed-set supervised setting. Although most existing ZSTAD methods report impressive results, they rely on such external classifiers, introducing additional supervision that violates zero-shot assumption. Therefore, we select following models that have publicly available implementations, removing only the external classification post-processing without modifying any other components: a foreground-based method STALE [20]; training-free method T3AL [16]; and a stateof-the-art foreground-free method Ti-FAD [13].

Evaluation Metric. We report mean Average Precision (mAP) as an evaluation metric, calculating mAP at different temporal intersection over union (tIoU) thresholds. For THU-MOS14, the tIoU thresholds are set at intervals of 0.1 from 0.3 to 0.7, while for ActivityNet v1.3 and HACS-Segment, they range from 0.5 to 0.95 at intervals of 0.05.

Implementation Details. We employ I3D [4], VideoMAE [26] and CoCA [31] as our video encoders to extract visual features following the existing methods [10, 13, 16, 20]. For THUMOS14, video features are extracted utilizing a sliding window of 16 frames with a stride of 4, while for ActivityNet v1.3 and HACS-Segment, we extract 16 frames as a segment with a stride of 16. For the text encoding, we employ the frozen pre-trained text encoders, such as CLIP [23] and CoCA [31]. We follow the prompt design of Ti-FAD [13], where the input prompt consists solely of the action class name. We train our model for 25 epochs on THUMOS14, and for 15 epochs on ActivityNet v1.3 and HACS-Segment using Adam with 5 epochs of linear warmup. The initial learning rate is 0.0001, updated with multi-step decay for THUMOS14 and cosine annealing [19] for ActivityNet v1.3 and HACS-Segment, respectively. All experiments are performed on a single NVIDIA A100 GPU.

Table 1. Performance comparison with the state-of-the-art methods on THUMOS14 and ActivityNet v1.3 under in-distribution setting. The average mAP is calculated to evaluate performance at different tIoU thresholds: [0.3:0.1:0.7] for THUMOS14 and [0.5:0.05:0.95] for ActivityNet v1.3. <sup>†</sup> indicates that the results are produced utilizing additional text descriptions (e.g., sentence-level captions by CoCa [31]).
<table><tr><td rowspan="2">Train Split</td><td rowspan="2">Model</td><td colspan="2">Feature</td><td colspan="6">THUMOS14</td><td colspan="4">ActivityNet v1.3</td></tr><tr><td>Video</td><td>Text</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>Avg.</td><td>0.5</td><td>0.75</td><td>0.95</td><td>Avg.</td></tr><tr><td rowspan="10">50% Seen 50% Unseen</td><td>EffPrompt [10]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>37.2</td><td>29.6</td><td>21.6</td><td>14.0</td><td>7.2</td><td>21.9</td><td>32.0</td><td>19.3</td><td>2.9</td><td>19.6</td></tr><tr><td>STALE [20]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>38.3</td><td>30.7</td><td>21.2</td><td>13.8</td><td>7.0</td><td>22.2</td><td>32.1</td><td>20.7</td><td>5.9</td><td>20.5</td></tr><tr><td>DeTAL [15]</td><td>I3D [4] / R(2+1)D [27]</td><td>CLIP-B [23]</td><td>38.3</td><td>32.3</td><td>24.4</td><td>16.3</td><td>9.0</td><td>24.1</td><td>34.4</td><td>23.0</td><td>4.0</td><td>22.4</td></tr><tr><td>STOV-TAL [7]</td><td>CLIP-B [23]</td><td>CLIP-B [23]</td><td>44.2</td><td>35.7</td><td>25.7</td><td>16.5</td><td>8.0</td><td>26.0</td><td>42.1</td><td>25.0</td><td>1.3</td><td>24.8</td></tr><tr><td>ZEETAD [22]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>45.2</td><td>38.8</td><td>30.8</td><td>22.5</td><td>13.7</td><td>30.2</td><td>39.2</td><td>25.7</td><td>3.1</td><td>24.9</td></tr><tr><td>mProTEA [24]</td><td>CLIP-B [23]</td><td>CLIP-B [23]</td><td>41.2</td><td>36.3</td><td>26.3</td><td>16.8</td><td>8.4</td><td>26.1</td><td>41.8</td><td>24.6</td><td>6.1</td><td>25.6</td></tr><tr><td>OVFormer [6]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>42.8†</td><td>37.3†</td><td>30.6†</td><td>23.5†</td><td>15.9†</td><td>30.5†</td><td>42.8†</td><td>27.3†</td><td>6.0†</td><td>27.2†</td></tr><tr><td>Ti-FAD [13]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>57.0</td><td>51.4</td><td>43.3</td><td>33.0</td><td>21.2</td><td>41.2</td><td>50.6</td><td>32.2</td><td>5.2</td><td>32.0</td></tr><tr><td>No external information</td><td>I3D [4]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>STALE [20]</td><td>CoCA [31]</td><td>CLIP-B [23] CoCA [31]</td><td></td><td></td><td></td><td></td><td></td><td></td><td>6.9</td><td>4.7</td><td>0.9</td><td>4.4</td></tr><tr><td>T3AL [16]</td><td></td><td></td><td>20.7†</td><td>14.3†</td><td>8.9†</td><td>5.3†</td><td>2.7†</td><td>10.4†</td><td>25.8†</td><td>13.9†</td><td>3.1†</td><td>14.3†</td></tr><tr><td>ZEAL [1]</td><td>LLaVA-OneVision [14]</td><td>CLIP-L [23]</td><td>21.1</td><td>15.0</td><td>9.9</td><td>5.0</td><td>2.6</td><td>10.7</td><td></td><td></td><td></td><td>一</td></tr><tr><td>Ti-FAD [13]</td><td>CoCA [31]</td><td>CoCA [31]</td><td>15.4</td><td>13.6</td><td>11.1</td><td>8.3</td><td>5.3</td><td>10.7</td><td>12.3</td><td>9.0</td><td>2.3</td><td>8.6</td></tr><tr><td>Ti-FAD [13]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>20.4</td><td>19.1</td><td>17.0</td><td>13.8</td><td>9.6</td><td>16.0</td><td>10.6</td><td>7.8</td><td>1.9</td><td>7.4</td></tr><tr><td>Ti-FAD [13] + Ours TF-CADE (Ours)</td><td>I3D [4] CoCA [31]</td><td>CLIP-B [23]</td><td>24.9</td><td>23.0</td><td>20.1 13.1</td><td>15.9</td><td>10.8</td><td>18.9</td><td>14.7</td><td>9.2</td><td>1.4 2.1</td><td>9.2</td></tr><tr><td>TF-CADE (Ours)</td><td>I3D [4]</td><td>CoCA [31] CLIP-B [23]</td><td>19.7 28.6</td><td>17.1 26.2</td><td>22.6</td><td>9.3 17.1</td><td>5.5 10.7</td><td>12.9 21.1</td><td>20.8 16.7</td><td>13.1 10.6</td><td>1.9</td><td>13.0 10.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>23.0</td><td>14.9</td><td></td><td>23.3</td><td></td><td></td><td>3.8</td><td>23.1</td></tr><tr><td rowspan="10">75% Seen</td><td>EffPrompt [10] STALE [20]</td><td>I3D [4] I3D [4]</td><td>CLIP-B [23] CLIP-B [23]</td><td>39.7 40.5</td><td>31.6 32.3</td><td>23.5 15.3</td><td>7.5</td><td>23.8</td><td>37.6 38.2</td><td>22.9 25.2</td><td>6.0</td><td>24.9</td></tr><tr><td>DeTAL [15]</td><td>I3D [4] / R(2+1)D [27]</td><td>CLIP-B [23]</td><td>39.8</td><td>33.6 25.9</td><td>17.4</td><td>7.6 9.9</td><td>25.3</td><td>39.3</td><td>26.4</td><td>5.0</td><td>25.8</td></tr><tr><td>STOV-TAL [7]</td><td>CLIP-B [23]</td><td>CLIP-B [23]</td><td>47.8</td><td>39.1 28.4</td><td>17.6</td><td>9.1</td><td>28.4</td><td>47.0</td><td>28.1</td><td>1.6</td><td>27.9</td></tr><tr><td>ZEETAD [22]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>61.4</td><td>53.9 44.7</td><td>34.5</td><td>20.5</td><td>43.2</td><td>51.0</td><td>33.4</td><td>5.9</td><td>32.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mProTEA [24]</td><td>CLIP-B [23]</td><td>CLIP-B [23]</td><td>43.1</td><td>38.2 28.2</td><td>18.1</td><td>8.7</td><td>27.9</td><td>44.5</td><td>27.4</td><td>7.9</td><td>27.6</td></tr><tr><td>OVFormer [6]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>49.8†</td><td>43.8† 35.8†</td><td>27.8†</td><td>19.2†</td><td>35.3†</td><td>46.7†</td><td>29.4†</td><td>6.1†</td><td>29.5†</td></tr><tr><td>Ti-FAD [13]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>64.0</td><td>58.5 49.7</td><td>37.7</td><td>24.1</td><td>46.8</td><td>53.8</td><td>34.8</td><td>7.0</td><td>34.7</td></tr><tr><td>No external information STALE [20]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>I3D [4] CoCA [31]</td><td>CLIP-B [23] CoCA [31]</td><td>19.2†</td><td>12.7† 7.4†</td><td></td><td></td><td>9.2†</td><td>14.0</td><td>9.8 14.9†</td><td>2.0 3.3†</td><td>9.5 15.4†</td></tr><tr><td>T3AL [16] ZEAL [1]</td><td>LLaVA-OneVision [14]</td><td>CLIP-L [23]</td><td>22.1</td><td>16.1</td><td>11.0</td><td>4.4† 5.7</td><td>2.2† 3.0</td><td>11.6</td><td>28.1†</td><td></td><td></td></tr><tr><td>Ti-FAD [13]</td><td>CoCA [31]</td><td>CoCA [31]</td><td>17.1</td><td>15.2</td><td>12.4</td><td>9.2</td><td>5.9</td><td>12.0</td><td>22.6</td><td>15.7</td><td>-</td></tr><tr><td>Ti-FAD [13]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>34.9</td><td>32.3</td><td>28.3</td><td>23.0</td><td>16.0</td><td>26.9</td><td>19.6</td><td>3.8</td><td>15.3 13.7</td></tr><tr><td>Ti-FAD [13] + Ours</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>42.4</td><td>38.7</td><td>33.1</td><td>25.7</td><td>16.4 31.2</td><td>25.6</td><td>14.2 16.5</td><td>3.5 3.2</td><td>16.4</td></tr><tr><td>TF-CADE (Ours)</td><td>CoCA [31]</td></table>

## 5.2. Main Results

In-distribution Evaluation. Table 1 shows a comparison of state-of-the-art ZSTAD methods on THUMOS14 and ActivityNet v1.3 under the in-distribution setting, where training and test classes are drawn from the same dataset but are disjoint. In the no external information scenario, our TF-CADE outperforms the existing models across both split settings (50%-50% and 75%-25%) with both I3D [4] and CoCA [31] features. Notably, under the 50%-50% split on THUMOS14, our method significantly outperforms the previous best by large margins. On ActivityNet v1.3, our model also improves the average mAP compared to previous methods. These results demonstrate the effectiveness of our method, even without relying on any external classifier.

Cross-dataset Generalization. We evaluate the generalization ability of TF-CADE in the cross-dataset setting, where models are trained on ActivityNet v1.3 and tested on THU-MOS14. Table 2 shows the performance on THUMOS14 under three different splits. The 50%-50% and 75%-25% settings are included solely to enable fair comparison with existing evaluation protocols. TF-CADE consistently outperforms prior works across all splits. Notably, under the most challenging 0%-100% setting, TF-CADE outperforms both

Table 2. Performance comparison with the state-of-the-art methods on THUMOS14 under zero-shot cross-dataset generalization setting. All methods are trained only on ActivityNet v1.3 and directly evaluated on THUMOS14 without any fine-tuning.
<table><tr><td rowspan="2">Evaluation Setting</td><td rowspan="2">Model</td><td colspan="2">Feature</td><td colspan="6">THUMOS14</td></tr><tr><td>Video</td><td>Text</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>Avg.</td></tr><tr><td rowspan="5">50%-50%</td><td>STALE [20]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>1.3</td><td>0.7</td><td>0.6</td><td>0.6</td><td>0.4</td><td>0.7</td></tr><tr><td>EffPrompt [10]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>5.4</td><td>4.4</td><td>3.5</td><td>2.7</td><td>1.9</td><td>3.6</td></tr><tr><td>T3AL [16]</td><td>CoCA [31]</td><td>CoCA [31]</td><td>20.7</td><td>14.3</td><td>8.9</td><td>5.3</td><td>2.7</td><td>10.4</td></tr><tr><td>Ti-FAD [13]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>19.3</td><td>15.8</td><td>11.7</td><td>7.7</td><td>4.1</td><td>11.7</td></tr><tr><td>TF-CADE (Ours)</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>40.5</td><td>34.1</td><td>26.5</td><td>17.2</td><td>8.9</td><td>28.2</td></tr><tr><td rowspan="5">75%-25%</td><td>STALE [20]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>0.5</td><td>0.3</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.3</td></tr><tr><td>EffPrompt [10]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>7.1</td><td>5.9</td><td>4.5</td><td>3.4</td><td>2.2</td><td>4.6</td></tr><tr><td>T3AL [16]</td><td>CoCA [31]</td><td>CoCA [31]</td><td>19.2</td><td>12.7</td><td>7.4</td><td>4.4</td><td>2.2</td><td>9.2</td></tr><tr><td>Ti-FAD [13]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>21.9</td><td>17.6</td><td>12.7</td><td>8.5</td><td>4.4</td><td>13.0</td></tr><tr><td>TF-CADE (Ours)</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>41.8</td><td>34.9</td><td>27.1</td><td>17.6</td><td>9.2</td><td>26.1</td></tr><tr><td rowspan="3">0%-100%</td><td>T3AL [16]</td><td>CoCA [31]</td><td>CoCA [31]</td><td>18.3</td><td>12.9</td><td>8.7</td><td>5.3</td><td>2.9</td><td>9.6</td></tr><tr><td>Ti-FAD [13]</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>17.9</td><td>15.0</td><td>11.1</td><td>7.4</td><td>4.0</td><td>11.1</td></tr><tr><td>TF-CADE (Ours)</td><td>I3D [4]</td><td>CLIP-B [23]</td><td>42.8</td><td>36.5</td><td>28.6</td><td>19.0</td><td>10.1</td><td>27.4</td></tr></table>

T3AL [16] and Ti-FAD [13] by large margins. In Table 3, we further evaluate the generalization ability, including the larger-scale dataset HACS-Segment by comparing it with THUMOS14 and ActivityNet v1.3. For fair cross-dataset evaluation, we use VideoMAE [26] features that are available for all three datasets under same feature extraction setting. As demonstrated in Table 3, TF-CADE achieves consistent improvements over Ti-FAD [13] on all datasets, indicating strong robustness in diverse cross-dataset scenarios.

Table 3. Performance comparison with the state-of-the-art methods under zero-shot cross-dataset generalization setting. All models are trained on only the training dataset and directly evaluated on the target datasets without any fine-tuning.
<table><tr><td rowspan="2">Training Dataset</td><td rowspan="2">Model</td><td colspan="2">Feature</td><td colspan="6">THUMOS14</td><td colspan="4">ActivityNet v1.3</td></tr><tr><td>Video</td><td>Text</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>Avg.</td><td>0.5</td><td>0.75</td><td>0.95</td><td>Avg.</td></tr><tr><td rowspan="2">HACS-Segment</td><td>Ti-FAD [13]</td><td>VideoMAE [26]</td><td>CLIP-B [23]</td><td>32.5</td><td>30.1</td><td>26.5</td><td>21.6</td><td>14.3</td><td>25.0</td><td>43.1</td><td>22.7</td><td>2.3</td><td>24.1</td></tr><tr><td>TF-CADE (Ours)</td><td>VideoMAE [26]</td><td>CLIP-B [23]</td><td>41.8</td><td>37.9</td><td>32.4</td><td>25.2</td><td>16.2</td><td>30.7</td><td>46.2</td><td>22.2</td><td>3.1</td><td>24.8</td></tr></table>

(a) Trained on HACS-Segment [33]

<table><tr><td rowspan="2">Training Dataset Model</td><td rowspan="2"></td><td colspan="2">Feature</td><td colspan="6">THUMOS14</td><td colspan="4">HACS-Segment</td></tr><tr><td>Video</td><td>Text</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>Avg.</td><td>0.5</td><td>0.75</td><td>0.95</td><td>Avg.</td></tr><tr><td rowspan="2">ActivityNet v1.3</td><td>Ti-FAD [13]</td><td>VideoMAE [26]</td><td>CLIP-B [23]</td><td>21.8</td><td>17.9</td><td>13.3</td><td>8.5</td><td>4.3</td><td>13.2</td><td>22.0</td><td>7.0</td><td>0.6</td><td>9.7</td></tr><tr><td>TF-CADE (Ours)</td><td>VideoMAE [26]</td><td>CLIP-B [23]</td><td>36.9</td><td>30.1</td><td>22.2</td><td>13.6</td><td>6.7</td><td>21.9</td><td>30.0</td><td>10.2</td><td>1.0</td><td>13.6</td></tr></table>

(b) Trained on ActivityNet v1.3 [5]

## 5.3. Further Analysis

Ablation Study on Cross-modal Update. As discussed in Sec. 3.2, we adopt the cross-modal adaptation proposed in Ti-FAD [13] as our baseline. This baseline updates both text and video features through a bi-directional cross-attention mechanism, where each modality alternately serves as the query. As shown in Table 4, the performance difference among using text, video, or both as the query is marginal. Therefore, in our framework, we adopt a simplified design that updates only the video features through cross-attention. Component Analysis. Table 5 shows an ablation study of each component in TF-CADE on THUMOS14. The second row denotes that the model is trained using the foregroundbased video-level similarity score $S _ { f g }$ produced by our ACA, which yields marginal improvement when used alone. When applying CCR alone, the model exhibits larger performance gain by suppressing irrelevant classes. Finally, combining both $\mathcal { L } _ { v i d e o }$ and CCR shows best performance, demonstrating that two components are complementary, as $S _ { f g }$ trained with $\mathcal { L } _ { v i d e o }$ provides a global prior that improves snippetlevel classification by re-weighting confidence scores.

Effectiveness of Smoothing Filter. To validate the effectiveness of applying a Gaussian smoothing filter to the temporal action certainty map, we compare performance under the zero-shot cross-dataset setting. Table 6 shows that applying the Gaussian smoothing filter leads to a clear improvement, demonstrating its effectiveness in reducing the impact of noise and overly sharp peaks in action segments. This smoothing process enables the model to concentrate on the foreground regions, leading to better generalization.

False Positive Analysis. Fig. 4 illustrates a false positive analysis on THUMOS14 using DETAD [2]. The analysis contains error types such as wrong label errors. Compared to Ti-FAD, our TF-CADE exhibits a notable reduction in wrong label errors. This reduction indicates that TF-CADE more effectively injects text features into the detection process, leading to improved action classification.

False Negative Analysis. Fig. 5 shows a false negative analysis on THUMOS14 using DETAD [2] across different action lengths. Without CCR, the model suffers from high false negatives, particularly for extremely short (XS) and long (XL) actions, where snippet-level classification alone fails to capture the complete temporal context. By introducing CCR, which re-weights snippet-level confidence using the video-level certainty score obtained from the action concentrate aggregation, TF-CADE effectively suppresses false negatives in these challenging cases. This demonstrates that video-level score re-weighting provides a complementary cue to refine snippet-wise predictions, enabling more reliable detection across varying temporal durations.

Table 4. Comparison of cross-attention configurations. The table shows how different modality update choices affect performance and support our simplified video-only update design.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Text</td><td rowspan="2">Video</td><td colspan="4">mAP@tIoU (%)</td></tr><tr><td>0.3</td><td>0.5</td><td>0.7</td><td>Avg.</td></tr><tr><td rowspan="3">Baseline</td><td>√</td><td></td><td>20.1</td><td>15.8</td><td>8.0</td><td>14.9</td></tr><tr><td></td><td>√</td><td>21.4</td><td>17.0</td><td>8.7</td><td>16.0</td></tr><tr><td>√</td><td>√</td><td>22.3</td><td>17.4</td><td>8.4</td><td>16.3</td></tr><tr><td rowspan="3">Ours</td><td>√</td><td></td><td>27.3</td><td>21.2</td><td>9.8</td><td>19.9</td></tr><tr><td></td><td>√</td><td>28.6</td><td>22.6</td><td>10.7</td><td>21.1</td></tr><tr><td>√</td><td>√</td><td>27.1</td><td>20.6</td><td>9.3</td><td>19.3</td></tr></table>

Table 5. Component analysis on THUMOS14. The average mAP is calculated to evaluate performance at all tIoU thresholds.
<table><tr><td rowspan="2">Method</td><td colspan="6">mAP@tIoU(%)</td></tr><tr><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>Avg.</td></tr><tr><td>Baseline</td><td>21.4</td><td>19.6</td><td>17.0</td><td>13.4</td><td>8.7</td><td>16.0</td></tr><tr><td> $+ \mathcal { L } _ { v i d e o }$ </td><td>22.0</td><td>20.2</td><td>17.4</td><td>13.6</td><td>8.6</td><td>16.4</td></tr><tr><td>+CCR</td><td>27.1</td><td>24.9</td><td>21.0</td><td>16.0</td><td>9.8</td><td>19.7</td></tr><tr><td> $+ \mathcal { L } _ { v i d e o }$  &amp; CCR</td><td>28.6</td><td>26.2</td><td>22.6</td><td>17.1</td><td>10.7</td><td>21.1</td></tr></table>

Table 6. Effectiveness of smoothing filter under zero-shot crossdataset generalization setting. The models are trained on ActivityNet v1.3 and evaluated on THUMOS14.
<table><tr><td rowspan="2">Method</td><td colspan="6">mAP@tIoU(%)</td></tr><tr><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>Avg.</td></tr><tr><td>w/o filter</td><td>35.7</td><td>28.6</td><td>21.7</td><td>13.5</td><td>6.7</td><td>21.3</td></tr><tr><td>w/ filter</td><td>42.8</td><td>36.5</td><td>28.6</td><td>19.0</td><td>10.1</td><td>27.4</td></tr></table>

Ablation and Design Analysis of ACA. Table 7 shows how different architectural choices in ACA, certainty maps, smoothing strategies, and kernel parameters. To validate the effectiveness of ACA, we compare our ACA with mean pooling, which is the simplest and most commonly used temporal aggregation method. Table 7 (a) shows that ACA consistently outperforms the baseline across all tIoU thresholds. This result highlights the importance of aggregating foreground regions into video-level representations. Table 7 (b) illustrates the types of action certainty maps. Combining $m _ { \mathrm { m a x } } ^ { ( l ) }$ and $m _ { \mathrm { f i l t e r } } ^ { ( l ) }$ yields the best performance, compared to using either map alone, indicating that their integration provides a more reliable aggregation of action certainty. Table 7 (c) compares three smoothing strategies applied to the action certainty map $m _ { \mathrm { f i l t e r } } ^ { ( l ) }$ . The first row uses a learnable 1D convolution filter, while the second initializes a learnable filter with a Gaussian kernel. The third row applies a fixed Gaussian smoothing filter without any learnable parameters. Among the three, the fixed Gaussian filter yields the highest performance, and thus we adopt this smoothing strategy in our final model. Furthermore, we analyze the hyperparameter σ in Gaussian kernel in Table 7 (d). Performance remains stable across different sigma values, demonstrating that our method is robust to the choice of σ.

![](images/c55a14d045d971587db1d3b8c7d2040be3bbd0f40ee9c9b680b1979499c68a03.jpg)

Table 7. Ablation studies of ACA on THUMOS14 in the 50%-50% setting.
<table><tr><td rowspan="2">Type</td><td colspan="4">mAP@tIoU (%)</td></tr><tr><td>0.3</td><td>0.5</td><td>0.7</td><td>Avg.</td></tr><tr><td>Mean pooling</td><td>26.4</td><td>20.2</td><td>9.0</td><td>18.9</td></tr><tr><td>Certainty</td><td>28.6</td><td>22.6</td><td>10.7</td><td>21.1</td></tr></table>

(a) Aggregation method

<table><tr><td rowspan="2">Type</td><td colspan="4">mAP@tIoU (%)</td></tr><tr><td>0.3</td><td>0.5</td><td>0.7</td><td> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td> $m _ { \mathrm { m a x } } ^ { ( l ) }$ </td><td>28.0</td><td>21.9</td><td>10.9</td><td>20.7</td></tr><tr><td>mfiher (l)</td><td>26.6</td><td>20.9</td><td>10.2</td><td>19.6</td></tr><tr><td>mmlax + m(lter (l) (l)</td><td>28.6</td><td>22.6</td><td>10.7</td><td>21.1</td></tr></table>

(b) Action certainty map

Background Err Localization Err Double Detection Err   
Confusion Err Wrong Label Err True Positive

![](images/65ec074589ae1cc83595a343e2cd4d4b3c5bf592f4182e65caef59fb3c9002ed.jpg)

![](images/0d4d9a04ecf6d702685361e49825430ba8e3c9922d1bff0e6aa6653b660406ca.jpg)  
(a) Ti-FAD [13]  
(b) TF-CADE (Ours)  
Figure 4. False positive analysis on THUMOS14 using DE-TAD [2]. Compared to Ti-FAD, our TF-CADE reduces the dependency on the Wrong Label Error.

Visualization of Certainty Map in ACA. Fig. 6 visualizes the temporal action certainty maps produced by our ACA. In Fig. 6 (a), the certainty map based on $m _ { \operatorname* { m a x } } ^ { ( l ) }$ exhibits sharp and fragmented activations. After applying the Gaussian filter, as shown in Fig. 6 (b), the certainty distribution of $m _ { \mathrm { f i l t e r } } ^ { ( l ) }$ becomes smoother and better highlights foreground action regions. This smoothing process reduces spurious peaks and preserves scores within action segments. ACA allows the model to focus on action-relevant regions while down-weighting ambiguous or background snippets. The refined certainty map provides a stable and reliable confidence basis for video-level re-weighting in CCR. Additional visualizations and analyses are presented in the Supp.

<table><tr><td rowspan="2">Type</td><td colspan="4">mAP@tIoU (%)</td></tr><tr><td>0.3</td><td>0.5</td><td>0.7</td><td>Avg.</td></tr><tr><td>Learnable (1D conv)</td><td>28.7</td><td>21.7</td><td>9.9</td><td>20.5</td></tr><tr><td>Gaussian + Learnable</td><td>27.5</td><td>21.3</td><td>10.5</td><td>20.2</td></tr><tr><td>Fix (Gaussian)</td><td>28.6</td><td>22.6</td><td>10.7</td><td>21.1</td></tr></table>

(c) Smoothing strategy

<table><tr><td rowspan="2">Type</td><td colspan="4">mAP@tIoU (%)</td></tr><tr><td>0.3</td><td>0.5</td><td>0.7</td><td>Avg.</td></tr><tr><td>σ = 1</td><td>28.6</td><td>22.6</td><td>10.7</td><td>21.1</td></tr><tr><td>σ = 2</td><td>28.8</td><td>22.1</td><td>10.1</td><td>20.7</td></tr><tr><td>σ = 3</td><td>28.7</td><td>22.3</td><td>10.2</td><td>20.8</td></tr></table>

(d) σ in Gaussian kernel

![](images/330c541623db700864447c42acdf9a2d5443351241765c5edc7cab5297b6015d.jpg)  
Figure 5. False negative analysis on THUMOS14 using DE-TAD [2]. CCR reduces false negatives consistently across actions of different temporal durations, especially for extremely short (XS) and long (XL) actions.

![](images/d144cf61751f7bf033c624063fc3fa86a69c86ee860fbfc3d0be78699e4ef072.jpg)  
Figure 6. Visualization of Temporal Action Certainty Maps on THUMOS14. (a) Certainty map $m _ { \operatorname* { m a x } } ^ { ( l ) }$ without smoothing. (b) Filtered Certainty map $m _ { \mathrm { { f i l t e r } } } ^ { ( l ) }$ by Gaussian kernel. The bar below plot indicates the ground-truth action segments.

## 6. Conclusion

In this paper, we address the issue of text-irrelevant predictions in ZSTAD. We propose TF-CADE, a novel framework that explicitly aligns textual information with action-relevant foreground regions. Our proposed Action Concentrate Aggregation (ACA) enhances text-video alignment by aggregating temporally informative segments into a foregroundweighted video embedding, while our Certainty-based Confidence Re-weighting (CCR) refines snippet-level predictions by leveraging video-level similarity score during inference. Extensive experiments on THUMOS14, ActivityNet v1.3, and HACS-Segment demonstrate that TF-CADE achieves state-of-the-art performance under both in-distribution and cross-dataset generalization settings, validating its effectiveness and robustness. We believe that our findings provide new insights into leveraging foreground concentration for more semantically grounded text-video alignment in ZSTAD.

## Acknowledgement

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant, funded by the Korea government (MSIT) (No. RS-2019-II190079, Artificial Intelligence Graduate School Program (Korea University), No. RS-2024-00457882, AI Research Hub Project, No. RS-2025-02304828, Artificial Intelligence Star Fellowship Support Program to Nurture the Best Talents, and No. RS-2022-II220984, Development of Artificial Intelligence Technology for Personalized Plug-and-Play Explanation and Verification of Explanation).

## References

[1] Josiah Aklilu, Xiaohan Wang, and Serena Yeung-Levy. Zeroshot action localization via the confidence of large visionlanguage models. arXiv preprint arXiv:2410.14340, 2024. 3, 6

[2] Humam Alwassel, Fabian Caba Heilbron, Victor Escorcia, and Bernard Ghanem. Diagnosing error in temporal action detectors. In Proceedings of the European Conference on Computer Vision, pages 256–272, 2018. 7, 8

[3] Navaneeth Bodla, Bharat Singh, Rama Chellappa, and Larry S Davis. Soft-nms–improving object detection with one line of code. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 5561–5569, 2017. 5

[4] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 3, 5, 6

[5] Bernard Ghanem Fabian Caba Heilbron, Victor Escorcia and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 961–970, 2015. 5, 7

[6] Akshita Gupta, Aditya Arora, Sanath Narayan, Salman Khan, Fahad Shahbaz Khan, and Graham W. Taylor. Openvocabulary temporal action localization using multimodal guidance. In Proceedings of the British Machine Vision Conference, 2024. 1, 3, 6

[7] Jeongseok Hyun, Su Ho Han, Hyolim Kang, Joon-Young Lee, and Seon Joo Kim. Exploring scalability of self-training for open-vocabulary temporal action localization. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, pages 9388–9397, 2025. 1, 3, 6

[8] Haroon Idrees, Amir R Zamir, Yu-Gang Jiang, Alex Gorban, Ivan Laptev, Rahul Sukthankar, and Mubarak Shah. The THUMOS challenge on action recognition for videos “in the wild”. Computer Vision and Image Understanding, 155:1–23, 2017. 2, 5

[9] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In Proceedings of the International Conference on Machine Learning, pages 4904– 4916, 2021. 1, 3

[10] Chen Ju, Tengda Han, Kunhao Zheng, Ya Zhang, and Weidi Xie. Prompting visual-language models for efficient video understanding. In Proceedings ofthe European Conference on Computer Vision, pages 105–124, 2022. 1, 3, 5, 6

[11] Kumara Kahatapitiya, Anurag Arnab, Arsha Nagrani, and Michael S Ryoo. Victr: Video-conditioned text representations for activity recognition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18547–18558, 2024. 3

[12] Ho-Joong Kim, Jung-Ho Hong, Heejo Kong, and Seong-Whan Lee. Te-tad: Towards full end-to-end temporal action detection via time-aligned coordinate expression. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18837–18846, 2024. 3

[13] Yearang Lee, Ho-Joong Kim, and Seong-Whan Lee. Textinfused attention and foreground-aware modeling for zeroshot temporal action detection. In Advances in Neural Information Processing Systems, pages 9864–9884, 2024. 1, 2, 3, 5, 6, 7, 8

[14] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 6

[15] Z. Li, Y. Zhong, R. Song, T. Li, L. Ma, and W. Zhang. Detal: Open-vocabulary temporal action localization with decoupled networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):7728–7741, 2024. 1, 3, 6

[16] Benedetta Liberatori, Alessandro Conti, Paolo Rota, Yiming Wang, and Elisa Ricci. Test-time zero-shot temporal action localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18720– 18729, 2024. 1, 3, 5, 6

[17] Chuming Lin, Chengming Xu, Donghao Luo, Yabiao Wang, Ying Tai, Chengjie Wang, Jilin Li, Feiyue Huang, and Yanwei Fu. Learning salient boundary feature for anchor-free temporal action localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3320–3329, 2021. 3

[18] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In´ Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2980–2988, 2017. 5

[19] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016. 5

[20] Sauradip Nag, Xiatian Zhu, Yi-Zhe Song, and Tao Xiang. Zero-shot temporal action detection via vision-language prompting. In Proceedings of the European Conference on Computer Vision, pages 681–697, 2022. 1, 3, 5, 6

[21] Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, and Haibin Ling. Expanding language-image pretrained models for general video recognition. In Proceedings of the European Conference on Computer Vision, pages 1–18, 2022. 3

[22] Thinh Phan, Khoa Vo, Duy Le, Gianfranco Doretto, Donald Adjeroh, and Ngan Le. Zeetad: Adapting pretrained visionlanguage model for zero-shot end-to-end temporal action

detection. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 7046–7055, 2024. 1, 3, 6

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, pages 8748–8763, 2021. 1, 3, 5, 6, 7

[24] A. Raza, B. Yang, and Y. Zou. Zero-shot temporal action detection by learning multimodal prompts and text-enhanced actionness. IEEE Transactions on Circuits and Systems for Video Technology, 34(11):11000–11012, 2024. 1, 3, 6

[25] Dingfeng Shi, Yujie Zhong, Qiong Cao, Lin Ma, Jia Li, and Dacheng Tao. Tridet: Temporal action detection with relative boundary modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18857–18866, 2023. 3

[26] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In Advances in Neural Information Processing Systems, pages 10078–10093, 2022. 5, 6, 7

[27] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6450–6459, 2018. 6

[28] Limin Wang, Yuanjun Xiong, Dahua Lin, and Luc Van Gool. Untrimmednets for weakly supervised action recognition and detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4325–4334, 2017. 3, 5

[29] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In Proceedings of the European Conference on Computer Vision, pages 396–416, 2024. 3

[30] Wenhao Wu, Zhun Sun, and Wanli Ouyang. Revisiting classifier: Transferring vision-language models for video recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2847–2855, 2023. 3

[31] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 3, 5, 6

[32] Chen-Lin Zhang, Jianxin Wu, and Yin Li. Actionformer: Localizing moments of actions with transformers. In Proceedings of the European Conference on Computer Vision, pages 492–510, 2022. 3, 5

[33] Hang Zhao, Antonio Torralba, Lorenzo Torresani, and Zhicheng Yan. Hacs: Human action clips and segments dataset for recognition and temporal localization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8668–8678, 2019. 5, 7

[34] Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, and Dongwei Ren. Distance-iou loss: Faster and better

learning for bounding box regression. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 12993– 13000, 2020. 5