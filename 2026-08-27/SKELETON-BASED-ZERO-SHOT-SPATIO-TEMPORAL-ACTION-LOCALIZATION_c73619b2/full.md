# SKELETON-BASED ZERO-SHOT SPATIO-TEMPORAL ACTION LOCALIZATION VIA WEAKLY-SUPERVISED PRETRAINING

Koshiro Nagano<sup>1,2∗</sup> Fumiaki Sato<sup>3†∗</sup> Ryo Hachiuma<sup>4†</sup> Kazuki Tsutsukawa<sup>1</sup> Taiki Sekii<sup>3†‡</sup>

<sup>1</sup>Konica Minolta, Inc. <sup>2</sup>Keio University <sup>3</sup>CyberAgent <sup>4</sup>NVIDIA

## ABSTRACT

We propose a novel pretraining strategy for skeleton-based zero-shot spatio-temporal action localization to estimate unseen actions for person instances while overcoming high annotation costs for training via new target actions and pretraining using large-scale action scenery datasets. Specifically, our approach, termed Skeleton-Language feature Pooling Switching, introduces a weakly-supervised vision-language pretraining mechanism. This mechanism transitions pooling kernels from pretraining, which aggregates skeleton features at the video level and aligns them with each video’s known action text embeddings, to the inference phase that computes instance-level features without training via target actions. Furthermore, we propose Scene-Mixed Discriminative Contrastive Learning to distinguish actions at the instance level within the combined scene through the MIL framework. Our experiments on four spatio-temporal action localization and classification datasets demonstrate that the proposed method effectively addresses annotation limitations.

Index Terms— Action Understanding, Skeleton-based Action Recognition, Human Pose Estimation

## 1. INTRODUCTION

Recognizing a person’s actions in a video is crucial for various applications such as robotics [1, 2] and surveillance cameras [3–5]. Three primary tasks have been extensively investigated in action recognition: action classification<sup>1</sup>, temporal action localization, and spatio-temporal action localization. In action classification, action labels are assigned to an entire video. In temporal action localization, action labels are assigned to each frame, and in spatio-temporal action localization, action labels are assigned to person instances, such as the bounding boxes or skeletons detected in each video frame. Two primary approaches have been the main focus of the study. The first approach leverages the appearance information extracted from videos [4, 6], and the second approach relies exclusively on human skeletons<sup>2</sup> detected in videos [3, 7, 8] to serve as inputs for deep neural networks (DNNs). The appearance-based approaches use appearance features obtained by applying DNNs to an input video. By contrast, the skeleton-based approaches use only low-information keypoints detected using the multi-person pose estimation methods [9], making these approaches relatively robust to such appearance changes in a scene or a person [10, 11]. Moreover, even recent multi-modal LLMs [12], which heavily rely on appearance cues from video inputs, often struggle to accurately capture subtle human actions or states [13], thereby revealing their limited robustness in realworld scenarios (cf. Sec. 4.3). Therefore, this study aims to robustly recognize each individual’s actions across diverse scenes by focusing on skeleton-based spatio-temporal action localization.

In conventional approaches, spatio-temporal action localization can be accomplished by feeding a cropped video clip [6] or a sequence of skeletons [3, 7, 8] into traditional action classification methods at predefined-frame intervals. This approach, called tracking-based, is contingent upon the precise processing of person instance detections using multiperson tracking methods. However, these tracking-based approaches have not learned the transition points of actions to accurately identify the start and end times of actions. Conversely, the supervised approaches [14, 15] achieve precise localization by employing action labels annotated for each individual in every frame during the DNN training. However, this method necessitates dense, instance-level annotations throughout the training process. To mitigate the burden of annotation costs, weakly-supervised methods have been proposed. These methods rely on a single label for the entire video for supervision [16,17]. The fully or weakly-supervised approaches encounter a significant challenge when there is a need to recognize new actions not included in the training dataset (referred to as target actions). This entails considerable human effort in collecting and annotating a vast number of video images to train the DNNs.

Furthermore, recent developments in computer vision have introduced a vision-language pretraining paradigm [18] utilizing contrastive learning between image features and text embeddings. This approach notably enhances the capability to recognize unknown object classes. Action recognition fields have recently seen an increasing application of such a pretraining approach. Several methods have been proposed to identify actions at the video [19–21], frame [22], or instance level [23] in a zero-shot manner without training on the target actions. These methods eliminate the need for the annotation costs that are typically required for the target actions. However, the conventional zero-shot spatio-temporal action localization method [23, 24] necessitates considerable human effort in the pretraining phase. The reason is that this method requires the annotation of action labels for each individual in every frame of a large number of pretraining videos, including actions other than the target actions, to acquire a generalized feature representation of diverse actions for DNNs.

![](images/5c0496c4f04f6c8f5c7329a076eadf4518dc3402029d9711d92922c5a568c8f2.jpg)  
Fig. 1: Overview of proposed SLPS learning framework.

## 1.1. Overview and Contributions

To reduce the annotation costs required for training new target actions and instance-level pretraining, we propose a novel approach to pretrain<sup>3</sup> DNNs in a weakly-supervised manner using only video-level action labels. This approach facilitates the zero-shot detection of instances and their unknown target actions in each frame. To achieve the weakly-supervised pretraining manner, we employ vision-language pretraining for action recognition at the video level and introduce a novel mechanism called Skeleton-Language feature Pooling Switching (SLPS) mechanism. It switches the pooling kernels from the pretraining, which aggregates skeleton features at the video level, to the inference phase to recognize instancelevel actions. The proposed DNN architecture incorporates a permutation-invariant structure [25] for pooling features per video in the pretraining phase and a permutation-equivariant structure [26] for extracting features for each instance in the inference phase. We apply contrastive learning to ensure that these features align with the text embeddings of action labels in a common space. Consequently, the proposed architecture eliminates the need for multi-person tracking steps typically required in conventional methods.

Additionally, the proposed weakly-supervised pretraining approach learns to classify actions performed by individuals within a video into a single class regardless of the number of people present, based on the Multiple Instance Learning (MIL) framework [27]. However, this assumption does not hold in complex scenes with multiple people. To mitigate this issue during pretraining, we propose a novel method called Scene-Mixed Discriminative Contrastive Learning (SM-DCL), which combines multiple scenes for pretraining. SM-DCL merges instances from different scenes into a unified context, enabling contrastive learning to differentiate actions across instances within the combined scene.

We performed experiments to verify the effectiveness of the proposed method in terms of the aforementioned limitations of annotation costs on four datasets for spatio-temporal action localization and classification.

The main contributions of this study are summarized as follows. (1) We define a novel task—zero-shot spatiotemporal action localization in a weakly-supervised pretraining manner—that requires no instance-level action labels in the pretraining phase. (2) Using the proposed SLPS mechanism, we demonstrate the feasibility of such a weaklysupervised pretraining that identifies actions for each instance in the inference phase without requiring instance-level annotations over the training phase and multi-person tracking methods. (3) We propose SM-DCL to distinguish actions at the instance level within the MIL framework.

## 2. RELATED WORK

## 2.1. Spatio-temporal Action Localization

Spatio-temporal action localization is categorized into fully-[14, 28, 29] and weakly-supervised approaches [16, 17, 30]. While supervised approaches achieve high precision, they demand costly frame-wise instance annotations. To mitigate this, weakly-supervised approaches have been proposed to utilize video-level labels. Tracking-based methods, which identify actions in an individual skeleton sequence generated using multi-person tracking methods, have also been proposed. However, these approaches require significant human costs for collecting many videos and annotating the target action labels when the target actions are not included in the training data.

## 2.2. Zero-shot Action Recognition

The field of vision and language has attracted considerable attention from researchers for the zero-shot visual recognition task [18,31–33] identifying the unseen target in the visual data with a text prompt that describes the target. The performance can be enhanced by introducing contrastive learning [18] between image features and text embeddings extracted from the prompts. Recently, the context of vision and language has also been introduced to action recognition [19–24, 34, 35]. In spatio-temporal action localization, the appearance-based approach has been proposed [23, 24, 35]. This approach requires a high human effort to prepare numerous videos that include various actions and annotate instance-level labels in the pretraining phase. Similar to the tracking-based approaches, zero-shot localization can be achieved using the information regarding individual person sequences [19–21] as inputs to zero-shot action classification methods. Our method achieves zero-shot localization using only video-level supervision for pretraining, without relying on trackers or instance-level annotations.

## 2.3. Data Mixing Augmentation

The field of image recognition has been actively researching the data mixing augmentation [36–44], which improves the robustness of the model by employing multiple images for data augmentation. For instance, CutMix [38] proposed to overlay a cropped area of an input image on another image to augment the data. This approach has been applied to the weakly-supervised spatio-temporal action localization task [45]. To further differentiate actions across instances in the context of MIL framework, we apply contrastive learning to the combined scenes.

## 3. PROPOSED METHOD

Fig. 1 shows the overview of the proposed framework. The DNNs in the proposed SLPS mechanism extract skeleton features for each skeletal instance (hereafter simply referred to as an instance) in the inference and pretraining phases. In the pretraining phase, the SLPS mechanism aggregates these features into a video-level feature using Global Max-Pooling (GMPool), and based on CLIP [18], it performs contrastive learning between the video-level features and text embeddings extracted from action label names using a text encoder, such as MPNet [46]. Such action labels can be found in large-scale action classification datasets like Kinetics-400 [47]. During inference, because each instance’s features are aligned with the text space, the target action score is computed by measuring the similarity between the instance feature and the text embedding of the user-provided prompt, following the CLIP framework.

## 3.1. Inference Phase

In the pretraining and inference phases, the multi-person pose estimation [9] is applied to the input video to extract FSK skeleton keypoints, where F, S, and K denote the number of frames in the video clip, the number of skeletal instances per frame, and the number of keypoints per instance, respectively. Each keypoint is then transformed into an input vector v for the DNNs. The input is a five-dimensional vector consisting of the two-dimensional keypoint coordinates on the image, time index, keypoint confidence, and keypoint index. If the number of detected skeletons or joints in each frame is fewer than S or $K$ , respectively, the input tensor is padded with zero vectors.

In the inference phase, the feature vector $\mathbf { x } _ { s }$ for each instance $s \in \{ 1 , \ldots , F S \}$ , aligned in a common space with the text embeddings described in section 3.2, is obtained by inputting the above input vectors into the architecture described in section 3.3. Consequently, given $T$ text embeddings $\mathcal { V } =$ $\left\{ \mathbf { y } _ { 1 } , \ldots , \mathbf { y } _ { T } \right\}$ extracted by a text encoder for each target action and the feature vector $\mathbf { x } _ { s }$ for each instance s, the score $P _ { s }$ representing each instance s includes the target action specified by the user, is formulated as follows:

$$
P _ { s } = \operatorname* { m a x } \left( \cos \left( f ( \mathbf { x } _ { s } ) , \mathbf { y } _ { 1 } \right) , \dots , \cos \left( f ( \mathbf { x } _ { s } ) , \mathbf { y } _ { T } \right) \right) ,\tag{1}
$$

where $\cos ( \cdot , \cdot )$ represents the cosine similarity between two vectors, and $f$ denotes pretrained multilayer perceptron (MLP) to align the dimension of $\mathbf { x } _ { s }$ and $\mathbf { y }$

## 3.2. Pretraining Phase

In the pretraining phase, the DNNs in the proposed SLPS mechanism extract the features $\mathbf { x } _ { s }$ for each instance s similar to the inference phase. In contrast to the inference phase, GM-Pool aggregates these features into a single video-level feature z representing the entire video. This enables the pretraining scheme to introduce a weakly-supervised learning framework based on MIL [27].

We use contrastive learning between the video-level skeleton features and the text embeddings extracted from action label names and multitask learning on the action classification task. To achieve the MIL-based pretraining that uses the video-level action labels, we define the total loss $\mathcal { L }$ consisting of the action classification loss $\mathcal { L } _ { \mathrm { c l s } }$ and the contrastive loss ${ \mathcal { L } } _ { \mathrm { c o n t } }$ in a batch of B videos as follows:

$$
\mathcal { L } = \alpha \sum _ { i = 1 } ^ { B } \mathcal { L } _ { \mathrm { c l s } , i } + ( 1 - \alpha ) \mathcal { L } _ { \mathrm { c o n t } } ,\tag{2}
$$

where α is the mixing ratio of the loss functions. The classification loss $\mathcal { L } _ { \mathrm { c l s } }$ uses the cross-entropy loss over C action classes, where the logits are computed by feeding the videolevel skeleton feature, aggregated by GMPool, into a fullyconnected classifier.

Based on the loss function proposed by CLIP [18], the contrastive loss is $\mathcal { L } _ { \mathrm { c o n t } } = ( \mathcal { L } _ { \mathrm { s 2 t } } + \mathcal { L } _ { \mathrm { t 2 s } } ) / 2$ . We use a projection head $f ( \cdot )$ to map z to the embedding space. Here, $\mathcal { L } _ { \mathrm { s 2 t } }$ contrasts each projected video feature $f ( \mathbf { z } _ { i } )$ with all text embeddings $\{ \mathbf { y } _ { j } \} _ { j = 1 } ^ { B }$ in the batch:

$$
\mathcal { L } _ { \mathrm { s 2 t } } = - \sum _ { i = 1 } ^ { B } \log \frac { \exp \left( \mathrm { C o s } \left( f \left( \mathbf { z } _ { i } \right) , \mathbf { y } _ { i } \right) / \tau \right) } { \sum _ { j = 1 } ^ { B } \exp \left( \mathrm { C o s } \left( f \left( \mathbf { z } _ { i } \right) , \mathbf { y } _ { j } \right) / \tau \right) } ,\tag{3}
$$

where $\left( \mathbf { z } _ { i } , \mathbf { y } _ { i } \right)$ is a positive pair and τ is a learnable temperature parameter. ${ \mathcal { L } } _ { \mathrm { t 2 s } }$ is defined symmetrically by swapping the roles of video and text.

## 3.3. SLPS DNN Architecture

Overview. For weakly-supervised pretraining, the DNN architecture of the proposed SLPS mechanism (Fig. 1) incorporates a Pooling-Switching mechanism. This mechanism employs GMPool to transition the unit of feature aggregation from the pretraining phase to the inference phase. In the pretraining and inference phases, the feature vectors for each instance are computed using a common set of multiple blocks. The Point Embedding block embeds the input per keypoint vector into a high-dimensional feature vector using MLPs. The Grouped Pool Block, described in section 6.3.2, aggregates the feature vectors transformed in the MLP block, considering the relationships between keypoints into a single feature vector for each group belonging to the same instance. Subsequently, the MLP and GMPool blocks compute the feature vectors for each instance by considering the relationships between instances. In the inference phase, the features for each instance are used to calculate scores using Eq. (1) as described in section 3.1. Meanwhile, in the pretraining phase, GMPool aggregates the features into a video-level feature, which is subsequently aligned with the text embedding space of action labels, as described in section 3.2.

![](images/5c99b19fda252512c3348cf9d0f3bce8369f07fc4401cce9e6f1d7afebeada47.jpg)  
Fig. 2: Overview of Scene-Mixed Discriminative Contrastive Learning for zero-shot spatio-temporal action localization. The modules same as those in Fig. 1 are abbreviated with the same color.

Notably, Max-Pooling only selects the features without requiring any transformation. This ensures that in the pretraining phase, the aggregated video-level feature, aligned with the text embedding space, is aligned with the same space as instance-level features just before being aggregated by GMPool. Each block is designed based on the keypoint pooling method [45], and for detailed information on the MLP and GMPool blocks, please refer to [45].

## 3.4. SM-DCL

To mitigate the pretraining assumption that each video contains only a single action class even when multiple people interact, we propose a novel contrastive learning method called SM-DCL (Fig. 2). This method artificially mixes scenes to ensure features remain discriminative across instances. SM-DCL employs two tensor operations: (a) reshaping the input $\textbf { V } \stackrel { \cdot } { \in } \bar { \mathbb { R } } ^ { \bar { B } \times F S K \times 5 } \mathrm { ~ t o ~ } \hat { \textbf { V } } \in \mathrm { ~ \mathbb { R } ~ } ^ { B / m \times m F S K \times 5 }$ to overlay m scenes (visualized as $m \ : = \ : 2$ in Fig. 2), and (b) reverting the backbone output X<sup>ˆ</sup> back to the unmixed state $\textbf { X } \in$ R $\sum _ { \Delta } \breve { B } \times F S \times D$ before pooling. This allows the loss calculation to remain identical to section 3.2 while enhancing instancelevel discriminability with minimal computational overhead.

## 4. EXPERIMENTS

## 4.1. Datasets and Evaluation Settings

Kinetics-400 is a large-scale action recognition dataset collected from YouTube videos with 400 action classes [47]. We utilize this dataset for pretraining the DNNs. Among the target action classes in the subsequent datasets, Kinetics-400 does not include labels for violent and falling behaviors.

UCF101-24 is a subset of the UCF101 dataset [53], a specific action classification dataset composed of videos from YouTube. This dataset consists of videos belonging to 24 action classes with bounding-box level annotations. We use this dataset to compare the proposed method with conventional weakly-supervised spatio-temporal action localization methods [16, 17]. In line with previous studies of zero-shot action classification [19, 20], the experiment excludes overlapped five action classes to compare performance using Kinetics-400 for pretraining.

Table 1: Performance comparison with conventional methods. (a) spatio-temporal action localization methods on UCF101- 24, (b) tracking-based methods on FDD, and (c) skeleton-based video classification methods on RWF and MF. Conventiona methods are trained in a supervised (†) or weakly-supervised manner (‡).
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>AP (%)</td><td rowspan=1 colspan=1>Learntarget act.in train. (X)or not (√)</td></tr><tr><td rowspan=1 colspan=1>Chéron et al. [17] Anurag et al. [16] </td><td rowspan=1 colspan=1>17.735.0</td><td rowspan=1 colspan=1>x</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>34.1</td><td rowspan=1 colspan=1>√</td></tr></table>

(b) FDD
<table><tr><td>Method</td><td>Frame AP (%)</td><td>Learn target act. in train. (X) or not (√)</td></tr><tr><td>HFDT ↑ CTR-GCN [48] †</td><td>61.6</td><td></td></tr><tr><td>ST-GCN++ [49] †</td><td>36.0 45.0</td><td>x</td></tr><tr><td>Ours</td><td>73.3</td><td>√</td></tr></table>

(c) RWF & MF
<table><tr><td>Method</td><td>RWF MF</td><td>Learn target act. in train. (X) or not (√)</td></tr><tr><td>PointNet++ [3] † DGCNN [3] †</td><td>78.2 89.2 91.3</td><td></td></tr><tr><td>SPIL [3] †</td><td>80.6 89.3 98.5</td><td>x</td></tr><tr><td>Ours</td><td>84.0 92.7</td><td>√</td></tr></table>

Table 2: Performance comparison for ablation studies on UCF101-24 (cf. Sec. 4.3).
<table><tr><td rowspan=1 colspan=1>Architecture</td><td rowspan=1 colspan=1>Training strategy</td><td rowspan=1 colspan=1>AP (%)</td></tr><tr><td rowspan=2 colspan=1>Ours</td><td rowspan=1 colspan=1>3-shot2-shot1-shot</td><td rowspan=1 colspan=1>33.929.923.4</td></tr><tr><td rowspan=1 colspan=1>w/o SM-DCLw/ SM-DCL</td><td rowspan=1 colspan=1>31.234.1</td></tr></table>

Table 3: Comparison with appearance-based methods on RWF-2000.
<table><tr><td>Method</td><td>Acc. (%)</td></tr><tr><td>X-CLIP [50]</td><td>69.3</td></tr><tr><td>EVA [51] ViFi-CLIP [19]</td><td>71.3</td></tr><tr><td>Qwen2.5-VL-7B-Instruct [52]</td><td>72.8 80.0</td></tr><tr><td>Ours</td><td>84.0</td></tr></table>

FDD is a spatio-temporal action localization dataset designed by the authors for comparing a SoTA conventional trackingbased approach $( \mathrm { H F D T ^ { 4 } } )$ that utilizes ST-GCN [7] with the proposed method. Since the supervised fall-action dataset Le2i [54], used to train HFDT, is not publicly available, we captured videos under shooting conditions similar to Le2i. We employ frame average precision (Frame AP) as the evaluation metric for spatio-temporal action localization, followed by the AVA dataset [55]. Among the seven actions, we categorize all labels related to falling behavior collectively as falls.

RWF-2000 and MF are violent action classification datasets. We evaluate the proposed method on these datasets by comparing it with supervised baselines. For fairness, we classify each video using the highest instance score for the violentaction prompt.

For additional details on the implementation and dataset, please refer to the Supplementary Material.

## 4.2. Comparative Experiments on Conventional Methods

Tab. 1 (a) to (c) summarize the performances of the proposed method and the conventional fully and weakly-supervised methods.

As shown in Tab. 1 (a), the proposed method outperforms the conventional weakly-supervised method [17] in terms of spatio-temporal action localization accuracy although UCF101-24 is an advantageous dataset for appearance-based approaches [45].

As shown in Tab. 1 (b), the proposed method considerably outperforms conventional tracking-based methods [30,48,49] in terms of spatio-temporal action localization accuracy. The conventional methods shown in Tabs. 1(b) and (c) use more accurate human pose detectors, AlphaPose $( 7 2 . 0 \mathrm { ~ \ } \% \mathrm { A P _ { k p } ) . }$ RMPE [56] $( 7 2 . 3 \ \% \mathrm { A P _ { k p } ) }$ , and HRNet $( 7 4 . 6 ~ \% \mathrm { { A P _ { k p } } ) }$ , compared with the PPNs $( 6 0 . 6 \% \mathrm { A P _ { k p } ) }$ employed in the proposed method. <sup>5</sup> The discrepancy in the accuracy of human pose detection shows that the tracking-based method yields poor spatio-temporal localization accuracy (Sec. 1).

![](images/99a5b3728525fa95807b8a1dee0569ede984a9cac9334a659420c0b454bb3264.jpg)  
Fig. 3: Robustness against degradation of input videos.

As shown in Tab. 1 (c), the proposed method outperforms many conventional supervised methods, including Point-Net++ [3] and DGCNN [3], in terms of violence action classification accuracy. Its accuracy is also comparable to that of SPIL [3]. This enhanced accuracy is achieved despite leveraging the outputs of our spatio-temporal action localization method, which differs fundamentally from action classification. These results indicate that the proposed method attains an accuracy comparable with existing supervised approaches that employ SoTA architectures for action classification tasks.

In summary, the proposed method successfully achieves zero-shot spatio-temporal action localization with a certain degree of accuracy without target action-specific DNN training and by exclusively relying on video-level annotations as supervision during pretraining. Consequently, it effectively reduces the annotation costs required for pretraining and training new target actions (Sec. 1).

## 4.3. Ablation Studies

In-depth Analysis of Individual Components. Tab. 2 shows the results of quantitatively validating each contribution described in section 1 using the UCF101-24. Rows 2–4 of Tab. 2 (denoted as n-shot in the training strategy) show a comparison of the spatio-temporal action localization accuracy of models trained with few-shot learning using instance-level labels. nshot corresponds to learning from randomly selected n videos per action. Each model is fine-tuned after pretraining on the Kinetics-400 dataset, similar to the proposed method, to ensure fair comparison. The average accuracy is reported for three models evaluated using different few-shot subsets. The accuracy in the three-shot setting (33.9%AP) is comparable with that of the proposed method (34.1%AP in Ours w/ SM-DCL), indicating that weakly-supervised pretraining reduces the annotation cost of manual three-video labeling per action (over ten hours).

SM-DCL. Rows 5–6 of Tab. 2 (denoted as Ours w/ or w/o SM-DCL in the training strategy) show a comparison of the accuracy of models trained with or without SM-DCL. Notably, the proposed method with SM-DCL outperforms that without SM-DCL (34.1%AP vs. 31.2%AP), indicating that the process of mixing scenes enhances the discriminability of actions for each instance during pretraining.

Comparison with video foundation models. Tab. 3 compares appearance-based zero-shot methods and representative large vision-language models (LVLMs) with ours on RWF-2000. Fig. 3 shows how the performance of the appearancebased method [19] and proposed method degrades with decreasing input image quality owing to blur and mask-induced frame drops on this dataset. Because of the diversity of scenes, varying appearances of individuals, and degradation of the input video, the proposed method exhibits higher robustness than appearance-based approaches.

While recent LVLMs typically contain more than 7B parameters and are pretrained on massive Internet-scale corpora, our model contains only about 70M parameters, approximately 100× fewer, and is trained using only Kinetics-400. On RWF-2000, Qwen2.5-VL-7B-Instruct achieves 80% accuracy, which is lower than that of our method. Moreover, Qwen2.5-VL-7B-Instruct runs at approximately 8 FPS, whereas our method achieves approximately 1900

FPS [45], corresponding to about a 240× speedup. These results demonstrate that our method achieves competitive or superior recognition performance with substantially lower computational and pretraining requirements.

## 5. CONCLUSION

This paper proposed a novel skeleton-based zero-shot spatiotemporal action localization method that estimates unseen actions for person instances detected in each video frame, addressing annotation limitations in existing localization methods. The core challenges are threefold: (1) Introducing a novel task—zero-shot spatio-temporal action localization in a weakly-supervised pretraining manner. This involves learning only video-level action labels during the pretraining phase without training via the target actions. (2) Demonstrating the feasibility of such a weakly-supervised pretraining that identifies actions for each instance in the inference phase without requiring instance-level annotations over the pretraining phase and without relying on multi-person tracking methods. (3) Proposing Scene-Mixed Discriminative Contrastive Learning (SM-DCL) to enable the distinction of actions at the instance level within the MIL framework. In the experiments, we evaluated the effectiveness of the proposed method against annotation limitations.

## 6. REFERENCES

[1] I. Rodomagoulakis et al., “Multimodal Human Action Recognition in Assistive Human-robot Interaction,” in ICASSP, 2016.

[2] Sang Uk Lee, Andreas Hofmann, and Brian Williams, “A Model-Based Human Activity Recognition for Human–Robot Collaboration,” in IROS, 2019.

[3] Yukun Su et al., “Human Interaction Learning on 3D Skeleton Point Clouds for Video Violence Recognition,” in ECCV, 2020.

[4] Ming Cheng et al., “RWF-2000: An Open Large Scale Video Database for Violence Detection,” in ICPR, 2021.

[5] Zahidul Islam, Mohammad Rukonuzzaman, Raiyan Ahmed, Md. Hasanul Kabir, and Moshiur Farazi, “Efficient Two-Stream Network for Violence Detection Using Separable Convolutional LSTM,” in IJCNN, 2021.

[6] AJ Piergiovanni et al., “Rethinking Video ViTs: Sparse Video Tubes for Joint Image and Video Learning,” in CVPR, 2023.

[7] Sijie Yan et al., “Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition,” in AAAI, 2018.

[8] Haodong Duan, Yue Zhao, Kai Chen, Dahua Lin, and Bo Dai, “Revisiting Skeleton-Based Action Recognition,” in CVPR, 2022.

[9] Taiki Sekii, “Pose Proposal Networks,” in ECCV, 2018.

[10] Philippe Weinzaepfel and Gregory Rogez, “Mimet-´ ics: Towards Understanding Human Actions out of Context,” IJCV, vol. 129, no. 5, pp. 1675–1690, 2021.

[11] Stephanie Kas, Anton Burenko, Louis Markert,¨ Onur Alp C¸ ulha, Dennis Mack, Timm Linder, and Bastian Leibe, “How do Foundation Models Compare to Skeleton-Based Approaches for Gesture Recognition in Human-Robot Interaction?,” in IEEE RO-MAN, 2025.

[12] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al., “Gemini: a family of highly capable multimodal models,” arXiv preprint arXiv:2312.11805, 2023.

[13] Mohammadreza Salehi et al., “ActionAtlas: A videoqa benchmark for domain-specialized action recognition,” in NeurIPS, 2024.

[14] Limin Wang et al., “VideoMAE V2: Scaling Video Masked Autoencoders With Dual Masking,” in CVPR, 2023.

[15] Jiaojiao Zhao, Yanyi Zhang, Xinyu Li, Hao Chen, Bing Shuai, Mingze Xu, Chunhui Liu, Kaustav Kundu, Yuanjun Xiong, Davide Modolo, Ivan Marsic, Cees G. M. Snoek, and Joseph Tighe, “TubeR: Tubelet Transformer for Video Action Detection,” in CVPR, 2022.

[16] Anurag Arnab et al., “Uncertainty-Aware Weakly Supervised Action Detection from Untrimmed Videos,” in ECCV, 2020.

[17] Guilhem Cheron et al., “A Flexible Model for Train-´ ing Action Localization with Varying Levels of Supervision,” in NeurIPS, 2018.

[18] Alec Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” in ICML, 2021.

[19] Hanoona Rasheed et al., “Fine-Tuned CLIP Models Are Efficient Video Learners,” in CVPR, 2023.

[20] Wei Lin, Leonid Karlinsky, Nina Shvetsova, Horst Possegger, Mateusz Kozinski, Rameswar Panda, Rogerio Feris, Hilde Kuehne, and Horst Bischof, “MAtch, eXpand and Improve: Unsupervised Finetuning for Zero-Shot Action Recognition with Language Knowledge,” in ICCV, 2023.

[21] Guy Tevet, Brian Gordon, Amir Hertz, Amit H Bermano, and Daniel Cohen-Or, “MotionCLIP: Exposing Human Motion Generation to CLIP Space,” in ECCV, 2022.

[22] Sauradip Nag et al., “Zero-shot temporal action detection via vision-language prompting,” in ECCV, 2022.

[23] Wei-Jhe Huang et al., “Interaction-Aware Prompting for Zero-Shot Spatio-Temporal Action Detection,” in IC-CVW, 2023.

[24] Wentao Bao, Kai Li, Yuxiao Chen, Deep Patel, Martin Renqiang Min, and Yu Kong, “Exploiting VLM Localizability and Semantics for Open Vocabulary Action Detection,” in WACV, 2025.

[25] Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas, “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation,” in CVPR, 2017.

[26] Manzil Zaheer et al., “Deep Sets,” in NeurIPS, 2017.

[27] Thomas G. Dietterich et al., “Solving the Multiple Instance Problem with Axis-parallel Rectangles,” Artificial Intelligence, vol. 89, no. 1, pp. 31–71, 1997.

[28] Gurkirt Singh, Suman Saha, and Fabio Cuzzolin, “TraMNet - Transition Matrix Network for Efficient Action Tube Proposals,” in ACCV, 2018.

[29] Gurkirt Singh, Suman Saha, Michael Sapienza, Philip H. S. Torr, and Fabio Cuzzolin, “Online Real-Time Multiple Spatiotemporal Action Localisation and Prediction,” in ICCV, 2017.

[30] “Human Falling Detection and Tracking,” https://github.com/GajuuzZ/ Human-Falling-Detect-Tracks, 2021.

[31] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut, “Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts,” in CVPR, 2021.

[32] Paola Cascante-Bonilla, Hui Wu, Letao Wang, Rogerio S. Feris, and Vicente Ordonez, “SimVQA: Exploring Simulated Environments for Visual Question Answering,” in CVPR, 2022.

[33] Vipul Gupta, Zhuowan Li, Adam Kortylewski, Chenyu Zhang, Yingwei Li, and Alan Yuille, “SwapMix: Diagnosing and Regularizing the Over-Reliance on Visual Context in Visual Question Answering,” in CVPR, 2022.

[34] Chen Ju, Tengda Han, Kunhao Zheng, Ya Zhang, and Weidi Xie, “Prompting Visual-Language Models for Efficient Video Understanding,” in ECCV, 2022.

[35] Wei-Jhe Huang, Min-Hung Chen, and Shang-Hong Lai, “Spatio-Temporal Context Prompting for Zero-Shot Action Detection,” in WACV, 2025.

[36] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz, “mixup: Beyond Empirical Risk Minimization,” in ICLR, 2018.

[37] Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada, “Between-Class Learning for Image Classification,” in CVPR, 2018.

[38] Sangdoo Yun et al., “CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features,” in ICCV, 2019.

[39] Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, David Lopez-Paz, and Yoshua Bengio, “Manifold Mixup: Better Representations by Interpolating Hidden States,” in ICML, 2019.

[40] Puneet Mangla, Neelabh Kumari, Aayush Sinha, Mayank Singh, Bala Krishnamurthy, and Piyush Rai, “Charting the Right Manifold: Manifold Mixup for Few-Shot Learning,” in WACV, 2020.

[41] Rui Xu, Xinchao Zhang, Pingping Cui, Jianfei Cheng, and Xiaojie Liu, “Adversarial Domain Adaptation with Domain Mixup,” Pattern Recognition Letters, vol. 136, pp. 316–322, 2020.

[42] Sungnyun Kim, Gihun Lee, Sangmin Bae, and Se-Young Yun, “Mixco: Mix-up Contrastive Learning for Visual Representation,” in NeurIPSW, 2020.

[43] Minsoo Kang and Suhyun Kim, “GuidedMixup: an efficient mixup strategy guided by saliency maps,” in AAAI, 2023.

[44] Khawar Islam, Muhammad Zaigham Zaheer, Arif Mahmood, and Karthik Nandakumar, “DiffuseMix: Label-Preserving Data Augmentation with Diffusion Models,” in CVPR, 2024.

[45] Ryo Hachiuma et al., “Unified Keypoint-based Action Recognition Framework via Structured Keypoint Pooling,” in CVPR, 2023.

[46] Kaitao Song et al., “MPNet: Masked and Permuted Pre-training for Language Understanding,” in NeurIPS, 2020.

[47] Joao Carreira and Andrew Zisserman, “Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset,” in CVPR, 2017.

[48] Yuxin Chen et al., “Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition,” in ICCV, 2021.

[49] Haodong Duan et al., “Pyskl: Towards good practices for skeleton action recognition,” in ACMMM, 2022.

[50] Bolin Ni et al., “Expanding Language-Image Pretrained Models for General Video Recognition,” in ECCV, 2022.

[51] Yuxin Fang et al., “EVA: Exploring the Limits of Masked Visual Representation Learning at Scale,” in CVPR, 2023.

[52] Shuai Bai et al., “Qwen2.5-VL Technical Report,” arXiv preprint arXiv:2502.13923, 2025.

[53] Khurram Soomro et al., “UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild,” CoRR, vol. abs/1212.0402, 2012.

[54] Imen Charfi et al., “Optimized spatio-temporal descriptors for real-time fall detection: Comparison of support vector machine and Adaboost-based classification,” Journal ofElectronic Imaging, vol. 22, no. 4, pp. 041106, 2013.

[55] Chunhui Gu et al., “AVA: A Video Dataset of Spatio-Temporally Localized Atomic Visual Actions,” in CVPR, 2018.

[56] Hao-Shu Fang et al., “RMPE: Regional Multi-Person Pose Estimation,” in ICCV, 2017.

[57] Gurkirt Singh, Suman Saha, Michael Sapienza, Philip H. S. Torr, and Fabio Cuzzolin, “Online Real-Time Multiple Spatiotemporal Action Localisation and Prediction,” in ICCV, 2017.

[58] Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, and Alex C Kot, “NTU RGB+D 120: A large-scale benchmark for 3D human activity understanding,” PAMI, vol. 42, no. 10, pp. 2684–2701, 2020.

[59] Enrique Bermejo Nievas et al., “Movies Fight Detection Dataset,” in CAIP, 2011.

[60] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang, “Deep High-Resolution Representation Learning for Human Pose Estimation,” in CVPR, 2019.

[61] Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, and Cordelia Schmid, “Action Tubelet Detector for Spatio-Temporal Action Localization,” in ICCV, 2017.

[62] Sergey Ioffe and Christian Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,” in ICML, 2015.

[63] Dan Hendrycks and Kevin Gimpel, “Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units,” CoRR, vol. abs/1606.08415, 2016.

[64] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou, “MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers,” in NeurIPS, 2020.

[65] Yun Wang, Juncheng Li, and Florian Metze, “A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling,” in ICASSP, 2019.

[66] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stephane Deny, “Barlow Twins: Self-Supervised Learn-´ ing via Redundancy Reduction,” in ICML, 2021.

[67] Dong Li, Zhaofan Qiu, Qi Dai, Ting Yao, and Tao Mei, “Recurrent Tubelet Proposal and Recognition Networks for Action Detection,” in ECCV, 2018.

[68] Jiaojiao Zhao and Cees G. M. Snoek, “Dance With Flow: Two-In-One Stream Action Detection,” in CVPR, 2019.

[69] Haodong Duan, Mingze Xu, Bing Shuai, Davide Modolo, Zhuowen Tu, Joseph Tighe, and Alessandro Bergamo, “SkeleTR: Towards Skeleton-based Action Recognition in the Wild,” in ICCV, 2023.

[70] Junting Pan, Siyu Chen, Mike Zheng Shou, Yu Liu, Jing Shao, and Hongsheng Li, “Actor-Context-Actor Relation Network for Spatio-Temporal Action Localization,” in CVPR, 2021.

[71] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He, “SlowFast Networks for Video Recognition,” in ICCV, 2019.

[72] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang, “VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training,” in NeurIPS, 2022.

[73] Victor Escorcia, Cuong D. Dao, Mihir Jain, Bernard Ghanem, and Cees Snoek, “Guess Where? Actorsupervision for Spatiotemporal Action Localization,” CVIU, vol. 192, pp. 102886, 2020.

[74] Sovan Biswas and Jurgen Gall, “Multiple Instance ¨ Triplet Loss for Weakly Supervised Multi-Label Action Localisation of Interacting Persons,” in ICCVW, 2021.

[75] Robert J. Wang, Xiang Li, and Charles X. Ling, “Pelee: A Real-Time Object Detection System on Mobile Devices,” in NeurIPS, 2018.

[76] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and´ C. Lawrence Zitnick, “Microsoft COCO: Common Objects in Context,” in ECCV, 2014.

[77] Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh, “OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields,” PAMI, vol. 43, no. 1, pp. 172–186, 2021.

[78] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” in NeurIPS, 2015.

[79] Shen Yan, Xuehan Xiong, Anurag Arnab, Zhichao Lu, Mi Zhang, Chen Sun, and Cordelia Schmid, “Multiview Transformers for Video Recognition,” in CVPR, 2022.

[80] Zhen Xing, Qi Dai, Han Hu, Jingjing Chen, Zuxuan Wu, and Yu-Gang Jiang, “SVFormer: Semi-Supervised Video Transformer for Action Recognition,” in CVPR, 2023.

[81] Jiayi Shao, Xiaohan Wang, Ruijie Quan, Junjun Zheng, Jiang Yang, and Yi Yang, “Action Sensitivity Learning for Temporal Action Localization,” in ICCV, 2023.

[82] Daisuke Miki, Shi Chen, and Kazuyuki Demachi, “Weakly Supervised Graph Convolutional Neural Network for Human Action Localization,” in WACV, 2020.

# SKELETON-BASED ZERO-SHOT SPATIO-TEMPORAL ACTION LOCALIZATION VIA WEAKLY-SUPERVISED PRETRAINING

Supplementary Material

## 6. EXPERIMENTS

## 6.1. Dataset

Kinetics-400. Kinetics-400 [47] is a large-scale action recognition dataset collected from YouTube<sup>6</sup> videos with 400 action classes. It contains 250K training and 19K validation 10-second video clips with 30 fps.

UCF101-24. The UCF101-24 dataset [53] is a subset of the UCF101 dataset, a specific action classification dataset composed of videos from YouTube. UCF101-24 consists of videos belonging to 24 action classes and comprises 3.2K videos. Additionally, its 24 class action labels are annotated for each bounding box in the videos. We use the corrected annotation [57] according to the standard practice [16, 17].

FDD. The FDD dataset is a spatio-temporal action localization dataset designed by the authors for comparing a SoTA conventional tracking-based approach (HFDT) [30] that utilizes ST-GCN [7] with the proposed method. The supervised fall action classification dataset, Le2i [54], is used in the training of the conventional method. This dataset is not publicly available at the moment. Thus, in the FDD dataset, we captured videos under shooting conditions similar to those of the Le2i dataset, including the size of the subjects, camera angle, and indoor environment, to compare localization accuracy fairly between the conventional and proposed methods.

The dataset comprises 62 video clips, of which 33 feature individuals simulating falls. Each clip lasts approximately 15 seconds and is recorded under conditions that ensure consistent visibility of 1 to 2 individuals in each video. The seven action labels, which semantically subdivide falls and non-falls similar to Le2i, are annotated for each bounding box in the videos.

Note that HFDT [30] is trained on the Le2i, while CTR-GCN [48] and ST-GCN++ [49] are trained on NTU RGB+D 120 [58], a relatively large-scale action classification dataset that includes fall actions.

RWF-2000. RWF-2000 [4] is the violent action classification dataset collected from YouTube videos. The videos feature two actions, violent or non-violent, captured by security cameras with different people and backgrounds. There are 1.6K training and 0.4K test 5-second video clips with 30 fps. Twoclass labels are annotated for each video.

MF. The MF dataset [59] is a collection of 200 video clips extracted from action movies, specifically designed for assessing violent action classification. A wider variety of fight scenes were captured at different resolutions, and nonfight videos were extracted from public action-recognition

datasets.

## 6.2. Evaluation Settings

UCF101-24 is used to compare the proposed method with conventional weakly-supervised spatio-temporal action localization methods [16,17]. Similar to conventional methods, we use video average precision (Video AP) (%) with 3D IoU=0.5 as the evaluation metric for the spatio-temporal action localization. We also employ the tubelet generated by Cheron et ´ al. [17] and link the tubelets using the linking algorithm [61], followed by the conventional method [16].

FDD is used to compare the conventional tracking-based methods [30, 48, 49] with the proposed method. We employ frame average precision (Frame AP) as the evaluation metric for spatio-temporal action localization, followed by the AVA dataset [55]. Among the seven actions, we categorize all labels related to falling behavior collectively as falls.

RWF-2000 and MF. Recently, skeleton-based methods have been actively studied in the field of action classification. In contrast to spatio-temporal action localization, SoTA action classification architectures are subject to frequent updates. We employ this dataset to validate the effectiveness of the proposed method by comparing its classification accuracy with that of conventional supervised action classification methods. For a fair comparison with the conventional methods, the proposed method classifies each video as either violent or non-violent using the highest instance score for violent action prompts within the video.

## 6.3. Implementation Details

## 6.3.1. Parameters of SLPS DNN Architecture

The dimension of the output feature vector at the point embedding block is set to 256 and those of the input feature vector at the MLP blocks are set to 256, 512, and 1024 because the Grouped Pool and GMPool blocks double the channels via concatenation. The MLP expansion ratio is set to 2.0. Batch normalization [62] is employed as the normalization layer, and GELU [63] is employed as the activation function in the MLP block.

## 6.3.2. Grouped pool block.

The Grouped Pool Block consists of GMPool $\phi _ { G }$ and Local Max-Pooling (LMPool) $\phi _ { L }$ , which applies Max-Pooling to each local group of the same instance to which the input feature vector per keypoint belong. The Grouped Pool Block outputs F S feature vectors containing the number of instances in the video. The Grouped Pool Block can be expressed as follows:

Table 4: Hyperparameters of each dataset during pretraining.
<table><tr><td>Evaluation dataset</td><td colspan="4">UCF101-24 [53] and MF [59] FDD and RWF-2000 [4]</td></tr><tr><td>Pretraining dataset</td><td colspan="4">Kinetics-400 [47] PPNs [9]</td></tr><tr><td>Pose Detector Mixing ratio</td><td colspan="3">HRNet [60] 1 0.6</td><td>0.6</td></tr><tr><td>of the loss functions α</td><td colspan="4">Stochastic Gradient Descent</td></tr><tr><td>Optimizer</td><td colspan="4">40</td></tr><tr><td>Number of epochs</td><td></td><td></td><td>150</td><td>40</td></tr><tr><td>Batch size</td><td></td><td></td><td>480</td><td>256</td></tr><tr><td>Learning rate</td><td></td><td></td><td>0.48</td><td>1.6</td></tr><tr><td>Weight decay</td><td>1.6 0.0000125</td><td colspan="2">0.00005</td><td>0.0001</td></tr><tr><td>Momentum</td><td colspan="4">0</td></tr><tr><td>LR scheduler</td><td colspan="4">linear</td></tr><tr><td>Joint scaling</td><td colspan="4">[0.8, 1.2]</td></tr><tr><td>Joint shift</td><td colspan="4">0.2</td></tr><tr><td>Joint rotate (°)</td><td colspan="4">5</td></tr><tr><td>Joint flip ratio</td><td colspan="4">0.5</td></tr><tr><td>Temporal crop window</td><td colspan="4"></td></tr><tr><td>Temporal FPS drop</td><td colspan="4">100</td></tr></table>

$$
\mathbf { U } _ { \mathrm { o u t } } = \left\{ \left[ \phi _ { L } \left( \mathbf { U } _ { \mathrm { i n } , s } \right) , \phi _ { G } \left( \mathbf { U } _ { \mathrm { i n } } \right) \right] \right\} _ { s \in \left\{ 1 , \ldots , F S \right\} } .\tag{5}
$$

${ \bf U } _ { \mathrm { i n } } \in \mathbb { R } ^ { F S K \times D }$ and $\mathbf { U } _ { \mathrm { o u t } } \in \mathbb { R } ^ { F S \times 2 D }$ are the matrices of the input and output feature vectors, respectively, as described below. Then, the input feature vector $\mathbf { u } _ { \mathrm { i n } } \in \mathbf { U } _ { \mathrm { i n } }$ per keypoint is grouped into $F S$ instances $\left( \mathbf { u } _ { \mathrm { i n } } \in \mathbb { R } ^ { 1 \times D } \right)$ , and $\mathbf { U } _ { \mathrm { i n } }$ can be expressed by a concatenated matrix as follows:

$$
{ \bf { U } } _ { \mathrm { { i n } } } = \left( { \bf { u } } _ { \mathrm { { i n } , 1 } } ; \ldots ; { \bf { u } } _ { \mathrm { { i n } } , F S K } \right) = \left( { \bf { U } } _ { \mathrm { { i n } , 1 } } ; \ldots ; { \bf { U } } _ { \mathrm { { i n } } , F S } \right) ,\tag{6}
$$

where K is the number of keypoints. $\mathbf { U } _ { \mathrm { i n } , s }$ represents a submatrix of $\mathbf { U } _ { \mathrm { i n } }$ for each instance s. Consequently, $\mathbf { U _ { \mathrm { o u t } } }$ is computed using the output vector $\mathbf { u } _ { \mathrm { o u t } } \in \mathbb { R } ^ { 1 \times \bar { D } }$ as follows:

$$
\mathbf { U } _ { \mathrm { o u t } } = \left( \mathbf { u } _ { \mathrm { o u t , 1 } } ; \ldots ; \mathbf { u } _ { \mathrm { o u t } , F S } \right) ^ { T } .\tag{7}
$$

In Eq. (5), we concatenate each feature vector ϕ $( \mathbf { U } _ { \mathrm { i n } , s } ) \in$ $\mathbb { R } ^ { 1 \times D }$ computed for the instance group s and the global feature vector $\phi _ { G } \left( \mathbf { U } _ { \mathrm { i n } } \right) \ \in \ \mathbb { R } ^ { 1 \times D }$ in a channel dimension. Moreover, LMPool $\phi _ { L } ( \cdot )$ can be expressed as follows:

$$
\phi _ { L } \left( \mathbf { U } _ { \mathrm { i n } , s } \right) = \mathrm { M a x P o o l } ( \mathbf { U } _ { \mathrm { i n } , s } ) ,\tag{8}
$$

where MaxPool(·) is the operation used to obtain the max value for each channel from the feature vectors matrices. $\phi _ { G } ( \cdot )$ is expressed as follows:

$$
\phi _ { G } ( \mathbf { U } _ { \mathrm { i n } } ) = \mathrm { M a x P o o l } ( \mathbf { U } _ { \mathrm { i n } } ) .\tag{9}
$$

In the pretraining phase, given the instance-level feature $( \mathbf { x } _ { 1 } , \hdots , \mathbf { x } _ { F S } )$ , GMPool aggregates these features in the video-level feature z as follows:

$$
\mathbf { z } = \operatorname { M a x P o o l } \left( \mathbf { x } _ { 1 } ; \ldots ; \mathbf { x } _ { F S } \right) .\tag{10}
$$

Note that GMPool operation in Eq. (10) is removed in the inference phase, based on the Pooling-Switching.

![](images/50a8507656467d930b0b2fe1263859da965417db9580dbd3331cf17074aa8653.jpg)  
Fig. 4: Qualitative zero-shot spatio-temporal action localization results of the proposed method on UCF101- 24 (cf. section 4.2). The input keypoints and the user prompts are visualized in the figure. Our approach does not require instance-level annotations during training or multi-person tracking methods, yet it can localize actions spatio-temporally.

## 6.3.3. Data Augmentation

We apply two types of data augmentation during pretraining: augmentation onto the image space and along the temporal axis to the input joints. We then randomly scale, shift, rotate, and flip the joint coordinates for augmentation onto the image space and randomly crop the input joints with a random size of the temporal window. Additionally, we drop joints within a random interval range.

## 6.3.4. Hyperparameters

Tab. 4 shows the hyperparameters employed in each experiment. We use stochastic gradient descent for DNN pretraining with a learning rate that decreases with the number of epochs. The model is pretrained for 150 epochs on Kinetics-400 with α = 1. After these epochs, we set α to 0.6 to pretrain the model for 40 epochs. In SM-DCL (section 3.4), m is set to 2. Furthermore, as shown in Tabs. 5 and 6, we experimented with various numbers of mixed scenes and different mixing timings in SM-DCL, and adopted the parameters that yielded the highest accuracy. The hyperparameters are simply found in a standard coarse-to-fine grid search or step-by-step tuning.

## 6.4. Ablation Studies

Single-action assumption per video. During pretraining, the proposed method assumes that each video contains only one action. However, many Kinetics-400 videos used for pretraining include scenes in which multiple people perform distinct actions (approximately 20%). Moreover, we conducted an experiment during pretraining to assess the accuracy degradation of the proposed method when the skeletons from another randomly selected video are added as noise. As shown in Tab. 7, the accuracy degradation is only 1.2% points on the UCF101-24 dataset, indicating that the assumption is not essential for the proposed method.

Table 5: Accuracy degradation with studies on increasing number of mixed scenes on UCF101-24.
<table><tr><td># mixed scenes</td><td>1</td><td>2</td><td>4</td></tr><tr><td>AP (%)</td><td>31.2</td><td>34.1</td><td>30.4</td></tr></table>

Table 6: Ablation study on varying the starting epoch for applying SM-DCL using UCF101-24.

![](images/538b6c98a72cc52ff64fb528571bc4b2beb431aafb640c39319648a14047d17b.jpg)  
Table 9: Ablation study on text encoders on UCF101-24.

<table><tr><td>Text encoder</td><td>AP (%)</td></tr><tr><td>MiniLM [64]</td><td>29.1</td></tr><tr><td>CLIP ViT-L/14 [18]</td><td>29.8</td></tr><tr><td>MPNet [46]</td><td>31.2</td></tr></table>

Table 10: Ablation study on scaling up the pretraining dataset using UCF101-24.

Fig. 5: Qualitative results for skeletons classified as violent behavior by the proposed method in RWF-2000.
<table><tr><td>Starting epoch</td><td>0</td><td>20</td><td>35</td></tr><tr><td>AP (%)</td><td>27.3</td><td>34.1</td><td>33.2</td></tr></table>

<table><tr><td>Ratio of scale</td><td>0.25</td><td>0.5</td><td>1.0</td></tr><tr><td>AP (%)</td><td>21.0</td><td>23.1</td><td>31.2</td></tr></table>

Table 11: Ablation study on text prompts using RWF-2000.

Table 7: Accuracy degradation with an increasing number of mixed scenes on UCF101-24.
<table><tr><td># mixed scenes Average # people</td><td>1 1.2</td><td>2 2.4</td></tr><tr><td>AP (%)</td><td>31.2</td><td>30.0</td></tr></table>

Table 8: UCF101-24 average accuracy for the three action classes with the lowest similarity to those included in the pretraining dataset.
<table><tr><td>Action class</td><td>AP (%)</td></tr><tr><td>Selected three classes</td><td>25.7</td></tr><tr><td>All</td><td>31.2</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Sentence</td><td rowspan=1 colspan=1>Acc. (%)</td></tr><tr><td rowspan=1 colspan=1>push or punchrelated toviolent scene</td><td rowspan=1 colspan=1>83.5</td></tr><tr><td rowspan=1 colspan=1>push, kick, punchor drag relatedto violent scene</td><td rowspan=1 colspan=1>82.0</td></tr><tr><td rowspan=1 colspan=1>push, kickor punch relatedto violent scene</td><td rowspan=1 colspan=1>84.0</td></tr></table>

![](images/e577eb4b295e38e7b260a2b09b0271a4b1d5fa5ad04a3f073c8375e7b7b36c96.jpg)  
Fig. 6: Qualitative zero-shot spatio-temporal action localization results of the proposed method on the FDD dataset. The input keypoints, including the fall behaviors, are visualized in the figure.

Table 12: Ablation study on the mixing ratios of action classification loss and contrastive loss.  
Table 13: Ablation study on pooling switching mechanism on UCF101-24.
<table><tr><td>Loss weight</td><td>0.0</td><td>0.2</td><td>0.4</td><td>0.6</td><td>0.8</td></tr><tr><td>AP (%)</td><td>24.8</td><td>30.4</td><td>30.5</td><td>31.2</td><td>30.1</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>AP (%)</td></tr><tr><td rowspan=1 colspan=1>Ave. Pool. [65]Attention [65]</td><td rowspan=1 colspan=1>30.626.4</td></tr><tr><td rowspan=1 colspan=1>Max Pool. (Ours)</td><td rowspan=1 colspan=1>31.2</td></tr></table>

Generalization to outlier actions. Among the actions shown in Tab. 1, some outlier actions do not resemble those learned during pretraining. Therefore, we analyzed the accuracy specifically for these outlier actions. We compared the text feature representations of the action class names in the UCF101-24 dataset with those derived during pretraining. Tab. 8 shows the accuracy for the three outlier actions with the lowest similarity. As the average accuracy for the outlier actions remains within a few percentage points of that for all action classes, we can conclude that the proposed method successfully detects outlier action classes.

Analysis of SM-DCL In the SM-DCL method (section 3.4), we set m, i.e., the number of mixed scenes, to 2 based on experiments with different numbers of mixed scenes (Tab. 5) and various mixing timings (Tab. 6). We then adopted the settings that achieved the highest accuracy.

Table 14: Ablation study on contrastive learning loss function using UCF101-24.
<table><tr><td>Method</td><td>AP (%)</td></tr><tr><td>L2 norm loss</td><td>28.8</td></tr><tr><td>Barlow Twins [66]</td><td>29.9</td></tr><tr><td>Ours</td><td>31.2</td></tr></table>

Analysis of Text Encoder Effectiveness Tab. 9 summarizes the differences in the performance of the proposed method on the UCF101-24 dataset using different text encoders. Results show the effectiveness of the proposed method with different module combinations.

Scaling up pretraining dataset We compared the accuracy of the proposed method on the UCF101-24 dataset using several subsets extracted at different ratios from the pretraining dataset (Kinetics-400). Tab. 10 shows that the accuracy of the proposed method increases as the number of subsets used for pretraining gradually increases.

Table 15: Summary of spatio-temporal action localization approaches.
<table><tr><td>Method</td><td>Input information</td><td>Learn target actions in training (X) or not (√)</td><td>Supervision at instance (X) or video (√) level</td><td>Multi-person tracking free</td></tr><tr><td>[28,29,61,67,68] [69]</td><td>RGB Skeleton</td><td rowspan="4">x</td><td>x</td><td>x x</td></tr><tr><td>[14,15,70–72]</td><td>RGB</td><td></td><td>√</td></tr><tr><td>[16,17,73]</td><td>RGB</td><td rowspan="2">√</td><td>x</td></tr><tr><td>[74]</td><td></td><td>√</td></tr><tr><td>[30]</td><td>RGB Skeleton</td><td></td><td></td><td>x</td></tr><tr><td>[23,24,35]</td><td>RGB</td><td>√</td><td>x</td><td>√</td></tr><tr><td>Ours</td><td>Skeleton</td><td>√</td><td>√</td><td>√</td></tr></table>

Text prompts. Tab. 11 shows the violent recognition accuracy of the proposed method using four text prompts based on the RWF-2000 dataset. Results show that the proposed method can localize unknown actions with some accuracy, regardless of the variations in text prompts.

Mixing ratios of the action classification loss and the contrastive loss. Tab. 12 shows the spatio-temporal action localization accuracy of the proposed method using different mixing ratios of action classification and contrastive losses based on the UCF101-24 dataset. Result shows that the proposed method that uses both loss terms achieves higher accuracy than when only the contrastive loss (α = 0) is employed, indicating the effectiveness of incorporating the action classification loss in the pretraining phase.

Analysis of individual components. The proposed method is designed based on ablation studies that could not be included in the main paper due to space constraints. Although the details are presented in the final version (Tabs. 13 and 14), the pooling switching mechanism and contrastive learning loss function are adopted after comparing multiple modules.

## 7. HUMAN POSE DETECTORS

In this section, we provide detailed information about the pose detectors.

PPNs. PPNs [9] detect human skeletons in a bottom-up manner from an RGB image at high speed. They comprise a ResNet-101 backbone [75] and trained using the MS-COCO dataset [76]. The definition of the human skeleton is the same as OpenPose [77]. As inputs to PPNs, we resize video frames to 640 × 480 px<sup>2</sup>.

HRNet. HRNet [60] is a top-down human pose detector. It achieves superior accuracy, and the computational cost includes a human detector (Faster R-CNN [78]), which is expensive. In the experiments on UCF101-24, we employ publicly available HRNet skeletons<sup>7</sup> by Haodong et al. [8].

## 8. QUALITATIVE RESULTS ON THE FDD DATASET

Fig. 6 shows that the qualitative results of action localization obtained using the FDD dataset. The proposed method localizes the falling actions of each person in each frame in a zero-shot manner.

## 9. RELATED WORK

Herein, we briefly review related work in action recognition to provide context for our contributions. Additionally, Tab. 15 compares the features of the proposed method with related works, including those discussed in section 2.

## 9.1. Action Recognition

As mentioned in section 1, three primary tasks have been extensively pursued in the field of action recognition: classification [3,6–8,45,79,80], temporal localization [81,82], and spatio-temporal localization [14–17, 23, 28–30, 61, 67–74] of actions. This study focuses on a skeleton-based spatiotemporal action localization task, wherein actions are recognized for each skeleton detected in every frame. To extract spatio-temporal features in a data-driven manner from videos over the three tasks, appearance-based approaches employ 3D convolutional neural networks (CNNs) for spatiotemporal convolutions [70,71] and transformers for capturing a global context of feature relationships across space and time [6, 14, 15, 72, 79] given the recent advancement of DNN. In contrast, skeleton-based approaches use graph convolutional networks [7, 69] and DNNs inspired by a point cloud deep learning paradigm [3, 45] to model the motion features from input skeleton sequences. The proposed method exploits the skeleton-based approach, which is more resistant to changes in a person’s appearance or background due to training [10].