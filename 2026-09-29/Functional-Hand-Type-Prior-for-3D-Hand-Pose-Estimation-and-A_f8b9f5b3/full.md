# Functional Hand Type Prior for 3D Hand Pose Estimation and Action Recognition from Egocentric View Monocular Videos

Wonseok Roh<sup>1</sup>   
6paulroh@korea.ac.kr   
<sup>2</sup>Seung Hyun Lee<sup>1</sup>   
easter3163@korea.ac.kr   
Won Jeong Ryoo<sup>1</sup>   
epetac@korea.ac.kr   
<sup>S</sup>Jakyung Lee<sup>1</sup>   
<sup>8</sup>2023020917@korea.ac.kr   
<sup>2</sup>Gyeongrok Oh<sup>1</sup>   
]dhrudfhr98@korea.ac.kr   
V<sub>Sooyeon</sub> <sub>Hwang</sub>1   
C<sub>hsy506@korea.ac.kr</sub>   
<sup>s</sup>Hyung-gun Chi<sup>2</sup>   
<sub>[</sub>chi45@purdue.edu   
Sangpil Kim<sup>1∗</sup>   
spk7@korea.ac.kr

<sup>1</sup> Department of Artificial Intelligence, Korea University, Seoul, Republic of Korea

<sup>2</sup> Electrical and Computer Engineering, Purdue University, West Lafayette, Indiana, USA

## Abstract

Current methods for egocentric view action recognition often face challenges in perceiving dynamic hand movements relying solely on geometrical or physical information. In this work, we effectively address this problem by gaining insights into the correlation between functional hand configurations and objects, which improves the detailed interpretation of real-world scenarios. To this end, we introduce a practical taxonomy of hand types based on the functioning perspective and utilize it for per-frame hand type labeling on existing datasets. We also propose a novel hand action recognition framework considering semantic details of the hand type as prior. This approach boosts the network’s understanding of the continuous hand interaction throughout the action sequence. Our whole pipeline consists of three main modules: (1) Feature Extraction, (2) Egocentric Knowledge Module, which estimates 3D hand pose, object category, and hand type leveraging short-term cues, and (2) Egocentric Action Module, which aggregates per-frame knowledge, including text embeddings of hand type, over a longer time. In our extensive experiments with large-scale benchmarks, FPHA and H2O, our model outperforms current state-of-the-art methods, demonstrating its superior performance.

![](images/c6367b18cec8d7ff2bda379028a1f647211ff542c94b4c047cf3af1a84acb793.jpg)  
Figure 1: Examples of actions and corresponding hand types from the FPHA dataset [15]. While each action shares the goal of opening objects, the specific interactions between hands and each object are different. Further, both activities involve holding and opening the lid, but the semantic functions of these motions differ significantly. To highlight the continuously repeated shape during the operation, we represent it using “Dynamic”.

## 1 Introduction

Understanding dynamic interacting human hands is a promising computer vision task because people use their hands to handle objects and communicate with others throughout daily activities. Recently, various hand-based visual applications have been proposed, such as human-robot collaboration [13, 14, 33, 47] and imitation learning [31, 36, 37]. Moreover, the advancement of low-cost wearable sensors and VR/AR technologies motivates the computer vision community to tackle egocentric view hand action recognition.

In real-world egocentric view scenarios, substantial occlusion and truncation often occur, especially when the hand actively interacts with other hands or objects. To address these issues, recent approaches [8, 24, 26, 28, 40, 43, 44, 49, 51, 52] in hand action recognition primarily focus on the temporal context of 3D hand position or high-level object labels. However, these methods still face challenges in perceiving dynamic hand movements relying solely on geometrical information. In this work, we go beyond mere physical information and focus on the semantical hand interactions to provide valuable details for a comprehensive understanding of hand action sequences. For example, as illustrated in Fig. 1, there are semantic differences between everyday hand actions such as “Open Juice Bottle” and “Open Liquid Soap”. Although each action shares the goal of opening objects, the specific interactions between hands and each object differ. Therefore, gaining insights into the semantic relationships between functional hand configurations and other hands or various objects is essential for addressing complex real-world hand scenarios. From this observation, we consider utilizing contextual knowledge of hand types to understand complex hand interactions above simply relying on the physical hand positions.

One of the best ways to specify hand type at semantic level is hand type taxonomy. The concept of hand type represents a figurative expression of human intention when conducting tasks with hands. Most previous studies [2, 6, 11, 12, 34, 39] organize hand type taxonomies based on the properties of objects since humans usually use the similar grasp types for specific objects. In other words, they design and utilize hierarchical hand type categorization criteria based on object grasp manner. Although this hierarchy helps explain the appearance of the hand, this approach overlooks the importance of hand functionality or how the hand operates in various tasks. The function of the hand plays a crucial role in effectively understanding each action, as it accurately reflects the hand-object correlation as well as the intention behind the movement. To this end, we carefully redefine a practical taxonomy of hand types focused on the functioning perspective for diverse hand-related vision tasks, including egocentric view action recognition. Then, we supervise the hand action recognition network with hand type labels annotated with newly defined taxonomy.

Built upon [49] framework, which considers the high-level action recognition task as a mixture of two low-level tasks: 3D hand pose estimation and object classification, we especially utilize hand type estimation network to provide valuable semantic cues. Our model consists of three main modules: (1) Feature Extraction, (2) Egocentric Knowledge Module, which estimates 3D hand pose, object category, and hand type leveraging short-term cues, and (3) Egocentric Action Module, which aggregates per-frame knowledge, including text embeddings of hand type, with long-term cues. To the best of our knowledge, we are the first to adopt the usefulness of semantic cues of hand type for hand action recognition. Our proposed framework outperforms the existing state-of-the-art works on 3D hand pose estimation and action recognition. To summarize, our main contributions are listed as follows:

• We newly define the practical taxonomy of functional hand types based on the functioning perspective for diverse hand-related vision tasks, including hand action recognition. We also provide precise hand type annotations for existing datasets.

• We present a novel hand action recognition framework utilizing semantic details of the hand type in each frame. This approach guides the entire network to learn deep understandings of the continuous interaction across hands and objects throughout the action sequence. To the best of our knowledge, our work is the first to leverage the semantic knowledge of hand type in the context of hand perception.

• We analyze the effectiveness of our proposed hand type prior framework on largescale benchmarks, including FPHA and H2O. Extensive experiments validate that our method generates a new State-of-the-Art score on 3D hand pose estimation and action recognition from egocentric view monocular videos.

## 2 Related Work

3D Hand Pose Estimation & Hand Action Recognition Hand action typically involves the interaction of hands and objects. 3D hand position provides important information about the object geometry and grasp type, which is positively correlated with hand action [3, 4, 41, 53]. Therefore, the hand pose is an influential feature for hand action recognition [15, 44]. Recently, CNN-based approaches [35, 46], and graph convolutional networkbased methods [24, 40, 51] have learned important hand pose information from extracted meaningful spatial features. However, since hand actions are not static representations, it is difficult to perceive the continuous movement of hands. Thus, it is crucial to consider temporal information to improve understanding of hand activities. Alternative action recognition models such as temporal CNN-based [20, 22, 30, 52], LSTM-based [27, 28, 44], and two-stream networks [5, 9, 10, 42] appear, yet they still rely on either information. To consider both spatial and temporal information, current state-of-the-art works [25, 29] introduce transformer-based approaches with the multi-head self-attention network that helps find the relationship between the input sequences. From these observations, we adopt hierarchical transformers model architecture to learn not only spatial information through positional encoding, but also geometric 3D hand position with short-term temporal cues and semantic action flow with long-term temporal cues hierarchically at once.

Hand Type Taxonomy The relation between hand grasp type and object has been widely studied for decades [3, 16, 23]. The grasp types reveal the characteristics of the object because people generally use the same or similar hand grasp types for particular things. Early work by Schelesinger et al. [39] first categorized hand grasp type into six major types based on object shape, hand surfaces, and hand shape. Also, Napier et al. [34] divided hand movements into two main groups, prehensile and non-prehensile, and established the concept of power and precision grasp types. Furthermore, in recent hand-related works [19, 32, 50], the grasp taxonomy proposed by Feix et al. [12] has been widely used in hand analysis. Understanding the hand grasp type helps the model recognize the user’s intention more accurately and improves the accuracy of hand action recognition. Therefore, we precisely redefine the taxonomy of hand grasp types based on functionality for practical action comprehension.

![](images/57fa9f698b2f840cca31bd642e5866e547dbc25e3115cff590c1204f834a351d.jpg)  
Figure 2: Taxonomy of hand types based on functionality. We categorize the mainly used hand types focusing on the role of the hand. Beyond static hand grasp types, we further define dynamic hand types to Control to explain more real-life hand behaviors.

## 3 Function-Based Hand Type Taxonomy

Insight into the context between functional hand configurations and objects could significantly advance understanding of hand action. In this work, we consider the semantic prior of the human-level hand type as a critical indicator of perceiving egocentric view scenes. To achieve this goal, we first carefully design a useful taxonomy of hand types based on the functioning perspective and make use of them for effective dataset curation.

Redesign Taxonomy The grasp type is a figurative expression of human intention when performing tasks with hands, and it is closely related to hand action. Furthermore, since we interact with the other person’s hand or object to perform many activities, it is very important to identify the hand type in understanding the relationship between the hand and the others. Most previous studies [1, 3, 18] have defined hand type taxonomies based on the characteristics of objects interacting with the hand because people usually use the same or similar grasp types for specific objects. They design and utilize hierarchical hand type categorization criteria based on grasp manner. We empirically observe that this hierarchy helps understand the appearance of the hand. However, to better understand the activity in the egocentric view video clip, it is more beneficial to categorize based on the function and role of the hand as well as the appearance of the hand. In other words, there are limitations to traditional taxonomy in explaining various hand action scenarios in everyday life. Our carefully designed taxonomy starts from this observation.

As shown in Fig. 2, we first categorize the mainly used hand types into two groups: Isolated and Interaction. Isolated represents independent hand types that do not interact with other people or objects, and Interaction describes hand types that are actively interacting with other people or objects. We then organize the hand types as Interaction into Grasp and Control, focusing on the functions in the activity. Grasp includes hand types that hold an object even over time, which are further divided into Power, Precision, and Intermediate depending on the hand’s appearance. We also distinguish between Circular and Prismatic pivoting on the shape of the interacting objects. On the other hand, Control includes hand types that manipulate objects over time. Here, we classify hand types, concentrating on the deformation of the object. For example, the hand type that turns the lid does not deform the object, but the hand type that crumples the paper or the hand type that opens the can deform the object. Thus, we differentiate them as Non-Deformable and Deformable according to these criteria.

Dataset Annotation Ultimately, we perform hand type annotation for all frames of the FPHA [15] and H2O [24] dataset based on the newly defined taxonomy. We implement hand type labeling on the right hand for FPHA, as it contains information only for the right hand. However, for the H2O dataset, which provides data for both hands, we label both hands. Note that more details about dataset annotation are provided in supplemental material.

## 4 Learning Functional Hand Type

In this section, we advocate for leveraging human-level hand type knowledge to provide semantically rich cues beyond using simply physical-level hand pose information. In particular, we explicitly guide the action recognition pipeline via rich semantic prior of the hand type based on hand type taxonomy that we proposed in Sec. 3.

It is noteworthy that temporal information for estimated 3D hand position and high-level object labels enhances action recognition accuracy. Although existing techniques [24, 44, 49] benefit from the generous geometric potential of 3D position, they do not cover diverse real-world scenarios. Specifically, they overlook the semantic role of hand motions in various scenarios and still face challenges in perceiving dynamic hand movements relying solely on geometrical or physical information. Therefore, we propose to utilize semantic knowledge of hand type in each frame to encourage the network to understand the continuous interaction between the primary hand and the assistive hand/object in the video clip.

## 4.1 Overview

We illustrate the outline of our framework in Fig 3. Built upon hierarchical temporal transformer (HTT [49]), which considers the high-level action recognition task as a mixture of two low-level tasks: 3D hand pose estimation and object classification, we especially utilize hand type estimation network to provide valuable semantic cues. Our model consists of three main modules: (1) Feature Extraction, (2) Egocentric Knowledge Module, which estimates 3D hand pose, object category and hand type leveraging the short-term temporal cue and (3) Egocentric Action Module, which aggregates per-frame pose, object and hand type information over a longer time span. First, our model takes aligned 2D video clip $V = \{ X _ { i } \in \mathbb { R } ^ { 3 \times H \times W } | i = 1 , . . . , K \}$ consisting of K frames as input, which are converted into feature vector $F _ { I }$ containing fine details. Then we employ temporal-dependent features $F _ { H }$ from the Local Transformer to estimate the per-frame 3D hand pose with feature vector

![](images/11421d9d3dd2d7b7bbef06463ad66a00296f8302ff0fe0763d4aa6b2a9d5665d.jpg)  
Figure 3: Overview of our proposed model. Our framework consists of three main modules: (1) Feature Extraction, (2) Egocentric Knowledge Module, which estimates 3D hand pose, object category and hand type leveraging short-term temporal cues, and (3) Egocentric Action Module, which aggregates per-frame pose, object and hand type information over a longer time span.

$F _ { P }$ , which contains geometric potential, and the category of interacting objects with feature vector $F _ { O }$ . Additionally, the hand type estimation network takes $F _ { H }$ and outputs hand type feature vectors $F _ { T }$ . Here, we utilize language models to provide the deep semantic cues of hand type based on our proposed functional hand type taxonomy. Subsequently, the Egocentric Action Module aggregates the predicted embeddings: hand position $F _ { P }$ , object category $F _ { O }$ , and hand type $F _ { T }$ for action recognition.

## 4.2 Egocentric Knowledge Module

To construct Egocentric Knowledge Module input sequence, we first divide the long video clip V into m consecutive segment $\mathsf { s e g } _ { \mathtt { k } } ( V ) = ( \bar { V } _ { 1 } , \bar { V } _ { 2 } , . . . , \bar { V } _ { m } )$ , where m denotes $\lceil K / k \rceil$ . In order to capture the temporal cue of consecutive segment for hand pose estimation, the module processes each segment $\bar { V } \in \mathsf { s e g } _ { \mathtt { k } } ( V )$ in parallel. Then transformer takes the sequence of per-frame feature vector $F _ { I }$ from the image encoder and outputs temporal-dependent features $F _ { H }$ . To decode the hand pose information, these features $F _ { H }$ are fed into simple MLP layers, yielding joint coordinates in the 2D image plane $P ^ { 2 D } \in \mathbb { R } ^ { J \times 2 }$ and the joint depth to the camera $P ^ { d e p } \in \mathbb { R } ^ { J \times 1 }$ . We train the 3D pose estimation module to minimize the following pose loss (L1-loss):

$$
\mathcal { L } _ { \mathrm { p o s e } } = \frac { 1 } { J } ( | | P ^ { 2 D } - P _ { g t } ^ { 2 D } | | _ { 1 } + \lambda _ { \mathrm { p o s e } } | | P ^ { d e p } - P _ { g t } ^ { d e p } | | _ { 1 } )\tag{1}
$$

where J denotes hand joints and $\lambda _ { \mathrm { p o s e } }$ is a hyperparameter to balance the different intensities of the 2D and depth losses. The 3D positions of the hand joints in the camera space $P ^ { 3 D } \in$ $\mathbb { R } ^ { J \times 3 }$ for I can be inferred operating the camera intrinsics.

In addition to using 3D hand pose information, which gives precise geometric knowledge about egocentric view scenarios, we advocate for leveraging hand type information to guide the entire network to learn deep semantic understanding. In particular, the context details of hand type can serve as a practical semantic key representing hand-object relationships for identifying hand actions. Thus, we introduce a simple but novel hand-type classification network $\phi _ { T }$ to predict the hand type $t _ { i }$ for $i = \left\{ 1 , 2 , \ldots , N _ { t } \right\}$ from temporal-dependent features $F _ { H }$ Given the ground truth hand type label $t _ { g t }$ from the dataset annotation process (see Sec. 3), the target probability is defined as a one-hot vector $w _ { t }$ . For training, we formulate the following cross-entropy loss to train the hand type classification network $\phi _ { T }$ :

$$
\mathcal { L } _ { t y p e } = - \mathbb { E } _ { t , w _ { t } \sim \mathbb { D } } \left[ \sum _ { r \in N _ { t } } w _ { t } [ r ] \log \phi _ { T } ( t ) [ r ] \right]\tag{2}
$$

where D is (input) data distribution. Also, we predict the object category $o _ { i }$ in each frame with the object classification network $\phi _ { O }$ . Similar to hand type classifier, the classifier $\phi _ { O }$ is supervised to minimize the cross entropy loss $\mathcal { L } _ { o b j }$

## 4.3 Boosting the Action Recognition with Hand Type Prior

We introduce the strategies using semantic prior of the human-level hand type, potentially improving not only action recognition but also 3D hand pose estimation. Specifically, we explicitly pilot the Egocentric Action Module to learn rich semantic hand type variant information via utilizing the power of the prevalent language model, Contrastive Language-Image Pre-Training [38] (CLIP). Given the candidate of hand type from the hand type classification network φ , we map the type number $t _ { i }$ with the corresponding text descriptions (e.g., “Hand Clench”, “Tip Pinch”). Next, the text label of hand type goes through the large-scale pre-trained language model, which outputs text embedding vector $F _ { T }$ . Importantly, we consider these text embedding vector of predicted hand type descriptions as critical indicators to deliver semantic knowledge to the action recognition pipeline.

## 4.4 Egocentric Action Module

We adopt the previous approaches [7, 21] presenting trainable tokens to aggregate the global information across the input video clip V. Each token encodes short-term temporal information such as temporal-dependent features, hand pose, object label, and hand type details. We design the fully connected layer for each cues, which outputs features of the same dimension, then concatenates these aligned features for input of the Global Transformer as follows:

$$
F _ { a g g } = \mathrm { F C } [ F _ { H } \oplus F _ { P } \oplus F _ { O } \oplus F _ { T } ]\tag{3}
$$

where ⊕ indicates channel-wise concatenation of feature vectors and FC[·] reduces the features into d-dim to fit in the token dimension of Global Transformer. After mixing features that potentially contain geometric and semantic knowledge, we feed these feature vector $F _ { a g g }$ to Global Transformer, which outputs action tokens. Here, we utilize an action classification head $\phi _ { A }$ to recognize action label $a _ { i }$ for $i = \left\{ 1 , 2 , \ldots , N _ { a } \right\}$ from action tokens. For supervision, we formulate the following cross-entropy loss to train the action classification head $\phi _ { A }$

$$
\mathcal { L } _ { a c t } = - \mathbb { E } _ { a , w _ { a } \sim \mathbb { D } } \left[ \sum _ { r \in N _ { a } } w _ { a } [ r ] \log \phi _ { A } ( a ) [ r ] \right]\tag{4}
$$

where $w _ { a }$ denotes one hot-encoded action labels and D represents (input) data distribution.

## 4.5 Training

Our entire network is trained end-to-end by minimizing the following loss $\mathcal { L } _ { \mathrm { t o t a l } }$

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { a c t } } \mathcal { L } _ { \mathrm { a c t } } + \frac { 1 } { K } \sum _ { \bar { V } \in \mathrm { s e g a r } ( V ) } \sum _ { I \in \bar { V } } ( \lambda _ { \mathrm { p o s e } } \mathcal { L } _ { \mathrm { p o s e } } + \lambda _ { \mathrm { o b j } } \mathcal { L } _ { \mathrm { o b j } } + \lambda _ { \mathrm { t y p e } } \mathcal { L } _ { \mathrm { t y p e } } )\tag{5}
$$

where $\lambda _ { \mathrm { a c t } } , \lambda _ { \mathrm { p o s e } } , \lambda _ { \mathrm { o b j } }$ and $\lambda _ { \mathrm { t y p e } }$ are the hyperparameters of respective loss terms.

## 5 Experiments

Table 1: Comparison of our novel hand action recognition framework and the state-of-theart models on the FPHA [15] and H2O [24] dataset. We report the classification accuracy of methods based on RGB videos. Note that the H2O dataset provides additional testing split videos, unlike the FPHA dataset, which provides only training and validation split.
<table><tr><td>Joule-color []</td><td>Two Stream []</td><td>H+O</td><td>[]</td><td>Collaborative []</td><td>HTT []</td><td>Trear []</td><td>Ours</td></tr><tr><td>Accuracy (↑) 66.78</td><td></td><td>75.30</td><td>82.43</td><td>85.22</td><td>94.09</td><td>94.96</td><td>95.13</td></tr></table>

(a) Action recognition accuracy (%) on FPHA.

<table><tr><td></td><td>C2D []</td><td>I3D []</td><td>SlowFast []</td><td>H+O []</td><td>ST-GCN []</td><td>TA-GCN []</td><td>HTT []</td><td>Ours</td></tr><tr><td>Val Accuracy (↑)</td><td>76.10</td><td>85.15</td><td>86.00</td><td>80.49</td><td>83.47</td><td>86.78</td><td>90.16</td><td>91.80</td></tr><tr><td>Test Accuracy (↑)</td><td>70.66</td><td>75.21</td><td>77.69</td><td>68.88</td><td>73.86</td><td>79.25</td><td>86.36</td><td>89.67</td></tr></table>

(b) Action recognition accuracy (%) on H2O.

## 5.1 Experimental Setups

Dataset We train and evaluate overall performance on two landmark datasets for action recognition from egocentric views: FPHA [15] and H2O [24]. These two datasets are collected in various indoor settings and have a frame rate of 30 frames per second. Both datasets provide the ground truth labels for hand pose, action, and object category, and we utilize them for supervision and evaluation. In this work, we annotate the hand type of all frames in these two datasets based on the newly defined taxonomy and use them for training. Note that detailed descriptions of each dataset are provided in the supplemental material.

Evaluation metrics To evaluate hand action recognition, we follow the official evaluation protocol of the hand action recognition task. We report classification accuracy over the validation and test split by comparing each video’s predicted and ground truth action categories. Also, we evaluate the 3D pose estimation performance in the camera space and the rootaligned (RA) space, which aligns the estimated wrist position with the ground truth for each frame. We report the Percentage of Correct Keypoints (PCK) for joints [54] against different error thresholds and the corresponding Area Under the Curve (AUC). We also utilize Mean End-Point Error (MEPE) metrics for hands [54] in the camera and root-aligned space following HTT [49]. We provide all implementation details in the supplemental material.

## 5.2 Comparison with the State-of-the-Arts

Hand Action Recognition We compare our proposed method with existing state-of-theart methods including Joule-color [17], Two Stream [9], H+O [44], Collaborative [52],

Table 2: 3D pose estimation performance in Root-Aligned space on the FPHA [15] and H2O [24]. We report AUC-RA for 3D PCK-RA at error thresholds ranging from 0 to 50 mm and the MEPE-RA in the unit of mm.
<table><tr><td>Model</td><td>AUC-RA(0-50) (↑)</td><td>MEPE-RA (↓)</td></tr><tr><td>HTT []</td><td>0.763</td><td>12.13</td></tr><tr><td>Ours</td><td>0.769</td><td>11.79</td></tr></table>

(a) FPHA Dataset

<table><tr><td rowspan="2">Model</td><td colspan="2">AUC-RA(0-50) (↑)</td><td colspan="2">MEPE-RA (↓)</td></tr><tr><td>Left</td><td>Right</td><td>Left</td><td>Right</td></tr><tr><td>HTT []</td><td>0.674</td><td>0.648</td><td>16.59</td><td>17.91</td></tr><tr><td>Ours</td><td>0.686</td><td>0.662</td><td>15.96</td><td>17.08</td></tr></table>

(b) H2O Dataset

Table 3: Ablative study of input features for Egocentric Action Module (EAM) on Hand Action Recognition Accuracy (%). We investigate the usage of the hand type feature in (a). Also, we analyze the effectiveness of each cue on the action recognition task in (b).
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Hand Type EAM Input Text Embedding</td><td colspan="2">Accuracy (↑)</td><td colspan="4">Input Feature for Egocentric Action Module</td><td colspan="2">Accuracy (↑)</td></tr><tr><td>FPHA [] H2O []</td><td></td><td>Image Feature Hand Pose Object Label Hand Type</td><td></td><td></td><td></td><td> FPHA [] H2O []</td><td></td></tr><tr><td>√</td><td>-</td><td></td><td>93.74</td><td>85.95</td><td>√</td><td>√</td><td>√</td><td>1</td><td>94.09</td><td>86.36</td></tr><tr><td>√</td><td>√</td><td></td><td>94.26</td><td>87.60</td><td>√</td><td>√</td><td></td><td>√ √</td><td>93.74 94.61</td><td>87.19</td></tr><tr><td>√</td><td>√</td><td>√</td><td>95.13</td><td>89.67</td><td>√ √</td><td>- √</td><td>√ √</td><td>√</td><td>95.13</td><td>86.78 89.67</td></tr></table>

(a) Effect of Text-based Hand Type Feature  
(b) Effect of Hand Type Cue

HTT [49], and Trear [25] on the FPHA [15] dataset (see Table 1 (a)). On the H2O dataset, we compare ours with C2D [48], I3D [45], SlowFast [10], H+O [44], ST-GCN [51], TA-GCN [24] and HTT [49] (see Table 1 (b)). As reported in Table 1 (a) and (b), our method generally outperforms other methods with state-of-the-art accuracy (FPHA [15]: 95.13%, H2O [24]: 89.67%). These results emphasize the effectiveness of our method in understanding the interaction between hands and objects.

Hand Pose Estimation We also demonstrate considerable performance on 3D hand pose estimation, scoring AUC for 3D PCK at error thresholds ranging from 0 to 50 mm and the MEPE measured in mm within the Root-Aligned (RA) space. Our method estimates hand pose more precisely than the current state-of-the-art method [49] (see Table 2). These experimental results verify that the semantic knowledge of the hand type benefits not only the action recognition network but also the entire network, resulting in high precision in 3D hand pose estimation.

## 5.3 Ablation Study

Effect of Text-based Hand Type Feature In this section, we evaluate the variants of our method across hand type features. As shown in Table 3 (a), we investigate the usage of the hand type feature. The Egocentric Action Module (EAM) performs better when the predicted probability distribution feature vector from the hand type estimation network is used as input than when the hand type is simply predicted as the auxiliary network. Further, utilizing text embedding of hand type improves accuracy over using the predicted feature vector.

Effect of Hand Type Cue With four types of cues (image feature, hand pose, object, and hand type), we analyze the effect of each cue on action recognition with and without each cue as input of the egocentric action module. Specifically, on the FPHA [15] and the H2O [24] datasets, hand type cue effectively enhances accuracy, as shown in Table 3 (b). We finally observe that employing hand type as a semantic prior plays a key role in action recognition.

(b)

![](images/01a6f2555e6352c577a640d378fb355b7aea9c712793117996c0e78c525ef7ae.jpg)

![](images/1dec0fdc479187e3a69b7fb088c239fbdfa478e4a9ef09cd1f17b140ac924992.jpg)

![](images/d2f6c62aad4380f081a6fd9a1dbb27c7936094523ff9b95dc20dfef5dff51acc.jpg)  
Figure 4: Qualitative result of our experiments. In (a), the green and blue line represents ground truth and estimated 3D hand pose, respectively. (b) shows the 3D PCK of hand pose estimation results on H2O [24] in Root-Aligned space. The blue line indicates the performance of our model, whereas the red line represents the HTT [49].

## 5.4 Qualitative Analysis

In this section, we qualitatively verify the usefulness of our novel framework. Fig. 4 shows the visualized results of our experiments. In Fig. 4 (a), the ground truth hand pose (see green lines) and our estimated hand pose (see blue lines) are projected in both 3D space and images. Ours show comparable results compared to ground truth. We also provide 3D PCK-RA graphs of left and right hand pose estimation results on the H2O test split of ours (see blue lines) vs. HTT [49] (see red lines) in Fig. 4 (b). This graph validates that our method generates reasonable results in both hands. Overall, our model is robust for estimating hand pose in 3D space. We provide more qualitative results and analysis in supplementary materials.

## 6 Conclusion

In this paper, we present a novel method applying the knowledge of hand type for hand action recognition based on the temporal transformer. This is the first attempt to regard the semantic details of the hand type as a critical indicator for enhancing the perception of egocentric view hand actions. To utilize the knowledge of hand type, we newly define the taxonomy based on hand functionality and annotate hand types for existing large-scale benchmarks. The experiments demonstrate the outstanding performance of our proposed approach.

## Acknowledgement

This work was supported by Electronics and Telecommunications Research Institute (ETRI) grant funded by the Korean government (23ZH1300, Research on hyper-realistic interaction technology for five senses and emotional experience, 95%), Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2019-0-00079, Artificial Intelligence Graduate School Program (Korea University), 3%), the National Research Foundation of Korea grant (NRF-2022R1F1A10 74334, 2%), and results of a study on the "Leaders in Industry-university Cooperation 3.0" Project, supported by the Ministry of Education and National Research Foundation of Korea. This work was supported by Artificial intelligence industrial convergence cluster development project funded by the Ministry of Science and ICT(MSIT, Korea)& Gwangju Metropolitan City.

## References

[1] Ian M Bullock, Júlia Borràs, and Aaron M Dollar. Assessing assumptions in kinematic hand models: a review. In 2012 4th IEEE RAS & EMBS International Conference on Biomedical Robotics and Biomechatronics BioRob), pages 139–146. IEEE, 2012.

[2] Ian M Bullock, Thomas Feix, and Aaron M Dollar. Finding small, versatile sets of human grasps to span common objects. In ICRA. IEEE, 2013.

[3] Minjie Cai, Kris M Kitani, and Yoichi Sato. Understanding hand-object manipulation with grasp types and object attributes. In Robotics: Science and Systems, volume 3. Ann Arbor, Michigan;, 2016.

[4] Minjie Cai, Kris M Kitani, and Yoichi Sato. An ego-vision system for hand grasp analysis. IEEE Transactions on Human-Machine Systems, 47(4):524–535, 2017.

[5] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 2017.

[6] Mark R Cutkosky et al. On grasp choice, grasp models, and the design of hands for manufacturing tasks. IEEE Transactions on robotics and automation, 5(3):269–279, 1989.

[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[8] Alireza Fathi, Ali Farhadi, and James M Rehg. Understanding egocentric activities. In ICCV. IEEE, 2011.

[9] Christoph Feichtenhofer, Axel Pinz, and Andrew Zisserman. Convolutional two-stream network fusion for video action recognition. In CVPR, 2016.

[10] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In ICCV, 2019.

[11] Thomas Feix, Roland Pawlik, Heinz-Bodo Schmiedmayer, Javier Romero, and Danica Kragic. A comprehensive grasp taxonomy. In Robotics, science and systems: workshop on understanding the human handfor advancing robotic manipulation, volume 2, pages 2–3. Seattle, WA, USA;, 2009.

[12] Thomas Feix, Javier Romero, Heinz-Bodo Schmiedmayer, Aaron M Dollar, and Danica Kragic. The grasp taxonomy of human grasp types. IEEE Transactions on humanmachine systems, 46(1):66–77, 2015.

[13] Qing Gao, Jinguo Liu, and Zhaojie Ju. Robust real-time hand detection and localization for space human–robot interaction based on deep learning. Neurocomputing, 390:198– 206, 2020.

[14] Qing Gao, Yongquan Chen, Zhaojie Ju, and Yi Liang. Dynamic hand gesture recognition based on 3d hand pose estimation for human–robot interaction. IEEE Sensors Journal, 22(18):17421–17430, 2021.

[15] Guillermo Garcia-Hernando, Shanxin Yuan, Seungryul Baek, and Tae-Kyun Kim. First-person hand action benchmark with rgb-d videos and 3d hand pose annotations. In CVPR, 2018.

[16] René Gilster, Constanze Hesse, and Heiner Deubel. Contact points during multidigit grasping of geometric objects. Experimental brain research, 217:137–151, 2012.

[17] Jian-Fang Hu, Wei-Shi Zheng, Jianhuang Lai, and Jianguo Zhang. Jointly learning heterogeneous features for rgb-d activity recognition. In CVPR, 2015.

[18] Thea Iberall. Human prehension and dexterous robot hands. The International Journal ofRobotics Research, 16(3):285–299, 1997.

[19] Georgiana Juravle, Heiner Deubel, and Charles Spence. Attention and suppression affect tactile perception in reach-to-grasp movements. Acta psychologica, 138(2):302– 310, 2011.

[20] Qiuhong Ke, Mohammed Bennamoun, Senjian An, Ferdous Sohel, and Farid Boussaid. A new representation of skeleton sequences for 3d action recognition. In CVPR, 2017.

[21] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pages 4171–4186, 2019.

[22] Tae Soo Kim and Austin Reiter. Interpretable 3d human action analysis with temporal convolutional networks. In CVPRW. IEEE, 2017.

[23] Roberta L Klatzky, Brian McCloskey, Sally Doherty, James Pellegrino, and Terence Smith. Knowledge about hand shaping and knowledge about objects. Journal of motor behavior, 19(2):187–213, 1987.

[24] Taein Kwon, Bugra Tekin, Jan Stühmer, Federica Bogo, and Marc Pollefeys. H2o: Two hands manipulating objects for first person interaction recognition. In ICCV, 2021.

[25] Xiangyu Li, Yonghong Hou, Pichao Wang, Zhimin Gao, Mingliang Xu, and Wanqing Li. Trear: Transformer-based rgb-d egocentric action recognition. IEEE Transactions on Cognitive and Developmental Systems, 14(1):246–252, 2021.

[26] Yin Li, Zhefan Ye, and James M Rehg. Delving into egocentric actions. In CVPR, 2015.

[27] Jun Liu, Amir Shahroudy, Dong Xu, and Gang Wang. Spatio-temporal lstm with trust gates for 3d human action recognition. In ECCV 2016. Springer, 2016.

[28] Jun Liu, Gang Wang, Ping Hu, Ling-Yu Duan, and Alex C Kot. Global context-aware attention lstm networks for 3d action recognition. In CVPR, 2017.

[29] Miao Liu, Siyu Tang, Yin Li, and James M Rehg. Forecasting human-object interaction: joint prediction of motor attention and actions in first person video. In ECCV. Springer, 2020.

[30] Diogo C Luvizon, David Picard, and Hedi Tabia. 2d/3d pose estimation and action recognition using multitask deep learning. In CVPR, 2018.

[31] Priyanka Mandikal and Kristen Grauman. Dexvip: Learning dexterous grasping with human hand pose priors from video. In Conference on Robot Learning. PMLR, 2022.

[32] T Manti. A novel type of compliant underactuated robotic hand for grasping. Soft Robotics, 35:161–185, 2015.

[33] Osama Mazhar, Benjamin Navarro, Sofiane Ramdani, Robin Passama, and Andrea Cherubini. A real-time human-robot interaction framework with robust background invariant hand gesture detection. Robotics and Computer-Integrated Manufacturing, 60:34–48, 2019.

[34] John R Napier. The prehensile movements of the human hand. The Journal of bone andjoint surgery. British volume, 38(4):902–913, 1956.

[35] Supreeth Narasimhaswamy, Zhengwei Wei, Yang Wang, Justin Zhang, and Minh Hoai. Contextual attention for hand detection in the wild. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9567–9576, 2019.

[36] Yuzhe Qin, Hao Su, and Xiaolong Wang. From one hand to multiple hands: Imitation learning for dexterous manipulation from single-camera teleoperation. IEEE Robotics and Automation Letters, 7(4):10873–10881, 2022.

[37] Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, Hanwen Jiang, Ruihan Yang, Yang Fu, and Xiaolong Wang. Dexmv: Imitation learning for dexterous manipulation from human videos. In ECCV. Springer, 2022.

[38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML. PMLR, 2021.

[39] Georg Schlesinger. Der mechanische aufbau der künstlichen glieder. Ersatzglieder und Arbeitshilfen: Für Kriegsbeschädigte und Unfallverletzte, pages 321–661, 1919.

[40] Lei Shi, Yifan Zhang, Jian Cheng, and Hanqing Lu. Two-stream adaptive graph convolutional networks for skeleton-based action recognition. In CVPR, 2019.

[41] Roy Shilkrot13, Supreeth Narasimhaswamy, Saif Vazir, and Minh Hoai12. Workinghands: A hand-tool assembly dataset for image segmentation and activity mining. 2019.

[42] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. NeurIPS, 27, 2014.

[43] Suriya Singh, Chetan Arora, and CV Jawahar. First person action recognition using deep learned descriptors. In CVPR, 2016.

[44] Bugra Tekin, Federica Bogo, and Marc Pollefeys. H+ o: Unified egocentric recognition of 3d hand-object poses and interactions. In CVPR, 2019.

[45] Quo Vadis, Joao Carreira, and Andrew Zisserman. Action recognition? a new model and the kinetics dataset. Joao Carreira, Andrew Zisserman.

[46] Limin Wang, Yu Qiao, Xiaoou Tang, et al. Action recognition and detection by combining motion and appearance features. THUMOS14 Action Recognition Challenge, 1 (2):2, 2014.

[47] Weitian Wang, Rui Li, Zachary Max Diekel, Yi Chen, Zhujun Zhang, and Yunyi Jia. Controlling object hand-over in human–robot collaboration via natural wearable sensing. IEEE Transactions on Human-Machine Systems, 49(1):59–71, 2018.

[48] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In CVPR, 2018.

[49] Yilin Wen, Hao Pan, Lei Yang, Jia Pan, Taku Komura, and Wenping Wang. Hierarchical temporal transformer for 3d hand pose estimation and action recognition from egocentric rgb videos. In CVPR, 2023.

[50] Cai-Hua Xiong, Wen-Rui Chen, Bai-Yang Sun, Ming-Jin Liu, Shi-Gang Yue, and Wen-Bin Chen. Design and implementation of an anthropomorphic hand for replicating human grasping functions. IEEE Transactions on Robotics, 32(3):652–671, 2016.

[51] Sijie Yan, Yuanjun Xiong, and Dahua Lin. Spatial temporal graph convolutional networks for skeleton-based action recognition. In AAAI, 2018.

[52] Siyuan Yang, Jun Liu, Shijian Lu, Meng Hwa Er, and Alex C Kot. Collaborative learning of gesture recognition and 3d hand pose estimation with multi-order feature analysis. In ECCV. Springer, 2020.

[53] Yezhou Yang, Cornelia Fermuller, Yi Li, and Yiannis Aloimonos. Grasp type revisited: A modern perspective on a classical feature for vision. In CVPR, 2015.

[54] Christian Zimmermann and Thomas Brox. Learning to estimate 3d hand pose from single rgb images. In ICCV, 2017.

# Supplementary Material: Functional Hand Type Prior for 3D Hand Pose Estimation and Action Recognition from Egocentric View Monocular Videos

Wonseok Roh<sup>1</sup> paulroh@korea.ac.kr Seung Hyun Lee<sup>1</sup> easter3163@korea.ac.kr

Won Jeong Ryoo<sup>1</sup> petac@korea.ac.kr

Jakyung Lee<sup>1</sup> 2023020917@korea.ac.kr

Gyeongrok Oh<sup>1</sup> dhrudfhr98@korea.ac.kr

Sooyeon Hwang<sup>1</sup> hsy506@korea.ac.kr

Hyung-gun Chi<sup>2</sup> chi45@purdue.edu

Sangpil Kim<sup>1∗</sup> spk7@korea.ac.kr

<sup>1</sup> Department of Artificial Intelligence, Korea University, Seoul, Republic of Korea

<sup>2</sup> Electrical and Computer Engineering, Purdue University, West Lafayette, Indiana, USA

## Overview

In this supplementary material, we supply further explanations and visualizations of our main paper, "Functional Hand Type Prior for 3D Hand Pose Estimation and Action Recognition from Egocentric View Monocular Videos". We first provide additional experimental results to validate our proposed framework (Section A). We also describe implementation details for training (Section B). Next, we explain elaborate descriptions of hand type annotation process (Section C). We further provide more details for large-scale datasets (FPHA [1] and H2O [2]) and analysis of hand type distributions based on functional hand type taxonomy (Section D). Moreover, we supply extra details on the following project website: https://kuai-lab.github.io/bmvc2023fhtp.

## A. Additional Experimental Results

Qualitative Results In this section, we provide additional qualitative examples and analysis of our proposed model. In Fig. 1, we show various visualized results of the 3D hand poses predicted by ours (see blue lines) and HTT [3] (see magenta lines) in 3D space and their projection to corresponding 2D image frames, compared with ground truth (see green lines). We observe that for both FPHA [1] (a) and H2O [2] (b), the gap between ours and ground truth is smaller than that between HTT and ground truth. These visualizations qualitatively verify that our model outperforms the baseline model by using functional hand type as semantic prior. In other words, utilizing hand type assists estimating hand pose in 3D space.

![](images/bf14dee8d54d6d1eb0cbf9e783d09a7ff9991b090dcce3aedb27b1cfcf230973.jpg)  
(a) FPHA Dataset  
(b) H2O Dataset  
Figure 1: Qualitative comparison of 3D hand pose estimation results between our model (blue lines) and HTT (magenta lines) on the (a) FPHA [1] and (b) H2O [2] dataset. Both ground truth (green lines) and estimated hand poses are visualized in 3D space and projected to the corresponding 2D image frames. Note that the H2O provides labels for both hands, while the FPHA only includes information for the right hand.

## B. Implementation Details

For training, we use two RTX 3090 GPUs with a batch size of 4, allowing for efficient processing and improved performance. Since the model converged at around 45 epochs, we trained it for a total of 45 epochs with a learning rate $3 \times 1 0 ^ { - 5 }$ , which is decreased 10 times every 15 epochs. We applied the Adam optimizer to train the dataset. Using PyTorch, we implement our experimental setup with the following settings. Our model architecture adopts K RGB frames, resized to $4 8 0 \times 2 7 0$ , as the input sequence. All K input sequence images are sliced into k segments and fed into ResNet-18 pre-trained on ImageNet to extract image features. Then transformer block takes these features and outputs temporal-dependent features. We set the dimension size of all features and tokens to 512. Moreover, to optimize our model efficiently, we apply data augmentation strategies following the previous method [3].

## C. Annotation Details

Beyond traditional hand grasp types, we categorize hand types based on functionality. In addition, we newly define additional hand types to describe more real-life hand activities and utilize deep semantic knowledge of hand types. For example, classic taxonomy includes the “Index Finger” category for holding an object while extending the index finger. However, as shown in Fig. 2, we add two more categories for extending the index finger; “Poking” which indicates to poking into an object without deforming it, and “Extended Index Curl”, which represents a hand type that utilizes the index finger to deform an object by applying pressure on its tip. Here, we explain the details of our annotation process based on these function-based taxonomy.

We manually annotate hand type all frames of landmark datasets, considering both the appearance of the hand and the various functions the hand performs. To make consistent annotation, we inspect which hand type certainly appears in specific actions before annotating each frame individually. Explanation in more detail; for the “Open Juice Bottle” action of the FPHA [1] dataset (left side of Fig. 1 in the main paper), we label “Tripod” as holding the juice lid and “Dynamic Tripod” as opening the lid. Therefore, the frames just holding the lid are marked as “Tripod” and from the frames that start to open, they are marked as “Dynamic Tripod” since the hand type changes to manipulate operation. More specifically, when the thumb finger is out while opening the lid, the hand type converts to “Thumb Up”. If the hand is out of frames or occluded by the object, it is annotated as “Invisible”. Also, for the challenging and unclear hand type, we go over whole videos implementing the same action by other subjects and then annotate as the most closely matched hand type.

As shown in Fig. 3, we show continuous annotation examples where hand types change over time in the video sequence, following the annotation details above. In Fig. 3 (a), we provide frames of the action called “Place Cappuccino” that interact with the object (cappuccino box). While the left hand keeps the hand on the desk (“Relaxed Hand”), the type of the right hand changes as it interacts with the cappuccino box. The right hand type starts as “Large Diameter” while holding the box and converts into a “Fist” through the intermediate stage of “Relaxed Hand” after putting the box down. Further, in Fig. 3 (b), we visualize frames of the action called “High Five” that interact with the other’s hand. Even though the shape of the hand is the same throughout high-fiving, hand type differs as “Open Hand” or “Dynamic Flatten” based on whether it interacts with the other person’s hand.

## D. Analysis of Datasets and Hand Type Distributions

We train and evaluate overall performance on two landmark datasets (FPHA [1] and H2O [2]). FPHA The FPHA [1] (First-Person Hand Action) dataset records 45 action categories from 6 subjects with an Intel RealSence SR300 RGB-D camera mounted on the subject’s shoulder. It designs various action scenarios interacting with 26 different objects. Only J=21 joints of the subject’s right hand are annotated using 3D poses from the wearable magnetic sensors. It contains 1,175 sequences and separates into 1:1 settings with 600 and 575 videos for the training and evaluation phase in each. Both steps include all subjects and actions.

H2O The H2O [2] (2 Hands and Objects) dataset captures 4 subjects performing 36 actions, interacting with 8 3D objects. Unlike the FPHA [1] dataset, which only provides labels for the right hand, H2O provides 3D labels for both hands with the total number of $J = 2 1 \times 2$ joints. It includes 569 videos for the training, including all actions for the first 3 subjects,

122 videos for the validation, and 242 videos of the remaining subject unseen in the training for the test.

Hand Type Distribution In this paper, we introduce a newly defined taxonomy of 31 hand types based on functionality. We visualize the distribution of our proposed hand type taxonomy with a radial diagram in Fig. 4. We first categorize the mainly used hand types into Isolated and Interaction based on whether the hands interact with objects (or other people). We then classify the Interaction types into Grasp and Control, reflecting the role of the hand in the scene. These categories are further subdivided according to the appearance of the hand and the degree of object manipulation. Finally, function-based taxonomy that comprehensively encompasses real-life hand movements allows explaining various hand action scenarios. With our new taxonomy, we annotate hand types across all frames of two datasets.

To investigate the hand type distributions of both FPHA [1] (Fig. 5 (a)) and H2O [2] (Fig. 5 (b)) datasets, we illustrate them via histograms and scatter plots. We first explore the number of frames per hand type with the histograms (Fig. 5 left) to observe how frequently each hand type occurs within whole hand video sequences. The x-axis represents the hand type, and the y-axis shows the number of frames. As shown in Fig. 5 (left), the distributions of the two datasets exhibit different patterns. In the case of the FPHA, 29 of the 31 hand types are identified, with “Large Diameter”, “Disk Grip” and “Thumb Tucked” being the most common. For the H2O, 24 of the 31 hand types are recognized, with “Large Diameter”, “Relaxed Hand” and “Parallel Extension” being the most prevalent. We find that the FPHA operates more diverse hand types than H2O for more action scenarios.

![](images/b8bb5e3025f838ab43244aaf4a4e8a4903183e4cb8fbdb72dac3f47d9291f130.jpg)  
Figure 4: Radial diagram of function-based hand type taxonomy. To explain more reallife hand behaviors, we carefully design a taxonomy based on the functioning perspective.

Also, we visualize the distribution of hand types across each action label with scatter plots (Fig. 5 right); the x-axis indicates the action of each dataset, and the y-axis represents the hand type. The circle size portrays the appearance frequency of each hand type within each action video clip. As shown in Fig. 5 (right), the FPHA presents monotonous distributions of hand types per action, while the H2O shows a relatively balanced distribution. This difference is because, unlike the H2O, which focuses on both hands, the FPHA annotates only on the right hand and records simple actions repeatedly. Ultimately, FPHA utilizes 29 hand types for 45 action scenarios but is monotonous, and H2O uses 24 hand types for 36 action scenarios but is relatively harmoniously distributed. Note that we further visualize hand type distributions of each left and right hand of the H2O dataset in Fig. 6.

![](images/3b5a90e78d84f8b64f7893799f76645100a7604a10ff93d09ff428eb272792ef.jpg)

![](images/a4d02dd421f87ed3e1e6cf5b27160bbc4e1461d3b3d6fe79b27d98847cf981c4.jpg)

![](images/98efc3d6ddab3eaefb8341ab2dd92250c3d9ac3e59b974b696fbed93711ef4a2.jpg)

![](images/719c605f9852be3726c40d1f9d32bd592e8227b054f32a5ba27ae7ce979b379a.jpg)

![](images/f232cac06d01f5bb27ca104a1885ed66ea94342e5d899afb88eda57eb6598853.jpg)  
(a) Index Finger

![](images/83e24b0ef49ec48394bf7ce549896f8a89b94e179914c065deba149467ac26f0.jpg)

![](images/d37b97d72bd7b2440e71a6b7f3ca43aef9e57c8e3d845e49c4e1534891121263.jpg)

![](images/d15f93424adbe6bef5c591e6ac2ed509663ecf997bed70c876896c82f5005779.jpg)

![](images/a16433d83c659c0dc503141b26f1503452f56af5457a706330a8c86bf1b76e96.jpg)

![](images/baf30812447d618514b263daeaf662c1dbd88cf96f2b40b5565c0bae997e1164.jpg)

![](images/c0586396b3388e300e5404141686d5110b9c249d19de4c11ac4087ea94e8a07b.jpg)  
(b) Poking

![](images/146b6972bd9a2770df71df9ef6e612df2873f52fb80b85750939dad667fee1c5.jpg)

![](images/f3019135cafc843c34d02a9aec97a3f61cc86e33574b601f05e5e2b7a59ea226.jpg)

![](images/8d3b63a74daecfa16beec1367b991eebc92041bbe86de771579101311146add9.jpg)

![](images/0fea99a42db78011c0f6b614f12a78ca872a5568900ec447bf85eeae0a77e862.jpg)

![](images/1895c83027f4c87bc743bbb9314098c01010a1b8299a14c1407b03ffd1617239.jpg)

(c) Extended Index Curl  
![](images/3beab23e056045fca27a4463532739211587d8384f58a6a9a0531eb45f769339.jpg)  
Figure 2: Examples of action scenes in which the variation of hand pose is similar, but the action labels are different according to the temporal context of hand function and object types. (a) “Index Finger”, (b) “Poking”, and (c) “Extended Index Curl” are all hand types that use index fingers, but their functions vary depending on the action and object they interact with. “Index Finger” represents holding an object while extending the index finger, “Poking” represents poking into an object without deforming it, and “Extended Index Curl” represents utilizing the index finger to deform an object by applying pressure on its tip.

![](images/6a6a3b62102b17bd65cfaac9c67e127aa3a2259712c8acdf9b8829ba37361441.jpg)

![](images/8d7036c935456d7c99477b5b3d0b3774ce6d57be4963b5b9fa6d94cb57916f27.jpg)

Relaxed Hand / Large Diameter  
![](images/7a6136bc6177dc94db97f22d27d4860d540ebb61925570ab861903af69322573.jpg)

![](images/d33754abcaa5d0173159705655f4e0265b2c4793b6fd825ed5379604afc9a170.jpg)  
Left Hand Type / Right Hand Type

![](images/fb2e14134b4e9b00d37a4f863c4335f4b96d51af7a2dc05c37e37acb99379765.jpg)

Relaxed Hand / Relaxed Hand  
![](images/2b5a48899610532f14e5b058c2ee25a142fcfd73da714d92231e460f4d601e55.jpg)

![](images/80daa89c44a27c59f7883871532117fa60feafddda70ffce1bbd6fcf7d43ba8f.jpg)  
(a) H2O Dataset

![](images/af8237fc97b58faaa546baf5542997df0045fbafccf61663d175cccc429efc2f.jpg)

![](images/451e79d755615be59daeffbeba07826205226a7c7e2f8f5c556a2b8fe845c499.jpg)

![](images/3d798adc343d8808b529d7e7bf1639d1218f305aacd3ba30046fa27af222b435.jpg)

![](images/9f2c745b08a5030c1277d911ddf734db163dda0f1619e764f1cba4539689233b.jpg)

![](images/0ef65bc76d01e887992545c8b8f8de311f73c1bd5583ec3792f23a9a7cfe5329.jpg)

![](images/2e315df15a10c9ae9e8503927c2bee017a7c1654423e2effa0e2265da859cbc8.jpg)

![](images/4acfcfa9586b1c059c639f86299c404678028cdfddd31d0913504c70ae0918a1.jpg)

![](images/be4abeedf358d2b307fad62db525e056cfe4bfedc748fd45cd409d5b7df20dee.jpg)

![](images/7be5ae29697b5df1bc1adf5416989cd924bbe6752c24f4dd31e97c73e2ca645c.jpg)  
(b) FPHA Dataset

![](images/2132290ed95da1d966969e190630182d79807327f803752c045686c24983be6e.jpg)  
Figure 3: Examples of continuous video frames with hand type annotation. The hand type of each frame changes over time in the video sequence. In (a) and (b), we show frames of the “Place Cappuccino” action (H2O [2]) in which the hand interacts with the box and the “High Five” action (FPHA [1]) in which the hand interacts with the other’s hand, respectively.

![](images/6c38f90d1d44cede6c18a82801d6deff6a220ae6d0472813b669937b2b6154a5.jpg)

![](images/56a4b408bd2258d30ff7dbf4cadec895af3f28220310e77fd1cb15ee40f79285.jpg)

![](images/a3989fe98707a4c2f4a03c500acf13d105695e06a8e8b443631fc79a5f5f969b.jpg)

(a) FPHA Dataset  
![](images/7a18563867c9b296f65f0a63364b5f6d9136d2bf25ac4ebfcdd6096997adfe57.jpg)  
(b) H2O Dataset  
Figure 5: Hand Type Distributions of the (a) FPHA [1] Dataset (Orange) and (b) H2O [2] Dataset (Blue). The histograms (left) show the number of frames per hand type; the x-axis represents the hand type, and the y-axis indicates the number of frames. The scatter plots (right) present the distribution of hand types across each action label; the x-axis represents the action of each dataset, and the y-axis indicates the hand type. The size of the circles illustrates how often each hand type appears within the hand action video sequences.

![](images/4cf8cd871e57cb60ee56c84f22d8f00fcc11051e79a57df3f48f54837033ef05.jpg)

![](images/42fbe3b04948157edea0889292bee9ce5b08e4939a773b6ded2c2f710ef25e7d.jpg)  
Hand Type (y) (a) H2O Dataset (Left)

![](images/b78664604bab47aaa964881fd8027d3c2f0043de9ce51da7e85c052476caf85c.jpg)

![](images/ddfba34e4b5a82f461c1d61882acd8c9b03fee2aa186a7c2e68c6463c657d1b3.jpg)  
(b) H2O Dataset (Right)  
Figure 6: Hand Type Distributions for the (a) Left hand and (b) Right hand of the H2O [2] Dataset. The histograms (left) show the number of frames per hand type; the x-axis represents the hand type, and the y-axis indicates the number of frames. The scatter plots (right) present the distribution of hand types across each action label; the x-axis represents the action of each dataset, and the y-axis indicates the hand type. The size of the circles illustrates how often each hand type appears within the hand action video sequences.

## References

[1] Guillermo Garcia-Hernando, Shanxin Yuan, Seungryul Baek, and Tae-Kyun Kim. Firstperson hand action benchmark with rgb-d videos and 3d hand pose annotations. In CVPR, 2018.

[2] Taein Kwon, Bugra Tekin, Jan Stühmer, Federica Bogo, and Marc Pollefeys. H2o: Two hands manipulating objects for first person interaction recognition. In ICCV, 2021.

[3] Yilin Wen, Hao Pan, Lei Yang, Jia Pan, Taku Komura, and Wenping Wang. Hierarchical temporal transformer for 3d hand pose estimation and action recognition from egocentric rgb videos. In CVPR, 2023.