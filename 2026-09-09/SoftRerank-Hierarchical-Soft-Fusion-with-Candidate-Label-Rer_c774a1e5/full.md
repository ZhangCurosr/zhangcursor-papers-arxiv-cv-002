# SoftRerank: Hierarchical Soft Fusion with Candidate-Label Reranking for Long-Tailed Micro-Action Recognition

Yichi Zhang<sup>∗</sup> University of Science and Technology of China Hefei, China charleszhang@mail.ustc.edu.cn

Lingsi Zhu University of Science and Technology of China Hefei, China ls-zhu24@mail.ustc.edu.cn

Qingsong Liu Unisound AI Technology Co., Ltd. Beijing, China liuqingsong@unisound.com

Zhichao Xia University of Science and Technology of China Hefei, China xiazc118@mail.ustc.edu.cn

Yuefeng Zou University of Science and Technology of China Hefei, China zouyuefeng.00@mail.ustc.edu.cn

Jianqing Sun Unisound AI Technology Co., Ltd. Beijing, China sunjianqing@unisound.com

Yanjun Chi University of Science and Technology of China Hefei, China yjChi@mail.ustc.edu.cn

Jun Yu<sup>†</sup>   
University of Science and Technology   
of China   
Hefei, China   
harryjun@ustc.edu.cn

Shengping Liu Unisound AI Technology Co., Ltd. Beijing, China liushengping@unisound.com

## Abstract

Micro-actions are subtle, low-intensity non-verbal behaviors that provide cues to fine-grained human states, including emotions and intentions. Recognizing them remains dificult because they are brief, contain weak visual changes, and often exhibit similar motion patterns across categories. This paper addresses these challenges with a fine-grained micro-action recognition method that combines full fine-tuning of InternVideo2.5, hierarchical soft fusion, and a lightweight candidate-label reranker. For the long-tailed label distribution in MA-52, we use class-balanced sampling and inversefrequency reweighting to reduce the efect of frequent classes during training. We fine-tune InternVideo2.5 end to end and attach coarse and group-conditional fine-grained classification heads to the shared video representation, improving the consistency between coarse and fine predictions. For ambiguous samples, the candidate-label reranker uses hard samples and video-label match ing to focus on easily confused fine-grained actions. Experiments validate the proposed method, which achieves a 79.99% F1-mean on MA-52 and ranks first in the 3rd Micro-Action Analysis Grand Challenge at ACM Multimedia 2026.

## CCS Concepts

• Computing methodologies → Artificial intelligence.

## Keywords

Micro-Action Recognition, Long-Tailed Learning, Hierarchical Soft Fusion, Candidate-Label Reranking

ACM Reference Format: Yichi Zhang, Zhichao Xia, Yanjun Chi, Lingsi Zhu, Yuefeng Zou, Jun Yu, Qingsong Liu, Jianqing Sun, and Shengping Liu. 2026. SoftRerank: Hierarchical Soft Fusion with Candidate-Label Reranking for Long-Tailed Micro-Action Recognition. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3767308.3837680

## 1 Introduction

Action recognition is a fundamental problem in computer vision, with applications in intelligent surveillance, medical assistance, human-computer interaction, and behavior analysis [8, 17, 18]. Micro-Action Recognition (MAR) [5, 9] focuses on subtle, shortduration movements, such as slight head motions, small hand gestures, and minor posture changes, which provide behavioral and psychological cues for emotion understanding, psychological assessment, and safety monitoring [7, 16, 19]. Unlike conventional actions, micro-actions exhibit weak motion intensity, sparse temporal evidence, and small inter-class variations, requiring models to capture brief, localized motion patterns while distinguishing visually similar categories. MA-52 is a representative MAR benchmark containing 52 fine-grained categories organized into 7 coarsegrained body-motion groups, with each fine-grained label assigned to a unique parent group [19, 23]. This hierarchy reflects the natural organization of micro-actions and supports evaluation at both body-motion and fine-action levels, but also requires separating actions across groups and distinguishing similar actions within each group. For example, turning the head, nodding, shaking the head, tilting the head, and raising the head share similar spatial regions and motion trajectories, making their discrimination dependent on subtle diferences in motion direction, temporal evolution, and visual context.

Recognition on MA-52 involves three closely related challenges. First, the long-tailed fine-grained label distribution allows frequent classes to dominate optimization and limits the representation qual ity of rare classes, particularly within body-motion groups where visually similar categories compete under substantially diferent sample frequencies. Second, discriminative evidence may occur in only a few frames and reflect minor appearance or motion changes, requiring precise spatiotemporal representation learning. Third, although the coarse-to-fine hierarchy provides useful structural information, incorporating it is non-trivial. A flat 52-way classifier ignores label-space organization, whereas a hard coarse-to-fine cascade restricts fine-grained prediction to one coarse group and may propagate an incorrect coarse prediction to the final decision. Moreover, even a strong fine-tuned model can produce unstable top-1 predictions when several semantically related labels receive similar probabilities.

To address these challenges, we propose a unified MAR framework based on full end-to-end fine-tuning of the InternVideo2.5 backbone [32]. Its long-context and fine-grained spatiotemporal modeling capacity enables the shared video representation to adapt to subtle motion diferences without an additional temporal contextualization module. To reduce optimization bias under the longtailed distribution, training incorporates class-balanced sampling, inverse-frequency loss reweighting within each parent group, stronge augmentation for rare classes, and hierarchical Mixup, which interpolates coarse- and fine-level supervision according to the label hierarchy to maintain semantic consistency during data mixing. On the shared representation, hierarchical soft fusion performs hierarchyconsistent global prediction. A coarse head estimates probabilities over the 7 body-motion groups, while a group-conditional finegrained head models fine-grained categories within each group. Each fine-grained probability is factorized into its coarse-group and group-conditional probabilities. Unlike hard cascading, the final decision globally searches all 52 fine-grained categories, preserving hierarchical consistency without excluding labels outside the coarse top-1 group. This formulation uses coarse information to regularize fine-grained prediction while reducing error propagation from incorrect coarse decisions.

We further introduce a lightweight candidate-label reranker as a selective correction stage for ambiguous predictions. The reranker is trained using out-of-fold predictions, which provide hard samples from models that have not observed the corresponding training samples. For each hard sample, a compact candidate set is constructed from high-scoring incorrect predictions, labels within the same body-motion group, and frequently confused labels identified from confusion statistics. During inference, reranking is activated only for low-confidence predictions or those with a small margin between the two highest probabilities. The reranker compares the video representation with candidate-label representations and refines the decision within this restricted set. Together, hierarchical soft fusion and selective candidate reranking form a two-stage inference process that first performs hierarchy-consistent global prediction and then corrects residual ambiguities among a small set of confusing fine-grained actions.

Our contributions are as follows:

• We propose an end-to-end fine-tuned micro-action recognition framework based on InternVideo2.5, where imbalanceaware sampling, inverse-frequency reweighting, and rareclass augmentation are jointly used to improve representation learning on the long-tailed MA-52 dataset.

• We introduce a hierarchical soft fusion strategy to jointly model coarse body-motion groups and fine-grained action labels, maintaining coarse–fine consistency while reducing error propagation from hard coarse-to-fine decisions.

• We design a lightweight candidate-label reranker trained with out-of-fold hard samples and confusion-aware candi date sets, which refines ambiguous predictions by comparing video features with a small set oflikely confused action labels under limited additional computation.

• We achieve an F1-mean of 79.99% on MA-52, ranking first in the 3rd Micro-Action Analysis Grand Challenge at ACM Multimedia 2026.

![](images/b851d399bec8582a35f42fdf968437ef4600e52cfacb0c8bb2c766a3b342fccb.jpg)  
Figure 1: Overview of MA-52 and the imbalance-aware data preprocessing strategies, including class-balanced sampling, inverse-frequency reweighting, and stronger augmentation for rare classes.

## 2 Related Work

Skeleton- and Pose-Based Methods. These methods represent the body as a graph of joints and bones and feed pose sequences (keypoints) extracted from video into the model. The design efort goes into graph neural networks or spatiotemporal models that capture both the spatial relations between joints and their motion over time. Early SkelAct work [13, 14, 35, 36] combined skeletal cues with RGB appearance, while toolkits such as PYSKL [4] and OpenPose [1] made pose extraction practical. Recent methods include BlockGCN [42], ProtoGCN [21], DeGCN [25], and Hyperformer [41], which models higher-order joint dependencies using hypergraph selfattention. SeFAR [11] targets semi-supervised recognition, whereas DSTA-Net [28] decouples spatial and temporal attention. Skeletal input is robust to background, lighting, and appearance shifts and is cheap to process, but it depends on pose-estimation quality and struggles under occlusion, fast motion, and exactly the fine appearance details that micro-actions hinge on.

Multimodal Integration Methods. To get past the limits of any single modality, these methods fuse complementary signals into a more robust action representation. Our framework draws on contrastive vision-language work: CLIP [26], M2-CLIP [30], and TC-CLIP [12], the first emphasizing intra and inter-modal alignment and the latter focusing on temporal alignment. MMCL-Action [22] similarly relies on multimodal contrastive learning. ExACT [40] and the DailyDVS-200 benchmark [31] explore event-camera input, and MM-CDFSL [10] studies cross-domain few-shot learning across RGB, flow, and audio. DualActNet [38] uses a SlowFast design to fuse information at diferent temporal rates. For MAR specifically, MANet [7] outperformed nine competing methods, and PCAN [2, 15] reduces the ambiguity between visually similar micro-actions. More recently, multimodal fusion that combines body-level context with fine motion cues has become a common recipe for MA-52, with strong systems pairing a body-level backbone and a motion-level backbone rather than relying on one model alone [6].

![](images/1dd53789255526ef13247100deae781f66c8bbc2a9bd6b1cd1bddf0417e8f981.jpg)  
Figure 2: Overview of the proposed hierarchical micro-action recognition framework, which integrates imbalance-aware preprocessing, InternVideo2.5-based spatiotemporal representation learning, hierarchical coarse-to-fine soft fusion, and candidate-label reranking for accurate fine-grained micro-action recognition.

Video Foundation Models and Eficient Supervision. Mul timodal Large Language Models (MLLMs) have reshaped video understanding and, with it, action recognition. MA-FSAR [34] adds temporal awareness to CLIP through a Global Timing Adapter and Text-guided Prototyping under a PEFT budget, while Motion-Sight [3] uses a zero-shot "visual spotlight" that pairs trajectory tracking with motion-blur enhancement to sharpen perception of subtle object and camera motion. AAPL [37] reduces annotation efort through action-agnostic frame sampling and point-level supervision while remaining competitive across temporal action detection benchmarks. InternVideo2.5 [32] trains progressively, unifying masked video modeling with next-token prediction to capture structure and semantics at several levels, and STIA [39] aggregates spatiotemporal information for end-to-end online feature learning. The broader trend is toward larger video foundation models and instruction-tuned video LLMs being adapted for fine-grained motion, where the open question is less about raw capacity and more about fine-tuning eficiency and countering the long-tail bias inherited from web-scale pretraining [27].

## 3 Methodology

As shown in Figure 2, our method builds a unified MAR framework with a fully fine-tuned InternVideo2.5 backbone, using imbalanceaware sampling, inverse-frequency reweighting, and rare-class augmentation to improve representation learning under the long-tailed label distribution of MA-52. Based on the shared video representation, hierarchical soft fusion jointly models coarse body-motion groups and fine-grained action labels, and a lightweight candidatelabel reranker refines ambiguous predictions by comparing video features with a compact set of likely confused labels.

## 3.1 Data Preprocessing

Imbalance-Aware Training. MA-52 has a long-tailed fine-grained label distribution, and the hierarchy in Sec. 3.3 handles cross-group confusion but not the intra-group tail. We address the tail on the data side with four strategies.

Class-balanced sampling. Let $N _ { f }$ be the sample count of finegrained class $f .$ We sample from a smoothed distribution

$$
q _ { f } \propto N _ { f } ^ { \tau } , \tau \in [ 0 , 1 ] ,
$$

where �=1 is instance-balanced and �=0 is class-balanced. An intermediate � keeps head classes from dominating the gradient while avoiding repeated tail samples.

Inverse-frequency reweighting. We weight the fine-grained crossentropy by �<sub>�</sub> ∝ $N _ { f } ^ { - Y } , \gamma \in [ 0 , 1 ]$ , computed within each parent group. The group-conditional head then does not collapse onto the most frequent member of its group.

Stronger augmentation on rare classes. We scale augmentation intensity by class frequency. Tail classes receive wider color perturbation and a larger motion-amplification factor �, which enlarges their efective support without new data.

Hierarchical Mixup/CutMix. For two samples $( x _ { a } , f _ { a } )$ and $( x _ { b } , f _ { b } )$ mixed with coeficient $\lambda ,$ we interpolate labels at both tree levels,

$$
\tilde { y } ^ { \mathrm { f i n e } } = \lambda \mathbf { e } _ { f _ { a } } + \left( 1 - \lambda \right) \mathbf { e } _ { f _ { b } } , \qquad \tilde { y } ^ { \mathrm { c o a r s e } } = \lambda \mathbf { e } _ { c ( f _ { a } ) } + \left( 1 - \lambda \right) \mathbf { e } _ { c ( f _ { b } ) } .
$$

The mixed target respects the parent assignment $c ( \cdot )$ and supervises both heads. Cross-group mixing places soft samples on group boundaries, where soft fusion does its arbitration.

## 3.2 Backbone and Full Fine-tuning

We build our model on a single InternVideo2.5 backbone $\Phi ( \cdot ) _ { i }$ , which maps an input video � to a pooled representation $z = \Phi ( x )$ shared by the coarse and fine-grained heads of Sec. 3.3. Compared with the InternVideo2 backbone used in our previous solution, InternVideo2.5 provides stronger long-context and fine-grained spatiotemporal modeling, so the external Temporal Contextualization module pre viously needed to inject cross-frame context is no longer required; the backbone already captures the temporal cues that micro-actions depend on.

In contrast to the parameter-eficient fine-tuning adopted last year, we fine-tune the backbone end to end. Micro-actions difer from each other by subtle appearance and motion details, and a frozen or adapter-only backbone leaves the shared features only partially aligned with the task. Full fine-tuning lets the representation adapt to these fine distinctions, and, since both heads read from the same �, it also lets the coarse objective regularize the shared features in the direction that benefits the fine-grained task, consistent with the joint objective described in Section 3.3.

Full fine-tuning of a billion-scale backbone on a long-tailed dataset of this size is prone to overfitting, so we pair it with the data-side rebalancing and augmentation of Sec. 3.1 and a small, layer-wise decayed learning rate on the backbone. Frames are sampled from each clip, center-cropped and resized to $2 2 4 \times 2 2 4 ,$ , and encoded by Φ(·); the two heads are then attached on top of � as described in Sec. 3.3. Training uses mixed precision and activation checkpointing to keep the memory footprint of end-to-end tuning tractable.

## 3.3 Hierarchical Soft Fusion

Let � ∈ $C ~ = ~ \{ 0 , \ldots , 6 \}$ denote the coarse label and $f \in { \mathcal { F } } =$ $\{ 0 , \ldots , 5 1 \}$ the fine-grained label. Each fine-grained class has a unique parent $c ( f )$ , and the parents partition $\mathcal { F }$ into groups $\mathcal { G } _ { c } =$ $\{ f : c ( f ) = c \}$ . We share a single InternVideo2.5 backbone $\Phi ( \cdot )$ and attach two heads on top of the pooled feature $z = \Phi ( x ) { \mathrm { : } }$ : a coarse head producing $P _ { \theta _ { c } } ( c \mid x )$ over the seven groups, and a fine-grained head implemented as a group-conditional softmax that normalizes within each parent group and outputs $P _ { \theta _ { f } } \left( f \mid c ( f ) , x \right)$

Following the tree structure, we recover the fine distribution by the chain rule and take the prediction over the full label set:

$$
P ( f \mid x ) = P { \big ( } c ( f ) \mid x { \big ) } P { \big ( } f \mid c ( f ) , x { \big ) } , \qquad { \hat { f } } = \arg \operatorname* { m a x } _ { f \in { \mathcal { F } } } P ( f \mid x ) .
$$

We term this combination soft fusion: the search ranges over all of $\mathcal { F }$ rather than being restricted to a single predicted group, so a fine class may be selected even when it lies outside the coarse top-1.

When the fine-grained head is instead a flat 52-way softmax $P _ { \theta _ { f } } ( f \mid x )$ , we fuse the two heads as a log-linear product of experts,

$$
s ( f ) = \log P _ { \theta _ { f } } ( f \mid x ) + \lambda \log P _ { \theta _ { c } } \big ( c ( f ) \mid x \big ) , \qquad \hat { f } = \arg \operatorname* { m a x } _ { f } s ( f ) ,
$$

where $\lambda \in \left[ 0 , 1 \right]$ weights the coarse signal and is selected on the validation set by macro-F1.

The two heads are trained jointly with

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { f i n e } } + \alpha \mathcal { L } _ { \mathrm { c o a r s e } } ,
$$

where $\mathcal { L } _ { \mathrm { c o a r s e } }$ is the cross-entropy on $P _ { \theta _ { c } } ( c \mid x )$ and $\mathcal { L } _ { \mathrm { f i n e } }$ is the cross-entropy on the group-conditional fine-grained head, paired with class-balanced reweighting so that rare fine-grained classes are not dominated within their group. The weight � is tuned on the validation set.

When a coarse–fine pair is required as output, we report the hierarchy-consistent pair $\textstyle ( c ( { \hat { f } } ) , { \hat { f } } )$ induced by the selected finegrained label.

## 3.4 Lightweight Candidate-Label Reranker

Fine-grained micro-action recognition is challenging because different categories can correspond to very similar motion patterns. A standard 52-class model may assign comparable probabilities to several related actions, especially when the visual diferences between them are subtle. In such cases, the top-1 prediction can be unstable. Repeating a full classification over all 52 categories would add unnecessary computation and would still not explicitly focus on the labels that are most likely to be confused. We address this problem with a lightweight candidate-label reranker. The module does not replace the base action model and does not perform another full 52-class prediction. Instead, it reranks only the candidate labels produced by the fine-tuned model and selects the most likely action category from this restricted set.

To obtain reliable hard samples for training the reranker, we do not directly use the incorrect predictions generated by the finetuned model on the training set. These predictions may be afected by sample memorization, because the model has already seen the same samples during optimization. The resulting errors can therefore be biased and may not match the error distribution at test time. We adopt an out-of-fold prediction strategy to reduce this bias. Specifically, the training set is divided into $F \operatorname { f o l d } s .$ . In each round, the fine-tuned model is trained on $F - 1$ folds and evaluated on the remaining fold. By rotating the validation fold, each training sample receives a prediction from a model that has not used this sample for training. We record the top-1 prediction, the top-� predictions, logits, prediction probabilities, and prediction correctness. Compared with direct inference on the training set, out-of-fold prediction provides a more realistic source of hard samples and reduces the bias caused by overfitting to the training data.

Based on the out-of-fold predictions, we construct a candidate label set for each sample. The candidate set always includes the ground-truth action label and several hard negative labels that are likely to be confused with it. These hard negatives are drawn from three sources. The first source is the high-scoring but incorrect top-� predictions produced by the fine-tuned model. The second source consists of action labels from the same body-motion group as the ground-truth label. The third source consists of frequently confused labels estimated from the validation-set confusion matrix. For example, for a sample with the ground-truth label shaking head, if the fine-tuned model often predicts turning head or nodding, the candidate set can be constructed as {shaking head, turning head, nodding, tilting head, head up}. This construction encourages the reranker to learn fine-grained diferences among similar actions, rather than relying on random negative labels that are already easy to separate.

The training data for the reranker are not restricted to samples that are incorrectly predicted by the fine-tuned model. Although incorrect samples are useful for learning how to correct errors made by the base model, using only such samples may introduce a correction bias. The model may then learn to override the prediction of the base model even when the original prediction is correct. To reduce this risk, the training data include both incorrect samples and a subset of low-confidence correct samples. Incorrect samples provide supervision for error correction, whereas low-confidence correct samples help the reranker learn when the original prediction should be preserved. This design better matches the inference setting, where both wrong predictions and uncertain but correct predictions may occur.

The reranker is implemented as a lightweight video-label matching network. For each video, we first extract a global video feature v from the fine-tuned video encoder. For each candidate action label $c _ { i } ,$ , a learnable label embedding matrix is used to obtain the label representation $\mathbf { t } _ { i } .$ Since MA-52 has a fixed label space, learnable label embeddings can capture implicit relations among action categories in the dataset. This design is suitable for fine-grained reranking, because the model only needs to compare a small set of candidate labels that are likely to be confused with each other.

Given the video feature v and the candidate label representation $\mathbf { t } _ { i } ,$ we first project them into the same feature dimension. We then construct an interaction feature by concatenating the projected video feature, the projected label feature, their element-wise product, and their absolute diference:

$$
\mathbf { z } _ { i } = \left[ \mathbf { v } , \mathbf { t } _ { i } , \mathbf { v } \odot \mathbf { t } _ { i } , | \mathbf { v } - \mathbf { t } _ { i } | \right] .
$$

The element-wise product models the compatibility between the video and the candidate label, whereas the absolute diference captures the discrepancy between their features. The interaction feature $\mathbf { z } _ { i }$ is fed into a lightweight MLP scorer, which outputs a matching score $s _ { i }$ for the video-label pair $( \mathbf { v } , c _ { i } )$ . For a candidate set containing � labels, the reranker produces � matching scores and ranks all candidate labels within the set.

The model is optimized using a candidate-level cross-entropy loss. For each training sample, the supervisory signal is defined as the relative position of the ground-truth label within the candi date set, rather than its original index in the 52-class label space. For instance, given the candidate set {shaking head, turning head, nodding}, if the ground-truth label shaking head is ranked first, the corresponding target index is set to 0. The loss is defined as

$$
\mathcal { L } _ { \mathrm { r a n k } } = - \log \frac { \exp ( s _ { y } ) } { \sum _ { i = 1 } ^ { M } \exp ( s _ { i } ) } .
$$

where $s _ { y }$ denotes the matching score of the ground-truth candidate. This objective directly optimizes the relative ranking within the candidate set and encourages the score of the ground-truth label to be higher than the scores of other candidate labels.

Table 1: Comparison of F1 scores on the 1,138-samples MAC 2026 test subset.
<table><tr><td>Method</td><td> $\operatorname { F 1 } _ { m a c r o } ^ { b o d y }$ </td><td> $\operatorname { F } 1 _ { m i c r o } ^ { b o d y }$ </td><td> $\mathrm { F } 1 _ { m a c r o } ^ { a c t i o n }$ </td><td> $\operatorname { F } 1 _ { m i c r o } ^ { a c t i o n }$ </td><td> $\mathrm { F } 1 _ { \mathrm { m e a n } }$ </td></tr><tr><td>VideoSwinT[24]</td><td>70.13</td><td>74.16</td><td>48.14</td><td>56.15</td><td>62.14</td></tr><tr><td>Li et al.[20]</td><td>83.49</td><td>86.38</td><td>64.47</td><td>71.79</td><td>76.54</td></tr><tr><td>Wang et al.[29]</td><td>81.47</td><td>87.13</td><td>64.46</td><td>74.85</td><td>76.98</td></tr><tr><td>STIA[39]</td><td>81.74</td><td>85.50</td><td>60.74</td><td>69.42</td><td>74.35</td></tr><tr><td>HiMEAR[33]</td><td>84.06</td><td>86.38</td><td>68.31</td><td>72.23</td><td>77.75</td></tr><tr><td>Our Method</td><td>85.90</td><td>87.96</td><td>71.57</td><td>74.52</td><td>79.99</td></tr></table>

During inference, the base model first predicts the probability distribution over the 52 action classes for an input video. The top-� classes are selected as candidate labels. For high-confidence samples, the top-1 prediction of the base model is directly retained as the final prediction. For low-confidence samples, or for samples where the probability gap between the top-1 and top-2 predictions is small, the reranker is used to refine the decision. It computes the matching score for each top-� candidate label and selects the candidate with the highest score as the final action prediction. In this way, the proposed module improves discrimination among confusing finegrained actions while keeping the additional computation limited to ambiguous samples

## 4 Experiments

## 4.1 Datasets and Metrics

The MA-52 dataset is a recent benchmark for micro-action recognition and provides a data basis for analyzing subtle human behaviors. It contains 22,422 video samples from 205 subjects, with a total duration of 12.29 hours. In the Micro-Action Analysis Grand Challenge, the final rankings are determined using a subset of 1,138 samples drawn from the MA-52 test set, which contains 5,586 samples in total.

Because sample counts across micro-action categories in MA-52 follow a long-tailed distribution, this study uses $F 1 _ { m i c r o }$ and $F 1 _ { m a c r o }$ to evaluate model performance. $F 1 _ { m i c r o }$ measures overall recognition performance at the sample level, whereas $F 1 _ { m a c r o }$ measures average recognition performance across categories. Using both metrics reduces the bias of $F 1 _ { m i c r o }$ toward frequent categories and gives a more reliable evaluation of categories with limited samples. To provide a unified assessment of coarse-grained action recognition and fine-grained micro-action recognition, this paper uses $F 1 _ { m e a n }$ as the final evaluation metric.

$$
F 1 _ { m e a n } = ( F 1 _ { m a c r o } ^ { b o d y } + F 1 _ { m i c r o } ^ { b o d y } + F 1 _ { m a c r o } ^ { a c t i o n } + F 1 _ { m i c r o } ^ { a c t i o n } ) / 4
$$

## 4.2 Implementation Details

We implement our method in PyTorch 2.1 with CUDA 12.1. For backbone fine-tuning, we choose the learning rate by grid search over 3e-6 to 3e-7 to keep optimization stable. We use mixed-precision training and activation checkpointing by default to reduce memory cost during training. For each training video, we center-crop each frame and resize it to 224 × 224 pixels before feeding it into the model. All experiments are run on 8 A100 GPUs.

Table 2: The ablation of fusion strategies on the MA-52 validation dataset
<table><tr><td>Strategy</td><td> $\operatorname { F 1 } _ { m a c r o } ^ { b o d y }$ </td><td> $\operatorname { F } 1 _ { m i c r o } ^ { b o d y }$ </td><td> $\mathrm { F } 1 _ { m a c r o } ^ { a c t i o n }$ </td><td> $\operatorname { F } 1 _ { m i c r o } ^ { a c t i o n }$ </td><td> $\mathrm { F } 1 _ { \mathrm { m e a n } }$ </td></tr><tr><td>Fine head only</td><td>81.54</td><td>85.22</td><td>65.03</td><td>69.22</td><td>75.25</td></tr><tr><td>Hard Cascade</td><td>84.72</td><td>86.70</td><td>67.25</td><td>70.15</td><td>77.20</td></tr><tr><td>Product-of-Experts</td><td>83.12</td><td>86.62</td><td>68.23</td><td>71.22</td><td>77.30</td></tr><tr><td>Soft Fusion</td><td>85.60</td><td>87.99</td><td>69.50</td><td>73.29</td><td>79.09</td></tr></table>

Table 3: The ablation of data processing and reranker on the validation MA-52 dataset
<table><tr><td>Method</td><td> $\operatorname { F 1 } _ { m a c r o } ^ { b o d y }$ </td><td> $\operatorname { F } 1 _ { m i c r o } ^ { b o d y }$ </td><td> $\mathrm { F } 1 _ { m a c r o } ^ { a c t i o n }$ </td><td> $\operatorname { F } 1 _ { m i c r o } ^ { a c t i o n }$ </td><td> $\mathrm { F } 1 _ { \mathrm { m e a n } }$ </td></tr><tr><td>Fine-tuning</td><td>85.60</td><td>87.99</td><td>69.50</td><td>73.29</td><td>79.09</td></tr><tr><td>+ data process</td><td>85.55</td><td>88.71</td><td>69.51</td><td>75.18</td><td>79.74</td></tr><tr><td>+ Reranker</td><td>86.84</td><td>88.71</td><td>71.51</td><td>74.67</td><td>80.44</td></tr><tr><td>+ Both</td><td>87.71</td><td>89.47</td><td>72.48</td><td>75.50</td><td>81.29</td></tr></table>

## 4.3 Comparison with State-of-the-Art Methods

We report our final results on the MA-52 test set (1,138 samples) and provide a comprehensive comparison with representative microaction recognition approaches. As shown in Table 1, our method achieves the best body-level, action-level, and overall performance, consistently outperforming existing approaches. The framework integrates end-to-end backbone fine-tuning, imbalance-aware training, hierarchical soft fusion, and selective candidate-label reranking. End-to-end fine-tuning adapts video representations to subtle spatiotemporal variations, while imbalance-aware strategies improve recognition reliability under long-tailed class distributions. Hierarchical soft fusion exploits the relationship between body-motion groups and fine-grained actions, whereas selective reranking resolves residual ambiguities among visually similar labels. Together, these components address representation adaptation, label imbalance, hierarchical consistency, and local class confusion within a unified framework. These results validate the integration of adaptive video representation learning, hierarchy-consistent prediction, and targeted ambiguity correction for fine-grained micro-action recognition.

## 4.4 Ablation Analysis

Efect of hierarchical fusion strategies. Table 2 compares strategies for exploiting the hierarchy between body-motion groups and fine-grained actions. All hierarchy-aware strategies outperform the fine-grained-head-only baseline, confirming that body-level supervision provides efective structural guidance. Hard cascade restricts predictions to one coarse group and may propagate coarse-level errors. Product-of-Experts avoids this constraint by combining coarse- and fine-level predictions but does not explicitly model their group-conditional dependency. Soft fusion performs best, in creasing $\mathrm { F } 1 _ { \mathrm { m e a n } }$ from 75.25 to 79.09 by combining coarse-group and group-conditional fine-grained probabilities while preserving global competition among action classes. Compared with the finegrained-head-only baseline, it also improves both body-level and action-level metrics, demonstrating consistent benefits across the two label levels. Soft fusion therefore maintains hierarchical consistency without prematurely excluding plausible actions and regularizes fine-grained classification through body-level predictions.

Efect of data preprocessing and the lightweight candidatelabel reranker. Table 3 evaluates the individual and combined contributions of imbalance-aware data preprocessing and selective candidate-label reranking. Data preprocessing increases F1<sub>mean</sub> from 79.09 to 79.74, mainly by improving body-level and action-level micro-F1, while both macro-F1 scores remain nearly unchanged. Specifically, action-level micro-F1 increases from 73.29 to 75.18. This suggests that class-balanced sampling, inverse-frequency reweighting, rare-class augmentation, and hierarchical mixup improve overall prediction reliability but are less efective at resolving residual confusion among fine-grained categories.

The reranker produces a larger gain in action-level macro-F1, increasing it from 69.50 to 71.51, and raises F1 to 80.44. This result shows that candidate-based correction is particularly efective for visually similar actions and uncertain predictions. By restricting comparisons to hard negatives, labels within the same body-motion group, and frequently confused categories, it directly targets errors remaining after global classification. Combining both components achieves the best performance across all metrics, with an overall $\mathrm { F } 1 _ { \mathrm { m e a n } }$ of 81.29. These complementary gains indicate that data preprocessing improves representation balance and reliability, whereas reranking refines ambiguous fine-grained decisions during inference. Together, they provide a more efective solution than either component alone.

## 5 Conclusion

This paper presents a unified micro-action recognition framework integrating full end-to-end fine-tuning of InternVideo2.5, hierarchical soft fusion, and a lightweight candidate-label reranker. To address the long-tailed label distribution of MA-52, class-balanced sampling and inverse-frequency reweighting are employed to mitigate the dominance of frequent classes during training. Intern-Video2.5 is fine-tuned end to end with coarse and group-conditional fine-grained classification heads over a shared video representation, thereby improving coarse-to-fine prediction consistency and reducing errors caused by premature coarse-level decisions. Hierarchical soft fusion preserves global competition among all finegrained classes while efectively incorporating coarse-group guidance. For ambiguous samples, the reranker leverages hard samples and candidate-level video-label matching to focus on easily confused fine-grained actions during final prediction. Our method achieves a 79.99% F1-mean on MA-52, surpassing the previous state of the art by 2.24 F1 points and ranking first in the MAR challenge.

## Acknowledgments

This work was supported by the Natural Science Foundation of China (62276242), Hefei Municipal Natural Science Foundation (HZR2431), CAAI-MindSpore Open Fund, developed on OpenI Community.

## References

[1] Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh. 2019. OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Afinity Fields. arXiv:1812.08008 [cs] doi:10.48550/arXiv.1812.08008

[2] Guoliang Chen, Fei Wang, Kun Li, Zhiliang Wu, Hehe Fan, Yi Yang, Meng Wang, and Dan Guo. 2024. Prototype learning for micro-gesture classification. arXiv preprint arXiv:2408.03097 (2024).

[3] Yipeng Du, Tiehan Fan, Kepan Nan, Rui Xie, Penghao Zhou, Xiang Li, Jian Yang, Zhenheng Yang, and Ying Tai. 2025. MotionSight: Boosting Fine-Grained Motion Understanding in Multimodal LLMs. arXiv:2506.01674 [cs.CV] https: //arxiv.org/abs/2506.01674

[4] Haodong Duan, Jiaqi Wang, Kai Chen, and Dahua Lin. 2022. PYSKL: Towards Good Practices for Skeleton Action Recognition. arXiv:2205.09443 [cs.CV] https: //arxiv.org/abs/2205.09443

[5] Jihao Gu, Kun Li, Fei Wang, Yanyan Wei, Zhiliang Wu, Hehe Fan, and Meng Wang. 2025. Motion matters: Motion-guided modulation network for skeleton-based micro-action recognition. In Proceedings of the 33rd ACM International Conference on Multimedia. 5461–5470.

[6] Jihao Gu, Fei Wang, Kun Li, Yanyan Wei, Zhiliang Wu, and Dan Guo. 2025. MM gesture: towards precise micro-gesture recognition through multimodal fusion. arXiv preprint arXiv:2507.08344 (2025).

[7] Dan Guo, Kun Li, Bin Hu, Yan Zhang, and Meng Wang. 2024. Benchmarking Micro-action Recognition: Dataset, Methods, and Applications. IEEE Transactions on Circuits and Systems for Video Technology 34, 7 (2024), 6238–6252.

[8] Dan Guo, Xiaobai Li, Kun Li, Haoyu Chen, Jingjing Hu, Guoying Zhao, Yi Yang, and Meng Wang. 2024. Mac 2024: Micro-action analysis grand challenge. In Proceedings of the 32nd ACM International Conference on Multimedia. 11304– 11305.

[9] Xiaochuan Guo, Jihao Gu, Haixu Liu, Yuxin Liu, Qi Wang, Yufei Wang, Fei Wang, Kun Li, and Dan Guo. 2026. Rethinking the Role of Feature Engineering and Learning Strategies in Few-Shot Hidden Emotion Recognition. arXiv preprint arXiv:2606.31249 (2026).

[10] Masashi Hatano, Ryo Hachiuma, Ryo Fujii, and Hideo Saito. 2025. Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition. Springer Nature Switzerland, 182–199.

[11] Yongle Huang, Haodong Chen, Zhenbang Xu, Zihan Jia, Haozhou Sun, and Dian Shao. 2025. SeFAR: Semi-Supervised Fine-Grained Action Recognition with Temporal Perturbation and Learning Stabilization. arXiv:2501.01245 [cs]

[12] Minji Kim, Dongyoon Han, Taekyung Kim, and Bohyung Han. 2024. Leveraging Temporal Contextualization for Video Action Recognition. arXiv:2404.09490 [cs]

[13] Chao Li, Qiaoyong Zhong, Di Xie, and Shiliang Pu. 2017. Skeleton-based Action Recognition with Convolutional Neural Networks. In 2017 IEEE International Conference on Multimedia & Expo Workshops. 597–600.

[14] Chao Li, Qiaoyong Zhong, Di Xie, and Shiliang Pu. 2018. Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierar chical Aggregation. In Proceedings ofthe 27th International Joint Conference on Artificial Intelligence. 786–792.

[15] Kun Li, Dan Guo, Guoliang Chen, Chunxiao Fan, Jingyuan Xu, Zhiliang Wu, Hehe Fan, and Meng Wang. 2025. Prototypical calibrating ambiguous samples for micro-action recognition. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 39. 4815–4823.

[16] Kun Li, Dan Guo, Guoliang Chen, Xinge Peng, and Meng Wang. 2023. Joint skeletal and semantic embedding loss for micro-gesture classification. arXiv preprint arXiv:2307.10624 (2023).

[17] Kun Li, Dan Guo, Jihao Gu, Pengyu Liu, Xiaobai Li, Haoyu Chen, Yanbin Hao, Guoying Zhao, and Meng Wang. 2026. MAC 2026: Advancing Micro-Action Analysis Towards Fine-Grained Understanding. In Proceedings ofthe 34th ACM International Conference on Multimedia.

[18] Kun Li, Dan Guo, Xiaobai Li, Haoyu Chen, Pengyu Liu, Fei Wang, Jingjing Hu, Guoying Zhao, and Meng Wang. 2025. MAC 2025: The 2nd Micro-Action Analysis Grand Challenge. In Proceedings of the 33rd ACM International Conference on Multimedia. 14216–14221.

[19] Kun Li, Pengyu Liu, Dan Guo, Fei Wang, Zhiliang Wu, Hehe Fan, and Meng Wang. 2025. Mmad: Multi-label micro-action detection in videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 13225–13236.

[20] Qiankun Li, Qiupu Chen, Huabao Chen, Feng He, Depeng Li, and Zhigang Zeng. 2025. Progressive Large-Scale Modeling via Temporal-Spatial Focus Connector for Micro-Action Recognition. In Proceedings ofthe 33rdACMInternational Conference on Multimedia. 14222–14228.

[21] Hongda Liu, Yunfan Liu, Min Ren, Hao Wang, Yunlong Wang, and Zhenan Sun. 2025. Revealing Key Details to See Diferences: A Novel Prototypical Perspective for Skeleton-based Action Recognition. In Proceedings of the Computer Vision and Pattern Recognition Conference. 29248–29257.

[22] Jinfu Liu, Chen Chen, and Mengyuan Liu. 2024. Multi-Modality Co-Learning for Eficient Skeleton-based Action Recognition. In Proceedings ofthe ACM Multimedia (ACM MM).

[23] Tingyi Liu, Kun Li, Fei Wang, Junjie Chen, Zhiliang Wu, Jihao Gu, Haixu Liu, and Dan Guo. 2026. Self-supervised Learning Matters: A Simple Ensemble Solution for Micro-Gesture Recognition. arXiv preprint arXiv:2606.09261 (2026).

[24] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. 2022. Video swin transformer. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 3202–3211.

[25] Woomin Myung, Nan Su, Jing-Hao Xue, and Guijin Wang. 2024. DeGCN: Deformable Graph Convolutional Networks for Skeleton-Based Action Recognition. IEEE Transactions on Image Processing 33 (2024), 2477–2490. doi:10.1109/tip.2024. 3378886

[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020 [cs.CV] https://arxiv.org/ abs/2103.00020

[27] Tuyun Shang, Yanbin Hao, Ming Pei, Kun Li, Huixia Ben, and Shuo Wang. 2025. Cross-modal Feature Enhancement and Contrastive Alignment for Micro-gesture Recognition. In Chinese Conference on Pattern Recognition and Computer Vision. 203–217.

[28] Lei Shi, Yifan Zhang, Jian Cheng, and Hanqing Lu. 2020. Decoupled Spatial-Temporal Attention Network for Skeleton-Based Action-Gesture Recognition. In Proceedings ofthe Asian Conference on Computer Vision (ACCV). 38–53. doi:10. 1007/978-3-030-69541-5\_3

[29] Chuang Wang, Weidong Chen, Xu Cui, Yiming Zhao, Zhaobo Qi, Pengqi Huang, Xinyan Liu, and Weigang Zhang. 2025. Combatting Data Imbalance and Noise in Micro-Action Recognition. In Proceedings of the 33rd ACM International Conference on Multimedia. 14229–14235.

[30] Mengmeng Wang, Jiazheng Xing, Boyuan Jiang, Jun Chen, Jianbiao Mei, Xingxing Zuo, Guang Dai, Jingdong Wang, and Yong Liu. 2024. M2-CLIP: A Multimodal, Multi-Task Adapting Framework for Video Action Recognition. arXiv:2401.11649 [cs] doi:10.48550/arXiv.2401.11649

[31] Qi Wang, Zhou Xu, Yuming Lin, Jingtao Ye, Hongsheng Li, Guangming Zhu, Syed Afaq Ali Shah, Mohammed Bennamoun, and Liang Zhang. 2024. DailyDVS-200: A Comprehensive Benchmark Dataset for Event-Based Action Recognition. arXiv:2407.05106 [cs] doi:10.48550/arXiv.2407.05106

[32] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, Min Dou, Kai Chen, Wenhai Wang, Yu Qiao, Yali Wang, and Limin Wang. 2025. InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling. arXiv preprint arXiv:2501.12386 (2025).

[33] Zhichao Xia, Yichi Zhang, Yanjun Chi, Lingsi Zhu, Mohan Jing, and Jun Yu. 2025. Hierarchical Multi-Feature Extraction and Aggregation for Micro-Action Recognition. In Proceedings of the 33rd ACM International Conference on Multimedia. 14236–14243.

[34] Jiazheng Xing, Chao Xu, Mengmeng Wang, Guang Dai, Baigui Sun, Yong Liu, Jingdong Wang, and Jian Zhao. 2024. MA-FSAR: Multimodal Adaptation of CLIP for Few-Shot Action Recognition. arXiv:2308.01532 [cs.CV] https://arxiv.org/ abs/2308.01532

[35] Kailin Xu, Fanfan Ye, Qiaoyong Zhong, and Di Xie. 2022. Topology-aware Convolutional Neural Network for Eficient Skeleton-based Action Recognition. In Proceedings ofthe AAAI Conference on Artificial Intelligence.

[36] Fanfan Ye, Shiliang Pu, Qiaoyong Zhong, Chao Li, Di Xie, and Huiming Tang. 2020. Dynamic GCN: Context-enriched Topology Learning for Skeleton-based Action Recognition. In Proceedings ofthe 28th ACM International Conference on Multimedia. 55–63.

[37] Shuhei M. Yoshida, Takashi Shibata, Makoto Terao, Takayuki Okatani, and Masashi Sugiyama. 2025. Action-Agnostic Point-Level Supervision for Temporal Action Detection. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 39. 9571–9579. doi:10.1609/aaai.v39i9.33037

[38] Churan Yu, Yiwei Ru, Zhenbo Xu, Huijia Wu, Hujiang Yang, and Zhaofeng He. 2024. DualActNet: Exploiting SlowFast Architecture for Micro-Action Recognition. In CCBR (2).

[39] Jun Yu, Mohan Jing, Guopeng Zhao, Keda Lu, Yifan Wang, Feng Zhao, Jiaqing Sun, Qingsong Liu, and Jiaen Liang. 2024. End-to-end Spatio-Temporal Information Aggregation For Micro-Action Detection. In Proceedings of the 32nd ACM International Conference on Multimedia. 11306–11312.

[40] Jiazhou Zhou, Xu Zheng, Yuanhuiyi Lyu, and Lin Wang. 2024. ExACT: Languageguided Conceptual Reasoning and Uncertainty Estimation for Event-based Action Recognition and More. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18633–18643.

[41] Yuxuan Zhou, Zhi-Qi Cheng, Chao Li, Yanwen Fang, Yifeng Geng, Xuansong Xie, and Margret Keuper. 2023. Hypergraph Transformer for Skeleton-Based Action Recognition. arXiv:2211.09590 [cs] doi:10.48550/arXiv.2211.09590

[42] Yuxuan Zhou, Xudong Yan, Zhi-Qi Cheng, Yan Yan, Qi Dai, and Xian-Sheng Hua. 2024. BlockGCN: Redefine Topology Awareness for Skeleton-Based Action Recognition. In CVPR.