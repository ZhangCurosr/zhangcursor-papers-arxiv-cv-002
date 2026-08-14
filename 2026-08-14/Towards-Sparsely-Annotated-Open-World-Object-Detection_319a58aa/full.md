# Towards Sparsely Annotated Open-World Object Detection

HeeJu Han , AJeong Kim , and Jinsun Park<sup>⋆</sup>

Pusan National University, Republic of Korea {hhjjoa2817, jeonk636, jspark}@pusan.ac.kr

Abstract. Real-world object detection operates under ambiguous supervision, where unlabeled regions may correspond to missing annotations of known objects or genuinely unknown categories. These challenges have been addressed separately in Sparsely Annotated Object Detection (SAOD) and Open-World Object Detection (OWOD). In practice, their co-occurrence remains an open problem. To address this problem, we introduce Sparsely Annotated Open-World Object Detection (SA-OWOD), a new task that jointly considers sparse supervision and the presence of unseen categories. We propose Dual-Perspective Object Discovery (DPOD), a unified framework that jointly models unlabeled known and unknown instances via two complementary mechanisms. The Known Target Recovery Module (KTRM) recovers supervision for unlabeled known instances and explicitly regularizes the feature space to separate known and unknown representations. Complementarily, the Dual-Disagreement Target Generator (DDTG) identifies reliable unknown candidates through cross-view semantic inconsistency. By integrating these modules, DPOD resolves contradictory supervision signals caused by ambiguous unlabeled regions. As a result, it prevents misclassification between known and unknown objects and stabilizes the decision boundaries. Experimental results on sparsely annotated open-world benchmarks demonstrate that the proposed method outperforms existing openworld detection methods, particularly in detecting unknown objects. The code is publicly available at: https://github.com/HelloHeeju/SA-OWOD

Keywords: Open-World Object Detection · Sparsely Annotated Object Detection · Ambiguous Supervision

## 1 Introduction

Object detection inevitably operates under incomplete and semantically open conditions. In practice, many object instances remain unlabeled due to annotation costs and human limitations. However, most existing object detection methods [1, 2, 8, 13, 17, 29] rely on two strong assumptions: (1) training datasets provide fully annotated labels without missing objects, and (2) the category set is closed, with no novel classes appearing after training. While recent progress has been driven by large-scale datasets [3, 7, 15, 40, 43] and advances in deep neural architectures, these assumptions rarely hold in realistic scenarios.

![](images/c55a1a51e7706d0132c6b01cb1f14cc2808d5eb6af7cd399202ac76bcd0ae223.jpg)  
Fig. 1: Object detection paradigms for discovering unlabeled data. (a) Open-World Object Detection aims to identify novel objects as unknown, assuming full annotation data. (b) Sparsely Annotated Object Detection focuses on recovering unlabeled known objects under sparse annotation settings, while assuming a closed set of categories. (c) Sparsely Annotated Open-World Object Detection considers sparse annotations and open-world scenarios simultaneously, where unlabeled regions may correspond to either unlabeled known objects or unknown objects.

Open-World Object Detection (OWOD) [5,14,24,26,49] aims to not only detect known objects but also identify unseen ones as unknown. To this end, existing OWOD methods adopt diverse strategies to model unknown objects, such as learning class-agnostic objectness scores and exploiting energy-based uncertainty measures. Nevertheless, these methods still depend on densely annotated training data, as illustrated in Fig. 1 (a). However, with sparse annotations where unlabeled known objects may exist, they are treated as background during training. As a result, the detector may incorrectly suppress valid known instances or confuse them with unknown objects, which blurs the decision boundary between known and unknown classes and degrades the accuracy of both.

On the other hand, Sparsely Annotated Object Detection (SAOD) [16,22,33, 38, 41] addresses object detection under incomplete annotation settings, where only a subset of object instances is labeled per image during training (i.e., sparse annotations). In this setting, missing annotations cause detectors to incorrectly treat unlabeled objects as background regions, leading to false-negative supervision during training. To address this issue, several studies [34, 36, 47] attempt to compensate for annotation sparsity by generating pseudo-labels from unlabeled regions. However, these methods are fundamentally developed under a closedworld assumption, as shown in Fig. 1 (b). Under this assumption, every object belongs to a predefined set of known categories. As a result, semantically novel objects outside the target classes are treated as background and therefore cannot be discovered.

Although SAOD and OWOD have been studied independently, real-world data contains both unlabeled known and unknown objects simultaneously. From the detector’s perspective, these instances appear as indistinguishable unlabeled regions during training. Without explicit supervision, the model cannot distinguish between unlabeled instances of known classes and unseen objects, leading to ambiguous training signals. To address this problem, we introduce a new task, Sparsely Annotated Open-World Object Detection (SA-OWOD), which assumes that unlabeled regions may contain both missing annotations of known categories and objects from unseen classes. As illustrated in Fig. 1 (c), SA-OWOD captures scenarios in which both types of unlabeled instances co-exist within a single image. Unlike OWOD and SAOD, which address these challenges in isolation, SA-OWOD explicitly models their interplay.

Solving SA-OWOD requires a framework that can both recover unlabeled known objects and identify unknown objects. For this purpose, we introduce Dual-Perspective Object Discovery (DPOD). DPOD comprises two complementary components: the Known Target Recovery Module (KTRM) and the Dual-Disagreement Target Generator (DDTG). KTRM recovers unlabeled known objects through pseudo-labeling and regularizes the decision boundary between known and unknown categories through feature-level alignment. Complementing this, DDTG identifies additional unknown candidates by exploiting crossview semantic disagreement. Proposals with high objectness but low logit similarity are treated as reliable unknowns. By coupling pseudo-label recovery with disagreement-driven unknown discovery and feature-level regularization, DPOD resolves the inherent ambiguity of unlabeled regions in SA-OWOD.

The main contributions of our work are summarized as follows:

– To the best of our knowledge, we are the first to formulate the Sparsely Annotated Open-World Object Detection (SA-OWOD) task. To address this challenge, we introduce a novel framework, DPOD.

– We propose two complementary modules, KTRM and DDTG, to resolve the ambiguity of unlabeled regions in SA-OWOD. KTRM recovers missing known objects to stabilize pseudo-label learning and refine decision boundaries, while DDTG discovers additional unknown instances via cross-view semantic disagreement.

– We construct sparsely annotated open-world benchmarks and an evaluation protocol that explicitly captures the dual nature of unlabeled regions. They allow quantitative evaluation of both missing known object recovery and unknown object discovery.

## 2 Related Work

Object detection has achieved remarkable progress, evolving from two-stage frameworks [11,30] and eficient one-stage detectors [19,29] to recent transformerbased architectures [2,48]. Despite their strong performance, these methods operate under a closed-world assumption — they recognize only predefined categories and treat all unlabeled or unknown objects as background. Consequently, these methods rely on static, fully annotated datasets. This limits their applicability to real-world scenarios, where annotations are often incomplete and novel categories continually emerge.

## 2.1 Open-World Object Detection (OWOD)

Open-World Object Detection (OWOD) [23, 37, 39, 44, 46] aims to not only recognize known object categories but also identify novel (i.e., unknown) objects and incrementally learn them over time.

Pseudo-labeling methods identify unknown objects and generate pseudolabels as additional supervision. ORE [14] introduces the OWOD setting and generates unknown pseudo-labels from high-objectness proposals. It further enhances unknown modeling using contrastive clustering and energy-based identification. OW-DETR [9] extends Deformable DETR [48] with attention-guided pseudo-labeling and explicit supervision to distinguish unknown objects from background regions.

Class-agnostic methods focus on objectness rather than class-specific supervision, treating both known and unknown objects as foreground. PROB [49] introduces a probabilistic objectness head to generalize from known to unseen categories. RandBox [35] mitigates the bias of label-dependent proposal generation by adopting randomly generated region proposals. OrthogonalDet [31] decouples objectness and classification by enforcing orthogonality between objectness and class features. CROWD [25] discovers diverse unknown instances via a submodular gain objective and then learns disentangled representations of known and unknown classes to reduce confusion.

However, despite their efectiveness, most existing OWOD methods still depend on densely annotated data. Under sparse annotation settings, unlabeled known objects obscure the boundary between known and unknown categories, leading to degraded detection performance.

## 2.2 Sparsely Annotated Object Detection (SAOD)

Sparsely Annotated Object Detection (SAOD) [10, 28, 32, 42, 45] addresses object detection when only a subset of object instances is annotated per image, resulting in sparse supervision. In such scenarios, detectors incorrectly interpret unlabeled objects as background regions, introducing false-negative supervision during training.

To alleviate this issue, several studies leverage pseudo-labeling to compensate for missing annotations. uDenseTeacher [47] utilizes dense pseudo-labels generated from the teacher’s raw dense predictions instead of sparse box-level pseudo-labels with post-processing. Co-mining [34] jointly trains two detectors that exchange each other’s high-confidence predictions. Co-Student [36] adopts a collaborative strong–weak student framework. The two students leverage each other’s predictions, while a teacher model refines them to provide more reliable supervision.

![](images/11ccced782e9f33c9992446644565fe74047c09a731e45bf1e2f18f9ffb669c3.jpg)  
Fig. 2: Overview of DPOD. (a) KTRM recovers supervision for unlabeled known objects and enforces known–unknown feature separation. (b) DDTG identifies additional unknown candidates via cross-view semantic inconsistency. The two signals jointly strengthen known–unknown separation and stabilize feature representations.

However, these methods are designed under a closed-world assumption, where every object belongs to a predefined class set. As a result, objects from previously unseen categories are implicitly treated as background, preventing their detection.

## 3 Method

## 3.1 SA-OWOD Problem Definition

In this work, we introduce a new task, SA-OWOD, which extends OWOD to a sparsely annotated scenario, as shown in Fig. 2. In the standard OWOD formulation [14], a detector is trained to recognize known classes, identify unseen objects as unknown, and incrementally incorporate them over time. Unlike conventional OWOD, SA-OWOD assumes partial supervision within known classes: not all instances of known categories are annotated during training. Consequently, unlabeled known objects and truly unknown instances coexist without labels, yet must be assigned distinct semantic roles. This ambiguity constitutes the core challenge of SA-OWOD.

Formally, the SA-OWOD is defined as follows: At stage t, a detector $f _ { t }$ is trained on a dataset $\mathcal { D } _ { t } = \{ \mathcal { T } _ { t } , \mathcal { L } _ { t } \}$ , where $\mathcal { T } _ { t }$ and $\mathcal { L } _ { t }$ denote a set of training images and their corresponding partial annotations, respectively. The annotations are provided only for the known class set ${ { K } _ { t } } \mathrm { ~ = ~ } \{ 1 , 2 , . . . , C \}$ . However, not all instances belonging to $\textstyle { \mathcal { K } } _ { t }$ are annotated; a subset of known-class objects may remain unlabeled in $\mathcal { T } _ { t }$ . In addition, objects from the unknown class set ${ \mathcal { U } } = \{ C + 1 , C + 2 , . . . \}$ may appear without labels during training. Consequently, unlabeled regions may correspond to either missing annotations of known classes or objects from unknown categories.

Similar to OWOD, the problem follows an incremental learning protocol. At each stage, a subset of previously detected unknown classes $\bar { \mathcal { U } } _ { t } \subset \mathcal { U }$ is annotated and incorporated into the known class set $\boldsymbol { \mathcal { K } } _ { t + 1 } = \boldsymbol { \mathcal { K } } _ { t } \cup \boldsymbol { \bar { \mathcal { U } } } _ { t }$ . Due to memory and computational constraints, the model is updated from $f _ { t }$ to $f _ { t + 1 }$ using limited samples from previously known classes rather than retraining from scratch. This process is repeated over T stages, requiring the detector to continuously discover novel objects while mitigating catastrophic forgetting.

In summary, SA-OWOD requires the detector to simultaneously: (1) recover supervision for unlabeled known instances and (2) detect truly unknown objects.

## 3.2 Known Target Recovery Module (KTRM)

In SA-OWOD, unlabeled regions may correspond to known objects that were missed during annotation. If left unaddressed, these unlabeled known instances are treated as background during training. This introduces false-negative supervision and blurs the decision boundary between known and unknown categories. KTRM is designed to explicitly resolve this ambiguity through two complementary mechanisms: pseudo-label recovery for unlabeled known objects and representation-level separation between unlabeled known and unknown proposals. In sparsely annotated open-world settings, supervision for known classes is inherently incomplete, which makes classifier-based pseudo-labeling unreliable due to ambiguous decision boundaries. To alleviate this issue, we adopt a classagnostic open-world detection framework based on CROWD [25], which produces object-centric proposals.

Based on these object-centric proposals, we recover supervision for unlabeled known objects using a pseudo-label generation strategy inspired by Co-Student [36]. A student detector first generates predictions from multiple augmented views of the same image. These predictions are then refined by a teacher model through consistency-aware filtering, producing a reliable unlabeled known proposal set $\mathcal { R } _ { u k }$ . We treat the elements of $\mathcal { R } _ { u k }$ as pseudo-labels and combine them with the sparse ground-truth annotations to compensate for missing labels in known categories. Each pseudo-label $p \in \mathcal { R } _ { u k }$ provides a class label $y _ { p } ^ { u k }$ and a bounding box $\mathbf { b } _ { p } ^ { u k }$ . After that, each pseudo-label $p$ searches for the object proposal $r _ { p }$ with the largest IoU and they are used for the training via classification loss $\ell _ { c l s }$ and regression loss $\ell _ { r e g }$ . The pseudo-label supervision is defined as:

$$
\mathcal { L } _ { u k } = \frac { 1 } { \vert \mathcal { R } _ { u k } \vert } \sum _ { p \in \mathcal { R } _ { u k } } \Big ( \ell _ { c l s } \big ( \mathbf { z } _ { r _ { p } } , y _ { p } ^ { u k } \big ) + \ell _ { r e g } \big ( \mathbf { b } _ { r _ { p } } , \mathbf { b } _ { p } ^ { u k } \big ) \Big ) ,\tag{1}
$$

where $\mathbf { z } _ { r _ { p } }$ and $\mathbf { b } _ { r _ { p } }$ denote class logits and bounding box of $r _ { p } ,$ respectively.

While pseudo-label recovery compensates for missing supervision, feature representations of unlabeled known objects can still overlap with unknown instances in the embedding space. To enforce representation-level separation, we adopt a conditional gain objective inspired by the Facility-Location formulation in CROWD. We adopt the Facility-Location formulation because it encourages each proposal to be separated from its nearest unknown counterpart rather than considering all unknown proposals simultaneously. In contrast, distance-based objectives aggregate similarities across multiple unknown instances, which tends to blur local decision boundaries and weaken fine-grained discrimination.

Let $\mathcal { R } _ { k } , \mathcal { R } _ { u k }$ , and $\mathcal { R } _ { u }$ denote labeled known (from sparse ground-truth annotations), unlabeled known (from pseudo-labeling), and unknown proposal sets, respectively. We define the unified known set as:

$$
\mathcal { A } = \mathcal { R } _ { k } \cup \mathcal { R } _ { u k } .\tag{2}
$$

Let V denote the set of all proposals. The separation objective is formulated as a facility-location conditional gain:

$$
\mathcal { L } _ { s e p } = - \frac { 1 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } \operatorname* { m a x } \bigg ( 0 , \operatorname* { m a x } _ { j \in \mathcal { A } } s _ { i j } - \eta \operatorname* { m a x } _ { j \in \mathcal { R } _ { u } } s _ { i j } \bigg ) ,\tag{3}
$$

where $s _ { i j }$ denotes a similarity metric between i-th and j-th proposals, and η is a weighting factor that encourages separation between known and unknown features.

Minimizing $\mathcal { L } _ { s e p }$ encourages known proposals to move away from unknown regions in the embedding space, thereby reducing feature entanglement and stabilizing decision boundaries under sparse supervision.

The overall KTRM objective is defined as:

$$
\mathcal { L } _ { K T R M } = \alpha _ { u k } \mathcal { L } _ { u k } + \alpha _ { s e p } \mathcal { L } _ { s e p } ,\tag{4}
$$

where $\alpha _ { u k }$ and $\alpha _ { s e p }$ are balancing hyperparameters.

## 3.3 Dual-Disagreement Target Generator (DDTG)

In open-world settings, unknown objects often manifest as regions where semantic predictions are inherently unstable [31]. Unlike known categories, unknown objects tend to reside near ambiguous decision boundaries. Motivated by this observation, we introduce the Dual-Disagreement Target Generator (DDTG), summarized in Algorithm 1. DDTG identifies unknown object candidates by measuring cross-view semantic inconsistency. Such inconsistency typically arises from unstable regions in the classification space associated with unknown objects.

Given two views of the same image, the detector produces class logits for each object proposal. We compare the class logit vectors obtained from the two views to identify prediction disagreement. However, reliable comparison requires that the corresponding regions of interest (RoIs) refer to the same spatial region.

Algorithm 1 Procedure for Selecting Unknown Candidates in DDTG   
Require: Detector $f _ { \theta }$ , two views $\{ x ^ { s _ { 1 } } , x ^ { s _ { 2 } } \}$ , objectness threshold $\tau _ { o } ,$ similarity thresh  
old $\tau _ { s }$   
Ensure: Disagreement proposal set $\mathcal { R } _ { d i s }$   
1: Initialize $\mathcal { R } _ { d i s } \gets \emptyset$   
2: for each ordered pair $( x ^ { i } , x ^ { j } ) \in \{ ( x ^ { s _ { 1 } } , x ^ { s _ { 2 } } ) , ( x ^ { s _ { 2 } } , x ^ { s _ { 1 } } ) \}$ do   
3: Obtain region proposals ${ \mathcal { R } } ^ { a }$ from $x ^ { a }$   
4: for each proposal $\boldsymbol { r } \in \mathcal { R } ^ { a }$ do   
5: Project r to $x ^ { j }$   
6: Extract logits $\mathbf { z } _ { r } ^ { i } = f _ { \theta } ( x ^ { i } , r ) , \mathbf { z } _ { r } ^ { j } = f _ { \theta } ( x ^ { j } , r )$   
7: Compute similarity $\begin{array} { r } { \boldsymbol { S } ( \boldsymbol { r } ) = \frac { ( \mathbf { z } _ { r } ^ { i } ) ^ { \top } \mathbf { z } _ { r } ^ { j } } { \| \mathbf { z } _ { r } ^ { i } \| _ { 2 } \| \mathbf { z } _ { r } ^ { j } \| _ { 2 } } } \end{array}$   
8: if $o _ { r } \ge \tau _ { o }$ and $S ( r ) \le \tau _ { s }$ then   
9: $\mathcal { R } _ { d i s } \gets \mathcal { R } _ { d i s } \cup \{ r \}$   
10: end if   
11: end for   
12: end for   
13: return $\mathcal { R } _ { d i s }$

Naively matching proposals by intersection-over-union (IoU) can be unstable, as even minor localization shifts may lead to substantially diferent RoI features. To ensure consistent cross-view comparison, we generate each RoI from a reference view and project it to the other view to extract geometrically aligned RoI features. Prediction disagreement is then quantified by computing the cosine similarity between the aligned class logit vectors.

Given a proposal $r ,$ let $\mathbf { z } _ { r } ^ { i } \in \mathbb { R } ^ { C }$ and $\mathbf { z } _ { r } ^ { j } \in \mathbb { R } ^ { C }$ denote the classification logits predicted from two diferent views. We measure their semantic consistency using cosine similarity:

$$
\boldsymbol { S } ( \boldsymbol { r } ) = \frac { \left( \mathbf { z } _ { \boldsymbol { r } } ^ { i } \right) ^ { \top } \mathbf { z } _ { \boldsymbol { r } } ^ { j } } { \| \mathbf { z } _ { \boldsymbol { r } } ^ { i } \| _ { 2 } \| \mathbf { z } _ { \boldsymbol { r } } ^ { j } \| _ { 2 } } .\tag{5}
$$

The set of disagreement proposals is defined as:

$$
\mathcal { R } _ { d i s } = \Big \{ r \in \mathcal { R } \Big | o _ { r } \geq \tau _ { o } , \ S ( r ) \leq \tau _ { s } \Big \} ,\tag{6}
$$

where $o _ { r }$ denotes the objectness score of proposal $r , \tau _ { o }$ is the objectness threshold, and $\tau _ { s }$ is the similarity threshold. Only proposals that are suficiently objectlike but exhibit low semantic consistency are treated as disagreement candidates. Similar to KTRM, we apply both classification and separation objectives to the disagreement set in order to enforce consistent unknown prediction. The resulting DDTG loss is defined as:

$$
\begin{array} { r l } & { \mathcal { L } _ { D D T G } = \beta _ { c l s } \frac { 1 } { | \mathcal { R } _ { d i s } | } \displaystyle \sum _ { d \in \mathcal { R } _ { d i s } } \ell _ { c l s } \big ( \mathbf { z } _ { r _ { d } } , y _ { d } ^ { d i s } \big ) } \\ & { \qquad - \beta _ { s e p } \frac { 1 } { | \mathcal { V } | } \displaystyle \sum _ { i \in \mathcal { V } } \operatorname* { m a x } _ { i } \Big ( 0 , \displaystyle \operatorname* { m a x } _ { j \in \mathcal { A } } s _ { i j } - \eta \displaystyle \operatorname* { m a x } _ { j \in \mathcal { R } _ { d i s } } s _ { i j } \Big ) , } \end{array}\tag{7}
$$

Table 1: Task composition in open-world evaluation protocol. The semantics of each task and the number of images and instances (objects) across splits are shown.
<table><tr><td></td><td>Task 1</td><td>Task 2</td><td>Task 3</td><td>Task 4</td></tr><tr><td>Semantic split</td><td>VOC Classes</td><td>Outdoor, Accessories, Appliance, Truck</td><td>Sports, Food</td><td>Electronic, Indoor, Kitchen, Furniture</td></tr><tr><td># training images</td><td>16,551</td><td>45,520</td><td>39,402</td><td>40,260</td></tr><tr><td># test images</td><td>4,952</td><td>1,914</td><td>1,642</td><td>1,738</td></tr><tr><td># train instances</td><td>47,223</td><td>113,741</td><td>114,452</td><td>138,996</td></tr><tr><td># test instances</td><td>14,976</td><td>4,966</td><td>4,826</td><td>6,039</td></tr></table>

where $y _ { d } ^ { d i s }$ denotes the unknown class label, and ${ \mathbf z } _ { r _ { d } }$ denotes the class logits of the proposal $r _ { d }$ that best matches the disagreement candidate d. $\beta _ { c l s }$ and $\beta _ { s e p }$ are balancing hyperparameters.

DDTG provides an additional supervisory signal for unknown object learning by modeling cross-view semantic instability. While objectness filtering ensures that candidate regions correspond to potential objects, it does not capture ambiguity in class predictions under view perturbations. By explicitly measuring disagreement between aligned logits, DDTG complements objectness-based supervision and enhances the robustness of unknown object discovery.

Accordingly, the overall loss for unlabeled object supervision is defined as the combination of the KTRM loss and the DDTG loss:

$$
\mathcal { L } _ { D P O D } = \gamma _ { K T R M } \mathcal { L } _ { K T R M } + \gamma _ { D D T G } \mathcal { L } _ { D D T G } ,\tag{8}
$$

where $\gamma _ { K T R M }$ and $\gamma _ { D D T G }$ are balancing hyperparameters.

## 4 Experiments

## 4.1 Datasets and Sparse Annotation Settings

We utilize the Open-World Object Detection dataset [14], which is constructed from Pascal VOC [6] and MS-COCO [20]. Following the standard OWOD dataset protocol, we organize all VOC classes as the first task $T _ { 1 }$ while the remaining 60 COCO classes are divided into three successive tasks. This division introduces semantic drifts between tasks (see Tab. 1). During the training of task $T _ { t } .$ classes from previous tasks $\{ T _ { \tau } : \tau < t \}$ are treated as known, while classes from future tasks $\{ T _ { \tau } : \tau > t \}$ are treated as unknown. This setup enables progressive learning under an open-world scenario, where the model incrementally encounters new classes while retaining knowledge of previously learned ones.

![](images/848c70a3d9c694602078aef49b28ffbef29e904e095e9cff95e4e6c744f7b360.jpg)  
Fig. 3: Remaining annotation ratio under each sparse configuration (all tasks aggregated).

To simulate sparse annotation scenarios, we consider five training configurations [18, 45] designed to reduce available annotations. Unlike conventional sparse annotation settings that remove annotations globally across the entire dataset, our framework follows an open-world protocol in which diferent tasks introduce diferent sets of classes. Accordingly, annotation removal is applied independently within each task. For example, under the Easy setting, one annotation is removed per image for each task individually: one annotation is removed for classes in Task 1, another for classes in Task 2, and so on. This ensures that each task observes its own sparse annotation setting, while images shared across tasks may have multiple annotations removed corresponding to the task-specific classes.

Specifically, five configurations are defined based on the number of retained annotations per image per task:

– Easy: Remove a single annotation, retaining at least one annotation per class in the current task; approximately 18.3% of all labels are removed.

– Hard: Remove half of the annotations, retaining at least one annotation per class in the current task; approximately 38.8% of all labels are removed.

– Coco50missp: Remove half of the annotations; approximately 46.7% of labels are removed. Unlike the Hard setting, this does not guarantee at least one annotation per class in each image.

– Keep1: Keep only one annotation for each category present in an image; approximately 50.0% of labels are removed.

Extreme: Retain only a single annotation; approximately 65.8% of labels are removed.

For evaluation, we adopt the Pascal VOC test split and the MS COCO validation split.

## 4.2 Implementation Details

In this study, we adopt a Faster R-CNN [30] as the baseline [25]. We employ a ResNet50 [12] backbone pre-trained on ImageNet [4]. The network is optimized using AdamW [21] with a learning rate of $2 . 5 \times \mathrm { 1 0 ^ { - 5 } }$ and a weight decay of $1 \times 1 0 ^ { - 4 }$ . For each task, the model is trained for 15,000 iterations, followed by 15,000 iterations of incremental learning, resulting in a total of 30,000 iterations (approximately 9 epochs). All experiments are conducted on four NVIDIA RTX A6000 GPUs with a batch size of 12. All weighting coeficients $\alpha _ { u k } , \alpha _ { s e p } , \beta _ { c l s }$ $\beta _ { s e p } ,$ γ<sub>KTRM</sub>, and γ<sub>DDTG</sub> are set to 1, and the facility-location parameter η is set to 1. During inference, predictions are obtained from 10,000 pre-defined bounding boxes, after which Non-Maximum Suppression (NMS) [27] is applied to remove redundant detections. Our model comprises 106.2M parameters with a computational cost of 965.4 GFLOPs, running at 2.52 FPS with a latency of 396.1 ms.

Table 2: Sparsely Annotated Open-World Object Detection (SA-OWOD) results. Existing open-world detectors are benchmarked on sparse annotation datasets for fair comparison. K-mAP and U-Recall denote Known mAP and Unknown Recall, respectively.
<table><tr><td rowspan="2">Set</td><td rowspan="2">Method</td><td colspan="2">Task 1</td><td colspan="2">Task 2</td><td colspan="2">Task 3</td><td>Task 4</td></tr><tr><td>K-mAP</td><td>U-Recall</td><td>K-mAP</td><td>U-Recall</td><td>K-mAP</td><td>U-Recall</td><td>K-mAP</td></tr><tr><td rowspan="4">Full</td><td>RandBox [35]</td><td>61.8</td><td>10.6</td><td>45.3</td><td>6.3</td><td>39.4</td><td>7.8</td><td>35.4</td></tr><tr><td>PROB [49]</td><td>59.5</td><td>19.4</td><td>44.0</td><td>17.4</td><td>36.0</td><td>19.6</td><td>31.5</td></tr><tr><td>OrthogonalDet [31]</td><td>61.3</td><td>24.6</td><td>47.0</td><td>26.3</td><td>41.3</td><td>29.1</td><td>37.9</td></tr><tr><td>CROWD [25]</td><td>61.7</td><td>57.9</td><td>47.8</td><td>53.6</td><td>42.5</td><td>69.6</td><td>38.5</td></tr><tr><td rowspan="5">Easy</td><td>RandBox</td><td>49.91</td><td>4.90</td><td>38.36</td><td>4.30</td><td>33.12</td><td>4.60</td><td>29.79</td></tr><tr><td>PROB</td><td>52.83</td><td>18.27</td><td>38.35</td><td>15.35</td><td>32.64</td><td>19.1</td><td>29.06</td></tr><tr><td>OrthogonalDet</td><td>56.19</td><td>17.05</td><td>41.91</td><td>22.51</td><td>36.73</td><td>25.13</td><td>34.60</td></tr><tr><td>CROWD</td><td>53.28</td><td>45.53</td><td>38.24</td><td>30.22</td><td>35.43</td><td>53.34</td><td>32.28</td></tr><tr><td>Ours</td><td>56.48</td><td>55.34</td><td>39.53</td><td>52.47</td><td>36.90</td><td>63.68</td><td>32.15</td></tr><tr><td rowspan="5">Hard</td><td>RandBox</td><td>49.96</td><td>2.69</td><td>27.66</td><td>3.62</td><td>29.86</td><td>3.72</td><td>27.26</td></tr><tr><td>PROB</td><td>50.18</td><td>18.26</td><td>36.24</td><td>13.91</td><td>30.98</td><td>17.57</td><td>27.01</td></tr><tr><td>OrthogonalDet</td><td>53.14</td><td>19.19</td><td>38.03</td><td>18.28</td><td>32.87</td><td>24.44</td><td>32.14</td></tr><tr><td>CROWD</td><td>49.29</td><td>49.80</td><td>34.83</td><td>33.35</td><td>33.63</td><td>50.80</td><td>29.91</td></tr><tr><td>Ours</td><td>54.09</td><td>51.85</td><td>33.36</td><td>48.25</td><td>35.79</td><td>64.00</td><td>31.15</td></tr><tr><td rowspan="5">Coco50missp</td><td>RandBox</td><td>45.43</td><td>0.63</td><td>28.02</td><td>0.76</td><td>16.93</td><td>1.47</td><td>21.09</td></tr><tr><td>PROB</td><td>48.84</td><td>18.27</td><td>34.58</td><td>14.46</td><td>28.96</td><td>16.96</td><td>25.65</td></tr><tr><td>OrthogonalDet</td><td>54.38</td><td>15.41</td><td>35.56</td><td>25.16</td><td>31.26</td><td>29.24</td><td>28.56</td></tr><tr><td>CROWD</td><td>49.90</td><td>43.91</td><td>32.03</td><td>24.14</td><td>29.92</td><td>41.12</td><td>26.13</td></tr><tr><td>Ours</td><td>53.78</td><td>51.39</td><td>30.73</td><td>47.69</td><td>30.90</td><td>59.00</td><td>24.74</td></tr><tr><td rowspan="5">Keep1</td><td>RandBox</td><td>44.35</td><td>2.95</td><td>31.34</td><td>3.70</td><td>28.72</td><td>4.18</td><td>25.05</td></tr><tr><td>PROB</td><td>48.81</td><td>17.84</td><td>34.67</td><td>14.06</td><td>29.59</td><td>16.95</td><td>26.47</td></tr><tr><td>OrthogonalDet</td><td>50.47</td><td>14.07</td><td>35.00</td><td>16.60</td><td>31.58</td><td>19.71</td><td>30.28</td></tr><tr><td>CROWD</td><td>46.45</td><td>44.59</td><td>32.76</td><td>33.27</td><td>32.16</td><td>52.07</td><td>28.34</td></tr><tr><td>Ours</td><td>47.97</td><td>48.03</td><td>33.09</td><td>48.28</td><td>32.37</td><td>52.22</td><td>28.21</td></tr><tr><td rowspan="5">Extreme</td><td>RandBox</td><td>36.62</td><td>0.68</td><td>21.93</td><td>1.50</td><td>19.55</td><td>2.39</td><td>17.95</td></tr><tr><td>PROB</td><td>37.78</td><td>17.86</td><td>26.09</td><td>14.83</td><td>23.41</td><td>17.60</td><td>19.38</td></tr><tr><td>OrthogonalDet</td><td>41.39</td><td>12.41</td><td>26.20</td><td>14.77</td><td>23.30</td><td>24.62</td><td>22.31</td></tr><tr><td>CROWD</td><td>34.37</td><td>40.66</td><td>24.84</td><td>31.17</td><td>24.46</td><td>48.64</td><td>20.47</td></tr><tr><td>Ours</td><td>45.37</td><td>45.56</td><td>25.98</td><td>44.32</td><td>26.58</td><td>57.53</td><td>20.19</td></tr></table>

## 4.3 Main Results

SA-OWOD Performance. Table 2 compares existing OWOD methods under sparse annotations. All methods are retrained under identical sparse annotation protocols for fair comparison. K-mAP denotes the mean Average Precision over known categories, and U-Recall evaluates unknown object discovery. As annotation sparsity increases, prior OWOD approaches exhibit substantial degradation, particularly in U-Recall. In contrast, our method achieves superior U-Recall while maintaining competitive K-mAP across all sparsity levels.

In the Easy setting, our method improves U-Recall by 9.81%p over CROWD and by 38.29%p over OrthogonalDet in Task 1, achieving the best K-mAP in Tasks 1 and 3. These results demonstrate that our framework improves unknown discovery without sacrificing known-class detection. As illustrated in Fig. 4, the baseline tends to mix unknown samples with known clusters in the feature space. This phenomenon arises because sparsely annotated datasets often contain unlabeled known objects that are unintentionally treated as unknown during training, leading to ambiguous feature representations. KTRM recovers these via objectness-guided pseudo-labeling, preventing them from being incorrectly treated as unknown samples. Simultaneously, DDTG leverages cross-view semantic disagreement to identify reliable unknown candidates. By disentangling unlabeled known and unknown instances, our framework reduces the boundary confusion that often degrades prior OWOD methods under sparse supervision.

![](images/def3d7ba9af785b06554c84c8e05de2f8bc7cf67c2329e4b219c667bd925ba6c.jpg)  
Fig. 4: t-SNE visualization of feature embeddings. The distributions of known and unknown samples are shown for Tasks 1–3. Known classes are represented by diferent colors, while unknown objects are shown in red. Our method shows clearer separation between known and unknown samples compared to the baseline.

As sparsity intensifies (i.e., Easy → Hard → Extreme), the performance gap widens further. In the Hard setting, our method surpasses CROWD by 14.90%p and OrthogonalDet by 29.97%p in Task 2. Under the Extreme sparsity, the unknown recall gap over OrthogonalDet exceeds 29.55%p in multiple tasks. This robustness indicates that KTRM efectively compensates for missing supervision, while DDTG prevents premature consolidation of ambiguous proposals into known categories.

Challenging Stages. In Task 2, our K-mAP is slightly lower than OrthogonalDet due to integrating novel classes under incomplete supervision, which causes unstable boundaries. However, DDTG improves robustness to ambiguity by delaying overconfident assignments under weak cross-view semantic consistency. This results in a 29.97%p gain in U-Recall over OrthogonalDet in Hard Task 2. In Extreme Task 2, OrthogonalDet achieves the highest absolute KmAP; however, its U-Recall drops by 43.84% compared to the fully annotated setting. In contrast, although our method exhibits a comparable K-mAP reduction, the degradation in U-Recall is limited to only 17.31%. This gap demonstrates that our framework achieves a better balance between known detection and unknown discovery under sparse annotations.

In contrast, Task 4, a later incremental stage, has few remaining unknowns. Consequently, the amount of supervision related to unknown discovery becomes limited, and the learning objective shifts toward fine-grained discrimination among many known categories. Since our framework is primarily designed to resolve ambiguity between unlabeled known and unknown objects, the relative contribution of disagreement-driven unknown modeling (DDTG) naturally decreases when unknown signals become scarce. Therefore, the smaller performance margin observed in Task 4 reflects a change in task characteristics rather than a limitation in modeling class separation. Even under this shift, our method maintains competitive performance, demonstrating stable behavior across incremental learning stages.

Discussion. The core challenge of SA-OWOD lies in resolving the ambiguity of unlabeled regions. KTRM reduces false-negative supervision caused by missing annotations, preserving K-mAP stability under sparsity. DDTG strengthens unknown modeling by explicitly regularizing semantically inconsistent proposals. The reduced U-Recall degradation and performance gap validate that explicitly modeling both mechanisms is essential for robust detection in sparsely annotated open-world environments.

## 4.4 Ablation Studies

We conduct ablation studies under the Hard setting to analyze the individual contributions of the proposed KTRM and DDTG, as well as the efect of key thresholds in DDTG. Moreover, we provide a qualitative comparison of the results with those of a baseline model.

Efect of KTRM and DDTG. As shown in Tab. 3, we perform an ablation study to analyze the individual contributions of the proposed KTRM and DDTG under the Hard setting. Starting from the CROWD baseline, we progressively enable each module to examine their impact on both known object detection performance (K-mAP) and unknown object recall (U-Recall).

When only KTRM is applied, improvements are mainly observed in known category detection, indicating that KTRM enhances discriminative representation learning and stabilizes training. Applying only DDTG also improves performance, suggesting that the proposed target generation mechanism contributes to boundary refinement and unknown exploration.

When both modules are jointly applied, consistent improvements are observed across both K-mAP and U-Recall. This demonstrates that KTRM and DDTG play complementary roles: KTRM improves feature quality for known categories, while DDTG facilitates unknown object exploration. The combined design achieves a better balance between known recognition and unknown discovery.

Table 3: Ablation study on KTRM and DDTG under the Hard setting.
<table><tr><td>Task 1</td><td>KTRM</td><td>DDTG</td><td> $\mathrm { K m A P }$ </td><td> $\varDelta$ </td><td>U-Recall</td><td> $\varDelta$ </td></tr><tr><td>CROWD</td><td>x</td><td>x</td><td>49.29</td><td></td><td>49.80</td><td></td></tr><tr><td>CROWD+KTRM</td><td>√</td><td>x</td><td>51.90</td><td> $+ 2 . 6 1$ </td><td>50.66</td><td> $+ 0 . 8 6$ </td></tr><tr><td>CROWD+DDTG</td><td>x</td><td>√</td><td>51.08</td><td> $+ 1 . 7 9$ </td><td>50.26</td><td> $+ 0 . 4 6$ </td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>54.09</td><td>+4.80</td><td>51.85</td><td>+2.05</td></tr></table>

Table 4: Ablation study on $\tau _ { o }$ and $\tau _ { s }$ in DDTG under the Hard setting.
<table><tr><td colspan="4">Objectness threshold  $\left( \tau _ { o } \right)$ </td><td colspan="4">Similarity threshold  $\left( \tau _ { s } \right)$ </td></tr><tr><td> $\tau _ { o }$ </td><td> $\tau _ { s }$ </td><td>K-mAP</td><td>U-Recall</td><td> $\tau _ { o }$ </td><td> $\tau _ { s }$ </td><td>K-mAP</td><td>U-Recall</td></tr><tr><td>0.1</td><td>0.95</td><td>52.97</td><td>47.68</td><td>0.2</td><td>0.95</td><td>54.09</td><td>51.85</td></tr><tr><td>0.2</td><td>0.95</td><td>54.09</td><td>51.85</td><td>0.2</td><td>0.90</td><td>52.24</td><td>46.26</td></tr><tr><td>0.3</td><td>0.95</td><td>53.62</td><td>51.40</td><td>0.2</td><td>0.85</td><td>52.53</td><td>50.46</td></tr></table>

Sensitivity to Objectness and Similarity Thresholds. We further analyze the sensitivity of DDTG to the objectness threshold $\left( \tau _ { o } \right)$ and similarity threshold $\left( \tau _ { s } \right)$ , as shown in Tab. 4. When fixing $\tau _ { s } = 0 . 9 5$ , increasing $\tau _ { o }$ from 0.1 to 0.2 significantly improves both $\mathrm { K m A P }$ and U-Recall, indicating that moderate filtering of low-objectness proposals helps suppress noise while preserving informative candidates. However, further increasing $\tau _ { o }$ to 0.3 slightly degrades performance, suggesting that overly strict filtering may discard useful unknown candidates.

When fixing $\tau _ { o } = 0 . 2$ , increasing $\tau _ { s }$ enlarges the set of proposals regarded as semantic disagreement. Although a larger disagreement set may potentially introduce noisy candidates, the results indicate that the additional proposals provide informative supervisory signals rather than degrading detection quality. We conjecture that richer disagreement cues encourage the model to learn a clearer separation between known and unknown representations, which benefits not only unknown recall but also known-class detection performance. Reducing $\tau _ { s }$ limits the diversity of disagreement candidates, leading to consistent drops in both metrics. These findings suggest that suficiently diverse semantic disagreement signals are important for stable and efective open-world detection.

Qualitative Results. The qualitative results in Fig. 5 visually support the qualitative improvements reported in Hard Task 3. In CROWD [25], known objects are sometimes misclassified as unknown, and some unknown instances fail to be clearly activated. In contrast, our method reduces ambiguity around object boundaries and produces more consistent detections. This suggests that our approach achieves improved stability even under incomplete supervision.

## 5 Conclusion

In this work, we introduced Sparsely Annotated Open-World Object Detection (SA-OWOD), where unlabeled known and unknown objects co-exist. In this setting, unlabeled regions are inherently ambiguous and cannot be reliably treated as either unlabeled known or unknown. To address this challenge, we proposed Dual-Perspective Object Discovery $( D P O D )$ , a unified framework that explicitly separates unlabeled regions into recoverable known objects and genuinely unknown ones. By explicitly modeling and separating unlabeled known and unknown instances, DPOD efectively resolves this ambiguity. Extensive experiments demonstrate consistent improvements in both known-class detection and unknown recall. Overall, SA-OWOD provides a practical benchmark for detection under ambiguous supervision and lays the foundation for robust open-world detection in real-world scenarios.

![](images/a46871e8a8a308095bdbf9cc2c31a859f4f668e4b033de8a888d5bb52c1353b3.jpg)  
Fig. 5: Qualitative Comparison on Hard Task 3. We compare CROWD [25] and our method in terms of known and unknown detections. Blue denotes unknown objects, and green denotes known objects.

Limitations. Despite these advances, DPOD currently relies on fixed objectness and score thresholds $\left( \tau _ { o } , \tau _ { s } \right)$ for proposal filtering. In open-world scenarios, object characteristics vary significantly across tasks, and certain categories — such as accessories (e.g., ties) — tend to yield lower objectness scores. Enforcing a fixed threshold under such noisy conditions is suboptimal and may introduce progressive learning bottlenecks. Replacing these fixed heuristics with task-adaptive thresholds remains a promising direction for future work.

## Acknowledgements

This work was supported in part by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (RS-2024-00358935, 90%) and in part by the Institute of Information & Communications Technology Planning & Evaluation (IITP) under the Artificial Intelligence Convergence Innovation Human Resources Development grant funded by the Korea government (MSIT) (IITP-2026-RS-2023-00254177, 10%).

## References

1. Cai, Z., Vasconcelos, N.: Cascade r-cnn: Delving into high quality object detection. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 6154–6162 (2018)

2. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: Endto-end object detection with transformers. In: Eur. Conf. Comput. Vis. pp. 213– 229. Springer (2020)

3. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 3213–3223 (2016)

4. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 248–255. IEEE (2009)

5. Doan, T., Li, X., Behpour, S., He, W., Gou, L., Ren, L.: Hyp-ow: Exploiting hierarchical structure learning with hyperbolic distance enhances open world object detection. In: AAAI. vol. 38, pp. 1555–1563 (2024)

6. Everingham, M., Van Gool, L., Williams, C.K., Winn, J., Zisserman, A.: The pascal visual object classes (voc) challenge 88(2), 303–338 (2010)

7. Geiger, A., Lenz, P., Stiller, C., Urtasun, R.: Vision meets robotics: The kitti dataset. The international journal of robotics research 32(11), 1231–1237 (2013)

8. Girshick, R., Donahue, J., Darrell, T., Malik, J.: Rich feature hierarchies for accurate object detection and semantic segmentation. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 580–587 (2014)

9. Gupta, A., Narayan, S., Joseph, K., Khan, S., Khan, F.S., Shah, M.: Ow-detr: Open-world detection transformer. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 9235–9244 (2022)

10. Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I., Sugiyama, M.: Co-teaching: Robust training of deep neural networks with extremely noisy labels. Adv. Neural Inform. Process. Syst. 31 (2018)

11. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: Int. Conf. Comput. Vis. pp. 2961–2969 (2017)

12. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 770–778 (2016)

13. Howard, A.G.: Mobilenets: Eficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 (2017)

14. Joseph, K.J., Khan, S., Khan, F.S., Balasubramanian, V.N.: Towards open world object detection. In: IEEE Conf. Comput. Vis. Pattern Recog. (2021)

15. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., et al.: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. Int. J. Comput. Vis. 128(7), 1956–1981 (2020)

16. Lee, C., Shin, S., Park, G.M., Kim, J.U.: Multispectral pedestrian detection with sparsely annotated label. In: AAAI. vol. 39, pp. 4482–4490 (2025)

17. Lee, S., Park, J., Park, J.: Crossformer: Cross-guided attention for multi-modal object detection. Pattern Recognition Letters 179, 144–150 (2024)

18. Li, H., Pan, X., Yan, K., Tang, F., Zheng, W.S.: Siod: Single instance annotated per category per image for object detection. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 14197–14206 (2022)

19. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object detection. In: Int. Conf. Comput. Vis. pp. 2980–2988 (2017)

20. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: Eur. Conf. Comput. Vis. pp. 740–755. Springer (2014)

21. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. Int. Conf. Learn. Represent. (2019)

22. Lu, Z., Wang, C., Xu, C., Zheng, X., Cui, Z.: Progressive exploration-conformal learning for sparsely annotated object detection in aerial images. Adv. Neural Inform. Process. Syst. 37, 40593–40614 (2024)

23. Ma, S., Wang, Y., Wei, Y., Fan, J., Li, T.H., Liu, H., Lv, F.: Cat: Localization and identification cascade detection transformer for open-world object detection. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 19681–19690 (2023)

24. Ma, Y., Li, H., Zhang, Z., Guo, J., Zhang, S., Gong, R., Liu, X.: Annealing-based label-transfer learning for open world object detection. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 11454–11463 (2023)

25. Majee, A., Gangrade, A., Iyer, R.: Looking beyond the known: Towards a data discovery guided open-world object detection. Adv. Neural Inform. Process. Syst. (2025)

26. Mullappilly, S.S., Gehlot, A.S., Anwer, R.M., Khan, F.S., Cholakkal, H.: Semisupervised open-world object detection. In: AAAI. vol. 38, pp. 4305–4314 (2024)

27. Neubeck, A., Van Gool, L.: Eficient non-maximum suppression. In: Int. Conf. Pattern Recog. vol. 3, pp. 850–855. IEEE (2006)

28. Niitani, Y., Akiba, T., Kerola, T., Ogawa, T., Sano, S., Suzuki, S.: Sampling techniques for large-scale object detection from sparsely annotated objects. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 6510–6518 (2019)

29. Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Unified, real-time object detection. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 779– 788 (2016)

30. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. Adv. Neural Inform. Process. Syst. 28 (2015)

31. Sun, Z., Li, J., Mu, Y.: Exploring orthogonality in open world object detection. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 17302–17312 (2024)

32. Suri, S., Rambhatla, S., Chellappa, R., Shrivastava, A.: Sparsedet: Improving sparsely annotated object detection with pseudo-positive mining. In: Int. Conf. Comput. Vis. pp. 6770–6781 (2023)

33. Wang, H., Liu, L., Zhang, B., Zhang, J., Zhang, W., Gan, Z., Wang, Y., Wang, C., Wang, H.: Calibrated teacher for sparsely annotated object detection. In: AAAI. vol. 37, pp. 2519–2527 (2023)

34. Wang, T., Yang, T., Cao, J., Zhang, X.: Co-mining: Self-supervised learning for sparsely annotated object detection. In: AAAI. vol. 35, pp. 2800–2808 (2021)

35. Wang, Y., Yue, Z., Hua, X.S., Zhang, H.: Random boxes are open-world object detectors. In: Int. Conf. Comput. Vis. pp. 6233–6243 (2023)

36. Wu, L., Han, J., Zheng, Z., Wang, X.: Co-student: Collaborating strong and weak students for sparsely annotated object detection. In: Eur. Conf. Comput. Vis. pp. 459–475. Springer (2024)

37. Wu, Y., Zhao, X., Ma, Y., Wang, D., Liu, X.: Two-branch objectness-centric open world detection. In: Proc. Int. Workshop Hum.-Centric Multimedia Anal. pp. 35–40 (2022)

38. Wu, Z., Bodla, N., Singh, B., Najibi, M., Chellappa, R., Davis, L.S.: Soft sampling for robust object detection (2019)

39. Wu, Z., Lu, Y., Chen, X., Wu, Z., Kang, L., Yu, J.: Uc-owod: Unknown-classified open world object detection. In: Eur. Conf. Comput. Vis. pp. 193–210. Springer (2022)

40. Xia, G.S., Bai, X., Ding, J., Zhu, Z., Belongie, S., Luo, J., Datcu, M., Pelillo, M., Zhang, L.: Dota: A large-scale dataset for object detection in aerial images. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 3974–3983 (2018)

41. Yao, S., Liu, Y., Jia, Q., Chen, S., Zhuo, W.: As pseudo-label free as possible: Leveraging adaptive feature generation for sparsely annotated object detection. In: AAAI. vol. 39, pp. 9418–9426 (2025)

42. Yoon, J., Hong, S., Choi, M.K.: Semi-supervised object detection with sparsely annotated dataset. In: IEEE Int. Conf. Image Process. pp. 719–723. IEEE (2021)

43. Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., Darrell, T.: Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 2636–2645 (2020)

44. Yu, J., Ma, L., Li, Z., Peng, Y., Xie, S.: Open-world object detection via discriminative class prototype learning. arXiv preprint arXiv:2302.11757 (2023)

45. Zhang, H., Chen, F., Shen, Z., Hao, Q., Zhu, C., Savvides, M.: Solving missingannotation object detection with background recalibration loss. In: ICASSP. pp. 1888–1892. IEEE (2020)

46. Zhang, S., Ni, Y., Du, J., Xue, Y., Torr, P., Koniusz, P., van den Hengel, A.: Openworld objectness modeling unifies novel object detection. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 30332–30342 (2025)

47. Zhou, H., Ge, Z., Liu, S., Mao, W., Li, Z., Yu, H., Sun, J.: Dense teacher: Dense pseudo-labels for semi-supervised object detection. In: Eur. Conf. Comput. Vis. pp. 35–50. Springer (2022)

48. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable detr: Deformable transformers for end-to-end object detection (2021)

49. Zohar, O., Wang, K.C., Yeung, S.: Prob: Probabilistic objectness for open world object detection. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 11444–11453 (2023)