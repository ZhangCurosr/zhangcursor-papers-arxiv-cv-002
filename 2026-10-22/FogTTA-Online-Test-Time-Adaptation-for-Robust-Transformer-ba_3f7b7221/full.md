# FogTTA: Online Test-Time Adaptation for Robust Transformer-based Object Detection in Foggy Weather

Ali Abedi<sup>†,\*</sup>, Q.M. Jonathan Wu<sup>†</sup>, Ning Zhang<sup>†</sup>, Shiqi Tian<sup>‡</sup>

<sup>†</sup> University of Windsor

<sup>‡</sup> University of Toronto Mississauga

## Abstract

Object detection models for autonomous driving commonly experience substantial performance drops when deployed under adverse weather due to the domain shift between training data and real-world operating conditions. This degradation is especially evident when models trained on clear-weather images encounter foggy environments with reduced visibility and contrast. To address this challenge, we introduce FogTTA, an online testtime adaptation framework designed to improve the robustness of Transformer-based object detectors in fog. Using RF-DETR as the underlying object detector, FogTTA enables real-time adaptation to the streaming target domain without requiring source data or retraining. The framework follows a teacher–student design, where the deployed model serves as the teacher and generates pseudo labels from weakly augmented target inputs. These predictions are subsequently refined through non-maximum suppression and confidence filtering. The student model then learns from strongly augmented target sample using the Varifocal loss to mitigate pseudo-label noise. The teacher is updated via exponential moving averaging to ensure stable and continuous adaptation. Experiments show that FogTTA outperforms prior baselines, delivering improved detection accuracy and stability while maintaining real-time performance.

Keywords: Test-time Domain Adaptation, Transformer-based Object Detection, Pseudo Label Generation

## 1. Introduction

Object detection is a fundamental task in autonomous driving, where reliable perception under diverse environmental conditions is essential for safety [1]. Accurate detection and localization of surrounding objects are essential for safe autonomous navigation. Early detection models are based on convolutional neural networks (CNNs) and can be categorized into two-stage and one-stage approaches. Two-stage detectors, such as Faster R-CNN [2], achieve high accuracy through region proposal refinement but incur heavy computational costs. As real-time performance is critical for autonomous vehicles, one-stage detectors such as SSD [3] and YOLO [4] were developed to deliver eficient detection through a single feedforward pass. CNN-based methods rely on local receptive fields and thus struggle to model long-range dependencies, which limits their adaptability to adverse conditions [5, 6].

Transformer-based architectures, on the other hand, leverage self-attention to capture long-range dependencies and global context across entire images. Early models such as DETR [6] demonstrated strong accuracy but sufered from high computational overhead and slow convergence. Subsequent detectors, including Deformable DETR [7], LW-DETR [8], and RT-DETR [9], improved speed and eficiency while maintaining competitive accuracy. RF-DETR [10] integrates DINOv2 into LW-DETR, which makes it a robust and eficient detector. However, models trained under clear-weather conditions still experience substantial performance degradation when deployed in foggy environments due to the domain shift.

Domain shift refers to the discrepancy between the distributions of the training (source) and deployment (target) environments [11]. In autonomous driving, this shift is critica when clear-weather models face fog, where reduced visibility and contrast hinder object detection [12]. Unsupervised domain adaptation (UDA) methods aim to mitigate this issue by aligning labeled source and unlabeled target data. However, as shown in Figure 1a, such methods [12, 13] assume access to both domains, which is impractical due to privacy and storage constraints. Source-free domain adaptation (SFDA) [14, 15] addresses this limitation by adapting pre-trained models to unlabeled target data without accessing source samples (Figure 1b), yet still requires ofline retraining on the entire target set. In contrast, online test-time adaptation (TTA) [16–18] enables real-time adaptation to a stream of unlabeled target inputs during deployment (Figure 1c). Although more practical, TTA remains challenging due to the absence of supervision, the noise in pseudo labels, and the need to balance adaptation stability and responsiveness.

![](images/c2f8806bad6621a5259451d9a318b8fd4f62b87ba9669808f0584156c23af2c7.jpg)  
Figure 1. Illustration of domain adaptation paradigms for object detection: (a) UDA uses labeled source and unlabeled target data jointly; (b) SFDA adapts a source-trained model without source access; (c) TTA adapts online to streaming target data during deployment.

In this paper, we present FogTTA, an online test-time adaptation framework that enhances the robustness of Transformer-based object detectors under foggy conditions. Built upon RF-DETR [10], FogTTA adapts continuously to unseen target data without requiring source access or retraining. It employs a teacher–student design, where the deployed model (teacher) generates and refines pseudo labels through confidence-based filtering and non-maximum suppression (NMS), while the student learns from strongly augmented target views using the Varifocal loss [19] to handle pseudo-label noise. The teacher is updated via exponential moving averaging (EMA) of the student’s parameters to ensure stable and progressive adaptation.

The main contributions of this work are as follows:

• We propose FogTTA, an online test-time adaptation framework that enhances the robustness of Transformer-based object detection under foggy weather conditions without requiring source data.

• We utilized a pseudo-label refinement strategy that integrates NMS and the Varifocal loss to mitigate noise and uncertainty during adaptation.

• We validate FogTTA through experiments, demonstrating improved detection accuracy in foggy conditions while maintaining real-time inference performance of RF-DETR.

The rest of this paper is organized as follows. Section 2 reviews related work. Section 3 presents the proposed methodology. Section 4 describes the experimental settings, and Section 5 discusses the results. Finally, Section 6 concludes the paper and outlines future directions.

## 2. Related Works

## 2.1. Transformer-based Object Detectors

The introduction of DETR [6] marked a turning point in object detection. As the first Transformer-based detector, DETR difers from previous approaches by utilizing learnable object queries and an end-to-end pipeline that eliminates the need for post-processing. However, the original DETR design is not suitable for real-time applications due to its computational complexity and relatively high latency. Deformable DETR [7], a follow-up work, adopts deformable attention to focus on key regions, accelerating training and reducing memory usage. To further improve real-time performance, RT-DETR [9] replaces the Transformer encoder with an eficient hybrid encoder to decouple the intra-scale interaction and cross-scale fusion of multi-scale features, improving performance while preserving the NMS-free design for faster inference. LW-DETR [8] uses a convolutional projector to connect a plain ViT encoder and a shallow DETR decoder, yielding a light-weight detector suitable for real-time applications. Building on LW-DETR and DINOv2, RF-DETR [10] leverages neural architecture search to create a scheduler-free object detector that surpasses prior state-of-the-art approaches as well as real-time methods.

## 2.2. Cross-weather domain adaptation

To systematically study clear-to-foggy adaptation, Foggy Cityscapes [20] dataset synthesizes foggy scenes on Cityscapes [21] dataset and is widely used as a benchmark for weather-robust object detection tasks. Early work on cross-weather domain adaptive object detection mainly follows the UDA setting. DA-Detect [12] augments Faster RCNN with image-level and object-level domain classifiers plus an auxiliary domain to reduce the influence of domain shift. This framework is further extended [13] to foggy and rainy conditions with a Dynamic Masking Process that exploits contextual information. However, these methods require access to both source and target data during training, thereby limiting their practicality in real-world deployment. To avoid accessing source data, DDT with AEMA [15] introduces a Dual-Rate Dynamic Teacher (DDT) with Asynchronous EMA (AEMA) to balance fast adaptation to domain shift and long-term knowledge in a source-free setting. FRANCK [14] shows a query-centric source-free framework for DETR with objectness-based sample reweighting module, a contrastive learning bank, and an uncertainty-weighted distillation. Nevertheless, these methods all rely on ofline adaptation on a pre-collected target set, which limits their applicability in online test-time scenarios.

## 2.3. Test-time Domain Adaptation

Recent work extends TTA and online adaptation ideas to object detection by adapting detectors during inference without source data. MemCLR [18] introduced a unified teacher-student framework that adapts the target domain in both ofline and online settings. MLFA [17] aligns informative global-level and cluster-level features between source and target domains while minimizing target entropy for realistic TTA object detection. DDT [16] introduce Dynamic Dual Teaching (DDT) scheme that combines a source detector and a vision language model teacher with dynamic prediction fusion and consistency regularization to achieve better generalization ability. However, existing TTA detectors still struggle with noisy pseudo labels and unstable adaptation under severe fog conditions. To address these limitations, we develop a transformer-based TTA framework that performs reliable adaptation without accessing source data.

## 3. Proposed Method

## 3.1. Problem Formulation

Let the labeled source domain be denoted as $\mathcal { D } ^ { S } = \{ ( x _ { i } ^ { S } , y _ { i } ^ { S } ) \} _ { i = 1 } ^ { N _ { S } }$ , drawn from the joint distribution $P _ { S } ( X , Y )$ , where $x _ { i } ^ { S }$ and $y _ { i } ^ { S }$ represent the input image and its corresponding annotation. The source domain corresponds to clear-weather driving conditions. A model $f _ { \theta }$ is first trained on the source domain by minimizing the supervised detection loss:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { s r c } } ( f _ { \theta } ) = \frac { 1 } { N _ { S } } \sum _ { i = 1 } ^ { N _ { S } } \ell ( f _ { \theta } ( x _ { i } ^ { S } ) , y _ { i } ^ { S } ) ,\tag{3.1}
$$

where $\ell ( \cdot , \cdot )$ denotes the overall detection loss that combines classification and localization terms.

At deployment, the pretrained model encounters a stream of unlabeled target samples $\mathcal { D } ^ { T } = \bar { \{ x _ { j } ^ { T } \} } _ { j = 1 } ^ { N _ { T } }$ , drawn from foggy-weather driving conditions following a distinct distribution $\bar { P _ { T } } ( \bar { X } )$ such that $P _ { T } ( X ) \neq P _ { S } ( X )$ . The goal of TTA is to adapt $f _ { \theta }$ online to the target distribution without access to $\mathcal { D } ^ { S }$ or its labels. Formally, the adaptation objective is expressed as:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { x ^ { T } \sim P _ { T } ( X ) } \left[ \ell ( f _ { \theta } ( x ^ { T } ) ) \right] ,\tag{3.2}
$$

where $\mathbb { E } _ { x ^ { T } \sim P _ { T } ( X ) } [ \cdot ]$ denotes the expected loss over target samples drawn from the target distribution, and $\operatorname { \rho } ( f _ { \boldsymbol { \theta } } ( x ^ { T } ) )$ represents the unsupervised adaptation loss computed from each incoming target image.

## 3.2. Model Overview

Figure 2 illustrates the overall framework of our proposed TTA method for autonomous driving under foggy conditions. The process begins with a model $f _ { \theta }$ pretrained on a labeled source dataset $\mathcal { D } ^ { S }$ that contains clear-weather driving scenes, as described in Section 3.3. During deployment, this pretrained model is installed on the vehicle and performs inference on a continuous stream of unlabeled target samples $\mathcal { D } ^ { T }$ captured under foggy-weather conditions. As the visual appearance of foggy scenes difers significantly from clear-weather training data, the model experiences performance degradation caused by the domain shift between $P _ { T } ( X )$ and $P _ { S } ( X )$ .

To maintain robustness under such domain shifts, the model is continuously adapted to the incoming target data in an online manner without accessing the source dataset. This process follows a teacher–student framework, where the pretrained model $f _ { \theta }$ serves as both the inference and teacher network, and a student model $f _ { \phi }$ , initialized with the teacher’s parameters, is used for adaptation. This architecture enables continuous online refinement as the environment evolves while ensuring that inference remains real-time through the eficiency of the RF-DETR backbone and is unafected by the adaptation updates.

As detailed in Section 3.4, for each target sample in the stream, the teacher generates predictions that are treated as supervision signals for the student. The student is optimized using the Varifocal loss [19] (Section 3.5) to minimize the discrepancy between its predictions and the teacher’s refined pseudo labels, thereby learning target-specific representations from unlabeled data. After each adaptation step, the teacher model is updated using an EMA of the student’s parameters to integrate the latest knowledge from the target domain. Consequently, the teacher model, which also serves as the deployed inference model, remains continuously adapted to the evolving target data stream while retaining the knowledge learned from the source domain.

![](images/7241e6db162bb6490f3ae0aca93d3c13953e0e905acdb180d60054e0e7f4c758.jpg)  
Figure 2. Model overview of the proposed FogTTA framework. (a) The model is trained on the labeled source domain using focal and detection losses. (b) The trained model is then deployed on the vehicle. (c) During deployment, the vehicle encounters a continuous stream of target data with a distribution diferent from the source domain. (d) The incoming target data are processed through the TTA pipeline. (e) Weak and strong augmentations of the target images are generated, where the deployed model serves as the teacher and processes weakly augmented inputs to produce refined pseudo labels. NMS is applied to the results, and the student model, which receives the strongly augmented inputs, is trained on these pseudo labels using the Varifocal loss. The teacher model is then updated via EMA of the student’s parameters.

## 3.3. Training Source Model

The training objective of source $f _ { \theta }$ on the labeled source dataset $\mathcal { D } ^ { S }$ follows the standard DETR [6] formulation and is optimized using a combination of classification and localization losses. Specifically, the overall loss is defined as:

$$
\mathcal { L } _ { \mathrm { s r c } } = \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { b o x } } \mathcal { L } _ { \mathrm { b o x } } + \lambda _ { \mathrm { g i o u } } \mathcal { L } _ { \mathrm { g i o u } } ,\tag{3.3}
$$

where $\lambda _ { \mathrm { c l s } } , \lambda _ { \mathrm { b o x } }$ , and $\lambda _ { \mathrm { g i o u } }$ are weighting coeficients that balance the individual loss terms. The classification loss $\mathcal { L } _ { \mathrm { c l s } }$ is computed using the focal loss [22], which addresses the foreground–background imbalance commonly observed in dense detection tasks. It is formulated as:

$$
\mathcal { L } _ { \mathrm { c l s } } = - \alpha _ { t } ( 1 - p _ { t } ) ^ { \gamma } \log ( p _ { t } ) ,\tag{3.4}
$$

where $p _ { t }$ represents the predicted probability for the target class, $\alpha _ { t }$ is a balancing factor, and $\gamma$ is the focusing parameter that down-weights easy samples.

For localization, the bounding box regression loss $\mathcal { L } _ { \mathrm { b o x } }$ and the generalized IoU loss $\mathcal { L } _ { \mathrm { g i o u } }$ [23] are employed to jointly optimize box position and shape alignment. The Hungarian matching algorithm [6] is used to associate each prediction with a corresponding groundtruth box before loss computation.

## 3.4. Test-Time Domain Adaptation

For each incoming image $x _ { t } ,$ , the teacher network generates a set of object predictions

$$
\mathcal { P } _ { t } = \{ ( b _ { i } , p _ { i } ) \} _ { i = 1 } ^ { N _ { t } } ,
$$

where $b _ { i }$ and $p _ { i }$ represent the predicted bounding box and its associated class probability vector. To improve stability, the teacher performs two inference passes on weakly augmented versions of $x _ { t } .$ , and the resulting predictions are merged and ranked according to their confidence scores.

Only the top K detections are retained, and low-confidence predictions are filtered using a pseudo-label confidence threshold $\tau _ { \mathrm { p l } }$ . The refined pseudo-label set is defined as

$$
\begin{array} { r } { \hat { \mathcal { Y } } _ { t } = \{ ( \hat { b } _ { k } , \hat { y } _ { k } , \hat { s } _ { k } ) \in \mathcal { P } _ { t } \mid \hat { s } _ { k } \geq \tau _ { \mathrm { p l } } \} , } \end{array}
$$

where $\hat { s } _ { k } = \operatorname* { m a x } _ { c } p _ { k , c }$ denotes the highest class confidence. The value of $\tau _ { \mathrm { p l } }$ determines a trade-of between precision and recall in the pseudo labels. A higher threshold increases precision by filtering out uncertain detections, however decreases recall, which means missing pseudo labels. Conversely, a lower threshold increases recall by retaining more true positives, but may introduce additional false positives that amplify noise during adaptation. As illustrated in Figure $^ { 3 , }$ selecting an appropriate $\tau _ { \mathrm { p l } }$ is crucial to balancing stability and adaptability in the test-time learning process.

![](images/bd165f90b64a5e69c5e694a2b7f94a20569d98d40d2824b8545961414e69ff2b.jpg)  
Figure 3. Precision and recall of pseudo labels with respect to the confidence threshold $\tau _ { \mathrm { p l } } .$

A non-maximum suppression with an IoU threshold $\tau _ { \mathrm { n m s } }$ is then applied to eliminate redundant detections. The resulting pseudo labels $\hat { \mathcal { V } } _ { t }$ are used to supervise the student model $f _ { \phi }$ , which is initialized with the teacher’s parameters and optimized using the adaptation losses detailed in Section 3.5. After each adaptation step, the teacher parameters are updated using an EMA of the student’s parameters:

$$
\theta _ { t + 1 }  \mu \theta _ { t } + ( 1 - \mu ) \phi _ { t } ,
$$

where $\mu$ is the EMA momentum coeficient. This update allows the teacher, which also serves as the deployed inference model, to remain continuously adapted to the evolving target domain while retaining the knowledge learned from the source data.

## 3.5. Handling Pseudo-Label Noise with Varifocal Loss

Pseudo labels generated by the teacher model inherently contain noise due to uncertain predictions under foggy conditions. Low visibility, degraded contrast, and domain shift often lead to inaccurate classification scores and imprecise localization of bounding boxes.

Directly training the student model on such noisy pseudo labels can result in confirmation bias [24, 25] and performance degradation. To mitigate this issue, we adopt the Varifocal loss [19], which adaptively weights the contribution of each sample based on its confidence and localization quality.

The Varifocal loss is formulated as:

$$
\mathcal { L } _ { \mathrm { v f } } = - \sum _ { i = 1 } ^ { N } \alpha _ { i } \left( q _ { i } ( 1 - p _ { i } ) ^ { \gamma } \log ( p _ { i } ) + ( 1 - q _ { i } ) p _ { i } ^ { \gamma } \log ( 1 - p _ { i } ) \right) ,
$$

where $p _ { i }$ denotes the predicted probability, $q _ { i }$ is the target confidence derived from the IoU between the predicted and pseudo-labeled boxes, and $\alpha _ { i }$ and $\gamma$ are balancing and focusing parameters, respectively. Unlike hard thresholding, which discards low-confidence pseudo labels and their potentially useful information, the Varifocal loss retains all samples but adjusts their gradient contribution according to their confidence and localization accuracy. This allows the model to exploit informative low-confidence predictions while reducing the impact of unreliable ones.

By preserving pseudo labels and re-weighting them rather than removing them entirely, the Varifocal loss maintains richer supervision and achieves a better balance between learning stability and adaptability. Consequently, the adaptation process becomes more robust to pseudo-label noise and yields improved detection performance in foggy scenes compared to conventional focal loss.

## 4. Experimental settings

## 4.1. Dataset

We conduct experiments on the clear-to-foggy domain adaptation scenario using the Cityscapes [21] and Foggy Cityscapes [20] datasets. The labeled training set of Cityscapes is employed as the source domain, whereas the unlabeled training set of Foggy Cityscapes is utilized as the target domain during test-time adaptation. Following prior work, model performance is evaluated on the validation set of Foggy Cityscapes to measure its efectiveness and generalization capability under foggy-weather conditions.

## 4.2. Implementation Details

To simulate the streaming nature of target data during test-time adaptation, the target images are processed sequentially with a batch size of one. The model is adapted for one epoch over the entire target dataset, ensuring that each image is seen exactly once in the adaptation phase. The learning rate is set to $1 \times 1 0 ^ { - 4 }$ . The pseudo-label confidence threshold $\tau _ { \mathrm { p l } }$ is set to 0.3, as discussed in Section 5.4.1, and the NMS threshold $\tau _ { \mathrm { n m s } }$ is fixed at 0.8 (Section 5.4.2). Based on common practice [26, 27], we set the EMA decay coeficient $\mu$ to 0.999. Following the default configuration of RF-DETR [10] for the maximum number of detected objects, we set $K = 3 0 0$ as the maximum number of pseudo labels retained after filtering.

Two types of data augmentation are applied to improve adaptation stability. For the teacher model, weak augmentation consists of light photometric and geometric transformations, including mild color jittering and random horizontal flipping. For the student model, a stronger augmentation is used to encourage robustness to appearance changes. It applies heavier color jittering, random grayscale conversion, Gaussian blur, and random erasing, combined with horizontal flipping.

All experiments are conducted on a single NVIDIA GeForce GTX 1080 Ti GPU.

Table 1. Comparison of detection performance on the Cityscapes → Foggy Cityscapes adaptation setting. † denotes Transformer-based object detection. TTA indicates testtime domain adaptation.
<table><tr><td>Method</td><td>TTA</td><td>Person</td><td>Rider</td><td>Car</td><td>Truck</td><td>Bus</td><td>Train</td><td>Motor</td><td>Bicycle</td><td>mAP</td></tr><tr><td>MLFA [17]</td><td>√</td><td>38.0</td><td>50.9</td><td>18.5</td><td>32.3</td><td>14.4</td><td>21.0</td><td>31.9</td><td>22.5</td><td>28.7</td></tr><tr><td>MemCLR [18]</td><td>√</td><td>32.1</td><td>41.4</td><td>43.5</td><td>21.4</td><td>33.1</td><td>11.5</td><td>25.5</td><td>32.9</td><td>29.8</td></tr><tr><td>DDT [16]</td><td>√</td><td>32.0</td><td>41.1</td><td>47.9</td><td>25.1</td><td>33.8</td><td>28.5</td><td>26.0</td><td>35.2</td><td>33.7</td></tr><tr><td>DA-Detect [12]</td><td>X</td><td>36.5</td><td>46.7</td><td>54.3</td><td>30.3</td><td>51.2</td><td>48.7</td><td>31.6</td><td>39.1</td><td>42.3</td></tr><tr><td>DA-Detect (AdvGRL) [13]</td><td>X</td><td>36.5</td><td>49.1</td><td>55.8</td><td>30.0</td><td>48.7</td><td>51.3</td><td>33.3</td><td>41.6</td><td>43.4</td></tr><tr><td>FRANCK [14]</td><td>X</td><td>48.1</td><td>49.3</td><td>60.6</td><td>33.9</td><td>48.2</td><td>36.9</td><td>34.0</td><td>47.9</td><td>44.9</td></tr><tr><td>DDT with AEMA [15]</td><td>X</td><td>49.3</td><td>53.0</td><td>65.4</td><td>25.8</td><td>43.0</td><td>39.7</td><td>40.0</td><td>47.9</td><td>45.5</td></tr><tr><td>Source Only †</td><td>√</td><td>52.7</td><td>54.9</td><td>71.6</td><td>43.1</td><td>66.7</td><td>61.6</td><td>47.6</td><td>50.3</td><td>56.1</td></tr><tr><td>FogTTA (Ours) †</td><td>√</td><td>53.8</td><td>58.0</td><td>72.6</td><td>45.2</td><td>68.3</td><td>63.1</td><td>48.1</td><td>52.0</td><td>57.6</td></tr><tr><td>Oracle †</td><td>√</td><td>57.1</td><td>58.8</td><td>78.6</td><td>47.0</td><td>76.4</td><td>67.2</td><td>48.5</td><td>52.6</td><td>60.8</td></tr></table>

## 4.3. Evaluation Metric

We evaluate performance using mAP, computed as the mean of class-wise AP values at an IoU threshold of 0.5. This metric jointly reflects detection accuracy and localization quality across all object categories.

## 5. Experimental Results

## 5.1. Results

Table 1 presents the detection performance under the Cityscapes → Foggy Cityscapes setting. Both the Source Only and Oracle models are based on the RF-DETR detector, where Source Only refers to the model trained solely on the clear-weather source domain without adaptation, and Oracle represents the model trained directly on labeled foggy data. The proposed FogTTA achieves the best overall performance across all classes, with an mAP of 57.6%, outperforming state-of-the-art methods while maintaining real-time eficiency. Compared to the source-only baseline, FogTTA yields a +1.5% mAP improvement and consistent gains across categories. These results confirm that online adaptation through pseudo-label refinement, Varifocal loss, and EMA-based teacher updates efectively mitigates domain shift and enhances robustness in foggy conditions.

## 5.2. Qualitative Analysis

Figure 4 shows detection results before (first row) and after (second row) adaptation. Green circles indicate new correct detections after adaptation, while yellow circles mark false detections removed by FogTTA, showing improved accuracy and robustness under foggy conditions.

## 5.3. Ablation Study

The contribution of each component in the proposed framework is examined to assess its efect on adaptation performance. Table 2 presents the results for diferent model variants, focusing on the influence of pseudo-label filtering and the Varifocal loss, which are both essential for achieving stable and reliable adaptation.

Removing pseudo-label filtering (FogTTA w/o Pseudo-Label Filtering) results in a clear decrease in accuracy. It shows the importance of discarding low-confidence detections to maintain high-quality supervision. Without this step, noisy pseudo labels with reduced precision are used during adaptation, which weakens the learning signal. Excluding the Varifocal loss (FogTTA w/o Varifocal Loss) also lowers performance, which highlights its efectiveness in mitigating residual noise by weighting samples according to their confidence and localization reliability. When both pseudo-label filtering and the Varifocal loss are removed, performance declines further, indicating that these two components complement each other in improving adaptation stability and overall detection accuracy.

![](images/57b9995ee3a2a00047f37b3e7f85048f748f0db6f53cc03c911346dd1b4b78cb.jpg)  
Figure 4. Detection results before (top) and after (bottom) adaptation. Green circles show new correct detections, while yellow circles show false positives removed by FogTTA.

Table 2. Ablation study on the impact of pseudo-label filtering and Varifocal loss under the Cityscapes → Foggy Cityscapes setting.
<table><tr><td>Method</td><td>Person</td><td>Rider</td><td>Car</td><td>Truck</td><td>Bus</td><td>Train</td><td>Motor</td><td>Bicycle</td><td>mAP</td></tr><tr><td>FogTTA (Ours)</td><td>53.8</td><td>58.0</td><td>72.6</td><td>45.2</td><td>68.3</td><td>63.1</td><td>48.1</td><td>52.0</td><td>57.6</td></tr><tr><td>FogTTA w/o Pseudo-Label Filtering</td><td>52.2</td><td>56.9</td><td>72.4</td><td>44.5</td><td>67.8</td><td>61.4</td><td>47.7</td><td>51.2</td><td>56.8</td></tr><tr><td>FogTTA w/o Varifocal Loss</td><td>52.1</td><td>57.5</td><td>71.0</td><td>43.1</td><td>67.2</td><td>63.8</td><td>46.9</td><td>51.1</td><td>56.6</td></tr><tr><td>FogTTA w/o Pseudo-Label Filtering and Varifocal Loss</td><td>52.1</td><td>55.9</td><td>70.1</td><td>44.5</td><td>68.9</td><td>62.3</td><td>45.8</td><td>50.6</td><td>56.3</td></tr></table>

## 5.4. Sensitivity Analysis

## 5.4.1. Efect of $\tau _ { \mathbf { p l } }$

As illustrated in Figure 5, the best overall performance is achieved when $\tau _ { \mathrm { p l } } = 0 . 3 .$ This observation aligns with the trend shown in Figure 3, where a threshold around 0.3 provides a desirable balance between precision and recall. Lower values of $\tau _ { \mathrm { p l } }$ introduce excessive noise due to low-confidence predictions, while higher values filter out potentially informative pseudo labels, leading to reduced coverage. Setting $\tau _ { \mathrm { p l } }$ to 0.3 ofers the best trade-of between stability and adaptability during test-time learning.

![](images/d2776afe296c62414c483bf3723d7cfd0695c3a2bc8ab969c3ec85dedb5d04dd.jpg)  
Figure 5. Efect of $\tau _ { \mathrm { p l } }$ on mAP.

## 5.4.2. Efect of τ<sub>nms</sub>

The NMS threshold $\tau _ { \mathrm { n m s } }$ controls the level of overlap allowed between detected bounding boxes and directly influences the trade-of between duplicate suppression and missed detections. Table 3 reports the results for diferent values of $\tau _ { \mathrm { n m s } }$ . As the threshold increases from 0.5 to 0.8, the mAP consistently improves, indicating that moderately higher overlap tolerance helps preserve valid detections that may otherwise be suppressed under dense or partially occluded conditions common in foggy scenes. When $\tau _ { \mathrm { n m s } }$ exceeds 0.8, performance slightly declines due to increased redundancy from overlapping boxes. The best performance is achieved at $\tau _ { \mathrm { n m s } } = 0 . 8$ , which provides a balanced compromise between duplicate suppression and recall preservation.

Table 3. Detection performance with diferent NMS thresholds $\tau _ { \mathrm { n m s } }$
<table><tr><td>τnms</td><td>Person</td><td>Rider</td><td>Car</td><td>Truck</td><td>Bus</td><td>Train</td><td>Motor</td><td>Bicycle</td><td>mAP</td></tr><tr><td>0.5</td><td>52.9</td><td>57.7</td><td>71.8</td><td>44.4</td><td>67.6</td><td>62.6</td><td>47.7</td><td>51.6</td><td>57.0</td></tr><tr><td>0.6</td><td>52.0</td><td>57.3</td><td>71.5</td><td>44.3</td><td>67.9</td><td>64.6</td><td>48.4</td><td>52.0</td><td>57.2</td></tr><tr><td>0.7</td><td>53.0</td><td>57.5</td><td>71.7</td><td>45.1</td><td>68.0</td><td>62.4</td><td>48.7</td><td>51.9</td><td>57.3</td></tr><tr><td>0.8</td><td>53.8</td><td>58.0</td><td>72.6</td><td>45.2</td><td>68.3</td><td>63.1</td><td>48.1</td><td>52.0</td><td>57.6</td></tr><tr><td>0.9</td><td>53.1</td><td>58.1</td><td>72.4</td><td>44.5</td><td>69.1</td><td>62.9</td><td>48.3</td><td>51.6</td><td>57.5</td></tr></table>

## 6. Conclusion

In this paper, we presented FogTTA, an online TTA framework designed to enhance the robustness of Transformer-based object detectors under foggy weather conditions. Building upon RF-DETR, FogTTA enables online adaptation without accessing source data by combining pseudo-label refinement, Varifocal loss, and EMA-based teacher updates. Experimental results demonstrated that FogTTA consistently outperforms all baseline methods in detection accuracy and stability compared to the source-only model, confirming the effectiveness of our approach for real-world autonomous driving scenarios. For future work, we plan to extend FogTTA to broader adaptation scenarios, including cross-dataset settings and simulation-to-real transfer, to evaluate its generalization under more diverse domain shifts. We also aim to explore additional adaptation cues, such as temporal information and uncertainty estimation, to further improve robustness in challenging driving environments.

## Acknowledgements

This research is partially funded by the NSERC CREATE TrustCAV program and the NSERC Discovery Grant program, and in part by the NSERC Canada Research Chair (CRC) Program.

The authors used AI-assisted tools solely for language polishing of portions of this manuscript.

## References

[1] J. Janai, F. Güney, A. Behl, and A. Geiger. “Computer Vision for Autonomous Vehicles: Problems, Datasets and State of the Art”. In: Found. Trends. Comput. Graph. Vis. 12.1–3 (July 2020), 1–308. issn: 1572-2740. doi: 10.1561/0600000079. url: https://doi.org/10. 1561/0600000079.

[2] S. Ren, K. He, R. Girshick, and J. Sun. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks”. In: Advances in Neural Information Processing Systems. Ed. by C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett. Vol. 28. Curran Associates, Inc., 2015. url: https://proceedings.neurips.cc/paper\_files/paper/2015/ file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf.

[3] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg. “SSD: Single Shot MultiBox Detector”. In: Computer Vision – ECCV 2016. Ed. by B. Leibe, J. Matas, N. Sebe, and M. Welling. Cham: Springer International Publishing, 2016, pp. 21–37. isbn: 978-3-319-46448-0.

[4] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. “You Only Look Once: Unified, Real-Time Object Detection”. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016.

[5] A. Dosovitskiy. “An image is worth 16x16 words: Transformers for image recognition at scale”. In: arXiv preprint arXiv:2010.11929 (2020).

[6] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko. “End-to-End Object Detection with Transformers”. In: Computer Vision – ECCV 2020. Ed. by A. Vedaldi, H. Bischof, T. Brox, and J.-M. Frahm. Cham: Springer International Publishing, 2020, pp. 213–229. isbn: 978-3-030-58452-8.

[7] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai. “Deformable detr: Deformable transformers for end-to-end object detection”. In: arXiv preprint arXiv:2010.04159 (2020).

[8] Q. Chen, X. Su, X. Zhang, J. Wang, J. Chen, Y. Shen, C. Han, Z. Chen, W. Xu, F. Li, S. Zhang, K. Yao, E. Ding, G. Zhang, and J. Wang. LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection. 2024. arXiv: 2406.03459 [cs.CV]. url: https://arxiv. org/abs/2406.03459.

[9] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen. “DETRs Beat YOLOs on Real-time Object Detection”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024, pp. 16965–16974.

[10] I. Robinson, P. Robicheaux, M. Popov, D. Ramanan, and N. Peri. RF-DETR: Neural Architecture Search for Real-Time Detection Transformers. 2025. arXiv: 2511.09554 [cs.CV]. url: https://arxiv.org/abs/2511.09554.

[11] J. Li, Z. Yu, Z. Du, L. Zhu, and H. T. Shen. “A Comprehensive Survey on Source-Free Domain Adaptation”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence 46.8 (2024), pp. 5743–5762. doi: 10.1109/TPAMI.2024.3370978.

[12] J. Li, R. Xu, J. Ma, Q. Zou, J. Ma, and H. Yu. “Domain Adaptive Object Detection for Autonomous Driving under Foggy Weather”. In: 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 2023, pp. 612–622. doi: 10.1109/WACV56688. 2023.00068.

[13] J. Li, R. Xu, X. Liu, J. Ma, B. Li, Q. Zou, J. Ma, and H. Yu. “Domain Adaptation Based Object Detection for Autonomous Driving in Foggy and Rainy Weather”. In: IEEE Transactions on Intelligent Vehicles 10.2 (2025), pp. 900–911. doi: 10.1109/TIV.2024.3419689.

[14] H. Yao, S. Zhao, S. Lu, H. Chen, Y. Li, G. Liu, T. Xing, C. Yan, J. Tao, and G. Ding. “Source-Free Object Detection With Detection Transformer”. In: IEEE Transactions on Image Processing 34 (2025), pp. 5948–5963. doi: 10.1109/TIP.2025.3607621.

[15] Q. He, X. Wu, J.-Y. He, and S. Li. “Dual-Rate Dynamic Teacher for Source-Free Domain Adaptive Object Detection”. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025, pp. 2067–2076.

[16] S. Zhang, L. Zhang, and Z. Liu. “Test-time adaptation for object detection via Dynamic Dual Teaching”. In: Image and Vision Computing 163 (2025), p. 105740. issn: 0262-8856. doi: https://doi.org/10.1016/j.imavis.2025.105740. url: https://www.sciencedirect. com/science/article/pii/S0262885625003282.

[17] Y. Liu, J. Wang, C. Huang, Y. Wu, Y. Xu, and X. Cao. “MLFA: Toward Realistic Test Time Adaptive Object Detection by Multi-Level Feature Alignment”. In: IEEE Transactions on Image Processing 33 (2024), pp. 5837–5848. doi: 10.1109/TIP.2024.3473532.

[18] V. VS, P. Oza, and V. M. Patel. “Towards Online Domain Adaptive Object Detection”. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 2023, pp. 478–488.

[19] H. Zhang, Y. Wang, F. Dayoub, and N. Sünderhauf. “VarifocalNet: An IoU-aware Dense Object Detector”. In: CVPR. 2021.

[20] M. Hahner, D. Dai, C. Sakaridis, J.-N. Zaech, and L. V. Gool. “Semantic Understanding of Foggy Scenes with Purely Synthetic Data”. In: 2019 IEEE Intelligent Transportation Systems Conference (ITSC). 2019, pp. 3675–3681. doi: 10.1109/ITSC.2019.8917518.

[21] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. “The Cityscapes Dataset for Semantic Urban Scene Understanding”. In: Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016.

[22] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár. “Focal Loss for Dense Object Detection”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence 42.2 (2020), pp. 318–327. doi: 10.1109/TPAMI.2018.2858826.

[23] H. Rezatofighi, N. Tsoi, J. Gwak, A. Sadeghian, I. Reid, and S. Savarese. “Generalized Intersection Over Union: A Metric and a Loss for Bounding Box Regression”. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2019, pp. 658–666. doi: 10.1109/CVPR.2019.00075.

[24] B. Chen, J. Jiang, X. Wang, P. Wan, J. Wang, and M. Long. “Debiased self-training for semi-supervised learning”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 32424–32437.

[25] E. Arazo, D. Ortego, P. Albert, N. E. O’Connor, and K. McGuinness. “Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning”. In: 2020 International Joint Conference on Neural Networks (IJCNN). 2020, pp. 1–8. doi: 10.1109/IJCNN48605.2020.9207304.

[26] A. Tarvainen and H. Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. 2018. arXiv: 1703.01780 [cs.NE]. url: https://arxiv.org/abs/1703.01780.

[27] D. Morales-Brotons, T. Vogels, and H. Hendrikx. Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits. 2024. arXiv: 2411.18704 [cs.LG]. url: https: //arxiv.org/abs/2411.18704.